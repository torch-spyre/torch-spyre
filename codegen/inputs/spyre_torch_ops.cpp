/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/EmptyTensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <logging.h>
#include <module.h>
#include <spyre_mem.h>
#include <spyre_sendnn_utils.h>
#include <spyre_storage_impl.h>
#include <spyre_tensor_impl.h>
#include <torch/library.h>

#include <cstdlib>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/graph/graph_serializer.hpp>
#include <sendnn/graph/graph_utils.hpp>
#include <sendnn/graph/senparms.hpp>
#include <sendnn/interface/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spyre {

namespace {

// Adapted from CUDA
at::Tensor spyre__view(const at::Tensor &self, c10::SymIntArrayRef size) {
  DEBUGINFO("self is on:", self.device());
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

// Adapted from CUDA
at::Scalar spyre___local_scalar_dense(const at::Tensor &self) {
  DEBUGINFO("Tensor is on:", self.device());
  TORCH_CHECK_NOT_IMPLEMENTED(self.is_cpu(),
                              "non-CPU device not supported yet!");
  std::optional<c10::Device> common_device = std::nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "spyre___local_scalar_dense", "self");
  const at::OptionalDeviceGuard device_guard(device_of(self));
  return at::native::_local_scalar_dense_cpu(self);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
getOutputShapeStrideForReduce(at::IntArrayRef inp_dims,
                              at::OptionalIntArrayRef dim, bool keepdim) {
  int n_inp_dim = static_cast<int>(inp_dims.size());
  std::vector<int64_t> out_shape;

  if (dim.has_value()) {
    for (int i = 0; i < n_inp_dim; i++) {
      bool found = false;
      for (size_t j = 0; j < dim->size(); j++) {
        if (i == dim->at(j)) {
          if (keepdim == true) out_shape.push_back(1);
          found = true;
          break;
        }
      }
      if (!found) out_shape.push_back(inp_dims[i]);
    }
  }

  if (out_shape.empty()) out_shape.push_back(1);

  int n_dim = out_shape.size();
  std::vector<int64_t> out_strides(n_dim);
  int64_t stride = 1;
  for (int i = n_dim - 1; i >= 0; i--) {
    out_strides[i] = stride;
    stride *= out_shape[i];
    DEBUGINFO("output shape: ", out_shape[i], "at dim: ", i);
    DEBUGINFO("output stride: ", out_strides[i], "at dim: ", i);
  }

  return std::make_pair(out_shape, out_strides);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
getOutputShapeStrideForMatmul(at::IntArrayRef inp1_dims,
                              at::IntArrayRef inp2_dims,
                              bool transpose_flag = false) {
  // Assuming n_dim1 is greater or equal than n_dim2
  int n_dim1 = static_cast<int>(inp1_dims.size());
  int n_dim2 = static_cast<int>(inp2_dims.size());
  std::vector<int64_t> out_shape(n_dim1);

  // This loop computes the right dimensions for a matmul output
  // 1. Batch dimensions stay the same (any dim i < n_dim1 - 2)
  // 2. The other 2 dims are decided based on the inputs to the matmul
  for (int i = 0; i < n_dim1; i++) {
    if (i < n_dim1 - 2) {
      out_shape[i] = inp1_dims[i];
    } else if (i == n_dim1 - 2) {
      if (transpose_flag) {
        out_shape[i] = inp2_dims[n_dim2 - 1];
      } else {
        out_shape[i] = inp1_dims[n_dim1 - 2];
      }
    } else {
      if (transpose_flag) {
        out_shape[i] = inp1_dims[n_dim1 - 2];
      } else {
        out_shape[i] = inp2_dims[n_dim2 - 1];
      }
    }
    DEBUGINFO("output shape: ", out_shape[i], "at dim: ", i);
  }

  std::vector<int64_t> out_strides(n_dim1);
  int64_t stride = 1;
  for (int i = n_dim1 - 1; i >= 0; i--) {
    out_strides[i] = stride;
    stride *= out_shape[i];
    DEBUGINFO("output stride: ", out_strides[i], "at dim: ", i);
  }

  return std::make_pair(out_shape, out_strides);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
getOutputShapeStrideForConcat(std::vector<at::IntArrayRef> inp_sizes_vector,
                              int64_t dim, bool stack) {
  int n_inps = static_cast<int>(inp_sizes_vector.size());
  int n_dim = static_cast<int>(inp_sizes_vector[0].size());

  DEBUGINFO("n_inps: ", n_inps, "n_dim: ", n_dim);

  std::vector<int64_t> out_shape;

  for (int i = 0; i < n_dim; i++) out_shape.push_back(inp_sizes_vector[0][i]);

  if (stack) {
    out_shape.insert(out_shape.begin() + dim, n_inps);
  } else {
    for (int j = 1; j < n_inps; j++) out_shape[dim] += inp_sizes_vector[j][dim];
  }

  std::vector<int64_t> out_strides(n_dim);
  int64_t stride = 1;
  for (int i = n_dim - 1; i >= 0; i--) {
    out_strides[i] = stride;
    stride *= out_shape[i];
    DEBUGINFO("output stride: ", out_strides[i], "at dim: ", i);
  }

  return std::make_pair(out_shape, out_strides);
}

// INSERT_CODEGEN_HERE

// TODO(filhan): Even though codegen generated version should work, causes an
// autograd error. Dummy bypass for now.
::std::tuple<at::Tensor, at::Tensor> spyre__native_dropout_default(
    const at::Tensor &input, double p, ::std::optional<bool> train) {
  DEBUGINFO("Tensor is on: ", input.device());
  return {input, input};
}

// TODO(filhan): Even though codegen generated version should work, causes an
// autograd error. Dummy bypass for now.
std::tuple<at::Tensor, at::Tensor, at::Tensor> spyre__native_layer_norm_default(
    const at::Tensor &input, at::IntArrayRef normalized_shape,
    const ::std::optional<at::Tensor> &weight,
    const ::std::optional<at::Tensor> &bias, double eps) {
  DEBUGINFO("Tensor is on: ", input.device());
  return {input, input, input};
}

void matmul_input_assignment(
    const at::Tensor &tensor,
    std::vector<flex::DeviceMemoryAllocationPtr> &eager_inputs,
    std::size_t eager_idx, bool transpose_flag) {
  if (transpose_flag) {
    eager_idx = 1 - eager_idx;
  }
  at::Tensor tmp_tensor = tensor;
  if (tensor.dim() == 0) {
    tmp_tensor = (at::ones({1}, tensor.dtype()) * tensor).to(tensor.device());
  }
  eager_inputs[eager_idx] = (static_cast<SharedOwnerCtx *>(
                                 tmp_tensor.storage().data_ptr().get_context()))
                                ->owner;
}
// TODO(filhan): Even though codegen generated version should work,
// this manually implemented version handles more failure cases.
at::Tensor spyre__bmm_default(const at::Tensor &self, const at::Tensor &mat2) {
  DEBUGINFO("Tensor is on: ", self.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat2.device());

  bool transpose_flag = (self.sizes()[1] > self.sizes()[2]) &&
                        (mat2.sizes()[1] > mat2.sizes()[2]);
  auto out_shape_stride =
      getOutputShapeStrideForMatmul(self.sizes(), mat2.sizes(), transpose_flag);
  auto result = spyre_empty_strided(
      out_shape_stride.first, out_shape_stride.second, self.scalar_type(),
      self.layout(), self.device(), self.is_pinned());

  std::string graph_label = "Bmm";
  graph_label += "__" + std::to_string(false);
  graph_label += "__" + std::to_string(false);

  DEBUGINFO("Operating on graph with label: ", graph_label);

  // create a dummy graph builder operation to produce the right size
  std::optional<sendnn::GraphLoader> opt_gl = spyre::getCachedGraphLoader(
      graph_label, result.sizes(), result.strides());

  sendnn::ConstTensor mat2_input_sendnn_tensor;
  sendnn::GraphLoader gl;
  if (opt_gl.has_value()) {
    gl = opt_gl.value();
    mat2_input_sendnn_tensor =
        createInputTensor(gl, mat2.storage().data_ptr().get(), 1, 2);
  } else {
    sendnn::GraphBuilder gb;

    sendnn::TensorInfo ti_1 = getTensorInfo(self);
    auto inp_1 = gb.PrimaryInput("Input1", ti_1);
    sendnn::TensorInfo ti_2 = getTensorInfo(mat2);
    auto inp_2 = gb.PrimaryInput("Input2", ti_2);
    auto inp_3 = transpose_flag;
    auto inp_4 = transpose_flag;

    sendnn::TensorInfo ti = getTensorInfo(result);

    if (transpose_flag) {
      auto r = gb.BatchMatMul("Bmm", ti, inp_2, inp_1, inp_3, inp_4);
      gb.PrimaryOutput("Output", r);
    } else {
      auto r = gb.BatchMatMul("Bmm", ti, inp_1, inp_2, inp_3, inp_4);
      gb.PrimaryOutput("Output", r);
    }

    gl = prepareGraphLoader(&gb);
    gl = parseGraphLoader(gl);
    mat2_input_sendnn_tensor =
        createInputTensor(gl, mat2.storage().data_ptr().get(), 1, 2);
    auto predict_s =
        gl.Predict(sendnn::Outputs(), {mat2_input_sendnn_tensor}, 1);
    storeCachedGraphLoader(graph_label, result.sizes(), result.strides(), gl);
  }

  std::vector<flex::DeviceMemoryAllocationPtr> eagerInputs(2, nullptr);
  matmul_input_assignment(self, eagerInputs, 0, transpose_flag);
  matmul_input_assignment(mat2, eagerInputs, 1, transpose_flag);
  sendnn::ConstTensor self_input_sendnn_tensor =
      createInputTensor(gl, self.storage().data_ptr().get(), 0, 2);

  self_input_sendnn_tensor.SetSpyreData(eagerInputs[0]);
  mat2_input_sendnn_tensor.SetSpyreData(eagerInputs[1]);

  auto output_sendnn_tensor =
      createOutputTensor(gl, result.storage().data_ptr().get(), 0, 2);
  output_sendnn_tensor.SetSpyreData(
      static_cast<SharedOwnerCtx *>(result.storage().data_ptr().get_context())
          ->owner);

  auto copy_status =
      gl.Compute({output_sendnn_tensor},
                 {self_input_sendnn_tensor, mat2_input_sendnn_tensor}, 2);

  if (transpose_flag)
    result = at::transpose(result, result.dim() - 2, result.dim() - 1);

  return result;
}

// TODO(filhan): Even though codegen generated version should work, throws
// DummyOp error. Decomposed for now.
at::Tensor &spyre__addmm_out(const at::Tensor &self, const at::Tensor &mat1,
                             const at::Tensor &mat2, const at::Scalar &beta,
                             const at::Scalar &alpha, at::Tensor &out) {
  DEBUGINFO("Tensor is on: ", self.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat1.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat2.device());
  // TORCH_INTERNAL_ASSERT(self.device() == out.device());

  at::Tensor mm_result = at::mm(mat1, mat2);
  out = spyre__add_Tensor(mm_result, self, alpha);  // add does not use alpha
  return out;
}

// TODO(filhan): Even though codegen generated version should work, throws
// DummyOp error. Decomposed for now.
at::Tensor spyre__addmm_default(const at::Tensor &self, const at::Tensor &mat1,
                                const at::Tensor &mat2, const at::Scalar &beta,
                                const at::Scalar &alpha) {
  DEBUGINFO("Tensor is on: ", self.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat1.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat2.device());

  at::Tensor mm_result = at::mm(mat1, mat2);
  at::Tensor result =
      spyre__add_Tensor(mm_result, self, alpha);  // add does not use alpha
  return result;
}

// TODO(filhan): Even though codegen generated version should work, throws
// DummyOp error. Decomposed for now.
at::Tensor spyre__addmm_dtype(const at::Tensor &self, const at::Tensor &mat1,
                              const at::Tensor &mat2, at::ScalarType out_dtype,
                              const at::Scalar &beta, const at::Scalar &alpha) {
  DEBUGINFO("Tensor is on: ", self.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat1.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat2.device());

  at::Tensor mm_result = at::mm(mat1, mat2);
  at::Tensor result =
      spyre__add_Tensor(mm_result, self, alpha);  // add does not use alpha
  return result;
}

// TODO(filhan): Even though codegen generated version should work, throws
// DummyOp error. Decomposed for now.
at::Tensor &spyre__addmm_dtype_out(const at::Tensor &self,
                                   const at::Tensor &mat1,
                                   const at::Tensor &mat2,
                                   at::ScalarType out_dtype,
                                   const at::Scalar &beta,
                                   const at::Scalar &alpha, at::Tensor &out) {
  DEBUGINFO("Tensor is on: ", self.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat1.device());
  // TORCH_INTERNAL_ASSERT(self.device() == mat2.device());
  // TORCH_INTERNAL_ASSERT(self.device() == out.device());

  at::Tensor mm_result = at::mm(mat1, mat2);
  out = spyre__add_Tensor(mm_result, self, alpha);  // add does not use alpha
  return out;
}

// TODO(filhan): Even though codegen generated version should work, throws
// DummyOp error. Decomposed for now.
at::Tensor spyre__mean_dim(const at::Tensor &self, at::OptionalIntArrayRef dim,
                           bool keepdim,
                           ::std::optional<at::ScalarType> dtype) {
  DEBUGINFO("Tensor is on: ", self.device());

  at::Tensor out;
  if (dim.has_value() && dim->size() == 1 && dim->at(0) != 0) {
    out =
        spyre__sum_dim_IntList(self.transpose(0, dim->at(0)), 0, keepdim, dtype)
            .transpose(0, dim->at(0) - 1);
  } else {
    out = spyre__sum_dim_IntList(self, dim, keepdim, dtype);
  }

  if (dim.has_value()) {
    for (size_t j = 0; j < dim->size(); j++) out /= self.sizes()[dim->at(j)];
  }
  return out;
}

// TODO(filhan): Even though codegen generated version should work, throws
// DummyOp error. Decomposed for now.
at::Tensor &spyre__mean_out(const at::Tensor &self, at::OptionalIntArrayRef dim,
                            bool keepdim, ::std::optional<at::ScalarType> dtype,
                            at::Tensor &out) {
  DEBUGINFO("Tensor is on: ", self.device());
  // TORCH_INTERNAL_ASSERT(self.device() == out.device());

  out = spyre__sum_IntList_out(self, dim, keepdim, dtype, out);

  if (dim.has_value()) {
    for (size_t j = 0; j < dim->size(); j++) out /= self.sizes()[dim->at(j)];
  }
  return out;
}

// TODO(filhan): Even though codegen generated version should work, fallback for
// now.
at::Tensor &spyre__ne_Scalar_out(const at::Tensor &self,
                                 const at::Scalar &other, at::Tensor &out) {
  out = at::ne(self.to(c10::kCPU), other).to(c10::kHalf).to(c10::kPrivateUse1);
  return out;
}

// TODO(filhan): Even though codegen generated version should work, fallback for
// now.
at::Tensor spyre__cumsum_default(const at::Tensor &self, int64_t dim,
                                 ::std::optional<at::ScalarType> dtype) {
  return at::cumsum(self.to(c10::kCPU), dim).to(c10::kPrivateUse1);
}

at::Tensor spyre__clone_default(
    const at::Tensor &self, ::std::optional<at::MemoryFormat> memory_format) {
  DEBUGINFO("Tensor is on: ", self.device());

  at::Tensor result;
  if (memory_format.has_value() &&
      memory_format == at::MemoryFormat::Contiguous) {
    int n_dim = self.sizes().size();
    std::vector<int64_t> out_strides(n_dim);
    int64_t stride = 1;
    for (int i = n_dim - 1; i >= 0; i--) {
      out_strides[i] = stride;
      stride *= self.sizes()[i];
    }
    result =
        spyre_empty_strided(self.sizes(), out_strides, self.scalar_type(),
                            self.layout(), self.device(), self.is_pinned());
    DEBUGINFO("Stride: ", out_strides);
  } else {
    result =
        spyre_empty_strided(self.sizes(), self.strides(), self.scalar_type(),
                            self.layout(), self.device(), self.is_pinned());
  }

  std::string graph_label = "Clone";

  DEBUGINFO("Operating on graph with label: ", graph_label);

  // create a dummy graph builder operation to produce the right size
  std::optional<sendnn::GraphLoader> opt_gl = spyre::getCachedGraphLoader(
      graph_label, result.sizes(), result.strides());

  sendnn::GraphLoader gl;
  if (opt_gl.has_value()) {
    gl = opt_gl.value();
  } else {
    sendnn::GraphBuilder gb;

    sendnn::TensorInfo ti_1 = getTensorInfo(self);
    auto inp_1 = gb.PrimaryInput("Input1", ti_1);

    sendnn::TensorInfo ti = getTensorInfo(result);

    auto r = gb.Identity("Clone", ti, inp_1);
    gb.PrimaryOutput("Output", r);

    gl = prepareGraphLoader(&gb);
    gl = parseGraphLoader(gl);
    auto predict_s = gl.Predict(sendnn::Outputs(), sendnn::Inputs());
    storeCachedGraphLoader(graph_label, result.sizes(), result.strides(), gl);
  }
  auto input_sendnn_tensor =
      createInputTensor(gl, self.storage().data_ptr().get());

  if (self.dim() == 0) {
    at::Tensor tmp_0 = (at::ones({1}, self.dtype()) * self).to(self.device());
    input_sendnn_tensor.SetSpyreData(
        (static_cast<SharedOwnerCtx *>(
             tmp_0.storage().data_ptr().get_context()))
            ->owner);
  } else {
    input_sendnn_tensor.SetSpyreData(
        (static_cast<SharedOwnerCtx *>(self.storage().data_ptr().get_context()))
            ->owner);
  }

  auto output_sendnn_tensor =
      createOutputTensor(gl, result.storage().data_ptr().get());
  output_sendnn_tensor.SetSpyreData(
      (static_cast<SharedOwnerCtx *>(result.storage().data_ptr().get_context()))
          ->owner);

  auto copy_status =
      gl.Compute({output_sendnn_tensor}, {input_sendnn_tensor}, 1);

  return result;
}

at::Tensor &spyre__fill_Scalar(at::Tensor &self, const at::Scalar &other) {
  DEBUGINFO("Tensor is on: ", self.device());
  at::Tensor tmp = (at::ones(self.sizes(), self.dtype()) * other);
  self = spyre::spyre_copy_from(tmp, self, false);
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("view", TORCH_FN(spyre__view));
  m.impl("_local_scalar_dense", TORCH_FN(spyre___local_scalar_dense));
  m.impl("native_dropout", TORCH_FN(spyre__native_dropout_default));
  m.impl("native_layer_norm", TORCH_FN(spyre__native_layer_norm_default));
  m.impl("bmm", TORCH_FN(spyre__bmm_default));
  m.impl("addmm.dtype_out", TORCH_FN(spyre__addmm_dtype_out));
  m.impl("addmm.dtype", TORCH_FN(spyre__addmm_dtype));
  m.impl("addmm", TORCH_FN(spyre__addmm_default));
  m.impl("addmm.out", TORCH_FN(spyre__addmm_out));
  m.impl("mean.dim", TORCH_FN(spyre__mean_dim));
  m.impl("mean.out", TORCH_FN(spyre__mean_out));
  m.impl("ne.Scalar_out", TORCH_FN(spyre__ne_Scalar_out));
  m.impl("cumsum", TORCH_FN(spyre__cumsum_default));
  m.impl("clone", TORCH_FN(spyre__clone_default));
  m.impl("fill_.Scalar", TORCH_FN(spyre__fill_Scalar));
}

}  // namespace

}  // namespace spyre
