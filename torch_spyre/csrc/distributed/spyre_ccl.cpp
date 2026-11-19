/*
 * Copyright 2026 The Torch-Spyre Authors.
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
#include "spyre_ccl.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_collective_utils.h"
#include "spyre_composite_address.h"
#include "spyre_mem.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace c10d {

namespace {

// Prove that physical element i has host storage offset i. Singleton device
// dimensions do not affect the address, but every other dimension must have
// the corresponding contiguous stride. This excludes tiled multi-row layouts
// such as device_size=[2, 8, 64], stride_map=[64, 96, 1].
bool can_borrow_linear_dl16_storage(const at::Tensor& tensor) {
  if (tensor.is_neg() || tensor.is_conj()) {
    return false;
  }
  const auto layout = spyre::get_spyre_tensor_layout(tensor);
  if (layout.device_dtype != DataFormats::SEN169_FP16 ||
      layout.element_arrangement != spyre::ElementArrangement::STANDARD) {
    return false;
  }
  int64_t stride = 1;
  for (int i = static_cast<int>(layout.device_size.size()) - 1; i >= 0; --i) {
    const auto size = layout.device_size[i];
    if (size <= 0 || (size > 1 && layout.stride_map[i] != stride)) {
      return false;
    }
    stride *= size;
  }
  return true;
}

}  // namespace

/***********************************************
 * Wrapper Backend for the Sypre Collective Library
 ***********************************************/
SpyreCCLBackend::SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store,
                                 int rank, int size)
    : Backend(rank, size), group_context_(nullptr) {
  SPYRE_RUNTIME_DEBUG() << "# [Spyre CCL]: Constructor for "
                        << getBackendName();

  /*
   * Start the communication library
   * Pass it the shared runtime library handle, and default stream.
   */
  spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                  spyre::getDefaultStreamRuntimeHandle());
  group_context_ = spyre_comms::get_world_context();
  if (nullptr == group_context_) {
    std::string _err_msg =
        "[" + getBackendName() + "]: Failed to capture the world context";
    throw std::runtime_error(_err_msg);
  }
}

SpyreCCLBackend::~SpyreCCLBackend() {
  spyre_comms::finalize_library();
}

/* **********************************************
 * Internal support functions
 ********************************************** */

/**
 * @brief Converts PyTorch reduction operation type to Spyre reduction operation
 * type.
 *
 * Maps c10d::ReduceOp enum values to spyre_comms::SpyreReductionOpType enum
 * values. Currently only supports SUM operation; other operations return
 * UNSUPPORTED.
 *
 * @param reduce_op The PyTorch reduction operation type to convert
 * @return The corresponding Spyre reduction operation type, or UNSUPPORTED if
 * not supported
 */
spyre_comms::SpyreReductionOpType convert_reduce_op_type(
    const ReduceOp reduce_op) {
  switch (reduce_op) {
    case ReduceOp::SUM:
      return spyre_comms::SpyreReductionOpType::SUM;
    default:
      return spyre_comms::SpyreReductionOpType::UNSUPPORTED;
  }
}

inline std::pair<spyre_comms::TensorDataTypeEnum,
                 spyre_comms::TensorDataTypeEnum>
convert_string_to_datatype_pair(const std::string& type_name) {
  /* val-1 = type on CPU-side
   * val-2 = type on Spyre-side
   */
  static const std::unordered_map<std::string,
                                  std::pair<spyre_comms::TensorDataTypeEnum,
                                            spyre_comms::TensorDataTypeEnum>>
      type_map = {
          // Boolean and string
          {"bool",
           {spyre_comms::TensorDataTypeEnum::boolean,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"string",
           {spyre_comms::TensorDataTypeEnum::string,
            spyre_comms::TensorDataTypeEnum::string}},

          // IEEE floats
          {"fp8_143",
           {spyre_comms::TensorDataTypeEnum::float8,
            spyre_comms::TensorDataTypeEnum::sen_fp8}},
          // TODO(tmhoangt): figure out why there is not FP8 variant specific in
          // sen_datatype_enum
          {"fp8_152",
           {spyre_comms::TensorDataTypeEnum::float8,
            spyre_comms::TensorDataTypeEnum::sen_fp8}},
          {"float16",
           {spyre_comms::TensorDataTypeEnum::float16,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"float32",
           {spyre_comms::TensorDataTypeEnum::float32,
            spyre_comms::TensorDataTypeEnum::float32}},
          {"float64",
           {spyre_comms::TensorDataTypeEnum::float64,
            spyre_comms::TensorDataTypeEnum::float64}},
          {"float128",
           {spyre_comms::TensorDataTypeEnum::float128,
            spyre_comms::TensorDataTypeEnum::float128}},
          {"float256",
           {spyre_comms::TensorDataTypeEnum::float256,
            spyre_comms::TensorDataTypeEnum::float256}},

          // Decimal
          {"decimal32",
           {spyre_comms::TensorDataTypeEnum::decimal32,
            spyre_comms::TensorDataTypeEnum::decimal32}},
          {"decimal64",
           {spyre_comms::TensorDataTypeEnum::decimal64,
            spyre_comms::TensorDataTypeEnum::decimal64}},
          {"decimal128",
           {spyre_comms::TensorDataTypeEnum::decimal128,
            spyre_comms::TensorDataTypeEnum::decimal128}},

          // bfloat
          {"bfloat16",
           {spyre_comms::TensorDataTypeEnum::bfloat16,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"bfloat16_compute",
           {spyre_comms::TensorDataTypeEnum::bfloat16,
            spyre_comms::TensorDataTypeEnum::float32}},

          // Signed ints
          {"int1",
           {spyre_comms::TensorDataTypeEnum::int1,
            spyre_comms::TensorDataTypeEnum::sen_int1}},
          {"int2",
           {spyre_comms::TensorDataTypeEnum::int2,
            spyre_comms::TensorDataTypeEnum::sen_int2}},
          {"int4",
           {spyre_comms::TensorDataTypeEnum::int4,
            spyre_comms::TensorDataTypeEnum::sen_int4}},
          {"int8",
           {spyre_comms::TensorDataTypeEnum::int8,
            spyre_comms::TensorDataTypeEnum::sen_int8}},
          {"int16",
           {spyre_comms::TensorDataTypeEnum::int16,
            spyre_comms::TensorDataTypeEnum::sen_int16}},
          {"int32",
           {spyre_comms::TensorDataTypeEnum::int32,
            spyre_comms::TensorDataTypeEnum::sen_int32}},
          {"int64",
           {spyre_comms::TensorDataTypeEnum::int64,
            spyre_comms::TensorDataTypeEnum::sen_int32}},

          // Unsigned ints
          {"uint1",
           {spyre_comms::TensorDataTypeEnum::uint1,
            spyre_comms::TensorDataTypeEnum::sen_uint1}},
          {"uint2",
           {spyre_comms::TensorDataTypeEnum::uint2,
            spyre_comms::TensorDataTypeEnum::sen_uint2}},
          {"uint4",
           {spyre_comms::TensorDataTypeEnum::uint4,
            spyre_comms::TensorDataTypeEnum::sen_uint4}},
          {"uint8",
           {spyre_comms::TensorDataTypeEnum::uint8,
            spyre_comms::TensorDataTypeEnum::sen_uint8}},
          {"uint16",
           {spyre_comms::TensorDataTypeEnum::uint16,
            spyre_comms::TensorDataTypeEnum::sen_uint16}},
          {"uint32",
           {spyre_comms::TensorDataTypeEnum::uint32,
            spyre_comms::TensorDataTypeEnum::sen_uint32}},
          {"uint64",
           {spyre_comms::TensorDataTypeEnum::uint64,
            spyre_comms::TensorDataTypeEnum::sen_uint32}},

          // Quantized ints
          {"qint1",
           {spyre_comms::TensorDataTypeEnum::qint1,
            spyre_comms::TensorDataTypeEnum::qint1}},
          {"qint2",
           {spyre_comms::TensorDataTypeEnum::qint2,
            spyre_comms::TensorDataTypeEnum::qint2}},
          {"qint4",
           {spyre_comms::TensorDataTypeEnum::qint4,
            spyre_comms::TensorDataTypeEnum::qint4}},
          {"qint8",
           {spyre_comms::TensorDataTypeEnum::qint8,
            spyre_comms::TensorDataTypeEnum::qint8}},
          {"qint16",
           {spyre_comms::TensorDataTypeEnum::qint16,
            spyre_comms::TensorDataTypeEnum::qint16}},
          {"qint32",
           {spyre_comms::TensorDataTypeEnum::qint32,
            spyre_comms::TensorDataTypeEnum::qint32}},
          {"qint64",
           {spyre_comms::TensorDataTypeEnum::qint64,
            spyre_comms::TensorDataTypeEnum::qint64}},

          {"quint1",
           {spyre_comms::TensorDataTypeEnum::quint1,
            spyre_comms::TensorDataTypeEnum::quint1}},
          {"quint2",
           {spyre_comms::TensorDataTypeEnum::quint2,
            spyre_comms::TensorDataTypeEnum::quint2}},
          {"quint4",
           {spyre_comms::TensorDataTypeEnum::quint4,
            spyre_comms::TensorDataTypeEnum::quint4}},
          {"quint8",
           {spyre_comms::TensorDataTypeEnum::quint8,
            spyre_comms::TensorDataTypeEnum::quint8}},
          {"quint16",
           {spyre_comms::TensorDataTypeEnum::quint16,
            spyre_comms::TensorDataTypeEnum::quint16}},
          {"quint32",
           {spyre_comms::TensorDataTypeEnum::quint32,
            spyre_comms::TensorDataTypeEnum::quint32}},
          {"quint64",
           {spyre_comms::TensorDataTypeEnum::quint64,
            spyre_comms::TensorDataTypeEnum::quint64}},

          // Complex
          {"complex64",
           {spyre_comms::TensorDataTypeEnum::complex64,
            spyre_comms::TensorDataTypeEnum::complex64}},
          {"complex128",
           {spyre_comms::TensorDataTypeEnum::complex128,
            spyre_comms::TensorDataTypeEnum::complex128}},

          // Sentient types
          {"sen_fp8",
           {spyre_comms::TensorDataTypeEnum::sen_fp8,
            spyre_comms::TensorDataTypeEnum::sen_fp8}},
          {"sen_fp16",
           {spyre_comms::TensorDataTypeEnum::sen_fp16,
            spyre_comms::TensorDataTypeEnum::sen_fp16}},
          {"sen_fp8_compute",
           {spyre_comms::TensorDataTypeEnum::sen_fp8,
            spyre_comms::TensorDataTypeEnum::float32}},
          {"sen_fp16_compute",
           {spyre_comms::TensorDataTypeEnum::sen_fp16,
            spyre_comms::TensorDataTypeEnum::float32}},

          {"sen_int1",
           {spyre_comms::TensorDataTypeEnum::sen_int1,
            spyre_comms::TensorDataTypeEnum::sen_int1}},
          {"sen_int2",
           {spyre_comms::TensorDataTypeEnum::sen_int2,
            spyre_comms::TensorDataTypeEnum::sen_int2}},
          {"sen_int4",
           {spyre_comms::TensorDataTypeEnum::sen_int4,
            spyre_comms::TensorDataTypeEnum::sen_int4}},
          {"sen_int8",
           {spyre_comms::TensorDataTypeEnum::sen_int8,
            spyre_comms::TensorDataTypeEnum::sen_int8}},
          {"sen_int16",
           {spyre_comms::TensorDataTypeEnum::sen_int16,
            spyre_comms::TensorDataTypeEnum::sen_int16}},
          {"sen_int24",
           {spyre_comms::TensorDataTypeEnum::sen_int24,
            spyre_comms::TensorDataTypeEnum::sen_int24}},
          {"sen_int32",
           {spyre_comms::TensorDataTypeEnum::sen_int32,
            spyre_comms::TensorDataTypeEnum::sen_int32}},
          {"sen_int4_compute",
           {spyre_comms::TensorDataTypeEnum::sen_int4,
            spyre_comms::TensorDataTypeEnum::int32}},
          {"sen_int8_compute",
           {spyre_comms::TensorDataTypeEnum::sen_int8,
            spyre_comms::TensorDataTypeEnum::int32}},

          {"sen_uint1",
           {spyre_comms::TensorDataTypeEnum::sen_uint1,
            spyre_comms::TensorDataTypeEnum::sen_uint1}},
          {"sen_uint2",
           {spyre_comms::TensorDataTypeEnum::sen_uint2,
            spyre_comms::TensorDataTypeEnum::sen_uint2}},
          {"sen_uint4",
           {spyre_comms::TensorDataTypeEnum::sen_uint4,
            spyre_comms::TensorDataTypeEnum::sen_uint4}},
          {"sen_uint8",
           {spyre_comms::TensorDataTypeEnum::sen_uint8,
            spyre_comms::TensorDataTypeEnum::sen_uint8}},
          {"sen_uint16",
           {spyre_comms::TensorDataTypeEnum::sen_uint16,
            spyre_comms::TensorDataTypeEnum::sen_uint16}},
          {"sen_uint24",
           {spyre_comms::TensorDataTypeEnum::sen_uint24,
            spyre_comms::TensorDataTypeEnum::sen_uint24}},
          {"sen_uint32",
           {spyre_comms::TensorDataTypeEnum::sen_uint32,
            spyre_comms::TensorDataTypeEnum::sen_uint32}},
      };

  auto it = type_map.find(type_name);
  if (it != type_map.end()) {
    return it->second;
  }
  return {spyre_comms::TensorDataTypeEnum::dt_undef,
          spyre_comms::TensorDataTypeEnum::dt_undef};
}

spyre_comms::TensorInfo SpyreCCLBackend::getTensorInfo(
    const at::Tensor& input) {
  spyre::check_collective_storage(input);
  auto str_type = spyre::torchScalarToString[input.scalar_type()];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      convert_string_to_datatype_pair(str_type);
  auto dtype = sen_dtype_cpu;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    // convert_string_to_datatype_pair("bfloat16") is {bfloat16, sen_fp16}.
    // check_collective_storage verified that the borrowed bytes are DL16.
    TORCH_INTERNAL_ASSERT(sen_dtype_dev ==
                          spyre_comms::TensorDataTypeEnum::sen_fp16);
    dtype = sen_dtype_dev;
  }
  const auto bytes =
      spyre::get_device_size_in_bytes(spyre::get_spyre_tensor_layout(input));
  const auto element_bytes = spyre_comms::DataTypeSizeBytes(dtype);
  TORCH_CHECK(element_bytes > 0 && bytes % element_bytes == 0,
              "Collective storage size is incompatible with its data type");
  // The comms buffer is flat device storage, including stick padding. Count
  // in the planner dtype's units so copy-only host/device width differences
  // (for example int64) retain exactly the same byte coverage.
  return spyre_comms::TensorInfo{
      dtype,
      spyre_comms::TensorShape({static_cast<int64_t>(bytes / element_bytes)})};
}

/**
 * @brief Prepares a PyTorch tensor for use with Spyre communication operations.
 *
 * Converts a PyTorch tensor to a spyre_comms::Tensor with proper metadata.
 *
 * @param input_tensor The PyTorch tensor to prepare (must be on device)
 * @param output_tensor Pointer to the spyre_comms::Tensor to populate with
 * prepared data
 */
void SpyreCCLBackend::prepare_tensor(const at::Tensor& input_tensor,
                                     spyre_comms::Tensor* output_tensor) {
  spyre_comms::TensorInfo tensor_info = getTensorInfo(input_tensor);
  *output_tensor =
      spyre_comms::Tensor(tensor_info, input_tensor.storage().data_ptr().get());
  // Update the data pointer on the object with a borrowed pointer to the
  // tensor's flex::CompositeAddress; the borrowing setter ensures spyre-comms
  // will never try to delete it.
  output_tensor->SetSpyreDeviceAddressBorrowed(
      spyre::get_composite_address(input_tensor));
}

/**
 * @brief Validates that a single tensor meets requirements for collective
 * operations.
 *
 * Checks that the tensor is contiguous, dense (not sparse), and suitable for
 * Spyre device operations. Throws an error if any validation fails.
 *
 * @param tensor The tensor to validate
 * @throws TORCH_CHECK exception if tensor is not contiguous or is sparse
 */
void SpyreCCLBackend::check_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "The tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "The tensor has to be dense");
  }
  // Add check for spyre device tensor
}

/**
 * @brief Validates a vector of tensors for collective operations.
 *
 * Checks that the number of tensors in the vector is within the specified range
 * [min_allowed, max_allowed], and validates each individual tensor using
 * check_single_tensor(). Throws an error if validation fails.
 *
 * @param tensors The vector of tensors to validate
 * @param min_allowed Minimum number of tensors allowed in the vector
 * @param max_allowed Maximum number of tensors allowed in the vector
 * @throws TORCH_CHECK exception if tensor count is out of range or any tensor
 * is invalid
 */
void SpyreCCLBackend::check_vector_tensor(
    const std::vector<at::Tensor>& tensors, int min_allowed, int max_allowed) {
  if (static_cast<int>(tensors.size()) < min_allowed) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Too few tensors. Expected at least " +
                           std::to_string(min_allowed) +
                           " Actual: " + std::to_string(tensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  if (static_cast<int>(tensors.size()) > max_allowed) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Too many tensors. Expected at most " +
                           std::to_string(max_allowed) +
                           " Actual: " + std::to_string(tensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  for (auto& tensor : tensors) {
    check_single_tensor(tensor);
  }
}

void SpyreCCLBackend::check_reduction(const at::Tensor& tensor,
                                      const std::string& op_name,
                                      const ReduceOp& reduce_op) {
  if (reduce_op != ReduceOp::SUM) {
    std::string _err_msg = "[" + getBackendName() + "]: " + op_name +
                           " only supports SUM operation." +
                           " Actual: " + std::to_string(reduce_op);
    TORCH_CHECK(false, _err_msg);
  }
  // FP16 and BF16 both use DL16 for device-side SUM.
  if (tensor.scalar_type() != c10::ScalarType::Half &&
      tensor.scalar_type() != c10::ScalarType::BFloat16) {
    std::string _err_msg =
        "[" + getBackendName() + "]: " + op_name +
        " only supports float16 and bfloat16 tensors." +
        " Actual: " + spyre::torchScalarToString[tensor.scalar_type()];
    TORCH_CHECK(false, _err_msg);
  }
}

void SpyreCCLBackend::check_dim0_concat(const at::Tensor& whole,
                                        const at::Tensor& part,
                                        const std::string& op_name) {
  const int64_t group_size = group_context_->getSize();
  TORCH_CHECK(whole.scalar_type() == part.scalar_type(), "[", getBackendName(),
              "]: ", op_name,
              ": input and output dtypes differ: ", whole.scalar_type(), " vs ",
              part.scalar_type());
  TORCH_CHECK(part.dim() > 0 && whole.dim() == part.dim() &&
                  whole.size(0) == group_size * part.size(0) &&
                  whole.sizes().slice(1) == part.sizes().slice(1),
              "[", getBackendName(), "]: ", op_name,
              ": expected the per-rank tensor ", part.sizes(), " to be 1/",
              group_size, " of the concatenated tensor along dim 0; got ",
              whole.sizes());
}

std::vector<at::Tensor> SpyreCCLBackend::make_rank_buffers(
    const at::Tensor& like) {
  const auto layout = spyre::get_spyre_tensor_layout(like);
  std::vector<at::Tensor> buffers;
  const auto group_size = static_cast<int64_t>(group_context_->getSize());
  for (int64_t r = 0; r < group_size; r++) {
    buffers.push_back(spyre::spyre_empty_with_layout(
        like.sizes(), like.strides(), like.scalar_type(), layout,
        like.device()));
  }
  return buffers;
}

/* **********************************************
 * Interface functions
 ********************************************** */
c10::intrusive_ptr<Work> SpyreCCLBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
  if (static_cast<int>(outputTensors.size()) != 1) {
    std::string _err_msg =
        "[" + getBackendName() +
        "]: Too many tensors in the output list. Expected exactly 1" +
        " Actual: " + std::to_string(outputTensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  if (static_cast<int>(outputTensors[0].size()) !=
      static_cast<int>(group_context_->getSize())) {
    std::string _err_msg =
        "[" + getBackendName() +
        "]: Incorrect output list size. The list size should be exactly " +
        std::to_string(group_context_->getSize()) +
        " Actual: " + std::to_string(outputTensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  // check_single_tensor enforces logical contiguity here, before either the
  // borrow or clone path. A linear base layout alone cannot validate a view's
  // logical order (for example, a zero-offset slice with step 2).
  check_vector_tensor(inputTensors, 1, 1);

  auto input = inputTensors[0];
  spyre::check_collective_device(input);
  const auto descriptor_numel =
      c10::multiply_integers(spyre::get_spyre_tensor_sizes(input));
  TORCH_INTERNAL_ASSERT(
      descriptor_numel >= input.numel(),
      "Collective input exceeds its device descriptor extent");

  std::vector<spyre_comms::Tensor> output_tensors;
  for (auto& outputTensor : outputTensors[0]) {
    spyre_comms::Tensor output_tensor;
    prepare_tensor(outputTensor, &output_tensor);
    output_tensors.push_back(output_tensor);
  }

  bool borrow_prefix = false;
  if (input.storage_offset() == 0 && descriptor_numel > input.numel()) {
    const auto input_bytes =
        spyre::get_device_size_in_bytes(spyre::get_spyre_tensor_layout(input));
    const auto output_bytes = spyre::get_device_size_in_bytes(
        spyre::get_spyre_tensor_layout(outputTensors[0][0]));
    borrow_prefix =
        (input.scalar_type() == c10::ScalarType::Half ||
         input.scalar_type() == c10::ScalarType::BFloat16) &&
        can_borrow_linear_dl16_storage(input) && output_bytes <= input_bytes &&
        std::all_of(outputTensors[0].begin(), outputTensors[0].end(),
                    [&](const at::Tensor& output) {
                      return output.sizes() == input.sizes() &&
                             output.scalar_type() == input.scalar_type() &&
                             can_borrow_linear_dl16_storage(output) &&
                             spyre::get_device_size_in_bytes(
                                 spyre::get_spyre_tensor_layout(output)) ==
                                 output_bytes;
                    });
    if (!borrow_prefix) {
      // Tiled prefixes need materialization. This handles the logits
      // allgather caller; mutating collectives would also need copy-back.
      input = input.clone(at::MemoryFormat::Contiguous);
    }
  }
  spyre_comms::Tensor input_tensor;
  if (borrow_prefix) {
    // Single-row logits are a linear device prefix. Limit the transfer to the
    // output allocation, including its last stick's padding, without changing
    // the input's descriptor or copying its data. Keep the original alive.
    input_tensor =
        spyre_comms::Tensor(spyre_comms::TensorInfo(output_tensors[0]),
                            input.storage().data_ptr().get());
    input_tensor.SetSpyreDeviceAddressBorrowed(
        spyre::get_composite_address(input));
  } else {
    prepare_tensor(input, &input_tensor);
  }

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::ALLGATHER);
  work->keep_alive_.push_back(input);
  work->add_schedule(group_context_->allgather(output_tensors, input_tensor));
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::all_gather_single(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  check_single_tensor(outputBuffer);
  check_single_tensor(inputBuffer);
  check_dim0_concat(outputBuffer, inputBuffer, "all_gather_single");
  // Unlike the list allgather, accept no view whose device descriptor covers
  // more than the tensor.
  spyre::check_collective_storage(inputBuffer);

  // spyre-comms' single-buffer allgather places rank r's input in the r-th
  // contiguous slab of the flat device output. Under the default stick layout
  // host dim 0 is not the outermost device dim ((4, 128) fp16 is
  // device_size=[2, 4, 64]), so that slab is not rows [r * n, (r + 1) * n) of
  // outputBuffer, and spyre-comms would move the wrong elements without an
  // error. As the compiled path does (spyre_allgather_run_impl), gather into
  // one buffer per rank, then concatenate them into outputBuffer once the
  // transfer completes. cat_out, unlike copying into outputBuffer.narrow(),
  // also handles rows whose offset is not stick-aligned (padded last dims).
  std::vector<std::vector<at::Tensor>> rank_outputs{
      make_rank_buffers(inputBuffer)};
  std::vector<at::Tensor> inputs{inputBuffer};
  AllgatherOptions start_opts = opts;
  start_opts.asyncOp = true;
  auto work = c10::static_intrusive_pointer_cast<SpyreCCLWork>(
      allgather(rank_outputs, inputs, start_opts));
  work->keep_alive_.insert(work->keep_alive_.end(), rank_outputs[0].begin(),
                           rank_outputs[0].end());
  work->on_complete_.emplace_back(
      [output = outputBuffer, buffers = rank_outputs[0]]() mutable {
        at::cat_out(output, buffers, 0);
      });
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
  TORCH_CHECK(outputTensorLists.size() == inputTensors.size(), "[",
              getBackendName(),
              "]: allgather_coalesced expects one output list per input; got ",
              outputTensorLists.size(), " output lists for ",
              inputTensors.size(), " inputs");
  // spyre-comms has no coalesced allgather: issue one allgather per input and
  // complete them as one work.
  auto work = c10::make_intrusive<SpyreCCLWork>(OpType::ALLGATHER_COALESCED);
  AllgatherOptions start_opts = opts;
  start_opts.asyncOp = true;
  for (size_t i = 0; i < inputTensors.size(); i++) {
    std::vector<std::vector<at::Tensor>> outputs{outputTensorLists[i]};
    std::vector<at::Tensor> inputs{inputTensors[i]};
    work->absorb(*c10::static_intrusive_pointer_cast<SpyreCCLWork>(
        allgather(outputs, inputs, start_opts)));
  }
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::all_gather_single_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  TORCH_CHECK(outputs.size() == inputs.size(), "[", getBackendName(),
              "]: all_gather_single_coalesced expects one output per input; "
              "got ",
              outputs.size(), " outputs for ", inputs.size(), " inputs");
  // spyre-comms has no coalesced allgather: issue one all_gather_single per
  // input and complete them as one work.
  auto work = c10::make_intrusive<SpyreCCLWork>(
      OpType::ALLGATHER_INTO_TENSOR_COALESCED);
  AllgatherOptions start_opts = opts;
  start_opts.asyncOp = true;
  for (size_t i = 0; i < inputs.size(); i++) {
    work->absorb(*c10::static_intrusive_pointer_cast<SpyreCCLWork>(
        all_gather_single(outputs[i], inputs[i], start_opts)));
  }
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  check_vector_tensor(tensors, 1, 1);
  check_reduction(tensors[0], "Allreduce", opts.reduceOp);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::ALLREDUCE);
  work->add_schedule(
      group_context_->allreduce(tensor, convert_reduce_op_type(opts.reduceOp)));
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  // Validate every tensor before any transfer starts.
  for (const auto& tensor : tensors) {
    check_single_tensor(tensor);
    check_reduction(tensor, "Allreduce_coalesced", opts.reduceOp);
  }
  // spyre-comms has no coalesced allreduce: issue one allreduce per tensor and
  // complete them as one work.
  auto work = c10::make_intrusive<SpyreCCLWork>(OpType::ALLREDUCE_COALESCED);
  AllreduceOptions start_opts = opts;
  start_opts.asyncOp = true;
  for (auto& tensor : tensors) {
    std::vector<at::Tensor> one{tensor};
    work->absorb(*c10::static_intrusive_pointer_cast<SpyreCCLWork>(
        allreduce(one, start_opts)));
  }
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts) {
  const int group_size = static_cast<int>(group_context_->getSize());
  check_vector_tensor(outputTensors, group_size, group_size);
  check_vector_tensor(inputTensors, group_size, group_size);
  const auto& like = outputTensors[0];
  std::vector<int64_t> output_rows;
  std::vector<int64_t> input_rows;
  for (const auto* list : {&outputTensors, &inputTensors}) {
    for (const auto& tensor : *list) {
      TORCH_CHECK(tensor.dim() > 0 && tensor.dim() == like.dim() &&
                      tensor.sizes().slice(1) == like.sizes().slice(1) &&
                      tensor.scalar_type() == like.scalar_type(),
                  "[", getBackendName(),
                  "]: alltoall: every tensor must have the dtype and the "
                  "sizes past dim 0 of the first output (",
                  like.scalar_type(), ", ", like.sizes(), "); got ",
                  tensor.scalar_type(), ", ", tensor.sizes());
      (list == &outputTensors ? output_rows : input_rows)
          .push_back(tensor.size(0));
    }
  }

  // spyre-comms has only the single-buffer alltoallw. Pack the inputs along
  // dim 0, exchange with all_to_all_single (which checks the layout limits),
  // and copy each received slice into its output once complete.
  at::Tensor packed_input = at::cat(inputTensors, 0);
  auto packed_sizes = like.sizes().vec();
  packed_sizes[0] =
      std::accumulate(output_rows.begin(), output_rows.end(), int64_t{0});
  at::Tensor packed_output = at::empty(packed_sizes, like.options());

  // A Spyre device-to-device copy out of a sliced view needs a stick-aligned
  // storage offset. Check it before any data moves.
  const int64_t row_numel = like.numel() / like.size(0);
  const int64_t stick =
      spyre::get_spyre_tensor_layout(packed_output).elems_per_stick();
  std::vector<int64_t> offsets;
  int64_t offset = 0;
  for (size_t j = 0; j < output_rows.size(); j++) {
    TORCH_CHECK((offset * row_numel) % stick == 0, "[", getBackendName(),
                "]: alltoall: output ", j, " starts at element ",
                offset * row_numel,
                " of the packed output, which is not stick-aligned (", stick,
                " elements); the copy into it is not supported");
    offsets.push_back(offset);
    offset += output_rows[j];
  }

  AllToAllOptions start_opts = opts;
  start_opts.asyncOp = true;
  auto work =
      c10::static_intrusive_pointer_cast<SpyreCCLWork>(all_to_all_single(
          packed_output, packed_input, output_rows, input_rows, start_opts));
  work->keep_alive_.push_back(packed_input);
  work->keep_alive_.push_back(packed_output);
  work->on_complete_.emplace_back(
      [outputs = outputTensors, packed_output, offsets, output_rows]() mutable {
        for (size_t j = 0; j < outputs.size(); j++) {
          outputs[j].copy_(packed_output.narrow(0, offsets[j], output_rows[j]));
        }
      });
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::all_to_all_single(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes, const AllToAllOptions& opts) {
  check_single_tensor(outputBuffer);
  check_single_tensor(inputBuffer);
  const std::string prefix = "[" + getBackendName() + "]: all_to_all_single";
  TORCH_CHECK(outputBuffer.scalar_type() == inputBuffer.scalar_type(), prefix,
              ": input and output dtypes differ: ", inputBuffer.scalar_type(),
              " vs ", outputBuffer.scalar_type());
  TORCH_CHECK(
      inputBuffer.dim() > 0 && outputBuffer.dim() == inputBuffer.dim() &&
          outputBuffer.sizes().slice(1) == inputBuffer.sizes().slice(1),
      prefix, ": input and output must have the same sizes past dim 0; got ",
      inputBuffer.sizes(), " and ", outputBuffer.sizes());

  const int64_t group_size = group_context_->getSize();
  // dist.all_to_all_single passes None split sizes as empty vectors, which
  // mean an even split of dim 0.
  auto resolve_rows = [&](std::vector<int64_t> rows, const at::Tensor& tensor,
                          const char* which) {
    if (rows.empty()) {
      TORCH_CHECK(tensor.size(0) % group_size == 0, prefix, ": ", which,
                  " dim 0 (", tensor.size(0),
                  ") must be divisible by the group size (", group_size,
                  ") when no split sizes are given");
      rows.assign(group_size, tensor.size(0) / group_size);
    }
    TORCH_CHECK(static_cast<int64_t>(rows.size()) == group_size, prefix,
                ": expected ", group_size, " ", which, " split sizes, got ",
                rows.size());
    TORCH_CHECK(
        std::accumulate(rows.begin(), rows.end(), int64_t{0}) == tensor.size(0),
        prefix, ": ", which, " split sizes ", c10::IntArrayRef(rows),
        " do not sum to dim 0 (", tensor.size(0), ")");
    return rows;
  };
  const auto output_rows =
      resolve_rows(outputSplitSizes, outputBuffer, "output");
  const auto input_rows = resolve_rows(inputSplitSizes, inputBuffer, "input");

  // spyre-comms splits the flat device buffer, so a dim-0 split is only
  // correct when dim-0 rows are contiguous on device. Same checks as
  // AllToAllKernel._plan_call (_inductor/ir.py); keep the two in sync.
  const int64_t row_bytes = spyre::dim0_row_device_bytes(inputBuffer, "input");
  const int64_t output_row_bytes =
      spyre::dim0_row_device_bytes(outputBuffer, "output");
  TORCH_CHECK(row_bytes == output_row_bytes, prefix,
              ": input and output rows span different device sizes (",
              row_bytes, " vs ", output_row_bytes, " bytes)");
  // Each split must be whole 128-byte sticks: spyre-comms has no algorithm
  // for partial sticks and can abort the process instead of raising.
  for (const auto* rows : {&output_rows, &input_rows}) {
    for (int64_t r : *rows) {
      TORCH_CHECK((r * row_bytes) % 128 == 0, prefix, ": a split of ", r,
                  " rows is ", r * row_bytes,
                  " bytes on device; each split must be a multiple of 128 "
                  "bytes (splits: output=",
                  c10::IntArrayRef(output_rows),
                  ", input=", c10::IntArrayRef(input_rows), ")");
    }
  }

  spyre_comms::Tensor input_tensor;
  prepare_tensor(inputBuffer, &input_tensor);
  spyre_comms::Tensor output_tensor;
  prepare_tensor(outputBuffer, &output_tensor);

  // prepare_tensor describes each tensor as a flat buffer of device elements;
  // spyre-comms takes the split sizes in those units.
  const int64_t buffer_bytes =
      static_cast<int64_t>(spyre::get_device_size_in_bytes(
          spyre::get_spyre_tensor_layout(inputBuffer)));
  TORCH_INTERNAL_ASSERT(input_tensor.Volume() > 0 &&
                        buffer_bytes % input_tensor.Volume() == 0);
  const int64_t elem_bytes = buffer_bytes / input_tensor.Volume();
  TORCH_INTERNAL_ASSERT(128 % elem_bytes == 0);
  auto to_elems = [&](const std::vector<int64_t>& rows) {
    std::vector<int64_t> elems;
    for (int64_t r : rows) {
      elems.push_back(r * row_bytes / elem_bytes);
    }
    return elems;
  };
  const auto output_splits =
      spyre::to_alltoall_splits(to_elems(output_rows), "output");
  const auto input_splits =
      spyre::to_alltoall_splits(to_elems(input_rows), "input");

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::ALLTOALL_BASE);
  // Note: alltoallw takes (output, input).
  work->add_schedule(group_context_->alltoallw(output_tensor, input_tensor,
                                               output_splits, input_splits));
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::barrier(const BarrierOptions& opts) {
  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::BARRIER);
  work->add_schedule(group_context_->barrier());
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  check_vector_tensor(tensors, 1, 1);
  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::BROADCAST);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  work->add_schedule(group_context_->broadcast(tensor, opts.rootRank));
  if (!opts.asyncOp) {
    work->wait();
  }

  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const GatherOptions& opts) {
  if (opts.rootRank == group_context_->getRank()) {
    if (static_cast<int>(outputTensors.size()) != 1) {
      std::string _err_msg =
          "[" + getBackendName() +
          "]: Too many tensors in the output list. Expected exactly 1" +
          " Actual: " + std::to_string(outputTensors.size());
      TORCH_CHECK(false, _err_msg);
    }
    if (static_cast<int>(outputTensors[0].size()) !=
        static_cast<int>(group_context_->getSize())) {
      std::string _err_msg =
          "[" + getBackendName() +
          "]: Incorrect output list size. The list size should be exactly " +
          std::to_string(group_context_->getSize()) +
          " Actual: " + std::to_string(outputTensors.size());
      TORCH_CHECK(false, _err_msg);
    }
  }
  check_vector_tensor(inputTensors, 1, 1);

  spyre_comms::Tensor input_tensor;
  prepare_tensor(inputTensors[0], &input_tensor);

  std::vector<spyre_comms::Tensor> output_tensors;
  if (opts.rootRank == group_context_->getRank()) {
    for (auto& outputTensor : outputTensors[0]) {
      spyre_comms::Tensor output_tensor;
      prepare_tensor(outputTensor, &output_tensor);
      output_tensors.push_back(output_tensor);
    }
  }

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::GATHER);
  work->add_schedule(
      group_context_->gather(output_tensors, input_tensor, opts.rootRank));
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::gather_single(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const GatherOptions& opts) {
  check_single_tensor(inputBuffer);
  spyre::check_collective_storage(inputBuffer);
  const bool is_root =
      opts.rootRank == static_cast<int64_t>(group_context_->getRank());
  // Same layout issue as all_gather_single: the single-buffer gather puts
  // rank r's input in the r-th flat device slab, which is not rows
  // [r * n, (r + 1) * n) of outputBuffer. Gather into one buffer per rank on
  // the root and concatenate them into outputBuffer once complete. Non-root
  // ranks get an empty placeholder output from c10d and do not touch it.
  std::vector<std::vector<at::Tensor>> rank_outputs;
  if (is_root) {
    check_single_tensor(outputBuffer);
    check_dim0_concat(outputBuffer, inputBuffer, "gather_single");
    rank_outputs.push_back(make_rank_buffers(inputBuffer));
  }
  std::vector<at::Tensor> inputs{inputBuffer};
  GatherOptions start_opts = opts;
  start_opts.asyncOp = true;
  auto work = c10::static_intrusive_pointer_cast<SpyreCCLWork>(
      gather(rank_outputs, inputs, start_opts));
  if (is_root) {
    work->keep_alive_.insert(work->keep_alive_.end(), rank_outputs[0].begin(),
                             rank_outputs[0].end());
    work->on_complete_.emplace_back(
        [output = outputBuffer, buffers = rank_outputs[0]]() mutable {
          at::cat_out(output, buffers, 0);
        });
  }
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  check_vector_tensor(tensors, 1, 1);
  check_reduction(tensors[0], "Reduce", opts.reduceOp);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::REDUCE);
  work->add_schedule(group_context_->reduce(
      tensor, convert_reduce_op_type(opts.reduceOp), opts.rootRank));
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  const int group_size = static_cast<int>(group_context_->getSize());
  TORCH_CHECK(inputTensors.size() == 1, "[", getBackendName(),
              "]: reduce_scatter expects exactly 1 input list; got ",
              inputTensors.size());
  check_vector_tensor(outputTensors, 1, 1);
  check_vector_tensor(inputTensors[0], group_size, group_size);
  const auto& output = outputTensors[0];
  for (const auto& input : inputTensors[0]) {
    TORCH_CHECK(input.sizes() == output.sizes() &&
                    input.scalar_type() == output.scalar_type(),
                "[", getBackendName(),
                "]: reduce_scatter: every input must match the output (",
                output.scalar_type(), ", ", output.sizes(), "); got ",
                input.scalar_type(), ", ", input.sizes());
  }
  // spyre-comms reduce_scatter takes one buffer: pack the per-rank inputs
  // along dim 0 and reuse reduce_scatter_single. That stages its own copy, so
  // the packed tensor need not outlive this call.
  at::Tensor packed = at::cat(inputTensors[0], 0);
  return reduce_scatter_single(outputTensors[0], packed, opts);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter_single(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const ReduceScatterOptions& opts) {
  check_single_tensor(outputBuffer);
  check_single_tensor(inputBuffer);
  check_reduction(inputBuffer, "Reduce_scatter", opts.reduceOp);
  check_dim0_concat(inputBuffer, outputBuffer, "reduce_scatter_single");
  const int64_t group_size = group_context_->getSize();
  const std::string prefix = "[" + getBackendName() + "]: reduce_scatter";
  TORCH_CHECK(outputBuffer.numel() > 0, prefix, ": tensors must be non-empty");

  // spyre-comms hands rank r the r-th contiguous 1/group_size of the flat
  // device input, and accumulates in the input buffer. Under the default
  // stick layout host dim 0 is not the outermost device dim, so that slab is
  // not inputBuffer.chunk(group_size)[r]. As the compiled path does
  // (_reduce_scatter in _inductor/lowering.py), stage a copy shaped
  // (A, group_size, B) -- B the last dim, A the rest of one rank's chunk --
  // whose default layout puts the group dim outermost. The copy also leaves
  // the caller's input unmodified.
  const int64_t b_dim = outputBuffer.size(-1);
  const int64_t a_dim = outputBuffer.numel() / b_dim;
  const auto dtype = inputBuffer.scalar_type();
  if (outputBuffer.dim() == 1) {
    // Same limit as the compiled path: staging a 1-D chunk that splits a
    // stick is not supported.
    const int64_t stick =
        spyre::SpyreTensorLayout({b_dim}, dtype).elems_per_stick();
    TORCH_CHECK(b_dim % stick == 0, prefix, ": a 1-D per-rank chunk (", b_dim,
                " elements) must be a multiple of the stick size (", stick,
                " elements)");
  }
  // Allocate with the default layout, then copy: contiguous() would return
  // the view itself when the permute is trivial (A == 1).
  at::Tensor staged =
      at::empty({a_dim, group_size, b_dim}, inputBuffer.options());
  staged.copy_(inputBuffer.view({group_size, a_dim, b_dim}).permute({1, 0, 2}));
  const auto staged_stl =
      spyre::check_reduce_scatter_staged_input(staged, group_size);

  // Each slab is the default device image of an (A, B) tensor. Receive
  // straight into outputBuffer when that is its layout, otherwise into a
  // temporary that is copied into outputBuffer once complete.
  const std::vector<int64_t> slab_sizes{a_dim, b_dim};
  const bool direct = outputBuffer.sizes() == c10::IntArrayRef(slab_sizes) &&
                      spyre::get_spyre_tensor_layout(outputBuffer) ==
                          spyre::SpyreTensorLayout(slab_sizes, dtype);
  at::Tensor result =
      direct ? outputBuffer : at::empty(slab_sizes, outputBuffer.options());
  spyre::check_reduce_scatter_output_tiles(result, staged_stl, group_size);

  spyre_comms::Tensor input_tensor;
  prepare_tensor(staged, &input_tensor);
  spyre_comms::Tensor output_tensor;
  prepare_tensor(result, &output_tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::REDUCE_SCATTER);
  work->keep_alive_.push_back(staged);
  if (!direct) {
    work->keep_alive_.push_back(result);
    work->on_complete_.emplace_back([output = outputBuffer, result]() mutable {
      output.copy_(result.view(output.sizes()));
    });
  }
  // Note: spyre-comms reduce_scatter takes (input, output).
  work->add_schedule(group_context_->reduce_scatter(
      input_tensor, output_tensor, convert_reduce_op_type(opts.reduceOp)));
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter_single_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputs.size() == inputs.size(), "[", getBackendName(),
              "]: reduce_scatter_single_coalesced expects one output per "
              "input; got ",
              outputs.size(), " outputs for ", inputs.size(), " inputs");
  // Validate every tensor before any transfer starts.
  for (const auto& input : inputs) {
    check_reduction(input, "Reduce_scatter_coalesced", opts.reduceOp);
  }
  // spyre-comms has no coalesced reduce_scatter: issue one
  // reduce_scatter_single per input and complete them as one work.
  auto work = c10::make_intrusive<SpyreCCLWork>(
      OpType::REDUCE_SCATTER_TENSOR_COALESCED);
  ReduceScatterOptions start_opts = opts;
  start_opts.asyncOp = true;
  for (size_t i = 0; i < inputs.size(); i++) {
    work->absorb(*c10::static_intrusive_pointer_cast<SpyreCCLWork>(
        reduce_scatter_single(outputs[i], inputs[i], start_opts)));
  }
  if (!opts.asyncOp) {
    work->wait();
  }
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  // Do not intend to support: spyre-comms has no scatter primitive.
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::send(std::vector<at::Tensor>& tensors,
                                               int dstRank, int tag) {
  // NOTE: The c10d::Backend::send() signature carries no opts struct and
  // therefore exposes no asyncOp flag.  We follow the same non-blocking
  // submission pattern used by all collectives and leave completion control
  // to the caller via Work::wait().
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::SEND);
  work->add_schedule(group_context_->send(tensor, dstRank, tag));
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recv(std::vector<at::Tensor>& tensors,
                                               int srcRank, int tag) {
  // NOTE: The c10d::Backend::recv() signature carries no opts struct and
  // therefore exposes no asyncOp flag.  We follow the same non-blocking
  // submission pattern used by all collectives and leave completion control
  // to the caller via Work::wait().
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::Tensor tensor;
  prepare_tensor(tensors[0], &tensor);

  c10::intrusive_ptr<SpyreCCLWork> work =
      c10::make_intrusive<SpyreCCLWork>(OpType::RECV);
  work->add_schedule(group_context_->recv(tensor, srcRank, tag));
  return work;
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  // Do not intend to support: Too much protocol overhead, and not commonly used
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Backend> SpyreCCLBackend::createSpyreCCLBackend(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<SpyreCCLBackend>(store, rank, size);
}

/***********************************************
 * Wrapper Work for the Sypre Collective Library
 ***********************************************/
SpyreCCLWork::SpyreCCLWork(OpType opType)
    : Work(-1, opType),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))) {}

void SpyreCCLWork::add_schedule(
    std::unique_ptr<spyre_comms::WorkSchedule> schedule) {
  TORCH_CHECK(schedule != nullptr,
              "[SpyreCCL]: spyre-comms returned no work schedule");
  schedule->start();
  work_schedules_.emplace_back(std::move(schedule));
}

void SpyreCCLWork::absorb(SpyreCCLWork& other) {
  for (auto& schedule : other.work_schedules_) {
    work_schedules_.push_back(std::move(schedule));
  }
  for (auto& tensor : other.keep_alive_) {
    keep_alive_.push_back(std::move(tensor));
  }
  for (auto& step : other.on_complete_) {
    on_complete_.push_back(std::move(step));
  }
  other.work_schedules_.clear();
  other.keep_alive_.clear();
  other.on_complete_.clear();
}

void SpyreCCLWork::finish() {
  if (exception_ == nullptr && !on_complete_.empty()) {
    // Take the steps first so that they run exactly once, even if one throws.
    auto steps = std::move(on_complete_);
    on_complete_.clear();
    try {
      for (auto& step : steps) {
        step();
      }
    }
    catch (...) {
      exception_ = std::current_exception();
      if (!future_->completed()) {
        future_->setError(exception_);
      }
      std::rethrow_exception(exception_);
    }
  }
  // The device no longer reads or writes the borrowed buffers.
  keep_alive_.clear();
  if (!future_->completed()) {
    future_->markCompleted(c10::IValue(std::vector<at::Tensor>{}));
  }
}

bool SpyreCCLWork::isCompleted() {
  // Non-blocking: if the future's waiter thread holds the lock, it is still
  // waiting, so the work is not complete yet.
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    return false;
  }
  if (!completed_) {
    try {
      completed_ =
          std::all_of(work_schedules_.begin(), work_schedules_.end(),
                      [](const auto& schedule) { return schedule->query(); });
    }
    catch (...) {
      exception_ = std::current_exception();
      completed_ = true;
      std::rethrow_exception(exception_);
    }
  }
  // Run the completion steps and resolve the future as soon as completion is
  // first detected, so that callers who poll isCompleted() rather than calling
  // wait() still see the result and a resolved future, matching the
  // c10d::Work contract.
  if (completed_) {
    finish();
  }
  return completed_;
}

bool SpyreCCLWork::isSuccess() const {
  return exception_ == nullptr;
}

bool SpyreCCLWork::wait(std::chrono::milliseconds timeout) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!completed_) {
    try {
      if (timeout == kUnsetTimeout || timeout == kNoTimeout) {
        // No deadline: block until the device signals completion.
        for (auto& schedule : work_schedules_) {
          schedule->wait();
        }
      } else {
        // Timed wait: poll query() until completion or deadline, then throw
        // on timeout to match the c10d::Work base-class contract.
        auto deadline = std::chrono::steady_clock::now() + timeout;
        for (auto& schedule : work_schedules_) {
          while (!schedule->query()) {
            if (std::chrono::steady_clock::now() >= deadline) {
              TORCH_CHECK(false, "[SpyreCCL]: Operation timed out!");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        }
      }
    }
    catch (...) {
      exception_ = std::current_exception();
      std::rethrow_exception(exception_);
    }
    completed_ = true;
  }

  // Assemble outputs and resolve the future so any getFuture()
  // chain-continuations are unblocked.
  finish();
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> SpyreCCLWork::getFuture() {
  std::call_once(future_waiter_started_, [this]() {
    // The thread holds a reference, so the work outlives the caller's handle.
    auto self = c10::intrusive_ptr<SpyreCCLWork>::reclaim_copy(this);
    std::thread([self]() {
      try {
        self->wait();
      }
      catch (...) {
        self->future_->setErrorIfNeeded(std::current_exception());
      }
    }).detach();
  });
  return future_;
}

}  // namespace c10d
