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

#include "spyre_sendnn_utils.h"

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "logging.h"
#include "module.h"
#include "types_mapping.h"

using json = nlohmann::json;

namespace spyre {

using spyre_ptr_t = uint64_t;

// TODO(tmhoangt): DTYPE
std::optional<sendnn::GraphLoader> getCachedGraphLoader(
    std::string name, c10::IntArrayRef size, c10::IntArrayRef stride) {
  std::vector<int64_t> size_copy;
  for (int64_t s : size) {
    size_copy.push_back(s);
  }
  std::vector<int64_t> stride_copy;
  for (int64_t s : stride) {
    stride_copy.push_back(s);
  }
  auto key = std::make_tuple(name, size_copy, stride_copy);
  try {
    return GlobalGraphLoaderCache::get().at(key);
  }
  catch (const std::out_of_range& e) {
    return std::nullopt;
  }
}

void storeCachedGraphLoader(std::string name, c10::IntArrayRef size,
                            c10::IntArrayRef stride, sendnn::GraphLoader gl) {
  std::vector<int64_t> size_copy;
  for (int64_t s : size) {
    size_copy.push_back(s);
  }
  std::vector<int64_t> stride_copy;
  for (int64_t s : stride) {
    stride_copy.push_back(s);
  }
  auto key = std::make_tuple(name, size_copy, stride_copy);
  GlobalGraphLoaderCache::get()[key] = gl;
}

/**
 * Create a dummy op for a single tensor in/out for use in allocation
 */
sendnn::GraphBuilder createDummyOp(c10::IntArrayRef sizes) {
  sendnn::GraphBuilder gb;

  std::vector<int64_t> shape;
  for (const int64_t& element : sizes) {
    shape.push_back(element);
  }

  sendnn::TensorShape t_shape(shape);

  sendnn::TensorInfo ti{sendnn::sen_datatype_enum::float16, t_shape,
                        sendnn::TensorLayout::NHWC};
  auto pi = gb.PrimaryInput("Input", ti);
  auto r = gb.Relu("Relu", ti, pi);

  gb.PrimaryOutput("Output", r);
  return gb;
}

sendnn::GraphLoader prepareGraphLoader(sendnn::GraphBuilder* gb) {
  sendnn::Graph graph;
  auto f_s = gb->Finalize(&graph);

  sendnn::GraphLoader gl(GlobalRuntime::get());

  auto l_s = gl.LoadGraph(graph);

  // hardcode to eager-mode for proper allocation when compiling eager graphs
  const char* prev_eager_env_str = std::getenv(EAGER_MODE_ENV);

  setenv(EAGER_MODE_ENV, "1", 1);

  auto c_s = gl.CompileGraph();

  // reset eager_mode
  setenv(EAGER_MODE_ENV,
         prev_eager_env_str != nullptr ? prev_eager_env_str : "0", 1);

  return gl;
}

sendnn::GraphLoader& parseGraphLoader(
    sendnn::GraphLoader& gl,
    std::optional<c10::IntArrayRef> out_shape /*= std::nullopt*/,
    std::optional<c10::IntArrayRef> out_stride /*= std::nullopt*/,
    std::optional<std::vector<c10::IntArrayRef>> inp_shapes /*= std::nullopt*/,
    std::optional<std::vector<c10::IntArrayRef>>
        inp_strides /*= std::nullopt*/) {
  // will be refactored

  int compute_op_idx = 1;
  int sen_host_compute_op_idx = 2;

  auto* attrs_super = dynamic_cast<sendnn::attributes::SenSuperNodeV2*>(
      gl.GetG2s()[0].compute_ops_[compute_op_idx]->Attrs());
  std::string attrs_super_json_str;
  sendnn::SerializeToString(&attrs_super_json_str, *attrs_super);
  json attrs_super_json = json::parse(attrs_super_json_str);

  if (inp_shapes.has_value() && inp_strides.has_value()) {
    compute_op_idx++;

    int n_inps = static_cast<int>(inp_shapes->size());

    for (int i = 0; i < n_inps; i++) {
      auto inp_shape = inp_shapes.value()[n_inps - i - 1];
      auto inp_stride = inp_strides.value()[n_inps - i - 1];

      auto* attrs_hostcompute =
          dynamic_cast<sendnn::attributes::SenHostCompute*>(
              attrs_super->execution_graph_.compute_ops_[i]->Attrs());
      auto payload_json =
          json::parse(attrs_super_json["execution_graph"]["compute_nodes"][i]
                                      ["attributes"]["attr_data"]["payload"]
                                          .get<std::string>());
      // DEBUGINFO("Input Payload json size_ before: ",
      // payload_json["dcsi_"][0]["size_"]);
      DEBUGINFO("Input Payload json stride_src_ before: ",
                payload_json["dcsi_"][0]["stride_src_"]);

      // auto shape_copy = payload_json["dcsi_"][0]["size_"];
      auto stride_copy = payload_json["dcsi_"][0]["stride_src_"];

      int n_dim = static_cast<int>(inp_stride.size());
      for (int j = 0; j < n_dim; j++) {
        if (n_dim == 1 || inp_shape[0] >= inp_shape[1]) {
          // shape_copy[j] = inp_shape[j];
          stride_copy[j] = inp_stride[j];
        } else {
          // shape_copy[n_dim - j - 1] = inp_shape[j];
          stride_copy[n_dim - j - 1] = inp_stride[j];
        }
      }

      // payload_json["dcsi_"][0]["size_"] = shape_copy;
      payload_json["dcsi_"][0]["stride_src_"] = stride_copy;
      DEBUGINFO("Input Payload json size_ after: ",
                payload_json["dcsi_"][0]["size_"])
      DEBUGINFO("Input Payload json stride_src_ after: ",
                payload_json["dcsi_"][0]["stride_src_"]);

      attrs_hostcompute->payload_ = payload_json.dump();
    }

    sen_host_compute_op_idx++;
  }

  if (out_shape.has_value() && out_stride.has_value()) {
    auto* attrs_hostcompute = dynamic_cast<sendnn::attributes::SenHostCompute*>(
        attrs_super->execution_graph_.compute_ops_[sen_host_compute_op_idx]
            ->Attrs());

    auto payload_json =
        json::parse(attrs_super_json["execution_graph"]["compute_nodes"]
                                    [sen_host_compute_op_idx]["attributes"]
                                    ["attr_data"]["payload"]
                                        .get<std::string>());

    // DEBUGINFO("Output Payload json size_ before: ",
    // payload_json["dcsi_"][0]["size_"]);
    DEBUGINFO("Output Payload json stride_dst_ before: ",
              payload_json["dcsi_"][0]["stride_dst_"]);

    // auto shape_copy = payload_json["dcsi_"][0]["size_"];
    auto stride_copy = payload_json["dcsi_"][0]["stride_dst_"];

    int n_dim = static_cast<int>(out_stride->size());
    for (int j = 0; j < n_dim; j++) {
      if (inp_shapes.has_value() && inp_strides.has_value()) {
        if (n_dim == 1 || out_shape->at(0) >= out_shape->at(1)) {
          // shape_copy[j] = out_shape->at(j);
          stride_copy[n_dim - j - 1] = out_stride->at(j);
        } else {
          // shape_copy[n_dim - j - 1] = out_shape->at(j);
          stride_copy[j] = out_stride->at(j);
        }
      } else {
        if (n_dim == 1 || out_shape->at(0) > out_shape->at(1)) {
          stride_copy[j] = out_stride->at(j);
        } else if (out_shape->at(0) < out_shape->at(1)) {
          stride_copy[n_dim - j - 1] = out_stride->at(j);
        }
      }
    }

    // payload_json["dcsi_"][0]["size_"] = shape_copy;
    payload_json["dcsi_"][0]["stride_dst_"] = stride_copy;
    DEBUGINFO("Output Payload json size_ after: ",
              payload_json["dcsi_"][0]["size_"]);
    DEBUGINFO("Output Payload json stride_dst_ after: ",
              payload_json["dcsi_"][0]["stride_dst_"]);

    attrs_hostcompute->payload_ = payload_json.dump();
  }

  auto p_s = gl.ParseGraph();

  return gl;
}

sendnn::ConstTensor createInputTensor(sendnn::GraphLoader& gl, void* data_ptr,
                                      unsigned int input_index /*= 0*/,
                                      uint64_t sn_index /*= 1*/) {
  auto inp_ti = gl.GetInputs(sn_index)[input_index];
  return sendnn::ConstTensor(inp_ti, data_ptr);
}

sendnn::Tensor createOutputTensor(sendnn::GraphLoader& gl, void* data_ptr,
                                  unsigned int output_index /*= 0*/,
                                  uint64_t sn_index /*= 1*/) {
  auto inp_ti = gl.GetOutputs(sn_index)[output_index];
  return sendnn::Tensor(inp_ti, data_ptr);
}

sendnn::TensorInfo getTensorInfo(const at::Tensor& input) {
  std::vector<int64_t> shape;
  if (input.dim() == 0) {
    shape = {1};
  } else {
    for (const int64_t& element : input.sizes()) {
      shape.push_back(element);
    }
  }
  auto str_type = torchScalarToString[input.scalar_type()];
  const auto [sen_dtype_cpu, sen_dtype_dev] = stringToSenDatatypePair(str_type);
  sendnn::TensorShape t_shape(shape);
  sendnn::TensorInfo ti{sen_dtype_cpu, t_shape, sendnn::TensorLayout::NHWC};
  return ti;
}

sendnn::TensorInfo getScalarTensorInfo(const at::Tensor& input) {
  // get the scalar shape, but type is matched with the given tensor
  std::vector<int64_t> shape = {1};
  sendnn::TensorShape t_shape(shape);
  auto str_type = torchScalarToString[input.scalar_type()];
  const auto [sen_dtype_cpu, sen_dtype_dev] = stringToSenDatatypePair(str_type);
  sendnn::TensorInfo ti{sen_dtype_cpu, t_shape, sendnn::TensorLayout::NHWC};
  return ti;
}

}  // namespace spyre
