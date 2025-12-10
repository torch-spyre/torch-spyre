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

#include "module.h"

#include <c10/core/ScalarType.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <util/sen_data_convert.h>
#include <util/sendefs.h>

#include <cstdlib>  // std::getenv
#include <dee_internal/dee_graph_converter.hpp>
#include <flex/flex_factory.hpp>
#include <memory>
#include <sendnn/graph.hpp>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/graph/graph_deserializer.hpp>
#include <sendnn/graph/graph_utils.hpp>
#include <sendnn/runtime/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
#include <vector>

#include "logging.h"
#include "spyre_mem.h"
#include "spyre_sendnn_utils.h"
#include "types_mapping.h"

namespace spyre {

std::atomic<bool> g_downcast_warn_enabled{true};

bool get_downcast_warn_enabled() {
  return g_downcast_warn_enabled.load(std::memory_order_relaxed);
}

void set_downcast_warn_enabled(bool enabled) {
  g_downcast_warn_enabled.store(enabled, std::memory_order_relaxed);
}

// Optional: initialize from env at module init
static void init_from_env() {
  if (const char *v = std::getenv(SPYRE_DOWNCAST_ENV)) {
    // Accept 0/1, true/false, on/off
    std::string s(v);
    for (auto &c : s) c = std::tolower(c);
    bool enable = !(s == "0" || s == "false" || s == "off");
    g_downcast_warn_enabled.store(enable, std::memory_order_relaxed);
  }
}

void _startRuntime() {
  DEBUGINFO("starting runtime");
  // TODO(tmhoangt): move sendnn::RuntimeInterface to flex to isolate from
  // sendnn
  std::shared_ptr<sendnn::RuntimeInterface> base_runtime;
  auto s = flex::CreateRuntimeInterface(&base_runtime);
  std::shared_ptr<flex::Runtime> runtime =
      std::dynamic_pointer_cast<flex::Runtime>(base_runtime);
  init_from_env();
  if (runtime) {
    GlobalRuntime::set(runtime);
    DEBUGINFO(s);
    DEBUGINFO("runtime started");
  } else {
    DEBUGINFO("runtime FAILED TO START.");
  }
}
void startRuntime() {
  static std::once_flag flag;
  std::call_once(flag, _startRuntime);
}

void freeRuntime() {
  GlobalRuntime::reset();
}
void launchKernel(std::string g2_path, std::vector<at::Tensor> args) {
  // Get global runtime from eager
  auto gl = sendnn::GraphLoader(GlobalRuntime::get());

  // Load compiled kernel
  auto g2 = sendnn::Graph();
  sendnn::Deserialize(&g2, g2_path);

  for (auto &super_node : g2.compute_ops_) {
    if (super_node->Name() != "DeviceInit" &&
        super_node->Name() != "PrepareModel") {
      auto *sn_attrs = dynamic_cast<sendnn::attributes::SenSuperNodeV2 *>(
          super_node->Attrs());
      auto &exec_graph = sn_attrs->execution_graph_;
      for (auto &node : exec_graph.compute_ops_) {
        auto *dev_attrs =
            dynamic_cast<sendnn::attributes::SenFusedDeviceNode *>(
                node->Attrs());
        auto &sub_graph = dev_attrs->sub_graph_;
        auto compute_node = sub_graph.compute_ops_.front();
        auto edge_count = 0;

        for (auto &arg : args) {
          if (&args.back() != &arg) {
            auto tensor = sendnn::Tensor(getTensorInfo(arg));
            exec_graph.AddInput(
                new sendnn::Node(sendnn::opcodes::PrimaryInput, {tensor}));
            sub_graph.AddInput(
                new sendnn::Node(sendnn::opcodes::PrimaryInput, {tensor}));
            exec_graph.NewEdge(edge_count, node, 0,
                               exec_graph.input_ops_[edge_count]);
            sub_graph.NewEdge(edge_count++, compute_node, 0,
                              sub_graph.input_ops_[edge_count]);
          } else {
            auto tensor = sendnn::Tensor(getTensorInfo(arg));
            exec_graph.NewOutput(sendnn::opcodes::PrimaryOutput, {});
            sub_graph.NewOutput(sendnn::opcodes::PrimaryOutput, {});

            auto *exec_edge =
                exec_graph.NewEdge(0, exec_graph.output_ops_.front(), 0, node);
            exec_edge->tensor_ = tensor;
            auto *sub_edge = sub_graph.NewEdge(0, sub_graph.output_ops_.front(),
                                               0, compute_node);
            sub_edge->tensor_ = tensor;
          }
        }
      }
    }
  }

  // Load/parse patched G2 graph
  auto status = gl.LoadGraph(g2, false);
  if (!status.IsOk()) throw std::runtime_error(status.Message());

  status = gl.CompileGraph();
  if (!status.IsOk()) throw std::runtime_error(status.Message());

  status = gl.ParseGraph();
  if (!status.IsOk()) throw std::runtime_error(status.Message());

  // Create sendnn tensors
  std::vector<sendnn::ConstTensor> sen_inputs;
  std::vector<sendnn::Tensor> sen_outputs;
  for (size_t i = 0; i < args.size() - 1; ++i) {
    auto arg = args[i];
    at::Tensor tmp_0;
    if (arg.dim() == 0) {
      tmp_0 = (at::ones({1}, arg.dtype()) * arg).to(arg.device());
      auto tensor = createInputTensor(gl, tmp_0.storage().data_ptr().get(), i,
                                      (args.size() >= 3) ? 2 : 1);
      tensor.SetSpyreData(static_cast<SharedOwnerCtx *>(
                              tmp_0.storage().data_ptr().get_context())
                              ->owner);
      sen_inputs.push_back(tensor);
    } else {
      auto tensor = createInputTensor(gl, arg.storage().data_ptr().get(), i,
                                      (args.size() >= 3) ? 2 : 1);
      tensor.SetSpyreData(
          static_cast<SharedOwnerCtx *>(arg.storage().data_ptr().get_context())
              ->owner);
      sen_inputs.push_back(tensor);
    }
  }
  auto tensor = createOutputTensor(gl, args.back().storage().data_ptr().get(),
                                   0, (args.size() >= 3) ? 2 : 1);
  tensor.SetSpyreData(static_cast<SharedOwnerCtx *>(
                          args.back().storage().data_ptr().get_context())
                          ->owner);
  sen_outputs.push_back(tensor);

  // Execute device init
  if (args.size() >= 3) {
    status = gl.Predict(sendnn::Outputs(), {sen_inputs[1]}, 1);
    if (!status.IsOk()) throw std::runtime_error(status.Message());

    status = gl.Compute(sen_outputs, sen_inputs, 2);
    if (!status.IsOk()) throw std::runtime_error(status.Message());
  } else {
    status = gl.Predict(sendnn::Outputs(), sendnn::Inputs(), 0);
    if (!status.IsOk()) throw std::runtime_error(status.Message());

    status = gl.Compute(sen_outputs, sen_inputs, 1);
    if (!status.IsOk()) throw std::runtime_error(status.Message());
  }

  return;
}
std::string getSenDataFormat(c10::ScalarType torch_dtype) {
  const auto [dtype_cpu, dtype_dev] =
      stringToDTDataFormatPair(torchScalarToString[torch_dtype]);
  return EnumsConversion::dataFormatsToString(dtype_dev);
}

uint32_t encodeConstant(float torch_const, const std::string &data_format) {
  uint32_t sen_const;
  DataFormats df;
  df = FromString<DataFormats>(data_format);

  if (df == DataFormats::IEEE_FP32) {
    sen_const =
        deeptools::BinaryConvert<uint32_t>(static_cast<float>(torch_const));
  } else {
    sen_const = deeptools::FloatToFp16Bin(torch_const);
  }
  return sen_const;
}
void convertArtifacts(std::string artifacts_path) {
  dee::PBD pbd;
  sendnn::Graph g2;

  setenv("DEEPRT_EXPORT_DIR", artifacts_path.c_str(), 1);
  senbfcc::GlobalTracedSettings::Get().UpdateValue("DEEPRT_EXPORT_DIR",
                                                   artifacts_path);

  setenv("SENDNN_SERIALIZER_FORMAT", "CBOR", 1);

  // Convert compiled artifacts to sendnn g2 graph
  pbd.FromGraph(&g2);

  // Serialize g2 graph
  sendnn::Serialize(g2, artifacts_path + "/g2");

  return;
}
}  // namespace spyre

PYBIND11_MODULE(_C, m) {
  m.doc() = "Spyre C++ bindings";
  m.def("start_runtime", &spyre::startRuntime);
  m.def("free_runtime", &spyre::freeRuntime);
  m.def("launch_kernel", &spyre::launchKernel);
  m.def("encode_constant", &spyre::encodeConstant);
  m.def("get_sen_data_format", &spyre::getSenDataFormat);
  m.def("convert_artifacts", &spyre::convertArtifacts);
  m.def("spyre_empty_with_layout", &spyre::spyre_empty_with_layout);

  py::class_<spyre::SpyreTensorLayout> dci_cls(m, "SpyreTensorLayout");

  py::enum_<spyre::SpyreTensorLayout::StickFormat>(m, "StickFormat")
      .value("Dense", spyre::SpyreTensorLayout::StickFormat::Dense)
      .value("Sparse", spyre::SpyreTensorLayout::StickFormat::Sparse)
      .value("SparseMulti", spyre::SpyreTensorLayout::StickFormat::SparseMulti);

  dci_cls.def_readwrite("device_size", &spyre::SpyreTensorLayout::device_size)
      .def_readwrite("dim_map", &spyre::SpyreTensorLayout::dim_map)
      .def_readwrite("num_stick_dims",
                     &spyre::SpyreTensorLayout::num_stick_dims)
      .def_readwrite("format", &spyre::SpyreTensorLayout::format)
      .def("__str__",
           [](const spyre::SpyreTensorLayout &c) { return c.toString(); })
      .def("__repr__",
           [](const spyre::SpyreTensorLayout &c) { return c.toString(); })
      .def("device_strides", &spyre::SpyreTensorLayout::device_strides)
      .def(py::self == py::self)
      .def(py::init<std::vector<int64_t>, c10::ScalarType>(),
           py::arg("host_size"), py::arg("dtype"))
      .def(py::init<std::vector<int64_t>, c10::ScalarType, std::vector<int32_t>,
                    spyre::SpyreTensorLayout::StickFormat>(),
           py::arg("host_size"), py::arg("dtype"), py::arg("dim_order"),
           py::arg("format") = spyre::SpyreTensorLayout::StickFormat::Dense)
      .def(py::init<std::vector<int64_t>, std::vector<int32_t>, int32_t,
                    spyre::SpyreTensorLayout::StickFormat>(),
           py::arg("device_size"), py::arg("dim_map"),
           py::arg("num_stick_dims"), py::arg("format"));

  m.def("get_spyre_tensor_layout", &spyre::get_spyre_tensor_layout);
  m.def("get_downcast_warning", &spyre::get_downcast_warn_enabled,
        "Return whether downcast warnings are enabled.");
  m.def("set_downcast_warning", &spyre::set_downcast_warn_enabled,
        "Enable/disable downcast warnings for this process.");
}
