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
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <util/sen_data_convert.h>
#include <util/sendefs.h>

#include <cstdlib>  // std::getenv
#include <flex/compiler_interface/dee_graph_converter.hpp>
#include <flex/runtime_graph/flex_factory.hpp>
#include <memory>
#include <sendnn/graph.hpp>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/graph/graph_deserializer.hpp>
#include <sendnn/graph/graph_utils.hpp>
#include <sendnn/interface/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
#include <vector>

#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_device_enum.h"
#include "spyre_guard.h"
#include "spyre_mem.h"
#include "spyre_sendnn_utils.h"
#include "spyre_stream.h"
#include "spyre_views.h"
#include "types_mapping.h"

namespace spyre {

static constexpr int32_t kSpyreTensorLayoutPickleVersion = 2;

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
  // Determine logical device index with priority:
  //   1. tls_idx (non-zero) — set via explicit set_device() call
  //   2. LOCAL_RANK env var — set by torchrun per process
  //   3. 0 — single-device / non-torchrun default
  int logical_device_id = 0;
  int tls_idx = static_cast<int>(SpyreGuardImpl::tls_idx);
  if (tls_idx != 0) {
    logical_device_id = tls_idx;
  } else if (const char *lr = std::getenv("LOCAL_RANK")) {
    logical_device_id = std::atoi(lr);
  }

  const int num_devices = getVisibleDeviceCount();
  TORCH_CHECK(logical_device_id < num_devices,
              "Device index out of bounds. logical_device_id=",
              logical_device_id, ", number of visible devices=", num_devices);

  std::shared_ptr<Runtime> runtime;
  auto s = flex::initializeRuntime(&runtime, logical_device_id);
  init_from_env();
  if (runtime) {
    GlobalRuntime::set(runtime);
    DEBUGINFO(s);
    DEBUGINFO("runtime started with logical_device_id ", logical_device_id);
  } else {
    DEBUGINFO("runtime FAILED TO START.");
    throw std::runtime_error("Failed to initialize Spyre runtime. ");
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
            sub_graph.NewEdge(edge_count, compute_node, 0,
                              sub_graph.input_ops_[edge_count]);
            edge_count++;
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
      auto tensor =
          createInputTensor(gl, tmp_0.storage().data_ptr().get(), i, 1);
      tensor.SetSpyreData(static_cast<SharedOwnerCtx *>(
                              tmp_0.storage().data_ptr().get_context())
                              ->owner);
      sen_inputs.push_back(tensor);
    } else {
      auto tensor = createInputTensor(gl, arg.storage().data_ptr().get(), i, 1);
      tensor.SetSpyreData(
          static_cast<SharedOwnerCtx *>(arg.storage().data_ptr().get_context())
              ->owner);
      sen_inputs.push_back(tensor);
    }
  }
  auto tensor =
      createOutputTensor(gl, args.back().storage().data_ptr().get(), 0, 1);
  tensor.SetSpyreData(static_cast<SharedOwnerCtx *>(
                          args.back().storage().data_ptr().get_context())
                          ->owner);
  sen_outputs.push_back(tensor);

  // Execute device init
  status = gl.Predict(sendnn::Outputs(), sendnn::Inputs(), 0);
  if (!status.IsOk()) throw std::runtime_error(status.Message());

  status = gl.Compute(sen_outputs, sen_inputs, 1);
  if (!status.IsOk()) throw std::runtime_error(status.Message());

  return;
}

uint32_t encodeConstant(float torch_const, DataFormats df) {
  uint32_t sen_const;

  if (df == DataFormats::IEEE_FP32) {
    sen_const =
        deeptools::BinaryConvert<uint32_t>(static_cast<float>(torch_const));
  } else {
    sen_const = deeptools::FloatToFp16Bin(torch_const);
  }
  return sen_const;
}

int64_t get_elem_in_stick(c10::ScalarType torch_dtype) {
  auto str_type = torchScalarToString[torch_dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  return elems_per_stick(sen_dtype_dev);
}

DataFormats get_device_dtype(c10::ScalarType torch_dtype) {
  auto str_type = torchScalarToString[torch_dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  return sen_dtype_dev;
}

bool is_supported_dtype(c10::ScalarType dtype) {
  // TODO(kmehant,yoheiueda): Replace this heuristic with a reliable method to
  // determine supported dtypes. Using elems_per_stick can miss certain
  // unsupported dtypes. See #950
  DataFormats sen_dtype_dev = get_device_dtype(dtype);
  return sen_dtype_dev != DataFormats::INVALID &&
         elems_per_stick(sen_dtype_dev) > 0;
}
int device_count() {
  return getVisibleDeviceCount();
}

}  // namespace spyre

namespace py = pybind11;
PYBIND11_MODULE(_C, m) {
  m.doc() = "Spyre C++ bindings";
  m.def("start_runtime", &spyre::startRuntime);
  m.def("free_runtime", &spyre::freeRuntime);
  m.def("launch_kernel", &spyre::launchKernel);
  m.def("encode_constant", &spyre::encodeConstant);
  m.def("convert_artifacts", &dee::convertArtifacts);

  py::class_<spyre::SpyreTensorLayout> dci_cls(m, "SpyreTensorLayout");

  dci_cls.def_readonly("device_size", &spyre::SpyreTensorLayout::device_size)
      .def_readonly("stride_map", &spyre::SpyreTensorLayout::stride_map)
      .def_readonly("device_dtype", &spyre::SpyreTensorLayout::device_dtype)
      .def("__str__",
           [](const spyre::SpyreTensorLayout &c) { return c.toString(); })
      .def("__repr__",
           [](const spyre::SpyreTensorLayout &c) { return c.toString(); })
      .def("elems_per_stick", &spyre::SpyreTensorLayout::elems_per_stick)
      .def(py::self == py::self)
      .def(py::init<std::vector<int64_t>, c10::ScalarType>(),
           py::arg("host_size"), py::arg("dtype"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, c10::ScalarType,
                    std::vector<int32_t>>(),
           py::arg("host_size"), py::arg("host_strides"), py::arg("dtype"),
           py::arg("dim_order"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, DataFormats>(),
           py::arg("device_size"), py::arg("stride_map"),
           py::arg("device_dtype"))
      .def(py::pickle(
          [](const spyre::SpyreTensorLayout &p) {  // __getstate__
            // Return a tuple that fully encodes the state of the object
            // If the pickle format changes, then update
            // kSpyreTensorLayoutPickleVersion but keep the tuple as the
            // returned object and the first element to be the
            // kSpyreTensorLayoutPickleVersion
            return py::make_tuple(spyre::kSpyreTensorLayoutPickleVersion,
                                  p.device_size, p.stride_map, p.device_dtype);
          },
          [](py::tuple t) {  // __setstate__
            int32_t version = t[0].cast<int32_t>();
            if (version == 1) {
              // Version 1 had: (version, device_size, dim_map, stride_map,
              // device_dtype) — discard dim_map
              if (t.size() != 5) {
                throw py::value_error(
                    "Invalid SpyreTensorLayout pickle v1: wrong tuple size");
              }
              return spyre::SpyreTensorLayout(t[1].cast<std::vector<int64_t>>(),
                                              t[3].cast<std::vector<int64_t>>(),
                                              t[4].cast<DataFormats>());
            } else if (version == 2) {
              // Version 2: (version, device_size, stride_map, device_dtype)
              if (t.size() != 4) {
                throw py::value_error(
                    "Invalid SpyreTensorLayout pickle v2: wrong tuple size");
              }
              return spyre::SpyreTensorLayout(t[1].cast<std::vector<int64_t>>(),
                                              t[2].cast<std::vector<int64_t>>(),
                                              t[3].cast<DataFormats>());
            } else {
              throw py::value_error(
                  "Unsupported SpyreTensorLayout pickle version: " +
                  std::to_string(version));
            }
          }));

  m.def("spyre_empty_with_layout", &spyre::spyre_empty_with_layout);
  m.def("empty_with_layout", &spyre::py_empty_with_layout);
  m.def("as_strided_with_layout", &spyre::as_strided_with_layout);
  m.def("reinterpret_tensor", &spyre::reinterpret_tensor);
  m.def("reinterpret_tensor_with_layout",
        &spyre::reinterpret_tensor_with_layout);

  py::enum_<DataFormats>(m, "DataFormats")
      .value("SEN169_FP16", DataFormats::SEN169_FP16)
      .value("IEEE_FP32", DataFormats::IEEE_FP32)
      .value("INVALID", DataFormats::INVALID)
      .value("SEN143_FP8", DataFormats::SEN143_FP8)
      .value("SEN152_FP8", DataFormats::SEN152_FP8)
      .value("SEN153_FP9", DataFormats::SEN153_FP9)
      .value("SENINT2", DataFormats::SENINT2)
      .value("SENINT4", DataFormats::SENINT4)
      .value("SENINT8", DataFormats::SENINT8)
      .value("SENINT16", DataFormats::SENINT16)
      .value("SENINT24", DataFormats::SENINT24)
      .value("IEEE_INT64", DataFormats::IEEE_INT64)
      .value("IEEE_INT32", DataFormats::IEEE_INT32)
      .value("SENUINT32", DataFormats::SENUINT32)
      .value("SENUINT2", DataFormats::SENUINT2)
      .value("IEEE_FP16", DataFormats::IEEE_FP16)
      .value("BOOL", DataFormats::BOOL)
      .value("BFLOAT16", DataFormats::BFLOAT16)
      .value("SEN18F_FP24", DataFormats::SEN18F_FP24)
      .def("elems_per_stick",
           [](const DataFormats &df) { return spyre::elems_per_stick(df); });

  m.def("get_spyre_tensor_layout", &spyre::get_spyre_tensor_layout);
  m.def("set_spyre_tensor_layout", &spyre::set_spyre_tensor_layout);
  m.def("get_downcast_warning", &spyre::get_downcast_warn_enabled,
        "Return whether downcast warnings are enabled.");
  m.def("set_downcast_warning", &spyre::set_downcast_warn_enabled,
        "Enable/disable downcast warnings for this process.");
  m.def("get_elem_in_stick", &spyre::get_elem_in_stick);
  m.def("get_device_dtype", &spyre::get_device_dtype);

  // Memory copy function
  m.def("copy_tensor", &spyre::spyre_copy_from,
        "Copy tensor between host and device using DMA", py::arg("self"),
        py::arg("dst"), py::arg("non_blocking") = false);

  // Stream management functions
  m.def("get_stream_from_pool", &spyre::getStreamFromPool, py::arg("device"),
        py::arg("priority") = 0,
        "Get a stream from the pool with specified device and priority");

  m.def("current_stream", &spyre::getCurrentStream, py::arg("device"),
        "Get the current stream for a device");

  m.def("set_current_stream", &spyre::setCurrentStream, py::arg("stream"),
        "Set the current stream and return the previous one");

  m.def("default_stream", &spyre::getDefaultStream, py::arg("device"),
        "Get the default stream for a device");

  m.def("synchronize", &spyre::synchronizeDevice,
        py::arg("device") = py::none(), "Synchronize a device or all devices");

  // Expose SpyreStream class to Python
  py::class_<spyre::SpyreStream>(m, "_SpyreStreamBase")
      .def("synchronize", &spyre::SpyreStream::synchronize,
           "Wait for all operations on this stream to complete")
      .def("query", &spyre::SpyreStream::query,
           "Check if all operations on this stream have completed")
      .def("device", &spyre::SpyreStream::device,
           "Get the device associated with this stream")
      .def("id", &spyre::SpyreStream::id, "Get the stream ID")
      .def("priority", &spyre::SpyreStream::priority, "Get the stream priority")
      .def("__repr__", [](const spyre::SpyreStream &stream) {
        return "<torch_spyre.Stream device=" +
               std::to_string(stream.device().index()) +
               " id=" + std::to_string(stream.id()) + ">";
      });
  m.def("set_device", [](int idx) {
    int count = spyre::device_count();
    TORCH_CHECK(idx >= 0 && idx < count, "Device index ", idx,
                " out of range [0, ", count, ")");
    c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->setDevice(c10::Device(c10::DeviceType::PrivateUse1,
                                static_cast<c10::DeviceIndex>(idx)));
  });
  m.def("current_device", []() {
    return c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->getDevice()
        .index();
  });
  m.def("device_count", &spyre::device_count);
}
