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

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/ScalarType.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <spyrecode-host-functions/sendataconvert/sen_data_convert.h>
#include <util/sendefs/sendefs.h>

#include <cstdint>
#include <cstdlib>     // std::getenv
#include <filesystem>  // NOLINT(build/c++17)
#include <flex/flex.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "job_plan.h"

#ifdef USE_SPYRE_CCL
#include <pybind11/chrono.h>

#include "distributed/spyre_ccl.hpp"
#endif

#include "logging.h"
#include "logging_bindings.h"
#include "logging_config.h"
#include "prepare_kernel.h"
#include "spyre_allocator.h"
#include "spyre_device_enum.h"
#include "spyre_error.h"
#include "spyre_generator_impl.h"
#include "spyre_guard.h"
#include "spyre_kernel.h"
#include "spyre_mem.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "spyre_views.h"
#include "types_mapping.h"

namespace fs = std::filesystem;

namespace spyre {

static constexpr int32_t kSpyreTensorLayoutPickleVersion = 3;

std::atomic<bool> g_downcast_warn_enabled{true};

bool get_downcast_warn_enabled() {
  return g_downcast_warn_enabled.load(std::memory_order_relaxed);
}

void set_downcast_warn_enabled(bool enabled) {
  g_downcast_warn_enabled.store(enabled, std::memory_order_relaxed);
}

// SPYRE_HAZARD_TRACKER latch: when on, torch-spyre skips emitting the static
// edge-3/4 event steps and instead splits the correction triple across
// S_prep/S_dev while flex's per-region hazard tracker inserts the cross-stream
// RAW/WAR events dynamically at enqueue. Default OFF => byte-identical to the
// static-edge overlap path. Read once from the env in init_from_env and applied
// to the flex RuntimeContext in _startRuntime.
std::atomic<bool> g_hazard_tracker_enabled{false};  // default OFF

bool get_hazard_tracker_enabled() {
  return g_hazard_tracker_enabled.load(std::memory_order_relaxed);
}

void set_hazard_tracker_enabled(bool enabled) {
  g_hazard_tracker_enabled.store(enabled, std::memory_order_relaxed);
}

// Optional: initialize from env at module init
static void init_from_env() {
  if (const char* v = std::getenv(SPYRE_DOWNCAST_ENV)) {
    // Accept 0/1, true/false, on/off
    std::string s(v);
    for (auto& c : s) c = std::tolower(c);
    bool enable = !(s == "0" || s == "false" || s == "off");
    g_downcast_warn_enabled.store(enable, std::memory_order_relaxed);
  }
  // SPYRE_HAZARD_TRACKER is a correctness gate: strict "1" semantics (matching
  // SPYRE_DISABLE_HC_DMA_OVERLAP), NOT the permissive downcast parser above.
  if (const char* v = std::getenv("SPYRE_HAZARD_TRACKER")) {
    g_hazard_tracker_enabled.store(std::string(v) == "1",
                                   std::memory_order_relaxed);
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
  } else if (const char* lr = std::getenv("LOCAL_RANK")) {
    logical_device_id = std::atoi(lr);
    // Match the current (c10) device to the rank so unqualified stream/pool
    // lookups don't fall back to spyre:0.
    SpyreGuardImpl::tls_idx = static_cast<c10::DeviceIndex>(logical_device_id);
  }

  const int num_devices = getVisibleDeviceCount();
  TORCH_CHECK(logical_device_id < num_devices,
              "Device index out of bounds. logical_device_id=",
              logical_device_id, ", number of visible devices=", num_devices);

  std::shared_ptr<flex::RuntimeContext> runtime;
  auto s = flex::initializeRuntime(&runtime, logical_device_id);
  init_from_env();
  if (runtime) {
    GlobalRuntime::set(runtime);
    // Apply the SPYRE_HAZARD_TRACKER latch (read in init_from_env above) to the
    // flex runtime: this enables flex's per-region hazard tracker iff the flag
    // is on. It is the ONLY coupled enable of flex's tracker; the python
    // set_hazard_tracker_enabled hook flips only torch-spyre's plan-shaping +
    // routing global and does NOT re-drive flex. setHazardTrackerEnabled only
    // stores an atomic in flex, so it cannot fail.
    runtime->setHazardTrackerEnabled(get_hazard_tracker_enabled());
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
  // Register PrivateUse1 hooks — tells PyTorch the device exists.
  // Loading _C.so does NOT trigger device initialization;
  // start_runtime() must be called explicitly (via _lazy_init()).
  {
    struct SpyreHooksArgs : public at::PrivateUse1HooksArgs {};
    struct SpyreHooksInterface : public at::PrivateUse1HooksInterface {
      SpyreHooksInterface() = default;
      explicit SpyreHooksInterface(SpyreHooksArgs) {}
      ~SpyreHooksInterface() override = default;
      bool hasPrimaryContext(c10::DeviceIndex) const override {
        return true;
      }
      bool isAvailable() const override {
        return true;
      }
      const at::Generator& getDefaultGenerator(
          c10::DeviceIndex device) const override {
        return spyre::detail::getDefaultSpyreGenerator(device);
      }
      at::Generator getNewGenerator(c10::DeviceIndex device) const override {
        return spyre::detail::createSpyreGenerator(device);
      }
    };
    static auto* hooks = new SpyreHooksInterface();
    at::RegisterPrivateUse1HooksInterface(hooks);
  }

  m.doc() = "Spyre C++ bindings";
  m.def("start_runtime", &spyre::startRuntime);
  m.def("free_runtime", &spyre::freeRuntime);
  m.def("device_count", &spyre::getVisibleDeviceCount);
  m.def("encode_constant", &spyre::encodeConstant);

  // Initialize logging bindings
  torch_spyre::logging::init_logging_bindings(m);

  py::enum_<spyre::ElementArrangement>(m, "ElementArrangement")
      .value("STANDARD", spyre::ElementArrangement::STANDARD)
      .value("DL16_TO_FP32", spyre::ElementArrangement::DL16_TO_FP32)
      .value("QFP8CH", spyre::ElementArrangement::QFP8CH)
      .value("EXX2", spyre::ElementArrangement::EXX2)
      .value("FP32_TO_DL16", spyre::ElementArrangement::FP32_TO_DL16)
      .value("QFP8WT", spyre::ElementArrangement::QFP8WT);

  py::class_<spyre::SpyreTensorLayout> dci_cls(m, "SpyreTensorLayout");

  dci_cls.def_readonly("device_size", &spyre::SpyreTensorLayout::device_size)
      .def_readonly("stride_map", &spyre::SpyreTensorLayout::stride_map)
      .def_readonly("device_dtype", &spyre::SpyreTensorLayout::device_dtype)
      .def_readonly("element_arrangement",
                    &spyre::SpyreTensorLayout::element_arrangement)
      .def("with_element_arrangement",
           &spyre::SpyreTensorLayout::with_element_arrangement,
           py::arg("element_arrangement"))
      .def("__str__",
           [](const spyre::SpyreTensorLayout& c) { return c.toString(); })
      .def("__repr__",
           [](const spyre::SpyreTensorLayout& c) { return c.toString(); })
      .def("elems_per_stick", &spyre::SpyreTensorLayout::elems_per_stick)
      .def(py::self == py::self)
      .def("__hash__",
           [](const spyre::SpyreTensorLayout& c) {
             return std::hash<spyre::SpyreTensorLayout>{}(c);
           })
      .def(py::init<std::vector<int64_t>, c10::ScalarType>(),
           py::arg("host_size"), py::arg("dtype"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, c10::ScalarType,
                    std::vector<int32_t>, spyre::ElementArrangement>(),
           py::arg("host_size"), py::arg("host_strides"), py::arg("dtype"),
           py::arg("dim_order"),
           py::arg("element_arrangement") = spyre::ElementArrangement::STANDARD)
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, DataFormats,
                    spyre::ElementArrangement>(),
           py::arg("device_size"), py::arg("stride_map"),
           py::arg("device_dtype"),
           py::arg("element_arrangement") = spyre::ElementArrangement::STANDARD)
      .def(py::pickle(
          [](const spyre::SpyreTensorLayout& p) {  // __getstate__
            // Return a tuple that fully encodes the state of the object
            // If the pickle format changes, then update
            // kSpyreTensorLayoutPickleVersion but keep the tuple as the
            // returned object and the first element to be the
            // kSpyreTensorLayoutPickleVersion
            return py::make_tuple(spyre::kSpyreTensorLayoutPickleVersion,
                                  p.device_size, p.stride_map, p.device_dtype,
                                  p.element_arrangement);
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
            } else if (version == 3) {
              // Version 3: (version, device_size, stride_map, device_dtype,
              // element_arrangement)
              if (t.size() != 5) {
                throw py::value_error(
                    "Invalid SpyreTensorLayout pickle v3: wrong tuple size");
              }
              return spyre::SpyreTensorLayout(
                  t[1].cast<std::vector<int64_t>>(),
                  t[2].cast<std::vector<int64_t>>(), t[3].cast<DataFormats>(),
                  t[4].cast<spyre::ElementArrangement>());
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
           [](const DataFormats& df) { return spyre::elems_per_stick(df); });

  m.def("get_spyre_tensor_layout", &spyre::get_spyre_tensor_layout);
  m.def("set_spyre_tensor_layout", &spyre::set_spyre_tensor_layout);
  m.def("get_downcast_warning", &spyre::get_downcast_warn_enabled,
        "Return whether downcast warnings are enabled.");
  m.def("set_downcast_warning", &spyre::set_downcast_warn_enabled,
        "Enable/disable downcast warnings for this process.");
  m.def("get_hazard_tracker_enabled", &spyre::get_hazard_tracker_enabled,
        "Whether the flex per-region hazard tracker is enabled.");
  m.def("set_hazard_tracker_enabled", &spyre::set_hazard_tracker_enabled,
        py::arg("enabled"),
        "Enable/disable the flex hazard-tracker path (TEST/tuning hook; the "
        "production value is latched from SPYRE_HAZARD_TRACKER at runtime "
        "start). NOTE: this flips only torch-spyre's plan-shaping + routing "
        "global; it does NOT re-drive flex's RuntimeContext, which was set "
        "once at _startRuntime.");
  m.def("get_elem_in_stick", &spyre::get_elem_in_stick);
  m.def("get_device_dtype", &spyre::get_device_dtype);

  // RNG functions
  m.def("manual_seed", &spyre::manual_seed, py::arg("seed"),
        py::arg("device") = -1);
  m.def("manual_seed_all", &spyre::manual_seed_all, py::arg("seed"));
  m.def("get_rng_state", &spyre::get_rng_state, py::arg("device") = -1);
  m.def("set_rng_state", &spyre::set_rng_state, py::arg("new_state"),
        py::arg("device") = -1);
  m.def("initial_seed", &spyre::initial_seed, py::arg("device") = -1);
  m.def("_get_default_generator", &spyre::detail::getDefaultSpyreGenerator,
        py::arg("device") = -1);

  // Memory copy function
  m.def("copy_tensor", &spyre::spyre_copy_from,
        "Copy tensor between host and device using DMA", py::arg("self"),
        py::arg("dst"), py::arg("non_blocking") = false);

  // Device-side fill using FillDMA (no host buffer or H2D copy)
  m.def("fill_tensor", &spyre::spyre_fill_tensor,
        "Fill a spyre tensor with a scalar value using device-side FillDMA",
        py::arg("self"), py::arg("value"));

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

  m.def("host_compute_stream", &spyre::getHostComputeStream, py::arg("device"),
        "Get a host compute stream for a device (round-robin)");

  m.def("host_compute_stream_by_id", &spyre::getHostComputeStreamById,
        py::arg("id"), py::arg("device"),
        "Get a specific host compute stream by stream ID");

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
      .def("__repr__", [](const spyre::SpyreStream& stream) {
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

#ifdef USE_SPYRE_CCL
  // Spyre CCL distributed backend
  m.def("createSpyreCCLBackend", &c10d::SpyreCCLBackend::createSpyreCCLBackend,
        "Create the Spyre Collective Library Backend object");
#endif

  py::class_<spyre::JobPlan>(m, "JobPlan")
      .def(
          "num_steps",
          [](const spyre::JobPlan& plan) { return plan.steps.size(); },
          "Get the number of steps in the JobPlan")
      .def(
          "job_allocation_size",
          [](const spyre::JobPlan& plan) {
            return plan.job_allocation.at(0).total_size();
          },
          "Get the size of the job allocation")
      .def(
          "get_step_type",
          [](const spyre::JobPlan& plan, size_t idx) {
            TORCH_CHECK(idx < plan.steps.size(), "Step index out of range");
            // Single source of truth for step-type identification (also used by
            // the P2-14 ordering validator). Recognizes the two-stream event
            // steps (SignalForward/WaitForward/SignalBack/WaitBack) that the
            // old dynamic_cast chain reported as "Unknown".
            return spyre::stepKindName(spyre::classifyStep(*plan.steps[idx]));
          },
          py::arg("idx"), "Get the type of step at the given index")
      .def(
          "get_step_pipeline_barrier",
          [](const spyre::JobPlan& plan, size_t idx) {
            TORCH_CHECK(idx < plan.steps.size(), "Step index out of range");
            return plan.steps[idx]->getPipelineBarrier();
          },
          py::arg("idx"),
          "Get the pipeline_barrier flag for the step at the given index")
      .def(
          "get_step_stream_role",
          [](const spyre::JobPlan& plan, size_t idx) {
            TORCH_CHECK(idx < plan.steps.size(), "Step index out of range");
            return plan.steps[idx]->role() == spyre::StreamRole::Prep ? "Prep"
                                                                      : "Dev";
          },
          py::arg("idx"),
          "Get the stream role (Prep/Dev) for the step at the given index")
      .def(
          "get_step_region_id",
          [](const spyre::JobPlan& plan, size_t idx) -> uint64_t {
            TORCH_CHECK(idx < plan.steps.size(), "Step index out of range");
            const auto* step = plan.steps[idx].get();
            if (const auto* sb =
                    dynamic_cast<const spyre::JobPlanStepEventSignalBack*>(
                        step)) {
              return sb->regionId();
            }
            if (const auto* wb =
                    dynamic_cast<const spyre::JobPlanStepEventWaitBack*>(
                        step)) {
              return wb->regionId();
            }
            TORCH_CHECK(false, "Step at index ", idx,
                        " is not a back-event step (SignalBack/WaitBack); has "
                        "no region_id");
          },
          py::arg("idx"),
          "Get the edge-4 region_id key for a SignalBack/WaitBack step")
      .def("__repr__", [](const spyre::JobPlan& plan) {
        return "<JobPlan steps=" + std::to_string(plan.steps.size()) +
               " job_allocation_size=" +
               std::to_string(plan.job_allocation.at(0).total_size()) +
               " expected_inputs=" +
               std::to_string(plan.expected_input_shapes.size()) +
               " pinned_buffers=" + std::to_string(plan.pinned_buffers.size()) +
               ">";
      });
  m.def("prepare_kernel", &spyre::prepareKernel, py::arg("spyrecode_dir"),
        py::arg("stream") = nullptr,
        "Prepare a kernel from a SpyreCode directory and return a JobPlan.\n\n"
        "Args:\n"
        "    spyrecode_dir (str): Path to the SpyreCode directory\n"
        "    stream (SpyreStream, optional): Stream to use for initialization "
        "transfers.\n"
        "        If None, uses the current stream. Defaults to None.\n\n"
        "Returns:\n"
        "    Prepared JobPlan ready for execution");
  // Bind the current-stream overload (resolves the current stream internally).
  m.def("launch_jobplan",
        static_cast<void (*)(const spyre::JobPlan&,
                             const std::vector<at::Tensor>&)>(
            &spyre::launchJobPlan),
        py::arg("job_plan"), py::arg("args"),
        "Launch a prepared JobPlan with the given tensor arguments.\n\n"
        "Args:\n"
        "    job_plan: The JobPlan to execute\n"
        "    args: Sequence of input/output tensors");

  // ── Two-stream overlap: step-ordering validator + edge-4 instrumentation ──

  // Direct binding of the pure P2-14 ordering checker so the
  // Placement-Invariant rejection can be tested without constructing real steps
  // (a real HostCompute needs a deeptools::Hcm + pinned buffers). Takes
  // parallel lists of StepKind names ("HostCompute"/"H2D"/... per stepKindName)
  // and StreamRole names
  // ("Prep"/"Dev"), returns "" when valid or a human-readable error otherwise.
  m.def(
      "check_job_plan_step_ordering",
      [](const std::vector<std::string>& kind_names,
         const std::vector<std::string>& role_names) {
        TORCH_CHECK(kind_names.size() == role_names.size(),
                    "kind_names/role_names length mismatch: ",
                    kind_names.size(), " vs ", role_names.size());
        std::vector<spyre::StepKind> kinds;
        std::vector<spyre::StreamRole> roles;
        kinds.reserve(kind_names.size());
        roles.reserve(role_names.size());
        for (const auto& n : kind_names) {
          kinds.push_back(spyre::stepKindFromName(n));
        }
        for (const auto& n : role_names) {
          roles.push_back(spyre::streamRoleFromName(n));
        }
        return spyre::checkJobPlanStepOrdering(kinds, roles);
      },
      py::arg("kind_names"), py::arg("role_names"),
      "Validate a projected two-stream step ordering; '' if valid else the "
      "error message.");

  m.def(
      "edge4_slot_hits", []() { return spyre::SpyreStream::edge4SlotHits(); },
      "Number of WaitBack rolling-slot HITS (found a published back-event).");
  m.def(
      "edge4_slot_skips", []() { return spyre::SpyreStream::edge4SlotSkips(); },
      "Number of WaitBack rolling-slot SKIPS (empty slot, no-op'd).");
  m.def(
      "reset_edge4_slot_stats",
      []() { spyre::SpyreStream::resetEdge4SlotStats(); },
      "Reset the edge-4 rolling-slot hit/skip counters to zero.");

  // ── Test-only edge-4 rolling-slot hooks (#2 cross-kernel WAR gate) ──
  // These drive the REAL SpyreStream::setEdge4Slot/getEdge4Slot map that
  // SignalBack::construct / WaitBack::construct use, keyed by a region_id the
  // test obtains from a REAL prepared plan (get_step_region_id). A device-free
  // test proves: after producer A publishes a back-event under region_id R, a
  // consumer B looking up the SAME derived R gets the SAME Event (pointer
  // identity) with edge4_slot_hits going 0->1 -- the cross-kernel WAR "hit"
  // that the shipped e2e never asserts -- and a disjoint region misses
  // (nullptr, edge4_slot_skips increments). NOT compiled out, but purely
  // additive and unused by the production launch path.
  m.def(
      "_test_edge4_signal_store",
      [](uint64_t region_id) -> uintptr_t {
        auto event = flex::createEvent();
        spyre::SpyreStream::setEdge4Slot(region_id, event);
        return reinterpret_cast<uintptr_t>(event.get());
      },
      py::arg("region_id"),
      "TEST-ONLY: publish a fresh back-event under region_id (mirrors "
      "SignalBack::construct's setEdge4Slot); returns the Event pointer.");
  m.def(
      "_test_edge4_wait_probe",
      [](uint64_t region_id) -> uintptr_t {
        auto event = spyre::SpyreStream::getEdge4Slot(region_id);
        return event ? reinterpret_cast<uintptr_t>(event.get())
                     : static_cast<uintptr_t>(0);
      },
      py::arg("region_id"),
      "TEST-ONLY: look up the back-event for region_id (mirrors "
      "WaitBack::construct's getEdge4Slot); returns the Event pointer or 0. "
      "Increments edge4_slot_hits on a hit, edge4_slot_skips on a miss.");
  m.def(
      "_test_clear_edge4_slots",
      []() { spyre::SpyreStream::clearEdge4Slots(); },
      "TEST-ONLY: drop all published edge-4 back-events (test isolation; the "
      "mock device address yields a deterministic region_id across tests).");

  // ── Test-only validator projection hook (#7a) ──
  // Projects a REAL prepared plan's steps through the SAME path validate()
  // uses -- classifyStep(*step) + step->role() -- in a caller-specified index
  // order, then runs the ordering checker. Exercises the JobPlanStep->(kind,
  // role) projection end-to-end, where a real edit-6 wiring bug would live (the
  // check_job_plan_step_ordering binding takes name lists and bypasses this
  // projection). Passing a permuted `order` injects a Placement-Invariant
  // violation over the real step objects. Returns '' if valid else the error.
  m.def(
      "_test_project_and_check_ordering",
      [](const spyre::JobPlan& plan, const std::vector<size_t>& order) {
        std::vector<spyre::StepKind> kinds;
        std::vector<spyre::StreamRole> roles;
        kinds.reserve(order.size());
        roles.reserve(order.size());
        for (size_t idx : order) {
          TORCH_CHECK(idx < plan.steps.size(),
                      "_test_project_and_check_ordering: step index ", idx,
                      " out of range (", plan.steps.size(), " steps)");
          const auto& step = plan.steps[idx];
          kinds.push_back(spyre::classifyStep(*step));
          roles.push_back(step->role());
        }
        return spyre::checkJobPlanStepOrdering(kinds, roles);
      },
      py::arg("plan"), py::arg("order"),
      "TEST-ONLY: project REAL plan steps (classifyStep + role(), exactly as "
      "validate() does) in the given index order, then run the ordering "
      "checker. Returns '' if valid else the error message.");

  // Allocator statistics functions
  m.def(
      "_spyre_get_allocator_stats",
      [](c10::DeviceIndex device) {
        auto& allocator = spyre::SpyreAllocator::instance();
        auto stats = allocator.getDeviceStats(device);
        py::dict result;
        result["allocated_bytes.all.current"] =
            stats
                .allocated_bytes[static_cast<size_t>(
                    c10::CachingAllocator::StatType::AGGREGATE)]
                .current;
        result["allocation.all.current"] =
            stats
                .allocation[static_cast<size_t>(
                    c10::CachingAllocator::StatType::AGGREGATE)]
                .current;
        return result;
      },
      py::arg("device"), "Get allocator statistics for a device");

  m.def(
      "_spyre_reset_accumulated_stats",
      [](c10::DeviceIndex device) {
        auto& allocator = spyre::SpyreAllocator::instance();
        allocator.resetAccumulatedStats(device);
      },
      py::arg("device"), "Reset accumulated allocator statistics");

  m.def(
      "_spyre_reset_peak_stats",
      [](c10::DeviceIndex device) {
        auto& allocator = spyre::SpyreAllocator::instance();
        allocator.resetPeakStats(device);
      },
      py::arg("device"), "Reset peak allocator statistics");

  // ── Typed stream error API ───────────────────────────────────────────────

  py::enum_<spyre::SpyreStreamError>(m, "SpyreStreamError")
      .value("Success", spyre::SpyreStreamError::Success)
      .value("Shutdown", spyre::SpyreStreamError::Shutdown);

  py::enum_<spyre::SpyreDeviceState>(m, "SpyreDeviceState")
      .value("Ok", spyre::SpyreDeviceState::Ok)
      .value("NotInitialized", spyre::SpyreDeviceState::NotInitialized)
      .value("StreamError", spyre::SpyreDeviceState::StreamError);

  m.def(
      "stream_get_error",
      [](const spyre::SpyreStream& stream) {
        return spyre::SpyreStreamGetError(stream);
      },
      py::arg("stream"),
      "Return the SpyreStreamError state of a single stream.");

  m.def(
      "stream_get_error_string",
      [](spyre::SpyreStreamError err) -> std::string {
        return spyre::SpyreStreamGetErrorString(err);
      },
      py::arg("error"),
      "Return a human-readable string for a SpyreStreamError value.");

  m.def(
      "get_device_state", []() { return spyre::SpyreGetDeviceState(); },
      "Return the SpyreDeviceState aggregate for the process. "
      "Returns NotInitialized when the runtime has not been created yet; "
      "callers should treat this as 'proceed / no error'.");
}
