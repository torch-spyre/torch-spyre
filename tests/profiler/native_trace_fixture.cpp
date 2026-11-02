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

// Synthetic AIUPTI records exercise the production session and Kineto exporter
// without initializing a device or collecting hardware activity.
#include <ATen/record_function.h>
#include <libaiupti/aiupti_runtime_cbid.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "AiuptiActivityApi.h"
#include "AiuptiActivityProfiler.h"

namespace {
using libkineto::ActivityType;
using libkineto::AiuptiActivityApi;
using libkineto::AiuptiActivityBuffer;
using libkineto::AiuptiActivityBufferDeque;
using libkineto::AiuptiActivityProfilerSession;
using libkineto::Config;
using libkineto::IActivityProfiler;
using libkineto::IActivityProfilerSession;
using libkineto::Pti_Activity;

class SyntheticApi : public AiuptiActivityApi {
 public:
  std::vector<AIUpti_ActivityAPI> runtime;
  std::vector<AIUpti_ActivityThread> threads;
  std::vector<AIUpti_ActivityCompute> kernels;
  std::vector<AIUpti_ActivityMemcpy> copies;

  std::unique_ptr<AiuptiActivityBufferDeque> activityBuffers() override {
    auto buffers = std::make_unique<AiuptiActivityBufferDeque>();
    // Use real buffers and the production processActivities implementation,
    // including libaiupti's record iterator. Device records precede runtime
    // attribution, and names arrive last across separate buffers.
    appendBuffer(*buffers, kernels);
    appendBuffer(*buffers, copies);
    appendBuffer(*buffers, runtime);
    appendBuffer(*buffers, threads);
    return buffers;
  }

 private:
  template <typename Record>
  static void appendBuffer(AiuptiActivityBufferDeque& buffers,
                           const std::vector<Record>& records) {
    if (records.empty()) return;
    auto buffer =
        std::make_unique<AiuptiActivityBuffer>(records.size() * sizeof(Record));
    std::memcpy(buffer->data(), records.data(), buffer->size());
    auto* data = buffer->data();
    buffers.emplace_back(data, std::move(buffer));
  }
};

SyntheticApi syntheticApi;
thread_local std::vector<uint64_t> externalIds;
std::vector<int32_t> cpuLookups;

// Kineto may create a new wrapper for each capture; deliberately reuse the
// production session underneath to exercise its capture-lifetime state.
class SyntheticSession : public IActivityProfilerSession {
 public:
  explicit SyntheticSession(
      std::shared_ptr<AiuptiActivityProfilerSession> session)
      : session_(std::move(session)) {}

  void start() override {
    syntheticApi.runtime.clear();
    syntheticApi.threads.clear();
    syntheticApi.kernels.clear();
    syntheticApi.copies.clear();
    externalIds.clear();
    cpuLookups.clear();
  }
  void stop() override {}
  std::vector<std::string> errors() override {
    return session_->errors();
  }
  void processTrace(libkineto::ActivityLogger& logger) override {
    session_->processTrace(logger);
  }
  void processTrace(libkineto::ActivityLogger& logger,
                    libkineto::getLinkedActivityCallback lookup, int64_t start,
                    int64_t end) override {
    session_->processTrace(
        logger,
        [lookup](int32_t id) {
          cpuLookups.push_back(id);
          return lookup(id);
        },
        start, end);
  }
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return session_->getDeviceInfo();
  }
  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return session_->getResourceInfos();
  }
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return session_->getTraceBuffer();
  }
  void pushCorrelationId(uint64_t id) override {
    externalIds.push_back(id);
  }
  void popCorrelationId() override {
    externalIds.pop_back();
  }
  void pushUserCorrelationId(uint64_t) override {}
  void popUserCorrelationId() override {}

 private:
  std::shared_ptr<AiuptiActivityProfilerSession> session_;
};

class SyntheticProfiler : public IActivityProfiler {
 public:
  const std::string& name() const override {
    return name_;
  }
  const std::set<ActivityType>& availableActivities() const override {
    return types_;
  }
  std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>&, const Config& config) override {
    if (!session_) {
      session_ = std::make_shared<AiuptiActivityProfilerSession>(
          syntheticApi, config, types_);
    }
    return std::make_unique<SyntheticSession>(session_);
  }
  std::unique_ptr<IActivityProfilerSession> configure(
      int64_t, int64_t, const std::set<ActivityType>& types,
      const Config& config) override {
    return configure(types, config);
  }

 private:
  const std::string name_{"synthetic_spyre"};
  const std::set<ActivityType> types_{ActivityType::CPU_OP};
  std::shared_ptr<AiuptiActivityProfilerSession> session_;
};

uint64_t emitBatch(uint32_t batch, const std::string& mode) {
  const uint64_t external = externalIds.empty() ? 0 : externalIds.back();
  const auto now = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  AIUpti_ActivityAPI submit{};
  submit.activity_kind = AIUPTI_ACTIVITY_KIND_RUNTIME;
  submit.cbid = AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_TO_HARDWARE;
  submit.process_id = getpid();
  submit.thread_id = getpid() + 1000;  // A different scheduler lane.
  submit.start = now;
  submit.end = now + 5000;
  submit.correlation_id = batch;
  submit.external_id = external;
  // One thread intentionally has no metadata, exercising the numeric fallback.
  if (mode == "orphan" || batch == 8) ++submit.thread_id;
  if (syntheticApi.threads.empty()) {
    AIUpti_ActivityThread thread{};
    thread.activity_kind = AIUPTI_ACTIVITY_KIND_THREAD;
    thread.process_id = submit.process_id;
    thread.thread_id = submit.thread_id;
    std::memcpy(thread.name, "test-scheduler", sizeof("test-scheduler"));
    syntheticApi.threads.push_back(thread);
  }
  if (mode == "orphan") submit.external_id = 0;
  if (mode == "wide") submit.external_id |= uint64_t{1} << 40;
  if (mode == "calibration") submit.calibration_id = 1;
  if (mode == "other_process") ++submit.process_id;
  syntheticApi.runtime.push_back(submit);
  if (mode == "normal") {
    auto launch = submit;
    launch.cbid = AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_CMPT;
    launch.start = now + 1000;
    launch.end = now + 2000;
    syntheticApi.runtime.push_back(launch);
    AIUpti_ActivityCompute kernel{};
    kernel.activity_kind = AIUPTI_ACTIVITY_KIND_CMPT;
    kernel.operation_kind = AIUPTI_ACTIVITY_CMPT_KIND_EXEC;
    kernel.correlation_id = batch;
    kernel.stream_id = 3;
    kernel.start = now + 2000;
    kernel.end = now + 4000;
    std::memcpy(kernel.name, "synthetic_kernel", sizeof("synthetic_kernel"));
    syntheticApi.kernels.push_back(kernel);
  }
  return external;
}

void emitMemcpy(uint32_t batch, bool launch, bool calibration = false) {
  const auto now = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  AIUpti_ActivityMemcpy copy{};
  copy.activity_kind = AIUPTI_ACTIVITY_KIND_MEMCPY;
  copy.copy_kind = AIUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
  copy.correlation_id = batch;
  copy.stream_id = 5;
  copy.start = now + 1000;
  copy.end = now + 3000;
  copy.bytes = 64;
  syntheticApi.copies.push_back(copy);
  if (launch) {
    AIUpti_ActivityAPI record{};
    record.activity_kind = AIUPTI_ACTIVITY_KIND_RUNTIME;
    record.cbid = AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMI;
    record.process_id = getpid();
    record.thread_id = getpid() + 1000;
    record.correlation_id = batch;
    record.calibration_id = calibration ? 1 : 0;
    record.start = now;
    record.end = now + 1000;
    syntheticApi.runtime.push_back(record);
  }
}

// A host-side runtime span whose `data` payload carries CBID-specific metadata.
void emitRuntime(uint32_t cbid, uint32_t correlation, uint64_t data) {
  const auto now = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  AIUpti_ActivityAPI record{};
  record.activity_kind = AIUPTI_ACTIVITY_KIND_RUNTIME;
  record.cbid = cbid;
  record.process_id = getpid();
  record.thread_id = getpid() + 1000;
  record.correlation_id = correlation;
  record.start = now;
  record.end = now + 1000;
  record.data = data;
  syntheticApi.runtime.push_back(record);
}

void checkBufferReplay() {
  auto buffers = syntheticApi.activityBuffers();
  std::vector<const Pti_Activity*> first;
  const auto counts = syntheticApi.processActivities(
      *buffers,
      [&first](const Pti_Activity* record) { first.push_back(record); });
  std::vector<const Pti_Activity*> second;
  const auto replayCounts = syntheticApi.processActivities(
      *buffers,
      [&second](const Pti_Activity* record) { second.push_back(record); });
  TORCH_CHECK(
      !first.empty() && first == second && counts == replayCounts,
      "Production buffer iteration must preserve all records on replay");
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("register_profiler", [] {
    libkineto::api().registerProfilerFactory(
        [] { return std::make_unique<SyntheticProfiler>(); });
  });
  module.def("emit_batches", [] {
    // Operator scopes use the external correlation stack, whereas Python's
    // record_function uses the separate user-correlation stack.
    at::RecordFunction origin(at::RecordScope::FUNCTION);
    origin.before("test_cpu_origin");
    uint32_t batch = 7;
    for (const auto* mode : {"normal", "normal", "orphan", "wide",
                             "calibration", "other_process"}) {
      TORCH_CHECK(emitBatch(batch++, mode) != 0,
                  "Expected an active CPU correlation ID");
    }
  });
  module.def("emit_flow_pairs", [] {
    emitMemcpy(13, true);
    emitMemcpy(14, false);
    emitMemcpy(16, true, true);
    emitBatch(15, "normal");
    syntheticApi.runtime.resize(2);  // Keep only the two memcpy launches.
    emitBatch(17, "orphan");
    emitMemcpy(17, false);  // A submit alone cannot start a device flow.
    auto unmatched = syntheticApi.runtime.front();
    unmatched.correlation_id = 18;
    syntheticApi.runtime.push_back(unmatched);
    checkBufferReplay();
  });
  module.def("emit_runtime_metadata", [] {
    // Distinct nonzero payloads catch both CBID and metadata-key drift.
    emitRuntime(AIUPTI_RUNTIME_TRACE_CBID_ISSUE_BARRIER, 40, 3);
    emitRuntime(AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_P2P_RDMA_SEND, 41, 5);
  });
  module.def("emit_capture_window", [](int window) {
    at::RecordFunction origin(at::RecordScope::FUNCTION);
    origin.before("test_cpu_origin");
    const auto external = emitBatch(window == 0 ? 20 : 30, "normal");
    emitMemcpy(window == 0 ? 21 : 31, true);
    if (window != 0) {
      // Reuse a previous launch ID without its device, and a previous device
      // ID without its launch. Neither side may link across captures.
      auto launch = syntheticApi.runtime[1];
      launch.correlation_id = 21;
      launch.external_id = 0;
      syntheticApi.runtime.push_back(launch);
      auto kernel = syntheticApi.kernels.front();
      kernel.correlation_id = 20;
      syntheticApi.kernels.push_back(kernel);
      emitMemcpy(20, false);
      syntheticApi.threads.clear();  // Names must not leak either.
    }
    return external;
  });
  module.def("cpu_lookups", [] { return cpuLookups; });
}
