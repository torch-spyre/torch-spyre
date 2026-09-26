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
 *
 * Portions derived from libkineto AIU plugin.
 */
#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "AiuptiProfilerMacros.h"

namespace KINETO_NAMESPACE {

// Each hardware stream is split into five lanes in the trace view.
// The composite resource ID is: stream_id * kLaneCount + lane_offset.
// Lane offsets must be contiguous starting at 0.
enum class StreamLane : uint32_t {
  H2D = 0,
  D2H = 1,
  Compute = 2,
  MemMgmt = 3,
  Unknown = 4
};
static constexpr uint32_t kLaneCount = 5;

// Pseudo-PID for host-initiated operations (memset, memory management).
// Placed one slot after the largest possible device PID in the sort order.
static constexpr int64_t kHostComputePid = libkineto::kExceedMaxPid + 1;

inline uint32_t streamLaneResourceId(uint32_t stream_id, StreamLane lane) {
  return stream_id * kLaneCount + static_cast<uint32_t>(lane);
}

constexpr std::string_view streamLaneName(StreamLane lane) {
  switch (lane) {
    case StreamLane::H2D:
      return "H2D";
    case StreamLane::D2H:
      return "D2H";
    case StreamLane::Compute:
      return "Compute";
    case StreamLane::MemMgmt:
      return "Memory Management";
    case StreamLane::Unknown:
    default:
      return "Unknown";
  }
}

inline std::string streamLaneLabel(uint32_t stream_id, StreamLane lane) {
  if (lane == StreamLane::MemMgmt) {
    return "Memory Management";
  }

  return fmt::format("Stream {} / {}", stream_id, streamLaneName(lane));
}

class AiuptiActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  AiuptiActivityProfilerSession() = delete;
  AiuptiActivityProfilerSession(
      AiuptiActivityApi& api, const libkineto::Config& config,
      const std::set<libkineto::ActivityType>& activity_types);
  AiuptiActivityProfilerSession(const AiuptiActivityProfilerSession&) = delete;
  AiuptiActivityProfilerSession& operator=(
      const AiuptiActivityProfilerSession&) = delete;

  ~AiuptiActivityProfilerSession();

  void start() override;
  void stop() override;
  std::vector<std::string> errors() override {
    return errors_;
  };
  void processTrace(libkineto::ActivityLogger& logger) override;
  void processTrace(libkineto::ActivityLogger& logger,
                    libkineto::getLinkedActivityCallback get_linked_activity,
                    int64_t captureWindowStartTime,
                    int64_t captureWindowEndTime) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

 private:
  void checkTimestampOrder(const libkineto::ITraceActivity* act1);
  void removeCorrelatedPtiActivities(const libkineto::ITraceActivity* act1);
  bool outOfRange(const libkineto::ITraceActivity& act);
  int64_t getMappedQueueId(uint64_t sycl_queue_id);
  const libkineto::ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);
  void handleRuntimeActivity(const AIUpti_ActivityAPI* activity,
                             libkineto::ActivityLogger* logger);
  void handleKernelActivity(const AIUpti_ActivityCompute* activity,
                            libkineto::ActivityLogger* logger);
  void handleMemcpyActivity(const AIUpti_ActivityMemcpy* activity,
                            libkineto::ActivityLogger* logger);
  void handleMemsetActivity(const AIUpti_ActivityMemset* activity,
                            libkineto::ActivityLogger* logger);
  void handleMemoryActivity(const AIUpti_ActivityMemory* activity,
                            libkineto::ActivityLogger* logger);
  void handlePtiActivity(const AIUpti_Activity* record,
                         libkineto::ActivityLogger* logger);

  template <class memory_activity_type>
  uint32_t getResourceId(memory_activity_type* activity);

  static uint32_t iterationCount_;
  static std::vector<std::array<unsigned char, 16>> deviceUUIDs_;
  static std::vector<std::string> correlateRuntimeOps_;

  std::set<uint32_t> observedDeviceIds_;

  int64_t captureWindowStartTime_{0};
  int64_t captureWindowEndTime_{0};
  int64_t profilerStartTs_{0};
  int64_t profilerEndTs_{0};
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;
  std::unordered_map<int64_t, const libkineto::ITraceActivity*>
      correlatedPtiActivities_;
  std::map<std::pair<int64_t, int64_t>, std::vector<int64_t>> activeThreadMap_;
  std::vector<std::string> errors_;

  libkineto::getLinkedActivityCallback cpuActivity_;

  AiuptiActivityApi& api_;
  libkineto::CpuTraceBuffer traceBuffer_;
  std::vector<uint64_t> sycl_queue_pool_;
  std::unique_ptr<const libkineto::Config> config_{nullptr};
  const std::set<libkineto::ActivityType>& activity_types_;

  std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo> resourceInfo_;
  bool hasDeviceResource(uint32_t device, uint32_t id);
  void ensureResource(uint32_t device, uint32_t stream_id, StreamLane lane);
  void recordStream(uint32_t device, uint32_t stream_id, StreamLane lane);
  void recordMemoryStream(uint32_t device, uint32_t stream_id, StreamLane lane);

  int64_t totalAllocatedBytes_{0};
};

class AIUActivityProfiler : public libkineto::IActivityProfiler {
 public:
  AIUActivityProfiler() = default;
  AIUActivityProfiler(const AIUActivityProfiler&) = delete;
  AIUActivityProfiler& operator=(const AIUActivityProfiler&) = delete;

  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms, int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"__aiu_profiler__"};
  int64_t AsyncProfileStartTime_{0};
  int64_t AsyncProfileEndTime_{0};
};

}  // namespace KINETO_NAMESPACE
