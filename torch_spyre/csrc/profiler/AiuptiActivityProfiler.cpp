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
#include "AiuptiActivityProfiler.h"

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "AiuptiActivityApi.h"

namespace KINETO_NAMESPACE {

uint32_t AiuptiActivityProfilerSession::iterationCount_ = 0;
std::vector<std::array<unsigned char, 16>>
    AiuptiActivityProfilerSession::deviceUUIDs_ = {};
std::vector<std::string> AiuptiActivityProfilerSession::correlateRuntimeOps_ = {
    "aiuLaunchControlBlocks"};

// =========== Session Constructor ============= //
AiuptiActivityProfilerSession::AiuptiActivityProfilerSession(
    AiuptiActivityApi& api, const libkineto::Config& config,
    const std::set<libkineto::ActivityType>& activity_types)
    : api_(api), config_(config.clone()), activity_types_(activity_types) {
  api_.setMaxBufferSize(config_->activitiesMaxGpuBufferSize());
}

AiuptiActivityProfilerSession::~AiuptiActivityProfilerSession() {
  api_.clearActivities();
}

// =========== Session Public Methods ============= //
void AiuptiActivityProfilerSession::start() {
  profilerStartTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
  api_.enableAiuptiActivities(activity_types_);
}

void AiuptiActivityProfilerSession::stop() {
  profilerEndTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
  api_.clearActivities();
  api_.disablePtiActivities(activity_types_);
}

void AiuptiActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger) {
  traceBuffer_.span = libkineto::TraceSpan(profilerStartTs_, profilerEndTs_,
                                           "__aiu_profiler__");
  traceBuffer_.span.iteration = iterationCount_++;

  auto aiuBuffer = api_.activityBuffers();
  if (aiuBuffer) {
    api_.processActivities(
        *aiuBuffer, std::bind(&AiuptiActivityProfilerSession::handlePtiActivity,
                              this, std::placeholders::_1, &logger));
  }

  // Emit one DeviceInfo (PID) per observed AIU device. Use
  // device_id + kExceedMaxPid as both the pid and the sort index so that:
  //   - AIU rows appear below CPU rows (kExceedMaxPid pushes them down)
  //   - device 0 pid != CPU process 0
  for (uint32_t device_id : observedDeviceIds_) {
    const int64_t aiu_pid =
        static_cast<int64_t>(device_id) + libkineto::kExceedMaxPid;
    logger.handleDeviceInfo(
        libkineto::DeviceInfo(aiu_pid, aiu_pid,
                              fmt::format("AIU {}", device_id),
                              fmt::format("AIU {}", device_id)),
        profilerStartTs_);
  }

  // Emit the "Host Compute" PID only if at least one memset or memory-release
  // activity was recorded (i.e. a resource on kHostComputePid was registered).
  const bool has_host_resources =
      resourceInfo_.lower_bound({kHostComputePid, 0}) !=
      resourceInfo_.lower_bound({kHostComputePid + 1, 0});
  if (has_host_resources) {
    logger.handleDeviceInfo(
        libkineto::DeviceInfo(kHostComputePid, kHostComputePid, "Host Compute",
                              "Host Compute"),
        profilerStartTs_);
  }
}

void AiuptiActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback get_linked_activity,
    int64_t captureWindowStartTime, int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime;
  captureWindowEndTime_ = captureWindowEndTime;
  cpuActivity_ = get_linked_activity;
  processTrace(logger);
}

std::unique_ptr<libkineto::DeviceInfo>
AiuptiActivityProfilerSession::getDeviceInfo() {
  // All DeviceInfo entries are emitted directly in processTrace().
  // Return a non-null sentinel so that CuptiActivityProfiler::finalizeTrace
  // sets use_default_device_info=false and skips the generic "GPU 0–15"
  // fallback that would overwrite our "AIU {N}" / "Host Compute" labels.
  if (observedDeviceIds_.empty()) {
    return nullptr;
  }
  uint32_t first_id = *observedDeviceIds_.begin();
  const int64_t aiu_pid =
      static_cast<int64_t>(first_id) + libkineto::kExceedMaxPid;
  return std::make_unique<libkineto::DeviceInfo>(
      aiu_pid, aiu_pid, fmt::format("AIU {}", first_id),
      fmt::format("AIU {}", first_id));
}

std::vector<libkineto::ResourceInfo>
AiuptiActivityProfilerSession::getResourceInfos() {
  std::vector<libkineto::ResourceInfo> resourceInfos;
  for (const auto& entries : resourceInfo_) {
    resourceInfos.push_back(entries.second);
  }
  return resourceInfos;
}

bool AiuptiActivityProfilerSession::hasDeviceResource(uint32_t device,
                                                      uint32_t id) {
  return resourceInfo_.find({device, id}) != resourceInfo_.end();
}

void AiuptiActivityProfilerSession::ensureResource(uint32_t device,
                                                   uint32_t stream_id,
                                                   StreamLane lane) {
  const uint32_t resource = streamLaneResourceId(stream_id, lane);
  const std::string label = streamLaneLabel(stream_id, lane);
  if (!hasDeviceResource(device, resource)) {
    resourceInfo_.emplace(
        std::make_pair(device, resource),
        libkineto::ResourceInfo(resource, resource, device, label));
  }
}

void AiuptiActivityProfilerSession::recordStream(uint32_t device,
                                                 uint32_t stream_id,
                                                 StreamLane lane) {
  ensureResource(device, stream_id, lane);
}

void AiuptiActivityProfilerSession::recordMemoryStream(uint32_t device,
                                                       uint32_t stream_id,
                                                       StreamLane lane) {
  ensureResource(device, stream_id, lane);
}

std::unique_ptr<libkineto::CpuTraceBuffer>
AiuptiActivityProfilerSession::getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

void AiuptiActivityProfilerSession::pushCorrelationId(uint64_t id) {
  api_.pushCorrelationID(id, AiuptiActivityApi::CorrelationFlowType::Default);
}

void AiuptiActivityProfilerSession::popCorrelationId() {
  api_.popCorrelationID(AiuptiActivityApi::CorrelationFlowType::Default);
}

void AiuptiActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  api_.pushCorrelationID(id, AiuptiActivityApi::CorrelationFlowType::User);
}

void AiuptiActivityProfilerSession::popUserCorrelationId() {
  api_.popCorrelationID(AiuptiActivityApi::CorrelationFlowType::User);
}

// =========== ActivityProfiler Public Methods ============= //
const std::set<libkineto::ActivityType> kAiuTypes{
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
};

const std::string& AIUActivityProfiler::name() const {
  return name_;
}

const std::set<libkineto::ActivityType>&
AIUActivityProfiler::availableActivities() const {
  throw std::runtime_error(
      "The availableActivities is legacy method and should not be called by "
      "kineto");
  return kAiuTypes;
}

std::unique_ptr<libkineto::IActivityProfilerSession>
AIUActivityProfiler::configure(
    const std::set<libkineto::ActivityType>& activity_types,
    const libkineto::Config& config) {
  return std::make_unique<AiuptiActivityProfilerSession>(
      AiuptiActivityApi::singleton(), config, activity_types);
}

std::unique_ptr<libkineto::IActivityProfilerSession>
AIUActivityProfiler::configure(
    int64_t ts_ms, int64_t duration_ms,
    const std::set<libkineto::ActivityType>& activity_types,
    const libkineto::Config& config) {
  AsyncProfileStartTime_ = ts_ms;
  AsyncProfileEndTime_ = ts_ms + duration_ms;
  return configure(activity_types, config);
}
}  // namespace KINETO_NAMESPACE
