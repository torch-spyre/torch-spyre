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

#include <libaiupti/aiupti_runtime_cbid.h>

#include <chrono>
#include <cstring>
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

  // Associations and lane metadata belong to this capture, even if the
  // session is reused and batch IDs or thread IDs appear again.
  cpuCorrelationMap_.clear();
  deviceActivityIds_.clear();
  launchActivityIds_.clear();
  threadNames_.clear();
  resourceInfo_.clear();
  activeThreadMap_.clear();
  auto aiuBuffer = api_.activityBuffers();
  if (aiuBuffer) {
    // Records from different queues need not arrive in causal order. Build the
    // batch -> CPU association before processing any device activities.
    api_.processActivities(*aiuBuffer, [this](const Pti_Activity* record) {
      if (record->kind == AIUPTI_ACTIVITY_KIND_THREAD) {
        const auto* a = reinterpret_cast<const AIUpti_ActivityThread*>(record);
        threadNames_[{a->process_id, a->thread_id}] =
            std::string(a->name, strnlen(a->name, sizeof(a->name)));
      } else if (record->kind == AIUPTI_ACTIVITY_KIND_RUNTIME) {
        const auto* a = reinterpret_cast<const AIUpti_ActivityAPI*>(record);
        if (a->calibration_id != 0) return;
        const bool isSubmit =
            a->cbid == AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_TO_HARDWARE;
        const bool isLaunch =
            a->cbid >= AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB &&
            a->cbid <= AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_DMO;
        if (isLaunch) launchActivityIds_.insert(a->correlation_id);
        // Comms schedule IDs have a different namespace; never use them here.
        if ((isSubmit || isLaunch) && a->external_id != 0) {
          cpuCorrelationMap_[a->correlation_id] = a->external_id;
        }
      } else if (record->kind == AIUPTI_ACTIVITY_KIND_CMPT) {
        deviceActivityIds_.insert(
            reinterpret_cast<const AIUpti_ActivityCompute*>(record)
                ->correlation_id);
      } else if (record->kind == AIUPTI_ACTIVITY_KIND_MEMCPY) {
        deviceActivityIds_.insert(
            reinterpret_cast<const AIUpti_ActivityMemcpy*>(record)
                ->correlation_id);
      }
    });
    api_.processActivities(
        *aiuBuffer, std::bind(&AiuptiActivityProfilerSession::handlePtiActivity,
                              this, std::placeholders::_1, &logger));
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

// TODO(mcalman): support multi-AIU
std::unique_ptr<libkineto::DeviceInfo>
AiuptiActivityProfilerSession::getDeviceInfo() {
  int32_t pid = libkineto::processId();
  std::string process_name = libkineto::processName(pid);
  int aiu = 0;
  return std::make_unique<libkineto::DeviceInfo>(
      aiu, aiu + libkineto::kExceedMaxPid, process_name,
      fmt::format("AIU {}", 0));
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

void AiuptiActivityProfilerSession::recordStream(uint32_t device, uint32_t id) {
  if (!hasDeviceResource(device, id)) {
    resourceInfo_.emplace(
        std::make_pair(device, id),
        libkineto::ResourceInfo(device, id, kExceedMaxTid + id,
                                fmt::format("Stream {}", id)));
  }
}

void AiuptiActivityProfilerSession::recordThreadStream(
    uint32_t device, uint32_t id, const std::string& name) {
  if (id == device) {
    return;
  }
  if (!hasDeviceResource(device, id)) {
    const int64_t sort_index = static_cast<int64_t>(device) + id;
    resourceInfo_.emplace(
        std::make_pair(device, id),
        libkineto::ResourceInfo{
            .id = id,
            .sortIndex = sort_index,
            .deviceId = device,
            .name = name.empty() ? fmt::format("Thread {}", id) : name});
  }
}

void AiuptiActivityProfilerSession::recordMemoryStream(uint32_t device,
                                                       uint32_t id,
                                                       std::string name) {
  if (!hasDeviceResource(device, id)) {
    resourceInfo_.emplace(std::make_pair(device, id),
                          libkineto::ResourceInfo(device, id, id, name));
  }
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
