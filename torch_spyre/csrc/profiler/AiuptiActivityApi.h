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

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

#include "AiuptiActivityBuffer.h"
#include "AiuptiProfilerMacros.h"

namespace KINETO_NAMESPACE {

using Pti_Activity = AIUpti_Activity;

class AiuptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  AiuptiActivityApi() = default;
  AiuptiActivityApi(const AiuptiActivityApi&) = delete;
  AiuptiActivityApi& operator=(const AiuptiActivityApi&) = delete;

  virtual ~AiuptiActivityApi() {}

  static AiuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableAiuptiActivities(
      const std::set<libkineto::ActivityType>& selected_activities);
  void disablePtiActivities(
      const std::set<libkineto::ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<AiuptiActivityBufferDeque> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      AiuptiActivityBufferDeque&,
      std::function<void(const Pti_Activity*)> handler);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
  int maxAiuBufferCount_{0};
  AiuptiActivityBufferDeque allocatedAiuTraceBuffers_;
  std::unique_ptr<AiuptiActivityBufferDeque> readyAiuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf, size_t validSize,
      std::function<void(const Pti_Activity*)> handler);

  static void bufferRequestedTrampoline(uint8_t** buffer, size_t* size,
                                        size_t* maxNumRecords);
  static void bufferCompletedTrampoline(uint8_t* buffer, size_t size,
                                        size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(uint8_t* buffer, size_t size, size_t validSize);
};

}  // namespace KINETO_NAMESPACE
