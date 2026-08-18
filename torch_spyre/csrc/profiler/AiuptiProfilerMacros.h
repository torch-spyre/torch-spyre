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

#include <libaiupti/aiupti_activity.h>
#include <libkineto.h>
#include <output_base.h>
#include <time_since_epoch.h>

#include <string>

namespace KINETO_NAMESPACE {

#define AIUPTI_CALL(returnCode)                                               \
  {                                                                           \
    if (returnCode != AIUPTI_SUCCESS) {                                       \
      std::string funcMsg(__func__);                                          \
      std::string codeMsg = std::to_string(returnCode);                       \
      std::string HeadMsg("Kineto Profiler on AIU got error from function "); \
      std::string Msg(". The error code is ");                                \
      throw std::runtime_error(HeadMsg + funcMsg + Msg + codeMsg);            \
    }                                                                         \
  }

class AiuptiActivityApi;
using DeviceIndex_t = int8_t;

}  // namespace KINETO_NAMESPACE
