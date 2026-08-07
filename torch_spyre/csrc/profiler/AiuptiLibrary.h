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
#pragma once

#ifdef HAS_AIUPTI

#include <libaiupti/aiupti_activity.h>

namespace KINETO_NAMESPACE {

// Lazily-resolved entry points into libaiupti.so.
//
// libaiupti is deliberately NOT linked as a DT_NEEDED dependency of
// torch_spyre._C. Merely loading it into the process arms the flex runtime's
// per-op telemetry path, which adds a fixed per-dispatch cost to every aten
// op. Under eager execution (CompilationMode.NONE) each op is its own tiny
// graph, so that cost is paid ~once per op per token and dominates the forward
// pass. By resolving these symbols via dlopen/dlsym on the first profiler
// session rather than at import, libaiupti stays unloaded — and telemetry
// dormant — until a trace is actually requested.
struct AiuptiLibrary {
  decltype(&aiuptiActivityRegisterCallbacks) activityRegisterCallbacks =
      nullptr;
  decltype(&aiuptiActivityEnable) activityEnable = nullptr;
  decltype(&aiuptiActivityDisable) activityDisable = nullptr;
  decltype(&aiuptiActivityGetNextRecord) activityGetNextRecord = nullptr;
  decltype(&aiuptiActivityGetNumDroppedRecords) activityGetNumDroppedRecords =
      nullptr;
  decltype(&aiuptiFlushAllActivities) flushAllActivities = nullptr;

  // Resolves and caches the library handle and symbols on first call. Throws
  // std::runtime_error if libaiupti.so cannot be loaded or a symbol is
  // missing. The handle is intentionally never dlclose'd: once a profiling
  // session loads libaiupti, it stays resident for the process lifetime.
  static const AiuptiLibrary& singleton();
};

}  // namespace KINETO_NAMESPACE

#endif  // HAS_AIUPTI
