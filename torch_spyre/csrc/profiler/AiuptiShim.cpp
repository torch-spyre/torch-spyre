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

// Thin resolver shim over libaiupti.
//
// This translation unit is compiled into a standalone shared library
// (torch_spyre/_aiupti_shim.so) that links libaiupti as a DT_NEEDED
// dependency. torch_spyre._C deliberately does NOT link libaiupti: merely
// loading it into the process arms the flex runtime's per-op telemetry, which
// adds a fixed per-dispatch cost to every aten op and dominates the forward
// pass under eager execution. The profiler dlopen's this shim on the first
// trace request (pulling libaiupti in at that point) and resolves the aiupti
// entry points through the single `extern "C"` function below.
//
// Taking the address of each aiupti function here forces the compiler to emit
// correctly-mangled references, so we never hardcode C++ mangled names in
// dlsym calls.

#include <libaiupti/aiupti_activity.h>

#include <cstring>

extern "C" void* ts_aiupti_resolve(const char* name) {
  if (std::strcmp(name, "aiuptiActivityRegisterCallbacks") == 0)
    return reinterpret_cast<void*>(&aiuptiActivityRegisterCallbacks);
  if (std::strcmp(name, "aiuptiActivityEnable") == 0)
    return reinterpret_cast<void*>(&aiuptiActivityEnable);
  if (std::strcmp(name, "aiuptiActivityDisable") == 0)
    return reinterpret_cast<void*>(&aiuptiActivityDisable);
  if (std::strcmp(name, "aiuptiActivityGetNextRecord") == 0)
    return reinterpret_cast<void*>(&aiuptiActivityGetNextRecord);
  if (std::strcmp(name, "aiuptiActivityGetNumDroppedRecords") == 0)
    return reinterpret_cast<void*>(&aiuptiActivityGetNumDroppedRecords);
  if (std::strcmp(name, "aiuptiFlushAllActivities") == 0)
    return reinterpret_cast<void*>(&aiuptiFlushAllActivities);
  return nullptr;
}
