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
#include "AiuptiLibrary.h"

#ifdef HAS_AIUPTI

#include <dlfcn.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace KINETO_NAMESPACE {

namespace {

using ResolveFn = void* (*)(const char*);

// Locate _aiupti_shim.so next to the _C extension that this code is linked
// into. dladdr on a symbol from this translation unit yields the path of _C,
// whose directory also contains the shim.
std::string shimPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&shimPath), &info) == 0 ||
      info.dli_fname == nullptr) {
    throw std::runtime_error(
        "Failed to locate the torch_spyre extension directory for AIU "
        "profiling (dladdr failed)");
  }
  std::string path(info.dli_fname);
  auto slash = path.find_last_of('/');
  std::string dir = (slash == std::string::npos) ? "." : path.substr(0, slash);
  return dir + "/_aiupti_shim.so";
}

void* resolve(ResolveFn resolveFn, const char* name) {
  void* sym = resolveFn(name);
  if (sym == nullptr) {
    throw std::runtime_error(
        std::string("_aiupti_shim could not resolve libaiupti symbol '") +
        name + "'");
  }
  return sym;
}

}  // namespace

const AiuptiLibrary& AiuptiLibrary::singleton() {
  static AiuptiLibrary lib;
  static std::once_flag once;
  std::call_once(once, [&]() {
    std::string path = shimPath();
    // Loading the shim pulls libaiupti in as its DT_NEEDED dependency; this is
    // the point at which flex per-op telemetry is armed, which is why it is
    // deferred until a profiler session actually starts.
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      throw std::runtime_error(
          std::string("Failed to load ") + path +
          " for AIU profiling: " + dlerror());
    }
    auto resolveFn =
        reinterpret_cast<ResolveFn>(dlsym(handle, "ts_aiupti_resolve"));
    if (resolveFn == nullptr) {
      throw std::runtime_error(
          std::string("_aiupti_shim is missing ts_aiupti_resolve: ") +
          dlerror());
    }
    lib.activityRegisterCallbacks =
        reinterpret_cast<decltype(lib.activityRegisterCallbacks)>(
            resolve(resolveFn, "aiuptiActivityRegisterCallbacks"));
    lib.activityEnable = reinterpret_cast<decltype(lib.activityEnable)>(
        resolve(resolveFn, "aiuptiActivityEnable"));
    lib.activityDisable = reinterpret_cast<decltype(lib.activityDisable)>(
        resolve(resolveFn, "aiuptiActivityDisable"));
    lib.activityGetNextRecord =
        reinterpret_cast<decltype(lib.activityGetNextRecord)>(
            resolve(resolveFn, "aiuptiActivityGetNextRecord"));
    lib.activityGetNumDroppedRecords =
        reinterpret_cast<decltype(lib.activityGetNumDroppedRecords)>(
            resolve(resolveFn, "aiuptiActivityGetNumDroppedRecords"));
    lib.flushAllActivities =
        reinterpret_cast<decltype(lib.flushAllActivities)>(
            resolve(resolveFn, "aiuptiFlushAllActivities"));
  });
  return lib;
}

}  // namespace KINETO_NAMESPACE

#endif  // HAS_AIUPTI
