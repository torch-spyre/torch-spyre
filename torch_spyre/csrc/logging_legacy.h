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

#pragma once

#include "logging_config.h"

namespace torch_spyre {
namespace logging {
namespace legacy {

inline bool is_legacy_debug_enabled() {
  return LoggingConfig::instance().is_enabled("spyre.runtime", LogLevel::DEBUG);
}

namespace detail {

template <typename... Args>
inline void log_variadic(const char* func, Args&&... args) {
  auto stream = Logger("spyre.runtime", LogLevel::DEBUG).debug();
  stream << func << ": ";
  ((stream << args), ...);
}

}  // namespace detail
}  // namespace legacy
}  // namespace logging
}  // namespace torch_spyre

#ifdef DEBUGINFO
#undef DEBUGINFO
#endif

#define DEBUGINFO(...)                                                 \
  do {                                                                 \
    if (torch_spyre::logging::legacy::is_legacy_debug_enabled()) {     \
      torch_spyre::logging::legacy::detail::log_variadic(__func__,     \
                                                         __VA_ARGS__); \
    }                                                                  \
  } while (0)
