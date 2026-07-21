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

#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace torch_spyre {
namespace logging {

// Transparent hash/equal for heterogeneous unordered_map lookup with
// string_view keys, avoiding temporary std::string allocations.
struct StringHash {
  using is_transparent = void;
  size_t operator()(std::string_view sv) const {
    return std::hash<std::string_view>{}(sv);
  }
};
struct StringEqual {
  using is_transparent = void;
  bool operator()(std::string_view a, std::string_view b) const {
    return a == b;
  }
};

// Log levels matching Python's logging module
enum class LogLevel : int {
  NOTSET = 0,
  DEBUG = 10,
  INFO = 20,
  WARNING = 30,
  ERROR = 40,
  CRITICAL = 50
};

// Convert log level to string
const char* log_level_to_string(LogLevel level);

// Convert string to log level
LogLevel string_to_log_level(const std::string& level_str);

/**
 * Unified logging configuration manager for C++ components.
 *
 * This class provides:
 * - Thread-safe access to logging configuration via shared_mutex
 * - Integration with Python logging_config module
 * - Hierarchical component lookup
 * - Generation-validated thread_local cache for lock-free steady-state reads
 */
class LoggingConfig {
 public:
  // Get singleton instance
  static LoggingConfig& instance();

  // Initialize from Python configuration
  // Called once during module initialization via pybind11
  void initialize_from_python(
      const std::vector<std::pair<std::string, int>>& config);

  // Get log level for a component.
  // Uses a generation-validated thread_local cache so steady-state reads
  // are lock-free. Falls back to shared_lock + hierarchy walk on cache miss.
  LogLevel get_log_level(const std::string& component) const;

  // Check if logging is enabled for a component at a given level.
  // Fast path hits thread_local cache (no lock, no hash).
  inline bool is_enabled(const std::string& component, LogLevel level) const {
    return get_log_level(component) <= level;
  }

  // Set log level programmatically (for testing)
  void set_log_level(const std::string& component, LogLevel level);

  // Get all configured components
  std::vector<std::string> get_components() const;

  // Set the output file path (empty = stderr).
  // Called from Python's _sync_cpp_config() to unify sinks.
  // NOT safe to call while other threads are actively logging — the old
  // stream is destroyed immediately.  In practice this is only called at
  // module init or from set_log_level (Python GIL held), so concurrent
  // C++ logging traffic has not yet started or is quiesced.
  void set_log_file(const std::string& path);

  // Get the current output sink (lock-free atomic read).
  // The returned reference is valid only while set_log_file() is not
  // called — callers must not hold it across a reconfiguration boundary.
  std::ostream& sink() const {
    auto* s = sink_ptr_.load(std::memory_order_acquire);
    return s ? *s : std::cerr;
  }

  // Generation counter — incremented on every config mutation.
  // Thread_local caches compare against this to detect staleness.
  uint64_t generation() const {
    return generation_.load(std::memory_order_acquire);
  }

 private:
  LoggingConfig() = default;
  ~LoggingConfig() = default;

  // Prevent copying
  LoggingConfig(const LoggingConfig&) = delete;
  LoggingConfig& operator=(const LoggingConfig&) = delete;

  // Configuration storage (transparent hash for string_view lookups)
  std::unordered_map<std::string, LogLevel, StringHash, StringEqual> config_;

  // Shared mutex for configuration updates
  mutable std::shared_mutex mutex_;

  // Initialization flag
  std::atomic<bool> initialized_{false};

  // Bumped on every set_log_level / initialize_from_python call.
  // thread_local caches validate against this generation counter.
  std::atomic<uint64_t> generation_{0};

  // Output file path and stream (empty path = stderr)
  std::string log_file_path_;
  mutable std::unique_ptr<std::ofstream> log_file_stream_;

  // Lock-free sink pointer — updated by set_log_file(), read by sink().
  // Points to log_file_stream_.get() when a file is active, nullptr = stderr.
  std::atomic<std::ostream*> sink_ptr_{nullptr};

  // Resolve log level with hierarchical lookup
  LogLevel resolve_log_level(const std::string& component) const;
};

/**
 * RAII logger class for structured logging.
 *
 * Usage:
 *   Logger log("spyre.runtime", LogLevel::DEBUG);
 *   if (log.is_enabled()) {
 *       log.debug() << "Message: " << value;
 *   }
 */
class Logger {
 public:
  Logger(const std::string& component, LogLevel level);

  // Fast-path constructor: caller already verified is_enabled, so skip
  // the redundant get_log_level call.
  struct AlreadyEnabled {};
  Logger(const std::string& component, LogLevel level, AlreadyEnabled)
      : component_(component), requested_level_(level), min_level_(level) {}

  // Check if logging is enabled at the requested level
  bool is_enabled() const;

  // Stream-based logging
  class LogStream {
   public:
    LogStream(const std::string& component, LogLevel level, bool enabled);
    ~LogStream();

    template <typename T>
    LogStream& operator<<(const T& value) {
      if (enabled_) {
        stream_ << value;
      }
      return *this;
    }

   private:
    std::string component_;
    LogLevel level_;
    bool enabled_;
    std::ostringstream stream_;
  };

  LogStream debug();
  LogStream info();
  LogStream warning();
  LogStream error();
  LogStream critical();

 private:
  std::string component_;
  LogLevel requested_level_;
  LogLevel min_level_;
};

// Convenience macros for logging.
// SPYRE_LOG_ENABLED checks the thread_local cache (lock-free fast path).
// SPYRE_LOG gates on it, then constructs a Logger whose constructor skips
// the redundant get_log_level call (level already known to be enabled).
#define SPYRE_LOG_ENABLED(component, level) \
  torch_spyre::logging::LoggingConfig::instance().is_enabled(component, level)

#define SPYRE_LOG(component, level)                                            \
  if (auto spyre_log_enabled_ =                                                \
          SPYRE_LOG_ENABLED(component, torch_spyre::logging::LogLevel::level); \
      !spyre_log_enabled_) {                                                   \
  } else /* NOLINT(readability/braces) */                                      \
    torch_spyre::logging::Logger(                                              \
        component, torch_spyre::logging::LogLevel::level,                      \
        torch_spyre::logging::Logger::AlreadyEnabled{})                        \
        .level()

// Component-specific macros
#define SPYRE_RUNTIME_DEBUG() SPYRE_LOG("spyre.runtime", DEBUG)
#define SPYRE_RUNTIME_INFO() SPYRE_LOG("spyre.runtime", INFO)
#define SPYRE_RUNTIME_WARNING() SPYRE_LOG("spyre.runtime", WARNING)
#define SPYRE_RUNTIME_ERROR() SPYRE_LOG("spyre.runtime", ERROR)
#define SPYRE_RUNTIME_CRITICAL() SPYRE_LOG("spyre.runtime", CRITICAL)

}  // namespace logging
}  // namespace torch_spyre
