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

#include "logging_config.h"

#include <ctime>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace torch_spyre {
namespace logging {

// Log level string conversion
const char* log_level_to_string(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARNING";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::CRITICAL:
      return "CRITICAL";
    default:
      return "NOTSET";
  }
}

LogLevel string_to_log_level(const std::string& level_str) {
  if (level_str == "DEBUG") return LogLevel::DEBUG;
  if (level_str == "INFO") return LogLevel::INFO;
  if (level_str == "WARNING") return LogLevel::WARNING;
  if (level_str == "ERROR") return LogLevel::ERROR;
  if (level_str == "CRITICAL") return LogLevel::CRITICAL;
  return LogLevel::NOTSET;
}

// LoggingConfig implementation
LoggingConfig& LoggingConfig::instance() {
  static LoggingConfig instance;
  return instance;
}

void LoggingConfig::initialize_from_python(
    const std::vector<std::pair<std::string, int>>& config) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  config_.clear();
  for (const auto& [component, level] : config) {
    config_[component] = static_cast<LogLevel>(level);
  }

  initialized_.store(true, std::memory_order_release);
  generation_.fetch_add(1, std::memory_order_release);
}

LogLevel LoggingConfig::get_log_level(const std::string& component) const {
  if (!initialized_.load(std::memory_order_acquire)) {
    return LogLevel::WARNING;
  }

  // 4-slot direct-mapped thread_local cache.  Steady-state reads for up to
  // 4 distinct components hit without any lock or hash-map lookup.
  static constexpr size_t kCacheSlots = 4;
  static constexpr size_t kCacheMask = kCacheSlots - 1;
  struct CacheEntry {
    uint64_t gen = 0;
    std::string component;
    LogLevel level = LogLevel::WARNING;
  };
  thread_local CacheEntry cache[kCacheSlots];

  // Simple hash to pick a slot (FNV-1a-style mixing of first few chars).
  size_t h = component.size();
  for (size_t i = 0; i < component.size() && i < 8; ++i) {
    h ^= static_cast<size_t>(component[i]);
    h *= 0x9e3779b97f4a7c15ULL;
  }
  auto& slot = cache[h & kCacheMask];

  uint64_t current_gen = generation_.load(std::memory_order_acquire);
  if (current_gen == slot.gen && component == slot.component) {
    return slot.level;
  }

  slot.level = resolve_log_level(component);
  slot.component = component;
  slot.gen = current_gen;
  return slot.level;
}

LogLevel LoggingConfig::resolve_log_level(const std::string& component) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  // Exact match — component is already a std::string, no allocation
  auto it = config_.find(component);
  if (it != config_.end()) {
    return it->second;
  }

  // Walk up hierarchy using string_view for rfind (no allocations for
  // position tracking).  Only the find() call constructs a key, which
  // the map's transparent hash resolves without an owning copy.
  std::string_view sv(component);
  while (true) {
    auto pos = sv.rfind('.');
    if (pos == std::string_view::npos) {
      break;
    }
    sv = sv.substr(0, pos);
    it = config_.find(sv);  // transparent lookup, no allocation
    if (it != config_.end()) {
      return it->second;
    }
  }

  return LogLevel::WARNING;
}

void LoggingConfig::set_log_level(const std::string& component,
                                  LogLevel level) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  config_[component] = level;
  initialized_.store(true, std::memory_order_release);
  generation_.fetch_add(1, std::memory_order_release);
}

std::vector<std::string> LoggingConfig::get_components() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  std::vector<std::string> components;
  components.reserve(config_.size());

  for (const auto& [component, _] : config_) {
    components.push_back(component);
  }

  return components;
}

void LoggingConfig::set_log_file(const std::string& path) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (path.empty()) {
    sink_ptr_.store(nullptr, std::memory_order_release);
    log_file_stream_.reset();
    log_file_path_.clear();
  } else {
    auto fs = std::make_unique<std::ofstream>(path, std::ios::app);
    if (fs->is_open()) {
      log_file_stream_ = std::move(fs);
      log_file_path_ = path;
      sink_ptr_.store(log_file_stream_.get(), std::memory_order_release);
    }
  }
}

// Logger implementation
Logger::Logger(const std::string& component, LogLevel level)
    : component_(component),
      requested_level_(level),
      min_level_(LoggingConfig::instance().get_log_level(component)) {}

bool Logger::is_enabled() const {
  return min_level_ <= requested_level_;
}

Logger::LogStream Logger::debug() {
  return LogStream(component_, LogLevel::DEBUG, min_level_ <= LogLevel::DEBUG);
}

Logger::LogStream Logger::info() {
  return LogStream(component_, LogLevel::INFO, min_level_ <= LogLevel::INFO);
}

Logger::LogStream Logger::warning() {
  return LogStream(component_, LogLevel::WARNING,
                   min_level_ <= LogLevel::WARNING);
}

Logger::LogStream Logger::error() {
  return LogStream(component_, LogLevel::ERROR, min_level_ <= LogLevel::ERROR);
}

Logger::LogStream Logger::critical() {
  return LogStream(component_, LogLevel::CRITICAL,
                   min_level_ <= LogLevel::CRITICAL);
}

// LogStream implementation
Logger::LogStream::LogStream(const std::string& component, LogLevel level,
                             bool enabled)
    : component_(component), level_(level), enabled_(enabled) {}

Logger::LogStream::~LogStream() {
  if (enabled_ && !stream_.str().empty()) {
    auto now = std::time(nullptr);
    std::tm tm_buf;              // stack-local buffer, one per thread
    localtime_r(&now, &tm_buf);  // thread-safe version
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                  &tm_buf);  // use our local buffer

    // Assemble into a local string so concurrent threads cannot interleave
    // within a single log record.
    std::string msg = stream_.str();
    std::string record;
    record.reserve(64 + msg.size());
    record += '[';
    record += log_level_to_string(level_);
    record += "] [";
    record += component_;
    record += "] ";
    record += time_buf;
    record += ' ';
    record += msg;
    record += '\n';

    // Single write to the configured sink (respects SPYRE_LOG_FILE).
    // Concurrent calls from multiple threads are safe on POSIX/Linux:
    // std::cerr (sync_with_stdio default) and ofstream both delegate to
    // FILE* whose fwrite() is internally serialized via flockfile().
    // The C++ standard alone does not guarantee this — do not port to a
    // platform without POSIX FILE* thread-safety without adding a mutex.
    auto& out = LoggingConfig::instance().sink();
    out.write(record.data(), static_cast<std::streamsize>(record.size()));
  }
}

}  // namespace logging
}  // namespace torch_spyre
