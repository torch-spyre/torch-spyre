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

#include "kernel_provenance_registry.h"

#include <c10/util/Exception.h>

#include <atomic>
#include <mutex>
#include <nlohmann/json.hpp>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spyre {
namespace {

// These literals are pinned to the Python constants by
// test_cpp_parser_literals_match_python_constants.
constexpr std::string_view kEventNamePrefix = "spyre_kernel_v1_";
constexpr size_t kKeyWidth = 16;

bool isDisplayCharacter(char value) {
  return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
         (value >= '0' && value <= '9') || value == '_';
}

bool isKeyCharacter(char value) {
  return (value >= 'a' && value <= 'z') || (value >= '2' && value <= '7');
}

bool isDecimalCharacter(char value) {
  return value >= '0' && value <= '9';
}

struct RegistryState {
  std::shared_mutex mutex;
  std::unordered_map<std::string, std::vector<std::string>> entries;
  std::atomic<size_t> hits{0};
  std::atomic<size_t> misses{0};
  std::atomic<size_t> conflicts{0};
};

RegistryState& registryState() {
  static RegistryState state;
  return state;
}

}  // namespace

std::optional<std::string> extractKernelProvenanceKey(
    const std::string& event_name) {
  std::string_view name(event_name);
  const size_t step_separator = name.rfind('#');
  if (step_separator != std::string_view::npos) {
    const std::string_view step = name.substr(step_separator + 1);
    if (step.empty()) {
      return std::nullopt;
    }
    for (const char value : step) {
      if (!isDecimalCharacter(value)) {
        return std::nullopt;
      }
    }
    name = name.substr(0, step_separator);
  }

  if (name.size() <= kEventNamePrefix.size() + kKeyWidth + 1 ||
      name.substr(0, kEventNamePrefix.size()) != kEventNamePrefix) {
    return std::nullopt;
  }

  const size_t key_start = name.size() - kKeyWidth;
  if (name[key_start - 1] != '_') {
    return std::nullopt;
  }

  const std::string_view display = name.substr(
      kEventNamePrefix.size(), key_start - kEventNamePrefix.size() - 1);
  if (display.empty()) {
    return std::nullopt;
  }
  for (const char value : display) {
    if (!isDisplayCharacter(value)) {
      return std::nullopt;
    }
  }

  const std::string_view key = name.substr(key_start);
  for (const char value : key) {
    if (!isKeyCharacter(value)) {
      return std::nullopt;
    }
  }
  return std::string(key);
}

bool registerKernelProvenance(const std::string& event_base_name,
                              std::vector<std::string> debug_handle_ids) {
  const auto key = extractKernelProvenanceKey(event_base_name);
  if (!key.has_value()) {
    TORCH_WARN_ONCE(
        "Cannot register Spyre kernel provenance for a noncanonical event "
        "name: ",
        event_base_name);
    return false;
  }

  auto& state = registryState();
  std::unique_lock<std::shared_mutex> lock(state.mutex);
  const auto existing = state.entries.find(*key);
  if (existing == state.entries.end()) {
    state.entries.emplace(*key, std::move(debug_handle_ids));
    return true;
  }
  if (existing->second == debug_handle_ids) {
    return true;
  }

  state.conflicts.fetch_add(1, std::memory_order_relaxed);
  TORCH_WARN_ONCE("Conflicting Spyre kernel provenance registration for key ",
                  *key, "; preserving the first debug-handle mapping ",
                  nlohmann::json(existing->second).dump(), " and rejecting ",
                  nlohmann::json(debug_handle_ids).dump());
  return false;
}

const std::vector<std::string>* lookupKernelProvenance(const std::string& key) {
  auto& state = registryState();
  std::shared_lock<std::shared_mutex> lock(state.mutex);
  const auto entry = state.entries.find(key);
  if (entry == state.entries.end()) {
    state.misses.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  state.hits.fetch_add(1, std::memory_order_relaxed);
  return &entry->second;
}

KernelProvenanceRegistryStats kernelProvenanceRegistryStats() {
  auto& state = registryState();
  std::shared_lock<std::shared_mutex> lock(state.mutex);
  return KernelProvenanceRegistryStats{
      state.entries.size(),
      state.hits.load(std::memory_order_relaxed),
      state.misses.load(std::memory_order_relaxed),
      state.conflicts.load(std::memory_order_relaxed),
  };
}

}  // namespace spyre
