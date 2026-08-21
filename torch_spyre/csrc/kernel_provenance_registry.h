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

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace spyre {

struct KernelProvenanceRegistryStats {
  size_t entries;
  size_t hits;
  size_t misses;
  size_t conflicts;
};

/** Extract the finalized-bundle key from a provenance-aware event name. */
std::optional<std::string> extractKernelProvenanceKey(
    const std::string& event_name);

/**
 * Register the direct debug handles for one provenance-aware event base name.
 *
 * Entries are immutable and retained for the process lifetime because AIUPTI
 * activities may be consumed asynchronously after kernel preparation.
 * Registration is unconditional, so entries remain even if no profiler
 * attaches. Growth is bounded by distinct prepared bundles, not trace sessions.
 *
 * An identical duplicate is accepted. A same-key, different-handle registration
 * preserves the first value and reports a conflict. Such a conflict indicates
 * either an extraordinarily unlikely collision in the public 80-bit key or a
 * defect in canonicalization or registration; it is not a normal merge path.
 *
 * Returns true for an insertion or identical duplicate, and false for an
 * invalid event name or a conflicting registration.
 */
bool registerKernelProvenance(const std::string& event_base_name,
                              std::vector<std::string> debug_handle_ids);

/**
 * Return a process-lifetime handle list, or nullptr when none is registered.
 * Insert-only storage keeps the returned pointer valid after the lock is
 * released.
 */
const std::vector<std::string>* lookupKernelProvenance(const std::string& key);

/** Return a snapshot of the process-lifetime registry counters. */
KernelProvenanceRegistryStats kernelProvenanceRegistryStats();

}  // namespace spyre
