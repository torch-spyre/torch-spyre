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

#include <ATen/core/CachingHostAllocator.h>

#include <cstddef>
#include <flex/memory_interface/pinned_staging_cache.hpp>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

namespace spyre {

struct PinnedAllocationState;

struct PinnedAllocation {
  std::shared_ptr<flex::PinnedBuffer> buffer;
  size_t offset = 0;
  size_t capacity = 0;
  std::shared_ptr<PinnedAllocationState> owner;

  explicit operator bool() const {
    return owner != nullptr;
  }
};

class SpyrePinnedAllocator : public at::HostAllocator {
 public:
  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override;
  void empty_cache() override;
  at::HostStats get_stats() override;
  void reset_accumulated_stats() override;
  void reset_peak_stats() override;

  bool isPinnedPtr(const void* ptr) const;
  PinnedAllocation retain(const void* ptr) const;

 private:
  static void deallocate(void* context);

  std::unordered_map<void*, std::shared_ptr<PinnedAllocationState>>
      allocations_;
  std::set<std::weak_ptr<flex::PinnedStagingCache>,
           std::owner_less<std::weak_ptr<flex::PinnedStagingCache>>>
      caches_;
  mutable std::mutex mutex_;
};

at::HostAllocator* GetSpyrePinnedAllocator();

}  // namespace spyre
