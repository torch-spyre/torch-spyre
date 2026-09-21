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

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace flex {
class PinnedBuffer;
class PinnedStagingCache;
}  // namespace flex

namespace spyre {

struct PinnedAllocationState;

struct PinnedAllocation {
  uint64_t iova = 0;
  std::shared_ptr<PinnedAllocationState> owner;

  explicit operator bool() const {
    return owner != nullptr;
  }
};

class SpyrePinnedAllocator : public at::HostAllocator {
 private:
  std::unordered_map<void*, std::shared_ptr<PinnedAllocationState>> buffer_map_;
  mutable std::mutex map_mutex_;

 public:
  SpyrePinnedAllocator() = default;

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;
  static void deallocate(void* ptr);

  bool isPinnedPtr(const void* ptr) const;
  PinnedAllocation retain(const void* ptr) const;

  bool record_event(void*, void*, c10::Stream) override {
    return true;
  }

  void empty_cache() override {}

  at::HostStats get_stats() override {
    return at::HostStats{};
  }

  void reset_accumulated_stats() override {}
  void reset_peak_stats() override {}
};

at::HostAllocator* GetSpyrePinnedAllocator();

}  // namespace spyre
