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

#include "spyre_pinned_allocator.h"

#include <cstring>
#include <flex/memory_interface/pinned_staging_cache.hpp>
#include <flex/runtime_stream/runtime_context.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "module.h"

namespace spyre {

struct PinnedAllocationState {
  PinnedAllocationState(std::shared_ptr<flex::PinnedBuffer> buffer,
                        std::shared_ptr<flex::PinnedStagingCache> cache)
      : buffer(std::move(buffer)), cache(std::move(cache)) {}

  ~PinnedAllocationState() noexcept {
    try {
      if (cache) {
        cache->release(std::move(buffer));
      }
    }
    catch (...) {
      // A deleter cannot report an exception. If cache return fails,
      // destruction of the remaining buffer ownership still unmaps and frees
      // the allocation.
    }
  }

  std::shared_ptr<flex::PinnedBuffer> buffer;
  std::shared_ptr<flex::PinnedStagingCache> cache;
};

c10::DataPtr SpyrePinnedAllocator::allocate(size_t size) {
  if (size == 0) {
    return {nullptr, nullptr, &deallocate, c10::DeviceType::CPU};
  }

  auto* runtime = GlobalRuntime::get();
  if (runtime == nullptr) {
    startRuntime();
    runtime = GlobalRuntime::get();
  }
  if (runtime == nullptr) {
    throw std::runtime_error(
        "SpyrePinnedAllocator: RuntimeContext is not initialized");
  }

  auto cache = runtime->getStagingCacheShared();
  if (!cache) {
    throw std::runtime_error(
        "SpyrePinnedAllocator: pinned staging cache is unavailable");
  }

  std::shared_ptr<flex::PinnedBuffer> buffer;
  try {
    buffer = cache->acquire(size);
  }
  catch (const std::exception& error) {
    std::ostringstream message;
    message << "SpyrePinnedAllocator: failed to allocate " << size
            << " pinned bytes: " << error.what();
    throw std::runtime_error(message.str());
  }
  if (!buffer) {
    throw std::runtime_error(
        "SpyrePinnedAllocator: pinned staging cache returned null");
  }

  void* const ptr = buffer->Hmva();
  auto state =
      std::make_shared<PinnedAllocationState>(buffer, std::move(cache));
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto [unused, inserted] = allocations_.emplace(ptr, state);
    if (!inserted) {
      throw std::runtime_error(
          "SpyrePinnedAllocator: cache returned an active allocation");
    }
  }

  return {ptr, ptr, &deallocate, c10::DeviceType::CPU};
}

c10::DeleterFnPtr SpyrePinnedAllocator::raw_deleter() const {
  return &deallocate;
}

void SpyrePinnedAllocator::copy_data(void* dest, const void* src,
                                     std::size_t count) const {
  std::memcpy(dest, src, count);
}

bool SpyrePinnedAllocator::record_event(void*, void*, c10::Stream) {
  // DMA submissions retain PinnedAllocationState explicitly through Flex's
  // host_lifetime token. This allocator does not maintain a separate event
  // queue, so it must not claim that an event was recorded.
  return false;
}

void SpyrePinnedAllocator::empty_cache() {
  // Free cache entries are owned by Flex's process-lifetime staging cache.
}

at::HostStats SpyrePinnedAllocator::get_stats() {
  return {};
}

void SpyrePinnedAllocator::reset_accumulated_stats() {}

void SpyrePinnedAllocator::reset_peak_stats() {}

void SpyrePinnedAllocator::deallocate(void* context) {
  if (context == nullptr) {
    return;
  }

  auto* allocator =
      static_cast<SpyrePinnedAllocator*>(GetSpyrePinnedAllocator());
  std::shared_ptr<PinnedAllocationState> state;
  {
    const std::lock_guard<std::mutex> lock(allocator->mutex_);
    auto it = allocator->allocations_.find(context);
    if (it == allocator->allocations_.end()) {
      return;
    }
    state = std::move(it->second);
    allocator->allocations_.erase(it);
  }
}

bool SpyrePinnedAllocator::isPinnedPtr(const void* ptr) const {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [base_ptr, state] : allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(base_ptr);
    if (address >= base && address - base < state->buffer->Capacity()) {
      return true;
    }
  }
  return false;
}

PinnedAllocation SpyrePinnedAllocator::retain(const void* ptr) const {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [base_ptr, state] : allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(base_ptr);
    if (address >= base && address - base < state->buffer->Capacity()) {
      const auto offset = address - base;
      return {
          state->buffer->Iova() + offset,
          state->buffer->Capacity() - offset,
          state,
      };
    }
  }
  return {};
}

at::HostAllocator* GetSpyrePinnedAllocator() {
  static auto* allocator = new SpyrePinnedAllocator();
  return allocator;
}

}  // namespace spyre
