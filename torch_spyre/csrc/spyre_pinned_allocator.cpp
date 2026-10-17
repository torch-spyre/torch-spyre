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
#include <flex/runtime_stream/runtime_context.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

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

  void* const ptr = buffer->hmva();
  auto state =
      std::make_shared<PinnedAllocationState>(buffer, std::move(cache));
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto [unused, inserted] = allocations_.emplace(ptr, state);
    if (!inserted) {
      throw std::runtime_error(
          "SpyrePinnedAllocator: cache returned an active allocation");
    }
    caches_.insert(std::weak_ptr<flex::PinnedStagingCache>(state->cache));
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
  // D2H retains PinnedAllocationState through Flex's host_lifetime token; H2D
  // converts into a Flex-owned staging buffer before asynchronous submission.
  // This allocator has no event queue, so it must not claim to record an event.
  return false;
}

void SpyrePinnedAllocator::empty_cache() {
  std::vector<std::shared_ptr<flex::PinnedStagingCache>> caches;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = caches_.begin(); it != caches_.end();) {
      if (auto cache = it->lock()) {
        caches.push_back(std::move(cache));
        ++it;
      } else {
        it = caches_.erase(it);
      }
    }
  }
  for (const auto& cache : caches) {
    cache->clear();
  }
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
    if (address >= base && address - base < state->buffer->capacity()) {
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
    if (address >= base && address - base < state->buffer->capacity()) {
      const auto offset = address - base;
      return {
          state->buffer,
          offset,
          state->buffer->capacity() - offset,
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
