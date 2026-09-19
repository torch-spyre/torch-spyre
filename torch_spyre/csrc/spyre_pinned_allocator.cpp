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

#include <ATen/ATen.h>

#include <cstring>
#include <flex/memory_interface/pinned_staging_cache.hpp>
#include <flex/runtime_stream/runtime_context.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "logging_config.h"
#include "module.h"  // For GlobalRuntime, startRuntime

namespace spyre {

struct PinnedAllocationState {
  PinnedAllocationState(std::shared_ptr<flex::PinnedBuffer> buffer_in,
                        std::shared_ptr<flex::PinnedStagingCache> cache_in)
      : buffer(std::move(buffer_in)), cache(std::move(cache_in)) {}

  std::shared_ptr<flex::PinnedBuffer> buffer;
  std::shared_ptr<flex::PinnedStagingCache> cache;

  ~PinnedAllocationState() noexcept {
    try {
      if (cache) {
        cache->release(std::move(buffer));
      }
    }
    catch (...) {
      // Destructors must not throw. Dropping buffer below still safely unmaps
      // and frees it if returning it to the cache fails.
    }
  }
};

c10::DataPtr SpyrePinnedAllocator::allocate(size_t size) {
  if (size == 0) {
    return {nullptr, nullptr, &deallocate, c10::Device(c10::DeviceType::CPU)};
  }

  // Ensure runtime is initialized (needed for PinnedStagingCache)
  // This is safe because startRuntime() is std::call_once (cheap after first
  // call)
  auto rtc = GlobalRuntime::get();
  if (!rtc) {
    // Runtime not initialized yet - trigger initialization
    startRuntime();
    rtc = GlobalRuntime::get();
    if (!rtc) {
      throw std::runtime_error(
          "SpyrePinnedAllocator: RuntimeContext not initialized");
    }
  }

  auto cache = rtc->getStagingCacheShared();
  if (!cache) {
    throw std::runtime_error(
        "SpyrePinnedAllocator: PinnedStagingCache not available");
  }

  // Acquire a pinned, pre-IOMMU-mapped buffer from flex's cache
  // This avoids per-transfer pin/map overhead
  std::shared_ptr<flex::PinnedBuffer> buffer;
  try {
    buffer = cache->acquire(size);
  }
  catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "SpyrePinnedAllocator: failed to acquire " << size
        << " bytes from cache: " << e.what();
    throw std::runtime_error(oss.str());
  }

  if (!buffer) {
    throw std::runtime_error(
        "SpyrePinnedAllocator: cache returned null buffer");
  }

  void* ptr = buffer->Hmva();

  // The state returns the buffer to the same runtime cache when the tensor and
  // all asynchronous transfer tokens release their shared ownership.
  auto state =
      std::make_shared<PinnedAllocationState>(buffer, std::move(cache));
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    buffer_map_[ptr] = std::move(state);
  }

  torch_spyre::logging::Logger("spyre.runtime",
                               torch_spyre::logging::LogLevel::DEBUG)
          .debug()
      << "SpyrePinnedAllocator: acquired " << size << " bytes (capacity "
      << buffer->Capacity() << ") from flex cache at " << ptr << " with IOVA 0x"
      << std::hex << buffer->Iova() << std::dec;

  // The context for the deleter is the raw pointer
  // We'll look up the buffer in deallocate()
  return {ptr, ptr, &deallocate, c10::Device(c10::DeviceType::CPU)};
}

c10::DeleterFnPtr SpyrePinnedAllocator::raw_deleter() const {
  return &deallocate;
}

void SpyrePinnedAllocator::copy_data(void* dest, const void* src,
                                     std::size_t count) const {
  std::memcpy(dest, src, count);
}

void SpyrePinnedAllocator::deallocate(void* ptr) {
  if (!ptr) {
    return;
  }

  auto* allocator =
      static_cast<SpyrePinnedAllocator*>(GetSpyrePinnedAllocator());

  // Erasing the tensor's ownership either returns the buffer immediately or
  // defers that release until the final asynchronous DMA token is destroyed.
  std::shared_ptr<PinnedAllocationState> state;
  {
    std::lock_guard<std::mutex> lock(allocator->map_mutex_);
    auto it = allocator->buffer_map_.find(ptr);
    if (it != allocator->buffer_map_.end()) {
      state = std::move(it->second);
      allocator->buffer_map_.erase(it);
    }
  }

  if (!state) {
    torch_spyre::logging::Logger("spyre.runtime",
                                 torch_spyre::logging::LogLevel::WARNING)
            .warning()
        << "SpyrePinnedAllocator: unknown pointer " << ptr
        << " (not from this allocator)";
    return;
  }

  const bool release_deferred = state.use_count() > 1;
  state.reset();
  torch_spyre::logging::Logger("spyre.runtime",
                               torch_spyre::logging::LogLevel::DEBUG)
          .debug()
      << "SpyrePinnedAllocator: "
      << (release_deferred ? "deferred release of in-flight buffer at "
                           : "released buffer back to flex cache at ")
      << ptr;
}

bool SpyrePinnedAllocator::isPinnedPtr(const void* ptr) const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& [base_ptr, state] : buffer_map_) {
    const auto base = reinterpret_cast<uintptr_t>(base_ptr);
    const auto capacity = state->buffer->Capacity();
    if (addr >= base && addr - base < capacity) {
      return true;
    }
  }
  return false;
}

PinnedAllocation SpyrePinnedAllocator::retain(const void* ptr) const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& [base_ptr, state] : buffer_map_) {
    const auto base = reinterpret_cast<uintptr_t>(base_ptr);
    const auto capacity = state->buffer->Capacity();
    if (addr >= base && addr - base < capacity) {
      const auto offset = addr - base;
      return {state->buffer->Iova() + offset, state};
    }
  }
  return {};
}

// Global allocator instance (singleton)
at::HostAllocator* GetSpyrePinnedAllocator() {
  static SpyrePinnedAllocator* allocator = new SpyrePinnedAllocator();
  return allocator;
}

}  // namespace spyre
