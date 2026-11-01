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

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace spyre {

// Defined in spyre_mem.cpp: a single-shard view whose DataPtr context is the
// SharedOwnerCtx below. Forward-declared so the ctx can point back at it.
struct ShardViewCtx;

// Sharding of a tensor's storage: shard i lives in the chunk whose
// segment-emulated placement has logical_chunk_id == i, encoded in that
// shard's own device layout (shard_device_bytes). Address-free.
struct ShardedStorageInfo {
  int64_t shard_dim;
  uint32_t num_shards;
  uint64_t shard_device_bytes;
};

struct SharedOwnerCtx {
  // Non-sharded storage: the owning allocation. Sharded storage: a non-owning
  // whole-storage view over the shards' chunks (hand-built, so its destructor
  // never frees); the owning per-shard allocations live in shard_addrs.
  flex::CompositeAddress composite_addr;
  // Sharded storage only: one owning plain allocation per shard, indexed by
  // shard ordinal (shard i backs core i). Empty for non-sharded storage.
  std::vector<flex::CompositeAddress> shard_addrs;
  signed char device_id;
  flex::MemoryType memory_type;
  // Set only for sharded storage.
  std::optional<ShardedStorageInfo> sharding;
  // Set only on the SharedOwnerCtx embedded in a ShardViewCtx: points back
  // at the enclosing view so the view's DataPtr deleter can free it.
  ShardViewCtx* shard_view = nullptr;

  SharedOwnerCtx(flex::CompositeAddress addr, signed char dev_id,
                 flex::MemoryType mem_type)
      : composite_addr(std::move(addr)),
        device_id(dev_id),
        memory_type(mem_type) {}
};

// A custom allocator for our custom device, which returns a handle to the
// allocated memory, not the actual pointer.
struct SpyreAllocator final : public c10::DeviceAllocator {
 private:
  SpyreAllocator();
  static c10::CachingDeviceAllocator::DeviceStats stats_;
  static c10::CachingDeviceAllocator::StatTypes
      stat_types;  // {AGGREGATE, SMALL_POOL, LARGE_POOL}
  static std::mutex stats_mutex_;

  static std::shared_ptr<flex::FlexAllocator> getFlexAllocator();

  // Memory pressure callback for FlexAllocator
  // Invoked when allocator exhausts all regions
  // Releases mutex, triggers Python GC, re-acquires mutex
  static void memoryPressureCallback(std::unique_lock<std::mutex>& lock);

 public:
  static SpyreAllocator& instance();
  bool initialized() override;

  void emptyCache(c10::MempoolId_t mempool_id) override;

  void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override;

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;

  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override;

  void resetAccumulatedStats(c10::DeviceIndex device) override;

  void resetPeakStats(c10::DeviceIndex device) override;

  void recordAlloc(size_t nbytes, void* data, int device,
                   flex::MemoryType memory_type);

  void recordRelease(size_t nbytes, void* data, int device,
                     flex::MemoryType memory_type);

  c10::DataPtr allocate(size_t nbytes) override;

  c10::DataPtr allocate(size_t nbytes,
                        const flex::AllocationDirective& directive);

  // Allocate sharded storage: logical_chunks segment-emulated chunks, each at
  // least shard_device_bytes, tagged with the sharding record. The returned
  // DataPtr owns the allocation through a SharedOwnerCtx.
  static c10::DataPtr allocateShardedStorage(int64_t shard_dim,
                                             uint32_t logical_chunks,
                                             uint64_t shard_device_bytes);

  static void ReportAndDelete(void* ctx_void);

  c10::DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const final;

  uint64_t compositeAddressToDeviceAddress(
      const flex::CompositeAddress& addr) const;
};

}  // namespace spyre
