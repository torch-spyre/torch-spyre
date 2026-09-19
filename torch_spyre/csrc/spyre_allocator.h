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

#include <ATen/core/Tensor.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace spyre {

struct SharedOwnerCtx {
  flex::CompositeAddress composite_addr;
  signed char device_id;

  SharedOwnerCtx(flex::CompositeAddress addr, signed char dev_id)
      : composite_addr(std::move(addr)), device_id(dev_id) {}
};

// ─── TEMPORARY (T5, 1p5-emulation epic) ──────────────────────────────────────
// Per-chunk description of a tensor's device allocation, exposed to Python so
// the thin-slice test can assert that an allocation really is interleaved
// (multi-chunk, one chunk per memory domain, every chunk backed by a real
// XLat segment) rather than silently passing on a single-chunk allocation.
//
// Debug-only introspection. Remove together with the interleave env gate when
// T6 replaces it with a real topology-derived eligibility policy.
struct CompositeChunkInfo {
  uint32_t domain_id;
  uint64_t region_id;
  uint64_t offset;
  uint64_t size;
  // Region's XLat segment id. Under an emulated multi-domain topology a surplus
  // tensor region can carry UNMAPPED_SEGMENT_ID, meaning it has no device
  // address and must never be dispatched (flex T3 Finding B / T4 backstop).
  uint32_t segment_id;
  bool segment_mapped;
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

  void resetAccumulatedStats(c10::DeviceIndex device) override;

  void resetPeakStats(c10::DeviceIndex device) override;

  void recordAlloc(size_t nbytes, void* data, int device);

  void recordRelease(size_t nbytes, void* data, int device);

  c10::DataPtr allocate(size_t nbytes) override;

  c10::DataPtr allocate(size_t nbytes,
                        const flex::AllocationDirective& directive);

  static void ReportAndDelete(void* ctx_void);

  c10::DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const final;

  uint64_t compositeAddressToDmva(const flex::CompositeAddress& addr) const;

  // ─── TEMPORARY (T5, 1p5-emulation epic) ────────────────────────────────────
  // Test-only hooks for the copy-only thin slice. Remove with the interleave
  // gate when T6 lands.
  //
  // Return the chunk layout of a Spyre tensor's device allocation.
  static std::vector<CompositeChunkInfo> debugCompositeChunks(
      const at::Tensor& tensor);
  // Flip the interleave gate at runtime; returns the previous value. Needed
  // because only one process may hold the device, so an interleaved run and its
  // Bind{0} baseline must both happen in this process.
  static bool debugSetEmulateInterleave(bool enabled);
  // Memory domains flex reports; 0 if the runtime is not started.
  static size_t debugNumMemoryDomains();
};

}  // namespace spyre
