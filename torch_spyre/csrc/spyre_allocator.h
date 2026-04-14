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

#include <flex/allocator/alloc_address.hpp>
#include <flex/allocator/flex_allocator.hpp>
#include <flex/device_types/device_memory_allocator.hpp>
#include <memory>
#include <utility>

namespace spyre {

struct SharedOwnerCtx {
  // Canonical allocation address from FlexAllocator (the "owner" per RFC).
  // On PF fallback path this is an empty CompositeAddress.
  flex::CompositeAddress owner;
  signed char device_id;
  size_t nbytes;
  // INTERIM: cached non-owning DeviceMemoryAllocationPtr for SetSpyreData().
  // Constructed once via makeInterimAllocationPtr() and reused for all
  // SetSpyreData() calls on this tensor.
  // Remove when SetSpyreData accepts CompositeAddress directly.
  // On PF fallback path this is an owning allocation (allocator_ != nullptr).
  flex::DeviceMemoryAllocationPtr interim_alloc_ptr;

  SharedOwnerCtx(flex::CompositeAddress addr, signed char dev_id, size_t nbytes,
                 flex::DeviceMemoryAllocationPtr interim)
      : owner(std::move(addr)),
        device_id(dev_id),
        nbytes(nbytes),
        interim_alloc_ptr(std::move(interim)) {}
};

// A custom allocator for our custom device, using FlexAllocator to manage
// device memory. Returns a handle (not a dereferenceable pointer) to the
// allocated memory via CompositeAddress stored in SharedOwnerCtx.
struct SpyreAllocator final : public c10::DeviceAllocator {
 private:
  SpyreAllocator();
  ~SpyreAllocator();
  static c10::CachingDeviceAllocator::DeviceStats stats_;
  static c10::CachingDeviceAllocator::StatTypes
      stat_types;  // {AGGREGATE, SMALL_POOL, LARGE_POOL}
  // Guard against use-after-destroy during static destruction.
  // Set to false in destructor; checked by ReportAndDelete.
  static bool alive_;

 public:
  static SpyreAllocator& instance();

  // Returns a mutable FlexAllocator from RuntimeContext.
  // Uses const_pointer_cast because RuntimeContext::getAllocator() returns
  // shared_ptr<const FlexAllocator>, but allocate()/deallocate() are
  // non-const (internally mutex-protected).
  static flex::FlexAllocator* getFlexAllocator();

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

  static void ReportAndDelete(void* ctx_void);

  c10::DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const final;
};

}  // namespace spyre
