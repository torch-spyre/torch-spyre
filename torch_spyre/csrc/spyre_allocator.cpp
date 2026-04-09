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
#include "spyre_allocator.h"

#include <utility>

#include "logging.h"
#include "module.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {

SpyreAllocator::SpyreAllocator() = default;
c10::CachingDeviceAllocator::DeviceStats SpyreAllocator::stats_;
c10::CachingDeviceAllocator::StatTypes SpyreAllocator::stat_types = {
    true, false, false};  // {AGGREGATE, SMALL_POOL, LARGE_POOL}

flex::DeviceMemoryAllocatorPtr SpyreAllocator::getAllocator(
    unsigned int /*dev_id*/) {
  // Each process has exactly one runtime with one device handle (index 0).
  // The PyTorch device index (dev_id) is the logical index across processes,
  // not an index into the runtime's device handle vector.
  return GlobalRuntime::get()->GetDeviceHandle(0)->GetDeviceMemoryAllocator();
}

SpyreAllocator& SpyreAllocator::instance() {
  static SpyreAllocator allocator;
  return allocator;
}

bool SpyreAllocator::initialized() {
  return true;
}

void SpyreAllocator::emptyCache(c10::MempoolId_t mempool_id) {}

void SpyreAllocator::recordStream(const c10::DataPtr& ptr, c10::Stream stream) {
}

c10::CachingDeviceAllocator::DeviceStats SpyreAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  return stats_;
}

void SpyreAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_accumulated();
        stats_.allocation[stat_type].reset_accumulated();
      });
}

void SpyreAllocator::resetPeakStats(c10::DeviceIndex device) {
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_peak();
        stats_.allocation[stat_type].reset_peak();
      });
}

void SpyreAllocator::recordAlloc(size_t nbytes, void* data, int device_id) {
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocation[stat_type].increase(1);
        stats_.allocated_bytes[stat_type].increase(nbytes);
      });

  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(
      &data,
      nbytes,  // alloc_size
      stats_
          .allocated_bytes[static_cast<size_t>(
              c10::CachingAllocator::StatType::AGGREGATE)]
          .current,  // total_allocated
      stats_
          .allocated_bytes[static_cast<size_t>(
              c10::CachingAllocator::StatType::AGGREGATE)]
          .current,  // total_reserved (currently same as total_allocated)
      curr_device);
}

void SpyreAllocator::recordRelease(size_t nbytes, void* data, int device_id) {
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocation[stat_type].decrease(1);
        stats_.allocated_bytes[stat_type].decrease(nbytes);
      });

  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(
      &data,
      -nbytes,  // alloc_size
      stats_
          .allocated_bytes[static_cast<size_t>(
              c10::CachingAllocator::StatType::AGGREGATE)]
          .current,  // total_allocated
      stats_
          .allocated_bytes[static_cast<size_t>(
              c10::CachingAllocator::StatType::AGGREGATE)]
          .current,  // total_reserved (currently same as total_allocated)
      curr_device);
}

c10::DataPtr SpyreAllocator::allocate(size_t nbytes) {
  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice();

  auto device_id = curr_device.index();

  DEBUGINFO("allocating ", nbytes, " (bytes) on Spyre", curr_device);
  if (nbytes == 0) {
    return {nullptr, nullptr, &ReportAndDelete, curr_device};
  }
  auto allocator = getAllocator(device_id);
  flex::DeviceMemoryAllocationPtr data;  // a smart-pointer object
  // NOTE: last argument should be set to 0
  allocator->TryAllocate(&data, nbytes, 0);
  TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on Spyre device.");
  auto* ctx = new SharedOwnerCtx{std::move(data), device_id, nbytes};
  void* ctx_void = static_cast<void*>(ctx);

  void* data_void = static_cast<void*>(ctx->owner.get());
  recordAlloc(nbytes, data_void, device_id);

  auto data_ptr_result =
      at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);

  return data_ptr_result;
}

void SpyreAllocator::ReportAndDelete(void* ctx_void) {
  if (!ctx_void) {
    return;
  }
  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  size_t nbytes = ctx->nbytes;

  SpyreAllocator::instance().recordRelease(
      nbytes, static_cast<void*>(ctx->owner.get()), ctx->device_id);
  delete ctx;
}

// The raw deleter only gets passed the data ptr, no context, so
// it would not work right now. To implement this, we first need to
// create a runtime interface that can correctly free an allocation
// only based on the data ptr, without the allocation idx from the
// context
c10::DeleterFnPtr SpyreAllocator::raw_deleter() const {
  return nullptr;
}

void SpyreAllocator::copy_data(void* dest, const void* src,
                               std::size_t count) const {
  py::gil_scoped_acquire acquire;
  DEBUGINFO("entering allocator->copy_data method");
  // do nothing -- look into when this is called
  // spyre_copy_from(reinterpret_cast<spyre_ptr_t>(dest),
  // reinterpret_cast<spyre_ptr_t>(src));
}

// Register our custom allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &SpyreAllocator::instance());

}  // namespace spyre
