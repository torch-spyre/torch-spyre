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

#include <memory>
#include <mutex>
#include <utility>

#include "logging.h"
#include "module.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {

SpyreAllocator::SpyreAllocator() {
  // Callback registration is deferred until first allocation
  // to avoid accessing RuntimeContext during static initialization
}

c10::CachingDeviceAllocator::DeviceStats SpyreAllocator::stats_;
c10::CachingDeviceAllocator::StatTypes SpyreAllocator::stat_types = {
    true, false, false};  // {AGGREGATE, SMALL_POOL, LARGE_POOL}
std::mutex SpyreAllocator::stats_mutex_;

std::shared_ptr<flex::FlexAllocator> SpyreAllocator::getFlexAllocator() {
  // FlexAllocator is owned by RuntimeContext (one per device per process).
  // RuntimeContext::getAllocator() returns shared_ptr<FlexAllocator>;
  auto flex_alloc = flex::getFlexRuntimeContext()->getAllocator();

  // Register memory pressure callback on first access (lazy initialization)
  static std::once_flag callback_registered;
  std::call_once(callback_registered, [&flex_alloc]() {
    if (flex_alloc) {
      flex_alloc->registerMemoryPressureCallback(
          &SpyreAllocator::memoryPressureCallback);
      DEBUGINFO(
          "SpyreAllocator: registered memory pressure callback with "
          "FlexAllocator");
    }
  });

  return flex_alloc;
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
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

void SpyreAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_accumulated();
        stats_.allocation[stat_type].reset_accumulated();
      });
}

void SpyreAllocator::resetPeakStats(c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_peak();
        stats_.allocation[stat_type].reset_peak();
      });
}

void SpyreAllocator::recordAlloc(size_t nbytes, void* data, int device_id) {
  int64_t total_allocated;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    c10::CachingAllocator::for_each_selected_stat_type(
        stat_types, [&](size_t stat_type) {
          stats_.allocation[stat_type].increase(1);
          stats_.allocated_bytes[stat_type].increase(nbytes);
        });
    total_allocated = stats_
                          .allocated_bytes[static_cast<size_t>(
                              c10::CachingAllocator::StatType::AGGREGATE)]
                          .current;
  }
  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(
      data,
      nbytes,           // alloc_size
      total_allocated,  // total_allocated
      total_allocated,  // total_reserved (currently same as total_allocated)
      curr_device);
}

void SpyreAllocator::recordRelease(size_t nbytes, void* data, int device_id) {
  int64_t total_allocated;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    c10::CachingAllocator::for_each_selected_stat_type(
        stat_types, [&](size_t stat_type) {
          stats_.allocation[stat_type].decrease(1);
          stats_.allocated_bytes[stat_type].decrease(nbytes);
        });
    total_allocated = stats_
                          .allocated_bytes[static_cast<size_t>(
                              c10::CachingAllocator::StatType::AGGREGATE)]
                          .current;
  }
  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(
      data,
      -static_cast<int64_t>(nbytes),  // alloc_size
      total_allocated,                // total_allocated
      total_allocated,  // total_reserved (currently same as total_allocated)
      curr_device);
}

c10::DataPtr SpyreAllocator::allocate(size_t nbytes) {
  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Tensor);
  return SpyreAllocator::allocate(nbytes, directive);
}

c10::DataPtr SpyreAllocator::allocate(
    size_t nbytes, const flex::AllocationDirective& directive) {
  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice();

  auto device_id = curr_device.index();

  DEBUGINFO("allocating ", nbytes, " (bytes) on Spyre", curr_device);
  if (nbytes == 0) {
    return {nullptr, nullptr, &ReportAndDelete, curr_device};
  }
  // Get shared_ptr to keep FlexAllocator alive during operations
  auto flex_alloc = getFlexAllocator();

  // Allocate first-class raw storage via CompositeAddress.
  flex::CompositeAddress composite_addr =
      flex_alloc->allocate(nbytes, directive);

  DEBUGINFO("allocated ", composite_addr);
  // FlexAllocator rounds up to DEVICE_ALIGNMENT (128 bytes), so the actual
  // allocation may be larger than the requested nbytes. Use total_size() for
  // accurate memory profiling.
  size_t actual_nbytes = composite_addr.total_size();

  auto* ctx = new SharedOwnerCtx(std::move(composite_addr), device_id);
  void* ctx_void = static_cast<void*>(ctx);

  // Use the SharedOwnerCtx pointer as the unique data handle for c10::DataPtr.
  // This pointer is never dereferenced — it serves only as a unique token for
  // memory profiling (recordAlloc/recordRelease).
  void* data_void = static_cast<void*>(ctx);
  recordAlloc(actual_nbytes, data_void, device_id);

  auto data_ptr_result =
      at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);

  return data_ptr_result;
}

void SpyreAllocator::ReportAndDelete(void* ctx_void) {
  if (!ctx_void) {
    return;
  }
  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  size_t nbytes = ctx->composite_addr.total_size();

  SpyreAllocator::instance().recordRelease(nbytes, static_cast<void*>(ctx),
                                           ctx->device_id);
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

uint32_t SpyreAllocator::segmentForRegion(uint64_t region_id) const {
  return getFlexAllocator()->getIdToRegionMap().at(region_id)->segment_id();
}

void SpyreAllocator::memoryPressureCallback(
    std::unique_lock<std::mutex>& lock) {
  // This callback is invoked by FlexAllocator while holding allocator_mutex
  // (via a unique_lock). We must:
  // 1. Release the allocator mutex
  // 2. Acquire the Python GIL
  // 3. Call PyGC_Collect()
  // 4. Release the GIL
  // 5. Re-acquire the allocator mutex
  //
  // Lock ordering: allocator_mutex -> (release) -> GIL -> (release) ->
  // allocator_mutex. This prevents deadlock with Python threads that hold GIL
  // before calling allocate().
  //
  // Exception safety: the caller holds a unique_lock whose destructor will
  // handle the mutex correctly if this callback throws, so no catch-to-relock
  // pattern is needed here.

  DEBUGINFO(
      "SpyreAllocator: memory pressure callback invoked, releasing allocator "
      "mutex");

  // Step 1: Release allocator mutex
  lock.unlock();

  // Step 2: Acquire Python GIL
  // PyGILState_Ensure() is safe to call from any thread, even if the thread
  // was not created by Python. It returns the previous GIL state.
  DEBUGINFO("SpyreAllocator: acquiring Python GIL for garbage collection");
  PyGILState_STATE gstate = PyGILState_Ensure();

  // Step 3: Trigger Python garbage collection
  // PyGC_Collect() runs a full collection cycle and returns the number of
  // unreachable objects found (or -1 on error)
  DEBUGINFO("SpyreAllocator: calling PyGC_Collect()");
  Py_ssize_t collected = PyGC_Collect();

  if (collected >= 0) {
    DEBUGINFO("SpyreAllocator: PyGC_Collect() completed, collected ", collected,
              " objects");
  } else {
    DEBUGINFO("SpyreAllocator: PyGC_Collect() returned error");
  }

  // Step 4: Release Python GIL
  DEBUGINFO("SpyreAllocator: releasing Python GIL");
  PyGILState_Release(gstate);

  // Step 5: Re-acquire allocator mutex before returning to FlexAllocator
  DEBUGINFO("SpyreAllocator: re-acquiring allocator mutex");
  lock.lock();
  DEBUGINFO("SpyreAllocator: memory pressure callback complete");
}

// Register our custom allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &SpyreAllocator::instance());

}  // namespace spyre
