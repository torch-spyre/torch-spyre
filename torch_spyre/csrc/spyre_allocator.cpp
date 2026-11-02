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

#include <atomic>
#include <cstdlib>  // std::getenv
#include <memory>
#include <mutex>
#include <numeric>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_composite_address.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {

namespace {

// Env gate that makes every *tensor* storage allocation interleave across all
// memory domains flex reports, e.g. under flex's emulated 4-domain topology:
//
//   FLEX_DEVICE=EMUL1p5 TORCH_SPYRE_EMULATE_INTERLEAVE=1
//
// It is not an eligibility policy: it interleaves every tensor storage.
//
// Scope:
//  - Only the c10::Allocator entry point (`allocate(nbytes)`) consults it, so
//    it affects tensor storages only. The program allocation uses the
//    directive-taking overload with its own Bind{0} (prepare_kernel.cpp) and is
//    untouched.
//  - Interleaved tensors are usable on the **eager copy path only**. The
//    compiled JobPlan builder requires single-chunk addresses, so do not set
//    this in a process that compiles kernels.
//  - Off (the default) leaves behavior byte-for-byte identical to Bind{0}.
//
// The env var seeds the initial value, but the flag is a process-global that
// `debugSetEmulateInterleave` can flip at runtime. That matters: only one
// process may hold the Spyre device, so a test cannot compare an interleaved
// run against a Bind{0} baseline by spawning a second process — both
// placements have to be exercised in the same process, under the same topology.
std::atomic<bool>& emulateInterleaveFlag() {
  static std::atomic<bool> flag{[]() {
    const char* env = std::getenv("TORCH_SPYRE_EMULATE_INTERLEAVE");
    return env != nullptr && std::string_view(env) != "" &&
           std::string_view(env) != "0";
  }()};
  return flag;
}

bool emulateInterleaveEnabled() {
  return emulateInterleaveFlag().load(std::memory_order_relaxed);
}

}  // namespace

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
  auto flex_alloc = flex::RuntimeContext::getInstance()->getAllocator();

  // Register memory pressure callback on first access (lazy initialization)
  static std::once_flag callback_registered;
  std::call_once(callback_registered, [&flex_alloc]() {
    if (flex_alloc) {
      flex_alloc->registerMemoryPressureCallback(
          &SpyreAllocator::memoryPressureCallback);
      SPYRE_RUNTIME_DEBUG()
          << "(getFlexAllocator) registered memory pressure callback with "
             "FlexAllocator";
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

std::pair<size_t, size_t> SpyreAllocator::getMemoryInfo(
    c10::DeviceIndex device) {
  int64_t active_device_id = flex::RuntimeContext::getInstance()->getDeviceID();
  TORCH_CHECK(static_cast<int64_t>(device) == active_device_id,
              "SpyreAllocator::getMemoryInfo: requested device index ", device,
              " does not match the active device (", active_device_id, ")");
  auto flex_alloc = getFlexAllocator();
  flex::MemoryStats mem_stats =
      flex_alloc->getMemoryStats(flex::MemoryType::Tensor);
  return {mem_stats.free_bytes, mem_stats.total_bytes};
}

void SpyreAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_accumulated();
        stats_.reserved_bytes[stat_type].reset_accumulated();
        stats_.allocation[stat_type].reset_accumulated();
      });
}

void SpyreAllocator::resetPeakStats(c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_peak();
        stats_.reserved_bytes[stat_type].reset_peak();
        stats_.allocation[stat_type].reset_peak();
      });
}

void SpyreAllocator::recordAlloc(size_t nbytes, void* data, int device_id,
                                 flex::MemoryType memory_type) {
  // User-visible accounting (stats and profiler) is Tensor-only.
  // Program and other non-Tensor allocations must not appear in user-visible
  // memory stats: on release the CompositeAddress has already been moved out,
  // so nbytes would be 0 and would leave an unmatched +N in the profiler.
  if (memory_type != flex::MemoryType::Tensor) {
    return;
  }
  int64_t total_allocated;
  int64_t total_reserved;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    c10::CachingAllocator::for_each_selected_stat_type(
        stat_types, [&](size_t stat_type) {
          stats_.allocation[stat_type].increase(1);
          stats_.allocated_bytes[stat_type].increase(nbytes);
          stats_.reserved_bytes[stat_type].increase(nbytes);
        });
    total_allocated = stats_
                          .allocated_bytes[static_cast<size_t>(
                              c10::CachingAllocator::StatType::AGGREGATE)]
                          .current;
    total_reserved = stats_
                         .reserved_bytes[static_cast<size_t>(
                             c10::CachingAllocator::StatType::AGGREGATE)]
                         .current;
  }
  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(data,
                                   nbytes,           // alloc_size
                                   total_allocated,  // total_allocated
                                   total_reserved,   // total_reserved
                                   curr_device);
}

void SpyreAllocator::recordRelease(size_t nbytes, void* data, int device_id,
                                   flex::MemoryType memory_type) {
  // User-visible accounting (stats and profiler) is Tensor-only.
  // Program and other non-Tensor allocations must not appear in user-visible
  // memory stats: on release the CompositeAddress has already been moved out,
  // so nbytes would be 0 and would leave an unmatched +N in the profiler.
  if (memory_type != flex::MemoryType::Tensor) {
    return;
  }
  int64_t total_allocated;
  int64_t total_reserved;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    c10::CachingAllocator::for_each_selected_stat_type(
        stat_types, [&](size_t stat_type) {
          stats_.allocation[stat_type].decrease(1);
          stats_.allocated_bytes[stat_type].decrease(nbytes);
          stats_.reserved_bytes[stat_type].decrease(nbytes);
        });
    total_allocated = stats_
                          .allocated_bytes[static_cast<size_t>(
                              c10::CachingAllocator::StatType::AGGREGATE)]
                          .current;
    total_reserved = stats_
                         .reserved_bytes[static_cast<size_t>(
                             c10::CachingAllocator::StatType::AGGREGATE)]
                         .current;
  }
  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(data,
                                   -static_cast<int64_t>(nbytes),  // alloc_size
                                   total_allocated,  // total_allocated
                                   total_reserved,   // total_reserved
                                   curr_device);
}

c10::DataPtr SpyreAllocator::allocate(size_t nbytes) {
  // See emulateInterleaveEnabled(). Skipped for nbytes == 0 so the empty
  // allocation keeps returning early below without touching the runtime.
  if (nbytes != 0 && emulateInterleaveEnabled()) {
    auto flex_alloc = getFlexAllocator();
    const size_t num_domains =
        flex_alloc ? flex_alloc->topology().num_domains() : 1;
    if (num_domains > 1) {
      std::vector<uint32_t> domain_ids(num_domains);
      std::iota(domain_ids.begin(), domain_ids.end(), 0U);
      flex::AllocationDirective interleaved(flex::PlacementPolicy::Interleave,
                                            std::move(domain_ids), std::nullopt,
                                            flex::MemoryType::Tensor);
      SPYRE_RUNTIME_DEBUG()
          << "TORCH_SPYRE_EMULATE_INTERLEAVE: interleaving " << nbytes
          << " (bytes) across " << num_domains << " memory domains";
      return SpyreAllocator::allocate(nbytes, interleaved);
    }
    // num_domains == 1: no domains to stripe across, fall through to Bind{0}.
  }

  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Tensor);
  return SpyreAllocator::allocate(nbytes, directive);
}

c10::DataPtr SpyreAllocator::allocate(
    size_t nbytes, const flex::AllocationDirective& directive) {
  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice();

  auto device_id = curr_device.index();

  SPYRE_RUNTIME_DEBUG() << "allocating " << nbytes << " (bytes) on Spyre"
                        << curr_device;
  if (nbytes == 0) {
    return {nullptr, nullptr, &ReportAndDelete, curr_device};
  }
  // Get shared_ptr to keep FlexAllocator alive during operations
  auto flex_alloc = getFlexAllocator();

  // Allocate first-class raw storage via CompositeAddress.
  flex::CompositeAddress composite_addr =
      flex_alloc->allocate(nbytes, directive);

  SPYRE_RUNTIME_DEBUG() << "allocated " << composite_addr;
  // FlexAllocator rounds up to DEVICE_ALIGNMENT (128 bytes), so the actual
  // allocation may be larger than the requested nbytes. Use total_size() for
  // accurate memory profiling.
  size_t actual_nbytes = composite_addr.total_size();

  auto* ctx = new SharedOwnerCtx(std::move(composite_addr), device_id,
                                 directive.memory_type);
  void* ctx_void = static_cast<void*>(ctx);

  // Use the SharedOwnerCtx pointer as the unique data handle for c10::DataPtr.
  // This pointer is never dereferenced — it serves only as a unique token for
  // memory profiling (recordAlloc/recordRelease).
  void* data_void = static_cast<void*>(ctx);
  recordAlloc(actual_nbytes, data_void, device_id, directive.memory_type);

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
                                           ctx->device_id, ctx->memory_type);
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
  SPYRE_RUNTIME_DEBUG() << "called";
  // do nothing -- look into when this is called
  // spyre_copy_from(reinterpret_cast<spyre_ptr_t>(dest),
  // reinterpret_cast<spyre_ptr_t>(src));
}

uint64_t SpyreAllocator::compositeAddressToDeviceAddress(
    const flex::CompositeAddress& addr) const {
  return flex::compositeAddressToDeviceAddress(
      addr, getFlexAllocator()->getIdToRegionMap());
}

// Flip the interleave gate for the rest of this process; returns the previous
// value so a caller can restore it. See emulateInterleaveFlag().
bool SpyreAllocator::debugSetEmulateInterleave(bool enabled) {
  return emulateInterleaveFlag().exchange(enabled, std::memory_order_relaxed);
}

// Number of memory domains flex reports (4 under FLEX_DEVICE=EMUL1p5, else 1),
// so a test can skip cleanly instead of asserting on a single-domain topology.
size_t SpyreAllocator::debugNumMemoryDomains() {
  auto flex_alloc = getFlexAllocator();
  return flex_alloc ? flex_alloc->topology().num_domains() : 0;
}

// Hardware segment of each chunk of a Spyre tensor's allocation, in
// get_composite_address(tensor).chunks() order; std::nullopt for a chunk whose
// region has no tensor segment (flex's UNMAPPED_SEGMENT_ID), which must never
// be dispatched.
//
// flex does not export MemoryRegion, so the segment is read back through the
// public addressing API instead: each chunk is encoded as a single-chunk
// device address and decoded again. A 1p0 DMVA has a 3-bit segment field, so
// a region beyond the tensor-segment budget (UNMAPPED_SEGMENT_ID) decodes to a
// segment id past it and reads as unmapped.
std::vector<std::optional<uint32_t>> SpyreAllocator::debugChunkSegmentIds(
    const at::Tensor& tensor) {
  const flex::CompositeAddress* addr = get_composite_address(tensor);
  auto flex_alloc = getFlexAllocator();
  const auto& region_map = flex_alloc->getIdToRegionMap();
  constexpr uint64_t kNumSegments = 8;

  std::vector<std::optional<uint32_t>> out;
  out.reserve(addr->chunks().size());
  for (const auto& chunk : addr->chunks()) {
    // Built from a bare Chunk, so it has no allocator and frees nothing.
    const flex::CompositeAddress single(chunk);
    const uint64_t device_address =
        flex::compositeAddressToDeviceAddress(single, region_map);
    const auto [segment, offset, type] =
        flex::decodeDevicePointer(device_address);
    // Segment 0 is never a tensor segment on 1p0; segment 7 decodes as Program.
    if (segment >= 1 && segment < kNumSegments &&
        type == flex::MemoryType::Tensor) {
      out.emplace_back(static_cast<uint32_t>(segment));
    } else {
      out.emplace_back(std::nullopt);
    }
  }
  return out;
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

  SPYRE_RUNTIME_DEBUG() << "releasing allocator mutex";

  // Step 1: Release allocator mutex
  lock.unlock();

  // Step 2: Acquire Python GIL
  // PyGILState_Ensure() is safe to call from any thread, even if the thread
  // was not created by Python. It returns the previous GIL state.
  SPYRE_RUNTIME_DEBUG() << "acquiring Python GIL for garbage collection";
  PyGILState_STATE gstate = PyGILState_Ensure();

  // Step 3: Trigger Python garbage collection
  // PyGC_Collect() runs a full collection cycle and returns the number of
  // unreachable objects found (or -1 on error)
  SPYRE_RUNTIME_DEBUG() << "calling PyGC_Collect()";
  Py_ssize_t collected = PyGC_Collect();

  if (collected >= 0) {
    SPYRE_RUNTIME_DEBUG() << "PyGC_Collect() completed, collected " << collected
                          << " objects";
  } else {
    SPYRE_RUNTIME_DEBUG() << "PyGC_Collect() returned error";
  }

  // Step 4: Release Python GIL
  SPYRE_RUNTIME_DEBUG() << "releasing Python GIL";
  PyGILState_Release(gstate);

  // Step 5: Re-acquire allocator mutex before returning to FlexAllocator
  SPYRE_RUNTIME_DEBUG() << "re-acquiring allocator mutex";
  lock.lock();
  SPYRE_RUNTIME_DEBUG() << "memory pressure callback complete";
}

// Register our custom allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &SpyreAllocator::instance());

}  // namespace spyre
