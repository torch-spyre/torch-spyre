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

#include "spyre_stream.h"

#include <ATen/record_function.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "job_plan.h"
#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_composite_address.h"
#include "spyre_error.h"
#include "spyre_guard.h"
#include "spyre_mem.h"
#include "spyre_pinned_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {
namespace {

// Bound the device descriptor to the whole sticks covering the view, then
// rebase DCI onto that staging buffer. When Flex's transfer spans this range
// from offset zero, it can reuse the DCI instead of constructing another
// compact conversion. Flex can still compact strided selections within the
// bounded descriptor. For a final-token logits view, the range is one row
// even when the tensor owns a complete prefill allocation.
std::optional<flex::CompositeAddress> narrowD2HRange(
    DataConversionInfo& dci, const SpyreTensorLayout& layout,
    const flex::CompositeAddress& allocation) {
  if (!allocation.is_single_chunk() ||
      layout.element_arrangement != ElementArrangement::STANDARD ||
      (layout.device_dtype != DataFormats::SEN169_FP16 &&
       layout.device_dtype != DataFormats::IEEE_FP32 &&
       layout.device_dtype != DataFormats::IEEE_INT32) ||
      dci.dcsi_.empty() || dci.input_shape_.empty()) {
    return std::nullopt;
  }

  const int64_t stick_elements = layout.elems_per_stick();
  const int64_t element_bytes = 128 / stick_elements;
  const int64_t allocation_elements = allocation.total_size() / element_bytes;
  int64_t first = allocation_elements;
  int64_t last = 0;
  for (const auto& stride : dci.dcsi_) {
    int64_t end = stride.offset_src_;
    if (end < 0 || end >= allocation_elements) {
      return std::nullopt;
    }
    for (size_t i = 0; i < stride.size_.size(); ++i) {
      const int64_t count = stride.size_[i];
      const int64_t step = stride.stride_src_[i];
      // Check against the allocation before multiplying to avoid overflow.
      if (count <= 0 || step < 0 ||
          (step > 0 && count - 1 > (allocation_elements - 1 - end) / step)) {
        return std::nullopt;
      }
      end += (count - 1) * step;
    }
    first = std::min(first, stride.offset_src_);
    last = std::max(last, end + 1);
  }
  first = first / stick_elements * stick_elements;
  last = (last + stick_elements - 1) / stick_elements * stick_elements;
  if (last > allocation_elements || last <= first ||
      (first == 0 && last == allocation_elements)) {
    return std::nullopt;
  }

  // Single-chunk STANDARD storage is a dense sequence of 128-byte sticks.
  // get_device_stride_infos builds contiguous physical device strides, so DCI
  // offsets already include layout padding; no extra inter-stick/tile gaps
  // need to be added when converting physical element offsets to byte offsets.
  auto chunk = allocation.chunks().front();
  chunk.addr.offset += first * element_bytes;
  chunk.size = (last - first) * element_bytes;
  for (auto& stride : dci.dcsi_) {
    stride.offset_src_ -= first;
  }
  // Describe the flat staged range while preserving the original DCI rank.
  // Shapes are innermost-first: collapse the outer dimensions to 1 so the
  // product remains the staged element count. dcsi_ retains the view strides.
  auto& shape = dci.input_shape_;
  std::fill(shape.begin() + 1, shape.end(), 1);
  shape.front() = last - first;
  return flex::CompositeAddress(chunk);
}

// TODO(tmhoangt): torch-spyre manages the pool and mapping; flex runtime just
// creates/destroys individual streams when asked.

// One pooled flex stream backing a c10::StreamId. The c10 stream id is
// stable for the process lifetime; the underlying flex handle may be
// recycled (see recycleStreamIfDue) to bound per-op state that flex retains
// for the lifetime of a stream.
struct PooledStream {
  flex::RuntimeStream* handle = nullptr;
  // Created by torch-spyre via RuntimeContext::createStream, so it can be
  // destroyed on recycle. The flex default stream that initially backs
  // c10 stream id 0 is borrowed and never destroyed.
  bool owned = false;
  // Handed out through getDefaultStreamRuntimeHandle(): spyre-comms keeps
  // the raw handle for the library lifetime and relies on sharing this
  // stream for ordering, so it must never be recycled.
  bool pinned = false;
  flex::RuntimeStreamPriority priority = flex::RuntimeStreamPriority::NORMAL;
  // Ops submitted while this handle is current. Flex keeps per-op host state
  // for every DMA, and for every launch on a hazard-tracking stream, until
  // the stream is destroyed (torch-spyre#6569), so this count gates
  // recycling.
  std::atomic<uint64_t> submitted_ops{0};

  PooledStream() = default;
  PooledStream(const PooledStream&) = delete;
  PooledStream& operator=(const PooledStream&) = delete;
};

// Global stream pool (shared across all threads)
struct StreamPool {
  mutable std::shared_mutex mutex;

  // Per-device stream pools
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      low_priority_streams;
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      high_priority_streams;
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      host_compute_streams;

  // Round-robin indices
  std::unordered_map<c10::DeviceIndex, size_t> next_low_priority_idx;
  std::unordered_map<c10::DeviceIndex, size_t> next_high_priority_idx;
  std::unordered_map<c10::DeviceIndex, size_t> next_host_compute_idx;

  // Mapping from c10::StreamId to the pooled flex stream backing it.
  // NOTE: this assumes one Spyre device per process (PyTorch's model for
  // multi-device use is one process per device), so the map is keyed by stream
  // id only. Supporting multiple devices in a single process would require
  // keying by (c10::DeviceIndex, c10::StreamId) so each device has its own
  // handles.
  std::unordered_map<c10::StreamId, PooledStream> stream_handle_map;

  // Per-device initialization flags
  std::unordered_map<c10::DeviceIndex, std::once_flag> device_init_flags;

  // Set to true once initializeStreamPoolImpl has been called.
  bool initialized = false;
  // Records the device_index passed to initializeStreamPoolImpl (valid only
  // when initialized == true), used in error messages.
  c10::DeviceIndex initialized_device_index = -1;
};

StreamPool& getStreamPool() {
  static StreamPool pool;
  return pool;
}

thread_local std::unordered_map<c10::DeviceIndex, c10::StreamId>
    current_streams;

// Resolves the pooled flex stream for id and runs fn while holding the
// pool's shared lock. Holding the lock across the flex call keeps a
// concurrent recycle from destroying the handle between lookup and use; the
// lock is shared, so submissions on the same stream still proceed
// concurrently.
template <typename F>
auto withResolvedStream(c10::StreamId id, F&& fn)
    -> decltype(fn(std::declval<PooledStream&>())) {
  auto& pool = getStreamPool();
  std::shared_lock<std::shared_mutex> lock(pool.mutex);
  auto it = pool.stream_handle_map.find(id);
  TORCH_CHECK(it != pool.stream_handle_map.end(),
              "SpyreStream: no flex handle for stream id ", id,
              " — was the stream pool initialized for this device?");
  return fn(it->second);
}

// Flex keeps per-op host state for a stream until the stream is destroyed
// (torch-spyre#6569), so host RSS grows linearly with the op count of any
// long-lived stream. Recycling swaps a fresh flex stream in behind the stable
// c10 stream id and destroys the old handle, bounding that retention at about
// TORCH_SPYRE_STREAM_RECYCLE_OPS submitted ops per stream; 0 disables it.
//
// A stream is recycled at the first of:
//   * a synchronization of it (stream or device): it is idle anyway;
//   * a submission on any stream while every pooled stream is idle, probed at
//     most once per kIdleProbeIntervalNs, which covers streams nobody
//     synchronizes (S_prep under SPYRE_HAZARD_TRACKER, every stream on a
//     non-driver tensor-parallel rank);
//   * a submission once it has carried kForcedRecycleFactor x the threshold:
//     the whole pool is drained first, so retention stays bounded even when
//     the pool never goes idle.
// Swapping only with the whole pool idle means no queued op, including the
// cross-stream edges flex's hazard tracker inserts, can refer to a destroyed
// handle; it is the state a device synchronize leaves.
constexpr uint64_t kDefaultStreamRecycleOps = 65536;
constexpr uint64_t kForcedRecycleFactor = 2;
constexpr int64_t kIdleProbeIntervalNs = 1'000'000;

// Re-read from the environment at every synchronization so tests can change
// it at runtime; the submission path only reads this cached copy.
std::atomic<uint64_t> g_recycle_ops{kDefaultStreamRecycleOps};
// Some recyclable stream has reached the threshold / the forced limit.
std::atomic<bool> g_recycle_due{false};
std::atomic<bool> g_recycle_overdue{false};
std::atomic<int64_t> g_last_idle_probe_ns{0};

std::atomic<uint64_t> g_recycles_at_sync{0};
std::atomic<uint64_t> g_recycles_when_idle{0};
std::atomic<uint64_t> g_recycles_forced{0};

uint64_t refreshStreamRecycleOps() {
  const char* env = std::getenv("TORCH_SPYRE_STREAM_RECYCLE_OPS");
  uint64_t threshold = kDefaultStreamRecycleOps;
  if (env != nullptr && *env != '\0') {
    char* end = nullptr;
    const int64_t parsed = std::strtoll(env, &end, 10);
    TORCH_CHECK(end != nullptr && *end == '\0' && parsed >= 0,
                "TORCH_SPYRE_STREAM_RECYCLE_OPS must be a non-negative "
                "integer, got '",
                env, "'");
    threshold = static_cast<uint64_t>(parsed);
  }
  g_recycle_ops.store(threshold, std::memory_order_relaxed);
  return threshold;
}

bool isRecycleDue(const PooledStream& entry, uint64_t threshold) {
  return threshold != 0 && !entry.pinned &&
         entry.submitted_ops.load(std::memory_order_relaxed) >= threshold;
}

// Swaps a fresh flex stream in for an idle entry. The caller holds the pool's
// exclusive lock, so no submission can start until the swap is visible.
// Returns the old handle when it must be destroyed once the lock is
// released; the borrowed flex default stream is never destroyed and stays
// idle for the rest of the process.
flex::RuntimeStream* swapIdleStreamLocked(PooledStream& entry) {
  flex::RuntimeStream* fresh = GlobalRuntime::get()->createStream(
      entry.priority, flex::RuntimeStreamMode::STRICT_ORDERING,
      /*track_hazards=*/get_hazard_tracker_enabled());
  flex::RuntimeStream* old = entry.handle;
  const bool destroy_old = entry.owned;
  entry.handle = fresh;
  entry.owned = true;
  entry.submitted_ops.store(0, std::memory_order_relaxed);
  return destroy_old ? old : nullptr;
}

// Re-derives the submission-path hints from the pool. Caller holds the lock.
void refreshRecycleHintsLocked(const StreamPool& pool, uint64_t threshold) {
  bool due = false;
  bool overdue = false;
  for (const auto& [id, entry] : pool.stream_handle_map) {
    if (isRecycleDue(entry, threshold)) {
      due = true;
      overdue =
          overdue || entry.submitted_ops.load(std::memory_order_relaxed) >=
                         kForcedRecycleFactor * threshold;
    }
  }
  g_recycle_due.store(due, std::memory_order_relaxed);
  g_recycle_overdue.store(overdue, std::memory_order_relaxed);
}

// Recycles stream id from a synchronization point. The old handle is drained
// again under the exclusive lock, so this stays correct when another thread
// submitted after the caller's synchronize.
void recycleStreamIfDue(c10::StreamId id) {
  const uint64_t threshold = refreshStreamRecycleOps();
  if (threshold == 0) {
    return;
  }
  auto& pool = getStreamPool();
  {
    std::shared_lock<std::shared_mutex> lock(pool.mutex);
    auto it = pool.stream_handle_map.find(id);
    if (it == pool.stream_handle_map.end() ||
        !isRecycleDue(it->second, threshold)) {
      return;
    }
  }
  flex::RuntimeStream* retired = nullptr;
  {
    std::unique_lock<std::shared_mutex> lock(pool.mutex);
    auto it = pool.stream_handle_map.find(id);
    if (it == pool.stream_handle_map.end() ||
        !isRecycleDue(it->second, threshold)) {
      return;
    }
    it->second.handle->synchronize();
    retired = swapIdleStreamLocked(it->second);
    refreshRecycleHintsLocked(pool, threshold);
  }
  g_recycles_at_sync.fetch_add(1, std::memory_order_relaxed);
  SPYRE_RUNTIME_DEBUG() << "recycled flex stream backing c10 stream " << id
                        << " at a synchronization";
  if (retired != nullptr) {
    GlobalRuntime::get()->destroyStream(retired);
  }
}

// Recycles every due stream while the whole pool is idle. Without force it
// gives up if another thread holds the pool lock or any stream is busy; with
// force it drains the pool first.
void recycleDueStreams(bool force) {
  const uint64_t threshold = g_recycle_ops.load(std::memory_order_relaxed);
  auto& pool = getStreamPool();
  std::vector<flex::RuntimeStream*> retired;
  uint64_t recycled = 0;
  {
    std::unique_lock<std::shared_mutex> lock(pool.mutex, std::defer_lock);
    if (force) {
      lock.lock();
    } else if (!lock.try_lock()) {
      return;
    }
    bool idle = true;
    for (auto& [id, entry] : pool.stream_handle_map) {
      if (force) {
        entry.handle->synchronize();
      } else if (!entry.handle->query()) {
        idle = false;
        break;
      }
    }
    if (idle) {
      for (auto& [id, entry] : pool.stream_handle_map) {
        if (isRecycleDue(entry, threshold)) {
          if (flex::RuntimeStream* old = swapIdleStreamLocked(entry)) {
            retired.push_back(old);
          }
          ++recycled;
        }
      }
    }
    refreshRecycleHintsLocked(pool, threshold);
  }
  if (recycled == 0) {
    return;
  }
  (force ? g_recycles_forced : g_recycles_when_idle)
      .fetch_add(recycled, std::memory_order_relaxed);
  SPYRE_RUNTIME_DEBUG() << "recycled " << recycled << " flex stream(s) "
                        << (force ? "after draining the pool"
                                  : "while the pool was idle");
  for (flex::RuntimeStream* old : retired) {
    GlobalRuntime::get()->destroyStream(old);
  }
}

// Called before every submission, with no pool lock held. Two relaxed loads
// unless some stream is due.
void maybeRecycleBeforeSubmit() {
  if (g_recycle_overdue.load(std::memory_order_relaxed)) {
    recycleDueStreams(/*force=*/true);
    return;
  }
  if (!g_recycle_due.load(std::memory_order_relaxed)) {
    return;
  }
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  int64_t last = g_last_idle_probe_ns.load(std::memory_order_relaxed);
  if (now - last < kIdleProbeIntervalNs ||
      !g_last_idle_probe_ns.compare_exchange_strong(
          last, now, std::memory_order_relaxed)) {
    return;
  }
  recycleDueStreams(/*force=*/false);
}

// Counts one submission on entry and raises the submission-path hints when it
// crosses the threshold or the forced limit. Caller holds the shared lock.
void countSubmission(PooledStream& entry) {
  const uint64_t n =
      entry.submitted_ops.fetch_add(1, std::memory_order_relaxed) + 1;
  const uint64_t threshold = g_recycle_ops.load(std::memory_order_relaxed);
  if (threshold == 0 || entry.pinned) {
    return;
  }
  if (n == threshold) {
    g_recycle_due.store(true, std::memory_order_relaxed);
  } else if (n == kForcedRecycleFactor * threshold) {
    g_recycle_overdue.store(true, std::memory_order_relaxed);
  }
}

}  // anonymous namespace

// Stream pool configuration
// Per device:
// - Stream 0: Default stream (always available, priority 0)
// - Streams 1-32: Low priority streams (priority 0)
// - Streams 33-64: High priority streams (priority -1)
// - Stream 65+: Host compute streams (priority -1)
constexpr int kStreamsPerDevice = 32;
constexpr int kHighPriorityStreamsPerDevice = 32;
constexpr int kHostComputeStreamStartPerDevice = 65;
constexpr int kDefaultHostComputeStreams = 4;
constexpr int kMaxHostComputeStreams = 8;

// Read from environment variable, defaulting to kDefaultHostComputeStreams when
// unset. An explicit value less than 1 (including malformed input, which parses
// to 0) is an error; values above kMaxHostComputeStreams are capped.
inline int getNumHostComputeStreams() {
  const char* env = std::getenv("TORCH_SPYRE_NUM_HOST_COMPUTE_STREAMS");
  int n = env ? std::atoi(env) : kDefaultHostComputeStreams;
  TORCH_CHECK(n >= 1,
              "TORCH_SPYRE_NUM_HOST_COMPUTE_STREAMS must be at least 1, got '",
              env, "'");
  if (n > kMaxHostComputeStreams) n = kMaxHostComputeStreams;
  return n;
}

// Constructor
SpyreStream::SpyreStream()
    : stream_(getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1,
                                           SpyreGuardImpl::tls_idx))
                  .unwrap()) {}
SpyreStream::SpyreStream(c10::Stream stream) : stream_(stream) {
  TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1,
              "SpyreStream requires PrivateUse1 device type, got ",
              stream_.device_type());
}

c10::StreamId SpyreStream::id() const {
  return stream_.id();
}

c10::Device SpyreStream::device() const {
  return stream_.device();
}

int SpyreStream::priority() const {
  // Determine priority from stream ID
  if (id() <= kStreamsPerDevice) {
    return 0;  // Low priority: stream 0 (default) and streams 1-32
  } else {
    return -1;
  }
}

bool SpyreStream::query() const {
  c10::DeviceGuard guard(stream_.device());

  SPYRE_RUNTIME_DEBUG() << "stream " << id() << " on device "
                        << static_cast<int>(device().index());

  return withResolvedStream(
      id(), [](PooledStream& entry) { return entry.handle->query(); });
}

void SpyreStream::synchronize() const {
  RECORD_FUNCTION("host::synchronize", {});
  c10::DeviceGuard device_guard(stream_.device());

  SPYRE_RUNTIME_DEBUG() << "stream " << id() << " on device "
                        << static_cast<int>(device().index());

  withResolvedStream(id(),
                     [](PooledStream& entry) { entry.handle->synchronize(); });
  // The stream is idle now, so a due recycle swaps the handle without
  // having to wait for in-flight work.
  recycleStreamIfDue(id());
}

c10::Stream SpyreStream::unwrap() const {
  return stream_;
}

void SpyreStream::copyProgramAsync(
    void* prog_cpu_ptr, const flex::CompositeAddress* device_address) const {
  // NOTE: the assumption is that the size of the program match the size of
  // device_address
  copyAsyncImpl(prog_cpu_ptr, /*host_capacity=*/0, device_address, nullptr,
                /*host2device=*/true);
}

void SpyreStream::copyAsync(const at::Tensor& src,
                            const at::Tensor& dst) const {
  SPYRE_RUNTIME_DEBUG() << "src (" << src.scalar_type()
                        << ") is on:" << src.device();
  SPYRE_RUNTIME_DEBUG() << "dst (" << dst.scalar_type()
                        << ") on:" << dst.device();

  // Determine copy direction
  bool host2device = src.is_cpu() && dst.is_privateuseone();
  bool device2host = src.is_privateuseone() && dst.is_cpu();

  const at::Tensor* dev_tensor = host2device ? &dst : &src;
  const at::Tensor* cpu_tensor = host2device ? &src : &dst;

  if (host2device || device2host) {
    // DCI offsets are relative to storage, so pass the storage base and its
    // complete accessible extent rather than the tensor's logical view.
    void* cpu_ptr = const_cast<void*>(cpu_tensor->storage().data());
    const size_t host_capacity = cpu_tensor->storage().nbytes();

    // Get SpyreTensorLayout using the public API
    SpyreTensorLayout stl = get_spyre_tensor_layout(*dev_tensor);

    DataConversionInfo dci =
        generate_dci(cpu_tensor, dev_tensor, stl, host2device);

    // A DCI that describes a dense same-format copy is a no-op conversion.
    // Dropping it skips flex's host-side staging (deeptools::ConvertData runs
    // a per-element strided loop even for identical formats) and, on H2D,
    // keeps a pinned source buffer attached -- attachStagedSource discards it
    // when a conversion stages into a fresh pageable buffer.
    //
    // The skip is only safe when the copied region is the whole device
    // allocation: the raw DMA is sized from the allocation's total_size(),
    // so a dense leading slice (e.g. d[:64].cpu()) would otherwise transfer
    // past the end of the smaller host buffer. Comparing the host storage
    // size is not sufficient either -- a same-sized host buffer backing a
    // slice view (y[:4096].copy_(d[:4096]) on an 8192-element y) would let
    // the whole-allocation DMA overwrite the rest of y. Require the copied
    // region extents themselves to match the allocation on both sides.
    // Slices keep their DCI and go through the conversion path, which
    // carries the region extents.
    //
    // The host-side and device-side extent checks are currently equivalent:
    // spyre_copy_ rejects mismatched region sizes up front
    // (TORCH_CHECK(cpu_sizes == dev_sizes)), and dci_is_identity_copy
    // requires matching data formats (hence equal element sizes), so the
    // two products are always equal here. Both halves are kept anyway as
    // defense-in-depth, in case the size precondition is ever relaxed or
    // copyAsync gains a caller that bypasses spyre_copy_'s check.
    //
    // On big-endian hosts the device is little-endian, and a raw DMA skips
    // the byte-order conversion that deeptools::ConvertData performs, so
    // multi-byte elements would arrive byte-swapped. Single-byte elements
    // have no byte order, so they may still take the raw-DMA fast path.
    // dci_is_identity_copy requires matching data formats, so the CPU and
    // device element sizes are equal here.
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    const bool byte_order_safe = cpu_tensor->element_size() == 1;
#else
    const bool byte_order_safe = true;
#endif
    const auto* allocation = get_composite_address(*dev_tensor);
    const size_t dev_total_size = static_cast<size_t>(allocation->total_size());
    const DataConversionInfo* dci_arg =
        (byte_order_safe && dci_is_identity_copy(dci) &&
         static_cast<size_t>(cpu_tensor->numel() *
                             cpu_tensor->element_size()) == dev_total_size &&
         static_cast<size_t>(dev_tensor->numel() *
                             dev_tensor->element_size()) == dev_total_size)
            ? nullptr
            : &dci;

    // The narrowed-range path is a conversion-path optimization: it rebases
    // the DCI onto the sticks covering the view. It has nothing to offer a
    // stripped identity copy (the region is the whole allocation by then),
    // so only take it when the DCI is kept.
    if (device2host && dci_arg != nullptr) {
      if (auto range = narrowD2HRange(dci, stl, *allocation)) {
        // Flex clones the descriptor when enqueuing; the tensor continues to
        // own the underlying allocation, just as in the full-storage path.
        copyAsyncImpl(cpu_ptr, host_capacity, &*range, &dci, host2device,
                      std::make_shared<c10::Storage>(cpu_tensor->storage()));
        return;
      }
    }

    // Hold the host storage alive across the DMA when flex will touch the
    // user pages after launchOperation returns: that is every D2H (the raw
    // DMA or the conversion's write-back runs on the scheduler thread) and
    // any DCI-less H2D (with the DCI stripped there is no caller-thread
    // staging copy, so the scheduler thread reads the source pages
    // directly). Hold c10::Storage rather than at::Tensor: a tensor copy
    // shares the TensorImpl, which resize_/set_ mutate in place by swapping
    // in a new storage -- the old pages would be freed under the in-flight
    // DMA. The pinned branches in copyAsyncImpl replace this token with the
    // pinned-buffer owner.
    std::shared_ptr<void> host_lifetime;
    if (device2host || dci_arg == nullptr) {
      host_lifetime = std::make_shared<c10::Storage>(cpu_tensor->storage());
    }
    copyAsyncImpl(cpu_ptr, host_capacity, allocation, dci_arg, host2device,
                  std::move(host_lifetime));

  } else {
    TORCH_CHECK(false, "Unsupported copy types: src on ", src.device(),
                " dst on ", dst.device());
  }
}

SpyreStreamError SpyreStream::getError() const {
  return withResolvedStream(id(), [](PooledStream& entry) {
    return entry.handle->needsShutdown() ? SpyreStreamError::Shutdown
                                         : SpyreStreamError::Success;
  });
}

void SpyreStream::copyAsyncImpl(void* cpu_ptr, size_t host_capacity,
                                const flex::CompositeAddress* device_address,
                                const DataConversionInfo* dci, bool host2device,
                                std::shared_ptr<void> host_lifetime) const {
  // Wrap dci in shared_ptr for flex API
  auto dci_ptr = dci ? std::make_shared<data_conversion_info>(*dci) : nullptr;

  // A pinned CPU source/destination skips flex's IOMMU shadow-buffer path:
  // the pre-mapped pinned buffer is submitted directly. The allocation state
  // rides along as host_lifetime so the buffer outlives in-flight DMA even if
  // the caller drops the tensor right after a non-blocking copy.
  auto* allocator =
      static_cast<SpyrePinnedAllocator*>(GetSpyrePinnedAllocator());
  const auto pinned_allocation = allocator->retain(cpu_ptr);

  // Create and launch operation through SpyreStream's typed launch methods.
  if (host2device) {
    flex::DmaParams* params = nullptr;
    if (pinned_allocation) {
      TORCH_CHECK(host_capacity <= pinned_allocation.capacity,
                  "H2D host storage exceeds pinned allocation capacity: ",
                  host_capacity, " vs ", pinned_allocation.capacity, " bytes");
      host_lifetime = std::static_pointer_cast<void>(pinned_allocation.owner);
      params = flex::createDmaParamsWithPinnedBuffer(
          pinned_allocation.buffer, pinned_allocation.offset, host_capacity,
          device_address->total_size(), host2device, device_address,
          std::move(dci_ptr),
          /*use_compute_pipeline=*/false,
          /*pipeline_barrier=*/false,
          /*skip_hazard=*/false, std::move(host_lifetime));
    } else if (host_capacity > 0) {
      params = flex::createDmaParamsWithHostCapacity(
          cpu_ptr, device_address->total_size(), host2device, device_address,
          host_capacity, std::move(dci_ptr),
          /*iova=*/nullptr,
          /*use_compute_pipeline=*/false,
          /*pipeline_barrier=*/false,
          /*skip_hazard=*/false, std::move(host_lifetime));
    } else {
      // host_capacity == 0 means the caller cannot state the accessible
      // extent (program uploads assume the buffer matches the device
      // allocation); skip the bounds-checked factory, as before.
      params = flex::createDmaParams(cpu_ptr, device_address->total_size(),
                                     host2device, device_address,
                                     std::move(dci_ptr));
    }
    try {
      launchH2D(params);
    }
    catch (...) {
      flex::destroyDmaParams(params);
      throw;
    }
    flex::destroyDmaParams(params);
  } else {
    flex::DmaParams* params = nullptr;
    if (pinned_allocation) {
      TORCH_CHECK(host_capacity <= pinned_allocation.capacity,
                  "D2H host storage exceeds pinned allocation capacity: ",
                  host_capacity, " vs ", pinned_allocation.capacity, " bytes");
      host_lifetime = std::static_pointer_cast<void>(pinned_allocation.owner);
      params = flex::createDmaParamsWithPinnedBuffer(
          pinned_allocation.buffer, pinned_allocation.offset, host_capacity,
          device_address->total_size(), host2device, device_address,
          std::move(dci_ptr),
          /*use_compute_pipeline=*/false,
          /*pipeline_barrier=*/false,
          /*skip_hazard=*/false, std::move(host_lifetime));
    } else {
      params = flex::createDmaParamsWithHostCapacity(
          cpu_ptr, device_address->total_size(), host2device, device_address,
          host_capacity, std::move(dci_ptr),
          /*iova=*/nullptr,
          /*use_compute_pipeline=*/false,
          /*pipeline_barrier=*/false,
          /*skip_hazard=*/false, std::move(host_lifetime));
    }
    try {
      launchD2H(params);
    }
    catch (...) {
      flex::destroyDmaParams(params);
      throw;
    }
    flex::destroyDmaParams(params);
  }
}

namespace {

// Submits one operation on the resolved flex stream under the pool's shared
// lock and counts it toward the recycle threshold. A due recycle runs first,
// with no lock held, so the operation lands on the fresh stream.
template <typename Params>
decltype(auto) submitCounted(c10::StreamId id, Params* params) {
  maybeRecycleBeforeSubmit();
  return withResolvedStream(id, [&](PooledStream& entry) -> decltype(auto) {
    countSubmission(entry);
    return entry.handle->launchOperation(params);
  });
}

}  // namespace

void SpyreStream::launchH2D(flex::DmaParams* params) const {
  TORCH_DCHECK_EQ(static_cast<int>(params->getType()),
                  static_cast<int>(flex::RuntimeOperationType::H2D))
      << "launchH2D called with params tagged as D2H — check createDmaParams' "
         "to_device argument";
  RECORD_FUNCTION("launch::H2D", {});
  submitCounted(id(), params);
}

void SpyreStream::launchD2H(flex::DmaParams* params) const {
  TORCH_DCHECK_EQ(static_cast<int>(params->getType()),
                  static_cast<int>(flex::RuntimeOperationType::D2H))
      << "launchD2H called with params tagged as H2D — check createDmaParams' "
         "to_device argument";
  RECORD_FUNCTION("launch::D2H", {});
  submitCounted(id(), params);
}

void SpyreStream::launchCompute(flex::ComputeParams* params) const {
  RECORD_FUNCTION("launch::Compute", {});
  submitCounted(id(), params);
}

void SpyreStream::launchHostCallback(flex::HostCallbackParams* params) const {
  RECORD_FUNCTION("launch::HostCallback", {});
  submitCounted(id(), params);
}

void SpyreStream::launchFillAsync(flex::FillAsyncParams* params) const {
  RECORD_FUNCTION("launch::Memset", {});
  submitCounted(id(), params);
}

void SpyreStream::launchCopyRaw(flex::CopyRawParams* params) const {
  RECORD_FUNCTION("launch::CopyRaw", {});
  submitCounted(id(), params);
}

flex::HostComputeBuffer* SpyreStream::launchHostCompute(
    flex::HostComputeParams* params) const {
  RECORD_FUNCTION("launch::HostCompute", {});
  return submitCounted(id(), params);
}

void SpyreStream::launch(const JobPlan& plan,
                         const std::vector<at::Tensor>& args,
                         std::vector<SymbolicArg> symbolic_args) const {
  // Validate all tensors are on Spyre device
  for (size_t i = 0; i < args.size(); ++i) {
    TORCH_CHECK(args[i].is_privateuseone(), "SpyreStream::launch: argument ", i,
                " must be on Spyre device, got ", args[i].device());
  }

  // Two-stream overlap topology:
  //   S_dev  = this stream (the default) — Compute (+ D2H).
  //   S_prep = the persistent host-compute stream — HostCompute + separate H2D.
  // Batched correction uploads instead travel with their compute on S_dev.
  // HostCompute runs inline on the caller; only uploads and compute are queued.
  // S_dev compute can overlap preparation for subsequent launches. Every op
  // keeps pipeline_barrier=true (per-stream FIFO). S_prep must be the same
  // persistent flex handle each launch: getHostComputeStreamById is a pure
  // lookup of the handle registered once in initializeStreamPoolImpl.
  const SpyreStream& s_dev = *this;
  const SpyreStream s_prep =
      getHostComputeStreamById(kHostComputeStreamStartPerDevice, device());

  // Create launch context with tensor arguments and typed symbolic payload.
  // symbolic_args is moved in so the closure in
  // JobPlanStepHostCompute::construct can capture it by value without an extra
  // copy.
  LaunchContext ctx{args, std::move(symbolic_args)};
  const size_t max_corrections =
      withResolvedStream(s_dev.id(), [](PooledStream& entry) {
        return entry.handle->maxProgramCorrections();
      });

  // Split Prep-role steps onto S_prep only when the flex tracker is on; flex
  // then inserts the cross-stream edges. Off = every step on S_dev (the
  // single-stream floor). Routing keys on role(), so all-Dev plans never split.
  // This switch controls stream splitting only: eligible correction batching is
  // enabled in both topologies, including the single-stream path.
  const bool should_split = get_hazard_tracker_enabled();
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    const auto& step = plan.steps[i];
    const auto kind = classifyStep(*step);
    // Group by operation kind: Prep also includes standalone H2D steps, which
    // must not be collected as host corrections. Pin the role assumptions used
    // by the routing below so new step variants cannot silently change them.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(kind != StepKind::HostCompute ||
                                     step->role() == StreamRole::Prep);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(kind != StepKind::Compute ||
                                     step->role() == StreamRole::Dev);
    if (max_corrections > 0 && kind == StepKind::HostCompute &&
        (i == 0 || classifyStep(*plan.steps[i - 1]) != StepKind::HostCompute)) {
      size_t end = i + 1;
      while (end < plan.steps.size() &&
             classifyStep(*plan.steps[end]) == StepKind::HostCompute) {
        ++end;
      }
      // Flex reports the supported correction count for the consuming stream.
      // Keep the whole group on separate uploads when it cannot be batched.
      ctx.batch_program_corrections =
          end - i <= max_corrections && end < plan.steps.size() &&
          classifyStep(*plan.steps[end]) == StepKind::Compute;
    }
    const SpyreStream& target =
        (should_split && step->role() == StreamRole::Prep) ? s_prep : s_dev;
    step->construct(ctx, target);
  }
  TORCH_CHECK(ctx.program_corrections.empty(),
              "Program correction has no consuming compute step");
}

void initializeStreamPoolImpl(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);

  // Check that this is the first and only device initialization
  TORCH_CHECK(!pool.initialized,
              "initializeStreamPoolImpl already called with device_index ",
              static_cast<int>(pool.initialized_device_index),
              "; cannot reinitialize with device_index ",
              static_cast<int>(device_index));
  pool.initialized = true;
  pool.initialized_device_index = device_index;
  refreshStreamRecycleOps();

  // Initialize mapping from StreamId → pooled flex stream.
  // RuntimeStream instances are owned by the RuntimeContext; the pool tracks
  // which handles it created (recyclable) versus borrowed (the flex default
  // stream, which destroyStream rejects).
  auto runtime = GlobalRuntime::get();

  // Register default stream (ID 0). Backed by flex's default stream until
  // the first recycle swaps in an owned handle.
  pool.stream_handle_map.try_emplace(0).first->second.handle =
      runtime->getDefaultStream();

  // Register host compute streams (IDs 65+)
  int num_host_streams = getNumHostComputeStreams();
  pool.host_compute_streams[device_index].reserve(num_host_streams);
  for (int i = 0; i < num_host_streams; ++i) {
    c10::StreamId sid = kHostComputeStreamStartPerDevice + i;
    // Create the flex handle only if this stream id has not been registered
    // yet. stream_handle_map is keyed by stream id, not device, so when the
    // pool is initialized under more than one device index in the same process
    // (all mapping to the same runtime) the handle is created once and reused.
    TORCH_CHECK(
        pool.stream_handle_map.find(sid) == pool.stream_handle_map.end(),
        "Host compute stream id ", sid,
        " is already registered; only one Spyre device per process is "
        "supported.");
    PooledStream& entry = pool.stream_handle_map.try_emplace(sid).first->second;
    entry.handle =
        runtime->createStream(flex::RuntimeStreamPriority::NORMAL,
                              flex::RuntimeStreamMode::STRICT_ORDERING,
                              /*track_hazards=*/get_hazard_tracker_enabled());
    entry.owned = true;
    entry.priority = flex::RuntimeStreamPriority::NORMAL;
    pool.host_compute_streams[device_index].push_back(sid);
  }
  pool.next_host_compute_idx[device_index] = 0;

  // Initialize low priority streams (IDs 1 to kStreamsPerDevice)
  pool.low_priority_streams[device_index].reserve(kStreamsPerDevice);
  for (int i = 1; i <= kStreamsPerDevice; ++i) {
    pool.low_priority_streams[device_index].push_back(i);
  }
  pool.next_low_priority_idx[device_index] = 0;

  pool.high_priority_streams[device_index].reserve(
      kHighPriorityStreamsPerDevice);
  for (int i = 1; i <= kHighPriorityStreamsPerDevice; ++i) {
    pool.high_priority_streams[device_index].push_back(kStreamsPerDevice + i);
  }
  pool.next_high_priority_idx[device_index] = 0;
}

void initializeStreamPool(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::call_once(pool.device_init_flags[device_index], initializeStreamPoolImpl,
                 device_index);
}

SpyreStream getDefaultStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }
  initializeStreamPool(device.index());
  return SpyreStream(c10::Stream(c10::Stream::DEFAULT, device));
}

flex::RuntimeStream* getDefaultStreamRuntimeHandle(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }
  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);
  auto it = pool.stream_handle_map.find(0);
  TORCH_CHECK(it != pool.stream_handle_map.end(),
              "Default stream handle not initialized for device ",
              device.index());
  // The caller (spyre-comms) holds this raw handle for the lifetime of its
  // library and relies on sharing torch-spyre's default stream for ordering,
  // so the handle must never be recycled.
  it->second.pinned = true;
  return it->second.handle;
}

SpyreStream getCurrentStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  auto it = current_streams.find(device.index());
  if (it == current_streams.end()) {
    return getDefaultStream(device);
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, it->second));
}

SpyreStream setCurrentStream(SpyreStream stream) {
  auto device = stream.device();
  auto old_stream = getCurrentStream(device);
  current_streams[device.index()] = stream.id();
  return old_stream;
}

SpyreStream getHostComputeStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when this is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);

  auto& streams = pool.host_compute_streams[device.index()];
  auto& idx = pool.next_host_compute_idx[device.index()];

  c10::StreamId stream_id = streams[idx];
  idx = (idx + 1) % streams.size();

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

SpyreStream getHostComputeStreamById(c10::StreamId id, c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when this is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  const c10::StreamId end =
      kHostComputeStreamStartPerDevice + getNumHostComputeStreams();
  TORCH_CHECK(id >= kHostComputeStreamStartPerDevice && id < end,
              "getHostComputeStreamById: stream id ", id,
              " is not a host compute stream id (valid range [",
              kHostComputeStreamStartPerDevice, ", ", end, "))");

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, id));
}

SpyreStream getStreamFromPool(c10::Device device, int priority) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when torch.Stream() is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);

  c10::StreamId stream_id;
  if (priority == 0) {
    // Low priority stream - round-robin from low priority pool
    auto& streams = pool.low_priority_streams[device.index()];
    auto& idx = pool.next_low_priority_idx[device.index()];

    stream_id = streams[idx];
    idx = (idx + 1) % streams.size();

  } else {
    // High priority stream - round-robin from high priority pool
    auto& streams = pool.high_priority_streams[device.index()];
    auto& idx = pool.next_high_priority_idx[device.index()];

    stream_id = streams[idx];
    idx = (idx + 1) % streams.size();
  }

  // Create corresponding flex stream handle (if not exists)
  if (pool.stream_handle_map.find(stream_id) == pool.stream_handle_map.end()) {
    auto runtime = GlobalRuntime::get();
    flex::RuntimeStreamPriority streamPriority =
        priority < 0 ? flex::RuntimeStreamPriority::HIGH
                     : flex::RuntimeStreamPriority::NORMAL;
    PooledStream& entry =
        pool.stream_handle_map.try_emplace(stream_id).first->second;
    entry.handle = runtime->createStream(
        streamPriority, flex::RuntimeStreamMode::STRICT_ORDERING,
        /*track_hazards=*/get_hazard_tracker_enabled());
    entry.owned = true;
    entry.priority = streamPriority;
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

StreamRecycleStats getStreamRecycleStats() {
  StreamRecycleStats stats;
  stats.at_sync = g_recycles_at_sync.load(std::memory_order_relaxed);
  stats.when_idle = g_recycles_when_idle.load(std::memory_order_relaxed);
  stats.forced = g_recycles_forced.load(std::memory_order_relaxed);
  return stats;
}

void synchronizeDevice(c10::optional<c10::Device> device) {
  auto sync_one_device = [](c10::Device dev) {
    if (dev.index() == -1) {
      dev = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
    }
    const auto device_index = dev.index();
    c10::DeviceGuard guard(dev);

    std::vector<c10::StreamId> synced_ids;
    {
      auto& pool = getStreamPool();
      // Synchronize under the pool's shared lock so a concurrent recycle
      // cannot destroy a handle between lookup and use.
      std::shared_lock<std::shared_mutex> lock(pool.mutex);

      // Default stream (ID 0) is always present when the pool is initialized
      auto default_it = pool.stream_handle_map.find(0);
      if (default_it != pool.stream_handle_map.end()) {
        default_it->second.handle->synchronize();
        synced_ids.push_back(0);
      }

      auto collect = [&](auto& stream_map) {
        auto it = stream_map.find(device_index);
        if (it == stream_map.end()) return;
        for (auto sid : it->second) {
          auto h = pool.stream_handle_map.find(sid);
          if (h != pool.stream_handle_map.end()) {
            h->second.handle->synchronize();
            synced_ids.push_back(sid);
          }
        }
      };
      collect(pool.low_priority_streams);
      collect(pool.high_priority_streams);
      collect(pool.host_compute_streams);
    }  // lock released

    for (auto sid : synced_ids) {
      recycleStreamIfDue(sid);
    }
  };
  if (device.has_value()) {
    sync_one_device(device.value());
  } else {
    sync_one_device(
        c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx));
  }
}

const char* SpyreStreamGetErrorString(SpyreStreamError error) noexcept {
  switch (error) {
    case SpyreStreamError::Success:
      return "Success";
    case SpyreStreamError::Shutdown:
      return "Shutdown";
    default:
      return "Unknown";
  }
}

SpyreStreamError SpyreStreamGetError(const SpyreStream& stream) {
  return stream.getError();
}

SpyreDeviceState SpyreGetDeviceState() {
  auto runtime = GlobalRuntime::get();
  if (!runtime) {
    return SpyreDeviceState::NotInitialized;
  }
  // NOTE: intentionally uses RuntimeContext::hasStreamError() (one locked
  // iteration) instead of rolling up SpyreStreamGetError per stream. These
  // agree only while both reduce to needsShutdown().
  // TODO(#3365): once flex exposes typed per-stream codes, roll up via
  // SpyreStreamGetError so the aggregate (and the pytest skip reason) can
  // surface the actual fault class.
  if (runtime->hasStreamError()) {
    return SpyreDeviceState::StreamError;
  }
  return SpyreDeviceState::Ok;
}

}  // namespace spyre
