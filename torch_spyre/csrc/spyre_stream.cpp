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

#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "flex/runtime_stream/runtime_operation.hpp"
#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_guard.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {
namespace {

// TODO(tmhoangt): torch-spyre manages the pool and mapping; flex runtime just
// creates/destroys individual streams when asked.

// Global stream pool (shared across all threads)
struct StreamPool {
  std::mutex mutex;

  // Per-device stream pools
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      low_priority_streams;
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      high_priority_streams;

  // Round-robin indices
  std::unordered_map<c10::DeviceIndex, size_t> next_low_priority_idx;
  std::unordered_map<c10::DeviceIndex, size_t> next_high_priority_idx;

  // Mapping from c10::StreamId to flex::RuntimeStream*
  std::unordered_map<c10::StreamId, flex::RuntimeStream*> stream_handle_map;

  // Per-device initialization flags
  std::unordered_map<c10::DeviceIndex, std::once_flag> device_init_flags;
};

StreamPool& getStreamPool() {
  static StreamPool pool;
  return pool;
}

thread_local std::unordered_map<c10::DeviceIndex, c10::StreamId>
    current_streams;

}  // anonymous namespace

// Stream pool configuration
// Per device:
// - Stream 0: Default stream (always available, priority 0)
// - Streams 1-32: Low priority streams (priority 0)
// - Streams 33-64: High priority streams (priority -1)
constexpr int kStreamsPerDevice = 32;
constexpr int kHighPriorityStreamsPerDevice = 32;

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

  DEBUGINFO("SpyreStream::query() - stream ", id(), " on device ",
            device().index());

  flex::RuntimeStream* handle = getRuntimeHandle();
  return handle->query();
}

void SpyreStream::synchronize() const {
  c10::DeviceGuard guard(stream_.device());

  DEBUGINFO("SpyreStream::synchronize() - stream ", id(), " on device ",
            device().index());

  flex::RuntimeStream* handle = getRuntimeHandle();
  handle->synchronize();
}

c10::Stream SpyreStream::unwrap() const {
  return stream_;
}

void SpyreStream::copyAsync(const at::Tensor& src,
                            const at::Tensor& dst) const {
  DEBUGINFO("src (", src.scalar_type(), ") is on:", src.device());
  DEBUGINFO("dst (", dst.scalar_type(), ") on:", dst.device());

  // TODO(tmhoangt): add type conversion node
  // Type checking - no type conversion support yet
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "Spyre backend does not support type conversion yet during copy.");

  // Determine copy direction
  bool host2device = src.is_cpu() && dst.is_privateuseone();
  bool device2host = src.is_privateuseone() && dst.is_cpu();

  const at::Tensor* dev_tensor = host2device ? &dst : &src;
  const at::Tensor* cpu_tensor = host2device ? &src : &dst;

  if (host2device || device2host) {
    // Host-to-device or device-to-host copy
    void* cpu_ptr = const_cast<void*>(cpu_tensor->storage().data());

    // Get SpyreTensorLayout using the public API
    SpyreTensorLayout stl = get_spyre_tensor_layout(*dev_tensor);

    // Extract device allocation from Spyre tensor storage
    auto* spyre_impl =
        static_cast<SpyreTensorImpl*>(dev_tensor->unsafeGetTensorImpl());
    auto& storage = spyre_impl->storage();
    auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());

    // Generate data conversion info
    // So we always pass &src, not cpu_tensor
    DataConversionInfo dci = generate_dci(
        dev_tensor, stl, cpu_tensor->storage_offset(), host2device);

    copyAsyncImpl(cpu_ptr, &ctx->composite_addr, dci, host2device);

  } else {
    TORCH_CHECK(false, "Unsupported copy types: src on ", src.device(),
                " dst on ", dst.device());
  }
}

flex::RuntimeStream* SpyreStream::getRuntimeHandle() const {
  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

  auto it = pool.stream_handle_map.find(id());
  TORCH_CHECK(it != pool.stream_handle_map.end(),
              "SpyreStream: no flex handle for stream id ", id(),
              " — was the stream pool initialized for this device?");
  return it->second;
}

void SpyreStream::copyAsyncImpl(void* cpu_ptr,
                                const flex::CompositeAddress* device_address,
                                const DataConversionInfo& dci,
                                bool host2device) const {
  // Wrap dci in shared_ptr for flex API
  auto dci_ptr = std::make_shared<data_conversion_info>(dci);

  // Get the flex runtime stream handle
  flex::RuntimeStream* flex_stream = getRuntimeHandle();

  // Create and launch operation
  if (host2device) {
    flex::RuntimeOperationH2D op(cpu_ptr, device_address, dci_ptr);
    flex_stream->launchOperation(op);
  } else {
    flex::RuntimeOperationD2H op(device_address, cpu_ptr, dci_ptr);
    flex_stream->launchOperation(op);
  }
}

void initializeStreamPoolImpl(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

  // Register the default stream (ID 0) using the concrete flex handle.
  // This ensures getRuntimeHandle() resolves stream 0 to the real RuntimeStream
  // instance owned by RuntimeContext.
  auto runtime = GlobalRuntime::get();
  pool.stream_handle_map[0] = runtime->getDefaultStream();

  // Initialize low priority streams (IDs 1 to kStreamsPerDevice)
  pool.low_priority_streams[device_index].reserve(kStreamsPerDevice);
  for (int i = 1; i <= kStreamsPerDevice; ++i) {
    pool.low_priority_streams[device_index].push_back(i);
  }
  pool.next_low_priority_idx[device_index] = 0;

  // Initialize high priority streams
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

SpyreStream getStreamFromPool(c10::Device device, int priority) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when torch.Stream() is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

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
    flex::RuntimeStream* flex_handle =
        runtime->createStream(runtime->toPriority(priority));
    pool.stream_handle_map[stream_id] = flex_handle;
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

void synchronizeDevice(c10::optional<c10::Device> device) {
  auto sync_one_device = [](c10::Device dev) {
    if (dev.index() == -1) {
      dev = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
    }
    const auto device_index = dev.index();

    std::vector<flex::RuntimeStream*> handles_to_sync;
    {
      auto& pool = getStreamPool();
      std::lock_guard<std::mutex> lock(pool.mutex);

      // Default stream (ID 0) is always present when the pool is initialized
      auto default_it = pool.stream_handle_map.find(0);
      if (default_it != pool.stream_handle_map.end()) {
        handles_to_sync.push_back(default_it->second);
      }

      auto collect = [&](auto& stream_map) {
        auto it = stream_map.find(device_index);
        if (it == stream_map.end()) return;
        for (auto sid : it->second) {
          auto h = pool.stream_handle_map.find(sid);
          if (h != pool.stream_handle_map.end()) {
            handles_to_sync.push_back(h->second);
          }
        }
      };
      collect(pool.low_priority_streams);
      collect(pool.high_priority_streams);
    }  // lock released

    auto runtime = GlobalRuntime::get();
    c10::DeviceGuard guard(dev);
    for (auto handle : handles_to_sync) {
      handle->synchronize();
    }
  };
  if (device.has_value()) {
    sync_one_device(device.value());
  } else {
    sync_one_device(
        c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx));
  }
}

}  // namespace spyre
