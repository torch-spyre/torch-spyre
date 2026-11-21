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

#include "host_transfer.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <spyrecode-host-functions/sendataconvert/sen_data_convert.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "module.h"
#include "spyre_composite_address.h"
#include "spyre_mem.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"

namespace spyre {
namespace {

size_t host_buffer_alignment() {
  auto* ctx = flex::RuntimeContext::getInstance();
  TORCH_CHECK(ctx != nullptr, "Spyre runtime not initialized");
  const auto type = ctx->getDeviceType();
  TORCH_CHECK(type == flex::DeviceType::PF || type == flex::DeviceType::VF,
              "Host transfers require an AF device");
  // PF and VF IOMMU mappers both align host mappings to the OS page size.
  const auto page_size = sysconf(_SC_PAGESIZE);
  TORCH_CHECK(page_size > 0, "Unable to query host page size");
  return std::max(sizeof(void*), static_cast<size_t>(page_size));
}

}  // namespace

size_t dma_bytes(const at::Tensor& dst) {
  TORCH_CHECK(dst.is_privateuseone(),
              "dma_bytes: dst must be a Spyre tensor, got ", dst.device());
  const flex::CompositeAddress* addr = get_composite_address(dst);
  return addr->is_single_chunk() ? addr->total_size() : 0;
}

void copy_raw_h2d_batch(
    const std::vector<std::pair<at::Tensor, at::Tensor>>& entries) {
  static std::mutex mutex;
  // A shut-down stream can return from synchronize before DMA finishes. Keep
  // uncertain storage even during static destruction, and reject further use.
  static auto* failed = new std::vector<std::pair<at::Tensor, at::Tensor>>();
  std::lock_guard<std::mutex> lock(mutex);
  TORCH_CHECK(failed->empty(),
              "raw H2D completion previously failed; restart the process");
  if (entries.empty()) return;

  const auto device = entries.front().second.device();
  TORCH_CHECK(device.is_privateuseone(),
              "copy_raw_h2d_batch: destination must be on Spyre");
  c10::DeviceGuard guard(device);
  const size_t alignment = host_buffer_alignment();
  for (const auto& [buffer, dst] : entries) {
    TORCH_CHECK(dst.device() == device,
                "copy_raw_h2d_batch: destinations must be on the same device");
    TORCH_CHECK(buffer.is_cpu() && buffer.scalar_type() == at::kByte &&
                    buffer.is_contiguous(),
                "copy_raw_h2d_batch: buffer must be contiguous CPU uint8");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(buffer.data_ptr()) % alignment == 0,
                "copy_raw_h2d_batch: buffer is not IOMMU-aligned");
    const size_t bytes = dma_bytes(dst);
    TORCH_CHECK(bytes != 0 && buffer.nbytes() == bytes,
                "copy_raw_h2d_batch: buffer size must equal the single-chunk "
                "allocation size, got ",
                buffer.nbytes(), " B for ", bytes, " B");
  }

  auto owned = entries;
  const auto stream = getCurrentStream(device);
  TORCH_CHECK(stream.getError() == SpyreStreamError::Success,
              "Raw H2D stream is shut down; completion is not guaranteed");
  std::exception_ptr submission_error;
  try {
    for (const auto& [buffer, dst] : owned) {
      auto params = std::unique_ptr<flex::CopyRawParams,
                                    decltype(&flex::destroyCopyRawParams)>(
          flex::createCopyRawParams(buffer.data_ptr(), buffer.nbytes(),
                                    get_composite_address(dst), true),
          &flex::destroyCopyRawParams);
      stream.launchCopyRaw(params.get());
    }
  }
  catch (...) {
    submission_error = std::current_exception();
  }
  try {
    TORCH_CHECK(stream.getError() == SpyreStreamError::Success,
                "Raw H2D stream is shut down; completion is not guaranteed");
    stream.synchronize();
    TORCH_CHECK(stream.getError() == SpyreStreamError::Success,
                "Raw H2D stream failed; completion is not guaranteed");
  }
  catch (...) {
    failed->swap(owned);
    throw;
  }
  if (submission_error) std::rethrow_exception(submission_error);
}

void convert_h2d_into(const std::vector<HostConvertTarget>& targets,
                      unsigned int num_threads) {
  if (targets.empty()) return;

  if (num_threads == 0) {
    num_threads = std::max(1u, std::thread::hardware_concurrency());
  }
  num_threads = static_cast<unsigned int>(
      std::min(static_cast<size_t>(num_threads), targets.size()));

  // Let each worker take the next tensor as it finishes a conversion.
  std::atomic<size_t> next{0};
  std::exception_ptr first_exception;
  std::mutex exc_mutex;
  auto worker = [&]() {
    while (true) {
      const size_t idx = next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= targets.size()) break;
      const auto& t = targets[idx];
      try {
        TORCH_CHECK(t.cpu_tensor.device().is_cpu(),
                    "convert_h2d_into: source must be on CPU, got ",
                    t.cpu_tensor.device());
        TORCH_CHECK(t.dst_tensor.is_privateuseone(),
                    "convert_h2d_into: dst_tensor must be a Spyre tensor, got "
                    "device ",
                    t.dst_tensor.device());
        const size_t bytes = dma_bytes(t.dst_tensor);
        TORCH_CHECK(bytes != 0 && bytes <= t.capacity,
                    "convert_h2d_into: dst_tensor must be a single-chunk "
                    "allocation of at most ",
                    t.capacity, " B, got ", bytes, " B");

        // Same DCI as SpyreStream::copyAsync (the copy_tensor path).
        const auto dci = generate_dci(&t.cpu_tensor, &t.dst_tensor,
                                      get_spyre_tensor_layout(t.dst_tensor),
                                      /*host2device=*/true);
        deeptools::ConvertData(t.dst, t.capacity, t.cpu_tensor.storage().data(),
                               dci);
      }
      catch (...) {
        std::lock_guard<std::mutex> lock(exc_mutex);
        if (!first_exception) first_exception = std::current_exception();
        // Drain the remaining entries before propagating the first failure.
      }
    }
  };

  if (num_threads == 1) {
    worker();
  } else {
    // jthread also joins started workers if creating another thread fails.
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (unsigned int i = 0; i < num_threads; ++i) {
      threads.emplace_back(worker);
    }
    for (auto& thread : threads) thread.join();
  }
  if (first_exception) std::rethrow_exception(first_exception);
}

}  // namespace spyre
