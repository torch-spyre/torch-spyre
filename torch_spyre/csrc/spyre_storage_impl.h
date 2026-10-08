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
#include <c10/core/StorageImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorImpl.h>

#include <mutex>
#include <optional>
#include <vector>

namespace spyre {

/**
 * An SpyreStorageImpl is a storage type which always returns
 * the Spyre device through device_type, regardless of whether
 * the data is on CPU or on Spyre.
 * For now, this is actually a CPU storage class, but eventually
 * it will be used to hold Spyre custom storage format conversions,
 * like Spyre specific stickification
 */
class SpyreStorageImpl : public c10::StorageImpl {
 public:
  // Shared by aliases. The version handle detects ordinary PyTorch writes;
  // raw DMA/fill entry points explicitly invalidate before writing.
  struct ZeroPaddingCertificate {
    std::vector<int64_t> device_size;
    std::vector<int64_t> stride_map;
    std::vector<int64_t> valid_size;
    std::vector<int64_t> host_size;
    std::vector<int64_t> host_stride;
    c10::VariableVersion version;
    uint32_t recorded_version;
  };
  mutable std::mutex zero_padding_mutex;
  std::optional<ZeroPaddingCertificate> zero_padding;

  SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes,
                   c10::DeviceAllocator* allocator, bool resizable);
};

}  // namespace spyre
