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

#include "spyre_storage_impl.h"

#include <utility>

namespace spyre {

SpyreStorageImpl::SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes,
                                   c10::DeviceAllocator* allocator,
                                   bool resizable)
    : c10::StorageImpl(use_byte_size_t(), size_bytes, allocator, resizable) {}

SpyreStorageImpl::SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes,
                                   c10::DataPtr data_ptr,
                                   c10::DeviceAllocator* allocator)
    : c10::StorageImpl(use_byte_size_t(), std::move(size_bytes),
                       std::move(data_ptr), allocator,
                       /*resizable=*/false) {}

}  // namespace spyre
