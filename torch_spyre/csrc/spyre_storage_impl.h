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
#include <ATen/ATen.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/SymInt.h>

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
  SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes,
                   at::Allocator* allocator, bool resizable);
};

}  // namespace spyre
