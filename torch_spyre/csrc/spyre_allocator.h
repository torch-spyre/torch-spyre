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

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <flex/device_memory_allocator.hpp>

namespace spyre {

// A custom allocator for our custom device, which returns a handle to the
// allocated memory, not the actual pointer.
struct SpyreAllocator final : public at::Allocator {
 private:
  SpyreAllocator();

  flex::DeviceMemoryAllocatorPtr getAllocator(unsigned int dev_id);

 public:
  static SpyreAllocator& instance();

  at::DataPtr allocate(size_t nbytes) override;

  static void ReportAndDelete(void* ctx_void);

  at::DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const final;
};

}  // namespace spyre
