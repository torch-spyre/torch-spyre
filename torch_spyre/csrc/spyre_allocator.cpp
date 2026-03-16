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
#include "spyre_mem.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"

namespace spyre {

SpyreAllocator::SpyreAllocator() = default;

flex::DeviceMemoryAllocatorPtr SpyreAllocator::getAllocator(
    unsigned int dev_id) {
  return GlobalRuntime::get()
      ->GetDeviceHandle(dev_id)
      ->GetDeviceMemoryAllocator();
}

SpyreAllocator& SpyreAllocator::instance() {
  static SpyreAllocator allocator;
  return allocator;
}

at::DataPtr SpyreAllocator::allocate(size_t nbytes) {
  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice();

  auto device_id = curr_device.index();
  auto current_stream = getCurrentStream(curr_device);

  DEBUGINFO("allocating ", nbytes, " (bytes) on Spyre", curr_device);
  if (nbytes <= 0) {
    return {nullptr, nullptr, &ReportAndDelete, curr_device};
  }
  auto allocator = getAllocator(device_id);
  flex::DeviceMemoryAllocationPtr data;  // a smart-pointer object
  // NOTE: last argument should be set to 0
  allocator->TryAllocate(&data, nbytes, 0);
  TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on Spyre device.");
  auto* ctx = new SharedOwnerCtx{std::move(data), device_id};
  void* ctx_void = static_cast<void*>(ctx);

  void* data_void = static_cast<void*>(ctx->owner.get());

  auto data_ptr_result =
      at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);

  return data_ptr_result;
}

void SpyreAllocator::ReportAndDelete(void* ctx_void) {
  if (!ctx_void) {
    return;
  }
  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  delete ctx;
}

at::DeleterFnPtr SpyreAllocator::raw_deleter() const {
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
