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

#include "spyre_guard.h"

#include <ATen/core/op_registration/adaption.h>

#include "module.h"
#include "spyre_stream.h"

namespace spyre {

c10::DeviceType SpyreGuardImpl::type() const {
  return c10::DeviceType::PrivateUse1;
}

c10::Device SpyreGuardImpl::exchangeDevice(c10::Device d) const {
  auto old = getDevice();
  setDevice(d);
  return old;
}

c10::Device SpyreGuardImpl::getDevice() const {
  return {type(), tls_idx};
}

void SpyreGuardImpl::setDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == type());
  // (optionally tell your runtime to switch)
  tls_idx = d.index();
}

void SpyreGuardImpl::uncheckedSetDevice(c10::Device) const noexcept {}

c10::DeviceIndex SpyreGuardImpl::deviceCount() const noexcept {
  //  FIXME (tmhoangt) - return actual device count
  return 1;
}

c10::Stream SpyreGuardImpl::getStream(c10::Device device) const {
  return getCurrentStream(device).unwrap();
}

c10::Stream SpyreGuardImpl::getNewStream(c10::Device device,
                                         int priority) const {
  return getStreamFromPool(device, priority).unwrap();
}

void SpyreGuardImpl::synchronizeStream(const c10::Stream& stream) const {
  TORCH_CHECK(stream.device().type() == this->type());
  SpyreStream(stream).synchronize();
}

bool SpyreGuardImpl::queryStream(const c10::Stream& stream) const {
  TORCH_CHECK(stream.device().type() == this->type());
  return SpyreStream(stream).query();
}

void SpyreGuardImpl::synchronizeDevice(c10::DeviceIndex device_index) const {
  c10::Device dev(c10::DeviceType::PrivateUse1, device_index);
  spyre::synchronizeDevice(dev);
}

c10::Stream SpyreGuardImpl::exchangeStream(c10::Stream stream) const {
  SpyreStream ss(stream);

  c10::Stream old = getCurrentStream(stream.device()).unwrap();

  // Set TLS current stream for THAT device index
  setCurrentStream(ss);

  return old;
}

void SpyreGuardImpl::recordDataPtrOnStream(const c10::DataPtr&,
                                           const c10::Stream&) const {}

c10::DeviceCapability SpyreGuardImpl::getDeviceCapability(
    c10::Device /*unused*/) const {
  c10::DeviceCapability cap{};

  cap.capability_data.capability_bits =
      (1ULL << c10::kIndex_Float) | (1ULL << c10::kIndex_Half) |
      (1ULL << c10::kIndex_Bool) | (1ULL << c10::kIndex_Char) |
      (1ULL << c10::kIndex_Byte) | (1ULL << c10::kIndex_Short) |
      (1ULL << c10::kIndex_Int4) | (1ULL << c10::kIndex_BFloat16) |
      (1ULL << c10::kIndex_Float8_e4m3fn) |
      (1ULL << c10::kIndex_Float8_e5m2fnuz);

  return cap;
}

thread_local c10::DeviceIndex SpyreGuardImpl::tls_idx = 0;

// Registration (runs at DSO load — after you import your module)
C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl);

}  // namespace spyre
