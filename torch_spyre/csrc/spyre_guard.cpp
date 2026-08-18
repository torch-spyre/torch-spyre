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

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "module.h"
#include "spyre_device_enum.h"
#include "spyre_stream.h"

namespace spyre {

c10::DeviceIndex parse_local_rank() {
  const char* env = std::getenv("LOCAL_RANK");
  if (!env || env[0] == '\0') {
    return 0;
  }

  errno = 0;
  char* end = nullptr;
  const int64_t val = std::strtoll(env, &end, 10);

  if (end == env || *end != '\0') {
    throw std::invalid_argument(
        std::string("LOCAL_RANK is not a valid integer: '") + env + "'");
  }
  if (errno == ERANGE || val < 0) {
    throw std::range_error(std::string("LOCAL_RANK value is out of range: '") +
                           env + "'");
  }
  // c10::DeviceIndex is int8_t — guard against silent truncation.
  constexpr int64_t kMaxDeviceIndex =
      static_cast<int64_t>(std::numeric_limits<c10::DeviceIndex>::max());
  if (val > kMaxDeviceIndex) {
    throw std::range_error(std::string("LOCAL_RANK value ") +
                           std::to_string(val) +
                           " exceeds the maximum supported device index (" +
                           std::to_string(kMaxDeviceIndex) + ")");
  }

  return static_cast<c10::DeviceIndex>(val);
}

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
  return c10::DeviceIndex(getVisibleDeviceCount());
}

c10::Stream SpyreGuardImpl::getStream(c10::Device device) const {
  return getCurrentStream(device).unwrap();
}

c10::Stream SpyreGuardImpl::getDefaultStream(c10::Device device) const {
  return spyre::getDefaultStream(device).unwrap();
}

c10::Stream SpyreGuardImpl::getStreamFromGlobalPool(c10::Device device,
                                                    bool isHighPriority) const {
  int priority = isHighPriority ? -1 : 0;
  return getStreamFromPool(device, priority).unwrap();
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

// Safe seed for the thread_local initializer.
// parse_local_rank() can throw (malformed / out-of-range LOCAL_RANK).  A
// thread_local dynamic initializer has no surrounding try/catch on worker
// threads, so an exception there calls std::terminate.  Return 0 on any
// error; _startRuntime re-calls parse_local_rank() on the guarded path and
// will surface the error as a catchable exception.
static c10::DeviceIndex parse_local_rank_safe() noexcept {
  try {
    return parse_local_rank();
  }
  catch (...) {
    return 0;
  }
}

thread_local c10::DeviceIndex SpyreGuardImpl::tls_idx = parse_local_rank_safe();

// Registration — runs when _C.so is loaded.
// Loading _C.so does NOT trigger device initialization; that only
// happens when start_runtime() is called via _lazy_init().
C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl);

}  // namespace spyre
