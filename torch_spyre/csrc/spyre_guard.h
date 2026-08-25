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

#include <c10/core/DeviceType.h>
#include <torch/library.h>

namespace spyre {

// Parse LOCAL_RANK from the environment.
//
// Uses strtoll + end-pointer check (not atoi) so that non-numeric or empty
// strings are detected. Validates that the result is non-negative and fits in
// c10::DeviceIndex (int8_t) without truncation. Returns 0 when LOCAL_RANK is
// unset. Throws std::invalid_argument on malformed input and std::range_error
// if the value is negative or exceeds c10::DeviceIndex's range.
//
// Do NOT call this directly from a thread_local initializer — use the
// file-local parse_local_rank_safe() wrapper in spyre_guard.cpp instead, which
// catches and returns 0 so that std::terminate is not called on worker threads
// that have no surrounding try/catch.  _startRuntime() calls this function on
// the guarded path where exceptions propagate normally.
c10::DeviceIndex parse_local_rank();

struct SpyreGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static thread_local c10::DeviceIndex
      tls_idx;  // your TLS (or delegate to your runtime)

  c10::DeviceType type() const override;

  c10::Device exchangeDevice(c10::Device d) const override;

  c10::Device getDevice() const override;

  void setDevice(c10::Device d) const override;

  void uncheckedSetDevice(c10::Device) const noexcept;

  c10::DeviceIndex deviceCount() const noexcept override;

  // Stream methods
  c10::Stream getStream(c10::Device device) const override;

  c10::Stream getDefaultStream(c10::Device device) const override;

  c10::Stream getStreamFromGlobalPool(
      c10::Device device, bool isHighPriority = false) const override;

  c10::Stream getNewStream(c10::Device device, int priority = 0) const override;

  c10::Stream exchangeStream(c10::Stream stream) const override;

  void synchronizeStream(const c10::Stream& stream) const override;
  void synchronizeDevice(c10::DeviceIndex device_index) const override;
  bool queryStream(const c10::Stream& stream) const override;
  void recordDataPtrOnStream(const c10::DataPtr&, const c10::Stream&) const;

  c10::DeviceCapability getDeviceCapability(c10::Device) const override;
};

}  // namespace spyre
