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

#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>

namespace spyre {

struct SpyreGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static thread_local c10::DeviceIndex
      tls_idx;  // your TLS (or delegate to your runtime)

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }
  c10::Device exchangeDevice(c10::Device d) const override {
    auto old = getDevice();
    setDevice(d);
    return old;
  }

  c10::Device getDevice() const override {
    return {type(), tls_idx};
  }
  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    // (optionally tell your runtime to switch)
    tls_idx = d.index();
  }
  void uncheckedSetDevice(c10::Device) const noexcept {}

  c10::DeviceIndex deviceCount() const noexcept override {
    //  FIXME (tmhoangt) - return actual device count
    return 1;
  }

  // Do Spyre have streams, override
  // getStream/exchangeStream/.../recordDataPtrOnStream
  c10::Stream getStream(c10::Device device) const override {
    return c10::Stream(c10::Stream::Default::DEFAULT, device);
  }
  c10::Stream exchangeStream(c10::Stream stream) const override {
    return stream;
  }
  void recordDataPtrOnStream(const c10::DataPtr&, const c10::Stream&) const {}
};

thread_local c10::DeviceIndex SpyreGuardImpl::tls_idx = 0;

// Registration (runs at DSO load — after you import your module)
C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl);

};  // namespace spyre
