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

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <util/sen_host_ops.h>

#include <flex/flex.hpp>
#include <memory>

using DataConversionStrideInfo = data_conversion_stride_info;
using DataConversionInfo = data_conversion_info;

namespace spyre {

class GlobalRuntime {
 public:
  static void set(const std::shared_ptr<flex::RuntimeContext>& runtime) {
    instance() = runtime;
  }
  static void reset() {
    instance().reset();  // sets the shared_ptr to nullptr
  }

  static const std::shared_ptr<flex::RuntimeContext>& get() {
    return instance();
  }

 private:
  GlobalRuntime() = delete;
  ~GlobalRuntime() = delete;

  static std::shared_ptr<flex::RuntimeContext>& instance() {
    static std::shared_ptr<flex::RuntimeContext> s;
    return s;
  }
};
bool get_downcast_warn_enabled();
bool is_supported_dtype(c10::ScalarType dtype);
DataFormats get_device_dtype(c10::ScalarType torch_dtype);

int device_count();
void startRuntime();
}  // namespace spyre
