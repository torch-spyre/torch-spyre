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

#include <flex/runtime.hpp>
#include <memory>

namespace spyre {

struct SharedOwnerCtx {
  flex::DeviceMemoryAllocationPtr owner;
  signed char device_id;
};

class GlobalRuntime {
 public:
  static void set(const std::shared_ptr<flex::Runtime>& runtime) {
    instance() = runtime;
  }
  static void reset() {
    instance().reset();  // sets the shared_ptr to nullptr
  }

  static const std::shared_ptr<flex::Runtime>& get() {
    return instance();
  }

 private:
  GlobalRuntime() = delete;
  ~GlobalRuntime() = delete;

  static std::shared_ptr<flex::Runtime>& instance() {
    static std::shared_ptr<flex::Runtime> s;
    return s;
  }
};
bool get_downcast_warn_enabled();

}  // namespace spyre
