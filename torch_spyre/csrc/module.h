/*
 * Copyright IBM Corp. 2025
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

}  // namespace spyre
