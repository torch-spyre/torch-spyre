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

#include "module.h"

#include <pybind11/pybind11.h>

#include <flex/flex_factory.hpp>
#include <memory>
#include <sendnn/graph.hpp>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/graph/graph_deserializer.hpp>
#include <sendnn/graph/graph_utils.hpp>
#include <sendnn/runtime/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/tensor_info.hpp>
#include <sendnn/util/status.hpp>

#include "logging.h"
#include "spyre_mem.h"
#include "spyre_sendnn_utils.h"

namespace spyre {

void _startRuntime() {
  DEBUGINFO("starting runtime");
  // TODO(tmhoangt): move sendnn::RuntimeInterface to flex to isolate from
  // sendnn
  std::shared_ptr<sendnn::RuntimeInterface> base_runtime;
  auto s = flex::CreateRuntimeInterface(&base_runtime);
  std::shared_ptr<flex::Runtime> runtime =
      std::dynamic_pointer_cast<flex::Runtime>(base_runtime);
  if (runtime) {
    GlobalRuntime::set(runtime);
    DEBUGINFO(s);
    DEBUGINFO("runtime started");
  } else {
    DEBUGINFO("runtime FAILED TO START.");
  }
}
void startRuntime() {
  static std::once_flag flag;
  std::call_once(flag, _startRuntime);
}

void freeRuntime() {
  GlobalRuntime::reset();
}

}  // namespace spyre

PYBIND11_MODULE(_C, m) {
  m.doc() = "Spyre C++ bindings";
  m.def("start_runtime", &spyre::startRuntime);
  m.def("free_runtime", &spyre::freeRuntime);
}
