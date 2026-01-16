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

#include <ATen/EmptyTensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include <memory>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/graph/graph_serializer.hpp>
#include <sendnn/graph/graph_utils.hpp>
#include <sendnn/graph/senparms.hpp>
#include <sendnn/interface/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "spyre_storage_impl.h"
#include "spyre_tensor_impl.h"

namespace flex {
class DeviceMemoryAllocation;
using DeviceMemoryAllocationPtr = std::shared_ptr<DeviceMemoryAllocation>;
}  // namespace flex

namespace spyre {

// todo tuple of op, sizes, strides -> keeping as string to be simple for now
using GraphLoaderCacheKey =
    std::tuple<std::string, std::vector<int64_t>, std::vector<int64_t>>;

struct GraphLoaderCacheHash {
  std::size_t operator()(
      const std::tuple<std::string, std::vector<int64_t>, std::vector<int64_t>>&
          graph_loader_key) const {
    auto result_hash = std::hash<std::string>{}(std::get<0>(graph_loader_key));
    for (auto s : std::get<1>(graph_loader_key)) {
      result_hash ^= (std::hash<int64_t>{}(s) << 1);
    }
    for (auto s : std::get<2>(graph_loader_key)) {
      result_hash ^= (std::hash<int64_t>{}(s * 2) << 1);
    }
    return result_hash;
  }
};

class GlobalGraphLoaderCache {
 public:
  static std::unordered_map<GraphLoaderCacheKey, sendnn::GraphLoader,
                            GraphLoaderCacheHash>&
  get() {
    return instance();
  }

 private:
  GlobalGraphLoaderCache() {}
  ~GlobalGraphLoaderCache() {}

  static std::unordered_map<GraphLoaderCacheKey, sendnn::GraphLoader,
                            GraphLoaderCacheHash>&
  instance() {
    static std::unordered_map<GraphLoaderCacheKey, sendnn::GraphLoader,
                              GraphLoaderCacheHash>
        s;
    return s;
  }
};

/**
 * Create a dummy op for a single tensor in/out for use in allocation
 */
sendnn::GraphBuilder createDummyOp(c10::IntArrayRef sizes);

sendnn::GraphLoader prepareGraphLoader(sendnn::GraphBuilder* gb);

void storeCachedGraphLoader(std::string name, c10::IntArrayRef size,
                            c10::IntArrayRef stride, sendnn::GraphLoader gl);
std::optional<sendnn::GraphLoader> getCachedGraphLoader(
    std::string name, c10::IntArrayRef size, c10::IntArrayRef stride);

sendnn::GraphLoader& parseGraphLoader(
    sendnn::GraphLoader& gl,
    std::optional<c10::IntArrayRef> out_shape = std::nullopt,
    std::optional<c10::IntArrayRef> out_stride = std::nullopt,
    std::optional<std::vector<c10::IntArrayRef>> inp_shapes = std::nullopt,
    std::optional<std::vector<c10::IntArrayRef>> inp_strides = std::nullopt);

sendnn::ConstTensor createInputTensor(sendnn::GraphLoader& gl, void* data_ptr,
                                      unsigned int input_index = 0,
                                      uint64_t sn_index = 1);

sendnn::Tensor createOutputTensor(sendnn::GraphLoader& gl, void* data_ptr,
                                  unsigned int output_index = 0,
                                  uint64_t sn_index = 1);

sendnn::TensorInfo getTensorInfo(const at::Tensor& input);

sendnn::TensorInfo getScalarTensorInfo(const at::Tensor& input);

}  // namespace spyre
