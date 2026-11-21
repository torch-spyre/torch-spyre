/*
 * Copyright 2026 The Torch-Spyre Authors.
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

#include <ATen/ATen.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace spyre {

/** Physical transfer size, or 0 for unsupported multi-chunk allocations. */
size_t dma_bytes(const at::Tensor& dst);

/** Submit all raw H2D copies on one stream, then wait before returning. */
void copy_raw_h2d_batch(
    const std::vector<std::pair<at::Tensor, at::Tensor>>& entries);

struct HostConvertTarget {
  void* dst;
  size_t capacity;
  at::Tensor cpu_tensor;
  at::Tensor dst_tensor;
};

/**
 * Convert host tensors directly into caller-owned device-format buffers.
 * Buffers may be partially written on failure; workers join before rethrow.
 */
void convert_h2d_into(const std::vector<HostConvertTarget>& targets,
                      unsigned int num_threads);

}  // namespace spyre
