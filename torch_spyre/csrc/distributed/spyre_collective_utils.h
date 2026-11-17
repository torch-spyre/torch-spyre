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
#include <c10/util/accumulate.h>

#include "../spyre_tensor_impl.h"

namespace spyre {

inline void check_collective_device(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "Collective tensor must be on the Spyre device");
  if (tensor.scalar_type() == c10::ScalarType::BFloat16) {
    TORCH_CHECK(
        get_spyre_tensor_layout(tensor).device_dtype ==
            DataFormats::SEN169_FP16,
        "BF16 collectives require DL16 device storage; explicit non-DL16 "
        "device layouts are not supported");
  }
}

inline void check_collective_storage(const at::Tensor& tensor) {
  check_collective_device(tensor);
  TORCH_CHECK(tensor.is_contiguous(), "Collective tensor must be contiguous");
  TORCH_CHECK(tensor.storage_offset() == 0,
              "Collectives do not support nonzero storage offsets");
  // A zero-offset prefix view may retain its base tensor's DMA descriptor.
  // Borrowed composite addresses cover that descriptor, not just the view.
  TORCH_CHECK(
      c10::multiply_integers(get_spyre_tensor_sizes(tensor)) == tensor.numel(),
      "Collective tensor requires matching device storage extent: logical "
      "elements=",
      tensor.numel(), ", descriptor elements=",
      c10::multiply_integers(get_spyre_tensor_sizes(tensor)),
      "; a view or allocation descriptor may cover a different logical shape");
}

}  // namespace spyre
