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

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>

namespace spyre {

class SpyreTensorLayout;

at::Tensor reinterpret_tensor(const at::Tensor& self, c10::IntArrayRef size,
                              c10::IntArrayRef stride,
                              int64_t offset_increment);

at::Tensor reinterpret_tensor_with_layout(const at::Tensor& self,
                                          c10::IntArrayRef size,
                                          c10::IntArrayRef stride,
                                          int64_t offset_increment,
                                          SpyreTensorLayout stl);

at::Tensor as_strided_with_layout(const at::Tensor& self, c10::IntArrayRef size,
                                  c10::IntArrayRef stride,
                                  std::optional<int64_t> storage_offset_,
                                  SpyreTensorLayout device_layout);

}  // namespace spyre
