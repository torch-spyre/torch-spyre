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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace torch_spyre {
namespace scratchpad {

// First-fit offsets for the boxes of a no-overlap model, in placement order.
// Box i lives over [x0[i], x1[i]) and needs size[i] bytes; an absent box
// gets offset 0. A box with pinned[i] >= 0 keeps that offset and is an
// obstacle to every box in ``order``. A box with a partner in ties[i] (an
// active merge) already placed takes that partner's offset. Every other box
// in ``order`` goes to the lowest offset clear of the placed boxes that share
// its time.
std::vector<int64_t> pack_boxes(const std::vector<int>& order,
                                const std::vector<int64_t>& x0,
                                const std::vector<int64_t>& x1,
                                const std::vector<int64_t>& size,
                                const std::vector<bool>& present,
                                const std::vector<int64_t>& pinned,
                                const std::vector<std::vector<int>>& ties);

// Registers ``pack_boxes`` on module ``m``. Called from the _C
// PYBIND11_MODULE block in module.cpp.
void register_box_packer(pybind11::module_& m);

}  // namespace scratchpad
}  // namespace torch_spyre
