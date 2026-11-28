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

#include <limits>
#include <vector>

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

// spyre-comms reduce_scatter hands rank r the r-th contiguous 1/group_size of
// the flat device input. Both the compiled path (_reduce_scatter in
// _inductor/lowering.py) and the eager SpyreCCLBackend::reduce_scatter_single
// therefore stage the input as (A, group_size, B), whose default device layout
// is [group_size, ceil(B/stick), A, stick]: each slab is then exactly the
// default device image of an (A, B) tensor. Reject anything else: a different
// layout scatters the wrong elements without any error from spyre-comms.
// Same condition as ReduceScatterKernel.codegen (_inductor/ir.py); keep the
// two in sync. Returns the staged layout.
inline SpyreTensorLayout check_reduce_scatter_staged_input(
    const at::Tensor& staged, int64_t group_size) {
  TORCH_CHECK(staged.dim() == 3 && staged.size(1) == group_size,
              "reduce_scatter: expected a staged (A, group_size, B) input, got "
              "sizes ",
              staged.sizes(), " for group_size ", group_size);
  const auto staged_stl = get_spyre_tensor_layout(staged);
  const SpyreTensorLayout expected(staged.sizes().vec(), staged.scalar_type());
  TORCH_CHECK(staged_stl == expected && expected.device_size[0] == group_size,
              "reduce_scatter: staged input must use the default device layout "
              "with the group dim outermost (",
              expected.toString(), "), got ", staged_stl.toString());
  return staged_stl;
}

// The per-rank output receives one slab of the staged input, so its device
// allocation must be exactly 1/group_size of the staged input's.
inline void check_reduce_scatter_output_tiles(
    const at::Tensor& output, const SpyreTensorLayout& staged_stl,
    int64_t group_size) {
  TORCH_CHECK(
      get_device_size_in_bytes(get_spyre_tensor_layout(output)) * group_size ==
          get_device_size_in_bytes(staged_stl),
      "reduce_scatter: per-rank output device size does not tile the "
      "staged input");
}

// Convert split sizes to the int vector spyre-comms alltoallw expects,
// checking each is positive and fits in an int.
inline std::vector<int> to_alltoall_splits(c10::IntArrayRef splits,
                                           const char* which) {
  std::vector<int> out;
  out.reserve(splits.size());
  for (int64_t s : splits) {
    TORCH_CHECK(s >= 0 && s <= std::numeric_limits<int>::max(),
                "alltoall: ", which, " split size out of range: ", s);
    // spyre-comms alltoallw accepts a zero split when the plan is built but
    // segfaults in Collective::Convert when tensors are applied.
    // TODO(flaviabeo): allow zero splits once spyre-comms alltoallw handles
    // them.
    TORCH_CHECK(s > 0, "alltoall: zero-size ", which,
                " splits are not supported yet (spyre-comms alltoallw crashes "
                "on them); got ",
                splits);
    out.push_back(static_cast<int>(s));
  }
  return out;
}

// Device bytes spanned by one dim-0 row of `tensor`, if its rows are
// contiguous in device memory.
//
// spyre-comms splits the *flat* device buffer, so an all_to_all along dim 0 is
// only correct when each dim-0 row is one contiguous run of device memory and
// rows are stored in order. That holds for a 1-D tensor without padding, and
// for an N-D tensor whose outermost non-trivial device dim walks dim 0
// (untiled) with everything else inside it. It does not hold for the default
// layout of e.g. (4, 128) fp16, device_size=[2, 4, 64], where the stick tiles
// of dim 1 are outside dim 0.
// Same condition as _dim0_row_device_elems in _inductor/ir.py (which counts
// in elements, not bytes); keep the two in sync.
inline int64_t dim0_row_device_bytes(const at::Tensor& tensor,
                                     const char* what) {
  TORCH_CHECK(tensor.dim() > 0, "all_to_all ", what,
              " must have at least 1 dim");
  const auto stl = get_spyre_tensor_layout(tensor);
  const auto total = static_cast<int64_t>(get_device_size_in_bytes(stl));
  const int64_t dim0 = tensor.size(0);
  TORCH_CHECK(dim0 > 0, "all_to_all ", what, " must have a non-empty dim 0");
  if (dim0 == 1) {
    return total;
  }
  if (tensor.dim() == 1) {
    const int64_t elem_bytes = 128 / stl.elems_per_stick();
    TORCH_CHECK(total == dim0 * elem_bytes, "all_to_all ", what,
                ": a 1-D tensor of ", dim0, " elements is padded to ",
                total / elem_bytes, " device elements; use a multiple of ",
                stl.elems_per_stick(), " elements");
    return elem_bytes;
  }
  const int64_t stride0 = tensor.stride(0);
  std::vector<size_t> non_trivial;
  for (size_t i = 0; i < stl.device_size.size(); i++) {
    if (stl.device_size[i] > 1) {
      non_trivial.push_back(i);
    }
  }
  // stride_map entries of 0 are synthetic (normalized from -1).
  bool rows_contiguous = !non_trivial.empty() &&
                         stl.stride_map[non_trivial[0]] == stride0 &&
                         stl.device_size[non_trivial[0]] == dim0;
  for (size_t k = 1; rows_contiguous && k < non_trivial.size(); k++) {
    rows_contiguous = stl.stride_map[non_trivial[k]] < stride0;
  }
  TORCH_CHECK(
      rows_contiguous, "all_to_all ", what,
      ": dim-0 rows are not contiguous in device memory (size=", tensor.sizes(),
      ", ", stl.toString(), "); the last dims must fit in one stick tile (",
      stl.elems_per_stick(), " elements) or the tensor must be 1-D");
  return total / dim0;
}

}  // namespace spyre
