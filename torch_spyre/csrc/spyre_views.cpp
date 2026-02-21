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

#include "spyre_views.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>
#include <util/sen_data_convert.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_sendnn_utils.h"
#include "spyre_storage_impl.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace spyre {

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
static at::Tensor spyre_alias_with_sizes_and_strides(
    const at::Tensor& self, const Vec& sizes, const Vec& strides,
    SpyreTensorLayout device_layout) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset());
  self_tmp_->set_sizes_and_strides(sizes, strides);
  static_cast<SpyreTensorImpl*>(self_tmp_)->spyre_layout = device_layout;
  return self_;
}

// specialization for symbolic shapes and strides.
// SymIntArrayRef/ArrayRef<c10::SymInt> and
// SmallVector<c10::SymInt>/SymDimVector
template <template <typename...> typename Container>
static at::Tensor spyre_alias_with_sizes_and_strides(
    const at::Tensor& self, const Container<c10::SymInt>& sizes,
    const Container<c10::SymInt>& strides, SpyreTensorLayout device_layout) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  self_.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides,
                                                     self.sym_storage_offset());
  static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl())->spyre_layout =
      device_layout;
  return self_;
}

// A group maps a set of old host dims to a set of new host dims.
// The product of sizes on each side must be equal.
struct DimGroup {
  std::vector<size_t> old_dims;
  std::vector<size_t> new_dims;
};

SpyreTensorLayout compute_view_layout(c10::IntArrayRef old_sizes,
                                      c10::IntArrayRef new_sizes,
                                      const SpyreTensorLayout& old_stl) {
  size_t old_rank = old_sizes.size();
  size_t new_rank = new_sizes.size();
  int sparse_to_dense_stick_dim = -1;

  // Phase 1: Identify old->new host dimension groups
  std::vector<DimGroup> groups;
  std::unordered_map<size_t, std::vector<size_t>> size_1_insertions;
  size_t old_i = 0, new_j = 0;
  while (old_i < old_rank || new_j < new_rank) {
    // Handle trailing size-1 dims
    if (old_i >= old_rank) {
      TORCH_CHECK(new_sizes[new_j] == 1,
                  "view: unsqueezed dimension must be size 1");
      groups.push_back({{}, {new_j}});
      if (old_stl.dim_map[old_stl.device_size.size() - 1] == -1) {
        sparse_to_dense_stick_dim = new_j;
      }
      new_j++;
      continue;
    }
    if (new_j >= new_rank) {
      TORCH_CHECK(old_sizes[old_i] == 1,
                  "view: squeezed dimension must be size 1");
      groups.push_back({{old_i}, {}});
      old_i++;
      continue;
    }

    // Handle size-1 removals
    if (old_sizes[old_i] == 1 && new_sizes[new_j] != 1) {
      groups.push_back({{old_i}, {}});
      old_i++;
      continue;
    }
    // Handle size-1 insertions
    if (new_sizes[new_j] == 1 && old_sizes[old_i] != 1) {
      std::vector<size_t> to_inject;
      while (new_sizes[new_j] == 1 && new_j < new_rank) {
        to_inject.push_back(new_j);
        new_j++;
      }
      size_1_insertions[old_i] = to_inject;
      continue;
    }

    // Match by accumulating products
    std::vector<size_t> old_group = {old_i};
    std::vector<size_t> new_group = {new_j};
    int64_t old_prod = old_sizes[old_i];
    int64_t new_prod = new_sizes[new_j];
    while (old_prod != new_prod) {
      if (old_prod < new_prod) {
        old_i++;
        TORCH_CHECK(old_i < old_rank, "view: cannot match dimension products");
        old_prod *= old_sizes[old_i];
        old_group.push_back(old_i);
      } else {
        new_j++;
        TORCH_CHECK(new_j < new_rank, "view: cannot match dimension products");
        new_prod *= new_sizes[new_j];
        new_group.push_back(new_j);
      }
    }
    groups.push_back({old_group, new_group});
    old_i++;
    new_j++;
  }

  // Build old_dim -> group index mapping
  std::unordered_map<int32_t, size_t> old_dim_to_group;
  for (size_t g = 0; g < groups.size(); g++) {
    for (int32_t d : groups[g].old_dims) {
      old_dim_to_group[d] = g;
    }
  }

  int64_t eps = spyre::elems_per_stick(old_stl.device_dtype);

  // Phase 2: Process groups to build new device_size and dim_map
  std::vector<int64_t> new_device_size;
  std::vector<int32_t> new_dim_map;

  size_t dev_rank = old_stl.device_size.size();
  for (size_t d = 0; d < dev_rank; d++) {
    int32_t mapped_dim = old_stl.dim_map[d];
    bool is_stick_dim = (d == dev_rank - 1);

    // Device dims not mapped to any host dim (e.g., -1 for scalars/sparse)
    if (mapped_dim == -1) {
      if (sparse_to_dense_stick_dim == -1) {
        // Remains sparse
        new_device_size.push_back(old_stl.device_size[d]);
        new_dim_map.push_back(-1);
      } else {
        // Converting sparse to dense.
        new_device_size.push_back(is_stick_dim ? eps : 1);
        new_dim_map.push_back(sparse_to_dense_stick_dim);
      }
      continue;
    }

    // Insert any new size 1 dimensions that go before mapped_dim
    // The mapped_dim for the stick dim appears twice; only do this once
    if (!is_stick_dim) {
      if (size_1_insertions.count(mapped_dim)) {
        auto to_inject = size_1_insertions[mapped_dim];
        for (auto i = 0; i < to_inject.size(); i++) {
          new_device_size.push_back(1);
          new_dim_map.push_back(to_inject[i]);
        }
      }
    }

    auto git = old_dim_to_group.find(mapped_dim);
    TORCH_CHECK(git != old_dim_to_group.end(), "view: dim_map entry ",
                mapped_dim, " not found in any group");
    const DimGroup& grp = groups[git->second];

    // 1:1 group
    if (grp.old_dims.size() == 1 && grp.new_dims.size() == 1) {
      new_device_size.push_back(old_stl.device_size[d]);
      new_dim_map.push_back(grp.new_dims[0]);
      continue;
    }

    // 1:0 group (size-1 removal): old dim was size-1
    if (grp.old_dims.size() == 1 && grp.new_dims.empty()) {
      // Can be removed entirely, unless it is the stick dim.
      // Squeezing a stick dimension of size 1 creates a sparse tensor.
      if (mapped_dim == old_stl.dim_map[old_stl.dim_map.size() - 1]) {
        new_device_size.push_back((d == dev_rank - 1) ? eps : 1);
        new_dim_map.push_back(-1);
      }
      continue;
    }

    // 0:1 group (size-1 insertion): no old dim maps here, so this device dim
    // can't reference it. This shouldn't happen.
    TORCH_CHECK(!grp.old_dims.empty(),
                "view: device dim maps to old dim that doesn't exist in "
                "any group with old dims");

    // N:1 merge
    if (grp.old_dims.size() > 1 && grp.new_dims.size() == 1) {
      // All device dims mapping to any old dim in this merge group
      // now map to the single new dim. Keep device_size as-is,
      // since the data arrangement doesn't change.

      // Check if device dims for merged old dims are adjacent
      // and could be merged into a single device dim.
      // For now, keep them separate (works for non-adjacent case).
      new_device_size.push_back(old_stl.device_size[d]);
      new_dim_map.push_back(grp.new_dims[0]);
      continue;
    }

    // N:M complex (N>1, M>1) - reject
    if (grp.old_dims.size() > 1 && grp.new_dims.size() > 1) {
      TORCH_CHECK(false,
                  "view: N:M dimension groups (N>1, M>1) are not supported for "
                  "Spyre yet.");
    }

    // 1:M split
    TORCH_CHECK(grp.old_dims.size() == 1 && grp.new_dims.size() > 1,
                "view: unexpected group configuration");
    int32_t old_dim = grp.old_dims[0];
    int64_t dev_size = old_stl.device_size[d];

    // Check if this is the stick dimension (last device dim)
    if (is_stick_dim) {
      // Stick dim split: the innermost new dim must be >= elems_per_stick
      int32_t innermost_new = grp.new_dims.back();
      TORCH_CHECK(new_sizes[innermost_new] >= eps,
                  "view: splitting stick dimension requires innermost new "
                  "dimension size (",
                  new_sizes[innermost_new], ") >= elems_per_stick (", eps, ")");
      // Stick dim stays mapped to innermost new dim, unchanged
      new_device_size.push_back(dev_size);
      new_dim_map.push_back(innermost_new);
      continue;
    }

    // Non-stick dim split: check if this is the tiling count dim for the
    // stick dimension (maps to the same old_dim as the stick dim)
    bool is_tiling_count = false;
    if (d < dev_rank - 1) {
      // Check if the stick dim (last device dim) maps to the same old host dim
      int32_t stick_mapped = old_stl.dim_map[dev_rank - 1];
      if (stick_mapped == old_dim) {
        is_tiling_count = true;
      }
    }

    if (is_tiling_count) {
      // This device dim is the tiling count for the stick dim.
      // The old_dim splits into new_dims. The stick covers elems_per_stick
      // elements of the innermost new dim. The tiling count covers
      // the remaining factor of the innermost new dim and all outer new dims.
      //
      // Total sticks = product(new_dims except innermost) * ceil(innermost /
      // eps) But dev_size = old_host_size / eps (for the non-padded case, or
      // ceil(old_host_size / eps)).
      //
      // We need to split dev_size into factors matching the new dims.
      // innermost contributes ceil(new_sizes[innermost] / eps) tiling entries.
      // The rest contribute their full size.
      int32_t innermost_new = grp.new_dims.back();
      TORCH_CHECK(new_sizes[innermost_new] >= eps,
                  "view: splitting stick dimension requires innermost new "
                  "dimension size (",
                  new_sizes[innermost_new], ") >= elems_per_stick (", eps, ")");
      int64_t innermost_tiling = (new_sizes[innermost_new] + eps - 1) / eps;

      // Check the split is clean
      int64_t outer_product = 1;
      for (size_t k = 0; k < grp.new_dims.size() - 1; k++) {
        outer_product *= new_sizes[grp.new_dims[k]];
      }
      TORCH_CHECK(dev_size == outer_product * innermost_tiling,
                  "view: tiling count split doesn't factor cleanly: "
                  "device_size=",
                  dev_size, " but expected ", outer_product, " * ",
                  innermost_tiling);

      // For 1D splits (old_rank == 1), the outermost device dim is the
      // stick iterator (tiling count), matching the standard rank-2 layout
      // produced by init(). For higher-rank tensors, outer dims come first
      // and the tiling count is emitted last, adjacent to the stick dim.
      if (old_rank == 1) {
        new_device_size.push_back(innermost_tiling);
        new_dim_map.push_back(innermost_new);
        for (size_t k = 0; k < grp.new_dims.size() - 1; k++) {
          new_device_size.push_back(new_sizes[grp.new_dims[k]]);
          new_dim_map.push_back(grp.new_dims[k]);
        }
      } else {
        for (size_t k = 0; k < grp.new_dims.size() - 1; k++) {
          new_device_size.push_back(new_sizes[grp.new_dims[k]]);
          new_dim_map.push_back(grp.new_dims[k]);
        }
        new_device_size.push_back(innermost_tiling);
        new_dim_map.push_back(innermost_new);
      }
      continue;
    }

    // Regular non-stick dim split: dev_size should equal old host size.
    // Split into factors matching new dims.
    int64_t remaining = dev_size;
    for (size_t k = 0; k < grp.new_dims.size(); k++) {
      int32_t nd = grp.new_dims[k];
      int64_t ns = new_sizes[nd];
      TORCH_CHECK(remaining % ns == 0,
                  "view: device_size split doesn't factor cleanly: "
                  "remaining=",
                  remaining, " not divisible by new dim size=", ns);
      new_device_size.push_back(ns);
      new_dim_map.push_back(nd);
      remaining /= ns;
    }
    TORCH_CHECK(remaining == 1,
                "view: device_size split has leftover factor: ", remaining);
  }

  auto res =
      SpyreTensorLayout(new_device_size, new_dim_map, old_stl.device_dtype);
  DEBUGINFO("old_size=", old_sizes, " new_size=", new_sizes,
            " old_stl=", old_stl.toString(), " new_stl=", res.toString())
  return res;
}

static inline at::Tensor spyre_view_impl(const at::Tensor& self,
                                         c10::IntArrayRef size) {
  c10::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  SpyreTensorLayout old_stl =
      static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl())->spyre_layout;
  SpyreTensorLayout new_stl =
      compute_view_layout(self.sizes(), inferred_size, old_stl);
  return spyre_alias_with_sizes_and_strides(self, inferred_size, *stride,
                                            new_stl);
}

at::Tensor spyre_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

at::Tensor spyre__unsafe_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("view", TORCH_FN(spyre_view));
  m.impl("_unsafe_view", TORCH_FN(spyre__unsafe_view));
}

}  // namespace spyre
