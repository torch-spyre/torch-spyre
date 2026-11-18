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

#include "spyre_mem.h"

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_composite_address.h"
#include "spyre_storage_impl.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace py = pybind11;

namespace spyre {

namespace {

// Contiguous (row-major) element strides over the device sizes.
std::vector<int64_t> contiguous_device_strides(const SpyreTensorLayout& stl) {
  std::vector<int64_t> strides(stl.device_size.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(strides.size()) - 1; i >= 0; i--) {
    strides[i] = running;
    running *= stl.device_size[i];
  }
  return strides;
}

/*
 * Fills out size and strides for each dimension of the tensor.
 *
 * The number of elements moved along each device dim comes from
 * stl.valid_elements:
 *   - singleton {j}: min(count, device_size[j]) when stride_map[j] > 0;
 *   - stick pair {outer, inner} holding V elements: inner moves
 *     min(V, inner_size) and outer moves V / inner_size full rows. When
 *     V % inner_size != 0 the last, partial row is a second descriptor.
 * A dim with stride_map == 0 (a broadcast) is not capped by its key and moves
 * its full device extent.
 *
 * @param cpu_offset: storage offset of the CPU tensor
 * @param device_offset: storage offset of the dev tensor
 * @param stl: SpyreTensorLayout of dev tensor
 * @param host2device: direction of data conversion
 * @return description of data conversion
 */
auto get_device_stride_infos(int64_t cpu_offset, int64_t device_offset,
                             const SpyreTensorLayout& stl, bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  for (const int64_t size : stl.device_size) {
    TORCH_CHECK(size >= 0, "Invalid device size ", size);
  }
  const std::vector<int64_t> device_strides = contiguous_device_strides(stl);
  // Every stride is non-negative after construction. A zero stride is a
  // broadcast or a size-1 dim.
  const std::vector<int64_t>& host_strides = stl.stride_map;
  std::vector<int64_t> sizes = stl.device_size;

  for (const auto& [dims, count] : stl.valid_elements) {
    if (dims.size() != 1) continue;
    const int64_t j = dims[0];
    if (stl.stride_map[j] > 0) sizes[j] = std::min(count, stl.device_size[j]);
  }

  // The first stick pair with a partial last row, if any: that row covers one
  // outer index and `count % inner_size` inner elements, after the full rows.
  std::optional<StickPair> partial_pair;
  for (const StickPair& pair : stick_pairs(stl)) {
    const int64_t inner_size = stl.device_size[pair.inner];
    TORCH_CHECK(inner_size > 0, "Invalid device size ", inner_size,
                " for device dimension ", pair.inner);
    sizes[pair.inner] = std::min(pair.count, inner_size);
    sizes[pair.outer] = pair.count / inner_size;
    if (pair.count % inner_size != 0) {
      TORCH_CHECK(!partial_pair.has_value(),
                  "valid_elements has more than one partial stick row");
      partial_pair = pair;
    }
  }

  // Builds one descriptor. The descriptors list the innermost device dim
  // first, the reverse of PyTorch ordering.
  const auto make_info = [&](std::vector<int64_t> info_sizes,
                             int64_t host_offset, int64_t device_off) {
    DataConversionStrideInfo info;
    info.size_ = std::move(info_sizes);
    info.stride_src_ = host2device ? host_strides : device_strides;
    info.stride_dst_ = host2device ? device_strides : host_strides;
    info.offset_src_ = host2device ? host_offset : device_off;
    info.offset_dst_ = host2device ? device_off : host_offset;
    std::reverse(info.size_.begin(), info.size_.end());
    std::reverse(info.stride_src_.begin(), info.stride_src_.end());
    std::reverse(info.stride_dst_.begin(), info.stride_dst_.end());
    return info;
  };

  std::vector<DataConversionStrideInfo> stride_infos;
  // With no full row (valid count below one stick) only the partial row moves.
  const bool has_full_rows = std::none_of(sizes.begin(), sizes.end(),
                                          [](int64_t s) { return s == 0; });
  if (has_full_rows) {
    stride_infos.push_back(make_info(sizes, cpu_offset, device_offset));
  }

  if (partial_pair) {
    const int64_t full_rows = sizes[partial_pair->outer];
    std::vector<int64_t> partial_sizes = sizes;
    partial_sizes[partial_pair->inner] =
        partial_pair->count % stl.device_size[partial_pair->inner];
    partial_sizes[partial_pair->outer] = 1;
    stride_infos.push_back(make_info(
        std::move(partial_sizes),
        cpu_offset + full_rows * host_strides[partial_pair->outer],
        device_offset + full_rows * device_strides[partial_pair->outer]));
  }
  return stride_infos;
}

// Element offset into the device_size coordinate space for a tensor whose
// storage offset is `storage_offset` elements.
int64_t device_offset_for_storage_offset(const SpyreTensorLayout& stl,
                                         int64_t storage_offset) {
  const int device_rank = static_cast<int>(stl.stride_map.size());
  const std::vector<int64_t> device_strides = contiguous_device_strides(stl);
  const int64_t device_elements = c10::multiply_integers(stl.device_size);

  // Walk the device dims from the coarsest host stride down, peeling off whole
  // multiples of each stride. Size-1 and non-positive-stride dims carry no
  // offset; of two dims sharing a stride only the stick (last) dim may be > 1.
  std::map<int64_t, int, std::greater<int64_t>> stride_map_to_j;
  for (int i = 0; i < device_rank; i++) {
    if (stl.stride_map[i] <= 0 || stl.device_size[i] <= 1) continue;
    if (stride_map_to_j.contains(stl.stride_map[i])) {
      TORCH_CHECK(i == (device_rank - 1),
                  "Invalid device sizes and stride map. Expected only 1 non-1 ",
                  "device_size value for stride_map value ", stl.stride_map[i],
                  "got: ", stl.device_size[i], " and ",
                  stl.device_size[stride_map_to_j[stl.stride_map[i]]]);
      continue;
    }
    stride_map_to_j.insert({stl.stride_map[i], i});
  }

  int64_t device_offset = 0;
  int64_t remaining = storage_offset;
  for (const auto& [stride, index] : stride_map_to_j) {
    if (stride > remaining) continue;
    const int64_t slice = remaining / stride;
    device_offset += slice * device_strides[index];
    remaining -= slice * stride;
    if (remaining <= 0) break;
  }

  TORCH_CHECK(remaining == 0, "Invalid device tensor storage offset");
  TORCH_CHECK(device_offset < device_elements, "Invalid device storage offset");
  return device_offset;
}

// Device extent at each positive host stride of `stl`: the largest device_size
// among the device dims sharing that stride.
std::map<int64_t, int64_t> device_extent_by_stride(
    const SpyreTensorLayout& stl) {
  std::map<int64_t, int64_t> extent;
  for (size_t k = 0; k < stl.stride_map.size(); k++) {
    const int64_t stride = stl.stride_map[k];
    if (stride <= 0) continue;
    auto [it, inserted] = extent.emplace(stride, stl.device_size[k]);
    if (!inserted) it->second = std::max(it->second, stl.device_size[k]);
  }
  return extent;
}

// What cap_valid_elements_to_view needs to know about a view.
struct ViewStrides {
  std::set<int64_t> used;  // positive strides of the view
  // Layout stride -> view size. Equal strides keep the larger size, since
  // unsqueeze(-1) repeats the inner stride on a size-1 dim.
  std::map<int64_t, int64_t> size_at_stride;
  int64_t numel = 1;
};

ViewStrides analyze_view_strides(
    const std::map<int64_t, int64_t>& layout_extent,
    const std::vector<int64_t>& view_sizes,
    const std::vector<int64_t>& view_strides) {
  ViewStrides view;
  for (const int64_t size : view_sizes) view.numel *= size;
  for (size_t i = 0; i < view_sizes.size(); i++) {
    const int64_t stride = view_strides[i];
    if (stride <= 0) continue;
    view.used.insert(stride);
    if (!layout_extent.contains(stride)) continue;
    // A view dim above a stride the layout does not have (b.view(64, 8, 512)
    // adds stride 4096 over layout stride 512) leaves the layout's count.
    const bool foreign_dim_above =
        std::ranges::any_of(view_strides, [&](int64_t other) {
          return other > stride && !layout_extent.contains(other);
        });
    if (foreign_dim_above) continue;
    auto [it, inserted] = view.size_at_stride.emplace(stride, view_sizes[i]);
    if (!inserted) it->second = std::max(it->second, view_sizes[i]);
  }
  return view;
}

// True when the view holds every layout stride at its full device extent, so
// it is the whole allocation and no valid element lies outside it.
bool view_covers_layout(const std::map<int64_t, int64_t>& layout_extent,
                        const std::vector<int64_t>& view_sizes,
                        const std::vector<int64_t>& view_strides) {
  return std::ranges::all_of(layout_extent, [&](const auto& stride_extent) {
    for (size_t i = 0; i < view_sizes.size(); i++) {
      if (view_strides[i] == stride_extent.first &&
          view_sizes[i] == stride_extent.second) {
        return true;
      }
    }
    return false;
  });
}

// The count a valid_elements key keeps in `view`. `finest` is the key's finest
// positive stride_map value (for a stick pair, the host stick stride).
int64_t capped_count(int64_t count, int64_t finest, const ViewStrides& view) {
  if (view.used.contains(finest)) {
    // The view has a dim at this stride: it limits the count to its size.
    const auto size = view.size_at_stride.find(finest);
    return size != view.size_at_stride.end() ? std::min(count, size->second)
                                             : count;
  }
  // The view has no dim at this stride. A stick key (finest stride 1) is then
  // indexed away. Any other key is still covered when the view holds the
  // count * finest host elements it spans (a flattened view walks through the
  // device dims below its innermost contiguous run); otherwise it is indexed
  // away and one element (or row) remains.
  if (finest == 1 || view.numel < count * finest) return 1;
  return count;
}

// Caps each valid_elements count of `stl` to what the view (view_sizes,
// view_strides) covers: a slice, or a view into a padded allocation, moves
// fewer elements than the layout describes. A key is matched to a view dim
// through its finest positive stride_map value.
void cap_valid_elements_to_view(SpyreTensorLayout& stl,
                                const std::vector<int64_t>& view_sizes,
                                const std::vector<int64_t>& view_strides) {
  const std::map<int64_t, int64_t> layout_extent = device_extent_by_stride(stl);
  if (view_covers_layout(layout_extent, view_sizes, view_strides)) return;
  const ViewStrides view =
      analyze_view_strides(layout_extent, view_sizes, view_strides);

  for (auto& [dims, count] : stl.valid_elements) {
    std::optional<int64_t> finest;
    for (const int64_t d : dims) {
      const int64_t stride = stl.stride_map[d];
      if (stride > 0 && (!finest || stride < *finest)) finest = stride;
    }
    if (finest) count = capped_count(count, *finest, view);
  }
}

// Maps each device stride that equals a view stride to the host stride of that
// view dim (the smaller of the two). Dims of size 1 and sparse or broadcast
// dims are left alone. Shared by the H2D and D2H descriptor paths.
void rebase_view_strides(std::vector<int64_t>& stride_map,
                         const SpyreTensorLayout& stl,
                         const std::vector<int64_t>& cpu_strides,
                         const std::vector<int64_t>& view_strides) {
  for (size_t i = 0; i < stride_map.size(); i++) {
    if (stl.stride_map[i] < 1 || stl.device_size[i] == 1) continue;
    for (size_t k = 0; k < view_strides.size(); k++) {
      if (view_strides[k] == stl.stride_map[i] && cpu_strides[k] > 0) {
        stride_map[i] = std::min(stl.stride_map[i], cpu_strides[k]);
        break;
      }
    }
  }
}

// Rewrites stride_map (and dst_strides, the host-side strides) for a host
// tensor that is an expanded view of the device tensor. Expanded host dims
// (cpu stride 0) are broadcast: each device dim with that view stride becomes
// stride 0, and the device strides above it shrink by the expand factor.
// valid_elements is not adjusted here; get_device_stride_infos moves the full
// extent of the zeroed (broadcast) dims.
void apply_expands(std::vector<int64_t>& stride_map,
                   std::vector<int64_t>& dst_strides,
                   const SpyreTensorLayout& stl,
                   const std::vector<int64_t>& cpu_sizes,
                   const std::vector<int64_t>& cpu_strides,
                   const std::vector<int64_t>& view_strides) {
  std::map<int64_t, int64_t> expands;
  for (size_t i = 0; i < cpu_sizes.size(); i++) {
    if (cpu_strides[i] != 0) continue;
    // The expand factor is later used as a divisor; zero would trap the
    // process (SIGFPE), which cannot be caught, so reject it here.
    TORCH_CHECK(cpu_sizes[i] > 0, "Invalid expanded host dimension of size ",
                cpu_sizes[i]);
    expands.insert({view_strides[i], cpu_sizes[i]});
    for (size_t j = 0; j < stl.stride_map.size(); j++) {
      if (stl.stride_map[j] == view_strides[i]) stride_map[j] = 0;
    }
  }
  for (size_t i = 0; i < stride_map.size(); i++) {
    if (stride_map[i] <= 0) continue;
    for (const auto& [stride, expand] : expands) {
      if (stl.stride_map[i] > stride) stride_map[i] /= expand;
    }
  }
  for (size_t i = 0; i < view_strides.size(); i++) {
    for (const auto& [stride, expand] : expands) {
      if (view_strides[i] > stride) dst_strides[i] /= expand;
    }
  }
}

// Slice multipliers for the host dims that are sliced relative to the view.
// Visits host dims from the finest cpu stride outward and returns, per device
// stride, the factor by which device strides at or above it scale.
std::multimap<int64_t, int64_t> slice_factors(
    const std::vector<int64_t>& cpu_sizes,
    const std::vector<int64_t>& cpu_strides,
    const std::vector<int64_t>& dst_strides) {
  std::vector<int64_t> cpu_order(cpu_sizes.size());
  std::iota(cpu_order.begin(), cpu_order.end(), 0);
  std::sort(cpu_order.begin(), cpu_order.end(), [&](int64_t i1, int64_t i2) {
    if (cpu_strides[i1] == cpu_strides[i2]) {
      if (cpu_sizes[i1] == 1 && cpu_sizes[i2] == 1) return i1 < i2;
      return cpu_sizes[i1] == 1;
    }
    return cpu_strides[i1] < cpu_strides[i2];
  });

  std::multimap<int64_t, int64_t> slices;
  int64_t current_slices = 1;
  for (const int64_t i : cpu_order) {
    const int64_t dev_stride = dst_strides[i];
    TORCH_CHECK(dev_stride > 0,
                "Invalid destination stride. Expected > 0, got: ", dev_stride);
    const int64_t slice = cpu_strides[i] / dev_stride / current_slices;
    if (slice > 1) {
      slices.insert({dev_stride, slice});
      current_slices *= slice;
    }
  }
  return slices;
}

// Scales each live device stride by the slice factors. The comparison is
// against the pre-slice snapshot (`unsliced`), not the map being scaled, so one
// slice does not change which dims the next one hits.
void apply_slices(std::vector<int64_t>& stride_map,
                  const std::multimap<int64_t, int64_t>& slices) {
  const std::vector<int64_t> unsliced = stride_map;
  for (size_t i = 0; i < stride_map.size(); i++) {
    if (stride_map[i] <= 0) continue;
    for (const auto& [stride, slice] : slices) {
      if (unsliced[i] >= stride) stride_map[i] *= slice;
    }
  }
}

// Returns stl.stride_map re-expressed in the strides of a host source tensor
// that differs from the device view by expansion or slicing.
std::vector<int64_t> stride_map_for_host_strides(
    const SpyreTensorLayout& stl, const std::vector<int64_t>& cpu_sizes,
    const std::vector<int64_t>& cpu_strides,
    const std::vector<int64_t>& view_strides) {
  std::vector<int64_t> stride_map = stl.stride_map;
  std::vector<int64_t> dst_strides = view_strides;

  rebase_view_strides(stride_map, stl, cpu_strides, view_strides);
  apply_expands(stride_map, dst_strides, stl, cpu_sizes, cpu_strides,
                view_strides);
  apply_slices(stride_map, slice_factors(cpu_sizes, cpu_strides, dst_strides));
  return stride_map;
}

// Descriptor for the FP8 [2, 64]-stick layout (QFP8WT), whose stick is
// interleaved in a way the generic stride description cannot express. Fills
// the DCI's descriptors and shapes. `cpu_sizes` is in PyTorch order [K, N].
void fill_qfp8wt_dci(DataConversionInfo& dci, int64_t elems_per_stick,
                     std::vector<int64_t> cpu_sizes,
                     std::vector<int64_t> device_sizes, int64_t cpu_offset,
                     bool host2device) {
  std::reverse(cpu_sizes.begin(), cpu_sizes.end());
  std::reverse(device_sizes.begin(), device_sizes.end());

  const int64_t si = 2;
  const int64_t so = elems_per_stick / si;
  // After the reverse, cpu_sizes = [N, K]: N columns, K rows.
  const int64_t N = cpu_sizes[0];
  const int64_t K = cpu_sizes[1];

  // Expanded device shape: [si, so, K/si, N/so]
  TORCH_CHECK(K % si == 0, "QFP8WT K=", K, " must be divisible by si=", si);
  TORCH_CHECK(N % so == 0, "QFP8WT N=", N, " must be divisible by so=", so);
  const int64_t dim2 = K / si;
  const int64_t dim3 = N / so;

  // Element [k, n] with k = d2*si + a, n = d3*so + b sits at host offset
  // a*N + b + d2*si*N + d3*so, and at device offset
  // a + b*si + d2*elems_per_stick + d3*(dim2*elems_per_stick): one stick holds
  // si x so elements, one N-strip holds dim2 sticks.
  const std::vector<int64_t> host_strides = {N, 1, si * N, so};
  const std::vector<int64_t> device_strides = {1, si, elems_per_stick,
                                               dim2 * elems_per_stick};

  DataConversionStrideInfo dcsi;
  dcsi.size_ = {si, so, dim2, dim3};
  dcsi.stride_src_ = host2device ? host_strides : device_strides;
  dcsi.stride_dst_ = host2device ? device_strides : host_strides;
  dcsi.offset_src_ = host2device ? cpu_offset : 0;
  dcsi.offset_dst_ = host2device ? 0 : cpu_offset;
  dci.dcsi_ = {dcsi};
  // H2D produces the expanded 4D device shape; D2H produces the 2D host shape.
  dci.output_shape_ = host2device ? dcsi.size_ : cpu_sizes;
  dci.input_shape_ = host2device ? cpu_sizes : device_sizes;
}

}  // namespace

// Validates the dtypes of a transfer and fills the source and destination
// data formats of `dci`. Refuses any combination the backend cannot convert.
void set_dtype_formats(DataConversionInfo& dci, const at::Tensor& cpu_tensor,
                       const at::Tensor& dev_tensor,
                       const SpyreTensorLayout& stl, bool host2device) {
  const auto cpu_str_type = torchScalarToString[cpu_tensor.scalar_type()];
  const auto dev_str_type = torchScalarToString[dev_tensor.scalar_type()];
  const auto [cpu_format_host, cpu_format_dev] =
      stringToDTDataFormatPair(cpu_str_type);
  TORCH_CHECK(cpu_format_host != DataFormats::INVALID &&
                  cpu_format_dev != DataFormats::INVALID,
              "Unsupported CPU tensor dtype for DMA transfer: ", cpu_str_type);

  // stl.device_dtype is the authoritative on-device element format.  For bool
  // tensors it may differ from the type-map default (SEN169_FP16): an fp32
  // comparison result is physically stored as IEEE_FP32 (32 elems/stick).
  //
  // IEEE_FP32 bool tensors are on-device intermediates produced by the
  // compiler; they are always read D2H and never written H2D.  Guard against
  // that assumption being violated so a future regression fails loudly rather
  // than silently transferring data with the wrong element size.
  TORCH_CHECK(
      !(host2device && dev_tensor.scalar_type() == c10::ScalarType::Bool &&
        stl.device_dtype == DataFormats::IEEE_FP32),
      "Unexpected H2D transfer to a bool tensor with IEEE_FP32 device format; "
      "fp32-format bool tensors are on-device intermediates and should never "
      "be H2D transfer destinations");
  dci.dci_dsName_ = "DCI-Tensor-0";
  dci.isHostToSen_ = host2device;
  dci.dataformat_src_ = host2device ? cpu_format_host : stl.device_dtype;
  dci.dataformat_dst_ = host2device ? stl.device_dtype : cpu_format_host;
  // deeptools reads/writes the host buffer as storage_bytes /
  // hostFormatBytes(cpu_format_host). When that width disagrees with PyTorch's
  // element size the access is out of bounds, and the resulting ConvertData
  // throw lands on the stream worker and shuts the stream down. Refuse up
  // front. quint8 still trips this: SENUINT32 claims 4 bytes for a 1-byte
  // buffer. Allocation is untouched: torch.distributed allocates uint8.
  const size_t host_bytes = hostFormatBytes(cpu_format_host);
  const size_t torch_bytes = c10::elementSize(cpu_tensor.scalar_type());
  TORCH_CHECK(host_bytes == 0 || host_bytes == torch_bytes,
              "Spyre backend does not support dtype ", cpu_tensor.scalar_type(),
              " for host transfers: its host data format declares ", host_bytes,
              " byte(s) per element but PyTorch stores ", torch_bytes,
              ". See https://github.com/torch-spyre/torch-spyre/issues/1588");
  TORCH_CHECK(
      isDCIConversionSupported(dci.dataformat_src_, dci.dataformat_dst_),
      "Unsupported DCI data format conversion: src=",
      static_cast<int>(dci.dataformat_src_),
      " dst=", static_cast<int>(dci.dataformat_dst_),
      " (cpu_type=", cpu_str_type, ", dev_type=", dev_str_type, ")");
}

/*
 * Generate description of data conversion for a tensor.
 *
 * @param cpu_tensor: CPU-side tensor (source for H2D, destination for D2H)
 * @param dev_tensor: device-side tensor (destination for H2D, source for D2H)
 * @param stl: SpyreTensorLayout of the device tensor
 * @param host2device: direction of the transfer
 * @return data conversion information
 */
auto generate_dci(const at::Tensor* cpu_tensor, const at::Tensor* dev_tensor,
                  SpyreTensorLayout stl, bool host2device)
    -> DataConversionInfo {
  DataConversionInfo dci{};
  set_dtype_formats(dci, *cpu_tensor, *dev_tensor, stl, host2device);

  const std::vector<int64_t> cpu_sizes = cpu_tensor->sizes().vec();
  const std::vector<int64_t> cpu_strides = cpu_tensor->strides().vec();
  const std::vector<int64_t> view_sizes = dev_tensor->sizes().vec();
  const std::vector<int64_t> view_strides = dev_tensor->strides().vec();

  // The logic below assumes the host and device tensors have the same sizes;
  // only their strides may differ (non-dense or overlapping source).
  TORCH_CHECK(cpu_sizes == view_sizes,
              "Invalid device sizes for host sizes. Expected: ", cpu_sizes,
              ", got: ", view_sizes);

  const int64_t cpu_offset = cpu_tensor->storage_offset();
  const int64_t device_offset =
      device_offset_for_storage_offset(stl, dev_tensor->storage_offset());

  // `stl` is this call's copy; the tensor's own layout is untouched.
  cap_valid_elements_to_view(stl, view_sizes, view_strides);
  if (cpu_strides != view_strides) {
    if (host2device) {
      stl.stride_map = stride_map_for_host_strides(stl, cpu_sizes, cpu_strides,
                                                   view_strides);
    } else {
      // D2H applies only the view-stride rebase: the staged buffer is the
      // stored allocation, so the expand and slice rescaling is not needed.
      rebase_view_strides(stl.stride_map, stl, cpu_strides, view_strides);
    }
  }

  if (stl.element_arrangement == ElementArrangement::QFP8WT) {
    fill_qfp8wt_dci(dci, stl.elems_per_stick(), cpu_sizes, stl.device_size,
                    cpu_offset, host2device);
  } else {
    dci.dcsi_ =
        get_device_stride_infos(cpu_offset, device_offset, stl, host2device);

    // Shapes list the innermost dim first (reverse of PyTorch ordering).
    const std::vector<int64_t> cpu_shape(cpu_sizes.rbegin(), cpu_sizes.rend());
    const std::vector<int64_t> device_shape(stl.device_size.rbegin(),
                                            stl.device_size.rend());
    dci.input_shape_ = host2device ? cpu_shape : device_shape;
    dci.output_shape_ = host2device ? device_shape : cpu_shape;
  }
  if (SPYRE_LOG_ENABLED("spyre.runtime",
                        torch_spyre::logging::LogLevel::DEBUG)) {
    std::stringstream s;
    dci.exportJson(s);
    SPYRE_RUNTIME_DEBUG() << "DataConversionInfo: " << s.str();
  }
  return dci;
}

/*
 * True when a DCI describes a pure memcpy: same source and destination
 * element format, a single descriptor, identical src/dst strides and zero
 * offsets, and strides that densely cover the element range (each
 * non-degenerate dimension sits exactly at the running element count, so
 * there are no gaps and no stick padding). Such a conversion moves the same
 * bytes a raw DMA would, so the caller can submit the copy without a DCI and
 * skip flex's host-side staging (deeptools::ConvertData) entirely.
 *
 * Anything else -- dtype conversion, strided/sliced views (non-zero offsets),
 * stickified or padded device layouts, specialized ops -- keeps the DCI.
 */
bool dci_is_identity_copy(const DataConversionInfo& dci) {
  if (dci.dataformat_src_ != dci.dataformat_dst_) return false;
  if (dci.dcOpName_ != DataConvertOpFuncs::GENERAL_SHUFFLE) return false;
  if (!dci.pre_input_shape_.empty() || !dci.post_output_shape_.empty())
    return false;
  if (!dci.pre_expand_dcsi_.empty() || !dci.post_slice_dcsi_.empty())
    return false;
  if (!dci.input_dimwise_ea_.empty() || !dci.output_dimwise_ea_.empty())
    return false;
  if (dci.useSpi_ || dci.usePca_ || dci.usePadVal_ || dci.useWli_) return false;
  if (dci.dcsi_.size() != 1) return false;

  const auto& dcsi = dci.dcsi_[0];
  // Non-zero (even if equal) offsets mean positions before the offset are
  // zero-filled by ConvertData but would hold stale bytes after a raw DMA;
  // keep the DCI for views with a storage offset.
  if (dcsi.offset_src_ != 0 || dcsi.offset_dst_ != 0) return false;
  if (dcsi.stride_src_ != dcsi.stride_dst_) return false;

  const auto& sizes = dcsi.size_;
  const auto& strides = dcsi.stride_src_;
  const size_t rank = sizes.size();
  if (rank == 0 || strides.size() != rank) return false;

  // Densely covering: ordered from the innermost (smallest) stride outward,
  // every non-degenerate dimension must start exactly where the inner
  // dimensions end. A padding gap or a broadcast stride fails this and keeps
  // the DCI.
  std::vector<std::pair<int64_t, int64_t>> dims;
  dims.reserve(rank);
  for (size_t i = 0; i < rank; i++) dims.emplace_back(strides[i], sizes[i]);
  std::sort(dims.begin(), dims.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  int64_t numel = 1;
  for (const auto& [stride, size] : dims) {
    if (size < 1) return false;
    if (size == 1) continue;
    if (stride != numel) return false;
    numel *= size;
  }

  // The covered elements must be the whole tensor, not a dense slice of it.
  int64_t volume = 1;
  for (const auto& dim : dci.output_shape_) volume *= dim;
  return volume == numel;
}

// Empty op needs C++ code and cannot be handled by python side fallback
// Bytes of allocation storage needed for a tensor of `size` and `dtype` on
// device: the larger of the device-layout bytes and the contiguous host bytes.
// The host bytes are overflow-checked; a huge shape throws instead of wrapping.
size_t allocation_bytes(const SpyreTensorLayout& device_layout,
                        c10::IntArrayRef size, c10::ScalarType dtype) {
  const size_t device_size_bytes = get_device_size_in_bytes(device_layout);
  const size_t cpu_size_bytes =
      at::detail::computeStorageNbytesContiguous(size, c10::elementSize(dtype));
  return std::max(device_size_bytes, cpu_size_bytes);
}

at::Tensor spyre_empty(c10::IntArrayRef size,
                       std::optional<c10::ScalarType> dtype_opt,
                       std::optional<c10::Layout> layout_opt,
                       std::optional<c10::Device> device_opt,
                       std::optional<bool> pin_memory_opt,
                       std::optional<c10::MemoryFormat> memory_format_opt) {
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  SPYRE_RUNTIME_DEBUG() << "shape=" << size << " on Spyre " << device;
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  const auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous ||
                  memory_format == c10::MemoryFormat::Preserve,
              "Spyre backend only supports contiguous memory format, got: ",
              memory_format);
  const c10::DeviceGuard device_guard(device);

  auto device_layout = SpyreTensorLayout(size.vec(), dtype);
  size_t size_bytes = allocation_bytes(device_layout, size, dtype);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(c10::make_intrusive<SpyreStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          &SpyreAllocator::instance(),
          /*resizeable=*/true)),
      pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_contiguous(size);
  spyre_tensor_impl->spyre_layout = device_layout;
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

/**
 * This method will determine the size of the tensor on Spyre, then allocate
 * that space on the Spyre and and set the handle for the tensor to that of the
 * memory in the Spyre. For now, it allocates a CPU tensor with the correct
 * size, as the actual storage will stay on CPU until the rest of the stack is
 * ready to filter out the allocation and deallocation of memory from the graph
 * processing.
 */
at::Tensor spyre_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                               std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt) {
  // SETUP FOR Spyre TENSOR
  at::detail::check_size_nonnegative(size);
  const auto scalar_type = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(spyre::is_supported_dtype(scalar_type),
              "Spyre backend does not support dtype ", scalar_type);
  caffe2::TypeMeta dtype = c10::scalarTypeToTypeMeta(scalar_type);
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  SPYRE_RUNTIME_DEBUG() << "Tensor info on CPU (Size:" << size
                        << ", Stride: " << stride << ", dtype: " << dtype
                        << ") to be mapped onto device " << device;
  auto device_layout = SpyreTensorLayout(size.vec(), stride.vec(), scalar_type,
                                         generic_stick_dim_order(size.size()));
  size_t size_bytes = allocation_bytes(device_layout, size, scalar_type);

  auto spyre_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      &SpyreAllocator::instance(),
      /*resizeable=*/true);
  auto spyre_storage = c10::Storage(spyre_storage_impl);

  // Create the Spyre Tensor
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      std::move(spyre_storage), pu1_dks, dtype);

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_and_strides(size, stride);

  spyre_tensor_impl->spyre_layout = device_layout;

  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

/**
 * Allocate a Spyre tensor sized/laid out for a ceiling (`max_size`) at
 * `dim` rather than for `size[dim]` itself. `size`/`stride` remain the
 * tensor's logical (PyTorch-visible) shape; only the SpyreTensorLayout and
 * the underlying storage are computed against the padded shape. Because
 * padding dim 0 (the outermost dimension) does not change stride_map (see
 * SpyreTensorLayout::init), the layout's valid_elements for `dim` is set to
 * `max_size`, so the layout is the same at every size in the declared range.
 * Only the real data is transferred, into the front of the larger buffer.
 *
 * Only dim 0 is supported. Padding any other non-stick dimension changes the
 * host stride of every dimension outside it, so stride_map (built from the
 * padded shape) would disagree with the real host strides and the
 * host-to-device copy would place data at the wrong offsets and can write
 * past the buffer. spyre_empty_reserved() rejects such a `dim`.
 */
at::Tensor spyre_empty_reserved(c10::IntArrayRef size, c10::IntArrayRef stride,
                                c10::ScalarType dtype, int64_t dim,
                                int64_t min_size, int64_t max_size,
                                int64_t granularity) {
  at::detail::check_size_nonnegative(size);
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  TORCH_CHECK(dim >= 0 && dim < static_cast<int64_t>(size.size()),
              "reserved dim ", dim, " out of range for tensor of rank ",
              size.size());
  TORCH_CHECK(granularity >= 1, "granularity must be >= 1, got ", granularity);
  TORCH_CHECK(min_size <= max_size, "min=", min_size,
              " must be <= max=", max_size, " at dim ", dim);
  TORCH_CHECK(max_size % granularity == 0, "max=", max_size,
              " is not a multiple of granularity=", granularity, " at dim ",
              dim,
              "; an unreachable ceiling would make every resize_ to max "
              "fail the per-resize granularity check");
  TORCH_CHECK(min_size <= size[dim] && size[dim] <= max_size, "current size ",
              size[dim], " at dim ", dim, " is not within [min=", min_size,
              ", max=", max_size, "]");
  TORCH_CHECK(size[dim] % granularity == 0, "current size ", size[dim],
              " at dim ", dim,
              " is not a multiple of granularity=", granularity);
  TORCH_CHECK(dim != static_cast<int64_t>(size.size()) - 1,
              "tensor.to(\"spyre\", dynamic={dim: {...}}) does not yet "
              "support reserving the innermost (stick) dimension ",
              dim,
              ": padding a stick dimension changes stride_map for every "
              "outer dimension, unlike a batch/outer dimension, so the "
              "DMA/guard logic implemented here does not apply");
  TORCH_CHECK(dim == 0,
              "tensor.to(\"spyre\", dynamic={dim: {...}}) does not yet "
              "support reserving dim ",
              dim,
              ": only dim 0 (the outermost dimension) is supported. Padding "
              "any other dimension changes the host stride of every "
              "dimension outside it, which the host-to-device copy does not "
              "handle");

  auto padded_size = size.vec();
  padded_size[dim] = max_size;
  auto device_layout = SpyreTensorLayout(padded_size, dtype);
  set_reserved_valid_elements(device_layout, dim, max_size);

  size_t device_size_bytes = get_device_size_in_bytes(device_layout);
  // Overflow-checked (throws), unlike a bare accumulate: a huge `max` must
  // not wrap size_bytes small and undersize the reservation.
  size_t cpu_size_bytes =
      at::detail::computeStorageNbytesContiguous(size, c10::elementSize(dtype));
  size_t size_bytes = std::max(device_size_bytes, cpu_size_bytes);

  c10::Device device =
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice();
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(c10::make_intrusive<SpyreStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          &SpyreAllocator::instance(),
          /*resizeable=*/true)),
      pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_and_strides(size, stride);
  spyre_tensor_impl->spyre_layout = device_layout;
  spyre_tensor_impl->reserved_dims = {
      {dim, ReservedDimInfo{min_size, max_size, granularity}}};
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout (reserved dim=" << dim
                        << " min=" << min_size << " max=" << max_size
                        << " granularity=" << granularity
                        << " reserved_bytes=" << size_bytes
                        << "): " << device_layout.toString();
  return tensor;
}

at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout,
                                   std::optional<c10::Device> device_opt) {
  at::detail::check_size_nonnegative(size);
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  TORCH_CHECK(device.is_privateuseone(),
              "spyre_empty_with_layout expected a Spyre device, got ", device);
  const c10::DeviceGuard device_guard(device);

  size_t size_bytes = allocation_bytes(device_layout, size, dtype);
  auto spyre_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      &SpyreAllocator::instance(),
      /*resizeable=*/true);
  auto spyre_storage = c10::Storage(spyre_storage_impl);

  // Create the Spyre Tensor
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      std::move(spyre_storage), pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_and_strides(size, stride);
  spyre_tensor_impl->spyre_layout = device_layout;
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

// A shard view's DataPtr context must be a SharedOwnerCtx* (the copy and
// compute paths read it through get_context()). The view owns a
// single-chunk CompositeAddress for one shard and a reference to the parent
// storage, which keeps the multi-chunk allocation alive. The SharedOwnerCtx
// points back at its enclosing view so the DataPtr deleter can free it.
struct ShardViewCtx {
  SharedOwnerCtx owner;
  c10::Storage parent;
  ShardViewCtx(flex::CompositeAddress addr, signed char device_id,
               c10::Storage parent_storage)
      : owner(std::move(addr), device_id, flex::MemoryType::Tensor),
        parent(std::move(parent_storage)) {}
};

namespace {

// The owner of a sharded tensor's storage: the DataPtr context with the
// sharding record present. Callers read ->sharding for the record itself.
SharedOwnerCtx* sharding_of(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(), "expected a Spyre tensor");
  auto* ctx =
      static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr && ctx->sharding.has_value(),
              "expected a sharded Spyre tensor");
  return ctx;
}

void delete_shard_view(void* owner_ctx) {
  auto* owner = static_cast<SharedOwnerCtx*>(owner_ctx);
  delete owner->shard_view;
}

}  // namespace

at::Tensor spyre_empty_sharded(c10::IntArrayRef size, c10::ScalarType dtype,
                               int64_t shard_dim, int64_t num_shards,
                               std::optional<c10::Device> device_opt) {
  at::detail::check_size_nonnegative(size);
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  TORCH_CHECK(device.is_privateuseone(),
              "spyre_empty_sharded expected a Spyre device, got ", device);
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  TORCH_CHECK(num_shards == 8 || num_shards == 16,
              "sharded tensors support 8 or 16 shards, got ", num_shards);
  const int64_t rank = static_cast<int64_t>(size.size());
  TORCH_CHECK(rank > 0, "sharded tensors must not be scalars");
  TORCH_CHECK(shard_dim >= 0 && shard_dim < rank, "shard_dim ", shard_dim,
              " is out of range for rank ", rank);
  TORCH_CHECK(size[shard_dim] > 0 && size[shard_dim] % num_shards == 0,
              "sharded dimension ", size[shard_dim],
              " must divide evenly into ", num_shards, " non-empty shards");
  const c10::DeviceGuard device_guard(device);

  std::vector<int64_t> shard_size = size.vec();
  shard_size[shard_dim] /= num_shards;
  SpyreTensorLayout shard_layout(shard_size, dtype);
  const uint64_t shard_device_bytes = get_device_size_in_bytes(shard_layout);

  c10::DataPtr data_ptr = SpyreAllocator::allocateShardedStorage(
      shard_dim, static_cast<uint32_t>(num_shards), shard_device_bytes);
  static_cast<SharedOwnerCtx*>(data_ptr.get_context())->sharding->global_shape =
      size.vec();
  static_cast<SharedOwnerCtx*>(data_ptr.get_context())->sharding->dtype = dtype;
  const size_t total_bytes =
      static_cast<SharedOwnerCtx*>(data_ptr.get_context())
          ->composite_addr.total_size();
  auto storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), static_cast<int64_t>(total_bytes),
      std::move(data_ptr), &SpyreAllocator::instance());

  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(std::move(storage_impl)), pu1_dks,
      c10::scalarTypeToTypeMeta(dtype));
  auto* impl = static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  impl->set_sizes_contiguous(size);
  // spyre_layout describes one shard; every shard uses the same layout.
  impl->spyre_layout = shard_layout;
  return tensor;
}

bool spyre_is_sharded(const at::Tensor& tensor) {
  if (!tensor.is_privateuseone() || !tensor.has_storage()) return false;
  auto* ctx =
      static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context());
  return ctx != nullptr && ctx->sharding.has_value();
}

uint64_t shard_device_address(const at::Tensor& sharded, int64_t shard_index,
                              int64_t expected_num_shards,
                              int64_t expected_shard_bytes) {
  // Launch-time check that the tensor is sharded the way the kernel was
  // compiled for: same shard count and per-shard device bytes.
  auto* owner = sharding_of(sharded);
  const auto& info = *owner->sharding;
  TORCH_CHECK(
      static_cast<int64_t>(info.num_shards) == expected_num_shards &&
          static_cast<int64_t>(info.shard_device_bytes) == expected_shard_bytes,
      "sharded tensor has ", info.num_shards, " shards of ",
      info.shard_device_bytes, " bytes, but the kernel was compiled for ",
      expected_num_shards, " shards of ", expected_shard_bytes, " bytes");
  TORCH_CHECK(shard_index >= 0 && shard_index < expected_num_shards,
              "shard index ", shard_index, " is out of range for ",
              expected_num_shards, " shards");
  // Shard ordinal indexes the shard table directly: shard i backs core i, in
  // an addressable plain allocation of at least one shard.
  TORCH_CHECK(static_cast<size_t>(shard_index) < owner->shard_addrs.size(),
              "sharded storage has no allocation for shard ", shard_index);
  flex::Chunk chunk = owner->shard_addrs[shard_index].chunks()[0];
  TORCH_CHECK(chunk.size >= info.shard_device_bytes, "chunk for shard ",
              shard_index, " is not an addressable chunk of one shard");
  chunk.size = info.shard_device_bytes;
  // Non-owning single-chunk address, used only for translation; the
  // tensor (held by the launch) keeps the allocation alive.
  return SpyreAllocator::instance().compositeAddressToDeviceAddress(
      flex::CompositeAddress(chunk));
}

at::Tensor& spyre_set_storage(at::Tensor& result, at::Storage storage,
                              int64_t storage_offset, c10::IntArrayRef size,
                              c10::IntArrayRef stride) {
  SPYRE_RUNTIME_DEBUG() << "set method";
  // Same hazard as resize_: reinterpreting sharded storage with new sizes
  // would leave the sharding record describing the old shape. A Spyre
  // storage's DataPtr context is always a SharedOwnerCtx (a shard view's
  // context points at its SharedOwnerCtx subobject, with no sharding).
  TORCH_CHECK(!spyre_is_sharded(result),
              "set_ is not supported on sharded Spyre tensors");
  if (storage.device().is_privateuseone() &&
      storage.data_ptr().get_context() != nullptr) {
    auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());
    TORCH_CHECK(!ctx->sharding.has_value(),
                "set_ is not supported on sharded Spyre storage");
  }
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

namespace {
/**
 * Derive the physical byte range of one KV page based on the block_id provided.
 */
flex::Range derive_kv_page_range(const at::Tensor& cache, size_t block_id) {
  const SpyreTensorLayout layout = get_spyre_tensor_layout(cache);
  const std::vector<int64_t>& device_size = layout.device_size;

  // Check logical shape
  TORCH_CHECK(cache.dim() == 4,
              "copy_kv_page_raw: expected a rank 4 KV cache, got rank ",
              cache.dim());

  // Check device shape
  TORCH_CHECK(device_size.size() == 4,
              "copy_kv_page_raw: expected a rank 4 device layout, got rank ",
              device_size.size(),
              ". A generic tiled layout is not a supported KV cache, pass a "
              "rank 4 row-major device_layout when allocating");

  const int64_t num_blocks = cache.size(0);
  const int64_t head_size = cache.size(3);
  const int64_t elems_per_stick = device_size[3];

  TORCH_CHECK(elems_per_stick > 0, "copy_kv_page_raw: invalid stick width");

  TORCH_CHECK(head_size % elems_per_stick == 0, "copy_kv_page_raw: head_size ",
              head_size, " is not a multiple of the stick width ",
              elems_per_stick,
              ", the device image is padded and one range cannot describe a "
              "page");

  // Strides must be row major over device_size, which is what makes a page one
  // contiguous interval. The default generated layout reorders dimensions.
  int64_t expected_stride = 1;
  for (int i = 3; i >= 0; i--) {
    TORCH_CHECK(layout.stride_map[i] == expected_stride,
                "copy_kv_page_raw: device layout is not row major at dim ", i,
                ", a page is not one contiguous interval");
    expected_stride *= device_size[i];
  }

  TORCH_CHECK(num_blocks > 0, "copy_kv_page_raw: cache has no pages");

  TORCH_CHECK(cache.is_contiguous(),
              "copy_kv_page_raw: cache must be logically contiguous");

  // Require the canonical KV cache layout:
  // logical [pages, X, Y, head_size] maps to
  // device [pages * X, Y, head_size / 64, 64] for fp16.
  // X/Y are slots/heads for slot-major, or heads/slots for head-major.
  const std::vector<int64_t> expected_device_size = {
      num_blocks * cache.size(1),
      cache.size(2),
      head_size / elems_per_stick,
      elems_per_stick,
  };

  TORCH_CHECK(
      device_size == expected_device_size,
      "copy_kv_page_raw: device layout does not match the full canonical "
      "KV cache shape; pass the full cache, not a partial view");

  TORCH_CHECK(cache.storage_offset() == 0,
              "copy_kv_page_raw: pass the full cache and a block_id, not a "
              "page view. Got storage_offset ",
              cache.storage_offset());
  TORCH_CHECK(block_id < static_cast<size_t>(num_blocks),
              "copy_kv_page_raw: block_id ", block_id, " out of range [0, ",
              num_blocks, ")");

  const size_t total_bytes = get_device_size_in_bytes(layout);
  const size_t page_bytes = total_bytes / static_cast<size_t>(num_blocks);
  const size_t page_offset = block_id * page_bytes;

  TORCH_CHECK(page_offset % flex::DEVICE_ALIGNMENT == 0 &&
                  page_bytes % flex::DEVICE_ALIGNMENT == 0,
              "copy_kv_page_raw: page range must be ", flex::DEVICE_ALIGNMENT,
              " byte aligned, got offset=", page_offset,
              " length=", page_bytes);

  return flex::Range(page_offset, page_bytes);
}
}  // namespace

void copy_kv_page_raw(const at::Tensor& cache, size_t block_id,
                      const flex::SharedPool& pool, size_t slot_id,
                      bool to_device) {
  TORCH_CHECK(cache.is_privateuseone(),
              "copy_kv_page_raw: cache must be a Spyre tensor, got one on ",
              cache.device());
  c10::Device device = cache.device();
  SpyreStream stream = getCurrentStream(device);

  const flex::CompositeAddress* composite_address =
      spyre::get_composite_address(cache);

  const flex::Range range = derive_kv_page_range(cache, block_id);

  // Owns params so they are destroyed even if launchCopyRaw() throws.
  std::unique_ptr<flex::CopyRawParams, decltype(&flex::destroyCopyRawParams)>
      params(flex::createCopyRawParams(&pool, slot_id, composite_address,
                                       to_device, range),
             &flex::destroyCopyRawParams);
  stream.launchCopyRaw(params.get());

  // Always wait: nothing keeps the pool mapped while a DMA is in flight, so
  // returning early would let the caller detach or destroy it mid-copy.
  stream.synchronize();
}

/**
 * This method handles copy between devices. When copying to Spyre, this method
 * marks the tensor to compute on Spyre, but continue to use CPU tensor for now
 * such that when we run an op on the tensor on the Spyre, it will have the
 * proper handle to the Spyre allocation
 */
namespace {

// A single-chunk, non-owning Spyre tensor over shard `shard_index` of a
// sharded tensor, with the shard-local shape and layout. It keeps the parent
// storage alive. Used by copy_sharded to copy one shard with the ordinary
// copy path.
at::Tensor spyre_shard_view(const at::Tensor& sharded, int64_t shard_index) {
  auto* owner = sharding_of(sharded);
  const auto& info = *owner->sharding;
  TORCH_CHECK(shard_index >= 0 && shard_index < info.num_shards, "shard index ",
              shard_index, " is out of range for ", info.num_shards, " shards");
  TORCH_CHECK(sharded.storage_offset() == 0 && sharded.is_contiguous(),
              "shard views require a contiguous base sharded tensor");
  // The shard table is indexed by shard ordinal by construction: shard i
  // backs core i. Exactly one shard's device bytes from the start of its
  // own allocation.
  TORCH_CHECK(static_cast<size_t>(shard_index) < owner->shard_addrs.size(),
              "sharded storage has no allocation for shard ", shard_index);
  flex::Chunk chunk = owner->shard_addrs[shard_index].chunks()[0];
  TORCH_CHECK(chunk.size >= info.shard_device_bytes, "chunk for shard ",
              shard_index, " is too small");
  chunk.size = info.shard_device_bytes;

  auto view_ctx = std::make_unique<ShardViewCtx>(
      flex::CompositeAddress(chunk), owner->device_id, sharded.storage());
  view_ctx->owner.shard_view = view_ctx.get();
  void* owner_ctx = static_cast<void*>(&view_ctx->owner);
  c10::DataPtr view_ptr(owner_ctx, owner_ctx, &delete_shard_view,
                        sharded.device());
  view_ctx.release();  // now owned by view_ptr's deleter

  auto storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      static_cast<int64_t>(info.shard_device_bytes), std::move(view_ptr),
      &SpyreAllocator::instance());
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto view = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(std::move(storage_impl)), pu1_dks, sharded.dtype());
  auto* impl = static_cast<SpyreTensorImpl*>(view.unsafeGetTensorImpl());
  std::vector<int64_t> shard_size = sharded.sizes().vec();
  shard_size[info.shard_dim] /= info.num_shards;
  impl->set_sizes_contiguous(shard_size);
  impl->spyre_layout =
      static_cast<SpyreTensorImpl*>(sharded.unsafeGetTensorImpl())
          ->spyre_layout;
  return view;
}

// Copy between a CPU tensor and a sharded Spyre tensor one shard at a time:
// each shard goes through the ordinary layout-aware copy into (or out of) a
// single-chunk view of its own chunk. Synchronous; the destination is valid
// only after every shard copy has completed.
void copy_sharded(const at::Tensor& self, const at::Tensor& dst) {
  const bool host2device = self.is_cpu() && spyre_is_sharded(dst);
  const at::Tensor& sharded = host2device ? dst : self;
  const at::Tensor& host = host2device ? self : dst;
  TORCH_CHECK(host.is_cpu(),
              "sharded Spyre tensors can only be copied to or from the CPU");
  TORCH_CHECK(sharded.storage_offset() == 0 && sharded.is_contiguous(),
              "sharded Spyre tensors only support whole-tensor copies");
  TORCH_CHECK(host.sizes() == sharded.sizes(),
              "copy shape mismatch: ", host.sizes(), " vs ", sharded.sizes());
  // Dtype-converting DMA is rejected: converting shard copies are untested
  // against the D2H conversion path, which bounds its host output by the
  // host_capacity the DMA factories set.
  TORCH_CHECK(host.dtype() == sharded.dtype(),
              "sharded Spyre tensors can only be copied without a dtype "
              "change (got ",
              sharded.dtype(), " -> ", host.dtype(), ")");
  const auto& info = *sharding_of(sharded)->sharding;
  TORCH_CHECK(sharded.sizes().equals(c10::IntArrayRef(info.global_shape)),
              "sharded Spyre tensors only support whole-tensor copies: tensor "
              "metadata no longer matches the sharded allocation (",
              sharded.sizes(), " vs ", info.global_shape, ")");
  // A same-size view(dtype) passes the identity view check but leaves a
  // tensor whose dtype disagrees with the recorded shard layout; reject it
  // before the copy misinterprets the bytes.
  TORCH_CHECK(sharded.scalar_type() == info.dtype,
              "sharded Spyre tensors only support whole-tensor copies: tensor "
              "dtype no longer matches the sharded allocation (",
              sharded.scalar_type(), " vs ", info.dtype, ")");
  const int64_t extent = sharded.size(info.shard_dim) / info.num_shards;

  SpyreStream stream = getCurrentStream(sharded.device());
  for (int64_t i = 0; i < static_cast<int64_t>(info.num_shards); ++i) {
    at::Tensor shard = spyre_shard_view(sharded, i);
    at::Tensor host_slice = host.narrow(info.shard_dim, i * extent, extent);
    if (host2device) {
      at::Tensor staging = host_slice.contiguous();
      stream.copyAsync(staging, shard);
      // staging must outlive the DMA; synchronize per shard.
      stream.synchronize();
    } else {
      at::Tensor staging = at::empty(shard.sizes(), host.options());
      stream.copyAsync(shard, staging);
      stream.synchronize();
      host_slice.copy_(staging);
    }
  }
}

}  // namespace

namespace {

// A dimension of the layout seen in host strides, for deciding whether a view
// can be read in place. All device dims sharing one host stride form one
// TileDim. For the [40, 60] tensor stored as device [8, 5, 64] with stride_map
// [300, 60, 1] there are three: stride 300 (size 8), 60 (size 5) and 1 (size
// 64, pitch 60: a row holds 60 valid elements).
struct TileDim {
  int64_t stride;  // host stride of the dim
  int64_t size;    // device extent (max over dims sharing the stride)
  // Host distance from one index of this dim to the next. For the stick dim
  // this is the valid row width (valid_elements), not the device stick size,
  // so a slice that spans a stick boundary is seen in host coordinates.
  int64_t pitch;
  // Host strides of the finer tiles inside this dim, ascending. A tile that
  // starts at one of these strides is split across device dims.
  std::vector<int64_t> tile_strides;
};

// Tile dims of `stl`, coarsest stride first.
std::vector<TileDim> tile_dims(const SpyreTensorLayout& stl) {
  const std::map<int64_t, int64_t> extent = device_extent_by_stride(stl);

  // Host row width of each stick pair: its valid count. The pitch of the dim
  // carrying the inner stride is that width. Other dims keep size * stride.
  std::map<int64_t, int64_t> row_width_of_stride;
  for (const StickPair& pair : stick_pairs(stl)) {
    const int64_t inner_stride = stl.stride_map[pair.inner];
    if (inner_stride > 0) row_width_of_stride[inner_stride] = pair.count;
  }

  std::vector<TileDim> dims;
  dims.reserve(extent.size());
  for (auto it = extent.rbegin(); it != extent.rend(); ++it) {
    const auto [stride, size] = *it;
    const auto width = row_width_of_stride.find(stride);
    const int64_t pitch =
        width != row_width_of_stride.end() ? width->second : stride * size;
    TileDim dim{stride, size, pitch, {}};
    for (const auto& stride_extent : extent) {
      const int64_t other = stride_extent.first;
      if (other >= stride && other < pitch) dim.tile_strides.push_back(other);
    }
    dims.push_back(std::move(dim));
  }
  return dims;
}

// True when a view with positive strides, smaller than its layout's stored
// elements, cannot be read in place: it starts inside a tile of a tiled dim and
// runs past the end of that tile, or it splits a layout dim into dims that
// do not cover all of it (a slice, or a view that drops part of the dim).
bool view_needs_staging(const std::vector<TileDim>& dims,
                        c10::IntArrayRef sizes, c10::IntArrayRef strides,
                        int64_t storage_offset) {
  // Per layout dim: the product of the view sizes that land in it, and
  // whether any of them is a plain split of the dim (not at a tile stride).
  std::vector<int64_t> covered(dims.size(), 1);
  std::vector<bool> split(dims.size(), false);
  for (size_t i = 0; i < sizes.size(); i++) {
    const int64_t stride = strides[i];
    const int64_t size = sizes[i];
    if (size == 1) continue;
    for (size_t j = 0; j < dims.size(); j++) {
      const int64_t pitch = dims[j].pitch;
      if (stride < dims[j].stride || stride >= pitch) continue;
      covered[j] *= size;
      const auto& tiles = dims[j].tile_strides;
      const auto tile = std::ranges::find(tiles, stride);
      if (tile == tiles.end()) {
        split[j] = true;
        break;
      }
      const int64_t tile_end =
          std::next(tile) != tiles.end() ? *std::next(tile) : pitch;
      const int64_t tile_size = tile_end / stride;
      const int64_t tile_offset =
          ((storage_offset % pitch) / stride) % tile_size;
      if (tile_offset != 0 && tile_offset + size > tile_size) return true;
      break;
    }
  }
  for (size_t j = 0; j < dims.size(); j++) {
    if (split[j] && covered[j] != dims[j].size) return true;
  }
  return false;
}

// True when two distinct indices of the view address the same element
// (as_strided, unfold). Dims are ordered by stride; each must start past the
// extent of the dims inside it.
bool view_overlaps_itself(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
  std::vector<std::pair<int64_t, int64_t>> dims;  // (stride, size)
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] > 1) dims.emplace_back(strides[i], sizes[i]);
  }
  std::sort(dims.begin(), dims.end());
  int64_t extent = 1;  // elements spanned by the finer dims
  for (const auto& [stride, size] : dims) {
    if (stride < extent) return true;
    extent = stride * (size - 1) + extent;
  }
  return false;
}

// True when D2H of `self` cannot DMA straight out of its allocation: it is
// expanded, it overlaps itself, or it is a slice that cannot be read in place.
// The whole allocation is then copied to the host and the view applied there.
bool needs_host_staging(const at::Tensor& self, const SpyreTensorLayout& stl) {
  const auto is_zero_or_negative = [](int64_t s) { return s < 1; };
  if (std::ranges::any_of(self.strides(), is_zero_or_negative)) {
    return true;  // expanded or broadcast
  }
  if (view_overlaps_itself(self.sizes(), self.strides())) return true;
  // A view covering every stored element is a plain DMA.
  if (valid_element_count(stl) == self.numel()) return false;
  // Only this last check needs the allocation geometry.
  return view_needs_staging(tile_dims(stl), self.sizes(), self.strides(),
                            self.storage_offset());
}

// D2H copy of `self` through a host copy of its whole allocation, with the view
// applied on the host. Always blocking: the host buffer is local, and the view
// is read from it before it is freed.
void copy_view_via_host_allocation(const at::Tensor& self,
                                   const at::Tensor& dst,
                                   const SpyreTensorLayout& stl,
                                   SpyreStream& stream) {
  // The layout's strides are host storage strides, as are the view's, so a flat
  // host buffer spanning the layout holds each element at its storage index and
  // the view applies to it directly. Spanning is the furthest host element any
  // device dim reaches, bounded by the storage.
  int64_t extent = 1;
  for (size_t j = 0; j < stl.stride_map.size(); j++) {
    if (stl.stride_map[j] > 0) {
      extent = std::max(extent, stl.device_size[j] * stl.stride_map[j]);
    }
  }
  extent =
      std::min<int64_t>(extent, self.storage().nbytes() / self.element_size());
  // The view must lie inside the staged span, or as_strided below reads past
  // the host buffer.
  int64_t reach = self.storage_offset() + 1;
  for (int64_t i = 0; i < self.dim(); i++) {
    reach += (self.size(i) - 1) * self.stride(i);
  }
  TORCH_CHECK(reach <= extent, "View reaches host element ", reach,
              " outside the staged layout span of ", extent, " elements");
  const at::Tensor device_alloc =
      at::as_strided(self, {extent}, {1}, /*storage_offset=*/0);
  const at::Tensor host_alloc = at::empty({extent}, dst.options());

  stream.copyAsync(device_alloc, host_alloc);
  stream.synchronize();
  const at::Tensor view_out = host_alloc.as_strided(
      self.sizes(), self.strides(), self.storage_offset());
  dst.copy_(view_out);
}

}  // namespace

at::Tensor spyre_copy_from(const at::Tensor& self, const at::Tensor& dst,
                           bool non_blocking) {
  if (spyre_is_sharded(self) || spyre_is_sharded(dst)) {
    // Sharded copies are blocking by contract; non_blocking is honored by
    // completing before return.
    copy_sharded(self, dst);
    return dst;
  }

  if (dst.is_privateuseone()) {
    SpyreStream stream = getCurrentStream(dst.device());
    stream.copyAsync(self, dst);
    if (!non_blocking) stream.synchronize();
    return dst;
  }

  SpyreStream stream = getCurrentStream(self.device());
  if (self.is_privateuseone()) {
    const auto& stl =
        static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl())->spyre_layout;
    if (needs_host_staging(self, stl)) {
      copy_view_via_host_allocation(self, dst, stl, stream);
      return dst;
    }
  }
  stream.copyAsync(self, dst);
  if (!non_blocking) stream.synchronize();
  return dst;
}

at::Tensor empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  SPYRE_RUNTIME_DEBUG() << "shape=" << size << " on Spyre " << device;
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  const auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous ||
                  memory_format == c10::MemoryFormat::Preserve,
              "Spyre backend only supports contiguous memory format, got: ",
              memory_format);
  const c10::DeviceGuard device_guard(device);

  size_t size_bytes = allocation_bytes(device_layout, size, dtype);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(c10::make_intrusive<SpyreStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          &SpyreAllocator::instance(),
          /*resizeable=*/true)),
      pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_contiguous(size);
  spyre_tensor_impl->spyre_layout = device_layout;
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

at::Tensor py_empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return empty_with_layout(size, device_layout, dtype_opt,
                           /*layout_opt=*/std::nullopt, device_opt,
                           pin_memory_opt, memory_format_opt);
}

// Checks a resize of a max-reserved tensor against its reservation and returns
// the layout size pinned at each reserved dim's max. Only reserved dims may
// change, the rank must be unchanged, and the new size must be a legal step.
std::vector<int64_t> pinned_reserved_layout_size(
    const SpyreTensorImpl& self_impl, const at::Tensor& self,
    c10::IntArrayRef size_int) {
  std::vector<int64_t> layout_size = size_int.vec();
  {
    // reserved_dims keys are absolute dim indices. If the shape, strides or
    // offset were changed in place since allocation (t_, squeeze_,
    // as_strided_, ...), a key can name a different axis than it did, and
    // every check below would validate the wrong dimension. Fail rather than
    // trust it; falling through as "unreserved" would reallocate instead.
    TORCH_CHECK(self_impl.has_intact_reservation(),
                "resize_ on a max-reserved Spyre tensor: its shape, strides "
                "or storage offset were changed in place after allocation "
                "(e.g. t_(), transpose_(), squeeze_(), as_strided_()), so the "
                "reservation no longer refers to the same axis. current "
                "shape=",
                self.sizes());
    // Scope today: at most one entry, so this loop runs once; the guard is
    // already written per-dim so a later second entry needs no changes here.
    for (const auto& [rdim, info] : self_impl.reserved_dims) {
      // The reservation's contract is "only reserved dims move": the buffer
      // was sized from the shape at allocation time with just `rdim` padded
      // to info.max, so every other dim must stay exactly what it currently
      // is, and the rank must be unchanged. Checking only
      // `rdim < size_int.size()` is not enough — e.g. resize_([56, 32] ->
      // [32]) keeps rdim=0 in bounds of the new 1-D shape while actually
      // collapsing away the reserved axis entirely.
      TORCH_CHECK(static_cast<int64_t>(size_int.size()) == self.dim(),
                  "resize_ on a max-reserved Spyre tensor (dim=", rdim,
                  ", max=", info.max,
                  ") must keep the same rank; current shape=", self.sizes(),
                  ", requested shape=", size_int);
      // Explicit bound before size_int[rdim] below: the rank check above
      // only says size_int matches self, not that rdim is a valid index.
      TORCH_CHECK(rdim >= 0 && rdim < static_cast<int64_t>(size_int.size()),
                  "resize_ on a max-reserved Spyre tensor: reserved dim ", rdim,
                  " is out of range for requested shape=", size_int);
      for (int64_t d = 0; d < static_cast<int64_t>(size_int.size()); d++) {
        if (d == rdim) continue;
        TORCH_CHECK(size_int[d] == self.sizes()[d],
                    "resize_ on a max-reserved Spyre tensor (dim=", rdim,
                    ") may only change the reserved dim; dim ", d,
                    " would change from ", self.sizes()[d], " to ",
                    size_int[d]);
      }
      TORCH_CHECK(size_int[rdim] >= info.min && size_int[rdim] <= info.max,
                  "resize_ to ", size_int[rdim], " at dim ", rdim,
                  " is outside the reservation [min=", info.min,
                  ", max=", info.max,
                  "] established by tensor.to(\"spyre\", dynamic=...)");
      TORCH_CHECK(size_int[rdim] % info.granularity == 0, "resize_ to ",
                  size_int[rdim], " at dim ", rdim,
                  " is not a multiple of the reservation's granularity=",
                  info.granularity);
      layout_size[rdim] = info.max;
    }
  }
  return layout_size;
}

// Builds the layout for `layout_size`, with each reserved dim's valid count set
// to its declared max, so the layout does not change across resize_.
SpyreTensorLayout layout_for_resize(const SpyreTensorImpl& self_impl,
                                    const std::vector<int64_t>& layout_size,
                                    c10::ScalarType dtype) {
  auto new_layout = SpyreTensorLayout(layout_size, dtype);
  for (const auto& [rdim, info] : self_impl.reserved_dims) {
    set_reserved_valid_elements(new_layout, rdim, info.max);
  }
  return new_layout;
}

const at::Tensor& spyre_resize_(
    const at::Tensor& self, c10::SymIntArrayRef size,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  auto size_int = c10::asIntArrayRefUnchecked(size);
  // Case 1: No-op.
  if (self.sizes() == size_int && self.is_contiguous()) {
    return self;
  }
  // In-place reinterpretation would leave the sharding record (shard dim,
  // count, per-shard bytes) describing the old shape, so per-shard copies
  // would gather the wrong rows; reallocation would silently convert the
  // tensor to unified storage. Fail closed instead.
  TORCH_CHECK(!spyre_is_sharded(self),
              "resize_ is not supported on sharded Spyre tensors");
  TORCH_CHECK(memory_format_opt != c10::MemoryFormat::Preserve,
              "aten::resize_ does not support MemoryFormat::Preserve");
  TORCH_CHECK(!memory_format_opt.has_value() ||
                  *memory_format_opt == c10::MemoryFormat::Contiguous,
              "aten::resize_ on Spyre only supports contiguous memory format");
  const auto dtype = c10::typeMetaToScalarType(self.dtype());
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);

  auto* self_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());

  // If this tensor was allocated via tensor.to("spyre", dynamic=...), its
  // storage/layout are already sized for each reserved dim's max.
  // Keep the layout pinned at that ceiling instead of recomputing it from
  // the concrete new size, so that:
  //  (a) growing or shrinking within [min, max] never reallocates,
  //      preserving the "single buffer, no realloc within bounds" contract
  //      the reservation exists for, and
  //  (b) the SpyreTensorLayout compared by the recompile guard stays
  //      identical across every concrete shape the reservation covers, so
  //      resizing never forces a spurious recompile.
  const std::vector<int64_t> layout_size =
      self_impl->reserved_dims.empty()
          ? size_int.vec()
          : pinned_reserved_layout_size(*self_impl, self, size_int);
  // Use STL device bytes (stick-padded) to determine if existing allocation
  // suffices.
  const SpyreTensorLayout new_layout =
      layout_for_resize(*self_impl, layout_size, dtype);
  const size_t new_device_bytes = get_device_size_in_bytes(new_layout);
  const size_t new_cpu_bytes =
      at::detail::computeStorageNbytesContiguous(size_int, self.itemsize());
  const size_t new_size_bytes = std::max(new_device_bytes, new_cpu_bytes);
  // Case 2: Same-numel or shrink — reinterpret storage in-place, no data
  // moved. Growth is also handled here (without reallocating) when the
  // tensor is max-reserved and the grown shape still fits the reservation.
  // Only valid when new last dim ≤ old last dim; otherwise D2H reads into
  // stick padding.
  const int64_t new_numel = c10::multiply_integers(size_int);
  // Trivially true on the reserved path: the innermost dim can never be
  // the reserved one (checked above in spyre_empty_reserved), and every
  // non-reserved dim is pinned by the per-dim checks above, so this only
  // actually discriminates on the non-reserved (plain resize_) path.
  const bool last_dim_ok = size_int.empty() || self.sizes().empty() ||
                           size_int.back() <= self.sizes().back();
  // Name is "reserved and growing", not "growing within the reservation":
  // the actual bound check already happened in the per-dim TORCH_CHECKs
  // above, and new_size_bytes <= self.storage().nbytes() below is what
  // keeps this safe. This flag only decides whether growth is eligible
  // for the no-realloc branch at all.
  const bool reserved_and_growing =
      !self_impl->reserved_dims.empty() && new_numel > self.numel();
  if (new_size_bytes <= self.storage().nbytes() &&
      (new_numel <= self.numel() || reserved_and_growing) && last_dim_ok) {
    self_impl->set_sizes_contiguous(size_int);
    self_impl->spyre_layout = new_layout;
    SPYRE_RUNTIME_DEBUG() << "to shape=" << size_int
                          << " layout=" << self_impl->spyre_layout.toString();
    return self;
  }
  // Expected unreachable: every in-bounds resize_ of a reserved tensor is
  // caught by the branch above, since new_size_bytes is always <= the
  // storage already sized for max. Kept as fail-closed defence in depth
  // rather than removed, in case that invariant is ever violated upstream.
  TORCH_CHECK(self_impl->reserved_dims.empty(),
              "resize_ on a max-reserved Spyre tensor would require "
              "reallocation for target shape=",
              size_int, ", which breaks the reservation invariant");
  // Case 3: Reallocate — D2H → CPU resize_ → H2D. Handles expand and any
  // reshape where the new last dim > old last dim (stick-layout incompatible).
  // TODO(kunuruabhishek): avoid round-trip once restickify supports
  // cross-layout D2D copies.
  at::Tensor cpu_buf = self.cpu();
  cpu_buf.resize_(size_int);
  auto new_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), new_size_bytes,
      &SpyreAllocator::instance(), /*resizeable=*/true);
  self_impl->set_storage_keep_dtype(c10::Storage(new_storage_impl));
  self_impl->set_sizes_contiguous(size_int);
  self_impl->spyre_layout = new_layout;
  at::_copy_from(cpu_buf, self, /*non_blocking=*/false);
  SPYRE_RUNTIME_DEBUG() << "expand to shape=" << size_int
                        << " layout=" << self_impl->spyre_layout.toString();
  return self;
}

at::Tensor spyre_fill_tensor(const at::Tensor& self, double value) {
  TORCH_CHECK(self.is_privateuseone(),
              "spyre_fill_tensor: tensor must be on spyre device");
  TORCH_CHECK(self.numel() > 0, "spyre_fill_tensor: cannot fill empty tensor");

  // Sharded storage is a table of per-shard allocations whose layout
  // describes one shard, so the checks below do not apply to it. Reject it
  // here with the sharding message; get_composite_address would too, but only
  // after the partial-view check below had misreported it.
  TORCH_CHECK(!spyre_is_sharded(self),
              "this operation does not support sharded Spyre tensors yet "
              "(sharded storage can only be copied to or from the CPU)");

  // The device-side MEMORY_FILL below writes every stick of the composite
  // address, i.e. the whole storage, so it is only correct when self covers
  // every element of the tensor that owns the storage, each exactly once. A
  // partial view (d[8:16], d[:, 32:], d[:8], ...) would otherwise silently
  // overwrite the elements outside it.
  //  - self must be non-overlapping and dense: a broadcast view such as
  //    d[:1].expand(64, 64) has the base's offset and element count but
  //    covers only row 0. A transpose (d.t()) is dense and still fills.
  //  - A tracked view is compared with its base: partial means a different
  //    storage offset or fewer elements than the base.
  //  - Anything else (a base, or a tensor sharing storage without view
  //    tracking, e.g. d[8:16].detach()) is compared with the elements its
  //    layout holds. For a reservation (to("spyre", dynamic=...)) the layout
  //    is sized for the reserved max, and the space beyond the current shape
  //    holds no other tensor's data, so the current shape is the whole
  //    tensor.
  //  - A layout whose valid_elements is unknown ({}) is not checked.
  // TODO(brunhe704): fill partial views that cover whole sticks with per-run
  // fills.
  TORCH_CHECK(self.is_non_overlapping_and_dense(),
              "fill_/zero_ of a broadcast or overlapping view of a Spyre "
              "tensor is not supported yet: the device fill writes every "
              "stick of the storage once (view: ",
              self.sizes(), " with strides ", self.strides(),
              "). Fill the underlying tensor, or a CPU copy, instead.");
  const auto* impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  int64_t whole_offset = 0;
  int64_t whole_elements = -1;  // -1: unknown, do not check
  if (self.is_view()) {
    const at::TensorBase& base = self._base();
    whole_offset = base.storage_offset();
    whole_elements = base.numel();
  } else if (impl->has_intact_reservation()) {
    whole_elements = self.numel();
  } else if (!impl->spyre_layout.valid_elements.empty()) {
    whole_elements = 1;
    for (const auto& [dims, count] : impl->spyre_layout.valid_elements) {
      whole_elements *= count;
    }
  }
  if (whole_elements >= 0) {
    TORCH_CHECK(
        self.storage_offset() == whole_offset && self.numel() == whole_elements,
        "fill_/zero_ of a partial view of a Spyre tensor is not supported "
        "yet: the "
        "device fill writes whole sticks of the storage, so it would also "
        "overwrite the elements outside the view (view: ",
        self.sizes(), " at storage offset ", self.storage_offset(),
        "; the whole tensor has ", whole_elements,
        " elements at storage offset ", whole_offset,
        "). Fill the whole tensor, or fill a CPU copy and copy it back.");
  }

  // Map torch dtype to DataFormats for the value->pattern conversion, which
  // flex::createFillAsyncParams performs internally.
  DataFormats dtype = get_device_dtype(self.scalar_type());

  // flex::createFillAsyncParams has no value->pattern conversion for the fp8
  // formats. Zero is the all-zero bit pattern in every format, and the fill
  // writes raw bytes, so a zero fill of fp8 storage can use any supported
  // format. A nonzero fp8 value would need an fp8 encoding, which we do not
  // have, so reject it here instead of failing inside the runtime.
  if (dtype == DataFormats::SEN143_FP8 || dtype == DataFormats::SEN152_FP8) {
    TORCH_CHECK(value == 0.0,
                "fill_ of a float8 Spyre tensor is only supported with the "
                "value 0 (the device fill has no fp8 pattern); got ",
                value);
    dtype = DataFormats::SEN169_FP16;
  }

  // Launch a device-side MEMORY_FILL DMA. createFillAsyncParams converts value
  // to the dtype-specific fill pattern; launchFillAsync submits it.
  auto* params = flex::createFillAsyncParams(get_composite_address(self), value,
                                             dtype, /*use_dmai=*/true);
  struct Guard {
    flex::FillAsyncParams* p;
    ~Guard() {
      flex::destroyFillAsyncParams(p);
    }
  } guard{params};
  SpyreStream stream;
  stream.launchFillAsync(params);

  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
  m.impl("resize_", TORCH_FN(spyre_resize_));
}

}  // namespace spyre
