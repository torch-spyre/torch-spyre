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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_sendnn_utils.h"
#include "spyre_storage_impl.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace py = pybind11;

namespace spyre {

/*
 * CPU stride for a dimension.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @param strides: dimension strides of the CPU tensor
 * @param device_sizes: dimesion sizes of dev tensor
 * @param stride_map: mapping of strides of the CPU tensor to sizes of dev
 *                    tensor
 * @return index in `strides` that the `stride_map` value corresponds to.
 */
auto get_dim_map(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                 c10::IntArrayRef device_sizes, c10::IntArrayRef stride_map)
    -> std::vector<int> {
  const int host_rank = strides.size();
  const int device_rank = stride_map.size();
  const int stick_dim_index = device_rank > 2 ? device_rank - 3 : 0;

  std::vector<int64_t> max_stride_le(device_rank, 0);
  std::vector<int> dim_map(device_rank, -1);

  for (int i = 0; i < host_rank; i++) {
    // Size 1 dimensions are ignored.
    if (sizes[i] == 1) continue;

    const int64_t hst = strides[i];

    // Expanded dimensions are ignored.
    if (hst == 0) continue;

    for (int j = 0; j < device_rank; j++) {
      // Size 1 dimensions are ignored.
      if (device_sizes[j] == 1) continue;

      const int64_t dst = stride_map[j];
      if (hst > max_stride_le[j] && hst <= dst) {
        max_stride_le[j] = hst;
        dim_map[j] = i;
      }
    }
  }

  if (dim_map[stick_dim_index] != -1) {
    dim_map[stick_dim_index] = dim_map[device_rank - 1];
  }

  return dim_map;
}

/* Generates the tile mapping between `strides` and `stride_map`.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @param strides: dimension strides of the CPU tensor
 * @param device_sizes: dimesion sizes of dev tensor
 * @param stride_map: mapping of strides of the CPU tensor to sizes of dev
 *                    tensor
 * @return ordered indices (from back-to-front) in `stride_map` that the
 *         `strides` value corresponds to
 */
auto get_tile_map(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                  c10::IntArrayRef device_sizes, c10::IntArrayRef stride_map)
    -> std::vector<std::vector<int>> {
  const std::vector<int> dim_map =
      get_dim_map(sizes, strides, device_sizes, stride_map);

  const int host_rank = strides.size();
  const int device_rank = stride_map.size();

  // Get the mapping of the indices of each dim in the dim map, ordered based
  // on increasing stride map value.
  //
  // Each pair in the inner vector comes in the form {stride, index}.
  //
  // For example:
  //   strides:       [320, 80, 1] ... which assumes sizes [*, 4, 80]
  //   device_sizes:  [4, 2, *, 64]
  //   stride_map:    [80, 64, 320, 1]
  //   dim_map:       [1, 2, 0, 2]
  //
  //   tile_pairs[0]: [(320, 2)]
  //   tile_pairs[1]: [(80, 0)]
  //   tile_pairs[2]: [(1, 3), (64, 1)]
  std::vector<std::map<int64_t, int>> tile_pairs(host_rank);

  const int stick_dim = dim_map[device_rank - 1];
  if (stick_dim != -1) {
    tile_pairs[stick_dim].insert({-1, device_rank - 1});
  }

  for (int i = device_rank - 2; i > -1; i--) {
    const int dim = dim_map[i];

    // Dimensions that do not appear in the dim map are ignored.
    if (dim == -1) continue;

    tile_pairs[dim].insert({stride_map[i], i});
  }

  // Reduce the tile pairs down to just the indices since the strides are no
  // longer needed now that mapping is ordered.
  //
  //   tile_pairs[0]: [(320, 2)]         ->  tile_map[0]: [2]
  //   tile_pairs[1]: [(80, 0)]          ->  tile_map[1]: [0]
  //   tile_pairs[2]: [(1, 3), (64, 1)]  ->  tile_map[2]: [3, 1]
  std::vector<std::vector<int>> tile_map(host_rank);
  for (int i = 0; i < host_rank; i++) {
    tile_map[i].reserve(tile_pairs[i].size());
    for (const auto& [stride, index] : tile_pairs[i]) {
      tile_map[i].push_back(index);
    }
  }

  return tile_map;
}

/*
 * Fills out size and strides for each dimension of the tensor.
 *
 * @param sizes: dimension sizes of the tensor
 * @param spyre_dma_strides: Spyre device tensor DMA strides (for structural
 * matching)
 * @param storage_offset: storage offset of the CPU tensor
 * @param stl: SpyreTensorLayout of dev tensor
 * @param host2device: direction of data conversion
 * @param cpu_tensor_strides: actual dimension strides of the CPU tensor
 * @return description of data conversion
 */
auto get_device_stride_infos(c10::IntArrayRef sizes,
                             c10::IntArrayRef spyre_dma_strides,
                             int64_t storage_offset, SpyreTensorLayout stl,
                             bool host2device,
                             c10::IntArrayRef cpu_tensor_strides)
    -> std::vector<DataConversionStrideInfo> {
  const std::vector<std::vector<int>> tile_map =
      get_tile_map(sizes, spyre_dma_strides, stl.device_size, stl.stride_map);

  const int host_rank = cpu_tensor_strides.size();
  const int device_rank = stl.stride_map.size();

  // The host strides based on stride_map, used for remainder calculation.
  std::vector<int64_t> host_strides(device_rank, 1);
  // The CPU layout strides (differs from stride_map for non-contiguous
  // tensors).
  std::vector<int64_t> cpu_layout_strides(device_rank, 1);
  // The device strides are always contiguous strides for device sizes.
  std::vector<int64_t> device_strides(device_rank, 1);
  // The sizes for the first DataConversionStrideInfo match the device sizes
  // except for dimensions with a remainder.
  std::vector<int64_t> dcsi_sizes(device_rank, 1);

  int64_t prev_size = 1;
  for (int i = device_rank - 1; i > -1; i--) {
    if (stl.stride_map[i] == 0) {
      dcsi_sizes[i] = stl.device_size[i];
    }
    device_strides[i] = prev_size;
    prev_size *= stl.device_size[i];
    // Size 1 dimensions are ignored.
    if (stl.stride_map[i] == -1) continue;
    host_strides[i] = stl.stride_map[i];
    cpu_layout_strides[i] = stl.stride_map[i];
  }

  // Map host stride to actual CPU stride via stride map.
  for (int dev_dim = 0; dev_dim < device_rank; dev_dim++) {
    int64_t stride_map_val = stl.stride_map[dev_dim];
    if (stride_map_val <= 0) continue;  // Skip non-data dims (like -1 or 0)

    for (int host_dim = 0; host_dim < host_rank; host_dim++) {
      if (spyre_dma_strides[host_dim] == stride_map_val) {
        // Update with CPU tensor stride.
        cpu_layout_strides[dev_dim] = cpu_tensor_strides[host_dim];
        break;
      }
    }
  }

  // The sizes for the subsequent DataConversionStrideInfo (remainders) match
  // the first DataConversionStrideInfo sizes except for dimensions with a
  // remainder.
  std::vector<std::vector<int64_t>> remainders;

  // The offsets for the host and device are at the start of each remainder.
  std::vector<int64_t> host_offsets;
  std::vector<int64_t> device_offsets;

  // Iterate over host dimensions from back-to-front.
  for (int i = host_rank - 1; i > -1; i--) {
    // Dimensions that do not appear in the tile map are ignored.
    if (tile_map[i].size() == 0) continue;

    const int64_t host_stride = spyre_dma_strides[i];
    int64_t host_size = sizes[i];

    // Fold leading host dimensions that do not appear in the tile map.
    for (int j = i - 1; j > -1 && tile_map[j].size() == 0; j--) {
      // Expanded dimensions are ignored.
      if (spyre_dma_strides[j] == 0) continue;

      host_size *= sizes[j];
    }

    int64_t elements_before = 1;

    // Iterate over the device dimension that come from the host dimension from
    // back-to-front.
    //
    // These are stored in the tile map from back-to-front, so we are in effect
    // iterting them from front-to-back.
    for (int j = tile_map[i].size() - 1; j > -1; j--) {
      const int tile_index = tile_map[i][j];
      const int64_t tile_size = stl.device_size[tile_index];
      const int64_t tile_stride = host_strides[tile_index] / host_stride;

      // Size 1 dimensions are ignored.
      if (tile_size == 1) continue;

      TORCH_CHECK(
          host_size % elements_before == 0,
          "Invalid device sizes and stride map for host sizes and strides");

      const int64_t current_elements = host_size / elements_before;
      const int64_t remaining_elements = current_elements / tile_stride;

      TORCH_CHECK(
          remaining_elements > 0,
          "Invalid device sizes and stride map for host sizes and strides");

      if (current_elements % tile_stride == 0) {
        // When the current elements is evenly divisible by the tile stride then
        // this tile has no remainder.

        dcsi_sizes[tile_index] = std::min(remaining_elements, tile_size);

        elements_before *= dcsi_sizes[tile_index];
      } else {
        // When the current elements is not evenly divisible by the tile stride
        // then this tile and the next tile have a remainder.
        //
        // In these cases we get both tile and compute the dcsi sizes and
        // remainders for this tile and the next tile using the information from
        // both tiles. We then update the remainders and offsets so they can be
        // used to populate subsequent DataConversionStrideInfo.

        TORCH_CHECK(j != 0, "Invalid tiling for dimension");
        j--;

        const int next_index = tile_map[i][j];
        const int64_t next_size = stl.device_size[next_index];
        const int64_t next_stride = host_strides[next_index] / host_stride;

        const int64_t tiled_elements = current_elements / next_stride;

        dcsi_sizes[tile_index] = remaining_elements;
        dcsi_sizes[next_index] = next_size;

        elements_before *= tiled_elements;

        std::vector<int64_t> remainder(device_rank, 0);
        remainder[tile_index] = 1;
        remainder[next_index] = tiled_elements % next_size;

        remainders.push_back(remainder);
        host_offsets.push_back(remaining_elements * host_strides[tile_index]);
        device_offsets.push_back(remaining_elements *
                                 device_strides[tile_index]);
      }
    }
  }

  // Create the first DataConversionStrideInfo.
  DataConversionStrideInfo stride_info;
  stride_info.size_ = dcsi_sizes;
  stride_info.stride_src_ = host2device ? cpu_layout_strides : device_strides;
  stride_info.stride_dst_ = host2device ? device_strides : cpu_layout_strides;
  stride_info.offset_src_ = host2device ? storage_offset : 0;
  stride_info.offset_dst_ = host2device ? 0 : storage_offset;

  std::reverse(stride_info.size_.begin(), stride_info.size_.end());
  std::reverse(stride_info.stride_src_.begin(), stride_info.stride_src_.end());
  std::reverse(stride_info.stride_dst_.begin(), stride_info.stride_dst_.end());

  std::vector<DataConversionStrideInfo> stride_infos = {stride_info};

  // Iterate through the remainders and create subsequent
  // DataConversionStrideInfo for each.
  for (auto i = 0; i < remainders.size(); i++) {
    std::reverse(remainders[i].begin(), remainders[i].end());
    const auto offset_src = host2device ? host_offsets[i] : device_offsets[i];
    const auto offset_dst = host2device ? device_offsets[i] : host_offsets[i];

    const auto num_infos = stride_infos.size();
    for (auto j = 0; j < num_infos; j++) {
      DataConversionStrideInfo info = stride_infos[j];
      for (auto k = 0; k < device_rank; k++) {
        info.size_[k] =
            remainders[i][k] == 0 ? info.size_[k] : remainders[i][k];
      }
      info.offset_src_ += offset_src;
      info.offset_dst_ += offset_dst;
      stride_infos.push_back(info);
    }
  }

  return stride_infos;
}

/*
 * Generate description of data conversion for a tensor.
 *
 * @param cpu_tensor: CPU-side tensor (source for H2D, destination for D2H)
 * @param dev_tensor: device-side tensor (destination for H2D, source for D2H)
 * @return data conversion information
 */
auto generate_dci(const at::Tensor* cpu_tensor, const at::Tensor* dev_tensor,
                  SpyreTensorLayout stl, int64_t cpu_offset, bool host2device)
    -> DataConversionInfo {
  auto str_type = torchScalarToString[cpu_tensor->scalar_type()];
  const auto [dtype_cpu, dtype_dev] = stringToDTDataFormatPair(str_type);

  DataConversionInfo dci{};
  dci.dci_dsName_ = "DCI-Tensor-0";
  dci.isHostToSen_ = host2device;
  dci.dataformat_src_ = host2device ? dtype_cpu : dtype_dev;
  dci.dataformat_dst_ = host2device ? dtype_dev : dtype_cpu;

  std::vector<int64_t> cpu_shape;
  std::vector<int64_t> dev_shape = stl.device_size;
  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(dev_tensor->unsafeGetTensorImpl());

  c10::IntArrayRef t_sizes;
  c10::IntArrayRef t_dev_strides;
  c10::IntArrayRef t_cpu_strides;

  if (host2device) {
    cpu_shape = cpu_tensor->sizes().vec();
    t_sizes = cpu_tensor->sizes();
    t_dev_strides = c10::IntArrayRef(spyre_tensor_impl->dma_strides);
    t_cpu_strides = cpu_tensor->strides();
  } else {
    // Transfer contiguous memory, deal with view on cpu
    cpu_shape = spyre_tensor_impl->dma_sizes;
    t_sizes = c10::IntArrayRef(spyre_tensor_impl->dma_sizes);
    t_dev_strides = c10::IntArrayRef(spyre_tensor_impl->dma_strides);
    t_cpu_strides = c10::IntArrayRef(spyre_tensor_impl->dma_strides);
  }
  // Reverse PyTorch ordering
  std::reverse(cpu_shape.begin(), cpu_shape.end());
  std::reverse(dev_shape.begin(), dev_shape.end());
  dci.dcsi_ = get_device_stride_infos(t_sizes, t_dev_strides, cpu_offset, stl,
                                      host2device, t_cpu_strides);

  dci.input_shape_ = host2device ? cpu_shape : dev_shape;
  dci.output_shape_ = host2device ? dev_shape : cpu_shape;
  if (g_debug_info_enabled) {
    std::stringstream s;
    dci.exportJson(s);
    DEBUGINFO("DataConversionInfo: ", s.str());
  }
  return dci;
}

// Empty op needs C++ code and cannot be handled by python side fallback
at::Tensor spyre_empty(c10::IntArrayRef size,
                       std::optional<c10::ScalarType> dtype_opt,
                       std::optional<c10::Layout> layout_opt,
                       std::optional<c10::Device> device_opt,
                       std::optional<bool> pin_memory_opt,
                       std::optional<c10::MemoryFormat> memory_format_opt) {
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  DEBUGINFO("shape=", size, " on Spyre ", device);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  const c10::DeviceGuard device_guard(device);

  auto device_layout = SpyreTensorLayout(size.vec(), dtype);
  size_t size_bytes = get_device_size_in_bytes(device_layout);
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
  spyre_tensor_impl->dma_sizes = size.vec();
  spyre_tensor_impl->dma_strides = tensor.strides().vec();
  DEBUGINFO("SpyreTensorLayout: ", device_layout.toString());
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
  DEBUGINFO("Tensor info on CPU (Size:", size, ", Stride: ", stride,
            ", dtype: ", dtype, ") to be mapped onto device ", device);
  auto device_layout = SpyreTensorLayout(size.vec(), stride.vec(), scalar_type,
                                         generic_stick_dim_order(size.size()));
  size_t size_bytes = get_device_size_in_bytes(device_layout);

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
  if (size.size() == 0) {
    std::vector<int64_t> one = {1};
    c10::IntArrayRef tmp_size(one);
    c10::IntArrayRef tmp_stride(one);
    spyre_tensor_impl->set_sizes_and_strides(tmp_size, tmp_stride);

  } else {
    spyre_tensor_impl->set_sizes_and_strides(size, stride);
  }

  spyre_tensor_impl->spyre_layout = device_layout;
  spyre_tensor_impl->dma_sizes = size.vec();
  spyre_tensor_impl->dma_strides = stride.vec();

  DEBUGINFO("SpyreTensorLayout: ", device_layout.toString());
  return tensor;
}

at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout) {
  at::detail::check_size_nonnegative(size);
  c10::Device device =
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice();
  size_t size_bytes = get_device_size_in_bytes(device_layout);
  auto spyre_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      &SpyreAllocator::instance(),
      /*resizeable=*/true);
  auto spyre_storage = c10::Storage(spyre_storage_impl);

  // Create the Spyre Tensor
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      std::move(spyre_storage), pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_and_strides(size, stride);
  spyre_tensor_impl->spyre_layout = device_layout;
  spyre_tensor_impl->dma_sizes = size.vec();
  spyre_tensor_impl->dma_strides = stride.vec();
  DEBUGINFO("SpyreTensorLayout: ", device_layout.toString());
  return tensor;
}

at::Tensor& spyre_set_storage(at::Tensor& result, at::Storage storage,
                              int64_t storage_offset, c10::IntArrayRef size,
                              c10::IntArrayRef stride) {
  DEBUGINFO("set method");
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

/**
 * This method handles copy between devices. When copying to Spyre, this method
 * marks the tensor to compute on Spyre, but continue to use CPU tensor for now
 * such that when we run an op on the tensor on the Spyre, it will have the
 * proper handle to the Spyre allocation
 */
at::Tensor spyre_copy_from(const at::Tensor& self, const at::Tensor& dst,
                           bool non_blocking) {
  SpyreStream stream;
  if (dst.is_privateuseone()) {
    stream = getCurrentStream(dst.device());
  } else {
    stream = getCurrentStream(self.device());
  }
  stream.copyAsync(self, dst);
  if (!non_blocking) {
    stream.synchronize();
  }
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
  DEBUGINFO("shape=", size, " on Spyre ", device);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  const c10::DeviceGuard device_guard(device);

  size_t size_bytes = get_device_size_in_bytes(device_layout);
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
  spyre_tensor_impl->dma_sizes = size.vec();
  spyre_tensor_impl->dma_strides = tensor.strides().vec();
  DEBUGINFO("SpyreTensorLayout: ", device_layout.toString());
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
}

}  // namespace spyre
