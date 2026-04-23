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
#include <util/sen_data_convert.h>

#include <algorithm>
#include <flex/runtime_graph/graph/graph_builder/flex_graph_builder.hpp>
#include <map>
#include <memory>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/interface/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_sendnn_utils.h"
#include "spyre_storage_impl.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace py = pybind11;

namespace spyre {

using DataConversionStrideInfo = data_conversion_stride_info;
using DataConversionInfo = data_conversion_info;

/* struct holding the parameters for DMA-based copy
   size_bytes: number of bytes to transfer
   src_offset: offset from src base pointer
   dst_offset: offset from destination base pointer
 */
struct DMAParameters {
  const int64_t size_bytes;
  const off64_t src_offset;
  const off64_t dst_offset;
};

/* Generates the dimension mapping between `strides` and `stride_map`.
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
 * @param sizes: dimension sizes of the CPU tensor
 * @param strides: dimension strides of the CPU tensor
 * @param storage_offset: storage offset of the CPU tensor
 * @param stl: SpyreTensorLayout of dev tensor
 * @param host2device: direction of data conversion
 * @return description of data conversion
 */
auto get_device_stride_infos(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                             int64_t storage_offset, SpyreTensorLayout stl,
                             bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  const std::vector<std::vector<int>> tile_map =
      get_tile_map(sizes, strides, stl.device_size, stl.stride_map);

  const int host_rank = strides.size();
  const int device_rank = stl.stride_map.size();

  // The host strides, which match the stride map except for size 1 dimensions.
  std::vector<int64_t> host_strides(device_rank, 1);
  // The device strides are always contiguous strides for device sizes.
  std::vector<int64_t> device_strides(device_rank, 1);
  // The sizes for the fist DataConversionStrideInfo match the device sizes
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

    const int64_t host_stride = strides[i];
    int64_t host_size = sizes[i];

    // Fold leading host dimensions that do not appear in the tile map.
    for (int j = i - 1; j > -1 && tile_map[j].size() == 0; j--) {
      // Expanded dimensions are ignored.
      if (strides[j] == 0) continue;

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
  stride_info.stride_src_ = host2device ? host_strides : device_strides;
  stride_info.stride_dst_ = host2device ? device_strides : host_strides;
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
 * @param tensor: tensor to convert
 * @return data conversion information in string
 */
auto generate_dci(const at::Tensor* tensor, SpyreTensorLayout stl,
                  int64_t cpu_offset, bool host2device) -> std::string {
  /*   host2device = true : then 'tensor' is CPU-tensor
   *   host2device = false: then 'tensor' is Spyre-tensor
   */
  auto str_type = torchScalarToString[tensor->scalar_type()];
  const auto [dtype_cpu, dtype_dev] = stringToDTDataFormatPair(str_type);
  std::stringstream s;

  DataConversionInfo dci{};
  dci.dci_dsName_ = "DCI-Tensor-0";
  dci.isHostToSen_ = host2device;
  dci.dataformat_src_ = host2device ? dtype_cpu : dtype_dev;
  dci.dataformat_dst_ = host2device ? dtype_dev : dtype_cpu;

  std::vector<int64_t> cpu_shape;
  std::vector<int64_t> dev_shape = stl.device_size;
  c10::IntArrayRef t_sizes;
  c10::IntArrayRef t_strides;
  if (host2device) {
    // Respect cpu shapes
    cpu_shape = tensor->sizes().vec();
    t_sizes = tensor->sizes();
    t_strides = tensor->strides();
  } else {
    // Transfer contiguous memory, deal with view on cpu
    auto spyre_tensor_impl =
        static_cast<SpyreTensorImpl*>(tensor->unsafeGetTensorImpl());
    cpu_shape = spyre_tensor_impl->dma_sizes;
    t_sizes = c10::IntArrayRef(spyre_tensor_impl->dma_sizes);
    t_strides = c10::IntArrayRef(spyre_tensor_impl->dma_strides);
  }
  // Reverse PyTorch ordering
  std::reverse(cpu_shape.begin(), cpu_shape.end());
  std::reverse(dev_shape.begin(), dev_shape.end());
  dci.dcsi_ =
      get_device_stride_infos(t_sizes, t_strides, cpu_offset, stl, host2device);

  dci.input_shape_ = host2device ? cpu_shape : dev_shape;
  dci.output_shape_ = host2device ? dev_shape : cpu_shape;
  dci.exportJson(s);
  DEBUGINFO("DataConversionInfo: ", s.str());
  return s.str();
}

auto create_dma_graph(const at::Tensor& self, const at::Tensor& dst,
                      bool host2device)
    -> std::shared_ptr<sendnn::GraphLoader> {
  /* self = source
   * dst  = destination
   */
  const at::Tensor* dev_tensor;
  const at::Tensor* cpu_tensor;
  if (host2device) {
    cpu_tensor = &self;
    dev_tensor = &dst;
  } else {
    cpu_tensor = &dst;
    dev_tensor = &self;
  }

  auto str_type = torchScalarToString[cpu_tensor->scalar_type()];
  const auto [sen_dtype_cpu, sen_dtype_dev] = stringToSenDatatypePair(str_type);
  auto layout = sendnn::TensorLayout::NHWC;
  SpyreTensorLayout stl = get_spyre_tensor_layout(host2device ? dst : self);
  sendnn::TensorShape dev_tensor_shape(stl.device_size);

  // ti = transfer info
  // dci = data conversion info
  sendnn::TensorInfo cpu_ti(sen_dtype_cpu,
                            sendnn::TensorShape(cpu_tensor->sizes().vec()),
                            layout, sendnn::TensorLocation::HOST());
  sendnn::TensorInfo dev_ti(sen_dtype_dev, dev_tensor_shape, layout,
                            sendnn::TensorLocation::DEVICE());
  sendnn::TensorInfo dci_ti(sen_dtype_dev, dev_tensor_shape, layout,
                            sendnn::TensorLocation::HOST());
  //  STAGE 1: execution graph
  sendnn::SubGraph sub_graph;
  const auto [elem_bytes_cpu, elem_bytes_spyre] =
      spyre::elementSize(cpu_tensor->scalar_type());
  int64_t xfer_size = dev_tensor_shape.Volume() * elem_bytes_spyre;
  {
    flex::FlexGraphBuilder gb;
    DMAParameters dma_param{xfer_size, 0, 0};
    if (host2device) {
      auto inp_node = gb.PrimaryInput("Input", dci_ti);
      auto xfer_node = gb.SenDataTransfer(
          "Host2Sen-Transfer",
          dev_ti,    // output (holding shape, type, and location DEVICE)
          inp_node,  // input (node created using PrimaryInput and on HOST)
          dev_ti.DataSize(), dma_param.src_offset, dma_param.dst_offset);
      auto out_node = gb.PrimaryOutput("Output", xfer_node);
    } else {
      auto inp_node = gb.PrimaryInput("Input", dev_ti);
      auto xfer_node = gb.SenDataTransfer(
          "Sen2Host-Transfer",
          dci_ti,    // output (holding shape, type and location HOST)
          inp_node,  // input (node created as a result of SenDataTransfer)
          dev_ti.DataSize(), dma_param.src_offset, dma_param.dst_offset);
      auto out_node = gb.PrimaryOutput("Output", xfer_node);
    }

    SEN_THROW_NOK(gb.Finalize(&sub_graph));
  }
  sendnn::SubGraph exec_graph;
  {  // add above subgraph as part of SenFusedDeviceCompute node
    flex::FlexGraphBuilder gb;
    auto dci = generate_dci(dev_tensor, stl, cpu_tensor->storage_offset(),
                            host2device);
    if (host2device) {
      auto inp_node = gb.PrimaryInput("Input", cpu_ti);
      auto dci_node = gb.SenHostCompute("Host2Sen-HostPrep", {dci_ti},
                                        {inp_node}, "SenDataConvert", dci);

      auto dev_node = gb.SenFusedDeviceCompute("SenFusedDeviceNode_0", {dci_ti},
                                               {dci_node}, sub_graph);
      gb.PrimaryOutput("Output", dev_node->OutputPort(0));
    } else {
      sendnn::Node* inp_node = gb.PrimaryInput("Input", dci_ti);
      auto dev_node = gb.SenFusedDeviceCompute("SenFusedDeviceNode_0", {dci_ti},
                                               {inp_node}, sub_graph);
      auto dci_node = gb.SenHostCompute("Sen2Host-HostPrep", cpu_ti, dev_node,
                                        "SenDataConvert", dci);

      gb.PrimaryOutput("Output", dci_node->OutputPort(0));
    }

    SEN_THROW_NOK(gb.Finalize(&exec_graph));
  }

  sendnn::SegmentTable segment_table = {
      sendnn::Segment::PRIMARY_OUT(xfer_size),
      sendnn::Segment::PRIMARY_IN(xfer_size),
      sendnn::Segment::INVALID,
      sendnn::Segment::INVALID,
      sendnn::Segment::INVALID,
      sendnn::Segment::INVALID,
      sendnn::Segment::INVALID,
      sendnn::Segment::PROGRAM(128),
  };
  // STAGE 2: SenSuperNodeV2 graph
  sendnn::Graph sn_graph;  // sn = supernode
  {                        // SenSuperNodeV2 graph
    flex::FlexGraphBuilder gb;

    sendnn::TensorInfo inp_ti =
        sendnn::TensorInfo(exec_graph.input_ops_.front()->OutputAt(0));
    sendnn::TensorInfo out_ti =
        sendnn::TensorInfo(exec_graph.output_ops_.front()->InputAt(0));
    sendnn::NodeOrIndexedNode inp_node = gb.PrimaryInput("Input", inp_ti);

    std::string k_uuid = "dma-network";
    sendnn::attributes::SenPartitionInit part_init;
    part_init.network_uuid_ = k_uuid;
    part_init.partition_idx_ = 0;
    part_init.segment_table_ = segment_table;

    auto sn =
        gb.SenSuperNodeV2("SenSuperNodeV2_0", {out_ti}, {inp_node}, k_uuid, 0,
                          1, part_init, exec_graph, {}, false, true, true);
    gb.PrimaryOutput("Output", {0, sn});

    SEN_THROW_NOK(gb.Finalize(&sn_graph));
  }

  // STAGE 3:
  std::shared_ptr<sendnn::GraphLoader> gl;
  gl = std::make_shared<sendnn::GraphLoader>(GlobalRuntime::get());
  {
    SEN_THROW_NOK(gl->LoadGraph(sn_graph));
    SEN_THROW_NOK(gl->CompileGraph());
    SEN_THROW_NOK(gl->ParseGraph());
  }
  return gl;
}

void copy_host_to_device(const at::Tensor& self, const at::Tensor& dst) {
  std::shared_ptr<sendnn::GraphLoader> gl = create_dma_graph(self, dst, true);
  if (!gl) {
    DEBUGINFO("GraphLoader is null!");
    return;
  }
  // execute
  constexpr int sn_idx = 0;
  constexpr int tensor_idx = 0;
  auto inp_tensor = createInputTensor(*gl, self.storage().data_ptr().get(),
                                      tensor_idx, sn_idx);
  auto* ctx =
      static_cast<SharedOwnerCtx*>(dst.storage().data_ptr().get_context());
  flex::DeviceMemoryAllocationPtr& dev_data = ctx->owner;
  inp_tensor.SetSpyreData(dev_data);  // ctx->owner;

  SEN_THROW_NOK(gl->Copy(sendnn::Outputs(), {inp_tensor}, sn_idx));
}

void copy_device_to_host(const at::Tensor& self, const at::Tensor& dst) {
  std::shared_ptr<sendnn::GraphLoader> gl = create_dma_graph(self, dst, false);
  // execute
  constexpr int sn_idx = 0;
  constexpr int tensor_idx = 0;
  auto out_tensor = createOutputTensor(*gl, dst.storage().data_ptr().get(),
                                       tensor_idx, sn_idx);
  auto* ctx =
      static_cast<SharedOwnerCtx*>(self.storage().data_ptr().get_context());
  out_tensor.SetSpyreData(ctx->owner);
  SEN_THROW_NOK(gl->Copy({out_tensor}, sendnn::Inputs(), sn_idx));
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
