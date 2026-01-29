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

// cpplint: disable=readability/braces

#include "spyre_mem.h"

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>
#include <util/sen_data_convert.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>  // check env vars
#include <flex/flex_graph_builder.hpp>
#include <memory>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/interface/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <stdexcept>  // throw exceptions
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
  // NOLINTNEXTLINE(readability/braces)
};
/*
 * CPU stride for a dimension.
 *
 * @param dim: dimension index
 * @param stick_size: stick length for the dtype
 * @param dev_dim_order: order of tensor dimensions on device
 * @param cpu_strides: strides of cpu tensor
 * @return CPU stride of the dimension
 */
auto get_dim_cpu_stride(int dim, int stick_size,
                        std::vector<int32_t> dev_dim_order,
                        std::vector<int64_t> cpu_strides) {
  int cpu_stride;
  if (dim == dev_dim_order.front()) {  // stick_dim
    cpu_stride = stick_size;
  } else {
    cpu_stride = cpu_strides[dim];
  }
  return cpu_stride;
}
/*
 * Device stride for a dimension.
 *
 * @param dim: dimension index
 * @param stick_size: stick length for the dtype
 * @param dev_dim_order: order of tensor dimensions on device
 * @param dev_strides: strides of device tensor
 * @param dev_shape: shape of tensor on device
 * @return device stride of the dimension
 */
auto get_dim_device_stride(int dim, int stick_size, SpyreTensorLayout stl,
                           std::vector<int64_t> dev_strides) {
  int dev_stride;
  if (dev_strides.size() == 1) {
    dev_stride = stick_size;
  } else {
    dev_stride = dev_strides.back() * stl.device_size[dev_strides.size() - 1];
  }
  return dev_stride;
}

/*
 * Fills out size and strides for each dimension of the tensor.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @param strides: dimension strides of the CPU tensor
 * @param stick_size: stick length for the dtype
 * @param host2device: direction of data conversion
 * @return description of data conversion
 */
auto get_device_stride_info(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                            SpyreTensorLayout stl, int stick_size,
                            bool host2device) -> DataConversionStrideInfo {
  DataConversionStrideInfo stride_info;
  auto cpu_shape = sizes.vec();
  auto cpu_strides = strides.vec();
  bool size_less_than_stick = cpu_shape[stl.dim_map.front()] < stick_size;
  bool requires_padding = cpu_shape[stl.dim_map.front()] % stick_size != 0;

  stride_info.size_ = stl.device_size;
  if (size_less_than_stick) {
    stride_info.size_[0] = cpu_shape[stl.dim_map.front()];
  }
  stride_info.stride_src_.push_back(1);
  stride_info.stride_dst_.push_back(1);

  for (int i = 1; i < stl.dim_map.size(); i++) {
    auto& dim = stl.dim_map[i];
    auto cpu_stride =
        get_dim_cpu_stride(dim, stick_size, stl.dim_map, cpu_strides);
    auto dev_stride = get_dim_device_stride(
        dim, stick_size, stl,
        host2device ? stride_info.stride_dst_ : stride_info.stride_src_);

    stride_info.stride_src_.push_back(host2device ? cpu_stride : dev_stride);
    stride_info.stride_dst_.push_back(host2device ? dev_stride : cpu_stride);
    if (dim == stl.dim_map.front() && requires_padding &&
        !size_less_than_stick) {  // stick_dim
      stride_info.size_[i] -= 1;
    }
  }
  stride_info.offset_src_ = 0;
  stride_info.offset_dst_ = 0;
  return stride_info;
}
/*
 * Generates one or more descriptions of data conversions based on padding
 * requirements.
 *
 * The stick dimension must be a multiple of the stick size. If the size of this
 * dimension on the CPU is not a multiple of the stick size, then padding is
 * added during the data conversion step. This padding is handled in two
 * different ways based on the size of the dimension:
 *    1. If the size of the stick dimension is less than the stick size, then
 *     a single DataConversionStrideInfo struct is created with the size of
 * that dimension being the cpu shape.
 *    2. If the size of the stick dimension is more than the stick size, then
 * two DataConversionStrideInfo are needed. The first is has the size of the
 * stick dimension being the cpu shape. The cpu and device offsets are 0. The
 * second DataConversionStrideInfo has the same cpu and device strides as the
 * first. For the second, the size of the stick dimension is the remainder of
 * the dimension size divided by the stick size (rounded down). The cpu offset
 * is the dimension size divided by the stick size (rounded up), multiplied by
 * the stick size. The device offset is the size of the stick size multiplied by
 * the volume of the dimensions preceeding the stick dim on the device.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @param strides: dimension strides of the CPU tensor
 * @param dev_shape: shape of tensor on device
 * @param stick_size: stick length for the dtype
 * @param host2device: direction of data conversion
 * @return descriptions of data conversions for the tensor
 */
auto get_device_stride_infos(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                             SpyreTensorLayout stl, int stick_size,
                             bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  std::vector<DataConversionStrideInfo> dcsi;
  auto cpu_shape = sizes.vec();
  bool requires_padding = cpu_shape[stl.dim_map.front()] % stick_size != 0;
  bool size_less_than_stick = cpu_shape[stl.dim_map.front()] < stick_size;
  DataConversionStrideInfo stride_info;

  stride_info =
      get_device_stride_info(sizes, strides, stl, stick_size, host2device);
  dcsi.push_back(stride_info);

  if (requires_padding && !size_less_than_stick) {
    /* Second DataConversionStrideInfo has same strides, so we can reuse the
     * stride information from the first DataConversionStrideInfo
     * and update the stick dim sizes and offsets
     */
    auto pad_stride_info = stride_info;
    auto dev_offset = stick_size;
    auto cpu_offset = stick_size;

    // Update host and device offsets
    for (int i = 1; i < stl.dim_map.size(); i++) {
      auto& dim = stl.dim_map[i];
      dev_offset *= pad_stride_info.size_[i];
      if (dim == stl.dim_map.front()) {
        cpu_offset *= pad_stride_info.size_[i];
        // Stick dimension is the size of the remainder of cpu_shape/stick_size
        pad_stride_info.size_[i] = 1;
        pad_stride_info.size_[0] = cpu_shape[stl.dim_map.front()] % stick_size;
        break;
      }
    }
    pad_stride_info.offset_src_ = host2device ? cpu_offset : dev_offset;
    pad_stride_info.offset_dst_ = host2device ? dev_offset : cpu_offset;
    dcsi.push_back(pad_stride_info);
  }
  return dcsi;
}
/*
 * Generate description of data conversion for a tensor.
 *
 * @param tensor: tensor to convert
 * @return data conversion information in string
 */
auto generate_dci(const at::Tensor* tensor, SpyreTensorLayout stl,
                  bool host2device) -> std::string {
  /*   host2device = true : then 'tensor' is CPU-tensor
   *   host2device = false: then 'tensor' is Spyre-tensor
   * TODO: support strided tensors
   */
  auto str_type = torchScalarToString[tensor->scalar_type()];
  const auto [dtype_cpu, dtype_dev] = stringToDTDataFormatPair(str_type);
  std::stringstream s;
  auto cpu_shape = tensor->sizes().vec();
  DataConversionInfo dci{};
  dci.dci_dsName_ = "DCI-Tensor-0";
  dci.isHostToSen_ = host2device;
  dci.dataformat_src_ = host2device ? dtype_cpu : dtype_dev;
  dci.dataformat_dst_ = host2device ? dtype_dev : dtype_cpu;
  // Reverse PyTorch ordering
  std::reverse(stl.device_size.begin(), stl.device_size.end());
  std::reverse(stl.dim_map.begin(), stl.dim_map.end());
  std::reverse(cpu_shape.begin(), cpu_shape.end());
  dci.dcsi_ = get_device_stride_infos(tensor->sizes(), tensor->strides(), stl,
                                      stl.elems_per_stick(), host2device);
  dci.input_shape_ = host2device ? cpu_shape : stl.device_size;
  dci.output_shape_ = host2device ? stl.device_size : cpu_shape;
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
    auto dci = generate_dci(dev_tensor, stl, host2device);
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
  };  // NOLINT
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
auto copy_host_to_device(const at::Tensor& self, const at::Tensor& dst) {
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
auto copy_device_to_host(const at::Tensor& self, const at::Tensor& dst) {
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

// A custom allocator for our custom device, what returns is a handle to the
// Forward declaration for test interface
namespace spyre {
namespace test {
class SpyreAllocatorTestInterface;
}  // namespace test
}  // namespace spyre

// allocated memory not the actual pointer
struct SpyreAllocator final : public at::Allocator {
  // Allow test interface to access private members
  friend class spyre::test::SpyreAllocatorTestInterface;

 private:
  flex::DeviceMemoryAllocatorPtr getAllocator(unsigned int dev_id) {
    return GlobalRuntime::get()
        ->GetDeviceHandle(dev_id)
        ->GetDeviceMemoryAllocator();
  }

  bool use_pf = false;
  bool alloc_debug = false;
  std::vector<SegmentInfo> segments;
  size_t segment_size;
  int n_segments;
  size_t vf_offset = 0;
  std::unordered_map<void*, SegmentInfo*> block_to_segment;
  // Segment size fixed to 12 GB (14 fails)
  static constexpr size_t DEFAULT_SEGMENT_SIZE = 12ULL * 1024 * 1024 * 1024;
  static constexpr size_t DEFAULT_N_SEGMENTS = 8;
  static constexpr size_t MIN_ALLOC_BYTES = 128;  // Spyre requirement

  bool get_function_mode() {
    const char* fmode_envvar = std::getenv("FLEX_DEVICE");
    TORCH_CHECK(fmode_envvar != nullptr, "FLEX_DEVICE env var is not set!")

    std::string fmode = fmode_envvar;
    if (fmode == "VF") {
      return false;
    } else if (fmode == "PF") {
      return true;
    } else {
      TORCH_CHECK(false, "Unsupported FLEX_DEVICE env var value: ", fmode);
    }
  }

  bool get_alloc_debug() {
    const char* alloc_envvar = std::getenv("TORCH_SPYRE_ALLOC_DEBUG");
    if (alloc_envvar == nullptr) return false;

    std::string alloc_debug_str = alloc_envvar;
    if (alloc_debug_str == "1") return true;
    return false;
  }

  SpyreAllocator(size_t seg_sz = DEFAULT_SEGMENT_SIZE,
                 int n_seg = DEFAULT_N_SEGMENTS)
      : segment_size(seg_sz), n_segments(n_seg) {
    /* This constructor determines if using VF of PF mode based on FLEX_DEVICE
     * env var. Alternatively to the following method, we could check if
     * allocator attribute vfw_ is nullptr. If so, PF is in use, otherwise VF.
     * However, this requires allocator object of type
     * flex::DeviceMemoryAllocatorPtr to exist, which doesn't yet in this
     * constructor. Can be checked within allocate() method though. However,
     * vfw_ is private and needs a getter created in Flex.
     */

    use_pf = get_function_mode();
    alloc_debug = get_alloc_debug();
  }

  at::DataPtr pf_allocation(flex::DeviceMemoryAllocatorPtr allocator,
                            size_t nbytes, c10::Device curr_device,
                            unsigned int device_id) {
    /* PF allocation implementation. Functionalities are preserved exactly from
     * earlier iteration of the code (PF-only).
     */

    DEBUGINFO("PF allocation");
    flex::DeviceMemoryAllocationPtr data;      // a smart-pointer object
    allocator->TryAllocate(&data, nbytes, 0);  // allocation request to Spyre
    TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on Spyre device.");

    // Instantiate object to live beyond SpyreAllocator scope
    auto* ctx = new SharedOwnerCtx{std::move(data), 0, device_id};
    void* ctx_void = static_cast<void*>(ctx);
    void* data_void = static_cast<void*>(ctx->owner.get());

    return at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);
  }

  at::DataPtr vf_allocation(flex::DeviceMemoryAllocatorPtr allocator,
                            size_t nbytes, c10::Device curr_device,
                            unsigned int device_id) {
    /* VF allocation implementation. A fixed number of Segments are
     * pre-allocated upon first call. Blocks are inserted into Segments
     * following round-robin strategy (memory-balanced), starting from vf_offset
     * = 0 and progressively increasing. No sub-Segment balancing is implemented
     * at this time.
     */

    flex::DeviceMemoryAllocationPtr data;  // a smart-pointer object

    if (segments.empty()) initializeSegments(allocator);

    size_t aligned_nbytes = setMinSpyreAllocation(nbytes);
    AllocationInfo alloc_info = findFreeBlock(aligned_nbytes);

    TORCH_CHECK(alloc_info.found,
                "Unable to find enough free memory for allocation. All ",
                n_segments, " segments are full.");

    allocateInSegment(alloc_info.segment, alloc_info.interval, aligned_nbytes,
                      vf_offset);

    // DeviceMemoryAllocationPtr shared within Segment
    data = alloc_info.segment->data;

    if (alloc_debug)
      logSegmentState(*alloc_info.segment, "After block allocation");

    TORCH_CHECK(data, "Failed to allocate ", aligned_nbytes,
                " bytes on Spyre device.");

    // Instantiate object to live beyond SpyreAllocator scope
    auto* ctx = new SharedOwnerCtx{std::move(data), vf_offset, device_id};
    void* ctx_void = static_cast<void*>(ctx);
    void* data_void = static_cast<void*>(ctx->owner.get());

    alloc_info.segment->blocks[ctx_void] =
        BlockInfo(vf_offset, vf_offset + aligned_nbytes);
    block_to_segment[ctx_void] = alloc_info.segment;

    return at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);
  }

  void initializeSegments(flex::DeviceMemoryAllocatorPtr allocator) {
    /* Request memory allocation on Spyre for `n_segments` of size
     * `segment_size`. */

    if (!segments.empty()) return;  // Already initialized

    DEBUGINFO("Initializing", n_segments, "segments");

    for (int i = 0; i < n_segments; i++) {
      flex::DeviceMemoryAllocationPtr data;
      allocator->TryAllocate(&data, segment_size, 0);
      TORCH_CHECK(data, "Failed to allocate segment ", i);
      segments.emplace_back(data->AllocIndex(), segment_size);
      segments.back().data = data;
      segments.back().free_intervals.insert(FreeInterval{0, segment_size});
      segments.back().free_interval_sizes.insert(segment_size);
    }
  }

  size_t setMinSpyreAllocation(size_t nbytes) const {
    /* Adjust allocation according to Spyre requirement. */

    // [AF] can nbytes request be ever misaligned?
    if (nbytes % MIN_ALLOC_BYTES != 0)
      return ((nbytes + MIN_ALLOC_BYTES - 1) / MIN_ALLOC_BYTES) *
             MIN_ALLOC_BYTES;
    return nbytes;
  }

  struct AllocationInfo {
    SegmentInfo* segment;
    FreeInterval interval;
    bool found;
  };  // NOLINT(readability/braces)

  AllocationInfo findFreeBlock(size_t nbytes) {
    /* Locate first memory interval that can accommodate a block of size nbytes.
     */

    TORCH_CHECK(nbytes <= segment_size, "Requested allocation (", nbytes,
                " bytes) exceeds segment size (", segment_size, " bytes)");

    SegmentInfo* best_seg = nullptr;
    size_t max_free_size = 0;

    for (SegmentInfo& seg : segments) {
      if (seg.free_size < nbytes || seg.free_interval_sizes.empty() ||
          *seg.free_interval_sizes.rbegin() < nbytes)
        continue;

      // Track segment with most free memory
      if (seg.free_size > max_free_size) {
        max_free_size = seg.free_size;
        best_seg = &seg;
      }
    }

    if (best_seg == nullptr) return {nullptr, {}, false};

    for (const FreeInterval& r : best_seg->free_intervals) {
      if (r.end - r.start >= nbytes)
        return {best_seg, r, true};  // free Block found
    }

    return {nullptr, {}, false};  // free Block not found
  }

  void allocateInSegment(SegmentInfo* seg, FreeInterval range, size_t nbytes,
                         size_t& vf_offset) {
    /* Given a predetermined Segment and a free memory range that accomodates at
     * least nbytes, mark this memory occupied, recalculate free range, and
     * update total Segment free memory.
     */

    DEBUGINFO("VF block allocation");
    vf_offset = range.start;
    seg->free_intervals.erase(
        range);  // remove FreeInterval selected to contain the new Block
    seg->free_interval_sizes.erase(range.end - range.start);

    if (range.end - range.start >
        nbytes) {  // if some space remains after Block creation
      FreeInterval new_range{range.start + nbytes, range.end};
      seg->free_intervals.insert(new_range);
      seg->free_interval_sizes.insert(range.end - range.start - nbytes);
    }
    seg->free_size -= nbytes;
  }

  void deallocateBlock(SegmentInfo& seg, void* ctx_void) {
    /* Deallocate a block from a segment and return its memory to the free pool.
     * Merges adjacent free intervals to reduce fragmentation. Updates segment's
     * free memory tracking and removes block from registry.
     */

    auto it = seg.blocks.find(ctx_void);
    if (it == seg.blocks.end()) return;

    DEBUGINFO("VF block deallocation");
    FreeInterval new_range{it->second.offset_init, it->second.offset_end};

    auto& fr = seg.free_intervals;
    auto fr_low = fr.lower_bound(new_range);

    // Merge with previous interval if touching at new_range.start
    if (fr_low != fr.begin()) {
      auto prev = std::prev(fr_low);
      if (prev->end == new_range.start) {
        new_range.start = prev->start;
        new_range.end = std::max(prev->end, new_range.end);
        seg.free_interval_sizes.erase(prev->end - prev->start);
        fr.erase(prev);
      }
    }

    // Merge with next interval if touching at new_range.end
    if (fr_low != fr.end() && fr_low->start == new_range.end) {
      new_range.end = std::max(fr_low->end, new_range.end);
      seg.free_interval_sizes.erase(fr_low->end - fr_low->start);
      fr.erase(fr_low);
    }

    fr.insert(new_range);
    seg.free_interval_sizes.insert(new_range.end - new_range.start);
    seg.free_size += it->second.offset_end - it->second.offset_init;
    seg.blocks.erase(it);
  }

  static void ReportAndDelete(void* ctx_void) {
    /* Called when DataPtr is being deallocated. */

    if (!ctx_void) return;

    auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
    if (!SpyreAllocator::instance().use_pf) {
      auto& allocator = SpyreAllocator::instance();

      if (allocator.alloc_debug)
        allocator.logAllSegments("Pre deallocation", true);

      // Using lookup map for blocks into segments (O(1))
      auto seg_it = allocator.block_to_segment.find(ctx_void);
      if (seg_it != allocator.block_to_segment.end()) {
        allocator.deallocateBlock(*seg_it->second, ctx_void);
        allocator.block_to_segment.erase(seg_it);
      }

      if (allocator.alloc_debug)
        allocator.logAllSegments("Post deallocation", true);
    }

    delete ctx;
  }

  void logSegmentState(const SegmentInfo& seg, const char* context,
                       bool include_blocks = false) {
    /* Log free and used memory in the specified Segment. */

    DEBUGINFO(context, "seg id", seg.segment_id, "free mem", seg.free_size);
    if (include_blocks) {
      for (const auto& [soc_ptr, block] : seg.blocks)
        DEBUGINFO("    ctx addr", soc_ptr,
                  "-> block bounds:", block.offset_init, block.offset_end);
    }
    for (const size_t& sz : seg.free_interval_sizes) DEBUGINFO("  free sz", sz);
    for (const FreeInterval& r : seg.free_intervals)
      DEBUGINFO("  free idx", r.start, "to", r.end);
  }

  void logAllSegments(const char* context, bool include_blocks = false) {
    /* Log free and used memory of all Segments. */

    DEBUGINFO(context);
    for (const SegmentInfo& seg : segments) {
      logSegmentState(seg, "", include_blocks);
    }
  }

 public:
  static SpyreAllocator& instance() {
    static SpyreAllocator allocator;
    return allocator;
  }

  at::DataPtr allocate(size_t nbytes) override {
    /* Allocation entry point. Implement branching to PF or VF allocation. */

    c10::Device curr_device =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
            ->getDevice();
    auto device_id = curr_device.index();
    DEBUGINFO("allocating", nbytes, "bytes on Spyre", curr_device);
    if (nbytes <= 0) return {nullptr, nullptr, &ReportAndDelete, curr_device};

    auto allocator = getAllocator(device_id);

    if (use_pf) {
      return pf_allocation(allocator, nbytes, curr_device, device_id);
    } else {
      return vf_allocation(allocator, nbytes, curr_device, device_id);
    }
  }

  // The raw deleter only gets passed the data ptr, no context, so
  // it would not work right now. To implement this, we first need to
  // create a runtime interface that can correctly free an allocation
  // only based on the data ptr, without the allocation idx from the
  // context
  at::DeleterFnPtr raw_deleter() const override {
    return nullptr;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    DEBUGINFO("entering allocator->copy_data method");
    // do nothing -- look into when this is called
    // spyre_copy_from(reinterpret_cast<spyre_ptr_t>(dest),
    // reinterpret_cast<spyre_ptr_t>(src));
  }
};

// Register our custom allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &SpyreAllocator::instance());

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

  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())->spyre_layout =
      device_layout;
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
  caffe2::TypeMeta dtype = c10::scalarTypeToTypeMeta(scalar_type);
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  DEBUGINFO("Size:", size, ", Stride: ", stride, " on device ", device);
  auto device_layout = SpyreTensorLayout(size.vec(), scalar_type);
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

  auto tensorImpl = tensor.unsafeGetTensorImpl();
  if (size.size() == 0) {
    std::vector<int64_t> one = {1};
    c10::IntArrayRef tmp_size(one);
    c10::IntArrayRef tmp_stride(one);
    tensorImpl->set_sizes_and_strides(tmp_size, tmp_stride);

  } else {
    tensorImpl->set_sizes_and_strides(size, stride);
  }

  static_cast<SpyreTensorImpl*>(tensorImpl)->spyre_layout = device_layout;
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

  auto tensorImpl = tensor.unsafeGetTensorImpl();
  tensorImpl->set_sizes_and_strides(size, stride);

  static_cast<SpyreTensorImpl*>(tensorImpl)->spyre_layout = device_layout;
  DEBUGINFO("SpyreTensorLayout: ", device_layout.toString());
  return tensor;
}
at::Tensor spyre_as_strided(const at::Tensor& self, c10::IntArrayRef size,
                            c10::IntArrayRef stride,
                            std::optional<int64_t> storage_offset_) {
  // Metadata-only change so we re-use the cpu impl
  return at::cpu::as_strided(self, size, stride, storage_offset_);
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
  DEBUGINFO("self (", self.scalar_type(), ") is on:", self.device());
  DEBUGINFO("dst (", dst.scalar_type(), ") on:", dst.device());
  at::Storage source_storage;
  at::Storage dest_storage;

  // TODO(tmhoangt): add type conversion node
  TORCH_CHECK(
      self.scalar_type() == dst.scalar_type(),
      "Spyre backend does not support type conversion yet during copy.");

  if (self.is_cpu() && dst.is_privateuseone()) {
    if (self.dim() == 0) {
      at::Tensor tmp_tensor = self.reshape({1});
      copy_host_to_device(tmp_tensor, dst);
    } else {
      copy_host_to_device(self, dst);
    }
    return dst;

  } else if (self.is_privateuseone() && dst.is_cpu()) {
    copy_device_to_host(self, dst);
    return dst;

  } else if (self.is_privateuseone() && dst.is_privateuseone()) {
    // Copy from Spyre to Spyre
    // FIXME: This will need to be addressed for proper spyre to spyre copy
    source_storage =
        (static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl()))->storage();
    dest_storage =
        (static_cast<SpyreTensorImpl*>(dst.unsafeGetTensorImpl()))->storage();
    DEBUGINFO("Copying", source_storage.nbytes(), "bytes from",
              source_storage.device(), "to", dest_storage.device());
    std::memcpy(dest_storage.data_ptr().get(), source_storage.data_ptr().get(),
                source_storage.nbytes());
    DEBUGINFO("Finished Copying ");
    return dst;
  } else {
    // For all other cases fallback to the upstream implementation
    return at::_copy_from(self, dst, non_blocking);
  }
}
at::Tensor to_with_layout(const at::Tensor& self,
                          SpyreTensorLayout device_layout) {
  DEBUGINFO(
      "Tensor info on CPU (Size:", self.sizes(), ", Stride: ", self.strides(),
      ", dtype: ", c10::typeMetaToScalarType(self.dtype()),
      ") and to be mapped onto device ",
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice(),
      " with layout ", device_layout.toString());
  auto dst = spyre_empty_with_layout(self.sizes(), self.strides(),
                                     c10::typeMetaToScalarType(self.dtype()),
                                     device_layout);
  return spyre_copy_from(self, dst, false);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("as_strided", TORCH_FN(spyre_as_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
  m.impl("_copy_from", TORCH_FN(spyre_copy_from));
}
}  // namespace spyre
