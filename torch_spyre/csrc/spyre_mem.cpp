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
#include <ATen/InferSize.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
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
#include <cstddef>
#include <flex/flex_graph_builder.hpp>
#include <memory>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/interface/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
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
  if (dim == dev_dim_order.back()) {  // stick_dim
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
auto get_dim_device_stride(int dim, int stick_size,
                           std::vector<int64_t> dev_size,
                           std::vector<int64_t> dev_strides) {
  int dev_stride;
  if (dev_strides.size() == 1) {
    dev_stride = stick_size;
  } else {
    dev_stride = dev_strides.back() * dev_size[dev_strides.size() - 1];
  }
  return dev_stride;
}

/*
 * Fills out size and strides for each dimension of the tensor.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @param strides: dimension strides of the CPU tensor
 * @param stick_size: stick length for the dtype
 * @param device_sizes: dimesion sizes of dev tensor
 * @param host2device: direction of data conversion
 * @return description of data conversion
 */
auto get_device_stride_info(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                            SpyreTensorLayout stl, int stick_size,
                            std::vector<int64_t> device_sizes, bool host2device)
    -> DataConversionStrideInfo {
  DataConversionStrideInfo stride_info;
  auto cpu_shape = sizes.vec();
  auto cpu_strides = strides.vec();

  // sparse tensors need no padding of the stick dimension
  bool sparse = stl.dim_map.back() == -1;
  bool requires_padding =
      !sparse && cpu_shape[stl.dim_map.back()] % stick_size != 0;
  bool size_less_than_stick =
      !sparse && cpu_shape[stl.dim_map.back()] < stick_size;

  stride_info.size_ = device_sizes;

  if (size_less_than_stick) {
    stride_info.size_[0] = cpu_shape[stl.dim_map.back()];
  }
  stride_info.stride_src_.push_back(1);
  stride_info.stride_dst_.push_back(1);

  for (int i = 1; i < stl.dim_map.size(); i++) {
    auto& dim = stl.dim_map[(stl.dim_map.size() - 1) - i];
    auto cpu_stride =
        get_dim_cpu_stride(dim, stick_size, stl.dim_map, cpu_strides);
    auto dev_stride = get_dim_device_stride(
        dim, stick_size, device_sizes,
        host2device ? stride_info.stride_dst_ : stride_info.stride_src_);

    stride_info.stride_src_.push_back(host2device ? cpu_stride : dev_stride);
    stride_info.stride_dst_.push_back(host2device ? dev_stride : cpu_stride);
    if (dim == stl.dim_map.back() && requires_padding &&
        !size_less_than_stick) {  // stick_dim
      stride_info.size_[i] -= 1;
    }
  }
  stride_info.offset_src_ = 0;
  stride_info.offset_dst_ = 0;

  // pull single value from stick if sparse tensor
  if (sparse) {
    stride_info.size_[0] = 1;
  }

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
 * @param host2device: direction of data conversion
 * @return descriptions of data conversions for the tensor
 */
auto get_device_stride_infos(c10::IntArrayRef sizes, c10::IntArrayRef strides,
                             SpyreTensorLayout stl,
                             std::vector<int64_t> dev_shape, bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  std::vector<DataConversionStrideInfo> dcsi;
  auto cpu_shape = sizes.vec();
  int stick_size = stl.elems_per_stick();

  // sparse tensors need no padding of the stick dimension
  bool sparse = stl.dim_map.back() == -1;
  bool requires_padding =
      !sparse && cpu_shape[stl.dim_map.back()] % stick_size != 0;
  bool size_less_than_stick =
      !sparse && cpu_shape[stl.dim_map.back()] < stick_size;
  DataConversionStrideInfo stride_info;

  stride_info = get_device_stride_info(sizes, strides, stl, stick_size,
                                       dev_shape, host2device);
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
      auto& dim = stl.dim_map[(stl.dim_map.size() - 1) - i];
      dev_offset *= pad_stride_info.size_[i];
      if (dim == stl.dim_map.back()) {
        cpu_offset *= pad_stride_info.size_[i];
        // Stick dimension is the size of the remainder of cpu_shape/stick_size
        pad_stride_info.size_[i] = 1;
        pad_stride_info.size_[0] = cpu_shape[stl.dim_map.back()] % stick_size;
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
  auto dev_shape = stl.device_size;
  std::reverse(cpu_shape.begin(), cpu_shape.end());
  std::reverse(dev_shape.begin(), dev_shape.end());
  dci.dcsi_ = get_device_stride_infos(tensor->sizes(), tensor->strides(), stl,
                                      dev_shape, host2device);

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
// allocated memory not the actual pointer
struct SpyreAllocator final : public at::Allocator {
 private:
  SpyreAllocator() = default;
  flex::DeviceMemoryAllocatorPtr getAllocator(unsigned int dev_id) {
    return GlobalRuntime::get()
        ->GetDeviceHandle(dev_id)
        ->GetDeviceMemoryAllocator();
  }

 public:
  static SpyreAllocator& instance() {
    static SpyreAllocator allocator;
    return allocator;
  }

  at::DataPtr allocate(size_t nbytes) override {
    c10::Device curr_device =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
            ->getDevice();

    auto device_id = curr_device.index();
    DEBUGINFO("allocating ", nbytes, " (bytes) on Spyre", curr_device);
    if (nbytes <= 0) {
      return {nullptr, nullptr, &ReportAndDelete, curr_device};
    }
    auto allocator = getAllocator(device_id);
    flex::DeviceMemoryAllocationPtr data;  // a smart-pointer object
    // NOTE: last argument should be set to 0
    allocator->TryAllocate(&data, nbytes, 0);
    TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on Spyre device.");
    auto* ctx = new SharedOwnerCtx{std::move(data), device_id};
    void* ctx_void = static_cast<void*>(ctx);

    void* data_void = static_cast<void*>(ctx->owner.get());

    auto data_ptr_result =
        at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);

    return data_ptr_result;
  }

  static void ReportAndDelete(void* ctx_void) {
    if (!ctx_void) {
      return;
    }
    auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
    delete ctx;
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
  DEBUGINFO("Tensor info on CPU (Size:", size, ", Stride: ", stride,
            ", dtype: ", dtype, ") to be mapped onto device ", device);
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
  // TODO(aviros): This as is will lead to many errors for views, fail for now
  TORCH_CHECK_NOT_IMPLEMENTED(
      !self.is_privateuseone(),
      "as_strided not implemented for Spyre tensors, implement the caller "
      "using as_strided_with_layout with the proper semantics");
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

static SpyreTensorLayout compute_view_layout(c10::IntArrayRef old_sizes,
                                             c10::IntArrayRef new_sizes,
                                             const SpyreTensorLayout& old_stl) {
  size_t old_rank = old_sizes.size();
  size_t new_rank = new_sizes.size();

  // Phase 1: Identify old->new host dimension groups
  std::vector<DimGroup> groups;
  size_t old_i = 0, new_j = 0;
  while (old_i < old_rank || new_j < new_rank) {
    // Handle trailing size-1 dims
    if (old_i >= old_rank) {
      TORCH_CHECK(new_sizes[new_j] == 1,
                  "view: unsqueezed dimension must be size 1");
      groups.push_back({{}, {new_j}});
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

    // Handle size-1 insertions/removals
    if (old_sizes[old_i] == 1 && new_sizes[new_j] != 1) {
      groups.push_back({{old_i}, {}});
      old_i++;
      continue;
    }
    if (new_sizes[new_j] == 1 && old_sizes[old_i] != 1) {
      groups.push_back({{}, {new_j}});
      new_j++;
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

    // Device dims not mapped to any host dim (e.g., -1 for scalars/sparse)
    if (mapped_dim == -1) {
      new_device_size.push_back(old_stl.device_size[d]);
      new_dim_map.push_back(-1);
      continue;
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
      // Size-1 dims are typically filtered before tiling in init(), so they
      // shouldn't appear in dim_map. But handle the all-size-1 special case.
      // Map to -1 since the host dim is gone.
      new_device_size.push_back(old_stl.device_size[d]);
      new_dim_map.push_back(-1);
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
    bool is_stick_dim = (d == dev_rank - 1) && (dev_size == eps);

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

  return SpyreTensorLayout(new_device_size, new_dim_map, old_stl.device_dtype);
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
  const c10::DeviceGuard device_guard(device);

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

at::Tensor as_strided_with_layout(const at::Tensor& self, c10::IntArrayRef size,
                                  c10::IntArrayRef stride,
                                  std::optional<int64_t> storage_offset_,
                                  SpyreTensorLayout device_layout) {
  // NOTE: This function does not check whether
  // the as_strided info and stl are compatible.
  // This is for the caller of this function to check
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  static_cast<SpyreTensorImpl*>(result.unsafeGetTensorImpl())->spyre_layout =
      device_layout;
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("as_strided", TORCH_FN(spyre_as_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
  m.impl("_copy_from", TORCH_FN(spyre_copy_from));
  m.impl("view", TORCH_FN(spyre_view));
  m.impl("_unsafe_view", TORCH_FN(spyre__unsafe_view));
}

}  // namespace spyre
