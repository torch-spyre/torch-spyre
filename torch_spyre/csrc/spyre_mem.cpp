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
#include <ATen/core/op_registration/adaption.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>
#include <util/sen_data_convert.h>

#include <algorithm>
#include <cassert>
#include <flex/flex_graph_builder.hpp>
#include <memory>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/runtime/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/sentensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
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
 * Ordering of tensor dimensions on the device for
 * default (generic stick) format.
 * Non-stick dimensions appear once, stick dimensions appear twice.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @return ordering of dimensions on device
 */
auto get_device_layout(c10::IntArrayRef sizes) -> std::vector<int64_t> {
  std::vector<int64_t> dim_order;
  switch (sizes.size()) {
    case 1:
      dim_order = {0, 0};
      break;
    case 2:
      dim_order = {1, 0, 1};
      break;
    case 3:
      dim_order = {2, 0, 2, 1};
      break;
    case 4:
      dim_order = {3, 2, 1, 3, 0};
      break;
    default:
      std::stringstream ss;
      ss << "Unsupported tensor rank: " << std::to_string(sizes.size());
      throw std::runtime_error(ss.str());
  }
  return dim_order;
}
/*
 * Shape of tensor on the device.
 *
 * @param sizes: dimension sizes of the CPU tensor
 * @param stick_size: stick length for the dtype
 * @return shape of tensor on the device
 */
auto get_device_shape(c10::IntArrayRef sizes, int stick_size)
    -> std::vector<int64_t> {
  auto cpu_shape = sizes.vec();
  std::vector<int64_t> dev_shape;
  auto dev_dim_order = get_device_layout(cpu_shape);
  auto stick_dim_shape = cpu_shape[dev_dim_order.front()];
  auto stick_dim = dev_dim_order.front();
  /* Pad the stick dimension if size of the dimension is
   * not a multiple of the stick size.
   */
  auto requires_padding = (stick_dim_shape % stick_size != 0);
  dev_shape.push_back(stick_size);

  for (int i = 1; i < dev_dim_order.size(); i++) {
    auto dim = dev_dim_order[i];
    if (dim == stick_dim) {
      dev_shape.push_back(requires_padding
                              ? ((stick_dim_shape / stick_size) + 1)
                              : (stick_dim_shape / stick_size));
    } else {
      dev_shape.push_back(cpu_shape[dev_dim_order[i]]);
    }
  }
  // Device shape is reversed for the G2 graph node
  std::reverse(dev_shape.begin(), dev_shape.end());
  return dev_shape;
}
/*
 * Shape of tensor on the device.
 *
 * @param tensor: CPU tensor
 * @return shape of tensor on the device
 */
auto get_device_shape(const at::Tensor* tensor) -> std::vector<int64_t> {
  const c10::IntArrayRef& sizes = tensor->sizes();
  constexpr auto bytesPerStick = 128;
  int stick_size = bytesPerStick / tensor->element_size();
  return get_device_shape(sizes, stick_size);
}
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
                        std::vector<int64_t> dev_dim_order,
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
auto get_dim_device_stride(int dim, int stick_size,
                           std::vector<int64_t> dev_dim_order,
                           std::vector<int64_t> dev_strides,
                           std::vector<int64_t> dev_shape) {
  int dev_stride;
  if (dev_strides.size() == 1) {
    dev_stride = stick_size;
  } else {
    dev_stride = dev_strides.back() * dev_shape[dev_strides.size() - 1];
  }
  return dev_stride;
}
/*
 * Size of dimension on the device.
 *
 * @param stick_size: stick length for the dtype
 * @param dim: dimensions idx
 * @param cpu_shape: dimension sizes of cpu tensor
 * @param dev_dim_order: order of tensor dimensions on device
 * @param size_less_than_stick: if the dimension is smaller than stick size
 * @return size of a dimension on the device
 */
auto get_dim_device_size(int stick_size, int dim,
                         std::vector<int64_t> cpu_shape,
                         std::vector<int64_t> dev_dim_order,
                         bool size_less_than_stick) {
  /* Returns the size for a given dimension on the device */
  int dim_size;
  if (dim == dev_dim_order.front()) {  // stick dim
    dim_size = size_less_than_stick ? 1 : cpu_shape[dim] / stick_size;
  } else {
    dim_size = cpu_shape[dim];
  }
  return dim_size;
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
                            std::vector<int64_t> dev_shape, int stick_size,
                            bool host2device) -> DataConversionStrideInfo {
  DataConversionStrideInfo stride_info;
  auto cpu_shape = sizes.vec();
  auto cpu_strides = strides.vec();
  auto dev_dim_order = get_device_layout(sizes);
  bool size_less_than_stick = cpu_shape[dev_dim_order.front()] < stick_size;

  stride_info.size_.push_back(
      size_less_than_stick ? cpu_shape[dev_dim_order.front()] : stick_size);
  stride_info.stride_src_.push_back(1);
  stride_info.stride_dst_.push_back(1);

  for (int i = 1; i < dev_dim_order.size(); i++) {
    auto& dim = dev_dim_order[i];
    auto cpu_stride =
        get_dim_cpu_stride(dim, stick_size, dev_dim_order, cpu_strides);
    auto dev_stride = get_dim_device_stride(
        dim, stick_size, dev_dim_order,
        host2device ? stride_info.stride_dst_ : stride_info.stride_src_,
        dev_shape);
    auto dim_size = get_dim_device_size(stick_size, dim, cpu_shape,
                                        dev_dim_order, size_less_than_stick);
    stride_info.size_.push_back(dim_size);
    stride_info.stride_src_.push_back(host2device ? cpu_stride : dev_stride);
    stride_info.stride_dst_.push_back(host2device ? dev_stride : cpu_stride);
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
                             std::vector<int64_t> dev_shape, int stick_size,
                             bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  std::vector<DataConversionStrideInfo> dcsi;
  auto cpu_shape = sizes.vec();
  auto dev_dim_order = get_device_layout(cpu_shape);
  bool requires_padding = cpu_shape[dev_dim_order.front()] % stick_size != 0;
  bool size_less_than_stick = cpu_shape[dev_dim_order.front()] < stick_size;
  DataConversionStrideInfo stride_info;

  stride_info = get_device_stride_info(sizes, strides, dev_shape, stick_size,
                                       host2device);
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
    for (int i = 1; i < dev_dim_order.size(); i++) {
      auto& dim = dev_dim_order[i];
      dev_offset *= pad_stride_info.size_[i];
      if (dim == dev_dim_order.front()) {
        cpu_offset *= pad_stride_info.size_[i];
        // Stick dimension is the size of the remainder of cpu_shape/stick_size
        pad_stride_info.size_[i] = 1;
        pad_stride_info.size_[0] =
            cpu_shape[dev_dim_order.front()] % stick_size;
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
auto generate_dci(const at::Tensor* tensor, bool host2device) -> std::string {
  /*   host2device = true : then 'tensor' is CPU-tensor
   *   host2device = false: then 'tensor' is Spyre-tensor
   * TODO: support strided tensors
   */
  auto str_type = torchScalarToString[tensor->scalar_type()];
  const auto [dtype_cpu, dtype_dev] = stringToDTDataFormatPair(str_type);
  std::stringstream s;
  auto cpu_shape = tensor->sizes().vec();
  auto cpu_strides = tensor->strides().vec();
  constexpr auto bytesPerStick = 128;
  int stick_size = bytesPerStick / tensor->element_size();
  std::vector<int64_t> dev_shape = get_device_shape(tensor);
  DataConversionInfo dci{};
  dci.dci_dsName_ = "DCI-Tensor-0";
  dci.isHostToSen_ = host2device;
  dci.dataformat_src_ = host2device ? dtype_cpu : dtype_dev;
  dci.dataformat_dst_ = host2device ? dtype_dev : dtype_cpu;
  std::reverse(cpu_shape.begin(), cpu_shape.end());
  std::reverse(dev_shape.begin(), dev_shape.end());
  dci.dcsi_ = get_device_stride_infos(tensor->sizes(), tensor->strides(),
                                      dev_shape, stick_size, host2device);
  dci.input_shape_ = host2device ? cpu_shape : dev_shape;
  dci.output_shape_ = host2device ? dev_shape : cpu_shape;

  dci.exportJson(s);
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

  sendnn::TensorShape dev_tensor_shape(get_device_shape(cpu_tensor));

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
  int64_t xfer_size = dev_tensor_shape.Volume() * cpu_tensor->element_size();
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
    if (host2device) {
      auto inp_node = gb.PrimaryInput("Input", cpu_ti);
      auto dci = generate_dci(cpu_tensor, host2device);
      auto dci_node = gb.SenHostCompute("Host2Sen-HostPrep", {dci_ti},
                                        {inp_node}, "SenDataConvert", dci);

      auto dev_node = gb.SenFusedDeviceCompute("SenFusedDeviceNode_0", {dci_ti},
                                               {dci_node}, sub_graph);
      gb.PrimaryOutput("Output", dev_node->OutputPort(0));
    } else {
      sendnn::Node* inp_node = gb.PrimaryInput("Input", dci_ti);
      auto dev_node = gb.SenFusedDeviceCompute("SenFusedDeviceNode_0", {dci_ti},
                                               {inp_node}, sub_graph);
      auto dci = generate_dci(dev_tensor, host2device);
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

struct SpyreGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static thread_local c10::DeviceIndex
      tls_idx;  // your TLS (or delegate to your runtime)

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }
  c10::Device exchangeDevice(c10::Device d) const override {
    auto old = getDevice();
    setDevice(d);
    return old;
  }

  c10::Device getDevice() const override {
    return {type(), tls_idx};
  }
  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    // (optionally tell your runtime to switch)
    tls_idx = d.index();
  }
  void uncheckedSetDevice(c10::Device) const noexcept {}

  c10::DeviceIndex deviceCount() const noexcept override {
    //  FIXME (tmhoangt) - return actual device count
    return 1;
  }

  // Do Spyre have streams, override
  // getStream/exchangeStream/.../recordDataPtrOnStream
  c10::Stream getStream(c10::Device device) const override {
    return c10::Stream(c10::Stream::Default::DEFAULT, device);
  }
  c10::Stream exchangeStream(c10::Stream stream) const override {
    return stream;
  }
  void recordDataPtrOnStream(const c10::DataPtr&, const c10::Stream&) const {}
};

thread_local c10::DeviceIndex SpyreGuardImpl::tls_idx = 0;

// Registration (runs at DSO load — after you import your module)
C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl);

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
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &SpyreAllocator::instance(), pu1_dks,
                                   dtype, memory_format_opt);
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
  constexpr auto bytesPerStick = 128;
  size_t size_bytes;
  size_t dev_elementsize_in_bytes = c10::elementSize(scalar_type);
  int stick_size = bytesPerStick / dev_elementsize_in_bytes;
  if (size.size() == 0) {
    size_bytes = bytesPerStick;
  } else {
    // int stick_size = 64;  // 128 / word size
    auto dev_sizes = get_device_shape(size, stick_size);
    size_bytes = bytesPerStick;
    for (auto it = dev_sizes.begin(); it != dev_sizes.end() - 1; ++it) {
      size_bytes *= *it;
    }
  }

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
    DEBUGINFO("device shape: ", get_device_shape(tmp_size, stick_size));
    DEBUGINFO("bytes on spyre: ", size_bytes);
    tensorImpl->set_sizes_and_strides(tmp_size, tmp_stride);
  } else {
    DEBUGINFO("device shape: ", get_device_shape(size, stick_size));
    DEBUGINFO("bytes on spyre: ", size_bytes);
    tensorImpl->set_sizes_and_strides(size, stride);
  }
  static_cast<SpyreTensorImpl*>(tensorImpl)->spyre_layout = device_layout;

  return tensor;
}

at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout) {
  // TEMP: forward to empty_strided for now.
  return spyre_empty_strided(size, stride, dtype, std::nullopt, std::nullopt,
                             std::nullopt);
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("as_strided", TORCH_FN(spyre_as_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
  m.impl("_copy_from", TORCH_FN(spyre_copy_from));
}

}  // namespace spyre
