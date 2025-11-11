/*
 * Copyright IBM Corp. 2025
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

#include <algorithm>
#include <sendnn/graph/graph_builder.hpp>
#include <sendnn/runtime/graph_loader.hpp>
#include <sendnn/runtime/runtime_interface.hpp>
#include <sendnn/tensor/tensor_info.hpp>
#include <sendnn/util/status.hpp>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_sendnn_utils.h"
#include "spyre_storage_impl.h"
#include "spyre_tensor_impl.h"

namespace spyre {

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
 * This method will run a dummy graph based on a single input/output to extract
 * the proper Spyre sizes, then allocate that space on the Spyre and and set the
 * handle for the tensor to that of the memory in the Spyre. For now, it
 * allocates a CPU tensor with the correct size, as the actual storage will stay
 * on CPU until the rest of the stack is ready to filter out the allocation and
 * deallocation of memory from the graph processing.
 */
at::Tensor spyre_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                               std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt) {
  // SETUP FOR Spyre TENSOR
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  DEBUGINFO("Size:", size, ", Stride: ", stride, " on device ", device);

  const auto scalar_type = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);

  at::detail::check_size_nonnegative(size);
  caffe2::TypeMeta dtype = c10::scalarTypeToTypeMeta(scalar_type);

  // handle tensors with min stride == 64 by altering the tensor size
  std::vector<int64_t> mutable_size(size.begin(), size.end());
  std::vector<int64_t> mutable_stride(stride.begin(), stride.end());
  auto min = std::min_element(stride.begin(), stride.end());
  if (*min == 64) {
    auto index = std::distance(stride.begin(), min);
    mutable_size[index] *= 64;
    mutable_stride[index] = 1;
    size = mutable_size;
    stride = mutable_stride;
  }
  DEBUGINFO("NEW Size: ", size, ", NEW Stride: ", stride);

  // create a dummy graph builder operation to produce the right size
  std::optional<sendnn::GraphLoader> opt_gl =
      spyre::getCachedGraphLoader("Relu", size, stride);

  sendnn::GraphLoader gl;
  if (opt_gl.has_value()) {
    gl = opt_gl.value();
  } else {
    sendnn::GraphBuilder gb = spyre::createDummyOp(size);
    gl = spyre::prepareGraphLoader(&gb);
    // gl = spyre::parseGraphLoader(gl, size, stride);
    gl = spyre::parseGraphLoader(gl);
    auto predict_s = gl.Predict(sendnn::Outputs(), sendnn::Inputs());
    spyre::storeCachedGraphLoader("Relu", size, stride, gl);
  }

  // get the input size based on Spyre input
  size_t size_bytes = gl.getTempAllocationNBytes().at(0);

  auto spyre_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      &SpyreAllocator::instance(),
      /*resizeable=*/true);
  auto spyre_storage = c10::Storage(spyre_storage_impl);

  // Create the Spyre Tensor
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      std::move(spyre_storage), pu1_dks, dtype);

  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

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
  DEBUGINFO("self is on:", self.device());
  DEBUGINFO("dst is on:", dst.device());
  at::Storage source_storage;
  at::Storage dest_storage;

  if (self.is_cpu() && dst.is_privateuseone()) {
    // Copy from CPU to Spyre
    std::optional<sendnn::GraphLoader> opt_gl =
        spyre::getCachedGraphLoader("Relu", self.sizes(), self.strides());
    sendnn::GraphLoader gl;
    if (opt_gl.has_value()) {
      gl = opt_gl.value();
    } else {
      sendnn::GraphBuilder gb = spyre::createDummyOp(self.sizes());
      gl = spyre::prepareGraphLoader(&gb);
      gl = spyre::parseGraphLoader(gl, self.sizes(), self.strides());
      auto predict_s = gl.Predict(sendnn::Outputs(), sendnn::Inputs());
      spyre::storeCachedGraphLoader("Relu", self.sizes(), self.strides(), gl);
    }

    // perform eager copy
    sendnn::ConstTensor constInput =
        createInputTensor(gl, self.storage().data_ptr().get());
    constInput.SetSpyreData(
        (static_cast<SharedOwnerCtx*>(dst.storage().data_ptr().get_context()))
            ->owner);

    auto copy_status = gl.Copy(sendnn::Outputs(), {constInput}, 1);

    return dst;
  } else if (self.is_privateuseone() && dst.is_cpu()) {
    // Copy from Spyre to CPU
    std::optional<sendnn::GraphLoader> opt_gl =
        spyre::getCachedGraphLoader("Relu", self.sizes(), self.strides());
    sendnn::GraphLoader gl;
    if (opt_gl.has_value()) {
      gl = opt_gl.value();
    } else {
      sendnn::GraphBuilder gb = spyre::createDummyOp(self.sizes());
      gl = spyre::prepareGraphLoader(&gb);
      gl = spyre::parseGraphLoader(gl, self.sizes(), self.strides());
      auto predict_s = gl.Predict(sendnn::Outputs(), sendnn::Inputs());
      spyre::storeCachedGraphLoader("Relu", self.sizes(), self.strides(), gl);
    }

    auto constOutput = createOutputTensor(gl, dst.storage().data_ptr().get());
    constOutput.SetSpyreData(
        (static_cast<SharedOwnerCtx*>(self.storage().data_ptr().get_context()))
            ->owner);

    auto copy_status = gl.Copy({constOutput}, sendnn::Inputs(), 1);

    return dst;
  } else if (self.is_privateuseone() && dst.is_privateuseone()) {
    // Copy from Spyre to Spyre
    // FIXME: This will need to be addressed for proper spyre to spyre copy
    source_storage =
        (static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl()))->storage();
    dest_storage =
        (static_cast<SpyreTensorImpl*>(dst.unsafeGetTensorImpl()))->storage();
  } else {
    // For all other cases fallback to the upstream implementation
    return at::_copy_from(self, dst, non_blocking);
  }
  DEBUGINFO("Copying", source_storage.nbytes(), "bytes from",
            source_storage.device(), "to", dest_storage.device());
  std::memcpy(dest_storage.data_ptr().get(), source_storage.data_ptr().get(),
              source_storage.nbytes());
  DEBUGINFO("Finished Copying ");
  return dst;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("as_strided", TORCH_FN(spyre_as_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
  m.impl("_copy_from", TORCH_FN(spyre_copy_from));
}

}  // namespace spyre
