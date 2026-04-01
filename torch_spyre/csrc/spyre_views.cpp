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
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>
#include <util/sen_data_convert.h>

#include <vector>

#include "spyre_tensor_impl.h"

namespace spyre {

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
static at::Tensor spyre_alias_with_sizes_and_strides(const at::Tensor& self,
                                                     const Vec& sizes,
                                                     const Vec& strides) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout stl = orig_impl->spyre_layout;
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto spyre_tensor_impl_ =
      static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  spyre_tensor_impl_->set_storage_offset(self.storage_offset());
  spyre_tensor_impl_->set_sizes_and_strides(sizes, strides);
  spyre_tensor_impl_->spyre_layout = stl;
  spyre_tensor_impl_->dma_sizes = orig_impl->dma_sizes;
  spyre_tensor_impl_->dma_strides = orig_impl->dma_strides;
  return self_;
}

// specialization for symbolic shapes and strides.
// SymIntArrayRef/ArrayRef<c10::SymInt> and
// SmallVector<c10::SymInt>/SymDimVector
template <template <typename...> typename Container>
static at::Tensor spyre_alias_with_sizes_and_strides(
    const at::Tensor& self, const Container<c10::SymInt>& sizes,
    const Container<c10::SymInt>& strides) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout stl = orig_impl->spyre_layout;
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto spyre_tensor_impl_ =
      static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  spyre_tensor_impl_->set_sizes_and_strides(sizes, strides,
                                            self.sym_storage_offset());
  spyre_tensor_impl_->spyre_layout = stl;
  spyre_tensor_impl_->dma_sizes = orig_impl->dma_sizes;
  spyre_tensor_impl_->dma_strides = orig_impl->dma_strides;
  return self_;
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
  return spyre_alias_with_sizes_and_strides(self, inferred_size, *stride);
}

at::Tensor spyre_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

at::Tensor spyre__unsafe_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

at::Tensor spyre_as_strided(const at::Tensor& self, c10::IntArrayRef size,
                            c10::IntArrayRef stride,
                            std::optional<int64_t> storage_offset_) {
  SpyreTensorLayout stl =
      (static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl()))->spyre_layout;
  return as_strided_with_layout(self, size, stride, storage_offset_, stl);
}

at::Tensor as_strided_with_layout(const at::Tensor& self, c10::IntArrayRef size,
                                  c10::IntArrayRef stride,
                                  std::optional<int64_t> storage_offset_,
                                  SpyreTensorLayout device_layout) {
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  auto spyre_impl = static_cast<SpyreTensorImpl*>(result.unsafeGetTensorImpl());
  spyre_impl->spyre_layout = device_layout;
  if (device_layout == orig_impl->spyre_layout) {
    spyre_impl->dma_sizes = orig_impl->dma_sizes;
    spyre_impl->dma_strides = orig_impl->dma_strides;
  } else {
    spyre_impl->dma_sizes = size.vec();
    spyre_impl->dma_strides = stride.vec();
  }

  return result;
}

// Similar to as_strided with the following differences
// - offset is added to the existing offset (rather than replacing it)
// - view tracking is disabled similar to unsafe_view
at::Tensor reinterpret_tensor(const at::Tensor& self, c10::IntArrayRef size,
                              c10::IntArrayRef stride,
                              int64_t offset_increment) {
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout stl = orig_impl->spyre_layout;
  return reinterpret_tensor_with_layout(self, size, stride, offset_increment,
                                        stl);
}

at::Tensor reinterpret_tensor_with_layout(const at::Tensor& self,
                                          c10::IntArrayRef size,
                                          c10::IntArrayRef stride,
                                          int64_t offset_increment,
                                          SpyreTensorLayout stl) {
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout orig_stl = orig_impl->spyre_layout;
  at::Tensor self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  auto* spyre_tensor_impl_ =
      static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  spyre_tensor_impl_->set_storage_offset(self.storage_offset() +
                                         offset_increment);
  spyre_tensor_impl_->set_sizes_and_strides(size, stride);
  spyre_tensor_impl_->spyre_layout = stl;
  if (stl == orig_stl) {
    spyre_tensor_impl_->dma_sizes = orig_impl->dma_sizes;
    spyre_tensor_impl_->dma_strides = orig_impl->dma_strides;
  } else {
    spyre_tensor_impl_->dma_sizes = size.vec();
    spyre_tensor_impl_->dma_strides = stride.vec();
  }
  return self_;
}

at::Tensor spyre_alias(const at::Tensor& self) {
  return spyre_alias_with_sizes_and_strides(self, self.sym_sizes(),
                                            self.sym_strides());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("view", TORCH_FN(spyre_view));
  m.impl("_unsafe_view", TORCH_FN(spyre__unsafe_view));
  m.impl("alias", TORCH_FN(spyre_alias));
  m.impl("as_strided", TORCH_FN(spyre_as_strided));
}

}  // namespace spyre
