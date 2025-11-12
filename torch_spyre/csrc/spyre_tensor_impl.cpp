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

#include "spyre_tensor_impl.h"

#include <utility>

#include "logging.h"

namespace spyre {

/**
 * Initialize a default format (generic stick) DCI for a tensor with num_dims
 * dimensions.
 */
SpyreDCI::SpyreDCI(int32_t num_dims)
    : dim_order({}), num_stick_dims(1), format(StickFormat::Dense) {
  for (int32_t i = 0; i < num_dims; i++) {
    dim_order.push_back(i);
  }
}

SpyreDCI::SpyreDCI(std::vector<int32_t> dim_order, int32_t num_stick_dims,
                   StickFormat format)
    : dim_order(dim_order), num_stick_dims(num_stick_dims), format(format) {}

SpyreTensorImpl::SpyreTensorImpl(c10::Storage&& storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta& dtype)
    : TensorImpl(std::move(storage), key_set, dtype) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

// FIXME: This is currently returning cpu storage as other methods use it, but
// will return Spyre storage in a later PR
const at::Storage& SpyreTensorImpl::storage() const {
  return storage_;
}

// FIXME: This is a temporary implementation to get the Spyre Tensor with CPU
// storage basic operation (view) to work
c10::intrusive_ptr<at::TensorImpl> SpyreTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  DEBUGINFO("Parent's implementation");
  return at::TensorImpl::shallow_copy_and_detach(version_counter,
                                                 allow_tensor_metadata_change);
}

// FIXME: This is a temporary implementation to get the Spyre Tensor with CPU
// storage basic operation (view) to work
at::intrusive_ptr<at::TensorImpl> SpyreTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  DEBUGINFO("Parent's implementation");
  return at::TensorImpl::shallow_copy_and_detach(version_counter,
                                                 allow_tensor_metadata_change);
}

// FIXME: This is a temporary implementation to get the Spyre Tensor with CPU
// storage basic operation (view) to work
void SpyreTensorImpl::shallow_copy_from(
    const at::intrusive_ptr<at::TensorImpl>& impl) {
  DEBUGINFO("Parent's implementation");
  at::TensorImpl::shallow_copy_from(impl);
}

/**
 * Custom metadata implementations
 * These are all temporary implementation to get the Spyre Tensor with CPU
 * storage working
 */

};  // namespace spyre
