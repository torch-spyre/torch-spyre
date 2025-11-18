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

#include <string>
#include <utility>
#include <vector>

#include "logging.h"

namespace spyre {

#define BYTES_IN_STICK 128

/**
 * Initialize a default (generic stick) layout for a tensor of host_size.
 */
SpyreTensorLayout::SpyreTensorLayout(std::vector<int64_t> host_size,
                                     c10::ScalarType dtype)
    : device_size({}),
      device_strides({}),
      dim_map({}),
      num_stick_dims(1),
      format(StickFormat::Dense) {
  int host_dims = static_cast<int>(host_size.size());
  int device_dims = host_dims + this->num_stick_dims;
  auto elem_bytes = c10::elementSize(dtype);
  auto elems_in_stick = BYTES_IN_STICK / elem_bytes;

  this->device_size.resize(device_dims);
  this->device_strides.resize(device_dims);
  this->dim_map.resize(device_dims);

  // Stick dim
  this->dim_map[0] = host_dims - 1;
  this->dim_map[device_dims - 1] = host_dims - 1;
  this->device_size[0] =
      (host_size[host_dims - 1] + elems_in_stick - 1) / elems_in_stick;
  this->device_size[device_dims - 1] = elems_in_stick;

  // Non-stick dims
  for (auto i = 1; i < device_dims - 1; i++) {
    this->dim_map[i] = i - 1;
    this->device_size[i] = host_size[i - 1];
  }

  int64_t cur_stride = elems_in_stick;
  device_strides[device_dims - 1] = 1;
  for (auto i = device_dims - 2; i >= 0; i--) {
    this->device_strides[i] = cur_stride;
    cur_stride = cur_stride * this->device_size[i];
  }
}

SpyreTensorLayout::SpyreTensorLayout(std::vector<int64_t> device_size,
                                     std::vector<int64_t> device_strides,
                                     std::vector<int32_t> dim_map,
                                     int32_t num_stick_dims, StickFormat format)
    : device_size(device_size),
      device_strides(device_strides),
      dim_map(dim_map),
      num_stick_dims(num_stick_dims),
      format(format) {}

std::string SpyreTensorLayout::toString() const {
  std::stringstream ss;
  ss << "SpyreTensorLayout(";
  ss << "device_size=[";
  for (size_t i = 0; i < this->device_size.size(); i++) {
    ss << this->device_size[i];
    if (i < this->device_size.size() - 1) {
      ss << ", ";
    }
  }
  ss << "], device_strides=[";
  for (size_t i = 0; i < this->device_strides.size(); i++) {
    ss << this->device_strides[i];
    if (i < this->device_strides.size() - 1) {
      ss << ", ";
    }
  }
  ss << "], dim_map =[";
  for (size_t i = 0; i < this->dim_map.size(); i++) {
    ss << this->dim_map[i];
    if (i < this->dim_map.size() - 1) {
      ss << ", ";
    }
  }
  ss << "], num_stick_dims=";
  ss << this->num_stick_dims;
  if (this->format == StickFormat::Dense) {
    ss << ", format=\"Dense\"";
  } else if (this->format == StickFormat::Sparse) {
    ss << ", format=\"Sparse\"";
  } else {
    ss << ", format=\"SparseMulti\"";
  }
  ss << ")";
  return ss.str();
}

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
