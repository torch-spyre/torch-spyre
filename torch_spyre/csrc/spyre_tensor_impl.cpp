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

#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "types_mapping.h"

namespace spyre {

#define BYTES_IN_STICK 128

int64_t elems_per_stick(const DataFormats& df) {
  // TODO(dgrove-oss): DeepTools dataFormatToStickSize map is incomplete!
  if (df == DataFormats::IEEE_INT32) {
    return 32;
  }
  auto fp_elems = dataFormatToStickSize[df];
  return static_cast<int64_t>(fp_elems);
}

/* Returns default ordering of tensor dimensions on the device (generic stick).
 * Non-stick dimensions appear once, stick dimensions appear twice.
 * Must be kept in synch with host_dim_order below.
 */
auto get_generic_stick_layout(int rank, std::vector<int32_t> host_dim_order)
    -> std::vector<int32_t> {
  std::vector<int32_t> dim_map;
  switch (rank) {
    case 1:
      dim_map = {host_dim_order[0], host_dim_order[0]};
      break;
    case 2:
      dim_map = {host_dim_order[1], host_dim_order[0], host_dim_order[1]};
      break;
    case 3:
      dim_map = {host_dim_order[1], host_dim_order[2], host_dim_order[0],
                 host_dim_order[2]};
      break;
    case 4:
      dim_map = {host_dim_order[0], host_dim_order[3], host_dim_order[1],
                 host_dim_order[2], host_dim_order[3]};
      break;
    default:
      std::stringstream ss;
      ss << "Unsupported tensor rank: " << std::to_string(rank);
      throw std::runtime_error(ss.str());
  }
  return dim_map;
}

/* This is the inverse function of get_generic_stick_layout.  Keep in sync
 */
std::vector<int32_t> SpyreTensorLayout::host_dim_order() {
  auto rank = this->dim_map.size() - 1;
  std::vector<int32_t> host_dim_order;
  switch (rank) {
    case 1:
      host_dim_order = {this->dim_map[1]};
      break;
    case 2:
      host_dim_order = {this->dim_map[1], this->dim_map[2]};
      break;
    case 3:
      host_dim_order = {this->dim_map[2], this->dim_map[0], this->dim_map[3]};
      break;
    case 4:
      host_dim_order = {this->dim_map[0], this->dim_map[2], this->dim_map[3],
                        this->dim_map[4]};
      break;
    default:
      std::stringstream ss;
      ss << "Unsupported tensor rank: " << std::to_string(rank);
      throw std::runtime_error(ss.str());
  }
  return host_dim_order;
}

void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                             c10::ScalarType dtype) {
  int host_dims = static_cast<int32_t>(host_size.size());
  std::vector<int32_t> dim_order;
  for (int32_t i = 0; i < host_dims; i++) {
    dim_order.push_back(i);
  }
  init(host_size, dtype, dim_order, Dense);
}

void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                             c10::ScalarType dtype,
                             std::vector<int32_t> dim_order,
                             StickFormat format) {
  auto str_type = torchScalarToString[dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  this->device_dtype = sen_dtype_dev;

  if (host_size.size() == 0) {
    // Degenerate case of 0-dimension tensor (ie, a scalar)
    this->device_size.resize(1);
    this->dim_map.resize(1);
    this->format = Dense;
    this->num_stick_dims = 1;
    this->device_size[0] = this->elems_per_stick();
    this->dim_map[0] = 0;  // host_size has no entries!

    return;
  }

  int host_dims = static_cast<int>(host_size.size());
  int device_dims = host_dims + 1;
  auto elems_in_stick = format == Dense ? this->elems_per_stick() : 1;

  TORCH_CHECK(host_size.size() == dim_order.size(),
              "Invalid arguments: host_size.size() != dim_order.size()");

  this->device_size.resize(device_dims);
  this->dim_map = spyre::get_generic_stick_layout(host_size.size(), dim_order);
  this->format = format;
  this->num_stick_dims = 1;

  // Stick dim
  auto stick_dim = this->dim_map[this->dim_map.size() - 1];
  this->device_size[this->dim_map.size() - 1] = this->elems_per_stick();

  // Pad stick dimension if necessary
  auto requires_padding = host_size[stick_dim] % elems_in_stick != 0;
  host_size[stick_dim] =
      requires_padding
          ? ((host_size[stick_dim] / elems_in_stick) + 1) * elems_in_stick
          : host_size[stick_dim];

  // Non-stick dims
  for (int i = 0; i < this->dim_map.size() - 1; i++) {
    auto dim = this->dim_map[i];
    if (dim == stick_dim) {
      this->device_size[i] =
          format == Dense
              ? (host_size[stick_dim] + elems_in_stick - 1) / elems_in_stick
              : host_size[stick_dim];
    } else {
      this->device_size[i] = host_size[dim];
    }
  }
}

std::vector<int64_t> SpyreTensorLayout::device_strides() {
  int device_dims = static_cast<int>(this->device_size.size());
  std::vector<int64_t> strides(device_dims);

  // Stick dim
  int64_t cur_stride = this->elems_per_stick();
  strides[device_dims - 1] = 1;

  // Non-stick dims
  for (int i = device_dims - 2; i >= 0; i--) {
    strides[i] = cur_stride;
    cur_stride = cur_stride * this->device_size[i];
  }
  return strides;
}

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
    ss << ", format=StickFormat.Dense, ";
  } else if (this->format == StickFormat::Sparse) {
    ss << ", format=StickFormat.Sparse, ";
  } else {
    ss << ", format=StickFormat.SparseMulti, ";
  }
  ss << "device_dtype=DataFormats."
     << EnumsConversion::dataFormatsToString(this->device_dtype);
  ss << ")";
  return ss.str();
}

SpyreTensorImpl::SpyreTensorImpl(c10::Storage&& storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta& dtype)
    : TensorImpl(std::move(storage), key_set, dtype) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
}

SpyreTensorImpl::SpyreTensorImpl(c10::Storage storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta& dtype,
                                 SpyreTensorLayout stl)
    : TensorImpl(std::move(storage), key_set, dtype) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
  this->spyre_layout = stl;
}

// FIXME: This is currently returning cpu storage as other methods use it, but
// will return Spyre storage in a later PR
const at::Storage& SpyreTensorImpl::storage() const {
  return storage_;
}

template <typename VariableVersion>
c10::intrusive_ptr<c10::TensorImpl>
SpyreTensorImpl::shallow_copy_and_detach_core(
    const VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  if (key_set_.has(c10::DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(c10::DispatchKey::Python)) {
    auto r = pyobj_slot_.load_pyobj_interpreter()->detach(this);
    if (r) {
      r->set_version_counter(version_counter);
      r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      return r;
    }
  }
  auto impl = c10::make_intrusive<SpyreTensorImpl>(storage_, key_set_,
                                                   data_type_, spyre_layout);

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> SpyreTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(version_counter,
                                      allow_tensor_metadata_change);
}

at::intrusive_ptr<c10::TensorImpl> SpyreTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(std::move(version_counter),
                                      allow_tensor_metadata_change);
}

// FIXME: This is a temporary implementation to get the Spyre Tensor with CPU
// storage basic operation (view) to work
void SpyreTensorImpl::shallow_copy_from(
    const at::intrusive_ptr<at::TensorImpl>& impl) {
  DEBUGINFO("Parent's implementation");
  at::TensorImpl::shallow_copy_from(impl);
}

int32_t get_device_size_in_bytes(SpyreTensorLayout stl) {
  int32_t size_bytes = BYTES_IN_STICK;
  for (int i = stl.device_size.size() - 2; i >= 0; i--) {
    size_bytes *= stl.device_size[i];
  }
  return size_bytes;
}
SpyreTensorLayout get_spyre_tensor_layout(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone());
  SpyreTensorLayout stl;
  SpyreTensorImpl* impl;
  if (impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())) {
    stl = impl->spyre_layout;
  } else {
    DEBUGINFO("Warning: Device tensor does not have SpyreTensorImpl");
    stl = SpyreTensorLayout(tensor.sizes().vec(),
                            c10::typeMetaToScalarType(tensor.dtype()));
  }
  return stl;
}

};  // namespace spyre
