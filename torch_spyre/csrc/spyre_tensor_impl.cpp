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

/* Returns default tiling of tensor dimensions on the device.
 * Non-stick dimensions appear once, stick dimensions appear twice.
 * Sparse sticks are encoded using a trailing -1 in the host_dim_order.
 */
auto get_generic_stick_layout(std::vector<int32_t> host_dim_order)
    -> std::vector<int32_t> {
  std::vector<int32_t> dim_map;
  auto rank = host_dim_order.size();
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
      dim_map = {host_dim_order[1], host_dim_order[2], host_dim_order[3],
                 host_dim_order[0], host_dim_order[3]};
      break;
    case 5:
      dim_map = {host_dim_order[1], host_dim_order[2], host_dim_order[3],
                 host_dim_order[4], host_dim_order[0], host_dim_order[4]};
      break;
    case 6:
      dim_map = {host_dim_order[1], host_dim_order[2], host_dim_order[3],
                 host_dim_order[4], host_dim_order[5], host_dim_order[0],
                 host_dim_order[5]};
      break;
    default:
      std::stringstream ss;
      ss << "Unsupported tensor rank: " << std::to_string(rank);
      throw std::runtime_error(ss.str());
  }
  return dim_map;
}

std::optional<int32_t> SpyreTensorLayout::host_stick_dim() {
  int32_t stick_dim = this->dim_map.back();
  if (stick_dim == -1) {
    return std::nullopt;
  } else {
    return stick_dim;
  }
}

void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                             c10::ScalarType dtype) {
  int host_dims = static_cast<int32_t>(host_size.size());
  std::vector<int32_t> dim_order;
  for (int32_t i = 0; i < host_dims; i++) {
    dim_order.push_back(i);
  }
  init(host_size, dtype, dim_order);
}

void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                             c10::ScalarType dtype,
                             std::vector<int32_t> dim_order) {
  TORCH_CHECK((host_size.size() == dim_order.size()) ||
                  (((host_size.size() + 1) == dim_order.size()) &&
                   dim_order.back() == -1),
              "Incompatible host_size and dim_order");

  auto str_type = torchScalarToString[dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  this->device_dtype = sen_dtype_dev;

  if (host_size.size() == 0) {
    // Degenerate case of 0-dimension tensor (ie, a scalar)
    this->device_size.resize(2);
    this->dim_map.resize(2);
    this->device_size[0] = this->elems_per_stick();
    this->device_size[1] = this->elems_per_stick();
    this->dim_map[0] = -1;
    this->dim_map[1] = -1;
    return;
  }

  // Computing tiling
  this->dim_map = spyre::get_generic_stick_layout(dim_order);
  this->device_size.resize(this->dim_map.size());
  bool sparse = dim_order.back() == -1;
  auto elems_in_stick = sparse ? 1 : this->elems_per_stick();
  auto stick_dim = this->dim_map.back();
  this->device_size[this->dim_map.size() - 1] = this->elems_per_stick();
  for (int i = 0; i < this->dim_map.size() - 1; i++) {
    auto dim = this->dim_map[i];
    if (dim == stick_dim) {
      if (sparse) {
        this->device_size[i] = 1;
      } else {
        this->device_size[i] =
            (host_size[stick_dim] + elems_in_stick - 1) / elems_in_stick;
      }
    } else {
      this->device_size[i] = host_size[dim];
    }
  }
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
  ss << "], device_dtype=DataFormats.";
  ss << EnumsConversion::dataFormatsToString(this->device_dtype);
  ss << ")";
  return ss.str();
}

SpyreTensorImpl::SpyreTensorImpl(c10::Storage&& storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta& dtype)
    : TensorImpl(std::move(storage), key_set, dtype) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
}

SpyreTensorImpl::SpyreTensorImpl(at::TensorImpl::ImplType unused,
                                 c10::Storage&& storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta data_type)
    : TensorImpl(unused, std::move(storage), key_set, data_type) {
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

uint64_t get_device_size_in_bytes(SpyreTensorLayout stl) {
  uint64_t size_bytes = BYTES_IN_STICK;
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
    TORCH_CHECK(false, "Error: Device tensor does not have SpyreTensorLayout");
  }
  return stl;
}

void set_spyre_tensor_layout(const at::Tensor& tensor,
                             const SpyreTensorLayout& stl) {
  TORCH_CHECK(tensor.is_privateuseone());
  SpyreTensorImpl* impl;
  if (impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())) {
    impl->spyre_layout = stl;
  } else {
    TORCH_CHECK(false,
                "Error: Attempting to set a STL for a device tensor that does "
                "not have SpyreTensorImpl");
  }
}

};  // namespace spyre
