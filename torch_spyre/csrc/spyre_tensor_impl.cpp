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
#include "types_mapping.h"

namespace spyre {

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
      dim_map = {host_dim_order[2], host_dim_order[0], host_dim_order[2],
                 host_dim_order[1]};
      break;
    case 4:
      dim_map = {host_dim_order[3], host_dim_order[2], host_dim_order[1],
                 host_dim_order[3], host_dim_order[0]};
      break;
    default:
      std::stringstream ss;
      ss << "Unsupported tensor rank: " << std::to_string(rank);
      throw std::runtime_error(ss.str());
  }
  return dim_map;
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
    this->format = Sparse;
    this->num_stick_dims = 1;
    this->device_size[0] = 1;
    this->dim_map[0] = 0;  // host_size has no entries!
    this->size_bytes = BYTES_IN_STICK;
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
  this->size_bytes = BYTES_IN_STICK;

  // Stick dim
  auto stick_dim = dim_order[host_dims - 1];
  this->device_size[0] = elems_in_stick;

  // Non-stick dims
  for (int i = 1; i < this->dim_map.size(); i++) {
    auto dim = this->dim_map[i];
    if (dim == stick_dim) {
      this->device_size[i] =
          format == Dense
              ? (host_size[stick_dim] + elems_in_stick - 1) / elems_in_stick
              : host_size[stick_dim];
    } else {
      this->device_size[i] = host_size[dim];
    }
    this->size_bytes *= this->device_size[i];
  }
  std::reverse(this->dim_map.begin(), this->dim_map.end());
  std::reverse(this->device_size.begin(), this->device_size.end());
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

int32_t get_device_size_in_bytes(SpyreTensorLayout stl) {
  int32_t size_bytes = BYTES_IN_STICK;
  for (int i = 0; i < stl.device_size.size() - 1; i++) {
    size_bytes *= stl.device_size[i];
  }
  return size_bytes;
}
SpyreTensorLayout get_spyre_tensor_layout(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone());
  auto* tensorImpl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());

  if (tensorImpl->spyre_layout.has_value()) {
    return tensorImpl->spyre_layout.value();
  } else {  // Initialize default layout
    int stick_size = BYTES_IN_STICK / tensor.element_size();
    auto sizes = tensor.sizes().vec();
    if (sizes.empty()) {
      sizes = {stick_size};
    } else if (sizes.back() % stick_size != 0) {
      sizes.back() =
          ((sizes.back() + stick_size - 1) / stick_size) * stick_size;
    }
    tensorImpl->spyre_layout = SpyreTensorLayout(sizes, tensor.scalar_type());
    return tensorImpl->spyre_layout.value();
  }
}

};  // namespace spyre
