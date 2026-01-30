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

#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <util/sendefs.h>

#include <string>
#include <vector>

#include "spyre_storage_impl.h"

namespace spyre {

int64_t elems_per_stick(const DataFormats& df);

class SpyreTensorLayout {
 public:
  enum StickFormat {
    Dense = 0,
    Sparse,
    SparseMulti,
  };

  /**
   * The on-device size array for the (Tiled) tensor.
   * The dimensions are in decreasing stride order with the stick dimension(s)
   * last.
   */
  std::vector<int64_t> device_size;

  /**
   * Record the mapping from host size to device_size.
   * It has len(device_size) entires whose values are indices in the host size
   * vector. Stick dimensions will appear twice; non-stick dimensions will
   * appear once.
   */
  std::vector<int32_t> dim_map;
  int32_t num_stick_dims;
  StickFormat format;
  DataFormats device_dtype;

  SpyreTensorLayout() = default;
  ~SpyreTensorLayout() = default;

  /**
   * Construct a SpyreTensorLayout in generic stick format for the argument
   * host_size. Generic stick format is row major with a single dense stick
   * dimension.
   */
  SpyreTensorLayout(std::vector<int64_t> host_size, c10::ScalarType dtype) {
    init(host_size, dtype);
  }

  /**
   * Construct a SpyreTensorLayout for the argument host_size with the given
   * order of dimensions in decreasing stride order and stick format.
   */
  SpyreTensorLayout(std::vector<int64_t> host_size, c10::ScalarType dtype,
                    std::vector<int32_t> dim_order,
                    StickFormat format = Dense) {
    init(host_size, dtype, dim_order, format);
  }

  SpyreTensorLayout(std::vector<int64_t> device_size,
                    std::vector<int32_t> dim_map, int32_t num_stick_dims,
                    StickFormat format, DataFormats device_dtype)
      : device_size(device_size),
        dim_map(dim_map),
        num_stick_dims(num_stick_dims),
        format(format),
        device_dtype(device_dtype) {}

  /**
   * Initialize a SpyreTensorLayout in generic stick format for the argument
   * host_size. Generic stick format is row major with a single dense stick
   * dimension.
   */
  void init(std::vector<int64_t> host_size, c10::ScalarType dtype);

  /**
   * Initialize a SpyreTensorLayout for the argument host_size with the given
   * order of dimensions in decreasing stride order and stick format.
   */
  void init(std::vector<int64_t> host_size, c10::ScalarType dtype,
            std::vector<int32_t> dim_order, StickFormat format = Dense);

  std::string toString() const;

  std::vector<int64_t> device_strides();

  /**
   * Return the host_dim_order that can be used as an argument to
   * SpyreTensorLayout::init to create a new SpyreTensorLayout that
   * will have the same dim_map as this SpyreTensorLayout.
   */
  std::vector<int32_t> host_dim_order();

  int64_t elems_per_stick() {
    return spyre::elems_per_stick(this->device_dtype);
  }

  bool operator==(const SpyreTensorLayout& other) const {
    return this->device_size == other.device_size &&
           this->dim_map == other.dim_map &&
           this->num_stick_dims == other.num_stick_dims &&
           this->format == other.format &&
           this->device_dtype == other.device_dtype;
  }
};

/**
 * A SpyreTensorImpl extends TensorImpl by adding a SpyreTensorLayout
 * that encapsulates the on-device layout of the Tensor.
 */
class SpyreTensorImpl : public at::TensorImpl {
 public:
  SpyreTensorImpl() = delete;
  ~SpyreTensorImpl() = default;

  SpyreTensorLayout spyre_layout;

  SpyreTensorImpl(c10::Storage&& storage, c10::DispatchKeySet key_set,
                  const caffe2::TypeMeta& dtype);

  SpyreTensorImpl(c10::Storage storage, c10::DispatchKeySet key_set,
                  const caffe2::TypeMeta& dtype, SpyreTensorLayout stl);
  const at::Storage& storage() const override;

  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      const VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const;

  void shallow_copy_from(
      const c10::intrusive_ptr<at::TensorImpl>& impl) override;
};

int32_t get_device_size_in_bytes(SpyreTensorLayout stl);
SpyreTensorLayout get_spyre_tensor_layout(const at::Tensor& tensor);

}  // namespace spyre
