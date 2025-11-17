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

#include <string>
#include <vector>

#include "spyre_storage_impl.h"

namespace spyre {

class SpyreTensorLayout {
 public:
  enum StickFormat {
    Dense = 0,
    Sparse,
    SparseMulti,
  };

  std::vector<int64_t> device_size;
  std::vector<int64_t> device_strides;
  /**
   * Record the mapping from host size to device_size.
   * It has len(device_size) entires whose values are indices in the host size
   * vector. Stick dimensions will appear twice; non-stick dimensions will
   * appear once.
   */
  std::vector<int32_t> dim_map;
  int32_t num_stick_dims;
  StickFormat format;

  /**
   * Construct a SpyreTensorLayout in generic stick format for the argument
   * host_size. Generic stick format is row major with a single dense stick
   * dimension.
   */
  SpyreTensorLayout(std::vector<int64_t> host_size, c10::ScalarType dtype);

  SpyreTensorLayout(std::vector<int64_t> device_size,
                    std::vector<int64_t> device_strides,
                    std::vector<int32_t> dim_mapping, int32_t num_stick_dims,
                    StickFormat format);

  std::string toString() const;
};

/**
 * An SpyreTensorImpl has extra information needed for Spyre tensors,
 * like what sticks are there.
 */
class SpyreTensorImpl : public at::TensorImpl {
 public:
  SpyreTensorImpl() = delete;
  ~SpyreTensorImpl() = default;

  SpyreTensorImpl(c10::Storage&& storage, c10::DispatchKeySet key_set,
                  const caffe2::TypeMeta& dtype);

  const at::Storage& storage() const override;

  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(
      const c10::intrusive_ptr<at::TensorImpl>& impl) override;
};

}  // namespace spyre
