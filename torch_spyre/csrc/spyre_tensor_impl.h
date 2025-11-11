/*
 * Copyright IBM Corp. 2025
 */
#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>

#include "spyre_storage_impl.h"

namespace spyre {

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
