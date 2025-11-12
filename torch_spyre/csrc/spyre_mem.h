/*
 * Copyright IBM Corp. 2025
 */
#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>

namespace spyre {
at::Tensor spyre_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                               std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt);

at::Tensor spyre_copy_from(const at::Tensor& self, const at::Tensor& dst,
                           bool non_blocking);
}  // namespace spyre
