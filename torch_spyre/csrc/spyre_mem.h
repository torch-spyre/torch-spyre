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

#include "module.h"

namespace spyre {

at::Tensor spyre_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                               std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt);

at::Tensor spyre_copy_from(const at::Tensor& self, const at::Tensor& dst,
                           bool non_blocking);

/**
 * Uses flex::RuntimeStream::fillAsync() to perform the fill entirely
 * on-device without allocating a host buffer or performing an H2D DMA.
 *
 * @param self The spyre tensor to fill (in-place)
 * @param value The fill value as a double (converted to the correct hardware
 *              pattern based on the tensor's dtype)
 * @return The filled tensor (same as self)
 */
at::Tensor spyre_fill_tensor(const at::Tensor& self, double value);

/**
 * Encode a CPU fp16 tensor (any strides) as SEN169_FP16 device bits, using
 * the intra-op threads. SEN169_FP16 is DeepTools' DLF16 (1 sign, 6 exponent
 * bits with bias 31, 9 mantissa bits); every element goes through a table
 * built from DeepTools' converter deeptools::Ieeef16BinToFp16bin. Returns a
 * contiguous CPU int16 tensor of the same shape.
 */
at::Tensor fp16_to_sen169_bits(const at::Tensor& src);

/**
 * Copy contiguous CPU tensor ``src`` as is into the whole device allocation
 * of ``dst`` (no conversion, no re-layout). ``src`` must hold the
 * allocation's bytes in device order. Synchronous.
 */
void spyre_copy_bytes_to_device(const at::Tensor& src, const at::Tensor& dst);

class SpyreTensorLayout;
at::Tensor spyre_empty_with_layout(
    c10::IntArrayRef size, c10::IntArrayRef stride, c10::ScalarType dtype,
    SpyreTensorLayout device_layout,
    std::optional<c10::Device> device_opt = std::nullopt);

at::Tensor empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor py_empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

auto generate_dci(const at::Tensor* cpu_tensor, const at::Tensor* dev_tensor,
                  SpyreTensorLayout stl, bool host2device)
    -> DataConversionInfo;
}  // namespace spyre
