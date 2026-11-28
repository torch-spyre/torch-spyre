/*
 * Copyright 2026 The Torch-Spyre Authors.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spyrecode-host-functions/sendataconvert/sen_data_convert.h>
#include <unistd.h>

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "spyre_composite_address.h"
#include "spyre_pinned_allocator.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"

namespace py = pybind11;
namespace spyre {
namespace {
constexpr uint64_t kLimit = uint64_t{4} << 30;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

py::buffer_info bytes(const py::handle& value, bool writable) {
  auto info = py::cast<py::buffer>(value).request(writable);
  require(info.ndim == 1 && info.itemsize == 1 && info.strides[0] == 1 &&
              info.size > 0 && info.ptr,
          "require nonempty contiguous byte buffer");
  const auto ptr = reinterpret_cast<uintptr_t>(info.ptr);
  require(static_cast<uint64_t>(info.size) <=
              std::numeric_limits<uintptr_t>::max() - ptr,
          "buffer address overflow");
  return info;
}

struct Transfer {
  py::buffer_info source;
  CompositeAddressHandle owner;
  flex::CompositeAddress address;
  PinnedAllocation pinned;
  size_t length;
};
}  // namespace

void copy_buffer_ranges(const py::list& entries,
                        const py::object& stream_object) {
  if (entries.empty()) return;
  // Accept the existing Python Stream wrapper or its existing _C.SpyreStream.
  const auto stream_value = py::hasattr(stream_object, "_cdata")
                                ? stream_object.attr("_cdata")
                                : stream_object;
  const auto stream = py::cast<SpyreStream>(stream_value);
  const auto device = stream.device();
  require(GlobalRuntime::get() && GlobalRuntime::get()->getDeviceIndex() ==
                                      unsigned(device.index()),
          "raw DMA runtime/device mismatch");
  std::vector<Transfer> transfers;
  transfers.reserve(entries.size());
  auto* allocator =
      static_cast<SpyrePinnedAllocator*>(GetSpyrePinnedAllocator());
  const auto page = sysconf(_SC_PAGESIZE);
  require(page > 0, "cannot query page size");
  // Validate the entire batch and retain exports/storage/mapping before launch.
  for (auto value : entries) {
    auto entry = py::cast<py::tuple>(value);
    require(entry.size() == 4,
            "raw entry requires buffer, handle, offset, length");
    auto source = bytes(entry[0], false);
    auto owner = py::cast<CompositeAddressHandle>(entry[1]);
    // Use only the existing public geometry API. Keeping the typed handle
    // alive retains storage; callers must not resize/rebind it or reset the
    // runtime. The base handle exposes no device/process/generation metadata.
    const auto chunks = owner.chunks();
    require(owner.num_chunks() == 1 && chunks.size() == 1,
            "raw DMA requires a single-chunk allocation");
    const auto& chunk = chunks.front();
    const auto total = owner.total_size();
    const auto offset = py::cast<size_t>(entry[2]);
    const auto length = py::cast<size_t>(entry[3]);
    require(total && total <= kLimit && chunk.size == total && length &&
                offset <= total && length <= total - offset &&
                length <= size_t(source.size) && offset % 128 == 0 &&
                length % 128 == 0 && chunk.offset % 128 == 0 &&
                chunk.offset <= std::numeric_limits<uint64_t>::max() - total &&
                reinterpret_cast<uintptr_t>(source.ptr) % page == 0,
            "invalid/alignment-violating raw DMA range");
    auto pinned = allocator->retain(source.ptr);
    // No pageable fallback, hidden staging or remapping in this API.
    require(
        static_cast<bool>(pinned) && size_t(source.size) <= pinned.capacity,
        "raw DMA requires a live Spyre pinned buffer of sufficient capacity");
    flex::CompositeAddress address(flex::Chunk(
        flex::LogicalAddress(chunk.region_id, chunk.offset + offset), length,
        chunk.domain_id));
    transfers.push_back({std::move(source), std::move(owner),
                         std::move(address), std::move(pinned), length});
  }
  // Enqueue only. The caller retains exports, storage handles and the stream
  // until completion, including after a partially submitted batch throws.
  py::gil_scoped_release release;
  require(stream.getError() == SpyreStreamError::Success,
          "raw H2D stream is shut down");
  for (const auto& item : transfers) {
    std::unique_ptr<flex::DmaParams, decltype(&flex::destroyDmaParams)> params(
        flex::createDmaParamsWithPinnedBuffer(
            item.pinned.buffer, item.pinned.offset, item.source.size,
            item.length, true, &item.address, nullptr, false, false, false,
            std::static_pointer_cast<void>(item.pinned.owner)),
        &flex::destroyDmaParams);
    require(static_cast<bool>(params),
            "pinned DMA descriptor allocation failed");
    stream.launchH2D(params.get());
  }
}

namespace {
size_t capacity(const std::vector<int64_t>& shape, size_t width) {
  size_t size = width;
  for (auto extent : shape) {
    require(extent > 0 && extent <= (int64_t{1} << 31) - 1 &&
                static_cast<uint64_t>(extent) <= kLimit / size,
            "conversion shape exceeds byte limit");
    size *= extent;
  }
  return size;
}
void span(const std::vector<int64_t>& size, const std::vector<int64_t>& stride,
          int64_t offset, size_t elements) {
  require(
      size.size() == stride.size() && static_cast<uint64_t>(offset) < elements,
      "conversion stride rank/offset mismatch");
  uint64_t last = offset;
  uint64_t count = 1;
  for (size_t i = 0; i < size.size(); ++i) {
    require(size[i] > 0 && stride[i] > 0 &&
                static_cast<uint64_t>(stride[i]) <= kLimit &&
                static_cast<uint64_t>(size[i]) <= kLimit / count &&
                static_cast<uint64_t>(size[i] - 1) <=
                    (elements - 1 - last) / stride[i],
            "conversion region exceeds buffer shape");
    count *= size[i];
    last += (size[i] - 1) * static_cast<uint64_t>(stride[i]);
  }
}
struct Conversion {
  data_conversion_info dci;
  size_t input_bytes, output_bytes;
};
Conversion dense_conversion(const py::handle& logical_shape,
                            const std::string& dtype) {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
  throw std::runtime_error("buffer conversion requires little-endian host");
#endif
  DataFormats host;
  c10::ScalarType scalar_type;
  size_t width;
  if (dtype == "F16" || dtype == "BF16") {
    host = dtype == "F16" ? DataFormats::IEEE_FP16 : DataFormats::BFLOAT16;
    scalar_type =
        dtype == "F16" ? c10::ScalarType::Half : c10::ScalarType::BFloat16;
    width = 2;
  } else if (dtype == "F32" || dtype == "I32") {
    host = dtype == "F32" ? DataFormats::IEEE_FP32 : DataFormats::IEEE_INT32;
    scalar_type =
        dtype == "F32" ? c10::ScalarType::Float : c10::ScalarType::Int;
    width = 4;
  } else {
    throw std::runtime_error("unsupported safetensors dtype for conversion");
  }
  // Only dense logical Tensor metadata crosses the Python boundary. Reject
  // oversized shapes before the existing layout constructor multiplies them.
  require(PyList_CheckExact(logical_shape.ptr()),
          "logical shape must be a list");
  auto values = py::reinterpret_borrow<py::list>(logical_shape);
  require(values.size() <= 16, "logical shape rank exceeds limit");
  std::vector<int64_t> shape;
  for (auto value : values) {
    require(PyLong_CheckExact(value.ptr()), "logical shape requires integers");
    shape.push_back(py::cast<int64_t>(value));
  }
  const auto input_bytes = capacity(shape, width);
  const SpyreTensorLayout layout(shape, scalar_type);
  const auto output_bytes = capacity(layout.device_size, width);
  const int64_t stick = layout.elems_per_stick();
  require(
      stick > 0 && layout.element_arrangement == ElementArrangement::STANDARD,
      "conversion requires standard Spyre layout");
  std::vector<int64_t> strides(layout.device_size.size(), 1);
  for (size_t i = strides.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * layout.device_size[i];
  }

  data_conversion_info dci{};
  dci.dci_dsName_ = "dense-safetensors";
  dci.isHostToSen_ = true;
  dci.dataformat_src_ = host;
  dci.dataformat_dst_ = layout.device_dtype;
  dci.input_shape_.assign(shape.rbegin(), shape.rend());
  if (shape.empty()) dci.input_shape_ = {1};
  dci.output_shape_.assign(layout.device_size.rbegin(),
                           layout.device_size.rend());
  const int rank = shape.size();
  const int64_t columns = rank ? shape.back() : 1;
  // The standard layout splits the last logical axis into sticks and puts
  // logical axis 0 immediately outside the lanes. At most two walks are needed:
  // complete sticks and the final, partially filled stick. No padded input or
  // tensor-sized temporary is constructed.
  auto add = [&](int64_t lanes, int64_t blocks, int64_t start) {
    if (!lanes || !blocks) return;
    const int64_t block_stride = rank > 1 ? shape[0] * stick : stick;
    data_conversion_stride_info info;
    info.size_ = {lanes, blocks};
    info.stride_src_ = {1, stick};
    info.stride_dst_ = {1, block_stride};
    info.offset_src_ = start;
    info.offset_dst_ = start / stick * block_stride;
    int64_t host_stride = columns;
    for (int axis = rank - 2; axis >= 0; --axis) {
      info.size_.push_back(shape[axis]);
      info.stride_src_.push_back(host_stride);
      info.stride_dst_.push_back(axis == 0 ? stick : strides[axis - 1]);
      host_stride *= shape[axis];
    }
    span(info.size_, info.stride_src_, info.offset_src_, input_bytes / width);
    span(info.size_, info.stride_dst_, info.offset_dst_, output_bytes / width);
    dci.dcsi_.push_back(std::move(info));
  };
  add(stick, columns / stick, 0);
  add(columns % stick, 1, columns / stick * stick);
  return {std::move(dci), input_bytes, output_bytes};
}

bool disjoint(const py::buffer_info& a, const py::buffer_info& b) {
  const auto x = reinterpret_cast<uintptr_t>(a.ptr);
  const auto y = reinterpret_cast<uintptr_t>(b.ptr);
  return x + a.size <= y || y + b.size <= x;
}
}  // namespace

void convert_buffers(const py::list& entries) {
  struct Target {
    py::buffer_info output, input;
    Conversion conversion;
  };
  std::vector<Target> targets;
  targets.reserve(entries.size());
  for (auto value : entries) {
    auto entry = py::cast<py::tuple>(value);
    require(entry.size() == 4,
            "conversion requires output, input, logical shape, dtype");
    auto output = bytes(entry[0], true);
    auto input = bytes(entry[1], false);
    auto conversion =
        dense_conversion(entry[2], py::cast<std::string>(entry[3]));
    require(size_t(input.size) >= conversion.input_bytes &&
                size_t(output.size) >= conversion.output_bytes,
            "conversion buffer capacity mismatch");
    require(disjoint(input, output),
            "conversion input/output must not overlap");
    for (const auto& prior : targets) {
      require(disjoint(output, prior.input) && disjoint(output, prior.output) &&
                  disjoint(input, prior.output),
              "conversion batch buffers must not overlap");
    }
    targets.push_back(
        {std::move(output), std::move(input), std::move(conversion)});
  }
  py::gil_scoped_release release;
  for (auto& target : targets) {
    std::memset(target.output.ptr, 0, target.conversion.output_bytes);
    deeptools::ConvertData(target.output.ptr, target.conversion.output_bytes,
                           target.input.ptr, target.conversion.dci);
  }
}
}  // namespace spyre
