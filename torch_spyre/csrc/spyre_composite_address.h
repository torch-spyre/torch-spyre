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

#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <flex/flex.hpp>
#include <vector>

namespace spyre {

// Per-chunk geometry of a device allocation, copied out of a
// flex::CompositeAddress for read-only inspection from Python. Field types
// mirror flex::Chunk / flex::LogicalAddress.
struct CompositeChunkInfo {
  uint64_t region_id;
  uint64_t offset;
  size_t size;
  uint32_t domain_id;
};

// An opaque, read-only handle over the flex::CompositeAddress that backs a
// device("spyre") tensor's storage.
//
// The CompositeAddress lives inside the tensor's SharedOwnerCtx, which is
// owned by the tensor's c10::DataPtr. To guarantee the address can never
// dangle while the handle is alive, the handle **keeps the source tensor
// alive** by holding its storage: as long as a CompositeAddressHandle exists,
// the underlying allocation (and its CompositeAddress) is kept alive too.
//
// The handle confers no ownership of device memory beyond that keepalive and
// exposes no raw host/device pointer to Python -- only the allocation's chunk
// geometry.
class CompositeAddressHandle {
 public:
  // Builds a handle for `tensor`, capturing a keepalive reference to its
  // storage and a pointer to the CompositeAddress inside its SharedOwnerCtx.
  // `tensor` must be a contiguous tensor on the Spyre device.
  explicit CompositeAddressHandle(const at::Tensor& tensor);

  // Total physical (padded/tiled) byte size of the allocation --
  // CompositeAddress::total_size(), not numel * itemsize.
  size_t total_size() const;

  // Number of device chunks the allocation spans (1 for a single-chunk
  // allocation).
  size_t num_chunks() const;

  // Per-chunk geometry, in order.
  std::vector<CompositeChunkInfo> chunks() const;

 private:
  // Keepalive: holds the tensor's storage so the SharedOwnerCtx (and the
  // CompositeAddress it owns) outlives this handle.
  c10::Storage storage_;
  // Borrowed pointer into storage_'s SharedOwnerCtx; valid for the handle's
  // lifetime by construction (guaranteed by storage_).
  const flex::CompositeAddress* composite_addr_;
};

// Raw-pointer accessor for C++ callers: returns a non-owning pointer to the
// flex::CompositeAddress inside `tensor`'s SharedOwnerCtx. Valid only while
// the tensor's storage stays alive -- caller must keep the tensor alive for
// as long as the returned pointer is used.
//
// Only storage-level invariants are validated (on-device, non-null storage
// and context) so that C++ callers that legitimately operate on
// non-contiguous views -- for example D2H copies of a transposed view whose
// layout is described separately via DataConversionInfo -- can still reach
// the CompositeAddress backing their view's storage.
//
// The Python-facing get_composite_address_handle layers a contiguity check
// on top for callers that need a well-defined chunk geometry.
//
// Returned as non-const because downstream flex / spyre_comms APIs
// (SetSpyreDeviceAddressBorrowed, createDmaParams, etc.) take mutable
// CompositeAddress*. None of them mutate through it; the ctx member itself
// is non-const on SharedOwnerCtx.
//
// This is the single definition used by every C++ site that needs the
// underlying CompositeAddress (distributed, job_plan, spyre_mem, spyre_stream).
flex::CompositeAddress* get_composite_address(const at::Tensor& tensor);

// Python-facing accessor: returns a read-only CompositeAddressHandle over the
// device address backing `tensor`'s storage. The returned handle keeps
// `tensor`'s allocation alive (see CompositeAddressHandle). This is the one
// tensor-aware step the KV-offload design assigns to torch-spyre. The pybind
// binding exposes this under the plain name "get_composite_address".
CompositeAddressHandle get_composite_address_handle(const at::Tensor& tensor);

}  // namespace spyre
