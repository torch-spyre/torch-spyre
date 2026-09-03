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

#include "spyre_composite_address.h"

#include <vector>

#include "spyre_allocator.h"

namespace spyre {

namespace {

// Storage-level safety checks: on-device, non-null storage and context.
// Contiguity is NOT checked here -- a non-contiguous view of a Spyre tensor
// still shares its storage's CompositeAddress with the base allocation, and
// C++ callers (D2H copies of transposed views, etc.) legitimately need it.
// The Python handle path layers a contiguity check on top for callers that
// want a well-defined chunk geometry.
SharedOwnerCtx* resolve_owner_ctx(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_composite_address: tensor must be on the Spyre device");

  const auto& storage = tensor.storage();
  TORCH_CHECK(storage.data_ptr().get() != nullptr,
              "get_composite_address: storage data pointer is null");

  auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "get_composite_address: SharedOwnerCtx is null (tensor has no "
              "Spyre device allocation)");
  return ctx;
}

}  // namespace

CompositeAddressHandle::CompositeAddressHandle(const at::Tensor& tensor)
    // Keepalive: retain the tensor's storage so the SharedOwnerCtx below (and
    // the CompositeAddress it owns) cannot be freed while this handle lives.
    : storage_(tensor.storage()), composite_addr_([&]() {
        TORCH_CHECK(tensor.is_contiguous(),
                    "get_composite_address: tensor must be contiguous");
        return &resolve_owner_ctx(tensor)->composite_addr;
      }()) {}

flex::CompositeAddress* get_composite_address(const at::Tensor& tensor) {
  return &resolve_owner_ctx(tensor)->composite_addr;
}

size_t CompositeAddressHandle::total_size() const {
  return composite_addr_->total_size();
}

size_t CompositeAddressHandle::num_chunks() const {
  return composite_addr_->chunks().size();
}

std::vector<CompositeChunkInfo> CompositeAddressHandle::chunks() const {
  const auto& chunks = composite_addr_->chunks();
  std::vector<CompositeChunkInfo> out;
  out.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    out.push_back(CompositeChunkInfo{
        /*region_id=*/chunk.addr.region_id,
        /*offset=*/chunk.addr.offset,
        /*size=*/chunk.size,
        /*domain_id=*/chunk.domain_id,
    });
  }
  return out;
}

CompositeAddressHandle get_composite_address_handle(const at::Tensor& tensor) {
  return CompositeAddressHandle(tensor);
}

}  // namespace spyre
