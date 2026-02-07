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

#include <flex/runtime.hpp>
#include <memory>
#include <set>
#include <unordered_map>

#include "module.h"

namespace spyre {

struct MemoryBlock {
  // Contiguous interval of memory within a Segment (either free or occupied).
  // VF only.

  size_t start;  // block starting offset
  size_t end;    // block ending offset (one past last byte)
  bool is_free;  // block represents memory occupied or free

  MemoryBlock() : start(0), end(0), is_free(true) {}
  MemoryBlock(size_t s, size_t e, bool free = true, void* ctx = nullptr)
      : start(s), end(e), is_free(free) {}

  size_t size() const {
    return end - start;
  }

  bool operator<(const MemoryBlock& other) const {
    return start < other.start;  // for std::set ordering
  }
};

struct MemorySegment {
  // One contiguous allocation on Spyre, via TryAllocate. VF only.
  // Allocated memory is subdivided into MemoryBlocks (free or occupied).

  size_t segment_id;  // same as alloc_idx. Type: AIUMsg::V1::AllocationIndex =
                      // senlib::v2::LittleEndian<unsigned long>
  flex::DeviceMemoryAllocationPtr
      data;  // in common across all ShareOwnerCtx associated with the same
             // MemorySegment

  size_t total_size;  // total allocated memory for this Segment
  size_t free_size;   // total available memory

  std::set<MemoryBlock>
      blocks;  // all memory blocks (free and occupied), ordered by start offset
  std::unordered_map<SharedOwnerCtx*, MemoryBlock*>
      ctx_to_block;  // quick lookup from context to occupied block
  std::multiset<size_t>
      free_sizes;  // track sizes of all free blocks for quick lookup

  MemorySegment(size_t idx, size_t sz)
      : segment_id(idx), total_size(sz), free_size(sz) {}
};

}  // namespace spyre
