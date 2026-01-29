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

// Test-only header for accessing internal allocator methods
// This header should only be included in test files
//
// Usage: Include this header in your test files to access private
// SpyreAllocator members and methods. The SpyreAllocator class
// declares this as a friend class.

#include <unordered_map>
#include <vector>

#include "module.h"

namespace spyre {
namespace test {

// Forward declaration - actual definition is in spyre_mem.cpp
struct SpyreAllocator;

// Test interface for SpyreAllocator
// This class is declared as a friend of SpyreAllocator, allowing
// tests to access private members and methods.
//
// Note: Implementation of these methods should be in a separate
// test source file that has access to the full SpyreAllocator definition.
class SpyreAllocatorTestInterface {
 public:
  // Access to internal methods for testing
  static size_t setMinSpyreAllocation(const SpyreAllocator& allocator,
                                      size_t nbytes);
  static SpyreAllocator::AllocationResult findFreeBlock(
      SpyreAllocator& allocator, size_t nbytes);
  static void allocateInSegment(SpyreAllocator& allocator, SegmentInfo* seg,
                                FreeInterval range, size_t nbytes,
                                size_t& vf_offset);
  static SegmentInfo& createNewSegment(
      SpyreAllocator& allocator, size_t nbytes,
      flex::DeviceMemoryAllocatorPtr allocator_ptr, size_t& vf_offset);
  static void deallocateBlock(SpyreAllocator& allocator, SegmentInfo& seg,
                              void* ctx_void);

  // Access to internal state for verification
  static const std::vector<SegmentInfo>& getSegments(
      const SpyreAllocator& allocator);
  static size_t getSegmentSize(const SpyreAllocator& allocator);
  static bool isVFMode(const SpyreAllocator& allocator);
  static size_t getMinAllocBytes(const SpyreAllocator& allocator);
  static const std::unordered_map<void*, SegmentInfo*>& getBlockToSegmentMap(
      const SpyreAllocator& allocator);
};

}  // namespace test
}  // namespace spyre
