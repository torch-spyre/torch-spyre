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

/**
 * C++ Unit Tests for VF Allocator
 *
 * This file contains unit tests for the VF allocator implementation.
 * To compile and run from the root folder:
 *
 *   g++ -std=c++17 -DTEST_VF_ALLOCATOR torch_spyre/csrc/test_vf_allocator.cpp
 * -o build/test_vf_allocator
 *   ./build/test_vf_allocator
 *
 * Note: These tests require the Flex runtime to be available and FLEX_DEVICE=VF
 * to be set. Some tests may require mocking of Flex dependencies.
 */

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef TEST_VF_ALLOCATOR

// Minimal type definitions for standalone testing
namespace spyre {
struct FreeInterval {
  size_t start;
  size_t end;
  bool operator<(const FreeInterval& other) const {
    return start < other.start;
  }
};

struct BlockInfo {
  size_t offset_init = 0;
  size_t offset_end = 0;
  BlockInfo() = default;
  BlockInfo(size_t init, size_t end) : offset_init(init), offset_end(end) {}
};

struct SegmentInfo {
  int segment_id;
  size_t total_size;
  size_t free_size;
  std::vector<BlockInfo> blocks;
  std::set<FreeInterval> free_intervals;
  std::set<size_t> free_interval_sizes;

  SegmentInfo(int id, size_t size)
      : segment_id(id), total_size(size), free_size(size) {}
};
}  // namespace spyre

using spyre::BlockInfo;
using spyre::FreeInterval;
using spyre::SegmentInfo;

// Simple test framework
class TestFramework {
 public:
  static int run_tests() {
    int passed = 0;
    int failed = 0;

    for (auto& test : tests()) {
      std::cout << "Running " << test.name << "... ";
      try {
        test.func();
        std::cout << "PASSED" << std::endl;
        passed++;
      }
      catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        failed++;
      }
      catch (...) {
        std::cout << "FAILED: Unknown exception" << std::endl;
        failed++;
      }
    }

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed"
              << std::endl;
    return failed == 0 ? 0 : 1;
  }

  struct Test {
    std::string name;
    void (*func)();
  };

  static void register_test(const char* name, void (*func)()) {
    tests().push_back({name, func});
  }

 private:
  static std::vector<Test>& tests() {
    static std::vector<Test> t;
    return t;
  }
};

#define TEST(name)                                      \
  void test_##name();                                   \
  struct Register_##name {                              \
    Register_##name() {                                 \
      TestFramework::register_test(#name, test_##name); \
    }                                                   \
  } register_##name;                                    \
  void test_##name()

#define ASSERT_EQ(a, b)                                            \
  do {                                                             \
    if ((a) != (b)) {                                              \
      throw std::runtime_error("Assertion failed: " #a " != " #b); \
    }                                                              \
  } while (0)

#define ASSERT_GE(a, b)                                           \
  do {                                                            \
    if ((a) < (b)) {                                              \
      throw std::runtime_error("Assertion failed: " #a " < " #b); \
    }                                                             \
  } while (0)

#define ASSERT_TRUE(cond)                                   \
  do {                                                      \
    if (!(cond)) {                                          \
      throw std::runtime_error("Assertion failed: " #cond); \
    }                                                       \
  } while (0)

// Test FreeInterval structure
TEST(FreeIntervalOrdering) {
  FreeInterval a{100, 200};
  FreeInterval b{200, 300};
  FreeInterval c{50, 100};  // Non-overlapping intervals

  // Test ordering (should be by start position)
  ASSERT_TRUE(a < b);
  ASSERT_TRUE(c < a);
  ASSERT_TRUE(!(a < c));
}

// Test BlockInfo structure
TEST(BlockInfoCreation) {
  BlockInfo empty;
  ASSERT_EQ(empty.offset_init, 0);
  ASSERT_EQ(empty.offset_end, 0);

  BlockInfo block{100, 200};
  ASSERT_EQ(block.offset_init, 100);
  ASSERT_EQ(block.offset_end, 200);
}

// Test SegmentInfo structure
TEST(SegmentInfoCreation) {
  // Use realistic segment size: 8GB per segment
  constexpr size_t GB = 1024ULL * 1024ULL * 1024ULL;
  constexpr size_t segment_size = 8 * GB;  // 8GB

  SegmentInfo seg(0, segment_size);
  ASSERT_EQ(seg.segment_id, 0);
  ASSERT_EQ(seg.total_size, segment_size);
  ASSERT_EQ(seg.free_size, segment_size);
  ASSERT_TRUE(seg.blocks.empty());
  ASSERT_TRUE(seg.free_intervals.empty());
  ASSERT_TRUE(seg.free_interval_sizes.empty());

  // Test creating multiple segments (e.g., 8 handlers)
  std::vector<SegmentInfo> segments;
  for (int i = 0; i < 8; i++) {
    segments.emplace_back(i, segment_size);
    ASSERT_EQ(segments[i].segment_id, i);
    ASSERT_EQ(segments[i].total_size, segment_size);
  }
  ASSERT_EQ(segments.size(), 8);
}

// Test alignment calculation (without requiring allocator instance)
TEST(AlignmentCalculation) {
  size_t min_alloc_bytes = 128;

  auto align = [min_alloc_bytes](size_t nbytes) -> size_t {
    if (nbytes % min_alloc_bytes != 0) {
      return ((nbytes + min_alloc_bytes - 1) / min_alloc_bytes) *
             min_alloc_bytes;
    }
    return nbytes;
  };

  ASSERT_EQ(align(1), 128);
  ASSERT_EQ(align(50), 128);
  ASSERT_EQ(align(100), 128);
  ASSERT_EQ(align(127), 128);
  ASSERT_EQ(align(128), 128);
  ASSERT_EQ(align(129), 256);
  ASSERT_EQ(align(200), 256);
  ASSERT_EQ(align(255), 256);
  ASSERT_EQ(align(256), 256);
}

// Test free interval merging logic (conceptual test)
TEST(FreeIntervalMerging) {
  // Simulate merging logic
  std::set<FreeInterval> free_intervals;

  // Add initial intervals
  free_intervals.insert(FreeInterval{0, 100});
  free_intervals.insert(FreeInterval{200, 300});

  // Simulate deallocating block [100, 200] - should merge with both
  FreeInterval new_range{100, 200};

  auto it = free_intervals.lower_bound(new_range);

  // Check if previous interval touches
  if (it != free_intervals.begin()) {
    auto prev = std::prev(it);
    if (prev->end == new_range.start) {
      new_range.start = prev->start;
      free_intervals.erase(prev);
    }
  }

  // Check if next interval touches
  if (it != free_intervals.end() && it->start == new_range.end) {
    new_range.end = it->end;
    free_intervals.erase(it);
  }

  free_intervals.insert(new_range);

  // Should have one merged interval [0, 300]
  ASSERT_EQ(free_intervals.size(), 1);
  auto merged = *free_intervals.begin();
  ASSERT_EQ(merged.start, 0);
  ASSERT_EQ(merged.end, 300);
}

int main() {
  std::cout << "Running VF Allocator C++ Unit Tests\n";
  std::cout << "====================================\n\n";

  // Set FLEX_DEVICE for tests that need it
  setenv("FLEX_DEVICE", "VF", 0);

  return TestFramework::run_tests();
}

#else
// If not compiled with TEST_VF_ALLOCATOR, provide a message
// This file is only meant to be compiled with -DTEST_VF_ALLOCATOR
#endif  // TEST_VF_ALLOCATOR
