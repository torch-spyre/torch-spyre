# VF Allocator Unit Tests Tutorial

This tutorial provides a comprehensive guide to the C++ unit tests implemented for the VF (Virtual Function) allocator in `torch_spyre/csrc/test_vf_allocator.cpp`.

## Table of Contents
1. [Overview](#overview)
2. [Test Framework](#test-framework)
3. [Core Data Structures](#core-data-structures)
4. [Implemented Tests](#implemented-tests)
5. [Building and Running](#building-and-running)
6. [Key Testing Patterns](#key-testing-patterns)

## Overview

The VF allocator unit tests validate the core memory management components used in Spyre's VF device backend. These tests are designed to be **standalone** and don't require the full allocator implementation, making them ideal for Test-Driven Development (TDD) and component isolation.

### Purpose
- Validate data structure initialization and behavior
- Test memory alignment calculations
- Verify interval merging algorithms
- Ensure ordering and comparison operators work correctly

## Test Framework

### Custom Lightweight Framework

The file includes a custom test framework activated when compiled with the `-DTEST_VF_ALLOCATOR` flag:

```cpp
class TestFramework {
  static int run_tests();
  static void register_test(const char* name, void (*func)());
};
```

### Key Components

#### 1. **TEST Macro**
Defines individual test cases:

```cpp
TEST(TestName) {
  // Test implementation
}
```

#### 2. **Assertion Macros**
- `ASSERT_EQ(a, b)` - Verifies equality
- `ASSERT_GE(a, b)` - Verifies a ≥ b
- `ASSERT_TRUE(cond)` - Verifies boolean condition

These macros throw exceptions on failure, which the framework catches and reports.

## Core Data Structures

The tests validate three fundamental allocator structures:

### 1. FreeInterval
Represents contiguous free memory regions:

```cpp
struct FreeInterval {
  size_t start;  // Starting byte offset
  size_t end;    // Ending byte offset (exclusive)
  bool operator<(const FreeInterval& other) const;
};
```

- Ordered by `start` position
- Used to track available memory ranges
- Must not overlap with other intervals

### 2. BlockInfo
Tracks allocated memory blocks:

```cpp
struct BlockInfo {
  size_t offset_init;  // Block start offset
  size_t offset_end;   // Block end offset
};
```

- Records allocation boundaries
- Used for deallocation and memory tracking

### 3. SegmentInfo
Manages entire memory segments (8GB each):

```cpp
struct SegmentInfo {
  int segment_id;                          // Unique identifier
  size_t total_size;                       // Total segment capacity
  size_t free_size;                        // Available bytes
  std::vector<BlockInfo> blocks;           // Allocated blocks
  std::set<FreeInterval> free_intervals;   // Free ranges
  std::set<size_t> free_interval_sizes;    // Size index
};
```

## Implemented Tests

### Test 1: FreeIntervalOrdering

**Purpose:** Validates that `FreeInterval` objects are correctly ordered by their start position.

**Test Case:**

```cpp
FreeInterval a{100, 200};  // [100, 200)
FreeInterval b{200, 300};  // [200, 300)
FreeInterval c{50, 100};   // [50, 100)
```

**Validations:**
- `c < a < b` (ordered by start position)
- Intervals are **non-overlapping** and adjacent
- This represents realistic free memory layout

**Why It Matters:** Correct ordering is essential for efficient best-fit and first-fit allocation strategies using `std::set`.

---

### Test 2: BlockInfoCreation

**Purpose:** Verifies proper initialization of allocated block metadata.

**Test Cases:**

1. **Default Construction:**

   ```cpp
   BlockInfo empty;
   // offset_init == 0
   // offset_end == 0
   ```

2. **Parameterized Construction:**
  
```cpp
   BlockInfo block{100, 200};
   // offset_init == 100
   // offset_end == 200
   ```

**Why It Matters:** Correct block tracking is critical for deallocation and memory leak prevention.

---

### Test 3: SegmentInfoCreation

**Purpose:** Validates segment initialization with realistic production sizes.

**Configuration:**
- **Segment Size:** 8GB (8 × 1024³ bytes)
- **Number of Segments:** 8 handlers

**Test Cases:**

1. **Single Segment Creation:**

   ```cpp
   SegmentInfo seg(0, 8GB);
   ```

   Validates:
   - Correct ID assignment
   - Total size = 8GB
   - Free size = 8GB (initially empty)
   - Empty collections (no allocations yet)

2. **Multiple Segments:**
  
```cpp
   for (int i = 0; i < 8; i++) {
     segments.emplace_back(i, 8GB);
   }
   ```
  
Validates:
- Sequential ID assignment (0-7)
- Each segment has 8GB capacity
- Total capacity: 64GB across 8 segments

**Why It Matters:** These are realistic production values for VF device memory management. 8GB segments align with hardware memory boundaries and provide efficient large-scale allocation.

---

### Test 4: AlignmentCalculation

**Purpose:** Tests memory alignment to 128-byte boundaries.

**Alignment Strategy:**

```cpp
aligned_size = ⌈nbytes / 128⌉ × 128
```

**Test Cases:**

| Input Size | Aligned Size | Reason |
|------------|--------------|--------|
| 1 byte     | 128 bytes    | Minimum allocation |
| 50 bytes   | 128 bytes    | Round up |
| 127 bytes  | 128 bytes    | Just under boundary |
| 128 bytes  | 128 bytes    | Already aligned |
| 129 bytes  | 256 bytes    | Round to next boundary |
| 256 bytes  | 256 bytes    | Already aligned |

**Why It Matters:**
- Prevents memory fragmentation
- Ensures efficient hardware access patterns
- Matches typical cache line sizes
- Simplifies pointer arithmetic in allocator

---

### Test 5: FreeIntervalMerging

**Purpose:** Demonstrates the interval coalescing algorithm that merges adjacent free memory blocks.

**Scenario:**

**Initial State:**

```
Free: [0, 100), [200, 300)
Allocated: [100, 200)
```

**Action:** Deallocate block [100, 200)

**Algorithm:**
1. Insert new freed interval [100, 200)
2. Check if previous interval [0, 100) is adjacent
   - `prev.end == new.start`? → Merge
3. Check if next interval [200, 300) is adjacent
   - `next.start == new.end`? → Merge
4. Result: Single merged interval [0, 300)

**Implementation Highlights:**

```cpp
auto it = free_intervals.lower_bound(new_range);

// Merge with previous interval
if (it != begin() && prev->end == new_range.start) {
  new_range.start = prev->start;
  erase(prev);
}

// Merge with next interval
if (it != end() && it->start == new_range.end) {
  new_range.end = it->end;
  erase(it);
}
```

**Why It Matters:**
- Prevents memory fragmentation
- Enables larger future allocations
- O(log n) complexity using `std::set`
- Critical for long-running applications

## Building and Running

### Prerequisites

Ensure you are in the project root directory:

```bash
cd /path/to/torch-spyre
```

### Option 1: Standalone (Recommended for Development)

This is the default and recommended approach for testing VF allocator components without external dependencies.

**Compilation:**

```bash
g++ -std=c++17 -DTEST_VF_ALLOCATOR torch_spyre/csrc/test_vf_allocator.cpp -o build/test_vf_allocator
```

**Running Tests:**

```bash
./build/test_vf_allocator
```

Or with environment variable explicitly set:

```bash
FLEX_DEVICE=VF ./build/test_vf_allocator
```

**Expected Output:**

```
Running VF Allocator C++ Unit Tests
====================================
Mode: Standalone (no external dependencies)
  To use gtest: compile with -DUSE_GTEST and link gtest
  To use module.h structs: compile with -DUSE_MODULE_H

Running FreeIntervalOrdering... PASSED
Running BlockInfoCreation... PASSED
Running SegmentInfoCreation... PASSED
Running AlignmentCalculation... PASSED
Running FreeIntervalMerging... PASSED

Results: 5 passed, 0 failed
```

### Option 2: With Google Test (Recommended for CI)

Use the standard gtest framework for integration with CI systems:

**Compilation:**

```bash
g++ -std=c++17 -DTEST_VF_ALLOCATOR -DUSE_GTEST \
    torch_spyre/csrc/test_vf_allocator.cpp \
    -lgtest -lgtest_main -pthread \
    -o build/test_vf_allocator_gtest
```

### Option 3: With Module.h Structs

Use actual struct definitions from `module.h` for validation:

```bash
g++ -std=c++17 -DTEST_VF_ALLOCATOR -DUSE_MODULE_H \
    -I/path/to/torch-spyre/torch_spyre/csrc \
    -I/path/to/flex/include \
    torch_spyre/csrc/test_vf_allocator.cpp \
    -o build/test_vf_allocator_integrated
```

### Option 4: SpyreAllocator Integration Tests

For tests that require the full SpyreAllocator, enable integration mode:

```bash
# Requires full build system with Flex runtime linked
cmake -DBUILD_TESTING=ON -DTEST_SPYRE_ALLOCATOR_INTEGRATION=ON ..
make test_vf_allocator
./test_vf_allocator
```

## Key Testing Patterns

### 1. Structural Testing
Tests validate data structure initialization and invariants:
- Default vs. parameterized constructors
- Initial state correctness
- Empty collection checks

### 2. Algorithmic Testing
Tests verify computational correctness:
- Alignment calculations
- Interval merging logic
- Ordering operators

### 3. Realistic Configuration
Tests use production-scale values:
- 8GB segments (not toy 1KB sizes)
- 8 segment handlers (realistic multi-segment scenarios)
- 128-byte alignment (hardware-appropriate)

### 4. Edge Case Coverage
Tests include boundary conditions:
- Already-aligned sizes
- Minimum allocations
- Adjacent interval merging
- Non-overlapping interval validation

### 5. Standalone Design
Tests are self-contained:
- No external dependencies beyond standard library
- Mock data structures for testing
- Can run before full allocator implementation
- Ideal for TDD workflow

## Future Test Additions

Consider adding tests for:
1. **Allocation/Deallocation Sequences** - Multi-step allocation patterns
2. **Fragmentation Scenarios** - Worst-case fragmentation tests
3. **Stress Tests** - Many small allocations
4. **Out-of-Memory Handling** - Exhausting segment capacity
5. **Multi-Segment Allocation** - Spanning allocations across segments
6. **Thread Safety** - Concurrent allocation/deallocation (if applicable)

## Contributing

When adding new tests:
1. Follow the existing `TEST(TestName)` pattern
2. Use descriptive names that indicate what's being tested
3. Include comments explaining the test scenario
4. Validate both success and edge cases
5. Keep tests focused on single concerns
6. Use realistic production values

## References

- Main allocator implementation: [spyre_mem.cpp](../torch_spyre/csrc/spyre_mem.cpp)
- VF device documentation: [0171-SpyreDeviceRFC.md](../RFCs/0171-SpyreDevice/0171-SpyreDeviceRFC.md)
- Pull request: [#355 - In-segment block allocation for VF](https://github.com/torch-spyre/torch-spyre/pull/355)
