# VF Allocator C++ Unit Tests

This document describes the C++ unit tests for the VF (Virtual Function) allocator in [torch_spyre/csrc/test_vf_allocator.cpp](../torch_spyre/csrc/test_vf_allocator.cpp).

For detailed explanations of individual tests, see the comments in the test file itself.

## Overview

The C++ unit tests validate core memory management components used in Spyre's VF device backend. These tests are designed to be **standalone** and don't require the full allocator implementation, making them ideal for Test-Driven Development (TDD) and component isolation.

### Purpose
- Validate data structure initialization and behavior
- Test memory alignment calculations
- Verify interval merging algorithms
- Ensure ordering and comparison operators work correctly

## Core Data Structures

The tests validate three fundamental allocator structures:

### FreeInterval
Represents contiguous free memory regions, ordered by start position:

```cpp
struct FreeInterval {
  size_t start;  // Starting byte offset
  size_t end;    // Ending byte offset (exclusive)
};
```

### BlockInfo
Tracks allocated memory block boundaries:

```cpp
struct BlockInfo {
  size_t offset_init;  // Block start offset
  size_t offset_end;   // Block end offset
};
```

### SegmentInfo
Manages entire memory segments (8GB each):

```cpp
struct SegmentInfo {
  int segment_id;
  size_t total_size;
  size_t free_size;
  std::vector<BlockInfo> blocks;
  std::set<FreeInterval> free_intervals;
};
```

## Test Categories

- **FreeIntervalOrdering** - Validates interval ordering by start position
- **BlockInfoCreation** - Verifies block metadata initialization
- **SegmentInfoCreation** - Tests segment initialization with production sizes (8GB)
- **AlignmentCalculation** - Validates 128-byte alignment logic
- **FreeIntervalMerging** - Tests adjacent free block coalescing algorithm

## Building and Running

### Standalone Mode (Recommended)

No external dependencies required:

```bash
# Compile
g++ -std=c++17 -DTEST_VF_ALLOCATOR torch_spyre/csrc/test_vf_allocator.cpp -o build/test_vf_allocator

# Run
./build/test_vf_allocator
```

### With Google Test

For CI integration:

```bash
g++ -std=c++17 -DTEST_VF_ALLOCATOR -DUSE_GTEST \
    torch_spyre/csrc/test_vf_allocator.cpp \
    -lgtest -lgtest_main -pthread \
    -o build/test_vf_allocator_gtest
```

### With Module.h Structs

Use actual struct definitions from module.h:

```bash
g++ -std=c++17 -DTEST_VF_ALLOCATOR -DUSE_MODULE_H \
    -I<path_to_csrc> -I<path_to_flex_include> \
    torch_spyre/csrc/test_vf_allocator.cpp \
    -o build/test_vf_allocator_integrated
```

### Expected Output

```
Running VF Allocator C++ Unit Tests
====================================
Mode: Standalone (no external dependencies)

Running FreeIntervalOrdering... PASSED
Running BlockInfoCreation... PASSED
Running SegmentInfoCreation... PASSED
Running AlignmentCalculation... PASSED
Running FreeIntervalMerging... PASSED

Results: 5 passed, 0 failed
```

## Comparison with Python Tests

| Aspect | C++ Tests | Python Tests |
|--------|-----------|--------------|
| **Focus** | Data structure validation | Integration with PyTorch |
| **Test Level** | Unit (individual components) | Integration (full stack) |
| **Dependencies** | Minimal (standalone mode) | Requires FLEX_DEVICE=VF |

## References

- Python Integration Tests: [VFAllocatorPythonTests.md](VFAllocatorPythonTests.md)
- Test File: [torch_spyre/csrc/test_vf_allocator.cpp](../torch_spyre/csrc/test_vf_allocator.cpp)
- VF Allocator Implementation: [torch_spyre/csrc/spyre_mem.cpp](../torch_spyre/csrc/spyre_mem.cpp)
