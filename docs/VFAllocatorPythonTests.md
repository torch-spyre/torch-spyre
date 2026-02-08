# VF Allocator Python Integration Tests

This document describes the Python integration tests for the VF (Virtual Function) allocator in [tests/test_vf_allocator_standalone.py](../tests/test_vf_allocator_standalone.py).

For detailed explanations of individual tests, see the docstrings and comments in the test file itself.

## Overview

The Python integration tests validate the VF allocator's behavior at the PyTorch level, ensuring that:
- Memory allocation and deallocation work correctly with PyTorch tensors
- Various tensor operations function properly with the spyre device
- Memory is properly reused after deallocation
- The allocator handles edge cases and stress scenarios

These tests complement the C++ unit tests by validating the full integration with PyTorch's tensor allocation system.

## Test Categories

The tests are organized into four categories:

### Initialization & Device Management
- VF mode detection from environment variable
- Basic tensor allocation on spyre device
- Zero-size tensor handling

### Memory Management
- 128-byte alignment verification
- Memory reuse after deallocation
- Sequential and interleaved allocation patterns
- Free interval merging (block coalescing)
- Mixed-size allocations
- Stress testing with many allocation cycles

### Data Type & Operation Testing
- Support for different dtypes (float16, float32, int32, bool)
- Minimum and large tensor allocations
- Tensor arithmetic operations
- Realistic allocation patterns

### Error Handling & Limits
- Out-of-memory error handling
- Memory limit stress testing

## Running the Tests

### Basic Execution

```bash
# Run all tests with pytest
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py -v

# Run a specific test
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_basic_allocation -v

# Run without pytest (direct unittest)
FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py
```

### With Verbose Output

```bash
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py -v -s
```

The `-s` flag captures print statements for tests with diagnostic output.

### Expected Output

```
tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_vf_mode_detection PASSED
tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_basic_allocation PASSED
tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_allocation_alignment PASSED
...
================================ 16 passed in X.XXs ==================================
```

## Key Testing Patterns

### Device Management
Always specify device and dtype when creating tensors:

```python
x = torch.empty(size, device="spyre", dtype=torch.float16)
```

### Garbage Collection
Trigger GC after deletions to ensure deallocation:

```python
del x
gc.collect()
```

### Device Transfer
Transfer to CPU for verification or unsupported operations:

```python
x_cpu = x.cpu()
```

### Alignment Verification
Check that allocations are properly aligned:

```python
storage_size = x.untyped_storage().nbytes()
self.assertEqual(storage_size % 128, 0)
```

## Comparison with C++ Tests

| Aspect | C++ Tests | Python Tests |
|--------|-----------|--------------|
| **Focus** | Data structure validation | Integration with PyTorch |
| **Test Level** | Unit (individual components) | Integration (full stack) |
| **Dependencies** | Minimal (standalone mode) | Requires FLEX_DEVICE=VF |

## References

- C++ Unit Tests: [VFAllocatorUnitTests.md](VFAllocatorUnitTests.md)
- Test File: [tests/test_vf_allocator_standalone.py](../tests/test_vf_allocator_standalone.py)
- VF Allocator Implementation: [torch_spyre/csrc/spyre_mem.cpp](../torch_spyre/csrc/spyre_mem.cpp)
