# VF Allocator Python Integration Tests Tutorial

This tutorial provides a comprehensive guide to the Python integration tests for the VF (Virtual Function) allocator in [tests/test_vf_allocator_standalone.py](../tests/test_vf_allocator_standalone.py).

## Table of Contents
1. [Overview](#overview)
2. [Test Framework](#test-framework)
3. [Test Categories](#test-categories)
4. [Building and Running](#building-and-running)
5. [Test Descriptions](#test-descriptions)
6. [Key Testing Patterns](#key-testing-patterns)
7. [Comparison with C++ Tests](#comparison-with-c-tests)

## Overview

The Python integration tests validate the VF allocator's behavior at the PyTorch level, ensuring that:
- Memory allocation and deallocation work correctly with PyTorch tensors
- Various tensor operations function properly with the spyre device
- Memory is properly reused after deallocation
- The allocator handles edge cases and stress scenarios

These tests complement the C++ unit tests by validating the full integration with PyTorch's tensor allocation system.

### Purpose
- Validate tensor allocation and device management
- Test memory reuse and fragmentation handling
- Verify alignment requirements are met
- Test realistic allocation patterns with varying tensor sizes
- Ensure compatibility with PyTorch operations

## Test Framework

The tests use Python's standard `unittest` framework with optional pytest integration:

```python
class TestVFAllocatorStandalone(unittest.TestCase):
    """Standalone test suite for VF allocator - can be run without pytest."""
```

### Key Components

#### 1. **Test Methods**
Each test method validates a specific aspect of allocator behavior:

```python
def test_basic_allocation(self):
    """Test basic memory allocation in VF mode."""
```

#### 2. **Environment Setup**
Tests require `FLEX_DEVICE=VF` environment variable:

```bash
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py
```

#### 3. **Tensor Operations**
Tests use PyTorch's `torch.empty()`, `torch.tensor()`, and `.to()` methods:

```python
x = torch.empty(100, device="spyre", dtype=torch.float16)
y = x.to("cpu")
```

## Test Categories

The 16 tests are organized into three categories:

### 1. **Initialization & Device Management** (3 tests)
- `test_vf_mode_detection` - Environment variable detection
- `test_basic_allocation` - Simple tensor creation
- `test_zero_size_allocation` - Empty tensor handling

### 2. **Memory Management** (8 tests)
- `test_allocation_alignment` - 128-byte alignment verification
- `test_memory_reuse` - Deallocation and reallocation patterns
- `test_sequential_allocation_deallocation` - Sequential patterns
- `test_interleaved_allocations` - Sparse deallocation patterns
- `test_free_interval_merging` - Adjacent free block merging
- `test_mixed_size_allocations` - Variable-sized allocations
- `test_allocation_after_many_deallocations` - Stress testing
- `test_concurrent_allocations` - Multiple simultaneous allocations

### 3. **Data Type & Operation Testing** (5 tests)
- `test_different_dtypes` - Multiple data type support
- `test_very_small_allocation` - Minimum size allocations
- `test_large_tensor_allocation` - Large memory allocations
- `test_tensor_operations_with_vf_allocator` - Tensor arithmetic
- `test_realistic_allocation_pattern` - Real-world scenario

## Building and Running

### Option 1: Direct Python Execution (Recommended)

```bash
# Navigate to workspace root
cd /home/chenw615/dt-inductor/torch-spyre

# Run all tests with pytest
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py -v

# Run specific test
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_basic_allocation -v

# Run without pytest (direct unittest)
FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py
```

### Option 2: With Verbose Output

```bash
FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator_standalone.py -v -s
```

The `-s` flag captures print statements from tests like `test_realistic_allocation_pattern`.

### Expected Output

```
================================== test session starts ==================================
...
tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_vf_mode_detection PASSED [ 6%]
tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_basic_allocation PASSED [ 12%]
tests/test_vf_allocator_standalone.py::TestVFAllocatorStandalone::test_allocation_alignment PASSED [ 18%]
...
================================ 16 passed in X.XXs ==================================
```

## Test Descriptions

### 1. **test_vf_mode_detection**
**Purpose:** Verify that VF mode is correctly detected from environment variable.

**Implementation:**

```python
def test_vf_mode_detection(self):
    self.assertEqual(os.environ.get("FLEX_DEVICE"), "VF")
    x = torch.empty(10, device="spyre", dtype=torch.float16)
    self.assertEqual(x.device.type, "spyre")
```

**What It Tests:**
- Environment variable is properly set
- Device creation succeeds in VF mode
- Tensor device type is correctly identified

---

### 2. **test_basic_allocation**
**Purpose:** Test basic memory allocation for a single tensor.

**Implementation:**

```python
def test_basic_allocation(self):
    x = torch.empty(100, device="spyre", dtype=torch.float16)
    self.assertEqual(x.device.type, "spyre")
    self.assertEqual(x.numel(), 100)
    self.assertEqual(x.dtype, torch.float16)
    
    storage_size = x.untyped_storage().nbytes()
    self.assertGreaterEqual(storage_size, 128)
```

**What It Tests:**
- Tensor creation on spyre device
- Correct device assignment
- Element count matches allocation
- Storage size is at least 128 bytes (minimum aligned size)

---

### 3. **test_allocation_alignment**
**Purpose:** Verify all allocations are aligned to 128-byte boundaries.

**Test Sizes:**

```
1 byte → 128 bytes
50 bytes → 128 bytes
127 bytes → 128 bytes
128 bytes → 128 bytes
129 bytes → 256 bytes
200 bytes → 256 bytes
256 bytes → 256 bytes
```

**Implementation:**

```python
for size in [1, 50, 100, 127, 128, 129, 200, 255, 256]:
    x = torch.empty(size, device="spyre", dtype=torch.float16)
    storage_size = x.untyped_storage().nbytes()
    self.assertEqual(storage_size % 128, 0)  # Must be multiple of 128
```

**Why It Matters:**
- Prevents memory fragmentation
- Matches hardware cache line sizes
- Simplifies pointer arithmetic
- Ensures consistent memory layout

---

### 4. **test_memory_reuse**
**Purpose:** Verify that deallocated memory can be reused for new allocations.

**Scenario:**
1. Allocate 5 tensors (500 elements each)
2. Delete all tensors and trigger garbage collection
3. Allocate 5 new tensors (same size)
4. Verify new allocations succeed

**Implementation:**

```python
tensors = []
for i in range(5):
    t = torch.empty(100, device="spyre", dtype=torch.float16)
    tensors.append(t)

del tensors
gc.collect()

new_tensors = []
for i in range(5):
    t = torch.empty(100, device="spyre", dtype=torch.float16)
    new_tensors.append(t)
```

**What It Tests:**
- Deallocation frees memory
- Freed memory can be reused
- No memory leaks occur
- Allocator tracks free blocks correctly

---

### 5. **test_zero_size_allocation**
**Purpose:** Test behavior with zero-size tensors.

**Implementation:**

```python
x = torch.empty(0, device="spyre", dtype=torch.float16)
self.assertEqual(x.device.type, "spyre")
self.assertEqual(x.numel(), 0)
```

**What It Tests:**
- Zero-size allocations don't crash
- Device assignment works for empty tensors
- Proper element count reporting

---

### 6. **test_different_dtypes**
**Purpose:** Verify allocation with different data types.

**Data Types Tested:**

```python
dtypes = [
    torch.float16,  # 2 bytes
    torch.float32,  # 4 bytes
    torch.int32,    # 4 bytes
    torch.bool,     # 1 byte
]
```

**What It Tests:**
- Allocator handles various data types
- Alignment works for different element sizes
- Dtype is correctly preserved

---

### 7. **test_sequential_allocation_deallocation**
**Purpose:** Test sequential allocate-deallocate patterns.

**Scenario:**

```
Iteration 1: Allocate → Use → Deallocate
Iteration 2: Allocate → Use → Deallocate
...
Iteration 5: Allocate → Use → Deallocate
```

**Implementation:**

```python
for iteration in range(5):
    x = torch.empty(500, device="spyre", dtype=torch.float16)
    x.fill_(iteration)
    x_cpu = x.cpu()
    self.assertTrue((x_cpu == iteration).all())
    del x, x_cpu
    gc.collect()
```

**What It Tests:**
- Repeated allocation/deallocation cycles work
- Garbage collection properly frees memory
- No resource exhaustion over iterations
- Data integrity during device transfers

---

### 8. **test_interleaved_allocations**
**Purpose:** Test allocation patterns with selective deallocation.

**Scenario:**

```
1. Allocate: t1 (200), t2 (200), t3 (200)
2. Deallocate: t2 (creates free gap)
3. Allocate: batch2[0] (200), batch2[1] (200)
4. Verify: t1, t3, and batch2 tensors are valid
```

**Implementation:**

```python
t1 = torch.empty(200, device="spyre", dtype=torch.float16)
t2 = torch.empty(200, device="spyre", dtype=torch.float16)
t3 = torch.empty(200, device="spyre", dtype=torch.float16)

del t2  # Create free gap
gc.collect()

batch2 = [torch.empty(200, device="spyre", dtype=torch.float16) for _ in range(2)]
```

**What It Tests:**
- Allocator handles fragmented memory
- New allocations can reuse holes
- Sparse deallocation patterns work correctly

---

### 9. **test_free_interval_merging**
**Purpose:** Test coalescing of adjacent free memory blocks.

**Scenario:**

```
Initial: [t1=1000] [t2=1000] [t3=1000]
         Allocated  Allocated  Allocated

Delete t2: [t1=1000] [FREE=1000] [t3=1000]

Delete t1: [FREE=1000] [FREE=1000] [t3=1000]
           Should merge into [FREE=2000]

Delete t3: [FREE=3000]
           Should merge into [FREE=3000]

Allocate large tensor (3000 elements) - should succeed
```

**Implementation:**

```python
t1 = torch.empty(1000, device="spyre", dtype=torch.float16)
t2 = torch.empty(1000, device="spyre", dtype=torch.float16)
t3 = torch.empty(1000, device="spyre", dtype=torch.float16)

del t2
gc.collect()

del t1
gc.collect()

del t3
gc.collect()

large = torch.empty(3000, device="spyre", dtype=torch.float16)
self.assertEqual(large.numel(), 3000)
```

**Why It Matters:**
- Adjacent free blocks merge automatically
- Prevents long-term fragmentation
- Enables allocation of large contiguous blocks
- Critical for applications with variable sizes

---

### 10. **test_very_small_allocation**
**Purpose:** Test minimum-size allocation behavior.

**Implementation:**

```python
x = torch.empty(1, device="spyre", dtype=torch.float16)
self.assertEqual(x.numel(), 1)

storage_size = x.untyped_storage().nbytes()
self.assertGreaterEqual(storage_size, 128)
self.assertEqual(storage_size % 128, 0)
```

**What It Tests:**
- Single-element tensors allocate correctly
- Minimum allocation respects 128-byte alignment
- Storage overhead for small tensors

---

### 11. **test_allocation_after_many_deallocations**
**Purpose:** Stress test allocator with many deallocation cycles.

**Scenario:**

```
for i in 1 to 50:
    Allocate tensor (100 elements)
    Deallocate tensor
    Trigger garbage collection

Finally: Allocate one more tensor to verify still working
```

**Implementation:**

```python
for _ in range(50):
    t = torch.empty(100, device="spyre", dtype=torch.float16)
    del t
    gc.collect()

final = torch.empty(100, device="spyre", dtype=torch.float16)
self.assertEqual(final.numel(), 100)
```

**What It Tests:**
- Allocator remains stable after many cycles
- No memory leaks accumulate
- Free interval management handles repeated churn
- Garbage collection doesn't cause issues

---

### 12. **test_different_dtypes** (Already described above)

---

### 13. **test_mixed_size_allocations**
**Purpose:** Test allocations with variable sizes to stress block management.

**Allocation Pattern:**

```
Sizes: [10, 100, 1000, 10000, 100000]

Allocate all → Delete 100 and 10000 → Allocate new 100 and 10000
```

**Implementation:**

```python
sizes = [10, 100, 1000, 10000, 100000]
tensors = []

for size in sizes:
    t = torch.empty(size, device="spyre", dtype=torch.float16)
    tensors.append(t)

del tensors[1], tensors[3]  # Delete 100 and 10000
gc.collect()

t_new1 = torch.empty(100, device="spyre", dtype=torch.float16)
t_new2 = torch.empty(10000, device="spyre", dtype=torch.float16)
```

**What It Tests:**
- Allocator handles mixed sizes
- Best-fit or first-fit strategies work
- Block management with non-uniform fragmentation
- Size-based allocation decisions

---

### 14. **test_concurrent_allocations**
**Purpose:** Test multiple simultaneous tensor allocations.

**Scenario:**

```
Allocate 20 tensors (100 elements each)
Fill each with its index value
Verify all are valid and have correct values
```

**Implementation:**

```python
tensors = []
num_tensors = 20

for i in range(num_tensors):
    t = torch.empty(100, device="spyre", dtype=torch.float16)
    t.fill_(i)
    tensors.append(t)

for i, t in enumerate(tensors):
    self.assertEqual(t.numel(), 100)
    t_cpu = t.cpu()
    self.assertEqual(t_cpu.numel(), 100)
```

**What It Tests:**
- Multiple allocations can coexist
- Data integrity across concurrent tensors
- No cross-talk between allocations
- Device transfer works for all tensors

---

### 15. **test_large_tensor_allocation**
**Purpose:** Test allocation of very large tensors.

**Implementation:**

```python
large_size = 10 * 1024 * 1024  # ~160MB for float16
x = torch.empty(large_size, device="spyre", dtype=torch.float16)
self.assertEqual(x.device.type, "spyre")
self.assertEqual(x.numel(), large_size)

x.fill_(1.0)
x_cpu = x.cpu()
self.assertTrue((x_cpu == 1.0).all())
```

**What It Tests:**
- Large memory blocks allocate successfully
- Data transfer from device works for large tensors
- Alignment works for large allocations
- Device fills and CPU transfers preserve data

---

### 16. **test_tensor_operations_with_vf_allocator**
**Purpose:** Test that tensor arithmetic operations work on spyre device.

**Operations Tested:**

```python
z = x + y      # Addition
w = x * y      # Element-wise multiplication
v = torch.sum(x)  # Reduction operation
```

**Implementation:**

```python
x = torch.randn(100, dtype=torch.float16).to("spyre")
y = torch.randn(100, dtype=torch.float16).to("spyre")

z = x + y
w = x * y
v = torch.sum(x)

# Verify on spyre, then transfer to CPU
self.assertEqual(z.device.type, "spyre")
z_cpu = z.cpu()
self.assertEqual(z_cpu.numel(), 100)
```

**What It Tests:**
- Device transfer (.to()) works
- Tensor operations execute on device
- Result tensors allocate correctly
- Device-to-CPU transfer preserves results

---

### 17. **test_realistic_allocation_pattern**
**Purpose:** Simulate a realistic real-world memory management scenario.

**Scenario (18 steps):**

```
1. Allocate a (1 element)
2. Allocate b (1 element)
3. Allocate c (2 elements)
4. Move b to CPU (deallocate from device)
5. Allocate d (2 elements)
6. Replace d with new d (4 elements)
7. Allocate e (64 elements = 4×16)
8. Delete a, c, d
9. Allocate k (2 elements) - from freed space
10. Allocate j (2 elements) - from freed space
11. Allocate l (4 elements) - from freed space
12. Allocate f (3 elements) - from freed space
13. Verify all tensors valid
```

**Implementation:**

```python
print("---------- allocate tensor a -------------")
a = torch.tensor([0], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor b -------------")
b = torch.tensor([0.], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor c -------------")
c = torch.tensor([1, 2], dtype=torch.float16, device="spyre")

print("\n\n---------- deallocate tensor b -------------")
b = b.to("cpu")

# ... more allocations ...

print("\n\n---------- deallocate tensor a, c, d -------------")
del a
del c
del d
gc.collect()

# ... reallocate from freed space ...

print("\n\n---------- verify all tensors are still valid -------------")
self.assertEqual(e.device.type, "spyre")
self.assertEqual(k.device.type, "spyre")
# ... more verifications ...
```

**What It Tests:**
- Real-world allocation patterns
- Mixed operations (allocate, deallocate, reallocate)
- Memory reuse for different sizes
- Device-to-CPU moves
- Long-lived allocations
- Fragmentation handling

**Output Example:**

```
---------- allocate tensor a -------------
---------- allocate tensor b -------------
---------- allocate tensor c -------------
---------- deallocate tensor b -------------
---------- allocate tensor d -------------
---------- allocate new tensor, then deallocate d -------------
---------- allocate tensor e -------------
---------- deallocate tensor a, c, d -------------
---------- allocate tensor k (from freed space) -------------
---------- allocate tensor j (from freed space) -------------
---------- allocate tensor l (from freed space) -------------
---------- allocate tensor f (from freed space) -------------
---------- verify all tensors are still valid -------------
All tensors verified successfully!
```

---

## Key Testing Patterns

### 1. **Device Management Pattern**
Always specify device and dtype:

```python
x = torch.empty(size, device="spyre", dtype=torch.float16)
```

### 2. **Garbage Collection Pattern**
Trigger GC after deletions to ensure deallocation:

```python
del x
gc.collect()
```

### 3. **Device Transfer Pattern**
For operations not supported on spyre, use CPU:

```python
# Create on CPU, transfer to device
x = torch.randn(100, dtype=torch.float16).to("spyre")

# Transfer back to CPU for verification
x_cpu = x.cpu()
```

### 4. **Storage Size Verification Pattern**
Check alignment of allocations:

```python
storage_size = x.untyped_storage().nbytes()
self.assertEqual(storage_size % 128, 0)
```

### 5. **Batch Operations Pattern**
Test multiple allocations:

```python
tensors = [torch.empty(size, device="spyre", dtype=torch.float16) 
           for _ in range(count)]
```

## Comparison with C++ Tests

| Aspect | C++ Tests | Python Tests |
|--------|-----------|--------------|
| **Focus** | Data structure validation | Integration with PyTorch |
| **Test Level** | Unit (individual components) | Integration (full stack) |
| **Memory Model** | Mock/simulated | Real PyTorch tensors |
| **Device** | Simulated | Actual spyre device |
| **Setup** | Minimal dependencies | Requires FLEX_DEVICE=VF |
| **Speed** | Fast (milliseconds) | Moderate (seconds each) |
| **Complexity** | Simple scenarios | Real-world patterns |
| **Coverage** | Algorithm correctness | Feature completeness |

### Complementary Coverage
- **C++ Tests**: Verify algorithms work in isolation
- **Python Tests**: Verify integration with PyTorch ecosystem

## Contributing

When adding new tests:
1. Follow the existing `def test_*` naming convention
2. Use descriptive docstrings explaining the scenario
3. Include comments for complex allocation patterns
4. Test both success and edge cases
5. Use realistic tensor sizes and types
6. Include garbage collection calls after deletions
7. Verify results with meaningful assertions

## References

- VF Allocator Implementation: [spyre_mem.cpp](../torch_spyre/csrc/spyre_mem.cpp)
- C++ Unit Tests: [docs/VFAllocatorUnitTests.md](VFAllocatorUnitTests.md)
- VF Device RFC: [RFCs/0171-SpyreDevice/0171-SpyreDeviceRFC.md](../RFCs/0171-SpyreDevice/0171-SpyreDeviceRFC.md)
- Test File: [tests/test_vf_allocator_standalone.py](../tests/test_vf_allocator_standalone.py)
