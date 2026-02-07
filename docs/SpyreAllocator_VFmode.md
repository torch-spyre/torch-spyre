# SpyreAllocator VF Mode: Design Summary

## Overview

`SpyreAllocator` is a custom memory allocator managing device memory allocation for tensors that execute on Spyre hardware, supporting both Physical Function (PF) and Virtual Function (VF) modes.

## Core Functionalities

### 1. Dual-Mode Memory Management

The new implementation adds VF Mode support, allowing the allocator to operate in two distinct modes. Mode selection is based on the `FLEX_DEVICE` environment variable. The selection takes place once, at the time `SpyreAllocator` is constructed.

#### **PF (Physical Function) Mode**
- This mode has been left mostly functionally-unaltered from previous implementations. The only I/O difference is a new `vf_offset` attribute in SharedOwnerCtx. `vf_offset` is set to 0 for PF Mode and should be ignored
- **Functionalities**
   - Each tensor allocation directly requests memory from Spyre hardware via `TryAllocate()`
   - Allocation on demand, deallocation when tensor is destroyed
   - Each allocation is independent

#### **VF (Virtual Function) Mode**
Implemented in PR #355
- **Functionalities**
   - Creates a pre-allocated fixed pool of large memory segments (default: 8 segments × 12 GB each)
   - Divides segments into blocks for individual tensor allocations
   - Freed blocks return to the pool for future allocations
   - Merges adjacent free blocks to reduce fragmentation
- **Additional details**
   - Called via `vf_allocation` method, through `allocate` (a request for `nbytes` allocation)
   - Returns `at::DataPtr` smart pointer comprising raw memory buffer, context/metadata
pointer needed by the deleter, cleanup function, and device identifier
   - Context is `SharedOwnerCtx` object, now augmented with `vf_offset` (block offset)
   - All blocks within the same segment share the same `flex::DeviceMemoryAllocationPtr data` pointer

### 2. Memory Segment Architecture (VF Mode)

**MemorySegment Structure:**
- Represents one contiguous allocation on Spyre hardware
- Tracks total size, free size, and allocation index
- Maintains ordered set of `MemoryBlock` objects (both free and occupied)
- Uses `std::multiset` to track free block sizes for efficient lookup
- Maps `SharedOwnerCtx` pointers to blocks for O(1) deallocation

**MemoryBlock Structure:**
- Represents contiguous memory intervals within a segment, either free or occupied
- Stores start/end offsets and free/occupied status
- Ordered by start offset for efficient merging operations

### 3. Allocation Strategy (VF Mode)

**Segment Load Balancing and Block First-Fit:**
1. **Segment selection**: Choose segment with most free memory (not round-robin)
2. **Block selection**: Find first free block (lowest offset) that fits the request
3. **Block splitting**: If selected block is larger than needed, split into occupied + free blocks
4. **Alignment**: All allocations aligned to 128-byte boundaries (Spyre hardware requirement)

### 4. Deallocation and Defragmentation (VF Mode)

**Coalescing Strategy:**
- When a block is freed, merge with adjacent free blocks (both previous and next)
- Uses `std::set::lower_bound()` iterator for efficient neighbor lookup
- Updates free block sizes multiset and metadata for selected segment
- Thread-safe via mutex protection

## VF Mode Design Considerations

### Segments Pre-allocation

- All segments are allocated at the time of the first call to `allocate` (see `initializeSegments`)
- Multi-tenant behavior may be incompatible with a fixed pre-allocation of the full amount of memory
- Alternative allocation scheme could allocate one segment at the time; if allocation fails, do
not attempt further segment allocations for given session and begin load balancing strategy; or,
if allocation fails, attempt new segment allocation for smaller amount
- Even allocating a segment at the time, assuming large segment size allocation, max memory
allocation will be achieved after few calls to `allocate`

### Segment number (8) and size (12 GB) = 96 GB allocation

- Factors empirically determined based on pre-allocation of full amount (tested 8 x 14 GB but allocation crashed)
- Unclear why this specific threshold was observed, upper limit is 128 GB
- If multi-tenant behavior prevents fixed pre-allocation, how would full allocation limit
change over time? Within what boundaries?

### Alignment to 128 bytes

- Supposedly, a Spyre hardware requirement (value unconfirmed)
- `nbytes` alignment implemented in `setMinSpyreAllocation`
- Unclear if `allocate` can ever receive an `nbytes` request that doesn't satisfy this requirement

### Extensibility

- Further updating the allocation design only needs to alter `SpyreAllocator` (in `spyre_mem.cpp`)
and core structures `MemoryBlock`, `MemorySegment`, and `SharedOwnerCtx` (in `module.h`).
There are no other dependencies.
- Current VF Mode implementation preserves `allocate` I/O used by PF Mode

## Design Strengths and Limitations

### Strengths

1. Supports both PF (preserved) and VF allocation strategies
2. Achieves load balancing by distributing allocations across segments based on available memory
3. First-fit strategy within best segment provides predictable allocation
4. Reduces fragmentation by coalescing freed memory blocks
5. Mutex protection ensures correctness in multi-threaded environments
6. Optional debug logging via `TORCH_SPYRE_ALLOC_DEBUG` environment variable

### Limitations

1. Fixed segment size and number of segments is unflexible, may waste memory, and lead to
crashes in multi-tenant scenarios; similar considerations apply to allocating all segments
on first use
2. First-fit block assignment could be improved upon by changing to best-fit assignment
3. Freed blocks are merged but not relocated, potentially leading to fragmentation over time
despite coalescing; relocation overhead may be substantial though
4. Single SpyreAllocator instance for all devices may need rework in multi-device scenarios
5. All allocations/deallocations are blocking (synchronous); could lead to delays under
high allocation rate; may be possible to improve on this by limiting the mutex scope
6. No handling of out of memory errors (all segments full) beyond TORCH_CHECK validation
that allocation succeeded
7. Logging is all-or-nothing; proposed to implement a finer-grain approach throughout the
whole codebase
