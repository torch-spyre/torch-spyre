# Work Division Code Generation

This document describes how work division plans are translated into executable code structures in Torch-Spyre. The code generation phase takes the core division from the planning phase and generates the necessary data structures for runtime execution on the hardware accelerator.

## Overview

After the planning phase determines how to split work across cores, the code generation phase must:

1. Map each core to the specific slice of data it should process
2. Calculate memory offsets for each core's data slice
3. Generate coordinate systems for accessing tensor elements
4. Handle padding and boundary conditions
5. Produce the SuperDSC (Super Data Stage Configuration) structures that drive hardware execution

The generated code must ensure that each core operates on the correct portion of data without overlap or gaps, and that all cores can execute independently without synchronization.

## Core-to-Slice Mapping

**For now, we use a simple mapping that makes sure each slice is mapped to a core 1-to-1. The mapping may not be optimal and we should consider deciding the mapping in an earlier pass.**

### Mapping Algorithm

Each core is assigned a unique identifier (core ID) from 0 to N-1, where N is the total number of cores used. The mapping algorithm converts this flat core ID into a multi-dimensional slice index, one index per parallelized dimension.

The mapping uses row-major ordering, where the rightmost dimension varies fastest. For example, with dimension splits [2, 4, 1]:
- Core 0 processes slice [0, 0, 0]
- Core 1 processes slice [0, 1, 0] (rightmost dimension increments)
- Core 4 processes slice [1, 0, 0] (rightmost wraps, next dimension increments)
- Core 7 processes slice [1, 3, 0]

### Memory Offset Calculation

Once a core knows which slice it should process, it must calculate the memory offset to the start of that slice. This involves:

1. Computing the stride for each dimension based on the tensor's device layout
2. For each dimension, multiplying the slice index by the stride and dividing by the number of splits
3. Summing the contributions from all dimensions

The division by the number of splits accounts for the fact that each slice is smaller than the full dimension. The result is an offset in elements from the tensor's base address, which is then converted to bytes based on the data type.

## Dimension Information

### Dimension Metadata

For each dimension in an operation, the code generator maintains:

- **Label**: A symbolic name (e.g., "mb" for mini-batch, "in" for input features, "out" for output features)
- **Index**: The position of this dimension in the ordering
- **Unpadded size**: The logical size of the dimension
- **Padded size**: The physical size including padding for hardware alignment
- **Number of splits**: How many cores this dimension is divided across
- **Scale**: A factor for coordinate transformations

This metadata is used throughout code generation to ensure consistent handling of dimensions across different tensors and operations.

### Dimension Reordering

Tensors may have different dimension orderings in host (PyTorch) layout versus device (hardware) layout. The code generator must reorder dimension information to match the device layout when generating hardware instructions, while maintaining the logical relationships between dimensions.

For example, a matrix multiplication might have logical dimensions [M, N] but device dimensions [N//64, M, 64] for the output tensor. The code generator tracks both orderings and applies the appropriate transformations.

## Operation-Specific Code Generation

### Pointwise Operations

Pointwise operations are the simplest case for work division. Each core processes a contiguous slice of the output tensor, reading corresponding slices from input tensors.

The code generator:
1. Determines which dimension is split (currently it's hard-coded as the stick dimension)
2. Creates a core-to-slice mapping where only the split dimension varies
3. Calculates memory offsets for each core
4. Generates coordinate information for accessing elements within each slice

All cores use the same computational kernel but operate on different data regions. No synchronization is needed because there are no dependencies between cores.

### Matrix Multiplication

Matrix multiplication is more complex because multiple dimensions are split and different tensors have different layouts.

For a matrix multiplication C = A × B:
- The left matrix A is split along the M dimension
- The right matrix B is split along the N dimension  
- The output matrix C is split along both M and N dimensions

The code generator must:
1. Map dimension labels (M, K, N) to device layout positions for each tensor
2. Create a core-to-slice mapping that coordinates splits across all three tensors
3. Calculate per-core memory offsets for each tensor independently
4. Generate layout specifications that describe how each tensor is organized in memory

Each core computes a rectangular block of the output matrix by reading the corresponding row slice from A and column slice from B. The K dimension (reduction dimension) is not split, so each core processes the full reduction.

### Batched Matrix Multiplication

Batched matrix multiplication extends regular matrix multiplication with an additional batch dimension. The code generation follows the same principles but with an extra dimension to coordinate.

The batch dimension is typically split first because batch elements are completely independent. The M and N dimensions are then split among the remaining cores. The code generator ensures that each core processes complete slices across all dimensions it's responsible for.

## Integration with Planning Phase

The code generation phase reads the work division plan produced by the planning phase. The plan specifies:
- The split count for each dimension of each tensor
- The total number of cores used

The code generator uses this information to:
1. Create the core-to-slice mapping
2. Calculate memory offsets
3. Generate dimension metadata
4. Produce the SuperDSC structure

The planning and code generation phases are loosely coupled: the planner makes high-level decisions about parallelization, while the code generator handles the low-level details of translating those decisions into executable structures.

## Performance Considerations

TODO

## Future Extensions

Potential enhancements to work division code generation include:

- Support for uneven splits with dynamic load balancing
- Multi-level memory hierarchy (HBM, LX)
- Fusion of multiple operations with coordinated work division

## See Also

- [Work Division Planning](work_division_planning.md) - How parallelization plans are created
- [Tensor Layouts](tensor_layouts.md) - Understanding device layouts and dimensions
