# Work Division Planning

This document describes the multi-dimensional parallelization planning in Torch-Spyre, which determines how computational work is distributed across multiple cores for parallel execution.

## Motivation

Spyre provide multiple processing cores that can execute operations in parallel. To maximize performance, the compiler must decide how to divide tensor operations across these cores. The challenges are to:

1. Maximize parallelism by using as many cores as possible
2. Ensure balanced workloads across all cores
3. Maintain correctness by respecting operation semantics

As a start, the current work division planning phase analyzes each operation in the computation graph and determines an optimal parallelization strategy based on the operation type, tensor dimensions, and available hardware resources. In the future we wish to combine it with LX scratchpad optimization and consider optimal work divisions beyond a single operation.

## Core Splitting Principles

### Multi-Dimensional Splitting

Many operations have multiple dimensions that can be parallelized independently. For example, a matrix multiplication can be split along both the row and column dimensions of the output matrix. The challenge is to distribute a fixed number of cores across multiple dimensions optimally.

Currently, the planner uses a priority-based greedy algorithm:

1. Assign priorities to dimensions based on operation semantics and performance characteristics
2. Sort dimensions by priority (higher priority first) and size (larger first)
3. Allocate cores to the highest priority dimension first
4. Continue allocating remaining cores to subsequent dimensions
5. Stop when all cores are allocated or no more even divisions are possible

The product of all dimension splits equals the total number of cores used. For example, with 8 cores and three dimensions, the splits might be [4, 2, 1], meaning 4 splits on the first dimension, 2 on the second, and 1 (no split) on the third.

## Operation-Specific Strategies

### Pointwise Operations

Pointwise operations perform element-wise computations where each output element depends only on corresponding input elements at the same position. Examples include addition, multiplication, and activation functions.

**Parallelization Strategy:**
- Split along the innermost dimension (stick dimension) in the device layout
- Only applicable when all tensors have identical shapes (no broadcasting)
- Each core processes a contiguous slice of the tensor

### Reduction Operations

TODO

### Matrix Multiplication

Matrix multiplication computes C = A × B where A is M×K, B is K×N, and C is M×N. The output can be parallelized along both the M and N dimensions.

**Parallelization Strategy:**
- Prioritize the M dimension (rows) over the N dimension (columns)
- Split both dimensions when sufficient cores are available
- The K dimension (reduction dimension) is not split

**Example:** With 8 cores and output size 128×256, the planner might choose splits [4, 2], dividing rows into 4 parts and columns into 2 parts. Each core computes a 32×128 block of the output.

### Batched Matrix Multiplication

Batched matrix multiplication extends matrix multiplication with an additional batch dimension, computing multiple independent matrix multiplications in parallel.

**Parallelization Strategy:**
- Prioritize the batch dimension highest (perfect parallelism)
- Then prioritize the N dimension (columns)
- Finally consider the M dimension (rows)

**Example:** With 8 cores, batch size 4, and output size 64×128 per batch, the planner might choose splits [4, 1, 2], splitting all 4 batches and dividing columns into 2 parts. Each core processes 2 complete batches with half the columns.

## Core Division Representation

The planning phase annotates each operation with a _core division_ specification. This is a list of split counts, one per tensor (inputs and outputs), where each split count list has one element per device dimension.

For example, a matrix multiplication with 8 cores might have:
- Input 0 (left matrix): [1, 4, 1] - split dimension 1 by 4
- Input 1 (right matrix): [2, 1, 1] - split dimension 0 by 2
- Output: [2, 4, 1] - split dimensions 0 and 1 by 2 and 4 respectively

Note that the length of `core_division` of the tensor corresponds to its `device_size`.

The product of splits in any tensor equals the total cores used (4 × 2 = 8 in this example). Different tensors may have different split patterns because they have different device layouts, but the splits are coordinated to ensure each core processes corresponding slices of all tensors.

## Planning Pipeline

The work division planner processes operations in topological order, ensuring that dependencies are handled before dependent operations. For each operation:

1. Determine if the operation type supports parallelization
2. Extract tensor dimensions and device layouts
3. Identify parallelizable dimensions based on operation semantics
4. Apply the appropriate splitting strategy
5. Annotate the operation with the core division specification

The maximum number of cores is configured via an environment variable and validated to be within hardware limits. Operations that don't support parallelization or have dimensions that don't divide evenly default to single-core execution.

## Limitations and Considerations

**Current Limitations:**
- Broadcasting in pointwise operations prevents parallelization
- Only specific operation types are supported (pointwise, matrix multiplication)
- Dimensions must divide evenly by the core count
- No dynamic adjustment based on runtime conditions

**Design Considerations:**
- Static planning enables compile-time optimization and code generation
- Even division simplifies implementation for now
- Operation-specific parallelization strategies

## Configuration

Work division is controlled by the SENCORES environment variable, which specifies the maximum number of cores available for parallelization. Valid values range from 1 (no parallelization) to 32 (maximum supported cores). The planner will use up to this many cores for each operation, subject to the constraints described above.

## Future Extensions

Potential enhancements to work division planning include:

- Support for uneven splits with padding or dynamic load balancing
- Parallelization of additional operation types (convolution, pooling, etc.)
- Cross-operation optimization considering data reuse and memory hierarchy

## See Also

- [Work Division Code Generation](work_division_codegen.md) - How division plans are translated to executable code
- [Tensor Layouts](tensor_layouts.md) - Understanding device layouts and dimensions
