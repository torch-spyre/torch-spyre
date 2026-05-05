# Work Division Planning

This document describes the multi-dimensional parallelization planning in
Torch-Spyre, which determines how computational work is distributed across
multiple cores for parallel execution.

## Motivation

Spyre provides multiple processing cores that can execute operations in
parallel. To maximize performance, the compiler must decide how to divide
tensor operations across these cores. The challenges are to:

1. Maximize parallelism by using as many cores as possible
2. Ensure balanced workloads across all cores
3. Respect hardware memory constraints per core
4. Maintain correctness by respecting operation semantics

The work division planning phase analyzes each operation in the computation
graph and determines a parallelization strategy based on the operation type,
tensor dimensions, device layouts, and available hardware resources. In the
future we wish to combine it with LX scratchpad optimization and consider
optimal work divisions beyond a single operation.

## Iteration Space

Each operation has an _iteration space_: the set of loop variables and their
ranges that together enumerate all output elements (for pointwise ops) or all
input elements (for reductions). For example, a 2D pointwise op over an output
of shape `[M, N]` has iteration space `{c0: M, c1: N}`.

Stick variables — iteration variables whose range maps to the innermost (stick)
device dimension of some tensor — are converted from element counts to stick
counts before planning. This ensures core splits always land on stick
boundaries, since each core must receive a whole number of sticks. When
multiple tensors of different dtypes share a stick variable, the conversion
uses the largest `elems_per_stick` across those tensors (conservative: fewer
sticks → smaller adjusted size → fewer cores assigned to that dimension).

## Hardware Memory Span Constraint

Each Spyre core has a 256 MB limit on the memory span it can access. The
_per-core span_ for a tensor is the contiguous range of device memory (in
sticks) that a single core must read or write, given a particular split
assignment. It is determined by the outermost device dimension that a core
touches: `per_core_size * stride`, where `per_core_size` is the number of
positions along that dimension each core covers.

If splitting is not applied, a large tensor may violate this limit. The
planner detects violations and computes the minimum number of slices required
on the responsible iteration variables to bring each tensor's span within the
limit.

For stick variables, valid slice counts are restricted to divisors of the
stick count, so each core always receives a whole number of sticks. If the
same iteration variable is a stick variable for one tensor and a span variable
for another, and no valid slice count satisfies both constraints simultaneously,
the compiler raises an error at compile time.

## Planning Algorithm

For each operation, `plan_splits` drives the planning in three steps:

**Step 1 — Span-required splits (`must_split_vars`).**
Process tensors one at a time. For each tensor whose per-core span exceeds
256 MB, iterate over device dimensions outer to inner and search for the best
split combination (Cartesian product of valid divisors for the variables
contributing to that dimension) that satisfies the hardware limit. The search
applies a two-tier selection: among combinations whose total core count does
not exceed `max_cores`, prefer the one with the **largest span that still fits
within the limit** (i.e. fewest cores used); if no combination brings the span
within the limit, fall back to the one with the **smallest span** (most
progress). Previously committed splits are carried forward as lower bounds,
narrowing the search for subsequent tensors.

**Step 2 — Priority ordering (`prioritize_dimensions`).**
Among the remaining dimensions (those not already committed by step 1), rank
variables for core assignment. Output dimensions (those present in the output's
device coordinates) are ranked first by decreasing stick-adjusted size.
Reduction dimensions follow, also by decreasing size. For non-matmul
reductions, reduction dimensions are excluded from candidates entirely due to a
known backend limitation.

**Step 3 — Core assignment (`multi_dim_iteration_space_split`).**
Assign cores in two passes:

1. Apply the span-required splits from step 1. These variables are excluded
   from the priority list — the two sets are disjoint.
2. Distribute remaining cores to the priority-ordered dimensions from step 2,
   greedily assigning the largest valid divisor of each dimension's size that
   fits within the remaining core budget.

The result is stored as `op_it_space_splits` on the `ComputedBuffer`. It is a `dict` keyed by the index coefficients of the buffer's read and write index expressions (computed by `splits_by_index_coeff` in [pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)), and each coefficient maps to its slice count. Downstream passes can recover an iteration-variable view by calling `apply_splits_from_index_coeff(splits, write_index, read_index, it_space)`.

:::{note}
**Two distinct memory limits.** The 256 MB span limit in step 1 is a per-core addressable device memory constraint, set by how much DDR each core can reach in its address space. It is not the same thing as the 2 MB on-core LX scratchpad. Scratchpad allocation is a separate decision, made by the `scratchpad_planning` pass when `LX_PLANNING` is enabled (see [scratchpad.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/scratchpad.py)).
:::

## Operation-Specific Strategies

### Pointwise Operations

The iteration space is that of the output tensor. All output dimensions are
candidates for splitting. There is no reduction dimension. Span-required
splits are computed jointly over all input and output tensors.

### Reduction Operations (non-matmul)

Reduction dimensions are excluded from work division candidates due to a known
backend limitation. Only output dimensions are split. Span-required splits are
asserted to not involve reduction variables; if they do, the compiler raises an
error.

### Matrix Multiplication

The iteration space covers the M (rows), K (reduction), and N (columns)
dimensions. All three are candidates. The priority order after span-required
splits is: output dimensions (M and N) by decreasing size, then K last. K is
only split when M and N cannot utilize all available cores.

### Batched Matrix Multiplication

Same as matrix multiplication, with additional batch dimensions prepended.
Batch dimensions appear as output dimensions and receive the highest priority
(largest size first), followed by N, M, and finally K.

## Configuration

Work division is controlled by the `SENCORES` environment variable, which
specifies the maximum number of cores available for parallelization. Valid
values range from 1 (no parallelization) to 32 (maximum supported cores).

## Limitations and Future Work

**Current limitations:**

- Dimensions must divide evenly by the slice count (no uneven splits)
- Only `Pointwise` and `Reduction` IR nodes are dispatched for work division;
  `ExternKernel` and `FallbackKernel` nodes are skipped
- Non-matmul reductions cannot split along the reduction dimension

**Potential future enhancements:**

- Retrieving correct padding instead of simplifying assumption
- Cross-operation optimization considering data reuse and memory hierarchy
- Integration with LX scratchpad memory planning

## See Also

- [Work Division Code Generation](work_division_codegen.md) — how division
  plans are translated to executable code
- [Tensor Layouts](../user_guide/tensors_and_layouts.md) — device layouts and
  the stick memory model
