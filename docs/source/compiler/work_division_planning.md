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

| Variable | Default | Purpose |
|---|---|---|
| `SENCORES` | 32 | Maximum number of cores for parallelization (1–32) |
| `SPYRE_INDUCTOR_IGNORE_HINTS` | 0 | Set to `1` to ignore all `work_division_hint` annotations and use the automatic planner |

## User-Specified Work Division Hints

For debugging and experimentation, users can override the automatic split
decisions on a per-operation basis using the `work_division_hint` context
manager.

### Usage

```python
from torch_spyre._inductor.work_division_hint import work_division_hint

@torch.compile
def model(x, y):
    with work_division_hint([2, 1, 2]):
        out = x @ y  # M split by 2, N unsplit, K split by 2
    return out
```

Different operations can receive different hints by using separate context
blocks:

```python
@torch.compile(options={"epilogue_fusion": False})
def linear_with_bias(x, w, b):
    with work_division_hint([4, 2, 1]):
        mm_out = x @ w.T        # matmul: M=4, N=2, K=1
    with work_division_hint([4, 2]):
        out = mm_out + b         # bias add: M=4, N=2
    return out
```

The hint is a list of split factors in **iteration-space order**: output
dimensions first (matching the output tensor shape), then reduction
dimensions appended.

| Operation | Iteration space | Hint format |
|---|---|---|
| 2D matmul `x:(M,K) @ y:(K,N)` | M, N, K | `[M_split, N_split, K_split]` |
| 3D batched matmul `x:(B,M,K) @ y:(K,N)` | B, M, N, K | `[B_split, M_split, N_split, K_split]` |
| 2D pointwise `x:(A,B) + y:(A,B)` | A, B | `[A_split, B_split]` |

Operations outside the context manager are unaffected and use the normal
planning algorithm.

### Validation

The compiler validates the hint at planning time. Structurally invalid hints
(wrong number of dimensions, non-positive values) are rejected with a warning,
and the operation falls back to automatic planning. Soft violations are warned
but still applied:

- Split factor does not evenly divide the dimension size
- Total core count (product of all splits) exceeds `SENCORES`
- Per-core memory span exceeds the 256 MB hardware limit

### Mechanism

The context manager wraps `torch.fx.traceback.annotate`, which stores the
hint in the FX graph node metadata under `node.meta["custom"]` during
`torch.compile` tracing. Users do not need to interact with this metadata
directly.

#### Propagation through decompositions

When a high-level operation decomposes into multiple lower-level ops (for
example, `F.linear(x, w, b)` becomes `mm` + `add`), PyTorch's
`preserve_node_meta` mechanism ensures that the `custom` metadata (including
the hint) is copied to all decomposed FX nodes automatically. Dynamo
activates `preserve_node_meta` whenever it encounters `annotate`, so users
do not need to enable it manually.

If the hint's length does not match a decomposed op's iteration-space
dimensionality, the compiler logs a warning and falls back to the automatic
planner for that op. In practice this means the hint naturally targets the
op it was designed for, while mismatched ops are left to the heuristic.

#### Recovery across re-tracing passes

Between tracing and core division planning, several graph transformation
passes (AOT Autograd re-tracing, post-grad passes) may create replacement
nodes that lose the original `custom` metadata despite `preserve_node_meta`.
The compiler recovers hints through a multi-stage propagation pipeline:

1. **Pre-grad collection** (`collect_work_division_hints`) — At
   `CustomPreGradPasses` time, before AOT Autograd, the compiler snapshots
   all hinted nodes keyed by graph ID and node name.

2. **Post-grad early recovery** (`propagate_work_division_hints`) — At
   `CustomPrePasses` time, the compiler traces each node's `from_node`
   provenance chain back to the pre-grad graph and restores hints from the
   snapshot. The compiler then takes a second snapshot
   (`collect_pre_pass_hints`) for the next stage.

3. **Post-grad late recovery** (`propagate_post_pass_hints`) — At
   `CustomPostPasses` time, replacement nodes (e.g. `mm_default`,
   `add_tensor`) may have lost both `custom` metadata and `from_node`
   provenance. The compiler recovers hints by stripping ATen overload
   suffixes from node names and matching against the second snapshot.

The core division pass then reads the recovered hint from each IR node's
origin metadata and applies it in place of the heuristic.

#### Scope and decomposed operations

To avoid ambiguity, keep the context manager scope as narrow as possible:

```python
a = F.linear(x, w, b)          # heuristic for both mm and add
with work_division_hint([2, 1, 2]):
    c = x @ y                   # only this matmul gets the hint
```

### Disabling Hints

To bypass all hints without modifying user code, set
`SPYRE_INDUCTOR_IGNORE_HINTS=1`. All operations fall back to the automatic
planner. This is useful for A/B comparisons between hinted and automatic
splits.

### Caveats

- Different hint values for the same compiled function may hit Dynamo's graph
  cache. Call `torch._dynamo.reset()` between experiments.
- Hints are specified in raw element-space dimensions, not stick-adjusted
  counts.

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
