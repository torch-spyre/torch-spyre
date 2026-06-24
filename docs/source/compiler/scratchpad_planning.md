# Scratchpad (LX) optimization

Where LX scratchpad planning sits in torch-spyre today, and what we are
working on next.

:::{admonition} Status
:class: note

Scratchpad planning runs by default. The pass is gated by `lx_planning`,
which has defaulted to `1` since [#2459](https://github.com/torch-spyre/torch-spyre/pull/2459).
The greedy solver (`config.layout_solver = "greedy"`) is the default.
First-fit and best-fit are available as opt-ins.

Co-optimization with work distribution is opt-in.
`config.co_optimizing_lx_planning` (`CO_OPTIMIZING_LX_PLANNING=1`)
defaults to off. The proof-of-concept handles only single-output-dim
flips today.
:::

**Quick navigation:**

- [Hardware context](#hardware-context)
- [Why scratchpad planning matters](#why-scratchpad-planning-matters)
- [Assumptions](#assumptions)
- [Pipeline position](#pipeline-position)
- [Optimizations on softmax](#optimizations-on-softmax)
- [Implementation](#implementation)
- [Solvers](#solvers)
- [Co-optimization with work-distribution](#co-optimization-with-work-distribution)
- [Current limitations](#current-limitations)
- [Target patterns](#target-patterns)
- [Future work](#future-work)

## Hardware context

Each Spyre core has a 2 MB on-core scratchpad (LX) alongside shared HBM.
LX reads are much cheaper than HBM and have no cross-core contention, so
the planner aims to keep reused tensors on-core and let HBM traffic happen
only at the graph boundary.

:::{figure} ../_static/images/lx/memory-hierarchy.svg
:alt: Spyre memory hierarchy. Large slow HBM shared by 32 cores, each with a 2 MB LX scratchpad.
:width: 480px
:align: center

HBM is plentiful but slow and shared. LX is small but fast and core-local.
The compiler picks which buffers live where.
:::

| Parameter | Value | Config |
|---|---|---|
| Total LX per core | 2 MB | fixed |
| Backend-reserved fraction | 20% | `DXP_LX_FRAC_AVAIL` |
| Usable LX per core | ~1.6 MB | `int((2<<20) * (1 - frac_avail))` |
| Alignment | 128-byte (stick) | implicit |
| Cores | 1 to 32 | `SENCORES` |
| Per-core HBM span limit | 256 MB | hardware, separate from LX |
| Inter-core data ring | yes | not yet used by compiler |
| Inter-core reduce-sum ring | yes | not yet used by compiler |

## Why scratchpad planning matters

Spyre is often memory-bound: compute cores stall waiting on HBM. Every byte
the compiler can keep on LX between producer and consumer is a byte the
runtime never has to fetch.

Take a single-core softmax over a `(512, 1024)` fp16 tensor, 1 MB of input.
The lowered op sequence is `max → sub → exp → sum → div`. Total HBM traffic
depends on which intermediates land on LX:

| Stage | What changes | HBM read+write | Speedup vs baseline |
|---|---|---|---|
| 1. baseline (HBM only) | every intermediate goes through HBM | 8MN + 4N | 1.0x |
| 2. pin reduction outputs to LX | `max` and `sum` outputs stay on-core | 8MN | ~1.0x (reductions are tiny) |
| 3. + in-place ops on LX | `exp`, `sub` reuse their input's address | 3MN | ~2.7x |
| 4. + clone the input to LX | one pass over HBM, everything else stays | 2MN | ~4.0x |

The ideal memory time after stage 4 is roughly 25% of baseline. End-to-end
measurements on this softmax kernel show the median runtime drop from
32.5 µs to 23.7 µs, a 27% reduction. The gap between the ideal and the
measured result is fixed per-bundle overhead.

The four stages map onto code under `torch_spyre/_inductor/scratchpad/`:
LX-eligible op outputs (stage 2), in-place reuse (stage 3), and
`CloneInputNodesPass` (stage 4).

## Assumptions

### LX state survives kernel boundaries

The planner assumes LX state persists across SuperDSC bundle boundaries. It
operates on the flat operations list before fusion and has no awareness of
where bundle boundaries will fall, so allocation decisions can span
multiple bundles.

There is a correctness gap under VF multi-tenancy: the runtime may wipe LX
on context switch at any bundle boundary. Once SpyreCode with symbolic
addresses is available, fusion will not be limited by the number of
tensors used by the bundle, and bundle boundaries should only land at
FallbackKernels, which are visible to the planner.

### Working sets are already right-sized

Tile size selection (BLOCK_M, BLOCK_N, BLOCK_K, etc.) to fit operands
within ~1.6 MB is a pre-Inductor concern, the same class of problem GPU
autotuners solve. Spad opt begins after tiling. Given operations whose
working sets are feasible, the planner decides which buffers to pin to
LX, at what addresses, and for how long. Tiling determines whether data
*can* fit; spad opt determines whether it *does* fit.

### No eviction from LX

Buffers placed on LX stay until end-of-life. There is no mechanism to
move a buffer to HBM and reload it later. This is deliberate. Eviction
only wins when a buffer is read many times on LX, goes dormant, then is
read many times again, which is rare in practice. Pre-Inductor tiling
already keeps per-op working sets small. The remaining problem (which
buffers to keep on LX when accumulated live buffers exceed capacity) is
better solved by smarter placement and spill decisions at allocation
time than by runtime eviction with its graph mutation complexity and
extra HBM round-trips.

## Pipeline position

Scratchpad planning runs at the end of `CustomPreSchedulingPasses`,
after work division has stamped per-op core splits:

```
deadcode_elimination
propagate_spyre_tensor_layouts        # assign FixedTiledLayout
optimize_restickify_locations
finalize_layouts
insert_restickify
insert_bmm_padding
dedup_and_promote_constants
chunk_large_tensors                   # conditional on config.chunk_large_tensors
propagate_named_dims                  # named-dimension metadata
assign_dim_hints
coarse_tile                           # runs when hints produce groups
span_reduction                        # work-division: enforce 256 MB span
cost_model_matmul_division            # work-division: matmul cost model
work_distribution                     # work-division: default distributor
scratchpad_planning                   # ← THIS PASS, gated by config.lx_planning
```

Two ordering constraints fix this slot:

- **Work division must run first.** Scratchpad planning needs
  `op_it_space_splits` to compute per-core buffer sizes. Work division
  also decides whether adjacent ops have compatible core splits.
  Incompatible splits trigger `core_div_mismatch` and disqualify shared
  buffers from LX (see [Current limitations](#current-limitations)).
- **Stickification must run first.** All buffers need `FixedTiledLayout`
  for device-memory size computation.

## Optimizations on softmax

The softmax example (`max → sub → exp → sum → div` over a `(512, 1024)`
input) is the easiest way to see what the planner does as each
optimization is added.

:::{figure} ../_static/images/lx/softmax-stages.svg
:alt: Four stages of LX optimization on softmax. Baseline, pin reductions, in-place, clone-input.
:width: 720px
:align: center

Each stage corresponds to one capability the planner gained. Boxes
coloured red touch HBM, green stays on LX, yellow is in-place reuse, and
blue is a clone inserted by the planner.
:::

**Stage 1, baseline.** No LX. Every op reads and writes HBM. For
`(M, N) = (512, 1024)` with a reduction along axis 0, total HBM I/O is
`8MN + 4N` bytes (eight full passes over the matrix plus four passes over
the reduction vector).

**Stage 2, pin reduction outputs to LX.** `max` and `sum` produce small
vectors (`1 × N`) that the next op reads immediately. Routing these
through LX instead of HBM costs almost no LX budget but eliminates
the `4N` term. On large `M × N` shapes this is a tiny relative win, but
it sets up the next two optimizations, which are large.

**Stage 3, in-place ops.** When a buffer is on LX and its last reader is
itself dying-after-this-op, the output of the next op can reuse the same
LX address. `exp` and `sub` are flagged
[OP_GOOD_FOR_LX_INPLACE](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/scratchpad/utils.py)
and therefore in-placeable. After stage 3, the only HBM access left is
the graph input and graph output, for `3MN` bytes total, a 62% reduction.

**Stage 4, clone the input to LX.** The graph input is read by several
ops. Without a clone each reader would re-fetch from HBM.
`CloneInputNodesPass` detects multi-use inputs that fit in LX and inserts
a `clone` op at the front of the graph. The clone reads HBM once and
writes LX; every subsequent op reads from LX. After stage 4 total HBM is
`2MN`, the input read plus the output write, which is the theoretical
minimum for this graph.

Numbers from a 1000-iteration measurement (after 200 warm-up runs):

| Variant | dim | M×N | cores | LX | clone | in-place | median (µs) |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 512×1024 | 1 | off | n/a | off | **32.51** |
| stage 2 | 0 | 512×1024 | 1 | on | n/a | off | 27.66 |
| stage 3 | 0 | 512×1024 | 1 | on | n/a | exp,sub | 23.93 |
| stage 4 | 0 | 512×1024 | 1 | on | yes | exp,sub | **23.67** |
| 4-core | 0 | 512×1024 | 4 | on | yes | exp,sub | 32.17 |

The 4-core run is *slower* on this small shape because communication and
work-distribution overhead dominate. Multi-core LX wins on larger
tensors, see below.

### Multi-core LX

A `(1024, 2048)` fp16 tensor is 4 MB, bigger than any single core's LX.
Splitting the rows over four cores gives each core a `(256, 2048)` slice
(~1 MB per core) that fits.

:::{figure} ../_static/images/lx/multicore-tiling.svg
:alt: Splitting a 4 MB tensor across four cores so each per-core slice fits in LX
:width: 580px
:align: center

For tensors larger than 2 MB, the same shape that overflows single-core
LX fits comfortably once it has been split across cores by work
distribution.
:::

Multi-core LX is not free. Adjacent ops can request different splits (one
sliced by rows, the next by columns), in which case the shared buffer is
stuck on HBM. That mismatch is what motivates co-optimization (below).

## Implementation

### Architecture

Scratchpad planning has three layers with separate concerns:

:::{figure} ../_static/images/lx/allocator-architecture.svg
:alt: ScratchpadAllocator delegates to a pluggable MemoryPlanSolver and runs optional graph passes around it
:width: 480px
:align: center

`DefaultAllocator` runs pre-passes (clone insertion), gathers
`LifetimeBoundBuffer`s, hands them to a pluggable solver, then writes the
chosen LX addresses onto buffer layouts. `StrategyBCoOptimizingAllocator`
extends this flow with a split-search step before the solver runs.
:::

The relevant code lives under `torch_spyre/_inductor/scratchpad/`:

| File | Responsibility |
|---|---|
| `passes.py` | `ScratchpadOptimizationPass` ABC, `CloneInputNodesPass` |
| `plan_solver.py` | `MemoryPlanSolver` ABC, `LifetimeBoundBuffer`, `GreedyLayoutSolver` |
| `firstfit_bestfit_solver.py` | `FirstFitLayoutSolver`, `BestFitLayoutSolver` |
| `allocator.py` | `ScratchpadAllocator` ABC, `DefaultAllocator`, `StrategyBCoOptimizingAllocator` |
| `utils.py` | liveness, in-place candidates, op eligibility lists |

### Entry point

```python
scratchpad_planning(graph, allocator=DefaultAllocator())
```

`DefaultAllocator` runs the following pipeline:

1. **Pre-passes.** `CloneInputNodesPass` walks graph inputs and inserts a
   `clone` for any HBM input that is read more than once *and* fits on
   LX. The clone output becomes a fresh LX-eligible buffer.
2. **Buffer analysis.** `_generate_buffers` produces a list of
   `LifetimeBoundBuffer(name, size, start_time, end_time, in_place_parents)`
   for every op that survives the eligibility filter (graph i/o is
   excluded; so are buffers whose users have incompatible core splits).
3. **Layout planning.** The solver assigns an `address` to each buffer
   it can fit; the rest get `address=None` and stay on HBM.
4. **Push allocation.** Successful placements are written to
   `layout.allocation["lx"] = addr` on each buffer's `FixedTiledLayout`.
5. **Post-passes.** Currently empty. Reserved for solver-driven graph
   mutations (output cloning, op re-ordering).

### Codegen integration

Once `layout.allocation["lx"]` is set:

- `spyre_kernel.py` removes LX-allocated buffers from kernel args
  (core-local, no HBM backing needed).
- `codegen/compute_ops.py` writes `component_` as `"lx"`, `memOrg_` as
  LX only, and `startAddressCoreCorelet_` as the baked-in LX address
  (the same address per core on their respective scratchpads).

## Solvers

`config.layout_solver` (`"greedy" | "firstfit" | "bestfit"`) picks the
solver.

### GreedyLayoutSolver (default)

Walks transition points in chronological order. At each point it
deallocates expired buffers, then for each newly-live buffer:

1. If a declared in-place parent is alive at the previous time step and
   the child fits in the parent's slot, reuse the parent's address.
2. Otherwise find a free block. Try address 0, then above the high-water
   mark, then gaps between live allocations.

It is simple, easy to reason about, and in-place reuse is automatic.
Decisions are local, though. Placing buffer A at address 0 can block a
later large buffer C that would have benefited from a low address.

### FirstFitLayoutSolver and BestFitLayoutSolver

Both solvers see *all* buffers up front, sort them topologically with
ties broken by ascending lifetime, and place them shortest-life-first
into the free address space.

For each buffer, free gaps during its lifetime are computed by
subtracting the address intervals of every overlapping placed buffer.
In-place parent addresses are kept as candidate gaps so the child can
land on top of them.

The two solvers differ only in the gap-selection policy:

- `FirstFitLayoutSolver` picks the first gap large enough.
- `BestFitLayoutSolver` picks the gap that leaves the smallest remainder
  after placement.

Both naturally avoid the "buffer at address 0 blocks everything else"
failure mode of the greedy solver. They are not yet selected by default.
Once a deeptools dependency clears, first-fit is the expected default.

## Co-optimization with work-distribution

Work division optimizes each op independently for parallelism. Adjacent
ops sharing a buffer can get different splits (different shapes mean
different optimal decompositions), which triggers `core_div_mismatch`
and disqualifies the shared buffer from LX even when it would have fit.

`StrategyBCoOptimizingAllocator` (gated by
`config.co_optimizing_lx_planning`, env var `CO_OPTIMIZING_LX_PLANNING=1`)
treats split choices and LX placement jointly:

:::{figure} ../_static/images/lx/co-optimization.svg
:alt: Co-optimization searches over alternative split assignments, scoring each by HBM bytes left unpinned
:width: 700px
:align: center

The co-optimizer enumerates split variants per op, scores each
combination by counting HBM bytes the solver could not pin, and commits
the winning assignment back before the standard allocator flow.
:::

The current POC (`v1`) only considers dim-flipping. When an op's seed
split has a single output-dim split factor, it generates variants that
move that factor onto each compatible alternative output dim.
Reduction-axis splits and multi-dim splits are skipped for now. The
search is bounded by `DEFAULT_VARIANT_CAP = 6` per op and uses DFS over
the cross-product with no early-stop pruning, so the search stays
compatible with future non-greedy solvers that can reach interior states.

The leaf-scoring function is intentionally cheap and solver-agnostic. It
runs the full `_generate_buffers + plan_layout` pass on the candidate
splits and counts the HBM bytes of every buffer the solver could not pin.

## Current limitations

### Greedy single-pass, no lookahead (default solver)

The greedy solver processes ops in topological order making irrevocable
placement decisions without considering future ops. First-fit and
best-fit mitigate this by sorting all buffers up front before placing.

### No defragmentation

`find_free_block` can locate holes between allocations but cannot
compact the address space. Allocate/deallocate cycles fragment LX.

### Co-optimization is a POC

`StrategyBCoOptimizingAllocator` implements the joint
work-division + LX planning idea, but the current variant generator only
flips a single output-dim split. Productionisation needs richer variant
generation (multi-dim splits, fewer-cores, reduction-axis), a
performance model that balances compute throughput against memory
traffic so we do not trade away compute parallelism for trivial LX wins,
and coverage when the `coarse_tiling` pass also drives split decisions.

### No cross-core ring utilization

The hardware has a data ring (core-to-core LX reads/writes) and a
reduce-sum ring (cross-core sum reduction, useful for matmul K-splits).
The compiler does not yet generate code that uses either ring. The
`core_div_mismatch` hard wall exists because without ring transfers, a
buffer split N ways in one op cannot be read by M cores in the next
(with M ≠ N). Ring support could remove this wall by redistributing
data across cores without going through HBM (the ring is always faster
than HBM). Enabling it requires compiler and codegen support to emit
ring transfer instructions in the SuperDSC schedule.

## Target patterns

The test suite `test_scratchpad_patterns.py` encodes patterns the greedy
allocator cannot handle (`@expectedFailure`). Each documents a class of
problem to be solved:

| Pattern | Problem | What's needed |
|---|---|---|
| Simple fragmentation | Greedy places A at addr 0, blocking later large allocation C | Placement aware of future deallocations |
| Staircase (up/down) | Increasing or decreasing buffer sizes overflow LX under greedy append | Lookahead and placement-order optimization |
| GQ attention | Large/small buffer lifecycle alternation (Q_K, scores vs. max, denominators) | Size-aware packing exploiting lifecycle patterns |
| MoE MLP | Many buffers of varying sizes and lifetimes, shared hidden state | Stack-like placement with complex lifetime management |

Best-fit and first-fit pass several patterns the greedy solver fails.
The remaining `@expectedFailure` cases motivate the items in
[Future work](#future-work).

## Future work

The items below are not in-tree. They sit on top of the
`MemoryPlanSolver` and `ScratchpadOptimizationPass` interfaces so they
can be plugged in without disturbing the rest of the planner.

### Non-greedy solvers

Two non-greedy solver families are being prototyped on top of the same
`MemoryPlanSolver` interface:

- **Simulated Annealing** (Imanishi-Xu) uses a first-fit or best-fit
  allocation as the initial guess, then perturbs the order to escape
  local minima.
- **Integer Linear Programming** via OR-Tools formulates placement as a
  2D bin-packing constraint and lets a general-purpose solver search
  exhaustively for graphs small enough to be tractable.

### Richer co-optimization

The current dim-flipping variant generator is a starting point. Planned
extensions:

- multi-dim splits, fewer-cores variants, reduction-axis splits;
- a performance model that balances compute throughput against memory
  traffic so we do not trade away parallelism for trivial LX wins;
- joint operation with the `coarse_tiling` pass when that pass also
  drives split decisions.

### Solver-driven graph mutations

`ScratchpadOptimizationPass` plug-ins run before or after the solver.
Candidates under evaluation:

- **Buffer evictions.** Move a buffer from LX to HBM and bring it back
  later. This is the counterpart of the "no eviction" assumption above
  and is only worthwhile when liveness shows it pays off.
- **Operation re-ordering.** Re-order independent ops to extend or
  shorten lifetimes for better packing.
- **Output node cloning.** Promote a producer to LX and clone to HBM
  only when an HBM-resident copy is required (draft PR
  [#2028](https://github.com/torch-spyre/torch-spyre/pull/2028)).
- **Driving cloning from the solver.** `CloneInputNodesPass` currently
  runs as a pre-pass with a heuristic. The longer-term plan is for the
  solver to decide which clones pay off based on the global layout.

### Cross-core ring transfers

Remove the `core_div_mismatch` hard wall by emitting data-ring or
reduce-sum-ring transfers in the SuperDSC schedule, so a buffer split N
ways in one op can feed a different M-way split in the next without
going through HBM. Requires compiler and codegen support.

### Non-terminal kernel hints

Extend the runtime to support a non-terminal kernel annotation. A
bundle marked non-terminal guarantees no context switch before the next
bundle, preserving LX state across the boundary. The compiler emits the
annotation based on cross-bundle LX liveness.

This buys real time on tightly coupled op sequences (for example,
softmax decomposed across bundles due to the 6-tensor limit). It needs
runtime scheduler support and compiler liveness tracking across bundle
boundaries.

## Testing

Three suites cover the planner:

- `tests/inductor/test_scratchpad_solver.py`: solver-level unit tests.
  Buffers are constructed directly as `LifetimeBoundBuffer` lists and
  fed to each solver.
- `tests/inductor/test_scratchpad_use.py`: end-to-end op-level checks
  that LX is actually used for representative graphs.
- `tests/inductor/test_scratchpad_patterns.py`: the `@expectedFailure`
  patterns above. Promoting one to passing is the typical signal that a
  new solver or pass is doing useful work.
- `tests/inductor/test_inductor_ops_lx_planning.py`: runs the full
  Inductor op suite under `LX_PLANNING=1` to catch regressions.

An auto-generated coverage suite expands op coverage beyond the
hand-written patterns above. It composes each supported op with simple
reduction or pointwise tails, so every supported op is exercised on
the planner without a hand-written test. The suite catches planning
bugs that the hand-written cases miss.

## Related documents

- [`work_division_planning.md`](work_division_planning.md) describes how
  work distribution decides per-op core splits that scratchpad planning
  then consumes.
- [`coarse_tiling_loops.md`](coarse_tiling_loops.md) describes coarse
  tiling, which reduces working sets so adjacent ops can fit on LX in
  the first place.
