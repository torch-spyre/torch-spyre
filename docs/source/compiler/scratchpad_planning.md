# Scratchpad (LX) Optimization — Design Document

Current state of LX scratchpad memory planning in torch-spyre and the
algorithmic space being explored for improvement.

## 1. Hardware Context

Each Spyre core has a **2 MB on-core scratchpad** ("LX") alongside shared HBM.
LX reads are ~10x cheaper than HBM with no cross-core contention. The goal is
to minimize HBM traffic by keeping reused tensors on-core.

| Parameter | Value | Config |
|---|---|---|
| Total LX per core | 2 MB | fixed |
| Backend-reserved fraction | 20% | `DXP_LX_FRAC_AVAIL` |
| Usable LX per core | ~1.6 MB | `int((2<<20) * (1 - frac_avail))` |
| Alignment | 128-byte (stick) | implicit |
| Cores | 1–32 | `SENCORES` |
| Per-core HBM span limit | 256 MB | hardware, separate from LX |
| Inter-core data ring | yes | not yet used by compiler |
| Inter-core reduce-sum ring | yes | not yet used by compiler |

## 2. Assumptions

### LX State Survives Kernel Boundaries

The current implementation assumes LX state persists across SuperDSC bundle
boundaries — the planner operates on the flat operations list *before* fusion
with no awareness of where bundle boundaries will fall, and makes allocation
decisions that can span multiple bundles.

**Correctness gap under VF multi-tenancy:** the runtime may wipe LX on
context switch at any bundle boundary.  However, once SpyreCode with
symbolic addresses is available, fusion will not be limited by the number
of tensors used by the bundle.  Therefore we expect that the bundle boundaries
will only occur at FallbackKernels which are visible to the planner.

### Working Sets Are Already Right-Sized

Tile size selection (BLOCK_M, BLOCK_N, BLOCK_K, etc.) to fit operands within
~1.6 MB is a **pre-Inductor** concern — the same class of problem GPU
autotuners solve. Spad opt begins *after* tiling: given operations whose
working sets are feasible, decide which buffers to pin to LX, at what
addresses, and for how long. Tiling determines whether data *can* fit; spad
opt determines whether it *does* fit.

### No Eviction from LX

Buffers placed on LX stay until end-of-life; there is no mechanism to evict
a buffer to HBM and reload it later. This is deliberate: eviction only wins
when a buffer is read many times on LX, goes dormant, then is read many times
again — rare in practice. Pre-Inductor tiling ensures per-op working sets
fit; the remaining problem (which buffers to keep on LX when accumulated live
buffers exceed capacity) is better solved by smarter placement and spill
decisions at allocation time (§7.1, §7.2) than by runtime eviction with its
graph mutation complexity and extra HBM round-trips.

## 3. Pipeline Position

Scratchpad planning runs as the **last pass** in `CustomPreSchedulingPasses`:

```
1. deadcode_elimination
2. propagate_spyre_tensor_layouts       # assign FixedTiledLayout
3. optimize_restickify_locations
4. finalize_layouts
5. insert_restickify
6. span_reduction                       # work division pass 1 (mandatory)
7. work_distribution                    # work division pass 2 (optional)
8. if config.lx_planning:
       scratchpad_planning              # ← THIS PASS
```

**Work division must run first**: spad opt needs `op_it_space_splits` to
compute per-core buffer sizes. Work division also determines whether adjacent
ops have compatible core splits — incompatible splits trigger
`core_div_mismatch`, disqualifying shared buffers from LX (see §5.4).

**Stickification must run first**: all buffers need `FixedTiledLayout` for
device-memory size computation.

Gated on `LX_PLANNING=1`. Disabled by default; experimental.

## 4. Current Implementation

### 4.1 Entry Point

```python
scratchpad_planning(operations, strategy=GreedyAllocationStrategy())
```

Operates on the topological operations list from `GraphLowering`.

### 4.2 GreedyAllocationStrategy

#### Phase 1: Buffer Analysis (`buf_analysis`)

Scans all operations to build:

- **`bufs_to_dealloc_at_idx`**: `{op_index: [buf_names]}` — deallocation
  schedule (one past last use)
- **`buf_users`**: `{buf_name: [Operation]}` — all readers of each buffer
- **`core_div_mismatch`**: `{buf_name: bool}` — True if any two operations
  on this buffer have different `op_it_space_splits`

#### Phase 2: Optional Clone Insertion

Disabled. When enabled, inserts clone ops for multi-use graph inputs that fit
in LX, allowing the clone output to be pinned on-core.

#### Phase 3: Greedy Forward Pass

Iterates operations in topological order. At each step, deallocates dead
buffers, then calls `try_allocate()` for each buffer used by the current op.

**Skip conditions** (never placed on LX):
- Graph input (not produced by a ComputedBuffer; clone insertion is the
  workaround, currently disabled), graph output (must reach HBM), or
  `core_div_mismatch` is True

**Allocation rules** (for each non-skipped buffer):

| Case | Condition | Action |
|---|---|---|
| Reuse input | Input already on LX, size matches | Reuse existing address |
| Inplace output | Output, dying input on LX with matching size/layout | Reuse input's address |
| New output | Output, no reusable input | Allocate new LX block |

### 4.3 ScratchPadAllocator

Flat allocator over the ~1.6 MB usable address space:

- **State**: `{tensor_name: {"addr": int, "size": int}}`
- **`find_free_block`**: tries address 0, then above high-water mark, then
  gaps between existing allocations. No defragmentation.
- **Deallocation**: removes entry from `usage` dict.

### 4.4 Codegen Integration

`ScratchPadAllocator.allocate()` writes `layout.allocation["lx"] = addr` on
the buffer's `FixedTiledLayout`. Downstream:

- **`spyre_kernel.py`**: LX-allocated buffers are removed from kernel args
  (core-local, no HBM backing needed)
- **`codegen/compute_ops.py`**: `component_` → `"lx"`, `memOrg_` → LX only,
  `startAddressCoreCorelet_` uses the baked-in LX address (same address per
  core on their respective scratchpads)

## 5. Current Limitations

### 5.1 Greedy Single-Pass, No Lookahead

The allocator processes ops in topological order making irrevocable placement
decisions without considering future ops.

### 5.2 No Defragmentation

`find_free_block` can locate holes between allocations but cannot compact the
address space. Allocate/deallocate cycles fragment LX.

### 5.3 No Joint Work-Division / Scratchpad Optimization

Work division optimizes each op independently for parallelism. Adjacent ops
sharing a buffer can get different splits (different shapes → different
optimal decompositions), triggering `core_div_mismatch` and disqualifying the
shared buffer. Joint optimization could choose compatible splits across a
sequence of ops, trading per-op parallelism for cross-op scratchpad
eligibility.

### 5.4 No Cross-Core Ring Utilization

The hardware has a **data ring** (core-to-core LX reads/writes) and a
**reduce-sum ring** (cross-core sum reduction, useful for matmul K-splits).
The compiler does not generate code that uses either ring. The
`core_div_mismatch` hard wall exists because without ring transfers, a buffer
split N ways in one op cannot be read by M cores in the next (M ≠ N). Ring
support could eliminate this wall by redistributing data across cores without
going through HBM (ring is always faster than HBM). Enabling this requires
compiler and codegen support to emit ring transfer instructions in the
SuperDSC schedule.

## 6. Target Patterns

The test suite `test_scratchpad_patterns.py` encodes patterns the greedy
allocator cannot handle (`@expectedFailure`). Each documents a class of
problem to be solved:

| Pattern | Problem | What's Needed |
|---|---|---|
| Simple fragmentation | Greedy places A at addr 0, blocking later large allocation C | Placement-aware-of-future-deallocations |
| Staircase (up/down) | Increasing/decreasing buffer sizes overflow LX under greedy append | Lookahead + placement-order optimization |
| GQ attention | Large/small buffer lifecycle alternation (Q_K, scores vs. max, denominators) | Size-aware packing exploiting lifecycle patterns |
| MoE MLP | Many buffers of varying sizes/lifetimes, shared hidden state | Stack-like placement with complex lifetime management |

## 7. Strategy Space

### 7.1 Improved Greedy (Short-Term)

Keep single-pass framework, improve placement: best-fit (vs. first-fit),
end-aligned allocation for short-lived buffers.

Minimal code change; fundamental limitation remains (no global view).

### 7.2 Global Lifetime-Based (Medium-Term)

Model as register allocation: compute liveness intervals, build interference
graph, assign addresses via graph coloring or linear scan. "Spilling" = leave
in HBM.

Handles fragmentation naturally. No eviction (buffer is all-LX or all-HBM
for its lifetime). NP-hard in general but practical at DL graph sizes.

This problem formulation is amenable to general numeric solvers. OpenTeams
is prototyping a solver-based approach with a clean separation that enables
multiple solver libraries to be plugged in.  Two solvers are being used thus far: a pure
Python solver and one that leverages an existing open source C/C++ solver library.

### 7.3 Joint Work-Division + Scratchpad (Long-Term)

Co-optimize core splits and scratchpad allocation. Find splits that are
compatible (or ring-transferable) across op groups sharing buffers, trading
per-op parallelism for scratchpad eligibility.

Requires a performance model balancing compute throughput and memory traffic.

### 7.4 Non-Terminal Kernel Hints (Long-Term)

Extend the runtime to support a **non-terminal kernel** annotation: a bundle
marked non-terminal guarantees no context switch before the next bundle,
preserving LX state across the boundary. The compiler emits the annotation
based on cross-bundle LX liveness.

Significant benefit for tightly coupled op sequences (e.g., softmax
decomposed across bundles due to the 6-tensor limit). Requires runtime
scheduler support and compiler liveness tracking across bundle boundaries.
