# HBM intermediates-pool planning

How torch-spyre packs intermediate tensors into a shared region of device
memory, and how that pass relates to LX scratchpad planning.

:::{admonition} Status
:class: note

HBM pool planning runs by default. The pass is gated by
`config.hbm_pool_planning`, which defaults to `True` and reads from the
`HBM_POOL_PLANNING` environment variable. It is implemented in
`torch_spyre/_inductor/hbm_pool_planning.py` and runs in
`CustomPostFusionPasses`.
:::

**Quick navigation:**

- [What the pass does](#what-the-pass-does)
- [Relationship to LX scratchpad planning](#relationship-to-lx-scratchpad-planning)
- [Pipeline position](#pipeline-position)
- [Pool candidates](#pool-candidates)
- [Allocation](#allocation)
- [Codegen integration](#codegen-integration)
- [Related documents](#related-documents)

## What the pass does

An inference graph produces many intermediate tensors that are written by
one operation and read by a later one. If every intermediate received its
own device-memory allocation, peak memory would scale with the number of
intermediates in the graph rather than with the number that are live at
the same time.

HBM pool planning assigns intermediates to offsets inside a single shared
region of device memory, the *intermediates segment*
(`constants.INTERMEDIATES_SEGMENT`, size `constants.SEGMENT_SIZE` = 16 GB).
Two intermediates whose live ranges do not overlap reuse the same offset,
so the segment only has to be as large as the peak concurrent footprint,
not the sum of all intermediates. The pass raises a `RuntimeError` if that
peak exceeds the segment size.

Only device memory is planned here. The 2 MB per-core LX scratchpad is
planned by a separate pass; see below.

## Relationship to LX scratchpad planning

Both passes decide where an intermediate tensor's data lives, but they
operate on different memory and at different points in the pipeline. A
buffer is a pool candidate only if LX planning did *not* already claim it
(`"lx" not in layout.allocation`), so the two passes are mutually
exclusive per buffer and are applied in that order.

| | LX scratchpad planning | HBM intermediates-pool planning |
|---|---|---|
| Memory | 2 MB on-core SRAM scratchpad, per core | Regular device memory (LPDDR5), the 16 GB intermediates segment |
| Pipeline stage | `CustomPreSchedulingPasses`, before the `Scheduler` is constructed | `CustomPostFusionPasses`, after Inductor's fusion pass has run |
| Gating | `config.lx_planning` (`LX_PLANNING`) | `config.hbm_pool_planning` (`HBM_POOL_PLANNING`) |
| Goal | Keep reused tensors on fast core-local memory to cut device-memory traffic | Share one device-memory region across non-overlapping intermediates to bound peak footprint |
| Allocation | Fixed per-core SRAM address | Bump/free-list offset in the intermediates segment |
| Entry point | `scratchpad_planning()` in `scratchpad/allocator.py` | `hbm_pool_planning()` in `hbm_pool_planning.py` |

The two passes are complementary. LX planning claims the buffers that fit
on the scratchpad and benefit from staying there; HBM pool planning then
packs every remaining intermediate into the shared segment.

## Pipeline position

`hbm_pool_planning` runs inside `CustomPostFusionPasses`, which executes
over the graph of LoopLevelIR nodes immediately after Inductor's fusion
pass. The pass order is:

```
demote_incoherent_lx_buffers    # re-check LX core->slice coherence with final loop orders
hbm_pool_planning               # ← THIS PASS, gated by config.hbm_pool_planning
spyre_fuse_nodes
```

`demote_incoherent_lx_buffers` runs first for a reason: it re-checks LX
core-to-slice coherence now that loop orders are final, and any buffer it
demotes off LX must still be visible to `hbm_pool_planning` as an
unclaimed intermediate.

## Pool candidates

The pass collects candidates from two sources:

- **Kernel intermediates.** Buffers both written and read within the
  graph, detected from the written and read dependency sets on
  `ComputedBuffer` nodes.
- **`SpyreEmptyFallback` full buffers** created by coarse tiling for
  non-outputs. These are `ExternKernel` nodes that emit no dependency-
  tracked write, so they are collected explicitly by the underlying
  buffer name.

A candidate must carry a `FixedTiledLayout`, must not have been claimed by
LX planning, and must not alias a graph input or output. Graph inputs and
outputs are excluded because they are addressed by the caller, not by the
pool. Buffers read by fallback, extern, or nop kernels are also excluded,
because those consumers require Python-side tensors rather than a pooled
offset.

Because mutation buffers share the same `layout.allocation` dictionary
object as their target, input/output exclusion is checked by the identity
of the allocation object (`id(layout.allocation)`), not by name.

## Allocation

Each candidate's live range is `(start_step, end_step)`, where the start
is the timestep of the node that writes the buffer and the end is the last
timestep at which any node reads it. Candidates are sorted by start step,
tie-broken by `(end_step, name)` for determinism, and processed in that
order.

The `Allocator` tracks free blocks within the segment as `(offset, size)`
pairs in bytes. For each buffer it first frees any block whose live range
has ended, then allocates:

1. Reuse the first free block large enough, returning any leftover
   fragment to the free list.
2. Otherwise extend the pool by appending at the current pool end.

Sizes are the buffer's stick-aligned device footprint: the product of the
device-side dimensions above the stick dimension, times 128 bytes per
stick, rounded up to a 128-byte (one-stick) boundary. The allocator tracks
peak concurrent usage and raises if it exceeds `SEGMENT_SIZE`.

The chosen offset is written to `layout.allocation["hbm_pool"]`. With
`config.bundle_symbolic_args` the raw offset is stored (the segment base is
added later during bundling); otherwise the absolute
`INTERMEDIATES_SEGMENT + offset` is stored. The final pool extent is
recorded on `V.graph.hbm_pool_size`.

A `SpyreEmptyFallback` buffer that is pool-allocated is added to
`V.graph.removed_buffers`. Its `should_allocate()` returns `False` once
pooled, so the wrapper never emits an allocate line for it; adding the name
to `removed_buffers` also keeps base Inductor's free machinery from
emitting a `del` for a buffer it never allocated.

## Codegen integration

A pooled buffer is not allocated as an independent tensor. Its
`layout.allocation["hbm_pool"]` offset places it within the shared segment,
and the generated wrapper addresses it against that segment rather than
emitting a standalone allocation and free.

## Related documents

- [`scratchpad_planning.md`](scratchpad_planning.md) describes LX
  scratchpad planning, the complementary pass that claims buffers for the
  on-core SRAM before HBM pool planning packs the remainder.
- [`coarse_tiling_loops.md`](coarse_tiling_loops.md) describes coarse
  tiling, which creates the `SpyreEmptyFallback` full buffers that this
  pass collects as a second candidate source.
