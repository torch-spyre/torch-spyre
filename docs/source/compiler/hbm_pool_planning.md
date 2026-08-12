# HBM Pool Planning

HBM pool planning is the compiler stage that allocates intermediate tensors
into a pool of bulk HBM memory, allowing non-overlapping tensors to reuse the
same storage region and reducing peak HBM pressure. This page describes what
the pool is, how it is scoped, how it interacts with LX scratchpad planning,
and the cross-bundle exclusion rule that prevents intermediates from being
pool-allocated across kernel invocation boundaries.

## What the HBM pool is

The HBM pool (also called the *intermediates segment*) is a contiguous region
of bulk HBM memory (not on-core SRAM) that houses intermediate tensors whose
data does not need to persist beyond the kernel invocation in which they are
produced. This is distinct from the LX scratchpad, which is a small, on-core
SRAM dedicated to holding reused operands within a single Spyre core.

:::{admonition} Terminology clarification
:class: warning

- **LX scratchpad:** 2 MB of on-core SRAM per core; managed by
  [scratchpad planning](scratchpad_planning.md); tiny but fast; core-local.
- **HBM pool:** bulk off-chip HBM memory; managed by this pass; plentiful but
  slower; globally accessible. The pool exists to allow safe temporary
  storage for intermediate computation results that will be consumed by the
  next operation and then discarded.

The two passes are independent tiers of the memory hierarchy. A single buffer
is placed in *either* LX *or* the HBM pool, never both.

:::

## Per-bundle scoping

HBM pool planning runs in `CustomPostFusionPasses`, after `spyre_fuse_nodes`
has already determined the final set of SDSC bundles (kernel invocations).
Each bundle is a separate `SpyreKernel` that compiles to a single `.run()`
call in the generated wrapper code.

The fundamental constraint is:

> A buffer's pool allocation is scoped to a single bundle. The pool tensor
> is allocated immediately before that bundle's `.run()` call and freed
> immediately after. Separate bundles do not share a pool.

This means:

- A buffer written and read entirely within one bundle (its producer and all
  its consumers in the same bundle) is eligible for pool allocation within
  that bundle's dedicated pool.
- A buffer written in bundle A and read by bundle B (a *cross-bundle buffer*)
  cannot be pool-allocated, because the two bundles execute as separate,
  sequential kernel invocations and no pool-scoped storage can persist
  between them. Such buffers fall back to standalone HBM allocation, the same
  as any other non-pool-eligible intermediate today.
- A buffer written by more than one bundle -- for example, a loop-carried
  accumulator whose in-place update in a later bundle is renamed by
  Inductor's scheduler (`mutation_renames`) to the same buffer name as its
  initializer in an earlier bundle -- is never pool-eligible in any bundle,
  regardless of where it is read. This exclusion is unconditional and
  independent of the read-based cross-bundle check above.

This design enables bundle-local pool lifetimes, which reduces peak HBM
pressure: instead of allocating pools for all bundles up front and keeping
them resident for the entire graph execution, each bundle's pool is freed as
soon as that bundle's computation is complete.

## Comparison with LX scratchpad planning

Both HBM pool planning and LX scratchpad planning are allocation passes that
decide where intermediate tensors live in the memory hierarchy. Here are the
key differences:

| Aspect | HBM Pool Planning | LX Scratchpad Planning |
|--------|-------------------|----------------------|
| **Memory tier** | Bulk HBM | 2 MB on-core SRAM (fast) |
| **Pipeline stage** | `CustomPostFusionPasses` (after fusion, after bundles exist) | `CustomPreSchedulingPasses` (before scheduler, before bundles exist) |
| **Allocation strategy** | Bump-allocator with free-list reuse; same offset reused by non-overlapping buffers | Core-local address assignment; each buffer gets a fixed on-core address |
| **Scope of sharing** | Per-bundle (a pool is local to one kernel invocation) | Per-core (an LX address persists across multiple ops executed on the same core) |
| **Exclusion rules** | Buffers with cross-bundle live ranges are excluded | Buffers read by CPU-fallback nodes are excluded |
| **Gating** | Enabled by `config.hbm_pool_planning` (default: on) | Enabled by `LX_PLANNING=1` (default: on) |

LX planning runs first (at `CustomPreSchedulingPasses`), so it sees a flat
operations list before bundle boundaries are known. Pool planning runs later
(at `CustomPostFusionPasses`) and sees the final, fused, bundle-organized
list. A buffer claimed by LX planning is excluded from pool allocation; the
two passes are mutually exclusive per buffer, applied in that order.

## Pipeline position

HBM pool planning runs in `CustomPostFusionPasses`, after both work-division
and fusion have already run:

```
CustomPreSchedulingPasses:
  work_division
  scratchpad_planning                  # LX allocation (phase 1)

... Inductor scheduler construction ...

CustomPostFusionPasses:
  demote_incoherent_lx_buffers         # LX fixups (phase 2)
  spyre_fuse_nodes                     # Determine bundle boundaries
  hbm_pool_planning                    # Per-bundle HBM pool allocation (this pass)
```

By the time this pass runs, `nodes` is the final, post-fusion top-level list.
Each entry in `nodes` is exactly one bundle:

- A `FusedSchedulerNode` wrapping multiple fused operations.
- A `CountedLoopSchedulerNode` (which is a `FusedSchedulerNode` subclass)
  wrapping a loop and its body operations.
- A standalone `SchedulerNode` or other node type that fusion left untouched
  because it had no fusible neighbors.

## Algorithm: per-bundle live-range analysis and allocation

For each top-level entry (bundle) in the post-fusion node list:

1. Flatten the bundle's tree to a list of leaf operations (to handle
   nested `FusedSchedulerNode` and `CountedLoopSchedulerNode` hierarchies).
2. Identify candidates: buffers that are:
   - Written and read (not only written or only read).
   - Not graph inputs or outputs.
   - Not already claimed by LX planning.
   - Not read by CPU-fallback nodes.
   - Fully contained within the bundle (same bundle writes and reads them).
   - Written by exactly one bundle. A buffer written by two or more bundles
     (e.g. an in-place accumulator update that Inductor's `mutation_renames`
     maps to the same buffer name as an earlier initializing write) is
     excluded unconditionally, even if all its reads happen to lie within
     the last-writing bundle.
3. For each bundle's local candidate set, compute live ranges: a buffer's
   live range is the interval from its producer op to its last consumer op
   within that bundle.
4. Sort by producer order (start step), then by end step and name for
   determinism.
5. Allocate sequentially with a bump-allocator and free-list reuse:
   - Process buffers in order.
   - Before allocating a new buffer, free any previously-allocated blocks
     whose live range ended before this buffer's start step.
   - Assign each buffer an offset within the bundle's dedicated pool.
6. Record the pool's final extent (total bytes needed) in
   `V.graph.hbm_pool_sizes[bundle_name]`.

Only bundles with at least one pool-eligible buffer get an entry in
`V.graph.hbm_pool_sizes`; bundles with no pool candidates get no pool tensor
and no allocation overhead.

## Integration with code generation

During code generation, the scheduler looks up each bundle's pool size from
`V.graph.hbm_pool_sizes` and passes it to `SpyreKernel`'s constructor. The
kernel then emits pool allocation and deallocation code around its own
`.run()` call:

```python
_pool_{bundle_name} = spyre_empty_with_layout((pool_size_bytes,), (1,), torch.uint8, ...)
sdsc_fused__{bundle_name}.run(_pool_{bundle_name}, ...)
del _pool_{bundle_name}
```

This per-bundle scoping ensures that the pool tensor's lifetime is limited to
the bundle's own kernel invocation, minimizing peak HBM usage across the
entire graph execution.

## Configuration and limitations

HBM pool planning is controlled by `config.hbm_pool_planning`, which defaults
to on. Set `HBM_POOL_PLANNING=0` to disable it globally; in that mode all
intermediates fall back to standalone HBM.

**Interaction with symbolic arguments:**

In `bundle_symbolic_args=False` mode (not the default -- the default is
`True`), `spyre_fuse_nodes` is
disabled and every scheduler node becomes its own single-op bundle. In this
degenerate case, any buffer read by a later node is cross-bundle by
construction and falls back to standalone HBM. Pool allocation is effectively
disabled, which is correct: the benefit of pooling (sharing HBM across
non-overlapping bundle-local temporaries) does not apply when there are no
multi-op bundles to localize to. No special handling is needed in the
planner; the cross-bundle exclusion logic automatically handles this edge
case.

## See Also

- [Scratchpad Planning](scratchpad_planning.md) covers LX allocation and the
  interaction between the two memory tiers.
- [Tensor Layouts](../user_guide/tensors_and_layouts.md) covers device
  layouts and memory hierarchy concepts.
- [Spyre Accelerator](../architecture/spyre_accelerator.md) gives the full
  hardware overview.
