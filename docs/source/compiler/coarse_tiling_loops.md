# Coarse-Tiling Loop IR for the Spyre Backend

## Background

Spyre's compilation pipeline runs a sequence of optimization passes over
`ir.Operation` objects in `CustomPreSchedulingPasses`, before Inductor's
`Scheduler` is constructed.  One optimization is **coarse-level
tiling**: take a sequence of operations that share an iteration space
dimension, split that dimension into K chunks (where K may be a symbolic
shape), and emit the body operations inside a counted outer loop.  This
is the key program transformation for working set reduction -- a tiling
of the computation in the time domain that enables effective scratchpad
utilization by reshaping the computation so that most tensors can be
allocated to the scratchpad.

The output of this pass needs to survive through:

1. Inductor's `Scheduler` (which wraps each `ir.Operation` in a
   `SchedulerNode`)
2. Spyre's `SuperDSCScheduling.codegen_node()` (which drives `SpyreKernel`
   to produce `OpSpec` objects)
3. Downstream SDSC compilation (which needs an explicit loop count to
   generate correct hardware instructions)

This document describes how that loop structure is represented, transported,
and consumed.  For the motivation — why the design has the shape it does and
what constraints forced each choice — see the companion RFC
[1358-CoarseTiling](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md).

**Quick navigation:**

- [Design Overview](#design-overview)
- [Small Example](#small-example)
- [Layer 1 — IR pass & `coarse_tile()` API](#layer-1--pre-scheduling-ir-pass)
  - [`reorder_unhinted_interlopers`](#reorder_unhinted_interlopers-pre-grouping-pass)
  - [Groups derivation and placement](#groups-derivation-and-placement-in-custompreschedulingpasses)
- [Layer 2 — `CountedLoopSchedulerNode`](#layer-2--countedloopschedulernode)
- [Layer 3 — `LoopSpec` & codegen](#layer-3--loopspec-and-codegen)
- [Key files](#key-files)
- [Invariants](#invariants-and-failure-modes)
- [Rejected alternatives](#rejected-design-alternatives)
- [Appendix: How IR rewiring works, and why it's sound](#appendix-how-ir-rewiring-works-and-why-its-sound)

## Design Overview

The tiling loop structure must be created early (before work division sees
the iteration space) and preserved intact through scheduling and codegen so
that the hardware executes the reduced per-iteration working set — not the
full pre-tiling range.  The design has three layers that correspond to the
three pipeline stages above.  At each layer the same concept — *these ops
are inside a counted loop* — takes the form demanded by that layer's type
system:

| Layer | Loop identity | Form |
|---|---|---|
| 1 — Pre-scheduling IR pass | `loop_info: CoarseTileInfo` on `ir.Operation` | Per-op tag |
| 2 — Scheduler | `CountedLoopSchedulerNode` | Perimeter wrapper |
| 3 — Codegen output | `LoopSpec` | Serializable tree node |

```
Pre-scheduling IR pass  (CustomPreSchedulingPasses)
  └─ stamps loop_info (CoarseTileInfo) on each ir.Operation
  └─ rewrites each op's ranges (divides the tiled dimension by K)

  ↓  Inductor Scheduler wraps each ir.Operation → SchedulerNode
  ↓  CustomPreFusionPasses fires (before Inductor's fusion pass)

Pre-fusion scheduler pass  (build_loop_scheduler_nodes)
  └─ scans list[BaseSchedulerNode] for runs sharing a loop_info.loop_group_id
  └─ wraps each run in a CountedLoopSchedulerNode(count=K, snodes=[...])
  └─ Inductor fusion runs after; CountedLoopSchedulerNode is opaque to it
  └─ spyre_fuse_nodes (CustomPostFusionPasses) also cannot cross group
     boundaries because CountedLoopSchedulerNode.can_fuse=False

  ↓  Scheduler calls SuperDSCScheduling.codegen_node()

codegen_node
  └─ receives CountedLoopSchedulerNode
  └─ drives SpyreKernel for the inner ops, collecting inner OpSpecs
  └─ wraps them in LoopSpec(count=K, body=[OpSpec, ...])
  └─ LoopSpec is serialized alongside OpSpec in codegen_kernel()
```

## Small Example

Consider two chained pointwise operations over `[1024, 4096]` tensors, where
`A=1024` names the row dimension and `B=4096` names the column dimension:

```python
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.wsr.propagate_named_dims import declare_tensor_dim, name_tensor_dims

A, B = 1024, 4096
declare_tensor_dim("A", A)
declare_tensor_dim("B", B)

a = torch.randn(A, B, dtype=torch.float16).to("spyre")
b = torch.randn(A, B, dtype=torch.float16).to("spyre")
c = torch.randn(A, B, dtype=torch.float16).to("spyre")
name_tensor_dims(a, ["A", "B"])
name_tensor_dims(b, ["A", "B"])
name_tensor_dims(c, ["A", "B"])

def f(a, b, c):
    with spyre_hint(num_tiles_per_dim={"A": 2}):     # outer loop: 2 iterations over rows
        with spyre_hint(num_tiles_per_dim={"B": 4}): # inner loop: 4 iterations over cols
            y = a + b
            z = y * c
            return z
```

Both operations are placed in a single tiling group with **K=2 in the outer
loop** (splitting the 1024 rows into 2 groups of 512) and **M=4 in the inner
loop** (splitting the 4096 columns into 4 groups of 1024).  Each inner-loop
iteration processes a 512 × 1024 tile (1/8th of the full tensor), enabling
the intermediate result `y` to remain in scratchpad across both operations
within the tile.

This example is the canonical small example tested by
`test_hint_nested_loop_with_scratchpad` in
`tests/inductor/test_coarse_tile_e2e.py`.  (`slices=` also works — it is a
deprecated alias for `num_tiles_per_dim=`.)

Every IR/OpSpec/`bundle.mlir` snippet below is real, captured output — not
hand-derived. When compiler internals drift and these snippets go stale,
regenerate them with `docs/tools/capture_coarse_tile_ir.py` rather than
hand-editing; see `docs/tools/README.md` for usage.

### What the coarse-tiling pass stamps

`coarse_tile()` sees this as a nested group spec and stamps a single
`loop_info: CoarseTileInfo` attribute on **both** `ir.Operation` objects. This
is the real, captured value of `buf0`'s (`y = a + b`'s) `loop_info`
immediately after `coarse_tile()` runs, before any later pass touches it:

```python
from torch_spyre._inductor.loop_info import CoarseTileInfo, PropagationPlan

op.loop_info = CoarseTileInfo(
    loop_group_id=(0, 0),          # depth-2 path: group 0, inner slot 0
    loop_count=[2, 4],             # [K_outer, M_inner]
    loop_tiled_dims=[[0], [1]],    # outer loop tiles dim 0; inner tiles dim 1
    loop_tiled_reduction_dims=[[], []],  # no reduction dims (pointwise op)
    tiled_dims_per_read=[
        [[(0, 512)], [(1, 1024)]],   # read of a: dim 0 tiled to 512 (outer),
                                     # dim 1 tiled to 1024 (inner)
        [[(0, 512)], [(1, 1024)]],   # read of b: same tiling
    ],
    output_tiled_dims=[],          # empty: buf0 is loop_internal, so its own
                                    # tile-sized buffer never advances
    propagation=PropagationPlan(
        kind="loop_internal",      # buf0 has no outside consumers, is not a
                                    # graph output
        full_ranges=None,
        reduction=None,
        outside_consumer_names=(),
        is_graph_output=False,
    ),
)
```

`tiled_dims_per_read` and `output_tiled_dims` are *decisions*, not
substituted index expressions — see [Stage 1 (decision, planning
time)](#treatment-by-consumer-topology) below for how `_general_tile_advance`
later substitutes them into each `TensorArg.device_tile_advance_expr`. The
`propagation` field is `_plan_tiling_propagation`'s own decision for how this
op's result crosses its loop boundary — `"loop_internal"` here because
planning already knows, before any transformation pass runs, that `buf0` has
no outside consumers and is not a graph output.

`buf1` (`z = y * c`) is tiled identically except its first read (`y`, i.e.
`buf0`) is loop-invariant at the outer level — `tiled_dims_per_read=[[[],
[(1, 1024)]], [[(0, 512)], [(1, 1024)]]]` — because `buf0`'s own per-tile
buffer is already fully divided by the time `buf1`'s dependency on it is
recorded. `buf1`'s `propagation` is planned `PropagationPlan(kind="copy_out",
full_ranges=[1024, 4096], reduction=None, outside_consumer_names=(),
is_graph_output=True)` — `copy_out` because `buf1` is the literal graph
output, decided at this same planning step, before Pass 3 has created the
copy op or the full-size buffer that will eventually hold `z`.

`_divide_ranges` is applied once per level in outermost-first order (the
`hint_id` in each `(hint_id, K)` pair is used only for per-op `dim_index`
lookup, not by `_divide_ranges` itself):

1. Outer level `(K=2, dim 0)`: `data.ranges [1024, 4096] → [512, 4096]`
2. Inner level `(M=4, dim 1)`: `data.ranges [512, 4096] → [512, 1024]`

The per-inner-iteration `data.ranges` for both ops is `[512, 1024]`.

### LoopLevel IR after CustomPreSchedulingPasses

After `coarse_tile` (which internally runs `_plan_tiling_propagation` and its
three transformation passes, and therefore already inserts
`coarse_tile_copy_buf1` via `_insert_all_write_copy_ops`, plus
`_insert_read_copy_ops` via `_insert_all_read_copy_ops` for each direct
graph-input read — see [Read-side adaptation: full-buffer inputs to a
loop-internal
op](#read-side-adaptation-full-buffer-inputs-to-a-loop-internal-op)),
`span_reduction`, `work_distribution` (`_distribute_work`), and
`scratchpad_planning` have all run, `graph.operations` contains seven ops.
This is the real, unedited output of `format_operations(graph.operations)`
(the same helper `CustomPreSchedulingPasses` itself logs at `INFO`) at
`sencores=4`, in topological order — the first op, `SpyreEmptyFallback`
(`op5` below; the op numbering reflects graph-construction order, not
topological position), is `coarse_tile_copy_buf1`'s eventual full-buffer
target (`z`); `i0`/`i1` are the `inner_fn` index variables for the outer and
inner tiled dims respectively. `a`'s read copy is shown in full below; `b`'s
and `c`'s read copies are elided since each is structurally identical to
`a`'s (same `FixedTiledLayout`, same `loop_info` shape, same single-`ops.load`
`inner_fn`) — only the name of the graph input being copied, the consuming
op, and the `lx` allocation offset differ. Every op's `loop_info` now carries
a `propagation=PropagationPlan(...)` field — `_plan_tiling_propagation`'s
decision, made once, before any transformation pass ran (see [What the
coarse-tiling pass stamps](#what-the-coarse-tiling-pass-stamps) above) — and
every buffer here, including the three read copies, is allocated in `lx`
scratchpad rather than falling back to `hbm_pool`:

```
op5: SpyreEmptyFallback

coarse_tile_read_copy_buf0_arg0_1: ComputedBuffer   # read copy: a → a_tile
  layout=FixedTiledLayout('spyre:0', torch.float16, size=[512, 1024], stride=[1024, 1],
      device_layout=SpyreTensorLayout(device_size=[16, 512, 64], stride_map=[64, 1024, 1],
                                       device_dtype=DataFormats.SEN169_FP16))
  allocation={'lx': 0}
  op_it_space_splits={d0: 4, d1: 1}
  dim_hints=[DimHint(dim_names=['A'], split_count=2, loop_var=d0, is_reduction=False, hint_id=0),
             DimHint(dim_names=['B'], split_count=4, loop_var=d1, is_reduction=False, hint_id=1)]
  loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
      loop_tiled_dims=[[0], [1]], loop_tiled_reduction_dims=[[], []],
      tiled_dims_per_read=[[[(0, 512)], [(1, 1024)]]],
      output_tiled_dims=[[], []],
      propagation=PropagationPlan(kind='loop_internal', full_ranges=None,
          reduction=None, outside_consumer_names=(), is_graph_output=False))
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(arg0_1, i1 + 4096 * i0)   # a, full-buffer read
        return tmp0
    ,
    ranges=[512, 1024],
    origin_node=None,
  )

# ... coarse_tile_read_copy_buf0_arg1_1 (read copy for b, allocation={'lx': 262144}) elided ...

buf0: ComputedBuffer                          # y = a + b
  layout=FixedTiledLayout('spyre:0', torch.float16, size=[512, 1024], stride=[1024, 1],
      device_layout=SpyreTensorLayout(device_size=[16, 512, 64], stride_map=[64, 1024, 1],
                                       device_dtype=DataFormats.SEN169_FP16))
  allocation={'lx': 0}
  op_it_space_splits={d0: 4, d1: 1}
  dim_hints=[DimHint(dim_names=['A'], split_count=2, loop_var=d0, is_reduction=False, hint_id=0),
             DimHint(dim_names=['B'], split_count=4, loop_var=d1, is_reduction=False, hint_id=1)]
  loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
      loop_tiled_dims=[[0], [1]], loop_tiled_reduction_dims=[[], []],
      tiled_dims_per_read=[[[], []], [[], []]],
      output_tiled_dims=[],
      propagation=PropagationPlan(kind='loop_internal', full_ranges=None,
          reduction=None, outside_consumer_names=(), is_graph_output=False))
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(coarse_tile_read_copy_buf0_arg0_1, i1 + 1024 * i0)   # a_tile
        tmp1 = ops.load(coarse_tile_read_copy_buf0_arg1_1, i1 + 1024 * i0)   # b_tile
        tmp2 = tmp0 + tmp1
        return tmp2
    ,
    ranges=[512, 1024],
    origin_node=add,
  )

# ... coarse_tile_read_copy_buf1_arg2_1 (read copy for c, allocation={'lx': 262144}) elided ...

buf1: ComputedBuffer                          # z = y * c
  layout=FixedTiledLayout('spyre:0', torch.float16, size=[512, 1024], stride=[1024, 1],
      device_layout=SpyreTensorLayout(device_size=[16, 512, 64], stride_map=[64, 1024, 1],
                                       device_dtype=DataFormats.SEN169_FP16))
  allocation={'lx': 0}
  op_it_space_splits={d0: 4, d1: 1}
  dim_hints=[DimHint(dim_names=['A'], split_count=2, loop_var=d0, is_reduction=False, hint_id=0),
             DimHint(dim_names=['B'], split_count=4, loop_var=d1, is_reduction=False, hint_id=1)]
  loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
      loop_tiled_dims=[[0], [1]], loop_tiled_reduction_dims=[[], []],
      tiled_dims_per_read=[[], [[], []]],
      output_tiled_dims=[],
      propagation=PropagationPlan(kind='copy_out', full_ranges=[1024, 4096],
          reduction=None, outside_consumer_names=(), is_graph_output=True))
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(buf0, i1 + 1024 * i0)                          # y
        tmp1 = ops.load(coarse_tile_read_copy_buf1_arg2_1, i1 + 1024 * i0)  # c_tile
        tmp2 = tmp0 * tmp1
        return tmp2
    ,
    ranges=[512, 1024],
    origin_node=mul,
  )

coarse_tile_copy_buf1: ComputedBuffer         # identity copy: z_tile → z
  layout=MutationLayoutSHOULDREMOVE('spyre:0', torch.float16, size=[1024, 4096], stride=[4096, 1])
  op_it_space_splits={d0: 4, d1: 1}
  loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
      loop_tiled_dims=[[0], [1]], loop_tiled_reduction_dims=[[], []],
      tiled_dims_per_read=[[[], []]], output_tiled_dims=[[(0, 512)], [(1, 1024)]],
      propagation=PropagationPlan(kind='copy_out', full_ranges=[1024, 4096],
          reduction=None, outside_consumer_names=(), is_graph_output=True))
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(buf1, i1 + 1024 * i0)
        return tmp0
    ,
    ranges=[512, 1024],
    origin_node=None,
  )
```

(`stack_traces` and `origins` fields that `format_operations` also prints are
omitted above for brevity — they only carry the originating Python source
line, not tiling-relevant information. `op5`'s own `ComputedBuffer` fields
are elided the same way: it carries no `loop_info` or `op_it_space_splits`
because it is never itself tiled, only mutated into — it is
`coarse_tile_copy_buf1`'s eventual full-buffer target, `z`. The two elided
read copies, `coarse_tile_read_copy_buf0_arg1_1` (for `b`) and
`coarse_tile_read_copy_buf1_arg2_1` (for `c`), are shown at full length in
the [Generated OpSpec](#generated-opspec-python-wrapper-source) section
below.)

This example uses `sencores=4` (rather than the default 32) purely for
readability: it keeps the per-core address expansion in the generated
`bundle.mlir` below small enough to quote in full while still being real,
unmodified compiler output. The mechanism is identical at any core count —
only the split factor in `op_it_space_splits` and the number of per-core
addresses in `bundle.mlir` scale with `sencores`.

Key points:

- **Every direct read of a graph input goes through a read copy first.**
  `a`, `b`, and `c` are each copied into their own tile-sized buffer
  (`coarse_tile_read_copy_buf0_arg0_1`, `coarse_tile_read_copy_buf0_arg1_1`,
  `coarse_tile_read_copy_buf1_arg2_1`) before `op0`/`op1` ever load from
  them — `_full_buffer_read_deps` flags a direct load from a graph input as
  needing this treatment for the same reason a tiled op can never write
  directly into a full buffer (see [Read-side adaptation: full-buffer
  inputs to a loop-internal
  op](#read-side-adaptation-full-buffer-inputs-to-a-loop-internal-op)): the
  op's own candidate layouts are tile-sized, and the graph input's is the
  one, full-size layout, so the two can never be made stick-compatible
  directly. Each read copy's own `inner_fn` still loads the full buffer
  with the original `4096` coefficient (`i1 + 4096 * i0`); it is `op0`'s and
  `op1`'s `inner_fn`s that now load the *copies* with the tile's own
  `1024` coefficient.
- **`buf1`'s read of `buf0` has coefficient `1024`, not `4096`.** This is the
  detail most likely to be misremembered: `buf0` (`y`) has already been
  divided down to its own per-tile `FixedTiledLayout` with `stride=[1024, 1]`
  by the time `buf1`'s dependency is computed, so the read index reflects
  `buf0`'s own (per-tile) stride, not the original full-tensor stride. This
  is the same `1024` coefficient the read copies use for their own
  consumers, for the same reason: both `buf0` and a read copy are tile-sized
  buffers being read by another op inside the same loop.
- All five tiled ops (the three read copies, `op0`, and `op1`) plus the
  output copy share the same `loop_info` with `loop_group_id = (0, 0)`,
  `loop_count = [2, 4]`, and `loop_tiled_dims = [[0], [1]]` — this is what
  `build_loop_scheduler_nodes` uses to wrap them together in a single
  `CountedLoopSchedulerNode`. `coarse_tile_copy_buf1` is tiled the same way
  even though its own layout is `MutationLayoutSHOULDREMOVE` over the full
  `[1024, 4096]` shape — see
  [MutationLayoutSHOULDREMOVE: the real contract](#mutationlayoutshouldremove-the-real-contract)
  for how that layout redirects without changing the loop's per-tile
  `Pointwise.ranges`.
- **`tiled_dims_per_read` and `output_tiled_dims` are already visible here**,
  not just at the moment `coarse_tile()` first stamps them (see [What the
  coarse-tiling pass stamps](#what-the-coarse-tiling-pass-stamps) above) —
  they survive `span_reduction`, `work_distribution`, and
  `scratchpad_planning` unchanged, since none of those passes touch
  `loop_info`. `buf1`'s `tiled_dims_per_read=[[], [[], []]]` still shows its
  read of `buf0` (`y`) as loop-invariant at the outer level, because `buf0`'s
  own buffer was already divided down before `buf1`'s dependency was
  recorded.
- **A per-level entry that does not advance is *omitted*, never given an
  extent-1 placeholder.** Every read-copy output above
  (`output_tiled_dims=[[], []]`) and every already-tile-local read
  (`buf1`'s read of `buf0`, `[[], []]`) uses this convention: the inner
  list for a given level is empty rather than something like `[(0, 1)]`.
  `_general_tile_advance` (in `spyre_kernel.py`) substitutes `0` for any
  `dep.index` free symbol that has no entry in a level's dict, so an
  omitted dim naturally contributes nothing to that `TensorArg`'s
  `device_tile_advance_expr` — and if *every* level is empty, the whole
  expression comes out `None` (see [Generated
  OpSpec](#generated-opspec-python-wrapper-source) below). An earlier
  version of this pipeline instead kept the dim with an explicit
  `sympy.Integer(1)` extent to mean the same thing, but
  `tiling_expr_to_device_expr` has no zero-coefficient special case, so a
  present-with-extent-1 entry still contributed a spurious nonzero
  `1 * level_symbol` advance term whenever the dependency's index happened
  to reference that dim — silently advancing a buffer that was supposed to
  stay at a fixed address. Dim omission is the only representation that is
  safe for every dependency, tiled or not.
- `ranges = [512, 1024]` is the *per-tile* iteration space (1/8th of the full
  tensor) for every tiled op, including the copy. Work division and codegen
  see only this reduced space; the loop trip counts carry the information
  needed to reconstruct the full addressing.
- `layout.size = [512, 1024]` for `buf0`/`buf1` matches the per-tile `ranges`.
  The layout describes the smaller per-tile output buffer allocated for each
  loop iteration. Per-iteration addressing into the full HBM region is
  handled by `tiled_symbols` / `affine.apply` in `bundle.mlir` at runtime.
  `coarse_tile_copy_buf1`'s layout, by contrast, has `size=[1024, 4096]` —
  the full tensor shape — because `MutationLayoutSHOULDREMOVE` always
  describes the mutation *target*'s shape, not the per-tile source.
- `op_it_space_splits={d0: 4, d1: 1}` is `format_operations`'s
  human-readable reconstruction (via `apply_splits_from_index_coeff`) of the
  `(dict, dict)` coefficient-keyed pair `work_distribution`
  (`_distribute_work`) actually stamps: `d0` (the outer, row-tiled loop
  symbol) is split 4 ways across `sencores`, and `d1` (the inner,
  column-tiled loop symbol) is not split (`1`) — every op in this example,
  including the copy, divides its per-tile work the same way. The
  internal storage is keyed by each symbol's coefficient in the relevant
  index expression rather than by the symbol itself, so that the split
  survives the scheduler's later symbol renaming; see `splits_by_index_coeff`
  / `apply_splits_from_index_coeff` in `pass_utils.py` for the encode/decode
  pair.
- `buf0` (`y`) is the intermediate result. At this point its layout is
  already a `FixedTiledLayout` with `size=[512, 1024]` and
  `allocation={'lx': 0}`, placing it in LX scratchpad memory at address 0.
  Because `y` is produced and fully consumed within the same tile iteration
  and its per-tile size fits in scratchpad, no HBM allocation is needed for
  it at all.
- `buf1` (`z`'s tile-sized producer) is planned `kind="copy_out"` rather than
  `kind="loop_internal"`, because `buf1` is itself the literal graph output
  (`_plan_tiling_propagation` sees `is_graph_output=True` for it — `z`/`op5`
  do not exist yet at planning time; they are created later, by Pass 3,
  `_insert_all_write_copy_ops`). Pass 3 also zeroes
  `buf1.loop_info.output_tiled_dims` when it inserts the copy: `buf1`'s own
  small buffer is loop-internal scratch by construction (written once per
  iteration, fully drained by the inserted `coarse_tile_copy_buf1` copy op
  before the next iteration overwrites it) regardless of why it took the
  copy-out path. An empty `output_tiled_dims` is what tells
  `_general_tile_advance` (`spyre_kernel.py`) to leave this tensor's
  `device_tile_advance_expr` as `None` — no per-iteration address advance —
  which is what lets `scratchpad_planning` place it in `lx` rather than
  falling back to `hbm_pool` — see the OpSpec and `bundle.mlir` sections
  below.

### Generated OpSpec (Python wrapper source)

The Python wrapper emitted by `codegen_kernel()` contains all five tiled ops
(the three read copies, `add`, `mul`) plus the output copy inside a single
nested `LoopSpec`.  Below is the actual output produced by running the e2e
test `test_hint_nested_loop_with_scratchpad` at `sencores=4` (which uses
`spyre_hint(num_tiles_per_dim=...)` / `declare_tensor_dim` / `name_tensor_dims`
with `allow_all_ops_in_lx_planning=True`; the `debug_handle=DebugHandle(...)`
field each real `OpSpec` carries is omitted below for brevity — it records the
originating source location and ATen op for each dispatch and carries no
tiling-relevant information). The read copy for `a` is shown in full; the
read copies for `b` and `c` are elided since each is structurally identical
to `a`'s copy — same `op='identity'` shape, same single input/output
`TensorArg` pair, same fixed (non-advancing) `lx` output — only the minted
`_tile_adv_*` symbol names, `arg_index`, and `allocation` offset differ.
Note the iteration-space symbols: every op except the final output copy uses
`d0`/`d1`, not `c0`/`c1` — only `coarse_tile_copy_buf1` (the last `OpSpec`
below) uses `c0`/`c1`, because it is generated from a different Inductor
scheduler node than the other five:

```python
sdsc_fused_add_mul_0 = async_compile.sdsc('sdsc_fused_add_mul_0',
    [
        LoopSpec(
            count=sympify('2'),        # outer K=2 loop
            body=[
                LoopSpec(
                    count=sympify('4'),    # inner M=4 loop
                    body=[
                        OpSpec(
                            op='identity',        # read copy: a → a_tile
                            is_reduction=False,
                            iteration_space={sympify('d0'): (sympify('512'), 4), sympify('d1'): (sympify('1024'), 1)},
                            op_info={},
                            tiled_symbols=[[sympify('_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl1')], [sympify('_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl0')]],
                            tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl0'): 2, sympify('_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl1'): 4},
                            symbolic_dim_bounds={},
                            args=[
                                TensorArg(              # input a (HBM, full tensor)
                                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[64, 1024, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'hbm': 0},
                                    device_tile_advance_expr=sympify('floor(32768*_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl0) + floor(1048576*_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl1)'),
                                ),
                                TensorArg(              # output a_tile (LX scratchpad)
                                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 0},
                                ),
                            ]
                        ),
                        # ... OpSpec(op='identity', ...) for coarse_tile_read_copy_buf0_arg1_1
                        # (read copy for b, allocation={'lx': 262144}) elided —
                        # structurally identical to a's above ...
                        OpSpec(
                            op='add',
                            is_reduction=False,
                            iteration_space={sympify('d0'): (sympify('512'), 4), sympify('d1'): (sympify('1024'), 1)},
                            op_info={},
                            tiled_symbols=[[sympify('_tile_adv_op0_lvl1')], [sympify('_tile_adv_op0_lvl0')]],
                            tiled_symbol_trip_counts={sympify('_tile_adv_op0_lvl0'): 2, sympify('_tile_adv_op0_lvl1'): 4},
                            symbolic_dim_bounds={},
                            args=[
                                TensorArg(              # input a_tile (LX scratchpad)
                                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 0},
                                ),
                                TensorArg(              # input b_tile (LX scratchpad)
                                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 262144},
                                ),
                                TensorArg(              # output y (LX scratchpad)
                                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 0},
                                ),
                            ]
                        ),
                        # ... OpSpec(op='identity', ...) for coarse_tile_read_copy_buf1_arg2_1
                        # (read copy for c, allocation={'lx': 262144}) elided —
                        # structurally identical to a's above ...
                        OpSpec(
                            op='mul',
                            is_reduction=False,
                            iteration_space={sympify('d0'): (sympify('512'), 4), sympify('d1'): (sympify('1024'), 1)},
                            op_info={},
                            tiled_symbols=[[sympify('_tile_adv_op1_lvl1')], [sympify('_tile_adv_op1_lvl0')]],
                            tiled_symbol_trip_counts={sympify('_tile_adv_op1_lvl0'): 2, sympify('_tile_adv_op1_lvl1'): 4},
                            symbolic_dim_bounds={},
                            args=[
                                TensorArg(              # input y (LX scratchpad)
                                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 0},
                                ),
                                TensorArg(              # input c_tile (LX scratchpad)
                                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 262144},
                                ),
                                TensorArg(              # output z tile (LX scratchpad)
                                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                                    allocation={'lx': 0},
                                ),
                            ]
                        ),
                        OpSpec(
                            op='identity',                 # coarse_tile_copy_buf1
                            is_reduction=False,
                            iteration_space={sympify('c0'): (sympify('512'), 4), sympify('c1'): (sympify('1024'), 1)},
                            op_info={},
                            tiled_symbols=[[sympify('_tile_adv_coarse_tile_copy_buf1_lvl1')], [sympify('_tile_adv_coarse_tile_copy_buf1_lvl0')]],
                            tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_copy_buf1_lvl0'): 2, sympify('_tile_adv_coarse_tile_copy_buf1_lvl1'): 4},
                            symbolic_dim_bounds={},
                            args=[
                                TensorArg(              # input: z tile (LX scratchpad)
                                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                                    allocation={'lx': 0},
                                ),
                                TensorArg(              # output z (HBM, full tensor)
                                    is_input=False, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[64, 1024, 64],
                                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                                    allocation={'hbm': 3},
                                    device_tile_advance_expr=sympify('floor(32768*_tile_adv_coarse_tile_copy_buf1_lvl0) + floor(1048576*_tile_adv_coarse_tile_copy_buf1_lvl1)'),
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ]
)
```

(`debug_handle=DebugHandle(...)`, which every real `OpSpec` above also
carries, is omitted from the listing for brevity — it records the
originating source location and ATen op for each dispatch and carries no
tiling-relevant information.)

Key observations:

- `d0`/`d1` (or, for the final output copy, `c0`/`c1`) are Inductor's
  iteration-space symbols for the two dimensions. `iteration_space` reflects
  the per-inner-iteration tile size `[512, 1024]`.
- **`tiled_symbols` no longer holds plain `c0`/`c1`.** Each op mints its own
  distinct symbols, one per `(op, nesting level)` pair, named
  `_tile_adv_{op_name}_lvl{level}` — e.g. `add`'s output buffer is `buf0`, so
  its symbols are `_tile_adv_op0_lvl0` (outer) and `_tile_adv_op0_lvl1`
  (inner); `a`'s read copy's are
  `_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl0`/`lvl1`; the output copy
  op's are `_tile_adv_coarse_tile_copy_buf1_lvl0`/`lvl1`.
  `tiled_symbols=[[lvl1], [lvl0]]` still records — innermost first — which
  minted symbol corresponds to which nesting level. Minting fresh symbols
  per `(op, level)` rather than reusing Inductor's `c0`/`c1` is what lets two
  different levels that happen to tile the *same* host dimension keep
  distinct, non-colliding terms when their contributions are summed into a
  single `device_tile_advance_expr` (see [Stage 1/Stage
  2](#treatment-by-consumer-topology) above for the full mechanism and the
  flattened-1D case that motivates it).
- **`tiled_symbol_trip_counts` is a new field** alongside `tiled_symbols`: a
  `{symbol: trip_count}` map (e.g. `{_tile_adv_op0_lvl0: 2,
  _tile_adv_op0_lvl1: 4}` for `add`) recording each minted level symbol's
  loop trip count, so downstream codegen can recover "how many steps does
  this level take" without a separate stored extent field on `TensorArg`.
- `symbolic_dim_bounds={}` is empty here because all loop counts are
  concrete integers.
- **Only the four full-tensor HBM `TensorArg`s carry a
  `device_tile_advance_expr`.** This is the substituted, per-arg sympy
  expression `_general_tile_advance` builds from `loop_info`'s
  `tiled_dims_per_read`/`output_tiled_dims` decisions — the sole
  tile-advance mechanism (see `op_spec.py`'s `TensorArg` docstring) — e.g.
  `a`'s read copy's *input* (the full-buffer read of `a` itself) gets
  `floor(32768*_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl0) +
  floor(1048576*_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl1)`, combining
  the outer level's per-step byte advance (`32768`) and the inner level's
  (`1048576`) into one expression, one additive term per level; `z`'s output
  (written by the final `identity` copy) gets the analogous
  `coarse_tile_copy_buf1`-keyed expression. Every other `TensorArg` —
  `a_tile`/`b_tile`/`c_tile` and `y`/`mul`'s own output, all in `lx` — has
  **no** `device_tile_advance_expr` field at all (it is omitted by the
  printer whenever the value is `None`; see
  `spyre_kernel.py::SpyreKernel._general_tile_advance`/`create_tensor_arg`).
  In every case this is because `loop_info.tiled_dims_per_read`/
  `output_tiled_dims` omits every level's dim entirely for that dependency
  (see the dim-omission point in [LoopLevel
  IR](#looplevel-ir-after-custompreschedulingpasses) above, and the
  Small Example's `buf1` paragraph above for how `output_tiled_dims` ends up
  empty for a loop-internal buffer) — `_general_tile_advance` substitutes
  `0` for every unlisted dim and, since no level contributes a term, returns
  `None` outright. A missing `device_tile_advance_expr` means
  `generate_bundle` addresses that `TensorArg` with a fixed, non-advancing
  address — no `affine.apply` per iteration; see
  `codegen/compute_ops.py`'s `_tensor_tiled_by_symbol`, which treats a
  `None` advance expression as the "does not advance" case for
  `bundle.mlir` purposes. `op0`'s (`add`'s) own two inputs reuse the same
  fixed `a_tile`/`b_tile` LX addresses the read copies just wrote — the read
  copy has already done the full-tensor addressing once per tile, so `op0`
  never touches the `32768`/`1048576`-scale expressions at all, and since it
  takes no HBM operand at all it needs no address operand whatsoever (see
  the next section).
- **All five tile-sized buffers fit in `lx`; none fall back to `hbm_pool`.**
  `a_tile`/`b_tile`/`c_tile` (`allocation={'lx': 0}`, `{'lx': 262144}`,
  `{'lx': 262144}` respectively — `b_tile` and `c_tile` alias the same
  offset because they are never live at the same time) plus `y` and `mul`'s
  own output are all produced and fully consumed within the same tile
  iteration. `scratchpad_planning`'s allocator fits all of them in LX
  scratchpad simultaneously at this tile size (`[512, 1024]` fp16, 4 cores),
  so no buffer here ever needs the bulk `hbm_pool` fallback
  (`constants.py`'s `INTERMEDIATES_SEGMENT`). Whether a given buffer lands
  in `lx` or spills to `hbm_pool` is a placement decision made per
  compilation by `scratchpad_planning`, based on how much same-lifetime
  data is live at once — it is not a fixed property of being a read-copy
  output, and a larger tile size or more simultaneously-live buffers could
  still force a spill to `hbm_pool` in a different example.
- The intermediate tensor `y` (output of `add`, input to `mul`) has
  `allocation={'lx': 0}` — it lives in LX scratchpad memory at address 0.
  Its `device_size=[16, 512, 64]` reflects the per-tile shape `[512, 1024]`.
  Its `loop_info.output_tiled_dims` is empty at every level (it is
  `add`'s loop-internal output, kind `"loop_internal"`), so it carries no
  `device_tile_advance_expr` — `generate_bundle` addresses it with a fixed
  base address, no `affine.apply` advance.  Because `y` is produced and
  fully consumed within the same tile iteration, no HBM allocation is
  needed.
- The final output `z` (output of `mul`) has no inside consumers, so it
  takes the **copy-out** path (planned `kind="copy_out"`): `mul` writes its
  per-tile result into its own small buffer, and a separate loop-tagged
  `identity` op (the last `OpSpec` above, generated from
  `coarse_tile_copy_buf1`) copies each tile into the correct slice of `z`'s
  own, separately-allocated full HBM buffer (`allocation={'hbm': ...}`,
  `arg_index=3`).  Because `mul`'s own small buffer is fully drained by that
  copy op every iteration before the next iteration overwrites it, it is
  loop-internal scratch by construction. Pass 3 (`_insert_all_write_copy_ops`)
  zeroes `mul`'s own `loop_info.output_tiled_dims` directly when it inserts
  the copy, and `scratchpad_planning` places it in `lx` (address 0, aliasing
  `y`'s slot since `y` and `mul`'s output are never live at the same time
  within scratchpad's allocator).  The identity copy is still the op whose
  `MutationLayoutSHOULDREMOVE` targets the full buffer; the per-iteration
  copy offset into *that* full buffer is computed by `affine.apply` in
  `bundle.mlir` (see next section) — only `mul`'s own small buffer has no
  advance, not `z`'s full-size target.
- HBM inputs `a`, `b`, `c` also have `device_size=[64, 1024, 64]` — the full
  tensor shape `[1024, 4096]` in Spyre stick layout.  Their
  `device_coordinates` use `d0`/`d1` (or, for `z`'s output copy, `c0`/`c1`)
  to index the per-iteration tile window into the full tensor.  Note that
  `add` and `mul` no longer read `a`/`b`/`c` directly — as described above,
  each is read once per tile by its own read-copy `OpSpec`, whose *output*
  (`a_tile`/`b_tile`/`c_tile`) is what `add`/`mul` actually consume.  The LX
  scratchpad tensor `y` and `mul`'s per-tile output both have
  `device_size=[16, 512, 64]`, the stick-layout shape for `[512, 1024]`
  fp16: 16 sticks of 64 columns across 512 rows, and both carry
  `allocation={'lx': 0}` with an empty `output_tiled_dims` — both are
  produced and fully consumed entirely inside the loop body (`y` by `mul`;
  `mul`'s own output by the `identity` copy), so both get a dedicated LX
  scratchpad slot with no HBM traffic and no advancing address, even though
  `mul`'s output also has an outside reader (the copy op) that `y` does
  not.  If the tile-sized buffers did not all fit in scratchpad
  simultaneously (e.g. a larger tile size, or more of them live at once),
  `scratchpad_planning` would fall back to `allocation={'hbm_pool': ...}`
  instead for whichever ones spill — the bulk-allocated HBM region
  (`constants.py`'s `INTERMEDIATES_SEGMENT`) — and the buffer would still
  carry an empty `output_tiled_dims`, since that reflects the
  loop-internal-scratch *lifetime* of the buffer, not which memory it
  happens to land in.

### Generated `bundle.mlir`

The SDSC compiler (`compile_op_spec`) translates `tiled_symbols` into per-loop
byte strides, producing a 2-dimensional `affine_map` for each `TensorArg`
whose `device_tile_advance_expr` is non-`None`. There is exactly one such
map, `#map_0`, covering the four full-tensor HBM operands (`a`, `b`, `c`,
`z`, bound to `%arg_0`..`%arg_3`). `a_tile`/`b_tile`/`c_tile` and `y`/`mul`'s
own output all live in `lx` (see [Key observations](#generated-opspec-python-wrapper-source)
above) and each has `device_tile_advance_expr=None`, so none of them get an
affine map, a per-core address, or any operand at all: `add`'s and `mul`'s
`sdsc_execute` dispatches below take **zero** operands, since every value
they touch is a fixed LX address baked directly into the compiled
`sdsc_2.json`/`sdsc_4.json` kernels rather than passed in from the bundle.
There is also no `%pool_base_addr` argument at all — the bundle's only
inputs are the four full-tensor HBM arguments:

```none
#map_0 = affine_map<(d0, d1)[s0] -> (s0 + 65536*d0 + 2097152*d1)>
module {
    func.func @sdsc_bundle(%arg_0_base_addr: !sdscbundle.input_arg<index>,
                            %arg_1_base_addr: !sdscbundle.input_arg<index>,
                            %arg_2_base_addr: !sdscbundle.input_arg<index>,
                            %arg_3_base_addr: !sdscbundle.input_arg<index>) {
        %arg_0 = sdscbundle.input_arg_extract value from %arg_0_base_addr : !sdscbundle.input_arg<index> -> index
        %arg_1 = sdscbundle.input_arg_extract value from %arg_1_base_addr : !sdscbundle.input_arg<index> -> index
        %arg_2 = sdscbundle.input_arg_extract value from %arg_2_base_addr : !sdscbundle.input_arg<index> -> index
        %arg_3 = sdscbundle.input_arg_extract value from %arg_3_base_addr : !sdscbundle.input_arg<index> -> index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %loop_bound_0 = arith.constant 2 : index
        %loop_bound_1 = arith.constant 4 : index

        // per-core address = base + core_index * 16384 bytes, for each of the
        // 4 cores (sencores=4); shown here for arg_0 (tensor a), identical
        // patterns repeat for arg_1 (b), arg_2 (c), arg_3 (z) — omitted here;
        // see the full real output linked below. These are all computed
        // once, outside the loop nest: none of them vary per iteration.
        %arg_0_core_offset_16384 = arith.constant 16384 : index
        %arg_0_core_16384 = arith.addi %arg_0, %arg_0_core_offset_16384 : index
        %arg_0_core_offset_32768 = arith.constant 32768 : index
        %arg_0_core_32768 = arith.addi %arg_0, %arg_0_core_offset_32768 : index
        %arg_0_core_offset_49152 = arith.constant 49152 : index
        %arg_0_core_49152 = arith.addi %arg_0, %arg_0_core_offset_49152 : index
        // ... (arg_1_core_*, arg_2_core_*, arg_3_core_* follow the same
        // pattern above — elided here ...)

        scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
            scf.for %i_1 = %c0 to %loop_bound_1 step %c1 {
                // read copy: a(hbm)→a_tile(lx)
                %addr_0 = affine.apply #map_0(%i_0, %i_1)[%arg_0]
                %addr_1 = affine.apply #map_0(%i_0, %i_1)[%arg_0_core_16384]
                %addr_2 = affine.apply #map_0(%i_0, %i_1)[%arg_0_core_32768]
                %addr_3 = affine.apply #map_0(%i_0, %i_1)[%arg_0_core_49152]
                sdscbundle.sdsc_execute (%addr_0, %addr_1, %addr_2, %addr_3)
                    {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1, -2, -3, -4]}

                // read copy: b(hbm)→b_tile(lx)
                %addr_4 = affine.apply #map_0(%i_0, %i_1)[%arg_1]
                %addr_5 = affine.apply #map_0(%i_0, %i_1)[%arg_1_core_16384]
                %addr_6 = affine.apply #map_0(%i_0, %i_1)[%arg_1_core_32768]
                %addr_7 = affine.apply #map_0(%i_0, %i_1)[%arg_1_core_49152]
                sdscbundle.sdsc_execute (%addr_4, %addr_5, %addr_6, %addr_7)
                    {sdsc_filename="sdsc_1.json", "symbol_ids"=[-5, -6, -7, -8]}

                // add: a_tile(lx)+b_tile(lx)→y(lx) — zero operands
                sdscbundle.sdsc_execute () {sdsc_filename="sdsc_2.json", "symbol_ids"=[]}

                // ... read copy: c(hbm)→c_tile(lx) (sdsc_3.json) elided —
                // same structure as sdsc_0/sdsc_1 above, using %arg_2 ...

                // mul: y(lx)*c_tile(lx)→mul_output(lx) — zero operands
                sdscbundle.sdsc_execute () {sdsc_filename="sdsc_4.json", "symbol_ids"=[]}

                // identity: mul_output(lx)→z(hbm) — z is the only operand
                %addr_12 = affine.apply #map_0(%i_0, %i_1)[%arg_3]
                %addr_13 = affine.apply #map_0(%i_0, %i_1)[%arg_3_core_16384]
                %addr_14 = affine.apply #map_0(%i_0, %i_1)[%arg_3_core_32768]
                %addr_15 = affine.apply #map_0(%i_0, %i_1)[%arg_3_core_49152]
                sdscbundle.sdsc_execute (%addr_12, %addr_13, %addr_14, %addr_15)
                    {sdsc_filename="sdsc_5.json", "symbol_ids"=[-13, -14, -15, -16]}
            }
        }
        return
    }
}
```

(This is a lightly elided version of the real captured output — the
read-copy dispatch for `c` (`sdsc_3.json`) is elided since it repeats
`sdsc_0`'s/`sdsc_1`'s structure exactly using `%arg_2`, and the per-core
address setup for `arg_1`/`arg_2`/`arg_3` is elided from the
constant-declaration block for the same reason. Nothing about `add`, `mul`,
or the final `identity` copy is elided — notice neither `add` nor `mul` has
any `affine.apply`, or indeed any operand at all: `symbol_ids=[]` on both.)

Key points:

- **Only one affine map, and no `%pool_base_addr` argument.** `#map_0`
  addresses the four full-tensor HBM operands (`a`, `b`, `c`, `z` — bound to
  `%arg_0`..`%arg_3`); the bundle takes exactly these four arguments, with
  no fifth pool-base-address parameter at all. `a_tile`/`b_tile`/`c_tile`
  and `y`/`mul`'s own output do not get a second map: each one's
  `TensorArg.device_tile_advance_expr` is `None` (see the OpSpec above — a
  direct consequence of the dim-omission convention for "this buffer does
  not advance"), so `compile_op_spec` never builds an `affine_map`, a
  per-core address, or any operand for them at all.
- **`add` and `mul` take zero operands.** Each inner-loop iteration runs, in
  order: the read copy for `a` (`sdsc_0.json`, 4 operands), the read copy
  for `b` (`sdsc_1.json`, 4 operands), `add` (`sdsc_2.json`, **0** operands
  — both its inputs and its output are fixed LX addresses baked into the
  compiled kernel), the read copy for `c` (`sdsc_3.json`, 4 operands), `mul`
  (`sdsc_4.json`, **0** operands, same reason), and finally the output
  `identity` copy (`sdsc_5.json`, 4 operands, since `z` is a full-tensor HBM
  output that does advance). This is the `bundle.mlir`-level consequence of
  the read-side adaptation described earlier in this doc: every full-buffer
  graph-input read is intercepted by its own dispatch before the op that
  actually needs the data ever runs, so by the time `add`/`mul` execute,
  every operand they touch is already a fixed, non-advancing LX slot with
  nothing left for the bundle to compute or pass in.
- **Every full-tensor HBM operand is still expanded into `sencores` per-core
  addresses**: at `sencores=4`, `arg_0` (tensor `a`) contributes its own
  base address plus three `arith.addi`-computed offsets
  (`arg_0_core_16384`, `arg_0_core_32768`, `arg_0_core_49152`, stepping by
  `16384` bytes — `65536 / 4`), and likewise for `arg_1`/`arg_2`/`arg_3` — 4
  operands per full-tensor dispatch (4 cores × 1 tensor). At the default
  `sencores=32` this would instead be 32 per-core addresses per operand —
  real, but too large to usefully quote in a doc, which is why this example
  fixes `sencores=4`.
- **Neither LX scratchpad tensor, nor the three tile-sized LX copies,
  appears as a symbol at all.** `a_tile`/`b_tile`/`c_tile`, `y`, and `mul`'s
  own output all have an empty `output_tiled_dims` (see the OpSpec above),
  so none of them needs an address computation of any kind, per-core or
  otherwise — each is just a compile-time-fixed offset inside the compiled
  `.json` kernel for its dispatch.

## Layer 1 — Pre-scheduling IR pass

### Attribute contract on `ir.Operation`

The coarse-tiling pass stamps a single `loop_info: CoarseTileInfo` attribute
onto each `ir.Operation` that participates in a loop group.  `CoarseTileInfo`
is a plain Python dataclass defined in
`torch_spyre/_inductor/loop_info.py` and attached with `setattr`; no Inductor
base class is modified.

```python
@dataclass
class CoarseTileInfo:
    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]] = field(default_factory=list)
```

| Field | Type | Meaning |
|---|---|---|
| `loop_group_id` | `tuple[int, ...]` | Nesting-path tuple identifying which loop group this op belongs to. Its length equals the nesting depth. All ops sharing the same tuple form the body of the innermost counted loop at that path. |
| `loop_count` | `list[sympy.Expr]` | Trip counts, one per nesting level from outermost to innermost. For a flat (depth-1) group this is a 1-element list `[K]`. For a two-level nested group it is `[K1, K2]`. All ops sharing the same `loop_group_id` must agree on the count at every level. |
| `loop_tiled_dims` | `list[list[int]]` | Per-level positional indices into `data.ranges` (the output iteration space) that are divided by the corresponding count. For a flat group: `[[0]]` (tile only dim 0). For a two-level nested group: `[[0], [1]]`. An empty sub-list means the op is loop-invariant at that level in the output space. |
| `loop_tiled_reduction_dims` | `list[list[int]]` | Per-level positional indices into `data.reduction_ranges` that are tiled at that level. Parallel to `loop_tiled_dims`. An empty sub-list means no reduction dim is tiled at that level. Defaults to `[]` for backward compatibility (pure output-dim tiling). |

The pass also **rewrites the op's iteration ranges**: for each level, the
dimensions at the corresponding indices in `loop_info.loop_tiled_dims` are
divided by the corresponding count in `loop_info.loop_count`, so that each
inner `OpSpec` describes only the work done per innermost-loop iteration.
For reduction-dim tiling, the indices in `loop_tiled_reduction_dims` drive
division of `data.reduction_ranges` instead of `data.ranges`.

`loop_group_id` is a tuple rather than a flat integer to support nested
loops.  See "Nested loops and the `loop_group_id` tree" below.

### Why these four fields are sufficient

`loop_count` is redundant across all ops sharing the same `loop_group_id`
(they must agree), but keeping it on each op means the post-fusion pass does
not need to maintain a separate side table.  The `loop_group_id` is the join
key.  `loop_tiled_dims` is the bridge between the pre-scheduling pass (which
operates on positional `data.ranges` indices) and the codegen phase (which
uses named sympy Symbols) — it is read by `create_op_spec` to identify, by
index, which scheduler-level symbols correspond to the tiled output dimensions
and should be recorded in `OpSpec.tiled_symbols`.  Each loop level gets its
own sublist (innermost first) so that `tiled_symbols` covers every loop
variable for the op.  Using a list-of-lists of indices (rather than a count
or a flag) allows
different ops in the same loop to tile non-contiguous or differently
positioned dimensions of their respective iteration spaces.

`loop_tiled_reduction_dims` plays the same bridging role for reduction-dim
tiling.  For a `Reduction` op, `iteration_space()` returns `reads.ranges`,
which has output-dim symbols first and reduction-dim symbols last.
`create_op_spec` determines the split point by counting the output-side write
dep's ranges (`n_output_syms = len(write_dep.ranges)`), then indexes
`it_space_keys[n_output_syms + r]` for each reduction-dim index `r` in the
flattened `loop_tiled_reduction_dims`.  These symbols are appended to
`tiled_syms` so the runtime correctly advances the input tensor pointer
between tiles.

Crucially, `loop_tiled_dims` is **per-op**: `plan_coarse_tile_groups` consults
each op's own `DimHint.loop_var` for each nesting level (via
`_loop_var_to_ranges_pos`/`_loop_var_to_reduction_ranges_pos`) rather than
applying a fixed spec-op index to every op.  This handles broadcast ops and
other ops whose iteration space lacks a
particular dimension — those ops get an empty sub-list `[]` for the
corresponding level and are not split along that axis (they become
loop-invariant at that depth, as detected by `_plan_tiling_propagation` and
planned `kind="loop_internal"`).

### `Loops` is a frozen dataclass

Inductor's `ir.Loops` (the base of `Pointwise` and `Reduction`) is
declared `@ir_dataclass(frozen=True)`, so `data.ranges = x` raises
`FrozenInstanceError`.  The tiling pass uses `object.__setattr__` to
bypass this:

```python
object.__setattr__(data, "ranges", ranges)
```

### Public API: `coarse_tile()`

```python
def coarse_tile(
    graph: GraphLowering,
    groups: list[tuple],
    group_idx_offset: int = 0,
) -> None:
```

`groups` is a pre-computed list of group tuples produced by
`hints_to_coarse_tile_groups`.  Each `ops` list must be a contiguous
sub-sequence of `graph.operations`; a gap indicates a data-flow dependency
crossing the group boundary and raises `RuntimeError`.  The full
`GraphLowering` is required (not just the operations list) because
`_insert_all_reduction_ops`/`_insert_all_write_copy_ops` call `V.graph`
APIs to allocate new buffers.
`group_idx_offset` lets a caller make a second `coarse_tile()` call on the
same graph (e.g. hint-driven groups stamped pre-stickification, followed by
a later span-overflow-driven call) without the new call's group IDs
colliding with IDs already stamped by the earlier one.

`coarse_tile()` itself is a thin plan-then-transform driver, and planning
is itself two calls, not one: `plan_coarse_tile_groups(operations, groups)`
decides each op's tiling attributes (loop nesting, which dims are tiled),
and `_plan_tiling_propagation(operations, groups, plan)` decides, for every
tiled op, how its result crosses its loop boundary — see "Buffer
propagation," below. Both run with zero IR mutation, up front, for every
group in the list, and both raise `Unsupported` if any op in any group
can't be tiled (see "Sequential recurrences are rejected at planning time"
below). Only if planning succeeds does `coarse_tile()` move on to
transformation: it loops over `groups` again and calls `_apply_plan` once
per group to perform the actual IR mutation (stamping `loop_info`, dividing
ranges). Transformation then runs three fixed, non-interleaved passes over
the whole op list — `_insert_all_read_copy_ops` (Pass 1), then
`_insert_all_reduction_ops` (Pass 2), then `_insert_all_write_copy_ops`
(Pass 3), each consuming the plan's decisions rather than making new ones —
followed by `_patch_retiled_load_indexes`, once per group.  There is no
per-group interleaving of planning and mutation — every group is planned
before any group is transformed — and no interleaving of the three
transformation passes either — every op is fully handled by Pass 1 before
Pass 2 starts, and by Pass 2 before Pass 3 starts.

Each group tuple has the form:

```python
(ops, levels)
```

where `levels` is a list of `(hint_id, K)` pairs, outermost first:

```python
(ops, [(hint_id_0, K1), (hint_id_1, K2)])
```

`hint_id` is the integer ID assigned by the enclosing `spyre_hint` scope
(smaller IDs are outer scopes).  Whether a level tiles an output dimension
or a reduction dimension is a **per-op** property: `plan_coarse_tile_groups`
consults each op's own `DimHint.is_reduction` for each level (building
`hint_id_to_ranges_pos`/`hint_id_to_reduction_ranges_pos` via
`_loop_var_to_ranges_pos`/`_loop_var_to_reduction_ranges_pos`) rather than
carrying `is_reduction` at the group level.  This means broadcast ops and
`Pointwise` ops inside a
reduction-level group get an empty sub-list for that level and are not
split along that axis.  `tiled_dims` are likewise **not** in the pair —
they are derived per-op inside `plan_coarse_tile_groups` by consulting each
op's `DimHint.loop_var`.

`plan_coarse_tile_groups` always receives this canonical list-of-pairs
representation; it is built by `_hints_levels()` inside
`hints_to_coarse_tile_groups` in `coarse_tile.py` before `coarse_tile()`
plans and then transforms each group.

### `reorder_unhinted_interlopers`: pre-grouping pass

Before `hints_to_coarse_tile_groups` walks the operation list,
`reorder_unhinted_interlopers` reorders any unhinted `ComputedBuffer` that
would otherwise break a contiguous run of same-hint ops into two separate groups.

#### Why it is needed

`hints_to_coarse_tile_groups` collects consecutive same-key ops into a group and
stops as soon as the key changes.  An unhinted op sandwiched between two
same-key ops would split what should be one group into two.  This pass attempts
to move ("reorder") such interlopers either before or after the run so the run
becomes contiguous.

#### Algorithm invariants enforced by the pass

The algorithm is a two-cursor scan.  The outer cursor `i` starts at the first
op of each new candidate run.  The inner cursor `j` walks forward, absorbing
same-key ops.  When it encounters an unhinted `ComputedBuffer` interloper it
applies one of three outcomes:

1. **Move before** (`_can_move_before` returns `True`): `ops.insert(run_start,
   ops.pop(j))`.  `run_start` is incremented by 1 to skip past the newly
   inserted op; `j` stays pointing at the next candidate.
2. **Move after** (`_can_move_after` returns `True`): `ops.insert(run_end - 1,
   ops.pop(j))`.  `run_end` is one past the *last* same-key op in the remainder
   (found by a backward scan), not merely the next one.  This ensures the entire
   remaining run is covered when later interlopers would otherwise still split it.
   After `pop(j)` shifts everything left, the insertion at `run_end - 1` lands
   just after the last hinted op.
3. **Neither** (both checks fail): raises `RuntimeError` with the op name and the
   hint group it is blocking.

When **both** directions are legal, the op is moved **before** the run (closer
to its original position).

#### Legality check: `_no_dep_conflict`

A move is legal when it introduces no new data-flow hazard between the interloper
and every op in the skipped range.  `_no_dep_conflict` checks four conditions:

- **RAW** (read-after-write): the interloper reads a buffer written by an op in
  the range (would observe a stale value after reordering).
- **WAW** (write-after-write): the interloper writes a buffer also written by an
  op in the range (order of writes matters; both directions are conservatively
  flagged).
- Symmetric versions: an op in the range reads or mutates a buffer written by the
  interloper.

`_no_dep_conflict` includes `op.get_mutation_names()` on both sides so that WAW
hazards through mutation aliases are detected.  The WAW check is deliberately
conservative: two ops mutating the same buffer cannot be safely reordered in
either direction.

#### Non-`ComputedBuffer` ops are hard stops

If the inner cursor `j` reaches an op that is not a `ComputedBuffer`, or a
`ComputedBuffer` whose hint key is different from the current run's key and
is non-`None` (i.e., it belongs to a *different* hint group), the scan stops
immediately.  Such ops cannot be moved by this pass.

#### Trailing consumer pattern

If no same-key op exists after position `j` (i.e. the unhinted op is after the
last hinted op in this group), `run_end` is `None` and the scan ends silently.
The unhinted op is not an interloper in this case — it is a trailing consumer.

#### Key invariant summary

| Invariant | How it is enforced |
|---|---|
| Every interloper is moved before or after the run | `RuntimeError` if neither direction is legal |
| Move-before uses the run start (not last position) | `run_start` used as insertion target |
| Move-after uses the last same-key op (not just the next) | Backward scan for `run_end` |
| WAW hazards are treated as conflicts in both directions | `get_mutation_names()` included in both `op_written` and `op_needs` |
| Non-`ComputedBuffer` ops are not moved | Type check in `_can_move_before` / `_can_move_after` |
| Only unhinted `ComputedBuffer`s are candidates | `ckey is not None` triggers hard stop |

### Groups derivation and placement in `CustomPreSchedulingPasses`

Groups are derived automatically from `spyre_hint(num_tiles_per_dim=...)` annotations
(`slices=` and `tiles=` are deprecated aliases that still work)
via `hints_to_coarse_tile_groups` (in `torch_spyre/_inductor/wsr/coarse_tile.py`),
which is a no-op when no hints are present.  `CustomPreSchedulingPasses`
maintains a `self.passes` list of uniform `Callable[[GraphLowering], None]`
entries, run in order by `__call__`.  Config-gated or multi-step groups are
wrapped in private helpers tagged with `@_runs(...)` for cache-key purposes:

```python
self.passes = [
    deadcode_elimination,
    #
    # Working Set Reduction (hint-driven, pre-stickification)
    propagate_named_dims,
    assign_dim_hints,
    _maybe_coarse_tile_hints,      # reorder_unhinted_interlopers + hints_to_coarse_tile_groups
                                   # + coarse_tile, on host-side FixedLayout
    #
    # Tensor Layout (Stickification)
    split_multi_ops,
    propagate_spyre_tensor_layouts,
    validate_ops,
    optimize_restickify_locations,
    finalize_layouts,
    insert_restickify,
    insert_post_mutation_restickify,
    insert_bmm_padding,
    #
    dedup_and_promote_constants,
    #
    # Working Set Reduction (device-layout-aware, post-stickification)
    _maybe_coarse_tile_span_overflow,  # span_overflow_groups + coarse_tile,
                                       # needs FixedTiledLayout.device_layout
    # Core Division
    span_reduction,
    _distribute_work,             # calls cost_model_matmul_division + work_distribution
    # LX Planning
    _maybe_scratchpad_planning,   # config-gated; calls scratchpad_planning
]
```

This ordering is required by several constraints:

**`propagate_named_dims` and `assign_dim_hints` must run before coarse tiling.**
`propagate_named_dims` propagates `name_tensor_dims()` annotations through the
op graph, attaching named dimension metadata to each `ir.Operation`.
`assign_dim_hints` then combines those named dimensions with the `spyre_hint`
scope annotations (attached to FX nodes as `meta["custom"]`) to produce
`op.dim_hints` — a flat list of `DimHint` objects consumed by
`hints_to_coarse_tile_groups` to form the coarse tiling groups.

**Coarse tiling is split into two slots, not one.** `_maybe_coarse_tile_hints`
(hint-derived loop groups) runs immediately after dead-code elimination,
before stickification: it only needs host-side `FixedLayout` (size/stride)
and loop-variable ranges, and running it here means `_divide_ranges` never
has to call a cross-phase `_resize_device_layout` correction step —
stickification computes the correct `SpyreTensorLayout` directly from the
already-divided ranges. This also removes a cross-phase contract that used
to exist between `insert_restickify` and hint-copy forwarding
(issue #3135). `_maybe_coarse_tile_span_overflow` (spans that overflow the
hardware memory budget, detected independently of hints) stays in the old
post-stickification slot below, because span arithmetic needs
`FixedTiledLayout.device_layout` (device size, stride map), which does not
exist yet pre-stickification.

**Must run after stickify and padding.**  `propagate_spyre_tensor_layouts`,
`insert_restickify`, and `insert_bmm_padding` establish the final tiled
memory layout for each tensor.  The span-overflow half of coarse tiling
must see the post-stickify, post-padding shapes or it will split on the
wrong dimension or produce a non-stick-aligned inner size.

**Must run before `work_distribution`.**  `work_distribution` stamps
`op_it_space_splits` on each `ir.Operation` to assign per-core work
slices.  It must see the already-reduced (inner) iteration space so that
cores divide the per-iteration work, not the full pre-tiling iteration
space.  Running coarse tiling after `work_distribution` would produce
`op_it_space_splits` values sized for the full range, which would then
be wrong relative to the reduced `ranges` written by the tiling pass.
`span_reduction` and `cost_model_matmul_division` have the same requirement
and already run before `work_distribution`, so placing `coarse_tile` with
them is consistent.

`scratchpad_planning` must run after coarse tiling because it sizes
scratchpad allocations to fit the per-iteration working set.  If it ran
before, it would see the full iteration space and allocate too much —
defeating the working-set reduction that coarse tiling is designed to
achieve.  `scratchpad_planning` receives the full `GraphLowering` object
(not just `operations`) because it needs access to graph-level metadata
for buffer lifetime analysis.

### Buffer propagation: planning and the three transformation passes

Its job is to ensure that any op whose result is consumed **outside** the
loop (or is a graph output) exposes a complete, fully-sized buffer to its
consumers.  Ops whose outputs are consumed only inside the loop are marked
so `generate_bundle` does not advance their base addresses.

This is split, like the rest of `coarse_tile()`, into a zero-mutation
planning step and a fixed sequence of transformation passes that only
consume the plan's decisions:

- **Planning — `_plan_tiling_propagation(operations, groups, plan)`.**
  Runs right after `plan_coarse_tile_groups`'s own per-op loop, over the
  same `groups`/`plan`. For every tiled op it decides a `PropagationPlan`
  (`torch_spyre/_inductor/loop_info.py`), stored on that op's
  `CoarseTileInfo.propagation`, with one of three `kind`s:
  `"loop_internal"`, `"copy_out"`, or `"reduction"` — see "Use-def analysis"
  and "Treatment by consumer topology," below, for how `kind` is chosen.
  A `Reduction` op tiled over a reduction dim always gets `kind="reduction"`
  (see "Reduction tiling," further below); this check runs first, before
  the use-def analysis that decides `loop_internal` vs. `copy_out` for
  every other tiled op.
- **Pass 1 — `_insert_all_read_copy_ops(operations)`.** For every tiled op
  that directly reads a full-size buffer, inserts a tile-sized read
  copy-in (see "Read-side adaptation," below). Runs first because Pass 2
  and Pass 3 read each op's *current* reads/loader, and a tiled-reduction
  or copy-out op may itself need a read copy-in before its own machinery is
  built.
- **Pass 2 — `_insert_all_reduction_ops(operations)`.** For every op whose
  plan has `kind == "reduction"`, builds the fill/combine/accumulator
  buffers from the plan's `ReductionPlan` data (see "Reduction tiling,"
  below).
- **Pass 3 — `_insert_all_write_copy_ops(operations)`.** For every op whose
  plan has `kind == "copy_out"`, allocates the full buffer and inserts the
  copy-out (see "Treatment by consumer topology," below). Runs last because
  `_allocate_full_buffer`/`_insert_copy_op` read the op's *current*
  reads/loader/layout, which Pass 1/2 may have already changed.

A `"reduction"` op is never also `"copy_out"` — the plan's `kind` routes
each op to exactly one of the three, so Pass 2 and Pass 3 never compete for
the same op. Each pass re-resolves "the current object for buffer name X"
fresh from `operations` at its own start, rather than trusting an object
reference captured before an earlier pass ran — this is why no per-op
resync hack is needed between passes: a name, once assigned, is stable
across any later replacement (see `PropagationPlan`'s own docstring on name
stability).

#### Use-def analysis

For each `ComputedBuffer` in a loop group, planning asks two questions:

1. **Does this buffer have outside consumers?**  A consumer is "outside" if
   it carries a different `loop_info.loop_group_id` prefix, or has no
   `loop_info` at all.  Graph outputs (recorded in the Inductor buffer's
   `users`/`get_alias_name` machinery) count as outside consumers.
   `_find_outside_consumers_planned` (the planning-time helper; the
   transformation-time original, `_find_outside_consumers`, still exists
   and is called by Pass 3) answers this by name, not object identity —
   see the name-stability note above.

2. **Does this buffer have inside consumers?**  A consumer is "inside" if it
   shares the same `loop_info.loop_group_id` tuple (i.e. it is another op in
   the same innermost loop body).

If a tiled op has no outside consumers and is not a graph output, planning
assigns `kind="loop_internal"` and stops — no buffer allocation, no copy
op, nothing to build. Otherwise it falls through to the copy-out treatment
below.

(treatment-by-consumer-topology)=

#### Treatment by consumer topology

The perimeter is shape-asymmetric.  On the producer side (tile → full), a
tiled op writes per-tile data while an outside consumer wants full data — a
genuine shape mismatch needing adaptation.  On the consumer side (full →
tile), the loop body reads from full HBM tensors using tile-sized windows
via `affine.apply` — no conversion, just addressing.  Only producer-side
crossings need adaptation.

For each tiled `ComputedBuffer`, planning classifies by consumer topology
(`kind`) and Pass 3 applies one of two treatments accordingly:

| Case | Inside consumers | Outside consumers | `kind` | Treatment |
|---|---|---|---|---|
| 1 | ✓ | ✗ | `loop_internal` | Nothing to build — Pass 3 skips this op entirely |
| 2 | any | ✓ | `copy_out` | Pass 3 allocates a full HBM buffer and inserts a loop-tagged copy op that publishes each tile into the correct slice |

Every cross-loop-group write, regardless of inside-consumer topology or
whether any of the op's real inputs are themselves loop-internal, takes
the copy-op path (Case 2 in this table, "Case 1" in `coarse_tile.py`'s own
comments and debug logging — see the code-level-naming note below). There
is no longer a direct-mutation treatment: an earlier version of this pass
(deleted as part of the unconditional-copy change; see
`coarse_tile.py`'s git history around the deletion of
`_has_loop_internal_real_input`) rewired the tiled op itself to write
directly into the full buffer via `MutationLayoutSHOULDREMOVE` whenever it
had no inside consumers and no loop-internal real input. That treatment
had a genuine post-stickify safety gap: `_allocate_full_buffer`'s
post-stickify branch derives the full buffer's device layout by scaling
the tiled op's own already-committed output layout, without ever
consulting the op's *input* layouts, and — unlike the pre-stickify path,
which goes through `finalize_layouts`'s explicit
`is_elided`/`is_carry_into_accum` compatibility assert — there was no
check that an external input's own committed layout was actually
stick-compatible with the newly-derived, scaled-up full-buffer layout. An
incompatible case could silently miscompile rather than raise.

Splitting every cross-boundary write into two ops (the real op's own
tile-sized output, then a single-input copy into the full buffer) avoids
this by construction: the real op only ever has to satisfy its own
input-derived layout (the same problem the pass already solves correctly
for ordinary loop-internal ops with no outside consumers at all), and the
new copy op only ever has to satisfy the full buffer's derived layout
against its own single, freshly-fixed input — there is no second edge
whose compatibility can be silently skipped.

An earlier version of this same always-copy idea (predating the
loop-internal-input narrowing entirely, i.e. before commit `8ac03da`)
forced the copy-op path for *any* tiled op with more than one real input
(`_num_real_inputs(op) > 1`); that rule was itself narrowed because it
over-triggered for ops whose several inputs were all external (e.g. two
graph inputs), producing what was judged at the time to be an unnecessary
identity copy. That perf argument is superseded now that the inserted
copy is understood to be scratchpad-resident (LX planning targets exactly
this kind of small, tile-sized, loop-internal buffer) — and the prior
direct-mutation treatment had a secondary cost of its own, forcing the
real op's output out of scratchpad-reuse eligibility entirely (see
`_op_output_good_for_lx_reuse` in
`torch_spyre/_inductor/scratchpad/allocator.py`, which
unconditionally excludes `MutationLayoutSHOULDREMOVE` outputs). Under the
current always-copy rule, the real op's own output is never a mutation
layout, so it never loses scratchpad eligibility on that account.

**Note on code-level naming**: `coarse_tile.py`'s Pass 3 executor
`_propagate_tiled_op` (called from `_insert_all_write_copy_ops` once per
op planned `kind="copy_out"`) carries the operative comment describing a
single unconditional path: "Every cross-loop-group write always takes the
copy-op path: the real compute op keeps its own natural, input-derived,
tile-sized layout, and a separate copy op takes
`MutationLayoutSHOULDREMOVE(full_buf)`." The planning-time decision behind
this — whether an op is `copy_out` at all — is made once, up front, by
`_plan_tiling_propagation`; Pass 3 only executes it. The single operative
treatment corresponds to the doc's Case 2 row above.

**Revisited and re-deferred (2026-08-08): skipping the copy for
span-overflow.** The span-overflow coarse-tiling path
(`coarse_tile_span_overflow.py`) runs post-stickify and is intentionally
narrow in scope — at most one loop level, at most one reduction per loop —
which raised the question of whether that narrowness makes it safe to skip
the copy op in at least some cases, avoiding the HBM round-trip. Three
candidates were investigated and all three were rejected or deferred
without becoming implemented:

1. **Rely on `_resize_device_layout` preserving compatibility.** The
   full buffer's device layout is derived from the tiled op's own
   already-input-reconciled output layout by resizing it
   (`_resize_device_layout`/`_stick_host_dim`), so perhaps that derivation
   preserves enough structure to stay compatible with the op's inputs.
   Rejected: `_resize_device_layout` preserves dim order and which host dim
   is the stick, but *recomputes* `stride_map`/`device_size` for the new
   size, and compatibility (`stick_compatible`, reached via
   `device_coordinates`/`compute_coordinates`) depends on a size-dependent
   coordinate-derivation step, not just dim order. Resizing does not
   provably preserve compatibility.
2. **Skip only when the tiled op has zero external-to-group reads.** If
   every one of an op's reads is satisfied by sibling ops in the same
   coarse-tile group (no graph inputs, no other-group outputs), there is
   structurally nothing external to reconcile the full buffer's layout
   against, so direct-write would be safe. Confirmed this shape is
   non-vacuous in the grouping logic (`_plan_tiling_propagation` classifies
   `copy_out` purely by output/consumer criteria, independent of an op's own
   reads, and `span_overflow_groups`'s multi-op pointwise-run logic does not
   require every op in a run to read an external buffer) and is exercised by
   mocked-IR unit tests (`test_span_overflow_hint_analysis.py`'s
   `test_chained_compatible_pointwise_ops_produce_one_group` and
   `test_chained_pointwise_ops_conform_to_producer_split`). But no
   end-to-end test compiling a real torch program through genuine
   span-overflow triggering produces this shape — every real multi-op chain
   found (e.g. `abs(a+b)*c`) has its last op read an additional external
   tensor, and single-op programs are the norm. Deferred as unproven against
   real workloads, not rejected as impossible.
3. **Explicit per-edge compatibility re-check.** Reuse the same mechanism
   `finalize_layouts` uses (`edge.layout(in_stl, target_stl)` via
   `EdgeCostMap`) to re-derive `device_coordinates` for the full buffer's
   actual resized layout against each of the tiled op's input edges, after
   `_allocate_full_buffer` runs, and skip the copy only if every edge is
   still compatible. This is well-scoped and would live entirely on the
   post-stickify path. Deferred, not rejected — the most promising direction
   for a follow-up, should a real workload surface case 2's shape or
   otherwise justify the added complexity.

**Case 1** is where most of the working-set-reduction win comes from.  An
intermediate like `y` in the small example flows from one tiled op to
another without ever leaving scratchpad.  Planning routes such an op to
`kind="loop_internal"`, and Pass 3 leaves its `loop_info.output_tiled_dims`
empty at every level:

```python
if propagation.kind == "loop_internal":
    pass  # output_tiled_dims already [] from planning; nothing to build.
```

An empty `output_tiled_dims` is what `_general_tile_advance`
(`spyre_kernel.py`) reads when building the op's own `TensorArg`: it
substitutes `0` for every dim at every level, so no term contributes and
the resulting `device_tile_advance_expr` is `None`.  `generate_bundle`
(`codegen/bundle.py`) then skips emitting an `affine.apply` address for
that `TensorArg` (the base address is fixed across iterations);
`device_size` already matches the tile, so no update is needed either.

**Case 2**: the copy op carries the same `loop_info` (same `loop_group_id`,
`loop_count`, and `loop_tiled_dims`) as the original op, so the scheduler
wraps both in the same `CountedLoopSchedulerNode`.  The `tiled_symbols` / `affine.apply`
machinery computes the per-iteration slice offset automatically.  All
outside consumers are patched to read the full buffer.

Beyond closing the post-stickify safety gap described above, splitting the
crossing into two ops buys two more things for free, because the copy op is
a fresh, single-input edge that nothing else depends on yet:

- **It gives `propagate_layouts` a clean point to insert a restickify.**
  `propagate_mutation_layouts` runs specifically on ops carrying
  `MutationLayoutSHOULDREMOVE` (i.e. exactly the inserted copy) and assigns
  the real `FixedTiledLayout` for the full buffer at that point — including
  restickifying if the device layout the full buffer needs (to satisfy its
  own outside consumers, or the hardware's stick-alignment requirements)
  differs from the tile's own device layout. Because the real compute op's
  output layout is never touched by this step, the copy op is the only
  place that has to reconcile "what layout does the tile have" against
  "what layout does the full buffer need" — there is no other edge in the
  graph where that reconciliation could silently be skipped.
- **It normalizes the tensor into row-major format.** The copy op's read
  uses the original op's own index function, which may encode any number
  of view operations (transpose, permute, slice) accumulated on the way
  into the loop — but the copy op controls the *write* into the full
  buffer, and always writes it row-major. This means that once a value has
  passed through a coarse-tiling copy, every later consumer can rely on a
  known, canonical dimension order: whoever reads out of the tile next
  does not have to re-derive or guess the tile's dimension order from an
  arbitrary chain of upstream views, because the copy that produced the
  full buffer already fixed it.

**Which supertile?** Case 2's copy op needs "which supertile" recoverable at
codegen time, since Inductor's IR has no side channel for it.  The original
tiled op leaves its own `inner_fn` completely untouched (per the wrap-never-
reconstruct convention — see the [IR-rewiring
appendix](#appendix-how-ir-rewiring-works-and-why-its-sound)), so the op's
write index is still computed against its own tile-local `ranges`.  Which
tile of the full buffer this iteration is writing is a fact Inductor's IR
cannot represent at all; it would otherwise be discarded the moment
`_propagate_tiled_op` returns.  The fix is split into two stages, one at
planning time and one at codegen time.

**Stage 1 (decision, planning time):** this is `plan_coarse_tile_groups`'s
own, older planning step — distinct from `_plan_tiling_propagation`'s
`kind`/`ReductionPlan` decisions described under [Buffer
propagation](#buffer-propagation-planning-and-the-three-transformation-passes)
above, and computed earlier in `coarse_tile()`'s pipeline. `plan_coarse_tile_groups`
(`coarse_tile.py`) records, per dependency, a per-level *decision* — not a
substituted expression — on `CoarseTileInfo.tiled_dims_per_read` (one entry
per read dependency) and `CoarseTileInfo.output_tiled_dims` (for the
write): a `list[list[tuple[int, Expr]]]`, outermost level first, where each
inner list is the `(host_dim, extent)` pairs tiled by that level for that
dependency. A host dim can be tiled at more than one level when the tensor
doesn't have enough real dims to give each level a distinct one — the
canonical example is a flattened 1-D `[Lq * D]` tensor coarse-tiled by two
independent hints (an outer `Lq` loop and an inner `D` loop), both of which
necessarily tile host dim 0, since there is no second host dim to tile.
Because the decision is a list of per-level `(dim, extent)` pairs rather
than a flat dict keyed by host dim, both levels' facts survive side by
side — one entry per level, keyed by list position, not host dim — rather
than one silently overwriting the other.

**Stage 2 (substitution, codegen time):** `SpyreKernel._general_tile_advance`
(`spyre_kernel.py`) does the actual sympy substitution, once per
`TensorArg`, independently — not once per op. An op's non-output
`TensorArg`s (its inputs) can have device layouts/`dim_order` that diverge
from the op's own output/mutation-target buffer — broadcast, permute, a
different rank, or a layout explicitly forced by an earlier pass — and a
single value shared across every arg of the op cannot represent each arg's
true per-iteration device-memory advance when that happens. For each
nesting level with a nonempty `(dim, extent)` list, `_general_tile_advance`
mints a fresh, distinct `sympy.Symbol` for that `(op, level)` pair (via
`_get_or_mint_level_symbol`, named `_tile_adv_{op_name}_lvl{level_idx}` —
distinct from Inductor's own `d0`, `d1`, ... convention by construction),
substitutes `d_i -> extent * level_symbol` for each tiled host dim `d_i` at
that level into this dependency's own `dep.index` (re-derived fresh at this
call, since every pass that could rewrite it via `WrapperHandler` has
already run), and reprojects the resulting host-space term to
device-element space via `views.tiling_expr_to_device_expr`, using this
arg's own `device_size`/`stride_map`. Every level's device-space term is
summed into one combined `sympy.Expr` — the arg's own
`TensorArg.device_tile_advance_expr` — preserving the single-Expr-per-arg
contract the rest of the pipeline depends on. Minting one symbol per
`(op, level)`, rather than per real dim, is what lets two levels that
happen to tile the *same* host dim (the flattened-1D case above) keep
distinct, non-colliding terms in the summed expression: `_tile_adv_add_lvl0`
carries the outer level's contribution and `_tile_adv_add_lvl1` the inner
level's, and sympy keeps them as separate addends rather than collapsing
them, since they are different symbols.

`OpSpec.tiled_symbols` (a `list[list[Symbol]]`, innermost-first, populated
by `create_op_spec`) and `OpSpec.tiled_symbol_trip_counts` (mapping each
minted level symbol to that level's trip count) travel alongside
`device_tile_advance_expr` and are what let downstream consumers recover
"how many bytes does this level's step actually advance, and how many
steps does it take" without a separate stored extent field on `TensorArg`.

The two are consumed together in two places, for two different purposes,
both already iterating per arg:

- **`superdsc.py`'s `_create_sdsc_tensors`** uses each arg's own
  `device_tile_advance_expr`/`tiled_symbol_trip_counts` to establish only
  that arg's **iteration-0 base** stick-dimension stride/backGap/offset —
  narrowly scoped to the stick dim, and only for computing where the very
  first tile starts, not for the per-iteration advance across supertiles.
  This replaces a reverse-engineering step (deriving the same fact from
  `device_coordinates`) that silently reads the wrong slot when
  `_get_device_dim_order`'s coordinate walk happens to place the stick
  dimension differently for a copy-op output arg than for its sibling
  input args. `device_coordinates` cannot represent "which supertile" for
  a copy-op arg at all, so no downstream mechanism can correct a wrong
  compile-time base offset — this is why the override survives here even
  though the harder problem (below) is already per-arg by construction.
  For each minted level symbol present in `op_spec.tiled_symbols`,
  `_create_sdsc_tensors` reads `arg.device_tile_advance_expr.coeff(sym)` to
  get that level's per-iteration byte advance (`tile_size`), and
  `op_spec.tiled_symbol_trip_counts[sym]` for that level's trip count
  (`supertile_count`) — `supertile_count` is host/op-level loop-structure
  metadata, not device-layout-derived, so it is legitimately the same
  across every arg of the op even though `tile_size` is not.
- **`codegen/compute_ops.py`'s `generate_sdsc`** uses each tensor's own
  `device_tile_advance_expr` to build `affine_strides` — the actual
  per-iteration advance for each nesting level. This is the one place in
  the whole pipeline that already iterates per level and per tensor arg,
  so it reads `tensor.device_tile_advance_expr` directly rather than a
  value passed in from outside that loop. For a symbol tiled at multiple
  levels, `tensor.strides[sym]` (`SDSCArgs`'s ordinary per-dim stride, a
  single flat scalar) already coincides with the *innermost* overridden
  level's advance — `coarse_tile.py` divides op ranges down to the
  innermost tile before `create_op_spec` runs — but cannot also represent
  an outer level's larger advance. `generate_sdsc` uses
  `_tensor_tiled_by_symbol` (a coefficient-based helper: true iff `sym`
  contributes a nonzero term to `tensor.device_tile_advance_expr`) to
  detect symbols that appear at more than one level and, for those only,
  reads each level's coefficient directly via `.coeff(sym)` — no
  ratio-scaling arithmetic needed, since the expression already keeps each
  level's contribution as a distinct addend. A symbol tiled at just one
  level is left alone; `tensor.strides[sym]` is already exactly right
  there.

See
[`MutationLayoutSHOULDREMOVE`: the real contract](#mutationlayoutshouldremove-the-real-contract)
below for the general soundness argument.

**This mechanism only produces a nonzero `device_tile_advance_expr` for an
arg whose dependency actually has tiled `(dim, extent)` pairs recorded on
it** — for Case 2 (the copy-op path), `_insert_copy_op` builds its own,
separate `ComputedBuffer` (`coarse_tile_copy_*`) with its own
`MutationLayoutSHOULDREMOVE` layout; whether its `TensorArg`s end up with a
nonzero `device_tile_advance_expr` depends on whether `loop_info` on that
copy op itself records tiled dims for the relevant dependency. The
[Small Example](#small-example) above takes this Case 2 path for
`coarse_tile_copy_buf1`, and its `bundle.mlir` affine map is already
correct via the ordinary `tiled_symbols`/`affine.apply` machinery
described under Case 2, independent of `device_tile_advance_expr`.
Tests like `test_hint_nested_tiling_copy_mutation_correct`
(`tests/inductor/test_coarse_tile_e2e.py`) now exercise the copy-op path
with nested tiling where the copy op itself needs `device_tile_advance_expr`
for its base offset calculation.

**The same multi-level-shared-host-dim pattern also covers a flattened 1-D
`[Lq * D]` tensor**, tracked by `test_hint_nested_tiling_copy_mutation_flat`
(same file): both the outer `Lq` and inner `D` coarse-tiling hints land on
the same (only) host dim here, unlike the 2-D case where each hint owns a
distinct host dim. This is exactly the multi-level-shared-host-dim scenario
described above — it needs the multi-term, per-`(op, level)`-minted-symbol
shape of `device_tile_advance_expr` (one term per nesting level, each with
its own distinct symbol) rather than a single-entry-per-host-dim shape,
since the latter cannot distinguish the outer level's advance from the
inner level's for the same dim.

(read-side-adaptation-full-buffer-inputs-to-a-loop-internal-op)=

#### Read-side adaptation: full-buffer inputs to a loop-internal op

The write-side perimeter above is not the whole story. Pass 1
(`_insert_all_read_copy_ops`) checks, for every op, whether it directly
reads a full-size buffer produced by a cross-loop-group producer — either a
full-size `SpyreEmptyFallback` buffer (typically an accumulator that the
copy-out path above already promoted to full size; see "Reduction tiling,"
below, for the nested output-dim + inner-reduction-dim case that produces
this), or any other `ComputedBuffer` whose own `loop_group_id` outer key
differs from `op`'s (see `_full_buffer_read_deps`). A loop-internal op
cannot read such a buffer directly: its own candidate layouts are
tile-sized, and the full-size buffer has only one, full-size candidate
layout, so the two can never be made stick-compatible. Pass 1 recomputes
`_full_buffer_read_deps(op)` fresh, post-division, for every op — rather
than relying on any planning-time snapshot — precisely because a
cross-loop-group producer's full-size promotion may itself be a product of
this same pipeline (a nested reduction's `accum_full`, built by Pass 2, or
an earlier op's `copy_out` target, built by Pass 3 in a prior compilation
pass over another group) and so is only guaranteed current once
transformation has actually run. `_insert_read_copy_ops` always
materializes a tile-sized copy `ComputedBuffer` per such read, rewriting
`op`'s `inner_fn` (via a `WrapperHandler` subclass, per the
wrap-never-reconstruct convention) to read the copy instead of the full
buffer. This means the "no conversion, just addressing" claim above holds
only when the full buffer being read is a genuine graph input or other
host-side tensor — not when it is itself the product of an earlier
tile→full promotion inside the same compilation.

#### Reduction tiling: stick and non-stick reduction dims

When a `Reduction` op has a non-empty `loop_tiled_reduction_dims`
(i.e. the hint named a reduction dimension), planning routes it to
`kind="reduction"` and computes its `ReductionPlan` (identity, nesting,
full/per-tile shapes — see `loop_info.py`). Pass 2
(`_insert_all_reduction_ops`, which calls the per-op executor
`_propagate_tiled_reduction_op` for every op so planned) then builds the
actual buffers using a **fill-initialize + per-tile combine** pattern,
purely executing the plan's shape/identity/nesting decisions rather than
making any new ones. The exact buffer allocation depends on whether tiling
is flat (reduction dim only) or nested (outer output dim + inner reduction
dim):

**Flat (K-only) tiling** — a single `accum_full` HBM buffer is allocated.
The fill and combine ops both target `accum_full` directly.

1. **Allocate `accum_full`** with the full output shape (`data.ranges`,
   which is already the full output since only `reduction_ranges` was
   divided by the tiling pass).
2. **Insert a fill op** (outside the loop, no `loop_info`) that writes the
   reduction's identity value into `accum_full`.  The identity value is
   produced by a `SpyreConstantFallback` scalar with a manually assigned
   `FixedTiledLayout` (necessary because `finalize_layouts` has already run
   by the time this pass executes).
3. **Insert a combine op** (inside the loop, same `loop_info` as the tiled
   reduction op) that merges each tile's partial result into `accum_full`
   using the appropriate pointwise binary operator.
4. **Leave the tiled reduction op's own `output_tiled_dims` empty** — it is
   a per-tile scratch buffer whose base address does not advance between
   iterations.
5. **Patch outside consumers** to read `accum_full`.

**Nested (outer output dim + inner reduction dim) tiling** — two buffers
are allocated to enable LX scratchpad placement of the inner accumulator
(e.g. outer-B + inner-K for bmm/mm):

1. **Allocate `accum_full`** (full HBM output, shape matching the full
   output across all outer tiles).
2. **Allocate `accum_tile`** (per-tile scratch, same per-tile output shape)
   with an empty `output_tiled_dims`, so `_general_tile_advance` gives it no
   `device_tile_advance_expr` and `generate_bundle` never advances its base
   address; `scratchpad_planning` can therefore place it in LX scratchpad
   memory.
3. **Insert a fill op** (inside the outer loop, carrying the outer
   `loop_info`) that writes the identity value into `accum_tile` once per
   outer-loop tile.
4. **Insert a combine op** (inside the inner loop, same `loop_info` as the
   tiled reduction op) that merges each inner-tile partial result into
   `accum_tile`.
5. **Insert a `coarse_tile_reduce_copy` op** (inside the outer loop, after
   the inner loop) that copies `accum_tile → accum_full`.  It carries the
   outer `loop_info` so `generate_bundle` advances `accum_full`'s HBM
   address once per outer-loop tile.  The copy uses `MutationLayoutSHOULDREMOVE`
   so no extra allocation is created.
6. **Leave the tiled reduction op's own `output_tiled_dims` empty** (the
   inner scratch for the reduction kernel itself).
7. **Patch outside consumers** to read `accum_full`.

Identity values and combine operators by `reduction_type`:

| `reduction_type` | Identity | Combine |
|---|---|---|
| `sum` | 0 | `add` |
| `prod` | 1 | `mul` |
| `max` | −∞ (`-torch.inf`) | `maximum` |
| `min` | +∞ (`torch.inf`) | `minimum` |
| `xor_sum` | 0 | `bitwise_xor` |
| `any` | 0 | `logical_or` |

`argmin` and `argmax` do not have element-wise combine operators and raise
`RuntimeError` when a user attempts to tile them.

Before any transformation runs, `plan_coarse_tile_groups` calls
`_validate_planned_reduction_tiling(op, tiled_dims, tiled_rdims)` at planning
time — a pure function of already-known shape data — which raises
`Unsupported` (from `torch_spyre/_inductor/errors.py`, a `RuntimeError`
subclass) for configurations not yet implemented:

- **Mixed output+reduction at the same nesting level** — `loop_tiled_dims[i]`
  and `loop_tiled_reduction_dims[i]` are both non-empty for some level `i`.
- **Multiple reduction indices at one level** — `len(loop_tiled_reduction_dims[i]) > 1`.

Stick-dim reduction tiling is fully supported: tiling the innermost (stick)
dimension of the input (e.g. `x.sum(dim=-1)` on a `[B, D]` tensor where D
maps to the stick, or K-tiling for `BATCH_MATMUL_OP`) uses the same
fill-initialize + per-tile combine pattern.  The output accumulator for a
scalar stick-dim reduction has shape `data.ranges` (e.g. `[B]`) — the stick
dim has been collapsed — and `_resize_device_layout` handles this "stick
eliminated" case correctly.

Nested tiling where outer level(s) tile output dims and the innermost level
tiles a reduction dim (e.g. outer-B + inner-K for bmm) is fully supported
and handled by the two-buffer pattern described above.

The device layout for `accum_full`/`accum_tile`'s `MutationLayoutSHOULDREMOVE`
target is not chosen uniformly: `propagate_spyre_tensor_layouts`
(`propagate_layouts.py`) dispatches mutation-target layout computation three
ways depending on what kind of op is writing into it — `BATCH_MATMUL_OP`
reductions use `_matmul_layouts`, other `Reduction` ops use
`_single_arg_op_layout`, and plain `Pointwise` ops (including the fill and
combine ops this section describes) use `_multi_arg_pointwise_layouts`, the
same `AllSameNode` stick-compatibility path used for ordinary Case 2
copy-op routing above. A reduction accumulator write does not fit the broadcast
relationship `_multi_arg_pointwise_layouts` otherwise assumes, which is why
the `Reduction`-specific paths exist as separate cases rather than folding
into the pointwise one.

(sequential-recurrences-are-rejected-at-planning-time)=

#### Sequential recurrences are rejected at planning time

The fill/combine pattern above is a **monoid combine**: each tile's partial
result is independent and can be merged into the accumulator in any order.
Online-softmax-style kernels (flash-attention's running max and
rescale-accumulate denominator/output) need something structurally
different — a **true recurrence**, where the value one loop iteration
writes must be visible, unmodified, as the *next* iteration's input.
Re-running the traced Python's fill on every tile would silently reset the
running max/denominator each iteration instead of carrying it forward.
There is no execution mechanism for this fourth regime: an op that needs it
is rejected outright, before any IR mutation happens.

**Detection, not propagation.** `plan_coarse_tile_groups` (the planning
phase — see
[Public API: `coarse_tile()`](#public-api-coarse_tile)) calls
`_seed_buffer_for_carry` on every op that is loop-invariant at the group's
reduction-tiled level(s) (`_plan_is_loop_invariant_at_reduction_levels`
gates this call). If `_seed_buffer_for_carry` identifies `op` as the
carry-producing step of such a recurrence, planning raises `Unsupported`
immediately — `coarse_tile()` never reaches the transformation phase for
that group, and no buffers are allocated or rewired for the recurrence.
`_seed_buffer_for_carry` exists purely to answer "does this op need a
pattern we don't support," not to drive any propagation; there is no
`accum_tile`/`carry_prev` machinery, no copy-in/copy-out placement, and no
entry-op/terminal-op walk — those all belonged to the deleted execution
mechanism and have no replacement.

**How detection works.** The recurrence's pre-loop initializer (e.g.
`M = torch.full((...), -inf)` for a running max) is a constant fill —
`_is_constant_fill` recognizes it (a `Pointwise` wrapper around a
`SpyreConstantFallback` scalar, the lowering of `torch.full`/`torch.zeros`/
`torch.zeros_like`). Detecting which such fill is a carry seed (as opposed
to an ordinary hoisted constant) is closure-based, not op-local:

- **Closure.** `_seed_closure` returns every op in the same outer loop
  group that reads the seed *directly* — e.g. both
  `max_running = maximum(M, block_max)` and
  `correction = exp(M - max_running)` read `M` directly, so both are in
  `M`'s closure, even though only the first is the actual recurrence
  update. This is deliberately non-transitive: an op that reads a closure
  member but not the seed itself is an ordinary downstream consumer, not
  part of the closure.
- **The unique externally-fed member.** `_seed_buffer_for_carry` requires
  `op` to be the *unique* closure member whose non-seed operands are all
  external to the closure
  (`_closure_member_has_external_operands_only`) — the step that combines
  the previous carry value with fresh, per-iteration data, as opposed to a
  downstream step that only combines the seed with an already-computed
  sibling (e.g. `correction` reads `max_running`, a closure member, so it
  is excluded even though it also reads `M` directly). If zero or more than
  one closure member satisfies this, `_seed_buffer_for_carry` returns
  `None` (not a carry step) rather than guessing — a known, accepted
  limitation for closures with more than one externally-fed member, not
  hit by any current test.

If `_seed_buffer_for_carry` returns non-`None` for an op that is
loop-invariant at a reduction-tiled level, that op is the carry-producing
step of a recurrence this pass cannot execute, and planning raises
`Unsupported` for the whole group.

## Layer 2 — `CountedLoopSchedulerNode`

### Class definition

`CountedLoopSchedulerNode` lives in
`torch_spyre/_inductor/scheduler.py` alongside `SuperDSCScheduling`.
It subclasses Inductor's `FusedSchedulerNode`:

```python
class CountedLoopSchedulerNode(FusedSchedulerNode):
    loop_count: sympy.Expr

    def __init__(
        self,
        scheduler,
        snodes: list[BaseSchedulerNode],
        loop_count: sympy.Expr,
    ) -> None:
        super().__init__(scheduler, snodes)
        self.loop_count = loop_count

    def unpack(self) -> list[BaseSchedulerNode]:
        # CountedLoopSchedulerNode is an atomic codegen unit; do not unpack.
        return [self]

    @classmethod
    def can_fuse(
        cls,
        producer: BaseSchedulerNode,
        consumer: BaseSchedulerNode,
    ) -> bool:
        return False
```

`unpack()` returns `[self]` to prevent Inductor's
`Scheduler.process_grouped_nodes()` from dissolving the node back into its
constituent `SchedulerNode`s before codegen.  `can_fuse` returns `False`
— a loop group is atomic; nothing can be fused into it from outside.

### Why `FusedSchedulerNode` is the right base

`CountedLoopSchedulerNode` subclasses `FusedSchedulerNode` rather than
`GroupedSchedulerNode` for two reasons:

1. **Dispatch**: `Scheduler._codegen` only dispatches
   `FusedSchedulerNode | SchedulerNode` to `codegen_node()`.  A
   `GroupedSchedulerNode` subclass falls through to
   `assert isinstance(node, NopKernelSchedulerNode)` and crashes.

2. **Unpack control**: `GroupedSchedulerNode` is unconditionally unpacked
   by `Scheduler.process_grouped_nodes()` at the start of codegen.
   `FusedSchedulerNode` is not subject to that unpack, so overriding
   `unpack()` is sufficient to keep the node intact.

`FusedSchedulerNode` already merges `unmet_dependencies` across all
constituent nodes, exposes `get_nodes()`, and registers all constituent
names in `scheduler.name_to_fused_node`.  Nothing needs to be
reimplemented.

### Pre-fusion pass placement and ordering

`CountedLoopSchedulerNode`s are created by `build_loop_scheduler_nodes`,
which is registered as the **second pass in `CustomPreFusionPasses`** —
running before Inductor's own fusion pass:

```python
class CustomPreFusionPasses(CustomNodePassBase):
    def get_passes(self):
        return [propagate_mutation_layouts, build_loop_scheduler_nodes]

class CustomPostFusionPasses(CustomNodePassBase):
    def get_passes(self):
        return [hbm_pool_planning, spyre_fuse_nodes]
```

**`build_loop_scheduler_nodes` must run before Inductor's fusion pass and
before `spyre_fuse_nodes`.**  Placing it in `CustomPreFusionPasses` means
`CountedLoopSchedulerNode`s are already present when Inductor calls
`can_fuse_vertical` / `can_fuse_horizontal` on `SuperDSCScheduling`
(both return `False`), so loop groups are never split by Inductor's own
fusion logic.  `spyre_fuse_nodes` is additionally protected because it
only fuses plain `SchedulerNode`s — a `CountedLoopSchedulerNode` forces
a bundle boundary automatically.  `can_fuse = False` on
`CountedLoopSchedulerNode` provides a belt-and-suspenders guard against
any future fusion path that might otherwise merge across group boundaries.

### The grouping algorithm

`build_loop_scheduler_nodes` first calls `_regroup_by_outer_loop_key`, then
scans the resulting node list and groups contiguous runs sharing the same
outermost `loop_group_id` key. The regroup step is necessary because
Inductor's own `Scheduler.topological_sort_schedule` runs (twice) before
this pass ever sees the node list, via a plain DFS over
`unmet_dependencies` — that DFS only guarantees a *valid* topological
order, not that mutually independent nodes keep their original relative
order, so it can interleave unrelated nodes into the middle of what
`coarse_tile.py` built as a single contiguous loop group.
`_regroup_by_outer_loop_key` merges every node sharing an outermost
`loop_group_id[0]` key into one virtual unit (dependency set = the union of
its members' real cross-group dependencies), runs a dependency-respecting
DFS over `{merged units, ungrouped nodes}`, then expands each unit back
into its original members in their original relative order — restoring
contiguity while still producing a valid topological order:

```
nodes = _regroup_by_outer_loop_key(nodes)
result = []
i = 0
while i < len(nodes):
    node = nodes[i]
    gid = _loop_group_id(node)   # reads loop_info.loop_group_id from the inner ir.Operation
    if gid is None:
        result.append(node)
        i += 1
        continue
    outer_key = gid[0]
    run = [node]; i += 1
    while i < len(nodes) and _loop_group_id(nodes[i])[0] == outer_key:
        run.append(nodes[i]); i += 1
    # Recursively wrap deeper nesting within this run.
    inner = _build_loop_group(run, depth=1)
    result.append(CountedLoopSchedulerNode.create(inner, loop_count))
return result
```

Key invariant: the pre-scheduling pass runs in topological order, but
Inductor's own topological sort does **not** by itself guarantee that a
loop group's `SchedulerNode`s stay contiguous — it only guarantees a valid
order among mutually independent nodes, which can interleave. Contiguity
is restored by `_regroup_by_outer_loop_key` before grouping runs. If
`build_loop_scheduler_nodes` still finds a non-contiguous run after that
call, it means either a bug in `_regroup_by_outer_loop_key` itself, or a
genuine data-flow constraint that makes the group's own op sequence
topologically invalid (which would be a tiling-pass bug). The post-fusion
pass asserts contiguity.

## Layer 3 — `LoopSpec` and codegen

### `LoopSpec` and `OpSpec.tiled_symbols` in `op_spec.py`

```python
@dataclasses.dataclass
class LoopSpec:
    count: sympy.Expr
    body: list[OpSpec | UnimplementedOp | LoopSpec]

@dataclasses.dataclass
class OpSpec:
    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]
    tiled_symbols: list[list[Symbol]] = field(default_factory=list)
    tiled_symbol_trip_counts: dict[Symbol, int] = field(default_factory=dict)
    symbolic_dim_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    debug_handle: DebugHandle | None = None

@dataclasses.dataclass
class TensorArg:
    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: Any
    name: str | None = None
    device_tile_advance_expr: Expr | None = None
    element_arrangement: Any = None
```

`device_tile_advance_expr` is the sole tile-advance mechanism (see the
docstring in `op_spec.py`): `None` means the address does not advance
across loop iterations, replacing the older, separate `per_tile_fixed`
flag entirely.

`LoopSpec` is a peer of `OpSpec` and `UnimplementedOp` in the list that
`SpyreKernel.codegen_kernel()` serializes.  It is not a subclass of `OpSpec`
because it has no `iteration_space`, `args`, or `op_info` of its own — those
belong to the inner `OpSpec`s.

The `body` type is recursive: a `LoopSpec` body may itself contain
`LoopSpec` entries, representing nested counted loops.

`OpSpec.tiled_symbols` is a `list[list[Symbol]]` containing per-loop-level
iteration-space symbols, **innermost first**.  `tiled_symbols[0]` lists
the symbols tiled by the innermost enclosing loop; `tiled_symbols[1]`
lists those tiled by the next-outer loop; and so on.  It is **empty for
ops not inside a `LoopSpec`**.  Every enclosing loop level has an entry
(even if empty `[]`) so that level indices stay aligned with nesting
depth.  Two ops in the same loop group can have different `tiled_symbols`
if work division or stickification places the batch dimension at
different positions in each op's iteration space.

`OpSpec.symbolic_dim_bounds` maps a PyTorch symbol name (e.g. `"s97"`) to
`(max, granularity)` bounds for dynamic-shape dims; it is populated by
`compute_symbolic_bounds` during `create_op_spec` and empty for concrete
dims.

`OpSpec.tiled_symbol_trip_counts` maps each minted level symbol appearing
in `tiled_symbols` to that level's own trip count
(`CoarseTileInfo.loop_count` for that level); it lets downstream codegen
recover a level's full (untiled) extent as `(per-step device-element
advance) * trip_count` without a separately tracked extent field on
`TensorArg`. Only correct when a symbol belongs to exactly one nesting
level.

`TensorArg.device_tile_advance_expr` is a single `sympy.Expr | None`,
computed independently for **each** `TensorArg` (not shared across the
op's `args`) by `SpyreKernel._general_tile_advance` — see [Which supertile?](#treatment-by-consumer-topology) above for how each arg's own
expression is derived and consumed; it is `None` for any arg whose
dependency has no tiled `(dim, extent)` pairs recorded on it. Its free
symbols are the minted, per-`(op, level)` `_tile_adv_{op_name}_lvl{level}`
placeholders (distinct from Inductor's own iteration-space symbols in
`tiled_symbols` by construction), so each level's term is picked out by
`expr.coeff(sym)` on that level's own minted symbol, not by list position
the way `tiled_symbols` is. A given real host dim can be tiled at more
than one level (the flattened 1-D case above); because each level mints
its own distinct symbol, `device_tile_advance_expr` keeps each such
level's contribution as a separate addend rather than collapsing them.

The `bundle.py` and `compile_op_spec` paths reverse `tiled_symbols` to
outermost-first order and build per-level `affine.apply` stride maps,
mapping each level's strides to the correct loop variable by index.

### Nested loops and the `loop_group_id` tree

Each `ir.Operation` carries a `loop_info.loop_group_id` that is a **path**
rather than a flat integer.  A path is a tuple of integers, one element per
nesting level:

| `loop_group_id` | Meaning |
|---|---|
| `(0,)` | outermost loop group 0, not nested |
| `(0, 0)` | single op nested two levels deep inside group 0 |
| `(0, 1)` | ops at depth 2 inside outer group 0, inner group 1 |

`loop_info.loop_count` is a **list** parallel to the path.  For a flat op at
`(0,)`, `loop_count = [K]`.  For a single op at `(0, 0)`,
`loop_count = [K1, K2]` — the scheduler reads `loop_count[0] = K1` when
building the outer `CountedLoopSchedulerNode` and `loop_count[1] = K2`
when building the inner one.  This allows a single op to supply the counts
for all its enclosing loops without requiring sibling ops at intermediate
depths.

The post-fusion pass (`_build_loop_group`) reconstructs the tree
recursively:

1. Group the flat `SchedulerNode` list into runs that share the same
   outermost group id element (index `depth`).
2. Read the count for this depth from `_loop_count(node, depth)`, which
   indexes `loop_info.loop_count[depth - base_depth]`.  All nodes in the run
   must agree on this count.
3. Recursively call `_build_loop_group(run, depth + 1)` to build the
   inner level.
4. Wrap the result in a `CountedLoopSchedulerNode(count=K_outer, ...)`.

Because every op carries the full `loop_count` list, the algorithm works
even when a run contains only a single op that spans all nesting levels —
there is no need for placeholder ops at intermediate depths.

### Bundle boundary constraint

A `CountedLoopSchedulerNode` (at any nesting depth) and all its
descendant `SchedulerNode`s must be codegen'd into a **single SuperDSC
bundle** — i.e., a single `codegen_node()` call must produce the entire
`LoopSpec` tree.  This is automatically satisfied because Inductor calls
`codegen_node()` once per `BaseSchedulerNode` in the topological order,
and a `CountedLoopSchedulerNode` is a single node that encapsulates all
its children.  No loop group can be split across two `codegen_node()`
calls.

The bundle boundary constraint also forbids a loop group from being split
by Inductor fusion: `can_fuse` returns `False` on
`CountedLoopSchedulerNode`, so no external node can be merged into or
absorb part of a loop group.

In `bundle.py`, `generate_bundle` iterates the flat `list[OpSpec]`
emitted by `codegen_kernel()`.  When it encounters a `LoopSpec` it
emits SDSC JSON files for each `OpSpec` in the body (recursively) and
wraps those executions in an `scf.for` in `bundle.mlir`.

### Changes to `SuperDSCScheduling.codegen_node()`

`codegen_node` already handles `FusedSchedulerNode | SchedulerNode`.
`CountedLoopSchedulerNode` is recognized by an `isinstance` check:

```python
def codegen_node(
    self,
    node: Union[FusedSchedulerNode, SchedulerNode, CountedLoopSchedulerNode],
) -> None:
    if isinstance(node, CountedLoopSchedulerNode):
        self._codegen_counted_loop(node)
        return
    # existing flat-list path unchanged
    ...

def _codegen_counted_loop(self, node: CountedLoopSchedulerNode) -> None:
    inner_nodes = [
        n for n in node.get_nodes()
        if n.get_name() not in self.scheduler.removed_ops
    ]
    kernel = SpyreKernel()
    all_schedule_nodes = []
    with kernel:
        for inner in inner_nodes:
            if isinstance(inner, CountedLoopSchedulerNode):
                self._codegen_loop_body(inner, kernel, all_schedule_nodes)
            else:
                sched = self.generate_node_schedule([inner])
                all_schedule_nodes.extend(sched)
                for snode in sched:
                    var_ranges = iteration_space(snode)
                    vs = list(var_ranges.keys())
                    index_vars = [vs[:len(snode._body.iter_vars)],
                                  vs[len(snode._body.iter_vars):]]
                    snode.codegen(index_vars)

    # Compute tiled symbols for depth 0 from any leaf SchedulerNode.
    outer_tiled_syms = []
    for inner in inner_nodes:
        ref = _find_leaf_sched_node(inner)
        if ref is not None:
            outer_tiled_syms = _tiled_syms_for_sched_node_at_depth(ref, 0)
            break

    # Wrap the collected inner specs in a LoopSpec
    kernel.wrap_op_specs_in_loop(node.loop_count)

    with V.set_kernel_handler(kernel):
        src_code = kernel.codegen_kernel()
    kernel_name = self.define_kernel(src_code, all_schedule_nodes, kernel)
    ...
```

`_codegen_loop_body` handles nested `CountedLoopSchedulerNode`s: it
codegens the body ops into the existing kernel, then wraps only the newly
added `op_specs` entries in an inner `LoopSpec`.  The outer
`_codegen_counted_loop` then wraps everything in the outer `LoopSpec` via
`wrap_op_specs_in_loop`.

`SpyreKernel.wrap_op_specs_in_loop(count)` replaces the flat `self.op_specs`
list with `[LoopSpec(count=count, body=self.op_specs)]`.

`generate_node_schedule` handles `FusedSchedulerNode`s that may appear
among the inner nodes (e.g. from earlier passes that fused nodes within
the same loop group) by flattening them into their constituent
`SchedulerNode`s.

### Serialization in `codegen_kernel()`

`codegen_kernel()` already iterates `self.op_specs` to emit Python source.
A `LoopSpec` entry is serialized as:

```python
LoopSpec(
    count=sympify('K'),
    body=[
        OpSpec(
            ...,
            tiled_symbols=[[sympify('c0')]],   # one level: innermost
        ),
        LoopSpec(          # nested loop
            count=sympify('J'),
            body=[
                OpSpec(..., tiled_symbols=[[sympify('c1')], [sympify('c0')]]),
                # tiled_symbols[0] = innermost loop symbols
                # tiled_symbols[1] = outer loop symbols
            ],
        ),
    ],
)
```

`OpSpec.tiled_symbols` is populated by `SpyreKernel.create_op_spec`: it
reads `loop_info.loop_tiled_dims` (a `list[list[int]]`) from the
`ir.Operation` (stamped by `coarse_tile()`), and for each loop level
selects the symbols at those indices from the scheduler-level
`iteration_space` dict.  The result is stored innermost-first.
`MemoryDep.ranges` preserves the `data.ranges` ordering, so this positional
correspondence is stable across the pre-scheduling to codegen boundary.

For reduction-dim tiling, `create_op_spec` also consults
`loop_info.loop_tiled_reduction_dims`.  For a `Reduction` op,
`iteration_space()` returns `reads.ranges`, which has output-dim symbols
first and reduction-dim symbols last.  `create_op_spec` finds the split
point as `n_output_syms = len(write_dep.ranges)` (the number of symbols in
the write dep's ranges), then appends `it_space_keys[n_output_syms + r]` for
each index `r` in the flattened `loop_tiled_reduction_dims`.  Without this,
`tiled_syms` would be empty for reduction-dim tiling (since
`loop_tiled_dims` is `[[]]`) and the runtime would not advance the input
tensor pointer between tiles, producing incorrect results.

`tiled_symbols` is omitted from the serialized source when empty (i.e. for ops
or loop specs where no dimension is tiled), keeping the generated output
identical to the pre-tiling baseline for non-tiled kernels.

The generated Python wrapper imports `LoopSpec` from `op_spec.py` so the
serialized source is re-loadable from the Inductor cache.

The `arg_index` fixup loop (which maps tensor names to kernel argument
positions) runs before serialization.  It must walk the `LoopSpec` tree
recursively to find all `TensorArg` objects inside nested bodies, not
just the top-level `self.op_specs` list.

### `bundle.mlir` generation for loops

`generate_bundle` in `bundle.py` emits one
`sdscbundle.sdsc_execute` line per `OpSpec`.  When a `LoopSpec` is
present it emits an `scf.for` block in `bundle.mlir` wrapping the
execute calls for the body ops.

The loop induction variable is an `index` type running from `0` to
`count` with step `1`.  For the current prototype, `count` must be a
concrete integer; symbolic loop counts raise `NotImplementedError`.

Emitted MLIR for a single-level loop with one body op:

```none
module {
  func.func @sdsc_bundle() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop_bound_0 = arith.constant 4 : index
    scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
      sdscbundle.sdsc_execute () {sdsc_filename="sdsc_a_0.json"}
    }
    return
  }
}
```

For nested loops, `scf.for` blocks are nested and induction variables are
numbered sequentially (`%i_0`, `%i_1`, ...):

```none
%loop_bound_0 = arith.constant 4 : index
%loop_bound_1 = arith.constant 8 : index
scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
  sdscbundle.sdsc_execute () {sdsc_filename="sdsc_a_0.json"}
  scf.for %i_1 = %c0 to %loop_bound_1 step %c1 {
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_a_1.json"}
  }
}
```

`generate_bundle` walks the `list[OpSpec | LoopSpec]` recursively,
maintaining an indentation level and a counter for SDSC JSON filenames.
The filenames are assigned in depth-first traversal order.

### Loop codegen: `scf.for` with late-bound addresses

Once the loop has reached `LoopSpec` form, `generate_bundle` in
`codegen/bundle.py` emits the loop intact — an `scf.for` wrapping, for each
tiled tensor, an `affine.apply` that computes the per-iteration HBM address
from the loop induction variable(s), followed by `sdsc_execute`, as shown in
the bundle.mlir section above.  `device_size` stays at the per-tile shape and
`tiled_symbols` records which iteration-space symbols the enclosing loop
levels advance; tensors whose `device_tile_advance_expr` is `None` (e.g. LX
scratchpad operands with an empty `output_tiled_dims`, see below) are
skipped entirely — no `affine.apply` is emitted for them since their base
address never changes across iterations.

This is the only loop-codegen path: nothing upstream of `generate_bundle`
branches on it, and there is no separate frontend loop-flattening step. An
earlier prototype ("unrolling") that expanded each `LoopSpec(K, body)` into K
flat copies of `body` with addresses baked into each `sdsc_*.json` has been
removed now that the backend symbol-table support this path relies on has
landed.

## Key files

| File | Role |
|---|---|
| `torch_spyre/_inductor/loop_info.py` | Layer 1: `CoarseTileInfo` dataclass; `copy_op_metadata` |
| `torch_spyre/_inductor/wsr/coarse_tile_hints.py` | `reorder_unhinted_interlopers()` reorders interlopers before grouping |
| `torch_spyre/_inductor/wsr/coarse_tile.py` | Layer 1: `coarse_tile()` stamps `loop_info` and rewrites ranges; `_plan_tiling_propagation` plus the `_insert_all_read_copy_ops`/`_insert_all_reduction_ops`/`_insert_all_write_copy_ops` passes handle the data perimeter |
| `torch_spyre/_inductor/insert_restickify.py` | `finalize_layouts` commits each op's chosen `FixedTiledLayout` and, for a tiled-reduction op, propagates that layout onto `accum_full` so fill/combine/copy all agree on device coordinates; also stamps a restickify node's `loop_info` from the op it feeds so the node lands in the same loop group |
| `torch_spyre/_inductor/scheduler.py` | Layer 2: `CountedLoopSchedulerNode`, `build_loop_scheduler_nodes`, `_codegen_counted_loop`, `_regroup_by_outer_loop_key` |
| `torch_spyre/_inductor/op_spec.py` | Layer 3: `LoopSpec` and `OpSpec` dataclasses |
| `torch_spyre/_inductor/spyre_kernel.py` | Layer 3: serializes `LoopSpec` tree in `codegen_kernel()`; `wrap_op_specs_in_loop()` |
| `torch_spyre/_inductor/codegen/bundle.py` | Layer 3: emits `scf.for` wrapping `affine.apply`/`sdsc_execute` in `bundle.mlir` |
| `torch_spyre/_inductor/passes.py` | Wires all passes into `CustomPreSchedulingPasses` and `CustomPreFusionPasses` |
| `torch_spyre/_inductor/propagate_hints.py` | `spyre_hint()` context manager; `DimHint`; hint collection/recovery across AOT re-tracing |
| `torch_spyre/_inductor/wsr/propagate_named_dims.py` | `propagate_named_dims()` and `assign_dim_hints()`: attach `dim_hints` to `ir.Operation` objects |
| `torch_spyre/_inductor/wsr/coarse_tile_hints.py` | `hints_to_coarse_tile_groups()`: converts `dim_hints` into `coarse_tile()` group tuples |
| `torch_spyre/_inductor/wsr/coarse_tile.py` | `coarse_tile()` entry point |
| `tests/inductor/test_coarse_tiling.py` | Unit tests: IR pass, propagation, scheduler node, bundle MLIR output |
| `tests/inductor/test_coarse_tile_e2e.py` | End-to-end compilation tests |

## Invariants and failure modes

**Pre-grouping contiguity** (`reorder_unhinted_interlopers`): before
`hints_to_coarse_tile_groups` runs, every unhinted `ComputedBuffer` that
sits between two same-hint ops is moved to just before or just after the
run.  If a data-flow dependency prevents both directions, a `RuntimeError`
is raised.  This ensures that all same-hint ops are contiguous in
`graph.operations` before grouping begins.

**Contiguity invariant**: all `SchedulerNode`s sharing a
`loop_info.loop_group_id` must be contiguous after the scheduler's
topological sort.  `_apply_plan` enforces this during transformation via
`_validate_contiguous`, which raises `RuntimeError` if the ops are not
a contiguous slice of the operation list.  The post-fusion pass
(`build_loop_scheduler_nodes`) also asserts this by processing a contiguous
run — a non-contiguous run indicates a bug in the tiling pass.

**Consistent `loop_count`**: all ops sharing a `loop_group_id` must agree on
`loop_info.loop_count` at every depth level.  The post-fusion pass asserts
this.

**`tiled_symbols` populated iff inside a loop**: `OpSpec.tiled_symbols` is
non-empty exactly when the op was codegen'd inside a `CountedLoopSchedulerNode`.
It is a `list[list[Symbol]]` (innermost first) derived from the per-level
tiled dims in `loop_info.loop_tiled_dims` on the corresponding
`ir.Operation`, selected from the scheduler-level `iteration_space` keys.

**Pass ordering**: coarse tiling must run after stickify/padding and
before `span_reduction`, `cost_model_matmul_division`, `work_distribution`,
and `scratchpad_planning`.  `build_loop_scheduler_nodes` must run in
`CustomPreFusionPasses` (before Inductor's own fusion pass and before
`spyre_fuse_nodes`) — see the ordering rationale above.

**Cache invalidation**: `coarse_tile.py`, `scratchpad_planning`, and all
other pass source files are included in `CustomPreSchedulingPasses.uuid()`
so the Inductor FX cache is invalidated when any pass changes.

## Rejected design alternatives

### Inductor's existing loop IR

Inductor has several loop-related constructs, none of which fit the
requirement.

**`ir.Loops` / `Pointwise` / `Reduction`** (`torch/_inductor/ir.py`).
These have a `ranges: Sequence[Expr]` field that describes the iteration
space of a *single* operation.  They model per-op loop bounds, not a loop
that groups multiple operations together.  There is no concept of "execute
this sequence of ops N times."

**`ir.WhileLoop`** (`torch/_inductor/ir.py`).  A while-loop IR node for
data-dependent control flow.  Trip count is not statically known; not
appropriate for the counted, coarse-tiling use case.

**`GroupedSchedulerNode`** (`torch/_inductor/scheduler.py`).  Groups a
sequence of `SchedulerNode`s so the scheduler cannot interleave other
nodes between them.  This is a pure scheduling constraint: it carries no
loop count, does not rewrite iteration spaces, and is **unconditionally
unpacked** by `Scheduler.process_grouped_nodes()` before codegen.  It also
does not appear in the `FusedSchedulerNode | SchedulerNode` isinstance
check in `Scheduler._codegen`, so a subclass of `GroupedSchedulerNode`
would not be dispatched to `codegen_node()` at all.  These limitations
make `FusedSchedulerNode` the correct base instead.

**`codegen.cpp.LoopLevel` / `LoopNest`** (`torch/_inductor/codegen/cpp.py`).
Codegen-time loop structures used by the C++ backend to emit nested
`for` loops.  They exist only during C++ code emission and have no
presence in the scheduler or IR layers where Spyre's optimization passes
run.

### Helion's `ForLoopGraphInfo`

Helion (`helion/_compiler/device_ir.py`) represents loops as
`ForLoopGraphInfo` nodes.  Each node wraps a nested FX sub-graph
(referenced by `graph_id`) and a `block_ids` list that determines which
tile dimensions participate in the loop.  The FX graph for the outer
scope contains a `_for_loop(graph_id, begin, end, args)` node
(`helion/language/_tracing_ops.py`) as a placeholder.  A companion
`ReductionLoopGraphInfo` handles reduction loops.

This design is well-suited to Helion's tile-strategy-driven GPU
compilation model, where the loop structure is discovered during tracing
and the body is a reusable sub-graph.  It is a poor fit for Spyre's
pipeline for three reasons:

1. **Wrong representation layer.**  Spyre's optimization passes operate
   on `list[ir.Operation]` before the Inductor `Scheduler` exists.
   Helion's loop nodes live in an FX graph; adopting that representation
   would require building and maintaining a parallel FX graph for the
   pre-scheduling IR, adding substantial complexity.

2. **Tile strategy coupling.**  `ForLoopGraphInfo` carries `block_ids`
   that reference Helion's tile strategy objects.  Spyre has no tile
   strategy layer; loop structure comes from the coarse-tiling pass
   decision, not from a tiling configuration object.

3. **Sub-graph identity vs. flat sequence.**  Helion identifies loop
   bodies by an opaque `graph_id` and looks them up in a registry.  For
   Spyre's use case — a contiguous run of `SchedulerNode`s that must stay
   together — a flat ordered list inside `CountedLoopSchedulerNode` is
   simpler and directly matches what `codegen_node` already iterates.

The key insight borrowed from Helion is that the loop body should be a
*separate, named structure* rather than an attribute on individual ops.
That insight shaped the decision to make `CountedLoopSchedulerNode` a
first-class scheduler node (rather than stamping a loop-count attribute
on each `SchedulerNode` and reconstructing the grouping at codegen time).

### Attribute-only approach (Option B)

An earlier candidate design stamped `loop_group_id` and `loop_count`
directly onto `ir.Operation` objects and deferred all grouping to
`codegen_node()`, which would scan the flat `node_schedule` list and
reconstruct loop boundaries at codegen time.

This was rejected because it is fragile in the face of correctness
requirements.  If the scheduler ever reorders nodes within what the
tiling pass intended to be a loop group — or if a group boundary does
not align perfectly with a fused-node boundary — the reconstruction in
`codegen_node()` silently produces wrong output: incorrect trip counts or
mismatched iteration spaces.  With coarse tiling these are correctness
bugs, not performance bugs.  `CountedLoopSchedulerNode` enforces the
grouping structurally: the scheduler cannot split or reorder within it,
and a mismatch is caught at post-fusion pass time rather than silently at
codegen time.

## Out of scope

- Loops whose trip count is data-dependent (use `ir.WhileLoop` for that).
- Fusing a non-tiled op into the body of a `CountedLoopSchedulerNode`.
- Passing the loop induction variable into an `OpSpec` body (ops inside a
  loop do not currently use the induction variable; each iteration executes
  identically on a different slice of the data determined by the reduced
  iteration space).
- Symbolic loop counts in `bundle.mlir` (currently raises
  `NotImplementedError`; requires runtime shape plumbing into the MLIR
  function signature).

## Appendix: How IR rewiring works, and why it's sound

The sections above describe *what* `coarse_tile.py` does semantically: which
buffers get promoted to full size, which get an `identity` copy op, which
get an empty `output_tiled_dims` and stay loop-internal scratch. This
appendix describes *how* those
outcomes are implemented as edits to live Inductor IR objects, and why those
edits cannot violate Inductor's own scheduler and dependency-tracking
invariants. It is written for developers who need to modify `coarse_tile.py`
itself or diagnose a wrong-code bug that might originate there — not a
restatement of the Case 1/2/3 classification, the reduction accum pattern, or
the carry seed/closure detection vocabulary, all covered above.

### The wrap-never-reconstruct convention in practice

CLAUDE.md states the rule plainly: *"Modifying `ComputedBuffer.inner_fn`:
wrap, never reconstruct. Use a `WrapperHandler` subclass ... installed with
`V.set_ops_handler(handler)` inside the original `inner_fn`."* The reason is
that `inner_fn` closes over symbolic index expressions computed against a
specific `ranges`/`reduction_ranges`; those expressions go stale the moment
anything about the op's shape changes, so hand-rebuilding them from scratch
is a silent wrong-code trap (issue #2797, cited directly in
`replace_computed_buffer_body`'s implementation comment in
`pass_utils.py`). Every rewrite site in `coarse_tile.py` and
`insert_restickify.py` follows the same four-line idiom instead:

```python
orig_inner = op.data.inner_fn

def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
    with V.set_ops_handler(SomeWrapperHandlerSubclass(V.ops, _map)):
        return _orig_inner(*args)

object.__setattr__(op.data, "inner_fn", new_inner_fn)
new_op = replace_computed_buffer_body(op, op.data, operations)
```

`object.__setattr__` is required here because `ir.Loops` (the base of
`Pointwise` and `Reduction`, which holds `inner_fn` and `ranges`) is declared
`@ir_dataclass(frozen=True)` — a plain `data.inner_fn = new_inner_fn` raises
`FrozenInstanceError`. This is the same escape hatch the doc already uses for
`_divide_ranges`'s `object.__setattr__(data, "ranges", ranges)` above. By
contrast, `Buffer` (the base of `ComputedBuffer`, which holds `.layout`) is
**not** frozen, so the `op.layout = MutationLayoutSHOULDREMOVE(...)`
assignments used elsewhere in this appendix are ordinary attribute sets, not
escape-hatch writes — the two mechanisms look similar but rest on different
class-level decisions.

`replace_computed_buffer_body` (in `pass_utils.py`) is the second half
of the idiom: because `ComputedBuffer` itself is also frozen, the mutated
`data` cannot simply be re-attached to the existing `op` object either — a
fresh `ComputedBuffer` is constructed with the new `data`, all metadata
fields downstream passes depend on (`operation_name`, `origins`,
`origin_node`, `_split_size`/`_original_*`) are copied across, the
`get_default_sizes_body` cache is explicitly cleared on the new object, and
the new buffer replaces the old one in `operations` by index. Every inner_fn
rewrite site ends with this call, not a raw dataclass mutation, specifically
so that stale per-object caches on the old buffer can never leak forward.

Call sites, all following this exact shape:

- `_insert_read_copy_ops` (in `coarse_tile.py`, with a local
  `_NameSwapHandler` defined just above it in the same file) —
  see
  [Read-side adaptation](#read-side-adaptation-full-buffer-inputs-to-a-loop-internal-op)
  above; detailed further below.
- `_patch_consumers` (in `coarse_tile.py`, `NameSwapHandler` imported
  from `insert_restickify.py`) — patches an outside consumer's `inner_fn` to
  read the newly-promoted full buffer instead of the original tile-sized one.
- `_patch_retiled_load_indexes` / `_RetileLoadIndexHandler`
  (both in `coarse_tile.py`) — a distinct
  mechanism from name-swapping, detailed in the next subsection.
- `insert_restickify_on_node_inputs` (in `insert_restickify.py`, using
  the canonical `NameSwapHandler` defined in the same file) —
  the example CLAUDE.md itself points to.

One site looks like an exception but is not: `_insert_copy_op`
(in `coarse_tile.py`) builds a **new** `Pointwise` via
`tiled_op.make_loader()` rather than editing `tiled_op`'s own `inner_fn`.
This is IR-safe by construction, not a violation of the convention — it
reuses Inductor's own `make_loader()` (which itself returns a closure over
the *existing* `inner_fn`/index machinery) instead of hand-assembling an
index expression, so the same "never reconstruct a stale index" property
holds even though no `WrapperHandler` is involved.

No site in either file reconstructs an index expression from scratch.
`_divide_ranges` (in `coarse_tile.py`) is the one place shape and
layout are mutated (via `object.__setattr__`) with `inner_fn` left completely
untouched — deliberately, and safely, for the reason given in the next
subsection.

### Index-expression remapping: `_divide_ranges` and `_patch_retiled_load_indexes`

Two distinct mechanisms handle index-expression correctness after tiling,
and they are staged deliberately rather than combined:

1. **`_divide_ranges`** (in `coarse_tile.py`) shrinks `data.ranges`
   (and the op's own `layout.size`/`layout.stride`) via `object.__setattr__`,
   leaving `inner_fn` completely untouched. This is correct because the op's
   own index arithmetic is expressed in terms of the loop variables that the
   surrounding (now smaller, per-tile) iteration space binds — the *op*
   never needs to know it was tiled; only its bounds shrink. `_divide_ranges`
   only ever runs from `_apply_plan` (the transformation phase); planning
   itself never mutates `data.ranges` — it computes what the post-mutation
   extents *would be* analytically via `_planned_tile_extents`, reading the
   still-untouched `data.ranges`/`data.reduction_ranges`.

2. **`_patch_retiled_load_indexes`** fixes a different problem: *other* ops
   whose captured load index still carries the pre-tiling stride
   coefficient for a buffer that has since been re-tiled. This is driven
   exactly once, at the very end of `coarse_tile()`, after every group in
   the call has been processed — not per-group. `_stride_rewrite_map`
   (in `coarse_tile.py`) builds the substitution from old to new
   stride coefficients; `_retile_load_index_from_strides`
   (in `coarse_tile.py`) checks that the load index is affine and
   separable in the rewritten variables before substituting, and — this is
   a real, flagged soft spot rather than a proven bug — conservatively
   *refuses and warns* rather than raising a hard compile error if a future
   index shape is not affine-separable. A refusal here degrades to a
   runtime warning plus likely-wrong output, not a caught error at compile
   time. `_RetileLoadIndexHandler` (in `coarse_tile.py`,
   a `WrapperHandler` subclass) is the mechanism that actually applies the
   substitution to the consumer's `inner_fn`, following the same
   wrap-never-reconstruct idiom as every other site in this appendix.

   The concrete case is exactly the Small Example above: before Pass 3's
   copy-out path is inserted for `buf1`, `buf1`'s
   captured load of `y` is `i1 + 4096*i0` — a coefficient computed against
   the *pre-tiling* full row stride (4096). Once `y`'s producer is tiled down
   to a `[512, 1024]` per-tile buffer, that captured `4096*i0` coefficient no
   longer matches `y`'s actual (now much smaller) tile layout, and
   `_patch_retiled_load_indexes` rewrites it to the coefficient consistent
   with the per-tile shape — the same information the `bundle.mlir` section's
   `affine_map<(d0, d1)[s0] -> (s0 + 4194304*d0 + 2048*d1)>` encodes at the
   byte-stride level for the final, fully-tiled program.

   Running the patch once, globally, after all groups are stamped (rather
   than per-group, immediately after each group is processed) is not a
   stylistic choice: the project's own test history found and fixed a
   double-application bug that resulted from patching too early, where a
   load index already rewritten by an earlier group's pass got rewritten a
   second time by a later group's pass touching an overlapping buffer.

### Read redirection: why a view buffer, not just an index edit

A recurring temptation when redirecting a read is to think of it as "leave
`inner_fn` alone, just edit the dependency the scheduler sees." That is not
possible in Inductor: as the next subsection proves in detail, dependency
information is *derived from* `inner_fn` by re-tracing it, not stored
independently — so the only way to actually redirect what an op reads is to
change what its `inner_fn` does when traced.

`_insert_read_copy_ops` (in `coarse_tile.py`) is the concrete instance
already introduced under
[Read-side adaptation](#read-side-adaptation-full-buffer-inputs-to-a-loop-internal-op)
above: when a loop-internal op reads a full-size `SpyreEmptyFallback` buffer
directly (typically an accumulator that an earlier Case-2/mutation rewrite
already promoted to full size), the two-step mechanism is (1) insert, before
the tiled op, a small tile-sized copy `ComputedBuffer` whose `inner_fn` loads
the full buffer's current tile slice using the *same* index expression the
tiled op already computes and the *same* `loop_info` (so the per-iteration
base address advances identically to the tiled op's own reads); then (2) wrap
the tiled op's own `inner_fn` with the local `_NameSwapHandler` so that its
load of the full buffer's name is retargeted to the new copy buffer's name
instead. This always materializes an actual copy — there is no conditional
path that instead installs a zero-copy "view" over the full buffer; every
call constructs a real `Pointwise`/`ComputedBuffer` that Inductor's own
scheduler treats as an ordinary tile-sized producer. The copy's own layout is
built from the full buffer's per-variable strides (extracted from the read
dependency's index, which is affine in its var_names) rather than fresh
contiguous strides, specifically so the tiled op's *unmodified* read index
still resolves correctly once `_NameSwapHandler` retargets only the buffer
name, not the index expression itself.

The reason a copy is needed at all, rather than simply changing which name
the tiled op loads from, is the same `AllSameNode` stick-compatibility
constraint that motivates the Case 1/2 split on the write side: a full-size
buffer has exactly one candidate layout (sized to the full buffer), while the
tiled op's own candidate layouts are all tile-sized — the two can never be
made stick-compatible without an intermediate buffer sized to match.

### `MutationLayoutSHOULDREMOVE`: the real contract

The doc above uses `MutationLayoutSHOULDREMOVE` several times (the copy-op
output in Case 2, and both the flat and nested reduction accum patterns) as
an already-understood primitive, each time asserting it is "a metadata
redirect, zero added data movement." This subsection explains why that claim
is true, from the actual upstream implementation (`torch/_inductor/ir.py:4373-4459`):

```python
class MutationLayoutSHOULDREMOVE(Layout):
    def __init__(self, target: IRNode) -> None:
        super().__init__(
            target.get_device_or_error(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)
```

Constructing one of these immediately calls `V.graph.mark_buffer_mutated`
on the target buffer's name — mutation is registered at construction time,
unconditionally, not lazily discovered later. `get_buffer()` recursively
unwraps through `MutationLayoutSHOULDREMOVE` → `BaseView` → `MutableBox`
chains to find the real underlying `Buffer`, and `real_layout()` always
defers to *that* buffer's own actual layout:

```python
    def real_layout(self) -> Layout:
        layout = self.get_buffer().layout
        assert isinstance(layout, Layout)
        return layout
```

This is what "metadata redirect, zero added data movement" concretely means:
the mutating op's `.layout.stride`/`.storage_size()` are computed by
deferring to the target's real layout, not by allocating or copying
anything. (`realize_into()`, the classmethod defined alongside it, is
Inductor's own factory for the common "materialize a copy into an existing
buffer" pattern; torch-spyre does not call it — every call site below
constructs `MutationLayoutSHOULDREMOVE` directly and assigns it to `.layout`.)

Marking the mutation matters beyond bookkeeping:
`ComputedBuffer.make_loader()` checks `self.name not in
V.graph.mutated_buffers` before deciding it is safe to inline a buffer's
computation into its consumer. Mutation marking is exactly what prevents
Inductor from incorrectly inlining away a buffer that is actually written in
place — without the constructor's `mark_buffer_mutated` call, nothing would
stop Inductor from treating the mutating op as a pure, inlinable pointwise
computation and silently dropping the in-place write.

**The single-writer invariant.** `Buffer.get_mutation_names()`
(`ir.py:4574-4577`) returns at most one name — `ComputedBuffer` inherits it
with no override:

```python
    def get_mutation_names(self) -> Sequence[str]:
        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            return [self.layout.target.get_name()]
        return ()
```

This is hard-enforced, not just documented, by an `assert` inside
`Scheduler.compute_dependencies` at `scheduler.py:3337` (comment on the line
above): `assert len(buf.get_mutations()) <= 1`. `compute_dependencies` is
called from `Scheduler._init` — i.e. it runs before the first topological
sort, before dead-code elimination, before any torch-spyre
`CustomPreFusionPasses` hook fires. Every torch-spyre call site that assigns
a `MutationLayoutSHOULDREMOVE` satisfies this by construction — `.layout` is
a single attribute, and no site chains a new `MutationLayoutSHOULDREMOVE`
onto a target that already carries one:

| Site | File | Target |
|---|---|---|
| `_insert_copy_op` | `coarse_tile.py` | full buffer (copy-out) |
| `_insert_combine_op` | `coarse_tile.py` | `accum_full`/`accum_tile` (per-tile combine) |
| `_insert_reduction_copy_op` | `coarse_tile.py` | `accum_full` (nested-tiling copy-out) |
| fill op inside `_propagate_tiled_reduction_op` (Pass 2) | `coarse_tile.py` | fill target (identity-value seed) |

This was checked directly against the current codebase and no violation was
found — but the invariant is currently upheld by convention (one assignment
per op, never revisited), not by an assertion or type-level guard. If this
pattern is ever extended to a new call site, it is worth adding an explicit
check rather than relying on the same discipline holding indefinitely.

**A documented-but-unenforced gap.** A comment in `coarse_tile.py` states
that `MutationLayoutSHOULDREMOVE` is incompatible with
`lx_planning` (LX scratchpad placement) — the two must never be combined on
the same buffer. There is no code-level guard preventing this combination;
it currently relies entirely on pass-ordering discipline (scratchpad
placement decisions and mutation-target rewrites are kept in separate,
non-overlapping cases by construction) rather than an assertion that would
catch a future regression.

**An open upstream-adjacent TODO.** `plan_span_overflow_tile` in
`span_overflow_hint_analysis.py`
carries its own open question, quoted directly rather than resolved here:

```python
        # TODO: decide whether MutationLayoutSHOULDREMOVE producers need
        # span-overflow planning, or whether they are safe to keep outside this
        # pass as copy-back/mutation intermediates.
```

This appendix does not resolve that TODO; it is flagged here so a reader
investigating a span-overflow-related bug touching a mutation-target buffer
knows this question is already on record as open, not newly discovered.

**Two mechanisms named "propagation" — do not conflate them.** The
pass-ordering section above already establishes when each runs; the naming
collision is worth calling out explicitly since both touch
`MutationLayoutSHOULDREMOVE`-adjacent state: `coarse_tile()`'s buffer
*propagation* machinery (`_plan_tiling_propagation` plus its three
transformation passes — `_insert_all_read_copy_ops`,
`_insert_all_reduction_ops`, `_insert_all_write_copy_ops`; the last of
these is the pass that *stamps* `MutationLayoutSHOULDREMOVE` — pre-
scheduling, inside `CustomPreSchedulingPasses`, before any `Scheduler`
object exists) is a completely different mechanism from
`propagate_mutation_layouts` (pre-fusion, the first entry in
`CustomPreFusionPasses`'s pass list shown above — it *unwraps*
`MutationLayoutSHOULDREMOVE` back to a real `FixedTiledLayout`, after
`Scheduler.__init__` has already consumed the mutation-marked state).

### Why dependency info never goes stale: no caching

The soundness of "mutate `inner_fn` in place and trust that Inductor sees
the update" rests on one fact: `ComputedBuffer.get_read_writes()`
(`ir.py:4768-4787`) has **no caching decorator**. Contrast this directly with
`get_free_symbol_uses`, defined on the very next lines, which *is*
`@cache_on_self_and_args("ComputedBuffer")`-decorated:

```python
    def get_read_writes(self) -> dependencies.ReadWrites:
        if not isinstance(self.data, (Reduction, Scan, Sort, Pointwise)):
            return dependencies.ReadWrites(
                reads=OrderedSet(),
                writes=OrderedSet(),
                index_exprs=OrderedSet(),
            )

        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_pointwise_size(),
                    self.data.get_reduction_size(),
                )
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                )

    @cache_on_self_and_args("ComputedBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        ...
```

`extract_read_writes()` (`dependencies.py:659-693`) — for this call path,
where `fn` is `self.get_store_function()`, a `partial`, not a `LoopBody` —
takes the "slow path tracing the function" branch:

```python
    else:
        # Slow path tracing the function
        rw = RecordLoadStore(var_ranges, normalize=normalize)
        with V.set_ops_handler(rw):
            fn(*args, *hidden_args)
        inner = rw.parent_handler
```

Every single call builds a fresh `RecordLoadStore`, installs it via
`V.set_ops_handler`, and literally re-invokes the store function — which
re-invokes `inner_fn` — from scratch. There is no memoized `ReadWrites`
object anywhere in this path that a `coarse_tile.py` rewrite could leave
stale. Mutating `op.data.inner_fn` in place is therefore automatically and
immediately reflected the next time anything calls `get_read_writes()` — and
there is no window in which Inductor's `Scheduler` could observe stale
dependency info, because `SchedulerNode.read_writes` is itself built once,
at `Scheduler.__init__` time, which runs strictly after all of
`coarse_tile.py`'s IR rewriting (`CustomPreSchedulingPasses`, by
construction) has already completed.

`pass_utils.py`'s own comment at the `replace_computed_buffer_body` call
site is the project's own prior articulation of this exact argument: *"Always
wrap the original inner_fn via WrapperHandler; never rebuild index
expressions from scratch (they go stale — see issue #2797)."*

### DCE liveness: why reduction copy-outs survive

`Scheduler.dead_node_elimination` (`scheduler.py:3528-3567`) is a single
reverse-topological-order linear sweep — not a separate reachability
analysis:

```python
    def dead_node_elimination(self) -> None:
        """
        Remove any nodes without users
        """
        if not config.use_dce:
            return

        # self.nodes is in topological order, so by iterating in reverse order
        # we have visited (and potentially removed) all users before visiting a
        # given node.
        updated_nodes = []
        for node in reversed(self.nodes):

            def can_eliminate_user(user: NodeUser) -> bool:
                return user.is_weak or user.get_name() in V.graph.removed_operations

            active_buffers = False
            for buf in node.get_outputs():
                can_eliminate = all(can_eliminate_user(u) for u in buf.users)
                if can_eliminate:
                    log.debug("removed dead buffer: %s", buf.get_name())
                    V.graph.removed_buffers.add(buf.get_name())
                else:
                    active_buffers = True

            can_eliminate = not node.has_side_effects() and not active_buffers
            ...
```

`active_buffers` becomes `True` for a node the instant any one of its output
buffers has a live (non-weak, non-removed) user; `can_eliminate_user`
propagates removal backward as later nodes are dropped in the same reverse
sweep. A node survives exactly when it has side effects, or at least one of
its outputs still has a live user at the point the sweep reaches it.

This runs exactly **once**, inside `Scheduler._init`, at step 8
(`scheduler.py:2953`) — strictly **before** `CustomPreFusionPasses` fires
(step 14, `scheduler.py:2966-2967`) and never again afterward in `_init`.
This is the fact that matters for correctness: any liveness protection a
torch-spyre pass wants to apply must already be in place by the time this
sweep runs, not applied afterward — `CustomPreFusionPasses` is too late to
save a node DCE has already dropped.

The real problem this creates: `_propagate_tiled_reduction_op`'s nested
output-dim + reduction-dim tiling (see "Reduction tiling," above) inserts a
copy-out op (`_insert_reduction_copy_op`) that mutates a pre-loop
accumulation buffer (`accum_full`) so its updated value is visible to the
*next outer-tile iteration's* copy-in. That copy-out has no downstream
reader *in the flat scheduler IR that DCE walks, before loop codegen ever
groups it under an `scf.for`* — the buffer it writes is read again only by
that next-iteration copy-in, a cross-iteration read with no representation
at this IR level. The same problem applies to the fill op inside
`_propagate_tiled_reduction_op` that seeds the accumulator with the
reduction's identity value before the loop: its write is only ever "read" by
the next iteration's use of the fill target as an accumulator seed, again
invisible to this IR level. From DCE's perspective both ops' outputs look
like dead buffers with zero live users, and they would be removed despite
being required for correctness — a real bug the project found and fixed.

The fix is a targeted monkeypatch inside `enable_spyre_context` in
`torch_spyre/_inductor/patches.py`:

```python
    # coarse_tile.py's nested output-dim + reduction-dim tiling
    # (_propagate_tiled_reduction_op) inserts a copy-out op
    # (_insert_reduction_copy_op) that mutates a pre-loop accumulation buffer
    # (accum_full) so its updated value is visible to the NEXT outer-tile
    # iteration's copy-in. That cross-iteration read has no representation in
    # the single-pass, pre-unroll IR the scheduler's own dead_node_elimination
    # walks, so a copy-out with no other downstream reader looks dead and is
    # removed — even though it is required for correctness. Mark such ops
    # with _coarse_tile_force_live (see _insert_reduction_copy_op) and force
    # SchedulerNode.has_side_effects() to report True for them, mirroring how
    # upstream itself protects effectful FallbackKernels from the same DCE
    # pass (torch/_inductor/lowering.py, effectful op handling).
    old_scheduler_node_has_side_effects = SchedulerNode.has_side_effects

    def _spyre_scheduler_node_has_side_effects(self: SchedulerNode) -> bool:
        if getattr(self.node, "_coarse_tile_force_live", False):
            return True
        return old_scheduler_node_has_side_effects(self)

    SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects
```

The patch's own comment already draws the right analogy: this is the same
technique upstream Inductor uses to keep effectful `FallbackKernel`s (ops
with observable side effects but no reader) alive across the same DCE pass —
`has_side_effects()` is precisely the escape hatch DCE consults
(`can_eliminate = not node.has_side_effects() and not active_buffers`,
quoted above) for exactly this situation.

The patch's scope is narrow, which matters for developer confidence that it
cannot mask an unrelated bug elsewhere: it patches `SchedulerNode.
has_side_effects` specifically — not `BaseSchedulerNode`, not
`ExternKernelSchedulerNode`, not `FusedSchedulerNode` — and even for
`SchedulerNode` it falls through unchanged to the original (`@cache_on_self`-
decorated) implementation (`scheduler.py:1818-1823`) for every node except
the ones explicitly stamped. The `_coarse_tile_force_live` attribute is
stamped at exactly two sites, both in `coarse_tile.py`: inside
`_insert_reduction_copy_op`, and on the fill buffer inside
`_propagate_tiled_reduction_op`.

### Summary: invariant-by-invariant soundness table

This table is additive to the [Invariants and failure modes](#invariants-and-failure-modes)
section above, not a replacement for it — that section covers loop-structure
invariants (contiguity, consistent `loop_count`, pass ordering); this one
covers the IR-rewrite mechanism this appendix describes.

| Inductor invariant | Where enforced upstream | How torch-spyre's rewiring respects it |
|---|---|---|
| Dependencies must reflect `inner_fn` | `get_read_writes()` re-traces every call, no cache (`ir.py:4768`) | No caching exists to go stale; wrap-in-place is automatically observed |
| ≤1 mutation target per op | `assert` at `scheduler.py:3337` | Every `MutationLayoutSHOULDREMOVE` call site assigns exactly one; `.layout` is a single attribute, never chained |
| Mutated buffers must not be silently inlined | `mark_buffer_mutated` called unconditionally in the constructor (`ir.py:4383`) | Constructor call fires on every instantiation, before `make_loader()` can ever see a stale view |
| Dead nodes are pruned before codegen | `dead_node_elimination`, `scheduler.py:3528`, runs once, before `CustomPreFusionPasses` | `_coarse_tile_force_live` + patched `has_side_effects()` (in `patches.py`) protects the two reduction copy-out/fill sites that need it |
| Loop-group contiguity after scheduling | (existing invariant, cross-referenced only) | See [Contiguity invariant](#invariants-and-failure-modes) above |
