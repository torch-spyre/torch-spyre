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
- [Layer 2 — `CountedLoopSchedulerNode`](#layer-2--countedloopschedulernode)
- [Layer 3 — `LoopSpec` & codegen](#layer-3--loopspec-and-codegen)
- [Key files](#key-files)
- [Invariants](#invariants-and-failure-modes)
- [Rejected alternatives](#rejected-design-alternatives)

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
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims

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
    with spyre_hint(slices={"A": 2}):     # outer loop: 2 iterations over rows
        with spyre_hint(slices={"B": 4}): # inner loop: 4 iterations over cols
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
`tests/inductor/test_coarse_tile_e2e.py`.

### What the coarse-tiling pass stamps

`coarse_tile()` sees this as a nested group spec and stamps a single
`loop_info: CoarseTileInfo` attribute on **both** `ir.Operation` objects:

```python
from torch_spyre._inductor.loop_info import CoarseTileInfo

op.loop_info = CoarseTileInfo(
    loop_group_id=(0, 0),        # depth-2 path: group 0, inner slot 0
    loop_count=[2, 4],           # [K_outer, M_inner]
    loop_tiled_dims=[[0], [1]],  # outer loop tiles dim 0; inner tiles dim 1
)
```

`_divide_ranges` is applied once per level in outermost-first order (the
`hint_id` in each `(hint_id, K, tiled_dims)` triple is used only for
per-op `dim_index` lookup, not by `_divide_ranges` itself):

1. Outer level `(K=2, dim 0)`: `data.ranges [1024, 4096] → [512, 4096]`
2. Inner level `(M=4, dim 1)`: `data.ranges [512, 4096] → [512, 1024]`

The per-inner-iteration `data.ranges` for both ops is `[512, 1024]`.

### LoopLevel IR after CustomPreSchedulingPasses

After `coarse_tile`, `span_reduction`, `work_distribution`, and
`scratchpad_planning` have all run, the two `ComputedBuffer` objects look like
this (the `_format_operations` representation with loop attributes added):

```
buf0: ComputedBuffer                          # y = a + b
  layout = FixedTiledLayout(size=[512, 1024], stride=[1024, 1],
                            device_size=[16, 512, 64])  # per-tile shape
  op_it_space_splits = {1024: 32}            # work division: 32 cores along dim 1
  loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
                           loop_tiled_dims=[[0], [1]])
  Pointwise(
    ranges=[512, 1024],                      # per-tile iteration space
    inner_fn: load(a, i1 + 4096*i0)
              load(b, i1 + 4096*i0)
              return a + b
  )

buf1: ComputedBuffer                          # z = y * c
  layout = FixedTiledLayout(size=[512, 1024], stride=[1024, 1],
                            device_size=[16, 512, 64])  # per-tile shape
  op_it_space_splits = {1024: 32}
  loop_info=CoarseTileInfo(loop_group_id=(0, 0), loop_count=[2, 4],
                           loop_tiled_dims=[[0], [1]])
  Pointwise(
    ranges=[512, 1024],
    inner_fn: load(buf0, i1 + 4096*i0)      # reads y
              load(c,    i1 + 4096*i0)
              return y * c
  )
```

Key points:

- Both ops share the same `loop_info` with `loop_group_id = (0, 0)`,
  `loop_count = [2, 4]`, and `loop_tiled_dims = [[0], [1]]` — this is what
  `build_loop_scheduler_nodes` uses to wrap them together in a
  `CountedLoopSchedulerNode`.
- `ranges = [512, 1024]` is the *per-tile* iteration space (1/8th of the full
  tensor).  Work division and codegen see only this reduced space; the loop
  trip counts carry the information needed to reconstruct the full addressing.
- `layout.size = [512, 1024]` matches the per-tile `ranges`.  The layout
  describes the smaller per-tile output buffer allocated for each loop
  iteration.  Per-iteration addressing into the full HBM region is handled
  by `tiled_symbols` / `affine.apply` in `bundle.mlir` at runtime.
- `op_it_space_splits = {1024: 32}` is stamped by `work_distribution`: the
  coefficient `1024` identifies the per-tile stride-1 dimension (columns after
  tiling), and `32` is the number of cores dividing that dimension's work.
- `buf0` (`y`) is the intermediate result.  At this point its layout is a
  `FixedTiledLayout` with `size=[512, 1024]`; `scratchpad_planning` later
  assigns it `allocation={'lx': 0}`, placing it in LX scratchpad memory at
  address 0.  Because `y` is produced and fully consumed within the same tile
  iteration and its per-tile size fits in scratchpad, no HBM allocation is
  needed for it at all.

### Generated OpSpec (Python wrapper source)

The Python wrapper emitted by `codegen_kernel()` contains both ops inside a
single nested `LoopSpec`.  Below is the actual output produced by running the e2e test
`test_hint_nested_loop_with_scratchpad` (which uses `spyre_hint(slices=...)` /
`declare_tensor_dim` / `name_tensor_dims` with `allow_all_ops_in_lx_planning=True`;
concrete HBM addresses replaced with symbolic names for readability):

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
                            op='add',
                            is_reduction=False,
                            iteration_space={
                                sympify('c0'): (sympify('512'), 32),
                                sympify('c1'): (sympify('1024'), 1),
                            },
                            op_info={},
                            tiled_symbols=[sympify('c0'), sympify('c1')],
                            args=[
                                TensorArg(              # input a
                                    is_input=True, arg_index=0,
                                    device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[64, 1024, 64],
                                    device_coordinates=[
                                        sympify('floor(c1/64)'),
                                        sympify('c0'),
                                        sympify('Mod(c1, 64)'),
                                    ],
                                    allocation={'hbm': <base_addr_a>},
                                ),
                                TensorArg(              # input b
                                    is_input=True, arg_index=1,
                                    device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[64, 1024, 64],
                                    device_coordinates=[
                                        sympify('floor(c1/64)'),
                                        sympify('c0'),
                                        sympify('Mod(c1, 64)'),
                                    ],
                                    allocation={'hbm': <base_addr_b>},
                                ),
                                TensorArg(              # output y (LX scratchpad)
                                    is_input=False, arg_index=-1,
                                    device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[
                                        sympify('floor(c1/64)'),
                                        sympify('c0'),
                                        sympify('Mod(c1, 64)'),
                                    ],
                                    allocation={'lx': 0},
                                ),
                            ]
                        ),
                        OpSpec(
                            op='mul',
                            is_reduction=False,
                            iteration_space={
                                sympify('c0'): (sympify('512'), 32),
                                sympify('c1'): (sympify('1024'), 1),
                            },
                            op_info={},
                            tiled_symbols=[sympify('c0'), sympify('c1')],
                            args=[
                                TensorArg(              # input y (LX scratchpad)
                                    is_input=True, arg_index=-1,
                                    device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[
                                        sympify('floor(c1/64)'),
                                        sympify('c0'),
                                        sympify('Mod(c1, 64)'),
                                    ],
                                    allocation={'lx': 0},
                                ),
                                TensorArg(              # input c
                                    is_input=True, arg_index=2,
                                    device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[64, 1024, 64],
                                    device_coordinates=[
                                        sympify('floor(c1/64)'),
                                        sympify('c0'),
                                        sympify('Mod(c1, 64)'),
                                    ],
                                    allocation={'hbm': <base_addr_c>},
                                ),
                                TensorArg(              # output z (HBM, per-tile)
                                    is_input=False, arg_index=3,
                                    device_dtype=DataFormats.SEN169_FP16,
                                    device_size=[16, 512, 64],
                                    device_coordinates=[
                                        sympify('floor(c1/64)'),
                                        sympify('c0'),
                                        sympify('Mod(c1, 64)'),
                                    ],
                                    allocation={'hbm': <base_addr_z>},
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

Key observations:

- `c0` and `c1` are Inductor's iteration-space symbols for the two dimensions.
  `iteration_space` reflects the per-inner-iteration tile size `[512, 1024]`.
- `tiled_symbols=[c0, c1]` records — outermost first — which symbols correspond
  to the tiled dimensions: `c0` drives the outer `scf.for`, `c1` the inner one.
- The intermediate tensor `y` (output of `add`, input to `mul`) has
  `allocation={'lx': 0}` — it lives in LX scratchpad memory at address 0.
  Its `device_size=[16, 512, 64]` reflects the per-tile shape `[512, 1024]`.
  Because `y` is produced and fully consumed within the same tile iteration,
  no HBM allocation is needed and its address is a fixed scratchpad offset
  that does not change between loop iterations (no `affine.apply` needed).
- The final output `z` (output of `mul`) has `allocation={'hbm': ...}` and
  `arg_index=3` — it lives in HBM.  Its `device_size=[16, 512, 64]` also
  reflects the per-tile shape; the per-iteration write address into the full
  HBM buffer is computed by `affine.apply` in `bundle.mlir` (see next
  section).
- HBM inputs `a`, `b`, `c` have `device_size=[64, 1024, 64]` — the full
  tensor shape `[1024, 4096]` in Spyre stick layout.  Their
  `device_coordinates` use `c0` and `c1` to index the per-iteration tile
  window into the full tensor.  The per-tile output buffers (`y`, `z`) have
  `device_size=[16, 512, 64]`, the stick-layout shape for `[512, 1024]`
  fp16: 16 sticks of 64 columns across 512 rows.

### Generated `bundle.mlir`

The SDSC compiler (`compile_op_spec`) translates `tiled_symbols` into per-loop
byte strides, producing a 2-dimensional `affine_map`.  For this `[1024, 4096]`
fp16 tensor with Spyre stick layout (128 bytes/stick, 64 elements/stick):

- Outer stride: 512 rows × 64 sticks/row × 128 bytes/stick = 4,194,304 bytes
- Inner stride: 1024 columns / 64 elements/stick × 128 bytes/stick = 2,048 bytes

```none
#map_0 = affine_map<(d0, d1)[s0] -> (s0 + 4194304*d0 + 2048*d1)>
module {
    func.func @sdsc_bundle() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %loop_bound_0 = arith.constant 2 : index
        %loop_bound_1 = arith.constant 4 : index
        %sym_1 = arith.constant <base_addr_a> : index
        %sym_2 = arith.constant <base_addr_b> : index
        %sym_3 = arith.constant <base_addr_c> : index
        %sym_4 = arith.constant <base_addr_z> : index
        scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
            scf.for %i_1 = %c0 to %loop_bound_1 step %c1 {
                %addr_0 = affine.apply #map_0(%i_0, %i_1)[%sym_1]
                %addr_1 = affine.apply #map_0(%i_0, %i_1)[%sym_2]
                sdscbundle.sdsc_execute (%addr_0, %addr_1) {sdsc_filename="sdsc_0.json", ...}  // add: a+b→y(lx)
                %addr_2 = affine.apply #map_0(%i_0, %i_1)[%sym_3]
                %addr_3 = affine.apply #map_0(%i_0, %i_1)[%sym_4]
                sdscbundle.sdsc_execute (%addr_2, %addr_3) {sdsc_filename="sdsc_1.json", ...}  // mul: y(lx)*c→z
            }
        }
        return
    }
}
```

Both operations share the same affine map because they operate on tensors of
the same shape and stride structure.  The scratchpad tensor `y` does not appear
as a symbol — it has a fixed `lx` address that does not change between
iterations.  Each inner-loop iteration dispatches `add` then `mul` at tile
`(i_0, i_1)`, keeping the intermediate result `y` in scratchpad between the
two dispatches.

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
```

| Field | Type | Meaning |
|---|---|---|
| `loop_group_id` | `tuple[int, ...]` | Nesting-path tuple identifying which loop group this op belongs to. Its length equals the nesting depth. All ops sharing the same tuple form the body of the innermost counted loop at that path. |
| `loop_count` | `list[sympy.Expr]` | Trip counts, one per nesting level from outermost to innermost. For a flat (depth-1) group this is a 1-element list `[K]`. For a two-level nested group it is `[K1, K2]`. All ops sharing the same `loop_group_id` must agree on the count at every level. |
| `loop_tiled_dims` | `list[list[int]]` | Per-level positional indices into `data.ranges` that are divided by the corresponding count. For a flat group: `[[0]]` (tile only dim 0). For a two-level nested group: `[[0], [1]]` (outer loop tiles dim 0, inner loop tiles dim 1). Different ops in the same group may carry different indices if their iteration spaces are shaped differently. |

The pass also **rewrites the op's iteration ranges**: for each level, the
dimensions at the corresponding indices in `loop_info.loop_tiled_dims` are
divided by the corresponding count in `loop_info.loop_count`, so that each
inner `OpSpec` describes only the work done per innermost-loop iteration.

`loop_group_id` is a tuple rather than a flat integer to support nested
loops.  See "Nested loops and the `loop_group_id` tree" below.

### Why these three fields are sufficient

`loop_count` is redundant across all ops sharing the same `loop_group_id`
(they must agree), but keeping it on each op means the post-fusion pass does
not need to maintain a separate side table.  The `loop_group_id` is the join
key.  `loop_tiled_dims` is the bridge between the pre-scheduling pass (which
operates on positional `data.ranges` indices) and the codegen phase (which
uses named sympy Symbols) — it is read by `create_op_spec` to identify, by
index, which scheduler-level symbols correspond to the tiled dimensions and
should be recorded in `OpSpec.tiled_symbols`.  All levels are flattened
(outermost first) so that `tiled_symbols` covers every loop variable for the
op.  Using a list-of-lists of indices (rather than a count or a flag) allows
different ops in the same loop to tile non-contiguous or differently
positioned dimensions of their respective iteration spaces.

Crucially, `loop_tiled_dims` is **per-op**: `_stamp_group` consults each
op's own `DimHint.dim_index` for each nesting level rather than applying a
fixed spec-op index to every op.  This handles broadcast ops and other ops
whose iteration space lacks a particular dimension — those ops get an empty
sub-list `[]` for the corresponding level and are not split along that axis
(they become loop-invariant at that depth, as detected by
`insert_tiling_propagation` and flagged `per_tile_fixed`).

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
    operations: list[Operation],
    groups: list[tuple],
) -> None:
```

`groups` is a pre-computed list of group tuples produced by
`hints_to_coarse_tile_groups`.  Each `ops` list must be a contiguous
sub-sequence of `operations`; a gap indicates a data-flow dependency
crossing the group boundary and raises `RuntimeError`.

Each group tuple has the form:

```python
(ops, levels)
```

where `levels` is a list of `(hint_id, K, tiled_dims)` triples, outermost
first:

```python
(ops, [(hint_id_0, K1, [dim_0]), (hint_id_1, K2, [dim_1])])
```

`hint_id` is the integer ID assigned by the enclosing `spyre_hint` scope
(smaller IDs are outer scopes).  It is used to match each level's tiling
against the per-op `DimHint` entries so that ops which lack a particular
dimension (e.g. broadcast ops) get an empty `tiled_dims` for that level
rather than an incorrect positional index.

`_stamp_group` always receives this canonical list-of-triples
representation; normalisation from user-facing flat syntax is performed by
`hints_to_coarse_tile_groups` in `temp_passes.py` before `coarse_tile()`
is called.

### Groups derivation and placement in `CustomPreSchedulingPasses`

Groups are derived automatically from `spyre_hint(slices=...)` annotations
via `hints_to_coarse_tile_groups` (in `torch_spyre/_inductor/temp_passes.py`),
which is a no-op when no hints are present.  Coarse tiling runs only when
the returned groups list is non-empty:

```python
deadcode_elimination(operations)
propagate_spyre_tensor_layouts(operations)
optimize_restickify_locations(operations)
finalize_layouts(operations)
insert_restickify(operations)
insert_bmm_padding(operations)
dedup_and_promote_constants(operations)
if config.chunk_large_tensors:
    chunk_large_tensors(operations)
propagate_named_dims(operations)
assign_dim_hints(operations)
groups = hints_to_coarse_tile_groups(operations)
if groups:
    coarse_tile(operations, groups=groups)
span_reduction(operations)
cost_model_ops = cost_model_matmul_division(operations)
work_distribution(operations, cost_model_ops)
if config.lx_planning:
    allocator = (
        StrategyBCoOptimizingAllocator()
        if config.co_optimizing_lx_planning
        else None
    )
    scratchpad_planning(graph, allocator=allocator)
```

This ordering is required by several constraints:

**`propagate_named_dims` and `assign_dim_hints` must run before coarse tiling.**
`propagate_named_dims` propagates `name_tensor_dims()` annotations through the
op graph, attaching named dimension metadata to each `ir.Operation`.
`assign_dim_hints` then combines those named dimensions with the `spyre_hint`
scope annotations (attached to FX nodes as `meta["custom"]`) to produce
`op.dim_hints` — a flat list of `DimHint` objects consumed by
`hints_to_coarse_tile_groups` to form the coarse tiling groups.

**Must run after stickify and padding.**  `propagate_spyre_tensor_layouts`,
`insert_restickify`, and `insert_bmm_padding` establish the final tiled
memory layout for each tensor.  The tiling pass must see the post-stickify,
post-padding shapes or it will split on the wrong dimension or produce a
non-stick-aligned inner size.

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

### Buffer propagation: `insert_tiling_propagation`

`coarse_tile()` calls `insert_tiling_propagation(operations, groups)`
immediately after stamping all loop attributes.  Its job is to ensure that
any op whose result is consumed **outside** the loop (or is a graph output)
exposes a complete, fully-sized buffer to its consumers.  Ops whose outputs
are consumed only inside the loop are marked so the unroller does not advance
their base addresses.

#### Use-def analysis

For each `ComputedBuffer` in a loop group the pass asks two questions:

1. **Does this buffer have outside consumers?**  A consumer is "outside" if
   it carries a different `loop_info.loop_group_id` prefix, or has no
   `loop_info` at all.  Graph outputs (recorded in the Inductor buffer's
   `users`/`get_alias_name` machinery) count as outside consumers.

2. **Does this buffer have inside consumers?**  A consumer is "inside" if it
   shares the same `loop_info.loop_group_id` tuple (i.e. it is another op in
   the same innermost loop body).

#### Treatment by consumer topology

The perimeter is shape-asymmetric.  On the producer side (tile → full), a
tiled op writes per-tile data while an outside consumer wants full data — a
genuine shape mismatch needing adaptation.  On the consumer side (full →
tile), the loop body reads from full HBM tensors using tile-sized windows
via `affine.apply` — no conversion, just addressing.  Only producer-side
crossings need adaptation.

For each tiled `ComputedBuffer`, the pass classifies by consumer topology
and applies the cheapest treatment that maintains correctness:

| Case | Inside consumers | Outside consumers | Treatment |
|---|---|---|---|
| 1 | ✓ | ✗ | Mark `per_tile_fixed` — flag only, no IR change |
| 2 | ✓ | ✓ | Allocate full HBM buffer; insert a loop-tagged copy op that publishes each tile into the correct slice |
| 3 | ✗ | ✓ | Rewire the tiled op to write directly into a full HBM buffer via `MutationLayoutSHOULDREMOVE` — a metadata redirect, zero added data movement |

**Case 1** is where most of the working-set-reduction win comes from.  An
intermediate like `y` in the small example flows from one tiled op to
another without ever leaving scratchpad.  `per_tile_fixed` is set on the
`FixedTiledLayout`:

```python
if isinstance(op.layout, FixedTiledLayout):
    op.layout.per_tile_fixed = True
```

This flag propagates to `TensorArg.per_tile_fixed` during codegen (in
`spyre_kernel.py`).  The unroller (`codegen/unroll.py`) then skips two
things for these args: **address advance** (the base address is fixed across
iterations) and **`device_size` update** (the allocation already matches the
tile).

**Case 2**: the copy op carries the same `loop_info` (same `loop_group_id`,
`loop_count`, and `loop_tiled_dims`) as the original op, so the scheduler
wraps both in the same `CountedLoopSchedulerNode`.  The `tiled_symbols` / `affine.apply`
machinery computes the per-iteration slice offset automatically.  All
outside consumers are patched to read the full buffer.

**Case 3**: `MutationLayoutSHOULDREMOVE` tells Inductor the op mutates an
existing storage in-place.  The full buffer's address is encoded in the
`TensorArg` via the `tiled_symbols` offset; no copy op is needed.  A
unified treatment that always inserted a copy would handle all three cases
correctly but waste a copy op here.

#### Reduction safety checks

Before running the propagation logic for a `Reduction` op, the pass calls
`_check_reduction_tiling_safety(op)` which raises `RuntimeError` in two
unsupported configurations:

- **Matrix multiply** (`reduction_type == "batchmatmul"`) inside a tiling
  loop — the accumulation semantics are not handled.
- **Tiled reduction dim** — if any entry in `loop_info.loop_tiled_dims` is
  `>= len(data.ranges)`, the tiled index falls in `reduction_ranges`.
  Accumulation-buffer support for this case is not yet implemented.

Both checks happen before any buffer allocation, so the error is clean.

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
        return [memory_planning, spyre_fuse_nodes]
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

`build_loop_scheduler_nodes` scans the flat node list and groups
contiguous runs sharing the same outermost `loop_group_id` key:

```
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

Key invariant: because the pre-scheduling pass runs in topological order
and the scheduler's topological sort preserves that order, a loop group's
`SchedulerNode`s will be contiguous in the post-fusion node list.  If they
are not contiguous it means a data-flow constraint separates them, which is a
bug in the tiling pass.  The post-fusion pass asserts contiguity.

## Layer 3 — `LoopSpec` and codegen

### `LoopSpec` and `OpSpec.tiled_symbols` in `op_spec.py`

```python
@dataclasses.dataclass
class LoopSpec:
    count: sympy.Expr
    body: list[OpSpec | UnimplementedOp | LoopSpec]
    tiled_symbols: list[Symbol] = field(default_factory=list)

@dataclasses.dataclass
class OpSpec:
    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]
    tiled_symbols: list[Symbol] = field(default_factory=list)
```

`LoopSpec` is a peer of `OpSpec` and `UnimplementedOp` in the list that
`SpyreKernel.codegen_kernel()` serializes.  It is not a subclass of `OpSpec`
because it has no `iteration_space`, `args`, or `op_info` of its own — those
belong to the inner `OpSpec`s.

The `body` type is recursive: a `LoopSpec` body may itself contain
`LoopSpec` entries, representing nested counted loops.

`LoopSpec.tiled_symbols` carries the loop-level tiling information: the
iteration-space symbols divided by `count` at this loop level.  It is
populated by `_codegen_counted_loop` and `_codegen_loop_body` using
`_tiled_syms_for_sched_node_at_depth` — a helper that maps
`loop_info.loop_tiled_dims[depth]` (host-range indices) to iteration-space
symbol keys, accounting for unit-size batch dimensions that the scheduler
skips.
The `bundle.py` and `compile_op_spec` paths use `LoopSpec.tiled_symbols`
to compute per-iteration HBM address offsets.

`OpSpec.tiled_symbols` provides per-op tiling information (present for
legacy / fallback paths): all iteration-space symbols divided by any
enclosing loop, listed **outermost first**.  It is **empty for ops not
inside a `LoopSpec`**.  Two ops in the same loop group can have different
`tiled_symbols` if work division or stickification places the batch
dimension at different positions in each op's iteration space.

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
    kernel.wrap_op_specs_in_loop(node.loop_count, tiled_symbols=outer_tiled_syms)

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
    tiled_symbols=[sympify('c0')],       # the symbol tiled at this loop level
    body=[
        OpSpec(
            ...,
            tiled_symbols=[sympify('c0')],   # per-op; emitted only when non-empty
        ),
        LoopSpec(          # nested loop
            count=sympify('J'),
            tiled_symbols=[sympify('c1')],   # symbol tiled at inner level
            body=[
                OpSpec(..., tiled_symbols=[sympify('c0'), sympify('c1')]),
            ],
        ),
    ],
)
```

`LoopSpec.tiled_symbols` is populated by `_codegen_counted_loop` (depth 0)
and `_codegen_loop_body` (deeper levels) via
`_tiled_syms_for_sched_node_at_depth`.  `OpSpec.tiled_symbols` is populated
by `SpyreKernel.create_op_spec`: it reads `loop_info.loop_tiled_dims` (a
`list[list[int]]`) from the `ir.Operation` (stamped by `coarse_tile()`),
flattens all levels outermost-first, and selects the symbols at those indices
from the scheduler-level `iteration_space` dict.  `MemoryDep.ranges`
preserves the `data.ranges` ordering, so this positional correspondence is
stable across the pre-scheduling to codegen boundary.

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

### Unrolling vs. `scf.for`: the `unroll_loops` flag

Once the loop has reached `LoopSpec` form, `config.unroll_loops` (default
`True`) controls whether the loop is resolved in the frontend or passed
intact to the backend.

**`unroll_loops=True` (default):** `unroll_loop_specs` in
`codegen/unroll.py` expands each `LoopSpec(K, body)` into K copies of
`body` before `generate_bundle` runs.  Per-iteration HBM addresses are
advanced by `iter * stride`, `device_size` is set to the per-tile shape,
and `tiled_symbols` is cleared.  The resulting `bundle.mlir` contains K
plain `sdsc_execute` calls with addresses baked into each `sdsc_*.json`.

**`unroll_loops=False`:** `generate_bundle` emits the loop intact — an
`scf.for` wrapping `affine.apply` per tiled tensor followed by
`sdsc_execute`, as shown in the bundle.mlir section above.  This path is
strictly more capable (smaller bundle, late-bound addresses) but requires
backend symbol-table support that is still under development.

Nothing upstream of `generate_bundle` knows or cares which path is active.
When backend support lands, `unroll_loops` will be flipped to default
`False` and `unroll_loop_specs` will become dead code.

## Key files

| File | Role |
|---|---|
| `torch_spyre/_inductor/loop_info.py` | Layer 1: `CoarseTileInfo` dataclass; `copy_op_metadata` |
| `torch_spyre/_inductor/coarse_tile.py` | Layer 1: `coarse_tile()` stamps `loop_info` and rewrites ranges; `insert_tiling_propagation` handles the data perimeter |
| `torch_spyre/_inductor/scheduler.py` | Layer 2: `CountedLoopSchedulerNode`, `build_loop_scheduler_nodes`, `_codegen_counted_loop` |
| `torch_spyre/_inductor/op_spec.py` | Layer 3: `LoopSpec` and `OpSpec` dataclasses |
| `torch_spyre/_inductor/spyre_kernel.py` | Layer 3: serializes `LoopSpec` tree in `codegen_kernel()`; `wrap_op_specs_in_loop()` |
| `torch_spyre/_inductor/codegen/bundle.py` | Layer 3: emits `scf.for` in `bundle.mlir` for the `unroll_loops=False` path |
| `torch_spyre/_inductor/codegen/unroll.py` | Layer 3: unrolls `LoopSpec` into flat `OpSpec`s for the `unroll_loops=True` (default) path |
| `torch_spyre/_inductor/passes.py` | Wires all passes into `CustomPreSchedulingPasses` and `CustomPreFusionPasses` |
| `torch_spyre/_inductor/propagate_hints.py` | `spyre_hint()` context manager; `DimHint`; hint collection/recovery across AOT re-tracing |
| `torch_spyre/_inductor/propagate_named_dims.py` | `propagate_named_dims()` and `assign_dim_hints()`: attach `dim_hints` to `ir.Operation` objects |
| `torch_spyre/_inductor/temp_passes.py` | `hints_to_coarse_tile_groups()`: converts `dim_hints` into `coarse_tile()` group tuples |
| `tests/inductor/test_coarse_tiling.py` | Unit tests: IR pass, propagation, scheduler node, bundle MLIR output |
| `tests/inductor/test_coarse_tile_e2e.py` | End-to-end compilation tests |
| `tests/inductor/test_unroll_loop_specs.py` | Unit tests for `unroll_loop_specs` |

## Invariants and failure modes

**Contiguity invariant**: all `SchedulerNode`s sharing a
`loop_info.loop_group_id` must be contiguous after the scheduler's
topological sort.  If the tiling pass stamps ops that have a data dependency
crossing the group boundary, the post-fusion pass will detect a non-contiguous
run and raise a `RuntimeError`.

**Consistent `loop_count`**: all ops sharing a `loop_group_id` must agree on
`loop_info.loop_count` at every depth level.  The post-fusion pass asserts
this.

**`tiled_symbols` populated iff inside a loop**: `OpSpec.tiled_symbols` is
non-empty exactly when the op was codegen'd inside a `CountedLoopSchedulerNode`.
Its elements are the flattened (outermost-first) per-level tiled dims from
`loop_info.loop_tiled_dims` on the corresponding `ir.Operation`, selected from
the scheduler-level `iteration_space` keys.

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
