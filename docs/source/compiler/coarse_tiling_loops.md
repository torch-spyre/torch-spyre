# Coarse-Tiling Loop IR for the Spyre Backend

## Background

Spyre's compilation pipeline runs a sequence of optimization passes over
`ir.Operation` objects in `CustomPreSchedulingPasses`, before Inductor's
`Scheduler` is constructed.  One planned optimization is **coarse-level
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
and consumed.

**Quick navigation:**

- [Design Overview](#design-overview)
- [Small Example](#small-example)
- [Layer 1 — IR pass & `coarse_tile()` API](#layer-1--pre-scheduling-ir-pass)
- [Layer 2 — `CountedLoopSchedulerNode`](#layer-2--countedloopschedulernode)
- [Layer 3 — `LoopSpec` & codegen](#layer-3--loopspec-and-codegen)
- [Files changed](#files-changed)
- [Invariants](#invariants-and-failure-modes)
- [Rejected alternatives](#rejected-design-alternatives)

## Design Overview

The tiling loop structure must be created early (before work division sees
the iteration space) and preserved intact through scheduling and codegen so
that the hardware executes the reduced per-iteration working set — not the
full pre-tiling range.  The design has three layers that correspond to the
three pipeline stages above.

```
Pre-scheduling IR pass  (CustomPreSchedulingPasses)
  └─ stamps loop_group_id + loop_count on each ir.Operation
  └─ rewrites each op's ranges (divides the tiled dimension by K)

  ↓  Inductor Scheduler wraps each ir.Operation → SchedulerNode
  ↓  CustomPostFusionPasses fires

Post-fusion scheduler pass  (build_loop_scheduler_nodes)
  └─ runs BEFORE spyre_fuse_nodes
  └─ scans list[BaseSchedulerNode] for runs sharing a loop_group_id
  └─ wraps each run in a CountedLoopSchedulerNode(count=K, snodes=[...])
  └─ spyre_fuse_nodes runs after; CountedLoopSchedulerNode.can_fuse=False
     prevents cross-group merging

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

`coarse_tile()` sees this as a nested group spec and stamps the following
attributes on **both** `ir.Operation` objects:

```python
op.loop_group_id   = (0, 0)        # depth-2 path: group 0, inner slot 0
op.loop_count      = [2, 4]        # [K_outer, M_inner]
op.loop_tiled_dims = [[0], [1]]    # outer loop tiles dim 0; inner tiles dim 1
```

`_divide_ranges` is applied once per level in outermost-first order:

1. Outer level `(K=2, [dim 0])`: `data.ranges [1024, 4096] → [512, 4096]`
2. Inner level `(M=4, [dim 1])`: `data.ranges [512, 4096] → [512, 1024]`

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
  loop_group_id   = (0, 0)
  loop_count      = [2, 4]
  loop_tiled_dims = [[0], [1]]
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
  loop_group_id   = (0, 0)
  loop_count      = [2, 4]
  loop_tiled_dims = [[0], [1]]
  Pointwise(
    ranges=[512, 1024],
    inner_fn: load(buf0, i1 + 4096*i0)      # reads y
              load(c,    i1 + 4096*i0)
              return y * c
  )
```

Key points:

- Both ops share the same `loop_group_id = (0, 0)`, `loop_count = [2, 4]`, and
  `loop_tiled_dims = [[0], [1]]` — this is what `build_loop_scheduler_nodes`
  uses to wrap them together in a `CountedLoopSchedulerNode`.
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

The coarse-tiling pass stamps two attributes onto each `ir.Operation` that
participates in a loop group.  These attributes are plain Python values
attached with `setattr`; no Inductor base class is modified.

| Attribute | Type | Meaning |
|---|---|---|
| `loop_group_id` | `tuple[int, ...]` | Nesting-path tuple identifying which loop group this op belongs to. Its length equals the nesting depth. All ops sharing the same tuple form the body of the innermost counted loop at that path. |
| `loop_count` | `list[sympy.Expr]` | Trip counts, one per nesting level from outermost to innermost. For a flat (depth-1) group this is a 1-element list `[K]`. For a two-level nested group it is `[K1, K2]`. All ops sharing the same `loop_group_id` must agree on the count at every level. |
| `loop_tiled_dims` | `list[list[int]]` | Per-level positional indices into `data.ranges` that are divided by the corresponding count. For a flat group: `[[0]]` (tile only dim 0). For a two-level nested group: `[[0], [1]]` (outer loop tiles dim 0, inner loop tiles dim 1). Different ops in the same group may carry different indices if their iteration spaces are shaped differently. |

The pass also **rewrites the op's iteration ranges**: for each level, the
dimensions at the corresponding indices in `loop_tiled_dims` are divided by
the corresponding count in `loop_count`, so that each inner `OpSpec`
describes only the work done per innermost-loop iteration.

`loop_group_id` is a tuple rather than a flat integer to support nested
loops.  See "Nested loops and the `loop_group_id` tree" below.

### Why these three attributes are sufficient

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
    *,
    tiled_dims: list[int] | None = None,
) -> None:
```

`groups` is a pre-computed list of group tuples supplied by the caller
(e.g., `config.coarse_tiling_groups_fn`).  Each `ops` list must be a
contiguous sub-sequence of `operations`; a gap indicates a data-flow
dependency crossing the group boundary and raises `RuntimeError`.

Each group tuple takes one of two forms:

**Flat (single loop):**

```python
(ops, K)               # tile dim 0 by K (default)
(ops, K, [0, 1])       # tile dims 0 and 1 by K
```

The optional third element overrides the `tiled_dims` keyword argument for
that specific group.  `None` (the default) divides only dimension 0.

**Nested (multiple independent loops on the same ops):**

```python
(ops, [(K1, [0]), (K2, [1])])    # outer K1 on dim 0; inner K2 on dim 1
(ops, [(K1, [0]), (K2, [0,1])]) # outer K1 on dim 0; inner K2 on dims 0 and 1
```

The second element is a list of `(count, tiled_dims)` pairs, outermost first.
The ops end up in the innermost loop body; each level's count divides the
corresponding dims independently (outermost pass applied first).

`coarse_tile()` normalises flat syntax to the list-of-pairs form internally,
so `_stamp_group` always works with the canonical representation.

### Feature flag and groups callable

```python
# config.py
coarse_tiling: bool = os.environ.get("COARSE_TILING", "0") == "1"
coarse_tiling_groups_fn: Optional[Callable] = None  # overrides hint-derived groups
```

When `coarse_tiling=True` and `coarse_tiling_groups_fn` is `None`, groups
are derived automatically from `spyre_hint(slices=...)` annotations via
`hints_to_coarse_tile_groups` — a no-op if no hints are present.  Setting
`coarse_tiling_groups_fn` to a callable overrides the hint-derived groups
entirely; this is intended for interim testing until the annotation framework
matures and will be removed once it is complete.

`coarse_tiling_groups_fn` must be a **module-level named function**, not a
lambda, because Inductor's FX graph cache pickles the config values.

### Placement in `CustomPreSchedulingPasses`

The coarse-tiling pass runs after layout finalization and before
`span_reduction`:

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
if config.coarse_tiling:
    groups = hints_to_coarse_tile_groups(operations)
    if config.coarse_tiling_groups_fn is not None:
        groups = config.coarse_tiling_groups_fn(operations)
    coarse_tile(operations, groups=groups)
span_reduction(operations)
k_fast_ops = (
    k_fast_division(operations) if config.core_id_k_fast_emission else []
)
work_distribution(operations, k_fast_ops)
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
`span_reduction` and `k_fast_division` have the same requirement and
already run before `work_distribution`, so placing `coarse_tile` with
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
   it carries a different `loop_group_id` prefix, or has no `loop_group_id`
   at all.  Graph outputs (recorded in the Inductor buffer's
   `users`/`get_alias_name` machinery) count as outside consumers.

2. **Does this buffer have inside consumers?**  A consumer is "inside" if it
   shares the same `loop_group_id` tuple (i.e. it is another op in the same
   innermost loop body).

#### `per_tile_fixed` — loop-internal buffers

If a buffer has **no** outside consumers and is not a graph output, it is a
per-tile scratch region that is fully overwritten and read within the same
loop iteration.  The pass marks it:

```python
if isinstance(op.layout, FixedTiledLayout):
    op.layout.per_tile_fixed = True
```

This flag propagates to `TensorArg.per_tile_fixed` during codegen (in
`spyre_kernel.py`).  The unroller (`codegen/unroll.py`) then skips two
things for these args:

- **Address advance** — the base address is fixed; no per-iteration offset
  is added.
- **`device_size` update** — the tile geometry is not applied; the hardware
  uses the original allocation size, which already matches the tile.

#### Case 1 — output used both inside and outside the loop

The tiled op writes into its small, per-iteration buffer as usual.  The pass
allocates a full-sized HBM buffer (sized to the original pre-division ranges)
and inserts a **copy op** immediately after the tiled op in the operations
list.  The copy op carries the same `loop_group_id`, `loop_count`, and
`loop_tiled_dims` as the original, so the scheduler wraps both ops in the
same `CountedLoopSchedulerNode`.  Its `TensorArg` for the destination uses
the full buffer's address; the existing `tiled_symbols` / `affine.apply`
machinery in `SpyreKernel` and `bundle.py` computes the per-iteration slice
offset automatically.

All outside consumers are then patched to read the full buffer instead of
the tiled one.

#### Case 2 — output used only outside the loop

When no inside consumer needs the per-iteration buffer, the simplest fix is
to rewire the tiled op itself to write directly into the full buffer:

```python
op.layout = MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf)))
```

`MutationLayoutSHOULDREMOVE` tells Inductor that the op mutates an existing
storage in-place.  Because the full buffer is pre-allocated and its address
is encoded in the `TensorArg` via the `tiled_symbols` offset, no copy op is
needed.

#### Reduction safety checks

Before running the propagation logic for a `Reduction` op, the pass calls
`_check_reduction_tiling_safety(op)` which raises `RuntimeError` in two
unsupported configurations:

- **Matrix multiply** (`reduction_type == "batchmatmul"`) inside a tiling
  loop — the accumulation semantics are not handled.
- **Tiled reduction dim** — if any entry in `loop_tiled_dims` is
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

### The post-fusion pass and ordering

`CountedLoopSchedulerNode`s are created by `build_loop_scheduler_nodes`,
which is registered as the first pass in `CustomPostFusionPasses`:

```python
class CustomPostFusionPasses(CustomNodePassBase):
    def get_passes(self):
        return [memory_planning, build_loop_scheduler_nodes, spyre_fuse_nodes]
```

**`build_loop_scheduler_nodes` must run before `spyre_fuse_nodes`.**
If `spyre_fuse_nodes` ran first, it could merge `SchedulerNode`s from
different loop groups into a single `FusedSchedulerNode`.  The loop-group
pass would then see one fused node spanning multiple groups rather than
the individual nodes with distinct `loop_group_id`s, and would wrap both
groups in a single `CountedLoopSchedulerNode` with a single `loop_count`.
Running loop grouping first ensures each group is already wrapped and
opaque before `spyre_fuse_nodes` runs.  `can_fuse = False` then prevents
`spyre_fuse_nodes` from merging across group boundaries.

### The grouping algorithm

`build_loop_scheduler_nodes` scans the flat node list and groups
contiguous runs sharing the same outermost `loop_group_id` key:

```
result = []
i = 0
while i < len(nodes):
    node = nodes[i]
    gid = _loop_group_id(node)   # reads loop_group_id from the inner ir.Operation
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

`OpSpec.tiled_symbols` carries the per-op tiling information: all
iteration-space symbols that are divided by any enclosing loop, listed
**outermost first**.  It is **empty for ops that are not inside a
`LoopSpec`**.  For a single-level tiled op, `tiled_symbols = [s0]`.  For
a two-level nested tiled op, `tiled_symbols = [s_outer, s_inner]`.  The
runtime uses this list together with the enclosing loop variables (also
outermost-first) to build the affine address formula:
`base + s_outer_stride * i_outer + s_inner_stride * i_inner`.

Tiling information is stored on `OpSpec` rather than on `LoopSpec` because
different body ops may tile different iteration-space dimensions.  Two ops in
the same loop group can have different `tiled_symbols` if, for example, work
division or stickification places the batch dimension at different positions in
each op's iteration space.  A single `int` on `LoopSpec` cannot express this;
per-op `list[Symbol]` can.

### Nested loops and the `loop_group_id` tree

Each `ir.Operation` carries a `loop_group_id` that is a **path** rather
than a flat integer.  A path is a tuple of integers, one element per
nesting level:

| `loop_group_id` | Meaning |
|---|---|
| `(0,)` | outermost loop group 0, not nested |
| `(0, 0)` | single op nested two levels deep inside group 0 |
| `(0, 1)` | ops at depth 2 inside outer group 0, inner group 1 |

`loop_count` is a **list** parallel to the path.  For a flat op at
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
   indexes `loop_count[depth - base_depth]`.  All nodes in the run must
   agree on this count.
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
            tiled_symbols=[sympify('c0')],   # emitted only when non-empty
        ),
        LoopSpec(          # nested loop
            count=sympify('J'),
            body=[
                OpSpec(..., tiled_symbols=[sympify('c0'), sympify('c1')]),
            ],
        ),
    ],
)
```

`tiled_symbols` is populated by `SpyreKernel.create_op_spec`: it reads
`loop_tiled_dims` (a `list[list[int]]`) from the `ir.Operation` (stamped
by `coarse_tile()`), flattens all levels outermost-first, and selects the
symbols at those indices from the scheduler-level `iteration_space` dict.
`MemoryDep.ranges` preserves the `data.ranges` ordering, so this
positional correspondence is stable across the pre-scheduling to codegen
boundary.

`tiled_symbols` is omitted from the serialized source when empty (i.e. for ops
that are not inside a loop), keeping the generated output identical to the
pre-tiling baseline for non-tiled kernels.

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

## Files changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/op_spec.py` | Add `LoopSpec` dataclass (recursive body type); add `tiled_symbols: list[Symbol]` to `OpSpec` |
| `torch_spyre/_inductor/spyre_kernel.py` | Add `SpyreKernel.wrap_op_specs_in_loop()`; extend `codegen_kernel()` to serialize `LoopSpec` recursively; populate `OpSpec.tiled_symbols` in `create_op_spec`; fix `arg_index` fixup to walk nested bodies |
| `torch_spyre/_inductor/scheduler.py` | Add `CountedLoopSchedulerNode(FusedSchedulerNode)` with `unpack()` override; add `build_loop_scheduler_nodes` and `_codegen_counted_loop`/`_codegen_loop_body` to `SuperDSCScheduling` |
| `torch_spyre/_inductor/passes.py` | Add `coarse_tile()` call (with `hints_to_coarse_tile_groups` fallback) in `CustomPreSchedulingPasses`; add `propagate_named_dims` and `resolve_hints` calls before coarse tiling; reorder `CustomPostFusionPasses` to `[memory_planning, build_loop_scheduler_nodes, spyre_fuse_nodes]` |
| `torch_spyre/_inductor/config.py` | Add `coarse_tiling: bool` flag, `coarse_tiling_groups_fn` override callable, `bundle_hbm_symbols: bool`, `unroll_loops: bool`, and `allow_all_ops_in_lx_planning: bool` |
| `torch_spyre/_inductor/propagate_hints.py` | Add `spyre_hint(slices=...)` context manager and `get_op_hints()`; `resolve_hints` stamps hint metadata on `ir.Operation` objects; `hints_to_coarse_tile_groups` converts resolved hints to a `coarse_tile` groups list |
| `torch_spyre/_inductor/propagate_named_dims.py` | `propagate_named_dims` propagates `name_tensor_dims()` annotations from FX nodes to `ir.Operation` objects; `assign_dim_hints` combines those named dimensions with `spyre_hint` scope metadata to produce `op.dim_hints`, consumed by `hints_to_coarse_tile_groups` |
| `torch_spyre/_inductor/coarse_tile.py` | New file: `coarse_tile(operations, groups)` pass; stamps `loop_group_id`, `loop_count`, and `loop_tiled_dims` on ops; rewrites `ranges` via `object.__setattr__`; `insert_tiling_propagation` allocates full buffers for outside consumers, marks loop-internal buffers `per_tile_fixed` |
| `torch_spyre/_inductor/ir.py` | Add `per_tile_fixed: bool = False` to `FixedTiledLayout` |
| `torch_spyre/_inductor/op_spec.py` | Add `per_tile_fixed: bool = False` to `TensorArg` |
| `torch_spyre/_inductor/codegen/unroll.py` | Add `_tile_device_size` helper; apply tile-sized `device_size` and skip address advance for `per_tile_fixed` args during loop unrolling |
| `torch_spyre/_inductor/wrapper.py` | Add `LoopSpec` to the generated wrapper's import line |
| `torch_spyre/_inductor/codegen/bundle.py` | Extend `generate_bundle()` to walk `LoopSpec` tree and emit `scf.for` in `bundle.mlir`; number SDSC JSON files in depth-first order |
| `torch_spyre/execution/async_compile.py` | `sdsc()` accepts `Sequence[OpSpec | UnimplementedOp | LoopSpec]`; delegates `_find_unimplemented` to `bundle.py` |
| `tests/inductor/test_coarse_tiling.py` | Consolidated unit test suite: `LoopSpec`/`OpSpec` data structures, `coarse_tile()` IR pass, `insert_tiling_propagation`, `CountedLoopSchedulerNode`, `generate_sdsc`/`compile_op_spec` symbol paths, `generate_bundle` MLIR output and snapshot tests (104 tests) |
| `tests/inductor/test_coarse_tile_e2e.py` | End-to-end compilation tests: baseline, single group, softmax-shaped, two groups, per-group tiled dims, unrolled execution |
| `tests/inductor/test_unroll_loop_specs.py` | Unit tests for `unroll_loop_specs`: address arithmetic, `per_tile_fixed` handling, nested loops, stride computation |

## Invariants and failure modes

**Contiguity invariant**: all `SchedulerNode`s sharing a `loop_group_id`
must be contiguous after the scheduler's topological sort.  If the tiling
pass stamps ops that have a data dependency crossing the group boundary,
the post-fusion pass will detect a non-contiguous run and raise a
`RuntimeError`.

**Consistent `loop_count`**: all ops sharing a `loop_group_id` must agree
on `loop_count` at every depth level.  The post-fusion pass asserts this.

**`tiled_symbols` populated iff inside a loop**: `OpSpec.tiled_symbols` is
non-empty exactly when the op was codegen'd inside a `CountedLoopSchedulerNode`.
Its elements are the flattened (outermost-first) per-level tiled dims from
`loop_tiled_dims` on the corresponding `ir.Operation`, selected from the
scheduler-level `iteration_space` keys.

**Pass ordering**: coarse tiling must run after stickify/padding and
before `span_reduction`, `k_fast_division`, `work_distribution`, and
`scratchpad_planning`.  `build_loop_scheduler_nodes` must run before
`spyre_fuse_nodes` in `CustomPostFusionPasses` — see the ordering
rationale above.

**Cache invalidation**: `coarse_tile.py`, `scratchpad_planning`, and all
other pass source files are included in `CustomPreSchedulingPasses.uuid()`
so the Inductor FX cache is invalidated when any pass changes.  The
`coarse_tiling_groups_fn` must be a module-level named function (not a
lambda) for Inductor's cache pickling to work.

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
