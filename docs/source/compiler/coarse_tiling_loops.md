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
and consumed.  For the motivation, why the design has the shape it does and
what constraints forced each choice, see the companion RFC
[1358-CoarseTiling](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md).

**Quick navigation:**

- [Design Overview](#design-overview)
- [Small Example](#small-example)
- [Layer 1: Pre-scheduling IR pass](#layer-1-pre-scheduling-ir-pass)
  - [The span-overflow lowering: `tile_groups`](#the-span-overflow-lowering-tile_groups)
- [Layer 2: `CountedLoopSchedulerNode`](#layer-2-countedloopschedulernode)
- [Layer 3: `LoopSpec` & codegen](#layer-3-loopspec-and-codegen)
- [Key files](#key-files)
- [Invariants](#invariants-and-failure-modes)
- [Appendix: How IR rewiring works, and why it's sound](#appendix-how-ir-rewiring-works-and-why-its-sound)

## Design Overview

The tiling loop structure must be created early (before work division sees
the iteration space) and preserved intact through scheduling and codegen so
that the hardware executes the reduced per-iteration working set rather than
the full pre-tiling range. The design has three layers that correspond to
the three pipeline stages above. At each layer the
same concept, that these ops are inside a counted loop, takes the form
demanded by that layer's type system:

| Layer | Loop identity | Form |
|---|---|---|
| 1, pre-scheduling IR pass | `loop_info: CoarseTileInfo` on `ir.Operation`, with the loop's ops wrapped in a `LoopNode` | Per-op tag plus a structural node |
| 2, scheduler | `LoopSchedulerNode` (a `CountedLoopSchedulerNode` subclass) | Perimeter wrapper |
| 3, codegen output | `LoopSpec` | Serializable tree node |

:::{figure} ../_static/images/coarse-tiling-layers.svg
:alt: Three layers of coarse-tiling loop IR: the pre-scheduling pass stamps CoarseTileInfo and divides ranges, the pre-fusion pass wraps runs in CountedLoopSchedulerNode, and codegen emits a LoopSpec that serializes to scf.for
:width: 100%
:align: center

The three layers, from the pre-scheduling IR pass through the pre-fusion
scheduler pass to codegen. Layer 1 stamps `loop_info` (`CoarseTileInfo`) on
each `ir.Operation`, divides the tiled dimension by `K`, and wraps the loop's
ops in a `LoopNode` so the loop is a structural unit from the first pass after
the splice. Layer 2 hands each `LoopNode` to Inductor's scheduler as one
`LoopSchedulerNode` (a `CountedLoopSchedulerNode` subclass), which is opaque to
Inductor fusion. Layer 3 drives `SpyreKernel` for the inner ops and wraps their
`OpSpec`s in a `LoopSpec` that serializes to an `scf.for` with late-bound
addresses.
:::

## Small Example

Consider two chained pointwise operations over `[1024, 4096]` tensors, tiled
along dim 0 in tiles of 128 rows with `for_each_tile`:

```python
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

a = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
b = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
c = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")

def fn(a, b, c):
    def body(_, ops):
        a_tile, b_tile, c_tile = ops
        y_tile = a_tile + b_tile
        return None, y_tile * c_tile

    _, z = for_each_tile(
        body, (a, b, c), dims=(0, 0, 0), tile_size=128, out_dim=0
    )
    return z
```

This tiles the 1024 rows into 8 tiles of 128 rows each. Each iteration
processes a 128 × 4096 tile (1/8th of the full tensor), enabling the
intermediate result `y_tile` to remain in scratchpad across both operations
within the tile. There is only one loop level here: `for_each_tile` expresses
one loop level per call (see [Composing and nesting](working_set_reduction.md#composing-and-nesting)
in `working_set_reduction.md` for how a second, nested level is built by
calling `for_each_tile` again inside `body`).

See the [for_each_tile example](https://github.com/torch-spyre/torch-spyre/blob/main/docs/source/user_guide/examples/for_each_tile.py) for
a runnable script that tiles a computation with `for_each_tile`, using a map
over one dimension and a split-K carry over another.

This example is exactly what `docs/tools/capture_for_each_tile_ir.py`
compiles. Every IR/OpSpec/`bundle.mlir` snippet below is real, captured
output, not hand-derived. When compiler internals drift and these snippets
go stale, regenerate them with that script rather than hand-editing; see
`docs/tools/README.md` for usage.

### Prove, splice, identify, stamp: how a `for_each_tile` call becomes `loop_info`

`for_each_tile` lowers to `torch._higher_order_ops.scan.scan`, which Inductor
traces into an `ir.WhileLoop`. `splice_while_loops`
(`for_each_tile_lowering.py`) recognizes the resulting while-loop shape,
proves its trip count statically via `try_prove_for_each_tile`/
`_extract_trip_count`, splices the loop body's ops directly into the
top-level `graph.operations` list, and calls `_stamp_direct_loop_info` to
attach `loop_info` to each spliced op, **directly**, with no intervening
planning step. This is the real, captured value of
`while_loop_body_graph_0_0_op8`'s (`y_tile = a_tile + b_tile`'s) `loop_info`,
immediately after `splice_while_loops` runs, before any later pass touches
it:

```python
op.loop_info = CoarseTileInfo(
    loop_group_id=(0,),            # depth-1 path: single loop level
    loop_count=[8],                # trip_count=8 (1024 rows / 128-row tiles)
    loop_tiled_dims=[[0]],         # dim 0 (rows) is tiled
    loop_tiled_reduction_dims=[[]],  # no reduction dims (pointwise op)
)
```

The record holds the loop's own facts only. How each access advances per trip
is not stored: `_stamp_direct_loop_info` also appends a `DimHint` with
`loop_var=u0` and `loop_var_range=8` to the op's `dim_hints`, and every reader
takes an access's advance from the coefficient of `u0` in that access's index
(`loop_facts.read_advance`, `SpyreKernel._index_tile_advance`). The
span-overflow lowering stamps a `LoopShim` with the same four fields and
follows the same rule (see [Layer 1](#layer-1-pre-scheduling-ir-pass) below),
so both frontends share every downstream layer.

`while_loop_body_graph_0_0_op9` (`z_tile = y_tile * c_tile`) is tiled
identically. Its read of `y_tile` (`op8`'s output) has no `u0` term, because
`op8`'s own buffer is tile-sized and rewritten every trip; that read does not
advance.

### `LoopNode`: the loop as a structural unit

The final step of `splice_while_loops` is `collapse` (`loop_node.py`), which
wraps each loop's ops in a `LoopNode` (nested for a nested loop). A `LoopNode`
is one `ir.Operation` held in `graph.operations`; its `body` holds the ordinary
ops and any nested `LoopNode`s, and it carries the per-loop facts (induction
variable, trip count, carries). `loop_info` stays on the leaf ops; the node
adds the loop structure and those facts without changing the leaf ops. So from
the first pass after the splice, the loop is a structural unit in the IR.

Pre-scheduling passes that know nothing about loop nodes run against an
`OpsView`, which iterates the leaf ops in program order and forwards every
mutation to the list that owns the op. The nested nodes therefore survive the
whole pass pipeline. `dedup_and_promote_constants` hoists constants across node
boundaries itself (`OpsView.hoist_to_front`). A pass that inserts or moves an op
names its anchor through the helpers in
[Placing ops relative to a loop](#placing-ops-relative-to-a-loop). At the hand-off to the scheduler,
`peel_non_members` moves any op a pass left at a body edge without the loop's
`loop_info` back out of the node, and each `LoopNode` becomes one
`LoopSchedulerNode` (gated by `loop_scheduler_supported()`; `SCHEDULE_LOOP_NODES`
defaults to true). Building the nodes is gated by `BUILD_LOOP_NODES`, which
defaults to true and is set false only by a no-behaviour-change test.

### The single-level `graph.operations` dump

After `splice_while_loops` runs (the second pass, ahead of
`deadcode_elimination`; see [pass ordering](#invariants-and-failure-modes)
below) and every later pass (`split_multi_ops`,
`propagate_spyre_tensor_layouts`, `insert_restickify`, `_distribute_work`,
`_maybe_scratchpad_planning`) has run,
`graph.operations` contains three tiled ops living inside the spliced loop
body, plus a `buf4` placeholder for the eventual output `z` (created earlier
by Dynamo's `getitem_3`/`empty_strided`). This is the real, unedited output of
`format_operations(graph.operations)` at `sencores=4`, in topological order,
using `u0` as the `inner_fn` index variable for the tiled dim:

```
buf4: ComputedBuffer                              # placeholder for z
  layout=FixedTiledLayout('spyre:0', torch.float16, size=[1024, 4096], stride=[4096, 1],
      device_layout=SpyreTensorLayout(device_size=[64, 1024, 64], stride_map=[64, 4096, 1],
                                       device_dtype=DataFormats.SEN169_FP16))

while_loop_body_graph_0_0_op8: ComputedBuffer     # y_tile = a_tile + b_tile
  layout=FixedTiledLayout('spyre:0', torch.float16, size=[128, 4096], stride=[4096, 1],
      device_layout=SpyreTensorLayout(device_size=[64, 128, 64], stride_map=[64, 4096, 1],
                                       device_dtype=DataFormats.SEN169_FP16))
  allocation={'lx': 0}
  dim_hints=[DimHint(dim_names=[], split_count=1, loop_var=u0, is_reduction=False, hint_id=0, loop_var_range=8)]
  loop_info=CoarseTileInfo(loop_group_id=(0,), loop_count=[8], loop_tiled_dims=[[0]],
      loop_tiled_reduction_dims=[[]])
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(arg0_1, i1 + 4096 * i0 + 524288 * u0)   # a, direct full-buffer read
        tmp1 = ops.load(arg1_1, i1 + 4096 * i0 + 524288 * u0)   # b, direct full-buffer read
        tmp2 = tmp0 + tmp1
        return tmp2
    ,
    ranges=[128, 4096],
    origin_node=add,
  )

while_loop_body_graph_0_0_op9: ComputedBuffer     # z_tile = y_tile * c_tile
  layout=FixedTiledLayout('spyre:0', torch.float16, size=[128, 4096], stride=[4096, 1],
      device_layout=SpyreTensorLayout(device_size=[64, 128, 64], stride_map=[64, 4096, 1],
                                       device_dtype=DataFormats.SEN169_FP16))
  allocation={'lx': 262144}
  dim_hints=[DimHint(dim_names=[], split_count=1, loop_var=u0, is_reduction=False, hint_id=0, loop_var_range=8)]
  loop_info=CoarseTileInfo(loop_group_id=(0,), loop_count=[8], loop_tiled_dims=[[0]],
      loop_tiled_reduction_dims=[[]])
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(while_loop_body_graph_0_0_buf8, i1 + 4096 * i0)   # y_tile, tile-local
        tmp1 = ops.load(arg2_1, i1 + 4096 * i0 + 524288 * u0)             # c, direct full-buffer read
        tmp2 = tmp0 * tmp1
        return tmp2
    ,
    ranges=[128, 4096],
    origin_node=mul,
  )

while_loop_body_graph_0_0_op15: ComputedBuffer    # identity copy: z_tile → z
  layout=MutationLayoutSHOULDREMOVE('spyre:0', torch.float16, size=[128, 4096], stride=[4096, 1])
  dim_hints=[DimHint(dim_names=[], split_count=1, loop_var=u0, is_reduction=False, hint_id=0, loop_var_range=8)]
  loop_info=CoarseTileInfo(loop_group_id=(0,), loop_count=[8], loop_tiled_dims=[[0]],
      loop_tiled_reduction_dims=[[]])
  Pointwise(
    'spyre', torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(while_loop_body_graph_0_0_buf9, i1 + 4096 * i0)
        return tmp0
    ,
    ranges=[128, 4096],
    origin_node=None,
  )
```

This example uses `sencores=4` (rather than the default 32) purely for
readability.

Key points worth reading closely:

- **There are no separate read-copy ops.** `op8` and `op9` load directly
  from the full-tensor graph inputs (`arg0_1`, `arg1_1`, `arg2_1`) with the
  tile advance baked straight into their own index expression
  (`i1 + 4096 * i0 + 524288 * u0`, the `524288 * u0` term is the per-tile
  row advance, `128 * 4096`). `for_each_tile`'s own tracing through `scan`
  already produces per-tile-sized reads of `a`/`b`/`c`
  directly, with no full-size intermediate buffer for a copy op to adapt.
- **`u0` is the (spliced-in) loop induction variable**, and it appears
  explicitly in the read index expressions above (`524288 * u0`) rather than
  being folded away into a separately-tracked `tiled_symbols` structure at
  this IR stage, that folding happens later, in codegen (see the OpSpec
  section below).
- `op9`'s read of `op8`'s output (`while_loop_body_graph_0_0_buf8`) uses
  coefficient `4096`, matching `op8`'s own per-tile `FixedTiledLayout` with
  `stride=[4096, 1]`, the "read a tile-local producer at its own per-tile
  stride" pattern, without any `_patch_retiled_load_indexes` involvement
  (there is no separate full-size-then-divided buffer here to retile).
- All three ops share `loop_group_id=(0,)` and `loop_count=[8]`, and all
  three are members of one `LoopNode`, which reaches the scheduler as one
  `LoopSchedulerNode`.
  `while_loop_body_graph_0_0_op15` (the identity copy that drains
  `op9`'s tile into `z`) is tiled the same way even though its own layout is
  `MutationLayoutSHOULDREMOVE` over the full `[1024, 4096]` shape; see
  [MutationLayoutSHOULDREMOVE: the real contract](#mutationlayoutshouldremove-the-real-contract).
- The write indexes of `op8` and `op9` have no `u0` term, so neither's own
  small buffer advances (see
  [Attribute contract on `ir.Operation`](#attribute-contract-on-iroperation)
  below): `SpyreKernel._tile_advance` returns `None` for those writes, which
  is what lets `scratchpad_planning` place both in `lx` (`op8` at offset `0`,
  `op9` at offset `262144`) instead of falling back to `hbm_pool`.
- `op15` writes its `MutationLayoutSHOULDREMOVE` target (`z`) at an index
  that carries the `u0` term, so `z` advances by 128 rows per iteration.

### Generated OpSpec (Python wrapper source)

The Python wrapper emitted by `codegen_kernel()` contains all three tiled ops
(`add`, `mul`, and the output `identity` copy) inside a single, single-level
`LoopSpec`. Below is the actual output captured by
`docs/tools/capture_for_each_tile_ir.py` at `sencores=4` (the
`debug_handle=DebugHandle(...)` field each real `OpSpec` carries is shown in
full below, since it is short enough here to be worth reading. It records
the full fusion/provenance chain back through `for_each_tile`'s own
`scan`/`view` tracing):

```python
sdsc_fused_add_copy__mul_select_view_0 = async_compile.sdsc('sdsc_fused_add_copy__mul_select_view_0',
    [
        LoopSpec(
            count=sympify('8'),
            body=[
                OpSpec(
                    op='add',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('128'), 1), sympify('c1'): (sympify('4096'), 4)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_while_loop_body_graph_0_0_op8_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_while_loop_body_graph_0_0_op8_lvl0'): 8},
                    core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)')},
                    symbolic_dim_bounds={},
                    args=[
                        TensorArg(              # input a (HBM, full tensor)
                            is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 1024, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 0},
                            device_tile_advance_expr=sympify('floor(8192*_tile_adv_while_loop_body_graph_0_0_op8_lvl0)'),
                        ),
                        TensorArg(              # input b (HBM, full tensor)
                            is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 1024, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 1},
                            device_tile_advance_expr=sympify('floor(8192*_tile_adv_while_loop_body_graph_0_0_op8_lvl0)'),
                        ),
                        TensorArg(              # output y_tile (LX scratchpad)
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 128, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 0},
                        ),
                    ]
                ),
                OpSpec(
                    op='mul',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('128'), 1), sympify('c1'): (sympify('4096'), 4)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_while_loop_body_graph_0_0_op9_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_while_loop_body_graph_0_0_op9_lvl0'): 8},
                    core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)')},
                    symbolic_dim_bounds={},
                    args=[
                        TensorArg(              # input y_tile (LX scratchpad)
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 128, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 0},
                        ),
                        TensorArg(              # input c (HBM, full tensor)
                            is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 1024, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 2},
                            device_tile_advance_expr=sympify('floor(8192*_tile_adv_while_loop_body_graph_0_0_op9_lvl0)'),
                        ),
                        TensorArg(              # output z_tile (LX scratchpad)
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 128, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 262144},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',                 # while_loop_body_graph_0_0_op15
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('128'), 1), sympify('c1'): (sympify('4096'), 4)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_while_loop_body_graph_0_0_op15_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_while_loop_body_graph_0_0_op15_lvl0'): 8},
                    core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)')},
                    symbolic_dim_bounds={},
                    args=[
                        TensorArg(              # input z_tile (LX scratchpad)
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 128, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 262144},
                        ),
                        TensorArg(              # output z (HBM, full tensor)
                            is_input=False, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[64, 1024, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 3},
                            device_tile_advance_expr=sympify('floor(8192*_tile_adv_while_loop_body_graph_0_0_op15_lvl0)'),
                        ),
                    ]
                ),
            ],
        ),
    ]
)
```

(`debug_handle=DebugHandle(...)` is omitted above for brevity. It carries
the fusion/provenance chain, not tiling-relevant information.)

Key observations:

- **`c0`/`c1` are used throughout, at every op.** All three ops are
  generated from the same single `CountedLoopSchedulerNode`, so there is
  only one iteration-space symbol pair for the whole loop body.
- **Single-level `tiled_symbols`.** Each op has exactly one level symbol,
  since there is only one loop level (`tiled_symbols=[[lvl0]]`).
  `SpyreKernel._get_or_mint_level_symbol` returns the loop's own induction
  variable when the op lists one loop-variable hint per level
  (`_induction_symbol`), and otherwise mints `_tile_adv_{op_name}_lvl{level}`,
  so each `(op, level)` pair has a distinct symbol and two levels that tile
  the same host dim are not summed into one coefficient. A nested tiling loop
  (composed of two `for_each_tile` calls, see `working_set_reduction.md`'s
  Composing and nesting section) produces the `[[lvl1], [lvl0]]` shape.
- **Only the four full-tensor HBM `TensorArg`s carry a
  `device_tile_advance_expr`**, `a`, `b`, `c` (each op's own HBM input) and
  `z` (the final identity copy's HBM output). The accesses of `y_tile` and
  `z_tile`, both in `lx`, have no loop-variable term in their index, so
  `SpyreKernel._tile_advance` returns `None` and the printer omits the
  field.
- **`add` and `mul` are *not* zero-operand.** Each still reads one HBM
  operand directly (`a`+`b_tile`'s scratchpad neighbor for `add`; `c` for
  `mul`) because there is no earlier op to have already absorbed that HBM
  read. This is the direct consequence of the "no separate read-copy ops"
  point called out in the `graph.operations` section above. Only the
  identity copy at the end carries a full-tensor HBM operand on its output
  side.

### Generated `bundle.mlir`

The SDSC compiler translates `tiled_symbols` into per-loop byte strides,
producing a 1-dimensional `affine_map` (only one loop level, so only one
induction variable) for each `TensorArg` whose `device_tile_advance_expr` is
non-`None`. There is exactly one such map, `#map_0`, covering the four
full-tensor HBM operands (`a`, `b`, `c`, `z`, bound to `%arg_0`..`%arg_3`).
`y_tile`/`z_tile` live in `lx` and have `device_tile_advance_expr=None`, so
neither gets an affine map, a per-core address, or any operand:

```none
#map_0 = affine_map<(d0)[s0] -> (s0 + 16384*d0)>
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
        %loop_bound_0 = arith.constant 8 : index

        // per-core address = base + core_index * 2097152 bytes, for each of
        // the 4 cores (sencores=4); shown here for arg_0 (tensor a),
        // identical patterns repeat for arg_1/arg_2/arg_3, elided here.
        // These are all computed once, outside the loop nest.
        %arg_0_core_offset_2097152 = arith.constant 2097152 : index
        %arg_0_core_2097152 = arith.addi %arg_0, %arg_0_core_offset_2097152 : index
        %arg_0_core_offset_4194304 = arith.constant 4194304 : index
        %arg_0_core_4194304 = arith.addi %arg_0, %arg_0_core_offset_4194304 : index
        %arg_0_core_offset_6291456 = arith.constant 6291456 : index
        %arg_0_core_6291456 = arith.addi %arg_0, %arg_0_core_offset_6291456 : index
        // ... (arg_1_core_*, arg_2_core_*, arg_3_core_* follow the same
        // pattern above, elided here ...)

        scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
            // add: a(hbm)+b(hbm)→y_tile(lx)
            %addr_0 = affine.apply #map_0(%i_0)[%arg_0]
            %addr_1 = affine.apply #map_0(%i_0)[%arg_0_core_2097152]
            %addr_2 = affine.apply #map_0(%i_0)[%arg_0_core_4194304]
            %addr_3 = affine.apply #map_0(%i_0)[%arg_0_core_6291456]
            %addr_4 = affine.apply #map_0(%i_0)[%arg_1]
            %addr_5 = affine.apply #map_0(%i_0)[%arg_1_core_2097152]
            %addr_6 = affine.apply #map_0(%i_0)[%arg_1_core_4194304]
            %addr_7 = affine.apply #map_0(%i_0)[%arg_1_core_6291456]
            sdscbundle.sdsc_execute (%addr_0, %addr_1, %addr_2, %addr_3, %addr_4, %addr_5, %addr_6, %addr_7)
                {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1, -2, -3, -4, -5, -6, -7, -8]}

            // mul: y_tile(lx)*c(hbm)→z_tile(lx)
            %addr_8 = affine.apply #map_0(%i_0)[%arg_2]
            %addr_9 = affine.apply #map_0(%i_0)[%arg_2_core_2097152]
            %addr_10 = affine.apply #map_0(%i_0)[%arg_2_core_4194304]
            %addr_11 = affine.apply #map_0(%i_0)[%arg_2_core_6291456]
            sdscbundle.sdsc_execute (%addr_8, %addr_9, %addr_10, %addr_11)
                {sdsc_filename="sdsc_1.json", "symbol_ids"=[-9, -10, -11, -12]}

            // identity: z_tile(lx)→z(hbm)
            %addr_12 = affine.apply #map_0(%i_0)[%arg_3]
            %addr_13 = affine.apply #map_0(%i_0)[%arg_3_core_2097152]
            %addr_14 = affine.apply #map_0(%i_0)[%arg_3_core_4194304]
            %addr_15 = affine.apply #map_0(%i_0)[%arg_3_core_6291456]
            sdscbundle.sdsc_execute (%addr_12, %addr_13, %addr_14, %addr_15)
                {sdsc_filename="sdsc_2.json", "symbol_ids"=[-13, -14, -15, -16]}
        }
        return
    }
}
```

This is the full, real captured output, nothing is elided in the loop body
above: there are only three dispatches total, so eliding a "repeats the same
pattern" block would save little.

Key points:

- **One affine map, no `%pool_base_addr` argument**, 1-D
  (`affine_map<(d0)[s0] -> ...>`), since there is only one loop level.
- **`add` and `mul` are not zero-operand.** Each dispatch carries the
  full-tensor HBM operand(s) it reads or writes directly, `add` takes both
  `a` and `b` (8 operands: 4 cores × 2 tensors), `mul` takes `c` (4
  operands), and the identity copy takes `z` (4 operands). This is the
  `bundle.mlir`-level consequence of "no separate read-copy ops" from the
  `graph.operations` section above: there is no earlier dispatch that has
  already absorbed the HBM read into a fixed LX address, so `add`/`mul`
  themselves carry the per-core addressing.
- **Every full-tensor HBM operand is expanded into `sencores` per-core
  addresses**: at `sencores=4`, each of `arg_0`/`arg_1`/`arg_2`/`arg_3`
  contributes its own base address plus three `arith.addi`-computed offsets,
  stepping by `2097152` bytes (`8388608 / 4`, the per-core share of the
  tensor's total byte size).
- **Neither `y_tile` nor `z_tile` appears as a symbol at all**, neither
  has a `device_tile_advance_expr` (see the OpSpec above), so neither needs any
  address computation, per-core or otherwise: each is a compile-time-fixed
  LX offset baked into the compiled `.json` kernel for its dispatch.

## Layer 1: Pre-scheduling IR pass

### Attribute contract on `ir.Operation`

Each `ir.Operation` inside a counted loop carries a `loop_info` attribute that
holds the loop's own facts. The `for_each_tile` lowering
(`_stamp_direct_loop_info` in `for_each_tile_lowering.py`) stamps a
`CoarseTileInfo`. The span-overflow lowering (`loop_tiling_lowering.py`)
stamps a `LoopShim`, a frozen dataclass with the same four fields. Both are
defined in `torch_spyre/_inductor/loop_info.py` and attached with `setattr`; no
Inductor base class is modified. A reader that needs only the loop facts tests
`loop_info` against `COUNTED_LOOP_INFO`, the tuple of both classes.

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
| `loop_tiled_dims` | `list[list[int]]` | Per-level positional indices into `data.ranges` (the output iteration space) that are divided by the corresponding count. An empty sub-list means the op is loop-invariant at that level in the output space. |
| `loop_tiled_reduction_dims` | `list[list[int]]` | Per-level positional indices into `data.reduction_ranges` that are tiled at that level. Parallel to `loop_tiled_dims`. An empty sub-list means no reduction dim is tiled at that level. |

Neither record stores how an access advances from one trip to the next. An
access's per-trip advance is the coefficient of the level's loop variable in
the access's index. `loop_facts.read_advance(op, dep)` and
`loop_facts.output_advance(op)` read it (through `_loop_coefficient`, which
refuses a non-affine use), and `SpyreKernel._index_tile_advance` reads it the
same way when it builds each `TensorArg.device_tile_advance_expr`. Level `i`'s
loop variable is the `i`-th loop-variable `DimHint` in the op's `dim_hints`
(`loop_var_ranges_from_dim_hints`), and both lowerings list one per level on
every member. A pass that rewrites an access keeps its advance as long as it
keeps the loop term in the index.

The tiling also **rewrites the op's iteration ranges**: the dims at the indices
in `loop_tiled_dims` are divided by the trip count (`_divide_ranges` in
`loop_tiling_common.py`), so that each inner `OpSpec` describes only the work
done per trip. For reduction-dim tiling, `_divide_reduction_ranges`
(`coarse_tile.py`) divides `data.reduction_ranges` instead.

`loop_group_id` is a tuple rather than a flat integer to support nested
loops. See "Nested loops and the `loop_group_id` tree" below.

`loop_count` is redundant across all ops sharing the same `loop_group_id`
(they must agree), but keeping it on each op means the scheduler does not need
a separate side table. `loop_tiled_dims` and `loop_tiled_reduction_dims` are
**per-op**: each op resolves its own tiled position from its own `DimHint`
(`resolve_level_positions` in `coarse_tile.py`, the same function both
lowerings use), so a broadcast op whose iteration space lacks the tiled
dimension gets an empty sub-list and is loop-invariant at that level.

### `Loops` is a frozen dataclass

Inductor's `ir.Loops` (the base of `Pointwise` and `Reduction`) is
declared `@ir_dataclass(frozen=True)`, so `data.ranges = x` raises
`FrozenInstanceError`.  The tiling passes use `object.__setattr__` to
bypass this:

```python
object.__setattr__(data, "ranges", ranges)
```

### Placing ops relative to a loop

A pre-scheduling pass that inserts or moves an op names the op it belongs
beside, and the helpers in `loop_node.py` put it in the list that holds that
anchor: `insert_before`, `insert_after`, `move_before`, `move_after`, and
`remove_op`, plus `insert_at_start` and `insert_on_back_edge` for a
`LoopNode`'s body. When the anchor is a `LoopNode`, the new op goes outside the
loop. `top_level_entry(operations, op)` returns `op` when it is at the top
level, otherwise the outermost `LoopNode` that holds it; an op that must run
once outside every loop of its anchor is placed beside that entry. The
span-overflow lowering places its copy-out, fill, scalar, and combine ops right
after the op they belong to, and the full buffer right before the group's first
op. The LX allocator places a hoisted clone or a drain beside the
`top_level_entry` of its anchor (`GraphEditor._place_clone` in
`scratchpad/graph_editor.py`).
`split_multi_ops` moves each op it creates right before the op it splits, in
the list that holds it. Restickify, padding, mutation-bracket, and
read-copy-elision edits use the same helpers.

### The span-overflow lowering: `tile_groups`

```python
def tile_groups(
    graph: Any, groups: list[tuple], group_idx_offset: int = 0
) -> dict[int, sympy.Symbol]:
```

`tile_groups` (`torch_spyre/_inductor/wsr/loop_tiling_lowering.py`) is the one
entry point of the span-overflow pass (`_maybe_coarse_tile_span_overflow` in
`passes.py`) and of the LX allocator's tiling (`apply_tilings` in
`scratchpad/coarse_tiling.py`). `for_each_tile` does not call it: its
`loop_info` is stamped by `_stamp_direct_loop_info` from the trip count and
loop variable of the traced `ir.WhileLoop`.

`groups` is a list of `(ops, levels)` tuples, where `levels` is a list of
`(hint_id, K)` pairs, outermost first. Each `ops` list must be a contiguous
sub-sequence of `graph.operations`. `span_overflow_groups`
(`coarse_tile_span_overflow.py`) builds them from the ops whose per-core span
overflows the hardware budget (see `span_overflow_hint_analysis.md`); it skips
any op that already carries `dim_hints`, so a `for_each_tile` body never
reaches this path. Both callers first run `validate_coarse_tile_groups`, which
raises `RuntimeError` when one `hint_id` appears in two groups, and
`reject_nested_loops`, which raises `Unsupported` ("nested loop tiling is not
supported in this release", #6088) for a group with more than one level.
`group_idx_offset` keeps the new group IDs clear of the IDs already stamped by
`for_each_tile` loops.

`tile_groups` plans, then lowers:

1. **Plan.** `build_loop_tiling_plan` returns a `LoopTilingPlan` without
   touching the IR. For each group it holds the trip count and, for each op,
   an `OpTiling` with the tiled position (`TiledDim`) or `None` for an op the
   loop runs unchanged. It raises `Unsupported` for a symbolic trip count,
   reduction tiling while `config.enable_reduction_tiling` is off, a mixed
   output and reduction level or more than one reduction dim at a level
   (`_validate_planned_reduction_tiling`), a pointwise op that reads a
   reduction result before it is complete (`_reads_incomplete_reduction`),
   and a matmul tiled across a broadcast batch
   (`_check_matmul_broadcast_batch_tiling`).
2. **Lower.** `lower_loop_tiling` mints one induction variable per group
   (`mint_induction`) and calls `_lower_group`, which reads everything it needs
   (consumers, ranges, strides, steps) from the graph as it stands, not from
   the plan. Its steps, in order:
   - `_validate_contiguous` checks that the members are a contiguous block of
     `operations`, and `_facts` classifies each member (below). A member that
     mutates a buffer, or that already carries a loop-variable hint, raises
     `Unsupported`.
   - Each tiled member's ranges are divided by the trip count and every member
     gets a `LoopShim`.
   - `_lower_reduction` handles each `reduction` member; `_insert_copy_out`
     drains each `copy_out` member into a full buffer.
   - `_patch_retiled_load_indexes` rewrites the stale stride coefficients of
     in-group consumers of a retiled buffer.
   - `_apply_edit` gives every member, copy, and combine the loop variable in
     its `dim_hints` and adds the term `step * u` to each advancing access: a
     read through `_AddLoopTermHandler`, a write through a wrapped
     `store_output`. The step is `_tile_step`: the change of the access's
     index over one tile, read before the divide.
   - Each accumulator becomes a loop carry (`_plan_carries`, `_apply_carries`).

`lower_loop_tiling` returns `{group index: loop variable}`, which
`request_loop_nodes` records; `build_pending_loop_nodes` wraps each group in a
`LoopNode` after the pass returns.

#### Treatment by consumer topology

`_facts` gives each member one of three kinds:

| Kind | When | Treatment |
|---|---|---|
| `loop_internal` | Not tiled, or tiled with no consumer outside the loop and not a graph output | Nothing is built. Its own tile-sized buffer does not advance, so it can be placed in LX. |
| `copy_out` | Tiled on an output dim, and read outside the loop or a graph output | `_allocate_full_buffer` allocates the full buffer before the group's first op; `_insert_copy_out` inserts, right after the member, a `coarse_tile_copy_<buf>` op that reads the tile and writes the full buffer through `MutationLayoutSHOULDREMOVE`. Only the copy's write advances. Outside consumers are patched to read the full buffer (`_patch_consumers`, which traces each consumer through `_ReadThroughLayout` so its index is derived from the full buffer's layout). |
| `reduction` | A `Reduction` tiled on a reduction dim | `_lower_reduction`, below. |

The read side needs no adaptation. A member reads a full input buffer through
its own index, which carries the advance term. A read of a tiled member's own
tile does not advance, because that buffer is rewritten every trip. A member
the loop does not tile may not read the tile of a member it does tile: that
raises `Unsupported`.

Splitting each crossing into the member's own tile-sized output and a separate
copy keeps the member's layout input-derived. The full buffer's device layout
is planned from the member's committed layout at its full size
(`_planned_full_device_layout`), and only the copy has to reconcile the tile's
layout with it. `propagate_mutation_layouts` (the first
`CustomPreFusionPasses` pass) then replaces the copy's
`MutationLayoutSHOULDREMOVE` with a `FixedTiledLayout`.

#### Reduction tiling

A `Reduction` tiled on its reduction dim lowers to an accumulator the size of
the output, filled once before the loop, and a combine op inside the loop:

1. `_allocate_full_buffer` allocates `accum_full` with the full output shape,
   before the group's first op.
2. `_insert_fill` writes the reduction's identity into `accum_full`. The fill
   has no `loop_info`, so it runs once, outside the loop. The identity is a
   `SpyreConstantFallback` scalar with a hand-built `FixedTiledLayout`, because
   layout finalization has already run. The fill is marked
   `_coarse_tile_force_live` (see the DCE subsection of the appendix).
3. `_make_combine_buffer` builds a `coarse_tile_combine_<buf>` pointwise op that
   folds each trip's partial result into `accum_full` in place
   (`MutationLayoutSHOULDREMOVE`). It carries the loop's `LoopShim` and is
   inserted right after the reduction.
4. The reduction records `_tiled_reduction_accum_name`, which
   `finalize_layouts` (`insert_restickify.py`) reads to give the accumulator
   the reduction's device layout.
5. Outside consumers are patched to read `accum_full`.

The accumulator is then declared a loop carry: its storage is `accum_full`
and its update is the combine. `_plan_carries` raises `Unsupported` when the
storage already declares a carry or is a loop member, or when any other member
touches it inside the loop. A member that reads the partial result inside the
loop that accumulates it raises `Unsupported` in `_facts`.

Identity values (`_reduction_identity_value`) and combine operators
(`_make_combine_buffer`) by `reduction_type`:

| `reduction_type` | Identity | Combine |
|---|---|---|
| `sum`, `BATCH_MATMUL_OP` | 0 | `add` |
| `prod` | 1 | `mul` |
| `max` | −∞ | `maximum` |
| `min` | +∞ | `minimum` |
| `xor_sum` | 0 | `bitwise_xor` |
| `any` | 0 | `logical_or` |

Any other reduction type (for example `argmin` or `argmax`) raises
`RuntimeError`.

### Pass placement

The full pre-scheduling pass list is in
[LoopLevelIR Passes](inductor_frontend.md#looplevelir-passes). The constraints
that place the two coarse-tiling slots:

**`splice_while_loops` runs second, after `apply_reserved_dims_contract` and
ahead of dead-code elimination.** It must prove each `for_each_tile`
`ir.WhileLoop` bounded and splice its body into `graph.operations` before any
other pass treats those ops as ordinary IR. It needs neither device layout nor
span arithmetic.

**The span-overflow lowering runs after stickification and padding.**
`_maybe_coarse_tile_span_overflow` needs `FixedTiledLayout.device_layout`
(device size, stride map) for span arithmetic, and must see the
post-stickify, post-padding shapes or it splits on the wrong dimension or
produces a non-stick-aligned inner size. It is skipped unless
`SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS=0`.

**Both run before work division.** `work_distribution` commits
`iteration_space_ownership` on each op, and it must see the reduced per-trip
iteration space so that cores divide the per-trip work. Each division computes
the op's span floors from the op as it is then, so they see the tiled
iteration space.

**`scratchpad_planning` runs after coarse tiling**, because it sizes
scratchpad allocations to the per-trip working set.

## Layer 2: `CountedLoopSchedulerNode`

Every counted loop, from `for_each_tile` or from the span-overflow lowering,
enters the scheduler as a `LoopSchedulerNode` (`loop_scheduler.py`), the
subclass of `CountedLoopSchedulerNode` that carries the loop's `LoopNode`
through the scheduler's dependency analysis and topological sorts via
per-compile patches on Inductor's `Scheduler`. Because a `LoopSchedulerNode` is
a `CountedLoopSchedulerNode`, `build_loop_scheduler_nodes` passes it through
unchanged, and fusion, kernel preparation, HBM-pool planning, and codegen all
operate on the counted loop described below. `build_loop_scheduler_nodes`
builds a `CountedLoopSchedulerNode` from flat `loop_info`-carrying nodes only
when the `LoopNode`s were expanded before the Scheduler
(`loop_scheduler.SCHEDULE_LOOP_NODES` off, which only the equivalence test
does).

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
constituent `SchedulerNode`s before codegen.  `can_fuse` returns `False`:
a loop group is atomic; nothing can be fused into it from outside.

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
which is registered as the **second pass in `CustomPreFusionPasses`**,
running before Inductor's own fusion pass:

```python
class CustomPreFusionPasses(_SpyreNodePassPipeline):
    def __init__(self):
        super().__init__([propagate_mutation_layouts, build_loop_scheduler_nodes])

class CustomPostFusionPasses(_SpyreNodePassPipeline):
    def __init__(self):
        super().__init__(
            [sink_collective_waits, spyre_fuse_nodes, prepare_spyre_kernels,
             hbm_pool_planning]
        )
```

**`build_loop_scheduler_nodes` must run before Inductor's fusion pass and
before `spyre_fuse_nodes`.**  Placing it in `CustomPreFusionPasses` means
`CountedLoopSchedulerNode`s are already present when Inductor calls
`can_fuse_vertical` / `can_fuse_horizontal` on `SuperDSCScheduling`
(both return `False`), so loop groups are never split by Inductor's own
fusion logic.  `spyre_fuse_nodes` accumulates a `CountedLoopSchedulerNode`
alongside plain `SchedulerNode`s, so a loop can share a bundle with adjacent
ops, but it never splits the loop's members.  `can_fuse = False` on
`CountedLoopSchedulerNode` provides a belt-and-suspenders guard against
any future fusion path that might otherwise merge across group boundaries.

### The grouping algorithm

`build_loop_scheduler_nodes` first calls `_regroup_by_outer_loop_key`, then
scans the resulting node list and groups contiguous runs sharing the same
outermost `loop_group_id` key. The regroup step is necessary because
Inductor's own `Scheduler.topological_sort_schedule` runs (twice) before
this pass ever sees the node list, via a plain DFS over
`unmet_dependencies`, that DFS only guarantees a *valid* topological
order, not that mutually independent nodes keep their original relative
order, so it can interleave unrelated nodes into the middle of what
a lowering built as a single contiguous loop group.
`_regroup_by_outer_loop_key` merges every node sharing an outermost
`loop_group_id[0]` key into one virtual unit (dependency set = the union of
its members' real cross-group dependencies), runs a dependency-respecting
DFS over `{merged units, ungrouped nodes}`, then expands each unit back
into its original members in their original relative order, restoring
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
loop group's `SchedulerNode`s stay contiguous, it only guarantees a valid
order among mutually independent nodes, which can interleave. Contiguity
is restored by `_regroup_by_outer_loop_key` before grouping runs. If
`build_loop_scheduler_nodes` still finds a non-contiguous run after that
call, it means either a bug in `_regroup_by_outer_loop_key` itself, or a
genuine data-flow constraint that makes the group's own op sequence
topologically invalid (which would be a tiling-pass bug). The pass raises
in that case.

## Layer 3: `LoopSpec` and codegen

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
because it has no `iteration_space`, `args`, or `op_info` of its own, those
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
`(max, granularity)` bounds for dynamic-shape dims. `create_op_spec` no
longer populates it: it refuses a `mark_dynamic(max=...)` size outside a
`for_each_tile` loop, so the field is empty in every compile.

`OpSpec.tiled_symbol_trip_counts` maps each minted level symbol appearing
in `tiled_symbols` to that level's own trip count
(`CoarseTileInfo.loop_count` for that level); it lets downstream codegen
recover a level's full (untiled) extent as `(per-step device-element
advance) * trip_count` without a separately tracked extent field on
`TensorArg`. Only correct when a symbol belongs to exactly one nesting
level.

`TensorArg.device_tile_advance_expr` is a single `sympy.Expr | None`,
computed independently for **each** `TensorArg` (not shared across the
op's `args`) by `SpyreKernel._tile_advance`. `_index_tile_advance` reads, for
each level, the coefficient of the level's loop variable in the access's
`MemoryDep` index, projects it to device elements
(`tiling_expr_to_device_expr`), and multiplies it by that level's symbol from
`_get_or_mint_level_symbol`. The result is `None` for an op in no loop, an op
that lists no loop variable, and an access whose index does not move with the
loop. An op stamped with a `LoopShim` that has no advance in its index raises
`RuntimeError`, because the span-overflow lowering lists the loop variable on
every member. Each level's term is picked out by `expr.coeff(sym)` on that
level's own symbol, not by list position the way `tiled_symbols` is; because
each level has its own symbol, two levels that tile the same host dim stay
separate addends.

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
`loop_count = [K1, K2]`, the scheduler reads `loop_count[0] = K1` when
building the outer `CountedLoopSchedulerNode` and `loop_count[1] = K2`
when building the inner one.  This allows a single op to supply the counts
for all its enclosing loops without requiring sibling ops at intermediate
depths.

When the flat nodes reach `build_loop_scheduler_nodes`, the pre-fusion
pass (`_build_loop_group`) reconstructs the tree recursively:

1. Group the flat `SchedulerNode` list into runs that share the same
   outermost group id element (index `depth`).
2. Read the count for this depth from `_loop_count(node, depth)`, which
   returns the trip count of the node's enclosing loop at that depth
   (`scheduler_enclosing_loops`).  All nodes in the run must agree on this
   count.
3. Recursively call `_build_loop_group(run, depth + 1)` to build the
   inner level.
4. Wrap the result in a `CountedLoopSchedulerNode` with that count.

Because every op carries the full `loop_count` list, the algorithm works
even when a run contains only a single op that spans all nesting levels,
there is no need for placeholder ops at intermediate depths.

### Bundle boundary constraint

A `CountedLoopSchedulerNode` (at any nesting depth) and all its
descendant `SchedulerNode`s must be codegen'd into a **single SuperDSC
bundle**, i.e., a single `codegen_node()` call must produce the entire
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

### Preparing and emitting counted-loop kernels

`prepare_kernel` handles both ordinary bundles and `CountedLoopSchedulerNode`s
before HBM-pool planning. `_codegen_into_kernel` runs leaf operations through the
existing handlers; `_codegen_loop_body` recursively adds nested loops to the same
kernel, wrapping only their newly added `op_specs` entries in an inner `LoopSpec`.
For the outer counted loop, `prepare_kernel` calls `wrap_op_specs_in_loop` once.

`SpyreKernel.wrap_op_specs_in_loop(count)` replaces the flat `self.op_specs`
list with `[LoopSpec(count=count, body=self.op_specs)]`.
`generate_node_schedule` flattens any fused nodes inside the loop group into
leaf `SchedulerNode`s.

`codegen_node` consumes this same prepared kernel. It does not rebuild the loop
body: it binds the final HBM pool size, addresses, and argument list, then emits
one SuperDSC bundle for the entire `LoopSpec` tree.

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

`OpSpec.tiled_symbols` is populated by `SpyreKernel.create_op_spec` from the
op's enclosing loops (`enclosing_loops`). A level gets an entry when anything
advances at it (`advancing_levels`) or when the op tiles an output or
reduction dim there (`loop_tiled_dims`, `loop_tiled_reduction_dims`). The
entry is the level's single symbol from `_get_or_mint_level_symbol`, the same
symbol `_index_tile_advance` uses for each `TensorArg`, and
`tiled_symbol_trip_counts` maps it to the level's trip count. The result is
stored innermost-first. A dim tiled down to an extent of 1 is squeezed out of
the op's iteration space and advances only through the index, which is why a
level is named whenever anything advances at it: otherwise the symbol would
appear in `device_tile_advance_expr` but not in `tiled_symbols`, and
`superdsc.py`'s affine-stride extraction, which walks only `tiled_symbols`,
would drop the advance.

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
`codegen/bundle.py` emits the loop intact, an `scf.for` wrapping, for each
tiled tensor, an `affine.apply` that computes the per-iteration HBM address
from the loop induction variable(s), followed by `sdsc_execute`, as shown in
the bundle.mlir section above.  `device_size` stays at the per-tile shape and
`tiled_symbols` records which iteration-space symbols the enclosing loop
levels advance; tensors whose `device_tile_advance_expr` is `None` (e.g. LX
scratchpad operands whose index has no loop-variable term) are
skipped entirely, no `affine.apply` is emitted for them since their base
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
| `torch_spyre/_inductor/loop_info.py` | Layer 1: `CoarseTileInfo` and `LoopShim` dataclasses, `COUNTED_LOOP_INFO`, `LoopCarryRecord`; `copy_op_metadata` |
| `torch_spyre/_inductor/loop_facts.py` | Layer 1: loop-structure accessors (`enclosing_loops`, `scheduler_enclosing_loops`) and the index-derived advances (`read_advance`, `output_advance`, `advancing_levels`) |
| `torch_spyre/_inductor/loop_node.py` | Layer 1: `LoopNode` structural node; `collapse` wraps each loop's ops, `request_loop_nodes`/`build_pending_loop_nodes` wrap the span-overflow groups, `OpsView` presents the body flat to passes, the edit helpers (`insert_before`, `insert_after`, `move_before`, `move_after`, `remove_op`, `top_level_entry`) place ops relative to loops, `expand`/`peel_non_members` for the scheduler hand-off |
| `torch_spyre/_inductor/loop_scheduler.py` | Layer 2: `LoopSchedulerNode` (a `CountedLoopSchedulerNode` subclass) and the per-compile `Scheduler` patches; `SCHEDULE_LOOP_NODES`, `loop_scheduler_supported` |
| `torch_spyre/_inductor/wsr/loop_tiling_lowering.py` | Layer 1 (span-overflow path): `tile_groups`, `build_loop_tiling_plan` (`LoopTilingPlan`), `lower_loop_tiling`; divides ranges, stamps `LoopShim`, inserts copy-outs and reduction accumulators, adds the loop terms to the indexes |
| `torch_spyre/_inductor/wsr/coarse_tile.py` | Layer 1: helpers shared by both lowerings: `validate_coarse_tile_groups`, `reject_nested_loops`, `resolve_level_positions`, `_divide_reduction_ranges`, `_allocate_full_buffer`, `_make_combine_buffer`, `_patch_consumers` |
| `torch_spyre/_inductor/wsr/loop_tiling_common.py` | Layer 1: `_divide_ranges`, `_patch_retiled_load_indexes`, `_AddLoopTermHandler`, `_ReadThroughLayout`, `_hint_ranges_pos` |
| `torch_spyre/_inductor/insert_restickify.py` | `finalize_layouts` commits each op's chosen `FixedTiledLayout` and, for a tiled-reduction op, propagates that layout onto `accum_full` so fill/combine/copy all agree on device coordinates; also stamps a restickify node's `loop_info` from the op it feeds so the node lands in the same loop group |
| `torch_spyre/_inductor/scheduler.py` | Layer 2: `CountedLoopSchedulerNode`, `build_loop_scheduler_nodes`, `prepare_kernel`, `_codegen_loop_body`, `_regroup_by_outer_loop_key` |
| `torch_spyre/_inductor/op_spec.py` | Layer 3: `LoopSpec` and `OpSpec` dataclasses |
| `torch_spyre/_inductor/spyre_kernel.py` | Layer 3: serializes `LoopSpec` tree in `codegen_kernel()`; `wrap_op_specs_in_loop()` |
| `torch_spyre/_inductor/codegen/bundle.py` | Layer 3: emits `scf.for` wrapping `affine.apply`/`sdsc_execute` in `bundle.mlir` |
| `torch_spyre/_inductor/passes.py` | Wires all passes into `CustomPreSchedulingPasses` and `CustomPreFusionPasses` |
| `torch_spyre/_inductor/propagate_hints.py` | `spyre_hint()` context manager (`work_div` and the named-dim keys; the removed tiling keys raise `ValueError`); `DimHint`; hint collection/recovery across AOT re-tracing |
| `torch_spyre/_inductor/wsr/propagate_named_dims.py` | `propagate_named_dims()` and `assign_dim_hints()`: attach `dim_hints` to `ir.Operation` objects |
| `torch_spyre/_inductor/wsr/for_each_tile.py` | `for_each_tile()` frontend: thin wrapper over `torch._higher_order_ops.scan.scan`; `Gather`/`Kind`/`TileSpec` types |
| `torch_spyre/_inductor/wsr/for_each_tile_lowering.py` | Layer 1 (for_each_tile path): `try_prove_for_each_tile`/`_extract_trip_count` prove the static trip count; `splice_while_loops` splices the traced `ir.WhileLoop` body into `graph.operations`, calls `_stamp_direct_loop_info`, and ends by calling `collapse` to wrap the loop's ops in a `LoopNode` |
| `torch_spyre/_inductor/wsr/while_loop_bridge.py` | Carry-rewiring for `for_each_tile`'s accumulator (`carry`) outputs, the true-recurrence mechanism |
| `tests/inductor/test_coarse_tiling.py` | Unit tests: IR pass, propagation, scheduler node, bundle MLIR output |
| `docs/tools/capture_for_each_tile_ir.py` | Regenerates this doc's for_each_tile Small Example snippets from real compiler output |

## Invariants and failure modes

**Contiguity invariant**: all ops sharing a `loop_info.loop_group_id` must be
contiguous. `_lower_group` checks this before it lowers a group, via
`_validate_contiguous`, which raises `RuntimeError` if the ops are not a
contiguous slice of the operation list. Each loop is then one `LoopNode`, so
its members stay together through the pre-scheduling passes, and
`build_loop_scheduler_nodes` raises `RuntimeError` if a loop group is still not
contiguous after `_regroup_by_outer_loop_key`.

**Consistent `loop_count`**: all ops sharing a `loop_group_id` must agree on
`loop_info.loop_count` at every depth level.  The post-fusion pass asserts
this.

**`tiled_symbols` populated iff inside a loop**: `OpSpec.tiled_symbols` is
non-empty exactly when the op was codegen'd inside a `CountedLoopSchedulerNode`.
It is a `list[list[Symbol]]` (innermost first) derived from the per-level
tiled dims in `loop_info.loop_tiled_dims` on the corresponding
`ir.Operation`, selected from the scheduler-level `iteration_space` keys.

**Pass ordering**: the span-overflow lowering must run after stickify/padding
and before `cost_model_matmul_division`, `work_distribution`, and
`scratchpad_planning`. `build_loop_scheduler_nodes` must run in
`CustomPreFusionPasses` (before Inductor's own fusion pass and before
`spyre_fuse_nodes`); see [Pass placement](#pass-placement) above.
`for_each_tile`'s `splice_while_loops` is the second
`CustomPreSchedulingPasses` pass, after `apply_reserved_dims_contract` and
ahead of `deadcode_elimination`: it must prove the trip count and splice the
loop body into `graph.operations` before any other pass sees those ops as
ordinary IR.

**Cache invalidation**: the source files of every pre-scheduling pass,
including the real passes behind each `@_runs(...)` wrapper (for example
`tile_groups` behind `_maybe_coarse_tile_span_overflow`), are included in
`CustomPreSchedulingPasses.uuid()`
so the Inductor FX cache is invalidated when any pass changes.

## Appendix: How IR rewiring works, and why it's sound

The sections above describe *what* the span-overflow lowering does
semantically: which buffers get promoted to full size, which get an `identity`
copy op, and which stay loop-internal scratch with no advance. This appendix
describes *how* those outcomes are implemented as edits to live Inductor IR
objects, and why those edits cannot violate Inductor's own scheduler and
dependency-tracking invariants. It is written for developers who need to modify
`loop_tiling_lowering.py`, `coarse_tile.py`, or `loop_tiling_common.py`, or
diagnose a wrong-code bug that might originate there, not a restatement of the
kind classification or the reduction accumulator pattern, both covered above.

**Scope note**: the copy-out, fill, and combine insertion below belongs to the
span-overflow lowering (`tile_groups`). The `for_each_tile` lowering
(`splice_while_loops` in `for_each_tile_lowering.py`) stamps `loop_info` onto
ops already produced by tracing `scan` and has no read-copy staging (see the
Small Example above), but it reuses `_divide_ranges`,
`_patch_retiled_load_indexes`, and `_NameSwapHandler` from
`loop_tiling_common.py`, so the wrap-never-reconstruct and index-remapping
subsections apply to it too.

### The wrap-never-reconstruct convention in practice

AGENTS.md states the rule plainly: *"Modifying `ComputedBuffer.inner_fn`:
wrap, never reconstruct. Use a `WrapperHandler` subclass ... installed with
`V.set_ops_handler(handler)` inside the original `inner_fn`."* The reason is
that `inner_fn` closes over symbolic index expressions computed against a
specific `ranges`/`reduction_ranges`; those expressions go stale the moment
anything about the op's shape changes, so hand-rebuilding them from scratch
is a silent wrong-code trap (issue #2797, cited directly in
`replace_computed_buffer_body`'s implementation comment in
`pass_utils.py`). Every rewrite site in the coarse-tiling modules and
`insert_restickify.py` follows the same idiom instead:

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
`@ir_dataclass(frozen=True)`, a plain `data.inner_fn = new_inner_fn` raises
`FrozenInstanceError`. This is the same escape hatch the doc already uses for
`_divide_ranges`'s `object.__setattr__(data, "ranges", ranges)` above. By
contrast, `Buffer` (the base of `ComputedBuffer`, which holds `.layout`) is
**not** frozen, so the `op.layout = MutationLayoutSHOULDREMOVE(...)`
assignments used elsewhere in this appendix are ordinary attribute sets, not
escape-hatch writes, the two mechanisms look similar but rest on different
class-level decisions.

`replace_computed_buffer_body` (in `pass_utils.py`) is the second half
of the idiom: because `ComputedBuffer` itself is also frozen, the mutated
`data` cannot simply be re-attached to the existing `op` object either, a
fresh `ComputedBuffer` is constructed with the new `data`, all metadata
fields downstream passes depend on (`operation_name`, `origins`,
`origin_node`, `_split_size`/`_original_*`) are copied across, the
`get_default_sizes_body` cache is explicitly cleared on the new object, and
the new buffer replaces the old one in `operations` by index. Every inner_fn
rewrite site ends with this call, not a raw dataclass mutation, specifically
so that stale per-object caches on the old buffer can never leak forward.

Call sites, all following this exact shape:

- `_patch_consumers` (in `coarse_tile.py`), patches an outside consumer's `inner_fn` to
  read the newly-promoted full buffer instead of the original tile-sized one.
  For a copy-out it also traces the consumer with the producer's layout
  presenting the full buffer's shape (`_ReadThroughLayout`), so the index is
  the full buffer's by construction; for a reduction accumulator it rescales
  the tile-traced index instead (`_NameAndIndexSwapHandler`).
- `_patch_retiled_load_indexes` / `_RetileLoadIndexHandler`
  (both in `loop_tiling_common.py`), a distinct
  mechanism from name-swapping, detailed in the next subsection.
- `insert_restickify_on_node_inputs` (in `insert_restickify.py`, using
  the canonical `NameSwapHandler` defined in the same file),
  the example AGENTS.md itself points to.

Two other edits look like exceptions but are not. `_insert_copy_out`
(in `loop_tiling_lowering.py`) builds a **new** `Pointwise` via
`tiled_op.make_loader()` rather than editing `tiled_op`'s own `inner_fn`.
This is IR-safe by construction, not a violation of the convention. It
reuses Inductor's own `make_loader()` (which itself returns a closure over
the *existing* `inner_fn`/index machinery) instead of hand-assembling an
index expression, so the same "never reconstruct a stale index" property
holds even though no `WrapperHandler` is involved. `_apply_edit` (in
`loop_tiling_lowering.py`) adds the loop term to a read by wrapping `inner_fn`
with `_AddLoopTermHandler`, and to a write by wrapping `store_output` so the
indexer it passes returns `indexer(idx) + term`; neither rebuilds an index.

No site reconstructs an index expression from scratch.
`_divide_ranges` (in `loop_tiling_common.py`) is the one place shape and
layout are mutated (via `object.__setattr__`) with `inner_fn` left completely
untouched, deliberately, and safely, for the reason given in the next
subsection.

### Index-expression remapping: `_divide_ranges` and `_patch_retiled_load_indexes`

Two distinct mechanisms handle index-expression correctness after tiling,
and they are staged deliberately rather than combined:

1. **`_divide_ranges`** (in `loop_tiling_common.py`) shrinks `data.ranges`
   (and the op's own `layout.size`/`layout.stride`) via `object.__setattr__`,
   leaving `inner_fn` completely untouched. This is correct because the op's
   own index arithmetic is expressed in terms of the loop variables that the
   surrounding (now smaller, per-tile) iteration space binds, the *op*
   never needs to know it was tiled; only its bounds shrink. In the
   span-overflow path `_divide_ranges` runs only from `_lower_group`;
   planning (`build_loop_tiling_plan`) never mutates `data.ranges`, it
   computes the per-tile extents analytically via
   `_planned_tile_extents_per_level`, reading the still-untouched
   `data.ranges`/`data.reduction_ranges`.

2. **`_patch_retiled_load_indexes`** (in `loop_tiling_common.py`) fixes a
   different problem: *other* ops in the same loop whose captured load index
   still carries the pre-tiling stride coefficient for a buffer that has since
   been retiled. `_lower_group` calls it once per group, after the group's
   ranges are divided and its copy-outs and accumulators are inserted.
   `_should_patch_retiled_load_indexes` selects only the ops that were already
   members of the group when it was divided; an op inserted later (such as the
   copy from `_insert_copy_out`) reads the retiled buffer's updated layout
   directly, and rewriting it would apply the stride correction twice.
   `_RetileLoadIndexHandler` (a `WrapperHandler` subclass) applies the
   substitution to the consumer's `inner_fn`, following the same
   wrap-never-reconstruct idiom as every other site in this appendix.
   `_retile_load_index` rewrites each index through `compute_tile_index` and
   raises `Unsupported` when the index cannot be decomposed (non-affine, or a
   stride not in the old layout).

### Read redirection: why a view buffer, not just an index edit

A recurring temptation when redirecting a read is to think of it as "leave
`inner_fn` alone, just edit the dependency the scheduler sees." That is not
possible in Inductor: as the next subsection proves in detail, dependency
information is *derived from* `inner_fn` by re-tracing it, not stored
independently, so the only way to actually redirect what an op reads is to
change what its `inner_fn` does when traced.

`_patch_consumers` (in `coarse_tile.py`) is the concrete instance: when an
outside consumer reads a tile-sized buffer that the copy-out path has just
promoted to a full-size buffer, the consumer's `inner_fn` is wrapped so its
load of the old buffer name is retargeted to the new full buffer's name. For a
copy-out the consumer is traced through `_ReadThroughLayout`, so the producer's
layout presents the full buffer's host shape and the index is derived from the
full buffer by construction; for a reduction accumulator
`_NameAndIndexSwapHandler` rescales the tile-traced index instead.

### `MutationLayoutSHOULDREMOVE`: the real contract

The doc above uses `MutationLayoutSHOULDREMOVE` several times (the copy-out
op's output, and the fill and combine of the reduction accumulator) as
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
on the target buffer's name, mutation is registered at construction time,
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
buffer" pattern; torch-spyre does not call it, every call site below
constructs `MutationLayoutSHOULDREMOVE` directly and assigns it to `.layout`.)

Marking the mutation matters beyond bookkeeping:
`ComputedBuffer.make_loader()` checks `self.name not in
V.graph.mutated_buffers` before deciding it is safe to inline a buffer's
computation into its consumer. Mutation marking is exactly what prevents
Inductor from incorrectly inlining away a buffer that is actually written in
place, without the constructor's `mark_buffer_mutated` call, nothing would
stop Inductor from treating the mutating op as a pure, inlinable pointwise
computation and silently dropping the in-place write.

**The single-writer invariant.** `Buffer.get_mutation_names()`
(`ir.py:4574-4577`) returns at most one name, `ComputedBuffer` inherits it
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
called from `Scheduler._init`, i.e. it runs before the first topological
sort, before dead-code elimination, before any torch-spyre
`CustomPreFusionPasses` hook fires. Every torch-spyre call site that assigns
a `MutationLayoutSHOULDREMOVE` satisfies this by construction, `.layout` is
a single attribute, and no site chains a new `MutationLayoutSHOULDREMOVE`
onto a target that already carries one:

| Site | File | Target |
|---|---|---|
| `_insert_copy_out` | `loop_tiling_lowering.py` | full buffer (copy-out) |
| `_make_combine_buffer` | `coarse_tile.py` | `accum_full` (per-trip combine) |
| `_insert_fill` | `loop_tiling_lowering.py` | `accum_full` (identity-value seed) |

This was checked directly against the current codebase and no violation was
found, but the invariant is currently upheld by convention (one assignment
per op, never revisited), not by an assertion or type-level guard. If this
pattern is ever extended to a new call site, it is worth adding an explicit
check rather than relying on the same discipline holding indefinitely.

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

**Stamping and unwrapping are separate steps.** The span-overflow lowering
*stamps* `MutationLayoutSHOULDREMOVE` on its copy-out, fill, and combine ops
pre-scheduling, inside `CustomPreSchedulingPasses`, before any `Scheduler`
object exists. `propagate_mutation_layouts` (pre-fusion, the first entry in
`CustomPreFusionPasses`) *unwraps* `MutationLayoutSHOULDREMOVE` back to a real
`FixedTiledLayout`, after `Scheduler.__init__` has already consumed the
mutation-marked state.

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

`extract_read_writes()` (`dependencies.py:659-693`), for this call path,
where `fn` is `self.get_store_function()`, a `partial`, not a `LoopBody`,
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
`V.set_ops_handler`, and literally re-invokes the store function, which
re-invokes `inner_fn`, from scratch. Upstream holds no memoized `ReadWrites`
object in this path. Mutating `op.data.inner_fn` in place is therefore
reflected the next time anything calls `get_read_writes()`, and Inductor's
`Scheduler` cannot observe stale dependency info, because
`SchedulerNode.read_writes` is built once, at `Scheduler.__init__` time, which
runs after all pre-scheduling IR rewriting has completed.

torch-spyre adds one cache of its own: `pass_utils.op_read_writes(op)` memoizes
`op.get_read_writes()` on the op instance, because the pre-scheduling passes
and the scratchpad planner ask for it very often. A pass that changes what an
op reads or writes on the same instance must call
`invalidate_op_read_writes(op)` right after the edit. `_apply_edit` and
`_patch_retiled_load_indexes` both do; `replace_computed_buffer_body` builds a
new instance, which starts with no memo.

`pass_utils.py`'s own comment at the `replace_computed_buffer_body` call
site is the project's own prior articulation of this exact argument: *"Always
wrap the original inner_fn via WrapperHandler; never rebuild index
expressions from scratch (they go stale; see issue #2797)."*

### DCE liveness: why reduction copy-outs survive

`Scheduler.dead_node_elimination` (`scheduler.py:3528-3567`) is a single
reverse-topological-order linear sweep, not a separate reachability
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
(`scheduler.py:2953`), strictly **before** `CustomPreFusionPasses` fires
(step 14, `scheduler.py:2966-2967`) and never again afterward in `_init`.
This is the fact that matters for correctness: any liveness protection a
torch-spyre pass wants to apply must already be in place by the time this
sweep runs, not applied afterward, `CustomPreFusionPasses` is too late to
save a node DCE has already dropped.

The problem this creates: the fill that `_insert_fill`
(`loop_tiling_lowering.py`) places before a tiled reduction's loop writes the
reduction's identity into `accum_full`. Its write is read only by the first
trip's combine, a dependency across iterations that the flat scheduler IR DCE
walks does not show before loop codegen groups the body under an `scf.for`.
From DCE's perspective the fill's output has no live user, and it would be
removed even though it is required for correctness.

The fix is a targeted monkeypatch inside `enable_spyre_context` in
`torch_spyre/_inductor/patches.py`. `_insert_fill` marks the fill with
`_coarse_tile_force_live`, and the patch makes `SchedulerNode.has_side_effects()`
report `True` for a node so marked:

```python
    old_scheduler_node_has_side_effects = SchedulerNode.has_side_effects

    def _spyre_scheduler_node_has_side_effects(self: SchedulerNode) -> bool:
        if getattr(self.node, "_coarse_tile_force_live", False):
            return True
        return old_scheduler_node_has_side_effects(self)

    SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects
```

The patch's own comment already draws the right analogy: this is the same
technique upstream Inductor uses to keep effectful `FallbackKernel`s (ops
with observable side effects but no reader) alive across the same DCE pass,
`has_side_effects()` is precisely the escape hatch DCE consults
(`can_eliminate = not node.has_side_effects() and not active_buffers`,
quoted above) for exactly this situation.

The patch's scope is narrow, which matters for developer confidence that it
cannot mask an unrelated bug elsewhere: it patches `SchedulerNode.
has_side_effects` specifically, not `BaseSchedulerNode`, not
`ExternKernelSchedulerNode`, not `FusedSchedulerNode`, and even for
`SchedulerNode` it falls through unchanged to the original (`@cache_on_self`-
decorated) implementation (`scheduler.py:1818-1823`) for every node except
the ones explicitly stamped. The `_coarse_tile_force_live` attribute is
stamped at one site: the fill buffer in `_insert_fill`
(`loop_tiling_lowering.py`).

### Summary: invariant-by-invariant soundness table

This table is additive to the [Invariants and failure modes](#invariants-and-failure-modes)
section above, not a replacement for it, that section covers loop-structure
invariants (contiguity, consistent `loop_count`, pass ordering); this one
covers the IR-rewrite mechanism this appendix describes.

| Inductor invariant | Where enforced upstream | How torch-spyre's rewiring respects it |
|---|---|---|
| Dependencies must reflect `inner_fn` | `get_read_writes()` re-traces every call, no cache (`ir.py:4768`) | Upstream has no cache to go stale; torch-spyre's `op_read_writes` memo is dropped with `invalidate_op_read_writes` after each in-place edit |
| ≤1 mutation target per op | `assert` at `scheduler.py:3337` | Every `MutationLayoutSHOULDREMOVE` call site assigns exactly one; `.layout` is a single attribute, never chained |
| Mutated buffers must not be silently inlined | `mark_buffer_mutated` called unconditionally in the constructor (`ir.py:4383`) | Constructor call fires on every instantiation, before `make_loader()` can ever see a stale view |
| Dead nodes are pruned before codegen | `dead_node_elimination`, `scheduler.py:3528`, runs once, before `CustomPreFusionPasses` | `_coarse_tile_force_live` + patched `has_side_effects()` (in `patches.py`) protects the reduction fill that needs it |
| Loop-group contiguity after scheduling | (existing invariant, cross-referenced only) | See [Contiguity invariant](#invariants-and-failure-modes) above |
