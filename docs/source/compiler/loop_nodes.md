# Counted Loops in the Spyre Compiler: `LoopNode`

## Background

Spyre's compilation pipeline runs a sequence of optimization passes over
`ir.Operation` objects in `CustomPreSchedulingPasses`, before Inductor's
`Scheduler` is constructed. One optimization is **coarse-level tiling**: take a
sequence of operations that share an iteration-space dimension, split that
dimension into K chunks (K may be a symbolic shape), and run the body once per
chunk inside a counted loop. This is the key program transformation for working
set reduction: it tiles the computation in the time domain, so that most tensors
fit in the scratchpad.

The loop has to survive three consumers:

1. Inductor's `Scheduler`, which wraps each `ir.Operation` in a `SchedulerNode`.
2. Spyre's `SuperDSCScheduling.codegen_node()`, which drives `SpyreKernel` to
   produce `OpSpec` objects.
3. Downstream SDSC compilation, which needs an explicit loop count to generate
   correct hardware instructions.

This document describes how a loop is represented, built, edited, scheduled and
emitted. For the motivation, and the constraints that shaped the design, see the
companion RFC
[1358-CoarseTiling](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md).
For the user-facing way to write a loop, see
[Working Set Reduction](working_set_reduction.md).

**Quick navigation:**

- [Overview](#overview)
- [Worked example](#worked-example)
- [Where loops come from](#where-loops-come-from)
- [The node](#the-node)
- [Reading loop facts](#reading-loop-facts)
- [Editing a graph that holds loops](#editing-a-graph-that-holds-loops)
- [The loop checker](#the-loop-checker)
- [Carries](#carries)

## Overview

A counted loop is one `LoopNode`, an `ir.Operation` held in `graph.operations`.
Its `body` holds the loop's ordinary ops and any nested `LoopNode`s, and the
node owns every fact that is one per loop (trip count, induction variable,
carries) and the facts about each member (which dims it tiles). The same
structure carries the loop from the first pass after the lowering to codegen.

Two lowerings produce loops. Both end by building nodes:

:::{figure} ../_static/images/wsr/loop-node-pipeline.svg
:alt: From a lowering to a LoopNode, a LoopSchedulerNode, a LoopSpec and an scf.for
:width: 60%
:align: center

Both lowerings put a `LoopMember` on each op; the node build turns those into one
`LoopNode`, which owns the loop facts. The passes read it with `loop_facts`, edit it
with the `loop_node` helpers and are checked by `loop_checker`. It is handed to the
scheduler as one `LoopSchedulerNode`, becomes one `LoopSpec` in codegen, and is
emitted as an `scf.for` with late-bound addresses.
:::

| Stage | Loop identity | Form |
|---|---|---|
| Lowering | `op.loop_member`, a `LoopMember` on each op the lowering puts in a loop | per-op record, temporary |
| Pre-scheduling passes | `LoopNode` in `graph.operations`; each leaf op points at its nodes | structural node |
| Scheduler | `LoopSchedulerNode`, a `CountedLoopSchedulerNode` subclass | one wrapper per loop |
| Codegen | `LoopSpec` | serializable tree node |

The lowering's record is consumed when the nodes are built and then deleted, so
from the first pass after the build the nodes are the only owner of the loop
facts. Every reader asks the nodes through `loop_facts`, and every pass that
changes the graph says where an op goes through the helpers in `loop_node.py`.

## Worked example

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

This tiles the 1024 rows into 8 tiles of 128 rows. Each iteration processes a
128 x 4096 tile, so the intermediate `y_tile` can stay in the scratchpad across
both operations. `for_each_tile` expresses one loop level per call; see
[Composing and nesting](working_set_reduction.md#composing-and-nesting) for how a
second, nested level is built by calling it again inside `body`. A runnable
script is
[`for_each_tile.py`](https://github.com/torch-spyre/torch-spyre/blob/main/docs/source/user_guide/examples/for_each_tile.py).

Every snippet below is real, captured output of this example. They are produced
by `docs/tools/capture_for_each_tile_ir.py`; regenerate them with that script
when compiler internals change (see `docs/tools/README.md`).

### The node, as the nodes are built

When the nodes are built, the loop's three body ops are wrapped in one
`LoopNode`, and the node records the facts:

```text
loopnode0: group_id=0 trip_count=8 induction=u0 strict_loop_vars=False carries=0
  tiling = {'while_loop_body_graph_0_0_buf8':  ((0,), ()),
            'while_loop_body_graph_0_0_buf9':  ((0,), ()),
            'while_loop_body_graph_0_0_buf11': ((0,), ())}
  while_loop_body_graph_0_0_buf8:  chain=['loopnode0']   # y_tile = a_tile + b_tile
  while_loop_body_graph_0_0_buf9:  chain=['loopnode0']   # z_tile = y_tile * c_tile
  while_loop_body_graph_0_0_buf11: chain=['loopnode0']   # the copy into the output
```

`tiling` maps each member's buffer name to `(tiled_dims, tiled_reduction_dims)`
at this level: here every member tiles output dim 0 and no reduction dim. `chain`
is the member's `_loop_chain`, the nodes around it, outermost first. The
lowering's `loop_member` record is deleted from the three ops once the node
holds the same facts.

### The `graph.operations` dump

After the pre-scheduling passes, the loop is one entry in the operation list.
`loop_levels` is what `loop_facts.enclosing_loops` reports for each op:
`(group path, trip count, tiled dims, tiled reduction dims)` per level.

```text
loopnode0: LoopNode trip_count=8 induction=u0 carries=0

  while_loop_body_graph_0_0_op8: ComputedBuffer
    buffer=while_loop_body_graph_0_0_buf8
    layout=FixedTiledLayout('spyre:0', torch.float16, size=[128, 4096], stride=[4096, 1], ...)
    allocation={'lx': 0}
    dim_hints=[DimHint(dim_names=[], split_count=1, loop_var=u0, is_reduction=False, hint_id=0, loop_var_range=8)]
    loop_levels=[((0,), 8, (0,), ())]
    Pointwise(
    'spyre',
    torch.float16,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(arg0_1, i1 + 4096 * i0 + 524288 * u0)
        tmp1 = ops.load(arg1_1, i1 + 4096 * i0 + 524288 * u0)
        tmp2 = tmp0 + tmp1
        return tmp2
    ,
    ranges=[128, 4096],
    ...
  )

  while_loop_body_graph_0_0_op9: ComputedBuffer
    buffer=while_loop_body_graph_0_0_buf9
    allocation={'lx': 524288}
    dim_hints=[DimHint(..., loop_var=u0, ..., loop_var_range=8)]
    loop_levels=[((0,), 8, (0,), ())]
    ...
        tmp0 = ops.load(while_loop_body_graph_0_0_buf8, i1 + 4096 * i0)
        tmp1 = ops.load(arg2_1, i1 + 4096 * i0 + 524288 * u0)
        tmp2 = tmp0 * tmp1
```

Three things to read off this dump:

- The tile ranges are `[128, 4096]`: the lowering divided dim 0 by the trip
  count, so every member works on one tile.
- The loop variable `u0` is in the index of each read of a full-size tensor
  (`524288 * u0` is 128 rows of 4096 elements), and absent from the read of the
  tile-sized intermediate `...buf8`. **A read's per-trip advance is the
  coefficient of the loop variable in its index**; nothing else stores it.
- `dim_hints` lists the loop variable of each level (`loop_var=u0,
  loop_var_range=8`), which is how a reader pairs a variable with its level.

### The generated `LoopSpec`

`SpyreKernel` builds an `OpSpec` per op and the scheduler wraps the three in one
`LoopSpec`. Each `TensorArg` that moves with the loop carries
`device_tile_advance_expr`, the per-trip advance projected onto the device
layout. The tile-sized intermediate, which lives in the scratchpad (`'lx'`),
carries none:

```python
LoopSpec(
    count=sympify('8'),
    body=[
        OpSpec(
            op='add',
            iteration_space={sympify('c0'): (sympify('128'), 1), sympify('c1'): (sympify('4096'), 2)},
            tiled_symbols=[[sympify('u0')]],
            tiled_symbol_trip_counts={sympify('u0'): 8},
            args=[
                TensorArg(is_input=True, arg_index=0, ..., allocation={'hbm': 0},
                          device_tile_advance_expr=sympify('8192*u0')),
                TensorArg(is_input=True, arg_index=1, ..., allocation={'hbm': 1},
                          device_tile_advance_expr=sympify('8192*u0')),
                TensorArg(is_input=False, arg_index=-1, ..., allocation={'lx': 0}),
            ],
        ),
        OpSpec(op='mul', ...),       # reads the scratchpad tile, advances 'c' by 8192*u0
        OpSpec(op='identity', ...),  # copies the tile into the output, advances by 8192*u0
    ],
)
```

### The generated `bundle.mlir`

The `LoopSpec` becomes an `scf.for` whose per-trip addresses are computed from
the induction variable and the base addresses the host passes in:

```text
#map_0 = affine_map<(d0)[s0] -> (s0 + 16384*d0)>
...
%loop_bound_0 = arith.constant 8 : index
scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
    %addr_0 = affine.apply #map_0(%i_0)[%arg_0]
    %addr_1 = affine.apply #map_0(%i_0)[%arg_0_core_4194304]
    %addr_2 = affine.apply #map_0(%i_0)[%arg_1]
    %addr_3 = affine.apply #map_0(%i_0)[%arg_1_core_4194304]
    sdscbundle.sdsc_execute (%addr_0, %addr_1, %addr_2, %addr_3) {sdsc_filename="sdsc_0.json", ...}
    ...
}
```

## Where loops come from

### `for_each_tile`

`for_each_tile` lowers to `torch._higher_order_ops.scan.scan`, which Inductor
traces into an `ir.WhileLoop`. `splice_while_loops`
(`wsr/for_each_tile_lowering.py`) is the first pass after
`apply_reserved_dims_contract`. For each loop it:

1. recognizes the while-loop shape and proves the trip count statically
   (`try_prove_for_each_tile`, `_extract_trip_count`);
2. splices the body's ops into the top-level `graph.operations`
   (`wsr/while_loop_bridge.py`);
3. gives every spliced op a `LoopMember`, one level per nesting level
   (`_stamp_direct_loop_info`), with the loop variable listed in its
   `dim_hints`;
4. declares each accumulator carry with a `LoopCarryRecord`
   (see [Carries](#carries));
5. calls `collapse`, which builds the nodes.

### The loop-tiling lowering

The loop-tiling lowering (`wsr/loop_tiling_lowering.py`) tiles ops that a pass
decided to tile, without a user-written loop. It has two entry points, both
through `tile_groups(graph, groups, group_idx_offset)`:

- the **span-overflow pass** (`_maybe_coarse_tile_span_overflow`), which tiles
  ops whose per-core span would exceed the hardware limit. It is opt-in
  (`config.ignore_span_overflow_hints = False`);
- the **allocator's tiling** (`apply_tilings` in `scratchpad/coarse_tiling.py`),
  which tiles to fit the working set in the scratchpad. It is opt-in
  (`config.auto_coarse_tiling`).

`tile_groups` is `build_loop_tiling_plan` followed by `lower_loop_tiling`. A
`LoopTilingPlan` holds only which position of each op a loop tiles and its trip
count; everything else (consumers, ranges, strides) is read from the graph at the
moment the plan is lowered, so the plan cannot go stale between planning and
lowering. The lowering divides each tiled op's range, puts the loop variable in
the op's `dim_hints` and a `step * u` term in each advancing index, adds a
copy-out for a tile something outside the loop reads, adds a fill and a combine
for an accumulator tiled on a reduction dim (declared as a `LoopCarryRecord`),
and gives each member a `LoopMember` with `strict_loop_vars=True`. It builds one
standalone loop per group; a nest, a group whose ops are already in a loop, and
the other cases named in `build_loop_tiling_plan`'s docstring raise
`Unsupported`.

The lowering runs in the middle of the pass pipeline, while a pass may still be
iterating `graph.operations`, so it does not wrap the ops itself. It returns
`{group index: loop variable}`, which the caller hands to `request_loop_nodes`;
the pass loop calls `build_pending_loop_nodes` once the pass has returned.

### The build-time record

Both lowerings describe what they did to an op with the same record:

```python
@dataclass
class LoopMember:
    loop_group_id: tuple[int, ...]            # nesting path, e.g. (0,) or (0, 0)
    loop_count: list[sympy.Expr]              # trip count per level, outermost first
    loop_tiled_dims: list[list[int]]          # per level: tiled positions in data.ranges
    loop_tiled_reduction_dims: list[list[int]]  # per level: tiled positions in reduction_ranges
    strict_loop_vars: bool = False            # True for the loop-tiling lowering's loops
```

`op.loop_member` exists only between the lowering and the node build. A reader
between the two (the lowering's own helpers, for example) goes through the same
`loop_facts` accessors as everyone else, which fall back to the record for an op
no node holds yet.

## The node

`LoopNode` (`loop_node.py`) is an `ir.Operation` with these fields:

| Field | Meaning |
|---|---|
| `body` | the loop's ops in execution order, and nested `LoopNode`s |
| `trip_count` | the loop's trip count (an `int` or a sympy expression) |
| `induction` | the loop variable (`u<n>`), or `None` for a node built without one |
| `carries` | the `LoopCarryRecord`s of the accumulators carried through the loop |
| `group_id` | this loop's entry in a member's group path |
| `tiling` | `{op name: (tiled_dims, tiled_reduction_dims)}` for each member at this level |
| `strict_loop_vars` | whether the loop-tiling lowering made the loop |
| `facts_recorded` | whether the pipeline built the node (a node built by hand in a unit test is not checked) |

Body ops keep their own names and stay in the graph's one buffer namespace; the
node is not registered in `graph.name_to_op`. Each leaf op in a built node has
`_loop_chain`, the tuple of nodes around it, outermost first.

### How nodes are built

`_wrap` groups runs of ops that share a `loop_group_id` prefix of length
`depth + 1` into a node, recursively, so a longer path yields a nested node. A
node's body is the span from its first member to its last. An op that sits between
two members without being one (an op of no loop, or an `ExternKernel`) stays in
the span where it is, since moving it would change program order. `collapse`
(called at the end of `splice_while_loops`) and `build_pending_loop_nodes` (for
the loop-tiling lowering) call `_wrap` and then `_record_node_facts`, which:

1. fills each node's `tiling` from its members' records and sets each leaf's
   `_loop_chain`;
2. sets `node.strict_loop_vars` from the members' records, and raises if the
   members of one node disagree (a loop is made by one lowering);
3. deletes `loop_member` from every op the node holds.

### What a member is

An op is a **member** of a loop when the loop's nodes hold a tiling entry for it.
Being placed in a loop's body is not enough: an LX clone the allocator puts beside
a body consumer runs every trip, but it tiles nothing and is not part of the loop's
computation. If it counted as looped, the relayout planner would treat it as part
of its consumer's tile group and offer a relayout for it. A copy of a member made
with `copy_op_metadata` is a member of the same loops (see
[Editing](#editing-a-graph-that-holds-loops)); a drain after the loop or a hoisted
pre-loop clone is not.

## Reading loop facts

`loop_facts.py` is the one read API for "which loops is this op in". Once an op's
nodes are built it answers from them; for an op no node holds yet it answers from
the lowering's `loop_member`.

| Function | Answer |
|---|---|
| `enclosing_loops(op)` | the `LoopLevel`s around `op`, outermost first: `group_path`, `trip_count`, `induction`, `tiled_dims`, `tiled_reduction_dims` |
| `loop_depth(op)` | the number of levels |
| `counted_loop_path(op, include_extern=False)` | the group path `op` runs in; `()` for an `ExternKernel` unless `include_extern` is set (it runs once, outside the loop) |
| `is_looped(op)` | whether `op` is a member of a counted loop |
| `strict_loop_vars(op)` | whether the loop-tiling lowering made the op's loops |
| `read_advance(op, dep)`, `output_advance(op)` | the per-trip advance of a read or of the write, one entry per level, in host elements |
| `read_advances_anywhere`, `output_advances_anywhere`, `advancing_levels` | whether and where an access moves with the loop |

### The advance comes from the index

A read's advance at a level is the coefficient of that level's loop variable in the
read's index. `read_advance` pairs each variable with its level through the op's
`dim_hints` (the entries with a `loop_var_range`, outermost first) and reads the
coefficient with `_loop_coefficient`, which refuses a non-affine use of the
variable with `Unsupported`. The kernel projects that host advance onto the device
layout (`SpyreKernel._index_tile_advance`) to get each `TensorArg`'s
`device_tile_advance_expr`. An op of a `for_each_tile` loop that lists no loop
variable does not move with the loop: its advance is zero at every level. An op
of the loop-tiling lowering always lists the variable of every level, so one that
lists none is a compiler bug and raises (`strict_loop_vars`).

## Editing a graph that holds loops

Passes read a flat view and edit through helpers.

### The flat read view

Pre-scheduling passes are handed `graph.operations` as an `OpsView`
(`passes._flat_operations`), a read-only view that iterates the leaf ops in program
order, as if the loops were not there. Its `insert`, `remove`, item assignment and
`del` raise `TypeError`: "next to this flat neighbour" does not say whether an op
belongs inside a loop body or beside the node. `append` (what
`GraphLowering.register_operation` does with a newly lowered op) goes to the top
level, and `hoist_to_front` moves ops to the head of the graph across node
boundaries on purpose.

### The edit helpers

Every structural edit names where the op goes, with a helper from `loop_node.py`:

| Helper | Effect |
|---|---|
| `insert_before(ops, anchor, op)`, `insert_after(ops, anchor, op)` | put `op` next to `anchor`, in the list that holds `anchor` |
| `move_before`, `move_after` | the same for an op already in the graph (it keeps its recorded tiling) |
| `insert_at_start(node, op)`, `insert_on_back_edge(node, op)` | first entry of the body (runs every trip, before the rest), or the end of the body (the point a carry copy needs) |
| `replace_op(ops, old, new)` | put `new` where `old` is; it takes `old`'s chain and its own stamp's tiling |
| `remove_op(ops, op)` | remove from whichever list holds it, and forget its tiling |
| `hoist_to_front(ops)` | move ops out of any loop to the head of the graph |
| `top_level_entry(ops, op)` | the entry of the top-level list that holds `op`: the op itself, or the outermost node around it |

**The placement rule:** an anchor that is a `LoopNode` puts the op outside the loop
(before it runs once; after it runs once, after the last trip); an anchor that is a
body op keeps the op in that body (it runs every trip). Each helper keeps the op's
`_loop_chain` and the nodes' `tiling` right, and each accepts the `OpsView` or the
real nested list.

### Telling the nodes what a new op is

An op placed in a body has a chain but is not yet a member. A pass that creates an
op the loop should treat as part of its computation says so:

- `record_member(op, [(tiled_dims, tiled_reduction_dims), ...])`, one entry per
  node around `op`. The restickify copy of a looped consumer, the padding ops of a
  looped matmul and the relayout copies of a looped scatter do this.
- `copy_op_metadata(src, dst)` marks `dst` with `src` as its donor when `src` is a
  member; when a helper places `dst` in `src`'s nodes it tiles what `src` tiles.
  The allocator's per-trip LX clones and the LX dump/restore copies are members
  this way. A pass that makes something that must not be a member (a drain, a
  hoisted clone) clears the marker (`_clear_loop_membership_metadata`).
- `inherit_loop_membership(source, target)` makes a probe copy of an op that is not
  placed in the graph look looped like its source.

### A checklist for a pass that adds or replaces ops in a loop

1. Place the op with a helper; never edit `graph.operations` or a node's `body`
   directly.
2. If the op advances with the loop or is otherwise part of its computation, call
   `record_member`, or build it with `copy_op_metadata` from a member.
3. Keep the loop variable in an advancing read's index and in the op's `dim_hints`
   (the advance is read from them).
4. Run with the loop checker on (it is on under pytest).

## The loop checker

`loop_checker.py` verifies the structure between passes. `check_loops(graph,
where=pass_name)` raises `LoopStructureError` naming the pass that left the tree
inconsistent. It is on under pytest and with `TORCH_SPYRE_CHECK_LOOPS=1`, and off
otherwise; `building_loop()` suspends it while a loop is half built. The rules:

1. Every op appears exactly once in the tree.
2. Every loop variable in an op's indices is the induction variable of a loop
   that encloses it; and an op whose index uses an enclosing loop's variable holds
   a tiling entry in those nodes (without one it would read as "not in the loop"
   and silently lose its advance).
3. Every carry's update is an op in its node's body.
4. An op that runs in a counted loop a node covers sits inside that node, not
   outside it and not in the body of another loop's node.
5. A node's body holds no op that is not a member of the loop (an `ExternKernel`
   is allowed: the scheduler hoists a once-op out).
6. An op in nodes the pipeline built points at exactly those nodes
   (`_loop_chain`) and carries no `loop_member`.

## Carries

A **carry** is an accumulator that lives across trips (a split-K matmul's partial
sum, online softmax's running max). The lowering declares it with a
`LoopCarryRecord(storage_name, update_name, loop_origin, storage_view)` on the
carry's storage (its pre-loop initial buffer) and on its update (the body op that
computes the next trip's value), and leaves the body pure: the update writes its
own buffer. `apply_loop_carries` (`loop_carries.py`) runs just before layout
propagation and turns each declaration into the in-place form the rest of the
pipeline expects: the update becomes a `MutationLayoutSHOULDREMOVE` write into the
storage, so the next trip reads what the last trip wrote, and every reader of the
update reads the storage. The record tells scratchpad planning to treat the closed,
compiler-created mutation differently from a user mutation, which must stay in HBM.
`LoopNode.carries` lists the records of a node's body, and the allocator, the loop
checker and dead-code elimination read them.

## Scheduling

### Handing the nodes to the scheduler

`CustomPreSchedulingPasses` ends with `_hand_off_loop_nodes`, which leaves
`graph.operations` as the nested list, with nothing flattened. On a supported torch
release (`SUPPORTED_RELEASES` in `loop_scheduler.py`, currently 2.14), each
`LoopNode` reaches `Scheduler.__init__` and becomes **one** `LoopSchedulerNode`; a
nested `LoopNode` becomes a nested `LoopSchedulerNode`. On any other release a
graph that holds a `LoopNode` is refused with `Unsupported` naming the release
(`refuse_release`), because flattening it would compile the loop without saying
that the release is unsupported. A graph with no loop is unaffected on every
release.

If a pass or `check_final_operation_order` raises, the loop is expanded back to
the flat list before the error propagates, so error reporting sees leaf ops.

### `LoopSchedulerNode`

`LoopSchedulerNode` (`loop_scheduler.py`) is the `CountedLoopSchedulerNode`
(`scheduler.py`) for one `LoopNode`. Its `snodes` are the body's scheduler nodes.
`CountedLoopSchedulerNode` is a `FusedSchedulerNode` subclass: Inductor treats it
as one node, which gives it the one property the codegen needs, that no Inductor
fusion can merge a node into the loop or take part of it out
(`can_fuse`, `can_fuse_vertical` and `can_fuse_horizontal` all return `False`).
`LoopSchedulerNode` does not call `FusedSchedulerNode.__init__`, because that
reads `min_order` and `max_order` before `compute_ancestors` has set them, and
leaves `node` unset where `compute_dependencies` expects one. It carries the
`LoopNode` as `node`, and `loop_count` is the node's `trip_count`.

- **Hoisting.** A no-op or extern body op becomes an ordinary scheduler node placed
  immediately before the outermost loop, since it runs once. One that reads a
  buffer the loop writes is refused with `Unsupported`.
- **Membership.** Every kernel op directly in a body must be a member of that
  loop, with one group path (`_check_members`). An op without membership would run
  every trip, so it is refused rather than run wrongly.
- **Dependencies.** The loop's reads are the union of its children's, narrowed to
  reads from outside the loop. The loop is the `defining_op` of every body buffer.
- **Patches.** Four per-compile patches on `Scheduler` (`_PATCHED`):
  `create_scheduler_node` (builds the loop node), `update_zero_dim_cpu_tensor`
  (splices the hoisted ops in front of the loop and restores `graph.operations` to
  the flat leaf list, because scheduler-time readers expect leaf ops),
  `compute_ancestors` and `dead_node_elimination` (a body buffer a sibling reads
  counts as live, and a loop whose outputs nothing reads is removed with its
  body). `PATCHED_SURFACE` lists the private Inductor names to re-verify on an
  upgrade.

`CustomPreFusionPasses` runs `propagate_mutation_layouts`; nothing regroups the
`LoopSchedulerNode`s. `CustomPostFusionPasses` runs `sink_collective_waits`,
`spyre_fuse_nodes`, `prepare_spyre_kernels` and `hbm_pool_planning`. A counted loop
may share a bundle with adjacent ops (`spyre_fuse_nodes` accepts a
`CountedLoopSchedulerNode` as well as a `SchedulerNode`); `hbm_pool_planning`
treats it as a top-level bundle member, not a plain `SchedulerNode`.

### Preparing and emitting the kernel

`prepare_spyre_kernels` runs `SuperDSCScheduling.prepare_kernel` on each bundle,
before pool planning. For a loop it builds the whole `LoopSpec` tree
(`_codegen_loop_body`, which reaches `SpyreKernel.counted_loop` for each nested
level; the outer level goes through `wrap_op_specs_in_loop`). `counted_loop` is the
one place a `LoopSpec` is built: it applies the ShapeEnv replacements, refuses a
symbolic count over an undeclared dimension (`require_declared_count`) and fills
`count_symbol_bounds`. If preparation hits `Unsupported`/`ValueError` because of an
LX buffer, it demotes that buffer and retries, bounded by the number of LX buffers
plus one. `codegen_node` then consumes the prepared kernel and binds only what
preparation could not know (pool size, addresses, argument list); it does not
rebuild the loop body. A prepared kernel whose leaf names no longer match the live
nodes raises `RuntimeError` ("changed after its kernel was prepared").

## `LoopSpec` and codegen

### `LoopSpec` and `OpSpec.tiled_symbols` (`op_spec.py`)

```python
@dataclass
class LoopSpec:
    count: sympy.Expr            # trip count; may be symbolic
    body: list[Any]              # OpSpec | UnimplementedOp | LoopSpec
    count_symbol_bounds: dict[str, tuple[int, int]]   # symbol -> (max, tile size)
    count_symbol_sources: dict[str, tuple[int, int]]  # symbol -> (arg_index, dim_index)

@dataclass
class OpSpec:
    ...
    tiled_symbols: list[list[sympy.Symbol]]          # one entry per enclosing level
    tiled_symbol_trip_counts: dict[sympy.Symbol, int]
```

- `count_symbol_bounds` and `count_symbol_sources` are filled only for a symbolic
  `count`: the first while the ShapeEnv exists, the second during kernel codegen,
  where the launch argument order is known. A symbol that cannot be placed is left
  out with a warning, and bundle generation then refuses to declare its parameter.
- `tiled_symbols` has one entry per enclosing loop level, innermost first (the
  bundle and `compile_op_spec` paths reverse it to outermost first). An entry is
  non-empty when something advances at that level or the op tiles a dim there
  (`advancing_levels`). The symbol is the loop's own induction variable when the
  op's `dim_hints` list one for the level (`_induction_symbol`), and a placeholder
  `_tile_adv_{op_name}_lvl{level}` otherwise.
- `tiled_symbol_trip_counts` maps each symbol to `max_trip_count(level trip count)`,
  because a symbolic count has no single value; the SDSC needs the dimension's
  largest extent.
- `TensorArg.device_tile_advance_expr` is the per-trip advance of that argument in
  device elements, a sympy expression whose free symbols are the entries of
  `tiled_symbols`. `SpyreKernel._tile_advance` computes it with
  `_index_tile_advance`: each level's loop-variable coefficient in the dep's index,
  projected to device elements by `tiling_expr_to_device_expr`. It is `None` for an
  op that is not looped, an op that lists no loop variable, a repeated load that
  consumes no new read, or a zero advance. An op whose `dim_hints` list a different
  number of loop variables than it has loop levels raises `RuntimeError`, and so
  does a `strict_loop_vars` op whose access has no advance.

`create_op_spec` takes the levels from `enclosing_loops(ir_node)` (`tiled_dims`,
`tiled_reduction_dims`, `trip_count`) and `advancing_levels(ir_node)`. The
validator `_check_tiled_symbols` (`op_spec_validation.py`) checks each symbol.

### A bundle boundary

A counted loop is emitted by one `codegen_node` call: the `LoopSchedulerNode` is
one node, so the whole `LoopSpec` tree is serialized together. `generate_bundle`
takes the resulting `list[OpSpec | LoopSpec]`, emits one SDSC JSON per distinct
`OpSpec` in the body (recursively), and wraps them in `scf.for`. With the SDSC
cache on (the default), a repeated op reuses an existing file, so filenames are not
one per `OpSpec`. `generate_bundle` requires `config.bundle_symbolic_args`, which
defaults on.

### Serialization

`SpyreKernel.codegen_kernel` serializes the tree into the generated Python wrapper
(`_codegen_op_spec_list`). `tiled_symbols` is written only when non-empty;
`tiled_symbol_trip_counts`, `count_symbol_bounds` and `count_symbol_sources` are
written when non-empty. `arg_index` is fixed up by a walk of the whole tree
(`_iter_op_specs`), and for a symbolic count `_resolve_loop_dimension_sources`
fills `count_symbol_sources` from the same list.

### `bundle.mlir` for a loop (`codegen/bundle.py`)

Each loop level becomes an `scf.for`. A concrete count emits
`%loop_bound_k = arith.constant N : index` and `scf.for %i_k = %c0 to
%loop_bound_k step %c1`. A symbolic count is supported when
`decompose_tiled_count(count)` gives `(symbol, tile)`, a bare symbol or a symbol
divided by an integer; any other shape, or a symbol with no `input_arg` parameter,
is `Unsupported`. It emits `to %dim_<sym>` with step `%c1` for tile 1 and a
`%step_k = arith.constant tile` otherwise; the loop variable then counts elements,
and every affine stride at that level is divided by the tile (`_scaled_strides`).
Each symbolic dimension is a bundle parameter
`%dim_<sym>_base: !sdscbundle.input_arg<index, granularity=G, max_value=M>`, and one
dimension driving loops of different tile sizes is refused.

Inside the loop, every tensor that advances gets an
`affine.apply #map_K(%i_..)[%base]`, which computes its address for this trip from
the induction variable and the base address the host passes in; the
`sdscbundle.sdsc_execute` takes those addresses as operands. A scratchpad (`'lx'`)
tensor has no advance and is not given an address map. Nested loops number their
induction variables and bounds in order (`%i_0`, `%i_1`, ...).

`generate_bundle` is the only loop codegen path. The KTIR emitter (`codegen/ktir.py`,
`TORCH_SPYRE_KTIR=1`) consumes the same `LoopSpec` contract and emits `scf.for` for it.

## Invariants and failure modes

- **Structure.** Every op appears once; each leaf points at exactly the nodes that
  hold it; an op that advances with a loop holds a tiling entry. `check_loops`
  verifies this between passes, and `LoopSchedulerNode._check_members` verifies at
  scheduling that every kernel op in a body is a member.
- **Contiguity.** The ops of a group the loop-tiling lowering lowers are one
  contiguous slice of `graph.operations` (`_validate_contiguous` raises
  `RuntimeError` otherwise).
- **Pass order.** `splice_while_loops` is the second pre-scheduling pass, after
  `apply_reserved_dims_contract`, so every later pass sees the spliced body.
  `apply_loop_carries` runs after `split_multi_ops` and before layout propagation.
  Span-overflow tiling runs after stickification and before work division and
  scratchpad planning. Nodes are built at the end of the pass that creates the loop.
- **Trip counts.** A loop has one trip count for all its members at each depth.
- **Dependencies.** Inductor's `ComputedBuffer.get_read_writes` is not cached, so a
  rewritten `inner_fn` is seen by the scheduler. torch-spyre's own memo
  (`pass_utils.op_read_writes`) is not: a pass that changes an `inner_fn` calls
  `invalidate_op_read_writes(op)`.
- **Liveness.** An op whose only reader is the next trip of its own loop (an
  accumulator's fill) looks dead to Inductor's dead-code sweep. `patches.py` makes
  `SchedulerNode.has_side_effects` true for an op flagged `_coarse_tile_force_live`
  and for any `ComputedBuffer` with a `MutationLayoutSHOULDREMOVE` layout, and
  `LoopSchedulerNode` patches `dead_node_elimination` so a loop's body buffers stay
  live while the loop is.
- **One mutation target.** Inductor allows at most one mutation target per op
  (`Scheduler.compute_dependencies`). The lowerings that write
  `MutationLayoutSHOULDREMOVE` (the copy-out, fill and combine ops of the
  loop-tiling lowering, `apply_loop_carries`, the carry bridge in
  `wsr/while_loop_bridge.py`, the restickify and non-stick-order relayouts,
  `copy_forced`, and LX context switching) each give an op one target.
  `propagate_mutation_layouts` (the pre-fusion pass) gives each such op a real
  `FixedTiledLayout` once the scheduler has consumed the mutation state: the
  target's own layout when it fits the op, a computed one otherwise. It descends
  into a `LoopSchedulerNode`'s body like any other group.
- **The cache key.** `_uuid` in `passes.py` hashes each pass's own source file, so
  editing a pass invalidates the Inductor FX cache. A change to a helper in a file
  that is not a pass (for example `wsr/loop_tiling_common.py`) is not seen by it;
  clear the cache (`/tmp/torchinductor_<user>/fxgraph/`) when a result looks stale.

### The wrap-never-reconstruct convention

A pass that changes an op's body wraps the original `inner_fn` with a
`WrapperHandler` subclass and installs it with `V.set_ops_handler`; it never
rebuilds `inner_fn` from index expressions, which go stale as soon as the pass
runs (see `AGENTS.md`). `replace_computed_buffer_body` (`pass_utils.py`) is the
helper. The handlers in use are `NameSwapHandler` (`pass_utils.py`), which swaps a
buffer name; `_RetileLoadIndexHandler` (`wsr/loop_tiling_common.py`), which
rewrites a load index from the old stride coefficients to the tile's; and
`_NameAndIndexSwapHandler` (`wsr/coarse_tile.py`), which does both for a reduction
accumulator. The ops the lowering creates (the copy-out, fill and combine) are new
`Pointwise`s built from loaders; they have no `inner_fn` to wrap.

`_divide_ranges` (`wsr/loop_tiling_common.py`) shrinks an op's `data.ranges` and
its layout size and stride and leaves `inner_fn` alone, because the op's index
arithmetic is already in terms of the per-tile loop variables. `_patch_retiled_load_indexes`
runs once per group at the end of `_lower_group` and patches only the ops that were
in the group before the division and read a retiled buffer; an index it cannot
rewrite raises `Unsupported`. An outside consumer of a tile-sized buffer is
redirected by `_patch_consumers`: `NameSwapHandler` swaps the name,
`_ReadThroughLayout` makes the index follow the full buffer for a copy-out, and
`_NameAndIndexSwapHandler` rescales the index for an accumulator.

## Key files

| File (under `torch_spyre/_inductor/`) | What it holds |
|---|---|
| `loop_node.py` | `LoopMember`, `LoopCarryRecord`, `LoopNode`, `OpsView`, the edit helpers, `request_loop_nodes` / `build_pending_loop_nodes` / `collapse` (build nodes), `expand` (flatten) |
| `loop_facts.py` | the read API: `LoopLevel`, `enclosing_loops`, `counted_loop_path`, `is_looped`, `loop_depth`, `strict_loop_vars`, `read_advance`, `output_advance`, `advancing_levels` |
| `loop_checker.py` | `check_loops`, `find_violations`, `LoopStructureError` |
| `loop_carries.py` | `apply_loop_carries` |
| `loop_scheduler.py` | `LoopSchedulerNode`, `_check_members`, the per-compile `Scheduler` patches, `SUPPORTED_RELEASES`, `loop_scheduler_supported`, `refuse_release` |
| `scheduler.py` | `CountedLoopSchedulerNode`, `prepare_spyre_kernels`, `SuperDSCScheduling.prepare_kernel` / `codegen_node` / `_codegen_loop_body` |
| `passes.py` | the pass lists (`CustomPreSchedulingPasses`, `CustomPreFusionPasses`, `CustomPostFusionPasses`), `_hand_off_loop_nodes`, the `check_loops` call between passes |
| `wsr/for_each_tile.py` | the `for_each_tile` frontend |
| `wsr/for_each_tile_lowering.py` | `try_prove_for_each_tile`, `_extract_trip_count`, `splice_while_loops`, `_stamp_direct_loop_info` |
| `wsr/while_loop_bridge.py` | splices a `WhileLoop` body and declares each accumulator's `LoopCarryRecord` |
| `wsr/loop_tiling_lowering.py` | `LoopTilingPlan`, `build_loop_tiling_plan`, `lower_loop_tiling`, `tile_groups` |
| `wsr/loop_tiling_common.py` | shared range, layout and index helpers: `_divide_ranges`, `_RetileLoadIndexHandler`, `_patch_retiled_load_indexes`, `_ReadThroughLayout` |
| `wsr/coarse_tile.py` | planning and validation helpers for the loop-tiling lowering, `_patch_consumers`, `_make_combine_buffer` |
| `wsr/coarse_tile_span_overflow.py` | the span-overflow groups handed to `tile_groups` |
| `scratchpad/coarse_tiling.py` | `apply_tilings`, the allocator's entry to `tile_groups` |
| `pass_utils.py` | `replace_computed_buffer_body`, `NameSwapHandler`, `copy_op_metadata`, `op_read_writes`, `invalidate_op_read_writes`, `decompose_tiled_count` |
| `op_spec.py` | `LoopSpec`, `OpSpec.tiled_symbols` |
| `spyre_kernel.py` | builds `tiled_symbols` and `device_tile_advance_expr`, `counted_loop`, `codegen_kernel` |
| `codegen/bundle.py` | emits `scf.for` around `sdsc_execute` in `bundle.mlir` |
| `codegen/ktir.py` | the KTIR emitter over the same `LoopSpec` contract |
| `patches.py` | the `has_side_effects` patch that keeps a loop's mutation-layout ops live |
