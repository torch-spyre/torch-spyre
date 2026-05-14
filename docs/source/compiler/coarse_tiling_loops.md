# Coarse-Tiling Loop IR for the Spyre Backend

## Background

Spyre's compilation pipeline runs a sequence of optimization passes over
`ir.Operation` objects in `CustomPreSchedulingPasses`, before Inductor's
`Scheduler` is constructed.  One planned optimization is **coarse-level
tiling**: take a sequence of operations that share an iteration space
dimension, split that dimension into K chunks (where K may be a symbolic
shape), and emit the body operations inside a counted outer loop.  This
allows the hardware to amortize setup cost over K iterations without
requiring the full iteration space to fit in on-chip memory at once.

The output of this pass needs to survive through:

1. Inductor's `Scheduler` (which wraps each `ir.Operation` in a
   `SchedulerNode`)
2. Spyre's `SuperDSCScheduling.codegen_node()` (which drives `SpyreKernel`
   to produce `OpSpec` objects)
3. Downstream SDSC compilation (which needs an explicit loop count to
   generate correct hardware instructions)

This document describes how that loop structure is represented, transported,
and consumed.

## Design Overview

The design has three layers that correspond to the three pipeline stages
above.

```
Pre-scheduling IR pass
  └─ stamps loop_group_id + loop_count on each ir.Operation
  └─ rewrites each op's ranges (divides the tiled dimension by K)

  ↓  Inductor Scheduler wraps each ir.Operation → SchedulerNode
  ↓  _post_fusion_custom_pass fires

Post-fusion scheduler pass
  └─ scans list[BaseSchedulerNode] for runs sharing a loop_group_id
  └─ wraps each run in a CountedLoopSchedulerNode(count=K, snodes=[...])

  ↓  Scheduler calls SuperDSCScheduling.codegen_node()

codegen_node
  └─ receives CountedLoopSchedulerNode
  └─ drives SpyreKernel for the inner ops, collecting inner OpSpecs
  └─ wraps them in LoopSpec(count=K, body=[OpSpec, ...])
  └─ LoopSpec is serialized alongside OpSpec in codegen_kernel()
```

## Layer 1 — Pre-scheduling IR pass

### Attribute contract on `ir.Operation`

The coarse-tiling pass stamps two attributes onto each `ir.Operation` that
participates in a loop group.  These attributes are plain Python values
attached with `setattr`; no Inductor base class is modified.

| Attribute | Type | Meaning |
|---|---|---|
| `loop_group_id` | `int` | Opaque integer identifying which loop group this op belongs to. All ops with the same id form the body of one counted loop. |
| `loop_count` | `sympy.Expr` | Trip count of the outer loop. Must be identical for every op sharing a `loop_group_id`. |

The pass also **rewrites the op's iteration ranges**: the dimension being
tiled is divided by `loop_count` so that each inner `OpSpec` describes only
the work done per loop iteration, not the full iteration space.

### Why these two attributes are sufficient

`loop_count` is redundant across all ops in a group (they must agree), but
keeping it on each op means the post-fusion pass does not need to maintain
a separate side table.  The `loop_group_id` is the join key; it does not
need to carry any other information because the reduced iteration space is
already embedded in the op's `ranges`.

### Placement in `CustomPreSchedulingPasses`

The coarse-tiling pass runs after `insert_padding_ir` and before
`span_reduction`:

```python
insert_padding_ir(operations)
coarse_tile(operations)          # new pass
span_reduction(operations)
k_fast_ops = (
    k_fast_division(operations) if config.core_id_k_fast_emission else []
)
work_distribution(operations, k_fast_ops)
if config.lx_planning:
    scratchpad_planning(operations)
```

This ordering is required by two constraints that pull in opposite
directions:

**Must run after stickify and padding.**  `propagate_spyre_tensor_layouts`,
`insert_restickify`, and `insert_padding_ir` establish the final tiled
memory layout for each tensor.  The tiling pass inspects tensor shapes and
strides to decide which dimension to split and by how much; it must see
the post-stickify, post-padding shapes or it will split on the wrong
dimension or produce a non-stick-aligned inner size.

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

`scratchpad_planning` must run after coarse tiling because scratchpad
allocation depends on the final (reduced) iteration space.

## Layer 2 — `CountedLoopSchedulerNode`

### Class definition

`CountedLoopSchedulerNode` lives in
`torch_spyre/_inductor/scheduler.py` alongside `SuperDSCScheduling`.
It subclasses Inductor's `GroupedSchedulerNode`:

```python
class CountedLoopSchedulerNode(GroupedSchedulerNode):
    loop_count: sympy.Expr

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: list[BaseSchedulerNode],
        loop_count: sympy.Expr,
    ) -> None:
        super().__init__(scheduler, snodes)
        self.loop_count = loop_count

    @classmethod
    def can_fuse(
        cls,
        producer: BaseSchedulerNode,
        consumer: BaseSchedulerNode,
    ) -> bool:
        return False
```

Everything else (`get_nodes`, `get_outputs`, `get_name`, dependency
merging) is inherited from `GroupedSchedulerNode` unchanged.
`can_fuse` returns `False` — a loop group is atomic; nothing can be
fused into it from outside.

### Why `GroupedSchedulerNode` is the right base

`GroupedSchedulerNode` already:

- Merges `unmet_dependencies` across all constituent nodes so the
  scheduler respects data-flow ordering.
- Sets `min_order` / `max_order` to prevent interleaving.
- Registers all constituent names in `scheduler.name_to_fused_node` so
  lookups work correctly.
- Exposes `get_nodes()` which `codegen_node` already iterates over.

None of this needs to be reimplemented.  The only thing `CountedLoopSchedulerNode`
adds is `loop_count`.

### The post-fusion pass

The post-fusion hook (`config._post_fusion_custom_pass`) is the correct
injection point.  It fires after Inductor's own fusion has completed but
before codegen begins, and it receives and returns
`list[BaseSchedulerNode]`.  Spyre sets this hook in `patches.py` alongside
the existing `_update_scheduler` monkey-patch.

The pass algorithm:

```
result = []
i = 0
while i < len(nodes):
    node = nodes[i]
    snode = node  # may be a plain SchedulerNode
    gid = getattr(snode.node, "loop_group_id", None)  # snode.node is ir.Operation
    if gid is None:
        result.append(node)
        i += 1
        continue
    # collect the run of nodes sharing this gid
    run = [node]
    loop_count = snode.node.loop_count
    i += 1
    while i < len(nodes):
        next_node = nodes[i]
        if getattr(next_node.node, "loop_group_id", None) != gid:
            break
        run.append(next_node)
        i += 1
    result.append(CountedLoopSchedulerNode.create(run, loop_count))
return result
```

Key invariant: because the pre-scheduling pass runs in topological order and
the scheduler's topological sort preserves that order, a loop group's
`SchedulerNode`s will be contiguous in the post-fusion node list.  If they
are not contiguous it means a data-flow constraint separates them, which is a
bug in the tiling pass (it tiled ops that have an inter-op dependency that
crosses the group boundary).  The post-fusion pass asserts contiguity.

## Layer 3 — `LoopSpec` and codegen

### `LoopSpec` in `op_spec.py`

```python
@dataclasses.dataclass
class LoopSpec:
    count: sympy.Expr
    body: list[OpSpec | UnimplementedOp]
```

`LoopSpec` is a peer of `OpSpec` and `UnimplementedOp` in the list that
`SpyreKernel.codegen_kernel()` serializes.  It is not a subclass of `OpSpec`
because it has no `iteration_space`, `args`, or `op_info` of its own — those
belong to the inner `OpSpec`s.

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
    node_schedule = self.generate_node_schedule(inner_nodes)
    kernel = SpyreKernel()
    with kernel:
        for snode in node_schedule:
            var_ranges = iteration_space(snode)
            vars = list(var_ranges.keys())
            index_vars = [
                vars[: len(snode._body.iter_vars)],
                vars[len(snode._body.iter_vars) :],
            ]
            snode.codegen(index_vars)

    # Wrap the collected inner specs in a LoopSpec
    kernel.wrap_op_specs_in_loop(node.loop_count)

    with V.set_kernel_handler(kernel):
        src_code = kernel.codegen_kernel()
    kernel_name = self.define_kernel(src_code, node_schedule, kernel)
    kernel.kernel_name = kernel_name
    kernel.code_hash = code_hash(src_code)

    with V.set_kernel_handler(kernel):
        for snode in node_schedule:
            snode.mark_run()

    self.codegen_comment(node_schedule, kernel_name)
    kernel.call_kernel(kernel.kernel_name)

    V.graph.removed_buffers |= kernel.removed_buffers
    V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
    self.free_buffers_in_scheduler()
```

`SpyreKernel.wrap_op_specs_in_loop(count)` replaces the flat `self.op_specs`
list with `[LoopSpec(count=count, body=self.op_specs)]`.

### Serialization in `codegen_kernel()`

`codegen_kernel()` already iterates `self.op_specs` to emit Python source.
A `LoopSpec` entry is serialized as:

```python
LoopSpec(
    count=sympify('K'),
    body=[
        OpSpec(...),
        OpSpec(...),
    ]
)
```

The `arg_index` fixup loop (which maps tensor names to kernel argument
positions) runs before serialization and is unaffected — `TensorArg` objects
inside the inner `OpSpec`s are still discovered by iterating
`self.spyre_kernel_args`.

## Files changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/op_spec.py` | Add `LoopSpec` dataclass |
| `torch_spyre/_inductor/spyre_kernel.py` | Add `SpyreKernel.wrap_op_specs_in_loop()`; extend `codegen_kernel()` to serialize `LoopSpec` |
| `torch_spyre/_inductor/scheduler.py` | Add `CountedLoopSchedulerNode`; add `_codegen_counted_loop()` to `SuperDSCScheduling` |
| `torch_spyre/_inductor/passes.py` | Add `coarse_tile()` call in `CustomPreSchedulingPasses.__call__()` |
| `torch_spyre/_inductor/patches.py` | Register `_post_fusion_custom_pass` hook alongside existing `_update_scheduler` patch |
| `torch_spyre/_inductor/coarse_tile.py` | New file: `coarse_tile(operations)` pass implementation |
| `torch_spyre/_inductor/codegen/superdsc.py` | Extend `parse_op_spec()` / `compile_op_spec()` to handle `LoopSpec` |
| `tests/inductor/test_coarse_tiling.py` | New file: tests for the end-to-end tiling pipeline |

## Invariants and failure modes

**Contiguity invariant**: all `SchedulerNode`s sharing a `loop_group_id`
must be contiguous after the scheduler's topological sort.  If the tiling
pass stamps ops that have a data dependency crossing the group boundary,
the post-fusion pass will detect a non-contiguous run and raise an error.

**Consistent `loop_count`**: all ops in a group must agree on `loop_count`.
The post-fusion pass asserts this.

**Pass ordering**: coarse tiling must run after stickify/padding and
before `span_reduction`, `k_fast_division`, `work_distribution`, and
`scratchpad_planning`.  All of these downstream passes must see the
reduced (per-iteration) ranges, not the full pre-tiling iteration space.
See "Placement in `CustomPreSchedulingPasses`" for the full rationale.

**Cache invalidation**: `CountedLoopSchedulerNode` does not affect the
Inductor FX graph cache key.  The `loop_group_id` and `loop_count`
attributes are added in a pre-scheduling pass whose source files are
already included in `CustomPreSchedulingPasses.uuid()`.

## Out of scope

- Nested counted loops (a loop body that itself contains a `LoopSpec`).
- Loops whose trip count is data-dependent (use `ir.WhileLoop` for that).
- Fusing a non-tiled op into the body of a `CountedLoopSchedulerNode`.
