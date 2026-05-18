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
    body: list[OpSpec | UnimplementedOp | LoopSpec]
```

`LoopSpec` is a peer of `OpSpec` and `UnimplementedOp` in the list that
`SpyreKernel.codegen_kernel()` serializes.  It is not a subclass of `OpSpec`
because it has no `iteration_space`, `args`, or `op_info` of its own — those
belong to the inner `OpSpec`s.

The `body` type is recursive: a `LoopSpec` body may itself contain
`LoopSpec` entries, representing nested counted loops.  Python's type
system requires a forward reference or a `TYPE_CHECKING` guard for the
self-referential annotation; in practice a `list[Any]` with a runtime
`isinstance` check is sufficient.

### Nested loops and the `loop_group_id` tree

To support nesting, each `ir.Operation` carries a `loop_group_id` that is
a **path** rather than a flat integer.  A path is a tuple of integers, one
element per nesting level:

| `loop_group_id` | Meaning |
|---|---|
| `(0,)` | outermost loop group 0, not nested |
| `(0, 1)` | inner loop group 1 inside outer loop group 0 |
| `(0, 1, 2)` | three levels deep |

`loop_count` remains a single `sympy.Expr` on each op — it is the trip
count of the **innermost** loop that directly contains this op.  Outer
loop counts are read from the ops at the corresponding prefix level.

The post-fusion pass reconstructs the tree by grouping on prefix:

1. Group flat `SchedulerNode` list into runs that share the same
   outermost group id (first path element).
2. For each such run, recursively group any nodes whose path length is
   greater than 1 into nested `CountedLoopSchedulerNode`s (stripping the
   first path element before recursing).
3. Wrap the top-level run in a `CountedLoopSchedulerNode` whose
   `loop_count` is the trip count from the outermost path element.

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
emitted by `codegen_kernel()`.  When it encounters a `LoopSpec` it must
emit SDSC JSON files for each `OpSpec` in the body (recursively) and
wrap those executions in an `scf.for` in `bundle.mlir`, as described
below.

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

For nested `CountedLoopSchedulerNode`s the inner node's `codegen_node()`
call produces an inner `LoopSpec`; the outer `wrap_op_specs_in_loop` then
wraps that `LoopSpec` (along with any sibling `OpSpec`s) into the outer
`LoopSpec`.  No special handling is needed: the recursion falls out
naturally from the `CountedLoopSchedulerNode` tree structure.

### Serialization in `codegen_kernel()`

`codegen_kernel()` already iterates `self.op_specs` to emit Python source.
A `LoopSpec` entry is serialized as:

```python
LoopSpec(
    count=sympify('K'),
    body=[
        OpSpec(...),
        LoopSpec(          # nested loop
            count=sympify('J'),
            body=[
                OpSpec(...),
            ]
        ),
    ]
)
```

The `arg_index` fixup loop (which maps tensor names to kernel argument
positions) runs before serialization.  It must walk the `LoopSpec` tree
recursively to find all `TensorArg` objects inside nested bodies, not
just the top-level `self.op_specs` list.

### `bundle.mlir` generation for loops

`generate_bundle` in `bundle.py` currently emits one
`sdscbundle.sdsc_execute` line per `OpSpec`.  When a `LoopSpec` is
present it must instead emit an `scf.for` block in `bundle.mlir`
wrapping the execute calls for the body ops.

The loop induction variable is an `index` type running from `0` to
`count` with step `1`.  Because `count` may be a symbolic shape, it is
passed as an MLIR `index` value materialized from the kernel's dynamic
shape arguments.  The concrete shape value is available at runtime via
the same mechanism used to pass dynamic tensor sizes to the Spyre
runtime.

Sketch of the emitted MLIR for a single-level loop with two body ops:

```mlir
func.func @sdsc_bundle() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %K  = <runtime shape value for loop count>
  scf.for %iv = %c0 to %K step %c1 {
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1.json"}
  }
  return
}
```

For nested loops, `scf.for` blocks are nested in the same way:

```mlir
scf.for %iv0 = %c0 to %K step %c1 {
  sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}
  scf.for %iv1 = %c0 to %J step %c1 {
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1.json"}
  }
}
```

`generate_bundle` is refactored to walk the `list[OpSpec | LoopSpec]`
recursively, maintaining an indentation level and a flat list of
`SDSCSpec`s to compile.  The SDSC JSON files are numbered in
depth-first order across all nesting levels.

The `sdsc_execute` attributes referencing filenames are unchanged; only
the surrounding control flow structure in `bundle.mlir` is new.

## Files changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/op_spec.py` | Add `LoopSpec` dataclass (recursive body type) |
| `torch_spyre/_inductor/spyre_kernel.py` | Add `SpyreKernel.wrap_op_specs_in_loop()`; extend `codegen_kernel()` to serialize `LoopSpec` recursively; fix `arg_index` fixup to walk nested bodies |
| `torch_spyre/_inductor/scheduler.py` | Add `CountedLoopSchedulerNode`; add `_codegen_counted_loop()` to `SuperDSCScheduling` |
| `torch_spyre/_inductor/passes.py` | Add `coarse_tile()` call in `CustomPreSchedulingPasses.__call__()` |
| `torch_spyre/_inductor/patches.py` | Register `_post_fusion_custom_pass` hook alongside existing `_update_scheduler` patch |
| `torch_spyre/_inductor/coarse_tile.py` | New file: `coarse_tile(operations)` pass; stamps tuple `loop_group_id` paths and `loop_count` on ops, rewrites `ranges` |
| `torch_spyre/_inductor/codegen/bundle.py` | Extend `generate_bundle()` to walk `LoopSpec` tree and emit `scf.for` in `bundle.mlir`; number SDSC JSON files in depth-first order |
| `tests/inductor/test_coarse_tiling.py` | New file: tests for the end-to-end tiling pipeline including nested loops |

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

## Implementation and testing plan

The goal of this plan is a working prototype that is sufficient for PR
#1984 (`chunk_large_tensors`) to be rewritten on top of it.  That pass
currently avoids the loop IR entirely — it emits a flat sequence of
`ComputedBuffer` + `Scatter` (overwrite) ops — and the prototype will
provide the alternative: stamp the ops with loop-group attributes and let
the loop IR carry the structure through to `bundle.mlir`.

The plan is divided into four milestones.  Each builds on the previous
and has an independent testability story.

---

### Milestone 1 — `LoopSpec` data structure and serialization

**Scope**: `op_spec.py` and `spyre_kernel.py` only.  No scheduler
changes, no IR pass changes, no `bundle.py` changes.

**Work items**:

1. Add `LoopSpec` to `op_spec.py`:

   ```python
   @dataclasses.dataclass
   class LoopSpec:
       count: sympy.Expr
       body: list  # list[OpSpec | UnimplementedOp | LoopSpec]
   ```

2. Extend `SpyreKernel.codegen_kernel()` to serialize `LoopSpec`
   recursively alongside the existing `OpSpec` / `UnimplementedOp`
   branches.  The serialized form is a nested Python literal:

   ```python
   LoopSpec(
       count=sympify('K'),
       body=[
           OpSpec(...),
       ]
   )
   ```

3. Fix the `arg_index` fixup loop to walk `LoopSpec.body` recursively
   so that `TensorArg` objects inside loop bodies get correct kernel
   argument indices.

4. Add `SpyreKernel.wrap_op_specs_in_loop(count: sympy.Expr)` which
   replaces `self.op_specs` with
   `[LoopSpec(count=count, body=self.op_specs)]`.

**Test**: Unit tests in `tests/inductor/test_loop_spec.py` that
construct `LoopSpec` / `OpSpec` objects directly, call
`codegen_kernel()` on a `SpyreKernel` with a manually assembled
`op_specs` list that includes `LoopSpec` entries, and assert the
serialized Python source round-trips correctly (i.e., `eval()` of the
output reconstructs the original objects).  No compilation or hardware
needed.

---

### Milestone 2 — `CountedLoopSchedulerNode` and post-fusion pass

**Scope**: `scheduler.py` and `patches.py`.  No IR pass, no codegen
changes beyond Milestone 1.

**Work items**:

1. Add `CountedLoopSchedulerNode(GroupedSchedulerNode)` to
   `torch_spyre/_inductor/scheduler.py` as described in the design.

2. Add `_build_loop_scheduler_nodes(nodes: list[BaseSchedulerNode])
   -> list[BaseSchedulerNode]` in `scheduler.py`.  This is the
   post-fusion pass body: scans for contiguous runs sharing a
   `loop_group_id` path prefix, asserts consistency of `loop_count`,
   and wraps each run in a `CountedLoopSchedulerNode`.  Handles
   nesting by recursing on the stripped path.

3. Register the pass in `patches.py` by setting
   `config._post_fusion_custom_pass = _build_loop_scheduler_nodes`
   alongside the existing `_update_scheduler` monkey-patch.

4. Extend `SuperDSCScheduling.codegen_node()` with the
   `isinstance(node, CountedLoopSchedulerNode)` branch and
   `_codegen_counted_loop()` as designed.  At this milestone
   `_codegen_counted_loop` calls `wrap_op_specs_in_loop` (Milestone 1)
   and then falls through to the same `define_kernel` / `call_kernel`
   path as today, so the serialized kernel source will contain a
   `LoopSpec` literal but the bundle compiler does not yet consume it.

**Test**: Unit tests in `tests/inductor/test_counted_loop_node.py`
that construct a minimal `Scheduler` over a small set of
`SchedulerNode`s whose underlying `ir.Operation` objects have been
stamped with `loop_group_id` and `loop_count` attributes, run the
post-fusion pass, and assert:

- The output list contains a `CountedLoopSchedulerNode` in the right
  position.
- Its `loop_count` is correct.
- Its `get_nodes()` returns the right constituent nodes.
- Non-grouped nodes are passed through unchanged.
- A non-contiguous group (dependency crossing a group boundary) raises
  an error.
- A two-level nested group produces a nested
  `CountedLoopSchedulerNode`.

These tests can be run without a Spyre device by constructing the IR
objects directly.

---

### Milestone 3 — IR-level loop group stamping (`coarse_tile.py`)

**Scope**: New file `torch_spyre/_inductor/coarse_tile.py` and a
one-line addition to `passes.py`.

**Work items**:

1. Implement `coarse_tile(operations: list[Operation]) -> None`.

   The function signature and pass placement are the same whether the
   caller is a standalone coarse-tiling pass or a rewritten
   `chunk_large_tensors`.  For the prototype, the implementation can be
   minimal: accept a pre-computed `list[tuple[list[Operation], Expr]]`
   (groups and their counts) as a parameter, and stamp the
   `loop_group_id` path and `loop_count` attributes.  The
   decision logic (which ops to group and what count to use) is
   intentionally left to the caller for the prototype so that PR #1984
   can supply its own grouping policy.

2. Insert `coarse_tile(operations)` in `CustomPreSchedulingPasses`
   after `insert_padding_ir` and before `span_reduction`, guarded by a
   feature flag (`config.coarse_tiling`, default `False`) so it is
   inert until callers opt in.

3. Add `coarse_tile.py` to the `uuid()` file list in
   `CustomPreSchedulingPasses` so the Inductor FX cache is invalidated
   when the pass changes.

**Connection to PR #1984**: `chunk_large_tensors._chunk_op()` currently
produces a flat sequence of `ComputedBuffer` + `Scatter` ops.  Once
Milestone 3 is in place, it can instead call `coarse_tile()` to stamp
the ops it produces with the appropriate `loop_group_id` and
`loop_count`, letting the loop IR carry the structure rather than
relying on the flat scatter pattern.  The two approaches can coexist
during transition: `chunk_large_tensors` can keep its current output
while the loop-IR path is validated, then switch.

**Test**: Integration tests in `tests/inductor/test_coarse_tile_pass.py`
that construct a small `list[ir.Operation]`, call `coarse_tile()` with
an explicit grouping, and assert:

- Each op in a group has the correct `loop_group_id` path and
  `loop_count`.
- Ops outside any group have no `loop_group_id`.
- Nested groups produce the correct path tuples.
- The pass is a no-op when passed an empty grouping.

No scheduler, no codegen, no hardware needed.

---

### Milestone 4 — `bundle.mlir` `scf.for` emission

**Scope**: `bundle.py` only.

**Work items**:

1. Refactor `generate_bundle` to separate SDSC JSON compilation (the
   flat `compile_op_spec` loop) from `bundle.mlir` emission.

2. Add a recursive helper
   `_emit_bundle_mlir(specs, indent, file, sdsc_files)` that walks
   `list[OpSpec | LoopSpec]` depth-first:
   - For each `OpSpec`: emit one `sdscbundle.sdsc_execute` line.
   - For each `LoopSpec`: emit an `scf.for` block, recurse into
     `body`, emit closing brace.
   - Number SDSC JSON files in the depth-first traversal order.

3. Emit the `arith.constant` and shape-value preamble for loop counts.
   For the prototype, symbolic `loop_count` expressions are
   concretized with `_concretize_for_sdsc` (already used in
   `superdsc.py`) and emitted as `arith.constant`.  Dynamic/symbolic
   emission (reading a runtime shape value) is a follow-up.

**Test**: Unit tests in `tests/inductor/test_bundle_mlir.py` that call
`generate_bundle` with a manually constructed `list[OpSpec | LoopSpec]`
(using stub `OpSpec` objects with enough data to pass `compile_op_spec`)
and assert the content of the generated `bundle.mlir`:

- A flat list produces the existing `sdsc_execute` pattern.
- A single `LoopSpec` wrapping two `OpSpec`s produces a single
  `scf.for` with two `sdsc_execute` lines inside.
- A nested `LoopSpec` produces nested `scf.for` blocks.
- SDSC JSON files are numbered in depth-first order.

---

### Milestone ordering and PR #1984 integration point

```
M1 (LoopSpec serialization)
  └─ M2 (CountedLoopSchedulerNode)
       └─ M3 (coarse_tile pass)   ← PR #1984 can use this
            └─ M4 (bundle.mlir scf.for)
```

PR #1984 needs Milestones 1–3 to be able to use the loop IR.
Specifically, after M3 merges, `chunk_large_tensors` can:

1. Compute its chunk grouping as it does today.
2. Call `coarse_tile(operations, groups=[(chunk_ops, loop_count)])`
   instead of inserting the `Scatter`/overwrite pattern.
3. The rest of the pipeline (M2's post-fusion pass → M1's
   `LoopSpec` serialization → M4's `scf.for`) handles the rest.

M4 is needed for end-to-end hardware correctness but not for the
pipeline integration test that validates the loop structure is correctly
threaded through the scheduler and codegen stages.

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
loop count, does not rewrite iteration spaces, and is unpacked back to a
flat list at codegen time.  It is nevertheless useful as the *base class*
for `CountedLoopSchedulerNode` because it already implements dependency
merging, ordering constraints, and `name_to_fused_node` bookkeeping.

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
