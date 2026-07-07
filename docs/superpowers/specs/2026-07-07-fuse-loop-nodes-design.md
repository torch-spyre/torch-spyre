# Design: Fuse `CountedLoopSchedulerNode`s with adjacent plain nodes

**Date:** 2026-07-07
**Branch:** `fuse-it-all`

## Background

`spyre_fuse_nodes` assembles sequences of `SchedulerNode`s into SDSC bundles
(each bundle compiles into one `SpyreKernel` / one `call_kernel` invocation).
Currently, a `CountedLoopSchedulerNode` always forces a bundle boundary: it is
not a `SchedulerNode` subclass, so it falls through to the `else` branch in
`spyre_fuse_nodes`, which flushes the current accumulator and appends the loop
node as its own isolated bundle.

This means a graph like:

```
MatMul → CountedLoop(GEMM×K) → LayerNorm
```

generates three separate kernel invocations, even though there is no
fundamental hardware reason they cannot share one bundle.

## Goal

Allow a SuperDSC bundle to contain any sequence of `CountedLoopSchedulerNode`s
and plain `SchedulerNode`s, in their original topological order.  The loop
node's ops emit as a `LoopSpec` entry inside the shared `SpyreKernel.op_specs`,
alongside the flat `OpSpec` entries from plain nodes.

## Approach

Option A (chosen): teach `spyre_fuse_nodes` to accumulate
`CountedLoopSchedulerNode`s alongside plain nodes, and refactor `codegen_node`
to drive all children of a mixed `FusedSchedulerNode` into one `SpyreKernel`.

Options B (new wrapper class) and C (flatten-then-re-wrap) were rejected: B
adds unnecessary indirection, C loses structural information and is
architecturally wrong.

## Design

### 1. `fusion.py` — accumulation change

Change the guard in `spyre_fuse_nodes` from:

```python
if isinstance(n, SchedulerNode):
```

to:

```python
if isinstance(n, (SchedulerNode, CountedLoopSchedulerNode)):
```

Widen the type annotation of `_make_fused`'s parameter from
`list[SchedulerNode]` to `list[SchedulerNode | CountedLoopSchedulerNode]`.

`_make_fused` itself is unchanged: when `cur_nodes` has more than one entry it
wraps them in a plain `FusedSchedulerNode`; a lone node is returned as-is.
This preserves the existing fast-path: a `CountedLoopSchedulerNode` that ends
up alone in a bundle still reaches `codegen_node` as a bare
`CountedLoopSchedulerNode`, and the existing `_codegen_counted_loop` path
handles it without modification.

The `else` branch (which forces a boundary for `FallbackKernel` and other
non-loop node types) is unchanged.

### 2. `scheduler.py` — shared codegen helper

Introduce a new private method:

```python
def _codegen_into_kernel(
    self,
    nodes: list[BaseSchedulerNode],
    kernel: SpyreKernel,
    all_schedule_nodes: list[SchedulerNode],
) -> None:
```

This iterates `nodes` in order and for each child:

- If `CountedLoopSchedulerNode`: call `_codegen_loop_body(child, kernel,
  all_schedule_nodes)`.  This already slices `kernel.op_specs` at `body_start`,
  codegens the inner nodes, and appends a `LoopSpec` covering only the newly
  added entries.  It is called unchanged.
- Otherwise (plain `SchedulerNode`): call `generate_node_schedule([child])`,
  extend `all_schedule_nodes`, and codegen each resulting `SchedulerNode` flat
  into the kernel.

Refactor `codegen_node` and `_codegen_counted_loop` to use this helper:

**`codegen_node`** (the mixed-bundle path):

```
codegen_node(node):
    if CountedLoopSchedulerNode → _codegen_counted_loop(node)   # unchanged dispatch
    children = node.get_nodes(), filtered for removed_ops
    if empty → return
    kernel = SpyreKernel()
    all_schedule_nodes = []
    with kernel:
        _codegen_into_kernel(children, kernel, all_schedule_nodes)
    finalize: codegen_kernel, define_kernel, mark_run, call_kernel,
              emit_layout_restores, removed_buffers, free_buffers_in_scheduler
```

Previously `codegen_node`'s else branch called `generate_node_schedule` on the
full node list and then codegenned the flattened result.  That approach would
raise `RuntimeError` on any `CountedLoopSchedulerNode` inside a mixed
`FusedSchedulerNode` (because `generate_node_schedule` raises on unexpected
node types).  The new path dispatches explicitly per child.

**`_codegen_counted_loop`** (the top-level loop path):

```
_codegen_counted_loop(node):
    inner_nodes = node.get_nodes(), filtered for removed_ops
    if empty → return
    kernel = SpyreKernel()
    all_schedule_nodes = []
    with kernel:
        _codegen_into_kernel(inner_nodes, kernel, all_schedule_nodes)
    kernel.wrap_op_specs_in_loop(node.loop_count)   # wraps entire kernel
    finalize: same pattern as codegen_node
```

The `wrap_op_specs_in_loop` call still happens here, after all inner nodes are
codegenned, wrapping the entire `kernel.op_specs` list in a single outer
`LoopSpec`.  This is unaffected by the refactor.

### 3. Downstream machinery — no changes required

**`unroll.py` (`unroll_loop_specs`):** Iterates a top-level spec list and
recursively descends into `LoopSpec.body`.  A mixed top-level list
`[OpSpec, OpSpec, LoopSpec(...), OpSpec]` is handled correctly: flat entries
pass through, `LoopSpec` entries are unrolled.

**`bundle.py` (`generate_bundle`):** `_compile_specs`, `_collect_loop_bounds`,
`_collect_affine_maps`, and `_emit_specs` are all depth-first walks that branch
on `isinstance(entry, LoopSpec)` vs `isinstance(entry, OpSpec)` at every
nesting level, including the top level.  Mixed top-level lists produce correct
`scf.for` / `sdsc_execute` interleaving in `bundle.mlir`.

One constraint holds by construction: a flat `OpSpec` at the top level of a
mixed bundle (not inside any `LoopSpec`) has `tiled_symbols=[]` and therefore
`affine_strides=[]` per tensor.  The assertion in `_collect_affine_maps` that
`level_idx < len(loop_var_depth)` (line 471) cannot fire for such ops because
their `affine_strides` lists are empty.  This is guaranteed because only ops
inside a `CountedLoopSchedulerNode` carry `tiled_symbols`, and those ops land
inside a `LoopSpec` body, not at the top level of the mixed bundle.

### 4. Testing

- Extend `tests/inductor/test_coarse_tiling.py` with unit tests for
  `spyre_fuse_nodes` covering:
  - A lone `CountedLoopSchedulerNode` still produces one bundle (regression).
  - A plain node followed by a `CountedLoopSchedulerNode` fuses into one bundle.
  - A `CountedLoopSchedulerNode` followed by a plain node fuses into one bundle.
  - A plain node, loop node, plain node sequence fuses into one bundle.
  - A `FallbackKernel` still forces a boundary even when adjacent to loop nodes.
- Extend `tests/inductor/test_building_blocks.py` (or a new compiled test) with
  an end-to-end compiled test that produces a mixed bundle and verifies
  numerical correctness against CPU.

## Files changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/fusion.py` | Widen `isinstance` guard; widen type annotation |
| `torch_spyre/_inductor/scheduler.py` | Add `_codegen_into_kernel`; refactor `codegen_node` and `_codegen_counted_loop` to use it |
| `tests/inductor/test_coarse_tiling.py` | Unit tests for new fusion behaviour |
| `tests/inductor/test_building_blocks.py` | End-to-end compiled test |
