# Spyre Inductor Operation Cookbook

This document describe the common patterns used to define operations
in the front-end compiler.

## Direct mapping from ATen to OpFunc

If a pointwise ATen operation can be implemented with a single Spyre OpFunc,
then enabling it in our backend only requires
adding a method to `SpyreOpFuncs` in [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py).
Canonical examples are `add` and `softplus` (see `softplus` for an example of using `op_info` for non-tensor arguments).

Note that some pointwise ATen operations that can be be implemented with a single Spyre OpFunc
have default decompositions defined by Inductor. Adding a method to
`SpyreOpFuncs` in [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py)
overrides the default decomposition and thus enables the desired direct mapping.
Canonical examples are `reciprocal` and `sigmoid`.

## Spyre-specific decompositions

We define Spyre-specific decompositions in [decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py)
using the `@register_spyre_decompositions` decorator.  Decompositions are graph transformations
that are performed before the graph is lowered to loop level IR.

## Spyre-specific lowerings

We define Spyre-specific lowerings from ATen operations to Inductor's
loop level IR in [lowering.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/lowering.py) using the `@register_spyre_lowering` decorator.

## Spyre-specific OpFuncs

For Spyre OpFuncs that do not have corresponding ATen operations, we use
the `@torch.library.custom_op` decorator to define a new operation in
[customops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/customops.py). This has two pieces:
+ defining the signature of the operation (using `@custom_op`)
+ defining its fake function (using the `@opname.register_fake` that is defined as part of the `@custom_op`)

In addition, when defining a custom op, you will also need to do one of:
+ register a lowering for the custom op in [lowering.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/lowering.py) and
  add a method to `SpyreOpFuncs` in [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py).
  A canonical example is `spyre.clamp`.
+ register a decomposition for the custom op in [decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py)
  that removes the custom op from the graph before lowering. We currently do not have any custom ops that use this option.
+ define a `CustomPrePass` or `CustomPostPass` that implements a more general graph
  rewrite that removes the custom op from the graph before lowering. We currently do not have any custom ops that use this option.

## Custom ops as CPU fallbacks

When an ATen operation cannot run natively on Spyre for certain dtypes but
can be decomposed on Spyre for others, we use a custom op to route the
unsupported cases to a CPU fallback. The pattern is:

1. Define a custom op in
   [customops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/customops.py)
   with `@torch.library.custom_op` and a `register_fake` that returns the
   expected output shape/dtype. The implementation body raises `RuntimeError`
   (it is never called directly).
2. Register a Spyre-specific decomposition in
   [decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py)
   that dispatches to either a native Spyre path or the custom fallback op
   based on dtype or other conditions.
3. Register a CPU fallback for the custom op in
   [fallbacks.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/ops/fallbacks.py)
   using `@register_fallback`.

Canonical examples are `spyre::max_dim_int64_fallback` and
`spyre::min_dim_int64_fallback`, which fall back to CPU for int64 reductions
while the fp16/fp32 cases run natively on Spyre via decomposition.

## Extending finalized kernel schemas

Profiler events identify a compiled kernel bundle by fingerprinting its
finalized `OpSpec`, `TensorArg`, and `LoopSpec` records. Adding an operation
that uses the existing fields requires no provenance-specific change. Changing
one of these dataclass schemas, or introducing a new value type inside
`op_info`, also changes the kernel-provenance contract.

When adding, removing, or changing a finalized-schema field:

1. Update the explicit field set and canonical representation in
   [kernel_provenance.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/kernel_provenance.py).
   Every field must be represented, or deliberately excluded with a comment
   explaining why it cannot affect bundle identity.
2. Bump `KERNEL_PROVENANCE_KEY_VERSION` if the canonical payload or its
   meaning changes. Retain parsing support for every released event-name
   version in `profiler_event.py`; do not reinterpret an existing version.
3. Add determinism and differentiation tests in
   [test_kernel_provenance.py](https://github.com/torch-spyre/torch-spyre/blob/main/tests/inductor/test_kernel_provenance.py).
   Independently reconstructed equivalent specs must produce the same key, and
   identity-relevant changes must produce different keys.
4. Verify the generated-wrapper round trip because cached and fresh compiles
   reconstruct specs before building the descriptor.

The runtime schema check intentionally rejects unhandled dataclass drift.
Descriptor construction then logs a warning and disables only the profiler
join for that kernel; compilation continues rather than emitting a key with
ambiguous meaning.

`op_info` values must have a deterministic canonical representation.
Currently supported values are `None`, booleans, integers, floats,
strings, bytes, SymPy expressions, mappings, and non-string sequences composed
from those types. Do not place process-local identities or volatile runtime
state in a finalized spec. If a new value type is necessary, extend
canonicalization and test equivalent independently constructed values.

## Modifying existing kernels: wrap, never reconstruct

When a compiler pass needs to alter how an existing `ComputedBuffer` computes
its values, always wrap the original `inner_fn` using a `WrapperHandler`
subclass — never attempt to rebuild `inner_fn` from scratch.

**Why:** `inner_fn` index expressions are symbolic; they are bound to the
specific `sympy` objects created during lowering. Rebuilding the function from
scratch produces fresh symbolic expressions that are structurally similar but
not the same objects, causing silent wrong-code bugs that are hard to detect
(see issue [#2797](https://github.com/torch-spyre/torch-spyre/issues/2797)).

**Correct pattern:**

```python
class _MyHandler(WrapperHandler):
    def load(self, name, index):
        # intercept specific loads; delegate everything else
        return super().load(self._name_map.get(name, name), index)

def new_inner_fn(*args, _orig=orig_inner):
    with V.set_ops_handler(_MyHandler(V.ops, ...)):
        return _orig(*args)
```

Canonical implementations: `NameSwapHandler` in
[insert_restickify.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/insert_restickify.py),
`_SplitOpsHandler` and `_IntermediateOpHandler` in
[split_multi_ops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/split_multi_ops.py).

## Preserving provenance in passes

Each op's `debug_handle` is built at codegen from the buffer's `origins`,
`origin_node`, and structured transformation history. These values tie the
emitted kernel back to its source and record how Spyre lower-IR passes derived
it. A buffer with no `origins` gets no handle, while `origins` without a stack
trace produce a handle with no source line. A pass that creates or rewrites a
`ComputedBuffer` must preserve or deliberately remap this provenance.

Use the existing helpers for new rewrite code and when modifying an existing
rewrite rather than setting `origins` by hand. Some legacy passes still copy
`origins` directly; this is migration guidance, not a claim that every existing
site has already been converted.

+ `replace_computed_buffer_body(op, new_data, operations, pass_name=...)` in
  [pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)
  when reconstructing a buffer's body. It forwards `operation_name`, provenance,
  and Spyre operation metadata.
+ `copy_op_metadata(src, dst)` in
  [loop_info.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/loop_info.py)
  carries only Spyre operation metadata such as loop and layout annotations.
  Provenance is intentionally owned by the provenance helpers, not this
  metadata-copy channel.
+ `preserve_provenance`, `merge_provenance`, and `decompose_provenance` in
  [provenance.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/provenance.py)
  at explicit rewrite sites:
  + `preserve_provenance(old, new, pass_name, reason=None)` carries origins, the
    primary node, and existing history through a 1-to-1 reconstruction, then
    appends a `rewrite` record.
  + `merge_provenance(sources, new, pass_name, reason=None)` unions the source
    origins, clears any stale primary node, and appends a `fusion` record.
  + `decompose_provenance(old, news, pass_name, reason=None)` gives each output
    the parent's origins and appends a `decomposition` record. Pass
    `inherit_origins=False` only when every child already has its own deliberate
    semantic FX origin, as `split_multi_ops` does.

Transformation history is an immutable tuple of `ProvenanceTransform` records.
Each record separates `kind`, `pass_name`, and optional `reason`; do not
reintroduce a scalar context attribute. `DebugHandle.transform_history` is the
authoritative serialized form for lower-IR transformation records. The current
helpers emit `rewrite`, `fusion`, and `decomposition`; the schema also supports
`clone` and `remap` for passes that explicitly implement those relationships.

When a lower-IR pass creates a fresh semantic FX node, retain the parent source
lineage while assigning the child's own operation identity. In practice, copy
`stack_trace`, add a `NodeSource` entry to `from_node`, set
`original_aten` to the child target, and make that child the new buffer's
origin. Do not union the parent FX node into the child's origins, because that
would turn a decomposition into an apparent fusion.

`SpyreGraphTransformObserver` wraps every pass in the node and pre-scheduling
pipelines. Node passes are reconciled against the list they return, with fused
and loop scheduler wrappers recursively resolved to their underlying buffers.
It emits a warning when a pass drops any of an existing buffer's `origins`
(even a partial loss, such as a fused buffer going from two sources to one),
clears its `origin_node`, drops transformation-history records, creates a
buffer with no provenance, or removes a buffer without forwarding its
attribution. The observer detects loss but never guesses rewrite semantics or
repairs provenance. If a warning appears, use an explicit helper at the rewrite
site.

Some passes legitimately create source-less buffers (for example padding via
`constant_pad_nd`). Those pass names are listed in
`SOURCELESS_CREATION_PASSES` in `provenance.py` and are exempt only when they
create a new buffer without provenance; the observer still reports provenance
loss on existing buffers reconstructed by the same pass. A pass that
intentionally remaps an existing buffer must separately declare that policy in
`INTENTIONAL_PROVENANCE_REMAP_PASSES`. A pass that intentionally deletes
buffers without forwarding provenance must be listed separately in
`INTENTIONAL_PROVENANCE_REMOVAL_PASSES`.

The observer is disabled by default and follows Inductor's
`trace.provenance_tracking_level`: every positive level, including level 2,
enables it. Set `INDUCTOR_PROVENANCE=1`, or set `TORCH_COMPILE_DEBUG=1` when
`INDUCTOR_PROVENANCE` is unset. Set `INDUCTOR_PROVENANCE=0` for an explicit
force-off override. This gate affects only drop detection; debug-handle
construction and the explicit forwarding helpers remain enabled.
