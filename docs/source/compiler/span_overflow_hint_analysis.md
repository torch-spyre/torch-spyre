# Span-Overflow Hint Analysis

## Background

Spyre work division must keep each core's memory span within the hardware
addressing limit.  For large tensors, a pointwise operation can have a physical
device layout whose per-core span is still too large after normal
`work_division` splitting.  User-authored `spyre_hint` scopes can fix this by
asking coarse tiling to run the operation in multiple outer loop iterations.

`span_overflow_hint_analysis.py` is the compiler-generated version of that
decision for pointwise operations.  It does not transform the graph directly.
Instead, it answers one question:

> Does this pointwise `ComputedBuffer` need coarse tiling, and if yes, which
> output dimension and split count should coarse tiling use?

The result is adapted into the same coarse-tiling group format used by
`spyre_hint`, so the downstream pipeline stays shared:

```
span_overflow_hint_analysis
  -> span_overflow_groups adapter
  -> coarse_tile
  -> CountedLoopSchedulerNode
  -> LoopSpec codegen
```

## Mental Model

Think of the pass as an automatic `spyre_hint` author.  It does not tile
tensors directly and it does not create a separate execution path.  It only:

1. detects that the current physical layout can exceed the 256 MB per-core span
   limit;
2. picks the output dimension that controls that span;
3. computes the smallest legal coarse-tile count for that dimension;
4. validates the real post-tile physical layout; and
5. hands the result to `coarse_tile` as a synthetic `DimHint`.

That means the correctness contract is small: if the planner returns a
`SpanOverflowTilePlan`, the downstream code should behave like a user had
written the equivalent `spyre_hint`.

## Entry Point and Pass Integration

The planner entry point is:

```python
from torch_spyre._inductor.span_overflow_hint_analysis import plan_span_overflow_tile

plan = plan_span_overflow_tile(op, max_cores=config.sencores)
```

`plan_span_overflow_tile()` is intentionally side-effect free: it inspects one
`ComputedBuffer` and returns either a `SpanOverflowTilePlan` or `None`.  It does
not attach hints, rewrite ranges, mutate layouts, or insert loop metadata.

The compiler consumes the planner through the coarse-tiling adapter:

```python
from torch_spyre._inductor.coarse_tile import span_overflow_groups

groups = span_overflow_groups(graph)
```

`span_overflow_groups(graph)` walks `graph.operations`, calls the planner for
each eligible op, attaches a synthetic `DimHint` to planned ops, and returns
groups in the exact format consumed by `coarse_tile()`:

```python
[([op], [(hint_id, split_count, is_reduction_level)])]
```

`passes.py` wires this into `_maybe_coarse_tile()` alongside user coarse-tiling hints:

```python
groups = []
if not config.ignore_wsr_hints:
    reorder_unhinted_interlopers(graph)
    groups += hints_to_coarse_tile_groups(graph)
if not config.ignore_span_overflow_hints:
    groups += span_overflow_groups(graph)
if groups:
    groups.sort(key=graph_order)
    coarse_tile(graph, groups=groups)
```

This ordering is important.  User-authored `spyre_hint` groups take precedence
per operation; automatic span-overflow hints are generated only for eligible ops
that do not already carry user dim hints.  From `coarse_tile()` onward,
automatic groups and manual `spyre_hint` groups share the same IR stamping,
scheduler wrapping, and `LoopSpec` codegen path.

## Scope

The pass is deliberately conservative:

- supports `ComputedBuffer` operations whose `data` is `Pointwise`;
- requires a `FixedTiledLayout`, because decisions are based on Spyre physical
  device layout;
- requires concrete/static layout metadata; symbolic layouts are skipped;
- produces one coarse-tile level over one selected output dimension;
- requires the split count to exactly divide that selected dimension;
- rejects candidates that would cut through physical sticks;
- raises `Unsupported` if the selected dimension cannot be tiled safely;
- does not auto-tile reductions or reduction ranges.

This keeps the pass as a planner.  Coarse tiling owns mutation of
`op.data.ranges`, layout sizes, `CoarseTileInfo`, and scheduler/codegen loop
structure.

## Planner Output

The planner returns a `SpanOverflowTilePlan`:

```python
SpanOverflowTilePlan(
    selected_host_dim=1,
    split_count=5,
    is_reduction=False,
    chunking_info=...,
    reason="output span overflow",
)
```

For example, a pointwise add over shape `[1, 8195, 256, 64]` can produce:

```text
selected_host_dim = 1
split_count       = 5
```

The adapter converts that into a synthetic `DimHint` and group equivalent to a
manual user hint:

```python
with spyre_hint(num_tiles_per_dim={"H": 5}):
    return x + y
```

The coarse-tile group has the usual shape:

```python
[([op], [(hint_id, 5, False)])]
```

where `False` means this is an output-dimension tile, not a reduction-dimension
tile.

## Choosing the Controlling Dimension

The pass walks physical device dimensions from outer to inner, skipping size-1
physical dimensions and the final stick dimension.  For each remaining device
dimension, it uses `stride_map` to find the corresponding logical host/output
dimension.  Exact host-stride matches are preferred over stick-scaled matches so
ambiguous non-standard strides do not select the wrong host dimension.

For a shape like `[1, 8195, 256, 64]`, the physical layout may expose:

```text
host size   = [1, 8195, 256, 64]
host stride = [134266880, 16384, 64, 1]
device size = [8195, 256, 1, 1, 64]
stride_map  = [16384, 64, 64, -1, 1]
```

The first useful physical dimension has `stride_map=16384`, matching host dim
1, so the selected dimension is the `H`-like dimension of size `8195`.

The current policy uses this first mapped span-controlling dimension only.  If
validation fails for that selected dimension, the planner raises `Unsupported`;
it does not try the next mapped dimension or build a multi-dimensional tile plan.

Skipped outer device dimensions are diagnostic context only.  They do not
multiply the span calculation, because constant outer coordinates do not
increase the per-core address span seen by `work_division`.

## Span Estimate

The pass estimates the same quantity that `work_division` cares about:

```python
per_core_span = (
    ceil(selected_device_dim_size / core_split_estimate)
    * selected_device_span_stride_elems
    * itemsize
)
```

where:

- `selected_device_dim_size` is the physical extent of the selected device dim;
- `core_split_estimate` comes from `work_division.core_split`, so analysis and
  work division use the same core-split estimate;
- `selected_device_span_stride_elems` is the product of all inner physical dims,
  including the stick dim;
- `itemsize` is the dtype size in bytes.

There is also a whole-tensor safety check:

```python
total_bytes > MAX_SPAN_BYTES * SENCORES
```

The required coarse-tile count is:

```python
required_count = max(
    ceil(per_core_span / MAX_SPAN_BYTES),
    ceil(total_bytes / (MAX_SPAN_BYTES * SENCORES)),
)
```

If `required_count <= 1`, the op is safe and no automatic hint is emitted.

## Split Count Selection

`coarse_tile` currently divides `op.data.ranges` by the loop count, so the
chosen split count must divide the selected host dimension exactly.  The planner
therefore rounds the required count up to the smallest divisor of the selected
dimension.

Example:

```text
selected host dim size = 8195
required_count         = 5
chosen split_count     = 5
per-tile host size     = 1639
```

If the required count is not itself a divisor, the pass considers larger exact
divisors of the selected dimension.  Candidates must also preserve physical
stick alignment.  If every legal candidate either still overflows or cuts
through a stick, the pass raises `Unsupported` rather than emitting a plan that
still overflows or would corrupt physical stick addressing.

## Internal Planner Flow

The pointwise planner follows this path:

```text
plan_span_overflow_tile()
  -> _plan_pointwise_span_overflow_tile()
       -> _find_outermost_span_dim()
       -> _needs_chunking()
       -> _compute_num_chunks()
       -> _post_tile_validated_split_count()
            -> _choose_divisible_split_count()
            -> _post_tile_stick_alignment_error()
            -> _post_tile_span_ok()
                 -> _post_tile_layout()
```

The important detail is that `_post_tile_validated_split_count()` is not just a
rounding helper.  It is the guardrail that keeps the planner from accepting a
split count until the candidate is exact-divisible, stick-aligned, and safe
under the rebuilt per-tile `SpyreTensorLayout`.

## Post-Tile Validation

The initial span estimate is not the final check.  Before returning a plan, the
pass rejects non-stick-aligned candidates, rebuilds the per-tile
`FixedTiledLayout` that coarse tiling will create, and runs the span check again
on that layout.

For the `[1, 8195, 256, 64]` example:

```text
before tiling: [1, 8195, 256, 64]
split_count:  5
after tiling: [1, 1639, 256, 64]
```

Only if the candidate is exact-divisible, stick-aligned, and passes the
post-tile span/total checks does the planner return the split count.  This
prevents the analysis from approving a tile count that looks safe in the
original layout but is unsafe after Spyre layout reconstruction.

## Adapter and Coarse-Tile Consumption

The adapter lives in `coarse_tile.span_overflow_groups()`.

User `spyre_hint` groups take precedence per operation.  Automatic
span-overflow groups are generated only for eligible pointwise ops that do not
already carry user `dim_hints`.  Mixed graphs can therefore contain both manual
hint groups and automatic span-overflow groups in the same `coarse_tile()` call.

`span_overflow_groups()` returns no groups when `config.chunk_large_tensors` or
`config.ignore_span_overflow_hints` is enabled.

For every returned plan, the adapter:

1. maps `selected_host_dim` to the concrete output loop symbol via
   `op_out_coords(op)`;
2. creates a synthetic `DimHint` with a reserved automatic hint id starting at
   `_SPAN_OVERFLOW_HINT_ID = 10000`;
3. attaches that hint to the op as `op.dim_hints`;
4. returns a group shaped like user-hint output:

```python
([op], [(hint_id, split_count, False)])
```

`coarse_tile()` then uses its normal path:

- resolves `hint_id` back to the op's tiled output dimension;
- divides `op.data.ranges` and `op.layout.size` by `split_count`;
- stamps `CoarseTileInfo` on the op.

For the example above, coarse tiling stamps:

```python
CoarseTileInfo(
    loop_group_id=(0,),
    loop_count=[5],
    loop_tiled_dims=[[1]],
    loop_tiled_reduction_dims=[[]],
)
```

and rewrites the per-tile shape to:

```text
ranges      = [1, 1639, 256, 64]
layout.size = [1, 1639, 256, 64]
```

## Scheduler and Codegen

After `coarse_tile`, the downstream layers do not know whether the loop came
from a user hint or automatic span-overflow analysis.

`build_loop_scheduler_nodes()` sees `CoarseTileInfo` and wraps the scheduled op
run in a `CountedLoopSchedulerNode(count=5)`.

`SpyreKernel` then emits a `LoopSpec`:

```python
LoopSpec(
    count=sympify("5"),
    body=[
        OpSpec(
            op="add",
            iteration_space={
                c0: (1639, 1),
                c1: (256, 4),
                c2: (64, 1),
            },
            tiled_symbols=[c1],
            ...
        )
    ],
    tiled_symbols=[c0],
)
```

This is intentionally the same loop structure produced by the equivalent
manual `spyre_hint`.

## Failure Policy

The pass raises `Unsupported` when automatic tiling would need behavior outside
the current pointwise contract.  Common cases are:

- the selected host dimension is not present in `op.data.ranges`;
- the op has symbolic layout metadata, which is skipped before planning;
- the selected range is symbolic or otherwise not an integer size;
- the selected range is size 1 and cannot be tiled;
- the required split count is larger than the selected dimension;
- every legal divisor cuts through physical sticks;
- no divisor of the selected dimension makes the post-tile layout safe; or
- the adapter cannot map the selected output coordinate to exactly one loop
  symbol.

These failures are intentional.  They prevent the pass from silently emitting a
coarse-tile group that still violates the hardware span limit or cannot be
represented by the existing `coarse_tile` machinery.

## Validation Strategy

Most validation should use small mock or patched-limit tests because true span
overflow usually requires large tensors.

Recommended coverage:

1. **Planner-level:** verify selected dim and split count.
2. **Adapter-level:** verify synthetic `DimHint` and coarse-tile group format.
3. **Coarse-tile IR-level:** verify rewritten ranges/layout and `CoarseTileInfo`.
4. **Scheduler/codegen-level:** verify generated source contains the expected
   `LoopSpec` count.
5. **One E2E smoke test:** use a real large pointwise tensor and compare Spyre
   output against CPU.

## Key Files

| File | Role |
|---|---|
| `torch_spyre/_inductor/span_overflow_hint_analysis.py` | Pointwise planner: choose selected dim and split count |
| `torch_spyre/_inductor/coarse_tile.py` | Adapter (`span_overflow_groups`) and coarse-tile IR stamping |
| `torch_spyre/_inductor/passes.py` | Combines user hint groups and automatic span-overflow groups, then invokes coarse tiling |
| `torch_spyre/_inductor/scheduler.py` | Wraps stamped ops in `CountedLoopSchedulerNode` |
| `torch_spyre/_inductor/spyre_kernel.py` | Emits `LoopSpec` around generated `OpSpec` objects |
| `tests/inductor/test_span_overflow_hint_analysis.py` | Unit/codegen coverage for the planner-to-LoopSpec path |

## Current Limitations

The pointwise implementation is intentionally narrow:

- Only `ComputedBuffer` ops whose `data` is `Pointwise` are planned
  automatically.
- The op must have a `FixedTiledLayout`; the policy depends on Spyre physical
  `device_size` and `stride_map`.
- Symbolic layout metadata is skipped because exact divisor selection and
  post-tile layout validation require concrete sizes.
- The planner emits one coarse-tile level over one output/host dimension.
- `span_overflow_groups()` emits one group per planned op; it does not yet try
  to coalesce producer/consumer chains into a shared automatic group.
- The selected output coordinate must resolve to exactly one loop symbol through
  `op_out_coords(op)`, so the adapter can create a valid synthetic `DimHint`.
- Exact divisibility is required because current `coarse_tile` range rewriting
  divides `op.data.ranges` and `op.layout.size` by the loop count.
- Candidate tiles must stay physical-stick aligned; the pass rejects candidates
  that would cut through sticks.
- If no divisor of the selected dimension makes the post-tile layout safe, the
  pass raises `Unsupported`; it does not try a later mapped dimension, emit
  nested multi-dimensional tile plans, or search split combinations across
  multiple host dimensions.
- If `config.chunk_large_tensors` or `config.ignore_span_overflow_hints` is
  enabled, `span_overflow_groups()` returns no groups.
- Reduction output-range tiling and reduction-range tiling are follow-up work.
