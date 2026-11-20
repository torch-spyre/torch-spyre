# tile_dim_marker resolution (WIP)

> **Status: work in progress.** This doc covers a narrow, currently-landed
> slice of the `for_each_tile`/`WhileLoop`-splice pipeline: how a
> `tile_dim_marker` op is resolved during marker consumption, and why that
> resolution matters to later passes. It intentionally does not attempt
> `coarse_tiling_loops.md`'s full depth yet — expect this doc to grow as
> the surrounding nested-`for_each_tile` work continues.

## Background

`torch.ops.spyre.tile_dim_marker(tile, dim)` (PR #4559) tags every
`for_each_tile` tile with the dim of its host operand, so that later passes
can recover which dim a tile-shaped read advances along inside a spliced
`WhileLoop` body -- information not otherwise recoverable from the body's
index expressions alone. `_consume_tile_dim_markers`
(`torch_spyre/_inductor/wsr/for_each_tile_lowering.py`) resolves every
marker during `splice_while_loops`, immediately after a `WhileLoop` body is
spliced in.

## The two resolution kinds

Every marker resolves to exactly one of two outcomes, recorded on the op
itself as `tile_marker_resolution` (a `MarkerResolution` enum member, read
via the `_marker_resolution` accessor -- same attribute-on-op pattern as
`tile_marker_dim`/`_marker_dim`):

| Resolution | When | What happens to the marker op |
|---|---|---|
| `INLINE_ERASED` | The marker's single consumer is a `ComputedBuffer` (or gets inlined) | Fused directly into the consumer's `inner_fn`; removed from `operations` and `group_ops` |
| `STAR_DEP_KEPT` | The consumer reaches the marker via a `StarDep`, not an ordinary read dependency | Stays live as a real, addressable buffer -- required both for the StarDep consumer to read it, and to satisfy `coarse_tile.py`'s `_validate_contiguous` gapless-block invariant |

## How later passes tell the two kinds apart (issue #4581)

A pass that only needs to know whether an op is a marker tests
`_marker_dim(op) is not None`. `_marker_resolution(op)` returns the finer
`INLINE_ERASED`/`STAR_DEP_KEPT` value. Only a `STAR_DEP_KEPT` marker stays in
`operations`, so it is the only kind a later pass sees. Its own write does not
carry the loop variable (only its read of the underlying tensor does), and the
marker map that `lookup_marker_dim` reads is keyed by the consumer's name, so
the map cannot resolve the marker's own tiled dim. `_stamp_direct_loop_info`
(`for_each_tile_lowering.py`) and `_hint_ranges_pos`
(`torch_spyre/_inductor/wsr/loop_tiling_common.py`) therefore take that
position from `tile_marker_dim`, which `lower_tile_dim_marker`
(`torch_spyre/_inductor/lowering.py`) stamps on the marker op.

## Where this fits in the pipeline

`splice_while_loops` is the second `CustomPreSchedulingPasses` pass, after
`apply_reserved_dims_contract`. Marker consumption happens inside it,
immediately after each `WhileLoop` body is spliced, and records the marker map
in `_MARKER_MAPS`. `_stamp_direct_loop_info` resolves each op's tiled dim
through `lookup_marker_dim` and appends a minimal `DimHint` (the loop variable
and its trip count) to the op's `dim_hints`. `_hint_ranges_pos` resolves those
hints in later passes.

## Open areas (not yet in this doc)

- A worked, real-IR example (in `coarse_tiling_loops.md`'s style) once a
  concrete nested-fixture trace is available to capture.
- How arbitrary nesting depth and sibling `WhileLoop`s interact with this
  mechanism in practice -- validated by the fixture family in
  `tests/inductor/for_each_tile_fixtures.py` (triple-nesting and
  sibling-nesting variants), but not yet written up here.
- The full marker-map registry lifecycle (`_MARKER_MAPS`,
  `clear_marker_maps`) and its per-compile scoping.
