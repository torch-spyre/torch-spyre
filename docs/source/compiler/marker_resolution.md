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

## Why the distinction matters (issue #4581)

Three sites test for "is this a marker" via `_marker_dim(op) is not None`:
`coarse_tile.py`'s coarse-tile planning and read-copy-planning sites
correctly mean "any marker, either kind" (a kept marker has nothing for
either of those passes to do). `_synthesize_dim_hints_for_group`
(`for_each_tile_lowering.py`) means something narrower -- "skip only a
marker whose transform has already been fused elsewhere" -- but used the
same broad test, silently starving a `STAR_DEP_KEPT` marker's own upstream
read of the synthesized `DimHint` a nesting level above it needs to resolve
provenance. `marker_resolution` lets that site ask the narrower question
directly.

## Where this fits in the pipeline

`splice_while_loops` is the pipeline's first `CustomPreSchedulingPasses`
pass. Marker consumption happens inside it, immediately after each
`WhileLoop` body is spliced. The synthesized `DimHint`s this produces feed
`coarse_tile.py`'s `_hint_ranges_pos`/`lookup_marker_dim`, several passes
later.

## Open areas (not yet in this doc)

- A worked, real-IR example (in `coarse_tiling_loops.md`'s style) once a
  concrete nested-fixture trace is available to capture.
- How arbitrary nesting depth and sibling `WhileLoop`s interact with this
  mechanism in practice -- validated by the fixture family in
  `tests/inductor/for_each_tile_fixtures.py` (triple-nesting and
  sibling-nesting variants), but not yet written up here.
- The full marker-map registry lifecycle (`_MARKER_MAPS`,
  `clear_marker_maps`) and its per-compile scoping.
