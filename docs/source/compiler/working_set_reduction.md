# Working Set Reduction - Design Document

Working set reduction decomposes operations or sequences of operations into
loops doing computations in a piecewise manner, for instance decomposing a
large matrix multiplication `x @ y` into a series of multiplications on groups
of `x`'s rows. The resulting operations operate on smaller tensors with the
following benefits:

- Smaller tensors help alleviate hardware limitations with respect to per-core,
  per-tensor DDR/HBM access span.
- Smaller tensors help reduce memory bandwidth pressure by making it possible
  to keep tensors in scratchpad memory.

This document motivates and walks through the working set reduction approach
adopted in torch-spyre.

**Quick navigation:**

- [Approach](#approach)
- [`for_each_tile`: an explicit, co-indexed tiling loop](#for_each_tile-an-explicit-co-indexed-tiling-loop)
- [Example: tiling `y = a + b; z = y * c`](#example-tiling-y--a--b-z--y--c)
- [Composing and nesting](#composing-and-nesting)
- [Implementation](#implementation)
- [Related documents](#related-documents)

## Approach

We intend to support both implicit (compiler generated) and explicit (source
code driven) working set reduction. Explicit working set reduction lets us
decouple the effort on working set reduction heuristics from downstream tasks
(intermediate representations, analyses, and transformations). Eventually, the
combination of the two can result in better performance and productivity than
either solution in isolation.

The classic illustration is a matrix multiplication. Given `z = x @ y` with
`x: [M, K]` and `y: [K, N]`, multiple tiling choices are valid: tile `x`
along `M`, tile `y` along `N`, tile both, or tile the reduction axis `K`.
Tiling along non-reduction axes produces independent output tiles. Tiling
along the reduction axis is qualitatively different: each tile produces a
partial sum, and an extra accumulation step combines them.

:::{figure} ../_static/images/wsr/matmul-tiling-options.png
:alt: Four tiling options for z = x @ y
:width: 760px
:align: center

Tiling options for a matrix multiplication. Options 1-3 tile non-reduction
axes; each tile is independent. Option 4 tiles the reduction axis K and
introduces an extra accumulation step.
:::

The explicit frontend for this is **`for_each_tile`**, a prototype
higher-order op (torch-spyre#3965) that expresses a tiling loop directly at
the source level.

## `for_each_tile`: an explicit, co-indexed tiling loop

`for_each_tile` is `scan` with the tiling made explicit: **one co-indexed loop
level** that reduces every operand to a per-step tile (a narrow view, a whole
invariant, or a gathered pool row), threads an optional carry, and optionally
lays each step's result tile back into a full-size output along one axis.

```python
def for_each_tile(
    body,
    operands,
    *,
    dims,
    tile_size: int,
    init=None,
    out_dim=None,
    reverse: bool = False,
):
    """Run `body` once per tile over a co-indexed tiling of `operands`.

    Args:
        body: ``(carry, tiles) -> (next_carry, out_tile)``. ``tiles`` arrives in
            operand order, each operand already reduced to its per-step tile.
            ``next_carry`` is ignored in map mode (``init=None``) and ``out_tile``
            in reduction mode (``out_dim=None``). Same restriction as ``scan``:
            the body may not alias input to output or output to output.
        operands: flat sequence of every input -- sliced, gathered and invariant.
        dims: per-operand tile spec as a tuple, or a single spec broadcast to all of
            them. ``int d`` slices the operand into ``tile_size``-wide contiguous views
            along ``d``; ``None`` passes it whole every step; ``Gather(axis, index)``
            takes one pool row per step.
        tile_size: the tile's size along each tiled axis. The loop's trip count is
            derived: ``shape[dim] // tile_size`` per sliced operand, ``len(index)`` per
            gathered one. All of them must agree.
        init: carry init (tensor or pytree of tensors); ``None`` means no carry.
        out_dim: ``int d`` lays step ``i``'s tile at ``narrow(d, i*extent, extent)``
            of the returned output. ``None`` means the body emits no tile.
        reverse: visit tiles high to low. The output still lands in natural order.

    Returns:
        ``(final_carry, out)``, either of which is ``None`` for the unused mode.
    """
```

`dims=`/`tile_size=` directly describe the tiling of the arguments passed to
`for_each_tile` itself, and the trip count is derived from the operand
shapes rather than supplied by hand. Three kinds of operand are supported
per the `dims` entry:

- **Sliced** (`dims[i]` is an `int`): the operand is cut into `tile_size`-wide
  contiguous views along that dimension; step `i` sees
  `narrow(dim, i * tile_size, tile_size)`.
- **Invariant** (`dims[i]` is `None`): the operand is passed whole, unchanged,
  every step.
- **Gathered** (`dims[i]` is a `Gather(axis, index)`): step `i` sees one row
  of a pool tensor, selected by `index[i]` along `axis`
  (`pool.index_select(axis, index[i])`).

`init`/`out_dim` cover the two directions data can cross the loop boundary
that a purely elementwise tiling doesn't need: `init` threads a **carry**
(e.g. a running accumulator) from one step to the next, and `out_dim` lays
each step's output tile back into the correct slice of a full-size result.
Either can be `None` independently: a pure reduction has no `out_dim`; a
pure per-tile map has no `init`.

Set `SPYRE_BACKEND_LOOP_UNROLL=0` before importing torch-spyre to preserve counted
loops in the SDSC-bundle backend. Unrolling is enabled by default. Values
`1`/`true`/`yes` enable it and `0`/`false`/`no` disable it (case-insensitive).
Invalid values raise an error.

Python callers can override the resolved setting during compilation:

```python
from torch_spyre._inductor import config

with config.patch(backend_loop_unroll=False):
    result = compiled_model(inputs)  # First call triggers compilation.
```

The resolved boolean is included in the kernel cache key and passed explicitly
to each compile worker as `dbo-opt --enable-loop-unroll=0` or `1`. Equivalent
environment spellings share a cache key; changing the effective setting cannot
reuse an artifact compiled with the other setting. This controls the SDSC-bundle
frontend only, not the direct `--from-ktir` path or every loop optimization.
Preserving device loops requires autopilot; Deeptools rejects disabling unrolling
with `DT_OPT=autopilot=0`.

For compiled SDPA, maps over batch, head, group, or query positions use this
carry-free form when K/V fits in one block: each map body computes stable
softmax over its complete K/V range. Only a scan over multiple K/V blocks
needs online-softmax running maximum, denominator, and output carries.

The SDPA tile selector (`_select_sdpa_tiling` in `decompositions.py`) admits
a plan only if its resident floor fits in LX: the two co-live score tiles
(the QK^T output and its exponentials), plus the scaled query, output carry,
and per-block P@V for a multi-block scan. A spilled score costs far more than
its HBM bytes, so such plans are rejected rather than priced. The selector also
limits compilation to two nested maps and 1024 tile iterations, guarding against
excessive default loop unrolling. If those limits leave no plan whose floor fits,
it takes the buildable plan that spills the fewest score tiles; coarse tiling
remains only for shapes with no buildable plan. Overflow above the floor is not
charged: across the measured plans it
had no measurable cost. The admitted plans are ranked by predicted device time
from four measured terms: a fixed cost per outer map tile, an extra cost per
tile when the plan splits query rows, the query-shaped carry each K/V scan
step reads and writes, and HBM traffic that depends on the plan (K/V replay
and head-tile staging). The coefficients come from forced-plan timings over
full and chunked prefill, including Gemma 4 global and sliding layers. The
selector chooses tiles before scratchpad planning, so it can only estimate the
allocator's placement; a placement-aware selector would remove that gap.

Sliding-window attention (`_select_swa_tiling`) applies the same resident
floor to its per-query-block K/V window and takes the fewest blocks whose
floor fits: each extra block adds a scan step that rescales the output carry.
A tiling that covers the window exactly with an odd number of blocks is
skipped, because the K/V loop then slices the cache view directly and its
strides are not divisible by the tile count.

## Example: tiling `y = a + b; z = y * c`

The example below tiles the same computation used throughout this document's
companion, [`coarse_tiling_loops.md`](coarse_tiling_loops.md), with
`for_each_tile`. `a`, `b`, `c` are `[1024, 4096]` tensors; the loop tiles
dimension 0 into 8 steps of 128 rows each:

```python
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

a = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
b = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
c = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")

def fn(a, b, c):
    def body(_, tiles):
        a_tile, b_tile, c_tile = tiles
        y_tile = a_tile + b_tile
        return None, y_tile * c_tile

    _, z = for_each_tile(body, (a, b, c), dims=(0, 0, 0), tile_size=128, out_dim=0)
    return z

print(torch.compile(fn)(a, b, c))
```

All three operands are sliced along dimension 0 with the same `tile_size`, so
the trip count is `1024 // 128 == 8`. `body` receives one `[128, 4096]` tile
of each input per step, has no carry (`init=None`, so its first return value
is ignored), and returns each step's `[128, 4096]` result tile, which
`out_dim=0` lays into the corresponding `narrow(0, i*128, 128)` slice of the
returned `[1024, 4096]` output.

This is the same tiling shape as the
[Small Example](coarse_tiling_loops.md#small-example) in the companion
document (a single loop, `y = a + b` then `z = y * c`) and it lowers to the
same downstream mechanism (`loop_info: CoarseTileInfo`, a `CountedLoopSchedulerNode`,
a `LoopSpec`). `docs/tools/capture_for_each_tile_ir.py` regenerates the real,
captured IR/OpSpec/`bundle.mlir` for this exact example, which the
[implementation reference](coarse_tiling_loops.md) quotes at length.

## Composing and nesting

`for_each_tile` expresses exactly **one** co-indexed loop level per call. A
nested tiling loop nest is expressed by composing two `for_each_tile` calls,
with the inner call inside the outer call's `body`:

```python
def fn(a, b, c):
    def outer_body(_, outer_tiles):
        a_row, b_row, c_row = outer_tiles

        def inner_body(_, inner_tiles):
            a_tile, b_tile, c_tile = inner_tiles
            y_tile = a_tile + b_tile
            return None, y_tile * c_tile

        _, z_row = for_each_tile(
            inner_body, (a_row, b_row, c_row), dims=(1, 1, 1), tile_size=1024, out_dim=1
        )
        return None, z_row

    _, z = for_each_tile(outer_body, (a, b, c), dims=(0, 0, 0), tile_size=512, out_dim=0)
    return z
```

The compiler's lowering pipeline (see [Layer 1's Prove → Splice → Identify →
Stamp
sequence](coarse_tiling_loops.md#prove-splice-identify-stamp-how-a-for_each_tile-call-becomes-loop_info))
runs to a fixed point over nested `for_each_tile`/`while_loop` structures, so
this composition is handled uniformly rather than as a special case. See
that section for how a two-level nest like this ends up with a two-entry
`loop_group_id` on the innermost ops, analogous to the Small Example's
`(0, 0)`.

## Implementation

`for_each_tile` is implemented as a thin frontend over PyTorch's `scan`
higher-order op: it validates and normalizes `dims`/`tile_size` into the
per-operand `TileSpec`s that decide how each operand is reduced to a tile
(slice, gather, or pass through invariant), builds the `scan` body, and calls
`torch._higher_order_ops.scan.scan`. This means a `for_each_tile` call reaches
Inductor as an `ir.WhileLoop` (`scan`'s own lowering), not as a distinct
"tiling loop" IR node in its own right.

The Spyre backend's job is then to recognize which `ir.WhileLoop`s are
provably bounded, tile-shaped loops (as opposed to genuinely
data-dependent `while_loop`s, which stay as `ir.WhileLoop`) and rewrite them
into a `loop_info`-carrying representation. The mechanics of that
recognition and rewrite (`try_prove_for_each_tile`, `splice_while_loops`,
`_stamp_direct_loop_info`, and the surrounding carry machinery in
`while_loop_bridge.py`) are described in detail in
[`coarse_tiling_loops.md`](coarse_tiling_loops.md#layer-1-pre-scheduling-ir-pass),
which covers everything past "a run of ops carries a `loop_info:
CoarseTileInfo`": the `CountedLoopSchedulerNode` scheduler wrapper (Layer 2)
and the `LoopSpec` codegen tree (Layer 3).

`splice_while_loops` runs in `CustomPreSchedulingPasses`, before dead-code
elimination, since it must resolve every `for_each_tile`-shaped
`ir.WhileLoop` down to a `loop_info`-carrying group before later passes see
a flat op list. Work-division and scratchpad planning run after, consuming
the resulting tiled iteration spaces. See
[`coarse_tiling_loops.md`](coarse_tiling_loops.md#groups-derivation-and-placement-in-custompreschedulingpasses)
for the full pass ordering.

### Transformation

The annotated IR is transformed into a tiled loop nest by the
**coarse-tiling** machinery. Each contiguous run of operations
sharing the same tiling decision is rewritten with reduced per-iteration
ranges, wrapped in a counted loop, and emitted as nested `LoopSpec` structures
that the SuperDSC codegen lowers to hardware MLIR (`scf.for` + `affine.apply`
- `sdsc_execute`).

The reduction in working set is what makes intermediates fit in LX
scratchpad: an intermediate buffer that is produced and consumed inside the
same loop iteration never needs an HBM allocation.

:::{figure} ../_static/images/wsr/memory-access-before-after.png
:alt: Memory access pattern before and after working set reduction
:width: 880px
:align: center

For `y = a + b; z = y * c`: without WSR, the intermediate `y` is full-size
and spills to HBM. With WSR, each tile of `y` lives in LX scratchpad for the
duration of one iteration and is consumed immediately by the next op.
:::

The full mechanics (how loop identity is carried through Inductor's
flat-list pipeline, how the loop perimeter prevents cross-group fusion, how
buffers crossing the loop boundary are classified) are documented in
[`coarse_tiling_loops.md`](coarse_tiling_loops.md).

A buffer that crosses the loop boundary and has a non-empty
`output_tiled_dims`/`tiled_dims_per_read` entry at a given level (see
[`coarse_tiling_loops.md`](coarse_tiling_loops.md#attribute-contract-on-iroperation))
advances its base address once per loop iteration at that level, so its HBM
pool allocation must be sized for every tile it will occupy across the
loop's run, not just one. `hbm_pool_planning.py`'s `_compute_size_bytes` sizes
each buffer from its full `FixedTiledLayout.device_layout.device_size`, which
spans every tile the buffer occupies across the loop; sizing it for a single
tile would let the loop overrun into whatever buffer the allocator packed
next to it. A buffer whose `output_tiled_dims`/`tiled_dims_per_read` entries
are empty at every level, by contrast, never advances and is a candidate for
a fixed-address LX scratchpad slot instead. See
[`coarse_tiling_loops.md`](coarse_tiling_loops.md) for the `accum_full` /
`accum_tile` buffers this distinction matters most for.

## Related documents

- [`coarse_tiling_loops.md`](coarse_tiling_loops.md): implementation
  reference for the transformation stage (Layer 1 IR pass, Layer 2
  scheduler wrapper, Layer 3 codegen tree) that turns `for_each_tile`'s
  `loop_info` into an executable tiled loop nest.
- [RFC 1358: Coarse-Tiling Loop IR Design
  Rationale](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md),
  which explains the reasoning behind the three-layer design.
- [`scratchpad_planning.md`](scratchpad_planning.md): how LX scratchpad
  allocation consumes the per-tile iteration spaces produced by WSR.
- [`work_division_planning.md`](work_division_planning.md): how work
  distribution across cores runs after WSR on the reduced ranges.
