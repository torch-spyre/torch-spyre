# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Predict the frame a tiling *would* produce, without applying it.

A coarse tiling rewrites an op's iteration ranges, its output layout, and the
indices its deps are written in. A per-core view taken on the committed
(untiled) layout therefore describes the wrong op, and residency is gated on
those views -- so a solver weighing a tiling it has not applied needs the
*tiled* frame as a pure prediction over the un-applied graph.

Every mutation coarse tiling performs already exists in ``coarse_tile.py``. What
is missing, and what this module supplies, is the **inverse**: a pure reading of
the frame ``_divide_ranges`` / ``_post_tile_layout_for_splits`` / ``_rescale_index``
would leave behind, composed from those same helpers rather than restated, so
prediction and application cannot drift.

Scope is deliberately the frame alone. Predicting the *buffers* a candidate
materializes (per-tile scratch, boundary ``full_buf``, reduction accumulator)
belongs with an objective that prices them; coarse tiling carries no objective
term of its own, so that predictor would be API with nothing to consume it and
is not built here.

The frame is stated in the op's *pre-tiling* iteration-symbol namespace, not
the one the tiled op will carry. Applying a tiling re-runs Inductor's
``extract_read_writes -> index_vars_squeeze``, which drops every dim whose
per-tile extent is 1 and renumbers the survivors from a fresh ``d0`` -- names
that do not exist yet at prediction time, and that a prediction could not use
anyway because it is paired with the op's still-committed deps. So the
symbol-carrying fields (``iter_space``, ``write_index``, ``read_index``) are
valid only against *untiled* deps, while the symbol-free ones (``ranges``,
``layout``) are exact against the applied op. See ``_predict_iter_space``.

A candidate that cannot be predicted is reported by returning ``None``, never
by raising. Callers enumerate candidates and price the ones that survive, so an
unpredictable spec is one to drop from the menu rather than a compilation
failure -- and ``_prepare_per_core_view`` already returns ``None`` for a buffer
no candidate can be viewed through, so the branch exists on the caller's side
either way. The reason is logged at debug. Lowering keeps the opposite contract:
by the time ``tile_spec_to_dim_hints`` runs the spec has been chosen, so it
raises ``Unsupported``. Both read the same authority,
``coarse_tile.try_resolve_tile_axis_loop_vars``.

Dependencies stay one-way (``tile_prediction -> coarse_tile``). Nothing here
mutates IR; the solver must not import this module -- the allocator calls the
predictor and hands results across, which is what keeps the solver IR-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy import Expr

from torch._inductor.ir import ComputedBuffer

from ..ir import FixedTiledLayout, _resize_device_layout
from ..logging_utils import get_inductor_logger
from ..pass_utils import (
    iteration_space_from_op,
    op_out_coords,
)
from ..scratchpad.plan_solver import TileSpec
from .coarse_tile import (
    _rescale_index,
    _stick_host_dim,
    reduction_loop_var_by_ranges_pos,
    try_resolve_tile_axis_loop_vars,
)
from .tile import compute_tile_stride

logger = get_inductor_logger("tile_prediction")


@dataclass
class PredictedFrame:
    """The tiled frame a candidate produces for one op -- measured, not applied.

    Only ever built for a candidate that predicts cleanly -- ``predict_frame``
    returns ``None`` rather than a partly-filled frame.

    ``ranges`` / ``reduction_ranges`` are the per-tile extents; ``layout`` is the
    per-tile output ``FixedTiledLayout`` (the op's own layout when untiled);
    ``write_index`` is rescaled to the tile strides while ``read_index`` is the
    op's committed read index unchanged (see ``predict_frame`` -- coarse tiling
    resizes the op's own output buffer, never the buffers it reads); and
    ``iter_space`` maps each loop symbol to its per-tile extent. These are
    exactly the pieces ``_prepare_per_core_view`` consumes via ``view_parts``.

    ``iter_space``, ``write_index`` and ``read_index`` are keyed by the op's
    *pre-tiling* loop symbols (see ``_predict_iter_space``), so they pair only
    with the committed, untiled ``MemoryDep`` that ``_prepare_per_core_view``
    reads. ``ranges``, ``reduction_ranges`` and ``layout`` carry no symbols and
    match the applied op exactly, including when a tiled dim divides to a
    per-tile extent of 1.
    """

    op_name: str
    tiling: TileSpec
    ranges: list
    reduction_ranges: list
    layout: FixedTiledLayout
    write_index: Expr
    read_index: Expr
    iter_space: dict

    def view_parts(self) -> tuple[dict, Expr, Expr]:
        """The ``(iter_space, write_index, read_index)`` tuple
        ``_prepare_per_core_view`` accepts as its ``parts`` argument."""
        return (self.iter_space, self.write_index, self.read_index)


def _try_exact_div(value, count: int):
    """Divide an extent by a tile count, or ``None`` if it does not divide.

    Coarse tiling emits equal-sized tiles, so a concrete extent that is not a
    multiple of the count has no per-tile frame to predict. That is a candidate
    to drop, not an error, so it is reported by value like every other
    rejection in this module -- see :func:`predict_frame`.

    A symbolic extent divides by construction: there is no residue to test
    without a hint, and the applier (``_divide_ranges``) likewise takes the
    symbolic quotient rather than refusing.
    """
    if isinstance(value, (int, sympy.Integer)):
        iv = int(value)
        if iv % count != 0:
            return None
        return sympy.Integer(iv // count)
    return sympy.sympify(value) / count


def _try_div_extents(extents, counts_by_pos: dict[int, int]) -> list | None:
    """``extents`` with each named position divided, or ``None`` if any position
    does not divide exactly."""
    divided = list(extents)
    for pos, count in counts_by_pos.items():
        quotient = _try_exact_div(divided[pos], count)
        if quotient is None:
            return None
        divided[pos] = quotient
    return divided


def _output_and_reduction_counts(tiling: TileSpec):
    """Split a TileSpec into total per-dim counts, output vs reduction."""
    output_counts: dict[int, int] = {}
    reduction_counts: dict[int, int] = {}
    for axis in tiling.axes:
        target = reduction_counts if axis.is_reduction else output_counts
        target[axis.host_dim] = target.get(axis.host_dim, 1) * axis.count
    return output_counts, reduction_counts


def _predict_output_layout(
    op: ComputedBuffer, tiling: TileSpec
) -> FixedTiledLayout | None:
    """The per-tile output ``FixedTiledLayout``, built exactly as
    ``_divide_ranges`` builds it.

    One resize **per tile level**, in ``TileSpec.axes`` order, chaining size,
    stride and device layout -- mirroring ``_divide_ranges``, which runs once
    per level and feeds each result to the next (coarse_tile.py:2211-2220).
    The composition is not associative, so a single full->tile resize is not
    equivalent: ``_resize_device_layout`` matches size-1 device dims to a size-1
    host dim by size alone (ir.py:236, no stride tiebreak and no one-to-one
    constraint), so once an earlier level drives a host dim to extent 1, a
    later resize can re-match a one-stick tile-count dim onto it and collapse
    its stride to the ``-1`` singleton sentinel. A single resize never sees
    that intermediate state and leaves the real stride in place. Measured: 4
    of 104 multi-level combinations diverge, all of that shape.

    Host strides come from ``compute_tile_stride``, not from
    ``contiguous_strides(new_size)``: the latter agrees only when the committed
    layout is contiguous, and silently reorders a transposed or channels-last
    layout (e.g. size [4, 128, 128] stride [128, 1, 16384] tiled to
    [4, 64, 128] yields [64, 1, 8192] applied vs [8192, 128, 1] contiguous).
    ``predict_frame`` feeds these straight to ``_rescale_index`` as the tile
    strides, so a reordered stride mismatches the applied per-core view.

    ``_stick_host_dim`` is re-resolved per level against the running device
    layout, as the applier does -- it recovers the *authoritative* stick host
    dim by coordinate identity, so transposed same-size dims resolve.

    Unlike ``_divide_ranges``, this does not re-check that the layout is a
    ``FixedTiledLayout`` before reading ``layout.device_layout``:
    :func:`_output_layout_rejection` has already rejected anything else, and
    restating the guard here would put a second (silently skipping) authority
    beside the gate.

    Returns ``None`` if any level's host extent does not divide by its tile
    count -- a candidate to drop, propagated by :func:`predict_frame`.
    """
    layout = op.layout
    cur_size = [int(s) for s in layout.size]
    cur_stride = [int(s) for s in layout.stride]
    cur_dev = layout.device_layout
    for axis in tiling.axes:
        if axis.is_reduction:
            continue
        extent = _try_exact_div(cur_size[axis.host_dim], axis.count)
        if extent is None:
            return None
        new_size = list(cur_size)
        new_size[axis.host_dim] = int(extent)
        cur_stride = [
            int(s) for s in compute_tile_stride(cur_size, cur_stride, new_size)
        ]
        cur_dev = _resize_device_layout(
            cur_dev,
            cur_size,
            new_size,
            stick_host_dim=_stick_host_dim(op, cur_dev),
        )
        cur_size = new_size
    return FixedTiledLayout(layout.device, layout.dtype, cur_size, cur_stride, cur_dev)


def _predict_iter_space(
    op: ComputedBuffer,
    output_counts: dict[int, int],
    reduction_counts: dict[int, int],
) -> dict | None:
    """The op's iteration space with each tiled symbol's extent divided down.

    An output axis's loop symbol is the sole free symbol of
    ``op_out_coords(op)[host_dim]``; a reduction axis's is
    ``reduction_loop_var_by_ranges_pos(op)[host_dim]`` -- the same resolution
    ``tile_spec_to_dim_hints`` uses, in the same unsqueezed
    ``reduction_ranges`` frame this function's caller divides.

    Resolves both unguarded: :func:`_rejection_reason` has already established
    that every ``host_dim`` here indexes in range and lands on exactly one loop
    symbol present in the iteration space. Any caller other than
    ``predict_frame`` must check that first -- these counts are positions in two
    different frames, and an unchecked one reads the wrong dim or raises
    ``IndexError``/``KeyError`` rather than being rejected. Returns ``None`` if
    a symbol's extent does not divide by its tile count.

    Keys stay the op's *pre-tiling* symbols; only extents move. That is
    deliberate. The applied op's symbols do not exist yet, and the caller pairs
    this dict with the op's committed (untiled) ``MemoryDep`` --
    ``_prepare_per_core_view`` builds ``dep_coeff`` as
    ``{sym: dep.index.coeff(sym) for sym in iter_space}``. Renaming the keys to
    what the tiled op will carry would break that pairing outright.

    The two namespaces are not interchangeable, and they overlap, so a mismatch
    reads the wrong dim rather than raising. Applying a tiling re-runs
    ``extract_read_writes -> index_vars_squeeze``, whose ``SqueezeView.squeezer``
    drops every dim of size 1 and mints ``d0, d1, ...`` from a fresh counter over
    the survivors; a dim tiled to per-tile extent 1 therefore loses its symbol
    and everything after it renumbers. Ranges ``[4, 128, 256]`` tiled on dim 1 by
    128 predicts ``{d0: 4, d1: 1, d2: 256}`` while the applied op carries
    ``{d0: 4, d1: 256}`` -- ``d1`` in both, meaning different dims. Never match a
    predicted frame against a post-apply dep by symbol. This is confined to the
    dep view: ``_divide_ranges`` keeps the unit dim at full rank, so ``ranges``,
    ``stride`` and ``device_size`` are unaffected.

    The surviving ``sym -> 1`` entry is inert in every consumer -- nothing splits
    a unit dim, and ``_per_core_view_from_prep`` skips ``split <= 1`` before
    device placement. Its one order-sensitive site is that function's
    ``contiguous_dim = len(dim_splits) - 1`` k-fast matmul reorder, which would
    select the phantom instead of the real trailing dim. That is unreachable
    rather than handled: it needs a ``Reduction`` (for ``is_matmul``), and both
    ``TileSpec`` producers reject Reduction unit tiles
    (``enumerate_tilings._reduction_split_counts`` and
    ``span_overflow_hint_analysis._split_candidates_for_host_dim``). If either
    filter is relaxed to admit them, drop the unit entry here instead.
    """
    iter_space = dict(iteration_space_from_op(op))
    out_coords = op_out_coords(op)
    syms_and_counts = [
        (next(iter(out_coords[host_dim].free_symbols)), count)
        for host_dim, count in output_counts.items()
    ]
    if reduction_counts:
        red_vars = reduction_loop_var_by_ranges_pos(op)
        # Both established by :func:`_rejection_reason`, which rejects an op
        # whose reduction positions do not map to loop variables and every
        # host_dim that resolves to none -- the same precondition the output
        # branch above relies on. Asserted rather than re-checked so a caller
        # that skipped the gate fails here instead of silently reading a
        # different dim.
        assert red_vars is not None
        syms_and_counts += [
            (sym, count)
            for host_dim, count in reduction_counts.items()
            if (sym := red_vars[host_dim]) is not None
        ]
    for sym, count in syms_and_counts:
        extent = _try_exact_div(iter_space[sym], count)
        if extent is None:
            return None
        iter_space[sym] = extent
    return iter_space


def _output_layout_rejection(op: ComputedBuffer, tiling: TileSpec) -> str | None:
    """Why ``tiling``'s per-tile output layout is not predictable, else ``None``.

    ``_divide_ranges`` gates its device-layout rebuild on
    ``isinstance(layout, FixedTiledLayout)`` (coarse_tile.py:2205) and returns
    quietly without it, having still rewritten ``ranges`` and the host
    size/stride. :func:`_predict_output_layout` does not mirror that gate -- it
    reads ``layout.device_layout`` unconditionally -- so without this check a
    plain ``FixedLayout`` op raises ``AttributeError: 'FixedLayout' object has
    no attribute 'device_layout'`` out of ``predict_frame``, which reports every
    other rejection by returning ``None``. A caller would have to wrap candidate
    pruning in a bare ``except`` to survive it.

    Rejecting is the conservative direction rather than the faithful one: the
    applier does tile these ops, just without rebuilding the device layout. A
    frame pairing tiled ranges with the *untiled* device layout is exactly the
    half-tiled frame the rest of this gate exists to prevent, and
    ``_prepare_per_core_view`` maps a non-``FixedTiledLayout`` buffer to
    unrepresentable for *every* candidate, so such a frame could never be priced
    anyway. Prediction stays strictly less permissive than application, which is
    the invariant that matters.

    Scoped to a spec carrying an output axis, mirroring the applier: a
    reduction-only spec never rebuilds the layout on either side, so gating it
    here would newly reject ops that predict fine today. The cost is that
    ``PredictedFrame.layout`` is only guaranteed to be a ``FixedTiledLayout``
    when the spec tiles an output dim; a reduction-only frame passes through
    whatever the op committed.

    ``_divide_ranges``'s *other* layout guard, ``len(layout.size) ==
    len(ranges)``, needs no counterpart. A rank mismatch does not survive far
    enough to reach either the applier or the predictor: ``store_output``
    indexes the layout with the op's own iteration vars, so
    ``op.get_read_writes()`` asserts inside ``_fixed_indexer``, and both
    ``resolve_tile_axis_loop_vars`` and ``tile_spec_to_dim_hints`` reach that
    through ``op_out_coords`` first. That guard exists for the symbolic-size
    plain-``FixedLayout`` case, which this function already rejects.
    """
    if all(axis.is_reduction for axis in tiling.axes):
        return None
    layout = getattr(op, "layout", None)
    if not isinstance(layout, FixedTiledLayout):
        return (
            f"{op.get_name()} tiles an output dim but has layout "
            f"{type(layout).__name__}, not FixedTiledLayout; coarse tiling "
            "would leave its device layout untiled, so the per-tile frame is "
            "not predictable."
        )
    return None


def _rejection_reason(op: ComputedBuffer, tiling: TileSpec) -> str | None:
    """Why ``tiling`` cannot be predicted onto ``op``, or ``None`` if it can.

    ``predict_frame``'s single gate, and the reason the private predictors it
    calls resolve each axis unguarded. Axis legality itself is not restated
    here: :func:`try_resolve_tile_axis_loop_vars` is the shared authority, so
    this rejects exactly what ``tile_spec_to_dim_hints`` rejects when it lowers
    the same spec. What is added is the extra reach *prediction* has -- the two
    positional lists, the iteration space it divides, and (via
    :func:`_output_layout_rejection`) the output layout it rebuilds, none of
    which lowering touches.

    The symmetry with lowering is the point. ``predict_frame`` divides
    ``ranges``, ``reduction_ranges`` and the output layout for *every* axis
    unconditionally, so an axis that quietly failed to resolve downstream would
    not drop out of the prediction -- it would return a frame whose ranges and
    layout say "tiled" while its ``iter_space`` still says "untiled", priced by
    the solver as though consistent and only refused much later, at apply time.

    The reduction bound is ``reduction_ranges`` on both sides: ``host_dim`` is
    an unsqueezed position, and :func:`try_resolve_tile_axis_loop_vars` bounds
    it against the same list (via ``reduction_loop_var_by_ranges_pos``), so the
    check here is the resolver's, restated for the extents this module divides
    rather than a second, differently-framed bound.

    Divisibility is deliberately not checked here. It is detected where it is
    computed instead -- ``_try_exact_div`` at each of the four division sites --
    so this gate does not have to restate the level-by-level layout walk or
    assume ``ranges``, ``layout.size`` and the iteration space agree on an
    extent. All four report the same way this does, by value.
    """
    if tiling.is_untiled:
        return None
    loop_vars, reason = try_resolve_tile_axis_loop_vars(op, tiling)
    if reason is not None:
        return reason
    assert loop_vars is not None
    iter_space = iteration_space_from_op(op)
    ranges = list(op.data.ranges)
    reduction_ranges = list(getattr(op.data, "reduction_ranges", []))
    for axis, sym in zip(tiling.axes, loop_vars):
        if axis.is_reduction:
            if axis.host_dim >= len(reduction_ranges):
                return (
                    f"reduction host_dim={axis.host_dim} is out of bounds for "
                    f"reduction ranges {reduction_ranges} on {op.get_name()}."
                )
        elif axis.host_dim >= len(ranges):
            return (
                f"host_dim={axis.host_dim} is out of bounds for data ranges "
                f"{ranges} on {op.get_name()}."
            )
        if sym not in iter_space:
            return (
                f"host_dim={axis.host_dim} on {op.get_name()} resolves to loop "
                f"var {sym}, which is absent from its iteration space "
                f"{dict(iter_space)}."
            )
    # Last, so a spec rejected for both reasons reports the axis reason lowering
    # would report rather than a prediction-only one.
    return _output_layout_rejection(op, tiling)


def predict_frame(op: ComputedBuffer, tiling: TileSpec) -> PredictedFrame | None:
    """The per-tile frame ``op`` would take under ``tiling``, or ``None`` -- no
    IR mutation.

    Output axes shrink ``op.data.ranges`` and the physical output layout (the
    same per-level resize real tiling uses -- see
    :func:`_predict_output_layout`); reduction axes shrink
    ``op.data.reduction_ranges`` only, since the op's own output buffer is the
    accumulator and keeps its full output extent.

    Returns ``None`` for a tiling that cannot be predicted onto ``op``: one
    :func:`try_resolve_tile_axis_loop_vars` could not resolve (so one
    ``tile_spec_to_dim_hints`` could not lower either), one whose output layout
    is not predictable even though lowering would accept it, and one whose
    extents do not divide evenly. Rejection is by return value, not by
    exception, because callers *enumerate* candidates: a spec that cannot be
    predicted is one to drop from the menu, not a compilation failure. The
    reason is logged at debug rather than discarded.

    ``None`` is also what ``_prepare_per_core_view`` returns for a buffer no
    candidate can be viewed through, so a caller pricing candidates already has
    this branch.

    Everything is checked before anything is divided, so there is no partially
    divided frame to return -- but the divisibility of a given extent is
    established at the point of division rather than up front, which is why the
    four division sites below each test for ``None``.
    """
    reason = _rejection_reason(op, tiling)
    if reason is not None:
        logger.debug("dropping tiling %s on %s: %s", tiling, op.get_name(), reason)
        return None

    output_counts, reduction_counts = _output_and_reduction_counts(tiling)
    ranges = _try_div_extents(op.data.ranges, output_counts)
    reduction_ranges = _try_div_extents(
        getattr(op.data, "reduction_ranges", []), reduction_counts
    )
    layout = _predict_output_layout(op, tiling) if output_counts else op.layout
    iter_space = _predict_iter_space(op, output_counts, reduction_counts)
    if ranges is None or reduction_ranges is None or layout is None:
        logger.debug(
            "dropping tiling %s on %s: extents do not divide evenly",
            tiling,
            op.get_name(),
        )
        return None
    if iter_space is None:
        logger.debug(
            "dropping tiling %s on %s: iteration-space extents do not divide evenly",
            tiling,
            op.get_name(),
        )
        return None

    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads if hasattr(d, "index")), write_index)
    if output_counts:
        full_strides = [sympy.sympify(s) for s in op.layout.stride]
        tile_strides = [sympy.sympify(s) for s in layout.stride]
        write_index = _rescale_index(write_index, full_strides, tile_strides)

    return PredictedFrame(
        op_name=op.get_name(),
        tiling=tiling,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
        layout=layout,
        write_index=write_index,
        read_index=read_index,
        iter_space=iter_space,
    )
