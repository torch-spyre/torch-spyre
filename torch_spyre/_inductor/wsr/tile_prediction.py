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
symbol-carrying fields (``iter_space``, ``write_index``) are valid only against
*untiled* deps, while the symbol-free ones (``ranges``, ``layout``) are exact
against the applied op. See ``_predict_iter_space``.

The frame covers the op's own loops and its own output buffer, and carries no
read index. What a read becomes is not the op's to decide: once a tiling is
applied, a load of a producer tiled in the same loop group addresses that
producer's per-tile buffer, at its tile strides, while a load of a buffer
outside the group keeps its committed index. Which one a read is depends on the
producer's tiling, which a per-op prediction is not given.

A candidate that cannot be predicted is reported by returning ``None``, never
by raising. Callers enumerate candidates and price the ones that survive, so an
unpredictable spec is one to drop from the menu rather than a compilation
failure -- and ``_prepare_per_core_view`` already returns ``None`` for a buffer
no candidate can be viewed through, so the branch exists on the caller's side
either way. The reason is logged at debug. Lowering keeps the opposite contract:
by the time ``tile_spec_to_dim_hints`` runs the spec has been chosen, so it
raises ``Unsupported``. Both read the same authority,
``scratchpad.coarse_tiling.try_resolve_tile_axis_loop_vars`` -- as does the
enumerator -- so prediction cannot disagree with lowering or enumeration about
which axis a spec names or whether it may name it. A reduction axis names a
position in the *squeezed* reduction loop variables, so on an op with a size-1
reduction dim the ``reduction_ranges`` entry it divides is not the one at
``host_dim``; prediction divides the entry the applier picks for that loop
variable (``_loop_var_to_reduction_ranges_pos``).

Dependencies stay one-way (``tile_prediction -> coarse_tile``, and the
resolver in ``scratchpad.coarse_tiling``). Nothing here
mutates IR; the solver must not import this module -- the allocator calls the
predictor and hands results across, which is what keeps the solver IR-free.

Nothing calls it yet: the allocator seam that prices candidates through it --
per-tiling division enumeration -- arrives separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy import Expr

from torch._inductor.ir import ComputedBuffer

from ..ir import FixedTiledLayout, _resize_device_layout
from ..logging_utils import get_inductor_logger
from ..pass_utils import iteration_space_from_op
from ..scratchpad.coarse_tiling import try_resolve_tile_axis_loop_vars
from ..scratchpad.plan_solver import TileSpec
from .coarse_tile import (
    _loop_var_to_reduction_ranges_pos,
    _rescale_index,
    _stick_host_dim,
)
from .tile import compute_tile_stride
from ..errors import Unsupported


logger = get_inductor_logger("tile_prediction")


@dataclass
class PredictedFrame:
    """The tiled frame a candidate produces for one op -- measured, not applied.

    Only ever built for a candidate that predicts cleanly -- ``predict_frame``
    returns ``None`` rather than a partly-filled frame.

    ``ranges`` / ``reduction_ranges`` are the per-tile extents; ``layout`` is the
    per-tile output ``FixedTiledLayout`` (the op's own layout when untiled);
    ``write_index`` is the op's write index rescaled to the tile strides; and
    ``iter_space`` maps each loop symbol to its per-tile extent. There is no
    read index: what a read becomes depends on its producer's tiling (module
    docstring).

    ``iter_space`` and ``write_index`` are keyed by the op's *pre-tiling* loop
    symbols (see ``_predict_iter_space``), so they pair only with the
    committed, untiled ``MemoryDep``. ``ranges``, ``reduction_ranges`` and
    ``layout`` carry no symbols and match the applied op exactly, including
    when a tiled dim divides to a per-tile extent of 1.
    """

    op_name: str
    tiling: TileSpec
    ranges: list
    reduction_ranges: list
    layout: FixedTiledLayout
    write_index: Expr
    iter_space: dict


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


def _output_and_reduction_counts(op: ComputedBuffer, tiling: TileSpec):
    """Split a TileSpec into total per-dim counts, output vs reduction.

    Output counts are keyed by ``host_dim``, a position in ``op.data.ranges``.
    A reduction ``host_dim`` counts only the non-size-1 reduction dims, so a
    reduction count is keyed by the ``op.data.reduction_ranges`` entry the
    applier divides for that axis's loop variable -- on ``[1, 8, 16]``,
    host_dim 1 is entry 2. Only reached behind ``_rejection_reason``, so every
    axis resolves.
    """
    loop_vars, _ = try_resolve_tile_axis_loop_vars(op, tiling)
    assert loop_vars is not None, "unreachable behind _rejection_reason"
    output_counts: dict[int, int] = {}
    reduction_counts: dict[int, int] = {}
    for axis, loop_var in zip(tiling.axes, loop_vars):
        if axis.is_reduction:
            pos = _loop_var_to_reduction_ranges_pos(op, loop_var)
            # _get_red_var refuses a loop variable the applier cannot place.
            assert pos is not None
            target = reduction_counts
        else:
            pos, target = axis.host_dim, output_counts
        target[pos] = target.get(pos, 1) * axis.count
    return output_counts, reduction_counts


def _predict_output_layout(
    op: ComputedBuffer, tiling: TileSpec
) -> FixedTiledLayout | None:
    """The per-tile output ``FixedTiledLayout``, built exactly as
    ``_divide_ranges`` builds it, or ``None`` where that would fail: an extent
    that does not divide, or a device layout ``_resize_device_layout`` cannot
    resize (e.g. one device dim folding two host dims). Sizes and strides are
    static here: ``_rejection_reason`` has already refused an op with a
    symbolic extent, as the resolver does.
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
            logger.debug(
                "dropping tiling %s on %s: output size does not divide evenly",
                tiling,
                op.get_name(),
            )
            return None
        new_size = list(cur_size)
        new_size[axis.host_dim] = int(extent)
        try:
            new_stride = [
                int(s) for s in compute_tile_stride(cur_size, cur_stride, new_size)
            ]
        except Unsupported as exc:
            logger.debug("dropping tiling %s on %s: %s", tiling, op.get_name(), exc)
            return None
        try:
            cur_dev = _resize_device_layout(
                cur_dev,
                cur_size,
                new_size,
                stick_host_dim=_stick_host_dim(op, cur_dev),
                old_host_stride=cur_stride,
                new_host_stride=new_stride,
            )
        except RuntimeError as exc:
            logger.debug("dropping tiling %s on %s: %s", tiling, op.get_name(), exc)
            return None
        cur_size, cur_stride = new_size, new_stride
    return FixedTiledLayout(
        layout.device, layout.dtype, cur_size, cur_stride, cur_dev, layout.offset
    )


def _predict_iter_space(op: ComputedBuffer, tiling: TileSpec) -> dict | None:
    """The op's iteration space with each tiled symbol's extent divided down."""
    loop_vars, _ = try_resolve_tile_axis_loop_vars(op, tiling)
    if loop_vars is None:  # unreachable behind _rejection_reason
        return None
    counts: dict[sympy.Symbol, int] = {}
    for axis, sym in zip(tiling.axes, loop_vars):
        counts[sym] = counts.get(sym, 1) * axis.count
    iter_space = dict(iteration_space_from_op(op))
    for sym, count in counts.items():
        extent = _try_exact_div(iter_space[sym], count)
        if extent is None:
            return None
        iter_space[sym] = extent
    return iter_space


def _output_layout_rejection(op: ComputedBuffer, tiling: TileSpec) -> str | None:
    """
    Why ``tiling``'s per-tile output layout is not predictable, else ``None``.
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
    """
    Why ``tiling`` cannot be predicted onto ``op``, or ``None`` if it can.
    """
    if tiling.is_untiled:
        return None
    _, reason = try_resolve_tile_axis_loop_vars(op, tiling)
    if reason is not None:
        return reason
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

    output_counts, reduction_counts = _output_and_reduction_counts(op, tiling)
    ranges = _try_div_extents(op.data.ranges, output_counts)
    reduction_ranges = _try_div_extents(
        getattr(op.data, "reduction_ranges", []), reduction_counts
    )
    layout = _predict_output_layout(op, tiling) if output_counts else op.layout
    iter_space = _predict_iter_space(op, tiling)
    if ranges is None or reduction_ranges is None:
        logger.debug(
            "dropping tiling %s on %s: extents do not divide evenly",
            tiling,
            op.get_name(),
        )
        return None
    if layout is None:  # _predict_output_layout logged why
        return None
    if iter_space is None:
        logger.debug(
            "dropping tiling %s on %s: iteration-space extents do not divide evenly",
            tiling,
            op.get_name(),
        )
        return None

    write_index = next(iter(op.get_read_writes().writes)).index
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
        iter_space=iter_space,
    )
