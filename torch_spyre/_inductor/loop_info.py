# Copyright 2025 The Torch-Spyre Authors.
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

"""Coarse-tiling loop metadata attached to ir.Operation objects.

``CoarseTileInfo`` is stamped onto ``ComputedBuffer`` ops by ``coarse_tile()``
and consumed by the scheduler, kernel codegen, and buffer-propagation pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from torch._inductor.ir import ComputedBuffer


@dataclass
class CoarseTileInfo:
    """Loop metadata stamped on a ``ComputedBuffer`` by the coarse-tiling pass.

    Attributes
    ----------
    loop_group_id:
        Tuple encoding the nesting path, e.g. ``(0,)`` for an outermost
        group, ``(0, 0)`` for a nested group inside group 0.
    loop_count:
        List of trip counts, one per nesting level from outermost to
        innermost.  ``len(loop_count) == len(loop_group_id)`` always holds.
    loop_tiled_dims:
        List of lists, one sub-list per nesting level.  Each sub-list
        contains the ``data.ranges`` positional indices that are tiled at
        that level.  An empty sub-list means the op is loop-invariant at
        that level.
    loop_tiled_reduction_dims:
        List of lists, one sub-list per nesting level.  Each sub-list
        contains the ``data.reduction_ranges`` positional indices that are
        tiled at that level.  An empty sub-list means no reduction dim is
        tiled at that level.  Parallel to ``loop_tiled_dims``.
    tile_advance_exprs:
        One sympy ``Expr`` per read dependency in
        ``op.get_read_writes().reads`` (in that iteration order, filtered to
        ``MemoryDep``), giving the element offset -- in that input tensor's
        *original, undivided* flat layout -- that one unit step of each
        tiled dim contributes, summed over every tiled dim.  Each term
        stays symbolic in Inductor's own iteration-space symbols (``d0,
        d1, ...``, continuous across output dims then reduction dims,
        matching ``Loops.get_reads()``'s own numbering): a tiled dim's
        term is ``coefficient * extent * d{i}``, left unevaluated rather
        than reduced to a plain number, so the expression is only resolved
        once a later compilation stage substitutes a concrete tile-index
        value for each ``d{i}``.  ``sympy.Integer(0)`` means that input
        does not advance (broadcast, or none of its dims are ever tiled).
        This is the sole tile-advance mechanism.
    output_tile_advance_expr:
        The analogous single ``Expr`` for this op's own output/write side,
        derived the same way from this op's pre-division write
        ``MemoryDep``.  Defaults to ``sympy.Integer(0)``.
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]] = field(default_factory=list)
    tile_advance_exprs: list[sympy.Expr] = field(default_factory=list)
    output_tile_advance_expr: sympy.Expr = field(
        default_factory=lambda: sympy.Integer(0)
    )


# ---------------------------------------------------------------------------
# Op-metadata helpers
# ---------------------------------------------------------------------------

_SPYRE_METADATA_ATTRS = (
    "dim_hints",
    "work_div_loop_info",
    "loop_info",
    "_restickify_plan",
    "_input_layout_overrides",
    "_emit_set_layout",
    # Deferred per_tile_fixed flag: set by coarse_tile._propagate_tiled_op when
    # the layout is FixedLayout (pre-stickify); consumed by finalize_layouts.
    "_pending_per_tile_fixed",
    # Links a tiled reduction op to its accumulation buffer; set by
    # coarse_tile._propagate_tiled_reduction_op, read by finalize_layouts in
    # insert_restickify.py to overwrite accum_full's generic layout.
    "_tiled_reduction_accum_name",
)


def copy_op_metadata(src: "ComputedBuffer", dst: "ComputedBuffer") -> None:
    """Copy all Spyre pass metadata from src to dst.

    Call this whenever a pass reconstructs a ComputedBuffer to ensure
    dim_hints, work-division hint metadata, and coarse-tiling attrs are not
    silently dropped.
    """
    for attr in _SPYRE_METADATA_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
