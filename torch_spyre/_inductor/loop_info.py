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
from typing import TYPE_CHECKING, Literal

import sympy

if TYPE_CHECKING:
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.ir import ComputedBuffer


@dataclass(frozen=True)
class ReductionPlan:
    """Planned shape/identity/nesting data for a tiled-reduction op.

    Computed once during planning (``_plan_tiling_propagation``) from pure
    functions of already-known shape data, and consumed by transformation's
    reduction-machinery pass to build the actual accumulator/fill/combine
    buffers -- the objects themselves don't exist until then, only the
    decisions about their shape/identity/nesting front-load.

    Attributes
    ----------
    reduction_type:
        ``op.data.reduction_type`` (e.g. ``"sum"``, ``"max"``).
    identity:
        Monoid identity value for ``reduction_type``, from
        ``_reduction_identity_value``.
    is_nested:
        True when an outer level tiles an output dim and an inner level
        tiles a reduction dim, requiring separate tile-sized and full-sized
        accumulators (see ``_compute_fill_loop_info_planned``). False for a flat
        (reduction-dim-only) tiling.
    full_output_ranges:
        Full (pre-division) output shape for the accumulation buffer --
        planning runs before ``_apply_plan`` divides ``op.data.ranges``, so
        this is just ``op.data.ranges`` at planning time, unchanged.
    per_tile_ranges:
        Per-outer-tile output shape: ``op.data.ranges`` at planning time with
        every tiled dim divided by its ``loop_count`` (mirrors the division
        ``_divide_ranges`` performs later, in place, during transformation).
    outer_fill_loop_info:
        ``CoarseTileInfo`` covering only the outer output-dim levels, to
        stamp on the fill op for a nested tiling
        (``_compute_fill_loop_info_planned``).
        ``None`` for a flat tiling, where the fill runs once before all loops.
        Its ``loop_group_id`` is planning-time, pre-offset numbering --
        transformation must re-slice it from the op's own real, stamped
        ``loop_group_id`` before use (see ``_propagate_tiled_reduction_op``).
    full_output_strides:
        Host strides of the full-sized accumulation buffer, captured from
        ``op.layout.stride`` at planning time (before ``_divide_ranges`` runs).
    per_tile_strides:
        Host strides of the per-outer-tile accumulation buffer, derived from
        ``full_output_strides`` via ``compute_tile_stride`` at planning time.
    """

    reduction_type: str
    identity: float | int
    is_nested: bool
    full_output_ranges: list[sympy.Expr]
    per_tile_ranges: list[sympy.Expr]
    outer_fill_loop_info: "CoarseTileInfo | None"
    full_output_strides: tuple[sympy.Expr, ...]
    per_tile_strides: tuple[sympy.Expr, ...]


@dataclass(frozen=True)
class PropagationPlan:
    """Decision for how a tiled op's result crosses its loop boundary.

    Computed once during planning (``_plan_tiling_propagation``) and
    consumed by transformation's fixed pass sequence, which only acts on
    these decisions -- it never makes new ones.

    Attributes
    ----------
    kind:
        ``"loop_internal"``: the op's own buffer is scratch reused every
        iteration; its write must not advance at any level.
        ``"copy_out"``: the op's result is consumed outside its loop group
        (or is a graph output) and needs a full-sized buffer + copy op.
        ``"reduction"``: the op is a Reduction tiled over a reduction dim;
        see ``reduction`` for the accumulator/fill/combine shape decisions.
    full_ranges:
        Full (pre-division) iteration ranges for the copy-out's full buffer.
        Only set when ``kind == "copy_out"``.
    full_strides:
        Original (pre-division) strides of the tiled op's layout, captured at
        planning time before ``_divide_ranges`` mutates the op.  Only set when
        ``kind == "copy_out"``.
    reduction:
        Shape/identity/nesting decisions for the reduction machinery. Only
        set when ``kind == "reduction"``.
    outside_consumer_names:
        Names (not object references -- see ``_find_outside_consumers``, and
        the module docstring on name stability) of ComputedBuffers outside
        this op's own outermost loop group that read this op's result.
    is_graph_output:
        True if this op's buffer name appears in the graph's output names.
    """

    kind: Literal["loop_internal", "copy_out", "reduction"]
    full_ranges: list[sympy.Expr] | None = None
    full_strides: tuple[sympy.Expr, ...] | None = None
    reduction: ReductionPlan | None = None
    outside_consumer_names: tuple[str, ...] = ()
    is_graph_output: bool = False


@dataclass(frozen=True)
class ReadCopyEntry:
    """One shared copy to insert for a cross-group buffer read.

    Attributes
    ----------
    copy_name:
        Qualified name to assign the inserted copy ComputedBuffer.
    dep:
        The canonical MemoryDep (buffer name + index + var_names + size)
        every equivalent read in the group shares -- the same object one of
        the consuming ops' own full_deps produced, used to size/index the
        copy exactly as _insert_all_read_copy_ops does today.
    insert_before_op_name:
        get_operation_name() of the first (operations order) consuming op
        in the group -- where the copy is inserted.
    sizing_op_name:
        get_operation_name() of the op supplying tiled_op.loop_info for
        the copy's own read/write-level-extent computation (the first
        consuming op, per the sizing-invariant in the design doc).
    consumer_op_names:
        Names of every op in the group that must have this dep's buffer
        name patched (via _NameSwapHandler) to load from copy_name instead.
    """

    copy_name: str
    dep: "MemoryDep"
    insert_before_op_name: str
    sizing_op_name: str
    consumer_op_names: tuple[str, ...]


@dataclass(frozen=True)
class ReadCopyPlan:
    """Per-group plan for Pass 1 (read-copy insertion).

    Computed once during planning (_plan_read_copies, run after every
    group's _apply_plan) and consumed by a slimmed _insert_all_read_copy_ops
    that only executes these decisions.

    Attributes
    ----------
    entries:
        One ReadCopyEntry per distinct (buffer name, canonical index expr)
        pair read cross-group by at least one op in this group.
    """

    entries: tuple["ReadCopyEntry", ...]


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
    tiled_dims_per_read:
        One entry per read dependency in ``op.get_read_writes().reads`` (in
        that iteration order, filtered to ``MemoryDep`` -- same positional
        convention the deleted ``tile_advance_exprs`` used). Each entry is a
        list of per-nesting-level ``(op_dim_index, extent)`` pairs,
        outermost level first (matching ``loop_count``/``loop_tiled_dims``):
        the host-range positional dims (or ``n_output_dims + reduction_pos``
        for reduction dims, matching ``loop_tiled_dims``'s own numbering)
        tiled *for this dependency* at that level, paired with that level's
        own tile extent: for a dim tiled at more than one level, an outer
        level's extent equals the final (innermost) extent times the
        product of every more-inner level's own count that also tiles that
        dim.
        An empty per-level list means the dep is loop-invariant at that
        level. This is a tiling *decision*, not a substituted index
        expression -- deferred substitution into the dependency's actual
        (possibly later-rewritten) index expression happens in
        spyre_kernel.py at OpSpec/TensorArg construction time, when the
        index is guaranteed final.
    output_tiled_dims:
        The analogous per-level ``(op_dim_index, extent)`` list for this
        op's own write dependency. Defaults to ``[]`` (no levels tiled).
    squeezed_advance_per_read:
        One entry per read dependency, parallel to ``tiled_dims_per_read``.
        Each entry is a list of per-nesting-level ``(host_stride, extent)``
        pairs for dims tiled down to extent 1 in this dep's own iteration
        space -- Inductor's ``SqueezeView.squeezer`` unconditionally drops
        any such dim from ``dep.index`` (called via
        ``ComputedBuffer.get_read_writes()`` -> ``extract_read_writes`` ->
        ``index_vars_squeeze``), so no ``d{i}`` symbol survives for
        ``_host_dim_to_index_symbol`` to substitute into -- unlike a dim
        merely absent from one read among several (genuine broadcast),
        which ``tiled_dims_per_read`` already handles correctly via
        substitution. ``host_stride`` is the dim's canonical index
        coefficient in the *squeezed* iteration space of the op whose
        ``data.ranges`` sizes this dim (product of that op's own
        ``data.ranges`` sizes strictly to the dim's right) -- the same units
        every surviving ``d{i}`` symbol in ``dep.index`` already carries,
        since Inductor mints those coefficients over the unsqueezed
        ``data.ranges`` and squeeze only renumbers/drops symbols, never
        rescales them. This is deliberately NOT the buffer's real PyTorch
        memory stride (``full_buf.layout.stride``) -- that is a different
        unit system that ``tiling_expr_to_device_expr``'s ``stride_map``-
        based dimension selection cannot be compared against, and picking
        the wrong device axis when the two happen to diverge silently
        advances the wrong dimension (see issue surfaced by
        ``test_flash_tile_B``). Independent of ``dep.index`` entirely, so
        ``SpyreKernel._general_tile_advance`` can add its device-address
        contribution as an extra term via ``tiling_expr_to_device_expr``
        rather than by substitution. Empty list means no such dims for this
        read (the common case).
    squeezed_advance_output:
        The analogous per-level ``(host_stride, extent)`` list for this op's
        own write dependency, parallel to ``output_tiled_dims`` the same way
        ``squeezed_advance_per_read`` is parallel to ``tiled_dims_per_read``.
        Defaults to ``[]`` (no levels tiled).
    propagation:
        Planned decision for how this op's result crosses its loop
        boundary, computed by ``_plan_tiling_propagation``. ``None`` until
        that planning stage runs (or for ops it doesn't cover).
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]] = field(default_factory=list)
    tiled_dims_per_read: list[list[list[tuple[int, sympy.Expr]]]] = field(
        default_factory=list
    )
    output_tiled_dims: list[list[tuple[int, sympy.Expr]]] = field(default_factory=list)
    squeezed_advance_per_read: list[list[list[tuple[sympy.Expr, sympy.Expr]]]] = field(
        default_factory=list
    )
    squeezed_advance_output: list[list[tuple[sympy.Expr, sympy.Expr]]] = field(
        default_factory=list
    )
    propagation: "PropagationPlan | None" = None


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
    # Links a tiled reduction op to its accumulation buffer; set by
    # coarse_tile._propagate_tiled_reduction_op, read by finalize_layouts in
    # insert_restickify.py to overwrite accum_full's generic layout.
    "_tiled_reduction_accum_name",
)


def copy_op_metadata(src: "ComputedBuffer", dst: "ComputedBuffer") -> None:
    """Copy non-provenance Spyre pass metadata from src to dst.

    Call this whenever a pass reconstructs a ComputedBuffer to ensure
    dim_hints, work-division hint metadata, and coarse-tiling attrs are not
    silently dropped. Source provenance is owned by the helpers in
    ``provenance.py`` and is deliberately excluded from this bulk copy.
    """
    for attr in _SPYRE_METADATA_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
