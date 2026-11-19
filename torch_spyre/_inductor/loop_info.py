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
from typing import TYPE_CHECKING, Any

import sympy

if TYPE_CHECKING:
    import torch.fx

    from torch._inductor.ir import ComputedBuffer


@dataclass(frozen=True)
class LoopCarryRecord:
    """Identity contract for one accumulator carried through a counted loop.

    ``for_each_tile`` lowering declares the carry with this record;
    ``loop_carries.apply_loop_carries`` then rewrites the update as an
    in-place write into the carry's initial storage.  Recording both ends
    lets scratchpad planning distinguish that closed, compiler-created
    mutation from an arbitrary user mutation, which must remain in HBM.

    ``loop_origin`` is the exact FX ``while_loop`` HOP node the carry belonged
    to, captured and validated at the splice site (see
    ``wsr/while_loop_bridge._validated_loop_origin``) before the WhileLoop IR
    node is deleted.  It is optional internal compiler metadata: ``None``
    whenever the origin could not be proven (nested loops, loops inside an
    ``invoke_subgraph``, or a chained carry whose origins gather an earlier
    loop's node), in which case the post-loop materialization plan declines
    and the carry keeps today's HBM behavior.  No other consumer of this
    record reads the field.

    ``storage_view`` is the view the update writes through when the carry's
    initial value is a view of the storage (a transposed init), else None.
    ``loop_carries.apply_loop_carries`` uses it as the mutation target, as the
    splice did when it made the write in place itself.
    """

    storage_name: str
    update_name: str
    loop_origin: "torch.fx.Node | None" = None
    storage_view: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ReadCopyElisionRecord:
    """Direct-read form saved while a consumer is redirected through a copy.

    Coarse tiling keeps the copy authoritative until layout selection, work
    division, and LX planning finish.  A later pass may restore
    ``direct_inner_fn`` only when the final physical plan proves that reading
    ``source_name`` directly is equivalent.  The record is deliberately not
    copied by :func:`copy_op_metadata`: rebuilding the consumer body
    invalidates the saved direct form unless that pass explicitly recreates
    the record.
    """

    consumer_name: str
    copy_name: str
    source_name: str
    direct_inner_fn: object
    direct_tiled_dims_per_level: (
        tuple[tuple[tuple[int, sympy.Expr], ...], ...] | None
    ) = None
    direct_squeezed_advance_per_level: (
        tuple[tuple[tuple[sympy.Expr, sympy.Expr], ...], ...] | None
    ) = None
    # Intermediate copies that become dead only if the deferred proposal is
    # accepted, ordered from the replacement consumer back toward the source.
    orphaned_copy_names: tuple[str, ...] = ()


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
        An empty per-level list means this read's address does not advance at
        that level. It does not by itself prove value invariance: a producer
        may rewrite fixed-address scratch on every trip. This is a tiling
        *decision*, not a substituted index expression -- deferred
        substitution into the dependency's actual
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
        rather than by substitution. An empty per-read list means that read
        has no such dims. The outer list may also be empty when no read in the
        operation needs squeezed-dimension metadata; that is complete "none
        needed" metadata, not a missing entry for every read.
    squeezed_advance_output:
        The analogous per-level ``(host_stride, extent)`` list for this op's
        own write dependency, parallel to ``output_tiled_dims`` the same way
        ``squeezed_advance_per_read`` is parallel to ``tiled_dims_per_read``.
        Defaults to ``[]`` (no levels tiled).
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


# ---------------------------------------------------------------------------
# Op-metadata helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopShim:
    """TEMPORARY, deleted with ``loop_info.py`` at stage 9.

    What the loop-tiling lowering (``wsr/loop_tiling_lowering.py``) stamps as an
    op's ``loop_info`` instead of a ``CoarseTileInfo``, for the readers that have
    not yet moved to the ``LoopNode``. It holds the plan's own content: which loop
    the op is in, how many trips, and which dim is tiled at each level. It holds
    NOTHING about advances: the four advance fields of ``CoarseTileInfo`` are
    absent on purpose, so a reader that wants them either has a shim branch
    (``loop_facts`` derives a shim op's advances from the op's own index) or fails
    in a test instead of reading an empty list.
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]]


# What a reader tests ``loop_info`` against when it needs the loop's own facts (group
# id, count, tiled dims) and nothing about advances. ``isinstance(x, CoarseTileInfo)``
# alone reads as False for a ``LoopShim``; stage 9 deletes this with the shim.
COUNTED_LOOP_INFO = (CoarseTileInfo, LoopShim)


_SPYRE_METADATA_ATTRS = (
    "dim_hints",
    "work_div_loop_info",
    "iteration_space_ownership",
    "loop_info",
    "_restickify_plan",
    "_input_layout_overrides",
    "_emit_set_layout",
    # Links a tiled reduction op to its accumulation buffer; set by
    # loop_tiling_lowering.lower_loop_tiling, read by finalize_layouts in
    # insert_restickify.py to promote accum_full to FixedTiledLayout when needed.
    "_tiled_reduction_accum_name",
    # One immutable record shared by a for_each_tile carry's persistent
    # storage and its in-loop update operation.
    "_loop_carry_record",
    # Ground-truth tiled-dim tag stamped by lower_tile_dim_marker
    # (lowering.py) on a for_each_tile tile read; consumed and erased by
    # _consume_tile_dim_markers (for_each_tile_lowering.py). Must survive
    # while_loop_bridge.splice_while_loop's own redirect_computed_buffer_
    # reads calls (e.g. "redirect while_loop carry/tile reads to persistent
    # scratch"), which reconstruct the marker op before
    # _consume_tile_dim_markers ever sees it -- confirmed empirically: the
    # marker attribute was silently dropped there before this attr was
    # added to this tuple, because copy_op_metadata only copies attrs
    # listed here.
    "tile_marker_dim",
    # work_division_constraints.JOINT_TILING_AND_DIVISION_ATTR, spelled out
    # since that module imports this one (through pass_utils).
    "_work_division_chosen_with_tiling",
)


# Per-op layout state set by propagate_spyre_tensor_layouts and
# nonstick_dim_order.  It describes ``src``'s own output and inputs, so it is
# only valid on a rebuild of ``src`` that keeps its name, layout and body
# (``copy_op_metadata(..., layout_state=True)``).  A new op built from another
# op (a drain, a restore, an inserted copy) must not inherit it.
_SPYRE_LAYOUT_STATE_ATTRS = (
    "layouts",
    "restick_cost_fn",
    "committed_stl",
)


def copy_op_metadata(
    src: "ComputedBuffer", dst: "ComputedBuffer", *, layout_state: bool = False
) -> None:
    """Copy non-provenance Spyre pass metadata from src to dst.

    Call this whenever a pass reconstructs a ComputedBuffer to ensure
    dim_hints, work-division hint metadata, and coarse-tiling attrs are not
    silently dropped. Source provenance is owned by the helpers in
    ``provenance.py`` and is deliberately excluded from this bulk copy.

    Pass ``layout_state=True`` only when ``dst`` replaces ``src`` with the same
    name, layout and body semantics (a name-swap or body-wrap rebuild): it also
    copies the layout-propagation results in ``_SPYRE_LAYOUT_STATE_ATTRS``.
    Without it, a rebuild after ``propagate_spyre_tensor_layouts`` loses them
    and later passes fail (``KeyError`` in optimize_restickify,
    ``FixedLayout`` without ``device_layout`` in insert_restickify).
    """
    attrs: tuple[str, ...] = _SPYRE_METADATA_ATTRS
    if layout_state:
        attrs = attrs + _SPYRE_LAYOUT_STATE_ATTRS
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
