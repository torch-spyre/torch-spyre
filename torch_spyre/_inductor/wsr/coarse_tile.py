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

"""Helpers of the loop-tiling lowering.

The lowering itself is ``wsr/loop_tiling_lowering.py`` (entry point
``tile_groups(graph, groups)``): it divides each tiled op's ranges by the loop's
trip count and gives every member a ``LoopMember``. This module keeps what that
lowering shares with the ``for_each_tile`` lowering:

  - the range and layout division (``_divide_ranges``, ``_divide_reduction_ranges``),
  - consumer patching (``_patch_consumers``) and full-buffer allocation,
  - ``validate_coarse_tile_groups`` / ``reject_nested_loops``.

``groups`` is a list of ``(ops, levels)`` tuples where ``levels`` is a list of
``(hint_id, count)`` pairs, outermost first (one level only: nesting is refused).
Each op resolves its own tiled dimension from its ``loop_var`` in ``dim_hints``.
Each ``ops`` list must be a contiguous sub-sequence of ``operations``.

Before touching any ``inner_fn``/``layout``/``MutationLayoutSHOULDREMOVE``
rewiring in this file, read "Invariants and failure modes" in
``docs/source/compiler/loop_nodes.md``. It documents the wrap-never-reconstruct
convention and why ``MutationLayoutSHOULDREMOVE`` sites must satisfy the
single-mutation-target invariant -- the same ground this file's rewrite sites
depend on.
"""

from __future__ import annotations


import contextlib

import sympy
from sympy import Expr

import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    Layout,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    ReinterpretView,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import SpyreTensorLayout

from ..constants import BATCH_MATMUL_OP, MATMUL_REDUCTION_OPS
from ..errors import Unsupported
from ..loop_facts import counted_loop_path
from ..loop_node import move_before
from ..pass_utils import (
    op_out_coords,
    identify_matmul_inputs,
    invalidate_op_read_writes,
    iteration_space_from_op,
    op_read_writes,
    replace_computed_buffer_body,
)
from ..ir import FixedTiledLayout
from ..ir import _resize_device_layout  # noqa: F401  (re-exported; tests import it here)
from .loop_tiling_common import (  # noqa: F401
    _ReadThroughLayout,
    _COMPUTED_BUF_FREE_SYMS_KEY,
    _COMPUTED_BUF_SIZES_KEY,
    _DivideRangesResult,
    _IterationSymbolRemap,
    _LAYOUT_FREE_SYMS_KEY,
    _LOOPS_FREE_SYMS_KEY,
    _LOOPS_INNER_FN_OPCOUNT_KEY,
    _LOOPS_INNER_FN_STR_KEY,
    _LogicalIterationSymbol,
    _NameSwapHandler,
    _REDUCTION_FREE_SYMS_KEY,
    _RetileLoadIndexHandler,
    _RetiledBufferInfo,
    _cache_key,
    _capture_logical_iteration_symbols,
    _clear_cache,
    _consumer_own_dim_symbol,
    _divide_ranges,
    _fused_iteration_symbol_remap,
    _hint_ranges_pos,
    _index_already_at_new_scale,
    _index_var_prefix,
    _is_dense_full_buffer_view,
    _loop_var_hinted_ranges,
    _loop_var_hinted_reduction_ranges,
    _loop_var_to_ranges_pos,
    _loop_var_to_reduction_ranges_pos,
    _op_reads,
    _order_preserving_symbol_remap,
    _patch_retiled_load_indexes,
    _raw_to_squeezed_pos,
    _replace_group_op,
    _rescale_index,
    _retile_load_index,
    _should_patch_retiled_load_indexes,
    _squeezed_retile_dims,
    _stick_host_dim,
    _undivided_ranges,
    logger,
    reduction_loop_vars,
)


# ---------------------------------------------------------------------------
# Group validation
# ---------------------------------------------------------------------------


def validate_coarse_tile_groups(groups: list[tuple]) -> None:
    """Raise RuntimeError if any hint_id appears in more than one group.

    Each spyre_hint scope has a unique hint_id.  All ops sharing a hint scope
    must be contiguous in the operation list and therefore land in a single group.
    A hint_id appearing in two groups means ops from the same hint scope were
    split — e.g. because an unrelated op migrated into the middle of the run —
    producing two separate loop nests over the same hint scope that would iterate
    different tiles in an unsynchronized fashion.
    """
    hint_id_to_group: dict[int, int] = {}
    for group_idx, (group_ops, _levels) in enumerate(groups):
        group_hint_ids: set[int] = set()
        for op in group_ops:
            for h in getattr(op, "dim_hints", []):
                group_hint_ids.add(h.hint_id)
        for hint_id in group_hint_ids:
            prior = hint_id_to_group.get(hint_id)
            if prior is not None:
                raise RuntimeError(
                    f"coarse_tile: hint_id={hint_id} appears in both group {prior} "
                    f"and group {group_idx}. Ops from the same hint scope were split "
                    "across two separate loop nests, which would produce unsynchronized "
                    "tiling."
                )
            hint_id_to_group[hint_id] = group_idx


def reject_nested_loops(groups: list[tuple], source: str) -> None:
    """Raise ``Unsupported`` for a group that nests more than one loop level.

    Span-overflow tiling and the allocator's own tiling build their groups for
    ``tile_groups``, which lowers one loop level per group (one standalone
    ``LoopNode``). A nested group is refused (#6088); lifting that means building
    nested nodes, one induction variable per level.

    Called by the two production callers and by ``build_loop_tiling_plan``; a
    ``for_each_tile`` nest never reaches either caller (``span_overflow_groups``
    skips an op that carries ``dim_hints``). ``source`` names the caller in the
    message.
    """
    for group_ops, levels in groups:
        if len(levels) > 1:
            names = ", ".join(op.get_name() for op in group_ops)
            raise Unsupported(
                f"{source}: nested loop tiling is not supported in this release "
                f"({len(levels)} levels over {names}; #6088)"
            )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def resolve_level_positions(
    op: ComputedBuffer, levels: list[tuple]
) -> tuple[list[list[int]], list[list[int]], dict[int, sympy.Symbol]]:
    """Where each of a group's loop levels tiles ``op``.

    Returns ``(tiled_dims, tiled_reduction_dims, unattributed_loop_vars)``:
    per level, the op's output-range position tiled there (a list of zero or one)
    and its reduction-range position; and the splice loop variables this op has no
    dim to attribute (the lowering reads their advance off the index).
    Shared by ``wsr.loop_tiling_lowering.build_loop_tiling_plan`` and the
    ``for_each_tile`` lowering, so the two agree on positions by construction.

    ``_hint_ranges_pos`` returns (position, is-a-reduction-dim) and is
    authoritative on which channel the dim lands in: see its docstring for why a
    WhileLoop-splice hint's own ``is_reduction`` cannot decide that per op.
    """
    op_out = op_out_coords(op)
    hint_id_to_ranges_pos: dict[int, int] = {}
    hint_id_to_reduction_ranges_pos: dict[int, int] = {}
    unattributed_loop_vars: dict[int, sympy.Symbol] = {}
    for h in getattr(op, "dim_hints", []):
        if h.loop_var is None:
            continue
        pos, resolved_is_reduction = _hint_ranges_pos(op, h, op_out)
        if pos is None:
            if h.loop_var_range is not None:
                unattributed_loop_vars[h.hint_id] = h.loop_var
            continue
        if resolved_is_reduction:
            hint_id_to_reduction_ranges_pos[h.hint_id] = pos
        else:
            hint_id_to_ranges_pos[h.hint_id] = pos
    tiled_dims: list[list[int]] = []
    tiled_reduction_dims: list[list[int]] = []
    for hint_id, _count in levels:
        opos = hint_id_to_ranges_pos.get(hint_id)
        rpos = hint_id_to_reduction_ranges_pos.get(hint_id)
        tiled_dims.append([opos] if opos is not None else [])
        tiled_reduction_dims.append([rpos] if rpos is not None else [])
    return tiled_dims, tiled_reduction_dims, unattributed_loop_vars


def _find_outside_consumers_planned(
    buf_name: str,
    group_loop_id: tuple[int, ...],
    operations: list[Operation],
    name_to_group_outer_key: dict[str, int],
) -> tuple[list[str], bool]:
    """Planning-time analog of _find_outside_consumers.

    Same decision (does any op outside buf_name's own outermost loop group
    read it, or is it a graph output), but returns consumer *names* instead
    of objects (planning is zero-mutation, so there's no reason to carry
    object references past this stage), and looks up each candidate's outer loop-group key from
    name_to_group_outer_key (built once by the caller from the planned
    LoopMember dict) instead of a not-yet-stamped op.loop_member attribute.
    """
    outer_key = group_loop_id[0]
    consumer_names: list[str] = []
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not _reads_buffer_cached(op, buf_name):
            continue
        candidate_outer_key = name_to_group_outer_key.get(op.get_name())
        if candidate_outer_key is None or candidate_outer_key != outer_key:
            consumer_names.append(op.get_name())

    is_graph_output = buf_name in _graph_output_names()
    return consumer_names, is_graph_output


def _planned_full_device_layout(
    op: ComputedBuffer, full_ranges: list[Expr]
) -> SpyreTensorLayout | None:
    """The device layout a copy-out's full buffer or a reduction's full
    accumulator takes: ``op``'s own, read before ``_divide_ranges`` replaces it
    with the tile's.

    ``None`` when ``op`` has no device layout yet (pre-stickify), or when the
    full buffer is not ``op``'s own shape -- a WhileLoop-splice dim, which
    the full-range computation multiplies up. ``_allocate_full_buffer``
    then raises for a post-stickify op.
    """
    layout = op.layout
    if not isinstance(layout, FixedTiledLayout):
        return None
    if list(layout.size) != list(full_ranges):
        return None
    return layout.device_layout


def _planned_tile_extents_per_level(
    op: ComputedBuffer,
    op_tiled_dims: list[list[int]],
    op_tiled_reduction_dims: list[list[int]],
    levels: list[tuple],
) -> list[dict[int, Expr]]:
    """Per-level (not merged) tile extents, outermost-first.

    Mirrors the arithmetic _divide_ranges/_divide_reduction_ranges perform
    in place, but reads pre-mutation op.data.ranges/reduction_ranges and
    never calls object.__setattr__ on op.data or touches op.layout. Raises
    Unsupported on non-even division, matching _divide_ranges's own check
    so the error surfaces at the same point in the pipeline it does today.

    Unlike the deleted _planned_tile_extents, a dim tiled at more than one
    level gets a DISTINCT extent value per level here: level i's extent is
    final_extent * (product of counts at every level strictly more-inner
    than i that also tiles this same dim).

    A dim carrying a WhileLoop-splice loop_var hint (see
    _loop_var_hinted_ranges/_loop_var_hinted_reduction_ranges) is exempt
    from the divide-by-count model entirely: op.data.ranges[d] (or
    reduction_ranges[d]) is ALREADY the per-iteration extent, and the
    hint's loop_var_range (the loop's trip count) is a multiplier over a
    larger extent that is never materialized as a real dim anywhere --
    never a divisor. For such a dim, final_extent is op.data.ranges[d]
    unchanged, and _per_level_extent_for uses loop_var_range (not the
    level's own count) when it extrapolates outward past that dim's own
    level.
    """
    hinted_ranges = _loop_var_hinted_ranges(op)
    hinted_reduction_ranges = _loop_var_hinted_reduction_ranges(op)

    counts_by_dim: dict[int, Expr] = {}
    counts_by_reduction_dim: dict[int, Expr] = {}
    for level_idx, (_, count) in enumerate(levels):
        for d in op_tiled_dims[level_idx]:
            if d in hinted_ranges:
                continue
            counts_by_dim[d] = counts_by_dim.get(d, sympy.Integer(1)) * count
        for d in op_tiled_reduction_dims[level_idx]:
            if d in hinted_reduction_ranges:
                continue
            counts_by_reduction_dim[d] = (
                counts_by_reduction_dim.get(d, sympy.Integer(1)) * count
            )

    def _divided(r: Expr, count: Expr, dim_desc: str) -> Expr:
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            count, (int, sympy.Integer)
        ):
            if int(r) % int(count) != 0:
                raise Unsupported(
                    f"coarse_tile: op {op.get_name()!r} {dim_desc} range {r} "
                    f"is not divisible by loop_count {count}.  All tiled "
                    f"dimensions must be evenly divisible by the loop trip count."
                )
            return sympy.Integer(int(r) // int(count))
        return sympy.sympify(r) / sympy.sympify(count)

    final_dim_extents = {
        d: _divided(op.data.ranges[d], count, f"loop var d{d}")
        for d, count in counts_by_dim.items()
    }
    for d in hinted_ranges:
        final_dim_extents[d] = op.data.ranges[d]
    final_reduction_extents = {}
    if isinstance(op.data, Reduction):
        final_reduction_extents = {
            d: _divided(op.data.reduction_ranges[d], count, f"reduction dim {d}")
            for d, count in counts_by_reduction_dim.items()
        }
        for d in hinted_reduction_ranges:
            final_reduction_extents[d] = op.data.reduction_ranges[d]

    n_output_dims = len(op.data.ranges) if hasattr(op.data, "ranges") else 0

    def _per_level_extent_for(
        final_extent: Expr,
        tiled_at_level: list[list[int]],
        dim_id: int,
        loop_var_range: Expr | None,
    ) -> dict[int, Expr]:
        # tiled_at_level[level_idx] is the list of dims tiled at that level
        # (op_tiled_dims or op_tiled_reduction_dims); find every level index
        # tiling dim_id, outermost first (levels is already outermost-first).
        levels_tiling_dim = [
            level_idx for level_idx, dims in enumerate(tiled_at_level) if dim_id in dims
        ]
        result: dict[int, Expr] = {}
        # Walk innermost-to-outermost; each step outward multiplies by the
        # next-inner level's own count, so an outer level's extent equals
        # the final extent times every more-inner level's count. For a
        # loop_var-hinted dim, use its loop_var_range (the trip count that
        # is a multiplier over a never-materialized larger extent) instead
        # of the level's own count -- there is exactly one such level per
        # WhileLoop-splice hint_id, so this only ever fires once, but stays
        # general in case a future caller nests one under another.
        running_extent = final_extent
        for level_idx in reversed(levels_tiling_dim):
            result[level_idx] = running_extent
            step = levels[level_idx][1] if loop_var_range is None else loop_var_range
            running_extent = running_extent * step
        return result

    per_level_output: list[dict[int, Expr]] = [dict() for _ in levels]
    for d, final_extent in final_dim_extents.items():
        level_extents = _per_level_extent_for(
            final_extent, op_tiled_dims, d, hinted_ranges.get(d)
        )
        for level_idx, extent in level_extents.items():
            per_level_output[level_idx][d] = extent
    for d, final_extent in final_reduction_extents.items():
        dim_key = n_output_dims + d
        level_extents = _per_level_extent_for(
            final_extent,
            op_tiled_reduction_dims,
            d,
            hinted_reduction_ranges.get(d),
        )
        for level_idx, extent in level_extents.items():
            per_level_output[level_idx][dim_key] = extent

    return per_level_output


def _check_matmul_broadcast_batch_tiling(
    op: ComputedBuffer,
    read_deps: list[MemoryDep],
    write_deps: list[MemoryDep],
    per_level_extents: list[dict[int, Expr]],
) -> None:
    """Reject coarse-tiling a matmul's broadcast batch dim with >1 elem/tile.

    torch-spyre#3888: when a matmul operand (x) is broadcast over a batch
    dim (e.g. torch.matmul(x.unsqueeze(0), w) with no real batch dim on x)
    and that dim is coarse-tiled with more than one element per tile, the
    backend's SDSC batched-matmul scheduling primitive cannot express the
    result: the native device compiler aborts with
    ``sbf-ddc: DtException: inp0_reuse_dim.size() == 1`` in
    ``L3DlOpsScheduler.cpp``. Confirmed via the generated ``sdsc_*.json``:
    tile size 1 (one batch element per tile) never materializes a reuse-dim
    key on the matmul's ``N_`` dims at all (128 separate kernel invocations);
    tile size 2 stamps ``N_.kj_ = 2`` and the native scheduler rejects it.
    This is a genuine backend limitation, not a torch-spyre layout bug --
    see issue #3927 for the writeup shared with the deeptools backend team.

    Rather than emit a kernel the native compiler will reject, raise
    Unsupported here at plan time so the failure is immediate and points at
    the actual cause instead of an opaque subprocess crash deep in codegen.
    """
    if not (
        isinstance(op.data, Reduction)
        and op.data.reduction_type in MATMUL_REDUCTION_OPS
    ):
        return
    if len(read_deps) != 2 or not write_deps:
        return

    out_dep = write_deps[0]
    x_dep, y_dep = identify_matmul_inputs(read_deps, out_dep)
    if x_dep is None or y_dep is None:
        return

    # Candidates for "absent from x, present in y and the output": the true
    # generated (N) dim of the matmul is always exactly one such var. A
    # *second* one is only possible when x is broadcast over an extra batch
    # dim it carries no symbol for at all (see issue #3888/#3927) -- that
    # excess var, not the legitimate N var, is what this check must flag.
    candidates = (
        y_dep.index.free_symbols & out_dep.index.free_symbols
    ) - x_dep.index.free_symbols
    if len(candidates) <= 1:
        return  # unambiguous N var (or none) -- nothing to flag.

    # Ambiguous: more than one var is absent from x but present in y and the
    # output. Exactly one is the true generated (N) dim; the rest are excess
    # broadcast-batch vars x carries no symbol for at all. broadcast_batch_vars
    # (issue #3888) resolves this via op.loop_member.loop_tiled_dims, but that
    # isn't stamped yet at plan time -- so here we can only tell "ambiguous"
    # from "unambiguous", not which candidate is the real N dim. Treat every
    # candidate whose *tiled* extent exceeds 1 element/tile as excess: the
    # real N dim's own generation loop is not a coarse-tile dim (it's the
    # matmul's inherent output dim, not something build_loop_tiling_plan
    # tiles down), so it will not appear in per_level_extents at all.
    #
    # Candidate symbols are always Inductor's dense "d{N}" iteration vars
    # (same numbering _raw_to_squeezed_pos/_host_dim_to_index_symbol assign
    # squeezed dims), so the str(sym).startswith("d") parse below recovers
    # the squeezed index. A future Inductor change to variable naming would
    # make this silently skip the candidate rather than raise -- if that
    # happens, the excess dim falls through to the native compiler's opaque
    # inp0_reuse_dim.size() == 1 abort instead of this clean Unsupported.
    excess_dims = {
        int(str(sym)[1:])
        for sym in candidates
        if str(sym).startswith("d") and str(sym)[1:].isdigit()
    }
    raw_to_squeezed = _raw_to_squeezed_pos(op)

    for level in per_level_extents:
        for raw_dim, extent in level.items():
            # .get(raw_dim) (no fallback): a raw_dim absent from
            # raw_to_squeezed was squeezed out (unit-size, extent == 1) and
            # has no d{i} symbol at all, so it can never legitimately match
            # an excess_dims entry (those are squeezed indices). Falling
            # back to raw_dim itself would risk an accidental collision with
            # an unrelated squeezed index; None never collides.
            if raw_to_squeezed.get(raw_dim) not in excess_dims:
                continue
            if isinstance(extent, (int, sympy.Integer)) and int(extent) <= 1:
                continue  # one broadcast element per tile -- backend handles this.
            raise Unsupported(
                f"matmul {op.get_name()!r}: coarse-tiling broadcast batch "
                f"dim d{raw_dim} with {extent} elements/tile is not "
                "supported -- the backend's batched-matmul scheduling "
                "primitive requires exactly 1 broadcast element per kernel "
                "invocation (inp0_reuse_dim.size() == 1). Retile this "
                "dimension with 1 element per tile (tile_size=1), or see "
                "issue #3927."
            )


def _group_reduction_tiled_levels_in_group(
    group_ops: list[Operation],
    levels: list[tuple],
) -> set[int]:
    """Planning-time helper for cross-op reduction-tiling checks.

    Level indices (positions into per-op loop_tiled_dims/
    loop_tiled_reduction_dims) where some Reduction op in group_ops tiles a
    reduction dim, computed directly from group_ops -- planning already has
    the group's ops together (unlike the post-stamp version, which only has
    the flat operations list and must filter by loop_group_id[0]).

    A Reduction op's own reduction-tiled-dims list is only ever non-empty at
    a level for that op (Pointwise ops never populate reduction dims -- see
    build_loop_tiling_plan's hint_id_to_reduction_ranges_pos, gated on
    isinstance(op.data, Reduction)), so this scan only needs to inspect
    Reduction ops; a Pointwise-only group always yields an empty set.

    A level whose hint is a WhileLoop-splice hint (``loop_var_range is not
    None``, see propagate_hints.py's DimHint docstring) is excluded even
    when some Reduction op tiles a reduction dim there. Both of this
    function's callers exist to catch a same-outer-group Pointwise sibling
    that would see a still-partial sum -- a real hazard for an ordinary
    spyre_hint() reduction group, where nothing else re-runs the Pointwise
    op once the reduction's own inner loop finishes accumulating. A
    WhileLoop-splice level has no such hazard: the group is one iteration
    of the spliced body, and a Pointwise op there (e.g. an accumulator's
    ``acc + p @ v_tile``) is SUPPOSED to fold each iteration's per-tile
    partial into the carry -- that folding, plus the carry-back across
    iterations, is exactly how the reduction completes over the whole
    loop. Flagging it as premature (or deferring it to copy_out) would
    treat the loop's own carry semantics as a bug.
    """
    reduction_levels: set[int] = set()
    for o in group_ops:
        if not isinstance(o, ComputedBuffer) or not isinstance(o.data, Reduction):
            continue
        o_out = op_out_coords(o)
        hint_id_to_reduction_ranges_pos: dict[int, int] = {}
        hint_id_to_loop_var_range: dict[int, object] = {}
        for h in getattr(o, "dim_hints", []):
            if h.loop_var is None:
                continue
            pos, resolved_is_reduction = _hint_ranges_pos(o, h, o_out)
            if pos is None or not resolved_is_reduction:
                continue
            hint_id_to_reduction_ranges_pos[h.hint_id] = pos
            hint_id_to_loop_var_range[h.hint_id] = h.loop_var_range
        for level_idx, (hint_id, _count) in enumerate(levels):
            if hint_id in hint_id_to_reduction_ranges_pos and (
                hint_id_to_loop_var_range.get(hint_id) is None
            ):
                reduction_levels.add(level_idx)
    return reduction_levels


def _reads_incomplete_reduction(
    op: ComputedBuffer,
    group_ops: list,
    group_op_names: set[str],
    plan: dict,
    group_reduction_tiled_levels: set[int],
) -> bool:
    """True if op reads a group-sibling whose result is still partial at any
    reduction-tiled level — i.e. the reduction hasn't accumulated yet when op runs."""
    name_to_buf = {o.get_name(): o for o in group_ops if isinstance(o, ComputedBuffer)}
    for n in _op_reads(op):
        if n not in group_op_names:
            continue
        buf = name_to_buf.get(n)
        if not isinstance(buf, ComputedBuffer):
            continue
        # group_ops is topologically ordered, so any in-group sibling is
        # already in plan by the time we reach op. A missing entry means
        # buf is outside this group (cross-group read) — not a partial result.
        entry = plan.get(id(buf))
        if entry is None:
            continue
        if any(
            entry.loop_tiled_reduction_dims[i] for i in group_reduction_tiled_levels
        ):
            return True
    return False


def _plan_is_loop_invariant_at_reduction_levels(
    op: ComputedBuffer,
    op_tiled_dims: list[list[int]],
    group_reduction_tiled_levels: set[int],
) -> bool:
    """True if op is loop-invariant at every level where some Reduction op
    in the same group tiles a reduction dim -- planning-time check using the
    group's own ops (already available during planning) instead of a
    flat-list post-stamp scan."""
    if not isinstance(op.data, Pointwise):
        return False
    if not group_reduction_tiled_levels:
        return False
    return all(not op_tiled_dims[i] for i in group_reduction_tiled_levels)


# ---------------------------------------------------------------------------
# Shared leaf helpers
# ---------------------------------------------------------------------------


def _reduction_identity_value(
    reduction_type: str, dtype: "torch.dtype"
) -> "float | int":
    """Return the monoid identity value for the given reduction type.

    Used to initialize the accumulation buffer before a tiled reduction loop.
    """
    if reduction_type in ("sum", "xor_sum", "any", BATCH_MATMUL_OP):
        return 0
    if reduction_type == "prod":
        return 1
    if reduction_type == "max":
        return float("-inf")
    if reduction_type == "min":
        return float("inf")
    raise RuntimeError(
        f"coarse_tile: unsupported reduction_type {reduction_type!r} for tiled "
        "reduction — no identity value is defined for this reduction type."
    )


def _validate_contiguous(
    ops: list[Operation],
    op_to_position: dict[str, int],
    group_id: tuple[int, ...],
) -> None:
    """Assert that ops form a contiguous slice of the operation list.

    A gap indicates a data-flow dependency that crosses the group boundary,
    which would violate the coarse-tiling model.
    """
    positions = []
    for op in ops:
        name = op.get_operation_name()
        if name not in op_to_position:
            raise RuntimeError(
                f"coarse_tile: operation {name!r} (group {group_id}) "
                "is not in the operations list"
            )
        positions.append(op_to_position[name])

    if not positions:
        return

    lo, hi = min(positions), max(positions)
    if hi - lo + 1 != len(ops):
        raise RuntimeError(
            f"coarse_tile: group {group_id} operations are not contiguous "
            f"in the operation list (positions {sorted(positions)}). "
            "A data-flow dependency crosses the group boundary."
        )


def _apply_work_div_symbol_remap(
    op: ComputedBuffer, remap: _IterationSymbolRemap | None
) -> None:
    """Move named work-division metadata through a proven symbol mapping."""

    if remap is None or not hasattr(op, "work_div_loop_info"):
        return

    old_names = op.work_div_loop_info  # type: ignore[attr-defined]
    unknown = set(old_names) - set(remap.before_symbols)
    if unknown:
        raise Unsupported(
            f"coarse_tile: work-division symbols are outside the captured "
            f"iteration space for {op.get_name()!r}: unknown={sorted(map(str, unknown))}, "
            f"captured={tuple(map(str, remap.before_symbols))}"
        )

    by_old_symbol = dict(remap.pairs)
    op.work_div_loop_info = {  # type: ignore[attr-defined]
        by_old_symbol[old_symbol]: list(names)
        for old_symbol, names in old_names.items()
        if old_symbol in by_old_symbol
    }


def _divide_reduction_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> _IterationSymbolRemap | None:
    """Divide the specified reduction_ranges entries of op by loop_count.

    Unlike _divide_ranges, does NOT update op.layout.size/stride — the
    output buffer shape is determined by data.ranges (non-reduction dims)
    and is unchanged by reduction-dim tiling.

    A reduction dim carrying a WhileLoop-splice loop_var hint is exempt --
    see _loop_var_hinted_reduction_ranges's docstring: reduction_ranges[i]
    is already the per-iteration extent for such a dim, so it is left
    unchanged.
    """
    data = op.data
    assert isinstance(data, Reduction)
    if not tiled_dims:
        return None
    hinted_reduction_ranges = _loop_var_hinted_reduction_ranges(op)
    before_symbols = None
    fused_before_symbols = None
    if hasattr(op, "work_div_loop_info"):
        try:
            before_symbols = _capture_logical_iteration_symbols(op)
        except Unsupported:
            fused_before_symbols = tuple(iteration_space_from_op(op))
    original_reduction_ranges = tuple(data.reduction_ranges)
    reduction_ranges = list(original_reduction_ranges)
    for i in tiled_dims:
        assert 0 <= i < len(reduction_ranges), (
            f"coarse_tile: op {op.get_name()!r} tiled reduction dim {i} out of bounds "
            f"(reduction_ranges has {len(reduction_ranges)} entries)"
        )
        if i in hinted_reduction_ranges:
            continue
        r = reduction_ranges[i]
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            loop_count, (int, sympy.Integer)
        ):
            if int(r) % int(loop_count) != 0:
                raise Unsupported(
                    f"coarse_tile: op {op.get_name()!r} reduction dim {i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"reduction dimensions must be evenly divisible by the loop trip count."
                )
            reduction_ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            reduction_ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)
    # Reduction is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "reduction_ranges", reduction_ranges)
    # reduction_ranges just changed unconditionally above, so any memoized
    # get_read_writes() result (pass_utils.op_read_writes) is stale
    # regardless of whether a symbol-remap capture path runs below --
    # invalidate unconditionally rather than only inside that branch.
    invalidate_op_read_writes(op)
    if before_symbols is None and fused_before_symbols is None:
        return None

    if before_symbols is not None:
        return _order_preserving_symbol_remap(
            op, before_symbols, _capture_logical_iteration_symbols(op)
        )

    active_before = [
        i
        for i, extent in enumerate(original_reduction_ranges)
        if sympy.sympify(extent) != 1
    ]
    removed = [i for i in active_before if sympy.sympify(data.reduction_ranges[i]) == 1]
    # Only reduction symbols at the end of the iteration space may disappear
    # without renumbering a survivor. Anything else remains ambiguous.
    trailing_removals = (
        len(removed) if removed and removed == active_before[-len(removed) :] else 0
    )
    assert fused_before_symbols is not None
    return _fused_iteration_symbol_remap(
        op, fused_before_symbols, max_trailing_removals=trailing_removals
    )


# ---------------------------------------------------------------------------
# Transformation entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Buffer propagation pass
# ---------------------------------------------------------------------------


def _validate_planned_reduction_tiling(
    op: ComputedBuffer,
    tiled_dims: list[list[int]],
    tiled_rdims: list[list[int]],
) -> None:
    """Raise Unsupported for unsupported Reduction tiling configurations.

    Supported:
      - A single level that tiles only a non-stick reduction dim.
      - A single level that tiles the stick (innermost) reduction dim, including
        the K dim of BATCH_MATMUL_OP and scalar reductions over dim=-1.
      - Multiple nesting levels where outer level(s) tile output dims and the
        innermost level tiles a reduction dim (e.g. outer M + inner K for mm).

    Deferred (raises Unsupported — reachable via a user-supplied spyre_hint,
    not an internal invariant violation):
      - Mixed output+reduction tiling at the same nesting level.
      - Multiple reduction range indices tiled at one level.

    Called from build_loop_tiling_plan (planning time): tiled_dims /
    tiled_rdims are the op's own per-level lists computed there, before any
    loop_member is stamped -- this check is a pure function of already-known
    shape data, so it doesn't need to wait for transformation to run.
    """
    # Pad both lists to the same length so zip covers all levels.
    n = max(len(tiled_dims), len(tiled_rdims))
    tiled_dims_padded = tiled_dims + [[]] * (n - len(tiled_dims))
    tiled_rdims_padded = tiled_rdims + [[]] * (n - len(tiled_rdims))

    for i, (out_dims, red_dims) in enumerate(
        zip(tiled_dims_padded, tiled_rdims_padded)
    ):
        if out_dims and red_dims:
            raise Unsupported(
                f"coarse_tile: op {op.get_name()!r} level {i} tiles both "
                f"output dim(s) {out_dims} and reduction dim(s) {red_dims} "
                "simultaneously (mixed output+reduction tiling at one level "
                "is not yet implemented)."
            )
        if len(red_dims) > 1:
            raise Unsupported(
                f"coarse_tile: op {op.get_name()!r} level {i} tiles multiple "
                f"reduction dims {red_dims} (tiling more than one reduction "
                "dim per level is not yet implemented)."
            )


# ---------------------------------------------------------------------------
# Consumer analysis
# ---------------------------------------------------------------------------


def _reads_buffer(op: ComputedBuffer, buf_name: str) -> bool:
    """Return True if op reads buf_name."""
    try:
        rw = op.get_read_writes()
    except Exception as e:
        logger.debug(
            "_reads_buffer: get_read_writes() raised for %s: %s", op.get_name(), e
        )
        return False
    return any(getattr(dep, "name", None) == buf_name for dep in rw.reads)


def _reads_buffer_cached(op: ComputedBuffer, buf_name: str) -> bool:
    """Planning-time analog of _reads_buffer that memoizes get_read_writes()
    via pass_utils.op_read_writes().

    Only safe to call from contexts that never mutate a *different* op's
    inner_fn between calls for the same op -- see the zero-mutation
    argument in coarse_tile_compile_time's design notes. Do NOT call this
    from transform-time helpers (the lowering, _find_outside_consumers) where earlier
    splices in the same pass can make another op's cached reads stale.
    """
    try:
        rw = op_read_writes(op)
    except Exception as e:
        logger.debug(
            "_reads_buffer_cached: get_read_writes() raised for %s: %s",
            op.get_name(),
            e,
        )
        return False
    return any(getattr(dep, "name", None) == buf_name for dep in rw.reads)


def _find_outside_consumers(
    buf_name: str,
    group_loop_id: tuple,
    operations: list[Operation],
) -> tuple[list[ComputedBuffer], bool]:
    """Return (consumer_ops, is_graph_output).

    consumer_ops: ComputedBuffers in operations that read buf_name and are
                  NOT in the same outermost loop group (loop_group_id[0]
                  differs or is absent).
    is_graph_output: True if buf_name appears in graph output names.
    """
    outer_key = group_loop_id[0]
    consumers: list[ComputedBuffer] = []
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not _reads_buffer(op, buf_name):
            continue
        path = counted_loop_path(op, include_extern=True)
        if not path or path[0] != outer_key:
            consumers.append(op)

    is_graph_output = buf_name in _graph_output_names()
    return consumers, is_graph_output


def _graph_output_names() -> set[str]:
    """Return the set of buffer names that appear in V.graph graph outputs."""
    try:
        return set(V.graph.get_output_names())
    except Exception as e:
        logger.debug("_graph_output_names: V.graph.get_output_names() raised: %s", e)
        return set()


# ---------------------------------------------------------------------------
# Full-buffer allocation
# ---------------------------------------------------------------------------


def _allocate_full_buffer(
    tiled_op: ComputedBuffer,
    full_ranges: list[Expr],
    full_strides: tuple[Expr, ...],
    operations: list[Operation],
    before: Operation,
    device_layout: SpyreTensorLayout | None = None,
) -> ComputedBuffer:
    """Allocate a full-sized HBM buffer for the tiled op's original shape.

    Creates a spyre.empty FX node, lowers it via V.graph.run_node(), assigns
    a layout matching tiled_op's layout type (FixedLayout pre-stickify,
    FixedTiledLayout post-stickify), splices it into operations
    immediately before ``before``, and returns the new ComputedBuffer.

    ``device_layout`` is the buffer's device layout, required post-stickify:
    the one planning recorded for a full buffer
    (the planned full layout of a copy-out or of an accumulator), or the
    tile's own for a tile-sized accumulator. It is not grown back from the
    tile's, which is ambiguous where the tile has two host dims of extent 1.
    """
    from ..ir import SpyreEmptyFallback  # deferred: avoids circular import

    orig_layout = tiled_op.layout
    if isinstance(orig_layout, FixedTiledLayout) and device_layout is None:
        raise RuntimeError(
            f"{tiled_op.get_name()} has a FixedTiledLayout but no "
            f"device_layout was given"
        )

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph
    device = tiled_op.get_device()
    dtype = tiled_op.get_dtype()

    # Evaluate full_ranges to concrete ints (they should be integer expressions).
    size = [int(r) for r in full_ranges]

    first_compute = next(n for n in fx_graph.nodes if n.op != "placeholder")
    with fx_graph.inserting_before(first_compute):
        empty_fx = fx_graph.create_node(
            "call_function",
            torch.ops.spyre.empty.default,
            args=(size, device, dtype),
        )
        empty_fx.meta["val"] = torch.empty(size, dtype=dtype, device="cpu")

    empty_tb = graph_lowering.run_node(empty_fx)
    graph_lowering.env[empty_fx] = empty_tb

    full_buf = empty_tb.data.data  # TensorBox → StorageBox → SpyreEmptyFallback
    assert isinstance(full_buf, SpyreEmptyFallback), (
        f"Expected SpyreEmptyFallback, got {type(full_buf).__name__}"
    )
    full_buf.origins = OrderedSet([empty_fx])

    # Assign a layout for the full-sized buffer.  Pre-stickify we use a plain
    # FixedLayout (stickification assigns the device layout later); post-stickify
    # we must build a FixedTiledLayout because stickification has already run.
    strides: list[Expr] = list(full_strides)

    layout: FixedTiledLayout | FixedLayout
    if isinstance(orig_layout, FixedTiledLayout):
        assert device_layout is not None  # checked above
        # Post-stickify path: the caller's device layout.
        layout = FixedTiledLayout(
            device,
            dtype,
            list(full_ranges),
            strides,
            device_layout,
        )
    else:
        # Pre-stickify path (for_each_tile groups): stickification has not yet
        # run, so assign a plain FixedLayout.  Stickification will propagate
        # SpyreTensorLayout to this buffer via the ExternKernel->generic_layout
        # path in propagate_spyre_tensor_layouts.
        #
        # This is logically a FlexibleLayout (the stride values below are
        # never read by stickification -- generic_layout builds
        # SpyreTensorLayout from .size alone), but it cannot be written that
        # way: full_buf gets read (via name-swapped consumer inner_fns)
        # before stickification runs, and
        # split_multi_ops traces those inner_fns by calling make_loader()/
        # make_indexer() on full_buf. Inductor's Layout.make_indexer()
        # (torch/_inductor/ir.py) asserts FlexibleLayout.allow_indexing --
        # a FlexibleLayout buffer cannot be indexed until frozen to a
        # concrete layout. Using FlexibleLayout here makes that assertion
        # fire, split_multi_ops silently drops the trace, and any scalar
        # constant in the consumer's inner_fn never gets materialized into a
        # SpyreConstantFallback buffer -- it survives as a raw Constant all
        # the way to codegen, which SpyreKernel.store() rejects. So
        # FixedLayout is required here despite the stride values being
        # otherwise meaningless.
        layout = FixedLayout(
            device,
            dtype,
            list(full_ranges),
            strides,
        )
    full_buf.layout = layout

    # Splice in immediately before ``before``; the lowering appended it at the end.
    move_before(operations, before, full_buf)

    return full_buf


# ---------------------------------------------------------------------------
# Case 1: copy op insertion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Case: reduction-dim tiling — combine op insertion
# ---------------------------------------------------------------------------


def _make_combine_buffer(
    tiled_op: ComputedBuffer, accum_buf: ComputedBuffer
) -> ComputedBuffer:
    """The pointwise op that folds ``tiled_op``'s partial result into ``accum_buf``
    in place (``MutationLayoutSHOULDREMOVE`` onto the accumulator), without loop
    metadata: the caller stamps that and inserts it."""
    from torch._inductor.virtualized import ops as vops

    reduction_type = tiled_op.data.reduction_type
    partial_loader = tiled_op.make_loader()
    accum_loader = accum_buf.make_loader()

    def combine_inner_fn(index):
        partial = partial_loader(index)
        accum = accum_loader(index)
        if reduction_type in ("sum", BATCH_MATMUL_OP):
            return vops.add(accum, partial)
        if reduction_type == "xor_sum":
            return vops.bitwise_xor(accum, partial)
        if reduction_type == "prod":
            return vops.mul(accum, partial)
        if reduction_type == "max":
            return vops.maximum(accum, partial)
        if reduction_type == "min":
            return vops.minimum(accum, partial)
        if reduction_type == "any":
            return vops.logical_or(accum, partial)
        raise RuntimeError(
            f"coarse_tile: _make_combine_buffer: unsupported reduction_type "
            f"{reduction_type!r}"
        )

    combine_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=combine_inner_fn,
        ranges=list(tiled_op.data.ranges),
    )
    combine_name = V.graph.qualify_name(f"coarse_tile_combine_{tiled_op.get_name()}")
    combine_buf = ComputedBuffer(
        name=combine_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(accum_buf))),
        data=combine_data,
    )
    combine_buf.origins = tiled_op.origins
    combine_buf.operation_name = combine_name
    return combine_buf


# ---------------------------------------------------------------------------
# Consumer / graph-output patching
# ---------------------------------------------------------------------------


def _patch_consumers(
    consumers: list[ComputedBuffer],
    old_name: str,
    new_name: str,
    operations: list[Operation],
    retile_info: _RetiledBufferInfo | None = None,
    *,
    producer_layout: Layout | None = None,
    read_layout: Layout | None = None,
) -> None:
    """Redirect consumers from old_name to new_name.

    Patches each consumer's inner_fn with a NameSwapHandler and reconstructs
    the ComputedBuffer to invalidate the sizes cache. The load index is
    handled one of two ways:

    ``read_layout`` (with ``producer_layout``, the layout object the
    consumers' loaders are bound to): the consumer is traced with
    ``producer_layout`` presenting ``read_layout``'s host shape
    (``_ReadThroughLayout``), so its index is derived from the buffer it now
    reads. This is the copy-out path: an outside consumer belongs to another
    loop nest, and the full buffer is the only thing the two nests share.

    ``retile_info`` carries the old (tile-local) and new (full-size) strides
    of the renamed buffer, and the index traced against the tile is rescaled
    (``_NameAndIndexSwapHandler``) when they differ. This is the reduction
    accumulator path, which redirects same-nest consumers as well.
    """
    if not consumers or old_name == new_name:
        return
    assert (producer_layout is None) == (read_layout is None), (
        "_patch_consumers: producer_layout and read_layout go together"
    )
    assert retile_info is None or read_layout is None, (
        "_patch_consumers: a consumer's index is either derived from the read "
        "buffer or rescaled from the tile, not both"
    )

    from ..pass_utils import NameSwapHandler

    name_map = {old_name: new_name}
    has_retile = (
        retile_info is not None and retile_info.old_stride != retile_info.new_stride
    )

    for consumer in consumers:
        orig_inner = consumer.data.inner_fn

        def new_inner_fn(
            *args,
            _map=name_map,
            _info=retile_info if has_retile else None,
            _orig=orig_inner,
            _consumer=consumer,
            _bound=producer_layout,
            _read=read_layout,
        ):
            if _info is not None:
                handler = _NameAndIndexSwapHandler(
                    V.ops, _map, {old_name: _info}, _consumer
                )
            else:
                handler = NameSwapHandler(V.ops, _map)
            read_through = (
                _ReadThroughLayout(_bound, _read)
                if _read is not None
                else contextlib.nullcontext()
            )
            with read_through, V.set_ops_handler(handler):
                return _orig(*args)

        object.__setattr__(consumer.data, "inner_fn", new_inner_fn)
        new_consumer = replace_computed_buffer_body(
            consumer,
            consumer.data,
            operations,
            pass_name="coarse_tile",
            reason="redirect outside consumer to full-sized buffer",
        )
        V.graph.name_to_buffer[new_consumer.get_name()] = operations[
            next(
                i
                for i, op in enumerate(operations)
                if isinstance(op, ComputedBuffer)
                and op.get_name() == new_consumer.get_name()
            )
        ]


class _NameAndIndexSwapHandler(WrapperHandler):
    """Redirect ops.load(name, index) to a new name and rewrite its index.

    Rewrites the index while `name` is still the old name (its coefficients
    are info.old_stride), then swaps the name.  Calls _retile_load_index,
    which raises Unsupported if the index cannot be retiled.
    """

    def __init__(
        self,
        inner,
        name_map: dict[str, str],
        infos_by_old_name: dict[str, _RetiledBufferInfo],
        consumer: "ComputedBuffer | None" = None,
    ):
        super().__init__(inner)
        self._name_map = name_map
        self._infos_by_old_name = infos_by_old_name
        self._consumer = consumer

    def load(self, name, index):
        if name in self._infos_by_old_name:
            index = _retile_load_index(
                name,
                index,
                self._infos_by_old_name[name],
                self._consumer,
                preserve_target_stride_atoms=True,
            )
        return super().load(self._name_map.get(name, name), index)


def _patch_graph_outputs(old_name: str, new_buf: ComputedBuffer) -> None:
    """Replace references to old_name in V.graph.graph_outputs with new_buf.

    A graph output is often not the tiled op's ComputedBuffer directly, but a
    ReinterpretView over it (e.g. from a trailing unsqueeze/view the caller
    applied to the reduction's result) wrapping a StorageBox wrapping the
    ComputedBuffer. That ReinterpretView's own layout is computed from the
    op's true (pre-coarse-tiling) shape, so it already describes the correct
    full-sized addressing -- only its underlying storage still points at the
    tile-local scratch buffer. Unwrap through StorageBox *and*
    ReinterpretView, and when a ReinterpretView is found, repoint its own
    `.data` in place (preserving its layout) rather than substituting a bare
    TensorBox that would discard the view's reshape/stride.
    """
    try:
        outputs = V.graph.graph_outputs
    except Exception:
        return

    new_tb = TensorBox(StorageBox(new_buf))
    for i, out in enumerate(outputs):
        # Unwrap StorageBox/ReinterpretView layers to reach ComputedBuffer
        # without going into the ComputedBuffer's inner data (Pointwise /
        # Reduction). Track the last ReinterpretView seen so we can patch its
        # storage in place instead of discarding its view metadata.
        candidate = out
        last_reinterpret_view = None
        while isinstance(candidate, (StorageBox, ReinterpretView)):
            if isinstance(candidate, ReinterpretView):
                last_reinterpret_view = candidate
            candidate = candidate.data
        if not (
            isinstance(candidate, ComputedBuffer) and candidate.get_name() == old_name
        ):
            continue
        if last_reinterpret_view is not None:
            object.__setattr__(last_reinterpret_view, "data", StorageBox(new_buf))
        else:
            outputs[i] = new_tb
