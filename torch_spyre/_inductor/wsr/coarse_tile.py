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

"""Coarse-tiling IR pass: stamp loop_group_id / loop_count on ir.Operation objects.

Each group of operations is wrapped in one or more nested counted loops.  For
every operation in the group the iteration ranges divided by each loop's trip
count are scaled down by that factor; the resulting (smaller) per-iteration
ranges are what the downstream scheduler and work-division passes will see.

A ``loop_group_id`` tuple encodes the nesting path:
  - ``(g,)``       — outermost loop group with index ``g``
  - ``(g, h)``     — inner loop group ``h`` nested inside outer group ``g``
  - etc.

``loop_count`` is a *list* of trip counts, one per nesting level from outermost
to innermost.  For a single-level group this is a 1-element list ``[K]``.
``loop_tiled_dims`` is a *list of lists*, one sub-list per nesting level.

Entry point::

    groups = hints_to_coarse_tile_groups(graph)
    coarse_tile(graph, groups)

``groups`` is a list of ``(ops, levels)`` tuples where ``levels`` is a list of
``(hint_id, count)`` pairs, outermost first.  Each op resolves its own
tiled dimension from its ``loop_var`` in ``dim_hints``.

Each ``ops`` list must be a contiguous sub-sequence of ``operations``.

After stamping, ``coarse_tile`` calls ``insert_tiling_propagation`` to allocate
full-sized output buffers and insert copy/mutation ops for Pointwise operations
whose results are consumed outside the loop.

Before touching any ``inner_fn``/``layout``/``MutationLayoutSHOULDREMOVE``
rewiring in this file, read "Appendix: How IR rewiring works, and why it's
sound" in ``docs/source/compiler/coarse_tiling_loops.md``. It documents the
wrap-never-reconstruct convention and why ``MutationLayoutSHOULDREMOVE``
sites must satisfy the single-mutation-target invariant -- the same ground
this file's rewrite sites depend on.
"""

from __future__ import annotations


import dataclasses
from collections import Counter
from typing import NamedTuple

import sympy
from sympy import Expr

import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import sympy_subs
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    IRNode,
    Layout,
    Loops,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import SpyreTensorLayout

from .. import config
from ..constants import BATCH_MATMUL_OP
from ..errors import Unsupported
from ..logging_utils import get_inductor_logger
from ..loop_info import CoarseTileInfo, copy_op_metadata
from ..propagate_hints import DimHint
from ..pass_utils import op_out_coords, host_coordinates, indirect_sizes_from_op
from ..ir import FixedTiledLayout, SpyreConstantFallback, _resize_device_layout

logger = get_inductor_logger("coarse_tile")


class _RetiledBufferInfo(NamedTuple):
    """Host strides before and after a buffer is resized for a coarse tile."""

    old_stride: tuple[Expr, ...]
    new_stride: tuple[Expr, ...]


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


# ---------------------------------------------------------------------------
# Cache-invalidation helpers
# ---------------------------------------------------------------------------


def _cache_key(cached_method: object) -> str:
    """Return the cache attribute name used by a cache_on_self / cache_on_self_and_args method.

    cache_on_self uses key ``f"__{fn.__name__}_cache"``; cache_on_self_and_args uses
    ``f"__{class_name}_{fn.__name__}_cache"``.  Both patterns are captured as the
    ``key`` free variable in the method's ``.clear_cache`` closure — extract it once
    at module load so misspellings or upstream renames fail loudly on import.
    """
    clear_fn = getattr(cached_method, "clear_cache")  # AttributeError if absent
    for i, name in enumerate(clear_fn.__code__.co_freevars):
        if name == "key":
            return clear_fn.__closure__[i].cell_contents
    raise AttributeError(
        f"Cannot find 'key' in clear_cache closure of {cached_method!r}"
    )


# Resolve cache keys once at import time — any rename in upstream IR will raise
# AttributeError here rather than silently no-oping at runtime.
_LOOPS_FREE_SYMS_KEY = _cache_key(Loops.get_free_symbol_uses)
_LOOPS_INNER_FN_STR_KEY = _cache_key(Loops.inner_fn_str)
_LOOPS_INNER_FN_OPCOUNT_KEY = _cache_key(Loops.inner_fn_opcount)
_REDUCTION_FREE_SYMS_KEY = _cache_key(Reduction.get_free_symbol_uses)
_LAYOUT_FREE_SYMS_KEY = _cache_key(Layout.get_free_symbol_uses)
_COMPUTED_BUF_FREE_SYMS_KEY = _cache_key(ComputedBuffer.get_free_symbol_uses)
_COMPUTED_BUF_SIZES_KEY = _cache_key(ComputedBuffer.get_default_sizes_body)


def _clear_cache(obj: object, key: str) -> None:
    # cache_on_self/cache_on_self_and_args store results via object.__setattr__ to
    # bypass frozen-dataclass guards (Loops, Reduction, Layout); clearing must also
    # use object.__delattr__ — plain delattr() raises FrozenInstanceError.
    if hasattr(obj, key):
        object.__delattr__(obj, key)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_coarse_tile_groups(
    operations: list[Operation],
    groups: list[tuple],
) -> dict[int, CoarseTileInfo]:
    """Decide every op's coarse-tiling attributes without mutating the IR.

    Performs the per-op decision logic (hint-to-position lookup, per-level
    tiled-dims bookkeeping, tiled_dims_per_read/output_tiled_dims) but never
    calls _divide_ranges/_divide_reduction_ranges -- those are real IR
    mutation and must only run during transformation (see _apply_plan).
    Extents are instead computed analytically by
    _planned_tile_extents_per_level, reading op.data.ranges/reduction_ranges
    as they exist before any mutation.

    Returns a dict mapping each tiled op's ``id(op)`` to its planned
    CoarseTileInfo. Keyed by ``id(op)`` rather than ``op`` itself because
    ir.Operation/ComputedBuffer are (unsafe_hash=False, eq=True) dataclasses
    -- Python therefore sets their __hash__ to None, so they cannot be used
    directly as dict keys (confirmed: ``{ComputedBuffer(...): 1}`` raises
    ``TypeError: unhashable type``). ``id(op)`` still gives exact identity
    semantics (the same object passed in via ``groups``, unmodified), and
    matches the existing ``{id(op): ...}`` convention already used for
    op-identity dicts elsewhere in this codebase (see
    ``torch_spyre/_inductor/passes.py``'s ``op_order`` dicts).

    Untiled/skipped ops (non-ComputedBuffer) have no entry.
    """
    plan: dict[int, CoarseTileInfo] = {}
    for group_idx, (group_ops, levels) in enumerate(groups):
        group_id: tuple[int, ...] = (group_idx,)
        nested_group_id: tuple[int, ...] = group_id + (0,) * (len(levels) - 1)
        counts = [count for _, count in levels]
        group_reduction_tiled_levels = _group_reduction_tiled_levels_in_group(
            group_ops, levels
        )

        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue

            op_out = op_out_coords(op)
            rw = op.get_read_writes()
            read_deps = [d for d in rw.reads if isinstance(d, MemoryDep)]
            write_deps = [d for d in rw.writes if isinstance(d, MemoryDep)]

            hint_id_to_ranges_pos: dict[int, int] = {
                h.hint_id: pos
                for h in getattr(op, "dim_hints", [])
                if h.loop_var is not None and not h.is_reduction
                if (pos := _loop_var_to_ranges_pos(op_out, h.loop_var)) is not None
            }
            hint_id_to_reduction_ranges_pos: dict[int, int] = {}
            if isinstance(op.data, Reduction):
                hint_id_to_reduction_ranges_pos = {
                    h.hint_id: pos
                    for h in getattr(op, "dim_hints", [])
                    if h.loop_var is not None and h.is_reduction
                    if (pos := _loop_var_to_reduction_ranges_pos(op, h.loop_var))
                    is not None
                }

            op_tiled_dims: list[list[int]] = []
            op_tiled_reduction_dims: list[list[int]] = []
            for hint_id, _count in levels:
                opos = hint_id_to_ranges_pos.get(hint_id)
                rpos = hint_id_to_reduction_ranges_pos.get(hint_id)
                op_tiled_dims.append([opos] if opos is not None else [])
                op_tiled_reduction_dims.append([rpos] if rpos is not None else [])

            has_tiled_reduction = any(op_tiled_reduction_dims)
            if has_tiled_reduction and not config.enable_reduction_tiling:
                raise Unsupported(
                    f"reduction-dim tiling for op {op.get_name()} "
                    "(disabled via enable_reduction_tiling)"
                )

            if _plan_is_loop_invariant_at_reduction_levels(
                op, op_tiled_dims, group_reduction_tiled_levels
            ):
                if _seed_buffer_for_carry(op, group_ops) is not None:
                    raise Unsupported(
                        f"reduction-dim tiling requiring carry propagation for "
                        f"op {op.get_name()}"
                    )

            per_level_extents = _planned_tile_extents_per_level(
                op, op_tiled_dims, op_tiled_reduction_dims, levels
            )

            tiled_dims_per_read = [
                _tiled_dims_for_dep(dep, per_level_extents) for dep in read_deps
            ]
            output_tiled_dims = (
                _tiled_dims_for_dep(write_deps[0], per_level_extents)
                if write_deps
                else []
            )

            plan[id(op)] = CoarseTileInfo(
                loop_group_id=nested_group_id,
                loop_count=counts,
                loop_tiled_dims=op_tiled_dims,
                loop_tiled_reduction_dims=op_tiled_reduction_dims,
                tiled_dims_per_read=tiled_dims_per_read,
                output_tiled_dims=output_tiled_dims,
            )

            logger.debug(
                "coarse_tile: planned %s loop_group_id=%s loop_count=%s "
                "loop_tiled_dims=%s loop_tiled_reduction_dims=%s "
                "tiled_dims_per_read=%s output_tiled_dims=%s",
                op.get_operation_name(),
                nested_group_id,
                counts,
                op_tiled_dims,
                op_tiled_reduction_dims,
                tiled_dims_per_read,
                output_tiled_dims,
            )

    return plan


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
    """
    counts_by_dim: dict[int, Expr] = {}
    counts_by_reduction_dim: dict[int, Expr] = {}
    for level_idx, (_, count) in enumerate(levels):
        for d in op_tiled_dims[level_idx]:
            counts_by_dim[d] = counts_by_dim.get(d, sympy.Integer(1)) * count
        for d in op_tiled_reduction_dims[level_idx]:
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
    final_reduction_extents = {}
    if isinstance(op.data, Reduction):
        final_reduction_extents = {
            d: _divided(op.data.reduction_ranges[d], count, f"reduction dim {d}")
            for d, count in counts_by_reduction_dim.items()
        }

    n_output_dims = len(op.data.ranges) if hasattr(op.data, "ranges") else 0

    def _per_level_extent_for(
        final_extent: Expr,
        tiled_at_level: list[list[int]],
        dim_id: int,
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
        # the final extent times every more-inner level's count.
        running_extent = final_extent
        for level_idx in reversed(levels_tiling_dim):
            result[level_idx] = running_extent
            running_extent = running_extent * levels[level_idx][1]
        return result

    per_level_output: list[dict[int, Expr]] = [dict() for _ in levels]
    for d, final_extent in final_dim_extents.items():
        level_extents = _per_level_extent_for(final_extent, op_tiled_dims, d)
        for level_idx, extent in level_extents.items():
            per_level_output[level_idx][d] = extent
    for d, final_extent in final_reduction_extents.items():
        dim_key = n_output_dims + d
        level_extents = _per_level_extent_for(final_extent, op_tiled_reduction_dims, d)
        for level_idx, extent in level_extents.items():
            per_level_output[level_idx][dim_key] = extent

    return per_level_output


def _tiled_dims_for_dep(
    dep: MemoryDep,
    per_level_extents: list[dict[int, Expr]],
) -> list[list[tuple[int, Expr]]]:
    """Filter per-level tiled-dim extents down to dims dep.index actually reads.

    A dim tiled at some level that this dependency's index does not depend
    on (broadcast, or simply not one of its dims) must not appear in its
    per-level list -- matching the implicit zeroing _tile_advance_expr_from_dep
    performs today for any free symbol absent from tiled_dim_extents.
    """
    dep_dims = {
        int(str(sym)[1:])
        for sym in dep.index.free_symbols
        if str(sym).startswith("d") and str(sym)[1:].isdigit()
    }
    return [
        [(d, extent) for d, extent in level.items() if d in dep_dims]
        for level in per_level_extents
    ]


def _stick_host_dim(op: ComputedBuffer, device_layout) -> int | None:
    """Authoritative stick host-dim index for ``op``'s output, recovered from
    coordinate identity (issue #3116).

    ``SpyreTensorLayout`` discards its ``dim_map`` at construction, so the
    host<->device dim identity is not carried on the layout object.  We recover
    only the stick host dim: the device layout's inner-stick coordinate has a
    single iteration symbol that also drives exactly one host coordinate, so
    ``matching_dim`` resolves it unambiguously — even when two host dims share a
    size (transposed flash-attn QK^T with ``Sq == Skv``), which defeats the
    size-based inference in ``_resize_device_layout``.

    This is the same identity mechanism ``_pick_stick_dim`` uses to choose a
    stick dim, so it is as reliable as the existing stick logic.  Returns
    ``None`` when identity cannot be resolved (single-symbol match not unique),
    so the caller falls back to size-based inference.

    The stick host dim is invariant under coarse tiling (tiling shrinks a range
    but does not change which axis is the stick), so this may be computed either
    before or after ``_divide_ranges`` mutates the ranges.
    """
    from ..pass_utils import (
        try_device_coordinates,
    )
    from ..views import matching_dim

    try:
        writes = op.get_read_writes().writes
        if not writes:
            return None
        out_dep = next(iter(writes))
        ind_sizes = indirect_sizes_from_op(op)
        dcoords = try_device_coordinates(device_layout, out_dep, ind_sizes)
        if not dcoords:  # None (unrepresentable stick) or empty → no identity
            return None
        hcoords = host_coordinates(op.get_layout(), out_dep, ind_sizes)
        return matching_dim(hcoords, dcoords[-1])
    except Exception:
        # Identity recovery is best-effort; any failure falls back to inference.
        return None


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
    plan_coarse_tile_groups's hint_id_to_reduction_ranges_pos, gated on
    isinstance(op.data, Reduction)), so this scan only needs to inspect
    Reduction ops; a Pointwise-only group always yields an empty set.
    """
    reduction_levels: set[int] = set()
    for o in group_ops:
        if not isinstance(o, ComputedBuffer) or not isinstance(o.data, Reduction):
            continue
        hint_id_to_reduction_ranges_pos: dict[int, int] = {
            h.hint_id: pos
            for h in getattr(o, "dim_hints", [])
            if h.loop_var is not None and h.is_reduction
            if (pos := _loop_var_to_reduction_ranges_pos(o, h.loop_var)) is not None
        }
        for level_idx, (hint_id, _count) in enumerate(levels):
            if hint_id in hint_id_to_reduction_ranges_pos:
                reduction_levels.add(level_idx)
    return reduction_levels


def _group_reduction_tiled_hint_ids(
    group_ops: list[Operation], levels: list[tuple]
) -> set[int]:
    """Return hint_ids among ``levels`` that tile a reduction dim for some
    Reduction op in group_ops."""
    reduction_hint_ids: set[int] = set()
    for hint_id, _count in levels:
        for op in group_ops:
            if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Reduction):
                continue
            _, has_reduction_pos = _op_hint_dim_positions(op, hint_id)
            if has_reduction_pos:
                reduction_hint_ids.add(hint_id)
                break
    return reduction_hint_ids


def _op_hint_dim_positions(op: ComputedBuffer, hint_id: int) -> tuple[bool, bool]:
    """Return (has_output_pos, has_reduction_pos) for op at the given hint_id.

    has_output_pos: op's own output ranges contain a dim driven by this hint.
    has_reduction_pos: op is a Reduction whose reduction_ranges contain a dim
    driven by this hint.  This is the hint-to-position lookup used to decide
    which of op's own dims a given hint_id tiles.
    """
    dim_hint = next(
        (h for h in getattr(op, "dim_hints", []) if h.hint_id == hint_id), None
    )
    if dim_hint is None or dim_hint.loop_var is None:
        return False, False
    if dim_hint.is_reduction:
        if not isinstance(op.data, Reduction):
            return False, False
        pos = _loop_var_to_reduction_ranges_pos(op, dim_hint.loop_var)
        return False, pos is not None
    op_out = op_out_coords(op)
    pos = _loop_var_to_ranges_pos(op_out, dim_hint.loop_var)
    return pos is not None, False


def _is_loop_invariant_at_reduction_levels(
    op: ComputedBuffer, group_ops: list[Operation], levels: list[tuple]
) -> bool:
    """True if op is a Pointwise that is loop-invariant at every hint level
    where the group tiles a Reduction's reduction dim, regardless of whether
    op is tiled at other (non-reduction) levels.

    This is the carry-candidate shape test: an online-softmax recurrence op
    (running max, rescale-accumulate) is tiled at outer output-dim levels
    (e.g. H, a real dim of its own output) exactly like any other op in the
    group, but is invariant in shape at the inner reduction-tiled level
    (e.g. Lk) because that dim never appears in its own output. Consulted
    here from dim_hints directly, rather than from stamped loop_info, because
    the caller (_replace_constant_fill_predecessors) runs before _apply_plan
    has stamped loop_info onto any op.
    """
    if not isinstance(op.data, Pointwise):
        return False
    reduction_hint_ids = _group_reduction_tiled_hint_ids(group_ops, levels)
    if not reduction_hint_ids:
        return False
    for hint_id in reduction_hint_ids:
        has_output_pos, _ = _op_hint_dim_positions(op, hint_id)
        if has_output_pos:
            return False
    return True


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


def _seed_buffer_for_carry(
    op: ComputedBuffer,
    group_ops: list[Operation],
) -> ComputedBuffer | None:
    """Return the pre-loop seed buffer op carries state through, or None.

    A Pointwise op that is loop-invariant at the group's reduction-tiled
    level(s) may be the carry-producing step of an online-softmax-style
    recurrence (running max, rescale-accumulate) rather than an ordinary
    broadcast/hoisted computation. Detection is closure-based rather than
    classifying op in isolation, because the seed's closure (the set of ops
    that read it, directly or transitively, without leaving the loop group)
    may have more than one member — see _seed_closure — and no single op's
    own consumer count or escape-the-loop status reliably identifies "the"
    carry (that correspondence to the traced Python's recurrence-variable
    rebinding is not recoverable from any one op in isolation):

      1. op must read exactly one pre-loop seed buffer directly (a constant
         fill — see _is_constant_fill — whose own reads all resolve to a
         SpyreConstantFallback scalar; torch.full/torch.zeros/
         torch.zeros_like lower to such a Pointwise wrapper. The seed may
         or may not have a stamped in-group loop_group_id, depending on
         whether its Python-source declaration sits inside or outside the
         tiled scope — that placement does not affect its seed status).
      2. op must be a member of that seed's closure (trivially true, since
         op reads the seed directly).
      3. op must be the *unique* closure member whose non-seed operands are
         all external to the closure (_closure_member_has_external_operands_only).
         If zero or more than one closure member satisfies this, return None
         rather than guessing — this is a known, accepted limitation for
         closures with more than one externally-fed member (not hit by any
         current test).

    Caller (plan_coarse_tile_groups) is responsible for the shape gate
    (_plan_is_loop_invariant_at_reduction_levels); this function only
    checks the seed-buffer data-flow shape.

    Uses the pre-stamp closure helper (_seed_closure_pre_stamp) because
    planning is zero-mutation and runs before _apply_plan stamps
    loop_info -- the post-stamp _seed_closure would find no matches here.
    """
    if not isinstance(op.data, Pointwise):
        return None

    seed_candidates = []
    for name in _op_reads(op):
        buf = V.graph.get_buffer(name)
        if not isinstance(buf, ComputedBuffer) or not _is_constant_fill(buf):
            continue
        # _is_constant_fill already requires every read of buf to come from
        # a SpyreConstantFallback scalar — an op with real in-group operands
        # can never satisfy it, so no additional loop_group_id check is
        # needed to exclude "produced inside the loop" buffers. A seed can
        # legitimately carry a stamped in-group loop_group_id (e.g. when its
        # Python-source declaration sits inside the tiled scope).
        seed_candidates.append(buf)

    if len(seed_candidates) != 1:
        return None
    seed_buf = seed_candidates[0]
    seed_name = seed_buf.get_name()

    closure = _seed_closure_pre_stamp(seed_name, group_ops)
    if op.get_name() not in closure:
        return None

    external_candidates = [
        name
        for name in closure
        if _closure_member_has_external_operands_only(
            name, seed_name, closure, group_ops
        )
    ]
    if len(external_candidates) != 1:
        logger.warning(
            "_seed_buffer_for_carry: ambiguous carry detection for seed %s "
            "(closure=%s) — found %d externally-fed closure members, "
            "expected exactly 1; treating %s as not a carry step. See "
            "_seed_buffer_for_carry's docstring, point 3.",
            seed_name,
            sorted(closure),
            len(external_candidates),
            op.get_name(),
        )
        return None

    return seed_buf if external_candidates[0] == op.get_name() else None


def _seed_closure(
    seed_name: str, loop_group_id: tuple, operations: list[Operation]
) -> set[str]:
    """Return the closure of seed_name within its outer loop group.

    The closure is every op in the same outer loop group that reads
    seed_name *directly* — e.g. both `max_running = maximum(M, block_max)`
    and `correction = exp(M - max_running)` read `M` directly, so both are
    in `M`'s closure. This is intentionally NOT transitive: an op that reads
    a closure member but not the seed itself (e.g.
    `denominator = denominator * correction + ...`, which reads `correction`
    but not `M`) is an ordinary downstream consumer, not part of the seed's
    closure — including it would pull in the whole downstream dependency
    cone, which is a different (much larger, wrong) set.

    Restricted to ops stamped with loop_info in the same outer group (i.e.
    after _apply_plan has stamped loop_info; see _seed_closure_pre_stamp for
    the pre-stamp equivalent used by _replace_constant_fill_predecessors).
    """
    outer_key = loop_group_id[0]
    closure: set[str] = set()
    for o in operations:
        if not isinstance(o, ComputedBuffer):
            continue
        li = getattr(o, "loop_info", None)
        if li is None or li.loop_group_id[0] != outer_key:
            continue
        if seed_name in _op_reads(o):
            closure.add(o.get_name())
    return closure


def _closure_member_has_external_operands_only(
    op_name: str,
    seed_name: str,
    closure: set[str],
    operations: list[Operation],
) -> bool:
    """True if op_name's non-seed read operands are all outside closure.

    This is the carry-producing member test: a true recurrence-update step
    combines the previous carry value (the seed) with fresh, externally
    derived per-iteration data. A step that combines the seed with an
    already-computed sibling closure member is downstream of the actual
    update, not the update itself (e.g. `correction = exp(M - max_running)`
    reads `max_running`, a closure member, so it is excluded even though it
    also reads the seed `M` directly).
    """
    op = V.graph.name_to_buffer.get(op_name)
    if op is None:
        return False
    non_seed_reads = _op_reads(op) - {seed_name}
    return not (non_seed_reads & closure)


def _op_reads(op: ComputedBuffer) -> set[str]:
    """Return the set of buffer names op reads (via MemoryDep)."""
    return {d.name for d in op.get_read_writes().reads if isinstance(d, MemoryDep)}


# ---------------------------------------------------------------------------
# Shared leaf helpers
# ---------------------------------------------------------------------------


def _loop_var_to_ranges_pos(out_coords: list, sym: sympy.Symbol) -> int | None:
    """Return the position of loop variable sym in op.data.ranges, or None.

    Looks up sym in the op's output coordinates — the only reliable mapping
    from a loop variable symbol to its data.ranges position, since dep var
    numbering skips size-1 dims while data.ranges does not.
    """
    for i, coord in enumerate(out_coords):
        if len(coord.free_symbols) == 1 and next(iter(coord.free_symbols)) == sym:
            return i
    return None


def _loop_var_to_reduction_ranges_pos(
    op: ComputedBuffer, sym: sympy.Symbol
) -> int | None:
    """Return position of loop variable sym in op.data.reduction_ranges, or None.

    Uses dep-tracking symbols (d0, d1, ...) rather than SymT.R0_INDEX symbols
    (r0_0, r0_1, ...) which are a different namespace.  Finds reduction symbols
    by set-subtracting output index symbols from input index symbols, in
    dep.ranges order (which matches reduction_ranges order).
    """
    assert isinstance(op.data, Reduction)
    rw = op.get_read_writes()
    out_dep = next(iter(rw.writes))
    out_syms = out_dep.index.free_symbols
    in_dep = next(d for d in rw.reads if hasattr(d, "index"))
    reduction_syms = [s for s in in_dep.ranges if s not in out_syms]
    try:
        return reduction_syms.index(sym)
    except ValueError:
        return None


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


def _divide_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> _RetiledBufferInfo | None:
    """Divide the specified iteration ranges of op by loop_count.

    For a ``Pointwise`` the full ranges are op.data.ranges.
    For a ``Reduction`` the non-reduction (outer) ranges are op.data.ranges;
    op.data.reduction_ranges are left untouched.

    ``tiled_dims`` is a list of positional indices into ``data.ranges``.
    All indices must be valid; an out-of-bounds index is a caller bug.

    Also updates ``op.layout.size``, ``op.layout.stride``, and
    ``op.layout.device_layout`` so the layout describes the smaller per-tile
    buffer, not the full tensor.  Contiguous host strides are recomputed from
    the new size; the ``SpyreTensorLayout`` is rebuilt from the new host size
    and strides, preserving the within-stick dimension from the original layout.
    """
    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        return None

    ranges = list(data.ranges)
    if not ranges:
        return None

    for i in tiled_dims:
        assert 0 <= i < len(ranges), (
            f"coarse_tile: op {op.get_name()!r} tiled dim {i} out of bounds "
            f"(ranges has {len(ranges)} entries)"
        )
        r = ranges[i]
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            loop_count, (int, sympy.Integer)
        ):
            if int(r) % int(loop_count) != 0:
                raise Unsupported(
                    f"coarse_tile: op {op.get_name()!r} loop var d{i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"dimensions must be evenly divisible by the loop trip count."
                )
            ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)

    # Loops is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "ranges", ranges)

    # Invalidate Loops-level caches that read ranges.
    _clear_cache(data, _LOOPS_FREE_SYMS_KEY)
    _clear_cache(data, _LOOPS_INNER_FN_STR_KEY)
    _clear_cache(data, _LOOPS_INNER_FN_OPCOUNT_KEY)
    if isinstance(data, Reduction):
        _clear_cache(data, _REDUCTION_FREE_SYMS_KEY)

    # Invalidate ComputedBuffer-level caches derived from data.ranges.
    _clear_cache(op, _COMPUTED_BUF_SIZES_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)

    # Sync layout.size, layout.stride, and layout.device_layout with the new ranges.
    layout = getattr(op, "layout", None)
    if not (isinstance(layout, FixedLayout) and len(layout.size) == len(ranges)):
        return None

    old_stride = tuple(layout.stride)
    new_size = list(layout.size)
    for i in tiled_dims:
        new_size[i] = ranges[i]
    layout.size = new_size

    # Recompute contiguous strides for the smaller buffer.
    layout.stride = list(FlexibleLayout.contiguous_strides(new_size))

    # Invalidate Layout- and ComputedBuffer-level caches that read size/stride.
    _clear_cache(layout, _LAYOUT_FREE_SYMS_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)
    retiled_info = (
        _RetiledBufferInfo(old_stride, tuple(layout.stride))
        if tiled_dims and old_stride != tuple(layout.stride)
        else None
    )

    # Rebuild SpyreTensorLayout for the new host size using device-native
    # reconstruction: transform the original device layout directly without
    # guessing a dim_order.
    if not isinstance(layout, FixedTiledLayout):
        return retiled_info
    # Capture old/new sizes as ints here, after the FixedTiledLayout guard,
    # so symbolic-size FixedLayout tests above are not affected.
    # layout.size is already the new (divided) size; reconstruct the old size
    # by multiplying tiled dims back up: old[i] = new[i] * loop_count.
    old_host_size = [int(s) for s in layout.size]
    for i in tiled_dims:
        old_host_size[i] = int(new_size[i] * loop_count)
    new_size_ints = [int(s) for s in new_size]
    # Recover the authoritative stick host dim from coordinate identity so
    # _resize_device_layout does not have to infer it by size (ambiguous for
    # transposed same-size dims — issue #3116). Tiling-invariant, so safe here.
    stick_hd = _stick_host_dim(op, layout.device_layout)
    layout.device_layout = _resize_device_layout(
        layout.device_layout, old_host_size, new_size_ints, stick_host_dim=stick_hd
    )
    return retiled_info


def _divide_reduction_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> None:
    """Divide the specified reduction_ranges entries of op by loop_count.

    Unlike _divide_ranges, does NOT update op.layout.size/stride — the
    output buffer shape is determined by data.ranges (non-reduction dims)
    and is unchanged by reduction-dim tiling.
    """
    data = op.data
    assert isinstance(data, Reduction)
    if not tiled_dims:
        return
    reduction_ranges = list(data.reduction_ranges)
    for i in tiled_dims:
        assert 0 <= i < len(reduction_ranges), (
            f"coarse_tile: op {op.get_name()!r} tiled reduction dim {i} out of bounds "
            f"(reduction_ranges has {len(reduction_ranges)} entries)"
        )
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


# ---------------------------------------------------------------------------
# Transformation entry point
# ---------------------------------------------------------------------------


def _apply_plan(
    ops: list[Operation],
    stamped_group_id: tuple[int, ...],
    levels: list[tuple],
    op_to_position: dict[str, int],
    plan: dict[int, CoarseTileInfo],
) -> dict[str, _RetiledBufferInfo]:
    """Apply planning's decisions: divide ranges and stamp loop_info.

    This is transformation's mutation step. All decisions (which
    dims/reduction levels are tiled, per CoarseTileInfo.tiled_dims_per_read /
    output_tiled_dims) already exist in
    `plan` (keyed by id(op) -- Operation/ComputedBuffer are unhashable, see
    plan_coarse_tile_groups) -- this function only performs the IR mutation
    _divide_ranges/_divide_reduction_ranges and the loop_info attribute
    assignment, using the plan's values instead of recomputing them.

    `stamped_group_id` is the caller's own group_id (with group_idx_offset
    and trailing per-level zeros already applied) -- it is NOT the same
    value plan_coarse_tile_groups used internally to compute each
    CoarseTileInfo.loop_group_id (that numbering starts at 0 and has no
    offset). This function overwrites loop_group_id with the caller's real
    value via dataclasses.replace before stamping, so the offset is never
    lost. Every other field of `info` is planning's decision, unchanged.
    """
    if not ops:
        return {}

    _validate_contiguous(ops, op_to_position, stamped_group_id)

    retiled_infos: dict[str, _RetiledBufferInfo] = {}
    for op in ops:
        if not isinstance(op, ComputedBuffer):
            continue
        info = plan.get(id(op))
        if info is None:
            continue

        for level_idx, (_, count) in enumerate(levels):
            opos_list = info.loop_tiled_dims[level_idx]
            rpos_list = info.loop_tiled_reduction_dims[level_idx]
            retiled_info = _divide_ranges(op, count, opos_list)
            if retiled_info is not None:
                name = op.get_name()
                prior = retiled_infos.get(name)
                retiled_infos[name] = (
                    _RetiledBufferInfo(prior.old_stride, retiled_info.new_stride)
                    if prior is not None
                    else retiled_info
                )
            if isinstance(op.data, Reduction):
                _divide_reduction_ranges(op, count, rpos_list)

        op.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
            info, loop_group_id=stamped_group_id
        )

        logger.debug(
            "coarse_tile: applied plan for %s loop_group_id=%s",
            op.get_operation_name(),
            stamped_group_id,
        )

    return retiled_infos


def _seed_closure_pre_stamp(seed_name: str, group_ops: list[Operation]) -> set[str]:
    """Pre-stamp equivalent of _seed_closure, over a plain group_ops list.

    Not transitive — see _seed_closure's docstring for why. Used by
    _replace_constant_fill_predecessors, which runs before _apply_plan (so
    ops in group_ops have no loop_info yet, and the outer-loop-group
    filtering _seed_closure does via stamped loop_info is unnecessary —
    group_ops is already scoped to the group).
    """
    return {
        o.get_name()
        for o in group_ops
        if isinstance(o, ComputedBuffer) and seed_name in _op_reads(o)
    }


def _replace_constant_fill_predecessors(
    group_ops: list[Operation],
    levels: list[tuple],
    operations: list[Operation],
    group_id: tuple[int, ...],
) -> dict[str, str]:
    """Create tile-sized constant-fill buffers for full-size fills feeding the group.

    full.default / zeros_like / zeros ops are often created outside a
    spyre_hint() scope (so they carry no dim_hints) but feed tiled ops inside
    the scope.  Rather than absorbing the full-size fill into the tiling loop
    (which causes DDL slice-size mismatches), we:

      1. Create a new tile-sized SpyreConstantFallback + Pointwise fill op
         with the same constant value but with the hinted dimension already
         divided by split_count.
      2. Insert the new tile-sized fill immediately before the tiling group in
         operations, stamped with loop_info (empty loop_tiled_dims) so
         build_loop_scheduler_nodes places it inside the loop group.

    The tile-sized fill is a loop-invariant constant: its value is identical
    across all iterations, so creating it once and reading it every iteration
    is semantically equivalent to slicing a per-iteration fill.

    Returns a name_map {old_fill_name: new_tile_fill_name} for the caller to
    apply via _apply_fill_name_swap after _apply_plan has run (so that
    replace_computed_buffer_body preserves the loop_info already stamped on
    each group op).
    """
    from torch._inductor.dependencies import MemoryDep

    # Build a reference dim_hints list from the first op in the group that has them.
    ref_dim_hints: list[DimHint] = []
    for op in group_ops:
        hints = getattr(op, "dim_hints", [])
        if hints:
            ref_dim_hints = hints
            break

    if not ref_dim_hints:
        return {}

    # Collect the set of buffer names already in the group so we don't confuse
    # intra-group data-flow edges with inter-group constant-fill edges.
    group_names: set[str] = {
        op.get_name() for op in group_ops if isinstance(op, ComputedBuffer)
    }

    # Find the insert position: just before the first op of the group.
    first_group_idx = next(
        (i for i, op in enumerate(operations) if op is group_ops[0]), None
    )
    if first_group_idx is None:
        return {}

    # name_map collects old_fill_name → new_tile_fill_name for the NameSwapHandler.
    name_map: dict[str, str] = {}

    # Track already-replaced fills so we don't create duplicates when the same
    # fill feeds multiple ops in the group.
    replaced: set[str] = set()

    for op in group_ops:
        if not isinstance(op, ComputedBuffer):
            continue
        try:
            rw = op.get_read_writes()
        except Exception:
            continue
        for dep in rw.reads:
            if not isinstance(dep, MemoryDep):
                continue
            old_name = dep.name
            if old_name in group_names or old_name in replaced:
                continue
            buf = V.graph.get_buffer(old_name)
            if not isinstance(buf, ComputedBuffer):
                continue
            if not _is_constant_fill(buf):
                continue

            # A constant fill whose closure (every op in the group that
            # transitively reads it, directly or through other closure
            # members) is entirely carry-shaped is not a hoistable broadcast
            # constant — it is an online-softmax-style recurrence's pre-loop
            # seed (e.g. `M`, read directly by both
            # `max_running = maximum(M, block_max)` and
            # `correction = exp(M - max_running)`), and every op in the
            # closure must keep reading it directly so plan_coarse_tile_groups's
            # _seed_buffer_for_carry check can find it and raise Unsupported.
            # Skip the tile-sized-copy rewrite for this (old_name, op) pair.
            # Checked via dim_hints directly (not stamped loop_info, which
            # does not exist yet at this point in the pass pipeline — this
            # function runs before _apply_plan).  A closure of size > 1 is
            # not itself a disqualifying signal (see _seed_buffer_for_carry)
            # — only "is every closure member carry-shaped" matters here.
            closure = _seed_closure_pre_stamp(old_name, group_ops)
            if closure and all(
                _is_loop_invariant_at_reduction_levels(
                    V.graph.name_to_buffer[name], group_ops, levels
                )
                for name in closure
            ):
                continue

            # Read the constant value from the SpyreConstantFallback scalar
            # that is the fill's only input.
            fill_rw = buf.get_read_writes()
            scalar_dep = next(
                (d for d in fill_rw.reads if isinstance(d, MemoryDep)), None
            )
            if scalar_dep is None:
                continue
            scalar_buf = V.graph.get_buffer(scalar_dep.name)
            if not isinstance(scalar_buf, SpyreConstantFallback):
                continue
            const_value = scalar_buf.constant_args[0]

            # Compute the tile-sized shape using the consumer op's authoritative
            # loop_var→ranges_pos mapping.  Size-based matching (_constant_fill_
            # ranges_pos) is unreliable when two named dims share the same value
            # (e.g. Lq=256 and Lk=256 in flash attention).  Instead, for each
            # hint we ask: which output-ranges position of the consumer op does
            # this hint tile?  If the fill has a dimension at that same position
            # with the expected size, divide it; otherwise skip (the fill doesn't
            # have that dimension and should not be divided on it).
            old_size = [int(r) for r in buf.data.ranges]
            tile_size = list(old_size)
            consumer_out = op_out_coords(op)
            for h in getattr(op, "dim_hints", None) or []:
                if h.loop_var is None or h.is_reduction or h.split_count <= 1:
                    continue
                consumer_pos = _loop_var_to_ranges_pos(consumer_out, h.loop_var)
                if consumer_pos is None:
                    continue
                if consumer_pos >= len(old_size):
                    continue
                if old_size[consumer_pos] != int(op.data.ranges[consumer_pos]):
                    # Fill dim size doesn't match consumer's full range; skip.
                    continue
                if tile_size[consumer_pos] % int(h.split_count) != 0:
                    logger.warning(
                        "coarse_tile: constant-fill %s dim %d size %d not "
                        "divisible by split_count %d; skipping replacement",
                        old_name,
                        consumer_pos,
                        tile_size[consumer_pos],
                        int(h.split_count),
                    )
                    tile_size = None  # type: ignore[assignment]
                    break
                tile_size[consumer_pos] = tile_size[consumer_pos] // int(h.split_count)

            if tile_size is None:
                continue

            dtype = buf.get_dtype()
            device = buf.get_device()

            # Create the tile-sized scalar + fill.
            new_scalar = SpyreConstantFallback(
                torch.ops.spyre.constant.default, float(const_value), dtype, device
            )
            scalar_stl = SpyreTensorLayout([], dtype)
            new_scalar.layout = FixedTiledLayout(device, dtype, [], [], scalar_stl)
            scalar_loader = TensorBox.create(new_scalar).make_loader()

            tile_ranges = [sympy.Integer(s) for s in tile_size]
            fill_data = Pointwise(
                device=device,
                dtype=dtype,
                inner_fn=lambda index, _loader=scalar_loader: _loader([]),
                ranges=tile_ranges,
            )
            tile_strides = [
                sympy.Integer(int(s))
                for s in FlexibleLayout.contiguous_strides(tile_size)
            ]
            fill_name = V.graph.qualify_name(f"ct_fill_{old_name}")
            # Logically a FlexibleLayout (tile_strides above is just the
            # contiguous row-major default; nothing downstream depends on
            # this specific stride value), but it must be FixedLayout: this
            # buffer is read by its consumer(s) before stickification runs,
            # and split_multi_ops traces consumer inner_fns via
            # make_loader()/make_indexer(), which asserts
            # FlexibleLayout.allow_indexing (torch/_inductor/ir.py). A
            # FlexibleLayout here makes that assertion fire, silently
            # dropping the trace and letting any scalar constant survive
            # ungrouped into codegen, where SpyreKernel.store() rejects it.
            # (See the identical hazard and failure mode documented on
            # _allocate_full_buffer's FixedLayout above.)
            fill_buf = ComputedBuffer(
                name=fill_name,
                layout=FixedLayout(device, dtype, list(tile_ranges), tile_strides),
                data=fill_data,
            )
            fill_buf.origins = buf.origins
            fill_buf.operation_name = fill_name
            # buf (the old, untiled fill) has no dim_hints by construction --
            # that's the whole reason this function exists (see docstring).
            # Copy dim_hints from op (the tiled consumer) instead, so
            # propagate_named_dims doesn't fall back to _untracked_* for this
            # buffer.
            copy_op_metadata(op, fill_buf)
            # Stamp loop_info so the fill is placed inside the loop group by
            # build_loop_scheduler_nodes, with empty loop_tiled_dims (loop-
            # invariant: executed once per loop body but no range is divided).
            # Use the same nested_group_id that plan_coarse_tile_groups assigns
            # to broadcast ops: group_id + (0,) * (len(levels) - 1).  loop_count
            # length must equal loop_group_id length (enforced by _loop_count
            # assertion).
            counts = [count for _, count in levels]
            nested_group_id = group_id + (0,) * (len(levels) - 1)
            fill_buf.loop_info = CoarseTileInfo(  # type: ignore[attr-defined]
                loop_group_id=nested_group_id,
                loop_count=counts,
                loop_tiled_dims=[[] for _ in levels],
                loop_tiled_reduction_dims=[[] for _ in levels],
            )
            V.graph.name_to_buffer[fill_name] = fill_buf

            # Splice new_scalar and fill_buf into operations just before the group.
            # new_scalar was appended to operations by register_operation() inside
            # SpyreConstantFallback.__init__; move it to the insert position.
            operations.remove(new_scalar)
            operations.insert(first_group_idx, new_scalar)
            first_group_idx += 1
            operations.insert(first_group_idx, fill_buf)
            first_group_idx += 1

            name_map[old_name] = fill_name
            replaced.add(old_name)
            logger.debug(
                "coarse_tile: created tile-sized fill %s (shape %s) replacing %s (shape %s)",
                fill_name,
                tile_size,
                old_name,
                old_size,
            )

    return name_map


def _is_constant_fill(op: ComputedBuffer) -> bool:
    """True if op is a Pointwise whose only reads come from SpyreConstantFallback.

    full.default / zeros_like / zeros lower to a SpyreConstantFallback scalar
    broadcast through a thin Pointwise wrapper.  These ops are position-
    independent, so shrinking their per-tile range to match the tiled group
    is semantically equivalent to slicing a full-sized fill.
    """
    if not isinstance(op.data, Pointwise):
        return False
    try:
        rw = op.get_read_writes()
    except Exception:
        return False
    from torch._inductor.dependencies import MemoryDep

    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if not reads:
        return False
    return all(
        isinstance(V.graph.get_buffer(d.name), SpyreConstantFallback) for d in reads
    )


def _apply_fill_name_swap(
    group_ops: list[Operation],
    name_map: dict[str, str],
    operations: list[Operation],
) -> None:
    """Patch group ops to read tile-sized fills instead of the original full-size ones.

    Must be called AFTER _apply_plan so that replace_computed_buffer_body
    copies the already-stamped loop_info onto the reconstructed ComputedBuffer.
    """
    if not name_map:
        return

    from torch._inductor.dependencies import MemoryDep
    from ..insert_restickify import NameSwapHandler
    from ..pass_utils import replace_computed_buffer_body

    for op in group_ops:
        if not isinstance(op, ComputedBuffer):
            continue
        reads = set()
        try:
            reads = {
                d.name for d in op.get_read_writes().reads if isinstance(d, MemoryDep)
            }
        except Exception:
            continue
        if not reads & set(name_map):
            continue

        orig_inner = op.data.inner_fn

        def new_inner_fn(*args, _map=name_map, _orig=orig_inner):
            with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
                return _orig(*args)

        object.__setattr__(op.data, "inner_fn", new_inner_fn)
        new_op = replace_computed_buffer_body(op, op.data, operations)
        V.graph.name_to_buffer[new_op.get_name()] = new_op


def coarse_tile(
    graph: GraphLowering,
    groups: list[tuple],
    group_idx_offset: int = 0,
) -> None:
    """Plan then transform: stamp loop_group_id / loop_count and scale ranges.

    Parameters
    ----------
    operations:
        The full ordered list of IR operations (as seen by
        CustomPreSchedulingPasses).  Modified in-place when the
        transformation phase inserts new buffer/copy ops.
    groups:
        Sequence of ``(ops, levels)`` tuples produced by
        ``hints_to_coarse_tile_groups``.  ``levels`` is a list of
        ``(hint_id, count)`` pairs, outermost first.
    group_idx_offset:
        Starting index for group IDs assigned to the first group.  Use this
        when making a second ``coarse_tile`` call on the same graph so that
        the new group IDs do not collide with IDs already stamped by an
        earlier call (e.g. hint-driven groups stamped pre-stickification).
    """
    operations = graph.operations

    # Planning: decide every op's tiling attributes with zero mutation.
    # If any op needs carry propagation or requests disabled reduction
    # tiling, this raises Unsupported before any transformation runs.
    # plan_coarse_tile_groups numbers groups starting at 0 internally, but
    # only uses that numbering to build each op's nested loop_group_id
    # shape (group_id + trailing zeros) -- it never compares group_id
    # values across calls, so an un-offset numbering here is safe. The
    # *real* group_id stamped onto ops (with group_idx_offset applied) is
    # recomputed below in the transformation loop and overwrites
    # info.loop_group_id via _apply_plan before it's ever read back out.
    plan = plan_coarse_tile_groups(operations, groups)

    # Transformation: apply the plan. Only reached if planning didn't raise.
    op_to_position: dict[str, int] = {
        op.get_operation_name(): i for i, op in enumerate(operations)
    }
    retiled_infos_by_group: list[
        tuple[tuple[int, ...], list[Operation], dict[str, _RetiledBufferInfo]]
    ] = []
    for group_idx, (group_ops, levels) in enumerate(groups, start=group_idx_offset):
        group_id: tuple[int, ...] = (group_idx,)
        name_map = _replace_constant_fill_predecessors(
            group_ops, levels, operations, group_id
        )
        op_to_position = {op.get_operation_name(): i for i, op in enumerate(operations)}
        stamped_group_id = group_id + (0,) * (len(levels) - 1)
        retiled_infos = _apply_plan(
            group_ops, stamped_group_id, levels, op_to_position, plan
        )
        _apply_fill_name_swap(group_ops, name_map, operations)
        retiled_infos_by_group.append((stamped_group_id, group_ops, retiled_infos))

    insert_tiling_propagation(operations, groups)
    _zero_fixed_tile_advance_exprs(operations)

    for group_id, group_ops, retiled_infos in retiled_infos_by_group:
        _patch_retiled_load_indexes(group_id, group_ops, retiled_infos, operations)


def _zero_fixed_tile_advance_exprs(operations: list[Operation]) -> None:
    """Drop tiled_dims_per_read / output_tiled_dims entries for fixed tiles.

    A per-tile-fixed buffer (flagged by insert_tiling_propagation, above, via
    either the pre-stickify _pending_per_tile_fixed or an already-committed
    post-stickify FixedTiledLayout.per_tile_fixed) is reused in place every
    tile iteration -- it never actually advances, regardless of what its
    dependency's index arithmetic would otherwise compute. Drop that
    dependency's tiled-dim entries entirely (an empty per-level list has the
    same effect as a zero advance expr did before this plan: no dims to
    substitute means no advance), both for the op's own output (if its own
    buffer is fixed) and for any op reading a fixed buffer as an input.

    This is the only place tiled_dims_per_read/output_tiled_dims reads
    per_tile_fixed/_pending_per_tile_fixed -- a deliberate one-way bridge
    kept isolated here so it can be deleted once a later stage derives
    per_tile_fixed from a zero advance expr instead of the reverse.
    """
    fixed_names = {
        op.get_name()
        for op in operations
        if isinstance(op, ComputedBuffer)
        and (
            getattr(op, "_pending_per_tile_fixed", False)
            or getattr(op.layout, "per_tile_fixed", False)
        )
    }
    if not fixed_names:
        return

    for op in operations:
        loop_info = getattr(op, "loop_info", None)
        if loop_info is None:
            continue
        if op.get_name() in fixed_names:
            loop_info.output_tiled_dims = []
        if not loop_info.tiled_dims_per_read:
            continue
        reads = [
            dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)
        ]
        for i, dep in enumerate(reads):
            if dep.name in fixed_names:
                loop_info.tiled_dims_per_read[i] = []


# ---------------------------------------------------------------------------
# Buffer propagation pass
# ---------------------------------------------------------------------------


def insert_tiling_propagation(
    operations: list[Operation],
    groups: list[tuple],
) -> None:
    """Insert full-sized buffers and copy/mutation ops for tiled ops.

    Handles Pointwise and Reduction ComputedBuffers.  For Reductions, tiled
    dims that fall in the reduction_ranges index range raise RuntimeError.

    For each eligible ComputedBuffer in a tiling group, if its result is
    consumed by any operation outside the loop (different loop_group_id or
    absent) or is a graph output, this pass ensures the outside consumer sees
    the complete result: allocate a full-sized buffer, then insert a copy op
    (same loop_group_id, same loop_tiled_dims) that writes each tile into the
    correct slice of the full buffer.  Patch outside consumers to read the
    full buffer.  This applies uniformly regardless of whether the tiled
    op's output is also consumed inside the loop.

    The existing tiled_symbols / affine.apply machinery in SpyreKernel and
    bundle.py handles the per-iteration address offset for both the tiled
    op and the inserted copy op.
    """
    for group_ops, _ in groups:
        for idx, op in enumerate(group_ops):
            if not isinstance(op, ComputedBuffer):
                continue
            if not isinstance(op.data, (Pointwise, Reduction)):
                continue
            _propagate_tiled_op(op, operations)
            # _propagate_tiled_op (and the copy-op insertion it may
            # delegate to) can replace op with a new ComputedBuffer object
            # spliced into `operations` under the same name (see
            # replace_computed_buffer_body).  group_ops is a separate list
            # snapshotted before this loop runs, so it must be kept in sync
            # here — otherwise a later pass reading group_ops (e.g.
            # _patch_retiled_load_indexes) sees the stale pre-rewrite object
            # and clobbers the rewrite when it reconstructs from it.  Look
            # up the current object in `operations` itself (already the
            # authoritative post-replacement list) rather than V.graph --
            # this pass runs from CustomPreSchedulingPasses, and unit tests
            # that call coarse_tile() directly never install a real V.graph.
            current = next(
                (
                    o
                    for o in operations
                    if isinstance(o, ComputedBuffer) and o.get_name() == op.get_name()
                ),
                None,
            )
            if current is not None and current is not op:
                group_ops[idx] = current


def _validate_reduction_tiling(op: ComputedBuffer) -> None:
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
    """
    data = op.data
    assert isinstance(data, Reduction)
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None:
        return

    tiled_dims = loop_info.loop_tiled_dims
    tiled_rdims = getattr(loop_info, "loop_tiled_reduction_dims", [])

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
                "is not yet implemented — Stage 2)."
            )
        if len(red_dims) > 1:
            raise Unsupported(
                f"coarse_tile: op {op.get_name()!r} level {i} tiles multiple "
                f"reduction dims {red_dims} (tiling more than one reduction "
                "dim per level is not yet implemented — Stage 2)."
            )


def _propagate_tiled_op(
    op: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Handle buffer propagation for a single tiled Pointwise or Reduction op."""
    loop_info = getattr(op, "loop_info", None)

    # A tiled op — Pointwise or Reduction, loop-internal or not — may read a
    # buffer produced outside its own outer loop group directly (e.g. a
    # SpyreEmptyFallback accumulator, or the output of an earlier, different
    # coarse-tile group's own copy op).  That buffer's layout was fixed by a
    # different loop group's (or no loop group's) constraints, sized to its
    # own full extent, while this op's own candidates are sized to its tile
    # — the two can never be stick-compatible.  Insert a tile-sized read copy
    # for each such input before doing anything else (mirrors the write-side
    # _insert_copy_op fix, but on the read side).  This must run before the
    # Reduction/has_tiled_reduction branch below, since
    # _propagate_tiled_reduction_op never touches op's own reads.
    full_deps = _full_buffer_read_deps(op)
    if full_deps:
        op = _insert_read_copy_ops(op, full_deps, operations)

    if isinstance(op.data, Reduction):
        _validate_reduction_tiling(op)
        has_tiled_reduction = loop_info is not None and any(
            dims for dims in getattr(loop_info, "loop_tiled_reduction_dims", [])
        )
        if has_tiled_reduction:
            _propagate_tiled_reduction_op(op, operations)
            return

    if loop_info is None:
        return
    loop_group_id = loop_info.loop_group_id

    buf_name = op.get_name()
    outside_consumers, is_graph_output = _find_outside_consumers(
        buf_name, loop_group_id, operations
    )

    # If no dims were tiled (loop_tiled_dims all empty), the op is loop-invariant.
    if all(not dims for dims in loop_info.loop_tiled_dims):
        return

    if not outside_consumers and not is_graph_output:
        # Loop-internal: the buffer is a per-tile scratch region reused every
        # iteration.  Mark it so the unroller does not advance its base address.
        if isinstance(op.layout, FixedTiledLayout):
            # Post-stickify (span-overflow path): layout is already FixedTiledLayout.
            op.layout.per_tile_fixed = True
        else:
            # Pre-stickify (hint path): layout is FixedLayout.  Defer to
            # finalize_layouts which sees the committed FixedTiledLayout and can
            # set the flag then.  MutationLayoutSHOULDREMOVE only appears in
            # the post-stickify span-overflow path and is handled by the branch
            # above (it would not reach this else).
            op._pending_per_tile_fixed = True  # type: ignore[attr-defined]
        return

    # Reconstruct the original (pre-division) ranges.
    full_ranges = _compute_full_ranges(op)

    # Insert the full buffer before the first op in the same outermost loop group
    # so it doesn't split the group's contiguous run in the operations list.
    outer_key = loop_group_id[0]
    group_start_idx = next(
        i
        for i, o in enumerate(operations)
        if isinstance(o, ComputedBuffer)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
    )
    full_buf = _allocate_full_buffer(op, full_ranges, operations, group_start_idx)

    # Capture before _insert_copy_op overwrites op.layout.
    old_stride = tuple(op.layout.stride)

    # Every cross-loop-group write always takes the copy-op path: the real
    # compute op keeps its own natural, input-derived, tile-sized layout,
    # and a separate copy op takes MutationLayoutSHOULDREMOVE(full_buf).
    # See docs/source/compiler/coarse_tiling_loops.md's "Treatment by
    # consumer topology" section for why the direct-mutation alternative
    # (formerly "Case 2"/"Case 3") is unsafe post-stickify: it derives
    # full_buf's layout from this op's own committed output layout without
    # reconciling the op's *input* layouts, and there is no compatibility
    # check analogous to finalize_layouts's is_elided/is_carry_into_accum
    # guard on that path.
    _insert_copy_op(op, full_buf, operations)
    # The tiled op's own buffer is always loop-internal scratch here: it is
    # fully drained by the copy op inserted above before the next iteration
    # overwrites it.
    if isinstance(op.layout, FixedTiledLayout):
        op.layout.per_tile_fixed = True
    else:
        op._pending_per_tile_fixed = True  # type: ignore[attr-defined]

    # Patch outside consumers and graph outputs to read full_buf.
    full_name = full_buf.get_name()
    retile_info = _RetiledBufferInfo(old_stride, tuple(full_buf.layout.stride))
    _patch_consumers(outside_consumers, buf_name, full_name, operations, retile_info)
    if is_graph_output:
        _patch_graph_outputs(buf_name, full_buf)

    logger.debug("coarse_tile: propagated %s → %s (copy)", buf_name, full_name)


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
        li = getattr(op, "loop_info", None)
        if li is None or li.loop_group_id[0] != outer_key:
            consumers.append(op)

    is_graph_output = buf_name in _graph_output_names()
    return consumers, is_graph_output


def _full_buffer_read_deps(op: ComputedBuffer) -> list[MemoryDep]:
    """Return op's MemoryDep reads whose producer is outside op's own loop group.

    A loop-internal op (own tile-sized layout) that reads a buffer produced
    outside its own outer loop group can never be made stick-compatible
    with it under AllSameNode: that producer's layout was fixed by a
    different loop group's (or no loop group's) constraints, sized to its
    own full extent, while op's own candidates are sized to its tile.

    This only applies to producers that go through coarse_tile's own
    candidate-layout machinery: a SpyreEmptyFallback buffer (coarse_tile's
    own full-extent accumulator, given a single generic_layout candidate and
    AnyInNode -- see _allocate_full_buffer -- that the optimizer can never
    relayout) or a ComputedBuffer with no loop_info (untiled, full-extent) or
    a different loop_group_id[0] (divided by a different group's loop_count
    -- e.g. the output of an earlier, different coarse-tile group's own copy
    op). Ordinary graph-input InputBuffers (and other non-ComputedBuffer,
    non-SpyreEmptyFallback producers such as constants/extern nodes) are
    deliberately excluded: those get exactly one candidate layout too, but
    _get_prop_args/AllSameNode's normal restickify path already reconciles a
    full-extent input against a tile-sized read (see insert_restickify.py),
    so flagging them here would insert a spurious copy for the common case
    of a plain graph input read directly by a tiled op. See
    _insert_read_copy_ops and _find_outside_consumers (same outer-key
    comparison, mirrored here on the read side).
    """
    from ..ir import SpyreEmptyFallback  # deferred: avoids circular import

    loop_info = getattr(op, "loop_info", None)
    if loop_info is None:
        return []
    outer_key = loop_info.loop_group_id[0]

    reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    result = []
    for d in reads:
        buf = V.graph.get_buffer(d.name)
        if isinstance(buf, SpyreEmptyFallback):
            result.append(d)
        elif isinstance(buf, ComputedBuffer):
            producer_li = getattr(buf, "loop_info", None)
            if producer_li is None or producer_li.loop_group_id[0] != outer_key:
                result.append(d)
    return result


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


def _compute_full_ranges(op: ComputedBuffer) -> list[Expr]:
    """Compute the original (pre-division) iteration ranges of op.

    op.data.ranges holds the already-divided ranges.  Reconstruct the full
    ranges by multiplying each tiled dimension back by its loop_count.
    """
    full_ranges = list(op.data.ranges)
    loop_count: list[Expr] = op.loop_info.loop_count
    loop_tiled_dims: list[list[int]] = op.loop_info.loop_tiled_dims
    for count, dims in zip(loop_count, loop_tiled_dims):
        for d in dims:
            if 0 <= d < len(full_ranges):
                full_ranges[d] = sympy.simplify(full_ranges[d] * count)
    return full_ranges


def _allocate_full_buffer(
    tiled_op: ComputedBuffer,
    full_ranges: list[Expr],
    operations: list[Operation],
    insert_at_idx: int,
) -> ComputedBuffer:
    """Allocate a full-sized HBM buffer for the tiled op's original shape.

    Creates a spyre.empty FX node, lowers it via V.graph.run_node(), assigns
    a layout matching tiled_op's layout type (FixedLayout pre-stickify,
    FixedTiledLayout post-stickify), splices it into operations at
    insert_at_idx, and returns the new ComputedBuffer.
    """
    from ..ir import SpyreEmptyFallback  # deferred: avoids circular import

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
    orig_layout = tiled_op.layout
    # Recompute strides for the full size (contiguous row-major).
    strides: list[Expr] = []
    stride: Expr = sympy.Integer(1)
    for s in reversed(full_ranges):
        strides.insert(0, stride)
        stride = stride * s

    if isinstance(orig_layout, FixedTiledLayout):
        # Post-stickify path (span-overflow groups): stickification has already
        # run, so we must assign a FixedTiledLayout now.  Derive the full
        # buffer's device layout by scaling the per-tile device layout up to
        # the full host size using _resize_device_layout.
        full_size_ints = [int(s) for s in full_ranges]
        tile_size_ints = [int(s) for s in orig_layout.size]
        # Authoritative stick host dim from coordinate identity (issue #3116);
        # None falls back to size-based inference inside _resize_device_layout.
        stick_hd = _stick_host_dim(tiled_op, orig_layout.device_layout)
        try:
            device_layout = _resize_device_layout(
                orig_layout.device_layout,
                tile_size_ints,
                full_size_ints,
                stick_host_dim=stick_hd,
            )
        except RuntimeError:
            # Non-standard device layout (e.g. post-restickify HBM strides that
            # don't correspond to contiguous host strides).  Fall back to a
            # default row-major allocation, preserving element_arrangement.
            logger.debug(
                "_allocate_full_buffer: _resize_device_layout could not classify "
                "%r (tile_size=%s full_size=%s); using row-major fallback",
                orig_layout.device_layout,
                tile_size_ints,
                full_size_ints,
            )
            ndim_full = len(full_size_ints)
            full_strides_ints = [int(s) for s in strides]
            device_layout = SpyreTensorLayout(
                full_size_ints,
                full_strides_ints,
                dtype,
                list(range(ndim_full)),
                orig_layout.device_layout.element_arrangement,
            )
        layout: FixedTiledLayout | FixedLayout = FixedTiledLayout(
            device,
            dtype,
            list(full_ranges),
            strides,
            device_layout,
        )
    else:
        # Pre-stickify path (hint-driven groups): stickification has not yet
        # run, so assign a plain FixedLayout.  Stickification will propagate
        # SpyreTensorLayout to this buffer via the ExternKernel->generic_layout
        # path in propagate_spyre_tensor_layouts.
        #
        # This is logically a FlexibleLayout (the stride values below are
        # never read by stickification -- generic_layout builds
        # SpyreTensorLayout from .size alone), but it cannot be written that
        # way: full_buf gets read (via name-swapped consumer inner_fns,
        # e.g. _insert_read_copy_ops) before stickification runs, and
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

    # Splice into operations at the correct position.
    operations.remove(full_buf)
    operations.insert(insert_at_idx, full_buf)

    return full_buf


# ---------------------------------------------------------------------------
# Case 1: copy op insertion
# ---------------------------------------------------------------------------


def _insert_copy_op(
    tiled_op: ComputedBuffer,
    full_buf: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Insert a copy op after tiled_op that writes each tile into full_buf.

    The copy op carries the same loop metadata as tiled_op (so it executes
    inside the same loop body) but its own freshly-derived
    tiled_dims_per_read/output_tiled_dims, since its reads/write don't
    correspond positionally to tiled_op's. Its layout is
    MutationLayoutSHOULDREMOVE pointing at full_buf so store_output writes
    into full_buf; loop_tiled_dims being set makes SpyreKernel stamp
    tiled_symbols on the OpSpec and bundle.mlir emit affine.apply for the
    per-iteration output address.
    """
    copy_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=tiled_op.make_loader(),
        ranges=list(tiled_op.data.ranges),
    )

    copy_name = V.graph.qualify_name(f"coarse_tile_copy_{tiled_op.get_name()}")
    copy_buf = ComputedBuffer(
        name=copy_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf))),
        data=copy_data,
    )
    copy_buf.origins = tiled_op.origins
    copy_buf.operation_name = copy_name

    # Fresh per-level tiled-dim decisions from copy_buf's own reads/write
    # (positionally different from tiled_op's). copy_buf's ranges are
    # identical to tiled_op's post-division ranges, so each tiled dim's
    # level-0..N extent is simply 1 -- the copy op is not itself re-divided,
    # it just re-reads tiled_op's already-divided buffer once per tile.
    tiled_op_info = tiled_op.loop_info  # type: ignore[attr-defined]
    per_level_extents = [
        {d: sympy.Integer(1) for d in level_dims}
        for level_dims in tiled_op_info.loop_tiled_dims
    ]
    copy_reads = [
        dep for dep in copy_buf.get_read_writes().reads if isinstance(dep, MemoryDep)
    ]
    copy_writes = [
        dep for dep in copy_buf.get_read_writes().writes if isinstance(dep, MemoryDep)
    ]
    tiled_dims_per_read = [
        _tiled_dims_for_dep(dep, per_level_extents) for dep in copy_reads
    ]
    output_tiled_dims = (
        _tiled_dims_for_dep(copy_writes[0], per_level_extents) if copy_writes else []
    )
    copy_buf.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
        tiled_op_info,
        tiled_dims_per_read=tiled_dims_per_read,
        output_tiled_dims=output_tiled_dims,
    )

    V.graph.name_to_buffer[copy_name] = copy_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, copy_buf)


class _NameSwapHandler(WrapperHandler):
    """Redirect ops.load(name, index) calls for names present in name_map.

    See NameSwapHandler in insert_restickify.py — same pattern (CLAUDE.md
    "Compiler Pass Conventions": wrap inner_fn via a WrapperHandler, never
    reconstruct it from index expressions). Duplicated locally rather than
    imported to avoid a coarse_tile <-> insert_restickify import-order
    dependency; the two run at different, non-adjacent pipeline stages.
    """

    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def _insert_read_copy_ops(
    tiled_op: ComputedBuffer,
    full_deps: list[MemoryDep],
    operations: list[Operation],
) -> ComputedBuffer:
    """Insert, before tiled_op, one tile-sized copy op per full-size real input.

    tiled_op reads one or more full-size cross-loop-group buffers directly
    (see _full_buffer_read_deps).  Those buffers get exactly one candidate
    layout (sized to the full buffer), while tiled_op's own candidates are
    sized to its tile — the two can never be stick-compatible under
    AllSameNode.  Mirroring _insert_copy_op's write-side fix: for each
    such input, insert a copy op that reads the full buffer's current tile
    slice (same index expression tiled_op already uses, same loop_info so
    the per-iteration base address advances identically) and writes it into
    a fresh tile-sized buffer.  tiled_op's own inner_fn is then patched
    (WrapperHandler, not reconstructed — see _NameSwapHandler) to read the
    copy instead of the full buffer.

    The copy's own ranges/index must match dep (dep.var_names/dep.size), not
    tiled_op.data.ranges: for a Reduction, the read spans output dims plus
    the reduction dim, so dep's iteration space has more vars than the op's
    own output-shaped ranges.  The copy's layout reuses full_buf's own
    per-var strides (extracted from dep.index, which is affine in
    dep.var_names) rather than fresh contiguous strides, sized down to
    dep.size — so tiled_op's unmodified read index (dep.index, computed
    against those same strides) still resolves correctly once _NameSwapHandler
    retargets the load at the copy buffer instead of full_buf.

    Mirrors _allocate_full_buffer's isinstance(orig_layout, FixedTiledLayout)
    branch: when full_buf already carries a FixedTiledLayout (post-stickify
    call site), the copy gets a FixedTiledLayout too, with device_layout
    resized down from full_buf's own device_layout via _resize_device_layout
    (shrink direction, mirroring _divide_ranges's use of the same helper).
    Otherwise (pre-stickify call site) the copy gets a plain FixedLayout, and
    stickification (which runs later) fills device_layout in normally.

    Returns the reconstructed ComputedBuffer that replaces tiled_op in
    operations (see replace_computed_buffer_body below) — callers must
    rebind their own reference since the original tiled_op object is stale
    after this call.
    """
    name_map: dict[str, str] = {}
    tiled_idx = operations.index(tiled_op)

    for dep in full_deps:
        full_buf = V.graph.get_buffer(dep.name)

        tile_ranges = list(dep.size)
        tile_strides = [dep.index.coeff(v) for v in dep.var_names]

        def _copy_inner_fn(idx, _dep=dep, _full_name=full_buf.get_name()):
            subs = dict(zip(_dep.var_names, idx))
            flat_index = sympy_subs(_dep.index, subs)
            return V.ops.load(_full_name, flat_index)

        # Construct under tiled_op's origins so data.origins is non-empty —
        # _single_arg_op_layout (propagate_layouts.py) unconditionally
        # dereferences next(iter(data.origins)) for ordinary (non-mutation)
        # Pointwise ops.  IRNode.origins is populated at construction time
        # from IRNode._current_origins, so it must be set via this context
        # manager rather than assigned after the fact (assigning
        # copy_buf.origins below only sets the ComputedBuffer's own origins,
        # not copy_data's).
        with IRNode.current_origins(tiled_op.origins):
            copy_data = Pointwise(
                device=tiled_op.get_device(),
                dtype=full_buf.get_dtype(),
                inner_fn=_copy_inner_fn,
                ranges=tile_ranges,
            )
        copy_name = V.graph.qualify_name(
            f"coarse_tile_read_copy_{tiled_op.get_name()}_{dep.name}"
        )

        # Mirror _allocate_full_buffer's isinstance(orig_layout, FixedTiledLayout)
        # branch: on the post-stickify call site (_maybe_coarse_tile_span_overflow),
        # full_buf already carries a FixedTiledLayout, and every sibling op at
        # this pipeline stage (span_reduction, work_distribution, LX scratchpad
        # planning) expects a copy op to carry one too. On the pre-stickify call
        # site (_maybe_coarse_tile_hints), full_buf is a plain FixedLayout and
        # stickification (which runs later) fills device_layout in normally, so
        # a plain FixedLayout on the copy is correct there.
        full_layout = full_buf.layout
        copy_layout: FixedLayout | FixedTiledLayout
        if isinstance(full_layout, FixedTiledLayout):
            full_size_ints = [int(s) for s in full_layout.size]
            tile_size_ints = [int(s) for s in tile_ranges]
            # Authoritative stick host dim from coordinate identity (issue
            # #3116); None falls back to size-based inference inside
            # _resize_device_layout.
            stick_hd = _stick_host_dim(full_buf, full_layout.device_layout)
            try:
                device_layout = _resize_device_layout(
                    full_layout.device_layout,
                    full_size_ints,
                    tile_size_ints,
                    stick_host_dim=stick_hd,
                )
            except RuntimeError:
                # Non-standard device layout (e.g. post-restickify HBM strides
                # that don't correspond to contiguous host strides).  Fall
                # back to a default row-major allocation, preserving
                # element_arrangement -- same fallback _allocate_full_buffer
                # uses for its own grow-direction resize failures.
                logger.debug(
                    "_insert_read_copy_ops: _resize_device_layout could not "
                    "classify %r (full_size=%s tile_size=%s); using "
                    "row-major fallback",
                    full_layout.device_layout,
                    full_size_ints,
                    tile_size_ints,
                )
                device_layout = SpyreTensorLayout(
                    tile_size_ints,
                    [int(s) for s in tile_strides],
                    full_buf.get_dtype(),
                    list(range(len(tile_size_ints))),
                    full_layout.device_layout.element_arrangement,
                )
            copy_layout = FixedTiledLayout(
                tiled_op.get_device(),
                full_buf.get_dtype(),
                tile_ranges,
                tile_strides,
                device_layout,
            )
        else:
            copy_layout = FixedLayout(
                tiled_op.get_device(),
                full_buf.get_dtype(),
                tile_ranges,
                tile_strides,
            )
        copy_buf = ComputedBuffer(name=copy_name, layout=copy_layout, data=copy_data)
        copy_buf.origins = tiled_op.origins
        copy_buf.operation_name = copy_name
        copy_op_metadata(tiled_op, copy_buf)
        copy_buf.loop_info = tiled_op.loop_info  # type: ignore[attr-defined]

        V.graph.name_to_buffer[copy_name] = copy_buf
        operations.insert(tiled_idx, copy_buf)
        tiled_idx += 1

        name_map[dep.name] = copy_name

    # Patch tiled_op's inner_fn once with the full name_map (wrap, not
    # reconstruct — see _NameSwapHandler docstring).  Rebuild via
    # replace_computed_buffer_body, matching every other inner_fn-rewrite
    # site in this file (_patch_consumers, _patch_retiled_load_indexes,
    # _apply_fill_name_swap): a fresh ComputedBuffer has no stale per-object
    # caches, sidestepping the need to enumerate every cache key by hand.
    from ..pass_utils import replace_computed_buffer_body

    orig_inner = tiled_op.data.inner_fn

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(_NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(tiled_op.data, "inner_fn", new_inner_fn)
    new_op = replace_computed_buffer_body(tiled_op, tiled_op.data, operations)
    V.graph.name_to_buffer[new_op.get_name()] = new_op
    return new_op


# ---------------------------------------------------------------------------
# Case: reduction-dim tiling — combine op insertion
# ---------------------------------------------------------------------------


def _insert_combine_op(
    tiled_op: ComputedBuffer,
    accum_buf: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Insert a pointwise combine op that accumulates tiled_op into accum_buf.

    The combine op reads both the partial result (tiled_op) and the current
    accumulation buffer and writes the combined value back into accum_buf via
    MutationLayoutSHOULDREMOVE.  It carries the same loop_info as tiled_op
    so the scheduler places it inside the same CountedLoopSchedulerNode.
    """
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
            # TODO: add vops.logical_or to SpyreOpFuncs before enabling
            # hardware-level 'any' support — it is currently absent.
            return vops.logical_or(accum, partial)
        raise RuntimeError(
            f"coarse_tile: _insert_combine_op: unsupported reduction_type "
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
    combine_buf.loop_info = tiled_op.loop_info  # type: ignore[attr-defined]
    V.graph.name_to_buffer[combine_name] = combine_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, combine_buf)


def _insert_reduction_copy_op(
    tiled_op: ComputedBuffer,
    accum_tile: ComputedBuffer,
    accum_full: ComputedBuffer,
    outer_loop_info: "CoarseTileInfo",
    operations: list[Operation],
    insert_after: ComputedBuffer | None = None,
    force_live: bool = False,
) -> None:
    """Insert a copy op that writes accum_tile → accum_full at the outer loop level.

    Reads accum_tile (per_tile_fixed=True, never advances) and writes into
    accum_full via MutationLayoutSHOULDREMOVE.  Carries outer_loop_info so
    the unroller advances accum_full per outer output-dim tile.

    By default inserts immediately after tiled_op (or its combine op, if
    any) — correct when nothing else in the inner loop group depends on
    tiled_op.  insert_after/force_live exist for a case this function's sole
    current caller (_propagate_tiled_reduction_op) never needs and always
    leaves at their defaults: reduction-dim tiling requiring cross-tile carry
    propagation is rejected with Unsupported at planning time
    (plan_coarse_tile_groups, via _seed_buffer_for_carry), so no caller ever
    needs to place the copy after anything but tiled_op/its combine op, or to
    force it live.
    """
    copy_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=accum_tile.make_loader(),
        ranges=list(tiled_op.data.ranges),
    )
    copy_name = V.graph.qualify_name(f"coarse_tile_reduce_copy_{tiled_op.get_name()}")
    copy_buf = ComputedBuffer(
        name=copy_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(accum_full))),
        data=copy_data,
    )
    copy_buf.origins = tiled_op.origins
    copy_buf.operation_name = copy_name
    copy_buf.loop_info = outer_loop_info  # type: ignore[attr-defined]
    if force_live:
        copy_buf._coarse_tile_force_live = True  # type: ignore[attr-defined]
    V.graph.name_to_buffer[copy_name] = copy_buf

    if insert_after is not None:
        insert_idx = operations.index(insert_after) + 1
    else:
        combine_name = V.graph.qualify_name(
            f"coarse_tile_combine_{tiled_op.get_name()}"
        )
        combine_buf = V.graph.name_to_buffer.get(combine_name)
        if combine_buf is not None and combine_buf in operations:
            insert_idx = operations.index(combine_buf) + 1
        else:
            insert_idx = operations.index(tiled_op) + 1
    operations.insert(insert_idx, copy_buf)


def _compute_fill_loop_info(op: ComputedBuffer) -> "CoarseTileInfo | None":
    """Return the loop_info to stamp on the fill op for a nested tiled reduction.

    For a flat (pure reduction) tiling the fill has no loop_info — it runs
    once before all loops.  Returns None.

    For a nested tiling where outer level(s) tile output dims and the inner
    level tiles a reduction dim, the fill must run inside the outer loop (once
    per outer tile) so the accumulator is per-outer-tile sized.  Returns a
    CoarseTileInfo covering only the outer output-dim levels.
    """
    loop_info = op.loop_info
    tiled_rdims = getattr(loop_info, "loop_tiled_reduction_dims", [])

    outer_counts: list[sympy.Expr] = []
    outer_tiled_dims: list[list[int]] = []
    outer_tiled_rdims: list[list[int]] = []
    for dims, rdims, count in zip(
        loop_info.loop_tiled_dims, tiled_rdims, loop_info.loop_count
    ):
        if dims:  # non-empty output-dim list → this is an output-dim level
            outer_counts.append(count)
            outer_tiled_dims.append(dims)
            outer_tiled_rdims.append([])

    if not outer_counts:
        return None  # flat: fill runs before all loops

    outer_gid = loop_info.loop_group_id[: len(outer_counts)]
    return CoarseTileInfo(
        loop_group_id=outer_gid,
        loop_count=outer_counts,
        loop_tiled_dims=outer_tiled_dims,
        loop_tiled_reduction_dims=outer_tiled_rdims,
        tiled_dims_per_read=[],
        output_tiled_dims=[],
    )


def _propagate_tiled_reduction_op(
    op: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Handle buffer propagation for a Reduction op tiled over a reduction dim.

    Strategy: fill-initialize + per-tile combine.
      1. Allocate a HBM accumulation buffer sized to the full
         (pre-outer-division) output shape (_compute_full_ranges), so that
         address advancement across outer tiles writes each tile into the
         correct slice.  For flat (reduction-only) tiling this equals
         op.data.ranges.
      2. Insert a fill op that writes the reduction's identity value into the
         accumulation buffer.  For flat reduction tiling the fill has no
         loop_info and runs before all loops.  For nested tiling (outer
         output-dim loop + inner reduction loop) the fill carries the outer
         loop's loop_info so it runs inside the outer loop — once per outer
         tile — keeping the accumulator sized to the per-tile output shape.
      3. Insert a combine op (inside the inner loop, same loop_info as the
         tiled reduction op) that merges each tile's partial result into the
         accumulation buffer using the reduction's combining fn.
      4. Mark the tiled reduction op's output as per_tile_fixed (inner-loop
         scratch, not advanced between inner iterations).
      5. Patch outside consumers and graph outputs to read the accumulation
         buffer.
    """
    loop_info = op.loop_info
    loop_group_id = loop_info.loop_group_id
    reduction_type = op.data.reduction_type
    identity = _reduction_identity_value(reduction_type, op.get_dtype())

    # Per-outer-tile output shape (ranges after any outer tiling divided them).
    per_tile_ranges = list(op.data.ranges)

    # Accumulation buffer uses the full (pre-outer-division) output shape so
    # that address advancement across outer output-dim tiles writes each tile's
    # result into the correct slice.  For reduction-dim-only tiling there is no
    # outer division, so full == per-tile.
    full_output_ranges = _compute_full_ranges(op)

    # Insert HBM buffer before the first op in the loop group.
    outer_key = loop_group_id[0]
    group_start_idx = next(
        i
        for i, o in enumerate(operations)
        if isinstance(o, ComputedBuffer)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
    )

    fill_loop_info = _compute_fill_loop_info(op)
    is_nested = fill_loop_info is not None

    if is_nested:
        # Nested case: allocate separate tile-sized and full-sized buffers.
        # accum_tile (per_tile_fixed=True) stays inside the inner K-loop;
        # accum_full accumulates across outer B-tiles via a copy op.
        accum_full = _allocate_full_buffer(
            op, full_output_ranges, operations, group_start_idx
        )
        group_start_idx_after_full = operations.index(accum_full) + 1
        accum_tile = _allocate_full_buffer(
            op, per_tile_ranges, operations, group_start_idx_after_full
        )
        if isinstance(accum_tile.layout, FixedTiledLayout):
            # Post-stickify (span-overflow path): set directly.
            accum_tile.layout.per_tile_fixed = True
        else:
            # Pre-stickify (hint path): layout is FixedLayout; defer to
            # finalize_layouts.  In the span-overflow path this branch is
            # unreachable because _allocate_full_buffer returns FixedTiledLayout
            # when the original layout is already FixedTiledLayout.
            accum_tile._pending_per_tile_fixed = True  # type: ignore[attr-defined]
        fill_target = accum_tile
        combine_target = accum_tile
    else:
        # Flat case: single full-sized buffer (unchanged behaviour).
        accum_full = _allocate_full_buffer(
            op, full_output_ranges, operations, group_start_idx
        )
        fill_target = accum_full
        combine_target = accum_full

    # Insert fill op immediately after the fill target buffer allocation
    # (outside the loop for flat, inside the outer loop for nested).
    # Use a SpyreConstantFallback scalar as the fill source so that Spyre's
    # kernel codegen can express this as an IDENTITY_OP broadcast.  For the
    # span-overflow path, finalize_layouts has already run so we must assign a
    # FixedTiledLayout manually here.  For the hint path (pre-stickify),
    # stickification will overwrite the layout; the manual assignment is
    # redundant but harmless.
    dtype = op.get_dtype()
    device = op.get_device()
    from ..ir import SpyreConstantFallback  # deferred: avoids circular import

    scalar_op = SpyreConstantFallback(
        torch.ops.spyre.constant.default, float(identity), dtype, device
    )
    # SpyreTensorLayout([], dtype) yields device_size=[1, 64], stride_map=[-1, -1]
    # — a 0-d broadcast scalar in Spyre's device coordinate system.
    scalar_stl = SpyreTensorLayout([], dtype)
    scalar_op.layout = FixedTiledLayout(device, dtype, [], [], scalar_stl)
    scalar_loader = TensorBox.create(scalar_op).make_loader()

    fill_data = Pointwise(
        device=device,
        dtype=dtype,
        inner_fn=lambda index, _loader=scalar_loader: _loader([]),
        ranges=per_tile_ranges,
    )
    fill_name = V.graph.qualify_name(f"coarse_tile_fill_{op.get_name()}")
    fill_buf = ComputedBuffer(
        name=fill_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(fill_target))),
        data=fill_data,
    )
    fill_buf.origins = op.origins
    fill_buf.operation_name = fill_name
    if fill_loop_info is not None:
        fill_buf.loop_info = fill_loop_info  # type: ignore[attr-defined]
    # else: no loop_info — fill runs once before all loops (flat reduction case).
    # fill_buf's write is only ever "read" by the NEXT loop iteration's use of
    # fill_target as an accumulator seed — a cross-iteration dependency
    # invisible to the single-pass, pre-unroll IR the scheduler's
    # dead_node_elimination walks, so without this it is (wrongly) seen as
    # dead and removed. Mirrors copy_buf's force_live handling above.
    fill_buf._coarse_tile_force_live = True  # type: ignore[attr-defined]
    V.graph.name_to_buffer[fill_name] = fill_buf
    fill_target_idx = operations.index(fill_target)
    # scalar_op was appended to graph.operations by register_operation(); move it
    # to just after fill_target, then insert fill_buf after scalar_op.
    operations.remove(scalar_op)
    operations.insert(fill_target_idx + 1, scalar_op)
    operations.insert(fill_target_idx + 2, fill_buf)

    # Insert combine op after the tiled reduction op (inside the loop).
    _insert_combine_op(op, combine_target, operations)

    # For nested case, insert a copy op at the outer loop level that writes
    # accum_tile → accum_full, advancing accum_full across outer output tiles.
    if is_nested:
        assert fill_loop_info is not None  # guaranteed by is_nested == True
        _insert_reduction_copy_op(
            op, accum_tile, accum_full, fill_loop_info, operations
        )

    # Mark the tiled reduction op's per-tile scratch as stationary.
    # Pre-stickify: op.layout is FixedLayout; per_tile_fixed will be set by
    # finalize_layouts (which inspects CoarseTileInfo) once stickification runs.
    # Post-stickify (span-overflow path): op.layout is already FixedTiledLayout.
    if isinstance(op.layout, FixedTiledLayout):
        op.layout.per_tile_fixed = True

    # Record the accumulation buffer name so finalize_layouts can propagate
    # the reduction op's post-stickify device layout to accum_full.  Pre-stickify,
    # accum_full gets a generic STL from propagate_spyre_tensor_layouts; we must
    # overwrite it with the actual reduction output STL so fill, combine, and copy
    # all agree on the device coordinate system.
    op._tiled_reduction_accum_name = accum_full.get_name()  # type: ignore[attr-defined]

    # Patch consumers to read accum_full (the fully-assembled output).
    buf_name = op.get_name()
    outside_consumers, is_graph_output = _find_outside_consumers(
        buf_name, loop_group_id, operations
    )
    accum_name = accum_full.get_name()
    retile_info = _RetiledBufferInfo(
        tuple(op.layout.stride), tuple(accum_full.layout.stride)
    )
    _patch_consumers(outside_consumers, buf_name, accum_name, operations, retile_info)
    if is_graph_output:
        _patch_graph_outputs(buf_name, accum_full)

    logger.debug(
        "coarse_tile: tiled reduction %s → accum_full %s (fill=%s, identity=%s, "
        "nested=%s)",
        buf_name,
        accum_name,
        fill_name,
        identity,
        is_nested,
    )


# ---------------------------------------------------------------------------
# Consumer / graph-output patching
# ---------------------------------------------------------------------------


def _patch_consumers(
    consumers: list[ComputedBuffer],
    old_name: str,
    new_name: str,
    operations: list[Operation],
    retile_info: _RetiledBufferInfo | None = None,
) -> None:
    """Redirect outside consumers from old_name to new_name.

    Patches each consumer's inner_fn via NameSwapHandler (or
    _NameAndIndexSwapHandler, when retile_info's old/new strides differ) and
    reconstructs the ComputedBuffer to invalidate the sizes cache.

    retile_info carries the old (tile-local) and new (full-size) strides of
    the renamed buffer, needed whenever new_name isn't addressing-equivalent
    to old_name (e.g. a coarse-tiled dim's stride scaled up for the full
    buffer) — plain NameSwapHandler forwards the load index unmodified,
    which computes wrong addresses when the strides differ.
    """
    if not consumers or old_name == new_name:
        return

    from ..insert_restickify import NameSwapHandler
    from ..pass_utils import replace_computed_buffer_body

    name_map = {old_name: new_name}
    rewrites = _stride_rewrite_map(retile_info) if retile_info is not None else {}

    for consumer in consumers:
        orig_inner = consumer.data.inner_fn

        def new_inner_fn(
            *args,
            _map=name_map,
            _rewrites_by_old_name={old_name: rewrites} if rewrites else {},
            _orig=orig_inner,
        ):
            if _rewrites_by_old_name:
                handler = _NameAndIndexSwapHandler(V.ops, _map, _rewrites_by_old_name)
            else:
                handler = NameSwapHandler(V.ops, _map)
            with V.set_ops_handler(handler):
                return _orig(*args)

        object.__setattr__(consumer.data, "inner_fn", new_inner_fn)
        consumer = replace_computed_buffer_body(consumer, consumer.data, operations)
        V.graph.name_to_buffer[consumer.get_name()] = operations[
            next(
                i
                for i, op in enumerate(operations)
                if isinstance(op, ComputedBuffer)
                and op.get_name() == consumer.get_name()
            )
        ]


def _stride_rewrite_map(info: _RetiledBufferInfo) -> dict[Expr, Expr]:
    """Map unique stale stride coefficients to their retiled coefficients."""

    old_counts = Counter(sympy.simplify(s) for s in info.old_stride)
    rewrites: dict[Expr, Expr] = {}
    for old, new in zip(info.old_stride, info.new_stride):
        old = sympy.simplify(old)
        new = sympy.simplify(new)
        if old_counts[old] == 1 and sympy.simplify(old - new) != 0:
            rewrites[old] = new
    return rewrites


def _retile_load_index_from_strides(
    buf_name: str,
    index: Expr,
    rewrites: dict[Expr, Expr],
) -> Expr:
    """Rewrite separable affine load-index terms from full strides to tile strides."""

    if not rewrites:
        return index

    loop_vars = index.free_symbols
    if not loop_vars:
        return index

    replacements = {var: sympy.S.Zero for var in loop_vars}
    offset = index.xreplace(replacements)
    projection_terms: dict[sympy.Symbol, Expr] = {}
    for var in sorted(loop_vars, key=str):
        other_vars = {other: sympy.S.Zero for other in loop_vars if other != var}
        projection_terms[var] = sympy.expand(index.xreplace(other_vars) - offset)

    residual = sympy.simplify(index - offset - sum(projection_terms.values()))
    if residual != 0:
        logger.warning(
            "coarse_tile: refusing to retile load index for %s: index=%s has "
            "mixed loop-variable residual %s",
            buf_name,
            index,
            residual,
        )
        return index

    adjusted_index = offset
    changed = False
    for var in sorted(loop_vars, key=str):
        term = projection_terms[var]
        coeff = term.coeff(var)
        remainder = sympy.simplify(term - coeff * var)
        if remainder != 0:
            logger.warning(
                "coarse_tile: refusing to retile load index for %s: projection "
                "for %s is non-affine in index=%s: %s",
                buf_name,
                var,
                index,
                term,
            )
            return index

        matches = [
            new_coeff
            for old_coeff, new_coeff in rewrites.items()
            if sympy.simplify(coeff - old_coeff) == 0
        ]
        if len(matches) == 1:
            adjusted_index += matches[0] * var
            changed = True
        else:
            adjusted_index += term

    if changed:
        logger.debug(
            "coarse_tile: retiled load index for %s: %s -> %s",
            buf_name,
            index,
            adjusted_index,
        )
        return sympy.simplify(adjusted_index)
    return index


class _RetileLoadIndexHandler(WrapperHandler):
    """Ops handler that retiles loads from buffers whose host strides changed."""

    def __init__(self, inner, rewrites_by_name: dict[str, dict[Expr, Expr]]):
        super().__init__(inner)
        self._rewrites_by_name = rewrites_by_name

    def load(self, name, index):
        if name in self._rewrites_by_name:
            index = _retile_load_index_from_strides(
                name, index, self._rewrites_by_name[name]
            )
        return super().load(name, index)


class _NameAndIndexSwapHandler(WrapperHandler):
    """Redirect ops.load(name, index) to a new name and rewrite its index.

    Rewrites index while `name` is still the old name (its coefficients are
    in the old buffer's strides), then swaps the name. Reuses
    _retile_load_index_from_strides.
    """

    def __init__(
        self,
        inner,
        name_map: dict[str, str],
        rewrites_by_old_name: dict[str, dict[Expr, Expr]],
    ):
        super().__init__(inner)
        self._name_map = name_map
        self._rewrites_by_old_name = rewrites_by_old_name

    def load(self, name, index):
        if name in self._rewrites_by_old_name:
            index = _retile_load_index_from_strides(
                name, index, self._rewrites_by_old_name[name]
            )
        return super().load(self._name_map.get(name, name), index)


def _should_patch_retiled_load_indexes(
    op: Operation,
    group_id: tuple[int, ...],
    retiled_names: set[str],
) -> bool:
    """Return True when op is an exact-loop consumer of a retiled buffer."""
    if not isinstance(op, ComputedBuffer):
        return False
    if not isinstance(op.data, (Pointwise, Reduction)):
        return False
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None or loop_info.loop_group_id != group_id:
        return False
    return any(_reads_buffer(op, name) for name in retiled_names)


def _replace_group_op(
    group_ops: list[Operation], old_op: Operation, new_op: Operation
) -> None:
    """Keep the tiling group list in sync after replacing a ComputedBuffer body."""
    old_name = old_op.get_operation_name()
    for idx, group_op in enumerate(group_ops):
        if group_op is old_op or group_op.get_operation_name() == old_name:
            group_ops[idx] = new_op
            return


def _patch_retiled_load_indexes(
    group_id: tuple[int, ...],
    group_ops: list[Operation],
    retiled_infos: dict[str, _RetiledBufferInfo],
    operations: list[Operation],
) -> None:
    """Rewrite stale load indexes for consumers of buffers retiled by coarse tiling."""
    rewrites_by_name = {
        name: rewrites
        for name, info in retiled_infos.items()
        if (rewrites := _stride_rewrite_map(info))
    }
    if not rewrites_by_name:
        return

    from ..pass_utils import replace_computed_buffer_body

    # Only ops that were already in the group when _apply_plan ran can hold a
    # stale (pre-divide) coefficient for a retiled buffer.  Ops inserted later
    # by insert_tiling_propagation (e.g. _insert_copy_op's copy_buf) read the
    # retiled buffer's already-updated layout directly, so rewriting them here
    # would double-apply the stride correction (see issue found while fixing
    # test_hint_restickify_stays_in_group).
    retiled_names = set(rewrites_by_name)
    for op in list(group_ops):
        if not _should_patch_retiled_load_indexes(op, group_id, retiled_names):
            continue

        orig_inner = op.data.inner_fn

        def new_inner_fn(*args, _rewrites=rewrites_by_name, _orig=orig_inner):
            with V.set_ops_handler(_RetileLoadIndexHandler(V.ops, _rewrites)):
                return _orig(*args)

        object.__setattr__(op.data, "inner_fn", new_inner_fn)
        new_op = replace_computed_buffer_body(op, op.data, operations)
        _replace_group_op(group_ops, op, new_op)
        V.graph.name_to_buffer[new_op.get_name()] = new_op


def _patch_graph_outputs(old_name: str, new_buf: ComputedBuffer) -> None:
    """Replace references to old_name in V.graph.graph_outputs with new_buf."""
    try:
        outputs = V.graph.graph_outputs
    except Exception:
        return

    new_tb = TensorBox(StorageBox(new_buf))
    for i, out in enumerate(outputs):
        # Unwrap StorageBox layers to reach ComputedBuffer without going into
        # the ComputedBuffer's inner data (Pointwise / Reduction).
        candidate = out
        while isinstance(candidate, StorageBox):
            candidate = candidate.data
        if isinstance(candidate, ComputedBuffer) and candidate.get_name() == old_name:
            outputs[i] = new_tb
