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
    coarse_tile_pre_stickify(graph, groups)

``groups`` is a list of ``(ops, levels)`` tuples where ``levels`` is a list of
``(hint_id, count)`` pairs, outermost first.  Each op resolves its own
tiled dimension from its ``loop_var`` in ``dim_hints``.

Each ``ops`` list must be a contiguous sub-sequence of ``operations``.

After stamping, each entry point runs its own sequence of passes.
``coarse_tile_pre_stickify`` runs ``_insert_all_read_copy_ops``,
``_insert_all_reduction_ops``, then ``_insert_all_write_copy_ops``;
``coarse_tile_post_stickify`` skips ``_insert_all_read_copy_ops`` and runs
only the latter two. All three passes allocate full-sized output buffers
and insert copy/mutation/reduction ops for tiled operations whose results
are consumed outside the loop, driven by the ``PropagationPlan`` each op's
``loop_info`` already carries from planning.

Before touching any ``inner_fn``/``layout``/``MutationLayoutSHOULDREMOVE``
rewiring in this file, read "Appendix: How IR rewiring works, and why it's
sound" in ``docs/source/compiler/coarse_tiling_loops.md``. It documents the
wrap-never-reconstruct convention and why ``MutationLayoutSHOULDREMOVE``
sites must satisfy the single-mutation-target invariant -- the same ground
this file's rewrite sites depend on.
"""

from __future__ import annotations


import collections
import dataclasses
import enum
import logging
from typing import NamedTuple

import sympy
from sympy import Expr

import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import sympy_index_symbol, sympy_subs
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    IRNode,
    Layout,
    Loops,
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

from .. import config
from ..constants import BATCH_MATMUL_OP, MATMUL_REDUCTION_OPS
from ..errors import Unsupported
from ..logging_utils import get_inductor_logger
from ..loop_info import (
    CarriedReductionRecord,
    CarriedReductionSpec,
    CoarseTileInfo,
    PropagationPlan,
    ReadCopyEntry,
    ReadCopyElisionRecord,
    ReadCopyPlan,
    ReductionPlan,
    copy_op_metadata,
)
from ..propagate_hints import get_op_hints
from ..pass_utils import (
    op_out_coords,
    host_coordinates,
    identify_matmul_inputs,
    indirect_sizes_from_op,
    invalidate_op_read_writes,
    iteration_space_from_op,
)
from ..ir import FixedTiledLayout, SpyreConstantFallback, _resize_device_layout
from .tile import compute_tile_index, compute_tile_stride, decompose_index_for_tiling

logger = get_inductor_logger("coarse_tile")


class _RetiledBufferInfo(NamedTuple):
    """Host strides before and after a buffer is resized for a coarse tile."""

    old_stride: tuple[Expr, ...]
    new_stride: tuple[Expr, ...]
    old_size: tuple[Expr, ...]


class _ReadCopyHoistDecision(enum.Enum):
    """Why a staged read may or may not move before its counted loop."""

    ELIGIBLE = enum.auto()
    CROSS_GROUP_SOURCE = enum.auto()
    LOOP_PRODUCED_SOURCE = enum.auto()
    IN_LOOP_WRITTEN_SOURCE = enum.auto()
    UNKNOWN_SOURCE = enum.auto()
    MISSING_STEP_METADATA = enum.auto()
    ADVANCING_READ = enum.auto()


class _ReadCopySourceKind(enum.Enum):
    """What planning has positively established about a staged-read source."""

    KNOWN_EXTERNAL = enum.auto()
    LOOP_PRODUCED = enum.auto()
    IN_LOOP_WRITTEN = enum.auto()
    UNKNOWN = enum.auto()


class _ReadCopySourceInfo(NamedTuple):
    kind: _ReadCopySourceKind
    loop_group_id: tuple[int, ...] | None = None


def _read_copy_hoist_decision(
    consumer_info: CoarseTileInfo,
    dep_idx: int,
    source_info: _ReadCopySourceInfo,
) -> _ReadCopyHoistDecision:
    """Classify whether a read sees the same source slice on every trip.

    Source stability and address stability are independent requirements. A
    graph input can still be read through an advancing window, while a fixed
    scratch address can be rewritten by another counted loop every trip.
    Hoisting is sound only when the source is loop-external and complete
    read-step metadata proves that the consumer's window does not advance.
    Reads synthesized after Pass 1 are outside this classifier's scope and
    must be validated by the pass that creates them.
    """
    if source_info.kind is _ReadCopySourceKind.LOOP_PRODUCED:
        if source_info.loop_group_id != consumer_info.loop_group_id:
            # Keep this distinct from the general loop-produced rejection:
            # cross-group scratch caused the correctness bug this guard fixes.
            return _ReadCopyHoistDecision.CROSS_GROUP_SOURCE
        return _ReadCopyHoistDecision.LOOP_PRODUCED_SOURCE
    if source_info.kind is _ReadCopySourceKind.IN_LOOP_WRITTEN:
        return _ReadCopyHoistDecision.IN_LOOP_WRITTEN_SOURCE
    if source_info.kind is _ReadCopySourceKind.UNKNOWN:
        return _ReadCopyHoistDecision.UNKNOWN_SOURCE
    if source_info.kind is not _ReadCopySourceKind.KNOWN_EXTERNAL:
        raise AssertionError(f"unexpected read-copy source kind {source_info.kind}")

    # tiled_dims_per_read is dense: planning records one entry for every read,
    # so a missing entry is missing evidence. squeezed_advance_per_read is
    # sparse by design: an empty outer list is complete "none needed" evidence.
    if dep_idx >= len(consumer_info.tiled_dims_per_read):
        return _ReadCopyHoistDecision.MISSING_STEP_METADATA
    if consumer_info.squeezed_advance_per_read and dep_idx >= len(
        consumer_info.squeezed_advance_per_read
    ):
        return _ReadCopyHoistDecision.MISSING_STEP_METADATA

    per_level_dims = consumer_info.tiled_dims_per_read[dep_idx]
    per_level_squeezed = (
        consumer_info.squeezed_advance_per_read[dep_idx]
        if consumer_info.squeezed_advance_per_read
        else []
    )
    if any(per_level_dims) or any(per_level_squeezed):
        return _ReadCopyHoistDecision.ADVANCING_READ
    return _ReadCopyHoistDecision.ELIGIBLE


def _loop_written_buffer_names(operations: list[Operation]) -> set[str]:
    """Return storage names mutated by an operation inside a counted loop."""
    written: set[str] = set()
    for op in operations:
        if hasattr(op, "loop_info"):
            # get_mutation_names is Inductor's authoritative record of storage
            # written by this op beyond its ordinary produced value.
            written.update(op.get_mutation_names())
    return written


def _read_copy_source_info(
    source_name: str,
    operations_by_name: dict[str, Operation],
    loop_written_names: set[str],
) -> _ReadCopySourceInfo:
    """Classify one source using positive producer and writer evidence."""
    if source_name in loop_written_names:
        return _ReadCopySourceInfo(_ReadCopySourceKind.IN_LOOP_WRITTEN)

    source = operations_by_name.get(source_name)
    if source is None:
        source = V.graph.try_get_buffer(source_name)
        if source is None:
            return _ReadCopySourceInfo(_ReadCopySourceKind.UNKNOWN)

    while isinstance(source, (TensorBox, StorageBox)):
        source = source.data

    if isinstance(source, ComputedBuffer):
        producer_info = getattr(source, "loop_info", None)
        if producer_info is not None:
            return _ReadCopySourceInfo(
                _ReadCopySourceKind.LOOP_PRODUCED,
                producer_info.loop_group_id,
            )
        return _ReadCopySourceInfo(_ReadCopySourceKind.KNOWN_EXTERNAL)

    # Imported lazily for the same circular-import reason as
    # _full_buffer_read_deps. A SpyreEmptyFallback is only external when the
    # writer scan above proved no counted-loop operation mutates it.
    from ..ir import SpyreEmptyFallback

    if isinstance(source, (InputBuffer, SpyreEmptyFallback)):
        return _ReadCopySourceInfo(_ReadCopySourceKind.KNOWN_EXTERNAL)
    return _ReadCopySourceInfo(_ReadCopySourceKind.UNKNOWN)


class _LogicalIterationSymbol(NamedTuple):
    """One active loop symbol keyed by its stable raw dimension identity."""

    logical_dim: tuple[str, int]
    extent: Expr
    symbol: sympy.Symbol


class _IterationSymbolRemap(NamedTuple):
    """Order-preserving loop-symbol translation produced by a range rewrite."""

    before_symbols: tuple[sympy.Symbol, ...]
    pairs: tuple[tuple[sympy.Symbol, sympy.Symbol], ...]


class _DivideRangesResult(NamedTuple):
    retiled_info: _RetiledBufferInfo | None
    symbol_remap: _IterationSymbolRemap | None


def _work_division_names(op: ComputedBuffer) -> dict[str, int]:
    """Return the explicit named work-division request attached to ``op``."""

    result: dict[str, int] = {}
    for _, hint_dict in sorted(get_op_hints(op).items()):
        result.update(hint_dict.get("work_div") or {})
    return result


def _plan_carried_sum(
    op: ComputedBuffer,
    info: CoarseTileInfo,
    *,
    is_graph_output: bool,
    outside_consumer_names: list[str],
    is_nested: bool,
) -> CarriedReductionSpec | None:
    """Recognize the one safe shape for an LX-resident loop-carried sum.

    The work-division hint names the output row dimension.  Resolve it here,
    before any IR is rewritten, so a misspelled or reduction-dimension hint
    cannot create a carried accumulator with ambiguous ownership.
    """

    data = op.data
    has_work_div = bool(getattr(op, "work_div_loop_info", None))
    if not (
        isinstance(data, Reduction)
        and data.reduction_type == "sum"
        and getattr(data, "src_dtype", object()) == getattr(data, "dtype", None)
        and not is_nested
        and (is_graph_output or has_work_div)
        and (not outside_consumer_names or has_work_div)
        and len(info.loop_count) == 1
        and all(not dims for dims in info.loop_tiled_dims)
        and info.loop_tiled_reduction_dims == [[0]]
        and len(data.reduction_ranges) == 1
    ):
        return None

    requested = _work_division_names(op)
    if not requested:
        return None

    named_symbols = getattr(op, "work_div_loop_info", {})
    output_symbols = set(list(iteration_space_from_op(op))[: len(data.ranges)])
    resolved: list[tuple[str, int]] = []
    unresolved: list[str] = []
    for name, split in requested.items():
        symbols = [sym for sym, names in named_symbols.items() if name in names]
        if len(symbols) != 1 or symbols[0] not in output_symbols:
            unresolved.append(name)
            continue
        if isinstance(split, bool) or not isinstance(split, (int, sympy.Integer)):
            raise Unsupported(
                f"coarse_tile: carried sum {op.get_name()} work_div {name!r} "
                f"must be a positive integer, got {split!r}"
            )
        split = int(split)
        if split < 1:
            raise Unsupported(
                f"coarse_tile: carried sum {op.get_name()} work_div {name!r} "
                f"must be positive, got {split}"
            )
        resolved.append((name, split))

    if unresolved or len(resolved) != 1:
        raise Unsupported(
            f"coarse_tile: carried sum {op.get_name()} requires exactly one "
            "work_div on an output row dimension; "
            f"resolved={resolved}, unresolved={unresolved}"
        )
    name, split = resolved[0]
    return CarriedReductionSpec(name, split)


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
        # Names of all ComputedBuffers in this group — used by the
        # per-op partial-scratch check below.
        group_op_names: set[str] = {
            o.get_name() for o in group_ops if isinstance(o, ComputedBuffer)
        }

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

            if has_tiled_reduction:
                _validate_planned_reduction_tiling(
                    op, op_tiled_dims, op_tiled_reduction_dims
                )

            if _plan_is_loop_invariant_at_reduction_levels(
                op, op_tiled_dims, group_reduction_tiled_levels
            ) and _reads_incomplete_reduction(
                op, group_ops, group_op_names, plan, group_reduction_tiled_levels
            ):
                raise Unsupported(
                    f"partial reduction result consumed before accumulation "
                    f"is complete (op {op.get_name()} reads a per-tile "
                    f"partial result from the same loop group)"
                )

            per_level_extents = _planned_tile_extents_per_level(
                op, op_tiled_dims, op_tiled_reduction_dims, levels
            )

            tiled_dims_per_read = [
                _tiled_dims_for_dep(dep, per_level_extents, op) for dep in read_deps
            ]
            output_tiled_dims = (
                _tiled_dims_for_dep(write_deps[0], per_level_extents, op)
                if write_deps
                else []
            )

            _check_matmul_broadcast_batch_tiling(
                op, read_deps, write_deps, per_level_extents
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
    object references past this stage -- see PropagationPlan's docstring on
    name stability), and looks up each candidate's outer loop-group key from
    name_to_group_outer_key (built once by the caller from the planned
    CoarseTileInfo dict) instead of a not-yet-stamped op.loop_info attribute.
    """
    outer_key = group_loop_id[0]
    consumer_names: list[str] = []
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not _reads_buffer(op, buf_name):
            continue
        candidate_outer_key = name_to_group_outer_key.get(op.get_name())
        if candidate_outer_key is None or candidate_outer_key != outer_key:
            consumer_names.append(op.get_name())

    is_graph_output = buf_name in _graph_output_names()
    return consumer_names, is_graph_output


def _compute_full_ranges_planned(
    op: ComputedBuffer, info: CoarseTileInfo
) -> list[Expr]:
    """Compute op's full (pre-division) output ranges without mutating.

    Planning runs *before* _apply_plan/_divide_ranges, so op.data.ranges is
    already the undivided, full shape here -- return it unchanged rather
    than multiplying by loop_count again (which would double the extent of
    every tiled dim).
    """
    return list(op.data.ranges)


def _compute_per_tile_ranges_planned(
    op: ComputedBuffer, info: CoarseTileInfo
) -> list[Expr]:
    """Compute op's post-division (per-tile) output ranges without mutating.

    Mirrors the arithmetic _divide_ranges performs in place, but reads
    pre-mutation op.data.ranges (planning runs before _apply_plan) and
    returns a fresh list instead of writing through op.data/op.layout. A dim
    tiled at more than one level is divided by every such level's count, matching
    _divide_ranges being called once per level in the transformation loop.
    """
    ranges = list(op.data.ranges)
    for count, dims in zip(info.loop_count, info.loop_tiled_dims):
        for d in dims:
            if 0 <= d < len(ranges):
                r = ranges[d]
                if isinstance(r, (int, sympy.Integer)) and isinstance(
                    count, (int, sympy.Integer)
                ):
                    assert int(r) % int(count) == 0, (
                        f"coarse_tile: op {op.get_name()!r} loop var d{d} range "
                        f"{r} is not divisible by loop_count {count}."
                    )
                    ranges[d] = sympy.Integer(int(r) // int(count))
                else:
                    ranges[d] = sympy.simplify(sympy.sympify(r) / sympy.sympify(count))
    return ranges


def _compute_fill_loop_info_planned(
    info: CoarseTileInfo,
) -> CoarseTileInfo | None:
    """Compute the fill op's trimmed loop_info from op's planned CoarseTileInfo.

    For a flat tiling (no output-dim levels, or every reduction-dim level is
    outer to every output-dim level) the fill has no loop_info — it runs once
    before all loops. Returns None.

    For a nested tiling where an output-dim level is outer to a
    reduction-dim level, the fill must run inside that outer loop (once per
    outer tile) so the accumulator is per-outer-tile sized. Returns a
    CoarseTileInfo covering only those outer output-dim levels.

    An output-dim level being outer to a reduction-dim level is what makes
    this nested: only then does the reduction re-run per outer tile, which is
    what requires a per-tile accumulator to be re-seeded on every outer
    iteration in the first place. An output-dim level that is inner to every
    reduction-dim level (e.g. softmax(dim=0) tiled A÷4 B÷4, with the A
    reduction outer and B's output tiling inner) does not carry this
    requirement: each inner B-tile still sees the reduction accumulate over
    the *entire* A range, exactly like the flat case, so a single
    full-output-sized accumulator initialized once is correct.
    """
    tiled_rdims = info.loop_tiled_reduction_dims

    output_level_indices = [i for i, dims in enumerate(info.loop_tiled_dims) if dims]
    reduction_level_indices = [i for i, rdims in enumerate(tiled_rdims) if rdims]

    if not output_level_indices:
        return None  # flat: no output-dim tiling at all

    outermost_output = min(output_level_indices)
    if not reduction_level_indices or outermost_output > max(reduction_level_indices):
        # Every reduction level is outer to every output level → the output
        # tiling is entirely inner to the reduction; flat case.
        return None

    # Interleaved topology: either (a) some output-dim level(s) are outer to
    # a reduction level while other output-dim level(s) are inner to that
    # *same* level, or (b) an output-dim level sits strictly between two
    # separate reduction levels (e.g. reduction/output/reduction nesting) —
    # it re-runs once per outer reduction tile just as surely as case (a)
    # would, but no single reduction level in isolation has output on both
    # sides of it, so (a) alone can't see it. Checking only the aggregate
    # outermost-output-vs-innermost-reduction boundary (as a prior version of
    # this function did) misses (b) entirely: that comparison only ever
    # relates the outermost output level to the *innermost* reduction level,
    # never to a reduction level in the interior of the reduction set.
    innermost_reduction = max(reduction_level_indices)
    outermost_reduction = min(reduction_level_indices)
    for r in reduction_level_indices:
        outer_output = [i for i in output_level_indices if i < r]
        inner_output = [i for i in output_level_indices if i > r]
        if outer_output and inner_output:
            raise Unsupported(
                f"coarse_tile: interleaved reduction tiling not supported — "
                f"output-dim level(s) {outer_output} are outer to reduction "
                f"level {r} but output-dim level(s) {inner_output} are inner "
                f"to it (reduction levels: {reduction_level_indices}). "
                f"Reorder spyre_hint scopes so all output dims are outer to "
                f"all reduction dims."
            )
    sandwiched = [
        i for i in output_level_indices if outermost_reduction < i < innermost_reduction
    ]
    if sandwiched:
        raise Unsupported(
            f"coarse_tile: interleaved reduction tiling not supported — "
            f"output-dim level(s) {sandwiched} are sandwiched between "
            f"reduction levels {reduction_level_indices} (between level "
            f"{outermost_reduction} and level {innermost_reduction}). "
            f"Reorder spyre_hint scopes so all output dims are outer to all "
            f"reduction dims."
        )

    # Nested: collect only the output-dim levels that are outer to a
    # reduction level.
    outer_counts: list[sympy.Expr] = []
    outer_tiled_dims: list[list[int]] = []
    outer_tiled_rdims: list[list[int]] = []
    for i, (dims, _rdims, count) in enumerate(
        zip(info.loop_tiled_dims, tiled_rdims, info.loop_count)
    ):
        if dims and i < innermost_reduction:
            outer_counts.append(count)
            outer_tiled_dims.append(dims)
            outer_tiled_rdims.append([])

    if not outer_counts:
        return None  # flat: fill runs before all loops

    outer_gid = info.loop_group_id[: len(outer_counts)]
    return CoarseTileInfo(
        loop_group_id=outer_gid,
        loop_count=outer_counts,
        loop_tiled_dims=outer_tiled_dims,
        loop_tiled_reduction_dims=outer_tiled_rdims,
        tiled_dims_per_read=[],
        output_tiled_dims=[],
    )


def _plan_tiling_propagation(
    operations: list[Operation],
    groups: list[tuple],
    plan: dict[int, CoarseTileInfo],
) -> None:
    """Decide how every tiled op's result crosses its loop boundary.

    Mirrors _propagate_tiled_op / _propagate_tiled_reduction_op's decision
    logic exactly, but makes zero mutation: it only reads op.data.ranges/
    op.get_read_writes() (unmutated at this point -- _apply_plan hasn't run
    yet) and each op's already-computed planned CoarseTileInfo (looked up by
    id(op) in `plan`), and stores the result on that same CoarseTileInfo's
    new `propagation` field.

    Called right after plan_coarse_tile_groups's own per-op loop, over the
    same `groups`/`plan` -- same zero-mutation contract, same id(op) keying.
    Untiled/skipped ops (no entry in `plan`) are left untouched.
    """
    # Every candidate consumer/producer's outer loop-group key, built once
    # up front so _find_outside_consumers_planned doesn't re-derive it per
    # candidate. Keyed by name (not id(op)) to match _reads_buffer/
    # _graph_output_names' own name-based buffer lookup.
    # Prefer this call's own plan (the freshest data, for ops this call is
    # about to stamp); fall back to a real, already-stamped loop_info
    # attribute for ops outside this plan (e.g. from an earlier
    # coarse_tile_pre_stickify()/coarse_tile_post_stickify() call on the
    # same graph, or already-processed groups within a chained call) --
    # exactly the candidates _find_outside_consumers/
    # _full_buffer_read_deps consult via getattr(op, "loop_info", None) at
    # transformation time.
    name_to_group_outer_key: dict[str, int] = {}
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        info = plan.get(id(op))
        if info is None:
            info = getattr(op, "loop_info", None)
        if info is not None:
            name_to_group_outer_key[op.get_name()] = info.loop_group_id[0]

    for group_ops, levels in groups:
        group_op_names: set[str] = {
            o.get_name() for o in group_ops if isinstance(o, ComputedBuffer)
        }
        group_reduction_tiled_levels = _group_reduction_tiled_levels_in_group(
            group_ops, levels
        )
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            info = plan.get(id(op))
            if info is None:
                continue

            has_tiled_reduction = any(info.loop_tiled_reduction_dims)
            if isinstance(op.data, Reduction) and has_tiled_reduction:
                reduction_type = op.data.reduction_type
                identity = _reduction_identity_value(reduction_type, op.get_dtype())
                per_tile_ranges = _compute_per_tile_ranges_planned(op, info)
                full_output_ranges = _compute_full_ranges_planned(op, info)
                outer_fill_loop_info = _compute_fill_loop_info_planned(info)
                is_nested = outer_fill_loop_info is not None
                full_output_strides = tuple(op.layout.stride)
                per_tile_strides = tuple(
                    compute_tile_stride(
                        list(full_output_ranges),
                        list(full_output_strides),
                        list(per_tile_ranges),
                    )
                )
                buf_name = op.get_name()
                if not is_nested:
                    # Flat reduction-tiling: _propagate_tiled_reduction_op's
                    # inside_consumers filter only redirects a same-group
                    # consumer of buf_name to the fully-combined accum_full
                    # buffer when that consumer's own loop_tiled_dims
                    # exactly equals this reduction op's loop_tiled_dims. A
                    # consumer that tiles the reduction dim as a genuine
                    # output dim (e.g. softmax's sub/div, which must write
                    # the full un-reduced shape) can never satisfy that —
                    # the reduction dim lives in loop_tiled_reduction_dims
                    # for this op but in loop_tiled_dims for the consumer.
                    # Left unredirected, such a consumer silently reads
                    # buf_name's raw, un-combined per-tile scratch on every
                    # tile instead of the accumulated result — a silent
                    # wrong-code bug, not a diagnosable one. A correct fix
                    # requires splitting this loop group into two
                    # sequential passes (reduce-and-combine, then consume)
                    # with a real barrier between them, which coarse_tile's
                    # single-fused-loop-body model cannot express today.
                    # Reject the pattern loudly instead of compiling it
                    # wrong.
                    outer_key = info.loop_group_id[0]
                    for candidate in operations:
                        if (
                            not isinstance(candidate, ComputedBuffer)
                            or candidate is op
                            or not _reads_buffer(candidate, buf_name)
                        ):
                            continue
                        if name_to_group_outer_key.get(candidate.get_name()) != (
                            outer_key
                        ):
                            continue
                        candidate_info = plan.get(id(candidate))
                        if (
                            candidate_info is not None
                            and candidate_info.loop_tiled_dims != info.loop_tiled_dims
                        ):
                            raise Unsupported(
                                f"coarse_tile: reduction op {buf_name!r} "
                                f"(reduction_type={reduction_type!r}) is "
                                f"tiled alongside a sibling output dim, and "
                                f"same-group consumer "
                                f"{candidate.get_name()!r} tiles the "
                                f"reduction dim as a real output dim "
                                f"(loop_tiled_dims={candidate_info.loop_tiled_dims} "
                                f"vs. reduction op's "
                                f"{info.loop_tiled_dims}). This consumer "
                                f"would read a partially-accumulated "
                                f"reduction result. Reorder spyre_hint "
                                f"scopes so the reduction dim is not tiled "
                                f"alongside another tiled output dim."
                            )
                consumer_names, is_graph_output = _find_outside_consumers_planned(
                    buf_name, info.loop_group_id, operations, name_to_group_outer_key
                )
                carried = _plan_carried_sum(
                    op,
                    info,
                    is_graph_output=is_graph_output,
                    outside_consumer_names=consumer_names,
                    is_nested=outer_fill_loop_info is not None,
                )
                reduction_plan = ReductionPlan(
                    reduction_type=reduction_type,
                    identity=identity,
                    is_nested=is_nested,
                    full_output_ranges=full_output_ranges,
                    per_tile_ranges=per_tile_ranges,
                    outer_fill_loop_info=outer_fill_loop_info,
                    full_output_strides=full_output_strides,
                    per_tile_strides=per_tile_strides,
                    carried=carried,
                )
                info.propagation = PropagationPlan(
                    kind="reduction",
                    reduction=reduction_plan,
                    outside_consumer_names=tuple(consumer_names),
                    is_graph_output=is_graph_output,
                )
                continue

            if all(not dims for dims in info.loop_tiled_dims):
                info.propagation = PropagationPlan(kind="loop_internal")
                continue

            buf_name = op.get_name()
            consumer_names, is_graph_output = _find_outside_consumers_planned(
                buf_name, info.loop_group_id, operations, name_to_group_outer_key
            )
            # A same-outer-group consumer that itself reads an unaccumulated
            # reduction sibling (e.g. softmax's div, reading sum) is deferred
            # to a separate loop nest after the reduction completes -- it
            # does not actually run in the same tile iteration as buf_name's
            # producer despite sharing loop_group_id[0]. Route buf_name to
            # copy_out for it exactly as if it were a genuine cross-group
            # consumer; see _consumers_reading_incomplete_reduction.
            reduction_consumer_names = _consumers_reading_incomplete_reduction(
                buf_name, group_ops, group_op_names, plan, group_reduction_tiled_levels
            )
            all_consumer_names = list(consumer_names) + [
                n for n in reduction_consumer_names if n not in consumer_names
            ]
            if not all_consumer_names and not is_graph_output:
                # A MutationLayoutSHOULDREMOVE op whose own buffer name isn't
                # a graph output may still write directly into a graph-input
                # buffer that is also a graph output (e.g. copy_forced(src, acc)
                # where acc is both a graph input and the graph output — the
                # op's own buffer is buf1, but arg0_1/acc is the graph output).
                # Detect that case and classify as mutation_write_back so
                # Pass 3 sets output_tiled_dims on the op itself rather than
                # inserting a separate copy-out.
                #
                # IMPORTANT: only trigger for graph-INPUT targets.  A locally-
                # created buffer that happens to be a graph output must still
                # go through the normal copy_out path (_insert_copy_op inserts
                # a coarse_tile_copy_* that advances through the full buffer).
                # If we classify that as mutation_write_back instead, the op
                # writes advancing tiles into the local scratch buffer while
                # the copy-out still reads from it — wrong values every tile
                # after the first.
                if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                    try:
                        mut_target = op.layout.get_buffer()
                        mut_target_name = mut_target.get_name()
                        target_is_graph_input = mut_target_name in V.graph.graph_inputs
                        target_consumer_names, target_is_output = (
                            _find_outside_consumers_planned(
                                mut_target_name,
                                info.loop_group_id,
                                operations,
                                name_to_group_outer_key,
                            )
                        )
                    except (AttributeError, TypeError):
                        target_is_graph_input = False
                        target_is_output = False
                        target_consumer_names = []
                        mut_target = None
                    if (
                        target_is_graph_input
                        and target_is_output
                        and mut_target is not None
                    ):
                        full_ranges = _compute_full_ranges_planned(op, info)
                        info.propagation = PropagationPlan(
                            kind="mutation_write_back",
                            full_ranges=full_ranges,
                            full_strides=tuple(mut_target.layout.stride),
                            is_graph_output=True,
                        )
                        continue
                    # Locally-created mutation target that IS the graph
                    # output, and/or is read by other ops outside this loop
                    # group (e.g. copy_forced(src, acc) where acc is a local
                    # buffer later read by another op, or is the function's
                    # return value, or both): must go through the normal
                    # copy_out path, keyed on the *target's* name -- not
                    # buf_name, which is the mutation op's own (irrelevant)
                    # buffer name -- for both consumer redirection
                    # (PropagationPlan.consumer_lookup_name) and, if
                    # applicable, graph-output patching
                    # (PropagationPlan.graph_output_name).
                    if (
                        target_is_output or target_consumer_names
                    ) and mut_target is not None:
                        full_ranges = _compute_full_ranges_planned(op, info)
                        info.propagation = PropagationPlan(
                            kind="copy_out",
                            full_ranges=full_ranges,
                            full_strides=tuple(op.layout.stride),
                            outside_consumer_names=tuple(target_consumer_names),
                            is_graph_output=target_is_output,
                            graph_output_name=mut_target_name,
                            consumer_lookup_name=mut_target_name,
                        )
                        continue
                    # Locally-created mutation target with NO outside
                    # consumer and NOT a graph output, but still read by
                    # another op inside this SAME loop group (e.g. flash
                    # attention's real_max: copy_forced(running_max,
                    # real_max) writes real_max, and the *next* tile
                    # iteration's `torch.maximum(real_max, block_max)` reads
                    # it back -- a pure intra-loop carry with no reader at
                    # all outside the loop). Such a target still needs its
                    # write to land at a fresh per-tile address each
                    # iteration -- otherwise every iteration after the first
                    # reads back the wrong (stale/aliased) tile's value.
                    # There is no separate copy-out to insert (nothing
                    # outside reads it), so mutation_write_back is the right
                    # shape: it sets output_tiled_dims on the op's own write
                    # directly, with no full-buffer allocation.
                    if mut_target is not None:
                        in_loop_carry = any(
                            isinstance(candidate, ComputedBuffer)
                            and candidate is not op
                            and _reads_buffer(candidate, mut_target_name)
                            and name_to_group_outer_key.get(candidate.get_name())
                            == info.loop_group_id[0]
                            for candidate in group_ops
                        )
                        if in_loop_carry:
                            full_ranges = _compute_full_ranges_planned(op, info)
                            info.propagation = PropagationPlan(
                                kind="mutation_write_back",
                                full_ranges=full_ranges,
                                full_strides=tuple(mut_target.layout.stride),
                                is_graph_output=False,
                            )
                            continue
                info.propagation = PropagationPlan(kind="loop_internal")
                continue

            full_ranges = _compute_full_ranges_planned(op, info)
            info.propagation = PropagationPlan(
                kind="copy_out",
                full_ranges=full_ranges,
                full_strides=tuple(op.layout.stride),
                outside_consumer_names=tuple(all_consumer_names),
                is_graph_output=is_graph_output,
            )

    _zero_reads_of_fixed_buffers_planned(operations, plan)


def _zero_reads_of_fixed_buffers_planned(
    operations: list[Operation],
    plan: dict[int, CoarseTileInfo],
) -> None:
    """Planning-time analog of _zero_reads_of_fixed_buffers.

    A buffer is "fixed" -- its own tiled write never advances, because
    something else drains/replaces it every iteration -- once
    _plan_tiling_propagation (just above, in the same call) has decided any
    kind at all for a tiled op: "loop_internal" (nothing advances it),
    "copy_out" (the Pass 3 copy op drains it), or "reduction" (the Pass 2
    combine op drains it, and _propagate_tiled_reduction_op zeroes it
    unconditionally). The reduction case matters here even though the
    accumulator buffer itself is never read by name: a tiled-reduction op's
    *own* buffer (e.g. an amax result feeding a same-loop subtraction, as in
    softmax) IS commonly read by name by a sibling op in the same group, and
    that sibling's tiled_dims_per_read entry for it must be zeroed exactly
    like the loop_internal/copy_out cases. Unlike the deleted
    transformation-time pass, every op's kind is already known for the
    whole plan at this point (computed above, before any mutation), so
    there is no reader-before-producer ordering hazard to work around --
    this always sees the complete, final fixed set on its one and only
    pass.

    "mutation_write_back" is deliberately excluded: that kind means the
    op's own write genuinely advances per iteration (its target -- e.g. a
    flash-attention running max/denominator carry -- is read back by name
    on a later iteration, see the in_loop_carry check above), just via the
    squeezed_advance_output side channel rather than output_tiled_dims
    when the advancing dim divides to per-tile extent 1. Treating such an
    op as "fixed" here would zero its own output_tiled_dims (when that
    happens to be nonempty pre-squeeze) and any sibling's
    tiled_dims_per_read of it, erasing the only signal
    _propagate_mutation_write_back and the scratchpad allocator have for
    routing it away from LX -- see issue #4126.
    """
    fixed_names = {
        op.get_name()
        for op in operations
        if isinstance(op, ComputedBuffer)
        and (info := plan.get(id(op))) is not None
        and info.propagation is not None
        and info.propagation.kind != "mutation_write_back"
        and any(dims for dims in info.loop_tiled_dims)
    }
    if not fixed_names:
        return

    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        info = plan.get(id(op))
        if info is None:
            continue
        if op.get_name() in fixed_names and any(info.output_tiled_dims):
            info.output_tiled_dims = []
        if not info.tiled_dims_per_read:
            continue
        reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
        for i, dep in enumerate(reads):
            if dep.name in fixed_names and info.tiled_dims_per_read[i]:
                # This makes the read address stationary while it names the
                # producer's per-tile scratch.  It does not make the value
                # invariant: that scratch is rewritten every trip, and an
                # outside consumer may later be redirected to a full buffer.
                info.tiled_dims_per_read[i] = []


def _log_propagation_plan(
    groups: list[tuple],
    plan: dict[int, CoarseTileInfo],
) -> None:
    """Checkpoint 1: log the complete plan before any transformation runs.

    Most valuable of the five checkpoints (see the plan/execute split
    design's "Logging checkpoints" section) because it is inspectable even
    if transformation later crashes -- every op's kind is already decided
    here, with zero mutation.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for group_idx, (group_ops, _levels) in enumerate(groups):
        tally: dict[str, int] = {
            "loop_internal": 0,
            "copy_out": 0,
            "reduction": 0,
            "mutation_write_back": 0,
        }
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            info = plan.get(id(op))
            propagation = info.propagation if info is not None else None
            if propagation is None:
                continue
            tally[propagation.kind] += 1
            if propagation.kind == "copy_out":
                logger.debug(
                    "coarse_tile: plan group=%d %s kind=copy_out "
                    "full_ranges=%s consumers=%s graph_output=%s",
                    group_idx,
                    op.get_name(),
                    propagation.full_ranges,
                    propagation.outside_consumer_names,
                    propagation.is_graph_output,
                )
            elif propagation.kind == "mutation_write_back":
                logger.debug(
                    "coarse_tile: plan group=%d %s kind=mutation_write_back "
                    "full_ranges=%s",
                    group_idx,
                    op.get_name(),
                    propagation.full_ranges,
                )
            elif propagation.kind == "reduction":
                reduction = propagation.reduction
                logger.debug(
                    "coarse_tile: plan group=%d %s kind=reduction "
                    "reduction_type=%s is_nested=%s consumers=%s "
                    "graph_output=%s",
                    group_idx,
                    op.get_name(),
                    reduction.reduction_type if reduction else None,
                    reduction.is_nested if reduction else None,
                    propagation.outside_consumer_names,
                    propagation.is_graph_output,
                )
        logger.debug(
            "coarse_tile: plan group=%d tally loop_internal=%d copy_out=%d "
            "reduction=%d mutation_write_back=%d",
            group_idx,
            tally["loop_internal"],
            tally["copy_out"],
            tally["reduction"],
            tally["mutation_write_back"],
        )


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


def _fixed_level_extents(loop_tiled_dims: list[list[int]]) -> list[dict[int, Expr]]:
    """Per-level extents for a dep that is loop-invariant (does not advance).

    The one and only "does not advance" convention in this pipeline is
    *omitting* the dim from its level's dict entirely -- see
    CoarseTileInfo.tiled_dims_per_read's docstring ("An empty per-level list
    means the dep is loop-invariant at that level") and
    SpyreKernel._general_tile_advance, which substitutes 0 for any dep.index
    free symbol with no entry in the level's dict. An extent of
    ``sympy.Integer(1)`` is NOT equivalent: _tiled_dims_for_dep keeps an
    entry whenever the dependency's own index references that dim
    (irrespective of the extent value attached), and
    tiling_expr_to_device_expr has no zero-coefficient special case, so a
    present-with-extent-1 entry still contributes a nonzero
    ``1 * level_symbol`` advance term whenever the dep's index happens to
    reference the dim -- exactly the per-tile-fixed scratch buffers this is
    meant for (issue: read-copy op's own output advancing when it must not).
    Only the empty dict is safe for every dependency, tiled or not.
    """
    return [{} for _ in loop_tiled_dims]


def _advancing_level_extents(
    loop_tiled_dims: list[list[int]],
    loop_count: list[int],
    ranges: list[Expr],
) -> list[dict[int, Expr]]:
    """Per-level extents for a dep that DOES advance across tiled levels.

    For each dim tiled at one or more levels, the innermost tiling level's
    extent is the dim's own (already-divided) range; each level further out
    multiplies by the next-inner level's trip count. This is the same
    per-level formula _planned_tile_extents_per_level's _per_level_extent_for
    uses at planning time, and what _insert_copy_op's write side (which
    targets full_buf, a real full-sized buffer that must advance a whole
    tile per iteration -- see its comment) computes inline. Use this
    whenever a dep addresses a full-sized (not per-tile-scratch) buffer
    whose own dims are loop_tiled_dims; use _fixed_level_extents instead for
    dims/deps that are genuine per-tile scratch reused in place.
    """
    extents: list[dict[int, Expr]] = [{} for _ in loop_tiled_dims]
    for d in {d for level in loop_tiled_dims for d in level}:
        levels_tiling_d = [i for i, dims in enumerate(loop_tiled_dims) if d in dims]
        running = sympy.sympify(ranges[d])
        for level_idx in reversed(levels_tiling_d):
            extents[level_idx][d] = running
            running = running * loop_count[level_idx]
    return extents


def _raw_to_squeezed_pos(ir_node: ComputedBuffer) -> dict[int, int]:
    """Map ir_node's raw host-range positions to their squeezed d{i} number.

    Mirrors SpyreKernel._host_dim_to_index_symbol's own squeeze arithmetic
    (same source of truth, duplicated here because that method maps one
    dim at a time and _tiled_dims_for_dep needs the whole table to build a
    dep_dims membership test): output dims are numbered densely over
    ir_node.data.ranges, skipping unit-size (==1) entries; reduction dims
    continue the same counter, offset by n_output_dims, over
    ir_node.data.reduction_ranges. A raw dim squeezed out entirely (range
    == 1) has no entry -- it has no d{i} symbol for any dep.index to
    reference.
    """
    pos: dict[int, int] = {}
    it_idx = 0
    ranges = getattr(getattr(ir_node, "data", None), "ranges", None)
    if ranges is None:
        return pos
    for host_idx, r in enumerate(ranges):
        if int(r) != 1:
            pos[host_idx] = it_idx
            it_idx += 1
    # Raw reduction-dim keys are stored offset by the RAW (un-squeezed)
    # output-dim count -- see _planned_tile_extents_per_level's
    # n_output_dims = len(op.data.ranges). But the d{i} symbol they map to
    # continues from the SQUEEZED output-dim count (it_idx above) -- see
    # SpyreKernel._host_dim_to_index_symbol's n_output_dims = it_idx. These
    # two offsets differ whenever ranges contains a unit dim, so they must
    # not be conflated.
    raw_n_output_dims = len(ranges)
    squeezed_n_output_dims = it_idx
    reduction_ranges = getattr(ir_node.data, "reduction_ranges", None) or []
    red_it_idx = 0
    for host_idx, r in enumerate(reduction_ranges):
        if int(r) != 1:
            pos[raw_n_output_dims + host_idx] = squeezed_n_output_dims + red_it_idx
            red_it_idx += 1
    return pos


def _tiled_dims_for_dep(
    dep: MemoryDep,
    per_level_extents: list[dict[int, Expr]],
    ir_node: ComputedBuffer,
) -> list[list[tuple[int, Expr]]]:
    """Filter per-level tiled-dim extents down to dims dep.index actually reads.

    A dim tiled at some level that this dependency's index does not depend
    on (broadcast, or simply not one of its dims) must not appear in its
    per-level list -- matching the implicit zeroing _tile_advance_expr_from_dep
    performs today for any free symbol absent from tiled_dim_extents.

    per_level_extents' keys are RAW positional indices into ir_node's own
    host-range space (ir_node.data.ranges / .reduction_ranges) -- the same
    convention every loop_tiled_dims/output_tiled_dims producer in this file
    uses, and the convention SpyreKernel._host_dim_to_index_symbol expects
    on its way out (see its docstring). But dep.index's free symbols are
    minted by Inductor's extract_read_writes -> index_vars_squeeze, which
    drops unit-size dims and renumbers the rest densely -- SQUEEZED-space
    numbers, not raw positions. Comparing a raw key directly against a
    squeezed symbol number silently mismatches whenever a lower-numbered
    dim was squeezed out ahead of it (issue #3613). Translate each raw key
    through ir_node's own raw->squeezed table before the membership test;
    the returned tuples keep the ORIGINAL raw key, since that -- not the
    squeezed number -- is what every caller stores and what
    _host_dim_to_index_symbol re-squeezes for itself later.
    """
    dep_dims = {
        int(str(sym)[1:])
        for sym in dep.index.free_symbols
        if str(sym).startswith("d") and str(sym)[1:].isdigit()
    }
    raw_to_squeezed = _raw_to_squeezed_pos(ir_node)
    return [
        [
            (d, extent)
            for d, extent in level.items()
            if raw_to_squeezed.get(d, d) in dep_dims
        ]
        for level in per_level_extents
    ]


def _predivision_unit_steps_for_dep(
    dep: MemoryDep,
    tiled_dims_per_level: list[list[tuple[int, Expr]]],
    ir_node: ComputedBuffer,
) -> list[list[tuple[int, Expr, Expr]]]:
    """Keep address steps for tiled dimensions that will be squeezed away.

    At planning time the original dimension and its index coefficient still
    exist. After division, a one-element expert tile has neither, so the
    address step cannot be reconstructed reliably from the smaller view.
    """
    raw_to_squeezed = _raw_to_squeezed_pos(ir_node)
    unit_dims = {
        dim for level in tiled_dims_per_level for dim, extent in level if extent == 1
    }
    result: list[list[tuple[int, Expr, Expr]]] = []
    for level in tiled_dims_per_level:
        steps: list[tuple[int, Expr, Expr]] = []
        for dim, extent in level:
            if dim not in unit_dims:
                continue
            squeezed_dim = raw_to_squeezed.get(dim)
            if squeezed_dim is None:
                continue
            symbol = sympy_index_symbol(f"d{squeezed_dim}")
            stride = dep.index.coeff(symbol)
            if stride != 0:
                steps.append((dim, stride, extent))
        result.append(steps)
    return result


def _capture_predivision_unit_steps(
    operations: list[Operation],
    plan: dict[int, CoarseTileInfo],
) -> dict[
    int,
    tuple[tuple[tuple[tuple[int, Expr, Expr], ...], ...], ...],
]:
    """Capture size-one source steps without stamping them onto operations.

    This runs while the original read indexes still contain every dimension.
    The result is deliberately local to the Pass-1 read-copy planner: after
    division, ``_plan_read_copies`` transfers only the selected sizing read's
    fact into its immutable ``ReadCopyEntry``.  General ``CoarseTileInfo`` and
    the transformed IR never carry this temporary observation.
    """

    result: dict[
        int,
        tuple[tuple[tuple[tuple[int, Expr, Expr], ...], ...], ...],
    ] = {}
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        info = plan.get(id(op))
        if info is None:
            continue
        read_deps = [
            dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)
        ]
        result[id(op)] = tuple(
            tuple(
                tuple(level)
                for level in _predivision_unit_steps_for_dep(dep, tiled_dims, op)
            )
            for dep, tiled_dims in zip(read_deps, info.tiled_dims_per_read)
        )
    return result


def _select_unit_steps(
    *,
    op_name: str,
    dep_name: str,
    dim: int,
    planned: list[list[tuple[Expr, Expr]]] | None,
    legacy: list[list[tuple[Expr, Expr]]],
) -> list[list[tuple[Expr, Expr]]]:
    """Use the pre-division fact when present; retain legacy as a check."""
    if planned is None:
        return legacy

    def equivalent() -> bool:
        if len(planned) != len(legacy):
            return False
        return all(
            len(planned_level) == len(legacy_level)
            and all(
                sympy.simplify(planned_stride - legacy_stride) == 0
                and sympy.simplify(planned_extent - legacy_extent) == 0
                for (planned_stride, planned_extent), (
                    legacy_stride,
                    legacy_extent,
                ) in zip(planned_level, legacy_level)
            )
            for planned_level, legacy_level in zip(planned, legacy)
        )

    if any(legacy) and not equivalent():
        logger.warning(
            "coarse_tile: pre-division unit step for %s read %s dim %s "
            "disagrees with legacy reconstruction (planned=%s legacy=%s); "
            "using the pre-division plan",
            op_name,
            dep_name,
            dim,
            planned,
            legacy,
        )
    return planned


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
    # (issue #3888) resolves this via op.loop_info.loop_tiled_dims, but that
    # isn't stamped yet at plan time -- so here we can only tell "ambiguous"
    # from "unambiguous", not which candidate is the real N dim. Treat every
    # candidate whose *tiled* extent exceeds 1 element/tile as excess: the
    # real N dim's own generation loop is not a coarse-tile dim (it's the
    # matmul's inherent output dim, not something plan_coarse_tile_groups
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
                "dimension with 1 element per tile (num_tiles_per_dim == "
                "the dim's full size), or see issue #3927."
            )


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


def _consumers_reading_incomplete_reduction(
    buf_name: str,
    group_ops: list,
    group_op_names: set[str],
    plan: dict,
    group_reduction_tiled_levels: set[int],
) -> list[str]:
    """Same-group consumers of buf_name that themselves read an
    unaccumulated reduction sibling.

    Such a consumer (e.g. softmax's ``div``, which reads ``sum``'s still-
    partial per-tile buffer) would need to be deferred to a separate loop
    nest that runs only after the reduction's own inner loop fully
    accumulates for its read of the reduction to be safe. No such deferral
    mechanism currently exists: _propagate_tiled_reduction_op's
    inside_consumers handling only redirects a same-group consumer to
    accum_full when is_nested is True, or (when flat) when the consumer's
    loop_tiled_dims exactly equals the reduction op's own — which can never
    hold for a consumer like ``div`` that tiles the reduction dim as a real
    output dim. For that flat-and-mismatched shape, _plan_tiling_propagation
    instead raises Unsupported (see the reduction branch's same-group
    consumer check) rather than silently reading a partially-accumulated
    value. A genuine two-pass deferral remains a possible future extension.
    """
    result = []
    for o in group_ops:
        if not isinstance(o, ComputedBuffer) or o.get_name() == buf_name:
            continue
        if not _reads_buffer(o, buf_name):
            continue
        if _reads_incomplete_reduction(
            o, group_ops, group_op_names, plan, group_reduction_tiled_levels
        ):
            result.append(o.get_name())
    return result


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


def reduction_loop_vars(op: ComputedBuffer) -> list[sympy.Symbol]:
    """Return the op's reduction loop variables, ordered as in
    ``op.data.reduction_ranges``.

    Uses dep-tracking symbols (d0, d1, ...) rather than SymT.R0_INDEX symbols
    (r0_0, r0_1, ...) which are a different namespace.  Finds reduction symbols
    by set-subtracting output index symbols from input index symbols, in
    dep.ranges order (which matches reduction_ranges order).

    This is the single source of truth for that derivation. Both directions go
    through it: ``_loop_var_to_reduction_ranges_pos`` (loop_var -> position) and
    coarse tiling's reduction-axis lowering (its inverse, position -> loop_var,
    in ``scratchpad.coarse_tiling.tile_spec_to_dim_hints``).
    """
    assert isinstance(op.data, Reduction)
    rw = op.get_read_writes()
    out_dep = next(iter(rw.writes))
    out_syms = out_dep.index.free_symbols
    in_dep = next(d for d in rw.reads if hasattr(d, "index"))
    return [s for s in in_dep.ranges if s not in out_syms]


def _loop_var_to_reduction_ranges_pos(
    op: ComputedBuffer, sym: sympy.Symbol
) -> int | None:
    """Return position of loop variable sym in op.data.reduction_ranges, or None."""
    try:
        return reduction_loop_vars(op).index(sym)
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


def _capture_logical_iteration_symbols(
    op: ComputedBuffer,
) -> tuple[_LogicalIterationSymbol, ...]:
    """Capture active symbols by raw output/reduction position.

    This identity is valid only for callers that preserve dimension order.
    Unit dimensions have no loop symbol and are deliberately omitted.
    """

    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        raise Unsupported(
            f"coarse_tile: cannot capture iteration symbols for "
            f"{op.get_name()!r} with data type {type(data).__name__}"
        )

    logical_extents: list[tuple[tuple[str, int], Expr]] = [
        (("output", idx), extent) for idx, extent in enumerate(data.ranges)
    ]
    if isinstance(data, Reduction):
        logical_extents.extend(
            (("reduction", idx), extent)
            for idx, extent in enumerate(data.reduction_ranges)
        )

    active = [
        (logical_dim, extent)
        for logical_dim, extent in logical_extents
        if sympy.sympify(extent) != 1
    ]
    symbols = tuple(iteration_space_from_op(op))
    if len(active) != len(symbols):
        raise Unsupported(
            f"coarse_tile: cannot match logical dimensions to iteration symbols "
            f"for {op.get_name()!r}: logical_dimensions={active}, "
            f"iteration_symbols={symbols}"
        )

    return tuple(
        _LogicalIterationSymbol(logical_dim, extent, symbol)
        for (logical_dim, extent), symbol in zip(active, symbols)
    )


def _order_preserving_symbol_remap(
    op: ComputedBuffer,
    before: tuple[_LogicalIterationSymbol, ...],
    after: tuple[_LogicalIterationSymbol, ...],
) -> _IterationSymbolRemap:
    """Return the surviving old-to-new symbols for an order-preserving rewrite."""

    before_by_dim = {entry.logical_dim: entry for entry in before}
    after_by_dim = {entry.logical_dim: entry for entry in after}
    after_dims = tuple(entry.logical_dim for entry in after)
    surviving_dims = tuple(
        entry.logical_dim for entry in before if entry.logical_dim in after_by_dim
    )

    pairs = tuple(
        (before_by_dim[logical_dim].symbol, after_by_dim[logical_dim].symbol)
        for logical_dim in surviving_dims
    )
    monotone = (
        surviving_dims == after_dims
        and len({old for old, _ in pairs}) == len(pairs)
        and len({new for _, new in pairs}) == len(pairs)
    )
    if not monotone:
        raise Unsupported(
            f"coarse_tile: order-preserving dimension mapping failed for "
            f"{op.get_name()!r}: old_dimensions={before}, "
            f"new_dimensions={after}, attempted_mapping={pairs}"
        )

    return _IterationSymbolRemap(
        before_symbols=tuple(entry.symbol for entry in before), pairs=pairs
    )


def _fused_iteration_symbol_remap(
    op: ComputedBuffer,
    before_symbols: tuple[sympy.Symbol, ...],
    *,
    max_trailing_removals: int,
) -> _IterationSymbolRemap:
    """Preserve names when fused-loop symbols demonstrably retain their identity.

    ``_capture_logical_iteration_symbols`` cannot describe a many-logical-dims to
    one-loop-symbol fusion.  For that case, accept only the narrower invariant we
    can still prove: the post-rewrite symbols are an unchanged prefix of the old
    symbols.  Reduction variables are appended to the iteration space, so a
    reduction-range rewrite may additionally remove a known number of trailing
    symbols.  Output-range rewrites pass zero and therefore cannot silently
    reinterpret a renumbered surviving dimension.
    """

    after_symbols = tuple(iteration_space_from_op(op))
    removed = len(before_symbols) - len(after_symbols)
    if (
        removed < 0
        or removed > max_trailing_removals
        or before_symbols[: len(after_symbols)] != after_symbols
    ):
        raise Unsupported(
            f"coarse_tile: cannot safely preserve fused work-division symbols "
            f"for {op.get_name()!r}: before={tuple(map(str, before_symbols))}, "
            f"after={tuple(map(str, after_symbols))}"
        )

    return _IterationSymbolRemap(
        before_symbols=before_symbols,
        pairs=tuple((symbol, symbol) for symbol in after_symbols),
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


def _divide_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> _DivideRangesResult:
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
        return _DivideRangesResult(None, None)

    ranges = list(data.ranges)
    if not ranges:
        return _DivideRangesResult(None, None)

    before_symbols = None
    fused_before_symbols = None
    if tiled_dims and hasattr(op, "work_div_loop_info"):
        try:
            before_symbols = _capture_logical_iteration_symbols(op)
        except Unsupported:
            fused_before_symbols = tuple(iteration_space_from_op(op))

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

    symbol_remap = None
    if before_symbols is not None or fused_before_symbols is not None:
        # Both capture paths call the memoized iteration-space helper before
        # ranges are rewritten.  Always invalidate it before observing the new
        # iteration space, including the fused-dimension fallback.
        invalidate_op_read_writes(op)
        if before_symbols is not None:
            symbol_remap = _order_preserving_symbol_remap(
                op, before_symbols, _capture_logical_iteration_symbols(op)
            )
        else:
            assert fused_before_symbols is not None
            symbol_remap = _fused_iteration_symbol_remap(
                op, fused_before_symbols, max_trailing_removals=0
            )

    # Sync layout.size, layout.stride, and layout.device_layout with the new ranges.
    layout = getattr(op, "layout", None)
    if not (isinstance(layout, FixedLayout) and len(layout.size) == len(ranges)):
        return _DivideRangesResult(None, symbol_remap)

    old_stride = tuple(layout.stride)
    old_size = tuple(layout.size)
    new_size = list(layout.size)
    for i in tiled_dims:
        new_size[i] = ranges[i]

    # Recompute strides for the smaller buffer preserving the order of dimensions
    layout.stride = compute_tile_stride(layout.size, old_stride, new_size)

    layout.size = new_size

    # Invalidate Layout- and ComputedBuffer-level caches that read size/stride.
    _clear_cache(layout, _LAYOUT_FREE_SYMS_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)
    retiled_info = (
        _RetiledBufferInfo(old_stride, tuple(layout.stride), old_size)
        if tiled_dims and old_stride != tuple(layout.stride)
        else None
    )

    # Rebuild SpyreTensorLayout for the new host size using device-native
    # reconstruction: transform the original device layout directly without
    # guessing a dim_order.
    if not isinstance(layout, FixedTiledLayout):
        return _DivideRangesResult(retiled_info, symbol_remap)
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
    return _DivideRangesResult(retiled_info, symbol_remap)


def _divide_reduction_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> _IterationSymbolRemap | None:
    """Divide the specified reduction_ranges entries of op by loop_count.

    Unlike _divide_ranges, does NOT update op.layout.size/stride — the
    output buffer shape is determined by data.ranges (non-reduction dims)
    and is unchanged by reduction-dim tiling.
    """
    data = op.data
    assert isinstance(data, Reduction)
    if not tiled_dims:
        return None
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
    if before_symbols is None and fused_before_symbols is None:
        return None

    invalidate_op_read_writes(op)
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
            divide_result = _divide_ranges(op, count, opos_list)
            _apply_work_div_symbol_remap(op, divide_result.symbol_remap)
            retiled_info = divide_result.retiled_info
            if retiled_info is not None:
                name = op.get_name()
                prior = retiled_infos.get(name)
                retiled_infos[name] = (
                    _RetiledBufferInfo(
                        prior.old_stride, retiled_info.new_stride, prior.old_size
                    )
                    if prior is not None
                    else retiled_info
                )
            if isinstance(op.data, Reduction):
                reduction_remap = _divide_reduction_ranges(op, count, rpos_list)
                _apply_work_div_symbol_remap(op, reduction_remap)

        op.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
            info, loop_group_id=stamped_group_id
        )

        logger.debug(
            "coarse_tile: applied plan for %s loop_group_id=%s",
            op.get_operation_name(),
            stamped_group_id,
        )

    return retiled_infos


def coarse_tile_pre_stickify(
    graph: GraphLowering,
    groups: list[tuple],
    group_idx_offset: int = 0,
) -> None:
    """Hint-driven coarse tiling.  Runs PRE-stickification.

    Parameters
    ----------
    graph:
        Provides ``operations``, the full ordered list of IR operations (as
        seen by CustomPreSchedulingPasses).  Modified in-place when the
        transformation phase inserts new buffer/copy ops.
    groups:
        Sequence of ``(ops, levels)`` tuples produced by
        ``hints_to_coarse_tile_groups``.  ``levels`` is a list of
        ``(hint_id, count)`` pairs, outermost first.
    group_idx_offset:
        Starting index for group IDs assigned to the first group.  Use this
        when making a second call on the same graph so that the new group
        IDs do not collide with IDs already stamped by an earlier call.

    Plans and inserts read copy-ins (Pass 1), reduction machinery (Pass 2),
    and write copy-outs (Pass 3). See coarse_tile_post_stickify for the
    post-stickification counterpart, which never needs Pass 1.
    """
    _coarse_tile_common(graph, groups, group_idx_offset, run_read_copies=True)


def coarse_tile_post_stickify(
    graph: GraphLowering,
    groups: list[tuple],
    group_idx_offset: int = 0,
) -> None:
    """Span-overflow coarse tiling.  Runs POST-stickification.

    Parameters
    ----------
    graph:
        Provides ``operations``, the full ordered list of IR operations (as
        seen by CustomPreSchedulingPasses).  Modified in-place when the
        transformation phase inserts new buffer/copy ops.
    groups:
        Sequence of ``(ops, levels)`` tuples produced by
        ``span_overflow_groups``.  ``levels`` is a list of
        ``(hint_id, count)`` pairs, outermost first.
    group_idx_offset:
        Starting index for group IDs assigned to the first group.  Use this
        so span-overflow group IDs do not collide with any hint-driven
        groups already stamped by an earlier coarse_tile_pre_stickify call.

    Every op's device layout is already committed by layout propagation by
    the time this runs, so Pass 1 (read copy-ins) is skipped
    unconditionally: a read-copy here would only produce an HBM-to-HBM copy
    with no layout-reconciliation benefit. See coarse_tile_pre_stickify for
    the pre-stickification counterpart.
    """
    _coarse_tile_common(graph, groups, group_idx_offset, run_read_copies=False)


def _coarse_tile_common(
    graph: GraphLowering,
    groups: list[tuple],
    group_idx_offset: int,
    run_read_copies: bool,
) -> None:
    """Plan then transform: stamp loop_group_id / loop_count and scale ranges.

    Shared plan-then-transform body for both stickify entry points --
    run_read_copies is an internal-only switch (never exposed publicly) so
    the two ~10-step orchestration bodies aren't duplicated. See
    coarse_tile_pre_stickify/coarse_tile_post_stickify for the two public
    entry points that call this.
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
    # A source dimension tiled to extent one disappears when _apply_plan
    # divides the operation.  Capture that one fact now, but keep it outside
    # CoarseTileInfo and the IR: Pass 1 is its only consumer and will attach
    # the selected read's value to the corresponding ReadCopyEntry.
    predivision_unit_steps_by_op = (
        _capture_predivision_unit_steps(operations, plan) if run_read_copies else {}
    )

    # Planning continued: decide every op's propagation kind (loop-internal
    # / copy-out / reduction) with zero mutation, consumed by Pass 1/2/3
    # below.
    _plan_tiling_propagation(operations, groups, plan)
    _log_propagation_plan(groups, plan)

    # Transformation: apply the plan. Only reached if planning didn't raise.
    retiled_infos_by_group: list[
        tuple[tuple[int, ...], list[Operation], dict[str, _RetiledBufferInfo]]
    ] = []
    for group_idx, (group_ops, levels) in enumerate(groups, start=group_idx_offset):
        group_id: tuple[int, ...] = (group_idx,)
        op_to_position = {op.get_operation_name(): i for i, op in enumerate(operations)}
        stamped_group_id = group_id + (0,) * (len(levels) - 1)
        retiled_infos = _apply_plan(
            group_ops, stamped_group_id, levels, op_to_position, plan
        )
        retiled_infos_by_group.append((stamped_group_id, group_ops, retiled_infos))

    # Pass 1: read copy-ins. _plan_read_copies runs here (after every
    # group's _apply_plan above, not alongside _plan_tiling_propagation)
    # because it needs op.loop_info stamped and ranges already divided --
    # see _plan_read_copies's own docstring. Skipped entirely when
    # run_read_copies is False (the post-stickify call site, where layout
    # propagation already ran and a read-copy buys nothing).
    if run_read_copies:
        read_copy_plans = _plan_read_copies(
            operations,
            retiled_infos_by_group,
            predivision_unit_steps_by_op,
        )
        _insert_all_read_copy_ops(operations, read_copy_plans)

    # Pass 2: reduction machinery (accumulator/fill/combine), using each
    # op's now-stamped loop_info.propagation.reduction. Must run after Pass
    # 1 (a tiled-reduction op may itself have needed a read copy-in) and
    # before Pass 3 -- a reduction op is never also copy_out (the plan's
    # kind routes each op to exactly one).
    _insert_all_reduction_ops(operations)

    # Pass 3: write copy-outs (full buffer + copy op + outside-consumer/
    # graph-output patching), using each op's now-stamped
    # loop_info.propagation.full_ranges/outside_consumer_names/
    # is_graph_output. Must run after Pass 1/2 -- _allocate_full_buffer/
    # _insert_copy_op read op's *current* reads/loader/layout.
    _insert_all_write_copy_ops(operations)

    # Checkpoint 5 wants each op's *planned* kind by name -- snapshot that
    # now, before the resync loop below overwrites `group_ops` (the same
    # list objects `groups` holds references to) with post-transformation
    # replacement objects, which would make the id(op)-keyed `plan` lookup
    # miss every replaced op.
    predicted_kind_by_name: dict[str, str] = {
        op.get_name(): info.propagation.kind
        for group_ops, _levels in groups
        for op in group_ops
        if isinstance(op, ComputedBuffer)
        and (info := plan.get(id(op))) is not None
        and info.propagation is not None
        and info.propagation.kind in ("copy_out", "reduction", "mutation_write_back")
    }

    # Pass 1/2/3 (all above) may have spliced a replacement ComputedBuffer
    # into `operations` under the same name for any op in a group's
    # `group_ops` snapshot (taken before those passes ran, back when
    # retiled_infos_by_group was built) -- e.g. a read copy-in redirect
    # (Pass 1) or a copy-out's output_tiled_dims zeroing that happens to
    # accompany a body rewrite. _patch_retiled_load_indexes must see each
    # op's *current* inner_fn/loop_info to decide whether it still needs
    # patching, so re-resolve every entry by name from `operations` (the
    # authoritative post-replacement list) before calling it -- the same
    # by-name resync idiom used throughout this module (see PropagationPlan's
    # docstring on name stability).
    name_to_op = {
        op.get_name(): op for op in operations if isinstance(op, ComputedBuffer)
    }
    for group_id, group_ops, retiled_infos in retiled_infos_by_group:
        for idx, op in enumerate(group_ops):
            if not isinstance(op, ComputedBuffer):
                continue
            group_ops[idx] = name_to_op.get(op.get_name(), op)
        _patch_retiled_load_indexes(group_id, group_ops, retiled_infos, operations)

    _log_propagation_self_check(operations, predicted_kind_by_name)
    validate_writer_tile_advance(operations)
    validate_reader_tile_advance(operations)


def validate_writer_tile_advance(operations: list[Operation]) -> None:
    """Every synthesized cross-loop writer must advance at each tiled level.

    For each op the plan routed to "copy_out" or nested "reduction", the
    real write into the full-sized output buffer happens in a synthesized
    copy op (`coarse_tile_copy_{name}` / `coarse_tile_reduce_copy_{name}`),
    never on the original op itself -- both _propagate_tiled_op and
    _propagate_tiled_reduction_op deliberately zero the original op's own
    `output_tiled_dims` (it is per-tile scratch, redrawn every iteration).
    If the synthesized copy's own `output_tiled_dims` is missing a level
    that its `loop_tiled_dims` says it tiles, that copy's write pointer
    would not advance at that level -- every tile after the first would
    land on top of tile 0 (the exact bug this function is named for; see
    _insert_reduction_copy_op's fix for a concrete instance).  A flat
    (non-nested) reduction has no synthesized copy at all: accum_full is
    written directly by the combine op, which by construction never
    advances (see _insert_combine_op) since a flat reduction has no outer
    output-dim tiling level to advance across.
    """
    name_to_op = {
        op.get_name(): op for op in operations if isinstance(op, ComputedBuffer)
    }
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        propagation = getattr(getattr(op, "loop_info", None), "propagation", None)
        if propagation is None:
            continue
        buf_name = op.get_name()
        if propagation.kind == "copy_out":
            writer_name = f"coarse_tile_copy_{buf_name}"
            writer = name_to_op.get(writer_name)
            if writer is None:
                continue
            writer_info = writer.loop_info  # type: ignore[attr-defined]
        elif propagation.kind == "reduction" and propagation.reduction.is_nested:
            writer_name = f"coarse_tile_reduce_copy_{buf_name}"
            writer = name_to_op.get(writer_name)
            if writer is None:
                continue
            writer_info = writer.loop_info  # type: ignore[attr-defined]
        elif propagation.kind == "mutation_write_back":
            # The op itself IS the writer — check its own output_tiled_dims.
            writer_info = op.loop_info  # type: ignore[attr-defined]
        else:
            continue
        output_tiled_dims = writer_info.output_tiled_dims
        squeezed_advance_output = (
            getattr(writer_info, "squeezed_advance_output", None) or []
        )
        for level_idx, tiled_dims in enumerate(writer_info.loop_tiled_dims):
            if not tiled_dims:
                continue
            level_extents = (
                output_tiled_dims[level_idx]
                if level_idx < len(output_tiled_dims)
                else []
            )
            squeezed_level_extents = (
                squeezed_advance_output[level_idx]
                if level_idx < len(squeezed_advance_output)
                else []
            )
            if not level_extents and not squeezed_level_extents:
                raise RuntimeError(
                    f"coarse_tile: writer-advance check failed for "
                    f"{writer_name!r} -- level {level_idx} tiles output dims "
                    f"{tiled_dims} but output_tiled_dims has no extents for "
                    f"that level, so its write pointer would not advance "
                    f"there."
                )


def validate_reader_tile_advance(operations: list[Operation]) -> None:
    """No op may read a tiled-reduction op's own (per-tile scratch) buffer.

    A Reduction op tiled over a reduction dim writes per-tile partial
    results into its own buffer every inner iteration -- that buffer is
    drained by the combine op and is never fully accumulated except at the
    very last inner iteration.  Any op other than the combine/reduce-copy
    machinery that reads it with a non-empty `tiled_dims_per_read` entry
    would advance alongside it and observe a partially-accumulated value
    for every iteration but the last -- silently wrong numerics.  True
    outside consumers are already redirected by _patch_consumers to read
    accum_full instead (see _propagate_tiled_reduction_op), and legitimate
    inside siblings get a structurally-empty tiled_dims_per_read entry for
    this buffer (squeeze of the collapsed reduction dim, or explicit
    zeroing by _zero_reads_of_fixed_buffers_planned) -- so this function
    asserts that invariant holds rather than establishing it.
    """
    reduction_names = set()
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        propagation = getattr(getattr(op, "loop_info", None), "propagation", None)
        if propagation is not None and propagation.kind == "reduction":
            reduction_names.add(op.get_name())
    if not reduction_names:
        return
    allowed_reader_prefixes = ("coarse_tile_combine_", "coarse_tile_reduce_copy_")
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        reader_name = op.get_name()
        if reader_name.startswith(allowed_reader_prefixes):
            continue
        if reader_name in reduction_names:
            continue
        try:
            reads = [
                dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)
            ]
        except Exception as e:
            # This validator exists to catch otherwise-silent wrong numerics,
            # so silently skipping an op whose deps couldn't even be computed
            # would itself be a blind spot -- log at warning, not debug.
            logger.warning(
                "validate_reader_tile_advance: get_read_writes() raised for %s: %s",
                reader_name,
                e,
            )
            continue
        loop_info = getattr(op, "loop_info", None)
        tiled_dims_per_read = getattr(loop_info, "tiled_dims_per_read", None) or []
        for dep_idx, dep in enumerate(reads):
            if getattr(dep, "name", None) not in reduction_names:
                continue
            level_extents = (
                tiled_dims_per_read[dep_idx]
                if dep_idx < len(tiled_dims_per_read)
                else []
            )
            if any(level_extents):
                raise RuntimeError(
                    f"coarse_tile: reader-advance check failed -- "
                    f"{reader_name!r} reads tiled-reduction op {dep.name!r}'s "
                    f"own per-tile scratch buffer with a non-empty "
                    f"tiled_dims_per_read entry {level_extents}, so it would "
                    f"observe a partially-accumulated value on every "
                    f"iteration but the last."
                )


def _log_propagation_self_check(
    operations: list[Operation],
    predicted_kind_by_name: dict[str, str],
) -> None:
    """Checkpoint 5: per-op cross-check of actual new buffers against the plan.

    For every op the plan routed to "copy_out" or "reduction", check by name
    that the buffers its kind requires actually exist in `operations`:
    copy_out needs a "coarse_tile_copy_{name}"; reduction needs both a
    "coarse_tile_fill_{name}" and a "coarse_tile_combine_{name}" (nested
    reduction additionally needs a "coarse_tile_reduce_copy_{name}", but that
    is not checked here since a missing outer-level copy would already
    surface as wrong numerics, not a silently-dropped buffer). This is a
    per-op existence check rather than a plan-vs-actual aggregate tally,
    because a coarse count comparison (e.g. counting fill buffers alone as a
    proxy for "reduction machinery is complete") cannot distinguish "combine
    op silently dropped" from "no bug" -- the count of fill buffers alone is
    unaffected by a missing combine op.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    existing_names = {
        op.get_name() for op in operations if isinstance(op, ComputedBuffer)
    }
    mismatches = []
    for name, kind in predicted_kind_by_name.items():
        if kind == "copy_out":
            required = [f"coarse_tile_copy_{name}"]
        elif kind == "mutation_write_back":
            # No synthesized buffers — the op itself IS the writer.
            required = []
        else:
            required = [f"coarse_tile_fill_{name}", f"coarse_tile_combine_{name}"]
        missing = [r for r in required if r not in existing_names]
        if missing:
            mismatches.append((name, kind, missing))

    predicted_copy_out = sum(
        1 for k in predicted_kind_by_name.values() if k == "copy_out"
    )
    predicted_reduction = sum(
        1 for k in predicted_kind_by_name.values() if k == "reduction"
    )
    predicted_mutation_write_back = sum(
        1 for k in predicted_kind_by_name.values() if k == "mutation_write_back"
    )
    logger.debug(
        "coarse_tile: self-check predicted copy_out=%d reduction=%d "
        "mutation_write_back=%d, %d mismatches",
        predicted_copy_out,
        predicted_reduction,
        predicted_mutation_write_back,
        len(mismatches),
    )
    if mismatches:
        logger.warning(
            "coarse_tile: propagation self-check mismatch -- %d op(s) "
            "missing their planned buffers: %s",
            len(mismatches),
            mismatches,
        )


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

    Called from plan_coarse_tile_groups (planning time): tiled_dims /
    tiled_rdims are the op's own per-level lists computed there, before any
    loop_info is stamped -- this check is a pure function of already-known
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


def _insert_all_write_copy_ops(operations: list[Operation]) -> None:
    """Pass 3: build full buffer + copy-out for every planned copy-out op.

    Transformation's Pass 3 (see the plan/execute split design). Every op
    was already stamped by _apply_plan with a loop_info carrying
    .propagation, computed by _plan_tiling_propagation -- this pass only
    consumes that decision (kind == "copy_out" and its accompanying
    full_ranges/outside_consumer_names/is_graph_output data), it makes no
    new ones. A "loop_internal" op needs nothing here: planning already
    determined it has no outside consumers, so its output_tiled_dims is
    left as _apply_plan stamped it (a loop-internal op's write is never
    tiled -- see _plan_tiling_propagation).

    Must run after Pass 1 (_insert_all_read_copy_ops) and Pass 2
    (_insert_all_reduction_ops) -- a copy-out op may itself have needed a
    read copy-in, and _allocate_full_buffer/_insert_copy_op read op's
    *current* reads/loader/layout.
    """
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        loop_info = getattr(op, "loop_info", None)
        propagation = getattr(loop_info, "propagation", None)
        if propagation is None:
            continue
        if propagation.kind == "copy_out":
            _propagate_tiled_op(op, propagation, operations)
        elif propagation.kind == "mutation_write_back":
            _propagate_mutation_write_back(op, propagation)


def _propagate_mutation_write_back(
    op: ComputedBuffer,
    propagation: PropagationPlan,
) -> None:
    """Set output_tiled_dims on a mutation_write_back op.

    The op already carries MutationLayoutSHOULDREMOVE targeting a
    graph-output buffer (e.g. copy_forced(src, acc) where acc is both a graph
    input and the graph output).  Its MutationLayout write IS the cross-tile
    write-back -- no separate copy op is needed, no full buffer allocation,
    no graph-output patching.

    The only missing piece is output_tiled_dims: _apply_plan left it as []
    because the op looked loop_internal to the planner (its own buffer name
    was not in graph outputs).  We set it now using the same
    write_level_extents math _insert_copy_op uses for its copy-out write
    side: the op's data.ranges are already divided, so the per-level extent
    for each tiled dim is the divided range times the product of all
    inner-level trip counts.
    """
    loop_info = op.loop_info  # type: ignore[attr-defined]
    op_ranges = list(op.data.ranges)  # already divided by _apply_plan
    mut_target = op.layout.get_buffer()  # type: ignore[attr-defined]
    full_sizes = list(mut_target.get_size())

    # A raw dim tiled to per-tile extent 1 is squeezed out of op_ranges
    # entirely -- index_vars_squeeze mints no d{i} symbol for it, so
    # _tiled_dims_for_dep's dep_dims membership test always drops it no
    # matter what extent we compute below (issue #4126's real_max/
    # denominator carries hit exactly this: their B-tile dim divides to
    # extent 1). Mirror _insert_copy_op's squeezed_advance_output
    # construction here: for each such dim, record (host_stride, extent)
    # pairs per tiling level, independent of dep.index's free symbols, so
    # SpyreKernel._general_tile_advance can add the device-address
    # contribution as an extra term instead of by substitution.
    squeeze_pos: dict[int, int] = {}
    it_idx = 0
    for host_idx, r in enumerate(op_ranges):
        if int(r) != 1:
            squeeze_pos[host_idx] = it_idx
            it_idx += 1

    write_level_extents: list[dict[int, sympy.Expr]] = [
        {} for _ in loop_info.loop_tiled_dims
    ]
    squeezed_advance: list[list[tuple[sympy.Expr, sympy.Expr]]] = [
        [] for _ in loop_info.loop_tiled_dims
    ]
    for d in {d for level in loop_info.loop_tiled_dims for d in level}:
        levels_tiling_d = [
            i for i, dims in enumerate(loop_info.loop_tiled_dims) if d in dims
        ]
        if d not in squeeze_pos:
            # Squeezed out of op's own write -- use mut_target's own
            # (undivided) sizes for the host_stride, not op_ranges (a dim
            # to the right that is itself tiled has already been divided
            # down in op_ranges, which would undercount the stride).
            host_stride = sympy.prod(full_sizes[d + 1 :])
            running = sympy.Integer(1)
            for level_idx in reversed(levels_tiling_d):
                squeezed_advance[level_idx].append((host_stride, running))
                running = running * loop_info.loop_count[level_idx]
            continue
        running = sympy.sympify(op_ranges[d])
        for level_idx in reversed(levels_tiling_d):
            write_level_extents[level_idx][d] = running
            running = running * loop_info.loop_count[level_idx]

    write_deps = [
        dep for dep in op.get_read_writes().writes if isinstance(dep, MemoryDep)
    ]
    output_tiled_dims = (
        _tiled_dims_for_dep(write_deps[0], write_level_extents, op)
        if write_deps
        else []
    )
    loop_info.output_tiled_dims = output_tiled_dims
    loop_info.squeezed_advance_output = squeezed_advance if write_deps else []

    logger.debug(
        "coarse_tile: mutation_write_back %s output_tiled_dims=%s "
        "squeezed_advance_output=%s",
        op.get_name(),
        output_tiled_dims,
        loop_info.squeezed_advance_output,
    )


def _propagate_tiled_op(
    op: ComputedBuffer,
    propagation: PropagationPlan,
    operations: list[Operation],
) -> None:
    """Allocate a full buffer + copy-out for a single planned copy-out op."""
    loop_info = op.loop_info
    loop_group_id = loop_info.loop_group_id
    buf_name = op.get_name()
    # A MutationLayoutSHOULDREMOVE op whose mutation target (not the op's
    # own buffer) is what outside ops actually read -- e.g.
    # copy_forced(src, c) where c is read later -- must have its consumers
    # re-resolved against the target's name; see PropagationPlan's
    # consumer_lookup_name docstring.
    read_lookup_name = propagation.consumer_lookup_name or buf_name

    # Resolve consumers at TRANSFORM time, by actual reads rather than by
    # the planning-time name list. Pass 1/2 may have spliced replacements
    # into `operations` under the same names since planning ran (see
    # PropagationPlan's docstring on name stability) -- and Pass 1 may have
    # rewired a planned consumer in another tiled group through a read-copy
    # staging op, which then performs the group's actual read of
    # read_lookup_name. Patching only the planned names would miss that
    # staging op, leaving it draining this op's per-tile scratch while the
    # full buffer goes unread (issue #4008: 94.6% wrong on two chained hint
    # groups). Any current reader outside this op's outermost loop group
    # needs the redirect; the in-group copy-out drain reads read_lookup_name
    # by design and is excluded by the group test exactly like the
    # planning-time analog (_find_outside_consumers_planned).
    own_outer_key = loop_group_id[0]
    planned_names = set(propagation.outside_consumer_names)
    outside_consumers = []
    for o in operations:
        if not isinstance(o, ComputedBuffer) or o is op:
            continue
        if not _reads_buffer(o, read_lookup_name):
            continue
        o_outer = getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        # Union of both consumer notions: a planned name that still reads
        # buf_name may legitimately share this group's outer key (a deferred
        # reduction consumer such as softmax's div - see
        # _consumers_reading_incomplete_reduction), so the group test alone
        # would wrongly drop it.
        if o_outer != own_outer_key or o.get_name() in planned_names:
            outside_consumers.append(o)
    resolved_names = {o.get_name() for o in outside_consumers}
    if resolved_names != planned_names:
        logger.debug(
            "coarse_tile: copy-out %s consumer set changed between planning "
            "and transform: planned=%s resolved=%s (read-copy staging ops "
            "take over their consumer's read)",
            buf_name,
            sorted(planned_names),
            sorted(resolved_names),
        )
    is_graph_output = propagation.is_graph_output

    full_ranges = propagation.full_ranges
    assert full_ranges is not None, "full_ranges must be planned for copy_out ops"
    full_strides = propagation.full_strides
    assert full_strides is not None, "full_strides must be planned for copy_out ops"

    # Insert the full buffer before the first op in the same outermost
    # loop group so it doesn't split the group's contiguous run in the
    # operations list.
    outer_key = loop_group_id[0]
    group_start_idx = next(
        i
        for i, o in enumerate(operations)
        if isinstance(o, ComputedBuffer)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
    )
    full_buf = _allocate_full_buffer(
        op, full_ranges, full_strides, operations, group_start_idx
    )

    # Capture before _insert_copy_op overwrites op.layout.
    old_stride = tuple(op.layout.stride)
    old_size = tuple(op.layout.size)

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
    _insert_copy_op(
        op,
        full_buf,
        operations,
        tiled_op_write_advances=propagation.consumer_lookup_name is not None,
    )
    if propagation.consumer_lookup_name is not None:
        # op.layout is MutationLayoutSHOULDREMOVE targeting a pre-existing,
        # locally-created buffer (e.g. copy_forced(src, acc) where acc is a
        # loop-carried accumulator also read later, by name, inside the SAME
        # loop group -- flash-attention's real_max/denominator/output). That
        # in-loop read already advances per tile (it goes through the normal
        # read-copy machinery keyed on the mutation target's name), so the
        # direct write into the target must ALSO advance per tile to stay
        # consistent with it -- leaving it at [] silently pins every tile's
        # write to the same address, so tile 1+ never actually lands and the
        # next iteration's read of the "accumulator" reads back tile 0's
        # value every time. Use the same write_level_extents math
        # _propagate_mutation_write_back uses for its own direct write.
        write_deps = [
            dep for dep in op.get_read_writes().writes if isinstance(dep, MemoryDep)
        ]
        if write_deps:
            op_ranges = list(op.data.ranges)
            # A raw dim tiled to per-tile extent 1 (e.g. flash attention's
            # B-tile dim on real_max/denominator/output) is squeezed out of
            # op_ranges entirely -- no d{i} symbol survives for it, so
            # _tiled_dims_for_dep always drops it below no matter what
            # extent write_level_extents carries. Mirror
            # _propagate_mutation_write_back's squeezed_advance_output
            # construction: use full_ranges (op's own undivided sizes, same
            # raw dim order) for the host_stride of such dims, independent
            # of dep.index's free symbols.
            squeeze_pos: dict[int, int] = {}
            it_idx = 0
            for host_idx, r in enumerate(op_ranges):
                if int(r) != 1:
                    squeeze_pos[host_idx] = it_idx
                    it_idx += 1
            write_level_extents: list[dict[int, Expr]] = [
                {} for _ in loop_info.loop_tiled_dims
            ]
            squeezed_advance: list[list[tuple[Expr, Expr]]] = [
                [] for _ in loop_info.loop_tiled_dims
            ]
            for d in {d for level in loop_info.loop_tiled_dims for d in level}:
                levels_tiling_d = [
                    i for i, dims in enumerate(loop_info.loop_tiled_dims) if d in dims
                ]
                if d not in squeeze_pos:
                    host_stride = sympy.prod(list(full_ranges)[d + 1 :])
                    running = sympy.Integer(1)
                    for level_idx in reversed(levels_tiling_d):
                        squeezed_advance[level_idx].append((host_stride, running))
                        running = running * loop_info.loop_count[level_idx]
                    continue
                running = sympy.sympify(op_ranges[d])
                for level_idx in reversed(levels_tiling_d):
                    write_level_extents[level_idx][d] = running
                    running = running * loop_info.loop_count[level_idx]
            loop_info.output_tiled_dims = _tiled_dims_for_dep(
                write_deps[0], write_level_extents, op
            )
            loop_info.squeezed_advance_output = squeezed_advance
        else:
            loop_info.output_tiled_dims = []
    else:
        # The tiled op's own buffer is loop-internal scratch here: it is
        # fully drained by the copy op inserted above before the next
        # iteration overwrites it, so its own write must not advance at any
        # level.
        loop_info.output_tiled_dims = []

    # Patch outside consumers and graph outputs to read full_buf.
    full_name = full_buf.get_name()
    retile_info = _RetiledBufferInfo(
        old_stride, tuple(full_buf.layout.stride), old_size
    )
    _patch_consumers(
        outside_consumers, read_lookup_name, full_name, operations, retile_info
    )
    if is_graph_output:
        _patch_graph_outputs(propagation.graph_output_name or buf_name, full_buf)

    logger.debug(
        "coarse_tile: write copy-out %s -> %s old_stride=%s new_stride=%s "
        "consumers=%s graph_output=%s",
        buf_name,
        full_name,
        old_stride,
        tuple(full_buf.layout.stride),
        [c.get_name() for c in outside_consumers],
        is_graph_output,
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
    op).

    Graph inputs (InputBuffer, including ConstantBuffer) are always
    full-extent and undivided, exactly like a SpyreEmptyFallback accumulator
    -- there is no loop_info question to ask, so they are always included.
    An older version of this docstring argued these could be skipped because
    insert_restickify's AllSameNode path "already reconciles a full-extent
    input against a tile-sized read." That reasoning predates the decision
    (see _insert_copy_op) to unconditionally insert a copy across any
    loop-group boundary that changes size (full <-> tile), and does not hold
    up under it: restickify only reconciles device layout/strides via a
    fresh spyre.restickify op; it never rewrites a consumer's *index
    expression*. A tiled op's own load index for a given read is sized to
    its tile regardless of what the producer is, so a direct read of a
    graph input inside the loop body is evaluated with a tile-scoped index
    against a full-size buffer -- the same indexing bug _insert_copy_op's
    write side had before that fix, just on the read side and against an
    external buffer instead of a freshly allocated one. See
    _insert_all_read_copy_ops and _find_outside_consumers (same outer-key
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
        # Graph inputs are TensorBox(StorageBox(InputBuffer))-wrapped in
        # V.graph.get_buffer's result (see graph_inputs); unwrap to check.
        unwrapped = buf
        if isinstance(unwrapped, TensorBox):
            unwrapped = unwrapped.data
        if isinstance(unwrapped, StorageBox):
            unwrapped = unwrapped.data
        if isinstance(unwrapped, (SpyreEmptyFallback, InputBuffer)):
            result.append(d)
        elif isinstance(unwrapped, ComputedBuffer):
            producer_li = getattr(unwrapped, "loop_info", None)
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


def _allocate_full_buffer(
    tiled_op: ComputedBuffer,
    full_ranges: list[Expr],
    full_strides: tuple[Expr, ...],
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
    strides: list[Expr] = list(full_strides)

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
        # e.g. _insert_all_read_copy_ops) before stickification runs, and
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
    tiled_op_write_advances: bool = False,
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

    tiled_op_write_advances must be True when the caller has routed
    tiled_op's OWN write through a per-tile-advancing target instead of
    loop-internal scratch -- the `propagation.consumer_lookup_name is not
    None` case in _propagate_tiled_op, where tiled_op's write lands in a
    real accumulator buffer (e.g. flash attention's real_max/denominator/
    output) that a later loop iteration reads back by name and that must
    therefore actually move each iteration. This copy op's READ side reads
    that same buffer, so it must advance too, or every iteration reads back
    tile 0's slice regardless of which tile is current (issue: flash-v2's
    B-tiled denominator/output collapsing every batch to batch 0's values).
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
    # (positionally different from tiled_op's).  The read and write sides need
    # DIFFERENT extents, because they address differently sized buffers:
    #
    # READS normally re-read tiled_op's already-divided per-tile buffer,
    # which is scratch reused in place every iteration -- it does not move,
    # so it must not advance at any level (the copy op is not itself
    # re-divided). See _fixed_level_extents for why "not advance" means
    # omitting the dim, not giving it extent 1.
    #
    # tiled_op_write_advances=True overrides this: the caller has routed
    # tiled_op's own write into a real per-tile-advancing accumulator
    # buffer instead of scratch (propagation.consumer_lookup_name is not
    # None in _propagate_tiled_op -- flash attention's real_max/
    # denominator/output carried across loop iterations). This copy op
    # reads that same buffer, so its read must advance in lockstep with
    # tiled_op's write, using the identical squeeze-aware extents
    # construction _propagate_tiled_op uses there (a raw dim tiled to
    # per-tile extent 1, e.g. the B-tile dim, is squeezed out of
    # tiled_op.data.ranges entirely and must go through squeezed_advance
    # instead of read_level_extents -- see squeezed_advance_output's
    # docstring). Leaving this at _fixed_level_extents here silently pins
    # every iteration's read to tile 0's address: the next iteration's
    # "fresh" tile read is actually tile 0's stale value every time
    # (confirmed via test_flash_v2_tile_B: spyre batch 1's entire output
    # was a verbatim copy of batch 0's).
    tiled_op_info = tiled_op.loop_info  # type: ignore[attr-defined]
    read_squeezed_advance: list[list[tuple[Expr, Expr]]] = [
        [] for _ in tiled_op_info.loop_tiled_dims
    ]
    if tiled_op_write_advances:
        tiled_op_ranges = list(tiled_op.data.ranges)
        tiled_op_squeeze_pos: dict[int, int] = {}
        it_idx = 0
        for host_idx, r in enumerate(tiled_op_ranges):
            if int(r) != 1:
                tiled_op_squeeze_pos[host_idx] = it_idx
                it_idx += 1
        read_level_extents: list[dict[int, Expr]] = [
            {} for _ in tiled_op_info.loop_tiled_dims
        ]
        full_sizes_for_read = list(full_buf.get_size())
        for d in {d for level in tiled_op_info.loop_tiled_dims for d in level}:
            levels_tiling_d = [
                i for i, dims in enumerate(tiled_op_info.loop_tiled_dims) if d in dims
            ]
            if d not in tiled_op_squeeze_pos:
                host_stride = sympy.prod(full_sizes_for_read[d + 1 :])
                running = sympy.Integer(1)
                for level_idx in reversed(levels_tiling_d):
                    read_squeezed_advance[level_idx].append((host_stride, running))
                    running = running * tiled_op_info.loop_count[level_idx]
                continue
            running = sympy.sympify(tiled_op_ranges[d])
            for level_idx in reversed(levels_tiling_d):
                read_level_extents[level_idx][d] = running
                running = running * tiled_op_info.loop_count[level_idx]
    else:
        read_level_extents = _fixed_level_extents(tiled_op_info.loop_tiled_dims)
    # The WRITE targets full_buf, which is NOT divided, so its store base must
    # advance a whole tile per iteration -- the same real per-level extents
    # plan_coarse_tile_groups derives for an op's own reads/write via
    # _planned_tile_extents_per_level, and what the deleted direct-mutation
    # branch used to get for free from the tiled op's own output_tiled_dims.
    # Reusing the extent-1 read decision here instead emitted an advance of a
    # single row rather than a full tile (e.g. 64 elements instead of 32768 for
    # a [1024, 4096] fp16 buffer tiled 2-ways), so every tile after the first
    # landed almost on top of tile 0 -- the multi-stick row-tiling and
    # softmax-row-tiling wrong-address failures.  Now that every
    # cross-loop-group write routes through this function unconditionally, this
    # is the only place that decision gets made.
    #
    # copy_buf.data.ranges are already divided, so a dim's innermost-level
    # extent is the range itself; each step outward multiplies by the
    # next-inner level's trip count (same per-level formula as
    # _planned_tile_extents_per_level's _per_level_extent_for).
    #
    # write_level_extents' dict keys stay RAW positional indices into
    # copy_data.ranges -- the same convention every other loop_tiled_dims/
    # output_tiled_dims producer uses (see the canonical plan[id(op)]
    # construction above). SpyreKernel._host_dim_to_index_symbol is the
    # sole consumer of output_tiled_dims's dim keys and does its OWN squeeze
    # mapping internally against ir_node.data.ranges (== copy_ranges, since
    # ir_node is copy_buf itself here) -- pre-squeezing the key before
    # storing it double-squeezes: re-running the squeeze loop on an
    # already-squeezed number maps it to a DIFFERENT dim's identity
    # whenever a lower-numbered dim was squeezed out (e.g. B squeezed out
    # of a (B, H, Lq, D) buffer makes squeeze_pos[Lq_raw=2] == 1, and
    # re-squeezing raw dim 1 (H) resolves to H's own symbol d0 -- silently
    # advancing H's device axis for what should have been Lq's advance;
    # this is exactly what broke test_tiled_in_place_accumulator). A raw
    # dim squeezed out of copy_buf's write entirely (e.g. B tiled to
    # per-tile extent 1) has no d{i} symbol at all and instead gets a
    # squeezed_advance-style entry below, exactly as _insert_one_read_copy
    # does for reads.
    copy_ranges = list(copy_data.ranges)
    squeeze_pos: dict[int, int] = {}
    it_idx = 0
    for host_idx, r in enumerate(copy_ranges):
        if int(r) != 1:
            squeeze_pos[host_idx] = it_idx
            it_idx += 1
    write_level_extents: list[dict[int, Expr]] = [
        {} for _ in tiled_op_info.loop_tiled_dims
    ]
    squeezed_advance: list[list[tuple[Expr, Expr]]] = [
        [] for _ in tiled_op_info.loop_tiled_dims
    ]
    for d in {d for level in tiled_op_info.loop_tiled_dims for d in level}:
        levels_tiling_d = [
            i for i, dims in enumerate(tiled_op_info.loop_tiled_dims) if d in dims
        ]
        if d not in squeeze_pos:
            # Tiled down to extent 1 in copy_buf's own write -- no d{i}
            # symbol survives squeeze for _host_dim_to_index_symbol to find.
            # full_buf's own canonical (squeezed-space) index coefficient
            # for this raw dim -- product of the sizes strictly to its
            # right, matching the units dep.index's surviving d{i} symbols
            # already carry (Inductor mints those coefficients over the
            # *unsqueezed* data.ranges, then squeeze only renumbers/drops
            # symbols, never rescales them) -- not full_buf.layout.stride,
            # which is a raw PyTorch memory stride in a different unit
            # system that tiling_expr_to_device_expr's stride_map-based
            # dimension selection cannot be compared against.
            #
            # Must use full_buf's own (pre-division) sizes here, NOT
            # copy_ranges (== copy_buf's already-divided data.ranges): a
            # dim strictly to the right of d that is ITSELF tiled by
            # another loop level has already been divided down in
            # copy_ranges, undercounting host_stride by that dim's
            # division factor. full_buf.get_size() is allocated directly
            # from full_ranges (see _allocate_full_buffer) in this same
            # raw dim order and is never divided, so it gives the correct
            # full extent for both tiled and untiled dims to the right of
            # d (a no-op substitution for the untiled ones).
            full_sizes = list(full_buf.get_size())
            host_stride = sympy.prod(full_sizes[d + 1 :])
            running = sympy.Integer(1)
            for level_idx in reversed(levels_tiling_d):
                squeezed_advance[level_idx].append((host_stride, running))
                running = running * tiled_op_info.loop_count[level_idx]
            continue
        # Key must stay the RAW host-range index (matching every other
        # loop_tiled_dims/output_tiled_dims producer, e.g. the canonical
        # plan[id(op)] construction above) -- SpyreKernel.
        # _host_dim_to_index_symbol re-squeezes this raw index itself
        # against ir_node.data.ranges (== copy_ranges here) when it later
        # consumes output_tiled_dims. Pre-squeezing here as well double-
        # squeezes: passing squeeze_pos[d] as if it were raw re-triggers
        # _host_dim_to_index_symbol's own squeeze loop, which (whenever a
        # lower dim is squeezed out) maps the already-squeezed number to a
        # DIFFERENT dim's identity -- e.g. B squeezed out of (B,H,Lq,D)
        # makes squeeze_pos[Lq_raw=2]==1, and re-squeezing raw dim 1 (H)
        # returns d0, silently advancing H's device axis for what should
        # have been Lq's advance (test_tiled_in_place_accumulator).
        running = sympy.sympify(copy_ranges[d])
        for level_idx in reversed(levels_tiling_d):
            write_level_extents[level_idx][d] = running
            running = running * tiled_op_info.loop_count[level_idx]
    copy_reads = [
        dep for dep in copy_buf.get_read_writes().reads if isinstance(dep, MemoryDep)
    ]
    copy_writes = [
        dep for dep in copy_buf.get_read_writes().writes if isinstance(dep, MemoryDep)
    ]
    tiled_dims_per_read = [
        _tiled_dims_for_dep(dep, read_level_extents, copy_buf) for dep in copy_reads
    ]
    output_tiled_dims = (
        _tiled_dims_for_dep(copy_writes[0], write_level_extents, copy_buf)
        if copy_writes
        else []
    )
    copy_buf.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
        tiled_op_info,
        tiled_dims_per_read=tiled_dims_per_read,
        output_tiled_dims=output_tiled_dims,
        squeezed_advance_output=squeezed_advance if copy_writes else [],
        squeezed_advance_per_read=(
            [read_squeezed_advance] * len(copy_reads)
            if tiled_op_write_advances and copy_reads
            else []
        ),
    )

    V.graph.name_to_buffer[copy_name] = copy_buf

    tiled_idx = operations.index(tiled_op)
    tiled_op_name = tiled_op.get_name()
    outer_key = tiled_op.loop_info.loop_group_id[0]  # type: ignore[attr-defined]

    # The copy-out must come AFTER any mutation ops in the same loop group
    # that write into tiled_op's scratch buffer (e.g. copy_forced with
    # MutationLayoutSHOULDREMOVE targeting tiled_op). Insert after the last
    # such mutation, not immediately after tiled_op.
    insert_after_idx = tiled_idx
    for i, op in enumerate(operations):
        if i <= tiled_idx:
            continue
        if not isinstance(op, ComputedBuffer):
            continue
        op_outer_key = getattr(getattr(op, "loop_info", None), "loop_group_id", (None,))
        if not op_outer_key or op_outer_key[0] != outer_key:
            break
        if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            try:
                mutation_target = op.layout.get_buffer().get_name()
            except Exception:
                mutation_target = None
            if mutation_target == tiled_op_name:
                insert_after_idx = i

    operations.insert(insert_after_idx + 1, copy_buf)


class _NameSwapHandler(WrapperHandler):
    """Redirect ops.load(name, index) calls for names present in name_map.

    See NameSwapHandler in insert_restickify.py — same pattern (CLAUDE.md
    "Compiler Pass Conventions": wrap inner_fn via a WrapperHandler, never
    reconstruct it from index expressions). Duplicated locally rather than
    imported to avoid a coarse_tile <-> insert_restickify import-order
    dependency; the two run at different, non-adjacent pipeline stages.

    Unlike insert_restickify's version, entries here also carry the
    full-buffer and tile-local strides for the swapped name (see
    _insert_all_read_copy_ops): the copy buffer being swapped in is physically
    smaller than the full buffer it replaces, so tiled_op's original index
    (affine in its own loop vars, using full_buf's stride coefficients) no
    longer resolves to a valid offset into it.

    The incoming `index` at call time is tiled_op's own inner_fn tracing
    through this exact load, so it is affine in whatever loop variables that
    particular trace happens to use -- inner_fn may be retraced multiple
    times (e.g. by scheduler fusion checks) with a *different* dummy
    variable each time (d0, i0, q0, ... have all been observed for the same
    load site), so a single precomputed replacement expression captured at
    _insert_all_read_copy_ops time would be wrong on every trace that doesn't
    happen to reuse those exact symbols. Instead, `index` is rescaled at
    call time: each additive term's coefficient is matched (by value) against
    full_strides to find its dimension, then replaced by that dimension's
    tile_strides coefficient -- this works for whatever free symbols this
    particular trace used, without needing to know them in advance.
    """

    def __init__(
        self,
        inner,
        name_map: dict[str, tuple[str, list[Expr], list[Expr]]],
    ):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        if name in self._name_map:
            new_name, full_strides, tile_strides = self._name_map[name]
            new_index = _rescale_index(index, full_strides, tile_strides)
            return super().load(new_name, new_index)
        return super().load(name, index)


def _rescale_index(
    index: Expr, full_strides: list[Expr], tile_strides: list[Expr]
) -> Expr:
    """Rescale an affine index's per-dimension coefficients.

    `index` is affine in some set of loop variables, with one additive term
    per dimension whose coefficient equals the matching entry in
    `full_strides` (plus, possibly, a constant offset term). Returns the
    same linear combination of the same variables with each dimension's
    coefficient replaced by the matching entry in `tile_strides`. Matching
    is by coefficient value rather than by variable identity because the
    variables `index` is expressed in are not known in advance -- see
    _NameSwapHandler.

    Each additive term is matched against a candidate `full_stride` by
    dividing the term by it and checking the quotient is free of the
    stride's own symbols (see `_divides_evenly` below) -- NOT via
    `index.as_coefficients_dict()`, which only isolates a term's
    "coefficient" correctly when that coefficient is a plain number. When
    `full_strides` contains a genuinely symbolic stride (e.g. a level/tile
    symbol) and the matching term is `loop_var * symbolic_stride`, sympy
    normalizes that whole product into a single atom with numeric
    coefficient 1 -- `as_coefficients_dict()` would report the *entire
    product* as the "term" and never find a `full_strides` entry equal to
    1, silently failing to match a case this function is specifically
    meant to support.

    Two further subtleties in the matching, both because dimensions are
    identified by their stride *value* rather than their position:

    - An extent-1 dimension's stride can coincide with a larger dimension's
      stride (e.g. a size-[1, N] shape's dim-0 stride equals dim-1's full
      extent, same as a size-[M, N] shape's dim-0 stride). Matching
      smallest-remaining-first would let a degenerate extent-1 stride steal
      a match that belongs to a real, larger dimension. Matching
      largest-first instead defers ambiguity among small/degenerate strides
      as long as possible, since a larger stride can only coincide with
      another dimension of at least that size.
    - Two full_strides can be symbolically equal but differently-formed
      expressions (e.g. ``2*(s0 + 1)`` vs ``2*s0 + 2``) -- `_divides_evenly`
      falls back to a simplified quotient/difference check rather than
      relying on structural equality alone.
    """

    def _divides_evenly(term: Expr, full_stride: Expr) -> tuple[bool, Expr]:
        """Return (matched, loop_var_part) if `full_stride` divides `term`.

        `full_stride` divides `term` cleanly when dividing it out of `term`
        leaves *exactly* the loop-variable part behind: no leftover free
        symbol from `full_stride` (it must fully cancel, not partially --
        e.g. dividing `c0*s0` by `s0` alone, not by some unrelated factor of
        it), and no leftover numeric scale factor (e.g. dividing `c0*128` by
        `4` leaves `32*c0`, i.e. still scaled by 32 -- not a clean divide,
        even though the quotient happens to be symbol-free). Checked both
        structurally and, if that's inconclusive, after simplifying the
        quotient (mirrors the structural-vs-simplified fallback this
        function has always used for coefficient matching).
        """
        stride_syms = full_stride.free_symbols

        def _is_clean(quotient: Expr) -> bool:
            coeff, _ = quotient.as_coeff_Mul()
            return coeff == 1 and not (quotient.free_symbols & stride_syms)

        quotient = term / full_stride
        if _is_clean(quotient):
            return True, quotient
        simplified = sympy.simplify(quotient)
        if _is_clean(simplified):
            return True, simplified
        return False, sympy.Integer(0)

    def _sort_key(pair: tuple[Expr, Expr]) -> tuple[int, Expr]:
        # Sort largest-first without calling `<` directly on sympy Exprs --
        # that raises TypeError for expressions with free symbols, which
        # full_strides commonly contains (level/tile symbols). Concrete
        # integers compare among themselves by value; every symbolic stride
        # sorts ahead of every concrete one (a symbolic stride is a
        # multiple of some concrete extent, so it is at least as large),
        # and symbolic-vs-symbolic keeps original relative order (stable
        # sort) rather than guessing a magnitude.
        full_stride = pair[0]
        is_concrete = full_stride.is_number
        return (
            0 if is_concrete else 1,
            full_stride if is_concrete else sympy.Integer(0),
        )

    remaining = sorted(zip(full_strides, tile_strides), key=_sort_key, reverse=True)
    new_index: Expr = sympy.Integer(0)
    for term in sympy.Add.make_args(index):
        if term.is_number:
            new_index += term
            continue
        for i, (full_stride, tile_stride) in enumerate(remaining):
            matched, loop_var_part = _divides_evenly(term, full_stride)
            if matched:
                new_index += tile_stride * loop_var_part
                del remaining[i]
                break
        else:
            raise RuntimeError(
                f"_rescale_index: no matching full_stride for term {term} "
                f"in index {index}; full_strides={full_strides}"
            )
    return new_index


def _insert_one_read_copy(
    sizing_op: ComputedBuffer,
    dep: MemoryDep,
    sizing_read_index: int,
    copy_name: str,
    operations: list[Operation],
    insert_before_op: Operation,
    *,
    predivision_unit_steps: tuple[tuple[tuple[int, Expr, Expr], ...], ...] = (),
    loop_invariant: bool = False,
) -> str:
    """Build and insert one tile-sized copy op for a single full-buffer read.

    sizing_op reads (or is the first of a group of ops that all read) a
    full-size cross-loop-group buffer directly (see _full_buffer_read_deps).
    That buffer gets exactly one candidate layout (sized to the full
    buffer), while sizing_op's own candidates are sized to its tile — the
    two can never be stick-compatible under AllSameNode.  Mirroring
    _insert_copy_op's write-side fix: insert a copy op that reads the full
    buffer's current tile slice (same index expression dep already
    describes, same loop_info as sizing_op so the per-iteration base
    address advances identically) and writes it into a fresh tile-sized
    buffer.

    The copy's own ranges/index must match dep (dep.var_names/dep.size), not
    sizing_op.data.ranges: for a Reduction, the read spans output dims plus
    the reduction dim, so dep's iteration space has more vars than the op's
    own output-shaped ranges.  The copy buffer's own layout gets fresh
    contiguous tile-local strides (it is a physically smaller allocation
    than full_buf, not an aliased view of it — see tile_strides below).

    Mirrors _allocate_full_buffer's isinstance(orig_layout, FixedTiledLayout)
    branch: when full_buf already carries a FixedTiledLayout (post-stickify
    call site), the copy gets a FixedTiledLayout too, with device_layout
    resized down from full_buf's own device_layout via _resize_device_layout
    (shrink direction, mirroring _divide_ranges's use of the same helper).
    Otherwise (pre-stickify call site) the copy gets a plain FixedLayout, and
    stickification (which runs later) fills device_layout in normally.

    insert_before_op is the plan's own insertion-point decision (see
    ReadCopyEntry.insert_before_op_name) -- it names the first
    (operations order) consuming op in the group and is authoritative
    for where the copy is spliced into `operations`, independent of
    sizing_op (which only supplies loop_info/ranges/origins here and may
    coincide with insert_before_op but is not guaranteed to by contract).

    Returns the inserted copy buffer's name (copy_buf.get_name()) -- callers
    patch consumers separately via _patch_consumer_to_read_copy.
    """
    insert_idx = operations.index(insert_before_op)
    full_buf = V.graph.get_buffer(dep.name)
    # Graph inputs come back TensorBox(StorageBox(InputBuffer))-wrapped
    # (see graph_inputs); get_dtype() resolves to self.dtype via IRNode
    # and is not delegated by TensorBox/StorageBox, so it raises
    # AttributeError on the wrapper -- unwrap to the real Buffer first.
    # get_name()/.layout are delegating and would work either way, but
    # unwrap once so every full_buf.* access below is on the real node.
    if isinstance(full_buf, TensorBox):
        full_buf = full_buf.data
    if isinstance(full_buf, StorageBox):
        full_buf = full_buf.data

    # Derive copy buffer strides using compute_tile_stride.
    # dep.size is the full loop iteration space (output + reduction dims) and
    # may have higher rank than the tensor (e.g. for a Reduction reading
    # a[M,K], dep.size=[M,N,K_tile] while the tensor has rank 2).
    # Use dep.index.coeff(v) to identify active dims (non-zero coeff) vs
    # broadcast/absent dims (zero coeff, e.g. N above).
    #
    # dep.size[i] is NOT the buffer's full range for an active dim: when
    # sizing_op's own loop is already tiled (the common case -- this
    # function copies a full-size buffer INTO a tile), dep.size reflects the
    # reader's tile-local extent (e.g. 128 for a Lq-tiled read), not the
    # buffer's true full/untiled extent (e.g. 256).  Passing that tile-local
    # value as both the "full size" and "tile size" arguments makes
    # compute_tile_stride's size//tile_size ratio 1 for every dim, a no-op
    # that leaves full_buf's own (too-large) stride on the physically
    # smaller copy buffer.  Look up each active dim's true full size from
    # full_buf.layout instead, matched by stride VALUE (full_coeff), not by
    # position -- dep.var_names/active_idx are in dep's squeezed iteration
    # space, while full_buf.layout.size/stride are in the buffer's own raw
    # space, and the two do not line up positionally in general (broadcast
    # inputs, reductions with more loop vars than the tensor has dims, etc).
    full_coeff = [dep.index.coeff(v) for v in dep.var_names]
    # Positions with non-zero coeff are active tensor dims; zero coeff means
    # broadcast/absent (e.g. the N dim in a Reduction reading a[M,K]).
    active_idx = [i for i, c in enumerate(full_coeff) if c != 0]
    # A broadcast input that is fixed for the whole counted loop needs one
    # compact staging copy of its real tensor dimensions.  Keeping the absent
    # loop dimension would materialize the expanded [E, ...] view in HBM.
    # A fully-broadcast scalar has no real dimensions to compact.
    compact_invariant = loop_invariant and bool(active_idx)
    tile_ranges = (
        [dep.size[i] for i in active_idx] if compact_invariant else list(dep.size)
    )

    tile_strides: list[Expr]
    active_full_strides: list[Expr] = []
    if not active_idx:
        tile_strides = [sympy.Integer(0)] * len(tile_ranges)
    else:
        # full_buf.layout.size/stride cannot be used to look up each active
        # dim's full size: full_buf is the RAW graph buffer dep.name names,
        # but dep.index's coefficients (full_coeff) are expressed in
        # whatever coordinate space the read was indexed in, which may be a
        # reshape/view of full_buf with no corresponding entry in
        # full_buf.layout at all (e.g. a 1D [Lq*D] input .view()-ed to
        # [Lq, D] before being read -- full_buf stays 1D/stride=[1] forever,
        # so no stride in its layout equals the viewed coordinate space's
        # per-dim strides). Derive full sizes purely from the active
        # strides themselves instead: in a dense coordinate space, each
        # dim's full size is (stride of the next-larger active dim) /
        # (this dim's own stride), and the outermost active dim's full size
        # is full_buf's total element count / its own stride -- true
        # regardless of any reshape, since numel is view-invariant.
        active_full_strides = [full_coeff[i] for i in active_idx]
        order = sorted(range(len(active_idx)), key=lambda k: active_full_strides[k])
        total_elems = sympy.prod(full_buf.get_size())
        active_full_sizes: list[Expr] = [sympy.Integer(0)] * len(active_idx)
        for pos, k in enumerate(order):
            next_stride = (
                active_full_strides[order[pos + 1]] if pos + 1 < len(order) else None
            )
            active_full_sizes[k] = (
                total_elems // active_full_strides[k]
                if next_stride is None
                else next_stride // active_full_strides[k]
            )
        active_tile_ranges = [dep.size[i] for i in active_idx]
        active_tile_strides = compute_tile_stride(
            active_full_sizes, active_full_strides, active_tile_ranges
        )
        # active_tile_strides[i] corresponds to active_idx[i]: both are indexed
        # by position in the compressed active-dims space, so zip pairs each
        # full dep.var_names position with its computed tile stride correctly.
        if compact_invariant:
            tile_strides = active_tile_strides
        else:
            tile_strides = [sympy.Integer(0)] * len(tile_ranges)
            for pos, ts in zip(active_idx, active_tile_strides):
                tile_strides[pos] = ts

    def _copy_inner_fn(
        idx,
        _dep=dep,
        _full_name=full_buf.get_name(),
        _active_idx=active_idx,
        _compact=compact_invariant,
    ):
        if _compact:
            full_idx = [sympy.Integer(0)] * len(_dep.var_names)
            for pos, value in zip(_active_idx, idx):
                full_idx[pos] = value
        else:
            full_idx = idx
        subs = dict(zip(_dep.var_names, full_idx))
        flat_index = sympy_subs(_dep.index, subs)
        return V.ops.load(_full_name, flat_index)

    # Construct under sizing_op's origins so data.origins is non-empty —
    # _single_arg_op_layout (propagate_layouts.py) unconditionally
    # dereferences next(iter(data.origins)) for ordinary (non-mutation)
    # Pointwise ops.  IRNode.origins is populated at construction time
    # from IRNode._current_origins, so it must be set via this context
    # manager rather than assigned after the fact (assigning
    # copy_buf.origins below only sets the ComputedBuffer's own origins,
    # not copy_data's).
    with IRNode.current_origins(sizing_op.origins):
        copy_data = Pointwise(
            device=sizing_op.get_device(),
            dtype=full_buf.get_dtype(),
            inner_fn=_copy_inner_fn,
            ranges=tile_ranges,
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
        # tile_ranges (== list(dep.size)) is dep's *squeezed* size --
        # extract_read_writes drops unit-size dims, so tile_ranges is
        # one shorter per unit dim in full_buf's own raw size and no
        # longer lines up positionally with full_size_ints.
        # _resize_device_layout indexes new_host_size exclusively by
        # positions derived from old_host_size (matched_host/pstar), so
        # it requires equal rank -- reinsert a 1 at each raw position
        # full_layout.size squeezed out, undoing the same squeeze
        # applied to sizing_op's own ranges elsewhere in this function
        # (see squeeze_pos below).
        # Count the non-unit dims and check the pairing *before* walking,
        # not after.  The walk consumes one tile_ranges entry per non-unit
        # dim, so a check placed after it only catches the direction where
        # entries are left over.  The opposite direction -- more non-unit
        # buffer dims than iteration extents -- would run off the end of
        # tile_ranges inside the loop and raise IndexError, which is the
        # same "error from deep inside a pass" this guard exists to
        # replace, just wearing a different exception type.
        non_unit_dims = sum(1 for s in full_size_ints if s != 1)
        if non_unit_dims != len(tile_ranges):
            # TODO(span-overflow-read-copy): support tiled ops whose
            # iteration space does not map one-to-one onto an input's
            # dimensions.  Replace this marker with the tracking issue
            # number once filed; the three xfailed tests named at the
            # bottom of this comment are the ones it unblocks.
            #
            # The walk below pairs each of full_buf's non-unit dims with
            # the next entry of tile_ranges (== dep.size, the op's
            # iteration extents), assuming the two have the same number of
            # non-unit entries.  That holds only when every input has the
            # output's shape.  Three ways it breaks, all the same failure:
            #
            #   * broadcast input of lower rank -- an rmsnorm weight
            #     (16384,) against an iteration space [16, 1000, 16384]:
            #     one buffer dim, three extents.  Note the walk does not
            #     merely run out, it pairs the 16384 dim with extent 16, so
            #     without this check it would build a wrong buffer rather
            #     than fail;
            #   * broadcast input with leading unit dims -- a rope cos
            #     (1, 1, 2048, 2048) against [32, 16, 1024, 2048]: the two
            #     1s are skipped, leaving two extents unconsumed;
            #   * a reduction whose input does not use every loop var --
            #     out[b, m, n] = sum_k A[b, m, k] * B[b, k, n] carries
            #     b/m/n/k while A has no n dimension at all.  B is worse
            #     still: its dims run b/k/n against a b/m/n/k loop, so even
            #     after dropping the unused var the orders disagree and a
            #     positional walk would hand k's extent to n's dimension.
            #
            # Both need dimension matching via dep.index's coefficients
            # rather than by position.  That was attempted and reverted:
            # it clears this check and compiles, but the results are still
            # numerically wrong, so a second positional assumption remains
            # further down (most likely in how the per-iteration address
            # advance is derived).  Reverted rather than shipped half-done.
            #
            # Only the POST-stickify caller reaches here -- manual
            # spyre_hint runs pre-stickify, gets a plain FixedLayout, and
            # takes the else branch below.  Manual tiling of this exact
            # 4-D bmm batch-dim case is verified working end to end
            # (test_bmm_to_pointwise_join_numeric_via_manual_hint), so what
            # is missing is this branch, not the tiling machinery.
            #
            # #3293 moved the hint path pre-stickify specifically so
            # stickification builds the layout from already-divided ranges,
            # "eliminating _resize_device_layout" -- and noted the
            # span-overflow path "retains the FixedTiledLayout construction
            # with _resize_device_layout as before".  Span-overflow cannot
            # follow: it needs device_layout to measure spans at all, so it
            # cannot run pre-stickify.  It is therefore the last caller on
            # that path, which is likely why this gap survived.  Worth
            # settling whether the fix is to repair the resize or to have
            # stickification supply this layout, before investing in the
            # former.
            #
            # Blocks: test_bmm_to_pointwise_join_numeric,
            # test_bmm_to_reduction_join_numeric (this branch), and
            # test_lm_head_matmul_join_numeric (xfailed since #3218 with
            # the identical failure).
            #
            # Raise Unsupported rather than letting the bare assert fire:
            # this is a known gap reachable from ordinary user code, so it
            # should surface as a clean backend limitation, not an
            # AssertionError from deep inside a pass.
            raise Unsupported(
                f"coarse_tile: cannot build a tile-sized read copy of "
                f"{dep.name!r} for {sizing_op.get_name()!r}: its iteration "
                f"extents {list(tile_ranges)} do not map one-to-one onto "
                f"the buffer's dimensions {full_size_ints}. Automatic "
                "span-overflow tiling is not yet supported for inputs "
                "that do not share the output's shape — a broadcast input "
                "(lower rank, or leading unit dims), or a reduction input "
                "that does not use every loop variable such as a "
                "batch-tiled bmm operand."
            )
        # Reinsert a 1 at each raw position full_layout.size squeezed out.
        # The guard above has established that the non-unit dims and
        # tile_ranges are the same length, so this walk cannot run off the
        # end.
        tile_size_ints = []
        it_idx = 0
        for s in full_size_ints:
            if s == 1:
                tile_size_ints.append(1)
            else:
                tile_size_ints.append(int(tile_ranges[it_idx]))
                it_idx += 1
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
            logger.warning(
                "_insert_one_read_copy: _resize_device_layout could not "
                "classify %r (full_size=%s tile_size=%s); using "
                "row-major fallback",
                full_layout.device_layout,
                full_size_ints,
                tile_size_ints,
            )
            # Row-major fallback describes the freshly allocated,
            # squeezed-rank copy buffer directly (unlike the
            # reconstruction above, it has no need for full_buf's raw
            # rank) -- use tile_ranges/tile_strides, not the
            # full-buf-rank-padded tile_size_ints.
            squeezed_size_ints = [int(s) for s in tile_ranges]
            device_layout = SpyreTensorLayout(
                squeezed_size_ints,
                [int(s) for s in tile_strides],
                full_buf.get_dtype(),
                list(range(len(squeezed_size_ints))),
                full_layout.device_layout.element_arrangement,
            )
        copy_layout = FixedTiledLayout(
            sizing_op.get_device(),
            full_buf.get_dtype(),
            tile_ranges,
            tile_strides,
            device_layout,
        )
    else:
        copy_layout = FixedLayout(
            sizing_op.get_device(),
            full_buf.get_dtype(),
            tile_ranges,
            tile_strides,
        )
    copy_buf = ComputedBuffer(name=copy_name, layout=copy_layout, data=copy_data)
    copy_buf.origins = sizing_op.origins
    copy_buf.operation_name = copy_name
    copy_op_metadata(sizing_op, copy_buf)
    # This is a new operation with its own iteration space.  The source
    # operation's d0/d1/... names have no positional meaning for the copy, so
    # let work-division planning choose from the copy's actual dimensions.
    if hasattr(copy_buf, "work_div_loop_info"):
        del copy_buf.work_div_loop_info  # type: ignore[attr-defined]

    if loop_invariant:
        # No loop metadata means codegen emits this copy once before the first
        # loop-body consumer.  The copy has its own iteration symbols, so the
        # consumer's named work-division request must not be reused on it
        # (it was removed unconditionally above).
        if hasattr(copy_buf, "loop_info"):
            del copy_buf.loop_info  # type: ignore[attr-defined]
        V.graph.name_to_buffer[copy_name] = copy_buf
        operations.insert(insert_idx, copy_buf)
        logger.debug(
            "coarse_tile: invariant read copy-in %s -> %s", dep.name, copy_name
        )
        return copy_buf.get_name()

    # Fresh per-level tiled-dim decisions for copy_buf's own read/write —
    # mirroring _insert_copy_op's read/write split (see its comment), but
    # with the roles swapped: there, the copy's READ (of sizing_op's
    # per-tile scratch) must not advance and its WRITE (to full_buf)
    # advances a whole tile; here, the copy's READ (of full_buf) advances
    # a whole tile per iteration and its WRITE (to this copy's own
    # freshly allocated, tile-sized buffer) must not advance -- the copy
    # buffer is scratch reused in place, it does not move.
    # See _fixed_level_extents for why "must not advance" means omitting
    # the dim, not giving it extent 1.
    #
    # Reusing sizing_op.loop_info verbatim here (as an earlier version of
    # this function did) is wrong for a different reason than
    # _insert_copy_op's original bug: it is not just the wrong extent, it
    # is tiled_dims_per_read/output_tiled_dims computed for sizing_op's own
    # reads/write (by then patched to read the copy buffers, not
    # full_buf) applied positionally to copy_buf's own reads/write (of
    # full_buf and of copy_buf's own output) via
    # SpyreKernel._general_tile_advance's positional dep-index lookup —
    # a semantic mismatch, not merely a magnitude one.
    sizing_op_info = sizing_op.loop_info  # type: ignore[attr-defined]
    dep_idx = sizing_read_index
    planned_unit_steps_by_dim: dict[int, list[list[tuple[Expr, Expr]]]] = {}
    for level_idx, level in enumerate(predivision_unit_steps):
        for dim, stride, extent in level:
            per_level = planned_unit_steps_by_dim.setdefault(
                dim, [[] for _ in sizing_op_info.loop_count]
            )
            per_level[level_idx].append((stride, extent))
    copy_ranges = list(copy_data.ranges)
    # sizing_op_info.loop_tiled_dims's dim keys are raw positional indices
    # into sizing_op.data.ranges (see CoarseTileInfo's docstring), which
    # may include unit-size (==1) dims (e.g. a unit B in BHLD). But
    # copy_ranges (== list(dep.size), dep being sizing_op's own *squeezed*
    # MemoryDep -- see extract_read_writes -> index_vars_squeeze) has
    # already dropped those unit dims, so copy_ranges is one shorter per
    # squeezed-out dim and its positions no longer line up with
    # loop_tiled_dims's raw numbering. Map each raw dim to its squeezed
    # position (mirroring SpyreKernel._host_dim_to_index_symbol's own
    # squeeze arithmetic) before indexing copy_ranges.
    squeeze_pos: dict[int, int] = {}
    it_idx = 0
    for host_idx, r in enumerate(sizing_op.data.ranges):
        if int(r) != 1:
            squeeze_pos[host_idx] = it_idx
            it_idx += 1
    write_level_extents = _fixed_level_extents(sizing_op_info.loop_tiled_dims)
    read_level_extents: list[dict[int, Expr]] = [
        {} for _ in sizing_op_info.loop_tiled_dims
    ]
    squeezed_advance: list[list[tuple[Expr, Expr]]] = (
        [list(level) for level in sizing_op_info.squeezed_advance_per_read[dep_idx]]
        if dep_idx < len(sizing_op_info.squeezed_advance_per_read)
        else [[] for _ in sizing_op_info.loop_tiled_dims]
    )
    for d in {d for level in sizing_op_info.loop_tiled_dims for d in level}:
        if d in planned_unit_steps_by_dim and d in squeeze_pos:
            # A captured unit-tile dim should have been squeezed out by
            # division. Keep the captured fact authoritative and make this
            # unexpected legacy shape visible.
            selected_steps: list[list[tuple[Expr, Expr]]] = _select_unit_steps(
                op_name=sizing_op.get_name(),
                dep_name=dep.name,
                dim=d,
                planned=planned_unit_steps_by_dim[d],
                legacy=[],
            )
            for level_idx, step_level in enumerate(selected_steps):
                squeezed_advance[level_idx].extend(step_level)
            continue
        levels_tiling_d = [
            i for i, dims in enumerate(sizing_op_info.loop_tiled_dims) if d in dims
        ]
        if d not in squeeze_pos:
            # sizing_op.data.ranges[d] == 1 for this tiled dim (e.g. a
            # B-tiled group where the per-tile B extent is 1) -- it was
            # squeezed out of sizing_op's own iteration space entirely
            # (Inductor's SqueezeView.squeezer, invoked unconditionally by
            # extract_read_writes/index_vars_squeeze whenever a dim's range
            # is 1, with no way to opt a specific dim out), so it has no
            # d{i} symbol in dep.index for _host_dim_to_index_symbol to
            # find -- unlike a dim genuinely absent from *this* read among
            # several (broadcast), which tiled_dims_per_read/
            # _tiled_dims_for_dep already handle correctly.
            #
            # But this raw dim being squeezed out of *sizing_op's own*
            # iteration space says nothing about whether *this specific
            # dep* (this copy's tensor) actually has that dimension at all.
            # sizing_op is one op reading potentially several tensors (e.g.
            # a matmul's x and w): a genuinely broadcast operand (x, with
            # no E dim in its own shape whatsoever, not even a squeezed-to-1
            # one) must NOT get a squeezed_advance term for E just because
            # E happens to be squeezed to a per-tile extent of 1 in the
            # *matmul's* own ranges (issue #3613's E-tiling case). Since d
            # has no d{i} symbol for ANY dep of sizing_op (it is squeezed
            # out globally, not per-dep), dep.index's coefficients cannot
            # answer this -- unlike the squeeze_pos branch below, which can
            # lean on _tiled_dims_for_dep's free-symbol check. Instead,
            # check dep's own buffer: d's full (pre-tile) extent is
            # `running`'s value after the accumulation below; the buffer
            # genuinely has dim d iff its total element count is divisible
            # by host_stride with quotient equal to that full extent --
            # numel is view-invariant (unlike full_buf.layout.stride, which
            # a reshape/view can leave with no entry matching host_stride
            # at all -- see the active_full_strides comment above), so this
            # holds regardless of any reshape/view between the graph input
            # and this read.
            #
            # The canonical (squeezed-space) index coefficient for this raw
            # dim -- product of sizing_op.data.ranges sizes strictly to its
            # right, matching the units dep.index's surviving d{i} symbols
            # already carry (Inductor mints those coefficients over the
            # *unsqueezed* data.ranges; squeeze only renumbers/drops
            # symbols, never rescales them) -- independent of dep.index
            # entirely, lets SpyreKernel._general_tile_advance add this
            # level's device-address contribution as an extra term via
            # tiling_expr_to_device_expr -- see squeezed_advance_per_read.
            # NOT full_buf.layout.stride: that's a raw PyTorch memory
            # stride, a different unit system tiling_expr_to_device_expr's
            # stride_map-based dimension selection cannot be compared
            # against (see _insert_copy_op's write-side analogue for the
            # collision this caused when the two happened to diverge).
            host_stride = sympy.prod(sizing_op.data.ranges[d + 1 :])
            d_full_size = sympy.Integer(1)
            for level_idx in reversed(levels_tiling_d):
                d_full_size = d_full_size * sizing_op_info.loop_count[level_idx]
            # A bare numel check (total_elems // host_stride == d_full_size)
            # is not sound: host_stride/d_full_size are properties of
            # sizing_op alone, and a genuinely-broadcast dep's own numel is
            # unrelated to either -- it is possible (if unlikely for any
            # one kernel) for a broadcast tensor's real numel to coincide
            # with host_stride * d_full_size purely by chance, wrongly
            # granting it an advance for a dim it does not have. Instead,
            # use full_buf's own RANK: active_idx (computed above from this
            # same dep) already gives every dim of full_buf that dep.index
            # can see; d, if present at all, is the ONE dim invisible to
            # dep.index (squeezed out of sizing_op's whole space, per the
            # comment above) -- so full_buf has dim d iff its rank exceeds
            # active_idx's count by exactly 1, AND the one dim not
            # accounted for by active_full_sizes actually has size
            # d_full_size (guards against an unrelated extra dim of some
            # other size, e.g. a genuine size-1 dim of full_buf's own that
            # active_idx also does not see). Rank and per-dim sizes are
            # exact structural properties of full_buf, unlike a numel
            # product, so this cannot collide the way the numel check can.
            full_sizes = list(full_buf.get_size())
            rank_excess = len(full_sizes) - len(active_idx)
            leftover_counts = collections.Counter(full_sizes)
            if active_idx:
                for s in active_full_sizes:
                    leftover_counts[s] -= 1
            leftover_sizes = [s for s, c in leftover_counts.items() if c > 0]
            dep_has_dim_d = (
                host_stride != 0
                and rank_excess == 1
                and len(leftover_sizes) == 1
                and leftover_sizes[0] == d_full_size
            )
            legacy_steps: list[list[tuple[Expr, Expr]]] = [
                [] for _ in sizing_op_info.loop_count
            ]
            if dep_has_dim_d:
                running = sympy.Integer(1)
                for level_idx in reversed(levels_tiling_d):
                    legacy_steps[level_idx].append((host_stride, running))
                    running = running * sizing_op_info.loop_count[level_idx]
            selected_steps = _select_unit_steps(
                op_name=sizing_op.get_name(),
                dep_name=dep.name,
                dim=d,
                planned=planned_unit_steps_by_dim.get(d),
                legacy=legacy_steps,
            )
            for level_idx, step_level in enumerate(selected_steps):
                squeezed_advance[level_idx].extend(step_level)
            continue
        # The dict key must be a raw positional index into copy_buf's own
        # data.ranges (what SpyreKernel._host_dim_to_index_symbol will
        # later squeeze again when it runs against copy_buf) -- i.e. the
        # squeezed position computed above, not sizing_op's raw d.
        #
        # squeeze_pos[d] is sizing_op's OWN squeezed symbol number for d --
        # it says nothing about whether THIS dep has d at all, or at that
        # same squeezed position, since dep can squeeze a different set of
        # unit dims out of its own iteration space than sizing_op does (e.g.
        # a broadcast operand like a [M, 1] scale read against a reduction
        # whose own ranges keep that dim non-unit at tile-extent > 1 --
        # issue #3613's family of mismatches). dep.var_names is dep's own
        # squeezed d{i} symbol list, positionally aligned with copy_ranges
        # (both derived from the same dep.size/dep.index); find sizing_op's
        # d{squeeze_pos[d]} symbol within it directly, mirroring
        # _tiled_dims_for_dep's free-symbol membership check rather than
        # assuming the two ops' squeezed numbering coincides.
        sizing_symbol = sympy_index_symbol(f"d{squeeze_pos[d]}")
        if sizing_symbol not in dep.var_names:
            # This dim is squeezed out of, or simply absent from, dep's own
            # space (broadcast) -- no advance term to contribute here.
            continue
        copy_dim = dep.var_names.index(sizing_symbol)
        running = sympy.sympify(copy_ranges[copy_dim])
        for level_idx in reversed(levels_tiling_d):
            read_level_extents[level_idx][copy_dim] = running
            running = running * sizing_op_info.loop_count[level_idx]
    reduction_squeeze_pos: dict[int, int] = {}
    red_it_idx = 0
    reduction_ranges = getattr(sizing_op.data, "reduction_ranges", None) or []
    for host_idx, r in enumerate(reduction_ranges):
        if int(r) != 1:
            reduction_squeeze_pos[host_idx] = red_it_idx
            red_it_idx += 1
    for d in {d for level in sizing_op_info.loop_tiled_reduction_dims for d in level}:
        levels_tiling_d = [
            i
            for i, dims in enumerate(sizing_op_info.loop_tiled_reduction_dims)
            if d in dims
        ]
        if d not in reduction_squeeze_pos:
            reduction_plan = getattr(
                getattr(sizing_op_info, "propagation", None), "reduction", None
            )
            if getattr(reduction_plan, "carried", None) is None:
                raise Unsupported(
                    "coarse_tile: size-one reduction tiles require an explicit "
                    "carried-reduction plan"
                )
            # The tiled reduction extent became one, so Inductor removed its
            # loop symbol.  Preserve its source-buffer address step explicitly
            # instead of indexing a symbol map that cannot contain it.
            d_full_size = sympy.Integer(1)
            for level_idx in reversed(levels_tiling_d):
                d_full_size *= sizing_op_info.loop_count[level_idx]

            full_sizes = list(full_buf.get_size())
            full_strides = list(full_buf.layout.stride)
            active_stride_counts = collections.Counter(active_full_strides)
            candidates: list[Expr] = []
            for size, stride in zip(full_sizes, full_strides):
                if active_stride_counts[stride] > 0:
                    active_stride_counts[stride] -= 1
                    continue
                if sympy.simplify(size - d_full_size) == 0:
                    candidates.append(stride)
            if len(candidates) != 1:
                raise Unsupported(
                    f"coarse_tile: cannot recover squeezed reduction dim {d} "
                    f"for read {dep.name}: full_size={d_full_size}, "
                    f"candidate_strides={candidates}"
                )
            running = sympy.Integer(1)
            for level_idx in reversed(levels_tiling_d):
                squeezed_advance[level_idx].append((candidates[0], running))
                running *= sizing_op_info.loop_count[level_idx]
            continue
        copy_dim_key = it_idx + reduction_squeeze_pos[d]
        running = sympy.sympify(copy_ranges[copy_dim_key])
        for level_idx in reversed(levels_tiling_d):
            read_level_extents[level_idx][copy_dim_key] = running
            running = running * sizing_op_info.loop_count[level_idx]
    copy_reads = [
        r for r in copy_buf.get_read_writes().reads if isinstance(r, MemoryDep)
    ]
    copy_writes = [
        w for w in copy_buf.get_read_writes().writes if isinstance(w, MemoryDep)
    ]
    tiled_dims_per_read = [
        _tiled_dims_for_dep(r, read_level_extents, copy_buf) for r in copy_reads
    ]
    output_tiled_dims = (
        _tiled_dims_for_dep(copy_writes[0], write_level_extents, copy_buf)
        if copy_writes
        else []
    )
    # copy_buf is its own op, not sizing_op under another name -- it must
    # not inherit sizing_op_info.propagation. That plan was computed for
    # sizing_op's own boundary-crossing decision (kind may be "reduction"
    # or "copy_out"); copy_buf is purely scratch, reused in place and
    # never read outside this loop group, so its own kind is always
    # "loop_internal". Leaving propagation unreplaced here previously let
    # a "reduction"-kind plan leak onto this Pointwise passthrough buffer,
    # causing Pass 2 (_insert_all_reduction_ops) to misdispatch it into
    # _propagate_tiled_reduction_op.
    #
    # loop_tiled_dims/loop_tiled_reduction_dims must ALSO be recomputed in
    # copy_buf's own raw-position space here, for the same reason
    # tiled_dims_per_read/output_tiled_dims are: sizing_op_info's raw
    # positions name dims in sizing_op's data.ranges, which can (and, for
    # any sizing_op whose read/write dims don't line up 1:1 with copy_buf's
    # own -- e.g. a transpose/reshape between them -- generally does) point
    # at a different dim of copy_buf's own data.ranges at the same raw
    # index. Downstream consumers of copy_buf.loop_info (e.g.
    # work_division_constraints.coarse_tile_local_dim_split_domains) read
    # loop_tiled_dims against copy_buf's own ranges/raw-squeeze table, so a
    # verbatim carry-over here silently mis-names which dim is tiled at
    # each level (confirmed by a minimal B+H coarse-tiled matmul repro
    # misreading the copy's own D axis as sizing_op's H axis). read_level_
    # extents/write_level_extents (built above) already key each tiled dim
    # by its correct raw position in copy_buf's own data.ranges -- reuse
    # those keys instead of sizing_op_info's.
    copy_loop_tiled_dims = [sorted(level) for level in read_level_extents]
    # Always empty, unconditionally: a generated copy_buf's own data is
    # always a Pointwise passthrough (a plain per-tile scratch read or
    # write-back), never a Reduction, so it has no reduction_ranges of its
    # own to tile regardless of whether the sizing op it copies for reduces
    # over a dim. work_division_constraints.coarse_tile_local_dim_split_
    # domains relies on this: it only indexes reduction_ranges when
    # ctx.op.data exposes it, so an empty list here is correct, not a
    # placeholder to fill in later.
    copy_loop_tiled_reduction_dims: list[list[int]] = [
        [] for _ in sizing_op_info.loop_count
    ]
    copy_buf.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
        sizing_op_info,
        loop_tiled_dims=copy_loop_tiled_dims,
        loop_tiled_reduction_dims=copy_loop_tiled_reduction_dims,
        tiled_dims_per_read=tiled_dims_per_read,
        output_tiled_dims=output_tiled_dims,
        squeezed_advance_per_read=[squeezed_advance] if copy_reads else [],
        # copy_buf's own write is always scratch reused in place every
        # iteration (never advancing) -- unlike squeezed_advance_per_read
        # above, there is no fresh per-write computation here to override
        # sizing_op_info.squeezed_advance_output with, so it must be forced
        # to [] explicitly or it silently carries over sizing_op's own
        # (unrelated) output advance via this dataclasses.replace, wrongly
        # marking this LX write as advancing.
        squeezed_advance_output=[],
        propagation=PropagationPlan(kind="loop_internal"),
    )

    V.graph.name_to_buffer[copy_name] = copy_buf
    operations.insert(insert_idx, copy_buf)

    logger.debug(
        "coarse_tile: read copy-in %s -> %s",
        dep.name,
        copy_name,
    )
    return copy_buf.get_name()


def _patch_consumer_to_read_copy(
    consumer: ComputedBuffer,
    dep: MemoryDep,
    copy_name: str,
    operations: list[Operation],
    *,
    loop_invariant: bool = False,
) -> None:
    """Patch consumer's inner_fn to read copy_name instead of dep.name.

    consumer's own read index for dep.name is affine against full_buf's
    full-sized strides (dep.index, structurally, though the free variables
    at any given trace of consumer's inner_fn may not be dep.var_names
    themselves -- see _NameSwapHandler). The copy (copy_name) is a smaller,
    freshly allocated buffer with its own contiguous tile-local strides, so
    _NameSwapHandler rescales the index's coefficients from full_strides
    (dep.index's own per-dimension coefficients) to tile_strides
    (recomputed here from copy_name's own current layout) at call time --
    see _NameSwapHandler and _rescale_index. Rebuilds consumer in place
    (splicing the fresh ComputedBuffer into operations, replacing the stale
    one) via replace_computed_buffer_body, exactly as today.
    """
    full_strides = [dep.index.coeff(v) for v in dep.var_names]
    # Keep the dependency-coordinate positions separate from buffer-layout
    # strides appended below.  The latter make _rescale_index complete, but
    # they are not dimensions of a compact invariant copy.
    pre_extension_strides = list(full_strides)
    # dep is sizing_op's own (upstream-Inductor-squeezed) MemoryDep for this
    # read, so dep.var_names only covers host dims sizing_op's tile-extent
    # keeps distinct -- a host dim tiled down to extent 1 (e.g. a B-tiled
    # group where sizing_op's own per-tile B range is 1) is squeezed out of
    # dep entirely and has no coefficient here at all. But consumer's own
    # load index is retraced independently of dep (see below) and may still
    # carry a real term for that host dim, using full_buf's actual stride --
    # it hasn't been squeezed at consumer's own trace site just because
    # sizing_op's tile happens to be 1 wide there. _rescale_index matches
    # every term in consumer's index by coefficient value, so every host dim
    # of full_buf needs an entry here, not just the ones dep kept. Such a
    # dim's tile_strides entry is 0: the copy buffer has no distinct index
    # for it either (same "no advance" convention as _fixed_level_extents),
    # so any load through it correctly always resolves to offset 0 for that
    # dim regardless of the loop variable's value.
    full_buf = V.graph.get_buffer(dep.name)
    if isinstance(full_buf, TensorBox):
        full_buf = full_buf.data
    if isinstance(full_buf, StorageBox):
        full_buf = full_buf.data
    dep_strides = set(full_strides)
    for host_stride in full_buf.layout.stride:
        if host_stride not in dep_strides:
            full_strides.append(host_stride)
            dep_strides.add(host_stride)
    copy_buf = next(
        op
        for op in operations
        if isinstance(op, ComputedBuffer) and op.get_name() == copy_name
    )
    copy_strides = list(copy_buf.layout.stride)
    active_idx = [i for i, stride in enumerate(pre_extension_strides) if stride != 0]
    if loop_invariant and len(copy_strides) == len(active_idx):
        # Expand a compact invariant copy's real strides back into the
        # consumer's iteration-space positions.  Broadcast loop dimensions
        # remain fixed at zero instead of shifting the real tensor strides.
        tile_strides = [sympy.Integer(0)] * len(pre_extension_strides)
        for pos, stride in zip(active_idx, copy_strides):
            tile_strides[pos] = stride
    else:
        tile_strides = copy_strides
    tile_strides.extend(
        sympy.Integer(0) for _ in range(len(full_strides) - len(tile_strides))
    )
    name_map: dict[str, tuple[str, list[Expr], list[Expr]]] = {
        dep.name: (copy_name, full_strides, tile_strides)
    }

    # Patch consumer's inner_fn once with the one-entry name_map (wrap, not
    # reconstruct — see _NameSwapHandler docstring).  Rebuild via
    # replace_computed_buffer_body, matching every other inner_fn-rewrite
    # site in this file (_patch_consumers, _patch_retiled_load_indexes):
    # a fresh ComputedBuffer has no stale per-object
    # caches, sidestepping the need to enumerate every cache key by hand.
    from ..pass_utils import replace_computed_buffer_body

    orig_inner = consumer.data.inner_fn
    existing_record = getattr(consumer, "_read_copy_elision_record", None)
    original_loop_info = consumer.loop_info  # type: ignore[attr-defined]
    original_reads = [
        read for read in consumer.get_read_writes().reads if isinstance(read, MemoryDep)
    ]
    original_dep_idx = next(
        (idx for idx, original_dep in enumerate(original_reads) if original_dep == dep),
        None,
    )
    original_read_advances = original_dep_idx is not None and (
        (
            original_dep_idx < len(original_loop_info.tiled_dims_per_read)
            and any(original_loop_info.tiled_dims_per_read[original_dep_idx])
        )
        or (
            original_dep_idx < len(original_loop_info.squeezed_advance_per_read)
            and any(original_loop_info.squeezed_advance_per_read[original_dep_idx])
        )
    )
    direct_read_candidate = (
        not loop_invariant
        and original_read_advances
        and dep.name in V.graph.graph_input_names
        and isinstance(consumer.data, Reduction)
        and consumer.data.reduction_type in MATMUL_REDUCTION_OPS
    )

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(_NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(consumer.data, "inner_fn", new_inner_fn)
    new_op = replace_computed_buffer_body(
        consumer,
        consumer.data,
        operations,
        pass_name="coarse_tile",
        reason="redirect consumer to copied inputs",
    )
    V.graph.name_to_buffer[new_op.get_name()] = new_op
    if isinstance(existing_record, ReadCopyElisionRecord):

        def updated_direct_inner(
            *args,
            _map=name_map,
            _direct_inner=existing_record.direct_inner_fn,
        ):
            with V.set_ops_handler(_NameSwapHandler(V.ops, _map)):
                return _direct_inner(*args)

        new_op._read_copy_elision_record = dataclasses.replace(  # type: ignore[attr-defined]
            existing_record,
            direct_inner_fn=updated_direct_inner,
        )
    if direct_read_candidate:
        new_op._read_copy_elision_record = ReadCopyElisionRecord(  # type: ignore[attr-defined]
            consumer_name=new_op.get_name(),
            copy_name=copy_name,
            source_name=dep.name,
            direct_inner_fn=orig_inner,
        )

    # new_op.loop_info (copied from consumer by copy_op_metadata inside
    # replace_computed_buffer_body) still carries tiled_dims_per_read as
    # planned when this op's own read of dep.name was full_buf directly --
    # whole-tile advance, correct at plan time. But new_op's actual read
    # (per get_read_writes(), re-derived from the now-patched inner_fn) is
    # the copy buffer: scratch reused in place every iteration, which must
    # not advance, exactly like _insert_copy_op's own read side.
    # SpyreKernel._general_tile_advance matches tiled_dims_per_read to
    # get_read_writes().reads purely positionally (see loop_info.py's
    # docstring), so the entry corresponding to the swapped-in copy-buffer
    # read must be zeroed (dim omitted, see _fixed_level_extents).
    new_loop_info = new_op.loop_info  # type: ignore[attr-defined]
    new_reads = [r for r in new_op.get_read_writes().reads if isinstance(r, MemoryDep)]
    new_tiled_dims_per_read = new_loop_info.tiled_dims_per_read
    if new_tiled_dims_per_read:
        assert len(new_reads) == len(new_loop_info.tiled_dims_per_read), (
            f"_patch_consumer_to_read_copy: positional mismatch between "
            f"new_op.get_read_writes().reads ({len(new_reads)} entries) and "
            f"new_loop_info.tiled_dims_per_read ({len(new_loop_info.tiled_dims_per_read)} "
            "entries) -- SpyreKernel._general_tile_advance matches these purely "
            "positionally, so a length mismatch means silently wrong tile-advance "
            "metadata rather than a loud failure."
        )
        fixed_level_extents = _fixed_level_extents(new_loop_info.loop_tiled_dims)
        new_tiled_dims_per_read = [
            (
                _tiled_dims_for_dep(read_dep, fixed_level_extents, new_op)
                if read_dep.name == copy_name
                else per_level
            )
            for read_dep, per_level in zip(new_reads, new_loop_info.tiled_dims_per_read)
        ]

    new_squeezed_advance_per_read = new_loop_info.squeezed_advance_per_read
    if new_squeezed_advance_per_read:
        assert len(new_reads) == len(new_squeezed_advance_per_read), (
            "_patch_consumer_to_read_copy: positional mismatch between "
            f"new_op.get_read_writes().reads ({len(new_reads)} entries) and "
            "new_loop_info.squeezed_advance_per_read "
            f"({len(new_squeezed_advance_per_read)} entries)"
        )
        new_squeezed_advance_per_read = [
            (
                [[] for _ in new_loop_info.loop_count]
                if read_dep.name == copy_name
                else per_level
            )
            for read_dep, per_level in zip(new_reads, new_squeezed_advance_per_read)
        ]

    new_op.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
        new_loop_info,
        tiled_dims_per_read=new_tiled_dims_per_read,
        squeezed_advance_per_read=new_squeezed_advance_per_read,
    )


def _plan_read_copies(
    operations: list[Operation],
    retiled_infos_by_group: list[
        tuple[tuple[int, ...], list[Operation], dict[str, "_RetiledBufferInfo"]]
    ],
    predivision_unit_steps_by_op: dict[
        int,
        tuple[tuple[tuple[tuple[int, Expr, Expr], ...], ...], ...],
    ]
    | None = None,
) -> dict[tuple[int, ...], ReadCopyPlan]:
    """Plan Pass 1's read-copy sharing, with zero mutation.

    For each group, collects every ComputedBuffer op's
    _full_buffer_read_deps and groups equivalent reads (same buffer name,
    same per-var index coefficients, same size -- see the canonical key
    below) into one ReadCopyEntry each, regardless of whether the
    equivalent reads came from the same op or from different ops in the
    group. The first op (operations order) with an equivalent read supplies
    both insert_before_op_name and sizing_op_name; every op in the group
    with an equivalent read is recorded in consumer_op_names.

    Must run after every group's _apply_plan (see _coarse_tile_common's body):
    _full_buffer_read_deps requires op.loop_info to be stamped and
    op.get_read_writes() to reflect post-division ranges, neither of which
    holds before _apply_plan runs for that op's group.
    """
    op_position = {op.get_operation_name(): i for i, op in enumerate(operations)}
    operations_by_name = {
        op.get_name(): op for op in operations if isinstance(op, ComputedBuffer)
    }
    loop_written_names = _loop_written_buffer_names(operations)
    predivision_unit_steps_by_op = predivision_unit_steps_by_op or {}
    plans: dict[tuple[int, ...], ReadCopyPlan] = {}

    for stamped_group_id, group_ops, _retiled_infos in retiled_infos_by_group:
        # canonical key -> list of (op, dep) in operations order.
        keyed: dict[tuple, list[tuple[Operation, MemoryDep]]] = {}
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            if not isinstance(op.data, (Pointwise, Reduction)):
                continue
            for dep in _full_buffer_read_deps(op):
                # dep.index.coeff(v) is a *linear* coefficient: it is blind
                # to any constant offset in the index (e.g. 64*d0 + d1 and
                # 64*d0 + d1 + 5 have identical coeffs). Two reads that
                # differ only in a constant offset (e.g. a shifted/windowed
                # read) must not collapse to the same key -- _insert_one_
                # read_copy sizes the shared copy from only the sizing
                # op's own dep.index, so a merged-in consumer at a
                # different real offset would silently read wrong or
                # out-of-bounds data. Include the offset explicitly.
                offset = dep.index - sum(dep.index.coeff(v) * v for v in dep.var_names)
                key = (
                    dep.name,
                    tuple(dep.index.coeff(v) for v in dep.var_names),
                    offset,
                    tuple(dep.size),
                )
                keyed.setdefault(key, []).append((op, dep))

        entries: list[ReadCopyEntry] = []
        for n, (key, op_deps) in enumerate(keyed.items()):
            # Order this key's (op, dep) pairs by their position in
            # `operations`, not by group_ops's own order, so
            # insert_before_op_name/sizing_op_name pick the op that is
            # actually first in the real operations list.
            op_deps.sort(key=lambda pair: op_position[pair[0].get_operation_name()])
            sizing_op, sizing_dep = op_deps[0]
            sizing_info = sizing_op.loop_info  # type: ignore[attr-defined]
            hoist_decisions: list[_ReadCopyHoistDecision] = []
            for candidate_op, candidate_dep in op_deps:
                candidate_info = candidate_op.loop_info  # type: ignore[attr-defined]
                candidate_reads = [
                    read
                    for read in candidate_op.get_read_writes().reads
                    if isinstance(read, MemoryDep)
                ]
                # reads is an OrderedSet: identical MemoryDeps collapse to one
                # equality class, so index() cannot select a different copy of
                # the same dependency with different movement metadata.
                dep_idx = candidate_reads.index(candidate_dep)
                decision = _read_copy_hoist_decision(
                    candidate_info,
                    dep_idx,
                    _read_copy_source_info(
                        candidate_dep.name,
                        operations_by_name,
                        loop_written_names,
                    ),
                )
                hoist_decisions.append(decision)
                logger.debug(
                    "coarse_tile: read-copy hoist decision consumer=%s "
                    "source=%s decision=%s",
                    candidate_op.get_operation_name(),
                    candidate_dep.name,
                    decision.name,
                )
            for other_op, _dep in op_deps[1:]:
                other_info = other_op.loop_info  # type: ignore[attr-defined]
                # Only loop_count (the per-level trip counts) is a genuine
                # per-*group* invariant that every op in the group shares by
                # construction -- and it is the only one of the two fields
                # _insert_one_read_copy actually consumes from sizing_op_info
                # to size the shared copy's read_level_extents (see its
                # body). loop_tiled_dims is *not* comparable across op kinds:
                # its dim keys are positions into each op's own data.ranges
                # (output-shaped), while a Reduction's own tiling of a
                # reduction dim shows up in loop_tiled_reduction_dims
                # instead, in a completely different numbering space. A
                # Reduction (e.g. softmax's max) and a sibling Pointwise
                # (e.g. softmax's x - max) can legitimately tile the very
                # same logical tensor dim through these two different
                # fields and still correctly share one copy -- the plan
                # doesn't need loop_tiled_dims to build the copy either way.
                if other_info.loop_count != sizing_info.loop_count:
                    raise Unsupported(
                        "_plan_read_copies: ops in the same coarse-tile "
                        f"group disagree on loop_count ({other_op.get_name()!r} "
                        f"vs {sizing_op.get_name()!r} sizing this shared copy "
                        f"of {key[0]!r}) -- ops in one group must share trip "
                        "counts by construction."
                    )
            group_tag = "_".join(str(i) for i in stamped_group_id)
            copy_name = V.graph.qualify_name(
                f"coarse_tile_read_copy_{group_tag}_{key[0]}_{n}"
            )
            assert copy_name.isidentifier(), f"invalid copy buffer name: {copy_name!r}"
            sizing_reads = [
                read
                for read in sizing_op.get_read_writes().reads
                if isinstance(read, MemoryDep)
            ]
            sizing_read_index = sizing_reads.index(sizing_dep)
            sizing_steps = predivision_unit_steps_by_op.get(id(sizing_op), ())
            predivision_unit_steps = (
                sizing_steps[sizing_read_index]
                if sizing_read_index < len(sizing_steps)
                else ()
            )
            entries.append(
                ReadCopyEntry(
                    copy_name=copy_name,
                    dep=sizing_dep,
                    insert_before_op_name=sizing_op.get_operation_name(),
                    sizing_op_name=sizing_op.get_operation_name(),
                    sizing_read_index=sizing_read_index,
                    consumer_op_names=tuple(
                        op.get_operation_name() for op, _dep in op_deps
                    ),
                    predivision_unit_steps=predivision_unit_steps,
                    # This one decision controls both preheader placement and
                    # compact staging in _insert_one_read_copy. They require
                    # the same proof: every consumer sees the same source
                    # slice on every counted-loop trip.
                    loop_invariant=all(
                        decision is _ReadCopyHoistDecision.ELIGIBLE
                        for decision in hoist_decisions
                    ),
                )
            )
        if entries:
            plans[stamped_group_id] = ReadCopyPlan(entries=tuple(entries))

    return plans


def _insert_all_read_copy_ops(
    operations: list[Operation],
    read_copy_plans: dict[tuple[int, ...], ReadCopyPlan],
) -> None:
    """Pass 1: execute a precomputed ReadCopyPlan per group.

    Transformation's Pass 1 (see the plan/execute split design and
    _plan_read_copies). All sharing/dedup decisions were already made by
    _plan_read_copies -- this function only builds and inserts the copy
    ops it named and patches the consumers it named. Must run after every
    group's _apply_plan (so op.loop_info is stamped -- see
    _plan_read_copies) and before Pass 2/3's Reduction/copy-out dispatch,
    which reads an op's *current* reads/loader.
    """
    for plan in read_copy_plans.values():
        for entry in plan.entries:
            name_to_op = {
                op.get_operation_name(): op
                for op in operations
                if isinstance(op, ComputedBuffer)
            }
            sizing_op = name_to_op[entry.sizing_op_name]
            insert_before_op = name_to_op[entry.insert_before_op_name]
            new_copy_name = _insert_one_read_copy(
                sizing_op,
                entry.dep,
                entry.sizing_read_index,
                entry.copy_name,
                operations,
                insert_before_op=insert_before_op,
                predivision_unit_steps=entry.predivision_unit_steps,
                loop_invariant=entry.loop_invariant,
            )
            for consumer_name in entry.consumer_op_names:
                consumer = name_to_op[consumer_name]
                _patch_consumer_to_read_copy(
                    consumer,
                    entry.dep,
                    new_copy_name,
                    operations,
                    loop_invariant=entry.loop_invariant,
                )


# ---------------------------------------------------------------------------
# Case: reduction-dim tiling — combine op insertion
# ---------------------------------------------------------------------------


def _insert_combine_op(
    tiled_op: ComputedBuffer,
    accum_buf: ComputedBuffer,
    operations: list[Operation],
    is_nested: bool,
) -> str:
    """Insert a pointwise combine op that accumulates tiled_op into accum_buf.

    The combine op reads both the partial result (tiled_op) and the current
    accumulation buffer and writes the combined value back into accum_buf via
    MutationLayoutSHOULDREMOVE.  It carries tiled_op's loop_group_id/
    loop_count/loop_tiled_dims/loop_tiled_reduction_dims (so the scheduler
    places it inside the same CountedLoopSchedulerNode), but its own
    freshly-derived tiled_dims_per_read/output_tiled_dims -- combine_buf's
    two reads and one write don't correspond positionally to tiled_op's own
    reads/write.

    tiled_op's own partial output is per-tile scratch reused in place every
    inner iteration, so the combine's read of it must not advance (see
    _fixed_level_extents).

    accum_buf's own reads/write depend on is_nested, because _caller_ passes a
    different buffer for each case (see _propagate_tiled_reduction_op):

    - Nested (is_nested=True): accum_buf is accum_tile, a per-outer-tile
      scratch buffer that a separate reduce-copy op drains into accum_full at
      the outer loop boundary -- accum_tile itself must not advance at any
      level, same as tiled_op's partial (_fixed_level_extents).
    - Flat (is_nested=False): accum_buf is accum_full itself, the real
      persistent output-shaped buffer with no separate copy op -- when a kept
      (non-reduction) output dim is tiled at some level, accum_full
      legitimately advances at that level so each iteration combines into the
      correct slice, exactly like _insert_copy_op's write side into full_buf
      (see _advancing_level_extents).

    Using _fixed_level_extents for accum_buf's reads/write unconditionally in
    both cases (as an earlier version of this function did) always pinned the
    flat case's mutation-write to the first slice, leaving every other slice
    stuck at the fill/identity value -- this previously surfaced as a
    spurious advancing-lx NotImplementedError once a since-removed per-buffer
    per_tile_fixed flag that had been masking it was taken away, and after
    that flag's removal regressed into silent wrong numerics instead
    (test_min_2d_512x256_reduce_dim0_A4_B4).
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

    # tiled_op's own partial output is per-tile-fixed scratch that never
    # advances across the inner tiled loop (all-empty per-level extents, same
    # convention _insert_copy_op's read side uses). accum_buf's own extents
    # depend on is_nested -- see this function's docstring.
    tiled_op_info = tiled_op.loop_info  # type: ignore[attr-defined]
    fixed_level_extents = _fixed_level_extents(tiled_op_info.loop_tiled_dims)
    accum_level_extents = (
        fixed_level_extents
        if is_nested
        else _advancing_level_extents(
            tiled_op_info.loop_tiled_dims,
            tiled_op_info.loop_count,
            list(combine_data.ranges),
        )
    )
    combine_reads = [
        dep for dep in combine_buf.get_read_writes().reads if isinstance(dep, MemoryDep)
    ]
    combine_writes = [
        dep
        for dep in combine_buf.get_read_writes().writes
        if isinstance(dep, MemoryDep)
    ]
    tiled_dims_per_read = [
        _tiled_dims_for_dep(
            dep,
            fixed_level_extents
            if dep.name == tiled_op.get_name()
            else accum_level_extents,
            combine_buf,
        )
        for dep in combine_reads
    ]
    output_tiled_dims = (
        _tiled_dims_for_dep(combine_writes[0], accum_level_extents, combine_buf)
        if combine_writes
        else []
    )
    combine_buf.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
        tiled_op_info,
        tiled_dims_per_read=tiled_dims_per_read,
        output_tiled_dims=output_tiled_dims,
    )
    V.graph.name_to_buffer[combine_name] = combine_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, combine_buf)

    logger.debug(
        "coarse_tile: combine %s accum_buf=%s is_nested=%s "
        "tiled_dims_per_read=%s output_tiled_dims=%s",
        combine_name,
        accum_buf.get_name(),
        is_nested,
        tiled_dims_per_read,
        output_tiled_dims,
    )
    return combine_name


def _collapse_loop_carried_unit_sum(
    tiled_op: ComputedBuffer,
    combine_name: str,
    operations: list[Operation],
) -> ComputedBuffer:
    """Replace a one-expert local sum by its single contribution.

    The combine op performs the real cross-expert sum.  Once coarse tiling has
    reduced the expert extent to one, retaining a second Reduction is both
    redundant and unrepresentable by the backend.
    """

    data = tiled_op.data
    info = getattr(tiled_op, "loop_info", None)
    plan = getattr(getattr(info, "propagation", None), "reduction", None)
    if not (
        isinstance(data, Reduction)
        and data.reduction_type == "sum"
        and data.src_dtype == data.dtype
        and info is not None
        and plan is not None
        and plan.carried is not None
        and len(data.reduction_ranges) == 1
        and sympy.sympify(data.reduction_ranges[0]) == 1
    ):
        raise Unsupported(
            f"coarse_tile: carried sum {tiled_op.get_name()} did not reduce "
            "to one contribution per loop iteration"
        )

    combine = V.graph.name_to_buffer.get(combine_name)
    if not (
        isinstance(combine, ComputedBuffer)
        and isinstance(combine.data, Pointwise)
        and isinstance(combine.layout, MutationLayoutSHOULDREMOVE)
        and _reads_buffer(combine, tiled_op.get_name())
    ):
        raise Unsupported(
            f"coarse_tile: carried sum {tiled_op.get_name()} has no matching "
            f"combine op {combine_name}"
        )

    reduction_inner_fn = data.inner_fn

    def contribution_inner_fn(index):
        return reduction_inner_fn(index, [sympy.Integer(0)])

    contribution_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=contribution_inner_fn,
        ranges=list(data.ranges),
    )
    from ..provenance import preserve_provenance

    preserve_provenance(
        data,
        contribution_data,
        pass_name="coarse_tile",
        reason="collapse one-expert sum to loop contribution",
    )
    from ..pass_utils import replace_computed_buffer_body

    before_symbols = _capture_logical_iteration_symbols(tiled_op)

    contribution = replace_computed_buffer_body(
        tiled_op,
        contribution_data,
        operations,
        pass_name="coarse_tile",
        reason="collapse one-expert sum to loop contribution",
    )
    _apply_work_div_symbol_remap(
        contribution,
        _order_preserving_symbol_remap(
            contribution,
            before_symbols,
            _capture_logical_iteration_symbols(contribution),
        ),
    )
    V.graph.name_to_buffer[contribution.get_name()] = contribution
    return contribution


def _copy_carried_output_dim_names(
    source: ComputedBuffer,
    target: ComputedBuffer,
) -> None:
    """Copy named dimensions across this order-preserving local rewrite."""

    target.work_div_loop_info = dict(  # type: ignore[attr-defined]
        getattr(source, "work_div_loop_info", {})
    )
    _apply_work_div_symbol_remap(
        target,
        _order_preserving_symbol_remap(
            target,
            _capture_logical_iteration_symbols(source),
            _capture_logical_iteration_symbols(target),
        ),
    )


def _insert_flat_reduction_drain_op(
    tiled_op: ComputedBuffer,
    accumulator: ComputedBuffer,
    output: ComputedBuffer,
    combine_name: str,
    operations: list[Operation],
) -> ComputedBuffer:
    """Drain a carried accumulator to its graph output once after the loop."""

    device = tiled_op.get_device()
    assert device is not None
    drain_data = Pointwise(
        device=device,
        dtype=tiled_op.get_dtype(),
        inner_fn=accumulator.make_loader(),
        ranges=list(tiled_op.data.ranges),
    )
    drain_name = V.graph.qualify_name(
        f"coarse_tile_reduction_drain_{tiled_op.get_name()}"
    )
    drain_buf = ComputedBuffer(
        name=drain_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(output))),
        data=drain_data,
    )
    drain_buf.origins = tiled_op.origins
    drain_buf.operation_name = drain_name
    drain_buf._coarse_tile_force_live = True  # type: ignore[attr-defined]
    V.graph.name_to_buffer[drain_name] = drain_buf

    combine_buf = V.graph.name_to_buffer[combine_name]
    assert isinstance(combine_buf, ComputedBuffer)
    operations.insert(operations.index(combine_buf) + 1, drain_buf)
    return drain_buf


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

    Reads accum_tile (never advances) and writes into accum_full via
    MutationLayoutSHOULDREMOVE.  Carries outer_loop_info so the unroller
    advances accum_full per outer output-dim tile.

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

    # outer_loop_info.output_tiled_dims is [] -- correct for the fill op
    # (which writes accum_tile in-place and never advances) but wrong here:
    # this copy op writes accum_full, which is NOT divided, so its store base
    # must advance a full outer tile per outer iteration. Derive real
    # per-level extents the same way _insert_copy_op does for its write side:
    # innermost tiled level's extent is the per-tile range itself, each level
    # out from there multiplies by the next-inner level's trip count.
    copy_ranges = list(copy_data.ranges)
    write_level_extents: list[dict[int, Expr]] = [
        {} for _ in outer_loop_info.loop_tiled_dims
    ]
    for d in {d for level in outer_loop_info.loop_tiled_dims for d in level}:
        levels_tiling_d = [
            i for i, dims in enumerate(outer_loop_info.loop_tiled_dims) if d in dims
        ]
        running = sympy.sympify(copy_ranges[d])
        for level_idx in reversed(levels_tiling_d):
            write_level_extents[level_idx][d] = running
            running = running * outer_loop_info.loop_count[level_idx]
    copy_writes = [
        dep for dep in copy_buf.get_read_writes().writes if isinstance(dep, MemoryDep)
    ]
    output_tiled_dims = (
        _tiled_dims_for_dep(copy_writes[0], write_level_extents, copy_buf)
        if copy_writes
        else []
    )
    copy_buf.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
        outer_loop_info, output_tiled_dims=output_tiled_dims
    )
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


def _insert_all_reduction_ops(operations: list[Operation]) -> None:
    """Pass 2: build reduction machinery for every planned reduction op.

    Transformation's Pass 2 (see the plan/execute split design). Every op
    was already stamped by _apply_plan with a loop_info carrying
    .propagation, computed by _plan_tiling_propagation -- this pass only
    consumes that decision (kind == "reduction" and its accompanying
    ReductionPlan shape/identity/nesting data), it makes no new ones.

    Must run after Pass 1 (_insert_all_read_copy_ops) -- a tiled-reduction
    op may itself have needed a read copy-in, and this pass's accumulator/
    fill/combine construction reads op's *current* reads/loader -- and
    before Pass 3, since a reduction op is never also copy_out (the plan's
    kind routes each op to exactly one).
    """
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        loop_info = getattr(op, "loop_info", None)
        propagation = getattr(loop_info, "propagation", None)
        if propagation is None or propagation.kind != "reduction":
            continue
        _propagate_tiled_reduction_op(op, operations)


def _propagate_tiled_reduction_op(
    op: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Handle buffer propagation for a Reduction op tiled over a reduction dim.

    Strategy: fill-initialize + per-tile combine.
      1. Allocate a HBM accumulation buffer sized to the full
         (pre-outer-division) output shape (planned as
         reduction.full_output_ranges), so that address advancement across
         outer tiles writes each tile into the correct slice.  For flat
         (reduction-only) tiling this equals op.data.ranges.
      2. Insert a fill op that writes the reduction's identity value into the
         accumulation buffer.  For flat reduction tiling the fill has no
         loop_info and runs before all loops.  For nested tiling (outer
         output-dim loop + inner reduction loop) the fill carries the outer
         loop's loop_info so it runs inside the outer loop — once per outer
         tile — keeping the accumulator sized to the per-tile output shape.
      3. Insert a combine op (inside the inner loop, same loop_info as the
         tiled reduction op) that merges each tile's partial result into the
         accumulation buffer using the reduction's combining fn.
      4. Mark the tiled reduction op's output as inner-loop scratch (not
         advanced between inner iterations).
      5. Patch outside consumers and graph outputs to read the accumulation
         buffer.
    """
    loop_info = op.loop_info
    loop_group_id = loop_info.loop_group_id
    reduction_plan = loop_info.propagation.reduction
    identity = reduction_plan.identity
    op_size = tuple(op.layout.size)

    # Per-outer-tile output shape (ranges after any outer tiling divided them).
    per_tile_ranges = reduction_plan.per_tile_ranges

    # Accumulation buffer uses the full (pre-outer-division) output shape so
    # that address advancement across outer output-dim tiles writes each tile's
    # result into the correct slice.  For reduction-dim-only tiling there is no
    # outer division, so full == per-tile.
    full_output_ranges = reduction_plan.full_output_ranges

    # Insert HBM buffer before the first op in the loop group.
    outer_key = loop_group_id[0]
    group_start_idx = next(
        i
        for i, o in enumerate(operations)
        if isinstance(o, ComputedBuffer)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
    )

    fill_loop_info = reduction_plan.outer_fill_loop_info
    is_nested = reduction_plan.is_nested
    carried = reduction_plan.carried
    use_loop_carried_accumulator = carried is not None

    if fill_loop_info is not None:
        # outer_fill_loop_info was built at planning time, before _apply_plan
        # stamped op's real, offset-adjusted loop_group_id -- its own
        # loop_group_id is still the pre-offset internal numbering
        # plan_coarse_tile_groups used only for its own bookkeeping (see
        # coarse_tile_pre_stickify's/coarse_tile_post_stickify's comment on
        # group_idx_offset). Re-slice from
        # op.loop_info's now-real loop_group_id so the fill/copy ops this
        # function stamps with fill_loop_info end up in the same outer group
        # as every other op here, not a stale, potentially colliding one.
        fill_loop_info = dataclasses.replace(
            fill_loop_info,
            loop_group_id=loop_group_id[: len(fill_loop_info.loop_count)],
        )

    if is_nested:
        # Nested case: allocate separate tile-sized and full-sized buffers.
        # accum_tile stays inside the inner K-loop (never advances there --
        # its only readers are the combine op and the outer copy op, both of
        # which build their own correct tiled_dims_per_read/output_tiled_dims
        # from their own loop_info, so accum_tile itself needs no flag);
        # accum_full accumulates across outer B-tiles via a copy op.
        accum_full = _allocate_full_buffer(
            op,
            full_output_ranges,
            reduction_plan.full_output_strides,
            operations,
            group_start_idx,
        )
        group_start_idx_after_full = operations.index(accum_full) + 1
        accum_tile = _allocate_full_buffer(
            op,
            per_tile_ranges,
            reduction_plan.per_tile_strides,
            operations,
            group_start_idx_after_full,
        )
        fill_target = accum_tile
        combine_target = accum_tile
    else:
        # Flat case: single full-sized buffer.
        accum_full = _allocate_full_buffer(
            op,
            full_output_ranges,
            reduction_plan.full_output_strides,
            operations,
            group_start_idx,
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
    assert device is not None

    scalar_op = SpyreConstantFallback(
        torch.ops.spyre.constant.default, float(identity), dtype, device
    )
    # SpyreTensorLayout([], dtype) yields device_size=[1, 64], stride_map=[-1, -1]
    # — a 0-d broadcast scalar in Spyre's device coordinate system.
    scalar_stl = SpyreTensorLayout([], dtype)
    scalar_op.layout = FixedTiledLayout(device, dtype, [], [], scalar_stl)
    scalar_loader = TensorBox.create(scalar_op).make_loader()

    # fill_target's shape matches per_tile_ranges when nested (accum_tile, a
    # per-outer-tile scratch buffer re-seeded every outer iteration) but
    # full_output_ranges when flat (accum_full itself, initialized once) --
    # for a flat tiling where an output dim is nonetheless divided (e.g. an
    # output-dim level inner to the reduction level), per_tile_ranges is
    # smaller than fill_target's actual full-sized allocation.
    fill_ranges = per_tile_ranges if is_nested else full_output_ranges
    fill_data = Pointwise(
        device=device,
        dtype=dtype,
        inner_fn=lambda index, _loader=scalar_loader: _loader([]),
        ranges=fill_ranges,
    )
    fill_name = V.graph.qualify_name(f"coarse_tile_fill_{op.get_name()}")
    if use_loop_carried_accumulator:
        # This ordinary buffer is the one value carried across expert-loop
        # iterations.  It is deliberately separate from the HBM graph output,
        # which the suffix drain writes once after the loop.
        if isinstance(fill_target.layout, FixedTiledLayout):
            fill_layout: FixedLayout = FixedTiledLayout(
                device,
                dtype,
                list(fill_target.layout.size),
                list(fill_target.layout.stride),
                fill_target.layout.device_layout,
            )
        else:
            fill_layout = FixedLayout(
                device,
                dtype,
                list(fill_target.layout.size),
                list(fill_target.layout.stride),
            )
        fill_buf = ComputedBuffer(
            name=fill_name,
            layout=fill_layout,
            data=fill_data,
        )
        _copy_carried_output_dim_names(op, fill_buf)
        combine_target = fill_buf
    else:
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
    combine_name = _insert_combine_op(
        op,
        combine_target,
        operations,
        is_nested or use_loop_carried_accumulator,
    )

    if carried is not None:
        combine_buf = V.graph.name_to_buffer[combine_name]
        assert isinstance(combine_buf, ComputedBuffer)
        _copy_carried_output_dim_names(op, combine_buf)
        op = _collapse_loop_carried_unit_sum(op, combine_name, operations)
        drain_buf = _insert_flat_reduction_drain_op(
            op, combine_target, accum_full, combine_name, operations
        )
        _copy_carried_output_dim_names(op, drain_buf)
        record = CarriedReductionRecord(
            accumulator_name=combine_target.get_name(),
            row_dim_name=carried.row_dim_name,
            required_row_split=carried.required_row_split,
            fill_name=fill_name,
            combine_name=combine_name,
            drain_name=drain_buf.get_name(),
        )
        fill_buf._carried_reduction_record = record  # type: ignore[attr-defined]
        combine_buf._carried_reduction_record = record  # type: ignore[attr-defined]
        drain_buf._carried_reduction_record = record  # type: ignore[attr-defined]

    # For nested case, insert a copy op at the outer loop level that writes
    # accum_tile → accum_full, advancing accum_full across outer output tiles.
    if is_nested:
        assert fill_loop_info is not None  # guaranteed by is_nested == True
        _insert_reduction_copy_op(
            op, accum_tile, accum_full, fill_loop_info, operations
        )

    # The tiled reduction op's own write is per-tile scratch: it is drained by
    # the combine op every inner iteration and never read directly by the
    # outer copy op (which reads accum_tile instead), so it must not advance
    # at any level.
    loop_info.output_tiled_dims = []

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

    # Consumers INSIDE the same outermost loop group may also need
    # redirecting: any such consumer that currently reads op's own per-tile
    # scratch buffer (buf_name) directly, rather than accum_full, sees
    # whatever partial value that scratch buffer holds at the point it
    # happens to run -- correct only once the reduction has fully
    # accumulated. The safety condition differs by nesting mode:
    #
    # Nested (is_nested=True): the reduce_copy op writes accum_tile ->
    # accum_full at the *outer* loop boundary, so any inside consumer that
    # runs after it within the same outer-tile iteration sees the fully
    # accumulated value for that tile. All inside consumers are safe to
    # redirect.
    #
    # Flat (is_nested=False): the combine op accumulates directly into
    # accum_full via MutationLayout inside the (possibly multi-level)
    # reduction loop itself. A consumer is safe to redirect only if its own
    # loop_tiled_dims exactly matches op's -- both then advance through
    # accum_full along exactly the same dimensions at exactly the same rate,
    # so by the time the consumer's tile is reached, every reduction-dim
    # tile contributing to it has already combined. A consumer with EXTRA
    # tiled dimensions (e.g. it also tiles a dim op's reduction loop doesn't,
    # or vice versa) could run before accum_full is fully combined for its
    # slice -- those consumers are left reading buf_name (per-tile scratch).
    combine_name = V.graph.qualify_name(f"coarse_tile_combine_{buf_name}")
    copy_name = V.graph.qualify_name(f"coarse_tile_reduce_copy_{buf_name}")
    outer_key = loop_group_id[0]
    inside_consumers = [
        o
        for o in operations
        if isinstance(o, ComputedBuffer)
        and o.get_name() not in (combine_name, copy_name)
        and _reads_buffer(o, buf_name)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
        and (
            is_nested
            or getattr(getattr(o, "loop_info", None), "loop_tiled_dims", None)
            == loop_info.loop_tiled_dims
        )
    ]

    all_consumers = outside_consumers + inside_consumers
    accum_name = accum_full.get_name()
    retile_info = _RetiledBufferInfo(
        tuple(op.layout.stride),
        tuple(accum_full.layout.stride),
        op_size,
    )
    _patch_consumers(all_consumers, buf_name, accum_name, operations, retile_info)
    if is_graph_output:
        _patch_graph_outputs(buf_name, accum_full)

    logger.debug(
        "coarse_tile: tiled reduction %s -> accum_full %s (fill=%s, combine=%s, "
        "identity=%s, nested=%s)",
        buf_name,
        accum_name,
        fill_name,
        combine_name,
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

    from ..pass_utils import NameSwapHandler, replace_computed_buffer_body

    name_map = {old_name: new_name}
    has_retile = (
        retile_info is not None and retile_info.old_stride != retile_info.new_stride
    )

    for consumer in consumers:
        orig_inner = consumer.data.inner_fn
        # _retile_load_index's squeezed-dim term injection (see
        # _squeezed_retile_dims) is only meaningful for a consumer with no
        # loop_info of its own -- an "outside" consumer whose incoming index
        # was traced against old_name's tile-local (squeezed) layout with no
        # enclosing coarse-tile loop nest to supply a term for a dim that
        # layout squeezed away. An "inside" consumer (has loop_info) already
        # derives its index from a real, enclosing loop nest that supplies a
        # correct term for every one of *its own* real dimensions; passing
        # it here anyway injects a bogus extra term for any dim this
        # consumer tiles as an output dim but that new_name's layout does
        # not vary over (e.g. a fully-reduced dim in an accum_full buffer),
        # double-counting/corrupting the address. See
        # test_copy_accum_with_reduction_512x256_A4_B4, where this caused a
        # spurious B-tile-index term in a redirected read of an
        # already-fully-B-reduced accum_full buffer.
        _index_consumer = consumer if not hasattr(consumer, "loop_info") else None

        def new_inner_fn(
            *args,
            _map=name_map,
            _info=retile_info if has_retile else None,
            _orig=orig_inner,
            _consumer=_index_consumer,
        ):
            if _info is not None:
                handler = _NameAndIndexSwapHandler(
                    V.ops, _map, {old_name: _info}, _consumer
                )
            else:
                handler = NameSwapHandler(V.ops, _map)
            with V.set_ops_handler(handler):
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

        # new_consumer.loop_info (copied verbatim from consumer by
        # copy_op_metadata inside replace_computed_buffer_body) still carries
        # tiled_dims_per_read as planned when this consumer's own read of
        # old_name was tile-local scratch -- fixed/non-advancing at plan
        # time (see _fixed_level_extents). But the read is now redirected to
        # new_name, a real full-sized buffer that must advance across every
        # loop_tiled_dims level that tiles its own dims, exactly like
        # _insert_combine_op's flat-case accum_buf write (see
        # _advancing_level_extents). Recompute just that read's entry so
        # SpyreKernel._general_tile_advance (which matches tiled_dims_per_read
        # to get_read_writes().reads purely positionally) does not silently
        # treat the redirected read as loop-invariant.
        #
        # A consumer that was never planned into any coarse-tile group (e.g.
        # a plain outside consumer of a copy_out buffer) never had
        # loop_info stamped at all -- there is no tiled_dims_per_read to fix
        # up, and this consumer isn't part of any loop group's
        # _general_tile_advance machinery in the first place.
        if not hasattr(new_consumer, "loop_info"):
            continue
        new_loop_info = new_consumer.loop_info  # type: ignore[attr-defined]
        new_reads = [
            r for r in new_consumer.get_read_writes().reads if isinstance(r, MemoryDep)
        ]
        if new_loop_info.tiled_dims_per_read:
            assert len(new_reads) == len(new_loop_info.tiled_dims_per_read), (
                "_patch_consumers: positional mismatch between "
                f"new_consumer.get_read_writes().reads ({len(new_reads)} entries) "
                "and new_loop_info.tiled_dims_per_read "
                f"({len(new_loop_info.tiled_dims_per_read)} entries) -- "
                "SpyreKernel._general_tile_advance matches these purely "
                "positionally, so a length mismatch means silently wrong "
                "tile-advance metadata rather than a loud failure."
            )
            advancing_level_extents = _advancing_level_extents(
                new_loop_info.loop_tiled_dims,
                new_loop_info.loop_count,
                list(new_consumer.data.ranges),
            )
            new_tiled_dims_per_read = [
                (
                    _tiled_dims_for_dep(read_dep, advancing_level_extents, new_consumer)
                    if read_dep.name == new_name
                    else per_level
                )
                for read_dep, per_level in zip(
                    new_reads, new_loop_info.tiled_dims_per_read
                )
            ]
            new_consumer.loop_info = dataclasses.replace(  # type: ignore[attr-defined]
                new_loop_info, tiled_dims_per_read=new_tiled_dims_per_read
            )
            logger.debug(
                "coarse_tile: patch_consumers %s old_name=%s new_name=%s "
                "loop_tiled_dims=%s loop_count=%s before=%s after=%s",
                new_consumer.get_name(),
                old_name,
                new_name,
                new_loop_info.loop_tiled_dims,
                new_loop_info.loop_count,
                new_loop_info.tiled_dims_per_read,
                new_tiled_dims_per_read,
            )


def _squeezed_retile_dims(
    info: _RetiledBufferInfo, consumer: ComputedBuffer
) -> list[int]:
    """Raw dims squeezed out of the old (tile-local) buffer but real in new.

    A dim with ``old_size[d] == 1`` has no ``d{i}`` symbol in an incoming
    load index at all -- Inductor's ``SqueezeView.squeezer`` drops unit-size
    dims unconditionally when the *reader's own* index for that dim was
    derived against this buffer's (by-then tile-local, size-1) layout, so
    ``compute_tile_index``/``_retile_load_index`` has no atom to rescale for
    that dim: rescaling can only touch coefficients already present in the
    incoming index (see ``_retile_load_index``'s docstring). Restricted to
    dims where ``new_stride[d] != 0`` -- a dim that stays size-1 (or
    genuinely strideless) in the new buffer too contributes nothing and
    needs no term.

    Also restricted to dims where the *consumer's own* output is non-unit
    for that raw dim (``consumer.data.ranges[d] != 1``). When the consumer's
    own output dim is unit-size too (e.g. a coarse-tiled dim of extent 1,
    such as B=1 when only H is tiled), there is no real loop variable for it
    anywhere in this trace -- the consumer's own write index squeezes it out
    exactly like the read did, so its only valid coordinate is the constant
    0, not a symbol. Minting one anyway causes a name collision with an
    unrelated, already-present symbol in the same dense numbering scheme
    (both derived independently, so nothing stops them picking the same
    name for two different logical slots), silently corrupting that other
    symbol's coefficient instead of erroring.
    """
    return [
        d
        for d in range(len(info.old_size))
        if info.old_size[d] == 1
        and info.new_stride[d] != sympy.S.Zero
        and int(consumer.data.ranges[d]) != 1
    ]


def _index_var_prefix(free_symbols: "OrderedSet[Expr] | set[Expr]") -> str:
    """Infer the live loop-variable naming prefix from an index's own symbols.

    ``index_vars_squeeze``/``index_vars_no_squeeze`` number loop variables
    densely as ``f"{prefix}{i}"`` -- but the prefix varies across the
    different retracing passes that reuse this same redirect machinery
    (``d0``, ``q0``, ``_i0``, ``i0``, ... have all been observed for the
    exact same logical dimension at different call sites/passes). Minting a
    hardcoded ``d{i}`` symbol is only correct when the live trace happens to
    use that exact prefix; otherwise the minted symbol is foreign to this
    trace's variable space and later stages that concretize "unknown"
    symbols (e.g. ``concretize_index`` treating it as a size symbol) will
    silently replace it with a constant, dropping the dimension again after
    _retile_load_index believed it had restored it. Detect the real prefix
    from any sibling symbol already present in the index instead of
    assuming one.
    """
    for sym in free_symbols:
        name = sym.name
        i = len(name)
        while i > 0 and name[i - 1].isdigit():
            i -= 1
        if i < len(name):
            return name[:i]
    return "d"


def _consumer_own_dim_symbol(
    consumer: ComputedBuffer, dim: int, prefix: str = "d"
) -> Expr:
    """The consumer's own loop-variable symbol for raw output dim ``dim``.

    Mirrors ``SpyreKernel._host_dim_to_index_symbol``'s squeeze arithmetic:
    Inductor's ``index_vars_squeeze`` numbers loop variables densely (as
    ``f"{prefix}{i}"``) over ``consumer.data.ranges``'s non-unit dims only,
    so ``{prefix}{dim}`` is the correct symbol only when no unit dim
    precedes it. ``consumer`` is the outside consumer being redirected
    (e.g. ``div``), not the buffer it reads -- its own output ranges are
    never squeezed by the redirect (only the *read* buffer's tile-local
    layout was), so this always yields a real, live loop symbol already
    used elsewhere in the consumer's index (e.g. by an unretiled sibling
    read), PROVIDED ``prefix`` matches the naming convention live in that
    index -- see ``_index_var_prefix``, which callers should use to derive
    it rather than assuming the default.
    """
    it_idx = 0
    mapped = dim
    for host_idx, r in enumerate(consumer.data.ranges):
        if int(r) != 1:
            if host_idx == dim:
                mapped = it_idx
            it_idx += 1
    return sympy_index_symbol(f"{prefix}{mapped}")


def _index_already_at_new_scale(
    index: Expr, loop_syms: set, info: "_RetiledBufferInfo"
) -> bool:
    """Return True when index's atom coefficients already match new_stride.

    A consumer in the same tiling group as a retiled buffer can be resynced
    (by name) to a *replacement* ComputedBuffer object spliced in by Pass
    1/2/3 after ``_apply_plan`` already ran -- see _coarse_tile_common's
    by-name resync comment above its ``_patch_retiled_load_indexes`` call.
    That replacement's inner_fn may have been retraced against the
    producer's already-mutated (new_stride) layout, in which case its load
    index is already correct and must not be decomposed against old_stride
    again -- doing so silently produces a wrong, merely plausible-looking
    index (two dims' coefficients can collide under old_stride's pairing
    even though the input was never stale -- see
    test_copy_running_max_4d_H4_Lq4).

    Detected by comparing the *set* of each atom's raw coefficient (before
    any tile-offset decomposition) against the set of old_stride vs.
    new_stride values for this buffer's real (non-irregular) dimensions.
    Each atom's coefficient is exactly one dimension's stride value in
    whatever scale the trace used -- a single loop variable that
    legitimately spans multiple dims (the "diagonal" case handled by
    compute_tile_offset's divmod chain, e.g. test_compute_tile_index_2d_diagonal)
    produces one atom whose coefficient is a *combined* value that matches
    neither set exactly, so this check only ever fires on the genuine
    already-new-scale case, never on a legitimate diagonal index. A
    coincidental match between the two sets can only happen when
    old_stride == new_stride for the dims involved (tiling never increases
    a dim's stride), which is a no-op either way.
    """
    try:
        atoms, _offset = decompose_index_for_tiling(
            index, {sym: 1 for sym in loop_syms}
        )
    except Unsupported:
        return False
    if not atoms:
        return False
    coeffs = {atom[0] for atom in atoms}
    dims = [
        d
        for d, (s, t) in enumerate(zip(info.old_size, info.old_stride))
        if s != 1 and t != 0
    ]
    old_set = {info.old_stride[d] for d in dims}
    new_set = {info.new_stride[d] for d in dims}
    return coeffs == new_set and coeffs != old_set


def _retile_load_index(
    buf_name: str,
    index: Expr,
    info: _RetiledBufferInfo,
    consumer: "ComputedBuffer | None" = None,
) -> Expr:
    """Rewrite a load index using compute_tile_index.  Raises Unsupported if
    the index cannot be decomposed (non-affine, or stride not in info.old_stride).

    Used by _RetileLoadIndexHandler and _NameAndIndexSwapHandler during real
    codegen.  In both cases the incoming index has coefficients equal to
    info.old_stride and the call rewrites them to info.new_stride.

    When ``consumer`` is given AND has no loop_info of its own (i.e. it is an
    "outside" consumer with no enclosing coarse-tile loop nest), any dim
    squeezed out of the old (tile-local) buffer but real in the new (full)
    buffer -- see _squeezed_retile_dims -- is added back as an explicit
    ``consumer``-own-symbol * new_stride term, since compute_tile_index
    cannot rescale a coefficient that isn't in the incoming index at all.
    Inside consumers (those with their own loop_info) are never patched this
    way: their incoming index is already derived from a real, enclosing loop
    nest via normal codegen and already carries a correct term for every
    real dimension -- adding another would double-count it. Only meaningful
    for the tile->full ("grow") direction (_NameAndIndexSwapHandler); the
    full->tile direction (_RetileLoadIndexHandler) never needs this --
    shrinking a buffer's own layout for a group-internal consumer doesn't
    lose dims from its index.

    The two handlers run in opposite directions:
    - _RetileLoadIndexHandler: full→tile (old_stride are the full strides,
      new_stride are the smaller tile strides).
    - _NameAndIndexSwapHandler: tile→full (old_stride are the tile strides,
      new_stride are the larger full strides).

    compute_tile_index is valid for both directions.  Its core,
    compute_tile_offset, does divmod(coeff, s) for each paired stride s in
    decreasing order.  Each atom's coefficient IS one of the old strides, so
    each divmod is exact (remainder zero).  Potential ambiguity from duplicate
    stride values is prevented by compute_tile_index's irregular-dimension
    filter: size==1 and stride==0 dimensions are excluded before pairing, and
    those are the only source of duplicate strides in a valid tensor layout.

    compute_tile_index uses var_ranges only as a key-set to distinguish loop
    variables from shape symbols.  We derive it from index.free_symbols so the
    call site never needs to know which prefix convention the calling inner_fn
    uses for its loop variables.
    """

    loop_syms = index.free_symbols
    if not loop_syms:
        new_index = index
    elif _index_already_at_new_scale(index, loop_syms, info):
        # This consumer's index was traced *after* the producer's layout was
        # already mutated to new_stride (e.g. built/retraced during Pass
        # 1/2/3, from a same-name replacement object resynced into group_ops
        # -- see _coarse_tile_common's by-name resync comment), so it is
        # already correct at the target scale. Rewriting it again would
        # decompose already-new-scale coefficients against old_stride and
        # silently produce a wrong (but plausible-looking) index -- see
        # issue found via test_copy_running_max_4d_H4_Lq4.
        new_index = index
    else:
        new_index = compute_tile_index(
            index,
            {sym: 1 for sym in loop_syms},
            info.old_size,
            info.old_stride,
            info.new_stride,
        )

    if consumer is not None:
        for d in _squeezed_retile_dims(info, consumer):
            # A consumer read by multiple _patch_consumers redirects (e.g.
            # buf24 reading both a _divide_ranges-mutated buffer and a
            # reduction accumulator) has its inner_fn wrapped more than
            # once. By the time a later wrap sees this index, an earlier
            # wrap may already have contributed a term for this exact dim
            # -- under whatever loop-symbol naming convention was live at
            # that earlier wrap's call site (which varies across retracing
            # passes: d0, i0, q0, ... are all seen for the same logical
            # dim). Detect that by coefficient, not by a hardcoded symbol
            # name: if some free symbol already carries coefficient
            # new_stride[d], this dim's term is already present and must
            # not be added again. Comparing by coefficient value (not
            # dimension identity) is safe only because strides are unique
            # across all non-irregular dims of a valid layout -- see this
            # function's docstring ("Potential ambiguity from duplicate
            # stride values"). If that uniqueness invariant is ever
            # weakened, this check must be revisited too.
            already_present = any(
                new_index.coeff(s) == info.new_stride[d] for s in new_index.free_symbols
            )
            if not already_present:
                prefix = _index_var_prefix(new_index.free_symbols)
                sym = _consumer_own_dim_symbol(consumer, d, prefix)
                new_index += sym * info.new_stride[d]

    logger.debug(
        "coarse_tile: retiled load index for %s: %s -> %s",
        buf_name,
        index,
        new_index,
    )
    return new_index


class _RetileLoadIndexHandler(WrapperHandler):
    """Ops handler that retiles loads from buffers whose host strides changed.

    Used during real codegen — calls _retile_load_index, which raises
    Unsupported if an index cannot be retiled.
    """

    def __init__(self, inner, infos_by_name: dict[str, _RetiledBufferInfo]):
        super().__init__(inner)
        self._infos_by_name = infos_by_name

    def load(self, name, index):
        if name in self._infos_by_name:
            index = _retile_load_index(name, index, self._infos_by_name[name])
        return super().load(name, index)


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
                name, index, self._infos_by_old_name[name], self._consumer
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
    infos_by_name = {
        name: info
        for name, info in retiled_infos.items()
        if info.old_stride != info.new_stride
    }
    if not infos_by_name:
        return

    from ..pass_utils import replace_computed_buffer_body

    # Only ops that were already in the group when _apply_plan ran can hold a
    # stale (pre-divide) coefficient for a retiled buffer.  Ops inserted later
    # by Pass 1/2/3 (e.g. _insert_copy_op's copy_buf) read the retiled
    # buffer's already-updated layout directly, so rewriting them here would
    # double-apply the stride correction (see issue found while fixing
    # test_hint_restickify_stays_in_group).
    retiled_names = set(infos_by_name)
    for op in list(group_ops):
        if not _should_patch_retiled_load_indexes(op, group_id, retiled_names):
            continue

        orig_inner = op.data.inner_fn

        def new_inner_fn(
            *args,
            _infos=infos_by_name,
            _orig=orig_inner,
        ):
            with V.set_ops_handler(_RetileLoadIndexHandler(V.ops, _infos)):
                return _orig(*args)

        object.__setattr__(op.data, "inner_fn", new_inner_fn)
        new_op = replace_computed_buffer_body(
            op,
            op.data,
            operations,
            pass_name="coarse_tile",
            reason="rewrite retiled load indexes",
        )
        _replace_group_op(group_ops, op, new_op)
        V.graph.name_to_buffer[new_op.get_name()] = new_op


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
