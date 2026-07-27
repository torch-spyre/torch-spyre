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
"""

from __future__ import annotations


import logging
from collections import Counter
from typing import NamedTuple

import sympy
from sympy import Expr

import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
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

from .constants import BATCH_MATMUL_OP
from .errors import Unsupported
from .logging_utils import get_inductor_logger
from .loop_info import CoarseTileInfo
from .propagate_hints import DimHint
from .pass_utils import op_out_coords, host_coordinates, indirect_sizes_from_op
from .span_overflow_hint_analysis import (
    SpanOverflowTilePlan,
    can_conform_pointwise_tile,
    plan_span_overflow_tile,
)
from .ir import FixedTiledLayout, _resize_device_layout

logger = get_inductor_logger("coarse_tile")
hints_logger = get_inductor_logger("assign_dim_hints")


_SPAN_OVERFLOW_HINT_ID = 10000


class _RetiledBufferInfo(NamedTuple):
    """Host strides before and after a buffer is resized for a coarse tile."""

    old_stride: tuple[Expr, ...]
    new_stride: tuple[Expr, ...]


def _auto_span_plan_signature(
    plan: SpanOverflowTilePlan,
) -> tuple[tuple[int, int, bool], ...]:
    """Return the grouping key for a span-overflow plan."""
    return tuple(
        (level.selected_host_dim, level.split_count, level.is_reduction)
        for level in plan.levels
    )


def _auto_span_read_deps(op: ComputedBuffer) -> set[str]:
    """Return direct MemoryDep read names for auto span-overflow grouping."""
    try:
        return {
            dep.name for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)
        }
    except (AttributeError, TypeError):
        return set()


def _reduction_shares_group_tiled_dim(
    op: ComputedBuffer,
    signature: tuple[tuple[int, int, bool], ...],
    current_group: list,
) -> bool:
    """True if a reduction's tiled output dim(s) are the *same logical dim(s)*
    as the tiled dim(s) of the producer(s) it reads in the open group.

    Joining ops into one group means they share a single loop nest: iteration
    ``t`` computes tile ``t`` of every member.  For that to be correct, the
    consumer's tiled dimension must be the dimension that — through its read of
    the producer — indexes the producer's tiled dimension.  Matching split
    counts is necessary but not sufficient (two unrelated dims could split into
    the same count), so verify the loop-variable correspondence explicitly: the
    symbol tiling the consumer's output dim must appear in the producer's tiled
    coordinate as seen through the read.

    Conservative: any failure to establish the correspondence returns False, so
    an unverifiable pair is left to the normal (Unsupported) conflict path
    rather than fused into a possibly-desynchronized loop.

    This check itself is reduction-type-agnostic — what makes the join safe is
    that the tiled dim is an **output range**, not the reduction range: tile
    ``t`` of an output dim is self-contained (it reads only tile ``t`` of the
    producer).  The caller (the reduction-join branch in
    ``span_overflow_groups``) applies this to any Reduction op, not just
    batch-matmul.

    The automatic span-overflow planner only ever tiles output ranges (see
    ``SpanOverflowTileLevel``: ``is_reduction`` is always False on the auto
    path, because reduction-range tiling would require partial-result
    accumulation), so every signature reaching here should already be
    output-only.  We assert that invariant explicitly below and fail closed if
    a future planner change ever emits a reduction-range tile — such a tile
    would break the loop-carried accumulation this join assumes away.
    """
    # Guard: only output-range tiles may join.  A reduction (K) range tile
    # would need cross-tile accumulation and cannot share a per-tile loop nest.
    if any(is_reduction for _host_dim, _split, is_reduction in signature):
        return False
    try:
        consumer_coords = op_out_coords(op)
        reads = {
            dep.name: dep
            for dep in op.get_read_writes().reads
            if isinstance(dep, MemoryDep)
        }
        indirect = indirect_sizes_from_op(op)
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError, IndexError):
        # op_out_coords internally calls host_coordinates, which can raise the
        # same ValueError/RuntimeError/IndexError as the direct call below, so
        # this list must cover that set too (plus AttributeError for the
        # get_read_writes()/indirect_sizes_from_op attribute access).
        return False

    consumer_tiled_syms: set = set()
    for host_dim, _split, _is_reduction in signature:
        if host_dim >= len(consumer_coords):
            return False
        consumer_tiled_syms |= consumer_coords[host_dim].free_symbols
    if not consumer_tiled_syms:
        return False

    group_by_name = {gop.get_name(): (gop, dims) for gop, dims in current_group}
    verified_any = False
    for name, dep in reads.items():
        if name not in group_by_name:
            continue
        producer, producer_dims = group_by_name[name]
        try:
            producer_coords = host_coordinates(producer.get_layout(), dep, indirect)
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
            return False
        for host_dim_p, _split, _is_reduction in producer_dims:
            if host_dim_p >= len(producer_coords):
                return False
            if not (producer_coords[host_dim_p].free_symbols & consumer_tiled_syms):
                # Consumer's tiled loop var does not index this producer's tiled
                # dim -> not the same logical dim -> unsafe to share a loop.
                return False
            verified_any = True
    return verified_any


def _dims_to_hints(
    op: ComputedBuffer,
    dims: tuple[tuple[int, int, bool], ...],
    hint_ids: list[int],
) -> list[DimHint]:
    """Create per-op DimHints from (host_dim, split_count, is_reduction) triples.

    ``dims`` is either ``op``'s own independently-searched plan signature, or
    — when ``op`` conforms to an already-open Pointwise chain — the chain's
    shared signature.  Either way, ``op`` resolves its own ``loop_var`` from
    its own output coordinates here, so a conforming op still gets a loop_var
    that is correct for its own indexing, not copied from the op it conforms
    to.
    """
    out_coords = op_out_coords(op)
    hints: list[DimHint] = []
    for (host_dim, split_count, is_reduction), hint_id in zip(dims, hint_ids):
        if host_dim >= len(out_coords):
            raise Unsupported(
                f"Cannot adapt span-overflow plan for {op.get_name()}: "
                f"host_dim={host_dim} is out of bounds for "
                f"{len(out_coords)} output coordinates."
            )

        coord = out_coords[host_dim]
        free_symbols = coord.free_symbols
        if len(free_symbols) != 1:
            raise Unsupported(
                f"Cannot adapt span-overflow plan for {op.get_name()}: "
                f"host_dim={host_dim} output coordinate {coord} has "
                f"{len(free_symbols)} free symbols; expected exactly one loop var."
            )

        loop_var = next(iter(free_symbols))
        logger.debug(
            "[span-overflow groups] op=%s host_dim=%d coord=%s "
            "loop_var=%s split_count=%s hint_id=%d is_reduction=%s",
            op.get_name(),
            host_dim,
            coord,
            loop_var,
            split_count,
            hint_id,
            is_reduction,
        )
        hints.append(
            DimHint(
                dim_names=["_span_overflow"],
                split_count=split_count,
                loop_var=loop_var,
                is_reduction=is_reduction,
                hint_id=hint_id,
            )
        )
    return hints


# ---------------------------------------------------------------------------
# Hint-driven group construction
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


def _hints_levels(ops: list[Operation]) -> list[tuple]:
    """Build (hint_id, K) level pairs by unioning across all ops.

    All ops in the group share the same hint IDs and split counts.  For each
    hint_id, pick the best DimHint across all ops: one with loop_var is not None
    beats one with loop_var=None.  Hints that are broadcast at every op
    (loop_var=None everywhere) are dropped.  Hints with split_count==1 are
    dropped (tiling by 1 is a no-op).  Returns pairs sorted by hint_id
    ascending (outermost-first).

    is_reduction is intentionally absent from the returned pairs: it is a
    per-op, per-dimension property consulted directly in _stamp_group via each
    op's own DimHint, not a group-level concept.
    """
    best: dict[int, DimHint] = {}
    for op in ops:
        for h in getattr(op, "dim_hints", []):
            prev = best.get(h.hint_id)
            if (
                prev is None
                or prev.loop_var is None
                or (prev.split_count == 1 and h.split_count > 1)
            ):
                best[h.hint_id] = h

    levels = []
    for h in sorted(best.values(), key=lambda x: x.hint_id):
        if h.loop_var is None:
            continue
        if h.split_count == 1:
            hints_logger.debug(
                "spyre_hint on [%s]: hint_id=%d dims=%s split_count=1"
                " — tiling by 1 is a no-op, dropping",
                ", ".join(o.get_name() for o in ops),
                h.hint_id,
                h.dim_names,
            )
            continue
        levels.append((h.hint_id, sympy.Integer(h.split_count)))
    return levels


def _hint_key(op: Operation) -> frozenset | None:
    """Return the frozenset of hint_ids on op, or None if op has no hints."""
    if not isinstance(op, ComputedBuffer):
        return None
    hints = getattr(op, "dim_hints", [])
    return frozenset(h.hint_id for h in hints) if hints else None


def _written_names(op: ComputedBuffer) -> set[str]:
    """Return all buffer names written by op: its output plus any mutation targets."""
    return {op.get_name()} | set(op.get_mutation_names())


def _no_dep_conflict(op: ComputedBuffer, others: list[Operation]) -> bool:
    """Return True if moving op past every op in others introduces no data-flow hazard.

    A conflict exists if any op in others reads or mutates a buffer written by op,
    or if op reads or mutates a buffer written by any op in others.

    op_needs intentionally includes op.get_mutation_names() alongside read names.
    This covers both RAW (op reads a buffer that other writes) and WAW (op mutates
    a buffer that other also writes) hazards.  The WAW case is conservative: two
    ops mutating the same buffer cannot be reordered safely regardless of direction,
    so conflating them here is deliberate.
    """
    op_written = _written_names(op)
    op_needs = op.get_read_names() | set(op.get_mutation_names())
    for other in others:
        if not isinstance(other, ComputedBuffer):
            continue
        if op_written & other.get_read_names():
            return False
        if _written_names(other) & op_needs:
            return False
    return True


def _can_move_before(
    op: Operation,
    ops: list[Operation],
    start: int,
    end: int,
) -> bool:
    """Return True if op (at ops[end]) can move to just before ops[start].

    Legal iff no data-flow conflict exists between op and ops[start..end-1].
    """
    # Defensive: _no_dep_conflict requires a ComputedBuffer; the sole caller
    # (reorder_unhinted_interlopers) already filters for this, but guard here
    # in case the function is called from a future context.
    if not isinstance(op, ComputedBuffer):
        return False
    return _no_dep_conflict(op, ops[start:end])


def _can_move_after(
    op: Operation,
    ops: list[Operation],
    start: int,
    end: int,
) -> bool:
    """Return True if op (at ops[start]) can move to just after ops[end-1].

    Legal iff no data-flow conflict exists between op and ops[start+1..end-1].
    """
    # Defensive: same rationale as _can_move_before.
    if not isinstance(op, ComputedBuffer):
        return False
    return _no_dep_conflict(op, ops[start + 1 : end])


def reorder_unhinted_interlopers(graph: GraphLowering) -> None:
    """Move unhinted ComputedBuffer ops that interrupt hint-group runs.

    ``hints_to_coarse_tile_groups`` treats unhinted ops as run-breakers.
    This pass attempts to move each such op either just before the run it
    splits or just after the last same-key op in the remainder, so the run
    becomes contiguous.

    Algorithm — two-cursor scan over ops:

    Outer cursor i: start of the next candidate run.  Advances to j when
    the inner loop exits.

    Inner cursor j: walks forward from i+1 building the run.  For each
    op at ops[j]:
      - Same hint key → absorb into run; j += 1.
      - Non-ComputedBuffer or differently-hinted → hard stop; break.
      - Unhinted ComputedBuffer (interloper) → one of three outcomes:
          (a) Move before: insert at run_start, run_start += 1, j stays
              (the rotate shifts subsequent ops left so ops[j] is fresh).
          (b) Move after: pop(j), insert at run_end-1, j stays.
          (c) Neither legal → RuntimeError.
        run_end is the index one past the *last* same-key op in ops[j+1:],
        found by scanning backward.  Using the last op (not just the next)
        ensures the move-after target span covers the full remaining run,
        which matters when interlopers further right would otherwise still
        split the run.

    When the inner loop exits, j points to the first op that could not be
    absorbed — a hard-stop or end-of-list.  Advancing i to j (not i+1)
    is correct because everything before j has already been processed.

    A move is legal when it introduces no new data-flow violation:
    no op in the skipped range reads or mutates the moved op's written
    buffers, and the moved op reads or mutates no buffer written in the
    skipped range.

    When both directions are legal the op is moved before the run (closer
    to its original position).

    Raises RuntimeError if an interloper cannot be moved in either
    direction (data-flow dependencies anchor it between hinted ops that
    share the same hint key).
    """
    ops = graph.operations
    i = 0
    while i < len(ops):
        op = ops[i]
        key = _hint_key(op)
        if key is None:
            i += 1
            continue

        run_start = i
        j = i + 1
        while j < len(ops):
            candidate = ops[j]
            ckey = _hint_key(candidate)
            if ckey == key:
                j += 1
                continue
            if not isinstance(candidate, ComputedBuffer) or ckey is not None:
                break
            # candidate is an unhinted ComputedBuffer interloper.
            # Scan backward for the last same-key op; run_end is one past it.
            # O(n) per interloper → O(n²) overall; acceptable for small graphs.
            run_end = None
            for k in range(len(ops) - 1, j, -1):
                if _hint_key(ops[k]) == key:
                    run_end = k + 1
                    break
            # No same-key op exists after j: trailing consumer, not an
            # interloper — end the run silently.
            if run_end is None:
                break
            if _can_move_before(candidate, ops, run_start, j):
                ops.insert(run_start, ops.pop(j))
                run_start += 1  # skip past the op we just inserted before the run
                continue
            if _can_move_after(candidate, ops, j, run_end):
                # pop(j) shifts everything after j left by one, so the last
                # same-key op (formerly run_end-1) is now at run_end-2.
                # Insert at run_end-1 to land just after that last hinted op.
                ops.insert(run_end - 1, ops.pop(j))
                continue
            run_ops = [ops[k].get_name() for k in range(run_start, j)]
            raise RuntimeError(
                f"Cannot reorder unhinted op '{candidate.get_name()}': "
                f"data-flow deps prevent moving it before or after the "
                f"hint-group run [{', '.join(run_ops)}] "
                f"(hint_ids={sorted(key)})"
            )
        i = j


def hints_to_coarse_tile_groups(graph: GraphLowering) -> list[tuple]:
    """Build coarse_tile() groups from op.dim_hints (set by assign_dim_hints).

    coarse_tile() requires ops to be grouped: all ops in a group share the same
    tiling spec and are tiled together inside the same loop nest.  We walk
    operations in topological order and collect consecutive ops that carry
    identical hints into one group, breaking whenever the hint changes or an
    op has no hint at all.
    """

    def _flush(groups, current_ops, current_key):
        if current_ops and current_key is not None:
            levels = _hints_levels(current_ops)
            if levels:
                groups.append((current_ops, levels))
            else:
                hints_logger.warning(
                    "spyre_hint on [%s]: no op iterates over the hinted dimension "
                    "— hint ignored",
                    ", ".join(o.get_name() for o in current_ops),
                )

    groups: list[tuple] = []
    current_ops: list[Operation] = []
    current_key = None

    operations = graph.operations
    for op in operations:
        key = _hint_key(op)

        if key is not None and key == current_key:
            current_ops.append(op)
        else:
            _flush(groups, current_ops, current_key)
            current_ops = [op] if key is not None else []
            current_key = key

    _flush(groups, current_ops, current_key)

    if hints_logger.isEnabledFor(logging.INFO):
        # Build an interleaved view: walk operations in order, emit group boundaries
        # and ungrouped ops so the reader can see what breaks each consecutive run.
        grouped_to_group_idx = {id(o): i for i, g in enumerate(groups) for o in g[0]}
        # Pre-compute hint descriptions per group — get_op_hints is called once per
        # group rather than once per op in the group.
        group_hint_descs: dict[int, str] = {}
        for g_idx, (group_ops, _group_levels) in enumerate(groups):
            # Collect all DimHints across the group, keyed by hint_id.
            # Prefer a hint whose loop_var is not None (op actually iterates
            # that dim) over a broadcast hint (loop_var=None), so that the
            # representative name/count reflects a real iteration.
            best: dict[int, "DimHint"] = {}
            for gop in group_ops:
                for h in getattr(gop, "dim_hints", []):
                    if h.hint_id not in best or best[h.hint_id].loop_var is None:
                        best[h.hint_id] = h
            descs = [
                f"hint_{h.hint_id}={{'tiles': {{"
                + ", ".join(f"'{n}': {h.split_count}" for n in h.dim_names)
                + "}}"
                for h in sorted(best.values(), key=lambda x: x.hint_id)
            ]
            group_hint_descs[g_idx] = ", ".join(descs)

        summary_lines = [f"coarse_tile_groups: {len(groups)} group(s) formed"]
        pending_ungrouped: list[str] = []
        last_group_idx: int | None = None
        for o in operations:
            if not isinstance(o, ComputedBuffer):
                continue
            op_group_idx = grouped_to_group_idx.get(id(o))
            if op_group_idx is None:
                hints = getattr(o, "dim_hints", [])
                if hints:
                    ids = sorted({h.hint_id for h in hints})
                    reason = f"hint_ids={ids}"
                else:
                    reason = "no hints"
                pending_ungrouped.append(f"{o.get_name()}({reason})")
            else:
                if op_group_idx != last_group_idx:
                    if pending_ungrouped:
                        summary_lines.append(
                            f"  ungrouped: [{', '.join(pending_ungrouped)}]"
                        )
                        pending_ungrouped = []
                    summary_lines.append(
                        f"  group {op_group_idx} scopes=[{group_hint_descs[op_group_idx]}]:"
                    )
                    last_group_idx = op_group_idx
                # Per-op tiling info.
                tiling_dims = [
                    f"{h.dim_names[0] if h.dim_names else '?'}x{h.split_count}"
                    for h in getattr(o, "dim_hints", [])
                    if h.loop_var is not None and not h.is_reduction
                ]
                aten_ops = [
                    str(n.target)
                    for n in getattr(o, "origins", [])
                    if hasattr(n, "target")
                ]
                summary_lines.append(
                    f"      {o.get_name()}  aten={aten_ops}"
                    + (f"  tiles={tiling_dims}" if tiling_dims else "  (no tiled dims)")
                )
        if pending_ungrouped:
            summary_lines.append(f"  ungrouped: [{', '.join(pending_ungrouped)}]")
        hints_logger.info("%s", "\n".join(summary_lines))

    return groups


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


def span_overflow_groups(graph: GraphLowering) -> list[tuple]:
    """Build coarse_tile() groups from automatic span-overflow plans.

    This adapter converts SpanOverflowTilePlans into the same group shape as
    user spyre_hint annotations: ``[(ops, [(hint_id, count)])]``.  Ops that
    already carry user dim hints are left for the user-hint grouping path.
    ``is_reduction`` is not carried in the group-level ``levels`` list; it
    lives on each op's own ``DimHint`` and is consulted directly by
    ``_stamp_group``.

    A contiguous run of Pointwise ops shares one group/loop when either:
      - each op's own independently-searched plan
        (``plan_span_overflow_tile``) produces the exact same
        ``(host_dim, split_count, is_reduction)`` signature as the run so
        far; or
      - an op's own plan disagrees, but the run reads into it (a real
        producer-consumer edge) and the run's existing split is *also* a
        legal, sufficient plan for that op on its own
        (``can_conform_pointwise_tile``) — the op then adopts the run's split
        instead of its own.

    A Reduction op does not *start* or extend a Pointwise run.  Any Reduction
    (matmul/BMM, sum, mean, ...) may **join** an open run's group when it
    reads a producer in that run and tiles the same shared logical (output)
    dim at the same split count — e.g. an F.linear matmul reading its
    auto-tiled restickified weight, or a plain ``sum`` reading a tiled
    pointwise producer (see the reduction-join branch below and
    ``_reduction_shares_group_tiled_dim``).  The join is reduction-type-
    agnostic: what makes it safe is that the tiled dim is an output range, not
    the reduction range (tile ``t`` is self-contained either way).  On
    joining, the group is flushed immediately, so a reduction is always the
    last member of its group and each auto-tiled producer feeds at most one
    reduction consumer.  A Reduction that cannot join gets an independent
    singleton group or, if it reads an auto-tiled producer, raises
    ``Unsupported``.  An op that
    reads a buffer from an already-closed group, or from the open run without
    being fusable into it, still raises ``Unsupported``: two independent loop
    nests over the same span-overflow-sized data can desynchronize, and for ops
    tiled specifically because their *full* buffer violates the hardware span
    limit, falling back to materializing that full buffer for an "outside
    consumer" would silently reintroduce the exact span violation tiling was
    meant to prevent.
    """
    from . import config

    if config.ignore_wsr_hints or config.ignore_span_overflow_hints:
        logger.debug(
            "[span-overflow groups] disabled ignore_wsr_hints=%s ignore_span_overflow_hints=%s",
            config.ignore_wsr_hints,
            config.ignore_span_overflow_hints,
        )
        return []

    logger.debug(
        "[span-overflow groups] begin ops=%d sencores=%s",
        len(graph.operations),
        config.sencores,
    )
    groups: list[tuple] = []
    next_hint_id = _SPAN_OVERFLOW_HINT_ID
    auto_tiled_producers: set[str] = set()
    # Producers whose group was closed by a Reduction consumer joining it (see
    # the reduction-join branch below).  These are a subset of
    # ``auto_tiled_producers``; tracked separately only so a *second* consumer
    # reading such a producer gets a precise "multi-consumer not yet supported"
    # error rather than the generic pointwise-only conflict message.
    reduction_joined_producers: set[str] = set()
    # Producers already tiled by a user spyre_hint (assign_dim_hints runs
    # before this pass and leaves dim_hints set; hints_to_coarse_tile_groups
    # only reads it, it never clears it). An op reading one of these has the
    # same unsynchronized-loop-nest risk as reading an auto_tiled_producers
    # entry, so both sets guard the same conflict checks below.
    manually_hinted_producers: set[str] = {
        op.get_name()
        for op in graph.operations
        if isinstance(op, ComputedBuffer) and getattr(op, "dim_hints", [])
    }
    _PwDims = tuple[tuple[int, int, bool], ...]
    current_group: list[tuple[ComputedBuffer, _PwDims]] = []
    current_signature: _PwDims | None = None

    def flush_current_group() -> None:
        nonlocal next_hint_id, current_group, current_signature
        if not current_group:
            return

        signature = current_signature
        assert signature is not None
        hint_ids = list(range(next_hint_id, next_hint_id + len(signature)))
        next_hint_id += len(signature)
        levels = [
            (hint_id, sympy.Integer(split_count))
            for hint_id, (_host_dim, split_count, _is_reduction) in zip(
                hint_ids, signature
            )
        ]

        group_ops: list[Operation] = []
        for grouped_op, dims in current_group:
            grouped_op.dim_hints = _dims_to_hints(  # type: ignore[attr-defined]
                grouped_op, dims, hint_ids
            )
            group_ops.append(grouped_op)
            auto_tiled_producers.add(grouped_op.get_name())

        groups.append((group_ops, levels))
        logger.debug(
            "[span-overflow groups] created group_index=%d ops=%s levels=%s",
            len(groups) - 1,
            [op.get_name() for op in group_ops],
            levels,
        )
        current_group = []
        current_signature = None

    for op in graph.operations:
        if not isinstance(op, ComputedBuffer):
            flush_current_group()
            continue
        if not isinstance(op.data, (Pointwise, Reduction)):
            flush_current_group()
            continue
        if isinstance(op.data, Reduction) and not list(op.data.ranges):
            flush_current_group()
            continue
        if not isinstance(op.layout, FixedTiledLayout):
            flush_current_group()
            continue
        if getattr(op, "dim_hints", []):
            flush_current_group()
            continue

        read_deps = _auto_span_read_deps(op)
        current_group_names = {grouped_op.get_name() for grouped_op, _ in current_group}

        plan = plan_span_overflow_tile(op, config.sencores)
        if plan is None:
            # op needs no coarse tiling of its own.  It's always safe to leave
            # it outside any loop: insert_tiling_propagation's outside-consumer
            # path (coarse_tile.py) already patches consumers of a tiled
            # producer to read a full, reassembled buffer, and
            # plan_span_overflow_tile returning None here means that op's own
            # full-size reads/writes are already known not to overflow.
            logger.debug("[span-overflow groups] op=%s no auto plan", op.get_name())
            flush_current_group()
            continue

        signature = _auto_span_plan_signature(plan)
        logger.debug(
            "[span-overflow groups] op=%s plan_levels=%s reasons=%s",
            op.get_name(),
            list(signature),
            [info.reason for info in plan.chunking_infos],
        )
        logger.debug(
            "[span-overflow groups] op=%s read_deps=%s auto_tiled_producers=%s "
            "current_group=%s",
            op.get_name(),
            sorted(read_deps),
            sorted(auto_tiled_producers),
            sorted(current_group_names),
        )

        completed_conflicts = sorted(
            read_deps & (auto_tiled_producers | manually_hinted_producers)
        )
        if completed_conflicts:
            logger.warning(
                "[span-overflow groups] op=%s rejected_conflicting_auto_producers=%s",
                op.get_name(),
                completed_conflicts,
            )
            joined_conflicts = sorted(
                set(completed_conflicts) & reduction_joined_producers
            )
            if joined_conflicts:
                # The producer was already auto-tiled *and* joined into a
                # synchronized loop by an earlier reduction consumer.  A single
                # auto-tiled producer can currently feed only one reduction
                # consumer; a second consumer would need its own tile loop over
                # the same producer, which is not yet supported.
                raise Unsupported(
                    f"Cannot auto-tile {op.get_name()}: it reads producer(s) "
                    f"{joined_conflicts} that were already auto-tiled and joined "
                    "by another reduction consumer. A single auto-tiled producer "
                    "can currently feed only one reduction consumer in one "
                    "synchronized group; multiple consumers sharing one "
                    "auto-tiled producer is not yet supported (#3217)."
                )
            raise Unsupported(
                f"Cannot auto-tile {op.get_name()}: it reads already auto-tiled "
                f"producer(s) {completed_conflicts}. Automatic span-overflow "
                "grouping currently only synchronizes compatible contiguous "
                "pointwise ops, so tiling this producer and consumer independently "
                "can produce unsynchronized loop nests."
            )

        is_reduction_op = isinstance(op.data, Reduction)

        can_join_pw_group = (
            not is_reduction_op
            and current_signature is not None
            and signature == current_signature
        )
        if can_join_pw_group:
            current_group.append((op, signature))
            logger.info(
                "[span-overflow groups] op=%s joined_matching_signature=%s",
                op.get_name(),
                list(signature),
            )
            continue

        # op's own independent plan disagrees with the open run.  If op
        # actually reads from the run (a real producer-consumer edge, not
        # just an adjacent unrelated op), check whether the run's split is
        # *also* legal and sufficient for op on its own — if so, op adopts
        # the run's split rather than opening a second, unsynchronized loop.
        conform_dims: tuple[tuple[int, int, bool], ...] | None = None
        if (
            not is_reduction_op
            and current_signature is not None
            and (read_deps & current_group_names)
        ):
            split_by_host_dim = {
                host_dim: split_count for host_dim, split_count, _ in current_signature
            }
            if can_conform_pointwise_tile(op, split_by_host_dim, config.sencores):
                conform_dims = current_signature

        if conform_dims is not None:
            current_group.append((op, conform_dims))
            logger.info(
                "[span-overflow groups] op=%s conformed_to_group_split=%s "
                "(own_independent_plan_was=%s)",
                op.get_name(),
                list(conform_dims),
                list(signature),
            )
            continue

        # Any Reduction consumer (e.g. an F.linear matmul reading its
        # restickified weight, or a plain sum reading a tiled pointwise
        # producer) can join its tiled producer's open group when it tiles the
        # same shared logical dimension at the same split count(s). The shared
        # dim sits at a different position in the consumer's output ranges
        # (the producer tiles its V output dim; the consumer tiles the
        # corresponding output N dim), so signatures match on split_count, not
        # host_dim.  Both are output-dim tiles, so they share one synchronized
        # loop nest and the producer's per-tile slice feeds the consumer's
        # per-tile compute — no unsynchronized second loop, no full-buffer
        # materialization.
        #
        # Scope: the join is reduction-type-agnostic — correct-by-construction
        # for any reduction tiled on a shared output range, since tile t is
        # self-contained (sum/mean/max pair slice-for-slice, same as matmul).
        # Unit coverage: test_non_matmul_reduction_joins_tiled_producer_group.
        # On-device numeric validation:
        # TestSpanOverflowNumericValidation.
        # test_pointwise_to_non_matmul_reduction_join_numeric.
        #
        # Split-count equality alone is insufficient: two unrelated dims could
        # split into the same count.  _reduction_shares_group_tiled_dim verifies
        # the consumer's tiled loop var actually indexes the producer's tiled dim
        # through the read, so the shared loop pairs matching slices.  It also
        # fails closed if the consumer tiles its reduction (K) range rather than
        # an output range (see its docstring) — only output-range tiles may join.
        #
        # The group is flushed immediately after the reduction joins: a
        # reduction terminates the extendable run (its output shape/tiling
        # differs from the producers'), so nothing further can be folded into
        # this loop nest. A consequence is one-consumer-per-group — a *second*
        # op reading the same producer is rejected below.  Supporting several
        # sibling reductions sharing one auto-tiled producer is a deliberate
        # non-goal here (matches the validated single-consumer LM-head case);
        # see #3217.
        if (
            is_reduction_op
            and current_signature is not None
            and (read_deps & current_group_names)
            and [s for _, s, _ in signature] == [s for _, s, _ in current_signature]
            and _reduction_shares_group_tiled_dim(op, signature, current_group)
        ):
            current_group.append((op, signature))
            reduction_joined_producers |= read_deps & current_group_names
            logger.info(
                "[span-overflow groups] op=%s joined_producer_group_as_reduction "
                "split=%s",
                op.get_name(),
                list(signature),
            )
            flush_current_group()
            continue

        pending_conflicts = sorted(read_deps & current_group_names)
        flush_current_group()
        if pending_conflicts:
            logger.warning(
                "[span-overflow groups] op=%s rejected_conflicting_auto_producers=%s",
                op.get_name(),
                pending_conflicts,
            )
            raise Unsupported(
                f"Cannot auto-tile {op.get_name()}: it reads already auto-tiled "
                f"producer(s) {pending_conflicts}. Automatic span-overflow "
                "grouping currently only synchronizes compatible contiguous "
                "pointwise ops, so tiling this producer and consumer independently "
                "can produce unsynchronized loop nests."
            )

        if not is_reduction_op:
            current_group.append((op, signature))
            current_signature = signature
            logger.info(
                "[span-overflow groups] op=%s started_new_pw_group split=%s",
                op.get_name(),
                list(signature),
            )
            continue

        # A Reduction/BMM op that did not join an open producer group (above)
        # gets an independent singleton group.
        hint_ids = list(range(next_hint_id, next_hint_id + len(signature)))
        next_hint_id += len(signature)
        op.dim_hints = _dims_to_hints(  # type: ignore[attr-defined]
            op, signature, hint_ids
        )
        levels = [
            (hint_id, sympy.Integer(split_count))
            for hint_id, (_host_dim, split_count, _is_reduction) in zip(
                hint_ids, signature
            )
        ]
        groups.append(([op], levels))
        auto_tiled_producers.add(op.get_name())
        logger.debug(
            "[span-overflow groups] created group_index=%d op=%s levels=%s",
            len(groups) - 1,
            op.get_name(),
            levels,
        )

        level_summary = [
            (host_dim, split_count) for host_dim, split_count, _ in signature
        ]
        max_total = max(info.total_bytes for info in plan.chunking_infos)
        max_span = max(info.per_core_span for info in plan.chunking_infos)
        logger.info(
            "[span-overflow groups] op=%s levels=%s total=%.2fGB per_tile_span=%.2fMB",
            op.get_name(),
            level_summary,
            max_total / (1024**3),
            max_span / (1024**2),
        )

    flush_current_group()
    return groups


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
# Public API
# ---------------------------------------------------------------------------


def coarse_tile(
    graph: GraphLowering,
    groups: list[tuple],
) -> None:
    """Stamp loop_group_id / loop_count on operations and scale their ranges.

    Parameters
    ----------
    operations:
        The full ordered list of IR operations (as seen by
        CustomPreSchedulingPasses).  Modified in-place when
        insert_tiling_propagation inserts new buffer/copy ops.
    groups:
        Sequence of ``(ops, levels)`` tuples produced by
        ``hints_to_coarse_tile_groups``.  ``levels`` is a list of
        ``(hint_id, count)`` pairs, outermost first.
    """
    operations = graph.operations
    op_to_position: dict[str, int] = {
        op.get_operation_name(): i for i, op in enumerate(operations)
    }

    retiled_infos_by_group: list[
        tuple[tuple[int, ...], list[Operation], dict[str, _RetiledBufferInfo]]
    ] = []
    for group_idx, (group_ops, levels) in enumerate(groups):
        group_id: tuple[int, ...] = (group_idx,)
        retiled_infos = _stamp_group(group_ops, group_id, levels, op_to_position)
        stamped_group_id = group_id + (0,) * (len(levels) - 1)
        retiled_infos_by_group.append((stamped_group_id, group_ops, retiled_infos))

    insert_tiling_propagation(operations, groups)

    for group_id, group_ops, retiled_infos in retiled_infos_by_group:
        _patch_retiled_load_indexes(group_id, group_ops, retiled_infos, operations)


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
    the complete result by one of two strategies:

    Case 1 — output used both inside and outside the loop:
        Allocate a full-sized buffer.  Insert a copy op (same loop_group_id,
        same loop_tiled_dims) that writes each tile into the correct slice of
        the full buffer.  Patch outside consumers to read the full buffer.

    Case 2 — output used only outside the loop:
        Allocate a full-sized buffer.  Rewire the tiled op to write directly
        into the full buffer via MutationLayoutSHOULDREMOVE.  Patch outside
        consumers to read the full buffer.

    In both cases the existing tiled_symbols / affine.apply machinery in
    SpyreKernel and bundle.py handles the per-iteration address offset.
    """
    for group_ops, _ in groups:
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            if not isinstance(op.data, (Pointwise, Reduction)):
                continue
            _propagate_tiled_op(op, operations)


def _validate_reduction_tiling(op: ComputedBuffer) -> None:
    """Raise RuntimeError for unsupported Reduction tiling configurations.

    Supported:
      - A single level that tiles only a non-stick reduction dim.
      - A single level that tiles the stick (innermost) reduction dim, including
        the K dim of BATCH_MATMUL_OP and scalar reductions over dim=-1.
      - Multiple nesting levels where outer level(s) tile output dims and the
        innermost level tiles a reduction dim (e.g. outer M + inner K for mm).

    Deferred (raises RuntimeError):
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
            raise RuntimeError(
                f"coarse_tile: op {op.get_name()!r} level {i} tiles both "
                f"output dim(s) {out_dims} and reduction dim(s) {red_dims} "
                "simultaneously (mixed output+reduction tiling at one level "
                "is not yet implemented — Stage 2)."
            )
        if len(red_dims) > 1:
            raise RuntimeError(
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

    # If no dims were tiled (loop_tiled_dims all empty), the op is loop-invariant —
    # mark per_tile_fixed so the unroller reuses the same address each tile.
    if all(not dims for dims in loop_info.loop_tiled_dims):
        from .ir import FixedTiledLayout

        if isinstance(op.layout, FixedTiledLayout):
            op.layout.per_tile_fixed = True
        return

    if not outside_consumers and not is_graph_output:
        # Loop-internal: the buffer is a per-tile scratch region reused every
        # iteration.  Mark it so the unroller does not advance its base address.
        from .ir import FixedTiledLayout

        if isinstance(op.layout, FixedTiledLayout):
            op.layout.per_tile_fixed = True
        # Non-FixedTiledLayout buffers (e.g. MutationLayoutSHOULDREMOVE from a
        # prior pass) are intentionally left unmarked — their addressing is
        # handled by the layout type itself, not by the unroller.
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

    has_inside = _has_inside_consumers(buf_name, loop_group_id, operations)

    if has_inside:
        # Case 1: keep tiled op writing to small buffer; insert copy op.
        _insert_copy_op(op, full_buf, operations)
    else:
        # Case 2: tiled op has no inside consumers — rewire it to write directly
        # into the full-size buffer.  Note: MutationLayoutSHOULDREMOVE is
        # incompatible with lx_planning (scratchpad); do not combine the two.
        op.layout = MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf)))

    # Patch outside consumers and graph outputs to read full_buf.
    full_name = full_buf.get_name()
    _patch_consumers(outside_consumers, buf_name, full_name, operations)
    if is_graph_output:
        _patch_graph_outputs(buf_name, full_buf)

    logger.debug(
        "coarse_tile: propagated %s → %s (case %s)",
        buf_name,
        full_name,
        "1 (copy)" if has_inside else "2 (mutation)",
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


def _has_inside_consumers(
    buf_name: str,
    group_loop_id: tuple,
    operations: list[Operation],
) -> bool:
    """Return True if any op inside the same outermost loop group reads buf_name."""
    outer_key = group_loop_id[0]
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        li = getattr(op, "loop_info", None)
        if li is None or li.loop_group_id[0] != outer_key:
            continue
        if _reads_buffer(op, buf_name):
            return True
    return False


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
    a FixedTiledLayout matching tiled_op's layout, splices it into operations
    at insert_at_idx, and returns the new ComputedBuffer.
    """
    from .propagate_layouts import generic_layout  # deferred: avoids circular import
    from .ir import (
        FixedTiledLayout,
        SpyreEmptyFallback,
    )  # deferred: avoids circular import

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

    # Assign a FixedTiledLayout with the full size.
    orig_layout = tiled_op.layout
    # Recompute strides for the full size (contiguous row-major).
    strides: list[Expr] = []
    stride: Expr = sympy.Integer(1)
    for s in reversed(full_ranges):
        strides.insert(0, stride)
        stride = stride * s

    if isinstance(orig_layout, FixedTiledLayout):
        # Derive the full buffer's device layout from the per-tile layout by
        # scaling device_size entries up to the full host size.  The stick
        # orientation (dim_order / element_arrangement) is propagated verbatim
        # from orig_layout — both buffers must agree on physical layout so the
        # scatter copy op computes correct device addresses.
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
    else:
        device_layout = generic_layout(full_buf)

    layout = FixedTiledLayout(
        device,
        dtype,
        list(full_ranges),
        strides,
        device_layout,
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

    The copy op carries the same loop metadata as tiled_op so it executes
    inside the same loop body.  Its layout is MutationLayoutSHOULDREMOVE
    pointing at full_buf so store_output writes into full_buf.  Because
    loop_tiled_dims is set, SpyreKernel stamps tiled_symbols on the OpSpec
    and bundle.mlir emits affine.apply for the per-iteration output address.
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

    # Stamp with the same loop metadata so this op is inside the same loop.
    copy_buf.loop_info = tiled_op.loop_info  # type: ignore[attr-defined]

    V.graph.name_to_buffer[copy_name] = copy_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, copy_buf)


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
) -> None:
    """Insert a copy op that writes accum_tile → accum_full at the outer loop level.

    Reads accum_tile (per_tile_fixed=True, never advances) and writes into
    accum_full via MutationLayoutSHOULDREMOVE.  Carries outer_loop_info so
    the unroller advances accum_full per outer output-dim tile.
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
    V.graph.name_to_buffer[copy_name] = copy_buf

    combine_name = V.graph.qualify_name(f"coarse_tile_combine_{tiled_op.get_name()}")
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
        from .ir import FixedTiledLayout

        if isinstance(accum_tile.layout, FixedTiledLayout):
            accum_tile.layout.per_tile_fixed = True
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
    # kernel codegen can express this as an IDENTITY_OP broadcast.  We must
    # assign a FixedTiledLayout manually here because finalize_layouts has
    # already run when this pass executes.
    dtype = op.get_dtype()
    device = op.get_device()
    from .ir import (
        FixedTiledLayout,
        SpyreConstantFallback,
    )  # deferred: avoids circular import

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

    # Mark tiled op's output as per-tile scratch (no address advance).
    if not isinstance(op.layout, FixedTiledLayout):
        raise RuntimeError(
            f"coarse_tile: tiled reduction op {op.get_name()!r} has layout "
            f"{type(op.layout).__name__}, expected FixedTiledLayout; "
            "cannot mark per_tile_fixed"
        )
    op.layout.per_tile_fixed = True

    # Patch consumers to read accum_full (the fully-assembled output).
    buf_name = op.get_name()
    outside_consumers, is_graph_output = _find_outside_consumers(
        buf_name, loop_group_id, operations
    )
    accum_name = accum_full.get_name()
    _patch_consumers(outside_consumers, buf_name, accum_name, operations)
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
) -> None:
    """Redirect outside consumers from old_name to new_name.

    Patches each consumer's inner_fn via NameSwapHandler and reconstructs
    the ComputedBuffer to invalidate the sizes cache.
    """
    if not consumers or old_name == new_name:
        return

    from .insert_restickify import NameSwapHandler
    from .pass_utils import replace_computed_buffer_body

    name_map = {old_name: new_name}
    for consumer in consumers:
        orig_inner = consumer.data.inner_fn

        def new_inner_fn(*args, _map=name_map, _orig=orig_inner):
            with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
                return _orig(*args)

        object.__setattr__(consumer.data, "inner_fn", new_inner_fn)
        replace_computed_buffer_body(consumer, consumer.data, operations)
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

    from .pass_utils import replace_computed_buffer_body

    retiled_names = set(rewrites_by_name)
    for op in list(operations):
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


# ---------------------------------------------------------------------------
# Original stamping helpers (unchanged)
# ---------------------------------------------------------------------------


def _stamp_group(
    ops: list[Operation],
    group_id: tuple[int, ...],
    levels: list[tuple],
    op_to_position: dict[str, int],
) -> dict[str, _RetiledBufferInfo]:
    """Stamp loop_group_id / loop_count / loop_tiled_dims and divide ranges.

    ``levels`` is a list of ``(hint_id, count)`` pairs, outermost first.  Each
    op resolves its own tiled dimension from its loop_var in dim_hints.  Ops
    that have no matching dim for a level are loop-invariant at that level.

    For each (op, hint_id) pair the dispatch is per-op:
    - If hint_id is in hint_id_to_ranges_pos (output dim for this op):
      populate loop_tiled_dims and call _divide_ranges.
    - If hint_id is in hint_id_to_reduction_ranges_pos (reduction dim for this
      op): populate loop_tiled_reduction_dims and call _divide_reduction_ranges.
    - These are mutually exclusive per op (enforced by _validate_reduction_tiling).
    - If hint_id is in neither (broadcast op): both lists get [] for this level.

    End-to-end correctness of the reduction path is covered by
    TestCoarseTileReductionDim0E2E in tests/inductor/test_coarse_tile_e2e.py.
    """
    if not ops:
        return {}

    _validate_contiguous(ops, op_to_position, group_id)

    nested_group_id: tuple[int, ...] = group_id + (0,) * (len(levels) - 1)
    counts = [count for _, count in levels]
    retiled_infos: dict[str, _RetiledBufferInfo] = {}

    for op in ops:
        if not isinstance(op, ComputedBuffer):
            logger.debug(
                "coarse_tile: skipping non-ComputedBuffer op %s (%s)",
                op.get_operation_name(),
                type(op).__name__,
            )
            continue

        op_out = op_out_coords(op)

        # Build lookup: hint_id → output-ranges position (non-reduction dims).
        hint_id_to_ranges_pos: dict[int, int] = {
            h.hint_id: pos
            for h in getattr(op, "dim_hints", [])
            if h.loop_var is not None and not h.is_reduction
            if (pos := _loop_var_to_ranges_pos(op_out, h.loop_var)) is not None
        }

        # Build lookup: hint_id → reduction_ranges position (reduction dims).
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
        for hint_id, count in levels:
            opos = hint_id_to_ranges_pos.get(hint_id)
            rpos = hint_id_to_reduction_ranges_pos.get(hint_id)
            op_tiled_dims.append([opos] if opos is not None else [])
            op_tiled_reduction_dims.append([rpos] if rpos is not None else [])
            # _divide_ranges with tiled_dims=[] is a no-op.
            retiled_info = _divide_ranges(op, count, [opos] if opos is not None else [])
            if retiled_info is not None:
                name = op.get_name()
                prior = retiled_infos.get(name)
                retiled_infos[name] = (
                    _RetiledBufferInfo(prior.old_stride, retiled_info.new_stride)
                    if prior is not None
                    else retiled_info
                )
            if isinstance(op.data, Reduction):
                # NOTE: _divide_reduction_ranges mutates data.reduction_ranges
                # before _validate_reduction_tiling runs in the later
                # insert_tiling_propagation pass.  If validation raises (e.g.
                # mixed output+reduction at one level), the mutated ranges are
                # never observed: the RuntimeError propagates uncaught through
                # the pass runner and aborts compilation.
                _divide_reduction_ranges(op, count, [rpos] if rpos is not None else [])

        op.loop_info = CoarseTileInfo(  # type: ignore[attr-defined]
            loop_group_id=nested_group_id,
            loop_count=counts,
            loop_tiled_dims=op_tiled_dims,
            loop_tiled_reduction_dims=op_tiled_reduction_dims,
        )

        logger.debug(
            "coarse_tile: stamped %s loop_group_id=%s loop_count=%s "
            "loop_tiled_dims=%s loop_tiled_reduction_dims=%s",
            op.get_operation_name(),
            nested_group_id,
            counts,
            op_tiled_dims,
            op_tiled_reduction_dims,
        )

    return retiled_infos


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
    from .pass_utils import (
        host_coordinates,
        indirect_sizes_from_op,
        try_device_coordinates,
    )
    from .views import matching_dim

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
                raise RuntimeError(
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
    from torch._inductor.ir import FixedLayout, FlexibleLayout

    from .ir import FixedTiledLayout

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
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} reduction dim {i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"reduction dimensions must be evenly divisible by the loop trip count."
                )
            reduction_ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            reduction_ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)
    # Reduction is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "reduction_ranges", reduction_ranges)


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
