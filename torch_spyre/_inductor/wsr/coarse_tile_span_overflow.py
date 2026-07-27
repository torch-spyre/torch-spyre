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

"""Span-overflow-driven coarse-tile group construction.

Builds coarse-tile groups automatically when an op's per-core working set
would overflow available span, for consumption by ``coarse_tile()`` in
``coarse_tile.py``. Runs POST-stickification (see ``passes.py``'s
``_maybe_coarse_tile_span_overflow``) — requires ``FixedTiledLayout``
(``device_layout``) on all ops.
"""

from __future__ import annotations

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise, Reduction

from ..errors import Unsupported
from ..logging_utils import get_inductor_logger
from ..propagate_hints import DimHint
from ..pass_utils import op_out_coords, host_coordinates, indirect_sizes_from_op
from ..ir import FixedTiledLayout
from .span_overflow_hint_analysis import (
    SpanOverflowTilePlan,
    can_conform_pointwise_tile,
    plan_span_overflow_tile,
)

logger = get_inductor_logger("coarse_tile")

_SPAN_OVERFLOW_HINT_ID = 10000


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


def span_overflow_groups(
    graph: GraphLowering,
) -> tuple[list[tuple], list[tuple[Operation, list[DimHint]]]]:
    """Build coarse_tile() groups from automatic span-overflow plans.

    This adapter converts SpanOverflowTilePlans into the same group shape as
    user spyre_hint annotations: ``[(ops, [(hint_id, count)])]``.  Ops that
    already carry user dim hints are left for the user-hint grouping path.
    ``is_reduction`` is not carried in the group-level ``levels`` list; it
    lives on each op's own ``DimHint`` and is consulted directly by
    ``plan_coarse_tile_groups`` (via ``_op_hint_dim_positions``).

    Returns a ``(groups, dim_hint_assignments)`` pair.  ``dim_hint_assignments``
    is a list of ``(op, dim_hints)`` pairs this function decided on but did
    NOT apply — applying them (setting ``op.dim_hints``) is the caller's
    responsibility, and must happen before ``coarse_tile()``/
    ``validate_coarse_tile_groups`` run, since ``dim_hints`` is an input those
    consume, not something they produce.  Keeping the assignment out of this
    function's own decision logic keeps ``span_overflow_groups`` a pure
    planning step: nothing here mutates ``op`` state.

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
    from .. import config

    if config.ignore_wsr_hints or config.ignore_span_overflow_hints:
        logger.debug(
            "[span-overflow groups] disabled ignore_wsr_hints=%s ignore_span_overflow_hints=%s",
            config.ignore_wsr_hints,
            config.ignore_span_overflow_hints,
        )
        return [], []

    logger.debug(
        "[span-overflow groups] begin ops=%d sencores=%s",
        len(graph.operations),
        config.sencores,
    )
    groups: list[tuple] = []
    dim_hint_assignments: list[tuple[Operation, list[DimHint]]] = []
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
            dim_hint_assignments.append(
                (grouped_op, _dims_to_hints(grouped_op, dims, hint_ids))
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
        dim_hint_assignments.append((op, _dims_to_hints(op, signature, hint_ids)))
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
    return groups, dim_hint_assignments
