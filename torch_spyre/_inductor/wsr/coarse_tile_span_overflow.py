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
would overflow available span, for consumption by
``coarse_tile_post_stickify()`` in ``coarse_tile.py``. Runs
POST-stickification (see ``passes.py``'s
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
from .coarse_tile import _loop_var_to_reduction_ranges_pos
from .span_overflow_hint_analysis import (
    SpanOverflowTilePlan,
    _bmm_k_symbol,
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


def _consumer_shares_group_tiled_dim(
    op: ComputedBuffer,
    signature: tuple[tuple[int, int, bool], ...],
    current_group: list,
) -> bool:
    """True if a consumer's tiled output dim(s) are the *same logical dim(s)*
    as the tiled dim(s) of the producer(s) it reads in the open group.

    Joining ops into one group means they share a single loop nest: iteration
    ``t`` computes tile ``t`` of every member.  For that to be correct, the
    consumer's tiled dimension must be the dimension that — through its read of
    the producer — indexes the producer's tiled dimension.  Matching split
    counts is necessary but not sufficient (two unrelated dims could split into
    the same count), so verify the loop-variable correspondence explicitly: the
    symbol tiling the consumer's output dim must appear in the producer's tiled
    coordinate as seen through the read.  A plan may carry several tile levels;
    the check is made **per level**, pairing producer level ``i`` with consumer
    level ``i`` the way the rest of the pass does, so a multi-level plan whose
    levels correspond only crosswise is rejected rather than accepted.

    Conservative: any failure to establish the correspondence returns False, so
    an unverifiable pair is left to the normal (Unsupported) conflict path
    rather than fused into a possibly-desynchronized loop.

    This check is agnostic to both ops' types — what makes the join safe is
    that the tiled dim is an **output range**, not the reduction range: tile
    ``t`` of an output dim is self-contained (it reads only tile ``t`` of the
    producer).  ``span_overflow_groups`` therefore applies it to every
    producer/consumer combination it groups: Pointwise or Reduction on either
    side.

    Once a **Reduction** sits on either side the check stops being a formality
    and becomes the thing keeping the pass correct, in two distinct ways.

    With a Reduction *producer* and a Pointwise consumer (BMM -> PW): the
    producer's output dim positions need not line up with the consumer's, yet
    the consumer inherits the group's ``host_dim`` *positionally* (see the
    conform branch in ``span_overflow_groups``).  If the numbering diverges,
    ``can_conform_pointwise_tile`` would validate the wrong dim, possibly pass
    by coincidence, and ``_dims_to_hints`` would resolve a ``loop_var`` for a
    dim unrelated to the producer's tile — a desynchronized loop nest computing
    wrong results silently.

    With a Reduction on *both* sides (BMM -> BMM, BMM -> sum): the producer's
    tiled dim may land on the consumer's **reduction (K) range** rather than an
    output range.  In ``bmm(bmm(q, k), v)``, a producer tiling its N dim feeds a
    consumer whose K *is* that N, so tile ``t`` of the producer is a partial
    slice of the consumer's reduction — pairing them per-iteration computes a
    partial sum.  K never appears in the consumer's output coordinates, so the
    intersection is empty and the pair is rejected, while the safe variants of
    the same chain (producer tiling B or M, or tiling N with the consumer
    reading it as its B operand) still verify.

    Both cases fail closed via the loop-variable correspondence below.

    The automatic planner normally tiles output ranges, but its BMM fallback can
    tile K.  We assert the output-range invariant explicitly below -- on both
    the consumer's signature and each producer's dims -- and fail closed if a
    reduction-range tile reaches this check, since such a tile would break the
    loop-carried accumulation this join assumes away.  K fallback plans must
    remain independent groups.

    Checking the producer side matters only because a Reduction can now root a
    run (see ``span_overflow_groups``).  Before that, an unjoined Reduction
    became a closed singleton and could never be an open group's producer, so
    the consumer-side check alone covered every reachable case.
    """
    # Guard: only output-range tiles may join.  A reduction (K) range tile
    # would need cross-tile accumulation and cannot share a per-tile loop nest.
    if any(is_reduction for _host_dim, _split, is_reduction in signature):
        return False
    try:
        consumer_coords = op_out_coords(op)
        # Group deps by name rather than keying a dict on it: one op can read
        # the same buffer through *several* deps at different indices (a matmul
        # taking the same tiled producer as both operands, e.g. ``x @ x.T``).
        # Collapsing them would verify one access pattern and silently leave
        # the others unchecked, which is exactly the case Reduction-to-Reduction
        # grouping makes reachable.  Every dep must correspond.
        reads: dict[str, list[MemoryDep]] = {}
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep):
                reads.setdefault(dep.name, []).append(dep)
        indirect = indirect_sizes_from_op(op)
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError, IndexError):
        # op_out_coords internally calls host_coordinates, which can raise the
        # same ValueError/RuntimeError/IndexError as the direct call below, so
        # this list must cover that set too (plus AttributeError for the
        # get_read_writes()/indirect_sizes_from_op attribute access).
        return False

    # Collect the tiling symbols *per level*, not unioned across levels.  Levels
    # are paired by position everywhere else in the pass -- the join branch
    # compares split counts positionally and ``_dims_to_hints`` zips levels to
    # hint_ids in the same order -- so producer level ``i`` shares its loop with
    # consumer level ``i`` and each pair must correspond on its own.  A union
    # would let a producer's level-0 dim satisfy the check by matching the
    # consumer's level-1 dim: a cross-level match that passes while the per-level
    # loop nests do not correspond.  The two are equivalent for single-level
    # plans, but the auto planner emits one level per output dim it must tile
    # (see ``plan_span_overflow_tile``), so multi-level plans are reachable and
    # the per-level form is the fail-closed one.
    consumer_level_syms: list[set] = []
    for host_dim, _split, _is_reduction in signature:
        if host_dim >= len(consumer_coords):
            return False
        level_syms = consumer_coords[host_dim].free_symbols
        if not level_syms:
            # A tiled dim with no loop var of its own cannot be shown to
            # correspond to anything -- fail closed rather than skip the level.
            return False
        consumer_level_syms.append(level_syms)
    if not consumer_level_syms:
        return False

    group_by_name = {gop.get_name(): (gop, dims) for gop, dims in current_group}
    verified_any = False
    for name, deps in reads.items():
        if name not in group_by_name:
            continue
        producer, producer_dims = group_by_name[name]
        if any(is_reduction for _host_dim, _split, is_reduction in producer_dims):
            # Same rule as the consumer-side guard at the top, applied to the
            # producer: tile t of a reduction-range tile is a partial
            # accumulation, so it cannot be paired with tile t of anything.
            return False
        if len(producer_dims) != len(consumer_level_syms):
            # Levels are paired by position; different level counts mean there is
            # no such pairing to verify.  (The callers only group ops with equal
            # split-count lists, so this is a guard against a future caller
            # rather than a case reachable today.)
            return False
        for dep in deps:
            try:
                producer_coords = host_coordinates(producer.get_layout(), dep, indirect)
            except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
                return False
            for level, (host_dim_p, _split, _is_reduction) in enumerate(producer_dims):
                if host_dim_p >= len(producer_coords):
                    return False
                if not (
                    producer_coords[host_dim_p].free_symbols
                    & consumer_level_syms[level]
                ):
                    # The consumer's loop var *at this level* does not index the
                    # producer's tiled dim at the same level -> not the same
                    # logical dim -> unsafe to share a loop.  Checking level by
                    # level is what makes a multi-level plan whose levels match
                    # only crosswise fail here rather than slip through.
                    # For a Reduction consumer this is also what
                    # rejects a producer whose tiled dim lands on the consumer's
                    # reduction (K) range: K never appears in the consumer's
                    # output coordinates, so the intersection is empty and tile
                    # t of the producer -- a partial slice of the consumer's
                    # reduction -- can never be paired with tile t of the
                    # consumer's output.
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
        if is_reduction:
            reduction_ranges = list(getattr(op.data, "reduction_ranges", []))
            if host_dim >= len(reduction_ranges):
                raise Unsupported(
                    f"Cannot adapt span-overflow reduction plan for {op.get_name()}: "
                    f"host_dim={host_dim} is out of bounds for reduction ranges "
                    f"{reduction_ranges}."
                )
            loop_var = _bmm_k_symbol(op)
            if loop_var is None:
                raise Unsupported(
                    f"Cannot adapt span-overflow reduction plan for {op.get_name()}: "
                    "could not identify the BMM K loop variable."
                )
            try:
                reduction_pos = _loop_var_to_reduction_ranges_pos(op, loop_var)
            except (StopIteration, AttributeError, TypeError, ValueError):
                reduction_pos = None
            if reduction_pos != host_dim:
                raise Unsupported(
                    f"Cannot adapt span-overflow reduction plan for {op.get_name()}: "
                    f"BMM K loop variable {loop_var} maps to reduction range "
                    f"position {reduction_pos}, expected {host_dim}."
                )
            coord = loop_var
        else:
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


# State machine walked by ``span_overflow_groups`` below.  It scans the ops in
# graph order carrying at most one *open run* -- a set of ops that will share one
# loop nest -- held in three variables: ``current_group`` (the members),
# ``current_signature`` (the shared tiling), and ``current_root_is_reduction``
# (which of the two run kinds it is).  ``flush_current_group()`` emits the open
# run as a group and returns to CLOSED.
#
#                          .---------------------------------.
#                          |             CLOSED              |<--------------.
#                          |   current_group == []           |               |
#                          |   current_signature is None     |               |
#                          '---------------------------------'               |
#                             |                         |                    |
#          Pointwise w/ plan  |                         |  Reduction w/ plan |
#                             |                         |  joining nothing   |
#                             v                         v                    |
#     .------------------------------.   .------------------------------.    |
#     |        PW-ROOTED RUN         |   |       RED-ROOTED RUN         |    |
#     |   root_is_reduction = False  |   |   root_is_reduction = True   |    |
#     '------------------------------'   '------------------------------'    |
#        ^      |                            ^      |                        |
#        |      |  PW, same signature        |      |  PW, same signature    |
#        '------'  -- or -- PW that reads    '------'  AND reads the run     |
#          stay    the run and conforms        stay    AND correspondence    |
#                  to its split (can_                   -- or -- PW that     |
#                  conform_pointwise_tile)               conforms AND        |
#                                                        correspondence      |
#              |                                |                            |
#              |  Reduction that reads the run, equal split counts, and       |
#              |  _consumer_shares_group_tiled_dim: append, then FLUSH        |
#              |  (a *joining* Reduction always ends its group; the Reduction |
#              |  that ROOTED the run is its first member, not its last) -----|
#              |                                |                            |
#              |  op reads the run but fits no branch above: FLUSH, then      |
#              |  raise Unsupported (two unsynchronized loop nests) ----------|
#              |                                |                            |
#              |  anything else (no plan, hinted, non-tileable, end of        |
#              |  graph): FLUSH -- the run may re-open on the next op --------'
#
# The two run kinds differ in exactly one way: a PW-rooted run lets the two
# Pointwise fast paths in on matching signatures alone, while a RED-rooted run
# additionally demands a real read edge plus the loop-var correspondence check,
# because a Reduction producer's output dim numbering need not match its
# consumer's.  The reduction-join branch is identical for both -- it always runs
# the correspondence check -- which is what makes BMM -> BMM and BMM -> sum work.
def span_overflow_groups(
    graph: GraphLowering,
) -> tuple[list[tuple], list[tuple[Operation, list[DimHint]]]]:
    """Build coarse_tile_post_stickify() groups from automatic span-overflow
    plans.

    This adapter converts SpanOverflowTilePlans into the same group shape as
    user spyre_hint annotations: ``[(ops, [(hint_id, count)])]``.  Ops that
    already carry user dim hints are left for the user-hint grouping path.
    ``is_reduction`` is not carried in the group-level ``levels`` list; it
    lives on each op's own ``DimHint`` and is consulted directly by
    ``plan_coarse_tile_groups``.

    Returns a ``(groups, dim_hint_assignments)`` pair.  ``dim_hint_assignments``
    is a list of ``(op, dim_hints)`` pairs this function decided on but did
    NOT apply — applying them (setting ``op.dim_hints``) is the caller's
    responsibility, and must happen before ``coarse_tile_post_stickify()``/
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

    A Reduction op does not extend a Pointwise run.  Any Reduction
    (matmul/BMM, sum, mean, ...) may **join** an open run's group when it
    reads a producer in that run and tiles the same shared logical (output)
    dim at the same split count — e.g. an F.linear matmul reading its
    auto-tiled restickified weight, a plain ``sum`` reading a tiled pointwise
    producer, or a second matmul reading the first's tiled output (see the
    reduction-join branch below and ``_consumer_shares_group_tiled_dim``).
    The join is reduction-type-agnostic: what makes it safe is that the tiled
    dim is an output range, not the reduction range (tile ``t`` is
    self-contained either way).  On joining, the group is flushed immediately,
    so a *joining* Reduction is always the last member of its group and each
    auto-tiled producer feeds at most one reduction consumer.  Note this says
    nothing about a Reduction that **roots** a run (below): that one is its
    group's first member, and the group may well end on a Pointwise op -- e.g.
    ``[bmm, pointwise]``, as in
    ``test_bmm_producer_groups_with_pointwise_consumer``.

    A Reduction that cannot join instead opens its own run when it tiles an
    output range, so a directly-connected consumer can fuse into its loop
    (BMM -> PW, BMM -> BMM, BMM -> sum).  Tile ``t`` of a Reduction's
    *output* dim is self-contained exactly as a Pointwise tile is, so the
    producer's per-tile slice feeds the consumer in the same iteration.  A
    K-only reduction-range plan is the exception: it remains an independent
    group and consumes producers through their materialized outputs.  A
    **Pointwise** consumer of a Reduction-rooted run must additionally read the
    run and pass ``_consumer_shares_group_tiled_dim`` -- the Pointwise fast
    paths otherwise reuse the run's ``host_dim`` positionally, which is unsound
    once the producer is a Reduction (see that helper).  A **Reduction**
    consumer already goes through that check on the join branch, which is also
    what rejects a producer whose tiled dim lands on the consumer's reduction
    (K) range.  If nothing joins, the run flushes to exactly the singleton
    group an unjoined Reduction used to produce eagerly.

    Any plan that reads a buffer from an already-closed group, or from the
    open run without being fusable into it, still raises ``Unsupported``: two
    independent loop nests over the same span-overflow-sized data can
    desynchronize, and for ops tiled specifically because their *full* buffer
    violates the hardware span limit, falling back to materializing that full
    buffer for an outside consumer would silently reintroduce the exact span
    violation tiling was meant to prevent. K-only plans also fail this guard:
    their reduction-range loop cannot be synchronized with an output-tiled
    producer, so independently nesting the two plans is unsafe.
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
    # Output-range-tiled producers whose group was closed by a Reduction consumer joining it (see
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
    # True when the open run was opened by a Reduction rather than a Pointwise
    # op (BMM -> PW, BMM -> BMM, BMM -> sum).  A consumer of such a run faces a
    # producer whose output dim numbering need not match its own, so the
    # positional reuse of the run's ``host_dim`` that the two **Pointwise** fast
    # paths rely on (matching-signature join and ``can_conform_pointwise_tile``)
    # is no longer self-justifying: this flag is what makes those two paths
    # additionally require a real read edge and
    # ``_consumer_shares_group_tiled_dim``.  A **Reduction** consumer needs no
    # such gate here because the reduction-join branch already runs that check
    # unconditionally -- Reduction -> Reduction chaining is supported, and the
    # join branch is deliberately *not* keyed on this flag.
    current_root_is_reduction = False

    def allocate_hint_ids(signature: _PwDims) -> list[int]:
        """Reserve one contiguous hint-ID range for a coarse-tile group."""
        nonlocal next_hint_id
        hint_ids = list(range(next_hint_id, next_hint_id + len(signature)))
        next_hint_id += len(signature)
        return hint_ids

    def flush_current_group() -> None:
        nonlocal current_group, current_signature
        nonlocal current_root_is_reduction
        if not current_group:
            current_root_is_reduction = False
            return

        signature = current_signature
        assert signature is not None
        hint_ids = allocate_hint_ids(signature)
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
        current_root_is_reduction = False

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
            # it outside any loop: Pass 3's outside-consumer path
            # (coarse_tile.py) already patches consumers of a tiled producer
            # to read a full, reassembled buffer, and plan_span_overflow_tile
            # returning None here means that op's own full-size reads/writes
            # are already known not to overflow.
            logger.debug("[span-overflow groups] op=%s no auto plan", op.get_name())
            flush_current_group()
            continue

        signature = _auto_span_plan_signature(plan)
        has_reduction_range = any(
            is_reduction for _host_dim, _split, is_reduction in signature
        )
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
        joined_conflicts = sorted(set(completed_conflicts) & reduction_joined_producers)
        if joined_conflicts:
            # A producer already synchronized with one reduction consumer
            # cannot safely feed a second independently tiled reduction.
            raise Unsupported(
                f"Cannot auto-tile {op.get_name()}: it reads producer(s) "
                f"{joined_conflicts} that were already auto-tiled and joined "
                "by another reduction consumer. A single auto-tiled producer "
                "can currently feed only one reduction consumer in one "
                "synchronized group; multiple consumers sharing one "
                "auto-tiled producer is not yet supported (#3217)."
            )
        if completed_conflicts:
            logger.warning(
                "[span-overflow groups] op=%s rejected_conflicting_auto_producers=%s",
                op.get_name(),
                completed_conflicts,
            )
            raise Unsupported(
                f"Cannot auto-tile {op.get_name()}: it reads already-tiled "
                f"producer(s) {completed_conflicts} that are not in an open "
                "group this op can join — either their coarse-tile group has "
                "already been flushed, or they were tiled by a user "
                "spyre_hint. Automatic span-overflow grouping can only fuse a "
                "consumer into a producer's group while that group is still "
                "open (the producers must be contiguous with the consumer), so "
                "tiling this producer and consumer independently can produce "
                "unsynchronized loop nests."
            )

        is_reduction_op = isinstance(op.data, Reduction)

        can_join_pw_group = (
            not is_reduction_op
            and current_signature is not None
            and signature == current_signature
        )
        if can_join_pw_group and current_root_is_reduction:
            # Two adjacent Pointwise ops that independently pick the same
            # signature share an iteration space, so matching signatures alone
            # justify their join.  A Reduction-rooted run gives no such
            # guarantee: the run's ``host_dim`` numbering is the *Reduction's*,
            # and an unrelated op that merely happens to agree numerically is
            # not tiling the same logical dim.  Require a real read edge plus
            # the loop-var correspondence, so only a genuine consumer joins.
            can_join_pw_group = bool(read_deps & current_group_names) and (
                _consumer_shares_group_tiled_dim(op, signature, current_group)
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
            # ``split_by_host_dim`` reuses the run's host_dim positions as
            # op's own.  That is sound for a Pointwise-rooted run (matching
            # iteration spaces), but a Reduction producer's output dims need
            # not sit at the same positions as its consumer's, so verify the
            # correspondence before letting op adopt the run's split — see
            # _consumer_shares_group_tiled_dim.
            if can_conform_pointwise_tile(op, split_by_host_dim, config.sencores) and (
                not current_root_is_reduction
                or _consumer_shares_group_tiled_dim(
                    op, current_signature, current_group
                )
            ):
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
        # split into the same count.  _consumer_shares_group_tiled_dim verifies
        # the consumer's tiled loop var actually indexes the producer's tiled dim
        # through the read, so the shared loop pairs matching slices.  It also
        # fails closed if the consumer tiles its reduction (K) range rather than
        # an output range (see its docstring) — only output-range tiles may join.
        #
        # The run may be Pointwise-rooted (PW -> BMM, the LM-head case) or
        # Reduction-rooted (BMM -> BMM, BMM -> sum): what licenses the join is
        # the shared *output*-range tile, not either op's type.  For a Reduction
        # producer the correspondence check carries real weight rather than
        # merely confirming an obvious pairing.  Consider chained matmuls,
        # ``bmm(bmm(q, k), v)``: if the producer tiles its N dim and the
        # consumer reads that buffer as its A operand, the producer's N *is* the
        # consumer's K, so tile t of the producer is a partial slice of the
        # consumer's reduction range and pairing them per-iteration would
        # compute a partial sum and call it a result.  K never appears in the
        # consumer's output coordinates, so the intersection is empty and
        # _consumer_shares_group_tiled_dim rejects it — while the safe variants
        # of the same chain (producer tiling B or M, or tiling N with the
        # consumer reading it as its B operand) still verify and join.
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
            and _consumer_shares_group_tiled_dim(op, signature, current_group)
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
                f"Cannot auto-tile {op.get_name()}: it reads auto-tiled "
                f"producer(s) {pending_conflicts} in the open group but cannot "
                "join them. Automatic span-overflow grouping requires the "
                "consumer to tile the same shared output dimension at the same "
                "split count(s) as the producer — its tiled loop variable must "
                "actually index the producer's tiled dim through the read (a "
                "Pointwise consumer may instead conform to the producer's "
                "split). Tiling this producer and consumer independently can "
                "produce unsynchronized loop nests."
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
        # either opens a run of its own (for output-only tiles) or stays as an
        # independent singleton (for any plan containing a reduction-range tile).
        # A K level is a partial accumulation, so K-only and combined output+K
        # plans cannot join or root a producer-consumer run.
        if has_reduction_range:
            hint_ids = allocate_hint_ids(signature)
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
        else:
            # Output-tiled Reductions open a run so a directly-connected
            # consumer can fuse into their loop (BMM -> PW, BMM -> BMM,
            # BMM -> sum).  Tile t of an output dim is self-contained just as
            # a Pointwise tile is.
            current_group.append((op, signature))
            current_signature = signature
            current_root_is_reduction = True
            logger.info(
                "[span-overflow groups] op=%s started_new_reduction_rooted_group split=%s",
                op.get_name(),
                list(signature),
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
