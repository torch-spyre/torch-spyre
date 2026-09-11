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

"""Per-candidate core-division machinery -- what stands in for a menu.

A solver can be handed an enumerated list of core-division candidates per op
plus a precomputed ``|D_p| x |D_c|`` compatibility table per edge, or it can
generate candidates as it goes and ask these functions the same questions one
candidate at a time. This module holds the per-candidate side:

* :func:`_core_division` classifies a symbol-keyed split map into the output /
  reduction split pair a :class:`CoreDivision` carries, and
  :func:`_division_splits` restores the complete map from it;
* :class:`OpSplitSpace` answers what the enumeration is a cross product over --
  the axes, each axis's legal factors, the coarse tilings the op may take, and
  whether a proposed (split, tiling) pair is legal -- so a caller can walk the
  space instead of materializing it, and :meth:`OpSplitSpace.neighbours` is the
  move alphabet that walk proposes from;
* :class:`ResidencyEdge` owns one producer-buffer -> consumer edge, both the
  geometry (does this pair of candidates slice the buffer identically) and the
  policy filters that decide a candidate can host a readable residency at all;
  :meth:`ResidencyEdge.consumer_division_for` and its mirror *construct* the
  other end's division rather than looking it up.

``allocator.py`` hands both to the solver on each buffer (``division_space`` and
``residency_edges``), and *also* materializes the edge relation as the
``cd_parent_matches`` pair table, which the CP-SAT and DFS engines still consume
and which serves as the fallback wherever a space could not be derived. Nothing
here decides *which* candidate to take: the space and the edge are pure oracles,
so a search owns its own proposal distribution and its own randomness.
"""

from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence
from typing import Optional

import sympy
from torch._inductor.dependencies import Dep, MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise, Reduction

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    invert_per_core_view,
    iteration_space_from_op,
    op_read_writes,
    op_short_name,
    _is_matmul_op,
    _per_core_view_from_prep,
    _prepare_per_core_view,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision, TileSpec
from torch_spyre._inductor.work_division import (
    WorkDivisionContext,
    work_division_context_for_op,
)
from torch_spyre._inductor.wsr.enumerate_tilings import TilingSpace

# The inert default every untiled division carries. Frozen, so one instance
# serves as a shared default argument.
UNTILED = TileSpec()


def _core_division(op: Operation, splits: dict[sympy.Symbol, int]) -> CoreDivision:
    """Classify one symbol-keyed candidate for its producing operation."""
    rw = op_read_writes(op)
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        return CoreDivision()
    output = {
        s: int(v) for s, v in splits.items() if write.index.coeff(s) != 0 and v > 1
    }
    reduction = {
        s: int(v) for s, v in splits.items() if write.index.coeff(s) == 0 and v > 1
    }
    return CoreDivision(output_splits=output, reduction_splits=reduction)


def _division_splits(op: Operation, division: CoreDivision) -> dict[sympy.Symbol, int]:
    """Restore a complete symbol-keyed split map from a sparse division."""
    return {
        sym: int(division.output_splits.get(sym, division.reduction_splits.get(sym, 1)))
        for sym in iteration_space_from_op(op)
    }


def _view_for_div(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    division: CoreDivision,
    prep_cache: dict,
):
    """One candidate division's per-core view of ``buf_name``.

    ``prep_cache`` holds the candidate-invariant (sympy-heavy) context, keyed by
    ``(op name, dep, buf_name)``: a producer's write-dep and a consumer's
    read-dep on the same buffer can be equal ``MemoryDep``s, so the op name
    keeps their preps distinct while a parent read by several consumers reuses
    its write-view prep.
    """
    return _per_core_view_from_prep(
        _prep_for(op, dep, buf_name, prep_cache),
        _division_splits(op, division),
        division.reduction_splits,
    )


def _prep_for(op: Operation, dep: MemoryDep, buf_name: str, prep_cache: dict):
    """The candidate-invariant view prep for one ``(op, dep, buf_name)``."""
    key = (op.get_name(), dep, buf_name)
    if key not in prep_cache:
        prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
    return prep_cache[key]


def _is_frame_changing_clone(op: Operation, buf_name: str) -> bool:
    """True if ``op`` is a clone whose output ``buf_name`` has an iteration
    dimension that none of its inputs carry -- i.e. it broadcasts a dim
    (e.g. GQA broadcasting K/V over the query-group axis). Such a clone reads
    its input in a different frame than it writes its output, so a per-core
    slice of the output cannot be produced from a core-local slice of the
    input; pinning the output mis-addresses (cf. the restickify barrier)."""
    if op_short_name(op) != "clone":
        return False
    rw = op_read_writes(op)
    write = next(
        (w for w in rw.writes if w.name == buf_name and hasattr(w, "index")), None
    )
    if write is None:
        return False
    read_syms: set = set()
    for r in rw.reads:
        if hasattr(r, "index"):
            read_syms |= set(r.index.free_symbols)
    # A write-only free symbol means the clone expands (broadcasts) that dim.
    return bool(set(write.index.free_symbols) - read_syms)


def undeclared_splits(division: CoreDivision, sym_core_divs: tuple[dict, dict]) -> set:
    """The split keys of ``division`` that ``sym_core_divs`` declares no symbol
    for.

    A solver prices a division through the symbols its buffer declares -- one
    per stride coefficient seen across an enumerated menu. A split on an axis
    outside that declaration is priced at the symbol's default of ``1`` rather
    than rejected, so it is a wrong answer and not a failure. Empty is the
    contract; a generator has to be held to it, which is why generation stays
    inside the declaration rather than widening it.
    """
    out_syms, red_syms = sym_core_divs
    return (set(division.output_splits) - set(out_syms)) | (
        set(division.reduction_splits) - set(red_syms)
    )


@dataclass
class OpSplitSpace:
    """One op's legal core divisions as a space to move in, not a list.

    Everything :func:`enumerate_work_division_candidates` needs, asked one
    candidate at a time: which axes there are, what factors each admits, and
    whether a proposed split map is legal. The enumerated menu is the cross
    product over exactly these answers, so a division this space admits is one
    the menu would have carried -- generation changes when a candidate is
    materialized, not which candidates exist.

    Two things narrow it beyond ``WorkDivisionContext.is_legal``: the
    ``declaration`` (see :func:`undeclared_splits`), and the split roles. Which
    axes are *output* axes and which are *reduction* axes is a property of the
    op's write index rather than of a candidate, so it is derived once here and
    :meth:`division` classifies without touching sympy again.

    **The coarse tiling rides on the same candidate**, and is the one part of
    the space the menu does not carry. Tiling rewrites index expressions and
    ``splits_by_index_coeff`` keys the output splits by each symbol's
    coefficient in the write index, so a :class:`CoreDivision` carried across
    tilings is uninterpretable rather than merely illegal -- the two have to be
    chosen together. That makes the space two-level and ragged: a tile level
    cuts its axis's per-tile extent, so the core splits that still divide it
    exactly are a strictly *narrower* set than the untiled one.

    Narrower in one direction only, and deliberately. A tiling also shrinks the
    per-core span, which would let ``MAX_SPAN_BYTES`` and the floor
    ``span_reduction_pass`` committed admit *smaller* split counts than they do
    untiled -- so the honest tiled domain drops large factors and gains small
    ones, and would be incomparable to the untiled one rather than a subset of
    it. That half is not modelled here (see
    :meth:`WorkDivisionContext.is_legal`), so these domains are nested. It
    costs an option, never a verdict.
    """

    op: Operation
    context: WorkDivisionContext
    # The buffer's ``sym_core_divs``; ``None`` for a caller with no cost
    # expression to price against, which then constrains nothing.
    declaration: Optional[tuple[dict, dict]]
    # Axes whose factor slices the op's output (the rest are reduction axes).
    output_axes: frozenset
    # Per axis, the legal factors *untiled* -- the widest domain, which a tiling
    # can only narrow (:meth:`factor_domain`). What the view inverse searches
    # over, and what a recolor anchor redraws from.
    factor_domains: dict[sympy.Symbol, list[int]]
    # The coarse tilings this op may take, or ``None`` when tilings are not this
    # caller's to choose -- which pins every division here to untiled, and is
    # what every engine but the SA co-optimizer gets.
    tiling: Optional[TilingSpace] = None
    # Output host dim -> the iteration axis it cuts, for the dims where the two
    # frames provably line up. Only these dims interact with a core split.
    axis_by_host_dim: dict[int, sympy.Symbol] = field(default_factory=dict)
    _neighbours: dict[tuple, list[CoreDivision]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def axes(self) -> list[sympy.Symbol]:
        return self.context.axes

    def splits(self, division: CoreDivision) -> dict[sympy.Symbol, int]:
        """``division`` as a complete factor per axis -- what this space moves
        in, where a :class:`CoreDivision` keeps only the factors above 1."""
        return {
            axis: int(
                division.output_splits.get(axis, division.reduction_splits.get(axis, 1))
            )
            for axis in self.axes
        }

    def division(
        self, splits: dict[sympy.Symbol, int], tiling: TileSpec = UNTILED
    ) -> CoreDivision:
        """``splits`` and ``tiling`` as one :class:`CoreDivision`, without
        re-deriving the roles per call. Owes the same answer as
        :func:`_core_division`, which ``test_work_division.py`` pins over the
        candidate corpus."""
        return CoreDivision(
            output_splits={
                axis: int(factor)
                for axis, factor in splits.items()
                if factor > 1 and axis in self.output_axes
            },
            reduction_splits={
                axis: int(factor)
                for axis, factor in splits.items()
                if factor > 1 and axis not in self.output_axes
            },
            tiling=tiling,
        )

    def tile_counts(self, tiling: TileSpec) -> dict[sympy.Symbol, int]:
        """``tiling``'s split counts keyed by the iteration axis each level
        cuts -- the extent a core split on that axis then has to divide.

        Reduction levels are skipped: their ``host_dim`` indexes the reduction
        loop vars rather than the output host dims, and :meth:`tiling_options`
        never proposes one, so nothing in a generated walk reaches this with a
        reduction-tiled spec.
        """
        counts: dict[sympy.Symbol, int] = {}
        for level in tiling.axes:
            if level.is_reduction:
                continue
            axis = self.axis_by_host_dim.get(level.host_dim)
            if axis is not None:
                counts[axis] = level.count
        return counts

    def factor_domain(
        self, axis: sympy.Symbol, tiling: TileSpec = UNTILED
    ) -> list[int]:
        """The legal factors for ``axis`` under ``tiling`` -- the untiled domain
        when nothing tiles that axis, narrowed to the divisors of the per-tile
        extent when something does."""
        count = self.tile_counts(tiling).get(axis, 1)
        if count == 1:
            return self.factor_domains[axis]
        return self.context.factor_domain(axis, count)

    def admits_tiling(self, tiling: TileSpec) -> bool:
        """Whether this op may take ``tiling`` at all, splits aside. Untiled is
        the only answer for a space built without a tiling half.

        A level on an output dim :attr:`axis_by_host_dim` could not resolve is
        refused rather than allowed: unresolved means :meth:`tile_counts`
        cannot narrow the axis it cuts, and an unnarrowed axis would admit a
        core split that does not divide the per-tile extent.
        """
        if self.tiling is None:
            return tiling.is_untiled
        if any(
            not level.is_reduction and level.host_dim not in self.axis_by_host_dim
            for level in tiling.axes
        ):
            return False
        return self.tiling.admits(tiling)

    def admits(
        self, splits: dict[sympy.Symbol, int], tiling: TileSpec = UNTILED
    ) -> bool:
        """Whether this op may take ``splits`` under ``tiling``: both legal on
        every count the contexts know, jointly exact, and priceable through the
        declaration."""
        if not self.admits_tiling(tiling):
            return False
        if not self.context.is_legal(splits, self.tile_counts(tiling)):
            return False
        if self.declaration is None:
            return True
        return not undeclared_splits(self.division(splits), self.declaration)

    def tiling_options(self, tiling: TileSpec) -> list[TileSpec]:
        """The tilings one level-edit from ``tiling`` -- add or remove a level,
        change a level's count, swap two adjacent levels. Empty for a space
        with no tiling half, which is what keeps such a search's trajectory
        identical to one that never knew about tilings."""
        return [] if self.tiling is None else self.tiling.neighbours(tiling)

    def neighbours(self, division: CoreDivision) -> list[CoreDivision]:
        """The divisions one step from ``division``: for each axis, every other
        factor its domain admits at this tiling, then every tiling one level
        away at these splits.

        This is the move alphabet a generating search proposes from -- a short
        list per axis (~7 factors, measured), so there is nothing to sample over
        with a temperature-dependent scale. One *step* is one axis's factor or
        one tile level, never both: the two are chosen jointly but moved on
        separately, which is what keeps the walk local in a ragged space.
        Ordered by axis then by factor, then by the tiling space's own order,
        and memoized, since a search revisits states.
        """
        current = self.splits(division)
        tiling = division.tiling
        key = (tuple(current[axis] for axis in self.axes), tiling)
        if key not in self._neighbours:
            out = []
            for axis in self.axes:
                for factor in self.factor_domain(axis, tiling):
                    if factor == current[axis]:
                        continue
                    candidate = {**current, axis: factor}
                    if self.admits(candidate, tiling):
                        out.append(self.division(candidate, tiling))
            for spec in self.tiling_options(tiling):
                if self.admits(current, spec):
                    out.append(self.division(current, spec))
            self._neighbours[key] = out
        return self._neighbours[key]


def _axis_by_host_dim(
    op: Operation, axes: Sequence[sympy.Symbol], it_space: dict
) -> dict[int, sympy.Symbol]:
    """Output host dim -> the iteration axis it cuts.

    ``iteration_space_from_op`` keys the space by the write dep's ranges, in
    order, and a :class:`TileAxis`'s ``host_dim`` indexes ``op.data.ranges`` --
    the same extents in the same order. The extents are *checked* rather than
    assumed: a dim where the two frames do not line up is simply absent here,
    and :meth:`OpSplitSpace.admits_tiling` then refuses to tile it, since
    without the correspondence nothing would narrow that axis's core splits to
    the per-tile extent. Fails closed on a frame this has never been seen in.
    """
    ranges = list(getattr(getattr(op, "data", None), "ranges", []))
    return {
        host_dim: axis
        for host_dim, axis in enumerate(axes[: len(ranges)])
        if it_space.get(axis) == ranges[host_dim]
    }


def build_op_split_space(
    op: Operation,
    max_cores: int,
    declaration: Optional[tuple[dict, dict]] = None,
    tiling: Optional[TilingSpace] = None,
) -> Optional[OpSplitSpace]:
    """The :class:`OpSplitSpace` for ``op``, or ``None`` when it has no
    enumerable one.

    The gate is ``_enumerate_core_divisions``': an op that is not a pointwise or
    reduction ``ComputedBuffer``, or whose context cannot be derived, keeps its
    committed division instead -- so exactly the ops the menu path leaves with a
    single candidate are the ops generation has nothing to offer.

    ``tiling`` is the op's :class:`TilingSpace` when the caller wants coarse
    tilings chosen here, and ``None`` when it does not -- the second is the
    default, since only a search that generates divisions can use one and the
    enumerated menu carries no tilings to begin with.
    """
    if not isinstance(op, ComputedBuffer) or not isinstance(
        op.data, (Pointwise, Reduction)
    ):
        return None
    try:
        context = work_division_context_for_op(op, max_cores)
    except Unsupported:
        return None
    rw = op_read_writes(op)
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        return None
    axes = context.axes
    return OpSplitSpace(
        op=op,
        context=context,
        declaration=declaration,
        output_axes=frozenset(a for a in axes if write.index.coeff(a) != 0),
        factor_domains={axis: context.factor_domain(axis) for axis in axes},
        tiling=tiling,
        axis_by_host_dim=_axis_by_host_dim(op, axes, context.it_space),
    )


@dataclass
class ResidencyEdge:
    """One producer-buffer -> consumer edge, with its residency policy applied.

    Owns both halves of "can these two candidates share a residency": the
    *geometry* -- the same per-core slicing of the buffer, compared in the
    buffer's own device-dim frame, on the same total core count -- and the
    *policy* filters that decide a candidate can host a readable residency at
    all. Built once per edge by :func:`build_residency_edge`, which returns
    ``None`` for an edge excluded outright, so a caller that generates
    candidates instead of enumerating them cannot apply the geometry and forget
    the filters.

    Excluded outright (the producer then falls back to HBM, always correct): a
    producer that can never be resident, and a frame-changing (broadcasting)
    clone, whose per-core slice cannot be produced core-locally at all -- the
    single-frame view comparison misses that, and the broadcast read from HBM
    is globally correct. Excluded per candidate: see :meth:`parent_view` and
    :meth:`consumer_view`.
    """

    buf_name: str
    parent_op: Operation
    consumer_op: Operation
    write_dep: MemoryDep
    read_dep: MemoryDep
    # An SDSC carries only a matmul's primary split, so a multi-dim-split matmul
    # output cannot be coherently LX-pinned even when views match -- a consumer
    # would read per-core LX holding only a fragment. (Mirrors #2745's
    # ``get_ncores_for_buffers`` matmul guard for the greedy path.)
    parent_is_matmul: bool
    prep_cache: dict

    def parent_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """The producer's write-view under ``division``, or ``None`` when that
        candidate cannot host a readable residency: a partial-reduction write
        (output not final), an unrepresentable slicing, or a matmul output
        split across more than one device dim."""
        view, partial, repr_ok = _view_for_div(
            self.parent_op, self.write_dep, self.buf_name, division, self.prep_cache
        )
        if not repr_ok or partial:
            return None
        if self.parent_is_matmul and len(view.work_slice_dims) > 1:
            return None
        return view

    def consumer_view(self, division: CoreDivision) -> Optional[PerCoreView]:
        """The consumer's read-view under ``division``, or ``None`` when its
        slicing of the buffer is unrepresentable -- we never pin on a slicing
        we cannot verify."""
        view, _partial, repr_ok = _view_for_div(
            self.consumer_op, self.read_dep, self.buf_name, division, self.prep_cache
        )
        return view if repr_ok else None

    def compatible(
        self, parent_division: CoreDivision, consumer_division: CoreDivision
    ) -> bool:
        """Whether the two candidates induce the same per-core slicing of the
        buffer on the same total core count. Equal views alone are not enough:
        a producer on N and a consumer on M > N cores can share a slicing while
        the consumer's extra (broadcast-axis) cores hold no copy and would read
        stale LX.

        Blind to the tiling on either side, because a per-core view is a
        function of the splits alone. That is *not* the whole residency
        question once tiling is on: a consumer in another tiling group reads
        the producer's whole output, not its per-tile scratch, and needs a
        companion buffer that nothing here accounts for. Pricing that is a
        separate step; this stays the slicing predicate it has always been."""
        if parent_division.cores_used != consumer_division.cores_used:
            return False
        parent_view = self.parent_view(parent_division)
        return parent_view is not None and parent_view.same_partition(
            self.consumer_view(consumer_division)
        )

    def consumer_division_for(
        self, parent_division: CoreDivision, consumer_space: OpSplitSpace
    ) -> Optional[CoreDivision]:
        """The consumer division that reads this buffer exactly the way
        ``parent_division`` writes it, or ``None`` if the consumer cannot read
        it that way at all.

        What a search propagating a division across this edge asks instead of
        scanning the consumer's menu for a compatible entry. The inverse
        proposes and :meth:`compatible` confirms -- on this side that is a
        tautology, since the inverse only returns a division whose read-view is
        the target, but it is the same call the other direction needs and it
        keeps the policy filters in one place.
        """
        target = self.parent_view(parent_division)
        if target is None:
            return None
        return self._inverse(
            self.consumer_op, self.read_dep, target, consumer_space, parent_division
        )

    def parent_division_for(
        self, consumer_division: CoreDivision, parent_space: OpSplitSpace
    ) -> Optional[CoreDivision]:
        """The producer division that writes this buffer the way
        ``consumer_division`` reads it, or ``None``.

        The mirror of :meth:`consumer_division_for`, for a search flooding
        upward. Here the confirmation bites: the geometry is blind to the
        write-side filters, so a partial-reduction or multi-dim-split-matmul
        division can invert cleanly and still be unable to host a residency.
        """
        target = self.consumer_view(consumer_division)
        if target is None:
            return None
        return self._inverse(
            self.parent_op, self.write_dep, target, parent_space, consumer_division
        )

    def _inverse(
        self,
        op: Operation,
        dep: MemoryDep,
        target: PerCoreView,
        space: OpSplitSpace,
        other: CoreDivision,
    ) -> Optional[CoreDivision]:
        """Invert ``target`` on ``op``'s side of this edge, then confirm the
        pair through :meth:`compatible`. ``space.admits`` rides along inside the
        inversion, so a geometrically valid but illegal candidate backtracks
        rather than losing the edge.

        The tiling is *carried across*, where this side can take it: a coarse
        tiling group is a run of consecutive ops sharing one ``TileSpec``
        (``derive_tiling_groups``), so propagating it along the residency
        relation is what forms a group at all -- flooding a region and leaving
        every op in it a different tiling would produce no group. It is only an
        attempt, though: the tiling narrows this side's split domains, so if it
        costs the edge the untiled inverse is taken instead. Losing a tiling
        level is a worse plan; losing the edge is a worse *state*.
        """
        prep = _prep_for(op, dep, self.buf_name, self.prep_cache)
        wanted = other.tiling if space.admits_tiling(other.tiling) else UNTILED
        attempts = (wanted,) if wanted.is_untiled else (wanted, UNTILED)
        for tiling in attempts:

            def accept(candidate: dict, tiling: TileSpec = tiling) -> bool:
                """``space.admits`` at the tiling this attempt is carrying.
                Bound as a default so the closure cannot pick up a later
                iteration's value."""
                return space.admits(candidate, tiling)

            splits = invert_per_core_view(
                prep, target, space.factor_domains, accept=accept
            )
            if splits is None:
                continue
            division = space.division(splits, tiling)
            is_parent_side = op is self.parent_op
            parent, consumer = (
                (division, other) if is_parent_side else (other, division)
            )
            if self.compatible(parent, consumer):
                return division
        return None

    def match_pairs(
        self,
        parent_divisions: Sequence[CoreDivision],
        consumer_divisions: Sequence[CoreDivision],
    ) -> list[tuple[int, int]]:
        """Compatible ``(parent index, consumer index)`` pairs, with each side's
        view computed once per candidate rather than once per pair."""
        parent_views = [self.parent_view(cd) for cd in parent_divisions]
        consumer_views = [self.consumer_view(cd) for cd in consumer_divisions]
        return [
            (i, j)
            for i, parent_view in enumerate(parent_views)
            if parent_view is not None
            for j, consumer_view in enumerate(consumer_views)
            if consumer_view is not None
            and parent_view.same_partition(consumer_view)
            and parent_divisions[i].cores_used == consumer_divisions[j].cores_used
        ]


def build_residency_edge(
    buf_name: str,
    parent_op: Operation,
    consumer_op: Operation,
    consumer_reads: Iterable[Dep],
    residency_reason: Optional[str],
    prep_cache: dict,
) -> Optional[ResidencyEdge]:
    """The :class:`ResidencyEdge` for this producer-consumer pair, or ``None``
    when the edge can never host a residency."""
    if residency_reason is not None:
        return None
    if _is_frame_changing_clone(parent_op, buf_name):
        return None
    write_dep = next(
        (
            w
            for w in op_read_writes(parent_op).writes
            if w.name == buf_name and hasattr(w, "index")
        ),
        None,
    )
    read_dep = next(
        (r for r in consumer_reads if r.name == buf_name and hasattr(r, "index")),
        None,
    )
    if write_dep is None or read_dep is None:
        return None
    return ResidencyEdge(
        buf_name=buf_name,
        parent_op=parent_op,
        consumer_op=consumer_op,
        write_dep=write_dep,
        read_dep=read_dep,
        parent_is_matmul=_is_matmul_op(parent_op),
        prep_cache=prep_cache,
    )
