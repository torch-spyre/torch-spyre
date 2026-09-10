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
  the axes, each axis's legal factors, and whether a proposed split is legal --
  so a caller can walk the space instead of materializing it, and
  :meth:`OpSplitSpace.neighbours` is the move alphabet that walk proposes from;
* :class:`ResidencyEdge` owns one producer-buffer -> consumer edge, both the
  geometry (does this pair of candidates slice the buffer identically) and the
  policy filters that decide a candidate can host a readable residency at all;
  :meth:`ResidencyEdge.consumer_division_for` and its mirror *construct* the
  other end's division rather than looking it up.

``allocator.py`` materializes the edge relation as the ``cd_parent_matches``
pair table every engine consumes today. Nothing here decides *which* candidate
to take: the space and the edge are pure oracles, so a search owns its own
proposal distribution and its own randomness.
"""

import math
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
    op_read_writes,
    _per_core_view_from_prep,
    _prepare_per_core_view,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision
from torch_spyre._inductor.work_division import (
    WorkDivisionContext,
    work_division_context_for_op,
)


def _reduction_syms(
    op: Operation, splits: dict[sympy.Symbol, int]
) -> frozenset[sympy.Symbol]:
    """Get reduction symbols for an operation."""
    rw = op_read_writes(op)
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        return frozenset()
    return frozenset(s for s in splits if write.index.coeff(s) == 0)


def _core_division(op: Operation, splits: dict[sympy.Symbol, int]) -> CoreDivision:
    """Classify one symbol-keyed candidate for its producing operation."""
    sparse = {s: v for s, v in splits.items() if v > 1}
    return CoreDivision(splits=sparse, reduction_syms=_reduction_syms(op, sparse))


def _view_for_div(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    splits: dict[sympy.Symbol, int],
    prep_cache: dict,
):
    """One candidate division's per-core view of ``buf_name``.

    ``prep_cache`` holds the candidate-invariant (sympy-heavy) context, keyed by
    ``(op name, dep, buf_name)``: a producer's write-dep and a consumer's
    read-dep on the same buffer can be equal ``MemoryDep``s, so the op name
    keeps their preps distinct while a parent read by several consumers reuses
    its write-view prep.
    """
    syms = _reduction_syms(op, splits)
    return _per_core_view_from_prep(
        _prep_for(op, dep, buf_name, prep_cache),
        splits,
        {k: v for k, v in splits.items() if k in syms},
    )


def _prep_for(op: Operation, dep: MemoryDep, buf_name: str, prep_cache: dict):
    """The candidate-invariant view prep for one ``(op, dep, buf_name)``."""
    key = (op.get_name(), dep, buf_name)
    if key not in prep_cache:
        prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
    return prep_cache[key]


def undeclared_splits(division: CoreDivision, sym_core_divs: dict) -> set:
    """The split keys of ``division`` that ``sym_core_divs`` declares no symbol
    for.

    A solver prices a division through the symbols its buffer declares -- one
    per stride coefficient seen across an enumerated menu. A split on an axis
    outside that declaration is priced at the symbol's default of ``1`` rather
    than rejected, so it is a wrong answer and not a failure. Empty is the
    contract; a generator has to be held to it, which is why generation stays
    inside the declaration rather than widening it.
    """
    return set(division.splits) - set(sym_core_divs)


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
    """

    op: Operation
    context: WorkDivisionContext
    # The buffer's ``sym_core_divs``; ``None`` for a caller with no cost
    # expression to price against, which then constrains nothing.
    declaration: Optional[dict]
    # Axes whose factor slices the op's output (the rest are reduction axes).
    output_axes: frozenset
    factor_domains: dict[sympy.Symbol, list[int]]
    _neighbours: dict[tuple, list[CoreDivision]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def axes(self) -> list[sympy.Symbol]:
        return self.context.axes

    def splits(self, division: CoreDivision) -> dict[sympy.Symbol, int]:
        """``division`` as a complete factor per axis -- what this space moves
        in, where a :class:`CoreDivision` keeps only the factors above 1."""
        return {axis: int(division.splits.get(axis, 1)) for axis in self.axes}

    def division(self, splits: dict[sympy.Symbol, int]) -> CoreDivision:
        """``splits`` as a :class:`CoreDivision`, without re-deriving the roles
        per call. Owes the same answer as :func:`_core_division`, which
        ``test_work_division.py`` pins over the candidate corpus."""
        sparse = {axis: int(factor) for axis, factor in splits.items() if factor > 1}
        return CoreDivision(
            splits=sparse,
            reduction_syms=frozenset(
                axis for axis in sparse if axis not in self.output_axes
            ),
        )

    def admits(self, splits: dict[sympy.Symbol, int]) -> bool:
        """Whether this op may take ``splits``: legal on every count the
        context knows, and priceable through the declaration."""
        if not self.context.is_legal(splits):
            return False
        if self.declaration is None:
            return True
        return not undeclared_splits(self.division(splits), self.declaration)

    def neighbours(self, division: CoreDivision) -> list[CoreDivision]:
        """The divisions one axis away from ``division``: for each axis, every
        other factor its domain admits, keeping only the legal results.

        This is the move alphabet a generating search proposes from -- a short
        list per axis (~7 factors, measured), so there is nothing to sample over
        with a temperature-dependent scale. Ordered by axis then by factor, and
        memoized, since a search revisits states.
        """
        current = self.splits(division)
        key = tuple(current[axis] for axis in self.axes)
        if key not in self._neighbours:
            out = []
            for axis in self.axes:
                for factor in self.factor_domains[axis]:
                    if factor == current[axis]:
                        continue
                    candidate = {**current, axis: factor}
                    if self.admits(candidate):
                        out.append(self.division(candidate))
            self._neighbours[key] = out
        return self._neighbours[key]


def build_op_split_space(
    op: Operation,
    max_cores: int,
    declaration: Optional[dict] = None,
) -> Optional[OpSplitSpace]:
    """The :class:`OpSplitSpace` for ``op``, or ``None`` when it has no
    enumerable one.

    The gate is ``_enumerate_core_divisions``': an op that is not a pointwise or
    reduction ``ComputedBuffer``, or whose context cannot be derived, keeps its
    committed division instead -- so exactly the ops the menu path leaves with a
    single candidate are the ops generation has nothing to offer.
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

    A producer rejected for LX is excluded outright. Otherwise, check each
    producer-consumer edge independently. A broadcasting clone may read its
    input from HBM and still keep its completed output in LX for a matching
    consumer. Candidate-specific checks are in :meth:`parent_view` and
    :meth:`consumer_view`.
    """

    buf_name: str
    parent_op: Operation
    consumer_op: Operation
    write_dep: MemoryDep
    read_dep: MemoryDep
    prep_cache: dict

    def parent_view(self, splits: dict[sympy.Symbol, int]) -> Optional[PerCoreView]:
        """The producer's write-view under ``division``, or ``None`` when that
        candidate cannot host a readable residency: a partial-reduction write
        (output not final) or an unrepresentable slicing. Matching compares
        the complete per-core views, including all split dimensions."""
        view, partial, repr_ok = _view_for_div(
            self.parent_op, self.write_dep, self.buf_name, splits, self.prep_cache
        )
        if not repr_ok or partial:
            return None
        return view

    def consumer_view(self, splits: dict[sympy.Symbol, int]) -> Optional[PerCoreView]:
        """The consumer's read-view under ``division``, or ``None`` when its
        slicing of the buffer is unrepresentable -- we never pin on a slicing
        we cannot verify."""
        view, _partial, repr_ok = _view_for_div(
            self.consumer_op, self.read_dep, self.buf_name, splits, self.prep_cache
        )
        return view if repr_ok else None

    @staticmethod
    def _cores_used(splits: dict[sympy.Symbol, int]) -> int:
        return math.prod(splits.values())

    def compatible(
        self,
        parent_splits: dict[sympy.Symbol, int],
        consumer_splits: dict[sympy.Symbol, int],
    ) -> bool:
        """Whether the two candidates induce the same per-core slicing of the
        buffer on the same total core count. Equal views alone are not enough:
        a producer on N and a consumer on M > N cores can share a slicing while
        the consumer's extra (broadcast-axis) cores hold no copy and would read
        stale LX.

        :meth:`match_pairs` answers this over two menus and caches each side's
        view across the cross product; this is the single-pair form, for a
        caller holding one candidate per side rather than a list.
        """
        if self._cores_used(parent_splits) != self._cores_used(consumer_splits):
            return False
        parent_view = self.parent_view(parent_splits)
        return parent_view is not None and parent_view == self.consumer_view(
            consumer_splits
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
        target = self.parent_view(parent_division.splits)
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
        upward. This is the side with write-side policy to apply -- a
        partial-reduction or multi-dim-split-matmul division inverts cleanly and
        still cannot host a residency -- so :meth:`_inverse` applies it inside
        the search rather than on the answer.
        """
        target = self.consumer_view(consumer_division.splits)
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
        pair through :meth:`compatible`.

        Everything that can reject a candidate rides along inside the inversion,
        so a geometrically valid one the policy turns down backtracks to the
        next rather than losing the edge. That is ``space.admits`` and, on the
        producer's side, :meth:`parent_view` -- a partial-reduction write or a
        multi-dim-split matmul output is invisible to the geometry, and the
        first solution the geometry offers is regularly one of those (two
        symbols on one device dim, where meeting ``target.num_cores`` forces a
        reduction factor above 1 under one placement and not under the next).
        Applying them afterwards instead cost the edge outright, and
        ``_ViewRelation`` memoizes that ``None`` for the whole solve.

        The trailing :meth:`compatible` is then a confirmation rather than a
        filter: it re-asks the same question of the pair as a whole, which keeps
        the "propose, then confirm" shape honest on both sides of the edge.
        """
        is_parent_side = op is self.parent_op

        def accept(splits: dict) -> bool:
            if not space.admits(splits):
                return False
            # Geometry-blind, side-specific policy. The consumer's side has none
            # -- an unrepresentable read cannot reproduce ``target`` anyway, so
            # the forward-map confirmation already covers it.
            if not is_parent_side:
                return True
            return self.parent_view(splits) is not None

        splits = invert_per_core_view(
            _prep_for(op, dep, self.buf_name, self.prep_cache),
            target,
            space.factor_domains,
            accept=accept,
        )
        if splits is None:
            return None
        division = space.division(splits)
        parent, consumer = (
            (splits, other.splits) if is_parent_side else (other.splits, splits)
        )
        return division if self.compatible(parent, consumer) else None

    def match_pairs(
        self,
        parent_divisions: Sequence[dict[sympy.Symbol, int]],
        consumer_divisions: Sequence[dict[sympy.Symbol, int]],
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
            and parent_view == consumer_view
            and self._cores_used(parent_divisions[i])
            == self._cores_used(consumer_divisions[j])
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
    write_dep = next(
        (
            w
            for w in op_read_writes(parent_op).writes
            if w.name == buf_name and isinstance(w, MemoryDep)
        ),
        None,
    )

    def wrapped_hasattr(obj, attr):
        try:
            return hasattr(obj, attr)
        except NotImplementedError:
            return False

    read_dep = next(
        (r for r in consumer_reads if r.name == buf_name and isinstance(r, MemoryDep)),
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
        prep_cache=prep_cache,
    )
