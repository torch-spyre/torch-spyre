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
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision
from torch_spyre._inductor.work_division import (
    WorkDivisionContext,
    work_division_context_for_op,
)


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
    """

    op: Operation
    context: WorkDivisionContext
    # The buffer's ``sym_core_divs``; ``None`` for a caller with no cost
    # expression to price against, which then constrains nothing.
    declaration: Optional[tuple[dict, dict]]
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
        return {
            axis: int(
                division.output_splits.get(axis, division.reduction_splits.get(axis, 1))
            )
            for axis in self.axes
        }

    def division(self, splits: dict[sympy.Symbol, int]) -> CoreDivision:
        """``splits`` as a :class:`CoreDivision`, without re-deriving the roles
        per call. Owes the same answer as :func:`_core_division`, which
        ``test_work_division.py`` pins over the candidate corpus."""
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
    declaration: Optional[tuple[dict, dict]] = None,
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
        stale LX."""
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
        rather than losing the edge."""
        splits = invert_per_core_view(
            _prep_for(op, dep, self.buf_name, self.prep_cache),
            target,
            space.factor_domains,
            accept=space.admits,
        )
        if splits is None:
            return None
        division = space.division(splits)
        is_parent_side = op is self.parent_op
        parent, consumer = (division, other) if is_parent_side else (other, division)
        return division if self.compatible(parent, consumer) else None

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
