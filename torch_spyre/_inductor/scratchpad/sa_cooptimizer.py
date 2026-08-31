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

"""Joint work-division + LX-layout simulated-annealing engine.

``SaCoOptimizingSolver`` is a third co-optimization engine alongside the
substrate's CP-SAT and DFS solvers. It anneals the joint state ``(pi, W)``:

* ``pi`` -- the layout permutation, held in a *composed* (not subclassed)
  :class:`PermutationBasedLayoutSolver` packer, because this loop mixes move
  types and scores a richer objective than the packer's own ``quality()``.
* ``W`` -- the work division, one ``chosen_division`` menu index per buffer.

Moves are reorder, atomic division flip, and region-recolor; each structural
move runs as a compound move+burst judged as a unit by one Metropolis test.
Region-recolor floods the ``cd_parent_matches`` relation bidirectionally from a
non-trivial (split) anchor tiling, so the region *is* the flood's reach and
boundaries emerge for free; an edge with no compatible index becomes an accepted
internal seam.

Best-seen over ``(pi, W)`` from the seed state (every op at index 0, ``pi`` from
FirstFit) keeps every returned state no worse than that baseline.

Determinism: a seeded ``Random`` over index-ordered domains and the integer
fixed-point score make a run bit-for-bit reproducible.

Design notes: ``docs/source/compiler/sa_co_optimization.md``.
"""

from __future__ import annotations

import copy
import heapq
import math
import random as rnd
import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import SolverToPermutation
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    ceil_div,
)
from torch_spyre._C import NativePermutationLayoutSolver
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    make_permutation_packer,
)
from torch_spyre._inductor.scratchpad import utils
from torch_spyre._inductor.cost_model import OpFeatures
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import iteration_space_from_op

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

logger = get_inductor_logger("scratchpad.sa_cooptimizer")

# RNG seed; fixes the (deterministic) search trajectory.
_SEED = 0

# Step budget: clamp(_STEPS_PER_BUFFER * n, _MIN_STEPS, _MAX_STEPS). The ceiling
# sits above the layout-only annealer's clamp (``SelfCalibratingReheatingSchedule
# .max_steps``, 5_000) since this engine searches divisions too, and binds only
# well past the validated corpus. It bounds *steps*, not wall-clock.
_STEPS_PER_BUFFER = 40
_MIN_STEPS = 200
_MAX_STEPS = 15_000

# Fixed proposal weights over the three move types. Reorder's weight is
# effectively 0 while every eligible buffer is resident (see
# :meth:`_applicable_moves`).
_MOVE_WEIGHTS = {"reorder": 0.5, "flip": 0.3, "recolor": 0.2}

# Layout-burst length as a fraction of the buffer count. The burst warms ``pi`` to
# the new footprints before a compound structural move is judged.
_BURST_FRACTION = 0.1

# The geometric cool spans t0 down to t0 / _COOLING_SPAN.
_COOLING_SPAN = 1000.0

# ``make_permutation_packer`` returns either the pure-Python or the native C++
# packer. Use ``.quality()`` (not the Python-only ``total_quality`` attribute) so
# both work.
Packer = Union[PermutationBasedLayoutSolver, NativePermutationLayoutSolver]

# Cause recorded for a buffer the SA engine left out of LX.
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"


def _work_slices(op, division: "CoreDivision") -> dict:
    """Restore a complete symbol-keyed split map from a sparse candidate."""
    return {
        symbol: (
            division.output_splits.get(symbol, division.reduction_splits.get(symbol, 1))
        )
        for symbol in iteration_space_from_op(op)
    }


def features_for_division(op, division: "CoreDivision") -> Optional[OpFeatures]:
    """``OpFeatures`` for ``op`` as if it were divided per ``division``.

    Returns ``None`` when the op cannot be featurized (the extractor is
    best-effort and swallows its own failures, so a ``None`` here means the op
    itself was rejected, not that the division was bad).

    The candidate is passed as a complete symbol-keyed map, leaving the live
    operation and its Scheduler-boundary transport untouched.
    """
    from torch_spyre._inductor.dump_cost_model import extract_op_features

    try:
        return extract_op_features(op, _work_slices(op, division))
    except Exception:  # noqa: BLE001 - featurization is best-effort by design
        logger.debug("could not featurize op for a candidate division", exc_info=True)
        return None


def features_for_menu(op, divisions) -> list[Optional[OpFeatures]]:
    """``features_for_division`` over a buffer's whole candidate menu, index for
    index with ``divisions`` so a menu index selects its features directly."""
    return [features_for_division(op, cd) for cd in divisions]


def _bundle_objective(
    buffers: Sequence["CoreDivisionBuffer"],
) -> Optional["BundleCostObjective"]:
    """Build a :class:`BundleCostObjective` from the live Inductor graph.

    Per-division ``OpFeatures`` and the fused-bundle grouping are read off
    ``V.graph``, which is ambient while the allocator runs -- this solver *is* an
    Inductor pass. Only the buffer order comes from the arguments.

    Returns ``None`` when there is no live graph, in which case the caller falls
    back to the memory-only objective and logs it.
    """
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.fusion import estimate_bundles
    from torch_spyre._inductor.scratchpad.cost_objective import BundleCostObjective

    # Unset, ``V.graph`` is a ``NullHandler`` rather than ``None``, so detect the
    # live graph by what this needs from it rather than by identity.
    graph: Any = getattr(V, "graph", None)
    if graph is None:
        return None
    if not hasattr(graph, "operations") or not hasattr(graph, "get_buffer"):
        return None

    features: dict[str, list] = {}
    for buf in buffers:
        try:
            op = graph.get_buffer(buf.name)
        except Exception:  # noqa: BLE001 - not every solver buffer is a graph buffer
            continue
        features[buf.name] = features_for_menu(op, buf.core_divisions)
    bundles = [
        [op.get_name() for op in group] for group in estimate_bundles(graph.operations)
    ]
    return BundleCostObjective([b.name for b in buffers], features, bundles)


class SaCoOptimizingSolver(CoreDivisionLayoutSolver):
    """SA joint core-division + LX-placement engine.

    The search is fully determined by the module constants above; there is
    nothing to configure per call.

    Args:
        buffers: the buffers to plan, in the allocator's order. Declared as
            ``Sequence[LifetimeBoundBuffer]`` so the class itself satisfies
            ``CoreDivisionSolverFactory`` (``Callable`` parameters are
            contravariant, so a narrower annotation would not), but every buffer
            passed must be a :class:`CoreDivisionBuffer` -- the engine reads the
            ``core_divisions`` menu and the ``cd_parent_matches`` relation off
            each one.

            **Mutated in place, and their order is an index.** The returned list
            is these same objects with ``chosen_division`` and ``address``
            written back, so a caller needing the input preserved must copy
            first. Position ``i`` is the index used by ``chosen``, by the packer's
            permutation, and by the cost objective. Solvers are single-use:
            construct a fresh one per buffer set.
        size: scratchpad capacity in bytes.
        alignment: placement alignment (128 = one Spyre stick).
    """

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
    ) -> None:
        super().__init__(buffers, size, alignment)
        # Narrowed from the contravariant parameter type (see the ``buffers``
        # arg). Same objects as the base's ``self.buffers``, so write-back
        # through either name is visible in both.
        self._bufs: Sequence[CoreDivisionBuffer] = cast(
            "list[CoreDivisionBuffer]", list(buffers)
        )
        # The cost model prices compute as well as traffic, so it replaces the
        # memory-only objective outright. It builds itself from the live graph,
        # which is what lets it be unconditional: the allocator's
        # ``CoreDivisionSolverFactory`` passes only (buffers, size, alignment), so
        # an objective could never arrive from the caller.
        self._cost_objective = _bundle_objective(self._bufs)
        if self._cost_objective is None:
            logger.info(
                "no live Inductor graph; falling back to the memory-only objective"
            )
        # Best-seen over the anneal (set in _anneal, read in _step); declared for
        # the types.
        self._best_score: int
        self._best_snap: tuple[Packer, list[int], int]
        # Number of buffers passing :meth:`_eligible` under the live ``W``. Kept
        # as a count, not a mask: the two ripple sites already evaluate
        # ``_eligible`` over the buffers a move can change, so they carry the
        # count by differencing that set before and after.
        self._n_eligible: int

    # -- public interface ----------------------------------------------------

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """Not supported: this engine is joint-only. :class:`MemoryPlanSolver`
        declares it abstract, but placement-only annealing belongs to the
        standalone layout-only annealer, and ``CoOptimizingAllocator`` only ever
        calls :meth:`plan_layout_and_core_divisions`."""
        raise NotImplementedError(
            "SaCoOptimizingSolver is a joint core-division + placement engine; "
            "use plan_layout_and_core_divisions, or "
            "SimulatedAnnealingLayoutSolver for placement-only annealing."
        )

    def plan_layout_and_core_divisions(
        self, cost_expr=None
    ) -> list[CoreDivisionBuffer]:
        """Anneal the joint ``(pi, W)`` state and write ``chosen_division`` /
        ``address`` back to each buffer; populate ``spill_reasons``. Returns the
        solver's own buffers. Single-use: construct a fresh solver per set."""
        # TODO: use cost_expr here -- states are scored by self._cost_objective.
        self.spill_reasons = {}
        n = len(self._bufs)
        if n == 0:
            return list(self._bufs)

        self._rng = rnd.Random(_SEED)
        self._precompute_topology()

        # Seed: every op at the committed division (index 0); pi from FirstFit.
        self.chosen = [0] * n
        self.packer = self._build_seed_packer()

        self._anneal()
        self._write_back()
        return list(self._bufs)

    # -- static topology (division-invariant) --------------------------------

    def _assert_unsized_buffers_are_pinned(self) -> None:
        """Assert every unsized buffer carries a ``residency_reason``.

        An unsized buffer carries the ``-1`` ``mem_usage`` sentinel
        ``mem_usage_by_buf`` (``utils.py``) emits when it cannot size a buffer.
        :meth:`_per_core_size` clamps that to ``0``, which passes
        :meth:`_eligible`'s capacity gate, so such a buffer reaching the search
        would be placed occupying no space and the buffer above it would land on
        the same address -- a wrong layout, not a crash.

        What prevents it is a coupling across three files: ``mem_usage_by_buf``
        emits ``-1`` on exactly the conditions ``_op_output_good_for_lx_reuse``
        (``allocator.py``) refuses, so the allocator pins every such buffer and
        the pin gate rejects it first. Nothing in the search re-derives that, so
        assert it rather than depend on the three staying in lockstep.
        """
        for b in self._bufs:
            assert b.size >= 0 or b.residency_reason is not None, (
                f"buffer {b.name} is unsized (size={b.size}) but carries no "
                "residency_reason, so nothing gates it out of LX residency; its "
                "per-core footprint would clamp to 0 and the buffer placed above "
                "it would land on the same address"
            )

    def _precompute_topology(self) -> None:
        """Precompute the division-invariant graph structure used every step:
        the name->index map, each buffer's parent indices, and -- keyed by parent
        index -- its children with the ``(parent_div, child_div)`` pairs that keep
        that edge tiling-compatible.

        No consumer *count* is derived here: :meth:`_spill_cost` scales by
        reads-served instead. ``_children`` remains available for the cohort
        multiplicity when op metadata is wired in.
        """
        self._assert_unsized_buffers_are_pinned()
        bufs = self._bufs
        self._name_to_idx = {b.name: i for i, b in enumerate(bufs)}
        n = len(bufs)
        self._parents_idx: list[set[int]] = [set() for _ in range(n)]
        # parent_idx -> list of (child_idx, frozenset of compatible (p_idx, c_idx))
        self._children: list[list[tuple[int, frozenset]]] = [[] for _ in range(n)]
        foreign_parents = 0
        for c_idx, c in enumerate(bufs):
            for p_name in c.parents:
                # A parent outside the solver's set is skipped, not asserted:
                # ``_build_cd_bound_buffers`` assigns ``parents`` unfiltered, so
                # graph inputs, constants and extern outputs appear here. The edge
                # only gates a child's division against reading the parent from
                # LX, and a buffer the solver does not own is never LX-resident.
                p_idx = self._name_to_idx.get(p_name)
                if p_idx is None:
                    foreign_parents += 1
                    continue
                self._parents_idx[c_idx].add(p_idx)
                pairs = frozenset(
                    (int(a), int(b)) for a, b in c.cd_parent_matches.get(p_name, [])
                )
                self._children[p_idx].append((c_idx, pairs))
        if foreign_parents:
            logger.debug(
                "dropped %d parent edge(s) naming buffers outside the solver's "
                "set (graph inputs / constants / externs)",
                foreign_parents,
            )

        # Region-recolor support. ``_edge_pairs[(p, c)]`` is the compatible
        # ``(p_div, c_div)`` set on the edge p->c; ``_children_idx`` lists each
        # op's children by index (deterministic flood order).
        self._children_idx = [sorted(c for c, _ in self._children[i]) for i in range(n)]
        self._edge_pairs: dict[tuple[int, int], frozenset] = {
            (i, c): pairs for i in range(n) for c, pairs in self._children[i]
        }
        # Non-trivial (split) menu indices per op -- the only legal recolor
        # anchors, so recolor stays a coordinated *splitting* move and leaves
        # undividing to atomic flips.
        self._nontrivial_menu = [
            sorted(
                j for j, cd in enumerate(b.core_divisions) if cd.output_partition > 1
            )
            for b in bufs
        ]
        self._anchor_candidates = [i for i in range(n) if self._nontrivial_menu[i]]
        self._precompute_spill_costs()

    def _precompute_spill_costs(self) -> None:
        """Cache the loop-invariant inputs to :meth:`_score`. A move changes only
        the *per-core* footprint the packer sees, never a buffer's total size, so
        neither the spill costs nor the bandwidth constant can move."""
        self._spill_costs = [self._spill_cost(b) for b in self._bufs]
        self._hbm_bytes_per_us = utils.hbm_bytes_per_us()

    # -- division-dependent derivations --------------------------------------

    def _per_core_size(self, idx: int, div_idx: int) -> int:
        """Per-core footprint of buffer ``idx`` under menu index ``div_idx``:
        ``ceil_div(total_size, output_partition)``, using the substrate's integer
        helper so this rounds identically to every other footprint-division site.

        Clamped non-negative so the packer never sees a negative size from the
        ``mem_usage`` ``-1`` sentinel; what stops an unsized buffer from looking
        *placeable* at zero footprint is
        :meth:`_assert_unsized_buffers_are_pinned`."""
        part = self._bufs[idx].core_divisions[div_idx].output_partition
        return max(0, ceil_div(self._bufs[idx].size, part))

    def _eligible(self, idx: int) -> bool:
        """Whether buffer ``idx`` may be LX-resident under the current ``W``
        (the three division-dependent gates, mirroring
        ``DfsLayoutSolver._evaluate``): the fixed residency pin, a per-core
        footprint that fits at all, and a division carrying a compatible
        ``cd_parent_matches`` pair on *every* child edge.

        Those pairs are per-core-view based, not ``is_clean`` based: a reduction
        split can appear as a *consumer* index (a K-split reading a clean parent
        via the PSUM ring) but never as a parent index, since a reduction-split
        producer writes a partial sum no child may read from LX -- so such a
        producer is always gated out here."""
        b = self._bufs[idx]
        # Not ``MemoryPlanSolver.excluded()``: that folds in a ``min_footprint >
        # limit`` test, which is division-dependent and is the next gate down.
        if b.residency_reason is not None:
            return False
        if self._per_core_size(idx, self.chosen[idx]) > self.limit:
            return False
        ci = self.chosen[idx]
        return all(
            (ci, self.chosen[c_idx]) in pairs for c_idx, pairs in self._children[idx]
        )

    def _all_eligible_resident(self) -> bool:
        """Whether every eligible buffer holds an address, i.e. nothing the solver
        could place is spilled. O(1): an ineligible buffer never has an address,
        so ``count_allocated()`` reaches ``_n_eligible`` exactly then."""
        return self.packer.count_allocated() == self._n_eligible

    # -- seed ----------------------------------------------------------------

    def _lifetime_buffers(self, sizes: list[int]) -> list[LifetimeBoundBuffer]:
        """Plain lifetime buffers the packer and FirstFit consume; ``sizes`` are
        the current per-core footprints.

        ``residency_reason`` is carried so ``MemoryPlanSolver.excluded()`` sees the
        fixed pins during the FirstFit seed pass; the packer ignores it, taking an
        explicit ``eligible`` mask instead.
        """
        out = []
        for i, b in enumerate(self._bufs):
            out.append(
                LifetimeBoundBuffer(
                    name=b.name,
                    size=sizes[i],
                    uses=list(b.uses),
                    first_use_is_read=b.first_use_is_read,
                    in_place_parents=[
                        p for p in b.in_place_parents if p in self._name_to_idx
                    ],
                    residency_reason=b.residency_reason,
                )
            )
        return out

    def _build_seed_packer(self) -> Packer:
        """Build the packer for the seed state: per-core sizes at index 0, a
        FirstFit-derived ``pi``, and the seed eligibility mask."""
        n = len(self._bufs)
        sizes = [self._per_core_size(i, 0) for i in range(n)]
        eligible = [self._eligible(i) for i in range(n)]
        self._n_eligible = sum(eligible)

        # pi from a FirstFit pass over the per-core sizes. FirstFit leaves the
        # fixed pins unplaced and ``SolverToPermutation`` sorts them after every
        # placed buffer, so they stop displacing eligible buffers upward. They keep
        # a slot, so pi stays a permutation of all n indices and lines up
        # index-for-index with the packer's ``eligible`` mask. Transient,
        # division-dependent ineligibility is deliberately *not* expressed here: it
        # must keep its slot so it can re-enter coherently.
        ff_bufs = self._lifetime_buffers(sizes)
        # Deep-copied so FirstFit lays out its own objects, never the ones the
        # solver mutates; SolverToPermutation reads addresses back by name.
        pi = SolverToPermutation(
            FirstFitLayoutSolver(copy.deepcopy(ff_bufs), self.limit, self.alignment)
        ).permutation(ff_bufs)

        return make_permutation_packer(
            self._lifetime_buffers(sizes),
            pi,
            self.limit,
            self.alignment,
            eligible=eligible,
        )

    # -- scoring (lower is better) -------------------------------------------

    @staticmethod
    def _spill_cost(buffer: CoreDivisionBuffer) -> int:
        """Differential HBM traffic a spill adds over residency, in bytes.

        Duplicates :meth:`_LifetimeBufferWithCpVars.spill_cost` in
        ``ilp_solver_ortools.py`` so the two engines score the same quantity;
        lifting the formula into ``plan_solver.py`` is a follow-up.

        The reads residency would have served from LX, plus the producer's write,
        which residency turns into a free LX write -- a graph input has no producer
        write to save and a graph output's write-out is unavoidable either way, so
        both cancel, exactly ``boundary != Intermediate``. The
        ``first_use_is_read`` discount drops an input's first read, the clone-in
        that pinning cannot avoid; a computed buffer's first use is the producing
        write, which ``read_count`` already excludes.
        """
        is_intermediate = buffer.boundary == BufferType.Intermediate
        reads_served = buffer.read_count - (1 if buffer.first_use_is_read else 0)
        return (reads_served + (1 if is_intermediate else 0)) * max(0, buffer.size)

    def _score(self) -> int:
        """The shared objective for the current state, in integer fixed-point
        time units. A buffer with a packer address is LX-resident (its address is
        ``None`` iff ineligible or spilled).

        Hot path: reads ``packer.addresses`` **once**. The native packer
        materializes a fresh list per ``addresses`` access, so a per-buffer read
        inside the loop was quadratic; hoisting it is 8-31x faster on the captures.

        The memory-only fallback is *differential* -- ``spill_cost`` is the traffic
        a spill adds **over** residency -- so a resident buffer contributes zero
        and only spilled ones are summed, the same shape as the CP-SAT engine's
        ``spill_cost() * (1 - in_buffer)``.
        """
        addresses = self.packer.addresses
        if self._cost_objective is not None:
            resident = frozenset(
                b.name
                for b, address in zip(self._bufs, addresses)
                if address is not None
            )
            return self._cost_objective.score(self.chosen, resident)

        traffic = sum(
            cost
            for cost, address in zip(self._spill_costs, addresses)
            if address is None
        )
        return utils.to_fixed_us(traffic / self._hbm_bytes_per_us)

    # -- moves ---------------------------------------------------------------

    def _flippable(self) -> list[int]:
        """Buffer indices whose division menu offers an alternative (>1 entry)."""
        return [
            i for i in range(len(self._bufs)) if len(self._bufs[i].core_divisions) > 1
        ]

    def _atomic_flip(self, idx: int, new_div: int) -> None:
        """Change buffer ``idx``'s division to ``new_div`` and ripple: resize its
        per-core footprint, then refresh eligibility for ``idx`` and its parents.
        Those are the only buffers a flip can change, since eligibility depends on
        an op's own division and its children's."""
        affected = sorted({idx} | self._parents_idx[idx])
        before = sum(self._eligible(x) for x in affected)
        self.chosen[idx] = new_div
        self.packer.resize(idx, self._per_core_size(idx, new_div))
        after = 0
        for x in affected:
            flag = self._eligible(x)
            after += flag
            self.packer.set_eligible(x, flag)
        self._n_eligible += after - before

    def _flood_region(self, anchor: int, tiling: int) -> dict[int, int]:
        """Flood the ``cd_parent_matches`` relation from ``(anchor, tiling)`` to a
        menu-index assignment over the reachable region.

        Bidirectional: from an assigned op ``u`` (index ``iu``), a child ``c`` joins
        at the smallest ``ic`` with ``(iu, ic)`` compatible, and a parent ``p`` at
        the smallest ``ip`` with ``(ip, iu)`` compatible. The reachable set *is* the
        region; an edge with no compatible index is simply not extended across --
        an accepted internal seam, never a failure.

        First-assignment-wins with a min-index frontier and sorted candidates makes
        this independent of ``cd_parent_matches`` list order.
        """
        assignment = {anchor: tiling}
        heap = [anchor]
        while heap:
            u = heapq.heappop(heap)
            iu = assignment[u]
            for c in self._children_idx[u]:  # down: u -> c
                if c in assignment:
                    continue
                cands = sorted(ic for ip, ic in self._edge_pairs[(u, c)] if ip == iu)
                if cands:
                    assignment[c] = cands[0]
                    heapq.heappush(heap, c)
            for p in sorted(self._parents_idx[u]):  # up: p -> u
                if p in assignment:
                    continue
                cands = sorted(ip for ip, ic in self._edge_pairs[(p, u)] if ic == iu)
                if cands:
                    assignment[p] = cands[0]
                    heapq.heappush(heap, p)
        return assignment

    def _apply_recolor(self, assignment: dict[int, int]) -> None:
        """Commit a flooded region coloring: set every region op's division, resize
        its footprint, and refresh eligibility for the region plus the parents of
        region ops (the same ripple as a flip, unioned over the region)."""
        # The affected set is division-invariant, so it is built (and its old
        # eligibility counted) before the coloring lands.
        affected = set(assignment)
        for op in assignment:
            affected |= self._parents_idx[op]
        affected_sorted = sorted(affected)
        before = sum(self._eligible(x) for x in affected_sorted)
        for op, div in assignment.items():
            self.chosen[op] = div
        for op in sorted(assignment):
            self.packer.resize(op, self._per_core_size(op, self.chosen[op]))
        after = 0
        for x in affected_sorted:
            flag = self._eligible(x)
            after += flag
            self.packer.set_eligible(x, flag)
        self._n_eligible += after - before

    def _recolor(self) -> None:
        """One region-recolor move: a uniform anchor op (so a region is hit
        ∝ its op-count), a random non-trivial anchor tiling, flood, recolor,
        burst."""
        anchor = self._rng.choice(self._anchor_candidates)
        tiling = self._rng.choice(self._nontrivial_menu[anchor])
        self._apply_recolor(self._flood_region(anchor, tiling))
        self._burst()

    def _burst(self) -> None:
        """A short cold layout burst: greedily accept layout steps that do not
        lower the packer's quality, letting ``pi`` adapt to the new footprints
        before the compound move is judged.

        Rejected steps are reverted rather than snapshotted, since
        ``rotate(j, i)`` undoes ``rotate(i, j)``.
        """
        n = len(self._bufs)
        if n < 2:
            return
        # The floor of 1 is there so a small graph still gets a burst.
        for _ in range(max(1, int(_BURST_FRACTION * n))):
            # Nothing left for pi to win once the structural move has left every
            # eligible buffer resident; the rest of the burst is noise.
            if self._all_eligible_resident():
                return
            i = self._rng.randrange(n)
            j = self._rng.randrange(n)
            if self.packer.rotate(i, j) < 0:
                self.packer.rotate(j, i)  # revert

    # -- state snapshots -----------------------------------------------------

    def _snapshot(self) -> tuple[Packer, list[int], int]:
        """An independent copy of the joint state ``(pi, W)``: the packer's
        dynamic layout (``copy`` shares only plan-lifetime structures) plus the
        division vector, and the eligible count ``W`` implies -- rebuilding that
        from ``W`` would cost an O(n) pass the restore does not otherwise need."""
        return (self.packer.copy(), list(self.chosen), self._n_eligible)

    def _adopt(self, snap: tuple[Packer, list[int], int]) -> None:
        """Install ``snap`` as the live state by *taking ownership* of it -- no
        copy, so the engine goes on mutating those objects and the caller must
        treat ``snap`` as dead from here on. Zero-copy because a step already pays
        one O(n) packer copy for its snapshot.

        Restoring rewinds ``(pi, W)`` behind the cost objective's back. Its cached
        bundle *values* stay valid (they are keyed on state), but the baseline it
        diffs against no longer describes the live state, so invalidate that.
        """
        self.packer, self.chosen, self._n_eligible = snap
        if self._cost_objective is not None:
            self._cost_objective.invalidate()

    # -- move selection & execution -----------------------------------------

    def _applicable_moves(self) -> list[str]:
        """Move types available this step, in fixed (deterministic) order: reorder
        needs >=2 buffers, flip a multi-entry menu, recolor a non-trivial anchor.

        Reorder additionally drops out (its proposal weight becomes 0) once every
        eligible buffer is resident: ``pi`` only decides which eligible buffers
        win LX, so with all of them already in there is nothing left for it to
        win, and only a structural move can still pay."""
        moves = []
        if len(self._bufs) >= 2 and not self._all_eligible_resident():
            moves.append("reorder")
        if self._flippable_ops:
            moves.append("flip")
        if self._anchor_candidates:
            moves.append("recolor")
        return moves

    def _choose_move(self) -> str:
        """Fixed-weight move choice."""
        applicable = self._applicable_moves()
        if not applicable:
            return "none"
        weights = [_MOVE_WEIGHTS[m] for m in applicable]
        return self._rng.choices(applicable, weights=weights)[0]

    def _execute_move(self, name: str) -> None:
        """Apply move ``name`` in place; structural moves carry their own burst."""
        n = len(self._bufs)
        if name == "reorder":
            self.packer.rotate(self._rng.randrange(n), self._rng.randrange(n))
        elif name == "flip":
            idx = self._rng.choice(self._flippable_ops)
            menu = len(self._bufs[idx].core_divisions)
            offset = self._rng.randrange(1, menu)  # a different index, wrap-around
            self._atomic_flip(idx, (self.chosen[idx] + offset) % menu)
            self._burst()
        elif name == "recolor":
            self._recolor()
        # "none": no applicable move; no-op.

    # -- annealing loop ------------------------------------------------------

    def _calibrate_temperature(self) -> float:
        """A crude scale estimate: the *median* absolute score delta over a sample
        of random moves -- the starting temperature ``T0``. Median, not mean, to
        survive region-recolor's large deltas; 1.0 when nothing moved. Restores
        state; consumes RNG deterministically."""
        base = self._score()
        deltas: list[int] = []
        for _ in range(min(64, 4 * len(self._bufs) + 8)):
            snap = self._snapshot()
            self._execute_move(self._choose_move())
            d = abs(self._score() - base)
            if d > 0:
                deltas.append(d)
            self._adopt(snap)  # snap dies here; a fresh one is taken next probe
        return float(statistics.median(deltas)) if deltas else 1.0

    def _choose_reinsertion_source(self, allocated: list[bool]) -> int:
        """Pick the permutation *position* to lift out for a sweep reorder, using
        the layout-only annealer's bias (weight ``n`` for a fully-allocated buffer,
        ``n_allocated + 1`` otherwise), which oversamples the buffers that miss LX
        -- the ones the objective prices."""
        n = len(allocated)
        n_allocated = sum(1 for a in allocated if a)
        return self._rng.choices(
            range(n), weights=[n if a else n_allocated + 1 for a in allocated]
        )[0]

    def _sweep_upper_bound(self, i: int, allocated: list[bool]) -> int:
        """Highest reinsertion position worth probing for the buffer at position
        ``i`` -- the layout-only annealer's monotonicity bound.

        A buffer's address is non-decreasing in its position, so one that is *not*
        legally allocated can only be made to fit by moving earlier: past the last
        legally-allocated position, nothing it reaches changes the outcome. An
        allocated buffer has no such bound and sweeps to the end.
        """
        n = len(allocated)
        if allocated[i]:
            return n - 1
        last = max((pos for pos, a in enumerate(allocated) if a), default=0)
        return min(n - 1, last + 1)

    def _step_reorder(self, temperature: float, cur: int) -> int:
        """One best-first reinsertion reorder, the layout-only annealer's move
        (:meth:`SimulatedAnnealingLayoutSolver.annealing_step_rotate`) ported to
        the joint objective. Returns the objective after the step.

        Lift the buffer at position ``i`` out, probe every reinsertion position by
        rotating it to 0 and bubbling it forward one adjacent swap at a time, then
        try the positions **best-first**, accepting the first that clears the
        Metropolis test.

        Ranking is by the packer's ``quality()``, O(1) per position so the sweep is
        O(n), paying a real ``_score()`` only for the candidates it tries. Quality
        is a *proxy* -- it weights a resident buffer by uses x size where the
        objective prices a spilled one by reads-served x size -- and ranking by it
        is deliberate: it breaks ties among the many score-identical positions
        (reorder acceptance runs at 96-100%) and steers ``pi`` toward states a
        later structural move can exploit.

        The probe walks the live packer and restores from the step's own snapshot
        rather than sweeping a copy: placement is a pure function of the
        permutation, so rotate-to-``j`` lands in the same state whichever
        intermediate positions the walk passed through.
        """
        packer = self.packer
        perm = packer.permutation
        n = len(self._bufs)
        allocated = [packer.is_fully_allocated(perm[k]) for k in range(n)]
        i = self._choose_reinsertion_source(allocated)
        upper = self._sweep_upper_bound(i, allocated)

        snap = self._snapshot()

        # keys[p] ranks position p (higher is better); None = not a candidate.
        keys: list[Optional[float]] = [None] * n
        if i != 0:
            packer.rotate(i, 0)
            keys[0] = packer.quality()
        for p in range(1, upper + 1):
            packer.swap(p - 1)  # bubble the lifted buffer from p-1 to p
            if p != i:
                keys[p] = packer.quality()
        pos = max(upper, 0)  # where the lifted buffer now sits

        order = sorted(
            (p for p, k in enumerate(keys) if k is not None),
            key=lambda p: -keys[p],  # type: ignore[operator]
        )
        for j in order:
            packer.rotate(pos, j)
            pos = j
            candidate = self._score()
            delta = candidate - cur
            if delta <= 0 or self._rng.random() < math.exp(-delta / temperature):
                if candidate < self._best_score:
                    self._best_score = candidate
                    self._best_snap = self._snapshot()
                return candidate

        self._adopt(snap)  # nothing accepted; this step's snapshot dies here
        return cur

    def _step(self, name: str, temperature: float, cur: int) -> int:
        """Execute one judged move: propose ``name``, apply the Metropolis test
        against ``temperature``, and update best-seen. Returns the objective after
        the step."""
        if name == "reorder":
            return self._step_reorder(temperature, cur)
        snap = self._snapshot()
        self._execute_move(name)
        new = self._score()
        delta = new - cur
        # `or` short-circuits, so the RNG is drawn only when delta > 0.
        if delta <= 0 or self._rng.random() < math.exp(-delta / temperature):
            if new < self._best_score:
                self._best_score = new
                self._best_snap = self._snapshot()
            return new
        self._adopt(snap)  # this step's snapshot dies here
        return cur

    def _anneal(self) -> None:
        """One geometric cool over the clamped step budget, at fixed proposal
        weights, publishing the best state seen."""
        n = len(self._bufs)
        steps = min(_MAX_STEPS, max(_MIN_STEPS, _STEPS_PER_BUFFER * n))
        if _STEPS_PER_BUFFER * n > _MAX_STEPS:
            logger.debug(
                "SA co-optimizer step budget clamped to %d for %d buffers (%d "
                "steps/buffer would ask for %d); layout quality is traded for "
                "bounded compile time.",
                _MAX_STEPS,
                n,
                _STEPS_PER_BUFFER,
                _STEPS_PER_BUFFER * n,
            )
        self._flippable_ops = self._flippable()

        cur = self._score()
        self._best_score = cur
        self._best_snap = self._snapshot()

        if self._applicable_moves():
            t0 = self._calibrate_temperature()
            t_end = max(t0 / _COOLING_SPAN, 1e-9)
            for step in range(steps):
                move = self._choose_move()
                # Only a move changes the state, so once none applies (every
                # eligible buffer resident and no structural move available) the
                # rest of the budget cannot find anything.
                if move == "none":
                    break
                frac = step / (steps - 1) if steps > 1 else 1.0
                cur = self._step(move, t0 * (t_end / t0) ** frac, cur)

        self.best_score = self._best_score
        # Adopting is safe only because nothing mutates the state after this: the
        # engine must not go on rewriting the layout that ``best_score`` describes.
        self._adopt(self._best_snap)

    # -- write-back ----------------------------------------------------------

    def _write_back(self) -> None:
        """Commit the best state to the buffers and record spill causes."""
        for i, b in enumerate(self._bufs):
            addr = self.packer.addresses[i]
            b.chosen_division = self.chosen[i]
            b.address = addr
            if addr is None:
                self.spill_reasons[b.name] = b.residency_reason or _SOLVER_CHOSE_SPILL
