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
* ``W`` -- the work division, one :class:`DivisionConfig` per buffer.

Moves are reorder, atomic division flip, and region-recolor; each structural
move runs as a compound move+burst judged as a unit by one Metropolis test.
Region-recolor floods the residency relation bidirectionally from a splitting
anchor config, so the region *is* the flood's reach and boundaries emerge for
free; an edge with no compatible division becomes an accepted internal seam.
A flip proposes one axis's factor, one step; a recolor draws a splitting
division outright and floods it, which is the search's long-range move.

Best-seen over ``(pi, W)`` from the seed state (every op at its seed config,
``pi`` from FirstFit) keeps every returned state no worse than that baseline.

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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import sympy

from torch_spyre._inductor.scratchpad.division_generation import (
    OpSplitSpace,
    ResidencyEdge,
    undeclared_splits,
)
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
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import iteration_space_from_op

if TYPE_CHECKING:  # pragma: no cover - typing only
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


def _canonical_key(division: "CoreDivision") -> tuple:
    """A hashable identity for ``division`` within its op's symbol namespace.

    Split keys are the producer's own iteration symbols, so this compares only
    within one operation -- the scope ``CoreDivision.signature_key`` already
    documents. Unlike that one it is *total*: it keeps the reduction splits and
    the tiling, so two divisions share a key only when they are the same choice.
    """

    def splits(mapping: dict) -> tuple:
        return tuple(sorted(mapping.items(), key=lambda item: str(item[0])))

    return (
        splits(division.output_splits),
        splits(division.reduction_splits),
        division.tiling,
    )


@dataclass(frozen=True, eq=False)
class DivisionConfig:
    """One op's work division as a value -- the annealer's state element.

    ``chosen[i]`` holds one of these rather than a menu position, so a config the
    engine *generates* rather than enumerates is usable wherever a menu entry is.
    Equality and hashing are :attr:`key`'s, which makes two configs equal exactly
    when they are the same *choice*, and lets a generated set be deduplicated or
    memoized. That is deliberately coarser than menu position: a menu carries the
    same choice at several positions (see
    :meth:`SaCoOptimizingSolver._build_configs`), and those compare equal.

    ``menu_index`` is provenance: the position this config came from in its
    buffer's ``core_divisions``, or ``None`` for a *generated* config, which
    came from no position at all. Only the write-back reads it -- the
    ``chosen_division`` the allocator re-indexes the menu with, which is the
    allocator's contract rather than the engine's, and the one place a
    generated config has to be given a position (see
    :meth:`SaCoOptimizingSolver._write_back`). The residency gate and the
    recolor flood, which used to read it, now ask an :class:`_EdgeRelation`.
    """

    division: "CoreDivision"
    key: tuple
    menu_index: Optional[int]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DivisionConfig) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def output_splits(self) -> dict:
        return self.division.output_splits

    @property
    def reduction_splits(self) -> dict:
        return self.division.reduction_splits

    @property
    def output_partition(self) -> int:
        return self.division.output_partition


def _one_axis_apart(left: "CoreDivision", right: "CoreDivision") -> bool:
    """Whether two divisions differ in exactly one axis's factor.

    Compared over the union of both split maps, since a factor of 1 is dropped
    from a sparse map: ``{d0: 2}`` and ``{d0: 2, d1: 2}`` are one axis apart.
    An axis neither division splits is 1 on both sides and so never counts.
    """

    def factors(division: "CoreDivision") -> dict:
        return {**division.output_splits, **division.reduction_splits}

    lhs, rhs = factors(left), factors(right)
    return sum(lhs.get(key, 1) != rhs.get(key, 1) for key in lhs | rhs) == 1


class _DivisionSource:
    """Where one buffer's candidate divisions come from, and what a move may
    reach from a given one.

    The engine asks only this, so it does not care whether the candidates were
    enumerated into a menu (:class:`_MenuDivisions`) or are generated on demand
    (:class:`_GeneratedDivisions`). The engine's seeded generator is the only
    randomness in the search; a source that has to draw (the recolor anchor) is
    handed it rather than holding one.

    The two structural moves ask for different scales, and measurably need to:
    :meth:`neighbours` is *one axis's factor*, which is what a flip takes -- and
    stage 0 measured those domains at ~7 legal factors per axis, a list to pick
    from rather than an interval to propose over with a cooling scale. But an
    op's legal divisions are not connected by one-axis moves (the core budget
    blocks a factor going up, a span floor blocks it coming down), so a search
    with only local moves does measurably worse: +0.9% on the corpus at four
    seeds. :meth:`anchor` is the long-range draw that pays for it, and recolor
    is where it belongs, since a flooded region is a coordinated change anyway.
    """

    def seed(self) -> DivisionConfig:
        """The buffer's committed division: where the search starts."""
        raise NotImplementedError

    def config_for(self, division: "CoreDivision") -> DivisionConfig:
        """``division`` as a config of this buffer's."""
        raise NotImplementedError

    def can_move(self) -> bool:
        """Whether this buffer has an alternative division at all -- a static
        filter, so the per-step move draw is over a fixed list. It is only an
        upper bound on :meth:`neighbours`, which is state-dependent."""
        raise NotImplementedError

    def can_split(self) -> bool:
        """Whether this buffer could take a *split* division, the only legal
        recolor anchor. Static, and for a generated source an
        over-approximation; :meth:`anchor` is what actually decides."""
        raise NotImplementedError

    def _one_axis_moves(self, config: DivisionConfig) -> list[DivisionConfig]:
        raise NotImplementedError

    def neighbours(self, config: DivisionConfig) -> list[DivisionConfig]:
        """The divisions one axis away from ``config``. Memoized by choice,
        since a search revisits states."""
        cached = self._neighbour_cache.get(config.key)
        if cached is None:
            cached = self._one_axis_moves(config)
            self._neighbour_cache[config.key] = cached
        return cached

    def anchor(self, config: DivisionConfig, rng) -> Optional[DivisionConfig]:
        """A *splitting* division for a recolor to flood from, drawn with
        ``rng``, or ``None`` if this buffer has none to offer.

        Splitting only, so recolor stays a coordinated splitting move and
        undividing is left to atomic flips. Unlike :meth:`neighbours` this is
        not restricted to one axis-step from ``config``: recolor is the search's
        long-range move.
        """
        raise NotImplementedError


@dataclass
class _MenuDivisions(_DivisionSource):
    """A buffer's divisions as its enumerated ``core_divisions`` menu carries
    them, deduplicated by choice.

    A menu holds the same division at several positions -- a factor-1 axis is
    dropped from the sparse split map, so ``{d0: 2, d1: 1}`` enumerates as a
    second copy of ``{d0: 2}`` -- and those positions are one choice. Collapsing
    them is what makes the move alphabet a set of choices rather than of
    positions: otherwise a duplicated choice is drawn twice as often, and which
    copy a flood met would decide the division it propagated. The surviving
    representative is the first, which is also the position written back.
    """

    configs: list[DivisionConfig]
    by_key: dict
    _neighbour_cache: dict = field(default_factory=dict, repr=False)

    @property
    def splitting(self) -> list[DivisionConfig]:
        """The menu's splitting divisions, in menu order -- which is the order
        the anchor draw indexes."""
        return [config for config in self.configs if config.output_partition > 1]

    def seed(self) -> DivisionConfig:
        return self.configs[0]

    def anchor(self, config: DivisionConfig, rng) -> Optional[DivisionConfig]:
        splitting = self.splitting
        return rng.choice(splitting) if splitting else None

    def config_for(self, division: "CoreDivision") -> DivisionConfig:
        return self.by_key[_canonical_key(division)]

    def can_move(self) -> bool:
        return len(self.configs) > 1

    def can_split(self) -> bool:
        return any(config.output_partition > 1 for config in self.configs)

    def _one_axis_moves(self, config: DivisionConfig) -> list[DivisionConfig]:
        return [
            candidate
            for candidate in self.configs
            if _one_axis_apart(candidate.division, config.division)
        ]


@dataclass
class _GeneratedDivisions(_DivisionSource):
    """A buffer's divisions generated from its op's split space.

    The space admits exactly what the enumeration carries, so this reaches the
    same divisions a menu would -- without materializing them, and without the
    ``|D_p| x |D_c|`` view comparisons an eager compatibility table costs. A
    config from here carries no ``menu_index``: the position is resolved once,
    against the menu, when the chosen division is written back.
    """

    space: OpSplitSpace
    seed_config: DivisionConfig
    _neighbour_cache: dict = field(default_factory=dict, repr=False)
    _config_cache: dict = field(default_factory=dict, repr=False)

    def seed(self) -> DivisionConfig:
        return self.seed_config

    def config_for(self, division: "CoreDivision") -> DivisionConfig:
        key = _canonical_key(division)
        if key not in self._config_cache:
            self._config_cache[key] = DivisionConfig(division, key, None)
        return self._config_cache[key]

    def can_move(self) -> bool:
        return any(len(factors) > 1 for factors in self.space.factor_domains.values())

    def can_split(self) -> bool:
        return any(
            axis in self.space.output_axes and any(factor > 1 for factor in factors)
            for axis, factors in self.space.factor_domains.items()
        )

    def _one_axis_moves(self, config: DivisionConfig) -> list[DivisionConfig]:
        return [
            self.config_for(division)
            for division in self.space.neighbours(config.division)
        ]

    def anchor(self, config: DivisionConfig, rng) -> Optional[DivisionConfig]:
        """Redraw every axis, keeping each draw that leaves the division legal.

        The generated stand-in for drawing uniformly from a menu's splitting
        entries: it reaches divisions many axis-steps away, and it costs one
        draw and one legality check per axis rather than a walk over the whole
        space. Starting from ``config`` -- which is legal -- means a rejected
        draw simply leaves that axis alone, so the result is always legal.
        """
        splits = self.space.splits(config.division)
        for axis in self.space.axes:
            candidate = dict(splits)
            candidate[axis] = rng.choice(self.space.factor_domains[axis])
            if self.space.admits(candidate):
                splits = candidate
        division = self.space.division(splits)
        if division.output_partition <= 1:
            return None
        return self.config_for(division)


class _EdgeRelation:
    """Which pairs of divisions let a consumer read a producer's buffer from
    LX, and how a search propagates one across the edge.

    The engine asks this in two places: the residency gate
    (:meth:`SaCoOptimizingSolver._eligible`), which needs the verdict for a
    pair, and the recolor flood, which needs the division on the other end.
    """

    def compatible(self, parent: DivisionConfig, child: DivisionConfig) -> bool:
        raise NotImplementedError

    def child_for(self, parent: DivisionConfig) -> Optional[DivisionConfig]:
        raise NotImplementedError

    def parent_for(self, child: DivisionConfig) -> Optional[DivisionConfig]:
        raise NotImplementedError


@dataclass
class _TableRelation(_EdgeRelation):
    """The edge relation as the allocator's ``cd_parent_matches`` pair table,
    re-keyed by choice.

    The table is keyed by menu position and several positions can carry one
    choice, so it is projected onto keys once. That also makes it exact for a
    *generated* config, which is why a graph where only some ops have a split
    space is not a mixture of two answers: a generated division is one the
    enumeration would have carried, so its key is a key the table knows.
    """

    pairs: frozenset
    down: dict
    up: dict

    def compatible(self, parent: DivisionConfig, child: DivisionConfig) -> bool:
        return (parent.key, child.key) in self.pairs

    def child_for(self, parent: DivisionConfig) -> Optional[DivisionConfig]:
        return self.down.get(parent.key)

    def parent_for(self, child: DivisionConfig) -> Optional[DivisionConfig]:
        return self.up.get(child.key)


def _table_relation(
    pairs: Iterable[tuple[int, int]],
    parent_menu: tuple[Sequence[tuple], dict],
    child_menu: tuple[Sequence[tuple], dict],
) -> _TableRelation:
    """Project a menu-position pair table onto choices.

    Each side is its menu's ``(keys by position, key -> representative config)``
    pair, which every buffer has whether or not its own source generates. The
    propagation direction keeps the pair table's tie-break -- the compatible
    division at the lowest menu position wins -- which is what makes a flood
    independent of ``cd_parent_matches`` list order.
    """
    parent_keys, parent_by_key = parent_menu
    child_keys, child_by_key = child_menu
    key_pairs = {
        (parent_keys[ip], child_keys[ic])
        for ip, ic in pairs
        if ip < len(parent_keys) and ic < len(child_keys)
    }
    # Ordered by menu position rather than by iterating the set, whose order is
    # a hash order and so not stable across interpreter runs.
    position = {
        pair: (parent_by_key[pair[0]].menu_index, child_by_key[pair[1]].menu_index)
        for pair in key_pairs
    }
    down: dict = {}
    up: dict = {}
    for pair in sorted(key_pairs, key=lambda kp: position[kp][::-1]):
        down.setdefault(pair[0], child_by_key[pair[1]])
    for pair in sorted(key_pairs, key=lambda kp: position[kp]):
        up.setdefault(pair[1], parent_by_key[pair[0]])
    return _TableRelation(frozenset(key_pairs), down, up)


@dataclass
class _ViewRelation(_EdgeRelation):
    """The edge relation computed per candidate, from the buffer's geometry.

    :class:`ResidencyEdge` owns both the view comparison and the residency
    policy filters, and inverts a view to *construct* the division on the other
    end. Memoized by choice, so the pair table this replaces is built lazily and
    only where the search actually looks -- which is the point of generating
    configs rather than enumerating them.
    """

    edge: "ResidencyEdge"
    parent_source: _GeneratedDivisions
    child_source: _GeneratedDivisions
    _compatible: dict = field(default_factory=dict, repr=False)
    _down: dict = field(default_factory=dict, repr=False)
    _up: dict = field(default_factory=dict, repr=False)

    def compatible(self, parent: DivisionConfig, child: DivisionConfig) -> bool:
        pair = (parent.key, child.key)
        if pair not in self._compatible:
            self._compatible[pair] = self.edge.compatible(
                parent.division, child.division
            )
        return self._compatible[pair]

    def child_for(self, parent: DivisionConfig) -> Optional[DivisionConfig]:
        if parent.key not in self._down:
            division = self.edge.consumer_division_for(
                parent.division, self.child_source.space
            )
            self._down[parent.key] = (
                None if division is None else self.child_source.config_for(division)
            )
        return self._down[parent.key]

    def parent_for(self, child: DivisionConfig) -> Optional[DivisionConfig]:
        if child.key not in self._up:
            division = self.edge.parent_division_for(
                child.division, self.parent_source.space
            )
            self._up[child.key] = (
                None if division is None else self.parent_source.config_for(division)
            )
        return self._up[child.key]


class SaCoOptimizingSolver(CoreDivisionLayoutSolver):
    """SA joint core-division + LX-placement engine.

    The search is fully determined by the module constants above; there is
    nothing to configure per call.

    Args:
        buffers: the buffers to plan, in the allocator's order. Declared as
            ``Sequence[LifetimeBoundBuffer]`` so the class itself satisfies
            ``CoreDivisionSolverFactory`` (``Callable`` parameters are
            contravariant, so a narrower annotation would not), but every buffer
            passed must be a :class:`CoreDivisionBuffer` -- the engine reads
            each one's ``core_divisions``, its residency relation to each parent
            (``division_space`` and ``residency_edges`` where the allocator
            built them, ``cd_parent_matches`` otherwise), and its cost symbols.

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
        # Built from ``cost_expr`` once ``plan_layout_and_core_divisions`` has it
        # (see :meth:`_build_score_fn`); ``None`` until then, which also means
        # "no usable cost expression" -- the memory-only objective's signal.
        self._score_fn: Any = None
        # The division vector ``W``: one config per buffer, positionally. Set at
        # the seed (see :meth:`_seed_configs`); declared here for the types.
        self.chosen: list[DivisionConfig]
        # Best-seen over the anneal (set in _anneal, read in _step); declared for
        # the types.
        self._best_score: int
        self._best_snap: tuple[Packer, list[DivisionConfig], int]
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
        self, cost_expr: Optional[sympy.Expr] = None
    ) -> list[CoreDivisionBuffer]:
        """Anneal the joint ``(pi, W)`` state and write ``chosen_division`` /
        ``address`` back to each buffer; populate ``spill_reasons``. Returns the
        solver's own buffers. Single-use: construct a fresh solver per set.
        """
        self.spill_reasons = {}
        n = len(self._bufs)
        if n == 0:
            return list(self._bufs)

        self._rng = rnd.Random(_SEED)
        # Before the score function, which prices the configs the topology pass
        # builds against the symbol set it freezes. Consumes no randomness, so
        # the search trajectory is unaffected by running first.
        self._precompute_topology()

        self._score_fn = self._build_score_fn(cost_expr)
        if self._score_fn is None:
            logger.info(
                "no usable cost expression; falling back to the memory-only objective"
            )

        # Seed: every op at its committed division; pi from FirstFit.
        self.chosen = self._seed_configs()
        self.packer = self._build_seed_packer()

        self._anneal()
        self._write_back()
        return list(self._bufs)

    def _build_score_fn(self, cost_expr: Optional[sympy.Expr]):
        """Compile ``cost_expr`` into a ``(chosen, resident) -> fixed-point ns``
        callable, or ``None`` if it can't be evaluated from only this solver's
        own buffers.

        Every free symbol becomes a getter over the live state: a residency
        symbol from whether its buffer's name is in ``resident``, a split symbol
        from ``chosen[idx]`` -- the config itself, so a generated one prices
        exactly as a menu entry does. The symbols come from the *declaration*
        :meth:`_build_configs` froze, which is also what makes that so: a config
        splitting an axis the declaration has no symbol for would be priced at
        the default of 1 rather than rejected.

        ``None`` (no expression, or a symbol this can't place -- e.g. a dynamic-
        shape symbol the allocator's build left in) falls back to the
        memory-only objective
        """
        if cost_expr is None:
            return None
        value_of: dict = {}  # sympy.Symbol -> (chosen, resident) -> number
        for idx, buf in enumerate(self._bufs):
            value_of[buf.sym_is_lx] = lambda chosen, resident, name=buf.name: (
                1 if name in resident else 0
            )
            out_syms, red_syms = self._sym_core_divs[idx]
            for key, sym in out_syms.items():
                value_of[sym] = lambda chosen, resident, idx=idx, key=key: (
                    chosen[idx].output_splits.get(key, 1)
                )
            for key, sym in red_syms.items():
                value_of[sym] = lambda chosen, resident, idx=idx, key=key: (
                    chosen[idx].reduction_splits.get(key, 1)
                )
        try:
            free = sorted(cost_expr.free_symbols, key=str)
            if any(sym not in value_of for sym in free):
                return None
            fn = sympy.lambdify(free, cost_expr, modules="math")
        except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
            return None

        def score(chosen, resident) -> int:
            ns = fn(*(value_of[sym](chosen, resident) for sym in free))
            return utils.to_fixed_us(max(0.0, ns) / 1000.0)

        return score

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
        the per-buffer division sources, the name->index map, each buffer's
        parent indices, and -- keyed by parent index -- its children with the
        relation that decides which of their divisions are compatible.

        No consumer *count* is derived here: :meth:`_spill_cost` scales by
        reads-served instead. ``_children`` remains available for the cohort
        multiplicity when op metadata is wired in.
        """
        self._assert_unsized_buffers_are_pinned()
        self._build_sources()
        bufs = self._bufs
        self._name_to_idx = {b.name: i for i, b in enumerate(bufs)}
        n = len(bufs)
        self._parents_idx: list[set[int]] = [set() for _ in range(n)]
        # parent_idx -> list of (child_idx, the p->c relation)
        self._children: list[list[tuple[int, _EdgeRelation]]] = [[] for _ in range(n)]
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
                self._children[p_idx].append(
                    (c_idx, self._edge_relation(p_idx, c_idx, p_name))
                )
        if foreign_parents:
            logger.debug(
                "dropped %d parent edge(s) naming buffers outside the solver's "
                "set (graph inputs / constants / externs)",
                foreign_parents,
            )

        # Region-recolor support. ``_relations[(p, c)]`` is the relation on the
        # edge p->c; ``_children_idx`` lists each op's children by index
        # (deterministic flood order).
        self._children_idx = [sorted(c for c, _ in self._children[i]) for i in range(n)]
        self._relations: dict[tuple[int, int], _EdgeRelation] = {
            (i, c): relation for i in range(n) for c, relation in self._children[i]
        }
        # Ops that could take a split division at all -- the only legal recolor
        # anchors, so recolor stays a coordinated *splitting* move and leaves
        # undividing to atomic flips. Static; what a given step can actually
        # draw is :meth:`_DivisionSource.anchor`.
        self._anchor_candidates = [i for i in range(n) if self._sources[i].can_split()]
        generated = sum(
            isinstance(source, _GeneratedDivisions) for source in self._sources
        )
        view_relations = sum(
            isinstance(relation, _ViewRelation) for relation in self._relations.values()
        )
        logger.debug(
            "division sources: %d generated / %d from the menu; edge relations: "
            "%d per candidate / %d from the pair table",
            generated,
            n - generated,
            view_relations,
            len(self._relations) - view_relations,
        )
        self._precompute_spill_costs()

    def _edge_relation(self, p_idx: int, c_idx: int, p_name: str) -> _EdgeRelation:
        """The relation on the edge ``p_idx -> c_idx``.

        Computed per candidate off the buffer's geometry where both ends
        generate their divisions and the allocator handed over the edge;
        otherwise the pair table, which is what a clone parent, a
        non-``ComputedBuffer`` op and a division-pinned op still have. The two
        answer the same question -- a generated division is one the enumeration
        would have carried, so the table knows its key -- so a graph that mixes
        them is not a mixture of two answers.
        """
        parent_source = self._sources[p_idx]
        child_source = self._sources[c_idx]
        edge = self._bufs[c_idx].residency_edges.get(p_name)
        if (
            edge is not None
            and isinstance(parent_source, _GeneratedDivisions)
            and isinstance(child_source, _GeneratedDivisions)
        ):
            return _ViewRelation(edge, parent_source, child_source)
        return _table_relation(
            (
                (int(a), int(b))
                for a, b in self._bufs[c_idx].cd_parent_matches.get(p_name, [])
            ),
            self._menu[p_idx],
            self._menu[c_idx],
        )

    def _build_sources(self) -> None:
        """Build each buffer's division source, and freeze the symbol set its
        divisions are priced over.

        The symbol set is the *declaration*: ``sym_core_divs`` carries one symbol
        per stride coefficient seen across a buffer's candidates, and
        :meth:`_build_score_fn` values only those. A division splitting an axis
        outside it would be priced at the symbol's default of 1 -- a wrong
        answer, silently. A menu meets the declaration by construction, since the
        declaration is derived from it; a generated one is held to it by
        :class:`OpSplitSpace`, which the allocator builds with the declaration in
        hand. Asserting it over the menu is what catches the two going out of
        step.

        The menu is kept as well as the source, keyed by choice: the pair-table
        relations are projected through it, and it is how the write-back turns
        the chosen division back into the position the allocator re-indexes.
        Positions are *not* distinct by choice -- the enumerator emits the same
        division at several of them, because a factor-1 axis is omitted from the
        sparse split map (measured on softmax: four of ``arg0_1``'s 20 candidates
        repeat an earlier one) -- so the first position carrying a choice is its
        representative everywhere.
        """
        self._sym_core_divs = [b.sym_core_divs for b in self._bufs]
        # Per buffer: the menu's keys by position, and the representative config
        # per key.
        self._menu: list[tuple[list[tuple], dict]] = []
        self._sources: list[_DivisionSource] = []
        for buf, declared in zip(self._bufs, self._sym_core_divs):
            keys: list[tuple] = []
            by_key: dict = {}
            configs: list[DivisionConfig] = []
            for index, cd in enumerate(buf.core_divisions):
                undeclared = undeclared_splits(cd, declared)
                assert not undeclared, (
                    f"buffer {buf.name}: candidate {index} splits "
                    f"{sorted(str(key) for key in undeclared)}, which the cost "
                    "expression declares no symbol for, so those splits would be "
                    "priced as unsplit"
                )
                key = _canonical_key(cd)
                keys.append(key)
                if key not in by_key:
                    by_key[key] = DivisionConfig(cd, key, index)
                    configs.append(by_key[key])
            self._menu.append((keys, by_key))
            space = buf.division_space
            self._sources.append(
                _GeneratedDivisions(space, configs[0])
                if space is not None
                else _MenuDivisions(configs, by_key)
            )

    def _seed_configs(self) -> list[DivisionConfig]:
        """The seed division vector: every op at its committed division, the
        candidate the allocator enumerates first."""
        return [source.seed() for source in self._sources]

    def _precompute_spill_costs(self) -> None:
        """Cache the loop-invariant inputs to :meth:`_score`. A move changes only
        the *per-core* footprint the packer sees, never a buffer's total size, so
        neither the spill costs nor the bandwidth constant can move."""
        self._spill_costs = [self._spill_cost(b) for b in self._bufs]
        self._hbm_bytes_per_us = utils.hbm_bytes_per_us()

    # -- division-dependent derivations --------------------------------------

    def _per_core_size(self, idx: int, config: DivisionConfig) -> int:
        """Per-core footprint of buffer ``idx`` under ``config``:
        ``ceil_div(total_size, output_partition)``, using the substrate's integer
        helper so this rounds identically to every other footprint-division site.

        Clamped non-negative so the packer never sees a negative size from the
        ``mem_usage`` ``-1`` sentinel; what stops an unsized buffer from looking
        *placeable* at zero footprint is
        :meth:`_assert_unsized_buffers_are_pinned`."""
        return max(0, ceil_div(self._bufs[idx].size, config.output_partition))

    def _eligible(self, idx: int) -> bool:
        """Whether buffer ``idx`` may be LX-resident under the current ``W``
        (the three division-dependent gates, mirroring
        ``DfsLayoutSolver._evaluate``): the fixed residency pin, a per-core
        footprint that fits at all, and a division every child edge's
        :class:`_EdgeRelation` calls compatible.

        That relation is per-core-view based, not ``is_clean`` based: a reduction
        split can appear on the *consumer* side (a K-split reading a clean parent
        via the PSUM ring) but never on the parent side, since a reduction-split
        producer writes a partial sum no child may read from LX -- so such a
        producer is always gated out here."""
        b = self._bufs[idx]
        # Not ``MemoryPlanSolver.excluded()``: that folds in a ``min_footprint >
        # limit`` test, which is division-dependent and is the next gate down.
        if b.residency_reason is not None:
            return False
        if self._per_core_size(idx, self.chosen[idx]) > self.limit:
            return False
        parent = self.chosen[idx]
        return all(
            relation.compatible(parent, self.chosen[c_idx])
            for c_idx, relation in self._children[idx]
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
        """Build the packer for the seed state: the per-core sizes ``chosen``
        implies, a FirstFit-derived ``pi``, and the seed eligibility mask."""
        n = len(self._bufs)
        sizes = [self._per_core_size(i, self.chosen[i]) for i in range(n)]
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
        if self._score_fn is not None:
            resident = frozenset(
                b.name
                for b, address in zip(self._bufs, addresses)
                if address is not None
            )
            return self._score_fn(self.chosen, resident)

        traffic = sum(
            cost
            for cost, address in zip(self._spill_costs, addresses)
            if address is None
        )
        return utils.to_fixed_us(traffic / self._hbm_bytes_per_us)

    # -- moves ---------------------------------------------------------------

    def _flippable(self) -> list[int]:
        """Buffer indices whose division source offers an alternative at all.

        Static, so the per-step draw is over a fixed list; whether the division
        the buffer currently holds has a *neighbour* is decided at move time."""
        return [i for i in range(len(self._bufs)) if self._sources[i].can_move()]

    def _atomic_flip(self, idx: int, config: DivisionConfig) -> None:
        """Change buffer ``idx``'s division to ``config`` and ripple: resize its
        per-core footprint, then refresh eligibility for ``idx`` and its parents.
        Those are the only buffers a flip can change, since eligibility depends on
        an op's own division and its children's."""
        affected = sorted({idx} | self._parents_idx[idx])
        before = sum(self._eligible(x) for x in affected)
        self.chosen[idx] = config
        self.packer.resize(idx, self._per_core_size(idx, config))
        after = 0
        for x in affected:
            flag = self._eligible(x)
            after += flag
            self.packer.set_eligible(x, flag)
        self._n_eligible += after - before

    def _flood_region(
        self, anchor: int, config: DivisionConfig
    ) -> dict[int, DivisionConfig]:
        """Flood the ``cd_parent_matches`` relation from ``(anchor, config)`` to a
        config assignment over the reachable region.

        Bidirectional: from an assigned op ``u``, a child ``c`` joins at the
        division that reads ``u``'s buffer the way ``u`` writes it, and a parent
        ``p`` at the one that writes ``p``'s buffer the way ``u`` reads it. Each
        is the edge relation's answer -- constructed by inverting the view where
        both ends generate, looked up in the pair table otherwise. The reachable
        set *is* the region; an edge with no compatible division is simply not
        extended across -- an accepted internal seam, never a failure.

        First-assignment-wins with a min-index frontier makes this independent of
        the order the edges are visited in.
        """
        assignment = {anchor: config}
        heap = [anchor]
        while heap:
            u = heapq.heappop(heap)
            for c in self._children_idx[u]:  # down: u -> c
                if c in assignment:
                    continue
                joined = self._relations[(u, c)].child_for(assignment[u])
                if joined is not None:
                    assignment[c] = joined
                    heapq.heappush(heap, c)
            for p in sorted(self._parents_idx[u]):  # up: p -> u
                if p in assignment:
                    continue
                joined = self._relations[(p, u)].parent_for(assignment[u])
                if joined is not None:
                    assignment[p] = joined
                    heapq.heappush(heap, p)
        return assignment

    def _apply_recolor(self, assignment: dict[int, DivisionConfig]) -> None:
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
        for op, config in assignment.items():
            self.chosen[op] = config
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
        ∝ its op-count), a random splitting anchor division, flood, recolor,
        burst.

        This is the search's long-range move -- the anchor is drawn from the
        whole space rather than one axis-step away, which is what keeps a
        one-axis flip from having to reach everything on its own (see
        :class:`_DivisionSource`). An op whose draw came out unsplit has nothing
        to propagate, which is a no-op step rather than a special case."""
        anchor = self._rng.choice(self._anchor_candidates)
        config = self._sources[anchor].anchor(self.chosen[anchor], self._rng)
        if config is None:
            return
        self._apply_recolor(self._flood_region(anchor, config))
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

    def _snapshot(self) -> tuple[Packer, list[DivisionConfig], int]:
        """An independent copy of the joint state ``(pi, W)``: the packer's
        dynamic layout (``copy`` shares only plan-lifetime structures) plus the
        division vector, and the eligible count ``W`` implies -- rebuilding that
        from ``W`` would cost an O(n) pass the restore does not otherwise need.
        Configs are immutable values, so the shallow list copy is a full copy of
        ``W``."""
        return (self.packer.copy(), list(self.chosen), self._n_eligible)

    def _adopt(self, snap: tuple[Packer, list[DivisionConfig], int]) -> None:
        """Install ``snap`` as the live state by *taking ownership* of it -- no
        copy, so the engine goes on mutating those objects and the caller must
        treat ``snap`` as dead from here on. Zero-copy because a step already pays
        one O(n) packer copy for its snapshot.
        """
        self.packer, self.chosen, self._n_eligible = snap

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
            # One axis's factor, drawn uniformly from the divisions an axis-step
            # away. Stage 0 measured those domains at ~7 factors per axis, which
            # is why there is a list to pick from here rather than a proposal
            # scale to cool.
            options = self._sources[idx].neighbours(self.chosen[idx])
            if not options:
                return
            self._atomic_flip(idx, self._rng.choice(options))
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
        """Commit the best state to the buffers and record spill causes.

        ``chosen_division`` is a position in ``core_divisions``, which is the
        allocator's contract and the one place a generated division has to be
        given a position. A generated division is normally one the enumeration
        already carries, so the position is looked up by choice; a division the
        menu does not carry is appended, which is what a truncated menu (the
        pruned enumeration) or a division carrying something the menu cannot
        express leaves. Appending here rather than when the division is proposed
        keeps the menu the search ran against unchanged.
        """
        for i, b in enumerate(self._bufs):
            addr = self.packer.addresses[i]
            b.chosen_division = self._menu_position(i, self.chosen[i])
            b.address = addr
            if addr is None:
                self.spill_reasons[b.name] = b.residency_reason or _SOLVER_CHOSE_SPILL

    def _menu_position(self, idx: int, config: DivisionConfig) -> int:
        """The position in buffer ``idx``'s ``core_divisions`` that names
        ``config``'s division, registering it if the menu has no such entry."""
        if config.menu_index is not None:
            return config.menu_index
        _keys, by_key = self._menu[idx]
        known = by_key.get(config.key)
        if known is not None:
            return known.menu_index
        divisions = self._bufs[idx].core_divisions
        divisions.append(config.division)
        return len(divisions) - 1
