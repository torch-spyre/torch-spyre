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
  :class:`NativePermutationLayoutSolver` packer, because this loop mixes move
  types and scores a richer objective than the packer's own ``quality()``.
* ``W`` -- the work division, one :class:`DivisionConfig` per buffer.

Moves are reorder, atomic division flip, and region-recolor; each structural
move runs as a compound move+burst judged as a unit by one Metropolis test.
Region-recolor floods the residency relation bidirectionally from a splitting
anchor config, so the region *is* the flood's reach and boundaries emerge for
free; an edge with no compatible division becomes an accepted internal seam.
A flip proposes one axis's factor, one step; a recolor draws a splitting
division any number of axes away and floods it, which is the search's
long-range move. A generated source draws it by redrawing the current one, so
unlike a menu draw it depends on the state.

Best-seen over ``(pi, W)`` from the seed state (every op at its seed config,
``pi`` from FirstFit) keeps every returned state no worse than that baseline.

Determinism: a seeded ``Random`` over index-ordered domains and the integer
fixed-point score make a run bit-for-bit reproducible.

Design notes: ``docs/source/compiler/sa_co_optimization.md``.
"""

from __future__ import annotations

import copy
import heapq
import keyword
import math
import random as rnd
import statistics
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence, Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import sympy

from torch_spyre._inductor.work_division import (
    UNTILED as _UNTILED,
    OpSplitSpace,
    ResidencyEdge,
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
    RelayoutCharge,
    RelayoutCopyBuffer,
    TileAxis,
    TileSpec,
    ceil_div,
)
from torch_spyre._inductor.scratchpad.lx_relayout import ChosenRelayout
from torch_spyre._C import NativePermutationLayoutSolver
from torch_spyre._inductor.scratchpad import utils
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import iteration_space_from_op

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_spyre._inductor.scratchpad.coarse_tiling import TileReads
    from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

logger = get_inductor_logger("scratchpad.sa_cooptimizer")

# RNG seed; fixes the (deterministic) search trajectory.
_SEED = 0

# Step budget, see _step_budget. The ceiling sits above the layout-only
# annealer's clamp (``SelfCalibratingReheatingSchedule.max_steps``, 5_000) since
# this engine searches divisions too, and binds only well past the validated
# corpus. It bounds *steps*, not wall-clock.
_STEPS_PER_BUFFER = 200
_MAX_STEPS = 50_000


def _step_budget(n: int) -> int:
    """The anneal's step count for ``n`` buffers that carry a division decision."""
    return min(_MAX_STEPS, _STEPS_PER_BUFFER * n)


# Fixed proposal weights over the move types. Reorder's weight is effectively 0
# while every eligible buffer is resident, and relayout's on a graph with no
# relayout copy (see :meth:`_applicable_moves`).
_MOVE_WEIGHTS = {"reorder": 0.5, "flip": 0.3, "recolor": 0.2, "relayout": 0.2}

# Layout-burst length as a fraction of the buffer count. The burst warms ``pi`` to
# the new footprints before a compound structural move is judged.
_BURST_FRACTION = 0.1

# How often a recolor anchor is drawn untiled outright (see
# ``_GeneratedDivisions._draw_tiling``). Flat, and deliberately not a per-dim
# opt-out: the probability of proposing an untiled region must not decay with the
# number of tileable dims.
_UNTILED_ANCHOR_PROB = 0.5

# The geometric cool spans t0 down to t0 / _COOLING_SPAN.
_COOLING_SPAN = 1000.0

# Cause recorded for a buffer the SA engine left out of LX.
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"

# What ``sym_division`` binds to for a division its buffer's menu does not carry:
# outside every table indexed by menu position, which ``RelayoutCharge`` reads as
# unpriced.
_UNPRICED = -1


def _work_slices(op, division: "CoreDivision") -> dict:
    """Restore a complete symbol-keyed split map from a sparse candidate."""
    return {
        symbol: division.splits.get(symbol, 1) for symbol in iteration_space_from_op(op)
    }


def _canonical_key(division: "CoreDivision") -> tuple:
    """A hashable identity for ``division`` within its op's symbol namespace.

    Split keys are the producer's own iteration symbols, so this compares only
    within one operation. It is *total*: the split map, which axes of it are
    reduction axes, and the tiling, so two divisions share a key only when they
    are the same choice. The reduction set is carried rather than derived even
    though one op's write index fixes it, because a clone's menu is not one op's
    namespace -- its entries are synthesized from different consumers' symbols.
    ``tile_splits`` is left out because it is not a further choice: it is the
    tiling resolved to the op's loop symbols, so within one op the tiling fixes it.
    """
    return (
        tuple(sorted(division.splits.items(), key=lambda item: str(item[0]))),
        tuple(sorted(division.reduction_syms, key=str)),
        division.tiling,
    )


def _split_key(key: tuple) -> tuple:
    """``key`` with the tiling dropped -- the part an *edge* compares.

    A per-core view is a function of the splits alone, so two configs differing
    only in their tiling slice the buffer identically and every edge relation
    owes them the same verdict. Projecting here is what lets the enumerated
    pair table, of which the relation reads only the untiled entries, answer
    for a tiled config at all. (What a tiling does change about an edge -- a consumer in another
    tiling group reading the whole output rather than the per-tile scratch --
    is a *cost*, not a compatibility, and is not priced yet.)

    Only the tiling is dropped. A key carrying a fourth element is a menu
    position disambiguated by :meth:`SaCoOptimizingSolver._build_sources`
    because it repeats an earlier entry's split map, and those two entries are
    physically distinct -- keeping them apart is exactly an edge's job, so the
    disambiguator survives the projection.
    """
    return key[:2] + key[3:]


@dataclass(frozen=True, eq=False)
class DivisionConfig:
    """One op's work division as a value -- the annealer's state element.

    ``chosen[i]`` holds one of these rather than a menu position, so a config the
    engine *generates* rather than enumerates is usable wherever a menu entry is.
    Equality and hashing are :attr:`key`'s, which makes two configs equal exactly
    when they are the same *choice*. The key is normally the division's split
    map (:func:`_canonical_key`), so a *generated* config compares equal to the
    menu entry making the same choice, and a generated set can be deduplicated
    or memoized by it.

    The exception is a clone's menu. Its entries are synthesized one per
    consumer out of that consumer's own iteration symbols, which are positional
    and repeat across ops, so two entries can share a split map while slicing
    the buffer differently. Merging them would report a division compatible
    with a clone entry the pair table never checked it against, so a position
    repeating an earlier split map ``position_disambiguates``: its key carries
    ``menu_index`` too. An op's menu is deduplicated by split map, so a buffer
    that generates never needs this.

    The key is derived and never supplied, so it cannot disagree with the
    config it identifies. It is derived once, so ``division``, held by
    reference, is treated as frozen: nothing mutates a division after the menu
    is built, and a mutation would leave the key stale.

    ``menu_index`` is provenance: the position this config came from in its
    buffer's ``core_divisions``, or ``None`` for a *generated* config. Only the
    write-back (see :meth:`SaCoOptimizingSolver._write_back`) and the
    objective's ``sym_division`` binding read it.
    """

    division: "CoreDivision"
    menu_index: Optional[int]
    position_disambiguates: bool = False
    key: tuple = field(init=False)

    def __post_init__(self) -> None:
        key = _canonical_key(self.division)
        if self.position_disambiguates:
            key = (*key, self.menu_index)
        object.__setattr__(self, "key", key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DivisionConfig) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def splits(self) -> dict:
        return self.division.splits

    @property
    def output_partition(self) -> int:
        return self.division.output_partition

    @property
    def tiling(self) -> "TileSpec":
        return self.division.tiling

    @property
    def output_tile_count(self) -> int:
        """Loop tiles the op's own output is cut into, over output axes only --
        the second factor its per-core footprint shrinks by."""
        return self.division.tiling.output_tile_count


def _one_axis_apart(left: "CoreDivision", right: "CoreDivision") -> bool:
    """Whether two divisions differ in exactly one axis's factor.

    Compared over the union of both split maps, since a factor of 1 is dropped
    from a sparse map: ``{d0: 2}`` and ``{d0: 2, d1: 2}`` are one axis apart.
    An axis neither division splits is 1 on both sides and so never counts.
    """

    lhs, rhs = left.splits, right.splits
    return sum(lhs.get(key, 1) != rhs.get(key, 1) for key in lhs | rhs) == 1


class _DivisionSource(ABC):
    """Where one buffer's candidate divisions come from, and what a move may
    reach from a given one.

    The engine asks only this, so it does not care whether the candidates were
    enumerated into a menu (:class:`_MenuDivisions`) or are generated on demand
    (:class:`_GeneratedDivisions`). The engine's seeded generator is the only
    randomness in the search; a source that has to draw (the recolor anchor) is
    handed it rather than holding one.

    The two structural moves ask for different scales. :meth:`neighbours` is
    *one step* -- one axis's factor, or one coarse tile level -- which is what a
    flip takes: ~7 legal factors per axis, a list to pick from rather than an
    interval to propose over with a cooling scale. But an op's legal divisions
    are not connected by one-axis moves (the core budget blocks a factor going
    up, a span floor blocks it coming down), so a search with only local moves
    does worse: +0.9% on the corpus. :meth:`anchor` is the long-range draw that
    pays for it, and recolor is where it belongs, since a flooded region is a
    coordinated change anyway -- and, once tilings are in, since a tiling group
    *is* a region that agrees on one ``TileSpec``.
    """

    @abstractmethod
    def seed(self) -> DivisionConfig:
        """The buffer's committed division: where the search starts."""
        raise NotImplementedError

    @abstractmethod
    def can_move(self) -> bool:
        """Whether this buffer has an alternative division at all -- a static
        filter, so the per-step move draw is over a fixed list. It is only an
        upper bound on :meth:`neighbours`, which is state-dependent."""
        raise NotImplementedError

    @abstractmethod
    def can_split(self) -> bool:
        """Whether this buffer could take a *split* division, the only legal
        recolor anchor. Static, and for a generated source an
        over-approximation; :meth:`anchor` is what actually decides."""
        raise NotImplementedError

    @abstractmethod
    def _step_moves(self, config: DivisionConfig) -> list[DivisionConfig]:
        raise NotImplementedError

    def neighbours(self, config: DivisionConfig) -> list[DivisionConfig]:
        """The divisions one step from ``config`` -- one axis's factor or one
        tile level. Memoized by choice, since a search revisits states."""
        cached = self._neighbour_cache.get(config.key)
        if cached is None:
            cached = self._step_moves(config)
            self._neighbour_cache[config.key] = cached
        return cached

    @abstractmethod
    def retiled(
        self, config: DivisionConfig, tiling: "TileSpec"
    ) -> Optional[DivisionConfig]:
        """``config``'s splits under ``tiling``, or ``None`` if this buffer
        cannot take that tiling at those splits.

        What both run-shaping steps ask: the boundary flip spreading one tiling
        along a run, and the recolor trim putting an op outside the anchor's run
        back to untiled. Splits are held fixed, so this is not a move in the
        division lattice -- a tiling the incoming splits cannot live with is
        refused here rather than silently paired with different ones.
        """
        raise NotImplementedError

    @abstractmethod
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
    """A buffer's divisions as its ``core_divisions`` menu carries them, one
    config per position, each its own choice (see :class:`DivisionConfig`)."""

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

    def can_move(self) -> bool:
        return len(self.configs) > 1

    def can_split(self) -> bool:
        return bool(self.splitting)

    def retiled(
        self, config: DivisionConfig, tiling: "TileSpec"
    ) -> Optional[DivisionConfig]:
        """A menu carries no tilings, so untiled is the only one on offer --
        which is also what makes a menu-backed op break a tiled run rather than
        join it."""
        return config if tiling.is_untiled else None

    def _step_moves(self, config: DivisionConfig) -> list[DivisionConfig]:
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
        """``division`` as a config of this buffer's."""
        key = _canonical_key(division)
        if key not in self._config_cache:
            self._config_cache[key] = DivisionConfig(division, None)
        return self._config_cache[key]

    def can_move(self) -> bool:
        return (
            any(len(factors) > 1 for factors in self.space.factor_domains.values())
            or self.can_tile()
        )

    def can_tile(self) -> bool:
        return self.space.tiling is not None and not self.space.tiling.is_empty

    def can_split(self) -> bool:
        return any(
            axis in self.space.output_axes and any(factor > 1 for factor in factors)
            for axis, factors in self.space.factor_domains.items()
        )

    def retiled(
        self, config: DivisionConfig, tiling: "TileSpec"
    ) -> Optional[DivisionConfig]:
        splits = self.space.splits(config.division)
        if not self.space.admits(splits, tiling):
            return None
        return self.config_for(self.space.division(splits, tiling))

    def _step_moves(self, config: DivisionConfig) -> list[DivisionConfig]:
        return [
            self.config_for(division)
            for division in self.space.neighbours(config.division)
        ]

    def anchor(self, config: DivisionConfig, rng) -> Optional[DivisionConfig]:
        """Redraw the tiling, then every axis, keeping each draw that leaves
        the division legal.

        The generated stand-in for drawing uniformly from a menu's splitting
        entries: it reaches divisions many axis-steps away, and it costs one
        draw and one legality check per axis rather than a walk over the whole
        space. Starting from ``config`` -- which is legal -- means a rejected
        draw simply leaves that axis alone, so the result is always legal.

        The axes go in a random order: a draw is judged against the factors the
        axes after it still hold, so a fixed order would block raising an early
        axis wherever a later one holds the core budget.

        The tiling is drawn *first* because the space is ragged in that order:
        a tile level narrows the axis it cuts, so drawing the splits under the
        chosen tiling reaches states that drawing them first cannot. A space
        with no tiling half draws nothing here and leaves the trajectory of a
        tiling-unaware search untouched.

        For that to hold, the draw is judged on the *tiling's* own legality and
        not against the incoming splits: a tiling whose only legal companions
        are smaller splits -- which is precisely the footprint-shrinking state
        the tiling axis exists to reach -- would otherwise be rejected before
        the redraw that would supply them. Splits the drawn tiling cannot take
        are dropped to all-ones first, so the redraw climbs out of a legal
        state rather than never starting. If even all-ones is illegal there
        (a committed span floor), the tiling is given up instead, where the
        incoming splits are legal untiled -- splits legal under a tiling need
        not be (see ``OpSplitSpace``) -- and the anchor otherwise.
        """
        splits = self.space.splits(config.division)
        tiling = self._draw_tiling(rng)
        if not self.space.admits(splits, tiling):
            floor = dict.fromkeys(self.space.axes, 1)
            if self.space.admits(floor, tiling):
                splits = floor
            elif self.space.admits(splits, _UNTILED):
                tiling = _UNTILED
            else:
                return None
        axes = list(self.space.axes)
        rng.shuffle(axes)
        for axis in axes:
            candidate = dict(splits)
            candidate[axis] = rng.choice(self.space.factor_domains[axis])
            if self.space.admits(candidate, tiling):
                splits = candidate
        division = self.space.division(splits, tiling)
        if division.output_partition <= 1:
            return None
        return self.config_for(division)

    def _draw_tiling(self, rng) -> "TileSpec":
        """A tiling for a recolor to flood, drawn one output dim at a time.

        Untiled is drawn *flat*, at ``_UNTILED_ANCHOR_PROB``, before the per-dim
        draw starts. Per-dim opt-outs alone would leave the untiled anchor at
        the **product** ``prod_d 1/(k_d + 1)`` over the tileable dims -- about
        1/289 at two dims and ``_MAX_SPLITS_PER_DIM`` counts, 1/4913 at three --
        so undividing a region would vanish exactly as the search gained room to
        over-divide it.

        *Whether* to tile is priced: :meth:`SaCoOptimizingSolver._companion_bytes`
        charges the full-extent companion an escaping op needs, so a tiling move
        is no longer accepted unconditionally. *How deep* to tile is priced
        only where the loop re-reads an operand whose index lacks the tiled
        axis (its ``loop_factor``). Otherwise the objective sees depth only
        through a monotone per-core footprint, and the companion charge is a
        function of the buffer's size and its outside readers rather than of
        the tile count, so a drawn tiling is as deep as the draw made it.
        Hence the flat untiled draw: it is what keeps undividing a region
        proposable at all, and recolor is the only long-range move there is.

        Each tileable dim then offers its legal counts plus ``None`` for "leave
        this one alone". A level the space does not admit is dropped, which
        keeps the result a legal tiling without a second pass; whether the
        *splits* can live with it is :meth:`anchor`'s to settle.
        """
        space = self.space.tiling
        if space is None or rng.random() < _UNTILED_ANCHOR_PROB:
            return _UNTILED
        tiling = _UNTILED
        for host_dim in space.output_dims:
            count = rng.choice([None, *space.output_counts[host_dim]])
            if count is None:
                continue
            candidate = TileSpec(
                tiling.axes + (TileAxis(host_dim=host_dim, count=count),)
            )
            if self.space.admits_tiling(candidate):
                tiling = candidate
        return tiling


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

    The table is keyed by menu position and the state is keyed by choice, so it
    is projected onto keys once. That also makes it exact for a generated
    untiled config, which is why a graph where only some ops have a split space
    is not a mixture of two answers: such a division is one the enumeration
    would have carried, so its key is a key the table knows.

    Only pairs of untiled entries are projected (a CP-SAT menu carries tiled
    ones too), onto :func:`_split_key` -- the table answers for a tiled config
    the way it answers for its untiled twin, which is what the geometry says.
    Where the twin is illegal, its splits being legal only in the tiled frame,
    the table does not know it and answers "incompatible": conservative. What it cannot do is *carry* a tiling across the edge: the
    division it hands back is a menu entry, so a flood crossing a table edge
    leaves the far side untiled. That is only ever a lost group, and the edges
    that take this path are the ones whose far side has no tiling space to
    speak of anyway (a clone parent, a non-``ComputedBuffer`` op, a
    division-pinned op).
    """

    pairs: frozenset
    down: dict
    up: dict

    def compatible(self, parent: DivisionConfig, child: DivisionConfig) -> bool:
        return (_split_key(parent.key), _split_key(child.key)) in self.pairs

    def child_for(self, parent: DivisionConfig) -> Optional[DivisionConfig]:
        return self.down.get(_split_key(parent.key))

    def parent_for(self, child: DivisionConfig) -> Optional[DivisionConfig]:
        return self.up.get(_split_key(child.key))


def _table_relation(
    pairs: Iterable[tuple[int, int]],
    parent_menu: _MenuDivisions,
    child_menu: _MenuDivisions,
) -> _TableRelation:
    """Project a menu-position pair table onto choices.

    Each side is its buffer's menu, which every buffer has whether or not its
    own source generates. The propagation direction keeps the pair table's
    tie-break -- the compatible division at the lowest menu position wins --
    which is what makes a flood independent of ``cd_parent_matches`` list order.
    """
    pc, cc = parent_menu.configs, child_menu.configs
    pairs = sorted(
        (ip, ic)
        for ip, ic in set(pairs)
        if pc[ip].tiling.is_untiled and cc[ic].tiling.is_untiled
    )
    key_pairs = frozenset(
        (_split_key(pc[ip].key), _split_key(cc[ic].key)) for ip, ic in pairs
    )
    down: dict = {}
    up: dict = {}
    for ip, ic in sorted(pairs, key=lambda p: (p[1], p[0])):
        down.setdefault(_split_key(pc[ip].key), cc[ic])
    for ip, ic in pairs:
        up.setdefault(_split_key(cc[ic].key), pc[ip])
    return _TableRelation(key_pairs, down, up)


@dataclass
class _ViewRelation(_EdgeRelation):
    """The edge relation computed per candidate, from the buffer's geometry.

    :class:`ResidencyEdge` owns both the view comparison and the residency
    policy filters, and inverts a view to *construct* the division on the other
    end. Memoized by choice, so the pair table this replaces is built lazily and
    only where the search actually looks -- which is the point of generating
    configs rather than enumerating them.

    Propagation picks a *different representative* than :class:`_TableRelation`
    does, and deliberately: the table's tie-break is the lowest menu position,
    while the inverse returns the first solution its own ordering reaches
    (placements by ``(host stride, name)``, then hidden symbols by ascending
    factor). Matching the table would mean exhausting the inversion and ranking
    its solutions by enumeration order -- re-attaching generation to the menu it
    exists to replace. Both picks are compatible and both are deterministic;
    which one a flood is better off with is unmeasured.

    Propagation is memoized on the *whole* key where compatibility is memoized
    on the split half: the division constructed on the far side carries the
    near side's tiling where it can, so which tiling was asked for changes the
    answer even though the verdict does not. It carries none that ``carries``
    refuses: a spec the reader does not read as written would put the two ends
    in different groups (``derive_tiling_groups``) and the trim would strip it.
    """

    edge: "ResidencyEdge"
    parent_source: _GeneratedDivisions
    child_source: _GeneratedDivisions
    carries: Callable[["TileSpec"], bool] = field(
        default=lambda _spec: True, repr=False
    )
    _compatible: dict = field(default_factory=dict, repr=False)
    _down: dict = field(default_factory=dict, repr=False)
    _up: dict = field(default_factory=dict, repr=False)

    def compatible(self, parent: DivisionConfig, child: DivisionConfig) -> bool:
        # Memoized on the split halves alone, matching ``ResidencyEdge`` --
        # the views it compares do not see a tiling.
        pair = (_split_key(parent.key), _split_key(child.key))
        if pair not in self._compatible:
            self._compatible[pair] = self.edge.compatible(
                parent.division.splits, child.division.splits
            )
        return self._compatible[pair]

    def _carried(self, division: "CoreDivision") -> "CoreDivision":
        if self.carries(division.tiling):
            return division
        return replace(division, tiling=_UNTILED)

    def child_for(self, parent: DivisionConfig) -> Optional[DivisionConfig]:
        if parent.key not in self._down:
            division = self.edge.consumer_division_for(
                self._carried(parent.division), self.child_source.space
            )
            self._down[parent.key] = (
                None if division is None else self.child_source.config_for(division)
            )
        return self._down[parent.key]

    def parent_for(self, child: DivisionConfig) -> Optional[DivisionConfig]:
        if child.key not in self._up:
            division = self.edge.parent_division_for(
                self._carried(child.division), self.parent_source.space
            )
            self._up[child.key] = (
                None if division is None else self.parent_source.config_for(division)
            )
        return self._up[child.key]


# The joint state ``(pi, W)`` as :meth:`SaCoOptimizingSolver._snapshot` takes it:
# the packer, the division vector, the eligible count and mask, the relayout copies
# switched on, and the ones eligible.
_Snapshot = tuple[
    NativePermutationLayoutSolver,
    list[DivisionConfig],
    int,
    bytearray,
    set[int],
    set[int],
]


class SaCoOptimizingSolver(CoreDivisionLayoutSolver):
    """SA joint core-division + LX-placement engine.

    Decides LX relayouts: the allocator hands it one ``RelayoutCopyBuffer`` per
    relayout group, the search switches each on or off, and a consumer reading
    a resident parent through a resident copy gets ``chosen_relayouts``.

    The search is fully determined by the module constants above; there is
    nothing to configure per call.

    Args:
        buffers: the buffers to plan, in the allocator's order. Declared as
            ``Sequence[LifetimeBoundBuffer]`` to match the base constructor the
            allocator calls, but every buffer passed must be a
            :class:`CoreDivisionBuffer` -- the engine reads each one's
            ``core_divisions``, its residency relation to each parent
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

    @classmethod
    def supports_paired_buffers(cls) -> bool:
        return False

    @classmethod
    def decides_lx_relayouts(cls) -> bool:
        return True

    @classmethod
    def generates_divisions(cls) -> bool:
        return True

    @classmethod
    def chooses_tilings(cls) -> bool:
        return True

    @classmethod
    def tilings_from_menu(cls) -> bool:
        return False

    @classmethod
    def linear_cost_only(cls) -> bool:
        return False

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
        self._best_snap: _Snapshot
        # Number of buffers passing :meth:`_eligible` under the live ``W``,
        # carried by the two ripple sites as the change over the buffers a move
        # can change.
        self._n_eligible: int
        # The eligibility the packer holds, per buffer (``_note_eligibility``).
        # A ripple reads the old flags of the buffers it changes here instead
        # of re-deriving them, which on a graph with thousands of relayout
        # copies was half of every move's eligibility work.
        self._eligible_mask: bytearray

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
        ``address`` / ``spill_reason`` back to each buffer. Returns the
        solver's own buffers. Single-use: construct a fresh solver per set.
        """
        n = len(self._bufs)
        if n == 0:
            return list(self._bufs)

        self._score_fn = self._build_score_fn(cost_expr)
        if self._score_fn is None:
            logger.info(
                "no usable cost expression; falling back to the memory-only objective"
            )

        self._rng = rnd.Random(_SEED)
        self._precompute_topology()

        # Seed: every op at its committed division; pi from FirstFit.
        self.chosen = self._seed_configs()
        self.packer = self._build_seed_packer()

        self._anneal()
        self._log_tilings()
        self._write_back()
        return list(self._bufs)

    def _build_score_fn(self, cost_expr: Optional[sympy.Expr]):
        """Compile ``cost_expr`` into a ``(chosen, resident) -> fixed-point ns``
        callable, or ``None`` if it can't be evaluated from only this solver's
        own buffers.

        Every free symbol becomes a getter over the live state: a residency
        symbol from whether its buffer's name is in ``resident``, a split symbol
        from ``chosen[idx]`` -- the config itself, so a generated one prices
        exactly as a menu entry does.

        ``None`` (no expression, or a symbol this can't place -- e.g. a dynamic-
        shape symbol the allocator's build left in) falls back to the
        memory-only objective

        A relayout copy's price term is summed directly rather than compiled
        (:meth:`_split_relayout_charges`): it is zero unless the copy is
        resident, and a graph can carry thousands of copies, each otherwise a
        getter and a table lookup on every score.
        """
        if cost_expr is None:
            return None
        value_of: dict = {}  # sympy.Symbol -> (chosen, resident) -> number
        for idx, buf in enumerate(self._bufs):
            value_of[buf.sym_is_lx] = lambda chosen, resident, name=buf.name: (
                1 if name in resident else 0
            )
            # The division's identity, for table terms over candidates (the
            # relayout price is one; see RelayoutCopyBuffer.cost_term): the menu
            # position naming the choice, which a generated config finds by its
            # key. One the menu does not carry binds outside every table, which
            # reads as unpriced (RelayoutCharge).
            value_of[buf.sym_division] = lambda chosen, resident, idx=idx: (
                _UNPRICED
                if (position := self._known_menu_position(idx, chosen[idx])) is None
                else position
            )
            for key, sym in buf.sym_core_divs.items():
                value_of[sym] = lambda chosen, resident, idx=idx, key=key: (
                    chosen[idx].splits.get(key, 1)
                )
            for key, sym in buf.sym_tile_counts.items():
                value_of[sym] = lambda chosen, resident, idx=idx, key=key: (
                    chosen[idx].tiling.count_for(key)
                )
        charges, cost_expr = self._split_relayout_charges(cost_expr)
        try:
            free = sorted(cost_expr.free_symbols, key=str)
            if any(sym not in value_of for sym in free):
                return None
            # lambdify renames a symbol that is not a Python identifier (a copy's
            # name carries ``:``) and searches the whole expression for a fresh
            # name each time; renaming them all here is one pass instead.
            renamed = {
                sym: sympy.Symbol(f"_sa_arg{i}")
                for i, sym in enumerate(free)
                if not str(sym).isidentifier() or keyword.iskeyword(str(sym))
            }
            fn = sympy.lambdify(
                [renamed.get(sym, sym) for sym in free],
                cost_expr.xreplace(renamed) if renamed else cost_expr,
                modules="math",
            )
        except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
            return None

        charge_of = {name: (source, prices) for name, source, prices in charges}
        charged = frozenset(charge_of)
        # Bound once, in argument order. ``value_of`` is keyed by the symbols
        # the buffers hand out, which equal the expression's but are other
        # objects, so every lookup misses the identity fast path for sympy's
        # ``__eq__``: looked up per score, that was 90% of a score's time.
        getters = [value_of[sym] for sym in free]

        def score(chosen, resident) -> int:
            ns = fn(*(getter(chosen, resident) for getter in getters))
            if charged:
                # Over the resident copies only, a handful against the thousands
                # a graph can carry; in name order, so the sum is the same
                # whatever order the sets iterate in.
                extra = 0.0
                for name in sorted(charged.intersection(resident)):
                    source, prices = charge_of[name]
                    position = self._known_menu_position(source, chosen[source])
                    if position is not None and 0 <= position < len(prices):
                        extra += prices[position]
                ns += extra
            return utils.to_fixed_us(max(0.0, ns) / 1000.0)

        return score

    def _split_relayout_charges(
        self, cost_expr: sympy.Expr
    ) -> tuple[list[tuple[str, int, tuple[float, ...]]], sympy.Expr]:
        """``cost_expr``'s top-level ``RelayoutCharge`` terms over this solver's
        own copies, as ``(copy name, source idx, prices)``, and the rest of the
        expression. Summed directly they price exactly as
        ``RelayoutCharge._imp_`` does: the price at the source's menu position
        while the copy is resident, else nothing. A term this cannot place
        stays in the expression, which still prices it."""
        is_lx_of = {
            buf.sym_is_lx: buf.name
            for buf in self._bufs
            if isinstance(buf, RelayoutCopyBuffer)
        }
        if not is_lx_of:
            return [], cost_expr
        division_of = {buf.sym_division: idx for idx, buf in enumerate(self._bufs)}
        charges: list[tuple[str, int, tuple[float, ...]]] = []
        rest = []
        for term in sympy.Add.make_args(cost_expr):
            if (
                isinstance(term, RelayoutCharge)
                and term.args[0] in is_lx_of
                and term.args[1] in division_of
            ):
                is_lx, division, *prices = term.args
                charges.append(
                    (
                        is_lx_of[is_lx],
                        division_of[division],
                        tuple(float(price) for price in prices),
                    )
                )
            else:
                rest.append(term)
        if not charges:
            return [], cost_expr
        return charges, sympy.Add(*rest)

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
        # draw is :meth:`_DivisionSource.anchor`. A relayout copy's one division
        # is its destination's slicing, not a choice, and it has no edge to
        # flood along.
        self._anchor_candidates = [
            i
            for i in range(n)
            if self._sources[i].can_split()
            and not isinstance(bufs[i], RelayoutCopyBuffer)
        ]
        self._build_operation_positions()
        # Whether any buffer can carry a tiling at all, so the companion term
        # costs one bool per score where it cannot -- which is every graph with
        # ``auto_coarse_tiling`` off.
        self._tilings_are_possible = any(
            isinstance(source, _GeneratedDivisions) and source.can_tile()
            for source in self._sources
        )
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
        self._build_relayout_topology()
        self._precompute_spill_costs()

    def _build_relayout_topology(self) -> None:
        """Index the relayout copies (:class:`RelayoutCopyBuffer`) the
        allocator handed over: each copy's source, each source's copies, and
        per edge which copy a consumer reads its parent through under which
        pair of divisions.

        A candidate is keyed by ``(source division, consumer division)`` as
        menu positions, and the consumer's division fixes the view it reads,
        so a pair names at most one group: on a given edge the divisions alone
        decide which copy serves, and the search decides only whether it is
        resident. A pair naming two copies is an enumeration error.
        """
        n = len(self._bufs)
        # copy idx -> source idx. A copy whose source is not in the solve is
        # absent and never eligible (the allocator builds none such).
        self._relayout_source: dict[int, int] = {}
        # source idx -> its copies.
        self._relayout_copies_of: list[list[int]] = [[] for _ in range(n)]
        # copy idx -> the consumers it may serve, in index order.
        self._relayout_consumers: dict[int, list[int]] = {}
        # (source idx, consumer idx) -> {(source pos, consumer pos): (copy idx,
        # the candidate)}.
        self._relayout_reads: dict[
            tuple[int, int], dict[tuple[int, int], tuple[int, RelayoutCandidate]]
        ] = {}
        for idx, buf in enumerate(self._bufs):
            if not isinstance(buf, RelayoutCopyBuffer):
                continue
            source = self._name_to_idx.get(buf.relayout_parent)
            if source is None:
                continue
            self._relayout_source[idx] = source
            self._relayout_copies_of[source].append(idx)
            consumers: set[int] = set()
            for candidate in buf.candidates:
                consumer = self._name_to_idx.get(candidate.consumer)
                # Only an edge the solve carries can be served: the coupling
                # prices the read by that edge's relation.
                if consumer is None or (source, consumer) not in self._relations:
                    continue
                consumers.add(consumer)
                pairs = self._relayout_reads.setdefault((source, consumer), {})
                pair = (candidate.source_division, candidate.consumer_division)
                known = pairs.setdefault(pair, (idx, candidate))
                assert known[0] == idx, (
                    f"{candidate.consumer} reads {buf.relayout_parent} under "
                    f"division pair {pair} through two copies: "
                    f"{self._bufs[known[0]].name} and {buf.name}"
                )
            self._relayout_consumers[idx] = sorted(consumers)
        # Whether each copy could hold its destination at all: no residency pin,
        # and a per-core span that fits. Constant, since a copy's one division
        # is never moved.
        self._relayout_fits = {
            idx: self._bufs[idx].residency_reason is None
            and self._per_core_size(idx, self._sources[idx].seed()) <= self.limit
            for idx in self._relayout_source
        }
        # The copies a relayout move draws from and a seam may buy: the ones
        # that could ever be resident and serve. One that does not fit, whose
        # source is pinned out of LX, or that no consumer in the solve reads,
        # is never eligible, so switching it on would only move divisions for
        # nothing.
        self._relayout_copy_list = sorted(
            idx
            for idx, source in self._relayout_source.items()
            if self._relayout_fits[idx]
            and self._bufs[source].residency_reason is None
            and self._relayout_consumers[idx]
        )
        self._relayout_buyable = frozenset(self._relayout_copy_list)
        # The copies switched on. All start off, so the seed is the plan
        # without relayouts.
        self._relayout_on: set[int] = set()
        # The copies eligible under the current state, kept by every site that
        # sets the packer's eligibility (:meth:`_note_eligibility`): all
        # the post-packing cleanup has to look at (:meth:`_resident_indices`).
        self._eligible_copies: set[int] = set()
        # The occupants the layout moves draw from (:meth:`_draw_layout_buffer`):
        # every buffer but the copies, which join while eligible.
        self._layout_buffers = [
            idx
            for idx, b in enumerate(self._bufs)
            if not isinstance(b, RelayoutCopyBuffer)
        ]
        self._has_copies = len(self._layout_buffers) < n
        if self._relayout_source:
            logger.debug(
                "%d relayout copies over %d sources",
                len(self._relayout_source),
                sum(1 for copies in self._relayout_copies_of if copies),
            )

    def _build_operation_positions(self) -> None:
        """Index the buffers by ``CoreDivisionBuffer.op_position``, both ways.

        A position with no buffer is untiled: nothing can carry a tiling to it.
        A buffer with no position is in no run, so it can hold no tiling:
        :meth:`_retile_boundary` declines and :meth:`_trim_tilings_to_anchor_run`
        strips.
        """
        self._position_of: list[Optional[int]] = []
        self._buffer_at: dict[int, int] = {}
        for idx, buf in enumerate(self._bufs):
            position = buf.op_position
            if position is None:
                self._position_of.append(None)
                continue
            assert position not in self._buffer_at, (
                f"buffers {self._bufs[self._buffer_at[position]].name!r} and "
                f"{buf.name!r} both claim operation position {position}"
            )
            self._buffer_at[position] = idx
            self._position_of.append(position)
        self._n_positions = max(self._buffer_at, default=-1) + 1
        # Reader position -> (earlier producer position, whether it reads that
        # producer tile by tile under a pair of specs), and the same by
        # producer: where a run breaks (see :meth:`_breaks_at`).
        self._tile_reads: dict[int, list[tuple[int, TileReads]]] = {}
        self._tile_readers: dict[int, list[tuple[int, TileReads]]] = {}
        for idx, buf in enumerate(self._bufs):
            reader = self._position_of[idx]
            if reader is None:
                continue
            for name, reads in sorted(buf.tile_reads.items()):
                p_idx = self._name_to_idx.get(name)
                producer = None if p_idx is None else self._position_of[p_idx]
                if producer is None or producer >= reader:
                    continue
                self._tile_reads.setdefault(reader, []).append((producer, reads))
                self._tile_readers.setdefault(producer, []).append((reader, reads))

    def _tiling_at(
        self,
        position: int,
        override: Optional[dict[int, DivisionConfig]] = None,
    ) -> "TileSpec":
        """The tiling in force at one operation position, under ``override`` if
        given. Untiled where no solver buffer is produced there."""
        idx = self._buffer_at.get(position)
        if idx is None:
            return _UNTILED
        config = (
            self.chosen[idx]
            if override is None
            else override.get(idx, self.chosen[idx])
        )
        return config.tiling

    def _breaks_at(
        self,
        position: int,
        stretch: int,
        override: Optional[dict[int, DivisionConfig]] = None,
    ) -> bool:
        """Whether ``derive_tiling_groups`` starts a group at ``position``
        inside a stretch of one nest starting at ``stretch``: it does not read
        some op of the stretch tile by tile, each under its own spec."""
        spec = self._tiling_at(position, override)
        if spec.is_untiled:
            return False
        return any(
            stretch <= producer
            and not reads.aligned(self._tiling_at(producer, override), spec)
            for producer, reads in self._tile_reads.get(position, ())
        )

    def _runs(
        self, override: Optional[dict[int, DivisionConfig]] = None
    ) -> Iterator[tuple[int, int]]:
        """Every run, left to right, under ``override`` if given -- inclusive on
        both ends: the groups ``derive_tiling_groups`` forms, i.e. the maximal
        stretches of positions agreeing on the loop nest, cut where
        :meth:`_breaks_at`.

        Untiled is a spec value like any other, so these runs partition the whole
        operation list. That is what lets a boundary move *create* a tiled region
        rather than only shrink one.
        """
        stretch = 0
        while stretch < self._n_positions:
            nest = self._tiling_at(stretch, override).level_counts
            lo = at = stretch
            while (
                at + 1 < self._n_positions
                and self._tiling_at(at + 1, override).level_counts == nest
            ):
                at += 1
                if self._breaks_at(at, stretch, override):
                    yield lo, at - 1
                    lo = at
            yield lo, at
            stretch = at + 1

    def _run_bounds(
        self,
        position: int,
        override: Optional[dict[int, DivisionConfig]] = None,
    ) -> tuple[int, int]:
        """The run of :meth:`_runs` holding ``position``."""
        return next(run for run in self._runs(override) if position <= run[1])

    def _edge_relation(self, p_idx: int, c_idx: int, p_name: str) -> _EdgeRelation:
        """The relation on the edge ``p_idx -> c_idx``.

        Computed per candidate off the buffer's geometry where both ends
        generate their divisions and the allocator handed over the edge;
        otherwise the pair table, which is what a clone parent, a
        non-``ComputedBuffer`` op and a division-pinned op still have.
        """
        parent_source = self._sources[p_idx]
        child_source = self._sources[c_idx]
        edge = self._bufs[c_idx].residency_edges.get(p_name)
        if (
            edge is not None
            and isinstance(parent_source, _GeneratedDivisions)
            and isinstance(child_source, _GeneratedDivisions)
        ):
            reads = self._bufs[c_idx].tile_reads.get(p_name)
            if reads is None:
                return _ViewRelation(edge, parent_source, child_source)

            def carries(spec: TileSpec) -> bool:
                return reads.aligned(spec, spec)

            return _ViewRelation(edge, parent_source, child_source, carries)
        return _table_relation(
            (
                (int(a), int(b))
                for a, b in self._bufs[c_idx].cd_parent_matches.get(p_name, [])
            ),
            self._menus[p_idx],
            self._menus[c_idx],
        )

    def _build_sources(self) -> None:
        """Build each buffer's division source, and keep its menu: the
        pair-table relations are projected through it, and it is how the
        write-back turns the chosen division back into the position the
        allocator re-indexes."""
        self._menus: list[_MenuDivisions] = []
        self._sources: list[_DivisionSource] = []
        for buf in self._bufs:
            configs: list[DivisionConfig] = []
            by_key: dict = {}
            split_maps: set[tuple] = set()
            for index, cd in enumerate(buf.core_divisions):
                canonical = _canonical_key(cd)
                config = DivisionConfig(cd, index, canonical in split_maps)
                split_maps.add(canonical)
                by_key[config.key] = config
                configs.append(config)
            menu = _MenuDivisions(configs, by_key)
            self._menus.append(menu)
            space = buf.division_space
            self._sources.append(
                menu if space is None else _GeneratedDivisions(space, configs[0])
            )

    def _seed_configs(self) -> list[DivisionConfig]:
        """The seed division vector: every op at its committed division, the
        candidate the allocator enumerates first."""
        return [source.seed() for source in self._sources]

    def _precompute_spill_costs(self) -> None:
        """Cache the loop-invariant inputs to :meth:`_score`. A move changes only
        the *per-core* footprint the packer sees, never a buffer's total size, so
        neither the spill costs nor the bandwidth constant can move."""
        # A relayout copy left out of LX moves nothing: the shuffle does not fire.
        self._spill_costs = [
            0 if isinstance(b, RelayoutCopyBuffer) else self._spill_cost(b)
            for b in self._bufs
        ]
        self._total_spill_cost = sum(self._spill_costs)
        self._names = [b.name for b in self._bufs]
        self._hbm_bytes_per_us = utils.hbm_bytes_per_us()

    # -- division-dependent derivations --------------------------------------

    def _per_core_size(self, idx: int, config: DivisionConfig) -> int:
        """Per-core footprint of buffer ``idx`` under ``config``:
        ``ceil_div(total_size, output_partition * output_tile_count)``, as
        :meth:`CoreDivisionBuffer.per_core_size` computes for a menu position.
        It is computed from ``config`` because a generated config has no
        position (``menu_index`` is ``None``).

        The tile count is the whole payoff channel for tiling. Every
        tiling-sensitive term in the cost model is a derate bounded by 1.0, and
        an untiled op has a working set of 0 by definition, so the objective
        can rank tilings against each other but never above not tiling. What a
        tiling can do is bring this footprint under :attr:`limit` in
        :meth:`_eligible`, which is an engine threshold rather than a cost
        term, and be paid for afterwards in the traffic residency frees.

        Only the per-tile scratch is sized: an escaping op's companion never
        enters LX (``coarse_tiling.apply_chosen_tilings``).

        Clamped non-negative so the packer never sees a negative size from the
        ``mem_usage`` ``-1`` sentinel; what stops an unsized buffer from looking
        *placeable* at zero footprint is
        :meth:`_assert_unsized_buffers_are_pinned`."""
        divisor = config.output_partition * config.output_tile_count
        return max(0, ceil_div(self._bufs[idx].size, divisor))

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
        if isinstance(b, RelayoutCopyBuffer):
            return self._relayout_copy_eligible(idx)
        if self._per_core_size(idx, self.chosen[idx]) > self.limit:
            return False
        parent = self.chosen[idx]
        if self._read_across_a_tiling_boundary(idx, parent):
            return False
        # A child the division pair does not match may still read this buffer
        # from LX, through the relayout copy that pair names.
        return all(
            relation.compatible(parent, self.chosen[c_idx])
            or self._relayout_copy_can_serve(idx, c_idx)
            for c_idx, relation in self._children[idx]
        )

    # -- relayout copies -----------------------------------------------------
    #
    # A RelayoutCopyBuffer is the LX destination of one relayout group, and its
    # residency is the decision that the shuffle fires; its price is the
    # RelayoutCharge term the allocator adds to ``cost_expr``. What the copy
    # cannot carry as data is the coupling: a resident copy needs its source
    # resident under a priced division, and a resident source a consumer does not
    # match needs that consumer's copy resident. The division-only half is
    # eligibility (below), consistent by construction: a copy is eligible only
    # while its source is, and a source only while every copy it needs could
    # hold. The placement half is the packer's outcome, so it is enforced after
    # packing by :meth:`_resident_addresses`.

    def _relayout_read(
        self, source: int, consumer: int
    ) -> Optional[tuple[int, RelayoutCandidate]]:
        """The copy ``consumer`` reads ``source`` through under the current
        divisions and the candidate that prices it, or ``None`` if the pair of
        divisions has none -- including a division the menu does not carry,
        which nothing priced."""
        pairs = self._relayout_reads.get((source, consumer))
        if not pairs:
            return None
        i = self._known_menu_position(source, self.chosen[source])
        j = self._known_menu_position(consumer, self.chosen[consumer])
        if i is None or j is None:
            return None
        return pairs.get((i, j))

    def _relayout_copy_admissible(self, copy_idx: int) -> bool:
        """Whether copy ``copy_idx`` could hold its destination at all, apart
        from its source and its consumers: switched on by the search (see
        :meth:`_toggle_relayout`), under the cost objective only, since the
        memory-only fallback has no price for a shuffle and would take every
        one as free, and at a per-core span that fits."""
        if self._score_fn is None or copy_idx not in self._relayout_on:
            return False
        return self._relayout_fits[copy_idx]

    def _relayout_copy_can_serve(self, source: int, consumer: int) -> bool:
        """Whether ``consumer`` could read ``source`` from LX through a copy
        under the current divisions."""
        read = self._relayout_read(source, consumer)
        return read is not None and self._relayout_copy_admissible(read[0])

    def _relayout_copy_serves(self, copy_idx: int) -> bool:
        """Whether some consumer reads its source through copy ``copy_idx``
        under the current divisions, i.e. does not match the source and names
        this copy. A copy no one reads is a shuffle for nothing."""
        source = self._relayout_source[copy_idx]
        parent = self.chosen[source]
        for consumer in self._relayout_consumers[copy_idx]:
            read = self._relayout_read(source, consumer)
            if (
                read is not None
                and read[0] == copy_idx
                and not self._relations[(source, consumer)].compatible(
                    parent, self.chosen[consumer]
                )
            ):
                return True
        return False

    def _relayout_copy_eligible(self, copy_idx: int) -> bool:
        """:meth:`_eligible` for a relayout copy: admissible, serving someone,
        and its source eligible. The last makes the predicate consistent, so
        when every eligible buffer is placed nothing has to be dropped
        (:meth:`_all_eligible_resident` stays exact)."""
        return self._relayout_copy_eligible_given(copy_idx, None)

    def _relayout_copy_eligible_given(
        self, copy_idx: int, source_eligible: Optional[bool]
    ) -> bool:
        """:meth:`_relayout_copy_eligible`, reusing ``source_eligible`` where the
        caller already evaluated the source in this state."""
        if not self._relayout_copy_admissible(copy_idx):
            return False
        if source_eligible is None:
            source_eligible = self._eligible(self._relayout_source[copy_idx])
        return source_eligible and self._relayout_copy_serves(copy_idx)

    def _eligibility(self, indices: Sequence[int]) -> dict[int, bool]:
        """:meth:`_eligible` over ``indices`` in the current state, evaluating
        each source once rather than again for every copy of it: a ripple over
        a recolored region holds hundreds of copies of a few sources."""
        flags = {
            x: self._eligible(x) for x in indices if x not in self._relayout_source
        }
        for x in indices:
            if x in self._relayout_source:
                flags[x] = self._relayout_copy_eligible_given(
                    x, flags.get(self._relayout_source[x])
                )
        return flags

    def _with_relayout_copies(self, affected: set[int]) -> set[int]:
        """``affected`` plus the switched-on relayout copies of every buffer in
        it: a copy's eligibility follows its source's, and a buffer whose
        eligibility a move can change is in ``affected`` already. A copy that is
        off is ineligible whatever the divisions, and only a toggle, which
        ripples it itself, can change that."""
        if not self._relayout_on:
            return affected
        on = self._relayout_on
        return affected.union(
            *(
                [c for c in self._relayout_copies_of[idx] if c in on]
                for idx in list(affected)
            )
        )

    def _note_eligibility(self, flags: dict[int, bool]) -> None:
        """Record the eligibility ``flags`` set on the packer: in the mask,
        and for the relayout copies among them, in the eligible copies."""
        mask = self._eligible_mask
        for idx, flag in flags.items():
            mask[idx] = flag
        if not self._relayout_source:
            return
        for idx, flag in flags.items():
            if idx in self._relayout_source:
                if flag:
                    self._eligible_copies.add(idx)
                else:
                    self._eligible_copies.discard(idx)

    def _switch_off_stale_copies(self, flags: dict[int, bool]) -> None:
        """Switch off the copies among ``flags`` that a division change left
        serving no one: switched on, and no consumer now reads its source
        through them.

        Such a copy is ineligible already, so nothing placed or counted
        changes; it would only be carried through every later ripple, and the
        anchoring, as a switched-on copy, which on a large graph is most of
        them. Should the divisions return to its pair, a relayout move buys it
        again. Called after a division change only, once any copies the move
        bought are switched on.
        """
        if not self._relayout_on:
            return
        stale = [
            idx
            for idx, flag in flags.items()
            if not flag
            and idx in self._relayout_on
            and not self._relayout_copy_serves(idx)
        ]
        self._relayout_on.difference_update(stale)

    def _resident_addresses(self) -> list[Optional[int]]:
        """``packer.addresses`` with every buffer :meth:`_resident_indices`
        drops set to ``None``: the addresses the write-back commits."""
        addresses = self.packer.addresses
        if not self._relayout_on:
            return addresses
        resident = self._resident_indices()
        return [
            address if i in resident else None for i, address in enumerate(addresses)
        ]

    def _resident_indices(self) -> set[int]:
        """The buffers actually resident: the ones the packer placed, with the
        relayout coupling the packer cannot see enforced.

        Eligibility keeps the coupling over divisions, but under capacity
        pressure the packer can place one side and spill the other. So, only
        ever *removing* what it placed: a source some consumer must read
        through a copy the packer did not place is dropped, then every copy
        whose source is not resident. That is a fixpoint already -- dropping a
        source orphans only its own copies, and dropping a copy strands no
        other source -- and the result does not depend on order. A dropped
        buffer wastes its LX in this state, which the objective prices; it never
        reaches the write-back.

        Only the eligible copies need looking at. A placed source is eligible,
        so each consumer it does not match reads it through a copy that is
        admissible and serves that consumer -- an eligible copy. So a placed
        source loses a consumer's copy exactly when one of its eligible copies
        is not placed; and every placed copy is an eligible one.

        Hot path: the packer reports the placed buffers only
        (``allocated_indices``), a few hundred against the thousands of
        occupants a graph with relayout copies hands it, and the eligible
        copies are a handful.
        """
        placed = set(self.packer.allocated_indices())
        if not self._eligible_copies:
            return placed
        placed.difference_update(
            [self._relayout_source[c] for c in self._eligible_copies if c not in placed]
        )
        placed.difference_update(
            [c for c in self._eligible_copies if self._relayout_source[c] not in placed]
        )
        return placed

    def _read_across_a_tiling_boundary(self, idx: int, parent: DivisionConfig) -> bool:
        """Whether buffer ``idx`` is read by a coarse-tiled consumer at an
        address that advances with the consumer's loop -- which its residency
        would have to express and cannot.

        LX addresses cannot advance: an LX start address is never registered as
        a symbol in the SDSC JSON, so ``affine.apply`` has nothing to target.
        ``compute_ops`` raises "Tiled (advancing) lx-allocated tensors are not
        yet supported" only where it recognizes the advance, and elsewhere the
        compiled graph returns wrong data -- so this is a correctness gate.

        The rule is "this buffer is untiled and some consumer is not". A tiled
        producer is never read across the boundary: in its consumer's run it is
        loop-internal scratch at a fixed address, and outside it the apply
        repoints the consumer at an HBM ``full_buf``. A clone is always untiled.

        A staged tile-local read copy would give the residency back: it does
        not advance, so it can be resident where its source cannot. Nothing
        places one yet.
        """
        if not self._tilings_are_possible or not parent.tiling.is_untiled:
            return False
        return any(
            not self.chosen[child_idx].tiling.is_untiled
            for child_idx, _relation in self._children[idx]
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
                    lifetime_start_override=b.lifetime_start_override,
                    lifetime_end_override=b.lifetime_end_override,
                )
            )
        return out

    def _build_seed_packer(self) -> NativePermutationLayoutSolver:
        """Build the packer for the seed state: the per-core sizes ``chosen``
        implies, a FirstFit-derived ``pi``, and the seed eligibility mask."""
        n = len(self._bufs)
        sizes = [self._per_core_size(i, self.chosen[i]) for i in range(n)]
        eligible = [self._eligible(i) for i in range(n)]
        self._n_eligible = sum(eligible)
        self._eligible_mask = bytearray(n)
        self._note_eligibility(dict(enumerate(eligible)))

        # pi from a FirstFit pass over the per-core sizes. FirstFit leaves the
        # fixed pins unplaced and ``SolverToPermutation`` sorts them after every
        # placed buffer, so they stop displacing eligible buffers upward. They keep
        # a slot, so pi stays a permutation of all n indices and lines up
        # index-for-index with the packer's ``eligible`` mask. Transient,
        # division-dependent ineligibility is deliberately *not* expressed here: it
        # must keep its slot so it can re-enter coherently.
        ff_bufs = self._lifetime_buffers(sizes)
        # FirstFit rejects an in-place child larger than its parent outright; the
        # packer instead skips such a pair per move, as sizes change with the
        # division. Drop those pairs for the seed pass only, as
        # ExhaustiveSearchSolver does (an unsized parent clamps to 0 here).
        for b in ff_bufs:
            child_size = sizes[self._name_to_idx[b.name]]
            b.in_place_parents = [
                p
                for p in b.in_place_parents
                if sizes[self._name_to_idx[p]] >= child_size
            ]
        # Every copy starts off, so ineligible, and an ineligible occupant is
        # transparent to the packer. Left in this pass it would still take an
        # address, and so shape the order of the buffers around it; pinned
        # here, it sorts after them, and the seed pi of the buffers is the one
        # the solve without relayouts starts from.
        for idx in range(n):
            if isinstance(self._bufs[idx], RelayoutCopyBuffer):
                ff_bufs[idx].residency_reason = "relayout copy (off in the seed)"
        # Deep-copied so FirstFit lays out its own objects, never the ones the
        # solver mutates; SolverToPermutation reads addresses back by name.
        pi = SolverToPermutation(
            FirstFitLayoutSolver(copy.deepcopy(ff_bufs), self.limit, self.alignment)
        ).permutation(ff_bufs)

        return NativePermutationLayoutSolver(
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

    def _companion_bytes(self, resident: AbstractSet[int]) -> int:
        """HBM bytes the apply's companion buffers move that the rest of the
        objective does not count, under the current state: the sum of
        :meth:`_companion_charges`.
        """
        return sum(nbytes for _run, nbytes in self._companion_charges(resident))

    def _companion_charges(
        self, resident: AbstractSet[int]
    ) -> Iterator[tuple[tuple[int, int], int]]:
        """``(run, bytes)`` for each escaping buffer of a tiled run: the HBM bytes
        its companion moves that the rest of the objective does not count.

        A tiling shrinks what the buffer holds at once; it does not shrink what
        the buffer moves. For an op whose output escapes its tiling group,
        ``apply_tilings`` allocates a full-extent ``full_buf``, inserts a copy
        op that drains one tile into it per iteration, and repoints every outside
        consumer and any graph output at it (``wsr/coarse_tile.py``). None of
        that is in the features: they were extracted from the untiled graph,
        before ``_post_solve`` applies anything, so the search sees the per-tile
        shrink and neither the copy nor the outside consumers' HBM reads. It
        therefore over-values tiling by a quantity that does not fall with the
        tile count.

        Per escaping buffer of ``size`` bytes, with ``r`` consumers outside its
        run, the difference between the applied graph and the extracted one is:

        * the copy op's read of the per-tile scratch, ``size`` when that scratch
          is in HBM and free when it is resident;
        * the copy op's write of ``full_buf``, ``size`` and always HBM -- except
          for a graph output, whose externally visible write the model already
          charges whether or not the buffer is resident (#4271), so there the
          copy replaces a write already counted and only the op's own write into
          scratch is new;
        * ``r * size`` when the buffer is resident, because those consumers read
          ``full_buf`` from HBM rather than the scratch residency freed them
          from. Not resident, the model already charges them.

        So an op whose consumers all sit inside its own run costs nothing here,
        which is the shape the recolor flood exists to build, while tiling a
        buffer that is not resident costs a full HBM round trip -- both of which
        are the point.

        :meth:`_score` adds this to either objective at the HBM rate, because
        neither can express it: the cost expression is built once from the
        untiled graph, so the full buffer the apply mints is not in it (its tile
        counts price only the loop over each op), and the fallback's spill costs
        are loop-invariant by construction.

        Two known under-corrections, neither expressible from what the solver
        holds: a consumer is counted once however many times it reads, and a
        matmul consumer's ``replication`` (#4454) would make its read of
        ``full_buf`` cost more than one pass.
        """
        if not self._tilings_are_possible:
            return
        # Untiled positions mint no companion.
        tiled = (
            (position, (lo, hi))
            for lo, hi in self._runs()
            if not self._tiling_at(lo).is_untiled
            for position in range(lo, hi + 1)
        )
        for position, (lo, hi) in tiled:
            idx = self._buffer_at[position]
            buf = self._bufs[idx]
            # A buffer with no operation position is in no run, so it is not
            # here; ``_position_of`` being None on a *consumer* means the same
            # thing, and puts it outside every run.
            outside = sum(
                1
                for child_idx, _relation in self._children[idx]
                if not (
                    (child_position := self._position_of[child_idx]) is not None
                    and lo <= child_position <= hi
                )
            )
            is_graph_output = buf.boundary is BufferType.Output
            if not (outside or is_graph_output):
                # Nothing escapes: the apply keeps the buffer as loop-internal
                # scratch and allocates no full buffer at all.
                continue
            size = max(0, buf.size)
            copy_read = 0 if idx in resident else size
            copy_write = copy_read if is_graph_output else size
            outside_reads = size * outside if idx in resident else 0
            yield (lo, hi), copy_read + copy_write + outside_reads

    def _companions(self, resident: AbstractSet[int]) -> int:
        """:meth:`_score`'s term outside ``cost_expr``, in its fixed-point units."""
        return utils.to_fixed_us(
            self._companion_bytes(resident) / self._hbm_bytes_per_us
        )

    def off_expression_ns(self) -> dict[str, float]:
        if not self._bufs:
            return {}
        ns = 1000.0 / utils.US_FIXED_POINT_SCALE
        return {"companions": self._companions(self._resident_indices()) * ns}

    def score_ns(self) -> Optional[float]:
        if not self._bufs:
            return None
        return self._score() * 1000.0 / utils.US_FIXED_POINT_SCALE

    def _score(self) -> int:
        """The shared objective for the current state, in integer fixed-point
        time units, over the buffers :meth:`_resident_indices` reports resident
        (the placed ones, less any the relayout coupling drops).

        Hot path: asks the packer once, for the placed buffers only. The native
        packer materializes a fresh list per query, so a per-buffer read inside
        the loop was quadratic; hoisting it is 8-31x faster on the captures.

        The memory-only fallback is *differential* -- ``spill_cost`` is the traffic
        a spill adds **over** residency -- so a resident buffer contributes zero
        and only spilled ones are summed, the same shape as the CP-SAT engine's
        ``spill_cost() * (1 - in_buffer)``.

        Both add :meth:`_companion_bytes` at the HBM rate.
        """
        resident_idx = self._resident_indices()
        companions = self._companions(resident_idx)
        if self._score_fn is not None:
            resident = frozenset(self._names[i] for i in resident_idx)
            return self._score_fn(self.chosen, resident) + companions
        # Spill costs are integers, so summing the resident ones off the total
        # is exact whatever the order.
        traffic = self._total_spill_cost - sum(
            self._spill_costs[i] for i in resident_idx
        )
        return utils.to_fixed_us(traffic / self._hbm_bytes_per_us) + companions

    # -- moves ---------------------------------------------------------------

    def _flippable(self) -> list[int]:
        """Buffer indices whose division source offers an alternative at all.

        Static, so the per-step draw is over a fixed list; whether the division
        the buffer currently holds has a *neighbour* is decided at move time."""
        return [i for i in range(len(self._bufs)) if self._sources[i].can_move()]

    def _atomic_flip(
        self, idx: int, config: DivisionConfig, *, buy_seams: bool = False
    ) -> set[int]:
        """Change buffer ``idx``'s division to ``config`` and ripple: resize its
        per-core footprint, then refresh eligibility for ``idx`` and its parents.
        Those are the only buffers a flip can change, since eligibility depends on
        an op's own division and its children's -- plus their relayout copies,
        whose eligibility follows their source's. The one-op case of
        :meth:`_apply_assignment`, whose ripple is exactly that. Returns the
        seams it bought."""
        return self._apply_assignment({idx: config}, buy_seams=buy_seams)

    def _retile_boundary(self, idx: int, config: DivisionConfig) -> set[int]:
        """Spread ``config``'s tiling from ``idx`` to one end of its run.

        The tiling arm of flip. ``idx`` sits in a uniform run ``A..Z``; this
        re-specs ``A..H`` or ``H..Z`` (inclusive), so the run splits in two, or
        -- where the new spec matches the neighbouring run's -- the boundary
        between them slides. What it writes is one contiguous stretch, and
        because untiled is a spec value like any other, the move creates tiled
        regions as readily as it shrinks them.

        **Truncating, not rejecting.** Whether an op can take a tiling is a
        per-op question (``OpSplitSpace.neighbours`` only offers a level the op's
        current splits survive), so over a run of any length the odds that every
        member agrees fall off fast. Stopping at the first op that refuses reads
        as sliding the boundary as far as it will go, and keeps the sub-run
        contiguous; rejecting the whole move would make long runs nearly immovable.
        It does not keep the old run whole: stopped short of ``Z``, it leaves
        ``[S][S'][S]``, and the apply forms the two ``S`` runs as separate groups,
        as it forms any two runs.

        An operation that produces no solver buffer stops the walk for the same
        reason it breaks a run: nothing can carry a tiling to it. So does an op
        the new spec would cut from the stretch written so far
        (:meth:`_misread_across`).

        The single-op flip survives as the degenerate case -- ``H`` at a run end
        -- and a mid-run split is two steps rather than one.

        Returns the seams it bought.
        """
        position = self._position_of[idx]
        if position is None:
            return set()  # no operation position, so no run to move a boundary in
        lo, hi = self._run_bounds(position)
        # The only randomness this move draws beyond the neighbour choice, and it
        # is drawn only here: a space with no tiling half never reaches this arm,
        # which is what keeps a tiling-unaware trajectory identical.
        forward = self._rng.random() < 0.5
        step, end = (1, hi) if forward else (-1, lo)
        assignment: dict[int, DivisionConfig] = {}
        at = position
        while True:
            target = self._buffer_at.get(at)
            if target is None:
                break
            if at != position and self._misread_across(
                at, position, config.tiling, forward
            ):
                break
            retiled = self._sources[target].retiled(self.chosen[target], config.tiling)
            if retiled is None:
                break
            assignment[target] = retiled
            if at == end:
                break
            at += step
        if not assignment:
            return set()
        return self._apply_assignment(assignment, buy_seams=True)

    def _misread_across(
        self, at: int, start: int, spec: "TileSpec", forward: bool
    ) -> bool:
        """Whether ``spec`` would cut ``at`` from the stretch it extends,
        ``start..at - 1`` walking forward or ``at + 1..start`` walking back,
        every op of which the walk gives ``spec``: :meth:`_breaks_at` over the
        reads ``at`` adds."""
        if spec.is_untiled:
            return False
        if forward:
            edges = [r for p, r in self._tile_reads.get(at, ()) if start <= p]
        else:
            edges = [r for p, r in self._tile_readers.get(at, ()) if p <= start]
        return any(not reads.aligned(spec, spec) for reads in edges)

    def _flood_region(
        self, anchor: int, config: DivisionConfig
    ) -> dict[int, DivisionConfig]:
        """Flood the residency relation from ``(anchor, config)`` to a config
        assignment over the reachable region.

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

    def _trim_tilings_to_anchor_run(
        self, anchor: int, assignment: dict[int, DivisionConfig]
    ) -> dict[int, DivisionConfig]:
        """Strip the ``TileSpec`` from every assigned op outside the anchor's
        contiguous run, leaving its splits alone.

        The flood's reach is the residency relation's, which is producer /
        consumer reachability; a coarse-tiling group has to be a contiguous run
        of the operation list. A region straddling an op the relation could not
        carry the tiling to would leave tiled runs apart from the anchor's, each
        its own group at apply; one reading another's output reads its *full*
        extent, through a companion buffer. So a recolor tiles the anchor's run
        only; this strips tilings and moves nothing else. The divisions are
        deliberately untouched -- they are what the flood is for, and narrowing
        them to the run would cost the long-range division move.

        An op whose splits are legal only under its tiling (see
        ``OpSplitSpace``) keeps it, and so forms a group of its own; it was
        already outside the anchor's run with that tiling, so the run's bounds
        stand.

        One pass suffices. If the anchor's run is tiled, stripping ops outside it
        to untiled can only keep them differing from it, so the boundary does not
        move; if it is untiled, everything tiled is outside and all of it goes.
        """
        if all(config.tiling.is_untiled for config in assignment.values()):
            return assignment
        position = self._position_of[anchor]
        # An anchor with no operation position (an input clone) is in no run, so
        # nothing in the region may stay tiled. No position is ever in [-1, -1].
        lo, hi = (
            (-1, -1) if position is None else self._run_bounds(position, assignment)
        )
        trimmed = dict(assignment)
        for idx, config in assignment.items():
            if config.tiling.is_untiled:
                continue
            at = self._position_of[idx]
            if at is not None and lo <= at <= hi:
                continue
            untiled = self._sources[idx].retiled(config, _UNTILED)
            if untiled is not None:
                trimmed[idx] = untiled
        return trimmed

    def _apply_assignment(
        self,
        assignment: dict[int, DivisionConfig],
        *,
        buy_seams: bool = False,
        switch_on: Sequence[int] = (),
    ) -> set[int]:
        """Commit a multi-op division assignment: set every op's division, resize
        its footprint, and refresh eligibility for the assigned set plus their
        parents (the same ripple as a flip, unioned over the set).

        Every structural move lands here -- a flooded region coloring, a
        boundary flip's sub-run, a single-op flip and a relayout move. With
        ``buy_seams`` it also switches on the copies the new divisions need
        (:meth:`_seam_copies`), and the copies in ``switch_on``, all in the same
        one re-place. Returns the seams it switched on, apart from
        ``switch_on``, for :meth:`_settle` to judge.
        """
        # The affected set is division-invariant, so it is built (and its old
        # eligibility counted, off the mask) before the coloring lands. A copy
        # switched on here was off, so ineligible before; its source is in the
        # set.
        affected = set(assignment)
        for op in assignment:
            affected |= self._parents_idx[op]
        affected.update(self._relayout_source[c] for c in switch_on)
        affected = self._with_relayout_copies(affected)
        before = sum(self._eligible_mask[x] for x in affected)
        for op, config in assignment.items():
            self.chosen[op] = config
        seams = self._seam_copies(assignment) if buy_seams else set()
        seams.difference_update(switch_on)
        bought = (seams | set(switch_on)) - self._relayout_on
        self._relayout_on |= bought
        affected_sorted = sorted(affected | bought)
        resizes = [
            (op, self._per_core_size(op, self.chosen[op])) for op in sorted(assignment)
        ]
        flags = self._eligibility(affected_sorted)
        # One re-place for the whole region rather than one per buffer.
        self.packer.update(resizes, [(x, flags[x]) for x in affected_sorted])
        self._note_eligibility(flags)
        self._switch_off_stale_copies(flags)
        self._n_eligible += sum(flags.values()) - before
        return seams

    def _toggle_relayout(self, copy_idx: int) -> None:
        """Switch relayout copy ``copy_idx`` on or off, and ripple.

        Residency cannot decide a shuffle on its own: the packer places every
        eligible buffer that fits, which is right for a buffer whose residency
        only saves traffic, and wrong for a copy, whose residency costs its
        price. Off, the copy is ineligible, and so is its source wherever a
        consumer reads it only through this copy. On, both may be placed.
        """
        self._switch_relayouts({copy_idx}, on=copy_idx not in self._relayout_on)

    def _switch_relayouts(self, copies: set[int], *, on: bool) -> None:
        """Switch every copy in ``copies`` on or off, and ripple, in one
        re-place.

        A switch changes the eligibility of the copy and its source, and the
        source's decides its other copies' (:meth:`_relayout_copy_eligible`).
        """
        sources = {self._relayout_source[c] for c in copies}
        affected = sorted(self._with_relayout_copies(sources) | copies)
        before = sum(self._eligible_mask[x] for x in affected)
        if on:
            self._relayout_on |= copies
        else:
            self._relayout_on -= copies
        flags = self._eligibility(affected)
        self.packer.update([], [(x, flags[x]) for x in affected])
        self._note_eligibility(flags)
        self._n_eligible += sum(flags.values()) - before

    def _relayout(self) -> None:
        """One relayout move: switch a uniformly drawn copy off, or on together
        with the divisions it is priced for, then burst.

        Switching a copy on alone rarely pays: a consumer reads through it only
        under one of its candidates' division pairs, and those are few among the
        pairs an edge can take, so the move that would make it pay is a second,
        independent one, and each half on its own is worth nothing to the score.
        So switching on also moves the copy's source and every consumer it can
        serve to one of its priced pairs (:meth:`_relayout_assignment`), and
        buys every seam that change opens (:meth:`_seam_copies`) -- the whole
        group, and the copies the new divisions need around it, bought in one
        re-place and judged as a unit, as recolor judges a flooded region. The
        drawn copy is the move; the seams around it are kept only where they
        pay (:meth:`_settle`).
        """
        copy_idx = self._rng.choice(self._relayout_copy_list)
        if copy_idx in self._relayout_on:
            self._toggle_relayout(copy_idx)
            self._burst()
        else:
            self._settle(
                self._apply_assignment(
                    self._relayout_assignment(copy_idx),
                    buy_seams=True,
                    switch_on=(copy_idx,),
                )
            )

    def _relayout_assignment(self, copy_idx: int) -> dict[int, DivisionConfig]:
        """The divisions that make copy ``copy_idx`` serve: its source at one of
        the source divisions its candidates price, and each consumer at a
        division priced into this group from that one -- its current division
        where that is priced, the first in candidate order otherwise.

        The source division is drawn uniformly among those that move the
        fewest *anchored* buffers: ones another eligible copy serves or reads
        from under their current divisions. Moving one unprices that copy,
        which is how a chain of relayouts (a buffer that is the consumer of one
        copy and the source of the next) used to lose one link for every link
        it gained.

        Only the consumers the copy can serve (``_relayout_consumers``, built
        with the topology): a candidate whose edge the solve does not carry
        would move a consumer no copy reads for."""
        copy = cast(RelayoutCopyBuffer, self._bufs[copy_idx])
        source = self._relayout_source[copy_idx]
        served = set(self._relayout_consumers[copy_idx])
        anchored = self._relayout_anchored(exclude=copy_idx)
        by_division: dict[int, dict[int, DivisionConfig]] = {}
        for candidate in copy.candidates:
            consumer = self._name_to_idx.get(candidate.consumer)
            if consumer not in served:
                continue
            assignment = by_division.setdefault(
                candidate.source_division,
                {source: self._menus[source].configs[candidate.source_division]},
            )
            config = self._menus[consumer].configs[candidate.consumer_division]
            if consumer not in assignment or config == self.chosen[consumer]:
                assignment[consumer] = config
        disturbance = {
            division: sum(
                idx in anchored and config != self.chosen[idx]
                for idx, config in assignment.items()
            )
            for division, assignment in by_division.items()
        }
        least = min(disturbance.values())
        division = self._rng.choice(
            sorted(d for d, moved in disturbance.items() if moved == least)
        )
        return by_division[division]

    def _relayout_anchored(self, exclude: int) -> set[int]:
        """The buffers whose current division an eligible copy other than
        ``exclude`` is priced for: its source and the consumers reading it.

        Eligible, not merely switched on: a copy that is on while its source is
        ineligible for another reason (too big at its division, another
        mismatched child with no copy) delivers nothing, and anchoring for it
        would steer the draw away from divisions no paying relayout depends
        on. Not resident either: placement follows ``pi``, which the burst
        after the move adjusts."""
        anchored: set[int] = set()
        for copy_idx in sorted(self._eligible_copies):
            if copy_idx == exclude:
                continue
            source = self._relayout_source[copy_idx]
            for consumer in self._relayout_consumers[copy_idx]:
                read = self._relayout_read(source, consumer)
                if read is not None and read[0] == copy_idx:
                    anchored.update((source, consumer))
        return anchored

    def _seam_copies(self, assignment: dict[int, DivisionConfig]) -> set[int]:
        """The switched-off copies an edge the ``assignment`` changed now needs,
        under the divisions it set -- after any structural move, not only a
        relayout one: the edges out of each reassigned buffer and into it from
        its parents that no longer match, where the new pair of divisions names
        a copy that could be resident.

        Without them, the next link of a chain, or a source whose reassignment
        mismatches several consumers, stays ineligible until another relayout
        move happens to draw exactly that copy, and that move alone is worth
        nothing to the score. A seam with no priced copy is left as it is: the
        source goes out of LX, which the score judges. Under the memory-only
        fallback there is no price for a shuffle, so none is bought.

        Only edges with an end in ``assignment``: an edge between a parent and
        a sibling of a reassigned buffer has the pair of divisions it had, so a
        copy the search switched off there stays off. Whether a bought seam
        pays is :meth:`_settle`'s call."""
        if self._score_fn is None or not self._relayout_buyable:
            return set()
        touched = set(assignment)
        for idx in assignment:
            touched |= self._parents_idx[idx]
        seams: set[int] = set()
        for parent in touched:
            if not self._relayout_copies_of[parent]:
                continue
            config = self.chosen[parent]
            for child, relation in self._children[parent]:
                if parent not in assignment and child not in assignment:
                    continue
                if relation.compatible(config, self.chosen[child]):
                    continue
                read = self._relayout_read(parent, child)
                if (
                    read is not None
                    and read[0] in self._relayout_buyable
                    and read[0] not in self._relayout_on
                ):
                    seams.add(read[0])
        return seams

    def _recolor(self) -> None:
        """One region-recolor move: a uniform anchor op (so a region is hit
        ∝ its op-count), a random splitting anchor division, flood, recolor,
        burst.

        The search's long-range move (see :class:`_DivisionSource`). A draw
        that came out unsplit is a no-op step."""
        anchor = self._rng.choice(self._anchor_candidates)
        config = self._sources[anchor].anchor(self.chosen[anchor], self._rng)
        if config is None:
            return
        assignment = self._trim_tilings_to_anchor_run(
            anchor, self._flood_region(anchor, config)
        )
        # The flood stops at every edge with no compatible pair, and those
        # seams are where a relayout can stand: buying them turns an accepted
        # seam into a priced choice rather than a source spilled.
        self._settle(self._apply_assignment(assignment, buy_seams=True))

    def _settle(self, seams: set[int]) -> None:
        """Burst, then judge the ``seams`` a structural move bought as a unit:
        keep them only if the state scores no worse with them than without.

        A seam is a choice between two plans for its edge: the copy on and its
        price paid, or the copy off and its source spilled. Bought blind, every
        structural move is scored as the first, so a move whose own gain beats
        the spill but not the shuffle is rejected for a price it need not pay.
        A per-copy price test (price below the source's spill) cannot replace
        this: two copies of one source can each pass it and together cost more
        than the spill, and the spill a copy saves depends on the source's
        other consumers. Scoring both is exact, for one more re-place and score
        on the moves that open a seam. A subset of the seams stays reachable
        through the relayout move.

        The variant without the seams keeps the burst's ``pi``: switching
        copies off only removes eligible buffers, so the packer re-places what
        is left from the same order. Draws no randomness, so a move that buys
        no seam -- every move on a graph without copies -- is just the burst.
        """
        self._burst()
        if not seams:
            return
        with_seams = self._score()
        snap = self._snapshot()
        self._switch_relayouts(seams, on=False)
        if self._score() > with_seams:
            self._adopt(snap)

    def _draw_layout_buffer(self) -> int:
        """A buffer for a layout move, uniformly over the buffers and the
        eligible relayout copies.

        A copy that is not eligible is transparent to the packer: moving it, or
        moving a buffer past it, changes no address. A graph with copies has
        many times more of them than buffers, and almost all are off, so a draw
        over every occupant would spend most layout moves on nothing. An
        ineligible *buffer* stays in the draw, as on a graph without copies.
        """
        movers = self._layout_buffers
        r = self._rng.randrange(len(movers) + len(self._eligible_copies))
        if r < len(movers):
            return movers[r]
        return sorted(self._eligible_copies)[r - len(movers)]

    def _draw_layout_position(self) -> int:
        """A permutation position for a layout move: the position of
        :meth:`_draw_layout_buffer`'s buffer, or, on a graph without copies,
        any position (the same draw, without the lookup)."""
        if not self._has_copies:
            return self._rng.randrange(len(self._bufs))
        return self.packer.position(self._draw_layout_buffer())

    def _layout_positions(self) -> list[int]:
        """The permutation positions :meth:`_draw_layout_buffer` can land on,
        in position order."""
        if not self._has_copies:
            return list(range(len(self._bufs)))
        movers = self._layout_buffers + sorted(self._eligible_copies)
        return sorted(self.packer.position(idx) for idx in movers)

    def _burst(self) -> None:
        """A short cold layout burst: greedily accept layout steps that do not
        lower the packer's quality, letting ``pi`` adapt to the new footprints
        before the compound move is judged.

        Rejected steps are reverted rather than snapshotted, since
        ``rotate(j, i)`` undoes ``rotate(i, j)``.
        """
        if len(self._bufs) < 2:
            return
        # Per buffer, as the step budget is: a copy carries no decision.
        # The floor of 1 is there so a small graph still gets a burst.
        for _ in range(max(1, int(_BURST_FRACTION * len(self._layout_buffers)))):
            # Nothing left for pi to win once the structural move has left every
            # eligible buffer resident; the rest of the burst is noise.
            if self._all_eligible_resident():
                return
            i = self._draw_layout_position()
            j = self._draw_layout_position()
            if self.packer.rotate(i, j) < 0:
                self.packer.rotate(j, i)  # revert

    # -- state snapshots -----------------------------------------------------

    def _snapshot(self) -> _Snapshot:
        """An independent copy of the joint state ``(pi, W)``: the packer's
        dynamic layout (``copy`` shares only plan-lifetime structures) plus the
        division vector, and the eligible count and mask ``W`` implies --
        rebuilding those from ``W`` would cost an O(n) pass of ``_eligible`` the
        restore does not otherwise need -- the relayout copies switched on,
        which nothing else can rebuild, and the ones eligible, kept for the same
        reason as the count."""
        return (
            self.packer.copy(),
            list(self.chosen),
            self._n_eligible,
            bytearray(self._eligible_mask),
            set(self._relayout_on),
            set(self._eligible_copies),
        )

    def _adopt(self, snap: _Snapshot) -> None:
        """Install ``snap`` as the live state by *taking ownership* of it -- no
        copy, so the engine goes on mutating those objects and the caller must
        treat ``snap`` as dead from here on. Zero-copy because a step already pays
        one O(n) packer copy for its snapshot.
        """
        (
            self.packer,
            self.chosen,
            self._n_eligible,
            self._eligible_mask,
            self._relayout_on,
            self._eligible_copies,
        ) = snap

    # -- move selection & execution -----------------------------------------

    def _applicable_moves(self) -> list[str]:
        """Move types available this step, in fixed (deterministic) order: reorder
        needs >=2 buffers, flip a multi-entry menu, recolor a non-trivial anchor,
        relayout a relayout copy and the cost objective that prices it (the
        memory-only fallback keeps every copy off).

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
        if self._relayout_copy_list and self._score_fn is not None:
            moves.append("relayout")
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
        if name == "reorder":
            i = self._draw_layout_position()
            self.packer.rotate(i, self._draw_layout_position())
        elif name == "flip":
            idx = self._rng.choice(self._flippable_ops)
            # One axis's factor, or one coarse tile level, drawn uniformly
            # from the divisions a step away.
            options = self._sources[idx].neighbours(self.chosen[idx])
            if not options:
                return
            config = self._rng.choice(options)
            # Two arms, two scopes. A step in the division lattice is this op's
            # alone; a step in the tiling lattice moves a *boundary*, re-speccing
            # a stretch that reaches one end of the op's run rather than the op
            # alone (see ``_retile_boundary`` for where it stops short).
            if config.tiling == self.chosen[idx].tiling:
                seams = self._atomic_flip(idx, config, buy_seams=True)
            else:
                seams = self._retile_boundary(idx, config)
            self._settle(seams)
        elif name == "recolor":
            self._recolor()
        elif name == "relayout":
            self._relayout()
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
        -- the ones the objective prices. Over the positions a layout move
        draws (:meth:`_layout_positions`): a copy that is off is never
        allocated, and would otherwise take the larger part of the draws."""
        positions = self._layout_positions()
        n = len(positions)
        n_allocated = sum(1 for p in positions if allocated[p])
        return self._rng.choices(
            positions,
            weights=[n if allocated[p] else n_allocated + 1 for p in positions],
        )[0]

    def _sweep_upper_bound(self, i: int, allocated: list[bool]) -> int:
        """Highest reinsertion position worth probing for the buffer at position
        ``i`` -- the layout-only annealer's monotonicity bound.

        A buffer's address is non-decreasing in its position, so one that is *not*
        legally allocated can only be made to fit by moving earlier: past the last
        legally-allocated position, nothing it reaches changes the outcome. An
        allocated buffer has no such bound and sweeps to the end.

        Nor past the last occupant a layout move draws: everything after it is
        a copy that is off, which no address depends on, so every position
        there is the one just after that occupant.
        """
        n = len(allocated)
        upper = n - 1
        if not allocated[i]:
            last = max((pos for pos, a in enumerate(allocated) if a), default=0)
            upper = min(upper, last + 1)
        if self._has_copies:
            last = max((p for p in self._layout_positions() if p != i), default=0)
            # Lifting ``i`` shifts an occupant after it down by one.
            upper = min(upper, last if i < last else last + 1)
        return upper

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
        # Per decision, not per slot: a relayout copy carries no division
        # decision (which one an edge reads is the divisions' call), and a graph
        # can carry thousands.
        n = sum(not isinstance(b, RelayoutCopyBuffer) for b in self._bufs)
        steps = _step_budget(n)
        if steps < _STEPS_PER_BUFFER * n:
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

    def _log_tilings(self) -> None:
        """Report the tiled runs the search kept, each with its companion
        traffic, or that it kept none although some buffer could tile."""
        if not self._tilings_are_possible:
            return
        charges: dict[tuple[int, int], int] = {}
        for run, nbytes in self._companion_charges(self._resident_indices()):
            charges[run] = charges.get(run, 0) + nbytes
        tiled = [run for run in self._runs() if not self._tiling_at(run[0]).is_untiled]
        if not tiled:
            logger.info(
                "SA co-optimizer kept no coarse tiling; %d buffer(s) could tile",
                sum(
                    isinstance(source, _GeneratedDivisions) and source.can_tile()
                    for source in self._sources
                ),
            )
        for lo, hi in tiled:
            logger.debug(
                "SA co-optimizer tiled %s at %s; companion HBM traffic %d bytes",
                ", ".join(
                    self._bufs[self._buffer_at[position]].name
                    for position in range(lo, hi + 1)
                ),
                self._tiling_at(lo).label,
                charges.get((lo, hi), 0),
            )

    def _log_coupling_drops(self, addresses: list[Optional[int]]) -> None:
        """Log the buffers the packer placed but the committed plan drops for
        the relayout coupling (:meth:`_resident_indices`), with the LX they held.

        The coupling is enforced after packing, so a dropped buffer's range
        still pushed others out while the packer ran. No final plan of the
        measured attention graphs drops anything, so this is the signal to look
        for before teaching the packer the coupling."""
        if not self._relayout_on:
            return
        dropped = sorted(
            i for i in self.packer.allocated_indices() if addresses[i] is None
        )
        if dropped:
            logger.debug(
                "relayout coupling dropped %d placed buffer(s), %d B of LX: %s",
                len(dropped),
                sum(self._per_core_size(i, self.chosen[i]) for i in dropped),
                ", ".join(self._bufs[i].name for i in dropped),
            )

    def _write_back(self) -> None:
        """Commit the best state to the buffers and record spill causes.

        ``chosen_division`` is a position in ``core_divisions``, which is the
        allocator's contract and the one place a generated division has to be
        given a position. A generated division is normally one the enumeration
        already carries, so the position is looked up by choice; one the menu
        does not carry (a tiled one, once the search chooses tilings) is
        appended. Appending here rather than when the division is proposed keeps
        the menu the search ran against unchanged.

        A consumer reading a resident parent it does not match reads it through
        the relayout copy its division pair names, which is resident too
        (:meth:`_resident_addresses`); that read is recorded on the consumer
        with the copy's address, as ``CpSatLayoutSolver`` records it, for the
        commit path to materialize.

        Both checks run before any buffer is touched, and raise rather than
        assert: a failure would otherwise commit a consumer reading LX in the
        wrong layout, and ``python -O`` strips an ``assert``.
        """
        self._check_tracked_eligibility()
        addresses = self._resident_addresses()
        self._log_coupling_drops(addresses)
        reads: list[tuple[int, int, int, RelayoutCandidate]] = []
        for source, copies in enumerate(self._relayout_copies_of):
            if not copies or addresses[source] is None:
                continue
            parent = self.chosen[source]
            for consumer, relation in self._children[source]:
                if relation.compatible(parent, self.chosen[consumer]):
                    continue
                read = self._relayout_read(source, consumer)
                if read is None or addresses[read[0]] is None:
                    raise RuntimeError(
                        f"SA co-optimizer: {self._bufs[source].name} is resident "
                        f"but {self._bufs[consumer].name} has no resident copy to "
                        "read it through"
                    )
                reads.append((source, consumer, *read))
        for i, b in enumerate(self._bufs):
            addr = addresses[i]
            b.chosen_division = self._menu_position(i, self.chosen[i])
            # Replaced, not merged, as ``CpSatLayoutSolver`` does: only this
            # solve's reads are recorded below.
            b.chosen_relayouts = {}
            b.address = addr
            spilled = b.residency_reason or _SOLVER_CHOSE_SPILL
            b.spill_reason = spilled if addr is None else None
        for source, consumer, copy_idx, candidate in reads:
            self._bufs[consumer].chosen_relayouts[self._bufs[source].name] = (
                ChosenRelayout(candidate, cast(int, addresses[copy_idx]))
            )

    def _check_tracked_eligibility(self) -> None:
        """Recount eligibility from scratch and raise if the tracked state
        disagrees.

        The anneal keeps the eligibility mask, the eligible copies and their
        count incrementally: each move updates only the buffers it can change.
        The coupling cleanup (:meth:`_resident_indices`) reads the eligible
        copies alone, so a buffer a move missed would let a source stay
        resident without the copy a consumer reads it through. A disagreement
        also means the search scored states under the wrong eligibility, so it
        is an error in the tracking, never repaired here. One O(n) pass, next
        to an anneal of thousands of moves.
        """
        n = len(self._bufs)
        recount = [self._eligible(i) for i in range(n)]
        wrong = {i for i in range(n) if bool(self._eligible_mask[i]) != recount[i]}
        wrong |= self._eligible_copies ^ {
            c for c in self._relayout_source if recount[c]
        }
        if not wrong and self._n_eligible == sum(recount):
            return
        names = sorted(self._bufs[i].name for i in wrong)
        logger.error(
            "SA co-optimizer: tracked eligibility drifted from a recount "
            "(eligible count %d, recount %d; buffers that differ: %s)",
            self._n_eligible,
            sum(recount),
            names,
        )
        raise RuntimeError(
            f"SA co-optimizer: tracked eligibility drifted from a recount: {names}"
        )

    def _known_menu_position(self, idx: int, config: DivisionConfig) -> Optional[int]:
        """The position in buffer ``idx``'s ``core_divisions`` that names
        ``config``'s division, or ``None`` if the menu carries no such entry.
        What the allocator indexed anything per division by -- the relayout
        candidates and their prices among them."""
        if config.menu_index is not None:
            return config.menu_index
        known = self._menus[idx].by_key.get(config.key)
        return None if known is None else known.menu_index

    def _menu_position(self, idx: int, config: DivisionConfig) -> int:
        """The position in buffer ``idx``'s ``core_divisions`` that names
        ``config``'s division, registering it if the menu has no such entry."""
        known = self._known_menu_position(idx, config)
        if known is not None:
            return known
        divisions = self._bufs[idx].core_divisions
        divisions.append(config.division)
        return len(divisions) - 1
