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

from torch_spyre._inductor.work_division import (
    UNTILED as _UNTILED,
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
    TileAxis,
    TileSpec,
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

# Step budget: min(_STEPS_PER_BUFFER * n, _MAX_STEPS). The ceiling sits above
# the layout-only annealer's clamp (``SelfCalibratingReheatingSchedule
# .max_steps``, 5_000) since this engine searches divisions too, and binds only
# well past the validated corpus (currently ~80 buffers). It bounds *steps*, not
# wall-clock.
_STEPS_PER_BUFFER = 200
_MAX_STEPS = 50_000

# Fixed proposal weights over the three move types. Reorder's weight is
# effectively 0 while every eligible buffer is resident (see
# :meth:`_applicable_moves`).
_MOVE_WEIGHTS = {"reorder": 0.5, "flip": 0.3, "recolor": 0.2}

# Layout-burst length as a fraction of the buffer count. The burst warms ``pi`` to
# the new footprints before a compound structural move is judged.
_BURST_FRACTION = 0.1

# How often a recolor anchor is drawn untiled outright (see
# ``_GeneratedDivisions._draw_tiling``). Flat, and deliberately not a per-dim
# opt-out: the probability of proposing an untiled region must not decay with the
# number of tileable dims. ``_companion_bytes`` prices whether to tile, but
# nothing prices how *deep*, so the draw is what sets depth.
_UNTILED_ANCHOR_PROB = 0.5

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
    pair table, whose entries are all untiled, answer for a tiled config at
    all. (What a tiling does change about an edge -- a consumer in another
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
    or memoized by it. Where a menu carries two entries that share a split map
    without being the same choice, :meth:`SaCoOptimizingSolver._build_sources`
    gives the second an identity of its own.

    The key is *derived* -- from ``division``, plus ``menu_index`` at a
    position that ``position_disambiguates`` -- and never supplied, so it cannot
    disagree with the config it identifies; what dedup and memoization need on
    top of that is that a ``CoreDivision`` handed to a config is not mutated
    afterwards, which nothing does.

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
    def output_splits(self) -> dict:
        return self.division.output_splits

    @property
    def reduction_splits(self) -> dict:
        return self.division.reduction_splits

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
    :meth:`neighbours` is *one step* -- one axis's factor, or one coarse tile
    level -- which is what a flip takes, and stage 0 measured those domains at
    ~7 legal factors per axis, a list to pick from rather than an interval to
    propose over with a cooling scale. But an op's legal divisions are not
    connected by one-axis moves (the core budget blocks a factor going up, a
    span floor blocks it coming down), so a search with only local moves does
    measurably worse: +0.9% on the corpus at four seeds. :meth:`anchor` is the
    long-range draw that pays for it, and recolor is where it belongs, since a
    flooded region is a coordinated change anyway -- and, once tilings are in,
    since a tiling group *is* a region that agrees on one ``TileSpec``.
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

    def retiled(
        self, config: DivisionConfig, tiling: "TileSpec"
    ) -> Optional[DivisionConfig]:
        """``config``'s splits under ``tiling``, or ``None`` if this buffer
        cannot take that tiling at those splits.

        What both halves of the contiguity invariant ask: the boundary flip
        spreading one tiling along a run, and the recolor trim putting an op
        outside the anchor's run back to untiled. Splits are held fixed, so
        this is not a move in the division lattice -- a tiling the incoming
        splits cannot live with is refused here rather than silently paired
        with different ones.
        """
        raise NotImplementedError

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
    config per position.

    Every menu is already duplicate-free on its own terms: an op's is
    deduplicated by split map where it is enumerated
    (``_enumerate_core_divisions``), and a clone's by *physical partition* where
    it is synthesized (``_clone_divisions_and_matches``). So every position is
    its own choice, and the move alphabet is the whole menu.
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
        """The first menu entry with ``division``'s split map.

        Exact only where the menu's entries are distinct by split map, which an
        op's are and a clone's need not be (see
        :meth:`SaCoOptimizingSolver._build_sources`). Nothing in the engine asks
        a menu for a config -- a flood propagates through an
        :class:`_EdgeRelation`, which answers with the entry itself.
        """
        return self.by_key[_canonical_key(division)]

    def can_move(self) -> bool:
        return len(self.configs) > 1

    def can_split(self) -> bool:
        return any(config.output_partition > 1 for config in self.configs)

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
        key = _canonical_key(division)
        if key not in self._config_cache:
            self._config_cache[key] = DivisionConfig(division, None)
        return self._config_cache[key]

    def can_move(self) -> bool:
        return any(
            len(factors) > 1 for factors in self.space.factor_domains.values()
        ) or not (self.space.tiling is None or self.space.tiling.is_empty)

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
        (a committed span floor), the tiling is given up instead: the incoming
        splits are legal untiled, since a tiled domain is a subset of the
        untiled one.
        """
        splits = self.space.splits(config.division)
        tiling = self._draw_tiling(rng)
        if not self.space.admits(splits, tiling):
            floor = dict.fromkeys(self.space.axes, 1)
            if self.space.admits(floor, tiling):
                splits = floor
            else:
                tiling = _UNTILED
        for axis in self.space.axes:
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
        is no longer accepted unconditionally. *How deep* to tile is not. The
        objective sees depth only through a monotone per-core footprint, and the
        companion charge is a function of the buffer's size and its outside
        readers rather than of the tile count, so a drawn tiling is as deep as
        the draw made it. Hence the flat untiled draw: it is what keeps
        undividing a region proposable at all, and recolor is the only
        long-range move there is.

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
    is projected onto keys once. That also makes it exact for a *generated*
    config, which is why a graph where only some ops have a split space is not a
    mixture of two answers: a generated division is one the enumeration would
    have carried, so its key is a key the table knows.

    Every entry is untiled, since the enumeration carries no tilings, so the
    projection is onto :func:`_split_key` -- the table answers for a tiled
    config the same way it answers for its untiled twin, which is what the
    geometry says. What it cannot do is *carry* a tiling across the edge: the
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
    parent_menu: tuple[Sequence[tuple], dict],
    child_menu: tuple[Sequence[tuple], dict],
) -> _TableRelation:
    """Project a menu-position pair table onto choices.

    Each side is its menu's ``(keys by position, key -> config)`` pair, which
    every buffer has whether or not its own source generates. The propagation
    direction keeps the pair table's tie-break -- the compatible division at the
    lowest menu position wins -- which is what makes a flood independent of
    ``cd_parent_matches`` list order.
    """
    parent_keys, parent_by_key = parent_menu
    child_keys, child_by_key = child_menu
    # Asserted rather than filtered: a pair naming a position no menu has is a
    # table built against a different candidate list, and silently dropping the
    # row would lose an edge the allocator meant to offer.
    key_pairs = set()
    for ip, ic in pairs:
        assert ip < len(parent_keys) and ic < len(child_keys), (
            f"cd_parent_matches names candidate ({ip}, {ic}), but the menus hold "
            f"{len(parent_keys)} and {len(child_keys)}"
        )
        key_pairs.add((_split_key(parent_keys[ip]), _split_key(child_keys[ic])))

    # First position wins the projection, as it does everywhere else a menu is
    # collapsed. Two full keys share a split key once some candidates carry a
    # committed tiling and others do not -- the hint-tiled and span-overflow
    # ops -- and the later one would otherwise silently take over both the
    # ``menu_index`` the tie-break below sorts on and the config a flood
    # propagates.
    def by_split(configs: dict) -> dict:
        out: dict = {}
        for key, config in configs.items():
            out.setdefault(_split_key(key), config)
        return out

    # Ordered by menu position rather than by iterating the set, whose order is
    # a hash order and so not stable across interpreter runs.
    parent_config = by_split(parent_by_key)
    child_config = by_split(child_by_key)
    position = {
        pair: (parent_config[pair[0]].menu_index, child_config[pair[1]].menu_index)
        for pair in key_pairs
    }
    down: dict = {}
    up: dict = {}
    for pair in sorted(key_pairs, key=lambda kp: position[kp][::-1]):
        down.setdefault(pair[0], child_config[pair[1]])
    for pair in sorted(key_pairs, key=lambda kp: position[kp]):
        up.setdefault(pair[1], parent_config[pair[0]])
    return _TableRelation(frozenset(key_pairs), down, up)


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
    answer even though the verdict does not.
    """

    edge: "ResidencyEdge"
    parent_source: _GeneratedDivisions
    child_source: _GeneratedDivisions
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
        :meth:`_build_sources` froze, which is also what makes that so: a config
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
            # The division's identity, for table terms over candidates (the
            # relayout price is one; see RelayoutCopyBuffer.cost_term). The
            # table is indexed by menu position, so the config's provenance is
            # what binds here, not the config itself.
            value_of[buf.sym_division] = lambda chosen, resident, idx=idx: (
                chosen[idx].menu_index
            )
            for key, sym in self._sym_core_divs[idx].items():
                value_of[sym] = lambda chosen, resident, idx=idx, key=key: (
                    chosen[idx].splits.get(key, 1)
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
        self._build_operation_positions()
        # Whether any buffer can carry a tiling at all, so the companion term
        # costs one bool per score where it cannot -- which is every graph with
        # ``auto_coarse_tiling`` off, and the parity gate's whole corpus.
        self._tilings_are_possible = any(
            isinstance(source, _GeneratedDivisions)
            and source.space.tiling is not None
            and not source.space.tiling.is_empty
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
        self._precompute_spill_costs()

    def _build_operation_positions(self) -> None:
        """Where each buffer's producing operation sits in ``graph.operations``,
        and the inverse map -- what a coarse-tiling *run* is measured over.

        A group has to occupy one contiguous stretch of the operation list, and
        buffer indices are not operation positions: the allocator prepends input
        clones, and an operation that produces no solver buffer has no index at
        all. So the run structure is read off ``op_position``, and a position
        with no buffer is untiled by definition -- which is exactly right, since
        nothing can carry a tiling to an operation the search does not own.

        A buffer without one (an input clone, or any buffer a caller built
        without supplying operation order) is in no run, so it can hold no
        tiling: :meth:`_retile_boundary` declines and
        :meth:`_trim_tilings_to_anchor_run` strips. Declining is the safe
        direction -- a tiling is only ever worth having if something can apply
        it, and nothing can apply one whose group is not known to be contiguous.
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

    def _run_bounds(
        self,
        position: int,
        override: Optional[dict[int, DivisionConfig]] = None,
    ) -> tuple[int, int]:
        """The maximal contiguous stretch of operation positions agreeing with
        ``position`` on the tiling -- inclusive on both ends.

        Untiled is a spec value like any other, so these runs partition the whole
        operation list. That is what lets a boundary move *create* a tiled region
        rather than only shrink one.
        """
        spec = self._tiling_at(position, override)
        lo = position
        while lo > 0 and self._tiling_at(lo - 1, override) == spec:
            lo -= 1
        hi = position
        while hi + 1 < self._n_positions and self._tiling_at(hi + 1, override) == spec:
            hi += 1
        return lo, hi

    def _edge_relation(self, p_idx: int, c_idx: int, p_name: str) -> _EdgeRelation:
        """The relation on the edge ``p_idx -> c_idx``.

        Computed per candidate off the buffer's geometry where both ends
        generate their divisions and the allocator handed over the edge;
        otherwise the pair table, which is what a clone parent, a
        non-``ComputedBuffer`` op and a division-pinned op still have.

        The two agree on the *verdict* -- a generated division is one the
        enumeration would have carried, so the table knows its key -- so a graph
        that mixes them is not a mixture of two answers about compatibility.
        They can differ on which compatible division they hand back; see
        :class:`_ViewRelation`.
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

        A choice is normally its split map, which identifies an entry only
        within one symbol namespace -- the scope :func:`_canonical_key`
        documents. An op's menu is one namespace and is deduplicated in it, so
        there the split map *is* an identity, and it has to be: a generated
        config is keyed by one, and the projection is what makes the two meet.
        A clone's menu is not one namespace. ``_clone_divisions_and_matches``
        synthesizes one entry per consumer out of that consumer's own iteration
        symbols and deduplicates them by physical partition, and Inductor's
        iteration symbols are positional and repeat across ops -- so two entries
        can share a split map while slicing the buffer differently. Merging them
        would leave the pair table with no positional information at all, and
        report a division compatible with a clone entry it was never checked
        against; a position whose split map is already taken therefore keeps an
        identity of its own.
        """
        self._sym_core_divs = [b.sym_core_divs for b in self._bufs]
        # Per buffer: the menu's keys by position, and the config per key.
        self._menu: list[tuple[list[tuple], dict]] = []
        self._sources: list[_DivisionSource] = []
        for buf, declared in zip(self._bufs, self._sym_core_divs):
            keys: list[tuple] = []
            by_key: dict = {}
            configs: list[DivisionConfig] = []
            split_maps: set[tuple] = set()
            for index, cd in enumerate(buf.core_divisions):
                undeclared = undeclared_splits(cd, declared)
                assert not undeclared, (
                    f"buffer {buf.name}: candidate {index} splits "
                    f"{sorted(str(key) for key in undeclared)}, which the cost "
                    "expression declares no symbol for, so those splits would be "
                    "priced as unsplit"
                )
                canonical = _canonical_key(cd)
                config = DivisionConfig(cd, index, canonical in split_maps)
                split_maps.add(canonical)
                keys.append(config.key)
                by_key[config.key] = config
                configs.append(config)
            self._menu.append((keys, by_key))
            space = buf.division_space
            assert space is None or len(split_maps) == len(keys), (
                f"buffer {buf.name}: its menu repeats a split map, but its "
                "generated configs are keyed by one, so a generated division "
                "would not meet the menu entry making the same choice"
            )
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
        ``ceil_div(total_size, output_partition * output_tile_count)``, using
        the substrate's integer helper so this rounds identically to every
        other footprint-division site -- ``CoreDivisionBuffer.min_footprint``
        divides by the same product.

        The tile count is the whole payoff channel for tiling. Every
        tiling-sensitive term in the cost model is a derate bounded by 1.0, and
        an untiled op has a working set of 0 by definition, so the objective
        can rank tilings against each other but never above not tiling. What a
        tiling can do is bring this footprint under :attr:`limit` in
        :meth:`_eligible`, which is an engine threshold rather than a cost
        term, and be paid for afterwards in the traffic residency frees.

        Only the *scratch* is sized here, and rightly so: the full-extent
        companion an escaping op also needs never enters LX (the apply mints it
        after the addresses are final), so it competes for no space. What it
        costs is traffic, which :meth:`_companion_bytes` charges.

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
                    lifetime_end_override=b.lifetime_end_override,
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

    def _tiled_runs(self) -> dict[int, tuple[int, int]]:
        """Inclusive bounds of the tiled run each tiled operation position sits
        in; untiled positions are absent, since they mint no companion.

        One left-to-right pass rather than :meth:`_run_bounds` per buffer, which
        re-walks the whole run each time and would be quadratic on a heavily
        tiled graph -- this runs once per score.
        """
        runs: dict[int, tuple[int, int]] = {}
        start = 0
        while start < self._n_positions:
            spec = self._tiling_at(start)
            end = start
            while end + 1 < self._n_positions and self._tiling_at(end + 1) == spec:
                end += 1
            if not spec.is_untiled:
                for position in range(start, end + 1):
                    runs[position] = (start, end)
            start = end + 1
        return runs

    def _companion_bytes(self, addresses: Sequence[Optional[int]]) -> int:
        """HBM bytes the apply's companion buffers move that the rest of the
        objective does not count, under the current state.

        A tiling shrinks what the buffer holds at once; it does not shrink what
        the buffer moves. For an op whose output escapes its tiling group,
        ``CoarseTilingPass`` allocates a full-extent ``full_buf``, inserts a copy
        op that drains one tile into it per iteration, and repoints every outside
        consumer and any graph output at it (``wsr/coarse_tile.py``). None of
        that is in the features: they were extracted from the untiled graph,
        before ``_post_solve`` applies anything, so the search sees the per-tile
        shrink and neither the copy nor the outside consumers' HBM reads. It
        therefore over-values tiling by a quantity that does not fall with the
        tile count -- measured as +24% of HBM traffic on
        ``test_mlp__simulated_annealing_sc32_coopt``.

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

        Two known under-corrections, neither expressible from what the solver
        holds: a consumer is counted once however many times it reads, and a
        matmul consumer's ``replication`` (#4454) would make its read of
        ``full_buf`` cost more than one pass.
        """
        if not self._tilings_are_possible:
            return 0
        total = 0
        for position, (lo, hi) in self._tiled_runs().items():
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
            copy_read = 0 if addresses[idx] is not None else size
            copy_write = copy_read if is_graph_output else size
            outside_reads = size * outside if addresses[idx] is not None else 0
            total += copy_read + copy_write + outside_reads
        return total

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

        :meth:`_companion_bytes` is added to both, at the HBM rate, because
        neither can express it: the cost expression is built once from the
        untiled graph over splits and residency, with no symbol for a tiling, and
        the fallback's spill costs are loop-invariant by construction. It is zero
        unless a buffer is tiled, so a run that chooses no tiling scores exactly
        as it did before this term existed.
        """
        addresses = self.packer.addresses
        companions = utils.to_fixed_us(
            self._companion_bytes(addresses) / self._hbm_bytes_per_us
        )
        if self._score_fn is not None:
            resident = frozenset(
                b.name
                for b, address in zip(self._bufs, addresses)
                if address is not None
            )
            return self._score_fn(self.chosen, resident) + companions

        traffic = sum(
            cost
            for cost, address in zip(self._spill_costs, addresses)
            if address is None
        )
        return utils.to_fixed_us(traffic / self._hbm_bytes_per_us) + companions

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

    def _retile_boundary(self, idx: int, config: DivisionConfig) -> None:
        """Spread ``config``'s tiling from ``idx`` to one end of its run.

        The tiling arm of flip. ``idx`` sits in a uniform run ``A..Z``; this
        re-specs ``A..H`` or ``H..Z`` (inclusive), so the run splits in two, or
        -- where the new spec matches the neighbouring run's -- the boundary
        between them slides. Contiguity therefore holds by construction rather
        than being repaired afterwards, and because untiled is a spec value like
        any other, the move creates tiled regions as readily as it shrinks them.

        **Truncating, not rejecting.** Whether an op can take a tiling is a
        per-op question (``OpSplitSpace.neighbours`` only offers a level the op's
        current splits survive), so over a run of any length the odds that every
        member agrees fall off fast. Stopping at the first op that refuses reads
        as sliding the boundary as far as it will go, and keeps the sub-run
        contiguous; rejecting the whole move would make long runs nearly immovable.

        An operation that produces no solver buffer stops the walk for the same
        reason it breaks a run: nothing can carry a tiling to it.

        The single-op flip survives as the degenerate case -- ``H`` at a run end
        -- and a mid-run split is two steps rather than one.
        """
        position = self._position_of[idx]
        if position is None:
            return  # no operation position, so no run to move a boundary in
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
            retiled = self._sources[target].retiled(self.chosen[target], config.tiling)
            if retiled is None:
                break
            assignment[target] = retiled
            if at == end:
                break
            at += step
        if assignment:
            self._apply_assignment(assignment)

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

    def _trim_tilings_to_anchor_run(
        self, anchor: int, assignment: dict[int, DivisionConfig]
    ) -> dict[int, DivisionConfig]:
        """Strip the ``TileSpec`` from every assigned op outside the anchor's
        contiguous run, leaving its splits alone.

        The flood's reach is the residency relation's, which is producer /
        consumer reachability; a coarse-tiling group has to be a contiguous run
        of the operation list. Left alone, a region straddling an op the relation
        could not carry the tiling to would be priced as one group where the
        apply round forms two, the second reading the first's *full* extent. The
        divisions are deliberately untouched -- they are what the flood is for,
        and narrowing them to the run would cost the long-range division move
        stage 2b measured at -0.71%.

        Always legal: the untiled factor domain contains the tiled one (tiling
        only removes large factors), so splits admitted under a tiling are
        admitted without it.

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
            assert untiled is not None, (
                f"buffer {self._bufs[idx].name}: splits admitted under "
                f"{config.tiling.label} are not admitted untiled, but the untiled "
                "domain contains the tiled one"
            )
            trimmed[idx] = untiled
        return trimmed

    def _apply_assignment(self, assignment: dict[int, DivisionConfig]) -> None:
        """Commit a multi-op division assignment: set every op's division, resize
        its footprint, and refresh eligibility for the assigned set plus their
        parents (the same ripple as a flip, unioned over the set).

        Both structural moves that touch more than one buffer land here -- a
        flooded region coloring, and a boundary flip's sub-run.
        """
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
        self._apply_assignment(
            self._trim_tilings_to_anchor_run(anchor, self._flood_region(anchor, config))
        )
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
        Configs are never mutated in place, so the shallow list copy is a full
        copy of ``W``."""
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
            # One axis's factor, or one coarse tile level, drawn uniformly
            # from the divisions a step away. Stage 0 measured those domains at
            # ~7 factors per axis, which is why there is a list to pick from
            # here rather than a proposal scale to cool.
            options = self._sources[idx].neighbours(self.chosen[idx])
            if not options:
                return
            config = self._rng.choice(options)
            # Two arms, two scopes. A step in the division lattice is this op's
            # alone; a step in the tiling lattice moves a *boundary*, because a
            # tiling group is a contiguous run and a single op re-specced in the
            # middle of one would split it into a shape the apply round prices
            # differently than the search did.
            if config.tiling == self.chosen[idx].tiling:
                self._atomic_flip(idx, config)
            else:
                self._retile_boundary(idx, config)
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
        steps = min(_MAX_STEPS, _STEPS_PER_BUFFER * n)
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
