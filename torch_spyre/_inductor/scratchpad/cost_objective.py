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

"""Cost-model objective for the SA co-optimizer, cached per bundle.

The engine's own objective is memory-only: a division only matters if it changes
what fits in LX. That makes several captures sit at the objective's floor, unable
to distinguish any move set or schedule, and it prices at zero the compute cost
of a division -- which is the whole reason to choose one. Scoring with the cost
model instead separates division choices by 4-6x on matmul-bearing buffers, where
the memory-only objective sees exact ties.

The cost is *per bundle* (one fused SuperDSC kernel), and non-separable within
one: external graph inputs are deduplicated across the bundle, the pointwise
arity derate counts its ops, the underfill derate takes its worst tile, and the
turnaround term uses bundle totals. So a bundle is the smallest unit that can be
scored, and the graph total is the sum over bundles.

Cost control. ``predict_ops`` runs 3-10us per bundle against the memory-only
objective's ~3us for a whole graph, and an anneal evaluates 10^5-10^6 times, so
scoring naively is not affordable. Two mechanisms:

* **Memoization.** A bundle's cost depends only on the divisions of its own ops
  and the residency of its own arguments, so it is keyed on exactly that and
  cached. Repeat states -- common once the schedule cools and most moves are
  rejected -- become dict hits.
* **Dirty tracking.** Between calls, only the buffers whose division or residency
  actually changed can dirty a bundle, so the rest keep their previous cost.

Determinism. Each bundle's microsecond prediction is converted to the shared
fixed-point integer scale *once*, then summed as integers. Float accumulation
would make an incrementally-updated total drift from a recomputed one, which
would break the engine's bit-for-bit reproducibility guarantee; integers make the
two agree exactly. :meth:`score_from_scratch` exists so tests can assert that.
"""

from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from typing import Optional

from torch_spyre._inductor.cost_model import OpFeatures, predict_ops
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.scratchpad import cooptimization_scorer as scorer
from torch_spyre._inductor.scratchpad.op_features import with_residency

logger = get_inductor_logger("scratchpad.cost_objective")


class BundleCostObjective:
    """Sum of per-bundle cost-model predictions, in fixed-point time units.

    Args:
        buffer_names: the solver's buffers, in solver index order. Index ``i``
            here is the same ``i`` the packer and ``chosen`` use.
        features: ``{buffer_name: [OpFeatures | None per division index]}``,
            index-aligned with that buffer's ``core_divisions`` menu. A ``None``
            entry means the op could not be featurized for that division.
        bundles: groups of buffer names, one per fused kernel (see
            ``fusion.estimate_bundles``). Names not in ``buffer_names`` are
            ignored, so a bundle may legitimately end up empty and is dropped.
    """

    def __init__(
        self,
        buffer_names: Sequence[str],
        features: dict[str, list[Optional[OpFeatures]]],
        bundles: Sequence[Sequence[str]],
    ) -> None:
        self._names = list(buffer_names)
        self._index = {name: i for i, name in enumerate(self._names)}
        self._features = features
        # Bundles as solver indices, dropping names this solver does not own
        # (graph inputs, constants) and any bundle left empty by that filter.
        self._bundles: list[tuple[int, ...]] = []
        for group in bundles:
            members = tuple(self._index[n] for n in group if n in self._index)
            if members:
                self._bundles.append(members)
        # buffer index -> bundles whose cost it can change. A buffer affects the
        # bundle it belongs to (via its division) and every bundle that reads or
        # writes it (via its residency), which is not the same set.
        # Argument names per bundle, computed once: they are read on every key
        # construction, and deriving them per call dominated the scoring cost.
        self._bundle_args: list[frozenset[str]] = [
            frozenset(self._arg_names_of(b)) for b in range(len(self._bundles))
        ]
        self._division_dirties: dict[int, set[int]] = {}
        self._residency_dirties: dict[int, set[int]] = {}
        for b, members in enumerate(self._bundles):
            for i in members:
                self._division_dirties.setdefault(i, set()).add(b)
            for name in self._bundle_args[b]:
                owner = self._index.get(name)
                if owner is not None:
                    self._residency_dirties.setdefault(owner, set()).add(b)
        self._cache: dict[tuple, int] = {}
        # Previous state, for dirty tracking. ``None`` forces a full evaluation
        # on the first call and after :meth:`invalidate`.
        self._prev_chosen: Optional[tuple[int, ...]] = None
        self._prev_resident: Optional[AbstractSet[str]] = None
        self._bundle_cost: list[int] = [0] * len(self._bundles)
        self._total = 0
        self.evaluations = 0  # bundles actually sent to predict_ops
        self.lookups = 0  # bundles whose cost was needed

    # -- structure -----------------------------------------------------------

    def _arg_names_of(self, b: int) -> set[str]:
        """Every argument name any op in bundle ``b`` touches, over all of its
        candidate divisions. Taken across the whole menu deliberately: the set
        must not change as the search moves, or the dirty map would go stale."""
        names: set[str] = set()
        for i in self._bundles[b]:
            for feat in self._features.get(self._name(i), ()) or ():
                if feat is not None:
                    names.update(a.name for a in feat.args)
        return names

    def _name(self, i: int) -> str:
        return self._names[i]

    # -- scoring -------------------------------------------------------------

    def _bundle_features(
        self, b: int, chosen: Sequence[int], resident: AbstractSet[str]
    ) -> list[OpFeatures]:
        out = []
        for i in self._bundles[b]:
            menu = self._features.get(self._name(i))
            if not menu:
                continue
            feat = menu[chosen[i]] if chosen[i] < len(menu) else None
            if feat is not None:
                out.append(with_residency(feat, resident))
        return out

    def _bundle_key(
        self, b: int, chosen: Sequence[int], resident: AbstractSet[str]
    ) -> tuple:
        members = self._bundles[b]
        # Residency restricted to this bundle's own arguments: a residency change
        # elsewhere must not evict this bundle's cached cost.
        local = tuple(sorted(self._bundle_args[b] & resident))
        return (b, tuple(chosen[i] for i in members), local)

    def _bundle_value(
        self, b: int, chosen: Sequence[int], resident: AbstractSet[str]
    ) -> int:
        key = self._bundle_key(b, chosen, resident)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        feats = self._bundle_features(b, chosen, resident)
        if not feats:
            value = 0
        else:
            # One rounding step per bundle, then integer accumulation -- see the
            # determinism note in the module docstring.
            value = scorer.to_fixed_us(max(0.0, predict_ops(feats)) / 1000.0)
        self._cache[key] = value
        self.evaluations += 1
        return value

    def score(self, chosen: Sequence[int], resident: AbstractSet[str]) -> int:
        """Total cost for division vector ``chosen`` and LX-resident set
        ``resident``, recomputing only the bundles those could have changed."""
        chosen_t = tuple(chosen)
        if self._prev_chosen is None or self._prev_resident is None:
            dirty: set[int] = set(range(len(self._bundles)))
        else:
            dirty = set()
            for idx, (new, old) in enumerate(zip(chosen_t, self._prev_chosen)):
                if new != old:
                    dirty |= self._division_dirties.get(idx, set())
            for name in resident ^ self._prev_resident:
                moved = self._index.get(name)
                if moved is not None:
                    dirty |= self._residency_dirties.get(moved, set())
        for b in dirty:
            value = self._bundle_value(b, chosen_t, resident)
            self._total += value - self._bundle_cost[b]
            self._bundle_cost[b] = value
        self.lookups += len(dirty)
        self._prev_chosen = chosen_t
        self._prev_resident = resident
        return self._total

    def score_from_scratch(
        self, chosen: Sequence[int], resident: AbstractSet[str]
    ) -> int:
        """``score`` without dirty tracking. Same value by construction -- the
        integer accumulation makes that exact rather than approximate -- so a
        test can assert the incremental path never drifts."""
        chosen_t = tuple(chosen)
        return sum(
            self._bundle_value(b, chosen_t, resident) for b in range(len(self._bundles))
        )

    def invalidate(self) -> None:
        """Forget the previous state so the next :meth:`score` recomputes every
        bundle.

        The engine restores snapshots on a rejected move, which rolls the state
        back without telling this object. The cached *values* stay valid (they
        are keyed on state, not on history); only the diff baseline is wrong, so
        this resets that alone.
        """
        self._prev_chosen = None
        self._prev_resident = None
        self._total = 0
        self._bundle_cost = [0] * len(self._bundles)
