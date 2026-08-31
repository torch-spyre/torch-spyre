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

"""Validation for the end-to-end SA co-optimization engine.

The two gates the engine must pass:

* **determinism** -- two runs on identical input give bit-for-bit identical
  ``chosen_division`` + ``address``; and
* **>= baseline on the shared scorer** -- the returned state never scores worse
  than the seed (index-0 divisions + FirstFit ``pi``), the seed-from-baseline +
  keep-best guarantee.

Plus the engine's output contract: every buffer gets a ``chosen_division`` and an
``address`` (``None`` == spilled), with ``spill_reasons`` populated for the
misses. Runs over the real-shaped captured graphs at several capacities, so
residency pressure (spills / eligibility toggles) is actually exercised.
"""

import copy
import json
import math
import os
import random as rnd
import subprocess
import sys
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad import utils
from torch_spyre._inductor.scratchpad.sa_cooptimizer import (
    _MAX_STEPS,
    _MIN_STEPS,
    _STEPS_PER_BUFFER,
    SaCoOptimizingSolver,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    make_permutation_packer,
)

from tests.inductor.cooptimization_capture_loader import load_captures
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
)
from tests.inductor.synthetic_cooptimization_graphs import synthetic_graphs


def _seed_footprint(buffers):
    """Total per-core footprint of the seed (index-0) divisions -- the scale used
    to pick exercise capacities."""
    return sum(
        math.ceil(b.size / b.core_divisions[0].output_partition) for b in buffers
    )


def _capacities(buffers):
    """A spread of scratchpad capacities: unbounded, roomy, and two tight ones
    that force spills / eligibility pressure."""
    tot = _seed_footprint(buffers)
    return [1 << 30, tot, max(1, tot // 2), max(1, tot // 4)]


def _all_cases():
    """The captured real corpus (softmax/mlp/swiglu/sdpa)."""
    for case, graphs in load_captures().items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


def _synthetic_cases():
    """Hand-built structural fixtures (long/short chains, wide join, multi-region,
    K-split, pins, big-n). They carry no ground-truth ``solved`` reference, so they
    exercise only the *shape-invariant* guarantees below. See
    ``synthetic_cooptimization_graphs``."""
    for case, graphs in synthetic_graphs().items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


# Large (25-100 buffer) real captures, kept OUT of CI: they are slow (~2s/solve at
# n~79). Opt in with ``SA_COOPT_LARGE_CAPTURES=1``.
_LARGE_CAPTURES_ENV = "SA_COOPT_LARGE_CAPTURES"
_LARGE_CAPTURES_PATH = os.path.join(
    os.path.dirname(__file__), "cooptimization_captures_large.json"
)


def _large_captures_enabled() -> bool:
    return os.environ.get(_LARGE_CAPTURES_ENV) == "1"


def _large_cases():
    """The env-gated large graphs (empty unless opted in)."""
    if not _large_captures_enabled():
        return
    for case, graphs in load_captures(_LARGE_CAPTURES_PATH).items():
        for gi, g in enumerate(graphs):
            yield case, gi, g.buffers


def _all_cases_incl_synthetic():
    """Real captures + synthetic fixtures (+ the large graphs when opted in): the
    fan-out for shape-invariant tests (output contract, >= baseline, determinism,
    region flood) that must hold for *any* valid graph."""
    yield from _all_cases()
    yield from _synthetic_cases()
    yield from _large_cases()


def _primed(buffers, capacity):
    """A solver primed to the seed state (index-0 divisions, FirstFit ``pi``): the
    prefix of ``plan_layout_and_core_divisions`` up to the anneal, so a unit test
    can drive the move / snapshot machinery -- or read the seed score -- directly.
    """
    solver = SaCoOptimizingSolver(buffers, capacity, 128)
    solver.spill_reasons = {}
    solver._rng = rnd.Random(0)
    solver._precompute_topology()
    solver.chosen = [0] * len(buffers)
    solver.packer = solver._build_seed_packer()
    solver._flippable_ops = solver._flippable()
    solver._best_score = solver._score()
    solver._best_snap = solver._snapshot()
    return solver


def _baseline_score(buffers, capacity):
    """The seed state's score -- the value ``best_score`` must never exceed."""
    return _primed(copy.deepcopy(buffers), capacity)._score()


def _geometry_violations(buffers, capacity, alignment):
    """Every way a solved layout can be geometrically wrong, as a list of
    human-readable strings (empty == the layout is realizable).

    Derived from the returned buffers alone -- lifetimes off ``uses``, per-core
    footprints off ``size`` and the chosen division's ``output_partition`` -- so
    it shares no code with the packer whose output it judges. That is the point:
    ``test_probe_walk_leaves_the_packer_consistent`` compares the incremental
    packer against a from-scratch rebuild, which catches bookkeeping drift but
    puts the same geometry rules on both sides, so a systematic placement bug
    would sit in both and go unseen.

    The properties, on the resident (addressed) buffers:

    * each address is a multiple of ``alignment`` (one Spyre stick);
    * each buffer fits entirely below ``capacity``;
    * two buffers alive at a common tick never share a byte -- with the one
      legitimate exception of an in-place pair, where the child takes over the
      parent's storage at the handoff tick and so must sit at *exactly* the
      parent's address.
    """
    resident = [b for b in buffers if b.address is not None]
    bad = []
    footprint = {}
    for b in resident:
        part = b.core_divisions[b.chosen_division].output_partition
        footprint[b.name] = max(0, -(-b.size // part))
        if b.address % alignment:
            bad.append(f"{b.name}: address {b.address} is not {alignment}-aligned")
        if b.address + footprint[b.name] > capacity:
            bad.append(
                f"{b.name}: [{b.address}, {b.address + footprint[b.name]}) crosses "
                f"the capacity {capacity}"
            )
    # A buffer with no uses is alive at no tick, so it can overlap nothing; the
    # alignment and capacity checks above still covered it.
    live = [b for b in resident if b.uses]
    for i, bi in enumerate(live):
        for bj in live[i + 1 :]:
            # Lifetimes are the half-open [uses[0], uses[-1] + 1), re-derived
            # here rather than taken from the buffer's own properties.
            if not (bi.uses[0] < bj.uses[-1] + 1 and bj.uses[0] < bi.uses[-1] + 1):
                continue
            lo_i, hi_i = bi.address, bi.address + footprint[bi.name]
            lo_j, hi_j = bj.address, bj.address + footprint[bj.name]
            if hi_i <= lo_j or hi_j <= lo_i:
                continue
            in_place = bj.name in bi.in_place_parents or bi.name in bj.in_place_parents
            if in_place and lo_i == lo_j:
                continue
            bad.append(
                f"{bi.name} [{lo_i}, {hi_i}) and {bj.name} [{lo_j}, {hi_j}) are "
                f"alive together and share bytes"
            )
    return bad


class OutputContractTest(TestCase):
    def test_every_buffer_gets_division_and_address(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(bufs, cap, 128)
                out = solver.plan_layout_and_core_divisions()
                tag = f"{case}[{gi}] cap={cap}"
                self.assertEqual(len(out), len(bufs), tag)
                for b in out:
                    self.assertIsNotNone(b.chosen_division, f"{tag} {b.name}")
                    self.assertTrue(0 <= b.chosen_division < len(b.core_divisions), tag)
                    # A spilled buffer (no address) must carry a spill reason;
                    # a resident one must not.
                    if b.address is None:
                        self.assertIn(b.name, solver.spill_reasons, f"{tag} {b.name}")
                    else:
                        self.assertNotIn(b.name, solver.spill_reasons, tag)

    def test_empty_graph(self):
        solver = SaCoOptimizingSolver([], 1024, 128)
        self.assertEqual(solver.plan_layout_and_core_divisions(), [])


class GeometricValidityTest(TestCase):
    """The returned layout is physically realizable: stick-aligned, inside the
    capacity, and free of overlap between buffers that are alive together.

    The output contract above says every buffer got *an* address; this says the
    addresses describe a placement the hardware could actually take. See
    :func:`_geometry_violations` for why this cannot be delegated to a rebuild-
    and-compare check.
    """

    @staticmethod
    def _placed(name, size, uses, address, in_place_parents=()):
        """A buffer already carrying a solved division and address, for the
        checks that hand the validator a layout instead of solving one."""
        buf = CoreDivisionBuffer(
            name=name,
            size=size,
            uses=list(uses),
            first_use_is_read=False,
            in_place_parents=list(in_place_parents),
            # The trivial division, so the per-core footprint is ``size``.
            core_divisions=[CoreDivision(output_splits={}, reduction_splits={})],
            boundary=BufferType.Intermediate,
        )
        buf.chosen_division = 0
        buf.address = address
        return buf

    def test_returned_layout_is_geometrically_valid(self):
        co_live = 0
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                bufs = copy.deepcopy(buffers)
                solver = SaCoOptimizingSolver(bufs, cap, 128)
                out = solver.plan_layout_and_core_divisions()
                self.assertEqual(
                    _geometry_violations(out, cap, 128), [], f"{case}[{gi}] cap={cap}"
                )
                resident = [b for b in out if b.address is not None]
                co_live += sum(
                    1
                    for i, a in enumerate(resident)
                    for b in resident[i + 1 :]
                    if a.overlaps_in_time(b)
                )
        # Non-overlap is vacuous on a corpus that never holds two buffers at
        # once, so pin that the fan-out really did exercise it.
        self.assertGreater(co_live, 0, "no two resident buffers were ever co-live")

    def test_validator_names_each_way_a_layout_can_be_wrong(self):
        """A validator nothing can fail proves nothing: break one property at a
        time on a hand-placed layout and confirm each is caught on its own."""
        cap = 1024
        a = self._placed("a", 256, (0, 4), 0)
        b = self._placed("b", 256, (1, 5), 256)
        self.assertEqual(_geometry_violations([a, b], cap, 128), [])

        b.address = 128  # aligned and in capacity, but overlaps a's [0, 256)
        self.assertEqual(len(_geometry_violations([a, b], cap, 128)), 1)

        b.address = 300  # clear of a, but not a multiple of 128
        self.assertEqual(len(_geometry_violations([a, b], cap, 128)), 1)

        b.address = 896  # aligned and clear of a, but [896, 1152) exceeds 1024
        self.assertEqual(len(_geometry_violations([a, b], cap, 128)), 1)

    def test_in_place_child_may_share_the_parent_address(self):
        """The one legitimate way two co-live buffers share bytes: the child
        takes the parent's storage at the handoff tick. It has to land on
        *exactly* the parent's address -- anywhere else is a real overlap."""
        parent = self._placed("p", 256, (0, 2), 0)
        child = self._placed("c", 128, (2, 3), 0, in_place_parents=["p"])
        self.assertEqual(_geometry_violations([parent, child], 1024, 128), [])

        child.address = 128  # inside the parent, but not its slot
        self.assertEqual(len(_geometry_violations([parent, child], 1024, 128)), 1)


class BaselineGuaranteeTest(TestCase):
    """The returned state never scores worse than the seed (lower is better)."""

    def test_never_worse_than_baseline(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                solver = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
                solver.plan_layout_and_core_divisions()
                self.assertLessEqual(
                    solver.best_score,
                    _baseline_score(buffers, cap),
                    f"{case}[{gi}] cap={cap}",
                )

    def test_best_score_describes_the_state_written_back(self):
        """``best_score`` is published beside the state ``_write_back`` walks, so
        re-scoring the solver's own live state must reproduce it exactly. This is
        what an aliased best-seen snapshot would break: the engine would report a
        score for a layout it had since overwritten."""
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                solver = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
                solver.plan_layout_and_core_divisions()
                self.assertEqual(
                    solver._score(), solver.best_score, f"{case}[{gi}] cap={cap}"
                )


class SeedPermutationTest(TestCase):
    """``pi`` is *ordered* over the buffers that can ever be resident.

    A fixed pin can never be resident for any ``(pi, W)``, so it must not occupy a
    prefix slot and displace an eligible buffer. It keeps its index -- ``pi`` stays
    a permutation of all ``n`` -- but sorts after everything the seed placed.
    """

    def test_pins_sort_after_every_placed_buffer(self):
        checked = 0
        for case, gi, buffers in _all_cases_incl_synthetic():
            bufs = copy.deepcopy(buffers)
            if not any(b.residency_reason is not None for b in bufs):
                continue
            for cap in _capacities(bufs):
                solver = _primed(copy.deepcopy(bufs), cap)
                pi = list(solver.packer.permutation)
                addrs = solver.packer.addresses
                pos = {idx: p for p, idx in enumerate(pi)}
                placed = [i for i in range(len(bufs)) if addrs[i] is not None]
                pinned = [
                    i
                    for i, b in enumerate(solver._bufs)
                    if b.residency_reason is not None
                ]
                if not placed or not pinned:
                    continue
                checked += 1
                tag = f"{case}[{gi}] cap={cap}"
                self.assertLess(
                    max(pos[i] for i in placed),
                    min(pos[i] for i in pinned),
                    f"{tag}: a pinned buffer sits before a placed one in pi",
                )
        self.assertGreater(checked, 0, "no pinned graph exercised")

    def test_pi_remains_a_permutation_of_every_buffer(self):
        """Pins are re-ordered, never dropped: the packer's ``eligible`` mask is
        index-aligned with the buffer list, so ``pi`` must keep all ``n`` slots."""
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                solver = _primed(copy.deepcopy(buffers), cap)
                pi = list(solver.packer.permutation)
                self.assertEqual(
                    sorted(pi), list(range(len(buffers))), f"{case}[{gi}] cap={cap}"
                )

    def test_pins_are_never_placed_by_the_seed(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                solver = _primed(copy.deepcopy(buffers), cap)
                for i, b in enumerate(solver._bufs):
                    if b.residency_reason is not None:
                        self.assertIsNone(
                            solver.packer.addresses[i],
                            f"{case}[{gi}] cap={cap} {b.name}: pinned but placed",
                        )


class UnsizedBufferTest(TestCase):
    """An unsized buffer (the ``mem_usage`` ``-1`` sentinel) is always a fixed pin.

    ``_per_core_size`` clamps ``-1`` to a ``0`` footprint, which passes the
    capacity gate -- so an unsized buffer that ever reached the search would be
    placed occupying no space and the buffer above it would land on the same
    address. What prevents that is a coupling across ``utils.mem_usage_by_buf``,
    ``allocator._op_output_good_for_lx_reuse`` and ``_eligible``'s pin gate,
    which no single file states. ``_assert_unsized_buffers_are_pinned`` states
    it; these pin that it holds on the corpus and that it actually bites.
    """

    @staticmethod
    def _graph(residency_reason):
        """A sized buffer alongside an unsized one, co-live, with the pin state
        under test carried by the unsized buffer."""

        def buf(name, size, reason):
            return CoreDivisionBuffer(
                name=name,
                size=size,
                uses=[0, 1],
                first_use_is_read=False,
                residency_reason=reason,
                core_divisions=[CoreDivision(output_splits={}, reduction_splits={})],
                boundary=BufferType.Intermediate,
            )

        return [buf("sized", 256, None), buf("unsized", -1, residency_reason)]

    def test_corpus_holds_the_invariant(self):
        unsized = 0
        for case, gi, buffers in _all_cases_incl_synthetic():
            for b in buffers:
                if b.size < 0:
                    unsized += 1
                    self.assertIsNotNone(
                        b.residency_reason, f"{case}[{gi}] {b.name}: unsized, unpinned"
                    )
        # The corpus has to actually carry the sentinel, or this proves nothing.
        self.assertGreater(unsized, 0, "no unsized buffer in the corpus")

    def test_unsized_and_unpinned_is_rejected(self):
        solver = SaCoOptimizingSolver(self._graph(None), 1024, 128)
        with self.assertRaisesRegex(AssertionError, "unsized"):
            solver.plan_layout_and_core_divisions()

    def test_unsized_but_pinned_solves_and_spills(self):
        solver = SaCoOptimizingSolver(self._graph("op not allowed"), 1024, 128)
        out = {b.name: b for b in solver.plan_layout_and_core_divisions()}
        # The pin is spilled under its own reason, and never occupies a slot the
        # sized buffer would then be stacked on top of.
        self.assertIsNone(out["unsized"].address)
        self.assertEqual(solver.spill_reasons["unsized"], "op not allowed")
        self.assertEqual(out["sized"].address, 0)


class DeterminismTest(TestCase):
    """Two runs on identical input are bit-for-bit identical."""

    def _run(self, buffers, cap):
        solver = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
        out = solver.plan_layout_and_core_divisions()
        return (
            [b.chosen_division for b in out],
            [b.address for b in out],
            solver.best_score,
            dict(solver.spill_reasons),
        )

    def test_repeated_solves_are_bit_identical(self):
        for case, gi, buffers in _all_cases_incl_synthetic():
            for cap in _capacities(buffers):
                self.assertEqual(
                    self._run(buffers, cap),
                    self._run(buffers, cap),
                    f"{case}[{gi}] cap={cap}",
                )


class ImprovementSmokeTest(TestCase):
    """At a tight capacity the search should usually *improve* on the seed for at
    least one captured graph -- evidence the moves actually do something, beyond
    the (trivially satisfied) >=-baseline guarantee. Not asserted per-graph (a
    graph whose seed is already optimal legitimately ties)."""

    def test_some_graph_improves_under_pressure(self):
        improved = False
        for case, gi, buffers in _all_cases():
            tot = _seed_footprint(buffers)
            for cap in (max(1, tot // 2), max(1, tot // 4)):
                solver = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
                solver.plan_layout_and_core_divisions()
                if solver.best_score < _baseline_score(buffers, cap):
                    improved = True
        self.assertTrue(improved, "SA never improved on the seed on any graph")


def _div(partition):
    """A core division with the given output partition (1 == trivial/whole)."""
    return CoreDivision(
        output_splits=({1: partition} if partition > 1 else {}),
        reduction_splits={},
    )


def _cdbuf(name, parents, matches, size=1024, uses=(0, 1)):
    """A minimal buffer with a 3-entry menu (index 0 trivial, 1 split-2, 2
    split-4) and the given parent-compatibility pairs. ``size`` / ``uses`` are
    overridable for the fixtures that need layout pressure (the flood tests do
    not care)."""
    return CoreDivisionBuffer(
        name=name,
        size=size,
        uses=list(uses),
        first_use_is_read=False,
        in_place_parents=[],
        residency_reason=None,
        core_divisions=[_div(1), _div(2), _div(4)],
        parents=parents,
        cd_parent_matches=matches,
        boundary=BufferType.Intermediate,
    )


def _flood(buffers, anchor_name, tiling):
    """Run ``_flood_region`` on a hand-built graph; return name -> chosen index."""
    solver = SaCoOptimizingSolver(buffers, 1 << 30, 128)
    solver._precompute_topology()
    result = solver._flood_region(solver._name_to_idx[anchor_name], tiling)
    return {buffers[i].name: d for i, d in result.items()}


class FloodRegionTest(TestCase):
    """The cd_parent_matches flood, on controlled synthetic graphs."""

    def test_chain_propagates_full_region(self):
        # A -> B -> C, every edge compatible at index 1: the split propagates end
        # to end.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 1)]}),
            _cdbuf("C", ["B"], {"B": [(1, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1), {"A": 1, "B": 1, "C": 1})

    def test_deterministic_tie_break_picks_smallest(self):
        # A's index 1 is compatible with both B-1 and B-2; the flood takes the
        # smallest, independent of pair list order.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 2), (1, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1)["B"], 1)

    def test_boundary_stops_flood(self):
        # The A->B edge carries no compatible pair for A's tiling 1 (only for 2),
        # so B is outside the region -- a boundary emerges for free.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(2, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1), {"A": 1})

    def test_upward_flood_reaches_parents(self):
        # Anchor the child; the flood must also go up the inverse relation.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 1)]}),
        ]
        self.assertEqual(_flood(bufs, "B", 1), {"A": 1, "B": 1})

    def test_join_accepts_internal_seam(self):
        # Diamond A->{B,C}->D with B,C forced to different indices. D is reachable
        # from both but assigned once (first-wins by frontier index: from B);
        # the C->D edge becomes an accepted internal seam, and the flood never
        # fails.
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A"], {"A": [(1, 1)]}),
            _cdbuf("C", ["A"], {"A": [(1, 2)]}),
            _cdbuf("D", ["B", "C"], {"B": [(1, 1)], "C": [(2, 2)]}),
        ]
        self.assertEqual(_flood(bufs, "A", 1), {"A": 1, "B": 1, "C": 2, "D": 1})


class RegionRecolorTest(TestCase):
    """Region-recolor on the real corpus: the flood spans genuine multi-op regions,
    and applying one is a coordinated division change the packer keeps up with."""

    def test_corpus_holds_multi_op_regions(self):
        # Floods every legal anchor/tiling on every graph rather than hoping the
        # search proposes one, so this is deterministic and independent of the
        # move weights. A corpus of singleton regions would make recolor pointless
        # and the bidirectional flood untested.
        largest = 0
        anchored = 0
        for case, gi, buffers in _all_cases_incl_synthetic():
            solver = _primed(copy.deepcopy(buffers), _seed_footprint(buffers))
            for anchor in solver._anchor_candidates:
                anchored += 1
                for tiling in solver._nontrivial_menu[anchor]:
                    largest = max(largest, len(solver._flood_region(anchor, tiling)))
        self.assertGreater(anchored, 0, "no graph offered a splittable anchor")
        self.assertGreater(largest, 1, "every region was a singleton")

    def test_recolor_recolors_the_whole_region_coherently(self):
        # After a recolor, every op the flood reached carries the flooded index and
        # the placement the packer holds for it reflects that division's footprint
        # -- i.e. the resize ripple in ``_apply_recolor`` reached everything
        # ``_flood_region`` assigned, not just the anchor.
        resized = 0
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            solver = _primed(copy.deepcopy(buffers), cap)
            for anchor in solver._anchor_candidates:
                tiling = solver._nontrivial_menu[anchor][0]
                assignment = solver._flood_region(anchor, tiling)
                solver._apply_recolor(assignment)
                addresses = solver.packer.addresses
                tag = f"{case}[{gi}] anchor={anchor}"
                for op, div in assignment.items():
                    self.assertEqual(solver.chosen[op], div, tag)
                    if addresses[op] is None:
                        continue  # spilled: the packer holds no extent to check
                    resized += 1
                    self.assertEqual(
                        solver.packer.top_or_inf(op) - addresses[op],
                        solver._per_core_size(op, div),
                        f"{tag}: packer footprint stale for op {op}",
                    )
        self.assertGreater(resized, 0, "no recolored op stayed resident")


def _chain(n=8):
    """A ``B0 -> ... -> B{n-1}`` chain, every edge compatible index-for-index, with
    varied sizes and staggered lifetimes -- so layout moves genuinely shift
    addresses and packer quality (equal-sized buffers sharing one lifetime are
    permutation-insensitive, which would make the assertions below vacuous)."""
    bufs = []
    for i in range(n):
        parents = [f"B{i - 1}"] if i else []
        matches = {f"B{i - 1}": [(0, 0), (1, 1), (2, 2)]} if i else {}
        bufs.append(
            _cdbuf(f"B{i}", parents, matches, size=1024 * (1 + i % 4), uses=[i, i + 3])
        )
    return bufs


def _chain_caps(buffers):
    """Roomy plus two spill-forcing capacities for a hand-built fixture."""
    tot = sum(b.size for b in buffers)
    return [tot, max(1, tot // 2), max(1, tot // 3)]


def _live_state(solver):
    """The observable joint state: layout addresses, packer quality, divisions,
    and the eligible count ``W`` implies (tracked, so a restore must rewind it)."""
    return (
        list(solver.packer.addresses),
        solver.packer.quality(),
        list(solver.chosen),
        solver._n_eligible,
    )


class SnapshotRestoreTest(TestCase):
    """``_adopt`` transfers ownership of a snapshot: it is the hot rejection path,
    where the snapshot was taken this iteration and dies with it, so a second O(n)
    packer copy would be pure overhead. What it must still do is restore the joint
    state exactly."""

    def _mutate(self, solver):
        """A division change (resize + eligibility ripple) plus a reinsertion --
        between them they move addresses, quality and ``chosen``."""
        solver._atomic_flip(2, 2)
        solver.packer.rotate(0, 5)

    def test_adopt_round_trips_state(self):
        for cap in _chain_caps(_chain()):
            solver = _primed(_chain(), cap)
            before = _live_state(solver)
            snap = solver._snapshot()
            self._mutate(solver)
            self.assertNotEqual(_live_state(solver), before, f"cap={cap}")
            solver._adopt(snap)  # snap is dead after this, by contract
            self.assertEqual(_live_state(solver), before, f"cap={cap}")


class StepBudgetTest(TestCase):
    """``clamp(_STEPS_PER_BUFFER * n, _MIN_STEPS, _MAX_STEPS)`` -- the same shape
    the layout-only annealer's schedule uses, so neither engine grows without
    bound."""

    @staticmethod
    def _budget(n):
        """The budget ``_anneal`` computes for ``n`` buffers."""
        return min(_MAX_STEPS, max(_MIN_STEPS, _STEPS_PER_BUFFER * n))

    def test_rate_applies_between_the_floor_and_the_ceiling(self):
        self.assertEqual(self._budget(100), _STEPS_PER_BUFFER * 100)

    def test_floor_applies_to_tiny_graphs(self):
        self.assertEqual(self._budget(1), _MIN_STEPS)

    def test_ceiling_caps_large_graphs(self):
        binds_at = _MAX_STEPS // _STEPS_PER_BUFFER
        self.assertEqual(self._budget(binds_at * 4), _MAX_STEPS)
        # Inert across the validated corpus: the largest captured graph is n=79,
        # far below where the ceiling starts binding.
        self.assertGreater(binds_at, 79)

    def test_ceiling_is_higher_than_the_layout_only_annealer(self):
        """The joint engine searches divisions too, so it wants a larger budget
        at the same buffer count (and must not silently inherit the smaller one).
        """
        from torch_spyre._inductor.scratchpad.cooling_schedules import (
            SelfCalibratingReheatingSchedule,
        )

        self.assertGreater(_MAX_STEPS, SelfCalibratingReheatingSchedule().max_steps)


def _n_eligible_recomputed(solver):
    """``_n_eligible`` from scratch -- the ground truth the incrementally tracked
    count is judged against."""
    return sum(solver._eligible(i) for i in range(len(solver._bufs)))


def _unsplittable_chain(n=4):
    """A chain whose buffers offer *no* alternative division (single-entry menus,
    trivial partition): no flip and no recolor anchor, so reorder is the only move
    the engine could ever propose."""
    bufs = []
    for i in range(n):
        parents = [f"B{i - 1}"] if i else []
        matches = {f"B{i - 1}": [(0, 0)]} if i else {}
        b = _cdbuf(f"B{i}", parents, matches, uses=[i, i + 3])
        b.core_divisions = [_div(1)]
        bufs.append(b)
    return bufs


class AllEligibleResidentTest(TestCase):
    """Once every eligible buffer is resident, ``pi`` has nothing left to win --
    it only decides *which* eligible buffers make LX -- so reorder is withdrawn
    from the proposal weights and a structural move's burst stops."""

    def test_gate_agrees_with_the_per_buffer_truth(self):
        # The O(1) count-vs-count test against the per-buffer definition it
        # stands in for, over roomy and spill-forcing capacities alike.
        for buffers in (_chain(), _chain(12)):
            for cap in [1 << 30] + _chain_caps(buffers):
                solver = _primed(buffers, cap)
                spilled = [
                    i
                    for i in range(len(buffers))
                    if solver._eligible(i) and solver.packer.addresses[i] is None
                ]
                self.assertEqual(
                    solver._all_eligible_resident(), not spilled, f"cap={cap}"
                )

    def test_reorder_is_withdrawn_only_when_all_eligible_are_resident(self):
        for cap in [1 << 30] + _chain_caps(_chain()):
            solver = _primed(_chain(), cap)
            self.assertEqual(
                "reorder" in solver._applicable_moves(),
                not solver._all_eligible_resident(),
                f"cap={cap}",
            )
            # The structural moves are unaffected by residency.
            self.assertIn("flip", solver._applicable_moves(), f"cap={cap}")
            self.assertIn("recolor", solver._applicable_moves(), f"cap={cap}")

    def test_burst_stops_at_the_first_all_resident_iteration(self):
        # A roomy capacity leaves the seed fully resident, so the burst returns
        # having drawn nothing and touched nothing; a tight one has it rotating.
        # The RNG state is the witness: the packer's methods are read-only on the
        # native build, so a rotate counter cannot be installed.
        tight = _chain_caps(_chain())[-1]
        for cap, expect_rotations in ((1 << 30, False), (tight, True)):
            solver = _primed(_chain(), cap)
            rng_before = solver._rng.getstate()
            perm_before = list(solver.packer.permutation)
            solver._burst()
            self.assertEqual(
                solver._rng.getstate() != rng_before, expect_rotations, f"cap={cap}"
            )
            if not expect_rotations:
                self.assertEqual(list(solver.packer.permutation), perm_before)

    def test_tracked_eligible_count_survives_a_move_storm(self):
        # Flips and recolors ripple eligibility over a buffer and its parents;
        # the tracked count differences that set, so it must still match a full
        # recompute after any sequence of them.
        for cap in _chain_caps(_chain(12)):
            solver = _primed(_chain(12), cap)
            self.assertEqual(solver._n_eligible, _n_eligible_recomputed(solver))
            for _ in range(50):
                if solver._rng.random() < 0.5:
                    idx = solver._rng.choice(solver._flippable_ops)
                    menu = len(solver._bufs[idx].core_divisions)
                    solver._atomic_flip(idx, solver._rng.randrange(menu))
                else:
                    solver._recolor()
                self.assertEqual(
                    solver._n_eligible, _n_eligible_recomputed(solver), f"cap={cap}"
                )

    def test_a_flip_that_shrinks_a_footprint_into_capacity_raises_the_count(self):
        # B7 is 4096 bytes and does not fit a 2048-byte scratchpad undivided, so
        # it is ineligible at menu index 0 and eligible at index 1 (partition 2).
        # It ends the chain, so no child edge can gate it, and its parent B6 is
        # already out on size (3072 > 2048) either way -- the count moves by
        # exactly the one buffer.
        solver = _primed(_chain(), 2048)
        idx = solver._name_to_idx["B7"]
        self.assertFalse(solver._eligible(idx))
        before = solver._n_eligible
        snap = solver._snapshot()
        solver._atomic_flip(idx, 1)
        self.assertTrue(solver._eligible(idx))
        self.assertFalse(solver._eligible(solver._name_to_idx["B6"]))
        self.assertEqual(solver._n_eligible, before + 1)
        self.assertEqual(solver._n_eligible, _n_eligible_recomputed(solver))
        solver._adopt(snap)
        self.assertEqual(solver._n_eligible, before)

    def test_a_graph_with_no_move_left_returns_the_seed(self):
        # Single-entry menus and a roomy capacity: reorder is withdrawn and
        # neither structural move applies, so the cool never starts.
        buffers = _unsplittable_chain()
        solver = _primed(buffers, 1 << 30)
        self.assertEqual(solver._applicable_moves(), [])
        self.assertEqual(solver._choose_move(), "none")

        solver = SaCoOptimizingSolver(copy.deepcopy(buffers), 1 << 30, 128)
        out = solver.plan_layout_and_core_divisions()
        self.assertEqual(solver.best_score, _baseline_score(buffers, 1 << 30))
        self.assertTrue(all(b.address is not None for b in out))

    def test_the_cool_stops_at_the_first_step_with_no_applicable_move(self):
        # Nothing can change the state once no move applies, so the remaining
        # budget is abandoned rather than spent on no-ops.
        solver = _primed(_chain(), _chain_caps(_chain())[-1])
        solver._calibrate_temperature = lambda: 1.0  # type: ignore[method-assign]
        moves = ["flip", "recolor", "none", "flip"]
        solver._choose_move = lambda: moves.pop(0)  # type: ignore[method-assign]
        stepped = []
        real_step = solver._step
        solver._step = lambda name, t, cur: (  # type: ignore[method-assign]
            stepped.append(name),
            real_step(name, t, cur),
        )[1]
        solver._anneal()
        self.assertEqual(stepped, ["flip", "recolor"])
        self.assertEqual(moves, ["flip"])  # the budget's tail went unused


# Snippet run in a subprocess to solve one graph (captured *or* synthetic, chosen
# by CASE) and print its result; used by the cross-process determinism test below.
_SOLVE_SNIPPET = """
import copy, json, math
from tests.inductor.cooptimization_capture_loader import load_captures
from tests.inductor.synthetic_cooptimization_graphs import synthetic_graphs
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver
case = {case!r}
src = load_captures() if case in load_captures() else synthetic_graphs()
g = src[case][0]
cap = max(1, sum(math.ceil(b.size / b.core_divisions[0].output_partition)
                 for b in g.buffers) // 2)
s = SaCoOptimizingSolver(copy.deepcopy(g.buffers), cap, 128)
out = s.plan_layout_and_core_divisions()
print("RESULT " + json.dumps({{
    "chosen": [b.chosen_division for b in out],
    "addr": [b.address for b in out],
    "best": s.best_score,
}}))
"""

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _solve_with_hashseed(hs, case="sdpa"):
    """Solve ``case`` in a subprocess with ``PYTHONHASHSEED=hs``."""
    env = dict(os.environ, PYTHONHASHSEED=str(hs), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    proc = subprocess.run(
        [sys.executable, "-c", _SOLVE_SNIPPET.format(case=case)],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


class CrossProcessDeterminismTest(TestCase):
    """The CI determinism test done right: solve twice in *separate processes*
    under different ``PYTHONHASHSEED`` values. In-process determinism tests share
    one hash seed and so cannot catch set-iteration-order bugs -- this one can (it
    caught the FirstFit seed nondeterminism)."""

    # ``sdpa`` is the richest captured graph (pins + reductions); ``big_chain`` is
    # the largest synthetic one (many regions / n~48), so between them they stress
    # the most set-ordered decisions (flood order, candidate lists, best-seen ties).
    def test_pythonhashseed_independent(self):
        for case in ("sdpa", "big_chain"):
            base = _solve_with_hashseed(0, case)
            for hs in (1, 2):
                self.assertEqual(
                    _solve_with_hashseed(hs, case), base, f"{case} PYTHONHASHSEED={hs}"
                )


@unittest.skipUnless(
    _large_captures_enabled(),
    f"large-capture experiments; set {_LARGE_CAPTURES_ENV}=1 to run",
)
class LargeCaptureTest(TestCase):
    """Opt-in (non-CI) coverage over the large 25-100 buffer captures: the engine
    must stay correct on big ``n``. Run with ``SA_COOPT_LARGE_CAPTURES=1``."""

    def test_large_graphs_valid_and_deterministic(self):
        for case, gi, buffers in _large_cases():
            cap = max(1, _seed_footprint(buffers) // 2)

            def run():
                s = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
                out = s.plan_layout_and_core_divisions()
                return (
                    [b.chosen_division for b in out],
                    [b.address for b in out],
                    s.best_score,
                )

            tag = f"{case}[{gi}]"
            a, b = run(), run()
            self.assertEqual(a, b, f"{tag} nondeterministic")
            s = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
            out = s.plan_layout_and_core_divisions()
            self.assertLessEqual(s.best_score, _baseline_score(buffers, cap), tag)
            self.assertEqual(_geometry_violations(out, cap, 128), [], tag)


def _score_after_rotate(s, i, j):
    """The objective reached by rotating position ``i`` to ``j``, leaving ``s``
    exactly as it was found."""
    snap = s._snapshot()
    s.packer.rotate(i, j)
    value = s._score()
    s._adopt(snap)  # a fresh snapshot is taken per call, so this transfer is safe
    return value


class ReorderSweepTest(TestCase):
    """The layout-only annealer's best-first reinsertion move, ported to the joint
    objective."""

    def test_probe_walk_leaves_the_packer_consistent(self):
        """The sweep walks the *live* packer and restores from the step snapshot,
        so a bookkeeping slip would show up as incremental state that disagrees
        with a packer rebuilt from scratch on the same permutation."""
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = _primed(copy.deepcopy(buffers), cap)
            if len(s._bufs) < 2:
                continue
            cur = s._score()
            for step in range(40):
                cur = s._step_reorder(1000.0, cur)
                tag = f"{case}[{gi}] step={step}"
                # The returned running score must be the state's real score.
                self.assertEqual(cur, s._score(), tag)
                # And the incrementally-maintained placement must match a
                # from-scratch rebuild on the permutation it ended up with.
                sizes = [s._per_core_size(i, s.chosen[i]) for i in range(len(s._bufs))]
                fresh = make_permutation_packer(
                    s._lifetime_buffers(sizes),
                    list(s.packer.permutation),
                    s.limit,
                    s.alignment,
                    eligible=[s._eligible(i) for i in range(len(s._bufs))],
                )
                self.assertEqual(list(fresh.addresses), list(s.packer.addresses), tag)
                self.assertEqual(fresh.quality(), s.packer.quality(), tag)

    def test_cold_sweep_takes_a_non_worsening_position_when_one_exists(self):
        """At a temperature that accepts nothing uphill, the sweep must accept some
        reinsertion whenever one does not worsen the score, and must leave the
        score untouched when every reachable position is uphill.

        Note it need not land on the *best* position: candidates are ranked by the
        packer's ``quality()`` proxy and the first to clear the Metropolis test
        wins, so a merely-equal position can beat the optimum to it.
        """
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = _primed(copy.deepcopy(buffers), cap)
            n = len(s._bufs)
            if n < 2:
                continue
            cur = s._score()
            for step in range(25):
                perm = s.packer.permutation
                allocated = [s.packer.is_fully_allocated(perm[k]) for k in range(n)]
                # Replay the source pick against a clone of the RNG so the brute
                # force below targets the same buffer the step will lift.
                probe_rng = copy.deepcopy(s._rng)
                saved, s._rng = s._rng, probe_rng
                i = s._choose_reinsertion_source(allocated)
                s._rng = saved
                upper = s._sweep_upper_bound(i, allocated)
                brute = [
                    _score_after_rotate(s, i, j) for j in range(upper + 1) if j != i
                ]
                before = cur
                cur = s._step_reorder(1e-12, cur)
                tag = f"{case}[{gi}] step={step} i={i}"
                if brute and min(brute) <= before:
                    self.assertLessEqual(cur, before, tag)
                else:
                    self.assertEqual(cur, before, tag)

    def test_monotonicity_bound_hides_no_better_position(self):
        """The sweep inherits the layout-only annealer's bound: an unallocated
        buffer is only probed up to the last allocated position + 1. Check on real
        graphs that nothing past the bound would have scored better."""
        for case, gi, buffers in _all_cases():
            cap = max(1, _seed_footprint(buffers) // 2)
            s = _primed(copy.deepcopy(buffers), cap)
            n = len(s._bufs)
            if n < 2:
                continue
            cur = s._score()
            for step in range(25):
                perm = s.packer.permutation
                allocated = [s.packer.is_fully_allocated(perm[k]) for k in range(n)]
                for i in range(n):
                    upper = s._sweep_upper_bound(i, allocated)
                    if upper >= n - 1:
                        continue  # unbounded; nothing was skipped
                    inside = min(
                        (
                            _score_after_rotate(s, i, j)
                            for j in range(upper + 1)
                            if j != i
                        ),
                        default=cur,
                    )
                    for j in range(upper + 1, n):
                        self.assertGreaterEqual(
                            _score_after_rotate(s, i, j),
                            inside,
                            f"{case}[{gi}] step={step} i={i} j={j} beat the bound",
                        )
                cur = s._step_reorder(1000.0, cur)


class ForeignParentTest(TestCase):
    """``parents`` naming buffers the solver does not own.

    ``_build_cd_bound_buffers`` assigns ``parents = info["op_inputs"]`` without
    intersecting the solver's buffer set, so graph inputs, constants and extern
    outputs land there on a real compile. This used to assert, which made the
    joint path unusable on 10 of the 11 corpus graphs.
    """

    def test_unowned_parent_is_skipped_not_asserted(self):
        bufs = [
            _cdbuf("A", [], {}),
            _cdbuf("B", ["A", "arg0_1"], {"A": [(1, 1)], "arg0_1": [(1, 1)]}),
        ]
        solver = SaCoOptimizingSolver(bufs, 1 << 30, 128)
        solver._precompute_topology()
        # The owned edge survives; the unowned one leaves no trace behind.
        self.assertEqual(solver._parents_idx[1], {0})
        self.assertEqual(solver._children_idx[0], [1])
        self.assertEqual(set(solver._edge_pairs), {(0, 1)})

    def test_graph_with_only_unowned_parents_still_solves(self):
        bufs = [
            _cdbuf("A", ["arg0_1"], {"arg0_1": [(1, 1)]}),
            _cdbuf("B", ["arg1_1"], {"arg1_1": [(2, 2)]}),
        ]
        solver = SaCoOptimizingSolver(bufs, 1 << 30, 128)
        out = solver.plan_layout_and_core_divisions()
        self.assertEqual(len(out), 2)
        self.assertTrue(all(b.chosen_division is not None for b in out))


class MemoryOnlyFallbackTest(TestCase):
    """With no live ``V.graph`` there are no per-division ``OpFeatures``, so the
    engine falls back to the memory-only spill-traffic objective.

    That is the path the whole capture-driven suite above runs on, so it has to be
    the memory-only formula exactly rather than an approximation of it. The
    cost-model objective is covered by ``test_cost_objective.py``.
    """

    def test_no_live_graph_means_no_cost_objective(self):
        buffers = [_cdbuf("A", [], {}), _cdbuf("B", ["A"], {"A": [(1, 1)]})]
        solver = SaCoOptimizingSolver(buffers, 1 << 30, 128)
        self.assertIsNone(solver._cost_objective)

    def test_fallback_scores_spilled_traffic_over_the_hbm_bandwidth(self):
        # Re-derives the objective from the returned layout, sharing nothing with
        # ``_score`` but the two constants: the differential spill cost of every
        # buffer that missed LX, converted by the HBM bandwidth.
        for case, gi, buffers in _all_cases():
            for cap in _capacities(buffers):
                solver = SaCoOptimizingSolver(copy.deepcopy(buffers), cap, 128)
                out = solver.plan_layout_and_core_divisions()
                traffic = 0
                for b in out:
                    if b.address is not None:
                        continue
                    reads = b.read_count - (1 if b.first_use_is_read else 0)
                    intermediate = b.boundary == BufferType.Intermediate
                    traffic += (reads + (1 if intermediate else 0)) * max(0, b.size)
                self.assertEqual(
                    solver.best_score,
                    utils.to_fixed_us(traffic / utils.hbm_bytes_per_us()),
                    f"{case}[{gi}] cap={cap}",
                )
