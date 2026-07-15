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

"""Tests for layout solvers"""

import unittest
from unittest import TestCase
from torch_spyre._inductor.scratchpad.plan_solver import (
    MemoryPlanSolver,
    CoreDivision,
    CoreDivisionBuffer,
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
)

try:
    from ortools.sat.python import cp_model  # noqa: F401

    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        CpSatLayoutSolver,
    )

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
    _assert_in_place_relationships,
    _topological_sort,
)

LARGE_SIZE = 512
SMALL_SIZE = 10
ALIGNMENT = 128


def _two_gap_buffers():
    """Buffers that leave two free gaps for x in a 120-byte scratchpad.

    Processing order by ascending lifetime: b_mid(2), b_left(4), b_right(5), x(5).
    b_right and x tie on lifetime; stable sort keeps b_right first.

    Placements: b_mid@0, b_left@40, b_right@70.
    b_mid lives [2,4) and x lives [4,9) — they do not overlap, so b_mid's
    address range (0,40) is not subtracted from x's gaps. After removing
    b_left(40,70) and b_right(70,100), x sees two gaps:
      (0,40)   waste = 30
      (100,120) waste = 10
    FirstFit picks (0,40) → addr=0; BestFit picks (100,120) → addr=100.
    """
    return [
        LifetimeBoundBuffer("b_mid", 40, [2, 3]),
        LifetimeBoundBuffer("b_left", 30, [1, 4]),
        LifetimeBoundBuffer("b_right", 30, [3, 7]),
        LifetimeBoundBuffer("x", 10, [4, 8]),
    ]


def _divs():
    # Two valid loop divisions: split output stride-256 axis four ways
    # (per-core footprint = total / 4), or keep the buffer whole.
    return [
        CoreDivision(output_splits={256: 4}),
        CoreDivision(),
    ]


def _whole():
    # A single whole-buffer division: per-core footprint == total size, so
    # no split can relieve capacity pressure and in-place reuse is actually
    # exercised (a /N split would shrink every footprint enough that the
    # merge -- a no-overlap relaxation used only when needed to fit -- would
    # never fire).
    return [CoreDivision()]


def _lifetimes_overlap(a, b) -> bool:
    return a.start_time < b.end_time and b.start_time < a.end_time


def _addr_overlap(a, b) -> bool:
    return a.address < b.address + b.size and b.address < a.address + a.size


class BaseLayoutSolverTests:
    solver_class: type[MemoryPlanSolver[LifetimeBoundBuffer]] = None  # type: ignore[assignment]

    def make_buffer(self, name, size, uses, **kwargs):
        """Build the buffer flavour the solver under test consumes.

        The placement-only solvers (greedy/first-fit/best-fit) read only the
        fields on :class:`LifetimeBoundBuffer`, so the base suite constructs
        those. :class:`JointDivisionSolverTests` overrides this to emit
        :class:`CoreDivisionBuffer`s carrying whole-buffer divisions, since the
        joint solver requires enumerated core divisions on every buffer.
        """
        return LifetimeBoundBuffer(name, size, uses, **kwargs)

    def solve(self, buffers, size=LARGE_SIZE, alignment=1):
        return self.solver_class(size, alignment).plan_layout(buffers)

    def check_result(self, result, expected_addresses, size, alignment):
        """Assert the solved layout matches the heuristic-solver expectation.

        :class:`JointDivisionSolverTests` overrides this: the joint solver is a
        satisfiability search, so it returns a *valid* packing rather than the
        specific addresses a gap heuristic picks, and it may evict a different
        buffer when capacity forces a spill. There the check is an invariant
        (no live overlap, within capacity, aligned) plus "places at least as
        many buffers as the heuristic", which holds because the solver minimises
        HBM transfers.
        """
        result_addresses = [p.address for p in result]
        if isinstance(expected_addresses, set):
            self.assertIn(tuple(result_addresses), expected_addresses)
        else:
            self.assertEqual(result_addresses, expected_addresses)

    def verify_layout(
        self,
        buffers: list[LifetimeBoundBuffer],
        expected_addresses: set[tuple[int | None]] | list[int | None],
        size=SMALL_SIZE,
        alignment=1,
    ):
        result = self.solve(buffers, size, alignment)
        self.check_result(result, expected_addresses, size, alignment)

    def test_simple_layout(self):
        # Three non-overlapping buffers fill memory sequentially.
        buffers = [
            self.make_buffer("buffer0", 3, [0, 1]),
            self.make_buffer("buffer1", 3, [0, 1]),
            self.make_buffer("buffer2", 4, [0, 1]),
        ]
        self.verify_layout(buffers, [0, 3, 6])

    def test_simple_layout_below_alignment(self):
        # Buffers smaller than the alignment boundary are evicted (address=None).
        buffers = [
            self.make_buffer("buffer0", 3, [0, 1]),
            self.make_buffer("buffer1", 3, [0, 1]),
            self.make_buffer("buffer2", 4, [0, 1]),
        ]
        self.verify_layout(buffers, [0, None, None], alignment=ALIGNMENT)

    def test_alignment_enforced(self):
        # Each buffer is placed at the next alignment boundary.
        buffers = [
            self.make_buffer("buffer0", 3, [0, 1]),
            self.make_buffer("buffer1", 3, [0, 1]),
            self.make_buffer("buffer2", 4, [0, 1]),
        ]
        self.verify_layout(buffers, [0, 128, 256], LARGE_SIZE, ALIGNMENT)

    def test_simple_eviction_layout(self):
        # buffer1 is evicted because it won't fit; buffer2 reuses buffer0's space.
        buffers = [
            self.make_buffer("buffer0", 7, [0, 1]),
            self.make_buffer("buffer1", 4, [0, 1]),
            self.make_buffer("buffer2", 3, [0, 1]),
        ]
        self.verify_layout(buffers, [0, None, 7])

    def test_realloc(self):
        # buffer1's lifetime starts after buffer0 ends, so it reuses address 0.
        buffers = [
            self.make_buffer("buffer0", 10, [0, 1]),
            self.make_buffer("buffer1", 3, [2]),
        ]
        self.verify_layout(buffers, [0, 0])

    def test_realloc_between(self):
        # buffer3's lifetime begins after buffer1 ends, so it reclaims buffer1's slot.
        buffers = [
            self.make_buffer("buffer0", 3, [0, 3]),
            self.make_buffer("buffer1", 3, [1, 2]),
            self.make_buffer("buffer2", 3, [2, 3]),
            self.make_buffer("buffer3", 3, [3]),
        ]
        self.verify_layout(buffers, {(0, 3, 6, 3), (6, 0, 3, 0)})

    def test_realloc_between_with_alignment(self):
        # Same reuse pattern as test_realloc_between, but with alignment padding applied.
        buffers = [
            self.make_buffer("buffer0", 200, [0, 3]),
            self.make_buffer("buffer1", 100, [1, 2]),
            self.make_buffer("buffer2", 100, [2, 3]),
            self.make_buffer("buffer3", 100, [3]),
        ]
        if self.solver_class == GreedyLayoutSolver:
            self.verify_layout(buffers, [0, 256, 384, 256], LARGE_SIZE, ALIGNMENT)
        else:
            # Other solvers are smarter than greedy
            self.verify_layout(buffers, [256, 0, 128, 0], LARGE_SIZE, ALIGNMENT)

    def test_inplace_allocation(self):
        # Test that adding inplace options allows for more efficient peak usage
        buffers = [
            self.make_buffer("buffer0", LARGE_SIZE, [0, 3]),
            self.make_buffer("buffer1", LARGE_SIZE, [3], in_place_parents=["buffer0"]),
        ]
        self.verify_layout(buffers, [0, 0], LARGE_SIZE + 1, ALIGNMENT)

    def test_without_inplace_allocation(self):
        # Test that buffer gets evicted without in_place
        buffers = [
            self.make_buffer("buffer0", LARGE_SIZE, [0, 3]),
            self.make_buffer("buffer1", LARGE_SIZE, [3]),
        ]
        self.verify_layout(buffers, {(0, None), (None, 0)}, LARGE_SIZE, ALIGNMENT)

    def test_multiple_evictions_do_not_corrupt_allocation(self):
        # buffer0 fills the entire scratchpad; buffer1 and buffer2 are evicted.
        # buffer3 starts after buffer0 ends and should reclaim address 0.
        buffers = [
            self.make_buffer("buffer0", SMALL_SIZE, [0, 1]),
            self.make_buffer("buffer1", SMALL_SIZE, [0, 1]),
            self.make_buffer("buffer2", SMALL_SIZE, [0, 1]),
            self.make_buffer("buffer3", SMALL_SIZE, [2]),
        ]
        self.verify_layout(buffers, [0, None, None, 0])

    def test_first_buffer_exceeds_limit_is_evicted(self):
        # A buffer whose size exceeds the scratchpad limit must be evicted even
        # when no other allocation is live (usage is empty, so address 0 would
        # otherwise be returned without the limit guard).
        buffers = [
            self.make_buffer("buffer0", SMALL_SIZE + 1, [0, 1]),
        ]
        self.verify_layout(buffers, [None], size=SMALL_SIZE)

    def test_empty_returns_empty_list(self):
        self.assertEqual(self.solve([]), [])

    def test_single_buffer_placed_at_zero(self):
        self.verify_layout([self.make_buffer("a", 10, [0, 4])], [0])

    def test_single_buffer_evicted_when_too_large(self):
        self.verify_layout([self.make_buffer("a", 11, [0, 4])], [None], size=10)

    def test_non_overlapping_lifetimes_reuse_address(self):
        # b1 ends at time 5 (exclusive); b2 starts at time 5 — they never coexist.
        self.verify_layout(
            [
                self.make_buffer("b1", 20, [0, 4]),
                self.make_buffer("b2", 20, [5, 9]),
            ],
            [0, 0],
            size=LARGE_SIZE,
        )

    def test_concurrent_buffers_packed_input_order(self):
        # Equal lifetimes: stable sort preserves input order, so a(10)@0, b(20)@10, c(30)@30.
        self.verify_layout(
            [
                self.make_buffer("a", 10, [0, 3]),
                self.make_buffer("b", 20, [0, 3]),
                self.make_buffer("c", 30, [0, 3]),
            ],
            [0, 10, 30],
            size=60,
        )

    def test_largest_buffer_evicted_when_full(self):
        # a(10)@0 and b(20)@10 consume 30 bytes; c(30) needs 30 but only 20 remain → evicted.
        self.verify_layout(
            [
                self.make_buffer("a", 10, [0, 3]),
                self.make_buffer("b", 20, [0, 3]),
                self.make_buffer("c", 30, [0, 3]),
            ],
            [0, 10, None],
            size=50,
        )

    def test_alignment_pads_between_buffers(self):
        # Two same-size concurrent buffers; the second is placed at the next
        # alignment boundary after the first.
        self.verify_layout(
            [
                self.make_buffer("a", 10, [0, 3]),
                self.make_buffer("b", 10, [0, 3]),
            ],
            [0, 128],
            alignment=128,
            size=LARGE_SIZE,
        )

    def test_alignment_can_cause_eviction(self):
        # a(13)@0 leaves a gap starting at 13; rounding up to alignment=10 gives
        # addr=20, but 20+12=32 > limit=30, so b is evicted.
        self.verify_layout(
            [
                self.make_buffer("a", 13, [0, 4]),
                self.make_buffer("b", 12, [0, 4]),
            ],
            [0, None],
            size=30,
            alignment=10,
        )

    def test_child_reuses_parent_address(self):
        # P.end_time==5 (uses[-1]+1); C.start_time==4 (uses[0]); 5==4+1, so in-place is valid.
        # Without in-place, P's [0,20) would be subtracted and C would land at 20.
        self.verify_layout(
            [
                self.make_buffer("P", 20, [0, 4]),
                self.make_buffer("C", 15, [4, 8], in_place_parents=["P"]),
            ],
            [0, 0],
            size=LARGE_SIZE,
        )

    def test_child_falls_back_when_parent_evicted(self):
        # P is too large to fit; C declared as in-place child of P.
        # P gets evicted (address=None), so C also cannot in-place and
        # must fall back to normal placement (placed independently, no overlap
        # conflict with evicted P).
        self.verify_layout(
            [
                self.make_buffer("P", 200, [0, 4]),
                self.make_buffer("C", 15, [4, 8], in_place_parents=["P"]),
            ],
            [None, 0],
            size=100,
        )

    def test_assert_rejects_wrong_end_time(self):
        p = LifetimeBoundBuffer("P", 20, [0, 4])
        c = LifetimeBoundBuffer(
            "C", 15, [3, 8], in_place_parents=["P"]
        )  # uses[0]=3, need P.uses[-1]+1==4
        with self.assertRaises(AssertionError):
            _assert_in_place_relationships([p, c])

    def test_assert_rejects_oversized_child(self):
        p = LifetimeBoundBuffer("P", 10, [0, 4])
        c = LifetimeBoundBuffer(
            "C", 15, [4, 8], in_place_parents=["P"]
        )  # child larger than parent
        with self.assertRaises(AssertionError):
            _assert_in_place_relationships([p, c])


class ScoreOrderingTests:
    """Tests for the priority-score ordering in FirstFit/BestFit.

    Buffers are placed in ascending order of ``(span - discount) / len(uses)``
    (lower = placed first), where ``discount`` is 0.25 per in-place
    relationship. These tests isolate the two terms the old
    shortest-lifetime-first heuristic ignored: ``len(uses)`` and the in-place
    discount. They do not apply to the Greedy solver, whose time-stepped
    plan_layout does not score buffers.
    """

    def test_higher_use_count_placed_first(self):
        # Two buffers with the same span (5) fully overlap and contend for the
        # single slot that fits one of them. Score is span / len(uses), so the
        # buffer with more uses scores lower and is placed first, winning the
        # slot — regardless of input order. The old span-only tiebreak would
        # tie (both span 5) and keep input order, pinning `few` instead.
        many = LifetimeBoundBuffer("many", 10, [0, 1, 2, 3, 4])  # 5 / 5 = 1.0
        few = LifetimeBoundBuffer("few", 10, [0, 4])  # 5 / 2 = 2.5
        # `few` first in input order, to prove ordering is by score not input.
        result = self.solve([few, many], size=10)
        by_name = {b.name: b.address for b in result}
        self.assertEqual(by_name["many"], 0)
        self.assertIsNone(by_name["few"])

    def test_inplace_discount_raises_priority(self):
        # `plain` and `parent` have identical span (5) and use count (2), so
        # their base scores tie at 2.5. `parent` is an in-place parent of
        # `child`, earning a 0.25 discount → score (5 - 0.25) / 2 = 2.375 <
        # 2.5, so it is placed first and wins the single contested slot. The
        # old heuristic would tie and keep input order, pinning `plain`.
        plain = LifetimeBoundBuffer("plain", 10, [0, 4])  # 5 / 2 = 2.5
        parent = LifetimeBoundBuffer("parent", 10, [0, 4])  # (5 - 0.25) / 2
        child = LifetimeBoundBuffer("child", 10, [4, 8], in_place_parents=["parent"])
        # `plain` first in input order; without the discount the tie would
        # keep input order and pin `plain`.
        result = self.solve([plain, parent, child], size=10)
        by_name = {b.name: b.address for b in result}
        self.assertEqual(by_name["parent"], 0)
        self.assertEqual(by_name["child"], 0)  # child reuses parent's slot
        self.assertIsNone(by_name["plain"])

    def test_write_first_buffer_placed_before_read_only(self):
        # Identical span (5) and use count (2). `writer`'s first use is a write
        # (first_use_is_read=False), so pinning it also saves the more expensive
        # first write to HBM; its use count is inflated by 0.5, giving score
        # 5 / 2.5 = 2.0 vs `reader`'s 5 / 2.0 = 2.5. `writer` is placed first and
        # wins the single contested slot. Without the write bonus the scores tie
        # and input order would pin `reader`.
        reader = LifetimeBoundBuffer("reader", 10, [0, 4], first_use_is_read=True)
        writer = LifetimeBoundBuffer("writer", 10, [0, 4], first_use_is_read=False)
        # `reader` first in input order, to prove ordering is by score not input.
        result = self.solve([reader, writer], size=10)
        by_name = {b.name: b.address for b in result}
        self.assertEqual(by_name["writer"], 0)
        self.assertIsNone(by_name["reader"])


class TestFirstFitLayoutSolver(ScoreOrderingTests, BaseLayoutSolverTests, TestCase):
    solver_class = FirstFitLayoutSolver

    def test_picks_first_gap_not_tightest(self):
        result = self.solver_class(120, 1).plan_layout(_two_gap_buffers())
        x_addr = next(b.address for b in result if b.name == "x")
        self.assertEqual(x_addr, 0)


class TestBestFitLayoutSolver(ScoreOrderingTests, BaseLayoutSolverTests, TestCase):
    solver_class = BestFitLayoutSolver

    def test_picks_tightest_gap(self):
        result = self.solver_class(120, 1).plan_layout(_two_gap_buffers())
        x_addr = next(b.address for b in result if b.name == "x")
        self.assertEqual(x_addr, 100)


class JointDivisionSolverTests(BaseLayoutSolverTests):
    """Shared tests for a joint core-division solver: it picks each buffer's
    division from its candidate list while keeping producer/consumer slicing
    consistent. This is a mixin (no ``TestCase`` base, so it is not collected on
    its own); concrete subclasses set ``solver_class``.

    It also inherits ``BaseLayoutSolverTests``. Those shared tests model flat
    placement (every live buffer gets an address), which the joint solver's
    LX-residency model does not: a buffer resides only if a consumer reads it
    from scratchpad, and the solver returns *a* valid packing rather than the
    specific addresses a gap heuristic picks. The overrides below bridge the
    gap -- ``make_buffer`` emits division-carrying buffers, ``solve`` attaches a
    synthetic sink so the base buffers are residency-eligible, and
    ``check_result`` validates the packing is legal and at least as full as the
    heuristic rather than asserting exact addresses.
    """

    def make_buffer(self, name, size, uses, **kwargs):
        # The joint solver requires every buffer to carry at least one core
        # division; the base suite's buffers are undivided, so attach the
        # whole-buffer division. Any in-place / producer edges a base test
        # declares need matching pairs so the slicing gate permits the
        # merge/residency (whole <-> whole is the (0, 0) pair).
        edges = set(kwargs.get("in_place_parents", [])) | set(kwargs.get("parents", []))
        matches = {p: [(0, 0)] for p in edges}
        matches.update(kwargs.pop("cd_parent_matches", {}))
        return CoreDivisionBuffer(
            name,
            size,
            uses,
            core_divisions=_whole(),
            cd_parent_matches=matches,
            **kwargs,
        )

    def solve(self, buffers, size=LARGE_SIZE, alignment=1):
        # The base buffers carry no consumer edges, so under the residency gate
        # the solver would force-spill every one of them. Attach a single
        # synthetic sink that consumes them all, making each residency-eligible.
        # The sink has no consumer of its own, so it is always spilled and
        # occupies nothing; strip it from the returned layout.
        if not buffers:
            return []
        if size // alignment < 1:
            # Below one alignment unit the solver's unit-scaled capacity rounds to
            # zero and the solver can't represent any placement; the layout is
            # trivially all-spilled, so return the buffers unplaced directly.
            return buffers
        names = [b.name for b in buffers]
        last = max(max(b.uses) for b in buffers)
        sink = CoreDivisionBuffer(
            "__sink__",
            1,
            [last + 1],
            core_divisions=_whole(),
            parents=names,
            cd_parent_matches={n: [(0, 0)] for n in names},
        )
        result = self.solver_class(size, alignment).plan_layout(buffers + [sink])
        return [b for b in result if b.name != "__sink__"]

    def check_result(self, result, expected_addresses, size, alignment):
        placed = [b for b in result if b.address is not None]
        # A legal packing: every placed buffer is aligned and within capacity.
        for b in placed:
            self.assertEqual(b.address % alignment, 0, f"{b.name} misaligned")
            self.assertLessEqual(b.address + b.size, size, f"{b.name} exceeds capacity")
        # No two lifetime-overlapping buffers may share addresses, except
        # in-place pairs, which intentionally share storage for the single tick
        # their lifetimes touch.
        for a in placed:
            for c in placed:
                if a.name == c.name:
                    continue
                if not _lifetimes_overlap(a, c):
                    continue
                if a.name in c.in_place_parents or c.name in a.in_place_parents:
                    continue
                self.assertFalse(
                    _addr_overlap(a, c), f"{a.name} and {c.name} overlap in memory"
                )
        # The solver minimises spilled HBM traffic, so it places at least as many buffers as the
        # heuristic expectation. Below one alignment unit of capacity the solver's
        # unit model rounds to zero and can't represent any placement, so the
        # count comparison does not apply.
        if size // alignment >= 1:
            expected = (
                next(iter(expected_addresses))
                if isinstance(expected_addresses, set)
                else expected_addresses
            )
            min_placed = sum(1 for a in expected if a is not None)
            self.assertGreaterEqual(len(placed), min_placed)

    def test_layout_with_inplace(self):
        # A producer->consumer chain (A->B->...->TERMINAL) gives every buffer a
        # consumer edge so it may reside; whole-only divisions keep footprint ==
        # size so capacity pressure forces the in-place merge to fire. P reuses
        # its in-place parent G's storage. The chain tail TERMINAL is
        # consumer-less and is correctly force-spilled by the no-consumer
        # residency gate, so it is the one buffer left without an address.
        # parents / cd_parent_matches are set by the chain loop below, so only
        # the fields that differ from the defaults are passed here: every buffer
        # carries the whole-only division, and P declares its in-place parents.
        buffers = [
            CoreDivisionBuffer("A", 60, [0, 2], core_divisions=_whole()),
            CoreDivisionBuffer("B", 30, [1, 4], core_divisions=_whole()),
            CoreDivisionBuffer("C", 30, [2, 13], core_divisions=_whole()),
            CoreDivisionBuffer("D", 30, [3, 4], core_divisions=_whole()),
            CoreDivisionBuffer("E", 30, [4, 5], core_divisions=_whole()),
            CoreDivisionBuffer("F", 60, [5, 6], core_divisions=_whole()),
            CoreDivisionBuffer("G", 30, [6, 15], core_divisions=_whole()),
            CoreDivisionBuffer("H", 30, [7, 8], core_divisions=_whole()),
            CoreDivisionBuffer("I", 30, [8, 9], core_divisions=_whole()),
            CoreDivisionBuffer("J", 15, [9, 16], core_divisions=_whole()),
            CoreDivisionBuffer("K", 15, [10, 12], core_divisions=_whole()),
            CoreDivisionBuffer("L", 15, [11, 12], core_divisions=_whole()),
            CoreDivisionBuffer("M", 15, [12, 13], core_divisions=_whole()),
            CoreDivisionBuffer("N", 30, [13, 15], core_divisions=_whole()),
            CoreDivisionBuffer("O", 45, [14, 15], core_divisions=_whole()),
            CoreDivisionBuffer(
                "P",
                30,
                [15, 16],
                in_place_parents=["G", "N"],
                core_divisions=_whole(),
            ),
            CoreDivisionBuffer("Q", 75, [16, 17], core_divisions=_whole()),
            CoreDivisionBuffer("TERMINAL", 75, [17, 18], core_divisions=_whole()),
        ]
        for i in range(1, len(buffers)):
            buffers[i].cd_parent_matches = {buffers[i - 1].name: [(0, 0)]}
            buffers[i].parents = [buffers[i - 1].name]
        buffers_by_name = {b.name: b for b in buffers}
        # The in-place merge gate matches on cd_parent_matches, not on the linear
        # parents chain, so P's in-place parents G/N need explicit pairs too.
        buffers_by_name["P"].cd_parent_matches.update({"G": [(0, 0)], "N": [(0, 0)]})

        results = self.solver_class(size=120, alignment=1).plan_layout(buffers)
        results_by_name = {b.name: b for b in results}
        # Every buffer is placed except the consumer-less chain tail TERMINAL.
        self.assertTrue(all(b.address is not None for b in results[:-1]))
        self.assertIsNone(results_by_name["TERMINAL"].address)
        # P reuses its in-place parent G's storage.
        self.assertEqual(results_by_name["P"].address, results_by_name["G"].address)

    def test_unset_core_divisions_raises(self):
        # The solver no longer has a placement-only fallback for unset divisions:
        # every buffer must carry at least one enumerated core division (the real
        # allocator always supplies at least the whole-buffer division), so an
        # undivided buffer is a usage error caught up front.
        plain = [
            CoreDivisionBuffer("x", 60, [0, 1]),
            CoreDivisionBuffer("y", 60, [1, 2]),
        ]
        with self.assertRaises(AssertionError):
            self.solver_class(size=120, alignment=1).plan_layout(plain)

    def test_picks_matching_division_to_fit(self):
        # Producer P (total 400) feeds consumer C (total 400); both overlap in
        # time so the whole (partition 1) division can't fit in 256. The only
        # feasible plan splits both /4 (per-core 100 each) AND picks the same
        # slicing signature so the shared buffer is locality-clean.
        P = CoreDivisionBuffer("P", 400, [0, 1], core_divisions=_divs())
        C = CoreDivisionBuffer(
            "C",
            400,
            [1, 3],
            core_divisions=_divs(),
            parents=["P"],
            cd_parent_matches={"P": [(0, 0), (1, 1)]},
        )
        # Give C a downstream consumer so it isn't force-spilled by no_consumer.
        D = CoreDivisionBuffer(
            "D",
            100,
            [3, 4],
            core_divisions=_divs(),
            parents=["C"],
            cd_parent_matches={"C": [(0, 0), (1, 1)]},
        )
        result = {
            b.name: b
            for b in self.solver_class(size=256, alignment=1).plan_layout([P, C, D])
        }

        self.assertIsNotNone(result["P"].address)
        self.assertIsNotNone(result["C"].address)
        # P resident => its division matches a consumer (C); both chose the /4
        # clean split, so their signatures agree.
        p_cd = result["P"].core_divisions[result["P"].chosen_division]
        c_cd = result["C"].core_divisions[result["C"].chosen_division]
        self.assertEqual(p_cd.signature_key(), c_cd.signature_key())
        self.assertEqual(p_cd.output_partition, 4)

    def test_no_consumer_division_buffer_is_spilled(self):
        # A buffer that carries divisions but has no local consumer edge can
        # never match anything, so it is force-spilled even when it would fit.
        leaf = CoreDivisionBuffer("leaf", 40, [0, 1], core_divisions=_divs())
        result = self.solver_class(size=256, alignment=1).plan_layout([leaf])
        self.assertIsNone(result[0].address)

    def test_oversized_min_footprint_is_spilled(self):
        # Even the smallest candidate footprint (total/4 = 250) exceeds the
        # tiny capacity, so the buffer is force-spilled.
        P = CoreDivisionBuffer("P", 1000, [0, 1], core_divisions=_divs())
        C = CoreDivisionBuffer(
            "C",
            1000,
            [1, 2],
            core_divisions=_divs(),
            parents=["P"],
        )
        result = {
            b.name: b
            for b in self.solver_class(size=200, alignment=1).plan_layout([P, C])
        }
        self.assertIsNone(result["P"].address)


@unittest.skipUnless(_HAS_ORTOOLS, "ortools not installed")
class TestCpSatJointDivision(JointDivisionSolverTests, TestCase):
    """The OR-Tools CP-SAT joint core-division + LX placement solver. Runs the
    shared ``JointDivisionSolverTests`` (legal packing, residency under the
    slicing gate, in-place reuse) via ``self.solver_class``.

    The cases below target CP-SAT-specific encoding details: in-place reuse is
    modelled by *shortening the parent's lifetime* by the handoff tick under a
    global ``AddNoOverlap2D``, so they pin down chain-sharing and the degenerate
    single-use (zero-width) parent."""

    solver_class = CpSatLayoutSolver

    def test_inplace_chain_shares_single_slot(self):
        # A 3-level in-place chain gp -> p -> c (each parent.end_time ==
        # child.start_time + 1) with whole-only divisions and capacity for just
        # one 100-byte buffer. gp & p are live together, as are p & c, so
        # without reuse the peak is 200 and something spills. The only feasible
        # full-residency plan merges the whole chain onto one shared slot --
        # exactly what the lifetime-shortening encoding must allow. A sink keeps
        # the tail residency-eligible.
        gp = CoreDivisionBuffer("gp", 100, [0, 1], core_divisions=_whole())
        p = CoreDivisionBuffer(
            "p", 100, [1, 2], core_divisions=_whole(), in_place_parents=["gp"]
        )
        c = CoreDivisionBuffer(
            "c", 100, [2, 3], core_divisions=_whole(), in_place_parents=["p"]
        )
        sink = CoreDivisionBuffer("__sink__", 1, [4], core_divisions=_whole())
        chain = [gp, p, c, sink]
        for i in range(1, len(chain)):
            chain[i].parents = [chain[i - 1].name]
            chain[i].cd_parent_matches = {chain[i - 1].name: [(0, 0)]}
        res = {
            b.name: b
            for b in self.solver_class(size=150, alignment=1).plan_layout(chain)
        }
        # The whole chain resides, sharing one address (the sink spills: no
        # consumer of its own).
        for n in ("gp", "p", "c"):
            self.assertIsNotNone(res[n].address, f"{n} should reside")
        self.assertEqual(res["gp"].address, res["p"].address)
        self.assertEqual(res["p"].address, res["c"].address)

    def test_single_use_parent_inplace_zero_width(self):
        # Degenerate case: the parent has a single use (uses=[0]), so its only
        # live tick IS the handoff. Shortening it by that tick yields a
        # zero-width time interval, which the 2D propagator ignores -- the child
        # still holds the shared slot. The merge must fire (capacity fits only
        # one 100-byte buffer) and the child must reuse the parent's address.
        gp = CoreDivisionBuffer("gp", 100, [0], core_divisions=_whole())  # end_time=1
        c = CoreDivisionBuffer(
            "c",
            100,
            [0, 1],
            core_divisions=_whole(),
            in_place_parents=["gp"],
            parents=["gp"],
            cd_parent_matches={"gp": [(0, 0)]},
        )
        sink = CoreDivisionBuffer(
            "__sink__",
            1,
            [2],
            core_divisions=_whole(),
            parents=["c"],
            cd_parent_matches={"c": [(0, 0)]},
        )
        res = {
            b.name: b
            for b in self.solver_class(size=150, alignment=1).plan_layout([gp, c, sink])
        }
        self.assertIsNotNone(res["gp"].address, "single-use parent should reside")
        self.assertIsNotNone(res["c"].address, "child should reside")
        self.assertEqual(res["gp"].address, res["c"].address)

    def test_spill_reasons_recorded(self):
        # The solver records a per-buffer drop cause for every spilled buffer so
        # the allocator can report why each landed in HBM. `leaf` has divisions
        # but no consumer edge (forced out by the residency gate); `big`'s
        # smallest footprint exceeds capacity (forced out up front).
        leaf = CoreDivisionBuffer("leaf", 40, [0, 1], core_divisions=_whole())
        big = CoreDivisionBuffer(
            "big", 1000, [0, 1], core_divisions=[CoreDivision(output_splits={256: 4})]
        )
        C = CoreDivisionBuffer(
            "C",
            100,
            [1, 2],
            core_divisions=[CoreDivision(output_splits={256: 4})],
            parents=["big"],
        )
        solver = self.solver_class(size=200, alignment=1)
        result = {b.name: b for b in solver.plan_layout([leaf, big, C])}

        # All three spill; each carries a reason keyed by buffer name.
        self.assertIsNone(result["big"].address)
        self.assertIn("big", solver.spill_reasons)
        self.assertIn("capacity", solver.spill_reasons["big"])
        self.assertIn("leaf", solver.spill_reasons)
        self.assertIn("no consumer", solver.spill_reasons["leaf"])
        # A resident buffer gets no spill reason.
        for name, buf in result.items():
            self.assertEqual(buf.address is None, name in solver.spill_reasons)


@unittest.skipUnless(_HAS_ORTOOLS, "cpsat placement unit tests need ortools")
class TestCpSatUnallocatedReads(TestCase):
    """Device-free coverage of the CP-SAT objective/residency gate for the
    placement-only path.

    ``_as_core_division_buffers`` hands the solver single-fixed-division buffers
    (the placement-only path: each buffer's only ``CoreDivision`` is the division
    the upstream passes already committed, so the solver cannot re-divide -- it
    only places) and records reads by consumers outside the candidate set
    (filtered-out ops, graph outputs) as ``unallocated_reads``. These check that
    such a read is enough to pin a buffer that has no candidate children, that a
    truly-unread buffer is still forced out, that an ordinary parent edge pins,
    and -- since the conversion now derives each edge's match from the two ops'
    fixed divisions -- that an edge whose divisions disagree (empty
    ``cd_parent_matches``) does *not* pin the producer. All without a Spyre device.
    """

    def _mk(self, name, uses, parents=(), unallocated_reads=0, matches=None):
        """A single-fixed-division ``CoreDivisionBuffer`` as emitted by
        ``_as_core_division_buffers``. ``matches`` overrides the per-parent match
        pairs; by default every parent edge is compatible (``[(0, 0)]``), matching
        a producer/consumer whose fixed divisions slice the buffer identically.
        """
        if matches is None:
            matches = {p: [(0, 0)] for p in parents}
        return CoreDivisionBuffer(
            name,
            128,
            list(uses),
            first_use_is_read=False,
            core_divisions=[CoreDivision()],
            parents=list(parents),
            cd_parent_matches=matches,
            unallocated_reads=unallocated_reads,
        )

    def _pinned(self, bufs):
        out = CpSatLayoutSolver(1 << 20).plan_layout(bufs)
        return {b.name for b in out if b.address is not None}

    def test_only_unallocated_reads_is_pinned(self):
        """A buffer read solely by a non-candidate consumer (no children) is
        pinned on the strength of its unallocated read."""
        self.assertIn("b0", self._pinned([self._mk("b0", [0, 1], unallocated_reads=1)]))

    def test_no_reads_is_not_pinned(self):
        """A buffer with no children and no unallocated reads is forced to HBM
        (nothing reads it from LX)."""
        self.assertNotIn("b0", self._pinned([self._mk("b0", [0, 1])]))

    def test_candidate_parent_edge_still_pins(self):
        """The ordinary producer->consumer edge still pins the producer."""
        pinned = self._pinned(
            [self._mk("b0", [0, 1]), self._mk("b1", [1, 2], parents=["b0"])]
        )
        self.assertIn("b0", pinned)

    def test_mismatched_fixed_divisions_do_not_pin(self):
        """When the producer and consumer fixed divisions slice the shared buffer
        differently, ``_as_core_division_buffers`` records an *empty* match for
        that edge. The producer then has no compatible child and no unallocated
        read, so the solver declines to pin it (it falls back to HBM) even though
        the consumer still lists it as a parent."""
        pinned = self._pinned(
            [
                self._mk("b0", [0, 1]),
                self._mk("b1", [1, 2], parents=["b0"], matches={"b0": []}),
            ]
        )
        self.assertNotIn("b0", pinned)


class TestGreedyLayoutSolver(BaseLayoutSolverTests, TestCase):
    solver_class = GreedyLayoutSolver


class TestTopologicalSort(TestCase):
    """Tests for the Kahn's-algorithm sort that orders in-place chains.

    These call the module-level helper directly with arbitrary lifetimes/sizes;
    the in-place invariants enforced elsewhere (parent.end_time ==
    child.start_time + 1, child.size <= parent.size) are not required here,
    since _topological_sort only consumes in_place_parents for edges.
    """

    @staticmethod
    def _names(buffers, f):
        return [b.name for b in _topological_sort(buffers, f)]

    def test_multi_level_chain_orders_parents_before_children(self):
        # A 3-level in-place chain gp -> p -> c. Each level has exactly one
        # ready node at a time, so topology alone fixes the order regardless of
        # the tie-break key or input order.
        gp = LifetimeBoundBuffer("gp", 100, 0, 2)
        p = LifetimeBoundBuffer("p", 100, 2, 4, in_place_parents=["gp"])
        c = LifetimeBoundBuffer("c", 100, 4, 6, in_place_parents=["p"])
        # Pass the inputs out of order to prove the result is driven by the
        # in-place edges, not the input order.
        self.assertEqual(self._names([c, p, gp], lambda b: 0), ["gp", "p", "c"])

    def test_tie_break_key_applied_below_the_root_frontier(self):
        # Regression test for the bug where the `f` tie-break was applied only
        # to the initial (root) frontier; nodes unlocked deeper in the sort
        # fell back to a hardcoded lifetime key.
        #
        # Chain root -> mid -> {a, b}. After root and mid are popped, a and b
        # become ready at the SAME step (third level), so their relative order
        # is decided purely by the tie-break key.
        #
        # f sorts ascending by size, so the smaller buffer `a` must come first.
        # `a` deliberately has the LONGER lifetime, so the old lifetime-keyed
        # tie-break would (incorrectly) emit `b` before `a`.
        root = LifetimeBoundBuffer("root", 100, 0, 1)
        mid = LifetimeBoundBuffer("mid", 100, 1, 2, in_place_parents=["root"])
        a = LifetimeBoundBuffer(
            "a", 1, 2, 102, in_place_parents=["mid"]
        )  # size 1, lifetime 100
        b = LifetimeBoundBuffer(
            "b", 100, 2, 3, in_place_parents=["mid"]
        )  # size 100, lifetime 1

        self.assertEqual(
            self._names([root, mid, a, b], lambda buf: buf.size),
            ["root", "mid", "a", "b"],
        )

        # Reversing the key flips a and b, confirming the key — not lifetime or
        # input order — drives the deep tie-break.
        self.assertEqual(
            self._names([root, mid, a, b], lambda buf: -buf.size),
            ["root", "mid", "b", "a"],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
