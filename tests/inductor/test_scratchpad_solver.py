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

from unittest import TestCase
from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
)
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
        LifetimeBoundBuffer("b_mid", 40, 2, 4),
        LifetimeBoundBuffer("b_left", 30, 1, 5),
        LifetimeBoundBuffer("b_right", 30, 3, 8),
        LifetimeBoundBuffer("x", 10, 4, 9),
    ]


class BaseLayoutSolverTests:
    solver_class: type[FirstFitLayoutSolver] = None  # type: ignore[assignment]

    def solve(self, buffers, size=LARGE_SIZE, alignment=1):
        return self.solver_class(size, alignment).plan_layout(buffers)

    def verify_layout(
        self,
        buffers: list[LifetimeBoundBuffer],
        expected_addresses: set[tuple[int | None]] | list[int | None],
        size=SMALL_SIZE,
        alignment=1,
    ):
        result = self.solve(buffers, size, alignment)
        result_addresses = [p.address for p in result]
        if isinstance(expected_addresses, set):
            self.assertIn(tuple(result_addresses), expected_addresses)
        else:
            self.assertEqual(result_addresses, expected_addresses)

    def test_simple_layout(self):
        # Three non-overlapping buffers fill memory sequentially.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 0, 2),
            LifetimeBoundBuffer("buffer2", 4, 0, 2),
        ]
        self.verify_layout(buffers, [0, 3, 6])

    def test_simple_layout_below_alignment(self):
        # Buffers smaller than the alignment boundary are evicted (address=None).
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 0, 2),
            LifetimeBoundBuffer("buffer2", 4, 0, 2),
        ]
        self.verify_layout(buffers, [0, None, None], alignment=ALIGNMENT)

    def test_alignment_enforced(self):
        # Each buffer is placed at the next alignment boundary.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 0, 2),
            LifetimeBoundBuffer("buffer2", 4, 0, 2),
        ]
        self.verify_layout(buffers, [0, 128, 256], LARGE_SIZE, ALIGNMENT)

    def test_simple_eviction_layout(self):
        # buffer1 is evicted because it won't fit; buffer2 reuses buffer0's space.
        buffers = [
            LifetimeBoundBuffer("buffer0", 7, 0, 2),
            LifetimeBoundBuffer("buffer1", 4, 0, 2),
            LifetimeBoundBuffer("buffer2", 3, 0, 2),
        ]
        self.verify_layout(buffers, [0, None, 7])

    def test_realloc(self):
        # buffer1's lifetime starts after buffer0 ends, so it reuses address 0.
        buffers = [
            LifetimeBoundBuffer("buffer0", 10, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 2, 3),
        ]
        self.verify_layout(buffers, [0, 0])

    def test_realloc_between(self):
        # buffer3's lifetime begins after buffer1 ends, so it reclaims buffer1's slot.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 4),
            LifetimeBoundBuffer("buffer1", 3, 1, 3),
            LifetimeBoundBuffer("buffer2", 3, 2, 4),
            LifetimeBoundBuffer("buffer3", 3, 3, 4),
        ]
        self.verify_layout(buffers, {(0, 3, 6, 3), (6, 0, 3, 0)})

    def test_realloc_between_with_alignment(self):
        # Same reuse pattern as test_realloc_between, but with alignment padding applied.
        buffers = [
            LifetimeBoundBuffer("buffer0", 200, 0, 4),
            LifetimeBoundBuffer("buffer1", 100, 1, 3),
            LifetimeBoundBuffer("buffer2", 100, 2, 4),
            LifetimeBoundBuffer("buffer3", 100, 3, 4),
        ]
        if self.solver_class == GreedyLayoutSolver:
            self.verify_layout(buffers, [0, 256, 384, 256], LARGE_SIZE, ALIGNMENT)
        else:
            # Other solvers are smarter than greedy
            self.verify_layout(buffers, [256, 0, 128, 0], LARGE_SIZE, ALIGNMENT)

    def test_inplace_allocation(self):
        # Test that adding inplace options allows for more efficient peak usage
        buffers = [
            LifetimeBoundBuffer("buffer0", LARGE_SIZE, 0, 4),
            LifetimeBoundBuffer(
                "buffer1", LARGE_SIZE, 3, 4, in_place_parents=["buffer0"]
            ),
        ]
        self.verify_layout(buffers, [0, 0], LARGE_SIZE + 1, ALIGNMENT)

    def test_without_inplace_allocation(self):
        # Test that buffer gets evicted without in_place
        buffers = [
            LifetimeBoundBuffer("buffer0", LARGE_SIZE, 0, 4),
            LifetimeBoundBuffer("buffer1", LARGE_SIZE, 3, 4),
        ]
        self.verify_layout(buffers, {(0, None), (None, 0)}, LARGE_SIZE, ALIGNMENT)

    def test_multiple_evictions_do_not_corrupt_allocation(self):
        # buffer0 fills the entire scratchpad; buffer1 and buffer2 are evicted.
        # buffer3 starts after buffer0 ends and should reclaim address 0.
        buffers = [
            LifetimeBoundBuffer("buffer0", SMALL_SIZE, 0, 2),
            LifetimeBoundBuffer("buffer1", SMALL_SIZE, 0, 2),
            LifetimeBoundBuffer("buffer2", SMALL_SIZE, 0, 2),
            LifetimeBoundBuffer("buffer3", SMALL_SIZE, 2, 3),
        ]
        self.verify_layout(buffers, [0, None, None, 0])

    def test_first_buffer_exceeds_limit_is_evicted(self):
        # A buffer whose size exceeds the scratchpad limit must be evicted even
        # when no other allocation is live (usage is empty, so address 0 would
        # otherwise be returned without the limit guard).
        buffers = [
            LifetimeBoundBuffer("buffer0", SMALL_SIZE + 1, 0, 2),
        ]
        self.verify_layout(buffers, [None], size=SMALL_SIZE)

    def test_empty_returns_empty_list(self):
        self.assertEqual(self.solve([]), [])

    def test_single_buffer_placed_at_zero(self):
        self.verify_layout([LifetimeBoundBuffer("a", 10, 0, 5)], [0])

    def test_single_buffer_evicted_when_too_large(self):
        self.verify_layout([LifetimeBoundBuffer("a", 11, 0, 5)], [None], size=10)

    def test_non_overlapping_lifetimes_reuse_address(self):
        # b1 ends at time 5 (exclusive); b2 starts at time 5 — they never coexist.
        self.verify_layout(
            [LifetimeBoundBuffer("b1", 20, 0, 5), LifetimeBoundBuffer("b2", 20, 5, 10)],
            [0, 0],
            size=LARGE_SIZE,
        )

    def test_concurrent_buffers_packed_input_order(self):
        # Equal lifetimes: stable sort preserves input order, so a(10)@0, b(20)@10, c(30)@30.
        self.verify_layout(
            [
                LifetimeBoundBuffer("a", 10, 0, 4),
                LifetimeBoundBuffer("b", 20, 0, 4),
                LifetimeBoundBuffer("c", 30, 0, 4),
            ],
            [0, 10, 30],
            size=60,
        )

    def test_largest_buffer_evicted_when_full(self):
        # a(10)@0 and b(20)@10 consume 30 bytes; c(30) needs 30 but only 20 remain → evicted.
        self.verify_layout(
            [
                LifetimeBoundBuffer("a", 10, 0, 4),
                LifetimeBoundBuffer("b", 20, 0, 4),
                LifetimeBoundBuffer("c", 30, 0, 4),
            ],
            [0, 10, None],
            size=50,
        )

    def test_alignment_pads_between_buffers(self):
        # Two same-size concurrent buffers; the second is placed at the next
        # alignment boundary after the first.
        self.verify_layout(
            [LifetimeBoundBuffer("a", 10, 0, 4), LifetimeBoundBuffer("b", 10, 0, 4)],
            [0, 128],
            alignment=128,
            size=LARGE_SIZE,
        )

    def test_alignment_can_cause_eviction(self):
        # a(13)@0 leaves a gap starting at 13; rounding up to alignment=10 gives
        # addr=20, but 20+12=32 > limit=30, so b is evicted.
        self.verify_layout(
            [LifetimeBoundBuffer("a", 13, 0, 5), LifetimeBoundBuffer("b", 12, 0, 5)],
            [0, None],
            size=30,
            alignment=10,
        )

    def test_child_reuses_parent_address(self):
        # P ends at 5; C.start_time=4 == P.end_time - 1, so in-place is valid.
        # Without in-place, P's [0,20) would be subtracted and C would land at 20.
        p = LifetimeBoundBuffer("P", 20, 0, 5)
        c = LifetimeBoundBuffer("C", 15, 4, 9, in_place_parents=["P"])
        result = self.solve([p, c])
        by_name = {b.name: b.address for b in result}
        self.assertEqual(by_name["P"], 0)
        self.assertEqual(by_name["C"], 0)

    def test_child_falls_back_when_parent_evicted(self):
        # P is too large to fit; C declared as in-place child of P.
        # P gets evicted (address=None), so C also cannot in-place and
        # must fall back to normal placement.
        p = LifetimeBoundBuffer("P", 200, 0, 5)
        c = LifetimeBoundBuffer("C", 15, 4, 9, in_place_parents=["P"])
        result = self.solve([p, c], size=100)
        by_name = {b.name: b.address for b in result}
        self.assertIsNone(by_name["P"])
        # C can still be placed independently (no overlap conflict with evicted P).
        self.assertEqual(by_name["C"], 0)

    def test_assert_rejects_wrong_end_time(self):
        p = LifetimeBoundBuffer("P", 20, 0, 5)
        c = LifetimeBoundBuffer(
            "C", 15, 3, 9, in_place_parents=["P"]
        )  # start_time=3, need P.end_time==4
        with self.assertRaises(AssertionError):
            _assert_in_place_relationships([p, c])

    def test_assert_rejects_oversized_child(self):
        p = LifetimeBoundBuffer("P", 10, 0, 5)
        c = LifetimeBoundBuffer(
            "C", 15, 4, 9, in_place_parents=["P"]
        )  # child larger than parent
        with self.assertRaises(AssertionError):
            _assert_in_place_relationships([p, c])


class TestFirstFitLayoutSolver(BaseLayoutSolverTests, TestCase):
    solver_class = FirstFitLayoutSolver

    def test_picks_first_gap_not_tightest(self):
        result = self.solver_class(120, 1).plan_layout(_two_gap_buffers())
        x_addr = next(b.address for b in result if b.name == "x")
        self.assertEqual(x_addr, 0)


class TestBestFitLayoutSolver(BaseLayoutSolverTests, TestCase):
    solver_class = BestFitLayoutSolver

    def test_picks_tightest_gap(self):
        result = self.solver_class(120, 1).plan_layout(_two_gap_buffers())
        x_addr = next(b.address for b in result if b.name == "x")
        self.assertEqual(x_addr, 100)


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
