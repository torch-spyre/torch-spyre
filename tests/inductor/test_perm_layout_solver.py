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

"""Tests for the capacity-bounded allocation plans."""

import copy
import gc
import itertools
import os
import random
import unittest
from unittest import TestCase
from typing import TYPE_CHECKING

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.scratchpad.contact_profile import Profile
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    ReferencePermutationBasedLayoutSolver,
)
from torch_spyre._C import NativePermutationLayoutSolver

ALIGNMENT = 128

# Exhaustive randomized differential runs: thousands of seeds, larger problems,
# dense in-place wiring. Skipped by default (slow); opt in with the env var.
_STRESS = os.environ.get("TORCH_SPYRE_STRESS_SCRATCHPAD") == "1"


def _random_buffers(rng, n, horizon=12, max_size=200, inplace_prob=0.25):
    """Generate ``n`` random buffers, occasionally wiring in-place pairs.

    Lifetimes are half-open ``[start, end)`` and non-empty (end > start).
    """
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        buffers.append(_buf(f"b{i}", size, start, end))
    # Turn a few buffers into in-place children of an earlier buffer: the child
    # starts at the parent's last live tick (parent.end - 1, so that
    # parent.end == child.start + 1) and clamps its size to fit.
    for child_i in range(1, n):
        if rng.random() < inplace_prob:
            parent_i = rng.randrange(child_i)
            parent = buffers[parent_i]
            child = buffers[child_i]
            if parent.read_count == 0:
                # A write-only parent has nothing to hand over, so the pair is
                # not expressible at all (see ``check_in_place_parent_is_read``).
                # Drawing one is possible because a base buffer may land on a
                # single-tick lifetime; skip rather than reshaping the parent,
                # which could invalidate a pair already wired to it.
                continue
            # The child is defined at the parent's last live tick, so that
            # parent.end_time == child.start_time + 1. Its own read has to fall
            # strictly after that write, both because uses is strictly increasing
            # and so that the child can itself act as an in-place parent; extend
            # by a tick when the child's own end is too early to supply one.
            handoff = parent.end_time - 1
            child.uses = [handoff, max(child.end_time - 1, handoff + 1)]
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _check_consistency(test, plan, tag=""):
    """Verify the live contact profiles: each is canonical and spans its
    buffer's lifetime, and the reverse-pointer invariant holds in both
    directions (iterating segments, not sampling columns):
    ``below_profile[x] == z`` over a segment iff ``above_profile[z] == x`` there.
    """
    n = len(plan.buffers)
    for i in range(n):
        for prof in (plan.below_profile[i], plan.above_profile[i]):
            prof.validate()
            test.assertEqual(prof.span_start, plan.buffers[i].start_time, tag)
            test.assertEqual(prof.span_end, plan.buffers[i].end_time, tag)
    for primary, reverse in (
        (plan.below_profile, plan.above_profile),
        (plan.above_profile, plan.below_profile),
    ):
        for x in range(n):
            p = primary[x]
            for k, z in enumerate(p.labels):
                if z is None:
                    continue
                a, b = p.starts[k], p.starts[k + 1]
                _, rev_labels = reverse[z].segments(a, b)
                for lab in rev_labels:
                    test.assertEqual(lab, x, f"{tag} reverse-pointer x={x} z={z}")


def _check_contact_faithful(test, plan, tag=""):
    """Verify ``contact_at(c, t)`` describes the slot directly below ``c`` at
    every column ``t``. In all cases ``c`` rests on the named buffer (the
    ``parent`` of a tuple), i.e. the max-top among ``c``'s earlier-positioned
    candidates alive there. A ``(parent, child)`` tuple appears iff that slot is
    shared by an in-place pair at its transition column: ``parent`` and ``child``
    are co-located (same address), both alive at ``t``, linked by reuse, and
    ``c``'s order-below neighbour is the higher-positioned of the two (which one
    that is depends on the reuse direction).
    """
    n = len(plan.buffers)
    pos = plan.position

    def top(w):
        # Exclusive top of a placed buffer; +inf for an evicted (None) one. Uses
        # the plan-local size so a resize is reflected.
        if plan.addresses[w] is None:
            return float("inf")
        return plan.addresses[w] + plan._sizes[w]

    def alive(w):
        return plan.buffers[w].start_time <= t < plan.buffers[w].end_time

    for c in range(n):
        bc = plan.buffers[c]
        if plan.addresses[c] is None:
            # c is evicted (or ineligible / in HBM): no concrete geometry to
            # validate. (Its order-based contact relation is still checked via
            # _check_consistency and the placed buffers below.)
            continue
        for t in range(bc.start_time, bc.end_time):
            contact = plan.contact_at(c, t)
            # Only eligible earlier buffers are in the stack (an ineligible one is
            # transparent), matching the candidate set placement actually uses.
            cand = [
                w
                for w in plan.overlap_dict[c]
                if pos[w] < pos[c] and alive(w) and plan._eligible[w]
            ]
            if not cand:
                test.assertIsNone(contact, f"{tag} c={c} t={t}")
                continue
            max_top = max(top(w) for w in cand)
            if isinstance(contact, tuple):
                parent, child = contact
                # c rests on the parent (the larger, max-top co-located buffer).
                test.assertEqual(top(parent), max_top, f"{tag} c={c} t={t} parent")
                # parent/child are an in-place pair, co-located, both alive at t,
                # linked by reuse, and one of them is c's order-below neighbour.
                test.assertEqual(
                    plan._in_place_pair(parent, child),
                    (parent, child),
                    f"{tag} c={c} t={t} pair",
                )
                test.assertEqual(
                    plan.addresses[parent], plan.addresses[child], f"{tag} colocated"
                )
                test.assertTrue(alive(parent) and alive(child), f"{tag} both-alive")
                test.assertTrue(
                    plan.inplace_reuse.get(parent) == child
                    or plan.inplace_reuse.get(child) == parent,
                    f"{tag} c={c} t={t} reuse-link",
                )
                test.assertIn(
                    plan.below_profile[c].label_at(t),
                    (parent, child),
                    f"{tag} c={c} t={t} order-below",
                )
            else:
                test.assertEqual(top(contact), max_top, f"{tag} c={c} t={t} int")


def _named_segments(plan, profile):
    """A profile as a readable list of (start, end, neighbour-name-or-None)."""
    return [
        (
            profile.starts[k],
            profile.starts[k + 1],
            None if profile.labels[k] is None else plan.buffers[profile.labels[k]].name,
        )
        for k in range(len(profile.labels))
    ]


def _below_named(plan, name):
    return _named_segments(plan, plan.below_profile[plan._name_to_idx[name]])


def _above_named(plan, name):
    return _named_segments(plan, plan.above_profile[plan._name_to_idx[name]])


def _buf(name, size, start, end, in_place_parents=None):
    # A lifetime of [start, end) is expressed as a write at start and a read at
    # end - 1. When end == start + 1 those are the same operation, so the buffer
    # has a single use: uses carries one distinct index per accessing op (see
    # LifetimeBoundBuffer), and such a buffer is written without being read.
    uses = [start] if end - 1 == start else [start, end - 1]
    return LifetimeBoundBuffer(
        name=name,
        size=size,
        uses=uses,
        in_place_parents=in_place_parents or [],
    )


def _rebuilt_with_state(plan, capacity, alignment):
    """A fresh ``PermutationBasedLayoutSolver`` reproducing ``plan``'s current
    permutation, *sizes*, and *eligibility* -- the from-scratch oracle for the
    resize / set_eligible differential checks.

    ``resize`` mutates the plan-local ``_sizes`` (never the shared buffer
    objects), so the rebuild is fed fresh buffers carrying those sizes; the live
    eligibility is passed through the constructor. This is the analogue of the
    ``rebuilt = PermutationBasedLayoutSolver(buffers, fast.permutation, ...)``
    oracle the swap/rotate tests use, extended to the co-opt state.
    """
    fresh = [
        LifetimeBoundBuffer(
            name=b.name,
            size=plan._sizes[i],
            uses=list(b.uses),
            first_use_is_read=b.first_use_is_read,
            in_place_parents=list(b.in_place_parents),
        )
        for i, b in enumerate(plan.buffers)
    ]
    return PermutationBasedLayoutSolver(
        fresh,
        list(plan.permutation),
        capacity,
        alignment,
        eligible=list(plan._eligible),
    )


# Both concrete plans share PermutationBasedLayoutSolverBase, so the Step 1 skeleton
# behaviour (field setup, helpers, finalize) is identical and tested for both.
if TYPE_CHECKING:
    MixinBase = TestCase
else:
    MixinBase = object


class SkeletonTestsMixin(MixinBase):
    plan_class: type = None  # type: ignore[assignment]

    def make_plan(self, buffers, permutation, capacity, alignment=ALIGNMENT):
        return self.plan_class(buffers, permutation, capacity, alignment)

    def test_init_stores_fields(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        plan = self.make_plan(buffers, [1, 0], capacity=256, alignment=ALIGNMENT)

        self.assertIs(plan.buffers, buffers)
        self.assertEqual(plan.permutation, [1, 0])
        self.assertEqual(plan.capacity, 256)
        self.assertEqual(plan.alignment, ALIGNMENT)
        self.assertEqual(plan._name_to_idx, {"a": 0, "b": 1})
        # addresses has one slot per buffer; its contents depend on _build.
        self.assertEqual(len(plan.addresses), 2)

    def test_permutation_is_copied(self):
        buffers = [_buf("a", 64, 0, 1)]
        perm = [0]
        plan = self.make_plan(buffers, perm, capacity=128)
        perm.append(99)
        self.assertEqual(plan.permutation, [0])

    def test_invalid_permutation_rejected(self):
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 0, 1)]
        with self.assertRaises(ValueError):
            self.make_plan(buffers, [0, 0], capacity=128)
        with self.assertRaises(ValueError):
            self.make_plan(buffers, [0], capacity=128)

    def test_single_use_input_in_place_parent_allowed(self):
        # The same shape is legal when the parent is a graph input: all its uses
        # are reads, so one use still means a genuine read before the handoff.
        parent = LifetimeBoundBuffer("a", 64, [0], first_use_is_read=True)
        child = LifetimeBoundBuffer("b", 64, [0, 2], in_place_parents=["a"])
        plan = self.make_plan([parent, child], [0, 1], capacity=256)
        self.assertEqual(plan._inplace_partners, [{1}, {0}])

    def test_in_place_chain_allowed(self):
        # Chains are explicitly supported: "b" is both a child of "a" and the
        # parent of "c". Every parent here has a read, so nothing is rejected.
        a = LifetimeBoundBuffer("a", 64, [0, 2])
        b = LifetimeBoundBuffer("b", 64, [2, 4], in_place_parents=["a"])
        c = LifetimeBoundBuffer("c", 64, [4, 6], in_place_parents=["b"])
        plan = self.make_plan([a, b, c], [0, 1, 2], capacity=256)
        self.assertEqual(plan._inplace_partners, [{1}, {0, 2}, {1}])

    def test_align_up(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128, alignment=128)
        self.assertEqual(plan._align_up(0), 0)
        self.assertEqual(plan._align_up(1), 128)
        self.assertEqual(plan._align_up(128), 128)
        self.assertEqual(plan._align_up(129), 256)

    def test_top(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=256)
        plan.addresses[0] = 128
        self.assertEqual(plan._top(0), 192)
        # An evicted buffer (no address) has no top.
        plan.addresses[0] = None
        self.assertIsNone(plan._top(0))

    def test_is_fully_allocated(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=100)
        # None is the single source of truth for eviction: a concrete address
        # means allocated, None means evicted (the capacity gate now lives in
        # placement, not here).
        plan.addresses[0] = 36
        self.assertTrue(plan.is_fully_allocated(0))
        plan.addresses[0] = 0
        self.assertTrue(plan.is_fully_allocated(0))
        plan.addresses[0] = None
        self.assertFalse(plan.is_fully_allocated(0))

    def test_quality_accessor(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128)
        plan.total_quality = 42
        plan.total_allocated_count = 3
        self.assertEqual(plan.quality(), 42)
        self.assertEqual(plan.count_allocated(), 3)

    def test_finalize_writes_back_only_fully_allocated(self):
        buffers = [
            _buf("fits", 64, 0, 1),
            _buf("evicted", 64, 0, 1),
            _buf("also_evicted", 64, 0, 1),
        ]
        plan = self.make_plan(buffers, [0, 1, 2], capacity=128)
        # addresses is already the single source of truth: a concrete address for
        # a placed buffer, None for an evicted one. finalize is a direct copy.
        plan.addresses = [0, None, None]

        plan.finalize()

        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)
        self.assertIsNone(buffers[2].address)


class ReferenceSolverSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = PermutationBasedLayoutSolver


def _addr(plan, name):
    return plan.addresses[plan._name_to_idx[name]]


class ReferencePlacementTests(TestCase):
    """Step 2: O(n^2) placement in ReferencePermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return ReferencePermutationBasedLayoutSolver(
            buffers, permutation, capacity, alignment
        )

    def test_disjoint_lifetimes_all_at_zero(self):
        # No two buffers are ever alive together, so each reuses address 0.
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3), _buf("c", 64, 4, 5)]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual([_addr(plan, n) for n in "abc"], [0, 0, 0])
        # Each buffer lives for a single tick, so it has one use (its write) and
        # weight 1 + 0.5. Contrast test_overlapping_lifetimes_stack, whose
        # two-tick buffers carry a write and a read and so weigh 2 + 0.5.
        self.assertEqual(plan.quality(), 288)  # 1.5 * (64 + 64 + 64)

    def test_overlapping_lifetimes_stack(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)  # stacked on top of a
        self.assertEqual(plan.quality(), 285)  # 2.5 * (64 + 50)

    def test_permutation_order_changes_layout(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [1, 0])  # b placed first
        self.assertEqual(_addr(plan, "b"), 0)
        self.assertEqual(_addr(plan, "a"), 50)  # a stacked on top of b

    def test_alignment_rounds_up(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], alignment=128)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 128)  # ceil(64/128)*128

    def test_freed_low_space_reused_by_later_disjoint_buffer(self):
        # a dies before c starts; c does not overlap a, so it drops back to 0
        # even though b (overlapping both) sits above.
        buffers = [
            _buf("a", 64, 0, 2),
            _buf("b", 64, 1, 5),
            _buf("c", 64, 3, 5),
        ]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)
        # c overlaps b (not a) -> stacks only on b.
        self.assertEqual(_addr(plan, "c"), 128)

    def test_in_place_child_reuses_parent_address(self):
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 0)  # reuses parent's address
        self.assertEqual(plan.quality(), 480)  # 2.5 * (128 + 64)

    def test_in_place_parent_placed_after_child_reuses_address(self):
        # Symmetric case: child allocated first, parent reuses its address.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [1, 0])  # child first
        self.assertEqual(_addr(plan, "c"), 0)
        self.assertEqual(_addr(plan, "p"), 0)  # parent reuses child's address

    def test_in_place_blocked_when_child_larger_than_parent(self):
        parent = _buf("p", 64, 0, 5)
        child = _buf("c", 128, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 64)  # cannot reuse; stacks on top

    def test_in_place_blocked_by_intruding_buffer(self):
        # The collision case from the design discussion: Z coexists with the
        # child but not the parent, so reusing the parent's address would
        # overlap Z. Placement must fall back to stacking.
        parent = _buf("p", 50, 0, 5)
        child = _buf("c", 30, 4, 10, in_place_parents=["p"])
        z = _buf("z", 20, 6, 10)
        plan = self.plan([parent, child, z], [0, 2, 1])  # order: p, z, c
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "z"), 0)  # z does not overlap p
        # c overlaps both p (top 50) and z (top 20); p is topmost and is the
        # in-place partner, but reusing addr 0 would hit z -> stack at 50.
        self.assertEqual(_addr(plan, "c"), 50)

    def test_over_capacity_buffer_evicted(self):
        # b would stack above a and cross the capacity line, so it is evicted:
        # its address is None and it contributes nothing to quality/count.
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertIsNone(_addr(plan, "b"))  # 64 + 64 = 128 > 100 -> evicted
        self.assertEqual(plan.quality(), 160)  # only a counts: 2.5 * 64
        self.assertEqual(plan.count_allocated(), 1)

    def test_finalize_after_build(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.finalize()
        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)  # over capacity, not committed


class ContactProfileTests(TestCase):
    """Order-based contact profiles in PermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_simple_stack(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual(_below_named(plan, "b"), [(1, 3, "a")])
        self.assertEqual(_above_named(plan, "a"), [(0, 1, None), (1, 3, "b")])
        self.assertEqual(_below_named(plan, "a"), [(0, 3, None)])
        self.assertEqual(_above_named(plan, "b"), [(1, 3, None)])

    def test_air_gap_neighbor(self):
        # low spans the whole range; tall is between low and high in order over
        # [0,3); after tall dies, high's below-neighbour is low (over an address
        # air gap -- the order-based relation does not care about the gap).
        low = _buf("low", 64, 0, 10)
        tall = _buf("tall", 256, 0, 3)
        high = _buf("high", 64, 0, 10)
        plan = self.plan([low, tall, high], [0, 1, 2])
        self.assertEqual(_below_named(plan, "high"), [(0, 3, "tall"), (3, 10, "low")])

    def test_in_place_pair_ordered_by_position(self):
        # Parent before child in the permutation -> parent below child over the
        # shared boundary tick, even though the child reuses the parent's slot.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_below_named(plan, "c"), [(4, 5, "p"), (5, 10, None)])
        self.assertEqual(_above_named(plan, "p"), [(0, 4, None), (4, 5, "c")])

    def test_contact_at_floor_and_plain_stack(self):
        # a is on the floor; b rests on a (plain int, no in-placement).
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        a, b = plan._name_to_idx["a"], plan._name_to_idx["b"]
        self.assertIsNone(plan.contact_at(a, 0))
        self.assertIsNone(plan.contact_at(a, 2))
        self.assertEqual(plan.contact_at(b, 1), a)
        self.assertEqual(plan.contact_at(b, 2), a)

    def test_contact_at_poke_through_tuple(self):
        # Order p, c, h. c in-places onto p; p (taller) pokes through, so h
        # rests on p at the shared tick (4) and is reported as (p, c). After p
        # dies (t >= 5) h rests on the now-plain child c.
        p = _buf("p", 128, 0, 5)
        c = _buf("c", 64, 4, 10, in_place_parents=["p"])
        h = _buf("h", 32, 4, 10)
        plan = self.plan([p, c, h], [0, 1, 2])
        ip, ic, ih = (plan._name_to_idx[n] for n in ("p", "c", "h"))
        self.assertEqual(_addr(plan, "c"), 0)  # reuses p
        self.assertEqual(_addr(plan, "h"), 128)  # rests on p, not on c
        self.assertEqual(plan.contact_at(ih, 4), (ip, ic))  # poke-through tuple
        self.assertEqual(plan.contact_at(ih, 5), ic)  # p gone -> plain child
        self.assertEqual(plan.contact_at(ih, 9), ic)
        self.assertEqual(plan.contact_at(ic, 4), ip)  # c's own contact is p
        self.assertIsNone(plan.contact_at(ip, 0))  # p on the floor

    def test_contact_at_parent_reused_child_tuple(self):
        # The other reuse direction: the child is placed first and the parent
        # reuses its slot, so the parent (larger) is h's order-below and the
        # child is buried inside it. At the transition tick the slot below h is
        # the pair (parent, child) -- the case the old semantics hid behind a
        # bare int. h still rests on the parent.
        p = _buf("p", 128, 0, 5)
        c = _buf("c", 64, 4, 10, in_place_parents=["p"])
        h = _buf("h", 32, 4, 6)
        plan = self.plan([p, c, h], [1, 0, 2])  # order: c, then p, then h
        ip, ic, ih = (plan._name_to_idx[n] for n in ("p", "c", "h"))
        self.assertEqual(_addr(plan, "c"), 0)
        self.assertEqual(_addr(plan, "p"), 0)  # parent reuses child's slot
        self.assertEqual(_addr(plan, "h"), 128)  # rests on the parent p
        self.assertEqual(plan.contact_at(ih, 4), (ip, ic))  # slot below = pair
        self.assertEqual(plan.contact_at(ih, 5), ic)  # p dead -> plain child
        self.assertEqual(plan.contact_at(ip, 4), ic)  # p's own order-below is c

    def test_contact_candidates_sufficient_exhaustive(self):
        """Exhaustively (all small configurations) verify the property that
        justifies the contact-based ``_recompute_address``: the candidate set
        built from ``contact_at`` over ``z``'s below-profile breakpoints yields
        the same ``_placement_decision`` as the full earlier-overlapping set.

        Placement reads neither capacity nor alignment, so the equivalence is
        independent of both and one of each suffices. ~82k configs, 33k of them
        carrying an in-place edge, in a few seconds. The heavier validation
        (n up to 4, larger sizes, live swap propagation) lives in StressTests
        and the one-off exhaustive sweep.

        ``horizon`` is 4 rather than 3 because requiring the parent to be read
        costs every in-place pair a tick: the parent must be written and then
        read before it hands over, so the earliest legal handoff is tick 1 and
        the earliest a child can start is 1, where it used to be 0.  (A child
        starting at 0 has no legal parent at any horizon -- that one is inherent
        to the rule, not a horizon effect.)  At horizon 3 there is no longer room
        for the shifted pairs and the wirings carrying an in-place edge collapse
        from 29k to 8k; horizon 4 gives them the tick back, at 33k.
        """

        def contact_cands(plan, z):
            s: set[int] = set()
            for t in plan.below_profile[z].starts[:-1]:
                co = plan.contact_at(z, t)
                if isinstance(co, tuple):
                    s.update(co)
                elif co is not None:
                    s.add(co)
            return list(s)

        def full_cands(plan, z):
            pz = plan.position[z]
            return [w for w in plan.overlap_dict[z] if plan.position[w] < pz]

        horizon = 4
        lifetimes = [(s, e) for s in range(horizon) for e in range(s + 1, horizon + 1)]
        for n in (2, 3):
            for life in itertools.product(lifetimes, repeat=n):
                starts = [s for s, _ in life]
                ends = [e for _, e in life]
                # in-place parent options: any other buffer whose lifetime makes
                # it a geometrically valid parent (parent.end == child.start + 1)
                # *and* that is read before the handoff. A single-tick parent has
                # only its write, so it has nothing to hand over and the pair is
                # not expressible (see ``check_in_place_parent_is_read``).
                parent_opts = [
                    [None]
                    + [
                        j
                        for j in range(n)
                        if j != i
                        and ends[j] == starts[i] + 1
                        and ends[j] - starts[j] > 1
                    ]
                    for i in range(n)
                ]
                for sizes in itertools.product((1, 2), repeat=n):
                    for wiring in itertools.product(*parent_opts):
                        ipp = [[f"b{j}"] if j is not None else [] for j in wiring]
                        for perm in itertools.permutations(range(n)):
                            bufs = [
                                _buf(
                                    f"b{i}",
                                    sizes[i],
                                    starts[i],
                                    ends[i],
                                    in_place_parents=list(ipp[i]),
                                )
                                for i in range(n)
                            ]
                            plan = PermutationBasedLayoutSolver(
                                bufs, list(perm), 10**9, 1
                            )
                            for z in range(n):
                                tag = (
                                    f"life={life} sizes={sizes} wiring={wiring} "
                                    f"perm={perm} z={z}"
                                )
                                self.assertEqual(
                                    plan._placement_decision(z, contact_cands(plan, z)),
                                    plan._placement_decision(z, full_cands(plan, z)),
                                    tag,
                                )

    # --- randomized differential checks ------------------------------------

    def _cases(self, seeds=300, max_n=8):
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(1, max_n)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            capacity = rng.choice([200, 600, 10_000])
            alignment = rng.choice([1, 64, 128])
            yield seed, buffers, perm, capacity, alignment

    def test_addresses_match_reference(self):
        for seed, buffers, perm, cap, align in self._cases():
            ref = ReferencePermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            self.assertEqual(fast.addresses, ref.addresses, f"seed={seed}")
            self.assertEqual(fast.quality(), ref.quality(), f"seed={seed}")

    def test_profiles_consistent(self):
        for seed, buffers, perm, cap, align in self._cases():
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            _check_consistency(self, fast, f"seed={seed}")
            _check_contact_faithful(self, fast, f"seed={seed}")


class RegressionFixtureTests(TestCase):
    """The exact contact-profile fixture from the spec."""

    def _fixture(self):
        # index: z=0, w=1, x=2, y=3; per-column order bottom->top: z, w, x, y.
        buffers = [
            _buf("z", 10, 0, 15),
            _buf("w", 10, 0, 10),
            _buf("x", 10, 0, 10),
            _buf("y", 10, 5, 15),
        ]
        return PermutationBasedLayoutSolver(buffers, [0, 1, 2, 3], 10_000, 1)

    def test_initial_profiles(self):
        plan = self._fixture()
        self.assertEqual(_below_named(plan, "z"), [(0, 15, None)])
        self.assertEqual(_below_named(plan, "w"), [(0, 10, "z")])
        self.assertEqual(_below_named(plan, "x"), [(0, 10, "w")])
        self.assertEqual(_below_named(plan, "y"), [(5, 10, "x"), (10, 15, "z")])
        _check_consistency(self, plan)

    def test_after_swapping_x_and_y(self):
        plan = self._fixture()
        plan.swap(2)  # swap x (pos 2) and y (pos 3); shared range I = [5, 10)
        self.assertEqual(_below_named(plan, "x"), [(0, 5, "w"), (5, 10, "y")])
        self.assertEqual(_below_named(plan, "y"), [(5, 10, "w"), (10, 15, "z")])
        self.assertEqual(_above_named(plan, "w"), [(0, 5, "x"), (5, 10, "y")])
        self.assertEqual(_above_named(plan, "z"), [(0, 10, "w"), (10, 15, "y")])
        self.assertEqual(_above_named(plan, "y"), [(5, 10, "x"), (10, 15, None)])
        self.assertEqual(_above_named(plan, "x"), [(0, 10, None)])
        _check_consistency(self, plan)


class SwapTests(TestCase):
    """Incremental swap in PermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_overlapping_swap_relayouts(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 64])
        delta = plan.swap(0)  # -> [b, a]
        self.assertEqual([_addr(plan, "b"), _addr(plan, "a")], [0, 50])
        self.assertEqual(delta, 0)  # both still fit

    def test_non_overlapping_swap_is_noop(self):
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3)]
        plan = self.plan(buffers, [0, 1])
        before = list(plan.addresses)
        delta = plan.swap(0)
        self.assertEqual(delta, 0)
        self.assertEqual(plan.addresses, before)
        self.assertEqual(plan.permutation, [1, 0])

    def test_swap_changes_total_size(self):
        # Only one of the two can fit fully below capacity; swapping which one
        # is placed first changes the total.
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(plan.quality(), 75)  # a@0 fits (2.5*30); b@30 (->120) does not
        delta = plan.swap(0)  # -> [b, a]: b@0 fits, a@90 (->120) does not
        self.assertEqual(plan.quality(), 225)  # 2.5 * 90
        self.assertEqual(delta, 150)

    def test_swap_back_restores(self):
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        d1 = plan.swap(0)
        d2 = plan.swap(0)
        self.assertEqual(d1 + d2, 0)
        self.assertEqual(plan.quality(), 75)  # 2.5 * 30
        # Back to [a, b]: a@0 fits; b@30 (-> 120) crosses cap 100 -> evicted.
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, None])

    def test_finalize_after_swaps_end_to_end(self):
        # Build, optimize via swaps, then commit: only buffers that fit below
        # capacity get an address written back.
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.swap(0)  # -> [b, a]: b@0 fits, a@90 (-> 120) does not
        plan.finalize()
        self.assertEqual(buffers[1].address, 0)  # b committed
        self.assertIsNone(buffers[0].address)  # a over capacity, dropped

    def test_random_swap_sequences_match_reference(self):
        for seed in range(3000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            for step in range(rng.randint(1, 2 * n)):
                i = rng.randrange(n - 1)
                before = fast.quality()
                delta = fast.swap(i)
                tag = f"seed={seed} step={step}"

                # Ground truth: a fresh reference build of the new permutation.
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.addresses, ref.addresses, tag)
                self.assertEqual(fast.quality(), ref.quality(), tag)
                self.assertEqual(delta, fast.quality() - before, tag)

                # The incrementally maintained contact profiles match a
                # from-scratch rebuild of the same permutation, exactly, and are
                # internally consistent.
                rebuilt = PermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
                self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
                self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
                _check_consistency(self, fast, tag)
                _check_contact_faithful(self, fast, tag)


class EvictionTests(TestCase):
    """`None`-as-eviction: the capacity gate in placement and its propagation."""

    def fast(self, buffers, permutation, capacity, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def ref(self, buffers, permutation, capacity, alignment=1):
        return ReferencePermutationBasedLayoutSolver(
            buffers, permutation, capacity, alignment
        )

    def test_lone_buffer_larger_than_capacity_evicted(self):
        # No candidates, but the buffer alone exceeds capacity -> evicted (the
        # one hole in the "on the floor => address 0" shortcut).
        for cls in (self.fast, self.ref):
            plan = cls([_buf("x", 150, 0, 1)], [0], 100)
            self.assertIsNone(_addr(plan, "x"))
            self.assertEqual(plan.quality(), 0)
            self.assertEqual(plan.count_allocated(), 0)
        # Exactly at the boundary fits.
        plan = self.fast([_buf("x", 100, 0, 1)], [0], 100)
        self.assertEqual(_addr(plan, "x"), 0)

    def test_aligned_address_crossing_capacity_evicted(self):
        # The capacity gate uses the *aligned* address. a@0 (64), b aligned to
        # 128; 128 + 64 = 192. cap 191 -> evicted; cap 192 -> fits exactly.
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        evicted = self.fast(buffers, [0, 1], 191, alignment=128)
        self.assertEqual(_addr(evicted, "a"), 0)
        self.assertIsNone(_addr(evicted, "b"))
        fits = self.fast(buffers, [0, 1], 192, alignment=128)
        self.assertEqual(_addr(fits, "b"), 128)

    def test_two_none_floor_vs_evicted_neighbour(self):
        # C's below-profile carries both kinds of None: a *floor* segment (label
        # None, no neighbour) and an *evicted-neighbour* segment (label E whose
        # address is None). The floor must not evict C; the evicted neighbour
        # must. E is too big to place, so over [1,2) C rests on the evicted E.
        E = _buf("E", 200, 1, 2)
        C = _buf("C", 10, 0, 3)
        plan = self.fast([E, C], [0, 1], 100)
        self.assertIsNone(_addr(plan, "E"))  # lone buffer > capacity
        # The profile literally shows floor (None) | E | floor (None).
        self.assertEqual(
            _below_named(plan, "C"), [(0, 1, None), (1, 2, "E"), (2, 3, None)]
        )
        self.assertIsNone(_addr(plan, "C"))  # rests on evicted E over [1, 2)
        # Contrast: the same C-shaped buffer that never overlaps E sits on the
        # floor at 0 -- a floor (None) neighbour does not evict.
        C2 = _buf("C2", 10, 0, 1)
        plan2 = self.fast([E, C2], [0, 1], 100)
        self.assertEqual(_addr(plan2, "C2"), 0)

    def test_swap_frees_space_refits_and_count_rises(self):
        # [a, b, c] with cap 100: a@0(60); b rests on a -> 120 evicted; c rests
        # on the evicted b -> evicted. Only a is allocated. Swapping a/b makes
        # b@0(60); a evicted; c (disjoint from a) rests on b -> c@60(90) fits.
        # So count_allocated rises 1 -> 2 and c goes None -> concrete.
        buffers = [_buf("a", 60, 0, 2), _buf("b", 60, 0, 4), _buf("c", 30, 2, 4)]
        plan = self.fast(buffers, [0, 1, 2], 100)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertIsNone(_addr(plan, "b"))
        self.assertIsNone(_addr(plan, "c"))
        self.assertEqual(plan.count_allocated(), 1)
        plan.swap(0)  # -> [b, a, c]
        self.assertEqual(_addr(plan, "b"), 0)
        self.assertIsNone(_addr(plan, "a"))
        self.assertEqual(_addr(plan, "c"), 60)  # re-fit: None -> concrete
        self.assertEqual(plan.count_allocated(), 2)
        # Matches the from-scratch oracle.
        ref = self.ref(buffers, [1, 0, 2], 100)
        self.assertEqual(plan.addresses, ref.addresses)

    def test_early_stop_saturated_interior_tail_all_none(self):
        # Eight buffers all alive over the single interval [0, 2), each size 40,
        # capacity 100: only the first two fit (0, 40); the third crosses 100 and
        # evicts, saturating the lone interval. The early-stop then bulk-evicts
        # the tail. Result is identical to the reference (no early-stop).
        n = 8
        buffers = [_buf(f"b{k}", 40, 0, 2) for k in range(n)]
        fast = self.fast(buffers, list(range(n)), 100)
        ref = self.ref(buffers, list(range(n)), 100)
        self.assertEqual(fast.addresses, ref.addresses)
        self.assertEqual(fast.addresses[:2], [0, 40])
        self.assertTrue(all(a is None for a in fast.addresses[2:]))
        self.assertEqual(fast.count_allocated(), 2)
        # Quality is exactly the placed prefix's contribution.
        self.assertEqual(fast.quality(), sum(fast._qualities[:2]))

    def test_sparse_end_interval_still_placed_after_saturated_interior(self):
        # The interior interval [1, 2) saturates (b0@0, b1@40, b2 evicted), but a
        # buffer H living only in the open head interval [0, 1) appears *last* in
        # the permutation. The per-interval early-stop keeps going (the head
        # interval is not yet done) and places H -- a coarse "whole top row None"
        # check would have stopped and wrongly evicted it.
        buffers = [
            _buf("b0", 40, 1, 2),
            _buf("b1", 40, 1, 2),
            _buf("b2", 40, 1, 2),
            _buf("H", 10, 0, 1),
        ]
        fast = self.fast(buffers, [0, 1, 2, 3], 100)
        ref = self.ref(buffers, [0, 1, 2, 3], 100)
        self.assertEqual(fast.addresses, ref.addresses)
        self.assertIsNone(_addr(fast, "b2"))  # interior saturated
        self.assertEqual(_addr(fast, "H"), 0)  # sparse head still placed
        self.assertEqual(fast.count_allocated(), 3)

    def test_n0_and_n1_edges(self):
        for cls in (
            PermutationBasedLayoutSolver,
            ReferencePermutationBasedLayoutSolver,
        ):
            empty = cls([], [], 100)
            self.assertEqual(empty.addresses, [])
            self.assertEqual(empty.quality(), 0)
            self.assertEqual(empty.count_allocated(), 0)
            empty.finalize()  # no-op, must not raise
            one_fits = cls([_buf("a", 40, 0, 1)], [0], 100)
            self.assertEqual(one_fits.addresses, [0])
            self.assertEqual(one_fits.count_allocated(), 1)
            one_evicted = cls([_buf("a", 200, 0, 1)], [0], 100)
            self.assertEqual(one_evicted.addresses, [None])
            self.assertEqual(one_evicted.count_allocated(), 0)


class RotateTests(TestCase):
    """rotate(i, j) and the single-element sweep it enables."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_rotate_noop(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = self.plan(buffers, [0, 1])
        before = list(plan.addresses)
        self.assertEqual(plan.rotate(1, 1), 0)
        self.assertEqual(plan.permutation, [0, 1])
        self.assertEqual(plan.addresses, before)

    def test_rotate_moves_element(self):
        # Three mutually overlapping buffers; move the first to the end.
        buffers = [_buf("a", 10, 0, 3), _buf("b", 20, 0, 3), _buf("c", 30, 0, 3)]
        plan = self.plan(buffers, [0, 1, 2])  # a@0, b@10, c@30
        plan.rotate(0, 2)  # -> [b, c, a]: b@0, c@20, a@50
        self.assertEqual(plan.permutation, [1, 2, 0])
        self.assertEqual([_addr(plan, n) for n in "abc"], [50, 0, 20])

    def test_random_rotations_match_reference(self):
        for seed in range(3000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            # Force the remove/reinsert fast path on every rotation (small n
            # gives small distances otherwise, so the chain would always win).
            fast._rotate_remove_insert_threshold = 1

            for step in range(rng.randint(1, 2 * n)):
                i, j = rng.randrange(n), rng.randrange(n)
                before = fast.quality()
                delta = fast.rotate(i, j)
                tag = f"seed={seed} step={step} i={i} j={j}"

                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.addresses, ref.addresses, tag)
                self.assertEqual(fast.quality(), ref.quality(), tag)
                self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
                self.assertEqual(delta, fast.quality() - before, tag)

                rebuilt = PermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
                self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
                self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
                _check_consistency(self, fast, tag)
                _check_contact_faithful(self, fast, tag)

    def test_single_element_sweep_matches_reference(self):
        # Sweep one element across every position (rotate it to 0, then bubble
        # it right), reading quality() at each stop. Each stop must match a
        # fresh build of that permutation, and a round trip must restore the
        # original state exactly -- the contract the annealing sweep relies on.
        for seed in range(500):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # exercise the fast path

            orig_perm = list(fast.permutation)
            orig_addr = list(fast.addresses)
            i = rng.randrange(n)
            x = orig_perm[i]
            others = [b for b in orig_perm if b != x]

            qualities = {}
            fast.rotate(i, 0)  # x to the front
            qualities[0] = fast.quality()
            for p in range(1, n):
                fast.swap(p - 1)  # bubble x from p-1 to p
                qualities[p] = fast.quality()

            # Every recorded objective matches a fresh build of "x inserted at p".
            for p in range(n):
                test_perm = others[:p] + [x] + others[p:]
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, test_perm, cap, align
                )
                self.assertEqual(qualities[p], ref.quality(), f"seed={seed} p={p}")

            # Round trip restores the exact original state (no hysteresis).
            fast.rotate(n - 1, i)
            self.assertEqual(fast.permutation, orig_perm, f"seed={seed}")
            self.assertEqual(fast.addresses, orig_addr, f"seed={seed}")
            rebuilt = PermutationBasedLayoutSolver(buffers, orig_perm, cap, align)
            self.assertEqual(fast.below_profile, rebuilt.below_profile, f"{seed}")
            self.assertEqual(fast.above_profile, rebuilt.above_profile, f"{seed}")
            self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, f"{seed}")
            _check_consistency(self, fast, f"seed={seed}")


class FastRotateTests(TestCase):
    """The remove-one / reinsert-elsewhere fast rotate in
    :class:`PermutationBasedLayoutSolver`, forced on for every rotation.

    The fast path is distance-gated in production; here we pin
    ``_rotate_remove_insert_threshold = 1`` so it fires on *every* rotate --
    including the large ``|i - j|`` moves it exists for -- and check its
    incremental profile patch against both oracles: a from-scratch reference
    build and a fresh ``PermutationBasedLayoutSolver`` on the same permutation
    (whose profiles come straight from ``_build_profiles``), which must agree
    exactly (see :meth:`_assert_matches`).
    """

    def _assert_matches(self, fast, cap, align, delta, before, tag):
        ref = ReferencePermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.addresses, ref.addresses, tag)
        self.assertEqual(fast.quality(), ref.quality(), tag)
        self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
        self.assertEqual(delta, fast.quality() - before, tag)
        rebuilt = PermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
        self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
        self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
        _check_consistency(self, fast, tag)
        _check_contact_faithful(self, fast, tag)

    def test_long_rotations_both_directions(self):
        # A handful of mutually overlapping buffers; sweep every (i, j) pair,
        # which includes the full-distance moves in both directions. Each
        # rotate is applied to a fresh plan.
        n = 7
        buffers = [_buf(f"b{k}", 10 * (k + 1), 0, 5) for k in range(n)]
        for i in range(n):
            for j in range(n):
                for cap, align in ((10_000, 1), (250, 64)):
                    fast = PermutationBasedLayoutSolver(
                        buffers, list(range(n)), cap, align
                    )
                    fast._rotate_remove_insert_threshold = 1
                    before = fast.quality()
                    delta = fast.rotate(i, j)
                    expected = list(range(n))
                    x = expected.pop(i)
                    expected.insert(j, x)
                    self.assertEqual(fast.permutation, expected)
                    self._assert_matches(
                        fast, cap, align, delta, before, f"i={i} j={j}"
                    )

    def test_dense_inplace_random(self):
        # Dense in-place wiring (inplace_prob up to ~0.7), random rotations
        # forced through the fast path.
        for seed in range(1500):
            rng = random.Random(seed)
            n = rng.randint(2, 12)
            buffers = _random_buffers(
                rng, n, horizon=15, max_size=300, inplace_prob=0.7
            )
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 800, 10**9])
            align = rng.choice([1, 32, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1
            for step in range(rng.randint(1, 2 * n)):
                i, j = rng.randrange(n), rng.randrange(n)
                before = fast.quality()
                delta = fast.rotate(i, j)
                tag = f"seed={seed} step={step} i={i} j={j}"
                self._assert_matches(fast, cap, align, delta, before, tag)

    def test_threshold_dispatch_agrees_with_chain(self):
        # The fast path and the swap-chain must produce identical results for
        # the same move. Run the same rotation on two clones, one forced to the
        # fast path and one forced to the chain, and compare.
        for seed in range(800):
            rng = random.Random(seed)
            n = rng.randint(3, 12)
            buffers = _random_buffers(rng, n, horizon=15, inplace_prob=0.5)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10**9])
            align = rng.choice([1, 64, 128])
            base = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            i, j = rng.randrange(n), rng.randrange(n)

            chain = base.copy()
            chain._rotate_remove_insert_threshold = n + 1  # never fast
            d_chain = chain.rotate(i, j)

            fast = base.copy()
            fast._rotate_remove_insert_threshold = 1  # always fast
            d_fast = fast.rotate(i, j)

            tag = f"seed={seed} i={i} j={j}"
            self.assertEqual(fast.permutation, chain.permutation, tag)
            self.assertEqual(fast.addresses, chain.addresses, tag)
            self.assertEqual(d_fast, d_chain, tag)
            self.assertEqual(fast.below_profile, chain.below_profile, tag)
            self.assertEqual(fast.above_profile, chain.above_profile, tag)
            self.assertEqual(fast.inplace_reuse, chain.inplace_reuse, tag)


class EligibilityConstructionMixin(MixinBase):
    """Constructing a plan with some buffers ineligible up front, for every
    packer. Run natively too because the native constructor's ``eligible=``
    branch (and its length check) is otherwise dead: every other native
    construction in the suite passes four positional arguments."""

    plan_class: type = None  # type: ignore[assignment]

    BUFFERS = [("a", 64), ("b", 50), ("c", 40)]

    def _buffers(self):
        return [_buf(name, size, 0, 3) for name, size in self.BUFFERS]

    def _plan(self, buffers, **kwargs):
        return self.plan_class(buffers, list(range(len(buffers))), 10_000, 1, **kwargs)

    def test_matches_the_reference_with_initial_eligibility(self):
        buffers = self._buffers()
        elig = [True, False, True]
        plan = self._plan(buffers, eligible=list(elig))
        ref = ReferencePermutationBasedLayoutSolver(
            buffers, [0, 1, 2], 10_000, 1, eligible=list(elig)
        )
        # b is transparent (HBM, no address) and c rests on a, not on b.
        self.assertEqual(list(plan.addresses), [0, None, 64])
        self.assertEqual(list(plan.addresses), list(ref.addresses))
        self.assertEqual(plan.quality(), ref.quality())
        self.assertEqual(plan.count_allocated(), ref.count_allocated())

    def test_all_eligible_matches_the_default(self):
        default = self._plan(self._buffers())
        explicit = self._plan(self._buffers(), eligible=[True, True, True])
        self.assertEqual(list(explicit.addresses), list(default.addresses))
        self.assertEqual(explicit.quality(), default.quality())

    def test_construction_matches_a_later_set_eligible(self):
        # The constructed state must be the state set_eligible reaches, not just
        # a plausible one: the two write the same flags through different paths.
        constructed = self._plan(self._buffers(), eligible=[True, False, True])
        toggled = self._plan(self._buffers())
        toggled.set_eligible(1, False)
        self.assertEqual(list(toggled.addresses), list(constructed.addresses))
        self.assertEqual(toggled.quality(), constructed.quality())

    def test_bad_eligible_length_rejected(self):
        buffers = self._buffers()
        for bad in ([], [True, False], [True] * 4):
            with self.assertRaises(ValueError):
                self._plan(buffers, eligible=bad)


class ReferenceSolverEligibilityConstructionTests(
    EligibilityConstructionMixin, TestCase
):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverEligibilityConstructionTests(
    EligibilityConstructionMixin, TestCase
):
    plan_class = PermutationBasedLayoutSolver


class NativeSolverEligibilityConstructionTests(EligibilityConstructionMixin, TestCase):
    plan_class = NativePermutationLayoutSolver


class EligibilityConstructionTests(TestCase):
    """What initial ineligibility does to the incremental packer's own state:
    the flags it stores and the contact profiles it keeps. The behaviour every
    packer shares is in :class:`EligibilityConstructionMixin`."""

    def plan(self, buffers, permutation, eligible, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(
            buffers, permutation, capacity, alignment, eligible=eligible
        )

    def test_default_all_eligible(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = PermutationBasedLayoutSolver(buffers, [0, 1], 10_000, 1)
        self.assertEqual(plan._eligible, [True, True])

    def test_ineligible_buffer_is_transparent(self):
        # b ineligible: routed to HBM (no address, no quality), and c stacks
        # directly on a as if b were absent -- not on b.
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3), _buf("c", 40, 0, 3)]
        plan = self.plan(buffers, [0, 1, 2], eligible=[True, False, True])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertIsNone(_addr(plan, "b"))
        self.assertEqual(_addr(plan, "c"), 64)  # rests on a, not on b
        self.assertEqual(plan.quality(), 2.5 * (64 + 40))
        self.assertEqual(plan.count_allocated(), 2)
        self.assertEqual(_below_named(plan, "c"), [(0, 3, "a")])
        # b's own profile is a trivial "nothing here" step function.
        self.assertEqual(_below_named(plan, "b"), [(0, 3, None)])
        self.assertEqual(_above_named(plan, "b"), [(0, 3, None)])
        _check_consistency(self, plan)
        _check_contact_faithful(self, plan)


class ResizeTests(TestCase):
    """resize(idx, new_size): change a footprint in place and re-place."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_resize_shifts_stacked_neighbour(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 64])
        delta = plan.resize(0, 100)  # a grows -> b moves up
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 100])
        # a's quality rises (2.5 * (100 - 64)); b's is unchanged.
        self.assertEqual(delta, 2.5 * (100 - 64))
        self.assertEqual(plan.quality(), 2.5 * (100 + 50))
        # The shared buffer object is NOT mutated.
        self.assertEqual(buffers[0].size, 64)

    def test_resize_over_capacity_evicts_and_uneviction(self):
        buffers = [_buf("a", 40, 0, 2), _buf("b", 40, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 40])
        plan.resize(0, 80)  # a@0..80, b@80..120 > 100 -> b evicted
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertIsNone(_addr(plan, "b"))
        self.assertEqual(plan.count_allocated(), 1)
        plan.resize(0, 40)  # shrink back -> b re-fits
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 40])
        self.assertEqual(plan.count_allocated(), 2)

    def test_resize_ineligible_is_bookkeeping_only(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3)]
        plan = PermutationBasedLayoutSolver(
            buffers, [0, 1], 10_000, 1, eligible=[True, False]
        )
        before = list(plan.addresses)
        delta = plan.resize(1, 5000)  # b is in HBM: nothing observable changes
        self.assertEqual(delta, 0.0)
        self.assertEqual(plan.addresses, before)
        self.assertEqual(plan._sizes[1], 5000)  # but the size is recorded

    def test_resize_crosses_inplace_fit_boundary(self):
        # child reuses parent while it fits; growing it past the parent forces a
        # stack, and shrinking it back restores the reuse.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "c"), 0)  # reuses p
        plan.resize(1, 200)  # c no longer fits in p
        self.assertEqual(_addr(plan, "c"), 128)  # stacks on p
        plan.resize(1, 64)
        self.assertEqual(_addr(plan, "c"), 0)  # reuse restored


class SetEligibleTests(TestCase):
    """set_eligible(idx, flag): toggle a buffer in/out of LX."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_toggle_out_then_in_restores(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3), _buf("c", 40, 0, 3)]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual([_addr(plan, n) for n in "abc"], [0, 64, 114])
        d_out = plan.set_eligible(1, False)  # b -> HBM
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertIsNone(_addr(plan, "b"))
        self.assertEqual(_addr(plan, "c"), 64)  # c drops onto a
        self.assertEqual(d_out, -2.5 * 50)
        d_in = plan.set_eligible(1, True)  # b -> LX at its slot
        self.assertEqual([_addr(plan, n) for n in "abc"], [0, 64, 114])
        self.assertEqual(d_in, 2.5 * 50)
        # Profiles match a fresh build after the round trip.
        rebuilt = _rebuilt_with_state(plan, 10_000, 1)
        self.assertEqual(plan.below_profile, rebuilt.below_profile)
        self.assertEqual(plan.above_profile, rebuilt.above_profile)
        _check_consistency(self, plan)
        _check_contact_faithful(self, plan)

    def test_toggle_unchanged_flag_is_noop(self):
        buffers = [_buf("a", 64, 0, 2)]
        plan = self.plan(buffers, [0])
        before = list(plan.addresses)
        self.assertEqual(plan.set_eligible(0, True), 0.0)  # already eligible
        self.assertEqual(plan.addresses, before)

    def test_toggle_out_uneviction(self):
        # a@0(60); b rests on a -> 120 > 100 evicted. Making a ineligible frees
        # the floor so b re-fits at 0.
        buffers = [_buf("a", 60, 0, 3), _buf("b", 60, 0, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertIsNone(_addr(plan, "b"))
        plan.set_eligible(0, False)
        self.assertIsNone(_addr(plan, "a"))
        self.assertEqual(_addr(plan, "b"), 0)  # un-evicted
        self.assertEqual(plan.count_allocated(), 1)


class ResizeEligibilityDifferentialTests(TestCase):
    """Randomized differential coverage for resize + set_eligible interleaved
    with swap / rotate, against both the from-scratch reference (O(n^2) scan) and
    a fresh rebuild carrying the same size/eligibility state."""

    def _assert_matches(self, fast, ref, cap, align, delta, before, tag):
        # Live reference (independent O(n^2) scan of the current state).
        self.assertEqual(fast.addresses, ref.addresses, tag)
        self.assertEqual(fast.quality(), ref.quality(), tag)
        self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
        self.assertEqual(delta, fast.quality() - before, tag)
        # Fresh rebuild reproducing the current sizes + eligibility: profiles and
        # in-place reuse (both size/eligibility dependent) must match exactly.
        rebuilt = _rebuilt_with_state(fast, cap, align)
        self.assertEqual(fast.addresses, rebuilt.addresses, tag)
        # Quality against the *rebuild*, not just ``ref``: ``_sync_reference``
        # copies ``_qualities`` into ``ref``, so ``ref.quality()`` only re-sums
        # whatever ``fast`` already computed and cannot catch a wrong size ->
        # quality formula in ``resize``. The rebuild derives it independently by
        # feeding the current ``_sizes`` through a fresh constructor.
        self.assertEqual(fast.quality(), rebuilt.quality(), tag)
        self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
        self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
        self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
        _check_consistency(self, fast, tag)
        _check_contact_faithful(self, fast, tag)

    def _apply(self, op, plan, rng, n):
        """Apply a random instance of ``op`` to ``plan``; return its delta."""
        if op == "swap":
            return plan.swap(rng.randrange(n - 1))
        if op == "rotate":
            return plan.rotate(rng.randrange(n), rng.randrange(n))
        if op == "resize":
            idx = rng.randrange(n)
            # A mix of shrink, grow-in-bounds, and grow-past-capacity (evicts).
            new = rng.choice([1, rng.randint(1, 300), rng.randint(300, 1200)])
            return plan.resize(idx, new)
        # toggle-eligibility
        idx = rng.randrange(n)
        return plan.set_eligible(idx, not plan._eligible[idx])

    def _sync_reference(self, ref, fast):
        """Rebuild ``ref`` to reproduce ``fast``'s post-move state, giving an
        independent O(n^2) oracle regardless of how the move was chosen."""
        ref.permutation = list(fast.permutation)
        ref._sizes = list(fast._sizes)
        ref._qualities = list(fast._qualities)
        ref._eligible = list(fast._eligible)
        ref._build()

    def test_random_mixed_sequences_match_oracles(self):
        seeds = 4000 if _STRESS else 800
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(1, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # exercise the fast path
            ref = ReferencePermutationBasedLayoutSolver(buffers, perm, cap, align)

            ops = ["swap", "rotate", "resize", "toggle"]
            for step in range(rng.randint(1, 3 * n + 3)):
                op = rng.choice(ops)
                if op == "swap" and n < 2:
                    op = "resize"  # no adjacent pair to swap
                before = fast.quality()
                d_fast = self._apply(op, fast, rng, n)
                self._sync_reference(ref, fast)
                tag = f"seed={seed} step={step} op={op} elig={fast._eligible}"
                self._assert_matches(fast, ref, cap, align, d_fast, before, tag)


class NativeSolverDifferentialTests(TestCase):
    """Differential coverage for the C++ ``NativePermutationLayoutSolver``
    accelerator against the canonical Python ``PermutationBasedLayoutSolver``.

    The C++ class is an opt-in accelerator; the Python packer stays canonical
    and is the correctness oracle. This drives the SAME random interleaved
    swap / rotate / resize / set_eligible sequences (the ``_apply`` mix from
    :class:`ResizeEligibilityDifferentialTests`) through both and asserts
    observable equality -- bit-for-bit identical ``addresses`` (None ==
    evicted / HBM), ``quality()`` and ``count_allocated()`` -- after every op,
    plus that the per-op quality delta each returns agrees.
    """

    def _apply_both(self, op, py_plan, cpp_plan, rng, n):
        """Apply one random instance of ``op`` identically to both plans.

        Returns ``(delta_py, delta_cpp)``. Every random parameter is drawn once
        and reused, so the two plans receive byte-identical operations.
        """
        if op == "swap":
            i = rng.randrange(n - 1)
            return py_plan.swap(i), cpp_plan.swap(i)
        if op == "rotate":
            i, j = rng.randrange(n), rng.randrange(n)
            return py_plan.rotate(i, j), cpp_plan.rotate(i, j)
        if op == "resize":
            idx = rng.randrange(n)
            new = rng.choice([1, rng.randint(1, 300), rng.randint(300, 1200)])
            return py_plan.resize(idx, new), cpp_plan.resize(idx, new)
        # toggle-eligibility: the Python plan is the source of truth for the
        # current flag; flip the same (idx, flag) on both.
        idx = rng.randrange(n)
        flag = not py_plan._eligible[idx]
        return py_plan.set_eligible(idx, flag), cpp_plan.set_eligible(idx, flag)

    def _assert_equal(self, py_plan, cpp_plan, tag):
        self.assertEqual(list(cpp_plan.addresses), list(py_plan.addresses), tag)
        self.assertEqual(cpp_plan.quality(), py_plan.quality(), tag)
        self.assertEqual(cpp_plan.count_allocated(), py_plan.count_allocated(), tag)

    def test_fast_path_reads_pokethrough_top_not_dead_parent(self):
        """The aggregate fast path's boundary case, pinned deterministically.

        ``RecomputeAll`` takes the interval-aggregate fast path for a buffer with
        no in-place partner and the gather + ``PlaceDecision`` path for one with,
        so the discriminating state is a partner-free buffer resting on a
        *poke-through*: ``c`` co-locates into ``p``'s slot, so its top (64) sits
        below ``p``'s (128), and ``h`` starts at 5 -- after ``p`` dies -- so its
        only candidate is ``c``. Its floor must therefore be 64, not 128.

        ``h`` overlapping ``p`` as well (as in the contact-profile poke-through
        test) makes both answers 128, which is why that case cannot distinguish a
        correct aggregate from one that folds in a dead-but-taller buffer. The
        alignment must stay 1: at 128-byte alignment ``align_up(64) ==
        align_up(128)`` and the discrimination vanishes.
        """
        p = _buf("p", 128, 0, 5)
        c = _buf("c", 64, 4, 10, in_place_parents=["p"])
        h = _buf("h", 32, 5, 10)
        buffers = [p, c, h]
        perm = [0, 1, 2]

        cpp_plan = NativePermutationLayoutSolver(buffers, perm, 10_000, 1)
        self.assertEqual(list(cpp_plan.addresses), [0, 0, 64])

        # Both Python packers agree, so the expectation is the specification's,
        # not just this implementation's. The from-scratch reference is the
        # independent oracle the randomized native tests never consult.
        py_plan = PermutationBasedLayoutSolver(buffers, perm, 10_000, 1)
        ref_plan = ReferencePermutationBasedLayoutSolver(buffers, perm, 10_000, 1)
        self._assert_equal(py_plan, cpp_plan, "pokethrough fast path")
        self._assert_equal(ref_plan, cpp_plan, "pokethrough fast path vs reference")

    def test_native_solver_honors_lifetime_end_override(self):
        persistent = LifetimeBoundBuffer(
            "persistent",
            64,
            [0, 1],
            lifetime_end_override=5,
        )
        later = LifetimeBoundBuffer("later", 64, [2, 3])
        buffers = [persistent, later]
        permutation = [0, 1]

        py_plan = PermutationBasedLayoutSolver(buffers, permutation, 10_000, 1)
        cpp_plan = NativePermutationLayoutSolver(buffers, permutation, 10_000, 1)

        self._assert_equal(py_plan, cpp_plan, "lifetime end override")
        self.assertNotEqual(cpp_plan.addresses[0], cpp_plan.addresses[1])

    def test_random_mixed_sequences_match_python(self):
        seeds = 4000 if _STRESS else 800
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(1, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            py_plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            py_plan._rotate_remove_insert_threshold = 1  # exercise the fast path
            cpp_plan = NativePermutationLayoutSolver(buffers, perm, cap, align)

            self._assert_equal(py_plan, cpp_plan, f"seed={seed} init")

            ops = ["swap", "rotate", "resize", "toggle"]
            for step in range(rng.randint(1, 3 * n + 3)):
                op = rng.choice(ops)
                if op == "swap" and n < 2:
                    op = "resize"  # no adjacent pair to swap
                d_py, d_cpp = self._apply_both(op, py_plan, cpp_plan, rng, n)
                tag = f"seed={seed} step={step} op={op}"
                self.assertEqual(d_cpp, d_py, tag)
                self._assert_equal(py_plan, cpp_plan, tag)

    def test_copy_is_independent_and_matches_python(self):
        # copy() yields an independent snapshot: mutating the clone leaves the
        # source observably unchanged, and the mutated clone still matches a
        # Python plan driven through the same swaps.
        for seed in range(400):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            cpp_plan = NativePermutationLayoutSolver(buffers, perm, cap, align)
            src_addr = list(cpp_plan.addresses)
            src_q = cpp_plan.quality()

            clone = cpp_plan.copy()
            py_plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            swaps = [rng.randrange(n - 1) for _ in range(rng.randint(1, 2 * n))]
            for i in swaps:
                clone.swap(i)
                py_plan.swap(i)

            # Source untouched by clone mutations.
            self.assertEqual(list(cpp_plan.addresses), src_addr, f"seed={seed}")
            self.assertEqual(cpp_plan.quality(), src_q, f"seed={seed}")
            # Mutated clone matches the Python plan run through the same swaps.
            self.assertEqual(
                list(clone.addresses), list(py_plan.addresses), f"seed={seed}"
            )
            self.assertEqual(clone.quality(), py_plan.quality(), f"seed={seed}")
            self.assertEqual(
                clone.count_allocated(), py_plan.count_allocated(), f"seed={seed}"
            )


class CopyTests(TestCase):
    """copy() makes an independent layout snapshot sharing static structures."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_static_shared_dynamic_independent(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3), _buf("c", 40, 1, 3)]
        plan = self.plan(buffers, [0, 1, 2])
        clone = plan.copy()
        # Static structures are shared by reference.
        self.assertIs(clone.buffers, plan.buffers)
        self.assertIs(clone.overlap_dict, plan.overlap_dict)
        self.assertIs(clone._name_to_idx, plan._name_to_idx)
        # Dynamic state is equal but independent.
        self.assertEqual(clone.addresses, plan.addresses)
        self.assertEqual(clone.below_profile, plan.below_profile)
        self.assertIsNot(clone.permutation, plan.permutation)
        self.assertIsNot(clone.below_profile, plan.below_profile)
        self.assertIsNot(clone.below_profile[0], plan.below_profile[0])

    def test_mutating_copy_leaves_original_intact(self):
        for seed in range(2000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            orig_perm = list(plan.permutation)
            orig_addr = list(plan.addresses)
            orig_below = {
                k: Profile(list(p.starts), list(p.labels))
                for k, p in plan.below_profile.items()
            }
            orig_quality = plan.quality()

            clone = plan.copy()
            for _ in range(rng.randint(1, 2 * n)):
                clone.swap(rng.randrange(n - 1))

            # Original is untouched by mutations on the clone.
            self.assertEqual(plan.permutation, orig_perm, seed)
            self.assertEqual(plan.addresses, orig_addr, seed)
            self.assertEqual(plan.below_profile, orig_below, seed)
            self.assertEqual(plan.quality(), orig_quality, seed)

            # The mutated clone is a valid plan: matches a fresh build.
            rebuilt = PermutationBasedLayoutSolver(
                buffers, list(clone.permutation), cap, align
            )
            self.assertEqual(clone.addresses, rebuilt.addresses, seed)
            self.assertEqual(clone.quality(), rebuilt.quality(), seed)
            self.assertEqual(clone.below_profile, rebuilt.below_profile, seed)
            self.assertEqual(clone.above_profile, rebuilt.above_profile, seed)


@unittest.skipUnless(
    _STRESS, "set TORCH_SPYRE_STRESS_SCRATCHPAD=1 to run scratchpad stress tests"
)
class StressTests(TestCase):
    """Exhaustive randomized differential coverage. Not run by default; these
    are the heavy versions of the SwapTests / RotateTests / CopyTests checks --
    thousands of seeds, larger n, dense in-place wiring -- against from-scratch
    reference and rebuild oracles."""

    def _stress_buffers(self, rng, n):
        return _random_buffers(rng, n, horizon=15, max_size=300, inplace_prob=0.4)

    def _cases(self, seeds, max_n=13):
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(2, max_n)
            buffers = self._stress_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 800, 10**9])
            align = rng.choice([1, 32, 64, 128])
            yield seed, rng, n, buffers, perm, cap, align

    def _assert_matches_rebuild(self, fast, cap, align, tag):
        ref = ReferencePermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.addresses, ref.addresses, tag)
        self.assertEqual(fast.quality(), ref.quality(), tag)
        self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
        rebuilt = PermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
        self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
        self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
        _check_contact_faithful(self, fast, tag)

    def test_swap_sequences(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(20000):
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            for step in range(rng.randint(1, 3 * n)):
                i = rng.randrange(n - 1)
                before = fast.quality()
                delta = fast.swap(i)
                tag = f"seed={seed} step={step}"
                self.assertEqual(delta, fast.quality() - before, tag)
                self._assert_matches_rebuild(fast, cap, align, tag)

    def test_rotation_sequences(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(10000):
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # force the fast path
            for step in range(rng.randint(1, 3 * n)):
                i, j = rng.randrange(n), rng.randrange(n)
                before = fast.quality()
                delta = fast.rotate(i, j)
                tag = f"seed={seed} step={step} i={i} j={j}"
                self.assertEqual(delta, fast.quality() - before, tag)
                self._assert_matches_rebuild(fast, cap, align, tag)

    def test_single_element_sweeps(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(3000):
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # force the fast path
            orig_perm = list(fast.permutation)
            orig_addr = list(fast.addresses)
            i = rng.randrange(n)
            x = orig_perm[i]
            others = [b for b in orig_perm if b != x]

            qualities = {}
            fast.rotate(i, 0)
            qualities[0] = fast.quality()
            for p in range(1, n):
                fast.swap(p - 1)
                qualities[p] = fast.quality()
            for p in range(n):
                test_perm = others[:p] + [x] + others[p:]
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, test_perm, cap, align
                )
                self.assertEqual(qualities[p], ref.quality(), f"seed={seed} p={p}")

            fast.rotate(n - 1, i)
            self.assertEqual(fast.permutation, orig_perm, seed)
            self.assertEqual(fast.addresses, orig_addr, seed)

    def test_copy_isolation(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(10000):
            plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            orig_addr = list(plan.addresses)
            orig_below = {
                k: Profile(list(p.starts), list(p.labels))
                for k, p in plan.below_profile.items()
            }
            clone = plan.copy()
            for _ in range(rng.randint(1, 3 * n)):
                clone.swap(rng.randrange(n - 1))
            self.assertEqual(plan.addresses, orig_addr, seed)
            self.assertEqual(plan.below_profile, orig_below, seed)
            self._assert_matches_rebuild(clone, cap, align, f"seed={seed} (clone)")


class ProfileTests(TestCase):
    """Unit tests for the Profile step-function, in isolation."""

    def test_uniform_and_label_at(self):
        p = Profile.uniform(0, 10, 7)
        self.assertEqual(p.span_start, 0)
        self.assertEqual(p.span_end, 10)
        self.assertEqual(p.label_at(0), 7)
        self.assertEqual(p.label_at(9), 7)
        p.validate()

    def test_from_segments_coalesces(self):
        p = Profile.from_segments([0, 3, 5, 9], [1, 1, 2])
        self.assertEqual(p, Profile([0, 5, 9], [1, 2]))
        p.validate()

    def test_segments_clips_and_copies(self):
        p = Profile([0, 5, 10, 15], [1, 2, 3])
        starts, labels = p.segments(3, 12)
        self.assertEqual(starts, [3, 5, 10, 12])
        self.assertEqual(labels, [1, 2, 3])
        # returned data must not alias internal state
        starts[0] = -999
        self.assertEqual(p.starts[0], 0)
        # whole-span and empty range
        self.assertEqual(p.segments(0, 15), ([0, 5, 10, 15], [1, 2, 3]))
        self.assertEqual(p.segments(7, 7), ([7], []))

    def test_splice_at_exact_breakpoints(self):
        p = Profile([0, 5, 10, 15], [1, 2, 3])
        p.splice(5, 10, [5, 10], [9])
        self.assertEqual(p, Profile([0, 5, 10, 15], [1, 9, 3]))
        p.validate()

    def test_splice_inside_one_segment(self):
        p = Profile([0, 10], [1])
        p.splice(3, 7, [3, 7], [2])
        self.assertEqual(p, Profile([0, 3, 7, 10], [1, 2, 1]))
        p.validate()

    def test_splice_spanning_several_segments(self):
        p = Profile([0, 5, 10, 15, 20], [1, 2, 3, 4])
        p.splice(3, 17, [3, 17], [9])
        self.assertEqual(p, Profile([0, 3, 17, 20], [1, 9, 4]))
        p.validate()

    def test_splice_coalesces_both_seams(self):
        p = Profile([0, 5, 10, 15], [1, 2, 1])
        # replace the middle [5,10) with label 1 -> whole thing coalesces to one
        p.splice(5, 10, [5, 10], [1])
        self.assertEqual(p, Profile([0, 15], [1]))
        p.validate()

    def test_splice_multi_segment_replacement(self):
        p = Profile([0, 10], [1])
        p.splice(2, 8, [2, 4, 6, 8], [2, 3, 2])
        self.assertEqual(p, Profile([0, 2, 4, 6, 8, 10], [1, 2, 3, 2, 1]))
        p.validate()

    def test_relabel_splits_straddling_segment(self):
        p = Profile([0, 10], [1])
        p.relabel(3, 7, {1: 5})
        self.assertEqual(p, Profile([0, 3, 7, 10], [1, 5, 1]))
        p.validate()

    def test_relabel_only_matching_labels(self):
        p = Profile([0, 5, 10, 15], [1, 2, 3])
        p.relabel(0, 15, {1: 9, 3: 9})  # 1->9, 3->9, 2 untouched
        self.assertEqual(p, Profile([0, 5, 10, 15], [9, 2, 9]))
        p.validate()

    def test_empty_range_noops(self):
        p = Profile([0, 5, 10], [1, 2])
        before = Profile(list(p.starts), list(p.labels))
        p.splice(5, 5, [5], [])
        self.assertEqual(p, before)
        p.relabel(7, 7, {2: 9})
        self.assertEqual(p, before)

    def test_label_set(self):
        p = Profile([0, 5, 10], [1, None])
        self.assertEqual(p.label_set(), {1, None})
        self.assertEqual(p.label_set() - {None}, {1})

    def test_validate_catches_corruption(self):
        bad_order = Profile([0, 5, 5, 10], [1, 2, 3])  # not strictly increasing
        with self.assertRaises(AssertionError):
            bad_order.validate()
        bad_adjacent = Profile([0, 5, 10], [1, 1])  # adjacent equal labels
        with self.assertRaises(AssertionError):
            bad_adjacent.validate()
        bad_len = Profile([0, 5, 10], [1])  # length mismatch
        with self.assertRaises(AssertionError):
            bad_len.validate()

    def test_none_labels_round_trip(self):
        p = Profile.uniform(0, 10, None)
        self.assertIsNone(p.label_at(4))
        p.splice(3, 7, [3, 7], [2])
        self.assertEqual(p, Profile([0, 3, 7, 10], [None, 2, None]))
        p.validate()


class IndexGuardTestsMixin(MixinBase):
    """Every packer rejects a buffer index outside ``range(len(buffers))``.

    The randomized differential tests draw only valid indices, so this is the
    one class of input where the packers could disagree unobserved -- and they
    did: Python read ``idx=-1`` as the last buffer and quietly operated on it
    where the native packer raised."""

    plan_class: type = None  # type: ignore[assignment]

    BAD_INDICES = (-1, 3, 999)

    def _plan(self, n=3):
        bufs = [_buf(f"b{i}", 64, 0, 3) for i in range(n)]
        return self.plan_class(bufs, list(range(n)), 10_000, 128)

    def test_resize_index_out_of_range_raises(self):
        plan = self._plan()
        for bad in self.BAD_INDICES:
            with self.assertRaises(ValueError):
                plan.resize(bad, 100)

    def test_set_eligible_index_out_of_range_raises(self):
        # ``flag=True`` is the flag the aliased buffer already carries, so an
        # unchecked ``-1`` would take the no-op path and return 0.0.
        plan = self._plan()
        for bad in self.BAD_INDICES:
            with self.assertRaises(ValueError):
                plan.set_eligible(bad, True)

    def test_top_or_inf_index_out_of_range_raises(self):
        plan = self._plan()
        for bad in self.BAD_INDICES:
            with self.assertRaises(ValueError):
                plan.top_or_inf(bad)

    def test_swap_index_out_of_range_raises(self):
        # A swap position needs a successor to swap with, so the last position
        # (2 of 0..2 here) is out of range as well.
        plan = self._plan()
        for bad in (-1, 2, 3, 999):
            with self.assertRaises(ValueError):
                plan.swap(bad)

    def test_rotate_index_out_of_range_raises(self):
        plan = self._plan()
        # (3, 0) and (0, 3) sit at exactly ``n``, the boundary a ``<=`` typo
        # would let through; (999, 999) is the pair the ``i == j`` no-op would
        # swallow.
        for i, j in ((3, 0), (0, 3), (5, 0), (0, 5), (-1, 0), (0, -1), (999, 999)):
            with self.assertRaises(ValueError):
                plan.rotate(i, j)

    def test_is_fully_allocated_index_out_of_range_raises(self):
        plan = self._plan()
        for bad in self.BAD_INDICES:
            with self.assertRaises(ValueError):
                plan.is_fully_allocated(bad)

    def test_overlaps_index_out_of_range_raises(self):
        # Both positions: either one alone reads as the last buffer when
        # unchecked, and ``overlaps`` is symmetric enough to hide that.
        plan = self._plan()
        for i, j in ((-1, 0), (0, -1), (3, 0), (0, 3), (999, 999)):
            with self.assertRaises(ValueError):
                plan.overlaps(i, j)


class ReferenceSolverIndexGuardTests(IndexGuardTestsMixin, TestCase):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverIndexGuardTests(IndexGuardTestsMixin, TestCase):
    plan_class = PermutationBasedLayoutSolver


class NativeSolverIndexGuardTests(IndexGuardTestsMixin, TestCase):
    plan_class = NativePermutationLayoutSolver


class ConstructorGuardTestsMixin(MixinBase):
    """Every packer rejects the degenerate constructor arguments. Regression
    coverage for the memory-safety review's ASan-confirmed findings on the
    native side; on the Python side these went unchecked, so a negative
    alignment produced aliased addresses rather than an error."""

    plan_class: type = None  # type: ignore[assignment]

    def test_empty_uses_raises(self):
        bad = LifetimeBoundBuffer(name="x", size=64, uses=[], in_place_parents=[])
        with self.assertRaises(ValueError):
            self.plan_class([bad], [0], 10_000, 128)

    def test_non_positive_alignment_raises(self):
        # -128 is the interesting one: _align_up would round *down*, seating two
        # co-live buffers at one address instead of stacking them.
        for bad in (0, -128):
            with self.assertRaises(ValueError):
                self.plan_class([_buf("a", 64, 0, 3)], [0], 10_000, bad)

    def test_negative_size_raises(self):
        with self.assertRaises(ValueError):
            self.plan_class([_buf("a", -64, 0, 3)], [0], 10_000, 128)

    def test_last_use_at_int64_max_raises(self):
        # end_time is uses[-1] + 1, which overflows the native packer's int64.
        huge = LifetimeBoundBuffer(
            name="x", size=64, uses=[0, 2**63 - 1], in_place_parents=[]
        )
        with self.assertRaises(ValueError):
            self.plan_class([huge], [0], 10_000, 128)


class ReferenceSolverConstructorGuardTests(ConstructorGuardTestsMixin, TestCase):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverConstructorGuardTests(
    ConstructorGuardTestsMixin, TestCase
):
    plan_class = PermutationBasedLayoutSolver


class NativeSolverConstructorGuardTests(ConstructorGuardTestsMixin, TestCase):
    plan_class = NativePermutationLayoutSolver


class InPlaceRejectionTestsMixin(MixinBase):
    """Every packer rejects the declared in-place pairs the plan invariants
    forbid, so that ``TORCH_SPYRE_NATIVE_PACKER=0`` selects the same packer and
    not a stricter one."""

    plan_class: type = None  # type: ignore[assignment]

    def _plan(self, buffers):
        return self.plan_class(buffers, [0, 1], 10_000, 128)

    def test_write_only_parent_rejected(self):
        # p is written at tick 0 and never read, so it has no live storage to
        # hand to c.
        buffers = [_buf("p", 64, 0, 1), _buf("c", 32, 0, 3, ["p"])]
        with self.assertRaises(ValueError):
            self._plan(buffers)

    def test_multi_tick_overlap_rejected(self):
        # p is live through tick 3 and c from tick 1: three ticks of overlap
        # rather than the single handoff tick, so co-locating them would alias
        # two buffers that are live together.
        buffers = [_buf("p", 64, 0, 4), _buf("c", 32, 1, 4, ["p"])]
        with self.assertRaises(ValueError):
            self._plan(buffers)


class ReferenceSolverInPlaceRejectionTests(InPlaceRejectionTestsMixin, TestCase):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverInPlaceRejectionTests(
    InPlaceRejectionTestsMixin, TestCase
):
    plan_class = PermutationBasedLayoutSolver


class NativeSolverInPlaceRejectionTests(InPlaceRejectionTestsMixin, TestCase):
    plan_class = NativePermutationLayoutSolver


class NativePermutationViewTests(TestCase):
    """Lifetime and reference semantics of the native ``permutation`` view.

    ``plan.permutation`` returns a ``PermutationView`` that *borrows* its solver
    and reads straight through to the solver's ``std::vector``. Two opposite
    things must hold, and the rest of the suite establishes neither:

    - It must stay **valid**. Nothing else in these tests lets a view outlive its
      solver, so a view left reading freed memory would go unnoticed -- and did:
      the ``py::keep_alive`` securing this was initially attached to the property
      rather than to the getter, where it silently does nothing, and the whole
      suite passed regardless.
    - It must stay **live**. ``annealing_step_swap`` captures the view once and
      re-reads it after each ``swap``, so a snapshot would make the search read
      pre-swap values. That degrades rather than breaks: the search would simply
      explore a different trajectory and still return a feasible layout, so only
      the cross-packer equivalence test could notice, and only because it demands
      bit-identical trajectories.

    These are failures where the program keeps running and produces plausible
    output, which is why they are asserted against the mechanism directly.
    """

    def _bufs(self):
        return [_buf("a", 64, 0, 4), _buf("c", 96, 0, 4), _buf("d", 32, 0, 4)]

    def _plan(self, perm=(0, 1, 2), capacity=200, alignment=1):
        return NativePermutationLayoutSolver(
            self._bufs(), list(perm), capacity, alignment
        )

    def test_view_outlives_its_solver(self):
        """keep_alive must make the view retain the solver it borrows."""
        plan = self._plan()
        view = plan.permutation
        del plan
        for _ in range(3):
            gc.collect()
        self.assertEqual(list(view), [0, 1, 2])

    def test_view_survives_reuse_of_the_solver_s_memory(self):
        """Stronger than the above: reading freed memory often *appears* to work
        because the block has not been handed out again. Force it to be reused."""
        plan = self._plan(perm=(2, 1, 0))
        view = plan.permutation
        del plan
        gc.collect()
        churn = [self._plan(perm=(1, 0, 2)) for _ in range(200)]
        self.assertEqual(list(view), [2, 1, 0])
        self.assertEqual(len(churn), 200)  # keep the churn alive to this point

    def test_captured_view_observes_later_mutation(self):
        """The liveness half: ``annealing_step_swap`` depends on this exactly."""
        plan = self._plan()
        view = plan.permutation  # captured ONCE, before the mutation
        plan.swap(0)
        self.assertEqual(list(view), [1, 0, 2])
        plan.rotate(0, 2)
        self.assertEqual(list(view), [0, 2, 1])

    def test_copy_detaches_from_the_live_order(self):
        """The mirror image of liveness. The search stores its best-so-far
        permutation with a copy and later compares the live order against it; were
        the copy to track the original, that test would be vacuously equal and
        "commit the best permutation seen" would quietly stop working."""
        plan = self._plan()
        view = plan.permutation
        snapshot = copy.copy(view)
        deep = copy.deepcopy(view)
        plan.swap(0)
        self.assertIsInstance(snapshot, list)
        self.assertIsInstance(deep, list)
        self.assertEqual(snapshot, [0, 1, 2])
        self.assertEqual(deep, [0, 1, 2])
        self.assertEqual(list(view), [1, 0, 2])
        self.assertNotEqual(snapshot, list(view))

    def test_clone_view_tracks_the_clone_not_the_original(self):
        """``copy()`` deep-copies the order, and the search clones plans."""
        original = self._plan()
        clone = original.copy()
        clone.swap(0)
        self.assertEqual(list(original.permutation), [0, 1, 2])
        self.assertEqual(list(clone.permutation), [1, 0, 2])

    def test_view_is_read_only(self):
        """Read-only by construction, so Python cannot reorder the permutation
        behind the solver's back (which would desync it from the addresses)."""
        view = self._plan().permutation
        with self.assertRaises(TypeError):
            view[0] = 5
        with self.assertRaises(TypeError):
            del view[0]

    def test_view_sequence_protocol(self):
        """len / indexing / negative offsets / equality against a plain list, and
        IndexError past the end so ``list(view)`` and iteration terminate."""
        plan = self._plan(perm=(2, 0, 1))
        view = plan.permutation
        self.assertEqual(len(view), 3)
        self.assertEqual(view[0], 2)
        self.assertEqual(view[-1], 1)
        self.assertEqual(list(view), [2, 0, 1])
        self.assertEqual([x for x in view], [2, 0, 1])
        self.assertEqual(view, [2, 0, 1])
        self.assertNotEqual(view, [0, 1, 2])
        self.assertEqual(view, plan.permutation)
        for bad in (3, 99, -4):
            with self.assertRaises(IndexError):
                view[bad]

    def test_finalize_writes_addresses_back_including_none(self):
        """``finalize`` writes to EVERY buffer, so an evicted one is cleared to
        ``None`` rather than left holding a stale address."""
        buffers = [_buf("x", 150, 0, 3), _buf("y", 150, 0, 3)]
        plan = NativePermutationLayoutSolver(buffers, [0, 1], 200, 1)
        plan.finalize()
        self.assertEqual(
            [(b.name, b.address) for b in buffers], [("x", 0), ("y", None)]
        )
        # The retained list is the caller's, not a copy.
        self.assertEqual([b.name for b in plan.buffers], ["x", "y"])
