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

"""End-to-end tests for the simulated-annealing layout solver."""

import copy
import math
import os
import random as rnd
import unittest
from unittest import TestCase

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    buffer_quality,
)
from torch_spyre._inductor.scratchpad.cooling_schedules import (
    SelfCalibratingReheatingSchedule,
    ExponentialCoolingSchedule,
    default_initial_temperature,
    peak_memory_load,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingLayoutSolver,
    SimulatedAnnealingSolverWithBuffers,
)

# Heavy randomized anneals over many seeds, larger problems and longer
# schedules. Skipped by default (slow); opt in with the env var.
_STRESS = os.environ.get("TORCH_SPYRE_STRESS_SCRATCHPAD") == "1"


def _random_buffers(rng, n, horizon=12, max_size=200):
    """Half-open lifetimes, some in-place children (parent.end == child.start+1)."""
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        uses = [start] if end == start + 1 else [start, end - 1]
        buffers.append(LifetimeBoundBuffer(f"b{i}", size, uses))
    for child_i in range(1, n):
        if rng.random() < 0.25:
            parent = buffers[rng.randrange(child_i)]
            child = buffers[child_i]
            new_start = parent.uses[-1]
            new_last = max(child.uses[-1], parent.uses[-1])
            child.uses = [new_start] if new_start == new_last else [new_start, new_last]
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _short_schedule():
    return ExponentialCoolingSchedule(
        t_initial=100.0, t_final=1.0, steps_per_epoch=5, epochs=4
    )


def _peak600_buffers():
    """Two buffers live at the same time: peak load 600, so a schedule that has
    not yet seen a move scale seeds its temperature at 600 / 300 = 2.0."""
    return [
        LifetimeBoundBuffer("a", 300, [0, 1]),
        LifetimeBoundBuffer("b", 300, [0, 1]),
    ]


def _assert_feasible(buffers, capacity):
    """Committed buffers fit below capacity and never address-overlap a
    time-overlapping peer (an in-place pair may share its base address)."""
    committed = [b for b in buffers if b.address is not None]
    for b in committed:
        assert b.address + b.size <= capacity, f"{b.name} exceeds capacity"
    for a in range(len(committed)):
        for c in range(a + 1, len(committed)):
            u, v = committed[a], committed[c]
            if not u.overlaps_in_time(v):
                continue
            if u.name in v.in_place_parents or v.name in u.in_place_parents:
                continue  # in-place pair may share an address
            assert u.address + u.size <= v.address or v.address + v.size <= u.address, (
                f"{u.name}@{u.address}+{u.size} overlaps {v.name}@{v.address}+{v.size}"
            )


def _committed_total(buffers):
    # The annealer optimizes the use-weighted quality, so the committed total it
    # is compared against must use the same weighting (not raw size).
    return sum(buffer_quality(b) for b in buffers if b.address is not None)


class CoolingScheduleTests(TestCase):
    def test_exponential_schedule_sequence(self):
        # alpha = (1/8) ** (1/3) = 0.5; cools once per epoch (at i=2, 4).
        s = ExponentialCoolingSchedule(
            t_initial=8.0, t_final=1.0, steps_per_epoch=2, epochs=4
        )
        traj = [s.reset()]
        t = traj[0]
        while t is not None:
            t = s.update(True, 0.0)  # ignores acceptance and move scale
            traj.append(t)
        self.assertEqual(traj, [8.0, 8.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, None])

    def test_reheating_schedule_trajectory(self):
        # accept_hi=e^-1, accept_lo=e^-4 -> A=1, B=4, so delta=2 and sqrt(A*B)=2.
        # Seed temperature is 2.0 (peak load 600 / 300). The first move_scale=8
        # snaps center to 8/2=4, giving band top 8 and cooling by alpha=0.5 per
        # step. Four cycles of length 2 over 8 steps: cool 8->4, reheat to 8, ...
        s = SelfCalibratingReheatingSchedule(
            total_steps=8, cycles=4, accept_hi=math.exp(-1), accept_lo=math.exp(-4)
        )
        s.set_buffers(_peak600_buffers())
        temps = [s.reset()]
        t = temps[0]
        while t is not None:
            t = s.update(True, 8.0)
            if t is not None:
                temps.append(t)
        expected = [2.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0]
        self.assertEqual(len(temps), len(expected))
        for got, want in zip(temps, expected):
            self.assertAlmostEqual(got, want)

    def test_reheating_center_snaps_to_move_scale_not_seed(self):
        # A huge peak-load seed governs only the first step: once real move
        # scales arrive, center snaps to d / sqrt(A*B), independent of the seed.
        s = SelfCalibratingReheatingSchedule(
            total_steps=8, cycles=4, accept_hi=math.exp(-1), accept_lo=math.exp(-4)
        )
        s.set_buffers([LifetimeBoundBuffer("big", 300_000, [0, 1])])  # seed 1000
        first = s.reset()
        self.assertAlmostEqual(first, 1000.0)  # the (huge) peak-load seed
        # snap_after == 1: the first sample sets center = 8/2 = 4, and at step 1
        # of a length-2 cycle the temperature equals center.
        self.assertAlmostEqual(s.update(True, 8.0), 4.0)

    def test_reheating_center_tracks_within_cycle(self):
        # center is recomputed from d_hat every step (not only at cycle
        # boundaries), so within one long cycle it follows a change in the move
        # scale at the EMA rate beta = horizons_per_cycle / cycle_len. Here
        # total_steps=10, cycles=1 -> cycle_len=10; horizons_per_cycle=5 ->
        # beta=0.5; snap_after = min(10 // 4, 20) = 2.
        s = SelfCalibratingReheatingSchedule(
            total_steps=10,
            cycles=1,
            horizons_per_cycle=5,
            accept_hi=math.exp(-1),  # A=1, B=4 -> sqrt(A*B)=2
            accept_lo=math.exp(-4),
        )
        s.set_buffers(_peak600_buffers())
        s.reset()
        s.update(True, 10.0)  # first sample: not yet snapped, center at seed
        s.update(True, 10.0)  # second sample -> snap: d_hat=10, center=10/2=5
        self.assertAlmostEqual(s._center, 5.0)
        # Move scale drops to 2; d_hat EMA-decays 10 -> 6 -> 4 -> 3 -> 2.5 at
        # beta=0.5, and center = d_hat / 2 tracks it every step (not frozen for
        # the rest of the cycle).
        for want in (3.0, 2.0, 1.5, 1.25):
            s.update(True, 2.0)
            self.assertAlmostEqual(s._center, want)

    def test_reheating_no_snap_when_no_move_changes_quality(self):
        # move_scale always 0 (degenerate instance): center never snaps off the
        # peak-load seed, but the band still reheats and the budget is honored.
        s = SelfCalibratingReheatingSchedule(
            total_steps=4, cycles=2, accept_hi=math.exp(-1), accept_lo=math.exp(-4)
        )
        s.set_buffers(_peak600_buffers())  # seed 2.0, delta 2, alpha 0.5
        temps = [s.reset()]
        t = temps[0]
        while t is not None:
            t = s.update(False, 0.0)
            if t is not None:
                temps.append(t)
        expected = [2.0, 1.0, 2.0, 1.0]  # cool 2->1, reheat to 2, repeat
        self.assertEqual(len(temps), len(expected))
        for got, want in zip(temps, expected):
            self.assertAlmostEqual(got, want)

    def test_reheating_adaptive_budget(self):
        def n_buffers(n):
            return [LifetimeBoundBuffer(f"b{i}", 1, [0]) for i in range(n)]

        # n=100 (the random-buffer example size): 30*n = 3000, under the 5000 cap.
        s = SelfCalibratingReheatingSchedule(cycles=4)
        s.set_buffers(n_buffers(100))
        self.assertEqual(s.total_steps, 3000)
        self.assertLessEqual(s.total_steps, 5000)  # the explicit budget ceiling
        # Large n is capped; tiny n hits the floor.
        capped = SelfCalibratingReheatingSchedule()
        capped.set_buffers(n_buffers(1000))
        self.assertEqual(capped.total_steps, 5000)
        floored = SelfCalibratingReheatingSchedule()
        floored.set_buffers(n_buffers(5))
        self.assertEqual(floored.total_steps, 500)
        # Explicit budget wins.
        ex = SelfCalibratingReheatingSchedule(total_steps=42)
        ex.set_buffers(n_buffers(100))
        self.assertEqual(ex.total_steps, 42)

    def test_reheating_emits_exactly_total_steps(self):
        # Cycle boundaries and the remainder-absorbing last cycle must still emit
        # exactly total_steps temperatures, across a range of budget/cycle mixes.
        for total_steps, cycles in [(20, 4), (17, 4), (7, 3), (5, 1), (10, 10), (3, 4)]:
            s = SelfCalibratingReheatingSchedule(total_steps=total_steps, cycles=cycles)
            s.set_buffers(_peak600_buffers())
            count = 0
            t = s.reset()
            while t is not None:
                count += 1
                t = s.update(True, 5.0)
            self.assertEqual(count, total_steps, (total_steps, cycles))

    def test_reheating_reset_without_buffers_errors(self):
        s = SelfCalibratingReheatingSchedule(total_steps=10)
        with self.assertRaises(ValueError):
            s.reset()  # set_buffers() never called

    def test_default_schedule_is_auto_feasible_and_deterministic(self):
        # The solver's default schedule is the self-calibrating reheating one;
        # with the default (seeded) RNG, two runs of the same instance must agree.
        for seed in range(15):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            b1, b2 = copy.deepcopy(buffers), copy.deepcopy(buffers)
            SimulatedAnnealingLayoutSolver(cap, 128).plan_layout(b1)  # default schedule
            SimulatedAnnealingLayoutSolver(cap, 128).plan_layout(b2)
            self.assertEqual(
                [b.address for b in b1], [b.address for b in b2], f"seed={seed}"
            )
            _assert_feasible(b1, cap)

    def test_peak_memory_load(self):
        # a:[0,2) b:[1,3) c:[2,4); peak live set is {b,c} at tick 2 = 50.
        buffers = [
            LifetimeBoundBuffer("a", 10, [0, 1]),
            LifetimeBoundBuffer("b", 20, [1, 2]),
            LifetimeBoundBuffer("c", 30, [2, 3]),
        ]
        self.assertEqual(peak_memory_load(buffers), 50)
        self.assertAlmostEqual(default_initial_temperature(buffers), 50 / 300.0)


class SimulatedAnnealingTests(TestCase):
    def _run(self, buffers, capacity, *, initial, seed, alignment=128):
        solver = SimulatedAnnealingLayoutSolver(
            capacity,
            alignment,
            initial=initial,
            schedule=_short_schedule(),
            random=rnd.Random(seed),
        )
        return solver.plan_layout(buffers)

    def test_solve_skips_annealing_when_initial_already_complete(self):
        # The capacity fits all three buffers in any order, so the initial
        # first_fit layout is already globally optimal. solve()'s up-front check
        # must return before calibrating or running a single anneal.
        buffers = [
            LifetimeBoundBuffer("a", 64, [0, 1]),
            LifetimeBoundBuffer("b", 64, [0, 1]),
            LifetimeBoundBuffer("c", 64, [0, 1]),
        ]
        cap = 10_000
        solver = SimulatedAnnealingSolverWithBuffers(
            buffers,
            cap,
            128,
            initial="first_fit",
            schedule=_short_schedule(),
            random=rnd.Random(0),
        )
        self.assertTrue(solver._is_optimal())  # precondition: initial is complete
        solver.solve()
        # anneal() appends exactly one log per call, so an empty list proves no
        # anneal ran.
        self.assertEqual(solver.quality_logs, [])
        solver.finalize()
        self.assertTrue(all(b.address is not None for b in buffers))
        _assert_feasible(buffers, cap)

    def test_anneal_stops_once_all_buffers_allocated(self):
        # Order [a, b, c] leaves c stacked above capacity (only 2 of 3 fit), but
        # placing c before b lets it drop to address 0 so all three fit. From
        # this order every buffer's best reinsertion reaches that complete
        # layout, so the first annealing step lands it regardless of the RNG --
        # and the cooling loop must then break immediately rather than run the
        # schedule out.
        buffers = [
            LifetimeBoundBuffer("a", 64, [0, 1]),
            LifetimeBoundBuffer("b", 64, [1, 4]),
            LifetimeBoundBuffer("c", 64, [3, 4]),
        ]
        cap = 128
        schedule = ExponentialCoolingSchedule(
            t_initial=8.0, t_final=1.0, steps_per_epoch=2, epochs=4
        )  # 8 cooling steps if never interrupted
        solver = SimulatedAnnealingSolverWithBuffers(
            buffers,
            cap,
            1,
            initial=[0, 1, 2],
            schedule=schedule,
            random=rnd.Random(0),
        )
        self.assertEqual(solver.plan.count_allocated(), 2)  # c does not yet fit
        solver.anneal()
        self.assertEqual(solver.plan.count_allocated(), 3)  # reached completeness
        # Broke after the first iteration instead of running all 8 steps.
        self.assertEqual(len(solver.quality_logs[0]), 1)

    def test_annealing_step_swap_handles_evicted_buffers(self):
        # With None-as-eviction, evicted buffers have address None. The clean-up
        # sweep compares adjacent *non-overlapping* buffers' tops; this must treat
        # an evicted buffer as +inf (sorting it last) rather than raising on
        # `None + size`. Here b stacks on a past capacity and is evicted, while c
        # is disjoint from both -- so the adjacent pair (b, c) is non-overlapping
        # with an evicted member, exactly the comparison that used to raise.
        buffers = [
            LifetimeBoundBuffer("a", 90, [0, 1]),  # [0, 2)
            LifetimeBoundBuffer("b", 90, [0, 1]),  # [0, 2), stacks on a -> evicted
            LifetimeBoundBuffer("c", 10, [5, 6]),  # [5, 7), disjoint
        ]
        solver = SimulatedAnnealingSolverWithBuffers(
            buffers,
            100,
            1,
            initial=[0, 1, 2],
            schedule=_short_schedule(),
            random=rnd.Random(0),
        )
        self.assertIsNone(solver.plan.addresses[1])  # b is evicted
        self.assertFalse(solver.plan.overlaps(1, 2))  # b and c do not overlap
        # Must not raise (the _top_or_inf guard); b (None top) sorts after c.
        solver.annealing_step_swap(0, 2)

    def test_finalized_layout_is_feasible(self):
        for seed in range(60):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            self._run(buffers, cap, initial="first_fit", seed=seed)
            _assert_feasible(buffers, cap)

    def test_annealing_never_worse_than_initial(self):
        # Starting from a known permutation, the tracked best (and thus the
        # finalized committed total) can only improve on the initial layout.
        for seed in range(60):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            self._run(buffers, cap, initial=initial, seed=seed)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)
            _assert_feasible(buffers, cap)

    def test_deterministic_with_seed(self):
        rng = rnd.Random(0)
        n = 7
        base = _random_buffers(rng, n)
        cap = max(b.size for b in base) * 3

        first = copy.deepcopy(base)
        self._run(first, cap, initial="first_fit", seed=42)
        second = copy.deepcopy(base)
        self._run(second, cap, initial="first_fit", seed=42)

        self.assertEqual([b.address for b in first], [b.address for b in second])

    def test_reheating_schedule_end_to_end(self):
        for seed in range(40):
            rng = rnd.Random(seed)
            n = rng.randint(2, 8)
            buffers = _random_buffers(rng, n)
            cap = max(b.size for b in buffers) * rng.randint(2, 4)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            # Exercises the full streaming path: online move scale, snap off the
            # peak-load seed, and reheating cycles.
            schedule = SelfCalibratingReheatingSchedule(total_steps=200, cycles=3)
            solver = SimulatedAnnealingLayoutSolver(
                cap, 128, initial=initial, schedule=schedule, random=rnd.Random(seed)
            )
            solver.plan_layout(buffers)

            _assert_feasible(buffers, cap)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)


@unittest.skipUnless(
    _STRESS, "set TORCH_SPYRE_STRESS_SCRATCHPAD=1 to run scratchpad stress tests"
)
class SimulatedAnnealingStressTests(TestCase):
    """Heavy version of SimulatedAnnealingTests: many seeds, larger problems and a
    longer cooling schedule. Not run by default."""

    def test_many_anneals_feasible_and_not_worse(self):
        for seed in range(500):
            rng = rnd.Random(seed)
            n = rng.randint(2, 14)
            buffers = _random_buffers(rng, n, horizon=20, max_size=300)
            cap = max(b.size for b in buffers) * rng.randint(2, 5)
            initial = list(range(n))
            rng.shuffle(initial)
            initial_quality = PermutationBasedLayoutSolver(
                copy.deepcopy(buffers), list(initial), cap, 128
            ).quality()

            schedule = ExponentialCoolingSchedule(
                t_initial=200.0, t_final=0.5, steps_per_epoch=8, epochs=6
            )
            solver = SimulatedAnnealingLayoutSolver(
                cap,
                128,
                initial=initial,
                schedule=schedule,
                random=rnd.Random(seed * 7 + 1),
            )
            solver.plan_layout(buffers)

            _assert_feasible(buffers, cap)
            self.assertGreaterEqual(_committed_total(buffers), initial_quality, seed)
