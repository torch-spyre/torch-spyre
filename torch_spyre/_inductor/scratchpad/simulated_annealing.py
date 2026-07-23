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


# Implement the algorithm from this paper:
#
# Imanishi, Akifumi, and Zijian Xu. "A heuristic for periodic memory allocation
# with little fragmentation to train neural networks." In Proceedings of the 2024
# ACM SIGPLAN International Symposium on Memory Management, pp. 82-94. 2024.
#
# The paper describes a few algorithms that work together to come up with a good
# allocation scheme. The problem setting differs slightly from ours in that they
# have a fixed set of buffers that are all to be allocated, and they want to do it
# in as little space as possible. By contrast, in our case, we have a fixed amount
# of space and we want to allocate those buffers that will give the best
# performance -- which we can probably approximate by saying, we want to minimize
# the volume of HBM transfers.
#
# Algorithm 4 is the simulated annealing algorithm that comes up with the
# permutation. It takes as inputs an annealing schedule, a list of buffers, and an
# initial permutation. One iteration randomly selects a buffer, and then cleverly
# compares all possible positions where the buffer could be reinserted. In effect,
# it cheaply considers (n-1) neighbours every iteration.
#
# In order to adjust this algorithm to our setting, we hold a
# PermutationBasedLayoutSolver from plan_solver as a member. It lets us use a
# permutation of buffers as a source of a layout plan, and modify the permutation
# and see the modification in the layout plan by repeated swapping. Each
# reinsertion sweep runs on a throwaway plan.copy(), so the live plan only performs
# the rotation that is actually accepted rather than sweeping and restoring. We
# also adjust our random sampling: a buffer that is currently allocated legally
# gets to consider being inserted into all other positions, whereas a buffer that
# is not currently allocated legally only gets to consider being reinserted in
# positions of (nearly) legally allocated buffers, so that we don't spend too much
# time on swaps that have no effect.

import math
import copy
from typing import Literal, Optional, Sequence, TypeAlias
import random as rnd

from torch_spyre._inductor.scratchpad.cooling_schedules import (
    CoolingSchedule,
    SelfCalibratingReheatingSchedule,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
)


class SolverToPermutation:
    def __init__(self, solver: MemoryPlanSolver):
        self.solver = solver

    def permutation(self, buffers: list[LifetimeBoundBuffer]) -> list[int]:
        """Lay out the given buffers, then sort them by their addresses. Any
        non-allocated buffers come after all allocated buffers. Return this ordering
        as a list of indices; the first index is i such that buffers[i] is one of the
        buffers allocated at address 0, etc. This yields a permutation that gives the
        given layout, or an equivalent one, or occasionally even a better one."""
        allocated_buffers = self.solver.plan_layout(copy.deepcopy(buffers))
        # Typically, allocated_buffers is just the argument to plan_layout, which has
        # been modified in-place. But we can't assume that. Moreover, we need to
        # protect the passed in buffers from being modified by the given solver.

        max_address = max(
            (b.address for b in allocated_buffers if b.address is not None), default=0
        )
        name_to_address = {
            b.name: (b.address if b.address is not None else max_address + 1)
            for b in allocated_buffers
        }
        return sorted(
            list(range(len(buffers))), key=lambda i: name_to_address[buffers[i].name]
        )


SolverInitialOption: TypeAlias = (
    list[int] | Literal["first_fit", "best_fit", "greedy"] | MemoryPlanSolver
)
SolverScheduleOption: TypeAlias = CoolingSchedule | Literal["auto"]


class SimulatedAnnealingLayoutSolver(MemoryPlanSolver):
    """We can only do the full initialization when we know the list of buffers, so
    this class is just a shim to create the actual solver."""

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        *,
        initial: SolverInitialOption = "first_fit",
        schedule: SolverScheduleOption = "auto",
        random: Optional[rnd.Random] = None,
    ):
        super().__init__(size, alignment)
        self.initial = initial
        self.schedule = schedule
        self.random = random

    def plan_layout(
        self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        _buffers = list(buffers)
        solver = SimulatedAnnealingSolverWithBuffers(
            _buffers,
            self.limit,
            self.alignment,
            initial=self.initial,
            schedule=self.schedule,
            random=self.random,
        )
        solver.solve()
        solver.finalize()
        return _buffers


class SimulatedAnnealingSolverWithBuffers:
    """Drives simulated annealing over a :class:`PermutationBasedLayoutSolver`.

    The layout is held as a *member* (``self.plan``), not a base class. This lets
    each reinsertion sweep run on a throwaway ``plan.copy()`` while the live plan
    only ever performs the single rotation that is actually accepted (and the
    cleanup swaps) -- so the live layout is never churned through a full sweep.
    """

    def __init__(
        self,
        buffers: list[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        *,
        initial: SolverInitialOption = "first_fit",
        schedule: SolverScheduleOption = "auto",
        random: Optional[rnd.Random] = None,
    ):
        if isinstance(initial, list):
            if sorted(initial) != list(range(len(buffers))):
                raise ValueError(
                    f"given initial list is not a permutation of range({len(buffers)})"
                )
            self.initial = initial
        else:
            if initial == "first_fit":
                initial = FirstFitLayoutSolver(size, alignment)
            elif initial == "best_fit":
                initial = BestFitLayoutSolver(size, alignment)
            elif initial == "greedy":
                initial = GreedyLayoutSolver(size, alignment)

            assert isinstance(initial, MemoryPlanSolver)
            convertor = SolverToPermutation(initial)
            self.initial = convertor.permutation(buffers)

        self.buffers = buffers
        self.size = size
        self.alignment = alignment
        self.plan = PermutationBasedLayoutSolver(buffers, self.initial, size, alignment)
        self.quality_logs: list[list[float]] = []
        self.temperature_logs: list[list[float]] = []
        self.best_quality = self.plan.quality()
        self.best_permutation = copy.copy(self.initial)

        if isinstance(schedule, str):
            self.schedule: CoolingSchedule
            if schedule == "auto":
                self.schedule = SelfCalibratingReheatingSchedule()
            else:
                raise ValueError(
                    f"this string does not describe a known schedule: {schedule}"
                )
        else:
            self.schedule = schedule
        # Let the schedule derive any buffer-dependent parameters (e.g. t0).
        self.schedule.set_buffers(buffers)

        if random is not None:
            self.random = random
        else:
            # Default to a fixed seed so layout planning is deterministic: the
            # same graph must compile to the same scratchpad layout across runs
            # (build reproducibility and compilation caching depend on it). Pass
            # an explicit Random to vary the search (e.g. in benchmarks/tests).
            self.random = rnd.Random(0)

    def finalize(self) -> None:
        self.plan.finalize()

    def _is_optimal(self) -> bool:
        """True once every buffer is fully allocated below capacity.

        Quality is then at its upper bound -- each buffer already contributes
        its full :func:`buffer_quality`, so no rotation or swap can improve it
        and the search can stop.
        """
        return self.plan.count_allocated() == len(self.buffers)

    def solve(self) -> None:
        # If the initial layout already fits every buffer it is globally
        # optimal, so skip annealing outright. (Once annealing is under way the
        # inner loop's own check terminates it; this guard is only ever reached
        # with the untouched initial plan.)
        if self._is_optimal():
            return
        self.anneal()
        # Commit the best permutation seen, so finalize() writes it rather than
        # whatever state annealing happened to end in.
        if self.plan.permutation != self.best_permutation:
            self.plan = PermutationBasedLayoutSolver(
                self.buffers, list(self.best_permutation), self.size, self.alignment
            )

    def anneal(self) -> None:
        quality_log: list[float] = []
        temperature_log: list[float] = []

        temperature = self.schedule.reset()
        while temperature is not None:
            move, move_scale = self.annealing_step_rotate(temperature)
            if move is not None:
                self.annealing_step_swap(*move)

            quality = self.plan.quality()
            quality_log.append(quality)
            temperature_log.append(temperature)
            if quality > self.best_quality:
                self.best_quality = quality
                self.best_permutation = copy.copy(self.plan.permutation)

            if self._is_optimal():
                # All buffers fit: quality is maximal, so stop cooling early --
                # the best layout above is this (globally optimal) one.
                break

            temperature = self.schedule.update(move is not None, move_scale)

        self.quality_logs.append(quality_log)
        self.temperature_logs.append(temperature_log)

    def annealing_step_swap(self, i: int, j: int) -> None:
        """This is the loop mentioned as Algorithms 5 and 6 in the paper."""
        plan = self.plan
        perm = plan.permutation
        assert i != j, (
            "for a rotation i -> i, we should return None from the rotation method"
        )
        assert 0 <= i < len(perm)
        assert 0 <= j < len(perm)

        if i > j:
            i, j = j, i
        # Now i < j, and perm[:i] and perm[j+1:] are "clean"; that is, there is no k
        # such that perm[k] and perm[k+1] are buffers that *do not overlap* in time,
        # and have perm[k] have a higher end point in memory than perm[k+1]. Because
        # perm[i] up to and including perm[j] changed, we need to examine
        # i-1 <= k <= j -- except if that would take us outside the bounds of perm,
        # of course.
        i -= 1

        # Ensure that both i and j+1 are valid indices.
        if i < 0:
            i = 0
        if j == len(perm) - 1:
            j = len(perm) - 2

        def _top_or_inf(p: int) -> float:
            # Exclusive top (address + size) of buffer ``p``, or +inf when ``p``
            # is evicted (address is None). An evicted buffer sorts as if it sits
            # arbitrarily high, so it is treated as "above" any placed buffer and
            # is never reordered below one; two placed buffers compare by their
            # real tops, unchanged from before eviction used None.
            addr = plan.addresses[p]
            if addr is None:
                return math.inf
            return addr + self.buffers[p].size

        while i <= j:
            pi = perm[i]
            pi1 = perm[i + 1]

            if (not plan.overlaps(pi, pi1)) and _top_or_inf(pi) > _top_or_inf(pi1):
                # Swap buffers pi and pi1. This makes no difference for the quality
                # of the result *now*, but it makes it easier to rotate to an
                # improved state.
                plan.swap(i)

                # Adjust the bounds of what we need to examine.
                if i == j and j < len(perm) - 2:
                    j += 1
                if i > 0:
                    i -= 1
                else:
                    i = 1
            else:
                i += 1

    def annealing_step_rotate(
        self, temperature: float
    ) -> tuple[Optional[tuple[int, int]], float]:
        """This is the inner loop of Algorithm 4 from the paper. The first return
        value is (i, j) iff we accepted a rotation inserting entry i of the
        permutation into position j != i; None if we accepted no rotation. We never
        accept a trivial rotation. The second return value is the move scale for this
        step -- the mean |Δquality| over the reinsertion positions probed, ignoring
        no-op positions -- which the schedule uses to size its temperatures online.

        The reinsertion sweep runs on a throwaway copy of the plan; only the accepted
        rotation (if any) is applied to the live plan, so the live layout never has
        to sweep-and-restore."""
        plan = self.plan
        n = len(self.buffers)
        allocated = [plan.is_fully_allocated(plan.permutation[i]) for i in range(n)]
        n_allocated = sum(1 if b else 0 for b in allocated)
        # Choose each allocated buffer with weight n and each non-allocated buffer
        # with weight n_allocated + 1.
        i = self.random.choices(
            range(n), weights=[n if b else n_allocated + 1 for b in allocated]
        )[0]

        # qualities[j] is the quality if we rotate i to position j in the
        # permutation, or None if we don't consider rotating i to position j.
        qualities: list[Optional[float]] = [None] * n
        quality_before = plan.quality()

        # Probe all reinsertion positions on a copy: rotate i to position 0, then
        # bubble it forward one step at a time, recording the quality at each
        # position it visits.
        probe = plan.copy()
        if i != 0:
            probe.rotate(i, 0)
            qualities[0] = probe.quality()
        if allocated[i]:
            upper_bound = n - 1
        else:
            # x is not legally allocated, so it can only be made to fit by moving it
            # earlier; the last legally-allocated buffer sits at position k, so only
            # positions 0..k+1 can change the quality. (See the monotonicity
            # argument: x's address is non-decreasing in its position. Eviction
            # preserves this: moving x later only adds earlier-positioned overlapping
            # buffers to what it must stack on, so its top -- and hence whether it is
            # evicted -- is monotone in position, with an evicted address read as
            # +inf.)
            upper_bound = (
                max((pos for pos, b in enumerate(allocated) if b), default=0) + 1
            )
            if upper_bound > n - 1:
                upper_bound = n - 1

        for p in range(1, upper_bound + 1):
            probe.swap(p - 1)  # bubble x from position p-1 to position p
            if p != i:
                qualities[p] = probe.quality()

        # Move scale streamed to the schedule: mean |Δquality| over the probed
        # reinsertion positions, ignoring no-op positions (which dominate the
        # set and would otherwise collapse the schedule's temperature). This is
        # the online analogue of the peak-load seed, in the right quality units.
        probed = [abs(q - quality_before) for q in qualities if q is not None]
        nonzero = [d for d in probed if d > 0.0]
        move_scale = sum(nonzero) / len(nonzero) if nonzero else 0.0

        insertion_points = [pos for pos, q in enumerate(qualities) if q is not None]
        insertion_points = sorted(
            insertion_points,
            key=lambda pos: -qualities[pos],  # type: ignore
        )

        for j in insertion_points:
            assert i != j
            qj = qualities[j]
            assert qj is not None
            if qj > quality_before or self.random.random() < math.exp(
                (qj - quality_before) / temperature
            ):
                # Apply only the accepted rotation to the live plan (others keep
                # their order).
                plan.rotate(i, j)
                return (i, j), move_scale

        # Nothing accepted: the live plan was never touched.
        return None, move_scale
