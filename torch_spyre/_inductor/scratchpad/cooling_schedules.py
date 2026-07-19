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


"""Cooling schedules for the simulated-annealing layout solver.

A :class:`CoolingSchedule` is a responsive temperature controller: the annealer
streams back each step's acceptance and move scale, so a schedule may adapt
online. This module holds the schedule ABC and its concrete implementations,
plus the peak-load helpers used to seed an initial temperature.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

from typing_extensions import override
from heapq import heappush, heappop

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer


def peak_memory_load(buffers: list[LifetimeBoundBuffer]) -> int:
    """Maximum total size of simultaneously-live buffers (a lower bound on the
    space any layout needs). Swept over lifetime start points."""
    by_start = sorted(buffers, key=lambda b: b.start_time)
    current_load = 0
    peak_load = 0
    end_points: list[tuple[int, int]] = []  # (end_time, size) min-heap
    for buffer in by_start:
        while end_points and end_points[0][0] <= buffer.start_time:
            current_load -= heappop(end_points)[1]
        current_load += buffer.size
        peak_load = max(peak_load, current_load)
        heappush(end_points, (buffer.end_time, buffer.size))
    return peak_load


def default_initial_temperature(buffers: list[LifetimeBoundBuffer]) -> float:
    """A principled starting temperature from the peak memory load -- the paper's
    tau_s. Used when a schedule is not given an explicit ``t0``."""
    return peak_memory_load(buffers) / 300.0


class CoolingSchedule(ABC):
    """A *responsive* temperature controller for simulated annealing.

    Unlike a blind temperature iterator, after every step the annealer reports
    both whether the step accepted a move and the *move scale* -- the mean
    ``|Δquality|`` over the reinsertion positions it probed (ignoring no-op
    positions) -- so a schedule may adapt: detect a stall, reheat, or size its
    temperatures to the instance's move magnitudes online. :meth:`reset` begins
    a fresh anneal and returns the first temperature; :meth:`update` consumes the
    latest step's acceptance and move scale and returns the next temperature, or
    ``None`` to stop. ``reset`` must fully reinitialize transient state (so a
    schedule can be reused across anneals).
    """

    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        """Preparation hook: the solver calls this with the buffer set before
        annealing, so a schedule may derive parameters (e.g. an initial
        temperature from the peak load). Default: no-op."""

    @abstractmethod
    def reset(self) -> Optional[float]:
        """Reinitialize and return the first temperature (None for no steps)."""

    @abstractmethod
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
        """Return the next temperature given the last step's acceptance and move
        scale (mean ``|Δquality|`` over probed reinsertions, ``0.0`` if none
        changed quality), or None to stop."""


class ExponentialCoolingSchedule(CoolingSchedule):
    """Geometric cooling over ``steps_per_epoch * epochs`` steps, dropping by a
    constant factor once per epoch. Ignores acceptance."""

    def __init__(
        self, *, t_initial: float, t_final: float, steps_per_epoch: int, epochs: int
    ):
        """A schedule that starts at temperature `t_initial` and ends at `t_final`,
        cooling down by a constant factor every `steps_per_epoch` steps. There are
        `epochs` such epochs.

        If `epochs == 1`, then the temperature stays at `t_initial`."""
        self.t_initial = t_initial
        if epochs <= 1:
            self.alpha = 1.0
        else:
            self.alpha = (t_final / t_initial) ** (1 / (epochs - 1))
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self._t = t_initial
        self._i = 0

    @override
    def reset(self) -> Optional[float]:
        self._t = self.t_initial
        self._i = 0
        return self._t

    @override
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
        self._i += 1
        if self._i >= self.steps_per_epoch * self.epochs:
            return None
        if self._i % self.steps_per_epoch == 0:
            self._t *= self.alpha
        return self._t


class SelfCalibratingReheatingSchedule(CoolingSchedule):
    """Self-calibrating simulated-annealing schedule with reheating cycles.

    This is the default schedule. It needs no tuning beyond the step budget: it
    sizes its temperatures to the instance online from the move scale the
    annealer streams back, locates the productive temperature, and spends the
    budget on reheating cycles around it -- concentrating moves where they are
    useful but not frozen, rather than cooling monotonically once.

    NOTE: like its predecessor this is a *reasonable* self-calibrating default,
    not a tuned or provably-good one -- we do not yet have representative example
    models to benchmark against. Two bets are unvalidated for our landscape:
    that reheating beats a single long cool, and that learning the move scale
    online beats a pre-committed warm-up. Both are bounded by the solver's
    best-seen tracking, so they can waste budget but never worsen the result.
    Expect to revisit ``cycles`` and the acceptance band once we can benchmark.

    Temperature scale (self-calibration). With ``A = -ln(accept_hi)`` and
    ``B = -ln(accept_lo)``, a band centered on temperature ``center`` accepts a
    mean-magnitude *worsening* move with probability ``accept_hi`` at its top
    ``center * delta`` and ``accept_lo`` at its bottom ``center / delta``, where::

        delta  = sqrt(B / A)              # band half-width, scale-independent
        center = d_hat / sqrt(A * B)      # productive temperature

    ``d_hat`` is an exponential moving average of the streamed move scale (mean
    ``|Δquality|`` over probed reinsertions), so ``center`` tracks the move scale
    as the layout improves and the landscape flattens.

    Bootstrap. Before any move scale is known, ``center`` is seeded from the
    peak-load estimate (:func:`default_initial_temperature`), placed at the band
    top. That estimate is in bytes rather than quality units, so it is only a
    rough magnitude -- but it governs a single step before the first real
    samples snap ``center`` onto the data scale, and best-seen tracking absorbs
    that step regardless.

    Reheating. The budget is split into ``cycles`` equal cycles (the last
    absorbing any remainder). Each cools geometrically from ``center * delta``
    down to ``center / delta``. ``center`` is recomputed from ``d_hat`` *every
    step*, so the band drifts downward (or re-expands) with the landscape
    continuously rather than jumping only at cycle boundaries -- which matters
    most when cycles are long (and for ``cycles = 1``, a single tracked cool,
    the only case where the boundary-only variant never re-centered at all).
    The EMA horizon is ``cycle_len / horizons_per_cycle``: with the default
    ``horizons_per_cycle = 2`` the center lags the move scale by about half a
    cycle, so a stale band never persists for a large fraction of a cycle.

    Budget knobs:
        total_steps: the annealing budget (temperatures emitted). ``None`` ->
            adaptive, ``clamp(steps_per_buffer * n, min_steps, max_steps)``.
        cycles: number of reheating cycles.
        horizons_per_cycle: EMA horizons per cycle (``H`` in the design notes);
            the move-scale EMA horizon is ``cycle_len / horizons_per_cycle``, so
            larger values track the landscape faster (at the cost of more noise
            and a stronger pull toward greedy cooling). Guessed default pending
            benchmarks, like ``cycles``.
        max_steps: hard cap on the adaptive budget (default 5000, keeping the
            n=100 random-buffer example bounded).
    """

    def __init__(
        self,
        *,
        total_steps: Optional[int] = None,
        cycles: int = 4,
        horizons_per_cycle: float = 2.0,
        steps_per_buffer: int = 30,
        min_steps: int = 500,
        max_steps: int = 5000,
        accept_hi: float = 0.8,
        accept_lo: float = 0.01,
    ):
        if not 0.0 < accept_lo < accept_hi < 1.0:
            raise ValueError("need 0 < accept_lo < accept_hi < 1")
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        if horizons_per_cycle <= 0.0:
            raise ValueError("horizons_per_cycle must be > 0")
        self._total_steps = total_steps
        self.cycles = cycles
        self.horizons_per_cycle = horizons_per_cycle
        self.steps_per_buffer = steps_per_buffer
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.accept_hi = accept_hi
        self.accept_lo = accept_lo
        # The reheat band is fixed by the two acceptance targets alone: a factor
        # `delta` above/below the center, with `sqrt(A*B)` converting the move
        # scale into the center temperature. Both are scale-independent.
        a = -math.log(accept_hi)
        b = -math.log(accept_lo)
        self._rt_ab = math.sqrt(a * b)
        self._delta = math.sqrt(b / a)
        # Sized in set_buffers (needs the buffer count and the peak load); _cycle_len
        # == 0 marks "not yet sized" so reset() can refuse to run uncalibrated.
        self.total_steps = total_steps or 0
        self._cycle_len = 0
        self._seed_center = 1.0

    @override
    def set_buffers(self, buffers: list[LifetimeBoundBuffer]) -> None:
        if self._total_steps is None:
            self.total_steps = min(
                self.max_steps,
                max(self.min_steps, self.steps_per_buffer * len(buffers)),
            )
        else:
            self.total_steps = max(1, self._total_steps)
        self._cycle_len = max(1, self.total_steps // self.cycles)
        # Cool by a factor delta^2 across one cycle (band top to band bottom).
        self._alpha = self._delta ** (-2.0 / self._cycle_len)
        # Move-scale EMA horizon of cycle_len / horizons_per_cycle steps, so the
        # center (recomputed every step) lags the landscape by that fraction of a
        # cycle. Clamped to a valid EMA rate for short cycles (where the ratio can
        # reach or exceed 1); there it degrades to "center = latest scale".
        self._ema_beta = min(1.0, self.horizons_per_cycle / self._cycle_len)
        # Average this many nonzero samples before snapping center off the seed
        # (a cheap, low-variance bootstrap; at least one).
        self._snap_after = min(self._cycle_len // 4, 20) or 1
        # Seed center so the peak-load estimate lands at the band top.
        self._seed_center = default_initial_temperature(buffers) / self._delta

    @override
    def reset(self) -> Optional[float]:
        if self._cycle_len == 0:
            raise ValueError(
                "SelfCalibratingReheatingSchedule must be given buffers before "
                "use; run it through SimulatedAnnealingSolverWithBuffers, or call "
                "set_buffers() first."
            )
        self._i = 0
        self._s = 0
        self._cycle = 0
        self._center = self._seed_center
        self._d_hat: Optional[float] = None
        self._sample_sum = 0.0
        self._n_samples = 0
        return self._temperature()

    def _temperature(self) -> float:
        # Position s within the cycle: s == 0 is the band top (center*delta),
        # cooling by alpha each step toward the band bottom (center/delta).
        return self._center * self._delta * self._alpha**self._s

    @override
    def update(self, accepted: bool, move_scale: float) -> Optional[float]:
        # Track the move scale, ignoring no-op reinsertions (move_scale == 0):
        # they dominate the sample and would collapse the center into a greedy
        # search. Before the first snap, average a few samples; after it, EMA.
        if move_scale > 0.0:
            if self._d_hat is None:
                self._sample_sum += move_scale
                self._n_samples += 1
                if self._n_samples >= self._snap_after:
                    self._d_hat = self._sample_sum / self._n_samples
            else:
                self._d_hat += self._ema_beta * (move_scale - self._d_hat)
        # Re-center from the current move scale every step, so the band tracks
        # the landscape continuously within a cycle rather than only at its
        # boundaries. Until the first snap ``d_hat`` is None and center stays at
        # the peak-load seed.
        if self._d_hat is not None:
            self._center = self._d_hat / self._rt_ab

        self._i += 1
        if self._i >= self.total_steps:
            return None
        self._s += 1
        # Cycle boundary: restart the cool at the band top. The last cycle
        # absorbs the budget remainder. (Center already tracks every step, so the
        # boundary only restarts the carrier phase.)
        if self._s >= self._cycle_len and self._cycle < self.cycles - 1:
            self._cycle += 1
            self._s = 0
        return self._temperature()
