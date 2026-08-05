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

"""Multi-run convergence study on an 18-buffer workload with in-place reuse.

Two in-place reuses chain: P reuses the memory of G or N (its in-place parents,
both last used at tick 15 when P starts), and the output buffer ``out`` reuses P
(P is last read at tick 17 when ``out`` starts).  Capacity is 120 -- the
theoretical peak load -- so correct in-place reuse is required to fit all 18
buffers.

Run::

    python inplace_annealing.py

Writes ``inplace_quality.png`` and ``inplace_layout.png``.  Requires
matplotlib and numpy.
"""

import time
import random as rnd
from collections import Counter

# Importing torch_spyre without torch currently sometimes fails.
import torch  # noqa: F401

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.cooling_schedules import (
    ExponentialCoolingSchedule,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingSolverWithBuffers,
)
from torch_spyre._inductor.scratchpad.utils import plot_buffers, quality_plot

# `uses` lists the ticks each buffer is accessed (its defining op and every op
# that reads it), from the 17-op chain plus its output. G and N are last read at
# op15 (by P, in-place); P, J, and Q are last read at op17 (by out, in-place onto
# P), so the output reuses P's slot while its other inputs are still live.
buffers = [
    LifetimeBoundBuffer("A", 60, [0, 1, 2]),
    LifetimeBoundBuffer("B", 30, [1, 3, 4]),
    LifetimeBoundBuffer("C", 30, [2, 3, 5, 7, 13]),
    LifetimeBoundBuffer("D", 30, [3, 4]),
    LifetimeBoundBuffer("E", 30, [4, 5]),
    LifetimeBoundBuffer("F", 60, [5, 6]),
    LifetimeBoundBuffer("G", 30, [6, 7, 15]),
    LifetimeBoundBuffer("H", 30, [7, 8]),
    LifetimeBoundBuffer("I", 30, [8, 9, 10]),
    LifetimeBoundBuffer("J", 15, [9, 11, 13, 16, 17]),
    LifetimeBoundBuffer("K", 15, [10, 11, 12]),
    LifetimeBoundBuffer("L", 15, [11, 12]),
    LifetimeBoundBuffer("M", 15, [12, 13]),
    LifetimeBoundBuffer("N", 30, [13, 14, 15]),
    LifetimeBoundBuffer("O", 45, [14, 15]),
    LifetimeBoundBuffer("P", 30, [15, 17], in_place_parents=["G", "N"]),
    LifetimeBoundBuffer("Q", 75, [16, 17]),
    LifetimeBoundBuffer("out", 30, [17], in_place_parents=["P"]),
]

CAPACITY = 120  # peak concurrent load; requires in-place reuse to fit all 17
N_RUNS = 100


def _schedule() -> ExponentialCoolingSchedule:
    # A single continuous cool. (This used to run 10 reset-to-hot "starts" of
    # 150 steps each; that facility was removed, so the budget is one 1500-step
    # cool instead.)
    return ExponentialCoolingSchedule(
        t_initial=10.0, t_final=1.0, steps_per_epoch=1, epochs=1500
    )


rng = rnd.Random(0)
all_logs: list[list[int]] = []
last_solver: SimulatedAnnealingSolverWithBuffers | None = None

t0 = time.perf_counter()
for _ in range(N_RUNS):
    solver = SimulatedAnnealingSolverWithBuffers(
        buffers,
        size=CAPACITY,
        alignment=1,
        initial="first_fit",
        schedule=_schedule(),
        random=rng,
    )
    solver.solve()
    all_logs.append([q for run in solver.quality_logs for q in run])
    last_solver = solver

elapsed = time.perf_counter() - t0
print(f"{N_RUNS} runs in {elapsed:.2f}s ({elapsed / N_RUNS * 1000:.1f}ms/run)")

final_qualities = [log[-1] for log in all_logs]
print(f"Final quality distribution (out of 18): {Counter(final_qualities)}")

assert last_solver is not None
# quality_plot uses the last solver's logs; the temperature overlay matches.
quality_plot(last_solver.quality_logs, last_solver.temperature_logs[0]).savefig(
    "inplace_quality.png", dpi=300
)
last_solver.finalize()
plot_buffers(buffers, CAPACITY).savefig("inplace_layout.png", dpi=300)
print("Saved inplace_quality.png, inplace_layout.png")
