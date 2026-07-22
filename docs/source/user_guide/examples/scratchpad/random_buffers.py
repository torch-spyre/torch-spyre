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

"""Compare first-fit vs simulated-annealing quality on a set of random buffers.

Capacity is set to the theoretical peak load (i.e. the minimum possible).
The simulated-annealing search improves on the first-fit ordering by reducing
fragmentation so that more buffers fit within the fixed capacity.

Run::

    python random_buffers.py

Writes ``random_buffers_quality.png`` and ``random_buffers_layout.png``.
Requires matplotlib and numpy.
"""

import copy
import math
import random as rnd

# Importing torch_spyre without torch currently sometimes fails.
import torch  # noqa: F401

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.cooling_schedules import (
    peak_memory_load,
    ExponentialCoolingSchedule,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingSolverWithBuffers,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.utils import plot_buffers, quality_plot


def _random_buffer(
    name: str,
    size_range: int,
    time_range: int,
    random: rnd.Random,
) -> LifetimeBoundBuffer:
    """Generate a random buffer biased towards large sizes and short lifetimes."""
    duration = random.randrange((time_range - 1) // 2)
    duration = duration * duration // (time_range - 1)
    t_start = random.randrange(time_range - duration)
    t_end = t_start + duration + 1
    size = random.randrange(size_range)
    size = max(1, math.isqrt(size * size_range))
    # Live at ticks [t_start, t_end] inclusive; uses records the first and last.
    uses = [t_start] if t_end == t_start else [t_start, t_end]
    return LifetimeBoundBuffer(name, size, uses)


random = rnd.Random(0)
N = 100
buffers = [_random_buffer(f"B{i}", 1_000_000, N, random) for i in range(N)]

peak_load = peak_memory_load(buffers)
capacity = peak_load // 2
total_size = sum(b.size for b in buffers)
print(
    f"N={N} buffers, peak load: {peak_load}, capacity: {capacity}, total size: {total_size}"
)

# First-fit baseline on a deep copy so buffer.address fields stay clean.
ff_buffers = copy.deepcopy(buffers)
FirstFitLayoutSolver(capacity).plan_layout(ff_buffers)
ff_quality = sum(b.size for b in ff_buffers if b.address is not None)
print(f"First-fit quality: {ff_quality}/{total_size}")

# Simulated annealing, seeded from the first-fit ordering.
solver = SimulatedAnnealingSolverWithBuffers(
    buffers,
    capacity,
    alignment=1,
    initial="first_fit",
    random=random,
    schedule=ExponentialCoolingSchedule(
        t_initial=500000.0, t_final=50000.0, steps_per_epoch=30, epochs=100
    ),
)
solver.solve()
solver.finalize()
print(f"Simulated-annealing quality: {solver.best_quality}/{total_size}")

quality_plot(solver.quality_logs, solver.temperature_logs[0]).savefig(
    "random_buffers_quality.png", dpi=300
)
plot_buffers(buffers, capacity).savefig("random_buffers_layout.png", dpi=300)
print("Saved random_buffers_quality.png, random_buffers_layout.png")
