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

"""Plot the layout for a fixed ordering of four buffers (no annealing).

Each buffer's ``uses`` list is the ticks at which it is accessed; here
``[first, last]``, so it is live over the half-open range ``[first, last + 1)``.

The identity ordering [B0, B1, B2, B3] stacks buffers by arrival and produces
a peak height of 22.  Run::

    python toy_layout.py

Writes ``toy_layout.png`` to the current directory.  Requires matplotlib.
"""

# Importing torch_spyre without torch currently sometimes fails.
import torch  # noqa: F401

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.cooling_schedules import (
    ExponentialCoolingSchedule,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingSolverWithBuffers,
)
from torch_spyre._inductor.scratchpad.utils import plot_buffers

buffers = [
    LifetimeBoundBuffer("B0", 8, [0, 1]),
    LifetimeBoundBuffer("B1", 4, [1, 4]),
    LifetimeBoundBuffer("B2", 2, [2, 5]),
    LifetimeBoundBuffer("B3", 8, [3, 5]),
]

solver = SimulatedAnnealingSolverWithBuffers(
    buffers,
    size=14,
    alignment=1,
    initial=[0, 1, 2, 3],
    schedule=ExponentialCoolingSchedule(
        t_initial=10.0, t_final=1.0, steps_per_epoch=10, epochs=10
    ),
)
print("Solving...")
solver.solve()
plot_buffers(buffers, 22).savefig("toy_layout.png", dpi=300)
print("Saved toy_layout.png")
