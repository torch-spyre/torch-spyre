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

# Dynamic shape version of gelu.py
# Dim 0 is declared on the tensor with x.to("spyre", dynamic={0: SPEC}) and the
# function is compiled with the default dynamic=None. gelu runs inside a
# for_each_tile loop, one TILE-row tile per step, so one compiled program
# serves every row count in [min, max]. torch.compile(dynamic=True) is refused
# on Spyre: it gives every dim a symbol with no declared range.

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters

from torch_spyre._inductor.wsr import for_each_tile

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

TILE = 64  # 64 fp16 values fill one 128-byte stick
COLS = 1024
# The minimum is two tiles: a one-tile call recompiles until #6060 lands (#6018).
SPEC = {"min": 2 * TILE, "max": 512, "granularity": TILE}
# The compile size first, then the minimum, the maximum and sizes between.
SIZES = [256, 128, 512, 448, 192]


def gelu_fn(a):
    # Restate the declared range inside the trace.
    rows = a.shape[0]
    torch._check(rows >= SPEC["min"])
    torch._check(rows <= SPEC["max"])
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, F.gelu(tile)

    _carry, out = for_each_tile(body, (a,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


compiled_fn = torch.compile(gelu_fn, fullgraph=True)

x = torch.rand(SPEC["max"], COLS, dtype=torch.float16)
# Reserves the device buffer at max rows, so resize_ within the range keeps
# the device layout and does not recompile.
x_device = x.to(DEVICE, dynamic={0: SPEC})

counters.clear()
for rows in SIZES:
    x_device.resize_((rows, COLS))
    compiled_result = compiled_fn(x_device).cpu().float()
    cpu_result = F.gelu(x[:rows].float())

    # Compare results
    cpu_delta = torch.abs(compiled_result - cpu_result).max()
    print(f"rows={rows}: Max delta Compiled Spyre vs. CPU: {cpu_delta}")
    torch.testing.assert_close(compiled_result, cpu_result, atol=1e-2, rtol=1e-2)

graphs = counters["stats"]["unique_graphs"]
print(f"Compiled graphs: {graphs}")
assert graphs == 1, "a size in the declared range caused a recompile"
