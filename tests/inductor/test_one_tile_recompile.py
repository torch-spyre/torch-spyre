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

"""A declared dim promises one binary for every legal size (#6018).

When the declared minimum equals the granularity, the smallest legal input is
exactly one tile. PyTorch's contiguity check asks "is this size 1?" before it
looks at the stride, and answering that writes an ``Ne(count, 1)`` guard that a
one-tile call then fails, which recompiles. ``enable_spyre_context`` installs a
stride-first check for the duration of the Spyre backend compile, so no such
guard is written.
"""

import unittest

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME

COLS = 64
TILE = 64
MIN_ROWS = 64
MAX_ROWS = 512

ATOL = 1e-2
RTOL = 1e-2


def tiled_gelu(x):
    """One tiled pointwise region over a dimension declared dynamic."""
    rows = x.shape[0]
    torch._check(rows >= MIN_ROWS)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


class TestOneTileReusesTheBinary(unittest.TestCase):
    """min == granularity: the one-tile call reuses the binary (#6018)."""

    def tearDown(self):
        torch.compiler.set_stance("default")
        torch._dynamo.reset()

    def test_one_tile_call_does_not_recompile(self):
        torch._dynamo.reset()
        contract = {0: {"min": MIN_ROWS, "max": MAX_ROWS, "granularity": TILE}}
        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
        for rows in (2 * TILE, MIN_ROWS):
            x = torch.randn(rows, COLS, dtype=torch.float16)
            out = compiled(x.to(DEVICE_NAME, dynamic=contract))
            torch.testing.assert_close(
                out.cpu().float(),
                torch.nn.functional.gelu(x.float()),
                atol=ATOL,
                rtol=RTOL,
            )
            torch.compiler.set_stance("fail_on_recompile")


if __name__ == "__main__":
    unittest.main()
