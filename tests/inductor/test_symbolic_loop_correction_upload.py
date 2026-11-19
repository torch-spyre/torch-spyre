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

"""The correction upload of a symbolic-count kernel follows the trip count.

A kernel compiled over a declared dimension carries one program-correction
slot per candidate iteration, ``max / granularity`` of them, and the host
fills the whole table on every launch. The device reads slot k at iteration k
only, so the launch uploads the address table plus the first n slot copies and
leaves the rest of the region as it is (``JobPlanStepHostCompute`` in
``csrc/job_plan.cpp``). What the scatter reads past a copy's own flits goes to
dummy addresses by construction, so the stale bytes beyond slot n never reach
the program.

WHAT THIS ASSERTS. A sequence of launches with a fresh declared tensor per
step (new buffers at new addresses, as a serving plugin allocates its page
tables per step) and trip counts going up and down, the maximum included, gives
the CPU's values at every step. The counters confirm the uploads below the
maximum were cut, and that ``TORCH_SPYRE_FULL_CORRECTION_UPLOAD=1`` restores
the whole upload.
"""

import os
import unittest
from unittest import mock

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._C import _correction_upload_stats
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME

COLS = 64
TILE = 64
MAX_TILES = 32
MAX_ROWS = MAX_TILES * TILE
SPEC = {"min": TILE, "max": MAX_ROWS, "granularity": TILE}
# Trip counts going up and down: the maximum twice, so a launch below it
# follows one that filled every slot, and the minimum in between.
TILES = [32, 4, 1, 9, 2, 32, 3, 1, 16, 5]
ATOL = 1e-2
RTOL = 1e-2


def tiled_gelu(x):
    rows = x.shape[0]
    torch._check(rows >= TILE)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


def _fresh_declared(rows, seed):
    """A new host tensor, declared at the maximum and resized to ``rows``."""
    generator = torch.Generator().manual_seed(seed)
    host = torch.randn(MAX_ROWS, COLS, dtype=torch.float16, generator=generator)
    on_device = host.to(DEVICE_NAME, dynamic={0: SPEC})
    on_device.resize_((rows, COLS))
    return host, on_device


def _delta(before, after):
    return {key: after[key] - before[key] for key in before}


class TestCorrectionUploadFollowsTheCount(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def _launch(self, tiles):
        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
        for step, count in enumerate(tiles):
            rows = count * TILE
            host, on_device = _fresh_declared(rows, seed=1000 + step)
            out = compiled(on_device).cpu().float()
            torch.testing.assert_close(
                out,
                torch.nn.functional.gelu(host[:rows].float()),
                atol=ATOL,
                rtol=RTOL,
                msg=f"step {step} at {count} tiles",
            )
        torch.spyre.synchronize()

    def test_values_right_with_counts_going_up_and_down(self):
        before = _correction_upload_stats()
        self._launch(TILES)
        delta = _delta(before, _correction_upload_stats())

        self.assertGreaterEqual(delta["launches"], len(TILES))
        self.assertEqual(delta["prefixed"], sum(1 for c in TILES if c < MAX_TILES))
        self.assertLess(delta["bytes_sent"], delta["bytes_full"])

    def test_the_whole_upload_can_be_forced(self):
        with mock.patch.dict(os.environ, {"TORCH_SPYRE_FULL_CORRECTION_UPLOAD": "1"}):
            before = _correction_upload_stats()
            self._launch(TILES[:4])
            delta = _delta(before, _correction_upload_stats())

        self.assertEqual(delta["prefixed"], 0)
        self.assertEqual(delta["bytes_sent"], delta["bytes_full"])


if __name__ == "__main__":
    unittest.main()
