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

"""for_each_tile(active_count=...) on the Spyre device: the loop runs as many
trips as the count in device memory says, at least one and at most the static
trip count, with one compile for every count.

Each tile adds one to its rows, so a row that is ``x + 1`` was run and a row
left at zero was not: the output itself says how many trips ran.
"""

import os
import tempfile
import unittest

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor.wsr.for_each_tile import Slice, for_each_tile

G, C = 64, 32  # rows per tile, columns (one FP32 stick)


def _tiles(x, cnt):
    def tile(_, t):
        return None, t[0] + 1.0

    return for_each_tile(
        tile, (x,), dims=(0,), tile_size=G, out_dim=0, active_count=cnt
    )[1]


def _flat(x, cnt):
    # The count is written in the same bundle as the loop (abs): a count
    # passed straight from the host is a kernel input, not supported yet.
    return _tiles(x, cnt.abs())


def _nested(trips):
    def fn(x, cnt):
        def outer(_, ops):
            x_o, c_o = ops
            return None, _tiles(x_o, c_o)

        return for_each_tile(
            outer,
            (x, cnt.abs()),
            dims=(0, Slice(0, 1)),
            tile_size=trips * G,
            out_dim=0,
        )[1]

    return fn


def _count_rows(counts):
    rows = torch.tensor(counts, dtype=torch.float32).view(-1, 1)
    return rows.expand(len(counts), C).contiguous()


def _trips_run(out, x, groups, trips):
    """Per group, how many tiles were run (in order from the first)."""
    done = ((out - (x + 1)).abs().amax(1) < 1e-3).view(groups, trips, G).all(2)
    for g in range(groups):
        ran = int(done[g].sum())
        assert done[g, :ran].all(), f"group {g}: the trips run are not the first"
    return done.sum(1).tolist()


class TestActiveCountOnDevice(unittest.TestCase):
    def setUp(self):
        self._cache = tempfile.TemporaryDirectory(prefix="active_count_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = self._cache.name
        torch._dynamo.reset()

    def tearDown(self):
        self._cache.cleanup()

    def _check(self, fn, counts, trips):
        compiled = torch.compile(fn, fullgraph=True)
        groups = len(counts)
        x = torch.randn(groups * trips * G, C)
        cnt = _count_rows(counts)
        out = compiled(x.to(DEVICE_NAME), cnt.to(DEVICE_NAME)).cpu()
        want = [min(max(c, 1), trips) for c in counts]
        self.assertEqual(_trips_run(out, x, groups, trips), want, f"counts {counts}")

    def test_flat_loop_runs_the_count(self):
        """Clamped to [1, 16]: 17 runs 16 trips and 0 runs one."""
        for count in (1, 3, 16, 17, 0):
            with self.subTest(count=count):
                self._check(_flat, [count], trips=16)

    def test_nested_loop_reads_one_count_per_outer_trip(self):
        self._check(_nested(32), [1, 4, 32], trips=32)


if __name__ == "__main__":
    unittest.main()
