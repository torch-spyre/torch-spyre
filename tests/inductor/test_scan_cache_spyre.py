# Copyright 2025 The Torch-Spyre Authors.
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


"""TORCH_SPYRE_CACHE_SCAN on the Spyre backend.

A cached ``for_each_tile`` graph is reused only when the tile size and the
torch-spyre config (here ``sencores``) are the same.
"""

import unittest
from unittest import mock

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.utils import fresh_cache

import torch_spyre  # noqa: F401  registers the "spyre" device
import torch_spyre._inductor.config as spyre_config
from torch_spyre._inductor import _allow_scan_graph_caching
from torch_spyre.constants import DEVICE_NAME

from for_each_tile_fixtures import COLS, ROWS, add_tiled_fn, add_tiled_reference


def _fx_counts():
    c = counters["inductor"]
    return c["fxgraph_cache_hit"], c["fxgraph_cache_miss"], c["fxgraph_cache_bypass"]


class TestScanCacheSpyre(unittest.TestCase):
    ATOL = 0.1
    RTOL = 0.1

    def _run(self, tile_size):
        torch._dynamo.reset()
        compiled = torch.compile(add_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(self.a, self.b, tile_size)
        torch.testing.assert_close(
            out.cpu().float(), self.ref, atol=self.ATOL, rtol=self.RTOL
        )
        return _fx_counts()

    def test_key_follows_tile_size_and_sencores(self):
        a = torch.randn(ROWS, COLS, dtype=torch.float16)
        b = torch.randn(ROWS, COLS, dtype=torch.float16)
        self.ref = add_tiled_reference(a.float(), b.float())
        self.a, self.b = a.to(DEVICE_NAME), b.to(DEVICE_NAME)
        with (
            mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_SCAN": "1"}),
            inductor_config.patch(
                unsafe_marked_cacheable_functions={},
                fx_graph_cache=True,
                fx_graph_remote_cache=False,
            ),
            fresh_cache(),
        ):
            _allow_scan_graph_caching()
            counters.clear()

            hit0, miss0, bypass0 = self._run(2)
            self.assertEqual((hit0, bypass0), (0, 0))
            self.assertGreater(miss0, 0)

            # Same tile size, same config: every graph is a hit.
            hit1, miss1, _ = self._run(2)
            self.assertEqual(miss1, miss0)
            self.assertGreater(hit1, hit0)

            # New tile size: a new entry, not the cached one.
            hit2, miss2, _ = self._run(4)
            self.assertGreater(miss2, miss1)

            # Same tile size, other sencores: a new entry too.
            with spyre_config.patch({"sencores": spyre_config.sencores // 2}):
                hit3, miss3, _ = self._run(2)
            self.assertGreater(miss3, miss2)
            self.assertEqual(_fx_counts()[2], 0)


if __name__ == "__main__":
    unittest.main()
