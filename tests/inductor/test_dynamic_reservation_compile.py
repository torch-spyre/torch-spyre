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

"""A compiled function over a ``tensor.to("spyre", dynamic=...)`` tensor.

What the reservation guarantees, and what these tests assert:

* Resizing a reserved tensor inside [min, max] never changes its
  SpyreTensorLayout, so a compiled graph is not recompiled when the size
  changes: one graph, one frame, one compile.
* At the size the graph was compiled for, the result is right and nothing is
  read from the reserved space above the logical size (poisoned tail).

Making one binary produce correct results across sizes needs symbolic loop bounds
in the compiler.
"""

import unittest

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor.utils import fresh_cache

import torch_spyre  # noqa: F401
from torch_spyre._C import get_spyre_tensor_layout

DEVICE = torch.device("spyre")

COLS = 1024
MAX_ROWS = 630
SPEC = {"min": 70, "max": MAX_ROWS, "granularity": 70}
# The first entry is the size the graph is compiled for. It is not the max, so
# the later sizes include both growing and shrinking relative to it.
SIZES = [140, 630, 280, 560, 70]

# fp16 results of ops on values in [0, 1): one ulp is about 1e-3 near 1, and
# sum(dim=0) accumulates 140 rows (measured relative error 3e-3). The reference
# is computed in fp32 so the bound reflects only device rounding.
ATOL = 1e-2
RTOL = 2e-2


def _sum_dim0(a):
    return a.sum(dim=0)


class TestReservedTensorCompile(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0xAFFE)
        torch._dynamo.reset()
        counters.clear()

    def test_resize_does_not_recompile(self) -> None:
        x = torch.rand(MAX_ROWS, COLS, dtype=torch.float16)
        y = x.to(DEVICE, dynamic={0: SPEC})
        layout = str(get_spyre_tensor_layout(y))
        compiled = torch.compile(F.gelu)

        with fresh_cache():
            for n in SIZES:
                if y.size(0) != n:
                    y.resize_((n, COLS))
                out = compiled(y)

                self.assertEqual(tuple(out.shape), (n, COLS))
                self.assertEqual(
                    str(get_spyre_tensor_layout(y)),
                    layout,
                    f"layout changed at {n} rows",
                )

        self.assertEqual(counters["stats"]["unique_graphs"], 1)
        self.assertEqual(counters["frames"]["ok"], 1)
        self.assertEqual(counters["frames"]["total"], 1)
        # One compile, and nothing served from an older cache entry.
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    def test_compile_size_result_ignores_poisoned_tail(self) -> None:
        # Reserve at the ceiling with NaN above the rows that matter, shrink to
        # the compile-time size, then run. The space above the logical size is
        # stale by contract, so nothing computed at this size may depend on it:
        # a kernel that read it would surface the NaNs.
        rows = SIZES[0]
        cases = {
            "gelu": (F.gelu, lambda a: F.gelu(a.float())),
            "sum(dim=0)": (_sum_dim0, lambda a: a.float().sum(dim=0)),
        }
        for name, (fn, ref_fn) in cases.items():
            with self.subTest(op=name):
                torch._dynamo.reset()
                x = torch.rand(MAX_ROWS, COLS, dtype=torch.float16)
                x[rows:] = float("nan")
                y = x.to(DEVICE, dynamic={0: SPEC})
                y.resize_((rows, COLS))

                with fresh_cache():
                    out = torch.compile(fn)(y).cpu().float()

                self.assertTrue(torch.isfinite(out).all())
                torch.testing.assert_close(out, ref_fn(x[:rows]), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
