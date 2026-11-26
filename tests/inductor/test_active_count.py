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

"""for_each_tile(active_count=...): the keyword's checks, and the marker that
reads the count once in the loop body so the capture keeps it.

No device needed.
"""

import unittest

import torch

from torch_spyre._inductor.wsr.for_each_tile import for_each_tile


class TestTheSpelling(unittest.TestCase):
    """The checks for_each_tile makes before any loop is built (plan 2.1)."""

    @staticmethod
    def _call(count, x=None, **kw):
        x = torch.ones(8, 4) if x is None else x

        def body(_, tiles):
            return None, tiles[0] + 1

        return for_each_tile(
            body, (x,), dims=(0,), tile_size=2, out_dim=0, active_count=count, **kw
        )

    def test_a_non_tensor_count_is_refused(self):
        with self.assertRaisesRegex(TypeError, "active_count must be a tensor"):
            self._call(3)

    def test_a_non_fp32_count_is_refused(self):
        with self.assertRaisesRegex(TypeError, "must be float32"):
            self._call(torch.ones(1, 32, dtype=torch.float16))

    def test_a_count_on_another_device_is_refused(self):
        with self.assertRaisesRegex(ValueError, "on the operands' device"):
            self._call(torch.ones(1, 32, device="meta"))

    def test_a_count_the_loop_tiles_is_refused(self):
        x = torch.ones(8, 4)
        with self.assertRaisesRegex(ValueError, "also an operand this loop tiles"):
            self._call(x, x=x)

    def test_reverse_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reverse=True"):
            self._call(torch.ones(1, 32), reverse=True)

    def test_eager_runs_every_trip(self):
        """Off the device the loop is the static one: all 4 tiles, whatever
        the count says, so trips past the count must be harmless there."""
        _, out = self._call(torch.full((1, 32), 1.0))
        torch.testing.assert_close(out, torch.full((8, 4), 2.0))


def _doubled(x, cnt):
    def body(_, tiles):
        return None, tiles[0] * 2

    return for_each_tile(
        body, (x,), dims=(0,), tile_size=2, out_dim=0, active_count=cnt
    )[1]


class TestTheKeywordReachesTheGraph(unittest.TestCase):
    """Plan 2.2: the lifted count is read once in the body (the marker);
    lifted but unread, the capture would drop it."""

    def test_the_marker_is_in_the_captured_body(self):
        graphs = []

        def backend(gm, _inputs):
            graphs.append(gm)
            return gm

        torch._dynamo.reset()
        x, cnt = torch.ones(8, 4), torch.full((1, 32), 2.0)
        out = torch.compile(_doubled, backend=backend, fullgraph=True)(x, cnt)
        torch.testing.assert_close(out, x * 2)
        text = "\n".join(
            m.print_readable(print_output=False)
            for gm in graphs
            for m in gm.modules()
            if isinstance(m, torch.fx.GraphModule)
        )
        self.assertIn("active_count_marker", text)

    def test_cpu_inductor_matches_eager(self):
        """Plan 2.7: on CPU the loop runs every trip, compiled or not."""
        torch._dynamo.reset()
        x, cnt = torch.randn(8, 4), torch.full((1, 32), 1.0)
        out = torch.compile(_doubled, fullgraph=True)(x, cnt)
        torch.testing.assert_close(out, x * 2)


class TestTheMarker(unittest.TestCase):
    def test_eagerly_it_reads_the_first_value(self):
        count = torch.tensor([[5.0, 9.0], [7.0, 1.0]])
        out = torch.ops.spyre.active_count_marker(count, 16)
        torch.testing.assert_close(out, torch.tensor([5.0]))

    def test_fx_keeps_it_although_nothing_reads_it(self):
        op = torch.ops.spyre.active_count_marker.default
        self.assertIn(op, torch.fx.node._side_effectful_functions)


if __name__ == "__main__":
    unittest.main()
