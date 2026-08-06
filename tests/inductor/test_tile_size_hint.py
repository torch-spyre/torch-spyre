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

"""spyre_hint(tile_size_per_dim=...) — declare the tile size, not the trip count.

The backend contract is a dynamic loop count over fixed-size tiles, so the hint
declares the size and the count falls out.  WSR requires every tile to be full:
the extent must already be padded to a multiple of the tile size (with
op-appropriate identity values), and a non-multiple is a loud error rather than a
short final tile.  See docs/wsr-tile-size-api-plan.md.
"""

import os
import sys

import torch

from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._inductor import config

# Inductor wraps backend Unsupported in InductorError, so assert on that and
# match the message text.
from torch._inductor.exc import InductorError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


class TestTileSizeHint(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        _pnd.reset()

    def _run(self, hint_kwargs, nrow=1024, ncol=4096):
        """y = (a + b) * c, tiled over row dim A per hint_kwargs. Returns (src, ok)."""
        from torch_spyre._inductor import spyre_hint

        a = torch.rand(nrow, ncol, dtype=torch.float16)
        b = torch.rand(nrow, ncol, dtype=torch.float16)
        c = torch.rand(nrow, ncol, dtype=torch.float16)
        ref = (a + b) * c

        _declare_tensor_dim("A", nrow)
        _declare_tensor_dim("B", ncol)
        ad, bd, cd = a.to("spyre"), b.to("spyre"), c.to("spyre")
        for t in (ad, bd, cd):
            _name_tensor_dims(t, ["A", "B"])

        def fn(x, y, z):
            with spyre_hint(**hint_kwargs):
                return (x + y) * z

        got, srcs = run_and_get_code(torch.compile(fn), ad, bd, cd)
        torch.testing.assert_close(got.cpu(), ref, equal_nan=True, atol=0.01, rtol=0.1)
        return srcs[0]

    def test_tile_size_matches_equivalent_num_tiles(self):
        """tile_size=256 on a 1024 extent must behave exactly like num_tiles=4."""
        _pnd.reset()
        torch.manual_seed(0xAFFE)
        src_count = self._run({"num_tiles_per_dim": {"A": 4}})

        torch._dynamo.reset()
        _pnd.reset()
        torch.manual_seed(0xAFFE)
        src_size = self._run({"tile_size_per_dim": {"A": 256}})

        # Same trip count in the generated LoopSpec, reached from either spelling.
        self.assertIn("LoopSpec(", src_count)
        self.assertIn("LoopSpec(", src_size)
        self.assertIn("sympify('4')", src_count)
        self.assertIn(
            "sympify('4')",
            src_size,
            "tile_size_per_dim={'A': 256} on extent 1024 must yield trip count 4",
        )

    def test_tile_size_one_tile_is_no_op(self):
        """tile_size == extent means one tile, i.e. no loop, like num_tiles=1."""
        src = self._run({"tile_size_per_dim": {"A": 1024}})
        self.assertNotIn(
            "LoopSpec(", src, "a single full-extent tile must not emit a loop"
        )

    def test_non_multiple_extent_is_untiled_by_default(self):
        """A size that does not divide leaves the dim untiled, with a warning.

        WSR needs full tiles, so a non-multiple cannot be honoured.  Default is
        graceful degradation rather than a hard error: decode has max_seqlen_q=1
        and batch=1, neither of which can be a multiple of a 64-element block, and
        those models should still compile.  config.strict_tile_size turns it into
        an error -- see the next test.
        """
        with self.assertLogs(
            "spyre.inductor.propagate_named_dims", level="WARNING"
        ) as logs:
            self._run({"tile_size_per_dim": {"A": 300}})
        blob = "\n".join(logs.output)
        self.assertIn("does not divide", blob)
        self.assertIn("full tiles", blob)
        self.assertIn("untiled", blob)

    @config.patch({"strict_tile_size": True})
    def test_non_multiple_extent_raises_under_strict(self):
        """strict_tile_size makes an infeasible tiling a hard error.

        Intended for performance CI and any model where tiling is expected to
        succeed, so a silent loss of tiling is caught rather than shipped.
        """
        with self.assertRaises(InductorError) as cm:
            self._run({"tile_size_per_dim": {"A": 300}})
        msg = str(cm.exception)
        self.assertIn("does not divide", msg)
        self.assertIn("full tiles", msg)
        self.assertIn("Pad", msg)

    def test_zero_or_negative_tile_size_raises(self):
        with self.assertRaises(InductorError) as cm:
            self._run({"tile_size_per_dim": {"A": 0}})
        self.assertIn("must be positive", str(cm.exception))

    def test_nested_tile_size_levels_on_distinct_dims(self):
        """Nested scopes on DISTINCT dims: each count comes from its own declaration.

        A is declared 1024 and tiled by 256 -> 4; B is declared 4096 and tiled by
        512 -> 8.  Deliberately different counts so the assertion can tell the two
        levels apart -- equal counts would pass even if one level were dropped or
        both resolved from the same dim.

        This also pins the independence rule: A's hint does not affect B's count,
        and neither consults the operation's iteration space.
        """
        from torch_spyre._inductor import spyre_hint

        _pnd.reset()
        torch.manual_seed(0xAFFE)
        nrow, ncol = 1024, 4096
        a = torch.rand(nrow, ncol, dtype=torch.float16)
        b = torch.rand(nrow, ncol, dtype=torch.float16)
        ref = a + b

        _declare_tensor_dim("A", nrow)
        _declare_tensor_dim("B", ncol)
        ad, bd = a.to("spyre"), b.to("spyre")
        for t in (ad, bd):
            _name_tensor_dims(t, ["A", "B"])

        def fn(x, y):
            with spyre_hint(tile_size_per_dim={"A": 256}):
                with spyre_hint(tile_size_per_dim={"B": 512}):
                    return x + y

        got, srcs = run_and_get_code(torch.compile(fn), ad, bd)
        torch.testing.assert_close(got.cpu(), ref, equal_nan=True, atol=0.01, rtol=0.1)
        src = srcs[0]
        self.assertIn("LoopSpec(", src)
        self.assertIn(
            "sympify('4')", src, "A declared 1024, tiled by 256 -> trip count 4"
        )
        self.assertIn(
            "sympify('8')", src, "B declared 4096, tiled by 512 -> trip count 8"
        )
