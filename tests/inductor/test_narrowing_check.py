# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""The narrowing check: guards that split a declared range, at compile 1.

The helper tests run anywhere. The device tests need a Spyre device and the
``dynamic=`` reservation of #5179.
"""

import os
import unittest
import warnings
from unittest import mock

import sympy
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import narrowing_check
from torch_spyre._inductor.narrowing_check import (
    MODE_ENV,
    DeclaredRangeNarrowedError,
    rejected_sizes,
)
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.ops.warning_types import DeclaredRangeNarrowedWarning

TILE = 64
MAX_ROWS = 512
COLS = 64
ROWS = 192


class _UnevaluableGuard:
    """A guard whose substitution raises, as Min(256, s) <= s does."""

    def __init__(self, sym):
        self.free_symbols = {sym}

    def xreplace(self, _rule):
        raise AttributeError("cannot evaluate")


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.s = sympy.Symbol("s0", integer=True, positive=True)
        self.sym_legal = {"s0": list(range(128, MAX_ROWS + 1, TILE))}

    def test_range_checks_reject_nothing(self):
        divisible = sympy.Eq(sympy.Mod(self.s, 64), 0)
        for expr in (self.s >= 128, self.s <= 512, divisible):
            self.assertEqual(rejected_sizes(expr, self.sym_legal), [])

    def test_branch_guard_rejects_upper_sizes(self):
        self.assertEqual(
            rejected_sizes(self.s <= 256, self.sym_legal), [320, 384, 448, 512]
        )

    def test_unrelated_and_two_symbol_guards_are_skipped(self):
        other = sympy.Symbol("s1", integer=True, positive=True)
        self.assertIsNone(rejected_sizes(other <= 256, self.sym_legal))
        self.assertIsNone(rejected_sizes(sympy.Eq(self.s, other), self.sym_legal))

    def test_unevaluable_guard_is_skipped_not_raised(self):
        self.assertIsNone(rejected_sizes(_UnevaluableGuard(self.s), self.sym_legal))

    def test_unknown_mode_falls_back_to_warn(self):
        # A bad value must not fail a compile: it is logged, and warn is used.
        with (
            mock.patch.dict(os.environ, {MODE_ENV: "not-a-mode"}),
            self.assertLogs(narrowing_check.logger.name, level="WARNING") as log,
        ):
            self.assertEqual(narrowing_check._mode(), "warn")
        self.assertIn("not-a-mode", log.output[0])


def _make_region(min_rows, narrowing=None):
    """A tiled region; ``narrowing`` picks the size-dependent code to add."""

    def region(x):
        rows = x.shape[0]
        torch._check(rows >= min_rows)
        torch._check(rows <= MAX_ROWS)
        torch._check(rows % TILE == 0)
        scale = 1.0
        if narrowing == "branch":
            scale = 2.0 if rows > 256 else 1.0
        elif narrowing == "assert":
            assert rows <= 448

        def body(_carry, tiles):
            (tile,) = tiles
            return None, (tile * scale).relu()

        _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
        return out

    return region


def _declared(min_rows, rows=ROWS, cols=COLS):
    spec = {0: {"min": min_rows, "max": MAX_ROWS, "granularity": TILE}}
    return torch.randn(rows, cols, dtype=torch.float16).to(DEVICE_NAME, dynamic=spec)


def _compile_and_run(min_rows, narrowing=None):
    compiled = torch.compile(_make_region(min_rows, narrowing), fullgraph=True)
    return compiled(_declared(min_rows))


def _reports(caught):
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeclaredRangeNarrowedWarning)
    ]


class TestNarrowingCheckOnDevice(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def _run_recording(self, *args, **kwargs):
        # The caches are off because a hit on the FX graph cache adds the cached
        # graph's guards to this compile: the branch test's graph at 192 rows
        # equals the clean region's, so its "rows <= 256" guard would be
        # reported in the clean test.
        with (
            warnings.catch_warnings(record=True) as caught,
            torch._inductor.config.patch(force_disable_caches=True),
        ):
            warnings.simplefilter("always")
            out = _compile_and_run(*args, **kwargs)
        return out, _reports(caught)

    def test_clean_region_gets_no_report(self):
        out, reports = self._run_recording(min_rows=128)
        self.assertEqual(tuple(out.shape), (ROWS, COLS))
        self.assertFalse(reports, "\n".join(reports))

    def test_branch_warns_by_default_naming_the_line(self):
        out, reports = self._run_recording(min_rows=128, narrowing="branch")
        self.assertEqual(tuple(out.shape), (ROWS, COLS))
        self.assertEqual(len(reports), 1)
        self.assertIn("dim values [320, 384, 448, 512]", reports[0])
        self.assertIn("rows > 256", reports[0])

    def test_assert_warns_naming_the_line(self):
        _out, reports = self._run_recording(min_rows=128, narrowing="assert")
        self.assertEqual(len(reports), 1)
        self.assertIn("dim values [512]", reports[0])
        self.assertIn("assert rows <= 448", reports[0])

    def test_error_mode_stops_the_compile(self):
        # Dynamo may wrap the error, so match the type's name in the message.
        expected = f"{DeclaredRangeNarrowedError.__name__}|causes a recompile"
        with mock.patch.dict(os.environ, {MODE_ENV: "error"}):
            with self.assertRaisesRegex(Exception, expected):
                _compile_and_run(min_rows=128, narrowing="branch")

    def test_off_mode_skips_the_check(self):
        with mock.patch.dict(os.environ, {MODE_ENV: "off"}):
            out, reports = self._run_recording(min_rows=128, narrowing="branch")
        self.assertEqual(tuple(out.shape), (ROWS, COLS))
        self.assertFalse(reports, "\n".join(reports))

    def test_encoder_like_region_gets_no_report(self):
        # Matmul, an RMS-style norm over the hidden axis, gelu and a residual
        # add. aten layer_norm is left out: inside a loop it needs the
        # layer-norm exemption, which is not in this tree.
        x = _declared(min_rows=128)
        w = torch.randn(COLS, COLS, dtype=torch.float16).to(DEVICE_NAME)

        def region(x, w):
            rows = x.shape[0]
            torch._check(rows >= 128)
            torch._check(rows <= MAX_ROWS)
            torch._check(rows % TILE == 0)

            def body(_carry, tiles):
                tile, weight = tiles
                h = tile @ weight
                h = h * torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + 1e-5)
                return None, torch.nn.functional.gelu(h) + tile

            _carry, out = for_each_tile(
                body, (x, w), dims=(0, None), tile_size=TILE, out_dim=0
            )
            return out

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = torch.compile(region, fullgraph=True)(x, w)
        self.assertEqual(tuple(out.shape), (ROWS, COLS))
        self.assertEqual(_reports(caught), [])


if __name__ == "__main__":
    unittest.main()
