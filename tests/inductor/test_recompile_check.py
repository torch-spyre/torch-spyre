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

"""The recompile check: recompiles that break a declared range are reported.

By default only a recompile whose input has one of its declared sizes is
reported; ``all`` mode reports every recompile of user code. torch-spyre's own
compiles are never reported. The CPU tests compile with ``backend="eager"`` and
stand in for a declaration by patching ``_declaration_of``. The device tests
need a Spyre device and the ``dynamic=`` reservation of #5179.
"""

import os
import unittest
import warnings
from unittest import mock

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import recompile_check
from torch_spyre._inductor.recompile_check import (
    MODE_ENV,
    RecompileError,
    RecompileWarning,
    compile_ledger,
    install_recompile_check,
    reset_compile_ledger,
)
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME

COLS = 8
SPEC = {0: {"min": 128, "max": 512, "granularity": 64}}


def _reports(caught):
    return [str(w.message) for w in caught if issubclass(w.category, RecompileWarning)]


def _run(fn, *inputs):
    """Call ``fn`` on each input in turn, recording recompile warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for x in inputs:
            fn(x)
    return _reports(caught)


def _mode(mode):
    return mock.patch.dict(os.environ, {MODE_ENV: mode})


def _declared_on_cpu():
    """Treat every 2-D CPU tensor as declared with ``SPEC``."""
    return mock.patch.object(
        recompile_check,
        "_declaration_of",
        lambda t: SPEC if t.dim() == 2 else None,
    )


class _Base(unittest.TestCase):
    def setUp(self):
        install_recompile_check()
        torch._dynamo.reset()
        reset_compile_ledger()


class TestRecompileCheckCPU(_Base):
    @staticmethod
    def _block():
        @torch.compile(backend="eager")
        def block(x):
            return x.relu() + 1

        return block

    def _sizes(self, *sizes):
        return [torch.randn(n, COLS) for n in sizes]

    def test_undeclared_recompile_is_silent_by_default(self):
        self.assertEqual(_run(self._block(), *self._sizes(128, 192)), [])

    def test_declared_recompile_at_a_promised_size_warns(self):
        with _declared_on_cpu():
            reports = _run(self._block(), *self._sizes(128, 192))
        self.assertEqual(len(reports), 1)
        self.assertIn("recompiled (compile 2)", reports[0])
        self.assertIn("Declared range broken: input 'x' dim 0 is 192", reports[0])
        self.assertIn("expected 128, actual 192", reports[0])

    def test_declared_recompile_outside_promised_sizes_is_silent(self):
        # 100 is not one of the declared sizes; the declaration refuses it
        # elsewhere, and it is not a broken promise.
        with _declared_on_cpu():
            self.assertEqual(_run(self._block(), *self._sizes(128, 100)), [])

    def test_error_mode_stops_a_broken_promise(self):
        block = self._block()
        with _declared_on_cpu(), _mode("error"):
            block(torch.randn(128, COLS))
            with self.assertRaisesRegex(RecompileError, "Declared range broken"):
                block(torch.randn(192, COLS))

    def test_error_mode_lets_undeclared_recompiles_through(self):
        with _mode("error"):
            self.assertEqual(_run(self._block(), *self._sizes(128, 192)), [])

    def test_all_mode_reports_any_recompile_with_the_reason(self):
        with _mode("all"):
            reports = _run(self._block(), *self._sizes(128, 192))
        self.assertEqual(len(reports), 1)
        self.assertIn("block", reports[0])
        self.assertIn("recompiled (compile 2)", reports[0])
        self.assertIn("expected 128, actual 192", reports[0])
        self.assertNotIn("Declared range broken", reports[0])

    def test_cached_call_is_not_reported(self):
        with _mode("all"):
            self.assertEqual(_run(self._block(), *self._sizes(128, 128)), [])

    def test_off_mode_reports_nothing(self):
        with _declared_on_cpu(), _mode("off"):
            self.assertEqual(_run(self._block(), *self._sizes(128, 192)), [])

    def test_ledger_counts_and_times_user_compiles(self):
        _run(self._block(), *self._sizes(128, 192))
        rows = [row for name, row in compile_ledger().items() if "block" in name]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["compiles"], 2)
        self.assertGreater(rows[0]["seconds"], 0.0)

    def test_torch_and_torch_spyre_files_are_internal(self):
        eager_py = os.path.join(recompile_check._SPYRE_DIR, "ops", "eager.py")
        torch_py = os.path.join(recompile_check._TORCH_DIR, "_compile.py")
        self.assertTrue(recompile_check._is_internal(eager_py))
        self.assertTrue(recompile_check._is_internal(torch_py))
        self.assertFalse(recompile_check._is_internal(__file__))

    def test_install_is_idempotent(self):
        import torch._dynamo.eval_frame as ef

        hook = ef._callback_from_stance
        install_recompile_check()
        self.assertIs(ef._callback_from_stance, hook)

    def test_graph_break_reports_both_pieces(self):
        @torch.compile(backend="eager")
        def broken(x):
            y = x.relu()
            torch._dynamo.graph_break()
            return y * 3

        with _mode("all"):
            reports = _run(broken, *self._sizes(128, 192))
        self.assertEqual(len(reports), 2)
        self.assertTrue(any("resume_in_broken" in r for r in reports))

    def test_graph_break_inside_torch_spyre_code_still_reports_user_pieces(self):
        # A helper whose code lives in the torch-spyre package, as
        # for_each_tile does. Dynamo resumes in the user's frame after the
        # break, so the user's pieces are still checked; only the helper's own
        # pieces are skipped.
        src = (
            "import torch\n"
            "def spyre_helper(x):\n"
            "    y = x * 2\n"
            "    torch._dynamo.graph_break()\n"
            "    return y + 1\n"
        )
        helper_file = os.path.join(recompile_check._SPYRE_DIR, "_test_helper.py")
        namespace = {}
        exec(compile(src, helper_file, "exec"), namespace)  # noqa: S102
        spyre_helper = namespace["spyre_helper"]

        @torch.compile(backend="eager")
        def calls_helper(x):
            return spyre_helper(x.relu()) * 3

        with _mode("all"):
            reports = _run(calls_helper, *self._sizes(128, 192))
        self.assertEqual(len(reports), 2)
        self.assertFalse(any("_test_helper.py" in r for r in reports))
        self.assertTrue(any("resume_in_calls_helper" in r for r in reports))


def _region(branch):
    """A tiled region over a declared dim; ``branch`` adds a size branch."""

    def region(x):
        rows = x.shape[0]
        torch._check(rows >= 128)
        torch._check(rows <= 512)
        torch._check(rows % 64 == 0)
        scale = (2.0 if rows > 256 else 1.0) if branch else 1.0

        def body(_carry, tiles):
            (tile,) = tiles
            return None, (tile * scale).relu()

        _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=64, out_dim=0)
        return out

    return region


def _declared(*sizes):
    return [
        torch.randn(n, 64, dtype=torch.float16).to(DEVICE_NAME, dynamic=SPEC)
        for n in sizes
    ]


class TestRecompileCheckOnDevice(_Base):
    def test_declared_region_serves_every_size_without_a_report(self):
        # The promise kept: one program for 128, 192, 256 and 320.
        region = torch.compile(_region(branch=False), fullgraph=True)
        self.assertEqual(_run(region, *_declared(128, 192, 256, 320)), [])

    def test_size_branch_in_a_declared_region_is_reported(self):
        # rows > 256 splits the range, so 320 recompiles: a broken promise.
        region = torch.compile(_region(branch=True), fullgraph=True)
        reports = _run(region, *_declared(128, 320))
        self.assertEqual(len(reports), 1)
        self.assertIn("Declared range broken", reports[0])
        self.assertIn("is 320", reports[0])

    def test_eager_ops_at_new_sizes_are_not_reported(self):
        # Each eager op is a small per-op compile that recompiles per size by
        # design. Even in all mode they must not be reported.
        with _mode("all"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for n in (128, 192):
                x = torch.randn(n, 64, dtype=torch.float16).to(DEVICE_NAME)
                x.clone()
                x + x
        self.assertEqual(_reports(caught), [])

    def test_lost_declaration_recompile_is_reported_in_all_mode(self):
        # A view drops the declaration, so the next block sees an ordinary
        # tensor. Under dynamic=None, the supported setting, its second size
        # is an automatic-dynamic recompile. Only all mode reports it, since
        # the input no longer carries a declaration. Spyre may then refuse
        # that compile (an unbounded size outside a loop); the report is made
        # before the compile either way.
        @torch.compile(dynamic=None)
        def block(h):
            return h * 2.0

        first, second = (x.view(-1, 64) for x in _declared(128, 192))
        with _mode("all"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            block(first)
            try:
                block(second)
            except Exception as e:  # noqa: BLE001
                if "kDimension" not in str(e):
                    raise
        reports = _reports(caught)
        self.assertEqual(len(reports), 1)
        self.assertIn("block", reports[0])

    def test_torch_spyre_internal_compiles_are_not_reported(self):
        # Custom ops and decompositions that torch-spyre compiles behind the
        # user's back, each at two sizes. They recompile per size by design,
        # so even all mode must not report them.
        w = torch.ones(64, dtype=torch.float16).to(DEVICE_NAME)
        paths = {
            "copy_from_d2d": lambda x: torch.empty_like(x).copy_(x),
            "to_dtype_d2d": lambda x: x.to(torch.float32),
            "rms_norm": lambda x: torch.ops.spyre.rms_norm(x, [64], w, 1e-5),
            "decomposition (sign)": lambda x: x.sign(),
        }
        for name, op in paths.items():
            with self.subTest(name):
                xs = [
                    torch.randn(n, 64, dtype=torch.float16).to(DEVICE_NAME)
                    for n in (128, 192)
                ]
                torch._dynamo.reset()
                try:
                    with _mode("off"):
                        for x in xs:
                            op(x)
                except Exception as e:  # noqa: BLE001
                    self.skipTest(f"{name} does not run here: {type(e).__name__}")
                torch._dynamo.reset()
                with _mode("all"):
                    reports = _run(op, *xs)
                self.assertEqual(reports, [])


if __name__ == "__main__":
    unittest.main()
