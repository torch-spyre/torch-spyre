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

# Owner(s): ["module: inductor"]

"""Tests that PyTorch's compiler bisector leaves Spyre's nested compiles alone.

A Spyre eager op is a compiled kernel. ``CompilerBisector`` overrides the
backend of every ``torch.compile`` call and counts every subsystem query, so
without an exemption it compiles those kernels with ``backend="eager"`` (which
re-enters the kernel) and blames ``eager`` for a program that is always correct.
See ``torch_spyre/ops/bisect_exemption.py``.

``TestSuspendedQueries`` and ``TestNestedCompileSitesAreExempt`` are CPU-only.
``TestBisectorOnDevice`` runs the bisector end to end and needs a card.
"""

import shutil
import tempfile
import threading
import unittest
from unittest import mock

import torch
import torch._dynamo
from torch._inductor import compiler_bisector
from torch._inductor.compiler_bisector import BisectionResult, CompilerBisector

import torch_spyre  # noqa: F401
from torch_spyre.ops import bisect_exemption, eager
from torch_spyre.ops.bisect_exemption import is_suspended, suspended
from torch_spyre.ops.eager import compile_once

# NB: plain unittest, not torch.testing._internal.common_utils.run_tests, which
# seeds the Spyre device and would turn the CPU-only classes into device tests.

_BACKEND = "inductor"
_SUBSYSTEM = "lowerings"


class _ActiveBisection:
    """Put the bisector in a chosen state, as ``do_bisect`` would mid-search."""

    def __init__(self, run_state: str):
        self.run_state = run_state

    def __enter__(self):
        self._saved = (
            CompilerBisector.bisection_enabled,
            CompilerBisector.in_process_cache,
        )
        self._dir = tempfile.mkdtemp()
        CompilerBisector.in_process_cache = self._dir
        CompilerBisector.bisection_enabled = True
        CompilerBisector.reset_counters()
        CompilerBisector.update_bisect_status(_BACKEND, _SUBSYSTEM)
        subsystem = CompilerBisector.get_subsystem_object(_BACKEND, _SUBSYSTEM)
        CompilerBisector.update_run_state(_BACKEND, subsystem, self.run_state)
        return self

    def __exit__(self, *exc):
        CompilerBisector.reset_counters()
        CompilerBisector.bisection_enabled, CompilerBisector.in_process_cache = (
            self._saved
        )
        shutil.rmtree(self._dir, ignore_errors=True)


def _applications_counted() -> int:
    return compiler_bisector.subsystem_call_counter[_SUBSYSTEM]


class TestSuspendedQueries(unittest.TestCase):
    def test_backend_override_is_hidden_inside_and_restored_after(self):
        with _ActiveBisection("test_disable"):
            self.assertEqual(CompilerBisector.get_backend(), _BACKEND)
            with suspended():
                self.assertIsNone(CompilerBisector.get_backend())
            self.assertEqual(CompilerBisector.get_backend(), _BACKEND)

    def test_disabled_subsystem_is_enabled_inside(self):
        with _ActiveBisection("test_disable"):
            self.assertTrue(CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM))
            with suspended():
                self.assertFalse(
                    CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM)
                )
            self.assertTrue(CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM))

    def test_applications_inside_are_not_counted(self):
        # ``find_max_bounds`` is the run that counts a subsystem's applications
        # to size the binary search. A nested kernel's lowerings must not
        # inflate that count.
        with _ActiveBisection("find_max_bounds"):
            CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM)
            self.assertEqual(_applications_counted(), 1)
            with suspended():
                for _ in range(5):
                    CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM)
            self.assertEqual(_applications_counted(), 1)
            CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM)
            self.assertEqual(_applications_counted(), 2)

    def test_config_change_is_hidden_inside(self):
        with _ActiveBisection("test_disable"):
            CompilerBisector.set_config_values(
                _BACKEND, _SUBSYSTEM, {"layout_optimization": False}
            )
            self.assertEqual(
                CompilerBisector.get_config_change("inductor"),
                {"layout_optimization": False},
            )
            with suspended():
                self.assertIsNone(CompilerBisector.get_config_change("inductor"))

    def test_regions_nest(self):
        with _ActiveBisection("test_disable"):
            with suspended():
                with suspended():
                    self.assertTrue(is_suspended())
                # Leaving the inner region must not end the outer one.
                self.assertTrue(is_suspended())
                self.assertIsNone(CompilerBisector.get_backend())
            self.assertFalse(is_suspended())

    def test_region_ends_when_the_body_raises(self):
        with self.assertRaises(ValueError):
            with suspended():
                raise ValueError
        self.assertFalse(is_suspended())

    def test_region_is_per_thread(self):
        # A compile on another thread is a different program; it must still be
        # bisected while this thread runs a nested kernel.
        seen = {}
        with _ActiveBisection("test_disable"):
            with suspended():
                thread = threading.Thread(
                    target=lambda: seen.update(
                        suspended=is_suspended(),
                        backend=CompilerBisector.get_backend(),
                    )
                )
                thread.start()
                thread.join()
        self.assertEqual(seen, {"suspended": False, "backend": _BACKEND})

    def test_queries_are_unchanged_when_no_bisection_is_active(self):
        with suspended():
            pass  # make sure the wrappers are installed
        self.assertFalse(CompilerBisector.bisection_enabled)
        self.assertFalse(CompilerBisector.disable_subsystem(_BACKEND, _SUBSYSTEM))

    def test_installing_twice_wraps_once(self):
        with suspended():
            pass
        wrapped = CompilerBisector.__dict__["get_backend"]
        bisect_exemption._install()
        self.assertIs(CompilerBisector.__dict__["get_backend"], wrapped)


class TestNestedCompileSitesAreExempt(unittest.TestCase):
    def test_compile_once_covers_both_the_compile_call_and_the_kernel_run(self):
        # The backend override is read when ``torch.compile`` is called; the
        # subsystem counters are read when the kernel first runs. Both must be
        # inside the region.
        seen = {}

        def fake_compile(fn, **kwargs):
            seen["at_compile_call"] = is_suspended()
            return fn

        @compile_once(torch.ops.aten.add.Tensor)
        def kernel(x, y, compiled=None):
            seen["at_kernel_run"] = is_suspended()
            return compiled(x, y)

        with mock.patch.object(torch, "compile", fake_compile):
            out = kernel(torch.ones(2), torch.ones(2))

        torch.testing.assert_close(out, torch.full((2,), 2.0))
        self.assertEqual(seen, {"at_compile_call": True, "at_kernel_run": True})
        self.assertFalse(is_suspended())

    def test_view_rebuilder_covers_both_the_compile_call_and_the_kernel_run(self):
        # The on-device view copy compiles its own kernel, outside
        # ``compile_once``. A CPU view stands in for the Spyre one: only the
        # region around the compile and the run is under test.
        seen = {}

        def fake_compile(fn, **kwargs):
            seen["at_compile_call"] = is_suspended()

            def run(base):
                seen["at_kernel_run"] = is_suspended()
                return fn(base)

            return run

        base = torch.arange(6.0).reshape(2, 3)
        view = base[1]
        with (
            mock.patch.dict(eager._view_rebuilders, clear=True),
            mock.patch.object(torch, "compile", fake_compile),
        ):
            out = eager._rebuild_view_on_device(view)

        torch.testing.assert_close(out, view)
        self.assertEqual(seen, {"at_compile_call": True, "at_kernel_run": True})
        self.assertFalse(is_suspended())


def _relu_sum(x, y):
    return torch.relu(x + y * 2.0).sum(dim=-1)


class TestBisectorOnDevice(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(64, 128, dtype=torch.float16)
        self.y = torch.randn(64, 128, dtype=torch.float16)
        self.expected = _relu_sum(self.x, self.y)

    def tearDown(self):
        torch._dynamo.reset()

    def _matches_cpu(self) -> bool:
        torch._dynamo.reset()
        try:
            out = torch.compile(_relu_sum)(self.x.to("spyre"), self.y.to("spyre"))
        except Exception:
            return False
        # Spyre fp16 is not IEEE half, and this sums 128 values.
        return torch.allclose(
            out.cpu().float(), self.expected.float(), atol=1.0, rtol=0.1
        )

    def test_a_correct_program_is_not_blamed(self):
        # Without the exemption this is BisectionResult(backend='eager'): the
        # per-op kernels are compiled with the eager backend and re-enter
        # themselves.
        self.assertIsNone(CompilerBisector.do_bisect(self._matches_cpu))

    def test_a_planted_graph_level_defect_is_found(self):
        # The exemption must not blind the bisector. Plant a defect in the
        # lowering of relu that only the program under test sees: the per-op
        # kernels, compiled inside the exemption, keep the correct lowering.
        # That is the shape of a real graph-level defect, and it makes
        # "disable this lowering" (fall back to the eager kernel) the fix.
        from torch._inductor import lowering

        relu = torch.ops.aten.relu.default
        correct_relu = lowering.lowerings[relu]

        def defective_relu(x):
            out = correct_relu(x)
            if is_suspended():
                return out
            return lowering.lowerings[torch.ops.aten.neg.default](out)

        with mock.patch.dict(lowering.lowerings, {relu: defective_relu}):
            result = CompilerBisector.do_bisect(self._matches_cpu)

        self.assertIsInstance(result, BisectionResult)
        self.assertEqual((result.backend, result.subsystem), ("inductor", "lowerings"))
        self.assertEqual(result.debug_info, "relu")


if __name__ == "__main__":
    unittest.main()
