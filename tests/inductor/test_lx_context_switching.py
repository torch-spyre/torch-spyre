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

"""Regression tests for LxContextSwitchingPass (scratchpad/lx_context_switching.py).

TestLxContextSwitching mirrors repro_lx_extern_clobber.py: an opaque custom
op with no native Spyre lowering (so Inductor emits a FallbackKernel) whose
eager body conditionally launches a *separate* compiled Spyre program. The
condition is a module-level global, not a Tensor argument, so the outer
program is traced/compiled exactly once and can never know at compile time
whether the nested launch will happen on a given call -- this is what makes
FallbackKernel dispatch opaque, and is the failure mode PR3683 and this pass
both address (from opposite ends: refuse residency vs. protect it).

The nested launch's own result is discarded; its only possible observable
effect is corrupting a buffer this graph still needs LX-resident at that
point. So correctness here means: the same compiled artifact, run once with
the nested launch suppressed and once with it firing, produces identical
output -- diff == 0, regardless of config.enable_lx_context_switching.

TestMarkLxSafe/TestIsCpuOnlyFallback cover the Layer-1 helpers directly and
need no compile or device.
"""

import unittest
from unittest.mock import patch

import torch

from torch._inductor import config as t_inductor_config

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor.scratchpad.lx_context_switching import (
    _is_cpu_only_fallback,
    mark_lx_safe,
)
from torch_spyre.ops.fallbacks import fallback_ops


def _pr3683_guard_disabled():
    """Force PR3683's blanket "never pin a buffer to LX across an extern
    kernel" guard off, independent of config.enable_lx_context_switching.

    Only for TestLxContextSwitching.test_neither_mechanism_reproduces_the_bug,
    to demonstrate the underlying failure mode when *neither* mechanism is
    active. Production code never runs with the guard forced off this way --
    the guard is otherwise only skipped via the config flag, which
    simultaneously turns LxContextSwitchingPass on (allocator.py's
    select_allocator()).

    Imports scratchpad.allocator lazily so importing this test module (and
    collecting TestMarkLxSafe/TestIsCpuOnlyFallback, which need neither
    allocator nor a device) doesn't require it to succeed.
    """
    from torch_spyre._inductor.scratchpad import allocator as ts_allocator

    return patch.object(
        ts_allocator, "_extern_kernel_in_live_range", lambda graph, uses: False
    )


DEVICE = "spyre"
DTYPE = torch.float16


def _ns_has_op(ns: str, op: str) -> bool:
    return hasattr(getattr(torch.ops, ns, None), op)


# See module docstring. Use FRAGMENT + a re-registration guard so this module
# is safe to import more than once (the test harness re-imports test files
# under different module names during analysis + execution).
_LIB = torch.library.Library("test_lx_context_switching", "FRAGMENT")
_launch = False
_inner_fn = None
_inner_arg = None

if not _ns_has_op("test_lx_context_switching", "opaque_clone"):
    _LIB.define("opaque_clone(Tensor x) -> Tensor")

    def _opaque_clone_impl(x: torch.Tensor) -> torch.Tensor:
        if _launch:
            _inner_fn(_inner_arg)  # discarded; only the nested launch matters
        return x.clone()

    _LIB.impl(
        "opaque_clone", _opaque_clone_impl, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB._register_fake("opaque_clone", lambda x: torch.empty_like(x))


def _model(x, ws, read_count=1):
    h = x
    for w in ws:
        r = h * 1.5 + 0.25  # live across the opaque op, at risk if LX-resident
        o = torch.ops.test_lx_context_switching.opaque_clone(h @ w)
        # read_count separate reads of r, not r*read_count -- calculate_liveness
        # (scratchpad/utils.py) counts distinct *operations* touching a buffer,
        # so folding read_count into one op (e.g. scaling r) would still count
        # as a single read; each += below is its own op in graph.operations,
        # straddling the opaque call same as the read_count=1 case.
        h = o
        for _ in range(read_count):
            h = h + r
    return h


class TestLxContextSwitching(unittest.TestCase):
    def setUp(self):
        global _launch
        _launch = False
        torch.manual_seed(0xC0FFEE)
        torch.compiler.reset()
        # This op/model pair is reused across every test method (and across
        # manual debugging runs), so the on-disk FxGraphCache can serve a
        # stale artifact compiled under a *different*
        # config.enable_lx_context_switching value -- that value isn't part
        # of upstream Inductor's own cache-key inputs. Matches
        # test_scratchpad_use.py's BaseTestScratchpadUsage.setUp.
        self._caches_disabled = t_inductor_config.patch("force_disable_caches", True)
        self._caches_disabled.__enter__()

    def tearDown(self):
        global _launch
        _launch = False
        self._caches_disabled.__exit__(None, None, None)
        torch.compiler.reset()

    def _run_launch_diff(self, layers: int, n: int, read_count: int = 1) -> float:
        global _launch, _inner_fn, _inner_arg

        _inner_fn = torch.compile(lambda t: t * 2.0 + 1.0, dynamic=False)
        # 256x256, not 64x64: the nested launch's own LX planning starts fresh
        # from address 0, independent of the outer graph's placement, so `r`
        # is only actually at risk if the nested launch's LX footprint is big
        # enough to reach r's offset. Empirically, 64x64 and 128x128 never
        # overlap it (this canary silently stopped reproducing after #5fdd1f08
        # made ExternKernel-adjacent buffers ineligible for LX elsewhere in
        # the graph, shifting r's offset); 256x256 reliably does.
        _inner_arg = torch.ones(256, 256, dtype=DTYPE, device=DEVICE)
        _launch = True
        _inner_fn(_inner_arg)  # compile + warm before the outer program runs
        _launch = False

        x = torch.randn(n, n, dtype=DTYPE, device=DEVICE)
        ws = [
            torch.randn(n, n, dtype=DTYPE, device=DEVICE) / (n**0.5)
            for _ in range(layers)
        ]

        compiled = torch.compile(_model, dynamic=False)
        without = compiled(x, ws, read_count).to("cpu").float()
        _launch = True
        with_launch = compiled(x, ws, read_count).to("cpu").float()
        _launch = False

        return (with_launch - without).abs().max().item()

    def test_neither_mechanism_reproduces_the_bug(self):
        """Baseline/canary, run first: with PR3683's guard forced off AND
        context switching off, nothing protects `r` across the opaque call --
        the nested launch's independent LX planning clobbers it, same as
        pre-PR3683. This is the failure mode both mechanisms exist to
        prevent; if this ever stops reproducing, suspect the test vehicle
        (op/model shape) has drifted, not that the bug is gone -- neither
        real mechanism is exercised in this configuration."""
        with (
            ts_inductor_config.patch({"enable_lx_context_switching": False}),
            _pr3683_guard_disabled(),
        ):
            diff = self._run_launch_diff(layers=1, n=128)
        self.assertGreater(diff, 0.0)

    def test_pr3683_guard_alone_is_correct(self):
        """PR3683's guard active, context switching off: refuses `r`
        residency on LX outright, so the opaque call has nothing to
        clobber."""
        with ts_inductor_config.patch({"enable_lx_context_switching": False}):
            diff = self._run_launch_diff(layers=1, n=128)
        self.assertEqual(diff, 0.0)

    def test_context_switching_alone_is_correct(self):
        """Default (on): PR3683's guard is skipped (see allocator.py's
        matching comments), so `r` can go LX-resident across the opaque
        call -- LxContextSwitchingPass brackets the call with dump/restore
        instead, protecting it directly."""
        with ts_inductor_config.patch({"enable_lx_context_switching": True}):
            diff = self._run_launch_diff(layers=1, n=128)
        self.assertEqual(diff, 0.0)

    def test_correct_at_higher_read_count(self):
        """Both mechanisms stay correct as read_count grows past the
        docs/benchmark table's read_count=1 -- only the *cost* story
        (dump/restore's fixed 2*size/BW vs. the guard's (1+read_count)*size/BW,
        scratchpad_planning.md's "LX context switching" section) is supposed
        to depend on read_count, not correctness. The canary (neither
        mechanism) is included for contrast: unprotected corruption grows
        with read_count too (each extra read of the clobbered buffer
        compounds into the output), which is expected and not itself a bug --
        see test_neither_mechanism_reproduces_the_bug for that mechanism on
        its own."""
        for read_count in (2, 4):
            with self.subTest(read_count=read_count):
                # _run_launch_diff doesn't reset dynamo/inductor state itself
                # (only setUp/tearDown do, once per test *method*) -- the
                # three separate test_*_is_correct methods each get that reset
                # for free between them, but three _run_launch_diff calls in
                # one method/subTest do not, so it is done explicitly here.
                # Skipping this makes the canary silently stop reproducing
                # (state left over from the guard/context-switching runs
                # above changes where `r` lands in LX).
                torch.compiler.reset()
                with ts_inductor_config.patch({"enable_lx_context_switching": False}):
                    guard_diff = self._run_launch_diff(
                        layers=1, n=128, read_count=read_count
                    )
                self.assertEqual(guard_diff, 0.0)

                torch.compiler.reset()
                with ts_inductor_config.patch({"enable_lx_context_switching": True}):
                    ctx_switch_diff = self._run_launch_diff(
                        layers=1, n=128, read_count=read_count
                    )
                self.assertEqual(ctx_switch_diff, 0.0)

                torch.compiler.reset()
                with (
                    ts_inductor_config.patch({"enable_lx_context_switching": False}),
                    _pr3683_guard_disabled(),
                ):
                    canary_diff = self._run_launch_diff(
                        layers=1, n=128, read_count=read_count
                    )
                self.assertGreater(canary_diff, 0.0)


class TestMarkLxSafe(unittest.TestCase):
    """No compile/device needed -- pure attribute/membership checks."""

    def test_mark_lx_safe_sets_attribute(self):
        op = torch.ops.aten.clone.default
        self.assertFalse(getattr(op, "_spyre_lx_safe", False))
        try:
            mark_lx_safe(op)
            self.assertTrue(op._spyre_lx_safe)
        finally:
            op._spyre_lx_safe = False


class TestIsCpuOnlyFallback(unittest.TestCase):
    def test_matches_registered_fallback_ops(self):
        self.assertIn(torch.ops.aten.sin.default, fallback_ops)
        self.assertTrue(_is_cpu_only_fallback(torch.ops.aten.sin.default))

    def test_rejects_non_fallback_ops(self):
        self.assertNotIn(torch.ops.aten.mm.default, fallback_ops)
        self.assertFalse(_is_cpu_only_fallback(torch.ops.aten.mm.default))


if __name__ == "__main__":
    unittest.main()
