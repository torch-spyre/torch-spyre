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

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import sympy
import torch
from torch.testing import FileCheck
from torch._inductor.exc import InductorError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    run_and_get_code,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.work_division import (
    _collect_symbol_metadata,
    _effective_size,
    _valid_divisor_basis,
    adjust_it_space_for_sticks,
)


class TestSpyreConfig(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_config_default(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        comp_fn = torch.compile(fn)
        out, source_codes = run_and_get_code(comp_fn, x)
        # print("test_config_default")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").check(
            f"sympify('c0'): (sympify('256'), {config.sencores})"
        ).run(source_codes[0])

    @config.patch({"sencores": 64})
    def test_config_too_many_sencores(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        with self.assertRaisesRegex(
            InductorError,
            "Unsupported: Spyre backend does not support: invalid SENCORES value 64",
        ):
            comp_fn = torch.compile(fn)
            comp_fn(x)

    @config.patch({"sencores": 16})
    def test_sencores_16(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")
        cfn = torch.compile(fn, dynamic=False)
        out, source_codes = run_and_get_code(cfn, x)
        # print("test_sencores 16")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").check(
            f"sympify('c0'): (sympify('256'), {config.sencores})"
        ).run(source_codes[0])

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_pointwise_split(self):
        """Symbolic batch dim must split by ``granularity`` not ``max_size`` (#2287).

        ``[s, 128]`` fp16 with ``s in [64, 1024]`` (granularity = 64). The planner picks the largest
        divisor of granularity ≤ SENCORES = 32, so the batch dim absorbs all
        32 cores and the static stick dim gets split 1.
        """
        fn = torch.add
        x = torch.randn((1024, 128), dtype=torch.float16)
        y = torch.randn_like(x)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        torch._dynamo.mark_dynamic(y, 0, min=64, max=1024)
        comp_fn = torch.compile(fn, dynamic=True)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"), y.to("spyre"))
        # Iteration space embeds (size_expr, split). The symbolic batch dim's
        # split must equal SENCORES=32; the static stick dim's split must be 1.
        FileCheck().check("sdsc_fused_add").check(", 32)").check(", 1)").run(
            source_codes[0]
        )

    # Need a test where changing dxp_lx_frac_avail changes the generated OpSpec
    # @config.patch({"dxp_lx_frac_avail": 0.01, "lx_planning": True})
    # def test_config_dxp_lx_frac_avail(self):
    #    fn = torch.abs
    #    x = torch.randn((256, 128, 512)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    #print("test_conf_dxp_lx_frac_avail")
    #    #print(source_codes[0])

    # Need a test where setting lx_planning to True generates a different OpSpec
    # @config.patch({'lx_planning': True})
    # def test_config_lx_planning(self):
    #    fn = torch.abs
    #    x = torch.randn((256, 128, 512)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    #print(source_codes[0])

    # ------------------------------------------------------------------
    # Unit tests for the symbolic-shape sidecar in work_division.py
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_v(lower=None, upper=None, size_hint=None):
        """
        Mock V whose ShapeEnv reports the given lower / upper bounds.
        """
        shape_env = SimpleNamespace(
            bound_sympy=lambda _e: SimpleNamespace(lower=lower, upper=upper)
        )
        sizevars = SimpleNamespace(shape_env=shape_env)
        if size_hint is not None:
            sizevars.size_hint = lambda _e: size_hint
        return SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))

    def test_collect_symbol_metadata_opt_in(self):
        """
        User-marked dynamic dim (finite max) enters the metadata dict.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch(
                "torch_spyre._inductor.pass_utils.V",
                self._mock_v(lower=sympy.Integer(2), upper=sympy.Integer(512)),
            ):
                result = _collect_symbol_metadata({s0: s0})
        # max comes straight from the ShapeEnv upper bound;
        # granularity is the smallest divisor of 512 with d >= 4 and
        # 512/d <= 32, which is 16.
        self.assertEqual(result, {s0: (512, 16)})

    def test_collect_symbol_metadata_auto_dynamic_skipped(self):
        """
        Dynamo-promoted symbols (no finite max) are skipped, not assigned.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        with patch(
            "torch_spyre._inductor.pass_utils.V",
            self._mock_v(lower=sympy.Integer(2), upper=sympy.oo, size_hint=1024),
        ):
            self.assertEqual(_collect_symbol_metadata({s0: s0}), {})

    def test_dispatch_helpers_symbolic_vs_concrete(self):
        """
        ``_effective_size`` and ``_valid_divisor_basis`` dispatch on ``v in meta``.
        """
        s0 = sympy.Symbol("s0")
        it_space = {s0: sympy.Integer(128)}
        meta = {s0: (512, 16)}
        # In meta: use the (max, granularity) tuple.
        self.assertEqual(_effective_size(s0, it_space, meta), 512)
        self.assertEqual(_valid_divisor_basis(s0, it_space, meta), 16)
        # Not in meta: fall through to concretize_expr(it_space[v]).
        self.assertEqual(_effective_size(s0, it_space, meta={}), 128)
        self.assertEqual(_valid_divisor_basis(s0, it_space, meta={}), 128)

    def test_symbolic_stick_dim_raises_unsupported(self):
        """
        A symbolic dim that lands on a tensor's stick coord is rejected.
        This is a follow up work.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        # Minimal TensorDep stand-in: the function only reads
        # device_coords[-1], dep.name, and layout.device_layout.elems_per_stick().
        fake_td = SimpleNamespace(
            dep=SimpleNamespace(name="fake_buf"),
            layout=SimpleNamespace(
                device_layout=SimpleNamespace(elems_per_stick=lambda: 64)
            ),
            device_coords=[s0],
        )
        with self.assertRaises(Unsupported) as cm:
            adjust_it_space_for_sticks(
                {s0: sympy.Integer(128)}, [fake_td], {s0: (512, 64)}
            )
        self.assertIn("symbolic stick dim", str(cm.exception))
