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

from collections.abc import Sequence
import logging
import logging.handlers
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch as mock_patch

import pytest
from sympy import Integer, Symbol
import torch
import torch.fx.traceback
from torch.fx.graph_module import GraphModule
from torch._decomp import get_decompositions
from torch._dynamo.test_case import (
    TestCase as DynamoTestCase,
)
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code, InputType


from torch_spyre._inductor import config, spyre_hint
import torch_spyre._inductor.work_division as _wd
import torch_spyre._inductor.propagate_named_dims as _pnd

_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


class TestNamedWorkDivisionHint(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        _pnd.reset()
        self.logger = logging.getLogger("spyre.inductor.work_division")
        self._original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        self.log_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.logger.setLevel(self._original_level)
        _pnd.reset()
        torch._dynamo.reset()
        super().tearDown()

    def _logs(self) -> list[str]:
        self.log_handler.flush()
        return [self.log_handler.format(record) for record in self.log_handler.buffer]

    def _assert_user_hint_logged(self):
        logs = self._logs()
        self.assertTrue(
            any("user-hint" in msg for msg in logs),
            f"Expected user-hint work-division log, got: {logs}",
        )

    def _fake_op(self, loop_var_dims):
        return SimpleNamespace(
            get_name=lambda: "fake_op",
            work_div_loop_info=loop_var_dims,
        )

    def _fake_output_td(self, coord_vars):
        return SimpleNamespace(device_coords=[*coord_vars, Integer(0)])

    def test_resolve_work_div_hint_preserves_hint_order(self):
        h = Symbol("H")
        lq = Symbol("Lq")
        lk = Symbol("Lk")
        op = self._fake_op({h: ["H"], lq: ["Lq"], lk: ["Lk"]})

        with mock_patch(
            "torch_spyre._inductor.work_division.get_op_hints",
            return_value={1: {"work_div": {"H": 4, "Lk": 8, "Lq": 8}}},
        ):
            splits = _wd._resolve_work_div_hint(op, {h: 64, lq: 512, lk: 512})

        self.assertEqual(list(splits.items()), [(h, 4), (lk, 8), (lq, 8)])

    def test_resolve_work_div_hint_filters_to_op_dims(self):
        h = Symbol("H")
        lk = Symbol("Lk")

        def resolve(loop_var_dims, it_space):
            op = self._fake_op(loop_var_dims)
            with mock_patch(
                "torch_spyre._inductor.work_division.get_op_hints",
                return_value={1: {"work_div": {"H": 4, "Lq": 8, "Lk": 8}}},
            ):
                return _wd._resolve_work_div_hint(op, it_space)

        self.assertEqual(
            resolve({h: ["H"], lk: ["Lk"]}, {h: 64, lk: 512}),
            {h: 4, lk: 8},
        )
        self.assertEqual(resolve({lk: ["Lk"]}, {lk: 512}), {lk: 8})

    def test_apply_work_div_hint_prunes_splits_over_sencores(self):
        h = Symbol("H")
        lq = Symbol("Lq")
        lk = Symbol("Lk")
        op = self._fake_op({h: ["H"], lq: ["Lq"], lk: ["Lk"]})

        splits = _wd._apply_user_hint(
            op,
            {h: 4, lq: 8, lk: 8},
            {h: 64, lq: 512, lk: 512},
            self._fake_output_td([h, lq, lk]),
            max_cores=32,
        )

        self.assertEqual(splits, {h: 4, lq: 8})
        logs = self._logs()
        self.assertTrue(
            any(
                "skipping named dim(s)" in msg
                and "fake_op" in msg
                and "['Lk']" in msg
                and "(split=8)" in msg
                and "cores would be 256" in msg
                and "SENCORES=32" in msg
                for msg in logs
            ),
            f"Expected skipped split warning, got: {logs}",
        )

    def test_apply_work_div_hint_prunes_by_priority_order(self):
        h = Symbol("H")
        lq = Symbol("Lq")
        lk = Symbol("Lk")
        op = self._fake_op({h: ["H"], lq: ["Lq"], lk: ["Lk"]})

        splits = _wd._apply_user_hint(
            op,
            {h: 4, lk: 8, lq: 8},
            {h: 64, lq: 512, lk: 512},
            self._fake_output_td([h, lq, lk]),
            max_cores=32,
        )

        self.assertEqual(splits, {h: 4, lk: 8})
        logs = self._logs()
        self.assertTrue(
            any("skipping named dim(s) ['Lq'] (split=8)" in msg for msg in logs),
            f"Expected Lq skip warning, got: {logs}",
        )

    def test_apply_work_div_hint_invalid_split_value_raises_unit(self):
        h = Symbol("H")
        op = self._fake_op({h: ["H"]})

        with self.assertRaisesRegex(Exception, "must be positive"):
            _wd._apply_user_hint(
                op,
                {h: 0},
                {h: 64},
                self._fake_output_td([h]),
                max_cores=32,
            )

    def test_apply_work_div_hint_non_divisible_split_raises_unit(self):
        h = Symbol("H")
        lq = Symbol("Lq")
        op = self._fake_op({h: ["H"], lq: ["Lq"]})

        with self.assertRaisesRegex(Exception, "not evenly divisible"):
            _wd._apply_user_hint(
                op,
                {h: 4, lq: 7},
                {h: 64, lq: 512},
                self._fake_output_td([h, lq]),
                max_cores=32,
            )

    def test_apply_work_div_hint_prunes_before_divisibility_check_unit(self):
        h = Symbol("H")
        lq = Symbol("Lq")
        op = self._fake_op({h: ["H"], lq: ["Lq"]})

        splits = _wd._apply_user_hint(
            op,
            {h: 32, lq: 7},
            {h: 64, lq: 512},
            self._fake_output_td([h, lq]),
            max_cores=32,
        )

        self.assertEqual(splits, {h: 32})

    def test_apply_work_div_hint_multiple_accepted_reduction_splits_raise_unit(self):
        m = Symbol("M")
        k = Symbol("K")
        ell = Symbol("L")
        op = self._fake_op({m: ["M"], k: ["K"], ell: ["L"]})

        with self.assertRaisesRegex(Exception, "reduction dimensions"):
            _wd._apply_user_hint(
                op,
                {k: 2, ell: 2},
                {m: 64, k: 32, ell: 16},
                self._fake_output_td([m]),
                max_cores=32,
            )

    @config.patch({"sencores": 8})
    def test_pointwise_work_div_hint_applied(self):
        M, N = 128, 64
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        y = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])
        _name_tensor_dims(y, ["M", "N"])

        def fn(x, y):
            with spyre_hint(work_div={"M": 4}):
                return x + y

        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x, y)
        self._assert_user_hint_logged()
        self.assertIn("sympify('c0'): (sympify('128'), 4)", source_codes[0])

    @config.patch({"sencores": 8})
    def test_matmul_work_div_hint_maps_by_name(self):
        M, K, N = 128, 256, 64
        x = torch.randn(M, K, dtype=torch.float16).to("spyre")
        y = torch.randn(K, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "K"])
        _name_tensor_dims(y, ["K", "N"])

        def fn(x, y):
            with spyre_hint(work_div={"K": 4, "M": 2}):
                return x @ y

        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x, y)
        self._assert_user_hint_logged()
        self.assertIn("sympify('c0'): (sympify('128'), 2)", source_codes[0])
        self.assertIn("sympify('c2'): (sympify('256'), 4)", source_codes[0])

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Named work-division hints do not yet distinguish component names "
            "inside a reshaped compound dimension."
        ),
    )
    @config.patch({"sencores": 8})
    def test_reshaped_matmul_work_div_hint_maps_component_name(self):
        B, M, K, N = 4, 32, 64, 128
        x = torch.randn(B, M, K, dtype=torch.float16).to("spyre")
        y = torch.randn(K, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["B", "M", "K"])
        _name_tensor_dims(y, ["K", "N"])

        def fn(x, y):
            x_flat = x.reshape(B * M, K)
            with spyre_hint(work_div={"M": 4}):
                return x_flat @ y

        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x, y)
        self._assert_user_hint_logged()
        self.assertIn("sympify('c0'): (sympify('32'), 4)", source_codes[0])
        self.assertNotIn("sympify('z0'): (sympify('4'), 4)", source_codes[0])

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Named work-division hints do not yet distinguish component names "
            "inside a reshaped compound dimension."
        ),
    )
    @config.patch({"sencores": 8})
    def test_reshaped_pointwise_work_div_hint_maps_component_name(self):
        B, M, K = 4, 32, 64
        x = torch.randn(B, M, K, dtype=torch.float16).to("spyre")
        y = torch.randn(B * M, K, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("BM", B * M)
        _name_tensor_dims(x, ["B", "M", "K"])
        _name_tensor_dims(y, ["BM", "K"])

        def fn(x, y):
            x_flat = x.reshape(B * M, K)
            with spyre_hint(work_div={"M": 4}):
                return x_flat + y

        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x, y)
        self._assert_user_hint_logged()
        self.assertIn("sympify('c0'): (sympify('32'), 4)", source_codes[0])
        self.assertNotIn("sympify('z0'): (sympify('4'), 4)", source_codes[0])

    @config.patch({"sencores": 8})
    def test_multiple_hint_blocks(self):
        M, K, N = 128, 64, 256
        x = torch.randn(M, K, dtype=torch.float16).to("spyre")
        w = torch.randn(N, K, dtype=torch.float16).to("spyre")
        b = torch.randn(N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "K"])
        _name_tensor_dims(w, ["N", "K"])
        _name_tensor_dims(b, ["N"])

        def fn(x, w, b):
            with spyre_hint(work_div={"M": 4, "N": 2}):
                mm_out = x @ w.T
            with spyre_hint(work_div={"M": 4, "N": 2}):
                return mm_out + b

        run_and_get_code(
            torch.compile(fn, options={"epilogue_fusion": False}, dynamic=False),
            x,
            w,
            b,
        )
        logs = self._logs()
        self.assertGreaterEqual(
            sum("user-hint" in msg for msg in logs),
            2,
            f"Expected both hint blocks to be consumed, got: {logs}",
        )

    @config.patch({"sencores": 8, "ignore_work_division_hints": True})
    def test_ignore_hints_flag_suppresses_hint(self):
        M, N = 128, 64
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])

        def fn(x):
            with spyre_hint(work_div={"M": 4}):
                return torch.abs(x)

        run_and_get_code(torch.compile(fn, dynamic=False), x)
        self.assertFalse(any("user-hint" in msg for msg in self._logs()))

    @config.patch({"sencores": 8})
    def test_work_div_does_not_create_loop_spec(self):
        M, N = 128, 64
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])

        def fn(x):
            with spyre_hint(work_div={"M": 4}):
                return torch.abs(x)

        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x)
        self.assertNotIn("LoopSpec(", source_codes[0])
        self._assert_user_hint_logged()

    @config.patch(
        {
            "bundle_symbolic_args": True,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "sencores": 8,
        }
    )
    def test_tiles_do_not_create_work_div_hint(self):
        M, N = 128, 64
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])

        def fn(x):
            with spyre_hint(tiles={"M": 4}):
                return torch.abs(x)

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x)
        self.assertIn("LoopSpec(", source_codes[0])
        self.assertFalse(any("user-hint" in msg for msg in self._logs()))

    @config.patch(
        {
            "bundle_symbolic_args": True,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "sencores": 8,
        }
    )
    def test_tiles_and_work_div_coexist(self):
        M, N = 128, 128  # N=128 = 2 sticks; work_div={"N": 2} splits into 1 stick/core
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])

        def fn(x):
            with spyre_hint(tiles={"M": 4}, work_div={"N": 2}):
                return torch.abs(x)

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), x)
        self.assertIn("LoopSpec(", source_codes[0])
        self._assert_user_hint_logged()

    @config.patch({"sencores": 8})
    def test_non_divisible_split_raises(self):
        M, N = 130, 64
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])

        def fn(x):
            with spyre_hint(work_div={"M": 4}):
                return torch.abs(x)

        with self.assertRaisesRegex(Exception, "not evenly divisible"):
            torch.compile(fn, dynamic=False)(x)

    @config.patch({"sencores": 4})
    def test_split_product_exceeding_sencores_skips_later_hint(self):
        # N=128 (2 sticks) so N has its own stick-level device coordinate and is
        # not misidentified as a reduction dim; total requested splits would be
        # 2*2*2 = 8 > sencores=4, so the final split should be skipped.
        M, K, N = 128, 128, 128
        x = torch.randn(M, K, dtype=torch.float16).to("spyre")
        y = torch.randn(K, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "K"])
        _name_tensor_dims(y, ["K", "N"])

        def fn(x, y):
            with spyre_hint(work_div={"M": 2, "N": 2, "K": 2}):
                return x @ y

        run_and_get_code(torch.compile(fn, dynamic=False), x, y)
        logs = self._logs()
        self.assertTrue(
            any(
                "skipping named dim(s)" in msg
                and "['K']" in msg
                and "(split=2)" in msg
                and "cores would be 8" in msg
                and "SENCORES=4" in msg
                for msg in logs
            ),
            f"Expected skipped split warning, got: {logs}",
        )
        self.assertFalse(any("exceeds SENCORES" in msg for msg in logs))

    @config.patch({"sencores": 8})
    def test_invalid_split_value_raises(self):
        M, N = 128, 64
        x = torch.randn(M, N, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x, ["M", "N"])

        def fn(x):
            with spyre_hint(work_div={"M": 0}):
                return torch.abs(x)

        with self.assertRaisesRegex(Exception, "must be positive"):
            torch.compile(fn, dynamic=False)(x)

    @config.patch({"sencores": 8})
    def test_multiple_reduction_splits_raise(self):
        # Keep L large enough that the failure is reduction-split validation,
        # not stick alignment.
        M, K, L = 64, 32, 128
        x = torch.randn(M, K, L, dtype=torch.float16).to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("L", L)
        _name_tensor_dims(x, ["M", "K", "L"])

        def fn(x):
            with spyre_hint(work_div={"K": 2, "L": 2}):
                return x.sum(dim=(1, 2))

        with self.assertRaisesRegex(
            Exception, "reduction dimensions|expected exactly 1 reduction variable"
        ):
            torch.compile(fn, dynamic=False)(x)


def aot_backend(gm: GraphModule, example_inputs: Sequence[InputType]):
    decompositions = get_decompositions(
        [
            torch.ops.aten.gelu.default,
            torch.ops.aten.gelu_backward.default,
        ]
    )

    def fw(gm: GraphModule, example_inputs: Sequence[InputType]) -> Any:
        for node in gm.graph.nodes:
            if node.op not in ["placeholder", "output"]:
                meta = node.meta.get("custom", {})
                assert meta.get("custom_hint", 0) == 1

        return make_boxed_func(gm.forward)

    def bw(gm: GraphModule, example_inputs: Sequence[InputType]) -> Any:
        return make_boxed_func(gm.forward)

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw,
        bw_compiler=bw,
        decompositions=decompositions,
    )  # type: ignore


class TestAOTAnnotationAssumptions(DynamoTestCase):
    def _compile_and_run(self, model: torch.nn.Module):
        x = torch.randn((64, 64), dtype=torch.float16, device="cpu")
        compiled = torch.compile(model, fullgraph=True, backend=aot_backend)
        for i in range(2):
            compiled(x)

    def test_dead_code_elimination(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                with torch.fx.traceback.annotate({"custom_hint": 1}):
                    y = torch.zeros_like(x)
                    y = torch.cos(y)
                    return x + 1

        self._compile_and_run(TestModule())

    def test_decomposition(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                with torch.fx.traceback.annotate({"custom_hint": 1}):
                    return torch.nn.functional.gelu(x)

        self._compile_and_run(TestModule())

    def test_functionalization(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                with torch.fx.traceback.annotate({"custom_hint": 1}):
                    y = torch.zeros_like(x)
                    y.add_(x)
                    return y

        self._compile_and_run(TestModule())


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
