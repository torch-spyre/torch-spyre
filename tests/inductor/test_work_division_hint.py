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
from dataclasses import replace
import logging
import logging.handlers
import regex as re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch as mock_patch

import pytest
from sympy import floor, Integer, Mod, Symbol
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
import torch_spyre._inductor.scratchpad.lx_relayout as lx_relayout_module
import torch_spyre._inductor.scheduler as scheduler_module
import torch_spyre._inductor.work_division as _wd
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec, parse_op_spec
from torch_spyre._inductor.constants import IDENTITY_OP
from torch_spyre._inductor.scratchpad.lx_relayout import (
    LXRelayoutPlan,
    work_division_from_view,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg, TensorWorkDivision
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.spyre_kernel import _remap_work_division, simplify_op_spec

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

    def test_apply_work_div_hint_rejects_illegal_split(self):
        m = Symbol("M")
        op = self._fake_op({m: ["M"]})

        with self.assertRaisesRegex(Exception, "legal splits are"):
            _wd._apply_user_hint(
                op,
                {m: 2},
                {m: 64},
                self._fake_output_td([m]),
                max_cores=32,
                allowed_splits={m: frozenset({1})},
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


_CORE_ID = Symbol("core_id")
_SOURCE_VIEW = PerCoreView(
    ((0, 4), (1, 2)), ((0, floor(_CORE_ID / 2)), (1, Mod(_CORE_ID, 2)))
)
_DESTINATION_VIEW = PerCoreView(((0, 8),), ((0, _CORE_ID),))


def _relayout_plan(source="source", consumers="consumer"):
    if isinstance(consumers, str):
        consumers = (consumers,)
    return LXRelayoutPlan(source, consumers, _SOURCE_VIEW, _DESTINATION_VIEW, 8)


def test_lx_relayout_activation_policy_is_source_wide():
    dep = SimpleNamespace(name="input")
    producer = SimpleNamespace()
    with (
        mock_patch.object(
            lx_relayout_module, "op_short_name", return_value="restickify"
        ),
        mock_patch.object(
            lx_relayout_module,
            "op_read_writes",
            return_value=SimpleNamespace(reads=[dep]),
        ),
        mock_patch.object(lx_relayout_module, "MemoryDep", SimpleNamespace),
        mock_patch.object(lx_relayout_module, "ComputedBuffer", SimpleNamespace),
    ):
        assert not lx_relayout_module._is_activation_source({}, producer)
        assert lx_relayout_module._is_activation_source({"input": dep}, producer)


def test_lx_relayout_planner_rejects_equal_projected_ownership():
    m = Symbol("m")
    source_view = PerCoreView(
        ((1, 32),),
        ((1, Mod(_CORE_ID, 32)),),
    )
    destination_view = PerCoreView(
        ((0, 32),),
        ((0, Mod(_CORE_ID, 32)),),
    )
    coordinates = [m, m]
    source_work_division = work_division_from_view(source_view, coordinates, (m,))
    destination_work_division = work_division_from_view(
        destination_view, coordinates, (m,)
    )
    assert source_view != destination_view
    assert source_work_division == destination_work_division

    source_dep = SimpleNamespace(name="source", is_indirect=lambda: False)
    producer = SimpleNamespace(
        layout=SimpleNamespace(device_layout=SimpleNamespace()),
        data=SimpleNamespace(),
        get_name=lambda: "source",
    )
    consumer = SimpleNamespace(
        layout=SimpleNamespace(),
        data=SimpleNamespace(),
        get_name=lambda: "consumer",
    )
    graph = SimpleNamespace(operations=[producer, consumer])

    def read_writes(op):
        if op is producer:
            return SimpleNamespace(reads=[], writes=[source_dep])
        return SimpleNamespace(reads=[source_dep], writes=[])

    with (
        mock_patch.object(lx_relayout_module, "MemoryDep", SimpleNamespace),
        mock_patch.object(lx_relayout_module, "ComputedBuffer", SimpleNamespace),
        mock_patch.object(lx_relayout_module, "FixedTiledLayout", SimpleNamespace),
        mock_patch.object(lx_relayout_module, "Pointwise", SimpleNamespace),
        mock_patch.object(
            lx_relayout_module, "op_read_writes", side_effect=read_writes
        ),
        mock_patch.object(
            lx_relayout_module,
            "_per_core_view_on_buf",
            side_effect=[
                (source_view, False, True),
                (destination_view, False, True),
            ],
        ),
        mock_patch.object(lx_relayout_module, "_op_num_cores", return_value=32),
        mock_patch.object(
            lx_relayout_module, "try_device_coordinates", return_value=coordinates
        ),
        mock_patch.object(
            lx_relayout_module, "iteration_space_from_op", return_value=(m,)
        ),
        mock_patch.object(
            lx_relayout_module, "op_short_name", return_value="pointwise"
        ),
    ):
        assert lx_relayout_module.collect_lx_relayout_plans(graph) == []


def _compile_spec(spec, normalize=True):
    if normalize:
        simplify_op_spec(spec)
    payload, *_ = compile_op_spec(0, spec, [])
    root = next(iter(payload.values()))
    dsc = next(iter(root["dscs_"][0].values()))
    return root, [
        node for node in dsc["scheduleTree_"] if node["nodeType_"] == "allocate"
    ]


def test_lx_relayout_normalizes_ownership_and_lowers_only_in_superdsc():
    m, n = Symbol("m"), Symbol("n")
    source_view = PerCoreView(
        ((1, 4), (2, 2)),
        ((1, floor(_CORE_ID / 2)), (2, Mod(_CORE_ID, 2))),
    )
    destination_view = PerCoreView(
        ((1, 2), (2, 4)),
        ((1, Mod(_CORE_ID, 2)), (2, floor(_CORE_ID / 2))),
    )
    coordinates = [Mod(n, 32), floor(n / 32), Mod(m, 64)]
    base = TensorArg(
        True, -1, DataFormats.SEN169_FP16, [32, 8, 64], coordinates, {"lx": 0}
    )
    args = [
        replace(
            base,
            work_division=work_division_from_view(source_view, coordinates, (m, n)),
        ),
        replace(
            base,
            is_input=False,
            allocation={"lx": 256},
            work_division=work_division_from_view(
                destination_view, coordinates, (m, n)
            ),
        ),
    ]
    spec = OpSpec(IDENTITY_OP, False, {n: (256, 8), m: (64, 1)}, args, {})
    root, allocations = _compile_spec(spec)
    assert spec.op == IDENTITY_OP
    assert set(root["dscs_"][0]) == {"shuffle"}
    assert root["dscs_"][0]["shuffle"]["labeledDs_"][0]["dsType_"] == "OUTPUT"
    assert [arg.work_division.work_slices for arg in spec.args] == [
        {Symbol("z0"): 4, m: 2},
        {Symbol("z0"): 2, m: 4},
    ]
    assert root["numWkSlicesPerDim_"] == {"mb": 1, "x": 8, "out": 1}
    maps = [node["coordinates_"]["coreIdToWkSlice_"] for node in allocations]
    assert [maps[0][str(i)]["x"] for i in range(8)] == [i // 2 for i in range(8)]
    assert [maps[0][str(i)]["out"] for i in range(8)] == [i % 2 for i in range(8)]
    assert [maps[1][str(i)]["x"] for i in range(8)] == [i % 2 for i in range(8)]
    assert [maps[1][str(i)]["out"] for i in range(8)] == [i // 2 for i in range(8)]
    coord_info = [node["coordinates_"]["coordInfo"] for node in allocations]
    assert coord_info[0]["x"]["folds"]["dim_prop_func"][0]["Affine"]["alpha_"] == 2
    assert coord_info[1]["x"]["folds"]["dim_prop_func"][0]["Affine"]["alpha_"] == 4
    with pytest.raises(ValueError, match="cannot map device dimension"):
        work_division_from_view(source_view, [Integer(0), m + n, Integer(0)], (m, n))

    for arg in spec.args:
        arg.work_division = None
    ordinary_root, ordinary_allocations = _compile_spec(spec, normalize=False)
    ordinary_sdsc, _ = parse_op_spec(spec)
    assert set(ordinary_root["dscs_"][0]) == {IDENTITY_OP}
    assert all(arg.work_division is not None for arg in ordinary_sdsc.args)
    assert all(
        not node["coordinates_"]["coreIdToWkSlice_"] for node in ordinary_allocations
    )

    old, inner, outer = Symbol("old"), Symbol("inner"), Symbol("outer")
    remapped = replace(
        base,
        work_division=TensorWorkDivision({old: 16}, {old: Mod(_CORE_ID, 16)}),
    )
    _remap_work_division(remapped, {old: ((inner, 2), (outer, 8))})
    assert remapped.work_division is not None
    core_three = {
        dim: int(slot.subs(_CORE_ID, 3))
        for dim, slot in remapped.work_division.core_id_to_work_slice.items()
    }
    assert core_three == {inner: 1, outer: 1}


@config.patch(
    {
        "sencores": 8,
        "lx_planning": True,
        "allow_all_ops_in_lx_planning": True,
        "lx_planner_relayout": True,
        # LX relayout needs a paired-buffer-capable solver; only the greedy
        # solver sets supports_paired_buffers. Pin it explicitly so this test
        # keeps exercising relayout regardless of the default layout_solver.
        "layout_solver": "greedy",
    }
)
@pytest.mark.parametrize("second_consumer", ["pointwise", "matmul_lhs", "matmul_rhs"])
def test_lx_relayout_consumers_share_destination_view(second_consumer):
    torch.manual_seed(0)
    m_size = 64 if second_consumer == "matmul_rhs" else 32
    x = torch.randn(8, m_size, 64, dtype=torch.float16)
    weight = torch.randn(8, 64, m_size, dtype=torch.float16)
    for name, size in (
        ("B", 8),
        ("M", m_size),
        ("K", 64),
        ("N", m_size),
        ("L", 64),
    ):
        _declare_tensor_dim(name, size)

    shares_destination = second_consumer in ("pointwise", "matmul_lhs")

    def fn(x, weight):
        with spyre_hint(work_div={"B": 4, "M": 2}):
            hidden = torch.neg(x)
        with spyre_hint(work_div={"B": 2, "M": 4}):
            pointwise = torch.relu(hidden)
        second_work_div = {"B": 2, "M": 4} if shares_destination else {"B": 8}
        with spyre_hint(work_div=second_work_div):
            if second_consumer == "matmul_lhs":
                second = torch.bmm(hidden, weight)
            elif second_consumer == "matmul_rhs":
                second = torch.bmm(weight, hidden)
            else:
                second = torch.abs(hidden)
        return pointwise, second

    device_x = _name_tensor_dims(x.to("spyre"), ["B", "M", "K"])
    weight_dims = (
        ["B", "L", "M"] if second_consumer == "matmul_rhs" else ["B", "K", "N"]
    )
    device_weight = _name_tensor_dims(weight.to("spyre"), weight_dims)
    torch._inductor.codecache.FxGraphCache.clear()
    actual, code = run_and_get_code(
        torch.compile(fn, dynamic=False, options={"epilogue_fusion": False}),
        device_x,
        device_weight,
    )
    for index, (got, expected) in enumerate(zip(actual, fn(x, weight))):
        tolerance = (
            {"rtol": 2e-2, "atol": 1e-1}
            if index == 1 and second_consumer.startswith("matmul")
            else {}
        )
        torch.testing.assert_close(got.cpu(), expected, **tolerance)
    generated = "\n".join(code)
    identities = [
        block for block in generated.split("OpSpec(") if "op='identity'" in block[:100]
    ]
    ordinary = [
        block
        for block in generated.split("OpSpec(")
        if "op='" in block[:100] and "op='identity'" not in block[:100]
    ]
    assert all("work_division=" not in block for block in ordinary)
    divisions = [
        re.findall(r"TensorWorkDivision\(work_slices=\{([^}]*)", block)
        for block in identities
    ]
    expected_copies = 1 if shares_destination else 2
    assert len(divisions) == expected_copies
    assert all(len(pair) == 2 for pair in divisions)
    assert {pair[0] for pair in divisions} == {"sympify('c1'): 2, sympify('c0'): 4"}
    expected_destinations = {"sympify('c1'): 4, sympify('c0'): 2"}
    if not shares_destination:
        expected_destinations.add("sympify('c0'): 8")
    assert {pair[1] for pair in divisions} == expected_destinations


def test_lx_relayout_allocation_is_atomic_in_one_greedy_solve(caplog):
    alternate_view = PerCoreView(((1, 8),), ((1, _CORE_ID),))
    plans = [
        _relayout_plan("source", ("consumer_a", "consumer_b")),
        LXRelayoutPlan(
            "source",
            ("consumer_c",),
            _SOURCE_VIEW,
            alternate_view,
            8,
        ),
    ]
    allocator = ScratchpadAllocator(GreedyLayoutSolver, 256)
    graph = SimpleNamespace(
        operations=[
            SimpleNamespace(get_name=lambda name=name: name)
            for name in (
                "producer",
                "consumer_a",
                "consumer_b",
                "consumer_c",
                "ordinary_consumer",
            )
        ]
    )
    source = LifetimeBoundBuffer("source", 128, [0, 1, 2, 3])
    source.lx_relayout_plans = list(plans)
    ordinary = LifetimeBoundBuffer("ordinary", 128, [0, 4])
    buffers = [source, ordinary]
    allocator._append_lx_relayout_destinations(graph, buffers)

    assert source.uses == [1, 2, 6]
    assert [buffer.uses for buffer in source.paired_with] == [[2, 3, 5], [6, 7]]

    solver = allocator._build_solver(buffers)
    with caplog.at_level(logging.DEBUG, logger="spyre.inductor.scratchpad.allocator"):
        allocation = allocator._solve(solver)
        allocator._finalize_lx_relayout_allocation(allocation)

    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["ordinary"].address == 0
    assert by_name["source"].address is None
    assert all(by_name[plan.destination_name].address is None for plan in plans)
    assert not by_name["source"].lx_relayout_plans
    assert any(
        "rejected LX relayout group source=source" in record.message
        for record in caplog.records
    )


def _assert_live_buffers_do_not_share_addresses(buffers, limit):
    allocator = ScratchpadAllocator(GreedyLayoutSolver, limit)
    allocation = allocator._solve(allocator._build_solver(buffers))
    assert all(buffer.address is not None for buffer in allocation)
    for index, left in enumerate(allocation):
        for right in allocation[index + 1 :]:
            if not left.overlaps_in_time(right):
                continue
            assert left.address + left.size <= right.address or (
                right.address + right.size <= left.address
            )


def test_lx_relayout_copies_loop_lifetime_to_every_destination():
    plans = [
        _relayout_plan("source", ("consumer_a", "consumer_b")),
        LXRelayoutPlan(
            "source",
            ("consumer_c",),
            _SOURCE_VIEW,
            PerCoreView(((1, 8),), ((1, _CORE_ID),)),
            8,
        ),
    ]
    graph = SimpleNamespace(
        operations=[
            SimpleNamespace(get_name=lambda name=name: name)
            for name in ("producer", "consumer_a", "consumer_b", "consumer_c")
        ]
    )
    source = LifetimeBoundBuffer("source", 64, [0, 1, 2, 3], lifetime_end_override=6)
    source.lx_relayout_plans = list(plans)
    buffers = [source]
    ScratchpadAllocator(GreedyLayoutSolver, 256)._append_lx_relayout_destinations(
        graph, buffers
    )

    assert source.lifetime_end_override is None
    assert [buffer.lifetime_end_override for buffer in source.paired_with] == [12, 12]
    tail = LifetimeBoundBuffer("tail", 64, [8, 11])
    buffers.append(tail)
    _assert_live_buffers_do_not_share_addresses(buffers, 384)


def test_lx_relayout_keeps_source_lifetime_for_later_original_reader():
    graph = SimpleNamespace(
        operations=[
            SimpleNamespace(get_name=lambda name=name: name)
            for name in ("producer", "relayout_consumer", "ordinary_consumer")
        ]
    )
    source = LifetimeBoundBuffer("source", 64, [0, 1, 2], lifetime_end_override=4)
    source.lx_relayout_plans = [_relayout_plan("source", "relayout_consumer")]
    buffers = [source]
    ScratchpadAllocator(GreedyLayoutSolver, 192)._append_lx_relayout_destinations(
        graph, buffers
    )

    assert source.lifetime_end_override == 8
    assert source.paired_with[0].lifetime_end_override == 8
    tail = LifetimeBoundBuffer("tail", 64, [6, 7])
    buffers.append(tail)
    _assert_live_buffers_do_not_share_addresses(buffers, 384)


@config.patch({"lx_planner_relayout": True})
def test_lx_relayout_warns_for_unsupported_solver(caplog):
    class UnsupportedSolver:
        pass

    allocator = ScratchpadAllocator(UnsupportedSolver, 256)
    allocator._generate_buffers = lambda _graph: []
    with caplog.at_level(logging.WARNING, logger="spyre.inductor.scratchpad.allocator"):
        assert allocator._prepare_buffers(SimpleNamespace()) == []
    assert any(
        "LX relayout is not supported by UnsupportedSolver" in record.message
        for record in caplog.records
    )


class _RelayoutNode:
    def __init__(self, name, reads=(), writes=(), layout=None):
        self.name = name
        self.node = SimpleNamespace(layout=layout or SimpleNamespace(allocation={}))
        self.read_writes = SimpleNamespace(reads=list(reads), writes=list(writes))

    def get_nodes(self):
        return [self]

    def get_name(self):
        return self.name


def _relayout_layout(address, view):
    return SimpleNamespace(allocation={"lx": address}, lx_view=view)


def test_lx_relayout_scheduler_checks_final_ownership_projection():
    m, n = Symbol("m"), Symbol("n")
    layout = SimpleNamespace(device_layout=object())
    graph = SimpleNamespace(
        try_get_buffer=lambda name: (
            SimpleNamespace(get_layout=lambda: layout) if name == "source" else None
        )
    )
    node, dep = SimpleNamespace(), SimpleNamespace()

    def projectable(coordinates):
        with (
            mock_patch.object(scheduler_module, "V", SimpleNamespace(graph=graph)),
            mock_patch.object(scheduler_module, "FixedTiledLayout", SimpleNamespace),
            mock_patch.object(
                scheduler_module,
                "try_device_coordinates",
                return_value=coordinates,
            ),
            mock_patch.object(
                scheduler_module, "iteration_space", return_value={m: 32, n: 64}
            ),
        ):
            return scheduler_module._ownership_projectable(
                node, dep, "source", _SOURCE_VIEW
            )

    assert projectable([m, n])
    assert not projectable([m + n, n])


def test_lx_relayout_scheduler_demotes_groups_but_not_ordinary_unary():
    def run_registered(drift):
        plan = _relayout_plan()
        src, dst = SimpleNamespace(name="source"), SimpleNamespace(name="destination")
        unary_src = SimpleNamespace(name="ordinary_source")
        unary_dst = SimpleNamespace(name="ordinary_unary")
        layouts = {
            "source": _relayout_layout(0, _SOURCE_VIEW),
            "destination": _relayout_layout(256, _DESTINATION_VIEW),
            "ordinary_source": _relayout_layout(512, _SOURCE_VIEW),
            "ordinary_unary": _relayout_layout(768, _DESTINATION_VIEW),
        }
        node = _RelayoutNode
        nodes = [
            node("source", writes=(src,), layout=layouts["source"]),
            node("destination", (src,), (dst,), layouts["destination"]),
            node("consumer", reads=(dst,)),
            node(
                "ordinary_source",
                writes=(unary_src,),
                layout=layouts["ordinary_source"],
            ),
            node(
                "ordinary_unary", (unary_src,), (unary_dst,), layouts["ordinary_unary"]
            ),
            node("ordinary_consumer", reads=(unary_dst,)),
        ]
        if drift == "missing":
            nodes = [node for node in nodes if node.name != "destination"]
        buffers = {
            name: SimpleNamespace(
                layout=SimpleNamespace(),
                get_layout=lambda layout=layout: layout,
            )
            for name, layout in layouts.items()
        }
        if drift == "missing_buffer":
            del buffers["destination"]
        graph = SimpleNamespace(
            _spyre_lx_relayout_copies={plan.edge: ("destination", plan)},
            try_get_buffer=buffers.get,
            get_buffer=buffers.__getitem__,
        )

        def view(node, _dep, name):
            if name == "ordinary_source":
                expected = (
                    _DESTINATION_VIEW if node.name == "ordinary_unary" else _SOURCE_VIEW
                )
            else:
                expected = _SOURCE_VIEW if name == "source" else _DESTINATION_VIEW
            return (
                PerCoreView((), ()) if node.name == drift else expected,
                False,
                True,
            )

        with (
            mock_patch.object(scheduler_module, "SchedulerNode", _RelayoutNode),
            mock_patch.object(scheduler_module, "MemoryDep", SimpleNamespace),
            mock_patch.object(scheduler_module, "FixedTiledLayout", SimpleNamespace),
            mock_patch.object(lx_relayout_module, "FixedTiledLayout", SimpleNamespace),
            mock_patch.object(scheduler_module, "V", SimpleNamespace(graph=graph)),
            mock_patch.object(scheduler_module, "per_core_view_scheduled", view),
            mock_patch.object(
                scheduler_module,
                "_ownership_projectable",
                side_effect=lambda node, _dep, _name, _view: (
                    not (drift == "projection" and node.name == "consumer")
                ),
            ),
            config.patch({"lx_planning": True}),
        ):
            scheduler_module.demote_incoherent_lx_buffers(nodes)
        assert graph._spyre_lx_relayout_copies == {}
        assert "lx" not in layouts["source"].allocation
        if drift != "missing_buffer":
            assert "lx" not in layouts["destination"].allocation
            assert layouts["destination"].lx_view is None
        assert "lx" not in layouts["ordinary_source"].allocation
        assert "lx" in layouts["ordinary_unary"].allocation

    run_registered("source")
    run_registered("consumer")
    run_registered("projection")
    run_registered("missing")
    run_registered("missing_buffer")


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
