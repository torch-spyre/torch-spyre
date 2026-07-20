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

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.codegen.compute_ops import (
    SymbolKind,
    _per_core_symbolic_dim_info,
)
from torch_spyre._inductor.codegen.superdsc import _resolve_sdsc_size, compile_op_spec
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
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
        cfn = torch.compile(fn)
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
        # dynamic=True not needed: mark_dynamic already makes dim 0 symbolic.
        comp_fn = torch.compile(fn, dynamic=False)
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

    def test_inplace_op_run_call_deduplicates_args(self):
        """An inplace op (x *= 2) must not pass the same tensor twice to .run().

        With symbolic args, the MLIR bundle emits one input_arg param per unique
        tensor.  Passing arg0_1 twice would cause a "Number of inputs mismatches"
        error at launch time.  This test verifies the generated .run() call
        contains no duplicate tensor arguments.
        """

        def fn(x):
            x *= 2
            return x

        x = torch.randn((4, 128), dtype=torch.float16, device="spyre")
        cfn = torch.compile(fn)
        _, source_codes = run_and_get_code(cfn, x)
        code = source_codes[0]
        # Find the .run(...) call for the fused kernel
        run_lines = [ln.strip() for ln in code.splitlines() if ".run(" in ln]
        self.assertTrue(run_lines, "No .run(...) call found in generated code")
        for line in run_lines:
            # Extract the argument list between the outermost parentheses
            args_str = line[line.index("(") + 1 : line.rindex(")")]
            args = [a.strip() for a in args_str.split(",")]
            self.assertEqual(
                len(args),
                len(set(args)),
                f"Duplicate args in .run() call: {line}",
            )


class TestResolveSdscSize(InductorTestCase):
    """Unit tests for superdsc._resolve_sdsc_size."""

    def test_concrete_sympy_integer(self):
        self.assertEqual(_resolve_sdsc_size(sympy.Integer(256), {}), 256)

    def test_concrete_python_int(self):
        self.assertEqual(_resolve_sdsc_size(128, {}), 128)

    def test_symbolic_in_bounds_returns_max(self):
        # bounds carries (max, granularity); index [0] is max.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        self.assertEqual(_resolve_sdsc_size(s0, {"s0": (1024, 64)}), 1024)

    def test_symbolic_not_in_bounds_falls_back_to_size_hint(self):
        # Symbol absent from bounds → _concretize_for_sdsc → size_hint.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        sizevars = SimpleNamespace(size_hint=lambda _: 128)
        mock_v = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with patch("torch_spyre._inductor.codegen.superdsc.V", mock_v):
            self.assertEqual(_resolve_sdsc_size(s0, {}), 128)


class TestSymbolKindDimension(InductorTestCase):
    """Unit tests for the dimension variant added to compute_ops.SymbolKind."""

    def test_factory_sets_all_fields(self):
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertEqual(sk.kind, "dimension")
        self.assertEqual(sk.granularity, 64)
        self.assertEqual(sk.max_value, 1024)
        self.assertEqual(sk.pytorch_sym, "s0")

    def test_is_dimension_true(self):
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertTrue(sk.is_dimension)

    def test_address_fields_are_sentinels(self):
        # Address-specific fields must not be set by the dimension factory so
        # they don't collide with kernel/pool symbol-table entries.
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertEqual(sk.arg_index, -1)
        self.assertEqual(sk.base_sym_idx, -1)
        self.assertEqual(sk.offset, 0)

    def test_kernel_is_not_dimension(self):
        self.assertFalse(SymbolKind.kernel(arg_index=0).is_dimension)

    def test_pool_is_not_dimension(self):
        self.assertFalse(SymbolKind.pool().is_dimension)


class TestPerCoreSymbolicDimInfo(InductorTestCase):
    """Unit tests for compute_ops._per_core_symbolic_dim_info."""

    def test_no_symbolic_dims_returns_empty(self):
        self.assertEqual(_per_core_symbolic_dim_info({}, {}), {})

    def test_single_dim_no_split(self):
        # work_slices == 1 means undivided: maxSize_/granularity_ pass through.
        symbolic_dims = {"c0": ("s0", 64, 1024)}
        work_slices = {sympy.Symbol("c0"): 1}
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {"c0": {"maxSize_": 1024, "granularity_": 64}},
        )

    def test_single_dim_split_across_cores(self):
        symbolic_dims = {"c0": ("s0", 64, 1024)}
        work_slices = {sympy.Symbol("c0"): 4}
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {"c0": {"maxSize_": 256, "granularity_": 16}},
        )

    def test_granularity_floors_at_one(self):
        # granularity // wk_slices would floor to 0; result must clamp to 1
        # so the runtime never sees a zero batch-size granularity.
        symbolic_dims = {"c0": ("s0", 1, 1024)}
        work_slices = {sympy.Symbol("c0"): 4}
        result = _per_core_symbolic_dim_info(symbolic_dims, work_slices)
        self.assertEqual(result["c0"], {"maxSize_": 256, "granularity_": 1})

    def test_multiple_symbolic_dims_independent(self):
        symbolic_dims = {
            "c0": ("s0", 64, 1024),
            "c1": ("s1", 32, 512),
        }
        work_slices = {
            sympy.Symbol("c0"): 4,
            sympy.Symbol("c1"): 2,
        }
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {
                "c0": {"maxSize_": 256, "granularity_": 16},
                "c1": {"maxSize_": 256, "granularity_": 16},
            },
        )


class TestSdscJsonSymbolicDimSmoke(InductorTestCase):
    """Smoke test: a symbolic iteration-space dim survives end-to-end through
    compile_op_spec (parse_op_spec + generate_sdsc) into the emitted SDSC
    JSON's dimToSymbolMapping_ / symbolicDimInfo_ fields.

    Fixture uses a [512, 256] fp16 stick-layout tensor with the row dim
    made symbolic. Because _resolve_sdsc_size resolves a symbolic dim to
    its declared max (512), every downstream computation (padding,
    stick-dim detection, core slicing) runs identically to the equivalent
    concrete case -- only the symbolic_dims side-channel asserted on here
    differs.
    """

    _DEVICE_SIZE = [4, 512, 64]
    _HBM_BASE = 0x400000000

    def _make_symbolic_op_spec(self) -> OpSpec:
        c_row, c_col = sympy.Symbol("c_row"), sympy.Symbol("c_col")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        coords = [c_col // 64, c_row, sympy.Mod(c_col, 64)]

        def _tensor_arg(is_input, arg_index, hbm_base):
            return TensorArg(
                is_input=is_input,
                arg_index=arg_index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=list(self._DEVICE_SIZE),
                device_coordinates=coords,
                allocation={"hbm": hbm_base},
            )

        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                c_row: (s0, 1),
                c_col: (sympy.Integer(256), 1),
            },
            args=[
                _tensor_arg(True, 0, self._HBM_BASE),
                _tensor_arg(True, 1, self._HBM_BASE + 0x1000),
                _tensor_arg(False, 2, self._HBM_BASE + 0x100000000),
            ],
            op_info={},
            symbolic_dim_bounds={"s0": (512, 64)},  # (max, granularity)
        )

    def test_symbolic_dim_fields_in_sdsc_json(self):
        op_spec = self._make_symbolic_op_spec()
        sdsc_json, _, _, _ = compile_op_spec(idx=0, op_spec=op_spec, symbols=[])

        top = next(iter(sdsc_json.values()))
        dsc = next(iter(top["dscs_"][0].values()))

        # "s0" is registered as dim-symbol id -1 and bound to the SDSC "mb"
        # dim (c_row maps to the first non-output dim label for a 2-dim op).
        self.assertEqual(dsc["dimToSymbolMapping_"], {"mb": [-1]})

        for stage in ("ss_", "el_"):
            sym_info = dsc["dataStageParam_"]["0"][stage]["symbolicDimInfo_"]
            self.assertEqual(sym_info, {"mb": {"maxSize_": 512, "granularity_": 64}})
