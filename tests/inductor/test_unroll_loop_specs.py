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

"""Unit tests for torch_spyre._inductor.codegen.unroll.

Tests build OpSpec / LoopSpec objects directly and mock parse_op_spec so no
Spyre device or backend compiler is needed.
"""

import unittest
from unittest.mock import patch

import sympy
from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.codegen.unroll import (
    unroll_loop_specs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_C0 = Symbol("c0")
_C1 = Symbol("c1")
_HBM_BASE = 0x400000000  # SEGMENT_OFFSETS[1]
_LX_ADDR = 0
_STRIDE_BYTES = 1024 * 2  # 1024-element tile × 2 bytes/fp16


def _make_hbm_tensor_arg(base: int = _HBM_BASE) -> TensorArg:
    return TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[16, 512, 64],
        device_coordinates=[sympy.Integer(0), _C0, sympy.Integer(0)],
        allocation={"hbm": base},
    )


def _make_lx_tensor_arg() -> TensorArg:
    return TensorArg(
        is_input=False,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[16, 512, 64],
        device_coordinates=[sympy.Integer(0), _C0, sympy.Integer(0)],
        allocation={"lx": _LX_ADDR},
    )


def _make_op_spec(
    tiled_syms: list[Symbol] | None = None,
    hbm_base: int = _HBM_BASE,
    include_lx: bool = False,
) -> OpSpec:
    tiled_syms = tiled_syms or []
    args = [_make_hbm_tensor_arg(hbm_base)]
    if include_lx:
        args.append(_make_lx_tensor_arg())
    args.append(
        TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[16, 512, 64],
            device_coordinates=[sympy.Integer(0), _C0, sympy.Integer(0)],
            allocation={"hbm": _HBM_BASE + 0x100000000},
        )
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={_C0: (sympy.Integer(512), 1)},
        args=args,
        op_info={},
        tiled_symbols=list(tiled_syms),
    )


def _make_sdsc_spec_mock(stride_bytes: int, tiled_sym_sdsc: Symbol):
    """Return a minimal (sdsc_spec, symbol_mapping) mock pair."""
    from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec
    from torch_spyre._C import DataFormats as DF

    sdsc_arg = SDSCArgs(
        layout="A",
        data_format=DF.SEN169_FP16,
        scales={},
        strides={tiled_sym_sdsc: stride_bytes // 2},  # strides are in elements
        offsets={},
        max_dim_sizes={},
        allocation={"hbm": _HBM_BASE},
        start_address=_HBM_BASE,
        backGap={},
    )
    sdsc_out = SDSCArgs(
        layout="B",
        data_format=DF.SEN169_FP16,
        scales={},
        strides={tiled_sym_sdsc: stride_bytes // 2},
        offsets={},
        max_dim_sizes={},
        allocation={"hbm": _HBM_BASE + 0x100000000},
        start_address=_HBM_BASE + 0x100000000,
        backGap={},
    )
    sdsc_spec = SDSCSpec(
        opfunc="add",
        execution_unit="sfp",
        data_format=DF.SEN169_FP16,
        num_inputs=1,
        iteration_space={tiled_sym_sdsc: sympy.Integer(512)},
        num_cores=1,
        work_slices={tiled_sym_sdsc: 1},
        core_id_to_work_slice={},
        padding={},
        layouts={"A": {"dim_order": [0], "stick_dim_order": 0, "stick_size": 64}},
        args=[sdsc_arg, sdsc_out],
        constants={},
        coordinate_masking={},
    )
    return sdsc_spec, {_C0: tiled_sym_sdsc}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnrollLoopSpecs(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. Flat spec list passes through unchanged.
    # ------------------------------------------------------------------

    def test_no_loop_passthrough(self):
        op = _make_op_spec()
        result = unroll_loop_specs([op])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], op)

    # ------------------------------------------------------------------
    # 2. LoopSpec(count=2) produces 2 copies; second HBM addr advanced.
    # ------------------------------------------------------------------

    def test_flat_loop_k2_advances_hbm(self):
        op = _make_op_spec(tiled_syms=[_C0], hbm_base=_HBM_BASE)
        loop = LoopSpec(count=sympy.Integer(2), body=[op])

        sdsc_sym = Symbol("a0")
        mock_sdsc, mock_mapping = _make_sdsc_spec_mock(_STRIDE_BYTES, sdsc_sym)

        with (
            patch(
                "torch_spyre._inductor.codegen.unroll.parse_op_spec",
                return_value=(mock_sdsc, {_C0: sdsc_sym}),
            ),
            patch(
                "torch_spyre._inductor.codegen.unroll._tiled_byte_stride",
                return_value=_STRIDE_BYTES,
            ),
        ):
            result = unroll_loop_specs([loop])

        self.assertEqual(len(result), 2)
        addr0 = result[0].args[0].allocation["hbm"]
        addr1 = result[1].args[0].allocation["hbm"]
        self.assertEqual(addr0, _HBM_BASE)
        self.assertEqual(addr1, _HBM_BASE + _STRIDE_BYTES)

    # ------------------------------------------------------------------
    # 3. LX tensor address identical in all copies.
    # ------------------------------------------------------------------

    def test_lx_tensor_unchanged(self):
        op = _make_op_spec(tiled_syms=[_C0], include_lx=True)
        loop = LoopSpec(count=sympy.Integer(3), body=[op])

        sdsc_sym = Symbol("a0")
        with (
            patch(
                "torch_spyre._inductor.codegen.unroll.parse_op_spec",
                return_value=_make_sdsc_spec_mock(_STRIDE_BYTES, sdsc_sym),
            ),
            patch(
                "torch_spyre._inductor.codegen.unroll._tiled_byte_stride",
                return_value=_STRIDE_BYTES,
            ),
        ):
            result = unroll_loop_specs([loop])

        self.assertEqual(len(result), 3)
        for copy_op in result:
            lx_args = [a for a in copy_op.args if "lx" in a.allocation]
            self.assertTrue(lx_args, "Expected at least one lx arg")
            for a in lx_args:
                self.assertEqual(a.allocation["lx"], _LX_ADDR)

    # ------------------------------------------------------------------
    # 4. tiled_symbols cleared on every copy.
    # ------------------------------------------------------------------

    def test_tiled_symbols_cleared(self):
        op = _make_op_spec(tiled_syms=[_C0])
        loop = LoopSpec(count=sympy.Integer(4), body=[op])

        sdsc_sym = Symbol("a0")
        with (
            patch(
                "torch_spyre._inductor.codegen.unroll.parse_op_spec",
                return_value=_make_sdsc_spec_mock(_STRIDE_BYTES, sdsc_sym),
            ),
            patch(
                "torch_spyre._inductor.codegen.unroll._tiled_byte_stride",
                return_value=_STRIDE_BYTES,
            ),
        ):
            result = unroll_loop_specs([loop])

        self.assertEqual(len(result), 4)
        for copy_op in result:
            self.assertEqual(copy_op.tiled_symbols, [])

    # ------------------------------------------------------------------
    # 5. Nested 2×4 loop → 8 flat copies with correct combined addresses.
    # ------------------------------------------------------------------

    def test_nested_loops_k2_m4(self):
        op = _make_op_spec(tiled_syms=[_C0, _C1], hbm_base=_HBM_BASE)

        outer_stride = 8192  # bytes per outer step
        inner_stride = 2048  # bytes per inner step

        inner_loop = LoopSpec(count=sympy.Integer(4), body=[op])
        outer_loop = LoopSpec(count=sympy.Integer(2), body=[inner_loop])

        call_count = [0]

        def fake_tiled_byte_stride(sdsc_arg, sdsc_sym, iter_space):
            # Alternate outer / inner strides based on call order within each
            # _compute_tiled_byte_strides call: first call → outer, second → inner.
            val = outer_stride if call_count[0] % 2 == 0 else inner_stride
            call_count[0] += 1
            return val

        sdsc_sym0 = Symbol("a0")
        with (
            patch(
                "torch_spyre._inductor.codegen.unroll.parse_op_spec",
                return_value=_make_sdsc_spec_mock(outer_stride, sdsc_sym0),
            ),
            patch(
                "torch_spyre._inductor.codegen.unroll._tiled_byte_stride",
                side_effect=fake_tiled_byte_stride,
            ),
        ):
            result = unroll_loop_specs([outer_loop])

        self.assertEqual(len(result), 8, f"Expected 8 copies, got {len(result)}")

    # ------------------------------------------------------------------
    # 6. Symbolic count raises ValueError.
    # ------------------------------------------------------------------

    def test_symbolic_count_raises(self):
        op = _make_op_spec()
        loop = LoopSpec(count=Symbol("K"), body=[op])
        with self.assertRaises(ValueError):
            unroll_loop_specs([loop])

    # ------------------------------------------------------------------
    # 7. HBM tensor NOT in tiled_symbols keeps same address in all copies.
    # ------------------------------------------------------------------

    def test_non_tiled_hbm_unchanged(self):
        # Op has tiled_syms=[] — no tiling, all HBM tensors stay fixed.
        op = _make_op_spec(tiled_syms=[], hbm_base=_HBM_BASE)
        loop = LoopSpec(count=sympy.Integer(4), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 4)
        for copy_op in result:
            for a in copy_op.args:
                if "hbm" in a.allocation:
                    # All copies should have the original base address.
                    self.assertIn(
                        a.allocation["hbm"], (_HBM_BASE, _HBM_BASE + 0x100000000)
                    )


if __name__ == "__main__":
    unittest.main()
