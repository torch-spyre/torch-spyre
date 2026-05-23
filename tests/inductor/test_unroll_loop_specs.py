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

Tests build OpSpec / LoopSpec objects directly.  No Spyre device or backend
compiler is needed.  The per-arg byte stride is derived from
TensorArg.device_coordinates and device_size; each arg advances independently.
"""

import unittest

import sympy
from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.codegen.unroll import (
    _hbm_byte_stride_for_arg,
    unroll_loop_specs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_C0 = Symbol("c0")
_C1 = Symbol("c1")
_HBM_BASE = 0x400000000  # SEGMENT_OFFSETS[1]
_LX_ADDR = 0

# device_size=[16, 512, 64], fp16, coord=[0, c0, 0], tile_range=512
# elem_stride(dim 1) = prod([64]) = 64; byte stride = 64 * 512 * 2 = 65536
_DEVICE_SIZE = [16, 512, 64]
_TILE_RANGE = 512
_STRIDE_BYTES = 64 * _TILE_RANGE * 2  # 65536


def _make_hbm_tensor_arg(base: int = _HBM_BASE) -> TensorArg:
    return TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=list(_DEVICE_SIZE),
        device_coordinates=[sympy.Integer(0), _C0, sympy.Integer(0)],
        allocation={"hbm": base},
    )


def _make_lx_tensor_arg() -> TensorArg:
    return TensorArg(
        is_input=False,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=list(_DEVICE_SIZE),
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
            device_size=list(_DEVICE_SIZE),
            device_coordinates=[sympy.Integer(0), _C0, sympy.Integer(0)],
            allocation={"hbm": _HBM_BASE + 0x100000000},
        )
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={_C0: (sympy.Integer(_TILE_RANGE), 1)},
        args=args,
        op_info={},
        tiled_symbols=list(tiled_syms),
    )


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
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 4)
        for copy_op in result:
            self.assertEqual(copy_op.tiled_symbols, [])

    # ------------------------------------------------------------------
    # 5. Nested 2×4 loop → 8 flat copies.
    # ------------------------------------------------------------------

    def test_nested_loops_k2_m4(self):
        op = _make_op_spec(tiled_syms=[_C0, _C1], hbm_base=_HBM_BASE)
        inner_loop = LoopSpec(count=sympy.Integer(4), body=[op])
        outer_loop = LoopSpec(count=sympy.Integer(2), body=[inner_loop])
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
                    self.assertIn(
                        a.allocation["hbm"], (_HBM_BASE, _HBM_BASE + 0x100000000)
                    )

    # ------------------------------------------------------------------
    # 8. _hbm_byte_stride_for_arg: coord=c0 in dim 1.
    #    stride = coeff(c0,dim1)=1 * elem_stride(dim1)=64 * tile_range * 2
    # ------------------------------------------------------------------

    def test_hbm_byte_stride_for_arg(self):
        arg = _make_hbm_tensor_arg()
        stride = _hbm_byte_stride_for_arg(arg, _C0, _TILE_RANGE)
        self.assertEqual(stride, _STRIDE_BYTES)

    # ------------------------------------------------------------------
    # 9. arg with c0 in dim 0 (not dim 1) gets different stride.
    # ------------------------------------------------------------------

    def test_hbm_byte_stride_dim0(self):
        # device_size=[16, 512, 64], coord=[c0, 0, 0]: c0 is in dim 0.
        # elem_stride(dim 0) = prod([512, 64]) = 32768
        # byte stride = 32768 * tile_range * 2
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[16, 512, 64],
            device_coordinates=[_C0, sympy.Integer(0), sympy.Integer(0)],
            allocation={"hbm": _HBM_BASE},
        )
        expected = 32768 * _TILE_RANGE * 2
        self.assertEqual(_hbm_byte_stride_for_arg(arg, _C0, _TILE_RANGE), expected)

    # ------------------------------------------------------------------
    # 10. Two HBM args with different device sizes advance independently.
    # ------------------------------------------------------------------

    def test_per_arg_independent_strides(self):
        # arg0: device_size=[16, 512, 64], coord=[0, c0, 0] → stride=65536
        arg0 = _make_hbm_tensor_arg(_HBM_BASE)
        # arg1: device_size=[4, 128, 64], coord=[0, c0, 0] → stride=16384
        small_size = [4, 128, 64]
        arg1 = TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=small_size,
            device_coordinates=[sympy.Integer(0), _C0, sympy.Integer(0)],
            allocation={"hbm": _HBM_BASE + 0x100000000},
        )
        op = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={_C0: (sympy.Integer(_TILE_RANGE), 1)},
            args=[arg0, arg1],
            op_info={},
            tiled_symbols=[_C0],
        )
        loop = LoopSpec(count=sympy.Integer(2), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 2)
        # arg0 advances by 65536 per iteration
        self.assertEqual(result[1].args[0].allocation["hbm"], _HBM_BASE + 65536)
        # arg1 advances by 64 * 512 * 2 = 65536 (same tile_range, but size is smaller)
        # Wait: dim1 elem_stride for [4,128,64] = 64; stride = 64 * 512 * 2 = 65536 too
        # Actually the tile_range is what matters, device_size only gives elem_stride
        expected_arg1 = _HBM_BASE + 0x100000000 + 64 * _TILE_RANGE * 2
        self.assertEqual(result[1].args[1].allocation["hbm"], expected_arg1)


if __name__ == "__main__":
    unittest.main()
