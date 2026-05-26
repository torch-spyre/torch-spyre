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

Tests build OpSpec / LoopSpec objects directly using realistic stick-layout
TensorArgs.  No Spyre device or backend compiler is needed.

Stick layout reference for a 2D fp16 tensor shaped [R, C] (C a multiple of 64):
  device_size        = [C//64, R, 64]      # [sticks_per_row, rows, elems_per_stick]
  stride_map         = [64, C, 1]          # elems per stick-col advance, row advance, within-stick
  device_coordinates = [c_col//64, c_row, c_col%64]

All fixtures use a [512, 256] fp16 tensor:
  device_size = [4, 512, 64],  stride_map = [64, 256, 1]

Tiling by c_row (T_ROW rows per iteration):
  byte_stride = T_ROW * stride_map[1] * 2 = T_ROW * 256 * 2

Tiling by c_col (T_COL elements per iteration, T_COL a multiple of 64):
  byte_stride = (T_COL // 64) * stride_map[0] * 2 = (T_COL // 64) * 64 * 2
"""

import unittest

import sympy
from sympy import Integer, Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.codegen.unroll import (
    _byte_stride_for_arg,
    unroll_loop_specs,
)

# ---------------------------------------------------------------------------
# Fixtures: [512, 256] fp16 tensor in stick layout
# ---------------------------------------------------------------------------

_C_ROW = Symbol("c_row")
_C_COL = Symbol("c_col")
_HBM_BASE = 0x400000000  # SEGMENT_OFFSETS[1]
_LX_ADDR = 0

# [512, 256] fp16 → device_size=[4, 512, 64], stride_map=[64, 256, 1]
_DEVICE_SIZE = [4, 512, 64]
_STRIDE_MAP = [64, 256, 1]
# Row tile: advance 512 rows; byte stride = 512 * 256 * 2
_T_ROW = 512
_STRIDE_BYTES = _T_ROW * 256 * 2  # 262144


def _device_coords():
    """Stick-layout device coordinates for the [512, 256] fixture tensor."""
    return [_C_COL // 64, _C_ROW, sympy.Mod(_C_COL, 64)]


def _make_hbm_tensor_arg(base: int = _HBM_BASE) -> TensorArg:
    return TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=list(_DEVICE_SIZE),
        device_coordinates=_device_coords(),
        allocation={"hbm": base},
        stride_map=list(_STRIDE_MAP),
    )


def _make_lx_tensor_arg() -> TensorArg:
    # per_tile_fixed=True: tile-local scratch reused every iteration.
    return TensorArg(
        is_input=False,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=list(_DEVICE_SIZE),
        device_coordinates=_device_coords(),
        allocation={"lx": _LX_ADDR},
        stride_map=list(_STRIDE_MAP),
        per_tile_fixed=True,
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
            device_coordinates=_device_coords(),
            allocation={"hbm": _HBM_BASE + 0x100000000},
            stride_map=list(_STRIDE_MAP),
        )
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={
            _C_ROW: (Integer(_T_ROW), 1),
            _C_COL: (Integer(256), 1),
        },
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
    #    Tiling c_row with T_ROW=512: byte_stride = 512 * 256 * 2 = 262144
    # ------------------------------------------------------------------

    def test_flat_loop_k2_advances_hbm(self):
        op = _make_op_spec(tiled_syms=[_C_ROW], hbm_base=_HBM_BASE)
        loop = LoopSpec(count=Integer(2), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 2)
        addr0 = result[0].args[0].allocation["hbm"]
        addr1 = result[1].args[0].allocation["hbm"]
        self.assertEqual(addr0, _HBM_BASE)
        self.assertEqual(addr1, _HBM_BASE + _STRIDE_BYTES)

    # ------------------------------------------------------------------
    # 3. per_tile_fixed LX tensor address identical in all copies.
    #    The lx arg has per_tile_fixed=True (tile-local scratch), so its
    #    address must not advance regardless of allocation type.
    # ------------------------------------------------------------------

    def test_lx_tensor_unchanged(self):
        op = _make_op_spec(tiled_syms=[_C_ROW], include_lx=True)
        loop = LoopSpec(count=Integer(3), body=[op])
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
        op = _make_op_spec(tiled_syms=[_C_ROW])
        loop = LoopSpec(count=Integer(4), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 4)
        for copy_op in result:
            self.assertEqual(copy_op.tiled_symbols, [])

    # ------------------------------------------------------------------
    # 5. Nested 2×4 loop → 8 flat copies.
    # ------------------------------------------------------------------

    def test_nested_loops_k2_m4(self):
        op = _make_op_spec(tiled_syms=[_C_ROW, _C_COL], hbm_base=_HBM_BASE)
        inner_loop = LoopSpec(count=Integer(4), body=[op])
        outer_loop = LoopSpec(count=Integer(2), body=[inner_loop])
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
        loop = LoopSpec(count=Integer(4), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 4)
        for copy_op in result:
            for a in copy_op.args:
                if "hbm" in a.allocation:
                    self.assertIn(
                        a.allocation["hbm"], (_HBM_BASE, _HBM_BASE + 0x100000000)
                    )

    # ------------------------------------------------------------------
    # 8. _byte_stride_for_arg: tiling c_row (row dimension).
    #    coord[1] = c_row; stride_map[1] = 256; tile_range = 512
    #    byte_stride = 512 * 256 * 2 = 262144
    # ------------------------------------------------------------------

    def test_byte_stride_for_arg(self):
        arg = _make_hbm_tensor_arg()
        stride = _byte_stride_for_arg(arg, _C_ROW, _T_ROW)
        self.assertEqual(stride, _STRIDE_BYTES)

    # ------------------------------------------------------------------
    # 9. _byte_stride_for_arg: tiling c_col (column dimension).
    #    coord[0] = c_col//64 (sticks_per_row), coord[2] = c_col%64 (within-stick).
    #    Advancing by T_COL=128 elements (2 sticks):
    #      delta[0] = 128//64 = 2; delta[2] = 128%64 = 0
    #      byte_stride = 2 * stride_map[0] * 2 = 2 * 64 * 2 = 256
    # ------------------------------------------------------------------

    def test_hbm_byte_stride_col_dim(self):
        arg = _make_hbm_tensor_arg()
        t_col = 128  # 2 sticks
        expected = (t_col // 64) * _STRIDE_MAP[0] * 2  # 2 * 64 * 2 = 256
        self.assertEqual(_byte_stride_for_arg(arg, _C_COL, t_col), expected)

    # ------------------------------------------------------------------
    # 10. Two HBM args with different tensor shapes advance independently.
    #     arg0: [512, 256] fp16, stride_map=[64, 256, 1] → row stride = 256*2
    #     arg1: [512, 128] fp16, stride_map=[64, 128, 1] → row stride = 128*2
    #     Tiling c_row with T_ROW=512:
    #       arg0 byte_stride = 512 * 256 * 2 = 262144
    #       arg1 byte_stride = 512 * 128 * 2 = 131072
    # ------------------------------------------------------------------

    def test_per_arg_independent_strides(self):
        arg0 = _make_hbm_tensor_arg(_HBM_BASE)
        # [512, 128] fp16: device_size=[2, 512, 64], stride_map=[64, 128, 1]
        arg1 = TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 512, 64],
            device_coordinates=[_C_COL // 64, _C_ROW, sympy.Mod(_C_COL, 64)],
            allocation={"hbm": _HBM_BASE + 0x100000000},
            stride_map=[64, 128, 1],
        )
        op = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                _C_ROW: (Integer(_T_ROW), 1),
                _C_COL: (Integer(128), 1),
            },
            args=[arg0, arg1],
            op_info={},
            tiled_symbols=[_C_ROW],
        )
        loop = LoopSpec(count=Integer(2), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 2)
        # arg0: [512, 256], byte_stride = 512 * 256 * 2 = 262144
        self.assertEqual(result[1].args[0].allocation["hbm"], _HBM_BASE + 512 * 256 * 2)
        # arg1: [512, 128], byte_stride = 512 * 128 * 2 = 131072
        self.assertEqual(
            result[1].args[1].allocation["hbm"],
            _HBM_BASE + 0x100000000 + 512 * 128 * 2,
        )


if __name__ == "__main__":
    unittest.main()
