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


class TestNestedReductionUnroll(unittest.TestCase):
    """Tests for nested outer-output + inner-reduction tiling.

    Models the bmm (B outer, K inner) scenario introduced in ct-stage2.  The
    key invariant: when the inner K-loop is unrolled, the combine op's accum_buf
    address must stay fixed (same slice for every K iteration).  Only the bmm's
    K-dimension input should advance per K-tile.

    Tensor geometry used throughout:
      accum_buf  [B=2, M=64, N=32] fp16 per outer tile → HBM at ACCUM_BASE
        device_size=[1, 2, 64], stride_map=[64, 64, 1]
        device_coords=[c_col//64, c_b, c_col%64]   (c_b tiles batch within tile)
      k_input    [M=64, K=128] fp16 per K-tile   → HBM at K_BASE (advances per K)
        device_size=[2, 64, 64], stride_map=[64, 64, 1]
        device_coords=[c_col//64, c_row, c_col%64]
      pool_scratch per_tile_fixed=True (bmm intermediate output, stays fixed)
    """

    # --- Symbols ---
    _C_B = Symbol("c_b")  # output batch symbol (appears in accum_buf coords)
    _C_M = Symbol("c_m")  # M rows
    _C_N = Symbol("c_n")  # N cols
    _C_K = Symbol("c_k")  # K reduction symbol

    # --- Base addresses ---
    _ACCUM_BASE = 0x1000000000
    _K_INPUT_BASE = 0x800000000
    _POOL_BASE = 0  # pool allocation; per_tile_fixed

    # Per-tile tensor geometry:
    #   accum_buf: [B=2, N=32] per row — but using simplified 2D layout:
    #     device_size=[1, 2, 64], stride_map=[64, 64, 1]
    #     device_coords=[c_n//64, c_b, c_n%64]
    #   k_input (K-dim): device_size=[2, 64, 64], stride_map=[64, 64, 1]
    #     device_coords=[c_k//64, c_m, c_k%64]
    #     per K-tile stride: 1 tile = 128 K-elems = 2 sticks
    #       byte_stride = 2 * 64 * 2 = 256  (one stick = 64 fp16 = 128 bytes)

    # For accum_buf: advancing 2 batches in c_b direction:
    #   device_coords[1] = c_b; stride_map[1] = 64
    #   byte_stride = 2 * 64 * 2 = 256

    # K-input advance per K-tile (128 K-elems = 2 sticks along c_k):
    #   device_coords[0] = c_k//64; stride_map[0] = 64
    #   byte_stride = (128//64) * 64 * 2 = 256

    def _make_accum_arg(self, base: int = _ACCUM_BASE, per_tile_fixed: bool = False):
        return TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[1, 2, 64],
            device_coordinates=[
                self._C_N // 64,
                self._C_B,
                sympy.Mod(self._C_N, 64),
            ],
            allocation={"hbm": base},
            stride_map=[64, 64, 1],
            per_tile_fixed=per_tile_fixed,
        )

    def _make_k_input_arg(self, base: int = _K_INPUT_BASE):
        return TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64, 64],
            device_coordinates=[
                self._C_K // 64,
                self._C_M,
                sympy.Mod(self._C_K, 64),
            ],
            allocation={"hbm": base},
            stride_map=[64, 64, 1],
        )

    def _make_pool_scratch(self):
        return TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[1, 64, 64],
            device_coordinates=[
                self._C_N // 64,
                self._C_M,
                sympy.Mod(self._C_N, 64),
            ],
            allocation={"pool": self._POOL_BASE},
            stride_map=[64, 64, 1],
            per_tile_fixed=True,
        )

    def _make_bmm_op(self) -> OpSpec:
        """Model bmm partial result: reads k_input, writes pool scratch."""
        return OpSpec(
            op="batchmatmul",
            is_reduction=True,
            iteration_space={
                self._C_B: (Integer(2), 1),
                self._C_M: (Integer(64), 1),
                self._C_N: (Integer(32), 1),
                self._C_K: (Integer(128), 1),
            },
            args=[self._make_k_input_arg(), self._make_pool_scratch()],
            op_info={},
            tiled_symbols=[self._C_B, self._C_K],
        )

    def _make_combine_op(self) -> OpSpec:
        """Model combine (add): reads pool + accum_buf, writes accum_buf.

        Output-only iteration space: no K symbol.
        """
        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                self._C_B: (Integer(2), 1),
                self._C_M: (Integer(64), 1),
                self._C_N: (Integer(32), 1),
            },
            args=[
                self._make_pool_scratch(),  # input: per_tile_fixed scratch
                self._make_accum_arg(),  # input: accum_buf (read)
                self._make_accum_arg(),  # output: accum_buf (write)
            ],
            op_info={},
            tiled_symbols=[self._C_B],
        )

    # ------------------------------------------------------------------
    # 11. Inner K-loop: combine's accum_buf does NOT advance per K-tile.
    #
    #     The LoopSpec for the K-loop carries tiled_symbols=[c_k].  The
    #     combine op's iteration_space has only {c_b, c_m, c_n} — no c_k.
    #     _arg_byte_strides_for_syms must produce stride=0 for accum_buf
    #     across K iterations, so every K-tile combine writes to the same
    #     address.
    # ------------------------------------------------------------------

    def test_combine_accum_fixed_across_k_iterations(self):
        bmm = self._make_bmm_op()
        combine = self._make_combine_op()
        # Inner K-loop: LoopSpec explicitly carries the K symbol.
        inner = LoopSpec(
            count=Integer(4), body=[bmm, combine], tiled_symbols=[self._C_K]
        )
        result = unroll_loop_specs([inner])

        self.assertEqual(len(result), 8, f"Expected 4*2=8 ops, got {len(result)}")
        combine_copies = result[1::2]  # every other op is a combine copy
        self.assertEqual(len(combine_copies), 4)

        # accum_buf is arg index 1 and 2 of the combine; both must stay at ACCUM_BASE.
        for i, copy_op in enumerate(combine_copies):
            for arg_idx in (1, 2):
                addr = copy_op.args[arg_idx].allocation["hbm"]
                self.assertEqual(
                    addr,
                    self._ACCUM_BASE,
                    f"K-iter {i}, combine arg[{arg_idx}] addr={hex(addr)}, "
                    f"expected {hex(self._ACCUM_BASE)} (must not advance per K)",
                )

    # ------------------------------------------------------------------
    # 12. Inner K-loop: pool scratch (per_tile_fixed) stays fixed.
    # ------------------------------------------------------------------

    def test_pool_scratch_fixed_across_k_iterations(self):
        bmm = self._make_bmm_op()
        combine = self._make_combine_op()
        inner = LoopSpec(
            count=Integer(4), body=[bmm, combine], tiled_symbols=[self._C_K]
        )
        result = unroll_loop_specs([inner])

        for op_copy in result:
            for arg in op_copy.args:
                if "pool" in arg.allocation:
                    self.assertEqual(
                        arg.allocation["pool"],
                        self._POOL_BASE,
                        f"pool scratch must stay at {self._POOL_BASE}",
                    )

    # ------------------------------------------------------------------
    # 13. Inner K-loop: bmm's K-input advances per K-tile.
    #
    #     k_input device_coords[0] = c_k//64; stride_map[0]=64; dtype=fp16.
    #     Per K-tile (128 K-elems = 2 sticks):
    #       byte_stride = (128//64) * 64 * 2 = 256
    # ------------------------------------------------------------------

    def test_bmm_k_input_advances_per_k_iteration(self):
        K_TILE_BYTES = (128 // 64) * 64 * 2  # 256
        bmm = self._make_bmm_op()
        combine = self._make_combine_op()
        inner = LoopSpec(
            count=Integer(4), body=[bmm, combine], tiled_symbols=[self._C_K]
        )
        result = unroll_loop_specs([inner])

        bmm_copies = result[::2]  # every other op is a bmm copy
        self.assertEqual(len(bmm_copies), 4)
        for i, copy_op in enumerate(bmm_copies):
            k_arg = copy_op.args[0]  # k_input
            expected = self._K_INPUT_BASE + i * K_TILE_BYTES
            self.assertEqual(
                k_arg.allocation["hbm"],
                expected,
                f"K-iter {i} k_input addr={hex(k_arg.allocation['hbm'])}, "
                f"expected {hex(expected)}",
            )

    # ------------------------------------------------------------------
    # 14. Nested outer-B + inner-K: full 2×4 layout.
    #
    #     Outer B-loop (tiled_symbols=[c_b], count=2):
    #       fill op: writes accum_buf, tiled_symbols=[c_b]
    #       Inner K-loop (tiled_symbols=[c_k], count=4):
    #         bmm + combine
    #
    #     After full unrolling:
    #     outer=2, inner=4 → 2*(1 fill + 4*(bmm+combine)) = 2+16 = 18 ops
    #       flat result: [fill_B0, bmm_K0, add_K0, bmm_K1, add_K1, ... x4, fill_B1, ...]
    #
    #     Key assertions:
    #       - fill_B0 accum addr = ACCUM_BASE
    #       - fill_B1 accum addr = ACCUM_BASE + B_TILE_BYTES
    #       - all add ops in B-tile 0 write to ACCUM_BASE
    #       - all add ops in B-tile 1 write to ACCUM_BASE + B_TILE_BYTES
    # ------------------------------------------------------------------

    def test_nested_outer_b_inner_k_full_layout(self):
        # B-tile byte advance: 2 batches in c_b, stride_map[1]=64, dtype=fp16
        B_TILE_BYTES = 2 * 64 * 2  # 256

        fill_accum = self._make_accum_arg()
        fill_op = OpSpec(
            op="identity",
            is_reduction=False,
            iteration_space={
                self._C_B: (Integer(2), 1),
                self._C_N: (Integer(32), 1),
            },
            args=[fill_accum],
            op_info={},
            tiled_symbols=[self._C_B],
        )

        bmm = self._make_bmm_op()
        combine = self._make_combine_op()

        inner = LoopSpec(
            count=Integer(4), body=[bmm, combine], tiled_symbols=[self._C_K]
        )
        outer = LoopSpec(
            count=Integer(2), body=[fill_op, inner], tiled_symbols=[self._C_B]
        )
        result = unroll_loop_specs([outer])

        # Expected flat layout: 2 × (1 fill + 4 × (1 bmm + 1 combine)) = 18 ops
        self.assertEqual(len(result), 18, f"Expected 18 ops, got {len(result)}")

        # Verify fill ops (indices 0 and 9)
        fill_b0 = result[0]
        fill_b1 = result[9]
        self.assertEqual(fill_b0.op, "identity")
        self.assertEqual(fill_b1.op, "identity")
        self.assertEqual(fill_b0.args[0].allocation["hbm"], self._ACCUM_BASE)
        self.assertEqual(
            fill_b1.args[0].allocation["hbm"], self._ACCUM_BASE + B_TILE_BYTES
        )

        # Verify combine ops in B-tile 0 (indices 2, 4, 6, 8 = ops 1,3,5,7 → add ops)
        # Inner body: [bmm, combine] × 4 = ops 1..8, starting at result[1]
        b0_adds = [result[i] for i in (2, 4, 6, 8)]
        for i, op_copy in enumerate(b0_adds):
            self.assertEqual(op_copy.op, "add", f"result[{2 + 2 * i}] should be add")
            accum_write = op_copy.args[2]
            self.assertEqual(
                accum_write.allocation["hbm"],
                self._ACCUM_BASE,
                f"B-tile0 K-iter {i} combine must write to ACCUM_BASE",
            )

        # Verify combine ops in B-tile 1 (indices 11, 13, 15, 17)
        b1_adds = [result[i] for i in (11, 13, 15, 17)]
        for i, op_copy in enumerate(b1_adds):
            self.assertEqual(op_copy.op, "add", f"result[{11 + 2 * i}] should be add")
            accum_write = op_copy.args[2]
            self.assertEqual(
                accum_write.allocation["hbm"],
                self._ACCUM_BASE + B_TILE_BYTES,
                f"B-tile1 K-iter {i} combine must write to ACCUM_BASE+B_TILE_BYTES",
            )


class TestNestedReductionTileAccum(unittest.TestCase):
    """Verify unroller behaviour for the tile-sized accum buffer pattern.

    Pattern (outer B=2 tiles, inner K=4 tiles):
      outer LoopSpec(count=2, tiled_symbols=[c_b]):
        fill: output=accum_tile (per_tile_fixed=True)
        inner LoopSpec(count=4, tiled_symbols=[c_k]):
          bmm partial: K-input advances; output per_tile_fixed=True
          combine: both args=accum_tile (per_tile_fixed=True)
        copy: input=accum_tile (per_tile_fixed=True),
              output=accum_full (advances per outer B-tile)

    After full unrolling:
      outer=2, inner=4 → 2*(1 fill + 4*(bmm+combine) + 1 copy) = 20 ops
    """

    _C_B = Symbol("c_b")
    _C_M = Symbol("c_m")
    _C_N = Symbol("c_n")
    _C_K = Symbol("c_k")

    _ACCUM_TILE_BASE = 0x0  # per_tile_fixed: always stays at 0
    _ACCUM_FULL_BASE = 0x1000000000  # advances per outer B-tile

    # accum_tile: [64, 32] fp16 per tile; simple device layout
    _TILE_DEVICE_SIZE = [1, 64, 32]  # 1 stick-group, 64 rows, 32 cols
    _TILE_STRIDE_MAP = [2048, 32, 1]

    # accum_full: [128, 32] fp16 = 2 tiles × [64, 32]; c_b in device coords
    # stride_map[0] = 64*32 = 2048 elements per B-tile
    # byte stride per outer tile = 1 * 2048 * 2 = 4096
    _FULL_DEVICE_SIZE = [1, 128, 32]
    _FULL_STRIDE_MAP = [2048, 32, 1]
    _OUTER_TILE_STRIDE_BYTES = 1 * 2048 * 2  # 4096

    def _make_accum_tile_arg(self) -> TensorArg:
        return TensorArg(
            is_input=True,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(self._TILE_DEVICE_SIZE),
            device_coordinates=[Integer(0), self._C_M, self._C_N],
            allocation={"hbm": self._ACCUM_TILE_BASE},
            stride_map=list(self._TILE_STRIDE_MAP),
            per_tile_fixed=True,
        )

    def _make_accum_full_arg(self) -> TensorArg:
        # c_b in device_coordinates so outer-loop unroller can compute byte stride.
        return TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(self._FULL_DEVICE_SIZE),
            device_coordinates=[self._C_B, self._C_M, self._C_N],
            allocation={"hbm": self._ACCUM_FULL_BASE},
            stride_map=list(self._FULL_STRIDE_MAP),
            per_tile_fixed=False,
        )

    def _make_fill_op(self) -> OpSpec:
        """Fill: zeros accum_tile once per outer B-tile."""
        return OpSpec(
            op="fill",
            is_reduction=False,
            iteration_space={
                self._C_B: (Integer(1), 1),
                self._C_M: (Integer(64), 1),
                self._C_N: (Integer(32), 1),
            },
            args=[self._make_accum_tile_arg()],
            op_info={},
            tiled_symbols=[],
        )

    def _make_copy_op(self) -> OpSpec:
        """Copy: accum_tile → accum_full after inner K-loop."""
        return OpSpec(
            op="copy",
            is_reduction=False,
            iteration_space={
                self._C_B: (Integer(1), 1),
                self._C_M: (Integer(64), 1),
                self._C_N: (Integer(32), 1),
            },
            args=[
                self._make_accum_tile_arg(),  # input: per_tile_fixed, never advances
                self._make_accum_full_arg(),  # output: advances per outer B-tile
            ],
            op_info={},
            tiled_symbols=[],
        )

    def _make_nested_loop(self) -> LoopSpec:
        bmm_input = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64, 64],
            device_coordinates=[
                self._C_K // 64,
                self._C_M,
                sympy.Mod(self._C_K, 64),
            ],
            allocation={"hbm": 0x800000000},
            stride_map=[64, 512, 1],
            per_tile_fixed=False,
        )
        bmm_output = TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(self._TILE_DEVICE_SIZE),
            device_coordinates=[Integer(0), self._C_M, self._C_N],
            allocation={"hbm": 0x2000000000},
            stride_map=list(self._TILE_STRIDE_MAP),
            per_tile_fixed=True,
        )
        bmm_partial = OpSpec(
            op="matmul",
            is_reduction=True,
            iteration_space={
                self._C_M: (Integer(64), 1),
                self._C_N: (Integer(32), 1),
                self._C_K: (Integer(128), 1),
            },
            args=[bmm_input, bmm_output],
            op_info={},
            tiled_symbols=[self._C_K],
        )
        combine_op = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                self._C_M: (Integer(64), 1),
                self._C_N: (Integer(32), 1),
            },
            args=[self._make_accum_tile_arg(), self._make_accum_tile_arg()],
            op_info={},
            tiled_symbols=[],
        )
        inner = LoopSpec(
            count=Integer(4),
            body=[bmm_partial, combine_op],
            tiled_symbols=[self._C_K],
        )
        return LoopSpec(
            count=Integer(2),
            body=[self._make_fill_op(), inner, self._make_copy_op()],
            tiled_symbols=[self._C_B],
        )

    def test_accum_tile_fixed_accum_full_advances(self):
        """accum_tile never advances; accum_full advances by outer-tile stride each B iter."""
        loop = self._make_nested_loop()
        result = unroll_loop_specs([loop])
        # 2 * (1 fill + 4*(bmm+combine) + 1 copy) = 20 ops
        self.assertEqual(len(result), 20, f"Expected 20 ops, got {len(result)}")
        # Positions: fill(0), bmm(1),comb(2),bmm(3),comb(4),bmm(5),comb(6),bmm(7),comb(8),
        #            copy(9), fill(10), bmm(11)..copy(19)
        copy_0 = result[9]
        copy_1 = result[19]
        # accum_tile input (index 0): per_tile_fixed, must never advance
        self.assertEqual(copy_0.args[0].allocation["hbm"], self._ACCUM_TILE_BASE)
        self.assertEqual(copy_1.args[0].allocation["hbm"], self._ACCUM_TILE_BASE)
        # accum_full output (index 1): advances by one tile per outer B iteration
        self.assertEqual(copy_0.args[1].allocation["hbm"], self._ACCUM_FULL_BASE)
        self.assertEqual(
            copy_1.args[1].allocation["hbm"],
            self._ACCUM_FULL_BASE + self._OUTER_TILE_STRIDE_BYTES,
        )

    def test_fill_always_targets_tile_base(self):
        """fill op always targets accum_tile (per_tile_fixed) regardless of outer tile."""
        loop = self._make_nested_loop()
        result = unroll_loop_specs([loop])
        fill_0 = result[0]
        fill_1 = result[10]
        self.assertEqual(fill_0.args[0].allocation["hbm"], self._ACCUM_TILE_BASE)
        self.assertEqual(fill_1.args[0].allocation["hbm"], self._ACCUM_TILE_BASE)


if __name__ == "__main__":
    unittest.main()
