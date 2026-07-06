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
  device_coordinates = [c_col//64, c_row, c_col%64]

All fixtures use a [512, 256] fp16 tensor:
  device_size = [4, 512, 64]

Tiling by c_row (T_ROW rows per iteration):
  byte_stride = T_ROW * device_stride[1] * 2 = T_ROW * 64 * 2

Tiling by c_col (T_COL elements per iteration, T_COL a multiple of 64):
  byte_stride = (T_COL // 64) * device_stride[0] * 2 = (T_COL // 64) * (512 * 64) * 2
"""

import unittest

import sympy
from sympy import Integer, Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.codegen.unroll import (
    _byte_stride_for_arg,
    _tile_device_size,
    unroll_loop_specs,
)

# ---------------------------------------------------------------------------
# Fixtures: [512, 256] fp16 tensor in stick layout
# ---------------------------------------------------------------------------

_C_ROW = Symbol("c_row")
_C_COL = Symbol("c_col")
_HBM_BASE = 0x400000000  # SEGMENT_OFFSETS[1]
_LX_ADDR = 0

# [512, 256] fp16 → device_size=[4, 512, 64]
_DEVICE_SIZE = [4, 512, 64]
# Row tile: advance 512 rows; device_stride[1]=64; byte stride = 512 * 64 * 2
_T_ROW = 512
_STRIDE_BYTES = _T_ROW * 64 * 2  # 65536


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
    #    Tiling c_row with T_ROW=512: byte_stride = 512 * 64 * 2 = 65536
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
    #    coord[1] = c_row; device_stride[1] = 64; tile_range = 512
    #    byte_stride = 512 * 64 * 2 = 65536
    # ------------------------------------------------------------------

    def test_byte_stride_for_arg(self):
        arg = _make_hbm_tensor_arg()
        stride = _byte_stride_for_arg(arg, _C_ROW, _T_ROW)
        self.assertEqual(stride, _STRIDE_BYTES)

    # ------------------------------------------------------------------
    # 9. _byte_stride_for_arg: tiling c_col (column dimension).
    #    coord[0] = c_col//64 (sticks_per_row), coord[2] = c_col%64 (within-stick).
    #    Advancing by T_COL=128 elements (2 sticks):
    #      delta[0] = 128//64 = 2; device_stride[0] = prod([512, 64]) = 32768
    #      byte_stride = 2 * 32768 * 2 = 131072
    # ------------------------------------------------------------------

    def test_hbm_byte_stride_col_dim(self):
        arg = _make_hbm_tensor_arg()
        t_col = 128  # 2 sticks
        # device_stride[0] = prod(device_size[1:]) = 512 * 64 = 32768
        expected = (t_col // 64) * (512 * 64) * 2  # 2 * 32768 * 2 = 131072
        self.assertEqual(_byte_stride_for_arg(arg, _C_COL, t_col), expected)

    # ------------------------------------------------------------------
    # 10. Two HBM args with different tensor shapes advance independently.
    #     arg0: [512, 256] fp16, device_size=[4, 512, 64] → device_stride[1] = 64
    #     arg1: [512, 128] fp16, device_size=[2, 512, 64] → device_stride[1] = 64
    #     Tiling c_row with T_ROW=512:
    #       arg0 byte_stride = 512 * 64 * 2 = 65536
    #       arg1 byte_stride = 512 * 64 * 2 = 65536
    # ------------------------------------------------------------------

    def test_per_arg_independent_strides(self):
        arg0 = _make_hbm_tensor_arg(_HBM_BASE)
        # [512, 128] fp16: device_size=[2, 512, 64]
        arg1 = TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 512, 64],
            device_coordinates=[_C_COL // 64, _C_ROW, sympy.Mod(_C_COL, 64)],
            allocation={"hbm": _HBM_BASE + 0x100000000},
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
        # arg0: [512, 256], device_stride[1]=64; byte_stride = 512 * 64 * 2 = 65536
        self.assertEqual(result[1].args[0].allocation["hbm"], _HBM_BASE + 512 * 64 * 2)
        # arg1: [512, 128], device_stride[1]=64; byte_stride = 512 * 64 * 2 = 65536
        self.assertEqual(
            result[1].args[1].allocation["hbm"],
            _HBM_BASE + 0x100000000 + 512 * 64 * 2,
        )


class TestNestedReductionUnroll(unittest.TestCase):
    """Tests for nested outer-output + inner-reduction tiling.

    Models the bmm (B outer, K inner) scenario introduced in ct-stage2.  The
    key invariant: when the inner K-loop is unrolled, the combine op's accum_buf
    address must stay fixed (same slice for every K iteration).  Only the bmm's
    K-dimension input should advance per K-tile.

    Tensor geometry used throughout:
      accum_buf  [B=2, M=64, N=32] fp16 per outer tile → HBM at ACCUM_BASE
        device_size=[1, 2, 64], device_stride=[128, 64, 1]
        device_coords=[c_col//64, c_b, c_col%64]   (c_b tiles batch within tile)
      k_input    [M=64, K=128] fp16 per K-tile   → HBM at K_BASE (advances per K)
        device_size=[2, 64, 64], device_stride=[4096, 64, 1]
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
    #     device_size=[1, 2, 64], device_stride=[128, 64, 1]
    #     device_coords=[c_n//64, c_b, c_n%64]
    #   k_input (K-dim): device_size=[2, 64, 64], device_stride=[4096, 64, 1]
    #     device_coords=[c_k//64, c_m, c_k%64]
    #     per K-tile stride: 1 tile = 128 K-elems = 2 sticks
    #       byte_stride = (128//64) * 4096 * 2 = 16384  (one stick = 64 fp16 = 128 bytes)

    # For accum_buf: advancing 2 batches in c_b direction:
    #   device_coords[1] = c_b; device_stride[1] = 64
    #   byte_stride = 2 * 64 * 2 = 256

    # K-input advance per K-tile (128 K-elems = 2 sticks along c_k):
    #   device_coords[0] = c_k//64; device_stride[0] = 64*64 = 4096
    #   byte_stride = (128//64) * 4096 * 2 = 16384

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
    #     k_input device_size=[2, 64, 64]; device_stride[0] = 64*64 = 4096; dtype=fp16.
    #     Per K-tile (128 K-elems = 2 sticks):
    #       byte_stride = (128//64) * (64*64) * 2 = 2 * 4096 * 2 = 16384
    # ------------------------------------------------------------------

    def test_bmm_k_input_advances_per_k_iteration(self):
        # k_input device_size=[2, 64, 64]; device_stride[0] = 64*64 = 4096
        # T_K=128 → delta[0]=2; byte_stride = 2 * 4096 * 2 = 16384
        K_TILE_BYTES = (128 // 64) * (64 * 64) * 2  # 16384
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
        # B-tile byte advance: 2 batches in c_b, device_stride[1]=64, dtype=fp16
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

    # accum_full: [128, 32] fp16 = 2 tiles × [64, 32]; c_b in device coords
    # device_size=[1, 128, 32]; device_stride[0] = 128*32 = 4096
    # byte stride per outer tile = 1 * 4096 * 2 = 8192
    _FULL_DEVICE_SIZE = [1, 128, 32]
    _OUTER_TILE_STRIDE_BYTES = 1 * 4096 * 2  # 8192

    def _make_accum_tile_arg(self) -> TensorArg:
        return TensorArg(
            is_input=True,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(self._TILE_DEVICE_SIZE),
            device_coordinates=[Integer(0), self._C_M, self._C_N],
            allocation={"hbm": self._ACCUM_TILE_BASE},
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
            per_tile_fixed=False,
        )
        bmm_output = TensorArg(
            is_input=False,
            arg_index=-1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(self._TILE_DEVICE_SIZE),
            device_coordinates=[Integer(0), self._C_M, self._C_N],
            allocation={"hbm": 0x2000000000},
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


class TestDeviceStrideFormula(unittest.TestCase):
    """Unit tests that verify _byte_stride_for_arg uses device_stride, not stride_map.

    These tests are parametrised over tensor shapes where stride_map and
    device_stride diverge, so they would fail with the old (wrong) formula.

    Fixture: [R, C] fp16 col-stick layout.
      device_size = [C//64, R, 64]
      device_coordinates = [c_col//64, c_row, Mod(c_col, 64)]
      device_stride[0] = R * 64   (advancing one stick group steps over R rows)
      device_stride[1] = 64       (advancing one row steps over 64 elems)
      device_stride[2] = 1
    """

    _C_COL = Symbol("c_col")
    _C_ROW = Symbol("c_row")

    def _make_arg(self, R: int, C: int, base: int = 0) -> TensorArg:
        return TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[C // 64, R, 64],
            device_coordinates=[
                self._C_COL // 64,
                self._C_ROW,
                sympy.Mod(self._C_COL, 64),
            ],
            allocation={"hbm": base},
        )

    # ------------------------------------------------------------------
    # Row-tiling: advancing T_ROW rows.
    # Correct: T_ROW * device_stride[1] * 2 = T_ROW * 64 * 2
    # Wrong (old formula): T_ROW * stride_map[1] * 2 = T_ROW * C * 2
    # These diverge whenever C != 64 (i.e. sticks_per_row > 1).
    # ------------------------------------------------------------------

    def test_row_tile_stride_narrow_tensor(self):
        """Row-tiling [512, 64] fp16: 1 stick per row, both formulas agree."""
        R, C, T_ROW = 512, 64, 128
        arg = self._make_arg(R, C)
        expected = T_ROW * 64 * 2  # device_stride[1]=64; = 16384
        self.assertEqual(_byte_stride_for_arg(arg, self._C_ROW, T_ROW), expected)

    def test_row_tile_stride_wide_tensor(self):
        """Row-tiling [512, 256] fp16: 4 sticks/row; old formula gave 4x too large."""
        R, C, T_ROW = 512, 256, 128
        arg = self._make_arg(R, C)
        # device_stride[1] = 64 regardless of C
        expected = T_ROW * 64 * 2  # 128 * 64 * 2 = 16384
        # old (wrong): T_ROW * C * 2 = 128 * 256 * 2 = 65536
        self.assertEqual(_byte_stride_for_arg(arg, self._C_ROW, T_ROW), expected)

    def test_row_tile_stride_very_wide_tensor(self):
        """Row-tiling [1024, 4096] fp16: 64 sticks/row; old formula was 64x too large."""
        R, C, T_ROW = 1024, 4096, 256
        arg = self._make_arg(R, C)
        expected = T_ROW * 64 * 2  # 256 * 64 * 2 = 32768
        # old (wrong): 256 * 4096 * 2 = 2097152
        self.assertEqual(_byte_stride_for_arg(arg, self._C_ROW, T_ROW), expected)

    # ------------------------------------------------------------------
    # Col-tiling: advancing T_COL elements (T_COL must be a multiple of 64).
    # delta[0] = T_COL // 64 (stick groups), delta[1]=0, delta[2]=0
    # Correct: (T_COL//64) * device_stride[0] * 2 = (T_COL//64) * R * 64 * 2
    # Wrong (old formula): (T_COL//64) * stride_map[0] * 2 = (T_COL//64) * 64 * 2
    # ------------------------------------------------------------------

    def test_col_tile_stride_one_stick(self):
        """Col-tiling [512, 256] by T_COL=64 (1 stick): correct advance."""
        R, C, T_COL = 512, 256, 64
        arg = self._make_arg(R, C)
        # delta[0] = 1; device_stride[0] = R * 64 = 32768
        expected = 1 * (R * 64) * 2  # 65536
        # old (wrong): 1 * 64 * 2 = 128
        self.assertEqual(_byte_stride_for_arg(arg, self._C_COL, T_COL), expected)

    def test_col_tile_stride_two_sticks(self):
        """Col-tiling [512, 256] by T_COL=128 (2 sticks): correct advance."""
        R, C, T_COL = 512, 256, 128
        arg = self._make_arg(R, C)
        # delta[0] = 2; device_stride[0] = 32768
        expected = 2 * (R * 64) * 2  # 131072
        # old (wrong): 2 * 64 * 2 = 256
        self.assertEqual(_byte_stride_for_arg(arg, self._C_COL, T_COL), expected)

    # ------------------------------------------------------------------
    # End-to-end unroll: row-tiled LoopSpec with multi-stick tensor.
    # Verifies that the correct address advances appear in the unrolled copies.
    # ------------------------------------------------------------------

    def test_unroll_row_tile_multi_stick_addresses(self):
        """Unrolling a row-tile loop over [1024, 256] fp16 gives correct HBM advances.

        [1024, 256] fp16: device_size=[4, 1024, 64], T_ROW=256, count=4.
        device_stride[1] = 64; byte_stride = 256 * 64 * 2 = 32768.
        Tile addresses: base, base+32768, base+65536, base+98304.
        """
        R, C, T_ROW = 1024, 256, 256
        c_row = Symbol("c_row")
        c_col = Symbol("c_col")
        base = 0x400000000
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[C // 64, R, 64],
            device_coordinates=[c_col // 64, c_row, sympy.Mod(c_col, 64)],
            allocation={"hbm": base},
        )
        op = OpSpec(
            op="abs",
            is_reduction=False,
            iteration_space={
                c_row: (Integer(T_ROW), 1),
                c_col: (Integer(C), 1),
            },
            args=[arg],
            op_info={},
            tiled_symbols=[c_row],
        )
        loop = LoopSpec(count=Integer(4), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), 4)
        tile_stride = T_ROW * 64 * 2  # 32768
        for i, copy_op in enumerate(result):
            expected_addr = base + i * tile_stride
            self.assertEqual(
                copy_op.args[0].allocation["hbm"],
                expected_addr,
                f"tile {i}: expected {hex(expected_addr)}, "
                f"got {hex(copy_op.args[0].allocation['hbm'])}",
            )


class TestTileDeviceSize(unittest.TestCase):
    """Unit tests for _tile_device_size covering all key tiling patterns.

    The function computes the device_size that describes the tile geometry
    after loop unrolling.  The key invariant: device_size[d] may only be
    shrunk to tile extent when all outer dims d' < d are degenerate (size 1).
    Shrinking a dimension that contributes to an outer dim's physical row
    stride corrupts that stride.

    Layout conventions used throughout:
      col-stick [R, C] fp16:
        device_size=[C//64, R, 64], device_coordinates=[c1//64, c0, Mod(c1,64)]
        d=0: sticks_per_row (C//64), outer stride contributor
        d=1: rows (R), inner stride
        d=2: within-stick (64)
      row-stick [R, C] fp16 (after restickify):
        device_size=[R//64, C, 64], device_coordinates=[c0//64, c1, Mod(c0,64)]
        d=0: sticks_per_col (R//64), outer stride contributor
        d=1: cols (C), inner stride
        d=2: within-stick (64)
    """

    _C0 = Symbol("c0")
    _C1 = Symbol("c1")
    _FP16 = DataFormats.SEN169_FP16

    def _col_stick_arg(self, R: int, C: int) -> TensorArg:
        """[R, C] fp16 col-stick: device_size=[C//64, R, 64]."""
        return TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=self._FP16,
            device_size=[C // 64, R, 64],
            device_coordinates=[self._C1 // 64, self._C0, sympy.Mod(self._C1, 64)],
            allocation={"hbm": 0},
        )

    def _row_stick_arg(self, R: int, C: int) -> TensorArg:
        """[R, C] fp16 row-stick: device_size=[R//64, C, 64]."""
        return TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=self._FP16,
            device_size=[R // 64, C, 64],
            device_coordinates=[self._C0 // 64, self._C1, sympy.Mod(self._C0, 64)],
            allocation={"hbm": 0},
        )

    # ------------------------------------------------------------------
    # Col-stick, tile the row dimension (c0).
    #
    # BUG case: old code shrank device_size[1] from R to T_ROW, but d=1
    # has a non-degenerate outer dim at d=0 (C//64 > 1 when C > 64).
    # The hardware uses device_size[1] as the inter-stick-group row stride;
    # shrinking it breaks every stick group after the first.
    # ------------------------------------------------------------------

    def test_col_stick_row_tiling_multi_stick_preserves_d1(self):
        """Col-stick multi-stick tensor: tiling c0 (rows) must NOT shrink device_size[1].

        [1024, 4096] fp16: device_size=[64, 1024, 64].
        Tiling c0 by T_ROW=512: tile has 512 rows.
        device_size[0]=64 > 1 → outer dim is non-degenerate → d=1 must stay at 1024.
        """
        R, C, T_ROW = 1024, 4096, 512
        arg = self._col_stick_arg(R, C)
        it_space = {self._C0: (Integer(T_ROW), 1), self._C1: (Integer(C), 1)}
        result = _tile_device_size(arg, [self._C0], it_space)
        # d=0: c0 has no delta in floor(c1/64) → unchanged
        self.assertEqual(result[0], C // 64, "sticks_per_row must not change")
        # d=1: must stay at full R, not shrink to T_ROW
        self.assertEqual(result[1], R, f"device_size[1] must stay {R}, got {result[1]}")
        self.assertEqual(result[2], 64, "within-stick size must not change")

    def test_col_stick_row_tiling_single_stick_shrinks_d1(self):
        """Col-stick single-stick tensor: tiling c0 (rows) MAY shrink device_size[1].

        [1024, 64] fp16: device_size=[1, 1024, 64].
        device_size[0]=1 → outer dim is degenerate → d=1 may be shrunk to T_ROW.
        """
        R, C, T_ROW = 1024, 64, 256
        arg = self._col_stick_arg(R, C)
        it_space = {self._C0: (Integer(T_ROW), 1), self._C1: (Integer(C), 1)}
        result = _tile_device_size(arg, [self._C0], it_space)
        self.assertEqual(result[0], 1, "single stick group must not change")
        self.assertEqual(result[1], T_ROW, f"device_size[1] must shrink to {T_ROW}")
        self.assertEqual(result[2], 64)

    # ------------------------------------------------------------------
    # Col-stick, tile the column dimension (c1).
    #
    # c1 appears in both floor(c1/64) (d=0) and Mod(c1,64) (d=2, the
    # within-stick coord).  Since c1 is in stick_syms it must be excluded
    # entirely — device_size is unchanged.
    # ------------------------------------------------------------------

    def test_col_stick_col_tiling_excluded_as_stick_sym(self):
        """Col-stick: tiling c1 (the stick symbol) leaves device_size unchanged.

        c1 appears in Mod(c1,64), the within-stick coordinate, so it is a
        stick symbol and must not affect device_size at all.
        """
        R, C, T_COL = 1024, 4096, 1024
        arg = self._col_stick_arg(R, C)
        it_space = {self._C0: (Integer(R), 1), self._C1: (Integer(T_COL), 1)}
        result = _tile_device_size(arg, [self._C1], it_space)
        self.assertEqual(result, [C // 64, R, 64])

    # ------------------------------------------------------------------
    # Row-stick, tile the column dimension (c1).
    #
    # After restickify, c1 is the non-stick (inner stride) dimension.
    # device_size=[R//64, C, 64]; d=0 sticks_per_col = R//64.
    # If R//64 > 1, tiling c1 must not shrink d=1.
    # ------------------------------------------------------------------

    def test_row_stick_col_tiling_multi_stick_preserves_d1(self):
        """Row-stick multi-stick tensor: tiling c1 (cols) must NOT shrink device_size[1].

        [1024, 4096] fp16 row-stick: device_size=[16, 4096, 64].
        Tiling c1 by T_COL=2048: device_size[0]=16 > 1 → d=1 must stay at 4096.
        """
        R, C, T_COL = 1024, 4096, 2048
        arg = self._row_stick_arg(R, C)
        it_space = {self._C0: (Integer(R), 1), self._C1: (Integer(T_COL), 1)}
        result = _tile_device_size(arg, [self._C1], it_space)
        self.assertEqual(result[0], R // 64, "sticks_per_col must not change")
        self.assertEqual(result[1], C, f"device_size[1] must stay {C}, got {result[1]}")
        self.assertEqual(result[2], 64)

    def test_row_stick_col_tiling_single_stick_shrinks_d1(self):
        """Row-stick single-stick tensor: tiling c1 (cols) MAY shrink device_size[1].

        [64, 4096] fp16 row-stick: device_size=[1, 4096, 64].
        device_size[0]=1 → d=1 may shrink to T_COL.
        """
        R, C, T_COL = 64, 4096, 1024
        arg = self._row_stick_arg(R, C)
        it_space = {self._C0: (Integer(R), 1), self._C1: (Integer(T_COL), 1)}
        result = _tile_device_size(arg, [self._C1], it_space)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], T_COL, f"device_size[1] must shrink to {T_COL}")
        self.assertEqual(result[2], 64)

    # ------------------------------------------------------------------
    # Row-stick, tile the row dimension (c0).
    #
    # c0 appears in Mod(c0,64), the within-stick coord → c0 in stick_syms.
    # Must be excluded.
    # ------------------------------------------------------------------

    def test_row_stick_row_tiling_excluded_as_stick_sym(self):
        """Row-stick: tiling c0 (the stick symbol) leaves device_size unchanged."""
        R, C, T_ROW = 1024, 4096, 256
        arg = self._row_stick_arg(R, C)
        it_space = {self._C0: (Integer(T_ROW), 1), self._C1: (Integer(C), 1)}
        result = _tile_device_size(arg, [self._C0], it_space)
        self.assertEqual(result, [R // 64, C, 64])

    # ------------------------------------------------------------------
    # 4D layout: batch + col-stick.
    # device_size=[B, C//64, R, 64]
    # device_coordinates=[c_b, c1//64, c0, Mod(c1,64)]
    # ------------------------------------------------------------------

    def test_4d_batch_col_stick_tile_batch_shrinks_d0(self):
        """4D layout: tiling c_b (batch, d=0) shrinks device_size[0] — no outer dims."""
        c_b = Symbol("c_b")
        c0, c1 = self._C0, self._C1
        B, R, C, T_B = 8, 256, 128, 4
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=self._FP16,
            device_size=[B, C // 64, R, 64],
            device_coordinates=[c_b, c1 // 64, c0, sympy.Mod(c1, 64)],
            allocation={"hbm": 0},
        )
        it_space = {
            c_b: (Integer(T_B), 1),
            c0: (Integer(R), 1),
            c1: (Integer(C), 1),
        }
        result = _tile_device_size(arg, [c_b], it_space)
        # d=0: c_b, no outer dims → may shrink to T_B
        self.assertEqual(result[0], T_B, f"batch dim must shrink to {T_B}")
        # d=1: c1//64, outer d=0 now has device_size[0]=B > 1 → must not shrink
        self.assertEqual(result[1], C // 64, "sticks_per_row must not change")
        self.assertEqual(result[2], R)
        self.assertEqual(result[3], 64)

    def test_4d_batch_col_stick_tile_col_excluded_as_stick_sym(self):
        """4D layout: tiling c1 (stick sym) leaves device_size unchanged."""
        c_b = Symbol("c_b")
        c0, c1 = self._C0, self._C1
        B, R, C, T_C = 8, 256, 128, 64
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=self._FP16,
            device_size=[B, C // 64, R, 64],
            device_coordinates=[c_b, c1 // 64, c0, sympy.Mod(c1, 64)],
            allocation={"hbm": 0},
        )
        it_space = {
            c_b: (Integer(B), 1),
            c0: (Integer(R), 1),
            c1: (Integer(T_C), 1),
        }
        result = _tile_device_size(arg, [c1], it_space)
        self.assertEqual(result, [B, C // 64, R, 64])

    # ------------------------------------------------------------------
    # Multiple simultaneously tiled symbols.
    #
    # When two non-stick symbols tile independent dimensions, each is
    # evaluated against the *original* device_size.  Only the outermost
    # dimension (no non-degenerate outer dims) may shrink per symbol.
    # ------------------------------------------------------------------

    def test_multi_sym_only_outermost_shrinks(self):
        """Tiling both c_b and c0 simultaneously: only c_b (d=0) may shrink.

        4D col-stick: device_size=[B, C//64, R, 64].
        c_b ticks d=0 (no outer non-degenerate dims → may shrink).
        c0 ticks d=2 (outer d=0 B>1 → must not shrink).
        """
        c_b = Symbol("c_b")
        c0, c1 = self._C0, self._C1
        B, R, C, T_B, T_R = 8, 512, 256, 4, 128
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=self._FP16,
            device_size=[B, C // 64, R, 64],
            device_coordinates=[c_b, c1 // 64, c0, sympy.Mod(c1, 64)],
            allocation={"hbm": 0},
        )
        it_space = {
            c_b: (Integer(T_B), 1),
            c0: (Integer(T_R), 1),
            c1: (Integer(C), 1),
        }
        result = _tile_device_size(arg, [c_b, c0], it_space)
        self.assertEqual(result[0], T_B, f"c_b (d=0) must shrink to {T_B}")
        self.assertEqual(result[1], C // 64, "sticks_per_row must not change")
        self.assertEqual(
            result[2], R, f"c0 (d=2) must stay at {R}: outer d=0 non-degenerate"
        )
        self.assertEqual(result[3], 64)

    def test_multi_sym_two_degenerate_outers_both_shrink(self):
        """Both c_a (d=0) and c0 (d=2) may shrink when outer dims between them are all size 1.

        device_size=[A, 1, R, 64]; d=1 is degenerate.
        c_a ticks d=0 (no outer → may shrink).
        c0 ticks d=2; outer check: device_size[0]=A > 1 → may NOT shrink d=2.
        """
        c_a = Symbol("c_a")
        c0, c1 = self._C0, self._C1
        A, R, C, T_A, T_R = 4, 512, 64, 2, 256
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=self._FP16,
            device_size=[A, 1, R, 64],
            device_coordinates=[c_a, Integer(0), c0, sympy.Mod(c1, 64)],
            allocation={"hbm": 0},
        )
        it_space = {
            c_a: (Integer(T_A), 1),
            c0: (Integer(T_R), 1),
            c1: (Integer(C), 1),
        }
        result = _tile_device_size(arg, [c_a, c0], it_space)
        # c_a at d=0: no outer dims → may shrink
        self.assertEqual(result[0], T_A)
        self.assertEqual(result[1], 1)
        # c0 at d=2: device_size[0]=A > 1 → blocked
        self.assertEqual(result[2], R, f"d=2 must stay at {R}: d=0 has A={A} > 1")
        self.assertEqual(result[3], 64)

    # ------------------------------------------------------------------
    # End-to-end unroll: verify device_size on unrolled copies.
    # This is the regression test for the original bug: a col-stick tensor
    # with >1 sticks/row tiled over rows must preserve device_size[1].
    # ------------------------------------------------------------------

    def test_unroll_preserves_device_size1_multi_stick(self):
        """Unrolled col-stick row-tile copies must carry the full-tensor device_size[1].

        [1024, 4096] fp16: device_size=[64, 1024, 64].  Tile c0 by 512, count=2.
        All copies must have device_size = [64, 1024, 64], NOT [64, 512, 64].
        """
        R, C, T_ROW, COUNT = 1024, 4096, 512, 2
        c0, c1 = Symbol("c0"), Symbol("c1")
        base = 0x400000000
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[C // 64, R, 64],
            device_coordinates=[c1 // 64, c0, sympy.Mod(c1, 64)],
            allocation={"hbm": base},
        )
        op = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                c0: (Integer(T_ROW), 1),
                c1: (Integer(C), 1),
            },
            args=[arg],
            op_info={},
            tiled_symbols=[c0],
        )
        loop = LoopSpec(count=Integer(COUNT), body=[op])
        result = unroll_loop_specs([loop])
        self.assertEqual(len(result), COUNT)
        for i, copy_op in enumerate(result):
            ds = copy_op.args[0].device_size
            self.assertEqual(
                ds,
                [C // 64, R, 64],
                f"tile {i}: device_size must be [{C // 64}, {R}, 64], got {ds}",
            )


if __name__ == "__main__":
    unittest.main()
