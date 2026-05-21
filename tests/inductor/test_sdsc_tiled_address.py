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

"""Unit tests for generate_sdsc with tiled_symbols.

These tests verify that when an OpSpec carries tiled_symbols, generate_sdsc:
  - registers the iteration-0 base HBM address (not the full per-core address)
    into the global symbols list
  - returns affine_strides with the correct per-iteration byte stride for each
    tiled symbol
  - stores a negative symbol ID (not the raw address) in startAddressCoreCorelet_

No device or backend compiler is required.
"""

import unittest

import os
import tempfile

from sympy import Integer, Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import (
    _tiled_byte_stride,
    generate_sdsc,
)
from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec, compile_op_spec
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg

_FP16 = DataFormats.SEN169_FP16  # 64 elems/stick → 2 bytes per element


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdsc_spec(
    s: Symbol,
    *,
    iter_range: int = 64,
    device_stride: int = 128,
    start_address: int = 0x1000,
    allocation: dict | None = None,
    num_cores: int = 1,
) -> SDSCSpec:
    """Build a minimal SDSCSpec with one HBM tensor and one iteration-space symbol."""
    if allocation is None:
        allocation = {"hbm": start_address}
    tensor = SDSCArgs(
        layout="A",
        data_format=_FP16,
        scales={s: 1},
        strides={s: device_stride},
        offsets={s: 0},
        max_dim_sizes={s: -1},
        allocation=allocation,
        start_address=start_address,
        backGap={},
    )
    # For a single core with no work splitting: core_id_to_work_slice maps each
    # dim to Integer(0) (the core always starts at slice 0).
    return SDSCSpec(
        opfunc="add",
        execution_unit="sfp",
        data_format=_FP16,
        num_inputs=1,
        iteration_space={s: iter_range},
        num_cores=num_cores,
        work_slices={s: 1},
        core_id_to_work_slice={s: Integer(0)},
        padding={},
        layouts={
            "A": {
                "dim_order": [s],
                "stick_dim_order": s,
                "stick_size": 64,
            }
        },
        args=[tensor],
        constants={},
        coordinate_masking={},
    )


# ---------------------------------------------------------------------------
# Tests for _tiled_byte_stride
# ---------------------------------------------------------------------------


class TestTiledByteStride(unittest.TestCase):
    def test_fp16_one_core(self):
        s = Symbol("s")
        tensor = SDSCArgs(
            layout="A",
            data_format=_FP16,  # 2 bytes per element
            scales={s: 1},
            strides={s: 128},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0},
            start_address=0,
            backGap={},
        )
        # stride = iter_range × device_stride × bytes = 64 × 128 × 2 = 16384
        stride = _tiled_byte_stride(tensor, s, {s: 64})
        self.assertEqual(stride, 64 * 128 * 2)

    def test_fp16_larger_stride(self):
        s = Symbol("s")
        tensor = SDSCArgs(
            layout="A",
            data_format=_FP16,
            scales={s: 1},
            strides={s: 512},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0},
            start_address=0,
            backGap={},
        )
        # 32 × 512 × 2 = 32768
        stride = _tiled_byte_stride(tensor, s, {s: 32})
        self.assertEqual(stride, 32 * 512 * 2)

    def test_stride_one(self):
        s = Symbol("s")
        tensor = SDSCArgs(
            layout="A",
            data_format=_FP16,
            scales={s: 1},
            strides={s: 1},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0},
            start_address=0,
            backGap={},
        )
        # 16 × 1 × 2 = 32
        stride = _tiled_byte_stride(tensor, s, {s: 16})
        self.assertEqual(stride, 16 * 1 * 2)


# ---------------------------------------------------------------------------
# Tests for generate_sdsc with tiled_symbols
# ---------------------------------------------------------------------------


class TestGenerateSdscTiledSymbols(unittest.TestCase):
    def test_tiled_tensor_affine_strides_correct(self):
        """affine_strides[0] should be {s: iter_range × device_stride × bytes}."""
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s, iter_range=64, device_stride=128)
        symbols: list[int] = []

        _, _, affine_strides = generate_sdsc(
            0, sdsc_spec, symbols, symbol_id_offset=0, tiled_symbols=[s]
        )

        self.assertEqual(len(affine_strides), 1)
        self.assertIn(s, affine_strides[0])
        expected = 64 * 128 * 2  # 16384
        self.assertEqual(affine_strides[0][s], expected)

    def test_tiled_tensor_base_address_registered(self):
        """The iteration-0 base address is appended to symbols (one entry)."""
        s = Symbol("s")
        start = 0x2000
        sdsc_spec = _make_sdsc_spec(s, start_address=start)
        symbols: list[int] = []

        generate_sdsc(0, sdsc_spec, symbols, symbol_id_offset=0, tiled_symbols=[s])

        # One symbol registered: the base address.
        # For a single core with no non-tiled work-slice contribution the base
        # equals start_address itself (tiled dim contribution zeroed at iter 0).
        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0], start)

    def test_tiled_tensor_json_stores_symbol_id(self):
        """startAddressCoreCorelet_ data_ values should be negative symbol IDs."""
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []

        sdsc_json, _, _ = generate_sdsc(
            0, sdsc_spec, symbols, symbol_id_offset=0, tiled_symbols=[s]
        )

        top_val = next(iter(sdsc_json.values()))
        node = top_val["dscs_"][0]["add"]["scheduleTree_"][0]
        data = node["startAddressCoreCorelet_"]["data_"]
        # The value stored must be a negative integer (symbol ID), not the
        # raw address.
        for v in data.values():
            self.assertLess(int(v), 0, f"Expected negative symbol ID, got {v!r}")

    def test_non_tiled_tensor_empty_affine_strides(self):
        """A tensor not in tiled_symbols gets an empty affine_strides dict."""
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []

        _, _, affine_strides = generate_sdsc(
            0, sdsc_spec, symbols, symbol_id_offset=0, tiled_symbols=[]
        )

        self.assertEqual(affine_strides, [{}])

    def test_lx_tensor_not_in_symbols(self):
        """An lx tensor's address is baked into JSON; it must NOT appear in symbols."""
        s = Symbol("s")
        lx_addr = 0xABC0
        sdsc_spec = _make_sdsc_spec(
            s, start_address=lx_addr, allocation={"lx": lx_addr}
        )
        symbols: list[int] = []

        _, local_sym_values, affine_strides = generate_sdsc(
            0, sdsc_spec, symbols, symbol_id_offset=0, tiled_symbols=[s]
        )

        self.assertEqual(symbols, [], "lx address must not be registered in symbols")
        self.assertEqual(local_sym_values, [])
        self.assertEqual(affine_strides, [{}])

    def test_symbol_id_offset_applied(self):
        """symbol_id_offset shifts the negative IDs assigned by this SDSC."""
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []

        sdsc_json, local_sym_values, _ = generate_sdsc(
            0, sdsc_spec, symbols, symbol_id_offset=5, tiled_symbols=[s]
        )

        # With offset=5 the first ID should be -(5+1) = -6
        top_val = next(iter(sdsc_json.values()))
        node = top_val["dscs_"][0]["add"]["scheduleTree_"][0]
        data = node["startAddressCoreCorelet_"]["data_"]
        ids = [int(v) for v in data.values()]
        self.assertTrue(all(i <= -6 for i in ids), f"Expected ids ≤ -6, got {ids}")

    def test_multi_core_tiled_per_core_symbols(self):
        """Two cores in a tiled op each get their own iter-0 base address symbol.

        With work_slices={s: 2} core 0 starts at start_address and core 1 at
        start_address + 1 * stride * bytes.  Both symbols are registered so
        that each core's per-iteration address is: sym + affine_stride * iter.
        """
        s = Symbol("s")
        core_id = Symbol("core_id")
        tensor = SDSCArgs(
            layout="A",
            data_format=_FP16,
            scales={s: 1},
            strides={s: 128},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0x1000},
            start_address=0x1000,
            backGap={},
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={s: 32},
            num_cores=2,
            work_slices={s: 2},
            core_id_to_work_slice={s: core_id},
            padding={},
            layouts={"A": {"dim_order": [s], "stick_dim_order": s, "stick_size": 64}},
            args=[tensor],
            constants={},
            coordinate_masking={},
        )
        symbols: list[int] = []

        _, local_sym_values, affine_strides = generate_sdsc(
            0, sdsc_spec, symbols, symbol_id_offset=0, tiled_symbols=[s]
        )

        # Each core gets its own iter-0 base symbol.
        # core_idx_to_slice_offset: wk_slice[s]=1, stride=128, work_slices[s]=2
        # → offset = 1 * 128 // 2 = 64 elements → 64 * 2 bytes = 128 bytes
        # core 0: 0x1000 + 0 = 0x1000
        # core 1: 0x1000 + 128 = 0x1080
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0], 0x1000)
        self.assertEqual(symbols[1], 0x1000 + 128)
        # Tiled stride is still present.
        self.assertIn(s, affine_strides[0])


# ---------------------------------------------------------------------------
# Tests for compile_op_spec: symbol_mapping translation of tiled_symbols
# ---------------------------------------------------------------------------


def _make_tiled_op_spec() -> OpSpec:
    """Minimal OpSpec with tiled_symbols that compile_op_spec can process."""
    c0 = Symbol("c0")
    fp16 = _FP16
    tensor_in = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x1000},
    )
    tensor_out = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x2000},
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={c0: (Integer(128), 1)},
        args=[tensor_in, tensor_out],
        op_info={},
        tiled_symbols=[c0],
    )


class TestCompileOpSpecSymbolMapping(unittest.TestCase):
    """Verify compile_op_spec translates tiled_symbols through symbol_mapping.

    parse_op_spec renames inductor symbols (c0, c1, ...) to SDSC dimension
    labels (mb, out, ...).  compile_op_spec must apply the same mapping to
    tiled_symbols before forwarding them to generate_sdsc.  Without this
    translation generate_sdsc finds no matching symbol in tensor.strides and
    returns affine_strides=[{}, ...], causing generate_bundle to fall back to
    static addresses instead of affine.apply.
    """

    def test_affine_strides_non_empty_for_tiled_op(self):
        """compile_op_spec returns non-empty affine_strides when tiled_symbols is set."""
        op_spec = _make_tiled_op_spec()
        symbols: list[int] = []
        _, _, affine_strides = compile_op_spec(0, op_spec, symbols)

        has_strides = any(len(d) > 0 for d in affine_strides)
        self.assertTrue(
            has_strides,
            f"Expected non-empty affine_strides; got {affine_strides}. "
            "tiled_symbols may not have been translated through symbol_mapping.",
        )

    def test_generate_bundle_emits_affine_apply_for_tiled_loop(self):
        """generate_bundle emits affine.apply when OpSpec carries tiled_symbols."""
        from torch_spyre._inductor.codegen.bundle import generate_bundle

        op_spec = _make_tiled_op_spec()
        loop = LoopSpec(count=Integer(4), body=[op_spec])
        tmpdir = tempfile.mkdtemp()
        generate_bundle("test_kernel", tmpdir, [loop])

        with open(os.path.join(tmpdir, "bundle.mlir")) as f:
            mlir = f.read()

        self.assertIn(
            "affine.apply",
            mlir,
            "Expected affine.apply in bundle.mlir. "
            "symbol_mapping translation may be broken in compile_op_spec.",
        )
        self.assertIn("affine_map", mlir)
        self.assertIn("scf.for", mlir)


if __name__ == "__main__":
    unittest.main()
