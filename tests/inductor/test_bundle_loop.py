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

"""Unit tests for generate_bundle / bundle.mlir with LoopSpec entries.

These tests exercise the refactored bundle.py without a Spyre device or
running the backend compiler.  They mock compile_op_spec so no actual SDSC
JSON generation is needed, and write bundle artifacts to a temporary directory.
"""

import os
import tempfile
import unittest
from sympy import Integer, Symbol
from unittest.mock import patch

from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
from torch_spyre._inductor.codegen.bundle import (
    generate_bundle,
    _collect_op_specs,
    _collect_loop_counts,
    _mlir_count_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_op_spec(name: str) -> OpSpec:
    """Return a minimal OpSpec sufficient for testing."""
    return OpSpec(
        op=name,
        is_reduction=False,
        iteration_space={},
        args=[],
        op_info={},
    )


def _fake_compile_op_spec(
    idx: int, op_spec: OpSpec, symbols: list, symbol_id_offset: int = 0
):
    """Stub that returns (json, [], []) — no real SDSC compilation."""
    return {f"{idx}_{op_spec.op}": {"op": op_spec.op}}, [], []


def _read_mlir(output_dir: str) -> str:
    with open(os.path.join(output_dir, "bundle.mlir")) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tests for _collect_op_specs
# ---------------------------------------------------------------------------


class TestCollectOpSpecs(unittest.TestCase):
    def test_flat_list(self):
        a, b = _make_op_spec("a"), _make_op_spec("b")
        result: list[OpSpec] = []
        _collect_op_specs([a, b], result)
        self.assertEqual(result, [a, b])

    def test_single_loop(self):
        a, b = _make_op_spec("a"), _make_op_spec("b")
        loop = LoopSpec(count=Integer(4), body=[a, b])
        result: list[OpSpec] = []
        _collect_op_specs([loop], result)
        self.assertEqual(result, [a, b])

    def test_nested_loop(self):
        a = _make_op_spec("a")
        b = _make_op_spec("b")
        inner = LoopSpec(count=Integer(2), body=[b])
        outer = LoopSpec(count=Integer(4), body=[a, inner])
        result: list[OpSpec] = []
        _collect_op_specs([outer], result)
        self.assertEqual(result, [a, b])

    def test_mixed_flat_and_loop(self):
        a, b, c = _make_op_spec("a"), _make_op_spec("b"), _make_op_spec("c")
        loop = LoopSpec(count=Integer(3), body=[b])
        result: list[OpSpec] = []
        _collect_op_specs([a, loop, c], result)
        self.assertEqual(result, [a, b, c])

    def test_empty(self):
        result: list[OpSpec] = []
        _collect_op_specs([], result)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Tests for _collect_loop_counts
# ---------------------------------------------------------------------------


class TestCollectLoopCounts(unittest.TestCase):
    def test_no_loops(self):
        a = _make_op_spec("a")
        counts = _collect_loop_counts([a])
        self.assertEqual(counts, [])

    def test_single_loop(self):
        loop = LoopSpec(count=Integer(4), body=[])
        counts = _collect_loop_counts([loop])
        self.assertEqual(counts, [Integer(4)])

    def test_nested_loops_depth_first_order(self):
        inner = LoopSpec(count=Integer(2), body=[])
        outer = LoopSpec(count=Integer(4), body=[inner])
        counts = _collect_loop_counts([outer])
        # outer count first, then inner
        self.assertEqual(counts, [Integer(4), Integer(2)])

    def test_two_sequential_loops(self):
        loop0 = LoopSpec(count=Integer(4), body=[])
        loop1 = LoopSpec(count=Integer(8), body=[])
        counts = _collect_loop_counts([loop0, loop1])
        self.assertEqual(counts, [Integer(4), Integer(8)])


# ---------------------------------------------------------------------------
# Tests for _mlir_count_value
# ---------------------------------------------------------------------------


class TestMlirCountValue(unittest.TestCase):
    def test_integer_count(self):
        self.assertEqual(_mlir_count_value(Integer(4)), "arith.constant 4 : index")

    def test_integer_count_one(self):
        self.assertEqual(_mlir_count_value(Integer(1)), "arith.constant 1 : index")

    def test_symbolic_count_raises(self):
        k = Symbol("K")
        with self.assertRaises(NotImplementedError):
            _mlir_count_value(k)


# ---------------------------------------------------------------------------
# Tests for generate_bundle (mlir output)
# ---------------------------------------------------------------------------


class TestGenerateBundleMlir(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patch = patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=_fake_compile_op_spec,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def _bundle(self, specs):
        generate_bundle("test_kernel", self.tmpdir, specs)
        return _read_mlir(self.tmpdir)

    def test_flat_ops_no_loop(self):
        a, b = _make_op_spec("a"), _make_op_spec("b")
        mlir = self._bundle([a, b])
        self.assertIn("sdscbundle.sdsc_execute", mlir)
        self.assertNotIn("scf.for", mlir)
        self.assertNotIn("arith.constant", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 2)

    def test_single_loop_emits_scf_for(self):
        a, b = _make_op_spec("a"), _make_op_spec("b")
        loop = LoopSpec(count=Integer(4), body=[a, b])
        mlir = self._bundle([loop])
        self.assertIn("scf.for", mlir)
        self.assertIn("arith.constant 4 : index", mlir)
        self.assertIn("%c0", mlir)
        self.assertIn("%c1", mlir)
        # Both body ops must appear inside the loop
        self.assertEqual(mlir.count("sdsc_execute"), 2)

    def test_single_loop_structure(self):
        a = _make_op_spec("a")
        loop = LoopSpec(count=Integer(3), body=[a])
        mlir = self._bundle([loop])
        # scf.for line should precede sdsc_execute line
        for_pos = mlir.index("scf.for")
        exec_pos = mlir.index("sdsc_execute")
        close_pos = mlir.rindex("}")
        self.assertLess(for_pos, exec_pos)
        self.assertLess(exec_pos, close_pos)

    def test_flat_op_before_and_after_loop(self):
        before = _make_op_spec("before")
        body = _make_op_spec("body")
        after = _make_op_spec("after")
        loop = LoopSpec(count=Integer(2), body=[body])
        mlir = self._bundle([before, loop, after])
        self.assertIn("scf.for", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 3)

    def test_nested_loops(self):
        a = _make_op_spec("a")
        b = _make_op_spec("b")
        inner = LoopSpec(count=Integer(2), body=[b])
        outer = LoopSpec(count=Integer(4), body=[a, inner])
        mlir = self._bundle([outer])
        self.assertEqual(mlir.count("scf.for"), 2)
        self.assertIn("arith.constant 4 : index", mlir)
        self.assertIn("arith.constant 2 : index", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 2)
        # Inner loop should appear after outer loop open
        outer_pos = mlir.index("scf.for")
        inner_pos = mlir.index("scf.for", outer_pos + 1)
        self.assertLess(outer_pos, inner_pos)

    def test_sdsc_json_files_written_depth_first(self):
        a = _make_op_spec("a")
        b = _make_op_spec("b")
        loop = LoopSpec(count=Integer(2), body=[a, b])
        generate_bundle("test_kernel", self.tmpdir, [loop])
        written = sorted(f for f in os.listdir(self.tmpdir) if f.endswith(".json"))
        # Two JSON files: idx 0 → a, idx 1 → b
        self.assertEqual(len(written), 2)

    def test_empty_specs_writes_minimal_bundle(self):
        mlir = self._bundle([])
        self.assertIn("func.func @sdsc_bundle", mlir)
        self.assertIn("return", mlir)
        self.assertNotIn("sdsc_execute", mlir)
        self.assertNotIn("scf.for", mlir)

    def test_symbolic_count_raises(self):
        k = Symbol("K")
        a = _make_op_spec("a")
        loop = LoopSpec(count=k, body=[a])
        with self.assertRaises(NotImplementedError):
            self._bundle([loop])


# ---------------------------------------------------------------------------
# Tests for async_compile._find_unimplemented
# ---------------------------------------------------------------------------


class TestFindUnimplemented(unittest.TestCase):
    def test_no_unimplemented(self):
        from torch_spyre.execution.async_compile import _find_unimplemented

        a = _make_op_spec("a")
        self.assertIsNone(_find_unimplemented([a]))

    def test_flat_unimplemented(self):
        from torch_spyre._inductor.op_spec import UnimplementedOp
        from torch_spyre.execution.async_compile import _find_unimplemented

        unimp = UnimplementedOp(op="missing")
        a = _make_op_spec("a")
        result = _find_unimplemented([a, unimp])
        self.assertIs(result, unimp)

    def test_unimplemented_inside_loop(self):
        from torch_spyre._inductor.op_spec import UnimplementedOp
        from torch_spyre.execution.async_compile import _find_unimplemented

        unimp = UnimplementedOp(op="missing")
        loop = LoopSpec(count=Integer(4), body=[unimp])
        result = _find_unimplemented([loop])
        self.assertIs(result, unimp)

    def test_unimplemented_in_nested_loop(self):
        from torch_spyre._inductor.op_spec import UnimplementedOp
        from torch_spyre.execution.async_compile import _find_unimplemented

        unimp = UnimplementedOp(op="missing")
        inner = LoopSpec(count=Integer(2), body=[unimp])
        outer = LoopSpec(count=Integer(4), body=[inner])
        result = _find_unimplemented([outer])
        self.assertIs(result, unimp)

    def test_returns_first_found(self):
        from torch_spyre._inductor.op_spec import UnimplementedOp
        from torch_spyre.execution.async_compile import _find_unimplemented

        u1 = UnimplementedOp(op="first")
        u2 = UnimplementedOp(op="second")
        result = _find_unimplemented([u1, u2])
        self.assertIs(result, u1)


class TestGenerateBundleMlirSnapshot(unittest.TestCase):
    """Snapshot tests: verify exact bundle.mlir output to catch regression in format."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patch = patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=_fake_compile_op_spec,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def _bundle(self, specs):
        generate_bundle("test_kernel", self.tmpdir, specs)
        return _read_mlir(self.tmpdir)

    def test_single_loop_snapshot(self):
        a = _make_op_spec("a")
        loop = LoopSpec(count=Integer(8), body=[a])
        mlir = self._bundle([loop])
        expected = (
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 8 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            '\t\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json", "symbol_ids"=[]}\n'
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)

    def test_flat_snapshot(self):
        a = _make_op_spec("a")
        mlir = self._bundle([a])
        expected = (
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json", "symbol_ids"=[]}\n'
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


# ---------------------------------------------------------------------------
# Tests for generate_bundle with non-empty affine_strides (tiled address path)
# ---------------------------------------------------------------------------


def _make_tiled_json(idx: int, sym_id: int) -> dict:
    """Return a minimal SDSC JSON with one HBM tensor whose symbol ID is sym_id."""
    return {
        f"{idx}_add": {
            "numCoresUsed_": 1,
            "dscs_": [
                {
                    "add": {
                        "scheduleTree_": [
                            {
                                "component_": "hbm",
                                "startAddressCoreCorelet_": {
                                    "data_": {"[0, 0, 0]": str(sym_id)}
                                },
                            }
                        ]
                    }
                }
            ],
        }
    }


class TestGenerateBundleMlirWithAffineStrides(unittest.TestCase):
    """Verify scf.for + affine.apply emission when compile_op_spec returns strides."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._s = Symbol("s")

    def _bundle(self, specs, fake_compile):
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=fake_compile,
        ):
            generate_bundle("test_kernel", self.tmpdir, specs)
        with open(os.path.join(self.tmpdir, "bundle.mlir")) as f:
            return f.read()

    def test_tiled_tensor_emits_affine_apply(self):
        """A tiled tensor inside a LoopSpec produces an affine.apply in the MLIR."""
        s = self._s
        stride = 16384

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return _make_tiled_json(idx, sym_id), [0x1000], [{s: stride}]

        op = _make_op_spec("a")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        self.assertIn("affine_map", mlir)
        self.assertIn(str(stride), mlir)
        self.assertIn("affine.apply", mlir)
        self.assertIn("%addr_0", mlir)
        self.assertIn(
            'sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json"', mlir
        )
        self.assertIn('"symbol_ids"=[-1]', mlir)

    def test_non_tiled_tensor_in_loop_no_affine_apply(self):
        """A non-tiled tensor inside a LoopSpec uses %sym_N directly, no affine.apply."""

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x2000)
            return _make_tiled_json(idx, sym_id), [0x2000], [{}]

        op = _make_op_spec("b")
        loop = LoopSpec(count=Integer(2), body=[op])
        mlir = self._bundle([loop], fake_compile)

        self.assertNotIn("affine.apply", mlir)
        self.assertNotIn("affine_map", mlir)
        self.assertIn("%sym_1", mlir)
        self.assertIn("sdscbundle.sdsc_execute (%sym_1)", mlir)

    def test_affine_map_stride_at_module_level(self):
        """affine_map definition must appear before 'module {' body."""
        s = self._s
        stride = 8192

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x3000)
            return _make_tiled_json(idx, sym_id), [0x3000], [{s: stride}]

        op = _make_op_spec("c")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        map_pos = mlir.index("affine_map")
        module_pos = mlir.index("module {")
        self.assertLess(map_pos, module_pos, "affine_map must precede module {")

    def test_affine_apply_inside_scf_for(self):
        """affine.apply must appear inside the scf.for body (after it, before })."""
        s = self._s

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x4000)
            return _make_tiled_json(idx, sym_id), [0x4000], [{s: 512}]

        op = _make_op_spec("d")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        for_pos = mlir.index("scf.for")
        apply_pos = mlir.index("affine.apply")
        execute_pos = mlir.index("sdsc_execute")
        self.assertLess(for_pos, apply_pos)
        self.assertLess(apply_pos, execute_pos)

    def test_tiled_snapshot(self):
        """Exact snapshot for a single tiled op in a loop."""
        s = self._s

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return _make_tiled_json(idx, sym_id), [0x1000], [{s: 256}]

        op = _make_op_spec("a")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        expected = (
            "#map_0 = affine_map<(d0)[s0] -> (s0 + 256*d0)>\n"
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 4 : index\n"
            "\t\t%sym_1 = arith.constant 4096 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            "\t\t\t%addr_0 = affine.apply #map_0(%i_0)[%sym_1]\n"
            '\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json",'
            ' "symbol_ids"=[-1]}\n'
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


# ---------------------------------------------------------------------------
# Tests for nested tiling: two-level affine_map with two loop variables
# ---------------------------------------------------------------------------


class TestGenerateBundleNestedTiling(unittest.TestCase):
    """Verify that nested LoopSpec with a two-entry affine_strides dict produces
    a 2-dimensional affine_map and nested scf.for loops."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.s0 = Symbol("s0")
        self.s1 = Symbol("s1")

    def _bundle(self, specs, fake_compile):
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=fake_compile,
        ):
            generate_bundle("test_kernel", self.tmpdir, specs)
        with open(os.path.join(self.tmpdir, "bundle.mlir")) as f:
            return f.read()

    def _fake_compile_two_strides(self, outer_stride, inner_stride):
        """Return a fake_compile that injects a two-entry affine_strides dict."""
        s0, s1 = self.s0, self.s1

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return (
                _make_tiled_json(idx, sym_id),
                [0x1000],
                [{s0: outer_stride, s1: inner_stride}],
            )

        return fake_compile

    def test_nested_loop_emits_two_scf_for(self):
        """Two nested LoopSpecs produce two scf.for blocks."""
        op = _make_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        self.assertEqual(mlir.count("scf.for"), 2)

    def test_nested_tiling_emits_2d_affine_map(self):
        """A two-entry affine_strides dict produces affine_map<(d0, d1)[s0] -> ...>."""
        op = _make_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        self.assertIn("affine_map<(d0, d1)[s0]", mlir)
        self.assertIn("512*d0", mlir)
        self.assertIn("64*d1", mlir)

    def test_nested_tiling_affine_apply_uses_both_loop_vars(self):
        """affine.apply inside the inner loop must reference both %i_0 and %i_1."""
        op = _make_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        self.assertIn("affine.apply", mlir)
        apply_line = next(line for line in mlir.splitlines() if "affine.apply" in line)
        self.assertIn("%i_0", apply_line)
        self.assertIn("%i_1", apply_line)

    def test_nested_tiling_snapshot(self):
        """Exact snapshot for a single op inside nested K=4/K=2 loops."""
        op = _make_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        expected = (
            "#map_0 = affine_map<(d0, d1)[s0] -> (s0 + 512*d0 + 64*d1)>\n"
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 4 : index\n"
            "\t\t%loop_bound_1 = arith.constant 2 : index\n"
            "\t\t%sym_1 = arith.constant 4096 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            "\t\t\tscf.for %i_1 = %c0 to %loop_bound_1 step %c1 {\n"
            "\t\t\t\t%addr_0 = affine.apply #map_0(%i_0, %i_1)[%sym_1]\n"
            '\t\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json",'
            ' "symbol_ids"=[-1]}\n'
            "\t\t\t}\n"
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


if __name__ == "__main__":
    unittest.main()
