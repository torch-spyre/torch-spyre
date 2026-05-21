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


def _fake_compile_op_spec(idx: int, op_spec: OpSpec):
    """Stub that returns a dict keyed by op name (no real SDSC compilation)."""
    return {f"{op_spec.op}_{idx}": {"op": op_spec.op}}


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
            '\t\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_a_0.json"}\n'
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
            '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_a_0.json"}\n'
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


if __name__ == "__main__":
    unittest.main()
