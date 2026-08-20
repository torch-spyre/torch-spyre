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

"""Unit tests for torch_spyre._inductor.op_spec_validation.

Tests exercise the validate_op_specs() public entry point and the
_is_unimplemented_op duck-type check.  No Spyre hardware is required.
"""

import dataclasses
import unittest

import sympy
from sympy import Integer, Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.op_spec_validation import (
    BINARY_OPS,
    OpSpecValidationError,
    _is_unimplemented_op,
    validate_op_specs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_C_ROW = Symbol("c_row")
_C_COL = Symbol("c_col")


def _make_tensor_arg(is_input: bool = True, arg_index: int = 0) -> TensorArg:
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 128, 64],
        device_coordinates=[_C_COL // 64, _C_ROW, sympy.Mod(_C_COL, 64)],
        allocation={"hbm": 0x400000000},
    )


def _make_valid_op_spec(op: str = "add", is_reduction: bool = False) -> OpSpec:
    """Build a minimal valid OpSpec for a binary pointwise op."""
    args = [
        _make_tensor_arg(is_input=True, arg_index=0),
        _make_tensor_arg(is_input=True, arg_index=1),
        _make_tensor_arg(is_input=False, arg_index=2),
    ]
    return OpSpec(
        op=op,
        is_reduction=is_reduction,
        iteration_space={
            _C_ROW: (Integer(128), 1),
            _C_COL: (Integer(256), 1),
        },
        args=args,
        op_info={},
        tiled_symbols=[],
    )


def _make_matmul_op_spec() -> OpSpec:
    """Build a valid matmul OpSpec."""
    args = [
        _make_tensor_arg(is_input=True, arg_index=0),
        _make_tensor_arg(is_input=True, arg_index=1),
        _make_tensor_arg(is_input=False, arg_index=2),
    ]
    return OpSpec(
        op="matmul",
        is_reduction=True,
        iteration_space={
            _C_ROW: (Integer(128), 1),
            _C_COL: (Integer(256), 1),
        },
        args=args,
        op_info={},
        tiled_symbols=[],
    )


# ---------------------------------------------------------------------------
# Tests: validate_op_specs — happy path
# ---------------------------------------------------------------------------


class TestValidateOpSpecsHappyPath(unittest.TestCase):
    def test_empty_list(self):
        validate_op_specs([], stage="test")

    def test_single_valid_op_spec(self):
        validate_op_specs([_make_valid_op_spec()], stage="test")

    def test_multiple_valid_op_specs(self):
        specs = [_make_valid_op_spec("add"), _make_valid_op_spec("mul")]
        validate_op_specs(specs, stage="test")

    def test_op_spec_unimplemented_op(self):
        specs = [_make_valid_op_spec(), UnimplementedOp(op="custom_op")]
        validate_op_specs(specs, stage="test")

    def test_loop_spec_with_valid_body(self):
        loop = LoopSpec(count=Integer(4), body=[_make_valid_op_spec()])
        validate_op_specs([loop], stage="test")

    def test_nested_loop_spec(self):
        inner_op = _make_valid_op_spec()
        inner_loop = LoopSpec(count=Integer(2), body=[inner_op])
        outer_loop = LoopSpec(count=Integer(4), body=[inner_loop])
        validate_op_specs([outer_loop], stage="test")

    def test_matmul_valid(self):
        validate_op_specs([_make_matmul_op_spec()], stage="test")

    def test_tiled_symbols_valid(self):
        op = _make_valid_op_spec()
        op.tiled_symbols = [[_C_ROW]]
        validate_op_specs([op], stage="test")

    def test_tiled_symbol_in_trip_counts_only(self):
        """Symbol in tiled_symbol_trip_counts but not iteration_space is valid."""
        op = _make_valid_op_spec()
        tile_sym = Symbol("_tile_adv_c0_0")
        op.tiled_symbols = [[tile_sym]]
        op.tiled_symbol_trip_counts = {tile_sym: 4}
        validate_op_specs([op], stage="test")

    def test_device_size_zero_allowed(self):
        """device_size dimension of 0 is valid (FP8 sub-stick layout)."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(op.args[0], device_size=[2, 0, 64])
        validate_op_specs([op], stage="after_creation_loop_wrapping")


# ---------------------------------------------------------------------------
# Tests: validate_op_specs — error cases
# ---------------------------------------------------------------------------


class TestValidateOpSpecsErrors(unittest.TestCase):
    def test_unexpected_type_in_list(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(["not_an_op_spec"], stage="test_stage")
        self.assertIn("unexpected type in spec list", str(ctx.exception))
        self.assertIn("test_stage", str(ctx.exception))

    def test_empty_op_name(self):
        op = _make_valid_op_spec()
        op.op = ""
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("op must be a non-empty string", str(ctx.exception))

    def test_empty_iteration_space_skips_validation(self):
        op = _make_valid_op_spec()
        op.iteration_space = {}
        # Incomplete specs (empty iteration_space) are silently skipped.
        validate_op_specs([op], stage="test")

    def test_iteration_space_non_symbol_key(self):
        op = _make_valid_op_spec()
        op.iteration_space["bad_key"] = (Integer(10), 1)
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("iteration_space keys must be sympy.Symbol", str(ctx.exception))

    def test_iteration_space_bad_value(self):
        op = _make_valid_op_spec()
        bad_sym = Symbol("bad")
        op.iteration_space[bad_sym] = (Integer(10),)
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("2-tuples", str(ctx.exception))

    def test_iteration_space_zero_work_division(self):
        op = _make_valid_op_spec()
        bad_sym = Symbol("bad")
        op.iteration_space[bad_sym] = (Integer(10), 0)
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("work_division must be a positive int", str(ctx.exception))

    def test_empty_args_skips_validation(self):
        op = _make_valid_op_spec()
        op.args = []
        # Incomplete specs (empty args) are silently skipped.
        validate_op_specs([op], stage="test")

    def test_arg_not_tensor_arg(self):
        op = _make_valid_op_spec()
        op.args = [_make_tensor_arg(), "not_a_tensor_arg", _make_tensor_arg(False, 2)]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("must be a TensorArg", str(ctx.exception))

    def test_no_output_arg(self):
        op = _make_valid_op_spec()
        op.args = [_make_tensor_arg(True, 0), _make_tensor_arg(True, 1)]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("at least one output TensorArg", str(ctx.exception))

    def test_device_coordinates_length_mismatch(self):
        op = _make_valid_op_spec()
        bad_arg = _make_tensor_arg(is_input=False, arg_index=2)
        bad_arg.device_coordinates = [_C_COL // 64, _C_ROW]
        op.args = [
            _make_tensor_arg(True, 0),
            _make_tensor_arg(True, 1),
            bad_arg,
        ]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn(
            "len(device_coordinates) must equal len(device_size)", str(ctx.exception)
        )

    def test_symbol_not_in_iteration_space(self):
        op = _make_valid_op_spec()
        foreign_sym = Symbol("foreign")
        bad_arg = _make_tensor_arg(is_input=False, arg_index=2)
        bad_arg.device_coordinates = [foreign_sym, _C_ROW, sympy.Mod(_C_COL, 64)]
        op.args = [
            _make_tensor_arg(True, 0),
            _make_tensor_arg(True, 1),
            bad_arg,
        ]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("symbols not in iteration_space", str(ctx.exception))

    def test_indirect_symbol_allowed_before_simplification(self):
        """Raw indirect0 symbol passes before IndirectAccess wrapping."""
        op = _make_valid_op_spec()
        indirect0 = Symbol("indirect0")
        bad_arg = _make_tensor_arg(is_input=False, arg_index=2)
        bad_arg.device_coordinates = [indirect0, _C_ROW, sympy.Mod(_C_COL, 64)]
        op.args = [
            _make_tensor_arg(True, 0),
            _make_tensor_arg(True, 1),
            bad_arg,
        ]
        validate_op_specs([op], stage="after_creation_loop_wrapping")

    def test_tiled_symbols_not_in_iteration_space(self):
        op = _make_valid_op_spec()
        foreign_sym = Symbol("foreign")
        op.tiled_symbols = [[foreign_sym]]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn(
            "tiled_symbols[0] references symbol not in iteration_space",
            str(ctx.exception),
        )

    def test_matmul_not_reduction(self):
        op = _make_matmul_op_spec()
        op.is_reduction = False
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("matmul ops must have is_reduction=True", str(ctx.exception))

    def test_matmul_too_few_inputs(self):
        op = _make_matmul_op_spec()
        op.args = [_make_tensor_arg(True, 0), _make_tensor_arg(False, 1)]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("at least 2 input TensorArgs", str(ctx.exception))

    def test_reduction_op_not_reduction(self):
        op = _make_valid_op_spec("sum", is_reduction=False)
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("must have is_reduction=True", str(ctx.exception))

    def test_pointwise_op_is_reduction(self):
        op = _make_valid_op_spec("add", is_reduction=True)
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("must have is_reduction=False", str(ctx.exception))

    def test_binary_op_single_input(self):
        op = _make_valid_op_spec("add")
        op.args = [_make_tensor_arg(True, 0), _make_tensor_arg(False, 1)]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("requires at least 2 input TensorArgs", str(ctx.exception))

    def test_where3_too_few_inputs(self):
        op = _make_valid_op_spec("where3")
        op.args = [
            _make_tensor_arg(True, 0),
            _make_tensor_arg(True, 1),
            _make_tensor_arg(False, 2),
        ]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("where3 requires at least 3 input TensorArgs", str(ctx.exception))

    def test_loop_spec_empty_body(self):
        loop = LoopSpec(count=Integer(4), body=[])
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([loop], stage="test")
        self.assertIn("LoopSpec has empty body", str(ctx.exception))

    def test_loop_spec_zero_count(self):
        loop = LoopSpec(count=Integer(0), body=[_make_valid_op_spec()])
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([loop], stage="test")
        self.assertIn("LoopSpec count must be positive", str(ctx.exception))

    def test_arg_index_negative_at_bundle_stage_hbm(self):
        """Non-pool-allocated arg with arg_index=-1 at bundle stage errors."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(op.args[0], arg_index=-1)
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="before_bundle_generation")
        self.assertIn("arg_index must be a non-negative int", str(ctx.exception))

    def test_arg_index_negative_pool_allocated_at_bundle_stage(self):
        """Pool-allocated arg with arg_index=-1 at bundle stage is valid."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(
            op.args[0], arg_index=-1, allocation={"hbm_pool": 0x0}
        )
        validate_op_specs([op], stage="before_bundle_generation")

    def test_arg_index_negative_lx_allocated_at_bundle_stage(self):
        """LX-allocated arg with arg_index=-1 at bundle stage is valid."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(
            op.args[0], arg_index=-1, allocation={"lx": 0x0}
        )
        validate_op_specs([op], stage="before_bundle_generation")

    def test_unknown_op_no_output_passes(self):
        """Unknown/synthetic ops without output args are valid."""
        c0 = Symbol("c0")
        op = OpSpec(
            op="synthetic_test_op",
            is_reduction=False,
            iteration_space={c0: (Integer(128), 1)},
            args=[_make_tensor_arg(is_input=True, arg_index=0)],
            op_info={},
            tiled_symbols=[],
        )
        validate_op_specs([op], stage="test")

    def test_allocation_invalid_keys(self):
        """allocation must contain exactly one of hbm/lx/hbm_pool."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(op.args[0], allocation={"bad_key": 42})
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("exactly one of hbm/lx/hbm_pool", str(ctx.exception))

    def test_allocation_multiple_keys(self):
        """allocation must not contain more than one valid key."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(
            op.args[0], allocation={"hbm": 0x1000, "lx": 0x2000}
        )
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="test")
        self.assertIn("exactly one of hbm/lx/hbm_pool", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: _is_unimplemented_op duck-type check
# ---------------------------------------------------------------------------


class TestIsUnimplementedOp(unittest.TestCase):
    def test_op_spec_unimplemented_op(self):
        self.assertTrue(_is_unimplemented_op(UnimplementedOp(op="custom")))

    def test_rvalue_unimplemented_op(self):
        """Simulates spyre_kernel.UnimplementedOp(RValue) without importing it."""

        @dataclasses.dataclass
        class UnimplementedOp:
            op: str

        obj = UnimplementedOp(op="unsupported_thing")
        self.assertTrue(_is_unimplemented_op(obj))

    def test_not_unimplemented_wrong_name(self):
        @dataclasses.dataclass
        class SomethingElse:
            op: str

        obj = SomethingElse(op="add")
        self.assertFalse(_is_unimplemented_op(obj))

    def test_not_unimplemented_no_op_attr(self):
        @dataclasses.dataclass
        class UnimplementedOp:
            value: int

        obj = UnimplementedOp(value=42)
        self.assertFalse(_is_unimplemented_op(obj))

    def test_not_unimplemented_op_not_str(self):
        @dataclasses.dataclass
        class UnimplementedOp:
            op: int

        obj = UnimplementedOp(op=42)
        self.assertFalse(_is_unimplemented_op(obj))

    def test_op_spec_is_not_unimplemented(self):
        op = _make_valid_op_spec()
        self.assertFalse(_is_unimplemented_op(op))

    def test_string_is_not_unimplemented(self):
        self.assertFalse(_is_unimplemented_op("not_an_op"))


# ---------------------------------------------------------------------------
# Tests: validate_op_specs with mixed UnimplementedOp types
# ---------------------------------------------------------------------------


class TestMixedUnimplementedOps(unittest.TestCase):
    def test_dataclass_unimplemented_op_accepted(self):
        specs = [UnimplementedOp(op="custom"), _make_valid_op_spec()]
        validate_op_specs(specs, stage="test")

    def test_rvalue_style_unimplemented_op_accepted(self):
        """Duck-typed UnimplementedOp (e.g. from spyre_kernel) passes validation."""

        @dataclasses.dataclass
        class UnimplementedOp:
            op: str

        specs = [UnimplementedOp(op="custom_kernel_op"), _make_valid_op_spec()]
        validate_op_specs(specs, stage="test")

    def test_loop_with_mixed_unimplemented_ops(self):
        @dataclasses.dataclass
        class UnimplementedOp:
            op: str

        body = [
            _make_valid_op_spec(),
            UnimplementedOp(op="kernel_op"),
        ]
        loop = LoopSpec(count=Integer(4), body=body)
        validate_op_specs([loop], stage="test")


# ---------------------------------------------------------------------------
# Tests: BINARY_OPS module-level constant
# ---------------------------------------------------------------------------


class TestBinaryOpsConstant(unittest.TestCase):
    def test_binary_ops_is_frozenset(self):
        self.assertIsInstance(BINARY_OPS, frozenset)

    def test_binary_ops_contains_expected(self):
        expected = {"add", "sub", "mul", "realdiv", "maximum", "minimum"}
        self.assertTrue(expected.issubset(BINARY_OPS))

    def test_binary_ops_contains_comparison(self):
        comparisons = {
            "equal",
            "notequal",
            "greaterthan",
            "greaterequal",
            "lesserthan",
            "lesserequal",
        }
        self.assertTrue(comparisons.issubset(BINARY_OPS))


if __name__ == "__main__":
    unittest.main()
