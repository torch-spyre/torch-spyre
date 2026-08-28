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
    STICK_STAGE,
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
        op="batchmatmul",
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
        op = OpSpec(
            op="synthetic_test_op",
            is_reduction=False,
            iteration_space={
                _C_ROW: (Integer(128), 1),
                _C_COL: (Integer(256), 1),
            },
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
            validate_op_specs([op], stage="before_bundle_generation")
        self.assertIn("exactly one of hbm/lx/hbm_pool", str(ctx.exception))

    def test_allocation_multiple_keys(self):
        """allocation must not contain more than one valid key."""
        op = _make_valid_op_spec()
        op.args[0] = dataclasses.replace(
            op.args[0], allocation={"hbm": 0x1000, "lx": 0x2000}
        )
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage="before_bundle_generation")
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


# ---------------------------------------------------------------------------
# Tests: stick constraints (OS-8)
# ---------------------------------------------------------------------------


class TestStickConstraints(unittest.TestCase):
    def test_uniform_same_stick_passes(self):
        validate_op_specs([_make_valid_op_spec("add")], stage=STICK_STAGE)

    def test_uniform_different_sticks_raises(self):
        c_other = Symbol("c_other")
        op = _make_valid_op_spec("add")
        op.iteration_space[c_other] = (Integer(64), 1)
        op.args[1] = dataclasses.replace(
            op.args[1],
            device_coordinates=[_C_COL // 64, _C_ROW, sympy.Mod(c_other, 64)],
        )
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage=STICK_STAGE)
        self.assertIn("different stick loop variables", str(ctx.exception))

    def test_restickify_different_sticks_passes(self):
        c_other = Symbol("c_other")
        op = _make_valid_op_spec("ReStickifyOpHBM")
        op.iteration_space[c_other] = (Integer(64), 1)
        op.args = [
            _make_tensor_arg(is_input=True, arg_index=0),
            dataclasses.replace(
                _make_tensor_arg(is_input=False, arg_index=1),
                device_coordinates=[_C_COL // 64, _C_ROW, sympy.Mod(c_other, 64)],
            ),
        ]
        validate_op_specs([op], stage=STICK_STAGE)

    def test_restickify_same_sticks_raises(self):
        op = _make_valid_op_spec("ReStickifyOpHBM")
        op.args = [
            _make_tensor_arg(is_input=True, arg_index=0),
            _make_tensor_arg(is_input=False, arg_index=1),
        ]
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs([op], stage=STICK_STAGE)
        self.assertIn(
            "ReStickifyOpHBM input and output must have different stick loop variables",
            str(ctx.exception),
        )


# ---------------------------------------------------------------------------
# Tests: _check_stick_norm (exx2)
# ---------------------------------------------------------------------------

# For these ops the input has the normalization (reduction) dim in its coords
# and the output does not — the reduction symbol is _C_ROW here.
# The stick symbol (_C_COL) must equal the reduction symbol to be valid, so
# we build fixtures where either _C_ROW or _C_COL is the stick.


def _make_norm_op_spec(op: str, stick_is_reduction_dim: bool) -> OpSpec:
    """Build a minimal exx2-like reduction OpSpec.

    Input has both _C_ROW and _C_COL in its coords; output only has _C_COL.
    So _C_ROW is the reduction symbol.

    If stick_is_reduction_dim=True, the input stick coord uses _C_ROW (valid).
    If False, the input stick coord uses _C_COL (invalid — wrong dim in stick).
    """
    if stick_is_reduction_dim:
        # Stick = _C_ROW (the reduction dim) — valid
        in_coords = [_C_COL // 64, sympy.Mod(_C_ROW, 64)]
        in_size = [4, 64]
    else:
        # Stick = _C_COL (a non-reduction dim) — invalid
        in_coords = [_C_ROW, sympy.Mod(_C_COL, 64)]
        in_size = [128, 64]

    out_coords = [_C_COL // 64, sympy.Mod(_C_COL, 64)]
    out_size = [4, 64]

    input_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=in_size,
        device_coordinates=in_coords,
        allocation={"hbm": 0x400000000},
    )
    output_arg = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=out_size,
        device_coordinates=out_coords,
        allocation={"hbm": 0x500000000},
    )
    return OpSpec(
        op=op,
        is_reduction=True,
        iteration_space={
            _C_ROW: (Integer(64), 1),
            _C_COL: (Integer(256), 1),
        },
        args=[input_arg, output_arg],
        op_info={},
        tiled_symbols=[],
    )


class TestNormStick(unittest.TestCase):
    def test_exx2_stick_is_reduction_dim_passes(self):
        validate_op_specs(
            [_make_norm_op_spec("exx2", stick_is_reduction_dim=True)],
            stage=STICK_STAGE,
        )

    def test_exx2_stick_is_non_reduction_dim_raises(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(
                [_make_norm_op_spec("exx2", stick_is_reduction_dim=False)],
                stage=STICK_STAGE,
            )
        self.assertIn("stick symbol must be the reduction", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: _check_topk_stick
# ---------------------------------------------------------------------------

# Topk coord layout:
#   input:  [_C_FEAT (reduction dim), Mod(_C_STICK, 64)]
#   output: [_C_K    (k dim),         Mod(_C_STICK, 64)]
# _C_FEAT is input-only  → reduction symbol.
# _C_K    is output-only → k symbol.
# _C_STICK is shared     → surviving stick dimension (valid in stick).

_C_FEAT = Symbol("c_feat")
_C_K = Symbol("c_k")
_C_STICK = Symbol("c_stick")


def _make_topk_op_spec(
    op: str,
    input_stick: sympy.Expr,
    output_stick: sympy.Expr,
) -> OpSpec:
    """Build a minimal topkvalue/topkindex-like OpSpec.

    Input coords: [_C_FEAT, input_stick]
    Output coords: [_C_K,   output_stick]

    _C_FEAT is the reduction symbol (input-only).
    _C_K is the k symbol (output-only).
    _C_STICK (used in the default valid case) is a shared surviving symbol.
    """
    input_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[_C_FEAT, input_stick],
        allocation={"hbm": 0x400000000},
    )
    output_arg = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 64],
        device_coordinates=[_C_K, output_stick],
        allocation={"hbm": 0x500000000},
    )
    return OpSpec(
        op=op,
        is_reduction=True,
        iteration_space={
            _C_FEAT: (Integer(128), 1),
            _C_K: (Integer(8), 1),
            _C_STICK: (Integer(256), 1),
        },
        args=[input_arg, output_arg],
        op_info={},
        tiled_symbols=[],
    )


class TestTopkStick(unittest.TestCase):
    def test_topkvalue_surviving_stick_passes(self):
        validate_op_specs(
            [
                _make_topk_op_spec(
                    "topkvalue", sympy.Mod(_C_STICK, 64), sympy.Mod(_C_STICK, 64)
                )
            ],
            stage=STICK_STAGE,
        )

    def test_topkvalue_reduction_dim_in_stick_raises(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(
                [
                    _make_topk_op_spec(
                        "topkvalue",
                        sympy.Mod(_C_FEAT, 64),
                        sympy.Mod(_C_STICK, 64),
                    )
                ],
                stage=STICK_STAGE,
            )
        self.assertIn("reduction or k dimension", str(ctx.exception))

    def test_topkindex_surviving_stick_passes(self):
        validate_op_specs(
            [
                _make_topk_op_spec(
                    "topkindex", sympy.Mod(_C_STICK, 64), sympy.Mod(_C_STICK, 64)
                )
            ],
            stage=STICK_STAGE,
        )

    def test_topkindex_k_dim_in_stick_raises(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(
                [
                    _make_topk_op_spec(
                        "topkindex",
                        sympy.Mod(_C_STICK, 64),
                        sympy.Mod(_C_K, 64),  # k dim in output stick — invalid
                    )
                ],
                stage=STICK_STAGE,
            )
        self.assertIn("reduction or k dimension", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: _check_matmul_stick
# ---------------------------------------------------------------------------

# Batchmatmul semantic dimensions:
#   _C_K_MM  = reduction_sym  (in Input1, Input2; absent from Output)
#   _C_N_MM  = generated_sym  (in Input2, Output; absent from Input1)
#   _C_M_MM  = preserved_sym  (in Input1, Output; absent from Input2)
#   _C_B_MM  = noreuse_sym    (in all three — batch dim)
#
# Required stick symbols:
#   Input1 stick = _C_K_MM  (reduction)
#   Input2 stick = _C_N_MM  (generated)
#   Output stick = _C_N_MM  (generated)

_C_K_MM = Symbol("c_k_mm")
_C_N_MM = Symbol("c_n_mm")
_C_M_MM = Symbol("c_m_mm")
_C_B_MM = Symbol("c_b_mm")


def _make_bmm_stick_op_spec(
    op: str,
    input1_stick: sympy.Expr,
    input2_stick: sympy.Expr,
    output_stick: sympy.Expr,
) -> OpSpec:
    """Build a minimal batchmatmul-like OpSpec.

    Input1 (x): coords [_C_B_MM, _C_M_MM, input1_stick]  (K is reduction_sym)
    Input2 (y): coords [_C_B_MM, _C_K_MM, input2_stick]  (N is generated_sym)
    Output:     coords [_C_B_MM, _C_M_MM, output_stick]
    """
    input1 = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 16, 64],
        device_coordinates=[_C_B_MM, _C_M_MM, input1_stick],
        allocation={"hbm": 0x400000000},
    )
    input2 = TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 64, 64],
        device_coordinates=[_C_B_MM, _C_K_MM, input2_stick],
        allocation={"hbm": 0x500000000},
    )
    output = TensorArg(
        is_input=False,
        arg_index=2,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 16, 64],
        device_coordinates=[_C_B_MM, _C_M_MM, output_stick],
        allocation={"hbm": 0x600000000},
    )
    return OpSpec(
        op=op,
        is_reduction=True,
        iteration_space={
            _C_B_MM: (Integer(8), 1),
            _C_K_MM: (Integer(64), 1),
            _C_N_MM: (Integer(64), 1),
            _C_M_MM: (Integer(16), 1),
        },
        args=[input1, input2, output],
        op_info={},
        tiled_symbols=[],
    )


class TestMatmulStick(unittest.TestCase):
    def test_batchmatmul_valid_passes(self):
        validate_op_specs(
            [
                _make_bmm_stick_op_spec(
                    "batchmatmul",
                    sympy.Mod(_C_K_MM, 64),
                    sympy.Mod(_C_N_MM, 64),
                    sympy.Mod(_C_N_MM, 64),
                )
            ],
            stage=STICK_STAGE,
        )

    def test_input1_wrong_stick_raises(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(
                [
                    _make_bmm_stick_op_spec(
                        "batchmatmul",
                        sympy.Mod(_C_N_MM, 64),  # wrong: N instead of K
                        sympy.Mod(_C_N_MM, 64),
                        sympy.Mod(_C_N_MM, 64),
                    )
                ],
                stage=STICK_STAGE,
            )
        self.assertIn("Input1 stick", str(ctx.exception))

    def test_input2_wrong_stick_raises(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(
                [
                    _make_bmm_stick_op_spec(
                        "batchmatmul",
                        sympy.Mod(_C_K_MM, 64),
                        sympy.Mod(_C_K_MM, 64),  # wrong: K instead of N
                        sympy.Mod(_C_N_MM, 64),
                    )
                ],
                stage=STICK_STAGE,
            )
        self.assertIn("Input2 stick", str(ctx.exception))

    def test_output_wrong_stick_raises(self):
        with self.assertRaises(OpSpecValidationError) as ctx:
            validate_op_specs(
                [
                    _make_bmm_stick_op_spec(
                        "batchmatmul",
                        sympy.Mod(_C_K_MM, 64),
                        sympy.Mod(_C_N_MM, 64),
                        sympy.Mod(_C_K_MM, 64),  # wrong: K instead of N
                    )
                ],
                stage=STICK_STAGE,
            )
        self.assertIn("Output stick", str(ctx.exception))

    def test_batchmatmulfp8_valid_passes(self):
        validate_op_specs(
            [
                _make_bmm_stick_op_spec(
                    "batchmatmulfp8",
                    sympy.Mod(_C_K_MM, 64),
                    sympy.Mod(_C_N_MM, 64),
                    sympy.Mod(_C_N_MM, 64),
                )
            ],
            stage=STICK_STAGE,
        )


if __name__ == "__main__":
    unittest.main()
