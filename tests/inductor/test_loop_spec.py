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

"""Unit tests for LoopSpec data structure and codegen_kernel serialization.

These tests exercise op_spec.LoopSpec and the helpers added to spyre_kernel
(_iter_op_specs, _codegen_op_spec_list, wrap_op_specs_in_loop) without
requiring a Spyre device or a full compilation pipeline.

The round-trip strategy: build an op_specs list by hand, call
_codegen_op_spec_list to produce Python source, then eval() it in a
namespace that has the required constructors in scope and assert that the
resulting objects match the originals.
"""

import unittest

from sympy import Integer, Symbol, sympify  # noqa: F401

from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list, _iter_op_specs
from torch._inductor.utils import IndentedBuffer
from torch_spyre._C import DataFormats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVAL_NS = {
    "LoopSpec": LoopSpec,
    "OpSpec": OpSpec,
    "TensorArg": TensorArg,
    "UnimplementedOp": UnimplementedOp,
    "DataFormats": DataFormats,
    "sympify": sympify,
}


def _make_tensor_arg(arg_index: int = 0, is_input: bool = True) -> TensorArg:
    x = Symbol("x0")
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 64],
        device_coordinates=[x, Integer(0)],
        allocation=None,
    )


def _make_op_spec(op: str = "add", arg_index: int = 0) -> OpSpec:
    x0 = Symbol("x0")
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={x0: (Integer(128), 1)},
        args=[
            _make_tensor_arg(arg_index=arg_index, is_input=True),
            _make_tensor_arg(arg_index=arg_index + 1, is_input=False),
        ],
        op_info={},
    )


def _roundtrip(specs):
    """Serialize specs to Python source and eval back."""

    def sympy_str(x):
        return "sympify('" + str(x) + "')"

    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(specs, buf, sympy_str)
    buf.writeline("]")
    return eval(buf.getvalue(), _EVAL_NS)  # noqa: S307


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoopSpecDataclass(unittest.TestCase):
    def test_flat_body(self):
        op = _make_op_spec()
        loop = LoopSpec(count=Integer(4), body=[op])
        self.assertEqual(loop.count, Integer(4))
        self.assertEqual(len(loop.body), 1)
        self.assertIs(loop.body[0], op)

    def test_nested_body(self):
        inner = LoopSpec(count=Integer(2), body=[_make_op_spec("mul")])
        outer = LoopSpec(count=Integer(4), body=[_make_op_spec("add"), inner])
        self.assertEqual(len(outer.body), 2)
        self.assertIsInstance(outer.body[1], LoopSpec)

    def test_empty_body(self):
        loop = LoopSpec(count=Integer(8), body=[])
        self.assertEqual(loop.body, [])


class TestIterOpSpecs(unittest.TestCase):
    def test_flat_list(self):
        specs = [_make_op_spec("add"), _make_op_spec("mul")]
        result = list(_iter_op_specs(specs))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].op, "add")
        self.assertEqual(result[1].op, "mul")

    def test_skips_unimplemented(self):
        specs = [UnimplementedOp(op="foo"), _make_op_spec("add")]
        result = list(_iter_op_specs(specs))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].op, "add")

    def test_single_level_loop(self):
        inner = [_make_op_spec("add"), _make_op_spec("mul")]
        specs = [LoopSpec(count=Integer(4), body=inner)]
        result = list(_iter_op_specs(specs))
        self.assertEqual([s.op for s in result], ["add", "mul"])

    def test_nested_loop_depth_first(self):
        innermost = [_make_op_spec("c")]
        middle = [_make_op_spec("b"), LoopSpec(count=Integer(2), body=innermost)]
        specs = [_make_op_spec("a"), LoopSpec(count=Integer(4), body=middle)]
        result = list(_iter_op_specs(specs))
        self.assertEqual([s.op for s in result], ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(list(_iter_op_specs([])), [])


class TestCodegenOpSpecListRoundtrip(unittest.TestCase):
    def test_flat_op_spec(self):
        original = [_make_op_spec("add")]
        result = _roundtrip(original)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], OpSpec)
        self.assertEqual(result[0].op, "add")

    def test_unimplemented_op(self):
        original = [UnimplementedOp(op="unknown")]
        result = _roundtrip(original)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], UnimplementedOp)
        self.assertEqual(result[0].op, "unknown")

    def test_single_loop_wrapping_two_ops(self):
        body = [_make_op_spec("add"), _make_op_spec("mul")]
        original = [LoopSpec(count=Integer(4), body=body)]
        result = _roundtrip(original)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], LoopSpec)
        self.assertEqual(result[0].count, Integer(4))
        self.assertEqual(len(result[0].body), 2)
        self.assertEqual(result[0].body[0].op, "add")
        self.assertEqual(result[0].body[1].op, "mul")

    def test_nested_loop(self):
        inner_loop = LoopSpec(count=Integer(2), body=[_make_op_spec("inner")])
        original = [
            LoopSpec(count=Integer(8), body=[_make_op_spec("outer"), inner_loop])
        ]
        result = _roundtrip(original)
        outer = result[0]
        self.assertIsInstance(outer, LoopSpec)
        self.assertEqual(outer.count, Integer(8))
        self.assertEqual(outer.body[0].op, "outer")
        inner = outer.body[1]
        self.assertIsInstance(inner, LoopSpec)
        self.assertEqual(inner.count, Integer(2))
        self.assertEqual(inner.body[0].op, "inner")

    def test_symbolic_count(self):
        s = Symbol("s0")
        original = [LoopSpec(count=s, body=[_make_op_spec("add")])]
        result = _roundtrip(original)
        self.assertIsInstance(result[0], LoopSpec)
        self.assertEqual(result[0].count, s)

    def test_mixed_flat_and_loop(self):
        original = [
            _make_op_spec("before"),
            LoopSpec(count=Integer(4), body=[_make_op_spec("body")]),
            _make_op_spec("after"),
        ]
        result = _roundtrip(original)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], OpSpec)
        self.assertIsInstance(result[1], LoopSpec)
        self.assertIsInstance(result[2], OpSpec)

    def test_arg_index_preserved(self):
        arg = _make_tensor_arg(arg_index=3)
        op = OpSpec(
            op="relu",
            is_reduction=False,
            iteration_space={Symbol("x0"): (Integer(64), 1)},
            args=[arg],
            op_info={},
        )
        original = [LoopSpec(count=Integer(2), body=[op])]
        result = _roundtrip(original)
        self.assertEqual(result[0].body[0].args[0].arg_index, 3)


if __name__ == "__main__":
    unittest.main()
