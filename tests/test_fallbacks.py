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


import torch
from torch.testing import assert_close
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
)
import warnings
from torch_spyre.fallbacks import FallbackWarning


class Op(subtest):
    def __init__(self, name, fn, rtol=None, atol=None):
        super().__init__(self, name)
        self.fn = fn
        self.rtol = rtol
        self.atol = atol


class FactoryOp(Op):
    """
    Represents an operator that creates a new tensor without consuming an existing
    tensor as input.

    Examples:
       - torch.arange
       - torch.full
    """

    inputs = [(64,)]

    def __init__(self, name, fn, inputs=None, rtol=None, atol=None):
        super().__init__(name, fn, rtol, atol)
        self.inputs = inputs or type(self).inputs


class UnaryOp(Op):
    inputs = [torch.rand(64, dtype=torch.float16)]

    def __init__(self, name, fn, inputs=None, rtol=None, atol=None):
        super().__init__(name, fn, rtol, atol)
        self.inputs = inputs or type(self).inputs


_factory_ops = [
    FactoryOp("arange", torch.arange, [(64.0,), (1.0, 65.0), (0.0, 128.0, 2.0)]),
]

_unary_ops = [
    UnaryOp("sin", torch.sin),
    UnaryOp("cos", torch.cos),
]


@instantiate_parametrized_tests
class TestFallbacks(TestCase):
    def setUp(self):
        self.rtol = 1e-2
        self.atol = 1e-3
        self.dtype = torch.float16

        torch.random.manual_seed(42)

        warnings.simplefilter("ignore", FallbackWarning)

    def _assert_close(self, op, output_spyre, output_cpu):
        rtol = op.rtol or self.rtol
        atol = op.atol or self.atol
        assert_close(output_spyre, output_cpu, rtol=rtol, atol=atol)

    @parametrize("op", _factory_ops)
    def test_factory_op(self, op):
        for input in op.inputs:
            output_cpu = op.fn(*input, dtype=self.dtype, device="cpu")
            output_spyre = op.fn(*input, dtype=self.dtype, device="spyre")

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _factory_ops)
    def test_factory_op_out(self, op):
        for input in op.inputs:
            buffer_cpu = torch.empty(0)
            output_cpu = op.fn(*input, out=buffer_cpu)

            buffer_spyre = torch.empty_like(output_cpu, device="spyre")
            output_spyre = op.fn(*input, out=buffer_spyre)

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _unary_ops)
    def test_unary_op(self, op):
        for input in op.inputs:
            output_cpu = op.fn(input)
            output_spyre = op.fn(input.to("spyre"))

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _unary_ops)
    def test_unary_op_out(self, op):
        for input in op.inputs:
            buffer_cpu = torch.empty_like(input)
            output_cpu = op.fn(input, out=buffer_cpu)

            buffer_spyre = torch.empty_like(input, device="spyre")
            output_spyre = op.fn(input.to("spyre"), out=buffer_spyre)

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _unary_ops)
    def test_unary_op_out_alias(self, op):
        for input in op.inputs:
            buffer_cpu = torch.clone(input)
            output_cpu = op.fn(buffer_cpu, out=buffer_cpu)

            buffer_spyre = torch.clone(input).to(device="spyre")
            output_spyre = op.fn(buffer_spyre, out=buffer_spyre)

            self._assert_close(op, output_spyre.cpu(), output_cpu)
            self.assertEqual(id(buffer_spyre), id(output_spyre))


if __name__ == "__main__":
    run_tests()
