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


import pytest
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
from torch_spyre.ops.fallbacks import FallbackWarning


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

        torch.manual_seed(0xAFFE)

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


@instantiate_parametrized_tests
class TestIntDispatchFallback(TestCase):
    """Integer-typed inputs to ops registered via `register_torch_compile_kernel`
    must transparently fall back to CPU. The SDSC scheduler has no op mapping
    for integer dtypes today and aborts with "Scheduler failed to find a
    suitable op mapping" if one reaches the device compiler; without this
    fallback the eager dispatch SIGABRTs (or, worse, silently returns
    uninitialized memory — see issue #2376).
    """

    def test_int_add_emits_fallback_warning(self):
        # The eager dispatcher should emit FallbackWarning on int dispatch
        # rather than reaching the SDSC compiler.
        a = torch.arange(16, dtype=torch.int64, device="spyre")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FallbackWarning)
            _ = a + 2
        assert any(issubclass(w.category, FallbackWarning) for w in caught), (
            "expected FallbackWarning when adding int64 tensors on Spyre"
        )

    @parametrize(
        "dtype",
        [
            torch.int8,
            subtest(
                torch.int16,
                decorators=[
                    pytest.mark.skip(
                        reason=(
                            "int16 H2D copy aborts in the SDK's "
                            "sen_data_convert.cpp with 'Unsupported data "
                            "format types' — the type table advertises "
                            "SENINT16 but the runtime has no converter for "
                            "it, so the tensor never reaches dispatch and "
                            "the CPU fallback can't fire."
                        )
                    )
                ],
            ),
            torch.int32,
            torch.int64,
        ],
    )
    def test_int_add_matches_cpu(self, dtype):
        a_cpu = torch.arange(16, dtype=dtype)
        b_cpu = torch.arange(16, dtype=dtype) + 1
        out_cpu = a_cpu + b_cpu
        out_spyre = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
        assert_close(out_spyre, out_cpu)

    def test_int_add_scalar_matches_cpu(self):
        # Covers the exact scalar-add pattern from issue #2376 — without the
        # fallback, Spyre returned uninitialized 0xAAAAAAAA bytes rather than
        # the correct sum.
        a_cpu = torch.arange(16, dtype=torch.int64)
        out_cpu = a_cpu + 2
        out_spyre = (a_cpu.to("spyre") + 2).cpu()
        assert_close(out_spyre, out_cpu)

    def test_int_cat_with_list_arg_matches_cpu(self):
        # `aten.cat`'s first arg is a `list[Tensor]`. The dispatcher must look
        # through nested containers when checking for unsupported dtypes;
        # otherwise the int tensors hide inside the list and the kernel still
        # reaches SDSC.
        tensors_cpu = [
            torch.arange(16, dtype=torch.int64),
            torch.arange(16, 32, dtype=torch.int64),
        ]
        out_cpu = torch.cat(tensors_cpu)
        out_spyre = torch.cat([t.to("spyre") for t in tensors_cpu]).cpu()
        assert_close(out_spyre, out_cpu)

    def test_float_add_does_not_fall_back(self):
        # Regression guard: fp16 add must stay on-device (no FallbackWarning).
        # The fallback is dtype-conditional, not a blanket override.
        a = torch.rand(16, dtype=torch.float16, device="spyre")
        b = torch.rand(16, dtype=torch.float16, device="spyre")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FallbackWarning)
            _ = a + b
        assert not any(issubclass(w.category, FallbackWarning) for w in caught), (
            "fp16 add should not fall back to cpu"
        )


if __name__ == "__main__":
    run_tests()
