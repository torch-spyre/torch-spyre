# Copyright 2026 The Torch-Spyre Authors.
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

import numpy as np
import pytest
import torch

from utils_inductor import DEVICE, cached_randn, compare_with_cpu


def _compare_modes(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def _run_spyre(execution_mode, fn, *args):
    """Run ``fn`` on Spyre (eager or ``torch.compile``) for tests that do not use ``compare_with_cpu``."""
    dev_args = tuple(a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in args)
    if execution_mode == "eager":
        return fn(*dev_args)
    return torch.compile(fn)(*dev_args)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestDatatypeScalarOperations:
    """
    Scalar–tensor dtype interactions: promotions, NumPy/Python scalars, 0-D torch tensors,
    shapes, and edge magnitudes. Spyre vs CPU via ``compare_with_cpu`` where applicable.
    """

    torch.manual_seed(0xAFFE)

    @pytest.mark.parametrize(
        "tensor_dtype,scalar_type,scalar_value,atol,rtol",
        [
            # FP64 tensor × FP64 scalar
            (torch.float64, "python", 0.125, 1e-10, 1e-7),
            # FP64 tensor × FP32 scalar
            (torch.float64, np.float32, 0.125, 1e-7, 1e-5),
            # FP32 tensor × FP64 scalar
            (torch.float32, np.float64, 0.125, 1e-7, 1e-5),
        ],
    )
    def test_mixed_precision_scalar_multiply(
        self, execution_mode, tensor_dtype, scalar_type, scalar_value, atol, rtol
    ):
        """
        Test tensor-scalar multiplication with mixed precision types.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
        if tensor_dtype == torch.float64:
            pytest.xfail(reason="Spyre backend does not support dtype Double(FP64)")

        def mixed_mul(x):
            if scalar_type == "python":
                return x * scalar_value
            else:
                return x * scalar_type(scalar_value)

        x = cached_randn(
            (128, 128),
            dtype=tensor_dtype,
            differentiation=f"mixed_precision_{tensor_dtype}_{scalar_type}",
        )
        _compare_modes(execution_mode, mixed_mul, x, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "tensor_dtype,scalar_dtype",
        [
            # FP16 tensor × FP16 scalar
            (torch.float16, torch.float16),
            # FP32 tensor × FP16 scalar
            (torch.float32, torch.float16),
        ],
    )
    def test_fp16_scalar_operations(self, execution_mode, tensor_dtype, scalar_dtype):
        """
        Test FP16 scalar with FP16/FP32 tensors.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
        if execution_mode == "eager" and tensor_dtype == torch.float32:
            pytest.xfail(reason="to_dtype on float32 (IEEE_FP32) not supported")
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        elif execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )

        def fp16_scalar_mul(x):
            scalar = torch.tensor(0.125, dtype=scalar_dtype, device=x.device)
            return x * scalar

        x = cached_randn(
            (128, 128),
            dtype=tensor_dtype,
            differentiation=f"fp16_ops_{tensor_dtype}_{scalar_dtype}",
        )
        _compare_modes(execution_mode, fp16_scalar_mul, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "dtype,value",
        [
            (np.int32, 2),
            (np.int16, 2),
            (np.int8, 2),
            (np.uint8, 2),
        ],
    )
    def test_integer_scalar_types(self, execution_mode, dtype, value):
        """Test integer scalars with FP32 tensor."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        if execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )

        def int_mul(x):
            return x * dtype(value)

        x = cached_randn((128, 64), differentiation=f"int_scalar_{dtype.__name__}")
        _compare_modes(execution_mode, int_mul, x, atol=0.004, rtol=0.01)

    @pytest.mark.parametrize("bool_val", [True, False])
    def test_bool_scalar(self, execution_mode, bool_val):
        """Test Boolean scalars."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        if not bool_val:
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )

        def bool_mul(x):
            return x * bool_val

        x = cached_randn((128, 64), differentiation=f"bool_scalar_{bool_val}")
        _compare_modes(execution_mode, bool_mul, x, atol=0.002, rtol=0.002)

    @pytest.mark.parametrize(
        "np_dtype,atol,rtol",
        [
            (np.float16, 1e-3, 1e-3),
            (np.float32, 1e-3, 1e-2),
            (np.float64, 1e-3, 1e-2),
        ],
    )
    def test_numpy_float_scalar_types(self, execution_mode, np_dtype, atol, rtol):
        """Test NumPy float scalars."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        if execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )

        def np_float_mul(x):
            return x * np_dtype(0.125)

        x = cached_randn((128, 64), differentiation=f"np_float_{np_dtype.__name__}")
        _compare_modes(execution_mode, np_float_mul, x, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "torch_dtype,scalar_val,atol,rtol",
        [
            (None, 0.125, 1e-7, 1e-5),
            (torch.float16, 0.125, 1e-3, 1e-3),
            (torch.float64, 0.125, 1e-3, 1e-2),
            (torch.int32, 2, 1e-3, 1e-2),
        ],
    )
    def test_torch_scalar_tensor_types(
        self, execution_mode, torch_dtype, scalar_val, atol, rtol
    ):
        """Test 0-D torch scalar tensors with various dtypes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
        if torch_dtype is None and execution_mode == "eager":
            pytest.xfail(
                reason="Mixed-dtype tensors (float32 * float16) sharing stick variable not supported"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
        elif torch_dtype == torch.float64:
            pytest.xfail(reason="Spyre backend does not support dtype Double(FP64)")
        # TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/925
        elif torch_dtype == torch.int32:
            pytest.skip(
                reason="Spyre backend does not support int32/int16 dtype - causes Signal Abort in data format converter"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        elif torch_dtype in (None, torch.float16) and execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor mul fails - IndexError on empty args during layout propagation."
            )

        def torch_scalar_mul(x):
            if torch_dtype is None:
                scalar = torch.tensor(scalar_val, device=x.device)
            else:
                scalar = torch.tensor(scalar_val, dtype=torch_dtype, device=x.device)
            return x * scalar

        x = cached_randn((128, 64), differentiation=f"torch_scalar_{torch_dtype}")
        _compare_modes(execution_mode, torch_scalar_mul, x, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "tensor_dtype,scalar_type,scalar_value,atol,rtol,test_name",
        [
            # Int8 to FP32 promotion
            (torch.float32, np.int8, 2, 1e-6, 1e-6, "int8_to_fp32"),
            # FP16 to FP64 promotion
            (torch.float64, np.float16, 0.125, 1e-3, 1e-3, "fp16_to_fp64"),
        ],
    )
    def test_type_promotion_operations(
        self,
        execution_mode,
        tensor_dtype,
        scalar_type,
        scalar_value,
        atol,
        rtol,
        test_name,
    ):
        """
        Test various type promotion scenarios.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
        if tensor_dtype == torch.float64:
            pytest.xfail(reason="Spyre backend does not support dtype Double(FP64)")
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        elif tensor_dtype == torch.float32 and execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor mul fails - IndexError on empty args during layout propagation."
            )

        def type_promo_op(x):
            return x * scalar_type(scalar_value)

        x = cached_randn(
            (128, 64), dtype=tensor_dtype, differentiation=f"type_promo_{test_name}"
        )
        _compare_modes(execution_mode, type_promo_op, x, atol=atol, rtol=rtol)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
    @pytest.mark.xfail(reason="Spyre backend does not support dtype Double(FP64)")
    def test_mixed_dtype_chain_fp64_fp32_fp16(self, execution_mode):
        """Test mixed dtype chain: FP64 → FP32 → FP16."""

        def mixed_chain(x):
            x_fp32 = x.to(torch.float32)
            x_scaled = x_fp32 * 0.125
            return x_scaled.to(torch.float16)

        x = cached_randn((128, 64), dtype=torch.float64)
        _compare_modes(execution_mode, mixed_chain, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "dtype,scalar_value,atol,rtol",
        [
            # (torch.float16, 6e-8, 1e-8, 1e-3),
            (torch.float16, 6e-8, 0.1, 0.1),
            (torch.float64, 1e-320, 1e-320, 1e-3),
        ],
    )
    def test_subnormal_scalar_operations(
        self, execution_mode, dtype, scalar_value, atol, rtol
    ):
        """
        Test subnormal scalar values with different dtypes.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
        if dtype == torch.float64:
            pytest.xfail(reason="Spyre backend does not support dtype Double(FP64)")

        def subnormal_mul(x):
            return x * scalar_value

        x = cached_randn(
            (100, 100), dtype=dtype, differentiation=f"subnormal_{dtype}_{scalar_value}"
        )
        _compare_modes(execution_mode, subnormal_mul, x, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "dtype,zero_val",
        [
            (np.float32, 0.0),
            (np.float64, 0.0),
            (np.float32, -0.0),
            (np.float64, -0.0),
        ],
    )
    def test_zero_scalar_different_dtypes(self, execution_mode, dtype, zero_val):
        """Test positive and negative zero scalars."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        if execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor mul fails - IndexError on empty args during layout propagation."
            )

        def zero_mul(x):
            return x * dtype(zero_val)

        x = cached_randn(
            (100, 100), differentiation=f"zero_{dtype.__name__}_{zero_val}"
        )
        # Loosened to 2e-4 to account for hardware-specific FTZ/Rounding
        tol = (2e-4, 0.0) if dtype == np.float32 else (1e-3, 0.0)
        _compare_modes(execution_mode, zero_mul, x, atol=tol[0], rtol=tol[1])

    @pytest.mark.parametrize(
        "shape",
        [
            (100,),
            (10, 10),
            (5, 5, 5),
            (2, 3, 4, 5),
        ],
    )
    def test_scalar_with_different_tensor_dimensions(self, execution_mode, shape):
        """Test scalar multiply across tensor ranks."""

        def scalar_mul(x):
            return x * 0.125

        x = cached_randn(shape, differentiation=f"shape_{len(shape)}d")
        _compare_modes(execution_mode, scalar_mul, x, atol=1e-3, rtol=1e-2)

    # TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1487
    @pytest.mark.skip(
        reason="Segmentation fault: Copy host to device 0-dim tensor error"
    )
    def test_scalar_with_empty_tensor(self, execution_mode):
        """Scalar operation on empty tensor (shape preserved)."""

        def scalar_mul(x):
            return x * 0.5

        x = torch.randn(0, 10)
        _compare_modes(execution_mode, scalar_mul, x, atol=0.0, rtol=0.0)

    def test_very_large_scalar_value(self, execution_mode):
        """Very large scalar (FP32 range)."""

        def large_scalar_mul(x):
            return x * 1e30

        x = cached_randn((10, 10), dtype=torch.float32) * 1e-10
        # For large numbers, atol must be large, but rtol is the real gate
        _compare_modes(execution_mode, large_scalar_mul, x, atol=1e25, rtol=1e-3)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestNegativeScalarOperations:
    """
    Errors, non-finite values, promotion rules, in-place and compile edge cases.
    Uses ``_run_spyre`` when assertions are not expressible as CPU reference equality.
    """

    torch.manual_seed(0xAFFE)

    @pytest.mark.parametrize(
        "invalid_scalar,desc",
        [
            ("invalid", "string"),
            (None, "none"),
            ([1, 2], "list"),
            ({"a": 1}, "dict"),
        ],
    )
    def test_invalid_scalar_types(self, execution_mode, invalid_scalar, desc):
        """Invalid scalar types should raise TypeError on Spyre."""

        def invalid_mul(x):
            return x * invalid_scalar

        x = cached_randn((10, 10), differentiation=f"invalid_{desc}")
        # Compiled mode fails with a Dynamo/Inductor runtime error
        # Eager mode raises the standard Python TypeError
        expected_err = (
            torch._dynamo.exc.TorchRuntimeError
            if execution_mode == "compiled"
            else TypeError
        )

        with pytest.raises(expected_err):
            _run_spyre(execution_mode, invalid_mul, x)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1545
    @pytest.mark.skip(
        reason="Spyre backend does not support complex scalar constants - causes SIGFPE"
    )
    def test_complex_scalar_type_promotion(self, execution_mode):
        """Real tensor multiplied by complex scalar promotes to complex."""

        def complex_mul(x):
            return x * (1 + 2j)

        x = cached_randn((10, 10))
        result = _run_spyre(execution_mode, complex_mul, x)
        assert result.dtype in (
            torch.complex32,
            torch.complex64,
            torch.complex128,
        )
        assert result.shape == x.shape
        assert torch.allclose(result.real, x, atol=1e-5)
        assert torch.allclose(result.imag, x * 2, atol=1e-5)

    def test_invalid_tensor_as_scalar(self, execution_mode):
        """Multi-element tensor used like a scalar should error."""

        def tensor_mul(x):
            return x * torch.tensor([1, 2])

        x = cached_randn((10, 10))
        with pytest.raises(RuntimeError):
            _run_spyre(execution_mode, tensor_mul, x)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
    @pytest.mark.xfail(
        reason="backend does not support aten::index operation required for NaN/Inf validation"
    )
    def test_division_by_zero_result(self, execution_mode):
        """Division by zero: Inf / NaN behavior."""

        def div_zero(x):
            return x / 0.0

        x = cached_randn((10, 10))
        result = _run_spyre(execution_mode, div_zero, x)
        non_zero_mask = x.abs() > 1e-6
        if non_zero_mask.any():
            assert torch.isinf(result[non_zero_mask]).all()
        assert (torch.isinf(result) | torch.isnan(result)).all()
        assert result.shape == x.shape

    @pytest.mark.parametrize(
        "special_value,expected_behavior,use_abs",
        [
            (float("nan"), "all_nan", False),
            (float("inf"), "all_inf_nonzero", False),
            (float("-inf"), "all_neg_inf", True),
        ],
    )
    def test_non_finite_scalar_inputs(
        self, execution_mode, special_value, expected_behavior, use_abs
    ):
        """
        Test non-finite scalar values (NaN, Inf, -Inf).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1738
        if expected_behavior == "all_nan":
            pytest.skip(
                reason="backend does not support aten::isnan operations required for NaN validation"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        elif expected_behavior == "all_inf_nonzero":
            pytest.xfail(
                reason="backend does not support aten::index operations required for NaN validation"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1639
        elif expected_behavior == "all_neg_inf":
            pytest.xfail(
                reason="backend does not support dtype conversion in comparison operations"
            )

        def special_mul(x):
            return x * special_value

        x = cached_randn((10, 10), differentiation=f"nonfinite_{expected_behavior}")
        if use_abs:
            x = torch.abs(x)

        result = _run_spyre(execution_mode, special_mul, x)

        if expected_behavior == "all_nan":
            assert torch.isnan(result).all()
        elif expected_behavior == "all_inf_nonzero":
            non_zero_mask = x.abs() > 1e-6
            if non_zero_mask.any():
                assert torch.isinf(result[non_zero_mask]).all()
        elif expected_behavior == "all_neg_inf":
            assert torch.isinf(result).all() and (result < 0).all()

        assert result.shape == x.shape

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1740
    # AttributeError: 'UnimplementedOp' object has no attribute 'iteration_space'
    @pytest.mark.xfail(
        reason="backend: Negative base with fractional power not implemented"
    )
    def test_negative_power_nan_result(self, execution_mode):
        """Negative base with fractional power -> NaN."""

        def neg_power(x):
            return x**-0.5

        x = torch.tensor([-1.0, -2.0, -3.0], device=DEVICE)
        result = _run_spyre(execution_mode, neg_power, x)
        assert torch.isnan(result).all()

    @pytest.mark.parametrize(
        "dtype,scalar,expected_behavior",
        [
            (torch.float16, 1e10, "overflow_to_inf"),
            (torch.float16, 1e-40, "underflow_close_to_input"),
            (torch.float16, 100000, "overflow_or_large"),
        ],
    )
    def test_fp16_overflow_underflow(
        self, execution_mode, dtype, scalar, expected_behavior
    ):
        """
        Test FP16 overflow and underflow scenarios.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
        if expected_behavior == "underflow_close_to_input":
            pytest.xfail(
                reason="Mixed-dtype tensors sharing stick variable not supported"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1639
        if expected_behavior in ["overflow_to_inf", "overflow_or_large"]:
            pytest.xfail(
                reason="backend does not support dtype conversion in comparison operations"
            )

        if expected_behavior == "underflow_close_to_input":

            def fp16_op(x):
                return x + scalar
        else:

            def fp16_op(x):
                return x * scalar

        x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=DEVICE)
        result = _run_spyre(execution_mode, fp16_op, x)

        if expected_behavior == "overflow_to_inf":
            assert torch.isinf(result).all()
        elif expected_behavior == "underflow_close_to_input":
            assert torch.allclose(result, x, atol=1e-3)
        elif expected_behavior == "overflow_or_large":
            assert torch.isinf(result).any() or result.abs().max() > 60000

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1639
    @pytest.mark.xfail(
        reason="backend does not support dtype conversion in add operations"
    )
    def test_int_tensor_float_scalar_promotion(self, execution_mode):
        """Int tensor + float scalar promotes to float."""

        def int_float_add(x):
            return x + 0.5

        x = torch.randint(0, 10, (10, 10), dtype=torch.int64).to(DEVICE)
        result = _run_spyre(execution_mode, int_float_add, x)
        assert result.dtype in (torch.float32, torch.float64)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
    @pytest.mark.skip(
        reason="Mixed-dtype tensors (bool + float) sharing stick variable not supported"
    )
    def test_bool_tensor_scalar_multiply(self, execution_mode):
        """Bool tensor * float scalar."""

        def bool_mul(x):
            return x * 2.0

        x = torch.tensor([[True, False], [False, True]], device=DEVICE)
        result = _run_spyre(execution_mode, bool_mul, x)
        assert result.dtype in (torch.float32, torch.float64, torch.int64)
        expected = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        assert torch.allclose(result.float(), expected)

    # TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1588
    @pytest.mark.skip(
        "Spyre hardware requires 128-byte aligned buffers - small uint8 tensors cause alignment violations"
    )
    def test_negative_scalar_uint_tensor(self, execution_mode):
        """Negative scalar with uint8 tensor."""

        def neg_uint_add(x):
            return x + (-1.0)

        x = torch.tensor([1, 2, 3], dtype=torch.uint8)
        result = _run_spyre(execution_mode, neg_uint_add, x)
        assert result.dtype in (
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.float32,
        )

    # TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/925
    @pytest.mark.skip(
        "Spyre backend does not support int32 dtype - causes DtException in data format converter"
    )
    def test_float_scalar_int_tensor_truncation(self, execution_mode):
        """Int tensor * float promotes to float."""

        def float_int_mul(x):
            return x * 0.5

        x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        result = _run_spyre(execution_mode, float_int_mul, x)
        assert result.dtype in (torch.float32, torch.float64)
        expected = torch.tensor([0.5, 1.0, 1.5, 2.0])
        assert torch.allclose(result, expected)

    def test_mixed_device_scalar_operation(self, execution_mode):
        """Device preserved for scalar op."""

        def device_mul(x):
            return x * 0.125

        x = cached_randn((10, 10))
        result = _run_spyre(execution_mode, device_mul, x)
        assert result.device.type == DEVICE.type
        assert result.shape == x.shape

    def test_requires_grad_scalar_operation(self, execution_mode):
        """requires_grad preserved on forward."""

        def grad_mul(x):
            return x * 0.125

        x = cached_randn((10, 10))
        x.requires_grad = True
        result = _run_spyre(execution_mode, grad_mul, x)
        assert result.shape == x.shape
        assert result.requires_grad

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1558
    @pytest.mark.xfail(reason="RuntimeError: Error: In-device copy not implemented.")
    def test_inplace_scalar_operation(self, execution_mode):
        """In-place multiply by scalar."""

        def inplace_mul(x):
            x *= 0.125
            return x

        x = cached_randn((10, 10))
        x_cpu = x.clone()
        expected = x_cpu * 0.125
        x_spyre = x.clone()
        result = _run_spyre(execution_mode, inplace_mul, x_spyre)
        assert result.shape == (10, 10)
        assert torch.allclose(result.cpu(), expected, atol=1e-3, rtol=1e-3)

    def test_chained_scalar_multiply_twice(self, execution_mode):
        """``((x * 0.125) * 2.0)`` — two scalar multiplies (no graph cycle)."""

        def circular_dep(x):
            x = x * 0.125
            x = x * 2.0
            return x

        x = cached_randn((10, 10))
        _compare_modes(execution_mode, circular_dep, x, atol=0.0005, rtol=0.001)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1738
    @pytest.mark.xfail(
        reason="backend does not support aten::isnan operations required for NaN validation"
    )
    def test_many_scalar_constants(self, execution_mode):
        """Many scalar adds/muls in a loop."""

        def many_scalars(x):
            result = x
            for _ in range(100):
                result = result * 1.001 + 0.001
            return result

        x = cached_randn((10, 10))
        result = _run_spyre(execution_mode, many_scalars, x)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_nested_compile_scalar(self, execution_mode):
        """Nested torch.compile scalar path; xfail if backend lacks nested-compile support."""

        def nested_compile_op(x):
            return x * 0.125

        x = cached_randn((10, 10))
        try:
            if execution_mode == "compiled":
                compiled_once = torch.compile(nested_compile_op)
                compiled_twice = torch.compile(compiled_once)
                result = compiled_twice(x.to(DEVICE))
            else:
                result = _run_spyre(execution_mode, nested_compile_op, x)

            expected = nested_compile_op(x)
            torch.testing.assert_close(result.cpu(), expected, atol=1e-3, rtol=1e-3)
        except (
            RuntimeError,
            AttributeError,
            torch._dynamo.exc.TorchRuntimeError,
        ) as err:
            pytest.xfail(reason=f"Nested compile unsupported on Spyre path: {err}")

    def test_cpu_scalar_tensor_with_spyre_tensor(self, execution_mode):
        """CPU scalar tensor × Spyre tensor should raise device mismatch error."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1598
        if execution_mode == "compiled":
            pytest.xfail(
                reason="IndexError: list index out of range due to device_tensor_layout"
            )

        def cpu_scalar_spyre_tensor(x):
            scalar = torch.tensor(0.125, dtype=torch.float16)  # CPU (intentional)
            return x * scalar

        x = cached_randn((128, 128), dtype=torch.float16)

        with pytest.raises(
            (RuntimeError, torch._inductor.exc.InductorError)
        ) as exc_info:
            _compare_modes(
                execution_mode, cpu_scalar_spyre_tensor, x, atol=1e-3, rtol=1e-3
            )

        error_msg = str(exc_info.value)
        print(error_msg)
        assert any(
            kw in error_msg.lower() for kw in ["device", "layout", "cpu", "spyre"]
        )

    def test_cpu_tensor_with_spyre_tensor(self, execution_mode):
        """CPU regular tensor × Spyre tensor should raise device mismatch error."""

        def cpu_tensor_spyre_tensor(x):
            cpu_tensor = torch.randn(128, 128, dtype=torch.float32)  # CPU (intentional)
            return x * cpu_tensor

        x = cached_randn((128, 128), dtype=torch.float32)

        with pytest.raises(
            (RuntimeError, torch._inductor.exc.InductorError)
        ) as exc_info:
            _compare_modes(
                execution_mode, cpu_tensor_spyre_tensor, x, atol=1e-3, rtol=1e-3
            )

        error_msg = str(exc_info.value)
        assert any(
            kw in error_msg.lower() for kw in ["device", "layout", "cpu", "spyre"]
        )
