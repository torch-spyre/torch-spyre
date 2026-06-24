# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Apache License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from utils_inductor import cached_randn, compare_with_cpu

SPYRE = torch.device("spyre")


def _compare_mode(
    execution_mode: str,
    fn,
    *args,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> None:
    """Run compare_with_cpu for exactly one Spyre path: eager XOR compiled."""
    compiled = execution_mode == "compiled"
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=compiled,
        run_eager=not compiled,
    )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestTransposeEdge:
    # --- Shape / invalid (parity where defined) ---

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((0, 64), id="zero_rows"),
            pytest.param((64, 0), id="zero_cols"),
        ],
    )
    def test_transpose_zero_extent(self, execution_mode, shape):
        """Transpose when one axis has size 0."""
        x = torch.empty(shape, dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)

    # --- Memory / aspect ---
    def test_transpose_very_large_tensor(self, execution_mode):
        try:
            x = torch.randn((16384, 16384), dtype=torch.float16)
            _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for this test")
            raise

    def test_transpose_extreme_aspect_ratio(self, execution_mode):
        x = cached_randn((1, 10000), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)

        y = cached_randn((10000, 1), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), y)

    def test_transpose_extreme_aspect_via_compare(self, execution_mode):
        """Very wide matrix -> transpose (distinct cache key vs other aspect tests)."""
        x = cached_randn((1, 4096), dtype=torch.float16, differentiation=2)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)

    # --- Dtypes ---

    @pytest.mark.xfail(reason="Issue #1545: Complex dtype not supported on Spyre")
    def test_transpose_complex_dtype(self, execution_mode):
        x = torch.randn((64, 128), dtype=torch.complex64)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)

    def test_transpose_bool_dtype(self, execution_mode):
        x = torch.randint(0, 2, (64, 128), dtype=torch.bool)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)

    def test_transpose_bfloat16_if_supported(self, execution_mode):
        x = torch.randn((48, 72), dtype=torch.bfloat16)
        _compare_mode(
            execution_mode, lambda t: t.transpose(0, 1), x, atol=1e-2, rtol=1e-2
        )

    # --- Non-contiguous / stride (parity on composed transpose graph) ---
    def test_transpose_already_non_contiguous(self, execution_mode):
        x = cached_randn((64, 128, 256), dtype=torch.float16)

        def fn(t):
            u = t.transpose(0, 1)
            return u.transpose(1, 2)

        _compare_mode(execution_mode, fn, x)

    @pytest.mark.skip(
        "Issue #1840: Signal Received: 11 (Segmentation fault) for eager and dxp_standalone for compile"
    )
    def test_transpose_strided_tensor(self, execution_mode):
        x = cached_randn((128, 256), dtype=torch.float16)

        def fn(t):
            return t[::2, ::2].transpose(0, 1)

        _compare_mode(execution_mode, fn, x)

    # --- Dimension semantics ---
    def test_transpose_1d_tensor(self, execution_mode):
        x = cached_randn((64,), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 0), x)

        with pytest.raises((IndexError, RuntimeError)):
            x.transpose(0, 1)

    def test_transpose_5d_tensor(self, execution_mode):
        x = cached_randn((2, 4, 8, 16, 32), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(1, 4), x)

        with pytest.raises((IndexError, RuntimeError)):
            x.transpose(1, 5)

    # --- Special values ---
    def test_transpose_nan_inf_preserving(self, execution_mode):
        """NaN/Inf slots preserved through transpose (utils assert_close uses equal_nan)."""
        # clone() is required: cached_randn is lru_cache'd, so mutating its return value
        # would corrupt the cached tensor for all subsequent tests sharing this cache key.
        x = cached_randn((64, 128), dtype=torch.float16).clone()
        x[0, 0] = float("nan")
        x[1, 1] = float("inf")
        x[2, 2] = float("-inf")
        _compare_mode(
            execution_mode, lambda t: t.transpose(0, 1), x, atol=2e-3, rtol=1e-3
        )

    def test_transpose_with_zero(self, execution_mode):
        x = torch.zeros((64, 128), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), x)

    # --- Boundaries ---
    # (1,1) 2D flip is already covered by test_t_2d* in test_inductor_ops_latest.py.

    def test_transpose_maximum_dimensions(self, execution_mode):
        x = cached_randn((2, 2, 2, 2, 2, 2), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 5), x)

    def test_transpose_stick_boundary_fp32_height_one(self, execution_mode):
        """Tall×1 fp32 flip; (64,1) fp16 is already in test_t_2d* in ops_latest."""
        y = cached_randn((16, 1), dtype=torch.float32)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1), y)

    # --- Indexed views (skipped: #1800) ---
    @pytest.mark.skip(
        reason="Issue #1800: Memory corruption in indexed views of transpose tensors"
    )
    def test_indexing_after_transpose(self, execution_mode):
        del execution_mode
        x = cached_randn((2, 4, 8, 16)).to(SPYRE)
        y = x.transpose(1, 3)

        slice1 = y[0, :, :, 0]
        assert slice1.shape == (16, 8)

        x_cpu = x.cpu()
        y_cpu = x_cpu.transpose(1, 3)
        assert torch.allclose(slice1.cpu(), y_cpu[0, :, :, 0])

    @pytest.mark.skip(
        reason="Issue #1800: Memory corruption in indexed views of transpose tensors"
    )
    def test_slicing_after_transpose(self, execution_mode):
        del execution_mode
        x = cached_randn((4, 8, 16, 32)).to(SPYRE)
        y = x.transpose(1, 3)

        slice1 = y[:2, :16, :8, :4]
        assert slice1.shape == (2, 16, 8, 4)

        x_cpu = x.cpu()
        y_cpu = x_cpu.transpose(1, 3)
        assert torch.allclose(slice1.cpu(), y_cpu[:2, :16, :8, :4])

    # --- Materialized copy after transpose (#1859) ---
    @pytest.mark.xfail(
        reason="Issue #1859: dxp_standalone SIGABRT on transpose+clone bundle generation.",
    )
    def test_transpose_then_clone(self, execution_mode):
        x = cached_randn((72, 91), dtype=torch.float16)
        _compare_mode(execution_mode, lambda t: t.transpose(0, 1).clone(), x)


# Tests below do not vary by execution_mode — moving them out of the
# parametrized TestTransposeEdge class avoids running each test twice
# (eager + compiled) with no added coverage.
@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestTransposeDeviceSemantics:
    """Device-level transpose semantics: error handling, view invariants, compilation."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    # --- Error / invalid ---
    def test_transpose_scalar_tensor(self):
        x = torch.tensor(3.14, dtype=torch.float16)
        with pytest.raises((IndexError, RuntimeError)):
            x.transpose(0, 1)

    # --- Stride parity CPU vs Spyre (.is_contiguous() flag) ---
    def test_transpose_contiguous_flag_matches_cpu(self):
        x_cpu = cached_randn((2, 4, 8, 16))
        x_sp = x_cpu.to(SPYRE)
        for d0, d1 in ((1, 3), (2, 3)):
            assert (
                x_sp.transpose(d0, d1).is_contiguous()
                == x_cpu.transpose(d0, d1).is_contiguous()
            )

    # --- View semantics (Spyre tensors) ---
    def test_transpose_creates_view(self):
        x = cached_randn((2, 4, 8, 16)).to(SPYRE)
        y = x.transpose(1, 3)
        assert y.data_ptr() == x.data_ptr()

    def test_shared_storage(self):
        x = cached_randn((2, 4, 8, 16)).to(SPYRE)
        y = x.transpose(1, 3)
        assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

    def test_view_chain(self):
        x = cached_randn((2, 4, 8, 16)).to(SPYRE)
        y = x.transpose(1, 3)
        z = y.transpose(0, 2)
        assert x.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()

    # --- Compilation: correctness across multiple compiled functions ---
    def test_multiple_compiles_transpose(self):
        @torch.compile
        def fn1(x):
            return x.transpose(1, 3)

        @torch.compile
        def fn2(x):
            return x.transpose(2, 3)

        x = cached_randn((2, 4, 8, 16)).to(SPYRE)
        y1 = fn1(x)
        y2 = fn2(x)

        x_ref = cached_randn((2, 4, 8, 16))
        assert y1.shape == (2, 16, 8, 4)
        assert y2.shape == (2, 4, 16, 8)
        assert torch.allclose(y1.cpu(), x_ref.transpose(1, 3), atol=1e-3, rtol=1e-3)
        assert torch.allclose(y2.cpu(), x_ref.transpose(2, 3), atol=1e-3, rtol=1e-3)

    # --- Compilation: cached compilation reuses the same graph for different inputs ---
    def test_compilation_cache_transpose(self):
        @torch.compile
        def fn(x):
            return x.transpose(1, 3)

        x1 = cached_randn((2, 4, 8, 16)).to(SPYRE)
        x2 = cached_randn((2, 4, 8, 16), differentiation=1).to(SPYRE)

        y1 = fn(x1)
        y2 = fn(x2)

        x1_ref = cached_randn((2, 4, 8, 16))
        x2_ref = cached_randn((2, 4, 8, 16), differentiation=1)
        assert y1.shape == y2.shape
        assert torch.allclose(y1.cpu(), x1_ref.transpose(1, 3), atol=1e-3, rtol=1e-3)
        assert torch.allclose(y2.cpu(), x2_ref.transpose(1, 3), atol=1e-3, rtol=1e-3)

    # --- Compiled transpose fused with pointwise ops ---
    def test_transpose_mul_add_compiled_matches_eager(self):
        def fn(x):
            y = x.transpose(1, 3).contiguous()
            return y * 2.0 + 1.0

        x_ref = cached_randn((2, 4, 8, 16))
        x = x_ref.to(SPYRE)
        out = torch.compile(fn)(x)
        ref = (x_ref.transpose(1, 3) * 2.0) + 1.0
        assert torch.allclose(out.cpu(), ref, rtol=1e-2, atol=5e-3)

    @pytest.mark.xfail(
        reason="Issue #2006: SpyreKernel.store() rejects Constant scalar in pointwise mul.",
    )
    def test_compile_with_operations_after_transpose(self):
        @torch.compile
        def fn(x):
            y = x.transpose(1, 3)
            z = y * 2.0
            return z + 1.0

        x = cached_randn((2, 4, 8, 16)).to(SPYRE)
        y_out = fn(x)

        expected = (x.transpose(1, 3) * 2.0) + 1.0
        assert torch.allclose(y_out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
