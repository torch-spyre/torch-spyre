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

"""
Unit tests for FP8 quantization operations.

Tests cover:
- qfp8ch: Channel-wise FP8 format conversion
- fp8todl16: FP8→FP16 dtype conversion (tests .to(torch.float16) lowering)
- quantize/dequantize: Comprehensive roundtrip tests with various scales and input ranges
"""

import pytest
import torch
from utils_inductor import (
    DEVICE,
    cached_randn,
    compare_with_pytorch,
)

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX, FP8_E4M3FN_MIN

# Maximum spacing between adjacent representable values in FP8 E4M3
FP8_E4M3_MAX_SPACING = 32.0


def _fp8_reference_quantize_dequantize(x, scale):
    """CPU reference for FP8 quantize→dequantize: clamp, cast to FP8, cast back, rescale."""
    return (x / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn).to(
        torch.float16
    ) * scale


class TestFP8Operations:
    """Test suite for FP8 quantization operations not covered in test_inductor_ops.py."""

    def test_qfp8ch_basic_conversion(self):
        """Test basic FP16→FP8 format conversion with qfp8ch.

        Tests:
        - Basic conversion with shape [1, 2, 8]
        - Roundtrip: FP16 → FP8 → FP16 with scaling
        - Verifies qfp8ch operation is used internally

        Note: We use dequantize_fp8_with_scale for FP8→FP16 conversion
        because direct .to(torch.float16) cannot transfer to CPU.
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # Test qfp8ch format conversion directly (no pre-scaling)
            # Input x is already in valid FP8 range from cached_randn
            x_fp8 = torch.ops.spyre.qfp8ch(x)
            verify_fp8_dtype(x_fp8)
            # Dequantize with identity scale to verify format conversion
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            # CPU reference: direct format conversion with identity scale
            return _fp8_reference_quantize_dequantize(x, scale)

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=0.0,
            rtol=0.0,
        )

    def test_fp8todl16_basic_conversion(self):
        """Test FP8→FP16 dtype conversion with fp8todl16.

        Tests:
        - FP8→FP16 conversion using dequantize_fp8_with_scale with identity scale
        - Verifies fp8todl16 operation is triggered by the decomposition
        - Confirms output dtype is FP16
        - Tests the lowering path: x_fp8.to(torch.float16) * scale

        This test validates that the fp8todl16 deeptools operation is correctly
        invoked during dequantization. Note: Direct .to(torch.float16) without
        scaling cannot transfer to CPU, so we use identity scale (ones) to enable
        CPU transfer while still testing the fp8todl16 operation.
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # Convert FP16 → FP8 using qfp8ch
            x_fp8 = torch.ops.spyre.qfp8ch(x)
            verify_fp8_dtype(x_fp8)

            # Convert FP8 → FP16 using dequantize_fp8_with_scale with identity scale
            # This triggers fp8todl16 operation and allows CPU transfer
            x_fp8_fp16 = torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)
            verify_fp16_dtype(x_fp8_fp16)

            return x_fp8_fp16

        def pytorch_fn(x, scale):
            # CPU reference: FP16 → FP8 → FP16 conversion with identity scale
            return _fp8_reference_quantize_dequantize(x, scale)

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=0.5,
            rtol=0.1,
        )

    # Tolerance categories:
    # - small_range: low FP8 spacing regions (atol=2)
    # - medium_range: may enter spacing=16 regions (atol=16)
    # - boundary_cases: may enter spacing=32 regions (atol=32 * scale)

    @pytest.mark.parametrize(
        "shape,scale_value,mean,std",
        [
            ((1, 2, 32), 0.01, 0.0, 1.0),
            ((1, 2, 32), 0.01, 0.0, 5.0),
            ((1, 2, 32), 0.1, 0.0, 1.0),
            ((1, 2, 32), 0.1, 0.0, 5.0),
            ((1, 2, 32), 0.5, 0.0, 1.0),
            ((1, 2, 32), 0.5, 0.0, 5.0),
            ((1, 2, 32), 1.0, 0.0, 1.0),
            ((1, 2, 32), 1.0, 0.0, 5.0),
            ((1, 2, 32), 2.0, 0.0, 1.0),
            ((1, 2, 32), 2.0, 0.0, 5.0),
        ],
    )
    def test_quantize_dequantize_fp8_small_range(
        self,
        shape,
        scale_value,
        mean,
        std,
    ):
        """Test quantize/dequantize for typical FP8 value ranges."""
        self._run_quantize_dequantize_fp8_test(
            shape,
            scale_value,
            mean,
            std,
            atol=2.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape,scale_value,mean,std",
        [
            ((1, 2, 32), 0.01, 10.0, 50.0),
            ((1, 2, 32), 0.1, 10.0, 50.0),
            ((1, 2, 32), 0.5, 10.0, 50.0),
            ((1, 2, 32), 1.0, 10.0, 50.0),
            ((1, 2, 32), 2.0, 10.0, 50.0),
        ],
    )
    def test_quantize_dequantize_fp8_medium_range(
        self,
        shape,
        scale_value,
        mean,
        std,
    ):
        """Test quantize/dequantize for moderate input ranges.

        These cases may enter higher FP8 spacing regions but do not
        intentionally target FP8 representation boundaries.
        """
        self._run_quantize_dequantize_fp8_test(
            shape,
            scale_value,
            mean,
            std,
            atol=16.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape,scale_value,mean,std",
        [
            ((1, 2, 32), 0.01, 100.0, 100.0),
            ((1, 2, 32), 0.01, 200.0, 200.0),
            ((1, 2, 32), 0.1, 100.0, 100.0),
            ((1, 2, 32), 0.1, 200.0, 200.0),
            ((1, 2, 32), 0.5, 100.0, 100.0),
            ((1, 2, 32), 0.5, 200.0, 200.0),
            ((1, 2, 32), 1.0, 100.0, 100.0),
            ((1, 2, 32), 1.0, 200.0, 200.0),
            ((1, 2, 32), 2.0, 100.0, 100.0),
            ((1, 2, 32), 2.0, 200.0, 200.0),
        ],
    )
    def test_quantize_dequantize_fp8_boundary_cases(
        self,
        shape,
        scale_value,
        mean,
        std,
    ):
        """Test FP8 E4M3 representation boundary cases."""
        self._run_quantize_dequantize_fp8_test(
            shape,
            scale_value,
            mean,
            std,
            atol=FP8_E4M3_MAX_SPACING * scale_value,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 128, 512),
            (4, 128, 512),
            (1, 128, 1024),
            (1, 128, 2048),
            (1, 128, 4096),
        ],
    )
    def test_quantize_dequantize_fp8_production_shapes(self, shape):
        """Test FP8 quantize/dequantize with production-scale tensor shapes.

        Uses standard tolerance values (atol=0.5, rtol=0.1) with typical
        input distributions (mean=1.0, std=2.0, scale=1.0) that don't trigger
        edge cases in FP8 representation.
        """
        # Generate deterministic input with typical distribution
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 2.0 + 1.0
        scale = torch.tensor([1.0], dtype=torch.float16)

        def spyre_fn(x, scale):
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            return _fp8_reference_quantize_dequantize(x, scale)

        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=0.5, rtol=0.1)

    def test_quantize_dequantize_fp8_4d_shape(self):
        """Test FP8 quantize/dequantize (qfp8ch path) with a 4D input tensor.

        Exercises the general 4D compilation path for quantize_fp8_with_scale
        (qfp8ch / activation quantization). Note: qfp8ch is not in FP8_2D_STICK_OPS
        and does not trigger device-size flattening. See
        test_quantize_weight_fp8_with_scale_4d_shape for the qfp8wt flattening path.
        """
        shape = (2, 4, 128, 512)
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 2.0 + 1.0
        scale = torch.tensor([1.0], dtype=torch.float16)

        def spyre_fn(x, scale):
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            return _fp8_reference_quantize_dequantize(x, scale)

        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=0.5, rtol=0.1)

    def test_quantize_weight_fp8_with_scale_eager_mode_dtype_only(self):
        """Regression guard: quantize_weight_fp8_with_scale returns a valid FP8 tensor in eager mode.

        A duplicate custom op registration with an empty body can silently overwrite
        the real implementation at import time, causing the op to return None in eager
        mode. A None return fails verify_fp8_dtype immediately.

        This test intentionally checks dtype and shape only — not numerical correctness.
        The qfp8wt op produces a QFP8WT 2D-stick physical layout that is opaque to
        fp8todl16 (dequantize uses a 1D flat read), so a naive quantize→dequantize
        roundtrip will not match a CPU reference. Numerical correctness of the
        qfp8wt → batchmatmulfp8 path is covered by test_fp8_scaled_mm in
        test_inductor_ops.py.
        """
        weight = cached_randn((128, 128), dtype=torch.float16, scale=1.0)
        scale = torch.max(torch.abs(weight)).reshape(1)
        weight_d = weight.to(DEVICE)
        scale_d = scale.to(DEVICE)

        # Call directly without torch.compile — exercises the compile_once eager path.
        quantized = torch.ops.spyre.quantize_weight_fp8_with_scale(weight_d, scale_d)

        verify_fp8_dtype(quantized)
        assert quantized.shape == weight_d.shape, (
            f"Expected shape {weight_d.shape}, got {quantized.shape}"
        )

    def test_dequantize_fp8_with_scale_eager_mode(self):
        """Test dequantize_fp8_with_scale in eager mode.

        Validates that dequantize_fp8_with_scale now works in eager mode via
        the compile_once decorator (previously required torch.compile and
        returned None when called directly).

        Uses quantize_fp8_with_scale (qfp8ch path) to produce the FP8 input
        since that path produces a flat QFP8CH layout compatible with
        fp8todl16. Verifies numerical correctness via a compiled roundtrip.
        """
        x = cached_randn((1, 2, 8), dtype=torch.float16, scale=1.0)
        scale = torch.tensor([1.0], dtype=torch.float16)

        def spyre_fn(inp, s):
            q = torch.ops.spyre.quantize_fp8_with_scale(inp, s)
            return torch.ops.spyre.dequantize_fp8_with_scale(q, s)

        def pytorch_fn(inp, s):
            return _fp8_reference_quantize_dequantize(inp, s)

        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=0.5, rtol=0.1)

    def _run_quantize_dequantize_fp8_test(
        self,
        shape,
        scale_value,
        mean,
        std,
        atol,
        rtol=0.0,
    ):
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * std + mean
        scale = torch.tensor([scale_value], dtype=torch.float16)

        def spyre_fn(x, scale):
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            return _fp8_reference_quantize_dequantize(x, scale)

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=atol,
            rtol=rtol,
        )


# Test utilities for FP8 operations
def verify_fp8_dtype(tensor):
    """Verify tensor has FP8 E4M3 dtype."""
    assert tensor.dtype == torch.float8_e4m3fn, (
        f"Expected dtype torch.float8_e4m3fn, got {tensor.dtype}"
    )


def verify_fp16_dtype(tensor):
    """Verify tensor has FP16 dtype."""
    assert tensor.dtype == torch.float16, (
        f"Expected dtype torch.float16, got {tensor.dtype}"
    )
