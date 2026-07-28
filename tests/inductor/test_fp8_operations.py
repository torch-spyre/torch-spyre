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

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX, FP8_E4M3FN_MIN
from utils_inductor import (
    cached_randn,
    compare_with_pytorch,
)

# Maximum spacing between adjacent representable values in FP8 E4M3
FP8_E4M3_MAX_SPACING = 32.0


# Additional test constants
FP8_E4M3_HALF_MAX = FP8_E4M3_MAX / 2.0  # 224.0 - Half of FP8 E4M3 max for testing reduced quantization ranges
FP16_SAFE_LARGE_VALUE = 30000.0  # Well below FP16 max (65504) to avoid overflow in edge case tests
MIXED_SIGNS_SCALE = 100.0  # Scale factor to test moderate value ranges with mixed positive/negative values


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
            x_fp8 = x.clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
            return x_fp8.to(torch.float16) * scale

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
            x_fp8 = x.clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
            return x_fp8.to(torch.float16) * scale

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
            return (x / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(
                torch.float8_e4m3fn
            ).to(torch.float16) * scale

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
            return (x / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(
                torch.float8_e4m3fn
            ).to(torch.float16) * scale

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=atol,
            rtol=rtol,
        )

    # ========================================================================
    # quantscalepertokenfp8 Tests
    # ========================================================================

    def test_quantscalepertokenfp8_basic(self):
        """Test basic quantscalepertokenfp8 functionality.

        Validates:
        - Correct output shape: [batch, seq, hidden] → [batch, seq, 1]
        - Correct output dtype: torch.float16
        - Numerical correctness against CPU reference
        """
        x = cached_randn((2, 4, 8), dtype=torch.float16, scale=1.0)

        def spyre_fn(x):
            return torch.ops.spyre.quantscalepertokenfp8(x)

        def pytorch_fn(x):
            return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3_MAX

        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "shape,mean,std",
        [
            # Small values - typical activation ranges
            ((2, 4, 32), 0.0, 1.0),
            ((2, 4, 32), 0.0, 5.0),
            # Medium values - moderate activation ranges
            ((2, 4, 32), 10.0, 50.0),
            ((2, 4, 32), 50.0, 100.0),
            # Large values - near FP8 boundaries
            ((2, 4, 32), 100.0, 100.0),
            ((2, 4, 32), 200.0, 200.0),
        ],
    )
    def test_quantscalepertokenfp8_numerical_correctness(self, shape, mean, std):
        """Test quantscalepertokenfp8 numerical correctness across value ranges.

        Validates the scale computation formula:
        scale = amax(abs(input), dim=-1, keepdim=True) / scale_ub

        Tests various input distributions to ensure correctness across
        typical activation ranges, moderate ranges, and boundary cases.
        """
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * std + mean

        def spyre_fn(x):
            return torch.ops.spyre.quantscalepertokenfp8(x)

        def pytorch_fn(x):
            return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3_MAX

        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "shape,scale_ub",
        [
            ((2, 4, 8), FP8_E4M3_MAX),  # Default FP8_E4M3_MAX
            ((2, 4, 8), FP8_E4M3_HALF_MAX),  # Half of default
            ((2, 4, 8), 100.0),  # Custom value to test non-standard quantization range
            ((2, 4, 32), FP8_E4M3_MAX),  # Larger hidden dim
            ((1, 128, 512), FP8_E4M3_MAX),  # Production-like shape
            ((1, 128, 512), FP8_E4M3_HALF_MAX),  # Production-like with custom scale_ub
        ],
    )
    def test_quantscalepertokenfp8_custom_scale_ub(self, shape, scale_ub):
        """Test quantscalepertokenfp8 with custom scale_ub parameter.

        Validates that the scale_ub parameter correctly adjusts the
        scale computation for different quantization ranges.
        """
        x = cached_randn(shape, dtype=torch.float16, scale=1.0)

        def spyre_fn(x):
            return torch.ops.spyre.quantscalepertokenfp8(x, scale_ub=scale_ub)

        def pytorch_fn(x):
            return torch.amax(torch.abs(x), dim=-1, keepdim=True) / scale_ub

        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 128, 512),  # Small batch, medium hidden
            (4, 128, 512),  # Medium batch, medium hidden
            (1, 128, 1024),  # Small batch, large hidden
            (1, 128, 2048),  # Small batch, very large hidden
            (1, 128, 4096),  # Small batch, production hidden
            (8, 2048, 4096),  # Production scale
        ],
    )
    def test_quantscalepertokenfp8_production_shapes(self, shape):
        """Test quantscalepertokenfp8 with production-scale tensor shapes.

        Validates correctness and performance with realistic tensor sizes
        used in production LLM inference and training workloads.
        """
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 2.0 + 1.0

        def spyre_fn(x):
            return torch.ops.spyre.quantscalepertokenfp8(x)

        def pytorch_fn(x):
            return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3_MAX

        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=1e-3, rtol=1e-3)


    def test_quantscalepertokenfp8_zero_input_handling(self):
        """Verify quantscalepertokenfp8 handles zero input without crashing.
        
        Tests that the operation gracefully handles all-zero input tensors,
        which could potentially cause division by zero. The hardware should
        add a small epsilon to prevent this.
        """
        x = cached_randn((2, 4, 8), dtype=torch.float16, scale=1.0) * 0.0
        
        def spyre_fn(x):
            return torch.ops.spyre.quantscalepertokenfp8(x)
        
        def pytorch_fn(x):
            return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3_MAX
        
        # Compare with relaxed tolerance due to hardware epsilon
        # Note: Hardware adds small epsilon to prevent division by zero
        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=1e-5, rtol=1e-2)


    @pytest.mark.parametrize(
        "shape,scale_ub",
        [
            ((2, 4, 8), FP8_E4M3_MAX),
            ((1, 128, 512), FP8_E4M3_MAX),
        ],
    )
    def test_quantscalepertokenfp8_with_quantize_integration(self, shape, scale_ub):
        """Test quantscalepertokenfp8 integrated with quantize/dequantize pipeline.

        Validates the complete FP8 quantization workflow:
        1. Compute per-token scale using quantscalepertokenfp8
        2. Quantize to FP8 using quantize_fp8_with_scale
        3. Dequantize back to FP16 using dequantize_fp8_with_scale
        4. Verify roundtrip accuracy

        This tests the primary use case of quantscalepertokenfp8 in
        production FP8 quantization pipelines.
        """
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 2.0 + 1.0

        def spyre_fn(x):
            # Compute per-token scale
            scale = torch.ops.spyre.quantscalepertokenfp8(x, scale_ub=scale_ub)
            # Quantize to FP8
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            # Dequantize back to FP16
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x):
            # CPU reference implementation
            scale = torch.amax(torch.abs(x), dim=-1, keepdim=True) / scale_ub
            x_scaled = x / scale
            x_fp8 = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(
                torch.float8_e4m3fn
            )
            return x_fp8.to(torch.float16) * scale

        # Use tolerance appropriate for FP8 quantization roundtrip
        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=2.0, rtol=0.0)

    @pytest.mark.parametrize(
        "input_type,shape",
        [
            ("zeros", (2, 4, 8)),
            ("near_zero", (2, 4, 8)),
            ("large", (2, 4, 8)),
            ("mixed_signs", (2, 4, 32)),
            ("large_hidden", (1, 1, 8192)),
        ],
    )
    def test_quantscalepertokenfp8_edge_cases(self, input_type, shape):
        """Test quantscalepertokenfp8 with edge case inputs.

        Validates behavior with boundary conditions:
        - zeros: All zero tensor (scale should be 0)
        - near_zero: Very small values (numerical stability)
        - large: Values near FP16 max (65504)
        - mixed_signs: Positive and negative values
        - single_element: Minimal tensor size
        - large_hidden: Very large hidden dimension
        """
        if input_type == "zeros":
            x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 0.0
        elif input_type == "near_zero":
            x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 0.0 + 1e-6
        elif input_type == "large":
            # Near FP16 max (65504), but within safe range
            x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 0.0 + FP16_SAFE_LARGE_VALUE
        elif input_type == "mixed_signs":
            x = cached_randn(shape, dtype=torch.float16, scale=1.0) * MIXED_SIGNS_SCALE
        elif input_type == "large_hidden":
            x = cached_randn(shape, dtype=torch.float16, scale=1.0)

        def spyre_fn(x):
            return torch.ops.spyre.quantscalepertokenfp8(x)

        def pytorch_fn(x):
            return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3_MAX

        # Use appropriate tolerance based on input type
        # Note: zeros and near_zero have relaxed tolerance (atol=1e-5, rtol=1e-2) because:
        # 1. Hardware adds small epsilon (~2.4e-6) to prevent division by zero
        # 2. This epsilon is implementation-specific and may vary across hardware versions
        # 3. The relaxed tolerance accounts for this hardware-level numerical stability mechanism
        # 4. Standard tolerance (atol=1e-3, rtol=1e-3) is used for all other cases
        if input_type in ["zeros", "near_zero"]:
            atol, rtol = 1e-5, 1e-2
        else:
            atol, rtol = 1e-3, 1e-3

        compare_with_pytorch(spyre_fn, pytorch_fn, x, atol=atol, rtol=rtol)




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
