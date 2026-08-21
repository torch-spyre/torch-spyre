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
Test bidirectional FP16↔FP32 type conversion with ElementArrangement.

Tests all 4 conversion cases:
1. FP16→FP32 with STANDARD → DL16_TO_FP32
2. FP16→FP32 with FP32_TO_DL16 → STANDARD
3. FP32→FP16 with STANDARD → FP32_TO_DL16
4. FP32→FP16 with DL16_TO_FP32 → STANDARD
"""

import pytest
import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("src_dtype", [torch.float16, torch.bfloat16])
def test_fp16_to_fp32_standard_input(device, src_dtype):
    """Test FP16/BF16→FP32 with STANDARD input creates DL16_TO_FP32 (#2843)."""

    @torch.compile
    def fn(x):
        return x.to(torch.float32)

    x = torch.randn(4, 128, device=device, dtype=src_dtype)
    result = fn(x)

    # Verify output EA
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32, (
        f"Expected DL16_TO_FP32, got {result_layout.element_arrangement}"
    )

    # Note: Cannot compare tensors with non-STANDARD EA directly with CPU
    # The result has DL16_TO_FP32 EA which differs from CPU's STANDARD EA

    print(f"✓ {src_dtype}→FP32 with STANDARD input produces DL16_TO_FP32")


@pytest.mark.parametrize("device", ["spyre"])
def test_mixed_bf16_fp32_add_rejects_ea_mismatch(device):
    """A bf16/fp32 add must raise on EA mismatch instead of silently executing (#2843)."""

    @torch.compile
    def fn(x, y):
        return torch.add(x, y)

    x_fp32 = torch.randn(5120, dtype=torch.float32, device=device)
    y_bf16 = torch.randn(5120, dtype=torch.bfloat16, device=device)

    with pytest.raises(Exception, match="element arrangement|EA"):
        fn(x_fp32, y_bf16)


@pytest.mark.parametrize("device", ["spyre"])
def test_fp32_to_fp16_standard_input(device):
    """Test FP32→FP16 with STANDARD input creates FP32_TO_DL16."""

    @torch.compile
    def fn(x):
        return x.to(torch.float16)

    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    result = fn(x)

    # Verify output EA
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.FP32_TO_DL16, (
        f"Expected FP32_TO_DL16, got {result_layout.element_arrangement}"
    )

    # Note: Cannot compare tensors with non-STANDARD EA directly with CPU
    # The result has FP32_TO_DL16 EA which differs from CPU's STANDARD EA

    print("✓ FP32→FP16 with STANDARD input produces FP32_TO_DL16")


@pytest.mark.parametrize("device", ["spyre"])
def test_fp16_to_fp32_restoration(device):
    """Test FP16→FP32 with FP32_TO_DL16 input restores to STANDARD."""

    @torch.compile
    def fn(x):
        # FP32 → FP16 (creates FP32_TO_DL16)
        x_fp16 = x.to(torch.float16)
        # FP16 → FP32 (should restore to STANDARD)
        return x_fp16.to(torch.float32)

    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    result = fn(x)

    # Verify output EA is STANDARD (restored)
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.STANDARD, (
        f"Expected STANDARD (restored), got {result_layout.element_arrangement}"
    )

    # Verify correctness
    x_cpu = x.cpu()
    result_cpu = fn(x_cpu)
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)

    print("✓ FP16→FP32 restoration (FP32_TO_DL16 → STANDARD) works")


@pytest.mark.parametrize("device", ["spyre"])
def test_fp32_to_fp16_restoration(device):
    """Test FP32→FP16 with DL16_TO_FP32 input restores to STANDARD."""

    @torch.compile
    def fn(x):
        # FP16 → FP32 (creates DL16_TO_FP32)
        x_fp32 = x.to(torch.float32)
        # FP32 → FP16 (should restore to STANDARD)
        return x_fp32.to(torch.float16)

    x = torch.randn(4, 128, device=device, dtype=torch.float16)
    result = fn(x)

    # Verify output EA is STANDARD (restored)
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.STANDARD, (
        f"Expected STANDARD (restored), got {result_layout.element_arrangement}"
    )

    # Verify correctness
    x_cpu = x.cpu()
    result_cpu = fn(x_cpu)
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)

    print("✓ FP32→FP16 restoration (DL16_TO_FP32 → STANDARD) works")


@pytest.mark.parametrize("device", ["spyre"])
def test_bidirectional_roundtrip_fp16_start(device):
    """Test FP16→FP32→FP16 roundtrip."""

    @torch.compile
    def fn(x):
        # FP16(STANDARD) → FP32(DL16_TO_FP32) → FP16(STANDARD)
        x_fp32 = x.to(torch.float32)
        return x_fp32.to(torch.float16)

    x = torch.randn(4, 128, device=device, dtype=torch.float16)
    result = fn(x)

    # Verify final EA is STANDARD
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.STANDARD, (
        f"Expected STANDARD after roundtrip, got {result_layout.element_arrangement}"
    )

    # Verify correctness
    x_cpu = x.cpu()
    result_cpu = fn(x_cpu)
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)

    print("✓ FP16→FP32→FP16 roundtrip works")


@pytest.mark.parametrize("device", ["spyre"])
def test_bidirectional_roundtrip_fp32_start(device):
    """Test FP32→FP16→FP32 roundtrip."""

    @torch.compile
    def fn(x):
        # FP32(STANDARD) → FP16(FP32_TO_DL16) → FP32(STANDARD)
        x_fp16 = x.to(torch.float16)
        return x_fp16.to(torch.float32)

    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    result = fn(x)

    # Verify final EA is STANDARD
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.STANDARD, (
        f"Expected STANDARD after roundtrip, got {result_layout.element_arrangement}"
    )

    # Verify correctness
    x_cpu = x.cpu()
    result_cpu = fn(x_cpu)
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)

    print("✓ FP32→FP16→FP32 roundtrip works")


def _stagger_fn(x):
    """fp32 → fp16(staggered) → stagger_to_standard_ea → standard EA fp16."""
    return torch.ops.spyre.stagger_to_standard_ea(x.to(torch.float16))


@pytest.mark.parametrize(
    "x",
    [
        # 1-D: stick-aligned
        torch.randn(64, dtype=torch.float32),
        torch.randn(128, dtype=torch.float32),
        # 1-D: non-stick-aligned (padded to 64-multiple)
        torch.nn.functional.pad(torch.randn(44, dtype=torch.float32), (0, 20)),
        # 2-D: stick-aligned
        torch.randn(4, 64, dtype=torch.float32),
        torch.randn(7, 128, dtype=torch.float32),
        # 2-D: non-stick-aligned (padded)
        torch.nn.functional.pad(torch.randn(7, 44, dtype=torch.float32), (0, 20)),
        # 3-D: stick-aligned
        torch.randn(2, 4, 64, dtype=torch.float32),
        torch.randn(3, 5, 128, dtype=torch.float32),
        # 3-D: non-stick-aligned (padded)
        torch.nn.functional.pad(torch.randn(2, 4, 44, dtype=torch.float32), (0, 20)),
        # 4-D: stick-aligned
        torch.randn(2, 3, 4, 64, dtype=torch.float32),
        torch.randn(2, 3, 4, 128, dtype=torch.float32),
        # 4-D: non-stick-aligned (padded)
        torch.nn.functional.pad(torch.randn(2, 3, 4, 44, dtype=torch.float32), (0, 20)),
    ],
)
@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
def test_stagger_to_standard_ea(x):
    """stagger_to_standard_ea restores standard EA after fp32→fp16 (fp32todl16).

    Verifies:
      1. Output values match a plain x.to(fp16) on CPU (logical correctness).
      2. Output EA is STANDARD (layout correctness).
    """
    expected = x.to(torch.float16)

    compiled_fn = torch.compile(_stagger_fn, backend="inductor")

    # 1. Value correctness: Spyre result matches CPU fp16 cast.
    # fp32→fp16 rounding differs slightly (Spyre uses DF16); use fp16 tolerances.
    result = compiled_fn(x.to("spyre")).cpu()
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    # 2. Layout correctness: output EA must be STANDARD.
    spyre_result = compiled_fn(x.to("spyre"))
    ea = get_spyre_tensor_layout(spyre_result).element_arrangement
    assert ea == ElementArrangement.STANDARD, f"Expected STANDARD EA, got {ea}"


# Made with Bob
