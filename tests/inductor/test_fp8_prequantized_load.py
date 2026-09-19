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

"""Tests for FP8 pre-quantized weight loading via load_fp8_model_to_spyre.

Validates that pre-quantized FP8 checkpoint weights (torch.float8_e4m3fn)
are loaded directly into QFP8WT KERNEL layout on Spyre — without lazy
decompression or runtime qfp8wt quantization.

Covers:
  1. _dma_to_spyre_fp8_kernel: transfers a synthetic FP8 tensor to Spyre
     with the correct dtype, shape, and device.
  2. load_fp8_model_to_spyre: loads FP8 Linear weights of an nn.Module
     into QFP8WT KERNEL layout; non-FP8 weights use the normal path.
  3. _scaled_mm with pre-quantized FP8 weight in KERNEL layout.
"""

import pytest
import torch
from torch import nn
from utils_inductor import DEVICE, compare_with_pytorch

import torch_spyre  # noqa: F401 — registers Spyre as the inductor backend

DEVICE_TYPE = DEVICE.type  # 'spyre' — used for device.type comparisons


# ---------------------------------------------------------------------------
# Unit tests — no model checkpoint required
# ---------------------------------------------------------------------------


class TestDmaToSpyreFp8Kernel:
    """Unit tests for _dma_to_spyre_fp8_kernel."""

    def test_basic_transfer(self):
        """FP8 weight transfers to Spyre with correct dtype and shape."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(128, 256, dtype=torch.float16).to(torch.float8_e4m3fn)
        dev = _dma_to_spyre_fp8_kernel(weight)

        assert dev.device.type == DEVICE_TYPE, (
            f"Expected device {DEVICE}, got {dev.device}"
        )
        assert dev.dtype == torch.float8_e4m3fn, f"Expected fp8, got {dev.dtype}"
        assert list(dev.shape) == [128, 256], f"Shape mismatch: {list(dev.shape)}"

    def test_non_contiguous_input(self):
        """Non-contiguous FP8 weight is transferred correctly without a CPU copy.

        The QFP8WT DCI path in spyre_mem.cpp derives host strides analytically
        from K and N, so it ignores the CPU tensor's actual strides. A transposed
        (non-contiguous) view can be passed directly without calling .contiguous().
        """
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        # t() produces a non-contiguous [128, 256] view with strides [1, 128]
        weight = torch.randn(256, 128, dtype=torch.float16).t().to(torch.float8_e4m3fn)
        assert not weight.is_contiguous(), (
            "pre-condition: weight must be non-contiguous"
        )
        dev = _dma_to_spyre_fp8_kernel(weight)

        assert dev.device.type == DEVICE_TYPE
        assert dev.dtype == torch.float8_e4m3fn

    def test_rejects_non_fp8(self):
        """Raises AssertionError for non-FP8 dtype."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(128, 128, dtype=torch.float16)
        with pytest.raises(AssertionError, match="float8_e4m3fn"):
            _dma_to_spyre_fp8_kernel(weight)

    def test_rejects_non_2d(self):
        """Raises AssertionError for non-2D tensor."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(4, 128, 128, dtype=torch.float16).to(torch.float8_e4m3fn)
        with pytest.raises(AssertionError, match="2D"):
            _dma_to_spyre_fp8_kernel(weight)

    def test_production_shapes(self):
        """Common Granite-3.3 8B Linear weight shapes transfer correctly."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        shapes = [
            (4096, 4096),  # q_proj, o_proj
            (1024, 4096),  # k_proj, v_proj (GQA)
            (12800, 4096),  # gate_proj, up_proj
            (4096, 12800),  # down_proj
        ]
        for out_f, in_f in shapes:
            weight = torch.randn(out_f, in_f, dtype=torch.float16).to(
                torch.float8_e4m3fn
            )
            dev = _dma_to_spyre_fp8_kernel(weight)
            assert dev.device.type == DEVICE_TYPE
            assert dev.dtype == torch.float8_e4m3fn
            assert list(dev.shape) == [out_f, in_f]


class TestLoadModelToSpyreUseFp8Weights:
    """Unit tests for load_model_to_spyre(use_fp8_weights=True) with a tiny model."""

    def _make_tiny_fp8_model(self):
        """Build a tiny 2-layer model with pre-quantized FP8 Linear weights.

        Shapes must satisfy the QFP8WT alignment constraints:
          - in_features (K after transpose) divisible by 2 (si=2)
          - out_features (N after transpose) divisible by 64 (so=64)
        """
        model = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 64, bias=False),
        )
        # Simulate pre-quantized FP8 weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight = nn.Parameter(
                    module.weight.data.to(torch.float8_e4m3fn),
                    requires_grad=False,
                )
        return model

    def test_fp8_weights_loaded_to_spyre(self):
        """FP8 Linear weights are moved to Spyre device."""
        from torch_spyre.model_utils import load_model_to_spyre

        model = self._make_tiny_fp8_model()
        load_model_to_spyre(model, use_fp8_weights=True)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert module.weight.device.type == DEVICE_TYPE, (
                    f"Expected weight on {DEVICE}, got {module.weight.device}"
                )
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"Expected fp8 dtype preserved, got {module.weight.dtype}"
                )

    def test_fp8_weights_dtype_preserved(self):
        """FP8 dtype is not upcast to BF16/FP16 during DMA."""
        from torch_spyre.model_utils import load_fp8_model_to_spyre

        model = self._make_tiny_fp8_model()
        load_fp8_model_to_spyre(model)

        fp8_weights = [
            (n, m.weight) for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]
        assert len(fp8_weights) == 2
        for name, w in fp8_weights:
            assert w.dtype == torch.float8_e4m3fn, (
                f"{name}: expected float8_e4m3fn, got {w.dtype}"
            )

    def test_non_fp8_weights_use_normal_path(self):
        """BF16 Linear weights still go through dim_order=[1,0] path."""
        from torch_spyre.model_utils import load_model_to_spyre

        model = nn.Sequential(nn.Linear(128, 64, bias=False))
        # BF16 weights — use_fp8_weights=True should NOT touch these
        model[0].weight = nn.Parameter(
            model[0].weight.data.to(torch.bfloat16), requires_grad=False
        )
        load_model_to_spyre(model, use_fp8_weights=True)

        assert model[0].weight.device.type == DEVICE_TYPE
        # dtype should be preserved (bfloat16 on device)
        assert model[0].weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# scaled_mm with pre-quantized FP8 weight
# ---------------------------------------------------------------------------


class TestScaledMmWithPrequantizedWeight:
    """Test the pre-quantized FP8 weight loading path for _scaled_mm.

    Uses synthetic FP8 weights (no model checkpoint required).  The weight is
    transferred directly to Spyre via _dma_to_spyre_fp8_kernel (QFP8WT KERNEL
    layout), bypassing any runtime qfp8wt quantization.

    ``test_kernel_layout_properties`` verifies the DMA transfer itself (dtype,
    shape, device) is correct.

    ``test_scaled_mm_prequantized_weight_xfail`` documents that a pre-loaded
    Spyre KERNEL tensor cannot be passed as a compiled-graph input — the weight
    is already on ``spyre:0`` before tracing, causing a device mismatch.

    The end-to-end correctness of _scaled_mm with a pre-quantized closed-over
    weight is covered by ``TestScaledMmPrequantizedClosedOver``.
    """

    @pytest.mark.parametrize(
        "n, k",
        [
            (128, 128),
            (4096, 4096),
            (12800, 4096),
        ],
    )
    def test_kernel_layout_properties(self, n, k):
        """_dma_to_spyre_fp8_kernel produces a Spyre FP8 tensor with correct properties."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight_fp8 = (
            torch.randn(n, k, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        q_weight = _dma_to_spyre_fp8_kernel(weight_fp8)

        assert q_weight.dtype == torch.float8_e4m3fn, (
            f"Expected float8_e4m3fn, got {q_weight.dtype}"
        )
        assert q_weight.device.type == DEVICE_TYPE, (
            f"Expected on {DEVICE_TYPE}, got {q_weight.device.type}"
        )
        assert q_weight.shape == torch.Size([n, k]), (
            f"Expected shape [{n}, {k}], got {q_weight.shape}"
        )

    @pytest.mark.xfail(
        reason=(
            "Weight as compiled-graph input causes device mismatch (cpu vs spyre:0) "
            "during tracing. Pre-loaded KERNEL weights must be closed over as frozen "
            "constants, not passed as graph inputs. See class docstring."
        ),
        strict=True,
    )
    @pytest.mark.parametrize(
        "m, k, n, scale_a, scale_b",
        [
            (1, 128, 128, 1.0, 1.0),
            (4, 4096, 4096, 2.0, 0.5),
        ],
    )
    def test_scaled_mm_prequantized_weight_xfail(self, m, k, n, scale_a, scale_b):
        """Documents that KERNEL FP8 weight as compiled-graph input is unsupported."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        torch.manual_seed(42)
        act = torch.randn(m, k, dtype=torch.float16)
        scale_a_t = torch.full((1,), scale_a, dtype=torch.float16)
        scale_b_t = torch.full((1,), scale_b, dtype=torch.float16)
        weight_fp8 = (
            torch.randn(n, k, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        q_weight = _dma_to_spyre_fp8_kernel(weight_fp8)

        def spyre_fn(act, q_weight, scale_a, scale_b):
            q_act = torch.ops.spyre.quantize_fp8_with_scale(act, scale_a)
            return torch.ops.aten._scaled_mm(
                q_act,
                q_weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                out_dtype=torch.float16,
            )

        def pytorch_fn(act, q_weight, scale_a, scale_b):
            q_a = (act / scale_a).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            a_f32 = q_a.to(torch.float32) * scale_a.item()
            b_f32 = q_weight.to(torch.float32) * scale_b.item()
            return (a_f32 @ b_f32.T).to(torch.float16)

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            act,
            q_weight,
            scale_a_t,
            scale_b_t,
            atol=1.0,
            rtol=0.1,
        )


# ---------------------------------------------------------------------------
# Pre-quantized FP8 weight closed-over as frozen constant
# ---------------------------------------------------------------------------


class TestScaledMmPrequantizedClosedOver:
    """Test _scaled_mm with a pre-quantized FP8 weight closed over in the compiled fn.

    This is the production pattern for pre-quantized checkpoint weights:
      1. Weight arrives on CPU as float8_e4m3fn from the checkpoint.
      2. _dma_to_spyre_fp8_kernel transfers it to Spyre in QFP8WT KERNEL layout.
      3. The on-device tensor is captured in the closure of the @torch.compile
         function — it is NOT passed as a graph input.

    Bug history:
      - Non-square shapes (K ≠ N) were silently wrong before two fixes:
        (a) spyre_mem.cpp QFP8WT DCI: K and N were swapped after the
            PyTorch→hardware axis reversal, scrambling bytes for K ≠ N.
        (b) work_division.py: `core_fold` was set to N//128 (= n_sticks)
            instead of the hardware constant 4, giving wrong SDSC coordInfo
            for all N except 512.
    """

    @pytest.mark.parametrize(
        "m, k, n",
        [
            # Square (K == N): passed before fixes (self-consistent wrong encoding)
            (4, 512, 512),
            (4, 4096, 4096),
            # Non-square (K ≠ N): failed before fixes — now the primary regression guard
            (4, 4096, 512),  # K > N  (e.g. k_proj/v_proj in Granite GQA)
            (4, 512, 1024),  # K < N
            (4, 4096, 1024),  # K > N  (e.g. k_proj/v_proj)
            (4, 2048, 4096),  # K < N  (e.g. down_proj)
        ],
    )
    def test_prequantized_closed_over(self, m, k, n):
        """Pre-quantized FP8 weight closed over as frozen constant produces correct output.

        Weight is quantized on CPU to float8_e4m3fn, transferred to Spyre via
        _dma_to_spyre_fp8_kernel, and closed over in the compiled function.
        CPU reference: quantize both inputs to FP8, dequantize, matmul in FP16.
        """
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        torch.manual_seed(42)
        act_cpu = torch.randn(m, k, dtype=torch.float16)
        weight_cpu = torch.randn(n, k, dtype=torch.float16)  # [n, k] → T → [k, n]
        scale_a = torch.tensor(1.0, dtype=torch.float16)
        scale_b = torch.tensor(1.0, dtype=torch.float16)

        # Simulate a checkpoint FP8 weight: quantize [n,k], transpose to [k,n] for matmul
        weight_fp8_T = (
            weight_cpu.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).T.contiguous()
        )
        # Transfer to Spyre as QFP8WT KERNEL tensor; close over in compiled fn
        q_w_spyre = _dma_to_spyre_fp8_kernel(weight_fp8_T)

        torch._dynamo.reset()

        @torch.compile(backend="inductor")
        def spyre_fn(act, sa, sb):
            q_act = torch.ops.spyre.quantize_fp8_with_scale(act, sa)
            # q_w_spyre is a frozen constant from the enclosing scope
            return torch.ops.aten._scaled_mm(
                q_act,
                q_w_spyre,
                scale_a=sa,
                scale_b=sb,
                bias=None,
                out_dtype=torch.float16,
            )

        result = spyre_fn(
            act_cpu.to(DEVICE),
            scale_a.to(DEVICE),
            scale_b.to(DEVICE),
        )

        # CPU reference: quantize both to FP8, dequantize, matmul
        q_a = act_cpu.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float16)
        q_b = weight_cpu.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float16)
        cpu_ref = (q_a @ q_b.T) * 1.0

        out_cpu = result.to("cpu").to(torch.float16)
        # atol=4.0: both paths quantize to FP8 E4M3 (max spacing ~0.5 in the
        # range used here) and accumulate K terms. For K=4096 the worst-case
        # absolute error is O(K * fp8_spacing) ≈ 4096 * 0.5 * eps_fp16 ≈ 2–4.
        torch.testing.assert_close(out_cpu, cpu_ref, atol=4.0, rtol=0.1)


# ---------------------------------------------------------------------------
# End-to-end: quantscalepertokenfp8 + pre-quantized KERNEL weight
# ---------------------------------------------------------------------------


class TestQuantScalePerTokenFp8WithPrequantizedWeight:
    """End-to-end test for the full FP8 inference pipeline:

        quantscalepertokenfp8(act)
          → quantize_fp8_with_scale(act, scale)
          → _scaled_mm(q_act, q_weight_prequantized, ...)

    The pre-quantized weight is loaded via _dma_to_spyre_fp8_kernel and closed
    over as a frozen constant in the compiled function — the production pattern
    for FP8 checkpoint inference.
    """

    @pytest.mark.parametrize(
        "m, k, n",
        [
            (1, 512, 512),
            (4, 4096, 512),
            (4, 4096, 1024),
        ],
    )
    def test_full_fp8_pipeline(self, m, k, n):
        """quantscalepertokenfp8 + pre-loaded KERNEL weight produces correct output."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        torch.manual_seed(0)
        act_cpu = torch.randn(m, k, dtype=torch.float16)
        weight_cpu = torch.randn(n, k, dtype=torch.float16)

        # Simulate checkpoint: quantize [n, k], transpose to [k, n] for matmul
        weight_fp8_T = (
            weight_cpu.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).T.contiguous()
        )
        q_w_spyre = _dma_to_spyre_fp8_kernel(weight_fp8_T)

        torch._dynamo.reset()

        @torch.compile(backend="inductor")
        def spyre_fn(act):
            # Step 1: compute per-token scale from activation amax
            scale = torch.ops.spyre.quantscalepertokenfp8(act)
            # Step 2: quantize activation using computed scale
            q_act = torch.ops.spyre.quantize_fp8_with_scale(act, scale)
            # Step 3: matmul with pre-loaded KERNEL weight
            return torch.ops.aten._scaled_mm(
                q_act,
                q_w_spyre,
                scale_a=scale,
                scale_b=torch.tensor(1.0, dtype=torch.float16, device=act.device),
                bias=None,
                out_dtype=torch.float16,
            )

        result = spyre_fn(act_cpu.to(DEVICE))

        # CPU reference: quantize both inputs, dequantize, matmul
        amax = act_cpu.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale_ref = amax / 448.0
        q_a = (act_cpu / scale_ref).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_a_fp16 = q_a.to(torch.float16) * scale_ref
        q_b_fp16 = (
            weight_cpu.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float16)
        )
        cpu_ref = q_a_fp16 @ q_b_fp16.T

        out_cpu = result.to("cpu").to(torch.float16)
        # atol=4.0: FP8 quantization error accumulated over K terms (see
        # test_prequantized_closed_over for tolerance justification).
        torch.testing.assert_close(out_cpu, cpu_ref, atol=4.0, rtol=0.1)
