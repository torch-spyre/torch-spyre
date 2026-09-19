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

from __future__ import annotations

import pytest
import torch
from torch_spyre.constants import DEVICE_NAME

E4M3 = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = float(torch.finfo(E4M3).max) if E4M3 is not None else 448.0
FP16 = torch.float16
BF16 = torch.bfloat16
FP32 = torch.float32
DEVICE = DEVICE_NAME
BACKEND = "inductor"


def _spyre_available() -> bool:
    try:
        import torch_spyre  # noqa: F401

        return torch.spyre.is_available()
    except (ImportError, AttributeError):
        return False


def _fp8_available() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def _quantize_op_available() -> bool:
    try:
        import torch_spyre  # noqa: F401

        _ = torch.ops.spyre.quantize_fp8_with_scale
        return True
    except (ImportError, AttributeError):
        return False


skip_no_spyre = pytest.mark.skipif(
    not _spyre_available(),
    reason="Spyre device not available",
)
skip_no_fp8 = pytest.mark.skipif(
    not _fp8_available(),
    reason="FP8 dtype not in this PyTorch build",
)
skip_no_quantize_ops = pytest.mark.skipif(
    not _quantize_op_available(),
    reason="quantize_fp8_with_scale not registered (requires PR #2401)",
)

pytestmark = [skip_no_spyre, skip_no_fp8]


def _amax_scale(t: torch.Tensor) -> float:
    """Amax-based scale: amax(|t|)/448 — maps max→FP8 max. OVERFLOWS FP16 for K≥1."""
    return max(float(t.float().abs().amax()) / FP8_MAX, 1e-6)


def _safe_dynamic_scale(t: torch.Tensor) -> float:
    """Data-driven scale: 2×amax(|t|) — maps max→0.5, safe for FP16 accumulation."""
    return max(float(t.float().abs().amax()) * 2.0, 1e-6)


def _st16(val: float, device: str = "cpu") -> torch.Tensor:
    """FP16 scalar scale tensor. quantize_fp8_with_scale requires FP16 scale."""
    return torch.tensor(val, dtype=FP16, device=device)


def _pipeline_ref(
    a: torch.Tensor,
    w: torch.Tensor,
    sa: float,
    sw: float,
    out_dtype: torch.dtype = FP16,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU oracle: (clip(a/sa)@clip(w/sw)) * sa*sw [+bias], all intermediate in FP32."""
    a_fp8 = (a.cpu().float() / sa).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    w_fp8 = (w.cpu().float() / sw).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    out = (a_fp8.float() @ w_fp8.float()) * (sa * sw)
    if bias is not None:
        out = out + bias.cpu().float()
    return out.to(out_dtype)


def _make_inputs(M: int, K: int, N: int, seed: int = 0, device: str = "cpu"):
    """FP16 randn (M,K) activation and (K,N) weight; seeded on CPU then moved to device."""
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=FP16)
    w = torch.randn(K, N, dtype=FP16)
    return a.to(device), w.to(device)


def _make_rand_inputs(M: int, K: int, N: int, seed: int = 0, device: str = "cpu"):
    """FP16 rand [0,1] (M,K) activation and (K,N) weight; seeded on CPU then moved to device."""
    torch.manual_seed(seed)
    a = torch.rand(M, K, dtype=FP16)
    w = torch.rand(K, N, dtype=FP16)
    return a.to(device), w.to(device)


def _compile(fn, fullgraph: bool = False):
    return torch.compile(fn, backend=BACKEND, fullgraph=fullgraph)


def _pipeline_fn(a, w, sa, sw, out_dtype=FP16):
    """Spyre FP8 linear pipeline: quantize_fp8_with_scale → quantize_weight → _scaled_mm."""
    a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)
    return torch._scaled_mm(a_fp8, w_fp8, scale_a=sa, scale_b=sw, out_dtype=out_dtype)


def _pipeline_bias_fn(a, w, sa, sw, bias, out_dtype=FP16):
    """FP8 linear pipeline with bias epilogue: quantize → mm → out + bias."""
    a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)
    return torch._scaled_mm(
        a_fp8, w_fp8, scale_a=sa, scale_b=sw, bias=bias, out_dtype=out_dtype
    )


class TestFP8Correctness:
    """Sunny-day pipeline correctness: K%128==0, Spyre quantize ops, compiled path."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_basic_2x128x128(self):
        """(2,128)@(128,128)→FP16 baseline: rand inputs, sa=sw=1.0."""
        M, K, N = 2, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any(), "NaN in pipeline output"
        assert not out.isinf().any(), "Inf in pipeline output"
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_zero_activation(self):
        """Zero activation → zero output regardless of weight (sa=sw=1.0)."""
        M, K, N = 16, 128, 128
        a = torch.zeros(M, K, dtype=FP16, device=DEVICE)
        _, w = _make_rand_inputs(M, K, N, seed=1, device=DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert not out.isnan().any()
        assert out.abs().max() == 0.0, "Zero activation must produce zero output"

    def test_uniform_input_unit_scale(self):
        """Uniform 0.25 input, scale=1.0: each output element = K × 0.0625 (exact in FP8)."""
        M, K, N = 16, 128, 128
        a = torch.full((M, K), 0.25, dtype=FP16, device=DEVICE)
        w = torch.full((K, N), 0.25, dtype=FP16, device=DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)
        expected = torch.full((M, N), K * 0.25 * 0.25, dtype=FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        torch.testing.assert_close(out, expected, atol=0.1, rtol=0.01)

    @pytest.mark.parametrize("M", [1, 8, 16, 32, 64])
    def test_various_m_at_k128(self, M):
        """K=128, M in {1,8,16,32,64}: M-dimension scaling at baseline K."""
        K, N = 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=M, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N), f"Expected ({M},{N}), got {out.shape}"
        assert out.dtype == FP16
        assert not out.isnan().any(), f"NaN at M={M}"
        assert not out.isinf().any(), f"Inf at M={M}"
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_out_dtype_fp16(self):
        """out_dtype=FP16: native Spyre accumulation dtype."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)
        fn_c = _compile(lambda a, w, sa, sw: _pipeline_fn(a, w, sa, sw, FP16))
        out = fn_c(a, w, sa, sw).cpu()
        assert out.dtype == FP16, f"Expected FP16, got {out.dtype}"
        assert out.shape == (M, N)
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_identity_weight(self):
        """Identity weight matrix: mm(a_fp8, I_fp8) = a_fp8, isolates layout and multiply bugs."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, _ = _make_rand_inputs(M, K, N, seed=3, device=DEVICE)
        w = torch.eye(K, dtype=FP16).to(DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any(), "NaN in identity-weight output"
        assert not out.isinf().any(), "Inf in identity-weight output"
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4311: non-contiguous activation causes unsupported stick expression"
    )
    def test_noncontiguous_activation(self):
        """Non-contiguous activation (strides=(1,K)): quantize_fp8_with_scale on transposed view."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        torch.manual_seed(7)
        a_noncontig = (
            torch.rand(K, M, dtype=FP16).to(DEVICE).t()
        )  # shape (M,K), strides=(1,K)
        w = torch.rand(K, N, dtype=FP16).to(DEVICE)
        assert not a_noncontig.is_contiguous(), (
            "Pre-condition: activation must be non-contiguous"
        )
        ref = _pipeline_ref(a_noncontig, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a_noncontig, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4305: non-unit dynamic scale epilogue gives wrong output"
    )
    def test_safe_dynamic_scale(self):
        """Dynamic safe scale (amax-computed) for both sa and sw."""
        M, K, N = 16, 128, 128
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=FP16)
        w = torch.randn(K, N, dtype=FP16)
        sa_val = _safe_dynamic_scale(a)
        sw_val = _safe_dynamic_scale(w)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)


class TestFP8CanonicalShapes:
    """Canonical pipeline shapes from test_fp8_scaled_mm_cpu."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_2x128x128_unit_scale(self):
        """M=2,K=128,N=128 with sa=sw=1.0 — canonical smoke test."""
        M, K, N = 2, 128, 128
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=0)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any(), "NaN in canonical 2x128x128 output"
        assert not out.isinf().any(), "Inf in canonical 2x128x128 output"
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_128x128x128_unit_scale(self):
        """M=128,K=128,N=128 with sa=sw=1.0 — large M case."""
        M, K, N = 128, 128, 128
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=1)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_4x128x1024_asymm_scale(self):
        """M=4,K=128,N=1024 with sa=1.0,sw=3.0 — wide N, asymmetric scale."""
        M, K, N = 4, 128, 1024
        sa_val, sw_val = 1.0, 3.0
        a, w = _make_rand_inputs(M, K, N, seed=2)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.parametrize(
        "sa_val,sw_val,seed,label",
        [
            (1.0, 1.0, 10, "unit"),
            (2.0, 1.0, 11, "sa2"),
        ],
    )
    def test_scale_variations_k128(self, sa_val, sw_val, seed, label):
        """K=128, M=16, N=128 with various fixed scales."""
        M, K, N = 16, 128, 128
        a, w = _make_rand_inputs(M, K, N, seed=seed)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N), f"[{label}] shape mismatch"
        assert out.dtype == FP16
        assert not out.isnan().any(), f"[{label}] NaN in output"
        assert not out.isinf().any(), f"[{label}] Inf in output"
        torch.testing.assert_close(
            out,
            ref,
            atol=0.1,
            rtol=0.05,
            msg=f"[{label}] pipeline output vs CPU oracle",
        )

    def test_2x128x100_unit_scale(self):
        """M=2,K=128,N=100 — from SCALED_MM_TESTS_SUPPORTED shapes."""
        M, K, N = 2, 128, 100
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=20)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_3x128x128_unit_scale(self):
        """M=3,K=128,N=128 — from SCALED_MM_TESTS_SUPPORTED shapes."""
        M, K, N = 3, 128, 128
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=21)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_17x128x128_unit_scale(self):
        """M=17 — non-power-of-2, non-multiple-of-16 leading dim."""
        M, K, N = 17, 128, 128
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=22)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.slow
    def test_2x4096x4096_unit_scale(self):
        """@slow — M=2,K=4096,N=4096 from SCALED_MM_TESTS_SUPPORTED."""
        M, K, N = 2, 4096, 4096
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=30)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.slow
    def test_4x4096x4096_unit_scale(self):
        """@slow — M=4,K=4096,N=4096 from SCALED_MM_TESTS_SUPPORTED."""
        M, K, N = 4, 4096, 4096
        sa_val, sw_val = 1.0, 1.0
        a, w = _make_rand_inputs(M, K, N, seed=31)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(
            a.to(DEVICE), w.to(DEVICE), _st16(sa_val, DEVICE), _st16(sw_val, DEVICE)
        ).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)


class TestFP8ScaleSemantics:
    """Scale divide-in-quantize / multiply-in-post-mm duality: net effect is out ≈ a @ w."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_unit_scale_net_effect_is_matmul(self):
        """sa=sw=1.0: scale cancels, output ≈ a @ w (quantization noise only)."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=0, device=DEVICE)
        # Unquantized reference: a.float() @ w.float() in FP16 (computed on CPU)
        raw_ref = (a.cpu().float() @ w.cpu().float()).to(FP16)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        # Pipeline output matches oracle
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)
        # Pipeline output also close to raw matmul (scale cancels out)
        torch.testing.assert_close(out.float(), raw_ref.float(), atol=0.1, rtol=0.05)

    def test_scale_cancel_duality_symmetric_nonunit(self):
        """sa=sw=2.0 produces same result as sa=sw=1.0: divide-in-quantize / multiply-post-mm cancel."""
        M, K, N = 16, 128, 128
        a, w = _make_rand_inputs(M, K, N, seed=7, device=DEVICE)
        sa1 = _st16(1.0, DEVICE)
        sw1 = _st16(1.0, DEVICE)
        sa2 = _st16(2.0, DEVICE)
        sw2 = _st16(2.0, DEVICE)
        ref = _pipeline_ref(a, w, 1.0, 1.0, FP16)

        fn_c1 = _compile(_pipeline_fn)
        out_sa1 = fn_c1(a, w, sa1, sw1).cpu()
        assert out_sa1.shape == (M, N)
        assert not out_sa1.isnan().any()
        assert not out_sa1.isinf().any()
        torch.testing.assert_close(out_sa1, ref, atol=0.1, rtol=0.05)

        torch._dynamo.reset()
        fn_c2 = _compile(_pipeline_fn)
        out_sa2 = fn_c2(a, w, sa2, sw2).cpu()
        assert out_sa2.shape == (M, N)
        assert not out_sa2.isnan().any()
        assert not out_sa2.isinf().any()
        torch.testing.assert_close(out_sa2, ref, atol=0.1, rtol=0.05)

        # Scale cancel-out duality: sa=sw=2.0 must match sa=sw=1.0
        torch.testing.assert_close(out_sa1, out_sa2, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4305: non-unit dynamic scale epilogue gives wrong output"
    )
    def test_dynamic_safe_scale_from_data(self):
        """Data-driven scale via _safe_dynamic_scale (maps max→0.5, no FP16 overflow)."""
        M, K, N = 16, 128, 128
        torch.manual_seed(11)
        a = torch.randn(M, K, dtype=FP16)
        w = torch.randn(K, N, dtype=FP16)
        # Dynamic scales must be computed from CPU tensors before device move
        sa_val = _safe_dynamic_scale(a)
        sw_val = _safe_dynamic_scale(w)
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)
        a, w = a.to(DEVICE), w.to(DEVICE)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4305: non-unit dynamic scale epilogue gives wrong output"
    )
    def test_asymmetric_dynamic_scales(self):
        """Asymmetric dynamic scales: sa >> sw (3× vs 0.1× inputs), applied independently."""
        M, K, N = 16, 128, 128
        torch.manual_seed(99)
        a = torch.randn(M, K, dtype=FP16) * 3.0
        w = torch.randn(K, N, dtype=FP16) * 0.1
        # Dynamic scales must be computed from CPU tensors before device move
        sa_val = _safe_dynamic_scale(a)
        sw_val = _safe_dynamic_scale(w)
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)
        a, w = a.to(DEVICE), w.to(DEVICE)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4305: test expects scale-drop overflow but scale is now applied"
    )
    def test_amax_scale_fp16_pipeline(self):
        """Amax-scale with K=128: K × 448² > FP16 max → NaN/Inf expected (documents overflow)."""
        M, K, N = 16, 128, 128
        torch.manual_seed(7)
        a = torch.randn(M, K, dtype=FP16)
        w = torch.randn(K, N, dtype=FP16)
        sa_val = _amax_scale(a)
        sw_val = _amax_scale(w)
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = a.to(DEVICE), w.to(DEVICE)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        # When https://github.com/torch-spyre/torch-spyre/issues/4305 is fixed, re-evaluate: the
        # sa×sw factor will reduce output magnitude and overflow may disappear.
        assert out.isnan().any() or out.isinf().any(), (
            "Expected FP16 overflow with amax scale — K × 448² > FP16 max (65504)"
        )

    @pytest.mark.parametrize(
        "sa_val,sw_val,seed",
        [
            (2.0, 1.0, 33),
            (1.0, 3.0, 34),
        ],
        ids=["sa2_sw1", "sa1_sw3"],
    )
    def test_static_asymmetric_scales(self, sa_val, sw_val, seed):
        """Static asymmetric scales (one non-unit): non-unit side must be applied by SDSC epilogue."""
        M, K, N = 16, 128, 128
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=seed, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)


class TestFP8AlignedShapes:
    """K%128==0 shape grid: irregular M, wide N, K=256, asymmetric sw — all K-aligned."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.parametrize(
        "M,K,N,sa_val,sw_val,seed,label",
        [
            (16, 128, 128, 1.0, 1.0, 40, "base"),
            pytest.param(
                8,
                128,
                1024,
                1.0,
                1.0,
                41,
                "wide_n",
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/4309: large N=1024 causes dxp_standalone SIGABRT"
                ),
            ),
            pytest.param(
                32,
                256,
                256,
                1.0,
                1.0,
                42,
                "k256",
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/4309: M=32 with K=256 causes dxp_standalone SIGABRT"
                ),
            ),
            (4, 128, 128, 1.0, 1.0, 43, "small_m4"),
            # Canonical K=128 shapes from test_fp8_scaled_mm_cpu param_sets:
            (128, 128, 128, 1.0, 1.0, 1, "m128"),
            (4, 128, 1024, 1.0, 3.0, 2, "asym_sw3"),
            (2, 128, 100, 1.0, 1.0, 20, "n100"),
            (3, 128, 128, 1.0, 1.0, 21, "m3"),
            (17, 128, 128, 1.0, 1.0, 22, "m17"),
        ],
        ids=[
            "base",
            "wide_n",
            "k256",
            "small_m4",
            "m128",
            "asym_sw3",
            "n100",
            "m3",
            "m17",
        ],
    )
    def test_aligned_k_shapes(self, M, K, N, sa_val, sw_val, seed, label):
        """K-aligned (K%128==0) shape and oracle correctness across M, N, and scale variants."""
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=seed, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N), f"[{label}] shape mismatch: {out.shape}"
        assert out.dtype == FP16
        assert not out.isnan().any(), f"[{label}] NaN in output"
        assert not out.isinf().any(), f"[{label}] Inf in output"
        torch.testing.assert_close(
            out, ref, atol=0.1, rtol=0.05, msg=f"[{label}] output vs CPU oracle"
        )


class TestFP8UnalignedShapes:
    """K%128!=0: insert_bmm_padding pads weight K to next multiple of 128."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4308: K%128!=0 causes dxp_standalone SIGABRT; K=64 scheduler fails"
    )
    def test_k64_hardcoded_scale(self):
        """K=64 (K%128=64), sa=sw=1.0: padding to K=128, value correctness vs oracle."""
        M, K, N = 16, 64, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()
        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4308: K%128!=0 causes dxp_standalone SIGABRT; K=16 scheduler fails"
    )
    def test_k16_hardcoded_scale(self):
        """K=16 (K%128=16), sa=sw=1.0: padding to K=128, value correctness vs oracle."""
        M, K, N = 16, 16, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()
        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4308: K%128!=0; K=2880 causes dxp_standalone SIGABRT"
    )
    def test_k2880_medium_unaligned(self):
        """K=2880 (K%128=64), sa=sw=1.0: padded 2880→2944, value correctness vs oracle."""
        M, K, N = 16, 2880, 2880
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=92, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()
        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4308: K%128!=0 causes dxp_standalone SIGABRT; K=4032 scheduler fails"
    )
    def test_k4032_large_padding(self):
        """@slow — K=4032 (K%128=64), padded 4032→4096, value correctness at production K."""
        M, K, N = 16, 4032, 4096
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=90, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()
        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4308: K%128!=0 causes dxp_standalone SIGABRT; K=3904 scheduler fails"
    )
    def test_k3904_large_padding(self):
        """@slow — K=3904 (K%128=64), padded 3904→3968, value correctness at production K."""
        M, K, N = 16, 3904, 4096
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=91, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()
        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)


class TestFP8LargeShapes:
    """Production LLM shapes (all @slow): M sweep at K=N=4096, Llama up_proj, Granite q_proj."""

    pytestmark = [skip_no_quantize_ops, pytest.mark.slow]

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.parametrize(
        "M",
        [
            1,
            2,
            4,
            8,
            16,
            pytest.param(
                32,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/4309: M=32 causes dxp_standalone SIGABRT at K=N=4096"
                ),
            ),
        ],
        ids=["m1", "m2", "m4", "m8", "m16", "m32"],
    )
    def test_m_sweep_k4096_n4096(self, M):
        """@slow — decode batch sizes M=1..32 at K=N=4096 (K-aligned), oracle correctness."""
        K, N = 4096, 4096
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=M * 100, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N), f"shape mismatch at M={M}"
        assert out.dtype == FP16
        assert not out.isnan().any(), f"NaN at M={M}"
        assert not out.isinf().any(), f"Inf at M={M}"
        torch.testing.assert_close(
            out, ref, atol=1.0, rtol=0.05, msg=f"oracle mismatch at M={M}"
        )

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4309: N=16384 causes dxp_standalone SIGABRT"
    )
    def test_wide_n_k4096_n16384(self):
        """@slow — (1,4096)@(4096,16384): Llama-style up_proj, large output width."""
        M, K, N = 1, 4096, 16384
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=200, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4309: large M=2048 causes dxp_standalone SIGABRT"
    )
    def test_large_m_k2048_n4096(self):
        """@slow — M=2048, K=2048, N=4096: large batch at medium K (K-aligned)."""
        M, K, N = 2048, 2048, 4096
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=300, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4309: large M=2048 and N=65536 causes dxp_standalone SIGABRT"
    )
    def test_m2048_k2048_n65536(self):
        """@slow — (2048,2048)@(2048,65536): mirrors test_large_matmul 2d shape, wide N."""
        M, K, N = 2048, 2048, 65536
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=310, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4311: non-contiguous activation causes unsupported stick expression"
    )
    def test_noncontiguous_activation_k4096(self):
        """@slow — non-contiguous activation (strides=(1,K)) at K=4096; extends K=128 TC."""
        M, K, N = 16, 4096, 4096
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        torch.manual_seed(400)
        a_noncontig = torch.rand(K, M, dtype=FP16).to(DEVICE).t()
        w = torch.rand(K, N, dtype=FP16).to(DEVICE)
        assert not a_noncontig.is_contiguous()
        ref = _pipeline_ref(a_noncontig, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a_noncontig, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    def test_granite_qproj(self):
        """@slow — (16,4096)@(4096,4096): Granite 3.x q_proj production decode shape."""
        M, K, N = 16, 4096, 4096
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=1, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)


class TestFP8Bias:
    """Bias epilogue: out = batchmatmulfp8(a, w) * sa * sw + bias, applied post-scale."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_bias_basic_fp16(self):
        """(16,128)@(128,128)+bias[N]→FP16: bias[5,15] large enough to detect a dropped epilogue."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, device=DEVICE)
        torch.manual_seed(3)
        bias = (torch.rand(N, dtype=FP16) * 10 + 5).to(DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16, bias=bias)

        fn_c = _compile(_pipeline_bias_fn)
        out = fn_c(a, w, sa, sw, bias).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_scalar_bias_unit_scale(self):
        """0-dim scalar bias broadcasts to all (M,N): documents SDSC scalar-epilogue path."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=8, device=DEVICE)
        bias_val = 11.0
        bias = torch.tensor(bias_val, dtype=FP16, device=DEVICE)
        ref = _pipeline_ref(
            a, w, sa_val, sw_val, FP16, bias=torch.tensor(bias_val, dtype=FP16)
        )

        fn_c = _compile(_pipeline_bias_fn)
        out = fn_c(a, w, sa, sw, bias).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_bias_with_nonunit_scales(self):
        """Bias[N] + sa=2.0, sw=4.0: verifies out = raw_mm * sa * sw + bias ordering."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 2.0, 4.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=15, device=DEVICE)
        torch.manual_seed(16)
        bias = (torch.rand(N, dtype=FP16) * 10 + 5).to(DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16, bias=bias)

        fn_c = _compile(_pipeline_bias_fn)
        out = fn_c(a, w, sa, sw, bias).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_negative_bias_zero_activation(self):
        """Zero activation + bias=-3.0: mm(zeros,ones)=0, so out = bias exactly."""
        M, K, N = 16, 128, 128
        bias_val = -3.0
        a = torch.zeros(M, K, dtype=FP16, device=DEVICE)
        w = torch.ones(K, N, dtype=FP16, device=DEVICE)
        bias = torch.full((N,), bias_val, dtype=FP16, device=DEVICE)
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)

        fn_c = _compile(_pipeline_bias_fn)
        out = fn_c(a, w, sa, sw, bias).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        expected = torch.full((M, N), bias_val, dtype=FP16)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_bias_exact_difference(self):
        """|out_with_bias - out_no_bias| == 4.0 element-wise: mirrors upstream test_float8_bias."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        bias_val = 4.0
        a, w = _make_rand_inputs(M, K, N, seed=42, device=DEVICE)
        bias = torch.full((N,), bias_val, dtype=FP16, device=DEVICE)

        fn_no_bias = _compile(_pipeline_fn)
        out_no_bias = fn_no_bias(a, w, sa, sw).cpu()

        torch._dynamo.reset()
        fn_with_bias = _compile(_pipeline_bias_fn)
        out_with_bias = fn_with_bias(a, w, sa, sw, bias).cpu()

        assert out_no_bias.shape == out_with_bias.shape == (M, N)
        diff = (out_with_bias.float() - out_no_bias.float()).abs()
        expected_diff = torch.full((M, N), bias_val)
        torch.testing.assert_close(diff, expected_diff, atol=0.1, rtol=0.0)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4305: non-unit dynamic scale epilogue gives wrong output"
    )
    def test_dynamic_asymmetric_scale_with_bias(self):
        """Dynamic asymmetric scales (a×3, w×0.1) + bias[N]: all three epilogues active."""
        M, K, N = 16, 128, 128
        torch.manual_seed(22)
        a = torch.randn(M, K, dtype=FP16) * 3.0
        w = torch.randn(K, N, dtype=FP16) * 0.1
        # Dynamic scales must be computed from CPU tensors before device move
        sa_val = _safe_dynamic_scale(a)
        sw_val = _safe_dynamic_scale(w)
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        torch.manual_seed(23)
        bias = torch.rand(N, dtype=FP16) * 10 + 5
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16, bias=bias)
        a, w, bias = a.to(DEVICE), w.to(DEVICE), bias.to(DEVICE)

        fn_c = _compile(_pipeline_bias_fn)
        out = fn_c(a, w, sa, sw, bias).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)


class TestFP8CompileIntegrity:
    """Compilation properties: fullgraph (no graph break), bit-identical repeated calls, fresh-recompile."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_fullgraph_compile(self):
        """fullgraph=True: quantize → _scaled_mm compiles as one graph, no graph break."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=42, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        fn_c = _compile(_pipeline_fn, fullgraph=True)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_repeated_calls_bit_identical(self):
        """5 consecutive compiled calls return bit-identical results: no DDL/SRAM state leak."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=11, device=DEVICE)

        fn_c = _compile(_pipeline_fn)
        first = fn_c(a, w, sa, sw).cpu()
        assert not first.isnan().any() and not first.isinf().any(), (
            "First call produced nan/inf — inputs are not numerically safe"
        )
        for i in range(4):
            out = fn_c(a, w, sa, sw).cpu()
            torch.testing.assert_close(
                out, first, atol=0.0, rtol=0.0, msg=f"Result changed at call {i + 2}"
            )

    def test_weight_change_changes_output(self):
        """Different weight tensors per call produce different outputs: no weight caching."""
        M, K, N = 16, 128, 128
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=FP16).to(DEVICE)
        w1 = torch.randn(K, N, dtype=FP16).to(DEVICE)
        w2 = torch.randn(K, N, dtype=FP16).to(DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)

        fn_c = _compile(_pipeline_fn)
        out1 = fn_c(a, w1, sa, sw).cpu()
        out2 = fn_c(a, w2, sa, sw).cpu()

        assert out1.shape == out2.shape == (M, N)
        # Two independent random weight matrices must produce distinct outputs
        assert not torch.allclose(out1, out2), (
            "Independent weight matrices produced identical output (weight caching bug)"
        )

    def test_recompile_after_dynamo_reset(self):
        """Fresh compile after dynamo reset matches oracle: no stale cache dependency."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=77, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        # First compile + call
        fn_c1 = _compile(_pipeline_fn)
        fn_c1(a, w, sa, sw)

        # Reset and recompile
        torch._dynamo.reset()
        fn_c2 = _compile(_pipeline_fn)
        out = fn_c2(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)


class TestFP8OptionalParams:
    """Optional _scaled_mm params: scale_result/use_fast_accum accepted (no-op); FP32 out_dtype unsupported on Spyre."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_scale_result_output_matches_oracle(self):
        """scale_result accepted but not applied: output matches the plain pipeline oracle."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=7, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)
        scale_result = torch.tensor(2.0, dtype=FP32, device=DEVICE)

        def _fn_scale_result(a, w, sa, sw, sr):
            a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
            w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)
            return torch._scaled_mm(
                a_fp8, w_fp8, scale_a=sa, scale_b=sw, scale_result=sr, out_dtype=FP16
            )

        fn_c = _compile(_fn_scale_result)
        out = fn_c(a, w, sa, sw, scale_result).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_fast_accum_output_matches_oracle(self):
        """use_fast_accum=True accepted but no-op: output matches the plain pipeline oracle."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=8, device=DEVICE)
        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)

        def _fn_fast_accum(a, w, sa, sw):
            a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
            w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)
            return torch._scaled_mm(
                a_fp8,
                w_fp8,
                scale_a=sa,
                scale_b=sw,
                use_fast_accum=True,
                out_dtype=FP16,
            )

        fn_c = _compile(_fn_fast_accum)
        out = fn_c(a, w, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == FP16
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    def test_fast_accum_identical_to_default(self):
        """use_fast_accum=True is bit-identical to default: no hardware distinction on Spyre."""
        M, K, N = 16, 128, 128
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a, w = _make_rand_inputs(M, K, N, seed=55, device=DEVICE)

        fn_default = _compile(_pipeline_fn)
        out_default = fn_default(a, w, sa, sw).cpu()
        assert not out_default.isnan().any() and not out_default.isinf().any()

        torch._dynamo.reset()

        def _fn_fast_accum(a, w, sa, sw):
            a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
            w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)
            return torch._scaled_mm(
                a_fp8,
                w_fp8,
                scale_a=sa,
                scale_b=sw,
                use_fast_accum=True,
                out_dtype=FP16,
            )

        fn_fast = _compile(_fn_fast_accum)
        out_fast = fn_fast(a, w, sa, sw).cpu()

        torch.testing.assert_close(out_default, out_fast, atol=0.0, rtol=0.0)


def _col_major(t: torch.Tensor) -> torch.Tensor:
    """Return mat2 in col-major layout (stride[0]==1) — upstream PyTorch contract."""
    return t.t().contiguous().t()


def _raw_quant(t: torch.Tensor, scale: float) -> torch.Tensor:
    """Raw FP8 cast via divide-clamp-convert (no Spyre quantize op)."""
    return (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(E4M3)


def _upstream_oracle(
    a_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sa: float,
    sw: float,
    out_dtype: torch.dtype = BF16,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU oracle for pre-quantized FP8: (a@w)*sa*sw [+bias], all in FP32 then cast."""
    out = (a_fp8.cpu().float() @ w_fp8.cpu().float()) * (sa * sw)
    if bias is not None:
        out = out + bias.cpu().float()
    return out.to(out_dtype)


class TestFP8UpstreamAPI:
    """Upstream _scaled_mm API: raw FP8 cast, col-major mat2, FP32/FP16 scales, no Spyre quantize ops."""

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4310: upstream _scaled_mm with FP8 inputs causes SIGABRT in ddl_conversion"
    )
    def test_upstream_k128_unit_scale_fp16(self):
        """K=128, FP16 unit scale, uniform 0.25 FP8 inputs: exact output match."""
        M, K, N = 16, 128, 128
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)
        a_fp8_cpu = torch.full((M, K), 0.25, dtype=E4M3)
        w_fp8_cpu = _col_major(torch.full((K, N), 0.25, dtype=E4M3))
        a_fp8 = a_fp8_cpu.to(DEVICE)
        w_fp8 = w_fp8_cpu.to(DEVICE)
        ref = _upstream_oracle(a_fp8_cpu, w_fp8_cpu, 1.0, 1.0, BF16)

        fn_c = _compile(
            lambda a, b, sa, sb: torch._scaled_mm(a, b, sa, sb, out_dtype=BF16)
        )
        out = fn_c(a_fp8, w_fp8, sa, sw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == BF16
        assert not out.isnan().any()
        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4310: upstream _scaled_mm SIGABRT; https://github.com/torch-spyre/torch-spyre/issues/4308: K=64 unaligned"
    )
    def test_upstream_k64_fp16_scale(self):
        """K=64 (K%128=64), FP16 scale: insert_bmm_padding pads 64→128, oracle correctness."""
        M, K, N = 16, 64, 128
        torch.manual_seed(5)
        a_f32 = torch.rand(M, K)
        w_f32 = torch.rand(K, N)
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)
        a_fp8_cpu = _raw_quant(a_f32, sa_val)
        w_fp8_cpu = _col_major(_raw_quant(w_f32, sw_val))
        a_fp8 = a_fp8_cpu.to(DEVICE)
        w_fp8 = w_fp8_cpu.to(DEVICE)
        ref = _upstream_oracle(a_fp8_cpu, w_fp8_cpu, sa_val, sw_val, BF16)

        fn_c = _compile(
            lambda a, b, sa, sb: torch._scaled_mm(a, b, sa, sb, out_dtype=BF16)
        )
        out = fn_c(a_fp8, w_fp8, sa, sw).cpu()
        assert out.shape == (M, N)
        assert out.dtype == BF16
        assert not out.isnan().any()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4310: rowwise scales trigger split_multi_ops KeyError"
    )
    def test_upstream_rowwise_scale_shape(self):
        """Per-token scales [M,1]/[1,N] (torchao/vLLM pattern): documents Spyre acceptance."""
        M, K, N = 16, 128, 128
        a_fp8_cpu = torch.full((M, K), 0.5, dtype=E4M3)
        w_fp8_cpu = _col_major(torch.full((K, N), 0.5, dtype=E4M3))
        a_fp8 = a_fp8_cpu.to(DEVICE)
        w_fp8 = w_fp8_cpu.to(DEVICE)
        scale_a_rw = torch.ones(M, 1, dtype=FP32, device=DEVICE)
        scale_b_rw = torch.ones(1, N, dtype=FP32, device=DEVICE)

        fn_c = _compile(
            lambda a, b, sa, sb: torch._scaled_mm(a, b, sa, sb, out_dtype=BF16)
        )
        out = fn_c(a_fp8, w_fp8, scale_a_rw, scale_b_rw).cpu()

        assert out.shape == (M, N)
        assert out.dtype == BF16
        assert not out.isnan().any()
        expected = torch.full((M, N), K * 0.5 * 0.5, dtype=BF16)
        torch.testing.assert_close(out, expected, atol=0.1, rtol=0.01)


class TestFP8Negative:
    """Invalid inputs: dtype guard, swapped quantize ops, K-dim mismatch — all must raise."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_non_fp8_mat1_dtype(self):
        """FP16 mat1 (unquantized) passed to _scaled_mm: rejected by lower_scaled_mm dtype guard."""
        M, K, N = 16, 128, 128
        a_fp16 = torch.rand(M, K, dtype=FP16, device=DEVICE)
        w_fp8 = _col_major(torch.rand(K, N, dtype=FP16).to(E4M3)).to(DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)

        def _wrong_mat1_dtype(a, w, sa, sw):
            return torch._scaled_mm(a, w, scale_a=sa, scale_b=sw, out_dtype=FP16)

        fn_c = _compile(_wrong_mat1_dtype)
        with pytest.raises((RuntimeError, ValueError)):
            fn_c(a_fp16, w_fp8, sa, sw)

    def test_swapped_quantize_ops(self):
        """Swapped quantize ops (qfp8wt on activation, qfp8ch on weight): DDL layout mismatch."""
        M, K, N = 16, 128, 128
        a, w = _make_rand_inputs(M, K, N, seed=53, device=DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)

        def _swapped_ops_fn(a, w, sa, sw):
            a_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(a, sa)
            w_fp8 = torch.ops.spyre.quantize_fp8_with_scale(w, sw)
            return torch._scaled_mm(
                a_fp8, w_fp8, scale_a=sa, scale_b=sw, out_dtype=FP16
            )

        fn_c = _compile(_swapped_ops_fn)
        with pytest.raises(RuntimeError):
            fn_c(a, w, sa, sw)

    def test_k_dimension_mismatch(self):
        """(M,K1)@(K2,N) with K1≠K2: rejected by _check_scaled_mm_sizes at PyTorch API layer."""
        M, K1, K2, N = 16, 128, 256, 128
        a_fp8 = torch.full((M, K1), 0.25, dtype=FP16).to(E4M3).to(DEVICE)
        w_fp8 = _col_major(torch.full((K2, N), 0.25, dtype=FP16).to(E4M3)).to(DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)

        fn_c = _compile(
            lambda a, b, sa, sb: torch._scaled_mm(a, b, sa, sb, out_dtype=FP16)
        )
        with pytest.raises((RuntimeError, ValueError)):
            fn_c(a_fp8, w_fp8, sa, sw)


class TestFP8EagerMode:
    """FP8 pipeline stages in eager mode (no torch.compile): shape, dtype, and value checks."""

    pytestmark = skip_no_quantize_ops

    def setup_method(self):
        torch._dynamo.reset()

    def test_compiled_quantize_then_eager_scaled_mm(self):
        """Quantize via compiled path, _scaled_mm in eager: validates against CPU _pipeline_ref."""
        M, K, N = 16, 128, 128
        a, w = _make_rand_inputs(M, K, N, seed=52, device=DEVICE)
        sa_val, sw_val = 1.0, 1.0
        sa = _st16(sa_val, DEVICE)
        sw = _st16(sw_val, DEVICE)

        def _quantize_only(a, w, sa, sw):
            return (
                torch.ops.spyre.quantize_fp8_with_scale(a, sa),
                torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw),
            )

        a_fp8, w_fp8 = _compile(_quantize_only)(a, w, sa, sw)

        out = torch._scaled_mm(a_fp8, w_fp8, scale_a=sa, scale_b=sw, out_dtype=FP16)

        ref = _pipeline_ref(a, w, sa_val, sw_val, FP16)
        torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.05)
