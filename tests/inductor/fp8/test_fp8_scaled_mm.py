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
from torch._inductor.exc import InductorError
from torch_spyre.constants import DEVICE_NAME
from utils_inductor import cached_randn, compare_with_pytorch, expects_raise

E4M3 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(E4M3).max)
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


pytestmark = pytest.mark.skipif(
    not _spyre_available(),
    reason="Spyre device not available",
)

# randn outputs reach ~45 and the device quantizes ~3% of values one FP8 step off the
# CPU oracle, so a fixed atol=0.1 is too tight. Observed error is <=2.7% of the output
# absmax; 5% leaves margin and still catches a dropped scale (off by 1/(sa*sw)).
_SCALED_ATOL_FRAC = 0.05


def _assert_close_scaled(out: torch.Tensor, ref: torch.Tensor) -> None:
    """assert_close with atol = 5% of the reference's absmax and rtol = 5%."""
    atol = _SCALED_ATOL_FRAC * float(ref.float().abs().max())
    torch.testing.assert_close(out, ref, atol=atol, rtol=0.05)


_XFAIL_4308 = pytest.mark.xfail(
    strict=True,
    raises=InductorError,
    reason="https://github.com/torch-spyre/torch-spyre/issues/4308: K%128!=0 fails to "
    "compile: padding K leaves a STANDARD weight, so the FP8 matmul cost model finds no "
    "QFP8WT input (StopIteration); past that, dbo-opt cannot zero-fill FP8 padding",
)

_XFAIL_4310 = pytest.mark.xfail(
    strict=True,
    raises=InductorError,
    reason="https://github.com/torch-spyre/torch-spyre/issues/4310: a raw FP8 weight "
    "moved with .to(device) is STANDARD, not QFP8WT, so the FP8 matmul cost model "
    "raises StopIteration at compile",
)

_XFAIL_4311 = pytest.mark.xfail(
    strict=True,
    raises=InductorError,
    reason="https://github.com/torch-spyre/torch-spyre/issues/4311: non-contiguous "
    "activation fails to compile: 'batchmatmulfp8: cannot restickify any input layout'",
)


def _amax_scale(t: torch.Tensor) -> float:
    """Amax-based scale: amax(|t|)/448 — maps max→FP8 max."""
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
    """CPU oracle: (clip(a/sa)@clip(w/sw)) * sa*sw [+bias], all intermediate in FP32.

    Tests compare with rtol=0.05, under half an E4M3 step (6.25%), and an atol of one to
    three FP16 ULPs at the output size for rand inputs, K/4 (atol=0.1 at K=128, 1.0 at K=4096).
    """
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

    @_XFAIL_4311
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
        _assert_close_scaled(out, ref)


class TestFP8CanonicalShapes:
    """Canonical pipeline shapes from test_fp8_scaled_mm_cpu."""

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
        _assert_close_scaled(out, ref)

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
        _assert_close_scaled(out, ref)

    def test_amax_scale_fp16_pipeline(self):
        """Amax scale (max→448): the unscaled K-term FP8 product exceeds FP16 max, so the
        output is finite only if the scales are applied before the FP16 store."""
        M, K, N = 16, 128, 128
        torch.manual_seed(7)
        a = torch.randn(M, K, dtype=FP16)
        w = torch.randn(K, N, dtype=FP16)
        sa_val = _amax_scale(a)
        sw_val = _amax_scale(w)
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
        _assert_close_scaled(out, ref)

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

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.parametrize(
        "M,K,N,sa_val,sw_val,seed,label",
        [
            (16, 128, 128, 1.0, 1.0, 40, "base"),
            (8, 128, 1024, 1.0, 1.0, 41, "wide_n"),
            (32, 256, 256, 1.0, 1.0, 42, "k256"),
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

    def setup_method(self):
        torch._dynamo.reset()

    @_XFAIL_4308
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

    @_XFAIL_4308
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

    @_XFAIL_4308
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
    @_XFAIL_4308
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
    @_XFAIL_4308
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

    pytestmark = pytest.mark.slow

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
            32,
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

    @_XFAIL_4311
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
        _assert_close_scaled(out, ref)


class TestFP8CompileIntegrity:
    """Compilation properties: fullgraph (no graph break), bit-identical repeated calls, fresh-recompile."""

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

    @_XFAIL_4310
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

    @pytest.mark.xfail(
        strict=True,
        raises=InductorError,
        reason="https://github.com/torch-spyre/torch-spyre/issues/4310: raw FP8 weight "
        "has no QFP8WT input, so the FP8 matmul cost model raises StopIteration; K=64 "
        "then also hits https://github.com/torch-spyre/torch-spyre/issues/4308",
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

    @_XFAIL_4310
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
    """Invalid inputs: dtype guard, swapped quantize ops — all must raise."""

    def setup_method(self):
        torch._dynamo.reset()

    @expects_raise
    def test_non_fp8_mat1_dtype(self):
        """FP16 mat1 (unquantized) passed to _scaled_mm: rejected by PyTorch's dtype check."""
        M, K, N = 16, 128, 128
        a_fp16 = torch.rand(M, K, dtype=FP16, device=DEVICE)
        w_fp8 = _col_major(torch.rand(K, N, dtype=FP16).to(E4M3)).to(DEVICE)
        sa = _st16(1.0, DEVICE)
        sw = _st16(1.0, DEVICE)

        def _wrong_mat1_dtype(a, w, sa, sw):
            return torch._scaled_mm(a, w, scale_a=sa, scale_b=sw, out_dtype=FP16)

        fn_c = _compile(_wrong_mat1_dtype)
        with pytest.raises(RuntimeError, match="Expected both inputs to be fp8"):
            fn_c(a_fp16, w_fp8, sa, sw)

    @expects_raise
    def test_swapped_quantize_ops(self):
        """Swapped quantize ops (qfp8wt on activation, qfp8ch on weight): rejected at compile."""
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
        with pytest.raises(
            InductorError, match="the matmul reduces over its stick dim"
        ):
            fn_c(a, w, sa, sw)


class TestFP8EagerMode:
    """FP8 pipeline stages in eager mode (no torch.compile): shape, dtype, and value checks."""

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


class TestFP8ScaledMMRegressions:
    """Model-shape and graph-pattern regressions for the FP8 ``_scaled_mm`` path.

    Moved from ``test_inductor_ops.py``. Each test compiles the Spyre pipeline and
    compares it with an uncompiled CPU reference via ``compare_with_pytorch``.
    """

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (1, 4096, 4096),
            (2, 4096, 4096),
            (1, 4096, 6144),
            (1, 4096, 25600),
            (4, 4096, 4096),
            (8, 4096, 4096),
            (16, 4096, 4096),
            (64, 4096, 4096),
            (8, 4096, 1024),
            (8, 4096, 12800),
        ],
        ids=[
            "decode_m1_n4096_baseline",
            "decode_m2_n4096_baseline",
            "fused_qkv_n6144",
            "fused_gate_up_n25600",
            "wide_m4_n4096",
            "wide_m8_n4096",
            "prefill_warmup_m16_n4096",
            "bench_prefill_m64_n4096",
            "m8_n1024",
            "m8_n12800",
        ],
    )
    def test_granite_fp8_shapes(self, m, k, n):
        """Regression test for SuperDSC aborts on FP8 _scaled_mm at Granite 8B shapes.

        Covers fused QKV (N=6144), fused gate_up (N=25600), and wide-M prefill
        shapes that previously caused compilation failures (issue #4179).

        Two independent fixes are exercised here:
          Fix 1 (compute_ops.py): corrects the out/N-dim elemArr constants
            emitted by gen_coord_info_value for the QFP8WT KERNEL weight.
          Fix 2 (work_division.py): routes batchmatmulfp8 through the analytic
            cost model (_cost_model_divide_op), which was previously skipped for
            FP8 BMM ops. The cost model now produces a correct split for all N.
        """
        x = cached_randn((m, k), dtype=FP16, differentiation=("x", m, k, n), scale=0.5)
        w = cached_randn((k, n), dtype=FP16, differentiation=("w", m, k, n), scale=0.5)
        scale_a = torch.tensor([0.1], dtype=FP16)
        scale_b = torch.tensor([0.1], dtype=FP16)

        def spyre_fn(x, w, scale_a, scale_b):
            xa = torch.ops.spyre.quantize_fp8_with_scale(x, scale_a)
            wb = torch.ops.spyre.quantize_weight_fp8_with_scale(w, scale_b)
            return torch.ops.aten._scaled_mm(
                xa, wb, scale_a=scale_a, scale_b=scale_b, out_dtype=FP16
            )

        def pytorch_fn(x, w, scale_a, scale_b):
            xa = (x / scale_a).clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
            wb = (w / scale_b).clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
            return (xa @ wb) * (scale_a * scale_b)

        compare_with_pytorch(
            spyre_fn, pytorch_fn, x, w, scale_a, scale_b, atol=2.0, rtol=0.2
        )

    @pytest.mark.parametrize(
        "m,k,n_hidden,n_out",
        [
            (2, 4096, 1024, 4096),
            (1, 4096, 1024, 4096),
            (2, 4096, 4096, 4096),
        ],
        ids=[
            "granite_m2_k4096_n1024_n4096",
            "granite_m1_k4096_n1024_n4096",
            "granite_m2_k4096_n4096_n4096",
        ],
    )
    def test_chained_scaled_mm(self, m, k, n_hidden, n_out):
        """Regression test for chained FP8 matmul (MLP / decoder block pattern).

        Two fp8_linear calls where the output of the first feeds the input of
        the second -- the MLP pattern in a Granite decoder block. Per-row
        dynamic x_scale computed from the activation; external per-column
        w_scale passed in.

        Previously failed with:
          NotImplementedError: no mechanism to resolve stick incompatibility

        Root cause: three combined issues unblocked by this fix:
          1. pass_utils.py: compute_restickify_needed blocked SEN143_FP8 (#4238)
          2. spyre_kernel.py: RESTICKIFY_OP gate excluded SEN143_FP8
          3. propagate_layouts.py: generate_stl_with_required_stick returned
             sparse QFP8CH without checking reduction_var was on the stick
        """
        scale_eps = 1e-4

        def fp8_linear(x, w, w_scale):
            x_scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
                min=scale_eps
            )
            wq = torch.ops.spyre.quantize_weight_fp8_with_scale(w, w_scale)
            xq = torch.ops.spyre.quantize_fp8_with_scale(x, x_scale)
            y = torch.ops.spyre.scaled_mm(
                xq.reshape(-1, xq.shape[-1]), wq, out_dtype=FP16
            )
            return (y.reshape(*x.shape[:-1], w.shape[-1]) * x_scale * w_scale).to(
                x.dtype
            )

        x = cached_randn(
            (1, m, k),
            dtype=FP16,
            differentiation=("x", m, k, n_hidden, n_out),
            scale=0.1,
        )
        g = cached_randn(
            (k,),
            dtype=FP16,
            differentiation=("g", m, k, n_hidden, n_out),
            scale=0.1,
        )
        w1 = cached_randn(
            (k, n_hidden),
            dtype=FP16,
            differentiation=("w1", m, k, n_hidden, n_out),
            scale=0.1,
        )
        w2 = cached_randn(
            (n_hidden, n_out),
            dtype=FP16,
            differentiation=("w2", m, k, n_hidden, n_out),
            scale=0.1,
        )
        ws1 = torch.full((n_hidden,), 0.1, dtype=FP16)
        ws2 = torch.full((n_out,), 0.1, dtype=FP16)

        def spyre_fn(x, g, w1, ws1, w2, ws2):
            h = x * g
            return fp8_linear(fp8_linear(h, w1, ws1), w2, ws2)

        def pytorch_fn(x, g, w1, ws1, w2, ws2):
            def ref_linear(x, w, ws):
                x_scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
                    min=scale_eps
                )
                xq = (x / x_scale).clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
                wq = (w / ws).clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
                y = (xq.reshape(-1, xq.shape[-1]) @ wq) * (x_scale * ws)
                return y.reshape(*x.shape[:-1], w.shape[-1]).to(x.dtype)

            h = x * g
            return ref_linear(ref_linear(h, w1, ws1), w2, ws2)

        compare_with_pytorch(
            spyre_fn, pytorch_fn, x, g, w1, ws1, w2, ws2, atol=2.0, rtol=0.2
        )

    @pytest.mark.parametrize(
        "b,m,k,n",
        [
            (2, 8, 128, 128),
            (2, 4, 256, 128),
            (4, 2, 128, 256),
        ],
        ids=["b2_m8_k128_n128", "b2_m4_k256_n128", "b4_m2_k128_n256"],
    )
    def test_3d_activation_reshape(self, b, m, k, n):
        """Regression test for _project_pointwise_dim_order rank_diff < 0 with QFP8CH.

        A 3D batched activation (B, M, K) is reshaped to 2D (B*M, K) before
        quantization and the matmul output is reshaped back to (B, M, N).
        The QFP8CH rank-2 buffer produced by quantize_fp8_with_scale is then
        consumed by rank-3 pointwise rescaling ops, giving rank_diff = 2 - 3 = -1
        in _project_pointwise_dim_order. The sparse-stick -1 marker was incorrectly
        shifted as if it were a dimension index, producing a dim_order of the wrong
        length and crashing SpyreTensorLayout.__init__.

        Previously failed with:
          RuntimeError: Incompatible host_size and dim_order

        Root cause:
          propagate_layouts.py: _project_pointwise_dim_order shifted the trailing
          -1 sparse-stick marker when rank_diff < 0, corrupting dim_order length
        """
        a_3d = cached_randn(
            (b, m, k), dtype=FP16, differentiation=("a3d", b, m, k, n), scale=0.1
        )
        w = cached_randn(
            (k, n), dtype=FP16, differentiation=("w", b, m, k, n), scale=0.1
        )
        # Non-unit scales: scale=1.0 makes quantization a numerical pass-through
        # and masks range-boundary errors.
        sa = torch.full((1,), 0.5, dtype=FP16)
        sw = torch.full((1,), 0.25, dtype=FP16)

        def spyre_fn(a_3d, w, sa, sw):
            a_2d = a_3d.reshape(a_3d.shape[0] * a_3d.shape[1], a_3d.shape[2])
            a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a_2d, sa)
            w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)
            out = torch.ops.aten._scaled_mm(
                a_fp8, w_fp8, scale_a=sa, scale_b=sw, out_dtype=FP16
            )
            return out.reshape(a_3d.shape[0], a_3d.shape[1], n)

        def pytorch_fn(a_3d, w, sa, sw):
            a_2d = a_3d.reshape(a_3d.shape[0] * a_3d.shape[1], a_3d.shape[2])
            a_fp8 = (a_2d / sa).clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
            w_fp8 = (w / sw).clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
            out = (a_fp8 @ w_fp8) * (sa * sw)
            return out.reshape(a_3d.shape[0], a_3d.shape[1], n)

        compare_with_pytorch(spyre_fn, pytorch_fn, a_3d, w, sa, sw, atol=2.0, rtol=0.2)
