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

import pytest
import torch

from utils_inductor import compare_with_cpu


def _compare_modes(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def _tol(dtype):
    return (1e-3, 1e-2) if dtype == torch.float16 else (1e-4, 1e-3)


def _sdp(q, k, v):
    """Scaled dot-product attention from raw matmul + softmax."""
    w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / q.shape[-1] ** 0.5, dim=-1)
    return torch.matmul(w, v)


def _mm_overflow_check(x, y, scale):
    """mm wrapper that asserts inf propagation for overflow-scale fp16 inputs."""
    out = torch.mm(x, y)
    assert out.isinf().any(), (
        f"Expected inf for scale={scale} in fp16 mm, got max={out.abs().max()}"
    )
    return out


def _two_bmm(a, b, c):
    """Two chained bmms — triggers HBM stickify/destickify round-trip on Spyre."""
    return torch.bmm(torch.bmm(a, b), c)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestMatmulOps:
    # ── Scenario 1 — Degenerate 1×1 [NEW] ────────────────────────────────────
    # Not in upstream op_db for custom backends. Tests Spyre single-element
    # tiling path.
    @pytest.mark.parametrize(
        "fn,a,b",
        [
            (
                torch.mm,
                torch.tensor([[3.0]], dtype=torch.float16),
                torch.tensor([[4.0]], dtype=torch.float16),
            ),
            (
                torch.bmm,
                torch.tensor([[[3.0]]], dtype=torch.float16),
                torch.tensor([[[4.0]]], dtype=torch.float16),
            ),
            (
                torch.matmul,
                torch.tensor([[5.0]], dtype=torch.float16),
                torch.tensor([[6.0]], dtype=torch.float16),
            ),
        ],
        ids=["mm", "bmm", "matmul"],
    )
    def test_one_by_one(self, execution_mode, fn, a, b):
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, fn, a, b, atol=atol, rtol=rtol)

    # ── Scenario 2 — Prime batch sizes [NEW] ─────────────────────────────────
    # Upstream never targets prime values 7, 13. Stresses Spyre partial-tile
    # remainder logic for batches that don't divide evenly into tile width.
    @pytest.mark.parametrize("fn", [torch.bmm, torch.matmul], ids=["bmm", "matmul"])
    @pytest.mark.parametrize("batch", [1, 7, 13])
    def test_prime_batch(self, execution_mode, fn, batch):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1793
        pytest.xfail(
            reason="fp16 bmm/matmul prime batch sizes produce wrong results on Spyre: "
            "spyre <-> cpu mismatch for batch=1,7,13 (Issue #1793)"
        )
        a, b = (
            torch.randn(batch, 16, 16, dtype=torch.float16),
            torch.randn(batch, 16, 16, dtype=torch.float16),
        )
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, fn, a, b, atol=atol, rtol=rtol)

    # ── Scenario 3 — Identity matrix correctness [NEW] ───────────────────────
    # Upstream has this for CPU/CUDA but not via compare_with_cpu on a custom
    # backend. Tests Spyre tile engine with non-uniform stride inputs.
    @pytest.mark.parametrize(
        "fn,left,batched",
        [
            (torch.mm, False, False),
            (torch.mm, True, False),
            (torch.bmm, False, True),
        ],
        ids=["mm_right", "mm_left", "bmm_batched"],
    )
    def test_identity(self, execution_mode, fn, left, batched):
        torch.manual_seed(0)
        a = (
            torch.randn(4, 8, 8, dtype=torch.float16)
            if batched
            else torch.randn(8, 8, dtype=torch.float16)
        )
        eye = (
            torch.eye(8, dtype=torch.float16)
            .unsqueeze(0)
            .expand(4, -1, -1)
            .contiguous()
            if batched
            else torch.eye(8, dtype=torch.float16)
        )
        atol, rtol = _tol(torch.float16)
        _compare_modes(
            execution_mode, fn, *(eye, a) if left else (a, eye), atol=atol, rtol=rtol
        )

    # ── Scenario 4 — Attention mechanisms [NEW] ──────────────────────────────
    # Upstream only tests fused F.sdpa. This manual matmul+softmax+matmul
    # decomposition forces two separate bmm dispatches through Spyre, testing
    # on-device buffer handoff and intermediate tensor lifetime management.
    @pytest.mark.parametrize("variant", ["self", "cross", "multi_head", "causal_mask"])
    def test_attention_fp16(self, execution_mode, variant):
        torch.manual_seed(0)
        B, S, D = 2, 8, 16
        dtype = torch.float16

        if variant in ("self", "cross", "multi_head") and execution_mode == "compiled":
            pytest.xfail(
                reason="dxp_standalone SIGABRT: no valid scheduling candidate "
                "for fused fp16 bmm+transpose kernel"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1796
        if variant == "multi_head" and execution_mode == "eager":
            pytest.xfail(
                reason="In-device copy not implemented: "
                "tensor.contiguous() after transpose not supported on Spyre"
            )
        if variant in ("self", "cross", "causal_mask") and execution_mode == "eager":
            pytest.xfail(
                reason="aten::_reshape_alias not registered on Spyre backend "
                "(required by torch.softmax)"
            )
        if variant == "causal_mask" and execution_mode == "compiled":
            pytest.xfail(
                reason="Spyre backend does not support: "
                "to_dtype on DataFormats.IEEE_FP32 (fp32 mask -> fp16)"
            )

        if variant in ("self", "cross"):
            Q = torch.randn(B, 4 if variant == "cross" else S, D, dtype=dtype)
            K = V = torch.randn(B, S, D, dtype=dtype)
            _compare_modes(execution_mode, _sdp, Q, K, V, atol=1e-1, rtol=1e-1)

        elif variant == "multi_head":
            H, dk = 4, D // 4
            x = torch.randn(B, S, D, dtype=dtype)
            Q = K = V = x.view(B, S, H, dk).transpose(1, 2)

            def fn(q, k, v):
                return _sdp(q, k, v).transpose(1, 2).contiguous().view(B, S, D)

            _compare_modes(execution_mode, fn, Q, K, V, atol=1e-1, rtol=1e-1)

        else:  # causal_mask
            x = torch.randn(B, S, D, dtype=dtype)
            mask = torch.tril(torch.ones(S, S)).masked_fill(
                torch.tril(torch.ones(S, S)) == 0, float("-inf")
            )

            def fn(q, k, v, m):
                s = torch.matmul(q, k.transpose(-2, -1)) / q.shape[-1] ** 0.5
                return torch.matmul(
                    torch.softmax(s + m.to(s.dtype).unsqueeze(0), dim=-1), v
                )

            _compare_modes(execution_mode, fn, x, x, x, mask, atol=1e-1, rtol=1e-1)

    # ── Scenario 5 — Tile-boundary M/N sizes [NEW, fp16] ─────────────────────
    # Spyre tiles at fixed width (believed 16 or 32). Sizes exactly at a tile
    # boundary use full-tile micro-code; one above forces a partial-tile
    # have no fixed tile size exposed at test level.
    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 16, 16),  # exactly one tile — clean path
            (17, 16, 16),  # M one above tile boundary — partial remainder
            (16, 17, 16),  # N one above tile boundary — partial remainder
            (32, 32, 32),  # two tiles exactly
            (33, 32, 32),  # M partial-tile at two-tile boundary
            (15, 16, 16),  # M one below tile boundary
            (16, 15, 16),  # N one below tile boundary
        ],
        ids=["exact_16", "M_17", "N_17", "exact_32", "M_33", "M_15", "N_15"],
    )
    def test_tile_boundary_mn_fp16(self, execution_mode, M, N, K):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1793
        pytest.xfail(
            reason="fp16 mm tile-boundary shapes produce wrong results on Spyre: "
            "spyre <-> cpu mismatch across all M/N boundary variants (Issue #1793)"
        )
        a = torch.randn(M, K, dtype=torch.float16)
        b = torch.randn(K, N, dtype=torch.float16)
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, torch.mm, a, b, atol=atol, rtol=rtol)

    # ── Scenario 6 — align_tensors() K-dim padding mm [NEW, fp16] ────────────
    # Spyre's align_tensors() pads K to an alignment boundary before dispatch.
    # Non-aligned K appends ghost zero-columns that can corrupt the fp16
    # accumulator if the kernel reads past the logical K — root cause of
    # Issue #984. Upstream never probes odd K values for custom backends.
    @pytest.mark.parametrize(
        "K",
        [1, 3, 5, 7, 9, 11, 13, 15],
        ids=["K1", "K3", "K5", "K7", "K9", "K11", "K13", "K15"],
    )
    def test_align_tensors_k_padding_fp16(self, execution_mode, K):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/984
        pytest.xfail(
            reason="align_tensors() pads K to alignment boundary; padded zeros "
            "may corrupt fp16 accumulator for non-aligned K (Issue #984)"
        )
        a = torch.randn(8, K, dtype=torch.float16)
        b = torch.randn(K, 8, dtype=torch.float16)
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, torch.mm, a, b, atol=atol, rtol=rtol)

    # ── Scenario 7 — align_tensors() K-dim padding bmm [NEW, fp16] ───────────
    # Same K-alignment issue through the batched path. Each batch slice shares
    # the same padded K so a wrong-accumulator bug appears identically across
    # all entries — easier to detect statistically than in a single mm.
    @pytest.mark.parametrize(
        "B,K",
        [
            (4, 3),
            (4, 7),
            (8, 5),
            (8, 13),
        ],
        ids=["B4_K3", "B4_K7", "B8_K5", "B8_K13"],
    )
    def test_align_tensors_bmm_k_padding_fp16(self, execution_mode, B, K):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/984
        pytest.xfail(
            reason="align_tensors() K-dim padding may corrupt bmm fp16 accumulator "
            "across all batch slices (Issue #984)"
        )
        a = torch.randn(B, 8, K, dtype=torch.float16)
        b = torch.randn(B, K, 8, dtype=torch.float16)
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, torch.bmm, a, b, atol=atol, rtol=rtol)

    # ── Scenario 8 — fp16 overflow boundary [NEW, fp16] ──────────────────────
    # fp16 max finite value is ~65504. Upstream test_mm_scalings tests large
    # scale for float32 only — float32 handles 1e10 fine. For fp16, scale=1e5
    # must produce inf. Tests whether Spyre tile engine propagates inf
    # consistently with CPU or silently clamps to a wrong finite value.
    @pytest.mark.parametrize(
        "scale,expect_inf",
        [
            (1e2, False),  # well within fp16 range
            (1e4, False),  # near fp16 max (~65504)
            (1e5, True),  # overflows fp16 — result must be inf
        ],
        ids=["scale_1e2", "scale_1e4", "scale_1e5_inf"],
    )
    def test_mm_fp16_overflow(self, execution_mode, scale, expect_inf):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1793
        # All three scale cases are broken on Spyre (verified on hardware):
        # - scale_1e2, scale_1e4: Spyre returns nan instead of finite values
        # - scale_1e5: Spyre returns nan instead of inf
        pytest.xfail(
            reason="fp16 mm produces wrong results on Spyre across all scales: "
            "nan instead of finite/inf values (Issue #1793)"
        )
        a = torch.randn(8, 16, dtype=torch.float16) * scale
        b = torch.randn(16, 8, dtype=torch.float16) * scale
        atol, rtol = _tol(torch.float16)
        if expect_inf:
            _compare_modes(
                execution_mode,
                lambda x, y: _mm_overflow_check(x, y, scale),
                a,
                b,
                atol=atol,
                rtol=rtol,
            )
        else:
            _compare_modes(execution_mode, torch.mm, a, b, atol=atol, rtol=rtol)

    # ── Scenario 9 — HBM stickify / destickify round-trip [NEW, fp16] ────────
    # Spyre stores activations in HBM in stickified layout. A second bmm
    # consuming the output of a first must destickify it via ReStickifyOpHBM.
    # Shapes exceeding on-chip SRAM force an HBM spill between the two ops —
    # a round-trip unique to Spyre not exercised by any upstream test.
    @pytest.mark.parametrize(
        "B,M,K,N",
        [
            (2, 128, 128, 128),  # fits in HBM — baseline
            (2, 256, 256, 256),  # forces HBM spill between ops
            (2, 512, 64, 64),  # wide batch, moderate matrix
        ],
        ids=["hbm_128", "hbm_256", "hbm_512_wide"],
    )
    def test_hbm_stickify_roundtrip_fp16(self, execution_mode, B, M, K, N):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1793
        pytest.xfail(
            reason="fp16 chained bmm produces wrong results on Spyre: "
            "spyre <-> cpu mismatch across all HBM shapes (Issue #1793)"
        )
        a = torch.randn(B, M, K, dtype=torch.float16)
        b = torch.randn(B, K, N, dtype=torch.float16)
        c = torch.randn(B, N, M, dtype=torch.float16)
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, _two_bmm, a, b, c, atol=atol, rtol=rtol)

    # ── Scenario 10 — SDP non-square sequence lengths [NEW, fp16] ────────────
    # Upstream F.sdpa tests use S_q == S_k only. Cross-attention with S_q !=
    # S_k produces a non-square weight matrix, changing the tile layout for
    # the second bmm. Prime values (7, 13) additionally stress partial-tile
    # logic in both dims simultaneously — not covered anywhere upstream.
    @pytest.mark.parametrize(
        "S_q,S_k",
        [
            (8, 16),  # S_q < S_k — tall weight matrix
            (16, 8),  # S_q > S_k — wide weight matrix
            (7, 13),  # both prime — partial tile in both dims
        ],
        ids=["sq_lt_sk", "sq_gt_sk", "both_prime"],
    )
    def test_sdp_nonsquare_fp16(self, execution_mode, S_q, S_k):
        torch.manual_seed(0)
        B, D = 2, 16
        atol, rtol = _tol(torch.float16)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1796
        pytest.xfail(
            reason="Spyre backend does not support: mutation op buf7 — "
            "cannot restickify non-square attention weight tensor "
            "across S_q != S_k shapes (Issue #1796)"
        )
        Q = torch.randn(B, S_q, D, dtype=torch.float16)
        K = torch.randn(B, S_k, D, dtype=torch.float16)
        V = torch.randn(B, S_k, D, dtype=torch.float16)
        _compare_modes(execution_mode, _sdp, Q, K, V, atol=atol, rtol=rtol)

    # ── Scenario 11 — baddbmm FixedTiledLayout output reuse [NEW, fp16] ──────
    # Issue #1795 shows addmm alpha=0 crashes on FixedLayout vs FixedTiledLayout
    # mismatch. baddbmm hits the same layout inference path through a different
    # op — not in upstream for custom backends.
    @pytest.mark.parametrize(
        "beta,alpha",
        [
            (1.0, 1.0),
            (0.5, 1.0),
        ],
        ids=["beta1_alpha1", "beta0p5_alpha1"],
    )
    def test_baddbmm_layout_fp16(self, execution_mode, beta, alpha):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1795
        pytest.xfail(
            reason="baddbmm fp16 hits same FixedTiledLayout vs FixedLayout "
            "mismatch as addmm (Issue #1795)"
        )
        bias = torch.randn(4, 8, 8, dtype=torch.float16)
        b1 = torch.randn(4, 8, 16, dtype=torch.float16)
        b2 = torch.randn(4, 16, 8, dtype=torch.float16)
        atol, rtol = _tol(torch.float16)
        _compare_modes(
            execution_mode,
            lambda bias, x, y: torch.baddbmm(bias, x, y, beta=beta, alpha=alpha),
            bias,
            b1,
            b2,
            atol=atol,
            rtol=rtol,
        )
