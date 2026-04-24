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

import torch
import math
import pytest

from utils_inductor import cached_randn, compare_with_cpu


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestTensorScalarCoreArithmetic:
    """
    Core tensor–scalar arithmetic on Spyre vs CPU
    """

    torch.manual_seed(0xAFFE)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1545
    @pytest.mark.xfail(
        reason="Spyre backend does not support dtype ComplexFloat (torch.complex64)"
    )
    def test_complex64_add_mul_div_python_complex_scalars(self, execution_mode):
        """
        ``complex64`` tensor with Python ``complex`` scalars: ``+ (1+2j)``, ``* (3-4j)``, ``/ (0.5+0.5j)``.
        """

        def complex_ops(x):
            # Complex scalar addition
            y1 = x + (1 + 2j)
            # Complex scalar multiplication
            y2 = y1 * (3 - 4j)
            # Complex scalar division
            y3 = y2 / (0.5 + 0.5j)
            return y3

        x = torch.randn(10, 10, dtype=torch.complex64)
        compare_with_cpu(
            complex_ops,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
    @pytest.mark.xfail(reason="Spyre backend does not support dtype Double (FP64)")
    def test_fp16_fp32_tensors_fp32_literal_and_fp64_scalar_tensor(
        self, execution_mode
    ):
        """
        ``fp16 * 2.5`` (Python scalar) and ``fp32 + tensor(1.5, float64)``; tuple return.
        """

        def mixed_precision_ops(x_fp16, x_fp32):
            # FP16 tensor with FP32 scalar
            y1 = x_fp16 * 2.5  # scalar is FP32 by default
            # FP32 tensor with explicit FP64 scalar
            y2 = x_fp32 + torch.tensor(1.5, dtype=torch.float64, device=x_fp16.device)
            return y1.float(), y2

        x_fp16 = cached_randn((10, 10), dtype=torch.float16)
        x_fp32 = cached_randn((10, 10), differentiation=1, dtype=torch.float32)
        compare_with_cpu(
            mixed_precision_ops,
            x_fp16,
            x_fp32,
            atol=0.01,
            rtol=0.01,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    # TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1172
    @pytest.mark.xfail(reason="Support 0-dim tensors in Spyre")
    def test_scalar_add_mul_sub_zero_one_and_ndim_tensors(self, execution_mode):
        """
        One scalar Python op per rank: 0-D ``+1``, 1-D ``*2.5``, 3-D ``-3``.
        """

        def broadcasting_ops(x0, x1, x2):
            # 0-dim tensor with scalar
            y0 = x0 + 1.0
            # 1-dim tensor with scalar
            y1 = x1 * 2.5
            # Multi-dim tensor with scalar
            y2 = x2 - 3.0
            return y0, y1, y2

        x0 = torch.tensor(5.0)  # 0-dim
        x1 = cached_randn((10))  # 1-dim
        x2 = cached_randn((10, 10, 10))  # 3-dim
        compare_with_cpu(
            broadcasting_ops,
            x0,
            x1,
            x2,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_inplace_add_mul_div_scalar_after_clone(self, execution_mode):
        """
        ``clone()`` then ``add_(1)``, ``mul_(2)``, ``div_(0.5)`` (in-place scalar chain).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1558
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: 'RuntimeError: In-device copy (.clone()) not implemented'"
            )

        def inplace_ops(x):
            y = x.clone()
            y.add_(1.0)
            y.mul_(2.0)
            y.div_(0.5)
            return y

        x = cached_randn((10, 10))
        compare_with_cpu(
            inplace_ops,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_parenthesized_chain_add_mul_sub_div_scalars(self, execution_mode):
        """
        ``((x + 1.5) * 2.0 - 0.5) / 3.0`` — single expression, Python float constants.
        """

        def chained_ops(x):
            return ((x + 1.5) * 2.0 - 0.5) / 3.0

        x = cached_randn((10, 10))
        compare_with_cpu(
            chained_ops,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestSdpaScalarHyperparameters:
    """
    Scaled dot-product attention (SDPA) variants on Spyre: scalar hyperparameters
    (mask fill value, dropout ``p``, temperature divisor, additive score bias, window size)
    and mask patterns (full, local, causal).
    """

    torch.manual_seed(0xAFFE)

    # Small shapes (B,H,S,D) keep memory down and reduce backend-specific FP16 overflow issues.
    _MASK_SMALL = dict(b=1, h=2, s=32, d=16)

    def test_sdpa_masked_fill_scalar_constant_finite_neg_fp16(self, execution_mode):
        """
        Causal SDPA: ``masked_fill`` with a negative **Python float** ``-1.0e4`` (in-range for FP16).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        elif execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )
        b, h, s, d = (
            self._MASK_SMALL["b"],
            self._MASK_SMALL["h"],
            self._MASK_SMALL["s"],
            self._MASK_SMALL["d"],
        )
        fill = -1.0e4

        def sdpa_finite_fill(q, k, v):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool)).to(q.device)
            allowed = causal_mask.view(1, 1, s, s).expand(b, h, s, s)
            scores = scores.masked_fill(~allowed, fill)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((b, h, s, d), dtype=torch.float16)
        k = cached_randn((b, h, s, d), dtype=torch.float16, differentiation=1)
        v = cached_randn((b, h, s, d), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            sdpa_finite_fill,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_masked_fill_zero_dim_tensor_scalar_fp16(self, execution_mode):
        """
        Causal mask with fill = **0-D tensor** ``torch.tensor(-1.0e4, dtype=float16)`` (tensor “scalar”).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        elif execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )
        b, h, s, d = (
            self._MASK_SMALL["b"],
            self._MASK_SMALL["h"],
            self._MASK_SMALL["s"],
            self._MASK_SMALL["d"],
        )

        def sdpa_tensor0d_fill(q, k, v, causal):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            allowed = causal.view(1, 1, s, s).expand(b, h, s, s)
            fill = torch.tensor(-1.0e4, dtype=torch.float16, device=scores.device)
            scores = scores.masked_fill(~allowed, fill)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((b, h, s, d), dtype=torch.float16)
        k = cached_randn((b, h, s, d), dtype=torch.float16, differentiation=1)
        v = cached_randn((b, h, s, d), dtype=torch.float16, differentiation=2)
        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool))

        compare_with_cpu(
            sdpa_tensor0d_fill,
            q,
            k,
            v,
            causal_mask,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_dropout_scalar_p(self, execution_mode):
        """SDPA with ``dropout(attn, p=0.1, training=False)`` — scalar dropout probability."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )

        def attention_with_dropout(q, k, v):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            attn = torch.softmax(scores, dim=-1)
            # Dropout with scalar rate
            attn = torch.nn.functional.dropout(attn, p=0.1, training=False)
            return torch.matmul(attn, v)

        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        k = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=1)
        v = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            attention_with_dropout,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_score_temperature_divisor_scalar(self, execution_mode):
        """SDPA: divide logits by ``(sqrt(d_k) * temperature)`` with scalar ``temperature=2.0``."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )

        def temperature_attention(q, k, v):
            d_k = q.size(-1)
            temperature = 2.0  # Temperature scalar
            scores = torch.matmul(q, k.transpose(-2, -1)) / (
                math.sqrt(d_k) * temperature
            )
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        k = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=1)
        v = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            temperature_attention,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_broadcast_kv_heads(self, execution_mode):
        """
        SDPA when K/V have fewer heads than Q: PyTorch **broadcasts** KV heads to Q heads.
        This is *not* a full GQA/MQA implementation (no explicit KV repeat); it exercises
        broadcast matmul + scalar ``1/sqrt(d_k)`` on Spyre.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1558
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: In-device copy not implemented (required for MQA operations)"
            )

        def sdpa_broadcast_kv(q, k, v):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        num_kv_heads: int = 1
        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)  # 8 query heads
        k = cached_randn(
            (2, num_kv_heads, 128, 64), dtype=torch.float16, differentiation=1
        )
        v = cached_randn(
            (2, num_kv_heads, 128, 64), dtype=torch.float16, differentiation=2
        )
        compare_with_cpu(
            sdpa_broadcast_kv,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_local_window_mask_scalar(self, execution_mode):
        """SDPA: local window via ``masked_fill(..., -inf)``; ``window_size=32`` sets mask support."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1482
        elif execution_mode == "compiled":
            pytest.xfail(
                reason="Spyre backend does not support type conversion yet during copy."
            )

        def local_attention(q, k, v):
            d_k = q.size(-1)
            window_size = 32  # Local window scalar
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            # Create local attention mask
            seq_len = scores.size(-1)
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
            for i in range(seq_len):
                start = max(0, i - window_size)
                end = min(seq_len, i + window_size + 1)
                mask[i, start:end] = False
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        k = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=1)
        v = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            local_attention,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_cross_sequence_length(self, execution_mode):
        """SDPA with Q seq len ≠ KV seq len (scalar ``1/sqrt(d_k)`` scaling only)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )

        def cross_attention(q, k, v):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        k = cached_randn((2, 8, 256, 64), dtype=torch.float16, differentiation=1)
        v = cached_randn((2, 8, 256, 64), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            cross_attention,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_self_attention_single_tensor(self, execution_mode):
        """SDPA: Q, K, V all from one tensor ``x`` (scalar ``1/sqrt(d_k)``)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )

        def self_attention(x):
            # Self attention: Q, K, V all from same source
            d_k = x.size(-1)
            scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(d_k)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, x)

        x = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        compare_with_cpu(
            self_attention,
            x,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_causal_triu_mask_neg_inf(self, execution_mode):
        """SDPA: causal mask via ``triu`` + ``masked_fill(..., -inf)``."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1334
        elif execution_mode == "compiled":
            pytest.xfail(
                reason="Constant tensor creation fails - IndexError on empty args during layout propagation."
            )

        def causal_attention(q, k, v):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            # Causal mask
            seq_len = scores.size(-1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        k = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=1)
        v = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            causal_attention,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_sdpa_additive_score_bias_scalar(self, execution_mode):
        """SDPA: add scalar ``0.1`` to all attention logits before softmax."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )

        def attention_with_bias(q, k, v):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            # Add scalar bias
            scores = scores + 0.1
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        q = cached_randn((2, 8, 128, 64), dtype=torch.float16)
        k = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=1)
        v = cached_randn((2, 8, 128, 64), dtype=torch.float16, differentiation=2)
        compare_with_cpu(
            attention_with_bias,
            q,
            k,
            v,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )
