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

import math

import pytest
import torch

from utils_inductor import cached_randn, compare_with_cpu


def _compare_modes(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestNormalizationScalarOperations:
    """
    Normalization-style graphs with explicit scalar ``eps`` / affine scales. Uses
    explicit mean/var decompositions (not ``nn.LayerNorm`` / ``nn.BatchNorm*`` modules)
    and identity running stats where noted. RMSNorm, GroupNorm, InstanceNorm patterns.
    Validates Spyre vs CPU.
    """

    torch.manual_seed(0xAFFE)

    @pytest.mark.parametrize(
        "eps,dtype,batch,seq,hidden",
        [
            (1e-5, torch.float32, 32, 512, 768),
            (1e-5, torch.float16, 128, 512, 768),
        ],
    )
    def test_layernorm(self, execution_mode, eps, dtype, batch, seq, hidden):
        """Last-dim normalization (mean/var), not ``nn.LayerNorm``; various ``eps`` and shapes."""

        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
        if dtype == torch.float32:
            pytest.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1688
        if dtype == torch.float16:
            pytest.xfail(
                reason="Variance (aten::var.correction) operation not implemented"
            )

        def layernorm(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + eps)

        x = cached_randn((batch, seq, hidden), dtype=dtype)
        tol = (1e-4, 1e-3) if dtype == torch.float32 else (1e-3, 1e-2)
        _compare_modes(execution_mode, layernorm, x, atol=tol[0], rtol=tol[1])

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
    @pytest.mark.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
    def test_layernorm_affine(self, execution_mode):
        """Last-dim layernorm with gamma/beta (affine)."""

        eps = 1e-5
        hidden_size = 768

        def layernorm_affine(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + eps)

            gamma = torch.ones(hidden_size, device=x.device)
            beta = torch.zeros(hidden_size, device=x.device)
            return gamma * normalized + beta

        x = cached_randn((32, 512, 768), dtype=torch.float32)
        _compare_modes(execution_mode, layernorm_affine, x, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize(
        "eps,dtype,batch,seq,hidden",
        [
            (1e-8, torch.float32, 128, 512, 768),
            (1e-6, torch.float16, 128, 512, 768),
            (1e-8, torch.float16, 32, 512, 768),
            (1e-6, torch.float16, 8, 2048, 768),
            (1e-5, torch.float16, 1, 1024, 4096),
            (1e-6, torch.float16, 1, 2048, 4096),
        ],
    )
    def test_rmsnorm(self, execution_mode, eps, dtype, batch, seq, hidden):
        """Test RMSNorm with various epsilon values and configurations."""

        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
        if dtype == torch.float32:
            pytest.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")

        def rmsnorm(x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            return x / rms

        x = cached_randn((batch, seq, hidden), dtype=dtype)
        tol = (1e-4, 1e-3) if dtype == torch.float32 else (1e-3, 1e-2)
        _compare_modes(execution_mode, rmsnorm, x, atol=tol[0], rtol=tol[1])

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
    @pytest.mark.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
    def test_rmsnorm_with_weight(self, execution_mode):
        """Test RMSNorm with learnable weight parameter."""

        eps = 1e-6
        hidden_size = 768

        def rmsnorm_with_weight(x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            normalized = x / rms
            weight = torch.ones(hidden_size, device=x.device)
            return weight * normalized

        x = cached_randn((32, 512, 768), dtype=torch.float32)
        _compare_modes(execution_mode, rmsnorm_with_weight, x, atol=1e-4, rtol=1e-3)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1377
    @pytest.mark.skip(
        reason="Spyre: Broadcasting size-1 dimensions - cannot map stick expr to host dimension"
    )
    @pytest.mark.parametrize(
        "dtype,num_channels",
        [
            (torch.float32, 64),
            (torch.float16, 64),
        ],
    )
    def test_batchnorm_identity_running_stats_2d(
        self, execution_mode, dtype, num_channels
    ):
        """2D inference-style norm with **identity** running mean/var (zeros/ones)."""

        eps = 1e-5

        def batchnorm_2d_inference(x):
            running_mean = torch.zeros(num_channels, device=x.device)
            running_var = torch.ones(num_channels, device=x.device)

            mean = running_mean.view(1, -1, 1, 1)
            var = running_var.view(1, -1, 1, 1)

            return (x - mean) / torch.sqrt(var + eps)

        x = cached_randn((32, num_channels, 224, 224), dtype=dtype)
        tol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-3, 1e-2)
        _compare_modes(
            execution_mode, batchnorm_2d_inference, x, atol=tol[0], rtol=tol[1]
        )

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1531
    @pytest.mark.xfail(
        reason="Square root operation on float32 (IEEE_FP32) not supported"
    )
    def test_batchnorm_identity_running_stats_1d(self, execution_mode):
        """1D inference-style norm with **identity** running mean/var."""

        eps = 1e-5
        num_features = 768

        def batchnorm_1d_inference(x):
            running_mean = torch.zeros(num_features, device=x.device)
            running_var = torch.ones(num_features, device=x.device)
            return (x - running_mean) / torch.sqrt(running_var + eps)

        x = cached_randn((32, 768), dtype=torch.float32)
        _compare_modes(execution_mode, batchnorm_1d_inference, x, atol=1e-4, rtol=1e-3)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1377
    @pytest.mark.xfail(
        reason="Spyre: Broadcasting size-1 dimensions - cannot map stick expr to host dimension"
    )
    def test_batchnorm_identity_affine_2d(self, execution_mode):
        """2D norm with identity running stats plus gamma/beta."""

        eps = 1e-5
        num_channels = 64

        def batchnorm_2d_affine(x):
            running_mean = torch.zeros(num_channels, device=x.device)
            running_var = torch.ones(num_channels, device=x.device)

            mean = running_mean.view(1, -1, 1, 1)
            var = running_var.view(1, -1, 1, 1)

            normalized = (x - mean) / torch.sqrt(var + eps)

            gamma = torch.ones(num_channels, device=x.device).view(1, -1, 1, 1)
            beta = torch.zeros(num_channels, device=x.device).view(1, -1, 1, 1)

            return gamma * normalized + beta

        x = cached_randn((32, 64, 224, 224), dtype=torch.float32)
        _compare_modes(execution_mode, batchnorm_2d_affine, x, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize(
        "num_groups,dtype",
        [
            (32, torch.float32),
            (32, torch.float16),
            (8, torch.float32),
            (8, torch.float16),
        ],
    )
    def test_groupnorm(self, execution_mode, num_groups, dtype):
        """Test GroupNorm with various group counts and dtypes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1722
        if dtype == torch.float32:
            pytest.xfail(
                reason="view() + mean() triggers Cannot satisfy hardware memory span limit (256MB) without splitting reduction dimensions."
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1688
        if dtype == torch.float16:
            pytest.xfail(
                reason="Variance (aten::var.correction) operation not implemented"
            )

        eps = 1e-5

        def groupnorm(x):
            batch, channels, height, width = x.shape
            x = x.view(batch, num_groups, channels // num_groups, height, width)

            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + eps)

            return normalized.view(batch, channels, height, width)

        x = cached_randn((32, 64, 224, 224), dtype=dtype)
        tol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-3, 1e-2)
        _compare_modes(execution_mode, groupnorm, x, atol=tol[0], rtol=tol[1])

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1722
    @pytest.mark.xfail(
        reason="view() + mean() triggers Cannot satisfy hardware memory span limit (256MB) without splitting reduction dimensions."
    )
    def test_groupnorm_affine(self, execution_mode):
        """Test GroupNorm with affine transformation."""

        eps = 1e-5
        num_groups = 32
        num_channels = 64

        def groupnorm_affine(x):
            batch, channels, height, width = x.shape
            x_grouped = x.view(batch, num_groups, channels // num_groups, height, width)

            mean = x_grouped.mean(dim=[2, 3, 4], keepdim=True)
            var = x_grouped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
            normalized = (x_grouped - mean) / torch.sqrt(var + eps)
            normalized = normalized.view(batch, channels, height, width)

            gamma = torch.ones(num_channels, device=x.device).view(1, -1, 1, 1)
            beta = torch.zeros(num_channels, device=x.device).view(1, -1, 1, 1)

            return gamma * normalized + beta

        x = cached_randn((32, 64, 224, 224), dtype=torch.float32)
        _compare_modes(execution_mode, groupnorm_affine, x, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_instancenorm_2d(self, execution_mode, dtype):
        """Test 2D InstanceNorm with epsilon constant."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
        if dtype == torch.float32:
            pytest.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1688
        if dtype == torch.float16:
            pytest.xfail(
                reason="Variance (aten::var.correction) operation not implemented"
            )

        eps = 1e-5

        def instancenorm(x):
            mean = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + eps)

        x = cached_randn((32, 64, 224, 224), dtype=dtype)
        tol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-3, 1e-2)
        _compare_modes(execution_mode, instancenorm, x, atol=tol[0], rtol=tol[1])

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
    @pytest.mark.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
    def test_instancenorm_1d(self, execution_mode):
        """Test 1D InstanceNorm variant."""

        eps = 1e-5

        def instancenorm_1d(x):
            mean = x.mean(dim=2, keepdim=True)
            var = x.var(dim=2, keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + eps)

        x = cached_randn((32, 768, 100), dtype=torch.float32)
        _compare_modes(execution_mode, instancenorm_1d, x, atol=1e-4, rtol=1e-3)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
    @pytest.mark.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
    def test_instancenorm_3d(self, execution_mode):
        """Test 3D InstanceNorm variant."""

        eps = 1e-5

        def instancenorm_3d(x):
            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + eps)

        x = cached_randn((32, 64, 16, 16, 16), dtype=torch.float32)
        _compare_modes(execution_mode, instancenorm_3d, x, atol=1e-4, rtol=1e-3)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1454
    @pytest.mark.xfail(reason="Mean reduction on float32 (IEEE_FP32) not supported")
    def test_instancenorm_affine(self, execution_mode):
        """Test InstanceNorm with affine transformation."""

        eps = 1e-5
        num_channels = 64

        def instancenorm_affine(x):
            mean = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + eps)

            gamma = torch.ones(num_channels, device=x.device).view(1, -1, 1, 1)
            beta = torch.zeros(num_channels, device=x.device).view(1, -1, 1, 1)

            return gamma * normalized + beta

        x = cached_randn((32, 64, 224, 224), dtype=torch.float32)
        _compare_modes(execution_mode, instancenorm_affine, x, atol=1e-4, rtol=1e-3)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestModelScalarOperations:
    """
    Representative **scalar-tensor** patterns (scale, eps, temperature, momentum) in
    shapes inspired by common models. Names describe the **operation**, not a full
    architecture implementation.
    """

    torch.manual_seed(0xAFFE)

    @pytest.mark.parametrize(
        "batch,heads,seq,d_k",
        [
            (1, 8, 1024, 64),
            (1, 12, 1024, 64),
            (1, 8, 2048, 64),
        ],
    )
    def test_scaled_qk_matmul_attention_scores(
        self, execution_mode, batch, heads, seq, d_k
    ):
        """``matmul(Q,K^T) / sqrt(d_k)`` — scaled dot-product logits only (no V); several (batch, heads, seq) configs."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )
        # TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1730
        if seq == 1024 and execution_mode == "compiled":
            pytest.xfail(
                reason="Assertion Error: Numerical mismatch (45-48% elements) for seq=1024"
            )
        scale = 1.0 / math.sqrt(d_k)

        def scaled_qk_logits(q, k):
            return torch.matmul(q, k.transpose(-2, -1)) * scale

        q = cached_randn((batch, heads, seq, d_k), dtype=torch.float16)
        k = cached_randn(
            (batch, heads, seq, d_k), dtype=torch.float16, differentiation=1
        )
        _compare_modes(execution_mode, scaled_qk_logits, q, k, atol=1e-1, rtol=1e-1)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1688
    @pytest.mark.xfail(
        reason="Variance (aten::var.correction) operation not implemented"
    )
    def test_layernorm_last_dim_eps1e12(self, execution_mode):
        """Last-dim layernorm with ``eps=1e-12`` (BERT-like 3D shape); not ``nn.LayerNorm``."""

        eps = 1e-12

        def bert_layernorm(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + eps)

        x = cached_randn((32, 512, 768), dtype=torch.float16)
        _compare_modes(execution_mode, bert_layernorm, x, atol=1e-3, rtol=1e-2)

    def test_scores_plus_tensor_bias_times_scalar(self, execution_mode):
        """``scores + bias * 0.0625`` — scalar multiplier on an additive bias tensor."""
        bias_scale = 0.0625

        def t5_relative_position(scores, bias):
            return scores + bias * bias_scale

        scores = cached_randn((1, 8, 512, 512), dtype=torch.float16)
        bias = cached_randn((1, 8, 512, 512), dtype=torch.float16, differentiation=1)
        _compare_modes(
            execution_mode, t5_relative_position, scores, bias, atol=1e-3, rtol=1e-2
        )

    def test_tensor_mul_sigmoid_gate(self, execution_mode):
        """Elementwise ``x * sigmoid(gate)`` (not full SwiGLU ``silu·x`` gating)."""

        def gated_mul_sigmoid(x, gate):
            return x * torch.sigmoid(gate)

        x = cached_randn((1, 1024, 4096), dtype=torch.float16)
        gate = cached_randn((1, 1024, 4096), dtype=torch.float16, differentiation=1)
        _compare_modes(execution_mode, gated_mul_sigmoid, x, gate, atol=1e-3, rtol=1e-2)

    def test_tensor_mul_scalar_patch_scale(self, execution_mode):
        """``x * 0.125`` on image-shaped tensor (ViT-like dimensions)."""
        patch_scale = 0.125

        def vit_patch_embed(x):
            return x * patch_scale

        x = cached_randn((1, 3, 224, 224), dtype=torch.float16)
        _compare_modes(execution_mode, vit_patch_embed, x, atol=1e-3, rtol=1e-2)

    def test_logits_div_scalar_temperature(self, execution_mode):
        """``logits / temperature`` with scalar ``temperature=0.07`` (contrastive-style)."""
        temperature = 0.07

        def clip_temperature(logits):
            return logits / temperature

        logits = cached_randn((32, 512), dtype=torch.float16)
        _compare_modes(execution_mode, clip_temperature, logits, atol=1e-3, rtol=1e-2)

    def test_ema_scalar_momentum995(self, execution_mode):
        """``center * m + features * (1-m)`` with scalar ``m=0.9`` (EMA-style mix)."""
        momentum = 0.995

        def blip_ema_update(current, ema):
            return ema * momentum + current * (1.0 - momentum)

        current = cached_randn((32, 768), dtype=torch.float16)
        ema = cached_randn((32, 768), dtype=torch.float16, differentiation=1)
        _compare_modes(
            execution_mode, blip_ema_update, current, ema, atol=1e-3, rtol=1e-2
        )

    def test_matmul_logits_times_scalar_scale(self, execution_mode):
        """``matmul(A,B^T) * 0.125`` — bilinear logits with a scalar scale."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/543
        if execution_mode == "eager":
            pytest.xfail(
                reason="Eager mode: aten::_reshape_alias operation not implemented"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1731
        elif execution_mode == "compiled":
            pytest.xfail(
                reason="Spyre: dxp_standalone SIGABRT in fused_bmm_transpose compilation"
            )

        scale = 0.125

        def clip_cross_attn(text_features, image_features):
            scores = torch.matmul(text_features, image_features.transpose(-2, -1))
            return scores * scale

        text_features = cached_randn((32, 77, 512), dtype=torch.float16)
        image_features = cached_randn(
            (32, 77, 512), dtype=torch.float16, differentiation=1
        )
        _compare_modes(
            execution_mode,
            clip_cross_attn,
            text_features,
            image_features,
            atol=1e-3,
            rtol=1e-2,
        )

    def test_tensor_mul_tanh_gate(self, execution_mode):
        """``tanh(gate) * x`` — multiplicative gating with tanh."""

        def flamingo_gating(x, gate):
            return torch.tanh(gate) * x

        x = cached_randn((1, 1024, 2048), dtype=torch.float16)
        gate = torch.rand((1, 1024, 2048), dtype=torch.float16)
        _compare_modes(execution_mode, flamingo_gating, x, gate, atol=1e-3, rtol=1e-2)

    def test_fused_multiply_add(self, execution_mode):
        """Test fused multiply-add (FMA) pattern."""
        scale = 0.125
        bias = 1.5

        def fused_multiply_add(x):
            return x * scale + bias

        x = cached_randn((256, 256))
        _compare_modes(execution_mode, fused_multiply_add, x, atol=1e-3, rtol=1e-3)

    def test_reciprocal_scaling(self, execution_mode):
        """Test reciprocal scaling pattern."""
        scale = 8.0

        def reciprocal_scale(x):
            return x * (1.0 / scale)

        x = cached_randn((256, 256))
        _compare_modes(execution_mode, reciprocal_scale, x, atol=1e-3, rtol=1e-2)

    def test_power_of_2_scaling(self, execution_mode):
        """Test power-of-2 scaling pattern."""

        def power_of_2_scale(x):
            return x * 16.0

        x = cached_randn((256, 256))
        _compare_modes(execution_mode, power_of_2_scale, x, atol=1e-3, rtol=1e-2)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1723
    @pytest.mark.xfail(reason="Round (aten::round.out) operation not implemented")
    def test_symmetric_quantization(self, execution_mode):
        """Symmetric quantization: round(x / scale) * scale."""

        scale = 0.1

        def symmetric_quant(x):
            return torch.round(x / scale) * scale

        x = cached_randn((256, 256))
        _compare_modes(execution_mode, symmetric_quant, x, atol=0.1, rtol=1e-3)

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1723
    @pytest.mark.xfail(reason="Round (aten::round.out) operation not implemented")
    def test_asymmetric_quantization(self, execution_mode):
        """Test asymmetric quantization pattern."""

        scale = 0.1
        zero_point = 128.0

        def asymmetric_quant(x):
            return (torch.round(x / scale + zero_point) - zero_point) * scale

        x = cached_randn((256, 256))
        _compare_modes(execution_mode, asymmetric_quant, x, atol=0.1, rtol=1e-2)

    def test_dequantization(self, execution_mode):
        """Test dequantization pattern."""
        scale = 0.1
        zero_point = 128.0

        def dequant(x):
            return (x - zero_point) * scale

        x = torch.randint(0, 256, (256, 256), dtype=torch.float32)
        _compare_modes(execution_mode, dequant, x, atol=1e-3, rtol=1e-3)

    def test_log_sum_exp_stability(self, execution_mode):
        """Stable log-sum-exp (``max`` + ``log`` + ``sum(exp)``), not ``torch.logsumexp``."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/453
        if execution_mode == "eager":
            pytest.xfail(reason="Max (aten::max.dim_max) operation not implemented")

        def log_sum_exp(x):
            max_val = torch.max(x, dim=-1, keepdim=True)[0]
            return max_val + torch.log(
                torch.sum(torch.exp(x - max_val), dim=-1, keepdim=True)
            )

        x = cached_randn((256, 256))
        _compare_modes(execution_mode, log_sum_exp, x, atol=0.01, rtol=0.01)

    def test_moe_load_balancing_loss(self, execution_mode):
        """Test MoE load balancing with scalar weight."""
        load_balance_weight = 0.01

        def moe_loss(main_loss, aux_loss):
            return main_loss + load_balance_weight * aux_loss

        main_loss = cached_randn((256, 256))
        aux_loss = cached_randn((256, 256), differentiation=1)
        _compare_modes(
            execution_mode, moe_loss, main_loss, aux_loss, atol=4e-3, rtol=4e-3
        )

    # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1387
    @pytest.mark.xfail(reason="Clamp (aten::clamp) operation not implemented")
    def test_quantization_scale_int8(self, execution_mode):
        """Test INT8 quantization with scale factor."""

        scale = 127.0

        def quantize_int8(x):
            return torch.clamp(x * scale, -128.0, 127.0)

        x = torch.rand((256, 256)) * 2.0 - 1.0
        _compare_modes(execution_mode, quantize_int8, x, atol=1e-3, rtol=1e-3)

    def test_attention_dense_scores_block_mask_scalar(self, execution_mode):
        """Dense attention scores + block-structured mask and scalar fill (not ``torch.sparse``)."""
        mask_value = -1e9

        def sparse_attention(scores, sparse_mask):
            return scores * sparse_mask + (1.0 - sparse_mask) * mask_value

        scores = cached_randn((2, 8, 128, 128), dtype=torch.float16)
        sparse_mask = torch.zeros((2, 8, 128, 128), dtype=torch.float16)
        for i in range(0, 128, 32):
            sparse_mask[:, :, i : i + 32, i : i + 32] = 1.0

        _compare_modes(
            execution_mode,
            sparse_attention,
            scores,
            sparse_mask,
            atol=1e-3,
            rtol=1e-3,
        )
