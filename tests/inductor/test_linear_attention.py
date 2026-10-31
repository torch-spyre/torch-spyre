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

"""Unit tests for Linear Attention kernel integration (Phase 1)."""

import math
import unittest
import torch
import torch.nn.functional as F

from torch_spyre._inductor.decompositions import spyre_linear_attention
from torch_spyre._inductor.errors import Unsupported
from torch_spyre.ops import linear_attention


def _reference_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    is_causal: bool = False,
    feature_map: str = "elu",
    scale: float | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference mathematical implementation of linear attention."""
    B, Hq, Lq, Dk = query.shape
    _, Hkv, Lkv, Dv = value.shape

    if scale is None:
        scale = 1.0 / math.sqrt(Dk)

    def phi(x):
        if feature_map == "elu":
            return F.elu(x) + 1.0
        elif feature_map == "relu":
            return F.relu(x)
        elif feature_map == "silu":
            return F.silu(x)
        elif feature_map in ("identity", "none"):
            return x
        raise ValueError(f"Unknown feature map {feature_map}")

    if Hq > Hkv:
        group_size = Hq // Hkv
        key = torch.repeat_interleave(key, group_size, dim=1)
        value = torch.repeat_interleave(value, group_size, dim=1)
        if initial_state is not None and initial_state.size(1) == Hkv:
            initial_state = torch.repeat_interleave(initial_state, group_size, dim=1)

    Q = phi(query * scale)
    K = phi(key)
    V = value

    if is_causal:
        delta_s = K.unsqueeze(-1) * V.unsqueeze(-2)  # [B, H, L, Dk, Dv]
        causal_s = torch.cumsum(delta_s, dim=2)
        if initial_state is not None:
            causal_s = causal_s + initial_state.unsqueeze(2)
        final_state = causal_s[:, :, -1, :, :]

        out = torch.matmul(Q.unsqueeze(-2), causal_s).squeeze(-2)
        k_cumsum = torch.cumsum(K, dim=2)
        z = torch.sum(Q * k_cumsum, dim=-1, keepdim=True)
        out = out / (z + eps)
        return out, final_state
    else:
        kv_state = torch.matmul(K.transpose(-2, -1), V)  # [B, H, Dk, Dv]
        if initial_state is not None:
            kv_state = kv_state + initial_state

        out = torch.matmul(Q, kv_state)
        k_sum = torch.sum(K, dim=-2, keepdim=True)  # [B, H, 1, Dk]
        z = torch.matmul(Q, k_sum.transpose(-2, -1))
        out = out / (z + eps)
        return out, kv_state


class TestLinearAttention(unittest.TestCase):
    def test_non_causal_mha_elu(self):
        batch, heads, seqlen, dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seqlen, dim)
        k = torch.randn(batch, heads, seqlen, dim)
        v = torch.randn(batch, heads, seqlen, dim)

        ref_out, ref_state = _reference_linear_attention(q, k, v, is_causal=False, feature_map="elu")
        act_out, act_state = spyre_linear_attention(q, k, v, is_causal=False, feature_map="elu")

        torch.testing.assert_close(act_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(act_state, ref_state, atol=1e-5, rtol=1e-5)

    def test_causal_mha_elu(self):
        batch, heads, seqlen, dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seqlen, dim)
        k = torch.randn(batch, heads, seqlen, dim)
        v = torch.randn(batch, heads, seqlen, dim)

        ref_out, ref_state = _reference_linear_attention(q, k, v, is_causal=True, feature_map="elu")
        act_out, act_state = spyre_linear_attention(q, k, v, is_causal=True, feature_map="elu")

        torch.testing.assert_close(act_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(act_state, ref_state, atol=1e-5, rtol=1e-5)

    def test_feature_maps(self):
        batch, heads, seqlen, dim = 1, 2, 16, 64
        q = torch.randn(batch, heads, seqlen, dim)
        k = torch.randn(batch, heads, seqlen, dim)
        v = torch.randn(batch, heads, seqlen, dim)

        for fmap in ["elu", "relu", "silu", "identity"]:
            ref_out, ref_state = _reference_linear_attention(q, k, v, is_causal=True, feature_map=fmap)
            act_out, act_state = spyre_linear_attention(q, k, v, is_causal=True, feature_map=fmap)
            torch.testing.assert_close(act_out, ref_out, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(act_state, ref_state, atol=1e-5, rtol=1e-5)

    def test_gqa(self):
        batch, q_heads, kv_heads, seqlen, dim = 2, 8, 2, 32, 64
        q = torch.randn(batch, q_heads, seqlen, dim)
        k = torch.randn(batch, kv_heads, seqlen, dim)
        v = torch.randn(batch, kv_heads, seqlen, dim)

        ref_out, ref_state = _reference_linear_attention(q, k, v, is_causal=False, feature_map="elu")
        act_out, act_state = spyre_linear_attention(q, k, v, is_causal=False, feature_map="elu")

        torch.testing.assert_close(act_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(act_state, ref_state, atol=1e-5, rtol=1e-5)

    def test_initial_state_carry(self):
        batch, heads, seqlen, dim = 1, 2, 16, 64
        init_state = torch.randn(batch, heads, dim, dim)
        q = torch.randn(batch, heads, seqlen, dim)
        k = torch.randn(batch, heads, seqlen, dim)
        v = torch.randn(batch, heads, seqlen, dim)

        ref_out, ref_state = _reference_linear_attention(
            q, k, v, initial_state=init_state, is_causal=True, feature_map="elu"
        )
        act_out, act_state = spyre_linear_attention(
            q, k, v, initial_state=init_state, is_causal=True, feature_map="elu"
        )

        torch.testing.assert_close(act_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(act_state, ref_state, atol=1e-5, rtol=1e-5)

    def test_cross_attention_non_causal(self):
        batch, heads, q_len, kv_len, dim = 2, 4, 16, 64, 64
        q = torch.randn(batch, heads, q_len, dim)
        k = torch.randn(batch, heads, kv_len, dim)
        v = torch.randn(batch, heads, kv_len, dim)

        ref_out, ref_state = _reference_linear_attention(q, k, v, is_causal=False)
        act_out, act_state = spyre_linear_attention(q, k, v, is_causal=False)

        torch.testing.assert_close(act_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(act_state, ref_state, atol=1e-5, rtol=1e-5)

    def test_custom_op_and_wrapper(self):
        batch, heads, seqlen, dim = 1, 2, 8, 64
        q = torch.randn(batch, heads, seqlen, dim)
        k = torch.randn(batch, heads, seqlen, dim)
        v = torch.randn(batch, heads, seqlen, dim)

        out, state = linear_attention(q, k, v, is_causal=False)
        self.assertEqual(out.shape, (batch, heads, seqlen, dim))
        self.assertEqual(state.shape, (batch, heads, dim, dim))

    def test_validation_errors(self):
        q = torch.randn(1, 4, 16, 64)
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn(1, 4, 32, 64)

        with self.assertRaises(Unsupported):
            spyre_linear_attention(q, k, v, is_causal=True)

        with self.assertRaises(Unsupported):
            spyre_linear_attention(q, k, v, is_causal=False, feature_map="invalid_map")


if __name__ == "__main__":
    unittest.main()
