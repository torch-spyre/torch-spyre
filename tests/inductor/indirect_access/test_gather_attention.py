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

"""Scaled dot-product attention (SDPA) with paged KV cache gather: decode/prefill, GQA/MQA, causal mask, attention bias, batched decode, RoPE+attention, LX planning, chunked prefill, and speculative decoding verification."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_SDPA = 2e-2


_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_POINTWISE_NO_LAYOUT = (
    4306,
    "compile Multi-arg pointwise no supported output layout found.",
)
_PAD_COMPILED = (
    4715,
    "compile SDPA lower_pad_sequence expected exactly dim=2 to be padded.",
)
_SDPA_LIN_MISMATCH = (
    4728,
    "compile paged-KV SDPA followed by output linear projection numerical mismatch.",
)
_RESTICKIFY_MOD = (
    4760,
    "compile 4D KV cache reshape-before-permute: insert_restickify_padding does not "
    "yet handle Mod/FloorDiv multi-symbol host coordinates produced by coarse tiling "
    "of a merged post-gather dimension.",
)


class TestGatherPagedAttentionAndSDPA:
    """Paged KV cache gather integrated with SDPA: MHA/GQA/MQA decode and prefill, causal mask, attention bias, multi-core, LX planning, chunked prefill, RoPE+attention, and speculative verification."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield
        os.environ.pop("LX_PLANNING", None)

    # ------------------------------------------------------------------

    def test_sdpa_paged_decode_basic(self, execution_mode):
        """Single-token decode: gather 32 KV slots from (512,8,64) pool → SDPA output."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 512, 8, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn01k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn01v", dtype=torch.float16
        )
        q = cached_randn((1, H, 1, D), differentiation="attn01q", dtype=torch.float16)
        slots = torch.randint(0, pool, (32,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_paged_prefill_full_context(self, execution_mode):
        """Prefill: gather 64 slots for context → SDPA self-attention over full context."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D, Lq = 512, 8, 64, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn02k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn02v", dtype=torch.float16
        )
        q = cached_randn((1, H, Lq, D), differentiation="attn02q", dtype=torch.float16)
        slots = torch.randint(0, pool, (Lq,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_gqa_32q_8kv(self, execution_mode):
        """GQA: 32 Q heads, 8 KV heads (group=4); gather KV → expand → SDPA."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H_q, H_kv, D = 512, 32, 8, 64
        k_cache = cached_randn(
            (pool, H_kv, D), differentiation="attn03k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H_kv, D), differentiation="attn03v", dtype=torch.float16
        )
        q = cached_randn((1, H_q, 1, D), differentiation="attn03q", dtype=torch.float16)
        slots = torch.randint(0, pool, (16,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = (
                k_cache[s]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .repeat_interleave(H_q // H_kv, dim=1)
            )
            v = (
                v_cache[s]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .repeat_interleave(H_q // H_kv, dim=1)
            )
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_mqa_single_kv_head(self, execution_mode):
        """MQA: 8 Q heads, 1 KV head; paged gather + broadcast + SDPA."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H_q, H_kv, D = 512, 8, 1, 64
        k_cache = cached_randn(
            (pool, H_kv, D), differentiation="attn04k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H_kv, D), differentiation="attn04v", dtype=torch.float16
        )
        q = cached_randn((1, H_q, 1, D), differentiation="attn04q", dtype=torch.float16)
        slots = torch.randint(0, pool, (32,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0).expand(1, H_q, 32, D)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0).expand(1, H_q, 32, D)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_bfloat16_paged_decode(self, execution_mode):
        """bfloat16 KV cache gather + SDPA; wordLength=2 throughout pipeline."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 512, 8, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn05k", dtype=torch.bfloat16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn05v", dtype=torch.bfloat16
        )
        q = cached_randn((1, H, 1, D), differentiation="attn05q", dtype=torch.bfloat16)
        slots = torch.randint(0, pool, (32,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_BF16,
            rtol=_ATOL_BF16,
        )

    def test_sdpa_attention_bias_neg_inf_padding(self, execution_mode, patch_sencores):
        """Attention bias with -inf for padded positions; gather + biased SDPA."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4715
        _xfail_existing(
            execution_mode,
            patch_sencores=patch_sencores,
            eager=_INDEX_EAGER,
            compiled_32=_PAD_COMPILED,
        )
        pool, H, D, Lq, Lk = 512, 8, 64, 4, 32
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn07k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn07v", dtype=torch.float16
        )
        q = cached_randn((1, H, Lq, D), differentiation="attn07q", dtype=torch.float16)
        slots = torch.randint(0, pool, (Lk,), dtype=torch.int64)
        attn_bias = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        attn_bias[0, 0, :, Lk // 2 :] = float("-inf")

        def fn(k_cache, v_cache, q, s, bias):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            attn_bias,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_batch_decode_b4(self, execution_mode):
        """Batched decode B=4; each request gathers 8 slots → SDPA per request."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D, Lk = 512, 8, 64, 8
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn08k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn08v", dtype=torch.float16
        )
        q = cached_randn((4, H, 1, D), differentiation="attn08q", dtype=torch.float16)
        slots = torch.randint(0, pool, (4 * Lk,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].reshape(4, Lk, H, D).permute(0, 2, 1, 3)
            v = v_cache[s].reshape(4, Lk, H, D).permute(0, 2, 1, 3)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_2d_slot_index_batched_decode(self, execution_mode):
        """2D slot_idxs (B=4, Lk=16): paged batch decode; each row is one request."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D, B, Lk = 512, 8, 64, 4, 16
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn12k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn12v", dtype=torch.float16
        )
        q = cached_randn((B, H, 1, D), differentiation="attn12q", dtype=torch.float16)
        slot_idxs = torch.randint(0, pool, (B, Lk), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(0, 2, 1, 3)
            v = v_cache[s].permute(0, 2, 1, 3)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slot_idxs,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_rope_gather_then_attention(self, execution_mode):
        """Gather RoPE position embeddings → apply rotate-half → SDPA."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4306
        _xfail_existing(
            execution_mode,
            eager=_INDEX_EAGER,
            compiled=_POINTWISE_NO_LAYOUT,
        )
        pool, H, D, Lk = 512, 8, 64, 32
        half = D // 2
        cos_sin = cached_randn(
            (4096, D), differentiation="attn13cs", dtype=torch.float16
        )
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn13k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn13v", dtype=torch.float16
        )
        q_raw = cached_randn(
            (1, H, 1, D), differentiation="attn13q", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (1,), dtype=torch.int64)
        slots = torch.randint(0, pool, (Lk,), dtype=torch.int64)

        def fn(cs, k_cache, v_cache, q_raw, pos, slots):
            cos = cs[pos, :half].unsqueeze(1)
            sin = cs[pos, half:].unsqueeze(1)
            q_rot = torch.cat(
                [
                    q_raw[:, :, :, :half] * cos - q_raw[:, :, :, half:] * sin,
                    q_raw[:, :, :, half:] * cos + q_raw[:, :, :, :half] * sin,
                ],
                dim=-1,
            )
            k = k_cache[slots].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[slots].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q_rot, k, v)

        compare_mode(
            execution_mode,
            fn,
            cos_sin,
            k_cache,
            v_cache,
            q_raw,
            pos,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_chunked_prefill_then_single_decode(self, execution_mode):
        """4 prefill chunks of 32 tokens each, then 1 single-token decode; all via paged gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 512, 8, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn14k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn14v", dtype=torch.float16
        )
        for chunk in range(4):
            Lq = 32
            q = cached_randn(
                (1, H, Lq, D), differentiation=f"attn14q{chunk}", dtype=torch.float16
            )
            slots = torch.randint(0, pool, (Lq,), dtype=torch.int64)

            def fn(k_cache, v_cache, q, s):
                k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
                v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

            compare_mode(
                execution_mode,
                fn,
                k_cache,
                v_cache,
                q,
                slots,
                atol=_ATOL_SDPA,
                rtol=_ATOL_SDPA,
            )
        q_dec = cached_randn(
            (1, H, 1, D), differentiation="attn14qd", dtype=torch.float16
        )
        dec_slot = torch.randint(0, pool, (128,), dtype=torch.int64)

        def fn_dec(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn_dec,
            k_cache,
            v_cache,
            q_dec,
            dec_slot,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_large_kv_pool_decode(self, execution_mode):
        """Large pool (1024 slots), 128-slot decode gather → SDPA correctness."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 1024, 8, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn15k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn15v", dtype=torch.float16
        )
        q = cached_randn((1, H, 1, D), differentiation="attn15q", dtype=torch.float16)
        slots = torch.randint(0, pool, (128,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_separate_k_v_caches_different_shapes(self, execution_mode):
        """Separate K (512,8,64) and V (512,8,64) cache shapes with different sequence slots; both gathered → SDPA."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 512, 8, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn16k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn16v", dtype=torch.float16
        )
        q = cached_randn((1, H, 1, D), differentiation="attn16q", dtype=torch.float16)
        slots = torch.randint(0, pool, (32,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_output_then_linear_projection(self, execution_mode, patch_sencores):
        """Gather KV → SDPA → linear output projection; full attention block pattern."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4715
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4728
        _xfail_existing(
            execution_mode,
            patch_sencores=patch_sencores,
            eager=_INDEX_EAGER,
            compiled_1=_SDPA_LIN_MISMATCH,
            compiled_32=_PAD_COMPILED,
        )
        pool, H, D = 512, 8, 64
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn17k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn17v", dtype=torch.float16
        )
        q = cached_randn((1, H, 4, D), differentiation="attn17q", dtype=torch.float16)
        out_proj = cached_randn(
            (H * D, H * D), differentiation="attn17p", dtype=torch.float16
        )
        slots = torch.randint(0, pool, (32,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, out_proj, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            B, H_, Lq, D_ = attn_out.shape
            flat = attn_out.transpose(1, 2).reshape(B * Lq, H_ * D_)
            return flat @ out_proj

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            out_proj,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_beam_reorder_then_attend(self, execution_mode):
        """Beam search KV reorder via index_select → SDPA on reordered KV."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D, B_beam = 512, 8, 64, 4
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn18k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn18v", dtype=torch.float16
        )
        q = cached_randn(
            (B_beam, H, 1, D), differentiation="attn18q", dtype=torch.float16
        )
        slots = torch.randint(0, pool, (B_beam * 32,), dtype=torch.int64)
        beam_idx = torch.randperm(B_beam, dtype=torch.int64)

        def fn(k_cache, v_cache, q, slots, beam_idx):
            k_flat = k_cache[slots].reshape(B_beam, 32, H, D).permute(0, 2, 1, 3)
            v_flat = v_cache[slots].reshape(B_beam, 32, H, D).permute(0, 2, 1, 3)
            k_reord = torch.index_select(k_flat, 0, beam_idx)
            v_reord = torch.index_select(v_flat, 0, beam_idx)
            return F.scaled_dot_product_attention(q, k_reord, v_reord)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            beam_idx,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_speculative_draft_verify(self, execution_mode, patch_sencores):
        """Speculative decode: draft tokens verified via SDPA score comparison."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4715
        _xfail_existing(
            execution_mode,
            patch_sencores=patch_sencores,
            eager=_INDEX_EAGER,
            compiled_32=_PAD_COMPILED,
        )
        pool, H, D, Ldraft = 512, 8, 64, 5
        k_cache = cached_randn(
            (pool, H, D), differentiation="attn19k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, D), differentiation="attn19v", dtype=torch.float16
        )
        q_draft = cached_randn(
            (1, H, Ldraft, D), differentiation="attn19q", dtype=torch.float16
        )
        slots = torch.randint(0, pool, (Ldraft + 4,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q_draft,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_4d_kv_layout_paged(self, execution_mode):
        """4D KV cache (pool, H, blk, D) gathered at dim=0 → permute → reshape → SDPA.

        Uses permute-before-reshape so the head dim is moved to the front before
        blk×slots are flattened into the sequence axis.  This keeps the restickify
        input coordinate linear and avoids the Mod/FloorDiv multi-symbol expression
        that insert_restickify_padding does not yet support.

        See test_sdpa_4d_kv_reshape_before_permute for the reshape-before-permute
        variant (tracked compiler gap).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, blk, D = 128, 8, 4, 32
        k_cache = cached_randn(
            (pool, H, blk, D), differentiation="attn20k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, blk, D), differentiation="attn20v", dtype=torch.float16
        )
        q = cached_randn((1, H, 1, D), differentiation="attn20q", dtype=torch.float16)
        slots = torch.randint(0, pool, (8,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            # permute(0,2,1,3): [slots,H,blk,D] → [slots,blk,H,D]
            # reshape(8, H*blk, D): merge blk×slots into sequence axis with H already outer
            k = k_cache[s].permute(0, 2, 1, 3).reshape(8, H * blk, D).unsqueeze(0)
            v = v_cache[s].permute(0, 2, 1, 3).reshape(8, H * blk, D).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )

    def test_sdpa_4d_kv_reshape_before_permute(self, execution_mode):
        """4D KV cache: reshape(Lk*blk, H, D) BEFORE permute(1, 0, 2) — compiler gap.

        The idiomatic real-world pattern (vLLM PagedAttention, TGI, Granite serving)
        merges the gathered Lk×blk dims first, then permutes H to the front:

            k_cache[s].reshape(Lk*blk, H, D).permute(1, 0, 2).unsqueeze(0)

        This is mathematically equivalent to permute-before-reshape but produces a
        Mod/FloorDiv multi-symbol host coordinate after coarse tiling that
        insert_restickify_padding does not yet handle.

        Tracked: see _RESTICKIFY_MOD issue marker.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4760
        _xfail_existing(execution_mode, eager=_INDEX_EAGER, compiled=_RESTICKIFY_MOD)
        pool, H, blk, D = 128, 8, 4, 32
        k_cache = cached_randn(
            (pool, H, blk, D), differentiation="attn21k", dtype=torch.float16
        )
        v_cache = cached_randn(
            (pool, H, blk, D), differentiation="attn21v", dtype=torch.float16
        )
        q = cached_randn((1, H, 1, D), differentiation="attn21q", dtype=torch.float16)
        slots = torch.randint(0, pool, (8,), dtype=torch.int64)

        def fn(k_cache, v_cache, q, s):
            k = k_cache[s].reshape(8 * blk, H, D).permute(1, 0, 2).unsqueeze(0)
            v = v_cache[s].reshape(8 * blk, H, D).permute(1, 0, 2).unsqueeze(0)
            return F.scaled_dot_product_attention(q, k, v)

        compare_mode(
            execution_mode,
            fn,
            k_cache,
            v_cache,
            q,
            slots,
            atol=_ATOL_SDPA,
            rtol=_ATOL_SDPA,
        )
