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

"""LLM inference serving patterns: MoE expert routing (random and topk-based), paged KV cache (pool shapes, 2D slot index, block table), vLLM serving scenarios, FMS 4D/5D layouts, multi-layer transformer KV gather, and real model shapes (Granite/Llama/Mistral)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5

_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_SYMBOLIC_INT_CONV = (
    4304,
    "compile dynamic symbolic size cannot convert symbols to int.",
)
_GATHER_OUT_EAGER = (4328, "aten::gather.out is not registered on Spyre.")
_RESTICKIFY_3ARGS = (4327, "restickify op_spec has 3 args.")
_STICK_INCOMPATIBLE = (
    3265,
    "buf0 (Reduction): no mechanism to resolve stick incompatibility.",
)
_CANNOT_RESCALE = (
    4392,
    "cannot rescale device layout [1, 4, 32] for conversion to torch.float16.",
)
_OUT_LAYOUTS_EMPTY = (4713, "AllSameNode.from_args: out_layouts is empty.")
_TOPK_K_PER_CORE = (
    4716,
    "topk(k=16): no divisor within 1 cores gives k_per_core <= 4.",
)
_TOPKINDEX_STICK = (
    4718,
    "topkindex stick must not contain the reduction or k dimension.",
)
_DXP_STANDALONE = (4321, "dxp_standalone returned non-zero exit status 1.")
_GATHER_EXPANDED_MISMATCH = (
    4840,
    "torch.compile gather along dim=1 with 3D expanded index produces numerical mismatch.",
)
_INDEX_PUT_IMPL_EAGER = (692, "aten::_index_put_impl_ is not registered on Spyre.")
_INDEX_PUT_NO_LAYOUT = (
    4636,
    "buf0 (aten.index_put.default): no supported output layout found.",
)


class TestGatherInferencePatternsEagerCompile:
    """MoE routing (random ids and topk-based), paged KV cache (diverse pool shapes, 2D slot index, block table), vLLM serving scenarios, FMS 4D/5D layouts, multi-layer transformer KV gather, and real model shapes (Granite/Llama/Mistral)."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape,P,dtype,atol,diff_key",
        [
            ((4, 64, 32), 8, torch.float16, _ATOL_F16, "moe01"),
            ((64, 64, 64), 32, torch.float16, _ATOL_F16, "moe02"),
            ((256, 64, 32), 16, torch.float16, _ATOL_F16, "moe03"),
            ((8, 64, 32), 32, torch.float16, _ATOL_F16, "moe04"),
            ((8, 64, 64), 32, torch.bfloat16, _ATOL_BF16, "moe05"),
            ((8, 64, 64), 32, torch.float16, _ATOL_F16, "moe06"),
            ((8, 64, 128), 32, torch.float16, _ATOL_F16, "moe07"),
        ],
    )
    def test_moe_expert_routing(self, execution_mode, shape, P, dtype, atol, diff_key):
        """MoE expert weight gather: expert_w[ids] across shapes and dtypes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn(shape, differentiation=diff_key, dtype=dtype)
        ids = torch.randint(0, shape[0], (P,), dtype=torch.int64)
        compare_mode(execution_mode, lambda w, i: w[i], w, ids, atol=atol, rtol=atol)

    @pytest.mark.skip(reason="This test is throwing crash error at copy_tensor.")
    def test_moe_expert_aggregation(self, execution_mode):
        """Route via gather then weighted-sum aggregation."""
        w = cached_randn((8, 64, 32), differentiation="moe09", dtype=torch.float16)
        ids = torch.randint(0, 8, (16,), dtype=torch.int64)
        scores = torch.randn(16, dtype=torch.float16).softmax(dim=0)
        compare_mode(
            execution_mode,
            lambda w, i, s: (w[i] * s[:, None, None]).sum(dim=0),
            w,
            ids,
            scores,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_moe_multi_layer(self, execution_mode):
        """Two consecutive MoE routing layers; two GATHER_OP_SPECs."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w1 = cached_randn((8, 64, 32), differentiation="moe10a", dtype=torch.float16)
        w2 = cached_randn((8, 32, 16), differentiation="moe10b", dtype=torch.float16)
        ids = torch.randint(0, 8, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w1, w2, i: (w1[i], w2[i]),
            w1,
            w2,
            ids,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "kv_shape,P,dtype,atol,diff_key",
        [
            ((256, 8, 64), 128, torch.float16, _ATOL_F16, "gpa01"),
            ((1024, 8, 64), 128, torch.float16, _ATOL_F16, "gpa02"),
            ((512, 16, 128), 256, torch.float16, _ATOL_F16, "gpa03"),
            ((512, 8, 64), 128, torch.float16, _ATOL_F16, "gpa05"),
            ((1024, 1, 64), 256, torch.float16, _ATOL_F16, "gpa08"),
            ((512, 8, 256), 128, torch.float16, _ATOL_F16, "gpa10"),
            ((512, 8, 64), 128, torch.bfloat16, _ATOL_BF16, "gpa11"),
            ((128, 8, 16, 64), 64, torch.float16, _ATOL_F16, "gpa23"),
        ],
    )
    def test_paged_kv_pool_shape(
        self, execution_mode, kv_shape, P, dtype, atol, diff_key
    ):
        """Paged KV pool gather: kv[idx] across pool shapes, head configs, and dtypes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(kv_shape, differentiation=diff_key, dtype=dtype)
        idx = torch.randint(0, kv_shape[0], (P,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], kv, idx, atol=atol, rtol=atol)

    def test_paged_kv_block_table(self, execution_mode):
        """Block table: slot = block_table[b, t] * page_size + offset."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn((512, 8, 64), differentiation="gpa12", dtype=torch.float16)
        block_table = torch.randint(0, 32, (4, 8), dtype=torch.int64)
        page_size = 16
        flat_slots = (block_table * page_size).flatten() % 512
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            kv,
            flat_slots,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_page_reuse(self, execution_mode):
        """Prefix caching: same page referenced by multiple requests."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn((512, 8, 64), differentiation="gpa13", dtype=torch.float16)
        shared_page = 42
        idx = torch.full((32,), shared_page, dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_paged_kv_online_softmax(self, execution_mode):
        """K gather + attention scores + softmax."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn((512, 8, 64), differentiation="gpa14", dtype=torch.float16)
        q = cached_randn((8, 4, 64), differentiation="gpa14q", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda kv, q, i: torch.softmax(
                (q @ kv[i].permute(1, 2, 0)).float(), dim=-1
            ).half(),
            kv,
            q,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_full_attention(self, execution_mode):
        """Full paged attention: gather K + V + attention + weighted V sum."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv_k = cached_randn((512, 8, 64), differentiation="gpa15k", dtype=torch.float16)
        kv_v = cached_randn((512, 8, 64), differentiation="gpa15v", dtype=torch.float16)
        q = cached_randn((8, 4, 64), differentiation="gpa15q", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda k, v, q, i: (
                torch.softmax((q @ k[i].permute(1, 2, 0)).float() / 8.0, dim=-1).half()
                @ v[i].permute(1, 0, 2)
            ),
            kv_k,
            kv_v,
            q,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_two_gathers(self, execution_mode):
        """K and V gathered from same pool; two GATHER_OP_SPECs."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        k = cached_randn((512, 8, 64), differentiation="gpa21k", dtype=torch.float16)
        v = cached_randn((512, 8, 64), differentiation="gpa21v", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda k, v, i: (k[i], v[i]),
            k,
            v,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_dynamic_seqlen(self, execution_mode):
        """Dynamic Lk; compile once, run at Lk=32 and Lk=64."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing(
            execution_mode, eager=_SYMBOLIC_INT_CONV, compiled=_SYMBOLIC_INT_CONV
        )
        kv = cached_randn((512, 8, 64), differentiation="gpa22", dtype=torch.float16)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for lk in (32, 64):
            idx = torch.randint(0, 512, (lk,), dtype=torch.int64)
            expected = kv[idx]
            result = fn(kv.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, expected, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_paged_kv_5d_split_kv(self, execution_mode):
        """FMS 5D combined K+V along dim=1; gather + split downstream."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(
            (128, 2, 8, 16, 64), differentiation="gpa24", dtype=torch.float16
        )
        idx = torch.randint(0, 128, (4 * 16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.unbind(x[i], dim=1),
            kv,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_sentinel_boundary(self, execution_mode):
        """Page boundary values; no off-by-one at slot boundaries."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn((512, 8, 64), differentiation="gpa25", dtype=torch.float16)
        page_size = 16
        boundary_slots = torch.tensor(
            [page_size - 1, page_size, 2 * page_size - 1, 2 * page_size],
            dtype=torch.int64,
        )
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            kv,
            boundary_slots,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_vllm_kv_cache_write(self, execution_mode):
        """KV cache write via index_put; written positions match."""
        cache = torch.zeros(1, 8, 128, 64, dtype=torch.float16)
        key = torch.randn(1, 8, 11, 64, dtype=torch.float16)
        pos = torch.arange(11, dtype=torch.int64)

        def fn(cache, key, pos):
            return cache.index_copy(2, pos, key[:, :, :11, :])

        compare_mode(
            execution_mode, fn, cache, key, pos, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_vllm_logit_extraction(self, execution_mode):
        """Decode-step per-request last-token gather: logits[b, last_pos[b], :] via torch.gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4840
        _xfail_existing(
            execution_mode, eager=_GATHER_OUT_EAGER, compiled=_GATHER_EXPANDED_MISMATCH
        )
        logits = cached_randn(
            (12, 4, 512), differentiation="vllm05", dtype=torch.float16
        )
        last_pos = torch.randint(0, 4, (12,), dtype=torch.int64)

        def fn(x, pos):
            idx = pos.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            return x.gather(1, idx).squeeze(1)

        compare_mode(
            execution_mode, fn, logits, last_pos, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_vllm_4d_kv_layout(self, execution_mode):
        """vLLM native 4D KV layout (num_blocks, block_sz, kv_h, head_d)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(
            (64, 16, 8, 64), differentiation="vllm07", dtype=torch.float16
        )
        idx = torch.randint(0, 64, (4 * 16,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_vllm_beam_search_reorder(self, execution_mode):
        """Beam search KV reordering via index_select at batch dim."""
        past_kv = cached_randn(
            (4, 32, 8, 64), differentiation="vllm10", dtype=torch.float16
        )
        beam_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            past_kv,
            beam_idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_vllm_llama31_8b(self, execution_mode):
        """Llama-3.1-8B shapes; token emb + KV cache dual gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(
            (512, 8, 128), differentiation="vllm11kv", dtype=torch.float16
        )
        emb = cached_randn((512, 128), differentiation="vllm11emb", dtype=torch.float16)
        slot_idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        tok_idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda kv, emb, si, ti: (kv[si], emb[ti]),
            kv,
            emb,
            slot_idx,
            tok_idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_vllm_mistral_small_24b(self, execution_mode):
        """Mistral-Small-3.2-24B sliding-window KV gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn((512, 16, 128), differentiation="vllm12", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 64,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_vllm_speculative_decode(self, execution_mode):
        """Speculative decode; gather target probs at draft positions."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode,
            eager=_GATHER_OUT_EAGER,
            compiled=_RESTICKIFY_3ARGS,
        )
        probs = cached_randn((512, 64), differentiation="vllm15", dtype=torch.float16)
        draft_tokens = torch.randint(0, 64, (5,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda p, i: torch.gather(
                p[:5], 1, i.unsqueeze(0).expand(5, -1)
            ).diagonal(),
            probs,
            draft_tokens,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "P,dtype,atol,diff_key",
        [
            (64, torch.float16, _ATOL_F16, "fms01"),
            (1, torch.float16, _ATOL_F16, "fms05"),
            (4 * 16, torch.bfloat16, _ATOL_BF16, "fms06"),
        ],
    )
    def test_fms_4d_layout(self, execution_mode, P, dtype, atol, diff_key):
        """FMS 4D KV (128,8,16,128) gather across dtypes and output sizes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn((128, 8, 16, 128), differentiation=diff_key, dtype=dtype)
        idx = torch.randint(0, 128, (P,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], kv, idx, atol=atol, rtol=atol)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "kv_shape,P,diff_key",
        [
            ((512, 8, 64), 1, "mdl02"),
            ((512, 8, 128), 41, "mdl03"),
            ((512, 16, 128), 128, "mdl05"),
        ],
    )
    def test_model_kv_prefill_decode(self, execution_mode, kv_shape, P, diff_key):
        """Model-specific KV gather at production prefill/decode sizes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(kv_shape, differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, kv_shape[0], (P,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_rope_index_select_float32(self, execution_mode):
        """SpyreRotaryEmbedding float32: cos_sin_cache.index_select(0, pos); 4-byte elements."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="mdl07", dtype=torch.float32
        )
        pos = torch.randint(0, 4096, (128,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_token_unpadding(self, execution_mode):
        """Packed token gather: each seq pos from flat packed buffer."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        packed = cached_randn((64, 128), differentiation="mdl12", dtype=torch.float16)
        pos_ids = torch.randint(0, 64, (4, 16), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda p, i: p[i],
            packed,
            pos_ids,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    @pytest.mark.parametrize(
        "kv_shape,idx_shape,diff_key",
        [
            ((512, 32, 128), (4, 16), "gpa27"),
            ((1024, 32, 128), (12, 64), "gpa28"),
        ],
    )
    def test_paged_kv_2d_slot_index(
        self, execution_mode, kv_shape, idx_shape, diff_key
    ):
        """Paged KV: 2D (B,Lk) slot_idxs into 3D cache — FRS paged gather pattern."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(kv_shape, differentiation=diff_key, dtype=torch.float16)
        slot_idxs = torch.randint(0, kv_shape[0], idx_shape, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            kv,
            slot_idxs,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_topk_moe_routing_top1(self, execution_mode):
        """MoE top-1 routing: softmax → topk(1) → index → gather expert weights."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3265
        _xfail_existing(execution_mode, eager=_STICK_INCOMPATIBLE)
        n_experts = 8
        expert_w = cached_randn(
            (n_experts, 64, 32), differentiation="topk01", dtype=torch.float16
        )
        logits = cached_randn(
            (4, n_experts), differentiation="topk01l", dtype=torch.float16
        )

        def fn(expert_w, logits):
            ids = (
                torch.topk(logits.float(), 1, dim=-1)
                .indices.squeeze(-1)
                .to(torch.int64)
            )
            return expert_w[ids]

        compare_mode(
            execution_mode, fn, expert_w, logits, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_topk_moe_routing_top2(self, execution_mode):
        """MoE top-2 routing: topk(2) flattened → gather 2 experts per token."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3265
        _xfail_existing(execution_mode, eager=_STICK_INCOMPATIBLE)
        n_experts = 8
        expert_w = cached_randn(
            (n_experts, 32, 64), differentiation="topk02", dtype=torch.float16
        )
        logits = cached_randn(
            (4, n_experts), differentiation="topk02l", dtype=torch.float16
        )

        def fn(expert_w, logits):
            ids = (
                torch.topk(logits.float(), 2, dim=-1).indices.flatten().to(torch.int64)
            )
            return expert_w[ids]

        compare_mode(
            execution_mode, fn, expert_w, logits, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_topk_moe_weighted_routing(self, execution_mode):
        """MoE top-2 weighted: topk scores × gathered expert outputs → weighted sum."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3265
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4392
        _xfail_existing(
            execution_mode,
            eager=_STICK_INCOMPATIBLE,
            compiled=_CANNOT_RESCALE,
        )
        n_experts = 8
        expert_w = cached_randn(
            (n_experts, 64), differentiation="topk03", dtype=torch.float16
        )
        logits = cached_randn(
            (4, n_experts), differentiation="topk03l", dtype=torch.float16
        )

        def fn(expert_w, logits):
            topk_out = torch.topk(logits.softmax(dim=-1).float(), 2, dim=-1)
            ids = topk_out.indices.to(torch.int64)
            weights = topk_out.values.half()
            selected = expert_w[ids.flatten()].view(4, 2, 64)
            return (selected * weights.unsqueeze(-1)).sum(dim=1)

        compare_mode(
            execution_mode, fn, expert_w, logits, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_topk_speculative_candidate_embed(self, execution_mode, patch_sencores):
        """Speculative decode: topk draft candidates → gather their token embeddings."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4713
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4716
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4718
        _xfail_existing(
            execution_mode,
            patch_sencores=patch_sencores,
            eager=_OUT_LAYOUTS_EMPTY,
            compiled_1=_TOPK_K_PER_CORE,
            compiled_32=_TOPKINDEX_STICK,
        )
        embed = cached_randn((512, 128), differentiation="topk04", dtype=torch.float16)
        scores = cached_randn((512,), differentiation="topk04s", dtype=torch.float16)

        def fn(embed, scores):
            candidate_ids = torch.topk(scores.float(), 16).indices.to(torch.int64)
            return embed[candidate_ids]

        compare_mode(execution_mode, fn, embed, scores, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_topk_beam_search_kv_reorder(self, execution_mode):
        """Beam search: topk beam scores → gather KV rows for top beams."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4713
        _xfail_existing(
            execution_mode,
            eager=_DXP_STANDALONE,
            compiled=_OUT_LAYOUTS_EMPTY,
        )
        kv = cached_randn((8, 8, 64), differentiation="topk05", dtype=torch.float16)
        beam_scores = cached_randn((8,), differentiation="topk05s", dtype=torch.float16)

        def fn(kv, scores):
            beam_ids = torch.topk(scores.float(), 4).indices.to(torch.int64)
            return kv[beam_ids]

        compare_mode(
            execution_mode, fn, kv, beam_scores, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_multi_layer_shared_slot_index(self, execution_mode):
        """4 transformer layers; same slot_idxs reused with separate KV cache per layer."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 256, 8, 64
        slot_idxs = torch.randint(0, pool, (32,), dtype=torch.int64)
        for layer_id in range(4):
            k = cached_randn(
                (pool, H, D), differentiation=f"mlt01k{layer_id}", dtype=torch.float16
            )
            v = cached_randn(
                (pool, H, D), differentiation=f"mlt01v{layer_id}", dtype=torch.float16
            )
            compare_mode(
                execution_mode,
                lambda k, v, s: (k[s], v[s]),
                k,
                v,
                slot_idxs,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_multi_layer_growing_decode_history(self, execution_mode):
        """Decode grows: same KV cache read at 4 increasing history lengths across layers."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 512, 8, 64
        kv = cached_randn((pool, H, D), differentiation="mlt02", dtype=torch.float16)
        for step in (1, 8, 32, 64):
            slot_idxs = torch.randint(0, pool, (step,), dtype=torch.int64)
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                kv,
                slot_idxs,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_multi_layer_4d_kv_per_layer(self, execution_mode):
        """4-layer 4D KV (pool, H, blk, D); per-layer gather at decode with unique tensors."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, blk, D = 64, 8, 4, 32
        for layer_id in range(4):
            kv = cached_randn(
                (pool, H, blk, D),
                differentiation=f"mlt03_{layer_id}",
                dtype=torch.float16,
            )
            idx = torch.randint(0, pool, (8,), dtype=torch.int64)
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                kv,
                idx,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_multi_layer_bfloat16_kv(self, execution_mode):
        """4 transformer layers; bfloat16 KV caches, same slot pattern per layer."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 256, 8, 64
        slot_idxs = torch.randint(0, pool, (16,), dtype=torch.int64)
        for layer_id in range(4):
            kv = cached_randn(
                (pool, H, D), differentiation=f"mlt04_{layer_id}", dtype=torch.bfloat16
            )
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                kv,
                slot_idxs,
                atol=_ATOL_BF16,
                rtol=_ATOL_BF16,
            )

    def test_float32_kv_cache_access(self, execution_mode):
        """float32 3D KV pool (512,8,64) gather; 4-byte element, int64 slot index."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        pool, H, D = 512, 8, 64
        kv = cached_randn((pool, H, D), differentiation="f32kv01", dtype=torch.float32)
        slots = torch.randint(0, pool, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, slots, atol=_ATOL_F32, rtol=_ATOL_F32
        )

    def test_index_put_view_n_32_32(self, execution_mode):
        """index_put_ on view [8, 32, 32] → [8, 1024]; compile Mod(d1, 32) (#4443)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/692
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4636
        _xfail_existing(
            execution_mode,
            eager=_INDEX_PUT_IMPL_EAGER,
            compiled=_INDEX_PUT_NO_LAYOUT,
        )
        x = torch.zeros(8, 32, 32, dtype=torch.float16)
        slots = torch.arange(4, dtype=torch.int32)
        values = torch.ones(4, 1024, dtype=torch.float16)

        def fn(x, slots, values):
            x = x.clone().view(8, 1024)
            x.index_put_((slots,), values, accumulate=False)
            return x

        compare_mode(execution_mode, fn, x, slots, values, atol=0, rtol=0)

    def test_index_put_3d_view_of_4d(self, execution_mode):
        """index_put_ on cache.view(-1, 8, 128) from 4-D dest (#4451)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/692
        _xfail_existing(execution_mode, eager=_INDEX_PUT_IMPL_EAGER)
        cache = torch.zeros(4, 4, 8, 128, dtype=torch.float16)
        slots = torch.tensor([0, 1], dtype=torch.int32)
        values = torch.ones(2, 8, 128, dtype=torch.float16)

        def fn(cache, slots, values):
            cache = cache.clone().view(-1, 8, 128)
            cache.index_put_((slots,), values)
            return cache

        compare_mode(execution_mode, fn, cache, slots, values, atol=0, rtol=0)
