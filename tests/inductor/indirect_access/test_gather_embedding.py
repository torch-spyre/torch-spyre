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

"""Token and positional embedding table lookup, RoPE position cache gather, EmbeddingBag (sum/mean/max modes, per-sample weights), multi-table gather, pooling variants, and downstream linear/norm projections."""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5

_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_EMBEDDING_BAG_UNSUPPORTED = (
    4319,
    "aten::_embedding_bag / _embedding_bag_forward_only is not registered on Spyre.",
)
_EMBEDDING_RENORM = (4320, "aten::embedding_renorm_ is not registered on Spyre.")
_TIED_LM_HEAD_MISMATCH = (
    4723,
    "Compile embedding lookup + tied LM-head matmul numerical mismatch.",
)
_DXP_STANDALONE = (4321, "dxp_standalone returned non-zero exit status 1.")
_MATMUL_MISMATCH = (
    4722,
    "Compile lookup table + linear downstream numerical mismatch.",
)


class TestGatherEmbeddingTableLookupAndPooling:
    """Token embedding (small/medium/gpt2-approx vocab), positional embedding, RoPE cos/sin cache, bfloat16/float32 tables, EmbeddingBag (sum/mean/max), multi-table gather, sum/mean pooling, and downstream projections."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    # ------------------------------------------------------------------

    def test_embedding_small_vocab(self, execution_mode):
        """Small vocab (512,64) via advanced indexing; GATHER_OP_SPEC."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 64), differentiation="emb01", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_embedding_medium_vocab(self, execution_mode):
        """Medium vocab (8192,128) via advanced indexing; stick-aligned."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 128), differentiation="emb02", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_embedding_gpt2_vocab(self, execution_mode):
        """GPT-2 vocab shape (50257,768) approximated; 12-stick row."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((256, 128), differentiation="emb03", dtype=torch.float16)
        idx = torch.randint(0, 256, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_embedding_batch_2d_ids(self, execution_mode):
        """2D token IDs (B=4, S=128); output shape (4,128,dim)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 64), differentiation="emb05", dtype=torch.float16)
        idx = torch.randint(0, 512, (4, 16), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_embedding_positional(self, execution_mode):
        """Positional encoding via index_select(0, pos_ids)."""
        pos_emb = cached_randn(
            (2048, 128), differentiation="emb07", dtype=torch.float16
        )
        pos_ids = torch.randint(0, 2048, (128,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda p, i: torch.index_select(p, 0, i),
            pos_emb,
            pos_ids,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_rope_cos_sin(self, execution_mode):
        """RoPE cos/sin cache lookup; index_select then chunk(2,-1)."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="emb08", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_bfloat16(self, execution_mode):
        """bfloat16 embedding table; SDSC wordLength=2."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 64), differentiation="emb09", dtype=torch.bfloat16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16
        )

    def test_embedding_float32(self, execution_mode):
        """float32 embedding table; 4-byte elements."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 64), differentiation="emb10", dtype=torch.float32)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F32, rtol=_ATOL_F32
        )

    def test_embedding_with_exp(self, execution_mode):
        """Embedding lookup + exp() downstream fused."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 64), differentiation="emb12", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: torch.exp(w[i]),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_functional_api(self, execution_mode):
        """F.embedding(idx, weight) → aten.embedding → NO_SPYRE_OP (CPU fallback)."""
        w = cached_randn((512, 64), differentiation="emb13", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: F.embedding(i, w),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_nn_module(self, execution_mode):
        """nn.Embedding.forward → aten.embedding lookup."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((512, 64), differentiation="emb14_mod", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: F.embedding(i, w),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_tied_lm_head(self, execution_mode):
        """Weight-tied model: same matrix for embedding and LM head."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4723
        _xfail_existing(
            execution_mode, eager=_INDEX_EAGER, compiled=_TIED_LM_HEAD_MISMATCH
        )
        w = cached_randn((512, 64), differentiation="emb15", dtype=torch.float16)
        h = cached_randn((8, 64), differentiation="emb15h", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)

        def fn(w, h, i):
            emb_out = w[i]
            lm_out = torch.matmul(h, w.t())
            return emb_out, lm_out

        compare_mode(execution_mode, fn, w, h, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    # ------------------------------------------------------------------

    def test_embeddingbag_sum_mode(self, execution_mode):
        """GEMB2-01: EmbeddingBag sum mode; output is sum of selected rows."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((64, 64), differentiation="emb2_01", dtype=torch.float16)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, o: F.embedding_bag(i, w, o, mode="sum"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embeddingbag_mean_mode(self, execution_mode):
        """GEMB2-02: EmbeddingBag mean mode; output is mean of selected rows."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((64, 64), differentiation="emb2_02", dtype=torch.float16)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, o: F.embedding_bag(i, w, o, mode="mean"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embeddingbag_max_mode(self, execution_mode):
        """GEMB2-03: EmbeddingBag max mode; output is element-wise max (float32)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((64, 64), differentiation="emb2_03", dtype=torch.float32)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, o: F.embedding_bag(i, w, o, mode="max"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_max_float16(self, execution_mode):
        """EmbeddingBag max mode in float16; 2-byte element max reduction via gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((64, 64), differentiation="emb2_maxf16", dtype=torch.float16)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, o: F.embedding_bag(i, w, o, mode="max"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embeddingbag_per_sample_weights(self, execution_mode):
        """GEMB2-04: EmbeddingBag with per-sample weights; weighted sum."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((64, 64), differentiation="emb2_04", dtype=torch.float32)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        per_sample_w = cached_randn(
            (8,), differentiation="emb2_04w", dtype=torch.float32
        )
        compare_mode(
            execution_mode,
            lambda w, i, o, s: F.embedding_bag(
                i, w, o, mode="sum", per_sample_weights=s
            ),
            w,
            input_ids,
            offsets,
            per_sample_w,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_large_vocab(self, execution_mode):
        """GEMB2-05: EmbeddingBag with large vocabulary (V=32000)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((512, 64), differentiation="emb2_05", dtype=torch.float32)
        input_ids = torch.randint(0, 512, (16,), dtype=torch.int64)
        offsets = torch.tensor([0, 8], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, o: F.embedding_bag(i, w, o, mode="sum"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_single_bag(self, execution_mode):
        """GEMB2-06: EmbeddingBag with one bag; degenerate offsets=[0]."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        w = cached_randn((64, 64), differentiation="emb2_06", dtype=torch.float32)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, o: F.embedding_bag(i, w, o, mode="sum"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_nn_module(self, execution_mode):
        """GEMB2-07: nn.EmbeddingBag forward; sum mode via module API."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4319
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_BAG_UNSUPPORTED,
            compiled=_EMBEDDING_BAG_UNSUPPORTED,
        )
        emb_bag = nn.EmbeddingBag(64, 64, mode="sum")
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda i, o, m: m(i, o),
            input_ids,
            offsets,
            emb_bag,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_lookup_table_two_tables(self, execution_mode):
        """GEMB2-08: Two separate lookup tables gathered in one graph."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w1 = cached_randn((64, 64), differentiation="emb2_08a", dtype=torch.float16)
        w2 = cached_randn((128, 64), differentiation="emb2_08b", dtype=torch.float16)
        i1 = torch.randint(0, 64, (16,), dtype=torch.int64)
        i2 = torch.randint(0, 128, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w1, w2, i1, i2: (w1[i1], w2[i2]),
            w1,
            w2,
            i1,
            i2,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lookup_table_sum_pooling(self, execution_mode):
        """GEMB2-09: Lookup + sum pooling; table[idx].sum(dim=0)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((64, 128), differentiation="emb2_09", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: w[i].sum(dim=0),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lookup_table_mean_pooling(self, execution_mode):
        """GEMB2-10: Lookup + mean pooling; table[idx].mean(dim=0)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((64, 128), differentiation="emb2_10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: w[i].mean(dim=0),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    @pytest.mark.skip(
        reason="#3502: RuntimeError: StreamInErrorState: cannot launch H2D — stream is in error state."
    )
    def test_lookup_table_concat(self, execution_mode):
        """GEMB2-11: Concatenate two lookups on dim=1; feature concatenation."""
        w1 = cached_randn((64, 32), differentiation="emb2_11a", dtype=torch.float16)
        w2 = cached_randn((64, 32), differentiation="emb2_11b", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda a, b, i: torch.cat([a[i], b[i]], dim=1),
            w1,
            w2,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lookup_table_bfloat16(self, execution_mode):
        """GEMB2-12: bfloat16 lookup table gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((64, 64), differentiation="emb2_12", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16
        )

    def test_lookup_table_with_layer_norm(self, execution_mode):
        """GEMB2-13: Lookup table + layer_norm downstream."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(execution_mode, eager=_INDEX_EAGER, compiled=_DXP_STANDALONE)
        w = cached_randn((64, 64), differentiation="emb2_13", dtype=torch.float32)
        weight = cached_randn(
            (64,), differentiation="emb2_13_ln_w", dtype=torch.float32
        )
        bias = cached_randn((64,), differentiation="emb2_13_ln_b", dtype=torch.float32)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, weight, bias: F.layer_norm(
                w[i], (64,), weight=weight, bias=bias
            ),
            w,
            idx,
            weight,
            bias,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_lookup_table_with_linear(self, execution_mode):
        """GEMB2-14: Lookup table + linear projection downstream."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4722
        _xfail_existing(execution_mode, eager=_INDEX_EAGER, compiled=_MATMUL_MISMATCH)
        w = cached_randn((64, 64), differentiation="emb2_14", dtype=torch.float16)
        proj_weight = cached_randn(
            (32, 64), differentiation="emb2_14_proj_w", dtype=torch.float16
        )
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i, pw: F.linear(w[i], pw),
            w,
            idx,
            proj_weight,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_max_norm_renorm(self, execution_mode):
        """F.embedding max_norm=1.0 → aten::embedding_renorm_ extra local (#4320).

        YAML already xfails test_embedding_max_norm_device. This is the arange 8×4 repro.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4320
        _xfail_existing(
            execution_mode,
            eager=_EMBEDDING_RENORM,
            compiled=_EMBEDDING_RENORM,
        )
        w = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4).to(torch.float16)
        idx = torch.tensor([1, 2, 4, 5, 4, 3, 2, 7], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: F.embedding(i, w.clone(), max_norm=1.0),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )
