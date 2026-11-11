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

"""End-to-end correctness of spyre::multi_latent_attention.

Compared against the definition: materialize per-head keys and values from the
latent cache, then run ordinary masked SDPA on them in fp32. The op has two
execution paths that compute the same function -- absorbed, which folds the key
up-projection into the query and never forms a per-head key, and materialized,
which projects each K/V tile inside the counted loop. Every shape here is run
through both, so a path that silently diverges fails rather than being masked
by the shape-based default.

Dimension names follow DeepSeek-V3's configuration fields: ``Dn`` is
qk_nope_head_dim, ``Dr`` is qk_rope_head_dim, ``R`` is kv_lora_rank and ``Dv``
is v_head_dim.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_multi_latent_attention.py -v
"""

import math
import unittest

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor.decompositions import (
    _mla_prefer_absorbed,
    spyre_multi_latent_attention,
)
from torch_spyre._inductor.errors import Unsupported
from utils_inductor import cached_randn, compare_with_cpu

# One scaled-down DeepSeek-V3 layer: the published dimensions keep their ratio
# (R = 4 * Dn, Dr = Dn / 2, Dv = Dn), only the head count and context shrink.
NOPE_DIM = 128
ROPE_DIM = 64
LATENT_RANK = 512
VALUE_DIM = 128


def _inputs(
    batch,
    heads,
    seqlen_q,
    seqlen_kv,
    nope_dim=NOPE_DIM,
    rope_dim=ROPE_DIM,
    latent_rank=LATENT_RANK,
    value_dim=VALUE_DIM,
):
    """Query, latent cache and up-projection weights for one MLA layer.

    The weights are scaled down by sqrt(latent_rank) so the projected keys and
    values keep roughly unit variance: an unscaled [R, D] projection of unit
    inputs produces values around sqrt(R) = 22, whose scores overflow fp16 long
    before the kernel is at fault.
    """
    weight_scale = 1.0 / math.sqrt(latent_rank)
    q_nope = cached_randn(
        (batch, heads, seqlen_q, nope_dim), differentiation=1, dtype=torch.float16
    )
    q_pe = cached_randn(
        (batch, heads, seqlen_q, rope_dim), differentiation=2, dtype=torch.float16
    )
    kv_latent = cached_randn(
        (batch, seqlen_kv, latent_rank), differentiation=3, dtype=torch.float16
    )
    k_pe = cached_randn(
        (batch, seqlen_kv, rope_dim), differentiation=4, dtype=torch.float16
    )
    w_uk = cached_randn(
        (heads, nope_dim, latent_rank),
        differentiation=5,
        dtype=torch.float16,
        scale=weight_scale,
    )
    w_uv = cached_randn(
        (heads, value_dim, latent_rank),
        differentiation=6,
        dtype=torch.float16,
        scale=weight_scale,
    )
    return q_nope, q_pe, kv_latent, k_pe, w_uk, w_uv


def _attention(
    q_nope,
    q_pe,
    kv_latent,
    k_pe,
    w_uk,
    w_uv,
    attention_mask,
    is_causal,
    scale,
    absorbed,
):
    """Dispatch: the latent op on Spyre, materialized fp32 SDPA on CPU."""
    if q_nope.device.type == "spyre":
        return torch.ops.spyre.multi_latent_attention(
            q_nope,
            q_pe,
            kv_latent,
            k_pe,
            w_uk,
            w_uv,
            attention_mask,
            is_causal,
            scale,
            absorbed,
        )

    # The definition, in fp32: per-head keys and values exist here even though
    # the op never materializes a cache-length copy of either. fp16 would
    # accumulate a 512-term latent dot product in the reference itself and
    # would not justify the tolerance below.
    num_heads = q_nope.size(1)
    if scale is None:
        scale = 1.0 / math.sqrt(q_nope.size(-1) + q_pe.size(-1))
    latent = kv_latent.to(torch.float32)
    k_nope = torch.einsum("blr,hdr->bhld", latent, w_uk.to(torch.float32))
    value = torch.einsum("blr,hdr->bhld", latent, w_uv.to(torch.float32))
    roped_key = k_pe.to(torch.float32).unsqueeze(1).expand(-1, num_heads, -1, -1)
    key = torch.cat([k_nope, roped_key], dim=-1)
    query = torch.cat([q_nope, q_pe], dim=-1).to(torch.float32)

    # F.scaled_dot_product_attention rejects a mask together with is_causal, so
    # fold the band into the mask instead of passing both.
    mask = None if attention_mask is None else attention_mask.to(torch.float32)
    if is_causal:
        band = torch.zeros(
            (1, 1, query.size(2), key.size(2)), dtype=torch.float32
        ).masked_fill(
            ~torch.tril(torch.ones(query.size(2), key.size(2), dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0),
            float("-inf"),
        )
        mask = band if mask is None else mask + band

    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)
    return out.to(q_nope.dtype)


def _compare_attention(
    inputs, *, attention_mask=None, is_causal=False, scale=None, absorbed=None
):
    compare_with_cpu(
        _attention,
        *inputs,
        attention_mask,
        is_causal,
        scale,
        absorbed,
        run_eager=False,
    )


def _causal_mask(batch, seqlen_q, seqlen_kv, dtype=torch.float16):
    """Additive causal mask over physical cache columns, query-end aligned."""
    q_pos = torch.arange(seqlen_kv - seqlen_q, seqlen_kv).view(1, 1, seqlen_q, 1)
    k_pos = torch.arange(seqlen_kv).view(1, 1, 1, seqlen_kv)
    mask = torch.zeros((batch, 1, seqlen_q, seqlen_kv), dtype=dtype)
    return mask.masked_fill(k_pos > q_pos, float("-inf"))


@pytest.mark.parametrize(
    "case,match",
    [
        ("query_rank", "q_nope and q_pe must be rank 4"),
        ("cache_rank", "kv_latent and k_pe must be rank 3"),
        ("weight_rank", "w_uk and w_uv must be rank 3"),
        ("query_disagreement", "q_pe must agree with q_nope"),
        ("cache_batch", "latent cache batch.*must match query batch"),
        ("rope_length", "k_pe length.*must match kv_latent length"),
        ("rope_dim", "k_pe head dim.*must match q_pe head dim"),
        ("weight_heads", "up-projection weights must carry one block per query head"),
        ("weight_nope_dim", r"w_uk must be \[H, 128, R\]"),
        ("weight_latent_rank", "must map from the latent rank"),
        ("empty_cache", "query and cache lengths must be positive"),
        ("dtype", "all inputs must share one dtype"),
        ("mask_shape", "attention_mask shape.*must broadcast"),
        ("mask_dtype", "attention_mask dtype.*must match"),
        ("mask_device", "attention_mask device.*must match"),
        ("negative_scale", "scale=.*must be non-negative"),
    ],
)
def test_validation_guards(case, match):
    """Malformed calls fail at the public decomposition boundary."""
    batch, heads, seqlen_q, seqlen_kv = 1, 4, 64, 128
    scale = None

    def empty(*shape, dtype=torch.float16):
        return torch.empty(shape, dtype=dtype)

    q_nope = empty(batch, heads, seqlen_q, NOPE_DIM)
    q_pe = empty(batch, heads, seqlen_q, ROPE_DIM)
    kv_latent = empty(batch, seqlen_kv, LATENT_RANK)
    k_pe = empty(batch, seqlen_kv, ROPE_DIM)
    w_uk = empty(heads, NOPE_DIM, LATENT_RANK)
    w_uv = empty(heads, VALUE_DIM, LATENT_RANK)
    attention_mask = None

    if case == "query_rank":
        q_nope = empty(heads, seqlen_q, NOPE_DIM)
    elif case == "cache_rank":
        kv_latent = empty(batch, 1, seqlen_kv, LATENT_RANK)
    elif case == "weight_rank":
        w_uk = empty(NOPE_DIM, LATENT_RANK)
    elif case == "query_disagreement":
        q_pe = empty(batch, heads, seqlen_q + 1, ROPE_DIM)
    elif case == "cache_batch":
        kv_latent = empty(batch + 1, seqlen_kv, LATENT_RANK)
    elif case == "rope_length":
        k_pe = empty(batch, seqlen_kv + 64, ROPE_DIM)
    elif case == "rope_dim":
        k_pe = empty(batch, seqlen_kv, ROPE_DIM * 2)
    elif case == "weight_heads":
        w_uk = empty(heads + 1, NOPE_DIM, LATENT_RANK)
    elif case == "weight_nope_dim":
        w_uk = empty(heads, NOPE_DIM * 2, LATENT_RANK)
    elif case == "weight_latent_rank":
        w_uv = empty(heads, VALUE_DIM, LATENT_RANK * 2)
    elif case == "empty_cache":
        kv_latent = empty(batch, 0, LATENT_RANK)
        k_pe = empty(batch, 0, ROPE_DIM)
    elif case == "dtype":
        kv_latent = empty(batch, seqlen_kv, LATENT_RANK, dtype=torch.float32)
    elif case == "mask_shape":
        attention_mask = empty(batch, 1, seqlen_q, seqlen_kv - 1)
    elif case == "mask_dtype":
        attention_mask = empty(batch, 1, seqlen_q, seqlen_kv, dtype=torch.float32)
    elif case == "mask_device":
        attention_mask = torch.empty(
            (batch, 1, seqlen_q, seqlen_kv), dtype=torch.float16, device="meta"
        )
    elif case == "negative_scale":
        scale = -1.0

    with pytest.raises(Unsupported, match=match):
        spyre_multi_latent_attention(
            q_nope,
            q_pe,
            kv_latent,
            k_pe,
            w_uk,
            w_uv,
            attention_mask,
            False,
            scale,
        )


class TestPathSelection(unittest.TestCase):
    """Which path the shape-based default picks. No device needed."""

    def _prefer(self, seqlen_q, seqlen_kv):
        return _mla_prefer_absorbed(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            nope_dim=NOPE_DIM,
            rope_dim=ROPE_DIM,
            latent_rank=LATENT_RANK,
            value_dim=VALUE_DIM,
        )

    def test_decode_absorbs(self):
        # The case MLA exists for: one query row against a long cache.
        self.assertTrue(self._prefer(1, 4096))

    def test_full_prefill_materializes(self):
        # Every cached token is scored by thousands of queries, so the
        # per-token projection is amortized many times over.
        self.assertFalse(self._prefer(4096, 4096))

    def test_crossover_is_monotonic_in_query_length(self):
        # Growing the query can only move the choice toward materialization,
        # never back: a non-monotonic rule would thrash between paths across
        # chunk sizes and recompile each time.
        seqlen_kv = 4096
        choices = [self._prefer(seqlen_q, seqlen_kv) for seqlen_q in range(1, 512)]
        self.assertEqual(choices, sorted(choices, reverse=True))

    def test_crossover_is_near_the_published_dimensions(self):
        # Solving the two counts in _mla_prefer_absorbed for equality gives
        #   Lq = Lk * R * (Dn + Dv) / (R * (Dn + Dv) + Lk * (2R - Dn - Dv))
        # which rises toward R * (Dn + Dv) / (2R - Dn - Dv) = 512 * 256 / 768,
        # about 170 rows, as the context grows.
        seqlen_kv = 32768
        crossover = next(
            seqlen_q
            for seqlen_q in range(1, 1024)
            if not self._prefer(seqlen_q, seqlen_kv)
        )
        self.assertGreater(crossover, 150)
        self.assertLess(crossover, 190)

    def test_short_query_against_a_short_cache_absorbs(self):
        # A 16-row chunk cannot amortize projecting 64 cached tokens, even
        # though those projections are cheap in absolute terms.
        self.assertTrue(self._prefer(16, 64))

    def test_square_block_materializes(self):
        # Lq == Lk == 64 already crosses over: each cached token is scored by
        # 64 queries, which pays for projecting it once.
        self.assertFalse(self._prefer(64, 64))


class TestMultiLatentAttention(unittest.TestCase):
    """Shapes the op supports, against the materialized reference."""

    def setUp(self):
        torch._dynamo.reset()

    def test_decode(self):
        # One query row, 512 cached tokens: the shape the absorbed path is for.
        inputs = _inputs(1, 8, 1, 512)
        _compare_attention(inputs, absorbed=True)

    def test_decode_materialized_path_agrees(self):
        # Same inputs through the path the default would not pick. The two
        # paths are algebraically equal, so a divergence here is a bug in one
        # of them rather than a property of the shape.
        inputs = _inputs(1, 8, 1, 512)
        _compare_attention(inputs, absorbed=False)

    def test_decode_long_cache(self):
        inputs = _inputs(1, 8, 1, 4096)
        _compare_attention(inputs, absorbed=True)

    def test_decode_batch(self):
        inputs = _inputs(4, 8, 1, 512)
        _compare_attention(inputs, absorbed=True)

    def test_prefill_causal(self):
        inputs = _inputs(1, 8, 256, 256)
        _compare_attention(inputs, is_causal=True, absorbed=False)

    def test_prefill_causal_absorbed_path_agrees(self):
        inputs = _inputs(1, 8, 256, 256)
        _compare_attention(inputs, is_causal=True, absorbed=True)

    def test_prefill_batch(self):
        inputs = _inputs(2, 4, 128, 128)
        _compare_attention(inputs, is_causal=True, absorbed=False)

    def test_chunked_prefill_with_runtime_mask(self):
        # Lq < Lk: a chunk continuing a warm cache, positioned by mask values
        # rather than by shape.
        inputs = _inputs(1, 8, 64, 512)
        mask = _causal_mask(1, 64, 512)
        _compare_attention(inputs, attention_mask=mask, absorbed=True)

    def test_chunked_prefill_materialized_path_agrees(self):
        inputs = _inputs(1, 8, 64, 512)
        mask = _causal_mask(1, 64, 512)
        _compare_attention(inputs, attention_mask=mask, absorbed=False)

    def test_causal_and_runtime_mask_together(self):
        # Both mask sources at once: the band is broadcast [1, 1, Lq, Lk] and
        # the runtime mask is per-batch, so they tile on different axes.
        inputs = _inputs(2, 4, 128, 128)
        valid = torch.zeros((2, 1, 128, 128), dtype=torch.float16)
        valid[1, :, :, :16] = float("-inf")  # left padding in the second row
        _compare_attention(inputs, attention_mask=valid, is_causal=True, absorbed=False)

    def test_explicit_scale(self):
        # A model with a YaRN-adjusted softmax scale must not be silently
        # rescaled by the absorbed path's wider latent query.
        inputs = _inputs(1, 8, 1, 512)
        _compare_attention(inputs, scale=0.0884, absorbed=True)

    def test_default_path_matches_the_reference(self):
        # absorbed=None: whatever _mla_prefer_absorbed picks still has to be
        # the same function.
        inputs = _inputs(1, 8, 1, 512)
        _compare_attention(inputs)

    def test_single_head(self):
        inputs = _inputs(1, 1, 1, 512)
        _compare_attention(inputs, absorbed=True)

    def test_small_latent_rank(self):
        # R == Dn: the absorbed query is no wider than q_nope, so the path's
        # advantage disappears while its arithmetic must stay correct.
        inputs = _inputs(1, 4, 1, 256, latent_rank=128)
        _compare_attention(inputs, absorbed=True)

    def test_nondivisible_cache_extent_uses_one_counted_loop(self):
        # 2112 cached tokens pad to a whole number of equal blocks and are
        # scanned by a single counted loop rather than an unrolled chain.
        inputs = _inputs(1, 8, 1, 2112)
        expected = _attention(*inputs, None, False, None, True)

        actual, sources = run_and_get_code(
            torch.compile(_attention, dynamic=False),
            *[tensor.to("spyre") for tensor in inputs],
            None,
            False,
            None,
            True,
        )

        torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 1)
        self.assertNotIn("while_loop_carry_snapshot", "\n".join(sources))

    def test_fully_masked_row_is_zero_not_nan(self):
        # A padding row with no legal key: the clamped denominator has to
        # produce zeros rather than a NaN that spreads through the layer.
        inputs = _inputs(1, 4, 64, 128)
        mask = torch.zeros((1, 1, 64, 128), dtype=torch.float16)
        mask[:, :, 0, :] = float("-inf")

        actual = torch.compile(_attention, dynamic=False)(
            *[tensor.to("spyre") for tensor in inputs],
            mask.to("spyre"),
            False,
            None,
            True,
        )

        row = actual.cpu()[:, :, 0, :]
        self.assertFalse(torch.isnan(row).any())
        torch.testing.assert_close(row, torch.zeros_like(row))


if __name__ == "__main__":
    unittest.main()
