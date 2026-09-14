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

"""End-to-end correctness of spyre::sliding_window_attention.

Compared against the definition: full SDPA over the whole cache behind a band
mask. Unsupported shapes raise rather than falling back, so numbers coming out
at all prove the windowed path ran; which shapes are refused is settled in
test_kv_window.py without a device.

This does NOT establish that the spyre_hints produce device loops — untiled
code returns the right answer with one large intermediate.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py -v
"""

import unittest

import torch
import torch._dynamo
import torch.nn.functional as F

from utils_inductor import cached_randn, compare_with_cpu


def _band_mask(
    seqlen_q: int, seqlen_kv: int, window_size: int, dtype=torch.float16
) -> torch.Tensor:
    """Full [1, 1, Lq, Lkv] causal sliding-window mask -- the definition."""
    q_pos = torch.arange(seqlen_kv - seqlen_q, seqlen_kv).unsqueeze(-1)
    k_pos = torch.arange(seqlen_kv).unsqueeze(0)
    delta = q_pos - k_pos
    allowed = (delta >= 0) & (delta < window_size)
    mask = torch.zeros(seqlen_q, seqlen_kv, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _attention(q, k, v, window_size):
    """Dispatch: the op on spyre, the masked reference on CPU."""
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(q, k, v, window_size, True)
    mask = _band_mask(q.size(2), k.size(2), window_size)
    return F.scaled_dot_product_attention(
        q, k, v, mask, enable_gqa=q.size(1) != k.size(1)
    )


def _inputs(batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim=64):
    query = cached_randn(
        (batch, heads, seqlen_q, head_dim), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (batch, kvheads, seqlen_kv, head_dim), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (batch, kvheads, seqlen_kv, head_dim), differentiation=3, dtype=torch.float16
    )
    return query, key, value


def _compact_kv(batch, kvheads, capacity, cache_seqlen, head_dim=64):
    """[B, Hkv, capacity, E] key/value for a compact (rolled or still-filling)
    cache -- capacity rows physically allocated, only ``min(cache_seqlen,
    capacity)`` of them real.

    Real data fills ``[0, cache_seqlen)``; the rest is zero. For a rolled
    cache (``cache_seqlen > capacity``) that is every row -- the whole
    point, a buffer that has filled and is sliding forward. For a
    still-filling one it leaves ``[cache_seqlen, capacity)`` zero, matching
    the precondition on ``spyre::sliding_window_attention`` (an additive
    ``-inf`` mask cannot rescue a ``NaN`` score) and on ``spyre::kv_window``
    (rows stay contiguous and time-ordered, oldest dropped from the front).
    """
    written = min(cache_seqlen, capacity)
    key = torch.zeros((batch, kvheads, capacity, head_dim), dtype=torch.float16)
    value = torch.zeros((batch, kvheads, capacity, head_dim), dtype=torch.float16)
    if written > 0:
        key[:, :, :written, :] = cached_randn(
            (batch, kvheads, written, head_dim), differentiation=2, dtype=torch.float16
        )
        value[:, :, :written, :] = cached_randn(
            (batch, kvheads, written, head_dim), differentiation=3, dtype=torch.float16
        )
    return key, value


def _compact_kv_at(batch, kvheads, capacity, buffer_origin, cache_seqlen, head_dim=64):
    """Like ``_compact_kv`` but for a buffer whose physical row 0 holds
    ``buffer_origin`` rather than the exactly-full ``cache_seqlen -
    capacity``. Rows ``[0, cache_seqlen - buffer_origin)`` are real; the rest
    is the zero-filled tail an evictor working at block granularity leaves.
    """
    written = cache_seqlen - buffer_origin
    assert 0 < written <= capacity, "buffer_origin outside the plannable range"
    key = torch.zeros((batch, kvheads, capacity, head_dim), dtype=torch.float16)
    value = torch.zeros((batch, kvheads, capacity, head_dim), dtype=torch.float16)
    key[:, :, :written, :] = cached_randn(
        (batch, kvheads, written, head_dim), differentiation=2, dtype=torch.float16
    )
    value[:, :, :written, :] = cached_randn(
        (batch, kvheads, written, head_dim), differentiation=3, dtype=torch.float16
    )
    return key, value


def _rolled_reference(query, key, value, window_size, cache_seqlen, buffer_origin=None):
    """CPU reference for a compact cache: key/value are ``[B, Hkv, capacity,
    E]``, not the full-length cache ``_band_mask`` assumes.

    Physical row ``j`` holds logical position ``buffer_origin + j``
    (``buffer_origin = max(0, cache_seqlen - capacity)``), per
    ``spyre::kv_window``'s row-order precondition -- the band is built
    directly in that coordinate space.

    The ``k_pos < cache_seqlen`` term is redundant with the causal one here,
    and kept deliberately: ``spyre::window_band_mask`` drops it on the
    argument that the two can never disagree, so stating it independently
    means a regression in the production causal band shows up as a
    disagreement rather than a shared mistake.
    """
    capacity = key.size(2)
    seqlen_q = query.size(2)
    if buffer_origin is None:
        buffer_origin = max(0, cache_seqlen - capacity)
    q_pos = torch.arange(cache_seqlen - seqlen_q, cache_seqlen).unsqueeze(-1)
    k_pos = torch.arange(capacity, dtype=torch.int64) + buffer_origin
    delta = q_pos - k_pos.unsqueeze(0)
    allowed = (delta >= 0) & (delta < window_size) & (k_pos.unsqueeze(0) < cache_seqlen)
    mask = torch.zeros(seqlen_q, capacity, dtype=torch.float16)
    mask.masked_fill_(~allowed, float("-inf"))
    mask = mask.unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(
        query, key, value, mask, enable_gqa=query.size(1) != key.size(1)
    )


def _reference_with_valid_start(query, key, value, window_size, valid_start):
    """``_rolled_reference`` for an exactly-full buffer, additionally excluding
    physical rows below ``valid_start`` -- the left-padding an offset-and-length
    window cannot express.

    One threshold per batch entry, so ``valid_start`` is a list even when uniform.
    """
    seqlen_q, capacity = query.size(2), key.size(2)
    rows = torch.arange(seqlen_q) + (capacity - seqlen_q)
    columns = torch.arange(capacity)
    delta = rows.unsqueeze(-1) - columns.unsqueeze(0)
    allowed = (delta >= 0) & (delta < window_size)
    starts = torch.tensor(valid_start).view(-1, 1, 1)
    allowed = allowed.unsqueeze(0) & (columns.view(1, 1, -1) >= starts)
    mask = torch.zeros(allowed.shape, dtype=query.dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask.unsqueeze(1)
    )


def _valid_start_attention(q, k, v, window_size, valid_start):
    """The op with an explicit valid_start on spyre, the reference on CPU."""
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(
            q, k, v, window_size, True, None, k.size(2), 0, valid_start
        )
    return _reference_with_valid_start(q, k, v, window_size, valid_start)


def _rolled_attention(q, k, v, window_size, cache_seqlen, buffer_origin=None):
    """Dispatch for a compact cache: the op on spyre with an explicit
    ``cache_seqlen`` (it cannot default to ``k.size(2)`` here -- that IS the
    distinction under test), the compact reference on CPU.
    """
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(
            q, k, v, window_size, True, None, cache_seqlen, buffer_origin
        )
    return _rolled_reference(q, k, v, window_size, cache_seqlen, buffer_origin)


class TestSlidingWindowAttention(unittest.TestCase):
    """Shapes the op supports, against the masked reference."""

    def setUp(self):
        torch._dynamo.reset()

    def test_prefill_mha(self):
        # 4 blocks of 64, a 128-row window each.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_mha_wider_window(self):
        # W=128 -> a 192-row window.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_prefill_gqa(self):
        # 8 query heads from 2 kv heads; the expand is inside the op.
        query, key, value = _inputs(1, 8, 2, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_batch(self):
        query, key, value = _inputs(2, 4, 4, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_head_dim_128(self):
        # Two sticks per row where 64 is one; the placement is in rows.
        query, key, value = _inputs(1, 8, 8, 256, 256, head_dim=128)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_long(self):
        # 32 blocks — a long unrolled loop rather than a handful.
        query, key, value = _inputs(1, 8, 8, 2048, 2048)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode(self):
        # One block reading exactly W rows: 64 of 4096.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode_gqa(self):
        query, key, value = _inputs(1, 8, 2, 1, 512)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_decode_long_cache(self):
        query, key, value = _inputs(1, 8, 8, 1, 8192)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_chunked_prefill(self):
        # Lq < Lkv: prefill continuing a warm cache.
        query, key, value = _inputs(1, 8, 8, 128, 512)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_query_length_not_a_multiple_of_the_block(self):
        # Lq=100 padded to 128 at the front. Back-padding would shift every
        # real row 28 positions and this would catch it.
        query, key, value = _inputs(1, 8, 8, 100, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode_window_not_a_multiple_of_the_stick(self):
        # The only decode case where the band add is emitted: W=64/128 mask
        # nothing and skip it.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)

    def test_window_not_a_multiple_of_the_stick(self):
        # W=100: buffer rounds up to a stick, band masks by the true window.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)

    def test_window_covering_the_whole_cache(self):
        # buffer_width == seqlen_kv: degenerate, not a separate code path.
        query, key, value = _inputs(1, 8, 8, 128, 128)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_explicit_cache_seqlen_matches_the_default(self):
        # cache_seqlen defaults to the cache's allocated rows, so passing that
        # same number explicitly must not move a single window.
        query, key, value = _inputs(1, 8, 8, 128, 512)

        def attention(q, k, v, window_size):
            if q.device.type == "spyre":
                return torch.ops.spyre.sliding_window_attention(
                    q, k, v, window_size, True, None, k.size(2)
                )
            mask = _band_mask(q.size(2), k.size(2), window_size)
            return F.scaled_dot_product_attention(q, k, v, mask)

        compare_with_cpu(attention, query, key, value, 64, run_eager=False)

    def test_ragged_query_and_window_together(self):
        # An off-by-one in the pad arithmetic can survive either alone.
        query, key, value = _inputs(1, 8, 2, 100, 512)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)

    def test_prefill_head_dim_256_gqa(self):
        # Gemma 4's sliding layers: 16 query heads from 8 KV heads, head_dim 256,
        # W=1024. head_dim 256 is four sticks per row where the rest of this file
        # uses one or two, and kv_window hands back a transposed slice.
        query, key, value = _inputs(1, 16, 8, 512, 512, head_dim=256)
        compare_with_cpu(_attention, query, key, value, 1024, run_eager=False)


class TestCompactCache(unittest.TestCase):
    """cache_seqlen != key.size(2): a compact cache, rolled or still filling.

    These check end-to-end numeric correctness of placement + masking +
    attention against a compact (not full-length) cache. They do NOT isolate
    the unwritten-tail question: an overshooting buffer is excluded by the
    causal term alone, since window_band_mask carries no cache_seqlen term.
    """

    def setUp(self):
        torch._dynamo.reset()

    def test_rolled_decode_at_a_non_aligned_position(self):
        # The design's goal: a compact rolled buffer read at an arbitrary
        # (non-stick) logical position. read_start is identical for every
        # such position -- TestArbitraryCacheSeqlen sweeps that; this is one
        # point on the line, verified end to end.
        batch, heads, kvheads = 1, 8, 8
        capacity, cache_seqlen, window = 4160, 5001, 4096
        key, value = _compact_kv(batch, kvheads, capacity, cache_seqlen)
        query = cached_randn(
            (batch, heads, 1, 64), differentiation=1, dtype=torch.float16
        )
        compare_with_cpu(
            _rolled_attention, query, key, value, window, cache_seqlen, run_eager=False
        )

    def test_warmup_cache_at_a_non_aligned_seqlen(self):
        # cache_seqlen=100 < capacity=256, not stick-aligned: the buffer
        # reaches column 128, past what is written (rows [100, 128) are the
        # zero-filled tail). Checks that placement and attention around that
        # overshoot are numerically correct end to end.
        batch, heads, kvheads = 1, 8, 8
        capacity, cache_seqlen, window = 256, 100, 64
        key, value = _compact_kv(batch, kvheads, capacity, cache_seqlen)
        query = cached_randn(
            (batch, heads, 1, 64), differentiation=1, dtype=torch.float16
        )
        compare_with_cpu(
            _rolled_attention, query, key, value, window, cache_seqlen, run_eager=False
        )

    def test_capacity_equals_window_decode(self):
        # HF's StaticSlidingWindowLayer geometry: exactly window_size rows for
        # decode, the minimal allocation this op requires. read_start
        # collapses to 0 once capacity == buffer_width, at any position.
        batch, heads, kvheads = 1, 8, 8
        capacity = window = 64
        cache_seqlen = 5001
        key, value = _compact_kv(batch, kvheads, capacity, cache_seqlen)
        query = cached_randn(
            (batch, heads, 1, 64), differentiation=1, dtype=torch.float16
        )
        compare_with_cpu(
            _rolled_attention, query, key, value, window, cache_seqlen, run_eager=False
        )

    def test_block_granular_eviction_with_an_explicit_buffer_origin(self):
        # A buffer that is NOT exactly full: an evictor freeing whole 64-row
        # blocks keeps everything from logical 896 on, so physical row 0 holds
        # 896 rather than the default's 1000-256=744. Rows stay contiguous and
        # time-ordered, so kv_window's ordering precondition holds -- only the
        # origin differs, and passing it is what keeps the read on real data.
        batch, heads, kvheads = 1, 8, 8
        capacity, cache_seqlen, window = 256, 1000, 64
        buffer_origin = 896
        key, value = _compact_kv_at(
            batch, kvheads, capacity, buffer_origin, cache_seqlen
        )
        query = cached_randn(
            (batch, heads, 1, 64), differentiation=1, dtype=torch.float16
        )
        compare_with_cpu(
            _rolled_attention,
            query,
            key,
            value,
            window,
            cache_seqlen,
            buffer_origin,
            run_eager=False,
        )

    def test_multiblock_rolled_prefill_with_distinct_read_starts(self):
        # 8 blocks of a 512-row prefill against a rolled, non-aligned
        # cache_seqlen -- every block reads a different physical offset
        # (448, 512, ..., 896 here). Window staggering on top of the
        # physical-space floor and the earliest-block-reach check, together.
        batch, heads, kvheads = 1, 8, 8
        capacity, cache_seqlen, window = 1024, 5001, 64
        seqlen_q = 512
        key, value = _compact_kv(batch, kvheads, capacity, cache_seqlen)
        query = cached_randn(
            (batch, heads, seqlen_q, 64), differentiation=1, dtype=torch.float16
        )
        compare_with_cpu(
            _rolled_attention, query, key, value, window, cache_seqlen, run_eager=False
        )

    def test_anchored_decode_gemma4(self):
        # The exact geometry hf-adapters will call every decode step: a 1088-row
        # compact buffer declared exactly full, and a single query row
        # (seqlen_q=1) fed straight to the op. The anchored design dropped the
        # 64-row query stick, so the real decode shape is (1, 16, 1, 256), with
        # head_dim 256 and W=1024. This is the one shape that must be right for
        # the integration to work at all.
        key, value = _compact_kv(1, 8, 1088, 1088, head_dim=256)
        query = cached_randn((1, 16, 1, 256), differentiation=1, dtype=torch.float16)
        compare_with_cpu(
            _rolled_attention, query, key, value, 1024, 1088, run_eager=False
        )

    def test_valid_start_excludes_padded_columns(self):
        # 17 rows of left padding inside the window: without valid_start they are
        # attended, and the reference proves the difference is visible.
        key, value = _compact_kv(1, 8, 1088, 1088)
        query = cached_randn((1, 8, 64, 64), differentiation=1, dtype=torch.float16)
        compare_with_cpu(
            _valid_start_attention, query, key, value, 1024, [17], run_eager=False
        )

    def test_valid_start_per_sequence(self):
        # Ragged batch: the band widens to [B, 1, q, W'] only for this case.
        key, value = _compact_kv(2, 8, 1088, 1088)
        query = cached_randn((2, 8, 64, 64), differentiation=1, dtype=torch.float16)
        compare_with_cpu(
            _valid_start_attention, query, key, value, 1024, [0, 40], run_eager=False
        )

    def test_all_zero_valid_start_matches_no_valid_start(self):
        # The fast path must be numerically identical to passing nothing.
        key, value = _compact_kv(1, 8, 1088, 1088)
        query = cached_randn((1, 8, 64, 64), differentiation=1, dtype=torch.float16)

        def attention(q, k, v, window_size):
            if q.device.type == "spyre":
                return torch.ops.spyre.sliding_window_attention(
                    q, k, v, window_size, True, None, k.size(2), 0, [0]
                )
            return _rolled_reference(q, k, v, window_size, k.size(2))

        compare_with_cpu(attention, query, key, value, 1024, run_eager=False)


if __name__ == "__main__":
    unittest.main()
