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

Tests that require structural guarantees also inspect the generated code for
the counted loops produced by ``for_each_tile``.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py -v
"""

import unittest

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor import config

from torch_spyre._inductor.decompositions import spyre_sliding_window_attention
from torch_spyre._inductor.errors import Unsupported
from utils_inductor import cached_randn, compare_with_cpu


def _attention_mask(
    batch,
    seqlen_q,
    capacity,
    window_size,
    *,
    query_end=None,
    buffer_origin=0,
    valid_start=None,
    dtype=torch.float16,
):
    """Runtime causal-window mask in physical cache coordinates."""
    query_end = capacity if query_end is None else query_end
    q_pos = torch.arange(query_end - seqlen_q, query_end).view(1, seqlen_q, 1)
    k_pos = torch.arange(capacity).view(1, 1, capacity) + buffer_origin
    allowed = (q_pos >= k_pos) & (q_pos - k_pos < window_size)
    if valid_start is not None:
        starts = torch.tensor(valid_start).view(batch, 1, 1)
        allowed = allowed & (k_pos >= starts)
    allowed = allowed.expand(batch, -1, -1)

    # Padding rows can have no legal key. Their outputs are discarded, but the
    # kernel still needs a defined softmax so they cannot poison later layers.
    has_attendable_key = allowed.any(dim=-1, keepdim=True)
    first_column = torch.arange(capacity).view(1, 1, capacity) == 0
    allowed = allowed | (~has_attendable_key & first_column)
    mask = torch.zeros((batch, seqlen_q, capacity), dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(1)


def _attention(q, k, v, attention_mask, window_size, scale=None, is_causal=True):
    """Dispatch: the runtime-mask op on Spyre, masked SDPA on CPU."""
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(
            q, k, v, attention_mask, window_size, is_causal, scale
        )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attention_mask,
        scale=scale,
        enable_gqa=q.size(1) != k.size(1),
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


def _to_cache_position_first(tensor):
    """Move a KV cache with the device layout used by model cache updates.

    The logical tensor remains ``[B, Hkv, capacity, D]``, but the device axes
    are ``[capacity, Hkv, ceil(D / eps), B, eps]``. Their stride map is the
    corresponding row-major logical stride ``[D, capacity*D, eps,
    Hkv*capacity*D, 1]``. This is the same mapping constructed by
    ``hf_common._cache_position_first_stl``: putting cache position at device
    dim 0 lets the model's logical-dim-2 ``index_copy`` update rows without a
    cache relayout before attention reads them.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    batch, kvheads, capacity, head_dim = tensor.shape
    eps = SpyreTensorLayout(list(tensor.shape), tensor.dtype).elems_per_stick()
    layout = SpyreTensorLayout(
        device_size=[capacity, kvheads, (head_dim + eps - 1) // eps, batch, eps],
        stride_map=[
            head_dim,
            capacity * head_dim,
            eps,
            kvheads * capacity * head_dim,
            1,
        ],
        device_dtype=get_device_dtype(tensor.dtype),
    )
    return tensor.to("spyre", device_layout=layout)


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


def _decode_mask(batch, capacity, write_row, window_size, valid_start, dtype):
    """Fixed-shape runtime mask over physical rows of an anchored cache."""
    columns = torch.arange(capacity).view(1, 1, 1, capacity)
    starts = torch.tensor(valid_start).view(batch, 1, 1, 1)
    allowed = (
        (columns <= write_row)
        & (columns > write_row - window_size)
        & (columns >= starts)
    )
    mask = torch.zeros((batch, 1, 1, capacity), dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


def _runtime_mask_attention(q, k, v, window_size, scale, decode_mask):
    """Compatibility wrapper with scalar arguments after tensor inputs."""
    return _attention(q, k, v, decode_mask, window_size, scale)


def _compare_attention(
    query,
    key,
    value,
    window_size,
    *,
    query_end=None,
    buffer_origin=0,
    valid_start=None,
    scale=None,
):
    mask = _attention_mask(
        query.size(0),
        query.size(2),
        key.size(2),
        window_size,
        query_end=query_end,
        buffer_origin=buffer_origin,
        valid_start=valid_start,
        dtype=query.dtype,
    )
    compare_with_cpu(
        _attention,
        query,
        key,
        value,
        mask,
        window_size,
        scale,
        run_eager=False,
    )


@pytest.mark.parametrize(
    "case,match",
    [
        ("negative_scale", "scale=.*must be non-negative"),
        ("nonpositive_window", "window_size=.*must be positive"),
        ("empty_query", "query and cache lengths must be positive"),
        ("empty_cache", "query and cache lengths must be positive"),
        ("query_exceeds_cache", "seqlen_q=.*exceeds cache_capacity"),
        ("unaligned_cache", "cache_capacity=.*must be a multiple"),
        ("nondivisible_heads", "query heads must be a whole multiple"),
        ("mask_shape", "attention_mask shape.*must be"),
        ("mask_dtype", "attention_mask dtype.*must match"),
        ("mask_device", "attention_mask device.*must match"),
    ],
)
def test_validation_guards(case, match):
    """Malformed calls fail at the public decomposition boundary."""
    batch, num_heads, num_kvheads = 1, 8, 2
    seqlen_q, cache_capacity, head_dim = 64, 64, 64
    window_size, scale = 64, None

    if case == "negative_scale":
        scale = -1.0
    elif case == "nonpositive_window":
        window_size = 0
    elif case == "empty_query":
        seqlen_q = 0
    elif case == "empty_cache":
        cache_capacity = 0
    elif case == "query_exceeds_cache":
        seqlen_q = 128
    elif case == "unaligned_cache":
        cache_capacity = 65
    elif case == "nondivisible_heads":
        num_kvheads = 3

    query = torch.empty(batch, num_heads, seqlen_q, head_dim, dtype=torch.float16)
    key = torch.empty(batch, num_kvheads, cache_capacity, head_dim, dtype=torch.float16)
    value = torch.empty_like(key)
    mask_shape = (batch, 1, seqlen_q, cache_capacity)
    if case == "mask_shape":
        mask_shape = (batch, 1, seqlen_q, max(0, cache_capacity - 1))
    mask_dtype = torch.float32 if case == "mask_dtype" else torch.float16
    mask_device = "meta" if case == "mask_device" else "cpu"
    attention_mask = torch.empty(mask_shape, dtype=mask_dtype, device=mask_device)

    with pytest.raises(Unsupported, match=match):
        spyre_sliding_window_attention(
            query,
            key,
            value,
            attention_mask,
            window_size,
            True,
            scale,
        )


class TestSlidingWindowAttention(unittest.TestCase):
    """Shapes the op supports, against the masked reference."""

    def setUp(self):
        torch._dynamo.reset()

    def test_prefill_mha(self):
        # 4 blocks of 64, a 128-row window each.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        _compare_attention(query, key, value, 64)

    def test_prefill_mha_wider_window(self):
        # W=128 -> a 192-row window.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        _compare_attention(query, key, value, 128)

    def test_prefill_gqa(self):
        # 8 query heads from 2 kv heads; the expand is inside the op.
        query, key, value = _inputs(1, 8, 2, 256, 256)
        _compare_attention(query, key, value, 64)

    def test_prefill_transposed_query_view(self):
        # Q projections arrive physically as [B, Lq, H, D] and are viewed as
        # [B, H, Lq, D]. The tiled window must follow logical, not stride, axes.
        query = cached_randn(
            (1, 256, 8, 64), differentiation=1, dtype=torch.float16
        ).transpose(1, 2)
        _, key, value = _inputs(1, 8, 8, 256, 256)
        _compare_attention(query, key, value, 64)

    def test_prefill_batch(self):
        query, key, value = _inputs(2, 4, 4, 256, 256)
        _compare_attention(query, key, value, 64)

    def test_prefill_batch_gqa(self):
        # Batch > 1 makes the unit GQA mask axis load-bearing: a rank-4 mask
        # cannot broadcast correctly against [B, Hkv, group, Lq, Lk].
        query, key, value = _inputs(2, 8, 2, 256, 256)
        _compare_attention(query, key, value, 64)

    def test_prefill_head_dim_128(self):
        # Two sticks per row where 64 is one; the placement is in rows.
        query, key, value = _inputs(1, 8, 8, 256, 256, head_dim=128)
        _compare_attention(query, key, value, 64)

    @unittest.skip("Temporarily disabled: slow SWA compile; re-enable after #4610")
    @config.patch({"cpsat_time_limit_seconds": 30})
    def test_prefill_long(self):
        # 32 blocks — a long unrolled loop rather than a handful.
        query, key, value = _inputs(1, 8, 8, 2048, 2048)
        _compare_attention(query, key, value, 64)

    def test_decode(self):
        # One block reading exactly W rows: 64 of 4096.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        _compare_attention(query, key, value, 64)

    def test_decode_gqa(self):
        query, key, value = _inputs(1, 8, 2, 1, 512)
        _compare_attention(query, key, value, 128)

    @unittest.skip("Temporarily disabled: slow SWA compile; re-enable after #4610")
    def test_decode_long_cache(self):
        query, key, value = _inputs(1, 8, 8, 1, 8192)
        _compare_attention(query, key, value, 64)

    def test_chunked_prefill(self):
        # Lq < Lkv: prefill continuing a warm cache.
        query, key, value = _inputs(1, 8, 8, 128, 512)
        _compare_attention(query, key, value, 64)

    def test_query_length_not_a_multiple_of_the_block(self):
        # Lq=100 padded to 128 at the front. Back-padding would shift every
        # real row 28 positions and this would catch it.
        query, key, value = _inputs(1, 8, 8, 100, 256)
        _compare_attention(query, key, value, 64)

    def test_decode_window_not_a_multiple_of_the_stick(self):
        # The only decode case where the band add is emitted: W=64/128 mask
        # nothing and skip it.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        _compare_attention(query, key, value, 100)

    def test_window_not_a_multiple_of_the_stick(self):
        # W=100: buffer rounds up to a stick, band masks by the true window.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        _compare_attention(query, key, value, 100)

    def test_window_covering_the_whole_cache(self):
        # buffer_width == seqlen_kv: degenerate, not a separate code path.
        query, key, value = _inputs(1, 8, 8, 128, 128)
        _compare_attention(query, key, value, 128)

    @unittest.skip("Temporarily disabled pending the SWA solver fixes in #4610")
    def test_ragged_query_and_window_together(self):
        # An off-by-one in the pad arithmetic can survive either alone.
        query, key, value = _inputs(1, 8, 2, 100, 512)
        _compare_attention(query, key, value, 100)

    def test_prefill_head_dim_256_gqa(self):
        # Gemma 4's sliding layers: 16 query heads from 8 KV heads, head_dim 256,
        # W=1024. head_dim 256 is four sticks per row where the rest of this file
        # uses one or two, and each K window is transposed tile-by-tile.
        query, key, value = _inputs(1, 16, 8, 512, 512, head_dim=256)
        _compare_attention(query, key, value, 1024)

    def test_nondivisible_kv_extent_uses_for_each_tile(self):
        """A nondivisible K/V extent is padded into one counted loop."""
        query, key, value = _inputs(1, 16, 8, 64, 1088)
        mask = _attention_mask(1, 64, 1088, 1024)
        expected = _attention(query, key, value, mask, 1024)

        actual, sources = run_and_get_code(
            torch.compile(_attention, dynamic=False),
            query.to("spyre"),
            key.to("spyre"),
            value.to("spyre"),
            mask.to("spyre"),
            1024,
        )

        torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 1)
        self.assertNotIn("while_loop_carry_snapshot", "\n".join(sources))

    def test_prefill_reads_a_prefix_of_a_larger_cache(self):
        # A short prefill can use the compact decode allocation already. Its KV
        # slice has a larger backing stride than its 64-row logical width.
        key, value = _compact_kv(1, 2, 1088, 64, head_dim=256)
        query = cached_randn((1, 4, 64, 256), differentiation=1, dtype=torch.float16)
        _compare_attention(query, key, value, 1024, query_end=64)

    def test_prefill_from_pinned_larger_cache_with_left_padding(self):
        # Model caches pin the sequence dimension outermost for indirect writes.
        # Exercise the fused recurrence with both the model's pinned cache layout
        # and a load-bearing left-padding band at head_dim=256.
        key, value = _compact_kv(1, 4, 256, 128, head_dim=256)
        query = cached_randn((1, 4, 128, 256), differentiation=1, dtype=torch.float16)
        mask = _attention_mask(1, 128, 256, 128, query_end=128, valid_start=[17])
        expected = _attention(query, key, value, mask, 128)

        compiled = torch.compile(_attention, backend="inductor")
        actual = compiled(
            query.to("spyre"),
            _to_cache_position_first(key),
            _to_cache_position_first(value),
            mask.to("spyre"),
            128,
        ).cpu()

        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    def test_prefill_from_pinned_larger_cache_without_left_padding(self):
        """Separate pinned-cache layout from the load-bearing padding band."""
        key, value = _compact_kv(1, 4, 256, 128, head_dim=256)
        query = cached_randn((1, 4, 128, 256), differentiation=1, dtype=torch.float16)
        mask = _attention_mask(1, 128, 256, 128, query_end=128)
        expected = _attention(query, key, value, mask, 128)

        compiled = torch.compile(_attention, backend="inductor")
        actual = compiled(
            query.to("spyre"),
            _to_cache_position_first(key),
            _to_cache_position_first(value),
            mask.to("spyre"),
            128,
        ).cpu()

        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    def test_prefill_left_padding_head_dim_256(self):
        """Separate the load-bearing padding band from the pinned-cache layout."""
        key, value = _compact_kv(1, 4, 256, 128, head_dim=256)
        query = cached_randn((1, 4, 128, 256), differentiation=1, dtype=torch.float16)
        _compare_attention(query, key, value, 128, query_end=128, valid_start=[17])

    def test_non_causal_prefill_reads_future_keys_outside_the_causal_plan(self):
        # A bidirectional vision block can make an early query attend a much
        # later key. The causal plan for block 0 reads only rows [0, 128), so
        # allowing row 255 proves the generic path scans the complete cache.
        query, key, value = _inputs(1, 8, 2, 256, 256)
        mask = _attention_mask(1, 256, 256, 64)
        mask[0, 0, 0, 255] = 0
        expected = _attention(query, key, value, mask, 64, None, False)

        compiled = torch.compile(_attention, backend="inductor")
        actual = compiled(
            query.to("spyre"),
            key.to("spyre"),
            value.to("spyre"),
            mask.to("spyre"),
            64,
            None,
            False,
        ).cpu()

        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)


class TestCompactCache(unittest.TestCase):
    """Runtime masks over compact caches with changing logical positions."""

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
        _compare_attention(
            query,
            key,
            value,
            window,
            query_end=cache_seqlen,
            buffer_origin=cache_seqlen - capacity,
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
        _compare_attention(query, key, value, window, query_end=cache_seqlen)

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
        _compare_attention(
            query,
            key,
            value,
            window,
            query_end=cache_seqlen,
            buffer_origin=cache_seqlen - capacity,
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
        _compare_attention(
            query,
            key,
            value,
            window,
            query_end=cache_seqlen,
            buffer_origin=buffer_origin,
        )

    @unittest.skip("Temporarily disabled: slow SWA compile; re-enable after #4610")
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
        _compare_attention(
            query,
            key,
            value,
            window,
            query_end=cache_seqlen,
            buffer_origin=cache_seqlen - capacity,
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
        _compare_attention(query, key, value, 1024, query_end=1088)

    def test_anchored_decode_gemma3(self):
        # Gemma 3's production compact cache is one 576-row calibrated block.
        key, value = _compact_kv(1, 4, 576, 576, head_dim=256)
        query = cached_randn((1, 8, 1, 256), differentiation=1, dtype=torch.float16)
        _compare_attention(query, key, value, 512, query_end=576)

    def test_runtime_decode_mask_reuses_one_graph_as_values_change(self):
        """Position and padding travel as tensor data, never Python guards."""
        batch, heads, kvheads, capacity, window = 2, 8, 2, 256, 128
        query, key, value = _inputs(batch, heads, kvheads, 1, capacity)
        masks = [
            _decode_mask(batch, capacity, 128, window, [0, 17], query.dtype),
            _decode_mask(batch, capacity, 191, window, [0, 3], query.dtype),
            _decode_mask(batch, capacity, 255, window, [0, 0], query.dtype),
        ]
        expected = [
            _runtime_mask_attention(query, key, value, window, 1.0, mask)
            for mask in masks
        ]

        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(_runtime_mask_attention, backend="inductor")
        device_args = [tensor.to("spyre") for tensor in (query, key, value)]
        actual = [
            compiled(*device_args, window, 1.0, mask.to("spyre")).cpu()
            for mask in masks
        ]

        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0.1, rtol=0.1)

    def test_runtime_prefill_mask_reuses_one_graph_as_values_change(self):
        """Chunk origin and per-sequence padding stay out of Python guards."""
        batch, heads, kvheads, seqlen_q, capacity, window = 2, 8, 2, 64, 128, 64
        query, key, value = _inputs(batch, heads, kvheads, seqlen_q, capacity)
        masks = [
            _attention_mask(
                batch,
                seqlen_q,
                capacity,
                window,
                query_end=64,
                valid_start=[0, 17],
            ),
            _attention_mask(
                batch,
                seqlen_q,
                capacity,
                window,
                query_end=128,
                valid_start=[0, 3],
            ),
        ]
        expected = [_attention(query, key, value, mask, window) for mask in masks]

        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(_attention, backend="inductor")
        device_args = [tensor.to("spyre") for tensor in (query, key, value)]
        actual = [
            compiled(*device_args, mask.to("spyre"), window).cpu() for mask in masks
        ]

        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0.1, rtol=0.1)

    def test_non_causal_prefill_mask_reuses_one_graph_as_values_change(self):
        """Different bidirectional regions remain runtime tensor data."""
        batch, heads, kvheads, seqlen, window = 1, 8, 2, 256, 64
        query, key, value = _inputs(batch, heads, kvheads, seqlen, seqlen)
        masks = []
        for future_key in (191, 255):
            mask = _attention_mask(batch, seqlen, seqlen, window)
            mask[0, 0, 0, future_key] = 0
            masks.append(mask)
        expected = [
            _attention(query, key, value, mask, window, None, False) for mask in masks
        ]

        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(_attention, backend="inductor")
        device_args = [tensor.to("spyre") for tensor in (query, key, value)]
        actual = [
            compiled(
                *device_args,
                mask.to("spyre"),
                window,
                None,
                False,
            ).cpu()
            for mask in masks
        ]

        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0.1, rtol=0.1)

    def test_runtime_mask_skips_fully_masked_leading_chunks(self):
        # A late chunked-prefill block can have no valid key in the first 512-row
        # KV chunk. The online softmax must carry zero weight across that chunk,
        # not form -inf - -inf and poison the later live window with NaNs.
        batch, heads, kvheads, seqlen_q, capacity, window = 1, 8, 2, 64, 1024, 64
        query, key, value = _inputs(batch, heads, kvheads, seqlen_q, capacity)
        mask = _attention_mask(
            batch,
            seqlen_q,
            capacity,
            window,
            query_end=capacity,
        )
        expected = _attention(query, key, value, mask, window)

        compiled = torch.compile(_attention, backend="inductor")
        actual = compiled(
            query.to("spyre"),
            key.to("spyre"),
            value.to("spyre"),
            mask.to("spyre"),
            window,
        ).cpu()

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    def test_runtime_mask_excludes_padded_columns(self):
        # Position and left padding are both runtime tensor data.
        key, value = _compact_kv(1, 8, 1088, 1088)
        query = cached_randn((1, 8, 64, 64), differentiation=1, dtype=torch.float16)
        _compare_attention(query, key, value, 1024, valid_start=[17])

    def test_runtime_mask_prefill_padding_rows_stay_finite(self):
        # The first 17 query rows are left padding. Each receives a harmless
        # diagonal rather than an all--inf band, so it cannot poison later layers
        # with NaN K/V values; callers discard or zero these query outputs.
        key, value = _compact_kv(1, 8, 1088, 64)
        query = cached_randn((1, 8, 64, 64), differentiation=1, dtype=torch.float16)
        _compare_attention(query, key, value, 1024, query_end=64, valid_start=[17])

    def test_runtime_mask_per_sequence_padding(self):
        # Each batch entry carries its own padding threshold in tensor data.
        key, value = _compact_kv(2, 8, 1088, 1088)
        query = cached_randn((2, 8, 64, 64), differentiation=1, dtype=torch.float16)
        _compare_attention(query, key, value, 1024, valid_start=[0, 40])


if __name__ == "__main__":
    unittest.main()
