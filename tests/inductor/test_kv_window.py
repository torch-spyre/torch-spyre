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

"""The window placement, the algorithm it implies, and the op that reads it.

Three levels, in the order a failure is worth reading:

  1. PLACEMENT -- pure integer arithmetic, no device. Where each Q block reads
     from and how wide its buffer is. Everything else rests on one invariant:
     every query row's window lies inside the buffer its block reads.

  2. ALGORITHM -- on CPU: does attending against the compact buffer compute
     the same thing as masking the full cache? Two claims, kept apart because
     they say different things. The set of unmasked columns is *identical*
     (combinatorial, no tolerance) -- that is what licenses the approach. The
     output values agree -- that only says the arithmetic was not fumbled.

  3. THE OP -- spyre::kv_window on device, against slices built
     independently here.

Bit-exactness is not available at level 2 and must not be asserted: the
windowed softmax reduces buffer_width terms where the reference reduces
seqlen_kv, and sum blocks its reduction differently at different widths.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_kv_window.py -v
"""

import math

import pytest
import torch

from torch_spyre._inductor.sliding_window_plan import (
    STICK,
    SlidingWindowPlan,
    band_batch,
    band_valid_start,
    check_valid_start,
    check_window_read,
    default_buffer_origin,
    plan_sliding_window,
    query_blocking,
    rejection_reason,
)
from utils_inductor import cached_randn, compare_with_cpu

# (seqlen_q, seqlen_kv, window): prefill, a window wider than one block, a warm
# cache, and decode against a long one.
SHAPES = [
    (512, 512, 64),
    (512, 512, 256),
    (256, 256, 64),
    (128, 512, 64),
    (1, 4096, 64),
    (512, 512, 100),  # window not a whole number of sticks
]
IDS = [f"q{q}_kv{kv}_w{w}" for q, kv, w in SHAPES]

BATCH, HEADS, HEAD_DIM = 1, 8, 64


def _plan(seqlen_q, seqlen_kv, window, q_block=None):
    q_block = query_blocking(seqlen_q)[0] if q_block is None else q_block
    plan = plan_sliding_window(seqlen_q, seqlen_kv, window, q_block=q_block)
    assert plan is not None, f"unsupported: {seqlen_q}/{seqlen_kv}/{window}"
    return plan


# --------------------------------------------------------------- 1. placement


class TestPlacement:
    """Where each block reads, and how wide its buffer is."""

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_every_row_window_lies_inside_its_buffer(self, seqlen_q, seqlen_kv, window):
        # The invariant everything downstream rests on. A row whose window
        # escaped its buffer would attend to columns that were never read.
        plan = _plan(seqlen_q, seqlen_kv, window)
        for qi in range(plan.num_q_blocks):
            start = plan.read_start(qi)
            q_start, q_end = plan.block_q_range(qi)
            for q_index in range(q_start, q_end):
                low, high = plan.row_window(q_index)
                assert start <= low and high <= start + plan.buffer_width, (
                    f"block {qi} row {q_index}: window [{low},{high}) escapes "
                    f"buffer [{start},{start + plan.buffer_width})"
                )

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_reads_are_stick_aligned_and_in_bounds(self, seqlen_q, seqlen_kv, window):
        # Unaligned would force a restickify; out of bounds would fault.
        plan = _plan(seqlen_q, seqlen_kv, window)
        starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
        for start in starts:
            assert start % STICK == 0
            assert 0 <= start and start + plan.buffer_width <= seqlen_kv
        assert starts == sorted(starts), "a window may not move backwards"

    def test_prefill_width_is_window_plus_one_block(self):
        # Rows inside a block have staggered windows spanning W + q_block - 1
        # columns, so the buffer is W + q_block. Assuming W is the mistake this
        # arithmetic exists to prevent.
        assert _plan(512, 512, 64).buffer_width == 64 + STICK
        assert _plan(512, 512, 256).buffer_width == 256 + STICK
        # A window that is not a whole number of sticks is fine -- the buffer
        # rounds up to one. 100 + 64 = 164 -> 192.
        assert _plan(512, 512, 100).buffer_width == 192

    def test_decode_width_is_exactly_the_window(self):
        # A single row has no stagger, so decode needs no extra block.
        plan = _plan(1, 4096, 64)
        assert plan.buffer_width == 64
        assert plan.num_q_blocks == 1

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_the_buffer_is_narrower_than_the_cache(self, seqlen_q, seqlen_kv, window):
        # The point of the exercise. Decode is the extreme: 64 rows of 4096.
        assert _plan(seqlen_q, seqlen_kv, window).buffer_width < seqlen_kv

    def test_blocks_tile_the_query_exactly(self):
        plan = _plan(512, 512, 64)
        covered = [
            i for qi in range(plan.num_q_blocks) for i in range(*plan.block_q_range(qi))
        ]
        assert covered == list(range(512))

    def test_row_window_is_the_causal_band(self):
        # The definition: row i at cache coordinate c attends [c-W+1, c].
        plan = _plan(512, 512, 64)
        assert plan.row_window(100) == (37, 101)
        assert plan.row_window(0) == (0, 1)  # clamped at the cache start

    def test_a_window_covering_the_cache_still_plans(self):
        # buffer_width == seqlen_kv is degenerate, not invalid: the read start
        # clamps to 0, every block reads the whole cache, and the band masks
        # it. It saves nothing there but it is not wrong, so it stays on the
        # one code path rather than being refused. Safe only because seqlen_kv
        # is required to be stick aligned -- otherwise ceil_stick could push
        # the buffer past the cache and the clamp negative.
        plan = _plan(128, 128, 128)
        assert plan.buffer_width == 128
        assert all(plan.read_start(n) == 0 for n in range(plan.num_q_blocks))

    def test_decode_masks_nothing_but_prefill_does(self):
        # What lets the body skip the band add at decode -- and only with a
        # stick-aligned window. An unaligned one floors the buffer origin below
        # the row's window start, so the band masks the difference and the add
        # is emitted after all. The body's skip is conditional on this, so both
        # directions are pinned.
        assert _plan(1, 4096, 64).block_is_fully_attended(0)
        assert _plan(1, 4096, 128).block_is_fully_attended(0)
        assert not _plan(1, 4096, 100).block_is_fully_attended(0)
        assert not _plan(1, 4096, 63).block_is_fully_attended(0)
        prefill = _plan(256, 256, 64)
        assert not any(
            prefill.block_is_fully_attended(n) for n in range(prefill.num_q_blocks)
        )


class TestRolledBuffer:
    """cache_capacity < seqlen_kv: a compact buffer that has filled and is
    sliding forward. Only reachable once the plan allows the logical
    position to run past the physical allocation.
    """

    def test_blocks_read_different_offsets_not_the_collapsed_bound(self):
        # Regression: read_start's clamp must not compare a logical origin
        # (thousands) against a physical bound (hundreds), which collapses
        # min() to the bound and makes every block read the same slice. Needs
        # cache_capacity != seqlen_kv to be visible at all.
        plan = plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=1024)
        assert plan is not None
        starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
        assert len(set(starts)) > 1, f"collapsed to one value: {starts}"
        assert starts == sorted(starts), "a window may not move backwards"
        assert all(0 <= s and s + plan.buffer_width <= 1024 for s in starts)

    def test_read_start_logical_is_read_start_plus_buffer_origin(self):
        plan = plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=1024)
        for qi in range(plan.num_q_blocks):
            assert plan.read_start(qi) + plan.buffer_origin == plan.read_start_logical(
                qi
            )

    def test_buffer_origin_is_zero_until_the_cache_outgrows_its_allocation(self):
        assert plan_sliding_window(512, 512, 64, q_block=64).buffer_origin == 0
        plan = plan_sliding_window(1, 5120, 64, q_block=1, cache_capacity=64)
        assert plan.buffer_origin == 5120 - 64

    def test_a_buffer_sized_to_exactly_one_window_always_reads_from_zero(self):
        # capacity == buffer_width: the buffer holds nothing but the live
        # window, so there is nowhere to shift to regardless of how far the
        # logical position has grown.
        plan = plan_sliding_window(1, 5120, 64, q_block=1, cache_capacity=64)
        assert all(plan.read_start(qi) == 0 for qi in range(plan.num_q_blocks))

    def test_a_buffer_too_small_for_a_multiblock_prefill_is_refused_up_front(self):
        # cache_capacity=192 clears the "narrower than one window" rejection
        # (buffer_width=128) but is still too small for the earliest block of
        # this 8-block prefill: its window reaches further back than a 192-row
        # rolled buffer holds.
        reason = rejection_reason(512, 5120, 64, True, 64, cache_capacity=192)
        assert reason is not None and "does not reach far enough back" in reason
        assert (
            plan_sliding_window(512, 5120, 64, q_block=64, cache_capacity=192) is None
        )

    def test_buffer_origin_defaults_to_the_exactly_full_derivation(self):
        # Backward compatibility: omitting the argument must reproduce the
        # value the plan used to derive internally, for every geometry.
        for lq, qb in ((1, 1), (64, STICK), (512, STICK)):
            for kv in (256, 512, 4096, 5120):
                for cap in (None, 64, 256, 1024, 4096):
                    plan = plan_sliding_window(
                        lq, kv, 64, q_block=qb, cache_capacity=cap
                    )
                    if plan is None:
                        continue
                    assert plan.buffer_origin == default_buffer_origin(kv, cap or kv)

    def test_a_single_block_plan_is_unaffected_by_the_reach_check(self):
        # Decode and single-chunk prefill only ever have one block, so the
        # earliest-block check is checking the only block -- it must not
        # reject a shape the buffer_width check already accepted.
        assert rejection_reason(1, 5120, 64, True, 1, cache_capacity=64) is None
        plan = plan_sliding_window(1, 5120, 64, q_block=1, cache_capacity=64)
        assert plan is not None


class TestExplicitBufferOrigin:
    """buffer_origin is given, not inferred.

    seqlen_kv and cache_capacity together do NOT determine where physical row
    0 sits. The default assumes an exactly-full buffer holding precisely the
    most recent cache_capacity positions -- true for HF's
    StaticSlidingWindowLayer, false for any evictor working at coarser
    granularity, which keeps whole blocks and so holds fewer live positions.
    """

    # capacity 256, window 64, decode at 1000, evictor freeing whole 64-blocks:
    # it must keep everything from 896 on, so row 0 holds 896, not 1000-256=744.
    CAPACITY, WINDOW, POSITION, EVICT_BLOCK = 256, 64, 1000, 64
    TRUE_ORIGIN = ((POSITION - WINDOW + 1) // EVICT_BLOCK) * EVICT_BLOCK

    def _plan(self, buffer_origin):
        return plan_sliding_window(
            1,
            self.POSITION,
            self.WINDOW,
            q_block=1,
            cache_capacity=self.CAPACITY,
            buffer_origin=buffer_origin,
        )

    def test_the_default_is_wrong_for_a_block_granular_evictor(self):
        # Not a bug in the default -- a demonstration that the two lengths
        # underdetermine the layout, which is why the argument exists. Under
        # the real layout the default's read points at rows holding logical
        # positions that were never written.
        default = self._plan(None)
        assert default.buffer_origin == 744 and self.TRUE_ORIGIN == 896
        read = default.read_start(0)
        actually_holds = self.TRUE_ORIGIN + read
        assert actually_holds >= self.POSITION, (
            "the default read should land past every written row here"
        )

    def test_the_true_origin_places_the_read_on_written_data(self):
        plan = self._plan(self.TRUE_ORIGIN)
        assert plan is not None
        low, high = plan.row_window(0)
        start = plan.read_start_logical(0)
        # The window is covered by the buffer...
        assert start <= low and high <= start + plan.buffer_width
        # ...and every column in it is a position the evictor actually kept.
        assert all(self.TRUE_ORIGIN <= c < self.POSITION for c in range(low, high))

    def test_an_overshooting_tail_is_still_banded_not_skipped(self):
        # The buffer reaches past cache_seqlen here, so the band must be
        # emitted -- skipping it would attend the unwritten tail.
        assert not self._plan(self.TRUE_ORIGIN).block_is_fully_attended(0)

    def test_reads_stay_stick_aligned_at_any_origin(self):
        # buffer_origin need not be stick-aligned: read_start floors AFTER
        # converting to physical coordinates, so the offset lands on a stick
        # whatever the origin is.
        for origin in range(744, 1000):
            plan = self._plan(origin)
            if plan is None:
                continue
            assert plan.read_start(0) % STICK == 0, origin

    def test_out_of_range_origins_are_refused_with_the_reason(self):
        def reason(origin):
            return rejection_reason(
                1, 1000, 64, True, 1, cache_capacity=256, buffer_origin=origin
            )

        # Below the default: the newest token would fall off the end.
        assert "too small" in reason(700)
        assert "too small" in reason(-1)
        # At or past cache_seqlen: no written row remains.
        assert "no written row" in reason(1000)
        assert "no written row" in reason(1001)
        # The default and the block-granular truth both stand.
        assert reason(744) is None and reason(896) is None


class TestStillFilling:
    """cache_seqlen < cache_capacity: a cache that has not filled its
    allocation yet, so it is left-aligned and buffer_origin is 0.
    """

    def test_a_still_filling_cache_plans_like_any_other(self):
        assert (
            plan_sliding_window(1, 128, 64, q_block=1, cache_capacity=4096) is not None
        )
        assert (
            plan_sliding_window(256, 256, 64, q_block=64, cache_capacity=4096)
            is not None
        )

    def test_buffer_origin_is_zero_while_still_filling(self):
        plan = plan_sliding_window(1, 128, 64, q_block=1, cache_capacity=4096)
        assert plan.buffer_origin == 0

    def test_per_block_read_start_is_monotonic(self):
        # A later block never reads from further back than an earlier one, so
        # buffer_width sized to the widest block covers every block. Asserted
        # against the plan's own read_start, never an inlined copy of the
        # formula, so it cannot drift from the code.
        for seqlen_q in (128, 256, 384, 512):
            for window in (63, 64, 100, 128, 150, 192, 201, 300):
                plan = plan_sliding_window(
                    seqlen_q, seqlen_q, window, q_block=STICK, cache_capacity=seqlen_q
                )
                if plan is None:
                    continue
                starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
                assert starts == sorted(starts), (seqlen_q, window, starts)

    @pytest.mark.parametrize(
        "seqlen_q", [1, 64, 128, 192, 256, 320, 384, 448, 512], ids=lambda q: f"q{q}"
    )
    def test_a_stick_aligned_cache_seqlen_needs_no_overshoot(self, seqlen_q):
        # Overshooting the written prefix is safe (TestArbitraryCacheSeqlen),
        # but when cache_seqlen happens to be stick-aligned no block should
        # need to: the reads are stick arithmetic that lands exactly on it.
        capacity = 4096
        q_block = 1 if seqlen_q == 1 else STICK
        for window in (1, 33, 63, 64, 100, 128, 150, 192, 201, 300):
            for cache_seqlen in range(max(seqlen_q, STICK), capacity, STICK):
                plan = plan_sliding_window(
                    seqlen_q,
                    cache_seqlen,
                    window,
                    q_block=q_block,
                    cache_capacity=capacity,
                )
                if plan is None:
                    continue
                for qi in range(plan.num_q_blocks):
                    stop = plan.read_start_logical(qi) + plan.buffer_width
                    assert stop <= cache_seqlen, (window, cache_seqlen, qi, stop)


class TestArbitraryCacheSeqlen:
    """cache_seqlen need not be stick-aligned -- only cache_capacity, the
    physical allocation, must be. At a non-aligned position a block's buffer
    CAN overshoot the written prefix; causal masking already excludes every
    overshot column (column <= row < cache_seqlen), so no tail mask is
    involved.
    """

    def test_a_non_aligned_cache_seqlen_is_accepted_with_a_capacity(self):
        # Only the allocation (4160, a stick multiple) needs alignment; 5001
        # is a token count, not a memory offset.
        assert rejection_reason(1, 5001, 4096, True, 1, cache_capacity=4160) is None
        assert (
            plan_sliding_window(1, 5001, 4096, q_block=1, cache_capacity=4160)
            is not None
        )

    def test_rolled_decode_read_start_is_position_independent(self):
        # The payoff of flooring in physical rather than logical coordinates:
        # one compiled graph serves every decode step on a rolled buffer,
        # rather than a recompile per logical position.
        starts = {
            plan_sliding_window(
                1, pos, 4096, q_block=1, cache_capacity=4160
            ).read_start(0)
            for pos in range(4160, 4160 + 600)
        }
        assert starts == {64}, starts

    def test_a_small_rolled_buffer_is_also_position_independent(self):
        starts = {
            plan_sliding_window(1, pos, 64, q_block=1, cache_capacity=64).read_start(0)
            for pos in range(64, 64 + 600)
        }
        assert starts == {0}, starts

    def test_a_buffer_that_overshoots_the_still_filling_tail_is_not_fully_attended(
        self,
    ):
        # seqlen_kv=127 is not stick-aligned: the W=63 buffer physically
        # reaches column 128 -- one past what is written. block_is_fully_
        # attended must say so (row_window's hi clamps to seqlen_kv, so a
        # buffer extending past it can never satisfy "fully attended"),
        # which is what makes the causal band get added at all.
        plan = plan_sliding_window(1, 127, 63, q_block=1, cache_capacity=128)
        assert plan is not None
        assert not plan.block_is_fully_attended(0)

    def test_no_leakage_or_coverage_gap_at_arbitrary_non_stick_cache_seqlen(self):
        # The regression net for the whole design, swept over arbitrary
        # (non-stick) cache_seqlen, against the two properties the design
        # rests on: (1) coverage --
        # every column a row is allowed to attend (row_window, already
        # clamped to cache_seqlen) lies inside this block's buffer, checked
        # against the plan's own (independently computed) row_window; and
        # (2) no leakage -- whenever block_is_fully_attended skips the band
        # entirely, every buffer column it reads must already be written,
        # since nothing will mask it otherwise.
        for cap in (64, 128, 256, 1024):
            for seq in range(1, 600, 7):  # arbitrary step, hits non-stick values
                for window in (63, 64, 100, 128, 300):
                    for lq, qb in ((1, 1), (64, 64)):
                        if seq < lq:
                            continue
                        plan = plan_sliding_window(
                            lq, seq, window, q_block=qb, cache_capacity=cap
                        )
                        if plan is None:
                            continue
                        for qi in range(plan.num_q_blocks):
                            start = plan.read_start_logical(qi)
                            stop = start + plan.buffer_width
                            if plan.block_is_fully_attended(qi):
                                assert stop <= seq, (cap, seq, window, lq, qi)
                            q_start, q_end = plan.block_q_range(qi)
                            for q_index in range(q_start, q_end):
                                lo, hi = plan.row_window(q_index)
                                assert start <= lo and hi <= stop, (
                                    cap,
                                    seq,
                                    window,
                                    lq,
                                    qi,
                                    q_index,
                                )


class TestQueryPadding:
    """A query that does not divide into blocks is padded, not refused."""

    @pytest.mark.parametrize(
        "seqlen_q,expected",
        [(1, (1, 1)), (64, (64, 64)), (100, (64, 128)), (257, (64, 320))],
    )
    def test_block_size_and_padded_length(self, seqlen_q, expected):
        # Decode takes a single-row block and no padding: padding one row up to
        # a full block would be 64x the work for one row of output.
        assert query_blocking(seqlen_q) == expected

    @pytest.mark.parametrize("seqlen_q", [33, 100, 200, 257])
    def test_front_padding_preserves_every_real_coordinate(self, seqlen_q):
        # The claim the whole scheme rests on. Row i sits at cache coordinate
        # seqlen_kv - seqlen_q + i; once seqlen_q is the PADDED length the two
        # shifts cancel, so every real row keeps the coordinate its window is
        # derived from. Back-padding would move all of them.
        seqlen_kv, window = 512, 64
        _, padded = query_blocking(seqlen_q)
        pad = padded - seqlen_q
        plan = _plan(padded, seqlen_kv, window)
        for i in range(seqlen_q):
            coord = seqlen_kv - seqlen_q + i
            assert plan.row_window(pad + i) == (
                max(0, coord - window + 1),
                min(seqlen_kv, coord + 1),
            )

    @pytest.mark.parametrize("seqlen_q", [33, 100, 200, 257])
    def test_no_padded_row_is_fully_masked(self, seqlen_q):
        # A row with an empty window would make softmax produce nan, which
        # would poison the real rows through no fault of theirs.
        plan = _plan(query_blocking(seqlen_q)[1], 512, 64)
        for j in range(plan.seqlen_q):
            low, high = plan.row_window(j)
            assert low < high


class TestRejection:
    """Shapes the placement declines, and the reason it gives for each."""

    REJECTED = [
        ((256, 256, 64, False, 64), "causal"),
        ((256, 256, 0, True, 64), "window_size=0 must be positive"),
        # No cache_capacity given: seqlen_kv doubles as the allocation
        # (effective_capacity), which must still be stick-aligned -- unlike
        # a passed cache_seqlen, which need not be (see rejection_reason).
        ((256, 257, 64, True, 64), "pad the cache allocation"),
        ((100, 256, 64, True, 64), "whole number of 64-row blocks"),
        ((512, 256, 64, True, 64), "exceeds seqlen_kv"),
    ]

    @pytest.mark.parametrize("args,expected", REJECTED)
    def test_each_reason_names_its_constraint(self, args, expected):
        # This message reaches a user in a compile-time warning, so
        # "unsupported shape" would not have been good enough.
        reason = rejection_reason(*args)
        assert reason is not None, f"{args} should be rejected"
        assert expected in reason, f"{args}: {reason}"

    def test_the_plan_agrees_with_the_reason(self):
        # rejection_reason is what plan_sliding_window decides with, so a shape
        # with a reason must not produce a plan. q_block=None means
        # query_blocking always returns a usable block size now, so a shape
        # with a reason is rejected on that reason rather than on the block.
        for args, _ in self.REJECTED:
            assert plan_sliding_window(*args[:4], q_block=args[4]) is None, args

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_supported_shapes_give_no_reason(self, seqlen_q, seqlen_kv, window):
        assert (
            rejection_reason(
                seqlen_q, seqlen_kv, window, True, query_blocking(seqlen_q)[0]
            )
            is None
        )

    def test_cache_capacity_constraints(self):
        # Omitted -- seqlen_kv doubles as the allocation, unchanged behaviour.
        assert rejection_reason(512, 5120, 64, True, 64) is None
        # The allocation must be stick-aligned; read_start floors against it.
        assert "cache_capacity=1000" in rejection_reason(
            512, 5120, 64, True, 64, cache_capacity=1000
        )
        # ...and must fit one buffer: window(64) + q_block(64) = 128.
        assert "narrower than the 128-row buffer" in rejection_reason(
            512, 5120, 64, True, 64, cache_capacity=64
        )
        assert rejection_reason(512, 5120, 64, True, 64, cache_capacity=1024) is None

    def test_all_three_orderings_of_seqlen_and_capacity_are_accepted(self):
        # Reading only cares which side of cache_capacity the cache is on:
        # exactly full, rolled past its allocation, still filling it.
        assert rejection_reason(1, 256, 64, True, 1, cache_capacity=256) is None
        assert rejection_reason(1, 5000, 64, True, 1, cache_capacity=64) is None
        assert rejection_reason(1, 100, 64, True, 1, cache_capacity=4096) is None

    def test_degenerate_lengths_are_rejected(self):
        assert "degenerate" in rejection_reason(1, 0, 64, True, 1, cache_capacity=256)
        assert "degenerate" in rejection_reason(1, -5, 64, True, 1, cache_capacity=256)
        assert rejection_reason(1, 256, 64, True, 1, cache_capacity=0) is not None
        # A negative capacity clears the stick check (-64 % 64 == 0), so the
        # message has to name cache_capacity rather than fall through.
        assert "cache_capacity=0" in rejection_reason(
            1, 256, 64, True, 1, cache_capacity=0
        )
        assert "cache_capacity=-64" in rejection_reason(
            1, 256, 64, True, 1, cache_capacity=-64
        )

    def test_an_explicit_origin_is_checked_without_a_capacity(self):
        # Regression: the capacity checks were guarded on cache_capacity being
        # passed, so an explicit buffer_origin skipped the reach check and
        # planned a negative read_start.
        assert "does not reach far enough back" in rejection_reason(
            1, 256, 64, True, 1, buffer_origin=200
        )
        assert plan_sliding_window(1, 256, 64, q_block=1, buffer_origin=200) is None
        # The largest origin the earliest block still reaches: its window
        # starts at 256 - 1 - 64 + 1 = 192.
        assert rejection_reason(1, 256, 64, True, 1, buffer_origin=192) is None

    def test_a_hand_built_read_is_validated(self):
        # The op takes its placement as plain ints, so a caller that does not
        # go through the plan can walk off the end of the cache.
        shape = (BATCH, HEADS, 256, HEAD_DIM)
        ok = dict(
            read_start=0,
            buffer_width=128,
            cache_capacity=256,
            num_heads=8,
            num_kv_heads=8,
            key_shape=shape,
            value_shape=shape,
        )
        assert check_window_read(**ok) is None
        assert check_window_read(**{**ok, "read_start": 128}) is None  # last legal
        assert "runs past the cache" in check_window_read(**{**ok, "read_start": 129})
        assert check_window_read(**{**ok, "read_start": -1}) is not None
        assert "whole multiple" in check_window_read(**{**ok, "num_kv_heads": 3})
        # MLA-style: V's head_dim differs from K's -- the fake kernel's shape
        # would otherwise silently come from the wrong tensor.
        mismatched = (BATCH, HEADS, 256, HEAD_DIM * 2)
        assert "does not match" in check_window_read(
            **{**ok, "value_shape": mismatched}
        )

    def test_a_read_overshooting_the_written_prefix_is_allowed(self):
        # check_window_read bounds the read against the ALLOCATION, never
        # against the cache's logical position: a still-filling cache's buffer
        # routinely extends past what is written, and causal masking excludes
        # those columns. A logical bound here would reject most valid plans.
        shape = (BATCH, HEADS, 256, HEAD_DIM)
        overshooting = dict(
            read_start=0,
            buffer_width=128,  # cache_seqlen would be 100: reads 28 unwritten rows
            cache_capacity=256,
            num_heads=8,
            num_kv_heads=8,
            key_shape=shape,
            value_shape=shape,
        )
        assert check_window_read(**overshooting) is None
        # And the plan really does produce such a read, so this is not
        # a hypothetical: cache_seqlen=100 into a 256-row allocation.
        plan = plan_sliding_window(1, 100, 64, q_block=1, cache_capacity=256)
        assert plan is not None
        assert plan.read_start(0) + plan.buffer_width > plan.seqlen_kv


# --------------------------------------------------------------- 2. algorithm


def _expand_kv(tensor, expansion):
    if expansion == 1:
        return tensor
    return tensor.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)


def _reference_allowed(seqlen_q, seqlen_kv, window):
    """Full [seqlen_q, seqlen_kv] boolean band -- the definition of SWA."""
    q_coord = torch.arange(seqlen_q) + (seqlen_kv - seqlen_q)
    delta = q_coord.unsqueeze(-1) - torch.arange(seqlen_kv).unsqueeze(0)
    return (delta >= 0) & (delta < window)


def _block_allowed(plan: SlidingWindowPlan, qi: int) -> torch.Tensor:
    """[q_len, buffer_width] boolean band inside block qi's buffer."""
    start = plan.read_start(qi)
    cols = torch.arange(start, start + plan.buffer_width)
    q_start, q_end = plan.block_q_range(qi)
    rows = []
    for q_index in range(q_start, q_end):
        low, high = plan.row_window(q_index)
        rows.append((cols >= low) & (cols < high))
    return torch.stack(rows)


def _additive(allowed, dtype):
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype),
        torch.full((), float("-inf"), dtype=dtype),
    )


def _windowed_attention(query, key, value, plan, scale):
    """Per block, attend only against the compact window."""
    expansion = query.size(1) // key.size(1)
    out_blocks = []
    for qi in range(plan.num_q_blocks):
        q_start, q_end = plan.block_q_range(qi)
        start = plan.read_start(qi)
        stop = start + plan.buffer_width
        scores = (
            torch.matmul(
                query[:, :, q_start:q_end, :],
                _expand_kv(key[:, :, start:stop, :], expansion).transpose(-1, -2),
            )
            * scale
        )
        scores = scores + _additive(_block_allowed(plan, qi), query.dtype)
        out_blocks.append(
            torch.matmul(
                torch.softmax(scores, dim=-1),
                _expand_kv(value[:, :, start:stop, :], expansion),
            )
        )
    return torch.cat(out_blocks, dim=2)


def _masked_attention(query, key, value, window, scale):
    """The definition, computed literally."""
    expansion = query.size(1) // key.size(1)
    scores = torch.matmul(query, _expand_kv(key, expansion).transpose(-1, -2)) * scale
    scores = scores + _additive(
        _reference_allowed(query.size(2), key.size(2), window), query.dtype
    )
    return torch.matmul(torch.softmax(scores, dim=-1), _expand_kv(value, expansion))


def _cpu_inputs(seqlen_q, seqlen_kv, dtype=torch.float64, kvheads=HEADS):
    generator = torch.Generator().manual_seed(0)

    def make(shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    return (
        make((BATCH, HEADS, seqlen_q, HEAD_DIM)),
        make((BATCH, kvheads, seqlen_kv, HEAD_DIM)),
        make((BATCH, kvheads, seqlen_kv, HEAD_DIM)),
    )


class TestAlgorithm:
    """Does the compact window compute the same attention as the full mask?"""

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_unmasked_column_sets_are_identical(self, seqlen_q, seqlen_kv, window):
        # Claim 1, exact. Combinatorial, so no tolerance is involved, and this
        # is what licenses the whole approach.
        plan = _plan(seqlen_q, seqlen_kv, window)
        reference = _reference_allowed(seqlen_q, seqlen_kv, window)
        for qi in range(plan.num_q_blocks):
            q_start, q_end = plan.block_q_range(qi)
            start = plan.read_start(qi)
            block = _block_allowed(plan, qi)
            for i, q_index in enumerate(range(q_start, q_end)):
                gathered = (block[i].nonzero().flatten() + start).tolist()
                assert gathered == reference[q_index].nonzero().flatten().tolist()
            # A fully masked row would make softmax produce nan.
            assert block.any(dim=-1).all()

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SHAPES, ids=IDS)
    def test_output_matches_the_masked_reference(self, seqlen_q, seqlen_kv, window):
        # Claim 2, roundoff. float64, so this is about the arithmetic rather
        # than about float16.
        plan = _plan(seqlen_q, seqlen_kv, window)
        query, key, value = _cpu_inputs(seqlen_q, seqlen_kv)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        actual = _windowed_attention(query, key, value, plan, scale)
        expected = _masked_attention(query, key, value, window, scale)
        assert torch.isfinite(actual).all()
        assert (actual - expected).abs().max().item() < 1e-12

    def test_gqa_matches_in_float16(self):
        # The dtype the device runs in, the GQA expand, and the scale split
        # across query and key the way the decomposition does it -- at the
        # tolerance the device tests use.
        plan = _plan(256, 256, 64)
        query, key, value = _cpu_inputs(256, 256, dtype=torch.float16, kvheads=2)
        split = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))
        actual = _windowed_attention(query * split, key * split, value, plan, 1.0)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            _additive(_reference_allowed(256, 256, 64), torch.float16),
            enable_gqa=True,
        )
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    def test_windowing_actually_changes_the_answer(self):
        # Guards the agreement tests above: if the band were a no-op they would
        # pass while proving nothing.
        plan = _plan(256, 256, 64)
        query, key, value = _cpu_inputs(256, 256)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        windowed = _windowed_attention(query, key, value, plan, scale)
        causal = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scale
        )
        assert (windowed - causal).abs().max().item() > 1e-3


# ------------------------------------------------------------------ 3. the op


def _device_cache(differentiation, seqlen_kv=256, kvheads=HEADS):
    return cached_randn(
        (BATCH, kvheads, seqlen_kv, HEAD_DIM),
        differentiation=differentiation,
        dtype=torch.float16,
    )


def _call_op(cache, plan, block_index, which, value_cache=None):
    """which: 0 -> k_win, 1 -> v_win, 2 -> band.

    value_cache defaults to cache: most callers only inspect k_win or the
    band, where key and value being the same tensor is harmless. A test that
    inspects v_win must pass a value_cache that actually differs from cache,
    or a v_win sourced from the wrong tensor would pass unnoticed.

    The window and the band are separate ops -- they share no arguments -- so
    2 dispatches to the mask rather than indexing a third return value.
    """
    value_cache = cache if value_cache is None else value_cache
    q_start, _ = plan.block_q_range(block_index)
    if which == 2:
        return torch.ops.spyre.window_band_mask(
            plan.read_start(block_index),
            plan.q_block,
            plan.buffer_width,
            plan.q_kv_offset + q_start,
            plan.window_size,
            plan.is_causal,
            cache.dtype,
            cache.device,
        )
    return torch.ops.spyre.kv_window(
        cache,
        value_cache,
        plan.read_start(block_index),
        plan.buffer_width,
        HEADS,
    )[which]


def _reference_window(cache, plan, block_index, transpose=False):
    start = plan.read_start(block_index)
    window = cache[:, :, start : start + plan.buffer_width, :]
    if transpose:
        window = window.transpose(-1, -2)
    return _expand_kv(window, HEADS // cache.size(1))


def _reference_band(plan, block_index):
    band = _additive(_block_allowed(plan, block_index), torch.float16)
    return band.unsqueeze(0).unsqueeze(0)


class TestKVWindowOp:
    """The op on device, against slices built here.

    GQA is checked one block at a time, which is what the body consumes.
    Cat-ing several expanded windows together zeroes the leading slots on
    device -- a backend defect unrelated to this op, and nothing on this path
    does it.
    """

    def test_window_and_band(self):
        # All three outputs of one call, per block, so a failure names both which
        # output was wrong and which block. Not cat-ed across blocks: that is the
        # defect the class docstring describes, and nothing on this path does it.
        plan = _plan(256, 256, 64)

        def fn(k, v):
            blocks = range(plan.num_q_blocks)
            if k.device.type == "spyre":
                return tuple(
                    _call_op(k, plan, n, which, value_cache=v)
                    for which in (0, 1, 2)
                    for n in blocks
                )
            return tuple(
                [_reference_window(k, plan, n, transpose=True) for n in blocks]
                + [_reference_window(v, plan, n) for n in blocks]
                + [_reference_band(plan, n) for n in blocks]
            )

        compare_with_cpu(fn, _device_cache(1), _device_cache(2), run_eager=False)

    def test_gqa_key_window_for_one_block(self):
        # Block 2 is the first whose read start is not 0, so a dropped window
        # shift shows up here rather than passing by accident.
        plan = _plan(256, 256, 64)
        assert plan.read_start(0) == 0 and plan.read_start(2) > 0

        def fn(k, v):
            if k.device.type == "spyre":
                return _call_op(k, plan, 2, 0)
            return _reference_window(k, plan, 2, transpose=True)

        compare_with_cpu(
            fn,
            _device_cache(1, kvheads=2),
            _device_cache(2, kvheads=2),
            run_eager=False,
        )

    def test_decode_window(self):
        # buffer_width == window: 64 rows out of a 4096-row cache.
        plan = _plan(1, 4096, 64, q_block=1)
        cache = _device_cache(3, seqlen_kv=4096)

        def fn(k, v):
            if k.device.type == "spyre":
                return _call_op(k, plan, 0, 0)
            return _reference_window(k, plan, 0, transpose=True)

        compare_with_cpu(fn, cache, cache, run_eager=False)

    def test_decode_band_is_all_zeros(self):
        # What the body relies on to skip the band add entirely.
        plan = _plan(1, 4096, 64, q_block=1)
        cache = _device_cache(3, seqlen_kv=4096)
        assert torch.equal(
            _reference_band(plan, 0), torch.zeros((1, 1, 1, 64), dtype=torch.float16)
        )

        def fn(k, v):
            if k.device.type == "spyre":
                return _call_op(k, plan, 0, 2)
            return _reference_band(plan, 0)

        compare_with_cpu(fn, cache, cache, run_eager=False)


class TestValidStart:
    """The three torch-free helpers behind the valid_start band."""

    def test_none_and_all_zero_mask_nothing(self):
        # A caller with no padding must not pay for a per-batch band.
        assert band_valid_start(None) is None
        assert band_valid_start([]) is None
        assert band_valid_start([0, 0]) is None
        assert band_batch([0, 0]) == 1

    def test_uniform_nonzero_stays_broadcast(self):
        # Same threshold for every sequence: one row of band, broadcast over batch.
        assert band_valid_start([40, 40]) == [40, 40]
        assert band_batch([40, 40]) == 1

    def test_ragged_widens_to_batch(self):
        assert band_batch([0, 40]) == 2
        assert band_batch([40, 40, 7]) == 3

    def test_check_rejects_wrong_length(self):
        assert "2 entries" in check_valid_start([0, 40], 3, 1088)

    def test_check_rejects_out_of_range(self):
        assert "outside" in check_valid_start([-1], 1, 1088)
        assert "outside" in check_valid_start([1089], 1, 1088)

    def test_check_accepts_valid(self):
        assert check_valid_start(None, 4, 1088) is None
        assert check_valid_start([0, 40], 2, 1088) is None
        assert check_valid_start([1088], 1, 1088) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
