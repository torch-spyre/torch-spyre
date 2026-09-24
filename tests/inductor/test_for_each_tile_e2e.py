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

"""End-to-end Spyre-device tests for WhileLoop -> OpSpec/LoopSpec lowering.

Compiles each fixture, runs it on the Spyre device, and compares against a
CPU reference -- for IR-level / mocked-IR unit tests of the lowering
machinery itself, see test_for_each_tile_lowering.py.

Minimum coverage per docs/superpowers/specs/2026-09-09-while-loop-lowering-design.md:
1. Single carry (this file: test_carry_mode_split_k) -- currently XFAIL on a
   read-copy/stick-layout gap; see that test's own docstring.
2 (carry + Kind.SLICE tile-advancing input): covered implicitly by
   test_carry_mode_split_k, whose X/Y operands are both Kind.SLICE.
3. Kind.GATHER: covered by test_gather_mode_paged_pages, which gathers one
   page per trip from inside the body the way paged attention does.
4. Multiple independent carries: covered by test_carry_mode_online_softmax
   (carry = (m, denom, acc), an online-softmax flash-attention inner loop).
Case 5 (nested for_each_tile) covers pure map/map nesting in
TestForEachTileNestedMapE2E, map/carry nesting in
test_batched_map_over_online_softmax_carry, and carry/carry nesting (depth=2
and depth=3) in TestForEachTileNestedCarryE2E. The map/carry case stages
full K/V buffers in the outer batch map, then slices those staged buffers in
the inner Lk carry, exercising per-level ownership of input advances. Case 6
(deliberate-decline) is still open, tracked as a follow-on item.

test_map_mode_split_m (map mode: Kind.SLICE + Kind.INVARIANT operands, a
stacking carry, no user carry) passes end to end with verified numerics and
is the case that exercises the full splice -> DimHint synthesis ->
coarse-tile -> single scf.for pipeline.

TestForEachTilePointwiseE2E adds simpler single-level pointwise/softmax
coverage (add, abs, a 3-operand abs(a+b)*c chain, row-tiled softmax),
ported down from test_coarse_tile_e2e.py's HINT-driven test_add_*/test_abs_*
and test_hint_softmax_row_tiling families -- for_each_tile doesn't need
coarse_tile's exhaustive combinatorial coverage, but benefits from its own
easy-to-debug cases, including a multi-stick tile_size variant of each
(multi-stick tiling has been a historical source of bugs; see
test_hint_softmax_row_tiling's docstring on the device_size[1] invariant).
"""

import functools
import unittest

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME

from for_each_tile_fixtures import (
    B,
    COLS,
    ROWS,
    D,
    K,
    LK,
    LQ,
    M,
    N,
    PAGE_HS,
    PAGE_LQ,
    PAGE_POOL,
    PAGE_SIZE,
    STICK_COLS,
    STICK_ROWS,
    abs_add_mul_tiled_fn,
    abs_add_mul_tiled_reference,
    abs_tiled_fn,
    abs_tiled_reference,
    add_tiled_fn,
    add_tiled_reference,
    batched_online_softmax_fn,
    nested_add_outer_row_inner_col_fn,
    nested_add_outer_row_inner_col_reference,
    nested_online_softmax_fn,
    nested_online_softmax_reference,
    nested_split_m_then_k_fn,
    nested_split_m_then_k_reference,
    nested_two_inner_loops_shared_init_fn,
    nested_two_inner_loops_shared_init_reference,
    online_softmax_fn,
    online_softmax_reference,
    paged_gather_fn,
    paged_gather_inputs,
    paged_gather_kv_fn,
    paged_gather_kv_inputs,
    paged_gather_kv_reference,
    paged_gather_nested_fn,
    paged_gather_nested_reference,
    paged_gather_reference,
    softmax_row_tiled_fn,
    softmax_row_tiled_reference,
    split_k_caller_init_fn,
    split_k_fn,
    split_k_transposed_caller_init_fn,
    split_k_transposed_caller_init_two_carries_fn,
    split_k_transposed_caller_init_two_carries_reference,
    split_m_fn,
    triple_nested_stardep_inner_fn,
    triple_nested_stardep_inner_reference,
    triple_nested_stardep_middle_fn,
    triple_nested_stardep_middle_reference,
    triple_nested_stardep_multilevel_fn,
    triple_nested_stardep_multilevel_reference,
    triple_nested_stardep_outer_fn,
    triple_nested_stardep_outer_reference,
)
from tests.inductor.utils_inductor import cached_randn, cached_xavier, dl16_round


def _with_dynamo_reset(test_fn):
    """Wrap a test method to reset Dynamo immediately before it runs."""

    @functools.wraps(test_fn)
    def wrapper(self, *args, **kwargs):
        torch._dynamo.reset()
        return test_fn(self, *args, **kwargs)

    return wrapper


class _DynamoResetTestCase(unittest.TestCase):
    """Resets Dynamo before each test.

    Every test in this file torch.compile()s one of a small set of shared
    fixture functions (e.g. add_tiled_fn is compiled by test_add_tiled_small
    AND test_add_tiled_multi_stick, at different shapes/tile_sizes). Without
    a reset, a later test's compile of the SAME function object can hit
    Dynamo's guard/recompile cache from an earlier test and skip re-entering
    Inductor's codegen entirely -- silently reusing a compiled artifact
    specialized for the wrong shape instead of recompiling for the new one.
    Confirmed: test_add_tiled_small passes in isolation but fails (wrong
    numerics, no new torch_compile_debug artifact) when run immediately
    after test_add_tiled_multi_stick in the same process; torch._dynamo.
    reset() before each test fixes it. This mirrors the reset
    capture_post_grad_while_loop (for_each_tile_fixtures.py) already does
    for the same reason.

    The reset is applied via a per-method decorator (__init_subclass__ below)
    rather than setUp(), because setUp() is not reliable here: the OOT test
    harness's instantiate_device_type_tests() builds each device-specific
    test class as type(name, (DeviceTypeTestBase, YourTestCase), {}), and
    DeviceTypeTestBase's own setUp() -- which never calls super().setUp() --
    wins the MRO over this class's setUp(), silently skipping the reset.
    A method decorator has no such hazard: it wraps the test function object
    itself, which survives instantiate_test()'s copy.deepcopy() untouched.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))


class TestForEachTileE2E(_DynamoResetTestCase):
    # Spyre's matmul actually runs in dl16 on-device (SEN169_FP16; fp16 <->
    # dl16 conversion happens implicitly at the H2D/D2H transfer boundary),
    # so the reference has to be dl16-faithful: round through (an
    # approximation of) dl16, then accumulate in fp32 on CPU. Comparing
    # against the fp32 product of fp32-precision operands would fail on
    # rounding alone, independently of anything this test is meant to check.
    #
    # Operands are generated directly at fp16 via cached_xavier rather than
    # `torch.randn(fp32).half()`, for two reasons: (1) it avoids a second,
    # independent double-rounding step (fp32 -> fp16) on top of the
    # fp16 -> dl16 conversion already happening at the device boundary, and
    # (2) xavier_uniform_ keeps output magnitude bounded regardless of the
    # contraction (K) dimension, whereas plain unit-variance randn summed
    # over K=256 produces outputs with magnitude ~sqrt(K) (~15-40 here) --
    # exactly where a loose rtol grants the most absolute slack and can mask
    # a real bug (see issue #4701's investigation). This is the same
    # rand_type="xavier" idiom test_inductor_ops.py already uses for its
    # large matmul cases.
    #
    # Tightened from the previous atol=0.1/rtol=0.1 now that both the
    # reference (dl16-rounded) and the operands (bounded-magnitude xavier)
    # remove the two effects that tolerance was compensating for.
    ATOL = 1e-2
    RTOL = 1e-2

    @staticmethod
    def _operands():
        X = cached_xavier((M, K))
        Y = cached_xavier((K, N))
        ref = dl16_round(X.float()) @ dl16_round(Y.float())
        return X.to(DEVICE_NAME), Y.to(DEVICE_NAME), ref

    def test_map_mode_split_m(self):
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_m_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_carry_mode_split_k(self):
        """Carry mode: accumulate a split-K matmul across tiles.

        Previously XFAIL (issue #4460) on a gap in read-copy layout
        reconciliation: ``for_each_tile``'s ``xs`` leaves for ``dims=(-1,
        0)`` are 3-D, transposed, ``movedim``-derived views of the operands
        (``[4, 3, 8]`` stride ``[3, 1, 12]`` for X, ``[4, 3, 6]`` stride
        ``[18, 6, 1]`` for Y). The K-advancing reads of those leaves route
        through ``coarse_tile.py``'s read-copy machinery, which built tile
        buffers whose own layouts (e.g. ``[8, 6, 3]`` stride ``[0, 1, 6]`` --
        a broadcast leading dim over transposed inner dims) then failed
        stick reconciliation in
        ``optimize_restickify.py``/``propagate_layouts.py`` ("No mechanism
        to scatter elements from one stick to multiple sticks").

        Now passes: confirmed via isolated stash/pop bisection that this is
        fixed by the splice-var/``loop_info`` symbol-consistency work in
        ``spyre_kernel.py``/``for_each_tile_lowering.py`` (the same issue
        #4706 OS-5/``_synthesize_dim_hints_for_group`` fixes described in
        ``test_nested_for_each_tile_value_correct``'s docstring, which
        closed the original stick-reconciliation crash for this fixture
        family) -- not by ``insert_restickify.py``'s unrelated online-
        softmax K-advance fix (see ``test_carry_mode_online_softmax``), which
        this test passes with or without. Without the
        ``spyre_kernel.py``/``for_each_tile_lowering.py``
        fixes, this now fails as a silent-wrong-answer (96% mismatched
        elements) rather than the original compile-time crash, confirming
        the stick-reconciliation gap itself is closed and any remaining
        exposure is in a different, already-covered layer.
        """
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_k_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    @staticmethod
    def _exact_split_k_operands(acc0_shape):
        """Small-integer operands whose split-K sums stay exact.

        X and Y are 0/1 and acc0 is 0..3, so every partial sum is an integer
        of at most K + 3 = 259: exact in the device's fp16 (SEN169, 10
        significant bits) and across the H2D/D2H round trip. The reference is
        therefore exact, and a wrong or reused accumulator shows up as a
        mismatch rather than as rounding.
        """
        g = torch.Generator().manual_seed(0)
        X = torch.randint(0, 2, (M, K), generator=g).half()
        Y = torch.randint(0, 2, (K, N), generator=g).half()
        acc0 = torch.randint(0, 4, acc0_shape, generator=g).half()
        return X, Y, acc0

    def _assert_exact_twice_inputs_unchanged(self, fn, inputs, refs):
        """Two calls with the same inputs: both exact, every input unchanged.

        If an accumulator wrote into a caller's init (#4838), that input
        changes and the second call starts from the first call's result.
        ``refs`` holds one exact reference per output. Each result is brought
        to the host before the next call, so the second call cannot overwrite
        the first result's memory.
        """
        on_device = [t.to(DEVICE_NAME) for t in inputs]
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for _ in range(2):
            outs = compiled(*on_device)
            outs = outs if isinstance(outs, tuple) else (outs,)
            self.assertEqual(len(outs), len(refs))
            for out, ref in zip(outs, refs):
                torch.testing.assert_close(out.cpu().float(), ref, atol=0.0, rtol=0.0)
        for before, after in zip(inputs, on_device):
            self.assertTrue(torch.equal(after.cpu(), before), "a caller input changed")

    def test_carry_mode_split_k_caller_init(self):
        """#4838: a caller-owned init is copied, never accumulated into."""
        X, Y, acc0 = self._exact_split_k_operands((M, N))
        ref = acc0.float() + X.float() @ Y.float()
        self._assert_exact_twice_inputs_unchanged(
            split_k_caller_init_fn, (X, Y, acc0), (ref,)
        )

    def test_carry_mode_split_k_transposed_caller_init(self):
        """#4838: a transposed view of a caller init is copied as its storage.

        acc0 is [N, M] and the carry starts from acc0.t() ([M, N], not
        square), so a copy that dropped the transpose, or wrote through to
        the caller's buffer, fails the exact comparison.
        """
        X, Y, acc0 = self._exact_split_k_operands((N, M))
        ref = acc0.t().float() + X.float() @ Y.float()
        self._assert_exact_twice_inputs_unchanged(
            split_k_transposed_caller_init_fn, (X, Y, acc0), (ref,)
        )

    def test_carry_mode_split_k_transposed_caller_init_old_value_reader(self):
        """#4838: the transposed caller init's old value has another reader.

        ``b`` adds ``a``'s pre-trip value, so ``a`` needs an in-loop snapshot
        on top of its pre-loop copy; both must copy the caller's storage and
        keep the transpose. Every sum is an integer of at most 396 (``b``
        adds four values of at most 3 + 64 * trip), so the check is exact.
        """
        X, Y, acc0 = self._exact_split_k_operands((N, M))
        refs = split_k_transposed_caller_init_two_carries_reference(X, Y, acc0)
        self._assert_exact_twice_inputs_unchanged(
            split_k_transposed_caller_init_two_carries_fn, (X, Y, acc0), refs
        )

    def test_nested_inner_loops_share_one_init(self):
        """#4838: an inner loop's pre-loop copy re-runs on every outer trip.

        Both inner loops start from one fill made in the outer loop's body, so
        the first one gets a pre-loop copy. If that copy ran once instead of on
        every outer trip, later outer trips would start from a stale sum.
        """
        X, Y, _ = self._exact_split_k_operands((M, N))
        ref = nested_two_inner_loops_shared_init_reference(X, Y)
        self._assert_exact_twice_inputs_unchanged(
            nested_two_inner_loops_shared_init_fn, (X, Y), (ref,)
        )

    def test_carry_mode_online_softmax(self):
        """Carry mode: 3-leaf carry (m, denom, acc), online-softmax over K/V tiles.

        Case 4 (multiple independent carries) from the design spec's minimum
        coverage list. carry_bindings_for/splice_while_loop's per-binding loop
        is already generic over an arbitrary-length carry list; this is the
        first fixture that actually drives a 3-leaf init= end to end, both to
        confirm the pytree carry survives decompose_scan_to_while_loop's
        scan -> while_loop decomposition intact, and to confirm
        _extra_readers_of_placeholder/_snapshot_carry_placeholder correctly
        handle the write-after-read hazard this body's own m carry hits:
        `correction = exp(m - m_new)` reads m's OLD value a second time,
        after m_new (m's per-iteration output) has already been computed --
        the exact case an in-place-only rewrite would silently corrupt.
        """
        Q = cached_xavier((LQ, D))
        K = cached_xavier((LK, D), differentiation=1)
        V = cached_xavier((LK, D), differentiation=2)
        ref = online_softmax_reference(
            dl16_round(Q.float()), dl16_round(K.float()), dl16_round(V.float())
        )

        Q_spyre = Q.to(DEVICE_NAME)
        K_spyre = K.to(DEVICE_NAME)
        V_spyre = V.to(DEVICE_NAME)

        compiled = torch.compile(online_softmax_fn, backend="inductor", fullgraph=True)
        out = compiled(Q_spyre, K_spyre, V_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_batched_map_over_online_softmax_carry(self):
        """An inner Lk advance stays on a full outer-staged K buffer."""
        Q = cached_xavier((B, 64, 128))
        K = cached_xavier((B, 256, 128), differentiation=1)
        V = cached_xavier((B, 256, 128), differentiation=2)
        Qb, Kb, Vb = (
            dl16_round(Q.float()),
            dl16_round(K.float()),
            dl16_round(V.float()),
        )
        ref = torch.softmax(Qb @ Kb.transpose(-1, -2), dim=-1)
        ref = ref @ Vb

        compiled = torch.compile(
            batched_online_softmax_fn, backend="inductor", fullgraph=True
        )
        out = compiled(Q.to(DEVICE_NAME), K.to(DEVICE_NAME), V.to(DEVICE_NAME))

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_gather_mode_paged_pages(self):
        """Kind.GATHER: the body gathers its own page, one per trip.

        Case 3 from the design spec's minimum coverage list, and the shape
        paged attention actually wants: tile the block table, keep the page
        pool invariant, and let the body read its page index out of the tile.
        The index is a point read whose address advances with the spliced loop
        and has no iteration dim, so this drives
        coarse_tile._point_splice_advance_for_dep (record the per-trip
        advance), _full_buffer_read_deps' point-read exclusion (leave the read
        direct instead of staging a 1-element int32 into scratch),
        _rebase_point_splice_reads (pin the index to iteration 0 so the
        advance is not applied twice), and insert_restickify's per-dep advance
        handover. Numerics, not just compilation: every one of those can be
        got wrong in a way that compiles and re-reads the same page.

        Two matmuls per trip and a distinct page per trip, so an advance
        applied to the wrong operand or dropped entirely shows up as a large
        mismatch rather than rounding.
        """
        _, table, _ = paged_gather_inputs()
        pages = cached_xavier((PAGE_POOL, PAGE_SIZE, PAGE_HS))
        q = cached_xavier((PAGE_LQ, PAGE_HS), differentiation=1)
        ref = paged_gather_reference(dl16_round(pages.float()), dl16_round(q.float()))

        compiled = torch.compile(paged_gather_fn, backend="inductor", fullgraph=True)
        out = compiled(pages.to(DEVICE_NAME), table.to(DEVICE_NAME), q.to(DEVICE_NAME))

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_gather_mode_paged_pages_kv(self):
        """Kind.GATHER with separate K and V pools: one marker, two consumers.

        Numerics for the shape spyre-inference's page_attn_kernel actually
        has (see paged_gather_kv_fn): the page index sliced out of the tiled
        block table feeds an index_select per cache, so the table's single
        tile_dim_marker has two consuming reads. Compilation alone is covered
        device-lessly by TestConsumeTileDimMarkers.test_marker_with_two_
        computed_buffer_consumers_maps_both; what only the device can show is
        whether the marker's per-trip advance was composed into BOTH gathers.
        Getting that wrong for one of them re-reads page PAGE_ORDER[0]'s K
        (or V) on every trip -- a large mismatch here, and invisible on CPU.

        Distinct from test_softmax_row_tiled_small's own multi-consumer
        coverage in the one respect that matters: softmax's two consumers
        read a marker on a whole-row tile, whereas the marker here carries a
        genuine per-trip advance, so a consumer resolved by renaming rather
        than by composition is silent there and loud here.
        """
        _, _, table, _ = paged_gather_kv_inputs()
        k_pages = cached_xavier((PAGE_POOL, PAGE_SIZE, PAGE_HS))
        v_pages = cached_xavier((PAGE_POOL, PAGE_SIZE, PAGE_HS), differentiation=1)
        q = cached_xavier((PAGE_LQ, PAGE_HS), differentiation=2)
        ref = paged_gather_kv_reference(
            dl16_round(k_pages.float()),
            dl16_round(v_pages.float()),
            dl16_round(q.float()),
        )

        compiled = torch.compile(paged_gather_kv_fn, backend="inductor", fullgraph=True)
        out = compiled(
            k_pages.to(DEVICE_NAME),
            v_pages.to(DEVICE_NAME),
            table.to(DEVICE_NAME),
            q.to(DEVICE_NAME),
        )

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


class TestForEachTilePointwiseE2E(_DynamoResetTestCase):
    """Single-level map-mode pointwise/softmax fixtures, simpler than TestForEachTileE2E.

    for_each_tile doesn't need coarse_tile_e2e's exhaustive combinatorial
    coverage (see test_add_*/test_abs_* there); these cases exist to give
    the lowering pipeline easy-to-debug pointwise/reduction coverage of its
    own. Each op is tested at a small, single-stick debug size and again at
    a tile_size that spans multiple 64-fp16-element sticks -- multi-stick
    tiling has historically been a source of bugs (see
    test_hint_softmax_row_tiling's device_size[1] docstring in
    test_coarse_tile_e2e.py), so both sizes are kept as separate tests
    rather than only covering the large one.

    References here skip dl16_round: unlike TestForEachTileE2E's matmuls,
    these ops don't contract over a dimension, so they don't amplify
    rounding error, and the kept ATOL/RTOL=0.1 tolerance already covers the
    fp16-vs-dl16 gap.
    """

    ATOL = 0.1
    RTOL = 0.1

    def test_add_tiled_small(self):
        A = cached_randn((ROWS, COLS))
        B = cached_randn((ROWS, COLS), differentiation=1)
        A_spyre, B_spyre = A.to(DEVICE_NAME), B.to(DEVICE_NAME)
        ref = add_tiled_reference(A.float(), B.float())

        compiled = torch.compile(add_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(A_spyre, B_spyre, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_add_tiled_multi_stick(self):
        A = cached_randn((STICK_ROWS, STICK_COLS))
        B = cached_randn((STICK_ROWS, STICK_COLS), differentiation=1)
        A_spyre, B_spyre = A.to(DEVICE_NAME), B.to(DEVICE_NAME)
        ref = add_tiled_reference(A.float(), B.float())

        compiled = torch.compile(add_tiled_fn, backend="inductor", fullgraph=True)
        # tile_size=128 rows, full 128-col width -> 2 sticks/row per tile.
        out = compiled(A_spyre, B_spyre, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_tiled_small(self):
        A = cached_randn((ROWS, COLS))
        A_spyre = A.to(DEVICE_NAME)
        ref = abs_tiled_reference(A.float())

        compiled = torch.compile(abs_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(A_spyre, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_tiled_multi_stick(self):
        A = cached_randn((STICK_ROWS, STICK_COLS))
        A_spyre = A.to(DEVICE_NAME)
        ref = abs_tiled_reference(A.float())

        compiled = torch.compile(abs_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(A_spyre, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_add_mul_tiled_small(self):
        A = cached_randn((ROWS, COLS))
        B = cached_randn((ROWS, COLS), differentiation=1)
        C = cached_randn((ROWS, COLS), differentiation=2)
        A_spyre = A.to(DEVICE_NAME)
        B_spyre = B.to(DEVICE_NAME)
        C_spyre = C.to(DEVICE_NAME)
        ref = abs_add_mul_tiled_reference(A.float(), B.float(), C.float())

        compiled = torch.compile(
            abs_add_mul_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(A_spyre, B_spyre, C_spyre, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_add_mul_tiled_multi_stick(self):
        A = cached_randn((STICK_ROWS, STICK_COLS))
        B = cached_randn((STICK_ROWS, STICK_COLS), differentiation=1)
        C = cached_randn((STICK_ROWS, STICK_COLS), differentiation=2)
        A_spyre, B_spyre, C_spyre = (
            A.to(DEVICE_NAME),
            B.to(DEVICE_NAME),
            C.to(DEVICE_NAME),
        )
        ref = abs_add_mul_tiled_reference(A.float(), B.float(), C.float())

        compiled = torch.compile(
            abs_add_mul_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(A_spyre, B_spyre, C_spyre, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_softmax_row_tiled_small(self):
        X = cached_randn((ROWS, COLS))
        X_spyre = X.to(DEVICE_NAME)
        ref = softmax_row_tiled_reference(X.float())

        compiled = torch.compile(
            softmax_row_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, 2)

        torch.testing.assert_close(out.cpu().float(), ref, atol=0.02, rtol=0.1)

    def test_softmax_row_tiled_multi_stick(self):
        """Row-tile size spans 2 sticks/row -- see test_hint_softmax_row_tiling."""
        # abs=True: softmax is invariant to input sign, so this is purely
        # fidelity to the old torch.rand ([0, 1)) input this test used
        # before switching to cached_randn, not a correctness requirement.
        X = cached_randn((STICK_ROWS, STICK_COLS), abs=True)
        X_spyre = X.to(DEVICE_NAME)
        ref = softmax_row_tiled_reference(X.float())

        compiled = torch.compile(
            softmax_row_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, 128)

        # Tight atol, same rationale as test_hint_softmax_row_tiling: a
        # per-tile device_size bug that shrinks the row-stride dim would
        # corrupt stick groups after the first with an error far exceeding
        # fp16 rounding noise on random inputs in [0, 1).
        torch.testing.assert_close(out.cpu().float(), ref, atol=0.02, rtol=0.1)


class TestForEachTileNestedMapE2E(_DynamoResetTestCase):
    """Two-level nested for_each_tile, both levels pure map mode (no carry).

    Separate tier from the carry-based nested fixtures in
    for_each_tile_fixtures.py (nested_split_m_then_k_fn,
    triple_nested_stardep_*): here the outer loop tiles one dimension and
    the inner loop tiles a DIFFERENT dimension of the same operands, and
    neither level carries a reduction -- the simplest shape that still
    requires dimension-provenance resolution across two nesting levels.
    """

    ATOL = 0.1
    RTOL = 0.1

    @unittest.expectedFailure
    def test_nested_add_outer_row_inner_col_small(self):
        """Nested tiling with a sub-stick (2-element) inner column tile.

        XFAIL at compile time: ``Unsupported: ... Unexpected stick expression
        Mod(d1, 2): expected Mod(var, 64), a bare variable, 0, or any of
        those with a constant offset``.

        Root cause: the innermost tile add's output is reshaped by
        ``for_each_tile`` lowering into ``[2, 4, 2]`` (splitting the row's
        flat 8-wide column axis into 4 tiles of width 2, matching
        ``inner_tile_size=2``). ``_clone_layout`` in
        ``propagate_layouts.py`` builds that buffer's device layout purely
        from its own reshaped shape, picking the size-2 last dim as the
        stick dim. But the consuming op one level up reads the same buffer
        back with the *flattened* ``8*d0 + d1`` index (ranges ``d0:2,
        d1:8``), expecting one whole 8-wide stick-compatible axis. No
        per-shape STL of ``[2, 4, 2]`` can satisfy that: the inner tile
        (2 elements) is far below ``elems_per_stick`` (64), so stick padding
        breaks the 4x replication needed to reconstruct the flat axis,
        producing the unrepresentable ``Mod(d1, 2)`` coordinate. This is a
        genuine sub-stick tiling gap in ``_clone_layout``'s output-STL
        construction, not specific to add or to this test's shape -- fixing
        it needs ``_clone_layout`` to offer (or inherit) a layout that keeps
        the flattened axis whole, deferred as a separate task.
        """
        A = cached_randn((4, 8))
        B = cached_randn((4, 8), differentiation=1)
        A_spyre, B_spyre = A.to(DEVICE_NAME), B.to(DEVICE_NAME)
        ref = nested_add_outer_row_inner_col_reference(A.float(), B.float())

        compiled = torch.compile(
            nested_add_outer_row_inner_col_fn, backend="inductor", fullgraph=True
        )
        out = compiled(A_spyre, B_spyre, 2, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_nested_add_outer_row_inner_col_multi_stick(self):
        A = cached_randn((STICK_ROWS, STICK_COLS))
        B = cached_randn((STICK_ROWS, STICK_COLS), differentiation=1)
        A_spyre, B_spyre = A.to(DEVICE_NAME), B.to(DEVICE_NAME)
        ref = nested_add_outer_row_inner_col_reference(A.float(), B.float())

        compiled = torch.compile(
            nested_add_outer_row_inner_col_fn, backend="inductor", fullgraph=True
        )
        # Outer tiles 128 rows at a time; inner tiles 128 cols (2 sticks) at
        # a time within each outer row-tile.
        out = compiled(A_spyre, B_spyre, 128, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


class TestForEachTileNestedCarryE2E(_DynamoResetTestCase):
    """Carry-based nested for_each_tile: nested_split_m_then_k_fn (depth=2)
    and the triple_nested_stardep_* family (depth=3), the fixtures named as
    a separate, not-yet-covered tier in TestForEachTileNestedMapE2E's own
    docstring. Unlike that class, every level here carries a matmul
    accumulation rather than being pure map, and the triple_nested_stardep_*
    variants each place a surviving STAR_DEP_KEPT tile_dim_marker at a
    different nesting level (see for_each_tile_fixtures.py's per-fixture
    docstrings), so a wrong per-level advance shows up as a large numeric
    mismatch rather than a compile-time error.

    Same dl16-rounded-reference, xavier-input tolerance rationale as
    TestForEachTileE2E: every level here carries a matmul accumulation, so
    the K-dimension magnitude blowup and fp16-vs-dl16 mismatch both apply.
    """

    ATOL = 1e-2
    RTOL = 1e-2

    def test_nested_split_m_then_k(self):
        """Depth=2: outer maps M, inner carries K (matmul accumulation)."""
        X = cached_xavier((256, 256))
        Y = cached_xavier((256, 64), differentiation=1)
        X_spyre, Y_spyre = X.to(DEVICE_NAME), Y.to(DEVICE_NAME)
        ref = nested_split_m_then_k_reference(
            dl16_round(X.float()), dl16_round(Y.float())
        )

        compiled = torch.compile(
            nested_split_m_then_k_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_triple_nested_stardep_outer(self):
        """Depth=3: surviving STAR_DEP_KEPT marker at the outer level."""
        X = cached_xavier((2, 256, 256))
        Y = cached_xavier((2, 256, 64), differentiation=1)
        X_spyre, Y_spyre = X.to(DEVICE_NAME), Y.to(DEVICE_NAME)
        ref = triple_nested_stardep_outer_reference(
            dl16_round(X.float()), dl16_round(Y.float())
        )

        compiled = torch.compile(
            triple_nested_stardep_outer_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_triple_nested_stardep_middle(self):
        """Depth=3: surviving STAR_DEP_KEPT markers at outer and middle."""
        X = cached_xavier((2, 256, 256))
        Y = cached_xavier((2, 256, 64), differentiation=1)
        X_spyre, Y_spyre = X.to(DEVICE_NAME), Y.to(DEVICE_NAME)
        ref = triple_nested_stardep_middle_reference(
            dl16_round(X.float()), dl16_round(Y.float())
        )

        compiled = torch.compile(
            triple_nested_stardep_middle_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_triple_nested_stardep_inner(self):
        """Depth=3: same outer STAR_DEP_KEPT marker as outer_fn; inner
        marker is INLINE_ERASED despite the fixture's name (see
        for_each_tile_fixtures.py's docstring)."""
        X = cached_xavier((2, 256, 256))
        Y = cached_xavier((2, 256, 64), differentiation=1)
        X_spyre, Y_spyre = X.to(DEVICE_NAME), Y.to(DEVICE_NAME)
        ref = triple_nested_stardep_inner_reference(
            dl16_round(X.float()), dl16_round(Y.float())
        )

        compiled = torch.compile(
            triple_nested_stardep_inner_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_triple_nested_stardep_multilevel(self):
        """Depth=3: same outer STAR_DEP_KEPT markers as outer_fn plus a
        middle-level `* 1.0`; the inner marker is lost regardless (see
        for_each_tile_fixtures.py's docstring)."""
        X = cached_xavier((2, 256, 256))
        Y = cached_xavier((2, 256, 64), differentiation=1)
        X_spyre, Y_spyre = X.to(DEVICE_NAME), Y.to(DEVICE_NAME)
        ref = triple_nested_stardep_multilevel_reference(
            dl16_round(X.float()), dl16_round(Y.float())
        )

        compiled = torch.compile(
            triple_nested_stardep_multilevel_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_nested_online_softmax(self):
        """Map-outer/carry-inner nesting with a multi-leaf carry.

        nested_online_softmax_fn maps Q-row-tiles around online_softmax_fn's
        own 3-leaf (m, denom, acc) carry over K/V tiles -- unlike
        test_batched_map_over_online_softmax_carry (which stages full K/V
        buffers in the outer loop and tiles Lk in the inner one), this
        fixture re-runs the ENTIRE online-softmax recurrence, including its
        own K/V tiling, once per outer Q-tile: the nesting wraps a full
        inner for_each_tile call rather than sharing one carry-tiling level
        across both loops.
        """
        Q = cached_xavier((LQ, D))
        K = cached_xavier((LK, D), differentiation=1)
        V = cached_xavier((LK, D), differentiation=2)
        ref = nested_online_softmax_reference(
            dl16_round(Q.float()), dl16_round(K.float()), dl16_round(V.float())
        )

        compiled = torch.compile(
            nested_online_softmax_fn, backend="inductor", fullgraph=True
        )
        out = compiled(Q.to(DEVICE_NAME), K.to(DEVICE_NAME), V.to(DEVICE_NAME))

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


class TestForEachTileNestedGatherE2E(_DynamoResetTestCase):
    """Kind.GATHER nested inside another for_each_tile level.

    paged_gather_fn/paged_gather_kv_fn (TestForEachTileE2E) exercise
    Kind.GATHER but only as a single loop level. paged_gather_nested_fn
    wraps an outer map over Q-row-tiles around that same gather-mode body
    (tiled block table, invariant page pool, one page gathered per trip),
    the shape paged attention would want with an outer query tile.
    """

    ATOL = 1e-2
    RTOL = 1e-2

    def test_gather_mode_nested_paged_pages(self):
        _, table, _ = paged_gather_inputs()
        pages = cached_xavier((PAGE_POOL, PAGE_SIZE, PAGE_HS))
        q = cached_xavier((PAGE_LQ, PAGE_HS), differentiation=1)
        ref = paged_gather_nested_reference(
            dl16_round(pages.float()), dl16_round(q.float())
        )

        compiled = torch.compile(
            paged_gather_nested_fn, backend="inductor", fullgraph=True
        )
        out = compiled(pages.to(DEVICE_NAME), table.to(DEVICE_NAME), q.to(DEVICE_NAME))

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


# --- trip-range vector gather (loop-trip ranges reach coordinate queries) -----

TRIP_POOL, TRIP_E, TRIP_SIZE, TRIP_HS, TRIP_LQ = 64, 4, 32, 64, 32


def trip_range_build(trips, e):
    pages = (
        torch.pow(torch.tensor(2.0), (torch.arange(TRIP_POOL) % 8).float()) / 64.0
    ).to(torch.float16)
    pages = (
        pages.reshape(TRIP_POOL, 1, 1)
        .expand(TRIP_POOL, TRIP_SIZE, TRIP_HS)
        .contiguous()
    )
    q = torch.full((TRIP_LQ, TRIP_HS), 1.0 / 64.0, dtype=torch.float16)
    table = torch.zeros(trips, 32, dtype=torch.int32)
    for t in range(trips):
        for j in range(e):
            table[t, j] = (t * e + j) % TRIP_POOL
    return pages, table, q


def trip_range_ref(pages, q, ids):
    pf, qf = pages.float(), q.float()
    acc = torch.zeros(TRIP_LQ, TRIP_HS)
    for p in ids:
        page = pf[int(p)]
        acc = acc + (qf @ page.T) @ page
    return acc


def trip_range_fn(e):
    from torch_spyre._inductor.wsr import for_each_tile

    def fn(pages, table, q):
        def body(acc, tiles):
            table_row, pages_all, q_whole = tiles
            idx = table_row[0, 0:e]
            pages_v = pages_all.index_select(0, idx)
            scores = torch.matmul(q_whole.unsqueeze(0), pages_v.transpose(-2, -1))
            out = torch.matmul(scores, pages_v)
            return acc + out.sum(0), None

        acc0 = torch.zeros(TRIP_LQ, TRIP_HS, dtype=q.dtype, device=q.device)
        final, _ = for_each_tile(
            body, (table, pages, q), dims=(0, None, None), tile_size=1, init=acc0
        )
        return final

    return fn


class TestForEachTileTripRangesE2E(_DynamoResetTestCase):
    """A VECTOR page gather per trip (vs paged_gather_fn's point read).

    On the base this fails at trips >= 2 with ``indirect symbol u0 not found in
    indirect_sizes``; with the fix it passes and the output is neither the first
    group repeated nor the advance applied twice.
    """

    ATOL = 1e-3
    RTOL = 1e-3

    def test_multi_trip_vector_page_gather(self):
        e = TRIP_E
        for trips in (1, 2, 4):
            with self.subTest(trips=trips):
                # Reset per case: without it Dynamo generalizes the fixed trip
                # counts across subtests and the for_each_tile splice is skipped.
                torch._dynamo.reset()
                pages, table, q = trip_range_build(trips, e)
                ids = [int(table[t, j]) for t in range(trips) for j in range(e)]
                want = trip_range_ref(pages, q, ids)
                compiled = torch.compile(
                    trip_range_fn(e), backend="inductor", fullgraph=True
                )
                out = (
                    compiled(
                        pages.to(DEVICE_NAME), table.to(DEVICE_NAME), q.to(DEVICE_NAME)
                    )
                    .cpu()
                    .float()
                )
                assert torch.isfinite(out).all()
                torch.testing.assert_close(out, want, atol=self.ATOL, rtol=self.RTOL)
                if trips >= 2:
                    rep_first = trip_range_ref(
                        pages,
                        q,
                        [int(table[0, j]) for _ in range(trips) for j in range(e)],
                    )
                    adv_twice = trip_range_ref(
                        pages,
                        q,
                        [
                            int(table[(2 * t) % trips, j])
                            for t in range(trips)
                            for j in range(e)
                        ],
                    )
                    assert (out - rep_first).abs().max().item() > 1e-2
                    assert (out - adv_twice).abs().max().item() > 1e-2


# --- sub-stick for_each_tile indirect-index advance (issue #4835) -----------


def substick_gather_fn(pool, ids, init):
    from torch_spyre._inductor.wsr import for_each_tile

    def fn(pool, ids, init):
        def body(carry, tiles):
            tile_ids, whole_pool = tiles
            return (carry[0] + whole_pool[tile_ids],), None

        (acc,), _ = for_each_tile(
            body, (ids, pool), dims=(0, None), tile_size=SUBSTICK_TILE, init=(init,)
        )
        return acc

    return fn(pool, ids, init)


SUBSTICK_POOL, SUBSTICK_WIDTH, SUBSTICK_TRIPS, SUBSTICK_TILE = 256, 128, 4, 2


class TestForEachTileSubStickAdvanceE2E(_DynamoResetTestCase):
    """A ``for_each_tile`` whose per-trip indirect-index advance is narrower
    than one physical stick must be refused at compile time, not silently
    compiled to a wrong answer.

    ``ids`` is an int32 tile-advancing (``Kind.SLICE``) operand tiled with
    ``tile_size=SUBSTICK_TILE=2``; each trip therefore advances the index
    tensor's device stick coordinate by 2 int32 elements, well inside a
    single 32-element stick. Before the ``UnalignedStickSplit`` guard added
    by PR #4829, ``SpyreKernel`` computed this sub-stick advance as if it
    were whole-stick, repeating the first tile's gather on every subsequent
    trip. #4829's own regression coverage of that guard
    (``TestIndirectIndexStepGuard`` in test_for_each_tile_lowering.py) only
    exercises it via a synthetic CPU ``sympy`` expression; no test compiles
    an actual sub-stick advance on device. This test closes that gap: it
    asserts that compiling ``substick_gather_fn`` raises the documented
    ``UnalignedStickSplit`` failure (surfaced through the wrapping
    ``InductorError``) instead of returning a plausible-looking wrong
    answer. See issue #4835 and the parent issue #4828.
    """

    def test_substick_indirect_advance_is_refused(self):
        import pytest
        from torch._inductor.exc import InductorError

        ids = torch.arange(SUBSTICK_TRIPS, dtype=torch.int32).to(DEVICE_NAME)
        pool = torch.randn(SUBSTICK_POOL, SUBSTICK_WIDTH, dtype=torch.float16).to(
            DEVICE_NAME
        )
        init = torch.zeros(SUBSTICK_TILE, SUBSTICK_WIDTH, dtype=torch.float16).to(
            DEVICE_NAME
        )

        compiled = torch.compile(substick_gather_fn, backend="inductor", fullgraph=True)
        with pytest.raises(InductorError, match="cuts tensor.*physical stick"):
            compiled(pool, ids, init)


if __name__ == "__main__":
    unittest.main()
