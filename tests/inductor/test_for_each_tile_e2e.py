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
Case 5 (nested for_each_tile) covers both pure map/map nesting in
TestForEachTileNestedMapE2E and map/carry nesting in
test_batched_map_over_online_softmax_carry.  The latter stages full K/V
buffers in the outer batch map, then slices those staged buffers in the inner
Lk carry, exercising per-level ownership of input advances. Case 6
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

from tests.inductor.for_each_tile_fixtures import (
    batched_online_softmax_fn,
    STICK_COLS,
    STICK_ROWS,
    abs_add_mul_tiled_fn,
    abs_add_mul_tiled_reference,
    abs_tiled_fn,
    abs_tiled_reference,
    add_tiled_fn,
    add_tiled_reference,
    attention_inputs,
    matmul_inputs,
    nested_add_outer_row_inner_col_fn,
    nested_add_outer_row_inner_col_reference,
    online_softmax_fn,
    online_softmax_reference,
    paged_gather_fn,
    paged_gather_inputs,
    paged_gather_kv_fn,
    paged_gather_kv_inputs,
    paged_gather_kv_reference,
    paged_gather_reference,
    pointwise_inputs,
    softmax_row_tiled_fn,
    softmax_row_tiled_reference,
    split_k_fn,
    split_m_fn,
)


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
    # Spyre's matmul runs in fp16, so the reference has to be an fp16-faithful
    # one: cast the operands first, then accumulate in fp32 on CPU. Comparing
    # against the fp32 product of fp32 operands would fail on rounding alone,
    # independently of anything this test is meant to check.
    #
    # Operands must also be cast to fp16 BEFORE the host->device transfer, not
    # after: `t.to(DEVICE_NAME).half()` (transfer fp32, cast on device)
    # currently produces garbage on this backend for reasons unrelated to
    # while_loop lowering -- a plain `torch.compile`d `a @ b` reproduces it
    # without any for_each_tile involved. `t.half().to(DEVICE_NAME)` is the
    # idiom the rest of the compiled-op suite uses (see
    # tests/inductor/test_inductor_matmul.py, whose inputs are constructed
    # `dtype=torch.float16` up front).
    #
    # rtol is the binding constraint here: operand/output magnitudes are
    # O(1)-O(10), so rtol * |expected| dominates atol (which only matters
    # near zero).
    ATOL = 0.1
    RTOL = 0.1

    @staticmethod
    def _operands():
        (X, Y), _ = matmul_inputs()
        ref = (X.half().float()) @ (Y.half().float())
        return X.half().to(DEVICE_NAME), Y.half().to(DEVICE_NAME), ref

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
        Q, K, V = attention_inputs()
        ref = online_softmax_reference(Q, K, V)

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
        torch.manual_seed(0)
        Q = torch.randn(2, 64, 128, dtype=torch.float16)
        K = torch.randn(2, 256, 128, dtype=torch.float16)
        V = torch.randn(2, 256, 128, dtype=torch.float16)
        ref = torch.softmax(Q.float() @ K.float().transpose(-1, -2), dim=-1)
        ref = ref @ V.float()

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
        pages, table, q = paged_gather_inputs()
        ref = paged_gather_reference(pages, q)

        compiled = torch.compile(paged_gather_fn, backend="inductor", fullgraph=True)
        out = compiled(pages.to(DEVICE_NAME), table.to(DEVICE_NAME), q.to(DEVICE_NAME))

        # Looser than the class defaults: the accumulator sums four
        # score-weighted pages of magnitude ~sqrt(head_size), so fp16 matmul
        # rounding alone reaches a couple of absolute units here.
        torch.testing.assert_close(out.cpu().float(), ref, atol=2.0, rtol=0.05)

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
        k_pages, v_pages, table, q = paged_gather_kv_inputs()
        ref = paged_gather_kv_reference(k_pages, v_pages, q)

        compiled = torch.compile(paged_gather_kv_fn, backend="inductor", fullgraph=True)
        out = compiled(
            k_pages.to(DEVICE_NAME),
            v_pages.to(DEVICE_NAME),
            table.to(DEVICE_NAME),
            q.to(DEVICE_NAME),
        )

        # Same tolerance rationale as test_gather_mode_paged_pages above.
        torch.testing.assert_close(out.cpu().float(), ref, atol=2.0, rtol=0.05)


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
    """

    ATOL = 0.1
    RTOL = 0.1

    def test_add_tiled_small(self):
        A, B, _ = pointwise_inputs()
        A_spyre, B_spyre = A.half().to(DEVICE_NAME), B.half().to(DEVICE_NAME)
        ref = add_tiled_reference(A.half().float(), B.half().float())

        compiled = torch.compile(add_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(A_spyre, B_spyre, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_add_tiled_multi_stick(self):
        A = torch.randn(STICK_ROWS, STICK_COLS)
        B = torch.randn(STICK_ROWS, STICK_COLS)
        A_spyre, B_spyre = A.half().to(DEVICE_NAME), B.half().to(DEVICE_NAME)
        ref = add_tiled_reference(A.half().float(), B.half().float())

        compiled = torch.compile(add_tiled_fn, backend="inductor", fullgraph=True)
        # tile_size=128 rows, full 128-col width -> 2 sticks/row per tile.
        out = compiled(A_spyre, B_spyre, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_tiled_small(self):
        A, _, _ = pointwise_inputs()
        A_spyre = A.half().to(DEVICE_NAME)
        ref = abs_tiled_reference(A.half().float())

        compiled = torch.compile(abs_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(A_spyre, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_tiled_multi_stick(self):
        A = torch.randn(STICK_ROWS, STICK_COLS)
        A_spyre = A.half().to(DEVICE_NAME)
        ref = abs_tiled_reference(A.half().float())

        compiled = torch.compile(abs_tiled_fn, backend="inductor", fullgraph=True)
        out = compiled(A_spyre, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_add_mul_tiled_small(self):
        A, B, C = pointwise_inputs()
        A_spyre = A.half().to(DEVICE_NAME)
        B_spyre = B.half().to(DEVICE_NAME)
        C_spyre = C.half().to(DEVICE_NAME)
        ref = abs_add_mul_tiled_reference(
            A.half().float(), B.half().float(), C.half().float()
        )

        compiled = torch.compile(
            abs_add_mul_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(A_spyre, B_spyre, C_spyre, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_abs_add_mul_tiled_multi_stick(self):
        A = torch.randn(STICK_ROWS, STICK_COLS)
        B = torch.randn(STICK_ROWS, STICK_COLS)
        C = torch.randn(STICK_ROWS, STICK_COLS)
        A_spyre, B_spyre, C_spyre = (
            A.half().to(DEVICE_NAME),
            B.half().to(DEVICE_NAME),
            C.half().to(DEVICE_NAME),
        )
        ref = abs_add_mul_tiled_reference(
            A.half().float(), B.half().float(), C.half().float()
        )

        compiled = torch.compile(
            abs_add_mul_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(A_spyre, B_spyre, C_spyre, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_softmax_row_tiled_small(self):
        X, _, _ = pointwise_inputs()
        X_spyre = X.half().to(DEVICE_NAME)
        ref = softmax_row_tiled_reference(X.half().float())

        compiled = torch.compile(
            softmax_row_tiled_fn, backend="inductor", fullgraph=True
        )
        out = compiled(X_spyre, 2)

        torch.testing.assert_close(out.cpu().float(), ref, atol=0.02, rtol=0.1)

    def test_softmax_row_tiled_multi_stick(self):
        """Row-tile size spans 2 sticks/row -- see test_hint_softmax_row_tiling."""
        X = torch.rand(STICK_ROWS, STICK_COLS)
        X_spyre = X.half().to(DEVICE_NAME)
        ref = softmax_row_tiled_reference(X.half().float())

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
        A, B, _ = pointwise_inputs(rows=4, cols=8)
        A_spyre, B_spyre = A.half().to(DEVICE_NAME), B.half().to(DEVICE_NAME)
        ref = nested_add_outer_row_inner_col_reference(
            A.half().float(), B.half().float()
        )

        compiled = torch.compile(
            nested_add_outer_row_inner_col_fn, backend="inductor", fullgraph=True
        )
        out = compiled(A_spyre, B_spyre, 2, 2)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_nested_add_outer_row_inner_col_multi_stick(self):
        A = torch.randn(STICK_ROWS, STICK_COLS)
        B = torch.randn(STICK_ROWS, STICK_COLS)
        A_spyre, B_spyre = A.half().to(DEVICE_NAME), B.half().to(DEVICE_NAME)
        ref = nested_add_outer_row_inner_col_reference(
            A.half().float(), B.half().float()
        )

        compiled = torch.compile(
            nested_add_outer_row_inner_col_fn, backend="inductor", fullgraph=True
        )
        # Outer tiles 128 rows at a time; inner tiles 128 cols (2 sticks) at
        # a time within each outer row-tile.
        out = compiled(A_spyre, B_spyre, 128, 128)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


if __name__ == "__main__":
    unittest.main()
