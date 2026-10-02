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

"""Device-free tests for the tiling-option space and its enumeration.

The predicates are pure, so these tests need no solver and no device: they
build the same lightweight ``FixedTiledLayout`` ops the span-overflow tests use
and assert the returned ``TileSpec`` set directly. ``TestTilingSpace`` pins the
seam a generating search asks instead of the list -- that the space and the
enumeration answer alike, and what one move-alphabet step reaches.
"""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.plan_solver import TileAxis, TileSpec
from torch_spyre._inductor.wsr.enumerate_tilings import (
    _MAX_AUTO_TILE_SPLIT_COUNT,
    _reduction_split_counts,
    build_tiling_space,
    enumerate_tile_options,
)
from utils_inductor import fixed_tiled_layout


# ---------------------------------------------------------------------------
# Device-free op builders (mirrors test_span_overflow_hint_analysis.py)
# ---------------------------------------------------------------------------
def _write_dep(name, shape, layout):
    # Integer, as Inductor's index symbols are, so the stick coordinate
    # simplifies to the form ``_stick_host_dim`` resolves.
    syms = sympy.symbols(
        " ".join(f"d{i}" for i in range(len(shape))), integer=True, nonnegative=True
    )
    if not isinstance(syms, tuple):
        syms = (syms,)
    index = sympy.Integer(0)
    for sym, stride in zip(syms, layout.stride):
        index += sym * int(stride)
    return MemoryDep(name, index, syms, tuple(shape)), syms


def _pointwise_op(shape, name="buf0"):
    data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = fixed_tiled_layout(shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name
    write, _ = _write_dep(name, shape, layout)
    op.get_read_writes = MagicMock(
        return_value=SimpleNamespace(reads=set(), writes={write})
    )
    return op


def _reduction_op(out_shape, reduction_ranges, name="buf0", reduction_type="sum"):
    """A Reduction op whose read dep carries real reduction loop vars.

    ``reduction_loop_vars`` derives the reduction symbols by subtracting the
    output write dep's symbols from the input read dep's symbols, so the read
    dep must range over both. The input has no resolvable device layout here, so
    the enumerator's per-input reduction stick check is skipped (returns clean).
    """
    data = MagicMock(spec=Reduction)
    data.ranges = list(out_shape)
    data.reduction_ranges = list(reduction_ranges)
    data.reduction_type = reduction_type
    layout = fixed_tiled_layout(out_shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name

    write, out_syms = _write_dep(name, out_shape, layout)
    red_syms = sympy.symbols(" ".join(f"r{i}" for i in range(len(reduction_ranges))))
    if not isinstance(red_syms, tuple):
        red_syms = (red_syms,)
    read_index = sympy.Integer(0)
    for sym, size in zip(out_syms + red_syms, list(out_shape) + list(reduction_ranges)):
        read_index += sym
    read = MemoryDep(
        f"in_{name}",
        read_index,
        out_syms + red_syms,
        tuple(out_shape) + tuple(reduction_ranges),
    )
    op.get_read_writes = MagicMock(
        return_value=SimpleNamespace(reads={read}, writes={write})
    )
    return op


def _exact_divisor_splits(n, max_split=_MAX_AUTO_TILE_SPLIT_COUNT):
    """Independent reference: exact divisors of ``n`` in ``(1, max_split]``."""
    return sorted(k for k in range(2, min(n, max_split) + 1) if n % k == 0)


def _expected_output_specs(dim_sizes, stick_dim, max_dims):
    """Brute-force reference set for a mock whose only stick constraint is that
    splitting a non-stick dim never cuts the last-dim sticks."""
    per_dim = {}
    for d, n in enumerate(dim_sizes):
        if d == stick_dim:
            continue
        splits = _exact_divisor_splits(n)
        if splits:
            per_dim[d] = splits
    specs = {TileSpec()}
    dims = sorted(per_dim)
    for k in range(1, min(max_dims, len(dims)) + 1):
        for combo in itertools.combinations(dims, k):
            for splits in itertools.product(*[per_dim[d] for d in combo]):
                specs.add(
                    TileSpec(tuple(TileAxis(d, s) for d, s in zip(combo, splits)))
                )
    return specs


class TestOutputEnumeration(unittest.TestCase):
    def test_untiled_option_present_and_first(self):
        opts = enumerate_tile_options(_pointwise_op((512, 256, 128)))
        self.assertTrue(opts[0].is_untiled)
        self.assertEqual(opts.count(TileSpec()), 1)

    def test_no_span_pressure_still_yields_more_than_untiled(self):
        # A splittable op with no overflow must still offer real tilings.
        opts = enumerate_tile_options(_pointwise_op((512, 256, 128)))
        self.assertGreater(len(opts), 1)

    def test_stick_dim_never_tiled(self):
        # The innermost dim (2) is the stick dim; it must never appear.
        opts = enumerate_tile_options(_pointwise_op((512, 256, 128)))
        for spec in opts:
            for axis in spec.axes:
                self.assertNotEqual(axis.host_dim, 2, spec.label)

    def test_an_unnamed_stick_dim_offers_no_output_tiling(self):
        # The size-based resolver would name dim 2 here; it is not consulted.
        op = _pointwise_op((512, 256, 128))
        self.assertFalse(build_tiling_space(op).is_empty)  # non-vacuity
        with patch(
            "torch_spyre._inductor.wsr.enumerate_tilings._stick_host_dim",
            return_value=None,
        ):
            self.assertTrue(build_tiling_space(op).is_empty)

    def test_no_device_layout_offers_no_output_tiling(self):
        op = _pointwise_op((512, 256, 128))
        op.layout.device_layout = None
        self.assertTrue(build_tiling_space(op).is_empty)

    def test_all_output_splits_are_exact_divisors(self):
        shape = (512, 256, 128)
        for spec in enumerate_tile_options(_pointwise_op(shape)):
            for axis in spec.axes:
                self.assertEqual(shape[axis.host_dim] % axis.count, 0, spec.label)
                self.assertLessEqual(axis.count, _MAX_AUTO_TILE_SPLIT_COUNT)

    def test_matches_brute_force_reference(self):
        # The returned set equals an independently computed divisor set.
        shape = (512, 256, 128)
        opts = enumerate_tile_options(_pointwise_op(shape), max_options=1000)
        expected = _expected_output_specs(shape, stick_dim=2, max_dims=2)
        self.assertEqual(set(opts), expected)
        # No duplicates.
        self.assertEqual(len(opts), len(set(opts)))

    def test_max_dims_one_gives_no_nested_specs(self):
        opts = enumerate_tile_options(_pointwise_op((512, 256, 128)), max_dims=1)
        self.assertTrue(all(spec.depth <= 1 for spec in opts))

    def test_max_options_truncates_but_keeps_untiled(self):
        opts = enumerate_tile_options(_pointwise_op((512, 256, 128)), max_options=5)
        self.assertEqual(len(opts), 5)
        self.assertTrue(opts[0].is_untiled)  # mandatory, never dropped

    def test_non_computed_buffer_returns_only_untiled(self):
        opts = enumerate_tile_options(MagicMock())
        self.assertEqual(opts, [TileSpec()])


class TestReductionEnumeration(unittest.TestCase):
    def setUp(self):
        self._patch = patch.object(config, "enable_reduction_tiling", True)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_reduction_split_counts_are_divisors_without_unit_tile(self):
        op = _reduction_op((256,), (64,))
        counts = _reduction_split_counts(op, 0)
        # exact divisors of 64 greater than 1, minus the unit-tile split (64).
        self.assertEqual(counts, [2, 4, 8, 16, 32])
        self.assertNotIn(64, counts)  # 64/64 == 1 element per tile: rejected

    def test_reduction_options_are_single_level(self):
        op = _reduction_op((256,), (64,))
        opts = enumerate_tile_options(op)
        red_opts = [s for s in opts if any(a.is_reduction for a in s.axes)]
        self.assertTrue(red_opts, "expected reduction options")
        for spec in red_opts:
            self.assertEqual(spec.depth, 1)
            self.assertTrue(spec.axes[0].is_reduction)

    def test_reduction_gated_on_config(self):
        op = _reduction_op((256,), (64,))
        with patch.object(config, "enable_reduction_tiling", False):
            opts = enumerate_tile_options(op)
        self.assertFalse(
            any(a.is_reduction for s in opts for a in s.axes),
            "reduction options must be gated on enable_reduction_tiling",
        )

    def test_reduction_split_counts_prime_extent_is_untileable(self):
        # A prime reduction extent has only the unit-tile split, which is
        # rejected -> no reduction options.
        op = _reduction_op((256,), (7,))
        self.assertEqual(_reduction_split_counts(op, 0), [])


class TestNoBadReductionOptions(unittest.TestCase):
    """Never a nested output+reduction spec or a multi-reduction spec."""

    def test_no_mixed_or_multi_reduction_specs(self):
        with patch.object(config, "enable_reduction_tiling", True):
            for shape, rranges in [((256,), (64,)), ((512, 256), (128,))]:
                op = _reduction_op(shape, rranges)
                for spec in enumerate_tile_options(op):
                    red_axes = [a for a in spec.axes if a.is_reduction]
                    out_axes = [a for a in spec.axes if not a.is_reduction]
                    # Never two reduction axes in one spec.
                    self.assertLessEqual(len(red_axes), 1, spec.label)
                    # Never an output axis and a reduction axis together.
                    self.assertFalse(red_axes and out_axes, spec.label)


def _has_reduction(spec: TileSpec) -> bool:
    return any(axis.is_reduction for axis in spec.axes)


class TestTilingSpace(unittest.TestCase):
    """The space a generating search asks instead of the list: what it admits,
    and what one move-alphabet step reaches."""

    def test_admits_rejects_what_the_enumeration_never_emits(self):
        space = build_tiling_space(_pointwise_op((512, 256, 128)))
        self.assertFalse(space.admits(TileSpec((TileAxis(2, 2),))))  # stick dim
        self.assertFalse(space.admits(TileSpec((TileAxis(0, 3),))))  # not a divisor
        self.assertFalse(  # the same dim twice
            space.admits(TileSpec((TileAxis(0, 2), TileAxis(0, 4))))
        )
        self.assertFalse(  # deeper than max_dims
            space.admits(TileSpec((TileAxis(0, 2), TileAxis(1, 2), TileAxis(0, 2))))
        )

    def test_a_reduction_level_is_never_admitted(self):
        with patch.object(config, "enable_reduction_tiling", True):
            op = _reduction_op((512, 256), (128,))
            space = build_tiling_space(op)
            red = TileSpec((TileAxis(host_dim=0, count=2, is_reduction=True),))
            self.assertIn(red, enumerate_tile_options(op))  # non-vacuity
        self.assertFalse(space.admits(red))

    def test_admits_is_exactly_membership_in_the_enumeration(self):
        # The safety argument for consuming the space's predicates instead of
        # the enumerator's list, asserted in the direction that can break it:
        # over *permutations*, not combinations, so a level order the
        # enumeration cannot emit has to be refused.
        space = build_tiling_space(_pointwise_op((512, 256, 128)))
        emitted = set(space.enumerate())
        for depth in range(1, space.max_dims + 2):
            for dims in itertools.permutations(space.output_dims, depth):
                for counts in itertools.product(*(space.counts(d) for d in dims)):
                    spec = TileSpec(tuple(TileAxis(d, c) for d, c in zip(dims, counts)))
                    self.assertEqual(space.admits(spec), spec in emitted, spec.label)

    def test_every_neighbour_is_admitted_and_one_edit_away(self):
        space = build_tiling_space(_pointwise_op((512, 256, 128)))
        emitted = set(space.enumerate())
        for spec in space.enumerate():
            for candidate in space.neighbours(spec):
                self.assertIn(candidate, emitted, candidate.label)
                self.assertNotEqual(candidate, spec)
                self.assertLessEqual(abs(candidate.depth - spec.depth), 1)

    def test_the_alphabet_offers_every_move_type(self):
        space = build_tiling_space(_pointwise_op((512, 256, 128)))
        start = TileSpec((TileAxis(0, 2), TileAxis(1, 2)))
        self.assertTrue(space.admits(start))
        moves = space.neighbours(start)
        self.assertIn(TileSpec((TileAxis(0, 4), TileAxis(1, 2))), moves)  # recount
        self.assertIn(TileSpec((TileAxis(1, 2),)), moves)  # remove
        # Add: reachable from a shallower spec, since ``max_dims`` is 2 here.
        self.assertIn(
            TileSpec((TileAxis(0, 2), TileAxis(1, 2))),
            space.neighbours(TileSpec((TileAxis(0, 2),))),
        )

    def test_no_move_reorders_and_an_added_level_lands_canonically(self):
        # Nest order is not a decision variable: a swap would be a free,
        # always-accepted step, so the alphabet does not carry one and an added
        # level sorts into place instead of nesting innermost.
        space = build_tiling_space(_pointwise_op((512, 256, 128)))
        start = TileSpec((TileAxis(0, 2), TileAxis(1, 2)))
        self.assertNotIn(
            TileSpec((TileAxis(1, 2), TileAxis(0, 2))), space.neighbours(start)
        )
        self.assertFalse(space.admits(TileSpec((TileAxis(1, 2), TileAxis(0, 2)))))
        self.assertIn(
            TileSpec((TileAxis(0, 2), TileAxis(1, 4))),
            space.neighbours(TileSpec((TileAxis(1, 4),))),
        )

    def test_untiled_is_reachable_and_reaches_back(self):
        # Undividing has to stay a single move, or the walk cannot leave a
        # region it tiled.
        space = build_tiling_space(_pointwise_op((512, 256, 128)))
        self.assertIn(TileSpec(), space.neighbours(TileSpec((TileAxis(0, 2),))))
        self.assertIn(TileSpec((TileAxis(0, 2),)), space.neighbours(TileSpec()))

    def test_no_move_ever_proposes_a_reduction_level(self):
        # v1 scope: output-axis tiling only, so a search seeded untiled never
        # reaches a reduction spec even where the enumeration offers one.
        with patch.object(config, "enable_reduction_tiling", True):
            op = _reduction_op((512, 256), (128,))
            space = build_tiling_space(op)
            # Non-vacuity: the enumeration does offer them.
            self.assertTrue(any(_has_reduction(s) for s in enumerate_tile_options(op)))
        seen = {TileSpec()}
        frontier = [TileSpec()]
        while frontier:
            for candidate in space.neighbours(frontier.pop()):
                if candidate not in seen:
                    seen.add(candidate)
                    frontier.append(candidate)
        self.assertFalse(any(_has_reduction(spec) for spec in seen))

    def test_an_untileable_op_has_an_empty_space(self):
        space = build_tiling_space(MagicMock())
        self.assertTrue(space.is_empty)
        self.assertEqual(space.enumerate(), [TileSpec()])
        self.assertEqual(space.neighbours(TileSpec()), [])

    def test_an_op_tileable_only_on_a_reduction_has_an_empty_space(self):
        with patch.object(config, "enable_reduction_tiling", True):
            op = _reduction_op((7,), (128,))
            space = build_tiling_space(op)
            # Non-vacuity: the enumeration does offer one.
            self.assertTrue(any(_has_reduction(s) for s in enumerate_tile_options(op)))
        self.assertTrue(space.is_empty)
        self.assertEqual(space.neighbours(TileSpec()), [])

    def test_an_op_the_hint_passes_already_tiled_is_offered_nothing(self):
        # ``CoarseTilingPass`` stamps ``op.dim_hints`` wholesale, so a tiling
        # chosen here for an already-hinted op would clobber the group it is
        # part of. The marker both earlier passes leave set is the guard.
        op = _pointwise_op((512, 256, 128))
        self.assertFalse(build_tiling_space(op).is_empty)  # non-vacuity
        op.dim_hints = [object()]
        space = build_tiling_space(op)
        self.assertTrue(space.is_empty)
        self.assertEqual(space.enumerate(), [TileSpec()])


if __name__ == "__main__":
    unittest.main()
