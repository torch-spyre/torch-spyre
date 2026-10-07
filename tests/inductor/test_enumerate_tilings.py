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

"""Device-free tests for the tiling-option enumerator.

The enumerator is pure and unconsumed, so these tests need no solver and no
device: they build the same lightweight ``FixedTiledLayout`` ops the
span-overflow tests use and assert the returned ``TileSpec`` set directly.
"""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sympy
import torch

from torch import fx
from torch._dynamo.source import ConstantSource
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scratchpad.plan_solver import TileAxis, TileSpec
from torch_spyre._inductor.wsr.enumerate_tilings import (
    _MAX_AUTO_TILE_SPLIT_COUNT,
    _reduction_split_counts,
    enumerate_tile_options,
)


# ---------------------------------------------------------------------------
# Device-free op builders (mirrors test_span_overflow_hint_analysis.py)
# ---------------------------------------------------------------------------
def _fixed_tiled_layout(shape, dtype=torch.float16):
    """A physical layout whose within-stick (innermost) dim is the last one."""
    size = list(shape)
    stride = list(FlexibleLayout.contiguous_strides(size))
    stride_ints = [int(s) for s in stride]
    size_ints = [int(s) for s in size]
    within_stick_dim = len(size_ints) - 1
    dim_order = [i for i in range(len(size_ints)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size_ints, stride_ints, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _write_dep(name, shape, layout):
    syms = sympy.symbols(" ".join(f"d{i}" for i in range(len(shape))))
    if not isinstance(syms, tuple):
        syms = (syms,)
    index = sympy.Integer(0)
    for sym, stride in zip(syms, layout.stride):
        index += sym * int(stride)
    return MemoryDep(name, index, syms, tuple(shape)), syms


def _pointwise_op(shape, name="buf0"):
    data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = _fixed_tiled_layout(shape)
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
    layout = _fixed_tiled_layout(out_shape)
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


def _reader(ranges):
    """A ComputedBuffer reading the tiled op's output, over ``ranges``."""
    reader = MagicMock(spec=ComputedBuffer)
    reader.data = SimpleNamespace(ranges=list(ranges))
    return reader


def _counts(options, host_dim):
    """The single-level counts ``options`` offers for output ``host_dim``."""
    return [
        spec.axes[0].count
        for spec in options
        if spec.depth == 1
        and not spec.axes[0].is_reduction
        and spec.axes[0].host_dim == host_dim
    ]


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


class TestApplyRefusals(unittest.TestCase):
    """Counts the coarse-tile apply would refuse are not offered."""

    def test_a_folded_device_dim_admits_no_count(self):
        # The attention output [1, 64, hq, 128] lays heads and head_dim's outer
        # stick out as one device dim, which ``_resize_device_layout`` cannot
        # resize for any tile.
        op = _pointwise_op((1, 64, 40, 128))
        self.assertNotEqual(enumerate_tile_options(op), [TileSpec()])  # non-vacuity
        dl = op.layout.device_layout
        op.layout.device_layout = SpyreTensorLayout(
            [64, 80, 1, 64], [5120, 64, -1, 1], dl.device_dtype, dl.element_arrangement
        )
        self.assertEqual(enumerate_tile_options(op), [TileSpec()])

    def test_a_unit_tile_is_offered(self):
        # A dim tiled all the way down is offered, also when the tile then has
        # a second unit host dim: [1, 1, 2048] of [1, 64, 2048].
        for shape in ((8, 64, 128), (1, 64, 2048)):
            with self.subTest(shape=shape):
                self.assertIn(
                    64, _counts(enumerate_tile_options(_pointwise_op(shape)), 1)
                )

    def test_a_one_stick_tile_is_offered_for_a_reduction_too(self):
        # Halving a two-stick dim leaves a one-stick tile, whose tile-count
        # device dim is then one of two with extent 1. Nothing is grown back
        # from that tile -- the full buffer and the accumulator take the layout
        # planning recorded -- so the split is as legal for a Reduction as for
        # a Pointwise.
        from torch_spyre._inductor.wsr.span_overflow_hint_analysis import (
            _split_candidates_for_host_dim,
        )

        shape = (1, 6, 128)
        for kind, op in (
            ("reduction", _reduction_op(shape, (8,))),
            ("pointwise", _pointwise_op(shape)),
        ):
            with self.subTest(kind=kind):
                self.assertIn(2, _split_candidates_for_host_dim(op, 2))

    _SHAPE = (2, 8, 5, 64, 128)

    def test_a_reshaping_reader_drops_only_the_unit_tile(self):
        untouched = enumerate_tile_options(_pointwise_op(self._SHAPE))
        self.assertIn(64, _counts(untouched, 3))  # non-vacuity
        options = enumerate_tile_options(
            _pointwise_op(self._SHAPE), readers=[_reader((2, 8, 5, 64, 64))]
        )
        for dim in (0, 1, 2, 3):
            self.assertEqual(
                _counts(options, dim),
                [c for c in _counts(untouched, dim) if c != self._SHAPE[dim]],
            )

    def test_a_rank_changing_reader_drops_the_unit_tile(self):
        options = enumerate_tile_options(
            _pointwise_op(self._SHAPE), readers=[_reader((2, 40, 64, 128))]
        )
        self.assertNotIn(64, _counts(options, 3))
        self.assertIn(32, _counts(options, 3))

    def test_a_same_shape_or_non_computed_reader_keeps_the_unit_tile(self):
        extern = SimpleNamespace(data=SimpleNamespace(ranges=[2, 40, 64, 128]))
        for readers in ([], [_reader(self._SHAPE)], [extern]):
            options = enumerate_tile_options(
                _pointwise_op(self._SHAPE), readers=readers
            )
            self.assertIn(64, _counts(options, 3))
            self.assertEqual(_counts(options, 2), [5])

    def test_a_symbolic_dim_is_offered_no_tiling(self):
        # A recompile for a second shape leaves the changed dim symbolic in the
        # host layout while the device layout is built from its actual value.
        # The applier reads every extent of a tiled op as an int, so the op is
        # offered no tiling, not even on its static dims. The split helper the
        # span-overflow planner shares still reads the symbol's value and
        # returns the static dim's counts instead of raising.
        from torch_spyre._inductor.wsr.span_overflow_hint_analysis import (
            _split_candidates_for_host_dim,
        )

        shape = (4, 8, 256)
        static = enumerate_tile_options(_pointwise_op(shape))
        self.assertEqual(_counts(static, 1), [2, 4, 8])  # non-vacuity
        static_splits = _split_candidates_for_host_dim(_pointwise_op(shape), 1)
        with V.set_graph_handler(GraphLowering(fx.symbolic_trace(lambda: None))):
            s0 = V.graph.sizevars.shape_env.create_symbol(4, ConstantSource("s0"))
            op = _pointwise_op(shape)
            op.data.ranges = [s0, *shape[1:]]
            op.layout.size = [s0, *shape[1:]]
            options = enumerate_tile_options(op)
            splits = _split_candidates_for_host_dim(op, 1)
        self.assertEqual(options, [TileSpec()])
        self.assertEqual(splits, static_splits)

    def test_a_symbolic_stride_is_offered_no_tiling(self):
        # A mutation op inherits its target view's strides, so a stride can be
        # symbolic over static sizes. The applier orders strides by value, so
        # the op is offered no tiling.
        shape = (4, 8, 256)
        with V.set_graph_handler(GraphLowering(fx.symbolic_trace(lambda: None))):
            s0 = V.graph.sizevars.shape_env.create_symbol(8, ConstantSource("s0"))
            op = _pointwise_op(shape)
            self.assertEqual(_counts(enumerate_tile_options(op), 1), [2, 4, 8])
            op.layout.stride = [256 * s0, 256, 1]
            options = enumerate_tile_options(op)
        self.assertEqual(options, [TileSpec()])

    def test_a_symbolic_output_dim_withholds_reduction_options(self):
        # Reduction axes are resolved one dim at a time, past the resolver's
        # own symbolic-extent check, so the enumerator has to refuse the op
        # itself: the reduction dim is static, the output dim beside it is not.
        with patch.object(config, "enable_reduction_tiling", True):
            static = enumerate_tile_options(_reduction_op((4, 256), (8,)))
            self.assertTrue(any(s.axes[0].is_reduction for s in static if s.axes))
            with V.set_graph_handler(GraphLowering(fx.symbolic_trace(lambda: None))):
                s0 = V.graph.sizevars.shape_env.create_symbol(4, ConstantSource("s0"))
                op = _reduction_op((4, 256), (8,))
                op.data.ranges = [s0, 256]
                op.layout.size = [s0, 256]
                options = enumerate_tile_options(op)
        self.assertEqual(options, [TileSpec()])


if __name__ == "__main__":
    unittest.main()
