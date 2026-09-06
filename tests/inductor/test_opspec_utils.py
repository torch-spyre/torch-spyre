# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Unit tests for the pure OpSpec-reading helpers in ``opspec_utils``.

These helpers (buffer identity, row-major strides, device-dim classification,
reshape/broadcast alignment) are pure sympy/int computations with no MLIR
emission, so this suite runs in CI without ``mlir_ktdp`` installed -- unlike
``test_ktir_emitter`` which pins the emitted MLIR text and skips without it.
"""

import unittest

import sympy
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.opspec_utils import (
    _DIM_BARE,
    _DIM_CONST,
    _DIM_OUTER_STICK,
    _DIM_WITHIN_STICK,
    PARALLEL,
    REDUCTION,
    _dim_info,
    _iteration_space_key,
    align_reshape_plan,
    buf_id,
    core_divisions,
    per_core_extent,
    placeholder_axes,
    reduction_indexing,
    row_major_strides,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _arg(name, arg_index=0) -> TensorArg:
    """Minimal TensorArg; only ``name`` matters for ``buf_id``."""
    return TensorArg(
        is_input=True,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[64],
        device_coordinates=[sympy.Symbol("d0")],
        allocation={"hbm": None},
        name=name,
    )


class TestRowMajorStrides(unittest.TestCase):
    def test_rank3(self):
        self.assertEqual(row_major_strides([16, 512, 64]), [32768, 64, 1])

    def test_rank1(self):
        self.assertEqual(row_major_strides([64]), [1])

    def test_rank2(self):
        self.assertEqual(row_major_strides([8, 3]), [3, 1])


class TestBufId(unittest.TestCase):
    def test_name_is_identity(self):
        self.assertEqual(buf_id(_arg("buf0")), "buf0")

    def test_same_name_same_id_across_input_output(self):
        # An intermediate appearing as both input and output (arg_index sentinel
        # -1 either side) must key on the shared name, not arg_index.
        as_in = _arg("buf7", arg_index=-1)
        as_out = _arg("buf7", arg_index=-1)
        self.assertEqual(buf_id(as_in), buf_id(as_out))

    def test_none_name_raises_value_error(self):
        # Missing name is a broken internal invariant, not a capability gap:
        # ValueError, not NotImplementedError.
        with self.assertRaises(ValueError):
            buf_id(_arg(None))


class TestDimInfo(unittest.TestCase):
    def test_const_dim(self):
        self.assertEqual(_dim_info(sympy.Integer(1)), (_DIM_CONST, None))

    def test_bare_symbol(self):
        d0 = sympy.Symbol("d0")
        self.assertEqual(_dim_info(d0), (_DIM_BARE, d0))

    def test_within_stick(self):
        d0 = sympy.Symbol("d0")
        kind, sym = _dim_info(sympy.Mod(d0, 64))
        self.assertEqual(kind, _DIM_WITHIN_STICK)
        self.assertEqual(sym, d0)

    def test_outer_stick(self):
        """Both spellings: torch's FloorDiv, and the sympy floor the projection
        actually produces (which used to be classified as neither)."""
        d0 = sympy.Symbol("d0")
        for coord in (FloorDiv(d0, 64), sympy.floor(d0 / 64)):
            with self.subTest(coord=coord):
                kind, sym = _dim_info(coord)
                self.assertEqual(kind, _DIM_OUTER_STICK)
                self.assertEqual(sym, d0)

    def test_multi_symbol_raises(self):
        d0, d1 = sympy.symbols("d0 d1")
        with self.assertRaises(NotImplementedError):
            _dim_info(d0 * 8 + d1)

    def test_single_symbol_unknown_form_raises(self):
        # A single-symbol coordinate that is neither bare, within-stick, nor
        # outer-stick (e.g. ``2*d0 + 1``) must raise, not fall through.
        d0 = sympy.Symbol("d0")
        with self.assertRaises(NotImplementedError):
            _dim_info(2 * d0 + 1)


class TestIterationSpaceKey(unittest.TestCase):
    def test_order_independent(self):
        d0, d1 = sympy.symbols("d0 d1")
        spec_a = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (16, 1), d1: (512, 1)},
            args=[],
            op_info={},
        )
        spec_b = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d1: (512, 1), d0: (16, 1)},
            args=[],
            op_info={},
        )
        self.assertEqual(_iteration_space_key(spec_a), _iteration_space_key(spec_b))

    def test_distinct_ranges_differ(self):
        d0 = sympy.Symbol("d0")
        spec_a = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (16, 1)},
            args=[],
            op_info={},
        )
        spec_b = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (32, 1)},
            args=[],
            op_info={},
        )
        self.assertNotEqual(_iteration_space_key(spec_a), _iteration_space_key(spec_b))


class TestAlignReshapePlan(unittest.TestCase):
    def test_identity_returns_none(self):
        d0, d1 = sympy.symbols("d0 d1")
        self.assertIsNone(align_reshape_plan([d0, d1], [16, 64], [d0, d1], [16, 64]))

    def test_broadcast_unmatched_output_axis(self):
        # Input [a, within] aligned into output [a, b, within]: the output's
        # ``b`` axis has no input counterpart, so it reshapes to extent 1 and
        # then broadcasts to the output block.
        a, b, c = sympy.symbols("a b c")
        plan = align_reshape_plan(
            [a, sympy.Mod(c, 64)],
            [16, 64],
            [a, b, sympy.Mod(c, 64)],
            [16, 8, 64],
        )
        self.assertEqual(plan, ([16, 1, 64], [16, 8, 64]))

    def test_transpose_raises(self):
        # Matched input axes in decreasing order -> would need a permute.
        a, b, c = sympy.symbols("a b c")
        with self.assertRaises(NotImplementedError):
            align_reshape_plan(
                [b, a, sympy.Mod(c, 64)],
                [2, 3, 64],
                [a, b, sympy.Mod(c, 64)],
                [3, 2, 64],
            )

    def test_dropped_extent_raises(self):
        # An input axis with extent > 1 that no output axis matches would lose
        # data -> needs a cross-stick transpose (restickify), not a reshape.
        a, d, c = sympy.symbols("a d c")
        with self.assertRaises(NotImplementedError):
            align_reshape_plan(
                [a, sympy.Mod(c, 64)],
                [4, 64],
                [d, sympy.Mod(c, 64)],
                [4, 64],
            )


class TestCoreDivisions(unittest.TestCase):
    """The grid, read off the iteration space's per-symbol work division."""

    def test_undivided_is_one_core(self):
        d0, d1 = sympy.symbols("d0 d1")
        self.assertEqual(core_divisions({d0: (16, 1), d1: (512, 1)}), ([], 1))

    def test_one_divided_symbol_owns_the_whole_grid(self):
        d0, d1 = sympy.symbols("d0 d1")
        divisions, cores = core_divisions({d0: (16, 1), d1: (512, 32)})
        self.assertEqual(cores, 32)
        self.assertEqual(divisions, [(d1, 32, 1)])

    def test_two_divided_symbols_are_mixed_radix_innermost_first(self):
        """``inner`` is the grid stride of one step of that symbol, so the flat
        id decodes as ``(id // inner) % div``."""
        d0, d1 = sympy.symbols("d0 d1")
        divisions, cores = core_divisions({d0: (16, 2), d1: (512, 4)})
        self.assertEqual(cores, 8)
        self.assertEqual(divisions, [(d1, 4, 1), (d0, 2, 4)])


class TestPerCoreExtent(unittest.TestCase):
    """One core's share of each device axis, and which division it follows."""

    @staticmethod
    def _arg(size, coords):
        return TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=size,
            device_coordinates=coords,
            allocation={"hbm": None},
            name="arg0",
        )

    def test_undivided_is_the_whole_extent(self):
        d0, d1 = sympy.symbols("d0 d1")
        arg = self._arg([16, 512, 64], [d0, d1, sympy.Mod(d1, 64)])
        self.assertEqual(per_core_extent(arg, {}), ([16, 512, 64], [None, None, None]))

    def test_a_divided_axis_shrinks_and_names_its_symbol(self):
        d0, d1 = sympy.symbols("d0 d1")
        arg = self._arg([16, 512, 64], [d0, d1, sympy.Mod(d1, 64)])
        self.assertEqual(
            per_core_extent(arg, {d1: 32}), ([16, 16, 64], [None, d1, None])
        )

    def test_the_within_stick_axis_is_never_divided(self):
        """A stick is the unit of transfer, so the last axis keeps its extent even
        when its symbol is divided."""
        d0 = sympy.symbols("d0")
        arg = self._arg([32, 64], [sympy.floor(d0 / 64), sympy.Mod(d0, 64)])
        self.assertEqual(per_core_extent(arg, {d0: 32}), ([1, 64], [d0, None]))

    def test_a_ragged_division_raises(self):
        d0, d1 = sympy.symbols("d0 d1")
        arg = self._arg([16, 512, 64], [d0, d1, sympy.Mod(d1, 64)])
        with self.assertRaises(NotImplementedError):
            per_core_extent(arg, {d1: 7})


class TestPlaceholderAxes(unittest.TestCase):
    """Which output axes stand in for something the op does not write.

    The rule is about the *extent*, not about reductions: a constant coordinate
    alone does not say "no elements here".  Both projections below carry a
    constant coordinate the store really walks, or would if the extent were the
    only thing consulted -- which is why this helper is unary and separate.
    """

    def test_a_nonstick_reduction_drops_only_its_unit_axis(self):
        """``sum(x[256, 2048], dim=0)``: the reduced axis survives in the output
        as a unit extent at a constant coordinate, and that is the one to go."""
        lanes = sympy.Symbol("c0")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        self.assertEqual(
            placeholder_axes([sympy.Integer(0), stick, lane], [1, 32, 64]), (0,)
        )

    def test_an_on_stick_reduction_keeps_its_broadcast_lane(self):
        """``sum(x[256, 128], dim=-1)``: device ``[1, 256, 64]`` at coordinates
        ``[0, c0, 0]``, i.e. **two** constant coordinates meaning different things.

        Axis 0 is the placeholder -- no elements, no stride.  Axis 2 is the
        broadcast lane, 64 real elements the D2H descriptor gathers across, and
        dropping it would describe a store of a fraction of the buffer.  Only the
        extent separates them, so this is the evidence that the rule is about
        extent rather than about reductions at all.
        """
        rows = sympy.Symbol("c0")
        self.assertEqual(
            placeholder_axes([sympy.Integer(0), rows, sympy.Integer(0)], [1, 256, 64]),
            (0,),
        )


class TestReductionIndexing(unittest.TestCase):
    """The iteration nest a reduction wants, read from the coordinates.

    Both patterns the frontend projects, against a *squeezed* output -- which is
    the helper's precondition, because an extent-1 constant is otherwise
    indistinguishable from a degenerate broadcast lane.
    """

    @staticmethod
    def _nonstick():
        """``sum(x[256, 2048], dim=0)``: input ``[32, 256, 64]`` -> ``[32, 64]``.

        The reduced axis (256 rows) sits between the two axes the reduced-over
        stick index splits, and the output keeps both of those.
        """
        lanes, rows = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        return ([stick, rows, lane], [32, 256, 64], [stick, lane], [32, 64])

    @staticmethod
    def _on_stick():
        """``sum(x[256, 128], dim=-1)``: input ``[2, 256, 64]`` -> ``[256, 64]``.

        The reduction runs along the *stick*, so it consumes both halves of the
        reduced symbol -- the outer-stick chunk index and the within-stick lane --
        and the output nonetheless has 64 lanes: a fresh iteration dim with no
        input axis behind it.
        """
        rows, reduced = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
        return (
            [stick, rows, lane],
            [2, 256, 64],
            [rows, sympy.Integer(0)],
            [256, 64],
        )

    def test_a_nonstick_reduction_is_one_reduced_dim(self):
        """Identity input map, output map the identity with dim 1 dropped -- which
        is exactly what ``linalg.reduce ... dimensions = [1]`` states."""
        iters, in_map, out_map = reduction_indexing(*self._nonstick())
        self.assertEqual(iters, (PARALLEL, REDUCTION, PARALLEL))
        self.assertEqual(in_map, (0, 1, 2))
        self.assertEqual(out_map, (0, 2))

    def test_an_on_stick_reduction_reduces_a_dim_and_keeps_a_lane(self):
        """Two reduced dims, a rank-3 input map against a rank-4 nest, and a dim
        (3) the output has that the input does not.

        This is the shape a set complement over kept axes cannot state: the lane
        axis is reduced on the way in and written on the way out, so no flat list
        of reduced axes describes it and the answer has to be per-operand maps.
        """
        iters, in_map, out_map = reduction_indexing(*self._on_stick())
        self.assertEqual(iters, (REDUCTION, PARALLEL, REDUCTION, PARALLEL))
        self.assertEqual(in_map, (0, 1, 2))
        self.assertEqual(out_map, (1, 3))

    def test_nothing_reduced_raises(self):
        """Not implied by the caller calling it a reduction: that is what was
        asked for, this is what the coordinates say.  An all-parallel nest would
        reach ``dimensions = []``, which does not build."""
        lanes, rows = sympy.symbols("c0 c1")
        with self.assertRaises(NotImplementedError) as ctx:
            reduction_indexing([rows, lanes], [256, 64], [rows, lanes], [256, 64])
        self.assertIn("no axis to reduce", str(ctx.exception))

    def test_a_resized_kept_axis_raises(self):
        """A kept axis whose extent changed is not a reduction of the others."""
        in_coords, in_extent, out_coords, _ = self._nonstick()
        with self.assertRaises(NotImplementedError) as ctx:
            reduction_indexing(in_coords, in_extent, out_coords, [16, 64])
        self.assertIn("does not resize", str(ctx.exception))

    def test_permuted_or_unmatched_kept_axes_raise(self):
        """A kept axis out of increasing order, or matching no input axis at all:
        both are a transpose (restickify), not a reduction."""
        lanes, rows = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        elsewhere = sympy.Symbol("d9")
        for out_coords, out_extent in (
            ([lane, stick], [64, 32]),  # matched in decreasing order
            ([elsewhere, lane], [32, 64]),  # a symbol the input does not carry
        ):
            with self.subTest(out_coords=out_coords):
                with self.assertRaises(NotImplementedError) as ctx:
                    reduction_indexing(
                        [stick, rows, lane], [32, 256, 64], out_coords, out_extent
                    )
                self.assertIn("transpose", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
