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

"""The loop form a symbolic trip count is emitted as, and its stride scale.

No device, no compile. These are pure functions on sympy expressions, so they
are checked directly.

The thing to understand before reading: a symbolic level is emitted as
``to <dim> step <G>``, so its loop variable counts ELEMENTS rather than tiles.
Every affine stride multiplying that variable therefore has to shrink by G.
The two decisions are one decision, and `TestStrideScaling` is where that is
pinned.
"""

import unittest

import sympy
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.codegen.bundle import (
    LoopLevel,
    _count_scale,
    _loop_level,
    _scaled_strides,
)
from torch_spyre._inductor.pass_utils import decompose_tiled_count
from torch_spyre._inductor.codegen.compute_ops import SymbolKind

G = 64
MAX = 512
SYM = "s0"
DIM_SSA = {SYM: "%sym_0_0"}


def _sym():
    return sympy.Symbol(SYM, integer=True, positive=True)


class TestDecompose(unittest.TestCase):
    """Which trip-count shapes the emitter recognises."""

    def test_floordiv_of_symbol_by_tile(self):
        self.assertEqual(decompose_tiled_count(FloorDiv(_sym(), G)), (_sym(), G))

    def test_bare_symbol_is_tile_size_one(self):
        self.assertEqual(decompose_tiled_count(_sym()), (_sym(), 1))

    def test_the_reload_spelling_of_the_same_division(self):
        """What the kernel serializer hands back, which is a different type.

        Expressions are written as ``sympify('<str>')`` and ``str(FloorDiv(s,
        G))`` prints ``(s//G)``, which sympy re-parses as ``floor(s/G)``. Same
        count, different class, so the reload path would refuse a loop it had
        just emitted if only one spelling were accepted.
        """
        reloaded = sympy.sympify(str(FloorDiv(_sym(), G)))

        self.assertNotIsInstance(
            reloaded,
            FloorDiv,
            "the round trip preserved FloorDiv on this sympy build, so this "
            "test is no longer exercising the second spelling",
        )

        symbol, tile = decompose_tiled_count(reloaded)
        # By name, because the round trip does not preserve symbol identity.
        # See test_the_round_trip_also_drops_the_symbol_assumptions.
        self.assertEqual((str(symbol), tile), (SYM, G))

    def test_the_round_trip_also_drops_the_symbol_assumptions(self):
        """Why nothing that outlives the reload may be keyed by the symbol.

        ``sympify`` builds a fresh ``Symbol`` with no assumptions, and sympy
        counts assumptions as part of a symbol's identity. So the reloaded
        symbol is unequal to the one the scheduler saw while printing the same,
        and a dict keyed by the object would miss every lookup after a reload
        in a way that looks like the entry was never written.
        """
        original = _sym()

        reloaded, _tile = decompose_tiled_count(
            sympy.sympify(str(FloorDiv(original, G)))
        )

        self.assertEqual(str(reloaded), str(original))
        self.assertNotEqual(
            reloaded,
            original,
            "the round trip preserved the symbol's assumptions on this sympy "
            "build, so the name-keyed maps could be keyed by the object "
            "instead. Verify that before relying on it",
        )

    def test_unrecognised_shapes_return_none(self):
        s = _sym()
        for expr in (
            s + 1,
            s * 2,
            FloorDiv(s + 1, G),  # not a bare symbol in the numerator
            FloorDiv(s, s),  # not an integer divisor
        ):
            self.assertIsNone(
                decompose_tiled_count(expr),
                f"{expr} should not be treated as a trip count",
            )


class TestLoopLevel(unittest.TestCase):
    """The emitted form, per kind of count."""

    def test_concrete_count_keeps_the_old_form(self):
        # Unchanged behaviour for every existing static kernel: a constant
        # bound, step 1, loop variable counting tiles.
        level = _loop_level(sympy.Integer(8), 0, {})
        self.assertEqual(level.bound, "%loop_bound_0")
        self.assertEqual(level.step, "%c1")
        self.assertEqual(level.stride_scale, 1)
        self.assertEqual(level.setup, ("%loop_bound_0 = arith.constant 8 : index",))

    def test_plain_int_count_is_accepted_too(self):
        level = _loop_level(4, 1, {})
        self.assertEqual(level.setup, ("%loop_bound_1 = arith.constant 4 : index",))

    def test_symbolic_count_uses_the_dim_as_the_bound(self):
        level = _loop_level(FloorDiv(_sym(), G), 0, DIM_SSA)
        # The bound is the dimension itself, never a precomputed count.
        self.assertEqual(level.bound, DIM_SSA[SYM])
        self.assertEqual(level.step, "%step_0")
        self.assertEqual(level.stride_scale, G)
        self.assertEqual(level.setup, (f"%step_0 = arith.constant {G} : index",))

    def test_no_division_is_authored(self):
        """The whole point. We emit no divide, the device derives the count."""
        level = _loop_level(FloorDiv(_sym(), G), 0, DIM_SSA)
        emitted = " ".join(level.setup) + level.bound + level.step
        for forbidden in ("divsi", "divui", "ceildiv", "floordiv"):
            self.assertNotIn(forbidden, emitted.lower())

    def test_tile_size_one_needs_no_step_or_scale(self):
        # A bare symbol already counts single elements.
        level = _loop_level(_sym(), 0, DIM_SSA)
        self.assertEqual(level.bound, DIM_SSA[SYM])
        self.assertEqual(level.step, "%c1")
        self.assertEqual(level.stride_scale, 1)
        self.assertEqual(level.setup, ())

    def test_unrecognised_count_raises_naming_the_symbol(self):
        with self.assertRaises(NotImplementedError) as cm:
            _loop_level(_sym() * 2, 0, DIM_SSA)
        msg = str(cm.exception)
        self.assertIn(SYM, msg, "the message must name the symbol")
        self.assertIn("decompose_tiled_count", msg, "and where to extend")

    def test_symbol_with_no_input_arg_raises_naming_it(self):
        with self.assertRaises(NotImplementedError) as cm:
            _loop_level(FloorDiv(_sym(), G), 0, {})
        msg = str(cm.exception)
        self.assertIn(SYM, msg)
        self.assertIn("count_symbol_bounds", msg, "and where to look")


class TestCountScale(unittest.TestCase):
    """The scale the affine-map pass needs, before any SSA name exists."""

    def test_scales(self):
        self.assertEqual(_count_scale(sympy.Integer(8)), 1)
        self.assertEqual(_count_scale(8), 1)
        self.assertEqual(_count_scale(FloorDiv(_sym(), G)), G)
        self.assertEqual(_count_scale(_sym()), 1)

    def test_unrecognised_count_scales_by_one(self):
        # Deliberately permissive: _loop_level is the place that refuses an
        # unrecognised shape. If this raised instead, the affine pass would
        # fail before the emitter could give the better message.
        self.assertEqual(_count_scale(_sym() * 2), 1)

    def test_agrees_with_loop_level(self):
        for count in (sympy.Integer(8), FloorDiv(_sym(), G), _sym()):
            self.assertEqual(
                _count_scale(count),
                _loop_level(count, 0, DIM_SSA).stride_scale,
                f"the two must not disagree for {count}",
            )


class TestStrideScaling(unittest.TestCase):
    """Strides shrink by the loop step, and a mismatch is caught loudly."""

    def test_unscaled_levels_pass_through(self):
        per_level = [{"a": 2048}, {"b": 64}]
        self.assertEqual(list(_scaled_strides(per_level, [1, 1])), [(0, 2048), (1, 64)])

    def test_a_symbolic_level_divides_its_strides(self):
        # 131072 is the per-tile stride for 64 rows of 1024 fp16 elements.
        # With the loop stepping 64, the emitted stride must be the per-row
        # 2048, since 64 * 2048 recovers the tile stride.
        self.assertEqual(list(_scaled_strides([{"a": 131072}], [G])), [(0, 2048)])

    def test_missing_scale_entries_default_to_one(self):
        self.assertEqual(list(_scaled_strides([{"a": 7}], [])), [(0, 7)])

    def test_empty_levels_are_skipped(self):
        per_level = [{}, {"b": 128}]
        self.assertEqual(list(_scaled_strides(per_level, [G, 1])), [(1, 128)])

    def test_an_indivisible_stride_raises(self):
        """The guard that stops a silently wrong address.

        A stride that is not a multiple of the step means the loop variable
        and the stride disagree about what one trip advances. That has to be
        loud: the result would be addresses off by a fraction of a tile, which
        is a wrong answer and not a crash.
        """
        with self.assertRaises(AssertionError) as cm:
            list(_scaled_strides([{"a": 100}], [G]))
        msg = str(cm.exception)
        self.assertIn("100", msg)
        self.assertIn(str(G), msg)


class TestLoopDimensionSymbolKind(unittest.TestCase):
    """The new symbol kind, and the gate it deliberately does not trip."""

    def test_loop_dimension_is_not_a_dimension(self):
        """This is load-bearing, not a naming detail.

        `execution/async_compile` refuses any bundle carrying an SDSC
        dimension symbol, because that route's runtime payload was never
        built. A loop bound never reaches an SDSC, so it must not answer True
        here or the whole feature is refused at the compile boundary.
        """
        sk = SymbolKind.loop_dimension(
            granularity=G, max_value=MAX, pytorch_sym=SYM, arg_index=0, dim_index=0
        )
        self.assertFalse(sk.is_dimension, "must not trip the async_compile gate")
        self.assertTrue(sk.is_loop_dimension)

    def test_the_old_dimension_kind_still_trips_it(self):
        sk = SymbolKind.dimension(granularity=G, max_value=MAX, pytorch_sym=SYM)
        self.assertTrue(sk.is_dimension, "the old route must stay gated")
        self.assertFalse(sk.is_loop_dimension)

    def test_loop_dimension_carries_where_to_read_the_value(self):
        # The runtime fills this from inputs_outputs[arg_index].size(dim_index).
        sk = SymbolKind.loop_dimension(
            granularity=G, max_value=MAX, pytorch_sym=SYM, arg_index=2, dim_index=1
        )
        self.assertEqual(sk.arg_index, 2)
        self.assertEqual(sk.dim_index, 1)
        self.assertEqual(sk.granularity, G)
        self.assertEqual(sk.max_value, MAX)

    def test_other_kinds_are_neither(self):
        for sk in (
            SymbolKind.kernel(0),
            SymbolKind.kernel_slice(0, 128),
            SymbolKind.pool(),
        ):
            self.assertFalse(sk.is_dimension)
            self.assertFalse(sk.is_loop_dimension)


class TestLoopLevelIsFrozen(unittest.TestCase):
    def test_frozen(self):
        level = LoopLevel(setup=(), bound="%x", step="%c1", stride_scale=1)
        with self.assertRaises(Exception):
            level.bound = "%y"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
