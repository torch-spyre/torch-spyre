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

"""How `for_each_tile` spells its views, and why the spelling is a correctness choice.

Three helpers reduce an operand to its per-step tile and fold the result back.
Each one has more than one spelling that is correct for a concrete length, and
exactly one that also survives a symbolic one. These tests pin that.

No device and no compile. The symbolic cases build a fake tensor over a real
`ShapeEnv`, so they run anywhere torch does.

Every symbolic test carries a control that exercises the spelling we did NOT
take, and asserts it behaves worse. Without the control a passing test says
nothing, because the assertion would hold for both spellings.
"""

import unittest

import sympy
import torch

from symbolic_shape_fixtures import (  # noqa: E402
    TILE,
    WIDTH,
    fake_symbolic_rows,
    is_symbolic,
    shape_env_attr,
)
from torch_spyre._inductor.wsr.for_each_tile import (
    Kind,
    TileSpec,
    _stacked_to_full,
    _tile_size_vector,
    _xs_leaf,
)


def _slice_spec(shape, dim, extent, num_tiles):
    return TileSpec(
        Kind.SLICE,
        dim,
        _tile_size_vector(shape, dim, extent),
        num_tiles,
    )


class TestConcreteBehaviourUnchanged(unittest.TestCase):
    """The spellings are views, and they still produce what they always did.

    This is the regression half of the change. A concrete length is the only
    thing every caller on main has today, including the SDPA decomposition, so
    these are the assertions that say the change is safe to land on its own.
    """

    def test_xs_leaf_is_a_view_and_splits_as_expected(self):
        x = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        spec = _slice_spec(x.shape, 0, 2, 4)

        leaf = _xs_leaf(x, spec)

        self.assertEqual(tuple(leaf.shape), (4, 2, 6))
        self.assertIs(leaf.untyped_storage(), x.untyped_storage())
        torch.testing.assert_close(leaf.reshape(8, 6), x)

    def test_the_trim_drops_nothing_for_a_concrete_length(self):
        x = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        spec = _slice_spec(x.shape, 0, 2, 4)

        self.assertEqual(_xs_leaf(x, spec).numel(), x.numel())

    def test_xs_leaf_on_a_moved_axis(self):
        x = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
        spec = _slice_spec(x.shape, 1, 4, 2)

        leaf = _xs_leaf(x, spec)

        # Tiled axis moved to the front, then split: [2 tiles, 4 wide, 3 rows].
        self.assertEqual(tuple(leaf.shape), (2, 4, 3))
        self.assertIs(leaf.untyped_storage(), x.untyped_storage())

    def test_stacked_to_full_dim0_matches_flatten_and_shares_storage(self):
        ys = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)

        folded = _stacked_to_full(ys, 0)

        torch.testing.assert_close(folded, ys.flatten(0, 1))
        self.assertEqual(tuple(folded.shape), (8, 6))
        self.assertIs(folded.untyped_storage(), ys.untyped_storage())

    def test_stacked_to_full_other_dim_still_takes_the_flatten_path(self):
        ys = torch.arange(3 * 8 * 2, dtype=torch.float32).reshape(3, 8, 2)

        folded = _stacked_to_full(ys, 1)

        torch.testing.assert_close(folded, ys.movedim(0, 1).flatten(1, 2))
        self.assertEqual(tuple(folded.shape), (8, 6))

    def test_a_non_contiguous_leading_axis_falls_back_and_is_still_correct(self):
        # as_strided would take the strides on trust here and be wrong, so the
        # contiguity test has to send this down the flatten path.
        ys = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)[:, ::2]

        folded = _stacked_to_full(ys, 0)

        torch.testing.assert_close(folded, ys.flatten(0, 1))


class TestSymbolicTileExtent(unittest.TestCase):
    """Trimming first is what keeps the tile extent a literal."""

    def test_the_extent_stays_literal(self):
        shape_env, mode, x = fake_symbolic_rows()
        rows = x.shape[0]
        self.assertTrue(is_symbolic(rows), "fixture did not produce a symbolic dim")
        spec = _slice_spec(x.shape, 0, TILE, rows // TILE)

        with mode:
            leaf = _xs_leaf(x, spec)

        extent = leaf.shape[1]
        self.assertFalse(
            is_symbolic(extent),
            f"tile extent came out symbolic as {extent}, so the trim did not hold",
        )
        self.assertEqual(int(extent), TILE)

    def test_control_the_untrimmed_spelling_derives_the_extent(self):
        """The control. Without it the test above would pass for either spelling.

        Splitting the untrimmed operand is what the helper used to do, and it
        re-derives the extent as `s // (s // G)` rather than keeping `G`.
        """
        shape_env, mode, x = fake_symbolic_rows()
        rows = x.shape[0]

        with mode:
            untrimmed = torch.unflatten(x, 0, (rows // TILE, TILE))

        extent = untrimmed.shape[1]
        self.assertTrue(
            is_symbolic(extent),
            "the untrimmed spelling kept the extent literal, so this control no "
            "longer proves anything and the test above is not measuring the trim",
        )

    def test_the_leaf_still_covers_every_whole_tile(self):
        shape_env, mode, x = fake_symbolic_rows()
        rows = x.shape[0]
        spec = _slice_spec(x.shape, 0, TILE, rows // TILE)

        with mode:
            leaf = _xs_leaf(x, spec)

        # num_tiles * extent, so no whole tile is lost to the trim.
        self.assertEqual(
            sympy.sympify(str(leaf.shape[0] * leaf.shape[1])),
            sympy.sympify(str((rows // TILE) * TILE)),
        )


class TestOutputFoldInstallsNoGuards(unittest.TestCase):
    """`as_strided` folds the output without installing a guard of its own.

    Scoped to this call. A single-tile call still compiles separately, because
    torch's own tensor-metadata extraction evaluates the same predicate later.
    This pins that the fold is not one of the sources.
    """

    @staticmethod
    def _stacked_ys(mode, shape_env, tiles):
        with mode:
            return torch.empty(tiles, TILE, WIDTH)

    def test_as_strided_path_adds_no_guards(self):
        shape_env, mode, x = fake_symbolic_rows()
        tiles = x.shape[0] // TILE
        ys = self._stacked_ys(mode, shape_env, tiles)
        before = len(shape_env_attr(shape_env, "guards"))

        with mode:
            _stacked_to_full(ys, 0)

        added = [str(g) for g in shape_env_attr(shape_env, "guards")[before:]]
        self.assertEqual(
            added,
            [],
            "folding the output installed a guard, so this fold is a source "
            f"of specialisation: {added}",
        )

    def test_control_flatten_adds_at_least_one(self):
        """The control for the test above. `flatten` is the spelling we rejected."""
        shape_env, mode, x = fake_symbolic_rows()
        tiles = x.shape[0] // TILE
        ys = self._stacked_ys(mode, shape_env, tiles)
        before = len(shape_env_attr(shape_env, "guards"))

        with mode:
            ys.flatten(0, 1)

        self.assertGreater(
            len(shape_env_attr(shape_env, "guards")) - before,
            0,
            "flatten installed no guard either, so the as_strided test above is "
            "not measuring anything",
        )


class TestDivisibilityIsStated(unittest.TestCase):
    """The ragged refusal is also recorded as a fact for the simplifier."""

    def test_a_ragged_concrete_length_is_still_refused(self):
        from torch_spyre._inductor.wsr.for_each_tile import _normalize_in_specs

        x = torch.empty(10, WIDTH)
        with self.assertRaisesRegex(ValueError, "not a multiple of tile_size"):
            _normalize_in_specs((x,), (0,), 4)

    def test_the_fact_reaches_the_shape_env(self):
        from torch_spyre._inductor.wsr.for_each_tile import _normalize_in_specs

        shape_env, mode, x = fake_symbolic_rows()
        with mode:
            _normalize_in_specs((x,), (0,), TILE)

        divisible = {str(expr) for expr in shape_env_attr(shape_env, "divisible")}
        self.assertTrue(
            any(str(TILE) in d for d in divisible),
            f"Mod(s, {TILE}) is not in the divisibility set, so the simplifier "
            f"cannot fold {TILE} * (s // {TILE}) back to s. Set was {divisible}",
        )


if __name__ == "__main__":
    unittest.main()
