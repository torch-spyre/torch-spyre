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
"""Unit tests for pass_utils.coeff_through_floor.

coeff_through_floor extends sympy.Expr.coeff(sym) to also find sym's
coefficient when sym only appears inside a floor(...) wrapper -- the
shape device_tile_advance_expr takes for stick-layout tensors (see
views.tiling_expr_to_device_expr). Plain sympy.Expr.coeff(sym) returns 0
for a symbol wrapped in floor(), even though it is a genuine free symbol.
"""

import unittest

import sympy
from sympy import Symbol

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import coeff_through_floor


class TestCoeffThroughFloor(unittest.TestCase):
    def test_plain_mul_term_matches_coeff(self):
        """Non-floor-wrapped case: behaves exactly like .coeff(sym)."""
        s = Symbol("s")
        expr = 64 * s
        self.assertEqual(coeff_through_floor(expr, s), 64)
        self.assertEqual(coeff_through_floor(expr, s), expr.coeff(s))

    def test_floor_wrapped_term_extracts_coeff(self):
        """The exact shape from Task 6's investigation:
        floor(65536*sym) -- plain .coeff(sym) returns 0 here, this must
        return 65536."""
        s = Symbol("_tile_adv_op0_lvl0")
        expr = sympy.floor(65536 * s)
        self.assertEqual(expr.coeff(s), 0)  # the bug this helper fixes
        self.assertEqual(coeff_through_floor(expr, s), 65536)

    def test_multi_level_sum_with_one_floor_wrapped_term(self):
        """device_tile_advance_expr is a sum of one term per level; only
        the symbol actually queried should be extracted, regardless of
        whether its own term or a sibling term is floor-wrapped."""
        lvl0 = Symbol("_tile_adv_add_lvl0")
        lvl1 = Symbol("_tile_adv_add_lvl1")
        expr = sympy.floor(65536 * lvl0) + 32 * lvl1
        self.assertEqual(coeff_through_floor(expr, lvl0), 65536)
        self.assertEqual(coeff_through_floor(expr, lvl1), 32)

    def test_symbol_absent_returns_zero(self):
        s = Symbol("s")
        other = Symbol("other")
        expr = 64 * other
        self.assertEqual(coeff_through_floor(expr, s), sympy.S.Zero)

    def test_floor_wrapped_non_integer_coeff_raises_unsupported(self):
        """Tiles are always a whole number of sticks, so floor()'s
        division inside device_tile_advance_expr must always be exact.
        A non-integer extracted coefficient means an earlier pass or
        spyre_hint produced an invalid sub-stick tile boundary -- this
        must fail loudly, not silently truncate via int()."""
        s = Symbol("s")
        expr = sympy.floor(4 * s / 3)  # 4/3 does not reduce to an integer
        with self.assertRaises(Unsupported):
            coeff_through_floor(expr, s)

    def test_floor_wrapped_integer_reducing_division(self):
        """floor(k*sym/d) where d evenly divides k must return the
        reduced integer coefficient, not raise."""
        s = Symbol("s")
        expr = sympy.floor(128 * s / 2)
        self.assertEqual(coeff_through_floor(expr, s), 64)


if __name__ == "__main__":
    unittest.main()
