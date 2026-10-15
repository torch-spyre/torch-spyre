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

"""Recovering a loop's trip count from its cond graph, concrete or symbolic.

The bound is not a field anywhere. `_extract_trip_count` replays the cond
graph's ``inner_fn`` under a recording ops handler and matches
``load(first_placeholder, 0) < bound``. A concrete bound arrives as
``ops.constant`` and a symbolic one as ``ops.index_expr``, so both are
candidates and the prover insists on seeing exactly one.

Each decline carries its own reason. That is not cosmetic: a decline is not an
error, the kernel just specialises, so a symbolic loop that quietly failed to
be recognised is indistinguishable from one that was never asked for unless
the reason says which.

Synthetic cond graphs here, so no compile and no device. The end-to-end claim,
that a real symbolic ``for_each_tile`` is recognised, lives beside the other
prover tests in test_for_each_tile_lowering.py, where the capture helper is.
"""

import unittest
from types import SimpleNamespace

import sympy
import torch
from torch._inductor.virtualized import ops
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.wsr.for_each_tile_lowering import _extract_trip_count

PLACEHOLDER = "iter_carry"
TILE = 64


def _cond_graph(inner_fn, *, n_ops=1, n_outputs=1, dtype=torch.bool, sizes=()):
    """The parts of a cond graph `_extract_trip_count` actually reads."""
    data = SimpleNamespace(inner_fn=inner_fn, get_size=lambda: list(sizes), dtype=dtype)
    return SimpleNamespace(
        graph_outputs=[object()] * n_outputs,
        operations=[SimpleNamespace(data=data)] * n_ops,
        graph_inputs={PLACEHOLDER: object()},
    )


def _compare(bound, kind="constant", placeholder=PLACEHOLDER, index=0, cmp="lt"):
    """An inner_fn issuing ``<placeholder>[index] <cmp> bound``."""

    def inner_fn(_index):
        lhs = ops.load(placeholder, index)
        if kind == "constant":
            rhs = ops.constant(bound, torch.int64)
        elif kind == "index_expr":
            rhs = ops.index_expr(bound, torch.int64)
        else:
            raise AssertionError(f"unknown bound kind {kind!r}")
        return getattr(ops, cmp)(lhs, rhs)

    return inner_fn


def _two_bounds():
    def inner_fn(_index):
        lhs = ops.load(PLACEHOLDER, 0)
        ops.constant(8, torch.int64)
        rhs = ops.index_expr(sympy.Symbol("s0", integer=True), torch.int64)
        return ops.lt(lhs, rhs)

    return inner_fn


class TestABoundIsAcceptedFromEitherChannel(unittest.TestCase):
    def test_a_concrete_bound_arrives_as_a_constant(self):
        result = _extract_trip_count(_cond_graph(_compare(8)))

        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.trip_count, sympy.Integer(8))

    def test_a_symbolic_bound_arrives_as_an_index_expr(self):
        count = FloorDiv(sympy.Symbol("s97", integer=True, positive=True), TILE)

        result = _extract_trip_count(_cond_graph(_compare(count, kind="index_expr")))

        self.assertTrue(
            result.accepted,
            "a symbolic bound was declined, so every symbolic loop would "
            f"specialise: {result.reason}",
        )
        self.assertEqual(result.trip_count, count)
        self.assertTrue(result.trip_count.free_symbols)

    def test_a_bare_symbol_bound_is_accepted_too(self):
        """A gather-driven loop steps one row per trip, so its count is bare."""
        count = sympy.Symbol("s97", integer=True, positive=True)

        result = _extract_trip_count(_cond_graph(_compare(count, kind="index_expr")))

        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.trip_count, count)


class TestEveryDeclineSaysWhy(unittest.TestCase):
    """The reason is the feature. A bare "not accepted" is not diagnosable."""

    def assert_declined(self, result, *expected):
        self.assertFalse(result.accepted, f"expected a decline, got {result}")
        for fragment in expected:
            self.assertIn(fragment, result.reason)

    def test_two_candidate_bounds(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_two_bounds())), "2 candidate bounds"
        )

    def test_no_bound_at_all(self):
        def inner_fn(_index):
            return ops.lt(ops.load(PLACEHOLDER, 0), ops.load(PLACEHOLDER, 1))

        self.assert_declined(_extract_trip_count(_cond_graph(inner_fn)), "2 loads")

    def test_a_bool_bound_is_not_a_trip_count(self):
        self.assert_declined(_extract_trip_count(_cond_graph(_compare(True))), "bool")

    def test_the_wrong_comparison(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8, cmp="gt"))), "comparisons", "gt"
        )

    def test_the_wrong_placeholder(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8, placeholder="something_else"))),
            "something_else",
        )

    def test_the_wrong_load_index(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8, index=3))), "index 3"
        )

    def test_more_than_one_operation(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8), n_ops=2)), "2 operations"
        )

    def test_more_than_one_output(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8), n_outputs=2)), "2 outputs"
        )

    def test_not_a_scalar(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8), sizes=(4,))), "not a scalar"
        )

    def test_not_a_bool(self):
        self.assert_declined(
            _extract_trip_count(_cond_graph(_compare(8), dtype=torch.int64)), "not bool"
        )


if __name__ == "__main__":
    unittest.main()
