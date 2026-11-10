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

"""IR-level unit tests for insert_bmm_padding.

Tests hook into CustomPreSchedulingPasses after insert_bmm_padding runs to inspect
the operations list directly, without requiring end-to-end compilation to succeed.
"""

import math
import unittest
from unittest.mock import patch
from typing import Any, Callable, Optional, TypeVarTuple, Unpack, override

import sympy
from sympy import FiniteSet, Interval, S, oo
import torch
import torch.fx as fx
from torch._inductor import config as t_inductor_config
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    Operation,
    Reduction,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V, ops
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.ir import FixedTiledLayout, SpyreConstantFallback
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor import padding
from torch_spyre._inductor.padding import (
    ArgNameToIndexTracer,
    DomainRangeSolver,
    InvalidDomain,
    SetDtype,
    SetDtypeTracer,
    _compute_digits,
    _is_dense_flattened_coordinate,
    _is_nonoverlapping_aligned_stick_coordinate,
    _restickify_input_required_extent,
)
from torch_spyre._inductor.scratchpad import allocator
from torch_spyre._inductor.scratchpad import utils as scratchpad_utils
from utils_inductor import try_reinterpret_host_shape

Ts = TypeVarTuple("Ts")


class TestRestickifyInputCoordinate(unittest.TestCase):
    def test_dense_flattened_coordinate(self) -> None:
        head, feature = sympy.symbols("head feature", integer=True)
        ranges = {head: 16, feature: 128}

        digits = _compute_digits(128 * head + feature, ranges)
        self.assertTrue(_is_dense_flattened_coordinate(digits, ranges))
        digits = _compute_digits(128 * head + feature + 7, ranges)
        self.assertTrue(_is_dense_flattened_coordinate(digits, ranges))

    def test_gapped_coordinate_is_not_dense(self) -> None:
        head, feature = sympy.symbols("head feature", integer=True)
        ranges = {head: 16, feature: 128}

        digits = _compute_digits(256 * head + feature, ranges)
        self.assertFalse(_is_dense_flattened_coordinate(digits, ranges))
        digits = _compute_digits(head + 32 * feature, ranges)
        self.assertFalse(_is_dense_flattened_coordinate(digits, ranges))

    def test_dense_flattened_required_extent(self) -> None:
        batch, sequence = sympy.symbols("batch sequence", integer=True)

        self.assertEqual(
            _restickify_input_required_extent(
                64 * batch + sequence,
                {batch: 2, sequence: 64},
                sequence,
                torch.float16,
            ),
            128,
        )
        self.assertEqual(
            _restickify_input_required_extent(
                64 * batch + sequence + 7,
                {batch: 2, sequence: 64},
                sequence,
                torch.float16,
            ),
            135,
        )

        with self.assertRaisesRegex(RuntimeError, "requires a re-base copy"):
            _restickify_input_required_extent(
                63 * batch + sequence,
                {batch: 2, sequence: 63},
                sequence,
                torch.float16,
            )

    def test_gapped_coarse_tile_has_nonoverlapping_stick_windows(self) -> None:
        batch, sequence = sympy.symbols("batch sequence", integer=True)
        coord = 512 * batch + sequence
        ranges = {batch: 4, sequence: 256}

        digits = _compute_digits(coord, ranges)
        self.assertFalse(_is_dense_flattened_coordinate(digits, ranges))
        self.assertTrue(
            _is_nonoverlapping_aligned_stick_coordinate(
                digits, coord, ranges, sequence, torch.float16
            )
        )
        self.assertEqual(
            _restickify_input_required_extent(coord, ranges, sequence, torch.float16),
            1792,
        )

        coord = 510 * batch + sequence
        digits = _compute_digits(coord, ranges)
        self.assertFalse(
            _is_nonoverlapping_aligned_stick_coordinate(
                digits,
                coord,
                {batch: 4, sequence: 255},
                sequence,
                torch.float16,
            )
        )
        coord = 128 * batch + sequence
        digits = _compute_digits(coord, ranges)
        self.assertFalse(
            _is_nonoverlapping_aligned_stick_coordinate(
                digits,
                coord,
                ranges,
                sequence,
                torch.float16,
            )
        )

    def test_single_symbol_required_extent_preserves_split_coordinate(self) -> None:
        sequence = sympy.symbols("sequence", integer=True)

        self.assertEqual(
            _restickify_input_required_extent(
                sympy.floor(sequence / 64),
                {sequence: 128},
                sequence,
                torch.float16,
            ),
            128,
        )


# ---------------------------------------------------------------------------
# Padding-value propagation: DomainRangeSolver, SetDtypeTracer,
# ArgNameToIndexTracer. These are pure set computations and need no device.
# ---------------------------------------------------------------------------

_I = sympy.Symbol("i")
_NON_NEG = Interval(0, oo)
_POS = Interval(0, oo, left_open=True)


def _rationalize(s):
    return s.xreplace({f: sympy.nsimplify(f) for f in s.atoms(sympy.Float)})


class TestDomainRangeSolver(unittest.TestCase):
    """The solver maps the set of possible padding values through each op.

    The image it returns must contain every value the op can produce from the
    input set (soundness); where it can, it is the exact image.
    """

    def setUp(self) -> None:
        self.s = DomainRangeSolver()

    def assertSetEqual(self, actual, expected) -> None:
        # Compare by inclusion: sympy compares sets structurally, so {0.0} and
        # {0}, or Range(1, 4) and {1, 2, 3}, are not ==. Floats become
        # rationals so that inclusion of float and integer bounds is decidable.
        actual, expected = _rationalize(actual), _rationalize(expected)
        self.assertTrue(
            actual.is_subset(expected) and expected.is_subset(actual),
            f"{actual} != {expected}",
        )

    def assertSubset(self, sub, sup) -> None:
        self.assertTrue(sub.is_subset(sup), f"{sub} is not a subset of {sup}")

    def assertRaisesDomain(
        self, fn, *args, func: str, domain=None, arg_index: int = 0
    ) -> InvalidDomain:
        with self.assertRaises(InvalidDomain) as ctx:
            fn(*args)
        self.assertEqual(ctx.exception.func, func)
        self.assertEqual(ctx.exception.arg_index, arg_index)
        if domain is not None:
            self.assertSetEqual(ctx.exception.domain, domain)
        return ctx.exception

    # -- constants and finite sets --------------------------------------------

    def test_constant_is_singleton(self) -> None:
        self.assertSetEqual(self.s.constant(2.5, torch.float16), FiniteSet(2.5))

    def test_unary_on_finite_set_maps_each_element(self) -> None:
        self.assertSetEqual(self.s.exp(FiniteSet(0)), FiniteSet(1.0))
        self.assertSetEqual(self.s.relu(FiniteSet(-2, 3)), FiniteSet(0, 3))
        self.assertSetEqual(self.s.neg(FiniteSet(1, -4)), FiniteSet(-1, 4))
        self.assertSetEqual(self.s.sigmoid(FiniteSet(0)), FiniteSet(0.5))
        self.assertSetEqual(self.s.reciprocal(FiniteSet(2)), FiniteSet(0.5))

    def test_binary_on_finite_sets_is_cartesian(self) -> None:
        self.assertSetEqual(
            self.s.add(FiniteSet(0, 1), FiniteSet(10)), FiniteSet(10, 11)
        )
        self.assertSetEqual(
            self.s.mul(FiniteSet(0, 2), FiniteSet(3, 4)), FiniteSet(0, 6, 8)
        )
        self.assertSetEqual(self.s.truediv(FiniteSet(0), FiniteSet(2)), FiniteSet(0))

    # -- intervals ------------------------------------------------------------

    def test_unary_on_interval_maps_endpoints(self) -> None:
        self.assertSetEqual(
            self.s.neg(Interval(1, 2, left_open=True)), Interval.Ropen(-2, -1)
        )
        self.assertSetEqual(self.s.reciprocal(Interval(1, 2)), Interval(0.5, 1))
        self.assertSetEqual(self.s.exp(S.Reals), Interval.open(0, oo))
        self.assertSetEqual(self.s.sigmoid(S.Reals), Interval.open(0, 1))
        self.assertSetEqual(self.s.tanh(S.Reals), Interval.open(-1, 1))
        self.assertSetEqual(self.s.sqrt(_NON_NEG), _NON_NEG)

    def test_binary_on_intervals_uses_corners(self) -> None:
        self.assertSetEqual(self.s.add(Interval(0, 1), FiniteSet(2)), Interval(2, 3))
        self.assertSetEqual(
            self.s.mul(Interval(-1, 2), Interval(-3, 1)), Interval(-6, 3)
        )
        self.assertSetEqual(self.s.sub(FiniteSet(0), Interval(0, 1)), Interval(-1, 0))
        self.assertSetEqual(
            self.s.maximum(FiniteSet(0), Interval(-1, 2)), Interval(0, 2)
        )
        self.assertSetEqual(
            self.s.minimum(FiniteSet(0), Interval(-1, 2)), Interval(-1, 0)
        )
        self.assertSetEqual(self.s.add(S.Reals, FiniteSet(0)), S.Reals)

    def test_binary_open_bound_only_when_every_corner_is_open(self) -> None:
        # [0, 1) + {1} = [1, 2): the upper corner comes only from the open end.
        self.assertSetEqual(
            self.s.add(Interval.Ropen(0, 1), FiniteSet(1)), Interval.Ropen(1, 2)
        )
        # [-1, 2] * [-3, 1): -6 is reached by the closed corner (2, -3).
        self.assertSetEqual(
            self.s.mul(Interval(-1, 2), Interval.Ropen(-3, 1)), Interval(-6, 3)
        )

    def test_mul_zero_by_unbounded(self) -> None:
        # [0, oo) excludes oo, so the 0 * oo corner is a limit: 0 * x is 0.
        self.assertSetEqual(self.s.mul(FiniteSet(0), _NON_NEG), FiniteSet(0))
        # A set that holds oo itself (an fp overflow) is not an interval; the
        # solver must fall back, not lose values.
        extended = _NON_NEG.union(FiniteSet(oo))
        self.assertSubset(FiniteSet(0, 1, 1000), self.s.mul(FiniteSet(0), extended))

    def test_max_min_bound_attained_where_one_argument_dominates(self) -> None:
        # max(x, 0) = 0 for every x <= 0, so 0 is attained though the corner
        # (-oo, 0) is open.
        self.assertSetEqual(self.s.maximum(S.Reals, Interval(0, 1)), _NON_NEG)
        self.assertSetEqual(self.s.maximum(S.Reals, FiniteSet(0)), _NON_NEG)
        self.assertSetEqual(self.s.minimum(S.Reals, Interval(0, 1)), Interval(-oo, 1))
        # relu is max(x, 0): 0 is attained for every x <= 0.
        self.assertSetEqual(self.s.relu(S.Reals), _NON_NEG)
        self.assertSetEqual(self.s.relu(Interval.open(-1, 1)), Interval.Ropen(0, 1))
        self.assertSetEqual(self.s.relu(Interval.open(-oo, 0)), FiniteSet(0))

    def test_max_min_exact_image(self) -> None:
        # Values of one argument below every value of the other are not in
        # the image of max.
        self.assertSetEqual(
            self.s.maximum(FiniteSet(0), FiniteSet(-1, 2)), FiniteSet(0, 2)
        )
        self.assertSetEqual(
            self.s.maximum(Interval.Lopen(0, 1), Interval.Ropen(0, 1)),
            Interval.Lopen(0, 1),
        )
        self.assertSetEqual(
            self.s.maximum(Interval.open(0, 1), Interval.open(0, 1)),
            Interval.open(0, 1),
        )
        self.assertSetEqual(self.s.maximum(S.EmptySet, S.Reals), S.EmptySet)

    def test_max_min_keep_infinities(self) -> None:
        # max(oo, 1) = oo: an extended set keeps its infinite points.
        extended = _NON_NEG.union(FiniteSet(oo))
        self.assertSetEqual(
            self.s.maximum(extended, FiniteSet(1)),
            Interval(1, oo).union(FiniteSet(oo)),
        )
        self.assertSetEqual(self.s.minimum(extended, FiniteSet(1)), Interval(0, 1))
        neg_extended = Interval(-oo, 0).union(FiniteSet(-oo))
        self.assertSetEqual(self.s.minimum(neg_extended, FiniteSet(1)), neg_extended)

    # A corner where the op is undefined (oo - oo, 0 * oo, 0 / 0, 1 / 0) is
    # dropped, or taken as a limit when only one of its bounds is open.
    def test_sub_of_unbounded_intervals_is_sound(self) -> None:
        self.assertSetEqual(self.s.sub(_NON_NEG, _NON_NEG), S.Reals)
        self.assertSetEqual(self.s.add(S.Reals, S.Reals), S.Reals)

    def test_mul_of_unbounded_intervals(self) -> None:
        self.assertSetEqual(self.s.mul(_NON_NEG, _NON_NEG), _NON_NEG)
        self.assertSetEqual(self.s.mul(Interval.Lopen(0, 1), _NON_NEG), _NON_NEG)
        self.assertSetEqual(self.s.mul(S.Reals, S.Reals), S.Reals)

    def test_truediv_by_interval_open_at_zero(self) -> None:
        # 0 / 0 and 1 / 0 at the excluded 0: the quotient grows without bound.
        divisor = Interval.Lopen(0, 1)
        self.assertSetEqual(self.s.truediv(Interval(0, 1), divisor), _NON_NEG)
        self.assertSetEqual(self.s.truediv(FiniteSet(1), divisor), Interval(1, oo))
        self.assertSetEqual(self.s.truediv(FiniteSet(-1), divisor), Interval(-oo, -1))
        self.assertSetEqual(self.s.truediv(Interval(1, 2), Interval.open(0, oo)), _POS)

    # abs, square and silu are not monotonic: an interval containing their
    # turning point has an extreme there, not at an endpoint.
    def test_abs_of_interval_spanning_zero(self) -> None:
        self.assertSetEqual(self.s.abs(S.Reals), _NON_NEG)
        self.assertSetEqual(self.s.abs(Interval(-2, 1)), Interval(0, 2))
        self.assertSetEqual(self.s.abs(Interval.open(-1, 3)), Interval.Ropen(0, 3))
        # The two endpoints reach the same bound: closed if either one is.
        self.assertSetEqual(self.s.abs(Interval.Lopen(-2, 2)), Interval(0, 2))
        self.assertSetEqual(self.s.abs(Interval.open(-2, 2)), Interval.Ropen(0, 2))

    def test_square_of_interval_spanning_zero(self) -> None:
        self.assertSetEqual(self.s.square(Interval(-2, 1)), Interval(0, 4))
        self.assertSetEqual(self.s.square(S.Reals), _NON_NEG)

    def test_non_monotonic_on_one_side_of_turning_point(self) -> None:
        # Not containing the turning point, the endpoints are the extremes.
        self.assertSetEqual(self.s.abs(Interval(-3, -1)), Interval(1, 3))
        self.assertSetEqual(self.s.square(Interval.Lopen(0, 2)), Interval.Lopen(0, 4))
        # The turning point as an excluded endpoint is not reached.
        self.assertSetEqual(self.s.square(Interval.open(-1, 0)), Interval.open(0, 1))

    def test_silu_of_interval_spanning_its_minimum(self) -> None:
        # silu's minimum, ~-0.2785, is at x ~-1.2785.
        r = self.s.silu(Interval(-2, 0))
        self.assertAlmostEqual(float(r.inf), -0.2785, 4)
        self.assertFalse(r.left_open)
        self.assertEqual(float(r.sup), 0.0)
        # Fully left of the minimum silu is decreasing.
        r = self.s.silu(Interval(-3, -2))
        self.assertAlmostEqual(float(r.inf), -2 / (1 + math.exp(2)))
        self.assertAlmostEqual(float(r.sup), -3 / (1 + math.exp(3)))

    # An open (or infinite) endpoint is not attained: the bound is the op's
    # limit there, even where the op is undefined (log(sigmoid(x)) for any x
    # reaches log on (0, 1)).
    def test_log_of_open_positive_interval(self) -> None:
        self.assertSetEqual(self.s.log(_POS), S.Reals)
        self.assertSetEqual(self.s.log(Interval.open(0, 1)), Interval.open(-oo, 0))

    def test_rsqrt_of_open_positive_interval(self) -> None:
        self.assertSetEqual(self.s.rsqrt(_POS), _POS)
        self.assertSetEqual(self.s.rsqrt(Interval.Lopen(0, 4)), Interval.Ropen(0.5, oo))

    def test_reciprocal_of_open_interval(self) -> None:
        self.assertSetEqual(self.s.reciprocal(_POS), _POS)
        self.assertSetEqual(
            self.s.reciprocal(Interval.open(-oo, 0)), Interval.open(-oo, 0)
        )

    def test_silu_of_reals(self) -> None:
        r = self.s.silu(S.Reals)
        self.assertAlmostEqual(float(r.inf), -0.2785, 4)
        self.assertFalse(r.left_open)
        self.assertEqual(r.sup, oo)

    def test_open_endpoint_bound_is_open(self) -> None:
        # The limit at an open endpoint is approached, not reached.
        r = self.s.exp(Interval.Lopen(0, 1))
        self.assertEqual(float(r.inf), 1.0)
        self.assertTrue(r.left_open)
        self.assertFalse(r.right_open)

    def test_clamp_floor_where(self) -> None:
        self.assertSetEqual(self.s.clamp(S.Reals, 0, 1), Interval(0, 1))
        self.assertSetEqual(self.s.clamp(S.Reals, min=0), _NON_NEG)
        # Values outside the range move to the nearest bound.
        self.assertSetEqual(self.s.clamp(FiniteSet(5), 0, 1), FiniteSet(1))
        self.assertSetEqual(
            self.s.clamp(FiniteSet(-3, 0.5, 9), 0, 1), FiniteSet(0, 0.5, 1)
        )
        self.assertSetEqual(self.s.clamp(Interval(2, 3), 0, 1), FiniteSet(1))
        self.assertSetEqual(self.s.floor(Interval(0.5, 3)), FiniteSet(1, 2, 3))
        self.assertSetEqual(
            self.s.where(FiniteSet(1), FiniteSet(0), FiniteSet(2)), FiniteSet(0, 2)
        )

    def test_comparisons(self) -> None:
        self.assertSetEqual(self.s.eq(FiniteSet(0), FiniteSet(0)), FiniteSet(True))
        self.assertSetEqual(self.s.lt(FiniteSet(1), FiniteSet(0)), FiniteSet(False))
        # Not both singletons: any boolean.
        self.assertSetEqual(self.s.lt(FiniteSet(0), Interval(0, 1)), FiniteSet(0, 1))
        self.assertSetEqual(self.s.ne(FiniteSet(0, 1), FiniteSet(0)), FiniteSet(0, 1))

    def test_range_fallbacks_contain_the_range(self) -> None:
        # Ops without an exact image return a superset of their range.
        self.assertSubset(Interval(-0.17, 5), self.s.gelu(S.Reals))
        self.assertSubset(_NON_NEG, self.s.layernormscale(S.Reals, 1e-5))
        self.assertSetEqual(self.s.to_dtype(FiniteSet(3), torch.float32), FiniteSet(3))

    def test_unknown_op_returns_user_values(self) -> None:
        with patch("torch_spyre._inductor.padding.padding_strict_mode", False):
            self.assertSetEqual(self.s._default("foo", (FiniteSet(0),), {}), S.Reals)
        with patch("torch_spyre._inductor.padding.padding_strict_mode", True):
            self.assertTrue(self.s._default("foo", (FiniteSet(0),), {}).has(_NAN))

    # -- domain checks ----------------------------------------------------------

    def test_domain_violations_raise(self) -> None:
        zero = FiniteSet(0)
        self.assertRaisesDomain(self.s.log, zero, func="log")
        self.assertRaisesDomain(self.s.log, FiniteSet(-1), func="log")
        self.assertRaisesDomain(self.s.rsqrt, zero, func="rsqrt")
        self.assertRaisesDomain(self.s.reciprocal, zero, func="reciprocal")
        self.assertRaisesDomain(
            self.s.sqrt, FiniteSet(-1), func="rsqrt", domain=_NON_NEG
        )
        # truediv blames the divisor and uses the realdiv mask key.
        self.assertRaisesDomain(
            self.s.truediv, FiniteSet(1), zero, func="realdiv", arg_index=1
        )

    def test_domain_satisfied_does_not_raise(self) -> None:
        self.assertSetEqual(self.s.log(FiniteSet(1)), FiniteSet(0))
        self.assertSetEqual(self.s.rsqrt(FiniteSet(4)), FiniteSet(0.5))
        self.assertSetEqual(self.s.sqrt(FiniteSet(0)), FiniteSet(0))
        # The mask values for these ops (inf, see _PADDING_MASK_VALUES) are
        # in the extended domain.
        self.assertSetEqual(self.s.reciprocal(FiniteSet(oo)), FiniteSet(0))
        self.assertSetEqual(self.s.rsqrt(FiniteSet(oo)), FiniteSet(0))
        self.assertSetEqual(self.s.truediv(FiniteSet(1), FiniteSet(oo)), FiniteSet(0))

    def test_indirect_indexing_requires_zero_padding(self) -> None:
        self.assertEqual(
            self.s.indirect_indexing(FiniteSet(0), 4), padding._INDIRECT_INDEX_SYM
        )
        self.assertRaisesDomain(
            self.s.indirect_indexing,
            FiniteSet(0, 1),
            4,
            func="indirect_indexing",
            domain=FiniteSet(0),
        )

    # -- reductions -------------------------------------------------------------

    def test_sum_reduction(self) -> None:
        self.assertSetEqual(
            self.s.reduction(torch.float16, torch.float16, "sum", FiniteSet(0)),
            S.Reals,
        )
        self.assertRaisesDomain(
            self.s.reduction,
            torch.float16,
            torch.float16,
            "sum",
            FiniteSet(0, 1),
            func="sum",
            domain=FiniteSet(0),
        )

    @parametrize(
        "reduction_type,dtype,pad",
        [
            ("max", torch.float16, padding._FP16_MIN),
            ("min", torch.float16, padding._FP16_MAX),
            ("max", torch.float32, padding._FP32_MIN),
            ("min", torch.float32, padding._FP32_MAX),
        ],
    )
    def test_max_min_reduction_requires_identity(
        self, reduction_type, dtype, pad
    ) -> None:
        self.assertRaisesDomain(
            self.s.reduction,
            dtype,
            dtype,
            reduction_type,
            FiniteSet(0),
            func=reduction_type,
            domain=FiniteSet(pad),
        )
        self.assertSetEqual(
            self.s.reduction(dtype, dtype, reduction_type, FiniteSet(pad)), S.Reals
        )

    def test_batchmatmul_needs_one_zero_padded_operand(self) -> None:
        for value in [
            (FiniteSet(0), S.Reals),
            (S.Reals, FiniteSet(0)),
            (FiniteSet(0), FiniteSet(0)),
        ]:
            self.assertSetEqual(
                self.s.reduction(torch.float16, torch.float16, BATCH_MATMUL_OP, value),
                FiniteSet(0),
            )
        self.assertRaisesDomain(
            self.s.reduction,
            torch.float16,
            torch.float16,
            BATCH_MATMUL_OP,
            (S.Reals, S.Reals),
            func=BATCH_MATMUL_OP,
            domain=FiniteSet(0),
        )


instantiate_parametrized_tests(TestDomainRangeSolver)


_NAN = padding._NAN
_NAN_SET = FiniteSet(_NAN)


class TestDomainRangeSolverNaN(unittest.TestCase):
    """In strict mode the padding of unknown memory may hold NaN.

    NaN is a set element (padding._NAN) that pointwise ops propagate, that
    comparisons turn into either boolean, and that no domain admits.
    """

    def setUp(self) -> None:
        self.s = DomainRangeSolver()

    def assertSetEqual(self, actual, expected) -> None:  # noqa: N802
        TestDomainRangeSolver.assertSetEqual(self, actual, expected)

    def test_strict_mode_user_values_hold_nan(self) -> None:
        with patch("torch_spyre._inductor.padding.padding_strict_mode", True):
            values = padding._user_values()
        self.assertTrue(values.has(_NAN))
        self.assertTrue(values.is_superset(S.Reals))

    def test_nan_mixes_with_numbers(self) -> None:
        # S.NaN cannot share a set with numbers; the marker can.
        self.assertEqual(len(FiniteSet(0, _NAN)), 2)
        self.assertEqual(
            padding._without_nan(Interval(0, oo).union(FiniteSet(oo, _NAN))),
            (Interval(0, oo).union(FiniteSet(oo)), True),
        )
        self.assertEqual(padding._without_nan(S.Reals), (S.Reals, False))

    @parametrize(
        "op",
        [
            "abs",
            "exp",
            "floor",
            "neg",
            "relu",
            "sigmoid",
            "silu",
            "square",
            "tanh",
            "gelu",
        ],
    )
    def test_unary_propagates_nan(self, op) -> None:
        for x in (_NAN_SET, S.Reals.union(_NAN_SET), FiniteSet(0, _NAN)):
            result = getattr(self.s, op)(x)
            self.assertTrue(result.has(_NAN), f"{op}({x}) = {result}")
            real, _ = padding._without_nan(result)
            self.assertSetEqual(real, getattr(self.s, op)(padding._without_nan(x)[0]))

    @parametrize("op", ["add", "sub", "mul", "truediv", "maximum", "minimum"])
    def test_binary_propagates_nan(self, op) -> None:
        # NaN from either argument; the real values still combine.
        fn = getattr(self.s, op)
        self.assertSetEqual(fn(_NAN_SET, FiniteSet(1)), _NAN_SET)
        self.assertSetEqual(fn(FiniteSet(1), _NAN_SET), _NAN_SET)
        result = fn(FiniteSet(2, _NAN), FiniteSet(1))
        self.assertSetEqual(result, fn(FiniteSet(2), FiniteSet(1)).union(_NAN_SET))

    def test_other_pointwise_propagate_nan(self) -> None:
        self.assertSetEqual(self.s.clamp(FiniteSet(5, _NAN), 0, 1), FiniteSet(1, _NAN))
        self.assertSetEqual(
            self.s.where(FiniteSet(1), FiniteSet(0, _NAN), FiniteSet(2)),
            FiniteSet(0, 2, _NAN),
        )
        self.assertSetEqual(self.s.softplus(_NAN_SET, 1.0, 20.0), _NAN_SET)

    @parametrize(
        "op", ["eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or"]
    )
    def test_comparison_with_nan_gives_any_boolean(self, op) -> None:
        fn = getattr(self.s, op)
        self.assertSetEqual(fn(_NAN_SET, FiniteSet(1)), FiniteSet(0, 1))
        self.assertSetEqual(fn(FiniteSet(1), FiniteSet(0, _NAN)), FiniteSet(0, 1))

    def test_nan_is_in_no_domain(self) -> None:
        # Real values outside the domain still raise, NaN alongside them or not.
        for op in ("log", "rsqrt", "reciprocal"):
            with self.assertRaises(InvalidDomain):
                getattr(self.s, op)(FiniteSet(0, _NAN))
        with self.assertRaises(InvalidDomain):
            self.s.truediv(FiniteSet(1), FiniteSet(0, _NAN))
        # A NaN-only operand has no real value to check: log(NaN) is NaN.
        self.assertSetEqual(self.s.log(_NAN_SET), _NAN_SET)

    def test_nan_reaching_valid_data_raises(self) -> None:
        # Reductions, matmul and gathers would mix NaN into valid lanes.
        nan_padding = FiniteSet(0, _NAN)
        with self.assertRaises(InvalidDomain):
            self.s.reduction(torch.float16, torch.float16, "sum", nan_padding)
        with self.assertRaises(InvalidDomain):
            self.s.reduction(
                torch.float16,
                torch.float16,
                BATCH_MATMUL_OP,
                (nan_padding, nan_padding),
            )
        with self.assertRaises(InvalidDomain):
            self.s.indirect_indexing(nan_padding, 4)
        # One zero-padded matmul operand still zeroes the products.
        self.assertSetEqual(
            self.s.reduction(
                torch.float16,
                torch.float16,
                BATCH_MATMUL_OP,
                (nan_padding, FiniteSet(0)),
            ),
            FiniteSet(0),
        )

    def test_pick_element_never_picks_nan(self) -> None:
        self.assertEqual(padding.pick_element(FiniteSet(2, _NAN)), 2)
        self.assertEqual(padding.pick_element(S.Reals.union(_NAN_SET)), 0)
        with self.assertRaises(ValueError):
            padding.pick_element(_NAN_SET)


instantiate_parametrized_tests(TestDomainRangeSolverNaN)


class _FixedLoadSolver(DomainRangeSolver):
    """A DomainRangeSolver whose loads return a fixed set per buffer."""

    def __init__(self, ranges: dict[str, Any]) -> None:
        super().__init__()
        self.ranges = ranges

    def load(self, name, index):
        return self.ranges[name]


class _DtypeGraph:
    """Stands in for V.graph: SetDtypeTracer.load only calls get_dtype."""

    def __init__(self, dtypes: dict[str, torch.dtype]) -> None:
        self.dtypes = dtypes

    def get_dtype(self, name: str) -> torch.dtype:
        return self.dtypes[name]


class TestSetDtypeTracer(unittest.TestCase):
    """SetDtypeTracer pairs each set with the value's dtype."""

    def _run(self, fn, inputs: dict[str, tuple[Any, torch.dtype]]):
        ranges = {name: s for name, (s, _) in inputs.items()}
        dtypes = {name: d for name, (_, d) in inputs.items()}
        tracer = SetDtypeTracer(_FixedLoadSolver(ranges))
        with V.set_graph_handler(_DtypeGraph(dtypes)), V.set_ops_handler(tracer):
            r = fn()
        # ops.* wraps the handler's result in an OpsValue.
        return getattr(r, "value", r)

    def _load(self, name: str):
        return ops.load(name, _I)

    def test_load_carries_buffer_dtype(self) -> None:
        r = self._run(lambda: self._load("x"), {"x": (FiniteSet(0), torch.float16)})
        self.assertEqual(r, SetDtype(FiniteSet(0), torch.float16))

    def test_constant_carries_its_dtype(self) -> None:
        r = self._run(lambda: ops.constant(2.0, torch.float32), {})
        self.assertEqual(r, SetDtype(FiniteSet(2.0), torch.float32))

    def test_pointwise_propagates_set_and_dtype(self) -> None:
        r = self._run(
            lambda: ops.mul(self._load("x"), ops.constant(2.0, torch.float16)),
            {"x": (FiniteSet(3), torch.float16)},
        )
        self.assertEqual(r, SetDtype(FiniteSet(6.0), torch.float16))

    def test_binary_promotes_dtype(self) -> None:
        r = self._run(
            lambda: ops.add(self._load("x"), self._load("y")),
            {"x": (FiniteSet(1), torch.float16), "y": (Interval(0, 1), torch.float32)},
        )
        self.assertEqual(r, SetDtype(Interval(1.0, 2.0), torch.float32))

    def test_comparison_is_bool(self) -> None:
        r = self._run(
            lambda: ops.lt(self._load("x"), self._load("y")),
            {"x": (FiniteSet(1), torch.float16), "y": (FiniteSet(2), torch.float16)},
        )
        self.assertEqual(r, SetDtype(FiniteSet(True), torch.bool))

    def test_to_dtype_changes_dtype_keeps_set(self) -> None:
        r = self._run(
            lambda: ops.to_dtype(self._load("x"), torch.float32),
            {"x": (FiniteSet(50), torch.float16)},
        )
        self.assertEqual(r, SetDtype(FiniteSet(50), torch.float32))

    def test_reduction_dtype(self) -> None:
        r = self._run(
            lambda: ops.reduction(torch.float32, torch.float16, "sum", self._load("x")),
            {"x": (FiniteSet(0), torch.float16)},
        )
        self.assertEqual(r, SetDtype(S.Reals, torch.float32))

    def test_unknown_input_dtype_stays_unknown(self) -> None:
        tracer = SetDtypeTracer(_FixedLoadSolver({}))
        with V.set_ops_handler(tracer):
            r = ops.add(SetDtype(FiniteSet(1), None), SetDtype(FiniteSet(2), None))
        self.assertEqual(r, SetDtype(FiniteSet(3), None))

    def test_wrapped_domain_checks_still_raise(self) -> None:
        with self.assertRaises(InvalidDomain) as ctx:
            self._run(
                lambda: ops.log(self._load("x")),
                {"x": (FiniteSet(0), torch.float16)},
            )
        self.assertEqual(ctx.exception.func, "log")

    @parametrize(
        "dtype,value,overflows",
        [
            (torch.float16, 11.0, False),
            (torch.float16, 12.0, True),
            (torch.bfloat16, 12.0, True),
            (torch.float32, 50.0, False),
            (torch.float32, 89.0, True),
        ],
    )
    def test_exp_domain_depends_on_dtype(self, dtype, value, overflows) -> None:
        def fn():
            return ops.exp(self._load("x"))

        inputs = {"x": (FiniteSet(value), dtype)}
        if overflows:
            with self.assertRaises(InvalidDomain) as ctx:
                self._run(fn, inputs)
            self.assertEqual(ctx.exception.func, "exp")
            self.assertFalse(ctx.exception.domain.contains(value))
        else:
            r = self._run(fn, inputs)
            self.assertEqual(r.dtype, dtype)
            self.assertAlmostEqual(float(next(iter(r.set))), math.exp(value), 2)

    def test_exp_uses_dtype_after_upcast(self) -> None:
        # 50 overflows exp in fp16 but not after an upcast to fp32.
        r = self._run(
            lambda: ops.exp(ops.to_dtype(self._load("x"), torch.float32)),
            {"x": (FiniteSet(50), torch.float16)},
        )
        self.assertEqual(r.dtype, torch.float32)

    def test_exp_mask_value_is_in_domain(self) -> None:
        # The fp16 exp mask value must pass the fp16 exp domain check.
        pad = padding.get_pad_value("exp", torch.float16)
        r = self._run(
            lambda: ops.exp(self._load("x")), {"x": (FiniteSet(pad), torch.float16)}
        )
        self.assertEqual(r.set, FiniteSet(0.0))


instantiate_parametrized_tests(TestSetDtypeTracer)


class TestArgNameToIndexTracer(unittest.TestCase):
    """inx2arg maps InvalidDomain.arg_index back to the buffer it blames."""

    def _trace(self, fn) -> list[str]:
        tracer = ArgNameToIndexTracer()
        with V.set_ops_handler(tracer):
            fn()
        return tracer.get_inx2arg()

    def _load(self, name: str):
        return ops.load(name, _I)

    def test_binary_op(self) -> None:
        self.assertEqual(
            self._trace(lambda: ops.mul(self._load("x"), self._load("y"))), ["x", "y"]
        )

    def test_operand_order_is_argument_order(self) -> None:
        # truediv(y, x): arg_index=1 (the divisor) must resolve to x.
        self.assertEqual(
            self._trace(lambda: ops.truediv(self._load("y"), self._load("x"))),
            ["y", "x"],
        )

    def test_same_buffer_twice(self) -> None:
        self.assertEqual(
            self._trace(lambda: ops.mul(self._load("x"), self._load("x"))), ["x", "x"]
        )

    def test_constant_takes_a_slot(self) -> None:
        inx = self._trace(
            lambda: ops.truediv(ops.constant(1.0, torch.float16), self._load("x"))
        )
        self.assertEqual(len(inx), 2)
        self.assertTrue(inx[0].startswith("__const_"))
        self.assertEqual(inx[1], "x")

    def test_ternary_op(self) -> None:
        self.assertEqual(
            self._trace(
                lambda: ops.where(self._load("c"), self._load("a"), self._load("b"))
            ),
            ["c", "a", "b"],
        )

    def test_store_is_ignored(self) -> None:
        self.assertEqual(
            self._trace(lambda: ops.store("out", _I, ops.exp(self._load("x")))), ["x"]
        )

    def test_reduction(self) -> None:
        self.assertEqual(
            self._trace(
                lambda: ops.store_reduction(
                    "out",
                    _I,
                    ops.reduction(
                        torch.float16,
                        torch.float16,
                        BATCH_MATMUL_OP,
                        (self._load("x"), self._load("y")),
                    ),
                )
            ),
            ["x", "y"],
        )

    def test_indirect_indexing(self) -> None:
        tracer = ArgNameToIndexTracer()
        with V.set_ops_handler(tracer):
            idx = ops.indirect_indexing(self._load("idx"), 4)
            self.assertEqual(idx, padding._INDIRECT_INDEX_SYM)
        self.assertEqual(tracer.get_inx2arg(), ["idx"])

    def test_buffer_passed_as_kwarg_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "kwargs"):
            self._trace(lambda: ops.clamp(self._load("x"), min=self._load("y")))

    def test_nested_operands_index_within_their_own_op(self) -> None:
        # Expected behavior, not a bug. Each load records its position among
        # the ArgName args of the op that consumes it, and the positions of
        # different ops are merged, so for exp(a) * (b / c) the indices are
        # a: 0 (in exp), b: 0 and c: 1 (in truediv). The merged list is then
        # not the operand order of any single op: index 1 resolves to b, not
        # to truediv's divisor c. This is fine on Spyre because split_multi_ops
        # runs before track_padding_value and leaves one arithmetic op per
        # body, so inx2arg only sees single-op bodies, where arg_index is the
        # op's own operand position. If padding analysis ever runs on unsplit
        # bodies, this mapping must change.
        tracer = ArgNameToIndexTracer()
        with V.set_ops_handler(tracer):
            ops.mul(
                ops.exp(self._load("a")),
                ops.truediv(self._load("b"), self._load("c")),
            )
        self.assertEqual(tracer.name2inx, {"a": [0], "b": [0], "c": [1]})
        self.assertEqual(tracer.get_inx2arg(), ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Hooks into CustomPreSchedulingPasses
# ---------------------------------------------------------------------------


class CustomPreSchedulingPassesWithCapture(CustomPreSchedulingPasses):
    """Subclass of CustomPreSchedulingPasses that captures the operations list
    after all built-in passes (including insert_bmm_padding) have run."""

    test_instance: Optional["TestInsertPaddingIR"] = None

    @classmethod
    def initialize(cls, test_instance: "TestInsertPaddingIR") -> None:
        cls.test_instance = test_instance

    @override
    def __call__(self, graph: GraphLowering) -> None:
        assert self.test_instance is not None
        super().__call__(graph)
        self.test_instance.captured_operations = list(graph.operations)
        self.test_instance.captured_name_to_buffer = dict(graph.name_to_buffer)
        # Reads must be traced while V.graph is live; record them per op here.
        self.test_instance.captured_read_names = {
            op.get_name(): {
                d.name for d in op.get_read_writes().reads if hasattr(d, "name")
            }
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
        }


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class TestInsertPaddingIR(unittest.TestCase):
    """IR-level structural tests for insert_bmm_padding.

    Each test compiles a small matmul function, captures the operations list
    after CustomPreSchedulingPasses finishes (which includes insert_bmm_padding),
    and asserts structural properties of the resulting operation sequence.
    """

    captured_operations: list[Operation] = []
    captured_name_to_buffer: dict[str, Any] = {}
    captured_read_names: dict[str, set[str]] = {}

    def setUp(self) -> None:
        torch.manual_seed(0xAFFE)
        self.patchers: list[Any] = []

        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(ts_inductor_config.patch("sencores", 1))
        # These tests assert the exact op sequence insert_bmm_padding produces.
        # Clone-at-boundaries runs later in the same pass pipeline and can
        # insert an unrelated aten.clone ComputedBuffer for a graph input read
        # by both the padding copy op and the matmul -- disable it here (in
        # both the allocator and utils bindings) so the captured ops reflect
        # only insert_bmm_padding's own output. tearDown's loop restores both
        # via the same patchers list as everything else here.
        self.patchers.extend(
            patch.object(mod, "clone_at_graph_boundaries", lambda: False)
            for mod in (allocator, scratchpad_utils)
        )

        CustomPreSchedulingPassesWithCapture.initialize(self)
        self.patchers.append(
            patch.object(
                passes,
                "CustomPreSchedulingPasses",
                CustomPreSchedulingPassesWithCapture,
            )
        )

        for p in self.patchers:
            p.__enter__()

        torch.compiler.reset()

    def tearDown(self) -> None:
        for p in self.patchers:
            p.__exit__(None, None, None)
        torch.compiler.reset()

    def compile_and_capture(
        self,
        fn: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ) -> list[Operation]:
        """Compile ``fn`` with the given Spyre-device args and return the
        captured operations list after CustomPreSchedulingPasses."""
        self.captured_operations = []
        compiled = torch.compile(fn, fullgraph=True)
        compiled(*args)
        return self.captured_operations

    def compile_and_run(
        self,
        fn: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ) -> tuple[list[Operation], torch.Tensor]:
        """Compile ``fn`` once, capture IR operations, and return (ops, result)."""
        self.captured_operations = []
        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(*args)
        return self.captured_operations, result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matmul_ops(operations: list[Operation]) -> list[ComputedBuffer]:
        """Return all ComputedBuffer operations with BATCH_MATMUL_OP reduction type."""
        result = []
        for op in operations:
            if not isinstance(op, ComputedBuffer):
                continue
            data = op.data
            if isinstance(data, Reduction) and data.reduction_type == BATCH_MATMUL_OP:
                result.append(op)
        return result

    @staticmethod
    def _ops_before(
        operations: list[Operation], target: ComputedBuffer
    ) -> list[Operation]:
        """Return all operations that appear before ``target`` in the list."""
        idx = operations.index(target)
        return operations[:idx]

    @staticmethod
    def _padded_buf_ops(operations: list[Operation]) -> list[ComputedBuffer]:
        """Return ComputedBuffer operations whose origin_node.target is aten.constant_pad_nd."""
        padded_buf_ops = []
        aten_op = torch.ops.aten.constant_pad_nd.default
        for op in operations:
            if isinstance(op, ComputedBuffer):
                if getattr(getattr(op, "origin_node", None), "target", None) == aten_op:
                    padded_buf_ops.append(op)
        return padded_buf_ops

    @staticmethod
    def _restickify_ops(operations: list[Operation]) -> list[ComputedBuffer]:
        """Return ComputedBuffer operations whose origin_node.target is spyre.restickify."""
        restickify_op = torch.ops.spyre.restickify.default
        return [
            op
            for op in operations
            if isinstance(op, ComputedBuffer)
            and getattr(getattr(op, "origin_node", None), "target", None)
            == restickify_op
        ]

    @staticmethod
    def _constant_ops(ops: list[Operation]) -> list[SpyreConstantFallback]:
        """Return SpyreConstantFallback ops (fill-value constants for padding)."""
        return [op for op in ops if isinstance(op, SpyreConstantFallback)]

    def _assert_constant_pad_nd_ops(self, ops: list[Operation]) -> None:
        """Assert the constant_pad_nd 4-op pattern."""

        # Expected 4 operations (all with FixedTiledLayout):
        # 1. ComputedBuffer: output buffer allocation
        # 2. SpyreConstantFallback: padding fill constant
        # 3. ComputedBuffer: fill padding region (reads constant, writes output)
        # 4. ComputedBuffer: copy input data (reads input, writes output)

        self.assertEqual(len(ops), 4, f"Expected 4 ops, got {len(ops)}")

        for op in ops:
            self.assertTrue(
                isinstance(op.get_layout(), FixedTiledLayout),
                f"{type(op).__name__} should have FixedTiledLayout",
            )

        computed_buffer_ops = [op for op in ops if isinstance(op, ComputedBuffer)]
        self.assertEqual(len(computed_buffer_ops), 3, "Expected 3 computed buffers")

        padded_buf_ops = next(iter(self._padded_buf_ops(computed_buffer_ops)), None)
        self.assertIsNotNone(padded_buf_ops, "Expected 1 output buffer")

        constant_op = next(iter(self._constant_ops(ops)), None)
        self.assertIsNotNone(constant_op, "Expected 1 constant buffer")

        mutation_ops = [op for op in computed_buffer_ops if op is not padded_buf_ops]
        self.assertEqual(len(mutation_ops), 2, "Expected 2 mutation ops")

        # Verify dependencies (mutation ops write to output buffer):
        # - Fill padding region: reads constant, writes output
        # - Copy input data: reads input, writes output

        for op in ops:
            self.assertTrue(
                padded_buf_ops.origin_node in op.origins,
                f"{op.name} origins should contain {padded_buf_ops.origin_node}",
            )
        for op in mutation_ops:
            self.assertTrue(
                op.get_layout() == padded_buf_ops.get_layout(),
                f"Mutation op {op.name} should have same layout as output buffer {padded_buf_ops.name}",
            )
        self.assertTrue(
            any(constant_op.name in op.get_read_names() for op in mutation_ops),
            "One mutation op should read from constant buffer",
        )
        self.assertTrue(
            any("arg" in name for op in mutation_ops for name in op.get_read_names()),
            "One mutation op should read from input buffer",
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_mm_unaligned_k_pads(self) -> None:
        """2D mm with K=67 (unaligned) — only y is padded before the matmul.

        x is untouched; reduction_ranges stays at K=67.
        y gets 2 overwrites (fill + copy).
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        # 67 is not a multiple of stick_size (64), so padding should occur.
        assert 67 % stick_size != 0

        x_cpu = torch.randn(55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]

        # reduction_ranges stays at K (not K_padded): the K→K_padded extension
        # happens at SDSC codegen time, not in the IR.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(
            k_actual,
            67,
            f"reduction_ranges should stay at K=67, got {k_actual}",
        )

        # 2 overwrite ops before the matmul: fill + copy for y only (x untouched).
        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_mm_aligned_k_no_padding(self) -> None:
        """2D mm with K=128 (aligned) — no padding ops inserted."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 128 % stick_size == 0

        x_cpu = torch.randn(55, 128, dtype=dtype)
        w_cpu = torch.randn(128, 64, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]

        # reduction_ranges should remain K=128.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(k_actual, 128, f"K should stay 128, got {k_actual}")

        # No overwrite ops should appear before the matmul.
        ops_before = self._ops_before(ops, mm)
        self.assertEqual(len(ops_before), 0, "Expected no padding ops for aligned K")

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_bmm_3d_unaligned_k_pads(self) -> None:
        """3D bmm (B,M,K)×(B,K,N) with K=67 — only y is padded before bmm."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 55, 67, dtype=dtype)
        w_cpu = torch.randn(2, 67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return torch.bmm(x, w)

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one batched matmul op")
        mm = matmuls[0]

        # reduction_ranges stays at K.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_bmm_3d_2d_unaligned_k_pads(self) -> None:
        """3D×2D bmm: (B,M,K)×(K,N) with K=67 — only y is padded."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        # reduction_ranges stays at K.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_matmul_4d_unaligned_k_pads(self) -> None:
        """4D matmul (B,H,M,K)×(B,H,K,N) with K=67 — only y is padded."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 3, 55, 67, dtype=dtype)
        w_cpu = torch.randn(2, 3, 67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        # reduction_ranges stays at K.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_einsum_mk_kn_mn_pads(self) -> None:
        """einsum('mk,kn->mn') with K=67 — y is padded; reduction_ranges stays at K."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return torch.einsum("mk,kn->mn", x, w)

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        # reduction_ranges stays at K=67.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_padding_constants_deduped(self) -> None:
        """Two matmuls with the same shapes yield exactly one spyre.constant after dedup.

        Both matmuls pad only y with fill_value=0.0 at the same dtype, so two
        spyre.constant FX nodes are created (one per pad sequence) and lowered to two
        SpyreConstantFallback IR ops.  dedup_and_promote_constants then merges them into
        one canonical constant and moves it to the head of operations.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 55, 67, dtype=dtype)
        w1_cpu = torch.randn(2, 67, 128, dtype=dtype)
        w2_cpu = torch.randn(2, 67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w1 = w1_cpu.to(device="spyre")
        w2 = w2_cpu.to(device="spyre")

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        ops, result = self.compile_and_run(fn, (x, w1, w2))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 2, "Expected 2 matmul ops")

        # dedup_and_promote_constants merges all (0.0, fp16, spyre) constants into one.
        constant_ops = self._constant_ops(ops)
        self.assertEqual(
            len(constant_ops),
            1,
            f"Expected 1 spyre.constant after IR dedup, got {len(constant_ops)}",
        )

        # The surviving constant must be at the head of operations.
        self.assertIs(
            ops[0],
            constant_ops[0],
            "Expected the surviving spyre.constant to be the first operation",
        )

        torch.testing.assert_close(
            fn(x_cpu, w1_cpu, w2_cpu), result.cpu(), atol=0.1, rtol=0.1
        )

    def test_origin_node_set_on_rebuilt_matmul(self) -> None:
        """Rebuilt matmul ComputedBuffer retains origin_node from the original.

        This is required by LX planning (scratchpad.py:298) which accesses
        op.origin_node.target._opname directly.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        self.assertIsNotNone(
            mm.origin_node,
            "origin_node should not be None after _rebuild_matmul",
        )

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_padded_buffer_sizes_y_only(self) -> None:
        """Only y is padded; x is untouched.  y_padded has host size [B, K_padded, N].

        For the 3D bmm shape [B=2, K=67, N=128]: SpyreTensorLayout for
        [B, K_padded, N] sticked on N lays out device dims as
        [K_padded, N_sticks, B, stick_size], so device_size[-4] == K_padded.

        Exactly one padded buffer op appears before the matmul.
        """

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        B, M, K, N = 2, 55, 67, 128
        k_padded = ((K + stick_size - 1) // stick_size) * stick_size

        x_cpu = torch.randn(B, M, K, dtype=dtype)
        w_cpu = torch.randn(B, K, N, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return torch.bmm(x, w)

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        ops_before = self._ops_before(ops, mm)

        padded_buf_ops = self._padded_buf_ops(ops_before)
        # Only y is padded — exactly one padded buffer.
        self.assertEqual(
            len(padded_buf_ops),
            1,
            f"Expected 1 padded buffer (y only), found {len(padded_buf_ops)}: "
            f"{[[int(s) for s in op.get_size()] for op in padded_buf_ops]}",
        )

        y_padded_buf_ops = padded_buf_ops[0]

        # y_padded's host size is [B, K_padded, N]: lower_pad_sequence builds it at the
        # padded shape and no host-downgrade is applied.  The IR iteration space (via
        # reduction_ranges) stays at K; only the buffer allocation is K_padded.
        host_size = [int(s) for s in y_padded_buf_ops.get_size()]
        self.assertEqual(
            host_size,
            [B, k_padded, N],
            f"y_padded host size should be [B,k_padded,N]=[{B},{k_padded},{N}], "
            f"got {host_size}",
        )

        # y's device_layout.device_size must reflect K_padded in the K-row device dim.
        # For [B, K_padded, N] sticked on N: device_size = [K_padded, N_sticks, B, stick_size].
        # K_padded sits at device_size[-4] (index -stride_idx-2 with stride_idx=2 for K).
        layout = y_padded_buf_ops.get_layout()
        self.assertIsInstance(layout, FixedTiledLayout)
        assert isinstance(layout, FixedTiledLayout)
        dev_size = list(layout.device_layout.device_size)
        self.assertEqual(
            dev_size[-4],
            k_padded,
            f"y device_size[-4] should be K_padded={k_padded}, "
            f"got {dev_size[-4]}; full device_size={dev_size}",
        )

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_padded_buffer_preserves_stick_dimension(self) -> None:
        """y's padded buffer preserves the original within-stick stride.

        ``lower_pad_sequence`` constructs the padded buffer's ``SpyreTensorLayout``
        from the padded host size/stride so that ``device_coordinates[-1]`` (the
        stick coordinate expression) is identical for both the original and padded
        buffers.  Concretely, ``stride_map[-1]`` must be 1 for the padded buffer.

        y is sticked on N (the output dim), which is contiguous, so
        ``stride_map[-1] == 1``.  The test catches a regression that confused the
        stick dim (e.g. producing ``stride_map[-1] == K_padded`` from a default
        layout with the wrong dim_order).
        """

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        def make_args(*shapes):
            cpu = tuple(torch.randn(*s, dtype=dtype) for s in shapes)
            dev = tuple(t.to(device="spyre") for t in cpu)
            return cpu, dev

        cpu0, dev0 = make_args((55, 67), (67, 128))
        cpu1, dev1 = make_args((2, 55, 67), (2, 67, 128))
        cpu2, dev2 = make_args((55, 67), (67, 128))

        cases: list[tuple[str, Callable[..., torch.Tensor], tuple, tuple]] = [
            ("mm [55,67]x[67,128]", lambda x, w: x @ w, dev0, cpu0),
            ("bmm [2,55,67]x[2,67,128]", lambda x, w: torch.bmm(x, w), dev1, cpu1),
            (
                "einsum mk,kn->mn [55,67]x[67,128]",
                lambda x, w: torch.einsum("mk,kn->mn", x, w),
                dev2,
                cpu2,
            ),
        ]

        for name, fn, args, cpu_args in cases:
            with self.subTest(case=name):
                ops, result = self.compile_and_run(fn, args)
                matmuls = self._matmul_ops(ops)
                self.assertEqual(len(matmuls), 1, f"{name}: expected 1 matmul")
                mm = matmuls[0]
                ops_before = self._ops_before(ops, mm)

                padded_buf_ops = self._padded_buf_ops(ops_before)
                self.assertEqual(
                    len(padded_buf_ops),
                    1,
                    f"{name}: expected exactly 1 padded buffer (y only)",
                )

                for op in padded_buf_ops:
                    layout = op.get_layout()
                    self.assertIsInstance(
                        layout,
                        FixedTiledLayout,
                        f"{name}: padded buffer has wrong layout type {type(layout)}",
                    )
                    sm_last = int(list(layout.device_layout.stride_map)[-1])
                    self.assertEqual(
                        sm_last,
                        1,
                        f"{name}: padded buffer stride_map[-1]={sm_last}, "
                        f"expected 1 (within-stick dim is contiguous); "
                        f"size={[int(s) for s in op.get_size()]}",
                    )

                torch.testing.assert_close(
                    fn(*cpu_args), result.cpu(), atol=0.1, rtol=0.1
                )

    def test_mm_square_unaligned_k_pads_y_only(self) -> None:
        """Square mm (M==K==N) with unaligned K — only y is padded, not x.

        Verifies that x/y identification works correctly when both inputs
        have the same shape (M==K==N).
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        K = 67
        assert K % stick_size != 0

        x_cpu = torch.randn(K, K, dtype=dtype)
        w_cpu = torch.randn(K, K, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), K)
        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)
        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_4d_matmul_xt_restickify_pads_y(self) -> None:
        """4D matmul where x is transposed (forcing restickify) and K is unaligned.

        x is stored as [B, H, K, M] and transposed to [B, H, M, K] for the
        matmul.  Restickify reorders x's device dims; insert_bmm_padding must
        preserve the original inner_fn's x loader unchanged so work_division
        sees valid device coordinates.  Regression test for that fix.
        """
        dtype = torch.float16
        B, H, M, K, N = 12, 32, 256, 4, 128

        x_cpu = torch.randn(B, H, K, M, dtype=dtype) * 0.01
        y_cpu = torch.randn(B, H, K, N, dtype=dtype) * 0.01
        x = x_cpu.to(device="spyre")
        y = y_cpu.to(device="spyre")

        def fn(x, y):
            return torch.matmul(x.transpose(-1, -2), y)

        ops, result = self.compile_and_run(fn, (x, y))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), K)
        # ops_before contains a restickify op for x in addition to the 4-op
        # padding sequence for y, so _assert_constant_pad_nd_ops (which expects
        # exactly 4 ops) cannot be used here.  Correctness is verified by assert_close.
        torch.testing.assert_close(fn(x_cpu, y_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_bmm_invariant_y_unaligned_k_pads_for_each_tile(self) -> None:
        """A loop-invariant weight with unaligned K inside a for_each_tile body.

        x is [E, M, K=67] tiled over E and w is a loop-invariant [K, N].  The
        padding ops for w join the loop group, but must not inherit the
        matmul's per-trip read advance: that record is positional in the
        matmul's reads (the tiled x), so on the pad ops it advanced the read of
        the invariant w and corrupted the padded copy on every trip after the
        first, leaving ~64% of the result wrong (#5440).
        """
        dtype = torch.float16
        E, M, K, N = 3, 64, 67, 128
        assert K % get_elem_in_stick(dtype) != 0

        x_cpu = torch.randn(E, M, K, dtype=dtype)
        w_cpu = torch.randn(K, N, dtype=dtype)

        def fn(x, w):
            def body(_, ops):
                x_tile, w_whole = ops
                return None, torch.matmul(x_tile, w_whole)

            return for_each_tile(body, (x, w), dims=(0, None), tile_size=1, out_dim=0)[
                1
            ]

        _, result = self.compile_and_run(fn, (x_cpu.to("spyre"), w_cpu.to("spyre")))
        torch.testing.assert_close(
            torch.matmul(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1
        )

    def test_linear_invariant_weight_unaligned_k_pads_for_each_tile(self) -> None:
        """An ``nn.Linear`` weight ``[N, K]`` with unaligned K, loop-invariant.

        The same padding-op bug as the ``x @ w`` case above, on the most common
        weight pattern: the weight is stored ``[N, K]``, padded, and restickified
        for the matmul, all inside the loop group.  Over 70% of the result was
        wrong before the padding ops stopped inheriting the matmul's read advance
        (#5440).
        """
        import torch.nn.functional as F

        dtype = torch.float16
        E, M, K, N = 3, 64, 67, 128
        assert K % get_elem_in_stick(dtype) != 0

        generator = torch.Generator().manual_seed(0)
        x_cpu = torch.randn(E, M, K, dtype=dtype, generator=generator)
        w_cpu = torch.randn(N, K, dtype=dtype, generator=generator)

        def fn(x, w):
            def body(_, ops):
                x_tile, w_whole = ops
                return None, F.linear(x_tile, w_whole)

            return for_each_tile(body, (x, w), dims=(0, None), tile_size=1, out_dim=0)[
                1
            ]

        _, result = self.compile_and_run(fn, (x_cpu.to("spyre"), w_cpu.to("spyre")))
        torch.testing.assert_close(
            F.linear(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1
        )

    def test_tiled_expert_weight_unaligned_k_pads_for_each_tile(self) -> None:
        """A per-expert weight ``[E, K, N]`` tiled with x, with unaligned K.

        Padding K to the stick boundary grows the per-expert stride of the weight
        (67 * 128 to 128 * 128).  The per-trip step ``splice_while_loops`` stamped
        before padding still held the unpadded stride, so every trip after the
        first read the wrong place and the device faulted (#5573).  At K=64 there
        is no padding and it was always correct.
        """
        dtype = torch.float16
        E, M, K, N = 4, 64, 67, 128
        assert K % get_elem_in_stick(dtype) != 0

        generator = torch.Generator().manual_seed(0)
        x_cpu = torch.randn(E, M, K, dtype=dtype, generator=generator)
        w_cpu = torch.randn(E, K, N, dtype=dtype, generator=generator)

        def fn(x, w):
            def body(_, ops):
                return None, torch.matmul(ops[0], ops[1])

            return for_each_tile(body, (x, w), dims=(0, 0), tile_size=1, out_dim=0)[1]

        _, result = self.compile_and_run(fn, (x_cpu.to("spyre"), w_cpu.to("spyre")))
        torch.testing.assert_close(
            torch.matmul(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1
        )

    def test_tiled_transposed_expert_weight_unaligned_k_pads_for_each_tile(
        self,
    ) -> None:
        """A per-expert weight stored ``[E, N, K]`` and used transposed, K unaligned.

        The same stale per-trip step as the ``[E, K, N]`` case, on a weight whose K
        is its innermost dim, so the padded buffer's per-expert stride and the
        within-expert strides all differ from the unpadded ones.  It returned NaN
        for about 70% of the result before the step was re-strided (#5573).
        """
        dtype = torch.float16
        E, M, K, N = 4, 64, 67, 128
        assert K % get_elem_in_stick(dtype) != 0

        generator = torch.Generator().manual_seed(0)
        x_cpu = torch.randn(E, M, K, dtype=dtype, generator=generator)
        w_cpu = torch.randn(E, N, K, dtype=dtype, generator=generator)

        def fn(x, w):
            def body(_, ops):
                return None, torch.matmul(ops[0], ops[1].transpose(-1, -2))

            return for_each_tile(body, (x, w), dims=(0, 0), tile_size=1, out_dim=0)[1]

        _, result = self.compile_and_run(fn, (x_cpu.to("spyre"), w_cpu.to("spyre")))
        expected = torch.einsum("emk,enk->emn", x_cpu.float(), w_cpu.float())
        torch.testing.assert_close(expected, result.cpu().float(), atol=0.1, rtol=0.1)

    def test_padded_read_step_shared_by_two_dims_is_rejected(self) -> None:
        """A per-trip step that two non-trivial dims of y could own is an error.

        The stamped step is a stride of y's layout; when two dims of size above one
        share it, there is no telling which one the loop walks, and re-striding the
        wrong one would read the wrong place after the first trip.  It must raise
        rather than leave the unpadded step in place (#5573).
        """
        from torch._inductor.ir import FixedLayout

        from torch_spyre._inductor.errors import Unsupported
        from torch_spyre._inductor.padding import (
            _refresh_loop_advance_for_padded_read,
        )

        class _Buf:
            def __init__(self, name, size, stride):
                self._name = name
                self._layout = FixedLayout(
                    torch.device("cpu"), torch.float16, size, stride
                )

            def get_name(self):
                return self._name

            def get_layout(self):
                return self._layout

        # Two dims of size 2 and 3 share stride 4 (an overlapping view).
        y = _Buf("y", [2, 3, 5], [4, 4, 1])
        padded = _Buf("y_padded", [2, 3, 8], [24, 8, 1])

        from torch_spyre._inductor.loop_info import CoarseTileInfo

        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[2],
            loop_tiled_dims=[[]],
            squeezed_advance_per_read=[[[(4, 1)]]],
        )

        class _Op:
            loop_info = info

            def get_read_writes(self):
                from torch._inductor.dependencies import MemoryDep
                import sympy

                x = sympy.Symbol("d0")
                dep = MemoryDep("y_padded", x, (x,), (6,))
                return type("RW", (), {"reads": [dep]})()

        with self.assertRaisesRegex(Unsupported, "cannot tell which dim"):
            _refresh_loop_advance_for_padded_read(_Op(), y, padded)

    def test_transposed_y_unaligned_k_pads_then_restickifies(self) -> None:
        """``x @ w.t()`` with unaligned K -- the ``nn.Linear`` weight pattern.

        The weight is stored ``[N, K]`` with K within the stick, so the matmul
        needs it restickified to put N within the stick.  Padding runs before
        stickification, so the pass pads the weight's K while it is a plain host
        buffer and stickification then restickifies the *padded* buffer.  The
        matmul must read that restickify output, and the restickify must read
        the padded buffer, so the zero rows K..K_padded-1 are what the matmul
        accumulates over (issue #4208).
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        M, K, N = 64, 784, 512
        assert K % stick_size != 0

        x_cpu = torch.randn(M, K, dtype=dtype)
        w_cpu = torch.randn(N, K, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w.t()

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), K)

        ops_before = self._ops_before(ops, mm)
        padded = self._padded_buf_ops(ops_before)
        self.assertEqual(len(padded), 1, "expected exactly one padded buffer")
        padded_buf = padded[0]
        self.assertEqual(
            [int(s) for s in padded_buf.get_size()],
            [N, K + stick_size - K % stick_size],
            "y is padded along its K dim, in y's own [N, K] host order",
        )

        resticks = self._restickify_ops(ops_before)
        self.assertEqual(len(resticks), 1, "expected one restickify of the padded y")
        restick = resticks[0]
        self.assertIn(
            padded_buf.get_name(),
            self.captured_read_names[restick.get_name()],
            "the restickify must read the padded buffer, not the raw weight",
        )
        self.assertIn(
            restick.get_name(),
            self.captured_read_names[mm.get_name()],
            "the matmul must read the restickify output",
        )

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.5, rtol=0.05)

    def test_linear_unaligned_in_features(self) -> None:
        """``F.linear`` (bias included) compiles and is correct for unaligned
        ``in_features`` -- e.g. ``nn.Linear(28 * 28, 512)`` (issue #4208)."""
        dtype = torch.float16
        M, K, N = 32, 28 * 28, 512
        x_cpu = torch.randn(M, K, dtype=dtype)
        w_cpu = torch.randn(N, K, dtype=dtype) * 0.05
        b_cpu = torch.randn(N, dtype=dtype)
        args = tuple(t.to(device="spyre") for t in (x_cpu, w_cpu, b_cpu))

        def fn(x, w, b):
            return torch.nn.functional.linear(x, w, b)

        ops, result = self.compile_and_run(fn, args)
        self.assertEqual(len(self._matmul_ops(ops)), 1)
        self.assertEqual(len(self._padded_buf_ops(ops)), 1)
        torch.testing.assert_close(
            fn(x_cpu, w_cpu, b_cpu), result.cpu(), atol=0.5, rtol=0.05
        )

    def test_rebuilt_matmul_is_registered_in_graph(self) -> None:
        """The matmul ``insert_bmm_padding`` rebuilds must be the object the graph
        registry returns for its name.

        Layout selection runs after padding and commits its choice through
        ``V.graph.get_buffer(name)``; a stale registry entry would receive the
        committed layout while the live op in ``operations`` got none.
        """
        dtype = torch.float16
        x = torch.randn(55, 67, dtype=dtype).to(device="spyre")
        w = torch.randn(67, 128, dtype=dtype).to(device="spyre")

        ops = self.compile_and_capture(lambda x, w: x @ w, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]
        self.assertIs(self.captured_name_to_buffer[mm.get_name()], mm)
        for op in ops:
            if isinstance(op, ComputedBuffer):
                self.assertIs(self.captured_name_to_buffer[op.get_name()], op)


class TestLowerPadSequenceBroadcastDim(unittest.TestCase):
    """White-box tests for ``lower_pad_sequence``'s host-stride ranking.

    A buffer can carry a broadcast dim -- stride 0, size > 1 -- alongside a
    dim that needs stick-boundary padding (an uncompacted-rank copy leaves
    absent/broadcast dims at stride 0).  ``lower_pad_sequence`` ranks the padded buffer's host dims
    by the source buffer's raw host stride to preserve fastest/slowest
    ordering; ranking a stride-0 broadcast dim by that raw value put it
    first (0 sorts smallest) and handed it a real nonzero stride, corrupting
    addressing for a dimension that must stay stride 0.

    These tests call ``lower_pad_sequence`` directly against a hand-built
    ``InputBuffer`` (the pre-stickification path used by
    ``insert_bmm_padding``, mirroring ``coarse_tile.py``'s own FX-node-free
    ``arg_buf`` call convention), independent of any end-to-end
    coarse-tiling or SDPA compilation -- exercising the bug's structural
    precondition directly rather than reproducing it through a full
    ``torch.compile`` call.
    """

    def setUp(self) -> None:
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self) -> None:
        self._graph_ctx.__exit__(None, None, None)

    def test_broadcast_dim_kept_at_stride_zero_when_padding_adjacent_dim(
        self,
    ) -> None:
        """A stride-0, size>1 dim must stay stride 0 in the padded buffer.

        arg_buf mirrors a coarse-tile read-copy of V broadcast across 4 head
        tiles (dim 0, stride 0, size 4), with a real row dim (dim 1, size
        64, stride 32) and a K dim (dim 2, size 13, unaligned) padded to 32.
        """
        from torch_spyre._inductor.pass_utils import lower_pad_sequence

        device = torch.device("cpu")
        dtype = torch.float16

        arg_buf = InputBuffer(
            name="arg0",
            layout=FixedLayout(device, dtype, [4, 64, 13], [0, 1, 32]),
        )
        V.graph.name_to_buffer["arg0"] = arg_buf
        V.graph.graph_inputs["arg0"] = arg_buf
        insert_before = next(iter(V.graph.graph.nodes))

        padded_buf, _new_ops = lower_pad_sequence(
            arg_fx_node=None,
            padded_size=[4, 64, 32],
            device=device,
            dtype=dtype,
            dim=2,
            insert_before=insert_before,
            orig_stl=None,
            fill_value=0.0,
            arg_buf=arg_buf,
        )

        size = [int(s) for s in padded_buf.get_size()]
        stride = [int(s) for s in padded_buf.get_stride()]
        self.assertEqual(size, [4, 64, 32])
        self.assertEqual(
            stride[0],
            0,
            f"broadcast dim (size 4) must stay stride 0; got padded stride {stride}",
        )
        # The two real dims must still be assigned distinct, non-overlapping
        # strides consistent with their original relative order (dim 1 was
        # fastest-varying at stride 1, dim 2 was next at stride 32).
        self.assertLess(stride[1], stride[2])
        self.assertNotEqual(stride[1], stride[0])
        self.assertNotEqual(stride[2], stride[0])


def full_half_stick(dtype: torch.dtype):
    elems = get_elem_in_stick(dtype)
    return elems, elems // 2


COORD_MASK_DTYPES = [torch.float32, torch.float16]

# Shapes with stick dimension that doesn't line up with stick size, i.e. requires padding
SHAPES = [
    (64, 8),
    # multiple sticks get split by work division. Coordinate masking in combination with splitting wasn't supported by the backend copmpiler
    (64, 100),
]


def unpad_shape(shape, dtype):
    elems = get_elem_in_stick(dtype)
    stick_dim = shape[-1]
    unpadded = int(math.ceil(stick_dim / elems) * elems)
    return (*shape[:-1], unpadded)


@instantiate_parametrized_tests
class TestCoordinateMasking(unittest.TestCase):
    """Regression tests for operations that can create NaNs in padding.

    The creation of NaNs is suporessed by the coordinate masking
    in `superdsc.py`.
    """

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("shape", SHAPES)
    def test_division(self, shape, dtype) -> None:
        # Copied from the host, so the padding is recorded as 0: 0 / 0 must be
        # coordinate-masked.
        a_spyre = torch.full(shape, 2, dtype=dtype).to("spyre")
        b_spyre = torch.full(shape, 2, dtype=dtype).to("spyre")

        def fn(x, y):
            return x / y

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre, b_spyre)

        unpadded = unpad_shape(shape, dtype)
        exposed_padding = try_reinterpret_host_shape(result, unpadded).cpu()[
            :, shape[-1] :
        ]
        assert (exposed_padding == 0).all().item()

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("broadcast", ["divisor", "dividend"])
    @parametrize("half_sticks", [1, 5, 11])
    def test_division_broadcast(self, dtype, broadcast, half_sticks) -> None:
        # An operand broadcast along the padded stick dim (e.g. a reduction
        # result) must not get a coordinate mask: SAMV skips it, so realdiv's
        # inf mask leaks into the padding (inf / y), and at 5.5 sticks the
        # backend also corrupted valid lanes.
        full, half = full_half_stick(dtype)
        width = half * half_sticks
        shapes = {
            "divisor": ((full, width), (full, 1)),
            "dividend": ((full, 1), (full, width)),
        }
        x_shape, y_shape = shapes[broadcast]
        a_spyre = torch.full(x_shape, 3, dtype=dtype, device="spyre")
        b_spyre = torch.full(y_shape, 2, dtype=dtype, device="spyre")

        def fn(x, y):
            return x / y

        # dynamic=False: a recompile with symbolic widths takes another path.
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre, b_spyre)

        torch.testing.assert_close(
            result.cpu(), torch.full((full, width), 1.5, dtype=dtype)
        )

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("shape", SHAPES)
    def test_exp(self, shape, dtype) -> None:
        # The reinterpreted view carries a fresh layout with no recorded
        # padding value, so the analysis must assume the padding (222s from
        # the full allocation) can hold anything.
        unpadded = unpad_shape(shape, dtype)
        a_spyre_full = torch.full(unpadded, 222, dtype=dtype, device="spyre")
        a_spyre = try_reinterpret_host_shape(a_spyre_full, shape)

        def fn(x):
            return torch.exp(x)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre)

        exposed_padding = try_reinterpret_host_shape(result, unpadded).cpu()[
            :, shape[-1] :
        ]
        assert (exposed_padding == 0).all().item()

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("shape", SHAPES)
    def test_pow_neg(self, shape, dtype) -> None:
        # Copied from the host, so the padding is recorded as 0, which the op
        # cannot take: the kernel must coordinate-mask it.
        a_spyre = torch.full(shape, 0.5, dtype=dtype).to("spyre")

        def fn(x):
            return torch.pow(x, -1.0)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre)

        unpadded = unpad_shape(shape, dtype)
        exposed_padding = try_reinterpret_host_shape(result, unpadded).cpu()[
            :, shape[-1] :
        ]
        assert (exposed_padding == 0).all().item()

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("shape", SHAPES)
    def test_reciprocal(self, shape, dtype) -> None:
        # Copied from the host, so the padding is recorded as 0, which the op
        # cannot take: the kernel must coordinate-mask it.
        a_spyre = torch.full(shape, 0.5, dtype=dtype).to("spyre")

        def fn(x):
            return torch.reciprocal(x)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre)

        unpadded = unpad_shape(shape, dtype)
        exposed_padding = try_reinterpret_host_shape(result, unpadded).cpu()[
            :, shape[-1] :
        ]
        assert (exposed_padding == 0).all().item()

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("shape", SHAPES)
    def test_rsqrt(self, shape, dtype) -> None:
        full, half = full_half_stick(dtype)
        # Copied from the host, so the padding is recorded as 0, which the op
        # cannot take: the kernel must coordinate-mask it.
        a_spyre = torch.full(shape, 0.5, dtype=dtype).to("spyre")

        def fn(x):
            return torch.rsqrt(x)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre)

        unpadded = unpad_shape(shape, dtype)
        exposed_padding = try_reinterpret_host_shape(result, unpadded).cpu()[
            :, shape[-1] :
        ]
        assert (exposed_padding < 1e-4).all().item()

    @parametrize("dtype", COORD_MASK_DTYPES)
    @parametrize("shape", SHAPES)
    def test_log(self, shape, dtype) -> None:
        # Copied from the host, so the padding is recorded as 0, which the op
        # cannot take: the kernel must coordinate-mask it.
        a_spyre = torch.full(shape, 0.5, dtype=dtype).to("spyre")

        def fn(x):
            return torch.log(x)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(a_spyre)

        unpadded = unpad_shape(shape, dtype)
        exposed_padding = try_reinterpret_host_shape(result, unpadded).cpu()[
            :, shape[-1] :
        ]
        if dtype == torch.float16:
            assert (exposed_padding == 0).all().item()
        else:
            assert (exposed_padding < 1e-2).all().item()


class TestMatmulPaddingNaN(unittest.TestCase):
    """Regression tests for matmul output corruption that surfaces as NaNs.

    When the padding inside the sticks becomes corrupted by the presence of
    NaNs, the result of the matmul in the valid region also becomes corrupted.
    """

    # For now batched matmuls aren't supported with float 32
    DTYPE = torch.float16

    def _assert_matches_cpu_and_has_no_nan(
        self,
        device_result: torch.Tensor,
        cpu_expected: torch.Tensor,
        atol: float = 0.1,
        rtol: float = 0.1,
    ) -> None:
        result_cpu = device_result.cpu()
        self.assertFalse(
            torch.isnan(result_cpu).any().item(),
            f"Compiled Spyre output contains NaN:\n{result_cpu}",
        )
        torch.testing.assert_close(result_cpu, cpu_expected, atol=atol, rtol=rtol)

    def test_division(self) -> None:
        """``(x / z) @ y`` -- division_compiled.py."""
        full, half = full_half_stick(self.DTYPE)
        a_cpu = torch.full((full, half), 2, dtype=self.DTYPE)
        b_cpu = torch.full((half, full), 2, dtype=self.DTYPE)
        c_cpu = torch.full((full, half), 2, dtype=self.DTYPE)
        a_spyre = a_cpu.to(device="spyre")
        b_spyre = b_cpu.to(device="spyre")
        c_spyre = c_cpu.to(device="spyre")

        def fn(x, y, z):
            return (x / z) @ y

        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(a_spyre, b_spyre, c_spyre)
        expected = fn(a_cpu, b_cpu, c_cpu)
        self._assert_matches_cpu_and_has_no_nan(result, expected)

    def test_pow_neg(self) -> None:
        """``torch.pow(x, -1.0) @ y`` -- pow_neg_compiled.py."""
        full, half = full_half_stick(self.DTYPE)
        a_cpu = torch.full((full, half), 0.5, dtype=self.DTYPE)
        b_cpu = torch.full((half, full), 2, dtype=self.DTYPE)
        a_spyre = a_cpu.to(device="spyre")
        b_spyre = b_cpu.to(device="spyre")

        def fn(x, y):
            r = torch.pow(x, -1.0)
            return r @ y

        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(a_spyre, b_spyre)
        expected = fn(a_cpu, b_cpu)
        self._assert_matches_cpu_and_has_no_nan(result, expected)

    def test_reciprocal(self) -> None:
        full, half = full_half_stick(self.DTYPE)
        a_cpu = torch.full((full, half), 0.5, dtype=self.DTYPE)
        b_cpu = torch.full((half, full), 2, dtype=self.DTYPE)
        a_spyre = a_cpu.to(device="spyre")
        b_spyre = b_cpu.to(device="spyre")

        def fn(x, y):
            r = torch.reciprocal(x)
            return r @ y

        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(a_spyre, b_spyre)
        expected = fn(a_cpu, b_cpu)
        self._assert_matches_cpu_and_has_no_nan(result, expected)

    def test_rsqrt(self) -> None:
        full, half = full_half_stick(self.DTYPE)
        a_cpu = torch.full((full, half), 0.5, dtype=self.DTYPE)
        b_cpu = torch.full((half, full), 2, dtype=self.DTYPE)
        a_spyre = a_cpu.to(device="spyre")
        b_spyre = b_cpu.to(device="spyre")

        def fn(x, y):
            r = torch.rsqrt(x)
            return r @ y

        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(a_spyre, b_spyre)
        expected = fn(a_cpu, b_cpu)
        self._assert_matches_cpu_and_has_no_nan(result, expected)

    def test_log(self) -> None:
        full, half = full_half_stick(self.DTYPE)
        a_cpu = torch.full((full, half), 0.5, dtype=self.DTYPE)
        b_cpu = torch.full((half, full), 2, dtype=self.DTYPE)
        a_spyre = a_cpu.to(device="spyre")
        b_spyre = b_cpu.to(device="spyre")

        def fn(x, y):
            r = torch.log(x)
            return r @ y

        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(a_spyre, b_spyre)
        expected = fn(a_cpu, b_cpu)
        self._assert_matches_cpu_and_has_no_nan(result, expected)


class TestRecordedPadding(unittest.TestCase):
    """The padding value recorded on a tensor's layout must stay true.

    A fill records its value as the padding, and the padding analysis of a
    later graph trusts it for a graph input. A graph that writes the input must
    update the record (the wrapper's set_tensor_padding_value), or the next
    graph reads padding it no longer holds.
    """

    # 3 fp16 elements in a 64-element stick: 61 padding elements.
    SIZE = 3

    def setUp(self) -> None:
        # Compile every graph afresh: a cached graph skips track_padding_value,
        # and the dynamic-shape tests would then pass without exercising it.
        torch._dynamo.reset()
        patcher = t_inductor_config.patch("force_disable_caches", True)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)

    def _filled(self, value: float) -> torch.Tensor:
        t = torch.empty(self.SIZE, dtype=torch.float16, device="spyre")
        t.fill_(value)
        self.assertEqual(t.device_tensor_layout().padding_value, value)
        return t

    def test_compiled_inplace_clears_recorded_padding(self) -> None:
        t = self._filled(5.0)

        def scale(x):
            x.mul_(2)
            return x

        torch.compile(scale, fullgraph=True, dynamic=False)(t)
        self.assertIsNone(t.device_tensor_layout().padding_value)
        # The sum must not count the padding, whatever it now holds.
        result = torch.compile(lambda x: x.sum(), dynamic=False)(t)
        torch.testing.assert_close(result.cpu(), torch.tensor(30.0).half())

    def test_compiled_double_inplace_clears_recorded_padding(self) -> None:
        # A second mutation must not reseed the input from its pre-graph
        # padding (5), which the first mutation already overwrote.
        t = self._filled(5.0)

        def scale_shift(x):
            x.mul_(2)
            x.add_(1)
            return x

        torch.compile(scale_shift, fullgraph=True, dynamic=False)(t)
        self.assertIsNone(t.device_tensor_layout().padding_value)
        torch.testing.assert_close(t.cpu(), torch.full((self.SIZE,), 11.0).half())

    def test_dynamic_recompile_of_inplace(self) -> None:
        # The second call recompiles with a symbolic size. The slice check in
        # track_padding_value must not compare it with the concrete
        # valid_elements of the layout ("cannot determine truth value").
        def scale(x):
            x.mul_(2)
            return x

        compiled = torch.compile(scale, fullgraph=True)
        compiled(self._filled(5.0))
        t = torch.empty(64, dtype=torch.float16, device="spyre")
        t.fill_(1.0)
        compiled(t)
        torch.testing.assert_close(t.cpu(), torch.full((64,), 2.0).half())

    def test_dynamic_slice_is_masked(self) -> None:
        # A symbolic slice of the stick dim leaves valid input elements in the
        # output padding; the following sum must still count only the slice.
        def fn(x):
            return (x[:, : x.shape[1] - 3] + 1).sum(dim=1)

        x = torch.ones(4, 64, dtype=torch.float16)
        result = torch.compile(fn, fullgraph=True, dynamic=True)(x.to("spyre"))
        torch.testing.assert_close(result.cpu(), fn(x))

    def test_unmaskable_op_on_unrecorded_padding_warns(self) -> None:
        # Depthwise conv needs zero padding on both inputs and cannot be
        # coordinate-masked. With no recorded padding on either input the
        # analysis must ask for zero padding on each input in turn: the
        # reduction has to name the input it rejects (arg_index), or the retries
        # keep blaming input 0 and end in "Infinite loop". No pass rewrites
        # padding yet, so the result is the "incompatible values" warning.
        from torch_spyre._C import SpyreTensorLayout, get_device_dtype

        dtype = get_device_dtype(torch.float16)
        x = torch.randn(1, 16, 64, dtype=torch.float16)
        w = torch.randn(16, 1, 3, dtype=torch.float16)
        # 16 channels in a 64-wide stick: 48 padding elements per stick.
        x_dev = x.to(
            device_layout=SpyreTensorLayout([64, 1, 1, 64], [1, -1, 1024, 64], dtype)
        )
        w_dev = w.to(
            device_layout=SpyreTensorLayout([3, 1, 1, 64], [1, -1, 3, 3], dtype)
        )
        # An in-place compiled graph leaves the padding unrecorded.
        for t in (x_dev, w_dev):
            torch.compile(lambda a: a.mul_(1), dynamic=False)(t)
            self.assertIsNone(t.device_tensor_layout().padding_value)

        def conv(x, w):
            return torch.conv1d(x, w, None, stride=1, padding=0, groups=16)

        with self.assertLogs("spyre.inductor.padding", level="WARNING") as logs:
            result = torch.compile(conv, dynamic=False)(x_dev, w_dev)
        self.assertTrue(any("incompatible values" in m for m in logs.output))
        # The padding is in fact still zero here, so the result is correct.
        torch.testing.assert_close(result.cpu(), conv(x, w), atol=0.5, rtol=0.1)

    def test_eager_inplace_clears_recorded_padding(self) -> None:
        # In-place eager ops write self through a compiled copy_from_d2d.
        t = self._filled(5.0)
        t.add_(1)
        self.assertIsNone(t.device_tensor_layout().padding_value)
        torch.testing.assert_close(t.sum().cpu(), torch.tensor(18.0).half())

    def test_view_does_not_inherit_recorded_padding(self) -> None:
        t = self._filled(5.0)
        self.assertIsNone(t.view(self.SIZE).device_tensor_layout().padding_value)
        self.assertIsNone(t.detach().device_tensor_layout().padding_value)
        self.assertEqual(t.device_tensor_layout().padding_value, 5.0)

    def test_h2d_records_zero_padding(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float16).to("spyre")
        self.assertEqual(t.device_tensor_layout().padding_value, 0.0)


if __name__ == "__main__":
    unittest.main()
