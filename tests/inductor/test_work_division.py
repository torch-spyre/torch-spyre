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

import math
import unittest
from collections import namedtuple
from contextlib import ExitStack
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import sympy
import torch
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    Pointwise,
    Reduction,
)

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.constants import (
    AVGPOOL2D_OP,
    CONV2D_FWD_OP,
    DEPTHWISE_CONV2D_OP,
)
from torch_spyre._inductor.pass_utils import SchedNodeArg
from torch_spyre._inductor.scratchpad import allocator as allocator_module
from torch_spyre._inductor.scratchpad.allocator import (
    CoOptimizingAllocator,
    CoreDivision,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivisionBuffer
from torch_spyre._inductor.work_division import (
    TensorDep,
    _cost_model_matmul_planner,
    _default_split,
    enumerate_work_division_candidates,
    work_division_context_for_op,
    work_division_splits_are_legal,
    multi_dim_iteration_space_split,
    span_reduction_pass,
)
from torch_spyre._inductor.work_division_constraints import (
    ConstraintResult,
    WorkDivConstraintContext,
    collect_work_division_constraints,
    conv_spatial_blocked_vars,
    coordinate_mask_blocked_vars,
    indirect_access_split_domains,
    keep_by_index_k_split_constraint,
    keep_by_index_pinned_search_space_vars,
    qfp8wt_matmul_k_split_domains,
    qfp8wt_split_domains,
    reduction_window_blocked_vars,
    restickify_padding_blocked_vars,
    topk_split_domains,
)


def _isym(name):
    """Symbol with the (integer, positive) assumptions real Inductor loop
    vars carry -- required for sympy's floor-division to simplify a stick
    coordinate down to a bare symbol instead of leaving it as floor(var)."""
    return Symbol(name, integer=True, positive=True)


def _fixed_tiled_layout(shape, dtype=torch.float16, element_arrangement=None):
    """Build the same kind of physical layout used by real Spyre lowering."""
    size = list(shape)
    stride = [int(s) for s in FlexibleLayout.contiguous_strides(size)]
    within_stick_dim = len(size) - 1
    dim_order = [i for i in range(len(size)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size, stride, dtype, dim_order)
    if element_arrangement is not None:
        device_layout = device_layout.with_element_arrangement(element_arrangement)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _tensor_dep(name, shape, symbols, element_arrangement=None):
    """Build a real TensorDep for a contiguous access over ``symbols``."""
    layout = _fixed_tiled_layout(shape, element_arrangement=element_arrangement)
    index = sympy.Integer(0)
    for sym, stride in zip(symbols, layout.stride):
        index += sym * int(stride)
    dep = MemoryDep(name, index, tuple(symbols), tuple(shape))
    return TensorDep(dep=dep, layout=layout)


def _computed_buffer(shape, name="buf0", reduction_type=None, reduction_ranges=()):
    if reduction_type is not None:
        data = MagicMock(spec=Reduction)
        data.reduction_type = reduction_type
        data.reduction_ranges = list(reduction_ranges)
    else:
        data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = _fixed_tiled_layout(shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name
    return op


def _make_context(
    op,
    output_td,
    input_tds=(),
    it_space=None,
    it_space_adjusted=None,
    stick_vars=None,
    reduction_vars=(),
    committed_splits=None,
):
    it_space = it_space or {}
    return WorkDivConstraintContext(
        op=op,
        it_space=it_space,
        it_space_adjusted=it_space_adjusted
        if it_space_adjusted is not None
        else it_space,
        output_td=output_td,
        input_tds=list(input_tds),
        stick_vars=stick_vars or {},
        reduction_vars=list(reduction_vars),
        committed_splits=committed_splits or {},
    )


class TestMultiDimIterationSpaceSplit(unittest.TestCase):
    def _reduction_split_vars(self, splits, output_dims):
        return {k for k, v in splits.items() if v > 1 and k not in output_dims}

    def test_output_dims_absorb_all_cores(self):
        o0, o1, r0 = Symbol("o0"), Symbol("o1"), Symbol("r0")
        splits = multi_dim_iteration_space_split(
            {o0: 16, o1: 16, r0: 8}, 32, [o0, o1], [r0]
        )
        self.assertLessEqual(len(self._reduction_split_vars(splits, [o0, o1])), 1)
        self.assertEqual(splits[o0] * splits[o1] * splits[r0], 32)

    def test_at_most_one_reduction_dim_split_when_output_dims_small(self):
        # output dims can absorb only 4 cores; 32 total with committed r0=2
        # leaves 4 cores for remaining reduction dims.
        # work_distribution_pass suppresses reduction_dims when a committed split
        # already covers a reduction var, so reduction_dims=[] is passed here.
        o0, r0, r1 = Symbol("o0"), Symbol("r0"), Symbol("r1")
        splits = multi_dim_iteration_space_split(
            {o0: 4, r0: 8, r1: 8},
            32,
            [o0],
            [],  # suppressed: r0 already committed, r1 must not also be split
            min_splits={r0: 2},
        )
        reduction_split = self._reduction_split_vars(splits, [o0])
        self.assertLessEqual(
            len(reduction_split),
            1,
            f"Expected at most 1 reduction dim split, got {reduction_split}",
        )

    def test_no_reduction_dims_uses_greedy_on_all_dims(self):
        o0, o1 = Symbol("o0"), Symbol("o1")
        splits = multi_dim_iteration_space_split({o0: 8, o1: 8}, 32, [o0, o1], [])
        self.assertEqual(splits[o0] * splits[o1], 32)

    def test_single_reduction_dim_split_when_output_exhausted(self):
        o0, r0 = Symbol("o0"), Symbol("r0")
        splits = multi_dim_iteration_space_split({o0: 4, r0: 8}, 32, [o0], [r0])
        self.assertEqual(splits[o0], 4)
        self.assertEqual(splits[r0], 8)

    def test_uses_legal_factor_per_dimension(self):
        o0, o1 = Symbol("o0"), Symbol("o1")
        splits = multi_dim_iteration_space_split(
            {o0: 16, o1: 16},
            32,
            [o0, o1],
            [],
            allowed_splits={o0: frozenset({1, 4}), o1: frozenset({1, 2})},
        )
        self.assertEqual(splits, {o0: 4, o1: 2})

    def test_mandatory_legal_factor_can_grow(self):
        o0, o1 = Symbol("o0"), Symbol("o1")
        splits = multi_dim_iteration_space_split(
            {o0: 16, o1: 16},
            32,
            [o0, o1],
            [],
            allowed_splits={o0: frozenset({4, 8}), o1: frozenset({1, 2})},
        )
        self.assertEqual(splits[o0], 8)

    def test_span_floor_can_use_a_larger_nonmultiple_factor(self):
        o0 = Symbol("o0")
        splits = multi_dim_iteration_space_split(
            {o0: 6},
            3,
            [o0],
            [],
            min_splits={o0: 2},
            allowed_splits={o0: frozenset({1, 2, 3, 6})},
        )
        self.assertEqual(splits[o0], 3)

    def test_rejects_two_mandatory_reduction_splits(self):
        r0, r1 = Symbol("r0"), Symbol("r1")
        with self.assertRaisesRegex(Unsupported, "at most one split reduction"):
            multi_dim_iteration_space_split(
                {r0: 8, r1: 8},
                32,
                [],
                [r0, r1],
                allowed_splits={r0: frozenset({2}), r1: frozenset({2})},
            )


class TestWorkDivisionCandidates(unittest.TestCase):
    def test_candidates_respect_span_floor(self):
        x = _isym("x")
        op = _computed_buffer((8,), name="span_floor")
        op._work_division_span_min_splits = {x: 2}
        output_td = _tensor_dep("span_floor", (8,), (x,))
        with (
            patch(
                "torch_spyre._inductor.work_division.iteration_space_from_op",
                return_value={x: 8},
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_tensor_deps",
                return_value=([], output_td),
            ),
            patch(
                "torch_spyre._inductor.work_division.op_read_writes",
                return_value=MagicMock(writes=[output_td.dep], reads=[]),
            ),
            patch(
                "torch_spyre._inductor.work_division.get_mem_deps_from_rw",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.work_division.adjust_it_space_for_sticks",
                return_value=({x: 8}, {}),
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_work_division_constraints",
                return_value=ConstraintResult(allowed_splits={x: frozenset({1, 2, 4})}),
            ),
        ):
            candidates = enumerate_work_division_candidates(op, 8)
        self.assertEqual(candidates, [{x: 2}, {x: 4}])

    def test_candidates_respect_legal_split_domain(self):
        x = _isym("x")
        op = _computed_buffer((8,), name="candidate_domain")
        output_td = _tensor_dep("candidate_domain", (8,), (x,))
        with (
            patch(
                "torch_spyre._inductor.work_division.iteration_space_from_op",
                return_value={x: 8},
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_tensor_deps",
                return_value=([], output_td),
            ),
            patch(
                "torch_spyre._inductor.work_division.op_read_writes",
                return_value=MagicMock(writes=[output_td.dep], reads=[]),
            ),
            patch(
                "torch_spyre._inductor.work_division.get_mem_deps_from_rw",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.work_division.adjust_it_space_for_sticks",
                return_value=({x: 8}, {}),
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_work_division_constraints",
                return_value=ConstraintResult(allowed_splits={x: frozenset({1, 2})}),
            ),
        ):
            candidates = enumerate_work_division_candidates(op, 8)
        self.assertEqual(candidates, [{x: 1}, {x: 2}])


class _Probe(NamedTuple):
    """One externally supplied split and the verdict each entry point owes it.

    ``committed`` is :func:`work_division_splits_are_legal` -- the op's own
    constraints plus the committed span floors, asked of a split that is
    already committed. ``proposed`` is :meth:`WorkDivisionContext.is_legal`,
    which adds the case's core budget and the ``MAX_SPAN_BYTES`` cap. They
    differ exactly where those two extra rules bite.
    """

    splits: dict
    committed: bool
    proposed: bool


def _by_name(splits):
    """A split keyed by symbol name. sympy symbols are unorderable, so a raw
    symbol-keyed dict makes ``assertEqual``'s diff machinery raise instead of
    printing what differs."""
    return {v.name: factor for v, factor in splits.items()}


class _CandidateCase:
    """One (op, patched context) scenario and the answers it must produce.

    ``candidates`` is the whole enumeration, in ``axes`` order; each entry in
    ``probes`` is a :class:`_Probe`, a split supplied from outside the
    enumeration. Both are literals, so a rule change fails here and is re-read
    rather than re-derived.
    """

    def __init__(
        self,
        name,
        op,
        it_space,
        output_td,
        max_cores,
        axes,
        candidates,
        probes,
        input_tds=(),
        it_space_adjusted=None,
        stick_vars=None,
        constraints=None,
        symbol_meta=None,
    ):
        self.name = name
        self.op = op
        self.it_space = it_space
        self.output_td = output_td
        self.max_cores = max_cores
        self.axes = tuple(axes)
        self.candidates = list(candidates)
        self.probes = list(probes)
        self.input_tds = list(input_tds)
        self.it_space_adjusted = (
            it_space if it_space_adjusted is None else it_space_adjusted
        )
        self.stick_vars = stick_vars or {}
        self.constraints = constraints or ConstraintResult()
        self.symbol_meta = symbol_meta

    def patches(self):
        """Patch the module inputs a work-division context is derived from."""
        rw = MagicMock(
            writes=[self.output_td.dep], reads=[td.dep for td in self.input_tds]
        )
        stack = ExitStack()
        for target, kwargs in [
            ("iteration_space_from_op", {"return_value": self.it_space}),
            (
                "collect_tensor_deps",
                {"return_value": (self.input_tds, self.output_td)},
            ),
            ("op_read_writes", {"return_value": rw}),
            ("get_mem_deps_from_rw", {"return_value": []}),
            (
                "adjust_it_space_for_sticks",
                {"return_value": (self.it_space_adjusted, self.stick_vars)},
            ),
            (
                "collect_work_division_constraints",
                {"return_value": self.constraints},
            ),
        ] + (
            []
            if self.symbol_meta is None
            else [("_collect_symbol_metadata", {"return_value": self.symbol_meta})]
        ):
            stack.enter_context(
                patch(
                    f"torch_spyre._inductor.work_division.{target}",
                    **kwargs,
                )
            )
        return stack


def _candidate_cases():
    """A corpus exercising every branch a candidate is judged by: the three
    factor bases, the core budget, the span cap, the span floor, the
    reduction-count rule, blocked dims and hard domains."""
    x, y, m, k0, k1 = (_isym(n) for n in ("x", "y", "m", "k0", "k1"))

    floor_op = _computed_buffer((8,), name="span_floor")
    floor_op._work_division_span_min_splits = {x: 2}

    red_out = _tensor_dep("reduction_out", (8,), (m,))
    red_in = _tensor_dep("reduction_in", (8, 4, 4), (m, k0, k1))

    blocked_out = _tensor_dep("blocked_out", (8, 16), (x, y))
    stick_out = _tensor_dep("stick_out", (4096, 65536), (x, y))

    return [
        _CandidateCase(
            name="span_floor",
            op=floor_op,
            it_space={x: 8},
            output_td=_tensor_dep("span_floor", (8,), (x,)),
            constraints=ConstraintResult(allowed_splits={x: frozenset({1, 2, 4})}),
            max_cores=8,
            axes=(x,),
            # The floor of 2 removes the unsplit candidate; 8 is outside the
            # allowed domain.
            candidates=[{x: 2}, {x: 4}],
            probes=[
                _Probe(
                    {x: 1}, committed=False, proposed=False
                ),  # below the committed floor of 2
                # Omits the floored axis, so no factor is checked against a
                # domain and only the floor itself can reject.
                _Probe({}, committed=False, proposed=False),
                _Probe({x: 2}, committed=True, proposed=True),
                _Probe({x: 4}, committed=True, proposed=True),
                _Probe(
                    {x: 8}, committed=False, proposed=False
                ),  # outside the allowed domain
            ],
        ),
        _CandidateCase(
            name="two_dims",
            op=_computed_buffer((8, 16), name="two_dims"),
            it_space={x: 8, y: 16},
            output_td=_tensor_dep("two_dims", (8, 16), (x, y)),
            max_cores=32,
            axes=(x, y),
            # The full cross product minus the corner where the product of the
            # factors exceeds 32 cores.
            candidates=[
                {x: 1, y: 1},
                {x: 1, y: 2},
                {x: 1, y: 4},
                {x: 1, y: 8},
                {x: 1, y: 16},
                {x: 2, y: 1},
                {x: 2, y: 2},
                {x: 2, y: 4},
                {x: 2, y: 8},
                {x: 2, y: 16},
                {x: 4, y: 1},
                {x: 4, y: 2},
                {x: 4, y: 4},
                {x: 4, y: 8},
                {x: 8, y: 1},
                {x: 8, y: 2},
                {x: 8, y: 4},
            ],
            # Splits are validated without a core budget, so the 128-core
            # {8, 16} is legal even though it is not an enumerated candidate.
            probes=[
                _Probe({x: 1, y: 1}, committed=True, proposed=True),
                _Probe({x: 4, y: 4}, committed=True, proposed=True),
                _Probe(
                    {x: 8, y: 16}, committed=True, proposed=False
                ),  # 128 cores: over budget, still committable
            ],
        ),
        _CandidateCase(
            name="two_reductions",
            op=_computed_buffer((8,), name="reduction_out"),
            it_space={m: 8, k0: 4, k1: 4},
            output_td=red_out,
            input_tds=[red_in],
            max_cores=32,
            axes=(m, k0, k1),
            # At most one of k0/k1 is ever split, so the (k0, k1) plane keeps
            # only its two axes and not their product.
            candidates=[
                {m: 1, k0: 1, k1: 1},
                {m: 1, k0: 1, k1: 2},
                {m: 1, k0: 1, k1: 4},
                {m: 1, k0: 2, k1: 1},
                {m: 1, k0: 4, k1: 1},
                {m: 2, k0: 1, k1: 1},
                {m: 4, k0: 1, k1: 1},
                {m: 8, k0: 1, k1: 1},
            ],
            probes=[
                _Probe({m: 2, k0: 1, k1: 1}, committed=True, proposed=True),
                _Probe({m: 1, k0: 4, k1: 1}, committed=True, proposed=True),
                _Probe(
                    {m: 1, k0: 2, k1: 2}, committed=False, proposed=False
                ),  # two split reduction dims
            ],
        ),
        _CandidateCase(
            name="blocked_dim",
            op=_computed_buffer((8, 16), name="blocked_out"),
            it_space={x: 8, y: 16},
            output_td=blocked_out,
            constraints=ConstraintResult(
                blocked={y}, allowed_splits={x: frozenset({1, 2, 8})}
            ),
            max_cores=32,
            axes=(x, y),
            # y is blocked, so it stays at 1 throughout; x is held to its
            # allowed domain.
            candidates=[{x: 1, y: 1}, {x: 2, y: 1}, {x: 8, y: 1}],
            probes=[
                _Probe({x: 2, y: 1}, committed=True, proposed=True),
                _Probe(
                    {x: 2, y: 2}, committed=False, proposed=False
                ),  # splits a blocked dim
                _Probe(
                    {x: 4, y: 1}, committed=False, proposed=False
                ),  # outside x's allowed domain
            ],
        ),
        _CandidateCase(
            name="stick_basis_and_span_cap",
            op=_computed_buffer((4096, 65536), name="stick_out"),
            it_space={x: 4096, y: 65536},
            it_space_adjusted={x: 4096, y: 1024},
            stick_vars={y: 64},
            output_td=stick_out,
            max_cores=32,
            axes=(x, y),
            # y unsplit would leave a per-core span over MAX_SPAN_BYTES, so
            # every candidate splits it at least twice.
            candidates=[
                {x: 1, y: 2},
                {x: 1, y: 4},
                {x: 1, y: 8},
                {x: 1, y: 16},
                {x: 1, y: 32},
                {x: 2, y: 2},
                {x: 2, y: 4},
                {x: 2, y: 8},
                {x: 2, y: 16},
                {x: 4, y: 2},
                {x: 4, y: 4},
                {x: 4, y: 8},
                {x: 8, y: 2},
                {x: 8, y: 4},
                {x: 16, y: 2},
            ],
            # The span cap is not one of an op's own constraints, so a
            # committed {x: 1, y: 1} stays legal despite being excluded above.
            probes=[
                _Probe(
                    {x: 1, y: 1}, committed=True, proposed=False
                ),  # over the span cap, still committable
                _Probe({x: 4, y: 1}, committed=True, proposed=False),  # likewise
                _Probe({x: 8, y: 4}, committed=True, proposed=True),
            ],
        ),
        _CandidateCase(
            name="symbolic_granularity",
            op=_computed_buffer((1024,), name="symbolic_out"),
            it_space={x: 1024},
            output_td=_tensor_dep("symbolic_out", (1024,), (x,)),
            symbol_meta={x: (1024, 256)},
            max_cores=32,
            axes=(x,),
            # Factors come off the granularity, capped by the core budget.
            candidates=[{x: 1}, {x: 2}, {x: 4}, {x: 8}, {x: 16}, {x: 32}],
            probes=[
                _Probe({x: 1}, committed=True, proposed=True),
                _Probe({x: 4}, committed=True, proposed=True),
                _Probe({x: 8}, committed=True, proposed=True),
            ],
        ),
    ]


class TestWorkDivisionContextAnswers(unittest.TestCase):
    """What the candidate seam answers, pinned to literals: the enumeration a
    core budget yields, the verdict an already-committed split gets, and the
    context's agreement with both."""

    def test_candidate_lists_are_the_expected_enumeration(self):
        rejected = []
        for case in _candidate_cases():
            with self.subTest(case.name):
                with case.patches():
                    actual = enumerate_work_division_candidates(case.op, case.max_cores)
                    ctx = work_division_context_for_op(case.op, case.max_cores)
                    domains = [ctx.factor_domain(v) for v in case.axes]
                self.assertEqual(
                    [_by_name(c) for c in actual],
                    [_by_name(c) for c in case.candidates],
                )
                self.assertTrue(
                    case.candidates, "case would prove nothing: no candidates"
                )
                rejected.append(
                    len(case.candidates) < math.prod(len(d) for d in domains)
                )
        # At least one case must exercise the whole-split predicate rather than
        # riding on the per-axis domains alone.
        self.assertTrue(any(rejected))

    def test_legality_verdicts_are_the_expected_verdicts(self):
        """Both entry points, on splits supplied from outside the enumeration:
        ``is_legal`` is asked directly, so a rule it forgets cannot hide behind
        candidates pre-filtered through :meth:`factor_domain`."""
        seen = set()
        for case in _candidate_cases():
            with self.subTest(case.name):
                for probe in case.probes:
                    with case.patches():
                        committed = work_division_splits_are_legal(
                            case.op, probe.splits
                        )
                        proposed = work_division_context_for_op(
                            case.op, case.max_cores
                        ).is_legal(probe.splits)
                    self.assertEqual(
                        (committed, proposed),
                        (probe.committed, probe.proposed),
                        _by_name(probe.splits),
                    )
                    seen.add((probe.committed, probe.proposed))
        # Agreeing everywhere, or agreeing with each other everywhere, would
        # prove nothing about the rules or about the difference between them.
        self.assertEqual(
            seen, {(True, True), (False, False), (True, False)}, sorted(seen)
        )

    def test_is_legal_rejects_splits_no_factor_domain_would_produce(self):
        """``is_legal`` asked about malformed splits, which only a caller that
        proposes rather than enumerates can supply. ``two_dims`` has no hard
        allowed-split domains, so the op's own domains constrain nothing here
        and the axis's factor domain is the only thing that can reject."""
        case = next(c for c in _candidate_cases() if c.name == "two_dims")
        x, y = case.axes
        foreign = _isym("not_an_axis")
        with case.patches():
            ctx = work_division_context_for_op(case.op, case.max_cores)
            self.assertEqual(ctx.constraints.allowed_splits, {})
            for splits, legal, why in [
                ({x: 4, y: 4}, True, "divisors of both axes"),
                ({x: 4}, True, "an omitted axis is unsplit, not illegal"),
                ({x: 3, y: 1}, False, "3 does not divide 8"),
                ({x: 0, y: 1}, False, "zero would divide by zero downstream"),
                ({x: -2, y: 1}, False, "negative factor"),
                ({foreign: 2}, False, "axis of no iteration space"),
            ]:
                with self.subTest(why):
                    self.assertEqual(ctx.is_legal(splits), legal, _by_name(splits))

    def test_context_answers_match_the_enumeration(self):
        """The seam itself: the context's axis order is the one a candidate is
        keyed by, and every enumerated candidate is one the context calls legal
        and whose factors come from its own per-axis domains."""
        for case in _candidate_cases():
            with self.subTest(case.name):
                with case.patches():
                    ctx = work_division_context_for_op(case.op, case.max_cores)
                    candidates = enumerate_work_division_candidates(
                        case.op, case.max_cores
                    )
                    self.assertEqual(ctx.axes, list(case.axes))
                    domains = {v: ctx.factor_domain(v) for v in ctx.axes}
                    self.assertTrue(all(ctx.is_legal(c) for c in candidates))
                    self.assertTrue(
                        all(
                            split in domains[v]
                            for c in candidates
                            for v, split in c.items()
                        )
                    )


class TestWorkDivisionSplitLegality(unittest.TestCase):
    def test_cpu_computed_buffer_is_not_constrained(self):
        op = _computed_buffer((8,), name="cpu_buf")
        op.layout = FixedLayout("cpu", torch.float16, [8], [1])
        self.assertTrue(work_division_splits_are_legal(op, {}))

    def test_symbol_keyed_splits_obey_allowed_domain(self):
        x = _isym("x")
        op = _computed_buffer((8,), name="domain")
        output_td = _tensor_dep("domain", (8,), (x,))
        rw = MagicMock(writes=[output_td.dep], reads=[])
        with (
            patch(
                "torch_spyre._inductor.work_division.iteration_space_from_op",
                return_value={x: 8},
            ),
            patch(
                "torch_spyre._inductor.work_division.op_read_writes", return_value=rw
            ),
            patch(
                "torch_spyre._inductor.work_division.get_mem_deps_from_rw",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_tensor_deps",
                return_value=([], output_td),
            ),
            patch(
                "torch_spyre._inductor.work_division.adjust_it_space_for_sticks",
                return_value=({x: 8}, {}),
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_work_division_constraints",
                return_value=ConstraintResult(allowed_splits={x: frozenset({2})}),
            ),
        ):
            self.assertTrue(work_division_splits_are_legal(op, {x: 2}))
            self.assertFalse(work_division_splits_are_legal(op, {x: 1}))

    def test_rejects_two_split_reduction_axes(self):
        o, r0, r1 = (_isym(name) for name in ("o", "r0", "r1"))
        op = _computed_buffer((8,), name="two_reductions")
        output_td = _tensor_dep("two_reductions", (8,), (o,))
        rw = MagicMock(writes=[output_td.dep], reads=[])
        with (
            patch(
                "torch_spyre._inductor.work_division.iteration_space_from_op",
                return_value={o: 8, r0: 8, r1: 8},
            ),
            patch(
                "torch_spyre._inductor.work_division.op_read_writes", return_value=rw
            ),
            patch(
                "torch_spyre._inductor.work_division.get_mem_deps_from_rw",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_tensor_deps",
                return_value=([], output_td),
            ),
            patch(
                "torch_spyre._inductor.work_division.adjust_it_space_for_sticks",
                return_value=({o: 8, r0: 8, r1: 8}, {}),
            ),
            patch(
                "torch_spyre._inductor.work_division.collect_work_division_constraints",
                return_value=ConstraintResult(),
            ),
        ):
            self.assertFalse(work_division_splits_are_legal(op, {r0: 2, r1: 2}))

    def test_uses_input_layout_override_for_qfp8wt_constraint(self):
        b, m, n = _isym("b"), _isym("m"), _isym("n")
        op = _computed_buffer((4, 8, 128), name="override_output")
        output_dep = MemoryDep(
            "override_output", b * 1024 + m * 128 + n, (b, m, n), (4, 8, 128)
        )
        kernel_dep = MemoryDep(
            "override_kernel", b * 1024 + n * 8 + m, (b, n, m), (4, 128, 8)
        )
        raw_args = [
            SchedNodeArg(
                MemoryDep(
                    "override_input", b * 1024 + m * 128 + n, (b, m, n), (4, 8, 128)
                ),
                _fixed_tiled_layout((4, 8, 128)),
            ),
            SchedNodeArg(kernel_dep, _fixed_tiled_layout((4, 128, 8))),
        ]
        override_layout = _fixed_tiled_layout(
            (4, 128, 8), element_arrangement=ElementArrangement.QFP8WT
        )
        constrained_var = next(
            iter(TensorDep(kernel_dep, override_layout).device_coords[-2].free_symbols)
        )
        op._input_layout_overrides = {"override_kernel": override_layout}
        rw = MagicMock(writes=[output_dep], reads=[raw_args[0].dep, kernel_dep])

        with (
            patch(
                "torch_spyre._inductor.work_division.iteration_space_from_op",
                return_value={b: 4, m: 8, n: 128},
            ),
            patch(
                "torch_spyre._inductor.work_division.op_read_writes", return_value=rw
            ),
            patch(
                "torch_spyre._inductor.work_division.get_mem_deps_from_rw",
                return_value=raw_args,
            ),
            patch(
                "torch_spyre._inductor.work_division.adjust_it_space_for_sticks",
                return_value=({b: 4, m: 8, n: 128}, {}),
            ),
        ):
            self.assertFalse(work_division_splits_are_legal(op, {constrained_var: 2}))
            del op._input_layout_overrides
            self.assertTrue(work_division_splits_are_legal(op, {constrained_var: 2}))


class TestKeepByIndexConstraints(unittest.TestCase):
    def test_k_is_minimally_split_and_search_axis_is_unsplit(self):
        batch, search, k = (_isym(name) for name in ("batch", "search", "k"))
        op = _computed_buffer(
            (8, 64), name="keep_by_index", reduction_type="keepbyindex"
        )
        output_td = _tensor_dep("keep_by_index", (8, 64), (batch, search))
        ctx = _make_context(
            op,
            output_td,
            input_tds=[
                _tensor_dep("values", (8, 64), (batch, search)),
                _tensor_dep("indices", (8, 8), (batch, k)),
            ],
            it_space={batch: 8, search: 64, k: 8},
            reduction_vars=(k,),
        )
        rw = MagicMock(writes=[output_td.dep])

        with patch(
            "torch_spyre._inductor.work_division_constraints.op_read_writes",
            return_value=rw,
        ):
            k_result = keep_by_index_k_split_constraint(ctx)
            search_result = keep_by_index_pinned_search_space_vars(ctx)

        self.assertEqual(k_result.allowed_splits, {k: frozenset({2})})
        self.assertEqual(search_result.allowed_splits, {search: frozenset({1})})

    def test_only_one_search_axis_is_pinned_when_indices_broadcast_batch(self):
        batch, search, k = (_isym(name) for name in ("batch", "search", "k"))
        op = _computed_buffer(
            (8, 64), name="keep_by_index", reduction_type="keepbyindex"
        )
        output_td = _tensor_dep("keep_by_index", (8, 64), (batch, search))
        ctx = _make_context(
            op,
            output_td,
            input_tds=[
                _tensor_dep("values", (8, 64), (batch, search)),
                _tensor_dep("indices", (8,), (k,)),
            ],
            it_space={batch: 8, search: 64, k: 8},
            reduction_vars=(k,),
        )
        rw = MagicMock(writes=[output_td.dep])

        with patch(
            "torch_spyre._inductor.work_division_constraints.op_read_writes",
            return_value=rw,
        ):
            result = keep_by_index_pinned_search_space_vars(ctx)

        self.assertEqual(result.allowed_splits, {batch: frozenset({1})})


class TestCostModelConstraints(unittest.TestCase):
    def test_restricted_batch_dim_stays_unsplit(self):
        batch, m, n, k = (_isym(name) for name in ("batch", "m", "n", "k"))
        op = _computed_buffer(
            (4, 64, 256),
            name="matmul_out",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("matmul_out", (4, 64, 256), (batch, m, n))
        input_tds = [
            _tensor_dep("lhs", (4, 64, 128), (batch, m, k)),
            _tensor_dep("rhs", (4, 128, 256), (batch, k, n)),
        ]
        it_space = {batch: 4, m: 64, n: 4, k: 2}

        def prefer_batch_split(batch_axis, *_args, **_kwargs):
            return 0 if batch_axis[1] > 1 else 1

        with patch(
            "torch_spyre._inductor.work_division._matmul_split_cost",
            side_effect=prefer_batch_split,
        ):
            unrestricted = _cost_model_matmul_planner(
                op,
                {sym: 1 for sym in it_space},
                it_space,
                output_td,
                {n: 64, k: 64},
                {},
                32,
                input_tds,
                set(),
                {},
            )
            restricted = _cost_model_matmul_planner(
                op,
                {sym: 1 for sym in it_space},
                it_space,
                output_td,
                {n: 64, k: 64},
                {},
                32,
                input_tds,
                {batch},
                {},
            )

        self.assertGreater(unrestricted[batch], 1)
        self.assertEqual(restricted[batch], 1)


class TestCoordinateMaskBlockedVars(unittest.TestCase):
    """coordinate_mask_blocked_vars only reads reduction_vars/stick_vars/it_space,
    so output_td/op are irrelevant here and stand in with a placeholder."""

    _PLACEHOLDER_OP = _computed_buffer((128,), name="placeholder_buf")
    _PLACEHOLDER_TD = _tensor_dep("placeholder_buf", (128,), (_isym("_placeholder"),))

    def test_padded_stick_aligned_reduction_dim_is_blocked(self):
        r0 = _isym("r0")
        ctx = _make_context(
            self._PLACEHOLDER_OP,
            self._PLACEHOLDER_TD,
            it_space={r0: 10},
            stick_vars={r0: 64},
            reduction_vars=[r0],
        )
        result = coordinate_mask_blocked_vars(ctx)
        self.assertEqual(result.blocked, {r0})

    def test_stick_aligned_reduction_dim_is_not_blocked(self):
        r0 = _isym("r0")
        ctx = _make_context(
            self._PLACEHOLDER_OP,
            self._PLACEHOLDER_TD,
            it_space={r0: 128},
            stick_vars={r0: 64},
            reduction_vars=[r0],
        )
        result = coordinate_mask_blocked_vars(ctx)
        self.assertEqual(result.blocked, set())

    def test_non_stick_var_is_not_blocked(self):
        r0 = _isym("r0")
        ctx = _make_context(
            self._PLACEHOLDER_OP,
            self._PLACEHOLDER_TD,
            it_space={r0: 10},
            stick_vars={},
            reduction_vars=[r0],
        )
        result = coordinate_mask_blocked_vars(ctx)
        self.assertEqual(result.blocked, set())


class TestConvSpatialBlockedVars(unittest.TestCase):
    _PATCH_TARGET = "torch_spyre._inductor.work_division_constraints.op_read_writes"
    _PLACEHOLDER_TD = _tensor_dep("conv_placeholder", (128,), (_isym("_conv"),))

    def _context(self, stride):
        mb, out, i, j = (_isym(name) for name in ("mb", "out", "i", "j"))
        op = _computed_buffer((2, 3, 8, 16), name="strided_conv")
        op.data.op_info = {
            "conv_params": {"stride_i": stride[0], "stride_j": stride[1]}
        }
        return (
            _make_context(
                op,
                self._PLACEHOLDER_TD,
                it_space={mb: 2, out: 3, i: 8, j: 16},
            ),
            i,
            j,
        )

    def test_blocks_spatial_dims_for_strided_conv(self):
        ctx, i, j = self._context((2, 1))
        rw = MagicMock()
        # Inductor stores ranges in OrderedSet, which does not support slices.
        rw.writes = [MagicMock(ranges=(_isym("mb"), _isym("out"), i, j))]
        with patch(self._PATCH_TARGET, return_value=rw):
            self.assertEqual(conv_spatial_blocked_vars(ctx).blocked, {i, j})

    def test_allows_spatial_dims_for_unstrided_conv(self):
        ctx, _, _ = self._context((1, 1))
        self.assertEqual(conv_spatial_blocked_vars(ctx).blocked, set())

    def test_span_commit_conflicting_with_spatial_block_raises_unsupported(self):
        ctx, i, j = self._context((2, 1))
        ctx.committed_splits = {i: 2}
        rw = MagicMock()
        rw.writes = [MagicMock(ranges=(_isym("mb"), _isym("out"), i, j))]
        with patch(self._PATCH_TARGET, return_value=rw):
            with self.assertRaisesRegex(Unsupported, "blocked dim"):
                collect_work_division_constraints(ctx)

    def test_blocked_spatial_dims_are_not_distributed(self):
        mb, out, i, j = (_isym(name) for name in ("mb", "out", "i", "j"))
        output_td = _tensor_dep("conv_out", (2, 32, 32, 32), (mb, out, i, j))
        splits, output_dims, _ = _default_split(
            _computed_buffer((2, 32, 32, 32), name="conv_out"),
            {mb: 2, out: 32, i: 32, j: 32},
            output_td,
            {},
            32,
            {},
            {i, j},
            {},
        )
        self.assertNotIn(i, output_dims)
        self.assertNotIn(j, output_dims)
        self.assertEqual(splits[i], 1)
        self.assertEqual(splits[j], 1)


class TestFinalMappingConstraints(unittest.TestCase):
    _PLACEHOLDER_TD = _tensor_dep("mapping_placeholder", (128,), (_isym("_mapping"),))

    def test_pool_window_dims_are_unsplit(self):
        ki, kj = (_isym(name) for name in ("ki", "kj"))
        op = _computed_buffer(
            (8,),
            name="pool",
            reduction_type=AVGPOOL2D_OP,
            reduction_ranges=(3, 3),
        )
        result = reduction_window_blocked_vars(
            _make_context(
                op,
                self._PLACEHOLDER_TD,
                reduction_vars=[ki, kj],
            )
        )

        self.assertEqual(result.blocked, {ki, kj})

    def test_conv_only_blocks_nontrivial_kernel_dims(self):
        channel, ki, kj = (_isym(name) for name in ("channel", "ki", "kj"))
        op = _computed_buffer(
            (8,),
            name="conv",
            reduction_type=CONV2D_FWD_OP,
            reduction_ranges=(64, 3, 1),
        )
        op.data.op_info = {"conv_params": {"kernel_h": 3, "kernel_w": 1}}
        result = reduction_window_blocked_vars(
            _make_context(
                op,
                self._PLACEHOLDER_TD,
                reduction_vars=[channel, ki],
            )
        )

        self.assertEqual(result.blocked, {ki})

    def test_depthwise_conv_does_not_block_trailing_group_dim(self):
        kh, kw, group = (_isym(name) for name in ("kh", "kw", "group"))
        op = _computed_buffer(
            (8,),
            name="depthwise_conv",
            reduction_type=DEPTHWISE_CONV2D_OP,
            reduction_ranges=(3, 3, 4),
        )
        result = reduction_window_blocked_vars(
            _make_context(
                op,
                self._PLACEHOLDER_TD,
                reduction_vars=[kh, kw, group],
            )
        )

        self.assertEqual(result.blocked, {kh, kw})

    def test_conv_spatial_and_window_blocks_compose(self):
        mb, out, i, j, channel, ki = (
            _isym(name) for name in ("mb", "out", "i", "j", "channel", "ki")
        )
        op = _computed_buffer(
            (2, 3, 8, 16),
            name="strided_windowed_conv",
            reduction_type=CONV2D_FWD_OP,
            reduction_ranges=(64, 3),
        )
        op.data.op_info = {
            "conv_params": {
                "stride_i": 2,
                "stride_j": 1,
                "kernel_h": 3,
                "kernel_w": 1,
            }
        }
        ctx = _make_context(
            op,
            self._PLACEHOLDER_TD,
            it_space={mb: 2, out: 3, i: 8, j: 16, channel: 64, ki: 3},
            reduction_vars=[channel, ki],
        )
        rw = MagicMock()
        rw.writes = [MagicMock(ranges=(mb, out, i, j))]

        with patch(TestConvSpatialBlockedVars._PATCH_TARGET, return_value=rw):
            result = collect_work_division_constraints(ctx)

        self.assertEqual(result.blocked, {i, j, ki})

    def test_window_block_conflicting_with_span_commit_raises_unsupported(self):
        ki, kj = (_isym(name) for name in ("ki", "kj"))
        op = _computed_buffer(
            (8,),
            name="pool_with_forced_window_split",
            reduction_type=AVGPOOL2D_OP,
            reduction_ranges=(3, 3),
        )
        ctx = _make_context(
            op,
            self._PLACEHOLDER_TD,
            reduction_vars=[ki, kj],
            committed_splits={ki: 2},
        )

        with self.assertRaisesRegex(Unsupported, "reduction_window_blocked_vars"):
            collect_work_division_constraints(ctx)

    def test_unaligned_restickify_stick_dim_is_unsplit(self):
        old_stick, new_stick = (_isym(name) for name in ("old_stick", "new_stick"))
        op = _computed_buffer((96,), name="restickify")
        input_td = MagicMock()
        input_td.device_coords = [
            sympy.floor(old_stick / 64),
            sympy.Mod(old_stick, 64),
        ]
        output_td = MagicMock()
        output_td.device_coords = [
            sympy.floor(new_stick / 64),
            sympy.Mod(new_stick, 64),
        ]
        result = restickify_padding_blocked_vars(
            _make_context(
                op,
                output_td,
                input_tds=[input_td],
                it_space={old_stick: 96, new_stick: 128},
                stick_vars={old_stick: 64, new_stick: 64},
            )
        )

        self.assertEqual(result.blocked, {old_stick})


class TestQfp8wtConstraints(unittest.TestCase):
    def test_output_second_stick_coord_restricted_for_qfp8wt_output(self):
        b, m, n = _isym("b"), _isym("m"), _isym("n")
        op = _computed_buffer((4, 8, 128), name="qfp8_out")
        output_td = _tensor_dep(
            "qfp8_out",
            (4, 8, 128),
            (b, m, n),
            element_arrangement=ElementArrangement.QFP8WT,
        )
        ctx = _make_context(op, output_td, it_space={b: 4, m: 8, n: 128})
        result = qfp8wt_split_domains(ctx)
        restricted_vars = set(output_td.device_coords[-2].free_symbols)
        self.assertTrue(restricted_vars)
        for v in restricted_vars:
            self.assertEqual(result.allowed_splits[v], frozenset({1}))

    def test_standard_output_yields_no_pins(self):
        b, m, n = _isym("b"), _isym("m"), _isym("n")
        op = _computed_buffer((4, 8, 128), name="std_out")
        output_td = _tensor_dep("std_out", (4, 8, 128), (b, m, n))
        ctx = _make_context(op, output_td, it_space={b: 4, m: 8, n: 128})
        result = qfp8wt_split_domains(ctx)
        self.assertEqual(result.allowed_splits, {})

    def test_matmul_k_restricted_for_batchmatmulfp8_with_qfp8wt_kernel(self):
        from torch_spyre._inductor.constants import BATCH_MATMUL_FP8_OP

        b, m, n, k = _isym("b"), _isym("m"), _isym("n"), _isym("k")
        op = _computed_buffer(
            (4, 8, 128),
            name="mm_out",
            reduction_type=BATCH_MATMUL_FP8_OP,
            reduction_ranges=(64,),
        )
        output_td = _tensor_dep("mm_out", (4, 8, 128), (b, m, n))
        kernel_td = _tensor_dep(
            "kernel",
            (4, 128, 64),
            (b, n, k),
            element_arrangement=ElementArrangement.QFP8WT,
        )
        ctx = _make_context(
            op,
            output_td,
            input_tds=[
                _tensor_dep("act", (4, 8, 64), (b, m, k)),
                kernel_td,
            ],
            it_space={b: 4, m: 8, n: 128, k: 64},
            reduction_vars=[k],
        )
        result = qfp8wt_matmul_k_split_domains(ctx)
        self.assertEqual(result.allowed_splits, {k: frozenset({1})})

    def test_matmul_k_unrestricted_for_plain_batchmatmul(self):
        from torch_spyre._inductor.constants import BATCH_MATMUL_OP

        b, m, n, k = _isym("b"), _isym("m"), _isym("n"), _isym("k")
        op = _computed_buffer(
            (4, 8, 128),
            name="mm_out2",
            reduction_type=BATCH_MATMUL_OP,
            reduction_ranges=(64,),
        )
        output_td = _tensor_dep("mm_out2", (4, 8, 128), (b, m, n))
        ctx = _make_context(
            op,
            output_td,
            input_tds=[
                _tensor_dep("act2", (4, 8, 64), (b, m, k)),
                _tensor_dep("kernel2", (4, 128, 64), (b, n, k)),
            ],
            it_space={b: 4, m: 8, n: 128, k: 64},
            reduction_vars=[k],
        )
        result = qfp8wt_matmul_k_split_domains(ctx)
        self.assertEqual(result.allowed_splits, {})


class TestCollectWorkDivisionConstraints(unittest.TestCase):
    _PATCH_TARGET = "torch_spyre._inductor.work_division_constraints"
    _PLACEHOLDER_OP = _computed_buffer((128,), name="constraint_placeholder_buf")
    _PLACEHOLDER_TD = _tensor_dep(
        "constraint_placeholder_buf", (128,), (_isym("_placeholder"),)
    )

    def _collect(self, results, **context_kwargs):
        rules = (
            "coordinate_mask_blocked_vars",
            "conv_spatial_blocked_vars",
            "qfp8wt_split_domains",
            "qfp8wt_matmul_k_split_domains",
            "indirect_access_split_domains",
        )
        with ExitStack() as stack:
            for rule, result in zip(rules, results):
                stack.enter_context(
                    patch(
                        f"{self._PATCH_TARGET}.{rule}",
                        lambda _ctx, result=result: result,
                    )
                )
            return collect_work_division_constraints(
                _make_context(
                    self._PLACEHOLDER_OP, self._PLACEHOLDER_TD, **context_kwargs
                )
            )

    def test_blocked_var_with_committed_split_raises_unsupported(self):
        r0 = _isym("r0")
        with self.assertRaisesRegex(Unsupported, "blocked dim"):
            self._collect(
                (
                    ConstraintResult(blocked={r0}),
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(),
                ),
                committed_splits={r0: 2},
            )

    def test_intersects_legal_split_domains(self):
        r0 = _isym("r0")
        result = self._collect(
            (
                ConstraintResult(allowed_splits={r0: frozenset({1, 2, 4})}),
                ConstraintResult(allowed_splits={r0: frozenset({2, 4, 8})}),
                ConstraintResult(),
                ConstraintResult(),
                ConstraintResult(),
            )
        )
        self.assertEqual(result.allowed_splits, {r0: frozenset({2, 4})})

    def test_empty_legal_split_domain_intersection_raises_unsupported(self):
        r0 = _isym("r0")
        with self.assertRaisesRegex(Unsupported, "conflicting legal split domains"):
            self._collect(
                (
                    ConstraintResult(allowed_splits={r0: frozenset({2})}),
                    ConstraintResult(allowed_splits={r0: frozenset({1})}),
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(),
                )
            )

    def test_qfp8wt_k_pin_conflicting_with_span_split_raises_unsupported(self):
        k = _isym("k")
        with self.assertRaisesRegex(Unsupported, "hardware memory-span limit"):
            self._collect(
                (
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(allowed_splits={k: frozenset({1})}),
                    ConstraintResult(),
                ),
                committed_splits={k: 2},
            )

    def test_indirect_pin_conflicting_with_span_split_raises_unsupported(self):
        i0 = _isym("i0")
        with self.assertRaisesRegex(Unsupported, "hardware memory-span limit"):
            self._collect(
                (
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(),
                    ConstraintResult(allowed_splits={i0: frozenset({1})}),
                ),
                committed_splits={i0: 2},
            )

    def test_combines_non_conflicting_rules(self):
        r0, r1, r2, r3 = (_isym(f"r{i}") for i in range(4))
        result = self._collect(
            (
                ConstraintResult(blocked={r0}, allowed_splits={r2: frozenset({1})}),
                ConstraintResult(blocked={r1}, allowed_splits={r3: frozenset({2})}),
                ConstraintResult(blocked={r1}),
                ConstraintResult(allowed_splits={r2: frozenset({1})}),
                ConstraintResult(),
            )
        )
        self.assertEqual(result.blocked, {r0, r1})
        self.assertEqual(
            result.allowed_splits, {r2: frozenset({1}), r3: frozenset({2})}
        )


class TestSpanReductionConstraints(unittest.TestCase):
    _PATCH_TARGET = "torch_spyre._inductor.work_division"

    def test_span_search_excludes_blocked_dimensions(self):
        o, r0, r1 = (_isym(name) for name in ("o", "r0", "r1"))
        op = _computed_buffer((8,), name="indirect_reduction")
        output_td = _tensor_dep("indirect_reduction", (8,), (o,))
        with (
            patch(
                f"{self._PATCH_TARGET}.iteration_space_from_op",
                return_value={o: 8, r0: 8, r1: 8},
            ),
            patch(
                f"{self._PATCH_TARGET}.collect_tensor_deps",
                return_value=([], output_td),
            ),
            patch(
                f"{self._PATCH_TARGET}.adjust_it_space_for_sticks",
                return_value=({o: 8, r0: 8, r1: 8}, {}),
            ),
            patch(
                f"{self._PATCH_TARGET}.must_split_vars", return_value={}
            ) as must_split,
            patch(
                f"{self._PATCH_TARGET}.collect_work_division_constraints",
                return_value=ConstraintResult(blocked={r0, r1}),
            ),
            patch(f"{self._PATCH_TARGET}.apply_splits") as apply_splits,
        ):
            span_reduction_pass(op, [], 32)
        self.assertEqual(apply_splits.call_args.args[1], {})
        self.assertEqual(must_split.call_args.args[-1], {r0, r1})


_FakeView = namedtuple("_FakeView", "work_slice_dims")


class TestResidencyEdgeMatching(unittest.TestCase):
    """The compatibility seam: the pairs :class:`ResidencyEdge` admits for a
    corpus of policy cases, and the pairwise :meth:`ResidencyEdge.compatible`
    a generator would call agreeing with the table it replaces."""

    def setUp(self):
        x, y = _isym("x"), _isym("y")
        self.view_a = _FakeView(((0, 4),))
        self.view_b = _FakeView(((0, 2),))
        self.view_wide = _FakeView(((0, 2), (1, 2)))

        def _div(splits, reduction=None):
            return CoreDivision(
                output_splits=dict(splits), reduction_splits=dict(reduction or {})
            )

        # Consumer: a 4-core slicing, a 2-core one, an 8-core one that slices
        # the buffer the same way as the first (the stale-LX case), and a
        # 4-core one slicing two device dims -- the only consumer the wide
        # parent candidate could pair with, so the matmul guard is what
        # rejects it rather than a view mismatch.
        self.consumer_divs = [
            _div({x: 4}),
            _div({x: 2}),
            _div({x: 8}),
            _div({x: 2, y: 2}),
        ]
        self.consumer_views = [
            self.view_a,
            self.view_b,
            self.view_a,
            self.view_wide,
        ]
        # Every parent offers the same two candidates; what differs is the
        # policy each one trips.
        self.parent_divs = [_div({x: 4}), _div({x: 2})]

        self.parents = {
            # Plain match, plus the cores_used guard on consumer index 2.
            "plain": ([self.view_a, self.view_b], [False, False], [True, True], False),
            # A partial-reduction write can't host a readable residency.
            "partial": ([self.view_a, self.view_b], [True, False], [True, True], False),
            # An unrepresentable slicing is never pinned on.
            "unrepr": (
                [self.view_a, self.view_b],
                [False, False],
                [False, True],
                False,
            ),
            # A matmul split across >1 device dim: only the primary split is
            # carried, so the wide candidate drops out and the narrow stays.
            "matmul": (
                [self.view_wide, self.view_b],
                [False, False],
                [True, True],
                True,
            ),
        }
        self.op_by_name = {
            name: self._op(name) for name in list(self.parents) + ["spilled", "clone"]
        }
        self.consumer_op = self._op("consumer")
        self.divisions = {
            name: self.parent_divs for name in list(self.parents) + ["spilled", "clone"]
        }
        self.residency = dict.fromkeys(self.op_by_name, None)
        self.residency["spilled"] = "no room"
        self.parent_names = list(self.parents) + ["spilled", "clone", "not_a_buffer"]
        self.rw = {
            self.consumer_op: MagicMock(
                reads=[
                    MemoryDep(name, x, (x,), (8,))
                    for name in list(self.parents) + ["spilled", "clone"]
                ],
                writes=[MemoryDep("consumer", x, (x,), (8,))],
            ),
            **{
                op: MagicMock(
                    writes=[MemoryDep(name, x, (x,), (8,))],
                    reads=[MemoryDep("src", x, (x,), (8,))],
                )
                for name, op in self.op_by_name.items()
            },
        }
        # A clone whose write carries a dim its reads do not: it broadcasts,
        # so no per-core slice of it is produced core-locally.
        self.rw[self.op_by_name["clone"]] = MagicMock(
            writes=[MemoryDep("clone", 16 * x + y, (x, y), (8, 16))],
            reads=[MemoryDep("src", x, (x,), (8,))],
        )

    @staticmethod
    def _op(name):
        op = MagicMock(spec=ComputedBuffer)
        op.get_name.return_value = name
        return op

    def _view_for_div(self, op, dep, buf_name, division, prep_cache):
        name = op.get_name()
        if name == "consumer":
            index = self.consumer_divs.index(division)
            return (self.consumer_views[index], False, True)
        views, partial, repr_ok, _matmul = self.parents.get(
            name, ([self.view_a, self.view_b], [False, False], [True, True], False)
        )
        index = self.parent_divs.index(division)
        return (views[index], partial[index], repr_ok[index])

    def _patches(self):
        stack = ExitStack()
        for target, kwargs in [
            ("_view_for_div", {"side_effect": self._view_for_div}),
            ("op_read_writes", {"side_effect": lambda op: self.rw[op]}),
            (
                "op_short_name",
                {
                    "side_effect": lambda op: (
                        "clone" if op.get_name() == "clone" else "pointwise"
                    )
                },
            ),
            (
                "_is_matmul_op",
                {"side_effect": lambda op: op.get_name() == "matmul"},
            ),
        ]:
            stack.enter_context(
                patch(
                    f"torch_spyre._inductor.scratchpad.allocator.{target}",
                    **kwargs,
                )
            )
        return stack

    def _table(self, allocator):
        return allocator._cd_parent_matches(
            self.consumer_op,
            self.consumer_divs,
            self.parent_names,
            self.divisions,
            self.op_by_name,
            {},
            self.residency,
        )

    def test_match_table_is_the_expected_pairs(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with self._patches():
            actual = self._table(allocator)
        # Producers excluded outright ("spilled", "clone") get no entry at all;
        # "plain" is the only one keeping the wide parent candidate. Consumer
        # index 2 slices the buffer like index 0 but on 8 cores, so the
        # cores_used guard drops it everywhere.
        self.assertEqual(
            actual,
            {
                "plain": [(0, 0), (1, 1)],
                "partial": [(1, 1)],
                "unrepr": [(1, 1)],
                "matmul": [(1, 1)],
            },
        )

    def test_compatible_agrees_with_the_table(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with self._patches():
            table = self._table(allocator)
            for parent, pairs in table.items():
                edge = allocator_module.build_residency_edge(
                    parent,
                    self.op_by_name[parent],
                    self.consumer_op,
                    self.rw[self.consumer_op].reads,
                    self.residency[parent],
                    {},
                )
                for i, parent_div in enumerate(self.parent_divs):
                    for j, consumer_div in enumerate(self.consumer_divs):
                        self.assertEqual(
                            edge.compatible(parent_div, consumer_div),
                            (i, j) in pairs,
                            f"{parent} ({i}, {j})",
                        )

    def test_excluded_edges_have_no_edge_object(self):
        with self._patches():
            for parent, reason in [
                ("spilled", "residency"),
                ("clone", "frame-changing clone"),
            ]:
                self.assertIsNone(
                    allocator_module.build_residency_edge(
                        parent,
                        self.op_by_name[parent],
                        self.consumer_op,
                        self.rw[self.consumer_op].reads,
                        self.residency[parent],
                        {},
                    ),
                    reason,
                )

    def test_no_consumer_op_matches_nothing(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with self._patches():
            self.assertEqual(
                allocator._cd_parent_matches(None, [], [], {}, {}, {}, self.residency),
                {},
            )


class TestCoOptimizingAllocator(unittest.TestCase):
    def test_fixed_illegal_split_raises_unsupported(self):
        op = MagicMock(spec=ComputedBuffer, name="fixed_op")
        op.data = MagicMock(spec=Pointwise)
        op.name = "fixed_op"
        graph = MagicMock(operations=[op])
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        fixed = CoreDivision()

        with (
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "ops_in_offset_mutation_component",
                return_value={op.name},
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "_find_distinct_matmul_splits",
                return_value=((), ()),
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._fixed_core_division",
                return_value=fixed,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._division_splits",
                return_value={},
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(
                Unsupported, "fixed split violates hard domain"
            ):
                allocator._division_map(graph)

    def test_pruned_candidates_and_commit_reject_illegal_division(self):
        batch, m = _isym("batch"), _isym("m")
        op = _computed_buffer((4, 64), name="constrained_out")
        graph = MagicMock(operations=[op])
        allocator = CoOptimizingAllocator(MagicMock(), size=1, prune=True)
        safe = {batch: 1, m: 8}
        unsafe = {batch: 4, m: 8}
        rw = MagicMock(
            writes=[MemoryDep(op.name, 64 * batch + m, (batch, m), (4, 64))],
            reads=[],
        )

        with (
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "ops_in_offset_mutation_component",
                return_value=set(),
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "_find_distinct_matmul_splits",
                return_value=((), ()),
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._enum_split_options",
                return_value=[safe, unsafe],
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator.op_read_writes",
                return_value=rw,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_fits_sticks",
                return_value=True,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                side_effect=lambda _op, splits: splits == safe,
            ) as is_legal,
        ):
            divisions = allocator._division_map(graph)[op.name]

        self.assertEqual(
            divisions, [CoreDivision(output_splits={m: 8}, reduction_splits={})]
        )
        self.assertEqual(is_legal.call_args_list[0].args[1], safe)
        self.assertEqual(is_legal.call_args_list[1].args[1], unsafe)

        op.iteration_space_ownership = MagicMock()
        allocation = [
            CoreDivisionBuffer(
                name=op.name,
                size=128,
                uses=[0],
                core_divisions=[
                    CoreDivision(output_splits={batch: 4}, reduction_splits={})
                ],
                chosen_division=0,
            )
        ]
        with (
            patch(
                "torch_spyre._inductor.scratchpad.allocator._division_splits",
                return_value={batch: 4},
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=False,
            ),
            self.assertRaisesRegex(Unsupported, "chosen split violates hard domain"),
        ):
            allocator._commit_divisions(graph, allocation)

    def test_no_enumerable_candidates_keeps_legal_fixed_division(self):
        op = MagicMock(spec=ComputedBuffer)
        op.name = "empty_candidates"
        op.data = MagicMock(spec=Pointwise)
        rw = MagicMock()
        rw.writes = [MagicMock(index=0)]
        rw.reads = []
        allocator = CoOptimizingAllocator(MagicMock(), size=1)

        fixed = CoreDivision(output_splits={1: 2}, reduction_splits={})
        with (
            patch(
                "torch_spyre._inductor.scratchpad.allocator.op_read_writes",
                return_value=rw,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._fixed_core_division",
                return_value=fixed,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._division_splits",
                return_value={},
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "enumerate_work_division_candidates",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=True,
            ),
        ):
            self.assertEqual(
                allocator._enumerate_core_divisions(op, max_cores=32), [fixed]
            )


class TestTopKConstraints(unittest.TestCase):
    def test_topk_uses_minimum_supported_split_domains(self):
        k, search = _isym("k"), _isym("search")
        op = _computed_buffer(
            (8, 16),
            reduction_type="topkvalue",
            reduction_ranges=(search,),
        )
        input_td = _tensor_dep("input", (16,), (search,))
        output_td = _tensor_dep("output", (8,), (k,))
        result = topk_split_domains(
            _make_context(
                op,
                output_td,
                input_tds=[input_td],
                it_space={k: 8, search: 16},
                reduction_vars=[search],
            )
        )
        self.assertEqual(result.allowed_splits[search], frozenset({1}))
        self.assertEqual(result.allowed_splits[k], frozenset({2}))

    def test_default_planner_uses_minimum_supported_k_split(self):
        k, search = _isym("k"), _isym("search")
        splits = multi_dim_iteration_space_split(
            {k: 8, search: 16},
            32,
            [k],
            [search],
            allowed_splits={search: frozenset({1}), k: frozenset({2})},
        )
        self.assertEqual(splits[k], 2)
        self.assertEqual(splits[search], 1)


class TestIndirectAccessSplitDomains(unittest.TestCase):
    _PATCH_TARGET = (
        "torch_spyre._inductor.work_division_constraints.indirect_forbidden_split_syms"
    )

    _PLACEHOLDER_OP = _computed_buffer((128,), name="indirect_placeholder_buf")
    _PLACEHOLDER_TD = _tensor_dep(
        "indirect_placeholder_buf", (128,), (_isym("_placeholder"),)
    )

    def test_restricts_only_indirect_forbidden_dims(self):
        data_dim, partial_entry = _isym("data_dim"), _isym("partial_entry")
        ctx = _make_context(
            self._PLACEHOLDER_OP,
            self._PLACEHOLDER_TD,
            it_space_adjusted={data_dim: 4, partial_entry: 8, _isym("entry"): 16},
        )
        with patch(self._PATCH_TARGET, return_value={data_dim, partial_entry}):
            result = indirect_access_split_domains(ctx)
        self.assertEqual(
            result.allowed_splits,
            {data_dim: frozenset({1}), partial_entry: frozenset({1})},
        )

    def test_non_indirect_op_yields_no_domains(self):
        ctx = _make_context(self._PLACEHOLDER_OP, self._PLACEHOLDER_TD)
        with patch(self._PATCH_TARGET, return_value=set()):
            result = indirect_access_split_domains(ctx)
        self.assertEqual(result.allowed_splits, {})
