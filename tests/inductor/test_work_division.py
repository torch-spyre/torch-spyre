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

import unittest
from contextlib import ExitStack
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
from torch_spyre._inductor.pass_utils import SchedNodeArg
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
    topk_split_domains,
    qfp8wt_split_domains,
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
