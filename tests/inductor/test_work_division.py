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
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.work_division import (
    TensorDep,
    _default_split,
    multi_dim_iteration_space_split,
    span_reduction_pass,
)
from torch_spyre._inductor.work_division_constraints import (
    ConstraintResult,
    WorkDivConstraintContext,
    collect_work_division_constraints,
    conv_spatial_blocked_vars,
    coordinate_mask_blocked_vars,
    indirect_access_pinned_vars,
    qfp8wt_matmul_k_pinned,
    qfp8wt_pinned_vars,
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
        rw.writes = [MagicMock(ranges=(_isym("mb"), _isym("out"), i, j))]
        with patch(self._PATCH_TARGET, return_value=rw):
            self.assertEqual(conv_spatial_blocked_vars(ctx).blocked, {i, j})

    def test_allows_spatial_dims_for_unstrided_conv(self):
        ctx, _, _ = self._context((1, 1))
        self.assertEqual(conv_spatial_blocked_vars(ctx).blocked, set())

    def test_span_commit_overrides_spatial_block(self):
        ctx, i, j = self._context((2, 1))
        ctx.committed_splits = {i: 2}
        rw = MagicMock()
        rw.writes = [MagicMock(ranges=(_isym("mb"), _isym("out"), i, j))]
        with patch(self._PATCH_TARGET, return_value=rw):
            self.assertEqual(collect_work_division_constraints(ctx).blocked, {j})

    def test_blocked_spatial_dims_are_not_distributed(self):
        mb, out, i, j = (_isym(name) for name in ("mb", "out", "i", "j"))
        output_td = _tensor_dep("conv_out", (2, 32, 32, 32), (mb, out, i, j))
        splits, output_dims, _ = _default_split(
            {mb: 2, out: 32, i: 32, j: 32},
            output_td,
            {},
            32,
            {},
            {i, j},
        )
        self.assertNotIn(i, output_dims)
        self.assertNotIn(j, output_dims)
        self.assertEqual(splits[i], 1)
        self.assertEqual(splits[j], 1)


class TestQfp8wtConstraints(unittest.TestCase):
    def test_output_second_stick_coord_pinned_for_qfp8wt_output(self):
        b, m, n = _isym("b"), _isym("m"), _isym("n")
        op = _computed_buffer((4, 8, 128), name="qfp8_out")
        output_td = _tensor_dep(
            "qfp8_out",
            (4, 8, 128),
            (b, m, n),
            element_arrangement=ElementArrangement.QFP8WT,
        )
        ctx = _make_context(op, output_td, it_space={b: 4, m: 8, n: 128})
        result = qfp8wt_pinned_vars(ctx)
        pinned_vars = set(output_td.device_coords[-2].free_symbols)
        self.assertTrue(pinned_vars)
        for v in pinned_vars:
            self.assertEqual(result.pinned[v], 1)

    def test_standard_output_yields_no_pins(self):
        b, m, n = _isym("b"), _isym("m"), _isym("n")
        op = _computed_buffer((4, 8, 128), name="std_out")
        output_td = _tensor_dep("std_out", (4, 8, 128), (b, m, n))
        ctx = _make_context(op, output_td, it_space={b: 4, m: 8, n: 128})
        result = qfp8wt_pinned_vars(ctx)
        self.assertEqual(result.pinned, {})

    def test_matmul_k_pinned_for_batchmatmulfp8_with_qfp8wt_kernel(self):
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
        result = qfp8wt_matmul_k_pinned(ctx)
        self.assertEqual(result.pinned, {k: 1})

    def test_matmul_k_not_pinned_for_plain_batchmatmul(self):
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
        result = qfp8wt_matmul_k_pinned(ctx)
        self.assertEqual(result.pinned, {})


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
            "qfp8wt_pinned_vars",
            "qfp8wt_matmul_k_pinned",
            "indirect_access_pinned_vars",
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

    def test_drops_blocked_var_with_committed_split(self):
        r0 = _isym("r0")
        result = self._collect(
            (
                ConstraintResult(blocked={r0}),
                ConstraintResult(),
                ConstraintResult(),
                ConstraintResult(),
                ConstraintResult(),
            ),
            committed_splits={r0: 2},
        )
        self.assertEqual(result.blocked, set())

    def test_conflicting_pins_raise_unsupported(self):
        r0 = _isym("r0")
        with self.assertRaisesRegex(Unsupported, "conflicting pinned split"):
            self._collect(
                (
                    ConstraintResult(pinned={r0: 2}),
                    ConstraintResult(pinned={r0: 1}),
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
                    ConstraintResult(pinned={k: 1}),
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
                    ConstraintResult(pinned={i0: 1}),
                ),
                committed_splits={i0: 2},
            )

    def test_combines_non_conflicting_rules(self):
        r0, r1, r2, r3 = (_isym(f"r{i}") for i in range(4))
        result = self._collect(
            (
                ConstraintResult(blocked={r0}, pinned={r2: 1}),
                ConstraintResult(blocked={r1}, pinned={r3: 2}),
                ConstraintResult(blocked={r1}),
                ConstraintResult(pinned={r2: 1}),
                ConstraintResult(),
            )
        )
        self.assertEqual(result.blocked, {r0, r1})
        self.assertEqual(result.pinned, {r2: 1, r3: 2})


class TestSpanReductionConstraints(unittest.TestCase):
    _PATCH_TARGET = "torch_spyre._inductor.work_division"

    def test_multidim_indirect_reduction_stays_single_core(self):
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
            patch(f"{self._PATCH_TARGET}.must_split_vars", return_value={}),
            patch(
                f"{self._PATCH_TARGET}.collect_work_division_constraints",
                return_value=ConstraintResult(pinned={r0: 1, r1: 1}),
            ),
            patch(f"{self._PATCH_TARGET}.apply_splits") as apply_splits,
        ):
            span_reduction_pass(op, [], 32)
        self.assertEqual(apply_splits.call_args.args[1], {r0: 1, r1: 1})


class TestIndirectAccessPinnedVars(unittest.TestCase):
    _PATCH_TARGET = (
        "torch_spyre._inductor.work_division_constraints.indirect_info_from_op"
    )

    _PLACEHOLDER_OP = _computed_buffer((128,), name="indirect_placeholder_buf")
    _PLACEHOLDER_TD = _tensor_dep(
        "indirect_placeholder_buf", (128,), (_isym("_placeholder"),)
    )

    def test_indirect_op_pins_every_dim_to_one(self):
        i0, i1 = _isym("i0"), _isym("i1")
        ctx = _make_context(
            self._PLACEHOLDER_OP,
            self._PLACEHOLDER_TD,
            it_space_adjusted={i0: 4, i1: 8},
        )
        with patch(self._PATCH_TARGET, return_value=(["value"], None, None)):
            result = indirect_access_pinned_vars(ctx)
        self.assertEqual(result.pinned, {i0: 1, i1: 1})

    def test_non_indirect_op_yields_no_pins(self):
        i0, i1 = _isym("i0"), _isym("i1")
        ctx = _make_context(
            self._PLACEHOLDER_OP,
            self._PLACEHOLDER_TD,
            it_space_adjusted={i0: 4, i1: 8},
        )
        with patch(self._PATCH_TARGET, return_value=([], None, None)):
            result = indirect_access_pinned_vars(ctx)
        self.assertEqual(result.pinned, {})
