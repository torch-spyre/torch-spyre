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
from unittest.mock import MagicMock, patch

import sympy
import torch
from sympy import Symbol

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.work_division import (
    TensorDep,
    multi_dim_iteration_space_split,
)
from torch_spyre._inductor.work_division_constraints import (
    WorkDivConstraintContext,
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
