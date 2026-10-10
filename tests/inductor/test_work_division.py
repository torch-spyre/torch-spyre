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

import dataclasses
import itertools
import math
import unittest
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import sympy
import torch
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep, StarDep, WeakDep
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    Pointwise,
    Reduction,
)
from torch._inductor.utils import fresh_cache
from torch.utils._sympy.functions import ModularIndexing
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    get_device_dtype,
)
from torch_spyre._inductor import passes
from torch_spyre._inductor import work_division_constraints
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.loop_info import CoarseTileInfo, LoopCarryRecord
from torch_spyre._inductor.constants import (
    AVGPOOL2D_OP,
    BATCH_MATMUL_FP8_OP,
    CONV2D_FWD_OP,
    DEPTHWISE_CONV2D_OP,
)
from torch_spyre._inductor import pass_utils as pass_utils_module
from torch_spyre._inductor.pass_utils import PerCoreView, SchedNodeArg, op_read_writes
from torch_spyre._inductor.scratchpad import allocator as allocator_module
from torch_spyre._inductor.scratchpad import coarse_tiling as coarse_tiling_module
from torch_spyre._inductor import work_division as work_division_module
from torch_spyre._inductor.scratchpad.allocator import (
    CoOptimizingAllocator,
    CoreDivision,
    ScratchpadAllocator,
)
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    TileAxis,
    TileSpec,
)
from torch_spyre._inductor.scratchpad.utils import (
    is_empty_tiled_layout,
)
from torch_spyre._inductor.wsr.enumerate_tilings import (
    TilingSpace,
    build_tiling_space,
    enumerate_tile_options,
)

from torch_spyre._inductor.work_division import (
    TensorDep,
    _cost_model_matmul_planner,
    _default_split,
    _HBM_BW_GBS,
    _matmul_split_cost,
    adjust_it_space_for_sticks,
    enumerate_work_division_candidates,
    work_division_context_for_op,
    work_division_splits_are_legal,
    multi_dim_iteration_space_split,
    span_reduction_pass,
)
from torch_spyre._inductor.work_division_constraints import (
    ConstraintResult,
    WorkDivConstraintContext,
    aligned_ownership_split_domains,
    collect_work_division_constraints,
    conv_spatial_blocked_vars,
    coordinate_mask_blocked_vars,
    direct_read_source_stick_split_domains,
    indirect_access_split_domains,
    keep_by_index_k_split_constraint,
    keep_by_index_pinned_search_space_vars,
    overlapping_coordinate_extents,
    overlapping_window_split_domains,
    qfp8wt_matmul_k_split_domains,
    qfp8wt_split_domains,
    reduction_window_blocked_vars,
    restickify_padding_blocked_vars,
    topk_split_domains,
)
from utils_inductor import mock_op_split_space


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
    return FixedTiledLayout(torch.device("spyre:0"), dtype, size, stride, device_layout)


def _tensor_dep(name, shape, symbols, element_arrangement=None, dtype=torch.float16):
    """Build a real TensorDep for a contiguous access over ``symbols``."""
    layout = _fixed_tiled_layout(
        shape, dtype=dtype, element_arrangement=element_arrangement
    )
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


class TestLoopCarryLxEligibility(unittest.TestCase):
    def setUp(self):
        self.allocator = ScratchpadAllocator(GreedyLayoutSolver, 2**20)

    @staticmethod
    def _tagged_carry(name="carry"):
        op = _computed_buffer((64, 64), name=name)
        op._loop_carry_record = LoopCarryRecord(
            storage_name=name,
            update_name="carry_update",
        )
        return op

    def test_loop_carry_bypasses_output_profitability_denylist(self):
        op = self._tagged_carry()
        with patch.object(self.allocator, "_get_op_name", return_value="convolution"):
            self.assertTrue(self.allocator._op_output_good_for_lx_reuse(op))

            op._loop_carry_record = LoopCarryRecord(
                storage_name="some_other_buffer",
                update_name="carry_update",
            )
            self.assertFalse(self.allocator._op_output_good_for_lx_reuse(op))

    def test_only_joint_solver_accepts_tagged_loop_carry_mutation_target(self):
        graph = SimpleNamespace(operations=[])
        common = dict(
            graph=graph,
            name="carry",
            uses=[0, 1],
            mutated_buffers={"carry"},
            graph_output_names=set(),
            reinterpret_output_names=set(),
            ncores={},
            ncores_reasons={},
            division_is_fixed=False,
            buf_user_deps={},
        )
        ordinary = _computed_buffer((64, 64), name="carry")
        self.assertEqual(
            self.allocator._buffer_residency_reason(op=ordinary, **common),
            "mutation target",
        )

        carry = self._tagged_carry()
        with (
            patch.object(allocator_module, "_is_tiled_advancing", return_value=False),
            patch.object(
                allocator_module, "_is_read_advancing_anywhere", return_value=False
            ),
            patch.object(
                allocator_module,
                "_multi_output_extern_kernel_in_live_range",
                return_value=False,
            ),
            patch.object(
                allocator_module, "buffer_not_read_in_full", return_value=False
            ),
            patch.object(
                allocator_module, "_would_produce_lx_back_gap", return_value=False
            ),
            patch.object(self.allocator, "_restickify_barrier", return_value=None),
            patch.object(
                self.allocator, "_is_index_or_indirectly_accessed", return_value=False
            ),
        ):
            self.assertIsNone(
                self.allocator._buffer_residency_reason(op=carry, **common)
            )
            common["division_is_fixed"] = True
            self.assertEqual(
                self.allocator._buffer_residency_reason(op=carry, **common),
                "mutation target",
            )

    def test_mutated_graph_output_clears_only_with_validated_drain_plan(self):
        """The graph-output-mutation refusal is lifted only by a validated plan.

        This is the exact guard that pins a returned ``for_each_tile``
        accumulator to HBM today (refusal reason "graph output mutated
        after production").  A ``drain_plans`` entry clears it; the plan is
        never produced on the fixed-division (placement) path, and the
        pre-existing mutation-target refusal above it is untouched.
        """
        graph = SimpleNamespace(operations=[])
        common = dict(
            graph=graph,
            name="carry",
            uses=[0, 1],
            mutated_buffers={"carry"},
            graph_output_names={"carry"},
            reinterpret_output_names=set(),
            ncores={},
            ncores_reasons={},
            division_is_fixed=False,
            buf_user_deps={},
        )
        carry = self._tagged_carry()
        with (
            patch.object(allocator_module, "_is_tiled_advancing", return_value=False),
            patch.object(
                allocator_module, "_is_read_advancing_anywhere", return_value=False
            ),
            patch.object(
                allocator_module,
                "_multi_output_extern_kernel_in_live_range",
                return_value=False,
            ),
            patch.object(
                allocator_module, "buffer_not_read_in_full", return_value=False
            ),
            patch.object(
                allocator_module, "_would_produce_lx_back_gap", return_value=False
            ),
            patch.object(self.allocator, "_restickify_barrier", return_value=None),
            patch.object(
                self.allocator, "_is_index_or_indirectly_accessed", return_value=False
            ),
        ):
            # No plan: today's refusal stands.
            self.assertEqual(
                self.allocator._buffer_residency_reason(op=carry, **common),
                "graph output mutated after production",
            )
            # A validated plan clears exactly this branch.
            self.assertIsNone(
                self.allocator._buffer_residency_reason(
                    op=carry, drain_plans={"carry"}, **common
                )
            )
            # The placement path never receives plans, but even if one leaked,
            # the earlier mutation-target refusal still fires first.
            common["division_is_fixed"] = True
            self.assertEqual(
                self.allocator._buffer_residency_reason(
                    op=carry, drain_plans={"carry"}, **common
                ),
                "mutation target",
            )


class TestRestickifyBarrierDeferredOnJointPath(unittest.TestCase):
    """Issue #4655: the restickify barrier tests one committed division

    (``op.iteration_space_ownership``) that reflects whatever a prior pass
    happened to leave on the op, not any division the joint solver could
    actually pick. On the joint path (``division_is_fixed=False``) that
    committed division may disagree with a perfectly compatible candidate the
    solver's own ``cd_parent_matches``/``constrain_residency`` gate would
    later select, permanently barring a buffer the solver would otherwise
    place. The fixed-division (placement) path has no such downstream gate,
    so it must still apply the barrier up front.
    """

    def setUp(self):
        self.allocator = CoOptimizingAllocator(lambda buffers, size: None, 2**20)
        self.op = _computed_buffer((64, 64), name="buf")
        self.common = dict(
            graph=SimpleNamespace(operations=[]),
            name="buf",
            uses=[0, 1],
            op=self.op,
            mutated_buffers=set(),
            graph_output_names=set(),
            reinterpret_output_names=set(),
            ncores={},
            ncores_reasons={},
            buf_user_deps={},
        )

    def test_joint_path_does_not_consult_the_barrier(self):
        with (
            patch.object(
                self.allocator, "_op_output_good_for_lx_reuse", return_value=True
            ),
            patch.object(allocator_module, "is_empty_tiled_layout", return_value=False),
            patch.object(allocator_module, "_is_tiled_advancing", return_value=False),
            patch.object(
                allocator_module, "_is_read_advancing_anywhere", return_value=False
            ),
            patch.object(
                allocator_module,
                "_multi_output_extern_kernel_in_live_range",
                return_value=False,
            ),
            patch.object(
                self.allocator, "_is_index_or_indirectly_accessed", return_value=False
            ),
            patch.object(
                allocator_module, "buffer_not_read_in_full", return_value=False
            ),
            patch.object(
                allocator_module, "_would_produce_lx_back_gap", return_value=False
            ),
            patch.object(
                self.allocator,
                "_restickify_barrier",
                return_value="read by restickify (local-read proof failed)",
            ) as barrier,
        ):
            self.assertIsNone(
                self.allocator._buffer_residency_reason(
                    division_is_fixed=False, **self.common
                )
            )
            barrier.assert_not_called()

            self.assertEqual(
                self.allocator._buffer_residency_reason(
                    division_is_fixed=True, **self.common
                ),
                "read by restickify (local-read proof failed)",
            )
            barrier.assert_called_once()


class TestEmptyLxEligibility(unittest.TestCase):
    def test_empty_tensors_are_rejected_before_lx_sizing(self):
        """A valid empty tensor clears no eligibility path.

        Native stickification preserves zero outer extents. A zero physical
        extent on a logically nonempty tensor (a one-stick FP16 tensor
        quantized to FP8 rescales to zero FP8 sticks) is not an empty tensor.
        """

        empty = _fixed_tiled_layout((0, 64))
        nonempty = _fixed_tiled_layout((64, 64))
        zero_extent = _fixed_tiled_layout((64, 64))
        zero_extent.device_layout = SpyreTensorLayout(
            [1, 0, 64],
            [64, 64, 1],
            DataFormats.SEN169_FP16,
            ElementArrangement.STANDARD,
        )
        self.assertEqual(
            [
                is_empty_tiled_layout(layout)
                for layout in (empty, nonempty, zero_extent)
            ],
            [True, False, False],
        )

        graph = SimpleNamespace(
            try_get_buffer=lambda _name: SimpleNamespace(layout=empty)
        )
        allocator = ScratchpadAllocator(GreedyLayoutSolver, 2**20)
        with patch.object(
            allocator_module, "clone_at_graph_boundaries", return_value=True
        ):
            reason = allocator._input_residency_reason(
                graph, "value", [0, 1], division_is_fixed=False
            )
        self.assertEqual(reason, "empty tensor")
        with patch.object(allocator, "_op_output_good_for_lx_reuse", return_value=True):
            reason = allocator._buffer_residency_reason(
                graph,
                "value",
                [0, 1],
                SimpleNamespace(layout=empty),
                mutated_buffers=set(),
                graph_output_names=set(),
                reinterpret_output_names=set(),
                ncores={},
                ncores_reasons={},
                division_is_fixed=False,
                buf_user_deps={},
            )
        self.assertEqual(reason, "empty tensor")


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


class TestAlignedOwnershipSplitDomains(unittest.TestCase):
    def _context(self, source_shape, source_index):
        rows, cols = _isym("d0"), _isym("d1")
        op = _computed_buffer((6, 128), name="repeat")
        output_td = _tensor_dep("repeat", (6, 128), (rows, cols))
        source = TensorDep(
            dep=MemoryDep("x", source_index(rows, cols), (rows, cols), (6, 128)),
            layout=_fixed_tiled_layout(source_shape),
        )
        ctx = _make_context(
            op,
            output_td,
            [source],
            it_space={rows: 6, cols: 128},
            it_space_adjusted={rows: 6, cols: 2},
            stick_vars={cols: 64},
        )
        return ctx, rows

    def test_repeat_rows_split_only_into_whole_blocks(self):
        # x.repeat(3, 2) over x of shape (2, 64): a 2-way row split would give
        # core 0 output rows {0, 2, 4} after alignment.
        ctx, rows = self._context(
            (2, 64),
            lambda d0, d1: 64 * ModularIndexing(d0, 1, 2) + ModularIndexing(d1, 1, 64),
        )
        result = aligned_ownership_split_domains(ctx)
        self.assertEqual(result.allowed_splits[rows], frozenset({1, 3, 6}))
        self.assertFalse(result.blocked)

    def test_affine_read_leaves_rows_unconstrained(self):
        ctx, rows = self._context((6, 128), lambda d0, d1: 128 * d0 + d1)
        result = aligned_ownership_split_domains(ctx)
        self.assertNotIn(rows, result.allowed_splits)


class TestOverlappingWindowSplitDomains(unittest.TestCase):
    """Issue #5347: an unfold-style read sweeps overlapping rows of one device dim."""

    def test_unfold_window_is_capped_at_its_device_dim(self):
        # x.unfold(d, 4, 2): row 2*w + k, k's device dim holds 2 of its 4 values.
        w, k = _isym("w"), _isym("k")
        self.assertEqual(
            overlapping_coordinate_extents(2 * w + k, {w: 7, k: 4}), {k: (2, w)}
        )

    def test_equal_steps_cap_the_shorter_window_axis(self):
        # x.unfold(0, 3, 1): row w + k; normalize_coordinates orders k (range 3)
        # inside w (range 8) and gives it a size-1 device dim.
        w, k = _isym("w"), _isym("k")
        self.assertEqual(
            overlapping_coordinate_extents(w + k, {w: 8, k: 3}), {k: (1, w)}
        )

    def test_non_overlapping_terms_are_free(self):
        w, k = _isym("w"), _isym("k")
        # unfold(size=2, step=2): windows tile the dim exactly.
        self.assertEqual(overlapping_coordinate_extents(2 * w + k, {w: 8, k: 2}), {})
        # A size-1 window variable cannot overlap.
        self.assertEqual(overlapping_coordinate_extents(2 * w + k, {w: 8, k: 1}), {})
        self.assertEqual(overlapping_coordinate_extents(w, {w: 8}), {})

    def test_only_the_overlapping_term_is_capped(self):
        a, b, c = _isym("a"), _isym("b"), _isym("c")
        self.assertEqual(
            overlapping_coordinate_extents(6 * a + 2 * b + c, {a: 2, b: 3, c: 4}),
            {c: (2, b)},
        )

    def test_unanalysable_coordinates_are_skipped(self):
        w, k = _isym("w"), _isym("k")
        self.assertEqual(
            overlapping_coordinate_extents(sympy.floor(w / 2) + k, {w: 8, k: 4}), {}
        )
        # Strides that do not nest (3 is not a multiple of 2).
        self.assertEqual(
            overlapping_coordinate_extents(3 * w + 2 * k, {w: 4, k: 4}), {}
        )

    def _unfold_sum_context(self):
        # sum(x.unfold(1, 4, 2), dim=0) over x of shape (8, 16, 6): the read
        # index 12*w + j + 6*k + 96*b puts 2*w + k on x's row dim.
        w, j, k, b = (_isym(n) for n in ("w", "j", "k", "b"))
        op = _computed_buffer((7, 6, 4), name="buf0", reduction_type="sum")
        output_td = _tensor_dep("buf0", (7, 6, 4), (w, j, k))
        source = TensorDep(
            dep=MemoryDep("x", 12 * w + j + 6 * k + 96 * b, (w, j, k, b), (7, 6, 4, 8)),
            layout=_fixed_tiled_layout((8, 16, 6)),
        )
        ctx = _make_context(
            op,
            output_td,
            [source],
            it_space={w: 7, j: 6, k: 4, b: 8},
            reduction_vars=[b],
        )
        return ctx, k

    def test_unfold_read_domains(self):
        ctx, k = self._unfold_sum_context()
        result = overlapping_window_split_domains(ctx)
        self.assertEqual(result.allowed_splits, {k: frozenset({2, 4})})
        self.assertFalse(result.blocked)

    def test_collected_with_the_other_constraints(self):
        ctx, k = self._unfold_sum_context()
        result = collect_work_division_constraints(ctx)
        self.assertEqual(result.allowed_splits[k], frozenset({2, 4}))

    def test_reduced_window_pins_the_outer_window_index(self):
        # sum(x.unfold(1, 4, 2), dim=-1): k is reduced, so rather than being
        # split it may stay whole on a core that sees a single window w.
        b, w, j, k = (_isym(n) for n in ("b", "w", "j", "k"))
        op = _computed_buffer((8, 7, 6), name="buf0", reduction_type="sum")
        output_td = _tensor_dep("buf0", (8, 7, 6), (b, w, j))
        source = TensorDep(
            dep=MemoryDep("x", 96 * b + 12 * w + j + 6 * k, (b, w, j, k), (8, 7, 6, 4)),
            layout=_fixed_tiled_layout((8, 16, 6)),
        )
        ctx = _make_context(
            op,
            output_td,
            [source],
            it_space={b: 8, w: 7, j: 6, k: 4},
            reduction_vars=[k],
        )
        domains = overlapping_window_split_domains(ctx).allowed_splits
        self.assertEqual(domains[k], frozenset({1, 2, 4}))
        self.assertEqual(domains[w], frozenset({7}))
        self.assertNotIn(j, domains)

    def test_contiguous_read_is_unconstrained(self):
        rows, cols = _isym("d0"), _isym("d1")
        op = _computed_buffer((6, 128))
        output_td = _tensor_dep("buf0", (6, 128), (rows, cols))
        source = _tensor_dep("x", (6, 128), (rows, cols))
        ctx = _make_context(op, output_td, [source], it_space={rows: 6, cols: 128})
        self.assertEqual(overlapping_window_split_domains(ctx), ConstraintResult())


class TestDirectReadSourceStickSplitDomains(unittest.TestCase):
    def test_only_whole_stick_partitions_are_legal(self):
        head, feature, key = (_isym(name) for name in ("head", "feature", "key"))
        op = _computed_buffer((2, 128, 64), name="direct_read_candidate")
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[sympy.Integer(3)],
            loop_tiled_dims=[[]],
        )
        output_td = _tensor_dep("output", (2, 128, 64), (head, feature, key))

        for feature_extent, expected in (
            (128, frozenset({1, 2})),
            (192, frozenset({1, 3})),
            (96, frozenset({1})),
        ):
            with self.subTest(feature_extent=feature_extent):
                source_layout = _fixed_tiled_layout((2, 8192, feature_extent))
                source_dep = MemoryDep(
                    "source",
                    8192 * feature_extent * head + feature + feature_extent * key,
                    (head, feature, key),
                    (2, feature_extent, 64),
                )
                ctx = _make_context(
                    op,
                    output_td,
                    it_space={head: 2, feature: feature_extent, key: 64},
                )
                with patch(
                    "torch_spyre._inductor.work_division_constraints."
                    "_direct_read_source_dep",
                    return_value=(source_dep, source_layout),
                ):
                    result = direct_read_source_stick_split_domains(ctx)

                self.assertEqual(result.allowed_splits, {feature: expected})

    def test_short_loop_does_not_force_source_compatible_division(self):
        head, feature, key = (_isym(name) for name in ("head", "feature", "key"))
        op = _computed_buffer((2, 128, 64), name="short_direct_read_candidate")
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[sympy.Integer(2)],
            loop_tiled_dims=[[]],
        )
        output_td = _tensor_dep("output", (2, 128, 64), (head, feature, key))
        source_layout = _fixed_tiled_layout((2, 128, 128))
        source_dep = MemoryDep(
            "source",
            128 * 128 * head + feature + 128 * key,
            (head, feature, key),
            (2, 128, 64),
        )
        ctx = _make_context(
            op,
            output_td,
            it_space={head: 2, feature: 128, key: 64},
        )
        with patch(
            "torch_spyre._inductor.work_division_constraints._direct_read_source_dep",
            return_value=(source_dep, source_layout),
        ):
            result = direct_read_source_stick_split_domains(ctx)

        self.assertEqual(result, ConstraintResult())


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


class TestMatmulRowOrderSplitDomains(unittest.TestCase):
    def test_flattened_staggered_rows_allow_backend_reordering(self):
        from torch_spyre._inductor.constants import BATCH_MATMUL_OP

        rows, n, k = (_isym(name) for name in ("rows", "n", "k"))
        op = _computed_buffer(
            (8, 64), reduction_type=BATCH_MATMUL_OP, reduction_ranges=(64,)
        )
        lhs = TensorDep(
            MemoryDep("lhs", 64 * rows + k, (rows, k), (8, 64)),
            _fixed_tiled_layout(
                (2, 4, 64), element_arrangement=ElementArrangement.FP32_TO_DL16
            ),
        )
        rhs = _tensor_dep("rhs", (64, 64), (n, k))
        output = _tensor_dep("out", (8, 64), (rows, n))
        ctx = _make_context(
            op,
            output,
            [lhs, rhs],
            it_space={rows: 8, n: 64, k: 64},
            it_space_adjusted={rows: 8, n: 1, k: 1},
            stick_vars={n: 64, k: 64},
            reduction_vars=[k],
        )
        self.assertEqual(
            aligned_ownership_split_domains(ctx).allowed_splits[rows],
            frozenset({1, 2, 4, 8}),
        )
        # Matching physical row order must not ban a one-core matmul.
        ctx.output_td = TensorDep(
            MemoryDep("out", 64 * rows + n, (rows, n), (8, 64)),
            _fixed_tiled_layout((2, 4, 64)),
        )
        self.assertEqual(
            aligned_ownership_split_domains(ctx).allowed_splits[rows],
            frozenset({1, 2, 4, 8}),
        )
        ctx.output_td = output
        # Non-matmuls retain the same contiguous-ownership split domain.
        ctx.op = _computed_buffer((8, 64))
        self.assertEqual(
            aligned_ownership_split_domains(ctx).allowed_splits[rows],
            frozenset({1, 2, 4, 8}),
        )
        # One row segment remains unrestricted, including a one-core matmul.
        ctx.op = op
        ctx.input_tds[0] = _tensor_dep(
            "lhs",
            (8, 64),
            (rows, k),
            element_arrangement=ElementArrangement.FP32_TO_DL16,
        )
        self.assertNotIn(rows, aligned_ownership_split_domains(ctx).allowed_splits)


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

    def test_fp8_cost_model_uses_correct_elems_per_stick(self):
        """#4466: N_e/K_e must come from the FP8 operand's stick (128
        elems/stick), not the FP16 output's (64) -- else they're halved."""
        m, n, k = (_isym(name) for name in ("m", "n", "k"))
        op = _computed_buffer(
            (8, 12800),
            name="scaled_mm_out",
            reduction_type=BATCH_MATMUL_FP8_OP,
            reduction_ranges=(4096,),
        )
        output_td = _tensor_dep("scaled_mm_out", (8, 12800), (m, n))
        input_tds = [
            _tensor_dep(
                "act",
                (8, 4096),
                (m, k),
                element_arrangement=ElementArrangement.QFP8CH,
                dtype=torch.float8_e4m3fn,
            ),
            _tensor_dep(
                "weight",
                (4096, 12800),
                (k, n),
                element_arrangement=ElementArrangement.QFP8WT,
                dtype=torch.float8_e4m3fn,
            ),
        ]
        it_space = {m: 8, n: 12800, k: 4096}
        it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
            it_space, input_tds + [output_td]
        )

        captured = {}

        def capture_axes(_b_axis, _m_axis, n_axis, k_axis, *_args, **_kwargs):
            captured["N_e"] = n_axis[0]
            captured["K_e"] = k_axis[0]
            return 1.0

        with patch(
            "torch_spyre._inductor.work_division._matmul_split_cost",
            side_effect=capture_axes,
        ):
            _cost_model_matmul_planner(
                op,
                {sym: 1 for sym in it_space_adjusted},
                it_space_adjusted,
                output_td,
                stick_vars,
                {},
                32,
                input_tds,
                set(),
                {},
            )

        self.assertEqual(captured["N_e"], 12800)
        self.assertEqual(captured["K_e"], 4096)

    def test_fp8_matmul_split_cost_uses_correct_byte_width(self):
        """#4465: activation/weight bytes must come from their own FP8
        elems_per_stick (1 byte/elem), not the flat fp16 _DTYPE_BYTES (2)."""
        m, n, k = (_isym(name) for name in ("m", "n", "k"))
        op = _computed_buffer(
            (8, 12800),
            name="scaled_mm_out",
            reduction_type=BATCH_MATMUL_FP8_OP,
            reduction_ranges=(4096,),
        )
        output_td = _tensor_dep("scaled_mm_out", (8, 12800), (m, n))
        input_tds = [
            _tensor_dep(
                "act",
                (8, 4096),
                (m, k),
                element_arrangement=ElementArrangement.QFP8CH,
                dtype=torch.float8_e4m3fn,
            ),
            _tensor_dep(
                "weight",
                (4096, 12800),
                (k, n),
                element_arrangement=ElementArrangement.QFP8WT,
                dtype=torch.float8_e4m3fn,
            ),
        ]
        it_space_adjusted = {m: 8, n: 100, k: 32}

        captured = {}

        def capture_bytes(_b_axis, _m_axis, _n_axis, _k_axis, *_args, **kwargs):
            captured["operand_bytes"] = kwargs.get("operand_bytes")
            captured["output_bytes"] = kwargs.get("output_bytes")
            return 1.0

        with patch(
            "torch_spyre._inductor.work_division._matmul_split_cost",
            side_effect=capture_bytes,
        ):
            _cost_model_matmul_planner(
                op,
                {sym: 1 for sym in it_space_adjusted},
                it_space_adjusted,
                output_td,
                {n: 128},
                {},
                32,
                input_tds,
                set(),
                {},
            )

        # Layer 1: planner must derive+pass these; reverted -> None != 1.0/2.0.
        self.assertEqual(captured["operand_bytes"], 1.0)
        self.assertEqual(captured["output_bytes"], 2.0)

        # Layer 2: old no-kwargs callers (e.g. cost_model.py) keep the flat default.
        B, M, K, N = 1, 8, 4096, 12800
        sm, sn, sk = 4, 8, 1  # fanout_split=max(sm,sn)=8 keeps cohort_penalty == 1.0
        legacy_cost_with_hbm = _matmul_split_cost(
            (B, 1),
            (M, sm),
            (N, sn),
            (K, sk),
            32,
            shared_weight=True,
        )
        legacy_cost_without_hbm = _matmul_split_cost(
            (B, 1),
            (M, sm),
            (N, sn),
            (K, sk),
            32,
            shared_weight=True,
            include_hbm=False,
        )
        legacy_bytes_total = (
            (legacy_cost_with_hbm - legacy_cost_without_hbm) * _HBM_BW_GBS * 1000
        )
        self.assertAlmostEqual(legacy_bytes_total, 105_127_936, delta=1.0)

        # Layer 3: the real (unmocked) function, given correct fp8 byte widths,
        # must itself compute the correct bytes_total.
        shared_cost_with_hbm = _matmul_split_cost(
            (B, 1),
            (M, sm),
            (N, sn),
            (K, sk),
            32,
            shared_weight=True,
            operand_bytes=1.0,
            output_bytes=2.0,
        )
        shared_cost_without_hbm = _matmul_split_cost(
            (B, 1),
            (M, sm),
            (N, sn),
            (K, sk),
            32,
            shared_weight=True,
            include_hbm=False,
            operand_bytes=1.0,
            output_bytes=2.0,
        )
        shared_bytes_total = (
            (shared_cost_with_hbm - shared_cost_without_hbm) * _HBM_BW_GBS * 1000
        )
        self.assertAlmostEqual(shared_bytes_total, 52_666_368, delta=1.0)

        # Separate-batched-weight branch (weight_batches=B, not 1): B=2,
        # M=8, K=4096, N=12800. fanout_split = n (shared_weight=False), so
        # n=8 again keeps cohort_penalty == 1.0.
        B2, M2, K2, N2 = 2, 8, 4096, 12800
        m2, n2, k2 = 1, 8, 1
        separate_cost_with_hbm = _matmul_split_cost(
            (B2, 1),
            (M2, m2),
            (N2, n2),
            (K2, k2),
            32,
            shared_weight=False,
            operand_bytes=1.0,
            output_bytes=2.0,
        )
        separate_cost_without_hbm = _matmul_split_cost(
            (B2, 1),
            (M2, m2),
            (N2, n2),
            (K2, k2),
            32,
            shared_weight=False,
            include_hbm=False,
            operand_bytes=1.0,
            output_bytes=2.0,
        )
        separate_bytes_total = (
            (separate_cost_with_hbm - separate_cost_without_hbm) * _HBM_BW_GBS * 1000
        )
        # weight_batches=B2=2 (not shared): (B2*M2*K2 + B2*K2*N2)*1 + B2*M2*N2*2
        self.assertAlmostEqual(separate_bytes_total, 105_332_736, delta=1.0)


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

    def _blocked(self, ctx, i, j):
        """Run the constraint against the (mb, out, i, j) output write ranges."""
        rw = MagicMock()
        # Inductor stores ranges in OrderedSet, which does not support slices.
        rw.writes = [MagicMock(ranges=(_isym("mb"), _isym("out"), i, j))]
        with patch(self._PATCH_TARGET, return_value=rw):
            return conv_spatial_blocked_vars(ctx).blocked

    def test_blocks_spatial_dims_for_strided_conv(self):
        ctx, i, j = self._context((2, 1))
        self.assertEqual(self._blocked(ctx, i, j), {i, j})

    def test_allows_spatial_dims_for_unstrided_conv(self):
        # An unstrided conv splits spatially per-core, so nothing is blocked --
        # including a collapsed (kernel-extent-1) axis, whose split is correct
        # (see conv_spatial_blocked_vars).
        ctx, i, j = self._context((1, 1))
        self.assertEqual(self._blocked(ctx, i, j), set())

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


class TestDepthwiseConvWindowBlocked(unittest.TestCase):
    """End-to-end: a depthwise conv's kernel window must stay unsplit.

    SuperDSC rejects a ki/kj split for every conv, and the scheduler transport
    cannot even carry one to it: the depthwise input read indexes the output
    position (window offsets live in conv_params), so a window split has no
    read coefficient and is dropped. The work-division guard is what keeps the
    solver from pricing a plan that cannot run.
    """

    _X_SHAPE = (1, 64, 32, 32)
    _W_SHAPE = (64, 1, 3, 3)

    @staticmethod
    def _conv(x, w):
        return torch.conv2d(x, w, None, stride=(1, 1), groups=x.shape[1])

    def _inputs(self):
        """CPU inputs and their device copies, both with channel as the stick."""
        fp16 = get_device_dtype(torch.float16)
        x = torch.randn(self._X_SHAPE, dtype=torch.float16)
        w = torch.randn(self._W_SHAPE, dtype=torch.float16)
        x_dev = x.to(
            device_layout=SpyreTensorLayout(
                [32, 32, 1, 1, 64], [1, 32, -1, 65536, 1024], fp16
            )
        )
        w_dev = w.to(
            device_layout=SpyreTensorLayout([3, 3, 1, 1, 64], [1, 3, -1, 9, 9], fp16)
        )
        return x, w, x_dev, w_dev

    def _compile_depthwise(self):
        """Compile and run the depthwise conv, recording each (ctx, result) of
        reduction_window_blocked_vars on it. Returns the device output, the
        CPU reference, and the recorded pairs."""
        captured = []
        real_window = work_division_constraints.reduction_window_blocked_vars

        def window(ctx):
            result = real_window(ctx)
            if getattr(ctx.op.data, "reduction_type", None) == DEPTHWISE_CONV2D_OP:
                captured.append((ctx, result))
            return result

        x, w, x_dev, w_dev = self._inputs()
        torch._dynamo.reset()
        with fresh_cache():
            with patch.object(
                work_division_constraints, "reduction_window_blocked_vars", window
            ):
                out = torch.compile(self._conv)(x_dev, w_dev).cpu()
        self.assertTrue(captured, "depthwise conv never reached work division")
        return out, self._conv(x, w), captured

    @staticmethod
    def _kernel_window(ctx):
        """The vars only the weight read indexes: the kernel window."""
        write_vars = op_read_writes(ctx.op).writes
        write_syms = set().union(*(d.index.free_symbols for d in write_vars))
        read_syms = set().union(
            *(d.index.free_symbols for d in op_read_writes(ctx.op).reads)
        )
        return {v for v in read_syms - write_syms if v in ctx.it_space}

    def test_blocks_exactly_the_kernel_window(self):
        out, ref, captured = self._compile_depthwise()
        for ctx, result in captured:
            window = self._kernel_window(ctx)
            self.assertEqual(sorted(int(ctx.it_space[v]) for v in window), [3, 3])
            self.assertEqual(result.blocked, window)
            # The channel stick dim sits first in reduction_vars (the output
            # stick is excluded from its coordinate vars) but is not reduced
            # over; a positional reduction_vars[:2] would block it and kh,
            # leaving kw free.
            channel = ctx.reduction_vars[0]
            self.assertNotIn(channel, window)
            self.assertNotIn(channel, result.blocked)
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.1)

    def test_unblocked_window_split_is_dropped_before_codegen(self):
        """What the guard prevents: permit (and force) a kh split, and the
        committed plan cannot be carried to codegen."""
        forced = {}
        committed = {}
        real_window = work_division_constraints.reduction_window_blocked_vars
        real_finalize = passes.finalize_work_division_for_scheduler

        def force_kh_split(ctx):
            if getattr(ctx.op.data, "reduction_type", None) != DEPTHWISE_CONV2D_OP:
                return real_window(ctx)
            kh = min(self._kernel_window(ctx), key=str)
            forced["kh"] = kh
            return ConstraintResult(allowed_splits={kh: frozenset({3})})

        def finalize(graph):
            for op in graph.operations:
                data = getattr(op, "data", None)
                if getattr(data, "reduction_type", None) == DEPTHWISE_CONV2D_OP:
                    committed.update(op.iteration_space_ownership.work_slices)
            real_finalize(graph)

        x, w, x_dev, w_dev = self._inputs()
        torch._dynamo.reset()
        with (
            fresh_cache(),
            patch.object(
                work_division_constraints,
                "reduction_window_blocked_vars",
                force_kh_split,
            ),
            patch.object(passes, "finalize_work_division_for_scheduler", finalize),
        ):
            with self.assertLogs("spyre.inductor.pass_utils", "WARNING") as logs:
                out = torch.compile(self._conv)(x_dev, w_dev).cpu()
        ref = self._conv(x, w)
        kh = forced["kh"]
        self.assertEqual(committed[kh], 3)
        self.assertTrue(
            any(
                "lossy work-division scheduler transport" in line
                and f"reduction:{kh}=absent" in line
                for line in logs.output
            ),
            logs.output,
        )
        # The dropped split leaves numerics intact -- the op simply runs on
        # fewer cores than the solver priced -- which is why only the guard,
        # not a numeric test, keeps this plan out.
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.1)


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


def _physical_view(*splits):
    """A real :class:`PerCoreView` whose cores own contiguous slices in
    row-major order of ``splits`` (device-dim index, split factor)."""

    core_id = sympy.Symbol("core_id")
    num_cores = math.prod(split for _, split in splits)
    slots = []
    stride = num_cores
    for dim, split in splits:
        stride //= split
        slots.append((dim, sympy.Mod(sympy.floor(core_id / stride), split)))
    return PerCoreView(tuple(splits), tuple(slots), num_cores=num_cores)


class TestResidencyEdgeMatching(unittest.TestCase):
    """The compatibility seam: the pairs :class:`ResidencyEdge` admits for a
    corpus of policy cases, and the pairwise :meth:`ResidencyEdge.compatible`
    a generator would call agreeing with the table it replaces."""

    def setUp(self):
        x, y = _isym("x"), _isym("y")
        self.view_a = _physical_view((0, 4))
        self.view_b = _physical_view((0, 2))
        self.view_wide = _physical_view((0, 2), (1, 2))

        def _div(splits, reduction=None):
            output = dict(splits)
            reduction = dict(reduction or {})
            return CoreDivision(
                splits={**output, **reduction}, reduction_syms=frozenset(reduction)
            )

        # Consumer: a 4-core slicing, a 2-core one, an 8-core one that slices
        # the buffer the same way as the first (the stale-LX case), and a
        # 4-core one slicing two device dims -- the only consumer the wide
        # parent candidate can pair with.
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
            # A matmul split across two device dims uses the same complete
            # ownership comparison as every other producer.
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
        self.divisions["matmul"] = [self.consumer_divs[3], self.parent_divs[1]]
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
        # Expanding a clone's input does not make its finished output partial.
        # Check input ownership separately from this output-consumer edge.
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
            index = [cd.splits for cd in self.consumer_divs].index(division.splits)
            return (self.consumer_views[index], False, True)
        views, partial, repr_ok, _matmul = self.parents.get(
            name, ([self.view_a, self.view_b], [False, False], [True, True], False)
        )
        index = [cd.splits for cd in self.divisions[name]].index(division.splits)
        return (views[index], partial[index], repr_ok[index])

    def _patches(self):
        # ``op_read_writes`` is called from both modules -- the allocator reads
        # the consumer's, the edge its producer's -- so each name is patched
        # wherever it is bound.
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
            for module in (allocator_module, work_division_module):
                if hasattr(module, target):
                    stack.enter_context(patch.object(module, target, **kwargs))
        return stack

    def _table(self, allocator):
        edges = allocator._parent_residency_edges(
            self.consumer_op, self.parent_names, self.op_by_name, {}, self.residency
        )
        return allocator._cd_parent_matches(edges, self.consumer_divs, self.divisions)

    def test_match_table_is_the_expected_pairs(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with self._patches():
            actual = self._table(allocator)
        # A rejected producer ("spilled") gets no entry. The wide matmul
        # matches only the same two-axis consumer. Consumer
        # index 2 slices the buffer like index 0 but on 8 cores, so the
        # cores_used guard drops it everywhere.
        self.assertEqual(
            actual,
            {
                "plain": [(0, 0), (1, 1)],
                "partial": [(1, 1)],
                "unrepr": [(1, 1)],
                "matmul": [(0, 3), (1, 1)],
                "clone": [(0, 0), (1, 1)],
            },
        )

    def test_multi_axis_matmul_requires_a_representable_finished_write(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        for partial, representable in ((True, True), (False, False)):
            with self.subTest(partial=partial, representable=representable):
                self.parents["matmul"] = (
                    [self.view_wide, self.view_b],
                    [partial, False],
                    [representable, True],
                    True,
                )
                with self._patches():
                    self.assertEqual(self._table(allocator)["matmul"], [(1, 1)])

    def test_a_producer_read_twice_pairs_only_where_both_reads_match(self):
        # The consumer reads "plain" a second time through another index, and
        # that read slices it differently: under consumer index 0 it owns the
        # buffer on two device dims, and under index 3 the way the first read
        # does under index 0. A pair has to be true of both reads, which
        # leaves (1, 1): (0, 0) holds for the first read alone and (0, 3) for
        # the second alone.
        y = _isym("y")
        second = MemoryDep("plain", y, (y,), (8,))
        self.rw[self.consumer_op].reads.append(second)
        second_views = [self.view_wide, self.view_b, self.view_a, self.view_a]

        def view_for_div(op, dep, buf_name, division, prep_cache):
            if dep == second:
                index = [cd.splits for cd in self.consumer_divs].index(division.splits)
                return (second_views[index], False, True)
            return self._view_for_div(op, dep, buf_name, division, prep_cache)

        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with (
            self._patches(),
            patch.object(
                work_division_module, "_view_for_div", side_effect=view_for_div
            ),
        ):
            pairs = self._table(allocator).get("plain", [])
        self.assertLessEqual(
            set(pairs), {(1, 1)}, "a pair must hold for every read of the producer"
        )

    def test_loop_carry_update_is_a_storage_ownership_edge(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        storage_op = self.op_by_name["plain"]
        record = LoopCarryRecord(
            storage_name=storage_op.get_name(),
            update_name=self.consumer_op.get_name(),
        )
        storage_op._loop_carry_record = record
        self.consumer_op._loop_carry_record = record

        with self._patches():
            edge = allocator._loop_carry_update_edge(
                self.consumer_op,
                self.op_by_name,
                {},
            )
            self.assertIsNotNone(edge)
            self.assertEqual(edge.buf_name, storage_op.get_name())
            self.assertEqual(
                [dep.name for dep in edge.read_deps], [storage_op.get_name()]
            )
            self.assertEqual(
                edge.match_pairs(self.parent_divs, self.consumer_divs),
                [(0, 0), (1, 1)],
            )

    def _carry_update(self):
        """Make ``plain`` a loop carry updated by a new op ``update``."""
        x = _isym("x")
        storage_op = self.op_by_name["plain"]
        update_op = self._op("update")
        record = LoopCarryRecord(storage_name="plain", update_name="update")
        storage_op._loop_carry_record = record
        update_op._loop_carry_record = record
        self.op_by_name["update"] = update_op
        self.divisions["update"] = self.parent_divs
        self.rw[update_op] = MagicMock(
            writes=[MemoryDep("update", x, (x,), (8,))],
            reads=[MemoryDep("plain", x, (x,), (8,))],
        )
        return update_op

    def test_reading_a_loop_carry_update_is_an_edge_on_its_storage(self):
        # The update writes through the carry's storage, so a later read of the
        # update's name reads the storage's LX bytes: it needs the storage's
        # ownership, although the update itself is never an LX buffer.
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        update_op = self._carry_update()
        x = _isym("x")
        self.rw[self.consumer_op].reads.append(MemoryDep("update", x, (x,), (8,)))

        with self._patches():
            carry_edges = {
                "update": allocator._loop_carry_update_edge(
                    update_op, self.op_by_name, {}
                )
            }
            edges = allocator._loop_carry_read_edges(self.consumer_op, carry_edges, {})
            self.assertEqual(set(edges), {"plain"})
            edge = edges["plain"]
            self.assertIs(edge.consumer_op, self.consumer_op)
            self.assertEqual([dep.name for dep in edge.read_deps], ["plain"])
            self.assertEqual(
                edge.match_pairs(self.parent_divs, self.consumer_divs),
                [(0, 0), (1, 1)],
            )
            # The update's own write is the carry edge, not a read edge.
            self.assertEqual(
                allocator._loop_carry_read_edges(update_op, carry_edges, {}), {}
            )

    @staticmethod
    def _rw(reads, writes):
        x = _isym("x")
        return SimpleNamespace(
            reads=OrderedSet(MemoryDep(n, x, (x,), (8,)) for n in reads),
            writes=OrderedSet(MemoryDep(n, x, (x,), (8,)) for n in writes),
        )

    def _drained_carry_graph(self, after_loop, outputs):
        """A small graph around one drained ``for_each_tile`` carry.

        ``plain`` is filled before the loop and updated in place by
        ``update``.  Inside the loop, ``consumer`` reads the update's name, so
        it reads the storage's bytes.  ``after_loop`` runs after the loop and
        ``outputs`` are the graph outputs.
        """
        from torch._inductor.ir import MutationLayoutSHOULDREMOVE

        storage_op = self.op_by_name["plain"]
        update_op = self._carry_update()
        reader_op = self.consumer_op
        ops = [storage_op, update_op, reader_op, *after_loop]
        for op in ops:
            op.name = op.get_name()
            op.layout = _fixed_tiled_layout((8, 64))
        op_by_name = {op.name: op for op in ops}
        graph = MagicMock()
        graph.operations = ops
        graph.graph_input_names = []
        graph.graph_outputs = list(outputs)
        graph.get_output_names.return_value = [op.get_name() for op in outputs]
        graph.get_buffer.side_effect = op_by_name.get
        # The drain's FX clone reads the storage's own FX node.
        graph.graph = torch.fx.Graph()
        storage_op.origins = OrderedSet([graph.graph.placeholder("plain")])

        # One counted loop holds the update and the reader. The update writes
        # through the storage.
        loop = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[sympy.Integer(4)], loop_tiled_dims=[[]]
        )
        update_op.loop_info = loop
        reader_op.loop_info = loop
        record = LoopCarryRecord(
            storage_name="plain",
            update_name="update",
            loop_origin=SimpleNamespace(graph=graph.graph),
        )
        storage_op._loop_carry_record = record
        update_op._loop_carry_record = record
        update_op.layout = MagicMock(spec=MutationLayoutSHOULDREMOVE)
        update_op.layout.target = storage_op
        self.rw.update(
            {
                storage_op: self._rw([], ["plain"]),
                update_op: self._rw(["plain"], ["update"]),
                reader_op: self._rw(["update"], ["consumer"]),
            }
        )
        return graph, ops

    def _plan_solve_push(self, graph, divisions, mem_usage, residency):
        """Run the real drain validator, buffer build, CP-SAT solve and push.

        Returns ``(plans, built, solved, editor_cls)``, with buffers by name.
        """
        allocator = CoOptimizingAllocator(
            allocator_module._make_cpsat_solver, size=4096
        )
        with ExitStack() as stack:
            stack.enter_context(self._patches())
            for target, kwargs in (
                ("utils.op_read_writes", {"side_effect": lambda op: self.rw[op]}),
                ("allocator.clone_at_graph_boundaries", {"return_value": True}),
                ("allocator.mem_usage_by_buf", {"return_value": mem_usage}),
                ("allocator.materialize_lx_relayouts", {}),
            ):
                stack.enter_context(
                    patch(f"torch_spyre._inductor.scratchpad.{target}", **kwargs)
                )
            stack.enter_context(
                patch.object(
                    allocator,
                    "_residency_by_buf",
                    side_effect=lambda *a, **k: residency,
                )
            )
            stack.enter_context(
                patch.object(
                    allocator, "_cd_parent_relayouts", side_effect=lambda *a: {}
                )
            )
            stack.enter_context(patch.object(allocator, "_set_one_allocation"))
            editor_cls = stack.enter_context(
                patch.object(allocator_module, "GraphEditor")
            )
            # The validator asks the real editor which collectives it rewrites.
            editor_cls.collective_operand_name.side_effect = (
                GraphEditor.collective_operand_name
            )

            plans = allocator_module.validated_drain_plans(
                graph, division_is_fixed=False
            )
            allocator._validated_drain_plans = plans
            built = allocator._build_cd_bound_buffers(
                graph, {}, allocator_module._DivisionMap(divisions, set())
            )
            solver = allocator.layout_planning(built, allocator.size)
            solved = {b.name: b for b in solver.plan_layout()}
            # commit_divisions would record the committed ownership here.
            for op in graph.operations:
                op.iteration_space_ownership = object()
            allocator._push_allocation(graph, list(solved.values()), [])
        return plans, {b.name: b for b in built}, solved, editor_cls

    def test_carry_read_gate_decides_whether_the_post_loop_drain_is_emitted(self):
        """A drained carry under the carry-read gate: resident only if readers agree.

        The carry of :meth:`_drained_carry_graph` is returned from the graph,
        and ``tail`` runs after the loop. The post-loop drain plan makes this
        mutated graph output eligible for LX, which is exactly when the read
        edge of ``consumer`` starts to matter. The real drain validator, buffer
        build, CP-SAT solve and push run on it:

        * the reader can slice the storage the way the storage is owned: the
          storage is resident, stays live to the graph exit, and the push emits
          one drain clone after the loop's last member;
        * every reader division slices it another way (same core count, other
          axes, as in #4990): the edge admits no pair, the solver keeps the
          storage in HBM, and the push emits nothing.
        """
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except ImportError:
            self.skipTest("the joint path needs the CP-SAT solver (ortools)")

        tail_op = self._op("tail")
        storage_op = self.op_by_name["plain"]
        graph, ops = self._drained_carry_graph([tail_op], [storage_op])
        self.rw[tail_op] = self._rw([], ["tail"])
        mem_usage = {
            "plain": {"size": 256, "op_inputs": []},
            # A mutation alias is unsized, as mem_usage_by_buf reports it.
            "update": {"size": -1, "op_inputs": ["plain"]},
            "consumer": {"size": 256, "op_inputs": ["update"]},
            "tail": {"size": 256, "op_inputs": []},
        }
        # The verdicts _residency_by_buf gives these ops: the plan clears the
        # storage's "graph output mutated after production" refusal
        # (TestLoopCarryLxEligibility covers that branch); the update is a
        # mutation alias, never an LX buffer itself.
        residency = {
            "plain": None,
            "update": "op not allowed",
            "consumer": None,
            "tail": None,
        }

        def run(reader_divs):
            divisions = {
                "plain": self.parent_divs,
                "update": self.parent_divs,
                "consumer": reader_divs,
                "tail": self.parent_divs[:1],
            }
            plans, built, solved, editor_cls = self._plan_solve_push(
                graph, divisions, mem_usage, residency
            )
            self.assertEqual(set(plans), {"plain"})
            self.assertIs(plans["plain"].anchor_op, self.consumer_op)
            self.assertIsNone(plans["plain"].collective)
            return plans["plain"], built, solved, editor_cls

        with self.subTest("reader splits like the storage"):
            plan, built, solved, editor_cls = run(self.consumer_divs[:2])
            storage = solved["plain"]
            self.assertIsNotNone(storage.address)
            editor = editor_cls.return_value
            editor.push_allocation_with_clone.assert_called_once_with(
                storage_op,
                [],
                input=False,
                private=True,
                after_fx=plan.loop_origin,
                lower_anchor=plan.anchor_op,
            )
            drain = editor.push_allocation_with_clone.return_value
            editor.change_graph_output.assert_called_once_with(storage_op, drain)
            # Why: the reader's edge on the storage admits the identical
            # slicings, and the solver committed one of them.
            matches = built["consumer"].cd_parent_matches.get("plain")
            self.assertEqual(matches, [(0, 0), (1, 1)])
            self.assertIn(
                (storage.chosen_division, solved["consumer"].chosen_division), matches
            )
            # The drain plan keeps the storage live to the graph exit (4 ops),
            # past the loop's end (3) that the counted-loop rule alone gives.
            self.assertEqual(built["plain"].end_time, len(ops))

        with self.subTest("reader splits the storage differently"):
            plan, built, solved, editor_cls = run([self.consumer_divs[3]])
            self.assertIsNone(solved["plain"].address)
            editor = editor_cls.return_value
            editor.push_allocation_with_clone.assert_not_called()
            editor.change_graph_output.assert_not_called()
            # Why: the reader's edge on the storage exists but admits no pair.
            self.assertEqual(built["consumer"].cd_parent_matches.get("plain"), [])

    def test_post_loop_all_reduce_reads_a_drain_of_the_resident_carry(self):
        """A carry whose only post-loop use is an all_reduce stays in LX.

        The all_reduce's whole-buffer read has no residency edge, so as a
        parent of the all_reduce the carry admits no division pair, which
        forces it -- and every in-loop update -- into HBM.  The plan gives the
        all_reduce a post-loop drain to read instead: the carry is no longer
        its parent, the carry is resident, the push repoints the all_reduce at
        the drain, and the carry's lifetime still ends at the all_reduce
        rather than the graph exit (``tail`` runs after it).
        """
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except ImportError:
            self.skipTest("the joint path needs the CP-SAT solver (ortools)")
        from torch._inductor.ir import StorageBox, TensorBox

        from torch_spyre._inductor.ir import AllReduceAsyncFallback

        storage_op = self.op_by_name["plain"]
        all_reduce = MagicMock(spec=AllReduceAsyncFallback)
        all_reduce.get_name.return_value = "all_reduce"
        all_reduce.inputs = [TensorBox(StorageBox(storage_op))]
        tail_op = self._op("tail")
        graph, ops = self._drained_carry_graph([all_reduce, tail_op], [tail_op])
        # An extern kernel reads and writes whole buffers.
        self.rw[all_reduce] = SimpleNamespace(
            reads=OrderedSet([StarDep("plain")]),
            writes=OrderedSet([StarDep("all_reduce")]),
        )
        self.rw[tail_op] = self._rw(["all_reduce"], ["tail"])
        mem_usage = {
            "plain": {"size": 256, "op_inputs": []},
            "update": {"size": -1, "op_inputs": ["plain"]},
            "consumer": {"size": 256, "op_inputs": ["update"]},
            # An extern kernel's FixedLayout output is unsized.
            "all_reduce": {"size": -1, "op_inputs": ["plain"]},
            "tail": {"size": 256, "op_inputs": ["all_reduce"]},
        }
        residency = {
            "plain": None,
            "update": "op not allowed",
            "consumer": None,
            "all_reduce": "op not allowed",
            "tail": None,
        }
        divisions = {
            "plain": self.parent_divs,
            "update": self.parent_divs,
            "consumer": self.consumer_divs[:2],
            "all_reduce": [CoreDivision(splits={})],
            "tail": self.parent_divs[:1],
        }

        def run(with_plan):
            real_plans = allocator_module.validated_drain_plans
            plans_fn = real_plans if with_plan else (lambda *a, **k: {})
            with patch.object(allocator_module, "validated_drain_plans", plans_fn):
                return self._plan_solve_push(graph, divisions, mem_usage, residency)

        with self.subTest("the collective is drained"):
            plans, built, solved, editor_cls = run(with_plan=True)
            self.assertEqual(set(plans), {"plain"})
            plan = plans["plain"]
            self.assertIs(plan.collective, all_reduce)
            self.assertIsNotNone(solved["plain"].address)
            self.assertNotIn("plain", built["all_reduce"].parents)
            # Live to the all_reduce (index 3), not to the graph exit (5).
            self.assertEqual(built["plain"].end_time, ops.index(all_reduce) + 1)
            editor = editor_cls.return_value
            editor.push_allocation_with_clone.assert_called_once_with(
                storage_op,
                [],
                input=False,
                private=True,
                after_fx=plan.loop_origin,
                lower_anchor=plan.anchor_op,
            )
            drain = editor.push_allocation_with_clone.return_value
            editor.replace_collective_operand.assert_called_once_with(
                all_reduce, "plain", drain
            )
            editor.change_graph_output.assert_not_called()

        with self.subTest("without a plan the all_reduce edge keeps it in HBM"):
            _plans, built, solved, editor_cls = run(with_plan=False)
            self.assertIsNone(solved["plain"].address)
            self.assertIn("plain", built["all_reduce"].parents)
            self.assertNotIn("plain", built["all_reduce"].cd_parent_matches)
            editor_cls.return_value.replace_collective_operand.assert_not_called()

    @staticmethod
    def _compatible(edge, parent_div, consumer_div):
        """Per-pair reimplementation of what ``match_pairs`` computes in
        batch, exercised against the same splits-dict API."""
        if edge._cores_used(parent_div) != edge._cores_used(consumer_div):
            return False
        parent_view = edge.parent_view(parent_div)
        if parent_view is None:
            return False
        consumer_view = edge.consumer_view(consumer_div)
        return consumer_view is not None and parent_view.same_partition(consumer_view)

    def test_compatible_agrees_with_the_table(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with self._patches():
            table = self._table(allocator)
            for parent, pairs in table.items():
                edge = work_division_module.build_residency_edge(
                    parent,
                    self.op_by_name[parent],
                    self.consumer_op,
                    self.rw[self.consumer_op].reads,
                    self.residency[parent],
                    {},
                )
                for i, parent_div in enumerate(self.divisions[parent]):
                    for j, consumer_div in enumerate(self.consumer_divs):
                        self.assertEqual(
                            self._compatible(edge, parent_div, consumer_div),
                            (i, j) in pairs,
                            f"{parent} ({i}, {j})",
                        )

    def test_expanding_clone_cannot_read_an_unreplicated_lx_input(self):
        producer = self.op_by_name["plain"]
        clone = self.op_by_name["clone"]
        self.divisions["clone"] = [self.consumer_divs[2]]
        read = self.rw[clone].reads[0].rename({"src": "plain"})
        with self._patches():
            edge = allocator_module.build_residency_edge(
                "plain", producer, clone, [read], None, {}
            )
            self.assertIsNotNone(edge)
            # Four owners cannot directly serve eight consumer cores.
            self.assertEqual(
                edge.match_pairs([self.parent_divs[0]], [self.consumer_divs[2]]),
                [],
            )

    def test_excluded_edges_have_no_edge_object(self):
        with self._patches():
            for parent, reason in [
                ("spilled", "residency"),
            ]:
                self.assertIsNone(
                    work_division_module.build_residency_edge(
                        parent,
                        self.op_by_name[parent],
                        self.consumer_op,
                        self.rw[self.consumer_op].reads,
                        self.residency[parent],
                        {},
                    ),
                    reason,
                )

    def test_residency_edge_requires_coordinate_dependencies(self):
        producer = self.op_by_name["plain"]
        memory = self.rw[producer].writes[0]
        for dependency in (StarDep("plain"), WeakDep("plain", "consumer")):
            for reads in ([dependency], [dependency, memory]):
                for writes in ([dependency], [dependency, memory]):
                    with self.subTest(reads=reads, writes=writes):
                        with (
                            self._patches(),
                            patch.object(self.rw[producer], "writes", writes),
                        ):
                            edge = allocator_module.build_residency_edge(
                                "plain", producer, self.consumer_op, reads, None, {}
                            )
                        if memory in reads and memory in writes:
                            self.assertEqual(edge.read_deps, (memory,))
                            self.assertIs(edge.write_dep, memory)
                        else:
                            self.assertIsNone(edge)

    def test_no_consumer_op_matches_nothing(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with self._patches():
            edges = allocator._parent_residency_edges(None, [], {}, {}, self.residency)
            self.assertEqual(allocator._cd_parent_matches(edges, [], {}), {})


class TestCloneDivisionMatching(unittest.TestCase):
    """The clone-in seam: the pairs a graph input's synthesized menu admits.

    The sibling of :class:`TestResidencyEdgeMatching` for the one edge with no
    producer: a clone's view *is* its consumer's, so nothing compares two views
    here and the broadcast check in ``_clone_divisions_and_matches`` is the
    only thing standing between a broadcast read and a plan ``_post_solve`` can
    only reject.
    """

    def setUp(self):
        x, y = _isym("x"), _isym("y")
        # One consumer, three candidates. The middle one splits ``y``, an axis
        # the input does not carry, so the split contracts out of the view and
        # all four cores read the whole buffer.
        self.consumer_divs = [
            CoreDivision(splits={x: 4}),
            CoreDivision(splits={y: 4}),
            CoreDivision(splits={x: 2}),
        ]
        self.views = [
            _physical_view((0, 4)),
            PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=4),
            _physical_view((0, 2)),
        ]
        self.consumer = MagicMock(spec=ComputedBuffer)
        self.consumer.get_name.return_value = "consumer"
        self.rw = MagicMock(
            reads=[MemoryDep("inp", x, (x,), (8,))],
            writes=[MemoryDep("consumer", x, (x,), (8,))],
        )

    def _view_for_div(self, op, dep, buf_name, division, prep_cache):
        index = [cd.splits for cd in self.consumer_divs].index(division.splits)
        return (self.views[index], False, True)

    def _menu(self):
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        with ExitStack() as stack:
            for target, kwargs in [
                ("_view_for_div", {"side_effect": self._view_for_div}),
                ("op_read_writes", {"return_value": self.rw}),
            ]:
                stack.enter_context(
                    patch(
                        f"torch_spyre._inductor.scratchpad.allocator.{target}",
                        **kwargs,
                    )
                )
            return allocator._clone_divisions_and_matches(
                "inp", [self.consumer], {"consumer": self.consumer_divs}, {}
            )

    def test_broadcast_read_reaches_neither_the_menu_nor_the_table(self):
        divs, matches = self._menu()
        self.assertEqual(
            [cd.splits for cd in divs], [{_isym("x"): split} for split in (4, 2)]
        )
        self.assertEqual(matches, {"consumer": [(0, 0), (1, 2)]})


class TestCoOptimizingAllocator(unittest.TestCase):
    def test_deferred_direct_read_restickify_keeps_committed_division(self):
        head, sequence = _isym("head"), _isym("sequence")
        op = _computed_buffer((2, 128, 1024), name="direct_read_restickify")
        op._read_copy_elision_record = MagicMock()
        graph = MagicMock(operations=[op])
        allocator = CoOptimizingAllocator(MagicMock(), size=1)
        fixed = CoreDivision(splits={head: 2, sequence: 16})

        with (
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "ops_in_offset_mutation_component",
                return_value=set(),
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._fused_layout_group_ops",
                return_value={},
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator."
                "_find_distinct_matmul_splits",
                return_value=((), ()),
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator.is_restickify_op",
                return_value=True,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._fixed_core_division",
                return_value=fixed,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=True,
            ),
            patch.object(allocator, "_enumerate_core_divisions") as enumerate_divs,
        ):
            divisions = allocator._division_map(graph).divisions

        self.assertEqual(divisions[op.name], [fixed])
        enumerate_divs.assert_not_called()

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
            # ``_core_division`` reads the write dep from ``work_division``.
            patch.object(work_division_module, "op_read_writes", return_value=rw),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_fits_sticks",
                return_value=True,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                side_effect=lambda _op, splits: splits == safe,
            ) as is_legal,
        ):
            division_map = allocator._division_map(graph)
            divisions = division_map.divisions[op.name]

        self.assertEqual(divisions, [CoreDivision(splits={m: 8})])
        self.assertEqual(is_legal.call_args_list[0].args[1], safe)
        self.assertEqual(is_legal.call_args_list[1].args[1], unsafe)

        op.iteration_space_ownership = MagicMock()
        allocation = [
            CoreDivisionBuffer(
                name=op.name,
                size=128,
                uses=[0],
                core_divisions=[CoreDivision(splits={batch: 4})],
                chosen_division=0,
            )
        ]
        with (
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=False,
            ),
            self.assertRaisesRegex(Unsupported, "chosen split violates hard domain"),
        ):
            allocator_module.commit_divisions(graph, allocation)

    def test_no_enumerable_candidates_keeps_legal_fixed_division(self):
        op = MagicMock(spec=ComputedBuffer)
        op.name = "empty_candidates"
        op.data = MagicMock(spec=Pointwise)
        rw = MagicMock()
        rw.writes = [MagicMock(index=0)]
        rw.reads = []
        allocator = CoOptimizingAllocator(MagicMock(), size=1)

        fixed = CoreDivision(splits={1: 2})
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
                "torch_spyre._inductor.scratchpad.allocator."
                "enumerate_work_division_candidates",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=True,
            ),
        ):
            # Not an enumeration, so a solver may not generate divisions for
            # this op: the committed one is all it is allowed.
            self.assertEqual(
                allocator._enumerate_core_divisions(op, max_cores=32),
                ([fixed], False),
            )

    def test_over_budget_candidate_menu_is_rejected(self):
        """An over-budget division must not reach the menu. Nothing downstream
        catches one: both engines pin an op's split symbols to a single enumerated
        candidate, and ``_matmul_split_cost``'s budget guard no-ops on symbolic
        splits, so an over-budget candidate is scored NEGATIVE and a minimizing
        solve prefers it (issue #4387). The prune path is the one that could admit
        one -- ``_legal_split_options`` tests stick validity and hard domains, never
        a core budget -- so it is the one exercised here.
        """
        batch, m = _isym("batch"), _isym("m")
        op = _computed_buffer((4, 64), name="over_budget_out")
        graph = MagicMock(operations=[op])
        allocator = CoOptimizingAllocator(MagicMock(), size=1, prune=True)
        over_budget = {batch: 4, m: 16}  # 64 cores against a 32-core budget
        rw = MagicMock(
            writes=[MemoryDep(op.name, 64 * batch + m, (batch, m), (4, 64))],
            reads=[],
        )

        with (
            patch.object(allocator_module.config, "sencores", 32),
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
                return_value=[over_budget],
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator.op_read_writes",
                return_value=rw,
            ),
            # ``_core_division`` reads the write dep from ``work_division``.
            patch.object(work_division_module, "op_read_writes", return_value=rw),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_fits_sticks",
                return_value=True,
            ),
            patch(
                "torch_spyre._inductor.scratchpad.allocator._split_option_is_legal",
                return_value=True,
            ),
            self.assertRaisesRegex(
                AssertionError, r"over_budget_out: .*over the 32-core budget"
            ),
        ):
            allocator._division_map(graph)


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


def _division_key(division):
    """A division as comparable literals -- symbols are unorderable."""
    return (
        tuple(sorted(_by_name(division.output_splits).items())),
        tuple(sorted(_by_name(division.reduction_splits).items())),
    )


@contextmanager
def _space_for(case):
    """The generated split space for one candidate case, under its patches."""
    with case.patches():
        yield work_division_module.build_op_split_space(case.op, case.max_cores)


class TestOpSplitSpace(unittest.TestCase):
    """The generation seam: a space that admits exactly what the enumeration
    carries, and a move alphabet over it."""

    def test_space_admits_exactly_the_enumerated_candidates(self):
        """Generation changes when a candidate is materialized, not which
        candidates exist -- so the space and the menu have to agree, over the
        corpus that exercises every rule a candidate is judged by."""
        narrowed = []
        for case in _candidate_cases():
            with self.subTest(case.name):
                with _space_for(case) as space:
                    admitted = [
                        splits
                        for combo in itertools.product(
                            *(space.factor_domains[axis] for axis in space.axes)
                        )
                        if space.admits(splits := dict(zip(space.axes, combo)))
                    ]
                    whole_product = math.prod(
                        len(space.factor_domains[axis]) for axis in space.axes
                    )
                self.assertEqual(
                    [_by_name(s) for s in admitted],
                    [_by_name(c) for c in case.candidates],
                )
                narrowed.append(len(admitted) < whole_product)
        # At least one case must be narrowed by the whole-split rules rather
        # than by the per-axis domains alone.
        self.assertTrue(any(narrowed))

    def test_space_division_agrees_with_the_classifier(self):
        """:meth:`OpSplitSpace.division` derives the output/reduction roles once
        instead of per candidate; it owes the same answer as the classifier the
        menu is built with."""
        reductions = 0
        for case in _candidate_cases():
            with self.subTest(case.name):
                with _space_for(case) as space:
                    for splits in case.candidates:
                        expected = work_division_module._core_division(case.op, splits)
                        actual = space.division(splits)
                        self.assertEqual(_division_key(actual), _division_key(expected))
                        reductions += bool(expected.reduction_splits)
        self.assertGreater(reductions, 0, "no case splits a reduction axis")

    def test_neighbours_are_the_one_axis_moves_inside_the_space(self):
        local = []
        for case in _candidate_cases():
            with self.subTest(case.name):
                with _space_for(case) as space:
                    for splits in case.candidates:
                        expected = {
                            _division_key(space.division(other))
                            for other in case.candidates
                            if sum(other[axis] != splits[axis] for axis in space.axes)
                            == 1
                        }
                        actual = {
                            _division_key(division)
                            for division in space.neighbours(space.division(splits))
                        }
                        self.assertEqual(actual, expected, _by_name(splits))
                        local.append(len(expected) < len(case.candidates) - 1)
        # A move alphabet that reached every candidate from every candidate
        # would not be a local one, and the test would say nothing.
        self.assertTrue(any(local))

    def test_no_space_where_the_menu_would_carry_one_candidate(self):
        """The ops generation has nothing to offer are exactly the ops
        ``_enumerate_core_divisions`` leaves at their committed division."""
        not_a_buffer = MagicMock()
        other_data = MagicMock(spec=ComputedBuffer)
        other_data.data = MagicMock()
        for op in (not_a_buffer, other_data):
            self.assertIsNone(work_division_module.build_op_split_space(op, 32))
        case = next(c for c in _candidate_cases() if c.name == "two_dims")
        with patch.object(
            work_division_module,
            "work_division_context_for_op",
            side_effect=Unsupported("no iteration space"),
        ):
            self.assertIsNone(
                work_division_module.build_op_split_space(case.op, case.max_cores)
            )


_SPEC_ON_DIM_0 = TileSpec((TileAxis(host_dim=0, count=2),))
_SPEC_ON_DIM_1 = TileSpec((TileAxis(host_dim=1, count=2),))


class TestOpSplitSpacePerTileFrame(unittest.TestCase):
    """A tiled division is judged in its per-tile frame. #4768's end-to-end
    shape: fp16 ``(1, 8195, 256, 64)`` on 4 cores overflows ``MAX_SPAN_BYTES``
    at every untiled split -- the span is measured on dim 1, and no factor of
    8195 = 5 * 11 * 149 fits 4 cores -- while dim 1 tiled 5 ways fits."""

    def test_a_tiling_admits_splits_the_untiled_op_cannot_take(self):
        shape = (1, 8195, 256, 64)
        op = _computed_buffer(shape)
        syms = [Symbol(f"d{i}", integer=True, nonnegative=True) for i in range(4)]
        index = sum(sym * stride for sym, stride in zip(syms, op.layout.stride))
        write = MemoryDep("buf0", index, tuple(syms), shape)
        op.get_read_writes = MagicMock(
            return_value=SimpleNamespace(reads=set(), writes={write})
        )
        space = work_division_module.build_op_split_space(
            op, 4, tiling=TilingSpace(max_dims=1, output_counts={1: [5]})
        )
        by_5 = TileSpec((TileAxis(host_dim=1, count=5),))

        def admitted(tiling):
            domains = [space.factor_domain(axis, tiling) for axis in space.axes]
            return [
                list(factors)
                for factors in itertools.product(*domains)
                if space.admits(dict(zip(space.axes, factors)), tiling)
            ]

        self.assertEqual(admitted(TileSpec()), [])
        self.assertEqual(admitted(by_5), [[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 4, 1]])


class TestOpSplitSpaceTiling(unittest.TestCase):
    """The tiling half of the space: a coarse tiling and a core division are
    one candidate, and the tiling narrows what the division may be."""

    def setUp(self):
        self.x, self.y = _isym("x"), _isym("y")
        # Host dim 0 has extent 8 -- four divisors untiled, fewer per tile.
        self.op = _computed_buffer((8, 128), name="tiled")
        self.case = _CandidateCase(
            name="tiling",
            op=self.op,
            it_space={self.x: 8, self.y: 128},
            output_td=_tensor_dep("tiled", (8, 128), (self.x, self.y)),
            max_cores=32,
            axes=(self.x, self.y),
            candidates=[],
            probes=[],
        )
        # The write dep the enumerator resolves the stick dim through.
        self.op.get_read_writes = MagicMock(
            return_value=SimpleNamespace(reads=set(), writes={self.case.output_td.dep})
        )
        self.tiling = TilingSpace(max_dims=2, output_counts={0: [2, 4]})

    @contextmanager
    def _space(self, tiling, coords=None):
        coords = [self.x, self.y] if coords is None else coords
        with (
            self.case.patches(),
            patch.object(work_division_module, "op_out_coords", return_value=coords),
            patch.object(coarse_tiling_module, "op_out_coords", return_value=coords),
            patch.object(
                coarse_tiling_module,
                "iteration_space_from_op",
                return_value=self.case.it_space,
            ),
        ):
            yield work_division_module.build_op_split_space(
                self.op, self.case.max_cores, tiling=tiling
            )

    def test_the_tiling_half_agrees_with_the_enumeration(self):
        """The seam ``test_space_admits_exactly_the_enumerated_candidates``
        pins for the split half: what the list carries, the space admits."""
        options = enumerate_tile_options(self.op, max_options=1000)
        self.assertGreater(len(options), 1)  # non-vacuity
        with self._space(build_tiling_space(self.op)) as space:
            for spec in options:
                self.assertTrue(space.admits_tiling(spec), spec.label)

    def test_a_dim_the_coords_do_not_resolve_is_not_offered(self):
        """``tile_counts`` could not narrow the axis such a dim cuts."""
        with self._space(self.tiling, coords=[sympy.Integer(0), self.y]) as space:
            self.assertTrue(space.tiling.is_empty)
            self.assertFalse(space.admits_tiling(_SPEC_ON_DIM_0))

    def test_a_tile_level_narrows_the_axis_it_cuts(self):
        """The ragged half: coarse tiling emits equal tiles, so a core split of
        a tiled axis has to divide the *per-tile* extent."""
        tiled = TileSpec((TileAxis(host_dim=0, count=4),))
        with self._space(self.tiling) as space:
            self.assertEqual(space.factor_domain(self.x), [1, 2, 4, 8])
            self.assertEqual(space.factor_domain(self.x, tiled), [1, 2])
            # The untouched axis keeps its whole domain.
            self.assertEqual(
                space.factor_domain(self.y, tiled), space.factor_domain(self.y)
            )
            self.assertTrue(space.admits({self.x: 4, self.y: 1}))
            self.assertFalse(space.admits({self.x: 4, self.y: 1}, tiled))
            self.assertTrue(space.admits({self.x: 2, self.y: 1}, tiled))

    def test_a_tiling_the_op_cannot_take_is_refused_with_its_splits(self):
        with self._space(self.tiling) as space:
            self.assertFalse(space.admits({self.x: 1, self.y: 1}, _SPEC_ON_DIM_1))
            self.assertFalse(space.admits_tiling(_SPEC_ON_DIM_1))

    def test_a_step_moves_one_axis_or_one_level_but_never_both(self):
        untiled = TileSpec()
        with self._space(self.tiling) as space:
            seed = space.division({self.x: 2, self.y: 1})
            moves = space.neighbours(seed)
            for division in moves:
                changed_tiling = division.tiling != untiled
                changed_splits = division.output_splits != seed.output_splits
                self.assertNotEqual(changed_tiling, changed_splits, division.label)
            # Both kinds are offered, and a tiling step keeps the splits.
            self.assertIn(
                (2, TileSpec((TileAxis(host_dim=0, count=2),))),
                [(d.output_splits.get(self.x, 1), d.tiling) for d in moves],
            )
            self.assertIn(4, [d.output_splits.get(self.x, 1) for d in moves])

    def test_the_split_steps_at_a_tiling_stay_inside_its_narrower_domain(self):
        tiled = TileSpec((TileAxis(host_dim=0, count=4),))
        with self._space(self.tiling) as space:
            at_tiling = space.neighbours(space.division({self.x: 1}, tiled))
            for division in at_tiling:
                if division.tiling == tiled:
                    self.assertIn(division.output_splits.get(self.x, 1), (1, 2))

    def test_without_a_tiling_space_nothing_is_tiled_and_nothing_is_offered(self):
        """What every engine but the SA co-optimizer gets: the answers are
        the ones a space that never knew about tilings gave."""
        with self._space(None) as space:
            self.assertFalse(space.admits_tiling(_SPEC_ON_DIM_0))
            self.assertFalse(space.admits({self.x: 1}, _SPEC_ON_DIM_0))
            self.assertEqual(space.tiling_options(TileSpec()), [])
            seed = space.division({self.x: 2, self.y: 1})
            self.assertTrue(all(d.tiling.is_untiled for d in space.neighbours(seed)))


class TestAxisByHostDim(unittest.TestCase):
    """``TileAxis.host_dim`` indexes ``op_out_coords``, and the axis a level
    cuts is read off it by the lowering's own resolver."""

    def setUp(self):
        self.d0, self.d1, self.d2 = _isym("d0"), _isym("d1"), _isym("d2")

    def _mapping(self, coords, axes):
        with (
            patch.object(work_division_module, "op_out_coords", return_value=coords),
            patch.object(coarse_tiling_module, "op_out_coords", return_value=coords),
            patch.object(
                coarse_tiling_module,
                "iteration_space_from_op",
                return_value=dict.fromkeys(axes, 1),
            ),
        ):
            return work_division_module._axis_by_host_dim(MagicMock(), axes)

    def test_a_size_1_dim_shifts_the_frame(self):
        # No symbol is minted for a size-1 dim, so host 1 is the axis ``d0``
        # cuts; a positional frame would map it to ``d1``.
        axes = [self.d0, self.d1, self.d2]
        self.assertEqual(
            self._mapping([sympy.Integer(0), self.d0, self.d1, self.d2], axes),
            {1: self.d0, 2: self.d1, 3: self.d2},
        )

    def test_only_a_single_known_axis_resolves(self):
        other = _isym("other")
        self.assertEqual(
            self._mapping([self.d0 * 8 + self.d1, other, self.d2], [self.d0, self.d2]),
            {2: self.d2},
        )


class TestResidencyEdgeInversion(unittest.TestCase):
    """Propagating a division across an edge by *constructing* the other end's
    division instead of scanning its menu for a compatible entry."""

    def setUp(self):
        self.x, self.y, self.k = _isym("x"), _isym("y"), _isym("k")
        self.r, self.c = _isym("r"), _isym("c")
        shape = (8, 128)  # 128 fp16 elements = 2 sticks, so both dims can split
        self.producer = _computed_buffer(shape, name="p")
        self.consumer = _computed_buffer(shape, name="cons")
        layout = _fixed_tiled_layout(shape)
        self.write_dep = MemoryDep("p", 128 * self.x + self.y, (self.x, self.y), shape)
        self.read_dep = MemoryDep("p", 128 * self.r + self.c, (self.r, self.c), shape)
        consumer_write = MemoryDep(
            "cons", 128 * self.r + self.c, (self.r, self.c), shape
        )
        # The producer carries a reduction axis its buffer does not see; the
        # consumer names its two axes differently. Both are what makes the
        # inverse a real inverse rather than a rename.
        self.iter_spaces = {
            "p": {self.x: 8, self.y: 128, self.k: 4},
            "cons": {self.r: 8, self.c: 128},
        }
        self.read_writes = {
            "p": MagicMock(writes=[self.write_dep], reads=[]),
            "cons": MagicMock(writes=[consumer_write], reads=[self.read_dep]),
        }
        self.graph = SimpleNamespace(
            _repeat_info={}, get_buffer=lambda name: SimpleNamespace(layout=layout)
        )
        self.parent_space = mock_op_split_space(
            {self.x: [1, 2, 4, 8], self.y: [1, 2], self.k: [1, 2, 4]},
            {self.x, self.y},
            op=self.producer,
        )
        self.consumer_space = mock_op_split_space(
            {self.r: [1, 2, 4, 8], self.c: [1, 2]},
            {self.r, self.c},
            op=self.consumer,
        )

    def _tiled_consumer_space(self, counts):
        return mock_op_split_space(
            {self.r: [1, 2, 4, 8], self.c: [1, 2]},
            {self.r, self.c},
            op=self.consumer,
            tiling=TilingSpace(max_dims=2, output_counts=counts),
        )

    def test_the_tiling_crosses_the_edge_where_the_far_side_can_take_it(self):
        """What forms a tiling group at all: the run of ops the residency
        relation reaches has to agree on one ``TileSpec``, so the inverse
        carries it rather than re-deciding on each side."""
        tiled = TileSpec((TileAxis(host_dim=0, count=2),))
        consumer_space = self._tiled_consumer_space({0: [2]})
        with self._geometry():
            division = self._edge().consumer_division_for(
                CoreDivision({self.x: 4}, tiling=tiled), consumer_space
            )
        self.assertIsNotNone(division)
        self.assertEqual(division.tiling, tiled)
        self.assertEqual(_by_name(division.output_splits), {"r": 4})

    def test_a_tiling_the_far_side_refuses_costs_the_level_not_the_edge(self):
        """Losing a tiling level is a worse plan; losing the edge is a worse
        state. So the untiled inverse is taken, and the edge survives."""
        tiled = TileSpec((TileAxis(host_dim=0, count=2),))
        consumer_space = self._tiled_consumer_space({0: [4]})  # 2 is not offered
        self.assertFalse(consumer_space.admits_tiling(tiled))  # non-vacuity
        with self._geometry():
            division = self._edge().consumer_division_for(
                CoreDivision({self.x: 4}, tiling=tiled), consumer_space
            )
        self.assertIsNotNone(division)
        self.assertTrue(division.tiling.is_untiled)
        self.assertEqual(_by_name(division.output_splits), {"r": 4})

    def _edge(self):
        return work_division_module.ResidencyEdge(
            buf_name="p",
            parent_op=self.producer,
            consumer_op=self.consumer,
            write_dep=self.write_dep,
            read_deps=(self.read_dep,),
            prep_cache={},
        )

    def _geometry(self):
        stack = ExitStack()
        stack.enter_context(pass_utils_module.V.set_graph_handler(self.graph))
        stack.enter_context(
            patch.object(
                pass_utils_module,
                "iteration_space_from_op",
                side_effect=lambda op: self.iter_spaces[op.get_name()],
            )
        )
        for module in (pass_utils_module, work_division_module):
            stack.enter_context(
                patch.object(
                    module,
                    "op_read_writes",
                    side_effect=lambda op: self.read_writes[op.get_name()],
                )
            )
        return stack

    def test_inverse_builds_the_other_end_of_the_edge(self):
        cases = [
            (CoreDivision({self.x: 4}), {"r": 4}),
            (CoreDivision({self.x: 4, self.y: 2}), {"r": 4, "c": 2}),
            (CoreDivision(), {}),
        ]
        with self._geometry():
            edge = self._edge()
            for parent_division, expected in cases:
                consumer_division = edge.consumer_division_for(
                    parent_division, self.consumer_space
                )
                self.assertIsNotNone(consumer_division, parent_division.label)
                self.assertEqual(_by_name(consumer_division.output_splits), expected)
                self.assertTrue(
                    edge.compatible(parent_division.splits, consumer_division.splits)
                )
                # And back: the mirror recovers the division it came from.
                self.assertEqual(
                    _division_key(
                        edge.parent_division_for(consumer_division, self.parent_space)
                    ),
                    _division_key(parent_division),
                )

    def test_a_partial_reduction_producer_hosts_nothing(self):
        """The write side's policy filters, which the geometry is blind to: a
        reduction-split producer leaves partial sums, so there is no division
        the consumer could read from LX."""
        with self._geometry():
            self.assertIsNone(
                self._edge().consumer_division_for(
                    CoreDivision(
                        splits={self.x: 4, self.k: 2},
                        reduction_syms=frozenset({self.k}),
                    ),
                    self.consumer_space,
                )
            )

    def test_compatibility_compares_partitions_not_records(self):
        """A view is a record of a slicing, and two records can describe one
        slicing -- so the edge asks ``same_partition``, not ``==``. Here the
        consumer's view is restated with its dims in the other order, which
        ``==`` calls a mismatch and the buffer's geometry does not."""
        parent_division = CoreDivision({self.x: 4, self.y: 2})
        consumer_division = CoreDivision({self.r: 4, self.c: 2})
        with self._geometry():
            edge = self._edge()
            view = edge.consumer_view(consumer_division)
            self.assertEqual(len(view.work_slice_dims), 2)
            restated = dataclasses.replace(
                view,
                work_slice_dims=view.work_slice_dims[::-1],
                core_to_slot=view.core_to_slot[::-1],
            )
            self.assertNotEqual(restated, view)
            with patch.object(
                work_division_module.ResidencyEdge,
                "consumer_view",
                return_value=restated,
            ):
                self.assertTrue(
                    edge.compatible(parent_division.splits, consumer_division.splits)
                )
                self.assertEqual(
                    edge.match_pairs([parent_division], [consumer_division]),
                    [(0, 0)],
                )

    def test_the_write_side_policy_rides_along_inside_the_inversion(self):
        """The producer's filters are geometry-blind, so they go in ``accept``
        rather than on the answer: a partial-reduction solution then backtracks
        to the next geometric one (``invert_per_core_view`` pins that a
        rejected candidate backtracks) instead of losing the edge outright --
        which ``_ViewRelation`` would memoize for the whole solve."""
        captured: list = []
        real = work_division_module.invert_per_core_view

        def spy(prep, target, domains, **kwargs):
            captured.append(kwargs["accept"])
            return real(prep, target, domains, **kwargs)

        with (
            self._geometry(),
            patch.object(work_division_module, "invert_per_core_view", spy),
        ):
            edge = self._edge()
            edge.parent_division_for(CoreDivision({self.r: 4}), self.parent_space)
            up = captured.pop()
            edge.consumer_division_for(CoreDivision({self.x: 4}), self.consumer_space)
            down = captured.pop()
            self.assertTrue(up({self.x: 4, self.y: 1, self.k: 1}))
            self.assertFalse(up({self.x: 4, self.y: 1, self.k: 2}))
            # The consumer's side holds a candidate to every read of the
            # buffer; with one read, that is only representability.
            self.assertTrue(down({self.r: 4, self.c: 2}))

    def test_an_illegal_candidate_loses_the_edge_rather_than_being_taken(self):
        """``admits`` rides along inside the inversion, so the only division
        that reproduces the geometry being illegal means no edge -- not an
        illegal division."""
        space = self.consumer_space
        space.context.is_legal.side_effect = (
            lambda splits, tile_counts=None: splits[self.r] != 4
        )
        with self._geometry():
            edge = self._edge()
            self.assertIsNone(
                edge.consumer_division_for(CoreDivision({self.x: 4}), space)
            )
            self.assertIsNotNone(
                edge.consumer_division_for(CoreDivision({self.x: 2}), space)
            )


class _TwoReadEdge(NamedTuple):
    """A producer/consumer edge whose consumer reads the buffer twice."""

    shape: tuple[int, ...]
    write_index: sympy.Expr
    read_indices: tuple[sympy.Expr, sympy.Expr]
    parent_domains: dict
    consumer_domains: dict
    consumer_output_axes: frozenset


class TestMultiReadEdgeInversion(unittest.TestCase):
    """Generation agrees with the pair table on an edge whose consumer reads the
    buffer more than once: for every producer division, ``consumer_division_for``
    finds a partner exactly when ``match_pairs`` finds one in the consumer's
    whole space."""

    def _sweep(self, case: _TwoReadEdge) -> set[str]:
        """Check every producer division; return the labels of those that pair."""
        parent_syms = list(case.parent_domains)
        consumer_syms = list(case.consumer_domains)
        sizes = dict(zip(parent_syms, case.shape))
        producer = _computed_buffer(case.shape, name="p")
        consumer = _computed_buffer(case.shape, name="cons")
        write_dep = MemoryDep("p", case.write_index, tuple(parent_syms), case.shape)
        consumer_sizes = {sym: max(case.consumer_domains[sym]) for sym in consumer_syms}
        read_deps = tuple(
            MemoryDep("p", index, tuple(consumer_syms), tuple(consumer_sizes.values()))
            for index in case.read_indices
        )
        iter_spaces = {"p": sizes, "cons": consumer_sizes}
        read_writes = {
            "p": MagicMock(writes=[write_dep], reads=[]),
            # Only which axes the consumer's write sees matters here.
            "cons": MagicMock(
                writes=[
                    MemoryDep("cons", sympy.Add(*case.consumer_output_axes), (), ())
                ],
                reads=list(read_deps),
            ),
        }
        layout = _fixed_tiled_layout(case.shape)
        graph = SimpleNamespace(
            _repeat_info={}, get_buffer=lambda name: SimpleNamespace(layout=layout)
        )
        parent_space = mock_op_split_space(
            case.parent_domains, parent_syms, op=producer
        )
        consumer_space = mock_op_split_space(
            case.consumer_domains, case.consumer_output_axes, op=consumer
        )

        def divisions(space):
            domains = space.factor_domains
            return [
                space.division(dict(zip(domains, factors)))
                for factors in itertools.product(*domains.values())
            ]

        with ExitStack() as stack:
            stack.enter_context(pass_utils_module.V.set_graph_handler(graph))
            stack.enter_context(
                patch.object(
                    pass_utils_module,
                    "iteration_space_from_op",
                    side_effect=lambda op: iter_spaces[op.get_name()],
                )
            )
            for module in (pass_utils_module, work_division_module):
                stack.enter_context(
                    patch.object(
                        module,
                        "op_read_writes",
                        side_effect=lambda op: read_writes[op.get_name()],
                    )
                )
            edge = work_division_module.ResidencyEdge(
                buf_name="p",
                parent_op=producer,
                consumer_op=consumer,
                write_dep=write_dep,
                read_deps=read_deps,
                prep_cache={},
            )
            consumers = divisions(consumer_space)
            paired_labels = set()
            for parent in divisions(parent_space):
                with self.subTest(parent=parent.label):
                    paired = bool(edge.match_pairs([parent], consumers))
                    if paired:
                        paired_labels.add(parent.label)
                    generated = edge.consumer_division_for(parent, consumer_space)
                    self.assertEqual(generated is not None, paired)
                    if generated is not None:
                        self.assertEqual(
                            edge.match_pairs([parent], [generated]), [(0, 0)]
                        )
        return paired_labels

    def test_a_windowed_read_beside_a_plain_one(self):
        """``sum_r a[x0 + r, x1] * a[x0, x1]``: ``r0_0`` and ``x0`` both walk
        rows of the first read, and ``r0_0`` sorts first. Splitting it
        reproduces the first read's slicing but leaves the second unsliced, so
        the inversion has to go on to ``x0``."""
        x, y = _isym("x"), _isym("y")
        r, x0, x1 = _isym("r0_0"), _isym("x0"), _isym("x1")
        paired = self._sweep(
            _TwoReadEdge(
                shape=(8, 128),
                write_index=128 * x + y,
                read_indices=(128 * (x0 + r) + x1, 128 * x0 + x1),
                parent_domains={x: [1, 2, 4, 8], y: [1, 2]},
                consumer_domains={r: [1, 2], x0: [1, 2, 4, 8], x1: [1, 2]},
                consumer_output_axes=frozenset({x0, x1}),
            )
        )
        # Every producer division pairs, through ``x0``.
        self.assertEqual(
            paired,
            {"whole", "sy/2"}
            | {f"sx/{n}" for n in (2, 4, 8)}
            | {f"sx/{n},sy/2" for n in (2, 4, 8)},
        )

    def test_a_transposed_read_beside_a_plain_one(self):
        """``a + a.permute(1, 0, 2)``: only a split of the dim both reads walk
        alike has a partner."""
        x, y, z = _isym("x"), _isym("y"), _isym("z")
        i, j, k = _isym("i"), _isym("j"), _isym("k")
        paired = self._sweep(
            _TwoReadEdge(
                shape=(4, 4, 128),
                write_index=512 * x + 128 * y + z,
                read_indices=(512 * i + 128 * j + k, 512 * j + 128 * i + k),
                parent_domains={x: [1, 2, 4], y: [1, 2, 4], z: [1, 2]},
                consumer_domains={i: [1, 2, 4], j: [1, 2, 4], k: [1, 2]},
                consumer_output_axes=frozenset({i, j, k}),
            )
        )
        self.assertEqual(paired, {"whole", "sz/2"})

    def test_a_gram_matrix_reads_both_ways(self):
        """``x @ x.T``: each read misses the other's row axis entirely."""
        x, y = _isym("x"), _isym("y")
        m, n, k = _isym("m"), _isym("n"), _isym("k")
        paired = self._sweep(
            _TwoReadEdge(
                shape=(8, 128),
                write_index=128 * x + y,
                read_indices=(128 * m + k, 128 * n + k),
                parent_domains={x: [1, 2, 4, 8], y: [1, 2]},
                consumer_domains={m: [1, 2, 4, 8], n: [1, 2, 4, 8], k: [1, 2]},
                consumer_output_axes=frozenset({m, n}),
            )
        )
        self.assertEqual(paired, {"whole", "sy/2"})
