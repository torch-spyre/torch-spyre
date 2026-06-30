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

"""Tests for automatic span-overflow coarse-tiling hints.

These tests intentionally mirror the compiler layers used by user
``spyre_hint`` coarse tiling:

1. Planner: span_overflow_hint_analysis returns a selected dim and split count.
2. Adapter: span_overflow_groups creates a synthetic DimHint/group.
3. Coarse-tile IR: coarse_tile consumes the group and stamps CoarseTileInfo.
4. Scheduler/codegen: generated source contains the expected LoopSpec count.

Coverage in this file:

- no-op behavior for small tensors and non-FixedTiledLayout ops;
- automatic group/DimHint structure, including the reserved hint-id sentinel;
- multiple independent overflowing pointwise ops producing separate groups;
- planner boundary errors when no legal divisor validates post-tile span;
- total-size fallback when no device dim maps through stride_map;
- adapter mapping with both constant and symbolic batch output coordinates;
- coarse_tile stamping of ranges/layout/CoarseTileInfo;
- equivalence between auto span-overflow hints and manual spyre_hint codegen.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sympy
import torch
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.coarse_tile import (
    _SPAN_OVERFLOW_HINT_ID,
    coarse_tile,
    span_overflow_groups,
)
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    build_loop_scheduler_nodes,
)
from torch_spyre._inductor.span_overflow_hint_analysis import (
    _find_outermost_span_dim,
    plan_span_overflow_tile,
)
import torch_spyre._inductor.propagate_named_dims as _pnd


_LAUNCH_KERNEL = "torch_spyre.execution.kernel_runner.launch_kernel"
_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


def _fixed_tiled_layout(shape, dtype=torch.float16):
    """Build the same kind of physical layout used by real Spyre lowering."""
    size = list(shape)
    stride = list(FlexibleLayout.contiguous_strides(size))
    stride_ints = [int(s) for s in stride]
    size_ints = [int(s) for s in size]
    within_stick_dim = len(size_ints) - 1
    dim_order = [i for i in range(len(size_ints)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size_ints, stride_ints, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _pointwise_op(shape, name="buf0"):
    """Return a real ComputedBuffer with a lightweight Pointwise mock."""
    data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    op = ComputedBuffer(
        name=name,
        layout=_fixed_tiled_layout(shape),
        data=data,
    )
    op.operation_name = name
    return op


def _graph(operations):
    return SimpleNamespace(operations=operations)


def _out_coords_for_bhld(_op):
    """Coordinates for shape [B, H, L, D] with B size 1 in these tests."""
    return [
        sympy.Integer(0),
        sympy.Symbol("h"),
        sympy.Symbol("l"),
        sympy.Symbol("d"),
    ]


def _out_coords_for_symbolic_bhld(_op):
    """Coordinates for shape [B, H, L, D] with B as a real loop var."""
    return [
        sympy.Symbol("b"),
        sympy.Symbol("h"),
        sympy.Symbol("l"),
        sympy.Symbol("d"),
    ]


def _run_span_overflow_groups(op):
    """Run span_overflow_groups with op_out_coords patched for one test op."""
    graph = _graph([op])

    with patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld):
        return span_overflow_groups(graph)


_E2E_SHAPE = (1, 8195, 256, 64)
_E2E_SPLIT_COUNT = 5
_E2E_TILE_SHAPE = [1, 1639, 256, 64]


def _manual_h_hint_group(op, hint_id=1, split_count=_E2E_SPLIT_COUNT):
    """Return the coarse-tile group produced by spyre_hint over dim H."""
    hint = DimHint(
        dim_names=["H"],
        split_count=split_count,
        loop_var=sympy.Symbol("h"),
        is_reduction=False,
        hint_id=hint_id,
    )
    op.dim_hints = [hint]
    return [([op], [(hint_id, sympy.Integer(split_count), False)])]


def _scheduler_node_for_op(op, name):
    """Return a minimal SchedulerNode mock wrapping one IR op."""
    scheduler = MagicMock()
    scheduler.name_to_fused_node = {}
    scheduler.removed_ops = set()

    snode = MagicMock(spec=SchedulerNode)
    snode.scheduler = scheduler
    snode.node = op
    snode.get_name.return_value = name
    snode.get_nodes.return_value = [snode]
    snode.ancestors = set()
    snode.min_order = 0
    snode.max_order = 0
    return snode


class TestSpanOverflowGroups(InductorTestCase):
    """Adapter-focused tests matching the user-hint group contract.

    These are intentionally close to the coarse-tiling draft tests: build one
    op, patch output coordinates, then inspect the generated group and DimHint.
    """

    def test_no_overflow_returns_empty(self):
        op = _pointwise_op((1, 2, 16, 64), name="small_op")

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])

    def test_overflow_pointwise_returns_one_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0][0][0], op)

    def test_group_structure(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        ops_list, levels = groups[0]
        self.assertEqual(ops_list, [op])
        self.assertEqual(len(levels), 1)
        hint_id, count, is_reduction_level = levels[0]
        self.assertEqual(hint_id, _SPAN_OVERFLOW_HINT_ID)
        self.assertIsInstance(count, sympy.Integer)
        self.assertEqual(count, sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertFalse(is_reduction_level)
        self.assertEqual(hint_id, op.dim_hints[0].hint_id)

    def test_two_overflow_ops_produce_two_groups(self):
        op0 = _pointwise_op(_E2E_SHAPE, name="buf0")
        op1 = _pointwise_op(_E2E_SHAPE, name="buf1")

        with patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld
        ):
            with config.patch({"sencores": 4, "chunk_large_tensors": False}):
                groups = span_overflow_groups(_graph([op0, op1]))

        self.assertEqual(len(groups), 2)
        self.assertIs(groups[0][0][0], op0)
        self.assertIs(groups[1][0][0], op1)
        self.assertEqual(groups[0][1][0][0], _SPAN_OVERFLOW_HINT_ID)
        self.assertEqual(groups[1][1][0][0], _SPAN_OVERFLOW_HINT_ID + 1)

    def test_dim_hint_attached_to_op(self):
        from torch_spyre._inductor.propagate_hints import DimHint

        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            _run_span_overflow_groups(op)

        self.assertTrue(hasattr(op, "dim_hints"))
        self.assertEqual(len(op.dim_hints), 1)
        hint = op.dim_hints[0]
        self.assertIsInstance(hint, DimHint)
        self.assertEqual(hint.dim_names, ["_span_overflow"])
        self.assertEqual(hint.split_count, _E2E_SPLIT_COUNT)
        self.assertEqual(hint.loop_var, sympy.Symbol("h"))
        self.assertFalse(hint.is_reduction)

    def test_trip_count_matches_level_and_hint(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        _, levels = groups[0]
        _, level_count, _ = levels[0]
        self.assertEqual(op.dim_hints[0].split_count, int(level_count))

    def test_non_fixed_tiled_layout_skipped(self):
        op = MagicMock(spec=ComputedBuffer)
        op.data = MagicMock(spec=Pointwise)
        op.data.ranges = [
            sympy.Integer(1),
            sympy.Integer(20),
            sympy.Integer(16),
            sympy.Integer(64),
        ]
        op.layout = MagicMock()
        op.get_name.return_value = "non_fixed_tiled"
        op.get_operation_name.return_value = "non_fixed_tiled"

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(groups, [])

    def test_symbolic_layout_skipped(self):
        op = _pointwise_op(_E2E_SHAPE)
        op.layout.size[1] = sympy.Symbol("s0")

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(groups, [])

    def test_chunk_large_tensors_config_suppresses_groups(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": True}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])

    def test_user_hinted_ops_do_not_block_unhinted_auto_groups(self):
        hinted_op = _pointwise_op(_E2E_SHAPE, name="hinted")
        hinted_op.dim_hints = [
            DimHint(
                dim_names=["H"],
                split_count=5,
                loop_var=sympy.Symbol("h"),
                is_reduction=False,
                hint_id=1,
            )
        ]
        unhinted_op = _pointwise_op(_E2E_SHAPE, name="unhinted")

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            with patch(
                "torch_spyre._inductor.coarse_tile.op_out_coords",
                _out_coords_for_bhld,
            ):
                groups = span_overflow_groups(_graph([hinted_op, unhinted_op]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [unhinted_op])
        self.assertEqual(getattr(hinted_op, "dim_hints")[0].hint_id, 1)

    def test_ignore_span_overflow_hints_config_suppresses_groups(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": True}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])


class TestSpanOverflowPointwisePlannerAndAdapter(InductorTestCase):
    """Mock-heavy tests for the first three compiler layers."""

    def test_planner_selects_dim_and_split_count(self):
        op = _pointwise_op(_E2E_SHAPE)

        plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.selected_host_dim, 1)
        self.assertEqual(plan.split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.is_reduction)
        self.assertEqual(plan.chunking_info.selected_device_dim_size, _E2E_SHAPE[1])

    def test_planner_skips_pointwise_with_indirect_reads(self):
        op = _pointwise_op(_E2E_SHAPE)

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis.indirect_info_from_op",
            return_value=({"arg1"}, {}, {sympy.Symbol("indirect0"): 8}),
        ):
            plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNone(plan)

    def test_planner_skips_size_one_device_dims(self):
        layout = SimpleNamespace(
            size=[1, 8195, 256, 64],
            stride=[8195 * 256 * 64, 256 * 64, 64, 1],
            device_layout=SimpleNamespace(
                device_size=[1, 8195, 256, 64],
                stride_map=[8195 * 256 * 64, 256 * 64, 64, 1],
                elems_per_stick=lambda: 64,
            ),
        )

        span_dim = _find_outermost_span_dim(layout, max_cores=4)

        self.assertIsNotNone(span_dim)
        self.assertEqual(span_dim.selected_device_dim_size, 8195)
        self.assertEqual(span_dim.selected_host_dim, 1)

    def test_planner_prefers_exact_stride_match(self):
        layout = SimpleNamespace(
            size=[4, 8, 64],
            stride=[64, 8, 1],
            device_layout=SimpleNamespace(
                device_size=[4, 8, 64],
                stride_map=[64, 8, 1],
                elems_per_stick=lambda: 64,
            ),
        )

        span_dim = _find_outermost_span_dim(layout, max_cores=4)

        self.assertIsNotNone(span_dim)
        self.assertEqual(span_dim.selected_host_dim, 0)

    def test_planner_rejects_when_stick_dim_tile_is_unaligned(self):
        # Granite-like vocab dim: 49159 is not 64-aligned.  The outermost
        # span dim maps to the vocab/within-stick host dim and would choose
        # split_count=11, producing tile size 4469, which cuts through a
        # physical stick.  The planner must reject this instead of emitting
        # an unsafe plan or falling back to an unrelated dimension.
        op = _pointwise_op((8192, 49159))

        with self.assertRaisesRegex(Unsupported, "stick alignment"):
            plan_span_overflow_tile(op, max_cores=4)

    def test_planner_allows_full_size_exact_divisor(self):
        op = _pointwise_op((1, 17, 16, 64))

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 32768
        ):
            plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.split_count, 17)

    @patch("torch_spyre._inductor.span_overflow_hint_analysis._post_tile_span_ok")
    def test_planner_raises_when_no_divisor_satisfies_post_tile_span(
        self,
        mock_post_tile_span_ok,
    ):
        mock_post_tile_span_ok.return_value = False
        op = _pointwise_op(_E2E_SHAPE)

        with self.assertRaisesRegex(Unsupported, "no divisor"):
            plan_span_overflow_tile(op, max_cores=4)

    @patch("torch_spyre._inductor.span_overflow_hint_analysis._post_tile_span_ok")
    @patch("torch_spyre._inductor.span_overflow_hint_analysis._find_outermost_span_dim")
    @patch("torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    def test_planner_falls_back_to_largest_host_dim_when_no_device_dim_maps(
        self,
        mock_find_outermost_span_dim,
        mock_post_tile_span_ok,
    ):
        mock_find_outermost_span_dim.return_value = None
        mock_post_tile_span_ok.return_value = True
        op = _pointwise_op((4, 128, 16, 64))

        plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.selected_host_dim, 1)
        self.assertEqual(plan.chunking_info.selected_device_dim_size, 0)
        self.assertIn("no device dimension", plan.chunking_info.reason)

    @patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld)
    def test_adapter_creates_dim_hint_and_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(len(groups), 1)
        group_ops, levels = groups[0]
        self.assertEqual(group_ops, [op])
        self.assertEqual(levels[0][1], sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertEqual(levels[0][2], False)
        self.assertEqual(len(op.dim_hints), 1)
        self.assertEqual(op.dim_hints[0].split_count, _E2E_SPLIT_COUNT)
        self.assertEqual(op.dim_hints[0].loop_var, sympy.Symbol("h"))

    @patch(
        "torch_spyre._inductor.coarse_tile.op_out_coords",
        _out_coords_for_symbolic_bhld,
    )
    def test_adapter_handles_nontrivial_batch_coord(self):
        op = _pointwise_op((4, 8195, 256, 64))

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(op.dim_hints), 1)
        # Batch is a real loop var in this test, but this shape's span-driving
        # physical dim still maps to H, so the adapter should choose h.
        self.assertEqual(op.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(groups[0][1][0][1], sympy.Integer(_E2E_SPLIT_COUNT))

    @patch("torch_spyre._inductor.coarse_tile.insert_tiling_propagation")
    @patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld)
    def test_coarse_tile_consumes_auto_group_and_stamps_op(
        self,
        _mock_insert_tiling_propagation,
    ):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            graph = _graph([op])
            groups = span_overflow_groups(graph)
            coarse_tile(graph, groups)

        self.assertEqual(list(op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(op.loop_info.loop_count, [sympy.Integer(_E2E_SPLIT_COUNT)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(op.loop_info.loop_tiled_reduction_dims, [[]])


class TestSpanOverflowLargeShapeContract(InductorTestCase):
    """Unit-style coverage for the real large shape used in E2E testing."""

    def test_large_shape_planner_adapter_and_coarse_tile_match_manual_hint(self):
        auto_op = _pointwise_op(_E2E_SHAPE, name="auto_buf")
        manual_op = _pointwise_op(_E2E_SHAPE, name="manual_buf")

        # Layer 1: planner chooses the same H split observed in the E2E run.
        plan = plan_span_overflow_tile(auto_op, max_cores=4)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.selected_host_dim, 1)
        self.assertEqual(plan.split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.is_reduction)

        with patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords",
            _out_coords_for_bhld,
        ):
            with patch("torch_spyre._inductor.coarse_tile.insert_tiling_propagation"):
                with config.patch({"sencores": 4, "chunk_large_tensors": False}):
                    # Layer 2: adapter emits the same group shape as user hints.
                    auto_graph = _graph([auto_op])
                    auto_groups = span_overflow_groups(auto_graph)
                    manual_graph = _graph([manual_op])
                    manual_groups = _manual_h_hint_group(manual_op)

                    self.assertEqual(len(auto_groups), 1)
                    self.assertEqual(len(manual_groups), 1)
                    self.assertEqual(auto_groups[0][1][0][1], sympy.Integer(5))
                    self.assertEqual(manual_groups[0][1][0][1], sympy.Integer(5))
                    self.assertFalse(auto_groups[0][1][0][2])
                    self.assertFalse(manual_groups[0][1][0][2])
                    self.assertEqual(auto_op.dim_hints[0].loop_var, sympy.Symbol("h"))
                    self.assertEqual(manual_op.dim_hints[0].loop_var, sympy.Symbol("h"))

                    # Layer 3: coarse_tile stamps identical per-tile IR shape.
                    coarse_tile(auto_graph, auto_groups)
                    coarse_tile(manual_graph, manual_groups)

        self.assertEqual(list(auto_op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(manual_op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(auto_op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(list(manual_op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(auto_op.loop_info.loop_count, [sympy.Integer(5)])
        self.assertEqual(manual_op.loop_info.loop_count, [sympy.Integer(5)])
        self.assertEqual(auto_op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(manual_op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(auto_op.loop_info.loop_tiled_reduction_dims, [[]])
        self.assertEqual(manual_op.loop_info.loop_tiled_reduction_dims, [[]])

        # Layer 4: scheduler wrapping sees the same counted loop on both paths.
        created = []

        def fake_create(snodes, loop_count):
            node = MagicMock(spec=CountedLoopSchedulerNode)
            node.snodes = snodes
            node.loop_count = loop_count
            node.get_nodes.return_value = snodes
            node.get_name.return_value = "_".join(n.get_name() for n in snodes)
            node.scheduler = snodes[0].scheduler
            created.append(node)
            return node

        auto_snode = _scheduler_node_for_op(auto_op, "auto_snode")
        manual_snode = _scheduler_node_for_op(manual_op, "manual_snode")
        with patch.object(
            CountedLoopSchedulerNode, "create", staticmethod(fake_create)
        ):
            auto_wrapped = build_loop_scheduler_nodes([auto_snode])
            manual_wrapped = build_loop_scheduler_nodes([manual_snode])

        self.assertEqual(len(auto_wrapped), 1)
        self.assertEqual(len(manual_wrapped), 1)
        self.assertEqual(created[0].loop_count, sympy.Integer(5))
        self.assertEqual(created[1].loop_count, sympy.Integer(5))
        self.assertEqual(auto_wrapped[0].loop_count, manual_wrapped[0].loop_count)


class TestSpanOverflowPointwiseCodegen(InductorTestCase):
    """Small codegen test for scheduler/codegen LoopSpec emission."""

    @patch("torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "chunk_large_tensors": False,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_codegen_contains_auto_span_overflow_loop_spec(self):
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        y = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")

        def fn(x, y):
            return x + y

        cfn = torch.compile(fn, dynamic=False)
        with (
            patch(_LAUNCH_KERNEL),
            patch(_LAUNCH_JOBPLAN),
            patch(_PREPARE_KERNEL),
            patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x, y)

        self.assertTrue(source_codes)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src)
        self.assertIn("sympify('5')", src)

    @patch("torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "chunk_large_tensors": False,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_auto_span_overflow_matches_equivalent_spyre_hint_loop_spec(self):
        from torch_spyre._inductor import spyre_hint

        shape = (1, 20, 16, 64)
        x = torch.randn(shape, dtype=torch.float16).to("spyre")
        y = torch.randn(shape, dtype=torch.float16).to("spyre")

        def auto_fn(x, y):
            return x + y

        def manual_hint_fn(x, y):
            with spyre_hint(num_tiles_per_dim={"SO_H": 5}):
                return x + y

        _pnd.declare_tensor_dim("SO_B", shape[0])
        _pnd.declare_tensor_dim("SO_H", shape[1])
        _pnd.declare_tensor_dim("SO_L", shape[2])
        _pnd.declare_tensor_dim("SO_D", shape[3])
        _pnd.name_tensor_dims(x, ["SO_B", "SO_H", "SO_L", "SO_D"])
        _pnd.name_tensor_dims(y, ["SO_B", "SO_H", "SO_L", "SO_D"])

        with (
            patch(_LAUNCH_KERNEL),
            patch(_LAUNCH_JOBPLAN),
            patch(_PREPARE_KERNEL),
            patch("subprocess.run"),
        ):
            _, auto_sources = run_and_get_code(
                torch.compile(auto_fn, dynamic=False), x, y
            )
            _, manual_sources = run_and_get_code(
                torch.compile(manual_hint_fn, dynamic=False), x, y
            )

        auto_src = auto_sources[0]
        manual_src = manual_sources[0]

        # Automatic span-overflow tiling should lower to the same one-level
        # counted loop shape as the equivalent explicit spyre_hint.
        self.assertEqual(auto_src.count("LoopSpec("), manual_src.count("LoopSpec("))
        self.assertEqual(auto_src.count("sympify('5')"), 1)
        self.assertEqual(manual_src.count("sympify('5')"), 1)
        self.assertIn("sympify('4')", auto_src)
        self.assertIn("sympify('4')", manual_src)
        self.assertIn("op='add'", auto_src)
        self.assertIn("op='add'", manual_src)
