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
- hard failure when output MemoryDep address math is unavailable;
- adapter mapping with both constant and symbolic batch output coordinates;
- coarse_tile stamping of ranges/layout/CoarseTileInfo;
- equivalence between auto span-overflow hints and manual spyre_hint codegen.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from torch_spyre._inductor.work_division import MAX_SPAN_BYTES

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
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
    ChunkingInfo,
    SpanOverflowTileLevel,
    SpanOverflowTilePlan,
    _bmm_output_symbol_to_dim,
    _candidate_host_dims,
    _input_read_deps,
    _input_span_infos_controlled_by_output_dims,
    _input_stick_alignment_error,
    plan_span_overflow_tile,
)
import torch_spyre._inductor.propagate_named_dims as _pnd
import torch_spyre._inductor.span_overflow_hint_analysis as soha


_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


def _fixed_tiled_layout(shape, dtype=torch.float16):
    """Build the same kind of physical layout used by real Spyre lowering."""
    size = list(shape)
    stride = list(FlexibleLayout.contiguous_strides(size))
    stride_ints = [int(s) for s in stride]
    size_ints = [int(s) for s in size]
    if not size_ints:
        device_layout = SpyreTensorLayout([], dtype)
        return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)

    within_stick_dim = len(size_ints) - 1
    dim_order = [i for i in range(len(size_ints)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size_ints, stride_ints, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _output_symbols_for_shape(shape):
    if len(shape) == 0:
        return ()
    if len(shape) == 4:
        return sympy.symbols("b h l d")
    return sympy.symbols(" ".join(f"d{i}" for i in range(len(shape))))


def _output_write_dep(name, shape, layout):
    symbols = _output_symbols_for_shape(shape)
    if not isinstance(symbols, tuple):
        symbols = (symbols,)
    index = sympy.Integer(0)
    for sym, stride in zip(symbols, layout.stride):
        index += sym * int(stride)
    return MemoryDep(name, index, symbols, tuple(shape))


def _default_read_writes_for_output(name, shape, layout):
    return SimpleNamespace(reads=set(), writes={_output_write_dep(name, shape, layout)})


def _pointwise_op(shape, name="buf0"):
    """Return a real ComputedBuffer with a lightweight Pointwise mock."""
    data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = _fixed_tiled_layout(shape)
    op = ComputedBuffer(
        name=name,
        layout=layout,
        data=data,
    )
    op.operation_name = name
    op.get_read_writes = MagicMock(
        return_value=_default_read_writes_for_output(name, shape, layout)
    )
    return op


def _reduction_op(shape, reduction_ranges=(64,), name="buf0", reduction_type="sum"):
    """Return a ComputedBuffer with a lightweight Reduction mock."""
    data = MagicMock(spec=Reduction)
    data.ranges = list(shape)
    data.reduction_ranges = list(reduction_ranges)
    data.reduction_type = reduction_type
    layout = _fixed_tiled_layout(shape)
    op = ComputedBuffer(
        name=name,
        layout=layout,
        data=data,
    )
    op.operation_name = name
    op.get_read_writes = MagicMock(
        return_value=_default_read_writes_for_output(name, shape, layout)
    )
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
    return [([op], [(hint_id, sympy.Integer(split_count))])]


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

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])

    def test_overflow_pointwise_returns_one_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0][0][0], op)

    def test_overflow_reduction_output_returns_one_group(self):
        op = _reduction_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0][0][0], op)
        self.assertFalse(op.dim_hints[0].is_reduction)

    def test_scalar_reduction_skipped(self):
        op = _reduction_op((), reduction_ranges=(8195, 256, 64))

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(groups, [])

    def test_group_structure(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        ops_list, levels = groups[0]
        self.assertEqual(ops_list, [op])
        self.assertEqual(len(levels), 1)
        hint_id, count = levels[0]
        self.assertEqual(hint_id, _SPAN_OVERFLOW_HINT_ID)
        self.assertIsInstance(count, sympy.Integer)
        self.assertEqual(count, sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertEqual(hint_id, op.dim_hints[0].hint_id)

    def test_two_compatible_pointwise_ops_produce_one_group(self):
        op0 = _pointwise_op(_E2E_SHAPE, name="buf0")
        op1 = _pointwise_op(_E2E_SHAPE, name="buf1")

        with patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld
        ):
            with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
                groups = span_overflow_groups(_graph([op0, op1]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [op0, op1])
        self.assertEqual(groups[0][1][0][0], _SPAN_OVERFLOW_HINT_ID)
        self.assertEqual(op0.dim_hints[0].hint_id, _SPAN_OVERFLOW_HINT_ID)
        self.assertEqual(op1.dim_hints[0].hint_id, _SPAN_OVERFLOW_HINT_ID)
        self.assertEqual(op0.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(op1.dim_hints[0].loop_var, sympy.Symbol("h"))

    def test_chained_compatible_pointwise_ops_produce_one_group(self):
        op0 = _pointwise_op(_E2E_SHAPE, name="buf0")
        op1 = _pointwise_op(_E2E_SHAPE, name="buf1")
        op1.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={
                    MemoryDep(
                        "buf0",
                        sympy.Symbol("h"),
                        (sympy.Symbol("h"),),
                        (8195,),
                    )
                },
                writes=_default_read_writes_for_output(
                    "buf1", _E2E_SHAPE, op1.layout
                ).writes,
            )
        )

        with patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld
        ):
            with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
                groups = span_overflow_groups(_graph([op0, op1]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [op0, op1])
        self.assertEqual(op0.dim_hints[0].hint_id, op1.dim_hints[0].hint_id)

    def _chained_pointwise_ops(self, shape1=_E2E_SHAPE):
        """Two Pointwise ops of the given shapes, op1 reading op0's buffer."""
        op0 = _pointwise_op(_E2E_SHAPE, name="buf0")
        op1 = _pointwise_op(shape1, name="buf1")
        op1.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={
                    MemoryDep(
                        "buf0",
                        sympy.Symbol("h"),
                        (sympy.Symbol("h"),),
                        (8195,),
                    )
                },
                writes=_default_read_writes_for_output(
                    "buf1", shape1, op1.layout
                ).writes,
            )
        )
        return op0, op1

    @staticmethod
    def _fake_plan(host_dim, split_count):
        return SpanOverflowTilePlan(
            levels=(
                SpanOverflowTileLevel(
                    selected_host_dim=host_dim, split_count=split_count
                ),
            ),
            chunking_infos=(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=split_count,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=host_dim,
                    stick_elems=64,
                    reason="output span overflow",
                ),
            ),
            reason="output span overflow",
        )

    def test_chained_pointwise_ops_conform_to_producer_split(self):
        """op1's own search disagrees with op0's, but op0's split is also
        legal and sufficient for op1 (identical shape/layout) -- op1 should
        adopt op0's split and join op0's group instead of raising."""
        op0, op1 = self._chained_pointwise_ops()

        def fake_plan(op, _max_cores):
            if op.get_name() == "buf0":
                return self._fake_plan(1, 5)
            return self._fake_plan(1, 11)

        with (
            patch(
                "torch_spyre._inductor.coarse_tile.plan_span_overflow_tile", fake_plan
            ),
            patch(
                "torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            groups = span_overflow_groups(_graph([op0, op1]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [op0, op1])
        self.assertEqual(op0.dim_hints[0].hint_id, op1.dim_hints[0].hint_id)
        # op1 adopted op0's split (5), not its own independently-searched one (11).
        self.assertEqual(op0.dim_hints[0].split_count, 5)
        self.assertEqual(op1.dim_hints[0].split_count, 5)

    def test_chained_pointwise_ops_conform_failure_still_raises(self):
        """op1's own search disagrees with op0's, and op0's split (5) does not
        evenly divide op1's H dim (8194) -- conform must fail and the
        producer-consumer read dependency must still raise Unsupported."""
        op0, op1 = self._chained_pointwise_ops(shape1=(1, 8194, 256, 64))

        def fake_plan(op, _max_cores):
            if op.get_name() == "buf0":
                return self._fake_plan(1, 5)
            return self._fake_plan(1, 11)

        with (
            patch(
                "torch_spyre._inductor.coarse_tile.plan_span_overflow_tile", fake_plan
            ),
            patch(
                "torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already auto-tiled producer.*buf0.*grouping currently only "
                "synchronizes compatible contiguous pointwise ops",
            ):
                span_overflow_groups(_graph([op0, op1]))

    def test_lm_head_auto_tiled_restickify_consumer_fails_safe(self):
        """LM-head restickify and BMM cannot be auto-tiled independently.

        This models F.linear(x[1,4096], weight[49216,4096]): lowering first
        creates a restickified weight buffer ``buf1`` and then a BMM reduction
        ``buf0`` that reads ``buf1``.  If both receive independent automatic
        span-overflow groups, their 769 vocab tiles are not one shared loop nest,
        so the adapter must fail loudly until producer-consumer fusion exists.
        """
        restickify_weight = _pointwise_op((49216, 4096), name="buf1")
        lm_head_bmm = _reduction_op(
            (1, 49216),
            reduction_ranges=(4096,),
            name="buf0",
            reduction_type=BATCH_MATMUL_OP,
        )
        restickify_weight.get_read_writes = MagicMock(
            return_value=SimpleNamespace(reads=set())
        )
        lm_head_bmm.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={
                    MemoryDep(
                        "buf1",
                        sympy.Symbol("d1"),
                        (sympy.Symbol("d1"),),
                        (49216,),
                    )
                }
            )
        )

        def fake_plan(op, _max_cores):
            return SpanOverflowTilePlan(
                levels=(
                    SpanOverflowTileLevel(
                        selected_host_dim=0 if op.get_name() == "buf1" else 1,
                        split_count=769,
                    ),
                ),
                chunking_infos=(
                    ChunkingInfo(
                        total_bytes=403177472,
                        per_core_span=403177472,
                        core_split_estimate=1,
                        selected_device_dim_size=769,
                        selected_device_span_stride_elems=262144,
                        selected_host_dim=0 if op.get_name() == "buf1" else 1,
                        stick_elems=64,
                        reason="output span overflow",
                    ),
                ),
                reason="output span overflow",
            )

        with (
            patch(
                "torch_spyre._inductor.coarse_tile.plan_span_overflow_tile", fake_plan
            ),
            patch(
                "torch_spyre._inductor.coarse_tile.op_out_coords",
                lambda op: (
                    [sympy.Symbol("d0"), sympy.Symbol("d1")]
                    if op.get_name() == "buf0"
                    else [sympy.Symbol("d0"), sympy.Symbol("d1")]
                ),
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already auto-tiled producer.*grouping currently only synchronizes "
                "compatible contiguous pointwise ops",
            ):
                span_overflow_groups(_graph([restickify_weight, lm_head_bmm]))

    def test_manually_hinted_producer_blocks_auto_tiled_consumer(self):
        """A user spyre_hint on a producer must also guard its auto-tiled consumer.

        auto_tiled_producers only tracks producers this pass tiles itself.
        assign_dim_hints runs earlier and leaves dim_hints set on any
        manually-hinted op, so a consumer this pass independently decides to
        auto-tile must also be checked against those -- reading a manually
        tiled producer has the exact same unsynchronized-loop-nest risk as
        reading one this pass auto-tiled itself.
        """
        producer = _pointwise_op(_E2E_SHAPE, name="buf1")
        _manual_h_hint_group(producer)  # simulates a user spyre_hint

        consumer = _pointwise_op(_E2E_SHAPE, name="buf0")
        consumer.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={
                    MemoryDep(
                        "buf1",
                        sympy.Symbol("h"),
                        (sympy.Symbol("h"),),
                        (_E2E_SHAPE[1],),
                    )
                },
                writes={_output_write_dep("buf0", _E2E_SHAPE, consumer.layout)},
            )
        )

        with (
            patch(
                "torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already auto-tiled producer.*grouping currently only synchronizes "
                "compatible contiguous pointwise ops",
            ):
                span_overflow_groups(_graph([producer, consumer]))

    def test_dim_hint_attached_to_op(self):
        from torch_spyre._inductor.propagate_hints import DimHint

        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
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

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _run_span_overflow_groups(op)

        _, levels = groups[0]
        _, level_count = levels[0]
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

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(groups, [])

    def test_symbolic_layout_skipped(self):
        op = _pointwise_op(_E2E_SHAPE)
        op.layout.size[1] = sympy.Symbol("s0")

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = span_overflow_groups(_graph([op]))

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

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            with patch(
                "torch_spyre._inductor.coarse_tile.op_out_coords",
                _out_coords_for_bhld,
            ):
                groups = span_overflow_groups(_graph([hinted_op, unhinted_op]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [unhinted_op])
        self.assertEqual(getattr(hinted_op, "dim_hints")[0].hint_id, 1)

    def test_ignore_wsr_hints_config_suppresses_groups(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_wsr_hints": True}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])


class TestSpanOverflowPointwisePlannerAndAdapter(InductorTestCase):
    """Mock-heavy tests for the first three compiler layers."""

    def test_planner_selects_dim_and_split_count(self):
        op = _pointwise_op(_E2E_SHAPE)

        plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 1)
        self.assertEqual(plan.levels[0].split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.levels[0].is_reduction)
        self.assertEqual(plan.chunking_infos[0].selected_device_dim_size, _E2E_SHAPE[1])

    def test_auto_planner_rejects_split_counts_above_codegen_cap(self):
        # (4096, 4096) fp16 is 32MB total; capping MAX_SPAN_BYTES well below
        # that proves a nontrivial split is required.  With the auto split cap
        # patched to 1, every nontrivial divisor is rejected before combo
        # validation, so the planner must fail instead of silently accepting an
        # uncapped split.
        op = _pointwise_op((4096, 4096))

        with (
            patch.object(soha, "MAX_SPAN_BYTES", 4 * 1024 * 1024),
            patch.object(soha, "_MAX_AUTO_TILE_SPLIT_COUNT", 1),
        ):
            with self.assertRaisesRegex(Unsupported, "no combined split"):
                plan_span_overflow_tile(op, max_cores=1)

    def test_planner_skips_pointwise_with_indirect_reads(self):
        op = _pointwise_op(_E2E_SHAPE)

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis.indirect_info_from_op",
            return_value=({"arg1"}, {}, {sympy.Symbol("indirect0"): 8}),
        ):
            plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNone(plan)

    def test_reduction_output_planner_selects_dim_and_split_count(self):
        op = _reduction_op(_E2E_SHAPE)

        plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 1)
        self.assertEqual(plan.levels[0].split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.levels[0].is_reduction)

    def test_scalar_reduction_planner_skips(self):
        op = _reduction_op((), reduction_ranges=(8195, 256, 64))

        plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNone(plan)

    def test_reduction_input_span_controlled_by_output_dim_plans_tile(self):
        op = _reduction_op((4_194_304,), reduction_ranges=(64,))
        m, k = sympy.symbols("m k")
        input_dep = MemoryDep(
            "arg0",
            m * 64 + k,
            (m, k),
            (4_194_304, 64),
        )
        input_layout = _fixed_tiled_layout((4_194_304, 64))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0},
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
                return_value=[],
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 0)
        self.assertEqual(plan.levels[0].split_count, 2)
        self.assertIn("input span overflow", plan.reason)

    def test_reduction_input_span_controlled_by_reduction_dim_is_known_gap(self):
        op = _reduction_op((64,), reduction_ranges=(65536,))
        n, k = sympy.symbols("n k")
        input_dep = MemoryDep(
            "arg0",
            k * 64 + n,
            (k, n),
            (65536, 64),
        )
        input_layout = _fixed_tiled_layout((65536, 64))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={n: 0},
            ),
        ):
            infos = _input_span_infos_controlled_by_output_dims(op, max_cores=1)
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertEqual(infos, [])
        self.assertIsNone(plan)

    def test_bmm_input_span_controlled_by_n_dim_plans_output_tile(self):
        op = _reduction_op(
            (1, 16, 4_194_304), reduction_ranges=(64,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k = sympy.symbols("b m n k")
        rhs_dep = MemoryDep(
            "rhs",
            n * 64 + k,
            (n, k),
            (4_194_304, 64),
        )
        rhs_layout = _fixed_tiled_layout((4_194_304, 64))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(rhs_dep, rhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
                return_value=[],
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 2)
        self.assertEqual(plan.levels[0].split_count, 2)
        self.assertFalse(plan.levels[0].is_reduction)

    def test_bmm_input_span_controlled_by_k_dim_skips(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(65536,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k = sympy.symbols("b m n k")
        lhs_dep = MemoryDep(
            "lhs",
            k * 64 + m,
            (k, m),
            (65536, 64),
        )
        lhs_layout = _fixed_tiled_layout((65536, 64))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(lhs_dep, lhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
        ):
            infos = _input_span_infos_controlled_by_output_dims(op, max_cores=1)
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertEqual(infos, [])
        self.assertIsNone(plan)

    def test_reduction_output_and_input_span_different_dims_can_emit_multilevel_plan(
        self,
    ):
        op = _reduction_op((8192, 8192), reduction_ranges=(64,))
        output_info = SimpleNamespace(
            total_bytes=512 * 1024 * 1024,
            per_core_span=512 * 1024 * 1024,
            core_split_estimate=1,
            selected_device_dim_size=8192,
            selected_device_span_stride_elems=32768,
            selected_host_dim=0,
            stick_elems=64,
            reason="output span overflow",
        )
        input_info = SimpleNamespace(
            total_bytes=512 * 1024 * 1024,
            per_core_span=512 * 1024 * 1024,
            core_split_estimate=1,
            selected_device_dim_size=8192,
            selected_device_span_stride_elems=32768,
            selected_host_dim=1,
            stick_elems=64,
            reason="input span overflow for arg0",
        )
        output_candidate = SimpleNamespace(chunking_info=output_info, source="output")
        input_candidate = SimpleNamespace(chunking_info=input_info, source="input:arg0")

        def remaining_after_tile(_op, _max_cores, split_by_host_dim):
            if set(split_by_host_dim) == {0, 1}:
                return []
            return [object()]

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[output_candidate],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_span_candidates",
                return_value=[input_candidate],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
                side_effect=remaining_after_tile,
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.levels), 2)
        self.assertEqual({level.selected_host_dim for level in plan.levels}, {0, 1})

    def test_pointwise_output_spans_different_dims_can_emit_multilevel_plan(self):
        op = _pointwise_op((8192, 8192, 64))
        dim0_info = SimpleNamespace(
            total_bytes=512 * 1024 * 1024,
            per_core_span=512 * 1024 * 1024,
            core_split_estimate=1,
            selected_device_dim_size=8192,
            selected_device_span_stride_elems=32768,
            selected_host_dim=0,
            stick_elems=64,
            reason="output span overflow",
        )
        dim1_info = SimpleNamespace(
            total_bytes=512 * 1024 * 1024,
            per_core_span=512 * 1024 * 1024,
            core_split_estimate=1,
            selected_device_dim_size=8192,
            selected_device_span_stride_elems=64,
            selected_host_dim=1,
            stick_elems=64,
            reason="output span overflow",
        )
        candidates = [
            SimpleNamespace(chunking_info=dim0_info, source="output"),
            SimpleNamespace(chunking_info=dim1_info, source="output"),
        ]

        def remaining_after_tile(_op, _max_cores, split_by_host_dim):
            if set(split_by_host_dim) == {0, 1}:
                return []
            return [object()]

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=candidates,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
                side_effect=remaining_after_tile,
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.levels), 2)
        self.assertEqual({level.selected_host_dim for level in plan.levels}, {0, 1})
        self.assertEqual([level.selected_host_dim for level in plan.levels], [0, 1])

    def test_input_read_deps_skips_bad_inputs_individually(self):
        op = MagicMock(spec=ComputedBuffer)
        bad_sym, good_sym = sympy.symbols("bad good")
        bad_dep = MemoryDep("bad", bad_sym, (bad_sym,), (16,))
        good_dep = MemoryDep("good", good_sym, (good_sym,), (16,))
        op.get_read_writes.return_value = SimpleNamespace(reads=[bad_dep, good_dep])
        good_layout = _fixed_tiled_layout((16,))

        def fake_fixed_read_layout(buf):
            if buf == "bad":
                raise RuntimeError("bad layout")
            return good_layout

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.V",
                SimpleNamespace(graph=SimpleNamespace(get_buffer=lambda name: name)),
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._fixed_read_layout",
                side_effect=fake_fixed_read_layout,
            ),
        ):
            deps = _input_read_deps(op)

        self.assertEqual(deps, [(good_dep, good_layout)])

    def test_planner_rejects_when_stick_dim_tile_is_unaligned(self):
        # Granite-like vocab dim: 49159 is not 64-aligned.  The output
        # span candidate maps to the vocab/within-stick host dim and would choose
        # split_count=11, producing tile size 4469, which cuts through a
        # physical stick.  The planner must reject this instead of emitting
        # an unsafe plan or falling back to an unrelated dimension.
        op = _pointwise_op((8192, 49159))

        with self.assertRaisesRegex(Unsupported, "no combined split"):
            plan_span_overflow_tile(op, max_cores=4)

    def test_within_stick_host_dim_returns_none_when_no_host_stride_matches(self):
        # No host stride equals the device layout's final stride-map entry.
        # An earlier revision guessed len(host_stride) - 1 here; that risked
        # silently validating stick alignment against the wrong host dim if
        # the guess was wrong. It must instead report "unknown" so the
        # caller can fail safe.
        fake_layout = SimpleNamespace(
            stride=[8, 4, 1],
            device_layout=SimpleNamespace(stride_map=[8, 4, 999]),
        )

        self.assertIsNone(soha._within_stick_host_dim(fake_layout))

    def test_post_tile_stick_alignment_error_rejects_when_stick_dim_unknown(self):
        fake_layout = SimpleNamespace(
            stride=[8, 4, 1],
            device_layout=SimpleNamespace(stride_map=[8, 4, 999]),
            size=[10, 20, 30],
        )

        error = soha._post_tile_stick_alignment_error(
            fake_layout, selected_host_dim=2, split_count=3
        )

        self.assertIsNotNone(error)

    def test_planner_allows_full_size_exact_divisor_for_pointwise(self):
        op = _pointwise_op((1, 17, 16, 64))

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 32768
        ):
            plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].split_count, 17)

    def test_planner_rejects_full_size_exact_divisor_for_reduction(self):
        # Reduction codegen/DDC can drop unit-size iteration dims before fixed
        # template matching.  Keep this rejection scoped to Reduction ops;
        # Pointwise full-size exact divisors are still legal.
        op = _reduction_op((1, 17, 16, 64))

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 32768
        ):
            with self.assertRaisesRegex(Unsupported, "no combined split"):
                plan_span_overflow_tile(op, max_cores=4)

    def test_planner_raises_when_no_combined_split_satisfies_post_tile_span(self):
        op = _pointwise_op(_E2E_SHAPE)

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
            return_value=[object()],
        ):
            with self.assertRaisesRegex(Unsupported, "no combined split"):
                plan_span_overflow_tile(op, max_cores=4)

    def test_reduction_skips_indirect_reads_even_when_span_overflows(self):
        op = _reduction_op(_E2E_SHAPE)

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis.indirect_info_from_op",
            return_value=({"arg1"}, {}, {sympy.Symbol("indirect0"): 8}),
        ):
            plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNone(plan)

    def test_reduction_indirect_guard_is_op_level_not_per_dim(self):
        op = _reduction_op(_E2E_SHAPE, reduction_ranges=(64,))
        m, n, k = sympy.symbols("m n k")
        input_dep = MemoryDep(
            "arg0",
            m * 256 * 64 + n * 64 + k,
            (m, n, k),
            (8192, 256, 64),
        )
        input_layout = _fixed_tiled_layout((8192, 256, 64))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.indirect_info_from_op",
                return_value=({"arg0"}, {}, {sympy.Symbol("indirect0"): 8}),
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0, n: 1},
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNone(plan)

    def test_input_span_scan_continues_after_reduction_controlled_dim(self):
        op = _reduction_op((4_194_304, 64), reduction_ranges=(65536,))
        k, m, n = sympy.symbols("k m n")
        input_dep = MemoryDep(
            "arg0",
            k * 4_194_304 * 64 + m * 64 + n,
            (k, m, n),
            (65536, 4_194_304, 64),
        )
        input_layout = _fixed_tiled_layout((65536, 4_194_304, 64))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0, n: 1},
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._device_coordinates_for_span",
                return_value=[k, m, n],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
                return_value=[],
            ),
        ):
            infos = _input_span_infos_controlled_by_output_dims(op, max_cores=1)
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].chunking_info.selected_host_dim, 0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 0)

    def test_bmm_symbol_map_requires_exactly_one_reduction_symbol(self):
        op = _reduction_op(
            (1, 16, 64), reduction_ranges=(64,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k0, k1 = sympy.symbols("b m n k0 k1")
        dep0 = MemoryDep("lhs", k0 * 16 + m, (k0, m), (64, 16))
        dep1 = MemoryDep("rhs", k1 * 64 + n, (k1, n), (64, 64))

        with patch(
            "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
            return_value={b: 0, m: 1, n: 2},
        ):
            symbol_to_dim = _bmm_output_symbol_to_dim(
                op,
                [
                    (dep0, _fixed_tiled_layout((64, 16))),
                    (dep1, _fixed_tiled_layout((64, 64))),
                ],
            )

        self.assertEqual(symbol_to_dim, {})

    def test_input_stick_alignment_rejects_split_legal_on_output_layout(self):
        op = _reduction_op((8190, 64), reduction_ranges=(64,))
        m, n, k = sympy.symbols("m n k")
        input_dep = MemoryDep(
            "transposed_rhs",
            k * 64 * 8192 + n * 8192 + m,
            (k, n, m),
            (64, 64, 8192),
        )
        input_layout = _fixed_tiled_layout((64, 64, 8192))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0, n: 1},
            ),
        ):
            error = _input_stick_alignment_error(op, host_dim=0, split_count=3)

        self.assertIsNotNone(error)
        self.assertIn("transposed_rhs", error)
        self.assertIn("host dim 2", error)

    def test_input_stick_alignment_checks_jointly_controlled_input_dim(self):
        # The target symbol (m) is not the sole free symbol of any input
        # coordinate -- it shares the within-stick dim's coordinate with
        # another symbol (n), e.g. an interleaved/collapsed physical stride
        # after a view or transpose. Requiring an exact coord.free_symbols
        # == {m} match would find no dimension at all and silently skip the
        # stick-alignment check entirely; checking every dimension m
        # contributes to (regardless of co-occurring symbols) still catches
        # the misaligned split.
        op = _reduction_op((8190, 64), reduction_ranges=(64,))
        m, n, k = sympy.symbols("m n k")
        input_dep = MemoryDep(
            "interleaved_rhs",
            k * 8192 + m + n,
            (k, m, n),
            (64, 8192, 8192),
        )
        input_layout = _fixed_tiled_layout((64, 8192))

        with (
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0, n: 1},
            ),
            patch(
                "torch_spyre._inductor.span_overflow_hint_analysis.host_coordinates",
                return_value=[k, m + n],
            ),
        ):
            error = _input_stick_alignment_error(op, host_dim=0, split_count=3)

        self.assertIsNotNone(error)
        self.assertIn("interleaved_rhs", error)
        self.assertIn("host dim 1", error)

    def test_candidate_host_dims_orders_by_decreasing_span_pressure(self):
        candidates = [
            SimpleNamespace(
                chunking_info=SimpleNamespace(selected_host_dim=1, per_core_span=512)
            ),
            SimpleNamespace(
                chunking_info=SimpleNamespace(selected_host_dim=0, per_core_span=2048)
            ),
            SimpleNamespace(
                chunking_info=SimpleNamespace(selected_host_dim=2, per_core_span=1024)
            ),
        ]

        self.assertEqual(_candidate_host_dims(candidates), [0, 2, 1])

    @patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld)
    def test_adapter_creates_dim_hint_and_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(len(groups), 1)
        group_ops, levels = groups[0]
        self.assertEqual(group_ops, [op])
        self.assertEqual(levels[0][1], sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertEqual(len(op.dim_hints), 1)
        self.assertEqual(op.dim_hints[0].split_count, _E2E_SPLIT_COUNT)
        self.assertEqual(op.dim_hints[0].loop_var, sympy.Symbol("h"))

    @patch(
        "torch_spyre._inductor.coarse_tile.op_out_coords",
        _out_coords_for_symbolic_bhld,
    )
    def test_adapter_handles_nontrivial_batch_coord(self):
        op = _pointwise_op((4, 8195, 256, 64))

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
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

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            graph = _graph([op])
            groups = span_overflow_groups(graph)
            coarse_tile(graph, groups)

        self.assertEqual(list(op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(op.loop_info.loop_count, [sympy.Integer(_E2E_SPLIT_COUNT)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(op.loop_info.loop_tiled_reduction_dims, [[]])


class TestSpanOverflowAdditionalPlannerCases(InductorTestCase):
    def test_output_symbol_mapping_keepdim_false(self):
        op = _reduction_op((1024, 4096), reduction_ranges=(128,))
        b, s = sympy.symbols("b s")

        with patch.object(soha, "op_out_coords", return_value=[b, s]):
            symbol_to_dim = soha._output_symbol_to_dim(op)

        self.assertEqual(symbol_to_dim[b], 0)
        self.assertEqual(symbol_to_dim[s], 1)

    def test_output_symbol_mapping_keepdim_true(self):
        op = _reduction_op((1024, 4096, 1), reduction_ranges=(128,))
        b, s = sympy.symbols("b s")

        with patch.object(soha, "op_out_coords", return_value=[b, s, sympy.Integer(0)]):
            symbol_to_dim = soha._output_symbol_to_dim(op)

        self.assertEqual(symbol_to_dim[b], 0)
        self.assertEqual(symbol_to_dim[s], 1)
        self.assertNotIn(sympy.Integer(0), symbol_to_dim)

    def test_single_reduction_dim_output_controlled_input_span_plans(self):
        op = _reduction_op((4_194_304,), reduction_ranges=(64,))
        m, k = sympy.symbols("m k")
        dep = MemoryDep("arg0", m * 64 + k, (m, k), (4_194_304, 64))
        layout = _fixed_tiled_layout((4_194_304, 64))

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(soha, "_input_read_deps", return_value=[(dep, layout)]),
            patch.object(soha, "_output_symbol_to_dim", return_value={m: 0}),
            patch.object(
                soha, "_remaining_span_candidates_after_tile", return_value=[]
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 0)
        self.assertEqual(plan.levels[0].split_count, 2)

    def test_multiple_reduction_dims_are_skipped_as_known_limitation(self):
        op = _reduction_op((64,), reduction_ranges=(8192, 8192))
        n, k0, k1 = sympy.symbols("n k0 k1")
        dep = MemoryDep(
            "arg0",
            k0 * 8192 * 64 + k1 * 64 + n,
            (k0, k1, n),
            (8192, 8192, 64),
        )
        layout = _fixed_tiled_layout((8192, 8192, 64))

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_input_read_deps", return_value=[(dep, layout)]),
            patch.object(soha, "_output_symbol_to_dim", return_value={n: 0}),
        ):
            infos = soha._input_span_infos_controlled_by_output_dims(op, max_cores=1)
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertEqual(infos, [])
        self.assertIsNone(plan)

    def test_full_scalar_reduction_returns_none(self):
        op = _reduction_op((), reduction_ranges=(4096, 4096, 128))

        self.assertIsNone(plan_span_overflow_tile(op, max_cores=4))

    def test_multiple_input_reads_aggregate_input_candidates(self):
        op = _reduction_op((4_194_304,), reduction_ranges=(64,))
        m, k = sympy.symbols("m k")
        dep0 = MemoryDep("arg0", m * 64 + k, (m, k), (4_194_304, 64))
        dep1 = MemoryDep("arg1", m * 64 + k, (m, k), (4_194_304, 64))
        layout = _fixed_tiled_layout((4_194_304, 64))

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(
                soha, "_input_read_deps", return_value=[(dep0, layout), (dep1, layout)]
            ),
            patch.object(soha, "_output_symbol_to_dim", return_value={m: 0}),
        ):
            infos = soha._input_span_infos_controlled_by_output_dims(op, max_cores=1)

        self.assertEqual(len(infos), 2)
        self.assertEqual({info.dep_name for info in infos}, {"arg0", "arg1"})

    def test_broadcasted_input_without_output_symbol_does_not_misfire(self):
        op = _reduction_op((4_194_304,), reduction_ranges=(64,))
        m, k = sympy.symbols("m k")
        dep = MemoryDep("bias", k, (k,), (64,))
        layout = _fixed_tiled_layout((64,))

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_input_read_deps", return_value=[(dep, layout)]),
            patch.object(soha, "_output_symbol_to_dim", return_value={m: 0}),
        ):
            infos = soha._input_span_infos_controlled_by_output_dims(op, max_cores=1)

        self.assertEqual(infos, [])

    def test_input_coordinate_jointly_controlled_by_two_symbols_becomes_two_candidates(
        self,
    ):
        """A coordinate mixing two output symbols must not be silently dropped.

        Some physical layouts interleave two logical dims into one physical
        stride (see the (4096, 4096, 4096, 64) repro).  Such a coordinate is
        still safely tileable by splitting either contributing dim, so it must
        produce a candidate for each dim instead of being skipped outright.
        """
        op = _reduction_op((2_000_000,), reduction_ranges=(64,))
        p, q = sympy.symbols("p q")
        dep = MemoryDep("arg0", p + q, (p, q), (2_000_000, 2_000_000))
        layout = SimpleNamespace(
            size=[2_000_000, 64],
            stride=[64, 1],
            dtype=torch.float16,
            device_layout=SimpleNamespace(
                device_size=[2_000_000, 64],
                stride_map=[64, 1],
                elems_per_stick=lambda: 64,
            ),
        )

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_input_read_deps", return_value=[(dep, layout)]),
            patch.object(soha, "_output_symbol_to_dim", return_value={p: 0, q: 1}),
            patch.object(
                soha,
                "_device_coordinates_for_span",
                return_value=[p + q, sympy.Integer(0)],
            ),
        ):
            infos = soha._input_span_infos_controlled_by_output_dims(op, max_cores=1)

        self.assertEqual(len(infos), 2)
        self.assertEqual(
            {info.chunking_info.selected_host_dim for info in infos}, {0, 1}
        )
        spans = {info.chunking_info.per_core_span for info in infos}
        self.assertEqual(len(spans), 1)

    def test_input_span_validation_uses_other_tiled_inner_output_dims(self):
        """Combined input span validation must shrink inner tiled coords too.

        The sum repro shape (2, 2, 257, 64, 64, 128) over the last dim has an
        input physical d1 coordinate whose inner span includes d2.  Splitting d1
        alone still leaves a 514 MB span, but splitting d1 and d2 together makes
        the d1 span small enough.
        """
        op = _reduction_op((2, 2, 257, 64, 64), reduction_ranges=(128,))
        d0, d1, d2, d3, d4, d5 = sympy.symbols("d0 d1 d2 d3 d4 d5")
        dep = MemoryDep(
            "arg0",
            269484032 * d0 + 134742016 * d1 + 524288 * d2 + 8192 * d3 + 128 * d4 + d5,
            (d0, d1, d2, d3, d4, d5),
            (2, 2, 257, 64, 64, 128),
        )
        layout = SimpleNamespace(
            size=[2, 2, 257, 64, 64, 128],
            stride=[269484032, 134742016, 524288, 8192, 128, 1],
            dtype=torch.float16,
            device_layout=SimpleNamespace(
                device_size=[2, 257, 64, 64, 2, 2, 64],
                stride_map=[134742016, 524288, 8192, 128, 64, 269484032, 1],
                elems_per_stick=lambda: 64,
            ),
        )
        device_coords = [
            d1,
            d2,
            d3,
            d4,
            sympy.floor(d5 / 64),
            d0,
            sympy.Mod(d5, 64),
        ]

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_input_read_deps", return_value=[(dep, layout)]),
            patch.object(
                soha,
                "_output_symbol_to_dim",
                return_value={d0: 0, d1: 1, d2: 2, d3: 3, d4: 4},
            ),
            patch.object(
                soha, "_device_coordinates_for_span", return_value=device_coords
            ),
        ):
            d1_only_infos = soha._input_span_infos_controlled_by_output_dims(
                op,
                max_cores=1,
                split_by_host_dim={1: 2},
            )
            d1_d2_infos = soha._input_span_infos_controlled_by_output_dims(
                op,
                max_cores=1,
                split_by_host_dim={1: 2, 2: 257},
            )

        self.assertIn(
            1,
            {info.chunking_info.selected_host_dim for info in d1_only_infos},
        )
        self.assertEqual(d1_d2_infos, [])

    def test_transposed_bmm_input_stick_alignment_rejects_split(self):
        op = _reduction_op(
            (8190, 64), reduction_ranges=(64,), reduction_type=BATCH_MATMUL_OP
        )
        m, n, k = sympy.symbols("m n k")
        dep = MemoryDep(
            "transposed_rhs",
            k * 64 * 8192 + n * 8192 + m,
            (k, n, m),
            (64, 64, 8192),
        )
        layout = _fixed_tiled_layout((64, 64, 8192))

        with (
            patch.object(soha, "_input_read_deps", return_value=[(dep, layout)]),
            patch.object(soha, "_output_symbol_to_dim", return_value={m: 0, n: 1}),
        ):
            error = soha._input_stick_alignment_error(op, host_dim=0, split_count=3)

        self.assertIsNotNone(error)
        self.assertIn("transposed_rhs", error)

    def test_output_coordinate_jointly_controlled_by_two_symbols_becomes_two_candidates(
        self,
    ):
        """Output-side counterpart of the input-side joint-coordinate test.

        ``_output_span_candidates_from_op`` must register a candidate for
        every output symbol that jointly controls an overflowing physical
        coordinate, not just bail out because more than one symbol is
        involved.
        """
        p, q = sympy.symbols("p q")
        out_dep = MemoryDep("buf0", p + q, (p, q), (2_000_000, 2_000_000))
        layout = SimpleNamespace(
            size=[2_000_000, 2_000_000],
            dtype=torch.float16,
            device_layout=SimpleNamespace(
                device_size=[2_000_000, 64],
                elems_per_stick=lambda: 64,
            ),
        )
        op = MagicMock(spec=ComputedBuffer)
        op.get_name.return_value = "buf0"

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_output_write_dep", return_value=out_dep),
            patch.object(soha, "_output_symbol_to_dim", return_value={p: 0, q: 1}),
            patch.object(
                soha,
                "_device_coordinates_for_span",
                return_value=[p + q, sympy.Integer(0)],
            ),
        ):
            candidates = soha._output_span_candidates_from_op(
                op, layout=layout, op_name="buf0"
            )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {c.chunking_info.selected_host_dim for c in candidates}, {0, 1}
        )
        spans = {c.chunking_info.per_core_span for c in candidates}
        self.assertEqual(len(spans), 1)

    def test_pointwise_post_tile_validation_uses_tiled_ranges(self):
        """Post-tile validation must model the per-tile iteration domain.

        This shape used to raise because validation rebuilt a shrunk output
        layout but kept the original full output ``MemoryDep.ranges``.  The
        mismatched domain made revalidation report an overflow tied to a dim
        outside the initial candidate set.  With tiled ranges, the selected
        split validates against the same domain the real tiled kernel executes.
        """
        op = _pointwise_op((4096, 4032, 4032, 64))

        with patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)

    def test_pointwise_too_many_overflow_dims_raises(self):
        op = _pointwise_op((64, 64, 64, 64, 64, 64))

        with patch.object(soha, "MAX_SPAN_BYTES", 1):
            with self.assertRaisesRegex(Exception, "bounded search limit"):
                plan_span_overflow_tile(op, max_cores=1)

    def test_missing_output_write_dep_skips_auto_tiling(self):
        op = _pointwise_op((1, 8195, 256, 64))

        with patch.object(soha, "_output_write_dep", return_value=None):
            self.assertIsNone(plan_span_overflow_tile(op, max_cores=4))

    def test_coordinate_span_elems_preserves_mod_coefficients(self):
        h = sympy.Symbol("h")
        dep = MemoryDep("buf0", h, (h,), (6000,))
        coord = sympy.floor(2 * sympy.Mod(h, 2048))

        span = soha._coordinate_span_elems(coord, dep, {h: 1})

        self.assertEqual(span, 4095)

    def test_coordinate_span_elems_multi_mod_same_symbol_uses_each_modulus_critical_point(
        self,
    ):
        # Two Mod() atoms on the same symbol with different moduli: the true
        # maximum occurs at the *larger* modulus's own wraparound point
        # (h=127: Mod(127,64)=63, Mod(127,128)=127, sum=190), not at the
        # smaller modulus's critical point (h=63: 63+63=126). Evaluating only
        # at a single critical point derived from the smallest modulus would
        # underestimate the span (127 instead of the true 191).
        h = sympy.Symbol("h")
        dep = MemoryDep("buf0", h, (h,), (200,))
        coord = sympy.Mod(h, 64) + sympy.Mod(h, 128)

        span = soha._coordinate_span_elems(coord, dep, {h: 1})

        self.assertEqual(span, 191)

    def test_coordinate_span_elems_returns_none_for_coefficient_inside_mod_argument(
        self,
    ):
        # The critical-point trick (evaluate at sym = modulus - 1) is only
        # exact when a Mod's argument is the bare symbol. A coefficient on
        # the argument shifts where the true wraparound maximum occurs:
        # Mod(3*h, 64) over h in [0, 100) has its true max (63) at h=21, not
        # at the naive critical point h=63 (which gives only 61). Silently
        # evaluating only at h=63 would underestimate the span (62 instead
        # of the true 64). This function must fail safe (return None) for
        # this shape instead, rather than accept an unproven bound.
        h = sympy.Symbol("h")
        dep = MemoryDep("buf0", h, (h,), (100,))
        coord = sympy.Mod(3 * h, 64)

        span = soha._coordinate_span_elems(coord, dep, {h: 1})

        self.assertIsNone(span)

    def test_reduction_indirect_read_guard(self):
        op = _reduction_op((1, 8195, 256, 64), reduction_ranges=(128,))

        with patch.object(
            soha,
            "indirect_info_from_op",
            return_value=({"arg0"}, {}, {sympy.Symbol("indirect0"): 8}),
        ):
            plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNone(plan)

    def test_bmm_ambiguous_reduction_symbol_map_returns_empty(self):
        op = _reduction_op(
            (1, 16, 64), reduction_ranges=(64,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k0, k1 = sympy.symbols("b m n k0 k1")
        dep0 = MemoryDep("lhs", k0 * 16 + m, (k0, m), (64, 16))
        dep1 = MemoryDep("rhs", k1 * 64 + n, (k1, n), (64, 64))

        with patch.object(
            soha, "_output_symbol_to_dim", return_value={b: 0, m: 1, n: 2}
        ):
            symbol_to_dim = _bmm_output_symbol_to_dim(
                op,
                [
                    (dep0, _fixed_tiled_layout((64, 16))),
                    (dep1, _fixed_tiled_layout((64, 64))),
                ],
            )

        self.assertEqual(symbol_to_dim, {})


class TestSpanOverflowLargeShapeContract(InductorTestCase):
    """Unit-style coverage for the real large shape used in E2E testing."""

    def test_large_shape_planner_adapter_and_coarse_tile_match_manual_hint(self):
        auto_op = _pointwise_op(_E2E_SHAPE, name="auto_buf")
        manual_op = _pointwise_op(_E2E_SHAPE, name="manual_buf")

        # Layer 1: planner chooses the same H split observed in the E2E run.
        plan = plan_span_overflow_tile(auto_op, max_cores=4)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 1)
        self.assertEqual(plan.levels[0].split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.levels[0].is_reduction)

        with patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords",
            _out_coords_for_bhld,
        ):
            with patch("torch_spyre._inductor.coarse_tile.insert_tiling_propagation"):
                with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
                    # Layer 2: adapter emits the same group shape as user hints.
                    auto_graph = _graph([auto_op])
                    auto_groups = span_overflow_groups(auto_graph)
                    manual_graph = _graph([manual_op])
                    manual_groups = _manual_h_hint_group(manual_op)

                    self.assertEqual(len(auto_groups), 1)
                    self.assertEqual(len(manual_groups), 1)
                    self.assertEqual(auto_groups[0][1][0][1], sympy.Integer(5))
                    self.assertEqual(manual_groups[0][1][0][1], sympy.Integer(5))
                    self.assertEqual(auto_groups[0][1][0][1], sympy.Integer(5))
                    self.assertEqual(manual_groups[0][1][0][1], sympy.Integer(5))
                    # Span-overflow tiling is always an output dim (never reduction).
                    self.assertFalse(auto_op.dim_hints[0].is_reduction)
                    self.assertFalse(manual_op.dim_hints[0].is_reduction)
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
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_codegen_contains_auto_span_overflow_loop_spec(self):
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        y = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")

        def fn(x, y):
            return x + y

        cfn = torch.compile(fn, dynamic=False)
        with (
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
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_input_span_codegen_contains_auto_loop_spec(self):
        x = torch.randn(2, 20, 16, 64, dtype=torch.float16).to("spyre")

        def fn(x):
            return x.sum(dim=0)

        cfn = torch.compile(fn, dynamic=False)
        with (
            patch(_LAUNCH_JOBPLAN),
            patch(_PREPARE_KERNEL),
            patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x)

        self.assertTrue(source_codes)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src)
        self.assertIn("count=sympify('10')", src)
        self.assertIn("op='sum'", src)
        self.assertIn("tiled_symbols=[[sympify('c0')]]", src)

    @config.patch(
        {
            "sencores": 4,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_multilevel_codegen_contains_nested_auto_loop_specs(self):
        x = torch.randn(20, 16, 64, dtype=torch.float16).to("spyre")

        def fn(x):
            return x.sum(dim=-1)

        fake_plan = SpanOverflowTilePlan(
            levels=(
                SpanOverflowTileLevel(selected_host_dim=0, split_count=2),
                # host_dim=1 has size 16 (x is (20, 16, 64)); split_count must
                # evenly divide it, unlike the un-checked scalar 5 this
                # replaced.
                SpanOverflowTileLevel(selected_host_dim=1, split_count=4),
            ),
            chunking_infos=(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=1,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=0,
                    stick_elems=64,
                    reason="output span overflow",
                ),
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=1,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=1,
                    stick_elems=64,
                    reason="input span overflow for arg0",
                ),
            ),
            reason="output span overflow; input span overflow for arg0",
        )

        cfn = torch.compile(fn, dynamic=False)
        with (
            patch(_LAUNCH_JOBPLAN),
            patch(_PREPARE_KERNEL),
            patch("subprocess.run"),
            patch(
                "torch_spyre._inductor.coarse_tile.plan_span_overflow_tile",
                return_value=fake_plan,
            ),
        ):
            _, source_codes = run_and_get_code(cfn, x)

        self.assertTrue(source_codes)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src)
        self.assertIn("count=sympify('2')", src)
        self.assertIn("count=sympify('4')", src)
        self.assertIn("op='sum'", src)

    # test_lm_head_restickify_codegen_contains_auto_loop_spec removed: its
    # "restickify producer tiled, BMM consumer stays untiled" premise is not
    # reachable for this op pair. buf0 (the BMM) always independently detects
    # the same overflow buf1 (the restickified weight) does, because buf0's
    # own candidate search reads buf1's full, undivided output size -- it has
    # no way to know buf1 will later be sliced. Confirmed empirically across
    # several (x, weight) shapes: buf1 always gets a plan, and whenever buf0's
    # own search completes, it does too, hitting the same producer-consumer
    # guard test_lm_head_auto_tiled_restickify_consumer_fails_safe already
    # covers -- so this test always asserted the same outcome as that one.

    @patch("torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
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
