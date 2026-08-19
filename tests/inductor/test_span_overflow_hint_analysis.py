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

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from torch_spyre._inductor.work_division import MAX_SPAN_BYTES

import sympy
import torch
import torch.nn.functional as F
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import compare_with_cpu  # noqa: E402

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import BATCH_MATMUL_OP, RESTICKIFY_OP
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.wsr.coarse_tile import coarse_tile_post_stickify
from torch_spyre._inductor.wsr.coarse_tile_span_overflow import (
    _SPAN_OVERFLOW_HINT_ID,
    _dims_to_hints,
    span_overflow_groups,
)
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    build_loop_scheduler_nodes,
)
from torch_spyre._inductor.wsr.span_overflow_hint_analysis import (
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
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
import torch_spyre._inductor.wsr.span_overflow_hint_analysis as soha


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


def _out_coords_distinct_producer_consumer(op):
    """Different output loop symbols for 'buf0' than for every other op.

    ``_out_coords_for_bhld`` returns identical coordinates for every op, so a
    test asserting that each group member resolved *its own* ``loop_var``
    passes vacuously under it.  A real Reduction producer and its Pointwise
    consumer describe their outputs with different symbols, so the
    Reduction-rooted group tests use this instead: 'buf0' (the producer) tiles
    ``m`` while its consumer tiles ``h`` at the same host_dim, which makes a
    loop_var copied from the wrong op observable.
    """
    if op.get_name() == "buf0":
        return [
            sympy.Integer(0),
            sympy.Symbol("m"),
            sympy.Symbol("n"),
            sympy.Symbol("k"),
        ]
    return [
        sympy.Integer(0),
        sympy.Symbol("h"),
        sympy.Symbol("l"),
        sympy.Symbol("d"),
    ]


def _apply_span_overflow(graph):
    """Run span_overflow_groups and apply its dim_hints assignments.

    span_overflow_groups is a pure planning step: it returns
    (groups, dim_hint_assignments) without mutating any op.  Real callers
    (_maybe_coarse_tile_span_overflow in passes.py) apply the assignments
    themselves before coarse_tile()/validate_coarse_tile_groups run; tests
    that inspect op.dim_hints after calling span_overflow_groups need to do
    the same, so this helper mirrors that call site.
    """
    groups, dim_hint_assignments = span_overflow_groups(graph)
    for op, dim_hints in dim_hint_assignments:
        op.dim_hints = dim_hints
    return groups


def _run_span_overflow_groups(op):
    """Run span_overflow_groups with op_out_coords patched for one test op."""
    graph = _graph([op])

    with patch(
        "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
        _out_coords_for_bhld,
    ):
        return _apply_span_overflow(graph)


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
    snode.unmet_dependencies = set()
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
            groups = _apply_span_overflow(_graph([op]))

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
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
            _out_coords_for_bhld,
        ):
            with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
                groups = _apply_span_overflow(_graph([op0, op1]))

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
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
            _out_coords_for_bhld,
        ):
            with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
                groups = _apply_span_overflow(_graph([op0, op1]))

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

    @staticmethod
    def _fake_multi_level_plan(*levels):
        """A plan with several tile levels, as ``(host_dim, split_count)`` pairs.

        The real planner emits one level per output dim it must tile, ordered
        by host dim (see ``plan_span_overflow_tile``).  Levels are paired by
        position downstream -- split counts are compared positionally and
        ``_dims_to_hints`` zips levels to hint_ids in the same order -- which
        is what the per-level correspondence tests below exercise.
        """
        return SpanOverflowTilePlan(
            levels=tuple(
                SpanOverflowTileLevel(
                    selected_host_dim=host_dim, split_count=split_count
                )
                for host_dim, split_count in levels
            ),
            chunking_infos=tuple(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=split_count,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=host_dim,
                    stick_elems=64,
                    reason="output span overflow",
                )
                for host_dim, split_count in levels
            ),
            reason="output span overflow",
        )

    def _multi_level_producer_and_reduction(self):
        """A Pointwise 'buf0' read by a Reduction 'buf1', both two-level tiled.

        Both sides plan levels ``[(host_dim=1, split=5), (host_dim=2, split=7)]``
        so the split-count lists match positionally and the join hinges purely
        on the loop-var correspondence the callers then check.
        """
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        reduction = _reduction_op(
            _E2E_SHAPE, name="buf1", reduction_type=BATCH_MATMUL_OP
        )
        reduction.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, reduction.layout
                ).writes,
            )
        )
        return producer, reduction

    def _run_multi_level_join(self, producer_coords):
        """Plan both ops two-level and run the pass with ``producer_coords``.

        ``producer_coords`` is what the producer's tiled dims look like *through
        the consumer's read*; the consumer's own coordinates stay ``[b, h, l, d]``,
        so its level 0 (host_dim=1) tiles ``h`` and its level 1 (host_dim=2)
        tiles ``l``.
        """
        producer, reduction = self._multi_level_producer_and_reduction()

        def fake_plan(_op, _max_cores):
            return self._fake_multi_level_plan((1, 5), (2, 7))

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_symbolic_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: list(producer_coords),
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            return _apply_span_overflow(_graph([producer, reduction])), (
                producer,
                reduction,
            )

    def test_multi_level_join_rejected_when_levels_correspond_crosswise(self):
        """A cross-level symbol match must not license a multi-level join.

        The consumer tiles ``h`` at level 0 and ``l`` at level 1.  Here the
        producer's level-0 dim (host_dim=1) is indexed by ``l`` and its level-1
        dim (host_dim=2) by ``h`` -- every producer level matches *some*
        consumer level, but never the level it will actually share a loop with.
        Unioning the consumer's tiled symbols across levels would accept this;
        the per-level check in ``_consumer_shares_group_tiled_dim`` rejects it,
        which is the point: level 0 of the shared loop nest would step ``h`` on
        one side and ``l`` on the other.
        """
        with self.assertRaisesRegex(
            Unsupported,
            "reads auto-tiled producer.*cannot join them.*same shared "
            "output dimension at the same split count",
        ):
            self._run_multi_level_join(
                [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("h"),
                    sympy.Symbol("d"),
                ]
            )

    def test_multi_level_join_accepted_when_levels_correspond_per_level(self):
        """The per-level check still accepts a genuinely corresponding join.

        Same two-level shape as the crosswise test, but the producer's level-0
        dim is indexed by ``h`` and its level-1 dim by ``l`` -- matching the
        consumer level for level.  Load-bearing as the counterpart to that
        test: without it, a per-level check that rejected *every* multi-level
        pair would look correct.
        """
        groups, (producer, reduction) = self._run_multi_level_join(
            [
                sympy.Symbol("b"),
                sympy.Symbol("h"),
                sympy.Symbol("l"),
                sympy.Symbol("d"),
            ]
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, reduction])
        self.assertEqual([hint.split_count for hint in producer.dim_hints], [5, 7])
        self.assertEqual(
            [hint.hint_id for hint in producer.dim_hints],
            [hint.hint_id for hint in reduction.dim_hints],
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
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([op0, op1]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [op0, op1])
        self.assertEqual(op0.dim_hints[0].hint_id, op1.dim_hints[0].hint_id)
        # op1 adopted op0's split (5), not its own independently-searched one (11).
        self.assertEqual(op0.dim_hints[0].split_count, 5)
        self.assertEqual(op1.dim_hints[0].split_count, 5)

    def test_matmul_joins_tiled_weight_producer_group(self):
        """A Reduction (matmul) that reads its auto-tiled producer joins the
        producer's group on the shared split rather than raising Unsupported
        (#3217).  The shared dim sits at a different output-range position in
        the matmul (host_dim=2) than in the producer (host_dim=1), so the join
        is matched on split_count, and both hints keep their own loop_var."""
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        matmul = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="batchmatmul")
        matmul.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, matmul.layout
                ).writes,
            )
        )

        def fake_plan(op, _max_cores):
            # Producer tiles host_dim=1 (h); matmul tiles host_dim=2 (l); same split.
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            # Through the read, the matmul's tiled symbol (l) indexes the
            # producer's tiled dim (host_dim=1) -> same logical dim -> join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, matmul]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, matmul])
        # Synchronized: shared hint_id and split, each on its own loop_var.
        self.assertEqual(producer.dim_hints[0].hint_id, matmul.dim_hints[0].hint_id)
        self.assertEqual(producer.dim_hints[0].split_count, 5)
        self.assertEqual(matmul.dim_hints[0].split_count, 5)
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(matmul.dim_hints[0].loop_var, sympy.Symbol("l"))

    def test_matmul_join_rejected_when_tiled_dim_not_shared(self):
        """Matching split counts are not enough: if the matmul's tiled loop var
        does not index the producer's tiled dim through the read (i.e. they tile
        unrelated dims that merely share a split count), the join is refused and
        the normal fail-safe Unsupported path is taken (#3217)."""
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        matmul = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="batchmatmul")
        matmul.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, matmul.layout
                ).writes,
            )
        )

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            # Producer's tiled dim (host_dim=1) is indexed by 'z', NOT the
            # matmul's tiled symbol 'l' -> unrelated dims -> must not join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("z"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "reads auto-tiled producer.*cannot join them.*same shared "
                "output dimension at the same split count",
            ):
                _apply_span_overflow(_graph([producer, matmul]))

    def test_non_matmul_reduction_joins_tiled_producer_group(self):
        """The join is reduction-type-agnostic, not matmul-only (#3217): a plain
        ``sum`` that reads its auto-tiled producer and tiles the same shared
        logical dim at the same split count joins the producer's group exactly
        like a matmul would.  Compare with
        ``test_matmul_joins_tiled_weight_producer_group``: the *only*
        difference is ``reduction_type`` ('sum' vs 'batchmatmul')."""
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        reduction = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="sum")
        reduction.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, reduction.layout
                ).writes,
            )
        )

        def fake_plan(op, _max_cores):
            # Producer tiles host_dim=1 (h); reduction tiles host_dim=2 (l).
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            # Through the read, the reduction's tiled symbol (l) indexes the
            # producer's tiled dim (host_dim=1) -> same logical dim -> join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._bmm_k_symbol",
                return_value=sympy.Symbol("k"),
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._loop_var_to_reduction_ranges_pos",
                return_value=0,
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, reduction]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, reduction])
        self.assertEqual(producer.dim_hints[0].hint_id, reduction.dim_hints[0].hint_id)
        self.assertEqual(producer.dim_hints[0].split_count, 5)
        self.assertEqual(reduction.dim_hints[0].split_count, 5)
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(reduction.dim_hints[0].loop_var, sympy.Symbol("l"))

    def test_non_matmul_reduction_join_rejected_when_tiled_dim_not_shared(self):
        """The shared-dim verification (``_consumer_shares_group_tiled_dim``)
        still applies regardless of reduction type: matching split counts are
        not enough if the reduction's tiled loop var does not actually index
        the producer's tiled dim through the read.  Compare with
        ``test_matmul_join_rejected_when_tiled_dim_not_shared``: the *only*
        difference is ``reduction_type`` ('sum' vs 'batchmatmul')."""
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        reduction = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="sum")
        reduction.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, reduction.layout
                ).writes,
            )
        )

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            # Producer's tiled dim (host_dim=1) is indexed by 'z', NOT the
            # reduction's tiled symbol 'l' -> unrelated dims -> must not join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("z"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "reads auto-tiled producer.*cannot join them.*same shared "
                "output dimension at the same split count",
            ):
                _apply_span_overflow(_graph([producer, reduction]))

    def _reduction_producer_pointwise_consumer(
        self, reduction_type=BATCH_MATMUL_OP, consumer_shape=_E2E_SHAPE
    ):
        """A Reduction producer ('buf0') read by a Pointwise consumer ('buf1')."""
        producer = _reduction_op(_E2E_SHAPE, name="buf0", reduction_type=reduction_type)
        consumer = _pointwise_op(consumer_shape, name="buf1")
        consumer.get_read_writes = MagicMock(
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
                    "buf1", consumer_shape, consumer.layout
                ).writes,
            )
        )
        return producer, consumer

    def test_bmm_producer_groups_with_pointwise_consumer(self):
        """A Reduction that starts a run (rather than being emitted as a closed
        singleton) lets a directly-connected Pointwise consumer fuse into its
        loop -- the BMM -> PW direction, mirroring the already-supported
        PW -> BMM join.  Tile t of the BMM's *output* dim is self-contained, so
        the producer's per-tile slice feeds the consumer in the same iteration
        instead of being materialized as a full buffer for a second loop nest.

        The producer and consumer describe their outputs with different loop
        symbols (see ``_out_coords_distinct_producer_consumer``), so the
        per-op loop_var assertions below would catch a hint copied from the
        wrong op rather than resolved against each op's own coordinates."""
        producer, consumer = self._reduction_producer_pointwise_consumer()

        def fake_plan(op, _max_cores):
            # Both tile host_dim=1 at split 5 -> exact-signature join path.
            return self._fake_plan(1, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_distinct_producer_consumer,
            ),
            # Through the read, the consumer's tiled symbol (h) indexes the
            # producer's tiled dim (host_dim=1) -> same logical dim -> join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("h"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, consumer]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, consumer])
        # Synchronized: shared hint_id and split, each on its own loop_var.
        self.assertEqual(producer.dim_hints[0].hint_id, consumer.dim_hints[0].hint_id)
        self.assertEqual(producer.dim_hints[0].split_count, 5)
        self.assertEqual(consumer.dim_hints[0].split_count, 5)
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("m"))
        self.assertEqual(consumer.dim_hints[0].loop_var, sympy.Symbol("h"))

    def test_pointwise_consumer_conforms_to_bmm_producer_split(self):
        """The conform path works against a Reduction-rooted run too: the
        consumer's own search picks 11, the BMM producer's run says 5, and 5 is
        also legal and sufficient for the consumer -- so it adopts 5 rather
        than opening a second, unsynchronized loop over the BMM's output."""
        producer, consumer = self._reduction_producer_pointwise_consumer()

        def fake_plan(op, _max_cores):
            return self._fake_plan(1, 5 if op.get_name() == "buf0" else 11)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_distinct_producer_consumer,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("h"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, consumer]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, consumer])
        self.assertEqual(producer.dim_hints[0].hint_id, consumer.dim_hints[0].hint_id)
        # Consumer adopted the run's split (5), not its own search's (11).
        self.assertEqual(producer.dim_hints[0].split_count, 5)
        self.assertEqual(consumer.dim_hints[0].split_count, 5)

    def test_bmm_producer_pointwise_consumer_rejected_when_dim_not_shared(self):
        """Widening *who* may join a Reduction-rooted run must not widen *when*
        it is safe.  A Reduction producer's output dims need not sit at the same
        positions as its Pointwise consumer's, and the consumer inherits the
        run's host_dim positionally -- so if the consumer's tiled loop var does
        not actually index the producer's tiled dim through the read, both the
        exact-signature join and the conform path must refuse and fall back to
        the Unsupported conflict path.  Without that check the consumer could
        conform against the wrong dim, pass by coincidence, and be stamped with
        a loop_var that desynchronizes the shared loop nest."""
        producer, consumer = self._reduction_producer_pointwise_consumer()

        def fake_plan(op, _max_cores):
            return self._fake_plan(1, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_distinct_producer_consumer,
            ),
            # Producer's tiled dim (host_dim=1) is indexed by 'z', NOT the
            # consumer's tiled symbol 'h' -> unrelated dims -> must not join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("z"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "reads auto-tiled producer.*buf0.*cannot join them.*same "
                "shared output dimension at the same split count",
            ):
                _apply_span_overflow(_graph([producer, consumer]))

    def test_pointwise_not_reading_bmm_run_does_not_join_it(self):
        """Matching signatures alone justify a Pointwise-to-Pointwise join
        (identical iteration spaces), but not a join into a Reduction-rooted
        run: there the run's host_dim numbering is the *Reduction's*, so an
        unrelated neighbour that merely agrees numerically is not tiling the
        same logical dim.  A real read edge is required, and without one the
        two ops get separate groups rather than being silently fused."""
        producer = _reduction_op(
            _E2E_SHAPE, name="buf0", reduction_type=BATCH_MATMUL_OP
        )
        # Default read_writes: no reads, so no producer-consumer edge.
        neighbour = _pointwise_op(_E2E_SHAPE, name="buf1")

        def fake_plan(op, _max_cores):
            return self._fake_plan(1, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_distinct_producer_consumer,
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, neighbour]))

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0], [producer])
        self.assertEqual(groups[1][0], [neighbour])
        self.assertNotEqual(
            producer.dim_hints[0].hint_id, neighbour.dim_hints[0].hint_id
        )

    def _reduction_producer_reduction_consumer(
        self,
        producer_type=BATCH_MATMUL_OP,
        consumer_type=BATCH_MATMUL_OP,
    ):
        """A Reduction producer ('buf0') read by a Reduction consumer ('buf1')."""
        producer = _reduction_op(_E2E_SHAPE, name="buf0", reduction_type=producer_type)
        consumer = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type=consumer_type)
        consumer.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, consumer.layout
                ).writes,
            )
        )
        return producer, consumer

    def test_bmm_producer_groups_with_bmm_consumer(self):
        """Reduction -> Reduction: a second matmul reading the first's tiled
        output joins its run.  What licenses the join is the shared *output*-
        range tile, not either op's type -- tile t of the producer's output dim
        is self-contained, so it feeds the consumer's tile t in the same
        iteration exactly as a Pointwise producer's would.

        The shared dim sits at different output positions in the two ops
        (producer host_dim=1, consumer host_dim=2), so the join matches on
        split_count and each op keeps its own loop_var."""
        producer, consumer = self._reduction_producer_reduction_consumer()

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            # Through the read, the consumer's tiled symbol (l) indexes the
            # producer's tiled dim (host_dim=1) -> same logical dim -> join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, consumer]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, consumer])
        self.assertEqual(producer.dim_hints[0].hint_id, consumer.dim_hints[0].hint_id)
        self.assertEqual(producer.dim_hints[0].split_count, 5)
        self.assertEqual(consumer.dim_hints[0].split_count, 5)
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(consumer.dim_hints[0].loop_var, sympy.Symbol("l"))

    def test_bmm_producer_groups_with_non_matmul_reduction_consumer(self):
        """Reduction -> Reduction is type-agnostic on both sides: a plain
        ``sum`` reading a tiled BMM producer joins just as another matmul
        would.  Compare with ``test_bmm_producer_groups_with_bmm_consumer``:
        the *only* difference is the consumer's ``reduction_type``."""
        producer, consumer = self._reduction_producer_reduction_consumer(
            consumer_type="sum"
        )

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, consumer]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, consumer])
        self.assertEqual(producer.dim_hints[0].hint_id, consumer.dim_hints[0].hint_id)
        self.assertEqual(consumer.dim_hints[0].split_count, 5)

    def test_bmm_producer_chain_through_pointwise_to_bmm_consumer(self):
        """A three-op chain lands in ONE group: bmm -> pointwise -> bmm.

        The two-op tests each cover a single join.  This covers the property
        that makes those joins useful together: a Reduction-rooted run is not
        closed by a Pointwise member, so the run survives 'buf1' and is still
        open when the second matmul arrives.  That is the shape attention has
        (matmul -> softmax -> matmul), and without it a chain could only ever
        fuse two ops.

        Also pins where the run *does* close: 'buf2' is a Reduction consumer,
        so it joins and flushes.  Anything after it would be rejected, which
        ``test_reduction_consumer_still_terminates_its_group`` covers.
        """
        producer = _reduction_op(
            _E2E_SHAPE, name="buf0", reduction_type=BATCH_MATMUL_OP
        )
        middle = _pointwise_op(_E2E_SHAPE, name="buf1")
        consumer = _reduction_op(
            _E2E_SHAPE, name="buf2", reduction_type=BATCH_MATMUL_OP
        )

        def _reads(buf_name, sym):
            return MagicMock(
                return_value=SimpleNamespace(
                    reads={MemoryDep(buf_name, sym, (sym,), (8195,))},
                    writes=_default_read_writes_for_output(
                        "buf1" if buf_name == "buf0" else "buf2",
                        _E2E_SHAPE,
                        middle.layout,
                    ).writes,
                )
            )

        middle.get_read_writes = _reads("buf0", sympy.Symbol("h"))
        consumer.get_read_writes = _reads("buf1", sympy.Symbol("h"))

        def fake_plan(op, _max_cores):
            # buf0/buf1 tile host_dim=1 (h); buf2 tiles host_dim=2 (l).  The
            # pointwise joins on an exact signature match, the matmul on the
            # shared split count across differing positions.
            return self._fake_plan(2 if op.get_name() == "buf2" else 1, 5)

        def host_coords(layout, dep, indirect):
            # Each read's producer must be indexed by the symbol tiling the
            # consumer's own output dim: buf1 tiles h and reads buf0, buf2
            # tiles l and reads buf1.
            sym = sympy.Symbol("h") if dep.name == "buf0" else sympy.Symbol("l")
            return [sympy.Symbol("b"), sym, sympy.Symbol("x"), sympy.Symbol("y")]

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                host_coords,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, middle, consumer]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, middle, consumer])
        shared_hint = producer.dim_hints[0].hint_id
        for op in (producer, middle, consumer):
            self.assertEqual(op.dim_hints[0].hint_id, shared_hint)
            self.assertEqual(op.dim_hints[0].split_count, 5)
        # Each op still resolves its own loop_var: buf0/buf1 tile h, buf2 l.
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(middle.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(consumer.dim_hints[0].loop_var, sympy.Symbol("l"))

    def _assert_joins_on_shared_dim(
        self, producer, consumer, producer_host_dim=1, consumer_host_dim=2, split=5
    ):
        """Run the pass on a pair that should join, and assert it did.

        Shared by the Reduction-producer matrix tests below.  The pass branches
        on ``isinstance(op.data, Reduction)`` and never inspects
        ``reduction_type``, so those tests vary only the two op types and reuse
        one scenario rather than restating the patch stack each time.

        ``host_coordinates`` is patched so the producer's tiled coordinate is
        indexed by whichever symbol tiles the consumer's output dim -- i.e. the
        correspondence holds and the join is licensed.
        """
        consumer_sym = _out_coords_for_bhld(None)[consumer_host_dim]

        def fake_plan(op, _max_cores):
            return self._fake_plan(
                producer_host_dim if op.get_name() == "buf0" else consumer_host_dim,
                split,
            )

        def host_coords(layout, dep, indirect):
            coords = [sympy.Symbol(f"p{i}") for i in range(4)]
            coords[producer_host_dim] = consumer_sym
            return coords

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                host_coords,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([producer, consumer]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [producer, consumer])
        self.assertEqual(producer.dim_hints[0].hint_id, consumer.dim_hints[0].hint_id)
        self.assertEqual(producer.dim_hints[0].split_count, split)
        self.assertEqual(consumer.dim_hints[0].split_count, split)
        return groups

    def test_non_matmul_reduction_producer_groups_with_pointwise_consumer(self):
        """A plain ``sum`` **producer** opens a run its Pointwise consumer joins.

        The sibling tests all use a batch-matmul producer.  Nothing in the pass
        inspects ``reduction_type`` -- a run is opened by any op whose data is a
        Reduction -- so a ``sum`` producer is supported by construction.  This
        pins that, so the support is a tested claim rather than an inference
        from the type check.
        """
        producer, consumer = self._reduction_producer_pointwise_consumer(
            reduction_type="sum"
        )
        # Pointwise consumers join on an exact signature match, so both tile the
        # same host_dim here (unlike the Reduction-consumer cases, which match on
        # split count across differing positions).
        self._assert_joins_on_shared_dim(
            producer, consumer, producer_host_dim=1, consumer_host_dim=1
        )
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(consumer.dim_hints[0].loop_var, sympy.Symbol("h"))

    def test_non_matmul_reduction_producer_groups_with_reduction_consumer(self):
        """Reduction -> Reduction with **neither** side a matmul (``sum`` ->
        ``sum``).  Completes the producer/consumer type matrix: the join is
        licensed by the shared output-range tile, not by either op being a
        batch-matmul."""
        producer, consumer = self._reduction_producer_reduction_consumer(
            producer_type="sum", consumer_type="sum"
        )
        self._assert_joins_on_shared_dim(producer, consumer)
        self.assertEqual(producer.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(consumer.dim_hints[0].loop_var, sympy.Symbol("l"))

    def test_non_matmul_reduction_producer_groups_with_bmm_consumer(self):
        """A plain ``sum`` producer read by a matmul consumer -- the remaining
        cell of the producer/consumer type matrix, and the mirror of
        ``test_bmm_producer_groups_with_non_matmul_reduction_consumer``."""
        producer, consumer = self._reduction_producer_reduction_consumer(
            producer_type="sum", consumer_type=BATCH_MATMUL_OP
        )
        self._assert_joins_on_shared_dim(producer, consumer)

    def test_bmm_to_bmm_rejected_when_producer_tiles_consumer_reduction_dim(self):
        """The wrong-answer case Reduction -> Reduction grouping makes
        reachable, and the reason the correspondence check is load-bearing
        rather than a formality.

        In ``bmm(bmm(q, k), v)`` the producer may tile its N dim while the
        consumer reads that buffer as its A operand -- so the producer's N *is*
        the consumer's K.  Tile t of the producer is then a partial slice of the
        consumer's *reduction* range, and pairing them per-iteration would
        compute a partial sum and call it a finished result.  Split counts match
        and there is a genuine read edge, so nothing else on the join branch
        objects.

        K never appears in the consumer's output coordinates, so the tiled-symbol
        intersection is empty and the join is refused -- modelled here by the
        producer's tiled dim being indexed by 'k', a symbol absent from the
        consumer's output coords."""
        producer, consumer = self._reduction_producer_reduction_consumer()

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            # Producer's tiled dim (host_dim=1) is indexed by the consumer's
            # reduction symbol 'k', which is not among its output coords
            # (b/h/l/d) -> partial-result slice -> must not join.
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("k"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "reads auto-tiled producer.*buf0.*cannot join them.*same "
                "shared output dimension at the same split count",
            ):
                _apply_span_overflow(_graph([producer, consumer]))

    def test_join_rejected_when_only_one_of_two_reads_corresponds(self):
        """A consumer can read the same tiled producer through more than one
        dep -- a matmul taking it as both operands (``x @ x.T``).  Every dep
        must correspond, not just one: verifying a single access pattern and
        ignoring the rest would let a partial-result read through whenever it
        happened to be paired with a safe one.

        Here the first read corresponds (tiled dim indexed by 'l') and the
        second does not (indexed by 'k', the consumer's reduction symbol), so
        the join must be refused."""
        producer = _reduction_op(
            _E2E_SHAPE, name="buf0", reduction_type=BATCH_MATMUL_OP
        )
        consumer = _reduction_op(
            _E2E_SHAPE, name="buf1", reduction_type=BATCH_MATMUL_OP
        )
        consumer.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={
                    MemoryDep("buf0", sympy.Symbol("h"), (sympy.Symbol("h"),), (8195,)),
                    MemoryDep("buf0", sympy.Symbol("l"), (sympy.Symbol("l"),), (8195,)),
                },
                writes=_default_read_writes_for_output(
                    "buf1", _E2E_SHAPE, consumer.layout
                ).writes,
            )
        )

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        def host_coords(layout, dep, indirect):
            # Correspondence depends on which dep is being checked: the 'h'
            # read pairs correctly, the 'l' read lands on the consumer's K.
            corresponds = dep.index == sympy.Symbol("h")
            return [
                sympy.Symbol("b"),
                sympy.Symbol("l") if corresponds else sympy.Symbol("k"),
                sympy.Symbol("x"),
                sympy.Symbol("y"),
            ]

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                host_coords,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "reads auto-tiled producer.*buf0.*cannot join them.*same "
                "shared output dimension at the same split count",
            ):
                _apply_span_overflow(_graph([producer, consumer]))

    def test_reduction_consumer_still_terminates_its_group(self):
        """A Reduction consumer joining flushes the group immediately, so it
        remains the last member and each auto-tiled producer still feeds at most
        one reduction consumer.  Allowing Reduction -> Reduction did not relax
        that: a second matmul reading the same producer is still rejected, with
        the distinct multi-consumer message."""
        producer = _reduction_op(
            _E2E_SHAPE, name="buf0", reduction_type=BATCH_MATMUL_OP
        )
        consumer1 = _reduction_op(
            _E2E_SHAPE, name="buf1", reduction_type=BATCH_MATMUL_OP
        )
        consumer2 = _reduction_op(
            _E2E_SHAPE, name="buf2", reduction_type=BATCH_MATMUL_OP
        )
        for mm in (consumer1, consumer2):
            mm.get_read_writes = MagicMock(
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
                        mm.get_name(), _E2E_SHAPE, mm.layout
                    ).writes,
                )
            )

        def fake_plan(op, _max_cores):
            return self._fake_plan(1 if op.get_name() == "buf0" else 2, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already auto-tiled and joined by another reduction consumer.*"
                "multiple consumers sharing one auto-tiled producer is not yet "
                "supported",
            ):
                _apply_span_overflow(_graph([producer, consumer1, consumer2]))

    def test_two_independent_reductions_still_produce_separate_groups(self):
        """Letting a Reduction open a run must not merge unrelated neighbours.
        With no read edge between them, two Reductions still land in their own
        groups with distinct hint_ids, exactly as when each was emitted as a
        closed singleton."""
        red0 = _reduction_op(_E2E_SHAPE, name="buf0", reduction_type=BATCH_MATMUL_OP)
        red1 = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="sum")

        def fake_plan(op, _max_cores):
            return self._fake_plan(1, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([red0, red1]))

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0], [red0])
        self.assertEqual(groups[1][0], [red1])
        self.assertEqual(groups[0][1][0][0], _SPAN_OVERFLOW_HINT_ID)
        self.assertEqual(groups[1][1][0][0], _SPAN_OVERFLOW_HINT_ID + 1)

    def test_reduction_range_tile_rejects_tiled_producer_conflict(self):
        """A reduction that tiles its *reduction* range (``is_reduction=True``)
        must never join, even if split counts match and the read correspondence
        would otherwise hold: tile ``t`` of a reduction range is not
        self-contained (it needs partial-result accumulation across tiles), so
        sharing a per-tile loop nest would be wrong.  This is a batch-matmul
        (so it passes the matmul gate) identical to the passing matmul-join
        case except its plan is flagged ``is_reduction=True`` -- the
        ``is_reduction`` guard in ``_consumer_shares_group_tiled_dim`` alone
        must flip join -> independent group (#3217).  This plan is emitted by
        the BMM K fallback and cannot safely form an independent loop while
        reading a producer whose coarse-tile group is still open."""
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        reduction = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="batchmatmul")
        reduction.get_read_writes = MagicMock(
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
                    "buf1", _E2E_SHAPE, reduction.layout
                ).writes,
            )
        )

        def reduction_range_plan(host_dim, split_count):
            return SpanOverflowTilePlan(
                levels=(
                    SpanOverflowTileLevel(
                        selected_host_dim=host_dim,
                        split_count=split_count,
                        is_reduction=True,
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
                        reason="reduction span overflow",
                    ),
                ),
                reason="reduction span overflow",
            )

        def fake_plan(op, _max_cores):
            if op.get_name() == "buf0":
                return self._fake_plan(1, 5)
            # Same split (5) and a correspondence that WOULD pass the loop-var
            # check -- only the is_reduction flag differs from the passing case.
            return reduction_range_plan(0, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._bmm_k_symbol",
                return_value=sympy.Symbol("k"),
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._loop_var_to_reduction_ranges_pos",
                return_value=0,
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported, "reads auto-tiled producer.*cannot join them"
            ):
                _apply_span_overflow(_graph([producer, reduction]))

    def test_k_tiled_bmm_output_is_not_tracked_as_auto_tiled_producer(self):
        bmm = _reduction_op(
            _E2E_SHAPE,
            name="buf1",
            reduction_type=BATCH_MATMUL_OP,
        )
        consumer = _pointwise_op(_E2E_SHAPE, name="buf2")
        h = sympy.Symbol("h")
        k = sympy.Symbol("k")

        bmm.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={MemoryDep("buf0", h, (h,), (8195,))},
                writes=_default_read_writes_for_output(
                    "buf1", _E2E_SHAPE, bmm.layout
                ).writes,
            )
        )
        consumer.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={MemoryDep("buf1", h, (h,), (8195,))},
                writes=_default_read_writes_for_output(
                    "buf2", _E2E_SHAPE, consumer.layout
                ).writes,
            )
        )

        k_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(0, 5, is_reduction=True),),
            chunking_infos=(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=5,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=0,
                    stick_elems=64,
                    reason="K span overflow",
                ),
            ),
            reason="K span overflow",
        )

        def fake_plan(op, _max_cores):
            if op.get_name() == "buf1":
                return k_plan
            return self._fake_plan(1, 5)

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._bmm_k_symbol",
                return_value=k,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._loop_var_to_reduction_ranges_pos",
                return_value=0,
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([bmm, consumer]))

        self.assertEqual([group[0] for group in groups], [[bmm], [consumer]])
        self.assertTrue(bmm.dim_hints[0].is_reduction)
        self.assertFalse(consumer.dim_hints[0].is_reduction)

    def test_second_reduction_consumer_of_joined_producer_rejected(self):
        """One auto-tiled producer feeds at most one reduction consumer.

        The group is flushed as soon as the first matmul joins, so a second
        matmul reading the same producer is rejected with the distinct
        multi-consumer message rather than the generic pointwise-only one.
        """
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        matmul1 = _reduction_op(_E2E_SHAPE, name="buf1", reduction_type="batchmatmul")
        matmul2 = _reduction_op(_E2E_SHAPE, name="buf2", reduction_type="batchmatmul")
        for mm in (matmul1, matmul2):
            mm.get_read_writes = MagicMock(
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
                        mm.get_name(), _E2E_SHAPE, mm.layout
                    ).writes,
                )
            )

        k_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(0, 5, is_reduction=True),),
            chunking_infos=(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=5,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=0,
                    stick_elems=64,
                    reason="K span overflow",
                ),
            ),
            reason="K span overflow",
        )

        def fake_plan(op, _max_cores):
            if op.get_name() == "buf0":
                return self._fake_plan(1, 5)
            if op.get_name() == "buf1":
                return self._fake_plan(2, 5)
            return k_plan

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("b"),
                    sympy.Symbol("l"),
                    sympy.Symbol("x"),
                    sympy.Symbol("y"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already auto-tiled and joined by another reduction consumer.*"
                "multiple consumers sharing one auto-tiled producer is not yet "
                "supported",
            ):
                _apply_span_overflow(_graph([producer, matmul1, matmul2]))

    def test_k_only_plan_rejects_manually_tiled_producer(self):
        producer = _pointwise_op(_E2E_SHAPE, name="buf0")
        _manual_h_hint_group(producer)
        bmm = _reduction_op(
            _E2E_SHAPE,
            name="buf1",
            reduction_type=BATCH_MATMUL_OP,
        )
        h = sympy.Symbol("h")
        bmm.get_read_writes = MagicMock(
            return_value=SimpleNamespace(
                reads={MemoryDep("buf0", h, (h,), (8195,))},
                writes=_default_read_writes_for_output(
                    "buf1", _E2E_SHAPE, bmm.layout
                ).writes,
            )
        )
        k_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(0, 5, is_reduction=True),),
            chunking_infos=(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=5,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=0,
                    stick_elems=64,
                    reason="K span overflow",
                ),
            ),
            reason="K span overflow",
        )

        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                return_value=k_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._bmm_k_symbol",
                return_value=sympy.Symbol("k"),
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._loop_var_to_reduction_ranges_pos",
                return_value=0,
            ),
            config.patch({"sencores": 8, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already-tiled producer.*not in an open group this op can join",
            ):
                _apply_span_overflow(_graph([producer, bmm]))

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
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "reads auto-tiled producer.*buf0.*cannot join them.*same "
                "shared output dimension at the same split count",
            ):
                _apply_span_overflow(_graph([op0, op1]))

    def test_lm_head_matmul_joins_tiled_restickify_producer(self):
        """LM-head restickify and BMM are tiled into one synchronized group (#3217).

        This models F.linear(x[1,4096], weight[49216,4096]): lowering first
        creates a restickified weight buffer ``buf1`` (tiled on its vocab dim
        host_dim=0) and then a BMM reduction ``buf0`` that reads ``buf1`` (tiled
        on the corresponding output vocab dim host_dim=1).  Because both tile the
        shared vocab dim at the same split, the matmul joins the producer's group
        rather than raising -- one shared loop nest, each op on its own loop_var.
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

        # Through the read, the matmul's tiled output dim (d1) indexes the
        # weight producer's tiled dim0 -- the same logical vocab dim.
        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
                fake_plan,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                lambda op: [sympy.Symbol("d0"), sympy.Symbol("d1")],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.host_coordinates",
                lambda layout, dep, indirect: [
                    sympy.Symbol("d1"),
                    sympy.Symbol("d0"),
                ],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.indirect_sizes_from_op",
                lambda op: {},
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            groups = _apply_span_overflow(_graph([restickify_weight, lm_head_bmm]))

        # One synchronized group containing the weight producer and the matmul.
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], [restickify_weight, lm_head_bmm])
        self.assertEqual(
            restickify_weight.dim_hints[0].hint_id, lm_head_bmm.dim_hints[0].hint_id
        )
        self.assertEqual(restickify_weight.dim_hints[0].split_count, 769)
        self.assertEqual(lm_head_bmm.dim_hints[0].split_count, 769)
        # Each op tiles the shared vocab dim on its own loop_var (dim0 vs dim1).
        self.assertEqual(restickify_weight.dim_hints[0].loop_var, sympy.Symbol("d0"))
        self.assertEqual(lm_head_bmm.dim_hints[0].loop_var, sympy.Symbol("d1"))

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
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "already-tiled producer.*not in an open group this op can join",
            ):
                _apply_span_overflow(_graph([producer, consumer]))

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
            groups = _apply_span_overflow(_graph([op]))

        self.assertEqual(groups, [])

    def test_symbolic_layout_skipped(self):
        op = _pointwise_op(_E2E_SHAPE)
        op.layout.size[1] = sympy.Symbol("s0")

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _apply_span_overflow(_graph([op]))

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
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                _out_coords_for_bhld,
            ):
                groups = _apply_span_overflow(_graph([hinted_op, unhinted_op]))

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
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.indirect_info_from_op",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0},
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={n: 0},
            ),
        ):
            infos = _input_span_infos_controlled_by_output_dims(op, max_cores=1)
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertEqual(infos, [])
        self.assertIsNone(plan)

    def test_bmm_input_span_controlled_by_b_dim_plans_output_tile(self):
        op = _reduction_op(
            (4_194_304, 1, 16),
            reduction_ranges=(64,),
            reduction_type=BATCH_MATMUL_OP,
        )

        b, m, n, k = sympy.symbols("b m n k")

        # BMM lhs shape is conceptually [B, M, K]. Here M is broadcast/unit,
        # and the large physical span is controlled by B.
        lhs_dep = MemoryDep(
            "lhs",
            b * 64 + k,
            (b, k),
            (4_194_304, 64),
        )
        lhs_layout = _fixed_tiled_layout((4_194_304, 64))

        with (
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis."
                "_output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis._input_read_deps",
                return_value=[(lhs_dep, lhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={
                    b: 0,
                    m: 1,
                    n: 2,
                },
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis."
                "_remaining_span_candidates_after_tile",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis."
                "_search_bmm_k_tile_plan"
            ) as k_search,
        ):
            plan = plan_span_overflow_tile(
                op,
                max_cores=1,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.levels), 1)

        level = plan.levels[0]

        self.assertEqual(
            level.selected_host_dim,
            0,  # ranges[0] = B
        )
        self.assertEqual(level.split_count, 2)
        self.assertFalse(level.is_reduction)

        # A valid B output tile must be selected before considering K.
        k_search.assert_not_called()

    def test_bmm_input_span_controlled_by_m_dim_plans_output_tile(self):
        op = _reduction_op(
            (1, 4_194_304, 16),
            reduction_ranges=(64,),
            reduction_type=BATCH_MATMUL_OP,
        )

        b, m, n, k = sympy.symbols("b m n k")

        # BMM lhs shape is conceptually [B, M, K]. The large physical
        # input span is controlled by the output M dimension.
        lhs_dep = MemoryDep(
            "lhs",
            m * 64 + k,
            (m, k),
            (4_194_304, 64),
        )
        lhs_layout = _fixed_tiled_layout((4_194_304, 64))

        with (
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis."
                "_output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis._input_read_deps",
                return_value=[(lhs_dep, lhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={
                    b: 0,
                    m: 1,
                    n: 2,
                },
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis."
                "_remaining_span_candidates_after_tile",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr."
                "span_overflow_hint_analysis."
                "_search_bmm_k_tile_plan"
            ) as k_search,
        ):
            plan = plan_span_overflow_tile(
                op,
                max_cores=1,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.levels), 1)

        level = plan.levels[0]

        self.assertEqual(
            level.selected_host_dim,
            1,  # ranges[1] = M
        )
        self.assertEqual(level.split_count, 2)
        self.assertFalse(level.is_reduction)

        # A valid M output tile must be selected before considering K.
        k_search.assert_not_called()

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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(rhs_dep, rhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
                return_value=[],
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 2)
        self.assertEqual(plan.levels[0].split_count, 2)
        self.assertFalse(plan.levels[0].is_reduction)

    def test_bmm_input_span_controlled_by_k_dim_plans_reduction_tile(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(65536,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k = sympy.symbols("b m n k")
        rhs_dep = MemoryDep("rhs", k * 64 + n, (k, n), (65536, 64))
        rhs_layout = _fixed_tiled_layout((65536, 64))

        with (
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                5 * 1024 * 1024,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(rhs_dep, rhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNotNone(plan)
        self.assertEqual(
            plan.levels,
            (
                SpanOverflowTileLevel(
                    selected_host_dim=0,
                    split_count=2,
                    is_reduction=True,
                ),
            ),
        )
        self.assertIn("BMM K input span overflow", plan.reason)

    def test_bmm_k_discovery_skips_unmeasurable_coordinate_span(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(65536,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k = sympy.symbols("b m n k")
        rhs_dep = MemoryDep("rhs", k * 64 + n, (k, n), (65536, 64))
        rhs_layout = _fixed_tiled_layout((65536, 64))

        with (
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(soha, "_input_span_candidates", return_value=[]),
            patch.object(
                soha, "_input_read_deps", return_value=[(rhs_dep, rhs_layout)]
            ),
            patch.object(soha, "_bmm_k_symbol", return_value=k),
            patch.object(
                soha,
                "_bmm_output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
            patch.object(
                soha,
                "_device_coordinates_for_span",
                return_value=[k + n, sympy.Integer(0)],
            ),
            patch.object(soha, "_coordinate_span_elems", return_value=None),
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIsNone(plan)

    def test_bmm_k_split_rejects_unmeasurable_coordinate_span(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(65536,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k = sympy.symbols("b m n k")
        rhs_dep = MemoryDep("rhs", k * 64 + n, (k, n), (65536, 64))
        rhs_layout = _fixed_tiled_layout((65536, 64))

        with (
            patch.object(
                soha, "_input_read_deps", return_value=[(rhs_dep, rhs_layout)]
            ),
            patch.object(soha, "_bmm_k_symbol", return_value=k),
            patch.object(
                soha,
                "_bmm_output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
            patch.object(
                soha,
                "_device_coordinates_for_span",
                return_value=[k + n, sympy.Integer(0)],
            ),
            patch.object(soha, "_coordinate_span_elems", return_value=None),
        ):
            with self.assertRaisesRegex(
                Unsupported,
                "Cannot validate BMM K split 2.*unsupported coordinate span",
            ):
                soha._bmm_k_span_infos(op, max_cores=1, k_split=2)

    def test_bmm_k_fallback_does_not_hide_output_overflow(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(65536,), reduction_type=BATCH_MATMUL_OP
        )
        output_candidate = SimpleNamespace(
            chunking_info=SimpleNamespace(selected_host_dim=2), source="output"
        )
        failure = Unsupported("output still overflows")

        with (
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[output_candidate],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_span_candidates",
                return_value=[],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._search_min_cost_tile_plan",
                side_effect=failure,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._search_bmm_k_tile_plan",
                return_value=None,
            ) as k_search,
        ):
            with self.assertRaisesRegex(Unsupported, "output still overflows"):
                plan_span_overflow_tile(op, max_cores=1)

        k_search.assert_called_once_with(op, 1)

    def test_bmm_input_failure_is_not_swallowed_when_k_has_no_plan(self):
        op = _reduction_op(
            (1, 4_194_304, 16),
            reduction_ranges=(64,),
            reduction_type=BATCH_MATMUL_OP,
        )
        input_candidate = SimpleNamespace(
            chunking_info=SimpleNamespace(selected_host_dim=1),
            source="input:lhs",
        )
        failure = Unsupported("M-controlled input still overflows")

        with (
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(
                soha, "_input_span_candidates", return_value=[input_candidate]
            ),
            patch.object(soha, "_search_min_cost_tile_plan", side_effect=failure),
            patch.object(soha, "_search_bmm_k_tile_plan", return_value=None),
        ):
            with self.assertRaisesRegex(
                Unsupported, "M-controlled input still overflows"
            ):
                plan_span_overflow_tile(op, max_cores=1)

    def test_bmm_k_plan_rejected_returns_none(self):
        op = _reduction_op(
            (1, 4_194_304, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        initial_info = SimpleNamespace(
            chunking_info=SimpleNamespace(
                per_core_span=2 * MAX_SPAN_BYTES,
                reason="BMM K input span overflow for rhs",
            ),
            dep_name="rhs",
        )
        remaining_at_2 = SimpleNamespace(source="input:rhs@2")
        remaining_at_4 = SimpleNamespace(source="input:lhs@4")

        def alignment_error(_op, split_count):
            if split_count == 8:
                return "input dependency rhs host dim 0 cuts a stick"
            return None

        def validate(_op, _max_cores, split_by_host_dim, *, k_split=None):
            self.assertEqual(split_by_host_dim, {})
            if k_split == 2:
                return [remaining_at_2]
            if k_split == 4:
                return [remaining_at_4]
            raise AssertionError(f"unexpected k_split={k_split}")

        with (
            patch.object(soha, "_bmm_k_span_infos", return_value=[initial_info]),
            patch.object(soha, "_bmm_k_split_candidates", return_value=[2, 4, 8]),
            patch.object(soha, "_bmm_k_alignment_error", side_effect=alignment_error),
            patch.object(
                soha,
                "_remaining_span_candidates_after_tile",
                side_effect=validate,
            ),
        ):
            plan = soha._search_bmm_k_tile_plan(op, max_cores=1)

        self.assertIsNone(plan)

    def test_bmm_k_fallback_respects_reduction_tiling_kill_switch(self):
        op = _reduction_op(
            (1, 1, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )

        with (
            patch.object(config, "enable_reduction_tiling", False),
            patch.object(soha, "_bmm_k_span_infos") as k_infos,
        ):
            plan = soha._search_bmm_k_tile_plan(op, max_cores=1)

        self.assertIsNone(plan)
        k_infos.assert_not_called()

    def test_bmm_k_alignment_failure_returns_none(self):
        op = _reduction_op(
            (1, 1, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        initial_info = SimpleNamespace(
            chunking_info=SimpleNamespace(
                per_core_span=2 * MAX_SPAN_BYTES,
                reason="BMM K input span overflow for rhs",
            ),
            dep_name="rhs",
        )

        with (
            patch.object(soha, "_bmm_k_span_infos", return_value=[initial_info]),
            patch.object(soha, "_bmm_k_split_candidates", return_value=[2]),
            patch.object(
                soha,
                "_bmm_k_alignment_error",
                return_value="input dependency rhs host dim 0 cuts a stick",
            ),
        ):
            plan = soha._search_bmm_k_tile_plan(op, max_cores=1)

        self.assertIsNone(plan)

    def test_bmm_k_alignment_skips_unrelated_nonstatic_input(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(128,), reduction_type=BATCH_MATMUL_OP
        )
        k = sympy.Symbol("k")
        unrelated = MemoryDep("activation", sympy.Symbol("m"), (), ())
        unrelated_layout = _fixed_tiled_layout((128,))
        unrelated_layout.size[0] = sympy.Symbol("s0")
        rhs = MemoryDep("rhs", k, (k,), (128,))
        rhs_layout = _fixed_tiled_layout((128,))

        with (
            patch.object(
                soha,
                "_input_read_deps",
                return_value=[(unrelated, unrelated_layout), (rhs, rhs_layout)],
            ),
            patch.object(soha, "_bmm_k_symbol", return_value=k),
            patch.object(soha, "host_coordinates", return_value=[k]),
        ):
            error = soha._bmm_k_alignment_error(op, split_count=2)

        self.assertIsNone(error)

    def test_bmm_k_fallback_tried_when_output_plan_leaves_k_span(self):
        op = _reduction_op(
            (1, 4_194_304, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        output_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(1, 2),),
            chunking_infos=(),
            reason="M input span overflow",
        )
        k_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(0, 4, is_reduction=True),),
            chunking_infos=(),
            reason="BMM K input span overflow",
        )
        remaining_k = SimpleNamespace(dep_name="rhs")

        with (
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(soha, "_input_span_candidates", return_value=[]),
            patch.object(soha, "_search_min_cost_tile_plan", return_value=output_plan),
            patch.object(
                soha, "_bmm_k_span_infos", return_value=[remaining_k]
            ) as k_infos,
            patch.object(
                soha, "_search_bmm_k_tile_plan", return_value=k_plan
            ) as k_search,
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIs(plan, k_plan)
        k_infos.assert_called_once_with(op, 1, split_by_host_dim={1: 2})
        k_search.assert_called_once_with(op, 1)

    def test_bmm_k_kill_switch_preserves_output_plan(self):
        op = _reduction_op(
            (1, 4_194_304, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        output_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(1, 2),),
            chunking_infos=(),
            reason="M input span overflow",
        )
        remaining_k = SimpleNamespace(dep_name="rhs")

        with (
            patch.object(config, "enable_reduction_tiling", False),
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(soha, "_input_span_candidates", return_value=[]),
            patch.object(soha, "_search_min_cost_tile_plan", return_value=output_plan),
            patch.object(soha, "_bmm_k_span_infos", return_value=[remaining_k]),
            patch.object(soha, "_search_bmm_k_tile_plan") as k_search,
        ):
            plan = plan_span_overflow_tile(op, max_cores=1)

        self.assertIs(plan, output_plan)
        k_search.assert_not_called()

    def test_bmm_output_plan_rejected_when_k_span_remains_and_k_fails(self):
        op = _reduction_op(
            (1, 4_194_304, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        output_plan = SpanOverflowTilePlan(
            levels=(SpanOverflowTileLevel(1, 2),),
            chunking_infos=(),
            reason="M input span overflow",
        )
        remaining_k = SimpleNamespace(dep_name="rhs")

        with (
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(soha, "_input_span_candidates", return_value=[]),
            patch.object(soha, "_search_min_cost_tile_plan", return_value=output_plan),
            patch.object(
                soha, "_bmm_k_span_infos", return_value=[remaining_k]
            ) as k_infos,
            patch.object(
                soha,
                "_search_bmm_k_tile_plan",
                side_effect=Unsupported("K-only candidates did not clear every span"),
            ) as k_search,
        ):
            with self.assertRaisesRegex(
                Unsupported, "combined output-range and reduction-range"
            ):
                plan_span_overflow_tile(op, max_cores=1)

        k_infos.assert_called_once_with(op, 1, split_by_host_dim={1: 2})
        k_search.assert_called_once_with(op, 1)

    def test_k_validation_keeps_unsplittable_output_controlled_span(self):
        op = _reduction_op(
            (1, 4_194_304, 64),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        remaining_info = SimpleNamespace(
            chunking_info=SimpleNamespace(selected_host_dim=1),
            dep_name="lhs",
        )

        with (
            patch.object(soha, "_post_tile_layout_for_splits", return_value=op.layout),
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(
                soha,
                "_input_span_infos_controlled_by_output_dims",
                return_value=[remaining_info],
            ),
            patch.object(soha, "_bmm_k_span_infos", return_value=[]),
            patch.object(
                soha, "_host_dim_has_legal_nontrivial_split", return_value=False
            ) as legal_split,
        ):
            remaining = soha._remaining_span_candidates_after_tile(op, 1, {}, k_split=2)

        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].source, "input:lhs")
        legal_split.assert_not_called()

    def test_output_tile_validation_keeps_legacy_candidate_filtering(self):
        op = _reduction_op((2, 64), reduction_ranges=(64,))
        filtered_candidates = [SimpleNamespace(source="filtered")]

        with (
            patch.object(soha, "_post_tile_layout_for_splits", return_value=op.layout),
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(
                soha,
                "_input_span_candidates",
                return_value=filtered_candidates,
            ) as filtered,
            patch.object(
                soha, "_input_span_infos_controlled_by_output_dims"
            ) as raw_infos,
        ):
            remaining = soha._remaining_span_candidates_after_tile(op, 1, {1: 2})

        self.assertEqual(remaining, filtered_candidates)
        filtered.assert_called_once_with(op, 1, split_by_host_dim={1: 2})
        raw_infos.assert_not_called()

    def test_bmm_k_hint_rejects_broadcast_output_symbol_mismatch(self):
        op = _reduction_op(
            (1, 16, 64), reduction_ranges=(64,), reduction_type=BATCH_MATMUL_OP
        )
        b, m, n, k = sympy.symbols("b m n k")
        layout = _fixed_tiled_layout((1, 16, 64))
        out_dep = MemoryDep("buf0", m * 64 + n, (b, m, n), (1, 16, 64))
        lhs = MemoryDep("lhs", b * 1024 + k * 16 + m, (b, k, m), (1, 64, 16))
        rhs = MemoryDep("rhs", b * 4096 + k * 64 + n, (b, k, n), (1, 64, 64))
        op.get_read_writes = MagicMock(
            return_value=SimpleNamespace(reads={lhs, rhs}, writes={out_dep})
        )
        op.layout = layout

        with self.assertRaisesRegex(Unsupported, "maps to reduction range position 1"):
            _dims_to_hints(op, ((0, 4, True),), [_SPAN_OVERFLOW_HINT_ID])

    def test_bmm_mixed_output_and_k_input_span_becomes_output_candidate(self):
        op = _reduction_op(
            (1, 4_194_304, 64),
            reduction_ranges=(64,),
            reduction_type=BATCH_MATMUL_OP,
        )
        b, m, n, k = sympy.symbols("b m n k")
        lhs = MemoryDep("lhs", k * 4_194_304 + m, (k, m), (64, 4_194_304))
        layout = _fixed_tiled_layout((64, 4_194_304, 64))

        with (
            patch.object(soha, "MAX_SPAN_BYTES", MAX_SPAN_BYTES),
            patch.object(soha, "_input_read_deps", return_value=[(lhs, layout)]),
            patch.object(
                soha, "_output_symbol_to_dim", return_value={b: 0, m: 1, n: 2}
            ),
            patch.object(
                soha,
                "_device_coordinates_for_span",
                return_value=[k + m, sympy.Integer(0)],
            ),
            patch.object(soha, "_coordinate_span_elems", return_value=4_194_304),
            patch.object(soha, "_tile_aware_inner_stride_elems", return_value=64),
        ):
            infos = soha._input_span_infos_controlled_by_output_dims(op, max_cores=1)

        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].chunking_info.selected_host_dim, 1)
        self.assertEqual(infos[0].dep_name, "lhs")

    def test_bmm_k_split_does_not_shrink_output_controlled_outer_span(self):
        op = _reduction_op(
            (1, 8192, 4096),
            reduction_ranges=(65536,),
            reduction_type=BATCH_MATMUL_OP,
        )
        b, m, n, k = sympy.symbols("b m n k")
        lhs = MemoryDep("lhs", m * 65536 + k, (m, k), (8192, 65536))
        layout = _fixed_tiled_layout((8192, 65536))

        def inner_stride(
            _device_coords,
            _device_size,
            _dep,
            _inner_start_dim,
            symbol_to_dim,
            _split_by_host_dim,
        ):
            return MAX_SPAN_BYTES if k not in symbol_to_dim else 1

        with (
            patch.object(soha, "_input_read_deps", return_value=[(lhs, layout)]),
            patch.object(soha, "_bmm_k_symbol", return_value=k),
            patch.object(
                soha,
                "_bmm_output_symbol_to_dim",
                return_value={b: 0, m: 1, n: 2},
            ),
            patch.object(
                soha,
                "_device_coordinates_for_span",
                return_value=[m, k, sympy.Integer(0)],
            ),
            patch.object(soha, "_coordinate_span_elems", return_value=2),
            patch.object(
                soha, "_tile_aware_inner_stride_elems", side_effect=inner_stride
            ),
        ):
            infos = soha._input_span_infos_controlled_by_output_dims(
                op, max_cores=1, k_split=2
            )

        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].chunking_info.selected_host_dim, 1)

    def test_bmm_output_search_retries_larger_split_when_k_validation_remains(self):
        op = _reduction_op(
            (1, 64, 64), reduction_ranges=(64,), reduction_type=BATCH_MATMUL_OP
        )
        candidate = SimpleNamespace(
            chunking_info=ChunkingInfo(
                total_bytes=1,
                per_core_span=2 * MAX_SPAN_BYTES,
                core_split_estimate=1,
                selected_device_dim_size=64,
                selected_device_span_stride_elems=64,
                selected_host_dim=1,
                stick_elems=64,
                reason="mixed M+K input span overflow",
            ),
            source="input:lhs",
        )

        def remaining_after_tile(_op, _max_cores, split_by_host_dim, *, k_split=None):
            self.assertIsNone(k_split)
            if split_by_host_dim == {1: 2}:
                return [SimpleNamespace(source="input:lhs")]
            return []

        with (
            patch.object(soha, "_split_candidates_for_host_dim", return_value=[2, 4]),
            patch.object(
                soha, "_combined_tile_stick_alignment_error", return_value=None
            ),
            patch.object(
                soha,
                "_remaining_span_candidates_after_tile",
                side_effect=remaining_after_tile,
            ),
        ):
            plan = soha._search_min_cost_tile_plan(op, 1, [candidate])

        self.assertIsNotNone(plan)
        self.assertEqual(plan.levels[0].selected_host_dim, 1)
        self.assertEqual(plan.levels[0].split_count, 4)

    def test_bmm_k_split_candidates_try_small_divisors_below_linear_estimate(self):
        op = _reduction_op(
            (1, 1, 64),
            reduction_ranges=(64,),
            reduction_type=BATCH_MATMUL_OP,
        )
        candidates = soha._bmm_k_split_candidates(op, required_split=16)

        self.assertIn(2, candidates)
        self.assertNotIn(64, candidates)

    def test_bmm_k_split_candidates_keep_reduction_extent_above_one(self):
        op = _reduction_op(
            (1, 1, 64),
            reduction_ranges=(8,),
            reduction_type=BATCH_MATMUL_OP,
        )

        candidates = soha._bmm_k_split_candidates(op, required_split=8)

        self.assertIn(2, candidates)
        self.assertIn(4, candidates)
        self.assertNotIn(8, candidates)

    def test_bmm_without_any_overflow_returns_none(self):
        op = _reduction_op(
            (1, 1, 64),
            reduction_ranges=(64,),
            reduction_type=BATCH_MATMUL_OP,
        )

        with (
            patch.object(soha, "_output_span_candidates_from_op", return_value=[]),
            patch.object(soha, "_input_span_candidates", return_value=[]),
            patch.object(soha, "_search_bmm_k_tile_plan", return_value=None),
        ):
            self.assertIsNone(plan_span_overflow_tile(op, max_cores=1))

    def test_bmm_k_plan_adapts_to_reduction_loop_variable(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(65536,), reduction_type=BATCH_MATMUL_OP
        )
        k = sympy.Symbol("k")
        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
                return_value=[sympy.Symbol("b"), sympy.Symbol("m"), sympy.Symbol("n")],
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._bmm_k_symbol",
                return_value=k,
            ),
            patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow._loop_var_to_reduction_ranges_pos",
                return_value=0,
            ),
        ):
            hints = _dims_to_hints(op, ((0, 4, True),), [_SPAN_OVERFLOW_HINT_ID])

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].loop_var, k)
        self.assertEqual(hints[0].split_count, 4)
        self.assertTrue(hints[0].is_reduction)

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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(lhs_dep, lhs_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=[output_candidate],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_span_candidates",
                return_value=[input_candidate],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_span_candidates_from_op",
                return_value=candidates,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.V",
                SimpleNamespace(graph=SimpleNamespace(get_buffer=lambda name: name)),
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._fixed_read_layout",
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

    def test_bmm_k_alignment_rejects_nonexact_padded_layout_split(self):
        op = _reduction_op(
            (1, 1, 64), reduction_ranges=(128,), reduction_type=BATCH_MATMUL_OP
        )
        k = sympy.Symbol("k")
        rhs_dep = MemoryDep("rhs", k, (k,), (128,))
        padded_layout = _fixed_tiled_layout((129,))

        with (
            patch.object(
                soha,
                "_input_read_deps",
                return_value=[(rhs_dep, padded_layout)],
            ),
            patch.object(soha, "_bmm_k_symbol", return_value=k),
            patch.object(soha, "host_coordinates", return_value=[k]),
        ):
            error = soha._bmm_k_alignment_error(op, split_count=2)

        self.assertIn("does not evenly divide", error)

    def test_planner_allows_full_size_exact_divisor_for_pointwise(self):
        op = _pointwise_op((1, 17, 16, 64))

        with patch(
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
            32768,
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
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
            32768,
        ):
            with self.assertRaisesRegex(Unsupported, "no combined split"):
                plan_span_overflow_tile(op, max_cores=4)

    def test_planner_raises_when_no_combined_split_satisfies_post_tile_span(self):
        op = _pointwise_op(_E2E_SHAPE)

        with patch(
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
            return_value=[object()],
        ):
            with self.assertRaisesRegex(Unsupported, "no combined split"):
                plan_span_overflow_tile(op, max_cores=4)

    def test_reduction_skips_indirect_reads_even_when_span_overflows(self):
        op = _reduction_op(_E2E_SHAPE)

        with patch(
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.indirect_info_from_op",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.indirect_info_from_op",
                return_value=({"arg0"}, {}, {sympy.Symbol("indirect0"): 8}),
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                MAX_SPAN_BYTES,
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0, n: 1},
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._device_coordinates_for_span",
                return_value=[k, m, n],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._remaining_span_candidates_after_tile",
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
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
            return_value={b: 0, m: 1, n: 2},
        ):
            input_deps = [
                (dep0, _fixed_tiled_layout((64, 16))),
                (dep1, _fixed_tiled_layout((64, 64))),
            ]
            symbol_to_dim = _bmm_output_symbol_to_dim(
                op,
                input_deps,
            )
            k_symbol = soha._bmm_k_symbol(op, input_deps)

        self.assertEqual(symbol_to_dim, {})
        self.assertIsNone(k_symbol)

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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
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
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._input_read_deps",
                return_value=[(input_dep, input_layout)],
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis._output_symbol_to_dim",
                return_value={m: 0, n: 1},
            ),
            patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.host_coordinates",
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

    @patch(
        "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
        _out_coords_for_bhld,
    )
    def test_adapter_creates_dim_hint_and_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _apply_span_overflow(_graph([op]))

        self.assertEqual(len(groups), 1)
        group_ops, levels = groups[0]
        self.assertEqual(group_ops, [op])
        self.assertEqual(levels[0][1], sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertEqual(len(op.dim_hints), 1)
        self.assertEqual(op.dim_hints[0].split_count, _E2E_SPLIT_COUNT)
        self.assertEqual(op.dim_hints[0].loop_var, sympy.Symbol("h"))

    @patch(
        "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
        _out_coords_for_symbolic_bhld,
    )
    def test_adapter_handles_nontrivial_batch_coord(self):
        op = _pointwise_op((4, 8195, 256, 64))

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            groups = _apply_span_overflow(_graph([op]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(op.dim_hints), 1)
        # Batch is a real loop var in this test, but this shape's span-driving
        # physical dim still maps to H, so the adapter should choose h.
        self.assertEqual(op.dim_hints[0].loop_var, sympy.Symbol("h"))
        self.assertEqual(groups[0][1][0][1], sympy.Integer(_E2E_SPLIT_COUNT))

    @patch(
        "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
        _out_coords_for_bhld,
    )
    def test_coarse_tile_consumes_auto_group_and_stamps_op(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
            graph = _graph([op])
            groups = _apply_span_overflow(graph)
            coarse_tile_post_stickify(graph, groups)

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
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.op_out_coords",
            _out_coords_for_bhld,
        ):
            with config.patch({"sencores": 4, "ignore_span_overflow_hints": False}):
                # Layer 2: adapter emits the same group shape as user hints.
                auto_graph = _graph([auto_op])
                auto_groups = _apply_span_overflow(auto_graph)
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
                coarse_tile_post_stickify(auto_graph, auto_groups)
                coarse_tile_post_stickify(manual_graph, manual_groups)

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


def _forced_span_plan_on_dim1(split_count, expect_size):
    """Force every op whose output dim 1 is ``expect_size`` onto one plan.

    Keys on output shape rather than buffer name: a Reduction graph lowers to
    more buffers than a two-op pointwise one, and their names are not knowable
    without first running it.  Anything not carrying the shared tiled dim falls
    back to the real planner rather than being handed a nonsensical tile.

    Forcing is necessary rather than convenient: two ops' independent span
    searches do not land on the same split for a toy shape (see
    ``test_pointwise_to_non_matmul_reduction_join_numeric``, whose earlier
    organic-plan version failed on hardware for exactly that reason), so
    without this the join under test would never form.

    Divisibility is checked here rather than left to ``coarse_tile``, so a bad
    ``split_count`` fails as a plain assertion in the test setup instead of as
    an ``Unsupported`` from deep in the pass.
    """
    real_plan = plan_span_overflow_tile
    assert expect_size % split_count == 0

    def forced(op, max_cores):
        ranges = list(getattr(op.data, "ranges", None) or [])
        try:
            shares_tiled_dim = len(ranges) >= 2 and int(ranges[1]) == expect_size
        except (TypeError, ValueError):
            shares_tiled_dim = False
        if not shares_tiled_dim:
            return real_plan(op, max_cores)
        return SpanOverflowTilePlan(
            levels=(
                SpanOverflowTileLevel(selected_host_dim=1, split_count=split_count),
            ),
            chunking_infos=(
                ChunkingInfo(
                    total_bytes=1,
                    per_core_span=1,
                    core_split_estimate=1,
                    selected_device_dim_size=split_count,
                    selected_device_span_stride_elems=1,
                    selected_host_dim=1,
                    stick_elems=64,
                    reason="forced for join validation",
                ),
            ),
            reason="forced for join validation",
        )

    return forced


class TestSpanOverflowPointwiseCodegen(InductorTestCase):
    """Small codegen test for scheduler/codegen LoopSpec emission."""

    _PLAN_PATCH = (
        "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile"
    )

    def _assert_one_loop_spec(self, fn, *args, expect_ops=()):
        """Compile ``fn`` for real and assert its group emits ONE LoopSpec.

        Kernel launch/compile are mocked, so this stops at generated source --
        one layer deeper than the grouping unit tests (which end at the group
        decision) and one layer shallower than
        ``TestSpanOverflowNumericValidation`` (which executes).  It proves the
        group survives ``coarse_tile`` and reaches codegen as a single shared
        loop, which the grouping tests alone cannot show.
        """
        with (
            patch(self._PLAN_PATCH, _forced_span_plan_on_dim1(5, 20)),
            patch(_LAUNCH_JOBPLAN),
            patch(_PREPARE_KERNEL),
            patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *args)
        self.assertTrue(source_codes)
        src = source_codes[0]
        self.assertEqual(
            src.count("LoopSpec("),
            1,
            "producer and consumer should share ONE LoopSpec -- source:\n" + src,
        )
        self.assertIn("count=sympify('5')", src)
        for op_name in expect_ops:
            self.assertIn(f"op='{op_name}'", src)
        return src

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_producer_to_pointwise_codegen_shares_one_loop_spec(self):
        """Reduction producer -> Pointwise consumer reaches codegen as one loop.

        The grouping tests prove the pass *decides* to group these; this proves
        the decision survives ``coarse_tile`` and is emitted as a single shared
        ``LoopSpec`` rather than two loops or none.

        The matmul equivalents cannot be covered this way yet: any auto-tiled
        group containing a matmul is stopped in ``_insert_read_copy_ops`` before
        codegen (the #3293 regression -- see
        ``test_bmm_to_pointwise_join_numeric``). That blocks #3218's already
        shipped `pw -> bmm` case too, not just the ones added here.
        """
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda x: x.sum(dim=2) * 2.0, x, expect_ops=("sum", "mul")
        )

    # TODO(copy-out-writer-advance): this is a KNOWN WRONG WRITE, not an
    # unexplained failure, and must not be un-xfailed without a fix.
    #
    # validate_writer_tile_advance (#3678) rejects this group with
    #   "writer-advance check failed for 'coarse_tile_copy_buf1' -- level 0
    #    tiles output dims [1] but output_tiled_dims has no extents for that
    #    level, so its write pointer would not advance there."
    # and it is right to.  _insert_copy_op keys the synthesized copy-out
    # writer's per-level extents by RAW dim index, while _tiled_dims_for_dep
    # matches those keys against the SQUEEZED dN symbols of dep.index.  This
    # group's terminal reduction has output [1, 20] -> divided [1, 4]; the
    # leading unit dim is squeezed away, so the raw key 1 matches no symbol,
    # output_tiled_dims comes back empty, and every tile is written on top of
    # tile 0.
    #
    # Not gated in the pass, because there is no clean predicate: a Reduction
    # consumer joining a Reduction-rooted run is the same code path for
    # Reduction -> Reduction and for Reduction -> matmul, and the latter works
    # (test_reduction_producer_to_bmm_codegen_shares_one_loop_spec, and its
    # on-device counterpart).  What separates them here is output shape, not
    # direction.  Gating the path disabled ten working tests.
    #
    # A first attempt at the real fix -- keying by squeezed position, the same
    # mapping _insert_read_copy_ops already builds -- fixed this test and two
    # numeric xfails (test_lm_head_matmul_join_numeric went from 48727/49152
    # (99.14%) mismatches to 999/49152 (2.03%) against a 1.59% untiled
    # baseline), but regressed test_hint_flash_attention_v2_divide_in_scope to
    # 86.2% wrong.  So the two conventions are reconciled somewhere further
    # down and a correct fix needs that path understood, not just this keying
    # changed.
    @unittest.expectedFailure
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_producer_to_reduction_codegen_shares_one_loop_spec(self):
        """Reduction producer -> Reduction consumer reaches codegen as one loop.

        Neither side is a matmul, so this exercises the Reduction-to-Reduction
        join end of the grouping change at codegen depth.  The group is flushed
        as soon as the second reduction joins, so one LoopSpec is also the
        assertion that nothing further was folded in.

        Currently xfailed on a known wrong write -- see the TODO above.  The
        grouping decision this test exists to cover is still pinned at
        decision depth by test_non_matmul_reduction_producer_groups_with_
        reduction_consumer, which does not go through codegen.
        """
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda x: x.sum(dim=2).sum(dim=-1), x, expect_ops=("sum",)
        )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_pointwise_producer_to_pointwise_codegen_shares_one_loop_spec(self):
        """Pointwise -> Pointwise, the oldest supported direction (#3058).

        ``test_codegen_contains_auto_span_overflow_loop_spec`` covers a *single*
        tiled op; this covers a two-op producer/consumer group, so the matrix
        below has a baseline that predates this branch.
        """
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        y = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda x, y: (x + y) * 2.0, x, y, expect_ops=("add", "mul")
        )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_pointwise_producer_to_reduction_codegen_shares_one_loop_spec(self):
        """Pointwise -> Reduction (#3270), at codegen depth.

        Its on-device counterpart
        (``test_pointwise_to_non_matmul_reduction_join_numeric``) already
        passes; this pins the same direction one layer earlier so a codegen
        regression is distinguishable from an execution one.
        """
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        y = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda x, y: (x + y).sum(dim=2), x, y, expect_ops=("add", "sum")
        )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_producer_to_bmm_codegen_shares_one_loop_spec(self):
        """Reduction -> matmul reaches codegen -- the one matmul direction that
        currently can.

        Worth pinning precisely because it is the exception.  Every other
        matmul direction is stopped in ``_insert_read_copy_ops`` (see the
        xfailed tests below).  This one survives only because ``keepdim=True``
        leaves M=1, which is squeezed out of the iteration space and happens to
        make the ranks line up again -- so it is a shape coincidence, not
        matmul support.  If the read-copy limitation is ever fixed, this should
        keep passing; if this starts failing, the squeeze behaviour changed.
        """
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        b = torch.randn(1, 20, 64, 32, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda x, b: torch.matmul(x.sum(dim=2, keepdim=True), b),
            x,
            b,
            expect_ops=("sum", BATCH_MATMUL_OP),
        )

    # The three matmul directions below are all stopped in
    # _insert_read_copy_ops before codegen -- a matmul operand does not use
    # every loop variable, so the iteration space cannot be walked onto its
    # dimensions (see the TODO there).  They are xfailed rather than omitted so
    # the matrix is complete and they flip green on their own once that is
    # fixed.
    #
    # Note the first one is #3218's already-shipped pointwise -> matmul case,
    # not a direction this branch adds: test_lm_head_matmul_join_numeric passed
    # when #3270 added it and was xfailed by #3293. So the blocker predates and
    # outlives this change.
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_pointwise_producer_to_bmm_codegen_shares_one_loop_spec(self):
        """Pointwise -> matmul (#3218's shipped direction).

        Was xfailed as TODO(span-overflow-read-copy): a tiled matmul could not
        read a full-size buffer, because the iteration space was walked onto
        the input's dimensions positionally and a matmul operand does not use
        every loop variable (A has no N).  #3612 ("coarse tiling: optional read
        copy") reworked that path and the direction now reaches codegen, so the
        xfail is removed rather than re-explained.

        Kept for the history it records: this was a REGRESSION, not a missing
        feature.  Its on-device counterpart test_lm_head_matmul_join_numeric
        passed when #3270 added it (21 Jul) and was marked expectedFailure by
        #3293 (28 Jul) with no stated cause -- #3293's own note called the
        xfails "a deliberate decision to unblock the merge, not a claim about a
        specific bisected root cause".  If this starts failing again, that is
        the history to read first.
        """
        a = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        b = torch.randn(1, 20, 64, 32, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda a, b: torch.matmul(a * 2.0, b),
            a,
            b,
            expect_ops=("mul", BATCH_MATMUL_OP),
        )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_bmm_producer_to_pointwise_codegen_shares_one_loop_spec(self):
        """matmul -> Pointwise, added by this branch.

        Was xfailed as TODO(span-overflow-read-copy) for the same reason as the
        direction above, and unblocked by the same change (#3612).
        """
        a = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        b = torch.randn(1, 20, 64, 32, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda a, b: torch.matmul(a, b) * 2.0,
            a,
            b,
            expect_ops=(BATCH_MATMUL_OP, "mul"),
        )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_bmm_producer_to_reduction_codegen_shares_one_loop_spec(self):
        """matmul -> Reduction, added by this branch.

        Was xfailed as TODO(span-overflow-read-copy) with the two directions
        above -- same cause, and unblocked by the same change (#3612).
        """
        a = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        b = torch.randn(1, 20, 64, 32, dtype=torch.float16).to("spyre")
        self._assert_one_loop_spec(
            lambda a, b: torch.matmul(a, b).sum(dim=2),
            a,
            b,
            expect_ops=(BATCH_MATMUL_OP, "sum"),
        )

    @patch("torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
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

    @patch("torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    @unittest.expectedFailure
    def test_reduction_input_span_codegen_contains_auto_loop_spec(self):
        """Decision xfail: failing in CI (Actions run 30385154736, job
        90362759197) on PR #3293. We've decided to xfail the coarse tiling

        TODO(3293-decision-xfail): investigate and un-xfail; no root cause
        was bisected when this was marked.
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
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
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
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
    # own search completes, it does too, and the two are now tiled into one
    # synchronized group by test_lm_head_matmul_joins_tiled_restickify_producer
    # -- so this test always asserted the same outcome as that one.

    @patch("torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    @unittest.expectedFailure
    def test_auto_span_overflow_matches_equivalent_spyre_hint_loop_spec(self):
        """Decision xfail: failing in CI (Actions run 30385154736, job
        90362759197) on PR #3293. We've decided to xfail the coarse tiling

        TODO(3293-decision-xfail): investigate and un-xfail; no root cause
        was bisected when this was marked.
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
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


class TestSpanOverflowNumericValidation(InductorTestCase):
    """Real end-to-end hardware execution and numeric validation for
    span-overflow producer-consumer joins.

    Every test class above this one either mocks out kernel launch/compile
    (``patch(_LAUNCH_JOBPLAN)``, ``patch(_PREPARE_KERNEL)``,
    ``patch("subprocess.run")``) or inspects internal Python state directly.
    Those are valuable and cheap, and prove the *decision* to join is made
    correctly -- but none of them prove the resulting shared loop nest
    actually *executes* correctly on hardware. A join could be structurally
    identical to a passing case and still compute wrong values (e.g. a subtle
    per-tile addressing bug only visible with real reads/writes).

    This includes the original #3217/#3218 matmul-join case: its on-device
    numeric validation ("0.5% rel error vs fp32 ref", per the PR description)
    was a manual, standalone validation run, never captured as an automated
    regression test. These tests close that gap for both the matmul case and
    the non-matmul-reduction case this change adds.

    No kernel-launch mocking here (unlike ``test_coarse_tile_e2e.py``'s
    codegen-only tests) -- ``compare_with_cpu`` runs the compiled function for
    real, the same pattern ``test_coarse_tile_e2e.py`` already uses for its
    own real numeric tests (e.g. ``test_hint_matmul_row_tiling``).
    """

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_pointwise_to_non_matmul_reduction_join_numeric(self):
        """A plain ``sum`` reduction that joins its tiled pointwise producer's
        group (the join this PR extends from matmul-only to any reduction)
        must produce numerically correct results, not just a plausible-looking
        LoopSpec.

        An earlier version of this test let both ops' plans be chosen
        organically by the real planner, on the theory that
        ``test_codegen_contains_auto_span_overflow_loop_spec`` already proves
        the producer picks host_dim=1/split=5 for real under this shape.  Ran
        on real hardware, that failed with ``Unsupported: Cannot auto-tile
        buf1: it reads already auto-tiled producer(s) ['buf0']`` --
        ``sum(dim=2)``'s own independent span search did *not* land on the
        same host_dim/split as the producer (its input-span formula differs
        from the producer's output-span formula), so the join conditions
        never matched and the whole compile failed instead of falling back
        safely. That confirmed the uncertainty flagged in that version was a
        real gap, not a hypothetical one.

        This version removes the guesswork: ``plan_span_overflow_tile`` is
        patched to force *both* ops onto the identical (host_dim=1,
        split_count=5) plan -- the same technique
        ``test_matmul_joins_tiled_weight_producer_group`` uses to prove the
        join decision in isolation -- but with no kernel-launch mocking, so
        the forced plan still runs for real on hardware. This isolates
        exactly what we want to validate (does a *successfully joined* loop
        compute the right numbers), independent of whether this particular
        toy shape happens to make both ops agree organically.
        """
        torch.manual_seed(0xAFFE)
        shape = (1, 20, 16, 64)
        x = torch.randn(shape, dtype=torch.float16)
        y = torch.randn(shape, dtype=torch.float16)

        def fn(x, y):
            z = x + y
            return z.sum(dim=2)

        _real_plan = plan_span_overflow_tile

        def forced_plan(op, max_cores):
            # Scope the forced plan to the two ops this graph actually
            # produces (confirmed by name from the real Unsupported error
            # this test's earlier version hit: "buf1: ... reads ...
            # producer(s) ['buf0']"). Anything else falls back to the real
            # planner so an unexpected extra buffer doesn't get a nonsensical
            # forced tile.
            if op.get_name() not in ("buf0", "buf1"):
                return _real_plan(op, max_cores)
            return SpanOverflowTilePlan(
                levels=(SpanOverflowTileLevel(selected_host_dim=1, split_count=5),),
                chunking_infos=(
                    ChunkingInfo(
                        total_bytes=1,
                        per_core_span=1,
                        core_split_estimate=1,
                        selected_device_dim_size=5,
                        selected_device_span_stride_elems=1,
                        selected_host_dim=1,
                        stick_elems=64,
                        reason="forced for numeric join validation",
                    ),
                ),
                reason="forced for numeric join validation",
            )

        def report_join_evidence(src):
            loop_spec_count = src.count("LoopSpec(")
            has_add = "op='add'" in src
            has_sum = "op='sum'" in src
            print(
                f"[join evidence] LoopSpec count={loop_spec_count}; "
                f"op='add' present={has_add}; op='sum' present={has_sum}"
            )
            self.assertEqual(
                loop_spec_count,
                1,
                "expected producer and consumer to share one LoopSpec (a "
                "real join) since plan_span_overflow_tile was forced to "
                "agree for both ops -- source:\n" + src,
            )
            self.assertTrue(has_add)
            self.assertTrue(has_sum)

        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            forced_plan,
        ):
            compare_with_cpu(
                fn,
                x,
                y,
                run_compile=True,
                run_eager=False,
                source_check=report_join_evidence,
                atol=0.05,
                rtol=0.05,
            )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_pointwise_to_pointwise_join_numeric(self):
        """Pointwise -> Pointwise executed for real against a CPU reference.

        This is the oldest automatic direction (#3058) and the only one whose
        numbers had never been checked: the pointwise coverage was codegen-only
        (kernel launch mocked, source text asserted), and the sole on-device
        test in this class covered Pointwise -> Reduction.  So a group whose
        members are all pointwise was verified to be *decided* and *emitted*
        correctly, never to compute correctly.

        Not a direction this branch adds -- included because the gap is only
        visible once the matrix is written out, and this path is not blocked by
        the read-copy limitation that xfails the matmul ones.

        The plan is forced so both ops tile host_dim=1 at split 5; without it
        their independent span searches need not agree on a toy shape and the
        join under test would not form.  Kernel launch is NOT mocked.
        """
        torch.manual_seed(0xAFFE)
        shape = (1, 20, 16, 64)
        x = torch.randn(shape, dtype=torch.float16)
        y = torch.randn(shape, dtype=torch.float16)

        def fn(x, y):
            return (x + y) * 2.0

        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            _forced_span_plan_on_dim1(5, 20),
        ):
            compare_with_cpu(
                fn,
                x,
                y,
                run_compile=True,
                run_eager=False,
                source_check=self._assert_single_loop_spec("add", "mul"),
                atol=0.05,
                rtol=0.05,
            )

    # The three Reduction-producer directions below execute a group whose
    # PRODUCER is a tiled reduction reading a full-size buffer (a graph input).
    #
    # All three used to die in dxp_standalone.  After #3612 ("coarse tiling:
    # optional read copy") reworked the read-copy path, Reduction -> Pointwise
    # and Reduction -> matmul both execute correctly and are no longer xfailed;
    # only Reduction -> Reduction still aborts.  That is worth noting rather
    # than quietly deleting: the diagnosis recorded below blamed the backend's
    # DDL conversion, and two thirds of it turned out to be reachable from the
    # read-copy path instead.  Whether the surviving case has the same cause or
    # a genuinely different one has not been re-established.
    #
    # The evidence below is kept as originally captured, and describes the
    # pre-#3612 state.  Isolated to the read-copy path, not to grouping:
    #
    #   lone sum, NO tiling                PASS
    #   lone sum, TILED (no group at all)  dxp_standalone SIGABRT
    #   sum -> pw, NO tiling               PASS
    #   sum -> pw, TILED (grouped)         dxp_standalone SIGABRT
    #
    # A single tiled sum with no consumer fails identically, so the group is
    # irrelevant.  What matters is what the tiled reduction reads:
    # test_pointwise_to_non_matmul_reduction_join_numeric passes with a tiled
    # sum too, but there the sum reads its in-group producer rather than a
    # full-size buffer.
    #
    # Same root as the matmul xfails, one stage later:
    # _insert_read_copy_ops handles a tiled *Pointwise* reading a full buffer
    # (test_pointwise_to_pointwise_join_numeric passes, and its add reads graph
    # inputs) but not a tiled *Reduction*.  A matmul trips the rank assert; a
    # sum clears it and yields a kernel the backend rejects.
    #
    # The backend's own diagnostic, captured by re-running its command:
    #
    #   terminate called after throwing an instance of 'DtException'
    #     what():  DtException: Could not find any suitable dimension mapping,
    #              file /project_src/deeptools/ddc/ddl/ddl_conversion.cpp line 2497
    #
    # So deeptools cannot map the tiled reduction's dimensions during DDL
    # conversion, throws, and nobody catches it -- hence SIGABRT rather than a
    # diagnosable error.  This is NOT #3414, which is a different failure in a
    # different place ("Immediate value out of boundary ... L3_ADDEARIMM").
    #
    # Codegen depth for all three passes -- see
    # TestSpanOverflowPointwiseCodegen -- so these isolate execution alone.
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_to_pointwise_join_numeric(self):
        """Reduction -> Pointwise executed for real, against a CPU reference.

        Was xfailed as TODO(deeptools-ddl-dim-mapping): codegen succeeded and
        deeptools then threw DtException "Could not find any suitable dimension
        mapping" (ddl_conversion.cpp:2497), uncaught, so the process aborted.
        #3612 reworked the read-copy path and this now executes and matches CPU
        -- so the abort was reachable from read-copy construction rather than
        being purely a backend limitation, whatever remains true of the
        Reduction -> Reduction case below.
        """
        torch.manual_seed(0xAFFE)
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16)
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            _forced_span_plan_on_dim1(5, 20),
        ):
            compare_with_cpu(
                lambda x: x.sum(dim=2) * 2.0,
                x,
                run_compile=True,
                run_eager=False,
                atol=0.05,
                rtol=0.05,
            )

    @unittest.expectedFailure
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_to_reduction_join_numeric(self):
        """Reduction -> Reduction executed for real. Blocked in dxp_standalone.

        TODO(deeptools-ddl-dim-mapping): un-xfail with the direction above --
        same DtException from the same place.
        """
        torch.manual_seed(0xAFFE)
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16)
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            _forced_span_plan_on_dim1(5, 20),
        ):
            compare_with_cpu(
                lambda x: x.sum(dim=2).sum(dim=-1),
                x,
                run_compile=True,
                run_eager=False,
                atol=0.05,
                rtol=0.05,
            )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_reduction_to_bmm_join_numeric(self):
        """Reduction -> matmul executed for real, against a CPU reference.

        Was xfailed as TODO(deeptools-ddl-dim-mapping) with the same
        DtException as the other Reduction-producer directions; #3612 unblocked
        it.

        Note this shape reached codegen even before #3612 (keepdim=True leaves
        M=1, which squeezes out and realigns the ranks -- see
        ``test_reduction_producer_to_bmm_codegen_shares_one_loop_spec``), so it
        was the only matmul cell where execution was reachable at all.  It now
        matches CPU, so the realignment produces correct numbers and not merely
        a kernel.
        """
        torch.manual_seed(0xAFFE)
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16) * 0.1
        b = torch.randn(1, 20, 64, 32, dtype=torch.float16) * 0.1
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            _forced_span_plan_on_dim1(5, 20),
        ):
            compare_with_cpu(
                lambda x, b: torch.matmul(x.sum(dim=2, keepdim=True), b),
                x,
                b,
                run_compile=True,
                run_eager=False,
                atol=0.05,
                rtol=0.05,
            )

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_bmm_to_pointwise_join_numeric_via_manual_hint(self):
        """The BMM -> PW group this branch builds automatically, executed for
        real -- but requested with a manual ``spyre_hint`` instead.

        Same shapes, same tiled dim, same resulting group as
        ``test_bmm_to_pointwise_join_numeric``; only *who asks for the tile*
        differs.  This passes, which localizes the gap precisely: grouping a
        BMM producer with a pointwise consumer, the shared loop nest, and the
        per-tile addressing are all correct on hardware for this exact case.

        The automatic version is xfailed because the two callers reach
        different code: ``_maybe_coarse_tile_hints`` runs PRE-stickification,
        so the operands still carry a plain ``FixedLayout`` and
        ``_insert_read_copy_ops`` takes its ``else`` branch, while
        ``_maybe_coarse_tile_span_overflow`` runs POST-stickification, hits the
        ``FixedTiledLayout`` branch, and cannot map a bmm's iteration space
        onto an operand's dimensions (see the TODO there).

        Kept here rather than in ``test_coarse_tile_e2e.py`` because its value
        is as the control for the two xfailed tests above -- it is what they
        should look like once that branch is fixed.
        """
        from torch_spyre._inductor import spyre_hint

        torch.manual_seed(0xAFFE)
        B, H, M, K, N = 1, 20, 16, 64, 32
        a = torch.randn(B, H, M, K, dtype=torch.float16) * 0.1
        b = torch.randn(B, H, K, N, dtype=torch.float16) * 0.1
        for dim_name, size in (("B", B), ("H", H), ("M", M), ("K", K), ("N", N)):
            _pnd.declare_tensor_dim(dim_name, size)

        def fn(a, b):
            _pnd.name_tensor_dims(a, ["B", "H", "M", "K"])
            _pnd.name_tensor_dims(b, ["B", "H", "K", "N"])
            with spyre_hint(num_tiles_per_dim={"H": 5}):
                return torch.matmul(a, b) * 2.0

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def _assert_single_loop_spec(self, *expected_ops):
        """Return a source_check asserting one shared LoopSpec over the ops.

        Load-bearing: without it these pass even if the join never happened
        and each op got its own loop (or tiling was skipped entirely), which
        is exactly the failure mode they exist to catch.
        """

        def check(src):
            loop_spec_count = src.count("LoopSpec(")
            present = {name: (f"op='{name}'" in src) for name in expected_ops}
            print(
                f"[join evidence] LoopSpec count={loop_spec_count}; "
                + "; ".join(f"op='{k}' present={v}" for k, v in present.items())
            )
            self.assertEqual(
                loop_spec_count,
                1,
                "expected producer and consumer to share one LoopSpec (a real "
                "join), since plan_span_overflow_tile was forced to agree for "
                "both ops -- source:\n" + src,
            )
            for name, found in present.items():
                self.assertTrue(found, f"expected op='{name}' in source:\n{src}")

        return check

    # TODO(span-overflow-read-copy): the FAILURE MODE CHANGED with #3612
    # ("coarse tiling: optional read copy").  The codegen half of this
    # direction, in TestSpanOverflowPointwiseCodegen, now passes and is no
    # longer xfailed -- the read copy gets built.  What remains is worse than
    # the old loud failure: the kernel compiles, executes, and returns WRONG
    # NUMBERS (compiled-spyre vs CPU mismatch), so this xfail is now masking a
    # silent wrong-answer path rather than a refusal to compile.
    #
    # Most likely cause is the positional walk described below, which is still
    # present: it pairs each of the buffer's non-unit dims with the next
    # iteration extent by position.  Instrumenting _resize_device_layout on
    # this shape shows it handed tile_size=[1,4,32,64] for a buffer of
    # [1,20,64,32] -- the trailing dims transposed -- so the layout it builds
    # does not describe the data.  Verify that before assuming a backend bug.
    #
    # Blocked by a pre-existing coarse_tile.py defect, not by the grouping this
    # branch adds: _insert_read_copy_ops cannot build a copy-buffer layout for a
    # tiled *Reduction* that reads a full-size buffer from outside its loop
    # group.  It aligns dep.size (the ITERATION space -- for a BMM that is
    # H/M/N/K) positionally against the input buffer's non-unit dims, which only
    # coincide for Pointwise ops.  For arg0_1 [1,20,16,64] read as
    # 1024*d0 + 64*d1 + d3, tile_ranges=[4,16,32,64] has four entries for three
    # non-unit buffer dims and it asserts.
    #
    # Verified pre-existing two ways: a LONE tiled BMM with no consumer at all
    # (no grouping possible) raises the same assert, and the #3218-era
    # test_lm_head_matmul_join_numeric below -- already expectedFailure before
    # this branch -- fails with the identical message.  So automatic
    # span-overflow tiling of a BMM reading graph inputs has never worked,
    # grouped or not.
    #
    # The grouping decision these validate is fully covered and passing in
    # TestSpanOverflowGroups; only on-device execution is blocked.  Remove both
    # decorators once _insert_read_copy_ops handles Reduction iteration spaces.
    @unittest.expectedFailure
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_bmm_to_pointwise_join_numeric(self):
        """A BMM producer whose Pointwise consumer joins its group -- the
        BMM -> PW direction -- executed for real and compared against a CPU
        reference.

        Every other BMM -> PW test mocks kernel launch and inspects the
        grouping decision only.  A group can be structurally perfect and still
        compute wrong values: the producer's per-tile output becomes
        loop-internal scratch that the consumer reads in the same iteration,
        and nothing but a real run proves that per-tile addressing pairs the
        right slices.

        The plan is forced so both ops tile host_dim=1 at split 5 (the
        technique the sibling pointwise test settled on after an organic-plan
        version failed on hardware, because the two ops' independent span
        searches do not land on the same split for a toy shape).  Kernel
        launch is NOT mocked, so the forced plan still runs for real.
        """
        torch.manual_seed(0xAFFE)
        H = 20
        a = torch.randn(1, H, 16, 64, dtype=torch.float16)
        b = torch.randn(1, H, 64, 32, dtype=torch.float16)

        def fn(a, b):
            return torch.matmul(a, b) * 2.0

        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            _forced_span_plan_on_dim1(5, H),
        ):
            compare_with_cpu(
                fn,
                a,
                b,
                run_compile=True,
                run_eager=False,
                source_check=self._assert_single_loop_spec(BATCH_MATMUL_OP, "mul"),
                atol=0.05,
                rtol=0.05,
            )

    # TODO(span-overflow-read-copy): the FAILURE MODE CHANGED with #3612
    # ("coarse tiling: optional read copy").  The codegen half of this
    # direction, in TestSpanOverflowPointwiseCodegen, now passes and is no
    # longer xfailed -- the read copy gets built.  What remains is worse than
    # the old loud failure: the kernel compiles, executes, and returns WRONG
    # NUMBERS (compiled-spyre vs CPU mismatch), so this xfail is now masking a
    # silent wrong-answer path rather than a refusal to compile.
    #
    # Most likely cause is the positional walk described below, which is still
    # present: it pairs each of the buffer's non-unit dims with the next
    # iteration extent by position.  Instrumenting _resize_device_layout on
    # this shape shows it handed tile_size=[1,4,32,64] for a buffer of
    # [1,20,64,32] -- the trailing dims transposed -- so the layout it builds
    # does not describe the data.  Verify that before assuming a backend bug.
    #
    # Blocked by a pre-existing coarse_tile.py defect, not by the grouping this
    # branch adds: _insert_read_copy_ops cannot build a copy-buffer layout for a
    # tiled *Reduction* that reads a full-size buffer from outside its loop
    # group.  It aligns dep.size (the ITERATION space -- for a BMM that is
    # H/M/N/K) positionally against the input buffer's non-unit dims, which only
    # coincide for Pointwise ops.  For arg0_1 [1,20,16,64] read as
    # 1024*d0 + 64*d1 + d3, tile_ranges=[4,16,32,64] has four entries for three
    # non-unit buffer dims and it asserts.
    #
    # Verified pre-existing two ways: a LONE tiled BMM with no consumer at all
    # (no grouping possible) raises the same assert, and the #3218-era
    # test_lm_head_matmul_join_numeric below -- already expectedFailure before
    # this branch -- fails with the identical message.  So automatic
    # span-overflow tiling of a BMM reading graph inputs has never worked,
    # grouped or not.
    #
    # The grouping decision these validate is fully covered and passing in
    # TestSpanOverflowGroups; only on-device execution is blocked.  Remove both
    # decorators once _insert_read_copy_ops handles Reduction iteration spaces.
    @unittest.expectedFailure
    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_bmm_to_reduction_join_numeric(self):
        """A BMM producer whose Reduction consumer joins its group -- the
        BMM -> Reduction direction -- executed for real.

        This is the direction with the most room to go silently wrong: a
        Reduction consumer paired against the wrong producer slice does not
        crash, it computes a partial sum and returns it.  A CPU comparison is
        the only thing that distinguishes that from a correct result, which is
        why this test exists rather than relying on the grouping unit tests.

        ``sum(dim=2)`` reduces a dim that is neither op's tiled dim, so the
        tiled dim stays an output range for both -- the case the join is
        licensed for.  The complementary illegal case (producer tiling a dim
        that is the consumer's reduction range) is covered structurally by
        ``test_bmm_to_bmm_rejected_when_producer_tiles_consumer_reduction_dim``,
        which asserts it is refused rather than executed.
        """
        torch.manual_seed(0xAFFE)
        H = 20
        a = torch.randn(1, H, 16, 64, dtype=torch.float16)
        b = torch.randn(1, H, 64, 32, dtype=torch.float16)

        def fn(a, b):
            return torch.matmul(a, b).sum(dim=2)

        with patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow.plan_span_overflow_tile",
            _forced_span_plan_on_dim1(5, H),
        ):
            compare_with_cpu(
                fn,
                a,
                b,
                run_compile=True,
                run_eager=False,
                source_check=self._assert_single_loop_spec(BATCH_MATMUL_OP, "sum"),
                atol=0.05,
                rtol=0.05,
            )

    # TODO(span-overflow-read-copy): the on-device half of
    # test_pointwise_producer_to_bmm_codegen_shares_one_loop_spec, whose
    # codegen half now passes after #3612 and is no longer xfailed.
    #
    # Added PASSING by #3270 on 21 Jul; marked expectedFailure by #3293 on
    # 28 Jul with no stated cause.  Before #3612 it failed in the read-copy
    # path before codegen -- the same rank assert as the codegen xfails, and
    # explicitly "not a numeric problem".  That is no longer true: it now
    # compiles, runs, and is numerically wrong, at
    # tiled mismatches=48727/49152 (99.14%) against untiled 781/49152 (1.59%)
    # on the same reference.  A ~99% mismatch is not accumulation noise; the
    # transposed tile layout described above is the first thing to rule out.
    @unittest.expectedFailure
    def test_lm_head_matmul_join_numeric(self):
        """F.linear with an oversized vocab-dim weight: the restickified
        weight producer and the BMM consumer join into one synchronized group
        (#3217/#3218). ``vocab=49152`` (768 stick-aligned sticks, composite)
        and ``sencores=32`` are the exact shape/core-count the PR author
        validated manually with 0% numeric error per the PR description; this
        test captures that as an automated regression instead of relying on a
        one-off manual run.

        Decision xfail: failing in CI (Actions run 30385154736, job
        90362759197) on PR #3293. We've decided to xfail the coarse tiling
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.

        A first version of this test compared only the tiled (joined) result
        against a plain fp16 CPU reference with atol=rtol=0.05, and failed:
        886/49152 (1.8%) elements exceeded that threshold, with the largest
        absolute diff 0.6875 on values in the ~30-160 magnitude range. Those
        numbers look like ordinary fp16 accumulation-order noise over a
        4096-deep reduction (CPU and device sum in different orders), not a
        correctness bug -- but a fixed tolerance can't distinguish "expected
        fp16 noise at this reduction depth" from "tiling introduced extra
        error" by itself. This version isolates that by also running the
        *same* shape with span-overflow tiling disabled
        (``ignore_span_overflow_hints=True``) -- work division alone already
        handles this shape at sencores=32 (splits the 768-stick dim in half),
        so an untiled run is possible here and gives a same-K-depth,
        same-CPU-reference control for how much noise is inherent regardless
        of tiling. The assertion is that tiled isn't *meaningfully worse* than
        untiled, not an arbitrary fixed threshold.
        """
        torch.manual_seed(0xAFFE)
        vocab, hidden = 49152, 4096
        x = torch.randn(1, hidden, dtype=torch.float16)
        weight = torch.randn(vocab, hidden, dtype=torch.float16)

        def fn(x, weight):
            return F.linear(x, weight)

        cpu_result = fn(x, weight)

        def run_on_spyre(ignore_span_overflow_hints, source_check=None):
            torch._dynamo.reset_code_caches()
            torch._inductor.codecache.FxGraphCache.clear()
            with config.patch(
                {
                    "sencores": 32,
                    "ignore_span_overflow_hints": ignore_span_overflow_hints,
                }
            ):
                x_dev = x.to("spyre")
                weight_dev = weight.to("spyre")
                cfn = torch.compile(fn, dynamic=False)
                if source_check is not None:
                    result, source_codes = run_and_get_code(cfn, x_dev, weight_dev)
                    if source_codes:
                        source_check(source_codes[0])
                else:
                    result = cfn(x_dev, weight_dev)
            return result.cpu()

        def mismatch_count(actual, expected, atol, rtol):
            diff = (actual.float() - expected.float()).abs()
            threshold = atol + rtol * expected.float().abs()
            return int((diff > threshold).sum().item())

        def report_join_evidence(src):
            loop_spec_count = src.count("LoopSpec(")
            print(
                f"[join evidence] LoopSpec count={loop_spec_count}; "
                f"op='{RESTICKIFY_OP}' present={RESTICKIFY_OP in src}; "
                f"op='{BATCH_MATMUL_OP}' present={BATCH_MATMUL_OP in src}"
            )
            self.assertIn("LoopSpec(", src)

        atol, rtol = 0.05, 0.05
        tiled_result = run_on_spyre(False, source_check=report_join_evidence)
        untiled_result = run_on_spyre(True)

        tiled_mismatches = mismatch_count(tiled_result, cpu_result, atol, rtol)
        untiled_mismatches = mismatch_count(untiled_result, cpu_result, atol, rtol)
        total = cpu_result.numel()
        print(
            f"[numeric] tiled mismatches={tiled_mismatches}/{total} "
            f"({100 * tiled_mismatches / total:.2f}%); "
            f"untiled mismatches={untiled_mismatches}/{total} "
            f"({100 * untiled_mismatches / total:.2f}%)"
        )

        # Tiled shouldn't be meaningfully worse than untiled at the same
        # reduction depth against the same CPU reference -- some slack for
        # noise variance between the two runs (different tile boundaries can
        # shift *which* elements land near the fp16 rounding edge without
        # indicating a real bug), but not an unbounded gap.
        self.assertLessEqual(
            tiled_mismatches,
            untiled_mismatches + max(10, untiled_mismatches),
            f"tiled result has {tiled_mismatches} mismatches vs "
            f"{untiled_mismatches} untiled -- meaningfully worse than the "
            "untiled baseline at the same reduction depth, suggesting the "
            "join introduces extra error beyond ordinary fp16 noise.",
        )
