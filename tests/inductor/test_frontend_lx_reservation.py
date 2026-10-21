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

"""Recovered frontend reservation liveness checks."""

from types import SimpleNamespace
from unittest import TestCase

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scratchpad.allocator import _lx_planning_size
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    TileAxis,
    TileSpec,
    ceil_div,
)
from torch_spyre._inductor.scratchpad.utils import (
    frontend_lx_high_water,
    invalidate_frontend_lx_high_water,
    publish_frontend_lx_footprints,
)

_D0 = sympy.Symbol("d0", integer=True, nonnegative=True)


def _dep(name: str) -> MemoryDep:
    return MemoryDep(name, _D0, (_D0,), (64,))


def _op(name: str, reads=(), writes=(), loop_group_id: tuple = ()):
    rw = SimpleNamespace(
        reads={_dep(r) for r in reads}, writes={_dep(w) for w in writes}
    )
    op = SimpleNamespace(get_operation_name=lambda: name, get_read_writes=lambda: rw)
    if loop_group_id:
        op.loop_info = SimpleNamespace(loop_group_id=loop_group_id)
    return op


def _lx_layout(address: int) -> FixedTiledLayout:
    layout = FixedTiledLayout(
        torch.device("cpu"),
        torch.float16,
        [64],
        [1],
        SimpleNamespace(device_size=[64]),
        sympy.Integer(0),
    )
    layout.allocation["lx"] = address
    return layout


def _graph(ops, layouts: dict[str, FixedTiledLayout]):
    holders = {
        name: SimpleNamespace(layout=layout) for (name, layout) in layouts.items()
    }
    return SimpleNamespace(
        operations=ops,
        graph_input_names=[],
        try_get_buffer=holders.get,
        get_buffer=holders.__getitem__,
    )


def _publish(graph, **footprints):
    publish_frontend_lx_footprints(graph, footprints)


class FrontendLxHighWaterTest(TestCase):
    def test_max_end_address_over_live_buffers_with_a_hole(self):
        """A gap between two buffers is not reserved; the top of the live set is."""
        ops = [
            _op("op0", writes=["a"]),
            _op("op1", reads=["a"]),
            _op("op2", reads=["a"], writes=["b"]),
            _op("op3", reads=["b"]),
            _op("op4", reads=["b"]),
        ]
        graph = _graph(ops, {"a": _lx_layout(0), "b": _lx_layout(512)})
        _publish(graph, a=128, b=256)
        self.assertEqual(
            frontend_lx_high_water(graph),
            {"op0": 128, "op1": 128, "op2": 768, "op3": 768, "op4": 768},
        )

    def test_inserted_copy_gets_a_value_and_extends_the_lifetime(self):
        """An inserted clone is an op like any other: it writes the LX buffer."""
        ops = [
            _op("op0", reads=["arg"], writes=["clone"]),
            _op("op1"),
            _op("op2", reads=["clone"]),
        ]
        graph = _graph(ops, {"clone": _lx_layout(256)})
        _publish(graph, clone=128)
        self.assertEqual(
            frontend_lx_high_water(graph), {"op0": 384, "op1": 384, "op2": 384}
        )

    def test_loop_backedge_keeps_a_carried_value_live_to_the_loop_end(self):
        """Textually a dead value survives its loop's backedge."""
        ops = [
            _op("op0", writes=["a"]),
            _op("op1", reads=["a"], loop_group_id=(7,)),
            _op("op2", loop_group_id=(7,)),
            _op("op3", loop_group_id=(7,)),
            _op("op4"),
        ]
        graph = _graph(ops, {"a": _lx_layout(384)})
        _publish(graph, a=128)
        self.assertEqual(
            frontend_lx_high_water(graph),
            {"op0": 512, "op1": 512, "op2": 512, "op3": 512, "op4": 0},
        )

    def test_loop_local_value_keeps_its_own_lifetime(self):
        """A value born and dying inside the loop is not widened to the loop."""
        ops = [
            _op("op0", loop_group_id=(3,)),
            _op("op1", writes=["t"], loop_group_id=(3,)),
            _op("op2", reads=["t"], loop_group_id=(3,)),
            _op("op3", loop_group_id=(3,)),
        ]
        graph = _graph(ops, {"t": _lx_layout(128)})
        _publish(graph, t=128)
        self.assertEqual(
            frontend_lx_high_water(graph), {"op0": 0, "op1": 256, "op2": 256, "op3": 0}
        )

    def test_zero_when_no_frontend_buffer_is_live(self):
        """Planning that placed nothing hands the whole phase to the backend."""
        ops = [_op("op0"), _op("op1")]
        graph = _graph(ops, {})
        _publish(graph)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 0, "op1": 0})

    def test_bound_is_rounded_up_to_the_allocation_granularity(self):
        """Sizes need not be 128-aligned; the emitted bound must be."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(0)})
        _publish(graph, a=129)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 256})

    def test_missing_footprint_record_emits_nothing(self):
        """An LX-resident buffer with no published footprint: no bound at all."""
        ops = [_op("op0", writes=["a"]), _op("op1", reads=["b"])]
        graph = _graph(ops, {"a": _lx_layout(0), "b": _lx_layout(128)})
        _publish(graph, a=128)
        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_unsized_record_emits_nothing(self):
        """A ``-1``/non-positive footprint is unknown, not zero occupancy."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(0)})
        _publish(graph, a=-1)
        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_no_allocator_records_emits_nothing(self):
        """Without a planning run there is no proof; every op keeps the default."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(0)})
        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_bound_above_planning_size_emits_nothing(self):
        """Placement never exceeds the planning size; a larger record is refused."""
        limit = _lx_planning_size()
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(limit)})
        _publish(graph, a=128)
        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_bound_equal_to_planning_size_is_kept(self):
        """A full-size bound is legal: comparing against the rounded default."""
        limit = _lx_planning_size()
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(limit - 128)})
        _publish(graph, a=128)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": limit})

    def test_bound_is_the_per_core_end_address_not_a_core_aggregate(self):
        """Every core owns the same LX offset, so the bound is one per-core end."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(1 << 20)})
        _publish(graph, a=128 << 10)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 1179648})

    def test_record_beats_any_tensor_size(self):
        """The packed relayout footprint is what gets counted, not the tensor size."""
        ops = [_op("op0", writes=["dst"])]
        graph = _graph(ops, {"dst": _lx_layout(1024)})
        _publish(graph, dst=8192)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 9216})

    def test_demoted_buffer_is_dropped_after_invalidation(self):
        """Demotion clears ``allocation["lx"]``; the cache must not survive it."""
        ops = [_op("op0", writes=["a"], reads=["b"])]
        layout_a = _lx_layout(0)
        layout_b = _lx_layout(4096)
        graph = _graph(ops, {"a": layout_a, "b": layout_b})
        _publish(graph, a=128, b=128)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 4224})
        del layout_b.allocation["lx"]
        invalidate_frontend_lx_high_water(graph)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 128})


class FrontendLxFootprintTest(TestCase):
    def test_recovered_footprint_keeps_current_output_tiling(self):
        for size, cores, tiles in ((1025, 4, 1), (4096, 4, 2), (4096, 4, 4)):
            with self.subTest(size=size, cores=cores, tiles=tiles):
                tiling = TileSpec((TileAxis(host_dim=0, count=tiles),))
                division = CoreDivision(splits={_D0: cores}, tiling=tiling)
                buffer = CoreDivisionBuffer(
                    name="a",
                    size=size,
                    uses=[0],
                    first_use_is_read=False,
                    core_divisions=[division],
                )
                self.assertEqual(buffer.per_core_size(0), ceil_div(size, cores * tiles))
                self.assertEqual(buffer.min_footprint, buffer.per_core_size(0))
