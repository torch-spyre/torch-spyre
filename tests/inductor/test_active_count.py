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

"""for_each_tile(active_count=...): the keyword's checks, the marker that reads
the count once in the loop body so the capture keeps it, the loop's
``count_source`` that the splice makes from it, its read edges, and the
resolver that turns it into the loop's ``CountSource`` after pooling.

No device needed: the IR is stand-ins. The device runs are in
test_active_count_e2e.py.
"""

import unittest
from types import SimpleNamespace

import sympy
import torch
from torch._inductor.dependencies import MemoryDep, ReadWrites, StarDep
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import ElementArrangement, SpyreTensorLayout

from torch_spyre._inductor.deadcode_elimination import live_operations
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.loop_checker import find_violations
from torch_spyre._inductor.loop_node import LoopCountSource, LoopNode, OpsView
from torch_spyre._inductor.op_spec import CountSource, LoopSpec
from torch_spyre._inductor.scheduler import _resolve_count_source
from torch_spyre._inductor.spyre_kernel import SpyreKernel
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile
from torch_spyre._inductor.wsr.for_each_tile_lowering import _take_active_counts


U0, U1 = sympy.Symbol("u0", integer=True), sympy.Symbol("u1", integer=True)


class _Op:
    """A stub op: a name, the buffers it reads and the index it reads them at."""

    def __init__(self, name, reads=(), index=sympy.Integer(0), **attrs):
        self._name = name
        self._reads = list(reads)
        self._index = sympy.sympify(index)
        self.__dict__.update(attrs)

    def get_name(self):
        return self._name

    get_operation_name = get_name

    def get_outputs(self):
        return [self]

    def get_read_writes(self):
        reads = OrderedSet(MemoryDep(r, self._index, (), ()) for r in self._reads)
        writes = OrderedSet([MemoryDep(self._name, sympy.Integer(0), (), ())])
        return ReadWrites(reads, writes, OrderedSet())

    def __repr__(self):
        return f"_Op({self._name})"


def _marker(name, count, index, max_value=4):
    return _Op(name, reads=[count], index=index, active_count_max=max_value)


def _node(name, body, induction=U0):
    return LoopNode(name, body, induction=induction, trip_count=4)


def _counted(name, body, buffer="cnt", index=sympy.Integer(0), induction=U0):
    node = _node(name, body, induction)
    node.count_source = LoopCountSource(buffer, index, 4)
    return node


def _graph(ops, inputs=()):
    return SimpleNamespace(
        operations=list(ops),
        removed_buffers=set(),
        graph_inputs={name: None for name in inputs},
        constants={},
        get_output_names=lambda: [],
    )


class TestTheSpelling(unittest.TestCase):
    """The checks for_each_tile makes before any loop is built (plan 2.1)."""

    @staticmethod
    def _call(count, x=None, **kw):
        x = torch.ones(8, 4) if x is None else x

        def body(_, tiles):
            return None, tiles[0] + 1

        return for_each_tile(
            body, (x,), dims=(0,), tile_size=2, out_dim=0, active_count=count, **kw
        )

    def test_a_non_tensor_count_is_refused(self):
        with self.assertRaisesRegex(TypeError, "active_count must be a tensor"):
            self._call(3)

    def test_a_non_fp32_count_is_refused(self):
        with self.assertRaisesRegex(TypeError, "must be float32"):
            self._call(torch.ones(1, 32, dtype=torch.float16))

    def test_a_count_on_another_device_is_refused(self):
        with self.assertRaisesRegex(ValueError, "on the operands' device"):
            self._call(torch.ones(1, 32, device="meta"))

    def test_a_count_the_loop_tiles_is_refused(self):
        x = torch.ones(8, 4)
        with self.assertRaisesRegex(ValueError, "also an operand this loop tiles"):
            self._call(x, x=x)

    def test_reverse_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reverse=True"):
            self._call(torch.ones(1, 32), reverse=True)

    def test_eager_runs_every_trip(self):
        """Off the device the loop is the static one: all 4 tiles, whatever
        the count says, so trips past the count must be harmless there."""
        _, out = self._call(torch.full((1, 32), 1.0))
        torch.testing.assert_close(out, torch.full((8, 4), 2.0))


def _doubled(x, cnt):
    def body(_, tiles):
        return None, tiles[0] * 2

    return for_each_tile(
        body, (x,), dims=(0,), tile_size=2, out_dim=0, active_count=cnt
    )[1]


class TestTheKeywordReachesTheGraph(unittest.TestCase):
    """Plan 2.2: the lifted count is read once in the body (the marker);
    lifted but unread, the capture would drop it."""

    def test_the_marker_is_in_the_captured_body(self):
        graphs = []

        def backend(gm, _inputs):
            graphs.append(gm)
            return gm

        torch._dynamo.reset()
        x, cnt = torch.ones(8, 4), torch.full((1, 32), 2.0)
        out = torch.compile(_doubled, backend=backend, fullgraph=True)(x, cnt)
        torch.testing.assert_close(out, x * 2)
        text = "\n".join(
            m.print_readable(print_output=False)
            for gm in graphs
            for m in gm.modules()
            if isinstance(m, torch.fx.GraphModule)
        )
        self.assertIn("active_count_marker", text)

    def test_cpu_inductor_matches_eager(self):
        """Plan 2.7: on CPU the loop runs every trip, compiled or not."""
        torch._dynamo.reset()
        x, cnt = torch.randn(8, 4), torch.full((1, 32), 1.0)
        out = torch.compile(_doubled, fullgraph=True)(x, cnt)
        torch.testing.assert_close(out, x * 2)


class TestTheMarker(unittest.TestCase):
    def test_eagerly_it_reads_the_first_value(self):
        count = torch.tensor([[5.0, 9.0], [7.0, 1.0]])
        out = torch.ops.spyre.active_count_marker(count, 16)
        torch.testing.assert_close(out, torch.tensor([5.0]))

    def test_fx_keeps_it_although_nothing_reads_it(self):
        op = torch.ops.spyre.active_count_marker.default
        self.assertIn(op, torch.fx.node._side_effectful_functions)


class TestTheSplice(unittest.TestCase):
    """Plan 2.3: the splice records the marker's read on its loop and removes
    the marker."""

    def test_the_read_becomes_the_loops_count_source(self):
        marker = _marker("m", "cnt", 32 * U0)
        inner = _node("inner", [_Op("t"), marker], induction=U1)
        outer = _node("outer", [inner])
        graph = _graph([_Op("cnt"), outer])
        _take_active_counts(graph)
        self.assertEqual(inner.count_source, LoopCountSource("cnt", 32 * U0, 4))
        self.assertIsNone(outer.count_source)
        self.assertNotIn(marker, inner.body)
        self.assertIn("m", graph.removed_buffers)

    def test_two_markers_in_one_loop_are_refused(self):
        loop = _node("loop", [_marker("m1", "a", 0), _marker("m2", "b", 0)])
        with self.assertRaisesRegex(Unsupported, "2 active_count markers"):
            _take_active_counts(_graph([loop]))

    def test_a_marker_outside_any_loop_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "outside any loop"):
            _take_active_counts(_graph([_marker("m", "cnt", 0)]))


class TestTheReadEdges(unittest.TestCase):
    """Item 4: the loop reads its count, so the count's writer stays live and
    runs first."""

    def test_the_loop_node_reads_the_count(self):
        loop = _counted("loop", [_Op("t")])
        self.assertIn(StarDep("cnt"), loop.get_read_writes().reads)

    def test_dead_code_elimination_keeps_the_count(self):
        """A pass sees the leaves only (OpsView), and no leaf reads the count."""
        graph = _graph([])
        graph.operations = OpsView(
            [_Op("cnt"), _counted("loop", [_Op("t", reads=["x"])])]
        )
        graph.get_output_names = lambda: ["t"]
        self.assertIn("cnt", live_operations(graph))

    def test_a_count_written_before_the_loop_is_fine(self):
        ops = [_Op("cnt"), _counted("loop", [_Op("t")])]
        self.assertEqual(find_violations(ops, inputs=set()), [])

    def test_a_count_written_inside_the_loop_is_flagged(self):
        ops = [_counted("loop", [_Op("cnt"), _Op("t")])]
        problems = find_violations(ops, inputs=set())
        self.assertTrue(any("written inside the loop" in p for p in problems))

    def test_a_count_nothing_writes_is_flagged(self):
        problems = find_violations([_counted("loop", [_Op("t")])], inputs=set())
        self.assertTrue(any("no op writes" in p for p in problems))

    def test_a_graph_input_count_is_not_a_structure_error(self):
        """It is refused at codegen instead, with a message for the user."""
        ops = [_counted("loop", [_Op("t")])]
        self.assertEqual(find_violations(ops, inputs={"cnt"}), [])


class TestCountedLoop(unittest.TestCase):
    """Plan 2.4: counted_loop runs before pooling, so it keeps the source."""

    def test_the_source_waits_on_the_loop_spec(self):
        source = LoopCountSource("cnt", 32 * U0, 4)
        loop = SpyreKernel.counted_loop(sympy.Integer(4), [], source, (U0,))
        self.assertEqual(loop._pending_count_source, (source, (U0,)))
        self.assertIsNone(loop.count_from_buffer)

    def test_an_ordinary_loop_has_none(self):
        loop = SpyreKernel.counted_loop(sympy.Integer(4), [])
        self.assertFalse(hasattr(loop, "_pending_count_source"))


class TestTheResolver(unittest.TestCase):
    """Plan 2.5. A count buffer of 8 rows, one 128-byte stick each, placed at
    pool byte 4096 by pooling; the read index says which row this loop uses."""

    ROWS = 8

    def _layout(self, arrangement=ElementArrangement.STANDARD, cols=32, pool=4096):
        stl = SpyreTensorLayout([self.ROWS, cols], torch.float32)
        if arrangement != ElementArrangement.STANDARD:
            stl = stl.with_element_arrangement(arrangement)
        layout = FixedTiledLayout(
            torch.device("cpu"), torch.float32, [self.ROWS, cols], [cols, 1], stl
        )
        if pool is not None:
            layout.allocation["hbm_pool"] = pool
        return layout

    def _resolve(self, index, enclosing=(), max_value=4, layout=None, trips=4):
        layout = layout or self._layout()
        graph = SimpleNamespace(
            try_get_buffer=lambda name: SimpleNamespace(get_layout=lambda: layout)
        )
        loop = LoopSpec(count=sympy.Integer(trips), body=[])
        source = LoopCountSource("buf_count", sympy.sympify(index), max_value)
        with V.set_graph_handler(graph):
            _resolve_count_source(loop, source, tuple(enclosing))
        return loop.count_from_buffer

    def test_a_per_trip_copy_is_a_fixed_address(self):
        """Prefill as lowered today: each expert trip copies its row to a
        buffer of its own, which the inner loop reads at index 0."""
        self.assertEqual(
            self._resolve(0, enclosing=[U0]),
            CountSource(pool_offset=4096, max_value=4, outer_strides=()),
        )

    def test_a_row_read_in_place_moves_one_stick_per_trip(self):
        self.assertEqual(
            self._resolve(32 * U0, enclosing=[U0]),
            CountSource(pool_offset=4096, max_value=4, outer_strides=(128,)),
        )

    def test_two_levels_give_one_stride_each(self):
        self.assertEqual(
            self._resolve(128 * U0 + 32 * U1 + 64, enclosing=[U0, U1]),
            CountSource(pool_offset=4096 + 256, max_value=4, outer_strides=(512, 128)),
        )

    def test_a_loop_the_count_does_not_move_with_gets_a_zero(self):
        self.assertEqual(
            self._resolve(32 * U1, enclosing=[U0, U1]),
            CountSource(pool_offset=4096, max_value=4, outer_strides=(0, 128)),
        )

    def test_a_count_outside_the_pool_is_refused(self):
        """A host count (a kernel input) lands here: CountSource has no field
        for it yet."""
        with self.assertRaisesRegex(Unsupported, "not in this bundle's pool"):
            self._resolve(0, layout=self._layout(pool=None))

    def test_a_staggered_count_is_refused(self):
        layout = self._layout(arrangement=ElementArrangement.DL16_TO_FP32)
        with self.assertRaisesRegex(Unsupported, "stored DL16_TO_FP32"):
            self._resolve(0, layout=layout)

    def test_a_mid_stick_read_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "not at the start of a stick"):
            self._resolve(1)

    def test_a_row_narrower_than_a_stick_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "one stick per row"):
            self._resolve(0, layout=self._layout(cols=16))

    def test_a_read_over_another_symbol_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "not only on enclosing loop"):
            self._resolve(32 * U1, enclosing=[U0])

    def test_a_maximum_other_than_the_trip_count_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "!= the loop's 4 trips"):
            self._resolve(0, max_value=8)


if __name__ == "__main__":
    unittest.main()
