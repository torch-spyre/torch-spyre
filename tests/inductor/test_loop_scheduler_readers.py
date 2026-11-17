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

"""The readers between the pass loop and the scheduler, on a LoopNode.

Since ``CustomPreSchedulingPasses`` stopped expanding the ``LoopNode``s, every
reader after the pass loop sees them: the work-division finalizer, the INFO
listing (``format_operations``), the cost-model report and dump, and, after
``Scheduler.__init__``, the pre-fusion passes (``propagate_mutation_layouts``,
``build_loop_scheduler_nodes``).  Each must read a loop's body exactly as it
read the flat list.  Everything runs on the CPU with real Inductor objects.
"""

import unittest
from unittest import mock

import sympy
import torch
from torch import fx
from torch._inductor.debug import DebugContext
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    StorageBox,
    TensorBox,
)
from torch._inductor.scheduler import FusedSchedulerNode, Scheduler, SchedulerNode
from torch._inductor.virtualized import V

from torch_spyre._inductor import cost_model_pass as cmp
from torch_spyre._inductor import dump_cost_model as dcm
from torch_spyre._inductor import loop_scheduler
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor import propagate_layouts
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.loop_node import LoopNode, all_ops, expand
from torch_spyre._inductor.loop_scheduler import (
    LoopSchedulerNode,
    apply_loop_scheduler_patches,
)
from torch_spyre._inductor.op_spec import TensorWorkDivision
from torch_spyre._inductor.pass_utils import (
    finalize_work_division_for_scheduler,
    format_operations,
)
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    _loop_group_id,
    _regroup_by_outer_loop_key,
    build_loop_scheduler_nodes,
)
from torch_spyre._inductor.scratchpad.utils import storage_has_single_writer
from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.hbm_pool_planning import hbm_pool_planning
from torch_spyre._inductor.ir import FixedTiledLayout

i0, i1 = sympy.symbols("i0 i1", integer=True, nonnegative=True)


class _Graph(unittest.TestCase):
    """A real CPU ``GraphLowering`` to build IR and a ``Scheduler`` in."""

    def setUp(self):
        graph = GraphLowering(fx.symbolic_trace(lambda: None))
        graph.graph_outputs = []
        self._ctx = [V.set_graph_handler(graph), DebugContext()]
        for ctx in self._ctx:
            ctx.__enter__()
        self.addCleanup(self._exit)
        self.outputs = []
        V.graph.get_output_names = lambda: list(self.outputs)
        self._v = mock.patch.object(torch, "__version__", "2.14.0")
        self._v.start()
        self.addCleanup(self._v.stop)

    def _exit(self):
        for ctx in reversed(self._ctx):
            ctx.__exit__(None, None, None)

    @staticmethod
    def _input(name):
        buf = V.graph.name_to_buffer.get(name)
        if buf is None:
            buf = InputBuffer(
                name=name,
                layout=FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
            )
            V.graph.name_to_buffer[name] = buf
            V.graph.graph_inputs[name] = TensorBox(StorageBox(buf))
        return buf

    def _buf(self, name, reads="x", gid=None):
        src = V.graph.name_to_buffer.get(reads) or self._input(reads)
        box = TensorBox(StorageBox(src))
        pw = Pointwise.create(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: box.make_loader()(index),
            ranges=[8],
        )
        buf = ComputedBuffer(
            name=name,
            layout=FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
            data=pw.data.data,
        )
        buf.operation_name = name
        V.graph.name_to_buffer[name] = buf
        V.graph.name_to_op[name] = buf
        if gid is not None:
            buf.loop_info = CoarseTileInfo(
                loop_group_id=gid,
                loop_count=[4] * len(gid),
                loop_tiled_dims=[[] for _ in gid],
            )
        return buf

    @staticmethod
    def _loop(name, body, trip=4):
        return LoopNode(
            name, body, induction=sympy.Symbol(f"u_{name}"), trip_count=trip
        )

    @staticmethod
    def _graph(operations):
        return type("G", (), {"operations": operations})()


# ---------------------------------------------------------------------------
# finalize_work_division_for_scheduler
# ---------------------------------------------------------------------------


class _OwnedOp:
    """An op with committed ownership, the shape the finalizer reads."""

    def __init__(self, name):
        self.name = name
        write = MemoryDep(f"{name}_out", 8 * i0 + i1, (i0, i1), (8, 8))
        self.rw = type("ReadWrites", (), {"writes": [write], "reads": []})()
        self.iteration_space_ownership = TensorWorkDivision(
            {i0: 2}, {i0: sympy.Integer(0)}
        )

    def get_name(self):
        return self.name

    def get_read_writes(self):
        return self.rw


class TestFinalizer(unittest.TestCase):
    def _finalize(self, operations):
        with mock.patch(
            "torch_spyre._inductor.pass_utils.iteration_space_from_op",
            return_value={i0: 8, i1: 8},
        ):
            finalize_work_division_for_scheduler(
                type("G", (), {"operations": operations})()
            )

    def test_body_ops_are_finalized_like_flat_ops(self):
        flat = [_OwnedOp(n) for n in ("a", "b", "c", "d")]
        self._finalize(flat)
        a, b, c, d = (_OwnedOp(n) for n in ("a", "b", "c", "d"))
        inner = LoopNode("loopnode1", [c], induction=None, trip_count=2)
        outer = LoopNode("loopnode0", [b, inner], induction=None, trip_count=4)
        self._finalize([a, outer, d])
        for f, n in zip(flat, (a, b, c, d)):
            self.assertEqual(n.op_it_space_splits, f.op_it_space_splits, n.name)
            self.assertEqual(n.op_it_space_splits, ({8: 2}, {}))
            self.assertFalse(hasattr(n, "iteration_space_ownership"), n.name)


# ---------------------------------------------------------------------------
# format_operations
# ---------------------------------------------------------------------------


class TestFormatOperations(_Graph):
    def test_flat_list_output_is_unchanged(self):
        # The text a flat list printed before nodes existed: one entry per op,
        # each its own block.  Built here by hand from the same ops, so a
        # change to the flat output (even whitespace) fails.
        ops = [self._buf("a"), self._buf("b", reads="a", gid=(0,))]
        text = format_operations(ops)
        self.assertEqual(text, "".join(format_operations([op]) for op in ops))
        self.assertTrue(text.startswith("a: ComputedBuffer\n  buffer=a\n  layout="))
        self.assertIn("\n  loop_info=CoarseTileInfo(loop_group_id=(0,)", text)
        self.assertTrue(text.endswith("\n\n"))
        self.assertNotIn("LoopNode", text)

    def test_loop_node_prints_its_body_indented(self):
        a = self._buf("a")
        b = self._buf("b", reads="a", gid=(0,))
        c = self._buf("c", reads="b", gid=(0, 0))
        inner = self._loop("loopnode1", [c], trip=2)
        outer = self._loop("loopnode0", [b, inner], trip=4)
        text = format_operations([a, outer])
        flat_b = format_operations([b])
        flat_c = format_operations([c])
        self.assertTrue(text.startswith(format_operations([a])))
        self.assertIn(
            "loopnode0: LoopNode trip_count=4 induction=u_loopnode0 carries=0\n\n",
            text,
        )
        self.assertIn(
            "  loopnode1: LoopNode trip_count=2 induction=u_loopnode1 carries=0\n\n",
            text,
        )

        def indent(block, n):
            return "".join(
                line if line == "\n" else " " * n + line
                for line in block.splitlines(keepends=True)
            )

        self.assertIn(indent(flat_b, 2), text)
        self.assertIn(indent(flat_c, 4), text)
        # Every body op is listed: nothing is dropped behind the node.
        for name in ("a", "b", "c"):
            self.assertIn(f"buffer={name}", text)


# ---------------------------------------------------------------------------
# cost_model_pass.build_report and dump_cost_model
# ---------------------------------------------------------------------------


class TestCostReaders(_Graph):
    def _nested(self):
        a = self._buf("a")
        b = self._buf("b", reads="a", gid=(0,))
        c = self._buf("c", reads="b", gid=(0, 0))
        d = self._buf("d", reads="c")
        flat = [a, b, c, d]
        inner = self._loop("loopnode1", [c], trip=2)
        return flat, [a, self._loop("loopnode0", [b, inner]), d]

    def test_build_report_prices_the_body_as_the_flat_list(self):
        flat, nested = self._nested()
        seen = []

        def extract(op):
            seen.append(op.get_name())
            return mock.MagicMock(name=op.get_name())

        priced = []

        def price(feats, index, params):
            priced.append([f._loop_group_id for f in feats])
            return None

        with (
            mock.patch.object(cmp, "_is_modellable", lambda op: True),
            mock.patch.object(cmp, "extract_op_features", extract),
            mock.patch.object(cmp, "_price", price),
        ):
            cmp.build_report(flat)
            flat_seen, flat_priced = list(seen), list(priced)
            seen.clear()
            priced.clear()
            cmp.build_report(nested)
        self.assertEqual(seen, ["a", "b", "c", "d"])
        self.assertEqual(seen, flat_seen)
        self.assertEqual(priced, flat_priced)
        self.assertEqual(priced, [[None, (0,), (0, 0), None]])

    def test_dump_extracts_the_body_as_the_flat_list(self):
        flat, nested = self._nested()
        got = []
        with (
            mock.patch.object(dcm, "cost_dump_enabled", lambda: True),
            mock.patch.object(
                dcm, "extract_features", lambda ops: got.append(list(ops)) or []
            ),
            mock.patch("torch_spyre._inductor.dump_common.emit", lambda text: None),
        ):
            dcm.dump_cost_model(flat)
            dcm.dump_cost_model(nested)
        self.assertEqual(got[1], got[0])
        self.assertEqual([op.get_name() for op in got[1]], ["a", "b", "c", "d"])


# ---------------------------------------------------------------------------
# Scheduler side: _loop_group_id, build_loop_scheduler_nodes, mutation layouts
# ---------------------------------------------------------------------------


class TestSchedulerPassThrough(_Graph):
    def setUp(self):
        super().setUp()
        self._p = apply_loop_scheduler_patches()
        self._p.__enter__()
        self.addCleanup(self._p.__exit__, None, None, None)

    def _front_half(self, operations):
        sched = Scheduler([])
        sched.nodes = [sched.create_scheduler_node(n) for n in operations]
        sched.update_zero_dim_cpu_tensor()
        for node in sched.nodes:
            node.prune_deps()
        sched.name_to_node = {n.get_name(): n for n in sched.nodes}
        sched.name_to_buf = {
            buf.get_name(): buf for n in sched.nodes for buf in n.get_outputs()
        }
        sched.name_to_fused_node = dict(sched.name_to_node)
        sched.mutation_real_name = {}
        sched.mutation_renames = {}
        sched.compute_dependencies()
        sched.nodes = sched.topological_sort_schedule(sched.nodes)
        sched.dead_node_elimination()
        sched.name_to_fused_node = {n.get_name(): n for n in sched.nodes}
        sched.compute_ancestors()
        return sched.nodes

    def test_loop_group_id_of_a_loop_node_is_none(self):
        b = self._buf("b", gid=(0,))
        c = self._buf("c", reads="b", gid=(0, 0))
        self.outputs = ["c"]
        loop = self._loop("loopnode0", [b, self._loop("loopnode1", [c], trip=2)])
        (snode,) = self._front_half([loop])
        self.assertIsInstance(snode, LoopSchedulerNode)
        self.assertIsNone(_loop_group_id(snode))
        self.assertIsNone(_loop_group_id(snode.snodes[1]))
        # Its leaves still report their own loop membership.
        self.assertEqual(_loop_group_id(snode.snodes[0]), (0,))

    def test_loop_group_id_of_a_flat_counted_loop_is_none(self):
        b = self._buf("b", gid=(0,))
        self.outputs = ["b"]
        (leaf,) = self._front_half([b])
        self.assertEqual(_loop_group_id(leaf), (0,))
        group = CountedLoopSchedulerNode.create([leaf], 4)
        self.assertIsNone(_loop_group_id(group))

    def test_build_loop_scheduler_nodes_passes_a_loop_node_through(self):
        pre = self._buf("pre")
        b = self._buf("b", reads="pre", gid=(0,))
        c = self._buf("c", reads="b", gid=(0, 0))
        post = self._buf("post", reads="b")
        self.outputs = ["post", "c"]
        loop = self._loop("loopnode0", [b, self._loop("loopnode1", [c], trip=2)])
        nodes = self._front_half([pre, loop, post])
        result = build_loop_scheduler_nodes(list(nodes))
        self.assertEqual([n.get_name() for n in result], ["pre", "loopnode0", "post"])
        self.assertIs(result[1], nodes[1])
        self.assertIsInstance(result[1], LoopSchedulerNode)
        self.assertIsInstance(result[1].snodes[1], LoopSchedulerNode)
        # Not wrapped again: one loop at each level.
        self.assertEqual(
            [type(n).__name__ for n in result[1].snodes],
            ["SchedulerNode", "LoopSchedulerNode"],
        )

    def test_regroup_keeps_a_loop_node_in_place_among_flat_loop_ops(self):
        # A mixed list: a structured loop and a FLAT loop (its ops carry
        # loop_info group 1 and still need regrouping, as span-overflow and
        # allocator tiling produce today).
        pre = self._buf("pre")
        b = self._buf("b", reads="pre", gid=(0,))
        f1 = self._buf("f1", reads="pre", gid=(1,))
        mid = self._buf("mid", reads="pre")
        f2 = self._buf("f2", reads="f1", gid=(1,))
        self.outputs = ["b", "f2", "mid"]
        loop = self._loop("loopnode0", [b])
        nodes = self._front_half([pre, loop, f1, mid, f2])
        names = [n.get_name() for n in nodes]
        self.assertEqual(names[0], "pre")
        ordered = _regroup_by_outer_loop_key(list(nodes))
        # The loop node stays where it was; the flat group-1 ops are made
        # contiguous around the independent ``mid``.
        self.assertEqual(
            [n.get_name() for n in ordered].index("loopnode0"),
            names.index("loopnode0"),
        )
        i1_, i2_ = (
            [n.get_name() for n in ordered].index("f1"),
            [n.get_name() for n in ordered].index("f2"),
        )
        self.assertEqual(i2_, i1_ + 1)
        result = build_loop_scheduler_nodes(list(nodes))
        kinds = [(type(n).__name__, n.get_name()) for n in result]
        self.assertIn(("LoopSchedulerNode", "loopnode0"), kinds)
        flat_loops = [n for n in result if type(n) is CountedLoopSchedulerNode]
        self.assertEqual(len(flat_loops), 1)
        self.assertEqual([n.get_name() for n in flat_loops[0].snodes], ["f1", "f2"])

    def test_a_flat_member_left_outside_its_loop_node_is_caught(self):
        # A flat op whose loop_info says it belongs to the loop the node
        # already is (group 0) would become a second, separate group-0 loop.
        b = self._buf("b", gid=(0,))
        mid = self._buf("mid", reads="b")
        stray = self._buf("stray", reads="mid", gid=(0,))
        self.outputs = ["stray"]
        nodes = self._front_half([self._loop("loopnode0", [b]), mid, stray])
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            build_loop_scheduler_nodes(list(nodes))

    def test_stray_member_is_caught_when_the_first_child_is_a_nested_loop(self):
        # The check reads the first member's own loop levels: a nested loop
        # node as first member has no _loop_group_id (it is a loop group).
        b = self._buf("b", gid=(0, 0))
        mid = self._buf("mid", reads="b")
        stray = self._buf("stray", reads="mid", gid=(0,))
        self.outputs = ["stray"]
        inner = self._loop("loopnode1", [b], trip=2)
        nodes = self._front_half([self._loop("loopnode0", [inner]), mid, stray])
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            build_loop_scheduler_nodes(list(nodes))

    def test_scheduler_time_readers_see_leaf_ops(self):
        # Scheduler._init reads graph.operations once, to make the scheduler
        # nodes; kernel preparation reads it again (the LX read contracts:
        # storage_has_single_writer, _is_activation_source).  Those readers
        # expect leaf ops: a LoopNode has no get_mutation_names.
        pre = self._buf("pre")
        b = self._buf("b", reads="pre", gid=(0,))
        c = self._buf("c", reads="b", gid=(0, 0))
        post = self._buf("post", reads="b")
        self.outputs = ["post", "c"]
        loop = self._loop("loopnode0", [b, self._loop("loopnode1", [c], trip=2)])
        V.graph.operations = [pre, loop, post]
        nodes = self._front_half(list(V.graph.operations))
        self.assertIsInstance(nodes[1], LoopSchedulerNode)
        self.assertEqual(V.graph.operations, [pre, b, c, post])
        # The body is still the node's: the structure lives in the scheduler.
        self.assertEqual(loop.body[0], b)
        for producer in (pre, b, c):
            self.assertTrue(storage_has_single_writer(V.graph, producer))

    def test_propagate_mutation_layouts_reaches_a_body_op(self):
        # A body op that mutates ``acc`` keeps its MutationLayoutSHOULDREMOVE
        # through Scheduler.__init__; the pre-fusion pass then resolves it to
        # the target's real layout.  Inside a LoopSchedulerNode it must too.
        acc = self._buf("acc")
        src = TensorBox(StorageBox(self._input("x")))
        pw = Pointwise.create(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: src.make_loader()(index),
            ranges=[8],
        )
        upd = ComputedBuffer(
            name="upd", layout=MutationLayoutSHOULDREMOVE(acc), data=pw.data.data
        )
        upd.operation_name = "upd"
        V.graph.name_to_buffer["upd"] = upd
        V.graph.name_to_op["upd"] = upd
        self.outputs = ["acc"]
        with mock.patch.object(
            SchedulerNode,
            "has_side_effects",
            lambda self: isinstance(self.node.layout, MutationLayoutSHOULDREMOVE),
        ):
            nodes = self._front_half([acc, self._loop("loopnode0", [upd])])
        self.assertIsInstance(nodes[1], LoopSchedulerNode)
        with (
            # The CPU target has a FixedLayout; stand it in for the Spyre one.
            mock.patch.object(propagate_layouts, "FixedTiledLayout", FixedLayout),
            mock.patch.object(
                propagate_layouts, "_real_layout_matches_op_size", lambda n, r: True
            ),
        ):
            propagate_layouts.propagate_mutation_layouts(nodes)
        self.assertIs(upd.layout, acc.layout)

    def _pool_layout(self, buf):
        # The pool planner sizes and places FixedTiledLayout buffers only.
        stl = SpyreTensorLayout(
            [8], [1], torch.float16, [0], ElementArrangement.STANDARD
        )
        buf.layout = FixedTiledLayout(torch.device("cpu"), torch.float16, [8], [1], stl)

    def test_hbm_pool_never_frees_a_body_buffer_before_its_loop_ends(self):
        # A LoopSchedulerNode keeps only EXTERNAL reads, so a buffer written
        # and read only inside the loop (``t``, read by ``u``) shows no read
        # at the loop's timestep.  hbm_pool_planning must then keep it live
        # for the rest of the bundle.  Ending its range at its last read in
        # the body (step 1) would free it before ``w`` (step 2) is placed and
        # hand ``w`` the same offset, but the body runs trip_count times:
        # the next trip writes ``w`` while ``t`` is still to be read.  Checked
        # with the loop inside a fused bundle and as a bare bundle.
        for bare in (False, True):
            with self.subTest(bare=bare):
                V.graph.removed_buffers.clear()
                t = self._buf("t", reads="x", gid=(0,))
                u = self._buf("u", reads="t", gid=(0,))
                w = self._buf("w", reads="x", gid=(0,))
                # ``u2`` keeps ``u`` an intermediate (read inside the bundle).
                u2 = self._buf("u2", reads="u", gid=(0,))
                w2 = self._buf("w2", reads="w", gid=(0,))
                body = [t, u, w, u2, w2]
                final = self._buf("final", reads="w2")
                outer = self._buf("outer", reads="u2")
                # Something outside reads the loop, so it survives DNE.
                self.outputs = ["u2", "w2"] if bare else ["final", "outer"]
                ops = [self._loop("loopnode0", body)]
                ops += [] if bare else [final, outer]
                nodes = self._front_half(ops)
                loop = nodes[0]
                self.assertIsInstance(loop, LoopSchedulerNode)
                self.assertNotIn("t", {d.name for d in loop.read_writes.reads})
                for buf in (*body, final, outer):
                    self._pool_layout(buf)
                if bare:
                    bundle = loop
                else:
                    bundle = FusedSchedulerNode(loop.scheduler, list(nodes))
                with config.patch({"hbm_pool_planning": True}):
                    hbm_pool_planning([bundle])
                offsets = {
                    b.get_name(): b.get_layout().allocation.get("hbm_pool")
                    for b in body
                }
                self.assertIsNotNone(offsets["t"], offsets)
                self.assertIsNotNone(offsets["w"], offsets)
                pooled = [o for o in offsets.values() if o is not None]
                self.assertEqual(len(pooled), len(set(pooled)), offsets)

    def test_hbm_pool_reuses_a_finished_loops_body_buffer(self):
        # Two sibling loops in one bundle.  A LoopSchedulerNode keeps only
        # external reads, so its body-only buffers showed no read; their
        # range ran to the end of the bundle and the second loop could not
        # reuse the first loop's space (the paged-gather graph: pool 163840
        # bytes vs 90112 on the flat path).  A buffer a loop writes is live
        # through that loop's step, and no longer unless read later.
        t1 = self._buf("t1", reads="x", gid=(0,))
        o1 = self._buf("o1", reads="t1", gid=(0,))
        t2 = self._buf("t2", reads="x", gid=(1,))
        o2 = self._buf("o2", reads="t2", gid=(1,))
        out = self._buf("out", reads="o1")
        out2 = self._buf("out2", reads="o2")
        self.outputs = ["out", "out2"]
        nodes = self._front_half(
            [
                self._loop("loopnode0", [t1, o1]),
                self._loop("loopnode1", [t2, o2]),
                out,
                out2,
            ]
        )
        for buf in (t1, o1, t2, o2, out, out2):
            self._pool_layout(buf)
        bundle = FusedSchedulerNode(nodes[0].scheduler, list(nodes))
        with config.patch({"hbm_pool_planning": True}):
            hbm_pool_planning([bundle])
        offset = {
            b.get_name(): b.get_layout().allocation.get("hbm_pool")
            for b in (t1, o1, t2, o2)
        }
        # t1 is dead once loopnode0 ends; t2 (written by loopnode1) reuses it.
        self.assertEqual(offset["t2"], offset["t1"], offset)
        # o1 is read after both loops: never shared with anything in loopnode1.
        self.assertNotIn(offset["o1"], (offset["t2"], offset["o2"]), offset)


# ---------------------------------------------------------------------------
# The hand-off in CustomPreSchedulingPasses
# ---------------------------------------------------------------------------


class _Pipeline(ts_passes.CustomPreSchedulingPasses):
    def __init__(self, body):
        self.passes = []
        self._body = body

    def _run_pass_loop(self, graph):
        self._body(graph)


class TestHandOff(_Graph):
    def _graph_with_loop(self):
        b = self._buf("b", gid=(0,))
        graph = self._graph([self._buf("a"), self._loop("loopnode0", [b])])
        return graph

    def _run(self, graph, body=lambda g: None):
        with (
            mock.patch.object(
                ts_passes, "_operations_have_spyre_device", lambda o: True
            ),
            mock.patch.object(ts_passes, "cost_model_pass", lambda g: None),
            mock.patch.object(ts_passes, "dump_cost_model", lambda o: None),
            mock.patch.object(
                ts_passes, "finalize_work_division_for_scheduler", lambda g: None
            ),
        ):
            _Pipeline(body)(graph)

    def test_supported_release_keeps_the_nodes(self):
        graph = self._graph_with_loop()
        self._run(graph)
        self.assertIsInstance(graph.operations[1], LoopNode)

    def test_switch_off_expands(self):
        graph = self._graph_with_loop()
        with mock.patch.object(loop_scheduler, "SCHEDULE_LOOP_NODES", False):
            self._run(graph)
        self.assertEqual([op.get_name() for op in graph.operations], ["a", "b"])

    def test_unsupported_release_with_a_loop_raises_naming_the_release(self):
        for text in ("2.12.0", "2.13.0", "2.15.1"):
            with self.subTest(text), mock.patch.object(torch, "__version__", text):
                graph = self._graph_with_loop()
                with self.assertRaisesRegex(Unsupported, text) as ctx:
                    self._run(graph)
                self.assertIn("supports only torch 2.14.x", str(ctx.exception))
                # Nothing is left behind for an error handler.
                self.assertFalse(
                    any(isinstance(op, LoopNode) for op in graph.operations)
                )

    def test_unsupported_release_without_a_loop_is_unaffected(self):
        with mock.patch.object(torch, "__version__", "2.15.0"):
            graph = self._graph([self._buf("a")])
            self._run(graph)
        self.assertEqual([op.get_name() for op in graph.operations], ["a"])

    def test_pass_error_flattens_then_reraises(self):
        graph = self._graph_with_loop()

        def boom(g):
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            self._run(graph, boom)
        self.assertEqual([op.get_name() for op in graph.operations], ["a", "b"])

    def test_expand_is_still_the_inverse(self):
        graph = self._graph_with_loop()
        leaves = list(all_ops(graph.operations))
        expand(graph)
        self.assertEqual(graph.operations, leaves)


if __name__ == "__main__":
    unittest.main()
