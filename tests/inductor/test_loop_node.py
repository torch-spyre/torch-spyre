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

"""Unit tests for ``loop_node``: the node, the flat view, collapse/expand."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sympy
import torch
from torch import fx
from torch._inductor.dependencies import MemoryDep, ReadWrites, StarDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    ExternKernel,
    FallbackKernel,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor import loop_node as ln
from torch_spyre._inductor.deadcode_elimination import (
    _has_side_effects,
    deadcode_elimination,
)
from torch_spyre._inductor.loop_info import CoarseTileInfo, LoopCarryRecord
from torch_spyre._inductor.loop_node import (
    LoopNode,
    OpsView,
    all_ops,
    building_loop,
    collapse,
    expand,
    expanded,
    in_building_loop,
    owning_list,
)


def _dep(name):
    return MemoryDep(name, sympy.Integer(0), (), ())


class _Op:
    """A stub op: a name, reads/writes, optional loop_info and layout."""

    def __init__(self, name, gid=None, counts=None, reads=(), writes=None):
        self._name = name
        if gid is not None:
            self.loop_info = CoarseTileInfo(
                loop_group_id=tuple(gid),
                loop_count=list(counts),
                loop_tiled_dims=[[] for _ in gid],
                loop_tiled_reduction_dims=[[] for _ in gid],
            )
        self._reads = [_dep(r) for r in reads]
        self._writes = [_dep(w) for w in (writes if writes is not None else [name])]
        self.layout = None

    def get_name(self):
        return self._name

    def get_operation_name(self):
        return self._name

    def get_device(self):
        return torch.device("cpu")

    def get_outputs(self):
        return [self]

    def get_read_writes(self):
        return ReadWrites(
            OrderedSet(self._reads), OrderedSet(self._writes), OrderedSet()
        )

    def __repr__(self):
        return f"_Op({self._name})"


def _graph(ops):
    return SimpleNamespace(operations=list(ops))


def _names(ops):
    return [op.get_name() for op in ops]


def _node(name, body, trip=4, induction=None):
    return LoopNode(name, body, induction=induction, trip_count=trip)


class TestLoopNode(unittest.TestCase):
    def test_construction_has_scheduler_attributes(self):
        a = _Op("a")
        node = _node("loopnode0", [a], induction=sympy.Symbol("u0"))
        for attr in ("origin_node", "origins", "stream_idx", "mempool"):
            self.assertTrue(hasattr(node, attr), attr)
        # torch's ir.Operation accessors assert the attribute exists.
        self.assertIsNone(node.get_origin_node())
        self.assertEqual(node.get_origins(), OrderedSet())
        self.assertIsNone(node.get_stream_idx())
        self.assertIsNone(node.mempool)
        self.assertEqual(node.get_name(), "loopnode0")
        self.assertEqual(node.get_operation_name(), "loopnode0")
        self.assertEqual(node.name, "loopnode0")
        self.assertFalse(node.is_extern())
        self.assertEqual(node.get_device(), torch.device("cpu"))
        self.assertEqual(node.induction, sympy.Symbol("u0"))
        self.assertIn("loopnode0", repr(node))

    def test_origins_and_origin_node_kept(self):
        origin = object()
        node = LoopNode(
            "n",
            [_Op("a")],
            induction=None,
            trip_count=2,
            origin_node=origin,
            origins=OrderedSet([origin]),
        )
        self.assertIs(node.get_origin_node(), origin)
        self.assertEqual(list(node.get_origins()), [origin])

    def test_nodes_compare_by_identity(self):
        n1, n2 = _node("n1", [_Op("a")]), _node("n2", [_Op("b")])
        self.assertNotEqual(n1, n2)
        lst = [n1, n2]
        lst.remove(n2)
        self.assertEqual(lst, [n1])

    def test_get_outputs_concatenates_body(self):
        a, b, c = _Op("a"), _Op("b"), _Op("c")
        node = _node("outer", [a, _node("inner", [b]), c])
        self.assertEqual(node.get_outputs(), [a, b, c])
        self.assertEqual(node.get_device(), torch.device("cpu"))

    def test_get_read_writes_union_and_mutation_stardep(self):
        target = _Op("acc")
        a = _Op("a", reads=["x"], writes=["a"])
        m = _Op("m", reads=["a", "acc"], writes=["m"])
        m.layout = MagicMock(spec=MutationLayoutSHOULDREMOVE)
        m.layout.get_buffer.return_value = target
        # _mutation_target only accepts a ComputedBuffer.
        m.__class__ = type("_MutOp", (_Op, ComputedBuffer), {})
        node = _node("n", [a, _node("inner", [m])])
        rw = node.get_read_writes()
        self.assertEqual({d.name for d in rw.reads}, {"x", "a", "acc"})
        self.assertIn(StarDep("acc"), rw.writes)
        self.assertEqual(
            {d.name for d in rw.writes if not isinstance(d, StarDep)}, {"a", "m"}
        )
        self.assertEqual(node.mutation_targets(), ["acc"])
        self.assertTrue(node.has_side_effects())
        self.assertTrue(_has_side_effects(node))

    def test_pure_node_has_no_side_effects(self):
        node = _node("n", [_Op("a"), _node("inner", [_Op("b")])])
        self.assertFalse(node.has_side_effects())
        self.assertFalse(_has_side_effects(node))
        self.assertEqual(node.mutation_targets(), [])

    def test_impure_fallback_in_body_has_side_effects(self):
        fb = MagicMock(spec=FallbackKernel)
        fb.has_side_effects.return_value = True
        node = _node("n", [_Op("a"), fb])
        self.assertTrue(node.has_side_effects())
        fb.has_side_effects.return_value = False
        self.assertFalse(node.has_side_effects())


class TestHelpers(unittest.TestCase):
    def test_all_ops_and_owning_list(self):
        a, b, c, d = (_Op(n) for n in "abcd")
        inner = _node("inner", [b, c])
        outer = _node("outer", [inner, d])
        ops = [a, outer]
        self.assertEqual(list(all_ops(ops)), [a, b, c, d])
        self.assertEqual(owning_list(ops, a), (ops, 0))
        self.assertEqual(owning_list(ops, c), (inner.body, 1))
        self.assertEqual(owning_list(ops, inner), (outer.body, 0))
        self.assertIsNone(owning_list(ops, _Op("z")))

    def test_building_loop(self):
        self.assertFalse(in_building_loop())
        with building_loop():
            self.assertTrue(in_building_loop())
            with building_loop():
                self.assertTrue(in_building_loop())
            self.assertTrue(in_building_loop())
        self.assertFalse(in_building_loop())

    def test_building_loop_resets_on_error(self):
        with self.assertRaises(ZeroDivisionError):
            with building_loop():
                1 / 0
        self.assertFalse(in_building_loop())


def _round_trip(test, ops):
    graph = _graph(ops)
    before = list(graph.operations)
    real = graph.operations
    collapse(graph)
    test.assertIs(graph.operations, real, "collapse must edit the list in place")
    collapsed = list(graph.operations)
    test.assertEqual(list(all_ops(collapsed)), before)
    expand(graph)
    test.assertIs(graph.operations, real)
    after = list(graph.operations)
    test.assertEqual(len(after), len(before))
    for x, y in zip(after, before):
        test.assertIs(x, y)
    test.assertEqual(_names(after), _names(before))
    test.assertFalse(any(isinstance(o, LoopNode) for o in after))
    return collapsed


class TestCollapse(unittest.TestCase):
    def test_flat_loop(self):
        pre, a, b, post = (
            _Op("pre"),
            _Op("a", (0,), [4]),
            _Op("b", (0,), [4]),
            _Op("post"),
        )
        out = _round_trip(self, [pre, a, b, post])
        self.assertEqual(len(out), 3)
        self.assertIs(out[0], pre)
        self.assertIs(out[2], post)
        node = out[1]
        self.assertIsInstance(node, LoopNode)
        self.assertEqual(node.body, [a, b])
        self.assertEqual(node.trip_count, 4)
        self.assertIsNone(node.induction)

    def test_nested_loop(self):
        a = _Op("a", (0,), [2])
        b = _Op("b", (0, 1), [2, 8])
        c = _Op("c", (0, 1), [2, 8])
        d = _Op("d", (0,), [2])
        out = _round_trip(self, [a, b, c, d])
        (outer,) = out
        self.assertEqual(outer.trip_count, 2)
        self.assertIs(outer.body[0], a)
        inner = outer.body[1]
        self.assertIsInstance(inner, LoopNode)
        self.assertEqual(inner.body, [b, c])
        self.assertEqual(inner.trip_count, 8)
        self.assertIs(outer.body[2], d)

    def test_two_sibling_loops(self):
        a = _Op("a", (0,), [4])
        mid = _Op("mid")
        b = _Op("b", (1,), [3])
        out = _round_trip(self, [a, mid, b])
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].body, [a])
        self.assertIs(out[1], mid)
        self.assertEqual(out[2].body, [b])
        self.assertEqual((out[0].trip_count, out[2].trip_count), (4, 3))

    def test_sibling_inner_loops(self):
        ops = [
            _Op("i0", (0, 1), [2, 4]),
            _Op("x", (0,), [2]),
            _Op("j0", (0, 2), [2, 5]),
        ]
        (outer,) = _round_trip(self, ops)
        self.assertEqual(len(outer.body), 3)
        self.assertEqual(outer.body[0].trip_count, 4)
        self.assertIs(outer.body[1], ops[1])
        self.assertEqual(outer.body[2].trip_count, 5)

    def test_ops_between_members_stay_in_span(self):
        # Ops without a group id that sit between two members of a loop are
        # kept in that loop's body, in place.
        a = _Op("a", (0,), [4])
        between = _Op("between")
        b = _Op("b", (0,), [4])
        (node,) = _round_trip(self, [a, between, b])
        self.assertEqual(node.body, [a, between, b])

    def test_other_loop_between_members_stays_in_span(self):
        a = _Op("a", (0,), [4])
        other = _Op("other", (1,), [3])
        b = _Op("b", (0,), [4])
        (node,) = _round_trip(self, [a, other, b])
        # ``other`` is a different top-level loop; it is kept in place, not
        # wrapped into a node of its own inside loop 0.
        self.assertEqual(node.body, [a, other, b])

    def test_op_with_no_loop_info(self):
        ops = [_Op("x"), _Op("y")]
        graph = _graph(ops)
        collapse(graph)
        self.assertEqual(graph.operations, ops)

    def test_extern_kernel_with_loop_info_is_not_a_member(self):
        ext = MagicMock(spec=ExternKernel)
        ext.loop_info = CoarseTileInfo((0,), [4], [[]], [[]])
        ext.get_name.return_value = "ext"
        a = _Op("a", (0,), [4])
        out = _round_trip(self, [ext, a])
        self.assertIs(out[0], ext)
        self.assertEqual(out[1].body, [a])
        # Between members, it stays in the span like any other non-member.
        b = _Op("b", (0,), [4])
        (node,) = _round_trip(self, [a, ext, b])
        self.assertEqual(node.body, [a, ext, b])

    def test_induction_and_carries(self):
        u0, u1 = sympy.Symbol("u0"), sympy.Symbol("u1")
        a = _Op("a", (0,), [2])
        b = _Op("b", (0, 1), [2, 8])
        record = LoopCarryRecord(storage_name="a", update_name="b")
        a._loop_carry_record = record
        b._loop_carry_record = record
        graph = _graph([a, b])
        collapse(graph, inductions={0: u0, 1: u1})
        (outer,) = graph.operations
        inner = outer.body[1]
        self.assertEqual((outer.induction, inner.induction), (u0, u1))
        self.assertEqual(outer.carries, (record,))
        self.assertEqual(inner.carries, (record,))

    def test_expanded_runs_flat_then_rebuilds(self):
        u0, u1 = sympy.Symbol("u0"), sympy.Symbol("u1")
        const = _Op("const")
        a = _Op("a", (0,), [2])
        b = _Op("b", (0, 1), [2, 8])
        graph = _graph([a, b, const])
        collapse(graph, inductions={0: u0, 1: u1})
        real = graph.operations
        with expanded(graph):
            self.assertEqual(graph.operations, [a, b, const])
            # A pass hoists the constant across the loop, as dedup does.
            graph.operations[:] = [const, a, b]
        self.assertIs(graph.operations, real)
        self.assertIs(real[0], const)
        (outer,) = real[1:]
        self.assertEqual(outer.induction, u0)
        self.assertEqual(outer.body[1].induction, u1)
        self.assertEqual(list(all_ops(real)), [const, a, b])

    def test_disabled_switch_builds_nothing(self):
        ops = [_Op("a", (0,), [4])]
        graph = _graph(ops)
        old = ln.BUILD_LOOP_NODES
        ln.BUILD_LOOP_NODES = False
        try:
            collapse(graph)
        finally:
            ln.BUILD_LOOP_NODES = old
        self.assertEqual(graph.operations, ops)

    def test_unique_node_names(self):
        ops = [
            _Op("a", (0, 1), [2, 2]),
            _Op("b", (0, 2), [2, 2]),
            _Op("c", (3,), [2]),
        ]
        graph = _graph(ops)
        collapse(graph)
        names = []

        def walk(lst):
            for o in lst:
                if isinstance(o, LoopNode):
                    names.append(o.get_name())
                    walk(o.body)

        walk(graph.operations)
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 4)


def _nested():
    """``[a, outer[b, inner[c, d], e], f]`` and its pieces."""
    a, b, c, d, e, f = (_Op(n) for n in "abcdef")
    inner = _node("inner", [c, d])
    outer = _node("outer", [b, inner, e])
    return [a, outer, f], outer, inner, (a, b, c, d, e, f)


class TestOpsView(unittest.TestCase):
    def test_iteration_is_flat(self):
        real, _, _, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        self.assertEqual(list(view), [a, b, c, d, e, f])
        self.assertEqual(list(reversed(view)), [f, e, d, c, b, a])
        self.assertEqual(len(view), 6)
        self.assertIs(view[2], c)
        self.assertEqual(view[1:3], [b, c])
        self.assertEqual(view.index(d), 3)
        self.assertIn(e, view)

    def test_append_goes_to_top_level(self):
        real, *_ = _nested()
        x = _Op("x")
        OpsView(real).append(x)
        self.assertIs(real[-1], x)

    def test_insert_before_op_in_body(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        x = _Op("x")
        view.insert(view.index(d), x)
        self.assertEqual(inner.body, [c, x, d])
        self.assertEqual(list(view), [a, b, c, x, d, e, f])

    def test_insert_after_op_in_body(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        x = _Op("x")
        # "after d" is "before e", which lives in the outer body.
        view.insert(view.index(d) + 1, x)
        self.assertEqual(outer.body, [b, inner, x, e])
        self.assertEqual(list(view), [a, b, c, d, x, e, f])

    def test_insert_at_end(self):
        real, *_ = _nested()
        view = OpsView(real)
        x, y = _Op("x"), _Op("y")
        view.insert(len(view), x)
        view.insert(100, y)
        self.assertEqual(real[-2:], [x, y])

    def test_insert_negative_index(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        x = _Op("x")
        view.insert(-1, x)  # before f, as list.insert does
        self.assertEqual(real, [a, outer, x, f])

    def test_remove(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        view.remove(c)
        self.assertEqual(inner.body, [d])
        view.remove(a)
        self.assertEqual(real, [outer, f])
        with self.assertRaises(ValueError):
            view.remove(_Op("z"))

    def test_setitem(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        x = _Op("x")
        view[view.index(d)] = x
        self.assertEqual(inner.body, [c, x])
        y = _Op("y")
        view[-1] = y
        self.assertIs(real[-1], y)

    def test_delitem(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        del view[view.index(e)]
        self.assertEqual(outer.body, [b, inner])
        del view[0]
        self.assertEqual(real, [outer, f])

    def test_mutation_while_iterating_matches_list(self):
        # A pass that removes the op it is visiting skips the next one on a
        # list; the view must behave the same.
        real, *_ = _nested()
        flat = list(all_ops(real))
        seen_view, seen_list = [], []
        view = OpsView(real)
        for op in view:
            seen_view.append(op)
            if op.get_name() in ("b", "d"):
                view.remove(op)
        for op in flat:
            seen_list.append(op)
            if op.get_name() in ("b", "d"):
                flat.remove(op)
        self.assertEqual(seen_view, seen_list)
        self.assertEqual(list(view), flat)

    def test_reorder_permutes_within_each_list(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        # Move f and d to the front, as a "hoist to front" rewrite would.
        view[:] = [f, d, a, b, c, e]
        # Each op stays in its own list, and a node sorts by its earliest op:
        # d (rank 1) drags inner ahead of b and outer ahead of a. So the flat
        # result differs from the same rewrite on a flat list.
        self.assertEqual(real, [f, outer, a])
        self.assertEqual(outer.body, [inner, b, e])
        self.assertEqual(inner.body, [d, c])
        self.assertEqual(list(view), [f, d, c, b, e, a])

    def test_reorder_node_sorts_by_earliest_body_op(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        view[:] = [c, a, b, d, e, f]
        # outer sorts by its earliest op (c), ahead of a.
        self.assertEqual(real, [outer, a, f])

    def test_reorder_rejects_added_or_dropped_ops(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        with self.assertRaises(RuntimeError):
            view[:] = [a, b, c, d, e]
        with self.assertRaises(RuntimeError):
            view[:] = [a, b, c, d, e, f, _Op("x")]
        with self.assertRaises(RuntimeError):
            view[:] = [a, b, c, d, e, e]
        with self.assertRaises(NotImplementedError):
            view[1:3] = [b, c]
        self.assertEqual(list(view), [a, b, c, d, e, f])

    def test_cache_invalidated_by_every_edit(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        self.assertEqual(len(view), 6)
        x = _Op("x")
        view.insert(2, x)
        self.assertEqual(len(view), 7)
        view.remove(x)
        view.append(x)
        self.assertIs(view[-1], x)
        del view[-1]
        self.assertEqual(len(view), 6)


class TestDeadcodeOnRealNestedList(unittest.TestCase):
    """``deadcode_elimination`` given the REAL nested list (no view)."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()
        self.addCleanup(self._graph_ctx.__exit__, None, None, None)

    @staticmethod
    def _buf(name, reads=None):
        src_name = reads if reads is not None else f"in_{name}"
        src = V.graph.name_to_buffer.get(src_name)
        if src is None:
            src = InputBuffer(
                name=src_name,
                layout=FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
            )
            V.graph.name_to_buffer[src_name] = src
        box = TensorBox(StorageBox(src))
        pw = Pointwise.create(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: box.make_loader()(index),
            ranges=[8],
        )
        buf = ComputedBuffer(
            name=name,
            layout=FixedLayout(torch.device("cpu"), torch.float32, [8], None),
            data=pw.data.data,
        )
        buf.operation_name = name
        V.graph.name_to_buffer[name] = buf
        return buf

    def _run(self, operations, output_names):
        V.graph.operations = operations
        V.graph.get_output_names = lambda: list(output_names)
        deadcode_elimination(V.graph)
        return V.graph.operations

    def test_mutation_only_loop_is_kept(self):
        # The loop writes only into ``acc`` (a mutation); nothing reads any of
        # its own outputs.  It must survive because ``acc`` is live.
        acc = self._buf("acc")
        tile = self._buf("tile")
        update = self._buf("update", reads="tile")
        update.layout = MutationLayoutSHOULDREMOVE(acc)
        node = _node("loopnode0", [tile, update])
        survivors = self._run([acc, node], output_names=["acc"])
        self.assertEqual(survivors, [acc, node])
        self.assertEqual(node.body, [tile, update])
        self.assertNotIn("update", V.graph.removed_buffers)

    def test_dead_loop_is_removed_with_every_leaf_output(self):
        acc = self._buf("acc")
        tile = self._buf("tile")
        update = self._buf("update", reads="tile")
        update.layout = MutationLayoutSHOULDREMOVE(acc)
        inner_op = self._buf("inner_op", reads="tile")
        node = _node("loopnode0", [tile, _node("loopnode1", [inner_op]), update])
        keep = self._buf("keep")
        survivors = self._run([acc, node, keep], output_names=["keep"])
        self.assertEqual(survivors, [keep])
        for name in ("tile", "update", "inner_op", "acc"):
            self.assertIn(name, V.graph.removed_buffers)

    def test_pure_loop_kept_through_its_outputs(self):
        tile = self._buf("tile")
        out = self._buf("out", reads="tile")
        node = _node("loopnode0", [tile, out])
        survivors = self._run([node], output_names=["out"])
        self.assertEqual(survivors, [node])

    def test_mutation_into_dead_target_loop_is_removed(self):
        acc = self._buf("acc")
        update = self._buf("update")
        update.layout = MutationLayoutSHOULDREMOVE(acc)
        node = _node("loopnode0", [update])
        keep = self._buf("keep")
        survivors = self._run([acc, node, keep], output_names=["keep"])
        self.assertEqual(survivors, [keep])

    def test_impure_fallback_keeps_loop(self):
        fb = MagicMock(spec=FallbackKernel)
        fb.has_side_effects.return_value = True
        fb.get_read_writes.return_value = ReadWrites(
            OrderedSet(), OrderedSet([_dep("fb")]), OrderedSet()
        )
        fb.get_operation_name.return_value = "fb"
        node = _node("loopnode0", [fb])
        survivors = self._run([node], output_names=[])
        self.assertEqual(survivors, [node])


if __name__ == "__main__":
    pytest.main([__file__])
