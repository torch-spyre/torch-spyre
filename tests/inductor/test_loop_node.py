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

from torch_spyre._inductor.deadcode_elimination import (
    _has_side_effects,
    deadcode_elimination,
)
from torch_spyre._inductor.loop_facts import (
    counted_loop_path,
    enclosing_loops,
    is_looped,
    node_levels,
    pending_levels,
)
from torch_spyre._inductor.loop_node import LoopMember
from torch_spyre._inductor.loop_node import LoopCarryRecord
from torch_spyre._inductor.loop_node import (
    LoopNode,
    OpsView,
    all_ops,
    build_pending_loop_nodes,
    building_loop,
    collapse,
    expand,
    in_building_loop,
    insert_after,
    insert_before,
    move_after,
    move_before,
    owning_list,
    inherit_loop_membership,
    record_member,
    remove_op,
    replace_op,
    request_loop_nodes,
    top_level_entry,
)


def _dep(name):
    return MemoryDep(name, sympy.Integer(0), (), ())


class _Op:
    """A stub op: a name, reads/writes, optional loop_member and layout."""

    def __init__(self, name, gid=None, counts=None, reads=(), writes=None):
        self._name = name
        if gid is not None:
            self.loop_member = LoopMember(
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
        ext.loop_member = LoopMember((0,), [4], [[]], [[]])
        ext.get_name.return_value = "ext"
        a = _Op("a", (0,), [4])
        out = _round_trip(self, [ext, a])
        self.assertIs(out[0], ext)
        self.assertEqual(out[1].body, [a])
        # Between members, it stays in the span like any other non-member. (Fresh
        # ops: the first collapse spent the lowering's record on ``a``.)
        a2, b2 = _Op("a2", (0,), [4]), _Op("b2", (0,), [4])
        (node,) = _round_trip(self, [a2, ext, b2])
        self.assertEqual(node.body, [a2, ext, b2])

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


class TestPendingLoopNodes(unittest.TestCase):
    """``request_loop_nodes`` / ``build_pending_loop_nodes``: a lowering records the
    loops it built mid-pipeline and the pass loop wraps them once the pass is over."""

    def test_only_the_requested_group_is_wrapped(self):
        u1 = sympy.Symbol("u1")
        a, b = _Op("a", (0,), [2]), _Op("b", (0,), [2])  # a flat loop nobody asked for
        c, d = _Op("c", (1,), [4]), _Op("d", (1,), [4])
        graph = _graph([a, b, c, d])
        request_loop_nodes(graph, {1: u1})
        build_pending_loop_nodes(graph)
        first, node = graph.operations[:2], graph.operations[2]
        self.assertEqual(first, [a, b])
        self.assertIsInstance(node, LoopNode)
        self.assertEqual((node.body, node.induction, node.trip_count), ([c, d], u1, 4))

    def test_an_existing_node_is_kept_and_the_new_one_is_numbered_after_it(self):
        u0, u1 = sympy.Symbol("u0"), sympy.Symbol("u1")
        a = _Op("a", (0,), [2])
        graph = _graph([a])
        collapse(graph, inductions={0: u0})
        (existing,) = graph.operations
        self.assertEqual(existing.get_name(), "loopnode0")
        c, d = _Op("c", (1,), [4]), _Op("d", (1,), [4])
        graph.operations.extend([c, d])
        request_loop_nodes(graph, {1: u1})
        build_pending_loop_nodes(graph)
        self.assertIs(graph.operations[0], existing)
        self.assertEqual(
            [o.get_name() for o in graph.operations], ["loopnode0", "loopnode1"]
        )
        self.assertEqual(graph.operations[1].body, [c, d])

    def test_a_later_build_keeps_an_earlier_nodes_strict_flag(self):
        """Two passes that tile build their nodes at different times (span-overflow
        tiling, then the allocator's). The second build must not read the first
        node's leaves, whose records were spent, as "not strict"."""
        a, b = _Op("a", (0,), [2]), _Op("b", (0,), [2])
        a.loop_member.strict_loop_vars = b.loop_member.strict_loop_vars = True
        c, d = _Op("c", (1,), [4]), _Op("d", (1,), [4])
        c.loop_member.strict_loop_vars = d.loop_member.strict_loop_vars = True
        graph = _graph([a, b])
        request_loop_nodes(graph, {0: sympy.Symbol("u0")})
        build_pending_loop_nodes(graph)
        first = graph.operations[0]
        self.assertTrue(first.strict_loop_vars)
        graph.operations.extend([c, d])
        request_loop_nodes(graph, {1: sympy.Symbol("u1")})
        build_pending_loop_nodes(graph)
        second = graph.operations[1]
        self.assertIsNot(first, second)
        self.assertTrue(first.strict_loop_vars)
        self.assertTrue(second.strict_loop_vars)
        self.assertFalse(hasattr(a, "loop_member"))

    def test_a_nested_strict_loop_gives_the_inner_node_its_flag_too(self):
        """The outer node's records are not deleted before the inner node reads
        its own flag."""
        a, b = _Op("a", (0, 0), [2, 3]), _Op("b", (0, 0), [2, 3])
        a.loop_member.strict_loop_vars = b.loop_member.strict_loop_vars = True
        graph = _graph([a, b])
        collapse(graph)
        (outer,) = graph.operations
        (inner,) = outer.body
        self.assertTrue(outer.strict_loop_vars)
        self.assertTrue(inner.strict_loop_vars)

    def test_building_twice_changes_nothing(self):
        c, d = _Op("c", (1,), [4]), _Op("d", (1,), [4])
        graph = _graph([c, d])
        request_loop_nodes(graph, {1: sympy.Symbol("u1")})
        build_pending_loop_nodes(graph)
        built = list(graph.operations)
        build_pending_loop_nodes(graph)
        self.assertEqual(graph.operations, built)

    def test_requests_accumulate_until_built(self):
        a, b = _Op("a", (0,), [2]), _Op("b", (1,), [3])
        graph = _graph([a, b])
        request_loop_nodes(graph, {0: sympy.Symbol("u0")})
        request_loop_nodes(graph, {1: sympy.Symbol("u1")})
        build_pending_loop_nodes(graph)
        self.assertEqual([type(o) for o in graph.operations], [LoopNode, LoopNode])

    def test_an_empty_request_is_not_recorded(self):
        graph = _graph([_Op("a")])
        request_loop_nodes(graph, {})
        build_pending_loop_nodes(graph)
        self.assertEqual(len(graph.operations), 1)

    def test_the_flat_view_is_refused(self):
        graph = _graph([_Op("c", (1,), [4])])
        request_loop_nodes(graph, {1: sympy.Symbol("u1")})
        graph.operations = OpsView(graph.operations)
        with self.assertRaises(AssertionError):
            build_pending_loop_nodes(graph)


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

    def test_the_list_mutators_refuse_and_change_nothing(self):
        """``insert`` / ``remove`` / item assignment / ``del`` place an op by its flat
        neighbour, which cannot say whether it belongs in a body or beside the node.
        They raise, and the graph is untouched."""
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        x = _Op("x")
        edits = {
            "insert": lambda: view.insert(view.index(d), x),
            "insert at end": lambda: view.insert(len(view), x),
            "remove": lambda: view.remove(c),
            "setitem": lambda: view.__setitem__(view.index(d), x),
            "slice assignment": lambda: view.__setitem__(slice(None), list(view)),
            "delitem": lambda: view.__delitem__(view.index(e)),
            "pop": lambda: view.pop(),
            "reverse": lambda: view.reverse(),
            "clear": lambda: view.clear(),
        }
        for what, edit in edits.items():
            with self.subTest(what), self.assertRaisesRegex(TypeError, "read-only"):
                edit()
        self.assertEqual(list(view), [a, b, c, d, e, f])
        self.assertEqual(real, [a, outer, f])
        self.assertEqual(outer.body, [b, inner, e])
        self.assertEqual(inner.body, [c, d])

    def test_helpers_edit_through_the_view_and_the_view_follows(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        self.assertEqual(len(view), 6)
        x, y = _Op("x"), _Op("y")
        insert_before(view, d, x)
        self.assertEqual(inner.body, [c, x, d])
        self.assertEqual(list(view), [a, b, c, x, d, e, f])
        remove_op(view, c)
        self.assertEqual(inner.body, [x, d])
        replace_op(view, x, y)
        self.assertEqual(inner.body, [y, d])
        self.assertEqual(len(view), 6)
        view.append(x)
        self.assertIs(view[-1], x)

    def test_removing_the_visited_op_matches_a_list_iteration(self):
        # A pass that removes the op it is visiting skips the next one on a
        # list; iterating the view while editing through the helpers must too.
        real, *_ = _nested()
        flat = list(all_ops(real))
        seen_view, seen_list = [], []
        view = OpsView(real)
        for op in view:
            seen_view.append(op)
            if op.get_name() in ("b", "d"):
                remove_op(view, op)
        for op in flat:
            seen_list.append(op)
            if op.get_name() in ("b", "d"):
                flat.remove(op)
        self.assertEqual(seen_view, seen_list)
        self.assertEqual(list(view), flat)


class TestOpsViewCrossingNodes(unittest.TestCase):
    """The two view edits that cross a node boundary on purpose."""

    def test_hoist_to_front_takes_ops_out_of_every_level(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        OpsView(real).hoist_to_front([d, b])
        # d left two levels, b one; they head the top-level list in the order given
        self.assertEqual(real[:2], [d, b])
        self.assertEqual(inner.body, [c])
        self.assertEqual(outer.body, [inner, e])
        self.assertEqual(real[2:], [a, outer, f])

    def test_hoist_to_front_drops_the_cached_flat_order(self):
        real, _, _, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        self.assertEqual(list(view), [a, b, c, d, e, f])  # fills the cache
        view.hoist_to_front([d])
        self.assertEqual(list(view), [d, a, b, c, e, f])
        self.assertEqual(list(view), list(all_ops(real)))

    def test_hoisting_every_op_leaves_an_empty_node_in_place(self):
        """The pass can take a body's only members out. The node is kept, empty:
        nothing here decides what an empty node means."""
        x, y = _Op("x"), _Op("y")
        node = _node("loop", [x, y])
        real = [node]
        OpsView(real).hoist_to_front([x, y])
        self.assertEqual(real, [x, y, node])
        self.assertEqual(node.body, [])

    def test_hoist_to_front_of_an_op_not_in_the_graph_raises(self):
        real, *_ = _nested()
        with self.assertRaises(ValueError):
            OpsView(real).hoist_to_front([_Op("stranger")])

    def test_insert_after_through_the_view_stays_beside_the_anchor(self):
        """After the last op of a body the op still belongs in the body, next to its
        anchor, not at the top level."""
        x, y, z = _Op("x"), _Op("y"), _Op("z")
        node = _node("loop", [x, y])
        real = [node, z]
        view = OpsView(real)
        w = _Op("w")
        insert_after(view, y, w)  # y is the last op of the body
        self.assertEqual(node.body, [x, y, w])
        self.assertEqual(real, [node, z])
        self.assertEqual(list(view), [x, y, w, z])  # and the cache was dropped

    def test_insert_after_a_node_is_outside_it(self):
        x, z = _Op("x"), _Op("z")
        node = _node("loop", [x])
        real = [node, z]
        w = _Op("w")
        insert_after(OpsView(real), node, w)
        self.assertEqual(real, [node, w, z])


class TestExplicitEdits(unittest.TestCase):
    """Edits that say where the op belongs (stage 8.0).

    A flat position cannot say whether "just before the loop's first member" means
    inside the body or beside the node. These say it explicitly: an op placed before
    a node is outside it, an op placed before a body op is in that body.
    """

    def test_insert_before_a_body_op_joins_its_body_through_the_view(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        self.assertEqual(list(view), [a, b, c, d, e, f])  # fills the cache
        x = _Op("x")
        insert_before(view, d, x)
        self.assertEqual(inner.body, [c, x, d])
        self.assertEqual(list(view), [a, b, c, x, d, e, f])  # the cache was dropped

    def test_insert_before_a_node_is_outside_it_even_when_its_first_member_is_next(
        self,
    ):
        """``c`` is the first flat op of ``inner``. Naming ``c`` puts the op inside
        the body; naming the node puts it before the loop, where it runs once."""
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        inside, outside = _Op("inside"), _Op("outside")
        insert_before(view, c, inside)
        self.assertEqual(inner.body, [inside, c, d])
        insert_before(view, inner, outside)
        self.assertEqual(outer.body, [b, outside, inner, e])
        self.assertEqual(real, [a, outer, f])

    def test_insert_before_a_top_level_node_goes_to_the_top_level(self):
        real, outer, _, (a, b, c, d, e, f) = _nested()
        x = _Op("x")
        insert_before(OpsView(real), outer, x)
        self.assertEqual(real, [a, x, outer, f])

    def test_insert_before_an_op_that_is_not_there_raises(self):
        real, *_ = _nested()
        with self.assertRaises(ValueError):
            insert_before(OpsView(real), _Op("stranger"), _Op("x"))

    def test_move_before_takes_an_op_out_of_its_list_and_into_the_anchors(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        move_before(view, d, f)  # a top-level op into the inner body
        self.assertEqual(inner.body, [c, f, d])
        self.assertEqual(real, [a, outer])
        self.assertEqual(list(view), [a, b, c, f, d, e])

    def test_move_before_a_node_takes_a_body_op_out_of_the_loop(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        move_before(OpsView(real), outer, d)
        self.assertEqual(real, [a, d, outer, f])
        self.assertEqual(inner.body, [c])

    def test_move_before_within_one_list_reorders_it(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        move_before(OpsView(real), c, d)
        self.assertEqual(inner.body, [d, c])

    def test_move_after_a_node_puts_the_op_after_the_loop(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        move_after(OpsView(real), inner, b)
        self.assertEqual(outer.body, [inner, b, e])

    def test_move_after_the_last_body_op_stays_in_the_body(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        move_after(OpsView(real), d, a)
        self.assertEqual(inner.body, [c, d, a])
        self.assertEqual(real, [outer, f])

    def test_moving_an_op_relative_to_itself_raises(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        with self.assertRaises(ValueError):
            move_before(OpsView(real), c, c)
        self.assertEqual(inner.body, [c, d])  # and the op was not taken out

    def test_move_of_an_op_that_is_not_there_raises_and_changes_nothing(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        with self.assertRaises(ValueError):
            move_before(OpsView(real), c, _Op("stranger"))
        self.assertEqual(inner.body, [c, d])

    def test_a_node_cannot_be_moved_relative_to_an_op_it_holds(self):
        """Detaching the node would detach the anchor: it raises before changing
        anything, and the node is still where it was."""
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        for move in (move_before, move_after):
            with self.assertRaises(ValueError):
                move(OpsView(real), c, outer)  # c is in a node nested in outer
            with self.assertRaises(ValueError):
                move(OpsView(real), inner, outer)
            self.assertEqual(real, [a, outer, f])
            self.assertEqual(outer.body, [b, inner, e])
            self.assertEqual(inner.body, [c, d])

    def test_a_node_moves_with_its_body_relative_to_an_outside_anchor(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        move_before(OpsView(real), a, inner)
        self.assertEqual(real, [inner, a, outer, f])
        self.assertEqual(inner.body, [c, d])
        self.assertEqual(outer.body, [b, e])

    def test_move_to_an_anchor_that_is_not_there_raises_and_changes_nothing(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        with self.assertRaises(ValueError):
            move_before(OpsView(real), _Op("stranger"), d)
        self.assertEqual(inner.body, [c, d])  # d is still where it was

    def test_remove_op_takes_the_op_from_the_list_that_holds_it(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        remove_op(view, d)
        self.assertEqual(inner.body, [c])
        self.assertEqual(list(view), [a, b, c, e, f])

    def test_remove_op_works_on_the_real_list_and_on_a_node(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        remove_op(real, inner)
        self.assertEqual(outer.body, [b, e])

    def test_removing_the_last_member_leaves_an_empty_body(self):
        """Nothing here decides what an empty node means (see the hoist test)."""
        x = _Op("x")
        node = _node("loop", [x])
        real = [node]
        remove_op(real, x)
        self.assertEqual(node.body, [])
        self.assertEqual(real, [node])

    def test_remove_op_of_an_op_that_is_not_there_raises(self):
        real, *_ = _nested()
        with self.assertRaises(ValueError):
            remove_op(OpsView(real), _Op("stranger"))


class TestTopLevelEntry(unittest.TestCase):
    """``top_level_entry``: the op, or the outermost node that holds it. A clone that
    must run once, outside every loop of its anchor, is placed beside this."""

    def test_a_top_level_op_is_its_own_entry(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        self.assertIs(top_level_entry(OpsView(real), a), a)
        self.assertIs(top_level_entry(real, f), f)

    def test_a_body_op_has_the_outermost_node_as_its_entry(self):
        real, outer, inner, (a, b, c, d, e, f) = _nested()
        view = OpsView(real)
        self.assertIs(top_level_entry(view, b), outer)  # one level down
        self.assertIs(top_level_entry(view, d), outer)  # two levels down

    def test_a_node_is_its_own_entry_at_the_top_level_and_nested(self):
        real, outer, inner, _ = _nested()
        self.assertIs(top_level_entry(real, outer), outer)
        self.assertIs(top_level_entry(real, inner), outer)

    def test_an_op_that_is_not_there_raises(self):
        real, *_ = _nested()
        with self.assertRaises(ValueError):
            top_level_entry(OpsView(real), _Op("stranger"))


class TestNodeFacts(unittest.TestCase):
    """Once the nodes are built they hold the loop facts: the group id, trip count and
    each member's tiled dims, and each leaf op points at its enclosing nodes. The
    lowering's ``loop_member`` record is spent and removed from the ops they hold."""

    def _tiled_pair(self):
        a, b = _Op("a", (0,), [4]), _Op("b", (0,), [4])
        a.loop_member.loop_tiled_dims = [[1]]
        b.loop_member.loop_tiled_reduction_dims = [[0]]
        graph = _graph([_Op("pre"), a, b, _Op("post")])
        collapse(graph)
        node = next(o for o in graph.operations if isinstance(o, LoopNode))
        return graph, node, a, b

    @staticmethod
    def _facts(levels):
        return [
            (x.group_path, x.trip_count, x.tiled_dims, x.tiled_reduction_dims)
            for x in levels
        ]

    def test_collapse_records_group_id_tiling_and_chain_and_spends_the_record(self):
        graph, node, a, b = self._tiled_pair()
        self.assertEqual(node.group_id, 0)
        self.assertEqual(node.tiling, {"a": ((1,), ()), "b": ((), (0,))})
        self.assertEqual(a._loop_chain, (node,))
        self.assertTrue(node.facts_recorded)
        self.assertEqual(self._facts(enclosing_loops(a)), [((0,), 4, (1,), ())])
        self.assertEqual(self._facts(enclosing_loops(b)), [((0,), 4, (), (0,))])
        self.assertEqual(enclosing_loops(a), node_levels(a))
        self.assertEqual(counted_loop_path(a), (0,))
        for op in (a, b):
            self.assertFalse(hasattr(op, "loop_member"))
            self.assertEqual(pending_levels(op), ())

    def test_an_op_no_node_holds_answers_from_its_pending_record(self):
        a = _Op("a", (0,), [4])
        a.loop_member.loop_tiled_dims = [[1]]
        self.assertEqual(node_levels(a), ())
        (level,) = enclosing_loops(a)
        self.assertEqual((level.group_path, level.tiled_dims), ((0,), (1,)))
        self.assertEqual(counted_loop_path(a), (0,))

    def test_an_op_moved_out_of_its_loop_is_no_longer_looped(self):
        graph, node, a, b = self._tiled_pair()
        move_after(graph.operations, node, a)
        self.assertEqual(a._loop_chain, ())
        self.assertEqual(node_levels(a), ())
        self.assertEqual(enclosing_loops(a), ())
        self.assertFalse(is_looped(a))

    def test_nested_chain_is_outermost_first(self):
        b = _Op("b", (0,), [4])
        c = _Op("c", (0, 0), [4, 2])
        c.loop_member.loop_tiled_dims = [[], [2]]
        graph = _graph([b, c])
        collapse(graph)
        (outer,) = graph.operations
        inner = outer.body[1]
        self.assertEqual(c._loop_chain, (outer, inner))
        self.assertEqual(b._loop_chain, (outer,))
        self.assertEqual(inner.tiling["c"], ((2,), ()))
        self.assertEqual(outer.tiling["c"], ((), ()))
        self.assertEqual([level.group_path for level in node_levels(c)], [(0,), (0, 0)])

    def test_an_op_placed_in_a_body_joins_its_chain_and_a_stale_record_is_dropped(
        self,
    ):
        graph, node, a, b = self._tiled_pair()
        new = _Op("new", (0,), [4])
        new.loop_member.loop_tiled_dims = [[0]]
        insert_before(graph.operations, b, new)
        self.assertEqual(new._loop_chain, (node,))
        self.assertEqual(node.tiling["new"], ((0,), ()))
        self.assertFalse(hasattr(new, "loop_member"))
        out = _Op("out")
        insert_after(graph.operations, node, out)
        self.assertEqual(out._loop_chain, ())

    def test_moving_an_op_out_of_a_body_clears_its_chain(self):
        graph, node, a, b = self._tiled_pair()
        move_after(graph.operations, node, a)
        self.assertEqual(a._loop_chain, ())
        self.assertEqual(node_levels(a), ())

    def test_record_member_makes_a_placed_op_a_member(self):
        graph, node, a, b = self._tiled_pair()
        late = _Op("late")
        insert_before(graph.operations, b, late)
        self.assertNotIn("late", node.tiling)
        self.assertFalse(is_looped(late))
        record_member(late, [([2], [])])
        self.assertEqual(node.tiling["late"], ((2,), ()))
        self.assertTrue(is_looped(late))
        self.assertEqual(self._facts(enclosing_loops(late)), [((0,), 4, (2,), ())])

    def test_record_member_needs_one_entry_per_node(self):
        graph, node, a, b = self._tiled_pair()
        with self.assertRaises(ValueError):
            record_member(a, [([], []), ([], [])])
        with self.assertRaises(ValueError):
            record_member(graph.operations[0], [([], [])])  # "pre": in no node

    def test_an_unplaced_member_op_inherits_its_stand_in_s_nodes(self):
        graph, node, a, b = self._tiled_pair()
        probe = _Op("a")
        inherit_loop_membership(a, probe)
        self.assertEqual(probe._loop_chain, (node,))
        self.assertTrue(is_looped(probe))

    def test_a_copy_of_a_member_placed_in_its_nodes_tiles_what_it_tiles(self):
        """``copy_op_metadata`` marks a copy with its donor; placing the copy in the
        donor's nodes makes it a member (an allocator clone beside its consumer)."""
        from torch_spyre._inductor.pass_utils import copy_op_metadata

        graph, node, a, b = self._tiled_pair()
        clone = _Op("clone")
        copy_op_metadata(a, clone)
        insert_before(graph.operations, b, clone)
        self.assertEqual(node.tiling["clone"], node.tiling["a"])
        self.assertTrue(is_looped(clone))
        self.assertFalse(hasattr(clone, "_member_like"))

    def test_a_copy_of_a_non_member_stays_a_non_member(self):
        from torch_spyre._inductor.pass_utils import copy_op_metadata

        graph, node, a, b = self._tiled_pair()
        pre = graph.operations[0]
        clone = _Op("clone")
        copy_op_metadata(pre, clone)
        insert_before(graph.operations, b, clone)
        self.assertNotIn("clone", node.tiling)
        self.assertFalse(is_looped(clone))

    def test_an_allocator_drain_or_hoisted_clone_is_not_a_member(self):
        """The allocator clears the donor marker for a clone that is neither a member
        nor a carry (``_clear_loop_membership_metadata``), so a copy of a member placed
        in the body still reads as not looped."""
        from torch_spyre._inductor.pass_utils import copy_op_metadata
        from torch_spyre._inductor.scratchpad.allocator import (
            _clear_loop_membership_metadata,
        )

        graph, node, a, b = self._tiled_pair()
        clone = _Op("clone")
        copy_op_metadata(a, clone)
        _clear_loop_membership_metadata(clone)
        insert_before(graph.operations, b, clone)
        self.assertNotIn("clone", node.tiling)
        self.assertFalse(is_looped(clone))

    def test_a_probe_that_inherits_membership_drops_its_donor_marker(self):
        from torch_spyre._inductor.pass_utils import copy_op_metadata

        graph, node, a, b = self._tiled_pair()
        probe = _Op("a")
        copy_op_metadata(a, probe)
        self.assertTrue(hasattr(probe, "_member_like"))
        inherit_loop_membership(a, probe)
        self.assertFalse(hasattr(probe, "_member_like"))
        self.assertTrue(is_looped(probe))

    def test_a_loop_that_mixes_the_tiling_lowering_with_another_is_refused(self):
        a, b = _Op("a", (0,), [4]), _Op("b", (0,), [4])
        a.loop_member.strict_loop_vars = True
        with self.assertRaisesRegex(RuntimeError, "holds ops of the loop-tiling"):
            collapse(_graph([a, b]))

    def test_a_loop_the_tiling_lowering_made_is_strict_as_a_whole(self):
        a, b = _Op("a", (0,), [4]), _Op("b", (0,), [4])
        a.loop_member.strict_loop_vars = b.loop_member.strict_loop_vars = True
        graph = _graph([a, b])
        collapse(graph)
        (node,) = graph.operations
        self.assertTrue(node.strict_loop_vars)

    def test_an_unstamped_op_in_a_body_is_not_a_member(self):
        """Being placed in a body is not being a member of the loop: an LX clone
        beside a body consumer runs every trip but tiles nothing and has no entry, so
        it answers "not looped" (counting it as looped made the same-tile-group
        relayout offer a wrong candidate for it)."""
        graph, node, a, b = self._tiled_pair()
        once = _Op("once")
        insert_before(graph.operations, b, once)
        self.assertEqual(once._loop_chain, (node,))
        self.assertEqual(len(node_levels(once)), 1)
        self.assertEqual(enclosing_loops(once), ())
        self.assertFalse(is_looped(once))
        self.assertTrue(is_looped(a))

    def test_replace_op_takes_the_new_ops_tiling(self):
        graph, node, a, b = self._tiled_pair()
        new = _Op("a", (0,), [4])
        new.loop_member.loop_tiled_dims = [[0, 2]]
        replace_op(graph.operations, a, new)
        self.assertEqual(new._loop_chain, (node,))
        self.assertEqual(node.tiling["a"], ((0, 2), ()))
        self.assertFalse(hasattr(new, "loop_member"))

    def test_remove_op_forgets_the_op_but_a_move_keeps_it(self):
        graph, node, a, b = self._tiled_pair()
        move_before(graph.operations, a, b)
        self.assertEqual(node.tiling["b"], ((), (0,)))
        self.assertEqual(b._loop_chain, (node,))
        remove_op(graph.operations, a)
        self.assertNotIn("a", node.tiling)
        self.assertEqual(a._loop_chain, ())
        self.assertEqual(node.tiling["b"], ((), (0,)))

    def test_hoist_to_front_clears_the_chain(self):
        graph, node, a, b = self._tiled_pair()
        OpsView(graph.operations).hoist_to_front([a])
        self.assertEqual(a._loop_chain, ())


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

    def test_a_dead_leaf_is_removed_from_its_body_through_the_flat_view(self):
        """As the pass runs it: ``graph.operations`` is the flat view, and a dead op
        inside a body leaves that body, not the top level."""
        keep = self._buf("keep")
        live = self._buf("live")
        dead = self._buf("dead")
        node = _node("loopnode0", [live, dead])
        real = [node, keep]
        self._run(OpsView(real), output_names=["keep", "live"])
        self.assertEqual(node.body, [live])
        self.assertEqual(real, [node, keep])
        self.assertIn("dead", V.graph.removed_buffers)

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
