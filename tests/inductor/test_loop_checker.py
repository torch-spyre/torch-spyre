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

"""Tests for ``loop_checker`` and the ``loop_node`` insertion points."""

import os
import unittest
import unittest.mock
from types import SimpleNamespace

import sympy
import torch
from torch._inductor.dependencies import MemoryDep, ReadWrites, StarDep
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor import loop_checker
from torch_spyre._inductor.loop_checker import (
    LoopStructureError,
    check_loops,
    find_violations,
)
from torch_spyre._inductor.loop_info import LoopCarryRecord
from torch_spyre._inductor.loop_node import (
    LoopNode,
    all_ops,
    building_loop,
    insert_after,
    insert_at_start,
    insert_before,
    insert_on_back_edge,
)

U0, U1 = sympy.symbols("u0 u1")


class _Op:
    """A stub op: a name and the index expression of one read and one write."""

    def __init__(self, name, index=sympy.Integer(0), reads=()):
        self._name = name
        self._index = index
        self._reads = list(reads)

    def get_name(self):
        return self._name

    def get_device(self):
        return torch.device("cpu")

    def get_outputs(self):
        return [self]

    def get_read_writes(self):
        reads = OrderedSet(MemoryDep(r, self._index, (), ()) for r in self._reads)
        writes = OrderedSet([MemoryDep(self._name, self._index, (), ())])
        return ReadWrites(reads, writes, OrderedSet())

    def __repr__(self):
        return f"_Op({self._name})"


class _StarOp(_Op):
    """An op whose only dependency is a StarDep, which has no index."""

    def get_read_writes(self):
        return ReadWrites(OrderedSet(), OrderedSet([StarDep(self._name)]), OrderedSet())


def _node(name, body, induction=U0, carries=()):
    return LoopNode(
        name, body, induction=induction, trip_count=4, carries=tuple(carries)
    )


def _graph(ops):
    return SimpleNamespace(operations=list(ops))


class TestFindViolations(unittest.TestCase):
    def test_a_well_formed_tree_has_none(self):
        outer = _node("outer", [_Op("a", U0), _node("inner", [_Op("b", U0 + U1)], U1)])
        self.assertEqual(find_violations([_Op("pre"), outer]), [])

    def test_an_op_in_two_bodies_is_reported(self):
        shared = _Op("shared")
        problems = find_violations([_node("n0", [shared]), _node("n1", [shared])])
        self.assertEqual(len(problems), 1)
        self.assertIn("shared appears more than once", problems[0])

    def test_an_index_using_another_loops_variable_is_reported(self):
        # u1 belongs to the second loop; "a" is in the first.
        problems = find_violations(
            [_node("n0", [_Op("a", U1)], U0), _node("n1", [_Op("b", U1)], U1)]
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("a indexes with ['u1']", problems[0])

    def test_an_outer_loops_variable_in_an_inner_body_is_allowed(self):
        inner_uses_outer = _node("outer", [_node("inner", [_Op("a", U0)], U1)], U0)
        self.assertEqual(find_violations([inner_uses_outer]), [])

    def test_an_op_outside_every_loop_using_a_loop_variable_is_reported(self):
        problems = find_violations([_Op("loose", U0), _node("n0", [_Op("a", U0)])])
        self.assertEqual(len(problems), 1)
        self.assertIn("loose indexes with ['u0']", problems[0])

    def test_a_stardep_has_no_index_and_is_ignored(self):
        self.assertEqual(find_violations([_node("n0", [_StarOp("s")])]), [])

    def test_a_carry_whose_update_is_not_in_the_body_is_reported(self):
        carry = LoopCarryRecord(storage_name="acc", update_name="gone")
        problems = find_violations([_node("n0", [_Op("a")], carries=[carry])])
        self.assertEqual(len(problems), 1)
        self.assertIn("carry update 'gone' is not an op in its body", problems[0])

    def test_a_carry_whose_storage_is_outside_the_body_is_allowed(self):
        # The storage is the carry's initial buffer, written before the loop.
        carry = LoopCarryRecord(storage_name="acc", update_name="a")
        tree = [_Op("acc"), _node("n0", [_Op("a")], carries=[carry])]
        self.assertEqual(find_violations(tree), [])


class TestCheckLoops(unittest.TestCase):
    def setUp(self):
        self.enabled_patch = unittest.mock.patch.object(
            loop_checker, "checking_enabled", return_value=True
        )
        self.enabled_patch.start()
        self.addCleanup(self.enabled_patch.stop)

    def bad_graph(self):
        # "a" is in the u0 loop but indexes with u1, which the second loop owns.
        return _graph([_node("n0", [_Op("a", U1)], U0), _node("n1", [_Op("b")], U1)])

    def test_it_raises_and_names_the_pass(self):
        with self.assertRaisesRegex(
            LoopStructureError, r"(?s)after my_pass: .*a indexes with \['u1'\]"
        ):
            check_loops(self.bad_graph(), where="my_pass")

    def test_building_a_loop_suspends_it(self):
        with building_loop():
            check_loops(self.bad_graph(), where="my_pass")

    def test_it_is_off_when_disabled(self):
        with unittest.mock.patch.object(
            loop_checker, "checking_enabled", return_value=False
        ):
            check_loops(self.bad_graph(), where="my_pass")

    def test_a_graph_without_loops_is_not_checked(self):
        check_loops(_graph([_Op("a", U0)]), where="my_pass")

    def test_it_reads_the_real_list_behind_a_view(self):
        graph = self.bad_graph()
        graph.operations = SimpleNamespace(real=graph.operations)
        with self.assertRaises(LoopStructureError):
            check_loops(graph, where="my_pass")

    def test_the_default_follows_the_environment(self):
        # Undo setUp's patch for this test only.
        self.enabled_patch.stop()
        self.addCleanup(self.enabled_patch.start)
        with unittest.mock.patch.dict("os.environ", {"TORCH_SPYRE_CHECK_LOOPS": "0"}):
            self.assertFalse(loop_checker.checking_enabled())
        with unittest.mock.patch.dict("os.environ", {"TORCH_SPYRE_CHECK_LOOPS": "1"}):
            self.assertTrue(loop_checker.checking_enabled())
        with unittest.mock.patch.dict("os.environ"):
            os.environ.pop("TORCH_SPYRE_CHECK_LOOPS", None)
            self.assertTrue(loop_checker.checking_enabled())  # pytest is loaded


class TestInsertionPoints(unittest.TestCase):
    def setUp(self):
        self.a, self.b = _Op("a"), _Op("b")
        self.node = _node("loop", [self.a, self.b])
        self.pre = _Op("pre")
        self.ops = [self.pre, self.node]

    def names(self):
        return [o.get_name() for o in all_ops(self.ops)]

    def test_before_a_node_is_outside_the_loop(self):
        new = _Op("new")
        insert_before(self.ops, self.node, new)
        self.assertEqual([o.get_name() for o in self.ops], ["pre", "new", "loop"])
        self.assertNotIn(new, self.node.body)

    def test_after_a_node_is_outside_the_loop(self):
        new = _Op("new")
        insert_after(self.ops, self.node, new)
        self.assertEqual([o.get_name() for o in self.ops], ["pre", "loop", "new"])
        self.assertNotIn(new, self.node.body)

    def test_before_and_after_a_body_op_stay_in_the_body(self):
        before, after = _Op("before_b"), _Op("after_a")
        insert_before(self.ops, self.b, before)
        insert_after(self.ops, self.a, after)
        self.assertEqual(
            [o.get_name() for o in self.node.body], ["a", "after_a", "before_b", "b"]
        )

    def test_before_a_nested_node_goes_to_the_enclosing_body(self):
        inner = _node("inner", [_Op("x")], U1)
        outer = _node("outer", [self.a, inner])
        ops = [outer]
        new = _Op("new")
        insert_before(ops, inner, new)
        self.assertEqual([o.get_name() for o in outer.body], ["a", "new", "inner"])

    def test_start_of_each_iteration_is_the_first_body_entry(self):
        new = _Op("new")
        insert_at_start(self.node, new)
        self.assertEqual([o.get_name() for o in self.node.body], ["new", "a", "b"])

    def test_the_back_edge_is_the_last_body_entry(self):
        new = _Op("copy")
        insert_on_back_edge(self.node, new)
        self.assertEqual([o.get_name() for o in self.node.body], ["a", "b", "copy"])

    def test_the_back_edge_follows_every_reader_of_the_old_value(self):
        # b reads "acc" (the old value); the copy overwrites "acc".
        reader = _Op("b", reads=["acc"])
        node = _node("loop", [_Op("a"), reader])
        copy = _Op("acc")
        insert_on_back_edge(node, copy)
        order = [o.get_name() for o in node.body]
        self.assertGreater(order.index("acc"), order.index("b"))

    def test_an_anchor_not_in_the_list_raises(self):
        with self.assertRaisesRegex(ValueError, "is not in the operations list"):
            insert_before(self.ops, _Op("stranger"), _Op("new"))

    def test_an_inserted_op_passes_the_checker(self):
        for place in (
            lambda n: insert_before(self.ops, self.node, n),
            lambda n: insert_after(self.ops, self.node, n),
            lambda n: insert_at_start(self.node, n),
            lambda n: insert_on_back_edge(self.node, n),
        ):
            place(_Op(f"new{id(place)}"))
        self.assertEqual(find_violations(self.ops), [])


if __name__ == "__main__":
    unittest.main()
