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

"""Unit tests for CountedLoopSchedulerNode and build_loop_scheduler_nodes.

These tests construct minimal fake scheduler/node objects using unittest.mock
so they can run without a Spyre device or a full compilation pipeline.
"""

import unittest
from unittest.mock import MagicMock, patch
from sympy import Integer, Symbol

from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    build_loop_scheduler_nodes,
    _loop_group_id,
    _loop_count,
)


# ---------------------------------------------------------------------------
# Helpers to build fake scheduler nodes
# ---------------------------------------------------------------------------


def _make_scheduler():
    """Return a minimal fake Scheduler."""
    sched = MagicMock()
    sched.name_to_fused_node = {}
    sched.removed_ops = set()
    return sched


def _make_ir_op(loop_group_id=None, loop_count=None, name="op"):
    """Return a fake ir.Operation optionally stamped with loop attributes.

    loop_count must be a list of trip counts (one per nesting level), matching
    the contract stamped by coarse_tile().  A bare Expr is accepted as a
    convenience shorthand and is wrapped in a 1-element list.
    """
    op = MagicMock()
    op.name = name
    if loop_group_id is not None:
        op.loop_group_id = loop_group_id
        op.loop_count = loop_count if isinstance(loop_count, list) else [loop_count]
    else:
        # Make getattr(..., None) work correctly by not having the attribute.
        del op.loop_group_id
        del op.loop_count
    return op


def _make_snode(scheduler, ir_op, name="buf0"):
    """Return a fake SchedulerNode wrapping ir_op."""
    from torch._inductor.scheduler import SchedulerNode

    snode = MagicMock(spec=SchedulerNode)
    snode.scheduler = scheduler
    snode.node = ir_op
    snode.get_name.return_value = name
    snode.get_nodes.return_value = [snode]
    snode.ancestors = set()
    snode.min_order = 0
    snode.max_order = 0
    snode.read_writes = MagicMock()
    snode.read_writes.reads_and_writes.return_value = []
    snode.outputs_by_name = {}
    return snode


# ---------------------------------------------------------------------------
# Tests for _loop_group_id and _loop_count helpers
# ---------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    def test_loop_group_id_present(self):
        sched = _make_scheduler()
        op = _make_ir_op(loop_group_id=(0,), loop_count=Integer(4))
        snode = _make_snode(sched, op)
        self.assertEqual(_loop_group_id(snode), (0,))

    def test_loop_group_id_absent(self):
        sched = _make_scheduler()
        op = _make_ir_op()
        snode = _make_snode(sched, op)
        self.assertIsNone(_loop_group_id(snode))

    def test_loop_count(self):
        sched = _make_scheduler()
        op = _make_ir_op(loop_group_id=(0,), loop_count=Integer(8))
        snode = _make_snode(sched, op)
        self.assertEqual(_loop_count(snode, depth=0), Integer(8))

    def test_loop_count_symbolic(self):
        sched = _make_scheduler()
        s = Symbol("s0")
        op = _make_ir_op(loop_group_id=(0,), loop_count=s)
        snode = _make_snode(sched, op)
        self.assertEqual(_loop_count(snode, depth=0), s)


# ---------------------------------------------------------------------------
# Tests for build_loop_scheduler_nodes
# ---------------------------------------------------------------------------


class TestBuildLoopSchedulerNodes(unittest.TestCase):
    def _run(self, nodes):
        """Patch GroupedSchedulerNode.__init__ and create so we can call build."""
        # We need CountedLoopSchedulerNode.create to work, which calls
        # GroupedSchedulerNode.__init__ -> init_group_node.  Instead of
        # mocking the entire chain, we patch CountedLoopSchedulerNode.create
        # to return a lightweight stand-in that still carries loop_count and snodes.
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

        with patch.object(
            CountedLoopSchedulerNode, "create", staticmethod(fake_create)
        ):
            result = build_loop_scheduler_nodes(nodes)
        return result, created

    def test_passthrough_no_loop_group(self):
        sched = _make_scheduler()
        nodes = [
            _make_snode(sched, _make_ir_op(), "a"),
            _make_snode(sched, _make_ir_op(), "b"),
        ]
        result, created = self._run(nodes)
        self.assertEqual(result, nodes)
        self.assertEqual(created, [])

    def test_single_group_two_nodes(self):
        sched = _make_scheduler()
        n1 = _make_snode(sched, _make_ir_op((0,), Integer(4)), "a")
        n2 = _make_snode(sched, _make_ir_op((0,), Integer(4)), "b")
        result, created = self._run([n1, n2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].loop_count, Integer(4))
        self.assertIn(n1, created[0].snodes)
        self.assertIn(n2, created[0].snodes)

    def test_non_group_nodes_pass_through_around_group(self):
        sched = _make_scheduler()
        before = _make_snode(sched, _make_ir_op(), "before")
        g1 = _make_snode(sched, _make_ir_op((0,), Integer(2)), "g1")
        g2 = _make_snode(sched, _make_ir_op((0,), Integer(2)), "g2")
        after = _make_snode(sched, _make_ir_op(), "after")
        result, created = self._run([before, g1, g2, after])
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], before)
        self.assertIsInstance(result[1], MagicMock)  # the CountedLoop stand-in
        self.assertIs(result[2], after)
        self.assertEqual(created[0].loop_count, Integer(2))

    def test_two_separate_groups(self):
        sched = _make_scheduler()
        g0a = _make_snode(sched, _make_ir_op((0,), Integer(4)), "g0a")
        g0b = _make_snode(sched, _make_ir_op((0,), Integer(4)), "g0b")
        g1a = _make_snode(sched, _make_ir_op((1,), Integer(8)), "g1a")
        g1b = _make_snode(sched, _make_ir_op((1,), Integer(8)), "g1b")
        result, created = self._run([g0a, g0b, g1a, g1b])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].loop_count, Integer(4))
        self.assertEqual(created[1].loop_count, Integer(8))

    def test_nested_group(self):
        sched = _make_scheduler()
        # Outer group (0,) contains an inner group (0, 0).
        # loop_count for a depth-2 op is [outer_count, inner_count].
        outer = _make_snode(sched, _make_ir_op((0,), Integer(4)), "outer")
        inner1 = _make_snode(
            sched, _make_ir_op((0, 0), [Integer(4), Integer(2)]), "inner1"
        )
        inner2 = _make_snode(
            sched, _make_ir_op((0, 0), [Integer(4), Integer(2)]), "inner2"
        )
        result, created = self._run([outer, inner1, inner2])
        # Outermost result has one CountedLoopSchedulerNode for group (0,)
        self.assertEqual(len(result), 1)
        outer_loop = result[0]
        # The outer loop should contain: outer node + a nested CountedLoop
        self.assertEqual(len(outer_loop.snodes), 2)
        # The second snode in the outer loop is the inner CountedLoop
        inner_loop = outer_loop.snodes[1]
        self.assertEqual(inner_loop.loop_count, Integer(2))
        self.assertIn(inner1, inner_loop.snodes)
        self.assertIn(inner2, inner_loop.snodes)

    def test_inconsistent_loop_count_raises(self):
        sched = _make_scheduler()
        n1 = _make_snode(sched, _make_ir_op((0,), Integer(4)), "a")
        n2 = _make_snode(sched, _make_ir_op((0,), Integer(8)), "b")  # different count
        with self.assertRaises(AssertionError):
            self._run([n1, n2])

    def test_empty_list(self):
        result, created = self._run([])
        self.assertEqual(result, [])
        self.assertEqual(created, [])

    def test_symbolic_loop_count(self):
        sched = _make_scheduler()
        s = Symbol("K")
        n1 = _make_snode(sched, _make_ir_op((0,), s), "a")
        n2 = _make_snode(sched, _make_ir_op((0,), s), "b")
        result, created = self._run([n1, n2])
        self.assertEqual(len(result), 1)
        self.assertEqual(created[0].loop_count, s)


if __name__ == "__main__":
    unittest.main()
