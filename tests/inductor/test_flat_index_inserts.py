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

"""Where an op inserted "after" an anchor lands, relative to a loop.

A pass handed the flat :class:`OpsView` that inserts at ``index(anchor) + k`` puts the
op before whatever follows the anchor, in the list that holds THAT op. When the anchor
is the last op of a loop body the follower is outside the loop, so the op lands outside
it: the bug #6101 fixed in ``reorder_nonstick_dims_mutation`` (a scatter's copy-back ran
once, after the loop, while carrying the loop's ``loop_group_id``).

This file records an audit of the other flat-index "insert after" sites:

* ``insert_restickify._bracket_mutation`` (the copy-back after a graph-input slice
  mutation) used ``index(mutation) + 2``. It now uses ``insert_after``. No compiled graph
  reaches it with a loop member today (a ``for_each_tile`` body cannot mutate a graph
  input: Dynamo refuses input mutation in a higher-order op, and a carry initialised
  from a graph input is copied to an internal buffer, which takes a branch that inserts
  nothing), so its placement is pinned here on a real tree.
* ``GraphEditor.push_allocation_with_clone`` with a ``lower_anchor`` (the post-loop
  drain) inserts at ``index(anchor) + 1`` on purpose: the anchor is the loop's last
  member and the clone belongs after the loop. That is pinned here so it is not
  "fixed".
* ``lx_context_switching``'s restore (``index(target_op) + 1``) is not changed. Its target
  is an extern kernel; the loop scheduler refuses one that reads a value the loop writes
  and hoists one that does not, so it is not the last op of a body. Not covered here.
"""

from types import SimpleNamespace

import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor.insert_restickify import _bracket_mutation
from torch_spyre._inductor.loop_node import LoopNode, OpsView, insert_after


class _Op:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def _tree(body_names=("first", "second", "last")):
    body = [_Op(n) for n in body_names]
    node = LoopNode("loop", body, induction=sympy.Symbol("u0"), trip_count=4)
    before, after = _Op("before_loop"), _Op("after_loop")
    graph = SimpleNamespace(operations=[before, node, after])
    return graph, node, body, before, after


def _where(graph, node, op):
    if op in node.body:
        return "inside the loop"
    if op in graph.operations:
        return "top level"
    return "nowhere"


class TestTheFlatIndexRule:
    def test_after_the_last_body_op_a_flat_index_lands_outside_the_loop(self):
        graph, node, body, *_ = _tree()
        view = OpsView(graph.operations)
        new = _Op("new")
        view.insert(list(view).index(body[-1]) + 1, new)
        assert _where(graph, node, new) == "top level"

    def test_after_a_middle_body_op_a_flat_index_lands_inside(self):
        graph, node, body, *_ = _tree()
        view = OpsView(graph.operations)
        new = _Op("new")
        view.insert(list(view).index(body[1]) + 1, new)
        assert _where(graph, node, new) == "inside the loop"

    def test_insert_after_keeps_the_op_next_to_its_anchor(self):
        graph, node, body, *_ = _tree()
        view = OpsView(graph.operations)
        new = _Op("new")
        insert_after(view, body[-1], new)
        assert _where(graph, node, new) == "inside the loop"
        assert node.body[-1] is new


class TestBracketMutation:
    """``insert_restickify._bracket_mutation``: buf_tmp before the mutation op, the
    copy-back after it, each in the list that holds the mutation."""

    def _run(self, graph, mutation):
        tmp, copyback = _Op("buf_tmp"), _Op("buf_copyback")
        # _create_restickify_node appends both at the end of the operations
        graph.operations.extend([tmp, copyback])
        _bracket_mutation(OpsView(graph.operations), mutation, tmp, copyback)
        return tmp, copyback

    def test_a_mutation_that_is_the_last_op_of_a_loop_body_keeps_both_inside(self):
        graph, node, body, *_ = _tree()
        tmp, copyback = self._run(graph, body[-1])
        assert _where(graph, node, tmp) == "inside the loop"
        assert _where(graph, node, copyback) == "inside the loop"
        assert node.body[-3:] == [tmp, body[-1], copyback]

    def test_a_mutation_in_the_middle_of_a_body_keeps_both_inside_and_in_order(self):
        graph, node, body, *_ = _tree()
        tmp, copyback = self._run(graph, body[1])
        assert [o.get_name() for o in node.body] == [
            "first",
            "buf_tmp",
            "second",
            "buf_copyback",
            "last",
        ]

    def test_a_mutation_outside_any_loop_is_bracketed_at_the_top_level(self):
        graph, node, body, before, after = _tree()
        tmp, copyback = self._run(graph, after)
        assert graph.operations == [before, node, tmp, after, copyback]

    def test_a_flat_list_is_bracketed_too(self):
        mutation = _Op("mutation")
        operations = [_Op("a"), mutation, _Op("b")]
        tmp, copyback = _Op("buf_tmp"), _Op("buf_copyback")
        operations.extend([tmp, copyback])
        _bracket_mutation(operations, mutation, tmp, copyback)
        assert [o.get_name() for o in operations] == [
            "a",
            "buf_tmp",
            "mutation",
            "buf_copyback",
            "b",
        ]


class TestThePostLoopDrain:
    """The allocator's clone with a ``lower_anchor`` is meant to land AFTER the loop:
    the anchor is the loop's last member. Inserting at ``index(anchor) + 1`` through the
    flat view does that, by the rule above. Pinned so it is not changed by mistake."""

    def test_a_clone_after_the_loops_last_member_lands_after_the_loop(self):
        graph, node, body, before, after = _tree()
        view = OpsView(graph.operations)
        clone = _Op("drain")
        view.insert(list(view).index(body[-1]) + 1, clone)
        assert _where(graph, node, clone) == "top level"
        assert graph.operations == [before, node, clone, after]
