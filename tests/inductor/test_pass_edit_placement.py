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

"""Where the restickify and padding passes put the ops they relocate (stage 8.3).

``_bracket_mutation`` (insert_restickify) and ``_move_ops_before`` (padding) take
freshly lowered ops, which ``register_operation`` appended at the top level, and put
them beside an anchor: in that anchor's loop body when it is a body op, before the
node when it is a node. Stub operations, no compile, no device.
"""

import unittest

import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor.insert_restickify import _bracket_mutation
from torch_spyre._inductor.loop_node import LoopNode, OpsView
from torch_spyre._inductor.padding import _move_ops_before


class _Op:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_Op({self.name})"


def _node(name, body):
    return LoopNode(name, body, induction=None, trip_count=4)


class TestMoveOpsBefore(unittest.TestCase):
    def test_pad_ops_before_the_first_member_of_a_loop_go_before_the_node(self):
        a, mm, z = _Op("a"), _Op("mm"), _Op("z")
        node = _node("loop", [mm, _Op("y")])
        pads = [_Op("p1"), _Op("p2")]
        real = [a, node, z, *pads]
        _move_ops_before(OpsView(real), pads, node)
        self.assertEqual(real, [a, *pads, node, z])
        self.assertEqual(node.body[0], mm)

    def test_pad_ops_before_a_body_op_stay_in_the_body_in_order(self):
        x, mm = _Op("x"), _Op("mm")
        node = _node("loop", [x, mm])
        pads = [_Op("p1"), _Op("p2")]
        real = [node, *pads]
        _move_ops_before(OpsView(real), pads, mm)
        self.assertEqual(real, [node])
        self.assertEqual(node.body, [x, *pads, mm])

    def test_pad_ops_before_a_top_level_op(self):
        a, mm = _Op("a"), _Op("mm")
        pads = [_Op("p1"), _Op("p2")]
        real = [a, mm, *pads]
        _move_ops_before(real, pads, mm)
        self.assertEqual(real, [a, *pads, mm])


class TestBracketMutation(unittest.TestCase):
    def test_the_copy_back_stays_in_the_body_when_the_mutation_ends_it(self):
        x, mut, after = _Op("x"), _Op("mut"), _Op("after")
        node = _node("loop", [x, mut])
        tmp, back = _Op("tmp"), _Op("back")
        real = [node, after, tmp, back]
        _bracket_mutation(OpsView(real), mut, tmp, back)
        self.assertEqual(real, [node, after])
        self.assertEqual(node.body, [x, tmp, mut, back])

    def test_a_mutation_that_opens_a_loop_body_keeps_both_copies_in_it(self):
        mut, y = _Op("mut"), _Op("y")
        node = _node("loop", [mut, y])
        tmp, back = _Op("tmp"), _Op("back")
        real = [node, tmp, back]
        _bracket_mutation(OpsView(real), mut, tmp, back)
        self.assertEqual(real, [node])
        self.assertEqual(node.body, [tmp, mut, back, y])

    def test_a_top_level_mutation(self):
        a, mut, z = _Op("a"), _Op("mut"), _Op("z")
        tmp, back = _Op("tmp"), _Op("back")
        real = [a, mut, z, tmp, back]
        _bracket_mutation(real, mut, tmp, back)
        self.assertEqual(real, [a, tmp, mut, back, z])


if __name__ == "__main__":
    unittest.main()
