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

"""Where ``GraphEditor`` puts an LX clone in the lowered order (stage 8.2).

A hoisted input clone runs once, before the loop its consumers run in; a drain runs
once, after the loop; any other clone sits beside its first consumer (so it runs every
trip when that consumer is in a loop body). Stub operations, no compile, no device.
"""

import unittest
from types import SimpleNamespace

import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor.loop_node import LoopNode, OpsView
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor


class _Op:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __repr__(self):
        return f"_Op({self.name})"


def _node(name, body):
    return LoopNode(name, body, induction=None, trip_count=4)


class TestPlaceClone(unittest.TestCase):
    def _place(self, real, clone, **kwargs):
        editor = GraphEditor.__new__(GraphEditor)
        editor.lowering = SimpleNamespace(operations=OpsView(real))
        editor._last_writer = kwargs.pop("last_writer", None) or (lambda name: None)
        # The lowering appended the clone at the top level, as register_operation does.
        real.append(clone)
        editor._place_clone(clone, "buf", **kwargs)

    def test_a_hoisted_input_clone_goes_before_the_loop_not_into_its_body(self):
        a, x, y, z = (_Op(n) for n in "axyz")
        node = _node("loop", [x, y])
        real = [a, node, z]
        clone = _Op("clone")
        self._place(
            real, clone, input=True, lower_anchor=None, lower_before=x, first_user=x
        )
        self.assertEqual(real, [a, clone, node, z])
        self.assertEqual(node.body, [x, y])

    def test_a_hoisted_clone_goes_before_the_outermost_loop(self):
        x, y = _Op("x"), _Op("y")
        inner = _node("inner", [x])
        outer = _node("outer", [inner, y])
        real = [outer]
        clone = _Op("clone")
        self._place(
            real, clone, input=True, lower_anchor=None, lower_before=x, first_user=x
        )
        self.assertEqual(real, [clone, outer])
        self.assertEqual(outer.body, [inner, y])

    def test_a_drain_goes_after_the_loop_and_before_a_loop_that_follows(self):
        """The flat view puts an op after the last member at the head of the NEXT
        loop's body; the drain belongs between the two loops."""
        x, y, p, q = (_Op(n) for n in "xypq")
        first = _node("first", [x, y])
        second = _node("second", [p, q])
        real = [first, second]
        clone = _Op("drain")
        self._place(
            real, clone, input=False, lower_anchor=y, lower_before=None, first_user=None
        )
        self.assertEqual(real, [first, clone, second])
        self.assertEqual(first.body, [x, y])
        self.assertEqual(second.body, [p, q])

    def test_a_drain_after_a_nested_last_member_goes_after_the_outermost_loop(self):
        x, y = _Op("x"), _Op("y")
        inner = _node("inner", [x])
        outer = _node("outer", [inner, y])
        after = _Op("after")
        real = [outer, after]
        clone = _Op("drain")
        self._place(
            real, clone, input=False, lower_anchor=x, lower_before=None, first_user=None
        )
        self.assertEqual(real, [outer, clone, after])

    def test_a_clone_with_no_anchor_sits_before_its_first_consumer_in_the_body(self):
        """Not hoisted: it runs every trip, because its consumer does."""
        x, y = _Op("x"), _Op("y")
        node = _node("loop", [x, y])
        real = [node]
        clone = _Op("clone")
        self._place(
            real, clone, input=True, lower_anchor=None, lower_before=None, first_user=y
        )
        self.assertEqual(node.body, [x, clone, y])

    def test_an_output_clone_follows_its_last_writer(self):
        a, w, z = _Op("a"), _Op("w"), _Op("z")
        real = [a, w, z]
        clone = _Op("clone")
        self._place(
            real,
            clone,
            input=False,
            lower_anchor=None,
            lower_before=None,
            first_user=a,
            last_writer=lambda name: w,
        )
        self.assertEqual(real, [a, w, clone, z])

    def test_an_output_clone_of_a_buffer_written_in_a_loop_stays_beside_its_writer(
        self,
    ):
        """It takes its writer's loop membership (``copy_op_metadata``), so it is a
        member of the loop its writer is in (a drain, which clears that, is the
        post-loop case)."""
        x, w = _Op("x"), _Op("w")
        node = _node("loop", [x, w])
        z = _Op("z")
        real = [node, z]
        clone = _Op("clone")
        self._place(
            real,
            clone,
            input=False,
            lower_anchor=None,
            lower_before=None,
            first_user=None,
            last_writer=lambda name: w,
        )
        self.assertEqual(node.body, [x, w, clone])
        self.assertEqual(real, [node, z])

    def test_a_hoist_from_a_loop_that_starts_with_non_counted_ops_stays_in_the_body(
        self,
    ):
        """The allocator's live-range gate looked only from the entry on. Hoisting
        before the whole loop would put the clone ahead of the ops that precede the
        entry in the body; it goes right before the entry instead."""
        extern, entry, y = _Op("extern"), _Op("entry"), _Op("y")
        node = _node("loop", [extern, entry, y])
        real = [node]
        clone = _Op("clone")
        self._place(
            real,
            clone,
            input=True,
            lower_anchor=None,
            lower_before=entry,
            first_user=entry,
        )
        self.assertEqual(node.body, [extern, clone, entry, y])
        self.assertEqual(real, [node])

    def test_a_clone_with_no_consumers_and_no_anchor_is_refused(self):
        real = [_Op("a")]
        with self.assertRaises(AssertionError):
            self._place(
                real,
                _Op("clone"),
                input=True,
                lower_anchor=None,
                lower_before=None,
                first_user=None,
            )


if __name__ == "__main__":
    unittest.main()
