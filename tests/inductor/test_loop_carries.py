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

"""Unit tests for ``loop_carries.apply_loop_carries`` on hand-built real IR.

Each graph is a carry in the pure form the splice leaves: a storage (the
carry's initial buffer), an update that reads it and writes its own buffer,
and readers of the update in the body and after the loop. The pass must turn
the update into an in-place write of the storage and send every reader to the
storage. No Spyre device is needed.
"""

import unittest

import sympy
import torch
from torch import fx
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    ReinterpretView,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V

from torch_spyre._inductor.loop_carries import apply_loop_carries
from torch_spyre._inductor.loop_info import LoopCarryRecord
from torch_spyre._inductor.loop_node import LoopNode
from torch_spyre._inductor.pass_utils import redirect_computed_buffer_reads


def _layout(size=(8,), stride=None):
    return FixedLayout(torch.device("cpu"), torch.float32, list(size), stride)


class _CarryGraph(unittest.TestCase):
    """A fresh GraphLowering per test, and helpers to build ops on it."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self.graph = GraphLowering(gm)
        ctx = V.set_graph_handler(self.graph)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.graph.operations = []
        self.graph.graph_outputs = []

    def _input(self, name, size=(8,)):
        buf = InputBuffer(name=name, layout=_layout(size))
        self.graph.name_to_buffer[name] = buf
        self.graph.graph_inputs[name] = TensorBox(StorageBox(buf))
        return buf

    def _op(self, name, reads, size=(8,), stride=None):
        """A Pointwise ComputedBuffer summing the buffers named in ``reads``."""

        def inner_fn(index, _reads=tuple(reads)):
            vals = [V.ops.load(r, index[-1]) for r in _reads]
            out = vals[0]
            for v in vals[1:]:
                out = V.ops.add(out, v)
            return out

        buf = ComputedBuffer(
            name=name,
            layout=_layout(size, stride),
            data=Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                inner_fn=inner_fn,
                ranges=list(size),
            ),
        )
        buf.operation_name = name
        self.graph.name_to_buffer[name] = buf
        self.graph.name_to_op[name] = buf
        return buf

    def _carry(self, view=None):
        """storage <- x;  [loop: upd = storage + x;  body_reader = upd];
        after = upd.  Returns the ops and the record."""
        self._input("x")
        storage = self._op("storage", ["x"])
        update = self._op("upd", ["storage", "x"])
        body_reader = self._op("body_reader", ["upd"])
        after = self._op("after", ["upd"])
        record = LoopCarryRecord("storage", "upd", None, storage_view=view)
        storage._loop_carry_record = record
        update._loop_carry_record = record
        return storage, update, body_reader, after, record

    @staticmethod
    def _reads(op):
        return {dep.name for dep in op.get_read_writes().reads}

    def _by_name(self, name):
        from torch_spyre._inductor.loop_node import all_ops

        return next(
            op for op in all_ops(self.graph.operations) if op.get_name() == name
        )


class TestApplyLoopCarries(_CarryGraph):
    def test_applies_mutation_and_redirects_body_and_outside_readers(self):
        storage, update, body_reader, after, _ = self._carry()
        self.graph.operations = [storage, update, body_reader, after]
        self.graph.graph_outputs = [TensorBox(StorageBox(update))]

        apply_loop_carries(self.graph)

        self.assertIsInstance(update.layout, MutationLayoutSHOULDREMOVE)
        self.assertIs(update.layout.get_buffer(), storage)
        self.assertIn("storage", self.graph.mutated_buffers)
        for name in ("body_reader", "after"):
            self.assertEqual(self._reads(self._by_name(name)), {"storage"}, name)
        # The update still reads the storage it now writes.
        self.assertEqual(self._reads(self._by_name("upd")), {"storage", "x"})
        # The graph output that was the update's buffer is the storage now.
        self.assertEqual(self.graph.graph_outputs[0].get_name(), "storage")

    def test_second_run_changes_nothing(self):
        storage, update, body_reader, after, _ = self._carry()
        self.graph.operations = [storage, update, body_reader, after]
        apply_loop_carries(self.graph)
        ops = list(self.graph.operations)
        layout = update.layout

        apply_loop_carries(self.graph)

        self.assertEqual([id(o) for o in self.graph.operations], [id(o) for o in ops])
        self.assertIs(update.layout, layout)

    def test_finds_the_update_by_name_after_a_rebuild(self):
        """A pass rebuilt the update under its name: the live object is applied,
        not the one the record was stamped on."""
        storage, update, body_reader, after, _ = self._carry()
        self.graph.operations = [storage, update, body_reader, after]
        rebuilt = redirect_computed_buffer_reads(
            update, {}, self.graph.operations, pass_name="test"
        )
        self.assertIsNot(rebuilt, update)

        apply_loop_carries(self.graph)

        self.assertIsInstance(rebuilt.layout, MutationLayoutSHOULDREMOVE)
        self.assertNotIsInstance(update.layout, MutationLayoutSHOULDREMOVE)
        self.assertEqual(self._reads(self._by_name("body_reader")), {"storage"})

    def test_finds_a_rebuilt_storage_by_name(self):
        storage, update, body_reader, after, _ = self._carry()
        self.graph.operations = [storage, update, body_reader, after]
        rebuilt = redirect_computed_buffer_reads(
            storage, {}, self.graph.operations, pass_name="test"
        )

        apply_loop_carries(self.graph)

        self.assertIs(update.layout.get_buffer(), rebuilt)

    def test_writes_through_the_recorded_view(self):
        """A carry whose init is a view of its storage writes through the view."""
        self._input("x", size=(2, 4))
        storage = self._op("storage", ["x"], size=(2, 4), stride=[4, 1])
        view = ReinterpretView(data=StorageBox(storage), layout=_layout((4, 2), [1, 4]))
        update = self._op("upd", ["storage"], size=(4, 2), stride=[2, 1])
        record = LoopCarryRecord("storage", "upd", None, storage_view=view)
        storage._loop_carry_record = record
        update._loop_carry_record = record
        self.graph.operations = [storage, update]

        apply_loop_carries(self.graph)

        self.assertIs(update.layout.target, view)
        self.assertIs(update.layout.get_buffer(), storage)
        # The store indexes through the view (transposed), not the storage.
        i, j = sympy.symbols("i j")
        self.assertEqual(update.layout.make_indexer()([i, j]), i + 4 * j)

    def test_redirects_a_reader_inside_a_loop_node(self):
        storage, update, body_reader, after, _ = self._carry()
        node = LoopNode(
            "loopnode0", [update, body_reader], induction=None, trip_count=4
        )
        self.graph.operations = [storage, node, after]

        apply_loop_carries(self.graph)

        self.assertIsInstance(update.layout, MutationLayoutSHOULDREMOVE)
        self.assertIs(node.body[0], update)
        self.assertEqual(self._reads(node.body[1]), {"storage"})
        self.assertEqual(self._reads(self.graph.operations[2]), {"storage"})

    def test_ignores_a_scatter_style_mutation_without_a_record(self):
        """A scatter carry stays a mutation from the splice and has no record."""
        self._input("x")
        dst = self._op("dst", ["x"])
        scatter = self._op("scatter", ["x"])
        scatter.layout = MutationLayoutSHOULDREMOVE(dst)
        reader = self._op("reader", ["scatter"])
        self.graph.operations = [dst, scatter, reader]
        layout = scatter.layout

        apply_loop_carries(self.graph)

        self.assertIs(scatter.layout, layout)
        self.assertIs(self.graph.operations[2], reader)
        self.assertEqual(self._reads(reader), {"scatter"})

    def test_leaves_a_loop_without_records_untouched(self):
        self._input("x")
        a = self._op("a", ["x"])
        b = self._op("b", ["a"])
        node = LoopNode("loopnode0", [a, b], induction=None, trip_count=4)
        self.graph.operations = [node]

        apply_loop_carries(self.graph)

        self.assertEqual(node.body, [a, b])
        self.assertNotIsInstance(a.layout, MutationLayoutSHOULDREMOVE)
        self.assertEqual(self._reads(b), {"a"})

    def test_a_dead_carry_is_skipped(self):
        """Dead-code elimination removed both ends: nothing to apply."""
        storage, update, *_ = self._carry()
        self._input("y")
        other = self._op("other", ["y"])
        self.graph.operations = [other]

        apply_loop_carries(self.graph)  # must not raise

        self.assertNotIsInstance(update.layout, MutationLayoutSHOULDREMOVE)

    def test_missing_storage_of_a_live_update_raises(self):
        storage, update, body_reader, after, _ = self._carry()
        self.graph.operations = [update, body_reader, after]
        with self.assertRaisesRegex(RuntimeError, "storage 'storage' is not"):
            apply_loop_carries(self.graph)

    def test_missing_update_of_a_live_storage_raises(self):
        storage, update, body_reader, after, _ = self._carry()
        self.graph.operations = [storage, body_reader, after]
        with self.assertRaisesRegex(RuntimeError, "update 'upd' is gone"):
            apply_loop_carries(self.graph)


if __name__ == "__main__":
    unittest.main()
