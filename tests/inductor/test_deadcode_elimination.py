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
"""Unit tests for the ``deadcode_elimination`` IR pass."""

import unittest
from unittest.mock import patch

import torch
from torch import fx
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
from torch._inductor.virtualized import V

from torch_spyre.constants import DEVICE_NAME

from torch_spyre._inductor import deadcode_elimination as dce_module
from torch_spyre._inductor.wsr import for_each_tile_lowering
from torch_spyre._inductor.deadcode_elimination import deadcode_elimination


class TestDeadMutationElimination(unittest.TestCase):
    """Unit tests for ``deadcode_elimination`` on a mutation whose target is dead.

    A mutation writes into a buffer it does not own, so it cannot be dropped for
    its own output being unread -- unless that buffer is dead too, when the write
    reaches nothing. Exercised against real IR rather than through a compile, so
    the rule stays pinned independently of the pass that creates such a mutation.
    """

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()
        self.addCleanup(self._graph_ctx.__exit__, None, None, None)

    @staticmethod
    def _make_buffer(name, reads=None):
        """A ComputedBuffer(Pointwise) reading ``reads`` (an InputBuffer name)."""
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
        # setUp builds a fresh GraphLowering per test, so the stub needs no undo.
        V.graph.operations = list(operations)
        V.graph.get_output_names = lambda: list(output_names)
        deadcode_elimination(V.graph)
        return {op.get_name() for op in V.graph.operations}

    def test_mutation_into_dead_target_is_collected(self):
        """The write cannot be observed, so the op and its reads go."""
        target = self._make_buffer("dead_target")
        producer = self._make_buffer("producer")
        mutation = self._make_buffer("mutation", reads="producer")
        mutation.layout = MutationLayoutSHOULDREMOVE(target)

        survivors = self._run([target, producer, mutation], output_names=[])

        self.assertNotIn("mutation", survivors)
        self.assertNotIn("producer", survivors)

    def test_mutation_into_graph_input_is_kept(self):
        """An ``out=`` destination is observable even though it is no output.

        Regression coverage for test_copy_back_elision, which an outputs-only
        liveness seed silently miscompiled by collecting such a write.
        """
        target = self._make_buffer("arg_target")
        producer = self._make_buffer("producer")
        mutation = self._make_buffer("mutation", reads="producer")
        mutation.layout = MutationLayoutSHOULDREMOVE(target)
        V.graph.graph_inputs["arg_target"] = target

        survivors = self._run([target, producer, mutation], output_names=[])

        self.assertIn("mutation", survivors)
        self.assertIn("producer", survivors)

    def test_mutation_into_live_target_is_kept(self):
        """A write that reaches a graph output keeps the op and its reads."""
        target = self._make_buffer("live_target")
        producer = self._make_buffer("producer")
        mutation = self._make_buffer("mutation", reads="producer")
        mutation.layout = MutationLayoutSHOULDREMOVE(target)

        survivors = self._run(
            [target, producer, mutation], output_names=["live_target"]
        )

        self.assertIn("mutation", survivors)
        self.assertIn("producer", survivors)

    def test_mutation_into_target_read_before_it_is_kept(self):
        """A target proven live only by an op the reverse walk reaches later.

        ``reader`` sits between the target and the mutation, so a single reverse
        pass judges the mutation before learning the target is live and drops the
        write. Pins the iteration to a fixed point.
        """
        target = self._make_buffer("target")
        reader = self._make_buffer("reader", reads="target")
        producer = self._make_buffer("producer")
        mutation = self._make_buffer("mutation", reads="producer")
        mutation.layout = MutationLayoutSHOULDREMOVE(target)

        survivors = self._run(
            [target, reader, producer, mutation], output_names=["reader"]
        )

        self.assertIn("mutation", survivors)
        self.assertIn("producer", survivors)


class TestForEachTileCarryIsCollected(unittest.TestCase):
    """A ``for_each_tile`` carry nobody reads is collected.

    ``scan`` requires a carry in map mode and a one-leaf reduction needs a
    second output leaf, so a step counter is supplied that every mode discards.
    Splicing leaves its whole chain behind, and the pass must collect it.

    What the chain lowers to is not the point and will change: whether its ops
    are pure or mutate the carry's own buffer in place decides which liveness
    rule retires them, and only the outcome is asserted here.
    """

    def test_spliced_carry_scaffolding_is_collected(self):
        from for_each_tile_fixtures import K, M, N, split_m_fn
        from utils_inductor import cached_xavier

        tallies = []
        original = dce_module.deadcode_elimination

        def record(graph):
            before = {op.get_operation_name() for op in graph.operations}
            original(graph)
            after = {op.get_operation_name() for op in graph.operations}
            tallies.append((before, after))

        torch._dynamo.reset()
        X = cached_xavier((M, K)).to(DEVICE_NAME)
        Y = cached_xavier((K, N)).to(DEVICE_NAME)
        # Patch the post-splice call site: the pipeline's own earlier call runs
        # before the carry scaffolding exists.
        with patch.object(for_each_tile_lowering, "deadcode_elimination", record):
            torch.compile(split_m_fn, backend="inductor", fullgraph=True)(X, Y)

        self.assertTrue(tallies, "post-splice deadcode_elimination never ran")
        before, after = max(tallies, key=lambda t: len(t[0]))
        removed = before - after

        carry = {name for name in before if "while_loop_carry_copy" in name}
        self.assertTrue(carry, "fixture produced no spliced carry copies")
        self.assertTrue(
            carry <= removed, f"carry copies survived: {sorted(carry - removed)}"
        )
        self.assertLess(len(after), len(before))


if __name__ == "__main__":
    unittest.main()
