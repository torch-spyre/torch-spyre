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
"""Unit tests for the Spyre collective IR nodes, built through the factories.

The lowerings build ``BroadcastKernel``, ``AllGatherKernel``, ``AllReduceKernel``
and ``WaitKernel`` with the upstream ``_CollectiveKernel`` factories
(``create_inplace``, ``create_out_of_place``, ``create_wait``), so their
dependency model is stock Inductor's and can change with the torch pin. These
tests build real nodes the same way, under a minimal graph handler and without
spyre-comms, so such a change fails here rather than on multi-rank hardware.
"""

import unittest
from contextlib import ExitStack

import torch
from torch import fx
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    FixedLayout,
    InputBuffer,
    NoneLayout,
    StorageBox,
    TensorBox,
)
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_spyre._inductor.ir import (
    AllGatherKernel,
    AllReduceKernel,
    BroadcastKernel,
    WaitKernel,
    _dtype_to_int,
)

_c10d = torch.ops._c10d_functional
_SIZE = [8, 64]
_FP16 = _dtype_to_int(torch.float16)


class _RecordingWrapper:
    """The two wrapper entry points collective codegen writes to."""

    def __init__(self):
        self.header = IndentedBuffer()
        self.lines = []

    def writeline(self, line):
        self.lines.append(line)

    def header_lines(self):
        return self.header.getvalue().splitlines()


class _CollectiveGraphTest(unittest.TestCase):
    """A fresh graph handler with a fake mode, as the factories need."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        stack = ExitStack()
        self.addCleanup(stack.close)
        # The factories run the op's Meta impl under the graph's fake mode.
        stack.enter_context(V.set_fake_mode(FakeTensorMode()))
        stack.enter_context(V.set_graph_handler(GraphLowering(gm)))

    @staticmethod
    def _input(name="x"):
        buf = InputBuffer(
            name=name,
            layout=FixedLayout(torch.device("cpu"), torch.float16, _SIZE, [64, 1]),
        )
        V.graph.name_to_buffer[name] = buf
        V.graph.graph_inputs[name] = buf
        return TensorBox(StorageBox(buf))

    @staticmethod
    def _last_op():
        return V.graph.operations[-1]


class TestFactoryBookkeeping(_CollectiveGraphTest):
    """What the factories record, which codegen and the passes rely on."""

    def test_all_reduce_is_in_place(self):
        x = self._input()
        AllReduceKernel.create_inplace(_c10d.all_reduce_.default, x, "sum", "group0")
        node = self._last_op()

        self.assertIsInstance(node, AllReduceKernel)
        # Codegen treats NoneLayout as "in place, bind nothing".
        self.assertIsInstance(node.layout, NoneLayout)
        self.assertEqual(node.constant_args, ("sum", "group0"))
        self.assertEqual(node.alias_names, ["x"])
        self.assertEqual(
            [
                [b.get_name() for b in out.get_mutation_buffers()]
                for out in node.mutation_outputs
            ],
            [["x"]],
        )
        self.assertTrue(node.has_side_effects())

    def test_out_of_place_nodes_own_their_output(self):
        cases = [
            (
                BroadcastKernel,
                _c10d.broadcast.default,
                (0, "group0"),
                _SIZE,
            ),
            (
                AllGatherKernel,
                _c10d.all_gather_into_tensor.default,
                (2, "group0"),
                [2 * _SIZE[0], _SIZE[1]],
            ),
        ]
        for cls, op, args, size in cases:
            with self.subTest(cls.__name__):
                result = cls.create_out_of_place(
                    op, self._input(f"in_{cls.__name__}"), *args
                )

                self.assertIsInstance(result, cls)
                self.assertNotIsInstance(result.layout, NoneLayout)
                self.assertEqual(list(result.get_size()), size)
                self.assertEqual(result.get_dtype(), torch.float16)
                self.assertEqual(result.constant_args, args)
                self.assertEqual(result.mutation_outputs, [])
                self.assertTrue(result.has_side_effects())

    def test_wait_mutates_the_collective_result(self):
        gathered = AllGatherKernel.create_out_of_place(
            _c10d.all_gather_into_tensor.default, self._input(), 2, "group0"
        )
        WaitKernel.create_wait(_c10d.wait_tensor.default, TensorBox.create(gathered))
        wait = self._last_op()

        self.assertIsInstance(wait, WaitKernel)
        self.assertIsInstance(wait.layout, NoneLayout)
        # This MutationOutput is what orders later readers after the wait.
        self.assertEqual(
            [
                [b.get_name() for b in out.get_mutation_buffers()]
                for out in wait.mutation_outputs
            ],
            [[gathered.get_name()]],
        )


class TestCodegen(_CollectiveGraphTest):
    """The plan/run/wait lines emitted, checked without spyre-comms."""

    def test_all_reduce_runs_in_place(self):
        AllReduceKernel.create_inplace(
            _c10d.all_reduce_.default, self._input(), "sum", "group0"
        )
        node = self._last_op()
        wrapper = _RecordingWrapper()

        node.codegen(wrapper)

        plan = f"_ar_plan_{node.get_name()}"
        self.assertEqual(
            wrapper.header_lines(),
            [f"{plan} = torch.ops.spyre.allreduce_plan(512, {_FP16}, 'sum', 'group0')"],
        )
        self.assertEqual(wrapper.lines, [f"torch.ops.spyre.allreduce_run(x, {plan})"])

    def test_out_of_place_run_binds_the_node(self):
        bcast = BroadcastKernel.create_out_of_place(
            _c10d.broadcast.default, self._input("b_in"), 0, "group0"
        )
        gathered = AllGatherKernel.create_out_of_place(
            _c10d.all_gather_into_tensor.default, self._input("g_in"), 2, "group0"
        )
        wrapper = _RecordingWrapper()

        bcast.codegen(wrapper)
        gathered.codegen(wrapper)

        b, g = bcast.get_name(), gathered.get_name()
        self.assertEqual(
            wrapper.header_lines(),
            [
                f"_bcast_plan_{b} = torch.ops.spyre.broadcast_plan("
                f"512, {_FP16}, 0, 'group0')",
                f"_ag_plan_{g} = torch.ops.spyre.allgather_plan("
                f"512, {_FP16}, 2, 'group0')",
            ],
        )
        self.assertEqual(
            wrapper.lines,
            [
                f"{b} = torch.ops.spyre.broadcast_run(b_in, _bcast_plan_{b}, 0)",
                f"{g} = torch.ops.spyre.allgather_run(g_in, _ag_plan_{g}, 2)",
            ],
        )

    def test_wait_is_a_bare_call(self):
        gathered = AllGatherKernel.create_out_of_place(
            _c10d.all_gather_into_tensor.default, self._input(), 2, "group0"
        )
        WaitKernel.create_wait(_c10d.wait_tensor.default, TensorBox.create(gathered))
        wrapper = _RecordingWrapper()

        self._last_op().codegen(wrapper)

        self.assertEqual(wrapper.header_lines(), [])
        self.assertEqual(
            wrapper.lines, [f"torch.ops.spyre.wait_work({gathered.get_name()})"]
        )

    def test_repeated_codegen_emits_the_plan_once(self):
        AllReduceKernel.create_inplace(
            _c10d.all_reduce_.default, self._input(), "sum", "group0"
        )
        node = self._last_op()
        wrapper = _RecordingWrapper()

        node.codegen(wrapper)
        node.codegen(wrapper)

        self.assertEqual(len(wrapper.header_lines()), 1)
        self.assertEqual(len(wrapper.lines), 2)

    def test_same_shape_nodes_each_emit_a_plan(self):
        """Different nodes get their own plan variable, so two plan lines.

        They share one plan at runtime through the C++ plan cache, not here.
        """
        x = self._input()
        AllReduceKernel.create_inplace(_c10d.all_reduce_.default, x, "sum", "group0")
        first = self._last_op()
        AllReduceKernel.create_inplace(_c10d.all_reduce_.default, x, "sum", "group0")
        second = self._last_op()
        wrapper = _RecordingWrapper()

        first.codegen(wrapper)
        second.codegen(wrapper)

        self.assertEqual(len(wrapper.header_lines()), 2)


if __name__ == "__main__":
    unittest.main()
