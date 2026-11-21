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

The lowerings build ``BroadcastKernel``, ``AllGatherKernel``, ``AllToAllKernel``,
``AllReduceKernel`` and ``WaitKernel`` with the upstream ``_CollectiveKernel`` factories
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
    InputBuffer,
    NoneLayout,
    StorageBox,
    TensorBox,
)
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode

from torch.spyre import SpyreTensorLayout

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import (
    AllGatherKernel,
    AllReduceKernel,
    AllToAllKernel,
    BroadcastKernel,
    FixedTiledLayout,
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
        # process_kernel reads the FX node being lowered (newer torch checks
        # its target); a lowering always has one, so provide the graph's own.
        stack.enter_context(V.set_current_node(next(iter(gm.graph.nodes))))

    @staticmethod
    def _input(name="x", size=None):
        # Collective codegen requires a finalized device layout, as layout
        # propagation leaves it on real inputs.
        size = list(size or _SIZE)
        stride = [1] * len(size)
        for i in range(len(size) - 2, -1, -1):
            stride[i] = stride[i + 1] * size[i + 1]
        layout = FixedTiledLayout(
            torch.device("cpu"),
            torch.float16,
            size,
            stride,
            SpyreTensorLayout(size, torch.float16),
        )
        buf = InputBuffer(name=name, layout=layout)
        V.graph.name_to_buffer[name] = buf
        V.graph.graph_inputs[name] = buf
        return TensorBox(StorageBox(buf))

    @staticmethod
    def _commit_default_layout(node):
        # What finalize_layouts does for an out-of-place collective whose
        # layout propagation picked the default STL for its own shape.
        layout = node.get_layout()
        node.layout = FixedTiledLayout(
            layout.device,
            layout.dtype,
            list(layout.size),
            list(layout.stride),
            SpyreTensorLayout([int(s) for s in layout.size], layout.dtype),
        )
        return node

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
            (
                AllToAllKernel,
                _c10d.all_to_all_single.default,
                ([4, 4], [4, 4], "group0"),
                _SIZE,
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
                # Split lists (all_to_all) are flattened into constant_args.
                flat = tuple(
                    v for a in args for v in (a if isinstance(a, list) else (a,))
                )
                self.assertEqual(result.constant_args, flat)
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

    def _a2a(self, size, output_splits, input_splits, name="a_in"):
        node = AllToAllKernel.create_out_of_place(
            _c10d.all_to_all_single.default,
            self._input(name, size),
            output_splits,
            input_splits,
            "g0",
        )
        return self._commit_default_layout(node)

    def test_all_to_all_codegen(self):
        # [8, 64] fp16: each row is one 64-element stick, so 4 rows = 256
        # device elements per peer.
        a2a = self._a2a(_SIZE, [4, 4], [4, 4])
        wrapper = _RecordingWrapper()

        a2a.codegen(wrapper)

        a = a2a.get_name()
        self.assertEqual(
            wrapper.header_lines(),
            [
                f"_a2a_plan_{a} = torch.ops.spyre.alltoall_plan("
                f"[256, 256], [256, 256], {_FP16}, 'g0')"
            ],
        )
        self.assertEqual(
            wrapper.lines,
            [f"{a} = torch.ops.spyre.alltoall_run(a_in, _a2a_plan_{a}, 8)"],
        )

    def test_all_to_all_uneven_codegen(self):
        # Send 2 rows to rank 0 and 6 to rank 1; receive 3 + 1 rows, so the
        # output has 4 rows.  Splits are passed in device elements.
        a2a = self._a2a(_SIZE, [3, 1], [2, 6])
        wrapper = _RecordingWrapper()

        a2a.codegen(wrapper)

        a = a2a.get_name()
        self.assertEqual(list(a2a.get_size()), [4, 64])
        self.assertEqual(
            wrapper.header_lines(),
            [
                f"_a2a_plan_{a} = torch.ops.spyre.alltoall_plan("
                f"[192, 64], [128, 384], {_FP16}, 'g0')"
            ],
        )
        self.assertEqual(
            wrapper.lines,
            [f"{a} = torch.ops.spyre.alltoall_run(a_in, _a2a_plan_{a}, 4)"],
        )

    def test_all_to_all_uneven_1d_codegen(self):
        # 1-D: rows are single elements; each split must be whole sticks.
        a2a = self._a2a([256], [64, 128], [192, 64])
        wrapper = _RecordingWrapper()

        a2a.codegen(wrapper)

        a = a2a.get_name()
        self.assertEqual(
            wrapper.header_lines(),
            [
                f"_a2a_plan_{a} = torch.ops.spyre.alltoall_plan("
                f"[64, 128], [192, 64], {_FP16}, 'g0')"
            ],
        )
        self.assertEqual(
            wrapper.lines,
            [f"{a} = torch.ops.spyre.alltoall_run(a_in, _a2a_plan_{a}, 192)"],
        )

    def test_all_to_all_rejects_rows_split_across_stick_tiles(self):
        # [4, 128] fp16 has device_size [2, 4, 64]: dim 1's stick tiles are
        # outside dim 0, so a flat split would interleave rows.
        a2a = self._a2a([4, 128], [2, 2], [2, 2])
        with self.assertRaisesRegex(Unsupported, "rows are not contiguous"):
            a2a.codegen(_RecordingWrapper())

    def test_all_to_all_rejects_split_not_stick_aligned(self):
        # A 1-D split of 32 fp16 elements is 64 bytes, half a stick.
        a2a = self._a2a([128], [32, 96], [32, 96])
        with self.assertRaisesRegex(Unsupported, "multiple of 128 bytes"):
            a2a.codegen(_RecordingWrapper())

    def test_all_to_all_rejects_padded_1d(self):
        # 96 fp16 elements pad to 128 on device; the padding would be sent.
        a2a = self._a2a([96], [32, 64], [32, 64])
        with self.assertRaisesRegex(Unsupported, "padded"):
            a2a.codegen(_RecordingWrapper())

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


class TestUpstreamRecognition(_CollectiveGraphTest):
    """Upstream's comm/compute passes must see both halves of a collective."""

    def test_is_collective_and_is_wait_see_spyre_nodes(self):
        import torch._inductor.comms as comms
        import torch._inductor.scheduler as scheduler
        import torch._inductor.utils as utils

        gathered = AllGatherKernel.create_out_of_place(
            _c10d.all_gather_into_tensor.default, self._input(), 2, "group0"
        )
        WaitKernel.create_wait(_c10d.wait_tensor.default, TensorBox.create(gathered))
        wait = self._last_op()

        self.assertTrue(utils.is_collective(gathered))
        self.assertFalse(utils.is_collective(wait))
        # Upstream compares type(node) exactly; ir.py patches every module that
        # imported is_wait by name, so the subclass is a wait everywhere.
        for mod in (utils, comms, scheduler):
            with self.subTest(mod.__name__):
                self.assertTrue(mod.is_wait(wait))
                self.assertFalse(mod.is_wait(gathered))


if __name__ == "__main__":
    unittest.main()
