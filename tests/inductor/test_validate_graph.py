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
"""Tests for the operation-order check run after every pre-scheduling pass,
and for the split_multi_ops ordering bug that motivated it (PR #5982)."""

from unittest.mock import patch

import pytest
import torch
from torch import fx
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
    FixedLayout,
    InputBuffer,
    MultiOutput,
    MultiOutputLayout,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.lowering import lowerings
from torch._inductor.virtualized import V

from torch_spyre._inductor import split_multi_ops as smo
from torch_spyre._inductor.passes import (
    CustomPreSchedulingPasses,
    InvalidGraphError,
    check_operation_order,
)

_CPU = torch.device("cpu")


def _layout(stride=(1,)):
    return FixedLayout(_CPU, torch.float32, [8], list(stride) if stride else None)


@pytest.fixture
def graph():
    gm = fx.symbolic_trace(lambda: None)
    gl = GraphLowering(gm)
    with V.set_graph_handler(gl):
        yield gl


def _input(name):
    buf = InputBuffer(name=name, layout=_layout())
    V.graph.graph_inputs[name] = TensorBox(StorageBox(buf))
    V.graph.name_to_buffer[name] = buf
    return buf


def _pointwise(name, *reads, layout=None):
    """A ComputedBuffer named ``name`` that adds the buffers in ``reads``."""
    boxes = [TensorBox(StorageBox(r)) for r in reads]

    def inner_fn(index):
        values = [b.make_loader()(index) for b in boxes]
        out = values[0]
        for v in values[1:]:
            out = V.ops.add(out, v)
        return out

    pw = Pointwise.create(
        device=_CPU, dtype=torch.float32, inner_fn=inner_fn, ranges=[8]
    )
    buf = ComputedBuffer(name=name, layout=layout or _layout(), data=pw.data.data)
    buf.operation_name = f"op_{name}"
    V.graph.name_to_buffer[name] = buf
    return buf


def _fallback_with_output(arg):
    """A FallbackKernel with a MultiOutputLayout and the MultiOutput reading it,
    registered in that order, as FallbackKernel.create does for a tuple result."""
    kernel = FallbackKernel(
        MultiOutputLayout(device=_CPU),
        torch.ops.aten.sort.default,
        [arg],
        [],
        lambda *a: a,
    )
    out = MultiOutput(_layout(), kernel, [(tuple, 0)])
    kernel.outputs = [out]
    return kernel, out


class TestCheckOperationOrder:
    def test_ordered_graph_passes(self, graph):
        x = _input("x")
        a = _pointwise("a", x)
        b = _pointwise("b", a, x)
        check_operation_order([a, b], "p")

    def test_consumer_before_producer_raises(self, graph):
        x = _input("x")
        a = _pointwise("a", x)
        b = _pointwise("b", a)
        with pytest.raises(
            InvalidGraphError,
            match=r"after pre-scheduling pass p: operation b .* at position 0 "
            r"reads a, which a .* produces later, at position 1",
        ):
            check_operation_order([b, a], "p")

    def test_reduction_body_is_traced(self, graph):
        # get_reads() raises NotImplementedError on a Reduction body; the check
        # must still see its reads.
        x = _input("x")
        a = _pointwise("a", x)
        red = lowerings[torch.ops.aten.sum.default](TensorBox(StorageBox(a)))
        red.realize()
        r = red.data.data
        assert isinstance(r.data, Reduction)
        check_operation_order([a, r], "p")
        with pytest.raises(InvalidGraphError, match="reads a"):
            check_operation_order([r, a], "p")

    def test_multi_output_before_parent_raises(self, graph):
        x = _input("x")
        kernel, out = _fallback_with_output(x)
        check_operation_order([kernel, out], "p")
        with pytest.raises(
            InvalidGraphError,
            match=rf"{out.get_name()} \(MultiOutput\) at position 0 reads "
            rf"{kernel.get_name()}",
        ):
            check_operation_order([out, kernel], "p")

    def test_inputs_and_unknown_names_are_outside_the_check(self, graph):
        # x is a graph input; y is named nowhere in the list (as for a buffer
        # of an enclosing graph). Neither has a producer to be out of order with.
        x = _input("x")
        y = InputBuffer(name="y", layout=_layout())
        V.graph.name_to_buffer["y"] = y
        check_operation_order([_pointwise("a", x, y)], "p")

    def test_mutation_may_read_its_target(self, graph):
        x = _input("x")
        a = _pointwise("a", x)
        mut = _pointwise(
            "mut", a, layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(a)))
        )
        check_operation_order([a, mut], "p")
        with pytest.raises(InvalidGraphError, match="reads a"):
            check_operation_order([mut, a], "p")


class TestLowerFxNodeOrder:
    """_lower_fx_node must move every operation a lowering registers, in order.

    A host fallback with a tuple result registers a FallbackKernel and then the
    MultiOutput that reads it. Before the fix only the MultiOutput was moved,
    which left it ahead of its parent and the parent after the consumer.
    """

    def test_fallback_and_multi_output_move_together(self, graph):
        x = _input("x")
        before = _pointwise("before", x)
        consumer = _pointwise("consumer", x)
        graph.operations[:] = [before, consumer]
        registered = []

        def run_node(node):
            kernel, out = _fallback_with_output(x)  # appended by register_*
            registered.extend([kernel, out])
            return TensorBox(StorageBox(out))

        node = graph.graph.create_node("placeholder", "n")
        with patch.object(graph, "run_node", run_node):
            buf, count = smo._lower_fx_node(node, graph, graph.operations, 1)

        kernel, out = registered
        assert buf is out
        assert count == 2
        assert graph.operations == [before, kernel, out, consumer]
        check_operation_order(graph.operations, "split_multi_ops")


def _out_of_order_pass(graph):
    a, b = graph.operations
    graph.operations[:] = [b, a]


class TestRunsAfterEveryPass:
    def _passes(self):
        obj = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        obj.passes = [_out_of_order_pass]
        return obj

    def _graph_ops(self, graph):
        x = _input("x")
        a = _pointwise("a", x)
        graph.operations[:] = [a, _pointwise("b", a)]

    def test_pass_that_breaks_order_is_named(self, graph):
        self._graph_ops(graph)
        with pytest.raises(InvalidGraphError, match="_out_of_order_pass"):
            self._passes()._run_pass_loop(graph)
