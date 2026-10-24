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

"""Tests for resolving a subgraph's input layouts from its invoking HOP.

``TestSubgraphInputLayouts`` runs on the host, on stand-in graphs holding only
what ``_assign_subgraph_input_layouts`` reads. ``TestSubgraphViewOperands``
compiles HOPs whose operands are views on the device and compares with CPU.
Further end-to-end coverage is in ``test_invoke_subgraph_split.py`` and the
``test_cmp_cond_subgraph`` cases of ``test_inductor_ops.py``.
"""

import unittest
from types import SimpleNamespace

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region
from torch._inductor.ir import (
    Buffer,
    Conditional,
    FixedLayout,
    InputBuffer,
    ReinterpretView,
    StorageBox,
    TensorBox,
)

from torch_spyre._C import SpyreTensorLayout
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.propagate_layouts import _assign_subgraph_input_layouts

SIZE = [64, 128]
STRIDE = [1, 64]
# The stick on dim 0, which a layout rebuilt from host strides would not pick.
STL = SpyreTensorLayout(SIZE, STRIDE, torch.float16, [1, 0])


def _host_layout(device, size=SIZE, stride=STRIDE, offset=0):
    return FixedLayout(torch.device(device), torch.float16, size, stride, offset)


def _tiled_layout():
    return FixedTiledLayout(torch.device("spyre"), torch.float16, SIZE, STRIDE, STL)


def _input(device="spyre", layout=None):
    layout = layout if layout is not None else _host_layout(device)
    return TensorBox(StorageBox(InputBuffer(name="arg0_1", layout=layout)))


def _conditional(operand):
    hop = Conditional.__new__(Conditional)
    hop.operands = [operand]
    hop.true_subgraph = SimpleNamespace(graph=None)
    hop.false_subgraph = None
    return hop


def _graphs(hop, parent_buffers, subgraph_input):
    """A parent graph holding ``hop`` and a subgraph it invokes."""
    parent = SimpleNamespace(
        operations=[hop],
        try_get_buffer=lambda name: parent_buffers.get(name),
    )
    subgraph = SimpleNamespace(
        name="true_graph_0",
        parent=parent,
        graph_input_names=["arg0_1"],
        graph_inputs={"arg0_1": subgraph_input},
    )
    hop.true_subgraph.graph = subgraph
    return subgraph


class TestSubgraphInputLayouts(unittest.TestCase):
    def test_takes_current_parent_buffer_layout_by_name(self):
        # The HOP holds a stale buffer object with a host layout; the parent's
        # current buffer of the same name carries the final device layout.
        stale = TensorBox(StorageBox(Buffer(name="buf0", layout=_host_layout("spyre"))))
        current = Buffer(name="buf0", layout=_tiled_layout())
        subgraph = _graphs(_conditional(stale), {"buf0": current}, _input())

        _assign_subgraph_input_layouts(subgraph)

        self.assertEqual(subgraph.graph_inputs["arg0_1"].layouts, [STL])

    def test_parent_buffer_without_device_layout_raises(self):
        operand = Buffer(name="buf0", layout=_host_layout("spyre"))
        subgraph = _graphs(_conditional(operand), {"buf0": operand}, _input())

        with self.assertRaisesRegex(Unsupported, "no final device layout"):
            _assign_subgraph_input_layouts(subgraph)

    def test_operand_missing_from_parent_raises(self):
        operand = Buffer(name="buf0", layout=_tiled_layout())
        subgraph = _graphs(_conditional(operand), {}, _input())

        with self.assertRaisesRegex(Unsupported, "no final device layout"):
            _assign_subgraph_input_layouts(subgraph)

    def test_unknown_hop_raises(self):
        unknown = SimpleNamespace(
            get_subgraphs=lambda: [SimpleNamespace(graph=subgraph)]
        )
        subgraph = SimpleNamespace(
            name="body_graph_0",
            parent=SimpleNamespace(operations=[unknown]),
            graph_input_names=["arg0_1"],
            graph_inputs={"arg0_1": _input()},
        )

        with self.assertRaisesRegex(Unsupported, "no Conditional, WhileLoop"):
            _assign_subgraph_input_layouts(subgraph)

    def test_host_input_gets_no_device_layout(self):
        operand = Buffer(name="arg1_1", layout=_host_layout("cpu"))
        subgraph = _graphs(_conditional(operand), {"arg1_1": operand}, _input("cpu"))

        _assign_subgraph_input_layouts(subgraph)

        self.assertFalse(hasattr(subgraph.graph_inputs["arg0_1"], "layouts"))

    def test_view_operand_takes_base_buffer_layout(self):
        # A row slice of the base buffer: the subgraph input's host layout
        # carries the view's offset, and the device layout is the base's.
        view_layout = _host_layout("spyre", [32, 128], STRIDE, offset=32)
        base = Buffer(name="buf0", layout=_host_layout("spyre"))
        view = ReinterpretView(data=StorageBox(base), layout=view_layout)
        current = Buffer(name="buf0", layout=_tiled_layout())
        subgraph = _graphs(
            _conditional(TensorBox(view)), {"buf0": current}, _input(layout=view_layout)
        )

        _assign_subgraph_input_layouts(subgraph)

        self.assertEqual(subgraph.graph_inputs["arg0_1"].layouts, [STL])

    def test_view_operand_with_other_host_layout_raises(self):
        view_layout = _host_layout("spyre", [32, 128], STRIDE, offset=32)
        base = Buffer(name="buf0", layout=_host_layout("spyre"))
        view = ReinterpretView(data=StorageBox(base), layout=view_layout)
        current = Buffer(name="buf0", layout=_tiled_layout())
        subgraph = _graphs(_conditional(TensorBox(view)), {"buf0": current}, _input())

        with self.assertRaisesRegex(Unsupported, "differs from the view operand"):
            _assign_subgraph_input_layouts(subgraph)


def _cond(operand_fn):
    """A ``torch.cond`` on an exact predicate whose operand is ``operand_fn(x)``."""

    def fn(x):
        return torch.cond(
            x.max() > 0, lambda y: y + 1.0, lambda y: y - 1.0, (operand_fn(x),)
        )

    return fn


class TestSubgraphViewOperands(unittest.TestCase):
    """A view passed into a HOP takes its base buffer's device layout."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def _check(self, fn, rows, cols):
        # Small integers, exact in fp16, so a wrong element cannot hide.
        x = ((torch.arange(rows * cols) % 7) - 3).reshape(rows, cols)
        x = x.to(torch.float16)
        got = torch.compile(fn, dynamic=False)(x.to(DEVICE_NAME)).cpu()
        torch.testing.assert_close(got, fn(x), atol=0.005, rtol=0.005)

    def test_cond_transposed_graph_input(self):
        self._check(_cond(lambda x: x.t()), 128, 64)

    def test_cond_leading_column_slice(self):
        self._check(_cond(lambda x: (x * 1.0)[:, :64]), 128, 128)

    def test_cond_offset_column_slice(self):
        self._check(_cond(lambda x: (x * 1.0)[:, 64:]), 128, 128)

    def test_cond_offset_row_slice(self):
        self._check(_cond(lambda x: (x * 1.0)[32:]), 128, 64)

    def test_invoke_subgraph_transposed_graph_input(self):
        @mark_compile_region
        def region(h):
            return h + 1.0

        # Two calls: a single call is inlined and no invoke_subgraph remains.
        self._check(lambda x: region(region(x.t())), 128, 64)


if __name__ == "__main__":
    unittest.main()
