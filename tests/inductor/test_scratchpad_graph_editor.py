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

from types import SimpleNamespace

import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    Pointwise,
    ReinterpretView,
    StorageBox,
    TensorBox,
)

from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor


def _buffer(name: str) -> ComputedBuffer:
    device = torch.device("spyre")
    return ComputedBuffer(
        name=name,
        layout=FixedLayout(device, torch.float16, [2, 3]),
        data=Pointwise(
            device=device,
            dtype=torch.float16,
            inner_fn=lambda _i0, _i1: 0,
            ranges=[2, 3],
        ),
    )


def test_change_graph_output_skips_unrelated_view_and_preserves_matching_view():
    old = _buffer("old")
    new = _buffer("new")
    unrelated = _buffer("unrelated")
    view_layout = FixedLayout(
        torch.device("spyre"), torch.float16, [3, 2], [1, 3], offset=1
    )
    view = ReinterpretView(data=StorageBox(old), layout=view_layout)
    output = TensorBox(StorageBox(view))
    unrelated_view = ReinterpretView(
        data=StorageBox(unrelated), layout=unrelated.layout
    )
    unrelated_output = TensorBox(StorageBox(unrelated_view))
    lowering = SimpleNamespace(graph_outputs=[unrelated_output, output])
    editor = object.__new__(GraphEditor)
    editor.lowering = lowering

    editor.change_graph_output(old, new)

    assert lowering.graph_outputs[0] is unrelated_output
    assert unrelated_view.data.data is unrelated
    assert lowering.graph_outputs[1] is output
    assert output.data.data is view
    assert view.layout is view_layout
    assert view.data.data is new
