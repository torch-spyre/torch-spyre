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

"""CPU-only regression tests for split_multi_ops helpers."""

import unittest
from types import SimpleNamespace

import sympy
import torch
import torch.fx as fx
from torch._inductor.ir import FixedLayout, InputBuffer, StorageBox, TensorBox

from torch_spyre._inductor.split_multi_ops import _find_fx_node


class TestFindFxNode(unittest.TestCase):
    def test_skips_symbol_env_entries(self):
        """Dynamic-shape symbols in GraphLowering.env are not tensor buffers."""
        graph = fx.Graph()
        symbolic_node = graph.placeholder("symbolic")
        tensor_node = graph.placeholder("arg0")

        input_buffer = InputBuffer(
            name="buf0",
            layout=FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
        )
        tensor_box = TensorBox(StorageBox(input_buffer))
        gl = SimpleNamespace(
            env={
                symbolic_node: sympy.Symbol("s0"),
                tensor_node: tensor_box,
            },
            graph=graph,
        )

        self.assertIs(_find_fx_node("buf0", gl), tensor_node)


if __name__ == "__main__":
    unittest.main()
