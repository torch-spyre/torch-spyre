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

import torch
from .logging_utils import get_inductor_logger
from torch_spyre._C import get_elem_in_stick

logger = get_inductor_logger("padding")
aten = torch.ops.aten

"""
Pass to add padding where useful for correctness or performance.  Must be a pre-grad pass 
if we want to leverage decomposition for constant_pad_nn.  If this pass must come later then
pad will need to be inserted in its decomposed form.
"""


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def pad_arg(graph: torch.fx.Graph, node: torch.fx.Node, arg_i: int, dim: int) -> None:
    arg = node.args[arg_i]
    example_value = arg.meta["example_value"]
    shape = example_value.shape
    ndim = len(shape)
    dim = dim if dim >= 0 else ndim + dim  # convert neg to pos indices

    pad = compute_padding(shape[dim], example_value.dtype)
    if pad > 0:
        # We are paddding dimension 'dim'.  F.pad takes padding inner to outer (until no more)
        # so put in zeros until dim is reached
        pad_list = [0, 0] * (ndim - 1 - dim) + [0, pad]
        with graph.inserting_after(arg):
            padded = graph.call_function(
                aten.constant_pad_nd.default,
                args=(arg, pad_list, 0.0),
            )
            new_shape = list(shape)
            new_shape[dim] += pad
            padded.meta["example_value"] = example_value.new_empty(new_shape)
        node.replace_input_with(arg, padded)


def insert_padding(graph: torch.fx.Graph) -> None:
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in [
            torch.matmul,
            torch.mm,
            torch.bmm,
        ]:
            args = node.args
            if not all(isinstance(arg, torch.fx.Node) for arg in args):
                continue

            x_val = args[0].meta.get("example_value")
            if x_val is None or not isinstance(x_val, torch.Tensor):
                continue
            # Skip if reduction dim size is 1 (special cased in in lowering, size 1 mm is converted to mul)
            if x_val.shape[-1] == 1:
                continue

            # Backend only requires padding arg_1 dim_-2 here, because arg_0 dim_-1 gets stick padding elsewhere.
            # However we are padding at the pytorch level so we also need to pad arg_0 dim_-1 or we generate
            # invalid matmul dimension errors.
            pad_arg(graph, node, arg_i=0, dim=-1)
            pad_arg(graph, node, arg_i=1, dim=-2)
