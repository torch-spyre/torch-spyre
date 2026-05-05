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
Pass to add padding where useful for correctness or performance.  Runs as the first pass
in CustomPostPasses on the post-grad FX graph, so it matches ATen default overloads
(aten.mm.default, aten.bmm.default).  Decompositions run before post-grad passes, so this
pass inserts the decomposed form directly: spyre.full + aten.expand + aten.clone +
spyre.overwrite (mirroring pad_decomp in decompositions.py).
"""


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def pad_arg(graph: torch.fx.Graph, node: torch.fx.Node, arg_i: int, dim: int) -> None:
    arg = node.args[arg_i]
    val = arg.meta["val"]
    shape = val.shape
    ndim = len(shape)
    dim = dim if dim >= 0 else ndim + dim  # convert neg to pos indices

    pad = compute_padding(shape[dim], val.dtype)
    if pad > 0:
        output_shape = list(shape)
        output_shape[dim] += pad
        device = val.device
        dtype = val.dtype
        # Replicate the pad_decomp transformation from decompositions.py directly
        # as FX nodes, since decompositions run before post-grad passes.
        # Insert all new nodes immediately after the arg node.
        with graph.inserting_after(arg):
            scalar = graph.call_function(
                torch.ops.spyre.full.default,
                args=([1], 0.0, device, dtype),
            )
            scalar.meta["val"] = torch.empty([1], dtype=dtype, device=device)
        with graph.inserting_after(scalar):
            expanded = graph.call_function(
                aten.expand.default,
                args=(scalar, output_shape),
            )
            expanded.meta["val"] = torch.empty(output_shape, dtype=dtype, device=device)
        with graph.inserting_after(expanded):
            output = graph.call_function(
                aten.clone.default,
                args=(expanded,),
            )
            output.meta["val"] = torch.empty(output_shape, dtype=dtype, device=device)
        with graph.inserting_after(output):
            padded = graph.call_function(
                torch.ops.spyre.overwrite_f.default,
                args=(arg, output, [dim], [0]),
            )
            padded.meta["val"] = torch.empty(output_shape, dtype=dtype, device=device)
        node.replace_input_with(arg, padded)


def insert_padding(graph: torch.fx.Graph) -> None:
    matmul_ops = {aten.mm.default, aten.bmm.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in matmul_ops:
            args = node.args
            if not all(isinstance(arg, torch.fx.Node) for arg in args):
                continue

            x_val = args[0].meta.get("val")
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
