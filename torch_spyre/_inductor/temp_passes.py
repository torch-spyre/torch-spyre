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

# This file contains inductor passes that are only needed as temp fixes

from typing import cast
import torch


def relayout_linear_weights(graph: torch.fx.Graph) -> None:
    """
    Transpose and realize nn.Linear weights so that they are compatible
    with the backend compiler as it is today. In the future, this pass
    should be eliminated for performance reasons when possible.
    """

    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
            input_t, kernel_t = node.args
            input_t = cast(torch.fx.Node, input_t)
            kernel_t = cast(torch.fx.Node, kernel_t)
            if not kernel_t.meta["val"].is_contiguous():
                with graph.inserting_before(node):
                    # transpose_node = graph.call_function(torch.ops.aten.permute.default, args=(kernel_t, [1, 0]))
                    contiguous_node = graph.call_function(
                        torch.ops.aten.clone.default,
                        args=(kernel_t,),
                        kwargs={"memory_format": torch.contiguous_format},
                    )
                    node.update_arg(1, contiguous_node)


def replace_scalar_with_tensor(graph: torch.fx.Graph) -> None:
    """
    Replace constant arguments to any operation with tensor.
    Scalars are converted to size=1 tensor and passed to the corresponding
    operations which was consuming the scalar value.
    """

    ops_support_list = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.true_divide.Tensor,
        torch.ops.aten.div.Tensor,
    ]

    # Created node cache for scalar values, and reuse the node when
    # the scalar found again.
    const_node_map: dict[int | float, torch.fx.node.Node] = {}

    for node in graph.nodes:
        if node.target not in ops_support_list:
            continue
        scalar_indexes = []
        for i in range(len(node.args)):
            in_arg = node.args[i]
            if not isinstance(in_arg, torch.fx.node.Node):
                if isinstance(in_arg, (int, float)):
                    scalar_indexes.append(i)
                else:
                    print(f"Warning: unhandled node type {type(in_arg)}")

        if len(scalar_indexes) > 0:
            with graph.inserting_before(node):
                for idx in scalar_indexes:
                    scalar_val = node.args[idx]
                    if scalar_val in const_node_map:
                        full_node = const_node_map[scalar_val]
                    else:
                        # Currently the dtype of the scalar tensor is set as same as the output dtype.
                        # TODO: Set the scalar tensor type same as scalar type after to_dtype supported
                        # (open issue: https://github.com/torch-spyre/torch-spyre/issues/41)
                        dtype = torch.float16
                        meta = node.meta.get("tensor_meta", None)
                        if meta:
                            dtype = meta.dtype
                        full_node = graph.call_function(
                            torch.ops.spyre.full.default,
                            args=((1,), scalar_val, torch.device("spyre"), dtype),
                        )
                        const_node_map[scalar_val] = full_node
                    node.update_arg(idx, full_node)
