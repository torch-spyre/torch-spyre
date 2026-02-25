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
