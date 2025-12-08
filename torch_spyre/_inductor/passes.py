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

from typing import Optional, Any, Callable, List

import torch
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_hash_for_files,
)
from torch._inductor.scheduler import BaseSchedulerNode
from .stickify import propagate_spyre_tensor_layouts


class CustomPrePasses(CustomGraphPass):
    """
    This inductor extension point enables Spyre-specific passes to run on the
    post-grad FX graph early in the sequence defined in `post_grad.post_grad_passes`.
    """

    """
    The list of custom passes to run
    """
    passes: List[Callable[[torch.fx.graph.Graph], None]] = []

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in CustomPrePasses.passes:
            p(graph)

    def uuid(self) -> Optional[Any]:
        files = [c.file() for c in CustomPrePasses.passes]
        return get_hash_for_files(tuple(set(files + [__file__])))


class CustomPostPasses(CustomGraphPass):
    """
    This inductor extension point enables Spyre-specific passes to run on the
    post-grad FX graph late in the sequence defined in `post_grad.post_grad_passes`.
    """

    """
    The list of custom passes to run
    """
    passes: List[Callable[[torch.fx.graph.Graph], None]] = []

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in CustomPostPasses.passes:
            p(graph)

    def uuid(self) -> Optional[Any]:
        files = [c.file() for c in CustomPostPasses.passes]
        return get_hash_for_files(tuple(set(files + [__file__])))


def scheduler_passes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    This inductor extension point enables Spyre-specific passes to run over
    the graph of LoopLevelIR nodes immediately before fusion is applied.

    The list of nodes is guarenteed by the caller to be in topological order.
    The returned list of nodes must also be in topological order.
    """
    nodes = propagate_spyre_tensor_layouts(nodes)
    return nodes
