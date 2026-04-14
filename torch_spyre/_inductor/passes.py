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

import inspect
from typing import Optional, Any, Callable, List
from abc import abstractmethod

import torch
import torch.fx.graph
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_hash_for_files,
)
from torch._inductor.ir import Operation
from torch._inductor.scheduler import BaseSchedulerNode

from .padding import insert_padding
from .temp_passes import (
    bmm_unflatten_pass,
    mm_to_bmm_pass,
    relayout_linear_weights,
    replace_scalar_with_tensor,
)
from . import config
from .stickify import propagate_mutation_layouts, propagate_spyre_tensor_layouts
from .core_division import core_division_planning
from .scratchpad import scratchpad_planning
from .fusion import spyre_fuse_nodes
from .constants import DEVICE_NAME


def _maybe_run_graph_pass(pass_fn, graph: torch.fx.graph.Graph) -> None:
    has_spyre_device = any(
        isinstance(node, torch.fx.Node)
        and isinstance(node.meta["val"], torch.Tensor)
        and node.meta["val"].device.type == DEVICE_NAME
        for node in graph.nodes
    )

    if has_spyre_device:
        return pass_fn(graph)


class CustomPreGradPasses:
    """
    This inductor extension point enables Spyre-specific passes to run on the
    pre-grad FX graph.
    """

    passes: List[Callable[[torch.fx.graph.Graph], None]] = [insert_padding]

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in self.passes:
            p(graph)


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
            _maybe_run_graph_pass(p, graph)

    def uuid(self) -> Optional[Any]:
        files = [inspect.getfile(c) for c in CustomPrePasses.passes]
        # Use dict.fromkeys instead of set for deterministic order
        return get_hash_for_files(tuple(dict.fromkeys(files + [__file__])))


class CustomPostPasses(CustomGraphPass):
    """
    This inductor extension point enables Spyre-specific passes to run on the
    post-grad FX graph late in the sequence defined in `post_grad.post_grad_passes`.
    """

    """
    The list of custom passes to run
    """
    passes: List[Callable[[torch.fx.graph.Graph], None]] = [
        replace_scalar_with_tensor,
        relayout_linear_weights,
        mm_to_bmm_pass.apply,
        bmm_unflatten_pass.apply,
    ]

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in CustomPostPasses.passes:
            _maybe_run_graph_pass(p, graph)

    def uuid(self) -> Optional[Any]:
        files = [inspect.getfile(c) for c in CustomPostPasses.passes]
        # Use dict.fromkeys instead of set for deterministic order
        return get_hash_for_files(tuple(dict.fromkeys(files + [__file__])))


def _maybe_run_scheduler_pass(
    pass_fn, nodes: list[BaseSchedulerNode]
) -> list[BaseSchedulerNode]:
    has_spyre_device = any(
        node.get_device() is not None and node.get_device().type == DEVICE_NAME
        for node in nodes
    )

    if has_spyre_device:
        return pass_fn(nodes)

    return nodes


class CustomNodePassBase(CustomGraphPass):
    def __call__(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        for _pass in self.get_passes():
            nodes = _maybe_run_scheduler_pass(_pass, nodes)
        return nodes

    @abstractmethod
    def get_passes(
        self,
    ) -> list[Callable[[list[BaseSchedulerNode]], list[BaseSchedulerNode]]]:
        pass

    def uuid(self) -> Optional[Any]:
        files = [inspect.getfile(c) for c in self.get_passes()]
        return get_hash_for_files(tuple(dict.fromkeys(files + [__file__])))


class CustomPreFusionPasses(CustomNodePassBase):
    """
    This inductor extension point enables Spyre-specific passes to run over
    the graph of LoopLevelIR nodes immediately before Inductor's fusion pass runs.

    The list of nodes is guarenteed by the caller to be in topological order.
    The returned list of nodes must also be in topological order.
    """

    def get_passes(self):
        return [propagate_mutation_layouts]


class CustomPostFusionPasses(CustomNodePassBase):
    """
    This inductor extension point enables Spyre-specific passes to run over
    the graph of LoopLevelIR nodes immediately after Inductor's fusion pass runs.

    The list of nodes is guarenteed by the caller to be in topological order.
    The returned list of nodes must also be in topological order.
    """

    def get_passes(self):
        return [spyre_fuse_nodes]


class CustomPreSchedulingPasses(CustomGraphPass):
    """
    Spyre-specific passes that run on IR operations immediately before the
    Scheduler is constructed (via the _update_scheduler monkey-patch).

    Operations are in topological order (guaranteed by GraphLowering).
    """

    def __call__(self, operations: list[Operation]) -> None:
        has_spyre_device = any(
            op.get_device() is not None and op.get_device().type == DEVICE_NAME
            for op in operations
        )
        if not has_spyre_device:
            return

        propagate_spyre_tensor_layouts(operations)
        core_division_planning(operations)
        if config.lx_planning:
            scratchpad_planning(operations)

    def uuid(self) -> Optional[Any]:
        files = [
            inspect.getfile(propagate_spyre_tensor_layouts),
            inspect.getfile(core_division_planning),
            inspect.getfile(scratchpad_planning),
        ]
        return get_hash_for_files(tuple(dict.fromkeys(files + [__file__])))
