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

import torch

from torch._inductor.ir import ComputedBuffer
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from . import config
from .constants import DEVICE_NAME
from .scheduler import CountedLoopSchedulerNode


def _make_fused(
    nodes: list[SchedulerNode | CountedLoopSchedulerNode],
) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return FusedSchedulerNode(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def _is_spyre_node(node: BaseSchedulerNode) -> bool:
    """True if the node computes on the Spyre device."""
    device = node.get_device()
    return device is not None and device.type == DEVICE_NAME


def _is_restickify_node(node: BaseSchedulerNode) -> bool:
    """True if the node is a restickify op inserted by insert_restickify.

    Restickify nodes must not be fused with surrounding ops.  When fused into a
    tiled attention SDSC the outer tile-loop variables inflate the SDSC iteration
    space, making the ReStickifyOpHBM initial chunk too large to fit in LX.

    A restickify node is a ComputedBuffer whose origins contain exactly one FX
    node targeting ``torch.ops.spyre.restickify.default``.
    """
    if not isinstance(node, SchedulerNode):
        return False
    op = node.node
    if not isinstance(op, ComputedBuffer):
        return False
    return any(
        getattr(fx_node, "target", None) is torch.ops.spyre.restickify.default
        for fx_node in op.origins
    )


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.

    Restickify nodes are never fused with other ops: they are emitted as
    standalone bundles so their SDSC iteration space contains only the
    restickify's own dimensions (not any outer tile-loop variables from a
    surrounding fused kernel), keeping the ReStickifyOpHBM initial chunk
    within the LX capacity limit.
    """
    if len(nodes) == 0:
        return nodes
    if not config.bundle_symbolic_args:
        # Without symbolic args, tensor addresses are baked-in constants from
        # SEGMENT_OFFSETS, which has a fixed number of slots.  Fusing ops could
        # exceed that slot count, so disable fusion when symbolic args are off.
        return nodes

    fused_nodes: list[BaseSchedulerNode] = []
    cur_nodes: list[SchedulerNode | CountedLoopSchedulerNode] = []

    for n in nodes:
        if _is_restickify_node(n):
            # Flush the current group, emit the restickify as its own bundle,
            # then start a fresh group for whatever follows.
            if fused := _make_fused(cur_nodes):
                fused_nodes.append(fused)
            cur_nodes = []
            fused_nodes.append(n)
        elif isinstance(n, (SchedulerNode, CountedLoopSchedulerNode)) and _is_spyre_node(
            n
        ):
            cur_nodes.append(n)
        else:
            # Non-Spyre nodes (Fallback nodes, CPU SchedulerNodes) force a
            # bundle boundary.
            if fused := _make_fused(cur_nodes):
                fused_nodes.append(fused)
            fused_nodes.append(n)
            cur_nodes = []

    if fused := _make_fused(cur_nodes):
        fused_nodes.append(fused)

    return fused_nodes
