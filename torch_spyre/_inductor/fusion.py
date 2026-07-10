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

from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from . import config
from .scheduler import CountedLoopSchedulerNode


def _op_count(node: BaseSchedulerNode) -> int:
    """Number of leaf SchedulerNodes represented by ``node``."""
    return len(node.get_nodes())


def _can_extend_bundle(
    cur_nodes: list[SchedulerNode | CountedLoopSchedulerNode],
    candidate: SchedulerNode | CountedLoopSchedulerNode,
) -> bool:
    """Return True if ``candidate`` may join the current (non-empty) bundle.

    Phase-1 policy (see the design doc):
      * Reduction isolation: a reduction op is never fused with anything else,
        keeping RMSNorm reduction dims out of a kernel that also holds GQA
        broadcast / KV-cache transpose dims (the DDL-unmappable combination).
      * Op-count cap: a bundle holds at most ``config.max_fused_ops`` leaf ops.
    """
    if candidate.is_reduction():
        return False
    if any(n.is_reduction() for n in cur_nodes):
        return False
    total = sum(_op_count(n) for n in cur_nodes) + _op_count(candidate)
    return total <= config.max_fused_ops


def _make_fused(
    nodes: list[SchedulerNode | CountedLoopSchedulerNode],
) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return FusedSchedulerNode(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.
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
        if isinstance(n, (SchedulerNode, CountedLoopSchedulerNode)):
            cur_nodes.append(n)
        else:
            # Other node types (eg Fallback nodes) force a bundle boundary.
            if fused := _make_fused(cur_nodes):
                fused_nodes.append(fused)
            fused_nodes.append(n)
            cur_nodes = []

    if fused := _make_fused(cur_nodes):
        fused_nodes.append(fused)

    return fused_nodes
