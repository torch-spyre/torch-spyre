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

from torch_spyre._inductor.logging_utils import _get_env_bool

# Temporary hook to easily disable
_FUSION_ENABLED = _get_env_bool("SPYRE_INDUCTOR_ENABLE_FUSION", True)


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.
    Fusion is limited by the following constraints.
     1. We only want to fuse SchedulerNodes (ie, nodes that generate OpSpecs).
     2. A SDSC Bundle can refer to at most 6 unique tensors (until we complete  https://github.com/torch-spyre/torch-spyre/issues/827).
    """
    if not _FUSION_ENABLED or len(nodes) == 0:
        return nodes

    fused_nodes: list[BaseSchedulerNode] = []
    cur_nodes: list[SchedulerNode] = []
    cur_tensors: set[str] = set()
    for n in nodes:
        if isinstance(n, SchedulerNode):
            n_tensors = {dep.name for dep in n.read_writes.reads_and_writes()}
            cur_tensors |= n_tensors
            if len(cur_tensors) < 7:
                # Ok to put in the current bundle
                cur_nodes.append(n)
            else:
                # Would be too many tensors in the Bundle; start a new one.
                fused_nodes.append(
                    FusedSchedulerNode(cur_nodes[0].scheduler, cur_nodes)
                )
                cur_nodes = [n]
                cur_tensors = n_tensors

        else:
            # Other node types (eg Fallback nodes) force a bundle boundary.
            if len(cur_nodes) > 0:
                fused_nodes.append(
                    FusedSchedulerNode(cur_nodes[0].scheduler, cur_nodes)
                )
            fused_nodes.append(n)
            cur_nodes = []
            cur_tensors = set()

    if len(cur_nodes) > 0:
        fused_nodes.append(FusedSchedulerNode(cur_nodes[0].scheduler, cur_nodes))

    return fused_nodes
