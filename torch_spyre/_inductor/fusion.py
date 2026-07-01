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
from torch._inductor.virtualized import V
from torch._inductor.ir import FallbackKernel
from torch_spyre._inductor.logging_utils import _get_env_bool
from .ir import FixedTiledLayout
from .constants import SEGMENT_OFFSETS
from .scheduler import CountedLoopSchedulerNode

# TODO: Temporary hook to easily disable
_FUSION_ENABLED = _get_env_bool("SPYRE_INDUCTOR_ENABLE_FUSION", True)


def _max_bundle_tensors() -> int:
    # Until https://github.com/torch-spyre/torch-spyre/issues/827 is completed.
    has_pool = getattr(V.graph, "pool_size", 0) > 0
    return len(SEGMENT_OFFSETS) - (2 if has_pool else 1)


def _make_fused(nodes: list[SchedulerNode]) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return FusedSchedulerNode(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def _is_non_intermediate(name: str) -> bool:
    buf = V.graph.get_buffer(name)
    if buf is None or isinstance(buf, FallbackKernel):
        return False
    # FallbackKernel may register companion buffers with NoneLayout /
    # MultiOutputLayout (MutationOutput sentinels for void/in-place ops, the
    # MultiOutputLayout buffer for tuple ops). These have no real tensor
    # layout, can't be FixedTiledLayout, and shouldn't count toward the
    # bundle's non-intermediate tensor budget.
    layout = buf.maybe_get_layout()
    return isinstance(layout, FixedTiledLayout) and not layout.allocation


def _count_non_intermediate_tensors(node: BaseSchedulerNode) -> int:
    """Count unique non-intermediate tensors referenced by node.

    For a CountedLoopSchedulerNode, node.read_writes is the recursively
    merged union of all inner nodes' read_writes (built by
    FusedSchedulerNode.__init__ → init_group_node →
    ReadWrites.merge_list).  Nested CountedLoopSchedulerNodes therefore
    contribute their full tensor sets automatically; no manual recursion
    is needed here.
    """
    names = {dep.name for dep in node.read_writes.reads_and_writes()}
    return sum(1 for name in names if _is_non_intermediate(name))


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.
    Fusion is limited by the following constraints.
     1. We only want to fuse SchedulerNodes (ie, nodes that generate OpSpecs).
     2. A SDSC Bundle can refer to at most 5 unique non-intermediate tensors
        (graph inputs/outputs). Intermediates don't count toward this limit.
    """
    if not _FUSION_ENABLED or len(nodes) == 0:
        return nodes

    max_tensors = _max_bundle_tensors()
    fused_nodes: list[BaseSchedulerNode] = []
    cur_nodes: list[SchedulerNode] = []
    cur_tensors: set[str] = set()
    cur_non_intermediate_count: int = 0

    for n in nodes:
        if isinstance(n, SchedulerNode):
            n_tensors = {dep.name for dep in n.read_writes.reads_and_writes()}
            new_tensors = n_tensors - cur_tensors
            new_non_intermediate = sum(
                1 for t in new_tensors if _is_non_intermediate(t)
            )
            if cur_non_intermediate_count + new_non_intermediate <= max_tensors:
                # Ok to put in the current bundle
                cur_nodes.append(n)
                cur_tensors |= n_tensors
                cur_non_intermediate_count += new_non_intermediate
            else:
                # Would be too many non-intermediate tensors; start a new bundle.
                if fused := _make_fused(cur_nodes):
                    fused_nodes.append(fused)
                cur_nodes = [n]
                cur_tensors = n_tensors
                cur_non_intermediate_count = sum(
                    1 for t in n_tensors if _is_non_intermediate(t)
                )

        else:
            # Other node types (eg Fallback nodes, CountedLoopSchedulerNode)
            # force a bundle boundary.  For CountedLoopSchedulerNodes (atomic
            # loop groups), verify the tensor budget since they cannot be split.
            if fused := _make_fused(cur_nodes):
                fused_nodes.append(fused)
            if isinstance(n, CountedLoopSchedulerNode):
                non_intermediate = _count_non_intermediate_tensors(n)
                if non_intermediate > max_tensors:
                    raise RuntimeError(
                        f"spyre_fuse_nodes: node {n.get_name()!r} references "
                        f"{non_intermediate} non-intermediate tensors but the "
                        f"bundle limit is {max_tensors}"
                    )
            fused_nodes.append(n)
            cur_nodes = []
            cur_tensors = set()
            cur_non_intermediate_count = 0

    if fused := _make_fused(cur_nodes):
        fused_nodes.append(fused)

    return fused_nodes
