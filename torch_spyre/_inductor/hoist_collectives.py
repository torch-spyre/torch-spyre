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

from torch._inductor.scheduler import BaseSchedulerNode, ExternKernelSchedulerNode

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("hoist_collectives")


def _get_async_collective_types() -> tuple:
    """Return the IR node types that represent async collective ops.

    Uses lazy import to tolerate branches where some classes don't exist yet.
    """
    from .ir import BroadcastAsyncFallback, WaitWorkFallback  # noqa: F401

    types: list[type] = [BroadcastAsyncFallback]

    try:
        from .ir import AllReduceAsyncFallback

        types.append(AllReduceAsyncFallback)
    except ImportError:
        pass

    try:
        from .ir import ReduceAsyncFallback

        types.append(ReduceAsyncFallback)
    except ImportError:
        pass

    try:
        from .ir import AllGatherAsyncFallback

        types.append(AllGatherAsyncFallback)
    except ImportError:
        pass

    return tuple(types)


def _is_async_collective(node: BaseSchedulerNode) -> bool:
    """Check whether a scheduler node wraps an async collective IR node."""
    if not isinstance(node, ExternKernelSchedulerNode):
        return False
    ir_node = node.node
    return isinstance(ir_node, _get_async_collective_types())


def _build_name_to_idx(nodes: list[BaseSchedulerNode]) -> dict[str, int]:
    """Build a mapping from buffer name to its position in the node list."""
    name_to_idx: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for name in node.get_buffer_names():
            name_to_idx[name] = i
    return name_to_idx


def hoist_collective_ops(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Reorder nodes to place async collective ops as early as possible.

    For each async collective (broadcast_async, all_reduce_async,
    all_gather_async), move it forward in the schedule to immediately after
    its last dependency is satisfied. This maximizes the window for
    communication-compute overlap before the corresponding wait_work.

    Maintains valid topological order: a node is only moved earlier than its
    current position, and never before any node that produces a buffer it
    depends on.
    """
    result = list(nodes)
    name_to_idx = _build_name_to_idx(result)

    i = 0
    while i < len(result):
        node = result[i]
        if not _is_async_collective(node):
            i += 1
            continue

        # Compute the earliest valid position: just after the last dependency.
        earliest_valid = 0
        for dep in node.unmet_dependencies:
            dep_idx = name_to_idx.get(dep.name)
            if dep_idx is not None:
                earliest_valid = max(earliest_valid, dep_idx + 1)

        if earliest_valid < i:
            logger.debug(
                "Hoisting %s from position %d to %d",
                node.get_name(),
                i,
                earliest_valid,
            )
            result.pop(i)
            result.insert(earliest_valid, node)
            # Rebuild index map since positions shifted.
            name_to_idx = _build_name_to_idx(result)
        else:
            i += 1
            continue

        # Advance past the node we just placed.
        i = earliest_valid + 1

    return result
