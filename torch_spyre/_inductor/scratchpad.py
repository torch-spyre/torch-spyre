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

import math
from torch._inductor.ir import (
    ComputedBuffer,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)

from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._C import SpyreTensorLayout


class ScratchPadAllocator:
    """
    A trivial bump pointer allocator
    """

    def __init__(self, size: int):
        self.current = 0
        self.limit = size

    def try_allocate(self, stl: SpyreTensorLayout) -> tuple[bool, int]:
        num_sticks = math.prod(stl.device_size[:-1])
        bytes = num_sticks * 128
        if self.current + bytes < self.limit:
            alloc = self.current
            self.current += bytes
            return (True, alloc)
        else:
            return (False, -1)


def consider_for_scratchpad(
    n: SchedulerNode, buf: ComputedBuffer, alloc: ScratchPadAllocator
):
    layout: FixedTiledLayout = buf.layout
    fits, offset = alloc.try_allocate(layout.device_layout)
    if fits:
        layout.allocation["lx"] = offset


def scratchpad_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    # Work division has already been done.
    # Stickification has already been done (therefore all ComputedBeffers have FixedTiledLayouts)

    alloc = ScratchPadAllocator(1024 * 1024)  # TODO -- don't hardwire size

    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            consider_for_scratchpad(n, n.node, alloc)

    return nodes
