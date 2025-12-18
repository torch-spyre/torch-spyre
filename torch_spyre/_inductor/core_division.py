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


import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    Pointwise,
    Reduction,
)
from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode

from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps


aten = torch.ops.aten
spyreop = torch.ops.spyre


def no_division(args: list[SchedNodeArg], output: FixedTiledLayout) -> list[list[int]]:
    result = []
    for a in args:
        result.append([1] * len(a.layout.size))
    return result


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg]):
    # pw: Pointwise = n.node.data
    # op = pw.get_origin_node().target
    output: FixedTiledLayout = n.node.get_layout()

    division = no_division(args, output)
    # TODO: Here is where we look for cases we support and update divsion

    n.spyre_core_division = division


def divide_reduction_op(n: SchedulerNode, args: list[SchedNodeArg]):
    # red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()

    division = no_division(args, output)
    # TODO: Here is where we look for cases we support and update divsion

    n.spyre_core_division = division


def core_division_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    for n in nodes:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            if isinstance(n.node.data, Pointwise):
                divide_pointwise_op(n, get_mem_deps(n))
            elif isinstance(n.node.data, Reduction):
                divide_reduction_op(n, get_mem_deps(n))
            else:
                # Core division not supported on other IRNode types
                pass

        else:
            print(f"Warning: unhandled scheduler node type {type(n)}")

    return nodes
