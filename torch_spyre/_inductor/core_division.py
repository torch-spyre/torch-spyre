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


import os
import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
    FixedLayout,
    MultiOutput,
    Pointwise,
    Reduction,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    SchedulerNode,
)

from . import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps


aten = torch.ops.aten
spyreop = torch.ops.spyre


def no_division(args: list[SchedNodeArg], output: FixedTiledLayout) -> list[list[int]]:
    result = []
    for a in args:
        result.append([1] * len(a.layout.device_layout.device_size))
    result.append([1] * len(output.device_layout.device_size))
    return result


def core_split(size, max_cores):
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    # pw: Pointwise = n.node.data
    # op = pw.get_origin_node().target
    output: FixedTiledLayout = n.node.get_layout()
    n.spyre_core_division = no_division(args, output)

    if max_cores == 1:
        return

    if len(n.node.get_outputs()) > 2:
        # Core division currently only implemented for 1 or 2 tensors
        return

    for a in args:
        if a.layout.size != output.size:
            # Core division not supported if there are broadcasts
            return

    device_size = output.device_layout.device_size
    split_idx = -3 if len(device_size) == 4 else 0  # split along stick dim
    num_cores = core_split(device_size[split_idx], max_cores)
    if num_cores > 1:
        for cd in n.spyre_core_division:
            cd[split_idx] = num_cores


def divide_reduction_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    n.spyre_core_division = no_division(args, output)

    if max_cores == 1:
        return

    if red.reduction_type == MATMUL_REDUCTION_OP:
        device_size = output.device_layout.device_size
        num_cores = core_split(device_size[-3], max_cores)
        if num_cores > 1:
            for cd in n.spyre_core_division:
                cd[-3] = num_cores

    if red.reduction_type == BATCH_MATMUL_OP:
        # [mb, out//64, x, 64]
        device_size = output.device_layout.device_size
        # try split along mb first
        mb_nsplit = core_split(device_size[0], max_cores)
        if mb_nsplit > 1:
            for cd in n.spyre_core_division:
                cd[0] = mb_nsplit


def core_division_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    max_cores = int(os.getenv("SENCORES", "1"))
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")

    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            if isinstance(n.node.data, Pointwise):
                divide_pointwise_op(n, get_mem_deps(n), max_cores)
            elif isinstance(n.node.data, Reduction):
                divide_reduction_op(n, get_mem_deps(n), max_cores)
            else:
                # Core division not supported on other IRNode types
                pass
        elif isinstance(n, ExternKernelSchedulerNode):
            if isinstance(n.node, FallbackKernel):
                n = next(it, None)
                if not (
                    isinstance(n, ExternKernelSchedulerNode)
                    and isinstance(n.node, MultiOutput)
                ):
                    raise RuntimeError("FallbackKernel must be followed by MultiOutput")

                # Core division not supported on fallback kernels
                pass
            else:
                print(f"Warning: unhandled node type {type(n.node)}")
        else:
            print(f"Warning: unhandled scheduler node type {type(n)}")

    return nodes
