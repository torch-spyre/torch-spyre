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
    NopKernelSchedulerNode,
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


def core_split(size: int, max_cores: int) -> int:
    """
    Find the largest divisor of size that doesn't exceed max_cores.

    Args:
        size: The dimension size to split
        max_cores: Maximum number of cores to use for this dimension

    Returns:
        Number of cores to use (always divides size evenly)
    """
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i
    return 1


def multi_dim_core_split(
    sizes: list[int], max_cores: int, priorities: list[int] | None = None
) -> list[int]:
    """
    Distribute max_cores across multiple dimensions optimally.

    This function tries to split cores across multiple dimensions to maximize
    parallelism while ensuring even division. It uses a greedy approach that
    prioritizes dimensions based on:
    1. User-specified priorities (if provided)
    2. Dimension size (larger dimensions get priority)
    3. Divisibility (dimensions that divide evenly get priority)

    Args:
        sizes: List of dimension sizes that can be parallelized
        max_cores: Total number of cores available
        priorities: Optional list of priority values (higher = more important)
                   If None, uses dimension sizes as priorities

    Returns:
        List of core splits for each dimension (same length as sizes)
        The product of all splits will be <= max_cores

    Example:
        >>> multi_dim_core_split([128, 64, 32], max_cores=8)
        [4, 2, 1]  # 4*2*1 = 8 cores total

        >>> multi_dim_core_split([100, 50], max_cores=10)
        [5, 2]  # 5*2 = 10 cores total
    """
    if not sizes:
        return []

    n_dims = len(sizes)
    splits = [1] * n_dims

    # Use provided priorities or default to the sizes of dimensions
    if priorities is None:
        priorities = sizes.copy()

    # Create list of (dimension_index, size, priority) tuples
    dim_info = [(i, sizes[i], priorities[i]) for i in range(n_dims)]

    # Sort by priority (descending), then by size (descending)
    dim_info.sort(key=lambda x: (x[2], x[1]), reverse=True)

    n_cores_to_split = max_cores

    # Greedy allocation: try to split highest priority dimensions first
    for dim_idx, size, _ in dim_info:
        if n_cores_to_split <= 1:
            break

        # Find the best split for this dimension given n_cores_to_split
        best_split = core_split(size, n_cores_to_split)

        if best_split > 1:
            splits[dim_idx] = best_split
            n_cores_to_split = n_cores_to_split // best_split

    return splits


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    # pw: Pointwise = n.node.data
    # op = pw.get_origin_node().target
    output: FixedTiledLayout = n.node.get_layout()
    n.spyre_core_division = no_division(args, output)
    n.n_cores_used = 1

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
        n.n_cores_used = num_cores
        for cd in n.spyre_core_division:
            cd[split_idx] = num_cores


def divide_reduction_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    n.spyre_core_division = no_division(args, output)
    n.n_cores_used = 1

    if max_cores == 1:
        return

    device_size = output.device_layout.device_size

    if red.reduction_type == MATMUL_REDUCTION_OP:
        assert len(args) == 2, "matmul has exactly 2 input args"

        # [M, K] @ [K, N] --> [M, N]

        # For MATMUL, we can split along output dimensions
        # Typically device_size is [N//64, M, 64]
        # We want to split M and possibly N//64
        # Choose dimensions to parallelize (exclude stick dimension -1)
        parallelizable_dims = [-3, -2]

        # Compute the splits
        sizes = [device_size[dim] for dim in parallelizable_dims]
        # Prioritize M dimension over N dimension
        priorities = [1, 2]
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        # Assign split values accordingly
        # arg 0: [K//64, M, 64]
        n.spyre_core_division[0][1] = splits[1]  # assign M split
        # arg 1: [N//64, K, 64]
        n.spyre_core_division[1][0] = splits[0]  # assign N split
        # output: [N//64, M, 64]
        n.spyre_core_division[2][0] = splits[0]  # assign N split
        n.spyre_core_division[2][1] = splits[1]  # assign M split

    if red.reduction_type == BATCH_MATMUL_OP and len(device_size) == 4:  # 3d bmm
        assert len(args) == 2, "bmm has exactly 2 input args"

        # Logical: [x, mb, in] @ [x, in, out] --> [x, mb, out]
        # where x=batch, mb=M, in=K, out=N

        # Device layout (3D rule: [x, mb, out] -> [mb, out//64, x, 64]):
        # Input 0:  [mb, in//64, x, 64]
        # Input 1:  [in, out//64, x, 64]
        # Output:   [mb, out//64, x, 64]

        # Choose dimensions to parallelize (exclude stick dimension -1)
        parallelizable_dims = [0, 1, 2]  # mb, out//64, x

        # Compute the splits
        sizes = [device_size[dim] for dim in parallelizable_dims]
        # Prioritize: x > out > mb
        priorities = [1, 2, 3]  # mb=1 (lowest), out=2, x=3 (highest)
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        # Assign split values accordingly
        # arg 0 device layout: [mb, in//64, x, 64]
        n.spyre_core_division[0][0] = splits[0]  # assign mb split
        n.spyre_core_division[0][2] = splits[2]  # assign x split
        # arg 1 device layout: [in, out//64, x, 64]
        n.spyre_core_division[1][1] = splits[1]  # assign out split
        n.spyre_core_division[1][2] = splits[2]  # assign x split
        # output device layout: [mb, out//64, x, 64]
        n.spyre_core_division[2][0] = splits[0]  # assign mb split
        n.spyre_core_division[2][1] = splits[1]  # assign out split
        n.spyre_core_division[2][2] = splits[2]  # assign x split

    if red.reduction_type == BATCH_MATMUL_OP and len(device_size) == 5:  # 4d bmm
        assert len(args) == 2, "bmm has exactly 2 input args"
        parallelizable_dims = [1, 2, 3, 0]  # mb, out//64, x, y

        # Compute the splits
        sizes = [device_size[dim] for dim in parallelizable_dims]

        # Prioritize: y > x > out > mb
        priorities = [1, 2, 3, 4]  # mb=1 (lowest), out=2, x=3, y=4 (highest)
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        n.spyre_core_division[0][0] = splits[2]  # assign x split
        n.spyre_core_division[0][1] = splits[3]  # assign y split
        n.spyre_core_division[0][2] = splits[0]  # assign mb split

        n.spyre_core_division[1][2] = splits[1]  # assign out split
        n.spyre_core_division[1][0] = splits[2]  # assign x split
        n.spyre_core_division[1][1] = splits[3]  # assign y split

        n.spyre_core_division[2][2] = splits[0]  # assign mb split
        n.spyre_core_division[2][3] = splits[1]  # assign out split
        n.spyre_core_division[2][0] = splits[2]  # assign x split
        n.spyre_core_division[2][1] = splits[3]  # assign y split


def core_division_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    max_cores = int(os.getenv("SENCORES", "32"))
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
        elif isinstance(n, NopKernelSchedulerNode):
            pass
        else:
            print(f"Warning: unhandled scheduler node type {type(n)}")

    return nodes
