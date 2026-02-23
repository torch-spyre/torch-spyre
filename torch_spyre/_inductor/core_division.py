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

from .errors import Unsupported
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


def get_host_dim_size(layout: FixedTiledLayout, host_dim_idx: int) -> int:
    """
    Get the size of an operation dimension from a tensor's host dimensions.

    Args:
        layout: The tensor's FixedTiledLayout
        host_dim_idx: The host dimension index (before canonicalization)

    Returns:
        The size of the operation dimension or number of sticks if it's
        the stick dimension
    """
    # layout.size is host size before canonicalization
    assert host_dim_idx < len(layout.size)

    if host_dim_idx != -1 and host_dim_idx != len(layout.size) - 1:
        return int(layout.size[host_dim_idx])
    else:  # stick dim
        return (
            int(layout.size[host_dim_idx])
            // layout.device_layout.device_dtype.elems_per_stick()
        )


def map_host_dim_to_device_dim(layout: FixedTiledLayout, host_dim_idx: int) -> int:
    """
    Map a tensor host dimension index to its corresponding device dimension(s).

    Args:
        layout: The tensor's FixedTiledLayout
        host_dim_idx:  The host dimension index (before canonicalization)

    Returns:
        The device dimension index that correspond to this operation dimension
    """
    # Assumptions:
    #   1. device layout is generic stick, so the last element is not considered
    #   2. dim_map elements have unique values
    dim_map = layout.device_layout.dim_map[:-1]
    return dim_map.index(host_dim_idx)


def divide_reduction_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    n.spyre_core_division = no_division(args, output)
    n.n_cores_used = 1

    if max_cores == 1:
        return

    if red.reduction_type == MATMUL_REDUCTION_OP:
        assert len(args) == 2, "matmul has exactly 2 input args"

        # Operation dimensions: [M, K] @ [K, N] --> [M, N]
        # Operation dimension indices: M=0, K=1, N=2

        # Get operation dimension sizes from host layouts
        M = get_host_dim_size(args[0].layout, 0)
        N = get_host_dim_size(args[1].layout, 1)

        # Parallelizable operation dimensions: M and N (not K, the reduction dim)
        sizes = [M, N]
        priorities = [2, 1]
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        # Create a mapping from operation dimension to its split count
        op_dim_splits = {"M": splits[0], "N": splits[1]}  # M: splits[0], N: splits[1]

        # Map operation dimension splits to device dimensions for each tensor
        # Safe to assume the dimension is not canonicalized if nsplit > 1, thus it must be mapped to an existing
        # device dim
        if op_dim_splits["M"] > 1:
            n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 0)] = (
                op_dim_splits["M"]
            )
            n.spyre_core_division[2][map_host_dim_to_device_dim(output, 0)] = (
                op_dim_splits["M"]
            )
        if op_dim_splits["N"] > 1:
            n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 1)] = (
                op_dim_splits["N"]
            )
            n.spyre_core_division[2][map_host_dim_to_device_dim(output, 1)] = (
                op_dim_splits["N"]
            )

    if red.reduction_type == BATCH_MATMUL_OP:
        assert len(args) == 2, "bmm has exactly 2 input args"

        # Determine if this is 3D or 4D BMM based on the number of dimensions
        num_dims = len(args[0].layout.size)

        if num_dims == 3:
            # 3D BMM: [B, M, K] @ [B, K, N] --> [B, M, N]
            # arg0 host layout: [B, M, K]
            # arg1 host layout: [B, K, N]
            # output host layout: [B, M, N]

            # Get operation dimension sizes from host layouts
            B = get_host_dim_size(args[0].layout, 0)
            M = get_host_dim_size(args[0].layout, 1)
            N = get_host_dim_size(args[1].layout, 2)

            # Parallelizable operation dimensions: B, M, N (not K, the reduction dim)
            sizes = [B, M, N]
            priorities = [3, 1, 2]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Create a mapping from operation dimension to its split count
            op_dim_splits = {"B": splits[0], "M": splits[1], "N": splits[2]}

            # Map operation dimension splits to device dimensions for each tensor
            # Safe to assume the dimension is not canonicalized if nsplit > 1, thus it must be mapped to an existing
            # device dim

            # arg0: [B, M, K] - B is host dim 0, M is host dim 1, K is host dim 2
            if op_dim_splits["B"] > 1:
                n.spyre_core_division[0][
                    map_host_dim_to_device_dim(args[0].layout, 0)
                ] = op_dim_splits["B"]
            if op_dim_splits["M"] > 1:
                n.spyre_core_division[0][
                    map_host_dim_to_device_dim(args[0].layout, 1)
                ] = op_dim_splits["M"]

            # arg1: [B, K, N] - B is host dim 0, K is host dim 1, N is host dim 2
            if op_dim_splits["B"] > 1:
                n.spyre_core_division[1][
                    map_host_dim_to_device_dim(args[1].layout, 0)
                ] = op_dim_splits["B"]
            if op_dim_splits["N"] > 1:
                n.spyre_core_division[1][
                    map_host_dim_to_device_dim(args[1].layout, 2)
                ] = op_dim_splits["N"]

            # output: [B, M, N] - B is host dim 0, M is host dim 1, N is host dim 2
            if op_dim_splits["B"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 0)] = (
                    op_dim_splits["B"]
                )
            if op_dim_splits["M"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 1)] = (
                    op_dim_splits["M"]
                )
            if op_dim_splits["N"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 2)] = (
                    op_dim_splits["N"]
                )

        elif num_dims == 4:
            # 4D BMM: [B1, B2, M, K] @ [B1, B2, K, N] --> [B1, B2, M, N]
            # arg0 host layout: [B1, B2, M, K]
            # arg1 host layout: [B1, B2, K, N]
            # output host layout: [B1, B2, M, N]

            # Get operation dimension sizes from host layouts
            B1 = get_host_dim_size(args[0].layout, 0)
            B2 = get_host_dim_size(args[0].layout, 1)
            M = get_host_dim_size(args[0].layout, 2)
            N = get_host_dim_size(args[1].layout, 3)

            # Parallelizable operation dimensions: B1, B2, M, N (not K, the reduction dim)
            sizes = [B1, B2, M, N]
            # NOTE: split priority can affect numerical error in unit tests
            priorities = [3, 4, 1, 2]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Create a mapping from operation dimension to its split count
            op_dim_splits = {
                "B1": splits[0],
                "B2": splits[1],
                "M": splits[2],
                "N": splits[3],
            }

            # Map operation dimension splits to device dimensions for each tensor
            # Safe to assume the dimension is not canonicalized if nsplit > 1, thus it must be mapped to an existing
            # device dim

            # arg0: [B1, B2, M, K] - B1 is host dim 0, B2 is host dim 1, M is host dim 2, K is host dim 3
            if op_dim_splits["B1"] > 1:
                n.spyre_core_division[0][
                    map_host_dim_to_device_dim(args[0].layout, 0)
                ] = op_dim_splits["B1"]
            if op_dim_splits["B2"] > 1:
                n.spyre_core_division[0][
                    map_host_dim_to_device_dim(args[0].layout, 1)
                ] = op_dim_splits["B2"]
            if op_dim_splits["M"] > 1:
                n.spyre_core_division[0][
                    map_host_dim_to_device_dim(args[0].layout, 2)
                ] = op_dim_splits["M"]

            # arg1: [B1, B2, K, N] - B1 is host dim 0, B2 is host dim 1, K is host dim 2, N is host dim 3
            if op_dim_splits["B1"] > 1:
                n.spyre_core_division[1][
                    map_host_dim_to_device_dim(args[1].layout, 0)
                ] = op_dim_splits["B1"]
            if op_dim_splits["B2"] > 1:
                n.spyre_core_division[1][
                    map_host_dim_to_device_dim(args[1].layout, 1)
                ] = op_dim_splits["B2"]
            if op_dim_splits["N"] > 1:
                n.spyre_core_division[1][
                    map_host_dim_to_device_dim(args[1].layout, 3)
                ] = op_dim_splits["N"]

            # output: [B1, B2, M, N] - B1 is host dim 0, B2 is host dim 1, M is host dim 2, N is host dim 3
            if op_dim_splits["B1"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 0)] = (
                    op_dim_splits["B1"]
                )
            if op_dim_splits["B2"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 1)] = (
                    op_dim_splits["B2"]
                )
            if op_dim_splits["M"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 2)] = (
                    op_dim_splits["M"]
                )
            if op_dim_splits["N"] > 1:
                n.spyre_core_division[2][map_host_dim_to_device_dim(output, 3)] = (
                    op_dim_splits["N"]
                )

        else:
            raise RuntimeError(f"Unsupported BMM dimension count: {num_dims}")


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
