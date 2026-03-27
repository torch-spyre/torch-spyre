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
from sympy import Expr, Symbol

import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
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
from .pass_utils import SchedNodeArg, get_mem_deps, device_coordinates, iteration_space
from .logging_utils import get_inductor_logger
from .work_division_utils import multi_dim_core_split, multi_dim_iteration_space_split
import logging

logger = get_inductor_logger("core_division")


aten = torch.ops.aten
spyreop = torch.ops.spyre


def get_host_dim_size(layout: FixedTiledLayout, host_dim_idx: int) -> int:
    """
    Get the parallelizable size of a host dimension.

    For non-stick dimensions this is simply the dimension size. For the stick
    dimension (the last host dimension), the elements are packed into sticks, so
    the parallelizable unit is the number of sticks rather than the number of
    elements.

    This function properly consults the dim_map to find which device dimension
    corresponds to the requested host dimension, handling tiling and sparse tensors.

    Args:
        layout: The tensor's FixedTiledLayout
        host_dim_idx: The host dimension index (negative indices are supported)

    Returns:
        The number of parallelizable units along this dimension
    """
    if host_dim_idx < 0:
        host_dim_idx = len(layout.size) + host_dim_idx

    assert host_dim_idx < len(layout.size)

    dl = layout.device_layout

    # Use dim_map to find the device dimension that corresponds to this host dimension
    # For tiled dimensions (appearing multiple times in dim_map), we use the first occurrence
    # which corresponds to the outermost device dimension for that host dimension
    try:
        device_dim_idx = dl.dim_map.index(host_dim_idx)
    except ValueError:
        raise RuntimeError(
            f"Host dimension {host_dim_idx} not found in dim_map {dl.dim_map}"
        )

    return dl.device_size[device_dim_idx]


def prioritize_dimensions(
    coords: list[Expr], iteration_space: dict[Symbol, Expr]
) -> list[Symbol]:
    """
    Return a list of the free variables in coords in the order they should be considered for core division.
    The order combines two considerations:
      1. If the iteration space is large, prioritize outer dimensions to reduce span-per-core
      2. After reducing the span, order by size of the dimension to maximize parallelism.
    """
    span = 1
    for e in iteration_space.values():
        span *= e

    priority = []
    # TODO: Don't hardwire this heuristic limit
    while span > 32 * 1024 * 1024:
        for e in coords:
            vars = e.free_symbols
            for v in vars:
                if v not in priority:
                    priority.append(v)
                    span /= iteration_space[v]

    # Prioritize all remaining dimensions by sorting them in decreasing size
    remaining = [(s, e) for s, e in iteration_space.items() if s not in priority]
    remaining.sort(key=lambda t: t[1], reverse=True)
    priority += [t[0] for t in remaining]

    return priority


def divide_pointwise_op_new(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    if max_cores == 1:
        return

    it_space = iteration_space(n)
    output_layout: FixedTiledLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    output_dev_coords = device_coordinates(output_layout, output_dep)
    stick_expr = output_dev_coords[-1]
    if len(stick_expr.free_symbols) != 1:
        # TODO: Can codegen handle core division for sparse tensors?
        return

    # Adjust the size of the stick dimension iteration space to be in sticks, not elements
    stick_var = next(iter(stick_expr.free_symbols))
    elems_per_stick = output_layout.device_layout.elems_per_stick()
    it_space[stick_var] = (it_space[stick_var] + elems_per_stick - 1) // elems_per_stick

    # Do the core division for this operation
    priorities = prioritize_dimensions(output_dev_coords[:-1], it_space)
    splits = multi_dim_iteration_space_split(it_space, max_cores, priorities)

    cores_used = math.prod(splits.values())

    if cores_used > 1:
        n.op_it_space_splits = splits

        # Consolidated DEBUG log for pointwise work division
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"pointwise work_division {n.node.get_name()}: cores={n.n_cores_used}, "
                f"iteration_space={it_space}, priorities={priorities}, op_it_space_splits={n.op_it_space_splits}"
            )


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    output: FixedTiledLayout = n.node.get_layout()
    ndim = len(output.size)
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

    # Collect parallelizable sizes for all host dimensions
    # For stick dimension: this returns the number of sticks
    # For non-stick dimensions: this returns the dimension size
    sizes = [get_host_dim_size(output, i) for i in range(ndim)]

    # Use sizes as priorities (larger dimensions get higher priority)
    priorities = sizes.copy()

    # Use multi-dimensional core splitting
    splits = multi_dim_core_split(sizes, max_cores, priorities)
    n.n_cores_used = math.prod(splits)

    if n.n_cores_used > 1:
        n.op_dim_splits = splits

        # Consolidated DEBUG log for pointwise work division
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"pointwise work_division {n.node.get_name()}: cores={n.n_cores_used}, "
                f"sizes={sizes}, priorities={priorities}, op_dim_splits={n.op_dim_splits}"
            )


def divide_reduction_op(
    n: SchedulerNode, args: list[SchedNodeArg], max_cores, enable_splitk=True
):
    red: Reduction = n.node.data
    n.n_cores_used = 1

    if max_cores == 1:
        return

    if red.reduction_type == MATMUL_REDUCTION_OP:
        assert len(args) == 2, "matmul has exactly 2 input args"

        # Operation dimensions: [M, K] @ [K, N] --> [M, N]
        # dim_labels in codegen: ["mb", "in", "out"] = [M, K, N]

        # Get operation dimension sizes from host layouts.
        M = get_host_dim_size(args[0].layout, 0)
        K = get_host_dim_size(args[0].layout, 1)
        N = get_host_dim_size(args[1].layout, 1)

        # Parallelizable operation dimensions: M, K, and N
        # K has lowest priority (1) - only split when M and N are exhausted
        # Use negative priority to exclude K from splitting when splitk is disabled
        sizes = [M, K, N]
        priorities = [3, 1 if enable_splitk else -1, 2]
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        # Store op_dim_splits directly matching dim_labels = ["mb", "in", "out"]
        n.op_dim_splits = splits

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"matmul work_division: M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                f"splits=[M={splits[0]}, K={splits[1]}, N={splits[2]}]"
            )

    if red.reduction_type == BATCH_MATMUL_OP:
        assert len(args) == 2, "bmm has exactly 2 input args"

        # Determine if this is 3D or 4D BMM based on the number of dimensions
        num_dims = len(args[0].layout.size)

        if num_dims == 3:
            # 3D BMM: [B, M, K] @ [B, K, N] --> [B, M, N]
            #     or  [B, M, K] @ [K, N] --> [B, M, N]
            # dim_labels in codegen: ["x", "mb", "in", "out"] = [B, M, K, N]

            # Get operation dimension sizes from host layouts
            B = get_host_dim_size(args[0].layout, 0)
            M = get_host_dim_size(args[0].layout, 1)
            K = get_host_dim_size(args[0].layout, 2)
            N = get_host_dim_size(args[1].layout, -1)

            # Parallelizable operation dimensions: B, M, K, and N
            # K has lowest priority (1) - only split when B, M, and N are exhausted
            # Use negative priority to exclude K from splitting when splitk is disabled
            sizes = [B, M, K, N]
            priorities = [4, 2, 1 if enable_splitk else -1, 3]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Store op_dim_splits directly matching dim_labels = ["x", "mb", "in", "out"]
            n.op_dim_splits = splits

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"bmm_3d work_division: B={B}, M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                    f"splits=[B={splits[0]}, M={splits[1]}, K={splits[2]}, N={splits[3]}]"
                )

        elif num_dims == 4:
            # 4D BMM: [B1, B2, M, K] @ [B1, B2, K, N] --> [B1, B2, M, N]
            # dim_labels in codegen: ["x", "y", "mb", "in", "out"] = [B1, B2, M, K, N]

            # Get operation dimension sizes from host layouts
            B1 = get_host_dim_size(args[0].layout, 0)
            B2 = get_host_dim_size(args[0].layout, 1)
            M = get_host_dim_size(args[0].layout, 2)
            K = get_host_dim_size(args[0].layout, 3)
            N = get_host_dim_size(args[1].layout, -1)

            # Parallelizable operation dimensions: B1, B2, M, K, and N
            # K has lowest priority (1) - only split when B1, B2, M, and N are exhausted
            # Use negative priority to exclude K from splitting when splitk is disabled
            # NOTE: split priority can affect numerical error in unit tests
            sizes = [B1, B2, M, K, N]
            priorities = [4, 5, 2, 1 if enable_splitk else -1, 3]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Store op_dim_splits directly matching dim_labels = ["x", "y", "mb", "in", "out"]
            n.op_dim_splits = splits

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"bmm_4d work_division: B1={B1}, B2={B2}, M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                    f"splits=[B1={splits[0]}, B2={splits[1]}, M={splits[2]}, K={splits[3]}, N={splits[4]}]"
                )

        else:
            raise RuntimeError(f"Unsupported BMM dimension count: {num_dims}")


def core_division_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guaranteed by caller).
    max_cores = int(os.getenv("SENCORES", "32"))
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")

    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            if isinstance(n.node.data, Pointwise):
                divide_pointwise_op(n, get_mem_deps(n), max_cores)
                divide_pointwise_op_new(n, get_mem_deps(n), max_cores)
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
                logger.warning(f"unhandled node type {type(n.node)}")
        elif isinstance(n, NopKernelSchedulerNode):
            pass
        else:
            logger.warning(f"unhandled scheduler node type {type(n)}")

    return nodes
