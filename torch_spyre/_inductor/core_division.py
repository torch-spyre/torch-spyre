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


import dataclasses
import math
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

from torch._inductor.dependencies import MemoryDep

from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps, device_coordinates, iteration_space
from .logging_utils import get_inductor_logger
from . import config
import logging

logger = get_inductor_logger("core_division")

# Maximum memory access span per core: 256MB hardware limit
MAX_SPAN_BYTES = 256 * 1024 * 1024
MAX_SPAN_STICKS = MAX_SPAN_BYTES // 128

aten = torch.ops.aten
spyreop = torch.ops.spyre


@dataclasses.dataclass
class TensorDep:
    """Bundles a MemoryDep with its FixedTiledLayout and pre-computes device coordinates."""

    dep: MemoryDep
    layout: FixedTiledLayout
    device_coords: list[Expr] = dataclasses.field(init=False)

    def __post_init__(self):
        self.device_coords = device_coordinates(self.layout, self.dep)


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


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    priorities: list[Symbol],
    min_splits: dict[Symbol, int] | None = None,
) -> dict[Symbol, int]:
    """
    Distribute max_cores across multiple dimensions of an iteration space.

    This function tries to split cores across multiple dimensions to maximize
    parallelism while ensuring even division. It uses a two-pass approach:
    1. First pass: satisfy minimum split requirements (hardware constraints)
    2. Second pass: distribute remaining cores by priority

    Args:
        iteration_space: The iteration space to be parallelized
        max_cores: Total number of cores available
        priorities: Order in which to consider the dimensions
        min_splits: Minimum splits required for each dimension (optional)

    Returns:
        The core splits for the iteration_space
        The product of all splits will be <= max_cores
    """
    splits = {v: 1 for v in iteration_space.keys()}
    n_cores_remaining = max_cores

    # First pass: satisfy minimum split requirements
    if min_splits:
        for var, min_split in min_splits.items():
            assert var not in priorities  # there shouldn't be an overlap

            # Check if we have enough cores for this minimum split
            if n_cores_remaining // min_split <= 0:
                logger.critical(
                    f"Cannot satisfy minimum split requirement for {var}: "
                    f"need {min_split} splits but only {n_cores_remaining} cores remaining. "
                    f"Skipping this constraint - hardware span limit may be violated."
                )
                continue  # Skip this variable, leave splits[var] = 1

            # Safe to apply the minimum split
            splits[var] = min_split
            n_cores_remaining = n_cores_remaining // min_split

    # Second pass: distribute remaining cores by priority
    for v in priorities:
        if n_cores_remaining <= 1:
            break

        best_split = core_split(iteration_space[v], n_cores_remaining)
        if best_split > 1:
            splits[v] = best_split
            n_cores_remaining = n_cores_remaining // best_split

    return splits


def adjust_it_space_for_sticks(
    it_space: dict[Symbol, Expr],
    tensor_deps: list[TensorDep],
) -> None:
    """Adjust iteration space sizes to count sticks rather than elements.

    For each tensor, find the variable that indexes its stick dimension and
    convert its size in it_space from elements to sticks. This ensures core
    division treats sticks as atomic units. Adjusts each variable at most once.
    """
    adjusted: dict[Symbol, int] = {}  # stick_var -> elems_per_stick used
    for td in tensor_deps:
        stick_expr = td.device_coords[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        if stick_var not in it_space:
            continue
        elems_per_stick = td.layout.device_layout.elems_per_stick()
        if stick_var in adjusted:
            assert adjusted[stick_var] == elems_per_stick, (
                f"Conflicting elems_per_stick for iteration variable {stick_var}: "
                f"previously seen {adjusted[stick_var]}, now {elems_per_stick}. "
                f"Mixed-dtype tensors sharing a stick variable are not supported."
            )
            continue
        # FIXME: here we assume padding to a full stick. It may not always be the
        #        case and we shouldn use a more robust way of computing the number
        #        of sticks
        it_space[stick_var] = (
            it_space[stick_var] + elems_per_stick - 1
        ) // elems_per_stick
        adjusted[stick_var] = elems_per_stick


def must_split_vars(
    tensor_deps: list[TensorDep] | None,
    it_space_adjusted: dict[Symbol, Expr],
) -> dict[Symbol, int]:
    """Return the minimum splits required per iteration variable to keep each
    tensor's memory span within MAX_SPAN_STICKS.

    For each violating tensor, finds the outermost non-size-1 device dimension
    (row-major layout means outer dims have larger strides and splitting them
    reduces contiguous span). The minimum split is rounded up to the nearest
    divisor of the stick-adjusted iteration space size so each core gets an
    equal integer-sized slice.

    Returns a dict mapping Symbol -> minimum split count, guaranteed to evenly
    divide the corresponding entry in it_space_adjusted.
    """
    if tensor_deps is None:
        return {}
    result: dict[Symbol, int] = {}
    for td in tensor_deps:
        total_sticks = math.prod(td.layout.device_layout.device_size[:-1])
        if total_sticks <= MAX_SPAN_STICKS:
            continue

        for coord in td.device_coords[:-1]:
            vars_ = coord.free_symbols
            if not vars_:
                continue  # skipping empty set (is it safe to assume no constant value > 1)?
            assert len(vars_) == 1, (
                f"Expected exactly 1 free symbol in device coord {coord!r}, got {vars_}."
            )
            adjusted_size = it_space_adjusted[next(iter(vars_))]
            if adjusted_size == 1:
                continue
            min_split_raw = math.ceil(total_sticks / MAX_SPAN_STICKS)
            min_split = next(
                (
                    d
                    for d in range(min_split_raw, adjusted_size + 1)
                    if adjusted_size % d == 0
                ),
                adjusted_size,
            )
            if min_split == adjusted_size and adjusted_size < min_split_raw:
                logger.warning(
                    f"Cannot fully satisfy span limit for {vars_} "
                    f"(adjusted_size={adjusted_size}, need {min_split_raw} splits): "
                    f"using full split of {adjusted_size}."
                )
            for var in vars_:
                result[var] = max(result.get(var, 1), min_split)
            break

    return result


def prioritize_dimensions(
    output: TensorDep,
    it_space: dict[Symbol, Expr],
    inputs: list[TensorDep] | None = None,
    exclude_reduction: bool = False,
) -> tuple[list[Symbol], dict[Symbol, int]]:
    """
    Return iteration variables in priority order for core division, along with
    minimum split requirements.

    Priority tiers:
      1. Must-split vars: outermost dims of tensors that violate MAX_SPAN_BYTES.
         Splitting these is required to bring memory span within hardware limits.
      2. Remaining output dims (present in output coords), by decreasing size.
      3. Reduction dims (absent from output coords), by decreasing size.

    Returns:
        tuple of (priority list, min_splits dict)
    """
    # Collect free symbols from all output device coords except the stick dim.
    # The stick dim is always the innermost device dimension and shares its host
    # dimension with an outer coord, so its free symbol is already captured here.
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    all_deps = (inputs + [output]) if inputs is not None else [output]
    # NOTE: it is possible that a reduction var is selected as must split
    min_splits = must_split_vars(all_deps, it_space)

    priority = []
    remaining_output = []
    reduction_dims: list[tuple[Symbol, Expr]] = []
    for s, e in it_space.items():
        if s in min_splits:
            assert not exclude_reduction or s in coord_vars, (
                f"Excluding reduction dimensions but {s} must be split"
            )
            continue
        if s in coord_vars:
            remaining_output.append((s, e))
        else:
            reduction_dims.append((s, e))

    remaining_output.sort(key=lambda t: t[1], reverse=True)
    reduction_dims.sort(key=lambda t: t[1], reverse=True)
    priority += [t[0] for t in remaining_output]
    if not exclude_reduction:
        priority += [t[0] for t in reduction_dims]

    return priority, min_splits


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    if max_cores == 1:
        return

    it_space = iteration_space(n)
    input_tds = [TensorDep(a.dep, a.layout) for a in args]
    output_td = TensorDep(next(iter(n.read_writes.writes)), n.node.get_layout())

    adjust_it_space_for_sticks(it_space, input_tds + [output_td])

    priorities, min_splits = prioritize_dimensions(output_td, it_space)
    splits = multi_dim_iteration_space_split(
        it_space,
        max_cores,
        priorities,
        min_splits,
    )

    cores_used = math.prod(splits.values())

    if cores_used > 1:
        n.op_it_space_splits = splits

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"pointwise work_division {n.node.get_name()}: cores={cores_used}, "
                f"iteration_space={it_space}, priorities={priorities}, "
                f"min_splits={min_splits}, op_it_space_splits={n.op_it_space_splits}"
            )


def divide_reduction_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    if max_cores == 1:
        return

    red: Reduction = n.node.data
    is_matmul = red.reduction_type in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP)

    it_space = iteration_space(n)
    input_tds = [TensorDep(a.dep, a.layout) for a in args]
    output_td = TensorDep(next(iter(n.read_writes.writes)), n.node.get_layout())

    # Adjust all stick dimension variables (inputs and output) to count sticks
    adjust_it_space_for_sticks(it_space, input_tds + [output_td])

    # FIXME: For non-matmul reduction, excluting reduction dimensions from work
    #        division candidates temporarily till known backend issue is fixed
    #        https://github.com/torch-spyre/torch-spyre/issues/1304
    priorities, min_splits = prioritize_dimensions(
        output_td, it_space, input_tds, exclude_reduction=not is_matmul
    )
    splits = multi_dim_iteration_space_split(
        it_space, max_cores, priorities, min_splits
    )

    cores_used = math.prod(splits.values())
    if cores_used > 1:
        n.op_it_space_splits = splits

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"reduction work_division {n.node.get_name()}: cores={cores_used}, "
                f"iteration_space={it_space}, priorities={priorities}, "
                f"min_splits={min_splits}, op_it_space_splits={n.op_it_space_splits}"
            )


def core_division_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guaranteed by caller).
    max_cores = config.sencores
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
                logger.warning(f"unhandled node type {type(n.node)}")
        elif isinstance(n, NopKernelSchedulerNode):
            pass
        else:
            logger.warning(f"unhandled scheduler node type {type(n)}")

    return nodes
