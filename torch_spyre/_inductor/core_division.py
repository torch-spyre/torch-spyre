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
import itertools
from sympy import Expr, Symbol, divisors

import torch
from torch._inductor.ir import (
    ComputedBuffer,
    ExternKernel,
    FallbackKernel,
    MultiOutput,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
)

from torch._inductor.dependencies import MemoryDep

from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    device_coordinates,
    iteration_space_from_op,
    splits_by_index_coeff,
)
from .logging_utils import get_inductor_logger
from . import config
import logging

logger = get_inductor_logger("core_division")

# Maximum memory access span per core: 256MB hardware limit
MAX_SPAN_BYTES = 256 * 1024 * 1024

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
        # TODO(issue#1372): with symbolic core division, concretize_expr
        #                   for core_split will not be needed.
        best_split = core_split(concretize_expr(iteration_space[v]), n_cores_remaining)
        if best_split > 1:
            splits[v] = best_split
            n_cores_remaining = n_cores_remaining // best_split

    return splits


def adjust_it_space_for_sticks(
    it_space: dict[Symbol, Expr],
    tensor_deps: list[TensorDep],
) -> tuple[dict[Symbol, Expr], dict[Symbol, int]]:
    """
    Return a copy of it_space with stick variables converted from elements to
    sticks, plus a dict mapping each stick variable to its max element per stick
    value.

    For each tensor, find the variable that indexes its stick dimension and
    convert its size in it_space from elements to sticks. This ensures core
    division treats sticks as atomic units.

    When tensors of different dtypes share a stick variable (e.g. a float16
    input and an int64 argmax output), the largest elems_per_stick is used
    so the adjustment is conservative (fewer sticks → smaller adjusted size →
    fewer cores assigned to the stick dimension).

    The original it_space is not mutated.
    """
    # Pass 1: find the largest elems_per_stick per stick variable.
    adjusted_space = dict(it_space)
    max_elems: dict[Symbol, int] = {}
    for td in tensor_deps:
        stick_expr = td.device_coords[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        if stick_var not in adjusted_space:
            continue
        elems_per_stick = td.layout.device_layout.elems_per_stick()
        if stick_var not in max_elems or elems_per_stick > max_elems[stick_var]:
            max_elems[stick_var] = elems_per_stick

    # Pass 2: adjust each variable once using the maximum.
    for stick_var, elems_per_stick in max_elems.items():
        # FIXME: here we assume padding to a full stick. It may not always be
        #        the case and we should use a more robust way of computing the
        #        number of sticks
        adjusted_space[stick_var] = (
            adjusted_space[stick_var] + elems_per_stick - 1
        ) // elems_per_stick

    return adjusted_space, max_elems


def get_per_core_span(
    td: TensorDep,
    splits: dict[Symbol, int],
    it_space_orig: dict[Symbol, Expr],
) -> int:
    """Compute per-core memory span in bytes for a tensor under the given splits.

    coordinate expressions from compute_coordinates() in views.py are sums of
    independent single-variable terms, so max of the full expression equals the
    sum of per-variable maxima obtained by zeroing out all other variables.
    min is always 0 since all variables start at 0. If this invariant in
    compute_coordinates() ever changes, this logic must be revisited.

    it_space_orig must be the original element-valued ranges, not the
    stick-adjusted copy, because device coordinate expressions are written in
    terms of element indices.
    """
    device_size = td.layout.device_layout.device_size
    itemsize = td.layout.dtype.itemsize
    for d, coord in enumerate(td.device_coords[:-1]):
        if not coord.free_symbols:
            continue
        per_core_max = 0
        for v in coord.free_symbols:
            term = coord.subs({u: 0 for u in coord.free_symbols - {v}})
            # Concretize the iteration-space size so R (and therefore the
            # ``int(term.subs(...))`` cast below) is a Python int.  Per-core
            # span is a hardware-bound quantity that must be compared against
            # MAX_SPAN_BYTES, so concretization here is the right boundary.
            # TODO(issue#1372): Symbolic core division will keep this symbolic.
            R = concretize_expr(it_space_orig[v]) // splits.get(v, 1)
            per_core_max += int(term.subs(v, R - 1))
        per_core_size = per_core_max + 1
        if per_core_size > 1:
            stride_elems = math.prod(device_size[d + 1 :])
            return per_core_size * stride_elems * itemsize
    return itemsize


def warn_if_per_core_overflow(
    tensor_deps: list[TensorDep],
    it_space_orig: dict[Symbol, Expr],
    splits: dict[Symbol, int],
    op_name: str,
) -> None:
    """Log CRITICAL if any tensor's per-core memory span exceeds MAX_SPAN_BYTES."""
    for td in tensor_deps:
        per_core_span = get_per_core_span(td, splits, it_space_orig)
        if per_core_span > MAX_SPAN_BYTES:
            dl = td.layout.device_layout
            logger.critical(
                f"{op_name}: per-core tensor span "
                f"{per_core_span / (1024 * 1024):.2f} MB "
                f"(shape={list(td.layout.size)}, dtype={td.layout.dtype}, "
                f"device_size={list(dl.device_size)}, splits={splits}) "
                f"exceeds hardware limit of {MAX_SPAN_BYTES / (1024 * 1024):.2f} MB"
            )


def must_split_vars(
    tensor_deps: list[TensorDep],
    it_space_orig: dict[Symbol, Expr],
    it_space_adjusted: dict[Symbol, Expr],
    stick_vars: dict[Symbol, int],
    max_cores: int,
) -> dict[Symbol, int]:
    """Return the minimum splits per iteration variable to keep each tensor's
    memory span within MAX_SPAN_BYTES.

    Processes tensors one at a time, carrying accumulated_splits forward so
    splits committed for one tensor reduce the search space for subsequent ones.
    For each violating tensor, iterates device dimensions outer to inner and
    searches for the joint split combination (Cartesian product over contributing
    variables) that brings the span closest to (but not exceeding) MAX_SPAN_BYTES.
    If no combo satisfies the limit, picks the one that minimizes the span.
    Gives up on a dimension when the committed splits still leave it evaluating
    to > 1, meaning inner dimensions cannot reduce the span further.

    Args:
        tensor_deps: List of tensor dependencies to check
        it_space_orig: Original iteration space (element-valued)
        it_space_adjusted: Adjusted iteration space (stick-valued for stick vars)
        stick_vars: Mapping of stick variables to elements per stick
        max_cores: Maximum number of cores available

    Returns a dict mapping Symbol -> number of slices.
    """
    accumulated_splits: dict[Symbol, int] = {}

    for td in tensor_deps:
        if get_per_core_span(td, accumulated_splits, it_space_orig) <= MAX_SPAN_BYTES:
            continue

        for coord in td.device_coords[:-1]:
            # Concretize for the ``> 1`` comparison: with symbolic ranges,
            # ``s0 > 1`` returns a sympy Relational whose truth value is
            # undefined.  Span filtering here is a structural decision that
            # needs a concrete answer.
            # TODO(issue#1372): Symbolic core division will keep this symbolic.
            vars = [
                v
                for v in coord.free_symbols
                if concretize_expr(it_space_orig.get(v, 1)) > 1
            ]
            if not vars:
                continue

            def valid_splits(v: Symbol) -> list[int]:
                current_min = accumulated_splits.get(v, 1)
                if v in stick_vars:
                    stick_count = concretize_expr(it_space_adjusted[v])
                    return [s for s in divisors(stick_count) if s >= current_min]
                return [
                    s
                    for s in divisors(concretize_expr(it_space_orig[v]))
                    if s >= current_min
                ]

            var_divisors = [valid_splits(v) for v in vars]

            for v, candidates in zip(vars, var_divisors):
                if not candidates:
                    raise Unsupported(
                        f"No valid split for variable {v} "
                        f"(orig_size={concretize_expr(it_space_orig[v])}, "
                        f"min_required={accumulated_splits.get(v, 1)}) "
                        f"for tensor {td.dep.name}."
                    )

            # NOTE: Exhaustive search of all combinations. It's probably ok
            #       assuming the search space is small. Can revisit if this
            #       becomes a bottleneck.
            #
            # Two-tier selection by span value:
            #   - Within-limit combos: prefer largest span (= fewest cores used)
            #   - Above-limit combos: prefer smallest span (= most progress)
            best_within: tuple[int, tuple] | None = None  # (span, combo)
            best_above: tuple[int, tuple] | None = None  # (span, combo)

            for combo in itertools.product(*var_divisors):
                trial = dict(accumulated_splits)
                for v, s in zip(vars, combo):
                    trial[v] = s

                if math.prod(trial.values()) > max_cores:
                    continue

                span = get_per_core_span(td, trial, it_space_orig)

                if span <= MAX_SPAN_BYTES:
                    if best_within is None or span > best_within[0]:
                        best_within = (span, combo)
                else:
                    if best_above is None or span < best_above[0]:
                        best_above = (span, combo)

            # Prefer within-limit; fall back to best partial progress
            best = best_within or best_above

            if best is None:
                logger.warning(
                    f"No valid split combo found for tensor {td.dep.name} "
                    f"coord={coord} under accumulated_splits={accumulated_splits}. "
                    f"Skipping."
                )
                break

            best_span, best_combo = best
            for v, s in zip(vars, best_combo):
                accumulated_splits[v] = s

            if best_span <= MAX_SPAN_BYTES:
                break

            # Still above the limit. If this coord still evaluates to > 1 under
            # the committed splits, inner dimensions cannot reduce the span further.
            # Concretize it_space_orig[v] so the ``int(coord.subs(...))`` cast
            # below succeeds with symbolic ranges.
            # TODO(issue#1372): Symbolic core division will keep this symbolic.
            per_core_coord_size = (
                max(
                    int(
                        coord.subs(
                            {
                                v: concretize_expr(it_space_orig[v])
                                // accumulated_splits.get(v, 1)
                                - 1
                                for v in coord.free_symbols
                            }
                        )
                    ),
                    0,
                )
                + 1
            )
            if per_core_coord_size > 1:
                logger.warning(
                    f"Cannot satisfy span limit for tensor {td.dep.name}: "
                    f"coord={coord} still evaluates to {per_core_coord_size} after splits. "
                    f"Inner dimensions cannot reduce span further. "
                    f"Best span={best_span}, limit={MAX_SPAN_BYTES}."
                )
                break

    return accumulated_splits


def prioritize_dimensions(
    output: TensorDep,
    it_space_adjusted: dict[Symbol, Expr],
    exclude_reduction: bool = False,
) -> list[Symbol]:
    """Return iteration variables in priority order for core division.

    Variables already committed as min_splits should be filtered out of
    it_space_adjusted before calling this function.

    Priority tiers:
      1. Output dims (present in output device coords), by decreasing size.
      2. Reduction dims (absent from output coords), by decreasing size.
    """
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    remaining_output = []
    reduction_dims: list[tuple[Symbol, Expr]] = []
    for s, e in it_space_adjusted.items():
        if s in coord_vars:
            remaining_output.append((s, e))
        else:
            reduction_dims.append((s, e))

    # Concretize sort keys: comparing two sympy Symbols returns a Relational
    # whose truth value is undefined and would raise inside Python's sort.
    # The priority order is a structural decision (largest dim first) that
    # needs a concrete numeric ordering.
    # TODO(issue#1372): Symbolic core division will keep this symbolic.
    remaining_output.sort(key=lambda t: concretize_expr(t[1]), reverse=True)
    reduction_dims.sort(key=lambda t: concretize_expr(t[1]), reverse=True)

    priority = [t[0] for t in remaining_output]
    if not exclude_reduction:
        priority += [t[0] for t in reduction_dims]

    return priority


def plan_splits(
    all_tds: list[TensorDep],
    output_td: TensorDep,
    it_space: dict[Symbol, Expr],
    max_cores: int,
    exclude_reduction: bool = False,
) -> tuple[dict[Symbol, int], dict[Symbol, Expr], list[Symbol], dict[Symbol, int]]:
    """Compute core splits for an op's iteration space.

    Returns (splits, it_space_adjusted, priorities, min_splits).

    When exclude_reduction is True, asserts that no reduction variable appears
    in min_splits — callers are responsible for only passing exclude_reduction=True
    when the backend supports it.
    """
    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(it_space, all_tds)

    # 1. Determine the minimum splits required to keep each tensor's memory
    #    span within the hardware limit.
    min_splits = must_split_vars(
        all_tds, it_space, it_space_adjusted, stick_vars, max_cores
    )

    if exclude_reduction:
        coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
        reduction_vars_to_split = set(min_splits) - coord_vars
        if reduction_vars_to_split:
            raise Unsupported(
                f"Cannot satisfy hardware memory span limit "
                f"({MAX_SPAN_BYTES // (1024 * 1024)}MB) without splitting reduction"
                f"dimensions {reduction_vars_to_split}, but the backend does not "
                f"support splitting reduction dimensions for this operation type."
            )

    # 2. Among the remaining dimensions, decide the priority order for
    #    distributing leftover cores.
    it_space_remaining = {
        s: e for s, e in it_space_adjusted.items() if s not in min_splits
    }
    priorities = prioritize_dimensions(
        output_td, it_space_remaining, exclude_reduction=exclude_reduction
    )

    # 3. Assign cores: satisfy min_splits first, then fill by priority.
    splits = multi_dim_iteration_space_split(
        it_space_adjusted, max_cores, priorities, min_splits
    )
    return splits, it_space_adjusted, priorities, min_splits


def _resolve_layout(op: ComputedBuffer) -> "FixedTiledLayout":
    """Return the FixedTiledLayout for op, unwrapping MutationLayoutSHOULDREMOVE.

    Mutation ops keep MutationLayoutSHOULDREMOVE at pre-scheduler time so the
    scheduler can identify them as in-place writes.  Their target buffer already
    has a FixedTiledLayout assigned by propagate_spyre_tensor_layouts, so
    real_layout() gives us the correct device layout for core division.
    """
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    assert isinstance(layout, FixedTiledLayout), (
        f"Expected FixedTiledLayout for {op.get_name()}, got {type(layout)}"
    )
    return layout


def collect_tensor_deps(
    op: ComputedBuffer, args: list[SchedNodeArg]
) -> tuple[list[TensorDep], TensorDep]:
    """Build TensorDep lists for inputs and the output of op."""
    input_tds = [TensorDep(a.dep, a.layout) for a in args]
    rw = op.get_read_writes()
    output_td = TensorDep(next(iter(rw.writes)), _resolve_layout(op))
    return input_tds, output_td


def apply_splits(
    op: ComputedBuffer,
    splits: dict,
    output_td: TensorDep,
    it_space: dict,
    it_space_adjusted: dict,
    priorities: list,
    min_splits: dict,
    kind: str,
) -> None:
    """Commit splits to op and emit a debug log entry.

    Does nothing when the product of splits is 1 (no parallelism).
    kind is a short label used in the log message (e.g. "pointwise" or "reduction").
    """
    cores_used = math.prod(splits.values())
    if cores_used <= 1:
        return

    rw = op.get_read_writes()
    write_index = output_td.dep.index
    first_read = next(iter(rw.reads), None)
    read_index = first_read.index if first_read is not None else write_index
    op.op_it_space_splits = splits_by_index_coeff(splits, write_index, read_index)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"{kind} work_division {op.get_name()}: cores={cores_used}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"priorities={priorities}, min_splits={min_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )


def divide_pointwise_op(op: ComputedBuffer, args: list[SchedNodeArg], max_cores):
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)

    splits: dict[Symbol, int] = {}
    if max_cores > 1:
        splits, it_space_adjusted, priorities, min_splits = plan_splits(
            input_tds + [output_td], output_td, it_space, max_cores
        )
        apply_splits(
            op,
            splits,
            output_td,
            it_space,
            it_space_adjusted,
            priorities,
            min_splits,
            kind="pointwise",
        )

    warn_if_per_core_overflow(input_tds + [output_td], it_space, splits, op.get_name())


def divide_reduction_op(op: ComputedBuffer, args: list[SchedNodeArg], max_cores):
    red: Reduction = op.data
    is_matmul = red.reduction_type in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP)

    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)

    # FIXME: For non-matmul reduction, excluding reduction dimensions from work
    #        division candidates temporarily till known backend issue is fixed
    #        https://github.com/torch-spyre/torch-spyre/issues/1304
    splits: dict[Symbol, int] = {}
    if max_cores > 1:
        splits, it_space_adjusted, priorities, min_splits = plan_splits(
            input_tds + [output_td],
            output_td,
            it_space,
            max_cores,
            exclude_reduction=not is_matmul,
        )
        apply_splits(
            op,
            splits,
            output_td,
            it_space,
            it_space_adjusted,
            priorities,
            min_splits,
            kind="reduction",
        )

    warn_if_per_core_overflow(input_tds + [output_td], it_space, splits, op.get_name())


def core_division_planning(
    operations: list[Operation],
) -> None:
    # Operations are in topological order (guaranteed by GraphLowering).
    max_cores = config.sencores
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")

    it = iter(operations)
    for op in it:
        if op.is_no_op():
            pass
        elif isinstance(op, ComputedBuffer):
            rw = op.get_read_writes()
            args = get_mem_deps_from_rw(rw)
            if isinstance(op.data, Pointwise):
                divide_pointwise_op(op, args, max_cores)
            elif isinstance(op.data, Reduction):
                divide_reduction_op(op, args, max_cores)
            else:
                # Core division not supported on other IRNode types
                pass
        elif isinstance(op, FallbackKernel):
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            # Core division not supported on fallback kernels
        elif isinstance(op, ExternKernel):
            logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")
