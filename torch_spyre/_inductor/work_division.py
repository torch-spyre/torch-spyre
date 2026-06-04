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
from .ir import SpyreConstantFallback, SpyreEmptyFallback

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
from .constants import BATCH_MATMUL_OP, TOPK_OPS
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    device_coordinates,
    iteration_space_from_op,
    splits_by_index_coeff,
    apply_splits_from_index_coeff,
)
from typing import Callable

from .logging_utils import get_inductor_logger
from . import config
import logging

logger = get_inductor_logger("work_division")

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
        self.device_coords = device_coordinates(self.layout.device_layout, self.dep)


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


def _most_splittable_dim(
    dims: list[Symbol],
    iteration_space: dict[Symbol, Expr],
    n_cores: int,
) -> tuple[Symbol, int] | None:
    """Return (dim, split) for the dim in dims that maximises core_split(size, n_cores).

    Returns None if no dim yields a split > 1.
    """
    best_dim, best_split = None, 0
    for d in dims:
        s = core_split(concretize_expr(iteration_space[d]), n_cores)
        if s > best_split:
            best_dim, best_split = d, s
    return (best_dim, best_split) if best_split > 1 else None


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    output_dims: list[Symbol],
    reduction_dims: list[Symbol],
    min_splits: dict[Symbol, int] | None = None,
) -> dict[Symbol, int]:
    """Distribute max_cores across the iteration space.

    Three-pass algorithm:
      1. Satisfy min_splits (span-reduction commitments).
      2. Distribute remaining cores to output_dims in priority order.
      3. If this is a reduction op, pick the single most-splittable reduction dim
         for any remaining cores.

    The product of all splits will be <= max_cores.
    """
    is_reduction_included = bool(reduction_dims)

    splits = {v: 1 for v in iteration_space.keys()}
    n_cores_remaining = max_cores

    if min_splits:
        # Sanity check: making sure that reduction_dims list is cleared up if
        #               any reduction dim is already selected during span reduction
        assert (
            not is_reduction_included  # not empty
            or not any(v in min_splits for v in reduction_dims)  # no overlap
        )

        for var, min_split in min_splits.items():
            assert var not in output_dims and var not in reduction_dims

            if n_cores_remaining // min_split <= 0:
                logger.critical(
                    f"Cannot satisfy minimum split requirement for {var}: "
                    f"need {min_split} splits but only {n_cores_remaining} cores remaining. "
                    f"Skipping this constraint - hardware span limit may be violated."
                )
                continue
            splits[var] = min_split
            n_cores_remaining = n_cores_remaining // min_split

    for v in output_dims:
        if n_cores_remaining <= 1:
            break
        # TODO(issue#1372): with symbolic work division, concretize_expr
        #                   for core_split will not be needed.
        best_split = core_split(concretize_expr(iteration_space[v]), n_cores_remaining)
        if best_split > 1:
            splits[v] = best_split
            n_cores_remaining = n_cores_remaining // best_split

    if is_reduction_included and n_cores_remaining > 1:
        result = _most_splittable_dim(
            reduction_dims, iteration_space, n_cores_remaining
        )
        if result is not None:
            best_dim, best_split = result
            splits[best_dim] = best_split

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
    convert its size in it_space from elements to sticks. This ensures work
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
        per_core_min = 0
        for v in coord.free_symbols:
            term = coord.subs({u: 0 for u in coord.free_symbols - {v}})
            # Concretize the iteration-space size so R (and therefore the
            # ``int(term.subs(...))`` cast below) is a Python int.  Per-core
            # span is a hardware-bound quantity that must be compared against
            # MAX_SPAN_BYTES, so concretization here is the right boundary.
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
            R = concretize_expr(it_space_orig[v]) // splits.get(v, 1)
            per_core_max += int(term.subs(v, R - 1))
            per_core_min += int(term.subs(v, 0))
        per_core_size = per_core_max - per_core_min + 1
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
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
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
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
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
) -> tuple[list[Symbol], list[Symbol]]:
    """Partition iteration variables into output dims and reduction dims.

    Output dims are those whose symbols appear in the output tensor's device
    coordinate expressions (excluding the stick coordinate). Reduction dims are
    the remainder. Both lists are sorted by decreasing concrete size.

    Variables already committed as min_splits should be filtered out of
    it_space_adjusted before calling this function.
    """
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    output_pairs: list[tuple[Symbol, Expr]] = []
    reduction_pairs: list[tuple[Symbol, Expr]] = []
    for s, e in it_space_adjusted.items():
        (output_pairs if s in coord_vars else reduction_pairs).append((s, e))

    # Concretize sort keys: comparing two sympy Symbols returns a Relational
    # whose truth value is undefined and would raise inside Python's sort.
    # The priority order is a structural decision (largest dim first) that
    # needs a concrete numeric ordering.
    # TODO(issue#1372): Symbolic work division will keep this symbolic.
    output_pairs.sort(key=lambda t: concretize_expr(t[1]), reverse=True)
    reduction_pairs.sort(key=lambda t: concretize_expr(t[1]), reverse=True)

    return [t[0] for t in output_pairs], [t[0] for t in reduction_pairs]


def _resolve_layout(op: ComputedBuffer) -> "FixedTiledLayout":
    """Return the FixedTiledLayout for op, unwrapping MutationLayoutSHOULDREMOVE.

    Mutation ops keep MutationLayoutSHOULDREMOVE at pre-scheduler time so the
    scheduler can identify them as in-place writes.  Their target buffer already
    has a FixedTiledLayout assigned by propagate_spyre_tensor_layouts, so
    real_layout() gives us the correct device layout for work division.
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
) -> None:
    """Commit splits to op.

    Does nothing when the product of splits is 1 (no parallelism).
    """
    cores_used = math.prod(splits.values())
    if cores_used <= 1:
        return

    rw = op.get_read_writes()
    write_index = output_td.dep.index
    first_read = next(iter(rw.reads), None)
    read_index = first_read.index if first_read is not None else write_index
    op.op_it_space_splits = splits_by_index_coeff(splits, write_index, read_index)


def span_reduction_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Mandatory per-op pass: compute minimum splits to satisfy the 256MB span limit.

    Writes results to op.op_it_space_splits. If no span violation exists,
    op.op_it_space_splits is left unset (apply_splits is a no-op for splits <= 1).
    """
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(it_space, all_tds)
    min_splits = must_split_vars(
        all_tds, it_space, it_space_adjusted, stick_vars, max_cores
    )

    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars_to_split = set(min_splits) - coord_vars
    # Each entry in Reduction.reduction_ranges maps to at most one Symbol via
    # index_vars_squeeze (size-1 entries are squeezed away). So len > 1 means
    # genuinely distinct reduction dimensions, not multiple symbols from one dim.
    if len(reduction_vars_to_split) > 1:
        raise Unsupported(
            f"Cannot satisfy hardware memory span limit "
            f"({MAX_SPAN_BYTES // (1024 * 1024)}MB) without splitting "
            f"{len(reduction_vars_to_split)} reduction dimension(s) "
            f"({reduction_vars_to_split}), but the backend supports at most 1."
        )

    apply_splits(op, min_splits, output_td)

    if logger.isEnabledFor(logging.DEBUG) and math.prod(min_splits.values()) > 1:
        logger.debug(
            f"span_reduction work_division {op.get_name()}: cores={math.prod(min_splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"priorities=[], min_splits={min_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )


def _default_split(
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    committed_splits: dict[Symbol, int],
    max_cores: int,
) -> tuple[dict[Symbol, int], list[Symbol], list[Symbol]]:
    """Distribute max_cores by priority on top of span_reduction's commits.

    Returns the chosen splits and the (output, reduction) priority dims the
    caller logs. Shared by work_distribution_pass and cost_model_matmul_division.
    """
    # TODO: The final dim committed by span_reduction_pass holds the minimum
    #       split that gets the span under the limit, so it may have headroom
    #       for additional parallelism (outer dims committed before it are
    #       already maximally split and have no headroom). Excluding it here
    #       leaves that parallelism on the table when other dims can't absorb
    #       the remaining cores.
    it_space_remaining = {
        s: e for s, e in it_space_adjusted.items() if s not in committed_splits
    }
    output_dims, reduction_dims = prioritize_dimensions(output_td, it_space_remaining)

    # If span_reduction_pass already committed a reduction split, suppress further
    # reduction splitting so the final result never exceeds one reduction dim split.
    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    if any(v not in coord_vars for v in committed_splits):
        reduction_dims = []

    # Pass max_cores, not remaining_cores: multi_dim_iteration_space_split
    # accounts for committed_splits in its first pass, consuming those cores
    # itself before distributing the rest by priority.
    splits = multi_dim_iteration_space_split(
        it_space_adjusted,
        max_cores,
        output_dims,
        reduction_dims,
        committed_splits,
    )
    return splits, output_dims, reduction_dims


def work_distribution_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Optional per-op pass: distribute remaining cores to maximize parallelism.

    Reads op.op_it_space_splits written by span_reduction_pass (if any) to
    recover the already-committed splits, then fills remaining cores by priority.
    """
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space_adjusted, _ = adjust_it_space_for_sticks(it_space, all_tds)

    # Recover splits committed by span_reduction_pass using the same
    # coeff-keyed encoding that codegen uses — stable across passes.
    if hasattr(op, "op_it_space_splits"):
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        min_splits = apply_splits_from_index_coeff(
            op.op_it_space_splits, write_index, read_index, it_space
        )
    else:
        min_splits = {}

    # apply_splits_from_index_coeff returns 1 for every unsplit dim; keep only
    # dims with actual committed splits so they don't overlap with priorities.
    committed_splits = {s: v for s, v in min_splits.items() if v > 1}

    splits, output_dims, reduction_dims = _default_split(
        it_space_adjusted, output_td, committed_splits, max_cores
    )

    apply_splits(op, splits, output_td)

    if logger.isEnabledFor(logging.DEBUG) and math.prod(splits.values()) > 1:
        logger.debug(
            f"work_distribution work_division {op.get_name()}: cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"priorities={output_dims + reduction_dims}, min_splits={committed_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )

    warn_if_per_core_overflow(all_tds, it_space, splits, op.get_name())


_PT_ROWS = 8  # PT block rows per corelet

# Constants for the matmul cost model (_matmul_split_cost). Each is either an
# AIU hardware limit or a coefficient fit to measured device kernel times.
_TARGET_PT_PASSES = 8  # per-core M that keeps the PT pipeline full = this * _PT_ROWS
_M_MIN = _PT_ROWS // 2  # below half a PT pass an m-split buys nothing
_PEAK_MACS_US_CORE = (98.304e12 / 2 / 32) / 1e6  # DL16 peak / 32 cores, MACs/us/core
_HBM_BW_GBS = 204.8  # LPDDR5 aggregate peak bandwidth
_DTYPE_BYTES = 2  # fp16
_PSUM_PER_ELEM_US = 1.4e-4  # per output element, per K-split ring reduction hop
_COHORT_LIMIT = 8  # cores sharing a broadcast before it contends for bandwidth
_BATCH_SPLIT_EXPONENT = 1.4  # batch-split cost grows ~ b ** this (fit to bmm sweeps)
_TARGET_M_PENALTY_US = 50.0  # tie-break weight, per log2 step off the target m-split


def _matmul_split_cost(
    b_axis: tuple[int, int],
    m_axis: tuple[int, int],
    n_axis: tuple[int, int],
    k_axis: tuple[int, int],
    max_cores: int,
) -> float:
    """Estimated kernel time in microseconds for ``[B,M,K]@[B,K,N]`` run with
    the given core split. Each axis is a ``(size, split)`` pair so a dim's size
    cannot be paired with another dim's split. Lower is better; inf if infeasible.
    """
    (B, b), (M, m), (N, n), (K, k) = b_axis, m_axis, n_axis, k_axis
    cores_used = b * m * n * k
    if cores_used == 0 or cores_used > max_cores:
        return float("inf")

    # Compute: per-core MACs over peak, derated when the per-core M tile is too
    # short to fill the PT pipeline. The PT array streams M in passes of
    # _PT_ROWS; below _TARGET_PT_PASSES passes its startup/drain overhead is
    # amortised over too little work, and that overhead grows sub-linearly,
    # hence the sqrt.
    m_t = M // m if m else 1
    pt_passes = max(1.0, m_t / _PT_ROWS)
    pt_eff = min(1.0, (pt_passes / _TARGET_PT_PASSES) ** 0.5)
    compute_us = (B * M * N * K / cores_used) / (_PEAK_MACS_US_CORE * pt_eff)

    # HBM: every input operand is broadcast to the cohort of cores splitting the
    # orthogonal dim. Past _COHORT_LIMIT the broadcasts contend for the shared
    # link, so effective bandwidth falls off linearly with cohort size.
    bytes_total = (B * M * K + B * K * N + B * M * N) * _DTYPE_BYTES
    cohort_penalty = max(1.0, max(m, n) / _COHORT_LIMIT)
    hbm_us = bytes_total / (_HBM_BW_GBS * 1000) * cohort_penalty

    # PSUM: a K-split spreads the reduction over k cores, costing (k-1) partial-
    # sum hops around the ring, each touching every output element.
    psum_us = max(0, k - 1) * (B * M * N) * _PSUM_PER_ELEM_US

    # Tie-break: among compute-equivalent splits prefer the m-split that lands
    # per-core M near the PT sweet spot, penalising log2-distance from it.
    target_m = max(
        _M_MIN, min(max_cores // 2, max(1, M // (_TARGET_PT_PASSES * _PT_ROWS)))
    )
    target_m_us = abs(math.log2(max(1, m) / target_m)) * _TARGET_M_PENALTY_US

    # Splitting batch across cores is strictly worse than tiling it in time on
    # one core (each item is independent), so charge a super-linear b penalty.
    batch_penalty = b**_BATCH_SPLIT_EXPONENT

    return (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty


def _cost_model_matmul_planner(
    op: ComputedBuffer,
    splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    max_cores: int,
    input_tds: list[TensorDep],
) -> dict[Symbol, int]:
    """Override the default split for a matmul / bmm with the lowest-cost
    feasible (b, m, n, k) per _matmul_split_cost.

    Returns ``splits`` unchanged for anything this planner does not model:
    non-matmuls, ops with a span-committed split already in place, multi-K
    matmuls, or a chosen split that would use fewer cores than the default.
    """
    if not isinstance(op.data, Reduction):
        return splits
    if op.data.reduction_type != BATCH_MATMUL_OP:
        return splits
    if committed_splits:
        return splits

    # Classify the output coord dims: the stickified one is N, the rest index
    # rows. Of those row dims, M is the one appearing in a single input (the
    # LHS); batch dims appear in both.
    output_coord_vars = {
        v for e in output_td.device_coords[:-1] for v in e.free_symbols
    }
    n_dims = [d for d in output_coord_vars if d in stick_vars]
    row_dims = [d for d in output_coord_vars if d not in stick_vars]
    if len(n_dims) != 1 or not row_dims:
        return splits
    n_dim = n_dims[0]

    def _appears_in_one_input(dim: Symbol) -> bool:
        hits = sum(
            dim in {v for e in td.device_coords for v in e.free_symbols}
            for td in input_tds
        )
        return hits == 1

    m_candidates = [d for d in row_dims if _appears_in_one_input(d)]
    # A bmm with a SHARED 2D weight broadcasts it across the batch, so the batch
    # dim "appears in one input" like M and m_candidates has two entries -> we
    # decline here and the default distributor handles it. Engaging the planner
    # for that case needs weight-rank awareness (which also fixes the B*K*N HBM
    # term over-counting the shared weight); tracked as a follow-up.
    if len(m_candidates) != 1:
        return splits
    m_dim = m_candidates[0]
    batch_dims = [d for d in row_dims if d is not m_dim]

    # K is the lone reduction dim (anything else this planner does not model).
    reduction = [d for d in it_space_adjusted if d not in output_coord_vars]
    if len(reduction) != 1:
        return splits
    k_dim = reduction[0]

    # The iteration space measures N and K in sticks; the cost model wants real
    # elements so its byte and MAC counts are physical.
    elems_per_stick = output_td.layout.device_layout.device_dtype.elems_per_stick()
    M_e = concretize_expr(it_space_adjusted[m_dim])
    n_sticks = concretize_expr(it_space_adjusted[n_dim])
    k_sticks = concretize_expr(it_space_adjusted[k_dim])
    N_e = n_sticks * elems_per_stick
    K_e = k_sticks * elems_per_stick

    batch_sizes = [concretize_expr(it_space_adjusted[bd]) for bd in batch_dims]
    B_total = math.prod(batch_sizes)

    b_combos = (
        list(itertools.product(*([int(d) for d in divisors(s)] for s in batch_sizes)))
        if batch_dims
        else [()]
    )
    m_divs = [int(d) for d in divisors(M_e)]
    n_divs = [int(d) for d in divisors(n_sticks)]
    k_divs = [int(d) for d in divisors(k_sticks)]

    best = None
    best_cost = float("inf")
    for b_combo in b_combos:
        b_prod = math.prod(b_combo)
        for mm in m_divs:
            for nn in n_divs:
                for kk in k_divs:
                    if b_prod * mm * nn * kk > max_cores:
                        continue
                    c = _matmul_split_cost(
                        (B_total, b_prod), (M_e, mm), (N_e, nn), (K_e, kk), max_cores
                    )
                    if c < best_cost:
                        best_cost = c
                        best = (b_combo, mm, nn, kk)

    if best is None:
        return splits

    b_combo, m_s, n_s, k_s = best
    new_splits = dict(splits)
    for bd, bs in zip(batch_dims, b_combo):
        new_splits[bd] = int(bs)
    new_splits[m_dim] = m_s
    new_splits[n_dim] = n_s
    new_splits[k_dim] = k_s

    # Never trade down to fewer cores than the default distributor already found.
    if math.prod(new_splits.values()) < math.prod(splits.values()):
        return splits

    logger.debug(
        f"cost_model work_division {op.get_name()}: "
        f"b={b_combo} m={m_s} n={n_s} k={k_s} cost={best_cost:.1f}us "
        f"[B={B_total} M={M_e} K={K_e} N={N_e}]"
    )
    return new_splits


def divide_pointwise_op(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
    pass_fn: Callable,
) -> None:
    pass_fn(op, args, max_cores)


def divide_reduction_op(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
    pass_fn: Callable,
) -> None:
    red: Reduction = op.data

    # Currently we support Topk for k<=4, which can be handled efficiently on single core
    # TODO: Modification will be required to enable Topk for k>4
    if red.reduction_type in TOPK_OPS:
        return

    pass_fn(op, args, max_cores)


def _validate_max_cores() -> int:
    max_cores = config.sencores
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")
    return max_cores


def _iter_computed_buffers(operations: list[Operation]):
    """Yield ComputedBuffer ops, handling FallbackKernel/ExternKernel dispatch."""
    it = iter(operations)
    for op in it:
        if op.is_no_op():
            pass
        elif isinstance(op, ComputedBuffer):
            yield op
        elif isinstance(op, FallbackKernel):
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            # Work division not supported on fallback kernels
        elif isinstance(op, ExternKernel):
            if isinstance(op, (SpyreConstantFallback, SpyreEmptyFallback)):
                # Work division not supported on allocation/constant kernels
                pass
            else:
                logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")


def span_reduction(operations: list[Operation]) -> None:
    """Pass 1: compute minimum per-op splits required by the 256MB span limit."""
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        rw = op.get_read_writes()
        args = get_mem_deps_from_rw(rw)
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, span_reduction_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, span_reduction_pass)


def work_distribution(
    operations: list[Operation], preassigned_ops: list[Operation] | None = None
) -> None:
    """Pass 3: distribute remaining cores across ops to maximize parallelism.

    Ops in `preassigned_ops` were already divided by cost_model_matmul_division;
    they are left untouched so every op is divided by exactly one pass.
    """
    preassigned_ops = preassigned_ops or []
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        if op in preassigned_ops:
            continue
        rw = op.get_read_writes()
        args = get_mem_deps_from_rw(rw)
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, work_distribution_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, work_distribution_pass)


def _cost_model_divide_op(op: ComputedBuffer, max_cores: int) -> bool:
    """Re-price one matmul's split with the analytic cost model.

    Runs between span_reduction and work_distribution, so op.op_it_space_splits
    still holds only span_reduction's commits. Computes the split
    work_distribution would pick, hands it to the cost model, and commits the
    cost model's choice when it differs — returning True so the caller excludes
    the op from work_distribution (every op is divided by exactly one pass).
    """
    if not isinstance(op.data, Reduction):
        return False
    if op.data.reduction_type != BATCH_MATMUL_OP:
        return False

    rw = op.get_read_writes()
    args = get_mem_deps_from_rw(rw)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space = iteration_space_from_op(op)
    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(it_space, all_tds)

    # op.op_it_space_splits holds span_reduction's commits here: span_reduction
    # runs before this pass, and work_distribution — which would overwrite it —
    # runs after and skips the ops this pass claims.
    if hasattr(op, "op_it_space_splits"):
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        span_splits = apply_splits_from_index_coeff(
            op.op_it_space_splits, write_index, read_index, it_space
        )
        committed_splits = {s: v for s, v in span_splits.items() if v > 1}
    else:
        committed_splits = {}

    default_splits, _, _ = _default_split(
        it_space_adjusted, output_td, committed_splits, max_cores
    )
    splits = _cost_model_matmul_planner(
        op,
        default_splits,
        it_space_adjusted,
        output_td,
        stick_vars,
        committed_splits,
        max_cores,
        input_tds,
    )
    if splits == default_splits:
        return False

    apply_splits(op, splits, output_td)
    warn_if_per_core_overflow(all_tds, it_space, splits, op.get_name())

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"cost_model_matmul_division work_division {op.get_name()}: "
            f"cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"min_splits={committed_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )
    return True


def cost_model_matmul_division(operations: list[Operation]) -> list[Operation]:
    """Pass 2: re-price matmul/bmm splits with the analytic hardware cost model.

    Runs after span_reduction and before work_distribution. Returns the ops it
    re-split so passes.py can exclude them from work_distribution — every op is
    divided by exactly one pass.
    """
    max_cores = _validate_max_cores()
    cost_model_ops: list[Operation] = []
    for op in _iter_computed_buffers(operations):
        if _cost_model_divide_op(op, max_cores):
            cost_model_ops.append(op)
    return cost_model_ops
