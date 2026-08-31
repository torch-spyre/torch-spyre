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


import builtins
import dataclasses
import itertools
import sympy
import logging
import math
from collections.abc import Callable

from sympy import Expr, Integer, Symbol, divisors
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    DeviceCopy,
    ExternKernel,
    FallbackKernel,
    MultiOutput,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
)

from torch_spyre._C import ElementArrangement

from . import config
from .constants import BATCH_MATMUL_FP8_OP, BATCH_MATMUL_OP, DEVICE_NAME
from .errors import Unsupported
from .ir import (
    AllGatherAsyncFallback,
    AllReduceAsyncFallback,
    BroadcastAsyncFallback,
    FixedTiledLayout,
    SpyreConstantFallback,
    SpyreEmptyFallback,
    WaitWorkFallback,
)
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    SchedNodeArg,
    compute_granularity,
    compute_max_size,
    concretize_expr,
    device_coordinates,
    finite_upper_or_none,
    get_mem_deps_from_rw,
    iteration_space_from_op,
    commit_iteration_space_ownership,
    op_read_writes,
)
from .propagate_hints import get_op_hints
from .work_division_constraints import (
    WorkDivConstraintContext,
    collect_work_division_constraints,
    has_qfp8wt_tensor,
)

logger = get_inductor_logger("work_division")

# Maximum memory-access span per core: 256MB hardware limit
MAX_SPAN_BYTES = 256 * 1024 * 1024


@dataclasses.dataclass
class TensorDep:
    """Bundles a MemoryDep with its FixedTiledLayout and pre-computes device coordinates."""

    dep: MemoryDep
    layout: FixedTiledLayout
    device_coords: list[Expr] = dataclasses.field(init=False)

    def __post_init__(self):
        self.device_coords = device_coordinates(
            self.layout.device_layout, self.dep, None
        )


# Per-symbol (max_size, granularity) bucket metadata for symbolic iteration vars.
# Concrete iteration vars are absent from the dict — lookups default to the
# concrete ``concretize_expr`` path via ``_effective_size`` / ``_valid_divisor_basis``.
SymbolMeta = dict[Symbol, tuple[int, int]]


def _collect_symbol_metadata(it_space: dict[Symbol, Expr]) -> SymbolMeta:
    """Build ``{symbol: (max_size, granularity)}`` for opted-in symbolic dims.

    An iteration var is "opted in" iff the user passed
    ``mark_dynamic(max=...)`` -- that's exactly when ShapeEnv records a
    finite upper bound. Auto-dynamic symbols (Dynamo promoting an int on
    retrace when a Python loop varies it) have no finite max, so we skip
    them here and let them fall through to the existing
    ``concretize_expr`` + ``optimization_hint`` path.

    Concrete dims (no free symbols) are also omitted, so callers can use
    ``v in meta`` to detect both cases.
    """
    meta: SymbolMeta = {}
    for sym, expr in it_space.items():
        if not (hasattr(expr, "free_symbols") and expr.free_symbols):
            continue
        if finite_upper_or_none(expr) is None:
            logger.debug(
                f"[work_division/symbolic] skipping auto-dynamic symbol "
                f"{sym}; use mark_dynamic(max=...) to enable symbolic planning"
            )
            continue
        max_size = compute_max_size(expr)
        granularity = compute_granularity(expr, max_size)
        meta[sym] = (max_size, granularity)
    if meta:
        logger.info(
            "[work_division/symbolic] collected symbol_meta: "
            + ", ".join(f"{sym}=(max={ms}, gran={g})" for sym, (ms, g) in meta.items())
        )
    return meta


def _effective_size(v: Symbol, it_space: dict[Symbol, Expr], meta: SymbolMeta) -> int:
    """Return the canonical size of ``v`` for ranking and the span check.

    For symbolic dims, this is ``max_size`` — the worst-case runtime footprint
    that the compiled plan must remain legal against. For concrete dims, it
    is the concretized integer range.
    """
    if v in meta:
        return meta[v][0]
    return concretize_expr(it_space.get(v, 1))


def _valid_divisor_basis(
    v: Symbol, it_space: dict[Symbol, Expr], meta: SymbolMeta
) -> int:
    """Return the integer whose divisors are valid split counts for ``v``.

    For symbolic dims, this is ``granularity`` — the divisibility invariant
    ``n | granularity`` ensures ``R / n`` stays integer for every admissible
    runtime value ``R = granularity * k``. For concrete dims, it is just the
    concretized size.

    Absent dims (e.g. pool reduction dims ki/kj stripped from the
    work-division iteration space) return 1 — no valid split beyond 1,
    matching the hardware constraint that pool window dims are never split.
    """
    if v in meta:
        return meta[v][1]
    return concretize_expr(it_space.get(v, 1))


def _legal_split_factors(
    v: Symbol,
    basis: int,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
    min_splits: dict[Symbol, int] | None = None,
) -> list[int]:
    """Return legal divisors of ``basis`` at or above a span-required floor."""
    factors = [int(s) for s in divisors(basis)]
    if allowed_splits is not None and v in allowed_splits:
        factors = [s for s in factors if s in allowed_splits[v]]
    return [s for s in factors if s >= (min_splits or {}).get(v, 1)]


def _span_min_splits(op: ComputedBuffer) -> dict[Symbol, int]:
    """Return the hard span floors committed by ``span_reduction_pass``."""
    return getattr(op, "_work_division_span_min_splits", {})


def _largest_legal_split(
    v: Symbol,
    basis: int,
    max_cores: int,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
) -> int:
    """Largest legal divisor of ``basis`` within ``max_cores``, else one."""
    legal_factors = _legal_split_factors(v, basis, allowed_splits)
    for split in reversed(legal_factors):
        if split <= max_cores:
            return split
    if allowed_splits is not None and v in allowed_splits:
        raise Unsupported(
            f"No legal split for {v} within {max_cores} cores; legal splits are "
            f"{sorted(allowed_splits[v])}."
        )
    return 1


def _largest_legal_split_from(
    v: Symbol,
    basis: int,
    current_split: int,
    cores_used: int,
    max_cores: int,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
    min_splits: dict[Symbol, int] | None = None,
) -> int:
    """Largest legal factor that replaces ``current_split`` within the budget."""
    return next(
        (
            split
            for split in reversed(
                _legal_split_factors(v, basis, allowed_splits, min_splits)
            )
            if split >= current_split
            and cores_used // current_split * split <= max_cores
        ),
        current_split,
    )


def _most_splittable_dim(
    dims: list[Symbol],
    iteration_space: dict[Symbol, Expr],
    splits: dict[Symbol, int],
    cores_used: int,
    max_cores: int,
    symbol_meta: SymbolMeta,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
    min_splits: dict[Symbol, int] | None = None,
) -> tuple[Symbol, int] | None:
    """Return dim and largest reachable split, or None if no dim can grow."""
    best_dim, best_split = None, 0
    for d in dims:
        split = _largest_legal_split_from(
            d,
            _valid_divisor_basis(d, iteration_space, symbol_meta),
            splits[d],
            cores_used,
            max_cores,
            allowed_splits,
            min_splits,
        )
        if split > splits[d] and split > best_split:
            best_dim, best_split = d, split
    return (best_dim, best_split) if best_dim is not None else None


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    output_dims: list[Symbol],
    reduction_dims: list[Symbol],
    min_splits: dict[Symbol, int] | None = None,
    symbol_meta: SymbolMeta | None = None,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
) -> dict[Symbol, int]:
    """Distribute max_cores across the iteration space.

    Three-pass algorithm:
      1. Satisfy min_splits (span-reduction commitments).
      2. Distribute remaining cores to output_dims in priority order.
      3. If this is a reduction op, pick the single most-splittable reduction dim
         for any remaining cores.

    ``symbol_meta`` carries ``(max_size, granularity)`` for any symbolic dim.
    A symbolic dimension uses ``granularity`` instead of its concretised size
    so the chosen split divides every admissible runtime bucket evenly.

    ``mandatory_splits`` is a local merge of ``min_splits`` and the smallest
    legal factor for every domain that excludes one. Each selected factor
    reserves core budget before greedy distribution. A later factor may replace
    it when it is legal, no smaller than the span floor, and fits the total
    core budget.

    The product of all splits will be <= max_cores.
    """
    symbol_meta = symbol_meta or {}
    is_reduction_included = bool(reduction_dims)

    splits = {v: 1 for v in iteration_space}
    cores_used = 1

    mandatory_splits = dict(min_splits or {})
    if allowed_splits:
        # A domain without one is a hard minimum split even without a span
        # commitment. Reserve its smallest legal factor before greedy selection.
        mandatory_splits.update(
            {
                var: min(allowed)
                for var, allowed in allowed_splits.items()
                if 1 not in allowed and var not in mandatory_splits
            }
        )

    for var, min_split in mandatory_splits.items():
        if cores_used * min_split > max_cores:
            raise Unsupported(
                f"Cannot satisfy mandatory split {min_split} for {var} within "
                f"{max_cores} cores."
            )
        if allowed_splits is not None and (
            var in allowed_splits and min_split not in allowed_splits[var]
        ):
            raise Unsupported(
                f"Mandatory split {min_split} for {var} is outside legal "
                f"domain {sorted(allowed_splits[var])}."
            )
        splits[var] = min_split
        cores_used *= min_split

    split_reduction_dims = [v for v in reduction_dims if splits[v] > 1]
    if len(split_reduction_dims) > 1:
        raise Unsupported(
            "The backend supports at most one split reduction dimension, got "
            f"{split_reduction_dims}."
        )

    for v in output_dims:
        if cores_used >= max_cores:
            break
        # Symbolic dims use granularity (divisibility invariant); concrete
        # dims use the concretised size. _valid_divisor_basis picks per dim.
        # TODO(issue#1372): remaining concrete sites use concretize_expr; once
        #                   symbolic work division is end-to-end, this comment
        #                   can be dropped.
        basis = _valid_divisor_basis(v, iteration_space, symbol_meta)
        best_split = _largest_legal_split_from(
            v,
            basis,
            splits[v],
            cores_used,
            max_cores,
            allowed_splits,
            min_splits,
        )
        if v in symbol_meta:
            logger.info(
                f"[work_division/symbolic] dim {v} (symbolic, max="
                f"{symbol_meta[v][0]}, gran={symbol_meta[v][1]}): "
                f"selected_split(basis={basis}, n_cores={max_cores // cores_used}) = "
                f"{best_split}"
            )
        if best_split > splits[v]:
            cores_used = cores_used // splits[v] * best_split
            splits[v] = best_split

    if is_reduction_included and cores_used < max_cores:
        eligible_reduction_dims = (
            split_reduction_dims if split_reduction_dims else reduction_dims
        )
        result = _most_splittable_dim(
            eligible_reduction_dims,
            iteration_space,
            splits,
            cores_used,
            max_cores,
            symbol_meta,
            allowed_splits,
            min_splits,
        )
        if result is not None:
            best_dim, best_split = result
            cores_used = cores_used // splits[best_dim] * best_split
            splits[best_dim] = best_split

    return splits


def adjust_it_space_for_sticks(
    it_space: dict[Symbol, Expr],
    tensor_deps: list[TensorDep],
    symbol_meta: SymbolMeta | None = None,
) -> tuple[dict[Symbol, Expr], dict[Symbol, int]]:
    """
    Return a copy of it_space with stick variables converted from elements to
    sticks, plus a dict mapping each stick variable to its max element per stick
    value.

    For each tensor, find the variable that indexes its stick dimension and
    convert its size in it_space from elements to sticks. This ensures work
    division treats sticks as atomic units.

    For QFP8WT tensors (2D stick layouts), both stick dimensions are treated as
    atomic units with 128-byte constraint.

    When tensors of different dtypes share a stick variable (e.g. a float16
    input and an int64 argmax output), the largest elems_per_stick is used
    so the adjustment is conservative (fewer sticks → smaller adjusted size →
    fewer cores assigned to the stick dimension).

    TODO: As of now, the stick dim cannot be symbolic. Granularity
    on a symbolic stick var would have to additionally be a multiple of
    ``elems_per_stick`` for the stick-count conversion to stay coherent; that
    is out of scope here. Raises ``Unsupported`` if any tensor's stick dim
    maps to a symbolic iteration variable.

    The original it_space is not mutated.
    """
    symbol_meta = symbol_meta or {}

    # Pass 1: find the largest elems_per_stick per stick variable.
    adjusted_space = dict(it_space)
    max_elems: dict[Symbol, int] = {}
    for td in tensor_deps:
        # Handle QFP8WT multi-dim stick
        if (
            hasattr(td.layout.device_layout, "element_arrangement")
            and td.layout.device_layout.element_arrangement == ElementArrangement.QFP8WT
        ):
            # For QFP8WT, last two device dimensions are the 2D stick [2, 64]
            # Both need to be treated as atomic 128-byte units
            stick_vars = []
            for coord in td.device_coords[-2:]:
                if len(coord.free_symbols) == 1:
                    var = next(iter(coord.free_symbols))
                    if var in adjusted_space:
                        stick_vars.append(var)

            for stick_var in stick_vars:
                # QFP8WT stick size is always 128 bytes (64 elements at fp8)
                fp8_stick_elems = td.layout.device_layout.elems_per_stick()
                if stick_var not in max_elems or fp8_stick_elems > max_elems[stick_var]:
                    max_elems[stick_var] = fp8_stick_elems
            continue

        stick_expr = td.device_coords[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        if stick_var not in adjusted_space:
            continue
        if stick_var in symbol_meta:
            logger.info(
                f"[work_division/symbolic] stick-dim guard raised: "
                f"stick_var={stick_var} on tensor {td.dep.name} is symbolic"
            )
            raise Unsupported(
                f"symbolic stick dim {stick_var} is not supported yet "
                f"(tensor {td.dep.name}); symbolic dims must be non-stick "
                f"(e.g. the leading batch dim)."
            )
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


def _is_indirectly_accessed(td: TensorDep) -> bool:
    """Return whether td has a data-dependent indirect coordinate."""
    return any(coord.has(IndirectAccess) for coord in td.device_coords[:-1])


def get_per_core_span(
    td: TensorDep,
    splits: dict[Symbol, int],
    it_space_orig: dict[Symbol, Expr],
    symbol_meta: SymbolMeta,
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

    For symbolic dims, the per-dim range ``R`` is the ``max_size`` from
    ``symbol_meta`` divided by the dim's split count — the worst-case runtime
    footprint that any compiled plan must remain legal against.
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
            # Per-core span is a hardware-bound quantity that must be checked
            # against MAX_SPAN_BYTES. For symbolic dims we use ``max_size``
            # (the worst-case footprint, also the HBM allocation footprint).
            R = _effective_size(v, it_space_orig, symbol_meta) // splits.get(v, 1)
            per_core_max += int(term.subs(v, R - 1))
            per_core_min += int(term.subs(v, 0))
        per_core_size = per_core_max - per_core_min + 1
        if per_core_size > 1:
            stride_elems = math.prod(device_size[d + 1 :])
            return per_core_size * stride_elems * itemsize
    return itemsize


def raise_if_per_core_overflow(
    tensor_deps: list[TensorDep],
    it_space_orig: dict[Symbol, Expr],
    splits: dict[Symbol, int],
    op_name: str,
    symbol_meta: SymbolMeta,
) -> None:
    """Raise Unsupported if any tensor's per-core memory span exceeds MAX_SPAN_BYTES."""
    for td in tensor_deps:
        if _is_indirectly_accessed(td):
            continue
        per_core_span = get_per_core_span(td, splits, it_space_orig, symbol_meta)
        if per_core_span > MAX_SPAN_BYTES:
            dl = td.layout.device_layout
            raise Unsupported(
                f"{op_name}: per-core tensor span "
                f"{per_core_span / (1024 * 1024):.3f} MB "
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
    symbol_meta: SymbolMeta,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
    blocked: set[Symbol] | None = None,
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

    For symbolic dims, ``symbol_meta`` supplies ``(max_size, granularity)``.
    The Cartesian search enumerates ``divisors(granularity)`` (not
    ``divisors(max_size)``) so every chosen split divides every admissible
    runtime bucket evenly. The span check itself uses ``max_size`` as the
    worst-case footprint.

    Args:
        tensor_deps: List of tensor dependencies to check
        it_space_orig: Original iteration space (element-valued)
        it_space_adjusted: Adjusted iteration space (stick-valued for stick vars)
        stick_vars: Mapping of stick variables to elements per stick
        max_cores: Maximum number of cores available
        symbol_meta: Per-symbol (max_size, granularity) for symbolic dims
        allowed_splits: Optional hard legal split factors per symbol
        blocked: Dimensions that must remain unsplit

    Returns a dict mapping Symbol -> number of slices.
    """
    # TODO: use compute_max_size(...) / compute_granularity(...) from pass_utils.py
    # for symbolic path. Refer to #2287 for details.
    accumulated_splits: dict[Symbol, int] = {}
    blocked = blocked or set()

    for td in tensor_deps:
        if (
            get_per_core_span(td, accumulated_splits, it_space_orig, symbol_meta)
            <= MAX_SPAN_BYTES
        ):
            continue

        for coord in td.device_coords[:-1]:
            # Concretize for the ``> 1`` comparison: with symbolic ranges,
            # ``s0 > 1`` returns a sympy Relational whose truth value is
            # undefined.  Span filtering here is a structural decision that
            # needs a concrete answer.
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
            split_vars = [
                v
                for v in coord.free_symbols
                if _effective_size(v, it_space_orig, symbol_meta) > 1
            ]
            if not split_vars:
                continue

            def valid_splits(v: Symbol) -> list[int]:
                if v in blocked:
                    return [1]
                current_min = accumulated_splits.get(v, 1)
                if v in symbol_meta:
                    basis = symbol_meta[v][1]
                elif v in stick_vars:
                    basis = concretize_expr(it_space_adjusted[v])
                else:
                    basis = concretize_expr(it_space_orig[v])
                return [
                    s
                    for s in _legal_split_factors(v, basis, allowed_splits)
                    if s >= current_min
                ]

            var_divisors = [valid_splits(v) for v in split_vars]

            for v, candidates in zip(split_vars, var_divisors):
                if not candidates:
                    raise Unsupported(
                        f"No valid split for variable {v} "
                        f"(orig_size={_effective_size(v, it_space_orig, symbol_meta)}, "
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
            best_within = None  # (span, combo)
            best_above = None  # (span, combo)

            for combo in itertools.product(*var_divisors):
                trial = dict(accumulated_splits)
                for v, s in zip(split_vars, combo):
                    trial[v] = s

                if math.prod(trial.values()) > max_cores:
                    continue

                span = get_per_core_span(td, trial, it_space_orig, symbol_meta)

                if span <= MAX_SPAN_BYTES:
                    if best_within is None or (math.prod(combo), -span) < (
                        math.prod(best_within[1]),
                        -best_within[0],
                    ):
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
            for v, s in zip(split_vars, best_combo):
                accumulated_splits[v] = s

            if best_span <= MAX_SPAN_BYTES:
                break

            # Still above the limit. If this coord still evaluates to > 1 under
            # the committed splits, inner dimensions cannot reduce the span further.
            # Use _effective_size so symbolic dims substitute their max_size
            # rather than a misleading optimization_hint.
            per_core_coord_size = (
                max(
                    int(
                        coord.subs(
                            {
                                v: _effective_size(v, it_space_orig, symbol_meta)
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


def prioritize_indirect_scatter_dimensions(
    op: ComputedBuffer,
    output: TensorDep,
    it_space_adjusted: dict[Symbol, Expr],
    symbol_meta: SymbolMeta | None = None,
) -> tuple[list[Symbol], list[Symbol]]:
    """Prioritize overwrite-scatter entry dims as output work.

    Scatter's runtime-selected destination row hides its entry dims from output
    coordinates. They otherwise look like reductions and never split for this
    non-reduction op. This is priority policy, not a legality constraint.
    """
    output_dims, reduction_dims = prioritize_dimensions(
        output, it_space_adjusted, symbol_meta
    )
    from .pass_utils import indirect_store_entry_syms

    entry_dims = indirect_store_entry_syms(op)
    promoted = [dim for dim in reduction_dims if dim in entry_dims]
    return promoted + output_dims, [
        dim for dim in reduction_dims if dim not in entry_dims
    ]


def prioritize_dimensions(
    output: TensorDep,
    it_space_adjusted: dict[Symbol, Expr],
    symbol_meta: SymbolMeta | None = None,
) -> tuple[list[Symbol], list[Symbol]]:
    """Partition iteration variables into output dims and reduction dims.

    Output dims are those whose symbols appear in the output tensor's device
    coordinate expressions (excluding the stick coordinate). Reduction dims are
    the remainder. Both lists are sorted by decreasing size — for symbolic
    dims the canonical size is ``max_size`` from ``symbol_meta``, preserving
    the existing "largest-output-dim-first" policy under the extension that a
    symbolic dim's size is its bucket upper bound.

    Variables already committed as min_splits should be filtered out of
    it_space_adjusted before calling this function.
    """
    symbol_meta = symbol_meta or {}
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    output_pairs: list[tuple[Symbol, Expr]] = []
    reduction_pairs: list[tuple[Symbol, Expr]] = []
    for s, e in it_space_adjusted.items():
        (output_pairs if s in coord_vars else reduction_pairs).append((s, e))

    # Sort by decreasing size (concrete for static dims, max_size for symbolic).
    def _size_key(t: tuple[Symbol, Expr]) -> int:
        sym, _ = t
        return _effective_size(sym, it_space_adjusted, symbol_meta)

    output_pairs.sort(key=_size_key, reverse=True)
    reduction_pairs.sort(key=_size_key, reverse=True)

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
    rw = op_read_writes(op)
    output_td = TensorDep(next(iter(rw.writes)), _resolve_layout(op))
    return input_tds, output_td


def apply_splits(op: ComputedBuffer, splits: dict) -> None:
    """Commit symbol-keyed work division; scheduler transport is finalized later."""
    commit_iteration_space_ownership(op, splits)


def enumerate_work_division_candidates(
    op: ComputedBuffer,
    max_cores: int,
) -> list[dict[Symbol, int]]:
    """Return every permissible core-division split for ``op``.

    A split (``dict[Symbol, int]``, same form as :func:`apply_splits`) is
    permissible iff: each per-dim factor divides its dim's size,
    ``prod(factors) <= max_cores``, every tensor's per-core span
    ``<= MAX_SPAN_BYTES``, and at most one reduction (K) dim is split. A factor
    of ``1`` means the dim is unsplit; the all-ones single-core split is
    included when it is itself permissible.

    Op-specific limits are supplied by ``collect_work_division_constraints``;
    this planner only consumes their generic legal split domains.
    """
    # TODO: Enumerate compute bound ops and for seeds or compute optimized
    # work division where HBM bandwidth can saturate compute.

    it_space = iteration_space_from_op(op)

    input_tds, output_td = collect_tensor_deps(
        op,
        _apply_input_layout_overrides(op, get_mem_deps_from_rw(op_read_writes(op))),
    )
    all_tds = input_tds + [output_td]

    symbol_meta = _collect_symbol_metadata(it_space)
    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )

    # Reduction (K) dims are absent from output device coordinates.
    coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]

    constraint_result = collect_work_division_constraints(
        WorkDivConstraintContext(
            op=op,
            it_space=it_space,
            it_space_adjusted=it_space_adjusted,
            output_td=output_td,
            input_tds=input_tds,
            stick_vars=stick_vars,
            reduction_vars=reduction_vars,
            committed_splits={},
        )
    )
    blocked = constraint_result.blocked
    allowed_splits = constraint_result.allowed_splits

    # Per-dim candidate factors, mirroring must_split_vars.valid_splits but with
    # no ``>= current_min`` floor (we want the full set, including 1).
    def factors(v: Symbol) -> list[int]:
        if v in symbol_meta:
            basis = symbol_meta[v][1]  # granularity
        elif v in stick_vars:
            basis = concretize_expr(it_space_adjusted[v])  # stick count
        else:
            basis = concretize_expr(it_space[v])  # element count
        return _legal_split_factors(v, basis, allowed_splits, _span_min_splits(op))

    def create_splits(axis_vars, combo):
        return dict(zip(axis_vars, combo))

    def valid_split(splits):
        if math.prod(splits.values()) > max_cores:  # core budget
            return False
        if sum(1 for v in reduction_vars if splits[v] > 1) > 1:  # <= 1 K-split
            return False
        if any(  # per-core span <= MAX_SPAN_BYTES, on element-valued it_space
            get_per_core_span(td, splits, it_space, symbol_meta) > MAX_SPAN_BYTES
            for td in all_tds
        ):
            return False
        if any(  # a coordinate-masked dim cannot be split across cores
            splits.get(v, 1) > 1 for v in blocked
        ):
            return False
        return all(splits.get(v, 1) in allowed for v, allowed in allowed_splits.items())

    vars_ = list(it_space_adjusted.keys())
    candidates = [
        splits
        for combo in itertools.product(*(factors(v) for v in vars_))
        if valid_split(splits := create_splits(vars_, combo))
    ]

    return candidates


def work_division_splits_are_legal(
    op: ComputedBuffer, splits: dict[Symbol, int]
) -> bool:
    """Return whether symbol-keyed splits obey this op's hard constraints."""
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    if not isinstance(layout, FixedTiledLayout):
        return True

    it_space = iteration_space_from_op(op)
    rw = op_read_writes(op)
    input_tds, output_td = collect_tensor_deps(
        op, _apply_input_layout_overrides(op, get_mem_deps_from_rw(rw))
    )
    symbol_meta = _collect_symbol_metadata(it_space)
    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, input_tds + [output_td], symbol_meta
    )
    coord_vars = {
        v
        for expr in output_td.device_coords[:-1]
        for v in expr.free_symbols
        if isinstance(v, Symbol)
    }
    result = collect_work_division_constraints(
        WorkDivConstraintContext(
            op=op,
            it_space=it_space,
            it_space_adjusted=it_space_adjusted,
            output_td=output_td,
            input_tds=input_tds,
            stick_vars=stick_vars,
            reduction_vars=[v for v in it_space_adjusted if v not in coord_vars],
            committed_splits={},
        )
    )
    min_splits = _span_min_splits(op)
    split_reduction_dims = [
        v for v in it_space_adjusted if v not in coord_vars and splits.get(v, 1) > 1
    ]
    return (
        len(split_reduction_dims) <= 1
        and not any(splits.get(v, 1) > 1 for v in result.blocked)
        and all(
            splits.get(v, 1) in allowed for v, allowed in result.allowed_splits.items()
        )
        and all(splits.get(v, 1) >= minimum for v, minimum in min_splits.items())
    )


def _work_div_hint_by_name(op: ComputedBuffer) -> dict[str, int]:
    dim_to_split: dict[str, int] = {}
    for _, hint_dict in sorted(get_op_hints(op).items()):
        dim_to_split.update(hint_dict.get("work_div") or {})
    return dim_to_split


def _has_work_div_hint(op: ComputedBuffer) -> bool:
    return any(hint_dict.get("work_div") for hint_dict in get_op_hints(op).values())


def _resolve_work_div_hint(
    op: ComputedBuffer,
    it_space: dict[Symbol, Expr],
) -> dict[Symbol, int] | None:
    dim_to_split = _work_div_hint_by_name(op)
    if not dim_to_split:
        return None

    loop_var_dims = getattr(op, "work_div_loop_info", {})
    splits: dict[Symbol, int] = {}
    for name, split in dim_to_split.items():
        for sym in it_space:
            if sym in splits:
                continue
            if name in loop_var_dims.get(sym, []):
                splits[sym] = split
                break
    return splits if splits else None


def _apply_user_hint(
    op: ComputedBuffer,
    user_splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    max_cores: int,
    blocked: set[Symbol] | None = None,
    allowed_splits: dict[Symbol, frozenset[int]] | None = None,
) -> dict[Symbol, int]:
    """Apply legal splits in insertion order, pruning lower-priority overflows."""
    op_name = op.get_name()
    blocked = blocked or set()
    allowed_splits = allowed_splits or {}
    min_splits = _span_min_splits(op)

    splits: dict[Symbol, int] = {}
    cores_used = 1
    loop_var_dims = getattr(op, "work_div_loop_info", {})
    for sym, split_val in user_splits.items():
        # bool is an int subclass in Python, but it is not a meaningful split.
        if isinstance(split_val, bool) or not isinstance(split_val, (int, Integer)):
            raise Unsupported(
                f"work_division_hint: {op_name} split value {split_val!r} "
                f"for dim {sym} must be an integer."
            )
        split = int(split_val)
        if split < 1:
            raise Unsupported(
                f"work_division_hint: {op_name} split value {split!r} "
                f"for dim {sym} must be positive."
            )
        if sym not in it_space_adjusted:
            raise Unsupported(
                f"work_division_hint: {op_name} dim {sym} is not in the "
                f"work-division iteration space."
            )
        if split > 1 and sym in blocked:
            raise Unsupported(
                f"work_division_hint: {op_name} cannot split constrained dim {sym}."
            )
        if sym in allowed_splits and split not in allowed_splits[sym]:
            raise Unsupported(
                f"work_division_hint: {op_name} dim {sym} legal splits are "
                f"{sorted(allowed_splits[sym])}."
            )
        if split < min_splits.get(sym, 1):
            raise Unsupported(
                f"work_division_hint: {op_name} dim {sym} must split at least "
                f"{min_splits[sym]} ways for the hardware memory-span limit."
            )

        next_cores = cores_used * split
        if next_cores > max_cores:
            logger.info(
                "work_division_hint: %s skipping named dim(s) %s (split=%s) "
                "because cores would be %s, exceeding SENCORES=%s",
                op_name,
                loop_var_dims.get(sym, []),
                split,
                next_cores,
                max_cores,
            )
            continue

        dim_size = concretize_expr(it_space_adjusted[sym])
        if dim_size % split != 0:
            raise Unsupported(
                f"work_division_hint: {op_name} dim {sym} size={dim_size} "
                f"is not evenly divisible by split={split}."
            )

        splits[sym] = split
        cores_used = next_cores

    coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    reduction_vars_to_split = {
        sym for sym, split in splits.items() if split > 1 and sym not in coord_vars
    }
    if len(reduction_vars_to_split) > 1:
        raise Unsupported(
            f"work_division_hint: {op_name} splits "
            f"{len(reduction_vars_to_split)} reduction dimensions "
            f"({reduction_vars_to_split}), but the backend supports at most 1."
        )

    conflicting_domains = {
        sym: allowed
        for sym, allowed in allowed_splits.items()
        if splits.get(sym, 1) not in allowed
    }
    below_span_floor = {
        sym: minimum
        for sym, minimum in min_splits.items()
        if splits.get(sym, 1) < minimum
    }
    if conflicting_domains or below_span_floor:
        raise Unsupported(
            f"work_division_hint: {op_name} conflicts with legal split domains "
            f"{conflicting_domains} or span floors {below_span_floor}."
        )

    return splits


def _commit_user_splits(op: ComputedBuffer, splits: dict[Symbol, int]) -> None:
    apply_splits(op, splits)


def span_reduction_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Mandatory per-op pass: compute hard minimum splits for MAX_SPAN_BYTES.

    Writes symbol-keyed ownership and persists the span floors separately so
    later planning may choose any legal factor at or above each floor. Unity
    splits are retained so later passes have a complete operation iteration-space
    ownership record.

    For indirect-access ops (gather / scatter), shared-table data dimensions
    (K, N of the value table or the scatter destination) have legal split domain
    `{1}`. Splitting those dims would give every core a different base address
    into the shared table, producing wrong results. If the memory span cannot be
    reduced within the hard domains, the pass raises `Unsupported`.
    """
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    # Symbolic-dim bucket metadata for the iteration vars in this op. Built
    # before stick adjustment so the stick-dim guard inside
    # adjust_it_space_for_sticks can raise on symbolic stick dims.
    symbol_meta = _collect_symbol_metadata(it_space)

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )
    coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]
    constraints = collect_work_division_constraints(
        WorkDivConstraintContext(
            op=op,
            it_space=it_space,
            it_space_adjusted=it_space_adjusted,
            output_td=output_td,
            input_tds=input_tds,
            stick_vars=stick_vars,
            reduction_vars=reduction_vars,
            committed_splits={},
        )
    )
    min_splits = must_split_vars(
        all_tds,
        it_space,
        it_space_adjusted,
        stick_vars,
        max_cores,
        symbol_meta,
        constraints.allowed_splits,
        constraints.blocked,
    )
    collect_work_division_constraints(
        WorkDivConstraintContext(
            op=op,
            it_space=it_space,
            it_space_adjusted=it_space_adjusted,
            output_td=output_td,
            input_tds=input_tds,
            stick_vars=stick_vars,
            reduction_vars=reduction_vars,
            committed_splits=min_splits,
        )
    )

    reduction_vars_to_split = {
        v for v, split in min_splits.items() if split > 1 and v not in coord_vars
    }
    # Each entry in Reduction.reduction_ranges maps to at most one Symbol via
    # index_vars_squeeze (size-1 entries are squeezed away). So len > 1 means
    # genuinely distinct reduction dimensions, not multiple symbols from one dim.
    if len(reduction_vars_to_split) > 1:
        raise Unsupported(
            f"Cannot satisfy hardware memory span limit "
            f"({MAX_SPAN_BYTES / (1024**2):.3f}MB) without splitting "
            f"{len(reduction_vars_to_split)} reduction dimension(s) "
            f"({reduction_vars_to_split}), but the backend supports at most 1."
        )

    op._work_division_span_min_splits = dict(min_splits)
    apply_splits(op, min_splits)

    if symbol_meta and math.prod(min_splits.values()) > 1:
        logger.info(
            f"[work_division/symbolic] span_reduction {op.get_name()}: "
            f"committed min_splits={ {str(k): v for k, v in min_splits.items()} }, "
            f"cores={math.prod(min_splits.values())}"
        )
    if logger.isEnabledFor(logging.DEBUG) and math.prod(min_splits.values()) > 1:
        logger.debug(
            f"span_reduction work_division {op.get_name()}: cores={math.prod(min_splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"symbol_meta={symbol_meta}, priorities=[], min_splits={min_splits}"
        )


def _default_split(
    op: ComputedBuffer,
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    committed_splits: dict[Symbol, int],
    max_cores: int,
    symbol_meta: SymbolMeta,
    blocked: set[Symbol],
    allowed_splits: dict[Symbol, frozenset[int]],
) -> tuple[dict[Symbol, int], list[Symbol], list[Symbol]]:
    """Distribute max_cores by priority on top of span_reduction's commits.

    Returns the chosen splits and the (output, reduction) priority dims the
    caller logs. Shared by work_distribution_pass and cost_model_matmul_division.
    """
    output_dims, reduction_dims = prioritize_indirect_scatter_dimensions(
        op, output_td, it_space_adjusted, symbol_meta
    )

    # If span reduction already committed a reduction split, grow only that
    # dimension; backend supports one reduction dimension split per op.
    coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    committed_reduction_vars = {v for v in committed_splits if v not in coord_vars}
    if committed_reduction_vars:
        reduction_dims = [v for v in reduction_dims if v in committed_reduction_vars]

    # Drop blocked dims before the greedy distributor commits them. Coordinate
    # masking only blocks reduction dims; strided conv also blocks output dims.
    output_dims = [v for v in output_dims if v not in blocked]
    reduction_dims = [v for v in reduction_dims if v not in blocked]

    # Pass max_cores, not remaining_cores: multi_dim_iteration_space_split
    # accounts for committed_splits in its first pass, consuming those cores
    # itself before distributing the rest by priority.
    splits = multi_dim_iteration_space_split(
        it_space_adjusted,
        max_cores,
        output_dims,
        reduction_dims,
        committed_splits,
        symbol_meta,
        allowed_splits,
    )
    return splits, output_dims, reduction_dims


def work_distribution_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Optional per-op pass: distribute remaining cores to maximize parallelism.

    Reads symbol-keyed ownership written by span_reduction_pass (if any), then
    fills remaining cores by priority.
    """
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    symbol_meta = _collect_symbol_metadata(it_space)

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )

    ownership = getattr(op, "iteration_space_ownership", None)
    committed_splits = (
        {s: v for s, v in ownership.work_slices.items() if v > 1}
        if ownership is not None
        else {}
    )

    coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]
    constraint_result = collect_work_division_constraints(
        WorkDivConstraintContext(
            op=op,
            it_space=it_space,
            it_space_adjusted=it_space_adjusted,
            output_td=output_td,
            input_tds=input_tds,
            stick_vars=stick_vars,
            reduction_vars=reduction_vars,
            committed_splits=committed_splits,
        )
    )
    blocked = constraint_result.blocked
    allowed_splits = constraint_result.allowed_splits

    if not config.ignore_work_division_hints:
        user_splits = _resolve_work_div_hint(op, it_space_adjusted)
        if user_splits is not None:
            user_splits = _apply_user_hint(
                op,
                user_splits,
                it_space_adjusted,
                output_td,
                max_cores,
                blocked,
                allowed_splits,
            )
            _commit_user_splits(op, user_splits)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"work_distribution(user-hint) work_division {op.get_name()}: "
                    f"cores={math.prod(user_splits.values())}, "
                    f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
                    f"min_splits={committed_splits}, user_splits={user_splits}"
                )
            raise_if_per_core_overflow(
                all_tds, it_space, user_splits, op.get_name(), symbol_meta
            )
            return

    splits, output_dims, reduction_dims = _default_split(
        op,
        it_space_adjusted,
        output_td,
        committed_splits,
        max_cores,
        symbol_meta,
        blocked,
        allowed_splits,
    )

    apply_splits(op, splits)

    if symbol_meta and math.prod(splits.values()) > 1:
        logger.info(
            f"[work_division/symbolic] work_distribution {op.get_name()}: "
            f"final splits={ {str(k): v for k, v in splits.items()} }, "
            f"cores={math.prod(splits.values())}, "
            f"priorities={[str(d) for d in output_dims + reduction_dims]}"
        )
    if logger.isEnabledFor(logging.DEBUG) and math.prod(splits.values()) > 1:
        logger.debug(
            f"work_distribution work_division {op.get_name()}: cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"symbol_meta={symbol_meta}, "
            f"priorities={output_dims + reduction_dims}, min_splits={committed_splits}"
        )

    raise_if_per_core_overflow(all_tds, it_space, splits, op.get_name(), symbol_meta)


def max(*args, **kwargs):
    """``max``, but symbolic-aware: dispatches to ``sympy.Max`` when an arg is
    a sympy expression (whose truth-valued comparisons the builtin can't
    resolve), otherwise defers to the builtin -- including its ``key``/
    ``default`` kwargs and single-iterable form, neither of which ``sympy.Max``
    supports."""
    if any(isinstance(a, sympy.Basic) for a in args):
        return sympy.Max(*args)
    return builtins.max(*args, **kwargs)


def min(*args, **kwargs):
    """``min`` counterpart of :func:`max`; see its docstring."""
    if any(isinstance(a, sympy.Basic) for a in args):
        return sympy.Min(*args)
    return builtins.min(*args, **kwargs)


def log2(arg):
    """``log2`` counterpart of :func:`max`; see its docstring."""
    if isinstance(arg, sympy.Basic):
        if isinstance(arg, sympy.Rational):
            return sympy.log(arg.n(), 2.0)
        else:
            return sympy.log(arg, 2.0)
    return math.log2(arg)


_PT_ROWS = 8  # PT block rows per corelet

# Constants for the matmul cost model (_matmul_split_cost). Each is either an
# AIU hardware limit or a coefficient fit to measured device kernel times.
_TARGET_PT_PASSES = 5  # per-core M that keeps the PT pipeline full = this * _PT_ROWS
_TARGET_M_TIE_PASSES = 4  # enough M lanes to keep the stationary weights fed
_PT_EFFICIENCY_EXPONENT = 0.25
_M_MIN = _PT_ROWS // 2  # below half a PT pass an m-split buys nothing
_PEAK_MACS_US_CORE = (98.304e12 / 2 / 32) / 1e6  # DL16 peak / 32 cores, MACs/us/core
_HBM_BW_GBS = 204.8  # LPDDR5 aggregate peak bandwidth
_DTYPE_BYTES = 2  # fp16
_PSUM_PER_CORE_ELEM_US = 1.0e-3
_BMM_PSUM_PER_CORE_ELEM_US = 1.0e-4
_COHORT_LIMIT = 8  # cores sharing a broadcast before it contends for bandwidth
_COHORT_PENALTY_EXPONENT = 0.75
_M_LANE_UNDERUSE_PENALTY_US = 10.0  # tie-break when too few M lanes are used
_M_TILE_UNDERFILL_TARGET = 16  # rows/core below this pay PT startup overhead
_M_TILE_UNDERFILL_PENALTY_US = 30.0
_TARGET_N_TILE_ELEMS = 512  # per-core N wider than this loses schedule efficiency
_WIDE_N_TILE_PENALTY_US = 25.0  # per log2 step over _TARGET_N_TILE_ELEMS
_CORE_UNDERUSE_PENALTY_US = (
    150.0  # soft replacement for the old hard full-core fallback
)
_BMM_BATCH_SPLIT_PENALTY_US = 10.0  # true-BMM batch split cost per log2 step
_LARGE_M_TILE_SHAPE_PENALTY_US = 20.0
_SHARED_DOWN_N_SPLIT_PENALTY_US = 10.0
_SHARED_NARROW_OUTPUT_REF = _TARGET_N_TILE_ELEMS * _COHORT_LIMIT
_SHARED_N_TILE_TARGET = _TARGET_N_TILE_ELEMS // 4


def _matmul_split_cost(
    b_axis: tuple[int, int],
    m_axis: tuple[int, int],
    n_axis: tuple[int, int],
    k_axis: tuple[int, int],
    max_cores: int,
    shared_weight: bool = False,
    include_hbm: bool = True,
) -> float:
    """Estimated kernel time in microseconds for ``[B,M,K]@[B,K,N]`` run with
    the given core split. Each axis is a ``(size, split)`` pair so a dim's size
    cannot be paired with another dim's split. Lower is better; inf if infeasible.

    ``include_hbm=False`` drops the operand/output HBM-traffic term for a caller
    that charges that traffic itself (``cost_model._matmul_ns_upstream``, whose
    bundle memory term counts the same bytes and knows about LX residency). The
    cohort bandwidth penalty scales only that term, so it drops out with it.
    """
    (B, b), (M, m), (N, n), (K, k) = b_axis, m_axis, n_axis, k_axis
    cores_used = b * m * n * k
    if cores_used == 0 or (isinstance(cores_used, int) and cores_used > max_cores):
        return math.inf

    num_elems = B * M * N * K
    is_symbolic = isinstance(cores_used, sympy.Basic) or isinstance(
        num_elems, sympy.Basic
    )

    # Compute: per-core MACs over peak, derated when the per-core M tile is too
    # short to fill the PT pipeline. The PT array streams M in passes of
    # _PT_ROWS; below _TARGET_PT_PASSES passes its startup/drain overhead is
    # amortised over too little work, and that overhead grows sub-linearly.
    m_t = M // m if m else 1
    pt_passes = max(1.0, m_t / _PT_ROWS)
    pt_eff = (
        # TODO: allow symbolic
        1.0
        if is_symbolic
        else min(1.0, (pt_passes / _TARGET_PT_PASSES) ** _PT_EFFICIENCY_EXPONENT)
    )
    compute_us = (num_elems / cores_used) / (_PEAK_MACS_US_CORE * pt_eff)

    # HBM: every input operand is broadcast to the cohort of cores splitting the
    # orthogonal dim. Past _COHORT_LIMIT the broadcasts contend for the shared
    # link, so effective bandwidth falls off linearly with cohort size.
    if include_hbm:
        weight_batches = 1 if shared_weight else B
        bytes_total = (B * M * K + weight_batches * K * N + B * M * N) * _DTYPE_BYTES
        fanout_split = max(m, n) if shared_weight else n
        # TODO: Remove special casing symbolic
        cohort_penalty = (
            1.0
            if is_symbolic
            else max(1.0, (fanout_split / _COHORT_LIMIT) ** _COHORT_PENALTY_EXPONENT)
        )
        hbm_us = bytes_total / (_HBM_BW_GBS * 1000) * cohort_penalty
    else:
        hbm_us = 0.0

    # PSUM: a K-split spreads the reduction over k cores, costing (k-1)
    # partial-sum hops. Charge each core's output tile rather than the whole
    # output, so useful K-splits are not over-penalized.
    psum_coeff = _PSUM_PER_CORE_ELEM_US if shared_weight else _BMM_PSUM_PER_CORE_ELEM_US
    output_elems_per_core = (B * M * N) / (b * m * n)
    psum_us = max(0, k - 1) * output_elems_per_core * psum_coeff

    # Tie-break: among compute-equivalent splits prefer exposing enough M lanes
    # to stream work over the stationary weight tile. PT efficiency above handles
    # the opposite case where an M split makes each per-core tile too short.
    target_m = max(
        _M_MIN,
        min(max_cores // 2, max(1, M // (_TARGET_M_TIE_PASSES * _PT_ROWS))),
    )
    m_lane_underuse_us = max(0.0, log2(target_m / m)) * _M_LANE_UNDERUSE_PENALTY_US
    m_tile_underfill_us = (
        max(0.0, log2(_M_TILE_UNDERFILL_TARGET / max(1, m_t)))
        * _M_TILE_UNDERFILL_PENALTY_US
    )

    # Very wide output tiles lose schedule efficiency in the generated kernel.
    # Charge only the over-wide side so short-M and small/moderate-N choices
    # are not pulled away from PT-friendly M tiles.
    n_t = N // n if n else N
    wide_n_us = (
        max(0.0, log2(max(1, n_t) / _TARGET_N_TILE_ELEMS)) * _WIDE_N_TILE_PENALTY_US
    )

    # Once M is large enough to feed the PT, prefer tile shapes that avoid
    # unnecessary layout/fusion fallout. For true BMMs this means avoiding
    # splitting a tiny output dimension when the reduction dimension is much
    # larger (value-matmul geometry: K >> N). For shared-weight projections this
    # means avoiding very wide per-core N tiles when the whole projection is
    # narrow enough that more N lanes are available. Both effects are expressed
    # as ratios rather than op names or workload-specific shapes.
    # filled_m_tile_factor = 1.0 if m_t >= _M_TILE_UNDERFILL_TARGET else 0.0
    # TODO: Remove special casing symbolic
    if not is_symbolic:
        filled_m_tile_factor = 1.0 if m_t >= _M_TILE_UNDERFILL_TARGET else 0.0
        true_bmm_value_split_us = (
            0.0
            if shared_weight or n <= 1
            else filled_m_tile_factor
            * max(0.0, log2(max(1, K) / max(1, N)))
            * log2(n)
            * _LARGE_M_TILE_SHAPE_PENALTY_US
        )
        shared_narrow_tile_us = (
            0.0
            if not shared_weight
            else filled_m_tile_factor
            * max(0.0, log2(_SHARED_NARROW_OUTPUT_REF / max(1, N)))
            * max(0.0, log2(max(1, n_t) / _SHARED_N_TILE_TARGET))
            * (_LARGE_M_TILE_SHAPE_PENALTY_US / 4)
        )
        shared_down_n_split_us = (
            0.0
            if not shared_weight or n <= 1
            else max(0.0, log2(max(1, K) / max(1, N)))
            * log2(n)
            * _SHARED_DOWN_N_SPLIT_PENALTY_US
        )
        large_m_tile_shape_us = (
            true_bmm_value_split_us + shared_narrow_tile_us + shared_down_n_split_us
        )
    else:
        large_m_tile_shape_us = 0.0

    # Prefer using the full core budget, but keep this soft so measured-good
    # lower-core candidates can still win.
    core_underuse_us = (
        max(0.0, log2(max_cores / cores_used)) * _CORE_UNDERUSE_PENALTY_US
    )

    # True BMMs often need batch parallelism to avoid tiny-M underfill. Charge a
    # small additive split overhead instead of multiplying the whole estimate.
    batch_split_us = 0.0 if shared_weight else log2(b) * _BMM_BATCH_SPLIT_PENALTY_US

    return (
        compute_us
        + hbm_us
        + psum_us
        + m_lane_underuse_us
        + m_tile_underfill_us
        + wide_n_us
        + large_m_tile_shape_us
        + core_underuse_us
        + batch_split_us
    )


def _single_input_row_dims(
    row_dims: list[Symbol],
    input_tds: list[TensorDep],
) -> list[Symbol]:
    def _appears_in_one_input(dim: Symbol) -> bool:
        hits = sum(
            dim in {v for e in td.device_coords for v in e.free_symbols}
            for td in input_tds
        )
        return hits == 1

    return [d for d in row_dims if _appears_in_one_input(d)]


def _pick_innermost_output_dim(
    dims: list[Symbol],
    output_index: Expr,
) -> Symbol | None:
    """Pick the row dim nearest the output's contiguous/stick dimension.

    Shared 2D weights are represented as broadcast views. Their stride-0 batch
    dimensions disappear from device coordinates, so both batch dims and the
    true M dim can look like "LHS-only" row dims. In the output's flat host
    index, the true M dim is the innermost row dimension: it has the smallest
    non-zero coefficient, while batch dims have coefficients multiplied by M
    and/or outer batch extents.
    """

    candidates: list[tuple[int, Symbol]] = []
    for dim in dims:
        coeff = output_index.coeff(dim)
        if coeff == 0:
            continue
        candidates.append((abs(concretize_expr(coeff)), dim))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _cost_model_matmul_planner(
    op: ComputedBuffer,
    splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    max_cores: int,
    input_tds: list[TensorDep],
    blocked: set[Symbol],
    allowed_splits: dict[Symbol, frozenset[int]],
) -> dict[Symbol, int]:
    """Override the default split for a matmul / bmm with the lowest-cost
    feasible (b, m, n, k) per _matmul_split_cost.

    Returns ``splits`` unchanged for anything this planner does not model:
    non-matmuls, ops with a span-committed split already in place, multi-K
    matmuls, or a chosen split that would use fewer cores than the default.
    """
    if not isinstance(op.data, Reduction):
        return splits
    if op.data.reduction_type not in (BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP):
        return splits
    if committed_splits:
        return splits

    # Classify the output coord dims: the stickified one is N, the rest index
    # rows. Of those row dims, M is the one appearing in a single input (the
    # LHS); batch dims appear in both.
    output_coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    ordered_output_coord_vars = [d for d in it_space_adjusted if d in output_coord_vars]
    n_dims = [d for d in ordered_output_coord_vars if d in stick_vars]
    row_dims = [d for d in ordered_output_coord_vars if d not in stick_vars]
    if len(n_dims) != 1 or not row_dims:
        return splits
    n_dim = n_dims[0]

    m_candidates = _single_input_row_dims(row_dims, input_tds)
    rhs_loaded_once = False
    if len(m_candidates) == 1:
        m_dim = m_candidates[0]
    elif len(m_candidates) > 1:
        m_dim = _pick_innermost_output_dim(m_candidates, output_td.dep.index)
        if m_dim is None:
            return splits
        rhs_loaded_once = True
    else:
        return splits
    batch_dims = [d for d in row_dims if d != m_dim]
    # Folded projection matmuls have no batch dims left by this stage, but the
    # RHS is still an unbatched weight that is loaded once. Treat them like the
    # broadcast/shared-weight path for cost purposes while keeping true BMMs
    # where RHS depends on batch dims on the non-shared path.
    if not batch_dims:
        rhs_loaded_once = True

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

    span_min_splits = _span_min_splits(op)

    def factors(dim: Symbol, size: int) -> list[int]:
        return (
            [1]
            if dim in blocked
            else _legal_split_factors(dim, size, allowed_splits, span_min_splits)
        )

    b_combos = (
        list(
            itertools.product(
                *(factors(dim, size) for dim, size in zip(batch_dims, batch_sizes))
            )
        )
        if batch_dims
        else [()]
    )
    m_divs = factors(m_dim, M_e)
    n_divs = factors(n_dim, n_sticks)
    k_divs = factors(k_dim, k_sticks)

    best = None
    best_cost = math.inf
    for b_combo in b_combos:
        b_prod = math.prod(b_combo)
        for mm in m_divs:
            for nn in n_divs:
                for kk in k_divs:
                    if b_prod * mm * nn * kk > max_cores:
                        continue
                    c = _matmul_split_cost(
                        (B_total, b_prod),
                        (M_e, mm),
                        (N_e, nn),
                        (K_e, kk),
                        max_cores,
                        shared_weight=rhs_loaded_once,
                    )
                    if c < best_cost:
                        best_cost = c
                        best = (b_combo, mm, nn, kk)

    if best is None:
        return splits

    b_combo, m_s, n_s, k_s = best
    new_splits = dict(splits)
    for bd, bs in zip(batch_dims, b_combo):
        new_splits[bd] = bs
    new_splits[m_dim] = m_s
    new_splits[n_dim] = n_s
    new_splits[k_dim] = k_s

    # Never trade down to fewer cores than the default distributor already found.
    if math.prod(new_splits.values()) < math.prod(splits.values()):
        if not has_qfp8wt_tensor(input_tds + [output_td]):
            return splits
        # For QFP8WT, force k_dim = 1 regardless of core count
        new_splits[k_dim] = 1

    logger.debug(
        f"cost_model work_division {op.get_name()}: "
        f"b={b_combo} m={m_s} n={n_s} k={k_s} rhs_loaded_once={rhs_loaded_once} "
        f"cost={best_cost:.1f}us "
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
    pass_fn(op, args, max_cores)


def _validate_max_cores() -> int:
    max_cores = config.sencores
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")
    return max_cores


def _iter_computed_buffers(operations: list[Operation]):
    """Yield ComputedBuffer ops, handling FallbackKernel/ExternKernel dispatch."""
    for op in operations:
        if op.is_no_op():
            pass
        elif isinstance(op, ComputedBuffer):
            layout = op.maybe_get_layout()
            if layout is None or layout.device.type != DEVICE_NAME:
                continue
            yield op
        elif isinstance(op, FallbackKernel):
            # FallbackKernel produces 0..N trailing MultiOutputs
            # (see torch_spyre/_inductor/propagate_layouts.py).
            # Work division is not supported on either; the MultiOutputs
            # are skipped in their own branch below.
            pass
        elif isinstance(op, MultiOutput):
            pass
        elif isinstance(op, ExternKernel):
            if isinstance(op, (SpyreConstantFallback, SpyreEmptyFallback, DeviceCopy)):
                # Work division not supported on allocation/constant kernels, nor
                # on DeviceCopy.
                pass
            elif isinstance(
                op,
                (
                    BroadcastAsyncFallback,
                    WaitWorkFallback,
                    AllGatherAsyncFallback,
                    AllReduceAsyncFallback,
                ),
            ):
                pass
            else:
                logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")


def _apply_input_layout_overrides(
    op: ComputedBuffer, args: list[SchedNodeArg]
) -> list[SchedNodeArg]:
    """Apply per-op input layout overrides stored in op._input_layout_overrides.

    insert_post_mutation_restickify uses this to make work division treat an
    input buffer with an override layout instead of its committed layout.

    The same tag is also used by SpyreKernel.create_tensor_arg, so work
    division and codegen agree on the input layout.
    """
    overrides: dict[str, FixedTiledLayout] = getattr(op, "_input_layout_overrides", {})
    if not overrides:
        return args
    return [SchedNodeArg(a.dep, overrides.get(a.dep.name, a.layout)) for a in args]


def span_reduction(graph: GraphLowering) -> None:
    """Pass 1: compute minimum per-op splits required by MAX_SPAN_BYTES."""
    operations = graph.operations
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        rw = op_read_writes(op)
        args = _apply_input_layout_overrides(op, get_mem_deps_from_rw(rw))
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, span_reduction_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, span_reduction_pass)


def work_distribution(
    graph: GraphLowering,
    preassigned_ops: list[Operation] | None = None,
) -> None:
    """Pass 3: distribute remaining cores across ops to maximize parallelism.

    Ops in `preassigned_ops` were already divided by cost_model_matmul_division;
    they are left untouched so every op is divided by exactly one pass.
    """
    operations = graph.operations
    preassigned_ops = preassigned_ops or []
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        if op in preassigned_ops:
            continue
        rw = op_read_writes(op)
        args = _apply_input_layout_overrides(op, get_mem_deps_from_rw(rw))
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, work_distribution_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, work_distribution_pass)


def _cost_model_divide_op(op: ComputedBuffer, max_cores: int) -> bool:
    """Re-price one matmul's split with the analytic cost model.

    Runs between span_reduction and work_distribution, so
    iteration_space_ownership still holds only span-reduction's commits. Computes the split
    work_distribution would pick, hands it to the cost model, and commits the
    cost model's choice when it differs — returning True so the caller excludes
    the op from work_distribution (every op is divided by exactly one pass).
    """
    if not isinstance(op.data, Reduction):
        return False
    if op.data.reduction_type != BATCH_MATMUL_OP:
        return False
    if not config.ignore_work_division_hints and _has_work_div_hint(op):
        # User hints take ownership of the split decision; do not override them.
        return False

    rw = op_read_writes(op)
    args = get_mem_deps_from_rw(rw)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space = iteration_space_from_op(op)

    symbol_meta = _collect_symbol_metadata(it_space)

    # Phase 1 covers Pointwise (and incidentally non-matmul Reduction) only;
    # symbolic batchmatmul needs symmetric changes inside the cost model
    # (_matmul_split_cost concretises M, N, K) and is tracked as a follow-up.
    # Raise loudly so users do not silently get a plan based on
    # the warmup optimization_hint.
    if symbol_meta:
        raise Unsupported(
            f"symbolic dim(s) {sorted(map(str, symbol_meta))} on batchmatmul "
            f"op {op.get_name()} are not supported yet; symbolic work "
            f"division currently covers pointwise (and non-matmul reduction) "
            f"ops only."
        )

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )

    ownership = getattr(op, "iteration_space_ownership", None)
    committed_splits = (
        {s: v for s, v in ownership.work_slices.items() if v > 1}
        if ownership is not None
        else {}
    )

    coord_vars = {
        v
        for e in output_td.device_coords[:-1]
        for v in e.free_symbols
        if isinstance(v, Symbol)
    }
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]
    constraint_result = collect_work_division_constraints(
        WorkDivConstraintContext(
            op=op,
            it_space=it_space,
            it_space_adjusted=it_space_adjusted,
            output_td=output_td,
            input_tds=input_tds,
            stick_vars=stick_vars,
            reduction_vars=reduction_vars,
            committed_splits=committed_splits,
        )
    )
    blocked = constraint_result.blocked
    allowed_splits = constraint_result.allowed_splits
    default_splits, _, _ = _default_split(
        op,
        it_space_adjusted,
        output_td,
        committed_splits,
        max_cores,
        symbol_meta,
        blocked,
        allowed_splits,
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
        blocked,
        allowed_splits,
    )
    if splits == default_splits:
        return False

    apply_splits(op, splits)
    raise_if_per_core_overflow(all_tds, it_space, splits, op.get_name(), symbol_meta)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"cost_model_matmul_division work_division {op.get_name()}: "
            f"cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"min_splits={committed_splits}"
        )
    return True


def cost_model_matmul_division(graph: GraphLowering) -> list[Operation]:
    """Pass 2: re-price matmul/bmm splits with the analytic hardware cost model.

    Runs after span_reduction and before work_distribution. Returns the ops it
    re-split so passes.py can exclude them from work_distribution — every op is
    divided by exactly one pass.
    """
    operations = graph.operations
    max_cores = _validate_max_cores()
    cost_model_ops: list[Operation] = []
    for op in _iter_computed_buffers(operations):
        if _cost_model_divide_op(op, max_cores):
            cost_model_ops.append(op)
    return cost_model_ops
