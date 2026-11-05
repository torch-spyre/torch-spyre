# Copyright 2026 The Torch-Spyre Authors.
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

"""Map a logical work division onto physical cores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from typing import Any, Callable

import sympy
from sympy import Expr, Integer, Mod, Symbol, floor, sympify
from sympy.utilities.misc import as_int

from torch_spyre._C import DataFormats, get_device_size_in_bytes
from .op_spec import TensorArg, TensorWorkDivision
from .slice_ownership import factor_slice_axis, project_slice_axis


# TensorWorkDivision imports the comparator only when its method is called.
_MAX_OWNER_PERMUTATION_DIMS = 5


# Conservative physical incoming-fragment bound; not a measured cost limit.
_MAX_SHUFFLE_FANIN = 8


_MAX_EXACT_OWNERSHIP_POINTS = 1024


_MAX_EXACT_DIRECT_AXIS_POINTS = 1 << 16
_LOOP_POINT = Symbol("direct_axis_loop", integer=True, nonnegative=True)
# Evaluating symbolic coordinates can fail in many ways; each is a rejected
# proof, never a compiler crash.
_EVALUATION_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    NameError,
    NotImplementedError,
    OverflowError,
    SyntaxError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)


# Room for 128 distinct formulas on a 32-core device; eviction only repeats work.
@lru_cache(maxsize=4096)
def _owner_at_core(expression: Expr, core: int) -> Expr:
    """Reuse pure substitution, not a validity or ownership decision.

    One core at a time preserves callers' short-circuit and error ordering.
    No buffer, layout, split count or graph state participates in this result.
    """
    return expression.subs(Symbol("core_id"), core)


def owner_slots(
    slots: Mapping[Any, Expr], splits: Mapping[Any, int], num_cores: int
) -> tuple[dict[Any, int], ...]:
    """Evaluate owner formulas on every core: one slot per split dimension.

    A slot that is not a concrete integer inside its split raises ``ValueError``.
    """

    if num_cores <= 0:
        raise ValueError(f"physical core count must be positive, got {num_cores}")
    try:
        formulas = {dim: sympify(slots[dim]) for dim in splits}
    except KeyError:
        raise ValueError(
            "ownership split and owner-slot dimensions differ: "
            f"{sorted(map(str, splits))} != {sorted(map(str, slots))}"
        )
    rows = []
    for core in range(num_cores):
        row = {}
        for dim, split in splits.items():
            value = _owner_at_core(formulas[dim], core)
            try:
                int_value = as_int(value, strict=True)
            except ValueError:
                raise ValueError(f"non-integral owner slot {value} on core {core}")
            if not 0 <= int_value < split:
                raise ValueError(
                    f"owner slot {int_value} outside split {split} on core {core}"
                )
            row[dim] = int_value
        rows.append(row)
    return tuple(rows)


def _overlap(a: int, an: int, b: int, bn: int) -> bool:
    return a * bn < (b + 1) * an and b * an < (a + 1) * bn


def transfer_edges(
    source_splits: Mapping[Any, int],
    destination_splits: Mapping[Any, int],
    source_map: Mapping[int, Mapping[Any, int]],
    destination_map: Mapping[int, Mapping[Any, int]],
) -> set[tuple[int, int]]:
    """Ownership intersections for ordinary and completed-result copies.

    Both partitions must describe the same coordinate domain.
    """
    all_splits = source_splits.keys() | destination_splits.keys()
    result = set()
    for s_core, s_slice in source_map.items():
        for d_core, d_slice in destination_map.items():
            for dim in all_splits:
                if not _overlap(
                    s_slice.get(dim, 0),
                    source_splits.get(dim, 1),
                    d_slice.get(dim, 0),
                    destination_splits.get(dim, 1),
                ):
                    break
            else:
                result.add((s_core, d_core))
    return result


def same_owner_maps(
    left_splits: Mapping[Any, int],
    left_slots: Mapping[Any, Expr],
    left_cores: int,
    right_splits: Mapping[Any, int],
    right_slots: Mapping[Any, Expr],
    right_cores: int,
) -> bool:
    """Whether two owner maps give every physical core the same slice.

    Unsplit dimensions describe no ownership and are ignored. Equivalent SymPy
    spellings compare equal; a missing owner formula is a mismatch.
    """
    if left_cores <= 0:
        return False
    if left_cores != right_cores:
        return False
    left = {dim: split for dim, split in left_splits.items() if split > 1}
    right = {dim: split for dim, split in right_splits.items() if split > 1}
    if not left and not right:
        return True
    if left != right:
        return False
    try:
        return core_mappings_equal(
            {dim: left_slots[dim] for dim in left},
            {dim: right_slots[dim] for dim in right},
            left_cores,
        )
    except KeyError:
        return False


@lru_cache(maxsize=256)
def _loop_regions(
    extent: int,
    coordinates: tuple[Expr, ...],
    device_extents: tuple[int, ...],
    split: int,
    rectangles: bool = False,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Bounds on each original axis for each contiguous loop partition.

    The ordinary index parser supplies quotient/remainder factors. Bounds use
    integer arithmetic at partition endpoints, including modulo wrap. Creating
    a view additionally requires an injective, hole-free rectangle; reading a
    view only requires containment, so diagonal and gapped reads remain legal.
    """
    from sympy import simplify
    from torch.utils._sympy.functions import FloorDiv

    from .errors import Unsupported
    from .views import Term

    terms = []
    try:
        for coordinate, size in zip(coordinates, device_extents):
            expr = coordinate.replace(floor, lambda x: x)
            offset = expr.subs(_LOOP_POINT, 0)
            term = Term.from_coordinate(
                expr - offset, _LOOP_POINT, Integer(extent), size
            )
            term.offset = offset
            num, den, mod, offset = map(
                int, (term.num, term.den, term.mod, term.offset)
            )
            if (
                (num, den, mod, offset) != (term.num, term.den, term.mod, term.offset)
                or num == 0
                or den <= 0
                or mod <= 0
            ):
                raise ValueError("not an integer term")
            # The aligner's historical floor/offset assumptions are not an
            # ownership proof. Check the unmodified expression before reuse.
            rebuilt = num * floor(Mod(_LOOP_POINT, mod) / den) + offset
            original = coordinate.replace(FloorDiv, lambda a, b: floor(a / b))
            delta = (original - rebuilt).xreplace(
                {Mod(_LOOP_POINT, extent): _LOOP_POINT}
            )
            if simplify(delta) != 0:
                raise ValueError("normalization changes this coordinate")
            terms.append((num, den, mod, offset))
    except (Unsupported, AssertionError, TypeError, ValueError):
        terms = []

    # A chain of quotient/remainder digits determines the original loop value.
    # Otherwise the bounded exact fallback checks uniqueness, not just bounds.
    known_modulus = 1
    for _, den, mod, _ in sorted(terms, key=lambda t: t[1]):
        if known_modulus % den == 0 and mod % known_modulus == 0:
            known_modulus = mod
    analytic = bool(terms) and (not rectangles or known_modulus >= extent)
    result = []
    width = extent // split
    for slot in range(split):
        first, last = slot * width, (slot + 1) * width - 1
        if analytic:
            bounds = []
            for num, den, mod, offset in terms:
                low, high = first % mod // den, last % mod // den
                if first // mod != last // mod:
                    low, high = 0, (mod - 1) // den
                bounds.append(tuple(sorted((num * low + offset, num * high + offset))))
        else:
            points = [
                tuple(sympify(c).subs(_LOOP_POINT, p) for c in coordinates)
                for p in range(first, last + 1)
            ]
            if any(value.is_integer is not True for row in points for value in row):
                raise ValueError("coordinates must be integral")
            bounds = list(zip(map(min, zip(*points)), map(max, zip(*points))))
            if rectangles and len(set(points)) != width:
                raise ValueError("fused partition repeats an element")
        if any(
            low < 0 or high >= size for (low, high), size in zip(bounds, device_extents)
        ):
            raise ValueError("loop partition is off-axis")
        if rectangles and math.prod(high - low + 1 for low, high in bounds) != width:
            raise ValueError("fused partitions are not one rectangle shape")
        result.append(tuple((int(low), int(high)) for low, high in bounds))
    return tuple(result)


def direct_axis_ownership_failure(
    extent: int, split: int, coordinate: Expr, device_extent: int, physical_split: int
) -> str | None:
    """Check the stride proposal: loop partition p must stay in physical slice p."""
    if extent > _MAX_EXACT_DIRECT_AXIS_POINTS:
        return f"proof limit: direct axis needs {extent} points; limit is {_MAX_EXACT_DIRECT_AXIS_POINTS}"
    if split <= 0 or extent % split or device_extent <= 0 or device_extent % split:
        return "unsupported ownership input: incompatible extents, splits or cores"
    if split != physical_split:
        return "ownership mismatch: logical and physical split counts differ"
    try:
        regions = _loop_regions(extent, (coordinate,), (device_extent,), split)
        width = device_extent // split
        signatures = []
        for ((low, high),) in regions:
            if low // width != high // width:
                return "ownership mismatch: one loop partition crosses physical slices"
            signatures.append(low // width)
        if len(set(signatures)) != split:
            return "ownership mismatch: loop partitions do not cover distinct physical slices"
        if signatures != list(range(split)):
            return "ownership mismatch: loop and physical slices have different core owners"
        return None
    except _EVALUATION_ERRORS as exc:
        return f"unsupported ownership evaluation: {type(exc).__name__}: {exc}"


def decompose_fused_split_view(
    fused_symbol: Symbol,
    fused_split: int,
    fused_slot_expr: Expr,
    tensor_ownership: TensorWorkDivision,
    loop_extents: Mapping[Symbol, int],
    device_size: Sequence[int],
    device_coordinates: Sequence[Expr],
    num_cores: int,
    *,
    rejection_reasons: list[str] | None = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, Expr], ...]] | None:
    """Express one contiguous fused-loop partition on physical device axes.

    Each loop partition must fill one rectangle of the same shape. The existing
    mapping generator must reproduce those rectangles' origins in partition
    order; substituting the committed loop owner then preserves physical cores.
    """

    def reject(reason: str) -> None:
        if rejection_reasons is not None:
            rejection_reasons.append(reason)

    try:
        fused_split = int(fused_split)
        num_cores = int(num_cores)
        if fused_split <= 1 or num_cores <= 0 or num_cores % fused_split:
            reject("unsupported ownership input: fused split must divide the cores")
            return None
        if len(device_size) != len(device_coordinates):
            reject("unsupported ownership input: sizes and coordinates differ in rank")
            return None
        extent_expr = sympify(loop_extents[fused_symbol])
        if extent_expr.free_symbols or extent_expr.is_integer is not True:
            reject(
                f"unsupported ownership input: fused extent {extent_expr} not concrete"
            )
            return None
        extent = int(extent_expr)
        if extent <= 0 or extent % fused_split:
            reject(f"unsupported ownership input: fused extent {extent} not divisible")
            return None
        if extent + fused_split > _MAX_EXACT_OWNERSHIP_POINTS:
            reject(
                f"proof limit: fused decomposition needs {extent + fused_split} states; limit is {_MAX_EXACT_OWNERSHIP_POINTS}"
            )
            return None
        core_id = Symbol("core_id")
        fused_slot_expr = sympify(fused_slot_expr)
        if fused_slot_expr.free_symbols - {core_id}:
            reject(
                "unsupported ownership expression: fused owner has unresolved symbols"
            )
            return None

        # Judge broadcast multiplicity using the complete tuple of tensor-owned
        # axes. A V page, for example, is owned by (batch x KV-head, D): either
        # axis alone looks repeated in non-contiguous groups, while the pair is
        # one canonical 16-owner partition broadcast to the two query groups.
        # Operation axes absent from this tensor are deliberately excluded, so
        # truly interleaved broadcast groups still require a richer ownership
        # model and remain fail-closed.
        if (
            int(tensor_ownership.work_slices.get(fused_symbol, 1)) != fused_split
            or tensor_ownership.physical_core_count != num_cores
            or not core_mappings_equal(
                {fused_symbol: tensor_ownership.core_id_to_work_slice[fused_symbol]},
                {fused_symbol: fused_slot_expr},
                num_cores,
            )
            or select_unique_partition_division(
                tuple(tensor_ownership.work_slices),
                tensor_ownership.work_slices,
                num_cores,
                tensor_ownership.same_ownership,
                rejection_reasons=rejection_reasons,
            )
            is None
        ):
            reject(
                "no certified decomposition: tensor ownership has no canonical order"
            )
            return None

        driven = tuple(
            axis
            for axis, coordinate in enumerate(device_coordinates)
            if fused_symbol in sympify(coordinate).free_symbols
        )
        if not 2 <= len(driven) <= 5:
            reject(
                f"proof limit: fused loop drives {len(driven)} axes; 2 to 5 supported"
            )
            return None
        if any(
            sympify(device_coordinates[axis]).free_symbols != {fused_symbol}
            for axis in driven
        ):
            reject(
                "unsupported ownership expression: fused axis depends on other loops"
            )
            return None
        device_extents = {}
        for axis in driven:
            device_extent = sympify(device_size[axis])
            if device_extent.free_symbols or device_extent.is_integer is not True:
                reject(
                    f"unsupported ownership input: axis {axis} extent is not concrete"
                )
                return None
            device_extents[axis] = int(device_extent)
        regions = _loop_regions(
            extent,
            tuple(
                sympify(device_coordinates[axis]).xreplace({fused_symbol: _LOOP_POINT})
                for axis in driven
            ),
            tuple(device_extents[axis] for axis in driven),
            fused_split,
            rectangles=True,
        )
        shapes = {tuple(high - low + 1 for low, high in bounds) for bounds in regions}
        if len(shapes) != 1:
            reject(
                "unsupported physical ownership: fused partitions are not one rectangle shape"
            )
            return None
        widths = next(iter(shapes))
        if any(device_extents[axis] % width for axis, width in zip(driven, widths)):
            reject(
                "unsupported physical ownership: rectangles do not divide device extents"
            )
            return None
        factors = {
            axis: device_extents[axis] // width for axis, width in zip(driven, widths)
        }
        split_dims = tuple(
            (axis, factor) for axis, factor in factors.items() if factor > 1
        )
        if not split_dims:
            reject("no certified decomposition: no physical dimension is split")
            return None

        synthetic = {axis: Symbol(f"physical_dim_{axis}") for axis in driven}
        origins = [tuple(low for low, _ in bounds) for bounds in regions]
        expected = tuple(
            {
                synthetic[axis]: lo // width
                for axis, lo, width in zip(driven, row, widths)
            }
            for row in origins
        )
        if any(lo % width for row in origins for lo, width in zip(row, widths)):
            reject("ownership mismatch: one loop partition crosses physical slices")
            return None
        for order in permutations(driven):
            mapping = core_to_slice_mapping(
                tuple(synthetic[axis] for axis in order),
                tuple(factors[axis] for axis in order),
                fused_split,
            )
            if (
                owner_slots(
                    mapping,
                    {synthetic[axis]: factor for axis, factor in factors.items()},
                    fused_split,
                )
                == expected
            ):
                return tuple(sorted(split_dims)), tuple(
                    sorted(
                        (axis, mapping[synthetic[axis]].subs(core_id, fused_slot_expr))
                        for axis, _ in split_dims
                    )
                )
        reject("no certified decomposition: no canonical physical order matched")
        return None
    except _EVALUATION_ERRORS as exc:
        reject(f"unsupported ownership evaluation: {type(exc).__name__}: {exc}")
        return None


def select_unique_partition_division(
    dimensions: Sequence[Symbol],
    work_slices: Mapping[Symbol, int],
    num_cores: int,
    matches: Callable[[TensorWorkDivision], bool],
    *,
    rejection_reasons: list[str] | None = None,
) -> TensorWorkDivision | None:
    """Return the sole standard dimension order accepted by ``matches``.

    This bounded search only tries mappings produced by the existing canonical
    partition generator. The caller supplies the exact ownership proof; two
    distinct accepted owner maps are ambiguity and fail closed.
    Optional reasons belong to this call only and never steer the search.
    """

    def reject(reason: str) -> None:
        if rejection_reasons is not None:
            rejection_reasons.append(reason)

    split_by_dim = {
        dim: int(work_slices[dim])
        for dim in dimensions
        if int(work_slices.get(dim, 1)) > 1
    }
    if set(split_by_dim) != {
        dim for dim, split in work_slices.items() if int(split) > 1
    }:
        reject("unsupported ownership input: candidate dimension keys differ")
        return None
    split_dims = tuple(split_by_dim)
    if not split_dims:
        reject("no canonical candidate: there are no split dimensions")
        return None
    if len(split_dims) > _MAX_OWNER_PERMUTATION_DIMS:
        reject(
            f"proof limit: canonical search has {len(split_dims)} split dimensions; "
            f"limit is {_MAX_OWNER_PERMUTATION_DIMS}"
        )
        return None

    # Mapping order and field order are separate. Keep the caller's field order
    # stable while trying the bounded set of canonical owner formulas.
    candidate_splits = {dim: int(split) for dim, split in work_slices.items()}
    accepted: list[TensorWorkDivision] = []
    for order in permutations(split_dims):
        try:
            mapping = derive_partition_mapping(
                order,
                tuple(split_by_dim[dim] for dim in order),
                num_cores,
            )
            candidate = TensorWorkDivision(
                candidate_splits,
                {dim: mapping.get(dim, Integer(0)) for dim in candidate_splits},
                num_cores=num_cores,
            )
        except ValueError as exc:
            reject(f"unsupported ownership candidate: {exc}")
            continue
        if matches(candidate) and not any(
            previous.same_ownership(candidate) for previous in accepted
        ):
            accepted.append(candidate)
            if len(accepted) > 1:
                reject("ambiguous ownership: multiple canonical maps matched")
                return None
    if accepted:
        return accepted[0]
    reject("no canonical candidate matched")
    return None


def core_to_slice_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
    *,
    contiguous_dim: int | None = None,
) -> dict[Symbol, Expr]:
    """Return the logical work slice assigned to each physical core.

    By default dimensions vary in iteration-space order. ``contiguous_dim``
    moves one caller-selected dimension first so its participants are adjacent.
    """

    dims = tuple(dims)
    splits = tuple(dim_splits)
    if len(dims) != len(splits):
        raise ValueError(f"dimension/split count differs: {len(dims)} != {len(splits)}")

    logical_cores = math.prod(splits)
    if num_cores < logical_cores or num_cores % logical_cores != 0:
        raise ValueError(
            "num_cores must be a multiple of the logical work split "
            f"({logical_cores}), got {num_cores}"
        )

    dim_order = list(range(len(dims)))
    if contiguous_dim is not None and splits[contiguous_dim] > 1:
        dim_order.remove(contiguous_dim)
        dim_order.insert(0, contiguous_dim)

    core_id: Expr = Symbol("core_id")
    stride = Integer(1)
    result: dict[Symbol, Expr] = {}
    for dim in dim_order:
        split = Integer(splits[dim])
        if split == 1:
            coordinate = Integer(0)
        elif stride == 1:
            coordinate = Mod(core_id, split)
        else:
            coordinate = Mod(floor(core_id / stride), split)
        result[dims[dim]] = coordinate
        stride *= split
    return result


def matmul_reduction_dim(
    iteration_space: Iterable[Symbol], args: Iterable[TensorArg]
) -> Symbol | None:
    """Return a matmul's reduction loop K: both inputs index it, the output does not.

    Find K by identity, not by position: alignment adds range-1 loops for
    elided device dimensions after K, so the last loop is not always K. The
    test on the inputs excludes an M or N of 1 that the output folds away.
    Returns None if no loop or more than one loop agrees with the rule.
    """
    inputs: list[set[Symbol]] = []
    outputs: set[Symbol] = set()
    for arg in args:
        indexed = {
            symbol
            for coordinate in arg.device_coordinates
            for symbol in sympify(coordinate).free_symbols
        }
        if arg.is_input:
            inputs.append(indexed)
        else:
            outputs |= indexed
    reduced = [
        dim
        for dim in iteration_space
        if dim not in outputs and all(dim in indexed for indexed in inputs)
    ]
    return reduced[0] if len(reduced) == 1 else None


def derive_core_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
    *,
    contiguous_dim: Symbol | None = None,
    grouped_splits: Mapping[Symbol, int] | None = None,
) -> dict[Symbol, Expr]:
    """Derive one complete mapping from final dimensions and group geometry.

    ``grouped_splits`` describes logical owners that must occupy contiguous,
    equal-size core groups. Its device-dimension order defines the group
    topology; final loop order does not. Dimensions outside that mapping divide
    work within each group. No planning-time core assignment is consumed.
    """

    dims = tuple(dims)
    splits = tuple(int(split) for split in dim_splits)
    if len(dims) != len(splits):
        raise ValueError(f"dimension/split count differs: {len(dims)} != {len(splits)}")
    split_by_dim = dict(zip(dims, splits))
    if math.prod(splits) != num_cores:
        raise ValueError(
            f"operation split product must equal num_cores: {math.prod(splits)} != {num_cores}"
        )

    grouped_splits = dict(grouped_splits or {})
    unknown_dims = grouped_splits.keys() - split_by_dim.keys()
    if unknown_dims:
        raise ValueError(
            "grouped dimensions are not in the operation: "
            f"{sorted(map(str, unknown_dims))}"
        )
    for dim, split in grouped_splits.items():
        if int(split) != split_by_dim[dim]:
            raise ValueError(
                f"grouped split {dim}={split} does not match operation split "
                f"{split_by_dim[dim]}"
            )

    if not grouped_splits:
        contiguous_index = (
            dims.index(contiguous_dim) if contiguous_dim in dims else None
        )
        return core_to_slice_mapping(
            dims,
            splits,
            num_cores,
            contiguous_dim=contiguous_index,
        )

    grouped_dims = tuple(grouped_splits)
    local_dims = tuple(dim for dim in dims if dim not in grouped_splits)
    owner_count = math.prod(grouped_splits[dim] for dim in grouped_dims)
    if owner_count <= 0 or num_cores % owner_count:
        raise ValueError("grouped ownership does not divide the operation")
    group_size = num_cores // owner_count
    if math.prod(split_by_dim[dim] for dim in local_dims) != group_size:
        raise ValueError("operation splits do not fill each owner group")

    core_id = Symbol("core_id")
    group_id = floor(core_id / group_size)
    local_core_id = Mod(core_id, group_size)
    owner_mapping = core_to_slice_mapping(
        grouped_dims,
        tuple(grouped_splits[dim] for dim in grouped_dims),
        owner_count,
    )
    local_contiguous = (
        local_dims.index(contiguous_dim) if contiguous_dim in local_dims else None
    )
    local_mapping = core_to_slice_mapping(
        local_dims,
        tuple(split_by_dim[dim] for dim in local_dims),
        group_size,
        contiguous_dim=local_contiguous,
    )
    return {
        **{
            dim: expression.subs(core_id, group_id)
            for dim, expression in owner_mapping.items()
        },
        **{
            dim: expression.subs(core_id, local_core_id)
            for dim, expression in local_mapping.items()
        },
    }


def derive_partition_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
) -> dict[Symbol, Expr]:
    """Derive tensor owners from final partition geometry.

    A partition may have fewer logical owners than physical cores. In that
    case each owner occupies one contiguous, equal-size core group.
    """

    dims = tuple(dims)
    splits = tuple(int(split) for split in dim_splits)
    owner_count = math.prod(splits)
    if owner_count <= 0 or num_cores <= 0 or num_cores % owner_count:
        raise ValueError(
            f"partition owner count must divide num_cores: {owner_count}, {num_cores}"
        )
    group_size = num_cores // owner_count
    core_id = Symbol("core_id")
    group_id = floor(core_id / group_size)
    mapping = core_to_slice_mapping(dims, splits, owner_count)
    return {
        dim: expression.subs(core_id, group_id) for dim, expression in mapping.items()
    }


def distribute_aligned_split(split: int, bases: Sequence[int]) -> tuple[list[int], int]:
    """Factor one loop dimension's ``split`` over its aligned segments.

    ``align_tensors`` cuts a loop dimension into segments, listed innermost
    first with their extents in ``bases``. The outermost segment takes
    ``gcd(split, basis)`` first and passes the rest inward. Returns the factor
    each segment receives, innermost first, and what is left undistributed.
    """
    factors = []
    remaining = int(split)
    for basis in reversed(bases):
        factor = math.gcd(remaining, int(basis))
        factors.append(factor)
        remaining //= factor
    factors.reverse()
    return factors, remaining


def aligned_split_keeps_blocks(split: int, bases: Sequence[int]) -> bool:
    """Whether ``split`` still gives each core one contiguous block once aligned.

    The scratchpad planner, and a consumer on the same division, take an
    ``n``-way split of a dimension to mean ``n`` contiguous blocks. The
    distribution in :func:`distribute_aligned_split` agrees only while every
    segment outside the innermost split one is split whole. Otherwise the
    split lands on an inner segment and interleaves the owners: a 2-way split
    of ``d0 < 6`` cut into segments of extent 2 and 3 (a ``repeat`` read of
    ``Mod(d0, 2)``) gives core 0 rows {0, 2, 4}, not {0, 1, 2}.
    """
    factors, remaining = distribute_aligned_split(split, bases)
    if remaining != 1:
        return False
    split_inside = False
    for factor, basis in zip(reversed(factors), reversed(bases)):
        if split_inside and factor > 1:
            return False
        split_inside = split_inside or factor < int(basis)
    return True


def remap_work_division(
    division: TensorWorkDivision,
    dimension_remap: Mapping[Symbol, Sequence[tuple[Symbol, int]]],
) -> TensorWorkDivision:
    """Express tensor ownership in an aligned iteration space.

    ``align_tensors`` may split one loop dimension into several dimensions. The
    physical partition does not change; only the symbols used to describe it
    do. That holds only for a split that :func:`aligned_split_keeps_blocks`
    admits, which the work-division constraints enforce upstream.
    """

    num_cores = division.physical_core_count
    new_splits: dict[Symbol, int] = {}
    new_core_map: dict[Symbol, Expr] = {}
    for old_dim, split in division.work_slices.items():
        new_dims = dimension_remap.get(old_dim)
        if new_dims is None:
            raise ValueError(f"tensor ownership dimension {old_dim} has no alignment")
        remaining_split = int(split)
        split_factors: list[tuple[Symbol, int]] = []
        if len(new_dims) == 1:
            split_factors = [(new_dims[0][0], remaining_split)]
            remaining_split = 1
        else:
            factors, remaining_split = distribute_aligned_split(
                remaining_split, [basis for _, basis in new_dims]
            )
            split_factors = [
                (new_dim, factor) for (new_dim, _), factor in zip(new_dims, factors)
            ]
        if remaining_split != 1:
            raise ValueError(f"cannot normalize {split}-way split on {old_dim}")

        slot = division.core_id_to_work_slice[old_dim]
        slot_stride = 1
        for new_dim, factor in split_factors:
            if factor == 1:
                continue
            new_slot = Mod(floor(slot / slot_stride), factor)
            previous_split = new_splits.get(new_dim)
            previous_slot = new_core_map.get(new_dim)
            if previous_split is not None and (
                previous_split != factor
                or previous_slot is None
                or not core_mappings_equal(
                    {new_dim: previous_slot},
                    {new_dim: new_slot},
                    num_cores,
                )
            ):
                raise ValueError(f"conflicting normalized ownership on {new_dim}")
            new_splits[new_dim] = factor
            new_core_map[new_dim] = new_slot
            slot_stride *= factor
    return TensorWorkDivision(
        new_splits,
        new_core_map,
        num_cores=num_cores,
    )


def finalize_tensor_work_divisions(
    iteration_space: Mapping[Symbol, tuple[Expr, int]],
    divisions: Sequence[TensorWorkDivision | None],
) -> tuple[TensorWorkDivision | None, ...]:
    """Verify committed tensor owners in the final aligned iteration space."""

    result: list[TensorWorkDivision | None] = []
    for division in divisions:
        if division is None:
            result.append(None)
            continue
        work_slices = {
            dim: int(split)
            for dim, split in division.work_slices.items()
            if int(split) > 1
        }
        unknown_dims = work_slices.keys() - iteration_space.keys()
        if unknown_dims:
            raise ValueError(
                "tensor ownership dimensions are not aligned: "
                f"{sorted(map(str, unknown_dims))}"
            )

        try:
            core_map = {dim: division.core_id_to_work_slice[dim] for dim in work_slices}
        except KeyError as exc:
            raise ValueError(
                f"tensor ownership has no owner for {exc.args[0]}"
            ) from exc
        verified = TensorWorkDivision(
            work_slices,
            core_map,
            num_cores=division.physical_core_count,
        )
        verified.to_core_slices(verified.physical_core_count)
        result.append(verified)
    return tuple(result)


def derive_operation_mapping(
    iteration_space: Mapping[Symbol, tuple[Expr, int]],
    tensor_divisions: Sequence[TensorWorkDivision | None] = (),
    *,
    contiguous_dim: Symbol | None = None,
) -> dict[Symbol, Expr]:
    """Derive one operation mapping that satisfies every LX tensor owner."""

    dims = tuple(iteration_space)
    splits = tuple(int(iteration_space[dim][1]) for dim in dims)
    num_cores = math.prod(splits)
    split_by_dim = dict(zip(dims, splits))
    constrained: dict[Symbol, Expr] = {}
    for division in tensor_divisions:
        if division is None:
            continue
        if division.physical_core_count != num_cores:
            raise ValueError(
                "LX tensor ownership and operation use different core domains: "
                f"{division.physical_core_count} != {num_cores}"
            )
        for dim, split in division.work_slices.items():
            if int(split) <= 1:
                continue
            if dim not in split_by_dim:
                raise ValueError(f"LX tensor dimension {dim} is not in the operation")
            if split_by_dim[dim] != int(split):
                raise ValueError(
                    f"LX tensor split for {dim} does not match the operation: "
                    f"{split} != {split_by_dim[dim]}"
                )
            expression = division.core_id_to_work_slice[dim]
            previous = constrained.setdefault(dim, expression)
            if not core_mappings_equal({dim: previous}, {dim: expression}, num_cores):
                raise ValueError(f"LX tensors disagree on core ownership for {dim}")

    if not constrained:
        return derive_core_mapping(
            dims,
            splits,
            num_cores,
            contiguous_dim=contiguous_dim,
        )

    # Preserve main's operation map whenever it already satisfies the physical
    # tensor owners. The grouped search below is only needed when it does not.
    default = derive_core_mapping(
        dims,
        splits,
        num_cores,
        contiguous_dim=contiguous_dim,
    )
    if all(
        core_mappings_equal({dim: default[dim]}, {dim: expression}, num_cores)
        for dim, expression in constrained.items()
    ):
        return default

    # Tensor-owned dimensions occupy the outer, contiguous groups. At most five
    # dimensions can be split on 32 cores, so trying their radix orders is small.
    if len(constrained) > _MAX_OWNER_PERMUTATION_DIMS:
        raise ValueError(
            "too many aligned tensor-owned dimensions for bounded core-order "
            f"search: {len(constrained)} > {_MAX_OWNER_PERMUTATION_DIMS}"
        )
    for order in permutations(sorted(constrained, key=str)):
        candidate = derive_core_mapping(
            dims,
            splits,
            num_cores,
            contiguous_dim=contiguous_dim,
            grouped_splits={dim: split_by_dim[dim] for dim in order},
        )
        if all(
            core_mappings_equal({dim: candidate[dim]}, {dim: expression}, num_cores)
            for dim, expression in constrained.items()
        ):
            return candidate

    raise ValueError("no operation core mapping satisfies every LX tensor owner")


def partition_lx_size_bytes(
    device_size: Sequence[int],
    device_dtype: DataFormats,
    split_by_device_dim: Mapping[int, int],
) -> int:
    """Size one core's packed slice of a normalized standard device layout.

    LX stores the local extents, not the gaps between cores' slices in the
    whole tensor. Use this member's physical splits: replicated consumers need
    more storage than tensor size divided by the communication core count.
    Device extents include padding; the final axis must be a complete unsplit
    stick. Reject uneven slices, whose local storage is not described here.
    """

    if not device_size or any(extent <= 0 for extent in device_size):
        raise ValueError("device extents must be positive")
    elems_per_stick = device_dtype.elems_per_stick()
    if elems_per_stick <= 0:
        raise ValueError("elems_per_stick must be positive")
    for dim, split in split_by_device_dim.items():
        if dim < 0 or dim >= len(device_size) or split <= 0:
            raise ValueError(f"invalid split {split} on device dimension {dim}")
    if device_size[-1] != elems_per_stick:
        raise ValueError("LX size requires one complete final stick dimension")
    if split_by_device_dim.get(len(device_size) - 1, 1) != 1:
        raise ValueError("the final stick dimension cannot be split")

    per_core_size = []
    for dim, extent in enumerate(device_size):
        split = split_by_device_dim.get(dim, 1)
        if extent % split:
            raise ValueError(f"device extent {extent} is not divisible by {split}")
        per_core_size.append(extent // split)
    return get_device_size_in_bytes(per_core_size, device_dtype)


def _comparable(expr):
    return expr if not isinstance(expr, Expr) else str(expr)


def core_mappings_equal(
    left: Mapping[Any, Expr],
    right: Mapping[Any, Expr],
    num_cores: int,
) -> bool:
    """Return whether two symbolic mappings assign every core identically.

    Memoized on the two mappings: the planner compares the same handful of
    owner formulas thousands of times (view interning, the slicing-match gate
    and the movement gate all go through here, and structurally identical ops
    such as attention's unrolled KV blocks induce identical views). Profiled
    on a 304-op graph this was 5 million ``sympify`` calls.
    """

    if num_cores <= 0:
        return False
    if left.keys() != right.keys():
        return False

    try:
        key_left = tuple(
            sorted(
                ((_comparable(d), sympify(e)) for d, e in left.items()),
                key=lambda kv: kv[0],
            )
        )
        key_right = tuple(
            sorted(
                ((_comparable(d), sympify(e)) for d, e in right.items()),
                key=lambda kv: kv[0],
            )
        )
    except (TypeError, ValueError):
        return False
    return _core_mappings_equal_cached(key_left, key_right, num_cores)


@lru_cache(maxsize=65536)
def _core_mappings_equal_cached(
    left: tuple[tuple[Any, Expr], ...],
    right: tuple[tuple[Any, Expr], ...],
    num_cores: int,
) -> bool:
    try:
        for (_, lf), (_, rf) in zip(left, right):
            for core in range(num_cores):
                try:
                    l_val = as_int(_owner_at_core(lf, core))
                    r_val = as_int(_owner_at_core(rf, core))
                    if l_val != r_val:
                        return False
                except ValueError:
                    return False
        return True
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PerCoreView:
    """Geometric description of a buffer's per-core slicing.

    - work_slice_dims: (device-dim index, split factor) pairs, one per
      split dim.
    - core_to_slot: (device-dim index, slice-index expression in core_id)
      pairs giving each core's position along that split dim.
    - num_cores: physical cores over which ``core_to_slot`` is defined. This
      can exceed the number of logical slices when data is replicated.
      If omitted, the split product is used. Views that omit a reduction axis
      or describe broadcast copies must retain the explicit physical count.

    The first two fields are keyed by the buffer's device-dim index — not by
    op-local iter symbols — so the value depends only on the buffer's physical
    slicing.

    Example: a 2D buffer split 4-ways on dim 0 across 4 cores has
        work_slice_dims = ((0, 4),)
        core_to_slot    = ((0, Mod(core_id, 4)),)
    so core_id=2 owns slot 2 along dim 0.
    """

    work_slice_dims: tuple[tuple[int, int], ...]
    core_to_slot: tuple[tuple[int, Expr], ...]
    split_product: int
    num_cores: int | None = None

    def __init__(
        self,
        work_slice_dims: tuple[tuple[int, int], ...],
        core_to_slot: tuple[tuple[int, Expr], ...],
        num_cores: int | None = None,
        *,
        split_product: int = 0,  # dummy arg to absorb automatically copied args
    ):
        object.__setattr__(self, "work_slice_dims", work_slice_dims)
        object.__setattr__(self, "core_to_slot", core_to_slot)
        object.__setattr__(
            self, "split_product", math.prod(split for _, split in work_slice_dims)
        )
        object.__setattr__(self, "num_cores", num_cores)

    def same_partition(self, other: "PerCoreView") -> bool:
        """Whether both views assign every physical core the same buffer slice."""

        return same_owner_maps(
            dict(self.work_slice_dims),
            dict(self.core_to_slot),
            self.num_cores if self.num_cores is not None else self.split_product,
            dict(other.work_slice_dims),
            dict(other.core_to_slot),
            other.num_cores if other.num_cores is not None else other.split_product,
        )


def core_slices(view: PerCoreView, num_cores: int) -> dict[int, dict[int, int]]:
    if view.num_cores is not None and view.num_cores <= 0:
        raise ValueError(f"physical core count must be positive, got {view.num_cores}")
    if view.num_cores is not None and view.num_cores != num_cores:
        raise ValueError(
            "ownership core count differs from the communication domain: "
            f"{view.num_cores} != {num_cores}"
        )
    rows = owner_slots(dict(view.core_to_slot), dict(view.work_slice_dims), num_cores)
    return dict(enumerate(rows))


def work_division_from_view(
    view: PerCoreView | None,
    device_size: Sequence[int],
    device_coordinates: Sequence[sympy.Expr],
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
    *,
    allow_extended: bool,
) -> TensorWorkDivision | None:
    """Interpret physical slices through an access, without choosing new owners."""
    if view is None:
        return None
    n = view.num_cores
    if n is None or n <= 0:
        raise ValueError("LX ownership must carry its physical core domain")
    if len(device_size) != len(device_coordinates):
        raise ValueError("sizes and coordinates differ in rank")
    physical_splits, slots = dict(view.work_slice_dims), dict(view.core_to_slot)
    if len(physical_splits) != len(view.work_slice_dims) or len(slots) != len(
        view.core_to_slot
    ):
        raise ValueError("duplicate physical dimensions")
    rows = owner_slots(slots, physical_splits, n)
    axes_by_loop: dict[sympy.Symbol, list[int]] = {}
    # Only the local proof coordinates may be reduced. Callers and emitted
    # tensor accesses retain the original multi-loop coordinate expressions.
    proof_coordinates = list(device_coordinates)
    for axis, split in physical_splits.items():
        if (
            not 0 <= axis < len(device_size)
            or sympy.sympify(device_size[axis]).is_Integer is not True
            or device_size[axis] <= 0
            or device_size[axis] % split
        ):
            raise ValueError(
                f"unsupported ownership input: axis {axis} not divisible by {split}"
            )
        symbols = device_coordinates[axis].free_symbols
        if len(symbols) == 1 and symbols <= iteration_space.keys():
            loop = next(iter(symbols))
        else:
            if not allow_extended:
                raise ValueError(f"cannot map device dimension {axis} to one loop")
            projected = project_slice_axis(
                device_coordinates[axis], int(device_size[axis]), split, iteration_space
            )
            if projected is None:
                # Keep the physical stick axis intact. Expand only the local
                # proof frame, then reuse the exact loop-region proof below.
                factors = (
                    factor_slice_axis(
                        device_coordinates[axis],
                        int(device_size[axis]),
                        split,
                        slots[axis],
                        iteration_space,
                    )
                    if axis != len(device_size) - 1
                    else None
                )
                if factors is None:
                    raise ValueError(f"cannot map device dimension {axis} to one loop")
                shift = len(factors) - 1
                new_splits, new_slots = {}, {}
                for a, count in physical_splits.items():
                    if a != axis:
                        new_axis = a + shift if a > axis else a
                        new_splits[new_axis], new_slots[new_axis] = count, slots[a]
                for offset, (_, _, count, owner) in enumerate(factors):
                    if count > 1:
                        new_splits[axis + offset] = count
                        new_slots[axis + offset] = owner
                return work_division_from_view(
                    PerCoreView(
                        tuple(sorted(new_splits.items())),
                        tuple(sorted(new_slots.items())),
                        num_cores=n,
                    ),
                    [
                        *device_size[:axis],
                        *(f[0] for f in factors),
                        *device_size[axis + 1 :],
                    ],
                    [
                        *device_coordinates[:axis],
                        *(f[1] for f in factors),
                        *device_coordinates[axis + 1 :],
                    ],
                    iteration_space,
                    allow_extended=allow_extended,
                )
            loop, proof_coordinates[axis] = projected
        axes_by_loop.setdefault(loop, []).append(axis)

    any_fused = any(len(axes) > 1 for axes in axes_by_loop.values())
    splits, owners, expected = {}, {}, {}
    fused_states = 0
    for loop, axes in axes_by_loop.items():
        extent = iteration_space[loop]
        extent = sympy.sympify(extent[0] if isinstance(extent, tuple) else extent)
        if extent.is_Integer is not True or extent <= 0:
            raise ValueError(
                f"unsupported ownership input: loop extent {extent} is not concrete"
            )
        extent = int(extent)
        first = axes[0]
        same = all(
            physical_splits[a] == physical_splits[first]
            and core_mappings_equal({loop: slots[a]}, {loop: slots[first]}, n)
            for a in axes
        )
        split = (
            physical_splits[first]
            if same
            else math.prod(physical_splits[a] for a in axes)
        )
        splits[loop] = split
        if len(axes) == 1:
            stick = len(device_size) - 1
            if (
                first != stick
                and stick not in physical_splits
                and loop in device_coordinates[-1].free_symbols
            ):
                padded = int(device_size[first] * device_size[-1])
                if padded - device_size[-1] < extent <= padded:
                    extent = padded
            if extent > _MAX_EXACT_DIRECT_AXIS_POINTS:
                raise ValueError(
                    f"proof limit: direct axis needs {extent} points; limit is {_MAX_EXACT_DIRECT_AXIS_POINTS}"
                )
        else:
            fused_states += extent + split
            if fused_states > _MAX_EXACT_OWNERSHIP_POINTS:
                raise ValueError(
                    f"proof limit: fused axes need {fused_states} states; limit is {_MAX_EXACT_OWNERSHIP_POINTS}"
                )
        if extent % split:
            raise ValueError(
                f"unsupported ownership input: loop {loop} not divisible by {split}"
            )
        try:
            bounds = _loop_regions(
                extent,
                tuple(proof_coordinates[a].xreplace({loop: _LOOP_POINT}) for a in axes),
                tuple(int(device_size[a]) for a in axes),
                split,
            )
        except _EVALUATION_ERRORS as exc:
            raise ValueError(
                f"unsupported ownership evaluation: {type(exc).__name__}: {exc}"
            ) from exc
        widths = [int(device_size[a]) // physical_splits[a] for a in axes]
        signatures = [
            tuple(low // width for (low, _), width in zip(region, widths))
            for region in bounds
        ]
        if any(
            low // width != high // width
            for region in bounds
            for (low, high), width in zip(region, widths)
        ):
            raise ValueError(
                "ownership mismatch: one loop partition crosses physical slices"
            )
        if len(set(signatures)) != split:
            raise ValueError(
                "ownership mismatch: loop partitions do not cover distinct physical slices"
            )
        try:
            table = tuple(signatures.index(tuple(row[a] for a in axes)) for row in rows)
        except ValueError:
            raise ValueError(
                "ownership mismatch: a core owns slices no loop partition covers"
            ) from None
        if set(table) != set(range(split)) or (
            not any_fused and signatures != [(p,) for p in range(split)]
        ):
            raise ValueError(
                "ownership mismatch: loop and physical slices have different core owners"
            )
        expected[loop] = table
        owners[loop] = slots[first]

    if not any_fused:
        return TensorWorkDivision(splits, owners, num_cores=n)
    # The physical slices already determine every loop owner. Search only for
    # the existing supported spelling, never re-prove the access per candidate.
    expected_rows = tuple(
        {loop: table[core] for loop, table in expected.items()} for core in range(n)
    )
    candidate = select_unique_partition_division(
        tuple(loop for loop in iteration_space if loop in splits),
        splits,
        n,
        lambda division: (
            owner_slots(division.core_id_to_work_slice, splits, n) == expected_rows
        ),
    )
    if candidate is None:
        raise ValueError("no unique certified canonical mapping for fused ownership")
    return candidate


@dataclass(frozen=True)
class SubrangeTransferFragment:
    """One exact rectangular intersection, with byte offsets in local storage.

    ``parent_origin`` and ``extent`` use the parent's physical coordinates.
    Offsets are relative to the full source allocation and the compact child
    allocation. Strides are dense *local* strides, not parent HBM strides.
    This record describes geometry; it is not a native shuffle descriptor.
    """

    source_core: int
    destination_core: int
    parent_origin: tuple[int, ...]
    extent: tuple[int, ...]
    source_offset_bytes: int
    destination_offset_bytes: int
    source_strides_bytes: tuple[int, ...]
    destination_strides_bytes: tuple[int, ...]


@dataclass(frozen=True)
class DenseSubrangeTransfer:
    """Checked geometry returned by :func:`certify_dense_subrange_transfer`.

    All fields are immutable values. A consumer must still prove fresh storage,
    single-writer and reader eligibility, scheduling, native emission and cost.
    This certificate alone must never enable LX residency or shorten a lifetime.
    """

    parent_shape: tuple[int, ...]
    origin: tuple[int, ...]
    extent: tuple[int, ...]
    element_bytes: int
    source_splits: tuple[int, ...]
    destination_splits: tuple[int, ...]
    source_owners: tuple[tuple[int, ...], ...]
    destination_owners: tuple[tuple[int, ...], ...]
    source_local_shape: tuple[int, ...]
    destination_local_shape: tuple[int, ...]
    source_reservation_bytes: int
    destination_reservation_bytes: int
    fragments: tuple[SubrangeTransferFragment, ...]

    @property
    def destination_identity(self) -> tuple:
        """Destination sharing key within one parent buffer's identity.

        Equal child shapes and owner maps do not identify equal source bytes.
        The caller must also key by the actual parent buffer identity.
        """
        return (
            self.parent_shape,
            self.origin,
            self.extent,
            self.element_bytes,
            self.destination_splits,
            self.destination_owners,
        )


def _dense_subrange_partition(
    shape: tuple[int, ...],
    splits: Mapping[int, int],
    owners: Mapping[int, Mapping[int, int]],
    *,
    allow_replicas: bool,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate already projected physical-axis owner maps without inferring order."""
    rank = len(shape)
    if any(type(axis) is not int or not 0 <= axis < rank for axis in splits):
        raise ValueError("partition has an invalid physical axis")
    full_splits = tuple(splits.get(axis, 1) for axis in range(rank))
    if (
        any(
            type(split) is not int or split <= 0 or size % split
            for size, split in zip(shape, full_splits)
        )
        or full_splits[-1] != 1
    ):
        raise ValueError("partition must divide each axis and retain complete sticks")
    if not owners or any(type(core) is not int for core in owners):
        raise ValueError("partition requires concrete physical core IDs")
    if set(owners) != set(range(len(owners))):
        raise ValueError(
            "partition core IDs must describe its complete physical domain"
        )
    partitions = math.prod(full_splits)
    if len(owners) % partitions or (not allow_replicas and len(owners) != partitions):
        raise ValueError("partition owners do not cover the complete domain")
    rows = []
    counts: dict[tuple[int, ...], int] = {}
    for core in range(len(owners)):
        row = owners[core]
        if any(type(axis) is not int or not 0 <= axis < rank for axis in row):
            raise ValueError("owner has an invalid physical axis")
        slots = tuple(row.get(axis, 0) for axis in range(rank))
        if any(
            (split > 1 and axis not in row)
            or type(slot) is not int
            or not 0 <= slot < split
            for axis, (slot, split) in enumerate(zip(slots, full_splits))
        ):
            raise ValueError(
                "owner slot is missing, nonintegral or outside its partition"
            )
        rows.append(slots)
        counts[slots] = counts.get(slots, 0) + 1
    expected = set(product(*(range(split) for split in full_splits)))
    if set(counts) != expected:
        raise ValueError("partition owners do not cover the complete domain")
    if len(set(counts.values())) != 1 or (
        not allow_replicas and max(counts.values()) != 1
    ):
        raise ValueError(
            "source owners must be unique and destination replicas uniform"
        )
    return (
        full_splits,
        tuple(rows),
        tuple(size // split for size, split in zip(shape, full_splits)),
    )


def certify_dense_subrange_transfer(
    parent_shape: Sequence[int],
    origin: Sequence[int],
    extent: Sequence[int],
    source_splits: Mapping[int, int],
    destination_splits: Mapping[int, int],
    source_map: Mapping[int, Mapping[int, int]],
    destination_map: Mapping[int, Mapping[int, int]],
    *,
    element_bytes: int,
) -> DenseSubrangeTransfer:
    """Certify a static dense, complete-stick rectangular child of a parent.

    Coordinates and owner maps must already be projected onto physical axes by
    the existing view/ownership machinery. The source partitions the *entire*
    parent uniquely; destinations partition the compact child, optionally with
    uniform replicas. Intersections share the ordinary transfer-edge overlap
    authority. Inactive source owners remain part of the parent's reservation.

    This certifies geometry, not a native transfer descriptor. Local pointwise
    reads additionally require ``local_subrange_byte_offset`` and the allocator's
    storage guards. Packed elements, incomplete sticks, symbolic coordinates,
    incomplete ownership and more than eight sources per destination fail closed.
    It does not replace storage, reduction or mutation proofs.
    Callers must separately check physical device limits and compatibility of
    source/destination core domains; contiguous map keys alone do not prove them.
    """
    parent_shape, origin, extent = map(tuple, (parent_shape, origin, extent))
    rank = len(parent_shape)
    if (
        rank < 2
        or len(origin) != rank
        or len(extent) != rank
        or type(element_bytes) is not int
        or element_bytes not in (1, 2, 4)
        or any(type(value) is not int for value in (*parent_shape, *origin, *extent))
        or any(
            size <= 0 or start < 0 or width <= 0 or start + width > size
            for size, start, width in zip(parent_shape, origin, extent)
        )
    ):
        raise ValueError("subrange requires positive static extents inside the parent")
    if (
        parent_shape[-1] * element_bytes != 128
        or origin[-1] != 0
        or extent[-1] != parent_shape[-1]
    ):
        raise ValueError("subrange must contain complete aligned 128-byte sticks")
    src_splits, src_owners, src_shape = _dense_subrange_partition(
        parent_shape, source_splits, source_map, allow_replicas=False
    )
    dst_splits, dst_owners, dst_shape = _dense_subrange_partition(
        extent, destination_splits, destination_map, allow_replicas=True
    )
    src_strides = tuple(
        math.prod(src_shape[axis + 1 :]) * element_bytes for axis in range(rank)
    )
    dst_strides = tuple(
        math.prod(dst_shape[axis + 1 :]) * element_bytes for axis in range(rank)
    )
    source_span = math.prod(src_shape) * element_bytes
    destination_span = math.prod(dst_shape) * element_bytes
    fragments = []
    for destination_core, dst_slots in enumerate(dst_owners):
        dst_start = tuple(
            start + slot * size
            for start, slot, size in zip(origin, dst_slots, dst_shape)
        )
        incoming = []
        for source_core, src_slots in enumerate(src_owners):
            src_start = tuple(slot * size for slot, size in zip(src_slots, src_shape))
            bounds = [
                (a, a + an, b, b + bn)
                for a, an, b, bn in zip(src_start, src_shape, dst_start, dst_shape)
            ]
            if not all(_intervals_overlap(*axis) for axis in bounds):
                continue
            start = tuple(max(a, b) for a, _, b, _ in bounds)
            widths = tuple(
                min(a_end, b_end) - low
                for (_, a_end, _, b_end), low in zip(bounds, start)
            )
            src_offset = sum(
                (low - base) * stride
                for low, base, stride in zip(start, src_start, src_strides)
            )
            dst_offset = sum(
                (low - base) * stride
                for low, base, stride in zip(start, dst_start, dst_strides)
            )
            # Inclusive last-element addresses are checked against full local
            # reservations, not a clipped rectangle's smaller byte count.
            for offset, strides, span in (
                (src_offset, src_strides, source_span),
                (dst_offset, dst_strides, destination_span),
            ):
                end = (
                    offset
                    + sum(
                        (width - 1) * stride for width, stride in zip(widths, strides)
                    )
                    + element_bytes
                )
                if offset < 0 or offset % 128 or end > span:
                    raise ValueError(
                        "subrange fragment exceeds its reserved local span"
                    )
            incoming.append(
                SubrangeTransferFragment(
                    source_core,
                    destination_core,
                    start,
                    widths,
                    src_offset,
                    dst_offset,
                    src_strides,
                    dst_strides,
                )
            )
        if not incoming or len(incoming) > _MAX_SHUFFLE_FANIN:
            raise ValueError(
                "subrange destination exceeds the eight-source fragment limit"
            )
        # Complete unique source ownership proves disjointness; the exact volume
        # check independently guards against an omitted clipped intersection.
        if sum(math.prod(fragment.extent) for fragment in incoming) != math.prod(
            dst_shape
        ):
            raise ValueError(
                "subrange fragments do not completely cover the destination"
            )
        fragments.extend(incoming)
    return DenseSubrangeTransfer(
        parent_shape,
        origin,
        extent,
        element_bytes,
        src_splits,
        dst_splits,
        src_owners,
        dst_owners,
        src_shape,
        dst_shape,
        source_span,
        destination_span,
        tuple(fragments),
    )


def local_subrange_byte_offset(certificate: DenseSubrangeTransfer) -> int:
    """Require one dense, equally offset local read on every physical core.

    This is the ordinary pointwise descriptor's contract, not permission to
    emit a shuffle. The parent keeps its complete reservation.
    """
    count = len(certificate.source_owners)
    if (
        len(certificate.destination_owners) != count
        or len(certificate.fragments) != count
    ):
        raise ValueError("local subrange needs one fragment on every source core")
    offsets = set()
    for core, fragment in enumerate(certificate.fragments):
        if (
            fragment.source_core != core
            or fragment.destination_core != core
            or fragment.destination_offset_bytes != 0
            or fragment.extent != certificate.destination_local_shape
            or any(
                size > 1 and source != destination
                for size, source, destination in zip(
                    fragment.extent,
                    fragment.source_strides_bytes,
                    fragment.destination_strides_bytes,
                )
            )
        ):
            raise ValueError("subrange is not a compact same-core read")
        offsets.add(fragment.source_offset_bytes)
    if len(offsets) != 1:
        raise ValueError("local subrange needs one common byte offset")
    return offsets.pop()


def _intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def dense_read_rectangle(device_size, coordinates, iteration_space):
    """Prove an injective physical rectangle using the existing loop parser.

    Each coordinate may depend on one loop. Coupled coordinates, padding,
    broadcasts and symbolic bounds are deliberately outside this local path.
    """
    shape = tuple(device_size)
    if len(shape) != len(coordinates) or any(
        sympy.sympify(size).is_Integer is not True or size <= 0 for size in shape
    ):
        raise ValueError("local read needs a static physical shape")
    origin, extent, axes = [0] * len(shape), [1] * len(shape), {}
    for axis, coordinate in enumerate(coordinates):
        coordinate = sympy.sympify(coordinate)
        symbols = coordinate.free_symbols
        if not symbols:
            if coordinate.is_Integer is not True or not 0 <= coordinate < shape[axis]:
                raise ValueError("local read constant is outside its parent")
            origin[axis] = int(coordinate)
        elif len(symbols) == 1 and symbols <= iteration_space.keys():
            axes.setdefault(next(iter(symbols)), []).append(axis)
        else:
            raise ValueError("local read needs independent physical coordinates")
    for loop, size in iteration_space.items():
        size = sympy.sympify(size)
        if size.is_Integer is not True or not 0 < size <= _MAX_EXACT_DIRECT_AXIS_POINTS:
            raise ValueError("local read loop exceeds the static proof domain")
        if loop not in axes:
            if size != 1:
                raise ValueError("local read broadcasts an iteration axis")
            continue
        active = axes[loop]
        (bounds,) = _loop_regions(
            int(size),
            tuple(coordinates[a].xreplace({loop: _LOOP_POINT}) for a in active),
            tuple(int(shape[a]) for a in active),
            1,
            rectangles=True,
        )
        for axis, (low, high) in zip(active, bounds):
            origin[axis], extent[axis] = low, high - low + 1
    return tuple(origin), tuple(extent)
