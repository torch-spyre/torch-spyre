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
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import permutations
from typing import Any, Callable

from sympy import Expr, Integer, Mod, Symbol, floor, sympify

from .op_spec import TensorWorkDivision


# pass_utils imports this module; keep its PerCoreView type out of this layer.
# TensorWorkDivision imports the comparator only when its method is called.
_MAX_OWNER_PERMUTATION_DIMS = 5


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
    if splits.keys() != slots.keys():
        raise ValueError(
            "ownership split and owner-slot dimensions differ: "
            f"{sorted(map(str, splits))} != {sorted(map(str, slots))}"
        )
    rows = []
    for core in range(num_cores):
        row = {}
        for dim, split in splits.items():
            value = _owner_at_core(sympify(slots[dim]), core)
            if value.free_symbols or value.is_integer is not True:
                raise ValueError(f"non-integral owner slot {value} on core {core}")
            if not 0 <= int(value) < int(split):
                raise ValueError(
                    f"owner slot {int(value)} outside split {split} on core {core}"
                )
            row[dim] = int(value)
        rows.append(row)
    return tuple(rows)


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

    left = {dim: int(split) for dim, split in left_splits.items() if int(split) > 1}
    right = {dim: int(split) for dim, split in right_splits.items() if int(split) > 1}
    if left != right or left_cores != right_cores:
        return False
    if not left:
        return True
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


def remap_work_division(
    division: TensorWorkDivision,
    dimension_remap: Mapping[Symbol, Sequence[tuple[Symbol, int]]],
) -> TensorWorkDivision:
    """Express tensor ownership in an aligned iteration space.

    ``align_tensors`` may split one loop dimension into several dimensions. The
    physical partition does not change; only the symbols used to describe it do.
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
            for new_dim, basis in reversed(new_dims):
                factor = math.gcd(remaining_split, int(basis))
                split_factors.append((new_dim, factor))
                remaining_split //= factor
            split_factors.reverse()
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


def core_mappings_equal(
    left: Mapping[Any, Expr],
    right: Mapping[Any, Expr],
    num_cores: int,
) -> bool:
    """Return whether two symbolic mappings assign every core identically."""

    if left.keys() != right.keys():
        return False
    if num_cores <= 0:
        return False
    try:
        for dim in left:
            for core in range(num_cores):
                values = [
                    _owner_at_core(sympify(mapping[dim]), core)
                    for mapping in (left, right)
                ]
                if any(
                    value.free_symbols or value.is_integer is not True
                    for value in values
                ):
                    return False
                if values[0] != values[1]:
                    return False
        return True
    except (TypeError, ValueError):
        return False
