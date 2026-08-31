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
from itertools import permutations

from sympy import Expr, Integer, Mod, Symbol, floor

from .op_spec import TensorWorkDivision


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
        raise ValueError(f"grouped dimensions are not in the operation: {unknown_dims}")
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
    if owner_count <= 0 or num_cores % owner_count:
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

    new_splits: dict[Symbol, int] = {}
    new_core_map: dict[Symbol, Expr] = {}
    for old_dim, split in division.work_slices.items():
        new_dims = dimension_remap[old_dim]
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
            previous = (new_splits.get(new_dim), new_core_map.get(new_dim))
            if previous[0] is not None and previous != (factor, new_slot):
                raise ValueError(f"conflicting normalized ownership on {new_dim}")
            new_splits[new_dim] = factor
            new_core_map[new_dim] = new_slot
            slot_stride *= factor
    return TensorWorkDivision(
        new_splits,
        new_core_map,
        num_cores=division.num_cores,
    )


def finalize_tensor_work_divisions(
    iteration_space: Mapping[Symbol, tuple[Expr, int]],
    divisions: Sequence[TensorWorkDivision | None],
) -> tuple[TensorWorkDivision | None, ...]:
    """Derive each tensor's owners from its final aligned partition."""

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
                f"tensor ownership dimensions are not aligned: {unknown_dims}"
            )

        if division.num_cores is None:
            raise ValueError(
                "tensor ownership must carry its physical core domain before alignment"
            )
        result.append(
            TensorWorkDivision(
                work_slices,
                derive_partition_mapping(
                    tuple(work_slices),
                    tuple(work_slices.values()),
                    division.num_cores,
                ),
                num_cores=division.num_cores,
            )
        )
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
        if division.work_slices and division.num_cores not in (None, num_cores):
            raise ValueError(
                "LX tensor ownership and operation use different core domains: "
                f"{division.num_cores} != {num_cores}"
            )
        for dim, split in division.work_slices.items():
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

    # Tensor-owned dimensions occupy the outer, contiguous groups. At most five
    # dimensions can be split on 32 cores, so trying their radix orders is small.
    for order in permutations(constrained):
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
    left: Mapping[Symbol, Expr],
    right: Mapping[Symbol, Expr],
    num_cores: int,
) -> bool:
    """Return whether two symbolic mappings assign every core identically."""

    if left.keys() != right.keys():
        return False
    core_id = Symbol("core_id")
    return all(
        int(left[dim].subs(core_id, core)) == int(right[dim].subs(core_id, core))
        for dim in left
        for core in range(num_cores)
    )
