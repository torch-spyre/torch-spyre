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
from collections.abc import Sequence

from sympy import Expr, Integer, Mod, Symbol, floor


def core_to_slice_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
    *,
    contiguous_dim: int | None = None,
) -> dict[str, Expr]:
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
    result: dict[str, Expr] = {}
    for dim in dim_order:
        split = Integer(splits[dim])
        if split == 1:
            coordinate = Integer(0)
        elif stride == 1:
            coordinate = Mod(core_id, split)
        else:
            coordinate = Mod(floor(core_id / stride), split)
        result[str(dims[dim])] = coordinate
        stride *= split
    return result
