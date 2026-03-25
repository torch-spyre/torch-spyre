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

# Helper methods to handle views

import sympy
from typing import Optional, Sequence


def compute_relative_stride(
    rank: int, device_size: Sequence[sympy.Expr], dim_map: Sequence[int]
) -> list[sympy.Expr]:
    """
    Compute strides of device dimensions with respect to host dimensions
    """
    acc = [sympy.S.One] * rank
    rel_stride = [-1] * len(dim_map)
    for device_dim in range(len(dim_map) - 1, -1, -1):
        dim = dim_map[device_dim]
        if dim != -1:
            rel_stride[device_dim] = acc[dim]
            acc[dim] *= device_size[device_dim]
    return rel_stride


def compute_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Compute an array of coordinate expressions from an index expression.

    Stride and index must be relative to the same storage (both host or device).
    Stride values<=0 are ignored.
    """
    # find stride immediately strictly larger that dim stride
    n = len(size)
    next_stride = [sympy.oo] * n
    for i in range(n):
        for j in range(n):
            # n^2 is ok since n is small
            if next_stride[i] > stride[j] and stride[j] > stride[i]:
                next_stride[i] = stride[j]
    # compute coordinate expressions
    coordinates = [sympy.S.Zero] * n
    vars = index.free_symbols
    for var in vars:
        if var_ranges[var] <= 1:
            continue  # ignore var with trivial range
        # isolate current var
        term = index.subs({v: 0 for v in vars - {var}})
        # compute index({var=1}) and index({var=var_ranges[var]})
        step = term.subs(var, 1)
        limit = term.subs(var, var_ranges[var])
        # find primary dim with largest stride less than or equal to step
        primary_stride = 0
        primary_dim = -1
        for dim in range(n):
            if size[dim] == 1:
                continue  # ignore dim with size 1
            st = stride[dim]
            if st <= step and st > primary_stride:
                # found candidate primary dim
                primary_stride = st
                primary_dim = dim
            elif st > step and st < limit:
                # var range intersects dim, add term
                if next_stride[dim] < limit:
                    # var range overflows dim
                    coordinates[dim] += var * step % next_stride[dim] // st
                else:
                    coordinates[dim] += var * step // st
        # add term for primary dim
        if next_stride[primary_dim] < limit:
            coordinates[primary_dim] += (
                # var range overflows primary dim
                var * step % next_stride[primary_dim] // primary_stride
            )
        else:
            coordinates[primary_dim] += var * step // primary_stride
    return coordinates


# deprecated: replace with compute_coordinates with stride_map
def compute_device_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    device_size: Sequence[sympy.Expr],
    dim_map: Sequence[int],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Derive an array of coordinate expressions into a device tensor from an index
    """
    rel_stride = compute_relative_stride(len(size), device_size, dim_map)
    host_coordinates = compute_coordinates(size, stride, var_ranges, index)
    coordinates = [sympy.S.Zero] * len(device_size)
    for dim in range(len(device_size)):
        if dim_map[dim] == -1:
            continue
        expr = host_coordinates[dim_map[dim]]
        vars = expr.free_symbols
        for var in vars:
            term = expr.subs({v: 0 for v in vars - {var}})
            step = term.subs(var, 1)
            limit = term.subs(var, var_ranges[var])
            if limit > rel_stride[dim] and step < rel_stride[dim] * device_size[dim]:
                coordinates[dim] += term // rel_stride[dim]
    return coordinates


def _is_range_subset(expr: sympy.Expr, coord: sympy.Expr, v: sympy.Symbol) -> bool:
    """
    Return True if the set of values expr can produce (as v varies) is a subset
    of the values coord can produce.

    Handles two cases:
    - coord == v: coord is unbounded, so any expr in v is a subset.
    - coord == Mod(v, b) and expr == Mod(v, a) with a <= b: [0,a-1] ⊆ [0,b-1].
    """
    if coord == v:
        return True
    if (
        isinstance(coord, sympy.Mod)
        and isinstance(expr, sympy.Mod)
        and coord.args[0] == v
        and expr.args[0] == v
    ):
        coord_mod = coord.args[1]
        expr_mod = expr.args[1]
        return bool(sympy.Le(expr_mod, coord_mod))
    return False


def matching_dim(coords: list[sympy.Expr], expr: sympy.Expr) -> Optional[int]:
    """
    Given a coordinate array and an expression, determine if there is a unique
    dimension in coords whose possible values are a superset of expr's possible
    values (both expressed in the single free variable of expr).  Return None if
    expr does not have exactly one free variable or if there is not exactly one
    matching dimension in coords.
    """
    if len(expr.free_symbols) != 1:
        return None
    v = next(iter(expr.free_symbols))
    dims = [d for d, e in enumerate(coords) if _is_range_subset(expr, e, v)]
    if len(dims) != 1:
        return None
    else:
        return dims[0]
