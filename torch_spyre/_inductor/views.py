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

import math
import sympy
from typing import Optional, Sequence


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
            if next_stride[i] > stride[j] and stride[j] > stride[i] and size[j] > 1:
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


def normalize_coordinates(var_ranges, size, coordinates):
    results = []
    for coordinate, dim_size in zip(coordinates, size):
        expr = coordinate.replace(sympy.floor, lambda x: x)
        vars = expr.free_symbols
        if len(vars) == 0:
            results.append([None, None, None, None, dim_size])
            continue
        result = []
        for var in vars:
            term = expr.subs({v: 0 for v in vars - {var}})
            if term.is_symbol:
                result.append(
                    [sympy.S.One, sympy.S.One, var, var_ranges[var], dim_size]
                )
            elif term.func == sympy.Mod:
                result.append([sympy.S.One, sympy.S.One, var, term.args[1], dim_size])
            elif term.func == sympy.Mul and term.args[0].is_rational:
                expr0, expr1 = term.args
                mod = expr1.args[1] if expr1.func == sympy.Mod else var_ranges[var]
                result.append([expr0.numerator, expr0.denominator, var, mod, dim_size])
            else:
                raise IndexError

        result.sort()

        r = []
        cum_size = result[0][0]
        for i in range(0, len(result) - 1):
            result[i][4] = result[i + 1][0]
            cum_size *= result[i][4]
            if i > 0:
                result[i][0] = 1
            r.append(result[i])
        if len(result) > 1:
            result[-1][0] = 1
        result[-1][4] = dim_size // cum_size
        r.append(result[-1])

        r.reverse()
        results += r

    new_results = []
    tmp = results[0]
    for result in results[1:-1]:
        if tmp[0] == 1 and tmp[1] == result[3] and tmp[2] == result[2]:
            tmp[0] = result[0]
            tmp[1] = result[1]
            tmp[4] *= result[4]
        else:
            if tmp[4] > 1:
                new_results.append(tmp)
            tmp = result
    if tmp[4] > 1:
        new_results.append(tmp)
    new_results.append(results[-1])
    return new_results


def align_tensors(iteration_space, tensors):
    var_ranges = {var: val[0] for var, val in iteration_space.items()}
    op_it_space_splits = {var: val[1] for var, val in iteration_space.items()}

    splits = {var: set() for var in var_ranges.keys()}
    breakdown = []
    stick_dim = []
    stick_size = []
    for t in tensors:
        intervals = normalize_coordinates(var_ranges, t["size"], t["coordinates"])
        stick_dim.append(intervals[-1][2])
        stick_size.append(intervals[-1][-1])
        breakdown.append(intervals)
        for num, den, var, mod, dim_size in intervals:
            if var is not None:
                if den != stick_size[-1] or var != stick_dim[-1]:
                    splits[var].add(den)
                if mod != stick_size[-1] or var != stick_dim[-1]:
                    splits[var].add(mod)
    splits = {var: sorted(val) for var, val in splits.items()}

    new_var_ranges = {}
    new_op_it_space_splits = {}
    n = 0
    remap = {}
    for var, split in splits.items():
        div = op_it_space_splits[var] if var in op_it_space_splits else 1
        if len(split) > 1:
            new_var_ranges[var] = split[1] // split[0]
            remap[var] = [var]
            for i in range(1, len(split) - 1):
                new_var = sympy.symbols(f"z{n}")
                n += 1
                new_var_ranges[new_var] = split[i + 1] // split[i]
                remap[var].append(new_var)
            for v in reversed(remap[var]):
                new_op_it_space_splits[v] = math.gcd(div, new_var_ranges[v])
                div //= new_op_it_space_splits[v]
        else:
            new_var_ranges[var] = var_ranges[var]
            new_op_it_space_splits[var] = (
                op_it_space_splits[var] if var in op_it_space_splits else 1
            )

    new_tensors = []
    for j, intervals in enumerate(breakdown):
        size = []
        coordinates = []
        for num, den, var, mod, dim_size in intervals[:-1]:
            if var is None:
                size.append(dim_size)
                coordinates.append(sympy.S.Zero)
                continue
            low = (
                0
                if var == stick_dim[j]
                and den == stick_size[j]
                and den not in splits[var]
                else splits[var].index(den)
            )
            for i in reversed(range(low, splits[var].index(mod))):
                if i == splits[var].index(mod) - 1:
                    size.append(dim_size * den // splits[var][i])
                else:
                    size.append(splits[var][i + 1] // splits[var][i])
                coordinates.append(remap[var][i])
            if var == stick_dim[j] and den == stick_size[j] and den not in splits[var]:
                size[-1] //= den
                coordinates[-1] //= den
            if num > 1:
                size.append(num)
                coordinates.append(sympy.S.Zero)
        num, den, var, mod, dim_size = intervals[-1]
        size.append(dim_size)
        coordinates.append(var % dim_size if var is not None else sympy.S.Zero)
        new_tensors.append({"size": size, "coordinates": coordinates})

    rank = 0
    for i, t in enumerate(new_tensors):
        if stick_dim[i] is None:
            not_found = 1
            for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                if c == 0 and s == 1:
                    not_found = 0
                    break
            rank = max(rank, len(t["size"]) + not_found)
            continue
        found = 1
        for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
            if stick_dim[i] in c.free_symbols or s == 1:
                found = 0
                break
        rank = max(rank, len(t["size"]) + found)

    for t in new_tensors:
        gap = rank - len(t["size"])
        t["size"] = [sympy.S.One] * gap + t["size"]
        t["coordinates"] = [sympy.S.Zero] * gap + t["coordinates"]

    for t in new_tensors:
        vars = t["coordinates"][-1].free_symbols
        if len(vars) == 1:
            stick_dim_var = next(iter(vars))
            found = False
            for i in range(len(t["coordinates"]) - 1):
                vars = t["coordinates"][i].free_symbols
                if stick_dim_var in vars:
                    found = True
                    continue
            if not found:
                for i in range(len(t["coordinates"]) - 1):
                    if t["size"][i] == 1:
                        t["coordinates"][i] = stick_dim_var // t["size"][-1]
                        t["coordinates"][-1] = stick_dim_var % t["size"][-1]
                        break

    new_iteration_space = {
        k: (v, new_op_it_space_splits[k]) for k, v in new_var_ranges.items()
    }

    return new_iteration_space, new_tensors
