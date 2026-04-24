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

from dataclasses import dataclass, astuple
import math
import sympy
from typing import Optional, Sequence, Dict, Tuple

from torch._inductor.virtualized import V


# NOTE: this is intentionally a local copy of pass_utils.concretize_expr.
# views.py cannot import from pass_utils because pass_utils imports
# compute_coordinates from views (circular dependency).  The duplication
# is acceptable because both are thin wrappers around V.graph.sizevars.size_hint.
def _concretize_for_cmp(expr):
    """Return a concrete numeric value for use in comparison operators only.

    Used for branching decisions inside ``compute_coordinates`` and
    ``align_tensors`` (e.g. choosing which dimension a loop variable maps to).
    The coordinate *output* expressions stay symbolic.

    Returns a Python ``int`` for ordinary values, and ``math.inf`` /
    ``-math.inf`` for sympy infinities (used as ``limit=sympy.oo`` sentinels
    in ``add_term`` when the index has a non-zero storage offset, e.g. for
    slice / split ops).  ``int(sympy.oo)`` would raise; ``math.inf`` works
    correctly in ``<`` / ``>`` comparisons against ints and sympy values.

    TODO(issue#1373): once these algorithms use sympy predicates or
    SizeVarAllocator guards, this function can be removed.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    # sympy.oo / -sympy.oo cannot be cast to int; preserve as Python infinity.
    if expr == sympy.oo:
        return math.inf
    if expr == -sympy.oo:
        return -math.inf
    if isinstance(expr, float):
        return expr  # passthrough (incl. math.inf); avoids int(math.inf) error
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        return V.graph.sizevars.size_hint(expr)
    return int(expr)


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

    ``size`` and ``stride`` must be concrete (int) values—callers such as
    ``host_coordinates`` concretize them before calling.  ``var_ranges``
    may contain symbolic expressions (e.g. a dynamic batch dimension); the
    algorithm concretizes range values only for comparison logic, while the
    output coordinate expressions remain symbolic.
    """
    assert all(isinstance(s, (int, sympy.Integer)) for s in stride), (
        f"compute_coordinates requires concrete strides, got {stride}"
    )
    assert all(isinstance(s, (int, sympy.Integer)) for s in size), (
        f"compute_coordinates requires concrete sizes, got {size}"
    )
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

    def add_term(var, step, limit):
        # Concretize step and limit for comparison logic only.  The symbolic
        # ``step`` and ``limit`` are still used in the coordinate *output*
        # expressions (``var * step // st``), preserving symbolic output.
        # TODO(issue#1373): replace with sympy predicates to avoid concretization.
        concrete_step = _concretize_for_cmp(step)
        concrete_limit = _concretize_for_cmp(limit)

        # find primary dim with largest stride less than or equal to step
        primary_stride = 0
        primary_dim = -1
        for dim in range(n):
            if size[dim] == 1:
                continue  # ignore dim with size 1
            st = stride[dim]
            if st <= concrete_step and st > primary_stride:
                # found candidate primary dim
                primary_stride = st
                primary_dim = dim
            elif st > concrete_step and st < concrete_limit:
                # var range intersects dim, add term
                if next_stride[dim] < concrete_limit:
                    # var range overflows dim
                    coordinates[dim] += var * step % next_stride[dim] // st
                else:
                    coordinates[dim] += var * step // st
        # add term for primary dim
        if primary_stride > 0:
            if next_stride[primary_dim] < concrete_limit:
                coordinates[primary_dim] += (
                    # var range overflows primary dim
                    var * step % next_stride[primary_dim] // primary_stride
                )
            else:
                coordinates[primary_dim] += var * step // primary_stride

    vars = index.free_symbols
    offset = index.xreplace({v: 0 for v in vars})
    if offset > 0:
        index = index - offset
        add_term(var=offset, step=sympy.S.One, limit=sympy.oo)

    for var in vars:
        # Skip symbols that are not loop variables (e.g. size symbols
        # injected by dynamic shapes that appear in the index expression
        # but are not iteration variables).
        if var not in var_ranges:
            continue

        range_val = var_ranges[var]

        # Skip vars with trivial range.  For symbolic ranges we cannot
        # statically determine triviality, so we assume they are non-trivial.
        if isinstance(range_val, (int, sympy.Integer)) and int(range_val) <= 1:
            continue

        # isolate current var
        term = index.xreplace({v: 0 for v in vars - {var}})
        # compute index({var=1}) and index({var=var_ranges[var]})
        step = term.xreplace({var: 1})
        limit = term.xreplace({var: range_val})
        add_term(var=var, step=step, limit=limit)

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


@dataclass(order=True)
class Term:
    """
    A term num*(var%mod)//den + offset in a coordinate expression.
    Includes the size of the dimension the expression is intended for.
    Constant including zero is represented as Term(None, None, None, None, dim_size, offset).
    """

    num: sympy.Expr | None  # numerator
    den: sympy.Expr | None  # denominator
    var: sympy.Expr | None  # variable
    mod: sympy.Expr | None  # modulo
    dim_size: sympy.Expr
    offset: sympy.Expr = sympy.S.Zero  # offset


def normalize_coordinates(
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    size: Sequence[sympy.Expr],
    coordinates: Sequence[sympy.Expr],
) -> list[Term]:
    """
    Normalize coordinate expressions obtained from compute_coordinates.

    If mod is absent from term assume term does not overflow dim_size.
    Assume num or den is 1.

    Break each expression into list of terms.
    If expr has no mod, use var_range instead.

    Split dimension into n dimensions if expression has n>1 terms.
    Split dim_size into n according to iteration range of each term.
    Fuse contiguous dimensions if corresponding terms can be fused.
    """
    # terms in non-increasing stride order
    terms = []

    for coordinate, dim_size in zip(coordinates, size):
        # sympy uses floor to encode integer divisions, remove
        expr = coordinate.replace(sympy.floor, lambda x: x)
        vars = expr.free_symbols
        offset = expr.xreplace({var: sympy.S.Zero for var in vars})
        if len(vars) == 0:
            terms.append(Term(None, None, None, None, dim_size, offset))
            continue
        dim_terms = []  # terms for current dimension
        for var in vars:
            # extract term for each var
            term = expr.xreplace({v: 0 for v in vars - {var}}) - offset
            # pattern match expression tree, there is small number of possibilities
            if term.is_symbol:
                dim_terms.append(
                    Term(sympy.S.One, sympy.S.One, var, var_ranges[var], dim_size)
                )
            elif term.func == sympy.Mod:
                dim_terms.append(
                    Term(sympy.S.One, sympy.S.One, var, term.args[1], dim_size)
                )
            elif term.func == sympy.Mul and term.args[0].is_rational:
                expr0, expr1 = term.args
                mod = expr1.args[1] if expr1.func == sympy.Mod else var_ranges[var]
                # TODO: handle non-unit fractions
                # https://github.com/torch-spyre/torch-spyre/issues/1353
                assert expr0.numerator == 1 or expr0.denominator == 1, (
                    f"Unsupported coordinate expression {expr}"
                )
                dim_terms.append(
                    Term(expr0.numerator, expr0.denominator, var, mod, dim_size)
                )
            else:
                assert False, f"Unsupported coordinate expression {expr}"

        # sort dim_terms in increasing num order
        dim_terms.sort()

        for dim_term in dim_terms[::-1]:
            dim_term.offset = offset // dim_term.num
            offset %= dim_term.num

        # split dims with n>1 terms
        split_dim_terms = []

        cum_size = 1
        # for all terms but the last
        for i in range(0, len(dim_terms) - 1):
            # set dim_size to numerator of next term
            dim_terms[i].dim_size = dim_terms[i + 1].num
            # set numerator of next term to 1
            dim_terms[i + 1].num = 1
            # compute cumulative dim_size of all terms up to current term
            cum_size *= dim_terms[i].dim_size
            # append corrected term
            split_dim_terms.append(dim_terms[i])
        # set last dim_size to residual size and append
        dim_terms[-1].dim_size = dim_size // cum_size
        split_dim_terms.append(dim_terms[-1])

        # accumulate terms in reverse order to ensure non-increasing device strides
        terms += reversed(split_dim_terms)

    # fuse contiguous dimensions when possible
    # never fuse last dimension = stick dimension!
    fused_terms = []
    fused_term = terms[0]
    for term in terms[1:-1]:
        if (
            fused_term.num == 1
            and fused_term.var == term.var
            and fused_term.den == term.mod
        ):
            # fuse terms
            fused_term.num = term.num
            fused_term.den = term.den
            fused_term.dim_size *= term.dim_size
            fused_term.offset += term.offset
        else:
            if fused_term.dim_size > 1 or fused_term.var is not None:
                fused_terms.append(fused_term)
            fused_term = term
    if fused_term.dim_size > 1 or fused_term.var is not None:
        fused_terms.append(fused_term)
    # add term for stick dimension
    fused_terms.append(terms[-1])

    return fused_terms


def align_tensors(
    iteration_space: Dict[sympy.Symbol, Tuple[sympy.Expr, int]],
    tensors: list[Dict[str, list[sympy.Expr]]],
) -> tuple[
    (dict[sympy.Symbol, tuple[sympy.Expr, int]], list[dict[str, list[sympy.Expr]]])
]:
    """
    Transform op iteration space and tensor arguments to satisfy codegen requirements.
    """

    # Concretize range values for the algorithm: align_tensors performs
    # sorting, math.gcd, and integer division that require concrete ints.
    # Coordinate *expressions* remain symbolic (they reference loop variable
    # Symbols, not range values).
    # TODO(issue#1373): make align_tensors symbolic-aware so concretization can
    #              be removed.

    var_ranges = {
        var: _concretize_for_cmp(val[0]) for var, val in iteration_space.items()
    }

    # core division for each variable
    op_it_space_splits = {var: val[1] for var, val in iteration_space.items()}

    # for each variable collect bounds (den and mod) for all terms involving variable
    # exclude the sick_size resulting from tiling the stick dimension
    splits: dict[sympy.Symbol, sympy.Expr] = {var: set() for var in var_ranges.keys()}

    all_terms = []  # terms for each tensor
    stick_dim = []  # stick var for each tensor
    stick_size = []  # stick size for each tensor

    n = 0  # next symbol number

    # TODO: Current support of size-1 dimensions are limited. We will generalize this
    # in follow-up PRs. See #1548
    is_pointwise_op = len(set([len(tensor["size"]) for tensor in tensors])) == 1
    if is_pointwise_op:
        rank = len(tensors[0]["size"])
        vars = var_ranges.keys()
        for i in range(rank):
            coords = [tensor["coordinates"][i] for tensor in tensors]
            if all(c == 0 for c in coords):
                continue
            if not any(c == 0 for c in coords):
                continue
            if not any(c.is_number and c > 0 for c in coords):
                continue

            new_var = sympy.symbols(f"z{n}")
            n += 1
            var_ranges[new_var] = 1
            splits[new_var] = set()

            for tensor in tensors:
                coord = tensor["coordinates"][i]
                if coord == 0:
                    tensor["coordinates"][i] = new_var
                else:
                    tensor["coordinates"][i] = new_var + coord

    for tensor in tensors:
        terms = normalize_coordinates(var_ranges, tensor["size"], tensor["coordinates"])
        stick_dim.append(terms[-1].var)
        stick_size.append(terms[-1].dim_size)
        all_terms.append(terms)
        for num, den, var, mod, dim_size, offset in [astuple(term) for term in terms]:
            if var is not None:
                if den != stick_size[-1] or var != stick_dim[-1]:
                    # add den to splits unless stick dim and stick size
                    splits[var].add(den)
                if mod != stick_size[-1] or var != stick_dim[-1]:
                    # add mod to splits unless stick dim and stick size
                    splits[var].add(mod)

    # sort splits
    splits = {var: sorted(val) for var, val in splits.items()}

    # create new vars, var ranges, and core division for each variable
    # with one var per segment (split[i], split[i+1])
    new_var_ranges = {}
    new_op_it_space_splits = {}
    remap = {}  # map old var to new vars in splits order
    for var, split in splits.items():
        div = op_it_space_splits[var] if var in op_it_space_splits else 1
        if len(split) > 1:
            new_var_ranges[var] = split[1] // split[0]
            remap[var] = [var]  # reuse variable name for 1st segment
            for i in range(1, len(split) - 1):
                new_var = sympy.symbols(f"z{n}")  # create new variable
                n += 1
                new_var_ranges[new_var] = split[i + 1] // split[i]
                remap[var].append(new_var)

            # distribute core division for old var to new vars
            for v in reversed(remap[var]):
                new_op_it_space_splits[v] = math.gcd(div, new_var_ranges[v])
                div //= new_op_it_space_splits[v]
        else:
            # no splits keep existing var, range, and core division
            # may happen with a single stick since the stick size is omitted
            new_var_ranges[var] = var_ranges[var]
            new_op_it_space_splits[var] = (
                op_it_space_splits[var] if var in op_it_space_splits else 1
            )

    # create new tensors with new sizes and coordinate expressions matching new vars
    new_tensors = []
    for j, terms in enumerate(all_terms):
        size = []
        coordinates = []
        for num, den, var, mod, dim_size, offset in [
            astuple(term) for term in terms[:-1]
        ]:
            # for each term except last one (stick dim)
            if var is None:
                # dimension is not iterated over, keep as is
                size.append(dim_size)
                coordinates.append(sympy.S.Zero)
                continue
            # decompose dimension according to splits and tiling of stick dim
            low = (
                0
                if var == stick_dim[j]
                and den == stick_size[j]
                and den not in splits[var]
                else splits[var].index(den)
            )  # replace split[var].index(stick_size) with 0 for stick dim
            high = splits[var].index(mod)
            if low == high:
                size.append(dim_size)
                coordinates.append(var + offset)
            for i in reversed(range(low, high)):
                if i == splits[var].index(mod) - 1:
                    # upper bound of iteration range is dim_size * den
                    size.append(dim_size * den // splits[var][i])
                else:
                    # upper bound of iteration range is split
                    size.append(splits[var][i + 1] // splits[var][i])
                coordinates.append(remap[var][i] + offset // splits[var][i])
                offset %= splits[var][i]
            if var == stick_dim[j] and den == stick_size[j] and den not in splits[var]:
                # outer stick dim
                size[-1] //= den
                (offset, term) = coordinates[-1].as_coeff_Add()
                coordinates[-1] = term // den + offset
            if num > 1:
                # iteration skips over elements in dim, realize gap as new dimension
                size.append(num)
                coordinates.append(sympy.S.Zero)
        # add stick dim
        num, den, var, mod, dim_size, offset = astuple(terms[-1])
        size.append(dim_size)
        coordinates.append(
            (var % dim_size if var is not None else sympy.S.Zero) + offset
        )
        new_tensors.append({"size": size, "coordinates": coordinates})

    # decide desired rank for all tensors
    rank = 0
    for i, t in enumerate(new_tensors):
        not_found = 1
        if stick_dim[i] is None:
            for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                if c == 0 and s == 1:
                    not_found = 0
                    break
            # if no candidate outer stick dim, add 1 to desired rank
            rank = max(rank, len(t["size"]) + not_found)
        else:
            for c, s in zip(t["coordinates"][:-1], t["size"][:-1]):
                if stick_dim[i] in c.free_symbols or s == 1:
                    not_found = 0
                    break
            # if no candidate outer stick dim, add 1 to desired rank
            rank = max(rank, len(t["size"]) + not_found)

    # extend each tensor to desired rank with outer dims of size 1
    for t in new_tensors:
        gap = rank - len(t["size"])
        t["size"] = [sympy.S.One] * gap + t["size"]
        t["coordinates"] = [sympy.S.Zero] * gap + t["coordinates"]

    # ensure stick dim var occurs twice if it occurs once using a dim of size 1
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
