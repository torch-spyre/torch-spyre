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
from typing import Optional, Sequence, Dict, Tuple, Callable
from torch.utils._sympy.functions import ModularIndexing, FloorDiv

from torch._inductor.virtualized import V

from .errors import Unsupported


def find_repeat_vars(index_exprs, var_ranges):
    repeat_info = {}
    for var, var_range in var_ranges.items():
        for expr in index_exprs:
            all_mods = expr.find(sympy.Mod)
            mods = []
            for m in all_mods:
                if m.has(var):
                    mods.append(m)
            if len(mods) != 1:
                continue
            node = mods[0]
            base, modulus = node.args
            if not sympy.simplify(modulus < var_range):
                continue

            vars_in_expr = expr.free_symbols
            term = expr.xreplace({v: 0 for v in vars_in_expr - {var}})

            if term == node:
                repeat_info[var] = {
                    "modulus": modulus,
                    "node": node,
                    "kind": "mod",
                }
                break
            if isinstance(term, sympy.Mul):
                coeff = sympy.S.One
                found = False
                for arg in term.args:
                    if not found and arg == node:
                        found = True
                    else:
                        coeff *= arg
                if found:
                    repeat_info[var] = {
                        "modulus": modulus,
                        "node": node,
                        "kind": "mul_mod",
                        "coeff": coeff,
                    }
                    break

    return repeat_info


def convert_modular_indexing(expr: sympy.Expr) -> sympy.Expr:
    """
    ModularIndexing(a, b, c) represents (a // b) % c
    If b == 1: Mod(a, c)
    Otherwise: Mod(FloorDiv(a, b), c)
    """
    if isinstance(expr, ModularIndexing):
        base, divisor, modulus = expr.args
        if divisor == 1:
            # ModularIndexing(a, 1, c) = a % c
            return sympy.Mod(base, modulus)
        else:
            # ModularIndexing(a, b, c) = (a // b) % c
            return sympy.Mod(FloorDiv(base, divisor), modulus)
    elif isinstance(expr, (sympy.Add, sympy.Mul)):
        new_args = [convert_modular_indexing(arg) for arg in expr.args]
        return expr.func(*new_args)
    else:
        return expr


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
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
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

    # Convert ModularIndexing expressions to sympy.Mod before processing
    index = convert_modular_indexing(index)
    repeat_info = find_repeat_vars([index], var_ranges)
    if not hasattr(V.graph, "_repeat_info"):
        V.graph._repeat_info = dict(repeat_info)
    else:
        V.graph._repeat_info.update(repeat_info)

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
            # Indirect index variables (tmp0/indirect0) are not loop vars.
            # Skip if indirect_sizes not provided — allows pre-scheduler
            # code that doesn't yet support indirect access to proceed.
            if indirect_sizes is not None and var in indirect_sizes:
                range_val = indirect_sizes[var]
            elif indirect_sizes is not None:
                raise Unsupported(
                    f"indirect symbol {var} not found in indirect_sizes {indirect_sizes}"
                )
            else:
                continue
        else:
            range_val = var_ranges[var]

        # Skip vars with trivial range.  For symbolic ranges we cannot
        # statically determine triviality, so we assume they are non-trivial.
        if isinstance(range_val, (int, sympy.Integer)) and int(range_val) <= 1:
            continue

        # isolate current var
        term = index.xreplace({v: 0 for v in vars - {var}})

        if var in repeat_info:
            info = repeat_info[var]
            if info["kind"] == "mod":
                add_term(var=info["node"], step=sympy.S.One, limit=info["modulus"])
            elif info["kind"] == "mul_mod":
                coeff = info["coeff"]
                add_term(var=info["node"], step=coeff, limit=coeff * info["modulus"])
            continue

        # compute index({var=1}) and index({var=var_ranges[var]})
        step = term.xreplace({var: 1})
        limit = term.xreplace({var: range_val})
        add_term(var=var, step=step, limit=limit)

    # NOTE: indirect_access_subs substitution is NOT applied here. It is deferred to
    # after align_tensors() so that indirect symbols are decomposed as regular variables.
    # The substitution is applied in simplify_op_spec() after align_tensors completes.
    return coordinates


def _is_range_subset(expr: sympy.Expr, coord: sympy.Expr, v: sympy.Symbol) -> bool:
    """
    Return True if the set of values expr can produce (as v varies) is a subset
    of the values coord can produce.

    Handles two cases:
    - coord == v: coord is unbounded, so any expr in v is a subset.
    - coord == Mod(v, b) and expr == Mod(v, a) with a <= b: [0,a-1] ⊆ [0,b-1].

    Both coord and expr can have optional constant offsets, but they must match.
    """
    if expr.free_symbols == {v} and coord.free_symbols == {v}:
        # Strip constant offsets if both have them
        expr_offset = expr.subs(v, 0)
        coord_offset = coord.subs(v, 0)
        if expr_offset != coord_offset:
            return False
        expr = expr - expr_offset
        coord = coord - coord_offset

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
    synthetic_var_fn: Callable[[], sympy.Symbol],
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
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

    for dim_idx, (coordinate, dim_size) in enumerate(zip(coordinates, size)):
        # sympy uses floor to encode integer divisions, remove
        expr = coordinate.replace(sympy.floor, lambda x: x)
        vars = expr.free_symbols
        offset = expr.xreplace({var: sympy.S.Zero for var in vars})

        if len(vars) == 0:
            if dim_size > 1 and dim_idx != len(size) - 1:
                # A non-stick dimension with no variables but size > 1 indicates an elided
                # dimension with offset/gap. Create a new variable to restore this dimension.
                var = synthetic_var_fn()
                var_ranges[var] = 1
                num = den = mod = sympy.S.One
                terms.append(Term(num, den, var, mod, dim_size, offset))
            else:
                assert offset == 0
                terms.append(Term(None, None, None, None, dim_size))
            continue
        # If any free symbols are not loop vars, check if they're indirect symbols
        # with known sizes (from indirect_sizes). If so, treat them like loop vars.
        if not vars.issubset(var_ranges.keys()):
            unknown_vars = vars - var_ranges.keys()
            if not (
                indirect_sizes is not None
                and unknown_vars.issubset(indirect_sizes.keys())
            ):
                # Symbols with unknown ranges: pass the raw coordinate through
                # as an opaque offset on a var=None term.
                terms.append(Term(None, None, None, None, dim_size, offset=coordinate))
                continue
        dim_terms = []  # terms for current dimension
        for var in vars:
            # Resolve the range for this variable: loop var from var_ranges, or indirect from indirect_sizes
            if var in var_ranges:
                var_range = var_ranges[var]
            elif indirect_sizes is not None and var in indirect_sizes:
                var_range = indirect_sizes[var]
            else:
                raise Unsupported(
                    f"Variable {var} in coordinate {expr} has no entry in var_ranges or indirect_sizes"
                )

            # extract term for each var
            term = expr.xreplace({v: 0 for v in vars - {var}}) - offset
            # pattern match expression tree, there is small number of possibilities
            if term.is_symbol:
                dim_terms.append(
                    Term(sympy.S.One, sympy.S.One, var, var_range, dim_size)
                )
            elif term.func == sympy.Mod:
                dim_terms.append(
                    Term(sympy.S.One, sympy.S.One, var, term.args[1], dim_size)
                )
            elif term.func == sympy.Mul and term.args[0].is_rational:
                expr0, expr1 = term.args
                mod = expr1.args[1] if expr1.func == sympy.Mod else var_range
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
        # sort dim_terms in increasing (num, mod) order so that z + offset
        # vars (num=1, mod=1) always sort before real iteration vars (num=1, mod=N)
        # when num is equal
        dim_terms.sort(
            key=lambda t: (
                _concretize_for_cmp(t.num),
                _concretize_for_cmp(t.mod),
            )
        )

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
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
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

    # Save original (possibly symbolic) range expressions before concretizing.
    # The algorithm below requires concrete ints for sorting and integer division,
    # but we must propagate symbolic expressions forward so downstream passes
    # (work_division, SDSC codegen) see the symbols to extract fields, not size_hints.
    orig_ranges = {var: val[0] for var, val in iteration_space.items()}
    # local import: pass_utils imports compute_coordinates/matching_dim from
    # this module, so importing at module scope would create a cycle.
    from .pass_utils import finite_upper_or_none

    def _bounded_or_hint(expr, hint):
        """Return ``expr`` unless it's an unbounded symbolic expression.

        Auto-dynamic symbols (Dynamo promoting an int on retrace, with no
        finite max and are not symbolic dims ) carry no bound downstream passes
        (work_division, SDSC codegen) can size buffers/loops with. Fall back to
        the concretized ``hint`` instead of propagating an unbounded symbol.
        """
        if hasattr(expr, "free_symbols") and expr.free_symbols:
            if finite_upper_or_none(expr) is None:
                return hint
        return expr

    repeat_info: dict = getattr(V.graph, "_repeat_info", {})

    var_ranges = {
        var: _concretize_for_cmp(val[0]) for var, val in iteration_space.items()
    }

    # work division for each variable
    op_it_space_splits = {var: val[1] for var, val in iteration_space.items()}

    new_vars: list[sympy.Symbol] = []
    _synthetic_var_idx: int = 0

    # return a synthetic variable, creating a new variable unless _synthetic_var_idx has been reset
    # there is no need for distinct synthetic variables for dimensions of size 1 across tensors
    def synthetic_var():
        nonlocal _synthetic_var_idx
        if _synthetic_var_idx < len(new_vars):
            var = new_vars[_synthetic_var_idx]
        else:
            var = sympy.symbols(f"z{len(new_vars)}")
            new_vars.append(var)
        _synthetic_var_idx += 1
        return var

    all_terms = []  # terms for each tensor
    stick_dim = []  # stick var for each tensor
    stick_size = []  # stick size for each tensor

    for tensor in tensors:
        _synthetic_var_idx = 0  # reuse synthetic_var across tensors
        terms = normalize_coordinates(
            var_ranges,
            tensor["size"],
            tensor["coordinates"],
            synthetic_var,
            indirect_sizes,
        )
        stick_dim.append(terms[-1].var)
        stick_size.append(terms[-1].dim_size)
        all_terms.append(terms)

    _synthetic_var_idx = len(new_vars)  # do not reuse synthetic vars after this point

    # for each variable collect bounds (den and mod) for all terms involving variable
    # exclude the sick_size resulting from tiling the stick dimension
    # Collect all variables that appear in terms (loop vars + indirect symbols).
    # dict.fromkeys preserves insertion order; set() does not. This matters for two
    # reasons: (1) frontend determinism; (2) backend workaround — the backend is
    # sensitive to iteration_space dim label order even though semantically it
    # should not be.
    all_vars = dict.fromkeys(var_ranges.keys())
    for terms in all_terms:
        for term in terms:
            if term.var is not None:
                all_vars[term.var] = None

    splits: dict[sympy.Symbol, sympy.Expr] = {var: set() for var in all_vars}

    for i, terms in enumerate(all_terms):
        for num, den, var, mod, dim_size, offset in [astuple(term) for term in terms]:
            if var is not None:
                if den != stick_size[i] or var != stick_dim[i]:
                    # add den to splits unless stick dim and stick size
                    splits[var].add(den)
                if (
                    mod != stick_size[i]
                    or var != stick_dim[i]
                    or var in repeat_info.keys()
                ):
                    # add mod to splits unless stick dim and stick size
                    splits[var].add(mod)

    if hasattr(V.graph, "_repeat_info"):
        V.graph._repeat_info.clear()

    # Insert restored size-1 dimensions with offset/gap to the other tensors
    for var in new_vars:
        assert var_ranges[var] == 1
        for terms in all_terms:
            if not any(term.var == var for term in terms):
                new_term = Term(sympy.S.One, sympy.S.One, var, sympy.S.One, sympy.S.One)
                terms.insert(0, new_term)

    # sort splits
    splits = {var: sorted(val) for var, val in splits.items()}

    # create new vars, var ranges, and work division for each variable
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
                new_var = synthetic_var()  # create new variable
                new_var_ranges[new_var] = split[i + 1] // split[i]
                remap[var].append(new_var)

            # distribute work division for old var to new vars
            for v in reversed(remap[var]):
                new_op_it_space_splits[v] = math.gcd(div, new_var_ranges[v])
                div //= new_op_it_space_splits[v]
        else:
            # no splits keep existing var, range, and work division
            # may happen with a single stick since the stick size is omitted
            # downstream passes receive the symbolic expression, not the concretized
            # hint -- unless the symbol is unbounded, see _bounded_or_hint.
            # Synthetic vars (z0, z1, …) are introduced by normalize_coordinates
            # for restored size-1 dims and are not in orig_ranges; fall back to
            # the concretized value (always 1) for those.
            new_var_ranges[var] = _bounded_or_hint(
                orig_ranges.get(var, var_ranges[var]), var_ranges[var]
            )
            # var can be a loop var or an indirect symbol
            if var in var_ranges:
                new_var_ranges[var] = var_ranges[var]
            elif indirect_sizes is not None and var in indirect_sizes:
                new_var_ranges[var] = indirect_sizes[var]
            else:
                raise Unsupported(
                    f"Variable {var} has no range in var_ranges or indirect_sizes"
                )
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
                # offset holds either 0 (broadcast/scalar dim) or an IndirectAccess
                # (indirect load access) that must pass through unchanged.
                size.append(dim_size)
                coordinates.append(offset)
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
                if stick_dim[i] in c.free_symbols or (s == 1 and c == 0):
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
                    if t["size"][i] == 1 and t["coordinates"][i] == 0:
                        t["coordinates"][i] = stick_dim_var // t["size"][-1]
                        t["coordinates"][-1] = stick_dim_var % t["size"][-1]
                        break
    # Restore original symbolic expressions wherever the algorithm left the
    # concretized value unchanged (i.e. no splits were applied to that dim).
    # Only restore when the symbol carries a finite bound (e.g. the user
    # passed mark_dynamic(max=...)) -- downstream passes (work_division,
    # SDSC codegen) need a bound to size buffers/loops. Auto-dynamic symbols
    # (Dynamo promoting an int on retrace, with no finite max) have no such
    # bound, so fall back to the concretized size-hint instead of
    # propagating an unbounded symbol. See work_division.py's symbol_meta
    # for the same distinction.
    for var, orig_expr in orig_ranges.items():
        if var not in new_var_ranges or new_var_ranges[var] != var_ranges[var]:
            continue
        new_var_ranges[var] = _bounded_or_hint(orig_expr, var_ranges[var])

    # Iteration space should only contain loop variables, not indirect symbols.
    # Filter out any indirect symbols that were added during normalization.
    indirect_syms = set(indirect_sizes.keys()) if indirect_sizes else set()
    new_iteration_space = {
        k: (v, new_op_it_space_splits[k])
        for k, v in new_var_ranges.items()
        if k not in indirect_syms
    }

    return new_iteration_space, new_tensors
