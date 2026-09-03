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
from typing import Callable, Dict, Optional, Sequence, Tuple, cast
from torch.utils._sympy.functions import ModularIndexing, FloorDiv

from torch._inductor.virtualized import V

from .errors import Unsupported


def _mixed_radix_digits(expr, var, var_range, mods):
    """Describe an exact quotient/remainder digit chain for ``var``.

    A flattened loop variable commonly reaches a pre-flatten view as adjacent
    mixed-radix digits, for example ``Mod(hd, 128)`` and
    ``Mod(FloorDiv(hd, 128), 32)`` for a flattened ``H*D`` axis. Multiple Mods
    are safe in that case: each digit addresses a distinct tensor dimension.

    Return the digits in low-to-high order, or ``None`` when the expressions
    overlap, leave a gap, or do not cover the variable's full range. Keeping
    this recognition deliberately strict preserves rejection of ambiguous
    combinations such as ``Mod(x, 4) + Mod(x, 6)``.
    """
    term = expr.xreplace({s: 0 for s in expr.free_symbols - {var}})
    var_terms = [addend for addend in sympy.Add.make_args(term) if addend.has(var)]
    if len(var_terms) != len(mods):
        return None

    digits = []
    for node in mods:
        containing = [addend for addend in var_terms if addend.has(node)]
        if len(containing) != 1:
            return None
        addend = containing[0]
        if len([m for m in addend.atoms(sympy.Mod) if m.has(var)]) != 1:
            return None

        base, modulus = node.args
        if base == var:
            divisor = sympy.S.One
        elif isinstance(base, FloorDiv) and base.args[0] == var:
            divisor = base.args[1]
        else:
            return None

        coeff = sympy.simplify(addend / node)
        if (
            coeff.has(var)
            or coeff.is_Rational is not True
            or coeff <= 0
            or (coeff.numerator != 1 and coeff.denominator != 1)
        ):
            return None
        if any(not value.is_Integer or value <= 0 for value in (divisor, modulus)):
            return None
        digits.append(
            {
                "node": node,
                "modulus": modulus,
                "divisor": divisor,
                "coeff": coeff,
            }
        )

    digits.sort(key=lambda digit: int(digit["divisor"]))
    if digits[0]["divisor"] != 1:
        return None
    for low, high in zip(digits, digits[1:]):
        if sympy.simplify(low["divisor"] * low["modulus"] - high["divisor"]) != 0:
            return None
    if sympy.simplify(digits[-1]["divisor"] * digits[-1]["modulus"] - var_range) != 0:
        return None
    return digits


def find_repeat_vars(index_exprs, var_ranges):
    repeat_info = {}
    for var, var_range in var_ranges.items():
        for expr in index_exprs:
            all_mods = expr.find(sympy.Mod)
            mods = []
            for m in all_mods:
                if m.has(var):
                    mods.append(m)
            if len(mods) > 1:
                digits = _mixed_radix_digits(expr, var, var_range, mods)
                if digits is None:
                    raise Unsupported(
                        f"variable {var} (range {var_range}) appears in multiple Mod "
                        f"expressions {mods} and cannot be mapped to coordinates."
                    )
                repeat_info[var] = {"kind": "mixed_radix", "digits": digits}
                break
            if len(mods) == 0:
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
# is acceptable because both are thin wrappers around V.graph.sizevars.optimization_hint.
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
        return V.graph.sizevars.optimization_hint(expr)
    return int(expr)


def _decompose_constant_offset(
    offset: sympy.Expr,
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    coordinates: list[sympy.Expr],
) -> bool:
    """Attribute a constant storage offset to device coordinates positionally.

    A storage offset is a fixed host-flat position, i.e. a mixed-radix number in
    the layout's strides.  We peel it greedily from the largest stride down
    (``digit = remaining // stride[d]``), so a whole-dimension offset lands
    entirely on that dimension.  This is unlike ``add_term``'s per-dim modular
    split, which assumes the strides form a clean nested radix
    (``stride[outer] == stride[inner] * size[inner]``).  A padded layout breaks
    that assumption -- e.g. an fp16 base whose row width is not a multiple of the
    64-element stick has row stride 100 but a stick pair spanning 128, so the
    modular split leaks ``offset % 64`` onto the stick coordinate.  Positional
    peeling keeps the stick coordinate offset-free.

    Mutates ``coordinates`` in place and returns True on success.  Returns False
    without touching ``coordinates`` if the offset cannot be fully peeled (never
    expected for a valid host position), so the caller can fall back to
    ``add_term``.
    """
    n = len(size)
    dims = sorted(
        (d for d in range(n) if size[d] > 1 and stride[d] > 0),
        key=lambda d: stride[d],
        reverse=True,
    )
    remaining = offset
    digits: list[tuple[int, sympy.Expr]] = []
    for d in dims:
        if remaining < stride[d]:
            continue
        digit = remaining // stride[d]
        digits.append((d, digit))
        remaining -= digit * stride[d]
    if remaining != 0:
        return False
    for d, digit in digits:
        coordinates[d] += digit
    return True


def compute_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
    repeat_info_out: "dict[sympy.Symbol, dict] | None" = None,
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

    Raises ``Unsupported`` if ``index`` walks a dimension backwards (a loop
    variable with a negative coefficient, as produced by ``prims.rev``): a
    device coordinate can only ascend.
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
    if repeat_info_out is not None:
        repeat_info_out.update(repeat_info)

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

        # ``limit`` below ``step`` means the access walks the dimension
        # backwards (the index carries a term like ``N - 1 - var``, as
        # ``prims.rev`` / ``Tensor.flip`` produces).  Every dim test below
        # compares a non-negative ``stride[dim]`` against ``concrete_step``, so
        # a descending term can match no dim and would be dropped from
        # ``coordinates`` entirely -- silently yielding the coordinate for
        # ``var == 0`` at every iteration.  Device coordinates can only ascend,
        # so reject it loudly instead (see issue #3558).
        #
        # Testing ``limit - step`` rather than ``step``'s sign isolates the
        # direction from any additive constant folded into both: for a term
        # ``a*var + b`` over range ``R``, ``limit - step == a*(R - 1)``, so the
        # comparison sees ``a``'s sign alone.
        if concrete_limit < concrete_step:
            raise Unsupported(
                f"index term for {var} runs backwards (step {step}, limit "
                f"{limit}): reversed traversal of a tensor dimension cannot "
                f"be expressed as a device coordinate"
            )

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
        # A concrete offset is decomposed positionally so a padded layout does
        # not leak a modular residual onto the stick coordinate (see
        # _decompose_constant_offset).  Symbolic offsets, or an offset that
        # cannot be fully peeled, fall back to add_term's original behavior.
        handled = not offset.free_symbols and _decompose_constant_offset(
            offset, size, stride, coordinates
        )
        if not handled:
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
            elif info["kind"] == "mixed_radix":
                for digit in info["digits"]:
                    add_term(
                        var=digit["node"],
                        step=digit["coeff"],
                        limit=digit["coeff"] * digit["modulus"],
                    )
            continue

        # compute index({var=1}) and index({var=var_ranges[var]})
        step = term.xreplace({var: 1})
        limit = term.xreplace({var: range_val})

        mods_with_var = [m for m in term.atoms(sympy.Mod) if m.has(var)]
        if len(mods_with_var) > 1:
            raise Unsupported(
                f"variable {var} (range {range_val}) appears in multiple Mod "
                f"expressions {mods_with_var} and cannot be mapped to coordinates."
            )

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
    compare_value: Callable[[sympy.Expr], int | float] = _concretize_for_cmp,
) -> list[Term]:
    """
    Normalize coordinate expressions obtained from compute_coordinates.

    If mod is absent from term assume term does not overflow dim_size.
    Assume num or den is 1.

    Break each expression into list of terms.
    If expr has no mod, use var_range instead.

    Split dimension into n dimensions if expression has n>1 terms.
    Split dim_size into n according to iteration range of each term.
    Fuse contiguous dimensions if corresponding terms can be fused.  Size-1
    device dims with a constant zero coordinate are dropped, and do not stop
    the dims on either side of them from fusing.
    """

    def normalize_var_expr(term, var, var_range, dim_size):
        """Convert one single-variable coordinate term to ``Term``.

        ``Mod(FloorDiv(var, divisor), radix)`` is one digit of a
        mixed-radix decomposition. Its equivalent normalized form is
        ``(var % (divisor * radix)) // divisor``; retaining both bounds is
        essential when ``align_tensors`` splits the original loop variable.
        """
        coeff = sympy.S.One
        body = term
        if term.func == sympy.Mul and term.args[0].is_rational:
            coeff, body = term.args
            # TODO: handle non-unit fractions
            # https://github.com/torch-spyre/torch-spyre/issues/1353
            assert coeff.numerator == 1 or coeff.denominator == 1, (
                f"Unsupported coordinate expression {term}"
            )

        divisor = sympy.S.One
        modulus = var_range
        if body == var:
            pass
        elif isinstance(body, FloorDiv) and body.args[0] == var:
            divisor = body.args[1]
        elif body.func == sympy.Mod:
            base, radix = body.args
            if base == var:
                modulus = radix
            elif isinstance(base, FloorDiv) and base.args[0] == var:
                divisor = base.args[1]
                modulus = divisor * radix
            else:
                raise Unsupported(
                    f"Unsupported modular coordinate expression {body} for {var}"
                )
        else:
            raise Unsupported(f"Unsupported coordinate expression {term}")

        return Term(
            coeff.numerator,
            coeff.denominator * divisor,
            var,
            modulus,
            dim_size,
        )

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
            dim_terms.append(normalize_var_expr(term, var, var_range, dim_size))
        # sort dim_terms in increasing (num, mod) order so that z + offset
        # vars (num=1, mod=1) always sort before real iteration vars (num=1, mod=N)
        # when num is equal
        dim_terms.sort(
            key=lambda t: (
                compare_value(t.num),
                compare_value(t.mod),
            )
        )

        for dim_term in dim_terms[::-1]:
            dim_term.offset = offset // dim_term.num
            offset %= dim_term.num

        # split dims with n>1 terms
        split_dim_terms = []

        cum_size = 1
        # Save original numerators before the loop resets them to 1.
        # dim_terms[i].num is the flat-index step for variable i.  The
        # device-dimension range for variable i equals the ratio of
        # consecutive steps: original_nums[i+1] // original_nums[i].
        # Using dim_terms[i+1].num directly (which has already been reset
        # to 1 for lower terms) would give the next variable's raw step,
        # producing inflated dim_sizes and spurious backGaps when 3+ vars
        # share a single flat device dimension (e.g. ho*96+kh*24+wo*4+kw).
        original_nums: list[sympy.Expr] = [cast(sympy.Expr, t.num) for t in dim_terms]
        # for all terms but the last
        for i in range(0, len(dim_terms) - 1):
            # range of variable i = step[i+1] / step[i]
            dim_terms[i].dim_size = original_nums[i + 1] // original_nums[i]
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
    # Whether a transparent placeholder term was skipped since ``fused_term``
    # was last (re)set.  A placeholder is a size-1 device dim with a constant
    # zero coordinate, e.g. the squeezed ``seq`` dim that
    # ``get_generic_stick_layout`` puts *between* ``H`` and the non-stick half
    # of ``D`` for a ``[B, H, 1, D]`` attention output.  It occupies no space
    # and is discarded by the flush guards below, but because this scan only
    # compares neighbouring list entries it would flush the pending term and
    # break a fusion run between two dims that ``compute_coordinates`` already
    # treats as stride-adjacent (``next_stride`` ignores ``size == 1`` dims).
    # Splitting an otherwise contiguous axis that way makes ``align_tensors``
    # mint an extra loop variable for it, and for a matmul that produces a
    # second reduction dim, which the backend cannot schedule (a bmm contracts
    # exactly one dim; deeptools aborts with ``out_reuse_dim.size() == 1``).
    skipped_placeholder = False
    for term in terms[1:-1]:
        if term.var is None and term.dim_size == 1 and term.offset == 0:
            skipped_placeholder = True
            continue
        if (
            fused_term.num == 1
            and fused_term.var == term.var
            and fused_term.den == term.mod
            # Fusing *across* a placeholder must not change the address that
            # any (dim, coordinate) pair encodes.  It provably does not when
            # the pair is densely packed (``den == term.den * term.dim_size``),
            # the inner term skips no elements (``num == 1``), and the outer
            # term carries no offset -- the outer offset counts in units of the
            # outer ``den`` and is not rescaled when ``den`` shrinks on fusion.
            # Adjacent terms keep the historical predicate unchanged.
            and (
                not skipped_placeholder
                or (
                    term.num == 1
                    and fused_term.offset == 0
                    and fused_term.den == term.den * term.dim_size
                )
            )
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
        skipped_placeholder = False
    if fused_term.dim_size > 1 or fused_term.var is not None:
        fused_terms.append(fused_term)
    # add term for stick dimension
    fused_terms.append(terms[-1])

    return fused_terms


@dataclass(frozen=True)
class AlignmentInputs:
    """Everything tensor alignment needs, captured without hidden graph state."""

    iteration_space: dict[sympy.Symbol, tuple[sympy.Expr, int]]
    tensors: list[dict[str, list[sympy.Expr]]]
    indirect_sizes: dict[sympy.Symbol, int] | None
    repeat_info: dict[sympy.Symbol, dict]
    concrete_ranges: dict[sympy.Symbol, int | float]
    restored_ranges: dict[sympy.Symbol, sympy.Expr | int | float]


def build_alignment_inputs(
    iteration_space: Dict[sympy.Symbol, Tuple[sympy.Expr, int]],
    tensors: list[Dict[str, list[sympy.Expr]]],
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
    repeat_info: "dict[sympy.Symbol, dict] | None" = None,
) -> AlignmentInputs:
    """Snapshot explicit inputs for a repeatable, side-effect-free alignment."""

    from .pass_utils import finite_upper_or_none

    concrete_ranges = {
        var: _concretize_for_cmp(value) for var, (value, _) in iteration_space.items()
    }
    restored_ranges: dict[sympy.Symbol, sympy.Expr | int | float] = {}
    for var, (value, _) in iteration_space.items():
        if (
            hasattr(value, "free_symbols")
            and value.free_symbols
            and finite_upper_or_none(value) is None
        ):
            restored_ranges[var] = concrete_ranges[var]
        else:
            restored_ranges[var] = value

    return AlignmentInputs(
        iteration_space=dict(iteration_space),
        tensors=[
            {
                "size": list(tensor["size"]),
                "coordinates": list(tensor["coordinates"]),
            }
            for tensor in tensors
        ],
        indirect_sizes=dict(indirect_sizes) if indirect_sizes is not None else None,
        repeat_info={
            symbol: dict(info) for symbol, info in (repeat_info or {}).items()
        },
        concrete_ranges=concrete_ranges,
        restored_ranges=restored_ranges,
    )


def _concrete_alignment_value(expr: sympy.Expr) -> int | float:
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        raise Unsupported(f"alignment input is not concrete: {expr}")
    if expr == sympy.oo:
        return math.inf
    if expr == -sympy.oo:
        return -math.inf
    return int(expr)


def align_tensors_pure(
    inputs: AlignmentInputs,
) -> tuple[
    dict[sympy.Symbol, tuple[sympy.Expr, int]],
    list[dict[str, list]],
    dict[sympy.Symbol, tuple[tuple[sympy.Symbol, int], ...]],
]:
    """
    Transform tensor coordinates using only the supplied immutable snapshot.
    """

    # Concretize range values for the algorithm: align_tensors performs
    # sorting, math.gcd, and integer division that require concrete ints.
    # Coordinate *expressions* remain symbolic (they reference loop variable
    # Symbols, not range values).

    iteration_space = inputs.iteration_space
    tensors = inputs.tensors
    indirect_sizes = inputs.indirect_sizes
    repeat_info = inputs.repeat_info
    var_ranges = dict(inputs.concrete_ranges)

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
            _concrete_alignment_value,
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
    work_division_remap = {}
    for var, split in splits.items():
        div = op_it_space_splits[var] if var in op_it_space_splits else 1
        if len(split) > 1:
            new_var_ranges[var] = split[1] // split[0]
            remap[var] = [var]  # reuse variable name for 1st segment
            for i in range(1, len(split) - 1):
                new_var = synthetic_var()  # create new variable
                new_var_ranges[new_var] = split[i + 1] // split[i]
                remap[var].append(new_var)

            bases = {}
            # distribute work division for old var to new vars
            for v in reversed(remap[var]):
                # Re-intersect the committed split against the basis work
                # division used for this var.
                if v == var and v in stick_dim:
                    # Stick var: stick count. The element range would drop a
                    # legal split when the size is not a multiple of it
                    # (e.g. gcd(2, 67) == 1).
                    eps = int(stick_size[stick_dim.index(v)])
                    basis = (int(new_var_ranges[v]) + eps - 1) // eps  # stick count
                else:
                    # Non-stick var (or synthetic sub-dim): element range.
                    basis = new_var_ranges[v]
                bases[v] = int(basis)
                new_op_it_space_splits[v] = math.gcd(div, basis)
                div //= new_op_it_space_splits[v]
            work_division_remap[var] = tuple((v, bases[v]) for v in remap[var])
        else:
            # no splits keep existing var, range, and work division
            # may happen with a single stick since the stick size is omitted
            # downstream passes receive the symbolic expression, not the concretized
            # hint -- unless the symbol is unbounded, see _bounded_or_hint.
            # Synthetic vars (z0, z1, …) are introduced by normalize_coordinates
            # for restored size-1 dims and are not in orig_ranges; fall back to
            # the concretized value (always 1) for those.
            new_var_ranges[var] = inputs.restored_ranges.get(var, var_ranges[var])
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
            work_division_remap[var] = ((var, 1),)
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
    for var, restored_expr in inputs.restored_ranges.items():
        if var not in new_var_ranges or new_var_ranges[var] != var_ranges[var]:
            continue
        new_var_ranges[var] = restored_expr

    # Iteration space should only contain loop variables, not indirect symbols.
    # Filter out any indirect symbols that were added during normalization.
    indirect_syms = set(indirect_sizes.keys()) if indirect_sizes else set()
    new_iteration_space = {
        k: (v, new_op_it_space_splits[k])
        for k, v in new_var_ranges.items()
        if k not in indirect_syms
    }

    return new_iteration_space, new_tensors, work_division_remap


def align_tensors(
    iteration_space: Dict[sympy.Symbol, Tuple[sympy.Expr, int]],
    tensors: list[Dict[str, list[sympy.Expr]]],
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
    repeat_info: "dict[sympy.Symbol, dict] | None" = None,
) -> tuple[
    dict[sympy.Symbol, tuple[sympy.Expr, int]],
    list[dict[str, list]],
    dict[sympy.Symbol, tuple[tuple[sympy.Symbol, int], ...]],
]:
    """Build explicit alignment inputs, then run the pure implementation."""

    return align_tensors_pure(
        build_alignment_inputs(iteration_space, tensors, indirect_sizes, repeat_info)
    )


def tiling_expr_to_device_expr(
    device_size: Sequence[sympy.Expr],
    stride_map: Sequence[sympy.Expr],
    index: sympy.Expr,
) -> sympy.Expr:
    """
    Convert a tile offset expression (index) to a device layout (device_size and
    stride_map)
    """

    assert all(isinstance(s, (int, sympy.Integer)) for s in device_size), (
        f"tiling_expr_to_device_expr requires a concrete device_size, got {device_size}"
    )
    assert all(isinstance(s, (int, sympy.Integer)) for s in stride_map), (
        f"tiling_expr_to_device_expr requires a concrete stride_map, got {stride_map}"
    )

    out = sympy.S.Zero
    n = len(stride_map)
    vars = index.free_symbols
    terms = index.args if isinstance(index, sympy.Add) else (index,)
    for var in vars:
        # step must be var's own coefficient, not index's value at var=1 --
        # those only coincide when index has zero constant term. `index` can
        # legitimately carry one here: _general_tile_advance builds it from
        # dep.index, which bakes in literal offsets from Python-level slicing
        # (e.g. key[..., start:end, :] for KV-block >= 1 contributes a
        # constant +start*row_stride term alongside the tiled-dim symbol).
        # Evaluating at var=1 folded that unrelated constant straight into
        # the per-level advance coefficient (issue: S=128 flash-attention,
        # second head-tile group's second KV block reading the wrong head).
        # Isolate var's own additive term first (mirrors coeff_through_floor
        # in pass_utils.py -- not reused directly to avoid a views<->pass_utils
        # import cycle), then take its coefficient, looking through one
        # floor() layer since a term can be floor(k*var/d).
        own_term = next((t for t in terms if var in t.free_symbols), sympy.S.Zero)
        if isinstance(own_term, sympy.floor):
            step = own_term.args[0].coeff(var)
        else:
            step = own_term.coeff(var)
        j = -1  # device dimension for var
        for i in range(n):
            if (
                device_size[i] > 1
                and stride_map[i] > (stride_map[j] if j != -1 else 0)
                and stride_map[i] <= step
            ):
                j = i
        out += var * math.prod(device_size[j + 1 : n]) * step // stride_map[j]
    return out
