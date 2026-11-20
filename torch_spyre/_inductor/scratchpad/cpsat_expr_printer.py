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

"""Lowering of a sympy cost expression (``cost_model``) into an OR-Tools CP-SAT
objective, kept apart from the layout model in ``ilp_solver_ortools`` that
calls it.

:func:`cost_expr_to_cpsat` reads the solver only through the two maps it is
given: ``sym_map`` (symbol name -> CP-SAT var, plus ``_division_of_<sym>`` ->
the division-choosing buffer wrapper) and ``buffer_map`` (split or tile symbol
name -> ``(wrapper, candidate values)``). A wrapper supplies ``buffer``,
``division`` and ``division_is``.
"""

from __future__ import annotations

from collections import Counter
import logging
import math
import operator
from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import sympy
from sympy.printing.printer import Printer

if TYPE_CHECKING:
    from ortools.sat.python import cp_model, cp_model_helper
else:
    try:
        from ortools.sat.python import cp_model, cp_model_helper
    except ImportError:  # pragma: no cover - only when ortools is absent
        cp_model = cp_model_helper = None

__all__ = ["cost_expr_to_cpsat"]

logger = logging.getLogger(__name__)

# constant to scale log of core split. error ~0.5%
_CORE_LOG_SCALE = 32.0
# cap on the scale for the inverse of a core split, and the scale itself when
# the split has no candidate values (see _SympyExprToCpSat._inv_scale).
# error <= ~4%
_CORE_INV_SCALE = 1024
# largest value _CORE_INV_SCALE is sized for: the core count. Past it (a tile
# count can be any divisor of its axis) the cap grows in step, see _inv_scale.
_CORE_INV_SCALE_MAX_VALUE = 32
# constant limit on product terms to avoid int32 overflow in CP-SAT
_MAX_PRODUCT_BOUND = 2**30


_operator_map = {
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _eq_fact(rel):
    """``(lhs, number)`` for ``Eq``/``Ne`` with a numeric side, else None."""
    lhs, rhs = rel.args
    if isinstance(rhs, sympy.Number):
        return lhs, rhs
    if isinstance(lhs, sympy.Number):
        return rhs, lhs
    return None


def _simplify_eq_conjunction(expr):
    """Cheap stand-in for ``sympy.simplify`` on a conjunction: resolve its
    ``Eq(x, c)`` terms against each other and against ``Ne(x, c')`` terms,
    e.g. ``And(Eq(x, 2), Ne(x, 4)) -> Eq(x, 2)`` and
    ``And(Eq(x, 2), Eq(x, 4)) -> false``. Anything else is kept as is."""
    if not isinstance(expr, sympy.And):
        return expr
    eqs: dict = {}
    for arg in expr.args:
        if isinstance(arg, sympy.Eq) and (fact := _eq_fact(arg)) is not None:
            lhs, val = fact
            if eqs.setdefault(lhs, val) != val:
                return sympy.false
    if not eqs:
        return expr
    kept = []
    for arg in expr.args:
        if isinstance(arg, sympy.Ne) and (fact := _eq_fact(arg)) is not None:
            lhs, val = fact
            if lhs in eqs:
                if eqs[lhs] == val:
                    return sympy.false
                continue  # implied by Eq(lhs, eqs[lhs])
        kept.append(arg)
    return sympy.And(*kept)


_inv_rel_op = {
    sympy.Eq: sympy.Eq,
    sympy.Ge: sympy.Le,
    sympy.Le: sympy.Ge,
    sympy.Gt: sympy.Lt,
    sympy.Lt: sympy.Gt,
}


class _LazyMin(sympy.Min):
    """``Min`` built without canonicalization. An evaluated ``Min``/``Max`` runs
    sympy's pairwise dominance check on its arguments (``_find_localzeros``),
    which for the split-symbol expressions of the cost objective goes through
    the assumptions system (``is_ge`` -> ``_monotonic_sign`` -> ``factor_terms``);
    every rewrite pass below rebuilds these nodes, so on the 304-op spyre_attn
    decode graph that check was ~60% of the planner's Python time (59.6 s of
    the rewrite passes; 2.0 s with these classes). The printer lowers a
    ``Min``/``Max`` structurally and never needs the canonical form."""

    def __new__(cls, *args, **kwargs):
        kwargs["evaluate"] = False
        return super().__new__(cls, *args, **kwargs)


class _LazyMax(sympy.Max):
    """``Max`` counterpart of :class:`_LazyMin`."""

    def __new__(cls, *args, **kwargs):
        kwargs["evaluate"] = False
        return super().__new__(cls, *args, **kwargs)


_LAZY = {sympy.Min: _LazyMin, sympy.Max: _LazyMax}


def _lazy_minmax(expr: sympy.Expr) -> sympy.Expr:
    """Rebuild every ``Min``/``Max`` node of ``expr`` as its lazy counterpart.

    A hand-rolled bottom-up rebuild rather than ``Basic.replace``: ``replace``
    reconstructs a node whose children changed with its ORIGINAL class before
    applying the substitution, so a ``Min`` nested inside a ``Max`` was
    canonicalized once more on the way up (13 s on the 304-op graph's
    objective). Here a ``Min``/``Max`` is built lazy in the first place and
    every other node is rebuilt only when a child actually changed."""
    if not expr.args:
        return expr
    args = [_lazy_minmax(a) for a in expr.args]
    lazy = _LAZY.get(type(expr))
    if lazy is not None:
        return lazy(*args)
    if all(a is b for a, b in zip(args, expr.args)):
        return expr
    return expr.func(*args)


class memoize_method:
    """Descriptor that memoizes a method separately for each instance.

    The cache lives in the instance ``__dict__``, so it is freed with the
    instance and never shared between instances.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        bound_cache = cache(self.func.__get__(instance, owner))
        # Shadow the descriptor so later lookups hit the cached method directly.
        instance.__dict__[self.func.__name__] = bound_cache
        return bound_cache


class _SympyExprToCpSat(Printer):
    """Translates a sympy cost expression into an OR-Tools CP-SAT expression
    over an existing ``sympy symbol -> CP-SAT var`` mapping.
    """

    def __init__(
        self,
        model: "cp_model.CpModel",
        sym_map: dict,
        buffer_map: dict,
    ) -> None:
        self._model = model
        self._count = 0
        self._sym_map = sym_map
        self._buffer_map = buffer_map
        self._allow_tables = True
        # These caches belong to one converter/model. A class-level @cache
        # would retain every converter and its CP model across compilations.
        self._expr_cache: dict = {}
        self._untabled_expr_cache: dict = {}
        self._conjunction_cache: dict = {}
        self._candidate_cache: dict = {}
        self._division_table_cache: dict = {}
        super().__init__()

    def _print(self, expr, **kwargs):
        if expr not in self._expr_cache:
            table = (
                self._division_table(expr) if self._allow_tables and expr.args else None
            )
            self._expr_cache[expr] = (
                super()._print(expr, **kwargs) if table is None else table
            )
        return self._expr_cache[expr]

    def _candidate_values(self, expr):
        """Evaluate an expression over at most one buffer's division menu.

        Residency and other free decisions stop evaluation at that node. Work
        on the rewritten expression, including its reciprocal/log scales and
        Min/Max rounding, so tabulation does not change the cost model.
        """
        if expr not in self._candidate_cache:
            self._candidate_cache[expr] = self._evaluate_candidates(expr)
        return self._candidate_cache[expr]

    def _evaluate_candidates(self, expr):
        if expr in (sympy.true, sympy.false):
            return None, bool(expr)
        if expr.is_Number:
            return None, int(expr) if expr.is_Integer else float(expr)
        if expr.is_Symbol:
            mapped = self._sym_map.get(expr.name)
            if isinstance(mapped, (int, float)):
                return None, mapped
            name = expr.name
            prefix = next((p for p in ("inv_", "log2_") if name.startswith(p)), "")
            raw_name = name[len(prefix) :]
            entry = self._buffer_map.get(raw_name)
            if entry is None:
                wrapper = self._sym_map.get(f"_division_of_{name}")
                if wrapper is None:
                    return None
                entry = wrapper, range(len(wrapper.buffer.core_divisions))
            wrapper, raw = entry
            if wrapper is None:
                # Domain-only entries can set reciprocal scales without a
                # shared division selector. Keep their arithmetic symbolic;
                # their candidate arrays are not scalar constants or a table.
                return None
            if prefix == "inv_":
                scale = self._inv_scale(raw_name)
                raw = [scale // v for v in raw]
            elif prefix == "log2_":
                raw = [int(round(_CORE_LOG_SCALE * math.log2(v))) for v in raw]
            values = np.asarray(raw, dtype=object)
            if all(v == values[0] for v in values):
                return None, values[0]
            return wrapper, values

        if expr.is_Piecewise and expr.args[-1][1] != sympy.true:
            return None  # Preserve the ordinary printer's required default.
        args = (
            [part for pair in expr.args for part in pair]
            if expr.is_Piecewise
            else expr.args
        )
        evaluated = [self._candidate_values(arg) for arg in args]
        if any(value is None for value in evaluated):
            return None
        owners = [value[0] for value in evaluated if value[0] is not None]
        owner = owners[0] if owners else None
        if any(other is not owner for other in owners):
            return None
        values = [value[1] for value in evaluated]
        if expr.is_Add:
            result = sum(values)
        elif expr.is_Mul:
            result = math.prod(values)
        elif expr.is_Pow:
            result = values[0] ** values[1]
        elif isinstance(expr, (sympy.Min, sympy.Max)):
            operation = np.minimum if isinstance(expr, sympy.Min) else np.maximum
            result = values[0]
            for value in values[1:]:
                result = operation(result, value)
        elif expr.is_Relational:
            result = _operator_map[expr.rel_op](*values)
        elif expr.func in (sympy.And, sympy.Or):
            operation = np.logical_and if expr.func is sympy.And else np.logical_or
            result = values[0]
            for value in values[1:]:
                result = operation(result, value)
        elif expr.is_Piecewise:
            result = values[-2]
            for value, cond in reversed(list(zip(values[:-2:2], values[1:-2:2]))):
                result = np.where(cond, value, result)
        elif expr.func is sympy.KroneckerDelta:
            result = np.equal(*values)
        else:
            return None
        return owner, result

    def _division_table(self, expr):
        evaluated = self._candidate_values(expr)
        if evaluated is None:
            return None
        owner, values = evaluated
        if owner is None:
            return np.asarray(values).item()
        raw = np.asarray(values, dtype=object).reshape(-1).tolist()
        # An exact affine encoding of the binary floating values, without
        # introducing another rounding scale. If its integer range is too
        # large, lower the expression normally and try its smaller children.
        if not all(math.isfinite(v) for v in raw):
            return None
        rationals = [Fraction(v) for v in raw]
        base = min(rationals)
        denominator = math.lcm(*(v.denominator for v in rationals))
        differences = [int((v - base) * denominator) for v in rationals]
        divisor = math.gcd(*differences)
        if divisor == 0:
            return raw[0]
        table = tuple(v // divisor for v in differences)
        if max(table) > _MAX_PRODUCT_BOUND:
            return None
        scale = Fraction(divisor, denominator)
        offset = int(base) if base.denominator == 1 else float(base)
        coefficient = int(scale) if scale.denominator == 1 else float(scale)
        if Fraction(offset) != base or Fraction(coefficient) != scale:
            return None
        key = owner.division.index, table
        if key not in self._division_table_cache:
            var = self._model.new_int_var_from_domain(
                cp_model.Domain.FromValues(table), f"division_cost_{self._count}"
            )
            self._count += 1
            self._model.add_element(owner.division, table, var)
            self._division_table_cache[key] = var
        return coefficient * self._division_table_cache[key] + offset

    def convert(self, cost_expr: sympy.Expr) -> "cp_model.LinearExpr":
        """Return the CP-SAT expression equivalent to ``cost_expr`` under
        ``sym_map`` (``sympy symbol -> CP-SAT var``)."""
        logger.debug("[CP-SAT layout solver] cost expr (raw): %s", cost_expr)
        cost_expr = self._rewrite(cost_expr)
        logger.debug("[CP-SAT layout solver] cost expr (linearized): %s", cost_expr)
        return self._print(cost_expr)

    def _rewrite(self, cost_expr: sympy.Expr) -> sympy.Expr:
        """The symbolic rewrites that bring ``cost_expr`` into the form the
        printer lowers: floors dropped, logs of Min/Max pushed inside, split
        logs and inverses replaced by table symbols, scalars pushed into
        Min/Max/Piecewise, floats truncated. Min/Max are rebuilt lazily first
        (see :class:`_LazyMin`)."""
        cost_expr = _lazy_minmax(cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.floor,
            lambda e: e.args[0],
        )
        cost_expr = sympy.expand(cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func in [sympy.log, sympy.Piecewise],
            lambda e: self._piecewise_canonical(self._log_min(e)),
        )
        cost_expr = sympy.expand(cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func in [sympy.log, sympy.Pow, sympy.Mul],
            self._inv_log_sym,
        )
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.Mul,
            lambda e: self._min_piecewise_expand(e),
        )
        cost_expr = self._integerize_minmax(cost_expr)
        return cost_expr

    @classmethod
    def _log_min(cls, expr):
        # rewrite log(min(a, b)) as min(log(a), log(b))
        if expr.func is not sympy.log:
            return expr
        arg = expr.args[0]
        if isinstance(arg, (sympy.Min, sympy.Max)):
            # n() here is to get a numeric value instead of log(2)
            return arg.func(*[sympy.log(a.n()) for a in arg.args])
        if (
            isinstance(arg, sympy.Mul)
            and len(arg.args) == 2
            and isinstance(arg.args[0], sympy.Number)
            and isinstance(arg.args[1], (sympy.Min, sympy.Max))
        ):
            return arg.func(
                *[sympy.log((a * arg.args[0]).n()) for a in arg.args[1].args]
            )
        else:
            return expr

    @staticmethod
    def _is_split_sym(expr):
        # A tile count is, like a core split, a positive integer tabulated over
        # its buffer's candidate divisions, so its log and inverse lower the
        # same way.
        return expr.is_Symbol and expr.name.startswith(("split_", "tile_"))

    def _inv_scale(self, name: str) -> int:
        """Fixed-point scale of ``inv_<name>``: the LCM of ``name``'s values
        across the candidate divisions, so every ``scale // v`` is exact and the
        variable spans only the bits it needs. ``_CORE_INV_SCALE`` when there
        are no values, or a value with many divisors if the LCM exceeds the cap.

        The cap is ``_CORE_INV_SCALE`` for values up to the core count and
        doubles with each doubling past it, so a tile count, which can far
        exceed any core split, keeps the resolution a split has rather than
        flattening to ``scale // v == 0``."""
        _, raw = self._buffer_map.get(name, (None, ()))
        if not raw or min(raw) < 1:
            return _CORE_INV_SCALE

        ints = [int(r) for r in raw]
        widen = max(
            0, (max(ints) - 1).bit_length() - _CORE_INV_SCALE_MAX_VALUE.bit_length() + 1
        )
        cap = _CORE_INV_SCALE << widen
        lcm = math.lcm(*ints)
        if lcm <= cap:
            return lcm

        # Otherwise: find the highest power of 2 in ints; among its multiples,
        # find which has the most entries of ints as divisors, and use that.
        # (We weight entries of ints by multiplicity.)
        cnt = Counter(ints)
        pow2 = max((a for a in cnt if a & (a - 1) == 0), default=1)
        # The cap is sized past every value, so this holds by construction.
        assert pow2 < cap, f"expected cap={cap} past every value, found {pow2}"
        scaled_core_inv_scale = cap // pow2
        counts = np.zeros(scaled_core_inv_scale + 1, dtype=np.int64)
        for a, mult in cnt.items():
            step = a // math.gcd(a, pow2)
            if step <= scaled_core_inv_scale:
                counts[step::step] += mult

        # Among those, the one whose worst-rounded entry rounds best: the error
        # of an inexact v is (scale % v) / scale, so width is what buys
        # accuracy. A worst case does not accumulate, so unlike the count above
        # this weighs each value once. Ties -- frequent, since a scale and its
        # multiples often round alike -- go to the narrowest, keeping the
        # products in _print_multiply clear of _MAX_PRODUCT_BOUND.
        candidates = pow2 * (np.flatnonzero(counts[1:] == counts[1:].max()) + 1)
        values = sorted(cnt)
        return min(
            (int(c) for c in candidates),
            key=lambda s: (Fraction(max(s % v for v in values), s), s),
        )

    def _inv_log_sym(self, expr):
        # replaces log(sym) with log2_sym and 1/sym with inv_sym
        arg = expr.args[0]
        if expr.func == sympy.log:
            if self._is_split_sym(arg):
                return (
                    sympy.Symbol(f"log2_{arg.name}", integer=True, nonnegative=True)
                    * sympy.log(2.0)
                    / _CORE_LOG_SCALE
                )
            elif arg.is_Number:
                return math.log(float(arg))
        elif expr.func == sympy.Pow:
            if not self._is_split_sym(arg):
                return expr
            # Core splits only: a tile count can exceed 32.
            if (
                expr.exp.is_Number
                and math.isclose(expr.exp, 0.25)
                and arg.name.startswith("split_")
            ):
                # Discrete piecewise linear approximation of x^(1/4) over the interval [1, 32],
                # pinned to return 1 at x=1, generated using tools/approximate-power.py.
                # since we check that the base is a _split_ symbol, it is an integer in the
                # range [1, 32]
                # Max 4.5% deviation for two segments;
                # we would get:
                #   max 1.6% deviation with 3 segments;
                #   max 0.074% deviation with 8 segments;
                #   max 0.019% deviation with 12 segments;
                #   no deviation with 16 segments.
                # script at https://github.com/user-attachments/files/31975789/approximate-power.py
                return sympy.Piecewise(
                    (0.139980295504224 * arg + 0.860019704495776, arg <= 5),
                    (0.0287191888771944 * arg + 1.45940018593522, True),
                )
            if expr.exp == -1:
                return sympy.Symbol(
                    f"inv_{arg.name}", integer=True, nonnegative=True
                ) / self._inv_scale(arg.name)
        elif expr.func == sympy.Mul:
            symbols = [
                arg
                for arg in expr.args
                if arg.is_Symbol and arg.name.startswith("inv_")
            ]
            if len(symbols) <= 2:
                return expr
            product = "_product_" + "_".join(
                sorted([symbol.name[4:] for symbol in symbols])
            )
            if product in self._sym_map:
                # The coefficient already divides by each factor's own scale;
                # swap those for the product's scale to keep the magnitude.
                result = sympy.Symbol(f"inv_{product}", integer=True, nonnegative=True)
                result *= sympy.Rational(
                    math.prod(self._inv_scale(s.name[4:]) for s in symbols),
                    self._inv_scale(product),
                )
                result *= math.prod([arg for arg in expr.args if arg not in symbols])
                return result
        return expr

    @classmethod
    def _piecewise_canonical(cls, expr):
        # re-write 1/x < 1/5 as x > 5, then tighten to an equivalent integer
        # bound (e.g. x < 4/3 as x <= 1) when x is integer-valued and the
        # bound is numeric.
        if not expr.is_Piecewise:
            return expr
        args = []
        for value, cond in expr.args:
            if (
                cond.is_Relational
                and cond.lhs.is_Pow
                and cond.lhs.exp == -1
                and cond.lhs.base.is_Symbol
                and cond.lhs.base.is_nonnegative
            ):
                args.append(
                    (
                        value,
                        cls._tighten_integer_bound(
                            _inv_rel_op[cond.func], 1 / cond.lhs, 1 / cond.rhs
                        ),
                    )
                )
            else:
                args.append((value, cond))
        return expr.func(*args)

    @staticmethod
    def _tighten_integer_bound(rel_op, lhs, rhs):
        if not (lhs.is_Symbol and lhs.is_integer and rhs.is_Number):
            return rel_op(lhs, rhs)
        if rel_op is sympy.Lt:
            return sympy.Le(lhs, sympy.ceiling(rhs) - 1)
        if rel_op is sympy.Gt:
            return sympy.Ge(lhs, sympy.floor(rhs) + 1)
        if rel_op is sympy.Le:
            return sympy.Le(lhs, sympy.floor(rhs))
        if rel_op is sympy.Ge:
            return sympy.Ge(lhs, sympy.ceiling(rhs))
        return rel_op(lhs, rhs)

    @staticmethod
    def _min_piecewise_expand(expr):
        # re-writes 2.1*Min(x, y) as Min(2.1*x, 2.1*y)
        # len(expr.args) may exceed 2 when extra scalar factors ride alongside
        # the leading number and the (single) Min/Max/Piecewise factor this
        # rewrites, e.g. 2.1*a*Min(x, y); only the first such factor is expanded
        # per call, with the rest folded back in as a plain multiplier.
        if len(expr.args) < 2 or not isinstance(expr.args[0], sympy.Number):
            return expr
        if any(
            isinstance(arg, (sympy.Min, sympy.Max, sympy.Piecewise))
            for arg in expr.args[1:]
        ):
            idx, arg = next(
                (
                    (idx, a)
                    for idx, a in enumerate(expr.args)
                    if isinstance(a, (sympy.Min, sympy.Max, sympy.Piecewise))
                )
            )
            m = expr.args[0]

            def apply(arg):
                if isinstance(arg, (tuple, sympy.Tuple)):
                    return (apply(arg[0]), *arg[1:])
                else:
                    new_arg = arg * abs(m)
                    return new_arg.replace(
                        lambda e: e.func == sympy.Mul,
                        lambda e: _SympyExprToCpSat._min_piecewise_expand(e),
                    )

            new_args = [apply(a) for a in arg.args]
            return (
                arg.func(*new_args)
                * sympy.sign(m)
                * sympy.Mul(*(expr.args[1:idx] + expr.args[idx + 1 :]))
            )
        return expr

    @classmethod
    def _integerize_minmax(cls, expr):
        # Only Min/Max constraints require integer operands. A conditional
        # objective supports float values directly; rounding its coefficients
        # can erase a large cost multiplied by scaled reciprocal variables.
        # Keep upstream's lazy Min/Max classes when rebuilding their subtrees.
        if isinstance(expr, (sympy.Min, sympy.Max)):
            return expr.replace(
                lambda e: isinstance(e, (sympy.Min, sympy.Max, sympy.Piecewise)),
                cls._truncate_floats_min,
            )
        if not expr.has(sympy.Min, sympy.Max):
            return expr
        return expr.func(*(cls._integerize_minmax(arg) for arg in expr.args))

    @staticmethod
    def _truncate_floats_min(expr):
        # re-writes Min(x*0.5, y*0.5) as Min(x, y)/2
        m = 10000
        func = expr.func

        def _process_inner(expr):
            if isinstance(expr, sympy.Mul) and isinstance(expr.args[0], sympy.Number):
                a = (expr.args[0] * m).round()
                r = sympy.Mul(a, *expr.args[1:])
            elif isinstance(expr, sympy.Number):
                r = (expr * m).round()
            else:
                r = expr * m
            return r

        def _process_outer(expr):
            if isinstance(expr, sympy.Add):
                return sympy.Add(*[_process_inner(a) for a in expr.args])
            elif isinstance(expr, sympy.Tuple):
                return (_process_outer(expr[0]), *expr[1:])
            else:
                return _process_inner(expr)

        result = list(map(_process_outer, expr.args))

        return func(*result) / m

    def _print_Integer(self, expr):
        return int(expr.p)

    def _print_Number(self, expr):
        return float(expr)

    def _print_Add(self, expr):
        return sum(self._print(arg) for arg in expr.args)

    def _print_Mul(self, expr):
        args = [self._print(arg) for arg in expr.args]
        try:
            return self._print_multiply(args)
        except ValueError:
            if not self._allow_tables:
                raise
            # A table can combine an affine sum into one wider integer, whose
            # product exceeds the bound even though each original term fits.
            # Keep the original factorization locally. Unused table helpers
            # are removed by presolve; other expressions can still tabulate.
            # Reuse this converter's variable namespace and guard caches.
            self._allow_tables = False
            self._expr_cache, self._untabled_expr_cache = (
                self._untabled_expr_cache,
                self._expr_cache,
            )
            try:
                return self._print(expr)
            finally:
                self._expr_cache, self._untabled_expr_cache = (
                    self._untabled_expr_cache,
                    self._expr_cache,
                )
                self._allow_tables = True

    def _print_multiply_two(self, a, b):
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            return a * b
        if isinstance(a, cp_model_helper.NotBooleanVariable):
            return b - self._print_multiply_two(a.Not(), b)
        if isinstance(b, cp_model_helper.NotBooleanVariable):
            return self._print_multiply_two(b, a)
        if isinstance(a, cp_model.IntVar) and isinstance(b, cp_model.IntVar):
            return self._print_multiply([a, b])
        if isinstance(a, (cp_model_helper.IntAffine, cp_model_helper.FloatAffine)):
            return (
                a.coefficient * self._print_multiply_two(a.expression, b) + a.offset * b
            )
        if isinstance(b, (cp_model_helper.IntAffine, cp_model_helper.FloatAffine)):
            return self._print_multiply_two(b, a)
        if hasattr(a, "num_exprs"):
            try:
                flat = cp_model.FlatIntExpr(a)
            except TypeError:
                flat = cp_model.FlatFloatExpr(a)
            result = flat.offset * b
            for var, c in zip(flat.vars, flat.coeffs):
                result = result + c * self._print_multiply_two(var, b)
            return result
        if hasattr(b, "num_exprs"):
            return self._print_multiply_two(b, a)
        raise NotImplementedError(f"multiplying {type(a)} by {type(b)}")

    def _print_multiply(self, args):
        ints = [arg for arg in args if isinstance(arg, cp_model.IntVar)]
        nonints = [arg for arg in args if not isinstance(arg, cp_model.IntVar)]
        if len(ints) == 1:
            return self._print_multiply_two(math.prod(nonints), ints[0])
        elif len(ints) == 0:
            return math.prod(nonints)

        # Names need not be unique: in particular, both Boolean constants
        # have an empty name. Cache by the variables' actual model identity.
        name = "_product_" + "_".join(str(arg.index) for arg in ints)
        if name in self._sym_map:
            return self._print_multiply_two(math.prod(nonints), self._sym_map[name])

        if any(arg.is_boolean for arg in ints):
            product = self._print_gated_product(ints, name)
            self._sym_map[name] = product
            return self._print_multiply_two(math.prod(nonints), product)

        # The product is multilinear (degree 1 in each factor), so its
        # extrema over the box of bounds occur at the box's vertices. Rather
        # than enumerating all 2**len(ints) vertices, fold the bounds
        # pairwise: at each step the running [lb, ub] is the exact image of
        # the partial product over its factors (a continuous function over a
        # connected box), so it can be treated as one more independent
        # interval factor and combined via standard interval multiplication.
        (lb, ub), *rest = [self._affine_bounds(arg) for arg in ints]
        for a, b in rest:
            candidates = (lb * a, lb * b, ub * a, ub * b)
            lb, ub = min(candidates), max(candidates)

        # A product past _MAX_PRODUCT_BOUND needs more dynamic range than the
        # model can carry. Rescaling its factors would trade the overflow for
        # rounding that can zero a small factor such as a split count, so
        # refuse it and let _minimize_cost_expr apply its fallback policy.
        if max(abs(lb), abs(ub)) > _MAX_PRODUCT_BOUND:
            raise ValueError(
                f"product {' * '.join(arg.name for arg in ints)} spans "
                f"[{lb}, {ub}], past the {_MAX_PRODUCT_BOUND} CP-SAT bound"
            )

        product = self._model.new_int_var(int(lb), int(ub), name)
        self._model.add_multiplication_equality(product, ints)
        self._sym_map[name] = product
        return self._print_multiply_two(math.prod(nonints), product)

    def _print_gated_product(self, ints, name):
        """A product with boolean factors as the product of the others, gated
        by each boolean: a variable equal to it while the booleans hold and 0
        otherwise, two linear constraints enforced on the literal (as in
        ``_print_RelayoutCharge``), instead of a ``multiplication_equality``
        the solver can only propagate through bounds."""
        gates = [arg for arg in ints if arg.is_boolean]
        rest = [arg for arg in ints if not arg.is_boolean]
        if not rest:
            value, gates = gates[0], gates[1:]
        elif len(rest) == 1:
            value = rest[0]
        else:
            value = self._print_multiply(rest)
        if len(gates) == 1:
            gate = gates[0]
        else:
            gate = self._model.new_bool_var(f"{name}_gate")
            self._model.add_bool_and(gates).only_enforce_if(gate)
            self._model.add_bool_or([g.Not() for g in gates]).only_enforce_if(
                gate.Not()
            )
        lb, ub = self._affine_bounds(value)
        gated = self._model.new_int_var(min(0, int(lb)), max(0, int(ub)), name)
        self._model.add(gated == value).only_enforce_if(gate)
        self._model.add(gated == 0).only_enforce_if(gate.Not())
        return gated

    def _print_Symbol(self, expr):
        if expr.name in self._sym_map:
            return self._sym_map[expr.name]
        if not expr.name.startswith(("log2_", "inv_")):
            raise NotImplementedError(f"not implemented. expr: {expr}")
        name = expr.name[5:] if expr.name.startswith("log2_") else expr.name[4:]
        b, raw = self._buffer_map[name]

        if expr.name.startswith("log2_"):
            values = [int(round(_CORE_LOG_SCALE * math.log2(v))) for v in raw]
            domain = cp_model.Domain.FromValues(values)
            cp_var = self._model.new_int_var_from_domain(domain, expr.name)
            self._model.add_element(b.division, values, cp_var)
        else:
            scale = self._inv_scale(name)
            values = [scale // v for v in raw]
            cp_var = self._model.new_int_var(min(values), max(values), expr.name)
            self._model.add_division_equality(cp_var, scale, self._sym_map[name])
        self._sym_map[expr.name] = cp_var
        return cp_var

    def _print_KroneckerDelta(self, expr):
        """``KroneckerDelta(division_X, k)`` -> the reified literal
        ``division == k`` of buffer X (``_CoreDivisionBufferWithCpVars.
        division_is``), so a table term such as the relayout price lowers to a
        product of literals. sympy canonicalises the argument order, so the
        symbol and the constant are found by type, not position."""
        symbols = [a for a in expr.args if isinstance(a, sympy.Symbol)]
        constants = [a for a in expr.args if isinstance(a, sympy.Integer)]
        if len(symbols) != 1 or len(constants) != 1:
            raise NotImplementedError(f"not implemented. expr: {expr}")
        wrapper = self._sym_map.get(f"_division_of_{symbols[0].name}")
        if wrapper is None:
            raise NotImplementedError(f"no division variable for {symbols[0]}")
        return wrapper.division_is(int(constants[0]))

    def _print_RelayoutCharge(self, expr):
        """``RelayoutCharge(is_lx, division_X, price_0, ...)`` (a relayout
        copy's ``cost_term``) -> one ``element`` lookup of X's division into the
        price table plus a charge variable equal to that price while the copy
        is resident and 0 otherwise: two constraints per copy, against one
        boolean product per (copy, priced division) had the table been written
        as a sum of deltas. Divisions past the table read 0; a source that is
        not a division-choosing buffer of this solve prices at 0, as it can
        never fire (``_constrain_relayout_copies``)."""
        is_lx, division, *prices = expr.args
        table = [int(p) for p in prices]
        lit = self._print(is_lx)
        if division.is_Integer:
            i = int(division)
            return lit * (table[i] if 0 <= i < len(table) else 0)
        wrapper = self._sym_map.get(f"_division_of_{division.name}")
        if wrapper is None:
            return 0
        table += [0] * (len(wrapper.buffer.core_divisions) - len(table))
        table = table[: len(wrapper.buffer.core_divisions)]
        top = max(table, default=0)
        if top <= 0:
            return 0
        name = is_lx.name if is_lx.is_Symbol else str(self._count)
        self._count += 1
        price = self._model.new_int_var(0, top, f"relayout_price_{name}")
        self._model.add_element(wrapper.division, table, price)
        charge = self._model.new_int_var(0, top, f"relayout_charge_{name}")
        self._model.add(charge == price).only_enforce_if(lit)
        self._model.add(charge == 0).only_enforce_if(lit.Not())
        return charge

    def _print_AppliedUndef(self, expr):
        """An implemented function of one buffer's splits (a functor of its work
        division, e.g. ``cost_model._decode_delivery_price``) -> one ``element``
        lookup of that buffer's division into the functor's ``_imp_`` evaluated
        at each of its candidate divisions. The owner is the one
        division-choosing buffer whose split symbols the arguments are written
        in; a functor of no buffer of this solve, or a candidate leaving an
        argument unresolved, prices 0. Memoized, since ``expand`` repeats the
        term once per residency gate."""
        key = f"_imp_{expr}"
        if key in self._sym_map:
            return self._sym_map[key]
        imp = getattr(expr, "_imp_", None)
        if imp is None:
            raise NotImplementedError(f"not implemented. expr: {expr}")
        if not expr.free_symbols:
            return self._print(sympy.sympify(imp(*(int(a) for a in expr.args))))
        owners = {
            id(w): w
            for w, _ in (
                self._buffer_map.get(s.name, (None, None)) for s in expr.free_symbols
            )
            if w is not None
        }
        if len(owners) != 1:
            return 0
        (wrapper,) = owners.values()
        buffer = wrapper.buffer
        if not expr.free_symbols <= set(buffer.sym_core_divs.values()):
            return 0
        table = []
        for cd in buffer.core_divisions:
            subs = {
                sym: sympy.Integer(cd.splits.get(key, 1))
                for key, sym in buffer.sym_core_divs.items()
            }
            args = [a.xreplace(subs) for a in expr.args]
            if not all(a.is_Integer for a in args):
                return 0
            table.append(int(imp(*(int(a) for a in args))))
        if min(table) == max(table):
            return table[0]
        name = f"{expr.func.__name__}_{buffer.sym_division.name}_{self._count}"
        self._count += 1
        var = self._model.new_int_var(min(table), max(table), name)
        self._model.add_element(wrapper.division, table, var)
        self._sym_map[key] = var
        return var

    def _print_Pow(self, expr):
        if expr.exp == 2:
            base = self._print(expr.base)
            return self._print_multiply_two(base, base)
        return self._print(expr.base) ** self._print(expr.exp)

    @memoize_method
    def _print_condition(self, cond):
        cond_expr = self._print(cond)
        if isinstance(cond_expr, (cp_model.IntVar, cp_model_helper.NotBooleanVariable)):
            return cond_expr
        if isinstance(cond_expr, (bool, int)):
            # Fixed divisions turn many symbolic predicates into constants.
            return self._model.new_constant(int(cond_expr))
        if not isinstance(cond, sympy.core.relational.Relational):
            return cond_expr
        not_cond_expr = self._print(sympy.Not(cond))
        var = self._model.new_bool_var(f"cond_{self._count}")
        self._count += 1
        self._model.Add(cond_expr).OnlyEnforceIf(var)
        self._model.Add(not_cond_expr).OnlyEnforceIf(var.Not())
        return var

    def _conjunction(self, lits):
        # Expanded cost terms often have different values under exactly the
        # same branch guard. Share the guard even when the Piecewise differs.
        key = tuple(sorted({lit.index for lit in lits}))
        if key not in self._conjunction_cache:
            var = self._model.new_bool_var(f"and_{self._count}")
            self._count += 1
            self._model.add_bool_and(lits).only_enforce_if(var)
            self._model.add_bool_or([lit.Not() for lit in lits]).only_enforce_if(
                var.Not()
            )
            self._conjunction_cache[key] = var
        return self._conjunction_cache[key]

    @memoize_method
    def _print_And(self, expr):
        lits = [self._print_condition(arg) for arg in expr.args]
        return self._conjunction(lits)

    @memoize_method
    def _print_Or(self, expr):
        lits = [self._print_condition(arg) for arg in expr.args]
        or_var = self._model.new_bool_var(f"or_{self._count}")
        self._count += 1
        self._model.AddBoolOr(lits).OnlyEnforceIf(or_var)
        self._model.AddBoolAnd([lit.Not() for lit in lits]).OnlyEnforceIf(or_var.Not())
        return or_var

    @memoize_method
    def _print_Piecewise(self, expr):
        args = expr.args
        assert args[-1][1] == sympy.true
        result = 0
        none_prev = sympy.true  # no earlier condition held
        for val, cond in args:
            lit = _simplify_eq_conjunction(sympy.And(none_prev, cond))
            if lit == sympy.false:
                continue
            if lit == sympy.true:
                result += self._print(val)
                break
            piecewise_var = self._print_condition(lit)
            result += self._print_multiply_two(piecewise_var, self._print(val))
            none_prev = sympy.And(none_prev, sympy.Not(cond))
        return result

    def _print_Not(self, expr):
        return self._print_condition(expr.args[0]).Not()

    def _print_Relational(self, expr):
        return _operator_map[expr.rel_op](*[self._print(arg) for arg in expr.args])

    def _print_log(self, expr):
        if isinstance(expr.args[0], sympy.Number):
            return math.log(float(expr.args[0]))
        raise NotImplementedError(f"log not implemented. expr: {expr}")

    @staticmethod
    def _affine_bounds(expr):
        if isinstance(expr, cp_model.IntVar):
            lb, ub = expr.domain.min(), expr.domain.max()
        elif isinstance(expr, (int, float)):
            lb, ub = expr, expr
        elif isinstance(expr, cp_model_helper.IntAffine):
            lb, ub = _SympyExprToCpSat._affine_bounds(expr.expression)
            c, o = int(expr.coefficient), int(expr.offset)
            lb, ub = (c * lb + o, c * ub + o) if c >= 0 else (c * ub + o, c * lb + o)
        elif isinstance(expr, cp_model.LinearExpr):
            # Flatten sums and folded constants (e.g. ``0 * is_lx``) to an
            # offset + per-var coefficients and bound each term.
            flat = cp_model.FlatIntExpr(expr)
            lb = ub = int(flat.offset)
            for var, c in zip(flat.vars, flat.coeffs):
                c = int(c)
                vlb, vub = _SympyExprToCpSat._affine_bounds(var)
                if c >= 0:
                    vlb, vub = c * vlb, c * vub
                else:
                    vlb, vub = c * vub, c * vlb
                lb, ub = lb + vlb, ub + vub
                assert lb <= ub
        else:
            raise TypeError(f"unsupported expr type: {type(expr)}")

        lb, ub = int(lb), int(ub)
        assert lb <= ub
        return lb, ub

    def _lin_max_operand(self, arg):
        """``arg`` as a ``lin_max`` operand: a constant or a single (affine)
        variable as is, a sum over several variables behind its own IntVar
        tied to it by a linear equality.

        Presolve reasons about a ``lin_max`` operand through its exact
        reachable domain. For a weighted sum of Booleans -- the HBM read and
        write totals behind the cost model's ``alpha * min(R, W)`` turnaround
        term sum ``bytes * (1 - is_lx)`` over a bundle's arguments -- that is
        the set of its subset sums, exponential in the number of distinct
        coefficients. On a Granite 4.0 decode block it was 4 s of
        ``PresolveToFixPoint`` (99% of the solve, on 1209 constraints) that
        neither probing, symmetry nor presolve-iteration limits shorten, and
        with presolve off it made the LNS
        workers, whose neighbourhood solves presolve, run out of memory. Behind
        an IntVar with interval bounds the same operand costs nothing and the
        optimum is unchanged. Float-coefficient operands pass through as
        before (``AddMaxEquality`` rejects them and ``_minimize_cost_expr``
        falls back)."""
        if isinstance(arg, (int, float)):
            return arg
        try:
            if len(cp_model.FlatIntExpr(arg).vars) <= 1:
                return arg
        except TypeError:
            return arg
        var = self._model.new_int_var(
            *self._affine_bounds(arg), f"minmax_arg_{self._count}"
        )
        self._count += 1
        self._model.add(var == arg)
        return var

    def _print_Max(self, expr):
        # max range is (max(mins), max(maxes))
        args = [self._print(arg) for arg in expr.args]
        if all(isinstance(a, (int, float)) for a in args):
            return max(args)  # a lazy Max of constants was never folded
        args = [self._lin_max_operand(arg) for arg in args]
        bounds = map(max, zip(*[self._affine_bounds(arg) for arg in args]))
        max_var = self._model.new_int_var(*bounds, f"max_var_{self._count}")
        self._model.AddMaxEquality(max_var, args)
        self._count += 1
        return max_var

    def _print_Min(self, expr):
        # min range is (min(mins), min(maxes))
        args = [self._print(arg) for arg in expr.args]
        if all(isinstance(a, (int, float)) for a in args):
            return min(args)  # a lazy Min of constants was never folded
        args = [self._lin_max_operand(arg) for arg in args]
        bounds = map(min, zip(*[self._affine_bounds(arg) for arg in args]))
        min_var = self._model.new_int_var(*bounds, f"min_var_{self._count}")
        self._model.AddMinEquality(min_var, args)
        self._count += 1
        return min_var


def cost_expr_to_cpsat(
    model: "cp_model.CpModel", sym_map: dict, buffer_map: dict, cost_expr: sympy.Expr
) -> "cp_model.LinearExpr":
    """``cost_expr`` as a CP-SAT expression over ``model``; the variables and
    constraints the lowering needs are added to ``model``, and ``sym_map`` gains
    the helper variables it creates."""
    return _SympyExprToCpSat(model, sym_map, buffer_map).convert(cost_expr)
