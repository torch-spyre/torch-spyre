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
from sympy.logic.boolalg import ITE
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
# Relative rounding error allowed when a curve's values, or a Min/Max's
# coefficients, are put on an integer scale.
_QUANTIZE_TOLERANCE = 0.01
# Largest scale a Min/Max that a variable multiplies may take to meet
# _QUANTIZE_TOLERANCE.
_MAX_MINMAX_SCALE = 2**14
# Relative rounding error allowed for the numbers of a Min/Max no variable
# multiplies, and the largest scale it may take to meet it: a wide scale costs
# it no product bound.
_MINMAX_TOLERANCE = 1e-5
_MAX_EXACT_MINMAX_SCALE = 2**40
# Largest a Min/Max's scaled coefficient may grow, leaving CP-SAT's 64-bit
# linear sums room for its variables' ranges.
_MAX_MINMAX_COEFFICIENT = 2**40
# Largest a Min/Max's scaled operand may grow over its variables' ranges: CP-SAT
# takes a variable's domain within half of int64.
_MAX_MINMAX_OPERAND = 2**60
# Resolution such a Min/Max starts from: the power of two nearest the old fixed
# 10000, which some objectives already fill near CP-SAT's 64-bit range.
_MIN_FLOAT_MINMAX_SCALE = 2**13


_operator_map = {
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


def power_of_two_scale(values, tolerance: float, limit: int, base: int = 1):
    """The smallest ``base * 2**k`` up to ``limit`` at which rounding
    ``scale * v`` to an integer misses every nonzero ``v`` by at most
    ``tolerance``, relative; None when no such scale is within ``limit``."""
    scale = base
    while scale <= limit:
        if all(
            abs(round(scale * v) - scale * v) <= tolerance * abs(scale * v)
            for v in values
            if v
        ):
            return scale
        scale *= 2
    return None


def round_scaled(number: sympy.Number, scale: int) -> sympy.Integer:
    """``number * scale`` rounded to the nearest integer (ties to even): exact
    for a rational, in double precision for a float."""
    if number.is_Rational:
        return sympy.Integer(round(Fraction(int(number.p), int(number.q)) * scale))
    return sympy.Integer(round(float(number) * scale))


class _CurveTable(sympy.Function):
    """``table[i]`` where ``arg == values[i]``: a fractional power over one
    buffer's menu, at each value its argument takes. Its entries are the
    power's floats until quantized, once, where it is printed: as integers on
    a Min/Max's scale (``_scale_node``) or on its own
    (``_print__CurveTable``). An integer table prints through
    ``_division_table`` as one element lookup."""

    nargs = 3
    # A number, like any value; its Tuple arguments would otherwise leave
    # sympy unable to tell, and a coefficient would not distribute over it.
    is_commutative = True

    @classmethod
    def eval(cls, arg, values, table):
        return None


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


class _LazyPiecewise(sympy.Piecewise):
    """``Piecewise`` counterpart of :class:`_LazyMin`: built without
    ``Piecewise.eval``, whose argument collapsing compares each condition with
    the earlier ones through the same assumptions queries."""

    def __new__(cls, *args, **kwargs):
        kwargs["evaluate"] = False
        return super().__new__(cls, *args, **kwargs)


_LAZY = {sympy.Min: _LazyMin, sympy.Max: _LazyMax}

# The nodes inside a Min/Max whose numbers _scale_node puts on an integer scale.
_TRUNCATED = (sympy.Min, sympy.Max, sympy.Piecewise, _CurveTable)


def _lazy_form(expr: sympy.Expr, hide) -> sympy.Expr:
    """``expr`` with floors dropped, every ``Min``/``Max``/``Piecewise`` rebuilt
    as its lazy counterpart, and each ``Piecewise`` condition replaced by
    ``hide(condition)``.

    A hand-rolled bottom-up rebuild rather than ``Basic.replace``: ``replace``
    reconstructs a node whose children changed with its ORIGINAL class before
    applying the substitution, so a ``Min`` nested inside a ``Max`` was
    canonicalized once more on the way up (13 s on the 304-op graph's
    objective). Here a node is built lazy in the first place and every other
    node is rebuilt only when a child actually changed."""
    done: dict = {}

    def walk(node):
        if not node.args:
            return node
        if node not in done:
            if node.func is sympy.floor:
                done[node] = walk(node.args[0])
            elif isinstance(node, sympy.Piecewise):
                pairs = ((walk(v), hide(walk(c))) for v, c in node.args)
                done[node] = _LazyPiecewise(*pairs)
            else:
                args = [walk(a) for a in node.args]
                lazy = _LAZY.get(type(node))
                if lazy is not None:
                    done[node] = lazy(*args)
                elif all(a is b for a, b in zip(args, node.args)):
                    done[node] = node
                else:
                    done[node] = node.func(*args)
        return done[node]

    return walk(expr)


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
        # Each Piecewise condition the rewrite hid (see _hide), both ways.
        self._conditions: dict = {}
        self._hidden: dict = {}
        # Each Min/Max, Piecewise or curve that took a scale of its own.
        self._own_scales: dict = {}
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
        if expr in self._conditions:
            return self._candidate_values(self._condition(expr))
        if isinstance(expr, _CurveTable):
            evaluated = self._candidate_values(expr.args[0])
            if evaluated is None:
                return None
            owner, values = evaluated
            lookup = {
                int(x): int(v) if v.is_Integer else float(v)
                for x, v in zip(expr.args[1], expr.args[2])
            }
            if owner is None:
                return None, lookup[int(values)]
            return owner, np.asarray([lookup[int(v)] for v in values], dtype=object)
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
            # As objects: a scaled integer can outgrow int64.
            result = np.asarray(values[0], dtype=object)
            for value in values[1:]:
                result = operation(result, np.asarray(value, dtype=object))
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
        if isinstance(result, np.ndarray) and result.dtype != object:
            # Comparisons and np.where give bool or int64 arrays; kept as
            # objects, a later sum or product can outgrow int64.
            result = result.astype(object)
        return owner, result

    def _division_table(self, expr, bound=_MAX_PRODUCT_BOUND):
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
        if bound is not None and max(table) > bound:
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
        Min/Max/Piecewise, floats truncated. Min/Max/Piecewise are rebuilt
        lazily and conditions hidden first (see :meth:`_hide`).

        Powers of a Min, Max, Piecewise or residency sum are pushed here, not
        where they are printed: the pushed Min/Max is the derate factor
        ``_push_into_byte_minmax`` multiplies into the byte Min, its operands
        are what ``_integerize_minmax`` puts on an integer scale, and the
        exponent each base form ends up with is the one ``_inv_log_sym``
        tabulates."""
        cost_expr = _lazy_form(cost_expr, self._hide)
        cost_expr = cost_expr.replace(
            lambda e: e.func is sympy.log or self._is_pushed_power(e),
            lambda e: self._log_min(self._push_power(e)),
        )
        cost_expr = sympy.expand(cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func in [sympy.log, sympy.Pow, sympy.Mul],
            self._inv_log_sym,
        )
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.Mul,
            lambda e: self._min_piecewise_expand(self._push_into_byte_minmax(e)),
        )
        cost_expr = self._integerize_minmax(cost_expr)
        return cost_expr

    def _hide(self, cond):
        """A Piecewise condition as the symbol the rewrite carries in its place,
        so the rewrite never rebuilds the condition or lowers a power inside
        it; the printer canonicalizes it (:meth:`_condition`). A condition
        with the same truth value at both values of its residency literal is
        that value: a branch it rules out never reaches the Min/Max integer
        scales."""
        lit = self._residency_literal(cond)
        if lit is not None:
            off, on = (cond.subs(lit, value) for value in (0, 1))
            if off == on:
                cond = off
        if cond in (sympy.true, sympy.false):
            return cond
        if cond not in self._hidden:
            self._hidden[cond] = sympy.Symbol(f"_cond_{len(self._hidden)}")
            self._conditions[self._hidden[cond]] = cond
        return self._hidden[cond]

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
            if expr.exp == -1:
                return sympy.Symbol(
                    f"inv_{arg.name}", integer=True, nonnegative=True
                ) / self._inv_scale(arg.name)
        elif expr.func == sympy.Mul:
            expr = self._merge_power_tables(expr)
            if not expr.is_Mul:
                return expr
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

    @memoize_method
    def _power_table(self, expr):
        """A fractional power of a split or tile symbol as the table of its
        values over its buffer's division menu. Any other node, or a power
        whose menu holds a pole, is left as is."""
        if not (
            expr.is_Pow
            and self._is_split_sym(expr.base)
            and expr.exp.is_Number
            and not expr.exp.is_Integer
        ):
            return expr
        owner, raw = self._buffer_map.get(expr.base.name, (None, ()))
        if owner is None or not raw:
            return expr
        xs = sorted({int(v) for v in raw})
        if expr.exp < 0 and xs[0] <= 0:
            return expr
        return self._curve(expr.base, xs, float(expr.exp))

    def _merge_power_tables(self, expr):
        """A product's fractional powers of one buffer's split and tile symbols
        under one exponent, as one table over the product of the symbols: it is
        then quantized once, not once per factor."""
        groups: dict = {}
        for factor in expr.args:
            if isinstance(self._power_table(factor), _CurveTable):
                owner = self._buffer_map[factor.base.name][0]
                groups.setdefault((id(owner), float(factor.exp)), []).append(factor)
        merged = [group for group in groups.values() if len(group) > 1]
        if not merged:
            return expr
        dropped = {factor for group in merged for factor in group}
        factors = [factor for factor in expr.args if factor not in dropped]
        for group in merged:
            syms = [factor.base for factor in group]
            columns = zip(*(self._buffer_map[sym.name][1] for sym in syms))
            xs = sorted({math.prod(int(v) for v in column) for column in columns})
            factors.append(self._curve(sympy.Mul(*syms), xs, float(group[0].exp)))
        return sympy.Mul(*factors)

    @staticmethod
    def _curve(arg, xs, exponent):
        """``arg**exponent`` at each value in ``xs`` that ``arg`` takes."""
        ys = [sympy.Float(x**exponent) for x in xs]
        if len(xs) == 1:
            return ys[0]
        return _CurveTable(arg, sympy.Tuple(*xs), sympy.Tuple(*ys))

    def _push_into_byte_minmax(self, expr):
        """``c * Min(R, W) * D`` as ``sign(c) * Min(|c| * R * D, |c| * W * D)``,
        where ``Min(R, W)`` is the product's bytes factor and D its other Min/Max
        factors (the derates): each operand is then a sum of terms times D,
        rather than a product of variables whose bounds compound. The bytes
        factor is the one Min/Max without a quantized curve, or, when none has
        one (a committed split leaves a derate linear in residency), the one
        with the largest coefficient. The factors are costs and derates, so D
        is nonnegative."""
        if not expr.is_Mul:
            return expr
        branching = [a for a in expr.args if isinstance(a, (sympy.Min, sympy.Max))]
        if len(branching) < 2:
            return expr
        plain = [a for a in branching if not self._has_curve(a)]
        if len(plain) == 1:
            (target,) = plain
        elif len(plain) == len(branching):
            target = max(branching, key=self._largest_coefficient)
        else:
            return expr
        coefficient, _ = expr.as_coeff_Mul()
        rest = sympy.Mul(*(a for a in expr.args if a is not target)) / coefficient
        func = _LAZY.get(type(target), type(target))
        return sympy.sign(coefficient) * func(
            *(
                sympy.Add(
                    *map(
                        self._push_into_byte_minmax,
                        sympy.Add.make_args(
                            sympy.expand(abs(coefficient) * arg * rest)
                        ),
                    )
                )
                for arg in target.args
            )
        )

    def _has_curve(self, expr):
        """Whether ``expr`` holds a curve over one buffer's menu, or a power
        that lowers to one."""
        return any(
            isinstance(self._power_table(a), _CurveTable)
            for a in expr.atoms(sympy.Pow, _CurveTable)
        )

    @staticmethod
    def _largest_coefficient(minmax):
        """The largest number multiplying a term of ``minmax``'s operands."""
        return max(
            abs(float(term.as_coeff_Mul()[0]))
            for arg in minmax.args
            for term in sympy.Add.make_args(arg)
        )

    @staticmethod
    def _residency_literal(expr):
        """The one residency literal (``is_lx_<buffer>``, 0 or 1) in ``expr``;
        None for none or several."""
        literals = [s for s in expr.free_symbols if s.name.startswith("is_lx_")]
        return literals[0] if len(literals) == 1 else None

    @classmethod
    def _is_pushed_power(cls, expr):
        """A power :meth:`_push_power` lowers: of a Min, Max or Piecewise, or of
        a sum over one residency literal."""
        if not (
            expr.is_Pow
            and expr.exp.is_Number
            and not (expr.exp.is_Integer and expr.exp > 0)
        ):
            return False
        if isinstance(expr.base, (sympy.Min, sympy.Max, sympy.Piecewise)):
            return True
        return expr.base.is_Add and cls._residency_literal(expr.base) is not None

    @classmethod
    def _push_power(cls, expr):
        """A fractional or negative power of a Min, Max or Piecewise as the Min,
        Max or Piecewise of its operands' powers: a power is monotone over the
        positive values a cost takes, so a negative exponent swaps Min and Max."""
        if not cls._is_pushed_power(expr):
            return expr
        base, exponent = expr.base, expr.exp

        def power(arg):
            if isinstance(arg, (sympy.Min, sympy.Max, sympy.Piecewise)):
                return cls._push_power(sympy.Pow(arg, exponent, evaluate=False))
            result = arg**exponent
            if result.is_Mul:
                return sympy.Mul(*(cls._push_power(f) for f in result.args))
            return cls._push_power(result)

        if base.is_Add:
            # Exact at both values of the literal: f(lit)**p as
            # f(0)**p + (f(1)**p - f(0)**p) * lit.
            lit = cls._residency_literal(base)
            off, on = (power(base.subs(lit, value)) for value in (0, 1))
            return off + (on - off) * lit
        if base.is_Piecewise:
            return base.func(*((power(v), c) for v, c in base.args))
        is_min = isinstance(base, sympy.Min)
        func = _LazyMin if is_min == (exponent > 0) else _LazyMax
        return func(*(power(arg) for arg in base.args))

    @classmethod
    def _canonical_condition(cls, cond):
        """A branch condition with each ITE as And/Or, and each comparison of a
        fractional or negative power of a product of symbols against a number
        as the product's comparison against the matching bound."""
        if cond.has(ITE):
            cond = cond.replace(lambda e: isinstance(e, ITE), lambda e: e.to_nnf())
        inverted = {}
        for rel in cond.atoms(sympy.core.relational.Relational):
            bound = cls._inverted_power_bound(rel)
            lit = cls._residency_literal(rel)
            if bound is None and lit is not None and rel.has(sympy.Pow):
                # One branch per value of the literal, each inverted on its own.
                branches = []
                for value in (0, 1):
                    branch = rel.subs(lit, value)
                    if branch.is_Relational:
                        bounded = cls._inverted_power_bound(branch)
                        branch = branch if bounded is None else bounded
                    branches.append(sympy.And(sympy.Eq(lit, value), branch))
                bound = sympy.Or(*branches)
            if bound is not None:
                inverted[rel] = bound
        return cond.xreplace(inverted) if inverted else cond

    @classmethod
    def _inverted_power_bound(cls, rel):
        """``c * prod(s)**p <op> k`` as ``prod(s) <op'> (k / c)**(1 / p)``, for
        nonnegative symbols ``s`` under a fractional or negative exponent ``p``;
        None for any other comparison."""
        lhs, rhs, op = rel.lhs, rel.rhs, type(rel)
        if lhs.is_Number:
            lhs, rhs, op = rhs, lhs, _inv_rel_op.get(op)
        if op not in (sympy.Ge, sympy.Gt, sympy.Le, sympy.Lt) or not rhs.is_Number:
            return None
        coefficient, rest = lhs.as_coeff_Mul()
        powers = [factor.as_base_exp() for factor in sympy.Mul.make_args(rest)]
        exponents = {exponent for _, exponent in powers}
        if coefficient <= 0 or rhs <= 0 or len(exponents) != 1:
            return None
        (exponent,) = exponents
        if not exponent.is_Number or exponent.is_Integer and exponent > 0:
            return None
        if not all(base.is_Symbol and base.is_nonnegative for base, _ in powers):
            return None
        bound = (rhs / coefficient) ** (1 / exponent)
        if not bound.is_Rational:
            bound = sympy.Float(bound)
        op = op if exponent > 0 else _inv_rel_op[op]
        product = sympy.Mul(*(base for base, _ in powers))
        return cls._tighten_integer_bound(op, product, bound)

    @staticmethod
    def _tighten_integer_bound(rel_op, lhs, rhs):
        if not (lhs.is_integer and rhs.is_Number):
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
        branching = [
            arg
            for arg in expr.args[1:]
            if isinstance(arg, (sympy.Min, sympy.Max, sympy.Piecewise))
        ]
        # A product of two such factors stays a product of two small variables;
        # the number pushed into one of them would scale that product past
        # _MAX_PRODUCT_BOUND.
        if len(branching) == 1:
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

    def _integerize_minmax(self, expr, multiplied=False):
        # Only Min/Max constraints require integer operands. A conditional
        # objective supports float values directly; rounding its coefficients
        # can erase a large cost multiplied by scaled reciprocal variables.
        # Keep upstream's lazy Min/Max classes when rebuilding their subtrees.
        # ``multiplied``: a variable multiplies ``expr``.
        if isinstance(expr, (sympy.Min, sympy.Max)):
            scaled, scale = self._on_own_scale(expr, multiplied)
            return scaled / scale
        if not expr.has(sympy.Min, sympy.Max):
            return expr
        return expr.func(
            *(
                self._integerize_minmax(
                    arg,
                    multiplied
                    or expr.is_Mul
                    and not all(self._is_gate(o) for o in expr.args if o is not arg),
                )
                for arg in expr.args
            )
        )

    def _is_gate(self, expr):
        """A number or a 0/1 literal: a factor that selects a product's other
        factor without widening its range."""
        if expr.is_Number:
            return True
        var = self._sym_map.get(expr.name) if expr.is_Symbol else None
        return isinstance(var, cp_model.IntVar) and var.is_boolean

    def _on_own_scale(self, node, multiplied):
        """``(node * scale, scale)`` for a Min/Max, Piecewise or curve: the
        node with integer numbers, on the scale ``_integer_scale`` picks for
        the numbers ``_scale_node`` rounds. The scale is the coarse one where a
        variable multiplies the node: it widens that product and divides the
        coefficient the product is rounded with."""
        key = (node, bool(multiplied))
        if key not in self._own_scales:
            numbers: list = []
            unscaled = self._scale_node(node, sympy.Integer(1), numbers)
            scale = self._integer_scale(numbers, key[1], self._magnitude(unscaled))
            scaled = self._scale_node(node, sympy.Integer(scale))
            self._own_scales[key] = (scaled, scale)
        return self._own_scales[key]

    @memoize_method
    def _magnitude(self, expr):
        """An upper bound on ``|expr|`` over the ranges of its variables, as
        the printer bounds it: a Piecewise prints as a sum of its branches,
        each under its own literal, and a symbol or any other node is bounded
        as it prints."""
        if expr.is_Number:
            return abs(float(expr))
        if isinstance(expr, _CurveTable):
            return max(abs(float(value)) for value in expr.args[2])
        if isinstance(expr, sympy.Piecewise):
            return sum(self._magnitude(value) for value, _ in expr.args)
        if isinstance(expr, (sympy.Min, sympy.Max)):
            return max(map(self._magnitude, expr.args))
        if expr.is_Add:
            return sum(map(self._magnitude, expr.args))
        if expr.is_Mul:
            return math.prod(map(self._magnitude, expr.args))
        if expr.is_Pow and expr.exp.is_Integer and expr.exp > 0:
            return self._magnitude(expr.base) ** int(expr.exp)
        return max(map(abs, self._affine_bounds(self._print(expr))))

    @staticmethod
    def _integer_scale(numbers, coarse, magnitude):
        """The scale that puts ``numbers`` on integers: a multiple of the LCM
        of the exact ones' denominators (an inverse's scale), so they stay
        exact at the narrowest width, times the smallest power of two that
        rounds every other within _MINMAX_TOLERANCE, or, ``coarse``, within
        _QUANTIZE_TOLERANCE. The scale stays under the mode's limit, keeps the
        largest number under _MAX_MINMAX_COEFFICIENT and the operands, of
        ``magnitude`` unscaled, under _MAX_MINMAX_OPERAND: exact denominators
        past that round like any other number, and numbers no scale within it
        meets round as finely as it allows."""
        tolerance, limit = _MINMAX_TOLERANCE, _MAX_EXACT_MINMAX_SCALE
        if coarse:
            tolerance, limit = _QUANTIZE_TOLERANCE, _MAX_MINMAX_SCALE
        largest = max((abs(float(n)) for n in numbers), default=1.0)
        limit = max(
            1,
            min(
                limit,
                int(_MAX_MINMAX_COEFFICIENT / max(largest, 1.0)),
                int(_MAX_MINMAX_OPERAND / max(magnitude, 1.0)),
            ),
        )
        exact = [n for n in numbers if n.is_Rational]
        inexact = [float(n) for n in numbers if not n.is_Rational]
        base = math.lcm(*(int(n.q) for n in exact)) if exact else 1
        if base > limit:
            inexact += [float(n) for n in exact]
            base = 1
        if inexact and not coarse:
            start = base * _MIN_FLOAT_MINMAX_SCALE
            while start > max(limit, base):
                start //= 2
            base = start
        return power_of_two_scale(inexact, tolerance, limit, base) or (
            base << max(0, (limit // base).bit_length() - 1)
        )

    def _scale_node(self, node, factor, numbers=None):
        """``node * factor`` for a Min/Max, Piecewise or curve and a positive
        ``factor``, each number rounded to an integer once, where it stands.

        A node inside it that only numbers and literals multiply takes the
        term's coefficient and ``factor`` in turn, so its numbers too are
        rounded on this scale; so does a choice among numbers, whatever
        multiplies it. Any other node a variable multiplies keeps a scale of
        its own (``_on_own_scale``): the scale would widen that product, so the
        term's coefficient divides by it instead. A power is lowered here, not
        in the printer: its table is quantized on this scale.

        With ``numbers``, a dry run: each number that would be rounded is
        appended to it."""
        if isinstance(node, _CurveTable):
            arg, xs, table = node.args
            values = [self._round(factor * value, numbers) for value in table]
            return _CurveTable(arg, xs, sympy.Tuple(*values))
        if isinstance(node, sympy.Piecewise):
            return node.func(
                *(
                    (self._scale_operand(value, factor, numbers), cond)
                    for value, cond in node.args
                )
            )
        func = _LAZY.get(type(node), type(node))
        return func(*(self._scale_operand(arg, factor, numbers) for arg in node.args))

    def _scale_operand(self, operand, factor, numbers):
        # A number times a sum: each term's own coefficient is rounded.
        if operand.is_Mul and any(arg.is_Add for arg in operand.args):
            operand = sympy.expand(operand)
        return sympy.Add(
            *(
                self._scale_term(term, factor, numbers)
                for term in sympy.Add.make_args(operand)
            )
        )

    def _scale_term(self, term, factor, numbers):
        coefficient, rest = term.as_coeff_Mul()
        nodes, others = [], []
        for arg in sympy.Mul.make_args(rest):
            arg = self._power_table(arg)
            if arg.is_Number:
                coefficient *= arg
            elif isinstance(arg, _TRUNCATED):
                nodes.append(arg)
            else:
                others.append(self._integerize_minmax(arg, True))
        # The node that takes the coefficient: the only one, when nothing but
        # numbers and literals multiply it, else a choice among numbers, which
        # prints as literals times numbers and widens no product.
        if len(nodes) == 1 and all(self._is_gate(other) for other in others):
            carrier = nodes[0]
        else:
            carrier = next((node for node in nodes if self._is_choice(node)), None)
        scale = 1
        for node in nodes:
            if node is not carrier:
                scaled, own = self._on_own_scale(node, True)
                others.append(scaled)
                scale *= own
        coefficient = coefficient * factor / scale
        if carrier is None:
            return self._round(coefficient, numbers) * sympy.Mul(*others)
        scaled = self._scale_node(carrier, abs(coefficient), numbers)
        return sympy.sign(coefficient) * sympy.Mul(*others, scaled)

    @staticmethod
    def _is_choice(node):
        """A Piecewise whose every branch is a number."""
        return isinstance(node, sympy.Piecewise) and all(
            value.is_Number for value, _ in node.args
        )

    @staticmethod
    def _round(number, numbers):
        if numbers is not None:
            numbers.append(number)
            return number
        return round_scaled(number, 1)

    def _print__CurveTable(self, expr):
        """A curve outside any Min/Max, quantized on its own: its values on the
        smallest power-of-two scale that keeps each within _QUANTIZE_TOLERANCE,
        an integer table ``_division_table`` prints. An integer curve is that
        one lookup even where the caller prints without tables, and past
        _MAX_PRODUCT_BOUND: one that wide took its Min/Max's scale, so only
        numbers and literals multiply it."""
        values = [float(v) for v in expr.args[2]]
        if all(float(v).is_integer() for v in values):
            table = self._division_table(expr, bound=None)
            if table is None:
                raise ValueError(f"curve over {expr.args[0]} is not tabulable")
            return table
        scale = power_of_two_scale(values, _QUANTIZE_TOLERANCE, _MAX_PRODUCT_BOUND)
        if scale is None:
            raise ValueError(
                f"curve over {expr.args[0]} has no integer scale within "
                f"{_QUANTIZE_TOLERANCE} under {_MAX_PRODUCT_BOUND}"
            )
        table = sympy.Tuple(*(int(round(scale * v)) for v in values))
        return self._print(_CurveTable(expr.args[0], expr.args[1], table)) * (1 / scale)

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
        exprs = [arg for arg in nonints if not isinstance(arg, (int, float))]
        if len(exprs) > 1:
            # Two linear factors (e.g. two tabulated derates): multiply them
            # variable by variable rather than as Python expressions.
            result = exprs[0]
            for factor in exprs[1:] + ints:
                result = self._print_multiply_two(result, factor)
            return math.prod(a for a in nonints if isinstance(a, (int, float))) * result
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
        lowered = self._power_table(expr)
        if lowered is not expr:
            return self._print(lowered)
        if expr.exp == 2:
            base = self._print(expr.base)
            return self._print_multiply_two(base, base)
        return self._print(expr.base) ** self._print(expr.exp)

    @memoize_method
    def _condition(self, cond):
        """A Piecewise condition, from behind the symbol the rewrite hid it
        behind, in the form ``_print_condition`` lowers."""
        return self._canonical_condition(self._conditions.get(cond, cond))

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
            cond = self._condition(cond)
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

    def _print__LazyPiecewise(self, expr):
        # The printer dispatches a renamed Function subclass on its own name.
        return self._print_Piecewise(expr)

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
