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

Every decision the cost depends on has a finite domain: a division index over
an enumerated menu, or a residency literal. So most of the expression is a table:
the cost model evaluated exactly at each point of the decisions it depends on,
looked up by one ``element`` constraint (see :meth:`_SympyExprToCpSat._axes`).
Only what spans too many decisions to enumerate jointly -- a bundle's memory
term, a sum over its residencies -- lowers term by term, as sums, gated
products and ``int_prod``s of tables.
"""

from __future__ import annotations

import functools
import logging
import math
import operator
import sys
from fractions import Fraction
from functools import cache
from typing import Optional, TYPE_CHECKING

import numpy as np
import sympy
from sympy.printing.printer import Printer

from torch_spyre._inductor import config

if TYPE_CHECKING:
    from ortools.sat.python import cp_model, cp_model_helper
else:
    try:
        from ortools.sat.python import cp_model, cp_model_helper
    except ImportError:  # pragma: no cover - only when ortools is absent
        cp_model = cp_model_helper = None

__all__ = ["cost_expr_to_cpsat"]

logger = logging.getLogger(__name__)

# constant limit on product terms to avoid int32 overflow in CP-SAT
_MAX_PRODUCT_BOUND = 2**30
# Bits each factor of a multiplication keeps, so the product of two of them
# stays inside int32. Past that presolve mis-propagates the multiplication:
# against a 30-bit value it reported a satisfiable model INFEASIBLE, and where
# it did solve it proved plans several times worse than the exact cost of a
# plan the same model admits.
_PRODUCT_FACTOR_BITS = 15
# CP-SAT rejects a linear constraint whose worst-case sum could pass this.
_INT64_MAX = 2**63 - 1
# Scale on a Min/Max operand with fractional coefficients, so a table's step
# survives rounding to an integer.
_MINMAX_OPERAND_SCALE = 1024
# Largest value such an operand may reach once scaled: _floored's two terms each
# carry it times a denominator of at most _MINMAX_OPERAND_SCALE.
_MINMAX_OPERAND_BOUND = _INT64_MAX // (2 * _MINMAX_OPERAND_SCALE)


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


def _reduce(op):
    return lambda args: functools.reduce(op, args)


def _piecewise(args):
    """A Piecewise's values at every point: the first branch whose condition
    holds. NaN where none does, so a table that reaches it is refused."""
    result = np.nan
    for value, cond in reversed(list(zip(args[::2], args[1::2]))):
        result = np.where(cond, value, result)
    return result


# The evaluator behind the tables: for each sympy node type, the numpy
# function _values applies to the node's operands, each already evaluated at
# every point of the node's decisions as an array broadcast against the others.
# An n-ary node folds its function over the operands; a Piecewise takes the
# first branch whose condition holds (_piecewise). A type absent here is one
# the tables cannot evaluate, so _find_axes returns None for it and the node
# lowers through its own printer method: a ``RelayoutCharge`` or
# ``KroneckerDelta`` onto the solver's own division literals, an implemented
# function through _print_AppliedUndef.
_TABLE_OPS = {
    sympy.Add: _reduce(np.add),
    sympy.Mul: _reduce(np.multiply),
    sympy.Pow: lambda a: np.power(np.asarray(a[0], dtype=float), a[1]),
    sympy.Min: _reduce(np.minimum),
    sympy.Max: _reduce(np.maximum),
    sympy.floor: lambda a: np.floor(a[0]),
    sympy.ceiling: lambda a: np.ceil(a[0]),
    sympy.log: lambda a: np.log(a[0]),
    sympy.exp: lambda a: np.exp(a[0]),
    sympy.Abs: lambda a: np.abs(a[0]),
    sympy.Mod: lambda a: np.mod(a[0], a[1]),
    sympy.And: _reduce(np.logical_and),
    sympy.Or: _reduce(np.logical_or),
    sympy.Not: lambda a: np.logical_not(a[0]),
    sympy.ITE: lambda a: np.where(a[0], a[1], a[2]),
    sympy.Piecewise: _piecewise,
}


def _table_op(expr):
    """The function that evaluates ``expr`` over its operands' tables, or None."""
    if expr.is_Relational:
        compare = _operator_map[expr.rel_op]
        return lambda a: compare(a[0], a[1])
    for cls in type(expr).__mro__:
        if cls in _TABLE_OPS:
            return _TABLE_OPS[cls]
    return None


def _table_operands(expr):
    if expr.is_Piecewise:
        return [part for pair in expr.args for part in pair]
    return expr.args


# A float holds every integer below this exactly; a larger "integer" entry is
# a rounded float and stays one.
_FLOAT_EXACT_INTEGER = 1 << sys.float_info.mant_dig


def _number(value):
    """A table entry as the constant the model takes: an int where it is one."""
    value = float(value)
    if value.is_integer() and abs(value) < _FLOAT_EXACT_INTEGER:
        return int(value)
    return value


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
        max_table_entries: Optional[int] = None,
    ) -> None:
        self._model = model
        self._count = 0
        self._sym_map = sym_map
        self._buffer_map = buffer_map
        # Points a joint table may span; the caller decides, see
        # config.cpsat_max_table_entries.
        self._max_table_entries = (
            config.cpsat_max_table_entries
            if max_table_entries is None
            else max_table_entries
        )
        # These caches belong to one converter/model. A class-level @cache
        # would retain every converter and its CP model across compilations.
        self._expr_cache: dict = {}
        self._conjunction_cache: dict = {}
        self._axes_cache: dict = {}
        self._values_cache: dict = {}
        self._lookup_cache: dict = {}
        self._domain_index: dict = {}
        # Axis id (a variable's index) -> (variable, first value, size, is a
        # residency literal). See _axes.
        self._axis: dict = {}
        # The indices of the table levels: see _is_table_factor.
        self._table_vars: set = set()
        super().__init__()

    def _print(self, expr, **kwargs):
        if expr not in self._expr_cache:
            self._expr_cache[expr] = self._lower(expr, **kwargs)
        return self._expr_cache[expr]

    def _lower(self, expr, **kwargs):
        """The one decision of the lowering, made for every node ``_print``
        meets.

        When :meth:`_axes` knows which decisions the node reads and there is at
        most one (a buffer's division index, a residency literal, or the rank
        of a wide Min/Max), the node is finished as one table: :meth:`_values`
        evaluates it at every point of that decision and :meth:`_emit` hands
        CP-SAT a constant, a linear expression or one ``element`` lookup, and
        nothing under the node is lowered. Otherwise -- the node reads several
        decisions, or an operand no table can evaluate -- the printer method
        for its type lowers it term by term and recurses into the operands
        through ``_print``, so each operand gets the same chance. Those methods
        take a joint table over several decisions where that pays (``_joint``
        in ``_print_Pow``, ``_print_Min``/``_print_Max`` and
        ``_print_Piecewise``, the factor groups of ``_print_Mul``); a sum never
        does, since a sum of tables is exact for free."""
        if expr.args:
            axes = self._axes(expr)
            if axes is not None and len(axes) <= 1:
                table = self._emit(axes, self._values(expr))
                if table is not None:
                    return table
        return super()._print(expr, **kwargs)

    def convert(self, cost_expr: sympy.Expr) -> "cp_model.LinearExpr":
        """Return the CP-SAT expression equivalent to ``cost_expr`` under
        ``sym_map`` (``sympy symbol -> CP-SAT var``)."""
        logger.debug("[CP-SAT layout solver] cost expr: %s", cost_expr)
        return self._print(cost_expr)

    # -- Tables ---------------------------------------------------------------
    #
    # An axis is one decision with a finite domain: a buffer's division index
    # (its splits and tile counts are the menu's columns), a residency literal,
    # or the rank of a Min/Max too wide to take jointly. A node's table is its
    # value at every point of its axes, evaluated from the cost expression as
    # written, so the objective prices each candidate as the cost model does.

    def _add_axis(self, var, first, size, literal=False):
        self._axis.setdefault(var.index, (var, first, size, literal))
        return var.index

    def _shape(self, axes):
        return tuple(self._axis[a][2] for a in axes)

    def _size(self, axes):
        return math.prod(self._shape(axes))

    def _axes(self, expr):
        """The axes ``expr`` is a function of, as a sorted tuple of axis ids, or
        ``None`` when it is not a table: it reads a variable with no small
        domain, holds a node :data:`_TABLE_OPS` does not evaluate, or spans
        more than ``max_table_entries`` points jointly."""
        if expr not in self._axes_cache:
            self._axes_cache[expr] = self._find_axes(expr)
        return self._axes_cache[expr]

    def _find_axes(self, expr):
        if expr.is_Number or expr in (sympy.true, sympy.false):
            return ()
        if expr.is_Symbol:
            return self._leaf_axes(expr)
        if _table_op(expr) is None:
            return None
        found: set = set()
        for arg in _table_operands(expr):
            axes = self._axes(arg)
            if axes is None:
                return None
            found.update(axes)
        axes = tuple(sorted(found))
        if len(axes) <= 1 or self._size(axes) <= self._max_table_entries:
            return axes
        if isinstance(expr, (sympy.Min, sympy.Max)):
            return self._rank_axes(expr)
        return None

    def _leaf_axes(self, symbol):
        """A symbol's axes, its values recorded in ``_values_cache``: a column
        of its buffer's menu, a division index, or a variable of the solve with
        a small domain. A constant has none."""
        name = symbol.name
        mapped = self._sym_map.get(name)
        if isinstance(mapped, (int, float)):
            self._values_cache[symbol] = np.asarray(float(mapped))
            return ()
        wrapper, raw = self._buffer_map.get(name, (None, None))
        if wrapper is None:
            wrapper = self._sym_map.get(f"_division_of_{name}")
            if wrapper is not None:
                raw = range(len(wrapper.buffer.core_divisions))
        if wrapper is not None:
            self._values_cache[symbol] = np.asarray(raw, dtype=float)
            return (self._add_axis(wrapper.division, 0, len(raw)),)
        if not isinstance(mapped, cp_model.IntVar):
            return None
        domain = list(mapped.proto.domain)
        lo, hi = domain[0], domain[-1]
        if len(domain) == 2:
            if hi - lo + 1 > self._max_table_entries:
                return None
            self._values_cache[symbol] = np.arange(lo, hi + 1, dtype=float)
            return (self._add_axis(mapped, lo, hi - lo + 1, mapped.is_boolean),)
        # A domain with holes, such as a split's divisors, is indexed by a
        # selector of its own: one per variable, so two such variables stay
        # independent decisions.
        values = [v for a, b in zip(domain[::2], domain[1::2]) for v in range(a, b + 1)]
        if len(values) > self._max_table_entries:
            return None
        if mapped.index not in self._domain_index:
            index = self._model.new_int_var(0, len(values) - 1, f"{name}_index")
            self._model.add_element(index, values, mapped)
            self._domain_index[mapped.index] = index
        self._values_cache[symbol] = np.asarray(values, dtype=float)
        return (self._add_axis(self._domain_index[mapped.index], 0, len(values)),)

    def _rank_axes(self, expr):
        """A ``Min``/``Max`` whose operands are tables too wide to take jointly,
        as one axis: the rank of its value among every value the operands take.
        Ranks keep the values' order, so the extreme of the operands' ranks is
        the rank of their extreme, and the table is the operands' values
        together rather than their product."""
        tables = [(self._axes(arg), self._values(arg)) for arg in expr.args]
        found = np.concatenate([np.ravel(values) for _, values in tables])
        if np.isnan(found).any():
            return None
        values = np.unique(found)
        ranks = []
        for axes, table in tables:
            rank = np.searchsorted(values, table)
            ranks.append(self._emit(axes, rank) if axes else int(rank))
        var = self._model.new_int_var(0, len(values) - 1, f"rank_{self._count}")
        self._count += 1
        if isinstance(expr, sympy.Min):
            self._model.add_min_equality(var, ranks)
        else:
            self._model.add_max_equality(var, ranks)
        self._values_cache[expr] = values
        return (self._add_axis(var, 0, len(values)),)

    def _values(self, expr):
        """``expr`` at every point of its :meth:`_axes`, one array dimension
        per axis."""
        if expr not in self._values_cache:
            if expr.is_Number:
                return np.asarray(float(expr))
            if expr in (sympy.true, sympy.false):
                return np.asarray(bool(expr))
            axes = self._axes(expr)  # a leaf or a rank records its own values
            if expr not in self._values_cache:
                operands = [self._broadcast(a, axes) for a in _table_operands(expr)]
                with np.errstate(all="ignore"):
                    values = _table_op(expr)(operands)
                self._values_cache[expr] = np.broadcast_to(values, self._shape(axes))
        return self._values_cache[expr]

    def _broadcast(self, expr, axes):
        """``expr``'s table shaped to broadcast over ``axes``, a superset of
        its own."""
        own = self._axes(expr)
        shape = [self._axis[a][2] if a in own else 1 for a in axes]
        return np.reshape(self._values(expr), shape)

    def _joint(self, expr):
        """``expr`` as one table over all of its axes, or None."""
        axes = self._axes(expr)
        return None if axes is None else self._emit(axes, self._values(expr))

    def _emit(self, axes, values):
        """A table as a CP-SAT expression: a constant, a linear expression
        where the table is linear in its axes, else a lookup. None when an
        entry is not finite -- a pole that a branch's guard excludes, which
        then lowers branch by branch."""
        values = np.asarray(values)
        if values.dtype == bool:
            return self._emit_condition(axes, values)
        values = values.astype(float)
        if not np.isfinite(values).all():
            return None
        flat = values.ravel()
        if not axes or (flat == flat[0]).all():
            return _number(flat[0])
        linear = self._linear(axes, values)
        if linear is not None:
            return linear
        return self._lookup(axes, flat)

    def _emit_condition(self, axes, values):
        flat = values.ravel()
        if not axes or (flat == flat[0]).all():
            return bool(flat[0])
        var, _, _, literal = self._axis[axes[0]]
        if len(axes) == 1 and literal:
            return var if flat[1] else var.Not()
        return self._lookup(axes, flat.astype(int))

    def _linear(self, axes, values):
        """``values`` as ``offset + sum(slope * axis)`` where every step along
        every axis is the same, else None."""
        slopes = []
        for dim, size in enumerate(values.shape):
            steps = np.diff(values, axis=dim) if size > 1 else np.zeros(1)
            if not (steps == steps.flat[0]).all():
                return None
            slopes.append(steps.flat[0])
        terms = [(self._axis[a], s) for a, s in zip(axes, slopes) if s]
        offset = values.flat[0] - sum(s * first for (_, first, *_), s in terms)
        if len(terms) == 1 and terms[0][1] == 1 and offset == 0:
            return terms[0][0][0]
        return sum((_number(s) * var for (var, *_), s in terms), _number(offset))

    def _index(self, axes):
        """The flat, row-major position of a point of ``axes``, as a linear
        expression ``element`` takes for its index."""
        index, stride = 0, 1
        for a in reversed(axes):
            var, first, size, _ = self._axis[a]
            index = index + stride * (var - first)
            stride *= size
        return index

    def _lookup(self, axes, flat):
        """One ``element`` lookup of a level variable, ``offset + coefficient *
        level``. Levels are exact where the entries' common denominator leaves
        them inside _MAX_PRODUCT_BOUND, else rounded against the entries' span:
        the differences are what order the points, so a fraction of the span is
        what the rounding may cost, not a fraction of the largest entry."""
        rationals = [Fraction(float(v)) for v in flat]
        base = min(rationals)
        denominator = math.lcm(*(v.denominator for v in rationals))
        differences = [int((v - base) * denominator) for v in rationals]
        divisor = math.gcd(*differences)
        table = tuple(v // divisor for v in differences)
        scale = Fraction(divisor, denominator)
        rounded = max(table) > _MAX_PRODUCT_BOUND
        if rounded:
            step = (max(rationals) - base) / _MAX_PRODUCT_BOUND
            table = tuple(round((v - base) / step) for v in rationals)
            offset, coefficient = _number(base), float(step)
        else:
            offset, coefficient = _number(base), _number(scale)
        key = axes, table
        if key not in self._lookup_cache:
            var = self._model.new_int_var_from_domain(
                cp_model.Domain.FromValues(table), f"table_{self._count}"
            )
            self._count += 1
            self._model.add_element(self._index(axes), table, var)
            self._lookup_cache[key] = var
            self._table_vars.add(var.index)
        var = self._lookup_cache[key]
        if coefficient == 1 and offset == 0:
            return var
        return coefficient * var + offset

    def _print_Integer(self, expr):
        return int(expr.p)

    def _print_Number(self, expr):
        return float(expr)

    def _print_Add(self, expr):
        return sum(self._print(arg) for arg in expr.args)

    def _grouped_factors(self, expr):
        """``expr``'s factors, printed, with the tables that fit together in
        ``max_table_entries`` taken as one table of their product: a bundle term's
        derates and a buffer's splits beside them are numbers at each point,
        which one table holds exactly where multiplying their variables would
        need an ``int_prod``. A factor over a residency literal stays out, so
        its product stays a gate."""
        groups, rest = [], []
        for arg in expr.args:
            axes = self._axes(arg)
            if not axes or any(self._axis[a][3] for a in axes):
                rest.append(arg)
                continue
            for group in groups:
                union = tuple(sorted({*group[0], *axes}))
                if len(union) == 1 or self._size(union) <= self._max_table_entries:
                    group[0] = union
                    group[1].append(arg)
                    break
            else:
                groups.append([axes, [arg]])
        printed = [self._print(arg) for arg in rest]
        for axes, args in groups:
            table = None
            if len(args) > 1:
                values = [self._broadcast(arg, axes) for arg in args]
                table = self._emit(
                    axes, np.broadcast_to(math.prod(values), self._shape(axes))
                )
            if table is None:
                printed.extend(self._print(arg) for arg in args)
            else:
                printed.append(table)
        return printed

    def _print_Mul(self, expr):
        return self._print_multiply(self._grouped_factors(expr))

    def _print_multiply_two(self, a, b):
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            return a * b
        for x, y in ((a, b), (b, a)):
            split = self._boolean_affine(x)
            if split is not None and self._boolean_affine(y) is None:
                return self._literal_times(split, y)
        if self._is_table_factor(a):
            return self._times_table(b, a)
        if self._is_table_factor(b):
            return self._times_table(a, b)
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
        # A sum: each of its variables times the other factor. Products of sums
        # reach here as written, since the cost expression is not expanded.
        if not isinstance(a, cp_model.LinearExpr) or not isinstance(
            b, cp_model.LinearExpr
        ):
            raise NotImplementedError(f"multiplying {type(a)} by {type(b)}")
        if isinstance(a, cp_model.IntVar):
            a, b = b, a
        try:
            flat = cp_model.FlatIntExpr(a)
        except TypeError:
            flat = cp_model.FlatFloatExpr(a)
        result = flat.offset * b
        for var, c in zip(flat.vars, flat.coeffs):
            result = result + c * self._print_multiply_two(var, b)
        return result

    def _prod(self, factors):
        """Product of constants and linear expressions; two rounded tables can
        meet here, each an affine of its level, which ``math.prod`` refuses."""
        constants = [f for f in factors if isinstance(f, (int, float))]
        result = math.prod(constants)
        for factor in factors:
            if not isinstance(factor, (int, float)):
                result = self._print_multiply_two(result, factor)
        return result

    def _is_table_factor(self, arg):
        """Whether ``arg`` is linear in table levels only (see :meth:`_lookup`),
        whose product with a wide factor is narrowed past the bound rather than
        refused: a level may span up to _MAX_PRODUCT_BOUND by design, and below
        the bound the product is exact."""
        if isinstance(arg, (int, float)):
            return False
        try:
            flat = cp_model.FlatFloatExpr(arg)
        except TypeError:
            return False
        return bool(flat.vars) and all(v.index in self._table_vars for v in flat.vars)

    def _print_multiply(self, args):
        tables = [arg for arg in args if self._is_table_factor(arg)]
        if tables:
            # Last: a bundle's memory term carries its underfill and spill
            # derates, which meet the value only once it is a sum of products.
            rest = [arg for arg in args if not self._is_table_factor(arg)]
            result = self._print_multiply(rest) if rest else 1
            for table in tables:
                result = self._times_table(result, table)
            return result

        ints = [arg for arg in args if isinstance(arg, cp_model.IntVar)]
        nonints = [arg for arg in args if not isinstance(arg, cp_model.IntVar)]
        if len(ints) == 1:
            return self._print_multiply_two(self._prod(nonints), ints[0])
        elif len(ints) == 0:
            return self._prod(nonints)

        # Names need not be unique: in particular, both Boolean constants
        # have an empty name. Cache by the variables' actual model identity.
        name = "_product_" + "_".join(str(arg.index) for arg in ints)
        if name in self._sym_map:
            return self._print_multiply_two(self._prod(nonints), self._sym_map[name])

        if any(arg.is_boolean for arg in ints):
            product = self._print_gated_product(ints, name)
            self._sym_map[name] = product
            return self._print_multiply_two(self._prod(nonints), product)

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
        return self._print_multiply_two(self._prod(nonints), product)

    def _narrowed_pair(self, a, b):
        """``(factors, bounds, divisor)``: ``a`` and ``b`` each floored so that
        their product keeps _MAX_PRODUCT_BOUND and ``divisor`` scales it back.

        A bit comes off whichever factor is wider at the time, so a narrow factor
        -- a derate, or a table's levels -- keeps all of its precision and the
        wide value it multiplies gives up only range it was not using. An even
        _PRODUCT_FACTOR_BITS each is what two equally wide factors settle on.
        """
        bounds = [self._affine_bounds(x) for x in (a, b)]
        widths = [max(abs(lb), abs(ub)).bit_length() for lb, ub in bounds]
        shifts = [0, 0]
        while sum(widths) > 2 * _PRODUCT_FACTOR_BITS:
            wider = 0 if widths[0] >= widths[1] else 1
            widths[wider] -= 1
            shifts[wider] += 1
        factors, narrowed = [], []
        for value, (lb, ub), shift in zip((a, b), bounds, shifts):
            divisor = 1 << shift
            if divisor > 1:
                lb, ub = lb // divisor, ub // divisor
                value = self._floored(value, divisor, lb, ub)
            factors.append(value)
            narrowed.append((lb, ub))
        return factors, narrowed, 1 << sum(shifts)

    def _times_table(self, value, table):
        """``value * table`` for a ``table`` linear in levels, each pair of their
        variables taking its own :meth:`_print_table_product`."""
        if isinstance(value, (int, float)):
            return value * table
        split = self._boolean_affine(value)
        if split is not None:
            return self._literal_times(split, table)
        flat = cp_model.FlatFloatExpr(value)
        levels = cp_model.FlatFloatExpr(table)
        result = levels.offset * value
        for level, weight in zip(levels.vars, levels.coeffs):
            result = result + weight * flat.offset * level
            for var, c in zip(flat.vars, flat.coeffs):
                result = result + weight * c * self._print_table_product(level, var)
        return result

    def _print_table_product(self, level, value):
        """``level * value``, past the bound a product of two wide factors is
        refused at: each factor keeps _PRODUCT_FACTOR_BITS, far finer than the
        objective resolves, so their product stays inside int32. Cached by the
        pair, as _print_multiply caches its products: a bundle's derate meets
        the same operand's bytes in more than one term."""
        key = f"_table_{level.index}_{value.index}"
        if key in self._sym_map:
            return self._sym_map[key]
        if value.is_boolean:
            result = self._print_gated_product([level, value], key)
        else:
            # ``a * b`` as ``lo_a * b + lo_b * (a - lo_a) + (a - lo_a) * (b -
            # lo_b)``: the two linear parts are exact, so only the factors'
            # spans meet in the product that has to be narrowed. A derate spans
            # a fraction of its value, and so loses a fraction of what flooring
            # it whole would cost.
            lo_level = max(0, self._affine_bounds(level)[0])
            lo_value = max(0, self._affine_bounds(value)[0])
            spans = (level - lo_level, value - lo_value)
            factors, ((low, high), (lb, ub)), divisor = self._narrowed_pair(*spans)
            corners = (low * lb, low * ub, high * lb, high * ub)
            product = self._model.new_int_var(min(corners), max(corners), key)
            self._model.add_multiplication_equality(product, factors)
            result = lo_level * value + lo_value * spans[0] + divisor * product
        self._sym_map[key] = result
        return result

    def _floored(self, value, divisor, lb, ub):
        """``value // divisor`` by linear bounds, not an ``int_div``: that is the
        chained shape that overflows presolve."""
        var = self._model.new_int_var(lb, ub, f"floored_{self._count}")
        self._count += 1
        self._model.add(value - divisor * var >= 0)
        self._model.add(value - divisor * var < divisor)
        return var

    @staticmethod
    def _boolean_affine(x):
        """``(literal, alpha, beta)`` when ``x`` is ``alpha + beta * literal`` for
        one Boolean variable (a negated literal counts), else None."""
        if isinstance(x, cp_model_helper.NotBooleanVariable):
            return x.Not(), 1, -1
        if isinstance(x, (int, float)) or not isinstance(x, cp_model.LinearExpr):
            return None
        try:
            flat = cp_model.FlatFloatExpr(x)
        except TypeError:
            return None
        if len(flat.vars) != 1 or not flat.vars[0].is_boolean:
            return None
        return flat.vars[0], flat.offset, flat.coeffs[0]

    def _literal_times(self, split, value):
        """``(alpha + beta * b) * value`` as ``alpha * [not b] value + (alpha +
        beta) * [b] value``, each part a gate on its own literal. Expanding it
        as ``alpha * value + beta * (b * value)`` instead leaves the linear
        relaxation free to cancel ``value`` against its gated copy: at a
        fractional ``b`` the copy can sit at the top of its range while
        ``value`` sits at the bottom, so ``(1 - b) * value`` relaxes far below
        zero although it is never negative. A gate per literal keeps each part
        inside ``[min(0, lb), max(0, ub)]`` of ``value``."""
        literal, alpha, beta = split
        result = 0
        for lit, weight in ((literal.Not(), alpha), (literal, alpha + beta)):
            if weight:
                result = result + weight * self._gate(lit, value)
        return result

    def _gate(self, lit, value):
        """``value`` while ``lit`` holds, else 0: per variable of ``value`` one
        gated variable, so a table's level is gated exactly and its scale stays
        a coefficient."""
        if isinstance(value, (int, float)):
            return value * (lit if isinstance(lit, cp_model.IntVar) else 1 - lit.Not())
        flat = cp_model.FlatFloatExpr(value)
        if flat.offset:
            base = lit if isinstance(lit, cp_model.IntVar) else 1 - lit.Not()
            result = flat.offset * base
        else:
            result = 0
        negated = not isinstance(lit, cp_model.IntVar)
        positive = lit.Not() if negated else lit
        for var, c in zip(flat.vars, flat.coeffs):
            key = f"_gate_{'n' if negated else 'p'}{positive.index}_{var.index}"
            if key not in self._sym_map:
                lb, ub = self._affine_bounds(var)
                gated = self._model.new_int_var(min(0, int(lb)), max(0, int(ub)), key)
                self._model.add(gated == var).only_enforce_if(lit)
                self._model.add(gated == 0).only_enforce_if(lit.Not())
                self._sym_map[key] = gated
                if var.index in self._table_vars:
                    self._table_vars.add(gated.index)
            result = result + c * self._sym_map[key]
        return result

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
        raise NotImplementedError(f"not implemented. expr: {expr}")

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
        argument unresolved, prices 0. Memoized, since a bundle repeats the
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
        if expr.exp.is_Integer and expr.exp > 1:
            base = self._print(expr.base)
            result = base
            for _ in range(int(expr.exp) - 1):
                result = self._print_multiply_two(result, base)
            return result
        return self._tabulated(expr)

    def _tabulated(self, expr):
        """``expr`` as one table, for a node with no lowering of its own: a
        fractional or negative power, a floor, a log. Raises when it is not
        a finite function of the decisions, since an approximation the solver
        then optimizes against is worse than no objective at all."""
        table = self._joint(expr)
        if table is None:
            raise ValueError(
                f"cannot tabulate {type(expr).__name__}: it is not a finite "
                f"function of the decisions within {self._max_table_entries} entries"
            )
        return table

    def _print_floor(self, expr):
        return self._tabulated(expr)

    def _print_ceiling(self, expr):
        return self._tabulated(expr)

    def _print_ITE(self, expr):
        return self._print(expr.to_nnf(simplify=False))

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
        table = self._joint(expr)
        if table is not None:
            return table
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
        args, _ = self._integer_operands([self._print(arg) for arg in expr.args])
        return _operator_map[expr.rel_op](*args)

    def _print_log(self, expr):
        return self._tabulated(expr)

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
        falls back); :meth:`_integer_operands` rounds those a table produced."""
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

    def _integer_operands(self, args):
        """``(args, divisor)``: Min/Max operands with integer coefficients,
        rounded at ``_MINMAX_OPERAND_SCALE`` where a table or the cost model's own
        coefficients left them fractional. Operands that are already integers
        come back unchanged."""

        def fractional(arg):
            if isinstance(arg, (int, float)):
                return not float(arg).is_integer()
            try:
                cp_model.FlatIntExpr(arg)
            except TypeError:
                return True
            return False

        if not any(fractional(arg) for arg in args):
            return args, 1

        # The scale gives way to the operands' size, so their bounds stay in
        # int64, and to their largest coefficient: past 2**31 one can make
        # presolve report a satisfiable model INFEASIBLE.
        def sizes(arg):
            if isinstance(arg, (int, float)):
                return abs(arg), 0.0
            flat = cp_model.FlatFloatExpr(arg)
            spans = (max(abs(v.domain.min()), abs(v.domain.max())) for v in flat.vars)
            size = sum(abs(c) * s for c, s in zip(flat.coeffs, spans))
            return abs(flat.offset) + size, max(map(abs, flat.coeffs), default=0.0)

        magnitude, coefficient = map(max, zip(*map(sizes, args)))
        scale = min(
            _MINMAX_OPERAND_SCALE,
            _MINMAX_OPERAND_BOUND / max(magnitude, 1.0),
            max(1.0, _MAX_PRODUCT_BOUND / max(coefficient, 1.0)),
        )

        def term(var, weight):
            # A table's product carries its levels' scale, which rounding the
            # weight alone would lose, so floor the weighted variable.
            if float(weight).is_integer():
                return int(weight) * var
            ratio = Fraction(abs(weight)).limit_denominator(_MINMAX_OPERAND_SCALE)
            if ratio == 0:
                ratio = Fraction(1, round(1 / abs(weight)))
            lb, ub = self._affine_bounds(ratio.numerator * var)
            floored = self._floored(
                ratio.numerator * var,
                ratio.denominator,
                lb // ratio.denominator,
                ub // ratio.denominator,
            )
            return floored if weight > 0 else -floored

        def rounded(arg):
            if isinstance(arg, (int, float)):
                return round(arg * scale)
            flat = cp_model.FlatFloatExpr(arg)
            return sum(
                (term(v, c * scale) for v, c in zip(flat.vars, flat.coeffs)),
                round(flat.offset * scale),
            )

        return [rounded(arg) for arg in args], scale

    def _print_Max(self, expr):
        table = self._joint(expr)
        if table is not None:
            return table
        # max range is (max(mins), max(maxes))
        args = [self._print(arg) for arg in expr.args]
        if all(isinstance(a, (int, float)) for a in args):
            return max(args)  # a lazy Max of constants was never folded
        args, divisor = self._integer_operands(args)
        args = [self._lin_max_operand(arg) for arg in args]
        bounds = map(max, zip(*[self._affine_bounds(arg) for arg in args]))
        max_var = self._model.new_int_var(*bounds, f"max_var_{self._count}")
        self._model.AddMaxEquality(max_var, args)
        self._count += 1
        return max_var if divisor == 1 else max_var * (1.0 / divisor)

    def _print_Min(self, expr):
        table = self._joint(expr)
        if table is not None:
            return table
        # min range is (min(mins), min(maxes))
        args = [self._print(arg) for arg in expr.args]
        if all(isinstance(a, (int, float)) for a in args):
            return min(args)  # a lazy Min of constants was never folded
        args, divisor = self._integer_operands(args)
        args = [self._lin_max_operand(arg) for arg in args]
        bounds = map(min, zip(*[self._affine_bounds(arg) for arg in args]))
        min_var = self._model.new_int_var(*bounds, f"min_var_{self._count}")
        self._model.AddMinEquality(min_var, args)
        self._count += 1
        return min_var if divisor == 1 else min_var * (1.0 / divisor)


def cost_expr_to_cpsat(
    model: "cp_model.CpModel",
    sym_map: dict,
    buffer_map: dict,
    cost_expr: sympy.Expr,
    max_table_entries: Optional[int] = None,
) -> "cp_model.LinearExpr":
    """``cost_expr`` as a CP-SAT expression over ``model``; the variables and
    constraints the lowering needs are added to ``model``, and ``sym_map`` gains
    the helper variables it creates. ``max_table_entries`` bounds a joint table
    (see ``config.cpsat_max_table_entries``, the default)."""
    printer = _SympyExprToCpSat(model, sym_map, buffer_map, max_table_entries)
    return printer.convert(cost_expr)
