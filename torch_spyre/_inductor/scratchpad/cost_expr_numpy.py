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

"""The numpy core of a sympy cost expression: one node's value from its operands'.

``NUMPY_FUNC_MAP`` maps a sympy class to the function that computes a node of that
class from the values of its operands, and ``numpy_eval`` applies it to the values of
``operands(expr)``. A value is a number, a truth value, or a numpy array of them with
one entry per candidate: every function works entry by entry, and a number stands for
itself at every entry. A Mul is computed as lambdify prints it, dividing once by
what it divides by: 98/a at a = 49 is 2.

The core reads no symbol and walks no expression: its caller evaluates a node's
operands and hands over their values. It has two callers. The CP-SAT printer
(``cpsat_expr_printer.py``) evaluates a node over one buffer's candidates, held as
exact ints. The table lowering (``cpsat_table_lowering.py``) computes each node's float
list from its operands' lists.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from functools import reduce
from typing import Any

import numpy as np
import sympy

from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge


def divides(factor: sympy.Basic) -> bool:
    """Whether factor, a factor of a Mul, is one lambdify prints it dividing by: a
    power with a negative rational exponent. A rational coefficient is not: lambdify
    prints a/49 as (1/49)*a."""
    return (
        isinstance(factor, sympy.Pow)
        and bool(factor.exp.is_Rational)
        and bool(factor.exp.is_negative)
    )


def factors(expr: sympy.Mul) -> Sequence[sympy.Basic]:
    """expr's factors in the order lambdify multiplies them, as_ordered_factors(), for
    three or more: two give the same product either way, and the sort costs."""
    return expr.as_ordered_factors() if len(expr.args) > 2 else expr.args


def operands(expr: sympy.Basic) -> list[sympy.Basic]:
    """The nodes whose values numpy_eval(expr, ...) takes, in order: a Piecewise's
    arms' values and conditions, value then condition; a Mul's factors in the order
    lambdify multiplies them (factors), one it divides by (divides) as its base; any
    other node's arguments."""
    if isinstance(expr, sympy.Piecewise):
        return [part for pair in expr.args for part in pair.args]
    if isinstance(expr, sympy.Mul):
        return [factor.base if divides(factor) else factor for factor in factors(expr)]
    return list(expr.args)


def _power(base: Any, exp: sympy.Basic, value: Any) -> Any:
    """base raised to exp, a sympy exponent of value value, as lambdify prints it: a
    power of one half is sqrt(base) and one of minus a half 1/sqrt(base), where
    base**0.5 is not always the correctly rounded root."""
    if exp == sympy.S.Half:
        return np.sqrt(np.asarray(base, dtype=float))[()]
    if exp == -sympy.S.Half:
        return 1 / np.sqrt(np.asarray(base, dtype=float))[()]
    return base**value


def _pow(expr: sympy.Pow, values: Sequence[Any]) -> Any:
    base, value = values
    return _power(base, expr.exp, value)


def _mul(expr: sympy.Mul, values: Sequence[Any]) -> Any:
    """A Mul from the values of operands(expr), as lambdify prints it: the product of
    the factors it does not divide by, in its printed order, divided once by the
    product of the bases of those it does, each raised to minus its exponent. 98/a is
    2 at a = 49, where 98 * (1/a) is under 2; a/49 is (1/49) * a."""
    above: list[Any] = []
    below: list[Any] = []
    for factor, value in zip(factors(expr), values):
        if divides(factor):
            exp = -factor.exp
            below.append(_power(value, exp, int(exp) if exp.is_Integer else float(exp)))
        else:
            above.append(value)
    return math.prod(above) / math.prod(below) if below else math.prod(above)


def _fold(binary: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    """An n-ary function from a binary numpy one, applied pairwise from the left."""
    return lambda *values: reduce(binary, values)


def _floats(function: Callable[..., Any]) -> Callable[..., Any]:
    """function over its operands as floats. numpy's log, exp and rint have no loop for
    a column of exact ints, which is how the printer holds a buffer's candidates."""
    return lambda *values: function(*(np.asarray(v, dtype=float) for v in values))


def _piecewise(*parts: Any) -> Any:
    """A Piecewise from its arms' values and conditions, value then condition, in
    order: each entry takes the first arm whose condition holds, and NaN where none
    does. An arm that holds everywhere, as a default arm's ``True`` does, is taken as
    it is."""
    values, conditions = parts[0::2], parts[1::2]
    result = values[-1]
    if not np.all(conditions[-1]):
        result = np.where(conditions[-1], result, np.nan)
    for value, condition in zip(reversed(values[:-1]), reversed(conditions[:-1])):
        result = np.where(condition, value, result)
    return result


def _relayout_charge(is_lx: Any, division: Any, *prices: Any) -> Any:
    """``RelayoutCharge(is_lx, division, *prices)``: is_lx times the price at the
    division, 0 at a division past the prices, the division rounded to the nearest
    integer as the class's own ``_imp_`` rounds it."""
    index = _floats(np.rint)(division)
    priced = [index == i for i in range(len(prices))]
    return is_lx * (np.select(priced, prices, default=0.0) if priced else 0.0)


# Add and the relationals are Python's own, as are a Mul's and a Pow's (NODE_FUNC_MAP):
# a numpy array takes them entry by entry, and a plain int stays exact at any size,
# where np.add(2**62, 2**62) wraps.
# floor and ceiling are numpy's, and round a column of exact ints or of floats alike.
# log and exp are numpy's over floats, so their operands are made floats first.
NUMPY_FUNC_MAP: dict[type, Callable[..., Any]] = {
    sympy.Add: lambda *values: sum(values),
    sympy.Abs: np.abs,
    sympy.Min: _fold(np.minimum),
    sympy.Max: _fold(np.maximum),
    sympy.floor: np.floor,
    sympy.ceiling: np.ceil,
    sympy.log: _floats(np.log),
    sympy.exp: _floats(np.exp),
    sympy.Eq: operator.eq,
    sympy.Ne: operator.ne,
    sympy.Lt: operator.lt,
    sympy.Le: operator.le,
    sympy.Gt: operator.gt,
    sympy.Ge: operator.ge,
    sympy.And: _fold(np.logical_and),
    sympy.Or: _fold(np.logical_or),
    sympy.Not: np.logical_not,
    sympy.ITE: np.where,
    sympy.Piecewise: _piecewise,
    sympy.KroneckerDelta: np.equal,
    RelayoutCharge: _relayout_charge,
}


# The classes lambdify prints by their node, not by a function of their operands'
# values: a Mul's divisions and order, a Pow's square roots.
NODE_FUNC_MAP: dict[type, Callable[[Any, Sequence[Any]], Any]] = {
    sympy.Mul: _mul,
    sympy.Pow: _pow,
}


def numpy_eval(expr: sympy.Basic, values: Sequence[Any]) -> Any:
    """expr's value from values, the values of operands(expr) in order. A Mul or a
    Pow is computed from its node as lambdify prints it (NODE_FUNC_MAP). Any other
    node's function is that of its class, or of the nearest class along its MRO that
    has one, or else an implemented function's ``_imp_``, which takes scalars, at each
    entry. Raises NotImplementedError, naming the class, when it has neither."""
    for cls in type(expr).__mro__:
        if cls in NODE_FUNC_MAP:
            return NODE_FUNC_MAP[cls](expr, values)
        if cls in NUMPY_FUNC_MAP:
            return NUMPY_FUNC_MAP[cls](*values)
    imp = getattr(expr, "_imp_", None)
    if imp is not None:
        return np.vectorize(imp, otypes=[object])(*values)
    raise NotImplementedError(f"no numpy function for {type(expr).__name__}: {expr}")
