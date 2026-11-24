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
class from the values of its operands, and ``numpy_eval`` applies it. A value is a
number, a truth value, or a numpy array of them with one entry per candidate: every
function works entry by entry, and a number stands for itself at every entry.

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


# Add, Mul, Pow and the relationals are Python's own: a numpy array takes them entry by
# entry, and a plain int stays exact at any size, where np.add(2**62, 2**62) wraps.
# floor and ceiling are numpy's, and round a column of exact ints or of floats alike.
# log and exp are numpy's over floats, so their operands are made floats first.
NUMPY_FUNC_MAP: dict[type, Callable[..., Any]] = {
    sympy.Add: lambda *values: sum(values),
    sympy.Mul: lambda *values: math.prod(values),
    sympy.Pow: operator.pow,
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


def numpy_eval(expr: sympy.Basic, values: Sequence[Any]) -> Any:
    """expr's value from values, its operands' values in order: a Piecewise's arms'
    values and conditions, value then condition, and any other node's arguments. The
    function is that of expr's class, or of the nearest class along its MRO that has
    one, or else an implemented function's ``_imp_``, which takes scalars, at each
    entry. Raises NotImplementedError, naming the class, when it has neither."""
    for cls in type(expr).__mro__:
        if cls in NUMPY_FUNC_MAP:
            return NUMPY_FUNC_MAP[cls](*values)
    imp = getattr(expr, "_imp_", None)
    if imp is not None:
        return np.vectorize(imp, otypes=[object])(*values)
    raise NotImplementedError(f"no numpy function for {type(expr).__name__}: {expr}")
