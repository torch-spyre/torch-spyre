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
operands and hands over their values. The CP-SAT printer's candidate evaluation
(``cpsat_expr_printer.py``) is that caller.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from functools import reduce
from typing import Any

import numpy as np
import sympy


def _fold(binary: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    """An n-ary function from a binary numpy one, applied pairwise from the left."""
    return lambda *values: reduce(binary, values)


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


# Add, Mul, Pow and the relationals are Python's own: a numpy array takes them entry by
# entry, and a plain int stays exact at any size, where np.add(2**62, 2**62) wraps.
NUMPY_FUNC_MAP: dict[type, Callable[..., Any]] = {
    sympy.Add: lambda *values: sum(values),
    sympy.Mul: lambda *values: math.prod(values),
    sympy.Pow: operator.pow,
    sympy.Min: _fold(np.minimum),
    sympy.Max: _fold(np.maximum),
    sympy.Eq: operator.eq,
    sympy.Ne: operator.ne,
    sympy.Lt: operator.lt,
    sympy.Le: operator.le,
    sympy.Gt: operator.gt,
    sympy.Ge: operator.ge,
    sympy.And: _fold(np.logical_and),
    sympy.Or: _fold(np.logical_or),
    sympy.Piecewise: _piecewise,
    sympy.KroneckerDelta: np.equal,
}


def numpy_eval(expr: sympy.Basic, values: Sequence[Any]) -> Any:
    """expr's value from values, its operands' values in order: a Piecewise's arms'
    values and conditions, value then condition, and any other node's arguments. The
    function is that of expr's class, or of the nearest class along its MRO that has
    one. Raises NotImplementedError, naming the class, when none has."""
    for cls in type(expr).__mro__:
        if cls in NUMPY_FUNC_MAP:
            return NUMPY_FUNC_MAP[cls](*values)
    raise NotImplementedError(f"no numpy function for {type(expr).__name__}: {expr}")
