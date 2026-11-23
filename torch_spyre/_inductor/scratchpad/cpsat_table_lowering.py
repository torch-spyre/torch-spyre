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

"""A sympy cost expression lowered to a CP-SAT objective from exact tables.

Every cost value is computed before the solve, in floating point, and CP-SAT only
works out which entry of a table applies: ``value == values[index]``. Three layers:

- values: ``Index``, a position, and ``AbstractRealization``, what a node is realized
  as, here always a ``TableRealization``. They hold no model;
- the printer: ``TableLowering``, a sympy ``Printer`` whose print of a node is the
  node's realization. It prints a node's operands first, reads their tables, and
  computes the node's list on a position it builds from the node's inputs;
- the build: a realization's ``cp_expr`` rounds it onto ``COST_SCALE`` and adds it to
  the model. Only ``cp_expr`` methods touch the model.

One table per node: an expression whose inputs fit the cap together is lowered, and
anything wider is refused. A node's list is the numpy core's function of its sympy
class (``cost_expr_numpy.py``, which the CP-SAT printer evaluates its candidates
with) applied to its operands' tables; a class the core has no function for is
lambdified. Nothing imports the module yet.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import sympy
from ortools.sat.python import cp_model
from sympy.logic.boolalg import Boolean
from sympy.printing.printer import Printer

from torch_spyre._inductor.scratchpad.cost_expr_numpy import numpy_eval

# One cost unit is COST_SCALE objective units: a table entry is
# rint(COST_SCALE * value).
COST_SCALE = 1024
# The largest table entry: a sum of thousands of entries stays inside int64.
MAX_ENTRY = 2**50

# What CP-SAT takes as a linear expression: a LinearExpr, an IntVar or an int.
CpExpr: TypeAlias = cp_model.LinearExprT


class TableLoweringError(Exception):
    """An objective the tables cannot lower. ``node`` is the sympy node refused, or
    None when the refusal names none; ``reason`` says why.

    It subclasses none of RuntimeError, TypeError and ValueError, so an ``except``
    around the objective that catches those cannot turn it into a different solve.
    """

    def __init__(self, node: sympy.Basic | None, reason: str) -> None:
        super().__init__(reason if node is None else f"{reason}: {node}")
        self.node = node
        self.reason = reason


# --------------------------------------------------------------------------- values


class Index(ABC):
    """A position: a map from the inputs' values to 0 .. size - 1. Each kind of step is
    a frozen dataclass holding the positions it is made from, so equal steps are one
    position. It holds no model."""

    size: int  # the number of positions

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The position as an expression in model."""

    @abstractmethod
    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        """The position at every point of grid, which holds each input position's
        coordinate there: what cp_expr builds, computed."""


@dataclass(frozen=True)
class InputIndex(Index):
    """A leaf: a variable's own position, v - lo, for a variable whose values run from
    lo to lo + size - 1."""

    var: cp_model.IntVar
    size: int
    lo: int = 0

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return self.var - self.lo if self.lo else self.var

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return grid[self]


@dataclass(frozen=True)
class UnitIndex(Index):
    """A leaf: the one position of a constant, on no input."""

    size: int = field(default=1, init=False)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return 0

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return 0


@dataclass(frozen=True)
class CombineIndex(Index):
    """left * right.size + right: linear, as right.size is a constant."""

    left: Index
    right: Index
    size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", self.left.size * self.right.size)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return self.left.cp_expr(model) * self.right.size + self.right.cp_expr(model)

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return self.left.on_grid(grid) * self.right.size + self.right.on_grid(grid)


UNIT = UnitIndex()


def _read_only(values: Any, index: Index) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.shape != (index.size,):
        raise ValueError(
            f"a list of shape {array.shape} on an index of size {index.size}"
        )
    array.setflags(write=False)
    return array


class AbstractRealization(ABC):
    """What a node is realized as: what the printer returns for it to its parent, and
    what the objective takes of the root. It holds no model."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, in COST_SCALE units."""

    @abstractmethod
    def table(self, cap: int) -> TableRealization:
        """The realization as one table, for a parent that computes its own list from
        it. cap bounds the entries of a table that has to be made."""


@dataclass(frozen=True, eq=False, kw_only=True)
class TableRealization(AbstractRealization):
    """values[index]: a read-only float list on a position, of the position's size.
    Compared by identity."""

    values: np.ndarray
    index: Index

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _read_only(self.values, self.index))

    @property
    def finite(self) -> bool:
        """Whether every value in its list is a finite number."""
        return bool(np.all(np.isfinite(self.values)))

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The list rounded once onto COST_SCALE: a number when it is constant, adding
        nothing to the model, and otherwise one add_element on its position. Raises
        TableLoweringError, naming no node, for an entry past MAX_ENTRY."""
        scaled = np.rint(COST_SCALE * self.values)
        largest = float(np.max(np.abs(scaled)))
        if largest > MAX_ENTRY:
            raise TableLoweringError(
                None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
            )
        entries = [int(v) for v in scaled]
        if len(set(entries)) == 1:
            return entries[0]
        target = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(entries))), "table"
        )
        model.add_element(self.index.cp_expr(model), entries, target)
        return target

    def table(self, cap: int) -> TableRealization:
        """Itself: none is made, so cap is not read."""
        return self


# Each symbol's input: the variable that picks its value, and the symbol's value at
# each of that variable's values, from its lowest to its highest.
Input: TypeAlias = tuple[cp_model.IntVar, Sequence[float]]
Inputs: TypeAlias = Mapping[sympy.Symbol, Input]


# --------------------------------------------------------------------- a node's list


# The form of an operand in its node's function: as it is (None), a condition
# (CONDITION), or the base of a Mul's factor with a negative rational exponent, which
# the Mul divides by (the exponent, a sympy.Rational).
CONDITION = "condition"
Form: TypeAlias = "str | sympy.Rational | None"


def _evaluate(
    expr: sympy.Basic, forms: tuple[Form, ...], arguments: Sequence[np.ndarray]
) -> Any:
    """expr's list from arguments, its operands' lists in the forms they take: by the
    numpy core's function of its class, or by its class lambdified (_lambdified) when
    the core has none. A condition is compared with 0, and a Mul divides its other
    factors' product by the bases of the factors it divides by, as lambdify prints a
    quotient: 98/a, where 98*(1/a) floors one low at a = 49."""
    plain = [
        argument != 0 if form == CONDITION else argument
        for argument, form in zip(arguments, forms)
        if form is None or form == CONDITION
    ]
    divisors = [
        argument ** float(-form)  # type: ignore[operator]
        for argument, form in zip(arguments, forms)
        if not (form is None or form == CONDITION)
    ]
    try:
        value = numpy_eval(expr, plain)
    except NotImplementedError:
        return _apply(_lambdified(expr.func, forms), arguments)
    return value / math.prod(divisors) if divisors else value


def _lambdified(func: Any, forms: tuple[Form, ...]) -> Callable[..., Any]:
    """The function of a node of sympy class func whose operands take forms, lambdified
    with the numpy module over one placeholder per operand: for a class the numpy core
    has no function for. Nothing keeps it.

    The placeholders are plain symbols with identifier names: when one is a Dummy,
    lambdify renames every argument and rebuilds the node each time (a Min of 35
    operands took 10.5 s, against 0.28 s). A condition is compared with 0, a Piecewise
    pairs its operands, and a Mul raises the base of a factor it divides by to the
    factor's exponent, so a quotient is divided as lambdify prints it: 98/a, where
    98*(1/a) floors one low at a = 49."""
    placeholders = [sympy.Symbol(f"operand{i}") for i in range(len(forms))]
    args = [
        p if form is None else sympy.Ne(p, 0) if form == CONDITION else p**form
        for p, form in zip(placeholders, forms)
    ]
    if func is sympy.Piecewise:
        expr = sympy.Piecewise(*zip(args[0::2], args[1::2]))
    else:
        expr = func(*args)
    return sympy.lambdify(placeholders, expr, modules="numpy")


def _apply(func: Callable[..., Any], arguments: Sequence[np.ndarray]) -> Any:
    """func on its operands' lists. A function that takes scalars only, as
    RelayoutCharge's ``_imp_`` does, raises TypeError or ValueError on arrays and is
    applied point by point."""
    try:
        return np.asarray(func(*arguments), dtype=float)
    except (TypeError, ValueError):
        return np.array([func(*point) for point in zip(*arguments)], dtype=float)


def _values(expr: sympy.Basic, count: int, compute: Callable[[], Any]) -> np.ndarray:
    """expr's list of count values, from compute(). Raises TableLoweringError naming
    expr when evaluating it raises. A value need not be finite here: the printer
    refuses that."""
    with np.errstate(all="ignore"):
        try:
            result = np.asarray(compute(), dtype=float)
        except NameError as exc:
            raise TableLoweringError(expr, f"no numeric value ({exc})") from exc
        except Exception as exc:
            raise TableLoweringError(
                expr, f"no value ({type(exc).__name__}: {exc})"
            ) from exc
    return np.broadcast_to(result, (count,)).astype(float)


def _grid(positions: Sequence[Index]) -> dict[Index, np.ndarray]:
    """Each position's coordinate at every joint assignment of positions, row-major,
    the last fastest."""
    shape = tuple(p.size for p in positions)
    return dict(zip(positions, np.indices(shape).reshape(len(shape), -1)))


# ---------------------------------------------------------------------- the printer


def _divides(expr: sympy.Basic, factor: sympy.Basic) -> bool:
    """Whether factor is a factor of the Mul expr with a negative rational exponent,
    which lambdify prints as a division."""
    return (
        isinstance(expr, sympy.Mul)
        and isinstance(factor, sympy.Pow)
        and bool(factor.exp.is_Rational)
        and bool(factor.exp.is_negative)
    )


def _is_condition(expr: sympy.Basic) -> bool:
    """Whether expr is a condition: a relational, a logical operation, true or false,
    and not a symbol."""
    return isinstance(expr, Boolean) and not isinstance(expr, sympy.Symbol)


class TableLowering(Printer):
    """A sympy Printer whose print of a node is the node's realization. print is its
    entry point. A print method, chosen by the node's sympy class, prints the node's
    operands, asks each for its table within the cap, and returns the node's own
    realization to its parent: here always one table over the node's inputs, on a
    position the printer builds.

    inputs gives each symbol's variable and values; cap bounds the entries of one
    table. The printer holds no model. It keeps each realization by its sympy node,
    so a shared subexpression is printed once, and a printer is for one set of inputs.
    """

    def __init__(self, inputs: Inputs, *, cap: int) -> None:
        super().__init__()
        self.inputs = inputs
        self.cap = cap
        self._realizations: dict[sympy.Basic, AbstractRealization] = {}

    def print(self, expr: sympy.Basic) -> AbstractRealization:
        """expr's realization, printed the first time expr is met and kept. Raises
        TableLoweringError, naming the node, for a node that cannot be lowered."""
        if expr not in self._realizations:
            self._realizations[expr] = self._print(expr)
        return self._realizations[expr]

    def _print_Symbol(self, expr: sympy.Symbol) -> AbstractRealization:
        """A symbol's column: its values on an InputIndex made here on its variable.
        Refuses a symbol with no input, a variable that is not an integer variable or
        has gaps in its domain (Stage 13 adds a position for one), and values that are
        not one finite number per value of the variable."""
        if expr not in self.inputs:
            raise TableLoweringError(expr, "a symbol with no input")
        var, values = self.inputs[expr]
        if not isinstance(var, cp_model.IntVar):
            raise TableLoweringError(
                expr, f"a variable that is not an IntVar ({type(var).__name__})"
            )
        domain = list(var.proto.domain)  # its intervals, flattened: lowest first
        if len(domain) != 2:
            raise TableLoweringError(expr, "a variable whose domain has gaps")
        lo, hi = domain
        if len(values) != hi - lo + 1:
            raise TableLoweringError(
                expr,
                f"{len(values)} values for a variable with {hi - lo + 1}, {lo} to {hi}",
            )
        index = InputIndex(var, size=len(values), lo=lo)
        try:
            column = TableRealization(values=values, index=index)
        except (TypeError, ValueError) as exc:
            raise TableLoweringError(
                expr, f"values that are not one number each ({exc})"
            ) from exc
        return self._finite(expr, column)

    def _print_Piecewise(self, expr: sympy.Piecewise) -> AbstractRealization:
        """Over its arms' values and conditions: an ExprCondPair is not printed."""
        return self._operation(expr, [part for pair in expr.args for part in pair.args])

    def emptyPrinter(self, expr: sympy.Basic) -> AbstractRealization:
        """Any other node, over its arguments."""
        return self._operation(expr, expr.args)

    def _operation(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> TableRealization:
        """expr as one table over its inputs, their positions combined in the order of
        its symbols, the last fastest. A number, true or false is its value on
        UnitIndex. An operation on numbers is one entry on UnitIndex, computed as any
        other operation: sqrt(a) and sqrt(19) are then equal at a = 19, where sympy's
        float(sqrt(19)) is the double below numpy's.

        A table past the cap is refused once its symbols' columns are read, before any
        operand is printed, so nothing under expr is evaluated. Then the operands are
        printed, each refusing a value of its own that is not finite, and expr's list
        is its function applied to their tables, each read at expr's entries
        (Index.on_grid). A condition is read as one, and a Mul reads the base of a
        factor it divides by in the factor's place."""
        if not operands:
            value = TableRealization(values=[_constant(expr)], index=UNIT)
            return self._finite(expr, value)
        symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
        columns = [self.print(symbol).table(self.cap) for symbol in symbols]
        # an operation on numbers has no input: its one entry is on UNIT
        positions = list(dict.fromkeys(column.index for column in columns)) or [UNIT]
        entries = math.prod(position.size for position in positions)
        if entries > self.cap:
            raise TableLoweringError(
                expr,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {self.cap}",
            )
        index = positions[0]
        for position in positions[1:]:
            index = CombineIndex(index, position)
        grid = _grid(positions)
        arguments = []
        forms: list[Form] = []
        for operand in operands:
            # prints the operand, which refuses its own value that is not finite
            table = self.print(operand).table(self.cap)
            form: Form = CONDITION if _is_condition(operand) else None
            if _divides(expr, operand):
                base, form = operand.base, operand.exp  # type: ignore[attr-defined]
                table = self.print(base).table(self.cap)
            # A constant operand is a broadcast of its one entry, under an operation on
            # numbers too: numpy computes x**0.5 as sqrt(x) for a broadcast exponent
            # only, and both sides of a tie must take one path.
            if table.index == index and index != UNIT:
                arguments.append(table.values)
            else:
                at = table.index.on_grid(grid)
                arguments.append(np.broadcast_to(table.values[at], (index.size,)))
            forms.append(form)
        values = _values(
            expr, index.size, lambda: _evaluate(expr, tuple(forms), arguments)
        )
        return self._finite(expr, TableRealization(values=values, index=index))

    def _finite(self, expr: sympy.Basic, table: TableRealization) -> TableRealization:
        """table, refused naming expr when a value of it is not finite."""
        if not table.finite:
            raise TableLoweringError(expr, "a value that is not finite")
        return table


def _constant(expr: sympy.Basic) -> float:
    """The value of a node with no operand: a number, true or false."""
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
