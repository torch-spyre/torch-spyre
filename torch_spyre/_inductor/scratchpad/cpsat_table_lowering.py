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
  as: a ``TableRealization``, or for a sum a ``LinearRealization``. They hold no model;
- the printer: ``TableLowering``, a sympy ``Printer`` whose print of a node is the
  node's realization. It prints a node's operands first, reads their tables, and
  computes the node's list on a position it builds from the node's inputs;
- the build: a realization's ``cp_expr`` rounds it onto ``COST_SCALE`` and adds it to
  the model. Only ``cp_expr`` methods touch the model.

A sum stays linear: its terms are added in the model, so no table crosses two of
them, and a constant factor is applied to a list before it is rounded. Any other node
is one table over its inputs, refused when they do not fit the cap together. Its list
is the numpy core's function of its sympy class (``cost_expr_numpy.py``, which the
CP-SAT printer evaluates its candidates with) applied to its operands' tables; a
class the core has no function for is lambdified. Nothing imports the module yet.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

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


def _rounded(values: Any) -> list[int]:
    """values rounded once onto COST_SCALE. Raises TableLoweringError, naming no node,
    for an entry past MAX_ENTRY."""
    scaled = np.rint(COST_SCALE * np.asarray(values, dtype=float))
    largest = float(np.max(np.abs(scaled)))
    if largest > MAX_ENTRY:
        raise TableLoweringError(
            None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
        )
    return [int(v) for v in scaled]


class AbstractRealization(ABC):
    """What a node is realized as: what the printer returns for it to its parent, and
    what the objective takes of the root. It holds no model."""

    @property
    @abstractmethod
    def finite(self) -> bool:
        """Whether every value it holds is a finite number."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, in COST_SCALE units."""

    @abstractmethod
    def table(self, cap: int) -> TableRealization:
        """The realization as one table, for a parent that computes its own list from
        it. cap bounds the entries of a table that has to be made."""

    @abstractmethod
    def scaled(self, factor: float) -> AbstractRealization:
        """factor times the realization, applied in float before any rounding."""


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
        return bool(np.all(np.isfinite(self.values)))

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The list rounded once onto COST_SCALE. A constant list is a number, and one
        that is affine in its position is its first entry plus a step times the
        position: neither adds a table. Any other is one add_element on its position.
        Raises TableLoweringError, naming no node, for an entry past MAX_ENTRY."""
        entries = _rounded(self.values)
        step = entries[1] - entries[0] if len(entries) > 1 else 0
        if all(entry == entries[0] + i * step for i, entry in enumerate(entries)):
            if step == 0:
                return entries[0]
            return entries[0] + step * self.index.cp_expr(model)
        target = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(entries))), "table"
        )
        model.add_element(self.index.cp_expr(model), entries, target)
        return target

    def table(self, cap: int) -> TableRealization:
        """Itself: none is made, so cap is not read."""
        return self

    def scaled(self, factor: float) -> TableRealization:
        with np.errstate(over="ignore"):  # inf is refused by the printer
            return TableRealization(values=factor * self.values, index=self.index)


@dataclass(frozen=True, eq=False)
class LinearRealization(AbstractRealization):
    """A sum kept linear: constant plus terms, each term a list on a position of its
    own. positions are the sum's input positions, in the order its table has them.
    Compared by identity."""

    constant: float
    terms: tuple[TableRealization, ...]
    positions: tuple[Index, ...]

    @classmethod
    def of(
        cls, operands: Sequence[AbstractRealization], positions: Sequence[Index]
    ) -> LinearRealization:
        """The sum of operands, over positions. A sum among the operands is taken
        term by term, a constant goes into the constant, and lists on one position
        are added: all in float, before any rounding."""
        constant = 0.0
        tables: list[TableRealization] = []
        for operand in operands:
            if isinstance(operand, LinearRealization):
                constant += operand.constant
                tables.extend(operand.terms)
            elif isinstance(operand, TableRealization):
                tables.append(operand)
            else:
                raise TypeError(f"a sum cannot hold a {type(operand).__name__}")
        lists: dict[Index, TableRealization] = {}
        for table in tables:
            if table.index == UNIT:
                constant += float(table.values[0])
            elif table.index in lists:
                with np.errstate(over="ignore"):  # inf is refused by the printer
                    values = lists[table.index].values + table.values
                lists[table.index] = TableRealization(values=values, index=table.index)
            else:
                lists[table.index] = table
        return cls(constant, tuple(lists.values()), tuple(positions))

    @property
    def finite(self) -> bool:
        return math.isfinite(self.constant) and all(term.finite for term in self.terms)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """Its constant, rounded, plus each term's cp_expr: a number when no term adds
        to the model."""
        total = _rounded([self.constant])[0]
        exprs = []
        for term in self.terms:
            expr = term.cp_expr(model)
            if isinstance(expr, int):
                total += expr
            else:
                exprs.append(expr)
        return cp_model.LinearExpr.sum(exprs) + total if exprs else total

    def table(self, cap: int) -> TableRealization:
        """Its constant plus its terms' lists, as one table over its positions. Raises
        TableLoweringError, naming no node, when that table would pass cap or holds a
        value that is not finite."""
        entries = math.prod(position.size for position in self.positions)
        if entries > cap:
            raise TableLoweringError(
                None,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {cap}",
            )
        index = _combined(self.positions)
        grid = _grid(self.positions)
        values = np.full(index.size, self.constant)
        with np.errstate(over="ignore"):  # inf is refused below
            for term in self.terms:
                values = values + _on(term, index, grid)
        table = TableRealization(values=values, index=index)
        if not table.finite:
            raise TableLoweringError(None, "a value that is not finite")
        return table

    def scaled(self, factor: float) -> LinearRealization:
        terms = tuple(term.scaled(factor) for term in self.terms)
        return LinearRealization(factor * self.constant, terms, self.positions)


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


def _combined(positions: Sequence[Index]) -> Index:
    """positions as one position, the last fastest; UNIT for none."""
    index = positions[0] if positions else UNIT
    for position in positions[1:]:
        index = CombineIndex(index, position)
    return index


def _grid(positions: Sequence[Index]) -> dict[Index, np.ndarray]:
    """Each position's coordinate at every joint assignment of positions, row-major,
    the last fastest."""
    shape = tuple(p.size for p in positions)
    return dict(zip(positions, np.indices(shape).reshape(len(shape), -1)))


def _on(
    table: TableRealization, index: Index, grid: Mapping[Index, np.ndarray]
) -> np.ndarray:
    """table's list at every entry of index, where grid holds the coordinates of
    index's input positions: as it is when table is on index, and otherwise read at
    table's own position (Index.on_grid)."""
    if table.index == index:
        return table.values
    at = table.index.on_grid(grid)
    return np.broadcast_to(table.values[at], (index.size,))


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


_Realization = TypeVar("_Realization", bound=AbstractRealization)


class TableLowering(Printer):
    """A sympy Printer whose print of a node is the node's realization. print is its
    entry point. A print method, chosen by the node's sympy class, prints the node's
    operands and returns the node's own realization to its parent. A sum is its
    operands' realizations, kept linear, and a constant times one operand is that
    operand's realization with the constant applied. Any other node asks each operand
    for its table within the cap and is one table over the node's inputs, on a
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

    def _print_Add(self, expr: sympy.Add) -> AbstractRealization:
        """A sum, kept linear: its operands' realizations, with no table across two
        of them."""
        if not expr.free_symbols:
            return self._operation(expr, expr.args)
        operands = [self.print(operand) for operand in expr.args]
        return self._finite(expr, LinearRealization.of(operands, self._inputs(expr)))

    def _print_Mul(self, expr: sympy.Mul) -> AbstractRealization:
        """A constant times one operand: the operand's realization with the constant
        applied in float, so it is rounded once. Any other product is one table, and
        so is a constant over an operand, which is divided."""
        constants = [factor for factor in expr.args if not factor.free_symbols]
        operands = [factor for factor in expr.args if factor.free_symbols]
        if len(operands) != 1 or _divides(expr, operands[0]):
            return self._operation(expr, expr.args)
        coefficient = math.prod(float(self._table(c).values[0]) for c in constants)
        return self._finite(expr, self.print(operands[0]).scaled(coefficient))

    def _print_Piecewise(self, expr: sympy.Piecewise) -> AbstractRealization:
        """Over its arms' values and conditions: an ExprCondPair is not printed."""
        return self._operation(expr, [part for pair in expr.args for part in pair.args])

    def emptyPrinter(self, expr: sympy.Basic) -> AbstractRealization:
        """Any other node, over its arguments."""
        return self._operation(expr, expr.args)

    def _operation(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> TableRealization:
        """expr as one table: its value on UnitIndex when it has no free symbol, and
        otherwise one table over its inputs, their positions combined in the order of
        its symbols, the last fastest.

        A table past the cap is refused before anything under expr is printed, so it
        is never evaluated. Then the operands are printed, each refusing a value of its
        own that is not finite, and expr's list is its function applied to their
        tables, each read at expr's entries (Index.on_grid). A condition is read as
        one, and a Mul reads the base of a factor it divides by in the factor's
        place."""
        if not expr.free_symbols:
            value = TableRealization(values=[_constant(expr)], index=UNIT)
            return self._finite(expr, value)
        positions = self._inputs(expr)
        entries = math.prod(position.size for position in positions)
        if entries > self.cap:
            raise TableLoweringError(
                expr,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {self.cap}",
            )
        index = _combined(positions)
        grid = _grid(positions)
        arguments = []
        forms: list[Form] = []
        for operand in operands:
            # prints the operand, which refuses its own value that is not finite
            table = self._table(operand)
            form: Form = CONDITION if _is_condition(operand) else None
            if _divides(expr, operand):
                table, form = self._table(operand.base), operand.exp  # type: ignore[attr-defined]
            arguments.append(_on(table, index, grid))
            forms.append(form)
        values = _values(
            expr, index.size, lambda: _evaluate(expr, tuple(forms), arguments)
        )
        return self._finite(expr, TableRealization(values=values, index=index))

    def _inputs(self, expr: sympy.Basic) -> list[Index]:
        """The positions of expr's inputs, in the order of its symbols."""
        symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
        return list(dict.fromkeys(self._table(symbol).index for symbol in symbols))

    def _table(self, expr: sympy.Basic) -> TableRealization:
        """expr's table within the cap, for a node that computes its list from it.
        A sum makes its table here, and a refusal of that names expr."""
        try:
            return self.print(expr).table(self.cap)
        except TableLoweringError as exc:
            if exc.node is not None:
                raise
            raise TableLoweringError(expr, exc.reason) from exc

    def _finite(self, expr: sympy.Basic, realization: _Realization) -> _Realization:
        """realization, refused naming expr when a value of it is not finite."""
        if not realization.finite:
            raise TableLoweringError(expr, "a value that is not finite")
        return realization


def _constant(expr: sympy.Basic) -> float:
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
