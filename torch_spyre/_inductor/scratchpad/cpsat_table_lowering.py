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
works out which entry of a table applies. A node's value is a *table*: a float list
``values`` and an ``Index``, the position in it that the decision variables fix,
``value == values[index]``. The lowering has three layers:

- values: ``Index``, a position, and ``AbstractRealization``, what a node is realized
  as, with its one kind here, ``TableRealization``, hold no model;
- the printer: ``TableLowering`` is a sympy ``Printer`` whose print of a node is the
  node's realization. A print method, dispatched on the node's sympy class, prints the
  node's operands first, asks each for its table, builds the node's own position and
  computes its list, and returns the realization to the node's parent. The printer
  makes the positions: a symbol's on its variable, and an operation's from those of
  its inputs. It refuses a node whose table would pass the cap before it prints
  anything under it, and a value that is not finite at the node that makes it,
  whether or not its parent's value depends on it. It holds no model;
- the build: the root's realization builds itself into the model
  (``TableRealization.cp_expr``), rounded onto ``COST_SCALE``, its position built by
  ``Index.cp_expr``. Only ``cp_expr`` methods touch the model.

One table per node, and the root read as one lookup. A node's list is its function
applied to its operands' lists, each read at every entry of the node's table. The
function is the node's sympy class over placeholders, lambdified when the node is
printed (``_lambdified``): nothing is kept once a lowering is done. It lowers an
expression whose inputs fit the cap together and refuses anything wider. Nothing imports
it yet.
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

# One cost unit is COST_SCALE objective units: a table entry is
# rint(COST_SCALE * value).
COST_SCALE = 1024
# The largest table entry: a sum of thousands of entries stays inside int64.
MAX_ENTRY = 2**50

# What CP-SAT takes as a linear expression: a LinearExpr, an IntVar or an int.
CpExpr: TypeAlias = cp_model.LinearExprT


class TableLoweringError(Exception):
    """An objective the tables cannot lower. ``node`` is the sympy node left with no
    realization, or None for a refusal outside the walk; ``reason`` says why.

    It subclasses none of RuntimeError, TypeError and ValueError, so an ``except``
    around the objective that catches those cannot turn it into a different solve.
    """

    def __init__(self, node: sympy.Basic | None, reason: str) -> None:
        super().__init__(reason if node is None else f"{reason}: {node}")
        self.node = node
        self.reason = reason


# --------------------------------------------------------------------------- values


class Index(ABC):
    """A position: a map from the inputs' values to 0 .. size - 1, as a CP-SAT
    expression. Each kind of step is a subclass that holds its parts, the positions it
    is made from, as its own fields, so a step depends on Index alone. A value, with
    no model in it. Steps are frozen dataclasses: two that make the same position from
    equal parts are equal and hash alike. TableRealization and the printer do not
    change when a step is added. The steps here are InputIndex, UnitIndex and
    CombineIndex."""

    # The number of positions, 0 .. size - 1: a field of each step, given or
    # computed from its parts when it is made.
    size: int

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The position as an expression in model, from its parts' cp_expr. A leaf over
        a variable reads the variable. Each call builds anew, and a part reached twice
        is built twice: no step of this stage adds to the model."""

    @abstractmethod
    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        """The position at every point of a grid, from its parts' on_grid: what
        cp_expr builds, computed. grid holds the coordinate of each input's position at
        every point, and a leaf over a variable reads its own."""


@dataclass(frozen=True)
class InputIndex(Index):
    """A leaf: a variable's own position, v -> v - lo, for a variable whose values run
    from lo to lo + size - 1. The printer makes one on a symbol's variable, which is
    already in the model. Two on one variable are equal: one position."""

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
    """(left, right) -> left * right.size + right. right.size is a constant, so this
    is linear: no two variables are multiplied."""

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
    """What a node is realized as: what the printer returns for the node to the
    node's parent, and what the objective takes of the root. A value, with no model in
    it, that builds its own CP-SAT side. Each kind of realization is a subclass: here
    a TableRealization, and from Stage 5 a LinearRealization too, what a sum is
    realized as."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, as a CP-SAT expression in COST_SCALE units."""

    @abstractmethod
    def table(self) -> TableRealization:
        """The realization as one table: what a parent asks for when it computes its
        own list from this one."""


@dataclass(frozen=True, eq=False, kw_only=True)
class TableRealization(AbstractRealization):
    """A node's table: values[index], made together and never apart. values is the
    value at each position, float until cp_expr rounds it, stored as a read-only
    copy; a list whose length is not the index's size is refused. Frozen, with no
    model in it, and compared by identity, its list being an array."""

    values: np.ndarray
    index: Index

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _read_only(self.values, self.index))

    @property
    def finite(self) -> bool:
        """Whether every value in its list is a finite number."""
        return bool(np.all(np.isfinite(self.values)))

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The table in model, values[index] in COST_SCALE units: the list rounded
        once, within half a unit of its float value, then a number when it is
        constant, and otherwise one add_element on its index's position into a new
        variable whose domain is the rounded values. The position is built only when
        the table is not constant, so a constant table adds nothing to the model.
        Raises TableLoweringError, naming no node, for an entry past MAX_ENTRY. Its
        values are finite: the printer refuses a table that has one that is not."""
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

    def table(self) -> TableRealization:
        """Itself."""
        return self


# Each symbol's input: the variable that picks its value, and the symbol's value at
# each of that variable's values, from its lowest to its highest.
Input: TypeAlias = tuple[cp_model.IntVar, Sequence[float]]
Inputs: TypeAlias = Mapping[sympy.Symbol, Input]


# --------------------------------------------------------------------- a node's list


# The form of an operand in its node's function: as it is (None), a condition
# (CONDITION), or the base of a Mul's factor with a negative rational exponent, which
# the function raises to that exponent (the exponent, a sympy.Rational).
CONDITION = "condition"
Form: TypeAlias = "str | sympy.Rational | None"


def _lambdified(func: Any, forms: tuple[Form, ...]) -> Callable[..., Any]:
    """The function of a node of sympy class func whose operands take forms,
    lambdified with the numpy module over one placeholder per operand. The printer
    calls it once for a node, when it prints it, and nothing keeps the function after
    that.

    The placeholders are plain symbols, operand0, operand1, ..., whose names are
    Python identifiers, so lambdify takes them as they are. It renames every argument
    when any is a Dummy, rebuilding the node once per argument: a Min of 35 operands
    took 10.5 s, against 0.28 s here. The expression holds nothing but placeholders,
    so no name of its own can clash with theirs.

    A condition is compared with 0, as a table holds it as 1.0 or 0.0. A Piecewise
    pairs its operands, value then condition. A Mul raises the base of a factor with
    a negative rational exponent to it, so a quotient is divided, as lambdify prints
    it: 98/a, where 98*(1/a) is 1.9999999999999998 at a = 49."""
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
    """func on its operands' lists, of one length. A function that takes scalars only,
    as RelayoutCharge's ``_imp_`` does, is applied point by point: on arrays it raises
    TypeError or, where it tests a value with ``if``, ValueError."""
    try:
        return np.asarray(func(*arguments), dtype=float)
    except (TypeError, ValueError):
        return np.array([func(*point) for point in zip(*arguments)], dtype=float)


def _values(expr: sympy.Basic, count: int, compute: Callable[[], Any]) -> np.ndarray:
    """expr's list of count values, from compute(). Raises TableLoweringError naming
    expr when it has no numeric value, and for anything else evaluating it raises. A
    value returned here need not be finite: a division by zero is infinite, and a
    Piecewise with no arm true is NaN. The printer refuses the list, naming expr
    (TableLowering._finite)."""
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
    the last position fastest: entry i1 * s1 + ... + im * sm, s_k the product of the
    sizes after k."""
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
    """Whether expr is a condition: a relational, a logical operation, true or false.
    A symbol is not one."""
    return isinstance(expr, Boolean) and not isinstance(expr, sympy.Symbol)


class TableLowering(Printer):
    """Lowers cost expressions to tables: a sympy Printer whose print of a node is the
    node's realization, an AbstractRealization. print, its public method, prints any
    node. A Printer dispatches on the node's sympy class, the nearest class along its
    MRO that has a print method, and falls back to emptyPrinter. A print method prints
    the node's operands first, which returns their realizations, asks each for its
    table, and returns the node's own realization to the node's parent: in this stage
    always one table over the node's inputs. The printer makes the positions itself:
    an InputIndex on a symbol's variable, and for an operation its inputs' positions
    combined.

    inputs gives each symbol's variable and values; cap bounds the entries of one
    table, the length of its list. The printer holds no model, so lowering cannot
    write to one. It keeps the realization of each node it has printed, by the sympy
    node: a subexpression shared by two parents, or by two calls of print, is printed
    once, and its parents read one realization. So inputs and cap are read when a node
    is first printed: a printer is made for one set of inputs, and a change to them
    afterwards does not reach a realization it has kept.
    """

    def __init__(self, inputs: Inputs, *, cap: int) -> None:
        super().__init__()
        self.inputs = inputs
        self.cap = cap
        self._realizations: dict[sympy.Basic, AbstractRealization] = {}

    def lower(self, root: sympy.Basic) -> AbstractRealization:
        """root's realization, with every node under root realized on the way; root
        is not changed. Raises TableLoweringError, naming the node, for a node that
        cannot be lowered. Its cp_expr builds it into a model."""
        return self.print(root)

    def print(self, expr: sympy.Basic) -> AbstractRealization:
        """expr's realization: printed the first time expr is met, by the print
        method of its sympy class, and kept. A print method prints its node's
        operands with it, and a caller any node."""
        if expr not in self._realizations:
            self._realizations[expr] = self._print(expr)
        return self._realizations[expr]

    def _print_Symbol(self, expr: sympy.Symbol) -> AbstractRealization:
        """A symbol's column: its values on its variable's position, an InputIndex
        made here from the variable's lowest and highest values, so the caller makes
        no position. Raises TableLoweringError naming the symbol when it has no input,
        when its variable is not an integer variable or its domain has gaps (a
        position lookup for one comes with Stage 13), when its values are not one
        number per value of its variable, and when one is not finite."""
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
        """Its operands are its arms' values and conditions, value then condition: an
        arm's ExprCondPair is not printed."""
        return self._operation(expr, [part for pair in expr.args for part in pair.args])

    def emptyPrinter(self, expr: sympy.Basic) -> AbstractRealization:
        """Any other node, over its arguments."""
        return self._operation(expr, expr.args)

    def _operation(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> TableRealization:
        """expr as one table: its value on the UnitIndex position when it has no free
        symbol (a number, true or false, or an operation on numbers), and otherwise
        one table over its inputs, their positions combined in the order of its
        symbols sorted by sympy.default_sort_key, the last fastest.

        Raises TableLoweringError naming expr when that table would pass the cap,
        before anything under expr is printed, so a table past the cap is never
        evaluated. Under the cap it prints its operands first, so an operand with a
        value that is not finite is refused at the operand, naming it, whether or not
        expr's value depends on it; then expr is refused if a value of its own is not
        finite.

        Its list is its function (_lambdified) applied to its operands' tables, each
        read at every entry of its own: an operand on expr's own position as it is,
        and one on fewer positions at its own position (Index.on_grid), a constant at
        every entry. Two symbols on one position move together. A condition is read
        as one, and in a Mul the base of a factor with a negative rational exponent
        is read in the factor's place, with the exponent: the one node read that is
        not an operand."""
        if not expr.free_symbols:
            value = TableRealization(values=[_constant(expr)], index=UNIT)
            return self._finite(expr, value)
        symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
        columns = [self.print(symbol).table() for symbol in symbols]
        positions = list(dict.fromkeys(column.index for column in columns))
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
            table = self.print(operand).table()
            form: Form = CONDITION if _is_condition(operand) else None
            if _divides(expr, operand):
                table, form = self.print(operand.base).table(), operand.exp  # type: ignore[attr-defined]
            if table.index == index:
                arguments.append(table.values)
            else:
                at = table.index.on_grid(grid)
                arguments.append(np.broadcast_to(table.values[at], (index.size,)))
            forms.append(form)
        values = _values(
            expr,
            index.size,
            lambda: _apply(_lambdified(expr.func, tuple(forms)), arguments),
        )
        return self._finite(expr, TableRealization(values=values, index=index))

    def _finite(self, expr: sympy.Basic, table: TableRealization) -> TableRealization:
        """table, refused naming expr when a value of it is not finite."""
        if not table.finite:
            raise TableLoweringError(expr, "a value that is not finite")
        return table


def _constant(expr: sympy.Basic) -> float:
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
