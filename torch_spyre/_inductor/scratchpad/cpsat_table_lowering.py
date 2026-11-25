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
  as: a list on a position or a linear form of its operands;
- the printer: ``TableLowering``, a sympy ``Printer`` whose print of a node is the
  node's realization. It prints a node's operands first, reads their tables, and
  computes the node's list on a position it builds from the node's inputs;
- the build: a realization's ``cp_expr`` rounds it onto ``COST_SCALE`` and adds it to
  the model. Each position and realization keeps its expression in the model it was
  last built in, so one held by several readers is built once. Only ``cp_expr``
  methods touch the model.

A sum stays linear, so no table crosses two of its terms. A product of several
operands is a switch on one of them where that has fewer entries than its one table,
so no table crosses that factor and the rest. Any other node is one table over its
inputs, refused when they do not fit the cap together. Its list
is the numpy core's function of its sympy class (``cost_expr_numpy.py``, which the
CP-SAT printer evaluates its candidates with) applied to its operands' tables; a
class the core has no function for is refused. Nothing imports the module yet.
"""

from __future__ import annotations

import math
import sys
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
# A switch writes a fraction p / q, q <= MAX_DENOMINATOR, each p * w < MAX_PRODUCT.
MAX_DENOMINATOR = 2**24
MAX_PRODUCT = 2**60

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
    position."""

    size: int  # the number of positions
    _built: tuple[cp_model.CpModel, CpExpr] | None = None  # not a dataclass field

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The position in model, built once per model."""
        built = self._built
        if built is None or built[0] is not model:
            built = (model, self.build(model))
            object.__setattr__(self, "_built", built)
        return built[1]

    @abstractmethod
    def build(self, model: cp_model.CpModel) -> CpExpr:
        """The position in model, its parts read through their cp_expr."""

    def entries(self) -> int:
        """The entries of its own lookup."""
        return 0

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

    def build(self, model: cp_model.CpModel) -> CpExpr:
        return self.var - self.lo if self.lo else self.var

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return grid[self]


@dataclass(frozen=True)
class UnitIndex(Index):
    """A leaf: the one position of a constant, on no input."""

    size: int = field(default=1, init=False)

    def build(self, model: cp_model.CpModel) -> CpExpr:
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

    def build(self, model: cp_model.CpModel) -> CpExpr:
        left = self.left.cp_expr(model)
        return left * self.right.size + self.right.cp_expr(model)

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return self.left.on_grid(grid) * self.right.size + self.right.on_grid(grid)


@dataclass(frozen=True)
class CompactIndex(Index):
    """remap[parent]: one lookup onto fewer positions, as an entry's value's rank."""

    parent: Index
    remap: tuple[int, ...]
    size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", max(self.remap) + 1)

    def entries(self) -> int:
        return len(self.remap)

    def build(self, model: cp_model.CpModel) -> CpExpr:
        rank = model.new_int_var(0, self.size - 1, "rank")
        model.add_element(self.parent.cp_expr(model), list(self.remap), rank)
        return rank

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return np.asarray(self.remap)[self.parent.on_grid(grid)]


UNIT = UnitIndex()


def _read_only(values: Any, index: Index) -> np.ndarray:
    """values as a read-only list of index's size: bool when they are bool, as a
    condition's are, and float otherwise."""
    array = np.asarray(values)
    array = np.array(array, dtype=bool if array.dtype == bool else float)
    if array.shape != (index.size,):
        raise ValueError(
            f"a list of shape {array.shape} on an index of size {index.size}"
        )
    array.setflags(write=False)
    return array


def _rounded(values: Any) -> list[int]:
    """values rounded onto COST_SCALE."""
    scaled = np.rint(COST_SCALE * np.asarray(values, dtype=float))
    largest = float(np.max(np.abs(scaled)))
    if largest > MAX_ENTRY:
        raise TableLoweringError(
            None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
        )
    return [int(v) for v in scaled]


class AbstractRealization(ABC):
    """What the printer returns for a node. It holds only finite values."""

    _built: tuple[cp_model.CpModel, CpExpr] | None = None  # not a dataclass field

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, built once per model: a realization that several
        readers hold is one expression."""
        built = self._built
        if built is None or built[0] is not model:
            built = (model, self.build(model))
            object.__setattr__(self, "_built", built)
        return built[1]

    @abstractmethod
    def build(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, in COST_SCALE units, its operands read through
        their cp_expr."""

    @abstractmethod
    def table(self, cap: int) -> TableRealization:
        """The realization as one table, for a parent that computes its own list from
        it. cap bounds the entries of a table that has to be made."""

    @abstractmethod
    def scaled(self, factor: float) -> AbstractRealization:
        """factor times the realization, applied in float before any rounding."""

    @abstractmethod
    def bounds(self) -> tuple[int, int]:
        """Its lowest and highest value in a model."""

    @abstractmethod
    def entries(self) -> int:
        """The table entries it adds to a model, which a node's choice weighs."""


@dataclass(frozen=True, eq=False, kw_only=True)
class TableRealization(AbstractRealization):
    """values[index]: a read-only list, one value per entry of index: bool for a
    condition, float for any other node."""

    values: np.ndarray
    index: Index

    def __post_init__(self) -> None:
        values = _read_only(self.values, self.index)
        if not np.all(np.isfinite(values)):
            raise TableLoweringError(None, "a value that is not finite")
        object.__setattr__(self, "values", values)

    def build(self, model: cp_model.CpModel) -> CpExpr:
        """A constant or affine list adds no table; any other is one add_element."""
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
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            return TableRealization(values=factor * self.values, index=self.index)

    def bounds(self) -> tuple[int, int]:
        entries = _rounded(self.values)
        return min(entries), max(entries)

    def entries(self) -> int:
        """Its list, counted before the rounding: none when it is one value."""
        return 0 if np.all(self.values == self.values[0]) else len(self.values)


class SymbolicRealization(AbstractRealization):
    """No list of its own: CP-SAT builds it from its operands' expressions. positions
    are its inputs', in its table's order."""

    positions: tuple[Index, ...]

    def table(self, cap: int) -> TableRealization:
        """An operand's inputs are among its own, so an operand's table fits when its
        own does."""
        entries = math.prod(position.size for position in self.positions)
        if entries > cap:
            raise TableLoweringError(
                None,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {cap}",
            )
        index = _combined(self.positions)
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            values = self._list(index, _grid(self.positions), cap)
        return TableRealization(values=values, index=index)

    @abstractmethod
    def _list(
        self, index: Index, grid: Mapping[Index, np.ndarray], cap: int
    ) -> np.ndarray:
        """Its value at each entry of index, from its operands' tables."""


@dataclass(frozen=True, eq=False)
class SumRealization(SymbolicRealization):
    """sum(terms). A constant is a term on UNIT."""

    terms: tuple[AbstractRealization, ...]
    positions: tuple[Index, ...]

    def build(self, model: cp_model.CpModel) -> CpExpr:
        total = 0
        exprs = []
        for term in self.terms:
            expr = term.cp_expr(model)
            if isinstance(expr, int):
                total += expr
            else:
                exprs.append(expr)
        return cp_model.LinearExpr.sum(exprs) + total if exprs else total

    def _list(
        self, index: Index, grid: Mapping[Index, np.ndarray], cap: int
    ) -> np.ndarray:
        values = np.zeros(index.size)
        for term in self.terms:
            values = values + _on(term.table(cap), index, grid)
        return values

    def scaled(self, factor: float) -> SumRealization:
        terms = tuple(term.scaled(factor) for term in self.terms)
        return SumRealization(terms, self.positions)

    def bounds(self) -> tuple[int, int]:
        ends = [term.bounds() for term in self.terms]
        return sum(end[0] for end in ends), sum(end[1] for end in ends)

    def entries(self) -> int:
        return sum(term.entries() for term in self.terms)


@dataclass(frozen=True, eq=False)
class SwitchRealization(SymbolicRealization):
    """A product as factor * rest, the factor's value values[index] (rule 4). The rest
    is collapsed once into w, and one true/false per value, on where the factor takes
    it, sets g to the value times w. No table crosses factor and rest."""

    values: tuple[float, ...]
    index: Index
    rest: AbstractRealization
    positions: tuple[Index, ...]

    def build(self, model: cp_model.CpModel) -> CpExpr:
        """A whole value is g == v * w; a fraction, p / q with q = denominator(), is
        the floor: q * g <= p * w <= q * g + q - 1."""
        w = self.rest.cp_expr(model)
        q = self.denominator()
        cases = [model.new_bool_var("case") for _ in self.values]
        model.add_map_domain(self.index.cp_expr(model), cases)
        g = model.new_int_var(*self.bounds(), "switch")
        for value, case in zip(self.values, cases):
            if value.is_integer():
                model.add(g == int(value) * w).only_enforce_if(case)
            else:
                p = round(value * q)
                model.add(q * g <= p * w).only_enforce_if(case)
                model.add(p * w <= q * g + q - 1).only_enforce_if(case)
        return g

    def _list(
        self, index: Index, grid: Mapping[Index, np.ndarray], cap: int
    ) -> np.ndarray:
        factor = np.asarray(self.values)[self.index.on_grid(grid)]
        return factor * _on(self.rest.table(cap), index, grid)

    def scaled(self, factor: float) -> SwitchRealization:
        rest = self.rest.scaled(factor)
        return SwitchRealization(self.values, self.index, rest, self.positions)

    def bounds(self) -> tuple[int, int]:
        q = self.denominator()
        ends = [
            int(value) * end if value.is_integer() else round(value * q) * end // q
            for value in self.values
            for end in self.rest.bounds()
        ]
        return min(ends), max(ends)

    def entries(self) -> int:
        return self.rest.entries() + len(self.values) + self.index.entries()

    def denominator(self) -> int:
        """q: 1 when every value is whole, otherwise the largest power of two up to
        MAX_DENOMINATOR that keeps every p * w under MAX_PRODUCT."""
        widest = max(abs(end) for end in self.rest.bounds())
        q = 1 if all(v.is_integer() for v in self.values) else MAX_DENOMINATOR
        while max(abs(round(v * q)) for v in self.values) * widest >= MAX_PRODUCT:
            if q == 1:
                raise TableLoweringError(None, "a switched product past 2^60")
            q //= 2
        return q


# --------------------------------------------------------------------- a node's list


def _values(expr: sympy.Basic, count: int, compute: Callable[[], Any]) -> np.ndarray:
    """expr's list of count values, from compute(): bool for a condition, float
    otherwise. A RecursionError is left to TableLowering.print."""
    dtype = bool if _is_condition(expr) else float
    with np.errstate(all="ignore"):
        try:
            result = np.asarray(compute(), dtype=dtype)
        except RecursionError:
            raise
        except Exception as exc:
            raise TableLoweringError(
                expr, f"no value ({type(exc).__name__}: {exc})"
            ) from exc
    return np.broadcast_to(result, (count,)).astype(dtype)


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
    """table's list at each entry of index; grid holds index's input coordinates."""
    # On UNIT too a constant is broadcast: numpy computes x**0.5 as sqrt(x) only for a
    # broadcast exponent, and both sides of a tie must take one path.
    if table.index == index and index != UNIT:
        return table.values
    at = table.index.on_grid(grid)
    return np.broadcast_to(table.values[at], (index.size,))


# ---------------------------------------------------------------------- the printer


def _divides(factor: sympy.Basic) -> bool:
    """Whether factor, a factor of a Mul, has a negative rational exponent, which
    lambdify prints as a division."""
    return (
        isinstance(factor, sympy.Pow)
        and bool(factor.exp.is_Rational)
        and bool(factor.exp.is_negative)
    )


def _is_condition(expr: sympy.Basic) -> bool:
    """Whether expr is a condition: a relational, a logical operation, true or false,
    and not a symbol."""
    return isinstance(expr, Boolean) and not isinstance(expr, sympy.Symbol)


class TableLowering(Printer):
    """A sympy Printer whose print of a node is its realization. A print method
    returns None to decline a node, which is then one table over its inputs.

    inputs gives each symbol's variable and the symbol's value at each of the
    variable's values, lowest first; cap bounds one table's entries.
    A printer is for one set of inputs: it keeps each node's realization, so a shared
    subexpression is printed once.
    """

    def __init__(
        self,
        inputs: Mapping[sympy.Symbol, tuple[cp_model.IntVar, Sequence[float]]],
        *,
        cap: int,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.cap = cap
        self._realizations: dict[sympy.Basic, AbstractRealization] = {}
        # one instance per rank, so equal ranks are one lookup
        self._ranks: dict[Index, Index] = {}

    def print(self, expr: sympy.Basic) -> AbstractRealization:
        """expr's realization; raises TableLoweringError for what cannot be lowered."""
        try:
            return self._print(expr)
        except RecursionError as exc:
            # No node: printing one this deep into the message would recurse too.
            raise TableLoweringError(
                None,
                "an expression nested too deep to print within the recursion limit "
                f"of {sys.getrecursionlimit()}",
            ) from exc

    def _print(self, expr: sympy.Basic, **kwargs: Any) -> AbstractRealization:
        """sympy's dispatch, kept by node. A declined node is one table."""
        if expr not in self._realizations:
            try:
                realization = super()._print(expr, **kwargs)
                if realization is None:
                    realization = self._operation(expr, expr.args)
            except TableLoweringError as exc:
                if exc.node is not None:
                    raise
                raise TableLoweringError(expr, exc.reason) from exc
            self._realizations[expr] = realization
        return self._realizations[expr]

    def doprint(self, expr: sympy.Basic) -> str:
        """Refused: this printer's print of a node is a realization, not a string."""
        raise NotImplementedError("TableLowering prints realizations: call print")

    def _print_Symbol(self, expr: sympy.Symbol) -> AbstractRealization:
        """A symbol's column on its variable. A domain with gaps waits for Stage 13."""
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
        return column

    def _print_Add(self, expr: sympy.Add) -> AbstractRealization | None:
        """A sum kept linear."""
        if not expr.free_symbols:
            return None
        operands = [self._print(operand) for operand in expr.args]
        return _sum(operands, tuple(self._inputs(expr)))

    def _print_Mul(self, expr: sympy.Mul) -> AbstractRealization | None:
        """A product of several operands is a switch (_switch) on the one with the
        fewest distinct values among those whose table fits the cap, the first in
        expr's order on a tie, over the others' product, its constants applied to that
        rest. Otherwise a quotient is one table (_quotient), a constant times one
        operand is the operand scaled in float, and any other product is declined."""
        constants = [factor for factor in expr.args if not factor.free_symbols]
        operands = [factor for factor in expr.args if factor.free_symbols]
        switch: AbstractRealization | None = None
        if len(operands) > 1:
            tables = {
                operand: self._table(operand)
                for operand in operands
                if math.prod(p.size for p in self._inputs(operand)) <= self.cap
            }
            if tables:
                factor = min(tables, key=lambda o: len(np.unique(tables[o].values)))
                values, index = _ranked(tables[factor])
                index = self._ranks.setdefault(index, index)
                rest = self._print(sympy.Mul(*(o for o in operands if o != factor)))
                positions = tuple(self._inputs(expr))
                switch = _switch(values, index, rest, positions, self.cap)
        if switch is None and any(_divides(factor) for factor in expr.args):
            return self._quotient(expr)
        realization = self._print(operands[0]) if len(operands) == 1 else switch
        if realization is None or not constants:
            return realization
        coefficient = math.prod(float(self._table(c).values[0]) for c in constants)
        return realization.scaled(coefficient)

    def _quotient(self, expr: sympy.Mul) -> TableRealization:
        """expr as one table: the product of its other factors divided by the product
        of the bases of its factors with a negative rational exponent, each raised to
        minus that exponent, as lambdify prints a quotient: 98/a, where 98*(1/a)
        floors one low at a = 49. A factor it divides by is still printed, refusing
        its own value that is not finite."""
        index, grid = self._position(expr)
        numerator: list[np.ndarray] = []
        denominator: list[tuple[np.ndarray, float]] = []
        for factor in expr.args:
            if _divides(factor):
                self._table(factor)
                base, exp = factor.args
                denominator.append((_on(self._table(base), index, grid), float(-exp)))
            else:
                numerator.append(_on(self._table(factor), index, grid))
        values = _values(
            expr,
            index.size,
            lambda: math.prod(numerator) / math.prod(b**e for b, e in denominator),
        )
        return TableRealization(values=values, index=index)

    def _print_Piecewise(self, expr: sympy.Piecewise) -> AbstractRealization:
        """Over its arms' values and conditions: an ExprCondPair is not printed."""
        return self._operation(expr, [part for pair in expr.args for part in pair.args])

    def emptyPrinter(self, expr: sympy.Basic) -> None:
        """Any other node is declined. sympy's own returns str(expr)."""
        return None

    def _operation(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> TableRealization:
        """expr as one table over its inputs, in symbol order. An operation on numbers
        is computed the same way, on UNIT, so sqrt(19) equals sqrt(a) at a = 19 (sympy's
        float is one double below). The cap is checked before any operand is printed,
        so nothing under a refused node is evaluated."""
        if not operands:
            return TableRealization(values=[_constant(expr)], index=UNIT)
        index, grid = self._position(expr)
        arguments = [_on(self._table(operand), index, grid) for operand in operands]
        values = _values(expr, index.size, lambda: numpy_eval(expr, arguments))
        return TableRealization(values=values, index=index)

    def _position(self, expr: sympy.Basic) -> tuple[Index, dict[Index, np.ndarray]]:
        """expr's position over its inputs, in symbol order, and the grid of their
        coordinates. Refuses a table past the cap before any operand is printed."""
        positions = self._inputs(expr) or [UNIT]
        entries = math.prod(position.size for position in positions)
        if entries > self.cap:
            raise TableLoweringError(
                expr,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {self.cap}",
            )
        return _combined(positions), _grid(positions)

    def _inputs(self, expr: sympy.Basic) -> list[Index]:
        """The positions of expr's inputs, in the order of its symbols."""
        symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
        return list(dict.fromkeys(self._table(symbol).index for symbol in symbols))

    def _table(self, expr: sympy.Basic) -> TableRealization:
        """expr's table within the cap; a refusal of that table names expr."""
        realization = self._print(expr)
        try:
            return realization.table(self.cap)
        except TableLoweringError as exc:
            raise TableLoweringError(expr, exc.reason) from exc


def _sum(
    operands: Sequence[AbstractRealization], positions: tuple[Index, ...]
) -> SumRealization:
    """operands added, kept linear. Lists on one position, constants among them, are
    added in float: one lookup, one rounding. Switches on one factor are one switch.
    Any other symbolic operand, as sqrt(2) * (a + b) is (sympy distributes only a
    number), stays one term."""
    lists: dict[Index, TableRealization] = {}
    switches: dict[tuple[Index, tuple[float, ...]], list[SwitchRealization]] = {}
    others: list[AbstractRealization] = []
    for operand in operands:
        if isinstance(operand, SwitchRealization):
            switches.setdefault((operand.index, operand.values), []).append(operand)
        elif isinstance(operand, TableRealization):
            index = operand.index
            if index in lists:
                # inf is refused by TableRealization
                with np.errstate(over="ignore"):
                    values = lists[index].values + operand.values
                lists[index] = TableRealization(values=values, index=index)
            else:
                lists[index] = operand
        else:
            others.append(operand)
    joined = [_joined(group) for group in switches.values()]
    return SumRealization((*lists.values(), *joined, *others), positions)


def _joined(switches: list[SwitchRealization]) -> SwitchRealization:
    """Switches on one factor as one, over the sum of their rests: one w."""
    if len(switches) == 1:
        return switches[0]
    positions = tuple(dict.fromkeys(p for s in switches for p in s.positions))
    rest = _sum([switch.rest for switch in switches], positions)
    return SwitchRealization(switches[0].values, switches[0].index, rest, positions)


def _switch(
    values: tuple[float, ...],
    index: Index,
    rest: AbstractRealization,
    positions: tuple[Index, ...],
    cap: int,
) -> AbstractRealization | None:
    """values[index] * rest. A factor of one value is a constant on its inputs: it
    scales the rest. Otherwise the switch where one table over positions is past cap
    or has more entries (rule 3), and None where that table is kept."""
    if len(values) == 1:
        return rest.scaled(values[0])
    switch = SwitchRealization(values, index, rest, positions)
    entries = math.prod(position.size for position in positions)
    return switch if entries > cap or switch.entries() < entries else None


def _ranked(table: TableRealization) -> tuple[tuple[float, ...], Index]:
    """table's distinct values, in the order of their first entries, and the rank of
    each entry's value: its own index when no value repeats, else a CompactIndex."""
    values = tuple(dict.fromkeys(table.values.tolist()))
    if len(values) == len(table.values):
        return values, table.index
    rank = {value: i for i, value in enumerate(values)}
    remap = tuple(rank[value] for value in table.values.tolist())
    return values, CompactIndex(table.index, remap)


def _constant(expr: sympy.Basic) -> float | bool:
    """The value of a node with no operand: a number, or true or false as a bool."""
    if expr is sympy.true or expr is sympy.false:
        return bool(expr)
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
