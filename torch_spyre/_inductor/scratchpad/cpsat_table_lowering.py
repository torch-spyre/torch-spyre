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
  the model. Each ``cp_expr`` but a leaf position's is a ``memoize_method``, so a
  position or realization held by several readers is built once per model. Only
  ``cp_expr`` methods touch the model.

A sum stays linear, so no table crosses two of its terms. A product of several
operands is gated by one that reads a boolean literal only and is 0 at one of its
values: its rest is the product with the literal at the other, so no table reads the
literal. A relayout charge is gated by its literal too. Any other product that
divides is one table, divided once. Any other product is a switch on one of its
operands where that has fewer entries than its one table, so no table crosses that
factor and the rest. Any other node is one table over its inputs, refused when they
do not fit the cap together. Its list is the numpy core's function of its sympy class
(``cost_expr_numpy.py``, which the CP-SAT printer evaluates its candidates with)
applied to its operands' tables; a class the core has no function for is refused.
Nothing imports the module yet.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Self, TypeAlias

import numpy as np
import sympy
from ortools.sat.python import cp_model
from sympy.logic.boolalg import Boolean
from sympy.printing.printer import Printer

from torch_spyre._inductor.scratchpad.cost_expr_numpy import (
    divides,
    factors,
    numpy_eval,
    operands,
)
from torch_spyre._inductor.scratchpad.cpsat_expr_printer import memoize_method

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

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The position in model, its parts read through their cp_expr. A step's is a
        memoize_method, built once per model; a leaf's is not, as its memo would keep
        every model UNIT is built in."""

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

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
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

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
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

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, in COST_SCALE units, its operands read through
        their cp_expr. Each kind's is a memoize_method: a realization that several
        readers hold is one expression per model."""

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

    def sum_key(self) -> Hashable:
        """In a sum, terms with one key are one term (joined). By default the
        realization itself: a term of its own."""
        return self

    def joined(self, others: Sequence[Self]) -> AbstractRealization:
        """It and others, which have its sum_key, as one term. By default others are
        itself again, so the term is it times their count."""
        return self.scaled(1 + len(others))


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

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
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

    def sum_key(self) -> Index:
        """Its position: lists on one position are added in float, one lookup and one
        rounding."""
        return self.index

    def joined(self, others: Sequence[Self]) -> TableRealization:
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            values = sum((other.values for other in others), self.values)
        return TableRealization(values=values, index=self.index)


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

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
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

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """A whole value is g == v * w; a fraction, p / q with q = denominator(), is
        the floor: q * g <= p * w <= q * g + q - 1. A rest of several terms is one
        variable w, so p multiplies one coefficient, not each term's: CP-SAT 9.15's
        presolve calls a model with p times a list's step past 2^31 infeasible."""
        w = self.rest.cp_expr(model)
        if not isinstance(w, (int, cp_model.IntVar)):
            rest = model.new_int_var(*self.rest.bounds(), "rest")
            model.add(rest == w)
            w = rest
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

    def sum_key(self) -> tuple[Index, tuple[float, ...]]:
        """Its factor: switches on one factor are one switch over their rests' sum."""
        return self.index, self.values

    def joined(self, others: Sequence[Self]) -> SwitchRealization:
        rest, positions = _joined((self, *others))
        return SwitchRealization(self.values, self.index, rest, positions)

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


@dataclass(frozen=True, eq=False)
class GateRealization(SymbolicRealization):
    """rest where a 0/1 literal is at kept, 0 or 1, and 0 where it is at its other
    value (rule 5). The literal is the true/false of the two cases, and rest does not
    read it."""

    literal: InputIndex
    kept: int
    rest: AbstractRealization
    positions: tuple[Index, ...]

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """g == w where the term is kept, g == 0 where it is not."""
        w = self.rest.cp_expr(model)
        g = model.new_int_var(*self.bounds(), "gate")
        var = self.literal.var
        on, off = (var, var.negated()) if self.kept else (var.negated(), var)
        model.add(g == w).only_enforce_if(on)
        model.add(g == 0).only_enforce_if(off)
        return g

    def _list(
        self, index: Index, grid: Mapping[Index, np.ndarray], cap: int
    ) -> np.ndarray:
        kept = self.literal.on_grid(grid) == self.kept
        return np.where(kept, _on(self.rest.table(cap), index, grid), 0.0)

    def scaled(self, factor: float) -> GateRealization:
        return replace(self, rest=self.rest.scaled(factor))

    def bounds(self) -> tuple[int, int]:
        lo, hi = self.rest.bounds()
        return min(lo, 0), max(hi, 0)

    def entries(self) -> int:
        return self.rest.entries()

    def sum_key(self) -> tuple[InputIndex, int]:
        """Its literal and kept value: gates on them are one gate over their rests'
        sum."""
        return self.literal, self.kept

    def joined(self, others: Sequence[Self]) -> GateRealization:
        rest, positions = _joined((self, *others))
        return replace(self, rest=rest, positions=positions)


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
                    realization = self._operation(expr, operands(expr))
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
        """A product of several operands is gated (_gated) where one of them is a
        gate, its constants in the gate's rest. Otherwise a product that divides
        (divides), 98*k/a, is declined: one table, divided once in the numpy core as
        lambdify prints it, and no table of 1/a, so 98*k/a is 2k at a = 49, where k
        times 98 * (1/49) is under 2k. Any other product of several operands is a
        switch (_switched), its constants applied to the rest. A constant times one
        operand is the operand scaled in float, as lambdify prints a/49: (1/49)*a, and
        any other product is declined."""
        constants = [factor for factor in factors(expr) if not factor.free_symbols]
        symbolic = [factor for factor in expr.args if factor.free_symbols]
        if len(symbolic) > 1:
            gated = self._gated(expr, expr.args)
            if gated is not None:
                return gated
        if any(divides(factor) for factor in expr.args):
            return None
        if len(symbolic) == 1:
            realization: AbstractRealization | None = self._print(symbolic[0])
        else:
            realization = self._switched(expr, symbolic)
        if realization is None or not constants:
            return realization
        coefficient = math.prod(float(self._table(c).values[0]) for c in constants)
        return realization.scaled(coefficient)

    def _gated(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> AbstractRealization | None:
        """expr, the product of operands, gated (_gate) by the first operand that is a
        gate. Its rest is the product with the gate's literal at its kept value, so
        nothing under the gate reads the literal. None when no operand is a gate."""
        for operand in operands:
            gate = self._literal(operand)
            if gate is not None:
                literal, kept = gate
                product = sympy.Mul(*operands)
                try:  # sympy evaluates the product at the kept value
                    kept_product = product.xreplace(self._at(product, literal, kept))
                except (TypeError, ValueError) as exc:
                    raise TableLoweringError(
                        expr, f"no value ({type(exc).__name__}: {exc})"
                    ) from exc
                rest = self._print(kept_product)
                return _gate(literal, kept, rest, tuple(self._inputs(expr)))
        return None

    def _literal(self, operand: sympy.Basic) -> tuple[InputIndex, int] | None:
        """The boolean literal that operand gates by, and its kept value: where operand
        reads that literal only and is 0 at exactly one of its values, as x, 1 - x and
        2 - 2*x are and 1 + x is not. None for any other operand."""
        positions = self._inputs(operand)
        one = len(positions) == 1 and not divides(operand)  # 1/a is never 0
        literal = positions[0] if one else None
        if isinstance(literal, InputIndex) and (literal.lo, literal.size) == (0, 2):
            zero = self._table(operand).values == 0
            if np.count_nonzero(zero) == 1:
                return literal, int(zero[0])
        return None

    def _at(
        self, expr: sympy.Basic, literal: InputIndex, value: int
    ) -> dict[sympy.Basic, sympy.Number]:
        """Each of expr's symbols on literal, at its value where literal is value."""
        return {
            symbol: _number(self._table(symbol).values[value])
            for symbol in expr.free_symbols
            if self._table(symbol).index == literal
        }

    def _switched(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> AbstractRealization | None:
        """expr, the product of operands, as a switch (_switch) on the operand with
        the fewest distinct values among those whose table fits the cap, the first in
        expr's order on a tie, over the others' product. None where no operand's table
        fits or the product keeps its table."""
        tables = {
            operand: self._table(operand) for operand in operands if self._fits(operand)
        }
        if not tables:
            return None
        factor = min(tables, key=lambda o: len(np.unique(tables[o].values)))
        values, index = _ranked(tables[factor])
        index = self._ranks.setdefault(index, index)
        rest = self._print(sympy.Mul(*(o for o in operands if o != factor)))
        return _switch(values, index, rest, tuple(self._inputs(expr)), self.cap)

    def _print_RelayoutCharge(self, expr: sympy.Function) -> AbstractRealization | None:
        """RelayoutCharge(is_lx, division, *prices), is_lx times the price at the
        division, gated (_gated) as that product over its prices, one list on the
        division. Declined where is_lx reads no variable or is no gate."""
        is_lx, *others = expr.args
        if not is_lx.free_symbols:
            return None
        return self._gated(expr, [is_lx, expr.func(sympy.S.One, *others)])

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

    def _fits(self, expr: sympy.Basic) -> bool:
        """Whether one table over expr's inputs is within the cap."""
        return math.prod(position.size for position in self._inputs(expr)) <= self.cap

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
    """operands added, kept linear. Terms with one sum_key are joined into one: lists
    on one position, constants among them, added in float, one lookup and one
    rounding; switches on one factor, and gates on one literal and kept value, one
    over their rests' sum. Any other operand, as sqrt(2) * (a + b) is (sympy
    distributes only a number), stays one term."""
    groups: dict[Hashable, list[AbstractRealization]] = {}
    for operand in operands:
        groups.setdefault(operand.sum_key(), []).append(operand)
    terms = (g[0] if len(g) == 1 else g[0].joined(g[1:]) for g in groups.values())
    return SumRealization(tuple(terms), positions)


def _joined(
    group: Sequence[SwitchRealization] | Sequence[GateRealization],
) -> tuple[SumRealization, tuple[Index, ...]]:
    """The rest and positions of group, switches on one factor or gates on one
    literal and kept value, joined into one: the sum of their rests, one w."""
    positions = tuple(dict.fromkeys(p for r in group for p in r.positions))
    return _sum([realization.rest for realization in group], positions), positions


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


def _gate(
    literal: InputIndex,
    kept: int,
    rest: AbstractRealization,
    positions: tuple[Index, ...],
) -> AbstractRealization:
    """rest where literal is at kept and 0 at its other value (rule 5). A rest of one
    value makes it one list on the literal, affine: no gate and no table."""
    if isinstance(rest, TableRealization) and len(np.unique(rest.values)) == 1:
        values = np.where(np.arange(2) == kept, float(rest.values[0]), 0.0)
        return TableRealization(values=values, index=literal)
    return GateRealization(literal, kept, rest, positions)


def _number(value: float) -> sympy.Number:
    """value as a sympy number: an Integer when it is whole."""
    return (
        sympy.Integer(int(value)) if float(value).is_integer() else sympy.Float(value)
    )


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
