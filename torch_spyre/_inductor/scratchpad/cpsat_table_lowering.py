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
literal, and it is a switch on the literal whose values are 0 and 1. A relayout
charge is gated by its literal too. Any other product that
divides is one table, divided once. Any other product is a switch on one of its
operands where that has fewer entries than its one table, so no table crosses that
factor and the rest. Any other node is a table, and a parent that reads a sum or a
switch as a list reads the node's table, which the printer makes from the node's
operands' tables (TableLowering.table). A table is over the node's inputs or
over its operands' positions, each ranked to its distinct values, whichever has
fewer entries (rule 3), the other where that one's list is not finite, and is
refused when neither fits the cap. Its list is the numpy core's function of its
sympy class (``cost_expr_numpy.py``, which the CP-SAT printer evaluates its
candidates with) applied to its operands' tables; a class the core has no function
for is refused. Nothing imports the module yet.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, TypeAlias, TypeGuard

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
        memoize_method, built once per model; a leaf's adds nothing to the model, so
        it is not."""

    @abstractmethod
    def entries(self) -> int:
        """The entries of the lookups it is read through."""

    @abstractmethod
    def inputs(self) -> tuple[InputIndex, ...]:
        """The input positions it is made from."""

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

    def entries(self) -> int:
        return 0

    def inputs(self) -> tuple[InputIndex, ...]:
        return (self,)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return self.var - self.lo if self.lo else self.var

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return grid[self]


@dataclass(frozen=True)
class UnitIndex(Index):
    """A leaf: the one position of a constant, on no input."""

    size: int = field(default=1, init=False)

    def entries(self) -> int:
        return 0

    def inputs(self) -> tuple[InputIndex, ...]:
        return ()

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

    def entries(self) -> int:
        return self.left.entries() + self.right.entries()

    def inputs(self) -> tuple[InputIndex, ...]:
        return self.left.inputs() + self.right.inputs()

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
        return len(self.remap) + self.parent.entries()

    def inputs(self) -> tuple[InputIndex, ...]:
        return self.parent.inputs()

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        rank = model.new_int_var(0, self.size - 1, "rank")
        model.add_element(self.parent.cp_expr(model), list(self.remap), rank)
        return rank

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return np.asarray(self.remap)[self.parent.on_grid(grid)]


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
    """How the printer builds a node into a model: a list on a position, a sum or a
    switch. It holds only finite values."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, in COST_SCALE units, its operands read through
        their cp_expr. Each kind's is a memoize_method: a realization that several
        readers hold is one expression per model."""

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

    def scaled(self, factor: float) -> TableRealization:
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            return TableRealization(values=factor * self.values, index=self.index)

    def bounds(self) -> tuple[int, int]:
        entries = _rounded(self.values)
        return min(entries), max(entries)

    def entries(self) -> int:
        """Its list, counted before the rounding, and its position's lookups: none
        when it is one value."""
        if np.all(self.values == self.values[0]):
            return 0
        return len(self.values) + self.index.entries()

    def sum_key(self) -> Index:
        """Its position: lists on one position are added in float, one lookup and one
        rounding."""
        return self.index

    def joined(self, others: Sequence[Self]) -> TableRealization:
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            values = sum((other.values for other in others), self.values)
        return TableRealization(values=values, index=self.index)


@dataclass(frozen=True, eq=False)
class SumRealization(AbstractRealization):
    """sum(terms). A constant is a term on a UnitIndex."""

    terms: tuple[AbstractRealization, ...]

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

    def scaled(self, factor: float) -> SumRealization:
        terms = tuple(term.scaled(factor) for term in self.terms)
        return SumRealization(terms)

    def bounds(self) -> tuple[int, int]:
        ends = [term.bounds() for term in self.terms]
        return sum(end[0] for end in ends), sum(end[1] for end in ends)

    def entries(self) -> int:
        return sum(term.entries() for term in self.terms)


@dataclass(frozen=True, eq=False)
class SwitchRealization(AbstractRealization):
    """values[index] * rest: a product on one of its factors (rule 4), or a gate on
    its 0/1 literal, values 0 and 1 (rule 5). The rest is collapsed once into w, and
    one true/false per value, on where index takes it, sets g to the value times w. No
    table crosses index and rest."""

    values: tuple[float, ...]
    index: Index
    rest: AbstractRealization

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """A whole value is g == v * w; a fraction, p / q with q = denominator(), is
        the floor: q * g <= p * w <= q * g + q - 1. A rest of several terms that a
        value other than 0 or 1 multiplies is one variable w, so p multiplies one
        coefficient, not each term's: CP-SAT 9.15's presolve calls a model with p
        times a list's step past 2^31 infeasible."""
        w = self.rest.cp_expr(model)
        scales = any(value not in (0.0, 1.0) for value in self.values)
        if scales and not isinstance(w, (int, cp_model.IntVar)):
            rest = model.new_int_var(*self.rest.bounds(), "rest")
            model.add(rest == w)
            w = rest
        q = self.denominator()
        g = model.new_int_var(*self.bounds(), "switch")
        for value, case in zip(self.values, self._cases(model)):
            if value.is_integer():
                model.add(g == int(value) * w).only_enforce_if(case)
            else:
                p = round(value * q)
                model.add(q * g <= p * w).only_enforce_if(case)
                model.add(p * w <= q * g + q - 1).only_enforce_if(case)
        return g

    def _cases(self, model: cp_model.CpModel) -> list[cp_model.IntVar]:
        """One true/false per value, true where index takes it: a 0/1 literal's own
        variable and its negation, or new ones mapped onto index."""
        if _is_literal(self.index):
            return [self.index.var.negated(), self.index.var]
        cases = [model.new_bool_var("case") for _ in self.values]
        model.add_map_domain(self.index.cp_expr(model), cases)
        return cases

    def scaled(self, factor: float) -> SwitchRealization:
        rest = self.rest.scaled(factor)
        return SwitchRealization(self.values, self.index, rest)

    def bounds(self) -> tuple[int, int]:
        q = self.denominator()
        ends = [
            int(value) * end if value.is_integer() else round(value * q) * end // q
            for value in self.values
            for end in self.rest.bounds()
        ]
        return min(ends), max(ends)

    def entries(self) -> int:
        """Its rest's, its index's lookups, and one per case a literal does not
        give."""
        cases = 0 if _is_literal(self.index) else len(self.values)
        return self.rest.entries() + cases + self.index.entries()

    def sum_key(self) -> tuple[Index, tuple[float, ...]]:
        """Its factor: switches on one factor are one switch over their rests' sum."""
        return self.index, self.values

    def joined(self, others: Sequence[Self]) -> SwitchRealization:
        rest = _sum([switch.rest for switch in (self, *others)])
        return SwitchRealization(self.values, self.index, rest)

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


def _values(expr: sympy.Basic, count: int, lists: list[np.ndarray]) -> np.ndarray:
    """expr's list of count values, the numpy core's function of it on its operands'
    lists: bool for a condition, float otherwise. A RecursionError is left to
    TableLowering.print."""
    dtype = bool if _is_condition(expr) else float
    with np.errstate(all="ignore"):
        try:
            result = np.asarray(numpy_eval(expr, lists), dtype=dtype)
        except RecursionError:
            raise
        except Exception as exc:
            raise TableLoweringError(
                expr, f"no value ({type(exc).__name__}: {exc})"
            ) from exc
    return np.broadcast_to(result, (count,)).astype(dtype)


def _combined(positions: Sequence[Index]) -> Index:
    """positions as one position, the last fastest; a UnitIndex for none."""
    index = positions[0] if positions else UnitIndex()
    for position in positions[1:]:
        index = CombineIndex(index, position)
    return index


def _grid(positions: Sequence[Index]) -> dict[Index, np.ndarray]:
    """Each position's coordinate at every joint assignment of positions, row-major,
    the last fastest."""
    shape = tuple(p.size for p in positions)
    return dict(zip(positions, np.indices(shape).reshape(len(shape), -1)))


def _values_at(
    table: TableRealization, index: Index, grid: Mapping[Index, np.ndarray]
) -> np.ndarray:
    """table's list at each entry of index; grid holds the coordinates of the
    positions index combines."""
    # On a UnitIndex too a constant is broadcast: numpy computes x**0.5 as sqrt(x)
    # only for a broadcast exponent, and both sides of a tie must take one path.
    if table.index == index and index != UnitIndex():
        return table.values
    at = grid[table.index] if table.index in grid else table.index.on_grid(grid)
    return np.broadcast_to(table.values[at], (index.size,))


def _compacted(parts: Sequence[TableRealization]) -> list[TableRealization]:
    """parts, where they are on two or more positions, each rewritten on its
    position's rank: one entry per unique row of the parts on that position
    (_compact_to_unique_rows). A single position is kept as it is, since its rank's
    lookup would be as long as the list it saves, and a constant stays a constant."""
    on: dict[Index, list[int]] = {}
    for i, part in enumerate(parts):
        if part.index != UnitIndex():
            on.setdefault(part.index, []).append(i)
    compacted = list(parts)
    if len(on) > 1:
        for index, members in on.items():
            rank, lists = _compact_to_unique_rows(
                index, [parts[i].values for i in members]
            )
            for i, values in zip(members, lists):
                compacted[i] = TableRealization(values=values, index=rank)
    return compacted


def _input_positions(parts: Sequence[TableRealization]) -> list[Index]:
    """The input positions parts are made from, ordered by variable; a UnitIndex for
    none."""
    inputs = {i for part in parts for i in part.index.inputs()}
    return sorted(inputs, key=lambda i: i.var.index) or [UnitIndex()]


def _own_positions(parts: Sequence[TableRealization]) -> list[Index]:
    """The positions parts are on, each once in the order met, constants skipped; a
    UnitIndex for none."""
    positions = (part.index for part in parts if part.index != UnitIndex())
    return list(dict.fromkeys(positions)) or [UnitIndex()]


def _size(positions: Sequence[Index]) -> int:
    return math.prod(position.size for position in positions)


def _tabulated(
    expr: sympy.Basic, parts: Sequence[TableRealization], cap: int
) -> TableRealization:
    """expr's table within cap from parts, its operands' tables, laid out over the
    parts' inputs or over the positions of the parts compacted (_compacted), where
    those are not the inputs: the layout with the fewer entries, its list's and its
    lookups' (rule 3), the inputs on a tie. Its list is expr's numpy core function of
    the parts' lists on it (_values), and only that table is computed, but where its
    list is not finite the other is: a compacted layout whose positions share an
    input also holds rows no assignment reaches."""
    inputs = _input_positions(parts)
    layouts = [(inputs, list(parts))]
    compacted = _compacted(parts)
    if set(_own_positions(compacted)) != set(inputs):
        layouts.append((_own_positions(compacted), compacted))
    sizes = [_size(positions) for positions, _ in layouts]
    if min(sizes) > cap:
        over_inputs, *own = sizes
        raise TableLoweringError(
            None,
            f"one table over its inputs would hold {over_inputs} entries, past the cap "
            f"of {cap}" + "".join(f", and one from its parts {size}" for size in own),
        )
    fitting = sorted(
        (layout for layout, size in zip(layouts, sizes) if size <= cap),
        key=lambda layout: _size(layout[0]) + sum(p.entries() for p in layout[0]),
    )
    for positions, read in fitting:
        index, grid = _combined(positions), _grid(positions)
        lists = [_values_at(part, index, grid) for part in read]
        values = _values(expr, index.size, lists)
        if np.all(np.isfinite(values)):
            break
    return TableRealization(values=values, index=index)


# ---------------------------------------------------------------------- the printer


def _is_condition(expr: sympy.Basic) -> bool:
    """Whether expr is a condition: a relational, a logical operation, true or false,
    and not a symbol."""
    return isinstance(expr, Boolean) and not isinstance(expr, sympy.Symbol)


class TableLowering(Printer):
    """A sympy Printer whose print of a node is its realization. A print method
    returns None to decline a node, which is then one table of its operands
    (_operation).

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
        self._tables: dict[sympy.Basic, TableRealization] = {}
        # a gated product's table is its gate's: a Dummy whose table is 1 where the
        # literal is kept and 0 where it is not, times the product at the kept value
        self._gates: dict[sympy.Basic, sympy.Mul] = {}
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
                    realization = self._operation(expr)
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
        return _sum(operands)

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
        coefficient = math.prod(float(self.table(c).values[0]) for c in constants)
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
                on = sympy.Dummy("kept")
                kept_where = np.where(np.arange(2) == kept, 1.0, 0.0)
                self._tables[on] = TableRealization(values=kept_where, index=literal)
                self._gates[expr] = sympy.Mul(on, kept_product, evaluate=False)
                return _gate(literal, kept, rest)
        return None

    def _literal(self, operand: sympy.Basic) -> tuple[InputIndex, int] | None:
        """The boolean literal that operand gates by, and its kept value: where operand
        reads that literal only and is 0 at exactly one of its values, as x, 1 - x and
        2 - 2*x are and 1 + x is not. None for any other operand."""
        positions = self._inputs(operand)
        one = len(positions) == 1 and not divides(operand)  # 1/a is never 0
        literal = positions[0] if one else None
        if literal is not None and _is_literal(literal):
            zero = self.table(operand).values == 0
            if np.count_nonzero(zero) == 1:
                return literal, int(zero[0])
        return None

    def _at(
        self, expr: sympy.Basic, literal: InputIndex, value: int
    ) -> dict[sympy.Basic, sympy.Number]:
        """Each of expr's symbols on literal, at its value where literal is value."""
        return {
            symbol: _number(self.table(symbol).values[value])
            for symbol in expr.free_symbols
            if self.table(symbol).index == literal
        }

    def _switched(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> AbstractRealization | None:
        """expr, the product of operands, as a switch (_switch) on the operand with
        the fewest distinct values among those whose table fits the cap, the first in
        expr's order on a tie, over the others' product. None where no operand's table
        fits or the product keeps its table."""
        tables = {
            operand: self.table(operand) for operand in operands if self._fits(operand)
        }
        if not tables:
            return None
        factor = min(tables, key=lambda o: len(np.unique(tables[o].values)))
        index, (values,) = _compact_to_unique_rows(
            tables[factor].index, [tables[factor].values]
        )
        index = self._ranks.setdefault(index, index)
        rest = self._print(sympy.Mul(*(o for o in operands if o != factor)))
        positions = tuple(self._inputs(expr))
        return _switch(tuple(values.tolist()), index, rest, positions, self.cap)

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

    def _operation(self, expr: sympy.Basic) -> TableRealization:
        """expr as its table (_tabulated) of its operands' tables. An operation on
        numbers is computed the same way, on a UnitIndex, so sqrt(19) equals sqrt(a)
        at a = 19 (sympy's float is one double below)."""
        if not operands(expr):
            return TableRealization(values=[_constant(expr)], index=UnitIndex())
        parts = [self.table(operand) for operand in operands(expr)]
        return _tabulated(expr, parts, self.cap)

    def _fits(self, expr: sympy.Basic) -> bool:
        """Whether one table over expr's inputs is within the cap."""
        return math.prod(position.size for position in self._inputs(expr)) <= self.cap

    def _inputs(self, expr: sympy.Basic) -> list[Index]:
        """The positions of expr's inputs, in the order of its symbols."""
        symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
        return list(dict.fromkeys(self.table(symbol).index for symbol in symbols))

    def table(self, expr: sympy.Basic) -> TableRealization:
        """expr's table within the cap, for a parent that computes its own list: its
        realization where that is a table, and otherwise its operation's (_operation),
        its gate's for a gated product (_gates). A refusal of that table names expr."""
        if expr not in self._tables:
            realization = self._print(expr)
            try:
                if isinstance(realization, TableRealization):
                    table = realization
                else:
                    table = self._operation(self._gates.get(expr, expr))
            except TableLoweringError as exc:
                raise TableLoweringError(expr, exc.reason) from exc
            self._tables[expr] = table
        return self._tables[expr]


def _sum(operands: Sequence[AbstractRealization]) -> SumRealization:
    """operands added, kept linear. Terms with one sum_key are joined into one: lists
    on one position, constants among them, added in float, one lookup and one
    rounding; switches on one factor, gates among them, one over their rests' sum. Any other operand, as sqrt(2) * (a + b) is (sympy
    distributes only a number), stays one term."""
    groups: dict[Hashable, list[AbstractRealization]] = {}
    for operand in operands:
        groups.setdefault(operand.sum_key(), []).append(operand)
    terms = (g[0] if len(g) == 1 else g[0].joined(g[1:]) for g in groups.values())
    return SumRealization(tuple(terms))


def _switch(
    values: tuple[float, ...],
    index: Index,
    rest: AbstractRealization,
    positions: tuple[Index, ...],
    cap: int,
) -> AbstractRealization | None:
    """values[index] * rest. A factor of one value is a constant on its inputs: it
    scales the rest. Otherwise the switch where one table over positions is past cap
    or has more entries (rule 3), and None where that table is kept. The table from
    the product's parts is not weighed: it pairs the factor's ranks with the rest's,
    which never has fewer entries than the switch."""
    if len(values) == 1:
        return rest.scaled(values[0])
    switch = SwitchRealization(values, index, rest)
    entries = math.prod(position.size for position in positions)
    return switch if entries > cap or switch.entries() < entries else None


def _gate(
    literal: InputIndex,
    kept: int,
    rest: AbstractRealization,
) -> AbstractRealization:
    """rest where literal is at kept and 0 at its other value (rule 5): a switch on
    literal whose values are 0 and 1. A rest of one value makes it one list on the
    literal, affine: no switch and no table."""
    if isinstance(rest, TableRealization) and len(np.unique(rest.values)) == 1:
        values = np.where(np.arange(2) == kept, float(rest.values[0]), 0.0)
        return TableRealization(values=values, index=literal)
    return SwitchRealization((0.0, 1.0) if kept else (1.0, 0.0), literal, rest)


def _is_literal(index: Index) -> TypeGuard[InputIndex]:
    """Whether index is a 0/1 variable's own position, the true/false of two cases."""
    return isinstance(index, InputIndex) and (index.lo, index.size) == (0, 2)


def _number(value: float) -> sympy.Number:
    """value as a sympy number: an Integer when it is whole."""
    return (
        sympy.Integer(int(value)) if float(value).is_integer() else sympy.Float(value)
    )


def _compact_to_unique_rows(
    index: Index, lists: Sequence[np.ndarray]
) -> tuple[Index, list[np.ndarray]]:
    """index compacted to the unique rows of lists, each a list on index, in the order
    of their first entries: the rank of each entry's row, a CompactIndex, or index
    itself when no row repeats; and each list's value at each rank."""
    rows = list(zip(*(values.tolist() for values in lists)))
    distinct = list(dict.fromkeys(rows))
    if len(distinct) == len(rows):
        return index, list(lists)
    rank = {row: i for i, row in enumerate(distinct)}
    remap = tuple(rank[row] for row in rows)
    return CompactIndex(index, remap), [np.asarray(column) for column in zip(*distinct)]


def _constant(expr: sympy.Basic) -> float | bool:
    """The value of a node with no operand: a number, or true or false as a bool."""
    if expr is sympy.true or expr is sympy.false:
        return bool(expr)
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
