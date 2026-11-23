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
works out which entry of a table applies. The objective uses three CP-SAT forms:
``element`` lookups whose entries are the expression's values rounded once onto
``COST_SCALE``, linear constraints (some enforced by a true/false), and
``lin_min``/``lin_max``. No two variables are multiplied. Three layers:

- values: ``Index``, a position, and ``AbstractRealization``, what a node is realized
  as: a table (a float list on a position), a sum, a switch or a gate (a product, rules
  4 and 5) or a Min/Max kept linear. Values hold no model;
- the nodes: ``TableLowering`` is a factory that makes one ``Node`` per node of the
  expression, of a class chosen by the node's sympy class. A node has two reads, each
  made the first time it is read and kept: its ``table``, the cheapest table of it in
  entries, for a parent that computes its own list from it, and its ``realization``,
  for a linear reader. A node's list is its function, its sympy class lambdified once
  per class and form of operands, applied to its operands' lists;
- the build: the root's ``cp_expr`` reads its realization, which builds itself into
  the model, each position and realization once per build.

``lower_cost_with_tables`` is the entry point from the printer's maps.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import Boolean

from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

if TYPE_CHECKING:
    from ortools.sat.python import cp_model
else:
    try:
        from ortools.sat.python import cp_model
    except ImportError:  # pragma: no cover - only when ortools is absent
        cp_model = None

# One cost unit is COST_SCALE objective units: a table entry is
# rint(COST_SCALE * value).
COST_SCALE = 1024
# The largest table entry: a sum of thousands of entries stays inside int64.
MAX_ENTRY = 2**50
# A switch holds its factor's values to 1/q: q is the largest power of two up to
# MAX_SWITCH_SCALE that keeps q * w under MAX_SWITCH_PRODUCT. Under MIN_SWITCH_SCALE
# it is coarse: a product with a table keeps the table instead.
MAX_SWITCH_SCALE = 2**30
MIN_SWITCH_SCALE = 2**22
MAX_SWITCH_PRODUCT = 2**60
# Values equal to this many significant digits share a rank or a rung.
SIGNIFICANT_DIGITS = 12
# A Min/Max with a table stays linear only when that takes under 1/LINEAR_MARGIN of
# the table's entries: a rank's small domain searches faster than a lin_max over wide
# values.
LINEAR_MARGIN = 2

# What CP-SAT takes as a linear expression: a LinearExpr, an IntVar or an int.
CpExpr: TypeAlias = "cp_model.LinearExprT"
# What one build has made, by what made it.
Built: TypeAlias = dict[Any, Any]


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


class _BuiltOnce(ABC):
    """A value, with no model in it, that builds its own CP-SAT side once per build:
    a position by its value, a realization by identity."""

    def cp_expr(self, model: cp_model.CpModel, built: Built | None = None) -> CpExpr:
        built = {} if built is None else built
        if self not in built:
            built[self] = self.build(model, built)
        return built[self]

    @abstractmethod
    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        """Its CP-SAT expression in model, its parts read through cp_expr."""


# ------------------------------------------------------------------------ positions


class Index(_BuiltOnce):
    """A position: a map from the inputs' values to 0 .. size - 1, as a CP-SAT
    expression. Each kind of step is a subclass that holds its parts, the positions it
    is made from, as its own fields. Steps are frozen dataclasses: two that make the
    same position from equal parts are equal and hash alike."""

    size: int

    @property
    def lookups(self) -> int:
        """The entries of its own lookup: 0 for a step with none."""
        return 0

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        """The position at every point of a grid: its own coordinate when it is one of
        the grid's positions, otherwise computed from its parts as build makes it."""
        if self in grid:
            return grid[self]
        return self.from_parts(grid)

    @abstractmethod
    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
        """The position at every point of grid, from its parts' on_grid."""

    def parts(self) -> tuple[Index, ...]:
        return ()

    def steps(self) -> dict[Index, None]:
        """Each distinct step under it, itself included."""
        steps: dict[Index, None] = {}
        pending: list[Index] = [self]
        while pending:
            step = pending.pop()
            if step not in steps:
                steps[step] = None
                pending.extend(step.parts())
        return steps


@dataclass(frozen=True)
class InputIndex(Index):
    """A leaf: a variable's own position, v -> v - lo, for a variable whose values run
    from lo to lo + size - 1."""

    var: cp_model.IntVar
    size: int
    lo: int = 0

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        return self.var - self.lo if self.lo else self.var

    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return grid[self]


@dataclass(frozen=True)
class LocateIndex(Index):
    """A leaf: the position of a variable whose domain has gaps, values[position] ==
    var, one lookup."""

    var: cp_model.IntVar
    values: tuple[int, ...]
    size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", len(self.values))

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        position = model.new_int_var(0, self.size - 1, "locate")
        model.add_element(position, list(self.values), self.var)
        return position

    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return grid[self]

    @property
    def lookups(self) -> int:
        return self.size


@dataclass(frozen=True)
class UnitIndex(Index):
    """A leaf: the one position of a constant, on no input."""

    size: int = field(default=1, init=False)

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        return 0

    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
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

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        left = self.left.cp_expr(model, built)
        return left * self.right.size + self.right.cp_expr(model, built)

    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return self.left.on_grid(grid) * self.right.size + self.right.on_grid(grid)

    def parts(self) -> tuple[Index, ...]:
        return (self.left, self.right)


@dataclass(frozen=True)
class CompactIndex(Index):
    """parent -> remap[parent]: one lookup that renumbers a position onto size
    positions, the distinct values (a rank), the rungs of a ladder or the states of a
    fold. A remap of one value is a constant: no lookup, its parent never built."""

    parent: Index
    remap: tuple[int, ...]
    size: int

    @property
    def constant(self) -> bool:
        return len(set(self.remap)) == 1

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        if self.constant:
            return self.remap[0]
        target = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(self.remap))), "compact"
        )
        model.add_element(self.parent.cp_expr(model, built), list(self.remap), target)
        return target

    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
        if self.constant:
            return self.remap[0]
        return np.asarray(self.remap)[self.parent.on_grid(grid)]

    def parts(self) -> tuple[Index, ...]:
        return () if self.constant else (self.parent,)

    @property
    def lookups(self) -> int:
        return 0 if self.constant else len(self.remap)


@dataclass(frozen=True)
class ExtremeIndex(Index):
    """The min (is_min) or max of rungs, positions on one ascending ladder of size
    positions: a ladder keeps order, so the Min of the rungs is the rung of the Min."""

    rungs: tuple[Index, ...]
    is_min: bool
    size: int

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        target = model.new_int_var(0, self.size - 1, "rung")
        exprs = [rung.cp_expr(model, built) for rung in self.rungs]
        if self.is_min:
            model.add_min_equality(target, exprs)
        else:
            model.add_max_equality(target, exprs)
        return target

    def from_parts(self, grid: Mapping[Index, np.ndarray]) -> Any:
        pick = np.minimum if self.is_min else np.maximum
        rungs = np.broadcast_arrays(*(np.asarray(r.on_grid(grid)) for r in self.rungs))
        return pick.reduce(rungs)

    def parts(self) -> tuple[Index, ...]:
        return self.rungs


UNIT = UnitIndex()


def _combined(positions: Sequence[Index]) -> Index:
    """positions as one, row-major, the last fastest; UNIT for none."""
    index = positions[0] if positions else UNIT
    for position in positions[1:]:
        index = CombineIndex(index, position)
    return index


def _entries_of(positions: Iterable[Index]) -> int:
    """The entries of the lookups under positions, each step once."""
    steps: dict[Index, None] = {}
    for position in positions:
        steps.update(position.steps())
    return sum(step.lookups for step in steps)


# --------------------------------------------------------------------- realizations


def _rounded(values: Any) -> list[int]:
    """values rounded once onto COST_SCALE, each within half a unit of its value; an
    entry past MAX_ENTRY raises TableLoweringError."""
    scaled = np.rint(COST_SCALE * np.atleast_1d(np.asarray(values, dtype=float)))
    largest = float(np.max(np.abs(scaled)))
    if largest > MAX_ENTRY:
        raise TableLoweringError(
            None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
        )
    return [int(v) for v in scaled]


class AbstractRealization(_BuiltOnce):
    """What a node is realized as: the one thing a reader and the objective take of
    it, in COST_SCALE units once built."""

    @abstractmethod
    def scaled(self, factor: float) -> AbstractRealization:
        """factor times it, applied in float before any rounding."""

    @abstractmethod
    def bounds(self) -> tuple[int, int]:
        """Its lowest and highest value in model."""

    @property
    @abstractmethod
    def entries(self) -> int:
        """The table entries it puts in the model, which rule 3 weighs."""

    def key(self) -> Hashable | None:
        """Terms of one sum with one key are one term (join); None for a term kept
        apart."""
        return None

    def join(self, other: Any) -> AbstractRealization | None:
        """This term and another of its key as one, or None to keep them apart."""
        raise TypeError(f"a {type(self).__name__} is kept apart")


@dataclass(frozen=True, eq=False, kw_only=True)
class TableRealization(AbstractRealization):
    """A node's table: values[index], made together and never apart. values is the
    value at each position, float until it is built, stored as a read-only copy; a
    list whose length is not the index's size is refused."""

    values: np.ndarray
    index: Index

    def __post_init__(self) -> None:
        array = np.array(self.values, dtype=float)
        if array.shape != (self.index.size,):
            raise ValueError(
                f"a list of shape {array.shape} on an index of size {self.index.size}"
            )
        array.setflags(write=False)
        object.__setattr__(self, "values", array)

    @property
    def finite(self) -> bool:
        """Whether every value in its list is a finite number."""
        return bool(np.all(np.isfinite(self.values)))

    @cached_property
    def rounded(self) -> list[int]:
        return _rounded(self.values)

    @cached_property
    def step(self) -> int | None:
        """The step of the rounded list when it is affine in its position, so it needs
        no table (0 when constant); None otherwise."""
        rounded = self.rounded
        step = rounded[1] - rounded[0] if len(rounded) > 1 else 0
        if all(v == rounded[0] + i * step for i, v in enumerate(rounded)):
            return step
        return None

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        """A number when constant, its position never built; table[0] + step *
        position when affine; otherwise one add_element on its position into a new
        variable whose domain is the rounded values."""
        entries = self.rounded
        if self.step == 0:
            return entries[0]
        position = self.index.cp_expr(model, built)
        if self.step is not None:
            return entries[0] + self.step * position
        target = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(entries))), "table"
        )
        model.add_element(position, entries, target)
        return target

    def scaled(self, factor: float) -> TableRealization:
        if factor == 1:
            return self
        return TableRealization(values=factor * self.values, index=self.index)

    def bounds(self) -> tuple[int, int]:
        return min(self.rounded), max(self.rounded)

    @property
    def entries(self) -> int:
        if self.step == 0:
            return 0
        own = len(self.values) if self.step is None else 0
        return own + _entries_of([self.index])

    def key(self) -> Hashable:
        return ("table", self.index)

    def join(self, other: Any) -> TableRealization:
        """Lists on one position, added before the rounding: one lookup."""
        return TableRealization(values=self.values + other.values, index=self.index)


@dataclass(frozen=True, eq=False)
class LinearRealization(AbstractRealization):
    """A sum read linearly: its constant plus its terms, each a realization."""

    constant: float
    terms: tuple[AbstractRealization, ...]

    @classmethod
    def of(
        cls, constant: float, terms: Iterable[AbstractRealization]
    ) -> LinearRealization:
        """constant plus terms: a sum among them flattened, a table on one position
        added to the constant, and terms of one key joined."""
        keyed: dict[Hashable, AbstractRealization] = {}
        apart: list[AbstractRealization] = []
        pending = list(terms)
        for term in pending:  # a sum's terms are appended as they are met
            if isinstance(term, LinearRealization):
                constant += term.constant
                pending.extend(term.terms)
            elif isinstance(term, TableRealization) and term.index.size == 1:
                constant += float(term.values[0])
            elif (key := term.key()) is None:
                apart.append(term)
            elif key not in keyed:
                keyed[key] = term
            elif (joined := keyed[key].join(term)) is None:
                apart.append(term)
            else:
                keyed[key] = joined
        return cls(constant, (*keyed.values(), *apart))

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        total = _rounded(self.constant)[0]
        exprs = []
        for term in self.terms:
            expr = term.cp_expr(model, built)
            if isinstance(expr, int):
                total += expr
            else:
                exprs.append(expr)
        return cp_model.LinearExpr.sum(exprs) + total if exprs else total

    def scaled(self, factor: float) -> LinearRealization:
        terms = [term.scaled(factor) for term in self.terms]
        return LinearRealization.of(factor * self.constant, terms)

    def bounds(self) -> tuple[int, int]:
        bounds = [term.bounds() for term in self.terms]
        constant = _rounded(self.constant)[0]
        return constant + sum(b[0] for b in bounds), constant + sum(
            b[1] for b in bounds
        )

    @property
    def entries(self) -> int:
        return sum(term.entries for term in self.terms)


def _is_literal(index: Index) -> bool:
    """Whether index is a 0/1 variable's own position, which is its own true/false."""
    return (
        isinstance(index, InputIndex)
        and (index.lo, index.size) == (0, 2)
        and bool(index.var.is_boolean)
    )


def _proxy(model: cp_model.CpModel, built: Built, term: AbstractRealization) -> Any:
    """term behind a variable of its own, unless it is one or a number."""
    expr = term.cp_expr(model, built)
    if isinstance(expr, (int, cp_model.IntVar)):
        return expr
    var = model.new_int_var(*term.bounds(), "term")
    model.add(var == expr)
    return var


@dataclass(frozen=True, eq=False)
class SwitchRealization(AbstractRealization):
    """factor * rest (rule 4). The factor's distinct values, none past 1 in size,
    stay numbers; one true/false per value is on where the factor takes it at index;
    and the rest is collapsed once into w. Under each true/false one variable g is
    p * w / q rounded, p/q the value: no table crosses factor and rest, and no two
    variables are multiplied."""

    values: tuple[float, ...]
    index: Index
    rest: AbstractRealization

    @cached_property
    def scale(self) -> int:
        """q: 1 when every value is whole, otherwise the largest power of two up to
        MAX_SWITCH_SCALE that keeps q * w, so every p * w, under MAX_SWITCH_PRODUCT;
        0 when none does."""
        widest = max(1, *(abs(bound) for bound in self.rest.bounds()))
        q = 1 if all(v.is_integer() for v in self.values) else MAX_SWITCH_SCALE
        while q * widest >= MAX_SWITCH_PRODUCT:
            q //= 2
        return q

    @property
    def precise(self) -> bool:
        """Whether p/q is within 1/MIN_SWITCH_SCALE of each value, relatively."""
        smallest = min(abs(v) for v in self.values if v)
        return self.scale == 1 or self.scale * smallest >= MIN_SWITCH_SCALE

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        """g under one true/false per value: a 0/1 variable is its own pair, any
        other position is mapped by add_map_domain. The rest sits behind a variable
        of its own when more than two values read it."""
        if not self.scale:
            raise TableLoweringError(None, "a switched product too wide for int64")
        if len(self.values) > 2:
            w = _proxy(model, built, self.rest)
        else:
            w = self.rest.cp_expr(model, built)
        if _is_literal(self.index):
            literals = [~self.index.var, self.index.var]  # type: ignore[attr-defined]
        else:
            literals = [model.new_bool_var("value") for _ in self.values]
            model.add_map_domain(self.index.cp_expr(model, built), literals)
        q = self.scale
        g = model.new_int_var(*self.bounds(), "switch")
        for value, on in zip(self.values, literals):
            p = round(value * q)
            if q == 1:
                model.add(g == p * w).only_enforce_if(on)
            else:  # g is p * w / q to the nearest: a floor would err one way
                model.add(q * g <= p * w + q // 2).only_enforce_if(on)
                model.add(q * g > p * w - q // 2).only_enforce_if(on)
        return g

    def scaled(self, factor: float) -> SwitchRealization:
        return SwitchRealization(self.values, self.index, self.rest.scaled(factor))

    def bounds(self) -> tuple[int, int]:
        q = self.scale
        ends = [
            (round(value * q) * w + q // 2) // q
            for value in self.values
            for w in self.rest.bounds()
        ]
        return min(ends), max(ends)

    @property
    def entries(self) -> int:
        return self.rest.entries + len(self.values) + _entries_of([self.index])

    def key(self) -> Hashable:
        return ("switch", self.index, self.values)

    def join(self, other: Any) -> SwitchRealization | None:
        """Products with one factor are one switch over their rests, unless the sum
        of the rests is too wide for it to stay precise."""
        rest = LinearRealization.of(0.0, (self.rest, other.rest))
        joined = SwitchRealization(self.values, self.index, rest)
        return None if self.precise and other.precise and not joined.precise else joined


@dataclass(frozen=True, eq=False)
class GateRealization(AbstractRealization):
    """rest where the 0/1 literal at literal is at kept, 0 elsewhere (rule 5): the
    literal is its own true/false, and the gate's value where it keeps the term is
    already in rest, so g == w is exact."""

    literal: InputIndex
    kept: int
    rest: AbstractRealization

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        var = self.literal.var
        on, off = (var, ~var) if self.kept else (~var, var)
        g = model.new_int_var(*self.bounds(), "gate")
        model.add(g == self.rest.cp_expr(model, built)).only_enforce_if(on)
        model.add(g == 0).only_enforce_if(off)
        return g

    def scaled(self, factor: float) -> GateRealization:
        return GateRealization(self.literal, self.kept, self.rest.scaled(factor))

    def bounds(self) -> tuple[int, int]:
        lo, hi = self.rest.bounds()
        return min(lo, 0), max(hi, 0)

    @property
    def entries(self) -> int:
        return self.rest.entries

    def key(self) -> Hashable:
        return ("gate", self.literal, self.kept)

    def join(self, other: Any) -> GateRealization:
        """Terms gated by one literal are one gate over their rests."""
        rest = LinearRealization.of(0.0, (self.rest, other.rest))
        return GateRealization(self.literal, self.kept, rest)


@dataclass(frozen=True, eq=False)
class ExtremeRealization(AbstractRealization):
    """A Min (is_min) or Max kept linear: a lin_min or lin_max over its operands,
    each behind a variable of its own (a long sum inside lin_max made CP-SAT's
    presolve run out of memory)."""

    is_min: bool
    operands: tuple[AbstractRealization, ...]

    def build(self, model: cp_model.CpModel, built: Built) -> CpExpr:
        exprs = [_proxy(model, built, operand) for operand in self.operands]
        target = model.new_int_var(*self.bounds(), "min" if self.is_min else "max")
        if self.is_min:
            model.add_min_equality(target, exprs)
        else:
            model.add_max_equality(target, exprs)
        return target

    def scaled(self, factor: float) -> ExtremeRealization:
        """A negative factor turns a Min into a Max."""
        operands = tuple(operand.scaled(factor) for operand in self.operands)
        return ExtremeRealization(self.is_min == (factor >= 0), operands)

    def bounds(self) -> tuple[int, int]:
        pick = min if self.is_min else max
        bounds = [operand.bounds() for operand in self.operands]
        return pick(lo for lo, _ in bounds), pick(hi for _, hi in bounds)

    @property
    def entries(self) -> int:
        return sum(operand.entries for operand in self.operands)


# Each symbol's input: the variable that picks its value, and the symbol's value at
# each of that variable's values, from its lowest to its highest.
Input: TypeAlias = "tuple[cp_model.IntVar, Sequence[float]]"
Inputs: TypeAlias = "Mapping[sympy.Symbol, Input]"


# ------------------------------------------------------------------ a node's list


# The form of an operand in its node's function: as it is (None), a condition
# (CONDITION), or the base of a Mul's factor with a negative rational exponent, which
# the function raises to that exponent (the exponent, a sympy.Rational).
CONDITION = "condition"
Form: TypeAlias = "str | sympy.Rational | None"


@lru_cache(maxsize=1024)
def _lambdified(func: Any, forms: tuple[Form, ...]) -> Callable[..., Any]:
    """The function of a node of sympy class func whose operands take forms,
    lambdified with the numpy module over one placeholder per operand. lambdify runs
    once for each class and forms, and every node of them shares the function.

    The placeholders are plain symbols, operand0, operand1, ..., whose names are
    Python identifiers, so lambdify takes them as they are. It renames every argument
    when any is a Dummy, rebuilding the node once per argument: a Min of 35 operands
    took 10.5 s, against 0.28 s here.

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
    value returned here need not be finite; the node that makes the list refuses
    it."""
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
    the last position fastest."""
    shape = tuple(p.size for p in positions)
    return (
        dict(zip(positions, np.indices(shape).reshape(len(shape), -1))) if shape else {}
    )


def _same_value(value: float) -> float:
    """value to SIGNIFICANT_DIGITS digits: values equal to that share a rank."""
    return 0.0 if value == 0 else float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _ranked(table: TableRealization) -> TableRealization:
    """table over its distinct values, ascending: equal values share one position,
    a CompactIndex from table's; table itself when its values are distinct."""
    first: dict[float, float] = {}
    keys = [_same_value(float(v)) for v in table.values]
    for key, value in zip(keys, table.values):
        first.setdefault(key, float(value))
    if len(first) == len(keys):
        return table
    distinct = sorted(first)
    rank = {key: i for i, key in enumerate(distinct)}
    index = CompactIndex(table.index, tuple(rank[k] for k in keys), len(distinct))
    return TableRealization(values=[first[k] for k in distinct], index=index)


def clamped_ladder(
    operands: Sequence[np.ndarray], is_min: bool
) -> tuple[list[float], float]:
    """The values a Min (is_min) or Max of operands can return, ascending, and the
    bound past which an operand's value is clamped: a Min never exceeds its smallest
    operand maximum, and a Max never falls below its largest operand minimum."""
    if is_min:
        bound = min(float(np.max(o)) for o in operands)
    else:
        bound = max(float(np.min(o)) for o in operands)
    clamp = np.minimum if is_min else np.maximum
    first: dict[float, float] = {}
    for values in operands:
        for v in clamp(values, bound):
            first.setdefault(_same_value(float(v)), float(v))
    return [first[k] for k in sorted(first)], bound


def rungs_on(
    ladder: Sequence[float], bound: float, values: np.ndarray, is_min: bool
) -> tuple[int, ...]:
    """Each of values, clamped at bound, as its position on ladder, to
    SIGNIFICANT_DIGITS digits."""
    rung = {_same_value(v): i for i, v in enumerate(ladder)}
    clamped = (np.minimum if is_min else np.maximum)(values, bound)
    return tuple(rung[_same_value(float(v))] for v in clamped)


def fold_reachable(
    ladders: Sequence[TableRealization], cap: int
) -> TableRealization | None:
    """Rule 2: Min/Max factors, each its ladder on the min or max of its operands'
    rungs, as one list over the tuples of their rungs that can occur together, the
    product of their values at each.

    Operands sharing an input join one group. Each group is one step: the state, the
    tuple of each factor's Min/Max over the groups read so far, is combined with the
    group's position and compacted onto the tuples that occur, so only tuples some
    assignment reaches are numbered. An operand on no input seeds the start state.
    None when a step or the states pass cap."""
    extremes = [ladder.index for ladder in ladders]
    picks = [min if e.is_min else max for e in extremes]  # type: ignore[attr-defined]

    def joined(f: int, a: int | None, b: int | None) -> int | None:
        return b if a is None else a if b is None else picks[f](a, b)

    start: list[int | None] = [None] * len(ladders)
    groups: list[tuple[list[Index], list[tuple[int, Index]]]] = []
    for f, extreme in enumerate(extremes):
        for rung in extreme.rungs:  # type: ignore[attr-defined]
            leaves: list[Index] = [
                s for s in rung.steps() if isinstance(s, (InputIndex, LocateIndex))
            ]
            if not leaves:
                start[f] = joined(f, start[f], int(rung.on_grid({})))
                continue
            hit = [g for g in groups if set(g[0]) & set(leaves)]
            positions = list(dict.fromkeys([p for g in hit for p in g[0]] + leaves))
            members = [m for g in hit for m in g[1]] + [(f, rung)]
            groups = [g for g in groups if all(g is not h for h in hit)]
            groups.append((positions, members))

    states: list[tuple[int | None, ...]] = [tuple(start)]
    index: Index = UNIT
    for positions, members in groups:
        joint = math.prod(p.size for p in positions)
        if len(states) * joint > cap:
            return None
        grid = _grid(positions)
        reached: list[list[int | None]] = [[None] * len(ladders) for _ in range(joint)]
        for f, rung in members:
            for j, at in enumerate(np.broadcast_to(rung.on_grid(grid), (joint,))):
                reached[j][f] = joined(f, reached[j][f], int(at))
        numbered: dict[tuple[int | None, ...], int] = {}
        remap = tuple(
            numbered.setdefault(
                tuple(joined(f, s, r) for f, (s, r) in enumerate(zip(state, row))),
                len(numbered),
            )
            for state in states
            for row in reached
        )
        if len(numbered) > cap:
            return None
        index = CompactIndex(
            CombineIndex(index, _combined(positions)), remap, len(numbered)
        )
        states = list(numbered)
    values = [
        math.prod(
            ladder.values[s] for ladder, s in zip(ladders, state) if s is not None
        )
        for state in states
    ]
    return TableRealization(values=values, index=index)


# -------------------------------------------------------------------------- nodes


def children_of(expr: sympy.Basic) -> tuple[sympy.Basic, ...]:
    """A node's children: its arguments, except a Piecewise, whose children are its
    arms' values and conditions, and a RelayoutCharge, whose children are its literal
    and its division (its prices are data)."""
    if isinstance(expr, sympy.Piecewise):
        return tuple(part for pair in expr.args for part in pair.args)
    if isinstance(expr, RelayoutCharge):
        return expr.args[:2]
    return expr.args


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


class Candidate(NamedTuple):
    """One way to make a node's table: its size and its entries, its lookups
    included, known before its list is computed, and make, which computes it."""

    size: int
    entries: int
    make: Callable[[], TableRealization]


@dataclass(eq=False)
class Node:
    """The CP-SAT counterpart of one sympy node, which realizes itself.

    expr is the sympy node, injected and only read. children holds the Nodes of
    children_of(expr), made before it; a subexpression shared by two parents is one
    Node that both hold. cap bounds one table. No model is involved until cp_expr.

    A node has two reads, each made the first time it is read and kept:

    - its table (tabulated): the cheapest of its ways to be one table, in entries
      (rule 3): one table over its inputs, the table from its parts, and what a
      subclass adds. A parent that computes its own list from it reads this;
    - its realization: what a linear reader takes, its table here; a subclass adds
      linear ways.

    A node refuses a value of its own that is not finite, so one below a finite root
    is refused at the node that makes it.
    """

    expr: sympy.Basic
    children: tuple[Node, ...]
    cap: int

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, TableRealization]:
        """Each free symbol's column on its input, gathered from the children."""
        columns: dict[sympy.Symbol, TableRealization] = {}
        for child in self.children:
            columns.update(child.columns)
        return columns

    @cached_property
    def realization(self) -> AbstractRealization:
        """What a linear reader takes; a refusal names the node."""
        try:
            return self.realize()
        except TableLoweringError as exc:
            if exc.node is not None:
                raise
            raise TableLoweringError(self.expr, exc.reason) from exc

    def realize(self) -> AbstractRealization:
        return self.table

    @property
    def table(self) -> TableRealization:
        """Its table; raises TableLoweringError naming the node when none fits the
        cap."""
        table = self.tabulated
        if table is None:
            raise TableLoweringError(
                self.expr,
                f"one table over its inputs would hold {self.candidates()[0].size} "
                f"entries, past the cap of {self.cap}, and no other table fits",
            )
        return table

    @cached_property
    def tabulated(self) -> TableRealization | None:
        """Its cheapest table in entries, or None when no way fits the cap. Only the
        way it takes is computed, so a table past the cap is never evaluated. A value
        that is not finite raises, naming the node."""
        fitting = [c for c in self.candidates() if c.size <= self.cap]
        if not fitting:
            return None
        return self._finite(min(fitting, key=lambda c: c.entries).make())

    def candidates(self) -> list[Candidate]:
        """Its ways to be one table: over its inputs, the first, and from its
        parts."""
        symbols = sorted(self.columns, key=sympy.default_sort_key)
        inputs = list(dict.fromkeys(self.columns[s].index for s in symbols))
        sources, forms = self.operands()
        ways = [self._candidate(inputs, sources, forms)]
        # From its parts: a function of one list on that list's position, and over
        # several lists their ranks combined, equal values sharing a position.
        tables = [s.tabulated if isinstance(s, Node) else s for s in sources]
        if all(t is not None for t in tables):
            if (
                len(
                    {t.index for t in tables if isinstance(t, TableRealization)}
                    - {UNIT}
                )
                > 1
            ):
                tables = [
                    _ranked(t) if isinstance(t, TableRealization) else t for t in tables
                ]
            ways.append(self._candidate(None, tables, forms))
        return ways

    def _candidate(
        self,
        positions: Sequence[Index] | None,
        operands: Sequence[Node | TableRealization | float | None],
        forms: tuple[Form, ...],
    ) -> Candidate:
        """Its function over operands on positions, the operands' own when None."""
        if positions is None:
            tables = [o for o in operands if isinstance(o, TableRealization)]
            positions = [p for p in dict.fromkeys(t.index for t in tables) if p != UNIT]
        size = math.prod(p.size for p in positions)
        return Candidate(
            size,
            size + _entries_of(positions),
            lambda: self._evaluate(positions, operands, forms),
        )

    def _evaluate(
        self,
        positions: Sequence[Index],
        operands: Sequence[Node | TableRealization | float | None],
        forms: tuple[Form, ...],
    ) -> TableRealization:
        """Its function (_lambdified) applied to operands at every joint assignment of
        positions: a node's table, an operand on the table's own position as it is,
        one on other positions at its own (Index.on_grid), and a constant at every
        entry."""
        index = _combined(positions)
        grid = _grid(positions)
        arguments = []
        for operand in operands:
            if isinstance(operand, Node):
                operand = operand.table
            if not isinstance(operand, TableRealization):
                arguments.append(np.full(index.size, operand))
            elif operand.index == index:
                arguments.append(operand.values)
            else:
                at = operand.index.on_grid(grid)
                arguments.append(np.broadcast_to(operand.values[at], (index.size,)))
        values = _values(
            self.expr,
            index.size,
            lambda: _apply(_lambdified(self.expr.func, forms), arguments),
        )
        return TableRealization(values=values, index=index)

    def operands(self) -> tuple[list[Node | float], tuple[Form, ...]]:
        """What the node's function reads, in order, and the form each takes in it
        (_lambdified): the arguments, a Piecewise's arms' values and conditions, value
        then condition, a RelayoutCharge's prices as constants, and in a Mul the base
        of a factor with a negative rational exponent in its place, with the exponent.
        Every other operand is a child, as it is or as a condition."""
        children = dict(zip(children_of(self.expr), self.children))
        parts = children_of(self.expr) if self.expr.is_Piecewise else self.expr.args
        sources: list[Node | float] = []
        forms: list[Form] = []
        for part in parts:
            if _divides(self.expr, part):
                sources.append(children[part].children[0])
                forms.append(part.exp)  # type: ignore[attr-defined]
            elif part in children:
                sources.append(children[part])
                forms.append(CONDITION if _is_condition(part) else None)
            else:
                sources.append(_constant(part))
                forms.append(None)
        return sources, tuple(forms)

    def _finite(self, table: TableRealization) -> TableRealization:
        """table, refused naming the node when a value of it is not finite."""
        if not table.finite:
            raise TableLoweringError(self.expr, "a value that is not finite")
        return table

    def cp_expr(self, model: cp_model.CpModel, built: Built | None = None) -> CpExpr:
        """The node in model, in COST_SCALE units: its realization built. Reading the
        realization realizes the node, so a refusal there comes before anything is
        added to the model; a refusal from the build names the node too."""
        realization = self.realization
        try:
            return realization.cp_expr(model, built)
        except TableLoweringError as exc:
            if exc.node is not None:
                raise
            raise TableLoweringError(self.expr, exc.reason) from exc


@dataclass(eq=False)
class InputNode(Node):
    """A symbol: input data. input is its variable and its value at each of that
    variable's values, injected by the factory from the inputs, or None for a symbol
    with no input, which raises when it is read. The node makes its own position on
    the variable: an InputIndex, or a LocateIndex when the domain has gaps."""

    input: Input | None = None

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, TableRealization]:
        """Its own column: its values on its variable's position. Raises
        TableLoweringError naming the symbol when it has no input, when its variable
        is not an integer variable, and when its values are not one number per value
        of its variable."""
        if self.input is None:
            raise TableLoweringError(self.expr, "a symbol with no input")
        var, values = self.input
        if not isinstance(var, cp_model.IntVar):
            raise TableLoweringError(
                self.expr, f"a variable that is not an IntVar ({type(var).__name__})"
            )
        bounds = list(var.proto.domain)  # its intervals, flattened: lowest first
        intervals = list(zip(bounds[0::2], bounds[1::2]))
        count = sum(hi - lo + 1 for lo, hi in intervals)
        if len(values) != count:
            raise TableLoweringError(
                self.expr,
                f"{len(values)} values for a variable with {count}, "
                f"{bounds[0]} to {bounds[-1]}",
            )
        index: Index
        if len(intervals) == 1:
            index = InputIndex(var, size=count, lo=bounds[0])
        else:
            index = LocateIndex(
                var, tuple(v for lo, hi in intervals for v in range(lo, hi + 1))
            )
        try:
            return {self.expr: TableRealization(values=values, index=index)}
        except (TypeError, ValueError) as exc:
            raise TableLoweringError(
                self.expr, f"values that are not one number each ({exc})"
            ) from exc

    @cached_property
    def tabulated(self) -> TableRealization:
        """Its column."""
        (column,) = self.columns.values()
        return self._finite(column)


@dataclass(eq=False)
class ConstantNode(Node):
    """A node with no free symbol: a number, true or false, or an operation on
    numbers. Its table is its value on the UnitIndex position."""

    @cached_property
    def tabulated(self) -> TableRealization:
        value = np.array([_constant(self.expr)])
        return self._finite(TableRealization(values=value, index=UNIT))


@dataclass(eq=False)
class AddNode(Node):
    """A sum: read linearly, a LinearRealization of its children's realizations."""

    def realize(self) -> LinearRealization:
        return LinearRealization.of(0.0, (c.realization for c in self.children))


def _scaled(realization: AbstractRealization, factor: float) -> AbstractRealization:
    """factor times realization; itself for 1, so its readers share one build."""
    return realization if factor == 1 else realization.scaled(factor)


def _gate(table: TableRealization | None) -> tuple[InputIndex, int, float] | None:
    """(literal, kept, value) when table is a list on one 0/1 literal that is 0 at one
    of its values and value at the other, kept; None otherwise."""
    if table is None or not _is_literal(table.index):
        return None
    off, on = (float(v) for v in table.values)
    if (off == 0) == (on == 0):
        return None
    return table.index, int(off == 0), on or off  # type: ignore[return-value]


@dataclass(eq=False)
class MulNode(Node):
    """A product. Read linearly: a constant times one child is the child's realization
    scaled; any other product is a gate on a 0/1 literal (rule 5), else a switch on its
    factor with the fewest values (rule 4), or its table when that has fewer entries
    (rule 3). Its table adds the fold of Min/Max factors (rule 2)."""

    def realize(self) -> AbstractRealization:
        coefficient = math.prod(
            float(c.table.values[0])
            for c in self.children
            if isinstance(c, ConstantNode)
        )
        factors = [c for c in self.children if not isinstance(c, ConstantNode)]
        if len(factors) == 1:
            return _scaled(factors[0].realization, coefficient)
        table = self.tabulated
        product = self._product(coefficient, factors)
        if isinstance(product, SwitchRealization) and not product.precise:
            product = product if table is None else None  # the table is exact
        fitting = [way for way in (table, product) if way is not None]
        if not fitting:
            raise TableLoweringError(
                self.expr, "no table fits the cap and no factor can be switched on"
            )
        return min(fitting, key=lambda way: way.entries)

    def _product(
        self, coefficient: float, factors: list[Node]
    ) -> AbstractRealization | None:
        """coefficient times factors, read linearly: a gate when a factor is one, else
        a switch on the factor with the fewest distinct values among those with a
        table. None when there is none, or its values pass the cap."""

        def rest(factor: Node, scale: float) -> AbstractRealization:
            others = [f for f in factors if f is not factor]
            if len(others) == 1:
                return _scaled(others[0].realization, scale)
            expr = sympy.Mul(*(other.expr for other in others), evaluate=False)
            return _scaled(MulNode(expr, tuple(others), self.cap).realization, scale)

        for factor in factors:
            gate = _gate(factor.tabulated)
            if gate is not None:
                literal, kept, value = gate
                return GateRealization(literal, kept, rest(factor, coefficient * value))
        ranked = {f: _ranked(t) for f in factors if (t := f.tabulated) is not None}
        if not ranked:
            return None
        factor = min(ranked, key=lambda f: ranked[f].index.size)
        table = ranked[factor]
        if table.index.size > self.cap:
            return None
        # The factor's size goes into the rest's lists before they are rounded, as a
        # power of two: the switch multiplies the rest's rounding by values up to 1.
        largest = float(np.max(np.abs(table.values)))
        size = 2.0 ** max(math.ceil(math.log2(largest)), 0) if largest else 1.0
        if table.index.size == 1:
            return rest(factor, coefficient * float(table.values[0]))
        values = tuple(float(v) / size for v in table.values)
        return SwitchRealization(values, table.index, rest(factor, coefficient * size))

    def candidates(self) -> list[Candidate]:
        """Rule 2 too: its Min/Max factors folded onto the tuples of their rungs that
        occur together, crossed with its other factors' ranks, when crossing the
        ladders themselves would pass the cap."""
        ways = super().candidates()
        sources, forms = self.operands()
        extreme = [
            isinstance(s, ExtremeNode) and f is None for s, f in zip(sources, forms)
        ]
        ladders = [s.ladder for s, e in zip(sources, extreme) if e]  # type: ignore[union-attr]
        if len(ladders) < 2 or None in ladders:
            return ways
        if math.prod(ladder.index.size for ladder in ladders) <= self.cap:
            return ways
        fold = fold_reachable(ladders, self.cap)
        others = [
            s.tabulated if isinstance(s, Node) else s
            for s, e in zip(sources, extreme)
            if not e
        ]
        if fold is None or None in others:
            return ways
        ranked = [_ranked(o) if isinstance(o, TableRealization) else o for o in others]
        folded_forms = (None, *(f for f, e in zip(forms, extreme) if not e))
        return [*ways, self._candidate(None, [fold, *ranked], folded_forms)]


@dataclass(eq=False)
class ExtremeNode(Node):
    """A Min or a Max. Its table adds the table by rung (rule 1); read linearly, it
    stays a lin_min/lin_max when that needs under 1/LINEAR_MARGIN of its table's
    entries, or when it has no table."""

    @property
    def is_min(self) -> bool:
        return isinstance(self.expr, sympy.Min)

    def realize(self) -> AbstractRealization:
        table = self.tabulated
        operands = tuple(child.realization for child in self.children)
        linear = ExtremeRealization(self.is_min, operands)
        if table is None or LINEAR_MARGIN * linear.entries < table.entries:
            return linear
        return table

    def candidates(self) -> list[Candidate]:
        ladder = self.ladder
        if ladder is None:
            return super().candidates()
        entries = ladder.index.size + _entries_of([ladder.index])
        return [
            *super().candidates(),
            Candidate(ladder.index.size, entries, lambda: ladder),
        ]

    @cached_property
    def ladder(self) -> TableRealization | None:
        """Rule 1: its ladder, the values it can return, read at the min or max of
        its operands' rungs, each operand's clamped values remapped onto the ladder.
        None when an operand has no table."""
        tables = [t for t in (c.tabulated for c in self.children) if t is not None]
        if len(tables) != len(self.children):
            return None
        ladder, bound = clamped_ladder([t.values for t in tables], self.is_min)
        rungs = tuple(
            CompactIndex(
                t.index, rungs_on(ladder, bound, t.values, self.is_min), len(ladder)
            )
            for t in tables
        )
        index = ExtremeIndex(rungs, self.is_min, len(ladder))
        return TableRealization(values=ladder, index=index)


@dataclass(eq=False)
class RelayoutChargeNode(Node):
    """``RelayoutCharge(is_lx, division, *prices)``, is_lx * prices[division]. Read
    linearly with a 0/1 literal, it is its prices on the division gated by the
    literal (rule 5), costing the division's menu once."""

    def realize(self) -> AbstractRealization:
        literal, division = self.children
        gate = _gate(literal.tabulated)
        if gate is None:
            return self.table
        index, kept, value = gate
        sources, forms = self.operands()
        prices = self._evaluate([division.table.index], [value, *sources[1:]], forms)
        return GateRealization(index, kept, self._finite(prices))


NODE_CLASSES: dict[type, type[Node]] = {
    sympy.Add: AddNode,
    sympy.Mul: MulNode,
    sympy.Min: ExtremeNode,
    sympy.Max: ExtremeNode,
    RelayoutCharge: RelayoutChargeNode,
}


# ------------------------------------------------------------------------ the walk


def _read_gated(mul: sympy.Basic, literals: set[sympy.Symbol]) -> sympy.Basic:
    """mul with its other factors read at each gate's kept value: a gate is a factor of
    one 0/1 literal that is 0 at one of its values, and where it is not, the product is
    0 whatever the others are."""
    args = list(mul.args)
    for i in range(len(args)):
        symbols = args[i].free_symbols
        if len(symbols) != 1 or not symbols <= literals:
            continue
        (literal,) = symbols
        off, on = (args[i].xreplace({literal: v}) for v in (sympy.S.Zero, sympy.S.One))
        if not (off.is_Number and on.is_Number) or (off == 0) == (on == 0):
            continue
        kept = sympy.S.One if off == 0 else sympy.S.Zero
        args = [
            a if j == i else a.xreplace({literal: kept}) for j, a in enumerate(args)
        ]
    return mul.func(*args)


class TableLowering:
    """Plans cost expressions as tables: a factory for Nodes. It makes one Node
    per node, of the class the node's sympy class calls for, and hands it what it
    reads; the nodes do the rest. It holds no model, so planning cannot write to one,
    and it keeps nothing between calls of lower.

    inputs gives each symbol's variable and values, which the factory hands to the
    symbol's InputNode; cap bounds the entries of one table, and is handed to every
    node.
    """

    def __init__(self, inputs: Inputs, *, cap: int) -> None:
        self.inputs = inputs
        self.cap = cap

    def lower(self, root: sympy.Basic) -> Node:
        """root as a plan: the Node of root normalized, over the graph of nodes under
        it. Nothing is realized here."""
        normalized = self.normalize(root)
        return self.rewrite(normalized)[normalized]

    def normalize(self, root: sympy.Basic) -> sympy.Basic:
        """root with a relayout charge on a division that has no input, a source this
        solve does not divide, as 0, and a gated product's other factors read at the
        gate's kept value, so no table under a gate reads its literal."""
        root = root.replace(
            lambda e: isinstance(e, RelayoutCharge)
            and e.args[1].is_Symbol
            and e.args[1] not in self.inputs,
            lambda e: sympy.S.Zero,
        )
        literals = {
            symbol
            for symbol, (var, values) in self.inputs.items()
            if isinstance(var, cp_model.IntVar)
            and var.is_boolean
            and [*values] == [0, 1]
        }
        return root.replace(
            lambda e: e.is_Mul and not literals.isdisjoint(e.free_symbols),
            lambda e: _read_gated(e, literals),
        )

    def rewrite(self, root: sympy.Basic) -> dict[sympy.Basic, Node]:
        """Each node under root, once, as a Node holding its children's Nodes,
        made after them, in the order of sympy.postorder_traversal with its repeats
        dropped. An arm's ExprCondPair gets no Node. Nothing is realized here;
        root is not changed."""
        nodes: dict[sympy.Basic, Node] = {}
        for expr in dict.fromkeys(sympy.postorder_traversal(root)):
            if isinstance(expr, ExprCondPair):
                continue
            nodes[expr] = self._node(expr, tuple(nodes[c] for c in children_of(expr)))
        return nodes

    def _node(self, expr: sympy.Basic, children: tuple[Node, ...]) -> Node:
        """expr's Node: an InputNode holding its symbol's input, a ConstantNode for a
        node with no free symbol, the class NODE_CLASSES gives its sympy class, and a
        Node for any other."""
        if isinstance(expr, sympy.Symbol):
            return InputNode(expr, children, self.cap, input=self.inputs.get(expr))
        if not expr.free_symbols:
            return ConstantNode(expr, children, self.cap)
        kind = next(
            (NODE_CLASSES[c] for c in type(expr).__mro__ if c in NODE_CLASSES), Node
        )
        return kind(expr, children, self.cap)


def _constant(expr: sympy.Basic) -> float:
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc


# --------------------------------------------------------------- the printer's maps


def inputs_from_maps(
    sym_map: Mapping[str, Any], buffer_map: Mapping[str, Any], *, cap: int
) -> dict[str, Input]:
    """Each symbol of the printer's two maps, by name, as its variable and its values:
    a buffer's split or tile symbol its value at each of its buffer's divisions, and
    any other integer variable its domain's values. A domain past the cap is no input.
    Raises TableLoweringError for two menu lengths on one division."""
    inputs: dict[str, Input] = {}
    lengths: dict[int, int] = {}
    for name, (wrapper, raw) in buffer_map.items():
        if wrapper is None:
            continue
        division = wrapper.division
        if lengths.setdefault(division.index, len(raw)) != len(raw):
            raise TableLoweringError(
                None, f"{name}: menus of two lengths on one division"
            )
        inputs[name] = (division, [float(v) for v in raw])
    for name, var in sym_map.items():
        if name in inputs or not isinstance(var, cp_model.IntVar):
            continue
        bounds = list(var.proto.domain)
        intervals = list(zip(bounds[0::2], bounds[1::2]))
        if sum(hi - lo + 1 for lo, hi in intervals) <= cap:
            values = [float(v) for lo, hi in intervals for v in range(lo, hi + 1)]
            inputs[name] = (var, values)
    return inputs


def lower_cost_with_tables(
    model: cp_model.CpModel,
    sym_map: Mapping[str, Any],
    buffer_map: Mapping[str, Any],
    cost_expr: sympy.Basic,
    *,
    cap: int,
) -> CpExpr:
    """cost_expr as a CP-SAT objective in COST_SCALE units, an int when it is
    constant. sym_map and buffer_map are the printer's: a number in sym_map is the
    symbol's value. Copies model first; on any failure swaps the copy back and raises
    TableLoweringError, from the original exception when that was of another type."""
    copy = model.clone()
    try:
        by_name = inputs_from_maps(sym_map, buffer_map, cap=cap)
        expr = sympy.sympify(cost_expr)
        numbers = {
            s: sympy.Integer(value) if isinstance(value, int) else sympy.Float(value)
            for s in expr.free_symbols
            if s.name not in by_name
            and isinstance(value := sym_map.get(s.name), (int, float))
        }
        expr = expr.xreplace(numbers)
        inputs = {s: by_name[s.name] for s in expr.free_symbols if s.name in by_name}
        return TableLowering(inputs, cap=cap).lower(expr).cp_expr(model)
    except Exception as exc:
        model.proto.copy_from(copy.proto)
        model.rebuild_constant_map()
        if isinstance(exc, TableLoweringError):
            raise
        raise TableLoweringError(None, f"{type(exc).__name__}: {exc}") from exc
