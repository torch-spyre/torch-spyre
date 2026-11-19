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
- the nodes: ``TableLowering`` is a factory that makes one ``Node`` per node of the
  expression, of a class chosen by the node's sympy class. Each node realizes
  itself, here as a table, the first time its realization is read, and reads its
  children's realizations as it does, so reading the root realizes every node under
  it. A node refuses a value of its own that is not finite, whether or not its
  parent's value depends on it. A symbol's ``InputNode`` is handed its variable and
  its values, and makes its own position on that variable. The plan, the root's
  ``Node``, holds no model either;
- the build: the root's ``cp_expr`` reads its realization, which builds itself into
  the model (``TableRealization.cp_expr``), rounded onto ``COST_SCALE``, its position
  built by ``Index.cp_expr``. Only ``cp_expr`` methods touch the model.

This is the base: one table per node, every value computed by ``sympy.lambdify`` over
the node's inputs, and the root read as one lookup. It lowers an expression whose
inputs fit the cap together and refuses anything wider. Nothing imports it yet.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import sympy
from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair

from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

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
    equal parts are equal and hash alike. TableRealization and the nodes do not change
    when a step is added. The steps here are InputIndex, UnitIndex and
    CombineIndex."""

    # The number of positions, 0 .. size - 1: a field of each step, given or
    # computed from its parts when it is made.
    size: int

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The position as an expression in model, from its parts' cp_expr. A leaf over
        a variable reads the variable. Each call builds anew, and a part reached twice
        is built twice: no step of this stage adds to the model."""


@dataclass(frozen=True)
class InputIndex(Index):
    """A leaf: a variable's own position, v -> v - lo, for a variable whose values run
    from lo to lo + size - 1. An InputNode makes one on its variable, which is already
    in the model. Two on one variable are equal: one position."""

    var: cp_model.IntVar
    size: int
    lo: int = 0

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return self.var - self.lo if self.lo else self.var


@dataclass(frozen=True)
class UnitIndex(Index):
    """A leaf: the one position of a constant, on no input."""

    size: int = field(default=1, init=False)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
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
    """What a node is realized as: the one thing a reader and the objective take of
    it. A value, with no model in it, that builds its own CP-SAT side. Each kind of
    realization is a subclass: here a TableRealization, and from Stage 3 a
    LinearRealization too, what a sum is realized as."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, as a CP-SAT expression in COST_SCALE units."""


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
        Raises TableLoweringError for an entry past MAX_ENTRY. Its values are finite:
        the node that makes a table refuses one that is not (Node._finite)."""
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


# Each symbol's input: the variable that picks its value, and the symbol's value at
# each of that variable's values, from its lowest to its highest.
Input: TypeAlias = tuple[cp_model.IntVar, Sequence[float]]
Inputs: TypeAlias = Mapping[sympy.Symbol, Input]


# ----------------------------------------------------------- values by lambdify


def _compile(expr: sympy.Basic) -> tuple[list[sympy.Symbol], Callable[..., Any]]:
    symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
    return symbols, sympy.lambdify(symbols, expr, modules="numpy")


def values_on_grid(
    expr: sympy.Basic,
    columns: Mapping[sympy.Symbol, TableRealization],
    positions: Sequence[Index],
) -> np.ndarray:
    """expr at every joint assignment of positions, row-major, the last position
    fastest: entry i1 * s1 + ... + im * sm, s_k the product of the sizes after k.

    Each free symbol of expr is read from its column in columns, at its own position,
    which must be one of positions; two symbols on one position move together. expr
    is lambdified with the numpy module on each call: a node makes one call, when it
    builds its table. A function that takes scalars only, as RelayoutCharge's
    ``_imp_`` does, is evaluated point by point: on arrays it raises TypeError or,
    where it tests a value with ``if``, ValueError. Raises TableLoweringError naming
    expr when it has no numeric value, and for anything else evaluating it raises.

    A value returned here need not be finite: a division by zero is infinite, and a
    Piecewise with no arm true is NaN. The node that makes the list refuses it
    (Node.realize).
    """
    shape = tuple(p.size for p in positions)
    grid = np.indices(shape).reshape(len(shape), -1) if shape else np.zeros((0, 1), int)
    count = grid.shape[1]
    axis = {p: k for k, p in enumerate(positions)}
    symbols, func = _compile(expr)
    arguments = []
    for symbol in symbols:
        table = columns[symbol]
        arguments.append(table.values[grid[axis[table.index]]])
    with np.errstate(all="ignore"):
        try:
            try:
                result = np.asarray(func(*arguments), dtype=float)
            except (TypeError, ValueError):
                result = np.array(
                    [func(*(a[i] for a in arguments)) for i in range(count)],
                    dtype=float,
                )
        except NameError as exc:
            raise TableLoweringError(expr, f"no numeric value ({exc})") from exc
        except Exception as exc:
            raise TableLoweringError(
                expr, f"no value ({type(exc).__name__}: {exc})"
            ) from exc
    return np.broadcast_to(result, (count,)).astype(float)


# ------------------------------------------------------------------------ the walk


def children_of(expr: sympy.Basic) -> tuple[sympy.Basic, ...]:
    """A node's children: its arguments, except a Piecewise, whose children are its
    arms' values and conditions, and a RelayoutCharge, whose children are its literal
    and its division (its prices are data)."""
    if isinstance(expr, sympy.Piecewise):
        return tuple(part for pair in expr.args for part in pair.args)
    if isinstance(expr, RelayoutCharge):
        return expr.args[:2]
    return expr.args


@dataclass(eq=False)
class Node:
    """The CP-SAT counterpart of one sympy node, which realizes itself.

    expr is the sympy node, injected and only read: it says which operation the node
    is, and gives its arguments and its free symbols. The Nodes make the same graph
    as the sympy nodes: children holds the Nodes of children_of(expr), made before
    it, and a subexpression shared by two parents is one Node that both hold. A
    node reads only its children: the columns of its inputs are gathered from them,
    and it reads their realizations when it is realized. cap bounds its table. No
    model is involved until cp_expr.

    A node's realization, an AbstractRealization, is what a reader and the objective
    take of it: in this stage always its table, one TableRealization. Its columns and
    its realization are computed the first time they are read, and kept. A node
    refuses a value of its own that is not finite, so one below a finite root is
    refused at the node that makes it: an infinite Piecewise arm at the arm, taken or
    not. This class is an operation over free symbols; InputNode and ConstantNode are
    the leaves. A subclass changes how a node is realized by overriding realize.
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
        """What the node is realized as, made the first time it is read and kept;
        realize raises TableLoweringError naming the node when it has none."""
        return self.realize()

    def realize(self) -> AbstractRealization:
        """Realizes the node: one table over its inputs, their positions combined in
        the order of their symbols sorted by sympy.default_sort_key, the last
        fastest. Raises TableLoweringError naming the node when that table would pass
        the cap, one past the cap never evaluated, and when a value of it is not
        finite. Under the cap it reads its children's realizations first, so a child
        with a value that is not finite is refused at the child, naming it, whether
        or not this node's value depends on it."""
        symbols = sorted(self.columns, key=sympy.default_sort_key)
        positions = list(dict.fromkeys(self.columns[s].index for s in symbols))
        entries = math.prod(p.size for p in positions)
        if entries > self.cap:
            raise TableLoweringError(
                self.expr,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {self.cap}",
            )
        for child in self.children:
            # realizes the child, which refuses its own value that is not finite
            child.realization
        index = positions[0]
        for position in positions[1:]:
            index = CombineIndex(index, position)
        values = values_on_grid(self.expr, self.columns, positions)
        return self._finite(TableRealization(values=values, index=index))

    def _finite(self, table: TableRealization) -> TableRealization:
        """table, refused naming the node when a value of it is not finite."""
        if not table.finite:
            raise TableLoweringError(self.expr, "a value that is not finite")
        return table

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The node in model, in COST_SCALE units: its realization built
        (TableRealization.cp_expr), one lookup or a number. Reading the realization
        realizes the node, so a refusal comes before anything is added to the model,
        and a refusal from the build, an entry past MAX_ENTRY, names the node too.
        Each call builds anew."""
        realization = self.realization
        try:
            return realization.cp_expr(model)
        except TableLoweringError as exc:
            if exc.node is not None:
                raise
            raise TableLoweringError(self.expr, exc.reason) from exc


@dataclass(eq=False)
class InputNode(Node):
    """A symbol: input data. input is its variable and its value at each of that
    variable's values, injected by the factory from the inputs, or None for a symbol
    with no input, which raises when it is read. The node makes its own position, an
    InputIndex on its variable, so the caller makes none."""

    input: Input | None = None

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, TableRealization]:
        """Its own column: its values on its variable's position, made here from the
        variable's lowest and highest values. Raises TableLoweringError naming the
        symbol, so a parent that reads it is refused by the symbol's name, when it has
        no input, when its variable is not an integer variable or its domain has gaps
        (a position lookup for one comes with Stage 11), and when its values are not
        one number per value of its variable."""
        if self.input is None:
            raise TableLoweringError(self.expr, "a symbol with no input")
        var, values = self.input
        if not isinstance(var, cp_model.IntVar):
            raise TableLoweringError(
                self.expr, f"a variable that is not an IntVar ({type(var).__name__})"
            )
        domain = list(var.proto.domain)  # its intervals, flattened: lowest first
        if len(domain) != 2:
            raise TableLoweringError(self.expr, "a variable whose domain has gaps")
        lo, hi = domain
        if len(values) != hi - lo + 1:
            raise TableLoweringError(
                self.expr,
                f"{len(values)} values for a variable with {hi - lo + 1}, {lo} to {hi}",
            )
        index = InputIndex(var, size=len(values), lo=lo)
        try:
            return {self.expr: TableRealization(values=values, index=index)}
        except (TypeError, ValueError) as exc:
            raise TableLoweringError(
                self.expr, f"values that are not one number each ({exc})"
            ) from exc

    def realize(self) -> TableRealization:
        """Its column."""
        (column,) = self.columns.values()
        return self._finite(column)


@dataclass(eq=False)
class ConstantNode(Node):
    """A node with no free symbol: a number, true or false, or an operation on
    numbers. It is realized as its value on the UnitIndex position."""

    def realize(self) -> TableRealization:
        """Its value, one entry on the UnitIndex position."""
        value = np.array([_constant(self.expr)])
        return self._finite(TableRealization(values=value, index=UNIT))


class TableLowering:
    """Plans cost expressions as tables: a factory for Nodes. It makes one Node
    per node, of the class the node's sympy class calls for, and hands it what it
    reads; the nodes do the rest. It holds no model, so planning cannot write to one,
    and it keeps nothing between calls of lower.

    inputs gives each symbol's variable and values, which the factory hands to the
    symbol's InputNode; cap bounds the entries of one table, the length of its list,
    and is handed to every node.
    """

    def __init__(self, inputs: Inputs, *, cap: int) -> None:
        self.inputs = inputs
        self.cap = cap

    def lower(self, root: sympy.Basic) -> Node:
        """root as a plan: root's Node, over the graph of nodes under it. Nothing is
        realized here; the root's cp_expr reads it, which realizes it, and through
        it every node under it."""
        return self.rewrite(root)[root]

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
        node with no free symbol, and a Node for every other node."""
        if isinstance(expr, sympy.Symbol):
            return InputNode(expr, children, self.cap, input=self.inputs.get(expr))
        if not expr.free_symbols:
            return ConstantNode(expr, children, self.cap)
        return Node(expr, children, self.cap)


def _constant(expr: sympy.Basic) -> float:
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
