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
"""The table lowering of a CP-SAT cost objective: one table per node and a sum kept
linear, printed by a sympy Printer, each node's list computed from its operands' lists.
No device and no compile: each test builds its inputs by hand and solves the lowered
model with CP-SAT.

*Exact* means: every assignment of the inputs has exactly one solution, so each
position and the cost are determined by the inputs, and that solution's cost is
COST_SCALE times the expression evaluated by sympy, within the rounding bound. A list
is computed in floating point and rounded once, where it enters CP-SAT: it is held to
the expression's value, not to the bits of another evaluator.
"""

import itertools
import math
import sys
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pytest
import sympy

# ortools is not installed on s390x or ppc64le (pyproject.toml), and the table module
# imports it at its top: skip the file there rather than fail to collect it.
pytest.importorskip("ortools.sat.python.cp_model")

from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.printing.printer import Printer

from torch_spyre._inductor.scratchpad.cpsat_table_lowering import (
    COST_SCALE,
    UNIT,
    CombineIndex,
    CompactIndex,
    AbstractRealization,
    Index,
    InputIndex,
    SumRealization,
    SwitchRealization,
    TableRealization,
    TableLowering,
    TableLoweringError,
    UnitIndex,
)
from torch_spyre._inductor.cost_model import ArgTraffic, OpFeatures, predict_ops
from torch_spyre._inductor.scratchpad import cpsat_table_lowering as lowering
from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS
from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

CAP = 8192


class Case:
    """A CP-SAT model with inputs made by hand: each symbol's input is its variable
    and its value at each of the variable's values."""

    def __init__(self) -> None:
        self.model = cp_model.CpModel()
        self.inputs: dict[sympy.Symbol, tuple[cp_model.IntVar, Sequence[float]]] = {}
        self.variables: list[cp_model.IntVar] = []
        self.values: dict[cp_model.IntVar, list[int]] = {}

    def add_input(self, name: str, **columns: list[float]) -> Index:
        """A buffer's division: a variable in 0 .. n-1 picking a row of its menu, and
        one column per symbol of the buffer. Returns the position the printer makes
        on that variable."""
        sizes = {len(column) for column in columns.values()}
        assert len(sizes) == 1
        (size,) = sizes
        var = self.model.new_int_var(0, size - 1, f"division_{name}")
        for symbol_name, column in columns.items():
            self.inputs[sympy.Symbol(symbol_name, positive=True)] = (var, column)
        self.variables.append(var)
        self.values[var] = list(range(size))
        return InputIndex(var, size=size)

    def integer(self, name: str, lo: int, size: int) -> Index:
        """An integer variable with its own values, lo .. lo + size - 1: its position
        is the value offset by lo. Returns the position the printer makes on it."""
        domain = list(range(lo, lo + size))
        var = self.model.new_int_var(lo, lo + size - 1, name)
        self.inputs[sympy.Symbol(name, integer=True)] = (var, domain)
        self.variables.append(var)
        self.values[var] = list(domain)
        return InputIndex(var, size=size, lo=lo)

    def symbol(self, name: str) -> sympy.Symbol:
        (symbol,) = (s for s in self.inputs if s.name == name)
        return symbol

    def lower(
        self, expr: sympy.Basic, cap: int = CAP
    ) -> tuple[AbstractRealization, cp_model.LinearExprT]:
        root = TableLowering(self.inputs, cap=cap).print(expr)
        return root, root.cp_expr(self.model)

    def solutions(self, exprs: list) -> list[tuple[int, ...]]:
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        found: list[tuple[int, ...]] = []

        class Collect(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self) -> None:
                found.append(tuple(self.value(e) for e in exprs))

        status = solver.solve(self.model, Collect())
        assert status == cp_model.OPTIMAL, solver.status_name(status)
        return found

    def sympy_value(self, expr: sympy.Basic, assignment: tuple[int, ...]) -> float:
        """expr evaluated by sympy where the variables take assignment."""
        values = dict(zip(self.variables, assignment))
        point = {}
        for symbol, (var, column) in self.inputs.items():
            value = float(column[values[var] - self.values[var][0]])
            point[symbol] = (
                sympy.Integer(int(value)) if value.is_integer() else sympy.Float(value)
            )
        return float(expr.xreplace(point))

    def assert_exact(self, expr: sympy.Basic, cap: int = CAP) -> AbstractRealization:
        plan, objective = self.lower(expr, cap)
        cost = self.model.new_int_var(-(2**60), 2**60, "cost")
        self.model.add(cost == objective)
        rows = self.solutions(self.variables + [cost])
        assignments = list(itertools.product(*(self.values[v] for v in self.variables)))
        assert len(rows) == len(assignments)
        assert sorted(row[:-1] for row in rows) == sorted(assignments)
        bound = _rounding_bound(plan)
        for row in rows:
            exact = COST_SCALE * self.sympy_value(expr, row[:-1])
            assert abs(row[-1] - exact) <= bound, (row, exact)
        return plan


class Recording(TableLowering):
    """A printer that keeps what it prints: each node's realization by its sympy node,
    in the order the realizations are returned, operands before their parents."""

    def __init__(
        self,
        inputs: dict[sympy.Symbol, tuple[cp_model.IntVar, Sequence[float]]],
        cap: int = CAP,
    ) -> None:
        super().__init__(inputs, cap=cap)
        self.printed: dict[sympy.Basic, AbstractRealization] = {}

    def _print(self, expr: sympy.Basic, **kwargs: Any) -> AbstractRealization:
        realization = super()._print(expr, **kwargs)
        self.printed.setdefault(expr, realization)
        return realization


def _printed(
    case: Case, expr: sympy.Basic, cap: int = CAP
) -> dict[sympy.Basic, TableRealization]:
    """Each node under expr as one printer lowers it: its table by its sympy node,
    operands before their parents."""
    printer = Recording(case.inputs, cap)
    printer.print(expr)
    return {
        node: realization.table(cap) for node, realization in printer.printed.items()
    }


def _rounding_bound(plan: AbstractRealization) -> float:
    """Half a unit per rounding: each list's and each sum's constant's. A switch
    scales its rest's bound by each value, and a fraction p / q adds under one unit
    for the floor plus |w| / (2q)."""
    if isinstance(plan, SumRealization):
        return 0.5 + sum(_rounding_bound(term) for term in plan.terms)
    if isinstance(plan, SwitchRealization):
        rest = _rounding_bound(plan.rest)
        fraction = 1 + max(map(abs, plan.rest.bounds())) / (2 * plan.denominator())
        return max(
            abs(v) * rest + (0 if v.is_integer() else fraction) for v in plan.values
        )
    return 0.5


def _tables(model: cp_model.CpModel) -> list[int]:
    """The entries of each table in model: one add_element per table."""
    return [len(c.element.exprs) for c in model.proto.constraints if c.has_element()]


def _operations(printed: Iterable[sympy.Basic]) -> list[sympy.Basic]:
    """The nodes whose lists the printer computes: those with operands, an operation on
    numbers among them, and neither a symbol nor a number."""
    return [e for e in printed if e.args]


# -------------------------------------------------------------------- the printer


def test_the_printer_prints_operands_first_and_a_shared_subexpression_once(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    shared = sympy.Min(a, b)
    expr = sympy.exp(shared) + sympy.log(shared)
    before = sympy.srepr(expr)
    calls = _count_evaluations(monkeypatch)
    printer = Recording(case.inputs)
    root = printer.print(expr)
    order = list(printer.printed)
    for node in order:
        for operand in node.args:
            assert order.index(operand) < order.index(node)
    assert order[-1] == expr and printer.printed[expr] is root
    assert calls.count(shared) == 1  # two parents read one realization
    assert printer.print(shared) is printer.printed[shared]
    assert sympy.srepr(expr) == before


def test_a_piecewise_arm_is_not_printed():
    """A Piecewise's operands are its arms' values and conditions."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    expr = sympy.Piecewise((a, a > 2), (a**2, True))
    printed = _printed(case, expr)
    assert not any(isinstance(e, ExprCondPair) for e in printed)
    assert {a, a > 2, a**2, sympy.true} <= set(printed)
    assert list(printed[expr].values) == [1.0, 4.0, 4.0]


def test_a_condition_is_a_bool_table():
    """A condition's table is bool, true and false included, and a parent takes it as
    it is; any other node's table is float."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    condition = sympy.And(a > 1, sympy.Not(a > 3))
    expr = sympy.Piecewise((a, condition), (-a, True))
    printed = _printed(case, expr)
    for node in (*condition.args, condition, sympy.true):
        assert printed[node].values.dtype == bool, node
    assert list(printed[condition].values) == [False, True, False]
    assert printed[a].values.dtype == float
    assert printed[expr].values.dtype == float
    assert list(printed[expr].values) == [-1.0, 2.0, -4.0]


def test_a_symbol_a_constant_and_an_operation_are_each_one_table():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    root2 = sympy.sqrt(2)
    expr = sympy.Max(a, 3) ** root2
    printer = Recording(case.inputs)
    printer.print(expr)
    for realization in printer.printed.values():
        assert type(realization) is TableRealization
        assert realization.table(CAP) is realization
    printed = _printed(case, expr)
    assert printed[a].index == position and list(printed[a].values) == [1.0, 2.0, 4.0]
    assert printed[sympy.Integer(3)].index == UNIT
    assert list(printed[sympy.Integer(3)].values) == [3.0]
    assert printed[root2].index == UNIT  # an operation on numbers
    assert list(printed[root2].values) == [2**0.5]
    assert printed[sympy.Max(a, 3)].index == position
    assert printed[expr].index == position


def test_the_lowering_is_a_sympy_printer_dispatching_on_the_sympy_class():
    """Dispatch follows the MRO, so a Dummy prints as a Symbol; a declined node is
    one table."""
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    dummy = sympy.Dummy("a")
    case.inputs[dummy] = case.inputs[a]
    printer = TableLowering(case.inputs, cap=CAP)
    assert isinstance(printer, Printer)
    column = printer.print(dummy).table(CAP)
    assert column.index == position and list(column.values) == [1.0, 2.0, 4.0]
    logs = printer.print(sympy.log(a)).table(CAP)
    assert logs.index == position
    assert list(logs.values) == pytest.approx([0.0, math.log(2), math.log(4)])
    number = sympy.log(sympy.Integer(2)) + sympy.pi
    constant = printer.print(number).table(CAP)
    assert constant.index == UNIT
    assert printer.emptyPrinter(sympy.log(a)) is None
    assert printer._print_Add(number) is None
    assert printer._print_Mul(a * sympy.log(a)) is None
    root = printer.print(sympy.log(a) + dummy)
    assert root is printer.print(sympy.log(a) + dummy)
    assert not hasattr(printer, "lower")  # print is the entry point
    assert printer.print(sympy.log(a)) is logs
    with pytest.raises(NotImplementedError, match="call print"):
        printer.doprint(sympy.log(a))


def test_the_printer_makes_a_symbols_position_on_its_variable():
    """The caller hands over a variable and values, and no position. Two symbols of
    one variable get equal positions, so one table over both is on one axis."""
    case = Case()
    position = case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    offset = case.integer("y", 2, 4)
    d0, d1, y = case.symbol("d0"), case.symbol("d1"), case.symbol("y")
    printed = _printed(case, d0 * d1 + y)
    assert printed[d0].index == printed[d1].index == position
    assert printed[y].index == offset and offset.lo == 2
    assert list(printed[d1].values) == [8.0, 4.0, 2.0]
    assert printed[d0 * d1].index == position  # one axis, 3 entries
    case.assert_exact(d0 * d1 + y)
    division = case.variables[0]
    gapped = case.model.new_int_var_from_domain(
        cp_model.Domain.from_values([1, 2, 4]), "gapped"
    )
    literal = case.model.new_bool_var("literal")
    for refused, match in (
        ((division, [1.0, 2.0]), "2 values for a variable with 3, 0 to 2"),
        ((division, [1.0, "two", 4.0]), "values that are not one number each"),
        ((gapped, [1.0, 2.0, 3.0, 4.0]), "a variable whose domain has gaps"),
        ((~literal, [1.0, 2.0]), "a variable that is not an IntVar"),
    ):
        printer = TableLowering({d0: refused}, cap=CAP)  # type: ignore[dict-item]
        with pytest.raises(TableLoweringError, match=match) as caught:
            printer.print(d0)
        assert caught.value.node == d0


def test_an_operations_position_combines_its_inputs_in_symbol_order():
    """The printer builds an operation's position from its inputs' positions, in the
    order of its symbols, however the expression is written."""
    case = Case()
    pa = case.add_input("a", a=[1.0, 2.0, 4.0])
    pb = case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = a**0.5 + 2 * b
    printed = _printed(case, expr)
    assert printed[expr].index == CombineIndex(pa, pb)
    assert printed[2 * b].index == pb
    assert printed[sympy.Integer(2)].index == UNIT
    assert _printed(case, sympy.log(b) * a)[sympy.log(b) * a].index == CombineIndex(
        pa, pb
    )


def test_a_table_at_the_cap_is_made_and_one_past_it_is_refused():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    root = TableLowering(case.inputs, cap=3).print(a**0.5).table(CAP)  # at the cap
    assert root.index == position
    assert list(root.values) == pytest.approx([1.0, 2**0.5, 2.0])
    with pytest.raises(
        TableLoweringError,
        match="one table over its inputs would hold 3 entries, past the cap of 2",
    ):
        TableLowering(case.inputs, cap=2).print(a**0.5)
    with pytest.raises(TableLoweringError, match="no input"):
        TableLowering({}, cap=CAP).print(a**0.5)


def _count_evaluations(monkeypatch) -> list[sympy.Basic]:
    """The nodes whose lists are computed from here on, in order: every list the
    printer computes goes through the lowering's _values, once."""
    calls: list[sympy.Basic] = []
    values = lowering._values

    def counting(expr, count, compute):
        calls.append(expr)
        return values(expr, count, compute)

    monkeypatch.setattr(lowering, "_values", counting)
    return calls


def test_a_table_past_the_cap_is_never_evaluated(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls = _count_evaluations(monkeypatch)
    with pytest.raises(
        TableLoweringError, match="6 entries, past the cap of 5"
    ) as caught:
        TableLowering(case.inputs, cap=5).print(sympy.log(a**0.5 * b))
    assert caught.value.node == sympy.log(a**0.5 * b)  # the root, refused first
    assert calls == []  # not the root's table, and nothing under it


def test_a_table_builds_its_own_lookup_and_a_constant_one_adds_nothing():
    case = Case()
    position = case.integer("y", 1, 4)
    before = len(case.model.proto.variables), len(case.model.proto.constraints)
    constant = TableRealization(values=[2.0, 2.0, 2.0, 2.0], index=position)
    assert constant.cp_expr(case.model) == 2 * COST_SCALE
    after = len(case.model.proto.variables), len(case.model.proto.constraints)
    assert after == before
    table = TableRealization(values=[1.0, 2.0, 4.0, 8.0], index=position)
    target = table.cp_expr(case.model)
    rows = dict(case.solutions([case.variables[0], target]))
    assert rows == {1: 1024, 2: 2048, 3: 4096, 4: 8192}


def test_print_touches_no_model_and_a_refusal_comes_from_print():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    size = len(case.model.proto.constraints), len(case.model.proto.variables)
    root = TableLowering(case.inputs, cap=CAP).print(sympy.Min(a, b))
    assert isinstance(root, TableRealization) and root.index.size == 6
    assert (len(case.model.proto.constraints), len(case.model.proto.variables)) == size
    root.cp_expr(case.model)  # only the build adds to the model
    size = len(case.model.proto.constraints), len(case.model.proto.variables)
    with pytest.raises(TableLoweringError, match="past the cap of 4"):
        TableLowering(case.inputs, cap=4).print(sympy.Min(a, b))
    assert (len(case.model.proto.constraints), len(case.model.proto.variables)) == size


def test_an_expression_nested_past_the_recursion_limit_is_refused():
    """The printer recurses a few frames per level of the expression. One nested
    deeper than the interpreter's recursion limit allows is refused as a
    TableLoweringError naming no node, and no RecursionError, a RuntimeError, leaves
    print, so an except around the objective cannot take it for another failure."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    expr = case.symbol("a")
    for _ in range(sys.getrecursionlimit()):  # unevaluated: sympy's own checks recurse
        expr = sympy.Pow(
            sympy.Add(expr, 1, evaluate=False), sympy.Rational(1, 2), evaluate=False
        )
    with pytest.raises(TableLoweringError, match="recursion limit") as caught:
        TableLowering(case.inputs, cap=CAP).print(expr)
    assert caught.value.node is None
    assert isinstance(caught.value.__cause__, RecursionError)


# ------------------------------------------------------------------------- values


def test_a_realization_is_abstract_and_a_table_is_one():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(TypeError):
        AbstractRealization()  # type: ignore[abstract]

    class Built(AbstractRealization):
        """A realization that builds itself but cannot be asked for its table."""

        def build(self, model: cp_model.CpModel) -> cp_model.LinearExprT:
            return 0

    with pytest.raises(TypeError):
        Built()  # type: ignore[abstract]
    table = TableRealization(values=[1.0, 2.0, 4.0], index=position)
    assert isinstance(table, AbstractRealization)
    assert table.table(CAP) is table
    a = case.symbol("a")
    lowered = TableLowering(case.inputs, cap=CAP).print(a**0.5)
    assert isinstance(lowered, AbstractRealization)


def test_a_realization_with_a_value_that_is_not_finite_is_not_made():
    """Refused naming no node: the printer names the node it was printing."""
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    for make in (
        lambda: TableRealization(values=[1.0, np.inf, 4.0], index=position),
        lambda: TableRealization(values=[1.0, np.nan, 4.0], index=position),
    ):
        with pytest.raises(TableLoweringError, match="not finite") as caught:
            make()
        assert caught.value.node is None


def test_a_parent_asks_an_operand_for_its_table_within_the_printers_cap(monkeypatch):
    """table takes the cap: the bound on a table that has to be made. The printer
    passes its own to every realization it asks."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    asked: list[int] = []
    original = TableRealization.table

    def table(self: TableRealization, cap: int) -> TableRealization:
        asked.append(cap)
        return original(self, cap)

    monkeypatch.setattr(TableRealization, "table", table)
    TableLowering(case.inputs, cap=7).print(a**0.5 * b + sympy.log(a))
    assert asked and set(asked) == {7}


def test_a_list_must_have_its_index_size():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(ValueError):
        TableRealization(values=[1.0, 2.0], index=position)


def test_a_step_takes_the_parts_it_names_and_equal_steps_are_one_position():
    case = Case()
    a = case.add_input("a", a=[1.0, 2.0, 4.0])
    b = case.add_input("b", b=[1.0, 3.0])
    with pytest.raises(TypeError):
        Index()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        CombineIndex(a)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CombineIndex(a, b, a)  # type: ignore[call-arg]
    pair = CombineIndex(a, b)
    assert pair.size == 6 and (pair.left, pair.right) == (a, b)
    assert pair == CombineIndex(a, b) and hash(pair) == hash(CombineIndex(a, b))
    assert pair != CombineIndex(b, a)
    assert UNIT.size == 1 and UNIT == UnitIndex()


def test_a_table_takes_values_and_index_only_and_its_list_is_read_only():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(TypeError):
        TableRealization(  # type: ignore[call-arg]
            values=[1.0, 2.0, 3.0], index=position, constant=1.0
        )
    source = np.array([1.0, 2.0, 3.0])
    table = TableRealization(values=source, index=position)
    source[0] = 9.0
    assert table.values[0] == 1.0
    with pytest.raises(ValueError):
        table.values[0] = 5.0
    with pytest.raises(AttributeError):
        table.values = np.zeros(3)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        table.index = UNIT  # type: ignore[misc]


def test_a_combined_position_is_row_major_at_every_assignment():
    case = Case()
    x = case.integer("x", 2, 4)
    y = case.integer("y", -1, 3)
    joint = CombineIndex(x, y)
    rows = case.solutions(case.variables + [joint.cp_expr(case.model)])
    assert all(position == (vx - 2) * 3 + (vy + 1) for vx, vy, position in rows)
    assert sorted(position for _, _, position in rows) == list(range(12))


# ------------------------------------------------------------------- a node's list


def _points(
    case: Case, expr: sympy.Basic
) -> tuple[list[sympy.Symbol], list[tuple[float, ...]]]:
    """expr's free symbols, sorted by sympy.default_sort_key, and their values at each
    entry of its table: their variables in that order, row-major, the last fastest."""
    symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
    columns = {s: np.asarray(case.inputs[s][1], dtype=float) for s in symbols}
    axis: dict[int, int] = {}  # each variable's axis, by the variable's identity
    sizes: list[int] = []
    for s in symbols:
        if id(case.inputs[s][0]) not in axis:
            axis[id(case.inputs[s][0])] = len(sizes)
            sizes.append(len(columns[s]))
    grid = np.indices(tuple(sizes)).reshape(len(sizes), -1)
    arguments = [columns[s][grid[axis[id(case.inputs[s][0])]]] for s in symbols]
    return symbols, list(zip(*arguments))


def _stage_1(case: Case, expr: sympy.Basic) -> np.ndarray:
    """expr's list as Stage 1 computed it: its whole expression lambdified with the
    numpy module over its inputs, point by point when it takes scalars only."""
    symbols, points = _points(case, expr)
    func = sympy.lambdify(symbols, expr, modules="numpy")
    with np.errstate(all="ignore"):
        try:
            result = np.asarray(func(*np.array(points).T), dtype=float)
        except (TypeError, ValueError):
            result = np.array([func(*point) for point in points], dtype=float)
    return np.broadcast_to(result, (len(points),))


def _list(case: Case, expr: sympy.Basic) -> list[float]:
    return list(TableLowering(case.inputs, cap=CAP).print(expr).table(CAP).values)


def test_a_list_is_row_major_with_the_last_position_fastest():
    """Each term is on one position, and the sum reads each at its own."""
    case = Case()
    case.add_input("a", a=[0.0, 1.0])
    case.add_input("b", b=[0.0, 1.0, 2.0])
    case.add_input("c", c=[0.0, 1.0, 2.0, 3.0])
    sa, sb, sc = case.symbol("a"), case.symbol("b"), case.symbol("c")
    expected = [
        100 * i + 10 * j + k for i in range(2) for j in range(3) for k in range(4)
    ]
    assert _list(case, 100 * sa + 10 * sb + sc) == expected


def test_two_symbols_of_one_position_move_together():
    case = Case()
    case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    d0, d1 = case.symbol("d0"), case.symbol("d1")
    assert _list(case, d0 * d1) == [8.0, 8.0, 8.0]


def test_a_constant_operand_is_read_at_every_entry():
    """Max is lambdified as numpy.amax over its operands, which takes them only of one
    length."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    assert _list(case, sympy.Max(a, 2)) == [2.0, 2.0, 4.0]


def test_a_value_that_is_not_finite_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    with pytest.raises(
        TableLoweringError, match="a value that is not finite"
    ) as caught:
        case.lower(1 / (a - 2))
    assert caught.value.node == 1 / (a - 2)


def test_a_value_that_is_not_finite_is_refused_at_its_node_under_a_finite_root():
    """Each root's own table is finite: a Min drops an infinite value, a comparison
    reads a NaN as false, and an arm is never taken. The child that makes the value
    refuses it, be it an operation, a constant or a symbol."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0], b=[1.0, np.inf, 4.0])
    a, b = case.symbol("a"), case.symbol("b")
    infinite = 1 / (a - 2)
    nan = sympy.sqrt(a - 3)
    for child, expr in (
        (infinite, sympy.Min(infinite, 5)),
        (nan, sympy.Piecewise((1, nan > 0), (2, True))),
        (sympy.oo, sympy.Piecewise((sympy.oo, a > 4), (a, True))),
        (b, sympy.Min(a, b)),
    ):
        assert np.all(np.isfinite(_stage_1(case, expr)))
        with pytest.raises(TableLoweringError, match="not finite") as caught:
            case.lower(expr)
        assert caught.value.node == child


def test_a_node_is_evaluated_once_its_operands_before_it(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls = _count_evaluations(monkeypatch)
    shared = sympy.Min(a, b)
    expr = sympy.Max(sympy.exp(shared), sympy.log(shared))
    printer = Recording(case.inputs)
    root = printer.print(expr)
    computed = _operations(printer.printed)
    # lowering the root evaluates every node under it, each once, the root last
    assert len(calls) == len(computed) and set(calls) == set(computed)
    assert calls[-1] == expr
    printer.print(expr)
    root.cp_expr(case.model)
    assert len(calls) == len(computed)  # kept: neither a second print nor the build


# ---------------------------------------------------------------- exact lowering


def test_one_table_over_two_inputs_of_different_sizes_is_exact():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 5.0, 7.0])
    a, b = case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(a**2 * b + b**0.5 / a)
    assert plan.table(CAP).index.size == 12


def test_variables_above_and_below_zero_read_their_entries():
    case = Case()
    case.integer("x", 2, 4)
    case.integer("y", -1, 3)
    x, y = case.symbol("x"), case.symbol("y")
    case.assert_exact(x**1.5 * y + 3 * x)


def test_an_infinite_piecewise_arm_is_refused_at_the_arm_taken_or_not():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    arm = 1 / (a - 2)  # infinite at a = 2
    for condition in (a > 2, a >= 2):  # at a = 2 the arm is not taken, then taken
        with pytest.raises(TableLoweringError, match="not finite") as caught:
            case.lower(sympy.Piecewise((arm, condition), (0.5, True)))
        assert caught.value.node == arm


def test_a_piecewise_with_no_arm_true_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 4.0])
    a = case.symbol("a")
    expr = sympy.Piecewise((a, a > 3))
    with pytest.raises(
        TableLoweringError, match="a value that is not finite"
    ) as caught:
        case.lower(expr)
    assert caught.value.node == expr


def test_a_relayout_charge_is_one_table_over_its_literal_and_division():
    case = Case()
    model = case.model
    lit = model.new_bool_var("is_lx")
    division = model.new_int_var(0, 2, "division_src")
    is_lx = sympy.Symbol("is_lx", integer=True)
    div = sympy.Symbol("division_src", integer=True, nonnegative=True)
    case.inputs[is_lx] = (lit, [0, 1])
    case.inputs[div] = (division, [0, 1, 2])
    case.variables += [lit, division]
    case.values[lit], case.values[division] = [0, 1], [0, 1, 2]
    expr = RelayoutCharge(is_lx, div, 0, 65, 147)
    plan = case.assert_exact(expr)
    assert plan.table(CAP).index.size == 6


@pytest.mark.parametrize(
    "make", [sympy.sec, sympy.sinc, sympy.Function("f")], ids=["sec", "sinc", "f"]
)
def test_a_class_the_numpy_core_has_no_function_for_is_refused(make):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    expr = make(case.symbol("a"))
    with pytest.raises(TableLoweringError, match="no numpy function for") as caught:
        case.lower(expr + 1)
    assert caught.value.node == expr


def test_a_coefficient_is_applied_before_the_rounding():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    _, objective = case.lower(7 * a**0.5)
    cost = case.model.new_int_var(0, 2**40, "cost")
    case.model.add(cost == objective)
    rows = dict((r[0], r[1]) for r in case.solutions([case.variables[0], cost]))
    assert rows[1] == 10137  # rint(1024 * 7 * 2**0.5), not 7 * rint(1024 * 2**0.5)


def test_a_constant_list_adds_no_variable_and_a_constant_objective_is_an_int():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    variables = len(case.model.proto.variables)
    _, objective = case.lower(a / a + 2)
    assert objective == 3 * COST_SCALE
    assert isinstance(objective, int)
    assert len(case.model.proto.variables) == variables


def test_an_entry_too_large_for_int64_sums_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    with pytest.raises(TableLoweringError, match="2\\^50") as caught:
        case.lower(1e14 * a)
    assert caught.value.node is None  # refused by the table as it is built


def test_a_symbol_with_no_input_is_refused_by_name():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a, z = case.symbol("a"), sympy.Symbol("z", positive=True)
    with pytest.raises(TableLoweringError) as caught:
        case.lower(a + z)
    assert caught.value.node == z
    assert "no input" in caught.value.reason


# -------------------------------------------------------------------- linear sums


def test_a_sum_is_exact_with_no_table_across_its_terms():
    """One table would hold 21 * 11 * 20 = 4620 entries, past the cap of 4096."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 22)])
    case.add_input("b", b=[float(v) for v in range(1, 12)])
    case.add_input("c", c=[float(v) for v in range(1, 21)])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    expr = a**0.5 + b**0.5 + c**0.25
    plan = case.assert_exact(expr, cap=4096)
    assert sorted(_tables(case.model)) == [11, 20, 21]
    assert plan.table(CAP).index.size == 4620
    with pytest.raises(
        TableLoweringError, match="4620 entries, past the cap of 4096"
    ) as caught:
        plan.table(4096)
    assert caught.value.node is None  # refused by the sum, which holds no node


def test_a_node_with_one_list_is_a_table():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    printer = TableLowering(case.inputs, cap=CAP)
    number = sympy.log(sympy.Integer(2)) + sympy.pi
    for expr in [a, sympy.log(a * b), 7 * a**0.5, number]:
        assert isinstance(printer.print(expr), TableRealization)


def test_a_sum_is_a_sum_realization_of_its_operands_realizations():
    case = Case()
    pa = case.add_input("a", a=[1.0, 2.0, 4.0])
    pb = case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    printer = TableLowering(case.inputs, cap=CAP)
    root = printer.print(sympy.log(b) + a**0.5 + 2.5)
    assert type(root) is SumRealization
    (constant,) = [term for term in root.terms if term.index == UNIT]
    assert list(constant.values) == [2.5]
    operands = {printer.print(a**0.5), printer.print(sympy.log(b))}
    assert set(root.terms) == {constant, *operands}
    assert root.positions == (pa, pb)
    assert list(root.table(CAP).values) == pytest.approx(
        [2.5 + va**0.5 + math.log(vb) for va in (1.0, 2.0, 4.0) for vb in (1.0, 3.0)]
    )
    number = printer.print(sympy.log(sympy.Integer(2)) + sympy.pi)
    assert type(number) is TableRealization and number.index == UNIT


def test_a_constant_times_one_operand_is_the_operand_with_the_constant_applied():
    """A quotient is divided, not scaled: 98 * (1 / 49) is under 2."""
    case = Case()
    pa = case.add_input("a", a=[1.0, 7.0, 49.0])
    pb = case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    printer = TableLowering(case.inputs, cap=CAP)
    roots = printer.print(a**0.5).table(CAP)
    scaled = printer.print(7 * a**0.5)
    assert type(scaled) is TableRealization and scaled.index == pa
    assert list(scaled.values) == [7 * value for value in roots.values]
    total = printer.print(-2 * (a**0.5 + b + 0.25))
    assert type(total) is SumRealization
    assert {term.index: list(term.values) for term in total.terms} == {
        pa: [-2 * value for value in roots.values],
        pb: [-2.0, -6.0],
        UNIT: [-0.5],
    }
    quotient = printer.print(98 / a)
    assert type(quotient) is TableRealization
    assert list(quotient.values) == [98.0, 14.0, 2.0] and 98 * (1 / 49) < 2


def test_a_coefficient_on_a_sum_is_applied_before_the_rounding():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = 7 * (a**0.5 + b**0.5 + sympy.Rational(1, 3))
    plan = case.assert_exact(expr)
    assert type(plan) is SumRealization and len(plan.terms) == 3
    _, objective = case.lower(expr)
    cost = case.model.new_int_var(0, 2**40, "at")
    case.model.add(cost == objective)
    rows = {row[:2]: row[2] for row in case.solutions(case.variables + [cost])}
    once = [round(1024 * 7 * value) for value in (2**0.5, 1.0, 1 / 3)]
    apart = [7 * round(1024 * value) for value in (2**0.5, 1.0, 1 / 3)]
    assert rows[1, 0] == sum(once) == 19694 and sum(apart) == 19691


def test_lists_on_one_position_are_added_before_the_rounding():
    """sympy distributes the 2 in 2 * (split**0.7 + b**0.5), so its lists join the
    others on their positions."""
    case = Case()
    case.add_input("a", split=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 9.0])
    split, b = case.symbol("split"), case.symbol("b")
    plan, objective = case.lower(split**0.5 + split**0.7)
    assert isinstance(plan, SumRealization) and len(plan.terms) == 1
    assert _tables(case.model) == [3]
    cost = case.model.new_int_var(0, 2**40, "cost")
    case.model.add(cost == objective)
    rows = dict(case.solutions([case.variables[0], cost]))
    assert rows[1] == 3112
    assert round(1024 * 2**0.5) + round(1024 * 2**0.7) == 3111
    distributed = case.assert_exact(split**0.5 + 2 * (split**0.7 + b**0.5) + b**0.25)
    assert isinstance(distributed, SumRealization)
    assert sorted(term.table(CAP).index.size for term in distributed.terms) == [3, 3]


def test_a_sum_under_a_constant_that_is_not_a_number_is_one_term():
    """sympy does not distribute sqrt(2), so the scaled inner sum is one term."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = a + sympy.sqrt(2) * (a + b)
    assert sympy.sqrt(2) * (a + b) in expr.args  # sympy has not distributed sqrt(2)
    plan = case.assert_exact(expr)
    assert type(plan) is SumRealization and len(plan.terms) == 2
    inner = plan.terms[1]
    assert type(inner) is SumRealization and len(inner.terms) == 2
    # a's list and sqrt(2) times it, each a lookup; sqrt(2) times b's is affine
    assert _tables(case.model) == [3, 3]
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    case.assert_exact(sympy.log(a + sympy.sqrt(2) * (a + b)))


def test_an_affine_list_adds_no_table():
    """3 * b, and any list of two entries, as a 0/1 literal's is."""
    case = Case()
    case.add_input("b", b=[float(v) for v in range(1, 7)])
    case.integer("x", 0, 2)
    b, x = case.symbol("b"), case.symbol("x")
    constraints = len(case.model.proto.constraints)
    variables = len(case.model.proto.variables)
    _, objective = case.lower(3 * b - 2.25 * x + sympy.exp(x))
    assert len(case.model.proto.constraints) == constraints
    assert len(case.model.proto.variables) == variables
    assert not isinstance(objective, int)
    case.assert_exact(3 * b - 2.25 * x + sympy.exp(x))
    assert _tables(case.model) == []
    case.assert_exact(b**0.5 + 3 * b)
    assert _tables(case.model) == [6]


def test_a_sum_under_a_power_is_read_as_its_table_and_refused_past_the_cap():
    """Past the cap the parent is refused before the sum is printed."""
    case = Case()
    pa = case.add_input("a", a=[1.0, 2.0, 4.0])
    pb = case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = (a + 2 * b) ** 0.5
    plan = case.assert_exact(expr)
    assert type(plan) is TableRealization and plan.index == CombineIndex(pa, pb)
    assert _tables(case.model) == [6]
    printer = Recording(case.inputs, cap=5)
    with pytest.raises(
        TableLoweringError, match="6 entries, past the cap of 5"
    ) as caught:
        printer.print(expr)
    assert caught.value.node == expr
    assert a + 2 * b not in printer.printed


def test_a_sum_that_is_not_finite_is_refused_at_the_sum():
    """Finite lists can add up to inf: refused at the node that adds them, though the
    root's own list is finite."""
    case = Case()
    case.add_input("a", d0=[1.0, 1.7e308], d1=[1.0, 1.7e308])
    case.add_input("b", b=[1.0, 1.7e308])
    d0, d1, b = case.symbol("d0"), case.symbol("d1"), case.symbol("b")
    for node, expr in (
        (d0 + d1 + b, d0 + d1 + b),
        (d0 + b, 1 / (d0 + b)),
        (10 * d0, sympy.Min(10 * d0, 5)),
    ):
        with pytest.raises(TableLoweringError, match="not finite") as caught:
            case.lower(expr)
        assert caught.value.node == node


# --------------------------------------------------------------------- the switch


def _named(model: cp_model.CpModel, name: str) -> list[int]:
    """The proto indices of model's variables named name."""
    return [i for i, v in enumerate(model.proto.variables) if v.name == name]


def _abc(case: Case) -> list[sympy.Symbol]:
    """Three buffers whose lists below are not affine, so each is one table."""
    case.add_input("a", a=[1.0, 2.0, 4.0, 9.0])
    case.add_input("b", b=[1.0, 3.0, 4.0])
    case.add_input("c", c=[1.0, 2.0, 3.0, 5.0, 8.0])
    return [case.symbol(name) for name in "abc"]


FACTORS = {
    "whole_fraction_zero": [2.0, 0.375, 0.0],
    "whole": [1.0, 2.0, 3.0],
    "mixed_sign": [-1.5, 0.0, 2.0],
    "negative": [-3.0, -0.25],
}


@pytest.mark.parametrize("factor", FACTORS.values(), ids=FACTORS)
def test_a_factor_times_a_sum_over_two_buffers_is_a_switch_and_exact(factor):
    """No table crosses factor and rest: one per list. Each case's true/false is on
    exactly where the factor takes its value, and sets g with one constraint for a
    whole value and two for a fraction."""
    case = Case()
    position = case.add_input("k", k=factor)
    k, (a, b, _) = case.symbol("k"), _abc(case)
    plan = case.assert_exact(k * (a**0.5 + sympy.log(b)))
    assert type(plan) is SwitchRealization and type(plan.rest) is SumRealization
    assert plan.values == tuple(factor) and plan.index == position  # no rank lookup
    assert sorted(_tables(case.model)) == [3, 4]
    model = case.model
    (g,) = _named(model, "switch")
    on_g = [c for c in model.proto.constraints if c.has_linear() and g in c.linear.vars]
    assert len(on_g) == 1 + sum(1 if v.is_integer() else 2 for v in factor)  # + cost
    cases = [model.get_bool_var_from_proto_index(i) for i in _named(model, "case")]
    for row in case.solutions([case.variables[0], *cases]):
        assert row[1:] == tuple(int(i == row[0]) for i in range(len(factor)))


def test_products_with_one_factor_are_one_switch_over_their_rests_sum():
    """One w, one set of true/falses and one g; lists on one position in the rests
    are added."""
    case = Case()
    case.add_input("k", k=[1.0, 2.5, 4.0])
    k, (a, b, _) = case.symbol("k"), _abc(case)
    plan = case.assert_exact(k * a**0.5 + k * sympy.log(b) + k * a**0.7)
    assert isinstance(plan, SumRealization) and len(plan.terms) == 1
    switch = plan.terms[0]
    assert isinstance(switch, SwitchRealization)
    assert isinstance(switch.rest, SumRealization) and len(switch.rest.terms) == 2
    assert sorted(_tables(case.model)) == [3, 4]
    assert [len(_named(case.model, name)) for name in ("switch", "case")] == [1, 3]


def test_a_position_and_a_realization_are_built_once_per_model():
    """Two factors on one buffer with equal ranks share one CompactIndex, and their
    switches one rest: one rank lookup and one list. A position that reads the rank
    twice adds one lookup. A second build in the model adds nothing; a build in
    another model adds it all again."""
    case = Case()
    position = case.add_input("k", f=[1.0, 1.0, 2.0], h=[5.0, 5.0, 7.0])
    f, h, (a, _, _) = case.symbol("f"), case.symbol("h"), _abc(case)
    plan = case.assert_exact(f * a**0.5 + h * a**0.5)
    rank = CompactIndex(position, (0, 0, 1))
    assert isinstance(plan, SumRealization) and len(plan.terms) == 2
    assert all(isinstance(t, SwitchRealization) and t.index == rank for t in plan.terms)
    assert sorted(_tables(case.model)) == [3, 4]
    CombineIndex(rank, rank).cp_expr(case.model)
    assert sorted(_tables(case.model)) == [3, 3, 4]
    plan.cp_expr(case.model)
    assert sorted(_tables(case.model)) == [3, 3, 4]
    other = cp_model.CpModel()
    other.proto.copy_from(case.model.proto)
    plan.cp_expr(other)
    assert sorted(_tables(other)) == [3, 3, 3, 4, 4]


def test_a_product_of_three_operands_is_a_switch_inside_a_switch():
    case = Case()
    a, b, c = _abc(case)
    plan = case.assert_exact(a**0.5 * sympy.log(b) * c**1.5)
    assert type(plan) is SwitchRealization and type(plan.rest) is SwitchRealization
    assert _tables(case.model) == [5]  # c's list; the factors are cases


def test_a_factor_over_two_buffers_is_ranked_on_their_combined_position():
    case = Case()
    a, b, c = _abc(case)
    plan = case.assert_exact(sympy.Min(a, b) * c**1.5)
    assert type(plan) is SwitchRealization and plan.values == (1.0, 2.0, 3.0, 4.0)
    assert isinstance(plan.index, CompactIndex) and plan.index.parent.size == 12


def test_a_literal_times_a_sum_is_the_two_value_switch():
    """The rest's entries count once, where Stage 5's one table held them twice."""
    case = Case()
    case.integer("x", 0, 2)
    x, (a, b, c) = case.symbol("x"), _abc(case)
    plan = case.assert_exact(x * (a**0.5 + sympy.log(b) + c**1.5))
    assert type(plan) is SwitchRealization and plan.values == (0.0, 1.0)
    assert plan.entries() == plan.rest.entries() + 2 == 4 + 3 + 5 + 2


def test_a_product_takes_its_switch_or_its_table_by_entries():
    """A factor with a repeated value times a short list keeps its table, 6 entries
    against 7; the step-by-step graph's product is a switch, 5 against 6. A factor of
    one value scales the rest, with no lookup. Under a log or a power a product is
    read as its table, and no switch is built."""
    case = Case()
    case.add_input("a", a=[1.0, 1.0, 2.0])
    case.add_input("b", b=[3.0, 5.0])
    a, b = case.symbol("a"), case.symbol("b")
    assert type(TableLowering(case.inputs, cap=CAP).print(a * b)) is TableRealization
    case = Case()
    case.add_input("k", k=[0.5, 0.5, 0.5])
    k, (a, b, _) = case.symbol("k"), _abc(case)
    assert type(case.assert_exact(k * (a**0.5 + sympy.log(b)))) is SumRealization
    assert sorted(_tables(case.model)) == [3, 4]

    def step_graph() -> tuple[Case, sympy.Basic]:
        case = Case()
        case.add_input("a", split_a=[1.0, 2.0, 4.0])
        case.add_input("b", split_b=[1.0, 3.0])
        return case, 9 * case.symbol("split_a") ** -2 * case.symbol("split_b") ** 0.5

    case, product = step_graph()
    plan = case.assert_exact(product)
    assert type(plan) is SwitchRealization and plan.entries() == 5
    for read in (sympy.log, lambda p: 2**p):
        case, product = step_graph()
        assert type(case.assert_exact(read(product))) is TableRealization
        assert _tables(case.model) == [6] and not _named(case.model, "switch")


def test_a_factor_with_more_values_than_the_cap_has_no_switch():
    """With no table either, the product is refused, naming it."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0, 9.0, 16.0])
    case.add_input("b", b=[1.0, 3.0, 4.0, 5.0, 8.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = a**0.5 * sympy.log(b)
    assert type(TableLowering(case.inputs, cap=5).print(expr)) is SwitchRealization
    with pytest.raises(
        TableLoweringError, match="25 entries, past the cap of 4"
    ) as caught:
        TableLowering(case.inputs, cap=4).print(expr)
    assert caught.value.node == expr


def test_a_switched_product_past_2_60_is_refused_as_it_is_built():
    """A fraction's q halves down to 1 first."""
    case = Case()
    case.add_input("k", k=[0.5, 2.0**20])
    case.add_input("a", a=[1.0, 1e11, 2e11])
    with pytest.raises(TableLoweringError, match="2\\^60") as caught:
        case.lower(case.symbol("k") * case.symbol("a"))
    assert caught.value.node is None


def test_a_constant_is_read_as_the_double_it_is():
    """A list is the expression's value in floating point, rounded once where it
    enters CP-SAT, and is not held to another evaluator's bits. lambdify prints a
    Float to 15 digits, 2/3 as 0.666666666666667, above the double, so its
    ceiling(2/3 * c) is one high at c = 3, 6 and 9. The printer reads the constant as
    the double it is."""
    case = Case()
    case.add_input("c", c=[1.0, 3.0, 6.0, 7.0, 9.0])
    c = case.symbol("c")
    expr = sympy.ceiling(sympy.Float(2) / 3 * c)
    root = case.assert_exact(expr)
    assert list(root.table(CAP).values) == [1.0, 2.0, 4.0, 5.0, 6.0]


def test_an_operation_on_numbers_is_computed_as_any_other_operation():
    """Only a number itself is read as its double: an operation on numbers is its
    node's function on its operands' entries, as numpy computes it over a symbol, so
    the two sides of a tie are computed alike. sympy's float(3*sqrt(2)) is the double
    below numpy's 3 * sqrt(2), which makes this ceiling one high at a = 3, and its
    float(sqrt(19)) the double below numpy's sqrt(19), which makes sqrt(a) greater
    than sqrt(19) at a = 19."""
    root2, root19 = sympy.sqrt(2), sympy.sqrt(19)
    assert math.ceil(2**0.5 * 3 - 4.242640687119285) == 1  # the double at 3*sqrt(2)
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 3.0])
    a = case.symbol("a")
    root = case.assert_exact(sympy.ceiling(root2 * a - 3 * root2))
    assert list(root.table(CAP).values) == [-2.0, -1.0, 0.0]
    case = Case()
    case.add_input("a", a=[18.0, 19.0, 20.0])
    a = case.symbol("a")
    root = case.assert_exact(sympy.Piecewise((1, sympy.sqrt(a) > root19), (0, True)))
    assert list(root.table(CAP).values) == [0.0, 0.0, 1.0]


def test_a_quotient_is_divided_as_lambdify_prints_it():
    """x/y is Mul(x, Pow(y, -1)), and from the lists of its children it would be
    x*(1/y): 98*(1/49) is 1.9999999999999998, whose floor is 1. The Mul divides by
    y, as lambdify prints it, and so for any negative rational exponent: 98/7**2."""
    case = Case()
    case.add_input("a", x=[98.0, 98.0, 49.0], y=[49.0, 7.0, 7.0])
    x, y = case.symbol("x"), case.symbol("y")
    assert math.floor(98 * (1 / 49)) == 1 and math.floor(98 * 7.0**-2) == 1
    case.assert_exact(sympy.floor(x / y) + sympy.floor(x / y**2))


# Expressions over a, b and c, each node's list compared with Stage 1's.
STAGE_1_LISTS = [
    lambda a, b, c: a**2 * b + b**0.5 / a,
    lambda a, b, c: (a**0.5 + b) * a + sympy.log(a**0.5 + b),
    lambda a, b, c: sympy.Piecewise((a**2, a > 1.5), (b, True)) + sympy.Max(a, b),
    lambda a, b, c: sympy.Min(a + b, 3 * a) * sympy.floor(4 / a) + sympy.Abs(b - 2),
    lambda a, b, c: sympy.Max(a * b, c) + sympy.floor(98 / a) + a / c**2,
    lambda a, b, c: sympy.Piecewise(
        (1 / b, sympy.Or(a >= 2, c < 3)), (sympy.ceiling(c / 3), True)
    ),
]


@pytest.mark.parametrize("make", STAGE_1_LISTS)
def test_every_list_is_stage_1s(make):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0, 49.0])
    case.add_input("b", b=[1.0, 3.0])
    case.add_input("c", c=[1.0, 3.0, 7.0])
    expr = make(case.symbol("a"), case.symbol("b"), case.symbol("c"))
    printed = _printed(case, expr)
    printed[expr].cp_expr(case.model)
    for node in _operations(printed):
        np.testing.assert_allclose(
            printed[node].values, _stage_1(case, node), rtol=1e-12, err_msg=str(node)
        )


# A gate_up projection, as test_cost_model_replication.py builds it: symbolic M, N and
# K splits and a weight's residency, priced by predict_ops.
GU_K, GU_N = 4096, 25600
SPLIT_M, SPLIT_N, SPLIT_K = sympy.symbols(
    "split_m split_n split_k", integer=True, positive=True
)
WEIGHT_LX = sympy.Symbol("is_lx_arg1_1", integer=True, nonnegative=True)
MENU = [(4, 8, 1), (8, 4, 1), (2, 16, 1), (1, 16, 1), (8, 2, 2), (2, 4, 4), (1, 8, 4)]


def _gate_up(rows: int) -> OpFeatures:
    m, n, k = SPLIT_M, SPLIT_N, SPLIT_K
    return OpFeatures(
        name="mm",
        is_reduction=True,
        out_elems=rows * GU_N,
        cores=m * n * k,
        dtype_bytes=2,
        reduction_cores=k,
        args=[
            ArgTraffic("buf0", "input", False, rows * GU_K, replication=n),
            ArgTraffic(
                "arg1_1", "input", WEIGHT_LX, GU_K * GU_N, broadcast=True,
                is_boundary=True, replication=m,
            ),
            ArgTraffic("buf1", "output", False, rows * GU_N),
        ],
        is_matmul=True,
        matmul_macs=rows * GU_K * GU_N,
        matmul_a_bytes=2 * rows * GU_K,
        matmul_b_bytes=2 * GU_K * GU_N,
        matmul_rows_per_core=rows / m,
        matmul_cols_per_core=GU_N / n,
        matmul_m_split=m,
        matmul_n_split=n,
    )  # fmt: skip


@pytest.mark.parametrize("rows", [8, 64], ids=["decode", "past-decode"])
def test_the_cost_models_objective_has_the_annealers_values(rows):
    """The objective predict_ops builds for the projection, its splits on one
    candidate variable and the weight's residency on another: each entry of the root's
    table is sympy.lambdify with the math module, the annealer's evaluator, at its
    candidate and residency. At decode it holds the delivery-price functions, which
    take scalars only."""
    expr = sympy.sympify(predict_ops([_gate_up(rows)], _COST_PARAMS))
    case = Case()
    candidate = case.model.new_int_var(0, len(MENU) - 1, "division_mm")
    lit = case.model.new_bool_var("is_lx")
    for k, split in enumerate((SPLIT_M, SPLIT_N, SPLIT_K)):
        case.inputs[split] = (candidate, [float(s[k]) for s in MENU])
    case.inputs[WEIGHT_LX] = (lit, [0.0, 1.0])
    root = TableLowering(case.inputs, cap=CAP).print(expr)
    prices = [f for f in expr.atoms(sympy.Function) if "delivery_price" in str(f)]
    assert bool(prices) == (rows == 8)
    symbols, points = _points(case, expr)
    annealer = sympy.lambdify(symbols, expr, modules="math")
    values = root.table(CAP).values
    assert len(values) == 2 * len(MENU)
    for value, point in zip(values, points):
        assert value == pytest.approx(annealer(*point), rel=1e-9), point


def test_the_projection_objective_differs_by_candidate():
    """The test above reads a real objective: one that depends on the splits and the
    weight's residency."""
    expr = sympy.sympify(predict_ops([_gate_up(8)], _COST_PARAMS))
    assert {SPLIT_M, SPLIT_N, SPLIT_K, WEIGHT_LX} <= expr.free_symbols
    at = sympy.lambdify([SPLIT_M, SPLIT_N, SPLIT_K, WEIGHT_LX], expr, modules="math")
    assert len({round(at(*split, 0), 3) for split in MENU}) > 1
    assert at(4, 8, 1, 1) != at(4, 8, 1, 0)


# ------------------------------------------ synthetic examples of later optimizations


def _with(objective, **inputs):
    """A Case with the inputs named. An int n is a buffer's division, its split taking
    the values 1 to n; a pair (lo, size) is an integer variable over lo to
    lo + size - 1. Returns the Case and objective(s), s the symbols by name."""
    case = Case()
    for name, spec in inputs.items():
        if isinstance(spec, tuple):
            case.integer(name, *spec)
        else:
            case.add_input(name, **{name: [float(v) for v in range(1, spec + 1)]})
    s = {name: case.symbol(name) for name in inputs}
    return case, objective(s)


def _refused(reason):
    return pytest.mark.xfail(raises=TableLoweringError, strict=True, reason=reason)


# One example per optimization of a later stage: an objective whose one table is past
# its cap, which no stage before that one can lower and that stage's optimization can.
# Each is refused today, for the reason its mark gives; the stage that lowers it drops
# its mark, so the marks show what each stage adds. Stage 3's caching and Stage 4's
# numpy steps change no lowering, so they have none, and the fold's example comes with
# Stage 10, since whether it fits where the switch and Stage 9's rungs do not depends
# on the fold's own counts. The first example lowers today, so the check below runs end
# to end on every stage.
EXAMPLES = [
    pytest.param(
        lambda: _with(
            lambda s: s["a"] ** 0.5 + 3 * s["b"] - 2 * s["x"],
            a=8,
            b=6,
            x=(0, 2),
        ),
        CAP,
        id="trivial",
    ),
    pytest.param(
        lambda: _with(
            lambda s: s["a"] ** 0.5
            + sympy.log(s["b"])
            + s["c"] ** -0.25
            + s["d"] ** 1.5,
            a=10,
            b=10,
            c=10,
            d=10,
        ),
        CAP,
        id="linear_sum",
    ),
    pytest.param(
        lambda: _with(
            lambda s: s["k"] * (s["a"] ** 0.5 + sympy.log(s["b"]) + s["c"] ** -0.25),
            k=(1, 3),
            a=16,
            b=16,
            c=16,
        ),
        CAP,
        id="switch",
    ),
    pytest.param(
        lambda: _with(
            lambda s: s["x"] * sympy.log(s["a"] + 1 + s["x"] * s["a"] / 2),
            x=(0, 2),
            a=40,
        ),
        40,
        id="gate",
        marks=_refused(
            "the rest of a term a 0/1 literal gates reads the literal: one table over "
            "the literal and a, 80 entries under a cap of 40, and reading the rest "
            "with the literal fixed, which halves it, is not built"
        ),
    ),
    pytest.param(
        lambda: _with(
            lambda s: (sympy.floor(s["a"] / 16) * sympy.floor(s["b"] / 32)) ** 0.5,
            a=128,
            b=128,
        ),
        CAP,
        id="ranks",
    ),
    pytest.param(
        lambda: _with(
            lambda s: sympy.Max(s["a"], 1.5 * s["b"], 2 * s["c"] + 0.25),
            a=24,
            b=20,
            c=18,
        ),
        CAP,
        id="rung",
        marks=_refused(
            "a Max is one table over every input of its operands: three buffers need "
            "8640 entries, past the cap of 8192, and a Max read from the order of its "
            "operands' values is not built"
        ),
    ),
    pytest.param(
        lambda: _with(
            lambda s: sympy.Min(
                1000 - sum((20 + j) * s[f"x{j}"] for j in range(15)),
                900 - sum((10 + j) * s[f"x{j}"] for j in range(14)),
            ),
            **{f"x{j}": (0, 2) for j in range(15)},
        ),
        CAP,
        id="lin_min",
        marks=_refused(
            "a Min of two sums of 15 residency literals is one table of 32768 "
            "entries, past the cap of 8192, neither sum fits a table either, and a "
            "Min kept as a linear constraint is not built"
        ),
    ),
]


def _solve_at(
    case: Case, objective, assignment: tuple[int, ...]
) -> tuple[float, float]:
    """The objective's minimum and maximum with every variable fixed."""
    on = case.model.new_bool_var("assignment")
    for var, value in zip(case.variables, assignment):
        case.model.add(var == value).only_enforce_if(on)
    case.model.clear_assumptions()
    case.model.add_assumptions([on])
    found = []
    for sense in (case.model.minimize, case.model.maximize):
        sense(objective)
        solver = cp_model.CpSolver()
        status = solver.solve(case.model)
        assert status == cp_model.OPTIMAL, solver.status_name(status)
        found.append(solver.objective_value)
    return found[0], found[1]


@pytest.mark.parametrize("example, cap", EXAMPLES)
def test_a_later_optimization_lowers_its_example_exactly(example, cap):
    """Each example is marked to fail with TableLoweringError until the stage whose
    optimization lowers it. The marks are strict: an example that starts to lower fails
    the run until its mark is dropped. An example that lowers is checked at three
    assignments, each variable at its lowest value, at its highest and at a seeded
    random one: minimizing and maximizing agree and equal COST_SCALE times sympy's
    value, within half a unit per constraint the lowering adds, plus half."""
    case, expr = example()
    before = len(case.model.proto.constraints)
    _, objective = case.lower(expr, cap)
    bound = (len(case.model.proto.constraints) - before + 1) / 2
    domains = [case.values[v] for v in case.variables]
    for assignment in (tuple(d[0] for d in domains), tuple(d[-1] for d in domains)):
        low, high = _solve_at(case, objective, assignment)
        assert low == high
        assert abs(low - COST_SCALE * case.sympy_value(expr, assignment)) <= bound
