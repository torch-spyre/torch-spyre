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
"""The table lowering of a CP-SAT cost objective: the base mechanics, one table per
node, each node's list computed from its operands' lists. No device and no compile:
each test builds its inputs by hand and solves the lowered model with CP-SAT.

*Exact* means: every assignment of the inputs has exactly one solution, so each
position and the cost are determined by the inputs, and that solution's cost is
COST_SCALE times the expression evaluated by sympy, within the rounding bound. A list
is computed in floating point and rounded once, where it enters CP-SAT: it is held to
the expression's value, not to the bits of another evaluator.
"""

import inspect
import itertools
import math

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
    ConstantNode,
    AbstractRealization,
    Index,
    Input,
    InputIndex,
    Node,
    InputNode,
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
        self.inputs: dict[sympy.Symbol, Input] = {}
        self.variables: list[cp_model.IntVar] = []
        self.values: dict[cp_model.IntVar, list[int]] = {}

    def add_input(self, name: str, **columns: list[float]) -> Index:
        """A buffer's division: a variable in 0 .. n-1 picking a row of its menu, and
        one column per symbol of the buffer. Returns the position an InputNode makes
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
        is the value offset by lo. Returns the position an InputNode makes on it."""
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
    ) -> tuple[Node, cp_model.LinearExprT]:
        root = TableLowering(self.inputs, cap=cap).lower(expr)
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

    def assert_exact(self, expr: sympy.Basic, cap: int = CAP) -> Node:
        plan, objective = self.lower(expr, cap)
        cost = self.model.new_int_var(-(2**60), 2**60, "cost")
        self.model.add(cost == objective)
        rows = self.solutions(self.variables + [cost])
        assignments = list(itertools.product(*(self.values[v] for v in self.variables)))
        assert len(rows) == len(assignments)
        assert sorted(row[:-1] for row in rows) == sorted(assignments)
        bound = 0.5  # one table, rounded once
        for row in rows:
            exact = COST_SCALE * self.sympy_value(expr, row[:-1])
            assert abs(row[-1] - exact) <= bound, (row, exact)
        return plan


def _nodes(case: Case, expr: sympy.Basic, cap: int = CAP) -> dict[sympy.Basic, Node]:
    """Each Node under expr lowered by one printer, by its sympy node, children
    before their parents."""
    nodes: dict[sympy.Basic, Node] = {}

    def gather(node: Node) -> None:
        if node.expr in nodes:
            assert nodes[node.expr] is node  # one Node per sympy node
            return
        for child in node.children:
            gather(child)
        nodes[node.expr] = node

    gather(TableLowering(case.inputs, cap=cap).lower(expr))
    return nodes


# -------------------------------------------------------------------- the printer


def test_the_printer_prints_children_first_and_shares_a_subexpression():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    before = sympy.srepr(expr)
    nodes = _nodes(case, expr)
    order = list(nodes)
    for expr_node, node in nodes.items():
        for child in node.children:
            assert order.index(child.expr) < order.index(expr_node)
    holders = [n for n in nodes.values() if any(c is nodes[shared] for c in n.children)]
    assert len(holders) == 2
    assert len(order) == len(set(order))
    assert sympy.srepr(expr) == before


def test_a_piecewise_arm_gets_no_node():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    expr = sympy.Piecewise((a, a > 2), (a**2, True))
    nodes = _nodes(case, expr)
    assert not any(isinstance(e, ExprCondPair) for e in nodes)
    assert [c.expr for c in nodes[expr].children] == [a, a > 2, a**2, sympy.true]


def test_each_node_is_an_input_a_constant_or_an_operation():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    root2 = sympy.sqrt(2)
    expr = 3 * a + root2 * a**0.5
    nodes = _nodes(case, expr)
    assert type(nodes[a]) is InputNode
    assert nodes[a].input is case.inputs[a]
    assert type(nodes[sympy.Integer(3)]) is ConstantNode
    assert type(nodes[root2]) is ConstantNode  # an operation on numbers
    assert type(nodes[a**0.5]) is Node
    assert type(nodes[expr]) is Node


def test_the_lowering_is_a_sympy_printer_dispatching_on_the_sympy_class():
    """A node is printed by the print method of the nearest class along its MRO that
    has one, so a Dummy, a subclass of Symbol, is an InputNode, and a class with none
    goes to emptyPrinter, a Node. A RelayoutCharge's prices are data, not children. A
    printer keeps its Nodes: what two calls of lower share is one Node."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    dummy = sympy.Dummy("a")
    case.inputs[dummy] = case.inputs[a]
    printer = TableLowering(case.inputs, cap=CAP)
    assert isinstance(printer, Printer)
    leaf = printer.lower(dummy)
    assert type(leaf) is InputNode and leaf.input is case.inputs[a]
    assert type(printer.lower(sympy.log(a))) is Node
    assert type(printer.lower(sympy.log(sympy.Integer(2)) + sympy.pi)) is ConstantNode
    charge = RelayoutCharge(dummy, a, 3, 4, 5)
    assert [child.expr for child in printer.lower(charge).children] == [dummy, a]
    root = printer.lower(sympy.log(a) + dummy)
    assert [child.expr for child in root.children] == list(root.expr.args)
    assert all(child is printer.lower(child.expr) for child in root.children)
    assert root is printer.lower(root.expr)


def test_an_input_node_makes_its_position_on_its_variable():
    """The caller hands over a variable and values, and no position. Two symbols of
    one variable get equal positions, so one table over both is on one axis."""
    case = Case()
    position = case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    offset = case.integer("y", 2, 4)
    d0, d1, y = case.symbol("d0"), case.symbol("d1"), case.symbol("y")
    nodes = _nodes(case, d0 * d1 + y)
    assert nodes[d0].realization.index == nodes[d1].realization.index == position
    assert nodes[y].realization.index == offset and offset.lo == 2
    assert list(nodes[d1].realization.values) == [8.0, 4.0, 2.0]
    assert nodes[d0 * d1].realization.index == position  # one axis, 3 entries
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
        node = InputNode(d0, (), CAP, input=refused)  # type: ignore[arg-type]
        with pytest.raises(TableLoweringError, match=match) as caught:
            node.realization
        assert caught.value.node == d0


def test_a_node_gathers_its_columns_from_its_children():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = a**0.5 + 2 * b
    nodes = _nodes(case, expr)
    assert set(nodes[expr].columns) == {a, b}
    assert all(nodes[expr].columns[s] is nodes[s].columns[s] for s in (a, b))
    assert dict(nodes[2 * b].columns) == {b: nodes[b].columns[b]}
    assert dict(nodes[sympy.Integer(2)].columns) == {}


def test_a_node_realizes_itself_when_it_is_read():
    """No lowering is passed in, and nothing visits: a node reads its children and
    its cap, the first time its realization is read."""
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    leaf = InputNode(a, (), CAP, input=case.inputs[a])
    half = ConstantNode(sympy.Float(0.5), (), CAP)
    root = Node(a**0.5, (leaf, half), CAP)
    assert not any("realization" in vars(node) for node in (root, leaf, half))
    assert root.realization is not None
    assert all("realization" in vars(node) for node in (leaf, half))  # read by the root
    assert root.realization.index == position
    assert list(root.realization.values) == pytest.approx([1.0, 2**0.5, 2.0])
    assert Node(a**0.5, (leaf, half), 3).realization.index == position  # at the cap
    small = Node(a**0.5, (leaf, half), 2)
    with pytest.raises(
        TableLoweringError,
        match="one table over its inputs would hold 3 entries, past the cap of 2",
    ):
        small.realization
    with pytest.raises(TableLoweringError, match="no input"):
        InputNode(a, (), CAP).realization


def _count_evaluations(monkeypatch) -> list[sympy.Basic]:
    """The nodes whose lists are computed from here on, in order: every list a node
    makes goes through the lowering's _values, once."""
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
    root = TableLowering(case.inputs, cap=5).lower(a**0.5 * b)
    with pytest.raises(TableLoweringError, match="6 entries, past the cap of 5"):
        root.realization
    assert calls == []  # not the root's table, and not a**0.5 under it


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


def test_lower_realizes_nothing_and_a_refusal_leaves_the_model_as_it_was():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    root = TableLowering(case.inputs, cap=CAP).lower(a**0.5 * b)
    assert type(root) is Node and root.expr == a**0.5 * b
    assert "realization" not in vars(root)
    root.cp_expr(case.model)
    size = len(case.model.proto.constraints), len(case.model.proto.variables)
    wide = TableLowering(case.inputs, cap=4).lower(a**0.5 * b)
    with pytest.raises(TableLoweringError, match="past the cap of 4"):
        wide.cp_expr(case.model)
    assert (len(case.model.proto.constraints), len(case.model.proto.variables)) == size


# ------------------------------------------------------------------------- values


def test_a_realization_is_abstract_and_a_table_is_one():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(TypeError):
        AbstractRealization()  # type: ignore[abstract]

    class Partial(AbstractRealization):
        """A realization with no cp_expr."""

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]
    table = TableRealization(values=[1.0, 2.0, 4.0], index=position)
    assert isinstance(table, AbstractRealization)
    assert table.finite
    assert not TableRealization(values=[1.0, np.inf, 4.0], index=position).finite
    a = case.symbol("a")
    assert isinstance(
        TableLowering(case.inputs, cap=CAP).lower(a**0.5).realization,
        AbstractRealization,
    )


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


def _points(node: Node) -> tuple[list[sympy.Symbol], list[tuple[float, ...]]]:
    """node's free symbols, sorted by sympy.default_sort_key, and their values at each
    entry of its table: their positions in that order, row-major, the last fastest."""
    columns = node.columns
    symbols = sorted(columns, key=sympy.default_sort_key)
    positions = list(dict.fromkeys(columns[s].index for s in symbols))
    grid = np.indices(tuple(p.size for p in positions)).reshape(len(positions), -1)
    axis = {p: k for k, p in enumerate(positions)}
    arguments = [columns[s].values[grid[axis[columns[s].index]]] for s in symbols]
    return symbols, list(zip(*arguments))


def _stage_1(node: Node) -> np.ndarray:
    """node's list as Stage 1 computed it: its whole expression lambdified with the
    numpy module over its inputs, point by point when it takes scalars only."""
    symbols, points = _points(node)
    func = sympy.lambdify(symbols, node.expr, modules="numpy")
    with np.errstate(all="ignore"):
        try:
            result = np.asarray(func(*np.array(points).T), dtype=float)
        except (TypeError, ValueError):
            result = np.array([func(*point) for point in points], dtype=float)
    return np.broadcast_to(result, (len(points),))


def _list(case: Case, expr: sympy.Basic) -> list[float]:
    table = TableLowering(case.inputs, cap=CAP).lower(expr).realization
    assert isinstance(table, TableRealization)
    return list(table.values)


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
        root = TableLowering(case.inputs, cap=CAP).lower(expr)
        assert np.all(np.isfinite(_stage_1(root)))
        with pytest.raises(TableLoweringError, match="not finite") as caught:
            case.lower(expr)
        assert caught.value.node == child


def test_a_node_is_evaluated_once_and_only_when_its_realization_is_read(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls = _count_evaluations(monkeypatch)
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    nodes = _nodes(case, expr)
    assert calls == []  # making the nodes realizes nothing
    computed = [e for e in nodes if e.free_symbols and not isinstance(e, sympy.Symbol)]
    for _ in range(2):
        for e in computed:
            assert nodes[e].realization is not None
    assert len(calls) == len(computed) == len(set(calls))
    calls.clear()
    TableLowering(case.inputs, cap=CAP).lower(expr).cp_expr(case.model)
    # reading the root reads every node under it, each once, the root last
    assert len(calls) == len(computed) and set(calls) == set(computed)
    assert calls[-1] == expr


# ---------------------------------------------------------------- exact lowering


def test_one_table_over_two_inputs_of_different_sizes_is_exact():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 5.0, 7.0])
    a, b = case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(a**2 * b + b**0.5 / a)
    index = plan.realization.index
    assert index.size == 12


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


def test_functions_numpy_does_not_share_give_sympys_values():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    case.assert_exact(sympy.sec(a / 4) + sympy.sinc(a))


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
    index = plan.realization.index
    assert index.size == 6


def test_an_undefined_function_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    expr = sympy.Function("f")(a)
    with pytest.raises(TableLoweringError, match="no numeric value") as caught:
        case.lower(expr)
    assert caught.value.node == expr


def test_a_function_that_takes_scalars_only_is_evaluated_point_by_point():
    """On an array, an implemented function that tests its value with ``if`` raises
    ValueError, and one that rounds raises TypeError; both are retried point by
    point. One that fails on a point too is refused, naming its node."""

    class Step(sympy.Function):
        @staticmethod
        def _imp_(x):
            return 3.0 if x > 2 else 1.0

    class Broken(sympy.Function):
        @staticmethod
        def _imp_(x):
            raise KeyError(x)

    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    case.assert_exact(Step(a) + a)
    with pytest.raises(TableLoweringError, match="no value .KeyError") as caught:
        case.lower(Broken(a))
    assert caught.value.node == Broken(a)


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
    assert caught.value.node == 1e14 * a


def test_a_symbol_with_no_input_is_refused_by_name():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a, z = case.symbol("a"), sympy.Symbol("z", positive=True)
    with pytest.raises(TableLoweringError) as caught:
        case.lower(a + z)
    assert caught.value.node == z
    assert "no input" in caught.value.reason


def test_the_naive_lowering_explodes_past_the_cap():
    """One table per node puts the sum over all three buffers in one table, 21 * 11 *
    20 = 4620 entries, though no term reads more than one buffer. Under a cap of 4096
    the sum is refused, by name; keeping a sum linear is the next optimization."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 22)])
    case.add_input("b", b=[float(v) for v in range(1, 12)])
    case.add_input("c", c=[float(v) for v in range(1, 21)])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    expr = a**0.5 + b**0.5 + c**0.25
    with pytest.raises(
        TableLoweringError, match="4620 entries, past the cap of 4096"
    ) as caught:
        case.lower(expr, cap=4096)
    assert caught.value.node == expr
    plan, _ = case.lower(expr, cap=8192)
    index = plan.realization.index
    assert index.size == 4620


# ---------------------------------------------------------------- a node's function


def _count_lambdify(monkeypatch) -> list[sympy.Basic]:
    """The expressions lambdified from here on."""
    calls: list[sympy.Basic] = []
    lambdify = sympy.lambdify

    def counting(args, expr, **kwargs):
        calls.append(expr)
        return lambdify(args, expr, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counting)
    return calls


def test_a_node_is_lambdified_only_when_the_core_has_no_function(monkeypatch):
    """A node's list comes from the numpy core's function of its class, with no
    lambdify: forty terms of one shape make hundreds of nodes and none is lambdified.
    A node of a class the core has no function for, here sinc and sec, lambdifies its
    class over placeholders once, when it is realized. Nothing is kept between
    lowerings, so a second lowering lambdifies as many again."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 65)])
    case.add_input("b", b=[float(v) for v in range(1, 9)])
    a, b = case.symbol("a"), case.symbol("b")
    terms = [
        (i + 1)
        * sympy.Piecewise(
            (sympy.floor((i + 64) / a) ** -0.25, a > i), (sympy.log(b + i), True)
        )
        + sympy.Max(a**0.5, b / (i + 1))
        for i in range(40)
    ]
    expr = sympy.Add(*terms) + sympy.sinc(a) * sympy.sec(b / 16)
    calls = _count_lambdify(monkeypatch)
    nodes = _nodes(case, expr)
    nodes[expr].cp_expr(case.model)
    assert len([n for n in nodes.values() if type(n) is Node]) > 300
    assert sorted(str(c) for c in calls) == ["sec(operand0)", "sinc(operand0)"]
    nodes[expr].cp_expr(cp_model.CpModel())  # a realization is kept by its node
    assert len(calls) == 2
    _nodes(case, expr)[expr].cp_expr(cp_model.CpModel())
    assert len(calls) == 4
    np.testing.assert_allclose(
        nodes[expr].realization.values, _stage_1(nodes[expr]), rtol=1e-12
    )


def test_a_function_takes_its_operands_by_their_own_names():
    """lambdify renames every argument when any is a Dummy, rebuilding the node once
    per argument: seconds for a Min or Max of a few dozen operands. Plain symbols
    whose names are identifiers are taken as they are, so the function's parameters
    are the placeholders' own names, and a Min of 35 is quick to lambdify."""
    func = lowering._lambdified(sympy.Min, (None,) * 35)
    assert list(inspect.signature(func).parameters) == [
        f"operand{i}" for i in range(35)
    ]
    assert func(*np.arange(35.0, 0.0, -1.0)) == 1.0


def test_a_constant_is_read_as_the_double_it_is():
    """A list is the expression's value in floating point, rounded once where it
    enters CP-SAT, and is not held to another evaluator's bits. lambdify prints a
    Float to 15 digits, 2/3 as 0.666666666666667, above the double, so its
    ceiling(2/3 * c) is one high at c = 3, 6 and 9. A node reads the constant as the
    double it is."""
    case = Case()
    case.add_input("c", c=[1.0, 3.0, 6.0, 7.0, 9.0])
    c = case.symbol("c")
    expr = sympy.ceiling(sympy.Float(2) / 3 * c)
    root = case.assert_exact(expr)
    assert list(root.realization.values) == [1.0, 2.0, 4.0, 5.0, 6.0]


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
    lambda a, b, c: sympy.sec(a / 4) + sympy.sinc(a),
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
    nodes = _nodes(case, expr)
    nodes[expr].cp_expr(case.model)
    for node in nodes.values():
        if type(node) is Node:
            np.testing.assert_allclose(
                node.realization.values,
                _stage_1(node),
                rtol=1e-12,
                err_msg=str(node.expr),
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
    root = TableLowering(case.inputs, cap=CAP).lower(expr)
    prices = [f for f in expr.atoms(sympy.Function) if "delivery_price" in str(f)]
    assert bool(prices) == (rows == 8)
    symbols, points = _points(root)
    annealer = sympy.lambdify(symbols, expr, modules="math")
    values = root.realization.values
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
# its mark, so the marks show what each stage adds. Stage 3's caching and Stage 4's numpy steps change no lowering,
# so they have none, and the fold's example comes with Stage 10, since whether it fits
# where the switch and Stage 9's rungs do not depends on the fold's own counts. The
# first example lowers today, so the check below runs end to end on every stage.
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
        marks=_refused(
            "a sum is one table over every input of its terms: four terms on a buffer "
            "each need 10^4 entries, past the cap of 8192, and a sum read term by "
            "term is not built"
        ),
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
        marks=_refused(
            "a product is one table over every input of its factors: a factor of 3 "
            "values times a sum over three buffers needs 12288 entries, past the cap "
            "of 8192, and switching the sum on the factor's value is not built"
        ),
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
            "a term a 0/1 literal gates is one table over the literal and the rest's "
            "inputs, 80 entries under a cap of 40, and reading the rest with the "
            "literal fixed, which halves it, is not built"
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
        marks=_refused(
            "a product under a power is one table over both buffers, 16384 entries, "
            "past the cap of 8192, and a table over the factors' distinct values, 45 "
            "entries, is not built"
        ),
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
