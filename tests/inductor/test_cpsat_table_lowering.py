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
node, every value by lambdify. No device and no compile: each test builds its inputs
by hand and solves the lowered model with CP-SAT.

*Exact* means: every assignment of the inputs has exactly one solution, so each
position and the cost are determined by the inputs, and that solution's cost is
COST_SCALE times the expression evaluated by sympy, within the rounding bound.
"""

import itertools

import numpy as np
import pytest
import sympy
from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair

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
    values_on_grid,
)
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

    def columns(self) -> dict[sympy.Symbol, TableRealization]:
        """Each symbol's column, as its InputNode makes it."""
        nodes = {s: InputNode(s, (), CAP, input=self.inputs[s]) for s in self.inputs}
        return {s: node.columns[s] for s, node in nodes.items()}

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


# ----------------------------------------------------------------------- the walk


def test_the_factory_makes_children_first_and_shares_a_subexpression():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    before = sympy.srepr(expr)
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
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
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert not any(isinstance(e, ExprCondPair) for e in nodes)
    assert [c.expr for c in nodes[expr].children] == [a, a > 2, a**2, sympy.true]


def test_each_node_is_an_input_a_constant_or_an_operation():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    root2 = sympy.sqrt(2)
    expr = 3 * a + root2 * a**0.5
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert type(nodes[a]) is InputNode
    assert nodes[a].input is case.inputs[a]
    assert type(nodes[sympy.Integer(3)]) is ConstantNode
    assert type(nodes[root2]) is ConstantNode  # an operation on numbers
    assert type(nodes[a**0.5]) is Node
    assert type(nodes[expr]) is Node


def test_an_input_node_makes_its_position_on_its_variable():
    """The caller hands over a variable and values, and no position. Two symbols of
    one variable get equal positions, so one table over both is on one axis."""
    case = Case()
    position = case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    offset = case.integer("y", 2, 4)
    d0, d1, y = case.symbol("d0"), case.symbol("d1"), case.symbol("y")
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(d0 * d1 + y)
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
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
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


def test_a_table_past_the_cap_is_never_evaluated(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls: list[sympy.Basic] = []
    lambdify = sympy.lambdify

    def counting(args, expr, **kwargs):
        calls.append(expr)
        return lambdify(args, expr, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counting)
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


# --------------------------------------------------------------- values_on_grid


def test_values_on_grid_is_row_major_with_the_last_position_fastest():
    case = Case()
    a = case.add_input("a", a=[0.0, 1.0])
    b = case.add_input("b", b=[0.0, 1.0, 2.0])
    c = case.add_input("c", c=[0.0, 1.0, 2.0, 3.0])
    sa, sb, sc = case.symbol("a"), case.symbol("b"), case.symbol("c")
    values = values_on_grid(100 * sa + 10 * sb + sc, case.columns(), [a, b, c])
    expected = [
        100 * i + 10 * j + k for i in range(2) for j in range(3) for k in range(4)
    ]
    assert list(values) == expected


def test_values_on_grid_moves_two_symbols_of_one_position_together():
    case = Case()
    p = case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    d0, d1 = case.symbol("d0"), case.symbol("d1")
    assert list(values_on_grid(d0 * d1, case.columns(), [p])) == [8.0, 8.0, 8.0]


def test_values_on_grid_broadcasts_a_constant():
    case = Case()
    p = case.add_input("a", a=[1.0, 2.0, 4.0])
    values = values_on_grid(sympy.Integer(5), case.columns(), [p])
    assert list(values) == [5.0, 5.0, 5.0]


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
    position = case.add_input("a", a=[1.0, 2.0, 4.0], b=[1.0, np.inf, 4.0])
    a, b = case.symbol("a"), case.symbol("b")
    infinite = 1 / (a - 2)
    nan = sympy.sqrt(a - 3)
    for child, expr in (
        (infinite, sympy.Min(infinite, 5)),
        (nan, sympy.Piecewise((1, nan > 0), (2, True))),
        (sympy.oo, sympy.Piecewise((sympy.oo, a > 4), (a, True))),
        (b, sympy.Min(a, b)),
    ):
        assert np.all(np.isfinite(values_on_grid(expr, case.columns(), [position])))
        with pytest.raises(TableLoweringError, match="not finite") as caught:
            case.lower(expr)
        assert caught.value.node == child


def test_a_node_is_lambdified_once_and_only_when_its_realization_is_read(
    monkeypatch,
):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls: list[sympy.Basic] = []
    lambdify = sympy.lambdify

    def counting(args, expr, **kwargs):
        calls.append(expr)
        return lambdify(args, expr, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counting)
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
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
