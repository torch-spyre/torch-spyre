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
"""The table lowering of a CP-SAT cost objective. No device and no compile: each test
builds its inputs by hand and solves the lowered model with CP-SAT.

*Exact* means: every assignment of the inputs has exactly one solution, so each
position, true/false and cost is determined by the inputs, and that solution's cost is
COST_SCALE times the expression evaluated by sympy, within the test's bound.
"""

import inspect
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import sympy

# ortools is not installed on s390x or ppc64le (pyproject.toml), and the table module
# imports it at its top: skip the file there rather than fail to collect it.
pytest.importorskip("ortools.sat.python.cp_model")

from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair

from torch_spyre._inductor.scratchpad.cpsat_table_lowering import (
    COST_SCALE,
    UNIT,
    AbstractRealization,
    AddNode,
    CombineIndex,
    CompactIndex,
    ConstantNode,
    ExtremeIndex,
    ExtremeNode,
    ExtremeRealization,
    GateRealization,
    Index,
    Input,
    InputIndex,
    InputNode,
    LinearRealization,
    LocateIndex,
    MulNode,
    Node,
    RelayoutChargeNode,
    SwitchRealization,
    TableLowering,
    TableLoweringError,
    TableRealization,
    UnitIndex,
    clamped_ladder,
    fold_reachable,
    inputs_from_maps,
    lower_cost_with_tables,
    rungs_on,
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

    def literal(self, name: str) -> Index:
        """A residency literal: a 0/1 variable that is its own value."""
        var = self.model.new_bool_var(name)
        self.inputs[sympy.Symbol(name, integer=True, nonnegative=True)] = (var, [0, 1])
        self.variables.append(var)
        self.values[var] = [0, 1]
        return InputIndex(var, size=2)

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

    def elements(self) -> list[int]:
        """The length of each element constraint's list, ascending."""
        return sorted(
            len(c.element.exprs)
            for c in self.model.proto.constraints
            if c.has_element()
        )

    def lin_maxes(self) -> list:
        """The lin_max constraints in the model."""
        return [c.lin_max for c in self.model.proto.constraints if c.has_lin_max()]

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
            value = float(column[self.values[var].index(values[var])])
            point[symbol] = (
                sympy.Integer(int(value)) if value.is_integer() else sympy.Float(value)
            )
        return float(expr.xreplace(point))

    def assert_exact(
        self, expr: sympy.Basic, cap: int = CAP, bound: float = 0.5
    ) -> Node:
        plan, objective = self.lower(expr, cap)
        cost = self.model.new_int_var(-(2**60), 2**60, "cost")
        self.model.add(cost == objective)
        rows = self.solutions(self.variables + [cost])
        assignments = list(itertools.product(*(self.values[v] for v in self.variables)))
        assert len(rows) == len(assignments)
        assert sorted(row[:-1] for row in rows) == sorted(assignments)
        for row in rows:
            exact = COST_SCALE * self.sympy_value(expr, row[:-1])
            assert abs(row[-1] - exact) <= bound, (row, exact)
        return plan


def _literals(case: Case, count: int) -> list[sympy.Symbol]:
    for j in range(count):
        case.literal(f"x{j}")
    return [case.symbol(f"x{j}") for j in range(count)]


def _on_inputs(node: Node) -> np.ndarray:
    """node's table read at every entry of one table over its inputs, row-major in
    the order of its symbols sorted by sympy.default_sort_key: its values whatever
    position it took."""
    symbols = sorted(node.columns, key=sympy.default_sort_key)
    positions = list(dict.fromkeys(node.columns[s].index for s in symbols))
    table = node.table
    size = math.prod(p.size for p in positions)
    return np.broadcast_to(table.values[table.index.on_grid(lowering._grid(positions))], (size,))


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


def test_each_node_has_the_class_its_sympy_class_calls_for():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.literal("x")
    a, x = case.symbol("a"), case.symbol("x")
    root2 = sympy.sqrt(2)
    charge = RelayoutCharge(x, a, 3, 4, 5)
    expr = 3 * a + root2 * a**0.5 + sympy.Max(a, 2) + charge
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert type(nodes[a]) is InputNode
    assert nodes[a].input is case.inputs[a]
    assert type(nodes[sympy.Integer(3)]) is ConstantNode
    assert type(nodes[root2]) is ConstantNode  # an operation on numbers
    assert type(nodes[a**0.5]) is Node
    assert type(nodes[3 * a]) is MulNode
    assert type(nodes[sympy.Max(a, 2)]) is ExtremeNode
    assert type(nodes[charge]) is RelayoutChargeNode
    assert type(nodes[expr]) is AddNode


def test_an_input_node_makes_its_position_on_its_variable():
    """The caller hands over a variable and values, and no position. Two symbols of
    one variable get equal positions, so one table over both is on one axis."""
    case = Case()
    position = case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    offset = case.integer("y", 2, 4)
    d0, d1, y = case.symbol("d0"), case.symbol("d1"), case.symbol("y")
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(d0 * d1 + y)
    assert nodes[d0].table.index == nodes[d1].table.index == position
    assert nodes[y].table.index == offset and offset.lo == 2
    assert list(nodes[d1].table.values) == [8.0, 4.0, 2.0]
    assert nodes[d0 * d1].table.index == position  # one axis, 3 entries
    case.assert_exact(d0 * d1 + y, bound=1.0)
    division = case.variables[0]
    literal = case.model.new_bool_var("literal")
    for refused, match in (
        ((division, [1.0, 2.0]), "2 values for a variable with 3, 0 to 2"),
        ((division, [1.0, "two", 4.0]), "values that are not one number each"),
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


def test_a_node_makes_its_table_when_it_is_read():
    """No lowering is passed in, and nothing visits: a node reads its children and
    its cap, the first time its table is read."""
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    leaf = InputNode(a, (), CAP, input=case.inputs[a])
    half = ConstantNode(sympy.Float(0.5), (), CAP)
    root = Node(a**0.5, (leaf, half), CAP)
    assert not any("tabulated" in vars(node) for node in (root, leaf, half))
    assert root.realization is root.table
    assert all("tabulated" in vars(node) for node in (leaf, half))  # read by the root
    assert root.table.index == position
    assert list(root.table.values) == pytest.approx([1.0, 2**0.5, 2.0])
    assert Node(a**0.5, (leaf, half), 3).table.index == position  # at the cap
    small = Node(a**0.5, (leaf, half), 2)
    assert small.tabulated is None
    with pytest.raises(
        TableLoweringError,
        match="one table over its inputs would hold 3 entries, past the cap of 2",
    ):
        small.realization
    with pytest.raises(TableLoweringError, match="no input"):
        InputNode(a, (), CAP).realization


def _count_evaluations(monkeypatch) -> list[sympy.Basic]:
    """The nodes whose lists are computed from here on, in order: every list a node
    computes goes through the lowering's _values, once."""
    calls: list[sympy.Basic] = []
    values = lowering._values

    def counting(expr, count, compute):
        calls.append(expr)
        return values(expr, count, compute)

    monkeypatch.setattr(lowering, "_values", counting)
    return calls


def test_a_table_past_the_cap_is_never_evaluated(monkeypatch):
    """Under a power the product is a table: one over its inputs and one from its
    parts are both 6 entries, past the cap of 5, so its own list is never computed.
    Read linearly it is a switch, which fits."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls = _count_evaluations(monkeypatch)
    root = TableLowering(case.inputs, cap=5).lower(sympy.log(a**0.5 * b))
    with pytest.raises(TableLoweringError, match="6 entries, past the cap of 5"):
        root.realization
    assert a**0.5 * b not in calls and root.expr not in calls
    product = TableLowering(case.inputs, cap=5).lower(a**0.5 * b)
    assert isinstance(product.realization, SwitchRealization)


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
    assert type(root) is MulNode and root.expr == a**0.5 * b
    assert "realization" not in vars(root) and "tabulated" not in vars(root)
    root.cp_expr(case.model)
    size = len(case.model.proto.constraints), len(case.model.proto.variables)
    wide = TableLowering(case.inputs, cap=4).lower(sympy.log(a**0.5 * b))
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
        """A realization with no build."""

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
    rank = CompactIndex(a, (1, 0, 1), 2)
    assert rank == CompactIndex(a, (1, 0, 1), 2) and rank != CompactIndex(a, (0, 1, 1), 2)


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


def test_a_position_read_twice_in_one_build_is_built_once():
    case = Case()
    a = case.add_input("a", a=[1.0, 2.0, 4.0, 8.0])
    rank = CompactIndex(a, (0, 0, 1, 1), 2)
    built: dict = {}
    first = CompactIndex(a, (0, 0, 1, 1), 2).cp_expr(case.model, built)
    assert rank.cp_expr(case.model, built) is first
    assert case.elements() == [4]
    rank.cp_expr(case.model)  # another build
    assert case.elements() == [4, 4]


# ------------------------------------------------------------------- a node's list


def _points(node: Node) -> tuple[list[sympy.Symbol], list[tuple[float, ...]]]:
    """node's free symbols, sorted by sympy.default_sort_key, and their values at each
    entry of one table over its inputs: their positions in that order, row-major, the
    last fastest."""
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
    return list(_on_inputs(TableLowering(case.inputs, cap=CAP).lower(expr)))


def test_a_list_is_row_major_with_the_last_position_fastest():
    """Each term is on one position, and one table of the sum reads each at its
    own."""
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


def test_a_node_is_evaluated_once_and_only_when_it_is_read(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls = _count_evaluations(monkeypatch)
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert calls == []  # making the nodes realizes nothing
    computed = [e for e in nodes if e.free_symbols and not isinstance(e, sympy.Symbol)]
    for _ in range(2):
        for e in computed:
            assert nodes[e].table is not None
    assert len(calls) == len(computed) == len(set(calls))
    calls.clear()
    TableLowering(case.inputs, cap=CAP).lower(expr).cp_expr(case.model)
    # every node computed is under the root, and computed once
    assert len(calls) == len(set(calls)) and set(calls) <= set(computed)
    assert sympy.log(shared) in calls


# ---------------------------------------------------------------- exact lowering


def test_one_table_over_two_inputs_of_different_sizes_is_exact():
    """Under a log the sum is read as one table over both inputs, not symmetric in
    them."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 5.0, 7.0])
    a, b = case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(sympy.log(a**2 * b + b**0.5 / a))
    assert plan.realization.index.size == 12


def test_variables_above_and_below_zero_read_their_entries():
    case = Case()
    case.integer("x", 2, 4)
    case.integer("y", -1, 3)
    x, y = case.symbol("x"), case.symbol("y")
    case.assert_exact(x**1.5 * y + 3 * x, bound=1.0)


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
    case.assert_exact(sympy.sec(a / 4) + sympy.sinc(a), bound=1.0)


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
    case.assert_exact(Step(a) + a, bound=1.0)
    with pytest.raises(TableLoweringError, match="no value .KeyError") as caught:
        case.lower(Broken(a))
    assert caught.value.node == Broken(a)


def test_a_coefficient_is_applied_before_the_rounding():
    """On a term, and on a sum: 7 * split ** 0.5 at split 2 is 10137, not 7 * 1448,
    and 7 * (split ** 0.5 + split ** 0.7) is one list, rounded once."""
    for on_sum, expected in (
        (False, 10137),
        (True, round(COST_SCALE * 7 * (2**0.5 + 2**0.7))),
    ):
        case = Case()
        case.add_input("a", a=[1.0, 2.0, 4.0])
        a = case.symbol("a")
        expr = sympy.Mul(7, a**0.5 + a**0.7, evaluate=False) if on_sum else 7 * a**0.5
        _, objective = case.lower(expr)
        rows = dict(case.solutions([case.variables[0], objective]))
        assert rows[1] == expected


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


# ----------------------------------------------------- one function for each class


def _count_lambdify(monkeypatch) -> list[sympy.Basic]:
    """The expressions lambdified from here on, the lowering's functions forgotten."""
    calls: list[sympy.Basic] = []
    lambdify = sympy.lambdify

    def counting(args, expr, **kwargs):
        calls.append(expr)
        return lambdify(args, expr, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counting)
    lowering._lambdified.cache_clear()
    return calls


def test_a_function_is_lambdified_once_for_each_class_and_form(monkeypatch):
    """Forty terms of one shape make hundreds of nodes and a few functions: every node
    of one class and form of operands shares one."""
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
    expr = sympy.Add(*terms)
    calls = _count_lambdify(monkeypatch)
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    np.testing.assert_allclose(_on_inputs(nodes[expr]), _stage_1(nodes[expr]), rtol=1e-12)
    computed = [n for n in nodes.values() if type(n) is not InputNode]
    assert len(computed) > 300
    assert len(calls) == len({str(c) for c in calls}) <= 12


def test_a_function_takes_its_operands_by_their_own_names():
    """lambdify renames every argument when any is a Dummy, rebuilding the node once
    per argument: seconds for a Min or Max of a few dozen operands. Plain symbols
    whose names are identifiers are taken as they are, so the function's parameters
    are the placeholders' own names, and a Min of 35 is quick to lambdify."""
    lowering._lambdified.cache_clear()
    func = lowering._lambdified(sympy.Min, (None,) * 35)
    assert list(inspect.signature(func).parameters) == [
        f"operand{i}" for i in range(35)
    ]
    assert func(*np.arange(35.0, 0.0, -1.0)) == 1.0


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
def test_every_table_is_stage_1s_at_every_input(make):
    """Whatever position a node's table takes (over its inputs, its operands' ranks
    or a ladder), read at each assignment of its inputs it is Stage 1's value."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0, 49.0])
    case.add_input("b", b=[1.0, 3.0])
    case.add_input("c", c=[1.0, 3.0, 7.0])
    expr = make(case.symbol("a"), case.symbol("b"), case.symbol("c"))
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    for node in nodes.values():
        if node.columns:
            np.testing.assert_allclose(
                _on_inputs(node), _stage_1(node), rtol=1e-12, err_msg=str(node.expr)
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


def _projection(rows: int) -> tuple[Case, sympy.Basic]:
    expr = sympy.sympify(predict_ops([_gate_up(rows)], _COST_PARAMS))
    case = Case()
    candidate = case.model.new_int_var(0, len(MENU) - 1, "division_mm")
    lit = case.model.new_bool_var("is_lx")
    for k, split in enumerate((SPLIT_M, SPLIT_N, SPLIT_K)):
        case.inputs[split] = (candidate, [float(s[k]) for s in MENU])
    case.inputs[WEIGHT_LX] = (lit, [0.0, 1.0])
    case.variables += [candidate, lit]
    case.values[candidate], case.values[lit] = list(range(len(MENU))), [0, 1]
    return case, expr


@pytest.mark.parametrize("rows", [8, 64], ids=["decode", "past-decode"])
def test_the_cost_models_objective_has_the_annealers_values(rows):
    """The objective predict_ops builds for the projection, its splits on one
    candidate variable and the weight's residency on another: its table at each
    candidate and residency is sympy.lambdify with the math module, the annealer's
    evaluator, and so is the lowered objective, within its rounding. At decode it
    holds the delivery-price functions, which take scalars only."""
    case, expr = _projection(rows)
    root = TableLowering(case.inputs, cap=CAP).lower(expr)
    prices = [f for f in expr.atoms(sympy.Function) if "delivery_price" in str(f)]
    assert bool(prices) == (rows == 8)
    symbols, points = _points(root)
    annealer = sympy.lambdify(symbols, expr, modules="math")
    values = _on_inputs(root)
    assert len(values) == 2 * len(MENU)
    for value, point in zip(values, points):
        assert value == pytest.approx(annealer(*point), rel=1e-9), point
    case.assert_exact(expr, bound=8.0)


def test_the_projection_objective_differs_by_candidate():
    """The test above reads a real objective: one that depends on the splits and the
    weight's residency."""
    expr = sympy.sympify(predict_ops([_gate_up(8)], _COST_PARAMS))
    assert {SPLIT_M, SPLIT_N, SPLIT_K, WEIGHT_LX} <= expr.free_symbols
    at = sympy.lambdify([SPLIT_M, SPLIT_N, SPLIT_K, WEIGHT_LX], expr, modules="math")
    assert len({round(at(*split, 0), 3) for split in MENU}) > 1
    assert at(4, 8, 1, 1) != at(4, 8, 1, 0)


# ------------------------------------------------------------ linear sums (Stage 5)


def test_a_sum_is_linear_with_no_table_across_two_buffers():
    """One table over the sum's three buffers is 21 * 11 * 20 = 4620 entries, past a
    cap of 4096: the sum is three lists, one per buffer. Read as a table, under a
    log, it is refused past the cap."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 22)])
    case.add_input("b", b=[float(v) for v in range(1, 12)])
    case.add_input("c", c=[float(v) for v in range(1, 21)])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    expr = a**0.5 + b**0.5 + c**0.25
    plan = case.assert_exact(expr, cap=4096, bound=1.5)
    assert isinstance(plan.realization, LinearRealization)
    assert case.elements() == [11, 20, 21]
    with pytest.raises(
        TableLoweringError, match="4620 entries, past the cap of 4096"
    ) as caught:
        case.lower(sympy.log(expr), cap=4096)
    assert caught.value.node == sympy.log(expr)


def test_lists_on_one_position_are_one_list_rounded_once():
    """split ** 0.5 + split ** 0.7 at split 2 is 3112, the rounding of 1448.15 +
    1663.49, not 1448 + 1663: one lookup."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    _, objective = case.lower(a**0.5 + a**0.7)
    assert case.elements() == [3]
    rows = dict(case.solutions([case.variables[0], objective]))
    assert rows[1] == 3112


def test_an_affine_list_adds_no_table():
    case = Case()
    case.add_input("a", a=[1.0, 3.0, 5.0])
    case.literal("x")
    a, x = case.symbol("a"), case.symbol("x")
    case.assert_exact(3 * a + 2 - 5 * x, bound=1.5)
    assert case.elements() == []


def test_a_sum_under_a_power_is_read_as_its_table():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = (a**0.5 + b) ** 2.5
    plan = case.assert_exact(expr)
    assert isinstance(plan.realization, TableRealization)
    assert case.elements() == [6]
    with pytest.raises(TableLoweringError, match="6 entries, past the cap of 5"):
        Case.lower(case, expr, cap=5)


def test_a_part_read_by_two_parents_is_built_once():
    """floor(a/2) is a term of the sum and an operand of the Min kept linear: one
    realization, one lookup."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0, 8.0])
    x = _literals(case, 5)
    a = case.symbol("a")
    shared = sympy.floor(a / 3) ** 0.5
    expr = shared + sympy.Min(shared, sum((j + 1) * x[j] for j in range(5)))
    case.assert_exact(expr, cap=16, bound=1.5)
    assert case.elements() == [4]


# -------------------------------------------------------------- the switch (Stage 6)


def test_a_switch_multiplies_its_collapsed_rest_by_each_factor_value():
    """A factor with a whole value, a fraction and 0 times a sum over two buffers:
    one list per buffer, and no table over factor and rest."""
    case = Case()
    case.add_input("k", k=[2.0, 0.5, 0.0])
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 7.0])
    k, a, b = case.symbol("k"), case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(k * (a**0.5 + b**0.25), bound=2.0)
    assert isinstance(plan.realization, SwitchRealization)
    assert case.elements() == [3, 3]


def test_two_products_with_one_factor_are_one_switch():
    case = Case()
    case.add_input("k", k=[1.0, 3.0, 4.0])
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 7.0])
    k, a, b = case.symbol("k"), case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(k * a**0.5 + k * b**0.25, bound=2.0)
    terms = plan.realization.terms
    assert sum(isinstance(t, SwitchRealization) for t in terms) == 1


def test_a_factor_with_repeated_values_is_ranked_once():
    """k takes 1, 2, 1, 2: both switches read its rank, one lookup of 4 between
    them, though k and k**2 are two nodes."""
    case = Case()
    case.add_input("k", k=[1.0, 2.0, 1.0, 2.0])
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 7.0])
    k, a, b = case.symbol("k"), case.symbol("a"), case.symbol("b")
    case.assert_exact(k * a**0.5 + k**2 * b**0.25, bound=3.0)
    assert case.elements() == [3, 3, 4]


@pytest.mark.parametrize("values", [[-1.5, 2.0, 0.5], [-1.0, -3.0]])
def test_a_factor_of_any_sign_switches(values):
    case = Case()
    case.add_input("k", k=values)
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 7.0])
    k, a, b = case.symbol("k"), case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(k * (a**0.5 + b**0.25), bound=3.0)
    assert isinstance(plan.realization, SwitchRealization)


def test_a_product_of_three_factors_is_a_switch_inside_a_switch():
    case = Case()
    case.add_input("j", j=[2.0, 3.0])
    case.add_input("k", k=[1.0, 0.5, 4.0])
    case.add_input("a", a=[1.0, 2.0, 4.0, 8.0])
    j, k, a = case.symbol("j"), case.symbol("k"), case.symbol("a")
    plan = case.assert_exact(j * k * a**0.5, cap=8, bound=3.0)
    assert isinstance(plan.realization, SwitchRealization)
    assert isinstance(plan.realization.rest, SwitchRealization)


def test_a_literal_times_a_factor_never_zero_is_a_two_value_switch():
    """1 + x is 1 or 2, so it is no gate: a switch whose true/falses are the literal
    and its negation, with no lookup of its own."""
    case = Case()
    case.literal("x")
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 7.0])
    x, a, b = case.symbol("x"), case.symbol("a"), case.symbol("b")
    plan = case.assert_exact((1 + x) * (a**0.5 + b**0.25), bound=2.0)
    assert isinstance(plan.realization, SwitchRealization)
    assert case.elements() == [3, 3]


def test_the_way_with_fewer_entries_wins():
    """k and a on one division: the product's one table is its 3 entries, against
    a switch over a rank and a list of 3. On two divisions the switch wins."""
    case = Case()
    case.add_input("d", k=[1.0, 1.0, 2.0], a=[1.0, 2.0, 4.0])
    k, a = case.symbol("k"), case.symbol("a")
    plan = case.assert_exact(k * a**0.5)
    assert isinstance(plan.realization, TableRealization)
    assert case.elements() == [3]
    case = Case()
    case.add_input("k", k=[1.0, 2.0, 3.0])
    case.add_input("a", a=[1.0, 2.0, 4.0, 8.0])
    k, a = case.symbol("k"), case.symbol("a")
    root = TableLowering(case.inputs, cap=CAP).lower(k * a**0.5)
    assert isinstance(root.realization, SwitchRealization)
    assert root.realization.entries < root.table.entries


def test_under_a_log_a_product_is_its_table():
    case = Case()
    case.add_input("k", k=[1.0, 2.0, 3.0])
    case.add_input("a", a=[1.0, 2.0, 4.0])
    k, a = case.symbol("k"), case.symbol("a")
    plan = case.assert_exact(sympy.log(k * a**0.5 + 1))
    assert isinstance(plan.realization, TableRealization)
    assert case.elements() == [9]


def test_a_factor_with_more_values_than_the_cap_has_no_switch():
    case = Case()
    case.add_input("k", k=[1.0, 2.0, 3.0, 5.0, 7.0])
    case.add_input("a", a=[1.0, 2.0, 4.0, 8.0, 16.0])
    k, a = case.symbol("k"), case.symbol("a")
    with pytest.raises(TableLoweringError, match="no factor can be switched") as caught:
        case.lower(k * a**0.5, cap=4)
    assert caught.value.node == k * a**0.5


def test_a_wide_rest_keeps_a_coarse_fraction_out_of_its_switch():
    """A factor of large values multiplies its rest's rounding: its magnitude goes
    into the rest's lists, and its switch reads values of at most 1."""
    case = Case()
    case.add_input("k", k=[3.0e6, 5.5e6, 7.25e6])
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 7.0])
    k, a, b = case.symbol("k"), case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(k * (a**0.5 + b**0.25), bound=1.5)
    assert max(abs(v) for v in plan.realization.values) <= 1


# -------------------------------------------------------------- the gate (Stage 7)


@pytest.mark.parametrize(
    "gate, kept",
    [
        (lambda x: x, 1),
        (lambda x: 1 - x, 0),
        (lambda x: 3 * x, 1),
        (lambda x: 2 - 2 * x, 0),
        (lambda x: 0.5 - 0.5 * x, 0),
    ],
    ids=["x", "1-x", "3x", "2-2x", "half"],
)
def test_a_gate_keeps_its_term_where_its_literal_is_kept(gate, kept):
    """One list of the menu's length on the division and no table over the literal;
    the gate's value is in the list, so a fraction adds no rounding."""
    case = Case()
    case.literal("x")
    case.add_input("a", a=[1.0, 2.0, 4.0])
    x, a = case.symbol("x"), case.symbol("a")
    plan = case.assert_exact(gate(x) * a**0.5)
    assert isinstance(plan.realization, GateRealization)
    assert plan.realization.kept == kept
    assert case.elements() == [3]


def test_gates_on_two_literals_keep_the_term_only_where_both_do():
    case = Case()
    case.literal("x")
    case.literal("y")
    case.add_input("a", a=[1.0, 2.0, 4.0])
    x, y, a = case.symbol("x"), case.symbol("y"), case.symbol("a")
    plan = case.assert_exact(x * (1 - y) * a**0.5)
    assert isinstance(plan.realization.rest, GateRealization)
    assert case.elements() == [3]


def test_terms_gated_by_one_literal_on_one_position_are_one_list():
    case = Case()
    case.literal("x")
    case.add_input("a", a=[1.0, 2.0, 4.0])
    x, a = case.symbol("x"), case.symbol("a")
    _, objective = case.lower(x * a**0.5 + x * a**0.7)
    assert case.elements() == [3]
    rows = case.solutions([*case.variables, objective])
    assert dict(((xv, av), cost) for xv, av, cost in rows)[(1, 1)] == 3112


def test_the_rest_of_a_gate_reads_no_literal():
    case = Case()
    case.literal("x")
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    x, a, b = case.symbol("x"), case.symbol("a"), case.symbol("b")
    lowering_ = TableLowering(case.inputs, cap=CAP)
    expr = x * sympy.Max(x * a, b)
    assert lowering_.normalize(expr) == x * sympy.Max(a, b)
    case.assert_exact(expr, bound=1.0)


def test_a_gate_with_only_constants_beside_it_is_affine():
    """(1 - x) * (c - c*x + c*x/t) is c * (1 - x): no gate, no table, and the tile's
    division is never read."""
    case = Case()
    case.literal("x")
    case.add_input("t", t=[1.0, 2.0, 4.0])
    x, t = case.symbol("x"), case.symbol("t")
    expr = (1 - x) * (3 - 3 * x + 3 * x / t)
    assert TableLowering(case.inputs, cap=CAP).normalize(expr) == 3 - 3 * x
    case.assert_exact(expr)
    assert case.elements() == []


def test_a_relayout_charge_is_its_prices_gated_by_its_literal():
    """Read linearly, the menu's prices once; under a power, one table over the
    literal and the division."""
    case = Case()
    case.literal("is_lx")
    position = case.integer("division_src", 0, 3)
    is_lx, division = case.symbol("is_lx"), case.symbol("division_src")
    charge = RelayoutCharge(is_lx, division, 0, 65, 147)
    plan = case.assert_exact(charge)
    assert isinstance(plan.realization, GateRealization)
    assert plan.realization.rest.index == position
    assert case.elements() == [3]
    plan = case.assert_exact(charge**0.5)
    assert plan.realization.index.size == 6


def test_a_gated_product_under_a_log_is_its_table():
    case = Case()
    case.literal("x")
    case.add_input("a", a=[1.0, 2.0, 4.0])
    x, a = case.symbol("x"), case.symbol("a")
    plan = case.assert_exact(sympy.log(x * a**0.5 + 1))
    assert isinstance(plan.realization, TableRealization)
    assert plan.realization.index.size == 6


# ---------------------------------------------------- ranks and pairs (Stage 8)


def test_equal_values_share_a_rank():
    case = Case()
    position = case.add_input("a", a=[8.0, 4.0, 4.0, 1.0])
    table = TableRealization(values=[8.0, 4.0, 4.0, 1.0], index=position)
    ranked = lowering._ranked(table)
    assert list(ranked.values) == [1.0, 4.0, 8.0]
    assert ranked.index == CompactIndex(position, (2, 1, 1, 0), 3)
    grid = lowering._grid([position])
    assert list(ranked.values[ranked.index.on_grid(grid)]) == list(table.values)
    distinct = TableRealization(values=[3.0, 1.0, 2.0, 5.0], index=position)
    assert lowering._ranked(distinct) is distinct


def test_a_product_under_a_log_is_exact_as_a_pair_of_ranks():
    """floor(a/4) and floor(b/3) take 4 values each on menus of 12 and 9: the
    product's table is the pair of their ranks, 16 entries, where one table over
    the divisions is 108; the sum and the log above it read that pair as it is, a
    function of one list adding no position."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 13)])
    case.add_input("b", b=[float(v) for v in range(1, 10)])
    a, b = case.symbol("a"), case.symbol("b")
    product = sympy.floor(a / 4) * sympy.floor(b / 3)
    plan = case.assert_exact(sympy.log(product + 1), cap=64)
    index = plan.realization.index
    assert isinstance(index, CombineIndex) and index.size == 16
    assert isinstance(index.left, CompactIndex) and isinstance(index.right, CompactIndex)
    assert case.elements() == [9, 12, 16]
    with pytest.raises(TableLoweringError, match="past the cap of 15"):
        Case.lower(case, sympy.log(product + 1), cap=15)


# ------------------------------------------------ a Min/Max by rung (Stage 9)


def test_a_ladder_holds_what_a_min_or_max_can_return():
    """A Min never exceeds its smallest operand maximum: values past it are clamped
    to it. A Max mirrors that with its largest operand minimum."""
    operands = [np.array([1.0, 5.0, 9.0]), np.array([3.0, 4.0])]
    assert clamped_ladder(operands, is_min=True) == ([1.0, 3.0, 4.0], 4.0)
    assert clamped_ladder(operands, is_min=False) == ([3.0, 4.0, 5.0, 9.0], 3.0)


def test_values_equal_to_twelve_digits_share_a_rung():
    ladder = [1.0, 2.0, 4.0]
    assert rungs_on(ladder, 4.0, np.array([1.0 + 1e-14, 2.0, 9.0]), True) == (0, 1, 2)


def test_the_min_of_rungs_is_the_rung_of_the_min():
    operands = [np.array([1.0, 5.0, 9.0, 2.5]), np.array([3.0, 4.0, 0.5])]
    for is_min, pick in ((True, min), (False, max)):
        ladder, bound = clamped_ladder(operands, is_min)
        rungs = [rungs_on(ladder, bound, o, is_min) for o in operands]
        for i, j in itertools.product(range(4), range(3)):
            rung = pick(rungs[0][i], rungs[1][j])
            assert ladder[rung] == pick(operands[0][i], operands[1][j])


def test_a_max_over_three_buffers_is_read_by_rung():
    """Under a log the Max is a table: its ladder, each operand remapped to its
    rungs, far smaller than one table over the three divisions."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 7)])
    case.add_input("b", b=[float(v) for v in range(1, 6)])
    case.add_input("c", c=[float(v) for v in range(1, 5)])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    extreme = sympy.Max(a, 1.5 * b, 2 * c + 0.25)
    plan = case.assert_exact(sympy.log(extreme))
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(sympy.log(extreme))
    assert isinstance(nodes[extreme].table.index, ExtremeIndex)
    assert max(case.elements()) < 6 * 5 * 4
    assert plan.realization.index == nodes[extreme].table.index


def test_the_graph_of_the_plan_is_exact():
    """"The process, step by step": four terms over three buffers, split_c with a
    repeated value, at all 24 assignments."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    case.add_input("c", c=[1.0, 2.0, 2.0, 4.0])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    expr = (
        7 * a**0.5
        + a**0.7
        + 9 * a**-2 * b**0.5
        + sympy.Max(a / 2, b**0.5, c**0.25)
    )
    case.assert_exact(expr, bound=2.5)


# ------------------------------------------------------------- the fold (Stage 10)


def _positions(*sizes: int) -> list[InputIndex]:
    """One input position per size, on variables of one model."""
    model = cp_model.CpModel()
    return [
        InputIndex(model.new_int_var(0, size - 1, f"p{i}"), size=size)
        for i, size in enumerate(sizes)
    ]


def _max(ladder: list[float], *operands: tuple[Index, tuple[int, ...]]) -> TableRealization:
    """A Max by rung: its ladder on the max of its operands' rungs, each a position
    and its rung at each of the position's values."""
    rungs = tuple(CompactIndex(p, remap, len(ladder)) for p, remap in operands)
    return TableRealization(values=ladder, index=ExtremeIndex(rungs, False, len(ladder)))


def test_a_fold_numbers_only_the_tuples_that_occur_together():
    """Two Max factors over one input: 3 tuples of rungs occur, of 9."""
    (p,) = _positions(3)
    factors = [
        _max([1.0, 2.0, 3.0], (p, (0, 1, 2))),
        _max([1.0, 2.0, 3.0], (p, (2, 1, 0))),
    ]
    fold = fold_reachable(factors, cap=100)
    assert fold is not None and fold.index.size == 3
    assert sorted(fold.values) == [3.0, 3.0, 4.0]
    assert fold_reachable(factors, cap=2) is None


def test_a_constant_operand_seeds_the_start_state():
    (p,) = _positions(3)
    factors = [
        _max([1.0, 2.0, 3.0], (UNIT, (1,)), (p, (0, 1, 2))),
        _max([5.0, 6.0], (p, (0, 0, 1))),
    ]
    fold = fold_reachable(factors, cap=100)
    assert fold is not None and sorted(fold.values) == [10.0, 18.0]


def test_operands_sharing_an_input_join_one_group():
    """Max(f(p, q), g(r)) * Max(h(q)): f and h share q, so p and q are read in one
    step of 4, and r in another."""
    p, q, r = _positions(2, 2, 3)
    factors = [
        _max([0.0, 1.0], (CombineIndex(p, q), (0, 1, 1, 0)), (r, (0, 0, 1))),
        _max([0.0, 1.0], (q, (1, 0))),
    ]
    fold = fold_reachable(factors, cap=100)
    assert fold is not None
    last = fold.index.parent  # the state before the last step, combined with its group
    first = last.left.parent
    groups = sorted((first.right, last.right), key=lambda group: group.size)
    assert groups[0] == r and {p, q} <= set(groups[1].steps()) and groups[1].size == 4


def test_a_product_of_max_factors_is_folded_under_a_cap_their_crossed_ladders_pass():
    """Four Max factors over three buffers under a log: one table over the three
    divisions and the crossed ladders both pass the cap; the fold fits."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 3.0, 4.0, 5.0])
    case.add_input("b", b=[1.5, 2.5, 3.5, 4.5, 5.5])
    case.add_input("c", c=[1.25, 2.25, 3.25, 4.25, 5.25])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    product = (
        sympy.Max(a, b, 3) * sympy.Max(b, c, 3) * sympy.Max(a, c, 3) * sympy.Max(a, 4)
    )
    nodes = TableLowering(case.inputs, cap=100).rewrite(product)
    ladders = [nodes[f].ladder for f in product.args if isinstance(f, sympy.Max)]
    assert math.prod(ladder.index.size for ladder in ladders) > 100
    plan = case.assert_exact(sympy.log(product), cap=100)
    assert isinstance(plan.realization.index, CompactIndex)


# ------------------------------------------------------ a Min kept linear (Stage 11)


def test_a_min_too_wide_for_a_table_stays_linear():
    case = Case()
    x = _literals(case, 6)
    expr = sympy.Min(
        100 - sum((20 + j) * x[j] for j in range(6)),
        90 - sum((10 + j) * x[j] for j in range(5)),
    )
    plan = case.assert_exact(expr, cap=16, bound=1.0)
    assert isinstance(plan.realization, ExtremeRealization)
    assert case.elements() == []
    (lin_max,) = case.lin_maxes()
    assert all(len(e.vars) <= 1 for e in lin_max.exprs)  # one variable each
    with pytest.raises(TableLoweringError, match="past the cap of 16"):
        Case.lower(case, sympy.log(expr + 200), cap=16)


def test_a_near_tie_keeps_the_table():
    """By rung, a Max of two lists is about twice its lin_max's entries: not under
    half, so it is a table."""
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0, 8.0])
    case.add_input("b", b=[1.5, 3.0, 7.0])
    a, b = case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(sympy.Max(a**0.5, b**0.25))
    assert isinstance(plan.realization, TableRealization)


@pytest.mark.parametrize("coefficient", [-2, 0.3, -0.7])
def test_a_coefficient_on_a_min_kept_linear(coefficient):
    """Applied to the operands' lists before they are collapsed; a negative one
    turns the Min into a Max."""
    case = Case()
    x = _literals(case, 4)
    inner = sympy.Min(10 - 3 * x[0] - 5 * x[1], 9 - 4 * x[2] - 2 * x[3])
    expr = sympy.Mul(coefficient, inner, evaluate=False)
    plan = case.assert_exact(expr, cap=4, bound=1.5)
    assert isinstance(plan.realization, ExtremeRealization)
    assert plan.realization.is_min == (coefficient > 0)


@pytest.mark.parametrize("values", [[-1.5, 2.0, 0.5], [-1.0, -3.0]])
def test_a_factor_of_any_sign_times_a_min_kept_linear(values):
    case = Case()
    case.add_input("k", k=values)
    x = _literals(case, 4)
    k = case.symbol("k")
    expr = k * sympy.Min(10 - 3 * x[0] - 5 * x[1], 9 - 4 * x[2] - 2 * x[3])
    plan = case.assert_exact(expr, cap=4, bound=3.0)
    assert isinstance(plan.realization, SwitchRealization)
    assert isinstance(plan.realization.rest, ExtremeRealization)


def test_a_max_too_wide_to_index_stays_linear_under_linear_readers():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    x = _literals(case, 4)
    a = case.symbol("a")
    wide = sympy.Max(3 * x[0] + 5 * x[1] + a, 4 * x[2] + 2 * x[3])
    for expr in (wide + a**0.5, 3 * wide, sympy.Min(wide, 7)):
        Case.lower(case, expr, cap=4)
    case.assert_exact(wide + a**0.5, cap=4, bound=2.0)
    with pytest.raises(TableLoweringError, match="past the cap of 4"):
        Case.lower(case, 1 / wide, cap=4)


# --------------------------------------------- integration with the maps (Stage 13)


def _maps(model: cp_model.CpModel) -> tuple[dict, dict]:
    """The printer's two maps for a buffer a of three divisions, its residency, a
    variable with gaps, a domain past the cap and a number."""
    division = model.new_int_var(0, 2, "div_a")
    wrapper = SimpleNamespace(division=division)
    sym_map = {
        "is_lx_a": model.new_bool_var("in_buffer_a"),
        "division_a": division,
        "_division_of_division_a": wrapper,
        "split_a": model.new_int_var(1, 32, "split_a"),
        "gapped": model.new_int_var_from_domain(cp_model.Domain.from_values([1, 2, 4]), "g"),
        "wide": model.new_int_var(0, 10**6, "wide"),
        "number": 3,
    }
    buffer_map = {"split_a": (wrapper, [1, 2, 4]), "tile_a": (wrapper, [1, 1, 2])}
    return sym_map, buffer_map


def test_the_inputs_come_from_the_printers_maps():
    model = cp_model.CpModel()
    sym_map, buffer_map = _maps(model)
    inputs = inputs_from_maps(sym_map, buffer_map, cap=CAP)
    assert set(inputs) == {"split_a", "tile_a", "is_lx_a", "division_a", "gapped"}
    assert inputs["split_a"] == (sym_map["division_a"], [1.0, 2.0, 4.0])
    assert inputs["is_lx_a"][1] == [0.0, 1.0]
    assert inputs["gapped"][1] == [1.0, 2.0, 4.0]
    buffer_map["tile_b"] = (buffer_map["split_a"][0], [1, 2])
    with pytest.raises(TableLoweringError, match="two lengths on one division"):
        inputs_from_maps(sym_map, buffer_map, cap=CAP)


def test_a_variable_with_gaps_in_its_domain_gets_a_position():
    case = Case()
    var = case.model.new_int_var_from_domain(cp_model.Domain.from_values([1, 2, 4]), "g")
    g = sympy.Symbol("g", integer=True)
    case.inputs[g] = (var, [1, 2, 4])
    case.variables.append(var)
    case.values[var] = [1, 2, 4]
    plan = case.assert_exact(sympy.log(g + 1))
    assert isinstance(plan.realization.index, LocateIndex)
    assert case.elements() == [3, 3]


def _objective(model: cp_model.CpModel, objective) -> float:
    model.minimize(objective)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    return solver.objective_value


def test_lower_cost_with_tables_reads_the_maps():
    """A number in sym_map is the symbol's value, and a relayout of a source this
    solve does not divide costs 0."""
    model = cp_model.CpModel()
    sym_map, buffer_map = _maps(model)
    split_a, tile_a, number, is_lx = sympy.symbols("split_a tile_a number is_lx_a")
    elsewhere = sympy.Symbol("division_b")
    expr = number * split_a**0.5 / tile_a + RelayoutCharge(is_lx, elsewhere, 50, 60)
    objective = lower_cost_with_tables(model, sym_map, buffer_map, expr, cap=CAP)
    assert _objective(model, objective) == round(COST_SCALE * 3 * 4**0.5 / 2)


def test_a_failure_leaves_the_model_as_it_was():
    """Refused in realizing, and refused in the build after variables were made:
    either way the model is its copy, and a later lowering into it solves."""
    model = cp_model.CpModel()
    sym_map, buffer_map = _maps(model)
    split_a, tile_a, is_lx = sympy.symbols("split_a tile_a is_lx_a")
    before = str(model.proto)
    for refused in (split_a + sympy.Symbol("nowhere"), split_a**0.5 + 1e14 * tile_a**0.5):
        with pytest.raises(TableLoweringError):
            lower_cost_with_tables(model, sym_map, buffer_map, refused, cap=CAP)
        assert str(model.proto) == before
    objective = lower_cost_with_tables(
        model, sym_map, buffer_map, is_lx * split_a**0.5 + 7, cap=CAP
    )
    assert _objective(model, objective) == 7 * COST_SCALE


def test_another_exception_leaves_as_a_table_lowering_error(monkeypatch):
    model = cp_model.CpModel()
    sym_map, buffer_map = _maps(model)
    split_a = sympy.Symbol("split_a")

    def broken(self, root):
        raise ValueError("broken")

    monkeypatch.setattr(TableLowering, "normalize", broken)
    with pytest.raises(TableLoweringError, match="ValueError: broken") as caught:
        lower_cost_with_tables(model, sym_map, buffer_map, split_a, cap=CAP)
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.node is None and caught.value.reason == "ValueError: broken"


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


# One example per optimization: an objective whose one table is past its cap, which no
# stage before that one lowers and that stage's optimization does. Each was marked to
# fail until its stage; every one lowers now. The first lowered from Stage 1 on.
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
    ),
    pytest.param(
        lambda: _with(
            lambda s: sympy.log(
                1 + sympy.floor(s["a"] / 16) * sympy.floor(s["b"] / 32)
            ),
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
    ),
    pytest.param(
        lambda: _with(
            lambda s: sympy.log(
                sympy.Max(s["a"], s["b"], 3)
                * sympy.Max(s["b"], s["c"], 3)
                * sympy.Max(s["a"], s["c"], 3)
                * sympy.Max(s["a"], 4)
            ),
            a=5,
            b=5,
            c=5,
        ),
        100,
        id="fold",
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
def test_each_optimization_lowers_its_example_exactly(example, cap):
    """Checked at two assignments, each variable at its lowest value and at its
    highest: minimizing and maximizing agree and equal COST_SCALE times sympy's value,
    within half a unit per constraint the lowering adds, plus half."""
    case, expr = example()
    before = len(case.model.proto.constraints)
    _, objective = case.lower(expr, cap)
    bound = (len(case.model.proto.constraints) - before + 1) / 2
    domains = [case.values[v] for v in case.variables]
    for assignment in (tuple(d[0] for d in domains), tuple(d[-1] for d in domains)):
        low, high = _solve_at(case, objective, assignment)
        assert low == high
        assert abs(low - COST_SCALE * case.sympy_value(expr, assignment)) <= bound
