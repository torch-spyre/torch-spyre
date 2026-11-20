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
"""The numpy core of a cost expression (cost_expr_numpy.py), and the CP-SAT printer's
candidate evaluation through it. No device and no compile. The tests of the printer
need ortools and skip without it.
"""

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sympy
from sympy.logic.boolalg import Boolean

from torch_spyre._inductor.scratchpad import cpsat_expr_printer
from torch_spyre._inductor.scratchpad.cost_expr_numpy import NUMPY_FUNC_MAP, numpy_eval
from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

x, y = sympy.symbols("split_x split_y", integer=True, positive=True)
# One division's menu: its two splits move together, as a buffer's candidates do.
MENU = {x: [1, 2, 4, 8], y: [8, 2, 4, 1]}


def _value(expr: sympy.Basic, env: dict[sympy.Symbol, Any]) -> Any:
    """expr's value where each symbol has its value in env: the core applied to each
    node's operands' values, from the leaves up, as a caller of the core does."""
    if expr in env:
        return env[expr]
    if expr is sympy.true or expr is sympy.false:
        return bool(expr)
    if expr.is_Number:
        return int(expr) if expr.is_Integer else float(expr)
    operands = (
        [part for pair in expr.args for part in pair.args]
        if expr.is_Piecewise
        else expr.args
    )
    return numpy_eval(expr, [_value(operand, env) for operand in operands])


def _sympy_value(expr: sympy.Basic, point: dict[sympy.Symbol, int]) -> Any:
    """expr at point as sympy computes it: a truth value, an int or a float."""
    value = expr.xreplace({s: sympy.Integer(v) for s, v in point.items()})
    if value is sympy.true or value is sympy.false:
        return bool(value)
    return int(value) if value.is_Integer else float(value)


EXPRESSIONS = [
    x + y + 3,
    2 * x * y,
    x**2 + y**3,
    sympy.Min(x, y, 3),
    sympy.Max(x, y, 3),
    sympy.Eq(x, y),
    sympy.Ne(x, y),
    x < y,
    x <= y,
    x > y,
    x >= y,
    sympy.And(x >= 2, y >= 2),
    sympy.Or(x <= 1, y <= 1),
    sympy.KroneckerDelta(x, y),
    sympy.Piecewise(
        (x * y, sympy.And(x >= 2, y >= 2)),
        (x**2, sympy.Or(x <= 1, y <= 1)),
        (3, True),
    ),
    sympy.Max(x * y, 5) + sympy.Piecewise((2 * x, x > y), (y - x, True)),
]


# Expressions of the classes whose functions take floats: the table lowering's, which
# the printer leaves to its print methods.
FLOAT_EXPRESSIONS = [
    sympy.Abs(x - y),
    sympy.floor(x / 4) + sympy.ceiling(y / 4),
    sympy.log(x + y) * sympy.exp(-x / 8),
    sympy.Piecewise((x, sympy.Not(sympy.And(x > 2, y > 2))), (y, True)),
    sympy.Piecewise((x, sympy.ITE(x > 2, y > 2, y < 2)), (y, True)),
]


@pytest.mark.parametrize("expr", FLOAT_EXPRESSIONS, ids=str)
def test_a_node_over_floats_is_sympys_value_at_every_entry(expr):
    """Over a menu held as a float array, as the table lowering holds a list."""
    env = {s: np.asarray(menu, dtype=float) for s, menu in MENU.items()}
    got = np.broadcast_to(_value(expr, env), (4,))
    for i in range(4):
        exact = _sympy_value(expr, {s: menu[i] for s, menu in MENU.items()})
        assert got[i] == pytest.approx(exact, rel=1e-12), i


def test_a_relayout_charge_is_its_literal_times_the_price_at_its_division():
    """At every literal and division, the value the class's own ``_imp_`` gives: 0
    where the literal is 0 and at a division past the prices. The prices are numbers,
    or lists of one length with the operands, as the table lowering hands them."""
    is_lx = np.repeat([0.0, 1.0], 5)
    division = np.tile([0.0, 1.0, 2.0, 3.0, 4.0], 2)
    prices = (0.0, 65.0, 147.5)
    expr = RelayoutCharge(sympy.Symbol("is_lx"), sympy.Symbol("division"), *prices)
    expected = [RelayoutCharge._imp_(*point, *prices) for point in zip(is_lx, division)]
    assert list(numpy_eval(expr, [is_lx, division, *prices])) == expected
    lists = [np.full(10, price) for price in prices]
    assert list(numpy_eval(expr, [is_lx, division, *lists])) == expected
    unpriced = RelayoutCharge(sympy.Symbol("is_lx"), sympy.Symbol("division"))
    assert list(numpy_eval(unpriced, [is_lx, division])) == [0.0] * 10


@pytest.mark.parametrize("expr", EXPRESSIONS, ids=str)
def test_a_node_is_sympys_value_at_every_entry(expr):
    """Over a menu held as an object array, as the printer holds its candidates, each
    entry is the value sympy computes there, exactly: an int stays an int."""
    env = {s: np.asarray(menu, dtype=object) for s, menu in MENU.items()}
    got = np.broadcast_to(_value(expr, env), (4,))
    for i in range(4):
        exact = _sympy_value(expr, {s: menu[i] for s, menu in MENU.items()})
        assert got[i] == exact and not isinstance(got[i], (float, np.floating)), i


def test_every_function_is_reached():
    """The expressions above, and the relayout charge, reach every class the core has
    a function for."""
    reached = {
        cls
        for expr in EXPRESSIONS + FLOAT_EXPRESSIONS
        for node in sympy.preorder_traversal(expr)
        for cls in type(node).__mro__
    }
    assert set(NUMPY_FUNC_MAP) - reached == {RelayoutCharge}


def test_a_plain_int_stays_exact_at_any_size():
    """Add, Mul and Pow are Python's own, so numbers past int64 are exact, alone and
    in an object array. A sum of floats is Python's sum and a product math.prod, the
    values the printer computed before the core."""
    big = 2**62
    assert numpy_eval(x + y, [big, big]) == 2**63
    assert numpy_eval(x * y, [big, big]) == 2**124
    assert numpy_eval(x**y, [3, 50]) == 3**50
    column = np.asarray([big, 3], dtype=object)
    assert list(numpy_eval(x * y, [column, column])) == [2**124, 9]
    floats = [0.1, 0.2, 0.3]
    assert numpy_eval(x + y + 1, floats) == sum(floats)
    assert numpy_eval(x * y * 2, floats) == math.prod(floats)


def test_a_number_stands_at_every_entry():
    column = np.asarray([1, 5, 9], dtype=object)
    assert list(numpy_eval(sympy.Max(x, 3), [column, 3])) == [3, 5, 9]
    assert list(numpy_eval(x + 3, [3, column])) == [4, 8, 12]
    assert list(numpy_eval(x < 3, [column, 5])) == [True, False, False]


def test_a_piecewise_takes_the_first_arm_that_holds():
    """Where two conditions hold, the earlier arm wins. A default arm is taken as it
    is, so its ints stay ints; with no default, an entry no arm holds at is NaN."""
    expr = sympy.Piecewise((x, x > 2), (y, x > 1), (0, True))
    first = np.asarray([10, 20, 30, 40], dtype=object)
    second = np.asarray([1, 2, 3, 4], dtype=object)
    holds = np.asarray([True, False, True, False])
    also = np.asarray([True, True, False, False])
    got = numpy_eval(expr, [first, holds, second, also, 7, True])
    assert list(got) == [10, 2, 30, 7]
    assert all(type(value) is int for value in got)
    open_ended = numpy_eval(expr, [first, holds, second, also])
    assert list(open_ended[:3]) == [10, 2, 30] and math.isnan(open_ended[3])


def test_a_subclass_is_evaluated_as_its_class():
    """The function is that of the nearest class along the MRO: the printer's lazy
    Min and Max are subclasses of sympy's."""
    lazy = cpsat_expr_printer._LazyMin(x, y)
    assert type(lazy) is not sympy.Min
    column = np.asarray([1, 5, 9], dtype=object)
    assert list(numpy_eval(lazy, [column, 4])) == [1, 4, 4]


def test_a_class_with_no_function_raises():
    with pytest.raises(NotImplementedError, match="no numpy function for sinc"):
        numpy_eval(sympy.sinc(x), [np.asarray([1.0, 2.0])])


# ------------------------------------------------------------------- the printer


def _printer(model: Any) -> tuple[Any, list[Any]]:
    """A printer over one division of four candidates that both splits follow."""
    division = model.new_int_var(0, 3, "division")
    wrapper = SimpleNamespace(division=division)
    sym_map: dict[str, Any] = {"resident": model.new_bool_var("resident")}
    buffer_map = {}
    for symbol, menu in MENU.items():
        var = model.new_int_var(min(menu), max(menu), symbol.name)
        model.add_element(division, menu, var)
        sym_map[symbol.name] = var
        buffer_map[symbol.name] = (wrapper, menu)
    printer = cpsat_expr_printer._SympyExprToCpSat(model, sym_map, buffer_map)
    return printer, [wrapper]


def test_the_printer_evaluates_its_candidates_through_the_core(monkeypatch):
    """The printer's candidate values of an expression over one division are the
    core's, node by node, and sympy's at every candidate."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    evaluated: list[sympy.Basic] = []

    def counting(expr, values):
        evaluated.append(expr)
        return numpy_eval(expr, values)

    monkeypatch.setattr(cpsat_expr_printer, "numpy_eval", counting)
    printer, (wrapper,) = _printer(cp_model.CpModel())
    for expr in EXPRESSIONS:
        evaluated.clear()
        owner, values = printer._candidate_values(expr)
        assert owner is wrapper
        assert expr in evaluated
        values = np.broadcast_to(values, (4,))
        for i in range(4):
            exact = _sympy_value(expr, {s: menu[i] for s, menu in MENU.items()})
            assert values[i] == exact, (expr, i)


def test_the_printer_evaluates_only_the_classes_it_did():
    """The core has functions the printer does not use. A node of a class outside the
    printer's own list, here Abs, has no candidate values, and neither has a node
    above it, so the printer lowers them as it did before the core. A free decision,
    here a residency literal, stops the evaluation the same way."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    printer, _ = _printer(cp_model.CpModel())
    resident = sympy.Symbol("resident", integer=True)
    assert printer._candidate_values(sympy.Abs(x - 3)) is None
    assert printer._candidate_values(sympy.Abs(x - 3) + y) is None
    assert printer._candidate_values(x * resident) is None
    assert printer._candidate_values(x * y) is not None


def test_the_printer_lowers_the_same_costs_through_the_core():
    """Each expression that is a cost, not a condition, lowered by the printer has,
    at every candidate of the division, the value sympy computes there."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    model = cp_model.CpModel()
    printer, (wrapper,) = _printer(model)
    exprs = [expr for expr in EXPRESSIONS if not isinstance(expr, Boolean)]
    costs = [printer.convert(expr) for expr in exprs]
    for i in range(4):
        fixed = model.clone()
        fixed.add(wrapper.division == i)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(fixed) == cp_model.OPTIMAL
        point = {s: menu[i] for s, menu in MENU.items()}
        for expr, cost in zip(exprs, costs):
            assert solver.float_value(cost) == pytest.approx(_sympy_value(expr, point))
