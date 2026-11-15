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

"""The table lowering of a cost expression, checked against the expression
evaluated at every assignment of its inputs."""

import itertools
import types

import pytest
import sympy
import torch  # noqa: F401

cp_model = pytest.importorskip("ortools.sat.python.cp_model")

from torch_spyre._inductor.scratchpad.cpsat_table_lowering import (  # noqa: E402
    COST_SCALE,
    FoldFactor,
    clamped_ladder,
    cost_expr_to_cpsat,
    fold_reachable,
    lower_cost_with_tables,
    rungs_on,
)


def _buffers(menus):
    """A model with one division per buffer; ``menus`` maps a buffer name to
    ``{symbol: value at each division}``."""
    model = cp_model.CpModel()
    buffer_map, divisions, symbols = {}, {}, {}
    for buffer, columns in menus.items():
        size = len(next(iter(columns.values())))
        division = model.new_int_var(0, size - 1, f"division_{buffer}")
        divisions[buffer] = division
        for name, raw in columns.items():
            buffer_map[name] = (types.SimpleNamespace(division=division), raw)
            symbols[name] = sympy.Symbol(name, positive=True)
    return model, buffer_map, divisions, symbols


def _element_sizes(model, first):
    return [
        len(c.element.exprs)
        for c in list(model.proto.constraints)[first:]
        if c.has_element()
    ]


def _assert_exact(model, sym_map, buffer_map, expr, choices, lowered):
    """``lowered`` is ``COST_SCALE * expr`` at every assignment of ``choices``
    (a CP-SAT variable -> its values), within one rounding per table."""
    tables = len(_element_sizes(model, 0)) + 1
    names = {s.name: s for s in expr.free_symbols}
    for assignment in itertools.product(*choices.values()):
        fixed = model.clone()
        values = {}
        for var, value in zip(choices, assignment):
            fixed.add(var == value)
            values[var.index] = value
        fixed.minimize(lowered)
        solver = cp_model.CpSolver()
        assert solver.solve(fixed) == cp_model.OPTIMAL
        subs = {}
        for name, symbol in names.items():
            if name in buffer_map:
                owner, raw = buffer_map[name]
                subs[symbol] = raw[values[owner.division.index]]
            else:
                subs[symbol] = values[sym_map[name].index]
        want = float(expr.xreplace(subs))
        assert solver.objective_value == pytest.approx(COST_SCALE * want, abs=tables), (
            assignment
        )


def test_clamped_ladder_keeps_only_what_a_min_can_return():
    ladder, bound = clamped_ladder([[1, 5, 9], [3, 4]], is_min=True)
    assert bound == 4
    assert ladder == [1, 3, 4]
    assert rungs_on(ladder, bound, [1, 5, 9], is_min=True) == [0, 2, 2]


def test_clamped_ladder_mirrors_for_a_max():
    ladder, bound = clamped_ladder([[1, 5, 9], [3, 4]], is_min=False)
    assert bound == 3
    assert ladder == [3, 4, 5, 9]
    assert rungs_on(ladder, bound, [3, 4], is_min=False) == [0, 1]


def test_fold_numbers_only_the_tuples_that_occur_together():
    # Two Max factors of the same per-buffer value p over buffers a and b, one
    # its square root: both follow max(p_a, p_b), so the reachable tuples are
    # the distinct values of that max, not the cross product of the ladders.
    p = [1.0, 4.0, 9.0, 16.0]
    ladder, bound = clamped_ladder([p, p], is_min=False)
    rungs = tuple(rungs_on(ladder, bound, p, is_min=False))
    first = FoldFactor(False, ((("a",), rungs), (("b",), rungs)), tuple(ladder))
    root = tuple(v**0.5 for v in ladder)
    second = FoldFactor(False, ((("a",), rungs), (("b",), rungs)), root)
    steps, states = fold_reachable([first, second], {"a": 4, "b": 4}, cap=64)
    assert [s.inputs for s in steps] == [("a",), ("b",)]
    reached = {
        (max(rungs[i], rungs[j]),) * 2 for i, j in itertools.product(range(4), repeat=2)
    }
    assert set(states) == reached
    assert len(states) == 4


def test_fold_refuses_a_step_past_the_cap():
    rungs = (0, 1, 2, 3)
    factor = FoldFactor(True, ((("a",), rungs), (("b",), rungs)), (0.0, 1.0, 2.0, 3.0))
    assert fold_reachable([factor], {"a": 4, "b": 4}, cap=15) is None


def test_a_sum_of_buffers_stays_linear():
    model, buffer_map, divisions, s = _buffers(
        {"a": {"split_a": [1, 2, 4, 8]}, "b": {"split_b": [1, 3, 9]}}
    )
    expr = 100 / s["split_a"] + 7 * s["split_b"] ** 0.5 + 3
    lowered = lower_cost_with_tables(model, {}, buffer_map, expr, cap=8192)
    # Each term is a function of one division; no table spans both.
    assert max(_element_sizes(model, 0), default=0) <= 4
    choices = {divisions["a"]: range(4), divisions["b"]: range(3)}
    _assert_exact(model, {}, buffer_map, expr, choices, lowered)


def test_a_min_max_is_indexed_by_its_ladder_rung():
    menus = {b: {f"split_{b}": [1, 2, 4, 8, 16, 32]} for b in "abc"}
    model, buffer_map, divisions, s = _buffers(menus)
    derate = sympy.Max(1, *(s[f"split_{b}"] ** 0.5 / 2 for b in "abc"))
    expr = 1000 * derate
    lowered = lower_cost_with_tables(model, {}, buffer_map, expr, cap=8192)
    # The joint table over three divisions would hold 216 entries; the rank
    # reads three 6-entry rungs and a ladder of 5 values.
    assert max(_element_sizes(model, 0)) < 216
    choices = {divisions[b]: range(6) for b in "abc"}
    _assert_exact(model, {}, buffer_map, expr, choices, lowered)


def test_a_min_of_literal_sums_stays_linear_in_the_objective():
    model, buffer_map, divisions, s = _buffers({"a": {"tile_a": list(range(1, 41))}})
    names = [f"is_lx_{i}" for i in range(5)]
    sym_map = {n: model.new_bool_var(n) for n in names}
    lits = [sympy.Symbol(n, integer=True) for n in names]
    reads = 70 - 3 * lits[0] - 5 * lits[1] + 2 * (1 - lits[2]) * (8 - 4 / s["tile_a"])
    writes = 90 - 7 * lits[3] - 2 * lits[4] - 4 * lits[2]
    expr = sympy.Rational(3, 7) * sympy.Min(reads, writes)
    lowered = lower_cost_with_tables(model, sym_map, buffer_map, expr, cap=8192)
    # One table over the Min's inputs would hold 2**5 * 40 entries; kept
    # linear, the literal terms cost none and the tile term 2 * 40.
    assert max(_element_sizes(model, 0)) <= 80
    choices = {divisions["a"]: (0, 1, 39), **{sym_map[n]: (0, 1) for n in names}}
    _assert_exact(model, sym_map, buffer_map, expr, choices, lowered)


def test_a_product_of_four_maxes_over_shared_buffers_folds():
    menus = {
        b: {f"split_{b}": [1, 2, 4, 8, 16, 32], f"tile_{b}": [1, 2, 1, 4, 2, 1]}
        for b in "abc"
    }
    model, buffer_map, divisions, s = _buffers(menus)
    per_buffer = [s[f"split_{b}"] * s[f"tile_{b}"] for b in "abc"]
    exponents = (0.15, 0.5, 0.25, 0.35)
    factors = [sympy.Max(1.0, *(q**e for q in per_buffer)) for e in exponents]
    expr = 1000 * sympy.Mul(*factors)
    # Crossing four ladders passes a cap of 200; the four follow one value,
    # the largest split * tile, so the fold reaches few states.
    lowered = lower_cost_with_tables(model, {}, buffer_map, expr, cap=200)
    assert lowered is not None
    assert max(_element_sizes(model, 0)) <= 200
    choices = {divisions[b]: range(6) for b in "abc"}
    _assert_exact(model, {}, buffer_map, expr, choices, lowered)


def test_a_negative_factor_turns_a_wide_min_into_a_max():
    model, buffer_map, divisions, s = _buffers({"a": {"split_a": [1, 2, 4, 8]}})
    x, y = sympy.symbols("x y", integer=True, nonnegative=True)
    sym_map = {"x": model.new_int_var(0, 50, "x"), "y": model.new_int_var(0, 50, "y")}
    derate = sympy.Max(1, 4 / s["split_a"] ** 0.5)
    expr = 1000 - 2 * sympy.Min(10 * x + 7, 6 * y + 20) * derate
    # At this cap each operand times the derate fits (51 * 4 entries), the
    # Min's 83-rung ladder times the derate does not, so the derate is pushed
    # into the Min's operands.
    lowered = lower_cost_with_tables(model, sym_map, buffer_map, expr, cap=250)
    assert lowered is not None
    assert any(c.has_lin_max() for c in model.proto.constraints)
    choices = {
        divisions["a"]: range(4),
        sym_map["x"]: (0, 1, 5, 50),
        sym_map["y"]: (0, 3, 50),
    }
    _assert_exact(model, sym_map, buffer_map, expr, choices, lowered)


def test_a_residency_gated_relayout_charge_is_exact():
    from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

    model, buffer_map, divisions, s = _buffers({"a": {"split_a": [1, 2, 4]}})
    lit = model.new_bool_var("is_lx_a")
    sym_map = {
        "is_lx_a": lit,
        "division_a": divisions["a"],
        "_division_of_division_a": types.SimpleNamespace(division=divisions["a"]),
    }
    is_lx = sympy.Symbol("is_lx_a", integer=True)
    division = sympy.Symbol("division_a", integer=True)
    expr = RelayoutCharge(is_lx, division, 5, 7) + 3 * (1 - is_lx) * s["split_a"]
    lowered = lower_cost_with_tables(model, sym_map, buffer_map, expr, cap=8192)
    choices = {divisions["a"]: range(3), lit: (0, 1)}
    _assert_exact(model, sym_map, buffer_map, expr, choices, lowered)


def test_an_expression_it_cannot_table_leaves_the_model_unchanged():
    model = cp_model.CpModel()
    names = "wxyz"
    sym_map = {n: model.new_int_var(0, 100, n) for n in names}
    w, x, y, z = sympy.symbols(" ".join(names), integer=True)
    before = (len(model.proto.variables), len(model.proto.constraints))
    # Two wide sums multiplied: no factor is short enough to push in.
    lowered = lower_cost_with_tables(model, sym_map, {}, (w + x) * (y + z), cap=50)
    assert lowered is None
    assert (len(model.proto.variables), len(model.proto.constraints)) == before


def test_an_expression_it_cannot_table_goes_to_the_printer():
    model = cp_model.CpModel()
    names = "w x y z"
    sym_map = {n: model.new_int_var(0, 100, n) for n in names.split()}
    w, x, y, z = sympy.symbols(names, integer=True)
    lowered = cost_expr_to_cpsat(model, sym_map, {}, (w + x) * (y + z))
    # The printer multiplies the two sums; the objective is the product at
    # every assignment, in the printer's units.
    assert any(c.has_int_prod() for c in model.proto.constraints)
    for values in ((0, 0, 0, 0), (3, 4, 5, 6), (100, 100, 100, 100)):
        fixed = model.clone()
        for name, value in zip(names.split(), values):
            fixed.add(sym_map[name] == value)
        fixed.minimize(lowered)
        solver = cp_model.CpSolver()
        assert solver.solve(fixed) == cp_model.OPTIMAL
        assert solver.objective_value == (values[0] + values[1]) * (
            values[2] + values[3]
        )
