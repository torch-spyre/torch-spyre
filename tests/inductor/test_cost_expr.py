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

"""Runnable prototype: sympy cost expression -> CP-SAT, with its tests.

Self-contained on purpose -- the lowering and the tests live in one file so it
can be run and stepped through before anything lands in the tree. When it does
land, it splits in two:

* ``CostExpressionError`` / ``Val`` / ``CpSatAnalysis`` / ``lower`` ->
  ``torch_spyre/_inductor/scratchpad/cost_expr.py``
* the ``TestCase``s -> ``tests/inductor/test_scratchpad_solver.py`` as
  ``TestCostExprLowering`` (device-free, so it needs no hardware)

Run::

    python3 -m pytest -x -v tests/inductor/test_cost_expr.py
    python3 tests/inductor/test_cost_expr.py           # same, via unittest
    python3 tests/inductor/test_cost_expr.py --demo    # dump one lowered model

Prefix with ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` if ``_C.so`` is stale — the
lowering never touches the device, so the backend does not need to load.

The load-bearing test is :class:`TestAgainstBruteForce`: it enumerates every
assignment over small domains and checks CP-SAT's optimum against sympy's own
evaluation. Everything else is a faster, more specific version of that check.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from itertools import product
from typing import Any

import sympy
import torch
from ortools.sat.python import cp_model
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis as VRA
from torch.utils._sympy.value_ranges import ValueRanges


# ---------------------------------------------------------------------------
# The lowering (-> scratchpad/cost_expr.py)
# ---------------------------------------------------------------------------
class CostExpressionError(Exception):
    """A cost expression outside the supported grammar, or a symbol we cannot
    bind. Always names the offending node or symbol."""


@dataclass
class Val:
    """An interpreted node: the CP-SAT object plus the range it lives in.

    Every reified node needs a *new* IntVar and CP-SAT needs an explicit domain
    for it. A blanket-wide domain is not a shortcut: at +/-2**55 the solver
    returns OPTIMAL with an objective that does not match the solution it hands
    back (see TestWideDomainsAreUnsafe). So the range rides along, and the
    arithmetic for it is torch's, not ours.
    """

    e: Any  # int | cp_model.LinearExpr | cp_model.IntVar
    vr: ValueRanges

    @property
    def is_const(self) -> bool:
        return isinstance(self.e, int)


def _is_concrete(bound) -> bool:
    """Whether a ValueRanges endpoint can become a CP-SAT domain edge.

    Test integrality, not finiteness: torch's ``int_oo`` reports
    ``is_finite == True``, so an ``is_finite`` guard lets infinity straight
    through and the failure surfaces much later, from inside the range
    arithmetic.
    """
    return bool(getattr(bound, "is_Integer", False))


def _bounds(vr: ValueRanges) -> tuple[int, int]:
    """ValueRanges -> a domain CP-SAT will accept, or a clean rejection."""
    lo, hi = vr.lower, vr.upper
    if not (_is_concrete(lo) and _is_concrete(hi)):
        raise CostExpressionError(
            f"unbounded subexpression {vr}: cannot size a CP-SAT variable; "
            "bound the leaf symbols"
        )
    lo, hi = int(lo), int(hi)
    if max(abs(lo), abs(hi)) > _MAX_MAGNITUDE:
        raise CostExpressionError(
            f"subexpression range {vr} exceeds the safe magnitude "
            f"{_MAX_MAGNITUDE}; rescale (COST_SCALE) before lowering"
        )
    return lo, hi


# CP-SAT rejects domains outside [-kint64max/2, +kint64max/2], and misbehaves
# well below that (TestWideDomainsAreUnsafe pins the observed failure at 2**55).
# 2**40 is comfortably inside both and far above any real cost magnitude.
_MAX_MAGNITUDE = 2**40


class CpSatAnalysis:
    """One method per supported sympy node. The supported grammar *is* this
    method set: an unimplemented node raises through ``lower`` rather than
    failing somewhere inside ortools."""

    def __init__(self, model: cp_model.CpModel) -> None:
        self.m = model
        self.aux: list[cp_model.IntVar] = []

    def _reify(self, vr: ValueRanges, tag: str) -> cp_model.IntVar:
        lo, hi = _bounds(vr)
        v = self.m.new_int_var(lo, hi, f"{tag}{len(self.aux)}")
        self.aux.append(v)
        return v

    # -- leaves --------------------------------------------------------------
    def constant(self, c, dtype) -> Val:
        if not isinstance(c, sympy.Integer):
            raise CostExpressionError(
                f"non-integer constant {c!r}: apply COST_SCALE before lowering"
            )
        return Val(int(c), VRA.constant(c, torch.int64))

    # -- linear: no variable, no constraint ----------------------------------
    def sym_sum(self, args: list[Val]) -> Val:
        """n-ary integer Add fastpath: one flat sum, not a fold chain."""
        vr = args[0].vr
        for a in args[1:]:
            vr = VRA.add(vr, a.vr)
        return Val(sum(a.e for a in args), vr)

    def add(self, a: Val, b: Val) -> Val:
        return Val(a.e + b.e, VRA.add(a.vr, b.vr))

    def mul(self, a: Val, b: Val) -> Val:
        vr = VRA.mul(a.vr, b.vr)
        if a.is_const or b.is_const:  # at most one non-constant factor
            return Val(a.e * b.e, vr)
        t = self._reify(vr, "mul")
        self.m.add_multiplication_equality(t, [a.e, b.e])
        return Val(t, vr)

    def pow_by_natural(self, a: Val, b: Val) -> Val:
        if b.is_const and b.e < 0:
            # sympy spells ``x / y`` as ``x * y**-1``, so this is where division
            # by a variable lands.
            raise CostExpressionError(
                f"unsupported division by a variable (Pow exponent {b.e})"
            )
        if not b.is_const:
            raise CostExpressionError(f"unsupported non-constant exponent {b.e}")
        acc = Val(1, VRA.constant(sympy.Integer(1), torch.int64))
        for _ in range(b.e):
            acc = self.mul(acc, a)
        return acc

    # -- reified: one variable + one constraint each -------------------------
    def minimum(self, a: Val, b: Val) -> Val:
        vr = VRA.minimum(a.vr, b.vr)
        t = self._reify(vr, "min")
        self.m.add_min_equality(t, [a.e, b.e])
        return Val(t, vr)

    def maximum(self, a: Val, b: Val) -> Val:
        vr = VRA.maximum(a.vr, b.vr)
        t = self._reify(vr, "max")
        self.m.add_max_equality(t, [a.e, b.e])
        return Val(t, vr)


def lower(model, expr, env: dict[sympy.Symbol, Val]) -> tuple[Val, CpSatAnalysis]:
    """Lower ``expr`` against ``env`` (symbol -> already-built CP-SAT var).

    Returns the interpreted root and the analysis, so callers can inspect the
    aux vars the lowering created.
    """

    def unbound(sym):
        known = ", ".join(sorted(str(s) for s in env))
        raise CostExpressionError(f"unbound symbol: {sym} (bound: {known})")

    # Check the leaves up front: an infinite range blows up inside the range
    # arithmetic before our own domain check could ever see it.
    for sym, val in sorted(env.items(), key=lambda kv: str(kv[0])):
        if not (_is_concrete(val.vr.lower) and _is_concrete(val.vr.upper)):
            raise CostExpressionError(f"unbounded leaf symbol {sym}: {val.vr}")

    analysis = CpSatAnalysis(model)
    try:
        out = sympy_interp(analysis, env, expr, missing_handler=unbound)
    except KeyError as exc:  # sympy node torch's dispatch table does not know
        raise CostExpressionError(f"unsupported sympy node: {exc}") from exc
    except AttributeError as exc:  # node we deliberately do not support
        raise CostExpressionError(f"unsupported operation: {exc}") from exc
    return out, analysis


def bind(model, name: str, lo: int, hi: int) -> Val:
    """A leaf: the CP-SAT var plus the range it already carries."""
    return Val(model.new_int_var(lo, hi, name), ValueRanges(lo, hi))


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
def dump(model, analysis=None, solver=None, root=None, status=None) -> None:
    """Print the model, and the solution if there is one. Call it from a
    breakpoint, or run this file with --demo."""
    print("--- variables ---")
    for v in model.proto.variables:
        val = ""
        if solver is not None:
            idx = list(model.proto.variables).index(v)
            val = f"  = {solver.value(model.get_int_var_from_proto_index(idx))}"
        print(f"  {v.name or '(anon)':>12}  domain={list(v.domain)}{val}")
    if analysis is not None:
        print(f"--- aux vars: {len(analysis.aux)} ---")
    if root is not None:
        print(f"--- root expr: {root.e}   range={root.vr} ---")
    if solver is not None:
        name = solver.status_name(status) if status is not None else "?"
        print(f"--- status={name} objective={solver.objective_value}")


def solve(model, objective, maximize: bool):
    """Solve deterministically so a failure reproduces."""
    if maximize:
        model.maximize(objective)
    else:
        model.minimize(objective)
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1
    s.parameters.random_seed = 0
    s.parameters.max_time_in_seconds = 30.0
    status = s.solve(model)
    return s, status


# ---------------------------------------------------------------------------
# Test corpus: (name, expression, {symbol: (lo, hi)})
# ---------------------------------------------------------------------------
_x, _y, _z, _w = sympy.symbols("x y z w")

CASES = [
    ("affine", 3 * _x + 5 * _y - 2, {_x: (0, 4), _y: (0, 4)}),
    ("min_max", sympy.Min(_x, _y) + sympy.Max(_x, 2 * _y), {_x: (0, 4), _y: (0, 4)}),
    (
        "nested_scaled_min",  # the shape unnest_min exists for
        sympy.Min(8, 5 * sympy.Min(_x, _y)),
        {_x: (0, 4), _y: (0, 4)},
    ),
    (
        "bool_gated_product",  # var * var -> reified
        3 * _z * (1 - _w),
        {_z: (0, 4), _w: (0, 1)},
    ),
    (
        "cost_model_shaped",
        sympy.Min(8, 5 * sympy.Min(_x, _y)) + 3 * _z * (1 - _w) + sympy.Max(_x, 2 * _y),
        {_x: (0, 4), _y: (0, 4), _z: (0, 4), _w: (0, 1)},
    ),
    (
        "negative_coeffs",
        4 * _x - 7 * _y + sympy.Max(_x - _y, 0),
        {_x: (0, 4), _y: (0, 4)},
    ),
    ("power", _x**2 - 3 * _x, {_x: (0, 4)}),
]


def _brute_force(expr, domains):
    """Every assignment over the (small) domains -> (min, max) of expr."""
    syms = sorted(domains, key=str)
    best_lo, best_hi = None, None
    for combo in product(*(range(domains[s][0], domains[s][1] + 1) for s in syms)):
        val = int(expr.subs(dict(zip(syms, combo))))
        best_lo = val if best_lo is None else min(best_lo, val)
        best_hi = val if best_hi is None else max(best_hi, val)
    return best_lo, best_hi


class _Base(unittest.TestCase):
    def build(self, domains):
        m = cp_model.CpModel()
        env = {s: bind(m, str(s), lo, hi) for s, (lo, hi) in domains.items()}
        return m, env


class TestAgainstBruteForce(_Base):
    """The load-bearing check: CP-SAT's optimum must equal the true optimum
    over an exhaustively enumerated domain, in both directions."""

    def test_optima_match_enumeration(self):
        for name, expr, domains in CASES:
            true_lo, true_hi = _brute_force(expr, domains)
            for maximize, truth in ((False, true_lo), (True, true_hi)):
                with self.subTest(case=name, maximize=maximize):
                    m, env = self.build(domains)
                    root, _analysis = lower(m, expr, env)
                    s, status = solve(m, root.e, maximize)
                    self.assertIn(
                        status,
                        (cp_model.OPTIMAL, cp_model.FEASIBLE),
                        f"{name}: {s.status_name(status)}",
                    )
                    sol = {sym: s.value(v.e) for sym, v in env.items()}
                    # The solver's objective, the model's own expression, and
                    # sympy must all agree -- disagreement between the first
                    # two is the wide-domain failure mode.
                    self.assertEqual(round(s.objective_value), s.value(root.e), name)
                    self.assertEqual(s.value(root.e), int(expr.subs(sol)), name)
                    self.assertEqual(round(s.objective_value), truth, f"{name} {sol}")


class TestBoundsAreSound(_Base):
    """Every aux domain must contain the value the solve gives it, and the
    root range must contain the true optimum."""

    def test_aux_domains_contain_solution(self):
        for name, expr, domains in CASES:
            with self.subTest(case=name):
                m, env = self.build(domains)
                root, analysis = lower(m, expr, env)
                s, status = solve(m, root.e, maximize=True)
                self.assertIn(status, (cp_model.OPTIMAL, cp_model.FEASIBLE))
                for v in analysis.aux:
                    val = s.value(v)
                    lo, hi = v.domain.min(), v.domain.max()
                    self.assertTrue(
                        lo <= val <= hi, f"{name}: {v.name}={val} not in [{lo},{hi}]"
                    )

    def test_root_range_contains_true_optima(self):
        for name, expr, domains in CASES:
            with self.subTest(case=name):
                m, env = self.build(domains)
                root, _ = lower(m, expr, env)
                true_lo, true_hi = _brute_force(expr, domains)
                self.assertLessEqual(int(root.vr.lower), true_lo, name)
                self.assertGreaterEqual(int(root.vr.upper), true_hi, name)


class TestWideDomainsAreUnsafe(unittest.TestCase):
    """Why Val carries a range at all.

    A blanket-wide aux domain is legal by CP-SAT's own validator but produces a
    wrong answer: OPTIMAL, objective 80, on a solution actually worth 77, when
    the true optimum is 81 (ortools 9.15.6755). Kept as a test so the day it
    starts passing we can simplify.
    """

    EXPR = (
        sympy.Min(40, 5 * sympy.Min(_x, _y)) + 3 * _z * (1 - _w) + sympy.Max(_x, 2 * _y)
    )
    DOMAINS = {_x: (0, 10), _y: (0, 10), _z: (0, 7), _w: (0, 1)}

    def _solve_with_aux_width(self, width):
        class Wide(CpSatAnalysis):
            def _reify(self, vr, tag):  # noqa: ARG002 - width, not range
                v = self.m.new_int_var(-width, width, f"{tag}{len(self.aux)}")
                self.aux.append(v)
                return v

        m = cp_model.CpModel()
        env = {s: bind(m, str(s), lo, hi) for s, (lo, hi) in self.DOMAINS.items()}
        root = sympy_interp(Wide(m), env, self.EXPR)
        self.assertEqual(m.validate(), "", "model itself is valid")
        s, _status = solve(m, root.e, maximize=True)
        sol = {sym: s.value(v.e) for sym, v in env.items()}
        return round(s.objective_value), s.value(root.e), int(self.EXPR.subs(sol))

    def test_derived_bounds_are_correct(self):
        m = cp_model.CpModel()
        env = {s: bind(m, str(s), lo, hi) for s, (lo, hi) in self.DOMAINS.items()}
        root, _ = lower(m, self.EXPR, env)
        s, _status = solve(m, root.e, maximize=True)
        self.assertEqual(round(s.objective_value), 81)
        self.assertEqual(s.value(root.e), 81)

    def test_wide_domain_reports_an_objective_it_cannot_produce(self):
        reported, actual, truth = self._solve_with_aux_width(2**55)
        self.assertNotEqual(
            reported,
            actual,
            "ortools now agrees with itself at 2**55 — re-check whether the "
            "range tracking can be simplified",
        )
        self.assertEqual((reported, actual, truth), (80, 77, 77))

    def test_moderate_widths_are_still_correct(self):
        for width in (2**40, 2**31, 2**20):
            with self.subTest(width=width):
                reported, actual, truth = self._solve_with_aux_width(width)
                self.assertEqual((reported, actual, truth), (81, 81, 81))


class TestModelSize(_Base):
    """R3.7: one reified var per Min/Max/var-times-var node, and nothing for
    the affine part."""

    def test_aux_count(self):
        expected = {
            "affine": 0,
            "min_max": 2,
            "nested_scaled_min": 2,
            "bool_gated_product": 1,
            "cost_model_shaped": 4,
            "negative_coeffs": 1,
            "power": 1,
        }
        for name, expr, domains in CASES:
            with self.subTest(case=name):
                m, env = self.build(domains)
                _root, analysis = lower(m, expr, env)
                self.assertEqual(len(analysis.aux), expected[name], name)

    def test_nary_min_costs_k_minus_1_vars(self):
        """An n-ary Min costs k-1 reified vars, not 1.

        sympy flattens ``Min(4, Min(x, y))`` to a 3-ary ``Min(4, x, y)``
        itself, but ``_run_sympy_handler`` folds associative ops pairwise, so
        we get two ``add_min_equality`` calls where CP-SAT would accept one
        over three operands. That is an optimisation left on the table -- the
        *answer* is right either way, which is the point: unlike the rewrite
        this replaces, nothing here can silently drop an operand.
        """
        flat = sympy.Min(4, sympy.Min(_x, _y))
        self.assertEqual(len(flat.args), 3, "sympy flattens nested Min itself")
        for expr in (flat, sympy.Min(4, _x, _y)):
            with self.subTest(expr=str(expr)):
                m, env = self.build({_x: (0, 9), _y: (0, 9)})
                root, analysis = lower(m, expr, env)
                s, _ = solve(m, root.e, maximize=True)
                self.assertEqual(len(analysis.aux), len(expr.args) - 1)
                self.assertEqual(s.value(root.e), 4)


class TestRejections(_Base):
    """Every unsupported construct must surface as CostExpressionError naming
    what it was -- never as a TypeError from inside ortools."""

    def test_rejected(self):
        cases = [
            ("transcendental", sympy.log(_x), "unsupported operation"),
            ("division by var", _x / _y, "division by a variable"),
            ("rational coefficient", _x / 2, "non-integer constant"),
            ("float coefficient", 0.5 * _x, "non-integer constant"),
        ]
        for name, expr, fragment in cases:
            with self.subTest(case=name):
                m, env = self.build({_x: (1, 4), _y: (1, 4)})
                with self.assertRaises(CostExpressionError) as cm:
                    lower(m, expr, env)
                self.assertIn(fragment, str(cm.exception), f"{name}: {cm.exception}")

    def test_unbound_symbol_names_itself_and_the_bound_set(self):
        m, env = self.build({_x: (0, 4)})
        with self.assertRaises(CostExpressionError) as cm:
            lower(m, _x + sympy.Symbol("is_lx_buf7"), env)
        self.assertIn("is_lx_buf7", str(cm.exception))
        self.assertIn("bound: x", str(cm.exception))

    def test_unbounded_leaf_is_rejected(self):
        """Caught at the boundary: an infinite range detonates inside the
        range arithmetic long before a domain could be derived from it."""
        m = cp_model.CpModel()
        env = {_x: Val(m.new_int_var(0, 10, "x"), ValueRanges(0, int_oo))}
        with self.assertRaises(CostExpressionError) as cm:
            lower(m, sympy.Max(_x, 3), env)
        self.assertIn("unbounded leaf symbol", str(cm.exception))


class TestDeterminism(_Base):
    """R8.4: the same expression must lower to the same model, twice."""

    def test_identical_protos(self):
        _name, expr, domains = CASES[4]
        protos = []
        for _ in range(2):
            m, env = self.build(domains)
            root, _ = lower(m, expr, env)
            m.minimize(root.e)
            protos.append(str(m.proto))
        self.assertEqual(protos[0], protos[1])


class TestSolverShaped(unittest.TestCase):
    """A miniature of the real integration: per-buffer residency bools and an
    inv_cores table selected by AddElement, exactly as the joint solver builds
    them, with the cost expression written in the buffers' own symbols.

    This is the one to step through -- it is the shape ``_run`` will have.
    """

    SCALE = 240  # divisible by every candidate core count, so 1/cores is exact

    def _model(self, buffers, capacity):
        m = cp_model.CpModel()
        env, is_lx, sizes = {}, {}, {}
        for name, size, cands in buffers:
            lx = m.new_bool_var(f"in_buffer_{name}")
            inv_table = [self.SCALE // c for c in cands]
            div = m.new_int_var(0, len(cands) - 1, f"div_{name}")
            inv = m.new_int_var(min(inv_table), max(inv_table), f"inv_cores_{name}")
            m.add_element(div, inv_table, inv)
            # symbols the producer would have minted off the buffer
            env[sympy.Symbol(f"is_lx_{name}")] = Val(lx, ValueRanges(0, 1))
            env[sympy.Symbol(f"inv_cores_{name}")] = Val(
                inv, ValueRanges(min(inv_table), max(inv_table))
            )
            is_lx[name] = lx
            sizes[name] = size
        m.add(sum(sizes[n] * is_lx[n] for n in is_lx) <= capacity)
        return m, env, is_lx

    def test_traffic_plus_time(self):
        buffers = [  # (name, size, candidate core counts)
            ("buf0", 60, [1, 2, 4]),
            ("buf1", 40, [1, 2]),
            ("buf2", 90, [1, 3, 6]),
        ]
        # cost = HBM traffic paid when spilled + compute time (~ work/cores)
        terms = []
        for name, size, _cands in buffers:
            spilled = 1 - sympy.Symbol(f"is_lx_{name}")
            terms.append(2 * size * spilled)
            terms.append(size * sympy.Symbol(f"inv_cores_{name}"))
        # Add(*terms), never sum(terms): incremental sympy accumulation is
        # superlinear (0.34 s at 500 terms, 37.7 s at 1500, vs 0.09 s here).
        expr = sympy.Add(*terms)

        m, env, is_lx = self._model(buffers, capacity=100)
        root, analysis = lower(m, expr, env)
        s, status = solve(m, root.e, maximize=False)
        self.assertIn(status, (cp_model.OPTIMAL, cp_model.FEASIBLE))

        sol = {sym: s.value(v.e) for sym, v in env.items()}
        self.assertEqual(round(s.objective_value), int(expr.subs(sol)))
        # capacity forces a spill; the model should keep the buffer whose
        # residency saves the most traffic per byte held.
        resident = {n for n, v in is_lx.items() if s.value(v)}
        self.assertLessEqual(sum(sz for n, sz, _ in buffers if n in resident), 100)
        self.assertTrue(resident, "expected at least one resident buffer")
        # affine in the decision vars -> nothing to reify
        self.assertEqual(len(analysis.aux), 0)


def _demo() -> None:
    _name, expr, domains = CASES[4]
    print(f"expression: {expr}\ndomains: { {str(k): v for k, v in domains.items()} }\n")
    m = cp_model.CpModel()
    env = {s: bind(m, str(s), lo, hi) for s, (lo, hi) in domains.items()}
    root, analysis = lower(m, expr, env)
    s, status = solve(m, root.e, maximize=True)
    dump(m, analysis, s, root, status)
    sol = {sym: s.value(v.e) for sym, v in env.items()}
    print(
        f"sympy at solution: {int(expr.subs(sol))}   brute force: {_brute_force(expr, domains)}"
    )


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        unittest.main(verbosity=2)
