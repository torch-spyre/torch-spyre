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

"""The solver interface the encoding strategy posts to (Phase 3a, prototype).

:class:`Backend` is shaped like ``pywraplp.Solver`` -- ``IntVar`` /
``BoolVar`` / ``Add`` / ``Minimize`` / ``Solve`` / ``Value`` -- plus the
operations the model uses that ``pywraplp`` lacks: enforcement
(``OnlyEnforceIf``), Boolean clauses, element and table, products, division,
max, optional intervals and the 2-D no-overlap. The strategy and the objective
ladder post only through it; backends differ in how they solve.

Every backend here posts through one recorded CP model (``CpSat``'s), which the
others then solve their own way: ``Mip`` linearizes it onto ``pywraplp``
(SCIP / HiGHS / CBC: big-M enforcement, one-hot element and table, McCormick
products, pairwise no-overlap disjunctions), ``BruteForce`` searches it
exhaustively. Recording once keeps the cost-expression printer, which builds
``cp_model`` expressions, working for every backend.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Protocol, Sequence

from google.protobuf import text_format
from ortools.linear_solver import pywraplp
from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model

Var = Any
LinearExpr = Any


class Constraint(Protocol):
    def OnlyEnforceIf(self, *lits: Var) -> "Constraint": ...


class Backend(Protocol):
    """What the strategy and the ladder post to."""

    model: cp_model.CpModel  # the recorded model (the printer builds on it)

    def IntVar(self, lb: int, ub: int, name: str) -> Var: ...
    def IntVarFromValues(self, values: Sequence[int], name: str) -> Var: ...
    def BoolVar(self, name: str) -> Var: ...
    def Constant(self, value: int) -> Var: ...
    def Add(self, bounded: object) -> Constraint: ...
    def AddBoolOr(self, lits: Sequence[Var]) -> Constraint: ...
    def AddBoolAnd(self, lits: Sequence[Var]) -> Constraint: ...
    def AddAtMostOne(self, lits: Sequence[Var]) -> Constraint: ...
    def AddImplication(self, a: Var, b: Var) -> Constraint: ...
    def AddElement(
        self, index: Var, values: Sequence[int], target: Var
    ) -> Constraint: ...
    def AddAllowedAssignments(
        self, vars: Sequence[Var], tuples: Sequence[Sequence[int]]
    ) -> Constraint: ...
    def AddMaxEquality(
        self, target: Var, exprs: Sequence[LinearExpr]
    ) -> Constraint: ...
    def AddDivisionEquality(
        self, target: Var, num: LinearExpr, den: LinearExpr
    ) -> Constraint: ...
    def AddMinEquality(
        self, target: Var, exprs: Sequence[LinearExpr]
    ) -> Constraint: ...
    def AddMultiplicationEquality(
        self, target: Var, factors: Sequence[LinearExpr]
    ) -> Constraint: ...
    def NewOptionalIntervalVar(
        self,
        start: LinearExpr,
        size: LinearExpr,
        end: LinearExpr,
        present: Var,
        name: str,
    ) -> Any: ...
    def AddNoOverlap2D(self, x: Sequence[Any], y: Sequence[Any]) -> Constraint: ...
    def Minimize(self, expr: LinearExpr) -> None: ...
    def Maximize(self, expr: LinearExpr) -> None: ...
    def SetHint(self, vars: Sequence[Var], values: Sequence[int]) -> None: ...
    def HintSolution(self) -> None: ...
    def Solve(self) -> int: ...
    def StatusName(self, status: int) -> str: ...
    def WallTime(self) -> float: ...
    def ObjectiveValue(self) -> float: ...
    def Value(self, expr: LinearExpr) -> int: ...
    def BooleanValue(self, lit: Var) -> bool: ...
    def NumVariables(self) -> int: ...
    def NumConstraints(self) -> int: ...


class CpSat:
    """Pass-through to ``cp_model``: the default and the reference."""

    def __init__(self, model: Optional[cp_model.CpModel] = None) -> None:
        self.model = model if model is not None else cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.record = None  # (solver, model) -> status; the solver's stats hook
        self._objective = False

    # ---- posting ----
    def IntVar(self, lb, ub, name):
        return self.model.new_int_var(lb, ub, name)

    def IntVarFromValues(self, values, name):
        return self.model.new_int_var_from_domain(
            cp_model.Domain.from_values(list(values)), name
        )

    def BoolVar(self, name):
        return self.model.new_bool_var(name)

    def Constant(self, value):
        return self.model.new_constant(value)

    def Add(self, bounded):
        return self.model.add(bounded)

    def AddBoolOr(self, lits):
        return self.model.add_bool_or(lits)

    def AddBoolAnd(self, lits):
        return self.model.add_bool_and(lits)

    def AddAtMostOne(self, lits):
        return self.model.add_at_most_one(lits)

    def AddImplication(self, a, b):
        return self.model.add_implication(a, b)

    def AddElement(self, index, values, target):
        return self.model.add_element(index, values, target)

    def AddAllowedAssignments(self, vars, tuples):
        return self.model.add_allowed_assignments(vars, tuples)

    def AddMaxEquality(self, target, exprs):
        return self.model.add_max_equality(target, exprs)

    def AddMultiplicationEquality(self, target, factors):
        return self.model.add_multiplication_equality(target, factors)

    def AddDivisionEquality(self, target, num, den):
        return self.model.add_division_equality(target, num, den)

    def AddMinEquality(self, target, exprs):
        return self.model.add_min_equality(target, exprs)

    def NewOptionalIntervalVar(self, start, size, end, present, name):
        return self.model.new_optional_interval_var(start, size, end, present, name)

    def AddNoOverlap2D(self, x, y):
        return self.model.add_no_overlap_2d(x, y)

    def Minimize(self, expr):
        self.model.minimize(expr)

    def Maximize(self, expr):
        self.model.maximize(expr)

    def SetHint(self, vars, values):
        for var, value in zip(vars, values):
            self.model.add_hint(var, value)

    def HintSolution(self):
        """Hint every variable with the last solution, so a stage that holds
        the previous optimum starts from a solution that meets it."""
        self.model.clear_hints()
        for i in range(len(self.model.proto.variables)):
            var = self.model.get_int_var_from_proto_index(i)
            self.model.add_hint(var, self.Value(var))

    # ---- solving ----
    def Solve(self, objective: bool = False) -> int:
        if self.record is not None:
            return self.record(self.solver, self.model, objective=objective)
        return self.solver.Solve(self.model)

    def StatusName(self, status):
        return self.solver.StatusName(status)

    def WallTime(self):
        return self.solver.WallTime()

    def ObjectiveValue(self):
        return self.solver.ObjectiveValue()

    def Value(self, expr):
        return self.solver.Value(expr)

    def BooleanValue(self, lit):
        return self.solver.BooleanValue(lit)

    def NumVariables(self):
        return len(self.model.proto.variables)

    def NumConstraints(self):
        return len(self.model.proto.constraints)


class _SolvedRecording(CpSat):
    """A backend that solves the recorded model with its own engine and
    answers ``Value`` from that solution."""

    def __init__(self, model=None):
        super().__init__(model)
        self._values: Optional[list] = None
        self._objective_value = 0.0
        self._wall = 0.0
        self._status = cp_model.UNKNOWN

    def _proto(self):
        return text_format.Parse(str(self.model.proto), cp_model_pb2.CpModelProto())

    def StatusName(self, status):
        return cp_model.CpSolver().StatusName(status)

    def WallTime(self):
        return self._wall

    def ObjectiveValue(self):
        return self._objective_value

    def Value(self, expr):
        if isinstance(expr, (int, float)):
            return int(expr)
        flat = cp_model.FlatIntExpr(expr)
        return int(flat.offset) + sum(
            int(c) * self._lit(v) for v, c in zip(flat.vars, flat.coeffs)
        )

    def _lit(self, v):
        if isinstance(v, cp_model_helper_not()):
            return 1 - self._values[v.negated().index]
        return self._values[v.index]

    def BooleanValue(self, lit):
        return bool(self.Value(lit))


def cp_model_helper_not():
    from ortools.sat.python import cp_model_helper

    return cp_model_helper.NotBooleanVariable


class Mip(_SolvedRecording):
    """The recorded model, linearized onto a ``pywraplp`` MIP solver with a
    zero relative gap, so OPTIMAL is exact."""

    def __init__(self, solver_id: str = "SCIP", time_limit_s: float = 60.0, model=None):
        super().__init__(model)
        self.solver_id = solver_id
        self.time_limit_s = time_limit_s

    def Solve(self, objective: bool = False) -> int:
        lin = _Linearizer(self._proto(), self.solver_id)
        lin.build()
        lin.s.SetTimeLimit(int(self.time_limit_s * 1000))
        params = pywraplp.MPSolverParameters()
        params.SetDoubleParam(params.RELATIVE_MIP_GAP, 0.0)
        t = time.perf_counter()
        status = lin.s.Solve(params)
        self._wall = time.perf_counter() - t
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            self._values = [int(round(v.solution_value())) for v in lin.x]
            self._objective_value = lin.s.Objective().Value() * lin.report
            self._status = (
                cp_model.OPTIMAL
                if status == pywraplp.Solver.OPTIMAL
                else cp_model.FEASIBLE
            )
        elif status == pywraplp.Solver.INFEASIBLE:
            self._status = cp_model.INFEASIBLE
        else:
            self._status = cp_model.UNKNOWN
        return self._status


class BruteForce(_SolvedRecording):
    """Exhaustive search over the recorded model: an oracle for tiny models."""

    def __init__(self, max_nodes: int = 3_000_000, model=None):
        super().__init__(model)
        self.max_nodes = max_nodes

    def Solve(self, objective: bool = False) -> int:
        search = _BruteForceSearch(self._proto(), self.max_nodes)
        t = time.perf_counter()
        search.solve()
        self._wall = time.perf_counter() - t
        if search.best is None:
            self._status = cp_model.INFEASIBLE
            return self._status
        self._values = search.best_a
        self._objective_value = search.objective_value()
        self._status = cp_model.OPTIMAL
        return self._status


# =============================================================================
# Engines (prototype): the MIP linearization and the exhaustive search.
# =============================================================================
class MipUnsupported(Exception):
    pass


def _vals(dom):
    out = []
    for lo, hi in zip(dom[::2], dom[1::2]):
        out.extend(range(lo, hi + 1))
    return out


class _Linearizer:
    def __init__(self, proto, solver_id):
        self.p = proto
        self.s = pywraplp.Solver.CreateSolver(solver_id)
        if self.s is None:
            raise MipUnsupported(solver_id)
        self.x = []
        self.bounds = []
        self.onehot_cache = {}
        for i, v in enumerate(proto.variables):
            dom = list(v.domain)
            lo, hi = dom[0], dom[-1]
            var = self.s.IntVar(lo, hi, f"v{i}")
            self.x.append(var)
            self.bounds.append((lo, hi))
            if len(dom) > 2:  # holes: pin to the listed values
                self._onehot(i)
        self.intervals = {}

    # ---- expressions ----
    def lit(self, r):
        return self.x[r] if r >= 0 else 1 - self.x[-r - 1]

    def lin(self, e):
        return sum(c * self.lit_int(v) for v, c in zip(e.vars, e.coeffs)) + e.offset

    def lit_int(self, r):
        return self.x[r] if r >= 0 else -self.x[-r - 1]

    def bnd(self, e):
        lo = hi = getattr(e, "offset", 0)
        for v, c in zip(e.vars, e.coeffs):
            a, b = self.bounds[v if v >= 0 else -v - 1]
            if v < 0:
                a, b = -b, -a
            lo += min(c * a, c * b)
            hi += max(c * a, c * b)
        return lo, hi

    def slack(self, enf):
        """0 when every enforcement literal is true, >= 1 otherwise."""
        return sum(1 - self.lit(lit) for lit in enf)

    def _onehot(self, v):
        if v not in self.onehot_cache:
            vals = _vals(list(self.p.variables[v].domain))
            if len(vals) > 512:
                raise MipUnsupported(f"domain of v{v} has {len(vals)} values")
            z = {val: self.s.BoolVar(f"oh{v}_{val}") for val in vals}
            self.s.Add(sum(z.values()) == 1)
            self.s.Add(self.x[v] == sum(val * zv for val, zv in z.items()))
            self.onehot_cache[v] = z
        return self.onehot_cache[v]

    def _single_var(self, e):
        if len(e.vars) == 1 and e.vars[0] >= 0:
            return e.vars[0], e.coeffs[0], e.offset
        return None

    # ---- constraints ----
    def add_range(self, expr, lo, hi, enf, ebnd):
        k = self.slack(enf) if enf else 0
        elo, ehi = ebnd
        if lo > elo:
            self.s.Add(expr >= lo - (lo - elo) * k)
        if hi < ehi:
            self.s.Add(expr <= hi + (ehi - hi) * k)

    def linear(self, c):
        lin = c.linear
        expr = sum(cf * self.lit_int(v) for v, cf in zip(lin.vars, lin.coeffs))
        eb = self.bnd(lin)
        dom = list(lin.domain)
        segs = list(zip(dom[::2], dom[1::2]))
        segs = [
            (max(a, eb[0]), min(b, eb[1])) for a, b in segs if b >= eb[0] and a <= eb[1]
        ]
        enf = list(c.enforcement_literal)
        if not segs:
            if enf:
                self.s.Add(self.slack(enf) >= 1)
                return
            self.s.Add(self.s.Sum([]) >= 1)  # infeasible
            return
        if len(segs) == 1:
            self.add_range(expr, *segs[0], enf, eb)
            return
        zs = [self.s.BoolVar("") for _ in segs]
        self.s.Add(sum(zs) == 1)
        for z, (lo, hi) in zip(zs, segs):
            k = (self.slack(enf) if enf else 0) + (1 - z)
            if lo > eb[0]:
                self.s.Add(expr >= lo - (lo - eb[0]) * k)
            if hi < eb[1]:
                self.s.Add(expr <= hi + (eb[1] - hi) * k)

    def element(self, c):
        e = c.element
        idx = self._single_var(e.linear_index)
        if idx is None or idx[1] != 1 or idx[2] != 0:
            raise MipUnsupported("element index")
        consts = []
        for x in e.exprs:
            if len(x.vars):
                raise MipUnsupported("element over vars")
            consts.append(x.offset)
        z = self._onehot(idx[0])
        tgt = self.lin(e.linear_target)
        self.s.Add(
            tgt == sum(consts[v] * zv for v, zv in z.items() if 0 <= v < len(consts))
        )
        for v, zv in z.items():
            if not 0 <= v < len(consts):
                self.s.Add(zv == 0)

    def table(self, c):
        t = c.table
        if t.negated:
            raise MipUnsupported("negated table")
        w = len(t.exprs)
        vals = list(t.values)
        rows = [vals[j : j + w] for j in range(0, len(vals), w)]
        enf = list(c.enforcement_literal)
        r = [self.s.BoolVar("") for _ in rows]
        on = 1 - self.slack(enf) if enf else 1
        self.s.Add(sum(r) <= 1)
        self.s.Add(sum(r) >= on)
        for j, x in enumerate(t.exprs):
            e = self.lin(x)
            lo, hi = self.bnd(x)
            k = self.slack(enf) if enf else 0
            pick = sum(rk * row[j] for rk, row in zip(r, rows))
            big = max(
                hi - min(row[j] for row in rows), max(row[j] for row in rows) - lo, 0
            )
            self.s.Add(e - pick <= big * k)
            self.s.Add(pick - e <= big * k)

    def product(self, c):
        g = c.int_prod
        if len(g.exprs) != 2:
            raise MipUnsupported("int_prod arity")
        a, b = g.exprs
        sa, sb = self._single_var(a), self._single_var(b)
        if sa is None or sb is None:
            raise MipUnsupported("int_prod of sums")
        # one-hot the smaller domain, McCormick the other
        va, vb = sa[0], sb[0]
        if len(_vals(list(self.p.variables[va].domain))) > len(
            _vals(list(self.p.variables[vb].domain))
        ):
            (va, sa), (vb, sb) = (vb, sb), (va, sa)
            a, b = b, a
        z = self._onehot(va)
        lo, hi = self.bnd(b)
        ybexpr = self.lin(b)
        terms = []
        for val, zv in z.items():
            w = self.s.NumVar(min(0, lo), max(0, hi), "")
            self.s.Add(w <= hi * zv)
            self.s.Add(w >= lo * zv)
            self.s.Add(w <= ybexpr - lo * (1 - zv))
            self.s.Add(w >= ybexpr - hi * (1 - zv))
            terms.append((sa[1] * val + sa[2]) * w)
        self.s.Add(self.lin(g.target) == sum(terms))

    def division(self, c):
        g = c.int_div
        n, d = g.exprs
        if len(n.vars) or self._single_var(d) is None:
            raise MipUnsupported("int_div form")
        dv, dc, do = self._single_var(d)
        z = self._onehot(dv)
        num = n.offset
        expr = 0
        for val, zv in z.items():
            den = dc * val + do
            if den == 0:
                self.s.Add(zv == 0)
                continue
            q = abs(num) // abs(den) * (1 if (num >= 0) == (den > 0) else -1)
            expr = expr + q * zv
        self.s.Add(self.lin(g.target) == expr)

    def lin_max(self, c):
        g = c.lin_max
        t = self.lin(g.target)
        tb = self.bnd(g.target)
        bs = [self.s.BoolVar("") for _ in g.exprs]
        self.s.Add(sum(bs) == 1)
        for b, x in zip(bs, g.exprs):
            e = self.lin(x)
            eb = self.bnd(x)
            self.s.Add(t >= e)
            self.s.Add(t <= e + (tb[1] - eb[0]) * (1 - b))

    def interval(self, ci, c):
        iv = c.interval
        enf = list(c.enforcement_literal)
        s, sz, e = self.lin(iv.start), self.lin(iv.size), self.lin(iv.end)
        lo, hi = self.bnd(iv.end)
        ls, hs = self.bnd(iv.start)
        lz, hz = self.bnd(iv.size)
        k = self.slack(enf) if enf else 0
        big = max(abs(hs + hz - lo), abs(hi - ls - lz), 1)
        self.s.Add(e - s - sz <= big * k)
        self.s.Add(s + sz - e <= big * k)
        pres = 1 - k if enf else 1
        self.intervals[ci] = (s, sz, e, (ls, hs), (lo, hi), (lz, hz), pres)

    def no_overlap_2d(self, c):
        no = c.no_overlap_2d
        boxes = [
            (self.intervals[xi], self.intervals[yi])
            for xi, yi in zip(no.x_intervals, no.y_intervals)
        ]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (xa, ya), (xb, yb) = boxes[i], boxes[j]
                ds = [self.s.BoolVar("") for _ in range(8)]
                # inactive if either box is absent
                self.s.Add(sum(ds) >= xa[6] + ya[6] + xb[6] + yb[6] - 3)
                pairs = [
                    (xa[2], xb[0], xa[4][1] - xb[3][0]),
                    (xb[2], xa[0], xb[4][1] - xa[3][0]),
                    (ya[2], yb[0], ya[4][1] - yb[3][0]),
                    (yb[2], ya[0], yb[4][1] - ya[3][0]),
                ]
                for d, (end, start, big) in zip(ds, pairs):
                    self.s.Add(end - start <= max(big, 0) * (1 - d))
                for d, box in zip(ds[4:], (xa, ya, xb, yb)):
                    self.s.Add(box[1] <= box[5][1] * (1 - d))

    def build(self):
        for ci, c in enumerate(self.p.constraints):
            k = c.WhichOneof("constraint")
            if k == "interval":
                self.interval(ci, c)
        for ci, c in enumerate(self.p.constraints):
            k = c.WhichOneof("constraint")
            enf = list(c.enforcement_literal)
            if k == "linear":
                self.linear(c)
            elif k == "bool_or":
                self.s.Add(
                    sum(self.lit(lit) for lit in c.bool_or.literals)
                    >= 1 - (self.slack(enf) if enf else 0)
                )
            elif k == "bool_and":
                for lit in c.bool_and.literals:
                    self.s.Add(self.lit(lit) >= 1 - (self.slack(enf) if enf else 0))
            elif k == "at_most_one":
                self.s.Add(sum(self.lit(lit) for lit in c.at_most_one.literals) <= 1)
            elif k == "exactly_one":
                self.s.Add(sum(self.lit(lit) for lit in c.exactly_one.literals) == 1)
            elif k == "element":
                self.element(c)
            elif k == "table":
                self.table(c)
            elif k == "int_prod":
                self.product(c)
            elif k == "int_div":
                self.division(c)
            elif k == "lin_max":
                self.lin_max(c)
            elif k == "no_overlap_2d":
                self.no_overlap_2d(c)
            elif k == "interval":
                pass
            else:
                raise MipUnsupported(k)
        if self.p.HasField("objective"):
            o = self.p.objective
            expr = (
                sum(cf * self.lit_int(v) for v, cf in zip(o.vars, o.coeffs)) + o.offset
            )
            self.report = o.scaling_factor or 1.0
            self.s.Minimize(expr)
        elif self.p.HasField("floating_point_objective"):
            o = self.p.floating_point_objective
            expr = (
                sum(cf * self.lit_int(v) for v, cf in zip(o.vars, o.coeffs)) + o.offset
            )
            self.report = 1.0
            (self.s.Maximize if o.maximize else self.s.Minimize)(expr)
        else:
            self.report = 1.0


class _DomainValues:
    """A domain's values in order, never listed: offset and cost domains span
    billions of values, and listing them exhausts memory."""

    def __init__(self, dom):
        self.spans = list(zip(dom[::2], dom[1::2]))
        self.size = sum(hi - lo + 1 for lo, hi in self.spans)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i < 0:
            i += self.size
        for lo, hi in self.spans:
            if i <= hi - lo:
                return lo + i
            i -= hi - lo + 1
        raise IndexError(i)

    def __iter__(self):
        for lo, hi in self.spans:
            yield from range(lo, hi + 1)


def _complementary(d1, d2) -> bool:
    """Whether two domains (flat [lo, hi, ...] lists) partition the int64 range."""
    spans = sorted([*zip(d1[::2], d1[1::2]), *zip(d2[::2], d2[1::2])])
    nxt = -(2**63)
    for lo, hi in spans:
        if lo != nxt:
            return False
        nxt = hi + 1
    return nxt == 2**63


class TooBig(Exception):
    pass


class _BruteForceSearch:
    def __init__(self, proto, max_nodes: int = 5_000_000):
        self.p = proto
        self.n = len(proto.variables)
        self.domains = [list(v.domain) for v in proto.variables]
        self.max_nodes = max_nodes
        self.cons = list(proto.constraints)
        self.intervals = {
            i: c.interval for i, c in enumerate(self.cons) if c.HasField("interval")
        }
        # which constraints mention each var
        self.vars_of = [self._vars(c) for c in self.cons]
        self.watch: list[list[int]] = [[] for _ in range(self.n)]
        for ci, vs in enumerate(self.vars_of):
            for v in vs:
                self.watch[v].append(ci)
        self.defs = self._definitions()
        self.objective = self._objective()

    # ---- structure ----
    @staticmethod
    def _ref(r):
        return r if r >= 0 else -r - 1

    def _expr_vars(self, e):
        return {self._ref(v) for v in e.vars}

    def _vars(self, c):
        k = c.WhichOneof("constraint")
        vs = {self._ref(x) for x in c.enforcement_literal}
        if k == "linear":
            vs |= {self._ref(v) for v in c.linear.vars}
        elif k in ("bool_or", "bool_and", "at_most_one", "exactly_one"):
            vs |= {self._ref(v) for v in getattr(c, k).literals}
        elif k == "element":
            e = c.element
            if e.HasField("linear_index"):
                vs |= self._expr_vars(e.linear_index) | self._expr_vars(e.linear_target)
                for x in e.exprs:
                    vs |= self._expr_vars(x)
            else:
                vs |= {e.index, e.target, *e.vars}
        elif k == "table":
            t = c.table
            if len(t.exprs):
                for x in t.exprs:
                    vs |= self._expr_vars(x)
            else:
                vs |= set(t.vars)
        elif k in ("int_prod", "int_div", "lin_max"):
            g = getattr(c, k)
            vs |= self._expr_vars(g.target)
            for x in g.exprs:
                vs |= self._expr_vars(x)
        elif k == "interval":
            iv = c.interval
            vs |= (
                self._expr_vars(iv.start)
                | self._expr_vars(iv.size)
                | self._expr_vars(iv.end)
            )
        elif k == "no_overlap_2d":
            for i in (*c.no_overlap_2d.x_intervals, *c.no_overlap_2d.y_intervals):
                vs |= self._vars(self.cons[i])
        else:
            raise NotImplementedError(k)
        return vs

    def _definitions(self):
        """var -> constraint index that computes it from other vars. One
        constraint defines one var: an equality defining all its vars leaves
        each waiting on the others, and none ever resolves."""
        defs = {}
        self._lookups: dict[int, dict] = {}
        for ci, c in enumerate(self.cons):
            if len(c.enforcement_literal):
                continue
            k = c.WhichOneof("constraint")
            if k == "element" and c.element.HasField("linear_target"):
                t = c.element.linear_target
                if len(t.vars) == 1 and abs(t.coeffs[0]) == 1:
                    defs.setdefault(self._ref(t.vars[0]), ci)
            elif k in ("int_prod", "int_div", "lin_max"):
                t = getattr(c, k).target
                if len(t.vars) == 1 and abs(t.coeffs[0]) == 1:
                    defs.setdefault(self._ref(t.vars[0]), ci)
            elif k == "linear":
                lin = c.linear
                if list(lin.domain) == [lin.domain[0], lin.domain[0]]:
                    # The newest var: an auxiliary is made after its inputs.
                    unit = [
                        self._ref(v)
                        for v, co in zip(lin.vars, lin.coeffs)
                        if abs(co) == 1 and self._ref(v) not in defs
                    ]
                    if unit:
                        defs[max(unit)] = ci
            elif k == "table":
                lookup = self._functional_table(c.table)
                if lookup is not None and lookup[0] not in defs:
                    defs[lookup[0]] = ci
                    self._lookups[ci] = lookup[1]
        self._reified = self._reifications(defs)
        for v, ci in self._reified.items():
            defs[v] = ci
        return defs

    def _functional_table(self, t):
        """(last var, row -> value) when a table's last column is a var that
        is a function of the other columns, as a lazy ``tie`` posts it (a
        constant column pins it outright)."""
        if t.negated:
            return None
        if len(t.exprs):
            last = t.exprs[-1]
            if (
                len(last.vars) != 1
                or last.coeffs[0] != 1
                or last.offset
                or last.vars[0] < 0
            ):
                return None
            var, w = last.vars[0], len(t.exprs)
        else:
            var, w = t.vars[-1], len(t.vars)
        if w < 2:
            return None
        vals = list(t.values)
        lookup: dict = {}
        for j in range(0, len(vals), w):
            key, out = tuple(vals[j : j + w - 1]), vals[j + w - 1]
            if lookup.setdefault(key, out) != out:
                return None
        return var, lookup

    def _reifications(self, defs):
        """bool var -> the constraint it enforces, for a ``b => lin in D``,
        ``not b => lin in complement(D)`` pair (b is whether lin lies in D),
        or a ``b => and(L)``, ``not b => or(not L)`` pair (b is and(L))."""
        out = {}
        ors = {
            (c.enforcement_literal[0], frozenset(c.bool_or.literals))
            for c in self.cons
            if c.WhichOneof("constraint") == "bool_or"
            and len(c.enforcement_literal) == 1
        }
        for ci, c in enumerate(self.cons):
            if (
                c.WhichOneof("constraint") != "bool_and"
                or len(c.enforcement_literal) != 1
            ):
                continue
            lit = c.enforcement_literal[0]
            if lit < 0 or lit in defs or lit in out:
                continue
            negated = frozenset(-x - 1 for x in c.bool_and.literals)
            if (-lit - 1, negated) in ors:
                out[lit] = ci
        by_expr: dict = {}
        for ci, c in enumerate(self.cons):
            if (
                c.WhichOneof("constraint") != "linear"
                or len(c.enforcement_literal) != 1
            ):
                continue
            key = (tuple(c.linear.vars), tuple(c.linear.coeffs))
            by_expr.setdefault(key, []).append(ci)
        for group in by_expr.values():
            for ci in group:
                lit = self.cons[ci].enforcement_literal[0]
                if lit < 0 or lit in defs or lit in out:
                    continue
                for cj in group:
                    if self.cons[cj].enforcement_literal[0] == -lit - 1 and (
                        _complementary(
                            self.cons[ci].linear.domain, self.cons[cj].linear.domain
                        )
                    ):
                        out[lit] = ci
                        break
        return out

    def _objective(self):
        """(vars, coeffs, offset, report): minimize offset + sum(c * x); the
        user-facing objective is report * that."""
        if self.p.HasField("objective"):
            o = self.p.objective
            return list(o.vars), list(o.coeffs), o.offset, o.scaling_factor or 1.0
        if self.p.HasField("floating_point_objective"):
            o = self.p.floating_point_objective
            sign = -1.0 if o.maximize else 1.0
            return list(o.vars), [c * sign for c in o.coeffs], o.offset * sign, sign
        return None

    # ---- evaluation ----
    def _val(self, a, r):
        x = a[self._ref(r)]
        if x is None:
            return None
        return x if r >= 0 else 1 - x

    def _lin(self, a, e):
        s = e.offset
        for v, c in zip(e.vars, e.coeffs):
            x = self._val(a, v)
            if x is None:
                return None
            s += c * x
        return s

    def _enforced(self, a, c):
        """True / False / None (unknown)."""
        out = True
        for lit in c.enforcement_literal:
            x = self._val(a, lit)
            if x is None:
                out = None
            elif x == 0:
                return False
        return out

    def _check(self, a, ci) -> bool:
        """False only when constraint ci is certainly violated."""
        c = self.cons[ci]
        k = c.WhichOneof("constraint")
        if k == "interval":
            return True
        enf = self._enforced(a, c)
        if enf is False:
            return True
        if k == "no_overlap_2d":
            return self._no_overlap(a, c.no_overlap_2d)
        vals = [
            a[v]
            for v in self.vars_of[ci]
            if not any(self._ref(lit) == v for lit in c.enforcement_literal)
        ]
        if any(x is None for x in vals) or enf is None:
            return True
        if k == "linear":
            lin = c.linear
            s = sum(co * self._val(a, v) for v, co in zip(lin.vars, lin.coeffs))
            return any(
                lo <= s <= hi for lo, hi in zip(lin.domain[::2], lin.domain[1::2])
            )
        if k == "bool_or":
            return any(self._val(a, x) for x in c.bool_or.literals)
        if k == "bool_and":
            return all(self._val(a, x) for x in c.bool_and.literals)
        if k == "at_most_one":
            return sum(self._val(a, x) for x in c.at_most_one.literals) <= 1
        if k == "exactly_one":
            return sum(self._val(a, x) for x in c.exactly_one.literals) == 1
        if k == "element":
            e = c.element
            i = self._lin(a, e.linear_index)
            if not 0 <= i < len(e.exprs):
                return False
            return self._lin(a, e.exprs[i]) == self._lin(a, e.linear_target)
        if k == "table":
            t = c.table
            row = tuple(self._lin(a, x) for x in t.exprs)
            w = len(row)
            vals = list(t.values)
            hit = any(tuple(vals[j : j + w]) == row for j in range(0, len(vals), w))
            return hit != bool(t.negated)
        if k == "int_prod":
            g = c.int_prod
            return math.prod(self._lin(a, x) for x in g.exprs) == self._lin(a, g.target)
        if k == "int_div":
            g = c.int_div
            n, d = (self._lin(a, x) for x in g.exprs)
            q = abs(n) // abs(d) * (1 if (n >= 0) == (d > 0) else -1)
            return q == self._lin(a, g.target)
        if k == "lin_max":
            g = c.lin_max
            return max(self._lin(a, x) for x in g.exprs) == self._lin(a, g.target)
        raise NotImplementedError(k)

    def _box(self, a, ii):
        c = self.cons[ii]
        pres = self._enforced(a, c)
        iv = c.interval
        s, e = self._lin(a, iv.start), self._lin(a, iv.end)
        return pres, s, e

    def _no_overlap(self, a, no):
        boxes = []
        for xi, yi in zip(no.x_intervals, no.y_intervals):
            px, x0, x1 = self._box(a, xi)
            py, y0, y1 = self._box(a, yi)
            if px is False or py is False:
                continue
            if None in (px, py, x0, x1, y0, y1):
                continue
            if x0 == x1 or y0 == y1:  # zero-area boxes never overlap
                continue
            boxes.append((x0, x1, y0, y1))
        for i in range(len(boxes)):
            a0, a1, b0, b1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                c0, c1, d0, d1 = boxes[j]
                if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
                    return False
        return True

    def _compute(self, a, v):
        """Value of a defined var from its definition, or None."""
        ci = self.defs[v]
        c = self.cons[ci]
        k = c.WhichOneof("constraint")
        if v in self._reified and k == "bool_and":
            xs = [self._val(a, x) for x in c.bool_and.literals]
            return None if None in xs else int(all(xs))
        if v in self._reified:
            lin = c.linear
            s = 0
            for r, co in zip(lin.vars, lin.coeffs):
                x = self._val(a, r)
                if x is None:
                    return None
                s += co * x
            return int(
                any(lo <= s <= hi for lo, hi in zip(lin.domain[::2], lin.domain[1::2]))
            )
        if k == "table":
            t = c.table
            cols = (
                [self._lin(a, x) for x in t.exprs]
                if len(t.exprs)
                else [a[x] for x in t.vars]
            )
            row = tuple(cols[:-1])
            return None if None in row else self._lookups[ci].get(row)
        if k == "element":
            e = c.element
            i = self._lin(a, e.linear_index)
            if i is None or not 0 <= i < len(e.exprs):
                return None
            val = self._lin(a, e.exprs[i])
            t = e.linear_target
        elif k in ("int_prod", "int_div", "lin_max"):
            g = getattr(c, k)
            xs = [self._lin(a, x) for x in g.exprs]
            if None in xs:
                return None
            if k == "int_prod":
                val = math.prod(xs)
            elif k == "lin_max":
                val = max(xs)
            else:
                n, d = xs
                if d == 0:
                    return None
                val = abs(n) // abs(d) * (1 if (n >= 0) == (d > 0) else -1)
            t = g.target
        else:  # linear equality
            lin = c.linear
            rest = 0
            coef = None
            for r, co in zip(lin.vars, lin.coeffs):
                if self._ref(r) == v and coef is None:
                    coef = co if r >= 0 else -co
                    if r < 0:
                        rest += co
                    continue
                x = self._val(a, r)
                if x is None:
                    return None
                rest += co * x
            num = lin.domain[0] - rest
            if num % coef:
                return None
            return num // coef
        # target = coeff * v + offset
        co, off = t.coeffs[0], t.offset
        if (val - off) % co:
            return None
        return (val - off) // co

    def _in_domain(self, v, x):
        d = self.domains[v]
        return any(lo <= x <= hi for lo, hi in zip(d[::2], d[1::2]))

    # ---- search ----
    def solve(self):
        order = [v for v in range(self.n) if v not in self.defs]
        # Booleans and small domains first, wide (offsets) last; defined vars
        # after everything, branched only if their definition never resolved.
        order.sort(
            key=lambda v: (
                len(_DomainValues(self.domains[v])) > 2,
                len(_DomainValues(self.domains[v])),
            )
        )
        order += [v for v in range(self.n) if v in self.defs]
        a: list = [None] * self.n
        self.best = None
        self.best_a = None
        self.nodes = 0
        self._dfs(a, order, 0)
        return self.best

    def _propagate(self, a, assigned):
        """Fill defined vars computable now; False on a violation."""
        changed = True
        while changed:
            changed = False
            for v in self.defs:
                if a[v] is None:
                    x = self._compute(a, v)
                    if x is not None:
                        if not self._in_domain(v, x):
                            return False
                        a[v] = x
                        assigned.append(v)
                        changed = True
        return True

    def _consistent(self, a, touched):
        seen = set()
        for v in touched:
            for ci in self.watch[v]:
                if ci not in seen:
                    seen.add(ci)
                    if not self._check(a, ci):
                        return False
        return True

    def _obj(self, a):
        if self.objective is None:
            return 0
        vs, cs, off, _ = self.objective
        return off + sum(c * self._val(a, v) for v, c in zip(vs, cs))

    def objective_value(self):
        return (
            None
            if self.best is None
            else self.best * (self.objective[3] if self.objective else 1)
        )

    def _dfs(self, a, order, i):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise TooBig(self.nodes)
        if i == len(order):
            assert all(x is not None for x in a)
            val = self._obj(a)
            if self.best is None or val < self.best - 1e-9:
                self.best, self.best_a = val, list(a)
            return
        v = order[i]
        if a[v] is not None:
            self._dfs(a, order, i + 1)
            return
        for x in _DomainValues(self.domains[v]):
            a[v] = x
            assigned = [v]
            ok = self._propagate(a, assigned) and self._consistent(a, assigned)
            if ok:
                self._dfs(a, order, i + 1)
            for u in assigned:
                a[u] = None


class _AnnealSearch(_BruteForceSearch):
    """Simulated annealing over the recorded model, scored natively (the
    exhaustive search's checks) on the objective plus weighted violations.

    A violated constraint costs its weight times its degree (1, plus a linear
    row's distance from its domain scaled below 1), so the search can cross
    infeasible plans; only feasible ones are kept. The weights start at the
    median objective coefficient and grow on the constraints still violated
    after ``STALL`` moves without a better score (Feasibility Jump's weights).

    A move reassigns one free var (while infeasible, ``FOCUS`` of the time one
    behind a violated constraint), floods a new value through the tables
    (every other column keeps its value where the table allows it, else takes
    an allowed one: a Kempe-chain recolor), or moves one box in the placement
    order. Up to ``REPAIR_DEPTH`` repairs follow, each the best-scoring single
    change behind a constraint the move broke, then one Metropolis test. The
    temperature cools over the budget from the median objective coefficient
    to ``COOLING`` of it; ``RESTART`` moves without a better plan resume from
    the best one.

    It starts from the hint, else with every objective var at its favourable
    bound; the all-minimum plan is the incumbent when it is feasible. Offsets
    are not searched: in placement order each resident box, an in-place group
    as one over its members' lifetimes, goes to the lowest offset clear of the
    boxes placed before it that share its time, so a packing never overlaps.
    Moves are differential and undone from a journal; ``SPYRE_ANNEAL_VERIFY=N``
    checks the running state against a from-scratch evaluation every N
    moves."""

    REPAIR_DEPTH = 6
    REPAIR_CANDIDATES = 8
    FOCUS = 0.5
    FLOOD = 0.2
    REORDER = 0.1
    COOLING = 1e-3
    BUMP = 1.5
    STALL = 300
    RESTART = 3000

    def anneal(self, seed: int, budget_s: float, start=None) -> bool:
        import os
        import random
        import statistics

        rng = random.Random(seed)
        verify = int(os.environ.get("SPYRE_ANNEAL_VERIFY", "0"))
        self._setup()
        coeffs = [abs(c) for c in self.objective[1]] if self.objective else []
        scale = max(1.0, float(statistics.median(coeffs))) if coeffs else 1.0
        self._w = [scale] * len(self.cons)
        self._wb = dict.fromkeys(self._topo, scale)
        searched = (*self._free_list, *self._packed)
        best = None
        a, order = self._fresh({})
        if not self._nviol:
            best = (self._objv, list(a))
        if start is not None:
            hint = {v: start[v] for v in searched if v < len(start)}
            a, order = self._fresh({v: x for v, x in hint.items() if x is not None})
        else:
            a, order = self._fresh(self._favourable())
        if not self._nviol and (best is None or self._objv < best[0] - 1e-9):
            best = (self._objv, list(a))
        bound = self._objective_bound()
        t0 = time.perf_counter()
        moves = last_better = last_best = 0
        while self._movable or len(order) > 1:
            elapsed = time.perf_counter() - t0
            if elapsed >= budget_s or (best is not None and best[0] <= bound + 1e-9):
                break
            temp = scale * self.COOLING ** (elapsed / budget_s)
            moves += 1
            journal: list = []
            flips: list = []
            state = (self._nviol, self._objv)
            score = self._objv + self._pen
            self._broken: set = set()
            saved_order = None
            r = rng.random()
            if self._tvars and r < self.FLOOD:
                v = rng.choice(self._tvars)
                assign = self._flood(a, v, self._other(v, a[v], rng), rng)
                if not self._try(a, order, assign, journal, flips):
                    continue
            elif len(order) > 1 and (
                not self._movable or self.FLOOD <= r < self.FLOOD + self.REORDER
            ):
                saved_order = list(order)
                i, j = rng.sample(range(len(order)), 2)
                order.insert(j, order.pop(i))
                moved = self._repack(a, order, journal)
                self._apply(a, order, moved, journal, flips, 0, repacked=True)
            else:
                v = self._pick(rng)
                self._try(a, order, [(v, self._other(v, a[v], rng))], journal, flips)
            self._repair(a, order, journal, flips, rng)
            delta = self._objv + self._pen - score
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                if delta < 0:
                    last_better = moves
                if not self._nviol and (best is None or self._objv < best[0] - 1e-9):
                    best = (self._objv, list(a))
                    last_best = moves
            else:
                self._rollback(a, journal, flips, 0, 0, state)
                if saved_order is not None:
                    order[:] = saved_order
            if self._nviol and moves - last_better > self.STALL:
                self._bump()
                last_better = moves
            if best is not None and moves - last_best > self.RESTART:
                a, order = self._fresh({v: best[1][v] for v in searched})
                last_best = moves
            if verify and moves % verify == 0:
                self._verify(a, order)
        self.moves = moves
        if best is not None and not self._feasible(best[1]):
            best = None
        if best is not None:
            self.best, self.best_a = best
        self.proved = best is not None and best[0] <= bound + 1e-9
        return best is not None

    # ---- moves ----

    def _pick(self, rng) -> int:
        """A free var to reassign: while infeasible, ``FOCUS`` of the time one
        behind a violated constraint."""
        if self._nviol and rng.random() < self.FOCUS:
            violated = sorted(self._unsat) + [self.defs[d] for d in sorted(self._bad)]
            sources = [
                v for v in self._sources(rng.choice(violated)) if v in self._moves_ok
            ]
            if sources:
                return rng.choice(sources)
        return rng.choice(self._movable)

    def _other(self, v, x, rng):
        """A value of v's domain other than x."""
        dom = self._doms[v]
        if len(dom) == 2:
            return dom[1] if x == dom[0] else dom[0]
        y = x
        while y == x:
            y = dom[rng.randrange(len(dom))]
        return y

    def _try(self, a, order, assigns, journal, flips) -> bool:
        """Apply ``assigns`` (var, value); False when nothing changes."""
        j0 = len(journal)
        seeds = []
        for v, x in assigns:
            if a[v] != x:
                journal.append((v, a[v]))
                a[v] = x
                seeds.append(v)
        if seeds:
            self._apply(a, order, seeds, journal, flips, j0)
        return bool(seeds)

    def _rollback(self, a, journal, flips, j0, f0, state):
        for u, old in reversed(journal[j0:]):
            a[u] = old
        for kind, key, old in reversed(flips[f0:]):
            (self._set_degree if kind == "deg" else self._set_bad)(key, old)
        del journal[j0:], flips[f0:]
        self._objv = state[1]

    def _flood(self, a, v, x, rng) -> list:
        """v := x, then across every table v is in, each other free column
        keeps its value where the table allows it, else takes an allowed one,
        and floods on from there."""
        assign = {v: x}
        queue = [v]
        while queue:
            u = queue.pop()
            for ci in self._tcols.get(u, ()):
                cols, index = self._table(ci)
                exprs = self.cons[ci].table.exprs
                cur = [
                    assign.get(c, a[c]) if c is not None else self._lin(a, exprs[k])
                    for k, c in enumerate(cols)
                ]
                for k, w in enumerate(cols):
                    if w is None or w in assign or w not in self._free:
                        continue
                    allowed = index[k].get(tuple(cur[:k] + cur[k + 1 :]), ())
                    if not allowed or cur[k] in allowed:
                        continue
                    cur[k] = assign[w] = allowed[rng.randrange(len(allowed))]
                    queue.append(w)
        return [(w, y) for w, y in assign.items() if y != a[w]]

    def _repair(self, a, order, journal, flips, rng):
        """While a constraint the move broke is still violated, apply the
        best-scoring single change behind it if that lowers the score."""
        for _ in range(self.REPAIR_DEPTH):
            broken = [ci for ci in self._broken if self._deg[ci]]
            if not broken:
                return
            ci = rng.choice(broken)
            base = self._objv + self._pen
            pick = None
            for cand in self._repair_candidates(a, ci, rng):
                j0, f0, state = len(journal), len(flips), (self._nviol, self._objv)
                was_broken = set(self._broken)
                if self._try(a, order, cand, journal, flips):
                    score = self._objv + self._pen
                    if pick is None or score < pick[0]:
                        pick = (score, cand)
                    self._rollback(a, journal, flips, j0, f0, state)
                self._broken = was_broken
            if pick is None or pick[0] >= base:
                self._broken.discard(ci)
                continue
            self._try(a, order, pick[1], journal, flips)

    def _repair_candidates(self, a, ci, rng) -> list:
        """Single changes that might fix ci: for a table, each value one free
        column may take given the others; else a free var behind ci at
        another value."""
        c = self.cons[ci]
        if c.WhichOneof("constraint") == "table" and not c.table.negated:
            cols, index = self._table(ci)
            exprs = c.table.exprs
            cur = [
                a[v] if v is not None else self._lin(a, exprs[k])
                for k, v in enumerate(cols)
            ]
            out = [
                [(v, x)]
                for k, v in enumerate(cols)
                if v is not None and v in self._free
                for x in index[k].get(tuple(cur[:k] + cur[k + 1 :]), ())
                if x != cur[k]
            ]
            if out:
                return rng.sample(out, min(len(out), self.REPAIR_CANDIDATES))
        sources = [v for v in self._sources(ci) if v in self._moves_ok]
        return [
            [(v, self._other(v, a[v], rng))]
            for v in rng.sample(sources, min(len(sources), self.REPAIR_CANDIDATES))
        ]

    def _bump(self):
        """Raise the weights of the constraints violated now."""
        for ci in self._unsat:
            self._pen += self._w[ci] * (self.BUMP - 1) * self._deg[ci]
            self._w[ci] *= self.BUMP
        for d in self._bad:
            self._pen += self._wb[d] * (self.BUMP - 1) * self._bdeg[d]
            self._wb[d] *= self.BUMP

    def _sources(self, ci):
        """The free vars constraint ci depends on, through definitions."""
        if ci not in self._sources_of:
            seen, todo, out = set(), list(self.vars_of[ci]), []
            while todo:
                u = todo.pop()
                if u in seen:
                    continue
                seen.add(u)
                if u in self._free:
                    out.append(u)
                elif u in self._pos:
                    todo.extend(self._deps[u])
            self._sources_of[ci] = sorted(out)
        return self._sources_of[ci]

    # ---- structure ----

    def _setup(self):
        self._boxes, self._ties, self._decoded = self._placement()
        self._doms = [_DomainValues(self.domains[v]) for v in range(self.n)]
        self._packed = {b[0] for b in self._boxes if b[0] not in self.defs}
        self._order_definitions()
        self._free_list = [
            v
            for v in range(self.n)
            if (v not in self.defs or v in self._floating) and v not in self._packed
        ]
        self._free = set(self._free_list)
        self._movable = [v for v in self._free_list if len(self._doms[v]) > 1]
        self._moves_ok = set(self._movable)
        self._sources_of: dict = {}
        self._tables: dict = {}
        self._rows: dict = {}
        self._pack_inputs = self._packing_inputs()
        self._box_exprs = [
            tuple(self._compiled(e.vars, e.coeffs, e.offset) for e in (xs, xe, ys))
            for _off, _lit, xs, xe, ys in self._boxes
        ]
        self._obj_terms: dict = {}
        if self.objective:
            for ref, c in zip(self.objective[0], self.objective[1]):
                self._obj_terms.setdefault(self._ref(ref), []).append((ref, c))
        self._tcols: dict = {}
        for ci, c in enumerate(self.cons):
            if c.WhichOneof("constraint") == "table" and not c.table.negated:
                for v in self._table(ci)[0]:
                    if v is not None and v in self._free:
                        self._tcols.setdefault(v, []).append(ci)
        self._tvars = [v for v in self._tcols if v in self._moves_ok]

    def _table(self, ci):
        """(columns, index): each column's var (None for any other
        expression) and, per column, the values the rows allow it keyed by
        the other columns' values."""
        if ci not in self._tables:
            t = self.cons[ci].table
            if len(t.exprs):
                cols = [
                    x.vars[0]
                    if len(x.vars) == 1
                    and x.coeffs[0] == 1
                    and not x.offset
                    and x.vars[0] >= 0
                    else None
                    for x in t.exprs
                ]
            else:
                cols = list(t.vars)
            w = len(cols)
            rows = [tuple(t.values[r : r + w]) for r in range(0, len(t.values), w)]
            index = {}
            for j in range(w):
                allowed: dict = {}
                for row in rows:
                    allowed.setdefault(row[:j] + row[j + 1 :], []).append(row[j])
                index[j] = allowed
            self._tables[ci] = (cols, index)
        return self._tables[ci]

    def _favourable(self) -> dict:
        """Each free var in the objective at the bound that lowers it."""
        out = {}
        for v in self._free_list:
            for ref, c in self._obj_terms.get(v, ()):
                dom = self._doms[v]
                out[v] = dom[-1] if (c < 0) == (ref >= 0) else dom[0]
        return out

    def _objective_bound(self):
        """The objective's least value over the domains alone."""
        if self.objective is None:
            return 0
        vs, cs, offset, _ = self.objective
        low = offset
        for r, c in zip(vs, cs):
            dom = self.domains[self._ref(r)]
            ends = (dom[0], dom[-1]) if r >= 0 else (1 - dom[0], 1 - dom[-1])
            low += min(c * ends[0], c * ends[1])
        return low

    def _order_definitions(self):
        """Each defined var after the vars its definition reads (Kahn). A var
        left in a cycle floats: the search moves it like a free one."""
        deps = {
            d: {u for u in self.vars_of[ci] if u != d} for d, ci in self.defs.items()
        }
        waiting = {d: sum(u in self.defs for u in ds) for d, ds in deps.items()}
        readers: dict = {}
        for d, ds in deps.items():
            for u in ds:
                readers.setdefault(u, []).append(d)
        ready = sorted(d for d, k in waiting.items() if k == 0)
        topo = []
        while ready:
            d = ready.pop()
            topo.append(d)
            for w in readers.get(d, ()):
                waiting[w] -= 1
                if waiting[w] == 0:
                    ready.append(w)
        self._topo = topo
        self._deps = deps
        self._pos = {d: i for i, d in enumerate(topo)}
        self._floating = set(self.defs) - set(topo)
        self._readers = {
            u: [d for d in ds if d in self._pos] for u, ds in readers.items()
        }

    def _packing_inputs(self):
        """Vars the packer reads: presences, lifetimes, sizes, merge
        literals, and offsets it does not choose."""
        out = set()
        for off, lit, xs, xe, ys in self._boxes:
            if lit is not None:
                out.add(self._ref(lit))
            for e in (xs, xe, ys):
                out |= self._expr_vars(e)
            if off not in self._packed:
                out.add(off)
        for ties in self._ties.values():
            out |= {self._ref(lit) for lit, _ in ties}
        return out

    def _placement(self):
        """The no-overlap boxes as (offset var, presence lit, x start, x end,
        y size); each offset's merge ties, (lit, other offset) for a
        ``lit => off == other``; and the no-overlap constraints whose every
        box the packer places."""
        boxes = []
        decoded = set()
        for ci, c in enumerate(self.cons):
            if c.WhichOneof("constraint") != "no_overlap_2d":
                continue
            no = c.no_overlap_2d
            whole = True
            for xi, yi in zip(no.x_intervals, no.y_intervals):
                x, y = self.cons[xi], self.cons[yi]
                s = y.interval.start
                if len(s.vars) != 1 or s.coeffs[0] != 1 or s.offset or s.vars[0] < 0:
                    whole = False
                    continue
                whole = whole and s.vars[0] not in self.defs
                lit = y.enforcement_literal[0] if len(y.enforcement_literal) else None
                boxes.append(
                    (s.vars[0], lit, x.interval.start, x.interval.end, y.interval.size)
                )
            if whole:
                decoded.add(ci)
        offsets = {b[0] for b in boxes}
        ties: dict = {}
        for c in self.cons:
            if (
                c.WhichOneof("constraint") != "linear"
                or len(c.enforcement_literal) != 1
            ):
                continue
            lin = c.linear
            if (
                len(lin.vars) == 2
                and sorted(lin.coeffs) == [-1, 1]
                and list(lin.domain) == [0, 0]
                and all(v >= 0 and v in offsets for v in lin.vars)
            ):
                p, q = lin.vars
                lit = c.enforcement_literal[0]
                ties.setdefault(p, []).append((lit, q))
                ties.setdefault(q, []).append((lit, p))
        return boxes, ties, decoded

    # ---- evaluation ----

    def _compiled(self, refs, coeffs, offset) -> tuple:
        """A linear expression as (offset, ((var, coeff), ...)), a negated
        literal folded into the offset."""
        terms = []
        for r, c in zip(refs, coeffs):
            if r >= 0:
                terms.append((r, c))
            else:
                offset += c
                terms.append((-r - 1, -c))
        return offset, tuple(terms)

    @staticmethod
    def _value(a, compiled):
        offset, terms = compiled
        return offset + sum(c * a[v] for v, c in terms)

    def _degree(self, a, ci) -> float:
        """0 when constraint ci holds, else 1, plus for a linear row its
        distance from the domain scaled into [0, 1)."""
        c = self.cons[ci]
        k = c.WhichOneof("constraint")
        if k == "interval" or ci in self._decoded or self._enforced(a, c) is False:
            return 0.0
        if k == "linear":
            if ci not in self._rows:
                lin = c.linear
                dom = list(lin.domain)
                finite = [abs(x) for x in dom if abs(x) < 2**62]
                self._rows[ci] = (
                    self._compiled(lin.vars, lin.coeffs, 0),
                    list(zip(dom[::2], dom[1::2])),
                    max(1, min(finite)) if finite else 1,
                )
            row, spans, scale = self._rows[ci]
            s = self._value(a, row)
            gap = min(lo - s if s < lo else s - hi if s > hi else 0 for lo, hi in spans)
            return 0.0 if gap == 0 else 1.0 + gap / (gap + scale)
        if k == "bool_and":
            return float(sum(not self._val(a, x) for x in c.bool_and.literals))
        if k == "at_most_one":
            return float(
                max(0, sum(self._val(a, x) for x in c.at_most_one.literals) - 1)
            )
        if k == "exactly_one":
            return float(abs(sum(self._val(a, x) for x in c.exactly_one.literals) - 1))
        return 0.0 if self._check(a, ci) else 1.0

    def _bad_degree(self, d, x) -> float:
        """A defined var's degree: 0 inside its domain, else 1 plus its scaled
        distance from it."""
        dom = self.domains[d]
        gap = min(
            lo - x if x < lo else x - hi if x > hi else 0
            for lo, hi in zip(dom[::2], dom[1::2])
        )
        return (
            0.0 if gap == 0 else 1.0 + gap / (gap + max(1, abs(dom[0]), abs(dom[-1])))
        )

    def _feasible(self, a) -> bool:
        """Every constraint and every domain holds for ``a``, from scratch."""
        return all(self._check(a, ci) for ci in range(len(self.cons))) and all(
            self._in_domain(v, a[v]) for v in range(self.n)
        )

    def _define(self, a, d):
        x = self._compute(a, d)
        return self._doms[d][0] if x is None else x

    def _evaluate(self, a, order):
        """Every defined var and offset, from scratch."""
        for d in self._topo:
            a[d] = self._define(a, d)
        self._pack(a, order)
        for d in self._topo:
            a[d] = self._define(a, d)

    def _fresh(self, assign):
        """The state from scratch: the searched vars at their least value
        unless assigned, the rest computed, with its degrees and score. First
        fit in offset order never places a box above its assigned offset, so
        an assigned plan still packs."""
        a = [None] * self.n
        for v in (*self._free_list, *self._packed):
            a[v] = self._doms[v][0]
        for v, x in assign.items():
            a[v] = x
        order = sorted(
            (i for i, b in enumerate(self._boxes) if b[0] in self._packed),
            key=lambda i: a[self._boxes[i][0]],
        )
        self._evaluate(a, order)
        self._deg = [self._degree(a, ci) for ci in range(len(self.cons))]
        self._bdeg = {d: self._bad_degree(d, a[d]) for d in self._topo}
        self._unsat = {ci for ci, g in enumerate(self._deg) if g}
        self._bad = {d for d, g in self._bdeg.items() if g}
        self._nviol = len(self._unsat) + len(self._bad)
        self._pen = sum(w * g for w, g in zip(self._w, self._deg)) + sum(
            self._wb[d] * g for d, g in self._bdeg.items()
        )
        self._objv = self._obj(a)
        self._broken = set()
        return a, order

    def _forward(self, a, seeds, journal):
        """Recompute, in definition order, the defined vars downstream of
        ``seeds``; return every var that changed."""
        import heapq

        heap: list = []
        queued: set = set()
        for u in seeds:
            for d in self._readers.get(u, ()):
                if d not in queued:
                    queued.add(d)
                    heapq.heappush(heap, (self._pos[d], d))
        changed = list(seeds)
        while heap:
            _, d = heapq.heappop(heap)
            x = self._define(a, d)
            if x == a[d]:
                continue
            journal.append((d, a[d]))
            a[d] = x
            changed.append(d)
            for w in self._readers.get(d, ()):
                if w not in queued:
                    queued.add(w)
                    heapq.heappush(heap, (self._pos[w], w))
        return changed

    def _repack(self, a, order, journal):
        offs = [self._boxes[i][0] for i in order]
        old = [a[o] for o in offs]
        self._pack(a, order)
        moved = [(o, x) for o, x in zip(offs, old) if a[o] != x]
        journal.extend(moved)
        return [o for o, _ in moved]

    def _apply(self, a, order, seeds, journal, flips, j0, repacked=False):
        """Settle what ``seeds`` (changed since journal[j0]) reach and update
        the degrees, the score and the constraints the change broke."""
        changed = self._forward(a, seeds, journal)
        if not repacked and self._pack_inputs.intersection(changed):
            moved = self._repack(a, order, journal)
            # An offset never feeds the packer back, so one pass settles it.
            changed += self._forward(a, moved, journal)
        first: dict = {}
        for u, old in journal[j0:]:
            first.setdefault(u, old)
        touched: set = set()
        for u, old in first.items():
            if a[u] == old:
                continue
            for ref, c in self._obj_terms.get(u, ()):
                self._objv += (c if ref >= 0 else -c) * (a[u] - old)
            if u in self._pos:
                deg = self._bad_degree(u, a[u])
                if deg != self._bdeg[u]:
                    flips.append(("bad", u, self._bdeg[u]))
                    self._set_bad(u, deg)
            touched.update(self.watch[u])
        for ci in touched:
            deg = self._degree(a, ci)
            old = self._deg[ci]
            if deg != old:
                flips.append(("deg", ci, old))
                self._set_degree(ci, deg)
                if not old:
                    self._broken.add(ci)

    def _set_degree(self, ci, deg):
        old = self._deg[ci]
        self._pen += self._w[ci] * (deg - old)
        self._deg[ci] = deg
        if deg and not old:
            self._unsat.add(ci)
            self._nviol += 1
        elif old and not deg:
            self._unsat.discard(ci)
            self._nviol -= 1

    def _set_bad(self, d, deg):
        old = self._bdeg[d]
        self._pen += self._wb[d] * (deg - old)
        self._bdeg[d] = deg
        if deg and not old:
            self._bad.add(d)
            self._nviol += 1
        elif old and not deg:
            self._bad.discard(d)
            self._nviol -= 1

    def _verify(self, a, order):
        """The running state equals a from-scratch evaluation, and every
        no-overlap the packer decides holds."""
        ref = list(a)
        self._evaluate(ref, order)
        assert ref == a, [
            (self.p.variables[v].name, a[v], ref[v])
            for v in range(self.n)
            if a[v] != ref[v]
        ][:10]
        assert self._deg == [self._degree(ref, ci) for ci in range(len(self.cons))]
        assert self._bdeg == {d: self._bad_degree(d, ref[d]) for d in self._topo}
        pen = sum(w * g for w, g in zip(self._w, self._deg)) + sum(
            self._wb[d] * g for d, g in self._bdeg.items()
        )
        assert abs(pen - self._pen) <= 1e-6 * max(1.0, abs(pen)), (pen, self._pen)
        assert abs(self._obj(ref) - self._objv) < 1e-6, (self._obj(ref), self._objv)
        assert all(self._check(ref, ci) for ci in self._decoded)

    def _pack(self, a, order):
        """Offsets for the boxes in ``order`` (``_C.pack_boxes``). An in-place
        group packs as one box over its members' lifetimes at their largest
        size (members are adjacent in time and equal in size), so no packing
        overlaps; a present box outside ``order`` keeps its offset as an
        obstacle."""
        from torch_spyre import _C

        n = len(self._boxes)
        index = {b[0]: i for i, b in enumerate(self._boxes)}
        packed = set(order)
        x0, x1, size, present, pinned = [], [], [], [], []
        for i, (off, lit, *_exprs) in enumerate(self._boxes):
            on = lit is None or bool(self._val(a, lit))
            xs, xe, ys = self._box_exprs[i]
            x0.append(self._value(a, xs) if on else 0)
            x1.append(self._value(a, xe) if on else 0)
            size.append(self._value(a, ys) if on else 0)
            present.append(on)
            pinned.append(a[off] if i not in packed and a[off] is not None else -1)
        group = list(range(n))

        def root(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        for i, (off, *_rest) in enumerate(self._boxes):
            if present[i]:
                for lit, q in self._ties.get(off, ()):
                    j = index[q]
                    if present[j] and self._val(a, lit):
                        group[root(i)] = root(j)
        gx0, gx1, gsize, gpinned = list(x0), list(x1), list(size), list(pinned)
        gpresent = [False] * n
        for i in range(n):
            if present[i]:
                g = root(i)
                if gpresent[g]:
                    gx0[g] = min(gx0[g], x0[i])
                    gx1[g] = max(gx1[g], x1[i])
                    gsize[g] = max(gsize[g], size[i])
                    gpinned[g] = max(gpinned[g], pinned[i])
                else:
                    gpresent[g] = True
                    gx0[g], gx1[g], gsize[g], gpinned[g] = (
                        x0[i],
                        x1[i],
                        size[i],
                        pinned[i],
                    )
        gorder, seen = [], set()
        for i in order:
            g = root(i)
            if present[i] and gpinned[g] < 0 and g not in seen:
                seen.add(g)
                gorder.append(g)
        offsets = _C.pack_boxes(
            gorder, gx0, gx1, gsize, gpresent, gpinned, [[] for _ in range(n)]
        )
        for i in order:
            a[self._boxes[i][0]] = offsets[root(i)] if present[i] else 0


class Annealing(_SolvedRecording):
    """Simulated annealing over the recorded model, scored natively."""

    def __init__(self, seed: int = 0, budget_s: float = 10.0, model=None):
        super().__init__(model)
        self.seed = seed
        self.budget_s = budget_s

    def Solve(self, objective: bool = False) -> int:
        search = _AnnealSearch(self._proto(), max_nodes=0)
        t = time.perf_counter()
        found = search.anneal(self.seed, self.budget_s, start=self._values)
        self._wall = time.perf_counter() - t
        if not found:
            self._status = cp_model.UNKNOWN
            return self._status
        self._values = search.best_a
        self._objective_value = search.objective_value()
        self._status = cp_model.OPTIMAL if search.proved else cp_model.FEASIBLE
        return self._status


def from_env() -> CpSat:
    """The backend ``SPYRE_LX_BACKEND`` names: ``cpsat`` (default),
    ``mip:<SCIP|HIGHS|CBC>``, ``bruteforce`` or ``anneal[:<budget s>]``."""
    import os

    choice = os.environ.get("SPYRE_LX_BACKEND", "cpsat")
    if choice.startswith("mip"):
        return Mip(choice.partition(":")[2] or "SCIP")
    if choice == "bruteforce":
        return BruteForce()
    if choice.startswith("anneal"):
        return Annealing(budget_s=float(choice.partition(":")[2] or 10))
    return CpSat()
