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

"""Joint core-division + LX-placement solver built on OR-Tools CP-SAT
(``config.layout_solver == "cpsat"``).

Selects each buffer's core division and its LX scratchpad placement in one
constraint model over :class:`CoreDivisionBuffer`s:

* **Joint core-division.** ``size`` is the *total* device footprint; a ``div``
  var indexes the buffer's candidate divisions (from
  ``enumerate_work_division_candidates``) and ``AddElement`` ties the chosen
  index to the per-core footprint (``eff_size = size / output_partition``) and
  total core usage (``cores = cores_used``, including any reduction-axis split).
* **Slicing-match residency gate.** A resident buffer's division must induce the
  same per-core slicing as *every* consumer's, using the precomputed
  ``cd_parent_matches`` pairs over the ``parents`` (producer/consumer) edges; a
  buffer with no consumer, or a consumer with no compatible pair, can never
  reside (``_CoreDivisionBufferWithCpVars.constrain_residency``).
* **Placement** is a global ``AddNoOverlap2D`` over optional rectangles
  (``[start_time, end_time) x [offset, offset + eff_size)``, present iff
  resident). In-place reuse (``in_place_parents`` -> per-edge ``merge_vars``) is
  encoded by *shortening the child's lifetime* by the single handoff tick when
  the merge fires, so the parent and its in-place child abut in time and may
  legally share an offset; the single-tick-overlap invariant
  (``_check_in_place_relationships``) makes this exact. The parent keeps its
  full lifetime, so the footprint above a smaller child stays protected on the
  handoff tick (``_add_no_overlap_2d``).
* **Objective** (lexicographic, in ``_run``; each level locks the prior
  level's optimum as a constraint before optimizing the next). *Residency
  is the hard priority.* It first minimizes total **HBM transfer traffic** via
  ``spill_cost(b) * (1 - in_buffer)`` -- the *differential* traffic a spill adds
  over residency (resident buffers contribute 0). An intermediate costs
  ``(num_consumers + 1) * size`` (the producer's HBM write, which residency turns
  into a free LX write, plus one re-read per consumer); a graph input drops the
  producer write it never had and the clone-in read residency cannot avoid
  (``(num_consumers - 1) * size``); a graph output drops its unavoidable
  write-out (``num_consumers * size``). This puts as much in LX as possible
  and chooses whatever division serves that (even no split, if that is what lets
  a buffer match its consumers and reside). It then *holds that residency
  optimum* and maximizes total core usage (``sum_b cores_b``) so every buffer --
  resident or spilled, the latter free of the slicing gate -- takes its most
  parallel division. Parallelism never costs a spill. It finally *holds the
  parallelism optimum* and breaks the remaining ties toward a **balanced**
  division by minimizing the summed squared split factors
  (``sum_b sum_axis split**2``): among divisions that use the same number of
  cores, one spreading the split across more axes with smaller factors scores
  lower than one that hammers a single axis (``2x2`` over ``4x1``). This only
  refines the division the allocator commits -- it can never spill a buffer or
  reduce its core count. Op shape is not yet visible to the solver, so this is a
  proxy for balance rather than a full cost model.

After the solve, ``_justify`` slides each in-place-merged placement unit down to
the lowest free address, squeezing out float gaps the search leaves. It coarsens
a merged unit to one rectangle over the union of its members' lifetimes, which is
conservative enough that the squeeze can occasionally need more room than the
solver's own answer; when it would not fit, the solver's offsets are kept.

The same model also serves plain :class:`LifetimeBoundBuffer`s via
``plan_layout`` (the ``MemoryPlanSolver`` contract the placement-only allocator
calls). Those buffers carry no candidate divisions, so the division-dependent
pieces -- per-core sizing, the slicing-match gate, the merge division gate and
the parallelism and balance objectives -- simply drop out: the
footprint is the buffer's ``size`` and the solve reduces to minimising HBM
traffic under the 2D no-overlap with in-place reuse. Residency is then gated
only by capacity and by the allocator's own ``residency_reason`` bars (which
both paths honour, since that field lives on the base buffer). That
specialisation lives on the buffer wrappers (``_LifetimeBufferWithCpVars`` and
its joint subclass ``_CoreDivisionBufferWithCpVars``), so the solver methods
below are written once against whichever wrapper ``_wrap`` chose.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, cast
import sympy
from sympy.printing.printer import Printer
import torch


if TYPE_CHECKING:
    from ortools.sat.python import cp_model, cp_model_helper
else:
    try:
        from ortools.sat.python import cp_model, cp_model_helper

    except ImportError:  # pragma: no cover - exercised only when ortools is absent
        cp_model = None

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    ceil_div,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    SolveError,
    BufferType,
    _check_in_place_relationships,
)
from torch_spyre._inductor import config

__all__ = ["CpSatLayoutSolver"]

logger = logging.getLogger(__name__)

# Drop cause for a buffer the solver chose to spill (rather than one pinned out
# up front by _add_core_division): it fit but residency gave no benefit, or
# there was no room once higher-value buffers were placed. Shared so the DEBUG
# log and the reasons surfaced to the allocator agree.
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"

# Buffer type the wrapper carries: the base placement wrapper holds any
# LifetimeBoundBuffer; the joint subclass binds this to CoreDivisionBuffer.
_BufT = TypeVar("_BufT", bound=LifetimeBoundBuffer)

# constant to scale log of core split. error ~0.5%
_CORE_LOG_SCALE = 32.0
# constant to scale inverse of core split. error ~1%
_CORE_INV_SCALE = 1024


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    original_offset: int  # offset the solver chose, before bottom-justify
    justified_offset: int = 0  # final justified offset


def _gate_divisions(model, compatible, src_div, dst_div, enforce_lit) -> None:
    """Enforce, when ``enforce_lit`` is true, that ``(src_div, dst_div)`` is
    one of the ``compatible`` (i, j) pairs. With no compatible pairs the
    relation is unsatisfiable, so ``enforce_lit`` is forced false."""
    if not compatible:
        model.Add(enforce_lit == 0)
        return
    pair_lits = []
    for i, j in compatible:
        lit = model.NewBoolVar("")
        model.Add(src_div == i).OnlyEnforceIf(lit)
        model.Add(dst_div == j).OnlyEnforceIf(lit)
        pair_lits.append(lit)
    model.AddBoolOr(pair_lits).OnlyEnforceIf(enforce_lit)


@dataclass
class _LifetimeBufferWithCpVars(Generic[_BufT]):
    """A :class:`LifetimeBoundBuffer` bundled with the CP-SAT variables the
    solver creates for it, so one object flows through the solve instead of a
    buffer list shadowed by a parallel ``name -> {var}`` dict.

    This is the *placement-only* wrapper backing :meth:`plan_layout`: the
    buffer's core division is already fixed upstream, so its footprint is the
    constant ``size`` (which the 2D no-overlap and capacity constraints accept
    wherever a var would go) and there is no division to choose. Every
    division-aware hook below is therefore a no-op or a fixed-size answer;
    :class:`_CoreDivisionBufferWithCpVars` overrides them to add the joint
    core-division model. Keeping the hooks on the wrapper is what lets ``_run``
    and its helpers serve both entry points unchanged.

    The buffer spans ``[buffer.start_time, buffer.end_time)``; the vars encode
    where (``offset``) and whether (``in_buffer``) it resides in LX.
    ``merge_vars`` maps each in-place parent name to the merge bool for that
    parent->this edge.

    CP-SAT variables must be created against a model, so this wrapper takes the
    model and the unit capacity ``M`` and creates only the variables here; the
    constraints tying them together are added by the solver methods."""

    buffer: _BufT
    model: "cp_model.CpModel"
    capacity_units: int

    def __post_init__(self):
        b = self.buffer
        m = self.model
        M = self.capacity_units
        self.name = b.name
        self.start_time = b.start_time
        self.end_time = b.end_time

        self.in_buffer = m.new_bool_var(f"in_buffer_{b.name}")
        # offset domain [0, M-1]; the resident => offset+eff_size<=M bound is
        # added in the in-place relaxation pass.
        self.offset = m.new_int_var(0, max(0, M - 1), f"off_{b.name}")
        # Fixed footprint -- no division to pick, so a constant stands in for
        # the joint solver's eff_size var.
        self.eff_size: object = b.size
        # Nothing to parallelise without candidate divisions; ``_run`` skips
        # the parallelism step when no buffer offers a core-usage term.
        self.cores = None
        self.merge_vars = {
            parent: m.new_bool_var(f"merge_{parent}_{b.name}")
            for parent in b.in_place_parents
        }
        self.core_cost = None

    # -- producer/consumer edges (joint model only; none when division-fixed) --
    @property
    def parents(self) -> list[str]:
        return []

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return []

    # ------------------------------ residency ------------------------------
    def spill_cost(self) -> int:
        """Differential HBM traffic a spill adds over residency: the reads
        residency would have served from LX plus the producer's write, which
        residency turns into a free LX write. A graph input has no producer write
        to save; a graph output's write-out is unavoidable either way, so it too
        cancels. Both cases are exactly ``boundary != Intermediate`` -- for a
        plain :class:`LifetimeBoundBuffer`, whose boundary is not tracked,
        ``first_use_is_read`` marks the same distinction for inputs.

        An input's first read is the clone-in that pinning cannot avoid, so it is
        not one of the reads residency serves and is discounted from
        ``read_count`` (which counts the buffer's reads, not the savings). For a
        computed buffer the first use is the write and ``read_count`` already
        excludes it, hence the discount is keyed on ``first_use_is_read``."""
        b = self.buffer
        boundary = getattr(b, "boundary", None)
        is_intermediate = (
            boundary == BufferType.Intermediate
            if boundary is not None
            else not b.first_use_is_read
        )
        reads_served = b.read_count - (1 if b.first_use_is_read else 0)
        return (reads_served + (1 if is_intermediate else 0)) * b.size

    def constrain_residency(self, model, kids, bufs) -> None:
        """Placement-only: any buffer may reside, so there is no slicing gate."""

    def constrain_merge(self, model, parent: "_LifetimeBufferWithCpVars", edge) -> None:
        """Extra conditions on an active in-place merge. None when the division
        is fixed: ``_check_in_place_relationships`` already checks the child
        fits in the parent's slot."""

    # ------------------------------- extract -------------------------------
    def footprint(self, solver: "cp_model.CpSolver") -> int:
        return self.buffer.size

    def record_division(self, solver: "cp_model.CpSolver") -> None:
        """Write the chosen division back onto the buffer (nothing to record
        when the division is fixed)."""


@dataclass
class _CoreDivisionBufferWithCpVars(_LifetimeBufferWithCpVars[CoreDivisionBuffer]):
    """The joint-model wrapper: a :class:`CoreDivisionBuffer` plus the vars for
    its chosen core division (``division``), the per-core footprint that
    division implies (``eff_size``) and its total core usage (``cores`` =
    ``cores_used``, including any reduction-axis split).

    On top of the base placement vars it supplies the division-aware pieces of
    the model: the slicing-match residency gate, the division gate on an
    in-place merge, and the edge-counted spill cost. The ``buffer`` field is
    narrowed to :class:`CoreDivisionBuffer` via the base's type parameter."""

    def __post_init__(self):
        super().__post_init__()
        b = self.buffer
        m = self.model

        per_core = [ceil_div(b.size, cd.output_partition) for cd in b.core_divisions]
        # Total cores the op runs on under each division -- includes any
        # reduction-axis split, so a reduction-parallel division counts its full
        # parallelism (``output_partition`` alone would score it as 1 core).
        cores_used = [cd.cores_used for cd in b.core_divisions]
        # Balance heuristic: the sum of squared per-axis split factors. For a
        # fixed core count (product of the factors, held at the parallelism
        # optimum) this is smallest when the split is spread across more axes
        # with smaller factors, so minimizing it favours a balanced division
        # over one that hammers a single axis (e.g. 2x2 over 4x1, both four
        # cores). The two split namespaces are coeff-keyed and iterated
        # separately -- a union would silently drop a factor whose coeff
        # collides across the two.
        core_cost = [
            sum(
                split**2
                for split in list(cd.output_splits.values())
                + list(cd.reduction_splits.values())
            )
            for cd in b.core_divisions
        ]
        self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
        self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")
        self.core_cost = m.new_int_var(0, max(core_cost), f"core_cost_{b.name}")
        # total cores this op uses under the chosen div
        self.cores = m.new_int_var(0, max(cores_used), f"occ_{b.name}")

        sym_core_divs = b.sym_core_divs

        cp_core_divs = ({}, {})
        cp_core_divs_raw = ({}, {})
        for i, split_type in enumerate(["output_splits", "reduction_splits"]):
            splits = sym_core_divs[i]
            for key, symbol in splits.items():
                assert isinstance(symbol, sympy.Symbol)
                raw = [getattr(cd, split_type).get(key, 1) for cd in b.core_divisions]
                cp_var = m.new_int_var(1, config.sencores, f"{symbol.name}")
                m.add_element(self.division, raw, cp_var)
                cp_core_divs[i][key] = cp_var
                cp_core_divs_raw[i][key] = raw

        self.cp_core_divs = cp_core_divs
        self.cp_core_divs_raw = cp_core_divs_raw

        # tie per-core footprint (output split only) and total core usage to the
        # chosen division index
        m.add_element(self.division, per_core, self.eff_size)
        m.add_element(self.division, cores_used, self.cores)
        m.add_element(self.division, core_cost, self.core_cost)

    @property
    def parents(self) -> list[str]:
        return self.buffer.parents

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return self.buffer.cd_parent_matches.get(parent, [])

    def constrain_residency(self, model, kids, bufs) -> None:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs.

        This is the part of residency that genuinely depends on the solver's
        free variables, so it stays here as a constraint. The precomputable
        parts -- having no LX reader at all, or a consumer with no compatible
        pair -- are decided by the allocator and arrive as ``read_count`` /
        ``residency_reason``. A consumer with no compatible pair still lands
        correctly if it slips through: ``_gate_divisions`` forces ``in_buffer``
        false when the pair list is empty."""
        for child, compatible in kids:
            _gate_divisions(
                model, compatible, self.division, bufs[child].division, self.in_buffer
            )

    def constrain_merge(self, model, parent, edge) -> None:
        """An active merge means the child reuses the parent's exact per-core
        storage, so their chosen divisions must have equal per-core footprints
        and must induce the same per-core slicing of that storage (the
        ``cd_parent_matches`` pairs; no pairs => merge forbidden)."""
        model.add(self.eff_size == parent.eff_size).OnlyEnforceIf(edge)
        _gate_divisions(
            model,
            self.match_pairs(parent.name),
            parent.division,
            self.division,
            edge,
        )

    def footprint(self, solver: "cp_model.CpSolver") -> int:
        t = self.buffer
        cd = t.core_divisions[solver.Value(self.division)]
        return ceil_div(t.size, cd.output_partition)

    def record_division(self, solver: "cp_model.CpSolver") -> None:
        self.buffer.chosen_division = solver.Value(self.division)


class _SympyExprToCpSat(Printer):
    """Translates a sympy cost expression into an OR-Tools CP-SAT expression
    over an existing ``sympy symbol -> CP-SAT var`` mapping.
    """

    def __init__(self, model: "cp_model.CpModel", sym_map: dict) -> None:
        self._model = model
        self._count = 0
        self._sym_map = sym_map
        super().__init__()

    def convert(self, cost_expr: sympy.Expr) -> "cp_model.LinearExpr":
        """Return the CP-SAT expression equivalent to ``cost_expr`` under
        ``sym_map`` (``sympy symbol -> CP-SAT var``)."""
        logger.debug("[CP-SAT layout solver] cost expr (raw): %s", cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.floor,
            lambda e: e.args[0],
        )
        cost_expr = sympy.expand(cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.log,
            lambda e: self._log_min(e),
        )
        cost_expr = sympy.expand(cost_expr)
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.log,
            lambda e: self._log_split(e),
        )
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.Pow,
            lambda e: self._inv_sym(e),
        )
        cost_expr = cost_expr.replace(
            lambda e: e.func == sympy.Mul,
            lambda e: self._min_expand(e),
        )
        cost_expr = cost_expr.replace(
            lambda e: e.func in [sympy.Min, sympy.Max],
            lambda e: self._truncate_floats_min(e),
        )
        logger.debug("[CP-SAT layout solver] cost expr (linearized): %s", cost_expr)
        return self._print(cost_expr)

    @classmethod
    def _log_min(cls, expr):
        # rewrite log(min(a, b)) as min(log(a), log(b))
        arg = expr.args[0]
        if isinstance(arg, (sympy.Min, sympy.Max)):
            # n() here is to get a numeric value instead of log(2)
            return arg.func(*[sympy.log(a.n()) for a in arg.args])
        if (
            isinstance(arg, sympy.Mul)
            and len(arg.args) == 2
            and isinstance(arg.args[0], sympy.Number)
            and isinstance(arg.args[1], (sympy.Min, sympy.Max))
        ):
            return arg.func(
                *[sympy.log((a * arg.args[0]).n()) for a in arg.args[1].args]
            )
        else:
            return expr

    @classmethod
    def _log_split(cls, expr):
        arg = expr.args[0]
        if isinstance(arg, sympy.Symbol) and "_split_" in arg.name:
            return (
                sympy.Symbol(f"log2_{arg.name}", integer=True, nonnegative=True)
                * sympy.log(2.0)
                / _CORE_LOG_SCALE
            )
        elif isinstance(arg, sympy.Number):
            return math.log(float(arg))
        else:
            return expr

    @classmethod
    def _inv_sym(cls, expr):
        if not isinstance(expr.base, sympy.Symbol):
            return expr
        if expr.exp != -1:
            return expr
        symbol = expr.base
        if (
            "_split_" in symbol.name
            and "log2_" not in symbol.name
            and "inv_" not in symbol.name
        ):
            return (
                sympy.Symbol(f"inv_{symbol.name}", integer=True, nonnegative=True)
                / _CORE_INV_SCALE
            )
        else:
            return expr

    @staticmethod
    def _min_expand(expr):
        # re-writes 2.1*Min(x, y) as Min(2.1*x, 2.1*y)
        if len(expr.args) != 2 or not isinstance(expr.args[0], sympy.Number):
            return expr
        arg = expr.args[1]
        if not isinstance(arg, (sympy.Min, sympy.Max)):
            return expr
        m = expr.args[0]
        new_args = [a * abs(m) for a in arg.args]
        new_args = [
            a.replace(
                lambda e: e.func == sympy.Mul,
                lambda e: _SympyExprToCpSat._min_expand(e),
            )
            for a in new_args
        ]
        return arg.func(*new_args) * sympy.sign(m)

    @staticmethod
    def _truncate_floats_min(expr):
        # re-writes Min(x*0.5, y*0.5) as Min(x, y)/2
        m = 10000
        result = []
        func = expr.func

        def _process(expr):
            if isinstance(expr, sympy.Mul) and isinstance(expr.args[0], sympy.Number):
                a = (expr.args[0] * m).round()
                r = sympy.Mul(a, *expr.args[1:])
            elif isinstance(expr, sympy.Number):
                r = (expr * m).round()
            else:
                r = expr * m
            return r

        for arg in expr.args:
            if isinstance(arg, sympy.Add):
                result.append(sympy.Add(*[_process(a) for a in arg.args]))
            else:
                result.append(_process(arg))

        return func(*result) / m

    def _print_Integer(self, expr):
        return int(expr.p)

    def _print_Number(self, expr):
        return float(expr)

    def _print_Add(self, expr):
        return sum(self._print(arg) for arg in expr.args)

    def _print_Mul(self, expr):
        args = [self._print(arg) for arg in expr.args]
        ints = [arg for arg in args if isinstance(arg, cp_model.IntVar)]
        if len(ints) <= 1:
            return math.prod(args)

        nonints = [arg for arg in args if not isinstance(arg, cp_model.IntVar)]
        name = "_product_" + "_".join([arg.name for arg in ints])
        if name in self._sym_map:
            return math.prod(nonints) * self._sym_map[name]

        lbs, ubs = list(zip(*[self._affine_bounds(arg) for arg in ints]))
        assert all(lb >= 0 for lb in lbs)
        assert all(ub >= 0 for ub in ubs)
        lb, ub = map(math.prod, [lbs, ubs])
        product = self._model.new_int_var(int(lb), int(ub), name)
        self._model.AddMultiplicationEquality(product, ints)
        self._sym_map[name] = product
        return math.prod(nonints) * product

    def _print_Symbol(self, expr):
        if expr.name in self._sym_map:
            return self._sym_map[expr.name]
        if not expr.name.startswith(("log2_", "inv_")):
            raise NotImplementedError(f"not implemented. expr: {expr}")
        name = expr.name[5:] if expr.name.startswith("log2_") else expr.name[4:]
        b = self._sym_map[f"_buffer_{name}"]
        raw = self._sym_map[f"_raw_{name}"]

        if expr.name.startswith("log2_"):
            values = [int(round(_CORE_LOG_SCALE * math.log2(v))) for v in raw]
            domain = cp_model.Domain.FromValues(values)
            cp_var = self._model.new_int_var_from_domain(domain, expr.name)
            self._model.add_element(b.division, values, cp_var)
        else:
            values = [int(round(_CORE_INV_SCALE // v)) for v in raw]
            cp_var = self._model.new_int_var(min(values), max(values), expr.name)
            self._model.AddDivisionEquality(
                cp_var, int(_CORE_INV_SCALE), self._sym_map[name]
            )
        self._sym_map[expr.name] = cp_var
        return cp_var

    def _print_Pow(self, expr):
        return self._print(expr.base) ** self._print(expr.exp)

    def _print_log(self, expr):
        if isinstance(expr.args[0], sympy.Number):
            return math.log(float(expr.args[0]))
        raise NotImplementedError(f"log not implemented. expr: {expr}")

    @staticmethod
    def _affine_bounds(expr):
        if isinstance(expr, cp_model.IntVar):
            lb, ub = expr.domain.min(), expr.domain.max()
        elif isinstance(expr, (int, float)):
            lb, ub = expr, expr
        elif isinstance(expr, cp_model_helper.IntAffine):
            lb, ub = _SympyExprToCpSat._affine_bounds(expr.expression)
            c, o = int(expr.coefficient), int(expr.offset)
            lb, ub = (c * lb + o, c * ub + o) if c >= 0 else (c * ub + o, c * lb + o)
        elif hasattr(expr, "num_exprs"):
            # SumArray (e.g. from ``a + b + c`` or ``sum(...)``): flatten to a
            # single offset + per-var coefficients and bound each term.
            flat = cp_model.FlatIntExpr(expr)
            lb = ub = int(flat.offset)
            for var, c in zip(flat.vars, flat.coeffs):
                c = int(c)
                vlb, vub = _SympyExprToCpSat._affine_bounds(var)
                if c >= 0:
                    vlb, vub = c * vlb, c * vub
                else:
                    vlb, vub = c * vub, c * vlb
                lb, ub = lb + vlb, ub + vub
                assert lb <= ub
        else:
            raise TypeError(f"unsupported expr type: {type(expr)}")

        lb, ub = int(lb), int(ub)
        assert lb <= ub
        return lb, ub

    def _print_Max(self, expr):
        # max range is (max(mins), max(maxes))
        args = [self._print(arg) for arg in expr.args]
        bounds = map(max, zip(*[self._affine_bounds(arg) for arg in args]))
        max_var = self._model.new_int_var(*bounds, f"max_var_{self._count}")
        self._model.AddMaxEquality(max_var, args)
        self._count += 1
        return max_var

    def _print_Min(self, expr):
        # min range is (min(mins), min(maxes))
        args = [self._print(arg) for arg in expr.args]
        bounds = map(min, zip(*[self._affine_bounds(arg) for arg in args]))
        min_var = self._model.new_int_var(*bounds, f"min_var_{self._count}")
        self._model.AddMinEquality(min_var, args)
        self._count += 1
        return min_var


class CpSatLayoutSolver(CoreDivisionLayoutSolver):
    """Joint core-division + LX placement via an OR-Tools CP-SAT search
    (``config.layout_solver == "cpsat"``). See the module docstring for the
    model (joint division, slicing-match residency gate, 2D no-overlap with
    in-place lifetime shortening) and the lexicographic objective
    (residency, then parallelism, then division balance).
    """

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        time_limit_seconds: float = 120.0,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, "
                "which is not installed. Install it with 'pip install ortools' "
                "or select a different layout_solver (e.g. 'greedy')."
            )
        super().__init__(buffers, size, alignment)
        # The solver works in alignment-sized units so every offset it picks is
        # automatically aligned; plan_layout scales sizes/offsets in and out.
        self._capacity_units = self.limit // self.alignment
        self._time_limit_seconds = time_limit_seconds
        self._bottom_justify = bottom_justify

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """Place buffers on their already-fixed core divisions (placement-only).

        Same model as :meth:`plan_layout_and_core_divisions` minus the joint
        division choice: each buffer's footprint is its ``size``, so there is no
        slicing gate on residency and no parallelism step -- the solve reduces
        to minimising HBM traffic under the 2D no-overlap with in-place reuse.
        Dispatch is per buffer and keys on whether it carries candidate
        divisions, not on its class, so a :class:`CoreDivisionBuffer` with an
        empty candidate list is placed here rather than divided."""
        return cast("list[LifetimeBoundBuffer]", list(self._plan_layout_generic()))

    def plan_layout_and_core_divisions(
        self, cost_expr: sympy.Expr | None = None
    ) -> list[CoreDivisionBuffer]:
        """Jointly choose each buffer's core division and its LX placement.

        The full model described in the module docstring. Every buffer must
        carry enumerated candidate divisions; the chosen index is written back
        to ``chosen_division`` for the allocator to commit."""
        buffers = cast("Sequence[CoreDivisionBuffer]", self.buffers)
        assert all(len(b.core_divisions) != 0 for b in buffers), (
            "All buffers must have at least 1 valid core division"
        )
        return cast(
            "list[CoreDivisionBuffer]",
            list(self._plan_layout_generic(cost_expr=cost_expr)),
        )

    def _wrap(
        self, model: "cp_model.CpModel", buffer: LifetimeBoundBuffer
    ) -> _LifetimeBufferWithCpVars:
        """Bundle a *copy* of ``buffer`` with its CP-SAT vars, scaled into the
        alignment units the solver works in.

        A buffer carrying enumerated core divisions gets the joint wrapper (its
        ``size`` is the total device footprint, divided down by the chosen
        division); anything else -- a plain :class:`LifetimeBoundBuffer`, or a
        :class:`CoreDivisionBuffer` with nothing to choose from -- gets the
        placement-only wrapper, whose footprint is ``size`` as given."""
        units = ceil_div(buffer.size, self.alignment)
        if isinstance(buffer, CoreDivisionBuffer) and buffer.core_divisions:
            return _CoreDivisionBufferWithCpVars(
                buffer=replace(buffer, size=units),
                capacity_units=self._capacity_units,
                model=model,
            )
        return _LifetimeBufferWithCpVars(
            buffer=replace(buffer, size=units),
            capacity_units=self._capacity_units,
            model=model,
        )

    def _plan_layout_generic(
        self,
        log_lx_usage: bool = False,
        cost_expr: sympy.Expr | None = None,
    ) -> list[LifetimeBoundBuffer | CoreDivisionBuffer]:
        buffers = self.buffers
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        _check_in_place_relationships(buffers)

        # Declarative exclusion, shared with every other solver: whatever the
        # allocator barred (each buffer's ``residency_reason``), plus the
        # no-LX-reader and capacity checks. Unlike the gap solvers -- which
        # ``partition`` these out -- we still hand the barred buffers to the
        # model (they must stay available for slicing matching and in-place
        # chains) but pin them non-resident below, so we only need the reasons.
        forced_reasons = dict(self.record_exclusions())

        model = cp_model.CpModel()
        # Solve on copies so we never mutate the caller's buffers.
        working = {b.name: self._wrap(model, b) for b in buffers}

        solved = self._run(model, working, forced_reasons, cost_expr=cost_expr)
        # Surface a drop cause for every spilled buffer: the pre-solve forced
        # reason when we have one, otherwise the solver chose to spill it.
        self.spill_reasons = {
            name: forced_reasons.get(name, _SOLVER_CHOSE_SPILL)
            for name, sb in solved.items()
            if sb.address is None
        }

        # Copy the solved results back onto the caller's buffers. Offsets come
        # back in alignment units (the solver works in aligned units), so scale
        # the address to bytes on the way out.
        for b in buffers:
            sb = solved[b.name]
            b.address = None if sb.address is None else sb.address * self.alignment
            if isinstance(b, CoreDivisionBuffer) and isinstance(sb, CoreDivisionBuffer):
                b.chosen_division = sb.chosen_division
        return list(buffers)

    # ------------------------------------------------------------------
    # Model build + solve
    # ------------------------------------------------------------------
    def _minimize_cost_expr(
        self,
        model: "cp_model.CpModel",
        solver: "cp_model.CpSolver",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        cost_expr: sympy.Expr,
    ) -> Optional["cp_model.CpSolverStatus"]:
        sym_map = {}
        for t in tensors.values():
            sym_map[t.buffer.sym_is_lx.name] = t.in_buffer
            sym_core_divs = t.buffer.sym_core_divs
            for splits, cp_splits, cp_splits_raw in zip(
                sym_core_divs,
                t.cp_core_divs,
                t.cp_core_divs_raw,
            ):
                for key, symbol in splits.items():
                    assert isinstance(symbol, sympy.Symbol)
                    sym_map[symbol.name] = cp_splits[key]
                    sym_map[f"_buffer_{symbol.name}"] = t
                    sym_map[f"_raw_{symbol.name}"] = cp_splits_raw[key]

        try:
            cp_cost = _SympyExprToCpSat(model, sym_map).convert(cost_expr)
            if not isinstance(cp_cost, (int, float)):
                # if the cost is non-constant, we minimize it
                # if the cost is constant, we use any solution
                model.minimize(cp_cost)
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError("CP-SAT memory planner found no feasible plan")
            return status
        except (RuntimeError, TypeError, ValueError):
            logger.warning("[CP-SAT layout solver] cannot linearize the sympy expr")
            if not config._cpsat_warn_on_cost_expr:
                raise
            return None

    def _run(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        forced_reasons: dict[str, str],
        cost_expr: sympy.Expr | None,
    ) -> dict[str, LifetimeBoundBuffer]:
        children_of = self._get_children(tensors)
        self._add_inplace_relaxation(model, tensors)
        self._add_core_division(model, tensors, children_of, forced_reasons)

        solver = cp_model.CpSolver()
        if self._time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(self._time_limit_seconds)
        solver.parameters.num_search_workers = (
            1 if torch.are_deterministic_algorithms_enabled() else (os.cpu_count() or 1)
        )
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0

        status = None
        core_terms = None
        occupancy: Optional[int] = None

        if cost_expr is not None:
            status = self._minimize_cost_expr(model, solver, tensors, cost_expr)

        if status is None:
            # TODO: Update objective to a maxmin optimization to optimize overall
            # throughput.
            #
            # The objective is a lexicographic solve: residency first, then
            # parallelism, then division balance. Each step locks the prior optimum
            # as a constraint before optimizing the next, so a later step only
            # breaks ties the earlier ones leave open.

            # Residency (the hard priority): minimize total HBM transfer traffic so
            # as much as possible stays resident in LX.
            hbm_terms = [
                sb.spill_cost() * (1 - sb.in_buffer) for sb in tensors.values()
            ]
            status = cp_model.INFEASIBLE
            if hbm_terms:
                model.minimize(sum(hbm_terms))
                status = solver.Solve(model)
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    raise SolveError("CP-SAT memory planner found no feasible plan")
                # Lock in the residency optimum (the traffic value, not just the
                # count) so the parallelism step can never trade a spill for
                # parallelism. Rounding avoids loss of precision as the objective is
                # a sum/product of ints.
                model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))

            # Parallelism: holding the residency optimum, maximize total core usage
            # so every buffer (resident or spilled) takes its most parallel
            # division. Placement-only buffers have no division to choose and so
            # contribute no term; with none at all there is nothing to maximize, so
            # we skip the re-solve and the extract below reads the residency
            # assignment still held by ``solver``.
            core_terms = [sb.cores for sb in tensors.values() if sb.cores is not None]
            # A core_cost term exists for exactly the same buffers as a core term
            # (both are set only on division-carrying buffers), so phase 3 runs
            # whenever phase 2 does.
            core_cost_terms = [
                sb.core_cost for sb in tensors.values() if sb.core_cost is not None
            ]
            if core_terms:
                model.maximize(sum(core_terms))
                status = solver.Solve(model)
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    raise SolveError("CP-SAT memory planner found no feasible plan")
                occupancy = round(solver.ObjectiveValue())

                # Shape balance: holding the parallelism optimum (the objective is
                # integer, so the round is exact), break the remaining ties toward a
                # balanced division by minimizing the summed squared split factors.
                # The parallelism solution still satisfies this lock, so this only
                # refines the choice among equally parallel divisions and can never
                # spill a buffer or lower its core count.
                model.add(sum(core_terms) >= occupancy)
                model.minimize(sum(core_cost_terms))
                status = solver.Solve(model)
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    raise SolveError("CP-SAT memory planner found no feasible plan")

        final_tensors = self._extract(solver, tensors)

        if logger.isEnabledFor(logging.DEBUG):
            if status is None:
                status = cp_model.INFEASIBLE
            spilled = [n for n, t in final_tensors.items() if t.address is None]
            # The final solve minimized the balance cost when there were
            # divisions to choose (with occupancy held at ``occupancy``);
            # otherwise only the residency solve ran and the objective is HBM
            # traffic.
            logger.debug(
                "[CP-SAT layout solver] tensors=%d resident=%d %s=%d "
                "occupancy=%s status=%s walltime=%.2f ms",
                len(tensors),
                len(tensors) - len(spilled),
                "balance" if core_terms else "hbm_traffic",
                round(solver.ObjectiveValue()),
                occupancy if occupancy is not None else "n/a",
                solver.StatusName(status),
                solver.WallTime() * 1e3,
            )
            # Per-buffer drop cause: a pre-solve forced reason when we have one,
            # otherwise the solver chose to spill it (residency gave no benefit,
            # or there was no room once higher-value buffers were placed).
            for name in sorted(spilled):
                logger.debug(
                    "[CP-SAT layout solver]   %s -> HBM: %s",
                    name,
                    forced_reasons.get(name, _SOLVER_CHOSE_SPILL),
                )

        return final_tensors

    def _add_inplace_relaxation(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> None:
        """In-place reuse as a relaxation of the no-overlap constraint: each
        parent->child edge gets a merge bool that, when active, pins the pair to
        one shared base. Rather than lifting a pairwise no-overlap, an active
        merge *shortens the child's lifetime by the single handoff tick* it
        shares with the parent (``_check_in_place_relationships`` guarantees the
        overlap is exactly that one tick): the two then become time-adjacent
        rectangles that may legally sit at the same offset under the global 2D
        no-overlap (see ``_add_no_overlap_2d``). Chains are induced transitively
        by the shared-offset equalities -- no merge groups, no path enumeration.
        The per-buffer ``merge_vars`` bools are read back in ``_extract`` to
        reconstruct placement units."""
        M = self._capacity_units

        # A storage slot is handed off linearly, so a buffer reuses at most one
        # parent and is reused by at most one child. ``incoming`` also drives the
        # lifetime shortening in ``_add_no_overlap_2d``.
        incoming: dict[str, list] = {}
        outgoing: dict[str, list] = {}
        for dst, c in bufs.items():
            for src, edge in c.merge_vars.items():
                src_v, dst_v = bufs[src], bufs[dst]
                # active merge => shared base and both endpoints resident
                model.add(src_v.offset == dst_v.offset).OnlyEnforceIf(edge)
                model.add_implication(edge, src_v.in_buffer)
                model.add_implication(edge, dst_v.in_buffer)
                # active merge => the child must be able to take over the
                # parent's exact storage (joint model: equal per-core footprints
                # under slicing-compatible divisions; nothing extra when the
                # division is fixed).
                dst_v.constrain_merge(model, src_v, edge)
                outgoing.setdefault(src, []).append(edge)
                incoming.setdefault(dst, []).append(edge)

        for ms in (*incoming.values(), *outgoing.values()):
            if len(ms) > 1:
                model.add_at_most_one(ms)

        for sb in bufs.values():
            # if a buffer is resident its top must be below the peak usage.
            model.add(sb.offset + sb.eff_size <= M).OnlyEnforceIf(sb.in_buffer)

        self._add_no_overlap_2d(model, bufs, incoming)

    def _add_no_overlap_2d(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        incoming: dict[str, list],
    ) -> None:
        """Global 2D no-overlap: each resident buffer is an optional rectangle
        ``[start_time, end_time) x [offset, offset + eff_size)`` and no two may
        intersect (touching edges are allowed). Residency is the interval
        presence (``in_buffer``), so spilled buffers drop out for free.

        In-place reuse is handled *inside* this constraint rather than by
        relaxing it: an active incoming merge shortens the child's time interval
        by the single handoff tick it shares with the parent
        (``start -> start + 1``). The parent and child then abut in time at the
        same offset (pinned equal by the merge), which the 2D constraint accepts
        as non-overlapping -- so the child legally reuses the parent's slot. With
        no active merge the child keeps its full lifetime and the shared-offset
        placement is correctly forbidden, exactly as the pairwise encoding did.

        It is the *child* that gives up the tick, never the parent: the parent's
        rectangle has to keep covering the handoff tick at full footprint. The
        child may be smaller than its parent (the placement-only model only
        requires ``child.size <= parent.size``), and the bytes above the child
        are still holding parent data that is read on that tick, so they are not
        free for a third buffer. Shortening the parent instead would expose them
        -- and a parent whose whole lifetime is that one tick would drop out of
        the propagator entirely, exposing its full slot.

        ``AddAtMostOne`` on the incoming edges bounds the shortening at one tick.
        A child whose entire lifetime is the handoff tick degenerates to a
        zero-width box the 2D propagator ignores, which is safe here: the tick is
        covered by the parent's box, whose footprint contains the child's at the
        shared offset."""
        x_intervals = []
        y_intervals = []
        for sb in bufs.values():
            ins = incoming.get(sb.name, [])
            if ins:
                # at most one incoming merge is active (AddAtMostOne), so the
                # sum is 0 or 1: shorten the child by the handoff tick exactly
                # when it takes over a parent's slot.
                start_var = model.new_int_var(
                    sb.start_time, sb.end_time, f"start_{sb.name}"
                )
                model.add(start_var == sb.start_time + sum(ins))
                x_start: object = start_var
                x_size: object = sb.end_time - start_var
            else:
                x_start = sb.start_time
                x_size = sb.end_time - sb.start_time
            x_intervals.append(
                model.new_optional_interval_var(
                    x_start, x_size, sb.end_time, sb.in_buffer, f"x_{sb.name}"
                )
            )
            # An interval's ``end`` must be affine (a single var), so the address
            # top ``offset + eff_size`` (a sum of two vars) needs its own var; the
            # interval ties it to start+size whenever the buffer is resident.
            y_end = model.new_int_var(0, self._capacity_units, f"top_{sb.name}")
            y_intervals.append(
                model.new_optional_interval_var(
                    sb.offset,
                    sb.eff_size,
                    y_end,
                    sb.in_buffer,
                    f"y_{sb.name}",
                )
            )
        model.add_no_overlap_2d(x_intervals, y_intervals)

    def _get_children(
        self, bufs: dict[str, _LifetimeBufferWithCpVars]
    ) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
        """parent name -> list of (child name, match_pairs), where ``match_pairs``
        is the child's ``cd_parent_matches[parent]`` (empty when the edge has no
        compatible division). The child's ``parents`` define the edges; a
        placement-only buffer declares none, so the map is empty there."""
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]] = {}
        for sb in bufs.values():
            for parent in sb.parents:
                children_of.setdefault(parent, []).append(
                    (sb.name, sb.match_pairs(parent))
                )
        return children_of

    def _add_core_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        forced: dict[str, str],
    ) -> None:
        """Pin out every buffer ``forced`` non-resident (decided declaratively by
        :meth:`MemoryPlanSolver.partition`) and install the per-buffer residency
        gate. In the joint model that gate is the slicing match, driven entirely
        by the precomputed ``cd_parent_matches`` pairs; placement-only buffers
        have no gate."""
        for name in forced:
            model.add(bufs[name].in_buffer == 0)
        for sb in bufs.values():
            sb.constrain_residency(model, children_of.get(sb.name, []), bufs)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    def _extract(
        self,
        solver: "cp_model.CpSolver",
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> dict[str, LifetimeBoundBuffer]:
        """Read the solution back onto each buffer and return ``name -> buffer``.

        Every buffer gets its ``chosen_division`` (a no-op for a placement-only
        buffer, whose division was fixed upstream) and, when resident, its LX
        ``address`` (in alignment units, as the solver works them; the caller
        scales to bytes). A spilled buffer gets ``address = None``. When
        bottom_justify is set, each in-place-merged placement unit is slid down
        to the lowest free address (preserving merges); if that squeeze cannot
        keep every unit inside capacity the solver's own offsets are kept, since
        those are always legal."""
        by_name = {name: sb.buffer for name, sb in bufs.items()}
        spilled = {
            name for name, sb in bufs.items() if not solver.BooleanValue(sb.in_buffer)
        }
        footprint = {name: sb.footprint(solver) for name, sb in bufs.items()}

        offsets: Optional[dict[str, int]] = None
        if self._bottom_justify:
            # A placement unit is a connected component of active merge edges: its
            # members share one base (the merge equalities), so the component
            # slides as a single block and in-place reuse is preserved.
            resident = [n for n in by_name if n not in spilled]
            parent = {n: n for n in resident}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for dst, c in bufs.items():
                for src, edge in c.merge_vars.items():
                    if solver.BooleanValue(edge):
                        parent[find(src)] = find(dst)

            components: dict[str, list[str]] = {}
            for n in resident:
                components.setdefault(find(n), []).append(n)

            units = [
                _PlacementUnit(
                    members=names,
                    footprint=max(footprint[n] for n in names),
                    start_time=min(by_name[n].start_time for n in names),
                    end_time=max(by_name[n].end_time for n in names),
                    original_offset=solver.Value(bufs[names[0]].offset),
                )
                for names in components.values()
            ]
            offsets = self._justify(units, self._capacity_units)

        if offsets is None:
            offsets = {
                name: solver.Value(sb.offset)
                for name, sb in bufs.items()
                if name not in spilled
            }

        for name, sb in bufs.items():
            t = sb.buffer
            sb.record_division(solver)
            if name in spilled:
                t.address = None
            else:
                t.address = offsets[name]
        return by_name

    @staticmethod
    def _justify(
        units: list[_PlacementUnit], capacity: int
    ) -> Optional[dict[str, int]]:
        """Slide each placement unit down to the lowest free address. Processing
        in current-base order and giving each the lowest non-conflicting slot
        preserves the relative stacking, so it mostly squeezes out the float gaps
        the search leaves. Returns a name -> address map, or ``None`` if the
        result would not fit in ``capacity``.

        A merged unit is coarsened to one rectangle spanning the union of its
        members' lifetimes at their largest footprint, which is conservative: it
        can make two units conflict here that did not conflict in the model, and
        the bump that resolves that conflict can push a unit's top past capacity.
        The caller then keeps the solver's own offsets, which the model
        constrained to fit. Returning ``None`` rather than clamping keeps this a
        pure optimisation -- it never decides residency, and never hands back an
        address outside the scratchpad."""
        placed: list[_PlacementUnit] = []
        offsets = {}
        for u in sorted(units, key=lambda u: (u.original_offset, u.start_time)):
            # lowest base whose [base, base+footprint) clears every already-placed
            # unit that overlaps this one in time. We don't need to worry about
            # tied offsets because blocks cannot have the same offset and also
            # overlap in time.
            obstacles = sorted(
                (p.justified_offset, p.justified_offset + p.footprint)
                for p in placed
                if u.start_time < p.end_time and p.start_time < u.end_time
            )
            base = 0
            for lo, hi in obstacles:
                if base + u.footprint <= lo:
                    break  # fits in the gap below this obstacle
                if base < hi:
                    base = hi  # otherwise bump above it
            if base + u.footprint > capacity:
                return None
            u.justified_offset = base
            placed.append(u)
            for n in u.members:
                offsets[n] = base
        return offsets
