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
  encoded by *shortening the parent's lifetime* by the single handoff tick when
  the merge fires, so the parent and its in-place child abut in time and may
  legally share an offset; the single-tick-overlap invariant
  (``_assert_in_place_relationships``) makes this exact (``_add_no_overlap_2d``).
* **Objective** (two-phase lexicographic, in ``_run``). *Residency is the hard
  priority.* Phase 1 minimizes total **HBM transfer traffic** via
  ``spill_cost(b) * (1 - in_buffer)`` -- the *differential* traffic a spill adds
  over residency (resident buffers contribute 0). An intermediate costs
  ``(num_consumers + 1) * size`` (the producer's HBM write, which residency turns
  into a free LX write, plus one re-read per consumer); a graph input drops the
  producer write it never had and the clone-in read residency cannot avoid
  (``(num_consumers - 1) * size``); a graph output drops its unavoidable
  write-out (``num_consumers * size``). Phase 1 puts as much in LX as possible
  and chooses whatever division serves that (even no split, if that is what lets
  a buffer match its consumers and reside). Phase 2 then *holds that residency
  optimum* and maximizes total core usage (``sum_b cores_b``) so every buffer --
  resident or spilled, the latter free of the slicing gate -- takes its most
  parallel division, which the allocator commits. Parallelism never costs a
  spill.

After the solve, ``_justify`` slides each in-place-merged placement unit down to
the lowest free address, squeezing out float gaps without raising the peak.

The same model also serves plain :class:`LifetimeBoundBuffer`s via
``plan_layout`` (the ``MemoryPlanSolver`` contract the placement-only allocator
calls). Those buffers carry no candidate divisions, so the division-dependent
pieces -- per-core sizing, the slicing-match gate, the merge division gate and
the phase-2 parallelism objective -- simply drop out: the footprint is the
buffer's ``size`` and the solve reduces to minimising HBM traffic under the 2D
no-overlap with in-place reuse. Residency is then gated only by capacity and by
the allocator's own ``residency_reason`` bars (which both paths honour, since
that field lives on the base buffer). That specialisation
lives on the buffer wrappers (``_LifetimeBufferWithCpVars`` and its joint
subclass ``_CoreDivisionBufferWithCpVars``), so the solver methods below are
written once against whichever wrapper ``_wrap`` chose.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar, cast
import torch
import numpy as np


if TYPE_CHECKING:
    from ortools.sat.python import cp_model
else:
    try:
        from ortools.sat.python import cp_model

    except ImportError:  # pragma: no cover - exercised only when ortools is absent
        cp_model = None

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    SolveError,
    BufferType,
    _assert_in_place_relationships,
)

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
        # phase 2 when no buffer offers a core-usage term.
        self.cores = None
        self.merge_vars = {
            parent: m.new_bool_var(f"merge_{parent}_{b.name}")
            for parent in b.in_place_parents
        }

    # -- producer/consumer edges (joint model only; none when division-fixed) --
    @property
    def parents(self) -> list[str]:
        return []

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return []

    # ------------------------------ residency ------------------------------
    @property
    def min_footprint(self) -> int:
        """Smallest footprint the buffer can take, for the capacity trim."""
        return self.buffer.size

    @property
    def residency_reason(self) -> "str | None":
        """Allocator-supplied reason the buffer may not reside, if any. Carried
        on the buffer itself, so both entry points honour the allocator's
        hard bars (e.g. the restickify cross-frame barrier) identically."""
        return self.buffer.residency_reason

    def spill_cost(self, num_children: int) -> int:
        """Differential HBM traffic a spill adds over residency. Without the
        producer/consumer edges of the joint model, the accesses are read
        straight off ``uses``: the producer's write (absent for a graph input,
        whose first use is a read) plus one re-read per later use."""
        b = self.buffer
        writes = 0 if b.first_use_is_read else 1
        reads = len(b.uses) - writes
        return (reads + writes) * b.size

    def constrain_residency(self, model, kids, bufs) -> "str | None":
        """Placement-only: any buffer may reside, so there is no slicing gate."""
        return None

    def constrain_merge(self, model, parent: "_LifetimeBufferWithCpVars", edge) -> None:
        """Extra conditions on an active in-place merge. None when the division
        is fixed: ``_assert_in_place_relationships`` already checks the child
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

        per_core = [
            int(np.ceil(b.size / cd.output_partition)) for cd in b.core_divisions
        ]
        # Total cores the op runs on under each division -- includes any
        # reduction-axis split, so a reduction-parallel division counts its full
        # parallelism (``output_partition`` alone would score it as 1 core).
        cores_used = [cd.cores_used for cd in b.core_divisions]
        self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
        self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")
        # total cores this op uses under the chosen div
        self.cores = m.new_int_var(0, max(cores_used), f"occ_{b.name}")

        # tie per-core footprint (output split only) and total core usage to the
        # chosen division index
        m.add_element(self.division, per_core, self.eff_size)
        m.add_element(self.division, cores_used, self.cores)

    @property
    def parents(self) -> list[str]:
        return self.buffer.parents

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return self.buffer.cd_parent_matches.get(parent, [])

    @property
    def min_footprint(self) -> int:
        t = self.buffer
        return min(
            int(np.ceil(t.size / cd.output_partition)) for cd in t.core_divisions
        )

    def spill_cost(self, num_children: int) -> int:
        b = self.buffer
        reads = num_children + b.unallocated_reads
        if b.boundary == BufferType.Input and reads > 0:
            reads -= 1
        writes = 1 if b.boundary == BufferType.Intermediate else 0
        return (reads + writes) * b.size

    def constrain_residency(self, model, kids, bufs) -> "str | None":
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs. A
        buffer with no consumer edge, or with a consumer that has no compatible
        pair, can never reside; the reason returned here is surfaced to the
        allocator as the buffer's drop cause."""
        t = self.buffer
        if not kids:
            if t.unallocated_reads:
                # Read only by consumers the solver never sees (filtered-out
                # ops / graph outputs). They still read it from LX when it
                # resides, so residency is worthwhile; there is no resident
                # consumer to constrain the division against, so no gate.
                # TODO: Remove this when the other solvers are brought to parity
                return None
            # Nothing consumes this buffer from LX -> it can never reside.
            model.add(self.in_buffer == 0)
            return "no consumer reads it from LX"
        for child, compatible in kids:
            if not compatible:
                # This child can never match -> the buffer cannot reside.
                model.add(self.in_buffer == 0)
                return f"consumer {child} has no slicing-compatible core division"
            _gate_divisions(
                model, compatible, self.division, bufs[child].division, self.in_buffer
            )
        return None

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
        return int(np.ceil(t.size / cd.output_partition))

    def record_division(self, solver: "cp_model.CpSolver") -> None:
        self.buffer.chosen_division = solver.Value(self.division)


class CpSatLayoutSolver(CoreDivisionLayoutSolver):
    """Joint core-division + LX placement via an OR-Tools CP-SAT search
    (``config.layout_solver == "cpsat"``). See the module docstring for the
    model (joint division, slicing-match residency gate, 2D no-overlap with
    in-place lifetime shortening) and the HBM-traffic objective.
    """

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        time_limit_seconds: float = 600.0,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, "
                "which is not installed. Install it with 'pip install ortools' "
                "or select a different layout_solver (e.g. 'greedy')."
            )
        super().__init__(size, alignment)
        # The solver works in alignment-sized units so every offset it picks is
        # automatically aligned; plan_layout scales sizes/offsets in and out.
        self._capacity_units = self.limit // self.alignment
        self._time_limit_seconds = time_limit_seconds
        self._bottom_justify = bottom_justify
        # Per-buffer drop cause for the most recent solve ({buffer name: reason},
        # spilled buffers only). The allocator reads this to populate its own
        # ``reject_reasons`` so cpsat spills show up in the LX-pinning debug log.
        self.spill_reasons: dict[str, str] = {}

    def plan_layout(
        self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        """Place buffers on their already-fixed core divisions (placement-only).

        Same model as :meth:`plan_layout_and_core_divisions` minus the joint
        division choice: each buffer's footprint is its ``size``, so there is no
        slicing gate on residency and no parallelism phase -- the solve reduces
        to minimising HBM traffic under the 2D no-overlap with in-place reuse.
        Dispatch is per buffer and keys on whether it carries candidate
        divisions, not on its class, so a :class:`CoreDivisionBuffer` with an
        empty candidate list is placed here rather than divided."""
        return cast(
            "list[LifetimeBoundBuffer]", list(self._plan_layout_generic(buffers))
        )

    def plan_layout_and_core_divisions(
        self, buffers: Sequence[CoreDivisionBuffer]
    ) -> list[CoreDivisionBuffer]:
        """Jointly choose each buffer's core division and its LX placement.

        The full model described in the module docstring. Every buffer must
        carry enumerated candidate divisions; the chosen index is written back
        to ``chosen_division`` for the allocator to commit."""
        assert all(len(b.core_divisions) != 0 for b in buffers), (
            "All buffers must have at least 1 valid core division"
        )
        return cast(
            "list[CoreDivisionBuffer]", list(self._plan_layout_generic(buffers))
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
        units = int(np.ceil(buffer.size / self.alignment))
        if isinstance(buffer, CoreDivisionBuffer) and buffer.core_divisions:
            return _CoreDivisionBufferWithCpVars(
                replace(buffer, size=units), model, self._capacity_units
            )
        return _LifetimeBufferWithCpVars(
            replace(buffer, size=units), model, self._capacity_units
        )

    def _plan_layout_generic(
        self,
        buffers: Sequence[LifetimeBoundBuffer | CoreDivisionBuffer],
        log_lx_usage: bool = False,
    ) -> list[LifetimeBoundBuffer | CoreDivisionBuffer]:
        self.spill_reasons = {}
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        _assert_in_place_relationships(buffers)

        model = cp_model.CpModel()
        # Solve on copies so we never mutate the caller's buffers.
        working = {b.name: self._wrap(model, b) for b in buffers}

        solved, forced_reasons = self._run(model, working)
        spilled = {name for name, sb in solved.items() if sb.address is None}
        # Surface a drop cause for every spilled buffer: the pre-solve forced
        # reason when we have one, otherwise the solver chose to spill it.
        self.spill_reasons = {
            name: forced_reasons.get(name, _SOLVER_CHOSE_SPILL) for name in spilled
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
    def _run(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
    ) -> tuple[dict[str, LifetimeBoundBuffer], dict[str, str]]:
        children_of = self._get_children(tensors)
        self._add_inplace_relaxation(model, tensors)
        forced_reasons = self._add_core_division(model, tensors, children_of)

        solver = cp_model.CpSolver()
        if self._time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(self._time_limit_seconds)
        solver.parameters.num_search_workers = (
            1 if torch.are_deterministic_algorithms_enabled() else (os.cpu_count() or 1)
        )
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0

        # TODO: Update objective to a maxmin optimization to optimize overall
        # throughput.
        hbm_terms = [
            sb.spill_cost(len(children_of.get(sb.name, []))) * (1 - sb.in_buffer)
            for sb in tensors.values()
        ]
        status = cp_model.INFEASIBLE
        if hbm_terms:
            model.minimize(sum(hbm_terms))
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError("CP-SAT memory planner found no feasible plan")
            # Lock in the residency optimum (the traffic value, not just the
            # count) so phase 2 can never trade a spill for parallelism.

            # Rounding avoids loss of precision as the objective function is
            # the sum and multiplication of integers.
            model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))

        # Phase 2 -- parallelism: holding the residency optimum, maximize total
        # core usage so every buffer (resident or spilled) takes its most
        # parallel division. Placement-only buffers have no division to choose
        # and so contribute no term; with none at all there is nothing to
        # maximize, so we skip the re-solve and the extract below reads the
        # phase-1 assignment still held by ``solver``.
        core_terms = [sb.cores for sb in tensors.values() if sb.cores is not None]
        if core_terms:
            model.maximize(sum(core_terms))
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError("CP-SAT memory planner found no feasible plan")

        final_tensors = self._extract(solver, tensors)

        if logger.isEnabledFor(logging.DEBUG):
            spilled = [n for n, t in final_tensors.items() if t.address is None]
            logger.debug(
                "[CP-SAT layout solver] tensors=%d resident=%d %s=%d "
                "status=%s walltime=%.2f ms",
                len(tensors),
                len(tensors) - len(spilled),
                "occupancy" if core_terms else "hbm_traffic",
                round(solver.ObjectiveValue()),
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

        return final_tensors, forced_reasons

    def _add_inplace_relaxation(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> None:
        """In-place reuse as a relaxation of the no-overlap constraint: each
        parent->child edge gets a merge bool that, when active, pins the pair to
        one shared base. Rather than lifting a pairwise no-overlap, an active
        merge *shortens the parent's lifetime by the single handoff tick* it
        shares with the child (``_assert_in_place_relationships`` guarantees the
        overlap is exactly that one tick): the two then become time-adjacent
        rectangles that may legally sit at the same offset under the global 2D
        no-overlap (see ``_add_no_overlap_2d``). Chains are induced transitively
        by the shared-offset equalities -- no merge groups, no path enumeration.
        The per-buffer ``merge_vars`` bools are read back in ``_extract`` to
        reconstruct placement units."""
        M = self._capacity_units

        # A storage slot is handed off linearly, so a buffer reuses at most one
        # parent and is reused by at most one child. ``outgoing`` also drives the
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

        self._add_no_overlap_2d(model, bufs, outgoing)

    def _add_no_overlap_2d(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        outgoing: dict[str, list],
    ) -> None:
        """Global 2D no-overlap: each resident buffer is an optional rectangle
        ``[start_time, end_time) x [offset, offset + eff_size)`` and no two may
        intersect (touching edges are allowed). Residency is the interval
        presence (``in_buffer``), so spilled buffers drop out for free.

        In-place reuse is handled *inside* this constraint rather than by
        relaxing it: an active outgoing merge shortens the parent's time
        interval by the single handoff tick it shares with the child
        (``end -> end - 1``). The parent and child then abut in time at the same
        offset (pinned equal by the merge), which the 2D constraint accepts as
        non-overlapping -- so the child legally reuses the parent's slot. With no
        active merge the parent keeps its full lifetime and the shared-offset
        placement is correctly forbidden, exactly as the pairwise encoding did.

        The handoff tick stays protected because the child's interval covers it
        at the shared offset; the merge ``eff_size`` equality means there is no
        footprint gap. ``AddAtMostOne`` on the outgoing edges bounds the
        shortening at one tick (a degenerate zero-width parent box is ignored by
        the 2D propagator, which is fine -- the child holds the slot)."""
        x_intervals = []
        y_intervals = []
        for sb in bufs.values():
            outs = outgoing.get(sb.name, [])
            if outs:
                # at most one outgoing merge is active (AddAtMostOne), so the
                # sum is 0 or 1: shorten the parent by the handoff tick exactly
                # when it hands its slot to an in-place child.
                end_var = model.new_int_var(
                    sb.start_time, sb.end_time, f"end_{sb.name}"
                )
                model.add(end_var == sb.end_time - sum(outs))
                x_size: object = end_var - sb.start_time
                x_end: object = end_var
            else:
                end_var = sb.end_time
                x_size = sb.end_time - sb.start_time
                x_end = sb.end_time
            x_intervals.append(
                model.new_optional_interval_var(
                    sb.start_time, x_size, x_end, sb.in_buffer, f"x_{sb.name}"
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

    def _trim_oversized_tensors(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> dict[str, str]:
        """Pin out of LX the buffers whose non-residency is fixed up front: those
        whose *smallest* candidate footprint still exceeds capacity, and those
        the allocator marked non-resident (``residency_reason`` set). Returns
        ``name -> reason`` for the buffers it forces out (drop-cause debug
        logging), using the allocator's specific reason when it has one."""
        forced: dict[str, str] = {}
        for sb in bufs.values():
            min_size = sb.min_footprint
            if min_size > self._capacity_units:
                forced[sb.name] = (
                    f"min per-core footprint {min_size} > LX capacity "
                    f"{self._capacity_units} (alignment units)"
                )
                model.add(sb.in_buffer == 0)
            elif sb.residency_reason is not None:
                forced[sb.name] = sb.residency_reason
                model.add(sb.in_buffer == 0)
        return forced

    def _add_core_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    ) -> dict[str, str]:
        """Wire up forced spills and the per-buffer residency gate. Returns
        ``name -> reason`` for every buffer pinned non-resident up front, so the
        solve can log why each buffer was dropped to HBM. In the joint model the
        gate is the slicing match, driven entirely by the precomputed
        ``cd_parent_matches`` pairs; placement-only buffers have no gate."""
        forced = self._trim_oversized_tensors(model, bufs)
        for sb in bufs.values():
            why = sb.constrain_residency(model, children_of.get(sb.name, []), bufs)
            if why is not None:
                forced.setdefault(sb.name, why)
        return forced

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
        to the lowest free address (preserving merges, never raising the
        peak)."""
        by_name = {name: sb.buffer for name, sb in bufs.items()}
        spilled = {
            name for name, sb in bufs.items() if not solver.BooleanValue(sb.in_buffer)
        }
        footprint = {name: sb.footprint(solver) for name, sb in bufs.items()}

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
            offsets = self._justify(units)
        else:
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
    def _justify(units: list[_PlacementUnit]) -> dict[str, int]:
        """Slide each placement unit down to the lowest free address. Processing
        in current-base order and giving each the lowest non-conflicting slot
        preserves the relative stacking, so the peak never increases -- it only
        squeezes out the float gaps the search leaves. Returns a name -> address
        map."""
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
            u.justified_offset = base
            placed.append(u)
            for n in u.members:
                offsets[n] = base
        return offsets
