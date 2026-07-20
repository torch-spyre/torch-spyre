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
  reside (``_implicate_core_division``).
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
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
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
    MemoryPlanSolver,
    _assert_in_place_relationships,
    SolveError,
    BufferType,
)

__all__ = ["CpSatLayoutSolver"]

logger = logging.getLogger(__name__)

# Drop cause for a buffer the solver chose to spill (rather than one pinned out
# up front by _add_core_division): it fit but residency gave no benefit, or
# there was no room once higher-value buffers were placed. Shared so the DEBUG
# log and the reasons surfaced to the allocator agree.
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    original_offset: int  # offset the solver chose, before bottom-justify
    justified_offset: int = 0  # final justified offset


def _assert_core_divisions_enumerated(buffers: Sequence[CoreDivisionBuffer]):
    """Assert that all buffers have enumerated core divisions."""
    for b in buffers:
        assert len(b.core_divisions) != 0, (
            "All buffers must have at least 1 valid core division"
        )


@dataclass
class _CoreDivisionBufferWithCpVars:
    """A :class:`CoreDivisionBuffer` bundled with the CP-SAT variables the solver
    creates for it, so one object flows through the solve instead of a buffer
    list shadowed by a parallel ``name -> {var}`` dict.

    The buffer spans ``[buffer.start_time, buffer.end_time)``; the vars encode
    where (``offset``) and whether (``in_buffer``) it resides in LX, and the
    chosen core division (``division``) with its per-core footprint
    (``eff_size``) and total core usage (``cores`` = ``cores_used``, including
    any reduction-axis split). ``merge_vars`` maps each in-place parent name to
    the merge bool for that parent->this edge.

    CP-SAT variables must be created against a model, so this wrapper takes the
    model and the unit capacity ``M`` and creates only the variables here; the
    constraints tying them together are added by the solver methods."""

    buffer: CoreDivisionBuffer
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
        self.merge_vars = {
            parent: m.new_bool_var(f"merge_{parent}_{b.name}")
            for parent in b.in_place_parents
        }

        # tie per-core footprint (output split only) and total core usage to the
        # chosen division index
        self.model.add_element(self.division, per_core, self.eff_size)
        self.model.add_element(self.division, cores_used, self.cores)


class CpSatLayoutSolver(MemoryPlanSolver[CoreDivisionBuffer]):
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
        self, buffers: Sequence[CoreDivisionBuffer], log_lx_usage: bool = False
    ) -> list[CoreDivisionBuffer]:
        self.spill_reasons = {}
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )
        _assert_in_place_relationships(buffers)
        _assert_core_divisions_enumerated(buffers)

        model = cp_model.CpModel()
        # Solve on copies so we never mutate the caller's buffers.
        working = {
            b.name: _CoreDivisionBufferWithCpVars(
                replace(
                    b,
                    size=int(np.ceil(b.size / self.alignment)),
                ),
                model,
                self._capacity_units,
            )
            for b in buffers
        }

        offsets, spilled, chosen_div, forced_reasons = self._run(model, working)
        offsets = {k: v * self.alignment for k, v in offsets.items()}
        # Surface a drop cause for every spilled buffer: the pre-solve forced
        # reason when we have one, otherwise the solver chose to spill it.
        self.spill_reasons = {
            name: forced_reasons.get(name, _SOLVER_CHOSE_SPILL) for name in spilled
        }

        for b in buffers:
            b.address = None if b.name in spilled else offsets.get(b.name)
            b.chosen_division = chosen_div.get(b.name, b.chosen_division)
        return list(buffers)

    # ------------------------------------------------------------------
    # Model build + solve
    # ------------------------------------------------------------------
    def _run(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> tuple[dict[str, int], set[str], dict[str, int], dict[str, str]]:
        children_of = self._get_children(tensors)
        self._add_inplace_relaxation(model, tensors)
        forced_reasons = self._add_core_division(model, tensors, children_of)

        solver = cp_model.CpSolver()
        if self._time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(self._time_limit_seconds)
        solver.parameters.num_search_workers = (
            1 if torch.are_deterministic_algorithms_enabled() else os.cpu_count()
        )
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0

        # TODO: Update objective to a maxmin optimization to optimize overall
        # throughput.
        def spill_cost(sb: "_CoreDivisionBufferWithCpVars") -> int:
            reads = len(children_of.get(sb.name, [])) + sb.buffer.unallocated_reads
            if sb.buffer.boundary == BufferType.Input and reads > 0:
                reads -= 1
            writes = 1 if sb.buffer.boundary == BufferType.Intermediate else 0
            return (reads + writes) * sb.buffer.size

        hbm_terms = [spill_cost(sb) * (1 - sb.in_buffer) for sb in tensors.values()]
        if hbm_terms:
            model.minimize(sum(hbm_terms))
            if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError("CP-SAT memory planner found no feasible plan")
            # Lock in the residency optimum (the traffic value, not just the
            # count) so phase 2 can never trade a spill for parallelism.

            # Rounding avoids loss of precision as the objective function is
            # the sum and multiplication of integers.
            model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))

        # Phase 2 -- parallelism: holding the residency optimum, maximize total
        # core usage so every buffer (resident or spilled) takes its most
        # parallel division.
        model.maximize(sum(sb.cores for sb in tensors.values()))
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise SolveError("CP-SAT memory planner found no feasible plan")

        offsets, spilled, chosen_div = self._extract(solver, tensors)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CP-SAT layout solver] tensors=%d resident=%d occupancy=%d "
                "status=%s walltime=%.2f ms",
                len(tensors),
                len(tensors) - len(spilled),
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

        return offsets, spilled, chosen_div, forced_reasons

    def _add_inplace_relaxation(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
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
                # active merge => child reuses the parent's exact per-core storage,
                # so their chosen divisions must have equal per-core footprints.
                model.add(dst_v.eff_size == src_v.eff_size).OnlyEnforceIf(edge)
                # active merge => parent and child must pick slicing-compatible divisions
                self._constrain_merge_division(model, bufs, src, dst, edge)
                outgoing.setdefault(src, []).append(edge)
                incoming.setdefault(dst, []).append(edge)

        for ms in (*incoming.values(), *outgoing.values()):
            if len(ms) > 1:
                model.add_at_most_one(ms)

        for sb in bufs.values():
            # if a buffer is resident its top must be below the peak usage.
            model.add(sb.offset + sb.eff_size <= M).OnlyEnforceIf(sb.in_buffer)

        self._add_no_overlap_2d(model, bufs, outgoing)

    def _constrain_merge_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
        src: str,
        dst: str,
        m,
    ) -> None:
        """Gate an in-place merge on slicing-compatible divisions: when ``m`` is
        active, parent (``src``) and child (``dst``) must pick divisions that
        induce the same per-core slicing of the shared storage. Uses the
        precomputed ``cd_parent_matches`` pairs (see ``_implicate_core_division``);
        no pairs => merge forbidden."""
        pv, cv = bufs[src], bufs[dst]
        compatible = bufs[dst].buffer.cd_parent_matches.get(src, [])
        self._gate_divisions(model, compatible, pv.division, cv.division, m)

    @staticmethod
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

    def _add_no_overlap_2d(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
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
        self, bufs: dict[str, _CoreDivisionBufferWithCpVars]
    ) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
        """parent name -> list of (child name, match_pairs), where ``match_pairs``
        is the child's ``cd_parent_matches[parent]`` (empty when the edge has no
        compatible division). The child's ``parents`` define the edges."""
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]] = {}
        for sb in bufs.values():
            t = sb.buffer
            for parent in t.parents:
                children_of.setdefault(parent, []).append(
                    (t.name, t.cd_parent_matches.get(parent, []))
                )
        return children_of

    def _trim_oversized_tensors(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> dict[str, str]:
        """Pin out of LX the buffers whose non-residency is fixed up front: those
        whose *smallest* candidate footprint still exceeds capacity, and those
        the allocator marked non-resident (``residency_reason`` set). Returns
        ``name -> reason`` for the buffers it forces out (drop-cause debug
        logging), using the allocator's specific reason when it has one."""
        forced: dict[str, str] = {}
        for sb in bufs.values():
            t = sb.buffer
            min_size = min(
                int(np.ceil(t.size / cd.output_partition)) for cd in t.core_divisions
            )
            if min_size > self._capacity_units:
                forced[t.name] = (
                    f"min per-core footprint {min_size} > LX capacity "
                    f"{self._capacity_units} (alignment units)"
                )
                model.add(sb.in_buffer == 0)
            elif not t.residency_allowed:
                forced[t.name] = (
                    t.residency_reason or "residency not allowed by allocator"
                )
                model.add(sb.in_buffer == 0)
        return forced

    def _implicate_core_division(
        self,
        model: "cp_model.CpModel",
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> dict[str, str]:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs. A
        buffer with no consumer edge, or with a consumer that has no compatible
        pair, can never reside. Returns ``name -> reason`` for the buffers the
        gate forces out (drop-cause debug logging)."""
        forced: dict[str, str] = {}
        for sb in bufs.values():
            t = sb.buffer
            kids = children_of.get(t.name, [])
            if not kids:
                if t.unallocated_reads:
                    # Read only by consumers the solver never sees (filtered-out
                    # ops / graph outputs). They still read it from LX when it
                    # resides, so residency is worthwhile; there is no resident
                    # consumer to constrain the division against, so no gate.
                    # TODO: Remove this when the other solvers are brought to parity
                    continue
                # Nothing consumes this buffer from LX -> it can never reside.
                forced.setdefault(t.name, "no consumer reads it from LX")
                model.add(sb.in_buffer == 0)
                continue
            for _child, compatible in kids:
                if not compatible:
                    # This child can never match -> the buffer cannot reside.
                    forced.setdefault(
                        t.name,
                        f"consumer {_child} has no slicing-compatible core division",
                    )
                    model.add(sb.in_buffer == 0)
                    break
                pair_lits = []
                for i, j in compatible:
                    lit = model.new_bool_var("")
                    model.add(sb.division == i).OnlyEnforceIf(lit)
                    model.add(bufs[_child].division == j).OnlyEnforceIf(lit)
                    pair_lits.append(lit)
                model.add_bool_or(pair_lits).OnlyEnforceIf(sb.in_buffer)
        return forced

    def _add_core_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    ) -> dict[str, str]:
        """Wire up forced spills and the slicing-match gate. Returns ``name ->
        reason`` for every buffer pinned non-resident up front, so the solve can
        log why each buffer was dropped to HBM. Matching is driven entirely by the
        precomputed ``cd_parent_matches`` pairs."""
        forced = self._trim_oversized_tensors(model, bufs)
        for name, why in self._implicate_core_division(
            model, children_of, bufs
        ).items():
            forced.setdefault(name, why)
        return forced

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    def _extract(
        self,
        solver: "cp_model.CpSolver",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        """Read the solution into (offsets, spilled, chosen_div). When
        bottom_justify is set, slide each placement unit down to the lowest free
        address (preserving in-place merges, never raising the peak)."""
        by_name = {name: sb.buffer for name, sb in bufs.items()}
        spilled = {
            name for name, sb in bufs.items() if not solver.BooleanValue(sb.in_buffer)
        }
        chosen_div = {name: solver.Value(sb.division) for name, sb in bufs.items()}

        def footprint(t: CoreDivisionBuffer) -> int:
            return int(
                np.ceil(t.size / t.core_divisions[chosen_div[t.name]].output_partition)
            )

        if not self._bottom_justify:
            return (
                {
                    name: solver.Value(sb.offset)
                    for name, sb in bufs.items()
                    if solver.BooleanValue(sb.in_buffer)
                },
                spilled,
                chosen_div,
            )

        # A placement unit is a connected component of active merge edges: its
        # members share one base (the merge equalities), so the component slides
        # as a single block and in-place reuse is preserved.
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
                footprint=max(footprint(by_name[n]) for n in names),
                start_time=min(by_name[n].start_time for n in names),
                end_time=max(by_name[n].end_time for n in names),
                original_offset=solver.Value(bufs[names[0]].offset),
            )
            for names in components.values()
        ]
        return self._justify(units), spilled, chosen_div

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
