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
  over residency (resident buffers contribute 0) -- plus, when the solve chooses
  tilings, each cut's copy traffic (``_cut_traffic_terms``). An intermediate costs
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

**LX relayouts** ride on the same machinery. The allocator hands the solver one
``RelayoutCopyBuffer`` per relayout group (a source and one destination per-core
view, however many consumers read it), live from the group's first consumer to
its last; the copy's residency IS the decision to shuffle, its rectangle sits in
the same 2D no-overlap as every other buffer, and its price is an ordinary term
of the shared objective (``RelayoutCopyBuffer.cost_term``: the fitted shuffle
cost of the source's chosen division, charged while the copy is resident). What
this module adds is only the coupling the data cannot carry
(``_constrain_relayout_copies`` and the relaxed gate in
``constrain_residency``): a resident copy needs its source resident under a
division it was priced for, a consumer reads the copy only under a division
pair its candidates list, and a resident copy serves at least one consumer. The
gate thus becomes "slicing match, or a resident copy serving this edge". Under
the fallback objective a shuffle is unpriced and would look free, so every copy
is pinned out there.

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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
)
import sympy
import torch


if TYPE_CHECKING:
    from ortools.sat.python import cp_model

    _CP_MODEL_IMPORT_ERROR = ""
else:
    # Why ortools failed to import: often a missing dependency of it, not ortools itself.
    _CP_MODEL_IMPORT_ERROR = ""
    try:
        from ortools.sat.python import cp_model

    except ImportError as exc:  # pragma: no cover - only when ortools is absent
        cp_model = None
        _CP_MODEL_IMPORT_ERROR = str(exc)

from torch_spyre._inductor.scratchpad.cpsat_expr_printer import cost_expr_to_cpsat
from torch_spyre._inductor.scratchpad.lx_relayout import ChosenRelayout
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    ceil_div,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    RelayoutCopyBuffer,
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

# CP-SAT subsolvers the layout solve never runs (see where the solver
# parameters are set).
_CRASHING_SUBSOLVERS = (
    "variables_shaving",
    "variables_shaving_no_lp",
    "variables_shaving_max_lp",
)

# Buffer type the wrapper carries: the base placement wrapper holds any
# LifetimeBoundBuffer; the joint subclass binds this to CoreDivisionBuffer.
_BufT = TypeVar("_BufT", bound=LifetimeBoundBuffer)

# A lifetime bound: fixed by op order, or chosen by the solve.
_Bound = Union[int, "cp_model.IntVar"]


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    original_offset: int  # offset the solver chose, before bottom-justify
    justified_offset: int = 0  # final justified offset


@dataclass(frozen=True)
class _TilingGroups:
    """The loop nests the solve may form, as built by ``_add_tiling_groups``."""

    # Loop group id per tileable op. Two ops share a nest exactly when their
    # ids agree; ids are only comparable within one segment.
    group: dict[str, "cp_model.IntVar"]
    # Segment index per tileable op: a maximal stretch of tileable ops at
    # consecutive positions. Ops in different segments never share a nest, and
    # an op with no entry can never be tiled.
    segment: dict[str, int]
    # Each segment's ops in op order: the longest nest they could form.
    runs: list[list[str]]
    # Per op except a run's first: true iff it shares its predecessor's nest.
    joined: dict[str, "cp_model.IntVar"]


def _chose_a_tiling(buffer: LifetimeBoundBuffer) -> bool:
    return (
        isinstance(buffer, CoreDivisionBuffer)
        and buffer.chosen_division is not None
        and not buffer.core_divisions[buffer.chosen_division].tiling.is_untiled
    )


def _op_readers(
    children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    tensors: dict[str, "_LifetimeBufferWithCpVars"],
    name: str,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """``name``'s consumers that are operations. A relayout copy lists the
    buffer it would copy as its parent too, but it is an option the solve may
    leave unused, not a reader the buffer has to serve."""
    return [
        (child, pairs)
        for child, pairs in children_of.get(name, [])
        if not isinstance(tensors[child].buffer, RelayoutCopyBuffer)
    ]


# TODO: replace the synthetic copy slots with variable lifetime bounds, so a
# cut's copy op gets a position of its own instead of a slot reserved after
# every op.
class _Slots:
    """Time in the placement model: op ``t`` writes in slot ``2 * t`` and the
    copy op a cut inserts right after it runs in slot ``2 * t + 1``."""

    @staticmethod
    def write_slot(time: int) -> int:
        return 2 * time

    @staticmethod
    def end_slot(end_time: int) -> int:
        """The slot form of an exclusive end: the last use's own slot, plus one."""
        return 2 * end_time - 1

    @staticmethod
    def horizon(tensors: dict[str, "_LifetimeBufferWithCpVars"]) -> int:
        """A slot no lifetime reaches: past the last buffer's end and one copy."""
        return max(_Slots.end_slot(sb.end_time) for sb in tensors.values()) + 2


def _gate_divisions(model, compatible, src_div, dst_div, enforce_lit) -> None:
    """Enforce, when ``enforce_lit`` is true, that ``(src_div, dst_div)`` is
    one of the ``compatible`` (i, j) pairs. With no compatible pairs the
    relation is unsatisfiable, so ``enforce_lit`` is forced false."""
    if not compatible:
        model.Add(enforce_lit == 0)
        return
    # Let CP-SAT encode the relation as a table, instead of introducing a
    # Boolean and two implications for every compatible pair on every edge.
    model.add_allowed_assignments([src_div, dst_div], compatible).only_enforce_if(
        enforce_lit
    )


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
        # Relayout state (populated only by the joint subclass; kept here so
        # every solver method can iterate uniformly). Per parent this buffer
        # could read through a relayout copy: the (served literal, copy wrapper)
        # pairs minted in constrain_residency. A served literal means "this
        # consumer reads the parent from that copy", which pins the division
        # pair and requires the copy resident.
        self.relayout_reads: dict[str, list[tuple[Any, Any]]] = {}
        # The (start, end) in slots the placement model gave this buffer's box,
        # set by ``_add_no_overlap_2d``.
        self.slot_span: Optional[tuple[_Bound, _Bound]] = None
        # Tiling variables. ``_add_tiling`` sets all but ``end`` on a buffer
        # with core divisions, and only when the menus offer a tiling; they
        # stay ``None``/empty otherwise.
        # Id of the loop nest the chosen division runs in, one id per distinct
        # ``tiling.level_counts``; 0 is the untiled nest.
        self.loop_id: Optional[cp_model.IntVar] = None
        # True iff ``loop_id != 0``: the op runs tile by tile.
        self.tiled: Optional[cp_model.IntVar] = None
        # True iff the op is tiled and some reader does not share its nest, so
        # a copy op writes each tile out to the full buffer in HBM. The
        # objective charges each cut.
        self.cut: Optional[cp_model.IntVar] = None
        # Slot at which this op's nest finishes: right after its own write
        # (plus one slot for the copy when cut), or the end of the next op's
        # nest when the two are joined in one nest. ``None`` marks an op that
        # cannot be tiled.
        self.nest_end: Optional[cp_model.IntVar] = None
        # Slot at which this op's nest begins: its own write, or the previous
        # op's ``nest_start`` when the two are joined. ``None`` as for
        # ``nest_end``.
        self.nest_start: Optional[cp_model.IntVar] = None
        # Per reader: true iff the reader takes the copied-out full buffer from
        # HBM instead of this buffer's tile. That waives the division match a
        # resident buffer owes the reader and lets it free LX before the read.
        self.via_full: dict[str, cp_model.IntVar] = {}
        # Slot at which the buffer frees its LX space, set by
        # ``_add_lifetime_ends`` when a tiling can change its lifetime: a tiled
        # buffer can free early, and one read inside a nest stays to its end.
        self.end: Optional[cp_model.IntVar] = None
        # Slot at which an input clone starts holding its LX space, set by
        # ``_add_lifetime_ends`` when a tiled op reads it: a clone hoisted to
        # the nest entry is written before the nest's first op.
        self.start: Optional[cp_model.IntVar] = None

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

    def constrain_residency(self, model, kids, bufs, copies) -> None:
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

    def decision_variables(self) -> list:
        """The CP-SAT variables that define this buffer's plan. The LX offset is
        left out: no objective prices an address, so every legal shift of a
        resident buffer would read as a tie."""
        served = [lit for reads in self.relayout_reads.values() for lit, _ in reads]
        return [self.in_buffer, *self.merge_vars.values(), *served]


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

        # Per-core LX footprint under each division: the output partition AND any
        # coarse tiling shrink it (a tiled op keeps only one tile resident at a
        # time), mirroring ``CoreDivisionBuffer.min_footprint``. Pricing tiling
        # here is what lets the residency objective prefer a tiled candidate.
        per_core = [b.per_core_size(i) for i in range(len(b.core_divisions))]
        # Total cores the op runs on under each division -- includes any
        # reduction-axis split, so a reduction-parallel division counts its full
        # parallelism (``output_partition`` alone would score it as 1 core).
        cores_used = [cd.cores_used for cd in b.core_divisions]
        # Balance heuristic: the sum of squared per-axis split factors. For a
        # fixed core count (product of the factors, held at the parallelism
        # optimum) this is smallest when the split is spread across more axes
        # with smaller factors, so minimizing it favours a balanced division
        # over one that hammers a single axis (e.g. 2x2 over 4x1, both four
        # cores).
        core_cost = [
            sum(split**2 for split in cd.splits.values()) for cd in b.core_divisions
        ]
        # For a relayout copy only: the served literals of the consumer edges
        # it can carry, filled by the sources' constrain_residency and consumed
        # by _constrain_relayout_copies ("a resident copy serves someone").
        self.serves: list[Any] = []
        self.cores_used = cores_used
        if len(b.core_divisions) == 1:
            # Nothing to choose: bind every division-derived quantity as a
            # constant instead of an element lookup on a fixed index. Every
            # relayout copy is such a buffer, and with hundreds of them the
            # free eff_size/cores/core_cost/split integers (domains up to a few
            # thousand) behind one-entry elements made CP-SAT's presolve scale
            # super-linearly in the copy count: 40 s at 160 copies, past the
            # 120 s limit at 312, on the 304-op spyre_attn decode graph.
            self.division = m.new_constant(0)
            self.eff_size = per_core[0]
            self.core_cost = core_cost[0]
            self.cores = cores_used[0]
            only = b.core_divisions[0]
            self.cp_core_divs = {
                key: only.splits.get(key, 1) for key in b.sym_core_divs
            }
            self.cp_core_divs_raw = {key: [v] for key, v in self.cp_core_divs.items()}
            self.cp_tile_counts = {
                key: only.tiling.count_for(key) for key in b.sym_tile_counts
            }
            self.cp_tile_counts_raw = {
                key: [v] for key, v in self.cp_tile_counts.items()
            }
            true, false = m.new_constant(1), m.new_constant(0)
            self.division_is = lambda i: true if i == 0 else false
            return
        self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
        self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")
        self.core_cost = m.new_int_var(0, max(core_cost), f"core_cost_{b.name}")
        self.cores = m.new_int_var(min(cores_used), max(cores_used), f"occ_{b.name}")

        cp_core_divs: dict = {}
        cp_core_divs_raw: dict = {}
        for key, symbol in b.sym_core_divs.items():
            assert isinstance(symbol, sympy.Symbol)
            raw = [cd.splits.get(key, 1) for cd in b.core_divisions]
            cp_var = m.new_int_var(1, config.sencores, f"{symbol.name}")
            m.add_element(self.division, raw, cp_var)
            cp_core_divs[key] = cp_var
            cp_core_divs_raw[key] = raw

        self.cp_core_divs = cp_core_divs
        self.cp_core_divs_raw = cp_core_divs_raw

        # The coarse-tiling counts, tied to the division index the same way:
        # each candidate division carries its own tiling.
        cp_tile_counts: dict = {}
        cp_tile_counts_raw: dict = {}
        for key, symbol in b.sym_tile_counts.items():
            raw = [cd.tiling.count_for(key) for cd in b.core_divisions]
            cp_var = m.new_int_var(min(raw), max(raw), symbol.name)
            m.add_element(self.division, raw, cp_var)
            cp_tile_counts[key] = cp_var
            cp_tile_counts_raw[key] = raw

        self.cp_tile_counts = cp_tile_counts
        self.cp_tile_counts_raw = cp_tile_counts_raw

        # tie per-core footprint (output split only) and total core usage to the
        # chosen division index
        m.add_element(self.division, per_core, self.eff_size)
        m.add_element(self.division, cores_used, self.cores)
        m.add_element(self.division, core_cost, self.core_cost)

        self.division_is = cache(self._division_is)

    def _division_is(self, i: int) -> Any:
        """The literal ``division == i`` (both directions enforced)."""
        lit = self.model.new_bool_var(f"div_{self.name}_is_{i}")
        self.model.add(self.division == i).only_enforce_if(lit)
        self.model.add(self.division != i).only_enforce_if(lit.Not())
        return lit

    @property
    def parents(self) -> list[str]:
        return self.buffer.parents

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return self.buffer.cd_parent_matches.get(parent, [])

    def constrain_residency(self, model, kids, bufs, copies) -> None:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs, or
        the consumer must read it through a resident relayout copy.

        ``via_full`` exempts a consumer that reads this buffer's copied-out
        full buffer: that buffer is in HBM, so the consumer puts no demand on
        how the tile is sliced.

        This is the part of residency that genuinely depends on the solver's
        free variables, so it stays here as a constraint. The precomputable
        parts -- having no LX reader at all, or a consumer with no compatible
        pair -- are decided by the allocator and arrive as ``read_count`` /
        ``residency_reason``. A consumer with no compatible pair still lands
        correctly if it slips through: ``_gate_divisions`` forces ``in_buffer``
        false when the pair list is empty.

        ``copies`` maps a relayout group key to the wrapper of its
        ``RelayoutCopyBuffer``. For each consumer edge with priced candidates
        on a group that has a copy in this solve, a *served* literal says "the
        consumer reads this buffer from that copy": it pins the division pair
        to one the candidates list (``_gate_divisions`` over the group's
        pairs) and requires the copy resident. The gate then relaxes to
        "match or served". The served literals are recorded on the consumer
        (``relayout_reads``) for extraction and on the copy's tally for the
        "a resident copy serves someone" constraint."""
        for child, compatible in kids:
            child_w = bufs[child]
            served: list = []
            by_group: dict[tuple[str, int], list] = {}
            for candidate in child_w.buffer.cd_parent_relayouts.get(self.name, ()):
                if candidate.group_key in copies:
                    by_group.setdefault(candidate.group_key, []).append(candidate)
            for key, candidates in sorted(by_group.items()):
                copy_w = copies[key]
                lit = model.new_bool_var(f"served_{self.name}__{child}__g{key[1]}")
                _gate_divisions(
                    model,
                    [(c.source_division, c.consumer_division) for c in candidates],
                    self.division,
                    child_w.division,
                    lit,
                )
                model.add_implication(lit, copy_w.in_buffer)
                served.append(lit)
                child_w.relayout_reads.setdefault(self.name, []).append((lit, copy_w))
                copy_w.serves.append(lit)
            # A consumer outside this tiled buffer's nest reads the copied-out
            # full buffer in HBM, not the tile, so its edge puts no division
            # demand on the tile (``via_full``; None when it cannot be tiled).
            exempt = self.via_full.get(child)
            if not served and exempt is None:
                _gate_divisions(
                    model, compatible, self.division, child_w.division, self.in_buffer
                )
                continue
            match_lit = model.new_bool_var(f"match_{self.name}__{child}")
            _gate_divisions(
                model, compatible, self.division, child_w.division, match_lit
            )
            ways = [match_lit, *served]
            if exempt is not None:
                ways.append(exempt)
            model.add_bool_or(ways).only_enforce_if(self.in_buffer)

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
        return self.buffer.per_core_size(solver.Value(self.division))

    def record_division(self, solver: "cp_model.CpSolver") -> None:
        self.buffer.chosen_division = solver.Value(self.division)

    def decision_variables(self) -> list:
        return [*super().decision_variables(), self.division]


class CpSatLayoutSolver(CoreDivisionLayoutSolver):
    """Joint core-division + LX placement via an OR-Tools CP-SAT search
    (``config.layout_solver == "cpsat"``). See the module docstring for the
    model (joint division, slicing-match residency gate, 2D no-overlap with
    in-place lifetime shortening) and the lexicographic objective
    (residency, then parallelism, then division balance).
    """

    @classmethod
    def supports_paired_buffers(cls) -> bool:
        return False

    @classmethod
    def decides_lx_relayouts(cls) -> bool:
        return True

    @classmethod
    def generates_divisions(cls) -> bool:
        return False

    @classmethod
    def chooses_tilings(cls) -> bool:
        return True

    @classmethod
    def tilings_from_menu(cls) -> bool:
        return True

    @classmethod
    def linear_cost_only(cls) -> bool:
        return True

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        time_limit_seconds: Optional[float] = None,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, which "
                f"could not be imported ({_CP_MODEL_IMPORT_ERROR or 'not installed'}). "
                "Install it with 'pip install ortools' "
                "or select a different layout_solver (e.g. 'greedy')."
            )
        super().__init__(buffers, size, alignment)
        # What the last solve cost and returned, for the cost-expression dump.
        # Here rather than on the base class: this is the only solver that
        # reports it, and the allocator reads it with a default, so the base
        # contract does not change. Empty until a solve, so a reader can tell
        # "not recorded" from "no solve".
        self.last_solve_stats: dict = {}
        # The solver works in alignment-sized units so every offset it picks is
        # automatically aligned; plan_layout scales sizes/offsets in and out.
        self._capacity_units = self.limit // self.alignment
        self._time_limit_seconds = (
            config.cpsat_time_limit_seconds
            if time_limit_seconds is None
            else time_limit_seconds
        )
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

        # Copy the solved results back onto the caller's buffers. Offsets come
        # back in alignment units (the solver works in aligned units), so scale
        # the address to bytes on the way out.
        for b in buffers:
            sb = solved[b.name]
            b.address = None if sb.address is None else sb.address * self.alignment
            # Surface a drop cause for every spilled buffer: the pre-solve forced
            # reason when we have one, the untiled bar when the op stayed
            # untiled, otherwise the solver chose to spill it.
            reason = forced_reasons.get(b.name)
            if reason is None and not _chose_a_tiling(sb):
                reason = b.untiled_bar
            b.spill_reason = (
                (reason or _SOLVER_CHOSE_SPILL) if sb.address is None else None
            )
            if isinstance(b, CoreDivisionBuffer) and isinstance(sb, CoreDivisionBuffer):
                b.chosen_division = sb.chosen_division
                b.chosen_relayouts = {
                    parent: chosen.scaled(self.alignment)
                    for parent, chosen in sb.chosen_relayouts.items()
                }
        return list(buffers)

    # ------------------------------------------------------------------
    # Model build + solve
    # ------------------------------------------------------------------
    def _is_unique_solution(
        self,
        model: "cp_model.CpModel",
        solver: "cp_model.CpSolver",
        tensors: dict[str, _LifetimeBufferWithCpVars],
    ) -> bool:
        """Whether the optimum ``solver`` holds is the only one over the plan's
        decision variables (residency, merges, served relayouts and division,
        not placement). False when a different plan costs the same, or when the
        probe ends without proving either way.
        """
        decision = [v for sb in tensors.values() for v in sb.decision_variables()]
        probe = model.clone()
        differs = []
        for var in decision:
            lit = probe.new_bool_var(f"differs_{var.name}")
            probe.add(var != solver.Value(var)).only_enforce_if(lit)
            differs.append(lit)
        probe.add_bool_or(differs)

        probe_solver = cp_model.CpSolver()
        probe_solver.parameters.copy_from(solver.parameters)

        # Only a probe run to completion proves anything, so it has no time
        # limit: OPTIMAL gives the runner-up's cost, INFEASIBLE means no other plan.
        probe_solver.parameters.max_time_in_seconds = float("inf")
        status = probe_solver.Solve(probe)
        if status == cp_model.INFEASIBLE:
            return True  # no other plan exists at all
        if status != cp_model.OPTIMAL:
            return False  # probe ran out of time we cannot prove uniqueness
        best, runner_up = solver.ObjectiveValue(), probe_solver.ObjectiveValue()
        if round(runner_up) <= round(best):
            return False  # a different plan costs the same to within 1
        return True  # another plan was found that is worse than the best one

    def _minimize_cost_expr(
        self,
        model: "cp_model.CpModel",
        solver: "cp_model.CpSolver",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        cost_expr: sympy.Expr,
    ) -> Optional["cp_model.CpSolverStatus"]:
        sym_map = {}
        buffer_map = {}
        for t in tensors.values():
            sym_map[t.buffer.sym_is_lx.name] = t.in_buffer
            if not isinstance(t, _CoreDivisionBufferWithCpVars):
                continue
            # The division index itself, and the wrapper behind it for the
            # KroneckerDelta lowering (a table over candidates, e.g. the
            # relayout price, selects by identity rather than by split shape).
            sym_map[t.buffer.sym_division.name] = t.division
            sym_map[f"_division_of_{t.buffer.sym_division.name}"] = t

            product = []
            for key, symbol in t.buffer.sym_core_divs.items():
                assert isinstance(symbol, sympy.Symbol)
                sym_map[symbol.name] = t.cp_core_divs[key]
                buffer_map[symbol.name] = (t, t.cp_core_divs_raw[key])
                product.append(symbol.name)
            product.sort()
            symbol = sympy.Symbol("_product_" + "_".join(product))
            sym_map[symbol.name] = t.cores
            buffer_map[symbol.name] = (t, t.cores_used)

            for key, symbol in t.buffer.sym_tile_counts.items():
                sym_map[symbol.name] = t.cp_tile_counts[key]
                buffer_map[symbol.name] = (t, t.cp_tile_counts_raw[key])

        try:
            cp_cost = cost_expr_to_cpsat(model, sym_map, buffer_map, cost_expr)
            if not isinstance(cp_cost, (int, float)):
                # if the cost is non-constant, we minimize it
                # if the cost is constant, we use any solution
                model.minimize(cp_cost)
            status = self._solve_and_record(solver, model, objective=True)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError(
                    f"CP-SAT returned {solver.StatusName(status)} without a plan "
                    f"after {solver.WallTime():.2f}s"
                )
            if (
                config.enable_uniqueness_check
                and status == cp_model.OPTIMAL
                and not self._is_unique_solution(model, solver, tensors)
            ):
                logger.warning(
                    "[CP-SAT layout solver] CP-SAT cannot prove uniqueness of the "
                    "optimal solution"
                )
            return status
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "[CP-SAT layout solver] cannot linearize the sympy expr: %s", exc
            )
            if not config._cpsat_warn_on_cost_expr:
                raise
            # The objective could not be lowered. A fallback solve follows and
            # records over this with ``objective_used`` False; this entry stands
            # only if no fallback runs, and says why.
            self.last_solve_stats = {
                "status": "NOT_LINEARIZABLE",
                "error": str(exc),
                "objective_used": False,
            }
            return None

    def _add_tiling(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    ) -> None:
        """The loop nests the chosen tilings form, as solver variables on the
        buffers (``loop_id``, ``tiled``, ``cut``, ``via_full``, ``nest_end``).

        One bool per buffer is true when that buffer is a coarse-tiling *cut*.

        A cut is a tiled op whose value has to be published into a full-sized
        buffer because some consumer sits outside its loop nest -- the
        ``kind="copy_out"`` classification ``_plan_tiling_propagation`` makes
        later, expressed over the solver's own division variables so it can be
        ranked *while* the tiling is being chosen rather than discovered after.

        Each candidate's loop nest (:attr:`TileSpec.level_counts`) is interned
        to a small integer id (the untiled nest is always 0, so ``loop_id !=
        0`` means "tiled"), and ``add_element`` ties a buffer's id to its
        chosen division exactly as ``eff_size`` and ``cores`` are already tied.
        Which ops share a nest is decided over op order, not the edge alone:
        see :meth:`_add_tiling_groups`. A consumer that shares its producer's
        nest reads it one tile at a time, so their chosen divisions must be a
        ``cd_parent_matches`` pair, whose views are owned per (tile, core); a
        pair that is not must be split by a cut. A tiled graph output, or a
        tiled buffer with no modelled consumer (one read only by an extern
        kernel), is a cut unconditionally, since its value must reach HBM
        either way.

        The copy op a cut inserts runs right after the tiled op and drains each
        tile into a full-sized HBM buffer, which is what a consumer outside the
        nest then reads (``via_full``). The tile itself stays an ordinary LX
        candidate, read by that copy. The copy walks the op's output and has
        no reduction axis, so a cut tile resides only under a division that
        splits the output alone.

        Adds nothing when no buffer carries a non-empty spec, which is every
        path except the joint solve with ``auto_coarse_tiling`` on; the tiling
        variables then stay ``None`` and every method that reads them is a
        no-op.
        """
        UNTILED_ID = 0
        divided = {
            name: sb
            for name, sb in tensors.items()
            if getattr(sb.buffer, "core_divisions", None)
        }
        # Tiled nests take ids from 1; 0 is the untiled nest.
        nest_ids: dict[tuple[int, ...], int] = {}
        for sb in divided.values():
            for cd in sb.buffer.core_divisions:
                if not cd.tiling.is_untiled:
                    nest_ids.setdefault(cd.tiling.level_counts, len(nest_ids) + 1)
        if not nest_ids:
            return

        max_id = len(nest_ids)
        tiled: dict[str, cp_model.IntVar] = {}
        for name, sb in divided.items():
            # Map each candidate in the buffer's division menu to its global
            # nest id, so ``loop_id`` follows the division the solve picks.
            ids = [
                UNTILED_ID if cd.tiling.is_untiled else nest_ids[cd.tiling.level_counts]
                for cd in sb.buffer.core_divisions
            ]
            sb.loop_id = model.new_int_var(0, max_id, f"loop_id_{name}")
            model.add_element(sb.division, ids, sb.loop_id)
            is_tiled = model.new_bool_var(f"tiled_{name}")
            model.add(sb.loop_id != UNTILED_ID).only_enforce_if(is_tiled)
            model.add(sb.loop_id == UNTILED_ID).only_enforce_if(is_tiled.negated())
            sb.tiled = tiled[name] = is_tiled

        groups = self._add_tiling_groups(model, divided, tiled)

        cuts: dict[str, cp_model.IntVar] = {}
        for name, is_tiled in tiled.items():
            sb = divided[name]
            diffs: list["cp_model.IntVar"] = []
            # A graph output is copied out whatever its consumers do.
            unshareable = getattr(sb.buffer, "boundary", None) == BufferType.Output
            seg = groups.segment.get(name)
            for child, pairs in _op_readers(children_of, tensors, name):
                # A consumer with no divisions of its own (placement-only), or
                # one an untileable op separates from this buffer, can never
                # share its nest, so reading it is always a cut.
                if seg is None or groups.segment.get(child) != seg:
                    unshareable = True
                    if seg is not None:
                        sb.via_full[child] = is_tiled
                    continue
                d = model.new_bool_var(f"apart_{name}_{child}")
                model.add(groups.group[name] != groups.group[child]).only_enforce_if(d)
                model.add(groups.group[name] == groups.group[child]).only_enforce_if(
                    d.negated()
                )
                _gate_divisions(
                    model, pairs, sb.division, divided[child].division, d.negated()
                )
                diffs.append(d)
                # An untiled producer's group id also differs from its
                # consumer's, but there is no full buffer then: the consumer
                # reads the producer itself.
                full = model.new_bool_var(f"via_full_{name}_{child}")
                model.add_bool_and([is_tiled, d]).only_enforce_if(full)
                model.add_bool_or([is_tiled.negated(), d.negated()]).only_enforce_if(
                    full.negated()
                )
                sb.via_full[child] = full

            cut = model.new_bool_var(f"cut_{name}")
            if unshareable or not diffs:
                # No modelled consumer that could share the nest: tiled => cut.
                model.add(cut == is_tiled)
            else:
                any_diff = model.new_bool_var(f"anydiff_{name}")
                model.add_max_equality(any_diff, diffs)
                model.add_bool_and([is_tiled, any_diff]).only_enforce_if(cut)
                model.add_bool_or(
                    [is_tiled.negated(), any_diff.negated()]
                ).only_enforce_if(cut.negated())
            sb.cut = cuts[name] = cut
            self._add_partial_reduction_cut_guard(model, sb)

        self._add_nest_ends(model, divided, groups, cuts)
        self._add_nest_starts(model, divided, groups)

    @staticmethod
    def _add_partial_reduction_cut_guard(
        model: "cp_model.CpModel", sb: _LifetimeBufferWithCpVars
    ) -> None:
        """A cut tile resides only under a division with no reduction split.

        A split reduction is not realized through HBM: its partial sums are
        combined core to core and the finished slice stays in LX on one core of
        each reduction group. The copy is a pointwise op that reads the tile
        core for core, so no division of its can read that layout.
        """
        menu = sb.buffer.core_divisions
        if not any(cd.reduction_splits for cd in menu):
            return
        assert sb.cut is not None, "a tileable op has a cut literal"
        copy_reads_it = model.new_bool_var(f"copy_reads_{sb.name}")
        model.add_element(
            sb.division,
            [int(not cd.reduction_splits) for cd in menu],
            copy_reads_it,
        )
        model.add_bool_or([sb.cut.negated(), sb.in_buffer.negated(), copy_reads_it])

    @staticmethod
    def _add_nest_ends(
        model: "cp_model.CpModel",
        divided: dict[str, _LifetimeBufferWithCpVars],
        groups: _TilingGroups,
        cuts: dict[str, "cp_model.IntVar"],
    ) -> None:
        """Set ``nest_end`` on every tileable op: the slot its nest ends, after
        the last op joined to it and after that op's copy when it is cut.

        Each run is walked backwards so an op takes its successor's end exactly
        when the two are joined.
        """
        for run in groups.runs:
            last_time = divided[run[-1]].buffer.uses[0]
            horizon = _Slots.write_slot(last_time) + 2
            following: Optional[_LifetimeBufferWithCpVars] = None
            for name in reversed(run):
                sb = divided[name]
                own = _Slots.write_slot(sb.buffer.uses[0]) + 1
                end = model.new_int_var(own, horizon, f"nest_end_{name}")
                if following is None:
                    model.add(end == own + cuts[name])
                else:
                    model.add(end == following.nest_end).only_enforce_if(
                        groups.joined[following.name]
                    )
                    model.add(end == own + cuts[name]).only_enforce_if(
                        groups.joined[following.name].negated()
                    )
                sb.nest_end = end
                following = sb

    @staticmethod
    def _add_nest_starts(
        model: "cp_model.CpModel",
        divided: dict[str, _LifetimeBufferWithCpVars],
        groups: _TilingGroups,
    ) -> None:
        """Set ``nest_start`` on every tileable op: the write slot of the first
        op in its nest. Each run is walked forwards so an op takes its
        predecessor's start exactly when the two are joined."""
        for run in groups.runs:
            first = _Slots.write_slot(divided[run[0]].buffer.uses[0])
            previous: Optional[_LifetimeBufferWithCpVars] = None
            for name in run:
                sb = divided[name]
                own = _Slots.write_slot(sb.buffer.uses[0])
                start = model.new_int_var(first, own, f"nest_start_{name}")
                if previous is None:
                    model.add(start == own)
                else:
                    model.add(start == previous.nest_start).only_enforce_if(
                        groups.joined[name]
                    )
                    model.add(start == own).only_enforce_if(
                        groups.joined[name].negated()
                    )
                sb.nest_start = start
                previous = sb

    @staticmethod
    def _add_tiling_groups(
        model: "cp_model.CpModel",
        divided: dict[str, _LifetimeBufferWithCpVars],
        tiled: dict[str, "cp_model.IntVar"],
    ) -> _TilingGroups:
        """The loop group each tileable op lands in, as solver variables.

        ``derive_tiling_groups`` fuses only *consecutive* ops that run the same
        non-empty loop nest, so any op between a producer and its consumer that
        does not run it splits them into two nests, whether or not it touches
        their edge. Op order is the solver's own time axis -- an op output's
        ``uses[0]`` is its producing write -- so each pair of adjacent ops gets
        a ``joined`` literal (both run the same non-empty nest) and a running
        group id that steps wherever it fails. Two ops share a nest exactly when
        their group ids agree.

        An op that can never be tiled -- one with only untiled candidates, or
        no op-output buffer in the solve at all -- always breaks the run, so it
        starts a new *segment* instead of a literal.
        """
        position = {
            name: sb.buffer.uses[0]
            for name, sb in divided.items()
            if sb.buffer.uses
            and not sb.buffer.first_use_is_read
            and not isinstance(sb.buffer, RelayoutCopyBuffer)
            and any(not cd.tiling.is_untiled for cd in sb.buffer.core_divisions)
        }
        group: dict[str, "cp_model.IntVar"] = {}
        segment: dict[str, int] = {}
        runs: list[list[str]] = []
        joined_to_prev: dict[str, "cp_model.IntVar"] = {}
        segments = 0
        prev: Optional[str] = None
        for name, pos in sorted(position.items(), key=lambda item: item[1]):
            if prev is None or pos != position[prev] + 1:
                group[name] = model.new_constant(0)
                segment[name] = segments
                segments += 1
                runs.append([name])
            else:
                same = model.new_bool_var(f"samenest_{prev}_{name}")
                loop_id, prev_loop_id = divided[name].loop_id, divided[prev].loop_id
                is_tiled = tiled[name]
                model.add(prev_loop_id == loop_id).only_enforce_if(same)
                model.add(prev_loop_id != loop_id).only_enforce_if(same.negated())
                joined = model.new_bool_var(f"joined_{prev}_{name}")
                model.add_bool_and([same, is_tiled]).only_enforce_if(joined)
                model.add_bool_or([same.negated(), is_tiled.negated()]).only_enforce_if(
                    joined.negated()
                )
                var = model.new_int_var(0, len(position), f"tile_group_{name}")
                model.add(var == group[prev] + 1 - joined)
                group[name] = var
                segment[name] = segment[prev]
                runs[-1].append(name)
                joined_to_prev[name] = joined
            prev = name
        return _TilingGroups(group, segment, runs, joined_to_prev)

    def _tile_count_terms(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
    ) -> list["cp_model.IntVar"]:
        """One int per buffer: the tile count of its chosen division's tiling.

        ``TileSpec.tile_count`` is 1 for the untiled spec, and ``add_element``
        ties it to the buffer's division the way :meth:`_tiling_model` ties
        ``loop_id``. A buffer whose candidates all tile alike (every one
        untiled, or a single division) is left out: its count is a constant
        and cannot move the sum.

        Returns an empty list when nothing carries a choice of tiling, which is
        every path except the joint solve with ``auto_coarse_tiling`` on, so the
        tile-count stage below vanishes there.
        """
        terms = []
        for name, sb in tensors.items():
            divisions = getattr(sb.buffer, "core_divisions", None)
            if not divisions:
                continue
            counts = [cd.tiling.tile_count for cd in divisions]
            if min(counts) == max(counts):
                continue
            var = model.new_int_var(min(counts), max(counts), f"tile_count_{name}")
            model.add_element(sb.division, counts, var)
            terms.append(var)
        return terms

    @staticmethod
    def _add_lifetime_ends(
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    ) -> None:
        """Set ``end``, the slot a buffer stops holding its LX space, on every
        buffer whose lifetime depends on the tiling chosen.

        The untiled graph gives a buffer one lifetime: from its write to its
        last read. Tiling changes that in two ways.

        - A tiled buffer holds one tile. A consumer outside its nest reads the
          copied-out full buffer instead, so only the consumers in the nest keep
          the tile alive, along with the copy op when the buffer is cut.
        - An untiled buffer read by a tiled op is read again on every iteration
          of that op's nest, so it stays alive until the nest ends
          (``counted_loop_lifetime_overrides`` on the applied graph). An input
          clone read by a tiled op may also be hoisted to the nest's entry, so
          its ``start`` reaches back to that op's ``nest_start``.

        Each end is bounded from below only, and each start from above only. A
        longer lifetime can only cost space, so the solve has no reason to pick
        one, and an in-place merge pins its parent's end exactly (see
        :meth:`_add_inplace_relaxation`).
        """
        horizon = _Slots.horizon(tensors)
        for name, sb in tensors.items():
            kids = [
                tensors[child] for child, _ in _op_readers(children_of, tensors, name)
            ]
            read_by_a_nest = [kid for kid in kids if kid.nest_end is not None]
            if sb.nest_end is None and not read_by_a_nest:
                # Neither this buffer nor any reader can be tiled, so no tiling
                # choice can move its end (or start): it keeps its untiled
                # lifetime and ``end`` stays None.
                continue
            untiled_end = _Slots.end_slot(sb.end_time)
            if sb.nest_end is None:
                sb.end = model.new_int_var(untiled_end, horizon, f"end_{name}")
                for kid in read_by_a_nest:
                    # It is read again on every pass of the kid's nest.
                    model.add(sb.end >= kid.nest_end).only_enforce_if(kid.tiled)
                if sb.buffer.first_use_is_read:
                    # An input clone may be hoisted to the entry of the nest
                    # that reads it, so its slot is held from the nest's start.
                    # Defense in depth: today a resident clone forces its readers
                    # untiled (``_views_for_divs``), so this never binds.
                    untiled_start = _Slots.write_slot(sb.start_time)
                    sb.start = model.new_int_var(0, untiled_start, f"start_{name}")
                    for kid in read_by_a_nest:
                        model.add(sb.start <= kid.nest_start).only_enforce_if(kid.tiled)
                continue
            assert sb.tiled is not None, "a tileable op has a tiled literal"
            untiled = sb.tiled.negated()
            written = _Slots.write_slot(sb.buffer.uses[0])
            sb.end = model.new_int_var(written + 1, horizon, f"end_{name}")
            model.add(sb.end >= untiled_end).only_enforce_if(untiled)
            # The copy op runs in the slot right after the write.
            model.add(sb.end >= written + 2).only_enforce_if(sb.cut)
            for kid in kids:
                if not kid.buffer.uses or kid.buffer.first_use_is_read:
                    continue
                read = _Slots.write_slot(kid.buffer.uses[0]) + 1
                model.add(sb.end >= read).only_enforce_if(
                    sb.via_full[kid.name].negated()
                )
            for kid in read_by_a_nest:
                model.add(sb.end >= kid.nest_end).only_enforce_if([kid.tiled, untiled])

    @staticmethod
    def _cut_traffic_terms(
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    ) -> list[cp_model.LinearExprT]:
        """HBM traffic a cut adds on top of each buffer's own spill cost.

        ``spill_cost`` prices a buffer as resident (free) or spilled (its write
        and every read). A cut buffer is neither. Its copy op writes the full
        buffer to HBM and every consumer outside the nest reads it from there,
        whether or not the tile resides; and a spilled tile is also written to
        HBM and read back by the copy. Without these terms a resident cut tile
        would look free and the residency stage would cut everywhere.
        """
        terms: list[cp_model.LinearExprT] = []
        for name, sb in tensors.items():
            cut = sb.cut
            if cut is None or sb.nest_end is None:
                continue  # never tiled, so never cut
            b = sb.buffer
            kids = [child for child, _ in _op_readers(children_of, tensors, name)]
            is_intermediate = (
                getattr(b, "boundary", BufferType.Intermediate)
                == BufferType.Intermediate
            )
            resident_cut = model.new_bool_var(f"resident_cut_{name}")
            model.add_bool_and([cut, sb.in_buffer]).only_enforce_if(resident_cut)
            model.add_bool_or([cut.negated(), sb.in_buffer.negated()]).only_enforce_if(
                resident_cut.negated()
            )
            # Spilled: the tile's own write and the copy's read of it. The
            # copy's write is in ``spill_cost`` already, as the buffer's write.
            terms.append(2 * b.size * (cut - resident_cut))
            # Resident: the copy's write, unless it is the graph output's own
            # write-out, and the reads of every consumer the model does not
            # carry (an extern kernel, say), which can only be outside the nest.
            unmodelled = max(0, b.read_count - len(kids))
            copy_out = (1 if is_intermediate else 0) + unmodelled
            if copy_out:
                terms.append(copy_out * b.size * resident_cut)
            for kid in kids:
                outside = model.new_bool_var(f"reads_full_{name}_{kid}")
                via_full = sb.via_full[kid]
                model.add_bool_and([via_full, sb.in_buffer]).only_enforce_if(outside)
                model.add_bool_or(
                    [via_full.negated(), sb.in_buffer.negated()]
                ).only_enforce_if(outside.negated())
                terms.append(b.size * outside)
        return terms

    def _solve_and_record(
        self,
        solver: "cp_model.CpSolver",
        model: "cp_model.CpModel",
        *,
        objective: bool = False,
    ) -> int:
        """Solve, and stash what it cost and returned for the cost-expression
        dump. The one way this class solves.

        Recording is bound to solving rather than left to each call site:
        ``_run`` solves in its own occupancy passes when
        ``_minimize_cost_expr`` returns no status, and a site that solved
        without recording would leave its plan described by an earlier call's
        numbers -- silently wrong data rather than an error. The two paths are
        exclusive (the fallbacks sit under ``if status is None``), so the last
        record always describes the solve that produced this plan.

        Nothing else in the pipeline records this, so "why was that compile
        slow" currently has no artifact behind it.
        """
        status = solver.Solve(model)
        self.last_solve_stats = {
            "status": solver.StatusName(status),
            "solve_s": round(solver.WallTime(), 3),
            "variables": len(model.proto.variables),
            "constraints": len(model.proto.constraints),
            "limit_s": solver.parameters.max_time_in_seconds or None,
            "objective_used": objective,
        }
        return status

    def _run(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        forced_reasons: dict[str, str],
        cost_expr: sympy.Expr | None,
    ) -> dict[str, LifetimeBoundBuffer]:
        children_of = self._get_children(tensors)
        # Relayout copies are ordinary buffers to the placement model (the
        # in-place relaxation and its 2D no-overlap need nothing special); the
        # residency gate and the coupling below reference them by group.
        copies = self._relayout_copies(tensors)
        self._add_tiling(model, tensors, children_of)
        self._add_lifetime_ends(model, tensors, children_of)
        self._add_inplace_relaxation(model, tensors)
        self._add_core_division(model, tensors, children_of, forced_reasons, copies)
        self._constrain_relayout_copies(model, tensors, copies)

        solver = cp_model.CpSolver()
        if self._time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(self._time_limit_seconds)
        # Presolve runs on every model, priced or not: the lin_max proxy
        # variables (see _SympyExprToCpSat._lin_max_operand) removed the
        # subset-sum domain work that let it consume the budget, and without it
        # the LNS workers on a priced model can run out of memory. The copy-count
        # threshold remains as an opt-in escape hatch (off by default).
        free_copies = sum(
            isinstance(
                tensors.get(copy_w.buffer.relayout_parent),
                _CoreDivisionBufferWithCpVars,
            )
            for copy_w in copies.values()
        )
        max_copies = config.lx_solver_relayout_presolve_max_copies
        if max_copies > 0 and free_copies > max_copies:
            solver.parameters.cp_model_presolve = False
            logger.info(
                "[CP-SAT layout solver] %d relayout copies exceed the presolve "
                "threshold of %d; solving without presolve",
                free_copies,
                max_copies,
            )
        solver.parameters.num_search_workers = (
            1
            if torch.are_deterministic_algorithms_enabled()
            else config.get_cpu_count()
        )
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0
        # OR-tools 9.15's variable-shaving workers can segfault on a model with
        # a NoOverlap2D, which every LX layout model has: when copying the
        # sub-model around one variable finds it infeasible, they still index
        # the interval mapping the failed copy left empty. The crash is rarer
        # than one in several hundred solves on an idle host and far more
        # frequent under CPU contention. They only tighten domains for the
        # other workers, so leaving them out costs no plan quality we measured.
        # TODO: drop this workaround once the pinned OR-tools has the fix for
        # google/or-tools#5532 (tracked in torch-spyre#5727).
        solver.parameters.ignore_subsolvers.extend(_CRASHING_SUBSOLVERS)

        # Loop-group boundaries the tiling implies. Empty unless the joint
        # solve is actually choosing tilings, which makes the cut stage inert.
        cut_terms = [sb.cut for sb in tensors.values() if sb.cut is not None]
        if cut_terms:
            logger.debug(
                "[CP-SAT layout solver] cut tiebreak over %d candidate cut(s)",
                len(cut_terms),
            )
        # Tile counts, so the last stage can prefer the coarsest tiling. Empty
        # unless the joint solve is choosing tilings, like ``cut_terms``.
        tile_terms = self._tile_count_terms(model, tensors)

        status = None
        core_terms = None
        occupancy: Optional[int] = None

        def _solve_stage(stage: str) -> int:
            result = self._solve_and_record(solver, model)
            if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError(
                    f"CP-SAT returned {solver.StatusName(result)} without a plan "
                    f"after {solver.WallTime():.2f}s ({stage})"
                )
            return result

        if cost_expr is not None:
            # Only reached with auto_coarse_tiling off: the allocator withholds the
            # expression when tiling is a solver axis, because the cost model is
            # flat in tile size and cut count. Unchanged behaviour otherwise --
            # a successful cost solve returns here and the ladder is skipped.
            status = self._minimize_cost_expr(model, solver, tensors, cost_expr)

        if status is None:
            # Fallback discipline: the traffic objective below knows no relayout
            # price, and an unpriced shuffle looks free - the exact degeneracy
            # the cost term exists to remove. No relayout decision may be made
            # under this objective, so every copy is pinned out.
            for copy_w in copies.values():
                model.add(copy_w.in_buffer == 0)

            # -- 1. LX residency ------------------------------------------------
            hbm_terms = [
                sb.spill_cost() * (1 - sb.in_buffer) for sb in tensors.values()
            ]
            hbm_terms += self._cut_traffic_terms(model, tensors, children_of)
            status = cp_model.INFEASIBLE
            if hbm_terms:
                model.minimize(sum(hbm_terms))
                status = _solve_stage("residency")
                # Lock in the residency optimum (the traffic value, not just the
                # count) so no later stage can trade a spill for its own metric.
                # Rounding avoids loss of precision as the objective is a
                # sum/product of ints.
                if cut_terms or any(sb.cores is not None for sb in tensors.values()):
                    model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))

            # -- 2. cut count ---------------------------------------------------
            if cut_terms:
                model.minimize(sum(cut_terms))
                status = _solve_stage("cut tiebreak")
                cuts = round(solver.ObjectiveValue())
                logger.debug(
                    "[CP-SAT layout solver] cut tiebreak: %d cut(s) at the "
                    "residency optimum",
                    cuts,
                )

            # -- 3. parallelism, then 4. division shape -------------------------
            # Placement-only buffers have no division to choose and so contribute
            # no term; with none at all there is nothing to rank, so we skip the
            # re-solve and the extract below reads the assignment the last solve
            # still holds.
            core_terms = [sb.cores for sb in tensors.values() if sb.cores is not None]
            # A core_cost term exists for exactly the same buffers as a core term
            # (both are set only on division-carrying buffers), so stage 4 runs
            # whenever stage 3 does.
            core_cost_terms = [
                sb.core_cost for sb in tensors.values() if sb.core_cost is not None
            ]
            if core_terms:
                if cut_terms:
                    model.add(sum(cut_terms) <= cuts)
                model.maximize(sum(core_terms))
                status = _solve_stage("parallelism")
                occupancy = round(solver.ObjectiveValue())

                # Shape balance: holding the parallelism optimum (the objective is
                # integer, so the round is exact), break the remaining ties toward a
                # balanced division by minimizing the summed squared split factors.
                # The parallelism solution still satisfies this lock, so this only
                # refines the choice among equally parallel divisions and can never
                # spill a buffer or lower its core count.
                model.add(sum(core_terms) >= occupancy)
                model.minimize(sum(core_cost_terms))
                status = _solve_stage("division shape")

                # -- 5. tile count ----------------------------------------------
                # Nothing above ranks how finely an op is tiled, so tilings that
                # differ only in count tie on every stage and the multi-worker
                # portfolio picks one arbitrarily (the same graph drew 4, 8 and
                # 64 run to run). Holding the division-shape optimum (integer,
                # so the round is exact), take the fewest tiles. Tilings exist
                # only on division-carrying buffers, so this runs only here.
                if tile_terms:
                    model.add(sum(core_cost_terms) <= round(solver.ObjectiveValue()))
                    model.minimize(sum(tile_terms))
                    status = _solve_stage("tile count")

        final_tensors = self._extract(solver, tensors)

        if logger.isEnabledFor(logging.DEBUG):
            if status is None:
                status = cp_model.INFEASIBLE
            spilled = [n for n, t in final_tensors.items() if t.address is None]
            # The final solve minimized the tile count when there were tilings
            # to choose, else the balance cost when there were divisions to
            # choose (with occupancy held at ``occupancy``); otherwise only the
            # residency solve ran and the objective is HBM traffic.
            if core_terms and tile_terms:
                final_objective = "tile_count"
            elif core_terms:
                final_objective = "balance"
            else:
                final_objective = "hbm_traffic"
            logger.debug(
                "[CP-SAT layout solver] tensors=%d resident=%d %s=%d "
                "occupancy=%s status=%s walltime=%.2f ms",
                len(tensors),
                len(tensors) - len(spilled),
                final_objective,
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

    @staticmethod
    def _relayout_copies(
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> dict[tuple[str, int], _CoreDivisionBufferWithCpVars]:
        """group key -> wrapper of the group's ``RelayoutCopyBuffer`` (whose
        ``serves`` tally the residency gate fills with the served literals of
        the consumer edges it can carry)."""
        copies: dict[tuple[str, int], _CoreDivisionBufferWithCpVars] = {}
        for w in bufs.values():
            if isinstance(w.buffer, RelayoutCopyBuffer):
                assert isinstance(w, _CoreDivisionBufferWithCpVars)
                copies[w.buffer.group_key] = w
        return copies

    @staticmethod
    def _constrain_relayout_copies(
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        copies: dict[tuple[str, int], _CoreDivisionBufferWithCpVars],
    ) -> None:
        """The coupling a ``RelayoutCopyBuffer`` cannot carry as data: a resident
        copy needs its source resident under one of the divisions it was priced
        for (its ``cost_term`` is a table over exactly those), and must serve at
        least one consumer (a copy nobody reads is a shuffle for nothing; the
        price already discourages it, this makes it infeasible). The consumer
        side -- reading the copy pins the division pair and requires the copy
        resident -- lives in ``constrain_residency``. A copy whose source is not
        in this solve, or has no division to choose, can never fire."""
        for copy_w in copies.values():
            source = bufs.get(copy_w.buffer.relayout_parent)
            if source is None or not isinstance(source, _CoreDivisionBufferWithCpVars):
                model.add(copy_w.in_buffer == 0)
                continue
            model.add_implication(copy_w.in_buffer, source.in_buffer)
            priced = [
                source.division_is(i)
                for i in sorted(copy_w.buffer.cost_by_source_division)
            ]
            model.add_bool_or(priced).only_enforce_if(copy_w.in_buffer)
            if copy_w.serves:
                model.add_bool_or(copy_w.serves).only_enforce_if(copy_w.in_buffer)
            else:
                model.add(copy_w.in_buffer == 0)

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
        reconstruct placement units.

        Under a tiling a merge also needs the handoff to exist in the tiled
        graph: the child reads the parent itself, not its copied-out full
        buffer, and nothing keeps the parent alive past the child's write."""
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
                if dst in src_v.via_full:
                    model.add_implication(edge, src_v.via_full[dst].negated())
                if src_v.end is not None:
                    model.add(
                        src_v.end == _Slots.write_slot(dst_v.start_time) + 1
                    ).only_enforce_if(edge)
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
        shared offset.

        The time axis is in :class:`_Slots`, which leave room after every op
        for the copy a cut inserts there; the mapping is monotone, so it
        changes no overlap between untiled lifetimes. A buffer with an ``end``
        stops at that variable instead of its untiled end."""
        x_intervals = []
        y_intervals = []
        for sb in bufs.values():
            ins = incoming.get(sb.name, [])
            assert sb.start is None or not ins, "an input clone has no parent"
            start_time = _Slots.write_slot(sb.start_time)
            end_time: _Bound = (
                sb.end if sb.end is not None else _Slots.end_slot(sb.end_time)
            )
            if ins:
                # at most one incoming merge is active (AddAtMostOne), so the
                # sum is 0 or 1: shorten the child by the handoff tick exactly
                # when it takes over a parent's slot.
                start_var = model.new_int_var(
                    start_time, start_time + 1, f"start_{sb.name}"
                )
                model.add(start_var == start_time + sum(ins))
                x_start: _Bound = start_var
            else:
                x_start = sb.start if sb.start is not None else start_time
            if sb.end is not None or sb.start is not None:
                # Each interval bound has to be affine in one variable, so a
                # length between two variables gets a variable of its own.
                x_size: object = model.new_int_var(
                    0, _Slots.horizon(bufs), f"length_{sb.name}"
                )
                model.add(x_size == end_time - x_start)
            else:
                x_size = end_time - x_start
            sb.slot_span = (x_start, end_time)
            x_intervals.append(
                model.new_optional_interval_var(
                    x_start, x_size, end_time, sb.in_buffer, f"x_{sb.name}"
                )
            )
            # An interval's ``end`` must be affine (a single var), so the top
            # of a division-dependent footprint needs its own var, tied to
            # ``offset + eff_size`` unconditionally. Its range covers every
            # offset/footprint pair, so a spilled buffer loses no assignment.

            # the top bound must be larger than capacity to account for buffers
            # which are larger than LX itself
            y_end = model.new_int_var(
                0,
                max(0, self._capacity_units - 1) + sb.buffer.size,
                f"top_{sb.name}",
            )
            model.add(y_end == sb.offset + sb.eff_size)
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
        copies: dict[tuple[str, int], _CoreDivisionBufferWithCpVars],
    ) -> None:
        """Pin out every buffer ``forced`` non-resident (decided declaratively by
        :meth:`MemoryPlanSolver.partition`) and install the per-buffer residency
        gate. In the joint model that gate is the slicing match, driven entirely
        by the precomputed ``cd_parent_matches`` pairs; placement-only buffers
        have no gate.

        A buffer with an ``untiled_bar`` resides only tiled; one the model
        cannot tile (no ``nest_end``) stays out."""
        for name in forced:
            model.add(bufs[name].in_buffer == 0)
        for sb in bufs.values():
            if sb.buffer.untiled_bar is None:
                continue
            if sb.nest_end is None:
                model.add(sb.in_buffer == 0)
            else:
                assert sb.tiled is not None, "a tileable op has a tiled literal"
                model.add_implication(sb.in_buffer, sb.tiled)
        for sb in bufs.values():
            sb.constrain_residency(model, children_of.get(sb.name, []), bufs, copies)

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
        # Relayout copies are resident buffers like any other here, so the
        # justify pass slides them with everything else and can never move a
        # buffer into a copy's space.
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

            def lifetime(name: str) -> tuple[int, int]:
                # The slide keeps clear of the lifetime the model placed, which
                # a tiling can change (``_add_no_overlap_2d``).
                span = bufs[name].slot_span
                assert span is not None, "_add_no_overlap_2d sets every span"
                start, end = (
                    bound if isinstance(bound, int) else solver.Value(bound)
                    for bound in span
                )
                return start, end

            lifetimes = {n: lifetime(n) for n in resident}
            units = [
                _PlacementUnit(
                    members=names,
                    footprint=max(footprint[n] for n in names),
                    start_time=min(lifetimes[n][0] for n in names),
                    end_time=max(lifetimes[n][1] for n in names),
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

        # Read back the relayouts: a consumer whose served literal is set reads
        # its parent from that copy, under the division pair the literal
        # pinned. Recorded on the consumer with the copy's FINAL address (after
        # the justify slide) for the commit path.
        for name, sb in bufs.items():
            for source_name, reads in sb.relayout_reads.items():
                fired = [copy_w for lit, copy_w in reads if solver.BooleanValue(lit)]
                if not fired:
                    continue
                assert len(fired) == 1, (
                    f"{name} reads {source_name} through {len(fired)} copies at once"
                )
                (copy_w,) = fired
                assert copy_w.name not in spilled, (
                    f"{name} reads {source_name} from a spilled copy {copy_w.name}"
                )
                i = solver.Value(bufs[source_name].division)
                j = solver.Value(sb.division)
                (candidate,) = [
                    c
                    for c in copy_w.buffer.candidates_for(name)
                    if c.source_division == i and c.consumer_division == j
                ]
                sb.buffer.chosen_relayouts[source_name] = ChosenRelayout(
                    candidate, offsets[copy_w.name]
                )
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
