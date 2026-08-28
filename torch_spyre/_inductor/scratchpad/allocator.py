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

import functools
import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any, Callable, cast, Optional

import sympy
import torch
from torch._inductor.ir import (
    TensorBox,
    ComputedBuffer,
    ExternKernel,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    ReinterpretView,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.pass_utils import (
    commit_iteration_space_ownership,
    concretize_expr,
    indirect_info_from_op,
    iteration_space_from_op,
    op_read_writes,
    _prepare_per_core_view,
    _per_core_view_from_prep,
    _is_matmul_op,
    op_short_name,
)
from torch_spyre._inductor.work_division import (
    enumerate_work_division_candidates,
    work_division_splits_are_legal,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
    SolveError,
    BufferType,
)
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingLayoutSolver,
)
from torch_spyre._inductor.scratchpad.exhaustive_search import (
    ExhaustiveSearchSolver,
)
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver
from torch_spyre._inductor.scratchpad.passes import (
    ScratchpadOptimizationPass,
)
from torch_spyre._inductor.scratchpad.utils import (
    round_up_to_alignment,
    clone_at_graph_boundaries,
    mem_usage_by_buf,
    calculate_liveness,
    get_buffer_users,
    ops_in_offset_mutation_component,
    get_op_pointwise_inputs,
    buffer_not_read_in_full,
    get_ncores_for_buffers,
    _is_tiled_advancing,
    _is_read_advancing_anywhere,
    _get_buffer_user_deps,
    _would_produce_lx_back_gap,
    OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE,
    counted_loop_lifetime_end_overrides,
)
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.ir import FixedTiledLayout, SpyreEmptyFallback

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.scratchpad.lx_relayout import (
    LXRelayoutPlan,
    collect_lx_relayout_plans,
    materialize_lx_relayouts,
)

logger = get_inductor_logger("scratchpad.allocator")


# Keep these values synchronized with Deeptools' LX memory tracker:
#
# * ``SenSystemDef`` removes 64 KiB of the physical 2 MiB LX for program and
#   debug data (``dsc/sysdef.cpp``).
# * ``MemTrackBundle::initializeMemoryTrackers`` uses one 128-byte stick as the
#   LX allocation granularity (``sharedtools/mem_track_bundle.cpp``).
#
# Torch and DXP independently consume ``DXP_LX_FRAC_AVAIL``.  These constants
# define the fixed part of that cross-compiler ownership contract.
_LX_PHYSICAL_CAPACITY_BYTES = 2 << 20
_LX_PROGRAM_DEBUG_RESERVATION_BYTES = 64 << 10
_LX_TRACKER_CAPACITY_BYTES = (
    _LX_PHYSICAL_CAPACITY_BYTES - _LX_PROGRAM_DEBUG_RESERVATION_BYTES
)
_LX_ALLOCATION_GRANULARITY_BYTES = 128


def _safe_in_place_parents(
    parents: Sequence[str], lifetime_end_overrides: dict[str, int]
) -> list[str]:
    """Drop handoffs from parents that remain live through a counted loop."""

    return [parent for parent in parents if parent not in lifetime_end_overrides]


def _extern_kernel_in_live_range(graph: GraphLowering, uses: list[int]) -> bool:
    """True if an opaque extern kernel runs at any point while the buffer is live.

    The LX scratchpad is a fixed per-core resource shared by *every* compiled
    Spyre program, and it is not threaded through the generated wrapper as a
    tensor -- a resident buffer is handed from one kernel launch to the next by
    its LX offset alone. An extern kernel is opaque: its body can launch other
    compiled programs (a nested ``torch.compile``, or any eager op, which
    torch-spyre compiles standalone via ``compile_once``), and those programs
    allocate the same LX offsets. A buffer left resident across such a call is
    therefore silently overwritten, and its consumer reads the other program's
    data.

    Being *accessed by* the extern kernel is the narrow case (already fatal,
    since the value must be a real HBM tensor to be passed to it); merely being
    live *across* one is equally fatal and is not visible from ``uses``
    membership alone.
    """
    if not uses:
        return False
    return any(
        isinstance(graph.operations[i], ExternKernel)
        and not isinstance(graph.operations[i], SpyreEmptyFallback)
        for i in range(min(uses), max(uses) + 1)
    )


# A ``MemoryPlanSolver`` is single-use (buffers are required at construction),
# so the allocators hold a factory -- how to build a solver for a given buffer
# set -- rather than a live instance, and build a fresh one per solve.
LayoutSolverFactory = Callable[[Sequence[LifetimeBoundBuffer], int], MemoryPlanSolver]
# Same argument type as ``LayoutSolverFactory`` (``Callable`` parameters are
# contravariant, and every ``CoreDivisionBuffer`` sequence is already a
# ``Sequence[LifetimeBoundBuffer]``); only the narrower return type differs.
CoreDivisionSolverFactory = Callable[
    [Sequence[LifetimeBoundBuffer], int], CoreDivisionLayoutSolver
]


class ScratchpadAllocator:
    """
    Class for allocating on scratchpad
    """

    def __init__(
        self,
        layout_planning: LayoutSolverFactory,
        size: int,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Configure the allocator with a solver factory and graph passes.

        Args:
            layout_planning: Factory that builds a solver (already bound to a
                given buffer set) that assigns LX addresses to lifetime-bound
                buffers. A solver is single-use -- buffers are required at its
                construction -- so the allocator builds a fresh one per solve
                (see :meth:`_build_solver`) rather than holding a live instance.
            size: LX size
            pre_optimization_passes: Graph passes applied before layout planning.
                Defaults to no passes.
            post_optimization_passes: Graph passes applied after layout planning.
                Defaults to no passes.
        """
        if pre_optimization_passes is None:
            pre_optimization_passes = []
        if post_optimization_passes is None:
            post_optimization_passes = []

        # Populated during plan_allocation: maps buffer/op name → reason string.
        # Stamped by _record_spill_reasons from the solver's own spill_reasons
        # (the declared residency verdict, or its capacity check)
        # (for the solver decision). Reset at the start of each plan_allocation.
        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning: Optional[LayoutSolverFactory] = layout_planning
        self.size = size

    @staticmethod
    def _planned_lx_buffer_names(
        plans: Sequence[LXRelayoutPlan],
    ) -> frozenset[str]:
        return frozenset(
            name for plan in plans for name in (plan.source_name, plan.destination_name)
        )

    def _build_solver(self, buffers: Sequence[Any]) -> MemoryPlanSolver:
        """Build a fresh solver over ``buffers`` from :attr:`layout_planning`."""
        assert self.layout_planning is not None
        return self.layout_planning(buffers, self.size)

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, assign LX addresses to eligible buffers, then run post-passes.

        This is a template method: the skeleton (pre-passes ->
        generate buffers -> solve -> commit -> record reasons -> push -> log ->
        post-passes) is fixed, while subclasses override the ``_prepare_buffers``
        / ``_solve`` / ``_post_solve`` / ``_record_spill_reasons`` hooks to swap
        in their buffer type, solver call, and post-solve commit. The base hooks
        implement the fixed-division, placement-only flow.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        self._run_passes(self.pre_optimization_passes, graph)
        buffers = self._prepare_buffers(graph)
        solver = self._build_solver(buffers)
        allocation = self._solve(solver)
        accepted_lx_relayouts = self._finalize_lx_relayout_allocation(allocation)
        self._post_solve(graph, allocation)
        reasons = self._get_spill_reasons(solver, allocation)
        self._push_allocation(graph, allocation, accepted_lx_relayouts)
        self._log_lx_pinning(graph, reasons)
        self._run_passes(self.post_optimization_passes, graph)

    @staticmethod
    def _run_passes(
        passes: Sequence[ScratchpadOptimizationPass], graph: GraphLowering
    ) -> None:
        for p in passes:
            p.apply_pass(graph)

    def _prepare_buffers(self, graph: GraphLowering) -> Sequence[Any]:
        """Buffers to hand the solver. Base: fixed-division LifetimeBoundBuffers."""
        assert self.layout_planning is not None
        if not getattr(self.layout_planning, "supports_paired_buffers", False):
            if config.lx_planner_relayout:
                solver_name = getattr(
                    self.layout_planning,
                    "__name__",
                    type(self.layout_planning).__name__,
                )
                logger.warning(
                    "LX relayout is not supported by %s; continuing without relayout",
                    solver_name,
                )
            return self._generate_buffers(graph)
        plans = collect_lx_relayout_plans(graph)
        buffers = self._generate_buffers(graph, lx_relayout_plans=plans)
        self._append_lx_relayout_destinations(graph, buffers)
        return buffers

    def _solve(self, solver: MemoryPlanSolver) -> Sequence[Any]:
        """Assign LX addresses. Base: placement-only ``plan_layout``."""
        return solver.plan_layout(log_lx_usage=True)

    def _finalize_lx_relayout_allocation(
        self,
        allocation: Sequence[LifetimeBoundBuffer],
    ) -> list[LXRelayoutPlan]:
        plans = [plan for buffer in allocation for plan in buffer.lx_relayout_plans]
        if not plans:
            return []
        complete = self._allocated_lx_relayout_sources(allocation)
        rejected = {plan.source_name for plan in plans} - complete
        if rejected:
            by_name = {buffer.name: buffer for buffer in allocation}
            for source_name in sorted(rejected):
                destinations = sorted(
                    plan.destination_name
                    for plan in plans
                    if plan.source_name == source_name
                )
                allocations = {
                    name: (by_name[name].address, by_name[name].size)
                    for name in (source_name, *destinations)
                }
                logger.debug(
                    "rejected LX relayout group source=%s allocations=%s; "
                    "every member must be allocated and destinations must not "
                    "overlap the source",
                    source_name,
                    allocations,
                )
            self._clear_lx_relayout_groups(allocation, rejected)
        return self._accepted_plans(allocation)

    def _post_solve(self, graph: GraphLowering, allocation: Sequence[Any]) -> None:
        """Hook run after the solve, before reasons/push. Base: nothing to commit."""

    def _get_spill_reasons(
        self, solver: MemoryPlanSolver, allocation: Sequence[LifetimeBoundBuffer]
    ) -> dict:
        """Get spill reasons for every buffer that did not land in LX.

        The solver's own :attr:`spill_reasons` is authoritative -- it carries the
        declared verdict (``residency_reason``) or its capacity check. Anything
        spilled without a reason there simply did not fit once the higher-value
        buffers were placed.
        """
        solver_reasons = dict(solver.spill_reasons)
        for b in allocation:
            if b.address is None:
                solver_reasons[b.name] = solver_reasons.get(
                    b.name,
                    f"no room on scratchpad (t={b.start_time}-{b.end_time},"
                    f" size={b.size // 1024} KB)",
                )
        return solver_reasons

    def _get_op_name(self, op: Any) -> str:
        return op_short_name(op)

    def _op_output_good_for_lx_reuse(
        self, op: Any, planned_lx_buffers: frozenset[str] = frozenset()
    ) -> bool:
        if not isinstance(op, ComputedBuffer):
            return False
        if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            return False
        # A CPU-resident ComputedBuffer has a plain FixedLayout
        # with no device_layout and can never be LX-pinned.
        if not isinstance(op.layout, FixedTiledLayout):
            return False
        # A planned source intentionally bypasses the profitability denylist:
        # the relayout planner has already applied its stricter structural gates.
        return config.allow_all_ops_in_lx_planning or (
            self._get_op_name(op) not in OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE
            or op.get_name() in planned_lx_buffers
        )

    @staticmethod
    def _read_count(uses: list[int]) -> int:
        """Reads residency would serve from LX. The first use is never one of
        them: it is either the producer's write (an intermediate) or the clone-in
        read a graph input cannot avoid.

        Deliberately not ``LifetimeBoundBuffer.read_count``, which counts the
        buffer's reads and so includes an input's clone-in; this is the savings,
        which discounts it in both cases (as ``spill_cost`` does)."""
        return max(0, len(uses) - 1)

    @staticmethod
    def _is_index_or_indirectly_accessed(
        graph: GraphLowering,
        name: str,
        uses: list[int],
        op: Optional[Operation],
    ) -> bool:
        """True if ``name`` is an index tensor, or is itself accessed
        indirectly (a gather/scatter value tensor), on either side of its
        lifetime: as ``op``'s own indirect write (a Scatter target), or as a
        read/index operand of any consumer in ``uses``.

        Both the index tensor and the tensor it indexes into must stay off
        the scratchpad and resolve from HBM instead.
        """
        if isinstance(op, ComputedBuffer):
            writes = op_read_writes(op).writes
            if any(isinstance(dep, MemoryDep) and dep.is_indirect() for dep in writes):
                return True
        for u in uses:
            consumer = graph.operations[u]
            if not isinstance(consumer, ComputedBuffer):
                continue
            index_names, _, _ = indirect_info_from_op(consumer)
            if name in index_names:
                return True
            reads = op_read_writes(consumer).reads
            if any(
                dep.name == name and isinstance(dep, MemoryDep) and dep.is_indirect()
                for dep in reads
            ):
                return True
        return False

    def _buffer_residency_reason(
        self,
        graph: GraphLowering,
        name: str,
        uses: list[int],
        op: Optional[Operation],
        *,
        mutated_buffers: set[str],
        graph_output_names: set[str],
        reinterpret_output_names: set[str],
        ncores: dict[str, int],
        ncores_reasons: dict[str, str],
        division_is_fixed: bool,
        buf_user_deps: dict[str, list[tuple[Operation, MemoryDep]]],
        planned_lx_buffers: frozenset[str] = frozenset(),
    ) -> Optional[str]:
        """The first check ``name`` fails, or ``None`` if it clears them all.

        Order matters. The unsized and op-kind guards come first because
        everything below assumes a placeable ``ComputedBuffer``; the back-gap
        probe comes last because it touches ``device_layout`` and is the most
        expensive. The graph-wide facts (``mutated_buffers``,
        ``graph_output_names``, ``ncores`` ...) are computed once per solve by
        :meth:`_residency_reasons` and passed in, so this stays O(1) in the graph.

        Args:
            graph: The lowered graph.
            name: Buffer name (an op name -- graph inputs go through
                :meth:`_input_residency_reason`).
            uses: ``name``'s liveness (the op indices where it is accessed).
            op: ``name``'s producing op, or ``None`` if it has none.
            division_is_fixed: True on the placement path, where each op's core
                division was committed upstream and a mismatch between a buffer's
                users is fatal. False on the joint path, where the solver chooses
                the division and its slicing gate decides instead.
            buf_user_deps: every buffer's ``(op, dep)`` users, from
                :func:`_get_buffer_user_deps`, for the read-side advancing check.
        """
        if op is None or not self._op_output_good_for_lx_reuse(op, planned_lx_buffers):
            return "op not allowed"
        if not hasattr(getattr(op, "layout", None), "device_layout"):
            # No device layout => no computable footprint (e.g. a
            # MultiOutputLayout tuple op). There is nothing to place, and the
            # checks below would raise.
            return "unsized (no device layout)"
        if name in mutated_buffers:
            return "mutation target"
        if _is_tiled_advancing(op) or _is_read_advancing_anywhere(name, buf_user_deps):
            # LX addresses cannot be expressed as affine.apply symbols today (see
            # compute_ops.py's is_tiled_lx check), so a buffer whose address
            # advances per coarse-tile iteration must stay in HBM, where that is
            # supported -- whether the advance is on this buffer's own write
            # (_is_tiled_advancing) or on some other op's read of it
            # (_is_read_advancing_anywhere, e.g. a fixed-write full buffer
            # copied into a nested tile every outer iteration).
            return "tiled (advancing)"
        restickify = self._restickify_barrier(graph, name, uses)
        if restickify is not None:
            return restickify
        if _extern_kernel_in_live_range(graph, uses):
            return "extern kernel user or live across extern kernel"
        if self._is_index_or_indirectly_accessed(graph, name, uses, op):
            # Index tensors and the value tensors they index into are read via
            # data-dependent (indirect) addressing, must stay in hbm.
            return "index tensor or indirectly accessed"
        if name in graph_output_names:
            # A graph output normally can't reside (the value must land back in
            # HBM), but with boundary cloning on it is pinned via an output clone
            # that still writes HBM once; that unavoidable write cancels from the
            # CP-SAT differential spill cost, so allow residency then.
            if not clone_at_graph_boundaries():
                return "graph output (no clone)"
            if name in reinterpret_output_names:
                return "graph output is a ReinterpretView"
        if buffer_not_read_in_full(graph, name):
            return "partial/offset read"
        if division_is_fixed and ncores.get(name, -1) < 0:
            reason = ncores_reasons.get(name, "core div mismatch")
            return f"core div mismatch: {reason}"
        if self._read_count(uses) == 0:
            # Only the producer's write touches it, so residency saves nothing.
            return "no consumer reads it from LX"
        if _would_produce_lx_back_gap(graph, name, uses):
            # backGap fires when device_size[d] > it_dim_size; the backend
            # supports it for HBM but not for LX.
            return "lx back gap"
        return None

    def _input_residency_reason(
        self,
        graph: GraphLowering,
        name: str,
        uses: list[int],
        *,
        ncores: Optional[dict[str, int]] = None,
        ncores_reasons: Optional[dict[str, str]] = None,
        division_is_fixed: bool,
    ) -> Optional[str]:
        """The residency verdict for a *graph input*, which is pinned by cloning
        it into LX rather than by placing it directly.

        An input has no producing op, so the op-kind checks do not apply; what
        does apply is that the clone must be substitutable at every use, and that
        residency has to beat the clone-in transfer it costs. An input read only
        once is already in HBM and would need one transfer to clone, so pinning
        it saves nothing -- which ``_read_count`` (first use excluded) states
        directly.

        ``ncores``/``ncores_reasons`` are consulted only on the placement path
        (``division_is_fixed``); the joint path defers the core division to the
        solver and passes neither.
        """
        if not clone_at_graph_boundaries():
            return "graph input (no clone)"
        if self._read_count(uses) == 0:
            return "no consumer reads it from LX"
        if self._is_index_or_indirectly_accessed(graph, name, uses, None):
            return "index tensor or indirectly accessed"
        if _extern_kernel_in_live_range(graph, uses):
            return "extern kernel user or live across extern kernel"
        if not GraphEditor.all_uses_are_rewritable(graph, uses):
            return "use is not rewritable to the clone"
        if buffer_not_read_in_full(graph, name):
            return "partial/offset read"
        restickify = self._restickify_barrier(graph, name, uses)
        if restickify is not None:
            return restickify
        if division_is_fixed and (ncores or {}).get(name, -1) < 0:
            reason = (ncores_reasons or {}).get(name, "core div mismatch")
            return f"core div mismatch: {reason}"
        if _would_produce_lx_back_gap(graph, name, uses):
            return "lx back gap"
        return None

    def _residency_reasons(
        self,
        graph: GraphLowering,
        names: "list[str] | set[str]",
        *,
        division_is_fixed: bool,
        lifetimes: Optional[dict[str, list[int]]] = None,
        ncores: Optional[dict[str, int]] = None,
        ncores_reasons: Optional[dict[str, str]] = None,
        planned_lx_buffers: frozenset[str] = frozenset(),
    ) -> dict[str, Optional[str]]:
        """:meth:`_buffer_residency_reason` over ``names``, as ``name -> reason``.

        Computes the graph-wide facts the per-buffer check needs once, here, and
        passes them down -- there is no shared context object. ``lifetimes`` and
        ``ncores`` are accepted so a caller that already has them (the placement
        path, the co-opt search) skips recomputing; ``ncores`` is built only on
        the placement path, since the joint path's slicing gate decides core
        division and ``_buffer_residency_reason`` skips that check when
        ``division_is_fixed`` is False.
        """
        if lifetimes is None:
            lifetimes = calculate_liveness(graph)
        op_by_name = {op.name: op for op in graph.operations}
        mutated_buffers = {
            op.layout.target.get_name()
            for op in graph.operations
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE)
        }
        graph_output_names = set(graph.get_output_names())
        reinterpret_output_names = {
            go.get_name()
            for go in graph.graph_outputs
            if isinstance(go, ReinterpretView)
            or isinstance(getattr(go, "data", None), ReinterpretView)
        }
        if division_is_fixed and ncores is None:
            ncores, ncores_reasons = get_ncores_for_buffers(graph)
        ncores = ncores or {}
        ncores_reasons = ncores_reasons or {}
        buf_user_deps = _get_buffer_user_deps(graph)
        return {
            name: self._buffer_residency_reason(
                graph,
                name,
                lifetimes.get(name, []),
                op_by_name.get(name),
                mutated_buffers=mutated_buffers,
                graph_output_names=graph_output_names,
                reinterpret_output_names=reinterpret_output_names,
                ncores=ncores,
                ncores_reasons=ncores_reasons,
                division_is_fixed=division_is_fixed,
                buf_user_deps=buf_user_deps,
                planned_lx_buffers=planned_lx_buffers,
            )
            for name in names
        }

    def _op_inputs_good_for_lx_inplace(self, op: Any) -> list[str]:
        target = getattr(getattr(op, "origin_node", None), "target", None)
        if target is None:
            return []
        reads = [dep.name for dep in op.get_read_writes().reads]
        # ``tags`` is an OpOverload attribute; some origin targets (e.g. builtin
        # functions behind int64 fallbacks) don't have it. Treat a tag-less
        # target as not-pointwise rather than crashing. The joint-division path
        # reaches this for ops the residency checks bar on the greedy path.
        if torch.Tag.pointwise in getattr(target, "tags", ()):
            # If the op is tagged as pointwise by pytorch upstream
            # allow all inputs. Does not work for all ops
            return reads
        if hasattr(op, "data"):
            return get_op_pointwise_inputs(op.data)
        return []

    def _restickify_barrier(
        self, graph: GraphLowering, name: str, uses: Sequence[int]
    ) -> Optional[str]:
        """The ``residency_reason`` for a buffer a restickify *reads*, else ``None``.

        Restickify moves the stick dimension: its per-core read frame and write
        frame are transposes, so a per-core (LX) slice of the OUTPUT can need
        bytes from another core's slice of the INPUT. The hazard is one-sided --
        it only bites when the input is core-sliced in LX -- so only a buffer a
        restickify reads is barred. The restickify's own output (the use whose op
        *is* this buffer's producer) is a normal core-local write and takes the
        ordinary residency path. Mirrors
        ``CoOptimizingAllocator._residency_reason``'s restickify guard so both
        allocators bar the same buffers; only :class:`CpSatLayoutSolver` acts on
        it, the gap heuristics ignore ``residency_reason``.
        """
        if any(
            graph.operations[u].name != name
            and self._get_op_name(graph.operations[u]) == "restickify"
            for u in uses
        ):
            return "read by restickify (cross-frame barrier)"
        return None

    def _build_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: dict[str, list[str]],
        mem_usage: dict,
        reasons: dict[str, Optional[str]],
        *,
        lifetimes: dict[str, list[int]],
        ncores: dict[str, int],
        ncores_reasons: dict[str, str],
        lifetime_end_overrides: Optional[dict[str, int]] = None,
    ) -> list[LifetimeBoundBuffer]:
        """Build one :class:`LifetimeBoundBuffer` per buffer, barred or not.

        Nothing is dropped for eligibility: an ineligible buffer is handed over
        carrying its ``residency_reason`` and the solver declines to place it
        (see :meth:`MemoryPlanSolver.excluded`). Only buffers with no lifetime at
        all are skipped -- an unused graph input has no ``uses[0]``, so there is
        no interval to reason about.

        Graph inputs are pinned by cloning them into LX rather than placed
        directly, so their verdict comes from
        :meth:`_input_residency_reason` and their footprint is computed
        here rather than read off ``mem_usage`` (which covers ops only).
        """
        lifetime_end_overrides = lifetime_end_overrides or {}
        buffers: list[LifetimeBoundBuffer] = []
        for output_name, info in mem_usage.items():
            uses = lifetimes.get(output_name, [])
            if not uses:
                continue
            buffers.append(
                LifetimeBoundBuffer(
                    output_name,
                    # An unsized (-1) or core-div-mismatched (negative) entry is
                    # always barred, so its footprint is never used; clamp it so
                    # a nonsense size can never look placeable.
                    max(0, info["size_per_core"]),
                    uses,
                    first_use_is_read=False,
                    # Copy: the reverse-parent block in the input loop below appends
                    # to a consumer's in_place_parents, which would otherwise mutate
                    # this list inside the shared ``in_place`` dict (matches the copy
                    # in ``_build_cd_bound_buffers``).
                    in_place_parents=_safe_in_place_parents(
                        in_place.get(output_name, []), lifetime_end_overrides
                    ),
                    residency_reason=reasons.get(output_name),
                    lifetime_end_override=lifetime_end_overrides.get(output_name),
                )
            )

        # Consumer buffers already built above (intermediates + graph outputs),
        # indexed for the reverse-parent edge below. A graph-input clone can only
        # ever be an in-place *parent* -- it is pinned to LX and dies at its last
        # read, and its source stays in HBM (not an LX candidate), so it has no
        # parent of its own. The value is letting the consumer that performs that
        # last read reuse the clone's slot for its own output.
        built_by_name = {b.name: b for b in buffers}
        for input_name in graph.graph_input_names:
            uses = lifetimes.get(input_name, [])
            if not uses:
                continue
            reason = self._input_residency_reason(
                graph,
                input_name,
                uses,
                ncores=ncores,
                ncores_reasons=ncores_reasons,
                division_is_fixed=True,
            )
            clone_size = self._input_footprint(graph, input_name, ncores)
            buffers.append(
                LifetimeBoundBuffer(
                    input_name,
                    clone_size,
                    uses,
                    first_use_is_read=True,
                    in_place_parents=[],
                    residency_reason=reason,
                    lifetime_end_override=lifetime_end_overrides.get(input_name),
                )
            )

            # Reverse-parent edge (issue #3212): let the input clone's last consumer
            # reuse the clone's LX slot in place. The op at the clone's last-use tick
            # both reads the clone and writes its own output, so the
            # single-handoff-tick invariant holds for that consumer alone (enforced
            # by ``_inplace_edge_ok``'s ``parent_end == child_start`` check). Only
            # when the clone can actually reside (``reason is None``) and the
            # consumer is a built candidate with matching per-core size, device
            # layout, a pointwise producer, and no core-division mismatch is there
            # anything safe to merge.
            if reason is not None or input_name in lifetime_end_overrides:
                continue
            last_use = uses[-1]
            consumer_op = graph.operations[last_use]
            consumer = built_by_name.get(consumer_op.name)
            if consumer is None or input_name in consumer.in_place_parents:
                continue
            # A multi-output op (e.g. max/aminmax) carries a MultiOutputLayout with
            # no single ``device_layout``, so it cannot alias one input clone in
            # place; skip it (matches the guard in
            # ``_determine_in_place_division_invariant``).
            consumer_layout = graph.get_buffer(consumer_op.name).get_layout()
            input_layout = graph.get_buffer(input_name).layout
            if not hasattr(consumer_layout, "device_layout") or not hasattr(
                input_layout, "device_layout"
            ):
                continue
            if self._inplace_edge_ok(
                child_pointwise_inputs=self._op_inputs_good_for_lx_inplace(consumer_op),
                parent_name=input_name,
                child_size_per_core=consumer.size,
                parent_size_per_core=clone_size,
                child_device_layout=consumer_layout.device_layout,
                parent_device_layout=input_layout.device_layout,
                child_start=lifetimes[consumer_op.name][0],
                parent_end=last_use,
                child_core_div_mismatch=mem_usage[consumer_op.name][
                    "core_div_mismatch"
                ],
            ):
                consumer.in_place_parents.append(input_name)

        return buffers

    @staticmethod
    def _inplace_edge_ok(
        *,
        child_pointwise_inputs: list[str],
        parent_name: str,
        child_device_layout: Any,
        parent_device_layout: Any,
        child_start: int,
        parent_end: int,
        child_size_per_core: Optional[int] = None,
        parent_size_per_core: Optional[int] = None,
        child_core_div_mismatch: bool = False,
        division_invariant: bool = False,
    ) -> bool:
        """Whether ``parent_name`` may be reused in place by the child buffer.

        The child (which writes at ``child_start``) reuses the parent's storage,
        so the parent must die exactly as the child is born. The conditions:

        - the parent is a pointwise-eligible read input of the child;
        - matching device layout (so the storage can alias);
        - single handoff tick (``parent_end == child_start``: the same op that reads
          the parent as its last use writes the child), the invariant the solvers'
          in-place relaxation relies on (see ``_check_in_place_relationships``);
        - matching per-core footprint and no core-division mismatch on the child.

        With ``division_invariant`` the last condition (per-core size + core-div) is
        skipped: those depend on a core division the joint solver has not chosen
        yet, so it enforces them itself (``eff_size`` equality + the
        ``cd_parent_matches`` gate). Only the first three (division-invariant)
        preconditions are checked.

        This is the sole definition of a legal in-place edge, shared by
        ``_determine_in_place`` / ``_build_bound_buffers`` (placement path) and
        ``_determine_in_place_division_invariant`` / ``_build_cd_bound_buffers``
        (co-optimizing path), so they cannot drift.
        """
        base_ok = (
            parent_name in child_pointwise_inputs
            and child_device_layout == parent_device_layout
            and parent_end == child_start
        )
        if division_invariant:
            return base_ok
        return (
            base_ok
            and child_size_per_core == parent_size_per_core
            and not child_core_div_mismatch
        )

    @staticmethod
    def _input_footprint(
        graph: GraphLowering, name: str, ncores: dict[str, int]
    ) -> int:
        """Per-core LX footprint of a cloned graph input, or 0 when it has no
        computable one (in which case ``input_residency_reason`` has already
        barred it, so the value is never used)."""
        layout = getattr(graph.get_buffer(name), "layout", None)
        dev_layout = getattr(layout, "device_layout", None)
        num_cores = ncores.get(name, -1)
        if dev_layout is None or num_cores < 1:
            return 0
        return math.prod(dev_layout.device_size[:-1]) * 128 // num_cores

    def _determine_in_place(
        self,
        graph: GraphLowering,
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
        reasons: dict[str, Optional[str]],
    ) -> dict[str, list[str]]:
        """In-place reuse candidates: ``buf -> [inputs whose slot it may take]``.

        Only buffers that may actually reside are considered on either side of
        the pair. A barred buffer has no LX slot to hand over or inherit, so
        pairing with one is meaningless -- and it would let two unsized (-1)
        sentinels match each other on size.
        """
        allow_inplace: dict[str, list[str]] = {}
        in_place_allowed = {
            op.name: self._op_inputs_good_for_lx_inplace(op) for op in graph.operations
        }
        for buf_name, info in mem_usage.items():
            allow_inplace[buf_name] = []
            if not in_place_allowed.get(buf_name):
                continue
            if reasons.get(buf_name) is not None or not lifetimes.get(buf_name):
                continue
            out_start = lifetimes[buf_name][0]
            out_ten_layout = graph.get_buffer(buf_name).get_layout().device_layout
            out_size = info["size_per_core"]
            for input_buf in info["op_inputs"]:
                if input_buf not in mem_usage or not lifetimes[input_buf]:
                    continue
                if reasons.get(input_buf) is not None:
                    continue
                in_ten_layout = graph.get_buffer(input_buf).get_layout().device_layout
                if self._inplace_edge_ok(
                    child_pointwise_inputs=in_place_allowed[buf_name],
                    parent_name=input_buf,
                    child_size_per_core=out_size,
                    parent_size_per_core=mem_usage[input_buf]["size_per_core"],
                    child_device_layout=out_ten_layout,
                    parent_device_layout=in_ten_layout,
                    child_start=out_start,
                    parent_end=lifetimes[input_buf][-1],  # inclusive last use
                    child_core_div_mismatch=info["core_div_mismatch"],
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _generate_buffers(
        self,
        graph: GraphLowering,
        cache: Optional[dict] = None,
        timings: Optional[dict[str, float]] = None,
        lifetimes: Optional[dict[str, list[int]]] = None,
        lx_relayout_plans: Sequence[LXRelayoutPlan] = (),
    ) -> list[LifetimeBoundBuffer]:
        # Compute the graph-wide residency facts + mem_usage once and share; the
        # helpers below treat them read-only. `lifetimes` is split-invariant, so
        # the co-opt search passes it in (computed here only for the single-shot
        # path). `ncores` is the placement path's split-dependent core-div check;
        # get_read_writes() is memoized per op by `op_read_writes`, so it doesn't
        # re-trace across leaves.
        t0 = time.perf_counter()
        if lifetimes is None:
            lifetimes = calculate_liveness(graph)
        lifetime_end_overrides = counted_loop_lifetime_end_overrides(graph)
        ncores, ncores_reasons = get_ncores_for_buffers(graph)
        t1 = time.perf_counter()
        mem_usage = mem_usage_by_buf(graph, cache)
        for plan in lx_relayout_plans:
            for name in (plan.source_name, plan.destination_name):
                if name not in mem_usage:
                    continue
                ncores[name] = plan.num_cores
                ncores_reasons.pop(name, None)
                mem_usage[name]["size_per_core"] = (
                    mem_usage[name]["size"] // plan.num_cores
                )
                mem_usage[name]["core_div_mismatch"] = False
        t2 = time.perf_counter()
        if timings is not None:
            timings["residency"] += t1 - t0
            timings["mem_usage"] += t2 - t1

        # Divisions are already committed on this path, so a core-division
        # mismatch between a buffer's users is fatal and is checked here.
        planned_lx_buffers = self._planned_lx_buffer_names(lx_relayout_plans)
        reasons = self._residency_reasons(
            graph,
            list(mem_usage),
            division_is_fixed=True,
            lifetimes=lifetimes,
            ncores=ncores,
            ncores_reasons=ncores_reasons,
            planned_lx_buffers=planned_lx_buffers,
        )
        in_place = self._determine_in_place(graph, mem_usage, lifetimes, reasons)
        buffers = self._build_bound_buffers(
            graph,
            in_place,
            mem_usage,
            reasons,
            lifetimes=lifetimes,
            ncores=ncores,
            ncores_reasons=ncores_reasons,
            lifetime_end_overrides=lifetime_end_overrides,
        )
        if lx_relayout_plans:
            by_name = {buffer.name: buffer for buffer in buffers}
            for plan in lx_relayout_plans:
                by_name[plan.source_name].lx_relayout_plans.append(plan)
        return buffers

    def _append_lx_relayout_destinations(
        self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]
    ) -> None:
        op_index = {op.get_name(): i for i, op in enumerate(graph.operations)}
        entries = []
        invalid = set()
        for source in buffers:
            for plan in source.lx_relayout_plans:
                consumer_ticks = [op_index[name] for name in plan.consumer_names]
                assert all(tick in source.uses for tick in consumer_ticks)
                if source.residency_reason is not None:
                    invalid.add(plan.source_name)
                else:
                    entries.append((source, plan, consumer_ticks))
        if invalid:
            entries = [entry for entry in entries if entry[0].name not in invalid]
            self._clear_lx_relayout_groups(buffers, invalid)
        planned_sources = {source.name for source, _, _ in entries}
        for buffer in buffers:
            buffer.in_place_parents = [
                parent
                for parent in buffer.in_place_parents
                if parent not in planned_sources
            ]
        if not entries:
            return
        for buffer in buffers:
            buffer.uses = [2 * use + 1 for use in buffer.uses]
            if buffer.lifetime_end_override is not None:
                buffer.lifetime_end_override *= 2

        # Adjacent half-ticks rely on DSCs within a bundle executing serially;
        # otherwise the allocator's lifetime reuse is unsound beyond relayout too.
        entries_by_source: dict[
            str, list[tuple[LifetimeBoundBuffer, LXRelayoutPlan, list[int]]]
        ] = defaultdict(list)
        for entry in entries:
            entries_by_source[entry[0].name].append(entry)

        for source_entries in entries_by_source.values():
            source = source_entries[0][0]
            original_end = source.lifetime_end_override
            transfer_ticks = []
            for _, plan, original_ticks in source_entries:
                consumer_ticks = [2 * tick + 1 for tick in original_ticks]
                transfer_tick = consumer_ticks[0] - 1
                transfer_ticks.append(transfer_tick)
                source.uses = sorted(
                    {use for use in source.uses if use not in consumer_ticks}
                    | {transfer_tick}
                )
                nominal_destination_end = consumer_ticks[-1] + 1
                destination_end = (
                    original_end
                    if original_end is not None
                    and original_end > nominal_destination_end
                    else None
                )
                destination = LifetimeBoundBuffer(
                    plan.destination_name,
                    round_up_to_alignment(
                        source.size, _LX_ALLOCATION_GRANULARITY_BYTES
                    ),
                    [transfer_tick, *consumer_ticks],
                    lifetime_end_override=destination_end,
                )
                buffers.insert(buffers.index(source), destination)
                source.paired_with.append(destination)

            # Once relayout owns every later read, the extended lifetime moves
            # from the source to its destinations. A same-view reader after the
            # last transfer still needs the original source through the loop.
            remaining_reads = source.uses[0 if source.first_use_is_read else 1 :]
            if not any(use > max(transfer_ticks) for use in remaining_reads):
                source.lifetime_end_override = None

    def _allocated_lx_relayout_sources(
        self, allocation: Sequence[LifetimeBoundBuffer]
    ) -> set[str]:
        by_name = {buffer.name: buffer for buffer in allocation}
        complete = set()
        for source in allocation:
            if not source.lx_relayout_plans:
                continue
            source_name = source.name
            plans = source.lx_relayout_plans
            destinations = [by_name[plan.destination_name] for plan in plans]
            allocated = [
                buffer.address is not None for buffer in (source, *destinations)
            ]
            assert all(allocated) or not any(allocated), (
                f"paired-buffer group for {source_name} was only partially allocated"
            )
            if not allocated[0]:
                continue
            assert source.address is not None
            assert all(
                destination.address is not None
                and not (
                    source.address < destination.address + destination.size
                    and destination.address < source.address + source.size
                )
                for destination in destinations
            ), f"paired-buffer group for {source_name} has overlapping placements"
            complete.add(source_name)
        return complete

    def _clear_lx_relayout_groups(
        self,
        allocation: Sequence[LifetimeBoundBuffer],
        sources: set[str],
    ) -> None:
        by_name = {buffer.name: buffer for buffer in allocation}
        names = set(sources)
        for source_name in sources:
            source = by_name[source_name]
            names.update(plan.destination_name for plan in source.lx_relayout_plans)
            source.lx_relayout_plans = []
        for buffer in allocation:
            if buffer.name in names:
                buffer.address = None

    def _accepted_plans(
        self, allocation: Sequence[LifetimeBoundBuffer]
    ) -> list[LXRelayoutPlan]:
        by_name = {buffer.name: buffer for buffer in allocation}
        return [
            replace(
                plan,
                source_address=by_name[plan.source_name].address,
                destination_address=by_name[plan.destination_name].address,
            )
            for buffer in allocation
            for plan in buffer.lx_relayout_plans
        ]

    def _log_lx_pinning(self, graph: GraphLowering, reasons: dict) -> None:
        """Log the final LX pinning decision for every op in the graph."""
        # Skip the per-op getattr walk unless DEBUG is on.
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for op in graph.operations:
            reason = reasons.get(op.name, "lx")
            logger.debug(
                "lx_pinning: %s (%s) → %s",
                op.name,
                self._get_op_name(op),
                reason,
            )

    def _push_allocation(
        self,
        graph: GraphLowering,
        buffers: Sequence[LifetimeBoundBuffer],
        accepted_lx_relayouts: list[LXRelayoutPlan],
    ):
        """Push the allocation into the code generation. This includes cloning graph inputs and
        graph outputs:

        - A graph input B that is allocated into LX means that it is cloned; call the clone C. The
        downstream users of B are now made to use C. The LX allocation is effectuated by assigning
        it to C.

        - A graph output B that is allocated into LX means that it is cloned; call the clone C.
        Nothing changes for the downstream users. The LX allocation is effectuated by assigning it
        to B itself. The graph is made to have C as its output.

        - A buffer that is neither a graph input nor a graph output gets the LX allocation assigned
        to itself."""
        outputs = set(graph.get_output_names())
        inputs = set(graph.graph_input_names)

        buffer_users = get_buffer_users(graph)
        graph_editor = GraphEditor(graph)

        for b in buffers:
            if b.address is None or b.name.startswith("__spyre_lx_relayout__:"):
                continue

            buf = graph.get_buffer(b.name)
            if b.name in inputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, buffer_users[b.name], input=True
                )
                self._set_one_allocation(new_buffer, b.address)

            elif b.name in outputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, buffer_users[b.name], input=False
                )
                self._set_one_allocation(buf, b.address)
                graph_editor.change_graph_output(buf, new_buffer)

            else:
                self._set_one_allocation(buf, b.address)

        # Keep graph mutation last and in pre-scheduling: solver retries require
        # the original graph, and post-grad no-op elimination has already run.
        materialize_lx_relayouts(graph, accepted_lx_relayouts)

    def _set_one_allocation(self, buf: TensorBox | ComputedBuffer, address: int):
        layout = buf.get_layout()
        layout.allocation["lx"] = address


def _lx_planning_size() -> int:
    """Return the frontend LX reservation, matching Deeptools exactly.

    The shared Torch/DXP contract partitions Deeptools' allocatable LX capacity,
    not the physical 2 MiB.  The frontend reserves
    ``1 - DXP_LX_FRAC_AVAIL`` from address zero, truncates the fractional byte
    count to an integer, and rounds that reservation up to the memory tracker's
    128-byte allocation granularity.  DXP marks that interval unavailable and
    allocates at or above the returned exclusive upper bound.  This is the
    ownership boundary whose mismatch was reported in torch-spyre issue #3222,
    not a safety margin.
    """
    backend_fraction = config.dxp_lx_frac_avail
    if not 0.0 <= backend_fraction <= 1.0:
        raise ValueError("DXP_LX_FRAC_AVAIL must be >=0 and <=1")

    frontend_reservation = int(_LX_TRACKER_CAPACITY_BYTES * (1.0 - backend_fraction))
    return round_up_to_alignment(frontend_reservation, _LX_ALLOCATION_GRANULARITY_BYTES)


def _core_division(op: Operation, splits: dict[sympy.Symbol, int]) -> CoreDivision:
    """Classify one symbol-keyed candidate for its producing operation."""
    rw = op_read_writes(op)
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        return CoreDivision()
    output = {
        s: int(v) for s, v in splits.items() if write.index.coeff(s) != 0 and v > 1
    }
    reduction = {
        s: int(v) for s, v in splits.items() if write.index.coeff(s) == 0 and v > 1
    }
    return CoreDivision(output_splits=output, reduction_splits=reduction)


def _division_splits(op: Operation, division: CoreDivision) -> dict[sympy.Symbol, int]:
    """Restore a complete symbol-keyed split map from a sparse division."""
    return {
        sym: int(division.output_splits.get(sym, division.reduction_splits.get(sym, 1)))
        for sym in iteration_space_from_op(op)
    }


def _fixed_core_division(op: Operation) -> CoreDivision:
    """The op's committed symbol-keyed division, or a one-core division."""
    ownership = getattr(op, "iteration_space_ownership", None)
    return _core_division(op, ownership.work_slices if ownership is not None else {})


def _legal_fixed_division(
    op: Operation, fixed: list[CoreDivision], reason: str
) -> list[CoreDivision]:
    """Return upstream division when it satisfies hard constraints."""
    division = fixed[0]
    if not isinstance(op, ComputedBuffer) or _split_option_is_legal(
        op, _division_splits(op, division)
    ):
        logger.debug("keep upstream division for %s: %s", op.name, reason)
        return fixed
    raise Unsupported(f"{op.name}: fixed split violates hard domain.")


def _split_option_is_legal(op: Operation, splits: dict[sympy.Symbol, int]) -> bool:
    """Return whether symbol-keyed splits satisfy hard domains."""
    return not isinstance(op, ComputedBuffer) or work_division_splits_are_legal(
        op, splits
    )


def _legal_split_options(
    op: Operation, options: Iterable[dict[sympy.Symbol, int]]
) -> list[dict[sympy.Symbol, int]]:
    """Return stick-valid candidates satisfying hard work-division domains."""
    return [
        option
        for option in options
        if _split_fits_sticks(op, option) and _split_option_is_legal(op, option)
    ]


DEFAULT_VARIANT_CAP = 6
# Try larger batch factors first. Keeping more of the batch axis whole offers
# the same reconciliation benefit with fewer co-optimization candidates.
_FACTORED_B_FACTORS: tuple[int, ...] = (8, 4, 2)


def _seed_splits(op: Operation) -> dict[sympy.Symbol, int]:
    ownership = getattr(op, "iteration_space_ownership", None)
    return {
        sym: int(ownership.work_slices.get(sym, 1)) if ownership is not None else 1
        for sym in iteration_space_from_op(op)
    }


def _output_stride_to_device_size(op: Operation) -> dict[int, int]:
    """Map each output host stride to the device size of the device dim it lands on.

    A stickified host dim decomposes into an outer-stick dim (size = stick count)
    at stride ``stick_host_stride * elems_per_stick`` and a within-stick dim at
    stride ``stick_host_stride``; sticks are atomic, so a split on that host dim
    uses the outer-stick dim. Keying by stride lets a caller look up the true
    splittable size for an output dim by its coefficient in the write index.
    (Mirrors _per_core_view_on_buf's stride→device-dim placement.)
    """
    layout = op.layout
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        # In-place mutations keep the mutation wrapper at pre-scheduler time;
        # the committed device layout lives on the mutation target.
        layout = layout.real_layout()
    dev_layout = layout.device_layout
    device_size = dev_layout.device_size
    stride_map = dev_layout.stride_map
    elems_per_stick = dev_layout.device_dtype.elems_per_stick()
    stride_to_size: dict[int, int] = {}
    for i, s in enumerate(stride_map):
        if s <= 0:  # sentinel for collapsed / broadcast dims
            continue
        if s not in stride_to_size or device_size[i] != 1:
            stride_to_size[s] = device_size[i]
    if stride_map[-1] > 0:  # stickified dim -> bound by the outer-stick count
        stride_to_size[stride_map[-1]] = stride_to_size.get(
            stride_map[-1] * elems_per_stick, 1
        )
    return stride_to_size


def _split_fits_sticks(op: Operation, splits: dict[sympy.Symbol, int]) -> bool:
    """True if every output split divides its physical device dimension.

    A split factor must divide the device dimension it lands on. For the
    stickified host dimension that is the outer-stick count, not the element
    extent: an extent of 128 with 64 elements per stick has only two splittable
    sticks. Reduction-only symbols are absent from the write index and are not
    constrained here; work-division bounds those separately.

    A positive-coefficient output symbol with no device-stride entry is
    unplaceable (for example, a collapsed or broadcast dimension), so reject it
    rather than relying on modulo arithmetic with a missing size.
    """
    layout = op.layout
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    # CPU ComputedBuffers participate in the joint division map but never in LX
    # placement. They have no device geometry to validate against.
    if not isinstance(layout, FixedTiledLayout):
        return True
    write = next(iter(op_read_writes(op).writes), None)
    if write is None:
        return False
    sizes = _output_stride_to_device_size(op)
    for sym, factor in splits.items():
        stride = int(write.index.coeff(sym))
        size = sizes.get(stride, 0)
        if factor > 1 and stride and (not size or size % factor):
            return False
    return True


def _matmul_axis_parse(op: Operation) -> dict[str, tuple[sympy.Symbol, int, int]]:
    """Parse a matmul into ``{B|M|N|K: (symbol, extent, seed_factor)}``.

    Output symbols sorted by ascending write-index stride are N, M, B (with B
    absent for 2D matmuls); the symbol added by a read index is K. Output
    extents come from stick-aware device geometry, so generated split factors
    divide stick counts rather than element extents. The returned symbols and
    factors remain local and symbol-keyed.
    """
    rw = op_read_writes(op)
    write = next(iter(rw.writes)).index
    read = next((dep.index for dep in rw.reads), write)
    out_syms = {int(write.coeff(sym)): sym for sym in write.free_symbols}
    k_syms = read.free_symbols - write.free_symbols
    if not k_syms:
        raise ValueError(f"matmul {op.get_name()} has no reduction axis")
    sizes = _output_stride_to_device_size(op)
    seed = _seed_splits(op)
    roles: dict[str, tuple[sympy.Symbol, int, int]] = {}
    for role, stride in zip(("N", "M", "B"), sorted(out_syms)):
        sym = out_syms[stride]
        roles[role] = (sym, sizes[stride], seed[sym])
    k_sym = next(iter(k_syms))
    roles["K"] = (
        k_sym,
        concretize_expr(iteration_space_from_op(op)[k_sym]),
        seed[k_sym],
    )
    return roles


def _bm_axes_from_roles(
    roles: dict[str, tuple[sympy.Symbol, int, int]],
) -> tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]] | None:
    """Return B/M ``(symbol, extent)`` pairs, or ``None`` when either is absent."""
    b, m = roles.get("B"), roles.get("M")
    return ((b[0], b[1]), (m[0], m[1])) if b is not None and m is not None else None


def _reduction_bm_axes(
    op: Operation,
) -> tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]] | None:
    """Return stick-aware B/M output axes for a non-matmul reduction.

    A reduction over N keeps B and M in its write. As in
    :func:`_matmul_axis_parse`, the largest output stride is B and the next is
    M. Reductions with fewer than two output axes cannot use this factorization.
    """
    write = next(iter(op_read_writes(op).writes)).index
    out_syms = {int(write.coeff(sym)): sym for sym in write.free_symbols}
    if len(out_syms) < 2:
        return None
    m_stride, b_stride = sorted(out_syms)[-2:]
    sizes = _output_stride_to_device_size(op)
    return (out_syms[b_stride], sizes[b_stride]), (out_syms[m_stride], sizes[m_stride])


def _factored_bm_splits(
    bm_axes: tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]] | None,
) -> list[dict[sympy.Symbol, int]]:
    """Offer at most one largest-B full-core B/M factorization.

    Smaller B factors do not add a useful reconciliation option but multiply the
    co-optimization search space. An empty result means B/M is absent or no
    factorization divides both stick-aware extents.
    """
    if bm_axes is None:
        return []
    (b_sym, b_size), (m_sym, m_size) = bm_axes
    for b_factor in _FACTORED_B_FACTORS:
        m_factor = config.sencores // b_factor
        if (
            b_factor * m_factor == config.sencores
            and b_size % b_factor == 0
            and m_size % m_factor == 0
        ):
            return [{b_sym: b_factor, m_sym: m_factor}]
    return []


def _candidate_key(splits: dict[sympy.Symbol, int]) -> tuple[tuple[str, int], ...]:
    """Return a stable, symbol-orderable deduplication key for one candidate."""
    return tuple(sorted(((str(sym), factor) for sym, factor in splits.items())))


def _output_profile(op: Operation, splits: dict[sympy.Symbol, int]) -> dict[int, int]:
    """Project output splits onto physical strides for cross-operation transfer.

    This is temporary candidate-generation metadata only; callers immediately
    reconstruct a symbol-keyed candidate for the target operation.
    """
    write = next(iter(op_read_writes(op).writes)).index
    return {
        int(write.coeff(sym)): factor
        for sym, factor in splits.items()
        if factor > 1 and write.coeff(sym) != 0
    }


def _from_output_profile(
    op: Operation, profile: dict[int, int]
) -> dict[sympy.Symbol, int]:
    """Apply a transient physical output profile as a symbol-keyed candidate."""
    write = next(iter(op_read_writes(op).writes)).index
    return {
        sym: profile.get(int(write.coeff(sym)), 1) if write.coeff(sym) != 0 else 1
        for sym in iteration_space_from_op(op)
    }


def _find_distinct_matmul_splits(
    ops: list[Operation],
) -> tuple[tuple[dict[int, int], ...], tuple[dict[str, int], ...]]:
    """Collect distinct physical profiles and B/M/N/K factors from matmul seeds.

    Profiles let intervening pointwise operations offer a matching output
    division. Role factors let another matmul transfer a split despite using
    different local iteration symbols. Both are transient inputs to candidate
    generation; committed divisions remain symbol-keyed.
    """
    profiles: list[dict[int, int]] = []
    roles: list[dict[str, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for op in ops:
        if not _is_matmul_op(op):
            continue
        parsed = _matmul_axis_parse(op)
        candidates = [_output_profile(op, _seed_splits(op))]
        candidates += [
            _output_profile(op, candidate)
            for candidate in _factored_bm_splits(_bm_axes_from_roles(parsed))
        ]
        for profile in candidates:
            key = tuple(sorted(profile.items()))
            if profile and key not in seen:
                seen.add(key)
                profiles.append(profile)
        if candidates[0]:
            roles.append(
                {role: factor for role, (_sym, _size, factor) in parsed.items()}
            )
    return tuple(profiles), tuple(roles)


def _check_and_add_matmul_options(
    op: Operation,
    seed: dict[sympy.Symbol, int],
    matmul_roles: tuple[dict[str, int], ...],
) -> list[dict[sympy.Symbol, int]]:
    """Offer seed, cross-matmul, and factored B/M candidates for ``op``.

    Work distribution may choose incompatible splits for two matmuls joined by
    pointwise/reduction operations. Transferring each source's B/M/N/K factors
    gives the co-optimizer a chance to choose a compatible assignment. Missing
    roles or non-divisible extents default to one; view matching later rejects
    candidates that cannot share a physical buffer view.
    """
    parsed = _matmul_axis_parse(op)
    options = {_candidate_key(seed): seed}
    for source in matmul_roles:
        candidate = {sym: 1 for sym in iteration_space_from_op(op)}
        for role, (sym, extent, _factor) in parsed.items():
            factor = source.get(role, 1)
            candidate[sym] = factor if factor > 1 and extent % factor == 0 else 1
        options.setdefault(_candidate_key(candidate), candidate)
    for candidate in _factored_bm_splits(_bm_axes_from_roles(parsed)):
        full_candidate = {sym: 1 for sym in iteration_space_from_op(op)}
        full_candidate.update(candidate)
        options.setdefault(_candidate_key(full_candidate), full_candidate)
    return _legal_split_options(op, options.values())


def _enum_split_options(
    op: Operation,
    extra_profiles: tuple[dict[int, int], ...] = (),
    matmul_roles: tuple[dict[str, int], ...] = (),
) -> list[dict[sympy.Symbol, int]]:
    """Enumerate symbol-keyed candidates for the pruned co-optimization path.

    Matmuls use role transfer plus B/M factorizations. Other reductions offer
    B/M factorizations only because their reduction axis is fixed. Pointwise
    ops can move a single output split to another divisible output axis and can
    adopt a matmul's transient physical output profile. The seed is always kept;
    non-seed candidates must fit physical stick geometry.
    """
    seed = _seed_splits(op)
    is_computed = isinstance(op, ComputedBuffer)
    is_reduction = is_computed and isinstance(op.data, Reduction)
    is_matmul = is_reduction and _is_matmul_op(op)
    seed_profile = _output_profile(op, seed)
    if is_matmul and matmul_roles:
        return _check_and_add_matmul_options(op, seed, matmul_roles)
    if is_reduction:
        if not seed_profile:
            return [seed]
        options = {_candidate_key(seed): seed}
        for candidate in _factored_bm_splits(_reduction_bm_axes(op)):
            full_candidate = {sym: 1 for sym in iteration_space_from_op(op)}
            full_candidate.update(candidate)
            options.setdefault(_candidate_key(full_candidate), full_candidate)
        return _legal_split_options(op, options.values())
    if not is_computed or not seed_profile:
        return [seed]

    write = next(iter(op_read_writes(op).writes)).index
    sliced = [
        sym for sym, factor in seed.items() if factor > 1 and write.coeff(sym) != 0
    ]
    options = {_candidate_key(seed): seed}
    if len(sliced) == 1:
        source = sliced[0]
        factor = seed[source]
        for sym, extent in iteration_space_from_op(op).items():
            if (
                sym is not source
                and write.coeff(sym) != 0
                and (size := concretize_expr(extent)) > 1
                and size % factor == 0
            ):
                candidate = dict(seed)
                candidate[source], candidate[sym] = 1, factor
                options.setdefault(_candidate_key(candidate), candidate)
                if len(options) >= DEFAULT_VARIANT_CAP:
                    break
    for profile in extra_profiles:
        candidate = _from_output_profile(op, profile)
        options.setdefault(_candidate_key(candidate), candidate)
    return _legal_split_options(op, options.values())


class CoOptimizingAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: CoreDivisionSolverFactory,
        size: int,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        prune: bool = False,
    ):
        """Joint core-division + LX-placement allocator.

        Args:
            layout_planning: Factory for a core-division-aware solver — either
                the OR-Tools ``CpSatLayoutSolver`` (ILP) or an
                ``ExhaustiveSearchSolver`` (DFS) wrapping a placement-only
                factory. This allocator drives the *joint* entry point, so it
                needs the ``CoreDivisionLayoutSolver`` interface rather than a
                plain ``MemoryPlanSolver``. The ortools-missing fallback to
                greedy placement lives in :func:`select_allocator`, which
                never constructs this allocator without a valid factory.
            pre_optimization_passes: Graph passes applied before layout planning.
            post_optimization_passes: Graph passes applied after layout planning.
            prune: Enable heuristic pruning of the core-division search space.
        """
        super().__init__(
            layout_planning=layout_planning,
            size=size,
            pre_optimization_passes=pre_optimization_passes,
            post_optimization_passes=post_optimization_passes,
        )
        # Narrow the base's ``LayoutSolverFactory`` annotation: the joint entry
        # point requires the core-division interface.
        self.layout_planning: Optional[CoreDivisionSolverFactory] = layout_planning
        self.prune = prune

    def _prepare_buffers(self, graph: GraphLowering) -> Sequence[Any]:
        in_place = self._determine_in_place_division_invariant(graph)
        buffers = self._build_cd_bound_buffers(
            graph, in_place, self._division_map(graph)
        )
        return buffers

    def _solve(self, solver: MemoryPlanSolver) -> Sequence[Any]:
        assert isinstance(solver, CoreDivisionLayoutSolver)
        result = solver.plan_layout_and_core_divisions()
        assert not any(buffer.lx_relayout_plans for buffer in result), (
            "CoOptimizingAllocator does not support LX relayout"
        )
        return result

    def _post_solve(self, graph: GraphLowering, allocation: Sequence[Any]) -> None:
        # The divisions must be committed such that any buffer clones can correctly
        # pull the selected core division from the dependent buffers when the graph
        # is updated with clones in ``_push_allocation``.
        self._commit_divisions(graph, allocation)

    def _get_spill_reasons(
        self,
        solver: MemoryPlanSolver,
        allocation: Sequence[LifetimeBoundBuffer],
    ) -> dict:
        # Surface the solver's per-buffer spill causes so the LX-pinning debug
        # log reports why each buffer landed in HBM, on par with the other
        # allocators. Both CoreDivisionLayoutSolver implementations expose it.
        assert isinstance(solver, CoreDivisionLayoutSolver)
        return solver.spill_reasons

    def _division_map(self, graph: GraphLowering) -> dict[str, list[CoreDivision]]:
        """Per-op core-division candidates for the joint-division solve.

        Every op gets at least one ``CoreDivision`` so the slicing-match gate can
        constrain it. Pointwise / Reduction ops get the enumerated candidates;
        every other op falls back to its committed symbol-keyed division. No
        op-kind pre-filter -- residency is
        gated per buffer (``_residency_by_buf``) and by the solver, so ineligible
        ops still participate as producers/consumers in the match.

        Exception: ops data-connected to a sliced in-place mutation (a constant-
        offset write, e.g. ``x[:, 32:96] = ...``) are pinned to their upstream
        (fixed) division. Re-slicing any op fused into the offset write's SDSC
        makes the deeptools scheduler reject it (``DtException: "There must be at
        least one valid candidate"``), the root cause of the
        ``slice_stick_mutation_*`` failures. Keeping the fixed division there
        matches the schedulable slicing the greedy path uses; it costs only a
        division optimization when that division also satisfies hard
        work-division constraints. Otherwise LX planning raises ``Unsupported``
        rather than committing an illegal division. See
        ``utils.ops_in_offset_mutation_component``.
        """
        max_cores = config.sencores
        fixed_division_ops = ops_in_offset_mutation_component(graph)
        profiles, matmul_roles = _find_distinct_matmul_splits(graph.operations)

        result = {}
        for op in graph.operations:
            if op.name in fixed_division_ops:
                divs = _legal_fixed_division(
                    op, [_fixed_core_division(op)], "offset mutation component"
                )
            elif self.prune and isinstance(op, ComputedBuffer):
                divs = [
                    _core_division(op, splits)
                    for splits in _legal_split_options(
                        op, _enum_split_options(op, profiles, matmul_roles)
                    )
                ]
            else:
                divs = self._enumerate_core_divisions(op, max_cores)
            if not divs:
                raise Unsupported(f"{op.name}: no legal core-division candidates.")
            result[op.name] = divs

        return result

    def _enumerate_core_divisions(
        self, op: Operation, max_cores: int
    ) -> list[CoreDivision]:
        """Enumerate and deduplicate symbol-keyed candidates for one operation.

        Operations without an enumerable concrete iteration space retain their
        committed division. Deduplication uses local symbol names only within
        this operation; cross-operation compatibility is derived from
        ``PerCoreView`` instead.
        """
        fixed = [_fixed_core_division(op)]
        if not isinstance(op, ComputedBuffer) or not isinstance(
            op.data, (Pointwise, Reduction)
        ):
            return fixed
        try:
            candidates = enumerate_work_division_candidates(op, max_cores)
        except Unsupported as exc:
            return _legal_fixed_division(op, fixed, str(exc))
        cds: list[CoreDivision] = []
        seen: set[tuple] = set()
        for candidate in candidates:
            division = _core_division(op, candidate)
            key = (
                tuple(
                    sorted(
                        division.output_splits.items(), key=lambda item: str(item[0])
                    )
                ),
                tuple(
                    sorted(
                        division.reduction_splits.items(), key=lambda item: str(item[0])
                    )
                ),
            )
            if key not in seen:
                seen.add(key)
                cds.append(division)
        return cds or _legal_fixed_division(op, fixed, "no enumerable candidate")

    def _commit_divisions(
        self,
        graph: GraphLowering,
        allocation: Sequence[CoreDivisionBuffer],
    ) -> None:
        """Commit the solver's chosen symbol-keyed division for every buffer.

        The solver optimizes a core division for all buffers, not just resident
        ones: a resident producer and its consumers are pinned by
        ``_CoreDivisionBufferWithCpVars.constrain_residency`` to one shared
        slicing (so those commits are mutually consistent), while a spilled
        buffer is free of that gate -- its accesses round-trip through HBM,
        which re-slices on load -- so it takes its most parallel candidate.
        Committing the spilled buffers' divisions too lets the joint solve
        optimize work division across the whole graph, not only the LX-resident
        region.
        """
        op_by_name = {op.name: op for op in graph.operations}
        for buf in allocation:
            op = op_by_name.get(buf.name)
            if op is None or buf.chosen_division is None:
                continue
            if not hasattr(op, "iteration_space_ownership"):
                continue
            cd = buf.core_divisions[buf.chosen_division]
            if not _split_option_is_legal(op, _division_splits(op, cd)):
                raise Unsupported(f"{op.name}: chosen split violates hard domain.")
            commit_iteration_space_ownership(op, _division_splits(op, cd))

    def _determine_in_place_division_invariant(
        self, graph: GraphLowering
    ) -> dict[str, list[str]]:
        """Co-opt in-place candidates: keep only the *division-invariant*
        preconditions here and defer the division-dependent ones to the solver.

        The per-core size match and core-division compatibility depend on the
        division the ILP has not yet chosen, so they are enforced in the solver
        (``eff_size`` equality + the ``cd_parent_matches`` gate). What stays as a
        pre-filter is division-invariant: lifetime adjacency
        (``in_end == out_start``, the single-tick-handoff invariant the solver's
        no-overlap relaxation relies on but cannot re-derive) and identical device
        layouts (required for the storage to alias).
        """
        allow_inplace: dict[str, list[str]] = {}
        mem_usage = mem_usage_by_buf(graph)
        in_place_allowed = {
            op.name: self._op_inputs_good_for_lx_inplace(op) for op in graph.operations
        }
        lifetimes = calculate_liveness(graph)
        lifetime_end_overrides = counted_loop_lifetime_end_overrides(graph)
        for buf_name, info in mem_usage.items():
            allow_inplace[buf_name] = []
            if not in_place_allowed[buf_name]:
                continue
            # Unplaceable producers (e.g. a ``MultiOutputLayout`` tuple op like
            # max-with-indices) carry no ``device_layout``: their storage cannot
            # alias an input, so skip rather than raise ``AttributeError``.
            out_layout = graph.get_buffer(buf_name).layout
            if not hasattr(out_layout, "device_layout"):
                continue
            out_start = lifetimes[buf_name][0]
            out_ten_layout = out_layout.device_layout
            for input_buf in info["op_inputs"]:
                # Graph inputs / constants now appear in ``op_inputs`` but are not
                # solver buffers, so they can't be in-place aliasing parents (the
                # solver's ``_check_in_place_relationships`` would fail to resolve
                # them). Skip them, matching the base allocator's guard.
                if input_buf not in mem_usage or not lifetimes[input_buf]:
                    continue
                if input_buf in lifetime_end_overrides:
                    continue
                in_layout = graph.get_buffer(input_buf).layout
                if not hasattr(in_layout, "device_layout"):
                    continue
                in_ten_layout = in_layout.device_layout
                # The division-invariant edge gate (layout match + single handoff
                # tick; per-core size and core-division deferred to the solver). The
                # ``division_invariant`` mode also applies the per-input
                # pointwise-eligibility test (``input_buf in in_place_allowed``): for
                # pointwise-tagged ops that is every read (a no-op), but for a
                # non-tagged Pointwise op it drops an input read at a different index
                # than the output write -- which must not be aliased over the output.
                if self._inplace_edge_ok(
                    child_pointwise_inputs=in_place_allowed[buf_name],
                    parent_name=input_buf,
                    child_device_layout=out_ten_layout,
                    parent_device_layout=in_ten_layout,
                    child_start=out_start,
                    parent_end=lifetimes[input_buf][-1],  # inclusive last use
                    division_invariant=True,
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _residency_by_buf(
        self,
        graph: GraphLowering,
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
    ) -> dict[str, Optional[str]]:
        """Per-buffer residency verdict: ``None`` if the buffer may be pinned in
        LX, else the reason it may not.

        Every buffer is handed to the solver so it participates in the slicing
        match, but participation is not residency. The predicate is the shared
        one in :meth:`_buffer_residency_reason` -- the same list the placement
        path uses -- with ``division_is_fixed=False``: this allocator *chooses*
        each op's core division, so pre-rejecting a buffer whose users disagree
        under the upstream-committed division would be premature. The solver's
        ``cd_parent_matches`` slicing gate decides that instead. ``ncores`` is
        therefore not needed here.
        """
        return self._residency_reasons(
            graph, list(mem_usage), division_is_fixed=False, lifetimes=lifetimes
        )

    def _build_cd_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
        divisions: dict[str, list[CoreDivision]],
    ) -> list[CoreDivisionBuffer]:
        """Build the ``CoreDivisionBuffer``s handed to the solver.

        Every buffer carries its candidate ``divisions`` and is sized by its
        *total* device footprint plus its producer edges (``parent_proj``); the
        solver picks a division and divides by its ``output_partition``.
        Counted-loop lifetimes still filter unsafe in-place handoffs before the
        solver compares those total footprints.
        """
        lifetimes = calculate_liveness(graph)
        lifetime_end_overrides = counted_loop_lifetime_end_overrides(graph)
        mem_usage = mem_usage_by_buf(graph)
        in_place = {} if in_place is None else in_place
        op_by_name = {op.name: op for op in graph.operations}
        graph_output_names = set(graph.get_output_names())

        prep_cache: dict = {}
        buffers: list[CoreDivisionBuffer] = []
        residency_by_buf = self._residency_by_buf(graph, mem_usage, lifetimes)

        input_clone_matches: dict[str, dict[str, list[tuple[int, int]]]] = {}
        # Consumer op name -> input clones for which it is the last reader, and so
        # may reuse the clone's LX slot in place (reverse-parent, #3212). Stays
        # empty unless cloning is on, making the reverse-parent block in the output
        # loop a no-op otherwise.
        last_consumer_clones: dict[str, list[str]] = {}
        if clone_at_graph_boundaries():
            buffer_users = get_buffer_users(graph)
            for input_name in self._eligible_clone_inputs(graph, lifetimes):
                consumers = [op for op in buffer_users.get(input_name, [])]
                divs, matches = self._clone_divisions_and_matches(
                    input_name, consumers, divisions, prep_cache
                )
                # No division matched any consumer -> the clone has no valid core
                # division and could never reside, so don't hand an unplaceable
                # buffer to the solver (it would trip the >=1-division invariant).
                # The input simply stays in HBM, as it would uncloned.
                if not divs:
                    continue
                input_clone_matches[input_name] = matches
                residency_by_buf[input_name] = None
                # Only the op at the clone's last-use tick both reads the clone and
                # writes its own output, so only it satisfies the single handoff
                # tick for reusing the clone's slot in place.
                last_use = lifetimes[input_name][-1]
                last_consumer_clones.setdefault(
                    graph.operations[last_use].name, []
                ).append(input_name)
                dev_layout = graph.get_buffer(input_name).layout.device_layout
                size = math.prod(dev_layout.device_size[:-1]) * 128
                buffers.append(
                    CoreDivisionBuffer(
                        input_name,
                        size,
                        lifetimes[input_name],
                        first_use_is_read=True,
                        in_place_parents=[],
                        core_divisions=divs,
                        parents=[],
                        cd_parent_matches={},
                        residency_reason=None,
                        lifetime_end_override=lifetime_end_overrides.get(input_name),
                        boundary=BufferType.Input,
                    )
                )

        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]

            op = op_by_name.get(output_name)
            residency_reason = residency_by_buf[output_name]

            buf_divisions = divisions[output_name]
            parents = _safe_in_place_parents(
                in_place.get(output_name, []), lifetime_end_overrides
            )
            size = info["size"]  # total footprint; solver divides per chosen cd
            parent_proj = info["op_inputs"].copy()
            cd_parent_matches = self._cd_parent_matches(
                op,
                buf_divisions,
                parent_proj,
                divisions,
                op_by_name,
                prep_cache,
                residency_by_buf,
            )

            for input_name in parent_proj:
                if input_name in input_clone_matches:
                    cd_parent_matches[input_name] = input_clone_matches[input_name][
                        output_name
                    ]

            # Reverse-parent edge (#3212): when this op is an input clone's last
            # reader, let it reuse the clone's LX slot in place. Division-invariant
            # gate (pointwise child reading the clone + matching device layout;
            # single tick guaranteed by "last reader"); per-core size and core
            # division are deferred to the solver, which also gates the merge on the
            # cd_parent_matches entry set just above. Multi-output ops carry a
            # MultiOutputLayout with no single device_layout and cannot alias one
            # clone, so they are skipped.
            out_layout = graph.get_buffer(output_name).layout
            for clone_name in last_consumer_clones.get(output_name, []):
                if clone_name in lifetime_end_overrides:
                    continue
                if clone_name in parents:
                    continue
                clone_layout = graph.get_buffer(clone_name).layout
                if (
                    op is None
                    or not hasattr(out_layout, "device_layout")
                    or not hasattr(clone_layout, "device_layout")
                ):
                    continue
                if self._inplace_edge_ok(
                    child_pointwise_inputs=self._op_inputs_good_for_lx_inplace(op),
                    parent_name=clone_name,
                    child_device_layout=out_layout.device_layout,
                    parent_device_layout=clone_layout.device_layout,
                    child_start=uses[0],
                    parent_end=lifetimes[clone_name][-1],
                    division_invariant=True,
                ):
                    parents.append(clone_name)

            buffers.append(
                CoreDivisionBuffer(
                    output_name,
                    size,
                    uses,
                    # An op output is a computed buffer: ``uses[0]`` is the
                    # producing write, as on the placement path above. (Only the
                    # input-clone loop above sets this True.)
                    first_use_is_read=False,
                    in_place_parents=parents,
                    core_divisions=buf_divisions,
                    parents=parent_proj,
                    cd_parent_matches=cd_parent_matches,
                    residency_reason=residency_reason,
                    lifetime_end_override=lifetime_end_overrides.get(output_name),
                    boundary=BufferType.Output
                    if output_name in graph_output_names
                    else BufferType.Intermediate,
                )
            )
        return buffers

    def _is_frame_changing_clone(self, op: Operation, buf_name: str) -> bool:
        """True if ``op`` is a clone whose output ``buf_name`` has an iteration
        dimension that none of its inputs carry -- i.e. it broadcasts a dim
        (e.g. GQA broadcasting K/V over the query-group axis). Such a clone reads
        its input in a different frame than it writes its output, so a per-core
        slice of the output cannot be produced from a core-local slice of the
        input; pinning the output mis-addresses (cf. the restickify barrier)."""
        if self._get_op_name(op) != "clone":
            return False
        rw = op_read_writes(op)
        write = next(
            (w for w in rw.writes if w.name == buf_name and hasattr(w, "index")), None
        )
        if write is None:
            return False
        read_syms: set = set()
        for r in rw.reads:
            if hasattr(r, "index"):
                read_syms |= set(r.index.free_symbols)
        # A write-only free symbol means the clone expands (broadcasts) that dim.
        return bool(set(write.index.free_symbols) - read_syms)

    def _eligible_clone_inputs(
        self, graph: GraphLowering, lifetimes: dict[str, list[int]]
    ) -> list[str]:
        """Graph inputs eligible to be cloned into LX.

        The same shared predicate the placement path's input loop uses, with
        ``division_is_fixed=False``: the core division is deferred to the solver,
        since a clone can take any division for which some valid choice
        satisfies all its children. That constraint is enforced by requiring
        each child to match the parent, not by pre-rejecting here.
        """
        return [
            name
            for name in graph.graph_input_names
            if self._input_residency_reason(
                graph, name, lifetimes.get(name, []), division_is_fixed=False
            )
            is None
        ]

    def _clone_divisions_and_matches(
        self,
        input_name: str,
        consumers: list[Operation],
        divisions: dict[str, list[CoreDivision]],
        prep_cache: dict,
    ) -> tuple[list[CoreDivision], dict[str, list[tuple[int, int]]]]:
        """Determine the core divisions which are applicable to the clone
        node based on the read per core views of the clone's consumers and
        equivalent core count (to cover the broadcasting case)

        The applicable core divisions are found and returned as a list of
        ``CoreDivision`` objects. The mapping such that the clone output
        per core view matches a given op's read per core view is returned
        where the mapping exists for each consumer. When solved the parent
        output per core view must match that of all consumers to be placed.
        This forces correctness at solve time rather than pre-pruning by
        finding the intersection of core divisions.
        """
        clone_divs: list[CoreDivision] = []
        clone_views: list[tuple] = []  # parallel: the view each clone div reproduces
        matches: dict[str, list[tuple[int, int]]] = {}
        for consumer in consumers:
            cname = consumer.get_name()
            consumer_divs = divisions[cname]
            rw = op_read_writes(consumer)
            read_dep = next(
                (r for r in rw.reads if r.name == input_name and hasattr(r, "index")),
                None,
            )
            write = next((w for w in rw.writes if hasattr(w, "index")), None)
            if read_dep is None or write is None:
                matches[cname] = []
                continue
            views = self._views_for_divs(
                consumer, read_dep, input_name, consumer_divs, prep_cache
            )
            pairs: list[tuple[int, int]] = []
            for j, (view, _, repr_ok) in enumerate(views):
                if not repr_ok:
                    continue
                k = next((idx for idx, v in enumerate(clone_views) if v == view), None)
                if k is None:
                    cd = consumer_divs[j]
                    per_sym = _division_splits(consumer, cd)
                    k = len(clone_divs)
                    clone_divs.append(
                        CoreDivision(
                            output_splits={
                                sym: split
                                for sym, split in per_sym.items()
                                if split > 1
                            },
                            reduction_splits={},
                        )
                    )  # a clone op cannot have a reduction split
                    clone_views.append(view)
                if clone_divs[k].cores_used == consumer_divs[j].cores_used:
                    pairs.append((k, j))
            matches[cname] = pairs
        # An empty ``clone_divs`` means no consumer matched the clone under any
        # division, so it has no valid core division. Return it empty rather than
        # fabricating a whole-buffer fallback that no consumer matches: the caller
        # drops such a clone (it can never reside), keeping it out of the solver's
        # >=1-division invariant.
        return clone_divs, matches

    def _cd_parent_matches(
        self,
        consumer_op: Optional[Operation],
        consumer_divs: list[CoreDivision],
        parent_names: list[str],
        divisions: dict[str, list[CoreDivision]],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
        residency_by_buf: dict[str, Optional[str]],
    ) -> dict[str, list[tuple[int, int]]]:
        """Physical slicing-match pairs for each divided producer this op reads.

        For producer ``P`` feeding this consumer, a ``(P_div_idx,
        consumer_div_idx)`` pair is compatible iff the two divisions induce the
        *same per-core slicing of ``P``* (``P``'s write-view equals the
        consumer's read-view, both via ``_per_core_view_on_buf`` in ``P``'s
        device-dim frame) AND use the *same total core count*. This is the
        per-core-view comparison ``get_ncores_for_buffers`` uses -- correct across
        reductions/reshapes, where a coeff-keyed signature would conflate axes.

        Excluded from matching (producer then falls back to HBM, always correct):
        a producer that can never be resident (``residency_by_buf`` reason is not
        ``None``); a producer candidate whose write carries a partial reduction
        (output not final); and either side's candidate whose slicing of ``P`` is
        unrepresentable -- we never pin on a slicing we cannot verify.
        """
        if consumer_op is None:
            return {}
        matches: dict[str, list[tuple[int, int]]] = {}
        consumer_reads = op_read_writes(consumer_op).reads
        for parent in parent_names:
            if parent not in op_by_name:
                continue
            if residency_by_buf.get(parent, "not in graph") is not None:
                continue
            parent_divs = divisions[parent]
            parent_op = op_by_name[parent]
            # Frame-changing (broadcasting) clone barrier: the output reads its
            # input in a different frame (e.g. GQA broadcasting K/V over the
            # query-group axis), so a per-core slice can't be produced
            # core-locally. The single-frame view comparison misses this; keep
            # it in HBM (the broadcast read is globally correct).
            if self._is_frame_changing_clone(parent_op, parent):
                continue
            write_dep = next(
                (
                    w
                    for w in op_read_writes(parent_op).writes
                    if w.name == parent and hasattr(w, "index")
                ),
                None,
            )
            read_dep = next(
                (r for r in consumer_reads if r.name == parent and hasattr(r, "index")),
                None,
            )
            if write_dep is None or read_dep is None:
                continue

            # Producer view per candidate; ``None`` marks one that can't host a
            # readable residency: a partial-reduction write, an unrepresentable
            # slicing, or a matmul output split across >1 device dim. The SDSC
            # for a matmul carries only the primary split, so a multi-dim-split
            # output (M-split x N-stick-split) can't be coherently LX-pinned even
            # when views match -- a consumer would read per-core LX holding only
            # a fragment. (Mirrors #2745's ``get_ncores_for_buffers`` matmul
            # guard for the greedy path.)
            parent_is_matmul = _is_matmul_op(parent_op)
            prod_views: list[Optional[tuple]] = [
                view
                if (
                    repr_ok
                    and not partial
                    and not (parent_is_matmul and len(view.work_slice_dims) > 1)
                )
                else None
                for view, partial, repr_ok in self._views_for_divs(
                    parent_op, write_dep, parent, parent_divs, prep_cache
                )
            ]
            cons_views: list[Optional[tuple]] = [
                view if repr_ok else None
                for view, _partial, repr_ok in self._views_for_divs(
                    consumer_op, read_dep, parent, consumer_divs, prep_cache
                )
            ]

            # A matched pair needs equal per-core slicing AND equal total core
            # count: equal views alone aren't enough, since a producer on N and
            # consumer on M>N cores can share a slicing while the consumer's
            # extra (broadcast-axis) cores hold no copy and would read stale LX.
            # The joint solver re-divides per buffer and can hit this; a rejected
            # pair just falls back to HBM.
            pairs = [
                (i, j)
                for i, pv in enumerate(prod_views)
                if pv is not None
                for j, cv in enumerate(cons_views)
                if cv is not None
                and pv == cv
                and parent_divs[i].cores_used == consumer_divs[j].cores_used
            ]
            matches[parent] = pairs
        return matches

    @staticmethod
    def _views_for_divs(op, dep, buf_name, divs, prep_cache: dict):
        """Per-core views of ``buf_name`` for each candidate division of ``op``.

        Prepares the candidate-invariant context once (``_prepare_per_core_view``
        -- the sympy-heavy op-level work) and evaluates every candidate from it
        via ``_per_core_view_from_prep``, so cost scales with the op rather than
        its candidate count.

        ``prep_cache`` is keyed by ``(op name, dep, buf_name)``: a producer's
        write-dep and a consumer's read-dep on the same buffer can be equal
        ``MemoryDep``s, so the op name keeps their preps distinct while a parent
        read by several consumers reuses its write-view prep.
        """
        key = (op.get_name(), dep, buf_name)
        out = []
        for cd in divs:
            if key not in prep_cache:
                prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
            splits = _division_splits(op, cd)
            out.append(
                _per_core_view_from_prep(prep_cache[key], splits, cd.reduction_splits)
            )
        return out


def _make_cpsat_solver(
    buffers: Sequence[LifetimeBoundBuffer], size: int
) -> MemoryPlanSolver:
    """Build the CP-SAT layout solver, or ``GreedyLayoutSolver`` when ortools
    is unavailable.

    Imported lazily so this module (and every non-cpsat path) loads without
    ortools installed; ``CpSatLayoutSolver.__init__`` raises ``ImportError``
    when ortools (``cp_model``) is missing, which we translate to a
    placement-only greedy fallback so callers never see an unusable factory.
    """
    try:
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        return CpSatLayoutSolver(buffers, size)
    except ImportError as exc:
        logger.warning(
            "cpsat layout solver unavailable (%s); falling back to the "
            "default greedy allocator.",
            exc,
        )
        return GreedyLayoutSolver(buffers, size)


_PLACEMENT_SOLVERS: dict[str, LayoutSolverFactory] = {
    "greedy": GreedyLayoutSolver,
    "bestfit": BestFitLayoutSolver,
    "firstfit": FirstFitLayoutSolver,
    "simulated_annealing": SimulatedAnnealingLayoutSolver,
    "cpsat": _make_cpsat_solver,
}


def select_allocator() -> ScratchpadAllocator:
    """Build the scratchpad allocator and inject its layout solver from config.

    This is the single place that maps config to an (allocator, solver) pair, so
    the allocators themselves take an explicit solver factory and never inspect
    config:

    * Without ``co_optimizing_lx_planning``, returns a :class:`ScratchpadAllocator`
      instance that solves for LX placement only.
    * With ``co_optimizing_lx_planning``, returns a :class:`CoOptimizingAllocator`
      instance. ``"simulated_annealing"`` is served by
      :class:`SaCoOptimizingSolver`, the joint work-division + LX engine.
      Otherwise a core-division-capable factory (currently only ``"cpsat"``, and
      only when ortools is available) is used directly; every other factory is
      wrapped in an :class:`ExhaustiveSearchSolver` that does an exhaustive
      search of all the core division options.

    The annealer is deliberately not wrapped in :class:`ExhaustiveSearchSolver`:
    that wrapper solves the layout once per enumerated division candidate, so
    nesting a full anneal there would cost one anneal per candidate. It is also a
    separate class from :class:`SimulatedAnnealingLayoutSolver` rather than one
    class in two modes (unlike the cpsat pair, where the same solver simply does
    less work): the layout-only annealer stays a usable
    :class:`MemoryPlanSolver`, while the joint engine composes the same packer
    and adds the division moves. Do not merge them.
    """
    size = _lx_planning_size()

    try:
        solver_cls = _PLACEMENT_SOLVERS[config.layout_solver]
    except KeyError:
        raise ValueError(
            f"Invalid layout_solver config option '{config.layout_solver}'."
        )

    if config.co_optimizing_lx_planning:
        if config.lx_planner_relayout:
            logger.warning(
                "LX relayout is not supported by CoOptimizingAllocator; "
                "continuing without relayout"
            )
        if config.layout_solver == "simulated_annealing":
            return CoOptimizingAllocator(
                layout_planning=SaCoOptimizingSolver, size=size
            )
        # Throwaway empty-buffer probe: cheap (no real solving happens in
        # __init__) and the only way to know whether this factory's solver is
        # core-division-capable when the factory may be a plain function (the
        # ortools-availability-aware cpsat factory) rather than a solver class.
        if not isinstance(solver_cls([], size), CoreDivisionLayoutSolver):
            return CoOptimizingAllocator(
                layout_planning=functools.partial(
                    ExhaustiveSearchSolver, inner_factory=solver_cls
                ),
                size=size,
                prune=True,
            )
        # The isinstance check above just proved this factory's solver is a
        # CoreDivisionLayoutSolver at runtime; narrow the static type to match.
        return CoOptimizingAllocator(
            layout_planning=cast(CoreDivisionSolverFactory, solver_cls), size=size
        )

    return ScratchpadAllocator(layout_planning=solver_cls, size=size)


def scratchpad_planning(
    graph: GraphLowering,
    allocator: Optional[ScratchpadAllocator] = None,
) -> None:
    """Assign LX scratchpad addresses to eligible buffers in a lowered graph.

    Called after stickification and core-division are complete. Graph operations
    are expected to be in topological order as guaranteed by GraphLowering.

    Args:
        graph: Lowered graph to plan scratchpad memory for.
        allocator: Allocator strategy to use. Defaults to the config-selected
            allocator (see :func:`select_allocator`).
    """
    if allocator is None:
        allocator = select_allocator()
    try:
        allocator.plan_allocation(graph)
    except SolveError:
        # When a solve error arises we assume a strong excpetion guarentee
        # meaning despite the solver failing. The allocator has not mutated
        # the state of the graph allowing a second attempt with a
        # greedy approach.
        logger.debug("solve error detected. falling back to greedy solver.")
        ScratchpadAllocator(
            GreedyLayoutSolver, size=_lx_planning_size()
        ).plan_allocation(graph)
