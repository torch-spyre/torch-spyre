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

import logging
import math
import time
from collections.abc import Sequence
from typing import Any, Optional

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
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    iteration_space_from_op,
    splits_by_index_coeff,
    op_read_writes,
    _prepare_per_core_view,
    _per_core_view_from_prep,
)
from torch_spyre._inductor.work_division import enumerate_work_division_candidates
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
    _would_produce_lx_back_gap,
    OP_OUTPUT_GOOD_FOR_LX_REUSE,
)
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import _is_matmul_op

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


class ScratchpadAllocator:
    """
    Class for allocating on scratchpad
    """

    def __init__(
        self,
        layout_planning: MemoryPlanSolver,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Configure the allocator with an optional solver and graph passes.

        Args:
            layout_planning: Solver that assigns LX addresses to lifetime-bound
                buffers. Defaults to GreedyLayoutSolver sized to available LX memory.
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
        self.reject_reasons: dict[str, str] = {}
        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning: Optional[MemoryPlanSolver] = layout_planning

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, assign LX addresses to eligible buffers, then run post-passes.

        This is a template method: the skeleton (reset reasons -> pre-passes ->
        generate buffers -> solve -> commit -> record reasons -> push -> log ->
        post-passes) is fixed, while subclasses override the ``_prepare_buffers``
        / ``_solve`` / ``_post_solve`` / ``_record_reject_reasons`` hooks to swap
        in their buffer type, solver call, and post-solve commit. The base hooks
        implement the fixed-division, placement-only flow.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        self.reject_reasons = {}
        self._run_passes(self.pre_optimization_passes, graph)
        buffers = self._prepare_buffers(graph)
        allocation = self._solve(buffers)
        self._post_solve(graph, allocation)
        self._record_reject_reasons(allocation)
        self._push_allocation(graph, allocation)
        self._log_lx_pinning(graph)
        self._run_passes(self.post_optimization_passes, graph)

    @staticmethod
    def _run_passes(
        passes: Sequence[ScratchpadOptimizationPass], graph: GraphLowering
    ) -> None:
        for p in passes:
            p.apply_pass(graph)

    def _prepare_buffers(self, graph: GraphLowering) -> Sequence[Any]:
        """Buffers to hand the solver. Base: fixed-division LifetimeBoundBuffers."""
        return self._generate_buffers(graph)

    def _solve(self, buffers: Sequence[Any]) -> Sequence[Any]:
        """Assign LX addresses. Base: placement-only ``plan_layout``."""
        assert self.layout_planning is not None
        return self.layout_planning.plan_layout(buffers, log_lx_usage=True)

    def _post_solve(self, graph: GraphLowering, allocation: Sequence[Any]) -> None:
        """Hook run after the solve, before reasons/push. Base: nothing to commit."""

    def _record_reject_reasons(self, allocation: Sequence[Any]) -> None:
        """Stamp ``reject_reasons`` for spilled buffers. Base: from the solver's
        per-buffer :meth:`_record_spill_reasons`."""
        self._record_spill_reasons(allocation)

    def _record_spill_reasons(self, allocation: Sequence[LifetimeBoundBuffer]) -> None:
        """Stamp ``reject_reasons`` for every buffer that did not land in LX.

        The solver's own :attr:`spill_reasons` is authoritative -- it carries the
        declared verdict (``residency_reason``) or its capacity check. Anything
        spilled without a reason there simply did not fit once the higher-value
        buffers were placed.
        """
        assert self.layout_planning is not None
        solver_reasons = self.layout_planning.spill_reasons
        for b in allocation:
            if b.address is None:
                self.reject_reasons[b.name] = solver_reasons.get(
                    b.name,
                    f"no room on scratchpad (t={b.start_time}-{b.end_time},"
                    f" size={b.size // 1024} KB)",
                )

    def _get_op_name(self, op: Any) -> str:
        return _op_short_name(op)

    def _op_output_good_for_lx_reuse(self, op: Any) -> bool:
        if not isinstance(op, ComputedBuffer):
            return False
        if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            return False
        return config.allow_all_ops_in_lx_planning or (
            self._get_op_name(op) in OP_OUTPUT_GOOD_FOR_LX_REUSE
        )

    @staticmethod
    def _read_count(uses: list[int]) -> int:
        """Reads residency would serve from LX. The first use is never one of
        them: it is either the producer's write (an intermediate) or the clone-in
        read a graph input cannot avoid. Mirrors
        ``LifetimeBoundBuffer.read_count``."""
        return max(0, len(uses) - 1)

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
        """
        if op is None or not self._op_output_good_for_lx_reuse(op):
            return "op not allowed"
        if not hasattr(getattr(op, "layout", None), "device_layout"):
            # No device layout => no computable footprint (e.g. a
            # MultiOutputLayout tuple op). There is nothing to place, and the
            # checks below would raise.
            return "unsized (no device layout)"
        if name in mutated_buffers:
            return "mutation target"
        if _is_tiled_advancing(op):
            # LX addresses cannot be expressed as affine.apply symbols today (see
            # compute_ops.py's is_tiled_lx check), so a buffer whose address
            # advances per coarse-tile iteration must stay in HBM, where that is
            # supported.
            return "tiled (advancing), not per_tile_fixed"
        restickify = self._restickify_barrier(graph, name, uses)
        if restickify is not None:
            return restickify
        if any(isinstance(graph.operations[u], ExternKernel) for u in uses):
            return "extern kernel user"
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
                    in_place_parents=in_place.get(output_name, []),
                    residency_reason=reasons.get(output_name),
                )
            )

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
            buffers.append(
                LifetimeBoundBuffer(
                    input_name,
                    self._input_footprint(graph, input_name, ncores),
                    uses,
                    first_use_is_read=True,
                    in_place_parents=[],
                    residency_reason=reason,
                )
            )

        return buffers

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
                in_end = lifetimes[input_buf][-1]  # inclusive last use
                in_ten_layout = graph.get_buffer(input_buf).get_layout().device_layout
                in_size = mem_usage[input_buf]["size_per_core"]
                inp_i_size_match = out_size == in_size
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start  # same op reads input and writes output
                no_core_div_mismatch = not info["core_div_mismatch"]
                if (
                    input_buf in in_place_allowed[buf_name]
                    and inp_i_size_match
                    and inp_i_lay_match
                    and inp_i_eol
                    and no_core_div_mismatch
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _generate_buffers(
        self,
        graph: GraphLowering,
        cache: Optional[dict] = None,
        timings: Optional[dict[str, float]] = None,
        lifetimes: Optional[dict[str, list[int]]] = None,
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
        ncores, ncores_reasons = get_ncores_for_buffers(graph)
        t1 = time.perf_counter()
        mem_usage = mem_usage_by_buf(graph, cache)
        t2 = time.perf_counter()
        if timings is not None:
            timings["residency"] += t1 - t0
            timings["mem_usage"] += t2 - t1

        # Divisions are already committed on this path, so a core-division
        # mismatch between a buffer's users is fatal and is checked here.
        reasons = self._residency_reasons(
            graph,
            list(mem_usage),
            division_is_fixed=True,
            lifetimes=lifetimes,
            ncores=ncores,
            ncores_reasons=ncores_reasons,
        )
        in_place = self._determine_in_place(graph, mem_usage, lifetimes, reasons)
        return self._build_bound_buffers(
            graph,
            in_place,
            mem_usage,
            reasons,
            lifetimes=lifetimes,
            ncores=ncores,
            ncores_reasons=ncores_reasons,
        )

    def _log_lx_pinning(self, graph: GraphLowering) -> None:
        """Log the final LX pinning decision for every op in the graph."""
        # Skip the per-op getattr walk unless DEBUG is on.
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for op in graph.operations:
            reason = self.reject_reasons.get(op.name, "lx")
            logger.debug(
                "lx_pinning: %s (%s) → %s",
                op.name,
                self._get_op_name(op),
                reason,
            )

    def _push_allocation(
        self, graph: GraphLowering, buffers: Sequence[LifetimeBoundBuffer]
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
            if b.address is None:
                continue

            buf = graph.get_buffer(b.name)
            if b.name in inputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, b.address, buffer_users[b.name], input=True
                )
                self._set_one_allocation(new_buffer, b.address)

            elif b.name in outputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, b.address, buffer_users[b.name], input=False
                )
                self._set_one_allocation(buf, b.address)
                graph_editor.change_graph_output(buf, new_buffer)

            else:
                self._set_one_allocation(buf, b.address)

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


def _fixed_core_division(op: Operation) -> CoreDivision:
    """The op's upstream-committed division (``op.op_it_space_splits``) as a single
    pinned :class:`CoreDivision`; a never-divided op yields a one-core empty split.
    """
    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", None) or ({}, {})
    return CoreDivision(output_splits=dict(seed[0]), reduction_splits=dict(seed[1]))


def _op_short_name(op: Any) -> str:
    """Resolve an op's short name from its ``origin_node`` target, falling back
    to each fused fx node in ``op.origins``; ``"None"`` when unresolvable.

    ``origin_node`` is tried first (independent of ``origins``, which may be
    empty), so a plain op still resolves; the ``origins`` fallback recovers a
    fused op like bmm+permute, whose ``origin_node`` target has no resolvable
    name and would otherwise resolve to ``"None"`` and be wrongly rejected as
    "op not allowed". Module-level so ``ScratchpadAllocator._get_op_name``
    delegates to one implementation.
    """
    name = None
    for fx_node in (getattr(op, "origin_node", None), *getattr(op, "origins", ())):
        target = getattr(fx_node, "target", None)
        name = (
            getattr(target, "_opname", None)
            or getattr(target, "__name__", None)
            or getattr(target, "name", None)
        )
        if name is not None:
            break
    return name if name is not None else "None"


DEFAULT_VARIANT_CAP = 6


def _output_stride_to_device_size(op: Operation) -> dict[int, int]:
    """Map each output host stride to the device size of the device dim it lands on.

    A stickified host dim decomposes into an outer-stick dim (size = stick count)
    at stride ``stick_host_stride * elems_per_stick`` and a within-stick dim at
    stride ``stick_host_stride``; sticks are atomic, so a split on that host dim
    uses the outer-stick dim. Keying by stride lets a caller look up the true
    splittable size for an output dim by its coefficient in the write index.
    (Mirrors _per_core_view_on_buf's stride→device-dim placement.)
    """
    dev_layout = op.layout.device_layout
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


def _split_fits_sticks(op: Operation, splits: tuple[dict, dict]) -> bool:
    """True if every output-dim factor in `splits` divides that dim's stick count.

    A split factor must divide the device size of the dim it lands on, which for
    the stickified dim is the stick count, not the element extent. Element-extent
    divisibility is not enough: N=128 with 64 elems/stick is only 2 sticks, yet
    128 % 4 == 0 would admit a 4-way split the SDSC bundler then rejects (SIGABRT).
    Checks output splits only; reduction (K) splits are bounded by the planner.

    A split whose stride has no entry in stride_to_size (e.g. it lands on a
    collapsed/broadcast device dim that _output_stride_to_device_size skips) is
    unplaceable and rejected: size defaults to 0, and `size <= 0` fails the check.
    (Plain `0 % factor == 0` would wrongly *admit* it.)
    """
    out_splits = splits[0]
    if not out_splits:
        return True
    stride_to_size = _output_stride_to_device_size(op)
    for stride, factor in out_splits.items():
        if factor <= 1:
            continue
        size = stride_to_size.get(int(stride), 0)
        if size <= 0 or size % factor != 0:
            return False
    return True


# TODO: helper for cross-matmul split transfer. Remove together with the
# block in _enum_split_options once work_dist assigns consistent splits.
def _matmul_axis_parse(
    op: Operation,
) -> dict[str, tuple[sympy.Symbol, int, int]]:
    """Parse a batched-matmul op into ``{role: (sym, extent, factor)}``.

    Role is one of "B", "M", "N", "K"; `sym` is the op's iter symbol for that
    axis, `extent` its splittable size, `factor` the current split from
    op.op_it_space_splits (1 if unsplit). Output is [B, M, N] (3D) or [M, N]
    (2D), so output symbols sorted by ascending stride spell N, M, B (B absent
    for 2D); the lone reduction symbol is K.

    For output dims, `extent` is the device size of the dim it maps to (the stick
    count for the stickified dim), via _output_stride_to_device_size — so a valid
    split must divide the stick count, not the element extent.
    """
    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)
    iter_space = iteration_space_from_op(op)

    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    per_sym = apply_splits_from_index_coeff(seed, write_index, read_index, iter_space)
    stride_to_size = _output_stride_to_device_size(op)

    # Derive axis symbols from the index free_symbols (not iter_space, which may
    # not enumerate every indexed symbol): output dims are in write_index, and K
    # is whatever the read index adds on top of the write.
    out_stride_sym = {int(write_index.coeff(s)): s for s in write_index.free_symbols}
    k_syms = read_index.free_symbols - write_index.free_symbols
    if not k_syms:
        raise ValueError(
            f"matmul {op.get_name()}: read index adds no reduction symbol over "
            f"the write index (read={read_index}, write={write_index})"
        )
    k_sym = next(iter(k_syms))

    roles: dict[str, tuple[sympy.Symbol, int, int]] = {}
    possible_roles = ["N", "M", "B"]
    for i, st in enumerate(sorted(out_stride_sym)):  # ascending, works for 2D and 3D
        sym = out_stride_sym[st]
        roles[possible_roles[i]] = (sym, stride_to_size[st], per_sym[sym])
    roles["K"] = (k_sym, concretize_expr(iter_space[k_sym]), per_sym[k_sym])

    return roles


# Batch factors to try, largest first. Only the largest one that fits is offered
# (see _factored_bm_splits): a bigger B split keeps more of the batch axis whole,
# and smaller-B variants don't aid reconciliation while multiplying the co-opt
# search space. m_fac = ncores // b_fac.
_FACTORED_B_FACTORS: tuple[int, ...] = (8, 4, 2)


def _bm_axes_from_roles(
    roles: dict[str, tuple[sympy.Symbol, int, int]],
) -> Optional[tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]]]:
    """B/M axes ((b_sym, b_extent), (m_sym, m_extent)) from _matmul_axis_parse
    roles, or None if either is absent."""
    b = roles.get("B")
    m = roles.get("M")
    if b is None or m is None:
        return None
    return (b[0], b[1]), (m[0], m[1])


def _reduction_bm_axes(
    op: Operation,
) -> Optional[tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]]]:
    """B/M axes for a reduction op, from its output dims.

    A reduction over N keeps B and M as output dims (e.g. write `512*d0 + d1`:
    B=d0, M=d1). Mirror _matmul_axis_parse's stride convention: sort output syms
    by ascending stride; the largest-stride dim is B (outermost), the next is M.
    Extents are stick-aware via _output_stride_to_device_size. None if < 2 output
    dims (nothing to factor).
    """
    write_index = next(iter(op.get_read_writes().writes)).index
    out_stride_sym = {int(write_index.coeff(s)): s for s in write_index.free_symbols}
    if len(out_stride_sym) < 2:
        return None
    stride_to_size = _output_stride_to_device_size(op)
    by_stride = sorted(out_stride_sym)  # ascending: [..., M, B]
    m_stride, b_stride = by_stride[-2], by_stride[-1]
    return (
        (out_stride_sym[b_stride], stride_to_size[b_stride]),
        (out_stride_sym[m_stride], stride_to_size[m_stride]),
    )


# TODO: companion to _matmul_axis_parse. Remove with the block in
# _enum_split_options once work_dist assigns consistent splits.
def _factored_bm_splits(
    op: Operation,
    bm_axes: Optional[tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]]],
) -> list[tuple[dict, dict]]:
    """Batch-major (B/b · M/m) full-core output split for `op`.

    `bm_axes` is ((b_sym, b_extent), (m_sym, m_extent)) for the op's batch and M
    output dims (from _bm_axes_from_roles for matmuls, _reduction_bm_axes for
    reductions). Returns at most ONE candidate: the largest-B full-core factoring
    (b_fac from _FACTORED_B_FACTORS, largest first; m_fac = ncores // b_fac) that
    divides both stick-count extents. Smaller-B factorings are not offered — they
    don't help reconciliation and only inflate the co-opt search space. Empty if
    no factoring fits (e.g. B too small). The caller's _split_fits_sticks is the
    final guard.
    """
    if bm_axes is None:
        return []
    (b_sym, b_extent), (m_sym, m_extent) = bm_axes
    ncores = config.sencores

    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)

    for b_fac in _FACTORED_B_FACTORS:
        m_fac = ncores // b_fac
        if (
            b_fac * m_fac != ncores
            or b_fac > b_extent
            or b_extent % b_fac != 0
            or m_extent % m_fac != 0
        ):
            continue
        per_sym = {b_sym: b_fac, m_sym: m_fac}
        return [splits_by_index_coeff(per_sym, write_index, read_index)]
    return []


# TODO: companion to _matmul_axis_parse. Remove with the block in
# _enum_split_options once work_dist assigns consistent splits.
def _find_distinct_matmul_splits(
    ops: list[Operation],
) -> tuple[tuple[tuple[dict, dict], ...], tuple[dict[str, int], ...]]:
    """Collect the distinct matmul output-splits in `ops`.

    Returns ``(bases, roles)`` deduped by canonical key. `bases` are raw
    (output_splits, {}) tuples for the pointwise path — the matmuls' seed splits
    plus the factored batch-major (B/b · M/m) splits, so the softmax chain between
    two matmuls can adopt a shared B/M tiling. `roles` are the seed splits as
    {role: factor} maps (e.g. {"M": 4, "N": 8}) for the cross-matmul transfer.
    """
    seen: set[tuple] = set()
    bases: list[tuple[dict, dict]] = []
    roles: list[dict[str, int]] = []
    for op in ops:
        if not _is_matmul_op(op):
            continue
        op_roles = _matmul_axis_parse(op)
        out: dict = getattr(op, "op_it_space_splits", ({}, {}))[0]
        candidates: list[tuple[dict, dict]] = [(dict(out), {})] if out != {} else []
        candidates += _factored_bm_splits(op, _bm_axes_from_roles(op_roles))
        for base in candidates:
            key = _canonical_key(base)
            if key in seen:
                continue
            seen.add(key)
            bases.append(base)
        if out != {}:
            roles.append({r: f for r, (_s, _e, f) in op_roles.items()})
    return tuple(bases), tuple(roles)


# TODO: companion to _matmul_axis_parse. Remove with the block in
# _enum_split_options once work_dist assigns consistent splits.
def _check_and_add_matmul_option(
    op: Operation,
    seed: tuple[dict, dict],
    matmul_roles: tuple[dict[str, int], ...],
) -> list[tuple[dict, dict]]:
    """Options for matmul `op`: its seed, each other matmul's split transferred
    into this op's coordinates by axis role, plus factored batch-major (B/b · M/m)
    splits.

    work_dist can assign two matmuls inconsistent splits (e.g. QK {4096:4, 1:8}
    vs AV {128:32}); a shared axis (here M) then disagrees with the PW/softmax
    ops between them, forcing a core-div mismatch and blocking LX pinning.
    Offering each matmul the other's split lets the co-opt search pick a
    consistent assignment. A role absent on this op, or whose extent is not
    divisible by the source factor, does not transfer. The factored B/M splits
    cover the case where the two matmuls' N/K roles map to different physical
    dims and so can't be cross-transferred, but they still share the B and M
    output axes. Candidates that fail to reconcile a shared buffer's PerCoreView
    self-eliminate during scoring.
    """
    self_roles = _matmul_axis_parse(op)
    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)

    options: dict[tuple, tuple[dict, dict]] = {_canonical_key(seed): seed}
    for src in matmul_roles:
        per_sym = {}
        for role, (sym, extent, _factor) in self_roles.items():
            factor = src.get(role, 1)
            per_sym[sym] = factor if factor > 1 and extent % factor == 0 else 1
        if not any(f > 1 for f in per_sym.values()):
            continue
        candidate = splits_by_index_coeff(per_sym, write_index, read_index)
        options.setdefault(_canonical_key(candidate), candidate)
    for candidate in _factored_bm_splits(op, _bm_axes_from_roles(self_roles)):
        options.setdefault(_canonical_key(candidate), candidate)
    # Here the filter is mostly defense-in-depth: _matmul_axis_parse already
    # reports stick-count extents, so the `extent % factor == 0` gate above keeps
    # candidates stick-divisible. _split_fits_sticks still catches the residual
    # case it can't — a factor landing on a collapsed/broadcast dim (no stick
    # count). The seed is always kept (work_dist's own choice).
    return [
        opt for opt in options.values() if opt == seed or _split_fits_sticks(op, opt)
    ]


def _enum_split_options(
    op: Operation,
    extra_bases: tuple[tuple[dict, dict], ...] = (),
    matmul_roles: tuple[dict[str, int], ...] = (),
) -> list[tuple[dict, dict]]:
    """Split options for a pointwise op: the seed (index 0) plus variants
    that flip the split onto another output dim (≤ DEFAULT_VARIANT_CAP).

    `extra_bases` (the matmuls' output-splits) are offered on top so the op
    can adopt a matmul's tiling and pin its shared buffer to LX. Matmuls take
    their dedicated cross-matmul + factored-B/M path; non-matmul reductions are
    offered the factored B/M splits (so a softmax chain's max/sum can reconcile
    with the chain instead of forcing a core-div mismatch). Invalid bases
    self-eliminate during scoring.

    `matmul_roles` map BMNK to splits, {"M": 4, "N: 8, ...} then apply to other matmuls.
    """
    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    is_output_splits_empty = seed[0] == {}
    is_computed_buf = isinstance(op, ComputedBuffer)
    is_reduction = is_computed_buf and isinstance(op.data, Reduction)
    is_matmul = is_reduction and _is_matmul_op(op)

    # TODO: let a matmul also consider the *other* matmuls' splits.
    # Remove once work_dist assigns consistent splits.
    if is_matmul and matmul_roles:
        return _check_and_add_matmul_option(op, seed, matmul_roles)

    # A non-matmul reduction (e.g. softmax max/sum) keeps B and M as output dims;
    # offer it the factored B/M splits so it can reconcile with a B/M-tiled chain
    # rather than being stuck on its seed and breaking the chain's per-core views.
    # We do NOT flip reductions onto other dims (their reduction axis is fixed).
    if is_reduction and not is_matmul and not is_output_splits_empty:
        red_options: dict[tuple, tuple[dict, dict]] = {_canonical_key(seed): seed}
        for base in _factored_bm_splits(op, _reduction_bm_axes(op)):
            red_options.setdefault(_canonical_key(base), base)
        return [
            opt
            for opt in red_options.values()
            if opt == seed or _split_fits_sticks(op, opt)
        ]

    # Only pointwise ops are flipped; reductions/matmuls keep work-division's
    # split. For compute-bound ops, prioritize PT utilization over LX pinning:
    # overriding a matmul's split to chase pinning regressed kernel time ~2.5x
    # (mlp-linear-kn.t, SENCORES=32; PT-util 66%→33%). Exclude future
    # compute-bound ops here too.
    # is_matmul implies is_reduction, so it's covered by the is_reduction term.
    if is_output_splits_empty or not is_computed_buf or is_reduction:
        return [seed]

    # Recover seed's per-symbol form to mutate the slicing.
    rw = op_read_writes(op)
    write_index = next(iter(rw.writes)).index
    first_read = next(iter(rw.reads), None)
    read_index = first_read.index if first_read is not None else write_index
    iter_space = iteration_space_from_op(op)
    seed_per_sym = apply_splits_from_index_coeff(
        seed, write_index, read_index, iter_space
    )

    sliced_output_syms = [
        s for s in seed_per_sym if seed_per_sym[s] > 1 and write_index.coeff(s) != 0
    ]

    # Dedup-and-collect in one dict: canonical key -> split tuple (the split
    # tuple itself is two dicts, so it can't be a key directly). Insertion
    # order is preserved, so the seed stays first.
    options: dict[tuple, tuple[dict, dict]] = {_canonical_key(seed): seed}

    # Only single output-dim splits are flipped. Multi-dim splits (e.g.
    # k_fast (1, n, k)) aren't yet handled.
    if len(sliced_output_syms) != 1:
        return [seed]
    seed_sym = sliced_output_syms[0]
    seed_factor = int(seed_per_sym[seed_sym])

    for sym, extent in iter_space.items():
        extent_int = concretize_expr(extent)
        if (
            sym is seed_sym
            or write_index.coeff(sym) == 0
            or extent_int <= 1
            or extent_int % seed_factor != 0
        ):
            continue
        variant_per_sym = dict(seed_per_sym)
        variant_per_sym[seed_sym] = 1
        variant_per_sym[sym] = seed_factor
        variant = splits_by_index_coeff(variant_per_sym, write_index, read_index)
        options.setdefault(_canonical_key(variant), variant)
        if len(options) >= DEFAULT_VARIANT_CAP:
            break

    # Let this pointwise op adopt a matmul's tiling to pin its shared buffer to
    # LX. High-value, so added regardless of DEFAULT_VARIANT_CAP (flips only).
    for base in extra_bases:
        options.setdefault(_canonical_key(base), base)
    # Load-bearing here (unlike the matmul path): the variant gate above tests
    # the element extent (extent_int % seed_factor), not the stick count, so it
    # can admit a factor that overflows the stickified dim's stick count — which
    # would SIGABRT the SDSC bundler. _split_fits_sticks drops those (and any
    # factor on a collapsed/broadcast dim). The seed is always kept: if it itself
    # is over-stick, that is work_dist's choice and not ours to discard here.
    return [
        opt for opt in options.values() if opt == seed or _split_fits_sticks(op, opt)
    ]


def _canonical_key(splits: tuple[dict, dict]) -> tuple:
    """Hashable key for a (output_splits, reduction_splits) pair."""
    out, red = splits
    return (tuple(sorted(out.items())), tuple(sorted(red.items())))


class StrategyBCoOptimizingAllocator(ScratchpadAllocator):
    """`Strategy B` assumes work_distribution committed one best option (seed). Here we
    first add a few variants based on the seed, pick the combination that minimizes HBM
    bytes among all, then defer to ScratchpadAllocator's flow. As seed is in the search
    space, the worst case matches ScratchpadAllocator.
    """

    def plan_allocation(self, graph: GraphLowering):
        self.reject_reasons = {}
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)

        # Enumerate options, run search, commit winners back to op_it_space_splits.
        ops = graph.operations

        # Distinct matmul output-splits (drop K) to seed the pointwise search.
        matmul_bases, matmul_roles = _find_distinct_matmul_splits(ops)

        options_per_op = [
            _enum_split_options(op, matmul_bases, matmul_roles) for op in ops
        ]
        t1 = time.perf_counter()
        best_chosen, timings, search_cache, search_lifetimes = self._search(
            graph, ops, options_per_op
        )
        t_search = time.perf_counter() - t1

        for op, opt_idx, options in zip(ops, best_chosen, options_per_op):
            chosen = options[opt_idx]
            if chosen != getattr(op, "op_it_space_splits", ({}, {})):
                op.op_it_space_splits = chosen

        n_paths = math.prod(len(o) for o in options_per_op)
        winner = {
            f"{ops[i].get_name()}({self._get_op_name(ops[i])})": options_per_op[i][
                best_chosen[i]
            ]
            for i in range(len(ops))
            if len(options_per_op[i]) > 1
        }
        logger.info(
            "co-opt search: %d paths in %.1fms (key components in "
            "_generate_buffers(): residency %.1fms + mem_usage %.1fms); "
            "winner=%s",
            n_paths,
            t_search * 1e3,
            timings["residency"] * 1e3,
            timings["mem_usage"] * 1e3,
            winner,
        )

        # try insert clone again, as what was incompatible could be compatible now
        # TODO simplify the previous pre-opt (at the beginning of this func), we will
        # run check core-div-mismatch a few times due to clone-insertion, speed-up?
        n_ops_before_clone = len(graph.operations)
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)

        # Standard downstream flow on the now-fixed winning splits. Mirrors
        # ScratchpadAllocator.plan_allocation past the pre-passes. Reuse the search's
        # per-core-view cache + liveness only if the clone pass left the graph
        # unchanged: a clone insertion both appends an op (shifts the
        # position-indexed liveness) and rewrites input consumers' MemoryDep to read
        # the clone (changes the (op, splits, dep) cache key), so on any op-count
        # change both are stale and we rebuild from scratch (cache=lifetimes=None).
        clone_inserted = len(graph.operations) != n_ops_before_clone
        buffers = self._generate_buffers(
            graph,
            cache=None if clone_inserted else search_cache,
            lifetimes=None if clone_inserted else search_lifetimes,
        )
        assert self.layout_planning is not None
        allocation = self.layout_planning.plan_layout(buffers, log_lx_usage=True)
        self._record_spill_reasons(allocation)
        self._push_allocation(graph, allocation)
        self._log_lx_pinning(graph)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        graph: GraphLowering,
        ops: list[Operation],
        options_per_op: list[list[tuple[dict, dict]]],
    ) -> tuple[list[int], dict[str, float], dict, dict[str, list[int]]]:
        """DFS over the option cross-product, scoring each leaf via
        _score_layout. Returns (best option index per op, timing breakdown in
        seconds, _per_core_view_on_buf cache, liveness). The timing dict has
        keys `residency` and `mem_usage` — the two split-dependent
        shared-object builds inside _generate_buffers, which dominate per-leaf
        cost. Liveness is split-invariant and computed once here, not per leaf.
        No early-stop pruning — bounded by ≤ K^N leaves where N counts ops with
        >1 option (most return [seed]). Per-leaf cost is one full
        _generate_buffers + plan_layout pass; the `cache` param on
        _per_core_view_on_buf amortizes sympy work if it ever becomes hot. The
        cache and liveness are returned so the final commit pass can reuse them
        when the post-search clone pass leaves the graph unchanged (see
        plan_allocation).
        """
        chosen: list[int] = [0] * len(ops)
        best_total: float = math.inf
        best_chosen: list[int] = list(chosen)
        timings: dict[str, float] = {
            "residency": 0.0,
            "mem_usage": 0.0,
        }

        buf_total_bytes: dict[str, int] = {
            name: math.prod(buf.layout.device_layout.device_size[:-1]) * 128
            for name, buf in graph.name_to_buffer.items()
        }

        # get_read_writes() re-traces the store function over the iteration space
        # on every call and is NOT memoized upstream, yet its result is
        # split-invariant (the symbolic deps don't depend on op_it_space_splits).
        # The per-leaf residency/get_ncores path calls it for every op, so across
        # ~K^N leaves it would dominate — but `op_read_writes` memoizes it per op
        # instance (split-invariant), so the first leaf warms the cache for all.

        # Liveness depends only on graph structure (not op.op_it_space_splits),
        # so compute it once for the whole search instead of per leaf.
        lifetimes = calculate_liveness(graph)

        # Memoize _per_core_view_on_buf across leaves. Keyed on
        # (op name, split values, dep) — see _per_core_view_on_buf for why
        # op name is required. A single dict is correct across the whole
        # search; scoped to this graph only since dep is not unique across
        # graphs.
        cache: dict = {}

        def recurse(op_idx: int) -> None:
            nonlocal best_total, best_chosen
            if op_idx == len(ops):
                hbm = self._score_layout(
                    graph, buf_total_bytes, cache, timings, lifetimes
                )
                if hbm < best_total:
                    best_total = hbm
                    best_chosen = list(chosen)  # list() makes a copy
                return

            op = ops[op_idx]
            options = options_per_op[op_idx]

            # Mutate-and-undo: stash and restore op.op_it_space_splits.
            # If the op originally lacked the attribute, restore it as
            # ({}, {}) — equivalent to "unset" for all readers (which use
            # getattr(..., ({}, {})) or hasattr+empty-dict default).
            prev_split: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
            for opt_idx, option in enumerate(options):
                op.op_it_space_splits = option
                chosen[op_idx] = opt_idx
                recurse(op_idx + 1)
            op.op_it_space_splits = prev_split

        recurse(0)
        return best_chosen, timings, cache, lifetimes

    # ------------------------------------------------------------------
    # Leaf scoring
    # ------------------------------------------------------------------

    def _score_layout(
        self,
        graph: GraphLowering,
        buf_total_bytes: dict[str, int],
        cache: Optional[dict] = None,
        timings: Optional[dict[str, float]] = None,
        lifetimes: Optional[dict[str, list[int]]] = None,
    ) -> int:
        """HBM bytes under the current split assignment: total device
        bytes of every buffer the solver couldn't pin. Non-committing
        (addresses land on throwaway buffers) and solver-agnostic.

        If `timings` is provided, _generate_buffers accumulates its
        `residency` / `mem_usage` sub-step seconds into it. `lifetimes`
        (split-invariant) is forwarded to avoid recomputing it per leaf.
        """
        buffers = self._generate_buffers(graph, cache, timings, lifetimes)
        assert self.layout_planning is not None
        allocation = self.layout_planning.plan_layout(buffers)
        pinned_names = {b.name for b in allocation if b.address is not None}

        return sum(
            total for name, total in buf_total_bytes.items() if name not in pinned_names
        )


class CoOptimizingAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: CoreDivisionLayoutSolver,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Joint core-division + LX-placement allocator.

        Args:
            layout_planning: A core-division-aware solver (the OR-Tools
                ``CpSatLayoutSolver``). This allocator drives the *joint* entry
                point, so it needs the ``CoreDivisionLayoutSolver`` interface
                rather than a plain ``MemoryPlanSolver``. The ortools-missing
                fallback to greedy placement lives in :func:`select_allocator`,
                which never constructs this allocator without a valid solver.
            pre_optimization_passes: Graph passes applied before layout planning.
            post_optimization_passes: Graph passes applied after layout planning.
        """
        super().__init__(
            layout_planning=layout_planning,
            pre_optimization_passes=pre_optimization_passes,
            post_optimization_passes=post_optimization_passes,
        )
        # Narrow the base's ``MemoryPlanSolver`` annotation: the joint entry
        # point requires the core-division interface.
        self.layout_planning: Optional[CoreDivisionLayoutSolver] = layout_planning

    def _prepare_buffers(self, graph: GraphLowering) -> Sequence[Any]:
        in_place = self._determine_in_place_division_invariant(graph)
        buffers = self._build_cd_bound_buffers(
            graph, in_place, self._division_map(graph)
        )
        return buffers

    def _solve(self, buffers: Sequence[Any]) -> Sequence[Any]:
        assert self.layout_planning is not None
        return self.layout_planning.plan_layout_and_core_divisions(buffers)

    def _post_solve(self, graph: GraphLowering, allocation: Sequence[Any]) -> None:
        # The divisions must be committed such that any buffer clones can correctly
        # pull the selected core division from the dependent buffers when the graph
        # is updated with clones in ``_push_allocation``.
        self._commit_divisions(graph, allocation)

    def _record_reject_reasons(self, allocation: Sequence[Any]) -> None:
        # Surface the solver's per-buffer spill causes so the LX-pinning debug
        # log reports why each buffer landed in HBM, on par with the other
        # allocators. ``getattr`` because only ``CpSatLayoutSolver`` exposes it.
        self.reject_reasons = dict(getattr(self.layout_planning, "spill_reasons", {}))

    def _division_map(self, graph: GraphLowering) -> dict[str, list[CoreDivision]]:
        """Per-op core-division candidates for the joint-division solve.

        Every op gets at least one ``CoreDivision`` so the slicing-match gate can
        constrain it. Pointwise / Reduction ops get the enumerated candidates;
        every other op falls back to a single fixed division read off its
        committed ``op_it_space_splits``. No op-kind pre-filter -- residency is
        gated per buffer (``_residency_by_buf``) and by the solver, so ineligible
        ops still participate as producers/consumers in the match.

        Exception: ops data-connected to a sliced in-place mutation (a constant-
        offset write, e.g. ``x[:, 32:96] = ...``) are pinned to their upstream
        (fixed) division. Re-slicing any op fused into the offset write's SDSC
        makes the deeptools scheduler reject it (``DtException: "There must be at
        least one valid candidate"``), the root cause of the
        ``slice_stick_mutation_*`` failures. Keeping the fixed division there
        matches the schedulable slicing the greedy path uses; it costs only a
        division optimization, never correctness. See
        ``utils.ops_in_offset_mutation_component``.
        """
        max_cores = config.sencores
        fixed_division_ops = ops_in_offset_mutation_component(graph)
        return {
            op.name: (
                [_fixed_core_division(op)]
                if op.name in fixed_division_ops
                else self._enumerate_core_divisions(op, max_cores)
            )
            for op in graph.operations
        }

    def _enumerate_core_divisions(
        self, op: Operation, max_cores: int
    ) -> list[CoreDivision]:
        """Core-division candidates for one eligible op (see ``_division_map``).

        Each ``enumerate_work_division_candidates`` split is encoded into the
        stride-keyed ``(output_splits, reduction_splits)`` form and deduped by
        slicing signature. Ops without a divisible iteration space, or whose
        space can't be enumerated, fall back to a single fixed division.
        """
        fixed = [_fixed_core_division(op)]
        if not isinstance(op, ComputedBuffer) or not isinstance(
            op.data, (Pointwise, Reduction)
        ):
            return fixed
        rw = op_read_writes(op)
        write = next(iter(rw.writes), None)

        # this is essentially a dead branch but serves as a type narrowing below
        if write is None:
            return fixed
        write_index = write.index
        first_read = next(iter(rw.reads), None)
        read_index = first_read.index if first_read is not None else write_index

        try:
            candidates = enumerate_work_division_candidates(op, max_cores)
        except Unsupported as exc:
            # Symbolic stick dims etc. can't be enumerated; leave the op on its
            # upstream-chosen split (fixed division).
            logger.debug("skip joint division for %s: %s", op.name, exc)
            return fixed

        cds: list[CoreDivision] = []
        seen: set[tuple] = set()
        for cand in candidates:
            out_s, red_s = splits_by_index_coeff(cand, write_index, read_index)
            key = (
                tuple(sorted(out_s.items())),
                tuple(sorted(red_s.items())),
            )
            if key in seen:
                continue
            seen.add(key)
            cds.append(CoreDivision(output_splits=out_s, reduction_splits=red_s))
        return cds or fixed

    def _commit_divisions(
        self,
        graph: GraphLowering,
        allocation: Sequence[CoreDivisionBuffer],
    ) -> None:
        """Write the solver's chosen division back to ``op.op_it_space_splits``
        for *every* buffer the solver assigned one.

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
            cd = buf.core_divisions[buf.chosen_division]
            op.op_it_space_splits = (
                dict(cd.output_splits),
                dict(cd.reduction_splits),
            )

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
                # solver's ``_assert_in_place_relationships`` would fail to resolve
                # them). Skip them, matching the base allocator's guard.
                if input_buf not in mem_usage or not lifetimes[input_buf]:
                    continue
                in_layout = graph.get_buffer(input_buf).layout
                if not hasattr(in_layout, "device_layout"):
                    continue
                in_end = lifetimes[input_buf][-1]  # inclusive last use
                in_ten_layout = in_layout.device_layout
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start  # same op reads input, writes output
                if inp_i_lay_match and inp_i_eol:
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
        solver picks a division and divides by its ``output_partition``. Because
        all buffers are on the same total scale, ``in_place_parents`` need no
        filtering."""
        lifetimes = calculate_liveness(graph)
        mem_usage = mem_usage_by_buf(graph)
        in_place = {} if in_place is None else in_place
        op_by_name = {op.name: op for op in graph.operations}
        graph_output_names = set(graph.get_output_names())

        prep_cache: dict = {}
        buffers: list[CoreDivisionBuffer] = []
        residency_by_buf = self._residency_by_buf(graph, mem_usage, lifetimes)

        input_clone_matches: dict[str, dict[str, list[tuple[int, int]]]] = {}
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
                        boundary=BufferType.Input,
                    )
                )

        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]

            op = op_by_name.get(output_name)
            residency_reason = residency_by_buf[output_name]

            buf_divisions = divisions[output_name]
            parents = in_place.get(output_name, [])
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

            buffers.append(
                CoreDivisionBuffer(
                    output_name,
                    size,
                    uses,
                    first_use_is_read=True,
                    in_place_parents=parents,
                    core_divisions=buf_divisions,
                    parents=parent_proj,
                    cd_parent_matches=cd_parent_matches,
                    residency_reason=residency_reason,
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
            iter_space = iteration_space_from_op(consumer)
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
                    per_sym = apply_splits_from_index_coeff(
                        (cd.output_splits, cd.reduction_splits),
                        write.index,
                        read_dep.index,
                        iter_space,
                    )
                    clone_out, _ = splits_by_index_coeff(
                        per_sym, read_dep.index, read_dep.index
                    )
                    k = len(clone_divs)
                    clone_divs.append(
                        CoreDivision(
                            output_splits=clone_out, reduction_splits={}
                        )  # a clone op cannot have a division split
                    )
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
            coeff = (cd.output_splits, cd.reduction_splits)
            # Build the op-level prep once per key, on first sight, regardless
            # of whether this candidate splits. ``_per_core_view_from_prep``
            # still short-circuits to the whole-buffer view for a no-split
            # candidate, but always populating the cache keeps an absent entry
            # distinct from a genuine ``None`` prep, so a later candidate (or a
            # cache reuse) can't silently get a stale/``None`` view.
            if key not in prep_cache:
                prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
            out.append(_per_core_view_from_prep(prep_cache[key], coeff))
        return out


_PLACEMENT_SOLVERS: dict[str, type[MemoryPlanSolver]] = {
    "greedy": GreedyLayoutSolver,
    "bestfit": BestFitLayoutSolver,
    "firstfit": FirstFitLayoutSolver,
    "simulated_annealing": SimulatedAnnealingLayoutSolver,
}


def _make_cpsat_solver(size: int) -> Optional[MemoryPlanSolver]:
    """Build the CP-SAT layout solver, or ``None`` when ortools is unavailable.

    Imported lazily so this module (and every non-cpsat path) loads without
    ortools installed; ``CpSatLayoutSolver.__init__`` raises ``ImportError`` when
    ortools (``cp_model``) is missing, which we translate to ``None`` so callers
    can fall back to a placement-only greedy solve.
    """
    try:
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        return CpSatLayoutSolver(size)
    except ImportError as exc:
        logger.warning(
            "cpsat layout solver unavailable (%s); falling back to the "
            "default greedy allocator.",
            exc,
        )
        return None


def select_allocator() -> ScratchpadAllocator:
    """Build the scratchpad allocator and inject its layout solver from config.

    This is the single place that maps config to an (allocator, solver) pair, so
    the allocators themselves take an explicit solver and never inspect config:

    * ``layout_solver == "cpsat"`` with ``co_optimizing_lx_planning`` -> joint
      core-division + LX placement via :class:`CoOptimizingAllocator`. Falls back
      to placement-only greedy :class:`ScratchpadAllocator` when ortools is absent.
    * ``layout_solver == "cpsat"`` without co-optimization -> placement-only
      :class:`ScratchpadAllocator` driven by the CP-SAT solver, placing buffers on
      each op's pre-determined core division (the buffers are converted to
      trivial ``CoreDivisionBuffer``s). Falls back to greedy when ortools is
      absent.
    * ``co_optimizing_lx_planning`` (non-cpsat solver) -> gap-based
      co-optimization via :class:`StrategyBCoOptimizingAllocator`.
    * otherwise -> placement-only :class:`ScratchpadAllocator` with the configured
      gap-based solver (greedy/bestfit/firstfit).
    """
    size = _lx_planning_size()
    if config.layout_solver == "cpsat":
        # Both cpsat paths share the same ortools-missing degradation: build the
        # CP-SAT solver here and fall back to greedy placement (still correct)
        # when it is unavailable, so the allocators never see a ``None`` solver.
        solver = _make_cpsat_solver(size)
        if config.co_optimizing_lx_planning:
            if solver is None:
                return ScratchpadAllocator(layout_planning=GreedyLayoutSolver(size))
            # CpSatLayoutSolver implements the core-division interface the joint
            # allocator drives; the narrowing is safe since this is the only
            # solver _make_cpsat_solver ever returns.
            assert isinstance(solver, CoreDivisionLayoutSolver)
            return CoOptimizingAllocator(layout_planning=solver)
        # Placement-only CP-SAT on the pre-determined core divisions.
        if solver is None:
            logger.debug(
                "falling back to greedy solver. Make sure Or-Tools is available"
            )
            return ScratchpadAllocator(layout_planning=GreedyLayoutSolver(size))
        return ScratchpadAllocator(layout_planning=solver)

    try:
        solver_cls = _PLACEMENT_SOLVERS[config.layout_solver]
    except KeyError:
        raise ValueError(
            f"Invalid layout_solver config option '{config.layout_solver}'."
        )
    solver = solver_cls(size)

    if config.co_optimizing_lx_planning:
        return StrategyBCoOptimizingAllocator(layout_planning=solver)
    return ScratchpadAllocator(layout_planning=solver)


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
        ScratchpadAllocator(GreedyLayoutSolver(_lx_planning_size())).plan_allocation(
            graph
        )
