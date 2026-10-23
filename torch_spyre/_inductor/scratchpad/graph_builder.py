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

"""The LX layout solve's graph, built from the IR, the same for every
encoding strategy.

:class:`GraphBuilder` puts on each node its candidate core divisions and
coarse tilings (a menu, and the space it is drawn from where the op has one)
and on each edge the relations a strategy encodes it by (``ResidencyEdge``,
``CloneRead``, ``RelayoutEdge``), with a pair table only where the builder
must fix one (the loop-carry intersection, input clones). It reads each op's
residency rules from ``residency_rules``, its lifetimes and drain plans from
``drain_plans`` and its split options from ``division_menus``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Collection, Sequence
from typing import Any, Optional

import sympy
from torch._inductor.ir import (
    ComputedBuffer,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    op_read_writes,
)
from torch_spyre._C import get_device_size_in_bytes
from torch_spyre._inductor.work_division import (
    ResidencyEdge,
    build_residency_edge,
    enumerate_work_division_candidates,
    has_resolved_work_div_hint,
    _core_division,
    _view_for_div,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.buffer_graph import (
    SpyreOptimizationGraph,
    empty_graph,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    TileSpec,
    BufferType,
)
from torch_spyre._inductor.scratchpad.utils import (
    clone_at_graph_boundaries,
    mem_usage_by_buf,
    calculate_liveness,
    get_buffer_users,
    ops_in_offset_mutation_component,
    counted_loop_lifetime_overrides,
)
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    KEEP_BY_INDEX_OP,
)
from torch_spyre._inductor import config
from torch_spyre._inductor.loop_info import (
    LoopCarryRecord,
    ReadCopyElisionRecord,
)
from torch_spyre._inductor.padding import is_restickify_op
from torch_spyre._inductor.scratchpad.lx_relayout import (
    _unsupported_relayout_transition_reason,
    core_domain_rejection,
    grouped_gather_rejection,
    partition_footprint,
    solver_relayout_edge_context,
    solver_relayout_pair_cost,
    work_division_from_view,
)
from torch_spyre._inductor.op_spec import TensorWorkDivision

from torch_spyre._inductor.scratchpad.division_menus import (
    _DivisionMap,
    _drop_read_distance_violations,
    _enum_split_options,
    _find_distinct_matmul_splits,
    _fixed_core_division,
    _fused_layout_group_ops,
    _is_cpu_host_buffer,
    _is_indirect_access_op,
    _is_windowed_pool,
    _legal_fixed_division,
    _legal_split_options,
    _reads_offset_slice,
)
from torch_spyre._inductor.scratchpad.drain_plans import (
    DrainPlan,
    _coarse_tile_loop_trip_count,
    _drain_lifetime_end_overrides,
    _drained_collectives,
    _handoff_child_start,
    _handoff_parent_end,
    validated_drain_plans,
)
from torch_spyre._inductor.scratchpad import residency_rules
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.scratchpad.pricing import _COST_PARAMS

logger = get_inductor_logger("scratchpad.graph_builder")


def residency_symbols(graph: SpyreOptimizationGraph) -> dict:
    """Each node's residency symbol (``is_lx_<name>``),
    what the cost model prices an LX read or write by."""
    return {n.name: n.sym_is_lx for n in graph.get_nodes()}


class RelayoutEdge:
    """One (producer, consumer) edge whose read may go through a relayout
    copy: the division-independent edge context, and the per-pair gates and
    price of a (source view, destination view) pair. Every per-pair gate is a
    function of the two per-core views and their core counts, so a strategy
    can tabulate it over view ids instead of division pairs (2d)."""

    def __init__(
        self,
        parent: str,
        parent_op: Operation,
        consumer_op: Operation,
        context: tuple,
        pair_costs: dict[tuple, Optional[float]],
        capacity: int,
    ) -> None:
        (
            self.write_dep,
            self.read_dep,
            self.prod_coords,
            self.cons_coords,
            self.prod_space,
            self.cons_space,
        ) = context
        self.parent = parent
        self.parent_op = parent_op
        self.consumer_op = consumer_op
        self.device_dims = list(parent_op.layout.device_layout.device_size)
        self.out_elems = math.prod(self.device_dims)
        self.dtype_bytes = parent_op.get_dtype().itemsize
        # solver_relayout_edge_context's same-tile-group exception admits a
        # coarse-tiled parent/consumer pair: that candidate's copy runs once per
        # loop trip, not once total, so its price must scale by the loop's own
        # trip count or the solver sees an under-priced option that could
        # out-rank a correctly-priced alternative whose real total cost is
        # actually lower (cyang49's PR #5057 review).
        self.loop_trip_count = _coarse_tile_loop_trip_count(parent_op)
        # Graph-wide cache: the same (source view, destination view, cores,
        # tensor geometry) recurs across structurally identical ops (the
        # unrolled KV blocks of attention), and pricing it re-runs the movement
        # gate's per-core owner comparison each time.
        self._pair_costs = pair_costs
        self.capacity = capacity
        self._projected: dict[tuple, Optional[TensorWorkDivision]] = {}
        self._spans: dict[PerCoreView, Optional[int]] = {}

    def source_views(
        self, divs: Sequence[CoreDivision], prep_cache: dict
    ) -> list[Optional[PerCoreView]]:
        """The producer's per-core view under each division; ``None`` where it
        cannot host a relayout source. Same screens as a matched producer: the
        source gets LX-pinned exactly like one, so the same coherence bars
        apply (partial-reduction write, unrepresentable slicing)."""
        return [
            view if repr_ok and not partial and not _candidate_tiled(cd) else None
            for cd, (view, partial, repr_ok) in zip(
                divs,
                GraphBuilder._views_for_divs(
                    self.parent_op, self.write_dep, self.parent, list(divs), prep_cache
                ),
            )
        ]

    def consumer_views(
        self, divs: Sequence[CoreDivision], prep_cache: dict
    ) -> list[Optional[PerCoreView]]:
        """The consumer's read view of the producer under each division;
        ``None`` where it cannot read through a copy."""
        return [
            view if repr_ok and not _candidate_tiled(cd) else None
            for cd, (view, _partial, repr_ok) in zip(
                divs,
                GraphBuilder._views_for_divs(
                    self.consumer_op,
                    self.read_dep,
                    self.parent,
                    list(divs),
                    prep_cache,
                ),
            )
        ]

    def _projected_on(self, view, frame: str) -> Optional[TensorWorkDivision]:
        """``view``'s ownership projected into loop symbols on the producer
        (``"prod"``) or consumer frame (the committed path's
        ``work_division_from_view`` gates), cached per view: several
        candidates often induce the same view. The projected division is kept
        (not just a bool) because the transition gate compares source and
        destination divisions for equality."""
        key = (view, frame)
        if key not in self._projected:
            coords, space = (
                (self.prod_coords, self.prod_space)
                if frame == "prod"
                else (self.cons_coords, self.cons_space)
            )
            try:
                self._projected[key] = work_division_from_view(
                    view, self.device_dims, coords, space
                )
            except ValueError:
                self._projected[key] = None
        return self._projected[key]

    def _span(self, view: PerCoreView) -> Optional[int]:
        """The per-core LX span of a view (#3440's reservation for a relayout
        member). Unavailable (non-standard arrangement, an unsplittable stick
        axis) is a decline, as on the committed path: without a size the copy
        cannot be placed, so the pair is never offered."""
        if view not in self._spans:
            try:
                self._spans[view] = partition_footprint(self.parent_op.layout, view)
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "[lx solver relayout] %s: span unavailable for view %s: %s",
                    self.parent,
                    dict(view.work_slice_dims),
                    exc,
                )
                self._spans[view] = None
        return self._spans[view]

    def price(
        self, pv: PerCoreView, cv: PerCoreView, ncores: int, dst_cores: int
    ) -> Optional[tuple[float, int, int]]:
        """``(cost_ns, source span, destination span)`` of reading through a
        copy from source view ``pv`` on ``ncores`` cores to destination view
        ``cv`` on ``dst_cores``, or ``None`` when the pair cannot relayout.
        The caller has already ruled out equal views (a match, not a
        shuffle)."""
        # The committed collector's consumer rules (#3440): the core domains,
        # then the gather rule on the destination view; the geometric half is
        # movement_supported's, inside the pair cost.
        if core_domain_rejection(ncores, dst_cores):
            return None
        if grouped_gather_rejection(self.consumer_op, ncores, cv):
            return None
        if self._projected_on(pv, "prod") is None:
            return None
        # The relayout copy iterates the producer's shape, so the destination
        # view must also project on the producer frame (materialize_lx_relayouts
        # commits it there and raises).
        if self._projected_on(cv, "prod") is None:
            return None
        src_division = self._projected_on(pv, "cons")
        dst_division = self._projected_on(cv, "cons")
        if src_division is None or dst_division is None:
            return None
        # Distinct per-core views that collapse to one logical work division
        # would codegen as a plain identity copy with the cross-core movement
        # silently omitted (#3926); the solver must never be offered such a pair.
        if (
            _unsupported_relayout_transition_reason(src_division, dst_division)
            is not None
        ):
            return None
        source_span, destination_span = self._span(pv), self._span(cv)
        if (
            source_span is None
            or destination_span is None
            or destination_span > self.capacity
        ):
            return None
        key = (
            pv,
            cv,
            ncores,
            dst_cores,
            tuple(self.device_dims),
            self.out_elems,
            self.dtype_bytes,
        )
        if key not in self._pair_costs:
            self._pair_costs[key] = solver_relayout_pair_cost(
                pv,
                cv,
                ncores,
                self.device_dims,
                self.out_elems,
                self.dtype_bytes,
                destination_num_cores=dst_cores,
            )
        cost = self._pair_costs[key]
        if cost is None:
            return None
        # The cache is graph-wide and keyed on geometry alone (shared across
        # parents), so the trip multiplier is applied to this pair's own cost,
        # never written back into the cached value.
        return cost * self.loop_trip_count, source_span, destination_span


def _candidate_tiled(cd: CoreDivision) -> bool:
    """A coarse-tiled candidate can never host a relayout, for the same
    reasons a coarse-tiled op cannot in general (its buffer is per-tile
    scratch)."""
    tiling = getattr(cd, "tiling", None)
    return tiling is not None and not getattr(tiling, "is_untiled", True)


class CloneRead:
    """How one consumer reads an input clone (2c). A clone has no write view
    of its own: its view is the one its consumers read it with, so each
    consumer division maps to the clone view its read has (an index into the
    clone's menu), or to none (-2) when that read is unrepresentable or a
    broadcast. The relation on the edge is then equality of the clone's index
    and the consumer's."""

    def __init__(
        self,
        consumer: Operation,
        read_dep: MemoryDep,
        input_name: str,
        clone_views: list[PerCoreView],
        prep_cache: dict,
    ) -> None:
        self.consumer = consumer
        self.read_dep = read_dep
        self.input_name = input_name
        self.clone_views = clone_views
        self.prep_cache = prep_cache

    def clone_index(self, divs: Sequence[CoreDivision]) -> list[int]:
        """Per consumer division, the clone view index its read has."""
        views = GraphBuilder._views_for_divs(
            self.consumer, self.read_dep, self.input_name, list(divs), self.prep_cache
        )
        out = []
        for view, _, repr_ok in views:
            k = _clone_view_index(view, repr_ok, self.clone_views)
            out.append(-2 if k is None else k)
        return out


def _clone_view_index(
    view: PerCoreView, repr_ok: bool, clone_views: list[PerCoreView]
) -> Optional[int]:
    """The clone view a consumer's read ``view`` equals, or ``None`` when no
    clone can serve the read: unrepresentable, a broadcast read (the division
    runs on more cores than the view covers: it splits an axis the input
    lacks), or a view no clone candidate has."""
    if not repr_ok or math.prod(f for _, f in view.work_slice_dims) != view.num_cores:
        return None
    return next((i for i, c in enumerate(clone_views) if c.same_partition(view)), None)


def _has_legal_division(context) -> bool:
    """Whether ``context`` admits any division (stops at the first)."""
    axes = context.axes
    return any(
        context.is_legal(dict(zip(axes, combo)))
        for combo in itertools.product(*(context.factor_domain(v) for v in axes))
    )


class GraphBuilder:
    """The solve's graph from the IR, for solves over ``capacity`` bytes of
    LX, with each node's candidates and each edge's relations, the same for
    every encoding strategy. An op whose divisions form a small space gets the
    space and its points as its menu (never enumerated down to the legal
    ones); others get their legal divisions. An edge carries the relations a
    strategy may derive from (``ResidencyEdge``, ``CloneRead``,
    ``RelayoutEdge``) and a pair table only where the builder must fix one (a
    loop carry, an input clone). ``committed``: every op keeps the division
    work division committed, a menu of one."""

    def __init__(
        self,
        capacity: int,
        prune: bool = False,
        committed: bool = False,
    ) -> None:
        self.capacity = capacity  # bytes
        self.prune = prune
        self.committed = committed
        # The post-loop drain plans validated for the last graph.
        self.drain_plans: dict[str, DrainPlan] = {}
        # Set for the re-plan after applied tilings: no further tilings.
        self.suppress_tiling = False
        self._relayout_pair_costs: dict[tuple, Optional[float]] = {}
        # Per enumerated op, its divisions as a space (a Lazy split's).
        self.spaces: dict[str, Any] = {}
        # Ops whose menu is their space's points, not the legal divisions.
        self.point_menus: set[str] = set()

    def prepare_graph(self, graph: GraphLowering) -> SpyreOptimizationGraph:
        """The solve's graph of ``graph``, its candidates on it."""
        # Validate the post-loop drain plans exactly once, here, before the
        # solver runs: the residency gate, the lifetime extension and the push
        # must all consume this same object.  Re-deriving it after the solve
        # could disagree with what residency actually admitted.
        self.drain_plans = validated_drain_plans(graph)
        in_place = self._determine_in_place_division_invariant(graph)
        division_map = self._division_map(graph, allow_deferred_read_candidates=True)
        divisions = division_map.divisions
        pending = {
            op.name: op
            for op in graph.operations
            if hasattr(op, "_read_copy_elision_record")
            and is_restickify_op(op, graph)
            and divisions[op.name] != [_fixed_core_division(op)]
        }
        if pending:
            for name in list(self.point_menus):
                self._materialize(divisions, name)
        while True:
            solve_graph = self._build_graph(graph, in_place, division_map)
            if not pending:
                return solve_graph
            pricing = {
                op.get_name(): op for op in self._pricing_operations(graph, solve_graph)
            }
            rejected = [
                name
                for name, op in pending.items()
                if pricing.get(name) is op
                or not self._direct_read_candidates_priced(
                    pricing.get(name), divisions[name], solve_graph
                )
            ]
            if not rejected:
                return solve_graph
            for name in rejected:
                op = pending.pop(name)
                fixed = _fixed_core_division(op)
                assert fixed.cores_used <= config.sencores, (
                    f"{name}: fixed direct-read division over the "
                    f"{config.sencores}-core budget"
                )
                divisions[name] = _legal_fixed_division(
                    op, [fixed], "unproved or unpriced direct read"
                )
                division_map.enumerated.discard(name)
            # Menus affect input clones and relayouts. Rebuild their actual
            # allocation context and recheck the remaining expanded reads.
            # Rejection is monotonic, so this terminates after at most one pin
            # per deferred read, without changing the graph or the late proof.

    @staticmethod
    def _direct_read_candidates_priced(op, divisions, graph) -> bool:
        from torch_spyre._inductor.cost_model import transport_dma_cost_available
        from torch_spyre._inductor.dump_cost_model import extract_op_features
        from torch_spyre._inductor.scratchpad.division_menus import _work_slices

        if op is None or op.get_name() not in graph.nodes:
            return False
        is_lx = residency_symbols(graph)
        return all(
            transport_dma_cost_available(
                extract_op_features(op, _work_slices(op, division), is_lx=is_lx),
                _COST_PARAMS,
            )
            for division in divisions
        )

    @staticmethod
    def _pricing_operations(graph, solve_graph):
        from torch_spyre._inductor.read_copy_elision import (
            project_transport_read_copies,
        )

        splits = {n.name: [cd.splits for cd in n.menu] for n in solve_graph.get_nodes()}
        # A copy that may be relaid out keeps its read: the relayout reads it.
        return project_transport_read_copies(
            graph,
            splits,
            relayout_sources={
                e.src.name
                for n in solve_graph.get_nodes()
                for e in n.inputs
                if e.relayout_edge is not None
            },
        )

    def _division_map(
        self, graph: GraphLowering, *, allow_deferred_read_candidates: bool = False
    ) -> "_DivisionMap":
        """Per-op core-division candidates for the joint-division solve.

        Every op gets at least one ``CoreDivision`` so the slicing-match gate can
        constrain it. Pointwise / Reduction ops get the enumerated candidates;
        every other op falls back to its committed symbol-keyed division. No
        op-kind pre-filter -- residency is
        gated per buffer (``_residency_by_buf``) and by the solver, so ineligible
        ops still participate as producers/consumers in the match.

        Constant-offset writes, their storage aliases and their forward
        dependency closure keep their committed division. This protects offset
        address arithmetic even in HBM, where residency edges do not constrain
        division choices. Upstream producers remain free to optimize. See
        ``utils.ops_in_offset_mutation_component``. As with every fixed pin,
        an illegal committed division raises ``Unsupported`` rather than
        bypassing hard work-division constraints.

        Whatever the path, every candidate returned is within the ``sencores`` budget
        -- asserted here because nothing downstream re-checks it (issue #4387).
        """
        max_cores = config.sencores
        profiles, matmul_roles = _find_distinct_matmul_splits(graph.operations)
        # Tilings are offered per op (``_tiling_candidates``), but whether an op
        # sits inside a for_each_tile region, and who reads its output, are
        # graph-level facts.
        self._prescribed_ops: frozenset[str] = frozenset()
        self._readers_by_name: dict[str, list[Operation]] = {}
        if config.auto_coarse_tiling:
            from torch_spyre._inductor.scratchpad.coarse_tiling import (
                prescribed_regions,
            )

            self._prescribed_ops = frozenset(
                name
                for region in prescribed_regions(graph.operations)
                for name in region.names
            )
            for reader in graph.operations:
                for name in {dep.name for dep in reader.get_read_writes().reads}:
                    self._readers_by_name.setdefault(name, []).append(reader)

        # Ops pinned to their committed (work-division) division: each guard
        # detects a distinct wrong-code or scheduling hazard the joint solver
        # would hit by re-slicing the op, and all share the one remedy -- keep the
        # fixed division. A resolved user work_div hint takes the same remedy,
        # last so a hazard guard's reason is the one logged: work division
        # already committed the hint, and the pin is whole-op -- unhinted dims
        # keep their committed split of 1. The graph-level group sets are
        # loop-invariant, so build them once here rather than rescanning
        # graph.operations for every op.
        offset_mutation_ops = ops_in_offset_mutation_component(graph)
        layout_group_reason = _fused_layout_group_ops(
            graph,
            {
                KEEP_BY_INDEX_OP: "keep_by_index layout group",
                BATCH_MATMUL_FP8_OP: "fp8 matmul layout group",
            },
        )
        result = {}
        enumerated = set()
        for op in graph.operations:
            if self.committed:
                result[op.name] = [_fixed_core_division(op)]
                continue
            reason: Optional[str] = None
            if _is_cpu_host_buffer(op):
                reason = "cpu/host buffer"
            elif op.name in offset_mutation_ops:
                reason = "offset mutation forward closure"
            elif _is_windowed_pool(op):
                reason = "windowed pool"
            elif op.name in layout_group_reason:
                reason = layout_group_reason[op.name]
            elif _is_indirect_access_op(op):
                reason = "indirect access entry split"
            elif _reads_offset_slice(op):
                reason = "offset slice read"
            elif (
                is_restickify_op(op, graph)
                and hasattr(op, "_read_copy_elision_record")
                and not (
                    allow_deferred_read_candidates
                    and config.read_copy_elision
                    and isinstance(op._read_copy_elision_record, ReadCopyElisionRecord)
                )
            ):
                # The preparation path may provisionally enumerate ordinary
                # legal candidates, but retains them only after proving and
                # pricing the direct read in the actual allocation context.
                # Other callers and unrecognized records keep the fixed pin.
                reason = "deferred direct graph-input read"
            elif (
                not config.ignore_work_division_hints
                and isinstance(op, ComputedBuffer)
                and has_resolved_work_div_hint(op)
            ):
                reason = "user work_div hint"

            if reason is not None:
                divs = _legal_fixed_division(op, [_fixed_core_division(op)], reason)
            elif self.prune and isinstance(op, ComputedBuffer):
                divs = [
                    _core_division(op, splits)
                    for splits in _legal_split_options(
                        op, _enum_split_options(op, profiles, matmul_roles)
                    )
                ]
                if not divs:
                    divs = _legal_fixed_division(
                        op, [_fixed_core_division(op)], "empty pruned candidate set"
                    )
            else:
                divs, is_enumeration = self._enumerate_core_divisions(op, max_cores)
                if is_enumeration:
                    enumerated.add(op.name)
            if not divs:
                raise Unsupported(f"{op.name}: no legal core-division candidates.")
            # The core budget is an invariant of the MENU, not of its consumers: both
            # engines pin an op's split symbols to one enumerated candidate, so nothing
            # downstream re-checks the product -- and `_matmul_split_cost`'s own budget
            # guard no-ops on symbolic splits, scoring an over-budget division NEGATIVE.
            # A minimizing solve then finds it maximally attractive rather than
            # rejecting it (issue #4387). Only two of the three paths above take
            # `max_cores` -- `_legal_split_options` asks nothing about a core budget --
            # so check the menu itself, unconditionally: it is one product per candidate.
            over = [d for d in divs if d.cores_used > max_cores]
            assert not over, (
                f"{op.name}: enumerated core divisions over the {max_cores}-core "
                f"budget: "
                + ", ".join(f"{d.label} ({d.cores_used} cores)" for d in over)
            )
            result[op.name] = divs

        return _DivisionMap(result, enumerated)

    def _enumerate_core_divisions(
        self, op: Operation, max_cores: int
    ) -> tuple[list[CoreDivision], bool]:
        """Enumerate and deduplicate symbol-keyed candidates for one operation,
        with whether they are the *whole* legal cross product.

        Operations without an enumerable concrete iteration space retain their
        committed division, and say so (see :class:`_DivisionMap`).
        Deduplication uses local symbol names only within this operation;
        cross-operation compatibility is derived from ``PerCoreView`` instead.
        """
        fixed = [_fixed_core_division(op)]
        if not isinstance(op, ComputedBuffer) or not isinstance(
            op.data, (Pointwise, Reduction)
        ):
            return fixed, False
        points = self._point_menu(op, max_cores)
        if points is not None:
            return points, True
        try:
            candidates = enumerate_work_division_candidates(op, max_cores)
        except Unsupported as exc:
            return _legal_fixed_division(op, fixed, str(exc)), False
        cds: list[CoreDivision] = []
        seen: set[tuple] = set()
        options: list = []
        # Each tiling option gets its own division enumeration: the legal split
        # set is tiling-relative -- a tiled dim has fewer/other divisors and a
        # smaller per-core span. The untiled option reuses the already-enumerated
        # `candidates`; a non-empty option re-enumerates on the tiled frame.
        # Divisions are symbol-keyed and a tiling rescales indices without
        # renaming loop symbols, so both frames' keys are directly comparable.
        for tiling in self._tiling_candidates(op, max_cores):
            tile_splits: tuple[tuple[sympy.Symbol, int], ...] = ()
            if tiling.is_untiled:
                tiled_candidates = candidates
            else:
                from torch_spyre._inductor.scratchpad.coarse_tiling import (
                    try_resolve_tile_axis_loop_vars,
                )

                loop_vars, reason = try_resolve_tile_axis_loop_vars(op, tiling)
                if loop_vars is None:
                    logger.debug(
                        "skip tiling %s for %s: %s", tiling.label, op.name, reason
                    )
                    continue
                tile_splits = tuple(
                    (sym, axis.count) for sym, axis in zip(loop_vars, tiling.axes)
                )
                try:
                    tiled_candidates = enumerate_work_division_candidates(
                        op, max_cores, tiling=tiling
                    )
                except Unsupported as exc:
                    logger.debug(
                        "skip tiled division for %s under %s: %s",
                        op.name,
                        tiling.label,
                        exc,
                    )
                    continue
            from torch_spyre._inductor.scratchpad.division_space import DivisionOption
            from torch_spyre._inductor.work_division import (
                work_division_context_for_op,
            )

            context = work_division_context_for_op(op, max_cores, tiling)
            if context is not None and tiled_candidates:
                options.append(DivisionOption(tiling, tile_splits, context))
            for candidate in tiled_candidates:
                division = _core_division(op, candidate, tiling, tile_splits)
                key = (
                    tuple(
                        sorted(
                            division.splits.items(),
                            key=lambda item: str(item[0]),
                        )
                    ),
                    division.reduction_syms,
                    tiling,
                )
                if key not in seen:
                    seen.add(key)
                    cds.append(division)
        if cds:
            if options:
                from torch_spyre._inductor.scratchpad.division_space import (
                    DivisionSpace,
                )

                self.spaces[op.get_name()] = DivisionSpace(op, options, max_cores)
            return cds, True
        return _legal_fixed_division(op, fixed, "no enumerable candidate"), False

    def _point_menu(
        self, op: ComputedBuffer, max_cores: int
    ) -> Optional[list[CoreDivision]]:
        """The op's space, and its points as its menu, without enumerating the
        legal divisions (a Lazy work split): the Lazy strategy posts legality
        instead. ``None`` sends the op to the ordinary enumeration."""
        from torch_spyre._inductor.scratchpad.coarse_tiling import (
            try_resolve_tile_axis_loop_vars,
        )
        from torch_spyre._inductor.scratchpad.division_space import (
            DivisionOption,
            DivisionSpace,
        )
        from torch_spyre._inductor.scratchpad.lazy_strategy import MAX_POINTS
        from torch_spyre._inductor.work_division import work_division_context_for_op

        try:
            untiled = work_division_context_for_op(op, max_cores)
        except Unsupported:
            return None
        options = []
        for tiling in self._tiling_candidates(op, max_cores):
            tile_splits: tuple[tuple[sympy.Symbol, int], ...] = ()
            context = untiled
            if not tiling.is_untiled:
                loop_vars, _ = try_resolve_tile_axis_loop_vars(op, tiling)
                if loop_vars is None:
                    continue
                tile_splits = tuple(
                    (sym, axis.count) for sym, axis in zip(loop_vars, tiling.axes)
                )
                try:
                    context = work_division_context_for_op(op, max_cores, tiling)
                except Unsupported:
                    continue
            # Only an option with some legal division: the menu has no other.
            if context is not None and _has_legal_division(context):
                options.append(DivisionOption(tiling, tile_splits, context))
        if not options:
            return None
        space = DivisionSpace(op, options, max_cores)
        if len(space.points()) > MAX_POINTS:
            return None
        self.spaces[op.get_name()] = space
        self.point_menus.add(op.get_name())
        return [space.division(k, s) for k, s in space.points()]

    def _materialize(self, divisions: dict, name: str) -> None:
        """Replace a point menu by the legal menu, for a reader that indexes
        it (a pair table, read-copy pricing). Same order as the enumeration."""
        if name not in self.point_menus:
            return
        space = self.spaces[name]
        divisions[name] = [
            space.division(k, s)
            for k, s in space.points()
            if space.options[k].context.is_legal(s)
        ]
        self.point_menus.discard(name)

    def _tiling_candidates(self, op: Operation, max_cores: int) -> list[TileSpec]:
        """Coarse-tiling options to pair with ``op``'s divisions.

        Unless ``auto_coarse_tiling`` is on the only option is the untiled
        ``TileSpec``, so enumeration and every downstream plan stay
        bit-identical to today. When it does, the op is offered the
        output-axis tilings it could take, minus any whose per-core read span
        would still exceed the read-distance limit (``MAX_SPAN_BYTES``); the
        untiled option is dropped too when the op's own full-size read
        overflows, and an op with no fitting tiling raises ``Unsupported`` (see
        :func:`_drop_read_distance_violations`).

        Filtered by op kind:

        - **Restickify** offers nothing: no tiling form is correct
          post-stickification.
        - **Everything else** -- pointwise, reduction, *and matmul* -- offers only
          output-axis tilings (``is_clean``). That single filter, plus the
          enumerator's refusal to emit the stick (innermost) dim, already drops
          the numerically fragile forms for every op kind, so matmul needs no
          guard of its own: a matmul's K/reduction axis is not an output axis and
          is excluded (reduction tiling routes through the accumulator/combine
          path, ~2 orders off CPU -- see ``_mlp_case``), and its N/stick dim is
          never emitted. What remains for a matmul is row/M-axis output tiling,
          which is correct and backend-accepted (see
          ``test_hint_matmul_row_tiling``), so a discovered tiling on that axis
          is honored the same as for any other op.

        (The ``_resize_device_layout`` gap #3218 that rejects a tile-sized read
        copy is confined to the span-overflow path, where a matmul feeds a
        differently-shaped consumer -- a separate mechanism, still xfailed, not
        reached by the ordinary M-axis output tiling offered here.)

        An op already tiled (``loop_info`` set -- by the ``spyre_hint`` pass
        pre-stickification, a ``for_each_tile`` loop, or a prior apply) is left
        untouched, so the solve never re-tiles or un-tiles it, and so is every
        other op inside a ``for_each_tile`` region (``_prescribed_ops``): the
        user's loop covers it even where no level stamped it. The solve reads no
        hints of its own.
        """
        untiled = [TileSpec()]
        if self.suppress_tiling:
            return untiled
        if not config.auto_coarse_tiling:
            return untiled
        if getattr(op, "loop_info", None) is not None:
            return untiled
        if op.get_operation_name() in getattr(self, "_prescribed_ops", ()):
            return untiled
        if residency_rules._get_op_name(op) == "restickify":
            return untiled
        from torch_spyre._inductor.wsr.enumerate_tilings import enumerate_tile_options

        try:
            # enumerate_tile_options returns untiled first; the is_clean filter
            # keeps it and the output-only specs, preserving that order.
            readers = getattr(self, "_readers_by_name", {}).get(op.get_name(), ())
            options = [
                t for t in enumerate_tile_options(op, readers=readers) if t.is_clean
            ]
        except Unsupported:
            options = list(untiled)

        # Discovery offers the op tilings the solve is free to pick, but
        # the CP-SAT cost model carries no span term, so a discovered TileSpec is
        # never otherwise checked against the hardware read-distance limit
        # (MAX_SPAN_BYTES) the span-overflow planner enforces. Drop any candidate
        # whose per-core read span would still overflow that limit -- including
        # the untiled option when the op's own full-size read overflows -- so the
        # solve can never choose a tiling that reintroduces the very span
        # violation coarse tiling exists to prevent, and abort when none fit.
        return _drop_read_distance_violations(op, options, max_cores)

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
            op.name: residency_rules._op_inputs_good_for_lx_inplace(op)
            for op in graph.operations
        }
        lifetimes = calculate_liveness(graph)
        lifetime_start_overrides, lifetime_end_overrides = (
            counted_loop_lifetime_overrides(graph)
        )
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
            out_start = _handoff_child_start(
                buf_name, lifetimes, lifetime_start_overrides
            )
            out_ten_layout = out_layout.device_layout
            for input_buf in info["op_inputs"]:
                # Graph inputs / constants now appear in ``op_inputs`` but are not
                # solver buffers, so they can't be in-place aliasing parents (the
                # solve graph has no node for them to merge into). Skip them,
                # matching the base allocator's guard.
                if input_buf not in mem_usage or not lifetimes[input_buf]:
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
                if residency_rules._inplace_edge_ok(
                    child_pointwise_inputs=in_place_allowed[buf_name],
                    parent_name=input_buf,
                    child_device_layout=out_ten_layout,
                    parent_device_layout=in_ten_layout,
                    child_start=out_start,
                    parent_end=_handoff_parent_end(
                        input_buf, lifetimes, lifetime_end_overrides
                    ),
                    division_invariant=True,
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _residency_by_buf(
        self,
        graph: GraphLowering,
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
        drain_plans: Collection[str] = (),
    ) -> dict[str, Optional[str]]:
        """Per-buffer residency verdict: ``None`` if the buffer may be pinned in
        LX, else the reason it may not.

        Every buffer is handed to the solver so it participates in the slicing
        match, but participation is not residency
        (``residency_rules._buffer_residency_reason``). Whether a buffer's
        users slice it alike is the solver's relation to decide, over the
        divisions it may choose.
        """
        return residency_rules._residency_reasons(
            graph,
            list(mem_usage),
            lifetimes=lifetimes,
            drain_plans=drain_plans,
        )

    def _build_graph(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
        division_map: "_DivisionMap",
    ) -> SpyreOptimizationGraph:
        """The open-mode graph of ``graph``, its candidates on it.

        Each node is an op output or an input clone of the IR, sized by its
        *total* device footprint, with an edge to each producer it reads
        (``parent_proj``); the solver picks a division and divides by its
        ``output_partition``. Counted-loop lifetimes still filter unsafe
        in-place handoffs before the solver compares those total footprints.

        Each node holds its candidate divisions, and each edge its relations,
        relayout candidates and any pair table the builder fixes.
        """
        divisions = division_map.divisions
        lifetimes = calculate_liveness(graph)
        lifetime_start_overrides, lifetime_end_overrides = (
            counted_loop_lifetime_overrides(graph)
        )
        # A planned drain is a new post-solve op that reads the carry storage
        # after the loop, so a graph-output storage must stay live to the graph
        # exit.  This is inside the solve, so the extension only removes reuse
        # and stays priced -- it adds no cost term and no capacity.
        drain_plans = self.drain_plans
        if drain_plans:
            _drain_lifetime_end_overrides(
                lifetime_end_overrides, drain_plans, len(graph.operations)
            )
        # A collective has no division pair with any resident buffer, so its
        # edge to a carry would force the carry into HBM.  A planned drain is
        # what that collective reads instead, so the edge does not exist.
        drained_collectives = _drained_collectives(drain_plans)
        mem_usage = mem_usage_by_buf(graph)
        in_place = {} if in_place is None else in_place
        op_by_name = {op.name: op for op in graph.operations}
        graph_output_names = set(graph.get_output_names())

        prep_cache: dict = {}
        g = empty_graph(self.capacity)
        node_reads: list[tuple[str, list[str], list[str], tuple]] = []
        residency_by_buf = self._residency_by_buf(
            graph, mem_usage, lifetimes, drain_plans
        )

        # Resolve every compiler-tagged carry before constructing any buffer.
        # If its aliased update cannot be represented as a physical-ownership
        # edge, fail closed by leaving the storage in HBM.  The storage usually
        # precedes its update in graph order, so doing this up front avoids
        # discovering the malformed contract after its solver record is built.
        carry_update_edges: dict[str, ResidencyEdge] = {}
        for update_op in graph.operations:
            record = getattr(update_op, "_loop_carry_record", None)
            if not isinstance(record, LoopCarryRecord):
                continue
            if record.update_name != update_op.get_name():
                continue
            edge = self._loop_carry_update_edge(update_op, op_by_name, prep_cache)
            if edge is None:
                if residency_by_buf.get(record.storage_name) is None:
                    residency_by_buf[record.storage_name] = (
                        "loop carry update ownership unavailable"
                    )
                continue
            if residency_by_buf.get(record.storage_name) is None:
                carry_update_edges[record.update_name] = edge
        # Carry edges keep pair tables over menu indices.
        for update_name, edge in carry_update_edges.items():
            self._materialize(divisions, update_name)
            self._materialize(divisions, edge.buf_name)
        if carry_update_edges:
            for reader in graph.operations:
                edges = self._loop_carry_read_edges(
                    reader, carry_update_edges, prep_cache
                )
                if edges:
                    self._materialize(divisions, reader.get_name())
                    for storage in edges:
                        self._materialize(divisions, storage)

        input_clone_matches: dict[str, dict[str, list[tuple[int, int]]]] = {}
        input_clone_reads: dict[str, dict[str, CloneRead]] = {}
        # Consumer op name -> input clones for which it is the last reader, and so
        # may reuse the clone's LX slot in place (reverse-parent, #3212). Stays
        # empty unless cloning is on, making the reverse-parent block in the output
        # loop a no-op otherwise.
        last_consumer_clones: dict[str, list[str]] = {}
        if clone_at_graph_boundaries():
            buffer_users = get_buffer_users(graph)
            for input_name in self._eligible_clone_inputs(graph, lifetimes):
                consumers = [op for op in buffer_users.get(input_name, [])]
                divs, matches, reads = self._clone_divisions_and_matches(
                    input_name, consumers, divisions, prep_cache
                )
                # No division matched any consumer -> the clone has no valid core
                # division and could never reside, so don't hand an unplaceable
                # buffer to the solver (it would trip the >=1-division invariant).
                # The input simply stays in HBM, as it would uncloned.
                if not divs:
                    continue
                input_clone_matches[input_name] = matches
                input_clone_reads[input_name] = reads
                residency_by_buf[input_name] = None
                # Only the op at the clone's last-use tick both reads the clone and
                # writes its own output, so only it satisfies the single handoff
                # tick for reusing the clone's slot in place.
                last_use = lifetimes[input_name][-1]
                last_consumer_clones.setdefault(
                    graph.operations[last_use].name, []
                ).append(input_name)
                dev_layout = graph.get_buffer(input_name).layout.device_layout
                size = get_device_size_in_bytes(dev_layout)
                g.add_ir_node(
                    input_name,
                    nbytes=size,
                    uses=lifetimes[input_name],
                    first_use_is_read=True,
                    boundary=BufferType.Input,
                    start_override=lifetime_start_overrides.get(input_name),
                    end_override=lifetime_end_overrides.get(input_name),
                    menu=divs,
                )

        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]

            op = op_by_name.get(output_name)
            residency_reason = residency_by_buf[output_name]

            buf_divisions = divisions[output_name]
            # Drain plans can extend a parent's lifetime after the in-place
            # candidates were computed. Recheck adjacency with the same final
            # bounds handed to the solver so stale handoffs cannot reach it.
            parents = [
                parent
                for parent in in_place.get(output_name, [])
                if _handoff_parent_end(parent, lifetimes, lifetime_end_overrides)
                == _handoff_child_start(
                    output_name, lifetimes, lifetime_start_overrides
                )
            ]
            size = info["size"]  # total footprint; solver divides per chosen cd
            parent_proj = [
                parent
                for parent in info["op_inputs"]
                if drained_collectives.get(parent) != output_name
            ]
            residency_edges = self._parent_residency_edges(
                op, parent_proj, op_by_name, prep_cache, residency_by_buf
            )
            # The strategy derives these edges' relation from their
            # ResidencyEdge (an input clone's from its CloneRead); only
            # loop-carry edges, whose table is overridden below, are
            # materialized here, unless every edge gets a pair table.
            carry_edge = carry_update_edges.get(output_name)
            special = set(
                self._loop_carry_read_edges(op, carry_update_edges, prep_cache)
            )
            if carry_edge is not None:
                special.add(carry_edge.buf_name)
            tabled = {p: e for p, e in residency_edges.items() if p in special}
            cd_parent_matches = self._cd_parent_matches(
                tabled, buf_divisions, divisions
            )

            # A for_each_tile update writes through a MutationLayout whose
            # dependency is named after the update op, even though the bytes
            # belong to the persistent carry storage.  Model that write as a
            # producer -> consumer edge so a resident carry is legal only when
            # the solver chooses identical physical ownership for the initial
            # storage and every update.  A relayout cannot satisfy an in-place
            # write, so this edge deliberately has match pairs only.
            #
            # Folded into cd_parent_matches BEFORE _cd_parent_relayouts runs
            # (not after, as a prior version of this did): a sibling parent's
            # own relayout candidates get capped against cd_parent_matches
            # inside that call, and a cap that ran before the carry's matches
            # existed could prune away the one candidate that would have let
            # that sibling land on a division the carry already occupies for
            # free (cyang49's PR #5057 review). original_parent_proj (the
            # pre-carry snapshot) is what every "already a direct read"
            # membership check below is against, so a storage this same op
            # also reads directly keeps getting the intersected (update AND
            # read) match pairs either block alone would have produced,
            # independent of which block runs first -- and is what
            # _cd_parent_relayouts is called with: no carry-touched storage
            # name is ever added to parent_proj before that call, so it is
            # never offered as a relayout candidate at all (a relayout copy
            # taken before the update would be stale, whether or not the
            # storage happens to also be an ordinary read). Carry-touched
            # names are appended to the real parent_proj only afterward.
            original_parent_proj = list(parent_proj)
            carry_storage_names: list[str] = []
            carry_edge = carry_update_edges.get(output_name)
            if carry_edge is not None:
                storage_name = carry_edge.buf_name
                update_matches = carry_edge.match_pairs(
                    divisions[storage_name],
                    buf_divisions,
                )
                if storage_name in original_parent_proj:
                    # If the update also reads the carry directly, both that
                    # read and the aliased write must agree with its storage.
                    read_matches = cd_parent_matches.get(storage_name)
                    if read_matches is None:
                        update_matches = []
                    else:
                        update_set = set(update_matches)
                        update_matches = [p for p in read_matches if p in update_set]
                carry_storage_names.append(storage_name)
                cd_parent_matches[storage_name] = update_matches

            # A read of a carry update reads the storage's bytes, so it must
            # agree with the storage's ownership too; otherwise a resident
            # carry reaches a reader that slices it differently, and kernel
            # preparation demotes it (#4990). A relayout copy taken before the
            # update would be stale, so this edge also has match pairs only.
            for storage_name, read_edge in self._loop_carry_read_edges(
                op, carry_update_edges, prep_cache
            ).items():
                read_matches = read_edge.match_pairs(
                    divisions[storage_name],
                    buf_divisions,
                )
                # Also checks carry_storage_names, not just original_parent_proj:
                # a storage that is THIS op's own carry-update target (recorded
                # above) but not a direct read never enters original_parent_proj,
                # so without this a self-overlap -- two sequential mutations of
                # one carry, where this op both updates storage S and reads
                # another update that also writes S -- would overwrite the
                # update-edge's match_pairs with the unfiltered read-edge ones
                # instead of intersecting them, same loss of constraint the
                # direct-read branch above exists to prevent.
                if storage_name in original_parent_proj or storage_name in (
                    carry_storage_names
                ):
                    known = set(cd_parent_matches.get(storage_name, []))
                    read_matches = [p for p in read_matches if p in known]
                if storage_name not in carry_storage_names:
                    carry_storage_names.append(storage_name)
                cd_parent_matches[storage_name] = read_matches

            relayout_parent_names = [
                parent for parent in parent_proj if parent not in carry_storage_names
            ]
            clone_reads: dict[str, CloneRead] = {}
            for input_name in parent_proj:
                if input_name not in input_clone_matches:
                    continue
                read = input_clone_reads[input_name].get(output_name)
                if read is not None:
                    clone_reads[input_name] = read
                else:
                    cd_parent_matches[input_name] = input_clone_matches[input_name][
                        output_name
                    ]
            relayout_edges: dict[str, RelayoutEdge] = {}
            if (
                config.lx_planner_relayout
                and not config.ktir_emitter
                and op is not None
            ):
                relayout_edges = self._relayout_edges(
                    graph, op, relayout_parent_names, op_by_name, residency_by_buf
                )
            for storage_name in carry_storage_names:
                if storage_name not in parent_proj:
                    parent_proj.append(storage_name)

            for input_name in parent_proj:
                if input_name in input_clone_matches and input_name not in clone_reads:
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
                if clone_name in parents:
                    continue
                clone_layout = graph.get_buffer(clone_name).layout
                if (
                    op is None
                    or not hasattr(out_layout, "device_layout")
                    or not hasattr(clone_layout, "device_layout")
                ):
                    continue
                if residency_rules._inplace_edge_ok(
                    child_pointwise_inputs=residency_rules._op_inputs_good_for_lx_inplace(
                        op
                    ),
                    parent_name=clone_name,
                    child_device_layout=out_layout.device_layout,
                    parent_device_layout=clone_layout.device_layout,
                    child_start=_handoff_child_start(
                        output_name, lifetimes, lifetime_start_overrides
                    ),
                    parent_end=_handoff_parent_end(
                        clone_name, lifetimes, lifetime_end_overrides
                    ),
                    division_invariant=True,
                ):
                    parents.append(clone_name)

            g.add_ir_node(
                output_name,
                nbytes=size,
                uses=uses,
                # An op output is a computed buffer: ``uses[0]`` is the
                # producing write. (Only the input-clone loop above reads first.)
                first_use_is_read=False,
                boundary=BufferType.Output
                if output_name in graph_output_names
                else BufferType.Intermediate,
                residency_reason=residency_reason,
                start_override=lifetime_start_overrides.get(output_name),
                end_override=lifetime_end_overrides.get(output_name),
                op=op,
                menu=buf_divisions,
                space=self.spaces.get(output_name)
                if output_name in division_map.enumerated
                else None,
                point_menu=output_name in division_map.enumerated
                and output_name in self.point_menus,
            )
            # Every clone edge keeps its pair table beside its CloneRead.
            pairs = dict(cd_parent_matches)
            for input_name in parent_proj:
                if input_name in input_clone_matches:
                    pairs.setdefault(
                        input_name, input_clone_matches[input_name][output_name]
                    )
            relations = (pairs, residency_edges, clone_reads, relayout_edges)
            node_reads.append((output_name, parent_proj, parents, relations))
        for name, parent_proj, in_place_parents, relations in node_reads:
            node = g.nodes[name]
            g.add_reads(node, parent_proj, in_place_parents)
            pairs, residency_edges, clone_reads, relayout_edges = relations
            for dep in node.inputs:
                parent = dep.src.name
                dep.pairs = pairs.get(parent)
                dep.relation = clone_reads.get(parent, residency_edges.get(parent))
                dep.relayout_edge = relayout_edges.get(parent)
        g.link_execution_order()
        return g

    @staticmethod
    def _loop_carry_update_edge(
        update_op: Optional[Operation],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
    ) -> Optional[ResidencyEdge]:
        """Return the physical-ownership edge for an aliased carry update.

        Inductor names the mutation write after ``update_op`` rather than the
        storage it aliases.  Renaming that dependency lets the ordinary
        :class:`ResidencyEdge` machinery interpret its index against the carry
        storage's device layout.
        """
        if update_op is None:
            return None
        record = getattr(update_op, "_loop_carry_record", None)
        if not isinstance(record, LoopCarryRecord):
            return None
        if record.update_name != update_op.get_name():
            return None
        storage_op = op_by_name.get(record.storage_name)
        if storage_op is None:
            return None
        storage_write = next(
            (
                dep
                for dep in op_read_writes(storage_op).writes
                if dep.name == record.storage_name and isinstance(dep, MemoryDep)
            ),
            None,
        )
        update_write = next(
            (
                dep
                for dep in op_read_writes(update_op).writes
                if dep.name == record.update_name and isinstance(dep, MemoryDep)
            ),
            None,
        )
        if storage_write is None or update_write is None:
            return None
        return ResidencyEdge(
            buf_name=record.storage_name,
            parent_op=storage_op,
            consumer_op=update_op,
            write_dep=storage_write,
            read_deps=(update_write.rename({record.update_name: record.storage_name}),),
            prep_cache=prep_cache,
        )

    @staticmethod
    def _loop_carry_read_edges(
        consumer_op: Optional[Operation],
        carry_update_edges: dict[str, ResidencyEdge],
        prep_cache: dict,
    ) -> dict[str, ResidencyEdge]:
        """Storage-ownership edges for ``consumer_op``'s reads of carry updates.

        A carry update writes through the carry's storage, so a later read of the
        update's name reads the storage's bytes. The update itself is never an LX
        buffer, so its ordinary producer -> consumer edge is excluded, and without
        this edge nothing ties a resident storage's ownership to that reader. Each
        returned edge, keyed by storage name, compares the storage's write with the
        reader's accesses renamed onto the storage, as
        :meth:`_loop_carry_update_edge` does for the update's own write. A
        reader that reaches one storage through several reads gets one edge
        holding them all.
        """
        if consumer_op is None:
            return {}
        edges: dict[str, ResidencyEdge] = {}
        for dep in op_read_writes(consumer_op).reads:
            carry_edge = carry_update_edges.get(dep.name)
            if carry_edge is None or not isinstance(dep, MemoryDep):
                continue
            storage_name = carry_edge.buf_name
            read = dep.rename({dep.name: storage_name})
            edge = edges.get(storage_name)
            if edge is not None:
                edge.read_deps += (read,)
                continue
            edges[storage_name] = ResidencyEdge(
                buf_name=storage_name,
                parent_op=carry_edge.parent_op,
                consumer_op=consumer_op,
                write_dep=carry_edge.write_dep,
                read_deps=(read,),
                prep_cache=prep_cache,
            )
        return edges

    def _eligible_clone_inputs(
        self, graph: GraphLowering, lifetimes: dict[str, list[int]]
    ) -> list[str]:
        """Graph inputs eligible to be cloned into LX.

        ``residency_rules._input_residency_reason``; the clone's division is
        the solver's to choose, constrained by requiring each child to match
        it.
        """
        return [
            name
            for name in graph.graph_input_names
            if residency_rules._input_residency_reason(
                graph, name, lifetimes.get(name, [])
            )
            is None
        ]

    def _clone_divisions_and_matches(
        self,
        input_name: str,
        consumers: list[Operation],
        divisions: dict[str, list[CoreDivision]],
        prep_cache: dict,
    ) -> tuple[
        list[CoreDivision], dict[str, list[tuple[int, int]]], dict[str, CloneRead]
    ]:
        """Determine the core divisions which are applicable to the clone
        node based on the read per core views of the clone's consumers.

        A consumer that *broadcast-reads* the input -- its view covers fewer
        cores than its division runs, because it splits an axis the input does
        not have -- is skipped. There is no single-base LX broadcast, so the
        cores without a local copy would read stale scratchpad; the same
        predicate rejects the buffer in ``get_ncores_for_buffers``, and by then
        the division is committed and ``_post_solve`` can only raise. This is
        the whole broadcast defense on this edge: unlike a producer-consumer
        edge, a clone has no write-view of its own for
        ``ResidencyEdge.match_pairs`` to compare against, since the clone's
        view *is* the consumer's.

        The applicable core divisions are found and returned as a list of
        ``CoreDivision`` objects. The mapping such that the clone output
        per core view matches a given op's read per core view is returned
        where the mapping exists for each consumer. When solved the parent
        output per core view must match that of all consumers to be placed.
        This forces correctness at solve time rather than pre-pruning by
        finding the intersection of core divisions. The third result is each
        consumer's :class:`CloneRead`, the same relation per division for a
        strategy that does not use the pair tables.
        """
        clone_divs: list[CoreDivision] = []
        clone_views: list[PerCoreView] = []
        matches: dict[str, list[tuple[int, int]]] = {}
        reads: dict[str, CloneRead] = {}
        for consumer in consumers:
            cname = consumer.get_name()
            consumer_divs = divisions[cname]
            rw = op_read_writes(consumer)
            read_dep = next(
                (
                    r
                    for r in rw.reads
                    if r.name == input_name and isinstance(r, MemoryDep)
                ),
                None,
            )
            write = next((w for w in rw.writes if isinstance(w, MemoryDep)), None)
            if read_dep is None or write is None:
                matches[cname] = []
                continue
            reads[cname] = CloneRead(
                consumer, read_dep, input_name, clone_views, prep_cache
            )
            views = self._views_for_divs(
                consumer, read_dep, input_name, consumer_divs, prep_cache
            )
            pairs: list[tuple[int, int]] = []
            for j, (view, _, repr_ok) in enumerate(views):
                # ``num_cores`` is the division's core count; the split product
                # is what the view covers. They differ exactly on a broadcast
                # read (see the docstring).
                if (
                    not repr_ok
                    or math.prod(f for _, f in view.work_slice_dims) != view.num_cores
                ):
                    continue
                k = _clone_view_index(view, repr_ok, clone_views)
                if k is None:
                    cd = consumer_divs[j]
                    per_sym = cd.splits
                    # Project onto the input: a consumer split on an axis the
                    # input lacks (e.g. a matmul's free dim) does not slice the
                    # input, so it must not count toward the clone's cores.
                    # Otherwise the cores_used check below cannot reject a
                    # broadcast read.
                    read_syms = read_dep.index.free_symbols
                    k = len(clone_divs)
                    clone_divs.append(
                        CoreDivision(
                            splits={
                                sym: split
                                for sym, split in per_sym.items()
                                if split > 1 and sym in read_syms
                            }
                        )
                    )  # a clone op cannot have a reduction split
                    clone_views.append(view)
                # No core-count check here: ``same_partition`` already demands
                # equal core counts, and the guard above ties each view's count
                # to its division's.
                pairs.append((k, j))
            matches[cname] = pairs
        # An empty ``clone_divs`` means no consumer matched the clone under any
        # division, so it has no valid core division. Return it empty rather than
        # fabricating a whole-buffer fallback that no consumer matches: the caller
        # drops such a clone (it can never reside), keeping it out of the solver's
        # >=1-division invariant.
        return clone_divs, matches, reads

    @staticmethod
    def _parent_residency_edges(
        consumer_op: Optional[Operation],
        parent_names: list[str],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
        residency_by_buf: dict[str, Optional[str]],
    ) -> dict[str, ResidencyEdge]:
        """One :class:`ResidencyEdge` per divided producer this op reads, which
        decides both which candidate pairs are compatible and which producers
        are excluded from matching altogether. A producer with no entry can host
        no residency for this consumer at all."""
        if consumer_op is None:
            return {}
        consumer_reads = op_read_writes(consumer_op).reads
        edges: dict[str, ResidencyEdge] = {}
        for parent in parent_names:
            if parent not in op_by_name:
                continue
            edge = build_residency_edge(
                parent,
                op_by_name[parent],
                consumer_op,
                consumer_reads,
                residency_by_buf.get(parent, "not in graph"),
                prep_cache,
            )
            if edge is not None:
                edges[parent] = edge
        return edges

    def _cd_parent_matches(
        self,
        edges: dict[str, ResidencyEdge],
        consumer_divs: list[CoreDivision],
        divisions: dict[str, list[CoreDivision]],
    ) -> dict[str, list[tuple[int, int]]]:
        """Physical slicing-match pairs for each divided producer this op reads:
        the pair-table materialization of :meth:`_parent_residency_edges`.

        The pairing is the per-core-view comparison ``get_ncores_for_buffers``
        uses -- correct across reductions/reshapes, where a coeff-keyed
        signature would conflate axes.
        """
        return {
            parent: edge.match_pairs(
                divisions[parent],
                consumer_divs,
            )
            for parent, edge in edges.items()
        }

    def _relayout_edges(
        self,
        graph: GraphLowering,
        consumer_op: Operation,
        parent_names: list[str],
        op_by_name: dict[str, Operation],
        residency_by_buf: dict[str, Optional[str]],
    ) -> dict[str, RelayoutEdge]:
        """The edges into ``consumer_op`` that may read through a relayout
        copy: the division-independent edge gates (single non-indirect write,
        activation source, pointwise-or-matmul consumer, ...;
        ``solver_relayout_edge_context``)."""
        edges: dict[str, RelayoutEdge] = {}
        for parent in parent_names:
            if parent not in op_by_name:
                continue
            # The source must be residency-eligible: a relayout keeps it in LX.
            if residency_by_buf.get(parent, "not in graph") is not None:
                continue
            parent_op = op_by_name[parent]
            context = solver_relayout_edge_context(
                graph, parent_op, consumer_op, parent, op_by_name
            )
            if context is None:
                continue
            edges[parent] = RelayoutEdge(
                parent,
                parent_op,
                consumer_op,
                context,
                self._relayout_pair_costs,
                self.capacity,
            )
        return edges

    @staticmethod
    def _views_for_divs(op, dep, buf_name, divs: list[CoreDivision], prep_cache: dict):
        """Per-core views of ``buf_name`` for each candidate division of ``op``.

        The candidate-invariant prep is computed once and shared through
        ``prep_cache``, so cost scales with the op rather than its candidate
        count. A tiled division is reported unrepresentable: these views pair
        clones and relayout copies on core ownership alone, and a tiled
        candidate would also have to agree tile by tile.
        """
        return [
            _view_for_div(op, dep, buf_name, cd, prep_cache)
            if not cd.tile_splits
            else (PerCoreView((), (), num_cores=cd.cores_used), False, False)
            for cd in divs
        ]
