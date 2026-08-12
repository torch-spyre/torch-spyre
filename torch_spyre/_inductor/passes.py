# Copyright 2025 The Torch-Spyre Authors.
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


import inspect
import logging
import time
from typing import Optional, Any, Callable

import torch
import torch.fx.graph
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

try:
    # valid for torch 2.13
    from torch._inductor.custom_graph_pass import CustomSchedulerPass
except ImportError:
    # torch < 2.13 has no dedicated scheduler-pass base. Fall back to
    # CustomGraphPass
    CustomSchedulerPass = CustomGraphPass


from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Operation
from torch._inductor.scheduler import BaseSchedulerNode

from .logging_utils import get_inductor_logger
from .provenance import SpyreGraphTransformObserver, reset_provenance_warnings

from .padding import insert_bmm_padding
from .temp_passes import (
    bmm_unflatten_pass,
    decompose_addmm,
    mark_direct_unit_bmm_pass,
    mm_to_bmm_pass,
)
from .wsr.coarse_tile import validate_coarse_tile_groups
from .wsr.coarse_tile_span_overflow import span_overflow_groups
from .wsr.coarse_tile_hints import (
    hints_to_coarse_tile_groups,
    reorder_unhinted_interlopers,
)
from . import config
from .propagate_hints import (
    collect_spyre_hints,
    recover_spyre_hints,
)
from .wsr.propagate_named_dims import (
    propagate_named_dims,
    validate_named_dims,
    assign_dim_hints,
)
from .propagate_layouts import (
    propagate_mutation_layouts,
    propagate_spyre_tensor_layouts,
)
from .optimize_restickify import optimize_restickify_locations
from .insert_restickify import (
    finalize_layouts,
    insert_post_mutation_restickify,
    insert_restickify,
)
from .enforce_indirect_access_layout import enforce_indirect_access_layout
from .hbm_pool_planning import hbm_pool_planning
from .work_division import (
    span_reduction,
    work_distribution,
    cost_model_matmul_division,
)
from .pass_utils import format_operations
from .scratchpad.allocator import (
    scratchpad_planning,
)
from .fusion import spyre_fuse_nodes
from .scheduler import (
    align_lx_producer_loop_order,
    build_loop_scheduler_nodes,
    demote_incoherent_lx_buffers,
)
from .constants import DEVICE_NAME
from .deadcode_elimination import deadcode_elimination
from .dedup_constants import dedup_and_promote_constants
from .wsr.coarse_tile import coarse_tile_post_stickify, coarse_tile_pre_stickify
from .split_multi_ops import split_multi_ops, validate_ops


logger = get_inductor_logger("passes")


def _get_pass_name(pass_fn: Callable) -> str:
    """Get a human-readable name for a pass function."""
    if hasattr(pass_fn, "__name__"):
        return pass_fn.__name__
    if hasattr(pass_fn, "__func__"):
        return pass_fn.__func__.__name__
    return type(pass_fn).__name__


def _should_log_pass(pass_name: str) -> bool:
    """Check if per-pass logging is enabled for the given pass name."""
    log_passes_cfg = config.log_passes
    if not log_passes_cfg:
        return False
    if log_passes_cfg in ("all", "1"):
        return True
    selected = {s.strip() for s in log_passes_cfg.split(",")}
    return pass_name in selected


def _graph_has_spyre_device(graph: torch.fx.graph.Graph) -> bool:
    return any(
        isinstance(node, torch.fx.Node)
        and isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].device.type == DEVICE_NAME
        for node in graph.nodes
    )


def _nodes_have_spyre_device(nodes: list[BaseSchedulerNode]) -> bool:
    return any(
        node.get_device() is not None and node.get_device().type == DEVICE_NAME
        for node in nodes
    )


def _operations_have_spyre_device(operations: list[Operation]) -> bool:
    return any(
        op.get_device() is not None and op.get_device().type == DEVICE_NAME
        for op in operations
    )


def _uuid(passes: list[Callable]) -> Optional[Any]:
    # A pass is hashed by its own source file, unless it is a wrapper that
    # declares the real passes it runs via @_runs — then we hash those.
    files = [
        inspect.getfile(fn) for p in passes for fn in getattr(p, "_pass_sources", (p,))
    ]
    # Use dict.fromkeys instead of set for deterministic order.
    return get_hash_for_files(tuple(dict.fromkeys(files + [__file__])))


class _SpyreGraphPassPipeline(CustomGraphPass):
    """Pipeline over a post-grad FX graph, guarded by Spyre-device presence."""

    def __init__(self, passes: list[Callable]):
        self.passes = passes

    def _has_spyre_device(self, target: torch.fx.graph.Graph) -> bool:
        return _graph_has_spyre_device(target)

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        if not self._has_spyre_device(graph):
            return
        # FX-graph passes are already observed by upstream Inductor's
        # GraphTransformObserver (populates node.meta["from_node"]); no Spyre
        # observer is wrapped here.
        for p in self.passes:
            p(graph)

    def uuid(self) -> Any | None:
        return _uuid(self.passes)


class _SpyreNodePassPipeline(CustomSchedulerPass):
    """Pipeline over a list of scheduler nodes; each pass returns the new list."""

    def __init__(self, passes: list[Callable]):
        self.passes = passes

    def _has_spyre_device(self, target: list[BaseSchedulerNode]) -> bool:
        return _nodes_have_spyre_device(target)

    def __call__(self, target: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        if not self._has_spyre_device(target):
            return target
        # This pipeline is a per-compile entry point for the observed passes,
        # so clear the dedup here so each compile warns afresh.
        reset_provenance_warnings()
        for pass_fn in self.passes:
            name = _get_pass_name(pass_fn)
            observer = SpyreGraphTransformObserver(target, name, kind="node")
            with observer:
                target = pass_fn(target)
                # Reconcile the returned list while recursively inspecting the
                # underlying buffers through scheduler get_nodes().
                observer.target = target
        return target

    def uuid(self) -> Any | None:
        return _uuid(self.passes)


class CustomPreGradPasses(_SpyreGraphPassPipeline):
    """
    This inductor extension point enables Spyre-specific passes to run on the
    pre-grad FX graph.
    """

    def __init__(self):
        super().__init__([])


class CustomPrePasses(_SpyreGraphPassPipeline):
    """
    This inductor extension point enables Spyre-specific passes to run on the
    post-grad FX graph early in the sequence defined in `post_grad.post_grad_passes`.
    """

    def __init__(self):
        super().__init__([collect_spyre_hints])


class CustomPostPasses(_SpyreGraphPassPipeline):
    """
    This inductor extension point enables Spyre-specific passes to run on the
    post-grad FX graph late in the sequence defined in `post_grad.post_grad_passes`.
    """

    def __init__(self):
        super().__init__(
            [
                recover_spyre_hints,
                # Undo the post-grad re-fusion of add(input, mm(a, b)) back into
                # aten.addmm, so the resulting mul.Scalar alpha/beta nodes (whose
                # constants are materialized later by the LoopLevel IR multi-ops
                # pass) and the mm flow through the Spyre lowerings instead of
                # falling back to extern_kernels.addmm.
                decompose_addmm,
                mm_to_bmm_pass.apply,
                mark_direct_unit_bmm_pass,
                bmm_unflatten_pass.apply,
            ]
        )


class CustomPreFusionPasses(_SpyreNodePassPipeline):
    """
    This inductor extension point enables Spyre-specific passes to run over
    the graph of LoopLevelIR nodes immediately before Inductor's fusion pass runs.

    The list of nodes is guarenteed by the caller to be in topological order.
    The returned list of nodes must also be in topological order.
    """

    # build_loop_scheduler_nodes runs unconditionally: it is a no-op when
    # no ops carry loop_group_id attributes (i.e. no spyre_hint annotations).
    # Running here (before Inductor's fusion pass) ensures CountedLoopSchedulerNodes
    # are visible to SuperDSCScheduling.can_fuse_vertical/horizontal (which return
    # False), so loop groups survive Inductor fusion intact.
    def __init__(self):
        # align_lx_producer_loop_order runs before build_loop_scheduler_nodes so
        # it still sees plain SchedulerNodes (the only kind that can reorder
        # their loops) rather than CountedLoopSchedulerNode wrappers.
        super().__init__(
            [
                propagate_mutation_layouts,
                align_lx_producer_loop_order,
                build_loop_scheduler_nodes,
            ]
        )


class CustomPostFusionPasses(_SpyreNodePassPipeline):
    """
    This inductor extension point enables Spyre-specific passes to run over
    the graph of LoopLevelIR nodes immediately after Inductor's fusion pass runs.

    The list of nodes is guarenteed by the caller to be in topological order.
    The returned list of nodes must also be in topological order.
    """

    def __init__(self):
        # demote_incoherent_lx_buffers runs first: it re-checks LX core->slice
        # coherence now that loop orders are final, and anything it demotes must
        # still be visible to hbm_pool_planning as an unclaimed intermediate.
        # hbm_pool_planning runs after spyre_fuse_nodes so it can compute
        # bundle-scoped live ranges.
        super().__init__(
            [demote_incoherent_lx_buffers, spyre_fuse_nodes, hbm_pool_planning]
        )


# Several pre-scheduling steps are config-gated or need arguments beyond the
# graph (coarse-tile groups, k-fast ops, a scratchpad allocator). They are
# wrapped below as uniform Callable[[GraphLowering], None] so the pipeline can
# run every step with a single ``pass_(graph)`` call. Each wrapper is tagged
# with @_runs(...) so uuid() still keys the Inductor cache on the source files
# of the real passes it invokes, not just this module.


def _runs(*passes: Callable) -> Callable[[Callable], Callable]:
    """Tag a wrapper with the underlying passes it invokes (for uuid keying)."""

    def annotate(wrapper: Callable) -> Callable:
        setattr(wrapper, "_pass_sources", passes)
        return wrapper

    return annotate


@_runs(
    reorder_unhinted_interlopers,
)
def _maybe_reorder_unhinted_interlopers(graph: GraphLowering) -> None:
    """Move unhinted ComputedBuffer ops that interrupt hint-group runs."""
    if config.ignore_wsr_hints:
        return
    reorder_unhinted_interlopers(graph)


@_runs(
    hints_to_coarse_tile_groups,
    validate_coarse_tile_groups,
    coarse_tile_pre_stickify,
)
def _maybe_coarse_tile_hints(graph: GraphLowering) -> None:
    """Hint-driven coarse tiling only.  Runs PRE-stickification.

    span_overflow_groups is intentionally absent: it requires FixedTiledLayout
    (device_layout) and must run post-stickification.
    """
    if config.ignore_wsr_hints:
        return
    groups = hints_to_coarse_tile_groups(graph)
    if not groups:
        return
    op_order = {id(op): idx for idx, op in enumerate(graph.operations)}
    groups.sort(key=lambda group: op_order.get(id(group[0][0]), len(op_order)))
    validate_coarse_tile_groups(groups)
    coarse_tile_pre_stickify(graph, groups=groups)


@_runs(
    span_overflow_groups,
    validate_coarse_tile_groups,
    coarse_tile_post_stickify,
)
def _maybe_coarse_tile_span_overflow(graph: GraphLowering) -> None:
    """Span-overflow coarse tiling only.  Runs POST-stickification.

    Requires FixedTiledLayout (device_layout) on all ops.
    hint-driven groups (hints_to_coarse_tile_groups) are intentionally
    absent: they have already run pre-stickification.

    Uses coarse_tile_post_stickify: layout propagation has already
    committed every op's device layout by this point, so a read-copy here
    would only produce an HBM-to-HBM copy with no layout-reconciliation
    benefit.
    """
    if config.ignore_span_overflow_hints:
        return
    groups, dim_hint_assignments = span_overflow_groups(graph)
    if not groups:
        return
    # span_overflow_groups is a pure planning step: it decides each op's
    # dim_hints but does not set them.  Apply them now, before
    # validate_coarse_tile_groups/coarse_tile_post_stickify run, since
    # dim_hints is an input those consume (via plan_coarse_tile_groups's
    # hint lookups), not something they produce.
    for op, dim_hints in dim_hint_assignments:
        op.dim_hints = dim_hints  # type: ignore[attr-defined]
    # Compute offset to avoid loop_group_id collision with any hint-driven
    # groups already stamped by _maybe_coarse_tile_hints.
    # E.g. if hints used groups 0 and 1, span-overflow groups start at 2.
    used_ids = [
        op.loop_info.loop_group_id[0]
        for op in graph.operations
        if hasattr(op, "loop_info") and op.loop_info is not None
    ]
    group_idx_offset = max(used_ids, default=-1) + 1
    op_order = {id(op): idx for idx, op in enumerate(graph.operations)}
    groups.sort(key=lambda group: op_order.get(id(group[0][0]), len(op_order)))
    validate_coarse_tile_groups(groups)
    coarse_tile_post_stickify(
        graph,
        groups=groups,
        group_idx_offset=group_idx_offset,
    )


@_runs(cost_model_matmul_division, work_distribution)
def _distribute_work(graph: GraphLowering) -> None:
    # cost_model_matmul_division claims a subset of ops; work_distribution skips
    # those so every op is divided by exactly one of the two passes.
    preassigned_ops = cost_model_matmul_division(graph)
    work_distribution(graph, preassigned_ops)


@_runs(scratchpad_planning)
def _maybe_scratchpad_planning(graph: GraphLowering) -> None:
    if not config.lx_planning:
        return
    # The allocator (and its layout solver) is selected from config by
    # scratchpad_planning -> select_allocator; no allocator wiring here.
    scratchpad_planning(graph)


class CustomPreSchedulingPasses:
    """
    Spyre-specific passes that run on the GraphLowering immediately before the
    Scheduler is constructed (via the _update_scheduler monkey-patch).

    Operations (``graph.operations``) are in topological order (guaranteed by
    GraphLowering). Each pass takes the GraphLowering so it can read and mutate
    ``graph.operations`` directly.

    :meth:`get_passes` is the single ordered pipeline: plain passes appear
    directly, while config-gated or parameterized steps are wrapped (see the
    ``_maybe_*`` / ``_distribute_work`` helpers above) so every entry is a
    uniform ``Callable[[GraphLowering], None]``. :meth:`__call__` just runs them
    in order, and the inherited :meth:`uuid` keys the cache on their sources.
    """

    def __init__(self):
        self.passes = [
            deadcode_elimination,
            #
            # Working Set Reduction (hint-driven, pre-stickification)
            # These passes only need host-side FixedLayout (size/stride) and
            # loop variable ranges.  Running before stickification means
            # _divide_ranges does not call _resize_device_layout: stickification
            # computes the correct SpyreTensorLayout from the already-divided
            # ranges.  This also dissolves the insert_restickify→hint cross-phase
            # contract (issue #3135).
            propagate_named_dims,
            validate_named_dims,
            assign_dim_hints,
            _maybe_reorder_unhinted_interlopers,
            _maybe_coarse_tile_hints,
            #
            # Tensor Layout (Stickification)
            split_multi_ops,
            propagate_spyre_tensor_layouts,
            validate_ops,
            optimize_restickify_locations,
            finalize_layouts,
            insert_restickify,
            enforce_indirect_access_layout,
            insert_post_mutation_restickify,
            insert_bmm_padding,
            #
            dedup_and_promote_constants,
            #
            # Working Set Reduction (device-layout-aware, post-stickification)
            # These passes require FixedTiledLayout.device_layout (device_size,
            # stride_map, elems_per_stick) for physical span arithmetic.
            _maybe_coarse_tile_span_overflow,
            #
            # Core Division
            span_reduction,
            _distribute_work,
            #
            # LX Planning
            _maybe_scratchpad_planning,
        ]

    def __call__(self, graph: GraphLowering) -> None:
        if not _operations_have_spyre_device(graph.operations):
            return

        # This pipeline is a per-compile entry point for the observed passes,
        # so clear the dedup here so each compile warns afresh.
        reset_provenance_warnings()

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "BEFORE PRE-SCHEDULING\n%s", format_operations(graph.operations)
            )

        for pass_fn in self.passes:
            pass_name = _get_pass_name(pass_fn)
            # `graph` is the same object throughout -- passes mutate
            # `graph.operations` in place -- so before/after reconciliation
            # is exact here.
            with SpyreGraphTransformObserver(graph, pass_name, kind="graphlowering"):
                t0 = time.perf_counter()
                pass_fn(graph)
                elapsed_ms = (time.perf_counter() - t0) * 1000

            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "elapsed %5dms  %s",
                    elapsed_ms,
                    pass_name,
                )
            if logger.isEnabledFor(logging.DEBUG) and _should_log_pass(pass_name):
                logger.debug(
                    "AFTER %s\n%s", pass_name, format_operations(graph.operations)
                )

        if logger.isEnabledFor(logging.INFO):
            logger.info("AFTER PRE-SCHEDULING\n%s", format_operations(graph.operations))

    def uuid(self) -> Any | None:
        return _uuid(self.passes)
