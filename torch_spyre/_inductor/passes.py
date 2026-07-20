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
import io
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
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.scheduler import BaseSchedulerNode

from .logging_utils import get_inductor_logger

from .padding import insert_bmm_padding
from .temp_passes import (
    bmm_unflatten_pass,
    mark_direct_unit_bmm_pass,
    mm_to_bmm_pass,
)
from .coarse_tile import (
    hints_to_coarse_tile_groups,
    reorder_unhinted_interlopers,
    span_overflow_groups,
)
from . import config
from .propagate_hints import (
    collect_spyre_hints,
    recover_spyre_hints,
)
from .propagate_named_dims import propagate_named_dims, assign_dim_hints
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
from .memory_planning import memory_planning
from .work_division import (
    span_reduction,
    work_distribution,
    cost_model_matmul_division,
)
from .pass_utils import apply_splits_from_index_coeff, iteration_space_from_op
from .scratchpad.allocator import (
    scratchpad_planning,
)
from .fusion import spyre_fuse_nodes
from .scheduler import build_loop_scheduler_nodes
from .constants import DEVICE_NAME
from .deadcode_elimination import deadcode_elimination
from .dedup_constants import dedup_and_promote_constants
from .coarse_tile import coarse_tile
from .split_multi_ops import split_multi_ops, validate_ops


logger = get_inductor_logger("passes")


def _format_operations(operations: list[Operation]) -> str:
    buf = io.StringIO()
    for op in operations:
        buf.write(f"{op.get_operation_name()}: {type(op).__name__}")
        if isinstance(op, ComputedBuffer):
            buf.write(f"\n  layout={op.layout}")
            if allocation := getattr(op.layout, "allocation", None):
                buf.write(f"\n  allocation={allocation}")
            if splits := getattr(op, "op_it_space_splits", None):
                rw = op.get_read_writes()
                write_index = next(iter(rw.writes)).index
                read_index = next((d.index for d in rw.reads), write_index)
                it_space = iteration_space_from_op(op)
                readable_splits = apply_splits_from_index_coeff(
                    splits, write_index, read_index, it_space
                )
                buf.write(f"\n  op_it_space_splits={readable_splits}")
            if dim_hints := getattr(op, "dim_hints", None):
                buf.write(f"\n  dim_hints={dim_hints}")
            if loop_info := getattr(op, "loop_info", None):
                buf.write(f"\n  loop_info={loop_info}")
            buf.write(f"\n  {op.data}")
        buf.write("\n\n")
    return buf.getvalue()


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
        for pass_fn in self.passes:
            target = pass_fn(target)
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
        super().__init__([propagate_mutation_layouts, build_loop_scheduler_nodes])


class CustomPostFusionPasses(_SpyreNodePassPipeline):
    """
    This inductor extension point enables Spyre-specific passes to run over
    the graph of LoopLevelIR nodes immediately after Inductor's fusion pass runs.

    The list of nodes is guarenteed by the caller to be in topological order.
    The returned list of nodes must also be in topological order.
    """

    def __init__(self):
        super().__init__([memory_planning, spyre_fuse_nodes])


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
    hints_to_coarse_tile_groups,
    span_overflow_groups,
    coarse_tile,
)
def _maybe_coarse_tile(graph: GraphLowering) -> None:
    groups = []
    if not config.ignore_wsr_hints:
        reorder_unhinted_interlopers(graph)
        groups += hints_to_coarse_tile_groups(graph)
    if not config.ignore_span_overflow_hints:
        groups += span_overflow_groups(graph)
    if groups:
        op_order = {id(op): idx for idx, op in enumerate(graph.operations)}
        groups.sort(key=lambda group: op_order.get(id(group[0][0]), len(op_order)))
        coarse_tile(graph, groups=groups)


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
            # Tensor Layout (Stickification)
            split_multi_ops,
            propagate_spyre_tensor_layouts,
            validate_ops,
            optimize_restickify_locations,
            finalize_layouts,
            insert_restickify,
            insert_post_mutation_restickify,
            insert_bmm_padding,
            #
            dedup_and_promote_constants,
            #
            # Working Set Reduction
            propagate_named_dims,
            assign_dim_hints,
            _maybe_coarse_tile,
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

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "BEFORE PRE-SCHEDULING\n%s", _format_operations(graph.operations)
            )

        for pass_fn in self.passes:
            t0 = time.perf_counter()
            pass_fn(graph)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "elapsed %5dms  %s",
                    (time.perf_counter() - t0) * 1000,
                    _get_pass_name(pass_fn),
                )

            pass_name = _get_pass_name(pass_fn)
            if logger.isEnabledFor(logging.DEBUG) and _should_log_pass(pass_name):
                logger.debug(
                    "AFTER %s\n%s", pass_name, _format_operations(graph.operations)
                )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "AFTER PRE-SCHEDULING\n%s", _format_operations(graph.operations)
            )

    def uuid(self) -> Any | None:
        return _uuid(self.passes)
