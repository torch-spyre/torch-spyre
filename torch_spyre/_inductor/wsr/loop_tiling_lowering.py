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

"""A small plan for tiling a group of ops along one loop, and its builder.

``LoopTilingPlan`` says which dim of each op a loop tiles and how many trips it
takes. It holds positions and a trip count and nothing else: no kind of tiling, no
consumers, no ranges, no per-read lists. Everything else is read from the graph at
the moment the plan is lowered, so the plan cannot go stale as other passes rewrite
the ops between planning and lowering.

``build_loop_tiling_plan`` resolves each op's hint to a position with the same
function the legacy planner uses (``resolve_level_positions``) and runs the same
refusals as ``plan_coarse_tile_groups``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation

from .. import config
from ..errors import Unsupported
from ..pass_utils import op_read_writes
from .coarse_tile import (
    _check_matmul_broadcast_batch_tiling,
    _group_reduction_tiled_levels_in_group,
    _plan_is_loop_invariant_at_reduction_levels,
    _planned_tile_extents_per_level,
    _reads_incomplete_reduction,
    _validate_planned_reduction_tiling,
    reject_nested_loops,
    resolve_level_positions,
)


@dataclass(frozen=True)
class TiledDim:
    """The dim of an op the loop tiles: a position in its output ranges, or in its
    reduction ranges when ``in_reduction``."""

    position: int
    in_reduction: bool


@dataclass(frozen=True)
class OpTiling:
    """One op of a group. ``tiled`` is None for an op the loop does not tile: it
    runs unchanged inside the loop."""

    op_name: str
    tiled: TiledDim | None


@dataclass(frozen=True)
class LoopTilingGroup:
    """The ops that share one loop: its index among the graph's loop groups, its
    trip count, and its ops in graph order."""

    index: int
    trip_count: int
    ops: tuple[OpTiling, ...]


@dataclass(frozen=True)
class LoopTilingPlan:
    groups: tuple[LoopTilingGroup, ...]


class _PartialResult(NamedTuple):
    """What ``_reads_incomplete_reduction`` asks of a planned sibling."""

    loop_tiled_reduction_dims: list[list[int]]


def build_loop_tiling_plan(
    operations: Sequence[Operation],
    groups: list[tuple],
    group_idx_offset: int = 0,
) -> LoopTilingPlan:
    """Plan ``groups`` (``(ops, [(hint_id, trip_count)])``) without touching the IR.

    ``op_name`` in the plan is ``get_operation_name()``. Raises ``Unsupported`` for
    what the legacy planner refuses: a nested group (one loop level only),
    reduction tiling while ``config.enable_reduction_tiling`` is off, a mixed or
    multi-dim reduction level, an op that reads a reduction before it is complete,
    and a matmul tiled across a broadcast batch.
    """
    reject_nested_loops(groups, "build_loop_tiling_plan")
    planned_groups: list[LoopTilingGroup] = []
    for group_idx, (group_ops, levels) in enumerate(groups):
        group_reduction_tiled_levels = _group_reduction_tiled_levels_in_group(
            group_ops, levels
        )
        group_op_names = {
            o.get_name() for o in group_ops if isinstance(o, ComputedBuffer)
        }
        siblings: dict[int, _PartialResult] = {}
        planned_ops: list[OpTiling] = []
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            tiled_dims, tiled_reduction_dims, _ = resolve_level_positions(op, levels)
            if any(tiled_reduction_dims) and not config.enable_reduction_tiling:
                raise Unsupported(
                    f"reduction-dim tiling for op {op.get_name()} "
                    "(disabled via enable_reduction_tiling)"
                )
            if any(tiled_reduction_dims):
                _validate_planned_reduction_tiling(op, tiled_dims, tiled_reduction_dims)
            if _plan_is_loop_invariant_at_reduction_levels(
                op, tiled_dims, group_reduction_tiled_levels
            ) and _reads_incomplete_reduction(
                op, group_ops, group_op_names, siblings, group_reduction_tiled_levels
            ):
                raise Unsupported(
                    f"partial reduction result consumed before accumulation "
                    f"is complete (op {op.get_name()} reads a per-tile "
                    f"partial result from the same loop group)"
                )
            rw = op_read_writes(op)
            read_deps = [d for d in rw.reads if isinstance(d, MemoryDep)]
            write_deps = [d for d in rw.writes if isinstance(d, MemoryDep)]
            _check_matmul_broadcast_batch_tiling(
                op,
                read_deps,
                write_deps,
                _planned_tile_extents_per_level(
                    op, tiled_dims, tiled_reduction_dims, levels
                ),
            )
            siblings[id(op)] = _PartialResult(tiled_reduction_dims)
            if tiled_dims[0]:
                tiled = TiledDim(tiled_dims[0][0], False)
            elif tiled_reduction_dims[0]:
                tiled = TiledDim(tiled_reduction_dims[0][0], True)
            else:
                tiled = None
            planned_ops.append(OpTiling(op.get_operation_name(), tiled))
        planned_groups.append(
            LoopTilingGroup(
                group_idx + group_idx_offset,
                int(levels[0][1]),
                tuple(planned_ops),
            )
        )
    return LoopTilingPlan(tuple(planned_groups))
