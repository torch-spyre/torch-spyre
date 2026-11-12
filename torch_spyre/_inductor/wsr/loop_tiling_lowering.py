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

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V

from .. import config
from ..errors import Unsupported
from ..loop_info import LoopShim
from ..pass_utils import op_read_writes
from .coarse_tile import (
    _allocate_full_buffer,
    _apply_work_div_symbol_remap,
    _check_matmul_broadcast_batch_tiling,
    _find_outside_consumers_planned,
    _group_reduction_tiled_levels_in_group,
    _patch_consumers,
    _patch_graph_outputs,
    _plan_is_loop_invariant_at_reduction_levels,
    _planned_full_device_layout,
    _planned_tile_extents_per_level,
    _reads_buffer,
    _reads_incomplete_reduction,
    _validate_contiguous,
    _validate_planned_reduction_tiling,
    reject_nested_loops,
    resolve_level_positions,
)
from .loop_tiling_common import (
    _divide_ranges,
    _loop_var_hinted_ranges,
    _patch_retiled_load_indexes,
    _RetiledBufferInfo,
)
from .span_overflow_loop_terms import (
    _OpEdit,
    _apply_edit,
    _taken_symbols,
    access_step,
    mint_induction,
)

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def _outer_key(op: Any) -> int | None:
    group = getattr(getattr(op, "loop_info", None), "loop_group_id", None)
    return group[0] if group else None


def _extent(op: ComputedBuffer, position: int, trip: sympy.Integer) -> sympy.Integer:
    """The tile's extent along ``position``, from the undivided range."""
    size = op.data.ranges[position]
    if not isinstance(size, sympy.Integer) or size % trip != 0:
        raise Unsupported(
            f"lower_loop_tiling: op {op.get_name()!r} loop var d{position} range "
            f"{size} is not divisible by loop_count {trip}.  All tiled dimensions "
            "must be evenly divisible by the loop trip count."
        )
    return sympy.Integer(int(size) // int(trip))


def _tile_step(
    op: ComputedBuffer, position: int, dep: MemoryDep, extent: sympy.Integer
) -> sympy.Expr:
    """The host advance of one access per trip: the coefficient of the tiled dim's
    index symbol times the tile extent, read from the access BEFORE the divide. A
    dim that will be squeezed out (tile extent 1) is no special case: its symbol is
    still in the undivided index. An access that does not mention the symbol (a
    broadcast read) does not advance."""
    from ..spyre_kernel import SpyreKernel

    symbol = SpyreKernel._host_dim_to_index_symbol(op, position)
    if symbol not in dep.index.free_symbols:
        return sympy.S.Zero
    return access_step(dep.index, symbol, extent, sympy.S.Zero)


@dataclass
class _Facts:
    """What the lowering reads from the live graph about one member, before it
    changes anything in the group."""

    op: ComputedBuffer
    tiled: TiledDim | None
    kind: str  # "copy_out" or "loop_internal"
    read_steps: dict[str, sympy.Expr]
    write_step: sympy.Expr | None
    full_ranges: list[sympy.Expr]
    full_strides: tuple
    full_device_layout: Any
    is_graph_output: bool


def _facts(
    group: LoopTilingGroup,
    members: list[ComputedBuffer],
    operations: list[Operation],
) -> list[_Facts]:
    trip = sympy.Integer(group.trip_count)
    tiling = {o.op_name: o.tiled for o in group.ops}
    key_of: dict[str, int] = {}
    for op in operations:
        key = _outer_key(op)
        if isinstance(op, ComputedBuffer) and key is not None:
            key_of[op.get_name()] = key
    for member in members:
        key_of[member.get_name()] = group.index
    scratch_names = {
        m.get_name() for m in members if tiling[m.get_operation_name()] is not None
    }
    facts: list[_Facts] = []
    for member in members:
        name = member.get_operation_name()
        tiled = tiling[name]
        if not isinstance(member.data, Pointwise) or (tiled and tiled.in_reduction):
            raise Unsupported(
                f"lower_loop_tiling: {name} ({type(member.data).__name__}"
                f"{', reduction dim' if tiled and tiled.in_reduction else ''}) is "
                "not lowered by the plan yet."
            )
        if isinstance(member.layout, MutationLayoutSHOULDREMOVE):
            raise Unsupported(
                f"lower_loop_tiling: {name} mutates a buffer; the plan does not "
                "lower a mutation yet."
            )
        buf_name = member.get_name()
        consumers, is_graph_output = (
            _find_outside_consumers_planned(
                buf_name, (group.index,), operations, key_of
            )
            if tiled
            else ([], False)
        )
        kind = (
            "copy_out" if tiled and (consumers or is_graph_output) else "loop_internal"
        )
        read_steps: dict[str, sympy.Expr] = {}
        write_step = None
        full_ranges = list(member.data.ranges)
        for d, loop_var_range in _loop_var_hinted_ranges(member).items():
            full_ranges[d] = full_ranges[d] * loop_var_range
        if tiled:
            extent = _extent(member, tiled.position, trip)
            rw = op_read_writes(member)
            for dep in (d for d in rw.reads if isinstance(d, MemoryDep)):
                # A read of a tiled member's own scratch does not advance: the
                # buffer is rewritten every trip.
                step = (
                    sympy.S.Zero
                    if dep.name in scratch_names
                    else _tile_step(member, tiled.position, dep, extent)
                )
                if read_steps.setdefault(dep.name, step) != step:
                    raise Unsupported(
                        f"lower_loop_tiling: {name} reads {dep.name} twice with "
                        "different advances."
                    )
            writes = [d for d in rw.writes if isinstance(d, MemoryDep)]
            if kind == "copy_out":
                if not writes:
                    raise Unsupported(f"lower_loop_tiling: {name} has no write.")
                write_step = _tile_step(member, tiled.position, writes[0], extent)
                if write_step == 0:
                    raise Unsupported(
                        f"lower_loop_tiling: the tiled dim of {name} is not in "
                        "its write index."
                    )
        facts.append(
            _Facts(
                member,
                tiled,
                kind,
                {n: t for n, t in read_steps.items() if t != 0},
                write_step,
                full_ranges,
                tuple(member.layout.stride),
                _planned_full_device_layout(member, full_ranges),
                is_graph_output,
            )
        )
    return facts


def _insert_copy_out(
    tiled_op: ComputedBuffer,
    full_buf: ComputedBuffer,
    operations: list[Operation],
    group: LoopTilingGroup,
    position: int,
) -> ComputedBuffer:
    """The op that drains ``tiled_op``'s tile into ``full_buf``, inside the loop.

    It reads the tile (scratch, rewritten every trip, so no advance) and writes
    ``full_buf`` through a ``MutationLayoutSHOULDREMOVE`` layout. Its write advances;
    the term is added with the others, after the bodies are final."""
    copy_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=tiled_op.make_loader(),
        ranges=list(tiled_op.data.ranges),
    )
    copy_name = V.graph.qualify_name(f"coarse_tile_copy_{tiled_op.get_name()}")
    copy_buf = ComputedBuffer(
        name=copy_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf))),
        data=copy_data,
    )
    copy_buf.origins = tiled_op.origins
    copy_buf.operation_name = copy_name
    copy_buf.loop_info = _shim(group, position)  # type: ignore[attr-defined]
    V.graph.name_to_buffer[copy_name] = copy_buf
    operations.insert(operations.index(tiled_op) + 1, copy_buf)
    return copy_buf


def _shim(group: LoopTilingGroup, position: int | None) -> LoopShim:
    return LoopShim(
        loop_group_id=(group.index,),
        loop_count=[sympy.Integer(group.trip_count)],
        loop_tiled_dims=[[] if position is None else [position]],
        loop_tiled_reduction_dims=[[]],
    )


def _resolve(operations: list[Operation], names: list[str]) -> list[ComputedBuffer]:
    by_name = {op.get_operation_name(): op for op in operations}
    return [by_name[n] for n in names]  # type: ignore[misc]


def _lower_group(graph: Any, group: LoopTilingGroup, induction: sympy.Symbol) -> None:
    operations = graph.operations
    trip = sympy.Integer(group.trip_count)
    names = [o.op_name for o in group.ops]
    members = _resolve(operations, names)
    already = [
        m.get_operation_name()
        for m in members
        if getattr(m, "loop_info", None) is not None
    ]
    if already:
        raise Unsupported(
            f"lower_loop_tiling: group {group.index} would overwrite the existing "
            f"loop_info of {', '.join(already)}."
        )
    _validate_contiguous(
        members,
        {op.get_operation_name(): i for i, op in enumerate(operations)},
        (group.index,),
    )
    facts = _facts(group, members, operations)

    # Divide the tiled ops and stamp the shim on every member.
    retiled: dict[str, _RetiledBufferInfo] = {}
    for fact in facts:
        if fact.tiled is not None:
            result = _divide_ranges(fact.op, trip, [fact.tiled.position])
            _apply_work_div_symbol_remap(fact.op, result.symbol_remap)
            if result.retiled_info is not None:
                retiled[fact.op.get_name()] = result.retiled_info
        fact.op.loop_info = _shim(  # type: ignore[attr-defined]
            group, None if fact.tiled is None else fact.tiled.position
        )

    # Drain every tile that something outside the loop reads into a full buffer.
    copies: dict[str, tuple[str, sympy.Expr]] = {}
    for fact in facts:
        if fact.kind != "copy_out" or fact.tiled is None:
            continue
        op = fact.op
        buf_name = op.get_name()
        outside = [
            o
            for o in operations
            if isinstance(o, ComputedBuffer)
            and o is not op
            and _reads_buffer(o, buf_name)
            and _outer_key(o) != group.index
        ]
        start = next(
            i
            for i, o in enumerate(operations)
            if isinstance(o, ComputedBuffer) and _outer_key(o) == group.index
        )
        full_buf = _allocate_full_buffer(
            op,
            fact.full_ranges,
            fact.full_strides,
            operations,
            start,
            device_layout=fact.full_device_layout,
        )
        bound_layout = op.layout
        copy_buf = _insert_copy_out(
            op, full_buf, operations, group, fact.tiled.position
        )
        assert fact.write_step is not None
        copies[copy_buf.get_operation_name()] = (buf_name, fact.write_step)
        _patch_consumers(
            outside,
            buf_name,
            full_buf.get_name(),
            operations,
            producer_layout=bound_layout,
            read_layout=full_buf.layout,
        )
        if fact.is_graph_output:
            _patch_graph_outputs(buf_name, full_buf)

    # An exact-loop consumer of a retiled buffer holds a stale stride.
    group_ops = _resolve(operations, names)
    _patch_retiled_load_indexes((group.index,), group_ops, retiled, operations)

    # The bodies are final: give every op the loop variable, and the advancing
    # accesses their term.
    steps = {fact.op.get_operation_name(): fact.read_steps for fact in facts}
    for member in _resolve(operations, names):
        _apply_edit(
            _OpEdit(member, trip, steps[member.get_operation_name()], None), induction
        )
    for copy in _resolve(operations, list(copies)):
        _apply_edit(
            _OpEdit(copy, trip, {}, copies[copy.get_operation_name()][1]), induction
        )


def lower_loop_tiling(graph: Any, plan: LoopTilingPlan) -> dict[int, sympy.Symbol]:
    """Tile every group of ``plan``: divide the tiled ops, drain each tile that
    something outside the loop reads into a full buffer, and give every op the loop
    variable and every advancing access its term.

    Returns ``{group index: loop variable}`` for ``request_loop_nodes``. Each op
    carries a ``LoopShim``, never a ``CoarseTileInfo``: the advances live in the
    index. Everything the lowering needs (consumers, ranges, strides, steps) is read
    from the graph as it stands when the group is lowered, never from the plan.
    """
    taken = _taken_symbols(graph, list(graph.operations))
    inductions: dict[int, sympy.Symbol] = {}
    for group in plan.groups:
        induction = mint_induction(taken)
        taken.append(induction)
        _lower_group(graph, group, induction)
        inductions[group.index] = induction
    return inductions
