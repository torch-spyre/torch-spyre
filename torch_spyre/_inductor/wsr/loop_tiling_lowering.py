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
function the ``for_each_tile`` lowering uses (``resolve_level_positions``) and
refuses what the lowering does not express: a nest, reduction tiling while
``config.enable_reduction_tiling`` is off, and the cases named in its docstring.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import regex
import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout

from .. import config
from ..errors import Unsupported
from ..ir import FixedTiledLayout, SpyreConstantFallback
from ..loop_facts import counted_loop_path, is_looped
from ..loop_node import insert_after, move_after
from ..loop_node import LoopMember
from ..loop_node import LoopCarryRecord
from ..pass_utils import invalidate_op_read_writes, op_read_writes
from ..propagate_hints import DimHint
from .coarse_tile import (
    _allocate_full_buffer,
    _apply_work_div_symbol_remap,
    _check_matmul_broadcast_batch_tiling,
    _divide_reduction_ranges,
    _find_outside_consumers,
    _find_outside_consumers_planned,
    _group_reduction_tiled_levels_in_group,
    _make_combine_buffer,
    _patch_consumers,
    _patch_graph_outputs,
    _plan_is_loop_invariant_at_reduction_levels,
    _planned_full_device_layout,
    _planned_tile_extents_per_level,
    _reads_buffer,
    _reads_incomplete_reduction,
    _reduction_identity_value,
    _validate_contiguous,
    _validate_planned_reduction_tiling,
    reject_nested_loops,
    resolve_level_positions,
)
from .loop_tiling_common import (
    _AddLoopTermHandler,
    _divide_ranges,
    _patch_retiled_load_indexes,
    _RetiledBufferInfo,
)

logger = logging.getLogger(__name__)

# The ops this lowering adds keep the names ``coarse_tile_copy_<buf>``,
# ``coarse_tile_fill_<buf>`` and ``coarse_tile_combine_<buf>``. No source file keys on
# those prefixes (comments mention them); 19 test files match them, so the names stay
# until those tests are rewritten together (stage 9).


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

    ``op_name`` in the plan is ``get_operation_name()``. Raises ``Unsupported`` for a
    nested group (one loop level only), reduction tiling while
    ``config.enable_reduction_tiling`` is off, a mixed or multi-dim reduction level,
    an op that reads a reduction before it is complete, and a matmul tiled across a
    broadcast batch.
    """
    reject_nested_loops(groups, "build_loop_tiling_plan")
    planned_groups: list[LoopTilingGroup] = []
    for group_idx, (group_ops, levels) in enumerate(groups):
        if not sympy.sympify(levels[0][1]).is_Integer:
            raise Unsupported(
                f"build_loop_tiling_plan: the trip count {levels[0][1]} of group "
                f"{group_idx + group_idx_offset} is symbolic; the plan holds an int."
            )
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


# ---------------------------------------------------------------------------
# The loop variable, the terms in the index, and the loop carries
# ---------------------------------------------------------------------------
# Everything here edits by wrapping: an op's ``inner_fn`` and ``store_output`` are
# wrapped, never rebuilt, because the index expressions go stale as soon as a pass runs
# (issue #2797).


_INDUCTION_NAME = regex.compile(r"u(\d+)")


def mint_induction(taken: Iterable[sympy.Basic]) -> sympy.Symbol:
    """A loop variable named ``u<n>``, with the smallest ``n`` no symbol in
    ``taken`` already uses.

    ``taken`` holds symbols or expressions; every free symbol counts. The name
    family is the one Inductor gives an unbacked scalar (``for_each_tile``'s loop
    variable is one), so a graph holding both kinds of loop needs distinct names.
    The variable is an integer, so ``floor(8*u)`` folds to ``8*u``.
    """
    used: set[int] = set()
    for item in taken:
        for symbol in getattr(item, "free_symbols", ()):
            match = _INDUCTION_NAME.fullmatch(symbol.name)
            if match:
                used.add(int(match.group(1)))
    n = 0
    while n in used:
        n += 1
    return sympy.Symbol(f"u{n}", integer=True, nonnegative=True)


def _taken_symbols(graph: Any, operations: Iterable[Any]) -> list[sympy.Basic]:
    """Every symbol a loop variable's name could collide with.

    The free symbols of each op's indexes and sizes, the loop variables its
    ``dim_hints`` already list, the graph inputs' sizes, and everything the shape
    environment has allocated (a ``for_each_tile`` loop variable is an unbacked
    scalar from there, and so is a data-dependent size that appears in no index).

    Raises ``Unsupported`` when an op's reads and writes cannot be read: a name
    that was not seen could be handed out twice, so the caller does nothing.
    """
    taken: list[sympy.Basic] = []
    shape_env = getattr(getattr(graph, "sizevars", None), "shape_env", None)
    taken.extend(getattr(shape_env, "var_to_range", None) or ())
    for value in (getattr(graph, "graph_inputs", None) or {}).values():
        get_size = getattr(value, "get_size", None)
        if get_size is not None:
            taken.extend(get_size())
    for op in operations:
        for hint in getattr(op, "dim_hints", None) or ():
            if hint.loop_var is not None:
                taken.append(hint.loop_var)
        get_size = getattr(op, "get_size", None)
        if get_size is not None:
            taken.extend(get_size())
        try:
            reads_writes = op_read_writes(op)
        except Exception as error:
            raise Unsupported(
                f"loop tiling lowering: cannot read the reads and writes of "
                f"{op.get_operation_name()} ({type(error).__name__}), so the names "
                "already in use are not known."
            ) from error
        taken.extend(
            dep.index
            for dep in (*reads_writes.reads, *reads_writes.writes)
            if isinstance(dep, MemoryDep)
        )
    return taken


@dataclass
class _OpEdit:
    """What to add to one op: planned first, applied only if the whole group plans."""

    op: ComputedBuffer
    trip: sympy.Integer
    read_terms: dict[str, sympy.Expr]  # buffer name -> step (before the variable)
    write_term: sympy.Expr | None


def _apply_edit(edit: _OpEdit, induction: sympy.Symbol) -> None:
    """Give the op the loop variable in ``dim_hints`` and the term in its index."""
    op = edit.op
    op.dim_hints = [  # type: ignore[attr-defined]
        *list(getattr(op, "dim_hints", None) or []),
        DimHint(
            dim_names=[],
            split_count=1,
            loop_var=induction,
            is_reduction=False,
            loop_var_range=edit.trip,
        ),
    ]
    read_terms = {name: term * induction for name, term in edit.read_terms.items()}
    if read_terms:
        original = op.data.inner_fn

        def inner_fn(*args, _original=original, _terms=read_terms):
            with contextlib.ExitStack() as stack:
                for name, term in _terms.items():
                    stack.enter_context(
                        V.set_ops_handler(_AddLoopTermHandler(V.ops, name, term))
                    )
                return _original(*args)

        object.__setattr__(op.data, "inner_fn", inner_fn)
    if edit.write_term is not None:
        write_term = edit.write_term * induction
        store_output = op.data.store_output

        def store(name, indexer, vars, _store=store_output, _term=write_term):
            return _store(name, lambda idx: indexer(idx) + _term, vars)

        object.__setattr__(op.data, "store_output", store)
    invalidate_op_read_writes(op)


@dataclass
class _CarryEdit:
    """A loop carry to declare: the accumulator's storage, the op that updates it
    each trip, and the record that names both."""

    update: ComputedBuffer
    storage: Any
    record: LoopCarryRecord


def _plan_carries(
    graph: Any,
    members: list[ComputedBuffer],
    reduction_names: Collection[str],
) -> list[_CarryEdit]:
    """The loop carries of a group of ``reduction`` kind, or ``[]`` for any other.

    A reduction tiled on its reduction dim lowers to a partial op
    (a ``Reduction`` into a scratch tile) and a combine (a ``Pointwise`` that
    writes the accumulator in place, ``MutationLayoutSHOULDREMOVE``); the fill that
    zeroes the accumulator runs once, before the loop. The accumulator is a carry:
    its storage is the buffer the combine mutates, its update the combine.

    Raises ``Unsupported`` when a
    partial op has no combine, when the storage is a loop member or already
    declares a carry (a ``for_each_tile`` one), or when anything but the combine
    touches the storage inside the loop: the record promises that the update is the
    storage's only in-loop access.
    """
    reductions = [op for op in members if op.get_operation_name() in reduction_names]
    if not reductions:
        return []
    partials = [op for op in reductions if isinstance(op.data, Reduction)]
    updates = [
        op
        for op in reductions
        if isinstance(op.data, Pointwise)
        and isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    ]
    if len(partials) != len(updates) or len(partials) + len(updates) != len(reductions):
        raise Unsupported(
            f"loop tiling lowering: {len(partials)} partial reductions and "
            f"{len(updates)} combines among {len(reductions)} reduction-kind ops "
            "of one group."
        )
    # By buffer name (``get_name``), as the record, the allocator and
    # ``apply_loop_carries`` key it: an extern op's name is op1, its buffer's buf1.
    live = {op.get_name(): op for op in graph.operations}
    touched = {}
    for op in members:
        rw = op_read_writes(op)
        touched[op.get_operation_name()] = {dep.name for dep in (*rw.reads, *rw.writes)}
    edits = []
    for update in updates:
        storage_name = update.layout.get_buffer().get_name()
        storage = live.get(storage_name)
        if storage is None:
            raise Unsupported(
                f"loop tiling lowering: the accumulator {storage_name} of "
                f"{update.get_operation_name()} is not an operation of the graph."
            )
        if getattr(storage, "_loop_carry_record", None) is not None:
            raise Unsupported(
                f"loop tiling lowering: {storage_name} already declares a carry."
            )
        if counted_loop_path(storage):
            raise Unsupported(
                f"loop tiling lowering: the accumulator {storage_name} is a "
                "member of the loop."
            )
        others = [
            name
            for name, names in touched.items()
            if name != update.get_operation_name() and storage_name in names
        ]
        if others:
            raise Unsupported(
                f"loop tiling lowering: {others} also touch the accumulator "
                f"{storage_name} inside the loop."
            )
        edits.append(
            _CarryEdit(
                update,
                storage,
                LoopCarryRecord(
                    storage_name=storage_name,
                    update_name=update.get_name(),
                    loop_origin=None,
                ),
            )
        )
    return edits


def _apply_carries(edits: list[_CarryEdit]) -> None:
    """Stamp each record on its storage and on its update, where the allocator, the
    loop checker and the node (``LoopNode.carries``) look for it."""
    for edit in edits:
        edit.storage._loop_carry_record = edit.record
        edit.update._loop_carry_record = edit.record


def _outer_key(op: Any) -> int | None:
    group = counted_loop_path(op, include_extern=True)
    return group[0] if group else None


def _extent(op: ComputedBuffer, tiled: TiledDim, trip: sympy.Integer) -> sympy.Integer:
    """The tile's extent along the tiled dim, from the undivided range."""
    position = tiled.position
    size = sympy.sympify(
        op.data.reduction_ranges[position]
        if tiled.in_reduction
        else op.data.ranges[position]
    )
    if not isinstance(size, sympy.Integer) or size % trip != 0:
        raise Unsupported(
            f"lower_loop_tiling: op {op.get_name()!r} loop var d{position} range "
            f"{size} is not divisible by loop_count {trip}.  All tiled dimensions "
            "must be evenly divisible by the loop trip count."
        )
    return sympy.Integer(int(size) // int(trip))


def _tile_step(
    op: ComputedBuffer,
    tiled: TiledDim,
    dep: MemoryDep,
    extent: sympy.Integer,
) -> sympy.Expr:
    """The host advance of one access per trip: the coefficient of the tiled dim's
    index symbol times the tile extent, read from the access BEFORE the divide. A
    dim that will be squeezed out (tile extent 1) is no special case: its symbol is
    still in the undivided index. The step is right for a read through a view that
    reorders or strides the op's dims too, because it comes from the access and not
    from the buffer's strides. An access that does not mention the symbol (a
    broadcast read) does not advance."""
    from ..spyre_kernel import SpyreKernel

    # A reduction dim is keyed after the op's output dims (see the kernel).
    key = len(op.data.ranges) + tiled.position if tiled.in_reduction else tiled.position
    symbol = SpyreKernel._host_dim_to_index_symbol(op, key)
    if symbol not in dep.index.free_symbols:
        return sympy.S.Zero
    # The delta of the whole index over one trip, which must be a number: the
    # coefficient at one point is not the step of an index that is only partly affine
    # in the symbol (``d0 + 7*floor(d0/8)``), and a symbolic one (``d0*d1``) has none.
    # The kernel reads the same advance back from the final index
    # (``loop_facts._loop_coefficient``, which evaluates it at u = 0, 1, 2); for the
    # ``step * u`` term added here the two agree by construction, and
    # ``test_loop_index_advance`` checks that they agree on real graphs.
    delta = sympy.simplify(dep.index.xreplace({symbol: symbol + extent}) - dep.index)
    if delta.free_symbols:
        raise Unsupported(
            f"lower_loop_tiling: the index {dep.index} of {dep.name} is not affine in "
            f"{symbol} with a numeric step, so the loop has no per-trip advance for it."
        )
    return delta


@dataclass
class _Facts:
    """What the lowering reads from the live graph about one member, before it
    changes anything in the group."""

    op: ComputedBuffer
    tiled: TiledDim | None
    kind: str  # "copy_out", "reduction" or "loop_internal"
    read_steps: dict[str, sympy.Expr]
    write_step: sympy.Expr | None
    full_ranges: list[sympy.Expr]
    full_strides: tuple
    full_device_layout: Any
    is_graph_output: bool


def _facts(
    graph: Any,
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
        if not isinstance(member.data, (Pointwise, Reduction)) or (
            tiled and tiled.in_reduction and not isinstance(member.data, Reduction)
        ):
            raise Unsupported(
                f"lower_loop_tiling: {name} ({type(member.data).__name__}) is not "
                "lowered by the plan."
            )
        if isinstance(member.layout, MutationLayoutSHOULDREMOVE):
            raise Unsupported(
                f"lower_loop_tiling: {name} mutates a buffer; the plan does not "
                "lower a mutation yet."
            )
        buf_name = member.get_name()
        if tiled and tiled.in_reduction:
            # The accumulator is complete only after the loop, so no member of the
            # loop may read the partial result.
            readers = [
                o.get_operation_name()
                for o in members
                if o is not member and _reads_buffer(o, buf_name)
            ]
            if readers:
                raise Unsupported(
                    f"lower_loop_tiling: {readers} read the partial result of "
                    f"{name} inside the loop that accumulates it."
                )
        consumers, is_graph_output = (
            _find_outside_consumers_planned(
                buf_name, (group.index,), operations, key_of
            )
            if tiled and not tiled.in_reduction
            else ([], False)
        )
        if tiled and tiled.in_reduction:
            kind = "reduction"
        elif tiled and (consumers or is_graph_output):
            kind = "copy_out"
        else:
            kind = "loop_internal"
        read_steps: dict[str, sympy.Expr] = {}
        write_step = None
        if any(
            h.loop_var_range is not None
            for h in getattr(member, "dim_hints", None) or ()
        ):
            raise Unsupported(
                f"lower_loop_tiling: {name} already carries a loop-variable hint (a "
                "spliced loop body); the plan does not tile it."
            )
        full_ranges = list(member.data.ranges)
        if tiled is None:
            reads_a_tile = sorted(
                {
                    d.name
                    for d in op_read_writes(member).reads
                    if isinstance(d, MemoryDep)
                }
                & scratch_names
            )
            if reads_a_tile:
                raise Unsupported(
                    f"lower_loop_tiling: the loop does not tile {name}, but it reads "
                    f"{reads_a_tile}, the tile of a member it does tile: that buffer "
                    "is rewritten every trip and is smaller than the range "
                    f"{name} iterates."
                )
        if tiled:
            extent = _extent(member, tiled, trip)
            rw = op_read_writes(member)
            for dep in (d for d in rw.reads if isinstance(d, MemoryDep)):
                # A read of a tiled member's own scratch does not advance: the
                # buffer is rewritten every trip.
                step = (
                    sympy.S.Zero
                    if dep.name in scratch_names
                    else _tile_step(member, tiled, dep, extent)
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
                write_step = _tile_step(member, tiled, writes[0], extent)
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
    tiled: TiledDim,
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
    copy_buf.loop_member = _loop_info(group, tiled)  # type: ignore[attr-defined]
    V.graph.name_to_buffer[copy_name] = copy_buf
    insert_after(operations, tiled_op, copy_buf)
    return copy_buf


def _loop_info(group: LoopTilingGroup, tiled: TiledDim | None) -> LoopMember:
    return LoopMember(
        loop_group_id=(group.index,),
        loop_count=[sympy.Integer(group.trip_count)],
        loop_tiled_dims=[[tiled.position] if tiled and not tiled.in_reduction else []],
        loop_tiled_reduction_dims=[
            [tiled.position] if tiled and tiled.in_reduction else []
        ],
        strict_loop_vars=True,
    )


def _insert_fill(
    op: ComputedBuffer,
    accum: ComputedBuffer,
    operations: list[Operation],
    identity: float,
    ranges: list[sympy.Expr],
) -> None:
    """The op that writes the reduction's identity into ``accum`` once, before the
    loop: it has no ``loop_member``, so it runs outside every loop.

    The identity is a ``SpyreConstantFallback`` scalar, which the kernel expresses as
    an IDENTITY_OP broadcast; its layout is built by hand because layout finalizing
    has already run."""
    dtype = op.get_dtype()
    device = op.get_device()
    assert device is not None
    scalar_op = SpyreConstantFallback(
        torch.ops.spyre.constant.default, float(identity), dtype, device
    )
    # A 0-d broadcast scalar in the device coordinate system.
    scalar_op.layout = FixedTiledLayout(
        device, dtype, [], [], SpyreTensorLayout([], dtype)
    )
    scalar_loader = TensorBox.create(scalar_op).make_loader()
    fill_name = V.graph.qualify_name(f"coarse_tile_fill_{op.get_name()}")
    fill_buf = ComputedBuffer(
        name=fill_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(accum))),
        data=Pointwise(
            device=device,
            dtype=dtype,
            inner_fn=lambda index, _loader=scalar_loader: _loader([]),
            ranges=ranges,
        ),
    )
    fill_buf.origins = op.origins
    fill_buf.operation_name = fill_name
    # Its write is read only by the next trip's combine: a dependency across
    # iterations that the pre-unroll IR does not show, so dead-code elimination
    # would drop it. The attribute keeps its legacy name because ``patches.py`` reads
    # it (the scheduler's dead-node check); it is live code, not a leftover.
    fill_buf._coarse_tile_force_live = True  # type: ignore[attr-defined]
    V.graph.name_to_buffer[fill_name] = fill_buf
    # The scalar was appended to the operations when it was created: move it to
    # just after the accumulator, then the fill after it.
    move_after(operations, accum, scalar_op)
    insert_after(operations, scalar_op, fill_buf)


def _lower_reduction(
    fact: _Facts, operations: list[Operation], group: LoopTilingGroup
) -> str:
    """A reduction tiled on its reduction dim: an accumulator the size of the output,
    filled with the identity once before the loop, and a combine op in the loop that
    folds each trip's partial result into it. Everything outside the loop that read
    the result reads the accumulator. Returns the combine's name."""
    op = fact.op
    buf_name = op.get_name()
    assert fact.tiled is not None
    start = next(
        o
        for o in operations
        if isinstance(o, ComputedBuffer) and _outer_key(o) == group.index
    )
    accum = _allocate_full_buffer(
        op,
        fact.full_ranges,
        fact.full_strides,
        operations,
        start,
        device_layout=fact.full_device_layout,
    )
    _insert_fill(
        op,
        accum,
        operations,
        _reduction_identity_value(op.data.reduction_type, op.get_dtype()),
        fact.full_ranges,
    )
    combine = _make_combine_buffer(op, accum)
    combine.loop_member = _loop_info(group, fact.tiled)  # type: ignore[attr-defined]
    V.graph.name_to_buffer[combine.get_name()] = combine
    insert_after(operations, op, combine)

    # insert_restickify.finalize_layouts reads this to give the accumulator the
    # reduction's device layout.
    op._tiled_reduction_accum_name = accum.get_name()  # type: ignore[attr-defined]

    outside, is_graph_output = _find_outside_consumers(
        buf_name, (group.index,), operations
    )
    _patch_consumers(
        outside,
        buf_name,
        accum.get_name(),
        operations,
        _RetiledBufferInfo(
            tuple(op.layout.stride),
            tuple(accum.layout.stride),
            tuple(op.layout.size),
            tuple(accum.layout.size),
        ),
    )
    if is_graph_output:
        _patch_graph_outputs(buf_name, accum)
    return combine.get_operation_name()


def _resolve(operations: list[Operation], names: list[str]) -> list[ComputedBuffer]:
    by_name = {op.get_operation_name(): op for op in operations}
    return [by_name[n] for n in names]  # type: ignore[misc]


def _lower_group(graph: Any, group: LoopTilingGroup, induction: sympy.Symbol) -> None:
    operations = graph.operations
    trip = sympy.Integer(group.trip_count)
    names = [o.op_name for o in group.ops]
    members = _resolve(operations, names)
    already = [m.get_operation_name() for m in members if is_looped(m)]
    if already:
        raise Unsupported(
            f"lower_loop_tiling: group {group.index} would overwrite the existing "
            f"loop_member of {', '.join(already)}."
        )
    _validate_contiguous(
        members,
        {op.get_operation_name(): i for i, op in enumerate(operations)},
        (group.index,),
    )
    facts = _facts(graph, group, members, operations)

    # Divide the tiled ops and stamp the shim on every member.
    retiled: dict[str, _RetiledBufferInfo] = {}
    for fact in facts:
        if fact.tiled is not None and fact.tiled.in_reduction:
            _apply_work_div_symbol_remap(
                fact.op, _divide_reduction_ranges(fact.op, trip, [fact.tiled.position])
            )
        elif fact.tiled is not None:
            result = _divide_ranges(fact.op, trip, [fact.tiled.position])
            _apply_work_div_symbol_remap(fact.op, result.symbol_remap)
            if result.retiled_info is not None:
                retiled[fact.op.get_name()] = result.retiled_info
        fact.op.loop_member = _loop_info(group, fact.tiled)  # type: ignore[attr-defined]

    # Accumulate every reduction tiled on its reduction dim.
    combines: dict[str, str] = {}
    for fact in facts:
        if fact.kind == "reduction":
            combines[fact.op.get_operation_name()] = _lower_reduction(
                fact, operations, group
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
            o
            for o in operations
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
        copy_buf = _insert_copy_out(op, full_buf, operations, group, fact.tiled)
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
    for combine in _resolve(operations, list(combines.values())):
        _apply_edit(_OpEdit(combine, trip, {}, None), induction)
    if combines:
        # Each accumulator is a loop carry: its storage is the buffer the combine
        # mutates, its update the combine.
        loop = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and _outer_key(op) == group.index
        ]
        _apply_carries(_plan_carries(graph, loop, {*combines, *combines.values()}))


def lower_loop_tiling(graph: Any, plan: LoopTilingPlan) -> dict[int, sympy.Symbol]:
    """Tile every group of ``plan``: divide the tiled ops, drain each tile that
    something outside the loop reads into a full buffer, and give every op the loop
    variable and every advancing access its term.

    Returns ``{group index: loop variable}`` for ``request_loop_nodes``. Each op
    carries a ``LoopMember`` with ``strict_loop_vars``: the advances live in the
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


def tile_groups(
    graph: Any, groups: list[tuple], group_idx_offset: int = 0
) -> dict[int, sympy.Symbol]:
    """Tile ``groups`` (``(ops, [(hint_id, trip_count)])``): the one entry point of
    the span-overflow pass and of the allocator's tiling. Plans them, lowers the plan
    and returns ``{group index: loop variable}`` for ``request_loop_nodes``."""
    return lower_loop_tiling(
        graph, build_loop_tiling_plan(graph.operations, groups, group_idx_offset)
    )
