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

"""Put a span-overflow group's loop in the index of its accesses (stage 7a.2).

The legacy lowering (``coarse_tile_post_stickify``) records each access's per-trip
advance in ``loop_info`` and leaves the index without a loop term. This module adds
the term, ``step * u``, to every advancing access of a group whose ops all
``copy_out``, and gives every op of the group the loop variable in its
``dim_hints``. The kernel then reads the advance from the index and checks it
against the record (``LoopAdvanceMismatch``), which stays until stage 7.4.

Everything here edits by wrapping: an op's ``inner_fn`` and ``store_output`` are
wrapped, never rebuilt, because the index expressions go stale as soon as a pass
runs (issue #2797).
"""

import contextlib
import dataclasses
import logging
from collections.abc import Collection, Iterable
from typing import Any

import regex
import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
)
from torch._inductor.virtualized import V

from ..errors import Unsupported
from ..loop_facts import counted_loop_path
from ..loop_info import LoopCarryRecord
from ..pass_utils import invalidate_op_read_writes
from ..propagate_hints import DimHint
from .loop_tiling_common import _AddLoopTermHandler

logger = logging.getLogger(__name__)

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


def access_step(
    index: sympy.Expr,
    dim_symbol: sympy.Symbol | None,
    extent: sympy.Expr,
    buffer_stride: sympy.Expr,
) -> sympy.Expr:
    """The host-element advance of one access per trip, along one tiled dim.

    ``index`` is the access's own index. ``dim_symbol`` is the index symbol of the
    tiled dim, or ``None``; ``extent`` is the tile's extent along that dim.
    When the symbol is in the index the step is its coefficient times the extent.
    A dim tiled down to extent one is squeezed out of the op, so no symbol
    survives; the step is then the buffer's own stride times the extent. It is
    never the product of the sizes to the right of the dim, which is the stride
    only for a contiguous buffer (#6093).

    Raises ``Unsupported`` for an extent that is not an integer, and for an index
    that holds the symbol without a plain numeric coefficient for it.
    """
    if not isinstance(extent, sympy.Integer):
        raise Unsupported(
            f"span-overflow loop term: the tile extent {extent} is not an integer "
            "(symbolic span-overflow tiling is not supported)."
        )
    if dim_symbol is not None and dim_symbol in index.free_symbols:
        coefficient = index.coeff(dim_symbol)
        if coefficient == 0 or coefficient.free_symbols:
            # The symbol is there but not as a plain term: inside a floor or modulo
            # (coefficient 0), or multiplied by another symbol (coefficient d1 in
            # d0*d1). There is no numeric step to read.
            raise Unsupported(
                f"span-overflow loop term: the index {index} is not affine in "
                f"{dim_symbol} with a numeric coefficient."
            )
        return coefficient * extent
    return sympy.sympify(buffer_stride) * extent


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
            reads_writes = op.get_read_writes()
        except Exception as error:
            raise Unsupported(
                f"span-overflow loop term: cannot read the reads and writes of "
                f"{op.get_operation_name()} ({type(error).__name__}), so the names "
                "already in use are not known."
            ) from error
        taken.extend(
            dep.index
            for dep in (*reads_writes.reads, *reads_writes.writes)
            if isinstance(dep, MemoryDep)
        )
    return taken


def _group_index(op: Any) -> int | None:
    info = getattr(op, "loop_info", None)
    group = getattr(info, "loop_group_id", None)
    return group[0] if group else None


def _kind(op: Any) -> str | None:
    propagation = getattr(getattr(op, "loop_info", None), "propagation", None)
    return getattr(propagation, "kind", None)


def _stride(graph: Any, buffer_name: str, rank: int, dim: int) -> sympy.Expr:
    """The stride of ``buffer_name`` along host dim ``dim``.

    Only meaningful when the buffer has the op's rank (the host dim numbers the
    op's ranges); anything else is refused rather than guessed.
    """
    buffer = graph.get_buffer(buffer_name)
    if len(buffer.get_size()) != rank:
        raise Unsupported(
            f"span-overflow loop term: {buffer_name} has rank "
            f"{len(buffer.get_size())} but the op iterates {rank} dims, so a "
            f"squeezed tiled dim {dim} cannot be mapped to its stride."
        )
    return sympy.sympify(buffer.get_stride()[dim])


def _require_same_dim_order(
    graph: Any, op: ComputedBuffer, dep: MemoryDep, buffer_name: str
) -> None:
    """Refuse a squeezed dim whose read is not a plain view of the buffer.

    A dim tiled down to extent one has no symbol left in the index, so its step is
    taken from the buffer's stride for the SAME dim number. That holds only when
    the op's dims are the buffer's dims in the same order: for each dim that
    survives, the index must step by the buffer's stride for that dim. A read
    through a permute, a slice with a step or a reshape fails that, and the group
    stays on the legacy record, which reads the step from the index before the
    divide.
    """
    from ..spyre_kernel import SpyreKernel

    rank = len(op.data.ranges)
    buffer = graph.get_buffer(buffer_name)
    if len(buffer.get_size()) != rank:
        raise Unsupported(
            f"span-overflow loop term: {buffer_name} has rank "
            f"{len(buffer.get_size())} but {op.get_operation_name()} iterates "
            f"{rank} dims."
        )
    for dim in range(rank):
        if op.data.ranges[dim] == 1:
            continue
        symbol = SpyreKernel._host_dim_to_index_symbol(op, dim)
        if symbol in dep.index.free_symbols and (
            dep.index.coeff(symbol) != sympy.sympify(buffer.get_stride()[dim])
        ):
            raise Unsupported(
                f"span-overflow loop term: {op.get_operation_name()} reads "
                f"{buffer_name} through a view that reorders or strides its dims "
                f"(dim {dim}: index step {dep.index.coeff(symbol)}, buffer stride "
                f"{buffer.get_stride()[dim]})."
            )


def _access_term(
    graph: Any,
    op: Any,
    dep: MemoryDep,
    buffer_name: str,
    pairs: list[tuple[int, sympy.Expr]],
    squeezed: list[tuple[sympy.Expr, sympy.Expr]],
) -> sympy.Expr:
    """The per-trip host advance of one access, from its own index and buffer.

    ``pairs`` are the planner's ``(host dim, tile extent)`` for the dims that keep a
    symbol in the index; ``squeezed`` has one entry per dim tiled down to extent
    one (its own stride is read from the buffer, not from the entry). The planner
    decides WHICH accesses advance; the step is not copied from its record, so the
    kernel's check compares two independent derivations.
    """
    from ..spyre_kernel import SpyreKernel

    rank = len(op.data.ranges)
    total: sympy.Expr = sympy.S.Zero
    for dim, extent in pairs:
        symbol = SpyreKernel._host_dim_to_index_symbol(op, dim)
        if symbol not in dep.index.free_symbols:
            # The access does not depend on this dim (a broadcast read), so it does
            # not advance along it. The stored record substitutes into the index and
            # gets the same zero; a buffer stride here would be wrong whenever the
            # op's dims are not the buffer's.
            continue
        total += access_step(dep.index, symbol, sympy.sympify(extent), sympy.S.Zero)
    if squeezed:
        _require_same_dim_order(graph, op, dep, buffer_name)
        dims = [d for d in op.loop_info.loop_tiled_dims[0] if op.data.ranges[d] == 1]
        if len(dims) != len(squeezed):
            raise Unsupported(
                f"span-overflow loop term: {op.get_operation_name()} records "
                f"{len(squeezed)} squeezed tiled dims but {len(dims)} have extent 1."
            )
        for dim in dims:
            total += access_step(
                dep.index,
                None,
                sympy.Integer(1),
                _stride(graph, buffer_name, rank, dim),
            )
    return total


def _level0(per_level: Any) -> list:
    """Level 0 of a per-level list, or ``[]`` when the list does not reach it."""
    return list(per_level[0]) if per_level else []


@dataclasses.dataclass
class _OpEdit:
    """What to add to one op: planned first, applied only if the whole group plans."""

    op: ComputedBuffer
    trip: sympy.Integer
    read_terms: dict[str, sympy.Expr]  # buffer name -> step (before the variable)
    write_term: sympy.Expr | None


def _plan_op(graph: Any, op: ComputedBuffer) -> _OpEdit:
    """Work out ``op``'s terms without editing it. Raises ``Unsupported`` for an
    access this module cannot express (the caller then leaves the group alone)."""
    info = op.loop_info  # type: ignore[attr-defined]
    trip = info.loop_count[0]
    if not isinstance(trip, sympy.Integer):
        raise Unsupported(
            f"span-overflow loop term: {op.get_operation_name()} has a trip count "
            f"{trip} that is not an integer."
        )
    reads_writes = op.get_read_writes()
    read_deps = [d for d in reads_writes.reads if isinstance(d, MemoryDep)]
    read_terms: dict[str, sympy.Expr] = {}
    for i, dep in enumerate(read_deps):
        squeezed = (
            _level0(info.squeezed_advance_per_read[i])
            if i < len(info.squeezed_advance_per_read)
            else []
        )
        pairs = (
            _level0(info.tiled_dims_per_read[i])
            if i < len(info.tiled_dims_per_read)
            else []
        )
        term = (
            _access_term(graph, op, dep, dep.name, pairs, squeezed)
            if pairs or squeezed
            else sympy.S.Zero
        )
        # The handler adds a term to every load of a name, so two reads of one
        # buffer must agree.
        if read_terms.setdefault(dep.name, term) != term:
            raise Unsupported(
                f"span-overflow loop term: {op.get_operation_name()} reads "
                f"{dep.name} twice with different advances."
            )
    write_term = None
    write_deps = [d for d in reads_writes.writes if isinstance(d, MemoryDep)]
    pairs = _level0(info.output_tiled_dims)
    squeezed = _level0(info.squeezed_advance_output)
    if write_deps and (pairs or squeezed):
        target = (
            op.layout.get_buffer().get_name()
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE)
            else write_deps[0].name
        )
        write_term = _access_term(graph, op, write_deps[0], target, pairs, squeezed)
    return _OpEdit(
        op, trip, {n: t for n, t in read_terms.items() if t != 0}, write_term
    )


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


@dataclasses.dataclass
class _CarryEdit:
    """A loop carry to declare: the accumulator's storage, the op that updates it
    each trip, and the record that names both."""

    update: ComputedBuffer
    storage: Any
    record: LoopCarryRecord


def _plan_carries(
    graph: Any,
    members: list[ComputedBuffer],
    reduction_names: Collection[str] | None = None,
) -> list[_CarryEdit]:
    """The loop carries of a group of ``reduction`` kind, or ``[]`` for any other.

    The legacy lowering of a reduction tiled on its reduction dim is a partial op
    (a ``Reduction`` into a scratch tile) and a combine (a ``Pointwise`` that
    writes the accumulator in place, ``MutationLayoutSHOULDREMOVE``); the fill that
    zeroes the accumulator runs once, before the loop. The accumulator is a carry:
    its storage is the buffer the combine mutates, its update the combine.

    Raises ``Unsupported`` (the group then stays whole on the legacy record) when a
    partial op has no combine, when the storage is a loop member or already
    declares a carry (a ``for_each_tile`` one), or when anything but the combine
    touches the storage inside the loop: the record promises that the update is the
    storage's only in-loop access.
    """
    reductions = [
        op
        for op in members
        if (
            _kind(op) == "reduction"
            if reduction_names is None
            else op.get_operation_name() in reduction_names
        )
    ]
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
            f"span-overflow loop term: {len(partials)} partial reductions and "
            f"{len(updates)} combines among {len(reductions)} reduction-kind ops "
            "of one group."
        )
    # By buffer name (``get_name``), as the record, the allocator and
    # ``apply_loop_carries`` key it: an extern op's name is op1, its buffer's buf1.
    live = {op.get_name(): op for op in graph.operations}
    touched = {
        op.get_operation_name(): {
            dep.name
            for dep in (*op.get_read_writes().reads, *op.get_read_writes().writes)
        }
        for op in members
    }
    edits = []
    for update in updates:
        storage_name = update.layout.get_buffer().get_name()
        storage = live.get(storage_name)
        if storage is None:
            raise Unsupported(
                f"span-overflow loop term: the accumulator {storage_name} of "
                f"{update.get_operation_name()} is not an operation of the graph."
            )
        if getattr(storage, "_loop_carry_record", None) is not None:
            raise Unsupported(
                f"span-overflow loop term: {storage_name} already declares a carry."
            )
        if counted_loop_path(storage):
            raise Unsupported(
                f"span-overflow loop term: the accumulator {storage_name} is a "
                "member of the loop."
            )
        others = [
            name
            for name, names in touched.items()
            if name != update.get_operation_name() and storage_name in names
        ]
        if others:
            raise Unsupported(
                f"span-overflow loop term: {others} also touch the accumulator "
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


def add_loop_terms(
    graph: Any, groups: list[tuple], group_idx_offset: int = 0
) -> dict[int, sympy.Symbol]:
    """Add the loop to every ``copy_out`` and ``reduction`` group the legacy lowering
    has just built.

    ``groups`` are the ``(ops, levels)`` the transform was given. A group with an
    op with no planned kind, with more than one level, or with an access this
    module cannot express (a symbolic trip count or extent, a squeezed dim of a
    buffer whose rank differs from the op's) is left alone, whole: the legacy
    record still drives it. Stage 7.4 turns that into an ``Unsupported``. For the
    others each member op, the copy-outs the transform added included, gets a loop
    variable in ``dim_hints`` and ``step * u`` in the index of every access the
    planner says advances. The stored record is untouched: the kernel compares the
    two (``LoopAdvanceMismatch``) until stage 7.4 deletes the record. A ``reduction``
    group also declares its accumulator as a loop carry (:func:`_plan_carries`).

    Returns ``{group index: loop variable}``, keyed by the first entry of the
    group's ``loop_group_id``, which is how ``loop_node.collapse`` keys inductions.
    """
    operations = list(graph.operations)
    try:
        taken = _taken_symbols(graph, operations)
    except Unsupported as error:
        logger.debug("no span-overflow group is converted: %s", error)
        return {}
    inductions: dict[int, sympy.Symbol] = {}
    for group_idx, (_, levels) in enumerate(groups, start=group_idx_offset):
        members = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and _group_index(op) == group_idx
        ]
        if not members or len(levels) != 1 or any(_kind(op) is None for op in members):
            continue
        try:
            edits = [_plan_op(graph, op) for op in members]
            carries = _plan_carries(graph, members)
        except Unsupported as error:
            # The legacy lowering still handles this group on its record alone, so
            # leave it whole: no hint and no term on any of its ops.
            logger.debug("group %d keeps the legacy lowering: %s", group_idx, error)
            continue
        induction = mint_induction(taken)
        taken.append(induction)
        for edit in edits:
            _apply_edit(edit, induction)
        _apply_carries(carries)
        inductions[group_idx] = induction
    return inductions
