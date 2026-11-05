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

"""Apply each declared loop carry: make its update an in-place write.

``splice_while_loops`` declares each accumulator carry of a ``for_each_tile``
loop with a :class:`~torch_spyre._inductor.loop_info.LoopCarryRecord` on the
carry's storage (its pre-loop initial buffer) and on its update (the body op
that computes the next trip's value). It leaves the body pure: the update
writes its own buffer, and the loop's result is that buffer.

:func:`apply_loop_carries` turns each declaration into the in-place form the
rest of the pipeline expects. The update becomes a
``MutationLayoutSHOULDREMOVE`` write into the storage, so the next trip reads
the value the last trip wrote, and every reader of the update reads the
storage instead. It runs as a pre-scheduling pass, before layout propagation:
from propagation on, every pass sees the same in-place form it saw when the
splice made the rewrite itself.
"""

from __future__ import annotations

from typing import Any

from torch._inductor import ir
from torch._inductor.graph import GraphLowering

from .logging_utils import get_inductor_logger
from .loop_info import LoopCarryRecord
from .loop_node import OpsView, all_ops, has_loop_nodes
from .pass_utils import invalidate_op_read_writes, redirect_computed_buffer_reads

logger = get_inductor_logger("loop_carries")

_PASS_NAME = "apply_loop_carries"


def _records(operations: list[ir.Operation]) -> list[LoopCarryRecord]:
    """Every ``LoopCarryRecord`` on an op in ``operations``, each once.

    A record sits on both its storage and its update, so a carry is found
    from either end.
    """
    records: list[LoopCarryRecord] = []
    for op in operations:
        record = getattr(op, "_loop_carry_record", None)
        if isinstance(record, LoopCarryRecord) and record not in records:
            records.append(record)
    return records


def _is_buffer(node: Any) -> bool:
    return isinstance(node, ir.Buffer)


def _storage_of(record: LoopCarryRecord, live: dict[str, ir.Operation]) -> ir.Buffer:
    """The carry's storage, by name now: earlier passes rebuild ops under the
    same name, so an object kept from the splice may be stale."""
    storage = live.get(record.storage_name)
    if not _is_buffer(storage):
        raise RuntimeError(
            f"{_PASS_NAME}: carry update {record.update_name!r} is in the graph "
            f"but its storage {record.storage_name!r} is not"
        )
    return storage  # type: ignore[return-value]


def _write_target(record: LoopCarryRecord, storage: ir.Buffer) -> Any:
    """What the update writes through: the storage, or the recorded view of it
    when the carry's initial value is a view (for example a transposed init).

    A view reaches its backing buffer through ``.data`` links. When a pass
    rebuilt the storage since the splice, the link still names the old object;
    point it at the live buffer, as ``pass_utils._repoint_mutation_targets``
    does for a mutation target.
    """
    view = record.storage_view
    if view is None:
        return storage
    holder = view
    while hasattr(holder, "data"):
        inner = holder.data
        if _is_buffer(inner):
            if inner.get_name() != storage.get_name():
                raise RuntimeError(
                    f"{_PASS_NAME}: carry update {record.update_name!r} writes "
                    f"through a view of {inner.get_name()!r}, not of its storage "
                    f"{storage.get_name()!r}"
                )
            if inner is not storage:
                object.__setattr__(holder, "data", storage)
            return view
        holder = inner
    raise RuntimeError(
        f"{_PASS_NAME}: the recorded view of carry storage "
        f"{record.storage_name!r} reaches no buffer"
    )


def _repoint_mutation_targets_by_name(
    operations: list[ir.Operation], names: dict[str, Any]
) -> None:
    """Point every mutation target that bottoms out at a buffer named in
    ``names`` at that name's replacement, at any depth of box/view wrapping.

    By name, not identity: an op the splice repointed at an update (a copy
    back into the loop's result) may hold an object that a later rebuild of
    the update left behind. Same walk as
    ``pass_utils._repoint_mutation_targets``.
    """

    def replacement(node: Any) -> Any:
        if _is_buffer(node):
            return names.get(node.get_name())
        return None

    for op in operations:
        layout = getattr(op, "layout", None)
        if not isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            continue
        new = replacement(layout.target)
        if new is not None:
            layout.target = new
            continue
        holder = layout.target
        while hasattr(holder, "data") and not _is_buffer(holder):
            inner = holder.data
            new = replacement(inner)
            if new is not None:
                object.__setattr__(holder, "data", new)
                break
            holder = inner


def _pending(
    graph: GraphLowering,
) -> list[tuple[LoopCarryRecord, ir.ComputedBuffer, Any]]:
    """``(record, update, write target)`` for each carry not yet in place."""
    live = {op.get_name(): op for op in all_ops(graph.operations)}
    out = []
    for record in _records(list(live.values())):
        update = live.get(record.update_name)
        if update is None:
            if record.storage_name in live:
                # A private carry's storage is read only through the loop, so
                # a live storage whose update is gone would read a stale
                # value on every trip after the first.
                raise RuntimeError(
                    f"{_PASS_NAME}: carry update {record.update_name!r} is gone "
                    f"but its storage {record.storage_name!r} is still in the "
                    "graph; the carry's write would be lost"
                )
            continue  # the whole carry was removed as dead
        if not isinstance(update, ir.ComputedBuffer):
            raise RuntimeError(
                f"{_PASS_NAME}: carry update {record.update_name!r} is a "
                f"{type(update).__name__}, not a ComputedBuffer"
            )
        if isinstance(update.layout, ir.MutationLayoutSHOULDREMOVE):
            continue  # already applied
        storage = _storage_of(record, live)
        out.append((record, update, _write_target(record, storage)))
    return out


def apply_loop_carries(graph: GraphLowering) -> None:
    """Make each declared carry's update an in-place write of its storage.

    Finds the work from the ``LoopCarryRecord`` declarations, not from objects
    kept at the splice: the update and the storage are looked up by name, as
    earlier passes rebuild ops under the same name. For each carry:

    - the update's layout becomes ``MutationLayoutSHOULDREMOVE`` of the
      storage (or of the recorded view of it);
    - every ``ComputedBuffer`` that reads the update's name, in the loop body
      or after it, is rebuilt to read the storage;
    - object references to the update (``.inputs`` of ops without an
      ``inner_fn``, mutation targets, graph outputs) are pointed at the
      storage.

    A carry already in place is left alone, so a second run changes nothing.
    A carry whose update and storage are both gone was removed as dead and has
    nothing to apply. One end missing while the other is still in the graph
    raises. A scatter
    carry has no record (the splice keeps its write in place itself) and is
    not seen here.

    Runs under ``passes._flat_operations``, or on a list holding
    ``LoopNode``s: a rebuilt op replaces the old one in the list that owns it.
    Every reader of an update is found with one trace of each op per run (the
    names are batched into one ``redirect_computed_buffer_reads`` call); the
    ``op_read_writes`` memo is not used, as a rebuilt op would not invalidate it.
    """
    from .wsr.while_loop_bridge import (
        _repoint_refs_to_buffer,
        _substitute_direct_input_refs,
    )

    pending = _pending(graph)
    if not pending:
        return

    for _record, update, target in pending:
        update.layout = ir.MutationLayoutSHOULDREMOVE(target)
        # The store index comes from the layout; drop what was computed from
        # the old one.
        ir.ComputedBuffer.get_default_sizes_body.clear_cache(update)
        invalidate_op_read_writes(update)

    name_map = {update.get_name(): r.storage_name for r, update, _ in pending}
    updates = {update.get_name() for _r, update, _ in pending}
    operations = graph.operations
    if not isinstance(operations, OpsView) and has_loop_nodes(operations):
        # Called on the real nested list (not through the pipeline's flat
        # view): edit through a view so a rebuilt op lands in its own body.
        operations = OpsView(operations)
    for op in list(all_ops(operations)):
        if not isinstance(op, ir.ComputedBuffer):
            continue
        own = op.get_name()
        reads = {dep.name for dep in op.get_read_writes().reads}
        if reads & (updates - {own}):
            redirect_computed_buffer_reads(
                op,
                {k: v for k, v in name_map.items() if k != own},
                operations,
                pass_name=_PASS_NAME,
                reason="redirect reads of a carry update to its storage",
            )

    targets = {update.get_name(): target for _r, update, target in pending}
    flat = list(all_ops(operations))
    _substitute_direct_input_refs(flat, targets)
    _repoint_mutation_targets_by_name(flat, targets)
    for name, target in targets.items():
        _repoint_refs_to_buffer(graph, name, target, redirect_readers=False)
    logger.debug("%s: applied %d carries: %s", _PASS_NAME, len(pending), name_map)
