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


"""Lifetimes read from the IR around coarse-tile loops: the single-tick
handoff between an in-place parent and child, a loop's trip count, and the
post-loop drain plans that extend a loop-carried buffer's lifetime to the
collective that drains it. The graph builder sets node lifetimes from these;
the commit path asserts the plans it consumed were committed."""

from __future__ import annotations
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from torch._inductor.ir import (
    Buffer,
    ExternKernel,
    MutationLayoutSHOULDREMOVE,
    Operation,
    ReinterpretView,
)
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.pass_utils import (
    op_read_writes,
    origin_in_graph,
)
from torch_spyre._inductor.scratchpad.utils import (
    clone_at_graph_boundaries,
)
from torch_spyre._inductor.scratchpad.graph_editor import (
    GraphEditor,
    unwrapped_buffer_name,
)

from torch_spyre._inductor.loop_info import (
    CoarseTileInfo,
    LoopCarryRecord,
)


def _handoff_child_start(
    name: str,
    lifetimes: dict[str, list[int]],
    lifetime_start_overrides: dict[str, int],
) -> int:
    """First tick of ``name`` as an in-place child, widened to a loop's start."""
    first = lifetimes[name][0]
    return min(first, lifetime_start_overrides.get(name, first))


def _handoff_parent_end(
    name: str,
    lifetimes: dict[str, list[int]],
    lifetime_end_overrides: dict[str, int],
) -> int:
    """Inclusive last tick of ``name`` as an in-place parent, widened to a loop's
    end (the override is exclusive)."""
    last = lifetimes[name][-1]
    return max(last, lifetime_end_overrides.get(name, last + 1) - 1)


def _coarse_tile_loop_trip_count(op: Any) -> int:
    """Total trip count of ``op``'s coarse-tiling loop, or 1 when untiled.

    ``CoarseTileInfo.loop_count`` is one entry per nesting level, outermost
    first; the real total is their product. An entry can be a symbolic
    ``sympy.Expr`` under dynamic shapes
    (``for_each_tile_lowering.py``'s ``_extract_trip_count``) rather than a
    concrete ``int`` -- coerced the same way ``dump_cost_model._loop_features``
    does for this exact field: ``int()`` when it resolves to a concrete
    value, 1 (not a propagated symbol that would later crash numeric code
    expecting a plain float) when it doesn't.
    """
    loop_info = getattr(op, "loop_info", None)
    if not isinstance(loop_info, CoarseTileInfo):
        return 1

    def _resolved(c) -> int:
        try:
            return int(c)
        except (TypeError, ValueError):
            return 1

    return max(1, math.prod(_resolved(c) for c in loop_info.loop_count))


@dataclass(frozen=True)
class DrainPlan:
    """Validated post-loop materialization plan for one resident loop carry.

    A ``for_each_tile`` accumulator returned from the graph is both a carry
    storage and a graph output.  The ordinary output clone runs after the
    storage's pre-loop initializer, so it would copy the initial value; the
    plan instead anchors one output clone after the whole counted loop.
    Computed once, pre-solve, by :func:`validated_drain_plans` and consumed by
    the residency gate, the lifetime extension and the post-solve push -- never
    re-derived from graph state that later passes mutate.

    The same drain serves a carry whose only post-loop access is a collective
    (e.g. the TP all_reduce of a traced MoE's expert sum).  A collective needs a
    materialized HBM operand, so without the drain the carry -- and with it
    every in-loop update -- stays in HBM.  With ``collective`` set, the
    drain is an ordinary HBM buffer and the collective is rewired to it.

    Attributes
    ----------
    storage_name:
        The carry's initial storage; for a graph-output carry also its single
        graph-output entry.
    update_name:
        Its one tagged in-loop mutator (``_loop_carry_record.update_name``).
    loop_group:
        The update's ``loop_info.loop_group_id`` -- exactly one level.
    anchor_op:
        The last operation of the loop's group subtree, as an ``Operation``
        object (identity survives other clones; a saved index would not).
        The drain is inserted immediately after it.
    loop_origin:
        The exact FX ``while_loop`` HOP node retained by the splice; the drain
        clone's FX node is inserted after it.
    collective:
        ``None`` for a graph-output carry.  Otherwise the single post-loop
        collective reading the storage, which the push repoints at the drain.
    """

    storage_name: str
    update_name: str
    loop_group: tuple[int, ...]
    anchor_op: Operation
    loop_origin: Any
    collective: Optional[Operation] = None


def _access_group_path(op: Operation) -> tuple[int, ...]:
    """The op's counted-loop group path, mirroring utils.group_path exactly."""

    if isinstance(op, ExternKernel):
        return ()
    return tuple(getattr(getattr(op, "loop_info", None), "loop_group_id", ()) or ())


def _is_reinterpret_output_entry(entry: Any) -> bool:
    """Whether a ReinterpretView sits anywhere between the entry and its buffer.

    ``GraphEditor.change_graph_output`` keeps such a view and repoints its
    ``.data`` in place.  The splice may have handed that same view object to the
    carry update as its mutation target (an init that is a view, e.g.
    ``zeros_like`` of a transposed tensor, reaches the output as
    ``TensorBox(StorageBox(view))``), so repointing it would make the in-loop
    update write the drain clone instead of the carry.
    """

    node = entry
    while not isinstance(node, Buffer):
        if isinstance(node, ReinterpretView):
            return True
        node = getattr(node, "data", None)
        if node is None:
            return False
    return False


def validated_drain_plans(graph: GraphLowering) -> dict[str, DrainPlan]:
    """The validated post-loop materialization plan for every eligible carry.

    Returns ``{}`` unless boundary cloning is on; the solver's relation gates
    the update's division against the carry's.  A storage is planned only if every predicate holds, and any
    failure means that storage is simply absent from the plan -- every consumer
    then keeps today's behavior (the carry stays in HBM and no clone is ever
    created):

    P1  boundary cloning is enabled;
    P2a (graph output) exactly one ``graph.graph_outputs`` entry names the
        storage, and no ReinterpretView sits anywhere on its wrapper chain
        (``change_graph_output`` replaces the first match, so an aliased entry
        declines, and it repoints a view in place); or
    P2b (collective) no entry names the storage, and its only access outside
        the loop is one post-loop collective whose operand
        :meth:`GraphEditor.collective_operand_name` resolves to it
        (:func:`_sole_post_loop_collective`);
    P3  the storage op carries a ``LoopCarryRecord`` whose ``storage_name`` is
        its own name and whose ``update_name`` resolves to exactly one op;
    P4  that update is the only op mutating the storage;
    P5  the update's group path is exactly one level, every in-loop access of
        the storage is that update, and the storage itself (the initializer)
        has no loop membership;
    P6  the retained FX ``while_loop`` origin lives in this graph;
    P7  the loop's group subtree is non-empty, so its last member -- the
        lowered insertion anchor -- exists;
    P8  the storage has an FX origin in this graph: the drain's FX clone reads
        that node.  The pre-loop ownership copy of a caller's init is built
        without origins by design (``while_loop_bridge._make_copying_buffer``),
        so such a carry declines.

    Intentional false negatives (safe declines, not bugs): a carry whose
    initializer is a view (the mutation target names the view, not the backing
    buffer), a carry with any second in-loop access, and a loop whose group is
    nested in another.  Every one of them keeps today's HBM behavior.

    The plan is data, not a decision: the existing solver still chooses
    residency and ownership, and an HBM selection emits no copy at all.
    """

    if not clone_at_graph_boundaries():
        return {}
    fx_graph = getattr(graph, "graph", None)
    if fx_graph is None:
        return {}
    op_by_name: dict[str, Operation] = {op.name: op for op in graph.operations}
    mutators: dict[str, list[Operation]] = defaultdict(list)
    for op in graph.operations:
        layout = getattr(op, "layout", None)
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            mutators[layout.target.get_name()].append(op)

    plans: dict[str, DrainPlan] = {}
    for name, storage_op in op_by_name.items():
        record = getattr(storage_op, "_loop_carry_record", None)
        if not isinstance(record, LoopCarryRecord):
            continue
        if record.storage_name != name:
            continue
        loop_origin = record.loop_origin
        if loop_origin is None or getattr(loop_origin, "graph", None) is not fx_graph:
            continue
        if origin_in_graph(getattr(storage_op, "origins", ()), fx_graph) is None:
            continue
        update_op = op_by_name.get(record.update_name)
        if update_op is None:
            continue
        update_group = _access_group_path(update_op)
        if len(update_group) != 1:
            continue
        if _access_group_path(storage_op):
            continue
        storage_mutators = mutators.get(name, [])
        if len(storage_mutators) != 1 or storage_mutators[0] is not update_op:
            continue
        # Every access but the initializer's and the update's.
        accessors = [
            op
            for op in graph.operations
            if op is not storage_op
            and op is not update_op
            and any(
                dep.name == name
                for dep in op_read_writes(op).reads | op_read_writes(op).writes
            )
        ]
        if any(_access_group_path(op) for op in accessors):
            continue
        anchor_op = None
        for op in graph.operations:
            op_group = _access_group_path(op)
            if op_group and op_group[0] == update_group[0]:
                anchor_op = op
        if anchor_op is None:
            continue
        output_entries = [
            index
            for index, entry in enumerate(graph.graph_outputs)
            if unwrapped_buffer_name(entry) == name
        ]
        collective = None
        if len(output_entries) == 1:
            if _is_reinterpret_output_entry(graph.graph_outputs[output_entries[0]]):
                continue
        elif output_entries:
            continue
        else:
            collective = _sole_post_loop_collective(graph, name, accessors, anchor_op)
            if collective is None:
                continue
        plans[name] = DrainPlan(
            storage_name=name,
            update_name=record.update_name,
            loop_group=update_group,
            anchor_op=anchor_op,
            loop_origin=loop_origin,
            collective=collective,
        )
    return plans


def _sole_post_loop_collective(
    graph: GraphLowering,
    name: str,
    accessors: Sequence[Operation],
    anchor_op: Operation,
) -> Optional[Operation]:
    """The collective that is ``name``'s only access outside its loop, or None.

    ``accessors`` are the ops, other than the storage's initializer and its
    tagged update, that read or write the storage.  The plan applies only when
    that is a single op running after the whole loop, and it is a collective
    whose operand :class:`GraphEditor` can repoint at the drain.  Any other
    post-loop access would still name the storage once the collective reads the
    drain -- and an in-place all_reduce's result lives in its operand -- so it
    declines.
    """

    if len(accessors) != 1:
        return None
    consumer = accessors[0]
    if GraphEditor.collective_operand_name(consumer) != name:
        return None
    if graph.operations.index(consumer) <= graph.operations.index(anchor_op):
        return None
    return consumer


def _drain_lifetime_end_overrides(
    lifetime_end_overrides: dict[str, int],
    drain_plans: Mapping[str, DrainPlan],
    graph_end: int,
) -> None:
    """Extend each graph-output drain storage's lifetime to the graph exit, in place.

    All solver intervals are pre-insertion indices and ``graph_end =
    len(graph.operations)`` is the exclusive end every pre-insertion op lies
    before.  Extending a storage's end to ``graph_end`` means no solver buffer
    can share its address after its birth (a sharer would have to be dead
    strictly before the fill), which is what makes the post-solve drain read
    safe regardless of where the scheduler eventually places it.  This only
    removes reuse; it adds no unpriced occupancy and no cost term.

    A collective drain is not extended: the collective reads the storage, so
    the storage already lives until the collective, which must run after the
    drain it reads.  Extending it would hold every layer's carry in LX to the
    graph exit.
    """

    for planned_name, plan in drain_plans.items():
        if plan.collective is not None:
            continue
        lifetime_end_overrides[planned_name] = max(
            lifetime_end_overrides.get(planned_name, 0), graph_end
        )


def _drained_collectives(drain_plans: Mapping[str, DrainPlan]) -> dict[str, str]:
    """Storage name -> the collective its drain plan repoints at the drain."""

    return {
        name: plan.collective.get_name()
        for name, plan in drain_plans.items()
        if plan.collective is not None
    }
