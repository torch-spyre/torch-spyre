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

from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
    MutationLayoutSHOULDREMOVE,
    Operation,
)
from torch._inductor.graph import GraphLowering


def live_operations(graph: GraphLowering) -> frozenset[str]:
    """Return the names of the operations whose effect the caller can observe.

    A buffer is live if it is a graph output, a graph input or a constant, or
    if a live operation reads it.  A mutation is live only while its target is,
    and keeps that target live in turn.  Liveness propagates backwards over
    ``graph.operations``, which is in topological order.
    """
    operations = graph.operations
    # A graph input or a constant counts as observable: an ``out=`` destination
    # or a frozen parameter is read again after the call while appearing in no
    # output list. ``mutated_buffers`` is deliberately not consulted -- it
    # records every mutation target, internal ones included, so seeding from it
    # would keep every mutation alive.
    live_bufs: set[str] = set(graph.get_output_names())
    live_bufs |= set(graph.graph_inputs)
    live_bufs |= set(graph.constants)
    live_ops: set[str] = set()

    # A mutation is judged by whether its target is live, and in a spliced
    # while-loop body list order is not execution order: a read at the top of
    # the body sees the write at its bottom on the next trip. So a reader that
    # precedes the mutation can be what makes its target live, and a single
    # reverse pass would drop that accumulator write. (In straight-line code
    # such a reader cannot see the write, and keeping it is merely
    # conservative.) Iterate to a fixed point; the sets only grow, so this
    # terminates.
    for _ in range(len(operations) + 1):
        before = (len(live_bufs), len(live_ops))
        _walk_once(operations, live_bufs, live_ops)
        if (len(live_bufs), len(live_ops)) == before:
            break

    return frozenset(live_ops)


def _walk_once(
    operations: list[Operation], live_bufs: set[str], live_ops: set[str]
) -> None:
    """One reverse liveness pass, accumulating into ``live_bufs``/``live_ops``."""
    for op in reversed(operations):
        rw = op.get_read_writes()
        # A side-effecting op is live whenever its write can still be observed.
        # A mutation into a dead target cannot be observed, so it is judged on
        # its own reachability like any pure op.
        observable = _has_side_effects(op) and _mutation_target_is_live(op, live_bufs)
        if not observable and not ({dep.name for dep in rw.writes} & live_bufs):
            continue
        live_ops.add(op.get_operation_name())
        # Its reads feed the write. A mutation's write also lands in its
        # target's storage, so the target has to survive with it.
        for dep in rw.reads:
            live_bufs.add(dep.name)
        target = _mutation_target(op)
        if target is not None:
            live_bufs.add(target)


def _mutation_target(op: Operation) -> str | None:
    """The buffer a ``MutationLayoutSHOULDREMOVE`` op writes into, else None."""
    if isinstance(op, ComputedBuffer) and isinstance(
        op.layout, MutationLayoutSHOULDREMOVE
    ):
        return op.layout.get_buffer().get_name()
    return None


def _mutation_target_is_live(op: Operation, live_bufs: set[str]) -> bool:
    """Whether a side-effecting op's write can still be observed.

    Only a ``MutationLayoutSHOULDREMOVE`` names a target, so anything else
    side-effecting (an impure ``FallbackKernel``) is assumed observable.
    """
    target = _mutation_target(op)
    return target is None or target in live_bufs


def _has_side_effects(op: Operation) -> bool:
    """Return True if op must not be eliminated regardless of whether its
    outputs are used.

    A mutation writes into a buffer it does not own, so it has side effects;
    whether that write can still be observed is ``_mutation_target_is_live``'s
    call.  FallbackKernel delegates to is_impure on the underlying op overload.
    All other op types are pure.
    """
    if _mutation_target(op) is not None:
        return True
    if isinstance(op, FallbackKernel):
        return op.has_side_effects()
    return False


def deadcode_elimination(graph: GraphLowering) -> None:
    """Remove dead operations from the list in-place, mirroring the
    scheduler's dead_node_elimination but running at pre-scheduler time.

    An operation is dead if none of its output buffers is transitively needed
    by the graph's outputs, inputs or constants, and it has no side effect that
    can still be observed (a mutation into a dead target has none).  Dead output
    buffer names are added to graph.removed_buffers so that downstream
    codegen skips them.

    Operations are expected to be in topological order.  The list is
    modified in-place; the relative order of surviving operations is
    preserved.
    """
    operations = graph.operations
    live_ops = live_operations(graph)

    dead: list[Operation] = []
    for op in operations:
        if op.get_operation_name() in live_ops:
            continue
        dead.append(op)

    for op in dead:
        rw = op.get_read_writes()
        for dep in rw.writes:
            graph.removed_buffers.add(dep.name)
        operations.remove(op)
