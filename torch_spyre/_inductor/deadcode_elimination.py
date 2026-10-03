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
from torch._inductor.virtualized import V


def live_operations(operations: list[Operation]) -> frozenset[str]:
    """Return the set of operation names that are transitively needed by the
    graph outputs.

    A buffer is live if it appears in V.graph.get_output_names() or if it is
    read by a live operation.  We walk the operation list in reverse
    (topological order) to propagate liveness backwards.
    """
    # A graph input counts as observable: an ``out=`` destination is read again
    # after the call while appearing in no output list. ``mutated_buffers`` is
    # deliberately not consulted -- it records every mutation target, internal
    # ones included, so seeding from it would keep every mutation alive.
    live_bufs: set[str] = set(V.graph.get_output_names())
    live_bufs |= set(V.graph.graph_inputs)
    live_ops: set[str] = set()

    # A mutation is judged by whether its target is live, and a reader of that
    # target can sit anywhere, including after the mutation in the walk order.
    # Iterating to a fixed point is what lets such a reader still be seen; the
    # sets only grow, so this terminates.
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
        # A side-effecting op is live whenever its write can still be observed,
        # and its reads are live with it because they feed that write. A
        # mutation into a dead target cannot be observed, so it falls through
        # to be judged on its own reachability like any pure op.
        if _has_side_effects(op) and _mutation_target_is_live(op, live_bufs):
            live_ops.add(op.get_operation_name())
            rw = op.get_read_writes()
            for dep in rw.reads:
                live_bufs.add(dep.name)
            continue
        rw = op.get_read_writes()
        writes = {dep.name for dep in rw.writes}
        if writes & live_bufs:
            live_ops.add(op.get_operation_name())
            for dep in rw.reads:
                live_bufs.add(dep.name)


def _mutation_target_is_live(op: Operation, live_bufs: set[str]) -> bool:
    """Whether a side-effecting op's write can still be observed.

    Only a ``MutationLayoutSHOULDREMOVE`` names a target, so anything else
    side-effecting (an impure ``FallbackKernel``) is assumed observable.
    """
    if not (
        isinstance(op, ComputedBuffer)
        and isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    ):
        return True
    return op.layout.get_buffer().get_name() in live_bufs


def _has_side_effects(op: Operation) -> bool:
    """Return True if op must not be eliminated regardless of whether its
    outputs are used.

    Mutation ops always have side effects.  FallbackKernel delegates to
    is_impure on the underlying op overload.  All other op types are pure.
    """
    if isinstance(op, ComputedBuffer) and isinstance(
        op.layout, MutationLayoutSHOULDREMOVE
    ):
        return True
    if isinstance(op, FallbackKernel):
        return op.has_side_effects()
    return False


def deadcode_elimination(graph: GraphLowering) -> None:
    """Remove dead operations from the list in-place, mirroring the
    scheduler's dead_node_elimination but running at pre-scheduler time.

    An operation is dead if none of its output buffers are transitively
    needed by the graph outputs and it has no side effects.  Dead output
    buffer names are added to graph.removed_buffers so that downstream
    codegen skips them.

    Operations are expected to be in topological order.  The list is
    modified in-place; the relative order of surviving operations is
    preserved.
    """
    operations = graph.operations
    live_ops = live_operations(operations)

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
