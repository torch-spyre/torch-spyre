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

"""Operation ordering used by LX planning and restored by the scheduler."""

from __future__ import annotations

from collections import defaultdict

from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Operation


def _loop_path(op: Operation) -> tuple[int, ...]:
    info = getattr(op, "loop_info", None)
    return tuple(getattr(info, "loop_group_id", ()) or ())


def _dependency_predecessors(ops: list[Operation]) -> list[set[int]]:
    """Build RAW/WAR/WAW edges while preserving the input order's semantics."""

    predecessors: list[set[int]] = [set() for _ in ops]
    last_writer: dict[str, int] = {}
    readers_since_write: dict[str, set[int]] = defaultdict(set)

    for index, op in enumerate(ops):
        reads = set(op.get_read_names())
        mutations = set(op.get_mutation_names())
        writes = {op.get_name()} | mutations

        for name in reads | mutations:
            if name in last_writer:
                predecessors[index].add(last_writer[name])
        for name in writes:
            if name in last_writer:
                predecessors[index].add(last_writer[name])
            predecessors[index].update(readers_since_write[name])

        for name in reads:
            readers_since_write[name].add(index)
        for name in writes:
            last_writer[name] = index
            readers_since_write[name].clear()

    return predecessors


def _stable_topological_units(
    operations: list[Operation],
) -> list[list[Operation]]:
    """Make every outer counted-loop group one dependency-ordered unit."""

    predecessors = _dependency_predecessors(operations)
    outer_units: dict[int, list[Operation]] = {}
    units: list[list[Operation]] = []
    unit_indices: dict[int, int] = {}
    unit_index_by_op: dict[int, int] = {}

    for op_index, op in enumerate(operations):
        path = _loop_path(op)
        if path:
            unit = outer_units.get(path[0])
            if unit is None:
                unit = []
                outer_units[path[0]] = unit
                units.append(unit)
                unit_indices[id(unit)] = len(units) - 1
            unit.append(op)
        else:
            unit = [op]
            units.append(unit)
            unit_indices[id(unit)] = len(units) - 1
        unit_index_by_op[op_index] = unit_indices[id(unit)]

    unit_predecessors: list[set[int]] = [set() for _ in units]
    for consumer, deps in enumerate(predecessors):
        consumer_unit = unit_index_by_op[consumer]
        for producer in deps:
            producer_unit = unit_index_by_op[producer]
            if producer_unit != consumer_unit:
                unit_predecessors[consumer_unit].add(producer_unit)

    ordered_units: list[list[Operation]] = []
    seen: set[int] = set()
    visiting: set[int] = set()

    def visit(unit_index: int) -> None:
        if unit_index in seen:
            return
        if unit_index in visiting:
            # A dependency cycle between an outside op and a loop group means
            # the group cannot be made atomic. Preserve the original order;
            # downstream validation will report the malformed loop scope.
            raise ValueError
        visiting.add(unit_index)
        for dependency in sorted(unit_predecessors[unit_index]):
            visit(dependency)
        visiting.remove(unit_index)
        seen.add(unit_index)
        ordered_units.append(units[unit_index])

    try:
        for unit_index in range(len(units)):
            visit(unit_index)
    except ValueError:
        return [[op] for op in operations]
    return ordered_units


def schedule_loop_body_for_liveness(graph: GraphLowering) -> None:
    """Preserve lowering's counted-loop order through LX planning and codegen.

    This runs immediately before scratchpad planning, so ``graph.operations``
    is both the allocator's liveness timeline and the order the scheduler must
    later restore. Inductor's FX locality reorder is disabled for Spyre, so lazy
    LoopIR lowering receives the decomposition's original block-major order.
    Inductor's later scheduler is still free to follow one recurrence branch
    across every unrolled KV block, extending every intervening LX lifetime, so
    this pass records the lowering order for restoration at that boundary.

    The only reordering here makes a counted-loop group atomic with respect to
    its preheader operations. Within every LoopSpec, a dependency-respecting
    order is left untouched.
    """

    operations = list(graph.operations)
    ordered: list[Operation] = []
    for unit in _stable_topological_units(operations):
        ordered.extend(unit)

    graph.operations[:] = ordered
    for index, op in enumerate(graph.operations):
        op._spyre_preschedule_order = index  # type: ignore[attr-defined]
