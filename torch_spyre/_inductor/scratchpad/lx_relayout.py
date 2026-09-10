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

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import cast

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
)

from .. import config
from ..core_mapping import (
    core_mappings_equal,
    owner_slots,
    _loop_regions,
    _LOOP_POINT,
    _MAX_EXACT_DIRECT_AXIS_POINTS,
    _MAX_EXACT_OWNERSHIP_POINTS,
    _EVALUATION_ERRORS,
    select_unique_partition_division,
)
from ..ir import FixedTiledLayout
from ..logging_utils import get_inductor_logger
from ..op_spec import TensorWorkDivision
from ..padding import is_restickify_op
from ..pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    iteration_space_from_op,
    op_read_writes,
    try_device_coordinates,
)
from .utils import _op_num_cores

logger = get_inductor_logger("lx_relayout")
_DESTINATION_PREFIX = "__spyre_lx_relayout__"
_REGISTRY = "_spyre_lx_relayout_copies"


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_names: tuple[str, ...]
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    source_address: int | None = None
    destination_address: int | None = None

    @property
    def destination_name(self) -> str:
        return f"{_DESTINATION_PREFIX}:{self.source_name}:{self.consumer_names[0]}"

    @property
    def edge(self) -> tuple[str, str]:
        return self.source_name, self.destination_name


def work_division_from_view(
    view: PerCoreView | None,
    device_size: Sequence[int],
    device_coordinates: Sequence[sympy.Expr],
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> TensorWorkDivision | None:
    """Interpret physical slices through an access, without choosing new owners."""
    if view is None:
        return None
    n = view.num_cores
    if n is None or n <= 0:
        raise ValueError("LX ownership must carry its physical core domain")
    physical_splits, slots = dict(view.work_slice_dims), dict(view.core_to_slot)
    if len(device_size) != len(device_coordinates):
        raise ValueError("sizes and coordinates differ in rank")
    if len(physical_splits) != len(view.work_slice_dims) or len(slots) != len(
        view.core_to_slot
    ):
        raise ValueError("duplicate physical dimensions")
    rows = owner_slots(slots, physical_splits, n)
    axes_by_loop: dict[sympy.Symbol, list[int]] = {}
    for axis, split in physical_splits.items():
        if (
            not 0 <= axis < len(device_size)
            or sympy.sympify(device_size[axis]).is_Integer is not True
            or device_size[axis] <= 0
            or device_size[axis] % split
        ):
            raise ValueError(
                f"unsupported ownership input: axis {axis} not divisible by {split}"
            )
        symbols = device_coordinates[axis].free_symbols
        if len(symbols) != 1 or not symbols <= iteration_space.keys():
            raise ValueError(f"cannot map device dimension {axis} to one loop")
        axes_by_loop.setdefault(next(iter(symbols)), []).append(axis)

    any_fused = any(len(axes) > 1 for axes in axes_by_loop.values())
    splits, owners, expected = {}, {}, {}
    fused_states = 0
    for loop, axes in axes_by_loop.items():
        extent = iteration_space[loop]
        extent = sympy.sympify(extent[0] if isinstance(extent, tuple) else extent)
        if extent.is_Integer is not True or extent <= 0:
            raise ValueError(
                f"unsupported ownership input: loop extent {extent} is not concrete"
            )
        extent = int(extent)
        first = axes[0]
        same = all(
            physical_splits[a] == physical_splits[first]
            and core_mappings_equal({loop: slots[a]}, {loop: slots[first]}, n)
            for a in axes
        )
        split = (
            physical_splits[first]
            if same
            else math.prod(physical_splits[a] for a in axes)
        )
        splits[loop] = split
        if len(axes) == 1:
            stick = len(device_size) - 1
            if (
                first != stick
                and stick not in physical_splits
                and loop in device_coordinates[-1].free_symbols
            ):
                padded = int(device_size[first] * device_size[-1])
                if padded - device_size[-1] < extent <= padded:
                    extent = padded
            if extent > _MAX_EXACT_DIRECT_AXIS_POINTS:
                raise ValueError(
                    f"proof limit: direct axis needs {extent} points; limit is {_MAX_EXACT_DIRECT_AXIS_POINTS}"
                )
        else:
            fused_states += extent + split
            if fused_states > _MAX_EXACT_OWNERSHIP_POINTS:
                raise ValueError(
                    f"proof limit: fused axes need {fused_states} states; limit is {_MAX_EXACT_OWNERSHIP_POINTS}"
                )
        if extent % split:
            raise ValueError(
                f"unsupported ownership input: loop {loop} not divisible by {split}"
            )
        try:
            bounds = _loop_regions(
                extent,
                tuple(
                    device_coordinates[a].xreplace({loop: _LOOP_POINT}) for a in axes
                ),
                tuple(int(device_size[a]) for a in axes),
                split,
            )
        except _EVALUATION_ERRORS as exc:
            raise ValueError(
                f"unsupported ownership evaluation: {type(exc).__name__}: {exc}"
            ) from exc
        widths = [int(device_size[a]) // physical_splits[a] for a in axes]
        signatures = [
            tuple(low // width for (low, _), width in zip(region, widths))
            for region in bounds
        ]
        if any(
            low // width != high // width
            for region in bounds
            for (low, high), width in zip(region, widths)
        ):
            raise ValueError(
                "ownership mismatch: one loop partition crosses physical slices"
            )
        if len(set(signatures)) != split:
            raise ValueError(
                "ownership mismatch: loop partitions do not cover distinct physical slices"
            )
        try:
            table = tuple(signatures.index(tuple(row[a] for a in axes)) for row in rows)
        except ValueError:
            raise ValueError(
                "ownership mismatch: a core owns slices no loop partition covers"
            ) from None
        if set(table) != set(range(split)) or (
            not any_fused and signatures != [(p,) for p in range(split)]
        ):
            raise ValueError(
                "ownership mismatch: loop and physical slices have different core owners"
            )
        expected[loop] = table
        owners[loop] = slots[first]

    if not any_fused:
        return TensorWorkDivision(splits, owners, num_cores=n)
    # The physical slices already determine every loop owner. Search only for
    # the existing supported spelling, never re-prove the access per candidate.
    expected_rows = tuple(
        {loop: table[core] for loop, table in expected.items()} for core in range(n)
    )
    candidate = select_unique_partition_division(
        tuple(loop for loop in iteration_space if loop in splits),
        splits,
        n,
        lambda division: owner_slots(division.core_id_to_work_slice, splits, n)
        == expected_rows,
    )
    if candidate is None:
        raise ValueError("no unique certified canonical mapping for fused ownership")
    return candidate


def materialized_lx_relayouts(
    graph: GraphLowering,
) -> dict[tuple[str, str], tuple[str, LXRelayoutPlan]]:
    return getattr(graph, _REGISTRY, {})


def materialized_lx_relayout_for_destination(
    graph: GraphLowering, destination_name: str
) -> LXRelayoutPlan | None:
    """Return the certified plan which created one destination copy."""

    return next(
        (
            plan
            for copy_name, plan in materialized_lx_relayouts(graph).values()
            if copy_name == destination_name
        ),
        None,
    )


def _discard_lx_relayout_group(graph: GraphLowering, source_name: str) -> set[str]:
    copies = materialized_lx_relayouts(graph)
    removed = set()
    for edge, (copy_name, _) in list(copies.items()):
        if edge[0] == source_name:
            removed.add(copy_name)
            del copies[edge]
    return removed


def _clear_lx_state(layout: FixedTiledLayout) -> None:
    """Clear an LX buffer's placement and physical ownership."""

    layout.allocation.pop("lx", None)
    layout.lx_view = None


def demote_lx_relayout_group(
    graph: GraphLowering, source_name: str, reason: str
) -> None:
    """Remove one relayout group from LX and its materialization registry."""

    names = {source_name, *_discard_lx_relayout_group(graph, source_name)}
    for name in names:
        buffer = graph.try_get_buffer(name)
        if buffer is None:
            continue
        layout = buffer.get_layout()
        if isinstance(layout, FixedTiledLayout):
            _clear_lx_state(layout)
    logger.info("demoted %s out of LX: %s", ", ".join(sorted(names)), reason)


def _core_slices(view: PerCoreView, num_cores: int) -> dict[int, dict[int, int]]:
    if view.num_cores is not None and view.num_cores <= 0:
        raise ValueError(f"physical core count must be positive, got {view.num_cores}")
    if view.num_cores is not None and view.num_cores != num_cores:
        raise ValueError(
            "ownership core count differs from the communication domain: "
            f"{view.num_cores} != {num_cores}"
        )
    rows = owner_slots(dict(view.core_to_slot), dict(view.work_slice_dims), num_cores)
    return dict(enumerate(rows))


def movement_supported(source, destination, source_num_cores, destination_num_cores):
    """The original relayout: two complete, distinct partitions of the same cores."""
    if source_num_cores != destination_num_cores or source_num_cores <= 0:
        return False
    if source.same_partition(destination):
        return False
    for view in (source, destination):
        if math.prod(dict(view.work_slice_dims).values()) != source_num_cores:
            return False
        rows = _core_slices(view, source_num_cores)
        if (
            len({tuple(sorted(row.items())) for row in rows.values()})
            != source_num_cores
        ):
            return False
    return True


def _single_write(op: ComputedBuffer, name: str) -> MemoryDep | None:
    writes = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == name
    ]
    if len(writes) != 1 or writes[0].is_indirect():
        return None
    return writes[0]


def _is_activation_source(
    graph: GraphLowering, operations: dict[str, Operation], op: Operation
) -> bool:
    """Exclude restickified graph inputs and weights from activation relayout."""

    return not is_restickify_op(op, graph) or any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def _unsupported_relayout_transition_reason(
    source_work_division: TensorWorkDivision,
    destination_work_division: TensorWorkDivision,
) -> str | None:
    """Reject ownership changes that the identity-copy emitter cannot represent.

    ``op_spec.is_lx_relayout_identity`` recognizes a physical shuffle only
    when the two tensor work divisions differ. If distinct per-core views
    project to the same work division, codegen would lower the materialized
    copy as an ordinary identity and silently omit the required cross-core
    movement. Dropping the optimization keeps consumers on the original,
    correctly addressed buffer.
    """

    if source_work_division.same_ownership(destination_work_division):
        return (
            "cannot emit: distinct physical ownerships collapse to the same "
            "logical work division"
        )
    return None


def collect_lx_relayout_plans(
    graph: GraphLowering,
) -> list[LXRelayoutPlan]:
    if not config.lx_planner_relayout or config.ktir_emitter:
        return []
    if materialized_lx_relayouts(graph):
        raise RuntimeError("LX relayout planning requires an unmaterialized graph")

    cache: dict = {}
    operations = {op.get_name(): op for op in graph.operations}
    reads: dict[str, list[tuple[Operation, MemoryDep]]] = {}
    for consumer in graph.operations:
        deps = [d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]
        for dep in deps:
            reads.setdefault(dep.name, []).append((consumer, dep))

    result: list[LXRelayoutPlan] = []
    for source_name, consumer_reads in reads.items():
        producer = operations.get(source_name)
        if (
            not isinstance(producer, ComputedBuffer)
            or not isinstance(producer.layout, FixedTiledLayout)
            or (write := _single_write(producer, source_name)) is None
        ):
            continue
        source_view, partial, representable = _per_core_view_on_buf(
            producer,
            write,
            source_name,
            cache,
        )
        source_num_cores = _op_num_cores(producer)
        if (
            source_view is None
            or partial
            or not representable
            or source_view.num_cores != source_num_cores
        ):
            continue

        # Activation eligibility belongs to the producer, not to an individual
        # edge. Never relayout a restickified graph input or weight.
        if not _is_activation_source(graph, operations, producer):
            continue

        producer_coordinates = try_device_coordinates(
            producer.layout.device_layout, write, None
        )
        if producer_coordinates is None:
            logger.debug(
                "rejected LX relayout candidate source=%s: "
                "cannot represent: producer coordinates are unavailable",
                source_name,
            )
            continue
        try:
            work_division_from_view(
                source_view,
                producer.layout.device_layout.device_size,
                producer_coordinates,
                iteration_space_from_op(producer),
            )
        except ValueError as exc:
            logger.debug(
                "rejected LX relayout candidate source=%s: "
                "cannot represent: source ownership cannot be projected to producer: %s",
                source_name,
                exc,
            )
            continue

        # Relayout copies sharing one source are allocated and materialized as
        # one atomic group. Any unsupported consumer therefore rejects the
        # group; supported consumers keep using the original buffer instead.
        transfers = []
        seen_consumers = set()
        rejection_reason = None
        for consumer, dep in consumer_reads:
            consumer_name = consumer.get_name()
            if consumer_name in seen_consumers:
                rejection_reason = (
                    "cannot emit: consumer reads the source more than once"
                )
                break
            if not isinstance(consumer, ComputedBuffer) or isinstance(
                consumer.layout, MutationLayoutSHOULDREMOVE
            ):
                rejection_reason = (
                    "cannot emit: consumer is not a supported computed buffer"
                )
                break
            seen_consumers.add(consumer_name)
            deps = [
                d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)
            ]
            if any(d.is_indirect() for d in deps):
                rejection_reason = "cannot emit: consumer uses indirect access"
                break
            view, consumer_partial, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            consumer_num_cores = _op_num_cores(consumer)
            if view is None or consumer_partial or not representable:
                rejection_reason = (
                    "cannot represent: consumer ownership is partial or unrepresentable"
                )
                break
            if consumer_num_cores != source_num_cores:
                rejection_reason = "cannot emit: different core counts"
                break
            consumer_coordinates = try_device_coordinates(
                producer.layout.device_layout, dep, None
            )
            if consumer_coordinates is None:
                rejection_reason = (
                    "cannot represent: consumer coordinates are unavailable"
                )
                break
            consumer_space = iteration_space_from_op(consumer)
            if view.same_partition(source_view):
                continue
            is_matmul = _is_matmul_op(consumer)
            if is_matmul and len(deps) != 2:
                rejection_reason = (
                    "cannot emit: matmul consumer does not have two inputs"
                )
                break
            if not is_matmul and not isinstance(consumer.data, Pointwise):
                rejection_reason = (
                    "cannot emit: consumer is neither pointwise nor matmul"
                )
                break

            failure = "cannot emit: unsupported ownership transfer"

            try:
                supported = movement_supported(
                    source_view, view, source_num_cores, consumer_num_cores
                )
            except (TypeError, ValueError) as exc:
                rejection_reason = (
                    f"cannot represent: invalid ownership partition: {exc}"
                )
                break
            if not supported:
                rejection_reason = failure
                break
            transfers.append(
                (consumer_name, consumer_coordinates, consumer_space, view)
            )

        # Reuse the ownership comparison and preserve first-consumer order.
        destinations: list[tuple[PerCoreView, list[str]]] = []
        if rejection_reason is None:
            for (
                consumer_name,
                consumer_coordinates,
                consumer_space,
                destination_view,
            ) in transfers:
                try:
                    source_work_division = work_division_from_view(
                        source_view,
                        producer.layout.device_layout.device_size,
                        consumer_coordinates,
                        consumer_space,
                    )
                except ValueError as exc:
                    rejection_reason = (
                        "cannot represent: source ownership cannot be projected "
                        f"to consumer: {exc}"
                    )
                    break
                if source_work_division is None:
                    raise RuntimeError(
                        "LX relayout source lost its certified physical ownership"
                    )
                try:
                    destination_work_division = work_division_from_view(
                        destination_view,
                        producer.layout.device_layout.device_size,
                        consumer_coordinates,
                        consumer_space,
                    )
                except ValueError as exc:
                    rejection_reason = (
                        "cannot represent: destination ownership cannot be projected "
                        f"to consumer: {exc}"
                    )
                    break
                if destination_work_division is None:
                    raise RuntimeError(
                        "LX relayout destination lost its certified physical ownership"
                    )
                if reason := _unsupported_relayout_transition_reason(
                    source_work_division, destination_work_division
                ):
                    rejection_reason = reason
                    break
                for group_view, consumers in destinations:
                    if group_view.same_partition(destination_view):
                        consumers.append(consumer_name)
                        break
                else:
                    destinations.append((destination_view, [consumer_name]))

        if rejection_reason is None:
            result.extend(
                LXRelayoutPlan(
                    source_name=source_name,
                    consumer_names=tuple(consumer_names),
                    source_view=source_view,
                    destination_view=destination_view,
                    num_cores=source_num_cores,
                )
                for (
                    destination_view,
                    consumer_names,
                ) in destinations
            )
        if rejection_reason is not None:
            logger.debug(
                "rejected LX relayout candidate source=%s consumers=%s: %s",
                source_name,
                tuple(consumer.get_name() for consumer, _ in consumer_reads),
                rejection_reason,
            )
    return result


def materialize_lx_relayouts(graph: GraphLowering, plans: list[LXRelayoutPlan]) -> None:
    if not plans:
        if materialized_lx_relayouts(graph):
            raise RuntimeError("LX relayouts were already materialized")
        return
    from .graph_editor import GraphEditor

    copies = materialized_lx_relayouts(graph)
    if copies:
        raise RuntimeError("LX relayouts were already materialized")
    editor = GraphEditor(graph)
    setattr(graph, _REGISTRY, copies)
    for plan in plans:
        if plan.source_address is None or plan.destination_address is None:
            raise RuntimeError("LX relayout plan is missing an allocated address")
        source = cast(ComputedBuffer, graph.get_buffer(plan.source_name))
        if plan.source_view.same_partition(plan.destination_view):
            raise RuntimeError("LX relayout plan has identical source and destination")
        source_layout = cast(FixedTiledLayout, source.layout)
        if (
            source_layout.allocation.get("lx") != plan.source_address
            or source_layout.lx_view is None
            or not source_layout.lx_view.same_partition(plan.source_view)
        ):
            raise RuntimeError("placed relayout source disagrees with its plan")
        consumers = [
            cast(ComputedBuffer, graph.get_buffer(name)) for name in plan.consumer_names
        ]
        copy = editor.insert_clone_before_consumers(
            source,
            consumers,
            lx_view=plan.destination_view,
        )
        copies[plan.edge] = (copy.get_name(), plan)

        copy_layout = cast(FixedTiledLayout, copy.layout)
        copy_layout.allocation["lx"] = plan.destination_address
        copy_layout.lx_view = plan.destination_view
        logger.debug(
            "accepted LX relayout %s -> %s: source=%s@%d destination=%s@%d",
            source.get_name(),
            copy.get_name(),
            plan.source_view,
            plan.source_address,
            plan.destination_view,
            plan.destination_address,
        )
