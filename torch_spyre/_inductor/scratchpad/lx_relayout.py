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
from collections.abc import Sequence
from typing import Literal, cast

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
from ..core_mapping import derive_partition_mapping, partition_physical_span_bytes
from ..ir import FixedTiledLayout
from ..logging_utils import get_inductor_logger
from ..op_spec import TensorWorkDivision
from ..pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    iteration_space_from_op,
    op_short_name,
    op_read_writes,
    try_device_coordinates,
)
from .utils import _op_num_cores

logger = get_inductor_logger("lx_relayout")
_DESTINATION_PREFIX = "__spyre_lx_relayout__"
_REGISTRY = "_spyre_lx_relayout_copies"


@dataclasses.dataclass(frozen=True)
class RelayoutDimension:
    """One device dimension's source-to-destination partition geometry."""

    device_dim: int
    source_split: int
    destination_split: int
    ordering_tag: Literal["contiguous_groups"]


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_names: tuple[str, ...]
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    kind: Literal["shuffle", "gather"] = "shuffle"
    group_geometry: tuple[RelayoutDimension, ...] = ()
    max_footprint_bytes: int = 0
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
    device_coordinates: Sequence[sympy.Expr],
    iteration_symbols: Sequence[sympy.Symbol],
) -> TensorWorkDivision | None:
    """Project physical per-core ownership into operation-loop symbols."""

    if view is None:
        return None
    if view.num_cores is None:
        raise ValueError("LX ownership must carry its physical core domain")
    loop_symbols = set(iteration_symbols)
    splits: dict[sympy.Symbol, int] = {}
    core_map: dict[sympy.Symbol, sympy.Expr] = {}
    slots = dict(view.core_to_slot)
    for device_dim, split in view.work_slice_dims:
        if device_dim >= len(device_coordinates):
            raise ValueError(f"missing device coordinate {device_dim}")
        matches = device_coordinates[device_dim].free_symbols & loop_symbols
        if len(matches) != 1:
            raise ValueError(f"cannot map device dimension {device_dim} to one loop")
        dim = next(iter(matches))
        slot = sympy.sympify(slots[device_dim])
        if dim in splits and (splits[dim], core_map[dim]) != (split, slot):
            raise ValueError(f"conflicting ownership for loop {dim}")
        splits[dim] = split
        core_map[dim] = slot
    return TensorWorkDivision(splits, core_map, num_cores=view.num_cores)


def materialized_lx_relayouts(
    graph: GraphLowering,
) -> dict[tuple[str, str], tuple[str, LXRelayoutPlan]]:
    return getattr(graph, _REGISTRY, {})


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
    core_id = sympy.Symbol("core_id")
    splits = dict(view.work_slice_dims)
    slots = dict(view.core_to_slot)
    result = {}
    for core in range(num_cores):
        row = {}
        for dim, split in splits.items():
            value = sympy.sympify(slots[dim]).subs(core_id, core)
            assert not value.free_symbols, f"non-concrete owner slot {value}"
            slot = int(value)
            assert 0 <= slot < split, f"owner slot {slot} outside split {split}"
            row[dim] = slot
        result[core] = row
    return result


def _view_from_splits(
    split_by_device_dim: dict[int, int], num_cores: int
) -> PerCoreView:
    """Build v1 planning ownership with the shared late-mapping formula."""

    dims = tuple(sympy.Symbol(f"device_dim_{dim}") for dim in split_by_device_dim)
    mapping = derive_partition_mapping(
        dims,
        tuple(split_by_device_dim.values()),
        num_cores,
    )
    device_dim_by_symbol = dict(zip(dims, split_by_device_dim))
    return PerCoreView(
        tuple(split_by_device_dim.items()),
        tuple(
            (device_dim_by_symbol[dim], expression)
            for dim, expression in mapping.items()
            if split_by_device_dim[device_dim_by_symbol[dim]] > 1
        ),
        num_cores=num_cores,
    )


def _grouped_gather_geometry(
    source: PerCoreView, destination: PerCoreView, num_cores: int
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Classify a full source partition contracting into repeated owners."""

    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    if math.prod(source_splits.values()) != num_cores:
        return None
    destination_owners = math.prod(destination_splits.values())
    if not 0 < destination_owners < num_cores:
        return None

    dimensions = tuple(dict.fromkeys((*source_splits, *destination_splits)))
    geometry = []
    for dim in dimensions:
        source_split = source_splits.get(dim, 1)
        destination_split = destination_splits.get(dim, 1)
        if source_split < destination_split or source_split % destination_split:
            return None
        geometry.append(
            RelayoutDimension(
                device_dim=dim,
                source_split=source_split,
                destination_split=destination_split,
                ordering_tag="contiguous_groups",
            )
        )

    if (
        destination_owners
        * math.prod(item.source_split // item.destination_split for item in geometry)
        != num_cores
    ):
        return None
    grouped_source = _view_from_splits(source_splits, num_cores)
    grouped_destination = _view_from_splits(destination_splits, num_cores)
    if not _compatible_partitions(grouped_source, grouped_destination, num_cores):
        return None
    return grouped_source, grouped_destination, tuple(geometry)


def _partition_footprint(layout: FixedTiledLayout, view: PerCoreView) -> int:
    device_layout = layout.device_layout
    return partition_physical_span_bytes(
        tuple(int(size) for size in device_layout.device_size),
        tuple(int(stride) for stride in device_layout.stride_map),
        int(device_layout.elems_per_stick()),
        dict(view.work_slice_dims),
    )


def _overlap(a: int, an: int, b: int, bn: int) -> bool:
    return a * bn < (b + 1) * an and b * an < (a + 1) * bn


def _compatible_partitions(
    source: PerCoreView, destination: PerCoreView, num_cores: int
) -> bool:
    """Whether every destination receives a uniform, complete partition."""

    source_map = _core_slices(source, num_cores)
    destination_map = _core_slices(destination, num_cores)
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    dims = set(source_splits) | set(destination_splits)
    edges = {
        (s_core, d_core)
        for s_core, s_slice in source_map.items()
        for d_core, d_slice in destination_map.items()
        if all(
            _overlap(
                s_slice.get(dim, 0),
                source_splits.get(dim, 1),
                d_slice.get(dim, 0),
                destination_splits.get(dim, 1),
            )
            for dim in dims
        )
    }
    fanout = [sum(src == core for src, _ in edges) for core in range(num_cores)]
    fanin = [sum(dst == core for _, dst in edges) for core in range(num_cores)]
    if not edges or len(set(fanout)) != 1 or len(set(fanin)) != 1:
        return False
    source_owners = len({tuple(sorted(row.items())) for row in source_map.values()})
    destination_owners = len(
        {tuple(sorted(row.items())) for row in destination_map.values()}
    )
    if source_owners != num_cores or math.prod(source_splits.values()) != num_cores:
        return False
    destination_slices = math.prod(destination_splits.values())
    if destination_owners != destination_slices or num_cores % destination_slices:
        return False
    multiplicity = num_cores // destination_slices
    return multiplicity == 1 or (fanout[0] == multiplicity and fanin[0] == multiplicity)


def _single_write(op: ComputedBuffer, name: str) -> MemoryDep | None:
    writes = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == name
    ]
    if len(writes) != 1 or writes[0].is_indirect():
        return None
    return writes[0]


def _is_activation_source(operations: dict[str, Operation], op: Operation) -> bool:
    """Exclude restickified graph inputs and weights from activation relayout."""

    return op_short_name(op) != "restickify" or any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def _unsupported_relayout_transition_reason(
    source_work_division: TensorWorkDivision,
    destination_work_division: TensorWorkDivision,
) -> str | None:
    """Reject ownership changes that the identity-copy emitter cannot represent.

    ``op_spec.is_lx_relayout_identity`` recognizes a physical reshuffle only
    when the two tensor work divisions differ. If distinct per-core views
    project to the same work division, codegen would lower the materialized
    copy as an ordinary identity and silently omit the required cross-core
    movement. Dropping the optimization keeps consumers on the original,
    correctly addressed buffer.
    """

    if source_work_division == destination_work_division:
        return "distinct physical ownerships collapse to the same logical work division"
    return None


def collect_lx_relayout_plans(graph: GraphLowering) -> list[LXRelayoutPlan]:
    if not config.lx_planner_relayout or config.ktir_emitter:
        return []
    assert not materialized_lx_relayouts(graph), (
        "LX relayout planning requires an unmaterialized graph"
    )

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
            producer, write, source_name, cache
        )
        num_cores = _op_num_cores(producer)
        if source_view is None or partial or not representable:
            continue

        # Activation eligibility belongs to the producer, not to an individual
        # edge. Never relayout a restickified graph input or weight.
        if not _is_activation_source(operations, producer):
            continue

        # Relayout copies sharing one source are allocated and materialized as
        # one atomic group. Any unsupported consumer therefore rejects the
        # group; supported consumers keep using the original buffer instead.
        consumer_views = []
        seen_consumers = set()
        rejection_reason = None
        for consumer, dep in consumer_reads:
            consumer_name = consumer.get_name()
            if consumer_name in seen_consumers:
                rejection_reason = "consumer reads the source more than once"
                break
            if not isinstance(consumer, ComputedBuffer) or isinstance(
                consumer.layout, MutationLayoutSHOULDREMOVE
            ):
                rejection_reason = "consumer is not a supported computed buffer"
                break
            seen_consumers.add(consumer_name)
            deps = [
                d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)
            ]
            if any(d.is_indirect() for d in deps):
                rejection_reason = "consumer uses indirect access"
                break
            view, consumer_partial, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            if (
                view is None
                or consumer_partial
                or not representable
                or _op_num_cores(consumer) != num_cores
            ):
                rejection_reason = (
                    "consumer ownership is partial, unrepresentable, or uses a "
                    "different core count"
                )
                break
            consumer_coordinates = try_device_coordinates(
                producer.layout.device_layout, dep, None
            )
            if consumer_coordinates is None:
                rejection_reason = "consumer coordinates are unavailable"
                break
            consumer_symbols = tuple(iteration_space_from_op(consumer))
            consumer_views.append(
                (
                    consumer_name,
                    consumer,
                    deps,
                    view,
                    consumer_coordinates,
                    consumer_symbols,
                )
            )

        grouped_source = None
        grouped_destinations = {}
        grouped_geometry = {}
        if rejection_reason is None:
            for consumer_name, consumer, _, view, _, _ in consumer_views:
                destination_owners = math.prod(dict(view.work_slice_dims).values())
                if destination_owners >= num_cores:
                    continue
                if not _is_matmul_op(consumer):
                    rejection_reason = "grouped gather requires a matmul consumer"
                    break
                grouped = _grouped_gather_geometry(source_view, view, num_cores)
                if grouped is None:
                    rejection_reason = (
                        "grouped destination does not evenly contract the source"
                    )
                    break
                candidate_source, destination, geometry = grouped
                if grouped_source is not None and candidate_source != grouped_source:
                    rejection_reason = (
                        "consumers require different grouped source geometry"
                    )
                    break
                grouped_source = candidate_source
                grouped_destinations[consumer_name] = destination
                grouped_geometry[consumer_name] = geometry

        source_view = grouped_source or source_view
        producer_coordinates = try_device_coordinates(
            producer.layout.device_layout, write, None
        )
        if rejection_reason is None:
            if producer_coordinates is None:
                rejection_reason = "producer coordinates are unavailable"
            else:
                try:
                    work_division_from_view(
                        source_view,
                        producer_coordinates,
                        tuple(iteration_space_from_op(producer)),
                    )
                except ValueError:
                    rejection_reason = (
                        "source ownership cannot be projected to producer"
                    )

        plans_by_destination = {}
        if rejection_reason is None:
            source_geometry = source_view.work_slice_dims
            for (
                consumer_name,
                consumer,
                deps,
                raw_view,
                consumer_coordinates,
                consumer_symbols,
            ) in consumer_views:
                destination_view = grouped_destinations.get(consumer_name, raw_view)
                if raw_view.work_slice_dims == source_geometry:
                    destination_view = source_view
                try:
                    source_work_division = work_division_from_view(
                        source_view, consumer_coordinates, consumer_symbols
                    )
                except ValueError:
                    rejection_reason = (
                        "source ownership cannot be projected to consumer"
                    )
                    break
                assert source_work_division is not None
                if destination_view == source_view:
                    continue
                is_matmul = _is_matmul_op(consumer)
                if is_matmul and len(deps) != 2:
                    rejection_reason = "matmul consumer does not have two inputs"
                    break
                if not is_matmul and not isinstance(consumer.data, Pointwise):
                    rejection_reason = "consumer is neither pointwise nor matmul"
                    break
                if not _compatible_partitions(source_view, destination_view, num_cores):
                    rejection_reason = (
                        "source and destination partitions are incompatible"
                    )
                    break
                try:
                    destination_work_division = work_division_from_view(
                        destination_view, consumer_coordinates, consumer_symbols
                    )
                except ValueError:
                    rejection_reason = (
                        "destination ownership cannot be projected to consumer"
                    )
                    break
                assert destination_work_division is not None
                if reason := _unsupported_relayout_transition_reason(
                    source_work_division, destination_work_division
                ):
                    rejection_reason = reason
                    break
                kind = "gather" if consumer_name in grouped_geometry else "shuffle"
                geometry = grouped_geometry.get(consumer_name, ())
                footprint = max(
                    _partition_footprint(producer.layout, source_view),
                    _partition_footprint(producer.layout, destination_view),
                )
                key = (destination_view, kind, geometry, footprint)
                plans_by_destination.setdefault(key, []).append(consumer_name)

        if rejection_reason is None:
            result.extend(
                LXRelayoutPlan(
                    source_name=source_name,
                    consumer_names=tuple(consumer_names),
                    source_view=source_view,
                    destination_view=destination_view,
                    num_cores=num_cores,
                    kind=kind,
                    group_geometry=geometry,
                    max_footprint_bytes=footprint,
                )
                for (
                    destination_view,
                    kind,
                    geometry,
                    footprint,
                ), consumer_names in plans_by_destination.items()
            )
        if rejection_reason is not None:
            logger.debug(
                "rejected LX relayout candidate source=%s consumer=%s: %s",
                source_name,
                consumer_name,
                rejection_reason,
            )
    return result


def materialize_lx_relayouts(graph: GraphLowering, plans: list[LXRelayoutPlan]) -> None:
    if not plans:
        assert not materialized_lx_relayouts(graph)
        return
    from .graph_editor import GraphEditor

    copies = materialized_lx_relayouts(graph)
    assert not copies, "LX relayouts were already materialized"
    editor = GraphEditor(graph)
    setattr(graph, _REGISTRY, copies)
    for plan in plans:
        source = cast(ComputedBuffer, graph.get_buffer(plan.source_name))
        consumers = [
            cast(ComputedBuffer, graph.get_buffer(name)) for name in plan.consumer_names
        ]
        copy = editor.insert_clone_before_consumers(source, consumers)
        copies[plan.edge] = (copy.get_name(), plan)

        assert plan.source_address is not None and plan.destination_address is not None
        assert plan.source_view != plan.destination_view
        source_layout = cast(FixedTiledLayout, source.layout)
        copy_layout = cast(FixedTiledLayout, copy.layout)
        source_layout.allocation["lx"] = plan.source_address
        copy_layout.allocation["lx"] = plan.destination_address
        source_layout.lx_view = plan.source_view
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
