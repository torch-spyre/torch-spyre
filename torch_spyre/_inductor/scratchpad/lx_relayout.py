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
_FINAL_VIEWS = "_spyre_lx_final_views"


@dataclasses.dataclass(frozen=True)
class RelayoutDimension:
    """One device dimension's source-to-destination partition geometry."""

    device_dim: int
    source_split: int
    destination_split: int
    group_count: int
    group_size: int
    multiplicity: int
    ordering_tag: Literal["contiguous_groups"]


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_names: tuple[str, ...]
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    kind: Literal["shuffle", "gather", "broadcast"] = "shuffle"
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


@dataclasses.dataclass(frozen=True)
class FinalLXView:
    """One tensor's final emitted physical shape and ownership."""

    ownership: PerCoreView
    device_size: tuple[int, ...]
    slice_shape: tuple[int, ...]
    physical_span_bytes: int | None


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


def final_lx_views(graph: GraphLowering) -> dict[tuple[str, str], FinalLXView]:
    """Final post-alignment physical views accepted by the graph-wide gate."""

    return getattr(graph, _FINAL_VIEWS, {})


def set_final_lx_views(
    graph: GraphLowering, views: dict[tuple[str, str], FinalLXView]
) -> None:
    setattr(graph, _FINAL_VIEWS, views)


def _discard_lx_relayout_group(graph: GraphLowering, source_name: str) -> set[str]:
    copies = materialized_lx_relayouts(graph)
    removed = set()
    for edge, (copy_name, _) in list(copies.items()):
        if edge[0] == source_name:
            removed.add(copy_name)
            del copies[edge]
    cached = final_lx_views(graph)
    for key in list(cached):
        if key[1] == source_name or key[1] in removed:
            del cached[key]
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


def view_from_work_division(
    division: TensorWorkDivision,
    device_coordinates: Sequence[sympy.Expr],
    iteration_space: Mapping[sympy.Symbol, tuple[sympy.Expr, int]],
) -> PerCoreView:
    """Convert final loop-symbol ownership back to a physical device view."""

    split_by_device_dim: dict[int, int] = {}
    slot_by_device_dim: dict[int, sympy.Expr] = {}
    for loop_dim, split in division.work_slices.items():
        matches = [
            device_dim
            for device_dim, coordinate in enumerate(device_coordinates)
            if loop_dim in coordinate.free_symbols
        ]
        # An operation may split a loop that this tensor broadcasts across
        # (for example the query dimension of a BMM input). That loop chooses
        # execution cores, but it does not further partition this tensor.
        if not matches:
            continue
        if len(matches) > 1:
            extent = int(iteration_space[loop_dim][0])
            slice_extent = math.ceil(extent / int(split))
            physical_matches = []
            for device_dim in matches:
                coordinate = device_coordinates[device_dim]
                other_dims = coordinate.free_symbols - {loop_dim}
                coordinate = coordinate.xreplace({dim: 0 for dim in other_dims})
                values = []
                for slot in range(int(split)):
                    start = slot * slice_extent
                    end = min(extent, start + slice_extent) - 1
                    if start >= extent:
                        break
                    first = sympy.simplify(coordinate.xreplace({loop_dim: start}))
                    last = sympy.simplify(coordinate.xreplace({loop_dim: end}))
                    if first != last:
                        break
                    values.append(first)
                if len(values) == int(split) and len(set(values)) == int(split):
                    physical_matches.append(device_dim)
            matches = physical_matches
        if len(matches) != 1:
            raise ValueError(
                f"cannot map final loop {loop_dim} to one device dimension"
            )
        device_dim = matches[0]
        slot = sympy.sympify(division.core_id_to_work_slice[loop_dim])
        previous = (
            split_by_device_dim.get(device_dim),
            slot_by_device_dim.get(device_dim),
        )
        if previous[0] is not None and previous != (int(split), slot):
            raise ValueError(
                f"conflicting final ownership on device dimension {device_dim}"
            )
        split_by_device_dim[device_dim] = int(split)
        slot_by_device_dim[device_dim] = slot
    return PerCoreView(
        tuple(sorted(split_by_device_dim.items())),
        tuple(sorted(slot_by_device_dim.items())),
        num_cores=division.num_cores,
    )


def final_view_from_work_division(
    division: TensorWorkDivision,
    device_size: Sequence[int],
    device_coordinates: Sequence[sympy.Expr],
    iteration_space: Mapping[sympy.Symbol, tuple[sympy.Expr, int]],
    *,
    physical_span_bytes: int | None,
) -> FinalLXView:
    """Build the view described by a finalized tensor argument.

    Alignment may factor or insert descriptor dimensions, so its ``device_size``
    cannot be combined with the original layout strides.  The exact physical
    span is therefore carried from the committed physical partition while the
    final ownership and slice shape are derived from the aligned descriptor.
    """

    size = tuple(int(extent) for extent in device_size)
    ownership = view_from_work_division(division, device_coordinates, iteration_space)
    kept_dims = [dim for dim, extent in enumerate(size) if extent != 1]
    canonical_dim = {dim: index for index, dim in enumerate(kept_dims)}
    ownership = PerCoreView(
        tuple((canonical_dim[dim], split) for dim, split in ownership.work_slice_dims),
        tuple((canonical_dim[dim], slot) for dim, slot in ownership.core_to_slot),
        num_cores=ownership.num_cores,
    )
    canonical_size = tuple(size[dim] for dim in kept_dims)
    splits = dict(ownership.work_slice_dims)
    extents = tuple(
        math.ceil(extent / int(splits.get(dim, 1)))
        for dim, extent in enumerate(canonical_size)
    )
    slice_shape = tuple(sorted(extents[:-1])) + (extents[-1],)
    return FinalLXView(
        ownership=ownership,
        device_size=canonical_size,
        slice_shape=slice_shape,
        physical_span_bytes=physical_span_bytes,
    )


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
                group_count=destination_split,
                group_size=source_split // destination_split,
                multiplicity=source_split // destination_split,
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


def _grouped_broadcast_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Classify a complete source partition spread over more physical cores."""

    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    if (
        source_num_cores >= destination_num_cores
        or math.prod(source_splits.values()) != source_num_cores
        or destination_num_cores % math.prod(destination_splits.values())
    ):
        return None

    geometry = []
    for dim in dict.fromkeys((*source_splits, *destination_splits)):
        source_split = source_splits.get(dim, 1)
        destination_split = destination_splits.get(dim, 1)
        if destination_split < source_split or destination_split % source_split:
            return None
        geometry.append(
            RelayoutDimension(
                device_dim=dim,
                source_split=source_split,
                destination_split=destination_split,
                group_count=source_split,
                group_size=destination_split // source_split,
                multiplicity=destination_split // source_split,
                ordering_tag="contiguous_groups",
            )
        )
    grouped_source = _view_from_splits(source_splits, source_num_cores)
    grouped_destination = _view_from_splits(destination_splits, destination_num_cores)
    if not _compatible_partitions(
        grouped_source,
        grouped_destination,
        source_num_cores,
        destination_num_cores,
    ):
        return None
    return grouped_source, grouped_destination, tuple(geometry)


def partition_footprint(layout: FixedTiledLayout, view: PerCoreView) -> int:
    device_layout = layout.device_layout
    return partition_physical_span_bytes(
        tuple(int(size) for size in device_layout.device_size),
        tuple(int(stride) for stride in device_layout.stride_map),
        int(device_layout.elems_per_stick()),
        dict(view.work_slice_dims),
    )


def _geometry_topology(
    geometry: Sequence[RelayoutDimension],
) -> tuple[tuple[int, int, int, int, int, str], ...]:
    """Return the layout-normalization-invariant collective geometry.

    Device-dimension numbers may change when a tensor layout is normalized.
    The split contraction and grouping structure must not.
    """

    return tuple(
        sorted(
            (
                item.source_split,
                item.destination_split,
                item.group_count,
                item.group_size,
                item.multiplicity,
                item.ordering_tag,
            )
            for item in geometry
        )
    )


def _overlap(a: int, an: int, b: int, bn: int) -> bool:
    return a * bn < (b + 1) * an and b * an < (a + 1) * bn


def _compatible_partitions(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int | None = None,
) -> bool:
    """Whether every destination receives a uniform, complete partition."""

    destination_num_cores = destination_num_cores or source_num_cores
    source_map = _core_slices(source, source_num_cores)
    destination_map = _core_slices(destination, destination_num_cores)
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
    fanout = [sum(src == core for src, _ in edges) for core in range(source_num_cores)]
    fanin = [
        sum(dst == core for _, dst in edges) for core in range(destination_num_cores)
    ]
    if not edges or len(set(fanout)) != 1 or len(set(fanin)) != 1:
        return False
    source_owners = len({tuple(sorted(row.items())) for row in source_map.values()})
    destination_owners = len(
        {tuple(sorted(row.items())) for row in destination_map.values()}
    )
    if (
        source_owners != source_num_cores
        or math.prod(source_splits.values()) != source_num_cores
    ):
        return False
    destination_slices = math.prod(destination_splits.values())
    if (
        destination_owners != destination_slices
        or destination_num_cores % destination_slices
    ):
        return False
    if source_num_cores != destination_num_cores:
        return fanout[0] == destination_num_cores // source_num_cores and fanin[0] == 1
    multiplicity = source_num_cores // destination_slices
    return multiplicity == 1 or (fanout[0] == multiplicity and fanin[0] == multiplicity)


def validate_final_views(
    graph: GraphLowering,
    plan: LXRelayoutPlan,
    views: dict[tuple[str, str], FinalLXView],
    *,
    destination_name: str | None = None,
) -> str | None:
    """Validate one complete relayout group after scheduler alignment preview."""

    destination_name = destination_name or plan.destination_name
    source_views = {
        view
        for (_, buffer_name), view in views.items()
        if buffer_name == plan.source_name
    }
    destination_views = {
        view
        for (_, buffer_name), view in views.items()
        if buffer_name == destination_name
    }
    if len(source_views) != 1:
        return f"source users derived {len(source_views)} final views"
    if len(destination_views) != 1:
        return f"destination users derived {len(destination_views)} final views"
    source = next(iter(source_views))
    destination = next(iter(destination_views))

    source_num_cores = plan.source_view.num_cores or plan.num_cores
    destination_num_cores = plan.destination_view.num_cores or plan.num_cores
    if (
        source.ownership.num_cores != source_num_cores
        or destination.ownership.num_cores != destination_num_cores
    ):
        return "final view physical core count changed"
    if source == destination:
        return "final source and destination views no longer require a relayout"
    if not _compatible_partitions(
        source.ownership,
        destination.ownership,
        source_num_cores,
        destination_num_cores,
    ):
        return "final source and destination partitions are not a complete transfer"

    if plan.kind == "gather":
        classified = _grouped_gather_geometry(
            source.ownership,
            destination.ownership,
            source_num_cores,
        )
        if classified is None:
            return "final views are not a grouped gather"
        geometry = classified[2]
        if _geometry_topology(geometry) != _geometry_topology(plan.group_geometry):
            return f"final grouped-gather geometry changed: {geometry}"
    elif plan.kind == "broadcast":
        classified = _grouped_broadcast_geometry(
            source.ownership,
            destination.ownership,
            source_num_cores,
            destination_num_cores,
        )
        if classified is None:
            return "final views are not a grouped broadcast"
        geometry = classified[2]
        if _geometry_topology(geometry) != _geometry_topology(plan.group_geometry):
            return f"final grouped-broadcast geometry changed: {geometry}"
    elif plan.kind != "shuffle":
        return f"unsupported relayout kind {plan.kind}"

    for name, view in (
        (plan.source_name, source),
        (destination_name, destination),
    ):
        buffer = graph.try_get_buffer(name)
        if buffer is None:
            return f"missing final buffer {name}"
        layout = buffer.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return f"final buffer {name} has no fixed tiled layout"
        planned_view = (
            plan.source_view if name == plan.source_name else plan.destination_view
        )
        planned_size = tuple(int(extent) for extent in layout.device_layout.device_size)
        planned_splits = dict(planned_view.work_slice_dims)
        planned_extents = tuple(
            math.ceil(extent / int(planned_splits.get(dim, 1)))
            for dim, extent in enumerate(planned_size)
            if extent != 1
        )
        planned_shape = tuple(sorted(planned_extents[:-1])) + (planned_extents[-1],)
        if view.slice_shape != planned_shape:
            return (
                f"final {name} slice shape {view.slice_shape} differs from planned "
                f"{planned_shape}"
            )
        if view.physical_span_bytes is None:
            return f"final {name} has no committed physical span"
        if view.physical_span_bytes > plan.max_footprint_bytes:
            return (
                f"final {name} physical span {view.physical_span_bytes} "
                f"exceeds planned "
                f"{plan.max_footprint_bytes}"
            )

    if plan.source_address is None or plan.destination_address is None:
        return "relayout group has no committed LX addresses"
    # The allocator validates disjointness using each concrete buffer size.
    # Repeating that check with this edge's maximum bound is incorrect for a
    # source shared by differently sized destinations.
    return None


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
        source_num_cores = _op_num_cores(producer)
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
            consumer_num_cores = _op_num_cores(consumer)
            if view is None or consumer_partial or not representable:
                rejection_reason = "consumer ownership is partial or unrepresentable"
                break
            if consumer_num_cores < source_num_cores:
                rejection_reason = "consumer uses fewer physical cores than producer"
                break
            if consumer_num_cores > source_num_cores and not _is_matmul_op(consumer):
                rejection_reason = "grouped broadcast requires a matmul consumer"
                break
            if (
                consumer_num_cores > source_num_cores
                and consumer_num_cores != config.sencores
            ):
                rejection_reason = "grouped broadcast must target all compute cores"
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
                    consumer_num_cores,
                )
            )

        grouped_source = None
        grouped_destinations = {}
        grouped_geometry = {}
        grouped_kinds = {}
        if rejection_reason is None:
            for (
                consumer_name,
                consumer,
                _,
                view,
                _,
                _,
                consumer_num_cores,
            ) in consumer_views:
                if consumer_num_cores > source_num_cores:
                    grouped = _grouped_broadcast_geometry(
                        source_view,
                        view,
                        source_num_cores,
                        consumer_num_cores,
                    )
                    if grouped is None:
                        rejection_reason = (
                            "grouped destination does not evenly replicate the source"
                        )
                        break
                    candidate_source, destination, geometry = grouped
                    if (
                        grouped_source is not None
                        and candidate_source != grouped_source
                    ):
                        rejection_reason = (
                            "consumers require different grouped source geometry"
                        )
                        break
                    grouped_source = candidate_source
                    grouped_destinations[consumer_name] = destination
                    grouped_geometry[consumer_name] = geometry
                    grouped_kinds[consumer_name] = "broadcast"
                    continue
                destination_owners = math.prod(dict(view.work_slice_dims).values())
                if destination_owners >= source_num_cores:
                    continue
                if not _is_matmul_op(consumer):
                    rejection_reason = "grouped gather requires a matmul consumer"
                    break
                grouped = _grouped_gather_geometry(source_view, view, source_num_cores)
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
                grouped_kinds[consumer_name] = "gather"

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

        plans_by_destination: dict[tuple, list[str]] = {}
        if rejection_reason is None:
            source_geometry = source_view.work_slice_dims
            for (
                consumer_name,
                consumer,
                deps,
                raw_view,
                consumer_coordinates,
                consumer_symbols,
                consumer_num_cores,
            ) in consumer_views:
                destination_view = grouped_destinations.get(consumer_name, raw_view)
                if (
                    raw_view.work_slice_dims == source_geometry
                    and consumer_num_cores == source_num_cores
                ):
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
                if not _compatible_partitions(
                    source_view,
                    destination_view,
                    source_num_cores,
                    consumer_num_cores,
                ):
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
                kind = grouped_kinds.get(consumer_name, "shuffle")
                geometry = grouped_geometry.get(consumer_name, ())
                footprint = max(
                    partition_footprint(producer.layout, source_view),
                    partition_footprint(producer.layout, destination_view),
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
                    num_cores=source_num_cores,
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
