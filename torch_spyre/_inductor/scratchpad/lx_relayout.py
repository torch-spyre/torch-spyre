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
from torch_spyre._C import ElementArrangement

from .. import config
from ..core_mapping import (
    partition_physical_span_bytes,
    select_partition_division_matching_physical_ownership,
    select_unique_partition_division,
    work_division_matches_physical_ownership,
)
from ..ir import FixedTiledLayout
from ..logging_utils import get_inductor_logger
from ..op_spec import TensorWorkDivision
from ..padding import is_restickify_op
from ..pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    commit_tensor_work_division,
    iteration_space_from_op,
    op_read_writes,
    try_device_coordinates,
)
from .utils import _op_num_cores

logger = get_inductor_logger("lx_relayout")
_DESTINATION_PREFIX = "__spyre_lx_relayout__"
_REGISTRY = "_spyre_lx_relayout_copies"


@dataclasses.dataclass(frozen=True)
class RelayoutDimension:
    """One axis's partition refinement, not the transfer graph's fanout.

    ``multiplicity`` measures this axis's split ratio. A broadcast can also
    add receiving cores without changing any axis splits; its receiver count
    comes from the source/destination owner maps and shared transfer edges.
    """

    device_dim: int
    source_split: int
    destination_split: int
    group_count: int
    group_size: int
    multiplicity: int
    ordering_tag: Literal["uniform_groups", "view_pair"]


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_names: tuple[str, ...]
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    group_geometry: tuple[RelayoutDimension, ...] = ()
    source_footprint_bytes: int = 0
    destination_footprint_bytes: int = 0
    source_address: int | None = None
    destination_address: int | None = None
    # Lowering names are registry keys rather than a closed type. Extension
    # PRs add certifiers without reopening the foundational kind field.
    kind: str = "shuffle"

    @property
    def destination_name(self) -> str:
        return f"{_DESTINATION_PREFIX}:{self.source_name}:{self.consumer_names[0]}"

    @property
    def edge(self) -> tuple[str, str]:
        return self.source_name, self.destination_name


def _owner_slots_equal(
    left: sympy.Expr,
    right: sympy.Expr,
    split: int,
    num_cores: int | None,
) -> bool:
    """Whether two slot formulas select the same slice on every core."""

    return PerCoreView(((0, split),), ((0, left),), num_cores=num_cores).same_partition(
        PerCoreView(((0, split),), ((0, right),), num_cores=num_cores)
    )


def work_division_from_view(
    view: PerCoreView | None,
    device_size: Sequence[int],
    device_coordinates: Sequence[sympy.Expr],
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> TensorWorkDivision | None:
    """Project physical per-core ownership into operation-loop symbols."""

    if view is None:
        return None
    if view.num_cores is None or view.num_cores <= 0:
        raise ValueError("LX ownership must carry its physical core domain")
    loop_symbols = set(iteration_space)
    splits: dict[sympy.Symbol, int] = {}
    core_map: dict[sympy.Symbol, sympy.Expr] = {}
    slots = dict(view.core_to_slot)
    if dict(view.work_slice_dims).keys() != slots.keys():
        raise ValueError("LX ownership split and owner-slot dimensions differ")
    ownership_by_loop: dict[sympy.Symbol, list[tuple[int, int, sympy.Expr]]] = {}
    for device_dim, split in view.work_slice_dims:
        if device_dim >= len(device_coordinates):
            raise ValueError(f"missing device coordinate {device_dim}")
        matches = device_coordinates[device_dim].free_symbols & loop_symbols
        if len(matches) != 1:
            raise ValueError(f"cannot map device dimension {device_dim} to one loop")
        dim = next(iter(matches))
        slot = sympy.sympify(slots[device_dim])
        ownership_by_loop.setdefault(dim, []).append((device_dim, split, slot))

    fused_loops: list[sympy.Symbol] = []
    for dim, ownerships in ownership_by_loop.items():
        _, split, slot = ownerships[0]
        same_ownership = all(
            other_split == split
            and _owner_slots_equal(slot, other_slot, split, view.num_cores)
            for _, other_split, other_slot in ownerships
        )
        if len(ownerships) == 1:
            splits[dim] = split
            core_map[dim] = slot
            continue
        # Several physical axes driven by one loop are a fused-axis claim even
        # when their split and slot formulas look identical.  Equal formulas
        # can still denote diagonal physical regions that no single contiguous
        # loop partition represents, so the exact ownership proof below must
        # judge every multi-axis case.
        fused_loops.append(dim)
        splits[dim] = (
            split if same_ownership else math.prod(item[1] for item in ownerships)
        )

    loop_extents = {
        dim: value[0] if isinstance(value, tuple) else value
        for dim, value in iteration_space.items()
    }
    if fused_loops:
        dimensions = tuple(dim for dim in iteration_space if dim in splits)
        candidate = select_partition_division_matching_physical_ownership(
            dimensions,
            splits,
            loop_extents,
            device_size,
            device_coordinates,
            view.work_slice_dims,
            view.core_to_slot,
            view.num_cores,
        )
        if candidate is None:
            raise ValueError(f"conflicting ownership for loop {fused_loops[0]}")
        return candidate
    candidate = TensorWorkDivision(splits, core_map, num_cores=view.num_cores)
    if not work_division_matches_physical_ownership(
        candidate,
        loop_extents,
        device_size,
        device_coordinates,
        view.work_slice_dims,
        view.core_to_slot,
        view.num_cores,
    ):
        raise ValueError("physical ownership is not exactly expressible in loop space")
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
    if num_cores <= 0:
        raise ValueError(f"physical core count must be positive, got {num_cores}")
    if view.num_cores is not None:
        if view.num_cores <= 0:
            raise ValueError(
                f"physical core count must be positive, got {view.num_cores}"
            )
        if view.num_cores != num_cores:
            raise ValueError(
                "ownership core count differs from the communication domain: "
                f"{view.num_cores} != {num_cores}"
            )
    core_id = sympy.Symbol("core_id")
    splits = dict(view.work_slice_dims)
    slots = dict(view.core_to_slot)
    if splits.keys() != slots.keys():
        raise ValueError(
            "ownership split and owner-slot dimensions differ: "
            f"{sorted(splits)} != {sorted(slots)}"
        )
    result = {}
    for core in range(num_cores):
        row = {}
        for dim, split in splits.items():
            value = sympy.sympify(slots[dim]).subs(core_id, core)
            if value.free_symbols or value.is_integer is not True:
                raise ValueError(f"non-integral owner slot {value} on core {core}")
            slot = int(value)
            if not 0 <= slot < split:
                raise ValueError(
                    f"owner slot {slot} outside split {split} on core {core}"
                )
            row[dim] = slot
        result[core] = row
    return result


def _grouped_gather_geometry(
    source: PerCoreView, destination: PerCoreView, num_cores: int
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Classify a gather ownership pattern.

    Each source core contributes one fragment to a complete destination slice.
    Destination ownership is validated separately from this communication
    class, including cases where several consumers need that slice.
    """

    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    if math.prod(source_splits.values()) != num_cores:
        return None
    destination_owners = math.prod(destination_splits.values())
    if not 0 < destination_owners < num_cores:
        return None

    dimensions = tuple(dict.fromkeys((*source_splits, *destination_splits)))
    for dim in dimensions:
        source_split = source_splits.get(dim, 1)
        destination_split = destination_splits.get(dim, 1)
        if source_split < destination_split or source_split % destination_split:
            return None

    if (
        destination_owners
        * math.prod(
            source_splits.get(dim, 1) // destination_splits.get(dim, 1)
            for dim in dimensions
        )
        != num_cores
    ):
        return None
    # Split counts classify the geometry; the actual views are the ownership
    # contract. Never replace their core order with a reconstructed default.
    if not _compatible_partitions(source, destination, num_cores):
        return None
    geometry = tuple(
        RelayoutDimension(
            device_dim=dim,
            source_split=source_splits.get(dim, 1),
            destination_split=destination_splits.get(dim, 1),
            group_count=destination_splits.get(dim, 1),
            group_size=(source_splits.get(dim, 1) // destination_splits.get(dim, 1)),
            multiplicity=(source_splits.get(dim, 1) // destination_splits.get(dim, 1)),
            # The views carry the physical core order.  This label records only
            # the certified equal-sized grouping, not a second owner mapping.
            ordering_tag="uniform_groups",
        )
        for dim in dimensions
    )
    return source, destination, geometry


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

    dimensions = tuple(dict.fromkeys((*source_splits, *destination_splits)))
    for dim in dimensions:
        source_split = source_splits.get(dim, 1)
        destination_split = destination_splits.get(dim, 1)
        if destination_split < source_split or destination_split % source_split:
            return None
    # Split counts classify the geometry; the actual views are the ownership
    # contract. Never replace their core order with a reconstructed default.
    if not _compatible_partitions(
        source,
        destination,
        source_num_cores,
        destination_num_cores,
    ):
        return None
    geometry = tuple(
        RelayoutDimension(
            device_dim=dim,
            source_split=source_splits.get(dim, 1),
            destination_split=destination_splits.get(dim, 1),
            group_count=source_splits.get(dim, 1),
            group_size=(destination_splits.get(dim, 1) // source_splits.get(dim, 1)),
            multiplicity=(destination_splits.get(dim, 1) // source_splits.get(dim, 1)),
            # The views carry the physical core order.  This label records only
            # the certified equal-sized grouping, not a second owner mapping.
            ordering_tag="uniform_groups",
        )
        for dim in dimensions
    )
    return source, destination, geometry


def partition_footprint(layout: FixedTiledLayout, view: PerCoreView) -> int:
    device_layout = layout.device_layout
    if device_layout.element_arrangement != ElementArrangement.STANDARD:
        raise ValueError("relayout footprint requires standard element arrangement")
    return partition_physical_span_bytes(
        tuple(int(size) for size in device_layout.device_size),
        int(device_layout.elems_per_stick()),
        dict(view.work_slice_dims),
    )


def _overlap(a: int, an: int, b: int, bn: int) -> bool:
    return a * bn < (b + 1) * an and b * an < (a + 1) * bn


def _transfer_edges(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> frozenset[tuple[int, int]]:
    """Derive movement from ownership, independent of collective names.

    An edge ``(s, d)`` exists exactly when source core ``s`` owns tensor
    elements needed by destination core ``d``. Shuffle, gather, broadcast, and
    gather-plus-broadcast are certified shapes of this one fact; they must never
    derive a different movement graph.
    """

    if source.num_cores != source_num_cores:
        raise ValueError(
            "source ownership core domain disagrees with the transfer domain"
        )
    if destination.num_cores != destination_num_cores:
        raise ValueError(
            "destination ownership core domain disagrees with the transfer domain"
        )

    source_map = _core_slices(source, source_num_cores)
    destination_map = _core_slices(destination, destination_num_cores)
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    dimensions = set(source_splits) | set(destination_splits)
    return frozenset(
        (source_core, destination_core)
        for source_core, source_slice in source_map.items()
        for destination_core, destination_slice in destination_map.items()
        if all(
            _overlap(
                source_slice.get(dim, 0),
                source_splits.get(dim, 1),
                destination_slice.get(dim, 0),
                destination_splits.get(dim, 1),
            )
            for dim in dimensions
        )
    )


def _compatible_partitions(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int | None = None,
) -> bool:
    """Whether every destination receives a uniform, complete partition."""

    if destination_num_cores is None:
        destination_num_cores = source_num_cores
    if (
        source_num_cores <= 0
        or destination_num_cores <= 0
        or source.num_cores != source_num_cores
        or destination.num_cores != destination_num_cores
    ):
        return False
    source_map = _core_slices(source, source_num_cores)
    destination_map = _core_slices(destination, destination_num_cores)
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    edges = _transfer_edges(
        source, destination, source_num_cores, destination_num_cores
    )
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


def _gather_broadcast_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Certify gather followed by broadcast from the complete owner maps."""

    if source.num_cores != num_cores or destination.num_cores != num_cores:
        return None
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    source_map = _core_slices(source, num_cores)
    destination_map = _core_slices(destination, num_cores)
    source_owners = {tuple(sorted(row.items())) for row in source_map.values()}
    destination_owners = {
        tuple(sorted(row.items())) for row in destination_map.values()
    }
    destination_slices = math.prod(destination_splits.values())
    if (
        math.prod(source_splits.values()) != num_cores
        or len(source_owners) != num_cores
        or not 0 < destination_slices <= num_cores
        or len(destination_owners) != destination_slices
        or num_cores % destination_slices
    ):
        return None

    dimensions = tuple(dict.fromkeys((*source_splits, *destination_splits)))
    if not any(
        source_splits.get(dim, 1) > destination_splits.get(dim, 1) for dim in dimensions
    ):
        return None
    edges = _transfer_edges(source, destination, num_cores, num_cores)
    fanout = [
        sum(source_core == core for source_core, _ in edges)
        for core in range(num_cores)
    ]
    fanin = [
        sum(destination_core == core for _, destination_core in edges)
        for core in range(num_cores)
    ]
    if (
        not edges
        or 0 in fanout
        or 0 in fanin
        or len(set(fanout)) != 1
        or len(set(fanin)) != 1
    ):
        return None

    destination_counts: dict[tuple[tuple[int, int], ...], int] = {}
    incoming_by_slice: dict[tuple[tuple[int, int], ...], set[int]] = {}
    for destination_core, destination_slice in destination_map.items():
        key = tuple(sorted(destination_slice.items()))
        destination_counts[key] = destination_counts.get(key, 0) + 1
        incoming = {
            source_core
            for source_core, edge_destination in edges
            if edge_destination == destination_core
        }
        previous = incoming_by_slice.setdefault(key, incoming)
        if previous != incoming:
            return None

    receivers_per_slice = num_cores // destination_slices
    # One receiver per slice is the foundational shuffle. This certifier covers
    # only the case where each completed slice is broadcast to several cores.
    if receivers_per_slice <= 1:
        return None
    if set(destination_counts.values()) != {receivers_per_slice}:
        return None
    geometry = tuple(
        RelayoutDimension(
            device_dim=dim,
            source_split=source_splits.get(dim, 1),
            destination_split=destination_splits.get(dim, 1),
            group_count=destination_slices,
            group_size=fanin[0],
            multiplicity=receivers_per_slice,
            ordering_tag="view_pair",
        )
        for dim in dimensions
        if source_splits.get(dim, 1) != destination_splits.get(dim, 1)
    )
    return source, destination, geometry


def _shuffle_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Certify #3439's cross-core shuffle ownership pattern.

    Each core starts with one unique source slice and ends with one unique
    destination slice.  When the two partitions cut different axes, cores
    exchange the pieces required by the destination partition. Ragged payload
    sizes may differ, but the ownership-level class remains a shuffle.
    """

    if source_num_cores != destination_num_cores or source.same_partition(destination):
        return None
    if not _compatible_partitions(
        source, destination, source_num_cores, destination_num_cores
    ):
        return None
    return source, destination, ()


def _gather_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Give gather the common classifier signature."""

    if source_num_cores != destination_num_cores:
        return None
    return _grouped_gather_geometry(source, destination, source_num_cores)


def _gather_broadcast_geometry_adapter(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Give gather-plus-broadcast the common classifier signature."""

    if source_num_cores != destination_num_cores:
        return None
    return _gather_broadcast_geometry(source, destination, source_num_cores)


def _broadcast_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    return _grouped_broadcast_geometry(
        source, destination, source_num_cores, destination_num_cores
    )


# Ownership determines movement through ``_transfer_edges``.  These existing
# lowerings are only certificates that the backend can emit that movement.
_LOWERING_CERTIFIERS = {
    "broadcast": _broadcast_geometry,
    "gather": _gather_geometry,
    "gather_broadcast": _gather_broadcast_geometry_adapter,
    "shuffle": _shuffle_geometry,
}
# More than one certifier may recognize a future transfer.  This order chooses
# one stable lowering only after every surviving fast path agrees on the edges.
_LOWERING_PRIORITY = ("broadcast", "gather_broadcast", "gather", "shuffle")


def classify_relayout_views(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int | None = None,
    *,
    allowed_kinds: Sequence[str] | None = None,
) -> tuple[str, tuple[RelayoutDimension, ...]] | None:
    """Choose the first certified lowering for an ownership-derived move."""

    if destination_num_cores is None:
        destination_num_cores = source_num_cores
    if (
        source_num_cores <= 0
        or destination_num_cores <= 0
        or source.num_cores != source_num_cores
        or destination.num_cores != destination_num_cores
    ):
        return None
    allowed = set(allowed_kinds) if allowed_kinds is not None else None
    for kind in _LOWERING_PRIORITY:
        if allowed is not None and kind not in allowed:
            continue
        classified = _LOWERING_CERTIFIERS[kind](
            source, destination, source_num_cores, destination_num_cores
        )
        if classified is not None:
            return kind, classified[2]
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


def _destination_plan_key(
    existing: Mapping[tuple, Sequence[str]],
    destination_view: PerCoreView,
    kind: str,
    geometry: tuple[RelayoutDimension, ...],
    source_footprint: int,
    destination_footprint: int,
) -> tuple:
    """Reuse a destination whose owner formulas mean the same thing.

    SymPy can spell the same mapping differently, for example ``core_id`` and
    ``Mod(core_id, 4)`` over four cores.  Structural dictionary equality would
    materialize two copies for those equivalent views and waste LX capacity.
    """

    key = (
        destination_view,
        kind,
        geometry,
        source_footprint,
        destination_footprint,
    )
    for candidate in existing:
        if candidate[1:] == key[1:] and candidate[0].same_partition(destination_view):
            return candidate
    return key


def collect_lx_relayout_plans(
    graph: GraphLowering,
    *,
    source_names: set[str] | None = None,
    ownership_overrides: Mapping[str, TensorWorkDivision] | None = None,
    unprojectable_sources: list[str] | None = None,
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
        if source_names is not None and source_name not in source_names:
            continue
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
            ownership_override=(ownership_overrides or {}).get(source_name),
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
        except ValueError:
            logger.debug(
                "rejected LX relayout candidate source=%s: "
                "cannot represent: source ownership cannot be projected to producer",
                source_name,
            )
            continue

        # Relayout copies sharing one source are allocated and materialized as
        # one atomic group. Any unsupported consumer therefore rejects the
        # group; supported consumers keep using the original buffer instead.
        transfers = []
        seen_consumers = set()
        rejection_reason = None
        source_unprojectable_to_consumer = False
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
            if consumer_num_cores < source_num_cores:
                rejection_reason = (
                    "cannot emit: consumer uses fewer physical cores than producer"
                )
                break
            if consumer_num_cores > source_num_cores and not _is_matmul_op(consumer):
                rejection_reason = (
                    "cannot emit: grouped broadcast requires a matmul consumer"
                )
                break
            if (
                consumer_num_cores > source_num_cores
                and consumer_num_cores != config.sencores
            ):
                rejection_reason = (
                    "cannot emit: grouped broadcast must target all compute cores"
                )
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

            destination_owners = math.prod(dict(view.work_slice_dims).values())
            allowed_kinds: Sequence[str]
            if consumer_num_cores > source_num_cores:
                allowed_kinds = ("broadcast",)
                failure = (
                    "cannot emit: grouped destination does not evenly "
                    "broadcast the source"
                )
            elif destination_owners < source_num_cores:
                if not is_matmul:
                    rejection_reason = (
                        "cannot emit: grouped gather requires a matmul consumer"
                    )
                    break
                allowed_kinds = ("gather", "gather_broadcast")
                failure = (
                    "cannot emit: grouped destination does not evenly contract "
                    "the source"
                )
            else:
                allowed_kinds = ("gather_broadcast", "shuffle")
                failure = "cannot emit: unsupported ownership transfer"

            try:
                classified = classify_relayout_views(
                    source_view,
                    view,
                    source_num_cores,
                    consumer_num_cores,
                    allowed_kinds=allowed_kinds,
                )
            except (TypeError, ValueError) as exc:
                rejection_reason = (
                    f"cannot represent: invalid ownership partition: {exc}"
                )
                break
            if classified is None:
                rejection_reason = failure
                break
            transfers.append(
                (
                    consumer_name,
                    consumer_coordinates,
                    consumer_space,
                    view,
                    *classified,
                )
            )

        plans_by_destination: dict[tuple, list[str]] = {}
        if rejection_reason is None:
            try:
                source_footprint = partition_footprint(producer.layout, source_view)
            except (TypeError, ValueError) as exc:
                rejection_reason = f"allocation: source footprint is unavailable: {exc}"

        if rejection_reason is None:
            for (
                consumer_name,
                consumer_coordinates,
                consumer_space,
                destination_view,
                kind,
                geometry,
            ) in transfers:
                try:
                    source_work_division = work_division_from_view(
                        source_view,
                        producer.layout.device_layout.device_size,
                        consumer_coordinates,
                        consumer_space,
                    )
                except ValueError:
                    rejection_reason = (
                        "cannot represent: source ownership cannot be projected "
                        "to consumer"
                    )
                    source_unprojectable_to_consumer = True
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
                except ValueError:
                    rejection_reason = (
                        "cannot represent: destination ownership cannot be projected "
                        "to consumer"
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
                try:
                    destination_footprint = partition_footprint(
                        producer.layout, destination_view
                    )
                except (TypeError, ValueError) as exc:
                    rejection_reason = (
                        f"allocation: destination footprint is unavailable: {exc}"
                    )
                    break
                key = _destination_plan_key(
                    plans_by_destination,
                    destination_view,
                    kind,
                    geometry,
                    source_footprint,
                    destination_footprint,
                )
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
                    source_footprint_bytes=source_footprint,
                    destination_footprint_bytes=destination_footprint,
                )
                for (
                    destination_view,
                    kind,
                    geometry,
                    source_footprint,
                    destination_footprint,
                ), consumer_names in plans_by_destination.items()
            )
        if rejection_reason is not None:
            if unprojectable_sources is not None and source_unprojectable_to_consumer:
                unprojectable_sources.append(source_name)
            logger.debug(
                "rejected LX relayout candidate source=%s consumers=%s: %s",
                source_name,
                tuple(consumer.get_name() for consumer, _ in consumer_reads),
                rejection_reason,
            )
    return result


def anchor_lx_relayout_ownership(graph: GraphLowering) -> None:
    """Choose the unique canonical producer order accepted by its consumers.

    Work-division split counts are already final here. This pass only changes
    their canonical owner order, and only when the ordinary relayout planner
    proves one unique order makes the complete source group expressible. The
    allocator later records the accepted physical view; the post-scheduler gate
    only preflights that same committed view through codegen's finalizer.
    """

    if (
        not config.lx_consumer_anchored_ordering
        or not config.lx_planner_relayout
        or config.co_optimizing_lx_planning
        or config.ktir_emitter
    ):
        return

    unprojectable_sources: list[str] = []
    collect_lx_relayout_plans(graph, unprojectable_sources=unprojectable_sources)
    operations = {op.get_name(): op for op in graph.operations}

    def direct_consumers_match(
        source_name: str,
        producer: ComputedBuffer,
        candidate: TensorWorkDivision,
    ) -> bool:
        writes = [
            dep
            for dep in op_read_writes(producer).writes
            if isinstance(dep, MemoryDep) and dep.name == source_name
        ]
        reads = [
            (consumer, dep)
            for consumer in graph.operations
            for dep in op_read_writes(consumer).reads
            if isinstance(dep, MemoryDep) and dep.name == source_name
        ]
        if len(writes) != 1 or not reads:
            return False
        source_view, partial, representable = _per_core_view_on_buf(
            producer,
            writes[0],
            source_name,
            ownership_override=candidate,
        )
        if partial or not representable:
            return False
        for consumer, dep in reads:
            view, consumer_partial, consumer_representable = _per_core_view_on_buf(
                consumer, dep, source_name
            )
            if (
                consumer_partial
                or not consumer_representable
                or not view.same_partition(source_view)
            ):
                logger.debug(
                    "direct LX owner mismatch source=%s consumer=%s source_view=%s "
                    "consumer_view=%s partial=%s representable=%s",
                    source_name,
                    consumer.get_name(),
                    source_view,
                    view,
                    consumer_partial,
                    consumer_representable,
                )
                return False
        return True

    for source_name in unprojectable_sources:
        producer = operations.get(source_name)
        if not isinstance(producer, ComputedBuffer):
            continue
        ownership = getattr(producer, "iteration_space_ownership", None)
        if ownership is None:
            continue
        num_cores = ownership.physical_core_count
        split_dims = tuple(
            dim
            for dim in iteration_space_from_op(producer)
            if int(ownership.work_slices.get(dim, 1)) > 1
        )
        if len(split_dims) <= 1:
            continue
        if _is_matmul_op(producer) and config.core_id_k_fast_emission:
            logger.debug(
                "keep %s owner order: matmul K-fast emission is active",
                source_name,
            )
            continue

        def accepted_by_consumers(candidate: TensorWorkDivision) -> bool:
            if direct_consumers_match(source_name, producer, candidate):
                return True
            # The planner is the one legality authority.  Re-run it for this
            # source only, with the candidate owner order, so classification,
            # complete-consumer coverage, and all fail-closed rules stay
            # identical to the real planning pass.
            return bool(
                collect_lx_relayout_plans(
                    graph,
                    source_names={source_name},
                    ownership_overrides={source_name: candidate},
                )
            )

        selected = select_unique_partition_division(
            split_dims,
            ownership.work_slices,
            num_cores,
            accepted_by_consumers,
        )
        if selected is None:
            logger.debug(
                "keep %s owner order: no unique consumer-compatible order",
                source_name,
            )
            continue
        commit_tensor_work_division(producer, selected)
        logger.debug(
            "consumer-anchored LX ownership source=%s mapping=%s",
            source_name,
            selected.core_id_to_work_slice,
        )


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
            "accepted LX relayout %s -> %s: "
            "source=%s@%d destination=%s@%d kind=%s geometry=%s",
            source.get_name(),
            copy.get_name(),
            plan.source_view,
            plan.source_address,
            plan.destination_view,
            plan.destination_address,
            plan.kind,
            plan.group_geometry,
        )
