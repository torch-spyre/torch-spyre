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
from ..core_mapping import partition_physical_span_bytes
from ..ir import FixedTiledLayout
from ..logging_utils import get_inductor_logger
from ..op_spec import TensorWorkDivision
from ..padding import is_restickify_op
from ..pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    completed_reduction_splits_on_buf,
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
    ordering_tag: Literal["uniform_groups"]


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
    # Planning derives completed-reduction roots once from the physical views.
    # Later stages carry this certificate; they never reconstruct it from the
    # aligned operation symbols.
    producer_consumers: tuple[tuple[int, tuple[int, ...]], ...] = ()
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
    device_coordinates: Sequence[sympy.Expr],
    iteration_symbols: Sequence[sympy.Symbol],
) -> TensorWorkDivision | None:
    """Project physical per-core ownership into operation-loop symbols."""

    if view is None:
        return None
    if view.num_cores is None or view.num_cores <= 0:
        raise ValueError("LX ownership must carry its physical core domain")
    loop_symbols = set(iteration_symbols)
    splits: dict[sympy.Symbol, int] = {}
    core_map: dict[sympy.Symbol, sympy.Expr] = {}
    slots = dict(view.core_to_slot)
    if dict(view.work_slice_dims).keys() != slots.keys():
        raise ValueError("LX ownership split and owner-slot dimensions differ")
    for device_dim, split in view.work_slice_dims:
        if device_dim >= len(device_coordinates):
            raise ValueError(f"missing device coordinate {device_dim}")
        matches = device_coordinates[device_dim].free_symbols & loop_symbols
        if len(matches) != 1:
            raise ValueError(f"cannot map device dimension {device_dim} to one loop")
        dim = next(iter(matches))
        slot = sympy.sympify(slots[device_dim])
        if dim in splits and (
            splits[dim] != split
            or not _owner_slots_equal(core_map[dim], slot, split, view.num_cores)
        ):
            raise ValueError(f"conflicting ownership for loop {dim}")
        splits[dim] = split
        core_map[dim] = slot
    return TensorWorkDivision(splits, core_map, num_cores=view.num_cores)


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


def _completed_reduction_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
    reduction_split: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Certify terminal reduction slices fanning out to complete consumers."""

    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    source_owners = math.prod(source_splits.values())
    destination_owners = math.prod(destination_splits.values())
    if (
        source.num_cores != source_num_cores
        or destination.num_cores != destination_num_cores
        or source_num_cores != destination_num_cores
        # The current backend PSUM path accepts only these group widths.
        or reduction_split not in (2, 3, 4)
        or source_owners * reduction_split != source_num_cores
        or destination_owners != destination_num_cores
        or destination_owners <= source_owners
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
                ordering_tag="uniform_groups",
            )
        )
    if destination_owners // source_owners != reduction_split:
        return None
    # Split counts describe the shape but do not prove which physical cores
    # hold the completed values.  Use the same exact edge proof that planning
    # carries forward, so this certifier cannot accept a count-compatible but
    # physically unsupported owner order.
    try:
        derive_completed_reduction_routes(source, destination, reduction_split)
    except ValueError:
        return None
    return source, destination, tuple(geometry)


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
    elements needed by destination core ``d``.  Shuffle, gather, and broadcast
    are certified shapes of this one fact; they must never derive a
    different movement graph.
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


def derive_completed_reduction_routes(
    source: PerCoreView,
    destination: PerCoreView,
    reduction_split: int,
    output_split: int = 1,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Choose the completed producer for each destination from physical ownership.

    ``_transfer_edges`` remains the only movement derivation. This step removes
    nonterminal contributors. DeepTools leaves the complete value on the last
    core of a contiguous reduction group when OUT is unsplit, and on the middle
    core when OUT is split.
    """

    source_num_cores = source.num_cores
    destination_num_cores = destination.num_cores
    if source_num_cores is None or destination_num_cores is None:
        raise ValueError("completed-reduction views require physical core domains")
    if reduction_split not in (2, 3, 4):
        raise ValueError(
            "a completed-reduction broadcast requires a backend-supported split"
        )
    if output_split <= 0:
        raise ValueError("the completed-reduction output split must be positive")
    if source_num_cores <= 0 or destination_num_cores <= 0:
        raise ValueError("physical core counts must be positive")

    source_slices = _core_slices(source, source_num_cores)
    destination_slices = _core_slices(destination, destination_num_cores)
    destination_owners = {
        tuple(sorted(owned_slice.items()))
        for owned_slice in destination_slices.values()
    }
    if len(destination_owners) != destination_num_cores:
        raise ValueError(
            "completed-reduction destinations must own distinct physical slices"
        )
    edges = _transfer_edges(
        source,
        destination,
        source_num_cores,
        destination_num_cores,
    )
    source_groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for core, owned_slice in source_slices.items():
        owner = tuple(sorted(owned_slice.items()))
        source_groups.setdefault(owner, []).append(core)
    if any(len(group) != reduction_split for group in source_groups.values()):
        raise ValueError("source owner groups do not match the reduction split")
    if any(
        group != list(range(group[0], group[-1] + 1))
        for group in source_groups.values()
    ):
        raise ValueError("completed-reduction source groups must be contiguous")

    terminal_offset = reduction_split // 2 if output_split > 1 else reduction_split - 1
    terminals = {group[0] + terminal_offset for group in source_groups.values()}
    routes: dict[int, list[int]] = {terminal: [] for terminal in terminals}
    sources_by_destination: dict[int, list[int]] = {
        core: [] for core in range(destination_num_cores)
    }
    for source_core, destination_core in edges:
        if source_core in terminals:
            sources_by_destination[destination_core].append(source_core)
    for destination_core, source_cores in sources_by_destination.items():
        if len(source_cores) != 1:
            raise ValueError(
                f"destination core {destination_core} has "
                f"{len(source_cores)} completed sources"
            )
        routes[source_cores[0]].append(destination_core)

    fanouts = {len(consumers) for consumers in routes.values()}
    if 0 in fanouts or len(fanouts) != 1:
        raise ValueError("completed-reduction routes require uniform fanout")
    return tuple(
        (source_core, tuple(sorted(consumers)))
        for source_core, consumers in sorted(routes.items())
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


def _broadcast_geometry(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> tuple[PerCoreView, PerCoreView, tuple[RelayoutDimension, ...]] | None:
    """Adapter giving grouped broadcast the common classifier signature."""

    return _grouped_broadcast_geometry(
        source, destination, source_num_cores, destination_num_cores
    )


# Existing lowering functions are the registry.  The table adds no plan IR: it
# only makes the authority order explicit.  A surviving fast path must certify
# the same edge set derived by ``_transfer_edges`` before its name is recorded
# in ``LXRelayoutPlan.kind``.
_LOWERING_CERTIFIERS = {
    "broadcast": _broadcast_geometry,
    "gather": _gather_geometry,
    "shuffle": _shuffle_geometry,
}
# More than one certifier may recognize a future transfer.  This order chooses
# one stable lowering only after every surviving fast path agrees on the edges.
_LOWERING_PRIORITY = ("broadcast", "gather", "shuffle")


def classify_relayout_views(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int | None = None,
    *,
    allowed_kinds: Sequence[str] | None = None,
    reduction_split: int | None = None,
) -> tuple[str, tuple[RelayoutDimension, ...]] | None:
    """Choose a certified lowering for one already-described view pair.

    Ownership is the fact.  The returned name is merely the first existing
    lowering, in deterministic priority order, whose certifier agrees with the
    shared movement graph.  Unsupported movement remains an HBM fallback.
    """

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
    if reduction_split is not None:
        if allowed is not None and "broadcast" not in allowed:
            return None
        classified = _completed_reduction_geometry(
            source,
            destination,
            source_num_cores,
            destination_num_cores,
            reduction_split,
        )
        return None if classified is None else ("broadcast", classified[2])
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
    producer_consumers: tuple[tuple[int, tuple[int, ...]], ...],
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
        producer_consumers,
    )
    for candidate in existing:
        if candidate[1:] == key[1:] and candidate[0].same_partition(destination_view):
            return candidate
    return key


def collect_lx_relayout_plans(graph: GraphLowering) -> list[LXRelayoutPlan]:
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
            producer, write, source_name, cache
        )
        source_num_cores = _op_num_cores(producer)
        reduction_info = (
            completed_reduction_splits_on_buf(producer, write, source_name)
            if partial
            else None
        )
        reduction_split, reduction_output_split = (
            reduction_info if reduction_info is not None else (None, None)
        )
        if (
            source_view is None
            or not representable
            or source_view.num_cores != source_num_cores
        ):
            continue
        if partial and (
            reduction_split is None
            or source_num_cores != config.sencores
            or not config.core_id_k_fast_emission
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
                producer_coordinates,
                tuple(iteration_space_from_op(producer)),
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
            consumer_symbols = tuple(iteration_space_from_op(consumer))
            if reduction_split is None and view.same_partition(source_view):
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
            if reduction_split is not None:
                expected_kind = "broadcast"
                failure = (
                    "cannot emit: completed reduction does not evenly cover "
                    "the destination"
                )
            elif consumer_num_cores > source_num_cores:
                expected_kind = "broadcast"
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
                expected_kind = "gather"
                failure = (
                    "cannot emit: grouped destination does not evenly contract "
                    "the source"
                )
            else:
                expected_kind = "shuffle"
                failure = "cannot emit: unsupported ownership transfer"

            try:
                classified = classify_relayout_views(
                    source_view,
                    view,
                    source_num_cores,
                    consumer_num_cores,
                    allowed_kinds=(expected_kind,),
                    reduction_split=reduction_split,
                )
            except (TypeError, ValueError) as exc:
                rejection_reason = (
                    f"cannot represent: invalid ownership partition: {exc}"
                )
                break
            if classified is None:
                rejection_reason = failure
                break
            try:
                if reduction_split is not None:
                    if reduction_output_split is None:
                        raise ValueError(
                            "completed-reduction output split was not certified"
                        )
                    producer_consumers = derive_completed_reduction_routes(
                        source_view,
                        view,
                        reduction_split,
                        reduction_output_split,
                    )
                else:
                    producer_consumers = ()
            except ValueError as exc:
                rejection_reason = (
                    f"cannot emit: invalid completed-reduction routes: {exc}"
                )
                break
            transfers.append(
                (
                    consumer_name,
                    consumer_coordinates,
                    consumer_symbols,
                    view,
                    *classified,
                    producer_consumers,
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
                consumer_symbols,
                destination_view,
                kind,
                geometry,
                producer_consumers,
            ) in transfers:
                try:
                    source_work_division = work_division_from_view(
                        source_view, consumer_coordinates, consumer_symbols
                    )
                except ValueError:
                    rejection_reason = (
                        "cannot represent: source ownership cannot be projected "
                        "to consumer"
                    )
                    break
                if source_work_division is None:
                    raise RuntimeError(
                        "LX relayout source lost its certified physical ownership"
                    )
                try:
                    destination_work_division = work_division_from_view(
                        destination_view, consumer_coordinates, consumer_symbols
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
                    producer_consumers,
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
                    producer_consumers=producer_consumers,
                )
                for (
                    destination_view,
                    kind,
                    geometry,
                    source_footprint,
                    destination_footprint,
                    producer_consumers,
                ), consumer_names in plans_by_destination.items()
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
