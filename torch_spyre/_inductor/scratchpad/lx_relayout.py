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

import collections
import dataclasses
from functools import cache
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    MutableBox,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
)
from torch_spyre._C import ElementArrangement

from .. import config
from ..core_mapping import (
    _MAX_SHUFFLE_FANIN,
    work_division_from_view as physical_work_division_from_view,
    owner_slots,
    partition_lx_size_bytes,
    select_unique_partition_division,
    transfer_edges,
)
from ..cost_model import CostParams, relayout_ns
from ..cost_features import relayout_cost_features
from ..ir import FixedTiledLayout
from ..loop_info import CoarseTileInfo
from ..logging_utils import get_inductor_logger
from ..op_spec import TensorWorkDivision
from ..padding import is_restickify_op
from ..pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    commit_tensor_work_division,
    completed_reduction_split_on_buf,
    iteration_space_from_op,
    op_read_writes,
    try_device_coordinates,
)
from .utils import _op_num_cores, storage_has_single_writer

logger = get_inductor_logger("lx_relayout")
_DESTINATION_PREFIX = "__spyre_lx_relayout__"
_REGISTRY = "_spyre_lx_relayout_copies"
# The current shuffle lowering can need one mandatory L3LU address-bound
# register per incoming fragment. There are eight LBRs on Spyre. Deeptools
# does not common these values per core for fully global programs, so a
# larger gather can fail register initialization even when individual cores
# need fewer distinct values. Apply this conservative bound before placement.
# Largest governing split the solver may be offered for a GROUPED movement.
_GROUPED_SPLIT_MAX = 8


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_names: tuple[str, ...]
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    source_footprint_bytes: int = 0
    destination_footprint_bytes: int = 0
    completed_producer_cores: tuple[int, ...] = ()
    source_address: int | None = None
    destination_address: int | None = None

    @property
    def destination_name(self) -> str:
        return f"{_DESTINATION_PREFIX}:{self.source_name}:{self.consumer_names[0]}"

    @property
    def edge(self) -> tuple[str, str]:
        return self.source_name, self.destination_name


@dataclasses.dataclass(frozen=True)
class RelayoutCandidate:
    """One priced way for a divided producer to stay LX-resident for one consumer.

    Born in the allocator's enumeration (``_cd_parent_relayouts``) and carried
    unchanged through the CP-SAT model, the extraction and the commit path: the
    solver keys its pair literal by this record, extraction attaches the solved
    placement (:class:`ChosenRelayout`), and the commit path folds the fired
    members of one fired group into a :class:`LXRelayoutPlan`
    (:class:`FiredRelayoutGroup`). Nothing downstream re-derives a view, a core
    count or a price from primitives, so a change to what a relayout *is*
    (another lowering kind, a measured footprint) is a change to this record
    and to the enumeration that builds it, nowhere else.

    ``group`` identifies the DESTINATION per-core view of ``parent``, interned
    per parent by the allocator for one solve: every candidate that lands on
    the same view of the same parent shares one shuffle and one LX destination
    (``RelayoutCopyBuffer``), so the solver prices and places the group once,
    not per edge.

    Each view carries the physical core count it was built for (every core's
    owner slot within its split), so ``num_cores`` is the source view's and
    ``destination_num_cores`` the destination view's; the enumeration's
    ``cores_used`` gates guarantee the views were built for the divisions
    being paired.

    ``source_footprint_bytes`` / ``destination_footprint_bytes`` are the
    per-core LX spans of the two views (:func:`partition_footprint`, the bound
    the committed path reserves for a relayout member since #3440), measured
    once at enumeration: the copy buffer is sized from the destination span and
    the plan hands both to the allocator, so the solver never reserves less for
    a shuffle than the committed path would.
    """

    parent: str
    consumer: str
    source_division: int
    consumer_division: int
    group: int
    source_view: PerCoreView
    destination_view: PerCoreView
    cost_ns: float
    source_footprint_bytes: int
    destination_footprint_bytes: int

    @property
    def num_cores(self) -> int:
        """The source's core count, carried by the source view (the plan's
        ``num_cores``, as the committed collector records it)."""
        return cast(int, self.source_view.num_cores)

    @property
    def destination_num_cores(self) -> int:
        """The consumer's core count, carried by the destination view: equal to
        ``num_cores`` for permutations and grouped gathers. Complete repartitions
        may use an integral expansion or contraction; the copy lives on the
        destination's cores while the plan's ``num_cores`` stays the source's."""
        return cast(int, self.destination_view.num_cores)

    def __post_init__(self) -> None:
        for side, view in (
            ("source", self.source_view),
            ("destination", self.destination_view),
        ):
            if view.num_cores is None:
                raise ValueError(
                    f"relayout candidate {self.parent} -> {self.consumer}: the {side} "
                    "view carries no physical core count"
                )
        if (
            min(self.destination_num_cores, self.num_cores) <= 0
            or max(self.destination_num_cores, self.num_cores)
            % min(self.destination_num_cores, self.num_cores)
            or (
                self.destination_num_cores < self.num_cores
                and (
                    self.source_view.split_product != self.num_cores
                    or self.destination_view.split_product != self.destination_num_cores
                )
            )
        ):
            raise ValueError(
                f"relayout candidate {self.parent} -> {self.consumer}: destination "
                f"on {self.destination_num_cores} cores is incompatible with the "
                f"source's {self.num_cores}"
            )
        if self.source_view.same_partition(self.destination_view):
            raise ValueError(
                f"relayout candidate {self.parent} -> {self.consumer} has equal "
                "views; that pair belongs to cd_parent_matches"
            )
        if self.source_footprint_bytes <= 0 or self.destination_footprint_bytes <= 0:
            raise ValueError(
                f"relayout candidate {self.parent} -> {self.consumer} has no "
                f"measured span (source {self.source_footprint_bytes}, destination "
                f"{self.destination_footprint_bytes} bytes); a candidate whose "
                "footprint is unavailable must be declined at enumeration"
            )

    @property
    def group_key(self) -> tuple[str, int]:
        """The solver's registry key: one destination view of one parent."""
        return self.parent, self.group


@dataclasses.dataclass(frozen=True)
class ChosenRelayout:
    """A fired :class:`RelayoutCandidate` with the solved address of its copy.

    Every consumer of one (parent, group) the solver served reads the group's
    single copy (``RelayoutCopyBuffer``), so all of them carry the same
    address; the commit path folds them into one plan
    (:meth:`FiredRelayoutGroup.from_chosen`).
    """

    candidate: RelayoutCandidate
    destination_address: int

    def scaled(self, alignment: int) -> ChosenRelayout:
        """The same choice with the address converted from alignment units."""
        return dataclasses.replace(
            self, destination_address=self.destination_address * alignment
        )


@dataclasses.dataclass(frozen=True)
class FiredRelayoutGroup:
    """The consumers of one (parent, destination view) the solver served from
    the group's copy: one shuffle, one LX residency at ``destination_address``.

    Members share the source division, both views and the address by
    construction (they read one buffer the solver placed once), which
    :meth:`from_chosen` verifies rather than trusts.
    """

    parent: str
    group: int
    members: tuple[ChosenRelayout, ...]

    @property
    def candidate(self) -> RelayoutCandidate:
        """A representative member; every field the plan needs agrees across
        the group (checked in :meth:`from_chosen`)."""
        return self.members[0].candidate

    @property
    def source_division(self) -> int:
        return self.candidate.source_division

    @property
    def destination_address(self) -> int:
        return self.members[0].destination_address

    @property
    def consumer_names(self) -> tuple[str, ...]:
        return tuple(m.candidate.consumer for m in self.members)

    def plan(self, source_address: int) -> LXRelayoutPlan:
        c = self.candidate
        return LXRelayoutPlan(
            self.parent,
            self.consumer_names,
            c.source_view,
            c.destination_view,
            c.num_cores,
            source_footprint_bytes=c.source_footprint_bytes,
            destination_footprint_bytes=c.destination_footprint_bytes,
            source_address=source_address,
            destination_address=self.destination_address,
        )

    @classmethod
    def from_chosen(cls, chosen: Iterable[ChosenRelayout]) -> list[FiredRelayoutGroup]:
        """Regroup fired edges by (parent, destination view).

        Deterministic order (sorted keys, members sorted by consumer name) so
        plan construction, and hence destination naming, is reproducible.
        """
        by_group: dict[tuple[str, int], list[ChosenRelayout]] = {}
        for ch in chosen:
            by_group.setdefault((ch.candidate.parent, ch.candidate.group), []).append(
                ch
            )
        groups: list[FiredRelayoutGroup] = []
        for (parent, group), members in sorted(by_group.items()):
            members.sort(key=lambda ch: ch.candidate.consumer)
            first = members[0]
            for m in members[1:]:
                agree = (
                    m.candidate.source_division == first.candidate.source_division
                    and m.candidate.source_view.same_partition(
                        first.candidate.source_view
                    )
                    and m.candidate.destination_view.same_partition(
                        first.candidate.destination_view
                    )
                    and m.candidate.num_cores == first.candidate.num_cores
                    and m.candidate.destination_num_cores
                    == first.candidate.destination_num_cores
                    and m.candidate.source_footprint_bytes
                    == first.candidate.source_footprint_bytes
                    and m.candidate.destination_footprint_bytes
                    == first.candidate.destination_footprint_bytes
                    and m.destination_address == first.destination_address
                )
                if not agree:
                    raise AssertionError(
                        f"relayout group {parent}/g{group}: members disagree on "
                        f"geometry or placement: {first} vs {m}"
                    )
            groups.append(cls(parent, group, tuple(members)))
        return groups


def work_division_from_view(
    view: PerCoreView | None,
    device_size: Sequence[int],
    device_coordinates: Sequence[sympy.Expr],
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> TensorWorkDivision | None:
    """Keep the planner's explicit rollout policy outside physical geometry."""
    return physical_work_division_from_view(
        view,
        device_size,
        device_coordinates,
        iteration_space,
        allow_extended=config.lx_extended_ownership,
    )


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


def partition_footprint(layout: FixedTiledLayout, view: PerCoreView) -> int:
    """Measure a relayout candidate in normalized standard device layout.

    FixedTiledLayout can wrap an explicit device shape, so its type alone does
    not guarantee a complete final stick axis. The size helper validates it.
    """
    device_layout = layout.device_layout
    if device_layout.element_arrangement != ElementArrangement.STANDARD:
        raise ValueError("relayout footprint requires standard element arrangement")
    return partition_lx_size_bytes(
        tuple(int(size) for size in device_layout.device_size),
        device_layout.device_dtype,
        dict(view.work_slice_dims),
    )


def movement_supported(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> bool:
    """Check transfer geometry with the current ownership rollout policy."""
    # Include the rollout flag in the memo key so toggling it cannot reuse ON results.
    return _movement_supported(
        source,
        destination,
        source_num_cores,
        destination_num_cores,
        config.lx_extended_ownership,
    )


@cache
def _movement_supported(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
    extended_ownership: bool,
) -> bool:
    """Certify uniform partition transfers, gathers and broadcasts.

    Edges are ownership intersections, never a separate geometry calculation.
    A complete source may feed uniformly repeated destination slices. Across
    unequal core counts, a complete destination partition may gather
    several disjoint source fragments. A full replicated destination may also
    gather a complete source partition (the cache-transpose to QK handoff).
    Partially partitioned replicas retain the one-source broadcast rule.
    Equal destination slices have identical sources by construction.
    Incoming fragments must also fit the current shuffle lowering's address
    register budget; broadcast fan-out does not consume this budget.
    """
    if (
        source_num_cores <= 0
        or destination_num_cores <= 0
        or (not extended_ownership and destination_num_cores < source_num_cores)
        or source.num_cores != source_num_cores
        or destination.num_cores != destination_num_cores
        or max(destination_num_cores, source_num_cores)
        % min(destination_num_cores, source_num_cores)
        or (
            source_num_cores == destination_num_cores
            and source.same_partition(destination)
        )
    ):
        return False

    destination_slices = destination.split_product
    if (
        destination_slices <= 0
        or destination_num_cores % destination_slices
        or source.split_product != source_num_cores
        or (
            destination_num_cores < source_num_cores
            and destination_slices != destination_num_cores
        )
    ):
        return False

    source_map = _core_slices(source, source_num_cores)
    # Every source slice is present exactly once.
    if (
        len({tuple(sorted(row.items())) for row in source_map.values()})
        != source_num_cores
    ):
        return False

    destination_map = _core_slices(destination, destination_num_cores)
    replicas = collections.Counter(
        tuple(sorted(row.items())) for row in destination_map.values()
    )

    # Every distinct destination slice is covered.
    if len(replicas) != destination_slices:
        return False
    # Within one core domain, each slice has equally many copies. A larger
    # full-replica gather has only one distinct slice, so is uniform as well.
    # Preserve the existing one-source broadcast rule for other expansions.
    if source_num_cores == destination_num_cores and len(set(replicas.values())) != 1:
        return False

    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    edges = transfer_edges(
        source_splits, destination_splits, source_map, destination_map
    )
    if not edges:
        return False

    fanout = [0] * source_num_cores
    fanin = [0] * destination_num_cores
    for src, dst in edges:
        fanout[src] += 1
        fanin[dst] += 1

    # Every source sends to the same number of destination cores.
    if len(set(fanout)) != 1:
        return False
    # Every destination receives from the same number of source cores.
    if len(set(fanin)) != 1:
        return False
    if max(fanin) > _MAX_SHUFFLE_FANIN:
        return False
    # Conservative workaround for issue #5811: complete repartitions where the
    # source splits a dropped axis (absent from destination.work_slice_dims) by
    # _MAX_SHUFFLE_FANIN produce wrong values with NDO=1.  The exact failure
    # mechanism is not yet established; this guard excludes the observed bad
    # geometry until the root cause is confirmed.
    _dropped_split_product = 1
    for _axis, _split in source_splits.items():
        if _axis not in destination_splits:
            _dropped_split_product *= _split
    if (
        destination_slices == destination_num_cores
        and _dropped_split_product >= _MAX_SHUFFLE_FANIN
    ):
        return False
    # A complete destination is an exact repartition, including contraction
    # onto fewer cores and expansion onto more cores. The shuffle's per-tensor
    # maps describe the two domains; STCDP intersects their pieces rather than
    # assuming one source per reader.
    # A full replica is the union of every complete, disjoint source slice.
    # STCDP already multicasts each source piece to every destination; the
    # same-core piece uses its local LX path. The fanin bound above still
    # applies. Do not generalize to partially partitioned replica expansions.
    return (
        source_num_cores == destination_num_cores
        or destination_slices == destination_num_cores
        or (
            extended_ownership
            and destination_slices == 1
            and fanin[0] == source_num_cores
        )
        or (fanout[0] == destination_num_cores // source_num_cores and fanin[0] == 1)
    )


def core_domain_rejection(source_num_cores: int, consumer_num_cores: int) -> str | None:
    """Why a producer on ``source_num_cores`` may not feed a consumer on
    ``consumer_num_cores`` through a relayout, from the core counts alone, or
    ``None`` when they are compatible. Equal counts are always fine
    (permutations and gathers). A larger consumer domain is either a complete
    repartition, a grouped broadcast, or a full replicated gather. The larger
    domain must use every compute core. Contraction requires complete
    partitions, never replicated output.
    The geometric half of that rule is :func:`movement_supported`'s.
    Shared by the committed collector and the solver's enumeration so the two
    paths cannot drift."""
    if not config.lx_extended_ownership and consumer_num_cores < source_num_cores:
        return "cannot emit: consumer uses fewer physical cores than producer"
    smaller, larger = sorted((source_num_cores, consumer_num_cores))
    if smaller <= 0 or larger % smaller:
        return "cannot emit: relayout domains must have an integral ratio"
    if smaller != larger and larger != config.sencores:
        return "cannot emit: larger relayout domain must use all compute cores"
    return None


def grouped_gather_rejection(
    consumer: Operation, source_num_cores: int, destination_view: PerCoreView
) -> str | None:
    """Reject a replicated grouped gather without a matmul consumer.

    A #3440 grouped gather repeats assembled destination slices across cores.
    This replicated layout supports matmul consumers only.
    A complete destination partition has no replicas. Shuffle lowering can
    assemble its disjoint source pieces for a pointwise consumer too, including
    expansion onto more cores. ``movement_supported`` checks its uniform
    routes and fan-in bound. Both candidate pricing and committed copies use
    this consumer rule.
    """
    owners = math.prod(dict(destination_view.work_slice_dims).values())
    if (
        owners < source_num_cores
        and (not config.lx_extended_ownership or owners != destination_view.num_cores)
        and not _is_matmul_op(consumer)
    ):
        return "cannot emit: grouped gather requires a matmul consumer"
    return None


def derive_completed_reduction_routes(
    source: PerCoreView,
    destination: PerCoreView,
    reduction_split: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Select finished producers, then intersect producer/consumer partitions.

    A matmul writes its completed piece on the last core of each contiguous
    K-fast group. Other cores' buffers are unwritten, not additional copies.
    The remaining producer map uses the same intersections as ordinary copies:
    several producers may supply disjoint pieces, never unfinished sums.
    """
    source_count, destination_count = source.num_cores, destination.num_cores
    splits, target = dict(source.work_slice_dims), dict(destination.work_slice_dims)
    owners = math.prod(splits.values())
    if (
        source_count is None
        or destination_count is None
        or reduction_split <= 1
        or owners * reduction_split != source_count
        or math.prod(target.values()) != destination_count
        or destination_count not in (source_count, owners)
        or (
            destination_count != source_count
            and any(
                target.get(d, 1) % splits.get(d, 1)
                for d in splits.keys() | target.keys()
            )
        )
    ):
        raise ValueError("unsupported completed-reduction ownership geometry")
    source_map = _core_slices(source, source_count)
    target_map = _core_slices(destination, destination_count)
    groups: dict[tuple, list[int]] = {}
    for core, row in source_map.items():
        groups.setdefault(tuple(sorted(row.items())), []).append(core)
    if (
        len(groups) != owners
        or len({tuple(sorted(row.items())) for row in target_map.values()})
        != destination_count
        or any(
            group != list(range(group[0], group[0] + reduction_split))
            for group in groups.values()
        )
    ):
        raise ValueError(
            "completed-reduction owners require contiguous source groups and distinct destinations"
        )
    # Mask unfinished producers before the common ownership intersection.
    source_map = {group[-1]: source_map[group[-1]] for group in groups.values()}
    routes: dict[int, list[int]] = {core: [] for core in sorted(source_map)}
    edges = transfer_edges(splits, target, source_map, target_map)
    fanins = set()
    for destination_core in range(destination_count):
        writers = [s for s, d in edges if d == destination_core]
        if not writers or (destination_count != source_count and len(writers) != 1):
            raise ValueError("destination has unsupported completed-result coverage")
        fanins.add(len(writers))
        for writer in writers:
            routes[writer].append(destination_core)
    counts = {len(consumers) for consumers in routes.values()}
    if 0 in counts or len(counts) != 1 or len(fanins) != 1:
        raise ValueError("completed-reduction routes require uniform fanin and fanout")
    return tuple((core, tuple(consumers)) for core, consumers in routes.items())


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
    """Allow computed activations and snapshots of an updated input cache.

    Only the immutable restickification result is eligible. The mutation and
    its target stay in place; cache contents, aliases, indexing and ordering
    still flow through the original load and scheduler dependencies. A cache
    read cannot be replaced with the new values alone: untouched history may
    be read too. Unmodified inputs/weights retain the existing exclusion.
    """

    if not is_restickify_op(op, graph):
        return True
    reads = [d for d in op_read_writes(op).reads if isinstance(d, MemoryDep)]
    if any(isinstance(operations.get(dep.name), ComputedBuffer) for dep in reads):
        return True
    if not config.lx_extended_ownership:
        return False
    if (
        len(reads) != 1
        or reads[0].is_indirect()
        or reads[0].name not in graph.graph_inputs
    ):
        return False
    # Match the storage name obtained by unwrapping mutation target views,
    # rather than an op name or a heuristic such as "cache" in the input name.
    # Require an earlier, ordinary in-graph mutation. This is eligibility,
    # not an ordering repair: we never rewrite the restickification's load.
    updated = False
    for previous in graph.operations:
        if previous is op:
            return updated
        if (
            isinstance(previous, ComputedBuffer)
            and isinstance(previous.layout, MutationLayoutSHOULDREMOVE)
            and getattr(previous, "loop_info", None) is None
            and previous.layout.get_buffer().get_name() == reads[0].name
        ):
            updated = True
    return False


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


def _persistent_scatter_consumer(
    graph: GraphLowering,
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    source_name: str,
) -> bool:
    """Permit only a fresh matmul value feeding an external cache scatter.

    A direct read is insufficient: the value must have independent storage,
    no aliasing users or later writes, and precede this mutation. Require a
    bare input-buffer target (no view), an indirect write to that same target,
    and no cache reads. The existing load-only clone rewrite preserves the
    mutation's target, indices, dependencies and position in the graph. The
    external target is never a candidate LX allocation.
    """
    if (
        not config.lx_extended_ownership
        or not _is_matmul_op(producer)
        or not isinstance(consumer.data, Pointwise)
        or getattr(producer, "loop_info", None) is not None
        or getattr(consumer, "loop_info", None) is not None
    ):
        return False
    try:
        if producer.get_mutation_names() or producer.get_inputs_that_alias_output():
            return False
    except NotImplementedError:
        return False
    target = consumer.layout.target
    while isinstance(target, MutableBox):
        target = target.data
    if not isinstance(target, InputBuffer):
        return False
    target_name = target.get_name()
    inputs = {
        value.get_name()
        for value in graph.graph_inputs.values()
        if hasattr(value, "get_name")
    }
    if (
        target_name not in inputs
        or source_name in inputs
        or not isinstance(producer.layout, FixedTiledLayout)
        or "lx" in getattr(target.get_layout(), "allocation", {})
    ):
        return False
    operations = graph.operations
    if producer not in operations or consumer not in operations:
        return False
    if operations.index(producer) >= operations.index(consumer):
        return False
    deps = op_read_writes(consumer)
    writes = list(deps.writes)
    if (
        len(writes) != 1
        or not isinstance(writes[0], MemoryDep)
        # Before scheduling, ComputedBuffer names its own write; the mutation
        # layout binds that name to the external target.
        or writes[0].name not in (consumer.get_name(), target_name)
        or not writes[0].is_indirect()
        or set(consumer.get_mutation_names()) != {target_name}
        or any(dep.name == target_name for dep in deps.reads)
    ):
        return False
    # FixedTiledLayout is fresh storage, but aliases introduced by other ops
    # could still mutate it or extend its lifetime behind the planner's back.
    # Reject those graphs rather than infer safety from a solver placement.
    if source_name != producer.get_name():
        return False
    return storage_has_single_writer(graph, producer)


def _supported_consumer_layout(graph, producer, consumer, source_name) -> bool:
    return isinstance(consumer, ComputedBuffer) and (
        not isinstance(consumer.layout, MutationLayoutSHOULDREMOVE)
        or _persistent_scatter_consumer(graph, producer, consumer, source_name)
    )


def solver_relayout_edge_context(
    graph: GraphLowering,
    producer: Operation,
    consumer: Operation,
    source_name: str,
    operations: dict[str, Operation],
) -> tuple | None:
    """Division-independent relayout eligibility of one producer->consumer edge.

    The same structural gates ``collect_lx_relayout_plans`` applies on the
    committed graph, restricted to what does not depend on a chosen division, so
    the solver's candidate enumeration can run them once per edge before any
    per-division-pair work. Returns ``(write_dep, read_dep, producer_coords,
    consumer_coords, producer_space, consumer_space)`` - the two spaces are the
    loop-symbol -> extent mappings ``work_division_from_view`` projects into -
    or ``None`` when the edge can never host a relayout.
    """
    # A coarse-tiled endpoint can usually never host a relayout: a tiled
    # consumer in general reads cross-group data through a per-iteration
    # staging op a relayout can't account for. That doesn't apply when
    # producer and consumer share the exact same loop_group_id -- same
    # iteration, same tile, direct read, no staging op between them -- so
    # that one case is let through, unless the consumer is already a
    # restickify under read-copy-elision's own deferred direct-read handling
    # (the same marker _distribute_work's division gate checks): that
    # mechanism already owns this edge's placement, so relayout must not
    # also offer a candidate for it. The fitted relayout law still prices
    # one tile's worth; _cd_parent_relayouts (allocator.py) scales that by
    # the parent's own loop trip count before handing a same-group candidate
    # to the solver, so a repeated shuffle is priced at its real total cost,
    # not once.
    producer_loop_info = getattr(producer, "loop_info", None)
    consumer_loop_info = getattr(consumer, "loop_info", None)
    same_tile_group = (
        config.lx_same_tile_group_relayout
        and isinstance(producer_loop_info, CoarseTileInfo)
        and isinstance(consumer_loop_info, CoarseTileInfo)
        and producer_loop_info.loop_group_id == consumer_loop_info.loop_group_id
        and not (
            hasattr(consumer, "_read_copy_elision_record")
            and is_restickify_op(consumer, graph)
        )
    )
    if (
        producer_loop_info is not None or consumer_loop_info is not None
    ) and not same_tile_group:
        return None
    if (
        not isinstance(producer, ComputedBuffer)
        or not isinstance(producer.layout, FixedTiledLayout)
        or (write_dep := _single_write(producer, source_name)) is None
        or not _is_activation_source(graph, operations, producer)
    ):
        return None
    if not _supported_consumer_layout(graph, producer, consumer, source_name):
        return None
    if not _is_matmul_op(consumer) and not isinstance(consumer.data, Pointwise):
        return None
    consumer_deps = [
        d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)
    ]
    if any(d.is_indirect() for d in consumer_deps):
        return None
    if _is_matmul_op(consumer) and len(consumer_deps) != 2:
        return None
    source_reads = [d for d in consumer_deps if d.name == source_name]
    if len(source_reads) != 1:
        return None
    read_dep = source_reads[0]
    producer_coords = try_device_coordinates(
        producer.layout.device_layout, write_dep, None
    )
    consumer_coords = try_device_coordinates(
        producer.layout.device_layout, read_dep, None
    )
    if producer_coords is None or consumer_coords is None:
        return None
    return (
        write_dep,
        read_dep,
        producer_coords,
        consumer_coords,
        iteration_space_from_op(producer),
        iteration_space_from_op(consumer),
    )


def prefix_lx_contraction_axis(producer: Operation, consumer: Operation) -> int | None:
    """Prove a zero-offset dense prefix read after a full-buffer LX gather.

    Ordinary IR shapes describe the producer's full width and the consumer's
    shorter K. Only the outermost stick axis may have a tail. The read index
    must retain the producer row stride exactly; arbitrary strided slices,
    offsets, broadcast/batched reads and counted loops are not admitted.
    The division-dependent gates separately require unsplit K and a disjoint
    copy of the complete producer domain. No width metadata or loop rewrite
    is involved.
    """
    if not isinstance(producer, ComputedBuffer) or not _is_matmul_op(consumer):
        return None
    layout = producer.layout
    if (
        not isinstance(layout, FixedTiledLayout)
        or layout.offset != 0
        or len(layout.size) < 2
        or any(not sympy.sympify(x).is_Integer for x in layout.size)
        or any(x != 1 for x in layout.size[:-2])
        or getattr(producer, "loop_info", None) is not None
        or getattr(consumer, "loop_info", None) is not None
        or getattr(producer, "_input_layout_overrides", None)
        or getattr(consumer, "_input_layout_overrides", None)
    ):
        return None
    m, physical = map(int, layout.size[-2:])
    ranges = consumer.data.ranges
    reduction = consumer.data.reduction_ranges
    if (
        m <= 1
        or list(layout.stride[-2:]) != [physical, 1]
        or len(ranges) < 2
        or any(x != 1 for x in ranges[:-2])
        or ranges[-2] != m
        or len(reduction) != 1
        or not sympy.sympify(reduction[0]).is_Integer
        or not sympy.sympify(ranges[-1]).is_Integer
    ):
        return None
    logical, n = int(reduction[0]), int(ranges[-1])
    rw = op_read_writes(consumer)
    reads = list(rw.reads)
    writes = list(rw.writes)
    if (
        not 0 < logical < physical
        or n <= 1
        or len(reads) != 2
        or len(writes) != 1
        or not all(isinstance(d, MemoryDep) for d in [*reads, *writes])
    ):
        return None
    out = writes[0]
    ms = [
        v for v, extent in out.ranges.items() if extent == m and out.index.coeff(v) == n
    ]
    ns = [
        v for v, extent in out.ranges.items() if extent == n and out.index.coeff(v) == 1
    ]
    if len(ms) != 1 or len(ns) != 1:
        return None
    ms, ns = ms[0], ns[0]
    ks = (reads[0].index.free_symbols | reads[1].index.free_symbols) - {ms, ns}
    if len(ks) != 1:
        return None
    (ks,) = ks
    source_reads = [d for d in reads if d.name == producer.get_name()]
    if len(source_reads) != 1:
        return None
    src = source_reads[0]
    weight = next(d for d in reads if d is not src)
    if (
        out.index != n * ms + ns
        or src.index != physical * ms + ks
        or src.ranges.get(ms) != m
        or src.ranges.get(ks) != logical
        or weight.index != logical * ns + ks
        or weight.ranges.get(ns) != n
        or weight.ranges.get(ks) != logical
    ):
        return None
    stl = layout.device_layout
    eps = stl.elems_per_stick()
    dims = list(stl.device_size)
    axes = [
        axis
        for axis in range(len(dims) - 1)
        if dims[axis] * eps == physical and stl.stride_map[axis] == eps
    ]
    if (
        stl.element_arrangement != ElementArrangement.STANDARD
        or logical % eps
        or physical % eps
        or stl.stride_map[-1] != 1
        or dims[-1] != eps
        or len(axes) != 1
        or any(size != 1 for size in dims[: axes[0]])
        or math.prod(dims) != m * physical
    ):
        return None
    return axes[0]


def prefix_lx_gather_source(graph: GraphLowering, name: str, uses: list[int]) -> bool:
    """One full-width producer and one zero-offset prefix matmul reader.

    Division-dependent proof remains in the relayout pair gate: the reader
    must own the whole padded feature axis, and a real copy must be allocated.
    Other users are deliberately refused, including external/collective users.
    """
    if name in graph.get_output_names():
        return False
    producer = graph.get_buffer(name)
    readers = [graph.operations[i] for i in uses if graph.operations[i] is not producer]
    return (
        len(readers) == 1
        and prefix_lx_contraction_axis(producer, readers[0]) is not None
        and storage_has_single_writer(graph, producer)
    )


def solver_relayout_pair_cost(
    source_view: PerCoreView,
    destination_view: PerCoreView,
    num_cores: int,
    device_dims: Sequence[int],
    out_elems: int,
    dtype_bytes: int,
    params=None,
    *,
    destination_num_cores: int | None = None,
) -> float | None:
    """Price one candidate relayout (source view -> destination view), in ns.

    Return ``None`` for matching ownership, unsupported movement, or a split
    outside the measured range. Permutations use the cost model's split range;
    grouped movements use [``relayout_split_min``, ``_GROUPED_SPLIT_MAX``].
    The caller also applies the consumer's core-domain and gather rules.

    Unequal complete partitions use the same estimate on the smaller domain;
    this is not a separately fitted law.
    It charges more than the larger-domain formula, not a proven upper bound
    on hardware time.

    Use the reporting path's feature vector and ``relayout_ns`` with the finer
    view's geometry. Build each view for its own core
    count: ``num_cores`` for the source and ``destination_num_cores`` for the
    destination (defaults to the source's count).
    """
    features = relayout_cost_features(
        source_view,
        destination_view,
        num_cores,
        device_dims,
        out_elems,
        dtype_bytes,
        destination_num_cores=destination_num_cores,
    )
    run_elems, split = features.relayout_run_elems, features.relayout_split
    p = params or CostParams()
    complete_partition = (
        source_view.split_product == num_cores
        and destination_view.split_product == (destination_num_cores or num_cores)
    )
    # A complete repartition moves each element once, with no destination
    # replicas. Use the partition law's range and retain the
    # smaller core count below: the larger slice governs send or receive work.
    # This is an estimate for unequal domains, not a new measured cost law.
    top = p.relayout_split_max if complete_partition else _GROUPED_SPLIT_MAX
    if run_elems <= 0 or not p.relayout_split_min <= split <= top:
        return None
    if not movement_supported(
        source_view, destination_view, num_cores, destination_num_cores or num_cores
    ):
        return None
    return relayout_ns(features, p)


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
        reduction = (
            completed_reduction_split_on_buf(producer, write, source_name)
            if partial
            else None
        )
        if (
            source_view is None
            or not representable
            or source_view.num_cores != source_num_cores
            or (
                partial
                and (
                    reduction is None
                    or source_num_cores != config.sencores
                    or not config.core_id_k_fast_emission
                )
            )
        ):
            continue

        # Activation eligibility belongs to the producer, not to an individual
        # edge. Only updated input caches escape the graph-input exclusion.
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
        source_unprojectable_to_consumer = False
        for consumer, dep in consumer_reads:
            consumer_name = consumer.get_name()
            if consumer_name in seen_consumers:
                rejection_reason = (
                    "cannot emit: consumer reads the source more than once"
                )
                break
            if not _supported_consumer_layout(graph, producer, consumer, source_name):
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
            view, _, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            # A split reduction makes the consumer's output partial, not its input.
            if view is None or not representable:
                rejection_reason = (
                    "cannot represent: consumer ownership is unrepresentable"
                )
                break
            consumer_num_cores = _op_num_cores(consumer)
            if reduction is None:
                rejection_reason = core_domain_rejection(
                    source_num_cores, consumer_num_cores
                )
                if rejection_reason is not None:
                    break
            consumer_coordinates = try_device_coordinates(
                producer.layout.device_layout, dep, None
            )
            if consumer_coordinates is None:
                rejection_reason = (
                    "cannot represent: consumer coordinates are unavailable"
                )
                break
            if reduction is None and view.same_partition(source_view):
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

            if reduction is None:
                rejection_reason = grouped_gather_rejection(
                    consumer, source_num_cores, view
                )
                if rejection_reason is not None:
                    break
            if consumer_num_cores > source_num_cores:
                failure = (
                    "cannot emit: grouped destination does not evenly "
                    "broadcast the source"
                )
            elif reduction is None and view.split_product < source_num_cores:
                failure = (
                    "cannot emit: grouped destination does not evenly contract "
                    "the source"
                )
            else:
                failure = "cannot emit: unsupported ownership transfer"

            try:
                routes = (
                    derive_completed_reduction_routes(source_view, view, reduction)
                    if reduction is not None
                    else ()
                )
                supported = (
                    bool(routes)
                    if reduction is not None
                    else movement_supported(
                        source_view, view, source_num_cores, consumer_num_cores
                    )
                )
            except (TypeError, ValueError) as exc:
                rejection_reason = (
                    f"cannot represent: invalid ownership partition: {exc}"
                )
                break
            if not supported:
                rejection_reason = failure
                break
            consumer_space = iteration_space_from_op(consumer)
            padded_axis = prefix_lx_contraction_axis(producer, consumer)
            if padded_axis is not None:
                if dict(view.work_slice_dims).get(padded_axis, 1) != 1:
                    rejection_reason = "cannot emit: padded LX contraction splits K"
                    break
                # Copy the full physical domain before the logical-K read.
                # Projecting 832-wide owners onto 800-wide logical quarters
                # would invent different ownership; the reader owns all K.
                consumer_coordinates = producer_coordinates
                consumer_space = iteration_space_from_op(producer)
            transfers.append(
                (consumer_name, consumer_coordinates, consumer_space, view, routes)
            )

        # Reuse the ownership comparison and preserve first-consumer order.
        destinations: list[tuple[PerCoreView, int, tuple, list[str]]] = []
        if rejection_reason is None:
            # Both footprint checks are before placement: an invalid or
            # unsupported candidate size declines this optional relayout,
            # with the exact reason logged below, leaving the original buffer.
            # This does not waive layout/codegen validation or catch assertions.
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
                routes,
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
                try:
                    destination_footprint = partition_footprint(
                        producer.layout, destination_view
                    )
                except (TypeError, ValueError) as exc:
                    rejection_reason = (
                        f"allocation: destination footprint is unavailable: {exc}"
                    )
                    break
                for group_view, footprint, group_routes, consumers in destinations:
                    if (
                        footprint == destination_footprint
                        and group_routes == routes
                        and group_view.same_partition(destination_view)
                    ):
                        consumers.append(consumer_name)
                        break
                else:
                    destinations.append(
                        (
                            destination_view,
                            destination_footprint,
                            routes,
                            [consumer_name],
                        )
                    )

        if rejection_reason is None:
            result.extend(
                LXRelayoutPlan(
                    source_name=source_name,
                    consumer_names=tuple(consumer_names),
                    source_view=source_view,
                    destination_view=destination_view,
                    num_cores=source_num_cores,
                    source_footprint_bytes=source_footprint,
                    destination_footprint_bytes=destination_footprint,
                    completed_producer_cores=tuple(core for core, _ in routes),
                )
                for (
                    destination_view,
                    destination_footprint,
                    routes,
                    consumer_names,
                ) in destinations
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


def anchor_lx_relayout_ownership(graph: GraphLowering) -> list[LXRelayoutPlan] | None:
    """Choose the unique canonical producer order accepted by its consumers.

    Work-division split counts are already final here. This pass only changes
    their canonical owner order, and only when the ordinary relayout planner
    proves one unique order makes the complete source group expressible. The
    allocator records the accepted physical view; kernel preparation consumes
    it without choosing another owner order. Return the collected plans for
    immediate allocation only if no ownership changed; otherwise recollect.
    """

    if (
        not config.lx_planner_relayout
        or config.co_optimizing_lx_planning
        or config.ktir_emitter
    ):
        return None

    unprojectable_sources: list[str] = []
    plans: list[LXRelayoutPlan] | None = collect_lx_relayout_plans(
        graph, unprojectable_sources=unprojectable_sources
    )
    operations = {op.get_name(): op for op in graph.operations}

    def direct_consumers_match(
        source_name: str,
        producer: ComputedBuffer,
        candidate: TensorWorkDivision,
    ) -> bool:
        """Whether every reader already agrees with ``candidate`` without movement.

        This is the direct path, not a zero-transfer planner result: it has
        fewer reader restrictions than movement planning and accepts on plain
        partition agreement.
        """

        write = _single_write(producer, source_name)
        readers = [
            (consumer, dep)
            for consumer in graph.operations
            for dep in op_read_writes(consumer).reads
            if isinstance(dep, MemoryDep) and dep.name == source_name
        ]
        if write is None or not readers:
            return False
        source_view, partial, representable = _per_core_view_on_buf(
            producer, write, source_name, ownership_override=candidate
        )
        if partial or not representable:
            return False
        for consumer, dep in readers:
            view, _, consumer_representable = _per_core_view_on_buf(
                consumer, dep, source_name
            )
            # Only the read partition matters, even if this reader splits its sum.
            if not consumer_representable or not view.same_partition(source_view):
                logger.debug(
                    "direct LX owner mismatch source=%s consumer=%s",
                    source_name,
                    consumer.get_name(),
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
        plans = None
        logger.debug(
            "consumer-anchored LX ownership source=%s mapping=%s",
            source_name,
            selected.core_id_to_work_slice,
        )
    return plans


def validate_relayout_spans(plan: LXRelayoutPlan) -> None:
    """Require disjoint positive byte spans on every shared physical core."""
    if plan.source_address is None or plan.destination_address is None:
        raise RuntimeError("LX relayout plan is missing an allocated address")
    if (
        plan.source_address < 0
        or plan.destination_address < 0
        or plan.source_footprint_bytes <= 0
        or plan.destination_footprint_bytes <= 0
    ):
        raise RuntimeError("LX relayout plan has an invalid physical span")
    if (
        plan.source_address
        < plan.destination_address + plan.destination_footprint_bytes
        and plan.destination_address < plan.source_address + plan.source_footprint_bytes
    ):
        raise RuntimeError("LX relayout source and destination spans overlap")


def materialize_lx_relayouts(graph: GraphLowering, plans: list[LXRelayoutPlan]) -> None:
    if not plans:
        if materialized_lx_relayouts(graph):
            raise RuntimeError("LX relayouts were already materialized")
        return
    from .graph_editor import GraphEditor

    copies = materialized_lx_relayouts(graph)
    if copies:
        raise RuntimeError("LX relayouts were already materialized")
    # Check every span before editing any consumer. A solver placement alone
    # must not authorize overwriting a live source during a cross-core gather.
    for plan in plans:
        validate_relayout_spans(plan)
    editor = GraphEditor(graph)
    setattr(graph, _REGISTRY, copies)
    for plan in plans:
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
        if any(prefix_lx_contraction_axis(source, c) is not None for c in consumers):
            for consumer in consumers:
                axis = prefix_lx_contraction_axis(source, consumer)
                if (
                    axis is None
                    or dict(plan.destination_view.work_slice_dims).get(axis, 1) != 1
                ):
                    raise RuntimeError("padded LX gather lost its full-K read contract")
        mutations = []
        for consumer in consumers:
            if isinstance(consumer.layout, MutationLayoutSHOULDREMOVE):
                if not _persistent_scatter_consumer(
                    graph, source, consumer, plan.source_name
                ):
                    raise RuntimeError(
                        "relayout cache mutation no longer has a safe source"
                    )
                mutations.append(
                    (
                        consumer,
                        consumer.layout,
                        op_read_writes(consumer).writes,
                        tuple(consumer.get_mutation_names()),
                    )
                )
        copy = editor.insert_clone_before_consumers(
            source,
            consumers,
            lx_view=plan.destination_view,
        )
        for consumer, layout, writes, names in mutations:
            if (
                consumer.layout is not layout
                or op_read_writes(consumer).writes != writes
                or tuple(consumer.get_mutation_names()) != names
            ):
                raise RuntimeError("LX input rewrite changed a persistent cache write")
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
