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
import math
from collections.abc import Iterable, Mapping, Sequence
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
from torch_spyre._C import ElementArrangement

from .. import config
from ..core_mapping import (
    core_mappings_equal,
    owner_slots,
    partition_physical_span_bytes,
    _loop_regions,
    _LOOP_POINT,
    _MAX_EXACT_DIRECT_AXIS_POINTS,
    _MAX_EXACT_OWNERSHIP_POINTS,
    _EVALUATION_ERRORS,
    select_unique_partition_division,
)
from ..cost_model import OpFeatures, relayout_ns
from ..dump_cost_model import governing_run_split
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
    source_footprint_bytes: int = 0
    destination_footprint_bytes: int = 0
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
    members of one segment into a :class:`LXRelayoutPlan`
    (:class:`RelayoutSegment`). Nothing downstream re-derives a view, a core
    count or a price from primitives, so a change to what a relayout *is*
    (another lowering kind, a measured footprint) is a change to this record
    and to the enumeration that builds it, nowhere else.

    ``group`` identifies the DESTINATION per-core view of ``parent``, interned
    per parent by the allocator for one solve: every candidate that lands on
    the same view of the same parent shares one shuffle and one LX destination,
    so the solver prices and places the group once, not per edge.

    Both views are built for ``num_cores`` (every core's owner slot within its
    split); the enumeration's ``cores_used`` equality gate guarantees that.
    """

    parent: str
    consumer: str
    source_division: int
    consumer_division: int
    group: int
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    cost_ns: float

    def __post_init__(self) -> None:
        if self.source_view.same_partition(self.destination_view):
            raise ValueError(
                f"relayout candidate {self.parent} -> {self.consumer} has equal "
                "views; that pair belongs to cd_parent_matches"
            )

    @property
    def group_key(self) -> tuple[str, int]:
        """The solver's registry key: one destination view of one parent."""
        return self.parent, self.group


@dataclasses.dataclass(frozen=True)
class ChosenRelayout:
    """A fired :class:`RelayoutCandidate` with its solved placement.

    ``run_head`` names the earliest consumer of the SEGMENT this consumer reads
    from: consumers of one group that the solver bridged onto one copy share a
    destination address and a head, and the commit path materializes one plan
    per head (:meth:`RelayoutSegment.from_chosen`).
    """

    candidate: RelayoutCandidate
    destination_address: int
    run_head: str

    def scaled(self, alignment: int) -> ChosenRelayout:
        """The same choice with the address converted from alignment units."""
        return dataclasses.replace(
            self, destination_address=self.destination_address * alignment
        )


@dataclasses.dataclass(frozen=True)
class RelayoutSegment:
    """A maximal run of consumers the solver bridged onto ONE relayout copy.

    One segment is one shuffle and one continuous LX residency at
    ``destination_address``; its members share the source division and the
    destination view by construction (they are members of one group whose
    rectangles were pinned to one offset by the bridge literals), which
    :meth:`from_chosen` verifies rather than trusts.
    """

    parent: str
    group: int
    run_head: str
    members: tuple[ChosenRelayout, ...]

    @property
    def candidate(self) -> RelayoutCandidate:
        """A representative member; every field the plan needs agrees across
        the segment (checked in :meth:`from_chosen`)."""
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
            source_address=source_address,
            destination_address=self.destination_address,
        )

    @classmethod
    def from_chosen(cls, chosen: Iterable[ChosenRelayout]) -> list[RelayoutSegment]:
        """Regroup fired edges by segment: (parent, destination view, head).

        Deterministic order (sorted keys, members sorted by consumer name) so
        plan construction, and hence destination naming, is reproducible.
        """
        by_segment: dict[tuple[str, int, str], list[ChosenRelayout]] = {}
        for ch in chosen:
            key = (ch.candidate.parent, ch.candidate.group, ch.run_head)
            by_segment.setdefault(key, []).append(ch)
        segments: list[RelayoutSegment] = []
        for (parent, group, head), members in sorted(by_segment.items()):
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
                    and m.destination_address == first.destination_address
                )
                if not agree:
                    raise AssertionError(
                        f"relayout segment {parent}/g{group}@{head}: members "
                        f"disagree on geometry or placement: {first} vs {m}"
                    )
            segments.append(cls(parent, group, head, tuple(members)))
        return segments


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


def partition_footprint(layout: FixedTiledLayout, view: PerCoreView) -> int:
    """Measure a relayout candidate in normalized standard device layout.

    FixedTiledLayout can wrap an explicit device shape, so its type alone does
    not guarantee a complete final stick axis. The span helper validates it.
    """
    device_layout = layout.device_layout
    if device_layout.element_arrangement != ElementArrangement.STANDARD:
        raise ValueError("relayout footprint requires standard element arrangement")
    return partition_physical_span_bytes(
        tuple(int(size) for size in device_layout.device_size),
        device_layout.device_dtype,
        dict(view.work_slice_dims),
    )


def _overlap(a: int, an: int, b: int, bn: int) -> bool:
    return a * bn < (b + 1) * an and b * an < (a + 1) * bn


def movement_supported(
    source: PerCoreView,
    destination: PerCoreView,
    source_num_cores: int,
    destination_num_cores: int,
) -> bool:
    """Extend the original full-partition check to gathers and broadcasts.

    Edges are ownership intersections, never a separate geometry calculation.
    A complete source may feed uniformly repeated destination slices. Across
    unequal core counts, only even broadcasts (one source per destination) are
    supported. Equal destination slices have identical sources by construction.
    """

    num_cores = source_num_cores
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    destination_slices = math.prod(destination_splits.values())
    if (
        num_cores <= 0
        or destination_num_cores < num_cores
        or source.num_cores != num_cores
        or destination.num_cores != destination_num_cores
        or destination_num_cores % num_cores
        or math.prod(source_splits.values()) != num_cores
        or destination_slices <= 0
        or destination_num_cores % destination_slices
        or (num_cores == destination_num_cores and source.same_partition(destination))
    ):
        return False
    if num_cores == destination_num_cores and destination_slices < num_cores:
        # This level only contracts existing source axes. #4152 lifts this
        # scope restriction for combined gather/broadcast on other axes.
        if any(
            source_splits.get(dim, 1) % destination_splits.get(dim, 1)
            for dim in source_splits.keys() | destination_splits.keys()
        ):
            return False
    source_map = _core_slices(source, num_cores)
    destination_map = _core_slices(destination, destination_num_cores)
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
    fanin = [
        sum(dst == core for _, dst in edges) for core in range(destination_num_cores)
    ]
    replicas = collections.Counter(
        tuple(sorted(row.items())) for row in destination_map.values()
    )
    return bool(edges) and all(
        (
            # Every source sends to the same number of destination cores.
            len(set(fanout)) == 1,
            # Every destination receives from the same number of source cores.
            len(set(fanin)) == 1,
            # Every source slice is present exactly once.
            len({tuple(sorted(row.items())) for row in source_map.values()})
            == num_cores,
            # Every distinct destination slice is covered.
            len(replicas) == destination_slices,
            # Within one core domain, each slice has equally many copies.
            num_cores != destination_num_cores or len(set(replicas.values())) == 1,
            # A larger domain only broadcasts: one source per destination.
            num_cores == destination_num_cores
            or (fanout[0] == destination_num_cores // num_cores and fanin[0] == 1),
        )
    )


def solver_relayout_movement_supported(
    source: PerCoreView, destination: PerCoreView, num_cores: int
) -> bool:
    """The movement shapes the solver may PRICE: uniform full permutations only.

    Two gates answer two different questions. The committed path's movement gate
    (``_compatible_partitions`` today; ``movement_supported`` once the
    ownership-flow rewrite lands, widened to grouped gathers and broadcasts by
    #3440) decides what the emitter CAN move. This gate decides what the fitted
    relayout law can price, which is narrower and must stay narrower however the
    committed gate grows: ``relayout_ns`` was fitted on uniform permutations,
    where every core sends to and receives from the same number of cores, both
    sides have ``num_cores`` distinct owners, and both split products equal
    ``num_cores``. Pricing a multicast or a broadcast with permutation constants
    would hand the objective a number the law never measured, so such pairs are
    declined here and stay unpriced until their own term is calibrated.

    Deliberately self-contained (it shares only ``_core_slices`` with the
    committed gate) so the committed gate can be replaced underneath without
    the solver's admission set changing by accident. The contract is
    "never looser than the committed gate", pinned by
    ``test_solver_gate_is_never_looser_than_the_committed_gate``.
    """
    if source.same_partition(destination):
        return False
    source_rows = _core_slices(source, num_cores)
    destination_rows = _core_slices(destination, num_cores)
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    if (
        math.prod(source_splits.values()) != num_cores
        or math.prod(destination_splits.values()) != num_cores
    ):
        return False
    distinct = lambda rows: len({tuple(sorted(r.items())) for r in rows.values()})  # noqa: E731
    if distinct(source_rows) != num_cores or distinct(destination_rows) != num_cores:
        return False

    def slices_overlap(a: int, an: int, b: int, bn: int) -> bool:
        # Slot a of an equal parts against slot b of bn equal parts, as
        # half-open intervals on the same unit axis.
        return a * bn < (b + 1) * an and b * an < (a + 1) * bn

    dims = set(source_splits) | set(destination_splits)
    edges = {
        (s_core, d_core)
        for s_core, s_slice in source_rows.items()
        for d_core, d_slice in destination_rows.items()
        if all(
            slices_overlap(
                s_slice.get(dim, 0),
                source_splits.get(dim, 1),
                d_slice.get(dim, 0),
                destination_splits.get(dim, 1),
            )
            for dim in dims
        )
    }
    if not edges:
        return False
    fanout = {sum(src == core for src, _ in edges) for core in range(num_cores)}
    fanin = {sum(dst == core for _, dst in edges) for core in range(num_cores)}
    return len(fanout) == 1 and len(fanin) == 1


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


def solver_relayout_edge_context(
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
    consumer_coords, producer_symbols, consumer_symbols)``, or ``None`` when the
    edge can never host a relayout.
    """
    # A coarse-tiled endpoint can never host a relayout. The fitted law has
    # no loop_trip factor (the committed-path planner already guarantees "a
    # relayout cannot be inside a coarse-tiling loop"), a tiled producer's
    # buffer is per-tile scratch rather than the full tensor, and a tiled
    # consumer reads cross-group data through a per-iteration staging op.
    # The MutationLayout check below only screens the loop's DRAIN op; the
    # staging and tiled compute ops are plain Pointwise buffers, so the
    # loop_info presence is the reliable marker.
    if (
        getattr(producer, "loop_info", None) is not None
        or getattr(consumer, "loop_info", None) is not None
    ):
        return None
    if (
        not isinstance(producer, ComputedBuffer)
        or not isinstance(producer.layout, FixedTiledLayout)
        or (write_dep := _single_write(producer, source_name)) is None
        or not _is_activation_source(operations, producer)
    ):
        return None
    if not isinstance(consumer, ComputedBuffer) or isinstance(
        consumer.layout, MutationLayoutSHOULDREMOVE
    ):
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
        tuple(iteration_space_from_op(producer)),
        tuple(iteration_space_from_op(consumer)),
    )


def solver_relayout_pair_cost(
    source_view: PerCoreView,
    destination_view: PerCoreView,
    num_cores: int,
    device_dims: Sequence[int],
    out_elems: int,
    dtype_bytes: int,
    params=None,
) -> float | None:
    """Price one candidate relayout (source view -> destination view), in ns.

    ``None`` when the pair cannot host a relayout, or should not be offered:

    - views with the same physical ownership need no relayout (that pair
      belongs to ``cd_parent_matches``), compared with ``same_partition`` so a
      differently spelled slot expression cannot masquerade as movement;
    - ``solver_relayout_movement_supported`` rejects everything but a uniform
      full permutation (``num_cores`` distinct owners on BOTH sides, split
      products equal to ``num_cores``, uniform fanout/fanin) - grouped gathers
      and broadcasts (#3440) fall out here and stay unpriced until their own
      term is calibrated, whatever the committed path's movement gate admits;
    - a governing split outside the law's fitted range [2, 8] is DECLINED, not
      clamped: the reporting path clamps because the shuffle it prices already
      exists, but the solver must never be offered an option at a price the
      law was not fitted for.

    The price is ``relayout_ns`` on a minimal feature vector - the same function
    the reporting path uses, so the two paths cannot drift.

    Both views must be built FOR ``num_cores`` (every core's owner slot within
    its split); the caller's cores_used equality gate guarantees that, and
    ``_core_slices`` asserts it rather than tolerating an out-of-range slot.
    """
    if not solver_relayout_movement_supported(source_view, destination_view, num_cores):
        return None
    run_elems, split = governing_run_split(source_view, destination_view, device_dims)
    if run_elems <= 0 or not 2 <= split <= 8:
        return None
    features = OpFeatures(
        name="lx_relayout",
        is_reduction=False,
        out_elems=out_elems,
        cores=num_cores,
        dtype_bytes=dtype_bytes,
        args=[],
        is_lx_relayout=True,
        relayout_run_elems=run_elems,
        relayout_split=split,
    )
    return relayout_ns(features, params)


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
            view, _, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            consumer_num_cores = _op_num_cores(consumer)
            # A split reduction makes the consumer's output partial, not its input.
            if view is None or not representable:
                rejection_reason = (
                    "cannot represent: consumer ownership is unrepresentable"
                )
                break
            if consumer_num_cores < source_num_cores:
                rejection_reason = (
                    "cannot emit: consumer uses fewer physical cores than producer"
                )
                break
            is_matmul = _is_matmul_op(consumer)
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
            if consumer_num_cores > source_num_cores:
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
                failure = (
                    "cannot emit: grouped destination does not evenly contract "
                    "the source"
                )
            else:
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
        destinations: list[tuple[PerCoreView, int, list[str]]] = []
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
                try:
                    destination_footprint = partition_footprint(
                        producer.layout, destination_view
                    )
                except (TypeError, ValueError) as exc:
                    rejection_reason = (
                        f"allocation: destination footprint is unavailable: {exc}"
                    )
                    break
                for group_view, footprint, consumers in destinations:
                    if footprint == destination_footprint and group_view.same_partition(
                        destination_view
                    ):
                        consumers.append(consumer_name)
                        break
                else:
                    destinations.append(
                        (destination_view, destination_footprint, [consumer_name])
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
                )
                for (
                    destination_view,
                    destination_footprint,
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
