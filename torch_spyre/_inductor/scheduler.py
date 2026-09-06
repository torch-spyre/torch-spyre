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

from typing import Iterator, Sequence, Union

import sympy

from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import (
    get_kernel_metadata,
    get_fused_kernel_name,
    sympy_product,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.scheduler import (
    BaseScheduling,
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch._inductor.codecache import code_hash
from torch.utils._ordered_set import OrderedSet

from .spyre_kernel import SpyreKernel
from .ir import FixedTiledLayout
from .pass_utils import (
    PerCoreView,
    iteration_space,
    iteration_space_from_op,
    per_core_view_scheduled,
    try_device_coordinates,
)
from .logging_utils import get_inductor_logger
from .scratchpad.lx_relayout import (
    demote_lx_relayout_group,
    materialized_lx_relayouts,
    work_division_from_view,
)
from .op_spec import LoopSpec
from . import config as _spyre_config
from .errors import Unsupported

logger = get_inductor_logger("scheduler")


def _ownership_projectable(
    node: SchedulerNode,
    dep: MemoryDep,
    name: str,
    view: PerCoreView,
) -> bool:
    """Whether final scheduled coordinates can carry ``view`` into codegen."""

    buffer = V.graph.try_get_buffer(name)
    if buffer is None:
        return False
    layout = buffer.get_layout()
    if not isinstance(layout, FixedTiledLayout):
        return False
    coordinates = try_device_coordinates(layout.device_layout, dep, None)
    if coordinates is None:
        return False
    try:
        work_division_from_view(view, coordinates, tuple(iteration_space(node)))
    except ValueError:
        return False
    return True


class CountedLoopSchedulerNode(FusedSchedulerNode):
    """A group of SchedulerNodes to be executed inside a counted outer loop.

    Produced by build_loop_scheduler_nodes from SchedulerNodes whose
    underlying ir.Operation has been stamped with a ``loop_info``
    (``CoarseTileInfo``) attribute by the coarse-tiling IR pass.

    loop_count is the trip count of the loop that directly contains this
    group's operations.  For nested loops, the snodes may themselves
    contain CountedLoopSchedulerNodes.
    """

    loop_count: sympy.Expr

    def __init__(
        self,
        scheduler,
        snodes: list[BaseSchedulerNode],
        loop_count: sympy.Expr,
    ) -> None:
        super().__init__(scheduler, snodes)
        self.loop_count = loop_count

    @classmethod
    def create(  # type: ignore[override]
        cls,
        snodes: list[BaseSchedulerNode],
        loop_count: sympy.Expr,
    ) -> "CountedLoopSchedulerNode":
        scheduler = snodes[0].scheduler
        assert all(node.scheduler is scheduler for node in snodes)
        grouped = cls(scheduler, snodes, loop_count)
        for snode in snodes:
            scheduler.name_to_fused_node[snode.get_name()] = grouped
        scheduler.name_to_fused_node[grouped.get_name()] = grouped
        return grouped

    def unpack(self) -> list[BaseSchedulerNode]:
        # CountedLoopSchedulerNode is an atomic codegen unit; do not unpack.
        return [self]

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        return False


def _loop_group_id(node: BaseSchedulerNode):
    """Return the loop_group_id of the ir.Operation inside node, or None."""
    for snode in node.get_nodes():
        if isinstance(snode, SchedulerNode) and snode.node is not None:
            loop_info = getattr(snode.node, "loop_info", None)
            if loop_info is not None:
                return loop_info.loop_group_id
    return None


def _loop_count(node: BaseSchedulerNode, depth: int) -> sympy.Expr:
    """Return the loop_count for ``depth`` from the ir.Operation inside node.

    ``loop_count`` on the ir.Operation is a list of trip counts, one per
    nesting level from outermost to innermost (stamped by
    coarse_tile_pre_stickify()/coarse_tile_post_stickify()).
    ``depth`` is the absolute nesting depth being queried (0 = outermost).

    For a flat (depth-1) op, ``loop_count = [K]`` and only depth 0 is valid.
    For a nested op with ``loop_group_id = (g, 0)``, ``loop_count = [K1, K2]``
    and depth 0 → K1, depth 1 → K2.
    """
    for snode in node.get_nodes():
        if isinstance(snode, SchedulerNode) and snode.node is not None:
            loop_info = getattr(snode.node, "loop_info", None)
            if loop_info is not None:
                counts: list = loop_info.loop_count
                gid = loop_info.loop_group_id
                # coarse_tile stamps one count per nesting level, so
                # len(counts) == len(gid) always holds.
                assert len(counts) == len(gid), (
                    f"loop_count length {len(counts)} != loop_group_id depth {len(gid)}"
                )
                if 0 <= depth < len(counts):
                    return counts[depth]
    raise AssertionError(f"Node {node.get_name()} has no loop_count for depth {depth}")


def _regroup_by_outer_loop_key(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Reorder ``nodes`` so every outermost loop_group_id run is contiguous.

    Inductor's own ``Scheduler.topological_sort_schedule`` runs (twice) before
    this pass ever sees the node list, via a plain DFS over
    ``unmet_dependencies``.  That DFS only guarantees a *valid* topological
    order — it does not preserve the original relative order of mutually
    independent nodes, so it can interleave unrelated nodes into the middle
    of what coarse_tile.py built as a single contiguous loop group.

    A naive "stable sort by first occurrence of the group key" is unsound: it
    can hoist a later group member forward past an interleaved node it
    genuinely depends on, producing an invalid order. Instead, merge every
    node sharing an outermost loop_group_id[0] key into one virtual unit for
    ordering purposes — its dependency set is the union of its members' real
    unmet_dependencies on buffers produced outside the group — and run a
    dependency-respecting DFS (mirroring topological_sort_schedule's own
    shape) over {merged units, ungrouped nodes}. Each unit then expands back
    into its original members in their original relative order, which is
    always safe because that intra-group order is coarse_tile's deliberate
    op sequence, not something this pass reorders.

    The result is a valid topological order (edges are the real edges of the
    original graph) in which every outermost loop group is contiguous by
    construction.
    """
    name_to_node: dict[str, BaseSchedulerNode] = {}
    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node

    outer_key_to_unit: dict[object, list[BaseSchedulerNode]] = {}
    units: list[Union[BaseSchedulerNode, list[BaseSchedulerNode]]] = []
    unit_of_node: dict[int, Union[BaseSchedulerNode, list[BaseSchedulerNode]]] = {}

    for node in nodes:
        gid = _loop_group_id(node)
        outer_key = gid[0] if gid is not None else None
        if outer_key is None:
            units.append(node)
            unit_of_node[id(node)] = node
            continue
        unit = outer_key_to_unit.get(outer_key)
        if unit is None:
            unit = []
            outer_key_to_unit[outer_key] = unit
            units.append(unit)
        unit.append(node)
        unit_of_node[id(node)] = unit

    def unit_key(unit) -> int:
        return id(unit)

    unit_deps: dict[int, OrderedSet] = {}
    unit_members: dict[int, list[BaseSchedulerNode]] = {}
    for unit in units:
        members = unit if isinstance(unit, list) else [unit]
        member_ids = {id(m) for m in members}
        deps: OrderedSet = OrderedSet()
        for member in members:
            for dep in member.unmet_dependencies:
                producer = name_to_node.get(dep.name)
                if producer is None or id(producer) in member_ids:
                    continue
                deps.add(unit_key(unit_of_node[id(producer)]))
        unit_deps[unit_key(unit)] = deps
        unit_members[unit_key(unit)] = members

    seen: set = set()
    ordered_units: list = []

    def visit(key: int) -> None:
        if key in seen:
            return
        seen.add(key)
        for dep_key in unit_deps[key]:
            visit(dep_key)
        ordered_units.append(key)

    for unit in units:
        visit(unit_key(unit))

    result: list[BaseSchedulerNode] = []
    for key in ordered_units:
        result.extend(unit_members[key])
    return result


def _build_loop_group(
    nodes: list[BaseSchedulerNode], depth: int
) -> list[BaseSchedulerNode]:
    """Recursively wrap contiguous runs sharing a loop_group_id into CountedLoopSchedulerNodes.

    depth is the nesting level being processed (0 = outermost).  Each node's
    loop_group_id is a tuple; we group on element [depth].

    Callers are expected to have already made outermost (depth 0) runs
    contiguous via ``_regroup_by_outer_loop_key`` — this function itself only
    scans linearly and does not tolerate gaps.
    """
    result: list[BaseSchedulerNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        gid = _loop_group_id(node)
        if gid is None or len(gid) <= depth:
            result.append(node)
            i += 1
            continue

        outer_key = gid[depth]
        # Every node in the run (regardless of path length) supplies the count
        # for this depth via its loop_count list.  Read it from the first node
        # and verify all others agree.
        count = _loop_count(node, depth)
        run = [node]
        i += 1
        while i < len(nodes):
            next_gid = _loop_group_id(nodes[i])
            if (
                next_gid is None
                or len(next_gid) <= depth
                or next_gid[depth] != outer_key
            ):
                break
            next_count = _loop_count(nodes[i], depth)
            assert next_count == count, (
                f"Loop group {outer_key} has inconsistent loop_count at depth "
                f"{depth}: {count} vs {next_count}"
            )
            run.append(nodes[i])
            i += 1

        # Recursively wrap any deeper nesting within this run.
        inner = _build_loop_group(run, depth + 1)
        result.append(CountedLoopSchedulerNode.create(inner, count))

    return result


def build_loop_scheduler_nodes(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Pre-fusion pass: wrap loop-group SchedulerNodes into CountedLoopSchedulerNodes.

    Reads the ``loop_info`` (``CoarseTileInfo``) attribute stamped on
    ir.Operation objects by the coarse-tiling IR pass.  Nodes without these attributes
    are passed through unchanged.

    loop_group_id is a tuple of ints encoding the nesting path, e.g.
    (0,) for an outermost group, (0, 1) for a nested group inside group 0.
    Nodes sharing the same outermost key are made contiguous by
    ``_regroup_by_outer_loop_key`` before grouping, since Inductor's own
    ``Scheduler.topological_sort_schedule`` (a DFS that runs twice before
    this pass ever sees the node list) does not preserve the tiling pass's
    intended contiguous ordering for mutually independent nodes.

    Running before Inductor's fusion pass ensures CountedLoopSchedulerNodes are
    visible to SuperDSCScheduling.can_fuse_vertical/horizontal (which return False),
    so loop groups survive Inductor fusion intact.  spyre_fuse_nodes is separately
    aware of CountedLoopSchedulerNodes: they are accumulated alongside plain
    SchedulerNodes and may share a bundle with adjacent ops.
    """
    nodes = _regroup_by_outer_loop_key(nodes)
    result = _build_loop_group(nodes, depth=0)

    # _regroup_by_outer_loop_key guarantees outermost runs are contiguous by
    # construction; this is a defensive check, not the primary correctness
    # mechanism.  A failure here would indicate a bug in that construction.
    seen: dict[tuple, str] = {}
    for node in result:
        if isinstance(node, CountedLoopSchedulerNode):
            gid = _loop_group_id(node.get_nodes()[0])
            if gid is not None:
                key = gid[0:1]
                name = node.get_name()
                if key in seen and seen[key] != name:
                    raise RuntimeError(
                        f"Loop group {key} is not contiguous in the scheduler node list "
                        "after _regroup_by_outer_loop_key. This indicates a bug in that "
                        "regrouping, not a data-flow issue in the tiling pass."
                    )
                seen[key] = name

    return result


def _lx_resident(node: SchedulerNode) -> bool:
    """True if ``node``'s output buffer was pinned into LX by scratchpad planning."""
    allocation = getattr(getattr(node.node, "layout", None), "allocation", None)
    return allocation is not None and "lx" in allocation


def _lx_view(name: str):
    buffer = V.graph.try_get_buffer(name)
    if buffer is None:
        return None
    layout = buffer.get_layout()
    if not isinstance(layout, FixedTiledLayout):
        return None
    if "lx" not in layout.allocation:
        return None
    return layout.lx_view


def align_lx_producer_loop_order(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Pre-fusion pass: match an LX buffer's producer loop order to its consumers'.

    LX is per-core scratchpad, so every op touching an LX-resident buffer must
    agree on which core owns which slice.  That mapping is *positional*:
    ``core_to_slice_mapping`` hands out ``core_id`` strides in iteration-space
    order, so a producer that walks the buffer in a different dim order than its
    consumers read it gets a transposed core->slice assignment.  The split
    factors still multiply to the same core count, so nothing downstream
    complains -- each core simply reads the slice a different core wrote, and the
    kernel silently returns another core's data.

    Scratchpad planning creates the clone that pins a graph input into LX, and it
    builds that clone in the buffer's natural dim order, which need not match how
    the consumers read it.  Through PyTorch 2.12 Inductor's
    ``loop_ordering_after_fusion`` happened to rewrite the clone into the
    consumers' order, so the assignments lined up by accident.  As of 2.13 the
    reorder is computed and then discarded (see
    ``Scheduler._try_reorder_loops_for_candidates``), which exposed the
    incoherence as wrong results for any two reductions sharing one LX-pinned
    input.  Align the orders here so correctness does not rest on an Inductor
    scoring heuristic.

    Consumers of an LX buffer are already known to agree with each other -- a
    disagreement is a core-division mismatch that keeps the buffer in HBM (see
    ``get_ncores_for_buffers``) -- so matching the first consumer matches all.
    """
    producers: dict[str, SchedulerNode] = {}
    for node in nodes:
        if isinstance(node, SchedulerNode) and _lx_resident(node):
            for dep in node.read_writes.writes:
                if isinstance(dep, MemoryDep) and _lx_view(dep.name) is None:
                    producers[dep.name] = node

    if not producers:
        return nodes

    # Keyed by producer, not by buffer: reordering a producer twice would leave
    # it matching only whichever consumer came last.  A ComputedBuffer has a
    # single output (multi-output ops carry no device_layout and never reach LX),
    # so one alignment per producer covers every LX buffer it writes.
    aligned: OrderedSet[str] = OrderedSet()
    for node in nodes:
        if not isinstance(node, SchedulerNode):
            continue
        for read in node.read_writes.reads:
            if not isinstance(read, MemoryDep):
                continue
            producer = producers.get(read.name)
            if producer is None or producer is node:
                continue
            if producer.get_name() in aligned:
                continue
            write = next(
                (
                    dep
                    for dep in producer.read_writes.writes
                    if isinstance(dep, MemoryDep) and dep.name == read.name
                ),
                None,
            )
            if write is None:
                continue
            # Reorders `producer`'s loops so its write dep matches `read`.
            if producer.reorder_loops_by_dep_pair(write, read):
                aligned.add(producer.get_name())
                logger.debug(
                    "align_lx_producer_loop_order: %s reordered to match %s's "
                    "read of LX buffer %s",
                    producer.get_name(),
                    node.get_name(),
                    read.name,
                )

    return nodes


def demote_incoherent_lx_buffers(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Post-fusion pass: drop an LX buffer whose users disagree on core->slice.

    LX planning runs before the Scheduler exists, so it reasons about each op's
    *pre*-scheduler ranges. ``core_to_slice_mapping`` is positional -- it hands
    ``core_id`` strides out in iteration-space order -- and Inductor's
    ``loop_ordering_after_fusion`` may permute a fused op's ranges after planning
    has already committed. When it permutes one user of an LX buffer and not
    another, the two disagree about which core owns which slice: each core writes
    one slice and reads back a different one. LX is per-core scratchpad with no
    other copy, so the read is silently wrong (#2062).

    Planning cannot see that permutation, so re-check here, where the ranges are
    final, and demote any buffer whose users no longer agree. Clearing ``"lx"``
    is all that is needed: this runs before ``hbm_pool_planning``, which claims
    exactly the intermediates LX did not, so a demoted buffer lands in the HBM
    intermediates segment on its way through.

    Deliberately verification-only -- it never *adds* residency and never
    rewrites a loop order, so it cannot perturb a graph whose users already
    agree.

    Complements :func:`align_lx_producer_loop_order`, which runs pre-fusion and
    rewrites a producer's loop order to match its consumers'. That pass fixes the
    incoherence it can reach; this one is the backstop for what it cannot -- a
    disagreement introduced after it ran, or a view too irregular to represent --
    where the only safe answer is to give up LX residency.
    """
    if not _spyre_config.lx_planning:
        return nodes

    # dep is needed per (node, buffer), including in-place read/write pairs.
    users: dict[str, list[tuple[SchedulerNode, MemoryDep]]] = {}
    lx_names: OrderedSet[str] = OrderedSet()
    scheduled = [
        inner
        for node in nodes
        for inner in node.get_nodes()
        if isinstance(inner, SchedulerNode)
    ]
    plans_by_copy = {
        copy_name: plan
        for copy_name, plan in materialized_lx_relayouts(V.graph).values()
    }
    source_by_copy = {
        copy_name: plan.source_name for copy_name, plan in plans_by_copy.items()
    }
    relayout_sources = set(source_by_copy.values())

    copy_reads = set()
    invalid_sources = {}
    seen_copies = set()
    for node in scheduled:
        if _lx_resident(node):
            for dep in node.read_writes.writes:
                if isinstance(dep, MemoryDep):
                    lx_names.add(dep.name)
        rw = node.read_writes
        reads = [dep for dep in rw.reads if isinstance(dep, MemoryDep)]
        writes = [dep for dep in rw.writes if isinstance(dep, MemoryDep)]
        for dep in [*reads, *writes]:
            users.setdefault(dep.name, []).append((node, dep))
        copies = [dep for dep in writes if dep.name in plans_by_copy]
        if not copies:
            continue
        for dep in copies:
            seen_copies.add(dep.name)
            plan = plans_by_copy[dep.name]
            source_view = _lx_view(plan.source_name)
            destination_view = _lx_view(dep.name)
            if (
                source_view is None
                or destination_view is None
                or source_view == destination_view
                or len(reads) != 1
                or len(writes) != 1
                or reads[0].name != plan.source_name
                or not _ownership_projectable(
                    node, reads[0], plan.source_name, source_view
                )
                or not _ownership_projectable(node, dep, dep.name, destination_view)
            ):
                invalid_sources[plan.source_name] = f"invalid relayout copy {dep.name}"
            else:
                copy_reads.add((node.get_name(), plan.source_name))
    for copy_name, plan in plans_by_copy.items():
        if copy_name not in seen_copies:
            invalid_sources[plan.source_name] = f"missing relayout copy {copy_name}"

    demoted = set()

    def demote(source_name: str, reason: str) -> None:
        if source_name in demoted:
            return
        demoted.add(source_name)
        if source_name in relayout_sources:
            # Scheduling supplies the final ownership verdict; the relayout
            # layer owns atomic group fallback and registry cleanup.
            demote_lx_relayout_group(V.graph, source_name, reason)
            return
        buffer = V.graph.try_get_buffer(source_name)
        if buffer is not None:
            layout = buffer.get_layout()
            if isinstance(layout, FixedTiledLayout):
                layout.allocation.pop("lx", None)
                layout.lx_view = None
        logger.info("demoted %s out of LX: %s", source_name, reason)

    for source_name, reason in invalid_sources.items():
        demote(source_name, reason)

    for name in lx_names:
        ref = None
        culprit = None
        expected = _lx_view(name)
        for node, dep in users.get(name, []):
            # The copy executes with its destination division; the source map is
            # carried by its input tensor and validated at the producer.
            if (node.get_name(), name) in copy_reads:
                continue
            view, _, representable = per_core_view_scheduled(node, dep, name)
            if not representable:
                culprit = f"{node.get_name()} view unrepresentable"
                break
            if expected is not None:
                if view != expected:
                    culprit = f"{node.get_name()} view {view} != {expected}"
                    break
                if not _ownership_projectable(node, dep, name, expected):
                    culprit = f"{node.get_name()} ownership unprojectable"
                    break
                continue
            if ref is None:
                ref = view
            elif view != ref:
                culprit = f"{node.get_name()} disagrees: {view} != {ref}"
                break
        if culprit is None:
            continue
        demote(source_by_copy.get(name, name), culprit)

    return nodes


def verify_carried_reduction_ownership(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Verify the final physical contract of every loop-carried reduction.

    This runs after fusion, LX demotion, and HBM-pool planning.  Earlier split
    metadata is only an input request; the scheduled per-core views checked
    here are the ownership codegen will actually emit.
    """

    def all_scheduler_nodes(
        items: Sequence[BaseSchedulerNode],
    ) -> Iterator[SchedulerNode]:
        for item in items:
            if isinstance(item, FusedSchedulerNode):
                yield from all_scheduler_nodes(item.get_nodes())
            elif isinstance(item, SchedulerNode):
                yield item

    grouped: dict[object, dict[str, SchedulerNode]] = {}
    for node in all_scheduler_nodes(nodes):
        record = getattr(node.node, "_carried_reduction_record", None)
        if record is not None:
            grouped.setdefault(record, {})[node.get_name()] = node

    for record, by_name in grouped.items():
        expected_names = {
            record.fill_name,
            record.combine_name,
            record.drain_name,
        }
        missing = expected_names - by_name.keys()
        if missing:
            raise Unsupported(
                "carried reduction lost physical stages after fusion: "
                f"accumulator={record.accumulator_name}, missing={sorted(missing)}"
            )

        accumulator = V.graph.try_get_buffer(record.accumulator_name)
        if accumulator is None:
            raise Unsupported(
                f"carried reduction accumulator {record.accumulator_name} is missing"
            )
        layout = accumulator.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(
                f"carried reduction accumulator {record.accumulator_name} has "
                f"non-device layout {type(layout).__name__}"
            )
        if "lx" not in layout.allocation:
            logger.warning(
                "carried reduction %s remained in HBM; execution is correct but "
                "the persistent-LX performance contract was not realized",
                record.accumulator_name,
            )
            continue

        checks = (
            (record.fill_name, "write"),
            (record.combine_name, "read"),
            (record.combine_name, "write"),
            (record.drain_name, "read"),
        )
        expected_view = layout.lx_view
        for op_name, access in checks:
            node = by_name[op_name]
            deps = (
                node.read_writes.reads if access == "read" else node.read_writes.writes
            )
            dep = next(
                (
                    candidate
                    for candidate in deps
                    if isinstance(candidate, MemoryDep)
                    and candidate.name == record.accumulator_name
                ),
                None,
            )
            if dep is None and access == "write" and op_name == record.combine_name:
                # MutationLayout's scheduled write is named after the combine
                # op, while its storage target is the carried accumulator.
                memory_deps = [
                    candidate for candidate in deps if isinstance(candidate, MemoryDep)
                ]
                dep = memory_deps[0] if len(memory_deps) == 1 else None
            if dep is None and access == "read" and op_name == record.drain_name:
                # Mutation propagation names this dependency after the latest
                # in-place writer, but it still reads the accumulator storage.
                memory_deps = [
                    candidate for candidate in deps if isinstance(candidate, MemoryDep)
                ]
                dep = memory_deps[0] if len(memory_deps) == 1 else None
            if dep is None:
                raise Unsupported(
                    f"carried reduction {op_name} lost its {access} of "
                    f"{record.accumulator_name}; deps="
                    f"{[(type(candidate).__name__, candidate.name) for candidate in deps]}"
                )
            view, _, representable = per_core_view_scheduled(
                node, dep, record.accumulator_name
            )
            if not representable:
                raise Unsupported(
                    f"carried reduction {op_name} {access} ownership is not "
                    "representable"
                )
            if expected_view is None:
                expected_view = view
            elif view != expected_view:
                raise Unsupported(
                    f"carried reduction {op_name} {access} ownership {view} "
                    f"does not match accumulator ownership {expected_view}"
                )

            # Equal physical views are necessary but not sufficient.  A
            # 32-way split over H and a 32-way split over T can have the same
            # total core count while assigning different logical rows to a
            # core.  Project the final physical view back into this stage's
            # scheduled loop symbols, then require the one split to be the
            # row dimension named by the immutable carried-reduction record.
            coordinates = try_device_coordinates(layout.device_layout, dep, None)
            if coordinates is None:
                raise Unsupported(
                    f"carried reduction {op_name} {access} cannot map final "
                    "accumulator coordinates back to operation loops"
                )
            try:
                realized_division = work_division_from_view(
                    view,
                    coordinates,
                    tuple(iteration_space(node)),
                )
            except ValueError as exc:
                raise Unsupported(
                    f"carried reduction {op_name} {access} cannot project final "
                    f"ownership into operation loops: {exc}"
                ) from exc
            if realized_division is None:
                raise Unsupported(
                    f"carried reduction {op_name} {access} has no final work division"
                )

            # Scheduler dependency extraction alpha-renames operation symbols
            # (for example, d0 -> c0) without changing dimension order.  The
            # carried-reduction rewrite is explicitly limited to
            # order-preserving pointwise stages, so translate the names across
            # that rename here.  This is not a general positional-remapping
            # facility: a rank change fails closed, and no other rewrite uses
            # this path.
            operation_symbols = tuple(iteration_space_from_op(node.node))
            scheduled_symbols = tuple(iteration_space(node))
            if len(operation_symbols) != len(scheduled_symbols):
                raise Unsupported(
                    f"carried reduction {op_name} {access} changed iteration rank "
                    "before final ownership verification"
                )
            operation_named_dims = getattr(node.node, "work_div_loop_info", {})
            named_dims = {
                scheduled_symbol: operation_named_dims.get(operation_symbol, [])
                for operation_symbol, scheduled_symbol in zip(
                    operation_symbols, scheduled_symbols
                )
            }
            row_symbols = [
                symbol
                for symbol in realized_division.work_slices
                if record.row_dim_name in named_dims.get(symbol, [])
            ]
            expected_splits = (
                {row_symbols[0]: record.required_row_split}
                if len(row_symbols) == 1
                else None
            )
            if realized_division.work_slices != expected_splits:
                raise Unsupported(
                    f"carried reduction {op_name} {access} expected only "
                    f"{record.row_dim_name} split={record.required_row_split}, but "
                    f"final logical splits are {realized_division.work_slices}"
                )

        assert expected_view is not None

    return nodes


class SuperDSCScheduling(BaseScheduling):
    def group_fn(self, sizes):
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def flush(self):
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        # Overrides superclass method that raises NotImplementedError.
        pass

    def can_buffer_be_removed_through_fusion(
        self, name: str, fused_node_names: OrderedSet[str]
    ) -> bool:
        """
        Spyre currently needs intermediate buffers to be allocated even if only used within a single Kernel.
        TODO: Revisit this as part of https://github.com/torch-spyre/torch-spyre/issues/1266
        """
        return False

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        # TODO: Revisit this as part of https://github.com/torch-spyre/torch-spyre/issues/826
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        # TODO: Revisit this as part of https://github.com/torch-spyre/torch-spyre/issues/826
        return False

    def generate_node_schedule(self, nodes: Sequence[BaseSchedulerNode]):
        node_schedule: list[SchedulerNode] = []
        done = OrderedSet[BaseSchedulerNode]()
        for node in nodes:
            if node in done:
                continue
            done.add(node)
            if isinstance(node, SchedulerNode):
                node_schedule.append(node)
            elif isinstance(node, FusedSchedulerNode):
                for inner in node.get_nodes():
                    if inner not in done and isinstance(inner, SchedulerNode):
                        done.add(inner)
                        node_schedule.append(inner)
            else:
                raise RuntimeError(f"Unexpected node type: {type(node)}")
        return node_schedule

    def _collect_layout_restores(self, node_schedule) -> list:
        """Select the layout restores to emit after a kernel call.

        Walks the kernel's nodes for _emit_set_layout tags set by
        insert_post_mutation_restickify and dedups them against the ones already
        emitted by earlier kernels, so each target restores once across the whole
        graph. Selection is the scheduler's job (it owns the node list and the
        cross-kernel dedup state); the kernel just emits the returned list.
        """
        # Dedup is graph-scoped: a target's device layout must be restored
        # exactly once across the whole generated program, not once per kernel.
        # The state lives on V.graph (one GraphLowering per compilation), so it
        # starts empty for each graph without any explicit reset.
        emitted = V.graph.__dict__.setdefault("_emitted_layout_targets", set())
        restores = []
        for snode in node_schedule:
            emit = getattr(getattr(snode, "node", None), "_emit_set_layout", None)
            if emit is not None and emit[0] not in emitted:
                emitted.add(emit[0])
                restores.append(emit)
        return restores

    def codegen_node(
        self, node: Union[FusedSchedulerNode, SchedulerNode, CountedLoopSchedulerNode]
    ) -> None:
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        if isinstance(node, CountedLoopSchedulerNode):
            self._codegen_counted_loop(node)
            return

        assert self.scheduler
        nodes = [
            n
            for n in node.get_nodes()
            if n.get_name() not in self.scheduler.removed_ops
        ]
        if len(nodes) == 0:
            return

        pool_sizes = getattr(V.graph, "hbm_pool_sizes", {})
        kernel = SpyreKernel(pool_size=pool_sizes.get(node.get_name(), 0))
        all_schedule_nodes: list[SchedulerNode] = []
        with kernel:
            self._codegen_into_kernel(nodes, kernel, all_schedule_nodes)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, all_schedule_nodes, kernel)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        with V.set_kernel_handler(kernel):
            for snode in all_schedule_nodes:
                snode.mark_run()

        self.codegen_comment(all_schedule_nodes, kernel_name)
        kernel.call_kernel(kernel.kernel_name)
        kernel.emit_layout_restores(self._collect_layout_restores(all_schedule_nodes))

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        self.free_buffers_in_scheduler()

    def _codegen_counted_loop(self, node: CountedLoopSchedulerNode) -> None:
        """Generate a kernel for a counted loop group."""
        assert self.scheduler
        inner_nodes = [
            n
            for n in node.get_nodes()
            if n.get_name() not in self.scheduler.removed_ops
        ]
        if len(inner_nodes) == 0:
            return

        pool_sizes = getattr(V.graph, "hbm_pool_sizes", {})
        kernel = SpyreKernel(pool_size=pool_sizes.get(node.get_name(), 0))
        all_schedule_nodes: list[SchedulerNode] = []
        with kernel:
            self._codegen_into_kernel(inner_nodes, kernel, all_schedule_nodes)

        kernel.wrap_op_specs_in_loop(node.loop_count)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, all_schedule_nodes, kernel)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        with V.set_kernel_handler(kernel):
            for snode in all_schedule_nodes:
                snode.mark_run()

        self.codegen_comment(all_schedule_nodes, kernel_name)
        kernel.call_kernel(kernel.kernel_name)
        kernel.emit_layout_restores(self._collect_layout_restores(all_schedule_nodes))

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        self.free_buffers_in_scheduler()

    def _codegen_loop_body(
        self,
        node: CountedLoopSchedulerNode,
        kernel: SpyreKernel,
        all_schedule_nodes: list[SchedulerNode],
        depth: int = 1,
    ) -> None:
        """Codegen the body of a nested CountedLoopSchedulerNode into an existing kernel.

        The inner ops are added to the kernel's op_specs list, then wrapped
        in a LoopSpec for the inner loop count.  Called from
        _codegen_counted_loop to handle nesting without creating a separate kernel.
        """
        assert self.scheduler
        inner_nodes = [
            n
            for n in node.get_nodes()
            if n.get_name() not in self.scheduler.removed_ops
        ]
        body_start = len(kernel.op_specs)
        for inner in inner_nodes:
            if isinstance(inner, CountedLoopSchedulerNode):
                self._codegen_loop_body(inner, kernel, all_schedule_nodes, depth + 1)
            else:
                sched = self.generate_node_schedule([inner])
                all_schedule_nodes.extend(sched)
                for snode in sched:
                    var_ranges = iteration_space(snode)
                    vs = list(var_ranges.keys())
                    index_vars = [
                        vs[: len(snode._body.iter_vars)],
                        vs[len(snode._body.iter_vars) :],
                    ]
                    snode.codegen(index_vars)

        # Wrap only the newly-added op_specs entries in this inner LoopSpec.
        body = kernel.op_specs[body_start:]
        kernel.op_specs = kernel.op_specs[:body_start]
        kernel.op_specs.append(LoopSpec(count=node.loop_count, body=body))

    def _codegen_into_kernel(
        self,
        nodes: list[BaseSchedulerNode],
        kernel: SpyreKernel,
        all_schedule_nodes: list[SchedulerNode],
    ) -> None:
        """Codegen a sequence of nodes into an existing kernel in order.

        Each CountedLoopSchedulerNode is driven via _codegen_loop_body so its
        ops land as a LoopSpec entry in kernel.op_specs.  Plain SchedulerNodes
        are codegenned flat.  The two types may appear in any order.
        """
        for node in nodes:
            if isinstance(node, CountedLoopSchedulerNode):
                self._codegen_loop_body(node, kernel, all_schedule_nodes)
            else:
                sched = self.generate_node_schedule([node])
                all_schedule_nodes.extend(sched)
                for snode in sched:
                    var_ranges = iteration_space(snode)
                    vs = list(var_ranges.keys())
                    index_vars = [
                        vs[: len(snode._body.iter_vars)],
                        vs[len(snode._body.iter_vars) :],
                    ]
                    snode.codegen(index_vars)

    def define_kernel(self, src_code, node_schedule, kernel):
        """
        Codegen kernel definition to go in output wrapper code
        """
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = get_fused_kernel_name(node_schedule, "original_aten")
            method = "ktir" if _spyre_config.ktir_emitter else "sdsc"
            kernel_name = "_".join([method, fused_name, wrapper.next_kernel_suffix()])
            wrapper.src_to_kernel[src_code] = kernel_name
            buf = IndentedBuffer()
            buf.writeline(f"async_compile.{method}('{kernel_name}',")
            with buf.indent():
                buf.splice(f"{src_code}")
            if method == "sdsc" and kernel._kernel_uses_hbm_pool():
                buf.writeline(f", pool_size={kernel.pool_size})")
            else:
                buf.writeline(")")
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
