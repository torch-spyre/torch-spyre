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
from torch._inductor.ir import NoneLayout
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
from .pass_utils import iteration_space
from .logging_utils import get_inductor_logger
from .scratchpad.lx_relayout import (
    demote_lx_relayout_group,
    materialized_lx_relayout_for_destination,
)
from .op_spec import LoopSpec
from . import config as _spyre_config
from .errors import Unsupported

logger = get_inductor_logger("scheduler")


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


def _lx_layout(name: str):
    buffer = V.graph.try_get_buffer(name)
    if buffer is None:
        return None
    # Fallback kernels can register dependency-only buffers whose NoneLayout
    # intentionally has no tensor descriptor.  Such a buffer cannot be LX
    # resident, and Buffer.get_layout() raises for it by contract.
    if isinstance(getattr(buffer, "layout", None), NoneLayout):
        return None
    layout = buffer.get_layout()
    if not isinstance(layout, FixedTiledLayout):
        return None
    if "lx" not in layout.allocation:
        return None
    return layout


def _all_scheduler_nodes(
    items: Sequence[BaseSchedulerNode],
) -> Iterator[SchedulerNode]:
    """Yield every leaf operation, including leaves inside counted loops."""

    for item in items:
        if isinstance(item, FusedSchedulerNode):
            yield from _all_scheduler_nodes(item.get_nodes())
        elif isinstance(item, SchedulerNode):
            yield item


def _touched_lx_names(node: SchedulerNode) -> list[str]:
    """The LX buffers one operation reads or writes."""

    return [
        dep.name
        for dep in (*node.read_writes.reads, *node.read_writes.writes)
        if isinstance(dep, MemoryDep) and _lx_layout(dep.name) is not None
    ]


def prepare_spyre_kernels(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Prepare every bundle's kernel once, while HBM fallback is still available.

    Planning has already chosen each LX buffer's physical ownership. This pass
    does not choose it again: it runs the real operation handlers and the real
    finalization over every bundle and keeps each successful kernel for
    emission. An operation whose LX placement cannot be finalized demotes the
    LX buffers it touches (a relayout member takes its whole connected group),
    and every kernel that touched a demoted buffer is prepared again, until no
    placement changes. Each retry removes at least one finite LX allocation.
    """
    scheduler = V.graph.scheduler
    bundles = [
        node
        for node in nodes
        if isinstance(node, (FusedSchedulerNode, SchedulerNode))
        and not (node.is_template() or node.is_extern() or node.is_foreach())
        and isinstance(scheduler.get_backend(node.get_device()), SuperDSCScheduling)
    ]
    leaves = list(_all_scheduler_nodes(bundles))

    def demote(name: str, reason: str) -> None:
        # Scheduling supplies the final ownership verdict; the relayout layer
        # owns the fallback for a whole connected group, and for a buffer with
        # no registered copies that group is the buffer itself. A destination
        # demoted after its group is already gone is cleared again, harmlessly.
        plan = materialized_lx_relayout_for_destination(V.graph, name)
        demote_lx_relayout_group(
            V.graph, name if plan is None else plan.source_name, reason
        )

    # One attempt prepares every bundle in order. A rejected placement demotes
    # the failing operation's LX buffers, the attempt's kernels and graph
    # removals are dropped, and the next attempt starts over, so every kernel
    # emission consumes was prepared against the final placement.
    removed_at_entry = OrderedSet(V.graph.removed_buffers)
    initial_lx_names = {name for node in leaves for name in _touched_lx_names(node)}
    for _ in range(len(initial_lx_names) + 1):
        kernels = []
        for bundle in bundles:
            backend = scheduler.get_backend(bundle.get_device())
            kernel = SpyreKernel()
            try:
                backend.prepare_kernel(bundle, kernel)
            except (Unsupported, ValueError) as exc:
                # A rejected attempt leaves nothing behind: its kernels are
                # dropped and the graph's removed-buffer set is restored.
                V.graph.removed_buffers -= V.graph.removed_buffers - removed_at_entry
                failed = kernel.failed_node
                touched = [] if failed is None else _touched_lx_names(failed)
                if failed is None or not touched:
                    raise
                reason = f"{failed.get_name()} LX ownership finalization failed: {exc}"
                # This is a preservation check, not another placement search.
                # Do not try subsets or keep a preferred writer: all LX buffers
                # touched by the failed operation fall back together.
                for touched_name in touched:
                    demote(touched_name, reason)
                break
            kernels.append((bundle, kernel))
        else:
            for bundle, kernel in kernels:
                setattr(bundle, "prepared_kernel", kernel)
            return nodes
    raise RuntimeError(
        "LX ownership finalization did not reach its monotone demotion fixed point"
    )


def verify_carried_reduction_ownership(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """Require carried-reduction stages and their committed LX ownership to survive.

    Kernel preparation already proved each access against the physical view.
    After HBM-pool planning, check that the stages, accesses, and view still exist.
    """

    grouped: dict[object, dict[str, SchedulerNode]] = {}
    for node in _all_scheduler_nodes(nodes):
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
            logger.info(
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
        if layout.lx_view is None:
            raise Unsupported(
                f"carried reduction accumulator {record.accumulator_name} has "
                "an LX address but no physical ownership"
            )
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
            # The row decision was made by carried_reduction_pinned_row before
            # placement. Kernel preparation above already proved this
            # stage can consume the committed physical view. Do not recreate a
            # symbol correspondence from post-scheduler loop position here.

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

    def _live_nodes(self, node: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        """The members of ``node`` that scheduling has not removed."""

        removed = getattr(self.scheduler, "removed_ops", ())
        return [n for n in node.get_nodes() if n.get_name() not in removed]

    def prepare_kernel(
        self, node: BaseSchedulerNode, kernel: SpyreKernel
    ) -> SpyreKernel:
        """Run the real handlers over one bundle into ``kernel``.

        Each operation is finished as it is constructed. On failure
        ``kernel.failed_node`` names the operation whose description could not
        be finalized; the kernel itself is not usable afterwards.
        """
        nodes = self._live_nodes(node)
        # Before any spec is built, which is what this method goes on to do:
        # create_tensor_arg reads this to decide whether a buffer is kernel-local.
        # _codegen_into_kernel may reach further inner nodes, so the set can be a
        # subset -- which errs towards declining.
        kernel.fused_node_names = OrderedSet(n.get_name() for n in nodes)
        with kernel:
            self._codegen_into_kernel(nodes, kernel)
        if isinstance(node, CountedLoopSchedulerNode):
            kernel.wrap_op_specs_in_loop(node.loop_count)
        kernel.check_op_specs()
        return kernel

    def codegen_node(
        self, node: Union[FusedSchedulerNode, SchedulerNode, CountedLoopSchedulerNode]
    ) -> None:
        """Emit one bundle from its prepared kernel.

        Preparation ran before HBM pooling; only what pooling decided afterwards
        is bound here: the pool size, the argument list and the HBM addresses.
        """
        assert self.scheduler
        nodes = self._live_nodes(node)
        if len(nodes) == 0:
            return
        name = node.get_name()
        leaf_names = {leaf.get_name() for leaf in _all_scheduler_nodes(nodes)}
        kernel: SpyreKernel | None = getattr(node, "prepared_kernel", None)
        if kernel is None:
            if any(_touched_lx_names(leaf) for leaf in _all_scheduler_nodes(nodes)):
                raise RuntimeError(f"{name} reached codegen without a prepared kernel")
            kernel = self.prepare_kernel(node, SpyreKernel())
        elif {leaf.get_name() for leaf in kernel.scheduled_nodes} != leaf_names:
            raise RuntimeError(f"{name} changed after its kernel was prepared")
        all_schedule_nodes = kernel.scheduled_nodes

        kernel.pool_size = getattr(V.graph, "hbm_pool_sizes", {}).get(name, 0)

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

    def _codegen_leaf(self, snode: SchedulerNode, kernel: SpyreKernel) -> None:
        """Run one operation's body through the kernel's handlers."""

        symbols = list(iteration_space(snode))
        index_vars = [
            symbols[: len(snode._body.iter_vars)],
            symbols[len(snode._body.iter_vars) :],
        ]
        kernel.scheduled_nodes.append(snode)
        try:
            snode.codegen(index_vars)
        except (Unsupported, ValueError):
            kernel.failed_node = snode
            raise

    def _codegen_loop_body(
        self, node: CountedLoopSchedulerNode, kernel: SpyreKernel
    ) -> None:
        """Codegen the body of a nested CountedLoopSchedulerNode into an existing kernel.

        The inner ops are added to the kernel's op_specs list, then wrapped
        in a LoopSpec for the inner loop count, so nesting needs no separate
        kernel.
        """
        body_start = len(kernel.op_specs)
        self._codegen_into_kernel(self._live_nodes(node), kernel)
        # Wrap only the newly-added op_specs entries in this inner LoopSpec.
        body = kernel.op_specs[body_start:]
        kernel.op_specs = kernel.op_specs[:body_start]
        kernel.op_specs.append(LoopSpec(count=node.loop_count, body=body))

    def _codegen_into_kernel(
        self, nodes: list[BaseSchedulerNode], kernel: SpyreKernel
    ) -> None:
        """Codegen a sequence of nodes into an existing kernel in order.

        Each CountedLoopSchedulerNode is driven via _codegen_loop_body so its
        ops land as a LoopSpec entry in kernel.op_specs.  Plain SchedulerNodes
        are codegenned flat.  The two types may appear in any order.
        """
        for node in nodes:
            if isinstance(node, CountedLoopSchedulerNode):
                self._codegen_loop_body(node, kernel)
            else:
                for snode in self.generate_node_schedule([node]):
                    self._codegen_leaf(snode, kernel)

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
