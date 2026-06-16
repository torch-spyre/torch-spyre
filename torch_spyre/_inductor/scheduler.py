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

from typing import Sequence, Union

import sympy

from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import (
    get_kernel_metadata,
    get_fused_kernel_name,
    sympy_product,
)
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
from .pass_utils import iteration_space
from .logging_utils import get_inductor_logger
from .op_spec import LoopSpec

logger = get_inductor_logger("scheduler")


def _find_leaf_sched_node(node: BaseSchedulerNode):
    """Recursively find the first leaf SchedulerNode inside a (possibly nested) node."""
    for snode in node.get_nodes():
        if isinstance(snode, SchedulerNode):
            return snode
        result = _find_leaf_sched_node(snode)
        if result is not None:
            return result
    return None


def _tiled_syms_for_sched_node_at_depth(sched_node: SchedulerNode, depth: int) -> list:
    """Return the OpSpec iteration-space symbols tiled at ``depth``.

    Uses ``loop_tiled_dims[depth]`` and ``loop_tiled_reduction_dims[depth]``
    from the IR node and the SchedulerNode's ``iteration_space`` (which
    produces the same symbols as ``create_op_spec`` uses to build
    ``OpSpec.tiled_symbols``).

    ``loop_tiled_dims`` stores *host-range* dimension indices (indices into
    ``op.data.ranges``), which include unit-size batch dimensions that are
    skipped in the iteration space.  We must map host-range indices to
    iteration-space key indices by walking ``op.data.ranges`` and counting
    only the non-unit entries.

    For reduction-dimension tiling (``loop_tiled_reduction_dims``), the
    reduction symbols follow the output symbols in the iteration space key
    list (the scheduler produces keys from reads.ranges for Reduction nodes,
    which has output dims first then reduction dims).  The offset is the
    number of non-unit output-dim ranges; indices in
    ``loop_tiled_reduction_dims`` are 0-based into the reduction portion.
    """
    ir_op = sched_node.node
    if ir_op is None:
        return []
    loop_info = getattr(ir_op, "loop_info", None)
    if loop_info is None:
        return []
    raw = loop_info.loop_tiled_dims
    raw_rdims = getattr(loop_info, "loop_tiled_reduction_dims", [])
    if not raw and not raw_rdims:
        return []
    dims_per_level: list[list[int]] = raw if raw else [[] for _ in raw_rdims]
    rdims_per_level: list[list[int]] = raw_rdims if raw_rdims else [[] for _ in raw]
    if depth >= len(dims_per_level):
        return []
    it_space = iteration_space(sched_node)
    keys = list(it_space.keys())

    # Build a map from host-range index → iteration-space key index.
    # loop_tiled_dims is only stamped on ComputedBuffer ops (Pointwise/Reduction),
    # so data.ranges is always present here.  The iteration space simply omits
    # unit-size dims, so we walk ranges and count only non-unit entries.
    host_to_it: dict[int, int] = {}
    it_idx = 0
    for host_idx, r in enumerate(ir_op.data.ranges):
        if int(r) != 1:
            host_to_it[host_idx] = it_idx
            it_idx += 1

    result = []
    for d in dims_per_level[depth]:
        mapped = host_to_it.get(d)
        if mapped is not None and mapped < len(keys):
            result.append(keys[mapped])

    # Map reduction-dimension indices to iteration-space symbols.  For
    # Reduction nodes the iteration space (from reads.ranges) has output-dim
    # symbols first, then reduction-dim symbols.  The offset is the count of
    # non-unit output-dim ranges.
    rdims_at_depth = rdims_per_level[depth] if depth < len(rdims_per_level) else []
    if rdims_at_depth:
        n_output_syms = sum(1 for r in ir_op.data.ranges if int(r) != 1)
        for rd in rdims_at_depth:
            sym_idx = n_output_syms + rd
            if sym_idx < len(keys):
                result.append(keys[sym_idx])

    return result


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
    nesting level from outermost to innermost (stamped by coarse_tile()).
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


def _build_loop_group(
    nodes: list[BaseSchedulerNode], depth: int
) -> list[BaseSchedulerNode]:
    """Recursively wrap contiguous runs sharing a loop_group_id into CountedLoopSchedulerNodes.

    depth is the nesting level being processed (0 = outermost).  Each node's
    loop_group_id is a tuple; we group on element [depth].
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
    Nodes sharing the same outermost key must be contiguous; a gap indicates
    a data-flow dependency crossing the group boundary, which is a bug in
    the tiling pass.

    Running before Inductor's fusion pass ensures CountedLoopSchedulerNodes are
    visible to SuperDSCScheduling.can_fuse_vertical/horizontal (which return False),
    so loop groups survive Inductor fusion intact.  spyre_fuse_nodes is separately
    protected because it only fuses plain SchedulerNodes (isinstance check), causing
    CountedLoopSchedulerNodes to force a bundle boundary.
    """
    result = _build_loop_group(nodes, depth=0)

    # Verify contiguity: no loop_group_id should appear in two separate runs.
    seen: dict[tuple, str] = {}
    for node in result:
        if isinstance(node, CountedLoopSchedulerNode):
            gid = _loop_group_id(node.get_nodes()[0])
            if gid is not None:
                key = gid[0:1]
                name = node.get_name()
                if key in seen and seen[key] != name:
                    raise RuntimeError(
                        f"Loop group {key} is not contiguous in the scheduler node list. "
                        "This indicates a data-flow dependency crossing a loop group boundary."
                    )
                seen[key] = name

    return result


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
            node
            for node in node.get_nodes()
            if node.get_name() not in self.scheduler.removed_ops
        ]
        if len(nodes) == 0:
            return

        node_schedule = self.generate_node_schedule(nodes)
        kernel = SpyreKernel()
        with kernel:
            for node in node_schedule:
                var_ranges = iteration_space(node)
                vars = list(var_ranges.keys())
                index_vars = [
                    vars[: len(node._body.iter_vars)],
                    vars[len(node._body.iter_vars) :],
                ]
                node.codegen(index_vars)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, node_schedule, kernel)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        with V.set_kernel_handler(kernel):
            for node in node_schedule:
                node.mark_run()

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel.kernel_name)

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

        # Each snode in the group may itself be a CountedLoopSchedulerNode
        # (nested loop) or a plain SchedulerNode.  Drive them all into the
        # same SpyreKernel so their OpSpecs land in one op_specs list.
        kernel = SpyreKernel()
        all_schedule_nodes: list[SchedulerNode] = []
        with kernel:
            for inner in inner_nodes:
                if isinstance(inner, CountedLoopSchedulerNode):
                    # Recurse: codegen the inner loop into the same kernel,
                    # which will call wrap_op_specs_in_loop on the inner body.
                    # We temporarily redirect codegen to this kernel.
                    self._codegen_loop_body(inner, kernel, all_schedule_nodes)
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

        # Compute per-level tiled symbols for the outer (depth=0) LoopSpec.
        # Find a leaf SchedulerNode to read loop_tiled_dims + iteration_space.
        outer_tiled_syms: list = []
        for inner in inner_nodes:
            ref = _find_leaf_sched_node(inner)
            if ref is not None:
                outer_tiled_syms = _tiled_syms_for_sched_node_at_depth(ref, 0)
                break

        kernel.wrap_op_specs_in_loop(
            node.loop_count,
            tiled_symbols=outer_tiled_syms,
        )

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

        # Determine this level's tiled symbols using the IR's loop_tiled_dims[depth].
        ref_sched_node = _find_leaf_sched_node(node)
        level_syms = (
            _tiled_syms_for_sched_node_at_depth(ref_sched_node, depth)
            if ref_sched_node is not None
            else []
        )

        # Wrap only the newly-added op_specs entries in this inner LoopSpec.
        body = kernel.op_specs[body_start:]
        kernel.op_specs = kernel.op_specs[:body_start]
        kernel.op_specs.append(
            LoopSpec(
                count=node.loop_count,
                body=body,
                tiled_symbols=level_syms,
            )
        )

    def define_kernel(self, src_code, node_schedule, kernel):
        """
        Codegen kernel definition to go in output wrapper code
        """
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = get_fused_kernel_name(node_schedule, "original_aten")
            kernel_name = "_".join(["sdsc", fused_name, wrapper.next_kernel_suffix()])
            wrapper.src_to_kernel[src_code] = kernel_name
            buf = IndentedBuffer()
            buf.writeline(f"async_compile.sdsc('{kernel_name}',")
            with buf.indent():
                buf.splice(f"{src_code}")
            buf.writeline(")")
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
