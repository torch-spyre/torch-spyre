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
    local_lx_emission_contract,
    materialized_lx_relayout_for_destination,
)
from . import config as _spyre_config
from .errors import Unsupported

logger = get_inductor_logger("scheduler")


class CountedLoopSchedulerNode(FusedSchedulerNode):
    """A group of SchedulerNodes to be executed inside a counted outer loop.

    Produced as a ``LoopSchedulerNode`` (``loop_scheduler.py``), one per
    ``LoopNode`` the pre-scheduling passes leave in ``graph.operations``.

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


class SuperDSCScheduling(BaseScheduling):
    """Inductor scheduling backend for Spyre.

    Inductor's own node-pairwise fusion (`can_fuse_vertical`/
    `can_fuse_horizontal` below) is permanently disabled, not merely
    unimplemented. Spyre does its own op-to-kernel grouping in a later,
    separate stage: `spyre_fuse_nodes` in `fusion.py` bundles
    `BaseSchedulerNode`s into `SuperDSCBundle`s, and each bundle maps
    one-to-one to a `SpyreKernel` device launch. See
    `docs/source/compiler/inductor_frontend.md` for the full pipeline.
    """

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
        Always False: Inductor-level fusion is disabled by design. Spyre's
        real op-to-kernel grouping happens later, in `spyre_fuse_nodes`
        (see the class docstring above).
        """
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Always False: Inductor-level fusion is disabled by design. Spyre's
        real op-to-kernel grouping happens later, in `spyre_fuse_nodes`
        (see the class docstring above).
        """
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
        kernel.local_read_contract_factory = lambda name, consumer: (
            local_lx_emission_contract(V.graph, name, consumer)
        )
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
        kernel.op_specs.append(SpyreKernel.counted_loop(node.loop_count, body))

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
            if method == "sdsc":
                from torch_spyre.execution.kernel_cache import precompute_kernel_key

                # pool_size is passed only for a pooled kernel (sdsc's default
                # 0 otherwise); the precomputed key hashes the value passed.
                tail = ""
                pool_size = 0
                if kernel._kernel_uses_hbm_pool():
                    pool_size = kernel.pool_size
                    tail = f", pool_size={pool_size}"
                # Emitted whatever config.precomputed_kernel_key says: the
                # switch is read by sdsc() at call time, not here.
                precomputed_key = precompute_kernel_key(
                    src_code, kernel_name, pool_size
                )
                buf.writeline(f"{tail}, precomputed_key={precomputed_key!r})")
            else:
                buf.writeline(")")
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
