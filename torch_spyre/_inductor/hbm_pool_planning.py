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

import math
from typing import Callable
from sympy import Symbol
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
)
from torch._inductor.ir import FallbackKernel
from torch._inductor.virtualized import V
from .constants import MAX_POOL_SIZE_BYTES, INTERMEDIATES_SEGMENT
from .ir import FixedTiledLayout, SpyreEmptyFallback
from .logging_utils import get_inductor_logger
from .scheduler import CountedLoopSchedulerNode
from . import config

logger = get_inductor_logger("HBM_POOL_PLANNING")
_STICK_BYTES = 128
_BYTES_PER_GB = 1024**3


class Allocator:
    """
    Tracks a set of free blocks within an hbm segment. Buffers
    whose live ranges do not overlap share the same region. Each block is a
    (offset, size) pair measured in bytes.

    Ensures the pool's high-water mark (`pool_end`, a bump pointer that never
    decreases even after `free()`) never exceeds the segment size limit --
    this is the same quantity `generate_bundle` reserves via
    `sdscbundle.device_mem_allocate`, so the two must agree.
    """

    def __init__(self, segment_size: int) -> None:
        self._free: list[tuple[int, int]] = []  # (offset, size) free blocks
        self._pool_end: int = 0  # current end of the pool
        self._segment_size: int = segment_size
        self._currently_allocated: int = 0  # bytes in-use right now
        self._peak_usage: int = 0  # peak concurrent usage

    def allocate(self, size: int) -> int | None:
        """Return a byte offset from INTERMEDIATES_SEGMENT for a block of
        `size` bytes. Reuses an existing free block when possible.

        Returns None, leaving all internal state untouched, if `size` would
        push the pool's high-water mark (`pool_end`) past the segment size
        limit -- callers fall back to standalone HBM allocation for that
        buffer instead. Gating on `pool_end` rather than concurrent usage
        matters because `pool_end` -- not peak concurrent usage -- is the
        quantity `generate_bundle` reserves via
        `sdscbundle.device_mem_allocate`; free-list fragmentation can push
        `pool_end` past the limit even while concurrent usage stays low.
        """
        for i, (blk_offset, blk_size) in enumerate(self._free):
            if blk_size >= size:
                offset = blk_offset
                new_pool_end = self._pool_end
                break
        else:
            # No suitable free block — extend the pool.
            offset = self._pool_end
            new_pool_end = self._pool_end + size
            i = None

        if new_pool_end > self._segment_size:
            return None

        if i is not None:
            self._free.pop(i)
            remainder = blk_size - size
            if remainder > 0:
                self._free.append((blk_offset + size, remainder))
        self._pool_end = new_pool_end
        self._currently_allocated += size
        self._peak_usage = max(self._currently_allocated, self._peak_usage)

        return offset

    def free(self, offset: int, size: int) -> None:
        """Return a previously allocated block to the free list."""
        self._free.append((offset, size))
        self._currently_allocated -= size

    def get_peak_usage(self) -> int:
        """Return the peak concurrent memory usage in bytes."""
        return self._peak_usage

    def get_pool_end(self) -> int:
        return self._pool_end


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _compute_size_bytes(name: str) -> int:
    """Return the stick-aligned device size in bytes for buffer `name`."""
    buf = V.graph.get_buffer(name)
    layout = buf.maybe_get_layout()
    assert isinstance(layout, FixedTiledLayout), (
        f"hbm_pool_planning: expected FixedTiledLayout for {name}, got {type(layout)}"
    )
    dev_layout = layout.device_layout
    num_sticks = math.prod(dev_layout.device_size[:-1])
    size_bytes = num_sticks * _STICK_BYTES
    return _align_up(size_bytes, _STICK_BYTES)


def _compute_live_ranges(
    nodes: list[BaseSchedulerNode],
    pool_candidates: set[str],
    alloc_id_of: "Callable[[str], int | None]",
) -> dict[str, tuple[int, int]]:
    """Return {buf_name: (start_step, end_step)} for each pool candidate.

    start_step: timestep of the node that writes the buffer.
    end_step: last timestep at which any node reads the buffer.

    Two or more candidate names can share the same underlying
    FixedTiledLayout.allocation dict object -- see `_alloc_id` -- when a
    MutationLayoutSHOULDREMOVE retarget (e.g. the copy-in/retarget/copy-back
    sequence enforce_indirect_access_layout.py's
    _insert_mutation_relayout_copy inserts for a non-compliant scatter
    destination) makes one op's layout literally *be* another buffer's
    layout instance. Since they are truly one physical storage location,
    their live range must be the union of every aliased name's individual
    range -- not each name's own range in isolation -- or the allocator
    could place another buffer's block on top of storage one of the
    aliased names still needs. Every name in such a group gets the exact
    same merged (start, end) here so the caller can allocate it once (via
    `alloc_id_of`) using a range that is safe for every alias.
    """
    start: dict[str, int] = {}
    end: dict[str, int] = {}

    for idx, node in enumerate(nodes):
        rw = node.read_writes
        for dep in rw.writes:
            if dep.name in pool_candidates:
                start[dep.name] = idx
        for dep in rw.reads:
            if dep.name in pool_candidates:
                end[dep.name] = idx

    live_ranges: dict[str, tuple[int, int]] = {}
    for name in pool_candidates:
        if name in start:
            live_ranges[name] = (start[name], end.get(name, len(nodes) + 1))

    # Merge ranges within each alias group (same id(layout.allocation)).
    group_range: dict[int, tuple[int, int]] = {}
    for name, (s, e) in live_ranges.items():
        alloc_id = alloc_id_of(name)
        if alloc_id is None:
            continue
        gs, ge = group_range.get(alloc_id, (s, e))
        group_range[alloc_id] = (min(gs, s), max(ge, e))
    for name in live_ranges:
        alloc_id = alloc_id_of(name)
        if alloc_id is not None and alloc_id in group_range:
            live_ranges[name] = group_range[alloc_id]

    return live_ranges


def hbm_pool_planning(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Pool-allocate buffers so non-overlapping tensors share the HBM intermediates segment.

    This is a *distinct* pass from LX scratchpad planning
    (`scratchpad/allocator.py`, `scratchpad_planning()`). Both passes decide
    where an intermediate tensor's data lives, but:
    - This pass runs in `CustomPostFusionPasses`, **after** Inductor's
      scheduler fusion has already run.
    - LX planning runs in `CustomPreSchedulingPasses`, **before** the
      Scheduler is even constructed.
    - This pass bump/free-list-allocates a region of regular HBM (the
      "intermediates segment", see `constants.INTERMEDIATES_SEGMENT`); LX
      planning allocates a fixed on-chip SRAM scratchpad per core.
    - A buffer is only an hbm_pool candidate if LX planning did *not* already
      claim it (`"lx" not in layout.allocation`); the two passes are
      mutually exclusive per buffer, applied in that order.

    Collects pool candidates from two sources:
    - Kernel intermediates: buffers both written and read within the graph,
      detected via written & read sets on ComputedBuffer nodes.
    - SpyreEmptyFallback full buffers created by coarse_tile.py (non-outputs).
      These are ExternKernel nodes invisible to the written & read path above.

    For each candidate, assigns layout.allocation["hbm_pool"] = INTERMEDIATES_SEGMENT + offset.
    Graph inputs/outputs and LX-allocated buffers are excluded.

    Bundle scoping: this pass runs after ``spyre_fuse_nodes`` (see
    ``CustomPostFusionPasses`` in passes.py), so ``nodes`` is the final,
    post-fusion top-level list -- each entry is exactly one SDSC bundle
    (one SpyreKernel / .run() call).  A buffer is pool-eligible only if
    the bundle that writes it is the same bundle that contains every read
    of it; a buffer written in one bundle and read from a different one
    falls back to standalone HBM, since bundle-scoped pools do not
    coexist across separate kernel invocations.  Pool offsets, sizes, and
    live-range analysis are all computed independently per bundle.

    See docs/source/compiler/hbm_pool_planning.md for the full design and a
    side-by-side comparison table with LX scratchpad planning.
    """

    if not config.hbm_pool_planning:
        V.graph.hbm_pool_sizes = {}
        return nodes

    graph_inputs: set[str] = set(V.graph.graph_inputs.keys())
    graph_outputs: set[str] = set(V.graph.get_output_names())
    io_names: set[str] = graph_inputs | graph_outputs

    _kernel_arg_types = (
        FallbackKernel,
        ExternKernelSchedulerNode,
        NopKernelSchedulerNode,
    )

    # Now that this pass runs after spyre_fuse_nodes (post-fusion), each
    # top-level entry in `nodes` is a bundle that may itself be a
    # FusedSchedulerNode wrapping several SchedulerNodes/CountedLoopScheduler
    # Nodes (or a CountedLoopSchedulerNode -- a FusedSchedulerNode subclass --
    # wrapping a counted-loop body).  Flatten both so individual ops are
    # visible at distinct timesteps for written/read detection and live-range
    # analysis.  FusedSchedulerNode.get_nodes() returns self.snodes verbatim
    # (not recursively flattened), so the recursion below handles nesting of
    # arbitrary depth.
    def _iter_all_nodes(
        node_list: list[BaseSchedulerNode],
    ):
        for n in node_list:
            if isinstance(n, FusedSchedulerNode):
                yield from _iter_all_nodes(n.get_nodes())
            else:
                yield n

    # Mutation buffers share the same allocation dict object as their target, so a
    # name-based check is insufficient.
    io_alloc_ids: set[int] = {
        id(layout.allocation)
        for io_name in io_names
        if (io_buf := V.graph.get_buffer(io_name)) is not None
        and not isinstance(io_buf, Symbol)
        # Use maybe_get_layout(): None-valued args carried by FallbackKernels
        # have no layout and raise from get_layout().
        and isinstance(layout := io_buf.maybe_get_layout(), FixedTiledLayout)
    }

    def _is_intermediate(name: str) -> bool:
        buf = V.graph.get_buffer(name)
        if buf is None:
            return False
        layout = buf.maybe_get_layout()
        return (
            isinstance(layout, FixedTiledLayout)
            and "lx" not in layout.allocation
            and id(layout.allocation) not in io_alloc_ids
        )

    def _alloc_id(name: str) -> int | None:
        """Return id(layout.allocation) for `name`, or None if unavailable.

        Buffers that alias another buffer's storage (e.g. a
        MutationLayoutSHOULDREMOVE target written by a later, differently-
        named op) share the same allocation dict object even though the two
        names are unrelated by any read/write dependency edge -- this is the
        same aliasing io_alloc_ids above already has to work around for
        graph I/O.  Resolving it here lets bundle attribution below catch
        the intermediate-to-intermediate case too.
        """
        buf = V.graph.get_buffer(name)
        if buf is None:
            return None
        layout = buf.maybe_get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return None
        return id(layout.allocation)

    # Buffers read by Fallback/Extern/Nop nodes must stay Python-side tensors,
    # regardless of which bundle they belong to.
    all_flat_nodes: list[BaseSchedulerNode] = list(_iter_all_nodes(nodes))
    fallback_read = {
        dep.name
        for node in all_flat_nodes
        if isinstance(node, _kernel_arg_types)
        for dep in node.read_writes.reads
    }

    # Build per-buffer writer-bundle / reader-bundles maps by walking each
    # top-level bundle's own flattened node list once.  A buffer normally has
    # exactly one writer (bundle), tracked in buffer_writer_bundle; it may
    # have readers in zero, one, or several bundles.  buffer_writer_bundles
    # additionally accumulates *every* bundle that ever wrote a given name,
    # so a buffer written by more than one bundle (e.g. a loop-carried
    # accumulator whose in-place update is mutation-renamed by Inductor's
    # scheduler to the same name as its initializer in an earlier bundle) can
    # be detected even though buffer_writer_bundle itself only ever retains
    # the last writer.
    buffer_writer_bundle: dict[str, str] = {}
    buffer_writer_bundles: dict[str, set[str]] = {}
    buffer_reader_bundles: dict[str, set[str]] = {}
    # id(layout.allocation) -> every bundle that wrote a name resolving to
    # that allocation dict.  Populated alongside buffer_writer_bundles so
    # aliased names (see _alloc_id) are attributed to the same underlying
    # storage even though they share no name-based dependency edge.
    #
    # Deliberately write-keyed only: the sole Inductor mechanism that makes
    # two distinct buffer names share the literal id(layout.allocation)
    # object is MutationLayoutSHOULDREMOVE, which always attaches to a
    # write (set via `src.data.layout = MutationLayoutSHOULDREMOVE(dst)` in
    # Inductor's realize_into/mark_buffer_mutated) -- so walking .writes
    # alone is provably complete. Inductor's read-only aliasing mechanism
    # (NonOwningLayout, used for views) builds a new Layout object rather
    # than sharing one, and such buffers are excluded from pool eligibility
    # entirely by the isinstance(layout, FixedTiledLayout) checks in
    # _is_intermediate/_alloc_id, independent of bundle analysis.
    alloc_id_bundles: dict[int, set[str]] = {}

    for bundle in nodes:
        bundle_name = bundle.get_name()
        bundle_flat = list(_iter_all_nodes([bundle]))
        bundle_non_kernel = [
            n for n in bundle_flat if not isinstance(n, _kernel_arg_types)
        ]
        for node in bundle_non_kernel:
            for dep in node.read_writes.writes:
                if dep.name not in graph_outputs:
                    buffer_writer_bundle[dep.name] = bundle_name
                    buffer_writer_bundles.setdefault(dep.name, set()).add(bundle_name)
                    if (alloc_id := _alloc_id(dep.name)) is not None:
                        alloc_id_bundles.setdefault(alloc_id, set()).add(bundle_name)
            for dep in node.read_writes.reads:
                if dep.name not in graph_inputs:
                    buffer_reader_bundles.setdefault(dep.name, set()).add(bundle_name)
        # SpyreEmptyFallback nodes allocate a buffer but emit no dep-tracked
        # write, so they never appear via the dep walk above.  Collect them
        # explicitly, using the underlying buffer's name (node.node.get_name())
        # rather than the scheduler node's own operation name
        # (node.get_name()) -- for an ExternKernelSchedulerNode these differ
        # (e.g. "op30" vs "buf27").
        for node in bundle_flat:
            if (
                isinstance(node, ExternKernelSchedulerNode)
                and isinstance(node.node, SpyreEmptyFallback)
                and node.node.get_name() not in graph_outputs
            ):
                buffer_writer_bundle[node.node.get_name()] = bundle_name
                buffer_writer_bundles.setdefault(node.node.get_name(), set()).add(
                    bundle_name
                )
                if (alloc_id := _alloc_id(node.node.get_name())) is not None:
                    alloc_id_bundles.setdefault(alloc_id, set()).add(bundle_name)

    written = set(buffer_writer_bundle)
    read = set(buffer_reader_bundles)

    def _is_cross_bundle(name: str) -> bool:
        if len(buffer_writer_bundles.get(name, set())) > 1:
            logger.debug(
                "hbm_pool_planning: %s written by multiple bundles %s -- "
                "excluding from pool eligibility",
                name,
                sorted(buffer_writer_bundles[name]),
            )
            return True
        alloc_id = _alloc_id(name)
        if alloc_id is not None and len(alloc_id_bundles.get(alloc_id, set())) > 1:
            logger.debug(
                "hbm_pool_planning: %s shares its allocation with a buffer "
                "written in another bundle %s -- excluding from pool "
                "eligibility",
                name,
                sorted(alloc_id_bundles[alloc_id]),
            )
            return True
        readers = buffer_reader_bundles.get(name, set())
        writer = buffer_writer_bundle.get(name)
        return bool(readers - {writer})

    all_candidates = {
        name
        for name in (written & read) - io_names - fallback_read
        if _is_intermediate(name) and not _is_cross_bundle(name)
    }

    V.graph.hbm_pool_sizes = {}

    # Sort by start step so the allocator processes tensors in execution
    # order.  Tie-break on (end_step, name) for determinism:
    def _alloc_sort_key(item: tuple[str, tuple[int, int]]) -> tuple[int, int, str]:
        name, (start, end) = item
        return (start, end, name)

    for bundle in nodes:
        bundle_name = bundle.get_name()
        # Restrict to buffers this bundle actually writes -- buffer_writer_
        # bundle[name] == bundle_name is implied by membership in
        # all_candidates plus this bundle's own written set, but recomputing
        # a local written set keeps this loop self-contained.
        bundle_candidates = {
            name
            for name in all_candidates
            if buffer_writer_bundle.get(name) == bundle_name
        }
        if not bundle_candidates:
            continue

        # Use the bundle's own direct, unflattened children here (not the
        # recursively-flattened list used above for candidate
        # identification). A CountedLoopSchedulerNode's merged read/write
        # set (see ReadWrites.merge_list) drops any buffer the loop both
        # writes and reads internally -- a loop-carried accumulator has no
        # visible read at all from this node's perspective. _compute_live_
        # ranges then falls back to `len(nodes) + 1` for such a buffer's
        # end_step, conservatively keeping it live through the rest of the
        # bundle, rather than the buggy alternative: fully flattening into
        # the loop body would expose its one internal read as an ordinary
        # timestep, ending its live range there and letting the allocator
        # free and reuse its offset for a different, still-live buffer
        # (e.g. the loop's own per-iteration scratch tile).
        #
        # This opacity relies on `bundle` being a FusedSchedulerNode whose
        # `get_nodes()` returns *other* nodes (the loop appearing as one
        # timestep among siblings). When spyre_fuse_nodes has no fusible
        # neighbors to group a loop with, `_make_fused` returns the bare
        # CountedLoopSchedulerNode itself as the top-level bundle -- in that
        # case `bundle.get_nodes()` returns the loop's own internal snodes,
        # bypassing the opacity entirely. Guard against that by treating a
        # bare loop bundle as a single opaque timestep.
        if isinstance(bundle, CountedLoopSchedulerNode):
            live_range_nodes = [bundle]
        else:
            live_range_nodes = bundle.get_nodes()
        live_ranges = _compute_live_ranges(
            live_range_nodes, bundle_candidates, _alloc_id
        )
        sorted_bufs = sorted(live_ranges.items(), key=_alloc_sort_key)

        # Two or more candidate names can resolve to the same
        # id(layout.allocation) (see _alloc_id / _compute_live_ranges) --
        # they are one physical storage location sharing one merged live
        # range, and must be allocated exactly once. Group by alloc_id
        # here (preserving sorted_bufs's (start, end, name) order for the
        # group's position) so the loop below allocates once per group and
        # then writes the resulting offset into every member name.
        alloc_groups: dict[str, list[str]] = {}
        group_of_alloc_id: dict[int, str] = {}
        alloc_order: list[str] = []
        for name, _ in sorted_bufs:
            alloc_id = _alloc_id(name)
            rep = group_of_alloc_id.get(alloc_id) if alloc_id is not None else None
            if rep is not None:
                alloc_groups[rep].append(name)
                continue
            if alloc_id is not None:
                group_of_alloc_id[alloc_id] = name
            alloc_groups[name] = [name]
            alloc_order.append(name)

        allocator = Allocator(MAX_POOL_SIZE_BYTES)

        # Track (end_step, offset, size) so we can free blocks promptly.
        pending_frees: list[tuple[int, int, int]] = []
        overflowed = 0

        for rep_name in alloc_order:
            start, end = live_ranges[rep_name]
            grouped_names = alloc_groups[rep_name]
            # Free any blocks whose live range ended before this start step.
            still_live = []
            for entry in pending_frees:
                e, off, sz = entry
                if e < start:
                    allocator.free(off, sz)
                else:
                    still_live.append(entry)
            pending_frees = still_live

            size = _compute_size_bytes(rep_name)
            offset = allocator.allocate(size)

            if offset is None:
                # Pool is full: leave this buffer on the standalone-HBM path
                # (no "hbm_pool" key) rather than failing the whole bundle --
                # it gets the same allocation graph inputs/outputs and
                # cross-bundle buffers already use.
                overflowed += 1
                logger.debug(
                    "hbm_pool_planning: bundle=%s  %s  live=[%d,%d]  size=%d  "
                    "does not fit in pool -- falling back to standalone HBM",
                    bundle_name,
                    grouped_names,
                    start,
                    end,
                    size,
                )
                continue

            # Assign pool offset directly to layout.allocation. Every name
            # in grouped_names shares the exact same layout/allocation dict
            # object, so this single write is visible to all of them.
            layout = V.graph.get_buffer(rep_name).maybe_get_layout()
            assert isinstance(layout, FixedTiledLayout)
            if config.bundle_symbolic_args:
                layout.allocation["hbm_pool"] = offset
            else:
                layout.allocation["hbm_pool"] = INTERMEDIATES_SEGMENT + offset

            pending_frees.append((end, offset, size))
            logger.debug(
                "hbm_pool_planning: bundle=%s  %s  live=[%d,%d]  size=%d  offset=%d",
                bundle_name,
                grouped_names,
                start,
                end,
                size,
                offset,
            )

            # SpyreEmptyFallback.should_allocate()/removed_buffers is a
            # per-*name* codegen concern (each name is still emitted as its
            # own statement even though several share one storage
            # location), so this must run once for every name in the group.
            for name in grouped_names:
                buf = V.graph.get_buffer(name)
                if isinstance(buf, SpyreEmptyFallback):
                    # SpyreEmptyFallback.should_allocate() returns False once
                    # pool-allocated, so the wrapper never emits an
                    # AllocateLine for it. Base Inductor's free machinery
                    # (Scheduler.free_buffers -> codegen_free -> can_reuse)
                    # does not consult should_allocate(); it frees any buffer
                    # whose last use has passed regardless of whether it was
                    # ever allocated. Without this, the generated wrapper
                    # emits `del buf27` with no prior `buf27 = ...`, raising
                    # UnboundLocalError. free_buffers() itself subtracts
                    # V.graph.removed_buffers before iterating, so adding the
                    # name here keeps it out of codegen entirely.
                    V.graph.removed_buffers.add(name)

        peak = allocator.get_peak_usage()
        pool_extent = allocator.get_pool_end()
        if overflowed:
            logger.warning(
                "hbm_pool_planning: bundle=%s  %d intermediate(s) did not fit in "
                "the %.2f GB pool budget and fell back to standalone HBM",
                bundle_name,
                overflowed,
                MAX_POOL_SIZE_BYTES / _BYTES_PER_GB,
            )
        logger.info(
            "hbm_pool_planning: bundle=%s assigned %d intermediates, peak concurrent "
            "usage %.2f GB, pool extent %.2f GB / %.2f GB",
            bundle_name,
            len(alloc_order) - overflowed,
            peak / _BYTES_PER_GB,
            pool_extent / _BYTES_PER_GB,
            MAX_POOL_SIZE_BYTES / _BYTES_PER_GB,
        )
        V.graph.hbm_pool_sizes[bundle_name] = pool_extent

    if not V.graph.hbm_pool_sizes:
        logger.info("hbm_pool_planning: no bundle had any pool-eligible intermediate")
    else:
        logger.info(
            "hbm_pool_planning: %d bundle(s) with pool allocations, "
            "total pool bytes across bundles %.2f GB",
            len(V.graph.hbm_pool_sizes),
            sum(V.graph.hbm_pool_sizes.values()) / _BYTES_PER_GB,
        )

    return nodes
