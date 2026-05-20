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
from sympy import Symbol
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    NopKernelSchedulerNode,
)
from torch._inductor.ir import FallbackKernel
from torch._inductor.virtualized import V
from .constants import SEGMENT_SIZE, INTERMEDIATES_SEGMENT
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger, _get_env_bool

logger = get_inductor_logger("MEMORY_PLANNING")
_STICK_BYTES = 128
_MEMORY_PLAN_ENABLED = _get_env_bool("SPYRE_INDUCTOR_MEMORY_PLAN", True)


class Allocator:
    """
    Tracks a set of free blocks within an hbm segment. Buffers
    whose live ranges do not overlap share the same region. Each block is a
    (offset, size) pair measured in bytes.

    Ensures peak concurrent memory usage does not exceed the segment size limit.
    """

    def __init__(self, segment_size: int) -> None:
        self._free: list[tuple[int, int]] = []  # (offset, size) free blocks
        self._pool_end: int = 0  # current end of the pool
        self._segment_size: int = segment_size
        self._currently_allocated: int = 0  # bytes in-use right now
        self._peak_usage: int = 0  # peak concurrent usage

    def allocate(self, size: int) -> int:
        """Return a byte offset from INTERMEDIATES_SEGMENT for a block of
        `size` bytes. Reuses an existing free block when possible."""
        for i, (blk_offset, blk_size) in enumerate(self._free):
            if blk_size >= size:
                self._free.pop(i)
                # Return any leftover fragment to the free list.
                remainder = blk_size - size
                if remainder > 0:
                    self._free.append((blk_offset + size, remainder))
                offset = blk_offset
                break
        else:
            # No suitable free block — extend the pool.
            offset = self._pool_end
            self._pool_end += size

        self._currently_allocated += size
        if self._currently_allocated > self._peak_usage:
            self._peak_usage = self._currently_allocated

        if self._peak_usage > self._segment_size:
            raise RuntimeError(
                f"HBM intermediate pool peak usage ({self._peak_usage} bytes, "
                f"{self._peak_usage / (1024**3):.2f} GB) exceeds segment size "
                f"({self._segment_size} bytes, {self._segment_size / (1024**3):.2f} GB)"
            )

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
    layout = buf.get_layout()
    assert isinstance(layout, FixedTiledLayout), (
        f"memory_planning: expected FixedTiledLayout for {name}, got {type(layout)}"
    )
    dev_layout = layout.device_layout
    num_sticks = math.prod(dev_layout.device_size[:-1])
    size_bytes = num_sticks * _STICK_BYTES
    return _align_up(size_bytes, _STICK_BYTES)


def _compute_live_ranges(
    nodes: list[BaseSchedulerNode],
    intermediates: set[str],
) -> dict[str, tuple[int, int]]:
    """Return {buf_name: (start_step, end_step)} for each intermediate.

    start_step: timestep of the node that writes the buffer.
    end_step: last timestep at which any node reads the buffer.
    """
    start: dict[str, int] = {}
    end: dict[str, int] = {}

    for idx, node in enumerate(nodes):
        rw = node.read_writes
        for dep in rw.writes:
            if dep.name in intermediates:
                start[dep.name] = idx
        for dep in rw.reads:
            if dep.name in intermediates:
                end[dep.name] = idx

    live_ranges: dict[str, tuple[int, int]] = {}
    for name in intermediates:
        if name in start:
            live_ranges[name] = (start[name], end.get(name, len(nodes) + 1))
    return live_ranges


def memory_planning(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Assign intermediate tensors addresses in the same segment.
    Identifies intermediate buffers (not graph inputs/outputs, not already LX-allocated), performs
    live range analysis, and assigns layout.allocation["pool"] = address so
    that non-overlapping intermediates share a hbm segment.
    """

    if not _MEMORY_PLAN_ENABLED:
        V.graph.pool_size = 0
        return nodes

    graph_inputs: set[str] = set(V.graph.graph_inputs.keys())
    graph_outputs: set[str] = set(V.graph.get_output_names())
    io_names: set[str] = graph_inputs | graph_outputs

    _kernel_arg_types = (
        FallbackKernel,
        ExternKernelSchedulerNode,
        NopKernelSchedulerNode,
    )
    non_kernel_nodes = [n for n in nodes if not isinstance(n, _kernel_arg_types)]

    written = {
        dep.name
        for node in non_kernel_nodes
        for dep in node.read_writes.writes
        if dep.name not in graph_outputs
    }
    read = {
        dep.name
        for node in non_kernel_nodes
        for dep in node.read_writes.reads
        if dep.name not in graph_inputs
    }

    # Mutation buffers share the same allocation dict object as their target, so a
    # name-based check is insufficient.
    io_alloc_ids: set[int] = {
        id(layout.allocation)
        for io_name in io_names
        if (io_buf := V.graph.get_buffer(io_name)) is not None
        and not isinstance(io_buf, Symbol)
        and isinstance(layout := io_buf.get_layout(), FixedTiledLayout)
    }

    def _is_intermediate(name: str) -> bool:
        buf = V.graph.get_buffer(name)
        if buf is None:
            return False
        layout = buf.get_layout()
        return (
            isinstance(layout, FixedTiledLayout)
            and "lx" not in layout.allocation
            and id(layout.allocation) not in io_alloc_ids
        )

    intermediates = {
        name for name in (written & read) - io_names if _is_intermediate(name)
    }
    if not intermediates:
        V.graph.pool_size = 0
        return nodes

    live_ranges = _compute_live_ranges(nodes, intermediates)

    # Sort by start step so the allocator processes tensors in execution order.
    sorted_bufs = sorted(live_ranges.items(), key=lambda kv: kv[1][0])

    allocator = Allocator(SEGMENT_SIZE)

    # Track (end_step, offset, size) so we can free blocks promptly.
    pending_frees: list[tuple[int, int, int]] = []

    for name, (start, end) in sorted_bufs:
        # Free any blocks whose live range ended before this start step.
        still_live = []
        for entry in pending_frees:
            e, off, sz = entry
            if e < start:
                allocator.free(off, sz)
            else:
                still_live.append(entry)
        pending_frees = still_live

        size = _compute_size_bytes(name)
        offset = allocator.allocate(size)

        # Assign HBM address directly to layout.allocation.
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        assert isinstance(layout, FixedTiledLayout)
        layout.allocation["pool"] = INTERMEDIATES_SEGMENT + offset

        pending_frees.append((end, offset, size))

        logger.debug(
            "memory_planning: %s  live=[%d,%d]  size=%d  offset=%d",
            name,
            start,
            end,
            size,
            offset,
        )

    peak = allocator.get_peak_usage()
    pool_extent = allocator.get_pool_end()
    logger.info(
        "memory_planning: assigned %d intermediates, peak concurrent usage %.2f GB, pool extent %.2f GB / %.2f GB",
        len(sorted_bufs),
        peak / (1024**3),
        pool_extent / (1024**3),
        SEGMENT_SIZE / (1024**3),
    )
    V.graph.pool_size = pool_extent

    return nodes
