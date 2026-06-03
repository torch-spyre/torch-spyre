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

"""Split oversized pointwise ops into memory-safe chunks.

Runs after ``propagate_spyre_tensor_layouts`` / ``insert_restickify`` and
before ``span_reduction``.  Each chunk becomes a normal
``ComputedBuffer`` that work-division handles without special-casing.
"""

import math

from dataclasses import dataclass
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Scatter,
)
from torch._inductor.virtualized import V

from . import config
from .work_division import MAX_SPAN_BYTES
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger


logger = get_inductor_logger("chunk_large_tensors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingInfo:
    total_bytes: int
    per_core_span: int
    best_split: int
    dev_dim_size: int
    dev_dim_stride: int
    host_dim: int
    stick_elems: int


def _find_best_split(dim_size: int, max_cores: int) -> int:
    """Return largest divisor of dim_size that is <= max_cores."""
    for i in range(min(max_cores, dim_size), 0, -1):
        if dim_size % i == 0:
            return i
    return 1


def _find_controlling_dim(
    layout: FixedTiledLayout,
    max_cores: int,
) -> tuple[int, int, int, int] | None:
    """Find the outermost splittable device dim and its best core split.

    Walks device dims outer-to-inner (skipping stick dim). For each
    device dim, uses stride_map to find the corresponding host dim.
    Returns the first host dim with size > 1.

    The controlling dim determines per-core memory span.
    Splitting inner dims increases parallelism but does NOT reduce span.
    Splitting the stick dim is not supported (atomic memory unit).

    Returns
    -------
    (host_dim_idx, dev_dim_size, dev_dim_stride, best_split)
        where best_split is the largest divisor of dev_dim_size
        that is <= max_cores.  Returns None if no valid splittable
        dim is found.

    Example::

        host_size   = [32,   8193,  1740]
        device_size = [8193, 28,    32,  64]
        stride_map  = [1740, 64, 14255820,  1]

        device_dim 0: stride_map=1740 -> host_dim 1 (M=8193), size>1
        8193 = 3 * 2731 -> best_split = 3 (largest divisor <= 32)
        dev_dim_stride = 28 * 32 * 64 = 57344

        -> returns (host_dim=1, dev_dim_size=8193,
                    dev_dim_stride=57344, best_split=3)
    """
    stl = layout.device_layout
    device_size = [int(s) for s in stl.device_size]
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]
    stick_elems = stl.elems_per_stick()

    for device_dim in range(len(device_size) - 1):  # skip last=stick
        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            continue

        # Collect all host dims matching this stride with size > 1.
        # In practice stride_map values map to unique host strides so
        # matching_dims has at most one element.  The list handles the
        # theoretical edge case where a size-1 dim shares a stride
        # with a valid dim.
        matching_dims = [
            d for d, s in enumerate(host_stride) if s == sm and host_size[d] > 1
        ]

        if not matching_dims:
            continue

        host_dim = matching_dims[0]
        dev_dim_size = device_size[device_dim]
        dev_dim_stride = math.prod(device_size[device_dim + 1 :])

        # Skip dims smaller than one stick — cannot produce
        # stick-aligned chunks for dims with fewer elements
        # than stick_elems (e.g. host_dim size=32 < 64).
        if dev_dim_size < stick_elems:
            logger.debug(
                "Skipping device_dim %d: host_dim %d size=%d "
                "< stick_elems=%d, cannot chunk",
                device_dim,
                host_dim,
                host_size[host_dim],
                stick_elems,
            )
            continue

        # Find the largest divisor of dev_dim_size that fits within
        # max_cores.  This simulates the best split work_division can
        # do on this dim.
        best_split = _find_best_split(dev_dim_size, max_cores)
        return host_dim, dev_dim_size, dev_dim_stride, best_split

    return None


def _compute_num_chunks(
    chunking_info: ChunkingInfo,
    max_cores: int,
) -> int:
    """Compute number of chunks needed.

    Uses two estimates and picks the best:
    - num_from_span:  per_core_span fits in 256MB
    - num_from_total: total fits in max_cores × 256MB

    If chunking by num_from_total improves dim divisibility
    (chunk dim gets better core split), total formula is used.
    Otherwise takes max of both.
    """

    # Fallback path: no controlling dim found
    if chunking_info.dev_dim_size == 0:
        return max(
            1, math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
        )

    num_from_span = math.ceil(chunking_info.per_core_span / MAX_SPAN_BYTES)
    num_from_total = math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))

    # Check if chunking by num_from_total improves divisibility
    total_sticks_preview = math.ceil(
        chunking_info.dev_dim_size / chunking_info.stick_elems
    )
    chunk_sticks_preview = math.ceil(total_sticks_preview / max(num_from_total, 1))
    chunk_dim_preview = chunk_sticks_preview * chunking_info.stick_elems
    chunk_best_split = _find_best_split(chunk_dim_preview, max_cores)

    if chunk_best_split > chunking_info.best_split and num_from_total > 1:
        return num_from_total
    return max(num_from_span, num_from_total)


def _needs_chunking(
    layout: FixedTiledLayout,
    max_cores: int,
    controlling: tuple[int, int, int, int] | None,
) -> ChunkingInfo | None:
    """Return total_device_bytes if this op needs chunking, else None.

    Uses the controlling dim's best_split (already computed in
    ``_find_controlling_dim``) to simulate work_division's per-core span.

    Two cases trigger chunking:

    1. ``per_core_span > 256 MB`` after best split on controlling dim
       -- catches small tensors with prime-like dims,
       e.g. [32, 8193, 1740]: best_split=3, per_core=313 MB > 256 MB

    2. ``total_bytes > 256 MB * max_cores``
       -- catches large tensors regardless of divisibility,
       e.g. [32, 8192, 17408]: total=9.17 GB > 8 GB

    Falls back to total_bytes threshold if no controlling dim found.
    """
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    total_bytes = math.prod(device_size) * itemsize
    stick_elems = layout.device_layout.elems_per_stick()

    if controlling is None:
        if total_bytes > MAX_SPAN_BYTES * max_cores:
            host_size = [int(s) for s in layout.size]
            fallback_host_dim = max(range(len(host_size)), key=lambda d: host_size[d])
            return ChunkingInfo(
                total_bytes=total_bytes,
                per_core_span=total_bytes,
                best_split=1,
                dev_dim_size=0,
                dev_dim_stride=0,
                host_dim=fallback_host_dim,
                stick_elems=stick_elems,
            )
        return None

    host_dim, dev_dim_size, dev_dim_stride, best_split = controlling

    per_core_span = math.ceil(dev_dim_size / best_split) * dev_dim_stride * itemsize

    needs_chunk_for_span = per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_total = total_bytes > MAX_SPAN_BYTES * max_cores

    if needs_chunk_for_span or needs_chunk_for_total:
        logger.info(
            "Op needs chunking: dev_dim_size=%d best_split=%d "
            "per_core_span=%.2fMB total=%.2fGB "
            "(span_limit=256MB total_limit=%.2fGB)",
            dev_dim_size,
            best_split,
            per_core_span / (1024**2),
            total_bytes / (1024**3),
            (MAX_SPAN_BYTES * max_cores) / (1024**3),
        )
        return ChunkingInfo(
            total_bytes=total_bytes,
            per_core_span=per_core_span,
            best_split=best_split,
            dev_dim_size=dev_dim_size,
            dev_dim_stride=dev_dim_stride,
            host_dim=host_dim,
            stick_elems=stick_elems,
        )
    return None


def _make_chunk_layout(
    original_ftl: FixedTiledLayout,
    split_dim_idx: int,
    chunk_size: int,
) -> FixedTiledLayout:
    """Build a ``FixedTiledLayout`` for a single chunk."""
    from torch_spyre._C import SpyreTensorLayout

    host_size = [int(s) for s in original_ftl.size]
    host_size[split_dim_idx] = chunk_size

    host_stride = [1] * len(host_size)
    for d in range(len(host_size) - 2, -1, -1):
        host_stride[d] = host_stride[d + 1] * host_size[d + 1]

    stl = SpyreTensorLayout(host_size, original_ftl.dtype)
    return FixedTiledLayout(
        original_ftl.device,
        original_ftl.dtype,
        host_size,
        host_stride,
        stl,
    )


def _make_chunk_fn(orig_fn, dim: int, offset: int):
    """Return an ``inner_fn`` that shifts the split dim by *offset*."""

    def inner_fn(index):
        idx = list(index)
        idx[dim] = idx[dim] + offset
        return orig_fn(idx)

    return inner_fn


def _make_output_indexer(offset: int, split_dim: int):
    """Return a scatter output indexer shifted by *offset* on *split_dim*."""

    def output_indexer(index):
        out = list(index)
        out[split_dim] = out[split_dim] + offset
        return out

    return output_indexer


def _register_and_insert(
    buf: ComputedBuffer,
    op: ComputedBuffer,
    operations: list[Operation],
    insert_pos: int,
) -> int:
    """Register *buf* in the graph and insert it at *insert_pos*.

    ``V.graph.register_operation`` appends to the same ``operations``
    list, so the duplicate is removed before the positioned insert.

    Returns the next insert position.
    """
    buf.name = V.graph.register_buffer(buf)
    V.graph.register_operation(buf)
    buf.origins = op.origins
    if buf in operations:
        operations.remove(buf)
    operations.insert(insert_pos, buf)
    return insert_pos + 1


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------


def _chunk_op(
    op: ComputedBuffer,
    max_cores: int,
    operations: list[Operation],
    op_index: int,
    chunking_info: ChunkingInfo,
    original_ftl: FixedTiledLayout,
) -> int:
    """Split *op* into memory-safe chunks along the controlling dim.

    Chunk 0 is the original op shrunk in-place (ranges only; layout
    stays full-size so the scheduler finds it by name). Chunks 1..N-1
    are direct ``Scatter`` mutation buffers that compute each chunk
    inline and write it into the corresponding region of the original
    output.

    Chunk sizes are stick-aligned (multiples of ``stick_elems``) so the
    hardware scheduler always finds valid chunk-parameter candidates.
    """

    original_ranges = list(op.data.ranges)
    original_inner_fn = op.data.inner_fn
    split_dim_idx = chunking_info.host_dim
    split_dim_full_size = int(original_ranges[split_dim_idx])
    stick_elems = chunking_info.stick_elems

    # -- Step 1: decide number of chunks --
    # Two estimates; take the one that gives the best result:
    #   num_from_span  : per_core_span fits in 256 MB
    #   num_from_total : total fits within max_cores * 256 MB
    num_chunks = _compute_num_chunks(chunking_info, max_cores)

    # -- Step 2: stick-aligned chunk size --
    # Chunk at stick level so every chunk is a multiple of stick_elems.
    # SpyreTensorLayout pads the last stick, so reading slightly beyond
    # split_dim_full_size is safe.
    total_sticks = math.ceil(split_dim_full_size / stick_elems)
    sticks_per_chunk = math.ceil(total_sticks / num_chunks)
    chunk_size = sticks_per_chunk * stick_elems
    num_chunks = math.ceil(total_sticks / sticks_per_chunk)

    logger.info(
        "Chunking %s: split_dim=%d full_size=%d "
        "sticks=%d sticks_per_chunk=%d "
        "chunk_size=%d num_chunks=%d total=%.2fGB",
        op.get_name(),
        split_dim_idx,
        split_dim_full_size,
        total_sticks,
        sticks_per_chunk,
        chunk_size,
        num_chunks,
        chunking_info.total_bytes / (1024**3),
    )

    # -- Chunk 0: shrink original op in-place --
    chunk0_size = min(chunk_size, split_dim_full_size)
    chunk0_ranges = list(original_ranges)
    chunk0_ranges[split_dim_idx] = chunk0_size
    object.__setattr__(op.data, "ranges", chunk0_ranges)

    insert_pos = op_index + 1
    n_inserted = 0

    # -- Chunks 1..N-1: direct scatter mutations into the original output --
    for chunk_idx in range(1, num_chunks):
        chunk_offset = chunk_idx * chunk_size
        remaining_elems = max(0, split_dim_full_size - chunk_offset)

        remaining_sticks = math.ceil(remaining_elems / stick_elems)
        this_chunk_size = min(remaining_sticks * stick_elems, chunk_size)

        chunk_ranges = list(original_ranges)
        chunk_ranges[split_dim_idx] = this_chunk_size

        mutation_data = Scatter(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=_make_chunk_fn(original_inner_fn, split_dim_idx, chunk_offset),
            ranges=chunk_ranges,
            output_indexer=_make_output_indexer(chunk_offset, split_dim_idx),
        )
        mutation_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=mutation_data,
        )
        insert_pos = _register_and_insert(mutation_buf, op, operations, insert_pos)
        n_inserted += 1
    return n_inserted


def chunk_large_tensors(operations: list[Operation]) -> None:
    """Split pointwise ops whose device footprint exceeds the limit.

    Must run **after** ``propagate_spyre_tensor_layouts`` /
    ``insert_restickify`` and **before** ``span_reduction``.

    Note: ``ir.Pointwise`` is broader than ``torch.Tag.pointwise``.
    ``inner_fn`` can in theory access non-corresponding input indices,
    making chunking unsafe.

    TODO: Use OpsHandler to verify output[i] only uses input[i].
    """
    max_cores = config.sencores
    i = 0
    while i < len(operations):
        op = operations[i]

        if (
            isinstance(op, ComputedBuffer)
            and isinstance(op.data, Pointwise)
            and isinstance(op.layout, FixedTiledLayout)
            and len(op.data.ranges) >= 2
        ):
            controlling = _find_controlling_dim(op.layout, max_cores)
            chunking_info = _needs_chunking(op.layout, max_cores, controlling)
            if chunking_info is not None:
                n_inserted = _chunk_op(
                    op,
                    max_cores,
                    operations,
                    i,
                    chunking_info,
                    op.layout,
                )
                i += n_inserted
        i += 1
