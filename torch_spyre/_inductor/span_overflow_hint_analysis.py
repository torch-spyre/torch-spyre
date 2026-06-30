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

"""Span-overflow tile planning for oversized pointwise ops."""

from __future__ import annotations

import math

from dataclasses import dataclass

from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch_spyre._C import SpyreTensorLayout

from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import indirect_info_from_op
from .work_division import MAX_SPAN_BYTES, core_split


logger = get_inductor_logger("span_overflow_hint_analysis")


@dataclass(frozen=True)
class ChunkingInfo:
    """Physical span facts for one op before coarse tiling."""

    total_bytes: int
    per_core_span: int
    core_split_estimate: int
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    selected_host_dim: int
    stick_elems: int
    reason: str | None = None


@dataclass(frozen=True)
class SpanOverflowTilePlan:
    """Coarse-tiling request produced by span-overflow analysis."""

    selected_host_dim: int
    split_count: int
    is_reduction: bool
    chunking_info: ChunkingInfo
    reason: str | None = None


@dataclass(frozen=True)
class SpanDimInfo:
    """Mapping from the first span-controlling device dim to a host dim."""

    selected_host_dim: int
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    core_split_estimate: int
    skipped_outer_device_dims: tuple[int, ...] = ()


def _find_max_divisible_core_split(dim_size: int, max_cores: int) -> int:
    """Return the work-division core split estimate for one dim."""
    return core_split(dim_size, max_cores)


def _layout_has_static_span_metadata(layout: FixedTiledLayout) -> bool:
    """Return True when span planning can use concrete layout metadata."""
    try:
        for values in (
            layout.size,
            layout.stride,
            layout.device_layout.device_size,
            layout.device_layout.stride_map,
        ):
            for value in values:
                int(value)
        int(layout.device_layout.elems_per_stick())
    except (TypeError, ValueError):
        return False
    return True


def _iter_span_dim_infos(
    layout: FixedTiledLayout,
    max_cores: int,
) -> list[SpanDimInfo]:
    """Return mappable span-controlling dims in outer-to-inner order."""
    stl = layout.device_layout
    device_size = [int(s) for s in stl.device_size]
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]
    stick_elems = stl.elems_per_stick()

    infos: list[SpanDimInfo] = []
    skipped_outer_device_dims: list[int] = []
    for device_dim in range(len(device_size) - 1):
        if device_size[device_dim] <= 1:
            skipped_outer_device_dims.append(device_dim)
            continue

        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            skipped_outer_device_dims.append(device_dim)
            continue

        exact_matches = [
            d for d, s in enumerate(host_stride) if host_size[d] > 1 and s == sm
        ]
        stick_scaled_matches = [
            d
            for d, s in enumerate(host_stride)
            if host_size[d] > 1 and s * stick_elems == sm and d not in exact_matches
        ]
        matching_dims = exact_matches or stick_scaled_matches

        if not matching_dims:
            skipped_outer_device_dims.append(device_dim)
            continue

        selected_host_dim = matching_dims[0]
        selected_device_dim_size = device_size[device_dim]
        infos.append(
            SpanDimInfo(
                selected_host_dim=selected_host_dim,
                selected_device_dim_size=selected_device_dim_size,
                selected_device_span_stride_elems=math.prod(
                    device_size[device_dim + 1 :]
                ),
                core_split_estimate=_find_max_divisible_core_split(
                    selected_device_dim_size, max_cores
                ),
                skipped_outer_device_dims=tuple(skipped_outer_device_dims),
            )
        )

    return infos


def _find_outermost_span_dim(
    layout: FixedTiledLayout,
    max_cores: int,
) -> SpanDimInfo | None:
    """Find the outermost mapped device dim that can reduce memory span."""
    infos = _iter_span_dim_infos(layout, max_cores)
    return infos[0] if infos else None


def _compute_num_chunks(
    chunking_info: ChunkingInfo,
    max_cores: int,
) -> int:
    """Return the minimum coarse-tile count required by span and total limits."""
    if chunking_info.selected_device_dim_size == 0:
        return max(
            1, math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
        )

    num_from_span = math.ceil(chunking_info.per_core_span / MAX_SPAN_BYTES)
    num_from_total = math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
    return max(num_from_span, num_from_total)


def _choose_divisible_split_count(full_size: int, required_count: int) -> int:
    """Choose the smallest divisor of ``full_size`` at least ``required_count``."""
    if required_count <= 1:
        return 1
    if required_count > full_size:
        raise Unsupported(
            f"Cannot choose coarse-tile split count for dimension size {full_size}: "
            f"required count {required_count} exceeds the dimension size."
        )

    best = full_size
    for divisor in range(1, math.isqrt(full_size) + 1):
        if full_size % divisor != 0:
            continue
        paired = full_size // divisor
        if required_count <= divisor < best:
            best = divisor
        if required_count <= paired < best:
            best = paired
    return best


def _post_tile_layout(
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    split_count: int,
    op_name: str,
) -> FixedTiledLayout:
    """Build the per-tile layout used for post-tile validation."""
    if split_count <= 0:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: split_count must be positive, "
            f"got {split_count}."
        )

    new_size = list(original_layout.size)
    if selected_host_dim >= len(new_size):
        raise Unsupported(
            f"Cannot auto-tile {op_name}: selected host dim {selected_host_dim} "
            f"is out of bounds for layout size {new_size}."
        )

    try:
        full_size = int(new_size[selected_host_dim])
    except (TypeError, ValueError) as exc:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: selected host dim "
            f"{selected_host_dim} has non-integral layout size "
            f"{new_size[selected_host_dim]!r}."
        ) from exc

    if full_size % split_count != 0:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: selected host dim size {full_size} "
            f"is not divisible by split_count {split_count}."
        )

    new_size[selected_host_dim] = full_size // split_count
    new_stride = list(FlexibleLayout.contiguous_strides(new_size))

    # TODO: Replace this copied coarse_tile layout-resize pattern with the
    # _resize_device_layout helper from PR #2912 once it lands.  The helper is
    # more robust for preserving Spyre physical layout details.
    orig_stl = original_layout.device_layout
    sm_last = int(list(orig_stl.stride_map)[-1])
    new_strides_ints = [int(s) for s in new_stride]
    new_size_ints = [int(s) for s in new_size]
    within_stick_dim = next(
        (i for i, s in enumerate(new_strides_ints) if s == sm_last), None
    )
    if within_stick_dim is None:
        within_stick_dim = len(new_size_ints) - 1

    ndim = len(new_size_ints)
    dim_order = [i for i in range(ndim) if i != within_stick_dim] + [within_stick_dim]
    device_layout = SpyreTensorLayout(
        new_size_ints,
        new_strides_ints,
        original_layout.dtype,
        dim_order,
    )
    return FixedTiledLayout(
        original_layout.device,
        original_layout.dtype,
        new_size,
        new_stride,
        device_layout,
    )


def _within_stick_host_dim(layout: FixedTiledLayout) -> int:
    """Return the host dim represented by the physical within-stick dim."""
    sm_last = int(list(layout.device_layout.stride_map)[-1])
    host_stride = [int(s) for s in layout.stride]
    return next(
        (i for i, s in enumerate(host_stride) if s == sm_last),
        len(host_stride) - 1,
    )


def _post_tile_stick_alignment_error(
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    split_count: int,
) -> str | None:
    """Return a diagnostic if coarse tiling cuts through physical sticks."""
    if split_count <= 1:
        return None

    within_stick_dim = _within_stick_host_dim(original_layout)
    if selected_host_dim != within_stick_dim:
        return None

    full_size = int(original_layout.size[selected_host_dim])
    tile_size = full_size // split_count
    stick_elems = original_layout.device_layout.elems_per_stick()
    if tile_size % stick_elems == 0:
        return None

    return (
        f"split_count {split_count} makes selected host dim {selected_host_dim} "
        f"tile size {tile_size}, which is not aligned to Spyre stick size "
        f"{stick_elems}; coarse-tile boundaries would cut through physical sticks"
    )


def _post_tile_span_ok(
    split_count: int,
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    max_cores: int,
    op_name: str,
) -> bool:
    """Return whether the post-tile layout fits span and total limits."""
    tiled_layout = _post_tile_layout(
        original_layout,
        selected_host_dim,
        split_count,
        op_name,
    )
    post_span_dim_info = _find_outermost_span_dim(tiled_layout, max_cores)
    return (
        _needs_chunking(
            tiled_layout,
            max_cores,
            post_span_dim_info,
            op_name=f"{op_name}:post_tile",
            trigger="post_tile_validation",
        )
        is None
    )


def _post_tile_validated_split_count(
    op_name: str,
    full_size: int,
    required_count: int,
    chunking_info: ChunkingInfo,
    original_layout: FixedTiledLayout,
    max_cores: int,
) -> int:
    """Return the smallest bounded, stick-safe split count that validates."""
    initial = _choose_divisible_split_count(full_size, required_count)
    selected_host_dim = chunking_info.selected_host_dim
    candidates = sorted(
        {
            d
            for i in range(1, math.isqrt(full_size) + 1)
            if full_size % i == 0
            for d in (i, full_size // i)
            if d >= required_count
        }
    )

    stick_alignment_errors: list[str] = []
    for candidate in candidates:
        stick_error = _post_tile_stick_alignment_error(
            original_layout, selected_host_dim, candidate
        )
        if stick_error is not None:
            stick_alignment_errors.append(stick_error)
            logger.debug(
                "plan_span_overflow_tile: %s candidate=%d rejected: %s",
                op_name,
                candidate,
                stick_error,
            )
            continue

        if _post_tile_span_ok(
            candidate, original_layout, selected_host_dim, max_cores, op_name
        ):
            if candidate != initial:
                logger.debug(
                    "plan_span_overflow_tile: %s bumped split_count %d -> %d "
                    "after post-tile layout validation",
                    op_name,
                    initial,
                    candidate,
                )
            return candidate

        logger.debug(
            "plan_span_overflow_tile: %s candidate=%d still overflows after "
            "post-tile layout validation; trying next divisor",
            op_name,
            candidate,
        )

    if stick_alignment_errors:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: selected_host_dim size {full_size} "
            f"has no legal split >= {required_count} that preserves Spyre "
            f"stick alignment. First rejected candidate: "
            f"{stick_alignment_errors[0]}."
        )

    raise Unsupported(
        f"Cannot auto-tile {op_name}: no divisor of selected_host_dim "
        f"size {full_size} >= {required_count} makes the post-tile layout fit "
        f"within span_limit={MAX_SPAN_BYTES / (1024**2):.0f} MB and "
        f"total_limit={(MAX_SPAN_BYTES * max_cores) / (1024**3):.2f} GB "
        f"(selected_device_dim_size={chunking_info.selected_device_dim_size}, "
        f"stride_elems={chunking_info.selected_device_span_stride_elems}, "
        f"dtype_itemsize={original_layout.dtype.itemsize})."
    )


def _needs_chunking(
    layout: FixedTiledLayout,
    max_cores: int,
    span_dim_info: SpanDimInfo | None,
    *,
    op_name: str | None = None,
    trigger: str = "output_span",
) -> ChunkingInfo | None:
    """Return chunking info when the layout exceeds span or total limits."""
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    total_bytes = math.prod(device_size) * itemsize
    stick_elems = layout.device_layout.elems_per_stick()

    if span_dim_info is None:
        if total_bytes > MAX_SPAN_BYTES * max_cores:
            host_size = [int(s) for s in layout.size]
            fallback_selected_host_dim = max(
                range(len(host_size)), key=lambda d: host_size[d]
            )
            return ChunkingInfo(
                total_bytes=total_bytes,
                per_core_span=total_bytes,
                core_split_estimate=1,
                selected_device_dim_size=0,
                selected_device_span_stride_elems=0,
                selected_host_dim=fallback_selected_host_dim,
                stick_elems=stick_elems,
                reason=(
                    "no device dimension could be mapped to a splittable "
                    "host dimension via stride_map; using largest host dim "
                    "as fallback"
                ),
            )
        return None

    selected_device_dim_size = span_dim_info.selected_device_dim_size
    selected_device_span_stride_elems = span_dim_info.selected_device_span_stride_elems
    core_split_estimate = span_dim_info.core_split_estimate
    per_core_span = (
        math.ceil(selected_device_dim_size / core_split_estimate)
        * selected_device_span_stride_elems
        * itemsize
    )

    needs_chunk_for_span = per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_total = total_bytes > MAX_SPAN_BYTES * max_cores
    if not (needs_chunk_for_span or needs_chunk_for_total):
        return None

    logger.info(
        "[span_overflow_hint_analysis] trigger=%s op=%s "
        "selected_host_dim=%d selected_device_dim_size=%d "
        "selected_device_span_stride_elems=%d core_split_estimate=%d "
        "per_core_span=%.2f MB total=%.2f GB "
        "(shape=%s, dtype=%s, device_size=%s, "
        "span_limit=%.2f MB, total_limit=%.2f GB)",
        trigger,
        op_name or "<unknown>",
        span_dim_info.selected_host_dim,
        selected_device_dim_size,
        selected_device_span_stride_elems,
        core_split_estimate,
        per_core_span / (1024**2),
        total_bytes / (1024**3),
        list(layout.size),
        layout.dtype,
        device_size,
        MAX_SPAN_BYTES / (1024**2),
        (MAX_SPAN_BYTES * max_cores) / (1024**3),
    )
    return ChunkingInfo(
        total_bytes=total_bytes,
        per_core_span=per_core_span,
        core_split_estimate=core_split_estimate,
        selected_device_dim_size=selected_device_dim_size,
        selected_device_span_stride_elems=selected_device_span_stride_elems,
        selected_host_dim=span_dim_info.selected_host_dim,
        stick_elems=stick_elems,
        reason=(
            "skipped unmapped outer device dims "
            f"{span_dim_info.skipped_outer_device_dims}"
            if span_dim_info.skipped_outer_device_dims
            else None
        ),
    )


def _has_indirect_reads(op: ComputedBuffer) -> bool:
    """Return True if the op uses indirect/gather-style input reads."""
    try:
        _, _, indirect_sizes = indirect_info_from_op(op)
    except (AttributeError, RuntimeError, TypeError, Unsupported):
        return False
    return indirect_sizes is not None


def plan_span_overflow_tile(
    op: ComputedBuffer,
    max_cores: int,
) -> SpanOverflowTilePlan | None:
    """Return an automatic coarse-tile plan for a pointwise op if needed."""
    if not (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Pointwise)
        and isinstance(op.layout, FixedTiledLayout)
    ):
        return None

    if not _layout_has_static_span_metadata(op.layout):
        return None

    # Gather/indirect-access ops can lower as Pointwise ComputedBuffers, but
    # they require the dedicated indirect-access SDSC path rather than automatic
    # output coarse tiling.
    if _has_indirect_reads(op):
        return None

    return _plan_pointwise_span_overflow_tile(op, max_cores)


def _plan_pointwise_span_overflow_tile(
    op: ComputedBuffer,
    max_cores: int,
) -> SpanOverflowTilePlan | None:
    """Plan one single-dimension coarse tile for an oversized pointwise op."""
    span_dim_info = _find_outermost_span_dim(op.layout, max_cores)
    chunking_info = _needs_chunking(
        op.layout,
        max_cores,
        span_dim_info,
        op_name=op.get_name(),
        trigger="output_span",
    )
    if chunking_info is None:
        return None

    selected_host_dim = chunking_info.selected_host_dim
    ranges = list(op.data.ranges)
    if selected_host_dim >= len(ranges):
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim "
            f"{selected_host_dim} is out of bounds for given data ranges {ranges}."
        )

    try:
        full_size = int(ranges[selected_host_dim])
    except (TypeError, ValueError) as exc:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim "
            f"{selected_host_dim} has non-integral range "
            f"{ranges[selected_host_dim]!r}."
        ) from exc

    if full_size <= 1:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim "
            f"{selected_host_dim} has unsplittable range {full_size}."
        )

    required_count = _compute_num_chunks(chunking_info, max_cores)
    if required_count <= 1:
        return None

    split_count = _post_tile_validated_split_count(
        op.get_name(), full_size, required_count, chunking_info, op.layout, max_cores
    )

    return SpanOverflowTilePlan(
        selected_host_dim=selected_host_dim,
        split_count=split_count,
        is_reduction=False,
        chunking_info=chunking_info,
        reason=chunking_info.reason or "output span overflow",
    )
