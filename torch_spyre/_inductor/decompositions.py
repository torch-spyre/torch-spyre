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

"""Spyre-specific decompositions and PrivateUse1 dispatch-key kernels.

There is exactly one public entry point: ``register_spyre_decompositions``.
It records a decomposition for ``torch.compile`` (consumed via Inductor's
``get_decomp_fn``); for aten ops, a PrivateUse1 kernel reaching the same
implementation is auto-installed at runtime init so eager-mode dispatch
reaches it too. Eager-only registration is intentionally not exposed.

The Spyre decomposition table built by ``get_spyre_decomp_table`` is
independent from PyTorch's global ``torch._inductor.decomposition.decompositions``
registry; Spyre never mutates the global table.
"""

import dataclasses
import math
import threading
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch._decomp as decomp
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes

from .constants import DEVICE_NAME, FP8_E4M3FN_MAX, FP8_E4M3FN_MIN
from .errors import Unsupported
from .sliding_window_plan import (
    MAX_QUERY_BLOCK,
    STICK,
    SlidingWindowPlan,
    check_window_read,
    plan_sliding_window,
    query_blocking,
    rejection_reason,
)
from . import config
from .logging_utils import get_inductor_logger

from . import customops  # noqa: F401
from .wsr import for_each_tile
from torch_spyre._C import DataFormats, get_device_dtype, get_elem_in_stick
import torch_spyre._inductor.customops  # noqa: F401

logger = get_inductor_logger("decompositions")


_SDPA_MAX_SEQUENCE_TILE_SIZE = 512
# for_each_tile feeds V directly to the second matmul. A partial-width V tile
# retains its transposed producer layout when padded to a full fp16/bf16 stick;
# the resulting stride amplification can exceed the 256 MiB per-core span.
_SDPA_SEQUENCE_TILE_ALIGNMENT = 64
_SDPA_MAX_TILE_PAIRS_PER_LOOP_GROUP = 16
_SDPA_PREFERRED_HEADS_PER_TILE = (4, 2, 1)
_SDPA_MHA_MAX_HEAD_WORK_DIVISION = 4
_SDPA_MHA_QUERY_ONLY_MAX_HEADS = 8
_SDPA_MHA_QUERY_ONLY_MIN_KV_BLOCKS = 8
# Decode and SWA use this narrower, separately calibrated live-set estimate.
# These constants do not represent a legacy non-HOP SDPA implementation.
_SDPA_NARROW_LIVE_SCORE_BUFFER_ALLOWANCE = 2
_SDPA_NARROW_LIVE_QUERY_BUFFER_ALLOWANCE = 2
# When every selected tile count is one, ``map_tiles`` and the K/V scan
# invoke their bodies directly. There is no HOP staging or carry handoff in
# that graph: the score allocation can be reused after its reduction and only
# the scaled query, weighted result, and normalized output overlap at peak.
_SDPA_DIRECT_LIVE_SCORE_BUFFER_ALLOWANCE = 1
_SDPA_DIRECT_LIVE_QUERY_BUFFER_ALLOWANCE = 3
_SDPA_TARGET_KV_BYTES_PER_CORE = 1024 * 1024
_SDPA_MAX_TARGET_KV_BYTES_PER_CORE = 2 * 1024 * 1024
_SDPA_GQA_HEADS_PER_KV_TARGET_MIB = 4
_SDPA_QUERY_ROWS_PER_KV_TARGET_MIB = 8
_SDPA_BASE_BURST_EFFICIENT_KV_BLOCK_SIZE = 512
_SDPA_MAX_BURST_EFFICIENT_KV_BLOCK_SIZE = 1024
# Nested HOP prefill keeps the scaled query and online-softmax carries live
# across the K loop. At the widest point, the dataflow contains four
# score-shaped values and seven query/output-shaped values once the enclosing
# map's tile staging and carry handoff are included. These are graph-derived
# buffer counts, rather than shape or model limits.
_SDPA_PREFILL_LIVE_SCORE_BUFFER_ALLOWANCE = 4
_SDPA_PREFILL_LIVE_QUERY_BUFFER_ALLOWANCE = 7

# A counted SWA K/V loop pays its carry handoff and loop-control costs for each
# query work partition.  Prefill sweeps show that retaining at least four query
# rows per core amortizes those costs while still exposing enough parallelism.
# This caps only GQA's query-only work division; MHA retains SDPA's joint
# head/query search.
_SWA_MIN_QUERY_ROWS_PER_CORE = 4

# DPO emits about seventeen additional sdsc_execute operations for every
# unrolled online-softmax block. Four K256 blocks can still win by retaining
# restickified K in LX, while four K1024 blocks lose badly to one long block.
# Permit two resident blocks at any measured size, and more only while their
# combined extent stays within the measured 1024-token efficient window.
_SDPA_DECODE_MIN_LX_RESIDENT_BLOCKS = 2
_SDPA_DECODE_MAX_MULTI_BLOCK_EXTENT = 1024

# Scratchpad allocation sweeps show that the restickified K is retained in LX
# while its per-core footprint is below roughly 2 KiB per reusing query head.
# This is deliberately a performance hint rather than an allocation guarantee;
# the allocator remains the authority on whether the value is actually kept.
_SDPA_RESTICK_LX_BYTES_PER_REUSING_HEAD = 2 * 1024


@dataclasses.dataclass(frozen=True)
class _SDPATilingConfig:
    """Static SDPA decomposition choices produced by the cost model."""

    strategy: str
    reason: str
    kv_block_size: int
    num_kv_blocks: int
    num_q_tiles: int
    q_tile_size: int
    num_batch_tiles: int
    num_head_tiles: int
    num_group_tiles: int
    kv_blocks_per_loop_group: int
    estimated_active_cores: int | None
    estimated_load_bursts: int
    estimated_hbm_bytes: int
    estimated_spill_buffers: int | None
    estimated_spill_bytes: int | None
    score_bytes_per_core: int | None
    estimated_live_bytes_per_core: int | None
    lx_budget_bytes: int


@dataclasses.dataclass(frozen=True)
class _SDPAPrefillPlan:
    """One exact nested-HOP tiling and its compiler-visible costs."""

    num_batch_tiles: int
    num_head_tiles: int
    num_group_tiles: int
    num_q_tiles: int
    q_tile_size: int
    kv: "_SDPAKVBlockCandidate"
    estimated_active_cores: int
    estimated_restick_active_cores: int
    estimated_restick_work_bytes: int
    estimated_load_bursts: int
    estimated_hbm_bytes: int
    estimated_hbm_transfer_waves: int
    estimated_spill_buffers: int
    estimated_spill_bytes: int
    estimated_work: int


@dataclasses.dataclass(frozen=True)
class _SWATilingConfig:
    """Static SWA decomposition choices produced by the cost model."""

    strategy: str
    reason: str
    kv_block_size: int
    num_kv_blocks: int
    num_head_tiles: int
    work_div: dict[str, int] | None
    score_bytes_per_core: int | None
    estimated_live_bytes_per_core: int | None
    kv_bytes_per_core: int | None
    restick_bytes_per_core: int | None
    restick_lx_eligible: bool | None
    estimated_dsc_executions: int | None
    lx_budget_bytes: int


def _sdpa_num_head_tiles(num_heads: int) -> int:
    """Use at most four heads per tile and return the required tile count."""
    for heads_per_tile in _SDPA_PREFERRED_HEADS_PER_TILE:
        if num_heads % heads_per_tile == 0:
            return num_heads // heads_per_tile
    return 1


def _sdpa_num_batch_tiles(batch_size: int) -> int:
    """Use two rows per exact tile, or one row when the batch size is odd."""
    return batch_size // 2 if batch_size % 2 == 0 else batch_size


def _sdpa_head_group_tiles(num_heads: int, num_kvheads: int) -> tuple[int, int]:
    """Keep at most four query heads in each fallback H/G tile."""
    if num_heads == num_kvheads:
        return _sdpa_num_head_tiles(num_heads), 1

    group_size = num_heads // num_kvheads
    return _sdpa_num_head_tiles(num_kvheads), group_size


def _sdpa_effective_group_tiles(
    num_group_tiles: int, num_q_tiles: int, num_kv_blocks: int
) -> int:
    """Return the number of group tiles that become an executed map loop.

    A group-only split does not reduce the sequence working set, so the lowering
    deliberately leaves the full G axis visible to work division when both
    sequence axes fit in one tile. Keep cost accounting and graph selection tied
    to that same effective count.
    """
    return num_group_tiles if num_q_tiles > 1 or num_kv_blocks > 1 else 1


def _sdpa_has_loop_boundary(
    *,
    num_non_group_outer_tiles: int,
    num_group_tiles: int,
    num_q_tiles: int,
    num_kv_blocks: int,
) -> bool:
    """Whether the selected counts emit any map or carry loop."""
    return (
        num_non_group_outer_tiles > 1
        or _sdpa_effective_group_tiles(num_group_tiles, num_q_tiles, num_kv_blocks) > 1
        or num_kv_blocks > 1
    )


def _num_tiles_for_max_extent(
    sequence_length: int, max_extent: int, *, tile_alignment: int = 1
) -> int:
    """Return an exact split count whose tile extent is at most ``max_extent``.

    Coarse tiling currently requires equal-sized tiles, so a simple ceiling is
    insufficient when it does not divide ``sequence_length``. Start with the
    minimum count that satisfies the extent cap and advance to the next exact
    divisor whose tile extent has the requested alignment. If the full extent
    is not aligned, or the extent cap is smaller than the alignment, no exact
    aligned split exists and only divisibility is enforced. Sequence lengths
    used by the adapters are normally stick-padded, so this resolves after only
    a few candidates.
    """
    if sequence_length < 1 or max_extent < 1 or tile_alignment < 1:
        raise ValueError(
            "sequence length, maximum extent, and alignment must be positive"
        )

    alignment_is_possible = (
        sequence_length % tile_alignment == 0 and max_extent >= tile_alignment
    )
    effective_alignment = tile_alignment if alignment_is_possible else 1
    minimum_tiles = max(1, (sequence_length + max_extent - 1) // max_extent)
    for num_tiles in range(minimum_tiles, sequence_length + 1):
        if (
            sequence_length % num_tiles == 0
            and (sequence_length // num_tiles) % effective_alignment == 0
        ):
            return num_tiles

    raise AssertionError("validated tiling inputs must have an exact tile")


def _padded_tiling_for_max_extent(
    sequence_length: int, max_extent: int, *, tile_alignment: int
) -> tuple[int, int]:
    """Return equal aligned tiles covering a possibly padded sequence."""
    if sequence_length < 1 or max_extent < 1 or tile_alignment < 1:
        raise ValueError(
            "sequence length, maximum extent, and alignment must be positive"
        )
    if max_extent < tile_alignment or max_extent % tile_alignment:
        raise ValueError("maximum extent must be a multiple of tile alignment")

    num_tiles = max(1, (sequence_length + max_extent - 1) // max_extent)
    unaligned_tile_size = (sequence_length + num_tiles - 1) // num_tiles
    tile_size = (
        (unaligned_tile_size + tile_alignment - 1) // tile_alignment * tile_alignment
    )
    return num_tiles, tile_size


def _kv_blocks_per_loop_group(num_q_tiles: int, num_kv_blocks: int) -> int:
    """Keep each SDPA backend bundle near the proven 4-by-4 size.

    The backend specializes a counted Lq loop across every unrolled Lk block.
    Bundle code size therefore scales with their product, not with the number of
    Lk blocks alone. Cap that product at sixteen when possible, while retaining at
    least one Lk block per group. Once Lq alone needs sixteen or more tiles,
    each explicit Lk block gets its own loop group.
    """
    return min(
        num_kv_blocks,
        max(1, _SDPA_MAX_TILE_PAIRS_PER_LOOP_GROUP // num_q_tiles),
    )


def _largest_exact_split(extent: int, limit: int) -> int:
    """Return the largest divisor of ``extent`` no larger than ``limit``."""
    for split in range(min(extent, limit), 0, -1):
        if extent % split == 0:
            return split
    return 1


def _sdpa_work_division(
    num_heads: int,
    num_kvheads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    num_cores: int,
) -> dict[str, int] | None:
    """Find the largest placeable head/query split for direct SDPA and SWA."""
    if num_cores < 1 or max_seqlen_q <= 1:
        return None

    if num_heads != num_kvheads:
        if num_heads % num_kvheads:
            return None
        query_split = _largest_exact_split(max_seqlen_q, num_cores)
        return {"max_seqlen_q": query_split} if query_split > 1 else None

    if (
        num_heads <= _SDPA_MHA_QUERY_ONLY_MAX_HEADS
        and max_seqlen_q % num_cores == 0
        and max_seqlen_kv
        >= _SDPA_MHA_QUERY_ONLY_MIN_KV_BLOCKS * _SDPA_MAX_SEQUENCE_TILE_SIZE
    ):
        return {"max_seqlen_q": num_cores}

    best: tuple[int, int] | None = None
    max_head_split = min(_SDPA_MHA_MAX_HEAD_WORK_DIVISION, num_heads, num_cores)
    for head_split in range(1, max_head_split + 1):
        if num_heads % head_split:
            continue
        query_split = _largest_exact_split(max_seqlen_q, num_cores // head_split)
        candidate = (head_split, query_split)
        if best is None or (math.prod(candidate), head_split) > (
            math.prod(best),
            best[0],
        ):
            best = candidate
    if best is None or math.prod(best) <= 1:
        return None
    return {"num_heads": best[0], "max_seqlen_q": best[1]}


def _sdpa_estimated_active_cores(tile_extents: tuple[int, ...], num_cores: int) -> int:
    """Estimate CP-SAT's largest exact split over visible inner axes."""
    products = {1}
    for extent in tile_extents:
        divisors = [
            split
            for split in range(1, min(extent, num_cores) + 1)
            if extent % split == 0
        ]
        products = {
            product * divisor
            for product in products
            for divisor in divisors
            if product * divisor <= num_cores
        }
    return max(products)


def _exact_tile_counts(extent: int) -> list[int]:
    """Return every exact map-loop trip count for a static axis."""
    return [count for count in range(1, extent + 1) if extent % count == 0]


def _axis_slice_is_dense(
    shape: tuple[int, ...], strides: tuple[int, ...], axis: int
) -> bool:
    """Whether one tile of ``axis`` is a contiguous region of the tensor."""
    expected_stride = 1
    for dim in range(len(shape) - 1, axis, -1):
        if shape[dim] > 1:
            if strides[dim] != expected_stride:
                return False
            expected_stride *= shape[dim]
    return strides[axis] == expected_stride


def _sdpa_mask_hbm_bytes(
    *,
    mask_shapes: tuple[tuple[int, ...], ...],
    axis_extents: tuple[int, ...],
    axis_tile_counts: tuple[int, ...],
    element_size: int,
) -> int:
    """Estimate mask loads, including replay along broadcast loop axes."""
    total = 0
    for shape in mask_shapes:
        if len(shape) != len(axis_extents):
            raise ValueError(
                f"SDPA mask rank {len(shape)} does not match rank {len(axis_extents)}"
            )
        replay = 1
        for size, extent, tile_count in zip(
            shape, axis_extents, axis_tile_counts, strict=True
        ):
            if size == 1 and extent != 1:
                replay *= tile_count
            elif size != extent:
                raise ValueError(
                    f"SDPA mask extent {size} is neither broadcast nor {extent}"
                )
        total += math.prod(shape) * element_size * replay
    return total


def _sdpa_hbm_transfer_waves(num_bytes: int, num_cores: int) -> int:
    """Convert aggregate traffic into calibrated full-card HBM load waves."""
    bytes_per_wave = num_cores * _SDPA_TARGET_KV_BYTES_PER_CORE
    return (num_bytes + bytes_per_wave - 1) // bytes_per_wave


def _sdpa_lx_budget_bytes() -> int:
    """Return the frontend-managed portion of each core's LX."""
    # Import lazily to avoid pulling the scratchpad planner into eager startup.
    from .scratchpad.allocator import _lx_planning_size

    return _lx_planning_size()


def _sdpa_estimated_live_bytes_per_core(
    *,
    batch_size: int,
    heads_per_core: int,
    query_rows_per_core: int,
    kv_block_size: int,
    head_dim: int,
    element_size: int,
    restick_bytes_per_core: int = 0,
    full_sdpa_prefill: bool = False,
    has_loop_boundary: bool = True,
) -> tuple[int, int]:
    """Estimate the co-live, non-streamed values for one SDPA iteration.

    V is streamed by the second matmul. K is restickified before the first
    matmul and therefore participates in full-SDPA prefill residency. Full
    nested prefill accounts for every score- and query-shaped value that can
    overlap at a map/carry boundary. Any plan with no effective loop uses the
    direct-body live set because ``map_tiles`` and the one-block scan bypass
    ``for_each_tile``. Looped decode and SWA retain their separately calibrated
    two-score/two-query estimate. These flags select a liveness accounting
    regime; they do not select a different SDPA implementation.
    """
    score_bytes = (
        batch_size * heads_per_core * query_rows_per_core * kv_block_size * element_size
    )
    query_bytes = (
        batch_size * heads_per_core * query_rows_per_core * head_dim * element_size
    )
    accumulator_bytes = batch_size * heads_per_core * query_rows_per_core * element_size
    if not has_loop_boundary:
        score_allowance = _SDPA_DIRECT_LIVE_SCORE_BUFFER_ALLOWANCE
        query_allowance = _SDPA_DIRECT_LIVE_QUERY_BUFFER_ALLOWANCE
    elif full_sdpa_prefill:
        score_allowance = _SDPA_PREFILL_LIVE_SCORE_BUFFER_ALLOWANCE
        query_allowance = _SDPA_PREFILL_LIVE_QUERY_BUFFER_ALLOWANCE
    else:
        score_allowance = _SDPA_NARROW_LIVE_SCORE_BUFFER_ALLOWANCE
        query_allowance = _SDPA_NARROW_LIVE_QUERY_BUFFER_ALLOWANCE
    estimated_live_bytes = (
        score_allowance * score_bytes
        + query_allowance * query_bytes
        + 2 * accumulator_bytes
        + restick_bytes_per_core
    )
    return score_bytes, estimated_live_bytes


@dataclasses.dataclass(frozen=True)
class _SDPAKVBlockCandidate:
    """Compiler-derived costs for one aligned K/V block size."""

    block_size: int
    num_blocks: int
    score_bytes_per_core: int
    query_bytes_per_core: int
    estimated_live_bytes_per_core: int
    kv_bytes_per_core: int
    estimated_restick_active_cores: int
    restick_bytes_per_core: int
    restick_lx_eligible: bool
    estimated_dsc_executions: int
    estimated_load_bursts: int


def _sdpa_kv_block_sizes(max_seqlen_kv: int) -> list[int]:
    """Enumerate power-of-two burst sizes plus the aligned full extent."""
    aligned_extent = max(64, ((max_seqlen_kv + 63) // 64) * 64)
    candidates = []
    block_size = 64
    while block_size < aligned_extent:
        candidates.append(block_size)
        block_size *= 2
    candidates.append(aligned_extent)
    return candidates


def _sdpa_query_tile_sizes(max_seqlen_q: int) -> list[int]:
    """Enumerate exact query tiles from the full chunk down to one row."""
    candidates = [max_seqlen_q]
    target = 1 << (max_seqlen_q.bit_length() - 1)
    while target >= 1:
        num_tiles = _num_tiles_for_max_extent(max_seqlen_q, target)
        tile_size = max_seqlen_q // num_tiles
        if tile_size not in candidates:
            candidates.append(tile_size)
        target //= 2
    return candidates


def _sdpa_target_kv_bytes_per_core(
    num_heads: int, num_kvheads: int, query_rows_per_core: int
) -> int:
    """Return the useful K/V streaming footprint for one core."""
    gqa_reuse = num_heads // num_kvheads if num_heads % num_kvheads == 0 else 1
    head_target = (
        _SDPA_TARGET_KV_BYTES_PER_CORE
        * max(_SDPA_GQA_HEADS_PER_KV_TARGET_MIB, gqa_reuse)
        // _SDPA_GQA_HEADS_PER_KV_TARGET_MIB
    )
    query_target = (
        _SDPA_TARGET_KV_BYTES_PER_CORE
        * max(_SDPA_QUERY_ROWS_PER_KV_TARGET_MIB, query_rows_per_core)
        // _SDPA_QUERY_ROWS_PER_KV_TARGET_MIB
    )
    return min(
        _SDPA_MAX_TARGET_KV_BYTES_PER_CORE,
        head_target,
        query_target,
    )


def _sdpa_burst_efficient_kv_block_limit(query_rows_per_core: int) -> int:
    """Bound K by the DPO BMM profile for the available query-row reuse."""
    scale = max(1, query_rows_per_core // _SDPA_QUERY_ROWS_PER_KV_TARGET_MIB)
    return min(
        _SDPA_MAX_BURST_EFFICIENT_KV_BLOCK_SIZE,
        _SDPA_BASE_BURST_EFFICIENT_KV_BLOCK_SIZE * scale,
    )


def _sdpa_restick_active_cores(
    *, num_heads: int, num_cores: int, core_split: dict[str, int] | None
) -> int:
    """Estimate the cores sharing a restickified K block.

    Explicit query work division determines this directly. Without one, the
    DPO schedules observed for decode use at most two lanes per query head.
    """
    if core_split is not None:
        return min(num_cores, math.prod(core_split.values()))
    return min(num_cores, max(1, 2 * num_heads))


def _sdpa_kv_candidates(
    *,
    batch_size: int,
    num_heads: int,
    num_kvheads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim: int,
    element_size: int,
    num_cores: int,
    query_tile_size: int | None = None,
    group_tile_size: int = 1,
    num_non_group_outer_tiles: int = 1,
    num_group_tiles: int = 1,
    num_q_tiles: int = 1,
    full_sdpa_prefill: bool = False,
    work_div: dict[str, int] | None = None,
    pad_extent: bool = False,
) -> list[_SDPAKVBlockCandidate]:
    """Estimate LX pressure, restick traffic, and DPO execution count.

    ``pad_extent`` lets SWA cost the aligned physical blocks it will materialize
    instead of requiring every candidate block to divide the logical extent.
    """
    if full_sdpa_prefill:
        assert query_tile_size is not None
        kv_heads_per_core = num_kvheads
        # K has no G or Q axis.  Be deliberately conservative here: generated
        # plans can use more lanes when a downstream Q split is compatible,
        # but the only parallel axes guaranteed before layout planning are B
        # and Hkv.  Overestimating this split makes a near-capacity plan look
        # resident and is much costlier than rejecting one marginal K tile.
        restick_active_cores = min(num_cores, batch_size * num_kvheads)
    else:
        # Preserve the calibrated decode model. Its operations are too narrow
        # for the head/query partition used by chunked prefill.
        head_split = work_div.get("num_heads", 1) if work_div is not None else 1
        query_split = work_div.get("max_seqlen_q", 1) if work_div is not None else 1
        heads_per_core = num_heads // head_split
        kv_heads_per_core = num_kvheads // head_split
        query_rows_per_core = max_seqlen_q // query_split
        restick_active_cores = _sdpa_restick_active_cores(
            num_heads=num_heads, num_cores=num_cores, core_split=work_div
        )
    gqa_reuse = num_heads // num_kvheads if num_heads % num_kvheads == 0 else 1
    restick_lx_limit = _SDPA_RESTICK_LX_BYTES_PER_REUSING_HEAD * gqa_reuse
    result = []
    seen_block_sizes = set()
    for max_block_size in _sdpa_kv_block_sizes(max_seqlen_kv):
        if pad_extent:
            num_blocks, effective_block_size = _padded_tiling_for_max_extent(
                max_seqlen_kv,
                max_block_size,
                tile_alignment=_SDPA_SEQUENCE_TILE_ALIGNMENT,
            )
        else:
            num_blocks = _num_tiles_for_max_extent(
                max_seqlen_kv,
                max_block_size,
                tile_alignment=_SDPA_SEQUENCE_TILE_ALIGNMENT,
            )
            effective_block_size = max_seqlen_kv // num_blocks
        if effective_block_size in seen_block_sizes:
            continue
        seen_block_sizes.add(effective_block_size)
        effective_group_tiles = _sdpa_effective_group_tiles(
            num_group_tiles, num_q_tiles, num_blocks
        )
        num_outer_tiles = num_non_group_outer_tiles * effective_group_tiles
        if full_sdpa_prefill:
            assert query_tile_size is not None
            # Model every axis visible in the innermost HOP body. When a G-only
            # map is elided, the complete G extent remains visible to CP-SAT.
            effective_group_tile_size = (
                group_tile_size * num_group_tiles // effective_group_tiles
            )
            active_cores = _sdpa_estimated_active_cores(
                (
                    batch_size,
                    num_kvheads,
                    effective_group_tile_size,
                    query_tile_size,
                ),
                num_cores,
            )
            head_rows_per_core = (
                batch_size
                * num_kvheads
                * effective_group_tile_size
                * query_tile_size
                // active_cores
            )
            heads_per_core = head_rows_per_core
            query_rows_per_core = 1
        restick_bytes_per_core = (
            batch_size * num_kvheads * effective_block_size * head_dim * element_size
            + restick_active_cores
            - 1
        ) // restick_active_cores
        kv_bytes_per_core = (
            batch_size
            * kv_heads_per_core
            * effective_block_size
            * head_dim
            * element_size
        )
        score_bytes, live_bytes = _sdpa_estimated_live_bytes_per_core(
            batch_size=1 if full_sdpa_prefill else batch_size,
            heads_per_core=heads_per_core,
            query_rows_per_core=query_rows_per_core,
            kv_block_size=effective_block_size,
            head_dim=head_dim,
            element_size=element_size,
            restick_bytes_per_core=(restick_bytes_per_core if full_sdpa_prefill else 0),
            full_sdpa_prefill=full_sdpa_prefill,
            has_loop_boundary=_sdpa_has_loop_boundary(
                num_non_group_outer_tiles=num_non_group_outer_tiles,
                num_group_tiles=num_group_tiles,
                num_q_tiles=num_q_tiles,
                num_kv_blocks=num_blocks,
            ),
        )
        blocks_per_group = _kv_blocks_per_loop_group(1, num_blocks)
        num_loop_groups = (num_blocks + blocks_per_group - 1) // blocks_per_group
        result.append(
            _SDPAKVBlockCandidate(
                block_size=effective_block_size,
                num_blocks=num_blocks,
                score_bytes_per_core=score_bytes,
                query_bytes_per_core=(score_bytes * head_dim // effective_block_size),
                estimated_live_bytes_per_core=live_bytes,
                kv_bytes_per_core=kv_bytes_per_core,
                estimated_restick_active_cores=restick_active_cores,
                restick_bytes_per_core=restick_bytes_per_core,
                restick_lx_eligible=restick_bytes_per_core <= restick_lx_limit,
                # Eight fixed executes, about seventeen for every unrolled
                # block, and one carry boundary per loop group in the DBO
                # bundles inspected during calibration.
                estimated_dsc_executions=num_outer_tiles
                * (8 + 17 * num_blocks + num_loop_groups),
                # One K and one V stream per online-softmax trip. Outer HOP
                # tiles replay both streams; wider resident blocks reduce the
                # number of independent load bursts without a token cutoff.
                estimated_load_bursts=2 * num_outer_tiles * num_blocks,
            )
        )
    return result


def _select_sdpa_tiling(
    *,
    batch_size: int,
    num_heads: int,
    num_kvheads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim: int,
    element_size: int,
    num_cores: int,
    lx_budget_bytes: int,
    mask_shapes: tuple[tuple[int, ...], ...] = (),
    head_tile_staging_bytes: int = 0,
) -> _SDPATilingConfig:
    """Choose SDPA tiling from compiler-visible costs.

    Prefill enumerates exact B/Hkv/G/Lq/Lk plans. It distinguishes the direct
    body from nested-HOP staging in its live-set estimate, permits at most one
    whole intermediate to spill, and balances loop/DSC executions, HBM load
    bursts, and aggregate transfer waves. Decode retains its separately
    calibrated policy. No model identity or sequence-length cutoff participates
    in the decision.
    """
    quarter_kv_stick_aligned = max(64, ((max_seqlen_kv + 3) // 4 + 63) // 64 * 64)
    fallback_kv_block_limit = min(
        _SDPA_MAX_SEQUENCE_TILE_SIZE, quarter_kv_stick_aligned
    )
    fallback_num_kv_blocks = _num_tiles_for_max_extent(
        max_seqlen_kv,
        fallback_kv_block_limit,
        tile_alignment=_SDPA_SEQUENCE_TILE_ALIGNMENT,
    )
    fallback_kv_block_size = max_seqlen_kv // fallback_num_kv_blocks
    fallback_num_q_tiles = _num_tiles_for_max_extent(
        max_seqlen_q, _SDPA_MAX_SEQUENCE_TILE_SIZE
    )
    fallback_q_tile_size = max_seqlen_q // fallback_num_q_tiles
    fallback_num_head_tiles, fallback_num_group_tiles = _sdpa_head_group_tiles(
        num_heads, num_kvheads
    )
    fallback_num_batch_tiles = _sdpa_num_batch_tiles(batch_size)
    fallback_kv_blocks_per_loop_group = _kv_blocks_per_loop_group(
        fallback_num_q_tiles, fallback_num_kv_blocks
    )
    score_bytes_per_core: int | None = None
    estimated_live_bytes_per_core: int | None = None
    is_decode = max_seqlen_q == 1
    group_extent = num_heads // num_kvheads if num_heads != num_kvheads else 1
    head_extent = num_kvheads if group_extent > 1 else num_heads
    query_output_bytes = (
        2 * batch_size * num_heads * max_seqlen_q * head_dim * element_size
    )
    kv_bytes = 2 * batch_size * num_kvheads * max_seqlen_kv * head_dim * element_size
    mask_axis_extents = (
        (batch_size, head_extent, group_extent, max_seqlen_q, max_seqlen_kv)
        if group_extent > 1
        else (batch_size, head_extent, max_seqlen_q, max_seqlen_kv)
    )

    def estimated_hbm_bytes(
        *,
        num_batch_tiles: int,
        num_head_tiles: int,
        num_group_tiles: int,
        num_q_tiles: int,
        num_kv_blocks: int,
    ) -> int:
        effective_group_tiles = _sdpa_effective_group_tiles(
            num_group_tiles, num_q_tiles, num_kv_blocks
        )
        axis_tile_counts = (
            (
                num_batch_tiles,
                num_head_tiles,
                effective_group_tiles,
                num_q_tiles,
                num_kv_blocks,
            )
            if group_extent > 1
            else (
                num_batch_tiles,
                num_head_tiles,
                num_q_tiles,
                num_kv_blocks,
            )
        )
        return (
            query_output_bytes
            + kv_bytes * effective_group_tiles * num_q_tiles
            + _sdpa_mask_hbm_bytes(
                mask_shapes=mask_shapes,
                axis_extents=mask_axis_extents,
                axis_tile_counts=axis_tile_counts,
                element_size=element_size,
            )
            + (head_tile_staging_bytes if num_head_tiles > 1 else 0)
        )

    if is_decode:
        num_batch_tiles = _sdpa_num_batch_tiles(batch_size)
        batch_tile_size = batch_size // num_batch_tiles
        num_head_tiles = 1
        num_group_tiles = group_extent
        candidates = _sdpa_kv_candidates(
            batch_size=batch_tile_size,
            num_heads=num_heads,
            num_kvheads=num_kvheads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim=head_dim,
            element_size=element_size,
            num_cores=num_cores,
            query_tile_size=1,
            group_tile_size=1,
            num_non_group_outer_tiles=num_batch_tiles,
            num_group_tiles=num_group_tiles,
            num_q_tiles=1,
            full_sdpa_prefill=False,
        )
        feasible_decode = [
            candidate
            for candidate in candidates
            if candidate.estimated_live_bytes_per_core <= lx_budget_bytes
        ]
        if feasible_decode:
            fewest_executions = min(
                feasible_decode,
                key=lambda candidate: (
                    candidate.estimated_dsc_executions,
                    -candidate.block_size,
                ),
            )
            lx_resident = [
                candidate
                for candidate in feasible_decode
                if candidate.restick_lx_eligible
                and candidate.block_size >= 256
                and candidate.num_blocks
                <= max(
                    _SDPA_DECODE_MIN_LX_RESIDENT_BLOCKS,
                    _SDPA_DECODE_MAX_MULTI_BLOCK_EXTENT // candidate.block_size,
                )
            ]
            selected = (
                min(
                    lx_resident,
                    key=lambda candidate: (
                        candidate.estimated_dsc_executions,
                        -candidate.block_size,
                    ),
                )
                if lx_resident
                else fewest_executions
            )
            reason = (
                "single-query decode; LX-resident restickified K"
                if selected.restick_lx_eligible
                else "single-query decode; longest bursts and fewest DSC executes"
            )
            return _SDPATilingConfig(
                strategy="decode" if selected.num_blocks == 1 else "decode_tiled",
                reason=reason,
                kv_block_size=selected.block_size,
                num_kv_blocks=selected.num_blocks,
                num_q_tiles=1,
                q_tile_size=1,
                num_batch_tiles=num_batch_tiles,
                num_head_tiles=num_head_tiles,
                num_group_tiles=num_group_tiles,
                kv_blocks_per_loop_group=_kv_blocks_per_loop_group(
                    1, selected.num_blocks
                ),
                estimated_active_cores=None,
                estimated_load_bursts=selected.estimated_load_bursts,
                estimated_hbm_bytes=estimated_hbm_bytes(
                    num_batch_tiles=num_batch_tiles,
                    num_head_tiles=num_head_tiles,
                    num_group_tiles=num_group_tiles,
                    num_q_tiles=1,
                    num_kv_blocks=selected.num_blocks,
                ),
                estimated_spill_buffers=0,
                estimated_spill_bytes=0,
                score_bytes_per_core=selected.score_bytes_per_core,
                estimated_live_bytes_per_core=selected.estimated_live_bytes_per_core,
                lx_budget_bytes=lx_budget_bytes,
            )
        smallest_candidate = min(
            candidates, key=lambda candidate: candidate.estimated_live_bytes_per_core
        )
    else:
        plans: list[_SDPAPrefillPlan] = []
        for num_batch_tiles in _exact_tile_counts(batch_size):
            batch_tile_size = batch_size // num_batch_tiles
            for num_head_tiles in _exact_tile_counts(head_extent):
                head_tile_size = head_extent // num_head_tiles
                for num_group_tiles in _exact_tile_counts(group_extent):
                    group_tile_size = group_extent // num_group_tiles
                    for query_tile_size in _sdpa_query_tile_sizes(max_seqlen_q):
                        num_q_tiles = max_seqlen_q // query_tile_size
                        num_non_group_outer_tiles = (
                            num_batch_tiles * num_head_tiles * num_q_tiles
                        )
                        candidates = _sdpa_kv_candidates(
                            batch_size=batch_tile_size,
                            num_heads=head_tile_size * group_tile_size,
                            num_kvheads=head_tile_size,
                            max_seqlen_q=query_tile_size,
                            max_seqlen_kv=max_seqlen_kv,
                            head_dim=head_dim,
                            element_size=element_size,
                            num_cores=num_cores,
                            query_tile_size=query_tile_size,
                            group_tile_size=group_tile_size,
                            num_non_group_outer_tiles=num_non_group_outer_tiles,
                            num_group_tiles=num_group_tiles,
                            num_q_tiles=num_q_tiles,
                            full_sdpa_prefill=True,
                        )
                        for candidate in candidates:
                            effective_group_tiles = _sdpa_effective_group_tiles(
                                num_group_tiles,
                                num_q_tiles,
                                candidate.num_blocks,
                            )
                            num_outer_tiles = (
                                num_non_group_outer_tiles * effective_group_tiles
                            )
                            effective_group_tile_size = (
                                group_tile_size
                                * num_group_tiles
                                // effective_group_tiles
                            )
                            live_overflow = max(
                                0,
                                candidate.estimated_live_bytes_per_core
                                - lx_budget_bytes,
                            )
                            # The analytical live count is conservative around
                            # the map/carry handoff by one query-shaped slot.
                            # Use that slot for admission, but charge the larger
                            # score/query buffer that allocation may evict.
                            spill_allowance = candidate.query_bytes_per_core
                            spill_buffer_size = max(
                                candidate.score_bytes_per_core,
                                candidate.query_bytes_per_core,
                            )
                            estimated_spill_buffers = (
                                (live_overflow + spill_allowance - 1) // spill_allowance
                                if live_overflow
                                else 0
                            )
                            # A spilled intermediate is written to HBM and read
                            # back on every innermost loop trip.
                            estimated_spill_bytes = (
                                2
                                * estimated_spill_buffers
                                * spill_buffer_size
                                * num_cores
                                * num_outer_tiles
                                * candidate.num_blocks
                            )
                            plan_hbm_bytes = (
                                estimated_hbm_bytes(
                                    num_batch_tiles=num_batch_tiles,
                                    num_head_tiles=num_head_tiles,
                                    num_group_tiles=num_group_tiles,
                                    num_q_tiles=num_q_tiles,
                                    num_kv_blocks=candidate.num_blocks,
                                )
                                + estimated_spill_bytes
                            )
                            plan_load_bursts = candidate.estimated_load_bursts + (
                                _sdpa_hbm_transfer_waves(
                                    head_tile_staging_bytes, num_cores
                                )
                                if num_head_tiles > 1
                                else 0
                            )
                            plan_hbm_transfer_waves = _sdpa_hbm_transfer_waves(
                                plan_hbm_bytes, num_cores
                            )
                            plans.append(
                                _SDPAPrefillPlan(
                                    num_batch_tiles=num_batch_tiles,
                                    num_head_tiles=num_head_tiles,
                                    num_group_tiles=num_group_tiles,
                                    num_q_tiles=num_q_tiles,
                                    q_tile_size=query_tile_size,
                                    kv=candidate,
                                    estimated_active_cores=(
                                        _sdpa_estimated_active_cores(
                                            (
                                                batch_tile_size,
                                                head_tile_size,
                                                effective_group_tile_size,
                                                query_tile_size,
                                            ),
                                            num_cores,
                                        )
                                    ),
                                    estimated_restick_active_cores=(
                                        candidate.estimated_restick_active_cores
                                    ),
                                    estimated_restick_work_bytes=(
                                        candidate.restick_bytes_per_core
                                        * num_outer_tiles
                                        * candidate.num_blocks
                                    ),
                                    estimated_load_bursts=plan_load_bursts,
                                    estimated_hbm_bytes=plan_hbm_bytes,
                                    estimated_hbm_transfer_waves=(
                                        plan_hbm_transfer_waves
                                    ),
                                    estimated_spill_buffers=(estimated_spill_buffers),
                                    estimated_spill_bytes=estimated_spill_bytes,
                                    estimated_work=(
                                        candidate.estimated_dsc_executions
                                        + plan_load_bursts
                                        + plan_hbm_transfer_waves
                                    ),
                                )
                            )

        bounded_prefill = [
            plan
            for plan in plans
            # Tolerate and price one query/output carry-sized uncertainty at
            # the map boundary. Requiring more means DWSRS did not right-size
            # the working set.
            if plan.estimated_spill_buffers <= 1
        ]
        for plan in plans:
            logger.debug(
                "SDPA tile candidate: B=%s H=%s G=%s Lq=%s K=%s "
                "active_cores=%s restick_cores=%s load_bursts=%s "
                "hbm_bytes=%s hbm_waves=%s spill_buffers=%s spill_bytes=%s "
                "restick_work_bytes=%s dsc_executes=%s estimated_work=%s "
                "score_bytes_per_core=%s live_bytes_per_core=%s feasible=%s",
                plan.num_batch_tiles,
                plan.num_head_tiles,
                plan.num_group_tiles,
                plan.q_tile_size,
                plan.kv.block_size,
                plan.estimated_active_cores,
                plan.estimated_restick_active_cores,
                plan.estimated_load_bursts,
                plan.estimated_hbm_bytes,
                plan.estimated_hbm_transfer_waves,
                plan.estimated_spill_buffers,
                plan.estimated_spill_bytes,
                plan.estimated_restick_work_bytes,
                plan.kv.estimated_dsc_executions,
                plan.estimated_work,
                plan.kv.score_bytes_per_core,
                plan.kv.estimated_live_bytes_per_core,
                plan.estimated_spill_buffers <= 1,
            )
        if bounded_prefill:
            selected_plan = min(
                bounded_prefill,
                key=lambda plan: (
                    plan.estimated_work,
                    plan.estimated_spill_buffers,
                    plan.estimated_load_bursts,
                    plan.estimated_hbm_bytes,
                    plan.kv.estimated_dsc_executions,
                    -plan.estimated_active_cores,
                    plan.estimated_restick_work_bytes,
                    plan.num_batch_tiles,
                    plan.num_head_tiles,
                    -plan.q_tile_size,
                    plan.num_group_tiles,
                    -plan.kv.block_size,
                ),
            )
            selected = selected_plan.kv
            return _SDPATilingConfig(
                strategy=(
                    "work_divided"
                    if not _sdpa_has_loop_boundary(
                        num_non_group_outer_tiles=(
                            selected_plan.num_batch_tiles
                            * selected_plan.num_head_tiles
                            * selected_plan.num_q_tiles
                        ),
                        num_group_tiles=selected_plan.num_group_tiles,
                        num_q_tiles=selected_plan.num_q_tiles,
                        num_kv_blocks=selected.num_blocks,
                    )
                    else "work_divided_tiled"
                ),
                reason=("lowest loop, HBM burst, and bounded-spill transfer cost"),
                kv_block_size=selected.block_size,
                num_kv_blocks=selected.num_blocks,
                num_q_tiles=selected_plan.num_q_tiles,
                q_tile_size=selected_plan.q_tile_size,
                num_batch_tiles=selected_plan.num_batch_tiles,
                num_head_tiles=selected_plan.num_head_tiles,
                num_group_tiles=selected_plan.num_group_tiles,
                kv_blocks_per_loop_group=_kv_blocks_per_loop_group(
                    selected_plan.num_q_tiles, selected.num_blocks
                ),
                estimated_active_cores=selected_plan.estimated_active_cores,
                estimated_load_bursts=selected_plan.estimated_load_bursts,
                estimated_hbm_bytes=selected_plan.estimated_hbm_bytes,
                estimated_spill_buffers=selected_plan.estimated_spill_buffers,
                estimated_spill_bytes=selected_plan.estimated_spill_bytes,
                score_bytes_per_core=selected.score_bytes_per_core,
                estimated_live_bytes_per_core=selected.estimated_live_bytes_per_core,
                lx_budget_bytes=lx_budget_bytes,
            )
        smallest_plan = min(
            plans, key=lambda plan: plan.kv.estimated_live_bytes_per_core
        )
        smallest_candidate = smallest_plan.kv

    score_bytes_per_core = smallest_candidate.score_bytes_per_core
    estimated_live_bytes_per_core = smallest_candidate.estimated_live_bytes_per_core
    reason = "estimated per-core live footprint exceeds the LX budget"

    fallback_effective_group_tiles = _sdpa_effective_group_tiles(
        fallback_num_group_tiles,
        fallback_num_q_tiles,
        fallback_num_kv_blocks,
    )
    return _SDPATilingConfig(
        strategy="coarse_tiled",
        reason=reason,
        kv_block_size=fallback_kv_block_size,
        num_kv_blocks=fallback_num_kv_blocks,
        num_q_tiles=fallback_num_q_tiles,
        q_tile_size=fallback_q_tile_size,
        num_batch_tiles=fallback_num_batch_tiles,
        num_head_tiles=fallback_num_head_tiles,
        num_group_tiles=fallback_num_group_tiles,
        kv_blocks_per_loop_group=fallback_kv_blocks_per_loop_group,
        estimated_active_cores=None,
        estimated_load_bursts=(
            2
            * fallback_num_batch_tiles
            * fallback_num_head_tiles
            * fallback_effective_group_tiles
            * fallback_num_q_tiles
            * fallback_num_kv_blocks
            + (
                _sdpa_hbm_transfer_waves(head_tile_staging_bytes, num_cores)
                if fallback_num_head_tiles > 1
                else 0
            )
        ),
        estimated_hbm_bytes=estimated_hbm_bytes(
            num_batch_tiles=fallback_num_batch_tiles,
            num_head_tiles=fallback_num_head_tiles,
            num_group_tiles=fallback_num_group_tiles,
            num_q_tiles=fallback_num_q_tiles,
            num_kv_blocks=fallback_num_kv_blocks,
        ),
        estimated_spill_buffers=None,
        estimated_spill_bytes=None,
        score_bytes_per_core=score_bytes_per_core,
        estimated_live_bytes_per_core=estimated_live_bytes_per_core,
        lx_budget_bytes=lx_budget_bytes,
    )


def _select_swa_tiling(
    *,
    batch_size: int,
    num_heads: int,
    num_kvheads: int,
    q_block: int,
    buffer_width: int,
    head_dim: int,
    element_size: int,
    num_cores: int,
    lx_budget_bytes: int,
) -> _SWATilingConfig:
    """Choose SWA tiling from the same compiler-visible costs as full SDPA.

    SWA presents one static physical K/V window to each query block, so its
    block selection has the same score-buffer, streaming, restickification,
    and DPO execution tradeoffs as SDPA.  Only the dimension names differ:
    SDPA's ``max_seqlen_q`` and ``max_seqlen_kv`` become ``q_block`` and
    ``kv_block`` in the SWA decomposition.

    Unlike full SDPA, SWA pads its complete K/V scan so every repeated BMM tile
    has a 64-row physical extent. Candidate ceilings are therefore normalized
    to equal aligned tiles that cover the logical buffer before their physical
    costs are evaluated.
    """
    fallback_num_kv_blocks, fallback_kv_block_size = _padded_tiling_for_max_extent(
        buffer_width,
        _SDPA_MAX_SEQUENCE_TILE_SIZE,
        tile_alignment=_SDPA_SEQUENCE_TILE_ALIGNMENT,
    )
    fallback_num_head_tiles = (
        1 if num_heads != num_kvheads else _sdpa_num_head_tiles(num_heads)
    )
    score_bytes_per_core: int | None = None
    estimated_live_bytes_per_core: int | None = None
    kv_bytes_per_core: int | None = None
    restick_bytes_per_core: int | None = None
    restick_lx_eligible: bool | None = None
    estimated_dsc_executions: int | None = None
    is_decode = q_block == 1
    work_div_num_cores = num_cores
    if num_heads != num_kvheads:
        work_div_num_cores = min(
            num_cores, max(1, q_block // _SWA_MIN_QUERY_ROWS_PER_CORE)
        )
    sdpa_work_div = (
        None
        if is_decode
        else _sdpa_work_division(
            num_heads,
            num_kvheads,
            q_block,
            buffer_width,
            work_div_num_cores,
        )
    )

    reason = "compiler-cost candidate available"
    if q_block > _SDPA_MAX_SEQUENCE_TILE_SIZE:
        reason = "query block exceeds the calibrated work-divided limit"
    elif not is_decode and sdpa_work_div is None:
        reason = "no exact head/query-block work division"
    else:
        candidates = _sdpa_kv_candidates(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kvheads=num_kvheads,
            max_seqlen_q=q_block,
            max_seqlen_kv=buffer_width,
            head_dim=head_dim,
            element_size=element_size,
            num_cores=num_cores,
            work_div=sdpa_work_div,
            pad_extent=True,
        )
        for candidate in candidates:
            logger.debug(
                "SWA KV candidate: K=%s blocks=%s estimated_dsc_executes=%s "
                "score_bytes_per_core=%s live_bytes_per_core=%s "
                "kv_bytes_per_core=%s restick_bytes_per_core=%s "
                "restick_lx_eligible=%s feasible=%s",
                candidate.block_size,
                candidate.num_blocks,
                candidate.estimated_dsc_executions,
                candidate.score_bytes_per_core,
                candidate.estimated_live_bytes_per_core,
                candidate.kv_bytes_per_core,
                candidate.restick_bytes_per_core,
                candidate.restick_lx_eligible,
                candidate.estimated_live_bytes_per_core <= lx_budget_bytes,
            )
        feasible = [
            candidate
            for candidate in candidates
            if candidate.estimated_live_bytes_per_core <= lx_budget_bytes
        ]
        if not feasible:
            smallest = candidates[0]
            score_bytes_per_core = smallest.score_bytes_per_core
            estimated_live_bytes_per_core = smallest.estimated_live_bytes_per_core
            kv_bytes_per_core = smallest.kv_bytes_per_core
            restick_bytes_per_core = smallest.restick_bytes_per_core
            restick_lx_eligible = smallest.restick_lx_eligible
            estimated_dsc_executions = smallest.estimated_dsc_executions
            reason = "estimated per-core live footprint exceeds the LX budget"
        else:
            fewest_executions = min(
                feasible,
                key=lambda candidate: (
                    candidate.estimated_dsc_executions,
                    -candidate.block_size,
                ),
            )
            if is_decode:
                lx_resident = [
                    candidate
                    for candidate in feasible
                    if candidate.restick_lx_eligible
                    and candidate.block_size >= 256
                    and candidate.num_blocks
                    <= max(
                        _SDPA_DECODE_MIN_LX_RESIDENT_BLOCKS,
                        _SDPA_DECODE_MAX_MULTI_BLOCK_EXTENT // candidate.block_size,
                    )
                ]
                selected = (
                    min(
                        lx_resident,
                        key=lambda candidate: (
                            candidate.estimated_dsc_executions,
                            -candidate.block_size,
                        ),
                    )
                    if lx_resident
                    else fewest_executions
                )
                reason = (
                    "single-query decode; LX-resident restickified K"
                    if selected.restick_lx_eligible
                    else "single-query decode; longest bursts and fewest DSC executes"
                )
            else:
                # Prefill starts from the generated candidates, keeps only
                # LX-feasible tiles, then selects the one closest to the
                # analytical K/V streaming target.
                assert sdpa_work_div is not None
                query_rows_per_core = q_block // sdpa_work_div["max_seqlen_q"]
                target_kv_bytes = _sdpa_target_kv_bytes_per_core(
                    num_heads,
                    num_kvheads,
                    query_rows_per_core,
                )
                target_kv_block_size = _sdpa_burst_efficient_kv_block_limit(
                    query_rows_per_core
                )
                kv_bytes_per_row = (
                    candidates[0].kv_bytes_per_core // candidates[0].block_size
                )
                target_kv_extent = min(
                    target_kv_block_size,
                    max(1, target_kv_bytes // kv_bytes_per_row),
                )
                # Equal HOP tiles can leave large gaps around the analytical
                # target.  Choose the closest physically runnable divisor,
                # rather than always rounding down and accidentally halving
                # the useful burst.  LX remains a hard feasibility bound;
                # execution count and burst length break equidistant ties.
                selected = min(
                    feasible,
                    key=lambda candidate: (
                        abs(candidate.block_size - target_kv_extent),
                        candidate.estimated_dsc_executions,
                        -candidate.block_size,
                    ),
                )
                reason = "closest exact tile to K/V streaming and burst targets"

            selected_work_div = (
                dict(sdpa_work_div) if sdpa_work_div is not None else None
            )
            # The HOP loop already owns the K/V traversal and its online
            # softmax reduction. Unlike full SDPA's statically unrolled
            # blocks, splitting that reduction again is neither legal nor
            # useful; preserve only the independent head/query splits.
            swa_work_div_names = {"max_seqlen_q": "q_block"}
            swa_work_div = (
                {
                    swa_work_div_names.get(name, name): split
                    for name, split in selected_work_div.items()
                }
                if selected_work_div is not None
                else None
            )
            strategy = "decode" if is_decode else "work_divided"
            if selected.num_blocks > 1:
                strategy += "_tiled"
            return _SWATilingConfig(
                strategy=strategy,
                reason=reason,
                kv_block_size=selected.block_size,
                num_kv_blocks=selected.num_blocks,
                num_head_tiles=1,
                work_div=swa_work_div,
                score_bytes_per_core=selected.score_bytes_per_core,
                estimated_live_bytes_per_core=selected.estimated_live_bytes_per_core,
                kv_bytes_per_core=selected.kv_bytes_per_core,
                restick_bytes_per_core=selected.restick_bytes_per_core,
                restick_lx_eligible=selected.restick_lx_eligible,
                estimated_dsc_executions=selected.estimated_dsc_executions,
                lx_budget_bytes=lx_budget_bytes,
            )

    if reason == "compiler-cost candidate available":
        raise AssertionError("SWA candidate selection did not produce a result")

    return _SWATilingConfig(
        strategy=("fallback" if fallback_num_kv_blocks == 1 else "fallback_tiled"),
        reason=reason,
        kv_block_size=fallback_kv_block_size,
        num_kv_blocks=fallback_num_kv_blocks,
        num_head_tiles=fallback_num_head_tiles,
        work_div=None,
        score_bytes_per_core=score_bytes_per_core,
        estimated_live_bytes_per_core=estimated_live_bytes_per_core,
        kv_bytes_per_core=kv_bytes_per_core,
        restick_bytes_per_core=restick_bytes_per_core,
        restick_lx_eligible=restick_lx_eligible,
        estimated_dsc_executions=estimated_dsc_executions,
        lx_budget_bytes=lx_budget_bytes,
    )


# Determine the float dtype for bool at module load time (not during tracing)
_BOOL_FLOAT_DTYPE = None


def _get_float_dtype_for_bool() -> torch.dtype:
    """
    Get the appropriate float dtype to convert boolean tensors on Spyre.
    Boolean tensors are stored as either FP16 or FP32 on the device.
    This is determined once at module load time to avoid tracing issues.
    """
    global _BOOL_FLOAT_DTYPE
    if _BOOL_FLOAT_DTYPE is None:
        device_dtype = get_device_dtype(torch.bool)
        # Map DataFormats to torch.dtype, defaulting to float16
        if device_dtype == DataFormats.IEEE_FP32:
            _BOOL_FLOAT_DTYPE = torch.float32
        else:
            _BOOL_FLOAT_DTYPE = torch.float16
    return _BOOL_FLOAT_DTYPE


# A module-level lock to make the CM thread-safe
_decompositions_lock = threading.RLock()

# Spyre-specific decompositions, populated by ``@register_spyre_decompositions``.
spyre_decompositions: dict = {}

# Inductor default decompositions to drop on Spyre. They produce code the
# backend cannot lower today; falling through to the CPU fallback is preferable
# until those issues are fixed.
spyre_decompositions_to_exclude = [
    torch.ops.aten.tril,
    # PT 2.12 broadened torch._inductor.decomposition.mm/bmm to decompose the
    # K==1 (unit-contraction) case into a broadcast ``self * other`` on all
    # non-cpu/mps devices. That AOT decomposition runs before Spyre's
    # ``mm_to_bmm_pass`` and produces a flatten-mul-unflatten shape whose
    # trailing view yields an unsupported ``floor(d0/N)`` stick expression.
    # Spyre has its own aten.mm / aten.bmm lowerings, so drop the upstream
    # decomps and let the mm/bmm survive to mm_to_bmm_pass (2.11 behavior).
    torch.ops.aten.mm,
    torch.ops.aten.bmm,
]

OpOrOps = Union[torch._ops.OperatorBase, Sequence[torch._ops.OperatorBase]]

# Module-level Library handles, kept alive for the lifetime of the process.
# ``torch.library.Library`` uses ``weakref.finalize`` to call ``m.reset()`` on
# GC, which would silently unregister every kernel from the C++ dispatcher.
_spyre_autograd_lib = None
_spyre_lib = None
_dispatchkey_kernels_registered = False


def register_spyre_decompositions(ops: OpOrOps):
    """Register a Spyre-specific decomposition for one or more operators.

    The function is added to the Spyre decomposition table; Inductor reads it
    via ``get_decomp_fn`` during ``torch.compile`` / ``make_fx``. For aten ops,
    ``_register_spyre_dispatchkey_kernels_permanently`` additionally installs a
    PrivateUse1 kernel pointing at the same function at runtime init, so
    eager-mode dispatch reaches it too. This is required for
    ``CompositeImplicitAutograd`` ops (``rms_norm``, ``layer_norm``, ...); it
    is harmless for the rest.
    """
    return decomp.register_decomposition(ops, spyre_decompositions)


def get_spyre_decomp_table() -> dict[Any, Callable[..., Any]]:
    """Return the decomposition table Inductor sees when compiling for Spyre.

    Builds a fresh dict on each call from ``select_decomp_table()`` (Inductor's
    default, itself cached upstream) plus Spyre additions and exclusions.
    Independent from ``torch._inductor.decomposition.decompositions`` — Spyre
    never mutates the global registry.
    """
    from torch._inductor.decomposition import select_decomp_table
    from torch._ops import OpOverload, OpOverloadPacket
    from torch_spyre.ops.fallbacks import fallback_ops

    table = dict(select_decomp_table())

    def _drop(op):
        if isinstance(op, OpOverloadPacket):
            for overload_name in op.overloads():
                table.pop(getattr(op, overload_name), None)
        elif isinstance(op, OpOverload):
            table.pop(op, None)

    for op in spyre_decompositions_to_exclude:
        _drop(op)
    for op in fallback_ops:
        _drop(op)
    table.update(spyre_decompositions)
    return table


class _OPWrapper:
    """PrivateUse1 kernel that lazily ``torch.compile``-s a Spyre decomposition.

    The first eager call compiles the decomposition (with ``dynamic=False``);
    subsequent eager calls reuse the compiled entry point. When invoked from
    inside an active ``torch.compile`` context, the wrapped function is called
    directly — re-entering ``torch.compile`` would be wrong.
    """

    def __init__(self, fn):
        self._fn = fn
        self._compiled_fn = None

    def __call__(self, *args, **kwargs):
        from torch.utils import _pytree as pytree

        leaves = pytree.tree_leaves(args) + pytree.tree_leaves(kwargs)
        # ``!=`` (not ``is not``) is deliberate: this compares the device *type*
        # string against the ``DEVICE_NAME`` constant, and string equality is a
        # value comparison. ``getattr(..., None)`` yields ``None`` for
        # non-tensors, but the ``isinstance`` guard short-circuits those first.
        if any(
            isinstance(x, torch.Tensor)
            and getattr(x.device, "type", None) != DEVICE_NAME
            for x in leaves
        ):
            devs = [x.device if isinstance(x, torch.Tensor) else None for x in leaves]
            raise RuntimeError(
                f"Spyre decomposition function called with inputs on a different "
                f"device! Args devices: {devs=}"
            )
        if torch.compiler.is_compiling():
            return self._fn(*args, **kwargs)
        if self._compiled_fn is None:
            self._compiled_fn = torch.compile(self._fn, dynamic=False)
        # ``scan`` lowers to a ``while_loop`` whose traced body extracts its
        # scalar induction variable with ``item()``.  Eager PrivateUse1
        # dispatch reaches that rewrite through this lazy compilation path,
        # outside Dynamo's usual scalar-capture setup.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            return self._compiled_fn(*args, **kwargs)


def _register_spyre_dispatchkey_kernels_permanently():
    """Install PrivateUse1 / AutogradPrivateUse1 kernels for every aten op
    that has a Spyre decomposition and no pre-existing PrivateUse1 kernel.

    Idempotent; called from ``_SpyreImpl._lazy_init`` after eager ops and
    custom ops have been imported, so the existing-kernel check sees the final
    set of registered backends.
    """
    global _spyre_autograd_lib, _spyre_lib, _dispatchkey_kernels_registered

    if _dispatchkey_kernels_registered:
        return

    from torch.library import Library, fallthrough_kernel

    _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    _spyre_lib = Library("aten", "IMPL", "PrivateUse1")
    has_pu1 = torch._C._dispatch_has_kernel_for_dispatch_key

    for op, fn in spyre_decompositions.items():
        if op.namespace != "aten" or has_pu1(op._name, "PrivateUse1"):
            continue
        # Autograd key: fall through so PrivateUse1 is reached.
        _spyre_autograd_lib.impl(op._name, fallthrough_kernel)
        # PrivateUse1 key: dispatch into a lazy-compile wrapper.
        _spyre_lib.impl(op._name, _OPWrapper(fn))

    _dispatchkey_kernels_registered = True


###############################################################################
##                       Spyre decompositions                                ##
###############################################################################


@register_spyre_decompositions([torch.ops.aten.ones.default])
def ones_decomp(
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.aten.full(size, 1, dtype=dtype, layout=layout, device=device)


@register_spyre_decompositions([torch.ops.aten.new_ones.default])
def new_ones_decomp(
    self: torch.Tensor,
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.aten.full(
        size,
        1,
        dtype=dtype if dtype is not None else self.dtype,
        layout=layout,
        device=device if device is not None else self.device,
    )


@register_spyre_decompositions([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


@register_spyre_decompositions([torch.ops.aten.sign.default])
def spyre_sign(input: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(input)
    return torch.where(
        torch.gt(input, zero),
        torch.ones_like(input),
        torch.where(torch.lt(input, zero), -torch.ones_like(input), zero),
    )


###############################################################################
##                    Spyre decompositions for aten ops                      ##
###############################################################################
# For aten ops, ``register_spyre_decompositions`` automatically installs a
# PrivateUse1 dispatch kernel as well (essential for CIA ops like rms_norm,
# layer_norm; harmless for the rest).
@register_spyre_decompositions([torch.ops.aten.rms_norm.default])
def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_rms_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )

    mean = torch.mean(input * input, dim=-1, keepdim=True)
    rsqrt_inp = torch.rsqrt(mean + eps)
    output = input * rsqrt_inp
    if weight is not None:
        output = output * weight
    return output


@register_spyre_decompositions([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_layer_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )
    # F.layer_norm treats weight=None as identity and bias=None as zero;
    # spyre.layernormnorm doesn't handle missing args, so substitute defaults.
    if weight is None:
        weight = input.new_ones(normalized_shape)
    if bias is None:
        bias = input.new_zeros(normalized_shape)
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decompositions([torch.ops.aten.silu.default])
def silu(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.silu(input)


@register_spyre_decompositions([torch.ops.aten.topk])
def spyre_topk(
    input: torch.Tensor,
    k: int,
    dim: Optional[int] = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k > 128:
        raise Unsupported(f"topk with k={k} is not supported (max k=128)")
    if not largest:
        raise Unsupported("topk with largest=False")
    # sorted=False is a no-op: our reduction always returns sorted output.
    # Index stays in the input dtype (not int64) all the way out; topkindex's
    # fake reports it so Dynamo traces it with no meta conflict.
    return torch.ops.spyre.topkvalue(input, k, dim), torch.ops.spyre.topkindex(
        input, k, dim
    )


@register_spyre_decompositions([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decompositions([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    if beta == 1.0:
        return torch.ops.spyre.softplus(input, beta, threshold)
    # The runtime primitive drops the outer 1/beta factor, so beta == 1 is its
    # only exact path. Scale into it and back out; the threshold branch stays
    # exact because 1 * (beta * x) > threshold is PyTorch's beta * x > threshold.
    # aten accepts beta == 0 and saturates every element to +-inf, so take the
    # reciprocal under IEEE rules rather than letting Python raise here.
    inv_beta = math.copysign(math.inf, beta) if beta == 0.0 else 1.0 / beta
    return torch.ops.spyre.softplus(input * beta, 1.0, threshold) * inv_beta


def _pow_by_squaring(input: torch.Tensor, exponent: int) -> torch.Tensor:
    """``input ** exponent`` for ``exponent >= 1`` as a chain of ``mul`` ops.

    Binary square-and-multiply, so ~2*log2(n) multiplies. That is optimal for
    every exponent below 15 and never more than one multiply above optimal
    through at least n=40, so the addition-chain search that would close the
    gap is not worth the table it needs.
    """
    result = None
    square = input
    while exponent:
        if exponent & 1:
            result = square if result is None else torch.mul(result, square)
        exponent >>= 1
        if exponent:
            square = torch.mul(square, square)
    return result


@register_spyre_decompositions([torch.ops.aten.pow.Tensor_Scalar])
def spyre_pow_tensor_scalar(
    input: torch.Tensor, exponent: Union[int, float]
) -> torch.Tensor:
    """``pow`` with a scalar exponent: an exact primitive where one exists, a
    multiply chain for any other integer, and ``exp(n * log(x))`` otherwise.

    Limitation: a negative base with a non-integer exponent returns a finite
    garbage value where CPU returns NaN.
    """
    if isinstance(exponent, bool):
        exponent = int(exponent)
    if not input.dtype.is_floating_point:
        # TODO: support integer bases; needs aten's promotion rules plus device
        # support for integer multiply chains.
        raise Unsupported(f"pow with a non-floating-point base: {input.dtype}")

    # int, float, or SymFloat depending on the trace, and not stable across runs.
    e = float(exponent)
    if e == 1.0:
        # Returning the input rather than a copy is legal because a
        # decomposition runs on a functionalized graph; callers must not rely on
        # either identity, since aten's contract is only that the value matches.
        return input
    if e == -1.0:
        return torch.reciprocal(input)
    if e == 0.5:
        return torch.sqrt(input)
    if e == -0.5:
        return torch.rsqrt(input)
    if e == 0.0:
        return torch.ones_like(input)
    if e.is_integer():
        magnitude = _pow_by_squaring(input, abs(int(e)))
        return torch.reciprocal(magnitude) if e < 0 else magnitude
    return torch.exp(e * torch.log(input))


@register_spyre_decompositions([torch.ops.aten.linear.default])
def spyre_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    weight = weight.transpose(-1, -2)
    while weight.dim() < input.dim():
        weight = torch.unsqueeze(weight, 0)
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


@register_spyre_decompositions(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default]
)
def spyre__sdpa_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = query.size(0)
    num_heads = query.size(1)
    num_kvheads = key.size(1)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)
    head_dim = query.size(3)

    query_scale = scale
    if query_scale is None:
        query_scale = 1.0 / math.sqrt(head_dim)

    if dropout_p > 0.0:
        raise Unsupported("Attention dropout not implemented for Spyre")

    # SDPA routinely receives logical [B, H, S, D] queries backed by physical
    # [B, S, H, D] storage. for_each_tile's explicit dims already preserve the
    # logical axes, but WhileLoop.create still requires exact input strides and
    # would otherwise synthesize this normalization. A projection result can be
    # logically contiguous while retaining a noncanonical Spyre device layout,
    # so ``contiguous()`` is not sufficient to materialize the normalization.
    # Keep the bounded, one-time query clone explicit before GQA unflattening.
    # Do not clone K/V wholesale: those copies scale with context length, so the
    # compiler streams their bounded tiles instead.
    original_query_strides = query.stride()
    if num_heads % num_kvheads != 0:
        raise Unsupported(
            "GQA requires the number of KV heads to divide the number of query "
            f"heads, got query={num_heads} and key/value={num_kvheads}"
        )
    gqa_group_size = num_heads // num_kvheads
    use_gqa = gqa_group_size != 1

    # Keep the GQA relationship explicit instead of repeating K/V up to Hq.
    # K/V receive only a unit view axis and broadcast over the within-group
    # query-head dimension in the native batched-matmul lowering. Preserve the
    # established rank-4 MHA path when there is no grouped-query expansion.
    query = (
        query.clone(memory_format=torch.contiguous_format)
        if use_gqa
        else query.contiguous()
    )
    if use_gqa:
        query = query.unflatten(1, (num_kvheads, gqa_group_size))

    # Precompute the causal additive mask once before entering the tiled loops.
    # Shape [1, 1, max_seqlen_q, max_seqlen_kv]: 0.0 = keep, -inf = masked.
    #
    # spyre::causal_mask builds the mask on CPU (tril + masked_fill_) and
    # transfers it to the query device. Wrapping this in a custom op makes the
    # CPU-side in-place ops opaque to torch.compile, so assert_functional_graph
    # is satisfied and the compiled graph sees only the resulting Spyre tensor.
    if is_causal:
        causal_mask = torch.ops.spyre.causal_mask(
            max_seqlen_q, max_seqlen_kv, query.dtype, query.device
        )
        if use_gqa:
            causal_mask = causal_mask.unsqueeze(2)

    if use_gqa and attn_bias is not None:
        if attn_bias.dim() == 2:
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif attn_bias.dim() == 3:
            attn_bias = attn_bias.unsqueeze(1).unsqueeze(2)
        elif attn_bias.dim() == 4:
            bias_heads = attn_bias.size(1)
            if bias_heads == num_heads:
                attn_bias = attn_bias.unflatten(1, (num_kvheads, gqa_group_size))
            elif bias_heads in (1, num_kvheads):
                attn_bias = attn_bias.unsqueeze(2)
            else:
                raise Unsupported(
                    "GQA attention bias head dimension must be 1, Hkv, or Hq; "
                    f"got {bias_heads} for Hkv={num_kvheads}, Hq={num_heads}"
                )
        elif attn_bias.dim() != 5:
            raise Unsupported(
                f"GQA attention bias must have rank 2-5, got rank {attn_bias.dim()}"
            )
    elif attn_bias is not None:
        while attn_bias.dim() < 4:
            attn_bias = attn_bias.unsqueeze(0)
        if attn_bias.dim() != 4:
            raise Unsupported(
                f"MHA attention bias must have rank 2-4, got rank {attn_bias.dim()}"
            )

    masks = tuple(
        mask
        for mask in (
            causal_mask if is_causal else None,
            attn_bias,
        )
        if mask is not None
    )

    # Tiling an interleaved H axis requires map inputs/outputs to be staged in
    # HBM. Price both sides of each staging copy. Query itself is normalized
    # before the loops, but the result must return in its original layout.
    head_tile_staging_bytes = 0
    for tensor in (key, value):
        if not _axis_slice_is_dense(tuple(tensor.shape), tuple(tensor.stride()), 1):
            head_tile_staging_bytes += 2 * tensor.numel() * tensor.element_size()
    if not _axis_slice_is_dense(
        (batch_size, num_heads, max_seqlen_q, head_dim), original_query_strides, 1
    ):
        head_tile_staging_bytes += 2 * query.numel() * query.element_size()

    tiling = _select_sdpa_tiling(
        batch_size=batch_size,
        num_heads=num_heads,
        num_kvheads=num_kvheads,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim=head_dim,
        element_size=query.dtype.itemsize,
        num_cores=config.sencores,
        lx_budget_bytes=_sdpa_lx_budget_bytes(),
        mask_shapes=tuple(tuple(mask.shape) for mask in masks),
        head_tile_staging_bytes=head_tile_staging_bytes,
    )
    # for_each_tile requires equal-sized tiles. The cost model already returns
    # an exact, stick-aligned block; retain the calculation here as a safety net
    # for configurations supplied by fallback or test overrides.
    num_kv_tiles = _num_tiles_for_max_extent(
        max_seqlen_kv,
        tiling.kv_block_size,
        tile_alignment=_SDPA_SEQUENCE_TILE_ALIGNMENT,
    )
    kv_tile_size = max_seqlen_kv // num_kv_tiles
    num_group_tiles = _sdpa_effective_group_tiles(
        tiling.num_group_tiles, tiling.num_q_tiles, num_kv_tiles
    )
    direct_plan = not _sdpa_has_loop_boundary(
        num_non_group_outer_tiles=(
            tiling.num_batch_tiles * tiling.num_head_tiles * tiling.num_q_tiles
        ),
        num_group_tiles=tiling.num_group_tiles,
        num_q_tiles=tiling.num_q_tiles,
        num_kv_blocks=num_kv_tiles,
    )
    logger.debug(
        "SDPA tiling: strategy=%s reason=%s Lq=%s q_tiles=%s "
        "q_tile_size=%s Lk=%s kv_blocks=%s kv_block_size=%s "
        "batch_tiles=%s head_tiles=%s group_tiles=%s kv_blocks_per_loop_group=%s "
        "estimated_active_cores=%s estimated_load_bursts=%s "
        "estimated_hbm_bytes=%s estimated_spill_buffers=%s "
        "estimated_spill_bytes=%s score_bytes_per_core=%s "
        "estimated_live_bytes_per_core=%s lx_budget_bytes=%s",
        tiling.strategy,
        tiling.reason,
        max_seqlen_q,
        tiling.num_q_tiles,
        tiling.q_tile_size,
        max_seqlen_kv,
        tiling.num_kv_blocks,
        tiling.kv_block_size,
        tiling.num_batch_tiles,
        tiling.num_head_tiles,
        tiling.num_group_tiles,
        tiling.kv_blocks_per_loop_group,
        tiling.estimated_active_cores,
        tiling.estimated_load_bursts,
        tiling.estimated_hbm_bytes,
        tiling.estimated_spill_buffers,
        tiling.estimated_spill_bytes,
        tiling.score_bytes_per_core,
        tiling.estimated_live_bytes_per_core,
        tiling.lx_budget_bytes,
    )

    packed_key_strides = (
        max_seqlen_kv * num_kvheads * head_dim,
        head_dim,
        num_kvheads * head_dim,
        1,
    )
    rebase_unaligned_packed_key = (
        batch_size > 1
        and key.stride() == packed_key_strides
        and kv_tile_size % get_elem_in_stick(key.dtype) != 0
    )

    def mask_dims(axis, extent):
        return tuple(axis if mask.size(axis) == extent else None for mask in masks)

    def map_tiles(body, operands, dims, tile_size, out_dim):
        sliced_operand, sliced_dim = next(
            (operand, dim) for operand, dim in zip(operands, dims) if dim is not None
        )
        if sliced_operand.size(sliced_dim) == tile_size:
            _, result = body(None, operands)
            return result
        _, result = for_each_tile(
            body,
            operands,
            dims=dims,
            tile_size=tile_size,
            out_dim=out_dim,
        )
        return result

    def kv_level(q_tile, k_tile, v_tile, *mask_tiles):
        # Q is invariant across the counted Lk loop.
        q_scaled = q_tile * query_scale

        # A loop-free plan has no online-softmax state to carry. Keep
        # its graph as the stable-softmax formula so the compiler sees the same
        # short live ranges that the cost model charges above. Tiled plans retain
        # the common carry path below.
        if direct_plan:
            if use_gqa:
                k_tile = k_tile.unsqueeze(2)
                v_tile = v_tile.unsqueeze(2)
            if rebase_unaligned_packed_key:
                k_tile = k_tile.contiguous()
            keys_t = k_tile.transpose(-1, -2).contiguous()
            scores = (
                torch.ops.spyre.batched_matmul(q_scaled, keys_t)
                if use_gqa
                else torch.matmul(q_scaled, keys_t)
            )
            for mask_tile in mask_tiles:
                scores = scores + mask_tile
            block_max = torch.amax(scores, dim=-1)
            exp_scores = torch.exp(scores - block_max.unsqueeze(-1)).contiguous()
            denominator = exp_scores.sum(dim=-1)
            output_tile = (
                torch.ops.spyre.batched_matmul(exp_scores, v_tile)
                if use_gqa
                else torch.matmul(exp_scores, v_tile)
            )
            return output_tile / denominator.unsqueeze(-1)

        # Keep the sparse accumulator representation used by production SDPA,
        # but size it to the current outer-loop tile.
        tile_accumulator_shape = (*q_tile.shape[:-1], 64)
        running_max_reduced = torch.full(
            tile_accumulator_shape,
            float("-inf"),
            device=q_tile.device,
            dtype=q_tile.dtype,
        )
        running_max = running_max_reduced.amax(dim=-1)
        denominator_reduced = torch.zeros(
            tile_accumulator_shape,
            device=q_tile.device,
            dtype=q_tile.dtype,
        )
        denominator = denominator_reduced.amax(dim=-1)
        output_tile = torch.zeros_like(q_tile)

        def sdpa_lk_body(carry, tiles):
            block_maximum, block_denominator, block_output = carry
            _, k_blk, v_blk, *block_masks = tiles
            if use_gqa:
                k_blk = k_blk.unsqueeze(2)
                v_blk = v_blk.unsqueeze(2)

            # A packed [B*S, H, D] input viewed as [B, H, S, D] shares one
            # physical row dimension between B and S. Rebase an unaligned S
            # tile before the transpose restickify so one batch's padded tail
            # cannot read the next batch's first row.
            if rebase_unaligned_packed_key:
                k_blk = k_blk.contiguous()
            keys_T = k_blk.transpose(-1, -2).contiguous()
            scores = (
                torch.ops.spyre.batched_matmul(q_scaled, keys_T)
                if use_gqa
                else torch.matmul(q_scaled, keys_T)
            )
            for block_mask in block_masks:
                scores = scores + block_mask

            block_max = torch.amax(scores, dim=-1)
            # Compute the old-max correction before producing new_max. The loop
            # lowering can then update running_max in place without snapshotting
            # its old value for a later reader.
            correction = torch.exp(torch.clamp_max(block_maximum - block_max, 0.0))
            new_max = torch.maximum(block_maximum, block_max)
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            new_denominator = block_denominator * correction + exp_scores.sum(dim=-1)
            exp_scores_c = exp_scores.contiguous()
            weighted = (
                torch.ops.spyre.batched_matmul(exp_scores_c, v_blk)
                if use_gqa
                else torch.matmul(exp_scores_c, v_blk)
            )
            new_output = block_output * correction.unsqueeze(-1) + weighted
            return (new_max, new_denominator, new_output), None

        kv_operands = (q_tile, k_tile, v_tile, *mask_tiles)
        kv_dims = (None, -2, -2, *mask_dims(-1, max_seqlen_kv))
        initial = (running_max, denominator, output_tile)
        if num_kv_tiles == 1:
            (_, denominator, output_tile), _ = sdpa_lk_body(initial, kv_operands)
        else:
            (_, denominator, output_tile), _ = for_each_tile(
                sdpa_lk_body,
                kv_operands,
                dims=kv_dims,
                tile_size=kv_tile_size,
                init=initial,
            )
        return output_tile / denominator.unsqueeze(-1)

    def query_level(q_tile, k_tile, v_tile, *mask_tiles):
        operands = (q_tile, k_tile, v_tile, *mask_tiles)
        dims = (-2, None, None, *mask_dims(-2, max_seqlen_q))

        def body(_, tiles):
            return None, kv_level(*tiles)

        return map_tiles(body, operands, dims, tiling.q_tile_size, -2)

    def group_level(q_tile, k_tile, v_tile, *mask_tiles):
        if not use_gqa:
            return query_level(q_tile, k_tile, v_tile, *mask_tiles)
        operands = (q_tile, k_tile, v_tile, *mask_tiles)
        dims = (2, None, None, *mask_dims(2, gqa_group_size))

        def body(_, tiles):
            return None, query_level(*tiles)

        # A G-only loop does not reduce the sequence working set: when both
        # sequence axes already fit in one tile, leave G visible to the normal
        # work-division pass and avoid paying a serial map invocation per
        # query-head group.  Nested prefill/decode plans still use the selected
        # G split whenever either Lq or Lk is tiled.
        group_tile_size = gqa_group_size // max(1, num_group_tiles)
        return map_tiles(body, operands, dims, group_tile_size, 2)

    def head_level(q_tile, k_tile, v_tile, *mask_tiles):
        head_extent = num_kvheads if use_gqa else num_heads
        head_tile_size = head_extent // max(1, tiling.num_head_tiles)
        operands = (q_tile, k_tile, v_tile, *mask_tiles)
        dims = (1, 1, 1, *mask_dims(1, head_extent))

        def body(_, tiles):
            return None, group_level(*tiles)

        return map_tiles(body, operands, dims, head_tile_size, 1)

    batch_tile_count = tiling.num_batch_tiles
    batch_tile_size = batch_size // batch_tile_count
    operands = (query, key, value, *masks)
    dims = (0, 0, 0, *mask_dims(0, batch_size))

    def batch_body(_, tiles):
        return None, head_level(*tiles)

    output = map_tiles(batch_body, operands, dims, batch_tile_size, 0)
    if use_gqa:
        output = output.flatten(1, 2)
    # The reference meta kernel for this op
    # (torch._meta_registrations.meta__scaled_dot_product_fused_attention_
    # overrideable -> alloc_with_matching_layout) declares the output layout to
    # MATCH THE QUERY's layout, not a fixed [B, S, H, D]-contiguous physical
    # layout. Inductor emits assert_size_stride against those meta strides, so the
    # decomp output must carry the same strides as ``query``. For a contiguous
    # query this is plain [B, H, S, D]-contiguous; for a transposed query
    # (physical [B, S, H, D]) it is the swapped-dim layout. Reproduce the meta's
    # dim-order permutation so both cases match exactly.
    dim_order = sorted(
        range(output.dim()), key=lambda i: original_query_strides[i], reverse=True
    )
    permuted = output.permute(dim_order).contiguous()
    inverse_permute = [dim_order.index(i) for i in range(len(dim_order))]
    output = permuted.permute(inverse_permute)
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    return (
        output,
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


@register_spyre_decompositions([torch.ops.spyre.kv_window.default])
def spyre_kv_window(
    key: torch.Tensor,
    value: torch.Tensor,
    read_start: int,
    buffer_width: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Q block's native-head K/V window."""
    reason = check_window_read(
        read_start=read_start,
        buffer_width=buffer_width,
        cache_capacity=key.size(2),
        num_heads=num_heads,
        num_kv_heads=key.size(1),
        key_shape=tuple(key.shape),
        value_shape=tuple(value.shape),
    )
    if reason is not None:
        raise Unsupported(f"kv_window: {reason}")

    # Preserve Hkv. Native GQA adds a unit broadcast axis at the matmul rather
    # than materializing Hq copies of each K/V window.
    k_win = key[:, :, read_start : read_start + buffer_width, :]
    v_win = value[:, :, read_start : read_start + buffer_width, :]
    return k_win, v_win


def _windowed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_plan: SlidingWindowPlan | None,
    q_block: int,
    padded_seqlen_q: int,
    query_scale: float,
    num_heads: int,
) -> torch.Tensor:
    """Blocked online attention over each query block's physical KV window.

    Query blocks remain static because each one has a separately planned cache
    window. Within a block, K/V traversal is represented structurally with
    ``for_each_tile`` rather than ``spyre_hint``. Batch, head, GQA-group, and
    query work division are left to the compiler, which avoids staging a full
    cache window in an outer loop before the inner K/V slice is formed.
    Functional SSA carries match ``spyre__sdpa_overrideable``; a mutation-based
    ``copy_forced`` carry is not tile-safe when the output row spans multiple
    sticks (Gemma's head_dim=256).
    """
    num_kvheads = key.size(1)
    gqa_group_size = num_heads // num_kvheads
    use_gqa = gqa_group_size != 1

    # Keep GQA explicit as [Hkv, group], exactly as full SDPA does. K/V retain
    # Hkv and gain only a unit view axis for the broadcast-aware matmuls.
    if use_gqa:
        query = query.contiguous().unflatten(1, (num_kvheads, gqa_group_size))
        attention_mask = attention_mask.unsqueeze(2)

    # A causal square prefill has a static narrow read plan. Generic masks and
    # runtime-positioned calls scan the complete compact allocation; their
    # tensor mask is the only source of geometry.
    buffer_width = key.size(2) if causal_plan is None else causal_plan.buffer_width

    tiling = _select_swa_tiling(
        batch_size=query.size(0),
        num_heads=num_heads,
        num_kvheads=key.size(1),
        q_block=q_block,
        buffer_width=buffer_width,
        head_dim=query.size(-1),
        element_size=query.dtype.itemsize,
        num_cores=config.sencores,
        lx_budget_bytes=_sdpa_lx_budget_bytes(),
    )
    physical_buffer_width = tiling.num_kv_blocks * tiling.kv_block_size
    pad_columns = physical_buffer_width - buffer_width
    logger.debug(
        "SWA tiling: strategy=%s reason=%s q_block=%s "
        "buffer_width=%s kv_blocks=%s kv_block_size=%s "
        "head_tiles=%s work_div=%s score_bytes_per_core=%s "
        "estimated_live_bytes_per_core=%s kv_bytes_per_core=%s "
        "restick_bytes_per_core=%s restick_lx_eligible=%s "
        "estimated_dsc_executions=%s lx_budget_bytes=%s",
        tiling.strategy,
        tiling.reason,
        q_block,
        buffer_width,
        tiling.num_kv_blocks,
        tiling.kv_block_size,
        tiling.num_head_tiles,
        tiling.work_div,
        tiling.score_bytes_per_core,
        tiling.estimated_live_bytes_per_core,
        tiling.kv_bytes_per_core,
        tiling.restick_bytes_per_core,
        tiling.restick_lx_eligible,
        tiling.estimated_dsc_executions,
        tiling.lx_budget_bytes,
    )
    # Spyre stores fp16 and bf16 tensors in its 16-bit floating-point format.
    # Use values representable in that storage rather than bf16's wider range.
    storage_dtype = (
        torch.float16 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    finite_min = torch.finfo(storage_dtype).min
    positive_min = torch.finfo(storage_dtype).tiny
    out_blocks = []
    num_q_blocks = padded_seqlen_q // q_block
    for block_index in range(num_q_blocks):
        q_start = block_index * q_block
        q_end = q_start + q_block
        assert q_end - q_start == q_block, (
            f"sliding_window_attention: Q block {block_index} is "
            f"{q_end - q_start} rows, expected {q_block} -- query padding "
            "should have produced a whole number of blocks"
        )

        read_start = 0 if causal_plan is None else causal_plan.read_start(block_index)
        mask_rows = attention_mask[..., q_start:q_end, :]

        # Keep the multi-output custom reads outside the coarse-tile scope.
        # FallbackKernel/MultiOutput nodes do not carry named dimensions; putting
        # them inside the scope can pull GQA expansion and batch dependencies into
        # the wrong loop group (and, for batch > 1, leave a dangling scheduler
        # dependency after grouping).
        k_window, v_window = torch.ops.spyre.kv_window(
            key,
            value,
            read_start,
            buffer_width,
            num_heads,
        )
        mask_window = mask_rows[..., read_start : read_start + buffer_width]
        if pad_columns:
            k_window = torch.cat(
                [
                    k_window,
                    torch.zeros(
                        (*k_window.shape[:-2], pad_columns, k_window.shape[-1]),
                        device=k_window.device,
                        dtype=k_window.dtype,
                    ),
                ],
                dim=-2,
            )
            v_window = torch.cat(
                [
                    v_window,
                    torch.zeros(
                        (*v_window.shape[:-2], pad_columns, v_window.shape[-1]),
                        device=v_window.device,
                        dtype=v_window.dtype,
                    ),
                ],
                dim=-2,
            )
            mask_window = torch.cat(
                [
                    mask_window,
                    torch.full(
                        (*mask_window.shape[:-1], pad_columns),
                        float("-inf"),
                        device=mask_window.device,
                        dtype=mask_window.dtype,
                    ),
                ],
                dim=-1,
            )

        q_rows = query[..., q_start:q_end, :]

        def kv_level(q_tile, k_tile, v_tile, mask_tile):
            if use_gqa:
                k_tile = k_tile.unsqueeze(2)
                v_tile = v_tile.unsqueeze(2)

            # Q and the carry seeds are invariant across the counted K/V loop.
            # A finite maximum keeps fully masked chunks defined: their
            # exponentials and weighted contribution are exactly zero.
            q_scaled = q_tile * query_scale
            output_tile = torch.zeros_like(q_tile)
            accumulator_shape = (*q_tile.shape[:-1], STICK)
            running_max_reduced = torch.full(
                accumulator_shape,
                finite_min,
                device=query.device,
                dtype=query.dtype,
            )
            running_max = running_max_reduced.amax(dim=-1)
            denominator_reduced = torch.zeros(
                accumulator_shape,
                device=query.device,
                dtype=query.dtype,
            )
            denominator = denominator_reduced.amax(dim=-1)

            def swa_kv_body(carry, tiles):
                running_max, denominator, output = carry
                _, k_blk, v_blk, mask_blk = tiles
                # Match full SDPA: the scan walks K in cache order and
                # materializes only the bounded, transposed tile required by
                # the score matmul.
                keys_t = k_blk.transpose(-1, -2).contiguous()
                scores = (
                    torch.ops.spyre.batched_matmul(q_scaled, keys_t)
                    if use_gqa
                    else torch.matmul(q_scaled, keys_t)
                )
                scores = scores + mask_blk

                # Clamp before reducing so fully masked chunks do not form
                # ``-inf - -inf`` in the online-softmax recurrence.
                block_max = torch.amax(torch.clamp_min(scores, finite_min), dim=-1)
                # Form the old-max correction first. The loop lowering can then
                # update running_max without a carry snapshot.
                correction = torch.exp(torch.clamp_max(running_max - block_max, 0.0))
                new_max = torch.maximum(running_max, block_max)
                exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
                new_denominator = denominator * correction + exp_scores.sum(dim=-1)
                exp_scores = exp_scores.contiguous()
                weighted = (
                    torch.ops.spyre.batched_matmul(exp_scores, v_blk)
                    if use_gqa
                    else torch.matmul(exp_scores, v_blk)
                )
                new_output = output * correction.unsqueeze(-1) + weighted
                return (new_max, new_denominator, new_output), None

            kv_operands = (q_tile, k_tile, v_tile, mask_tile)
            initial = (running_max, denominator, output_tile)
            if tiling.num_kv_blocks == 1:
                (_, denominator, output_tile), _ = swa_kv_body(initial, kv_operands)
            else:
                (_, denominator, output_tile), _ = for_each_tile(
                    swa_kv_body,
                    kv_operands,
                    dims=(None, -2, -2, -1),
                    tile_size=tiling.kv_block_size,
                    init=initial,
                )
            safe_denominator = torch.clamp_min(denominator, positive_min)
            return output_tile / safe_denominator.unsqueeze(-1)

        out_blocks.append(kv_level(q_rows, k_window, v_window, mask_window))

    output = torch.cat(out_blocks, dim=-2)
    return output.flatten(1, 2) if use_gqa else output


@register_spyre_decompositions([torch.ops.spyre.sliding_window_attention.default])
def spyre_sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    window_size: int,
    is_causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    """Sliding-window attention with all runtime geometry in a tensor mask.

    Strictly causal square prefill uses static per-block KV windows. Generic
    masks and every other shape read the complete cache allocation in bounded
    chunks, so changing mask contents never changes the graph. A ragged query
    length is padded up rather than refused. In particular, decode with a
    runtime mask scales with the physical cache width, not only
    ``window_size``. Callers should therefore keep decode caches compact (about
    one window plus query-block staggering); a full-length cache remains
    correct but intentionally receives no hidden position-dependent fast path.
    """
    num_heads = query.size(1)
    head_dim = query.size(3)
    batch_size = query.size(0)
    seqlen_q = query.size(2)
    cache_capacity = key.size(2)
    num_kvheads = key.size(1)

    if scale is not None and scale < 0:
        # math.sqrt would otherwise raise a bare ValueError.
        raise Unsupported(
            f"sliding_window_attention: scale={scale} must be non-negative"
        )
    if window_size <= 0:
        raise Unsupported(
            f"sliding_window_attention: window_size={window_size} must be positive"
        )
    if seqlen_q <= 0 or cache_capacity <= 0:
        raise Unsupported(
            "sliding_window_attention: query and cache lengths must be positive, "
            f"got seqlen_q={seqlen_q}, cache_capacity={cache_capacity}"
        )
    if seqlen_q > cache_capacity:
        raise Unsupported(
            f"sliding_window_attention: seqlen_q={seqlen_q} exceeds "
            f"cache_capacity={cache_capacity}"
        )
    if cache_capacity % STICK != 0:
        raise Unsupported(
            f"sliding_window_attention: cache_capacity={cache_capacity} must be a "
            f"multiple of {STICK}"
        )
    if num_kvheads <= 0 or num_heads % num_kvheads != 0:
        raise Unsupported(
            "sliding_window_attention: query heads must be a whole multiple "
            f"of KV heads, got Hq={num_heads}, Hkv={num_kvheads}"
        )

    expected_mask_shape = (batch_size, 1, seqlen_q, cache_capacity)
    if tuple(attention_mask.shape) != expected_mask_shape:
        raise Unsupported(
            "sliding_window_attention: attention_mask shape "
            f"{tuple(attention_mask.shape)} must be {expected_mask_shape}"
        )
    if attention_mask.dtype != query.dtype:
        raise Unsupported(
            "sliding_window_attention: attention_mask dtype "
            f"{attention_mask.dtype} must match query dtype {query.dtype}"
        )
    if attention_mask.device != query.device:
        raise Unsupported(
            "sliding_window_attention: attention_mask device "
            f"{attention_mask.device} must match query device {query.device}"
        )

    query_scale = scale
    if query_scale is None:
        query_scale = 1.0 / math.sqrt(head_dim)

    # Pad at the front so real rows keep their relative order. Synthetic rows
    # receive an all-zero mask and are sliced from the result; this guarantees a
    # finite softmax without imposing semantics on discarded outputs.
    has_narrow_static_plan = (
        is_causal and seqlen_q == cache_capacity and window_size < cache_capacity
    )
    q_block, padded_seqlen_q = query_blocking(
        seqlen_q,
        max_query_block=(STICK if has_narrow_static_plan else MAX_QUERY_BLOCK),
    )
    pad_rows = padded_seqlen_q - seqlen_q

    # Only a strictly causal square prefill has position and its upper bound
    # fully determined by tensor shapes. A generic mask can allow future keys,
    # while non-square calls can carry a changing query origin in mask values;
    # both must scan the complete allocation. A compact decode cache keeps that
    # bounded by the configured window.
    causal_plan = None
    if is_causal and seqlen_q == cache_capacity:
        causal_plan = plan_sliding_window(
            padded_seqlen_q,
            cache_capacity,
            window_size,
            is_causal=True,
            q_block=q_block,
            cache_capacity=cache_capacity,
        )
        if causal_plan is None:
            # Never None when the plan is valid, but the type says otherwise.
            reason = (
                rejection_reason(
                    padded_seqlen_q,
                    cache_capacity,
                    window_size,
                    True,
                    q_block,
                    cache_capacity,
                )
                or "the window placement cannot express this shape"
            )
            raise Unsupported(f"sliding_window_attention: {reason}")
    if pad_rows:
        query = torch.cat(
            [
                torch.zeros(
                    (batch_size, num_heads, pad_rows, head_dim),
                    device=query.device,
                    dtype=query.dtype,
                ),
                query,
            ],
            dim=2,
        )
        attention_mask = torch.cat(
            [
                torch.zeros(
                    (batch_size, 1, pad_rows, cache_capacity),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
                attention_mask,
            ],
            dim=2,
        )

    output = _windowed_attention(
        query,
        key,
        value,
        attention_mask,
        causal_plan,
        q_block,
        padded_seqlen_q,
        query_scale,
        num_heads,
    )
    return output[:, :, pad_rows:, :] if pad_rows else output


@register_spyre_decompositions([torch.ops.aten.max.default])
def spyre_max_default_decomp(input):
    """
    Decompose torch.max(input) with conditional CPU fallback for int64.

    For int64 tensors, use custom op spyre::max_default_int64_fallback which has
    a CPU fallback registered in fallbacks.py.
    For other dtypes (float16, float32, etc.), use amax.
    """
    if input.dtype == torch.int64:
        # Use custom op with CPU fallback to avoid recursive decomposition
        # Returns a scalar (0D) tensor
        return torch.ops.spyre.max_default_int64_fallback(input)
    else:
        # Use amax for supported dtypes (can run on Spyre)
        # Returns a scalar (0D) tensor
        return torch.ops.aten.amax(input)


@register_spyre_decompositions([torch.ops.aten.max.dim])
def spyre_max_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.max(input, dim) with conditional handling for bool and int64.
    For bool: convert to float16, perform max, convert back (bool stored as fp16 on Spyre).
    For int64: use CPU fallback custom op (not supported on Spyre).
    For other dtypes: use default PyTorch decomposition (amax + argmax).
    """
    if input.dtype == torch.bool:
        # Reinterpret bool as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
        float_dtype = _get_float_dtype_for_bool()
        input_float = torch.ops.prims.convert_element_type(input, float_dtype)
        values_float = torch.ops.aten.amax(input_float, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input_float, dim=dim, keepdim=keepdim)
        values = torch.ops.prims.convert_element_type(values_float, torch.bool)
        return torch.return_types.max((values, indices))
    elif input.dtype == torch.int64:
        # Use CPU fallback custom op for int64
        return torch.ops.spyre.max_dim_int64_fallback(input, dim=dim, keepdim=keepdim)
    else:
        # Use amax and argmax for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amax(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input, dim=dim, keepdim=keepdim)
        return torch.return_types.max((values, indices))


@register_spyre_decompositions([torch.ops.aten.min.dim])
def spyre_min_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.min(input, dim) with conditional handling for bool and int64.
    For bool: convert to float16, perform min, convert back (bool stored as fp16 on Spyre).
    For int64: use CPU fallback custom op (not supported on Spyre).
    For other dtypes: use default PyTorch decomposition (amin + argmin).
    """
    if input.dtype == torch.bool:
        # Reinterpret bool as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
        float_dtype = _get_float_dtype_for_bool()
        input_float = torch.ops.prims.convert_element_type(input, float_dtype)
        values_float = torch.ops.aten.amin(input_float, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input_float, dim=dim, keepdim=keepdim)
        values = torch.ops.prims.convert_element_type(values_float, torch.bool)
        return torch.return_types.min((values, indices))
    elif input.dtype == torch.int64:
        # Use CPU fallback custom op for int64
        return torch.ops.spyre.min_dim_int64_fallback(input, dim=dim, keepdim=keepdim)
    else:
        # Use amin and argmin for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amin(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input, dim=dim, keepdim=keepdim)
        return torch.return_types.min((values, indices))


@register_spyre_decompositions([torch.ops.aten.amax.default])
def spyre_amax_decomp(
    input: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    """
    Decompose torch.amax for boolean tensors.
    For bool tensors: convert to float16, perform amax, convert back (bool stored as fp16 on Spyre).
    For other dtypes: return NotImplemented to use default behavior.
    """
    if input.dtype != torch.bool:
        # For non-bool types, don't decompose - use default lowering
        return NotImplemented

    # For bool tensors: reinterpret as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
    float_dtype = _get_float_dtype_for_bool()
    input_float = torch.ops.prims.convert_element_type(input, float_dtype)
    if dim is None:
        result_float = torch.ops.aten.amax(input_float, keepdim=keepdim)
    else:
        result_float = torch.ops.aten.amax(input_float, dim=dim, keepdim=keepdim)
    return torch.ops.prims.convert_element_type(result_float, torch.bool)


@register_spyre_decompositions([torch.ops.aten.amin.default])
def spyre_amin_decomp(
    input: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    """
    Decompose torch.amin for boolean tensors.
    For bool tensors: convert to float16, perform amin, convert back (bool stored as fp16 on Spyre).
    For other dtypes: return NotImplemented to use default behavior.
    """
    if input.dtype != torch.bool:
        # For non-bool types, don't decompose - use default lowering
        return NotImplemented

    # For bool tensors: reinterpret as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
    float_dtype = _get_float_dtype_for_bool()
    input_float = torch.ops.prims.convert_element_type(input, float_dtype)
    if dim is None:
        result_float = torch.ops.aten.amin(input_float, keepdim=keepdim)
    else:
        result_float = torch.ops.aten.amin(input_float, dim=dim, keepdim=keepdim)
    return torch.ops.prims.convert_element_type(result_float, torch.bool)


@register_spyre_decompositions([torch.ops.aten.ceil.default])
def spyre_ceil(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.neg.default(
        torch.ops.aten.floor.default(torch.ops.aten.neg.default(input))
    )


# ---------------------------------------------------------------------------
# cos / sin via Cody-Waite range reduction + degree-9 Taylor series
#
# All RoPE call sites use fp32 inputs (confirmed from tests/resource/models/).
# Ops used: floor, mul, add, sub — all natively lowered on Spyre.
# torch.round is NOT used: it is not implemented in the Spyre codegen;
# round-to-nearest is expressed as floor(x + 0.5).
#
# Accuracy (fp32 input, measured against an fp64 reference): worst-case absolute
# error ~5.3e-5 for both cos and sin on RoPE-realistic inputs (|x| ≤ 1063, i.e.
# seq_len=1064 with inv_freq[0]=1.0).  On a dense linspace sweep the two differ:
# cos is the limiting op at ~2.5e-5 even for |x| ≤ π, because the polynomial
# partially cancels near |x_r| = π/2; sin stays at ~3.7e-6 there.  Safe
# tolerance for both: 1e-4.  The downstream fp16 cast in model inference absorbs
# this entirely (fp16 ULP at 1.0 is ~1e-3).
#
# Narrower dtypes are computed in fp32 and cast back.  The range reduction needs
# k = floor(x/π + 0.5) to resolve x against π, and a 10- or 7-bit mantissa runs
# out of bits as |x| grows: evaluated at fp16 width the error reaches ~0.75 at
# |x| ≤ 1000, i.e. unusable rather than merely coarse.  Computing the whole body
# in fp32 fixes it — measured on device, against cos of the value the device
# actually holds, the error is 4.9e-4 (one fp16 ULP, i.e. optimal) at every range
# from π to 1000, versus 0.75 without the upcast.  A reduction-only fp32 window
# is also honored but is weaker (2.4e-3), so the whole body is upcast.
#
# Note when validating fp16 against a CPU reference: Spyre's fp16 is a 1-6-9
# format (9 mantissa bits, not IEEE's 10), so H2D re-rounds an IEEE fp16 input by
# up to 1 ULP.  That shifts cos/sin by up to 0.5 at |x| ~ 1000 no matter how the
# op is implemented — it hits a CPU fallback identically (measured 0.495 for
# both) — so a *device* result must be compared against cos of the round-tripped
# input, not of the original.  Tolerances below are stated on that basis.
#
# KNOWN LIMITATION for fp16/bf16, issue #2818, not introduced here: the fp32
# window this decomposition opens closes with an fp32 -> fp16/bf16 cast, and that
# cast is wrong on device unless the innermost dim spans an EVEN number of fp32
# sticks, i.e. ceil(size[-1] / 32) % 2 == 0.  Two 32-element fp32 sticks pair
# into one 64-element fp16 stick, and an odd count leaves a dangling half stick
# where "the fp32 tensor is allocated with a stick of 32 elements but the SDSC
# shapes are asking for 64" (#2818).  Measured on a sweep of (4, N): correct at
# N = 48, 64, 112, 128, 192, 240, 256, 320 (even stick counts); wrong or a hard
# "Invalid device sizes and stride map" at N = 16, 32, 65, 80, 96, 129, 160, 224
# (odd), with over half the elements taking values that are not in the correct
# result at all.  Note the rule is the stick pairing, not "multiple of 64":
# N = 48 is correct and N = 96 is not.
#
# This is a backend cast defect, not a cos/sin one, and it is not this
# decomposition's to fix: a bare ``(x.float() * 2.0).to(torch.float16)`` on the
# same shapes is wrong by the same amount with no cos/sin in the graph, while a
# pure fp16 pointwise op (abs, mul) is bit-exact there.  It is also not #4392
# (no zero-sized device dim appears on these graphs, and a ceil in
# ``rescale_stl_for_dtype`` changes nothing) and not #4393 (the output carries a
# STANDARD element arrangement, and a following compiled graph reads back the
# same wrong values, so the device data itself is wrong rather than merely
# permuted on copy-out).  RoPE head dims (64, 128) are even-stick and unaffected.
#
# PI_HI: nearest fp32 to π (stored as a Python float / fp64 constant so the
#         compiler sees the exact value rather than a rounded literal).
# PI_LO: fp64 residual (π − PI_HI), used in the two-term subtraction to
#         suppress range-reduction error accumulated across large k values.
# ---------------------------------------------------------------------------

_PI_HI = 3.1415927410125732  # float32(π) as fp64
_PI_LO = -8.742278012618954e-8  # π − PI_HI in fp64


def _taylor_range_reduce(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cody-Waite two-term range reduction.

    Returns (x_r, sign) where x_r ∈ [−π/2, π/2] and sign = (−1)^k.
    k is computed as round(x / π) using floor(x/π + 0.5) to avoid
    torch.round (not supported in the Spyre codegen).
    """
    k = torch.ops.aten.floor.default(x * (1.0 / math.pi) + 0.5)
    x_r = (x - k * _PI_HI) - k * _PI_LO
    k_mod2 = k - 2.0 * torch.ops.aten.floor.default(k * 0.5)
    sign = 1.0 - 2.0 * k_mod2
    return x_r, sign


def _taylor_cos(x: torch.Tensor) -> torch.Tensor:
    """Degree-9 Horner cos. Caller must widen to fp32 (see the accuracy note)."""
    x_r, sign = _taylor_range_reduce(x)
    x2 = x_r * x_r
    poly = 1.0 + x2 * (
        -0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0)))
    )
    return sign * poly


def _taylor_sin(x: torch.Tensor) -> torch.Tensor:
    """Degree-9 Horner sin. Caller must widen to fp32 (see the accuracy note)."""
    x_r, sign = _taylor_range_reduce(x)
    x2 = x_r * x_r
    poly = x_r * (
        1.0
        + x2
        * (
            -1.0 / 6.0
            + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))
        )
    )
    return sign * poly


def _taylor_dtypes(input: torch.Tensor) -> tuple[torch.dtype, torch.dtype]:
    """Compute and result dtypes for a cos/sin body, as aten defines them.

    The result dtype is aten's, asked of aten rather than reimplemented:
    ``ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT`` is the kind
    ``torch._refs.cos`` is built with, so integral and boolean input yield the
    default floating dtype (nothing is truncated back to an integral dtype) and
    floating input is returned at its own width.

    The compute dtype is *not* taken from the same call, and this is deliberate.
    ``elementwise_dtypes`` derives it through ``get_computation_dtype``, which
    reads ``torch._prims_common._computation_dtype_map`` -- and
    ``torch_spyre._inductor.patches.spyre_data_types`` deliberately replaces that
    map with identity entries for the whole Inductor compile, so refs do not widen
    fp16 to fp32 on a device whose native dtype is fp16.  Inside a Spyre compile
    it therefore reports ``compute=float16`` for fp16 input where an eager call
    reports ``compute=float32``; wearing ``elementwise_type_promotion_wrapper``
    here looks idiomatic but widens nothing on the path that matters (measured:
    0.7501 max error at fp16, identical to no upcast at all, against 5.0e-4 for
    the explicit ``.to`` below).  cos/sin need the wider window for the range
    reduction regardless of that policy, so they ask for it in the graph with an
    explicit cast, which lowers to ``aten._to_copy`` and survives.

    fp64 falls out of ``promote_types``: computation and result are both fp64,
    which is not a claim that Spyre executes fp64 -- H2D rejects a Double tensor
    outright.  It matters only because the bodies above are plain torch
    functions the CPU-side accuracy tests call directly, and this never narrows
    one of those.  Eager dispatch cannot arrive here with fp64 either, even
    though ``_register_spyre_dispatchkey_kernels_permanently`` installs a
    PrivateUse1 kernel for ``aten.cos`` / ``aten.sin``: that key selects on
    device, not dtype, so a CPU fp64 tensor takes the CPU kernel, and no fp64
    tensor can sit on a Spyre device to route here -- placement raises
    ``Spyre backend does not support dtype Double``, and widening an
    already-placed tensor via ``.to(torch.float64)`` dies with SIGFPE inside the
    cast (measured; issue #1201's territory, nothing to do with this body).

    Complex is the one dtype aten accepts that these bodies do not serve, and it
    needs no guard because it cannot arrive: ``.to("spyre")`` rejects a complex
    tensor outright (``Spyre backend does not support dtype ComplexFloat``), so
    no Spyre compile ever sees one.  Called directly on CPU it raises
    ``NotImplementedError`` from ``torch.floor``, which is the right answer --
    range-reducing a complex argument against pi is meaningless, ``cos(a+bi)``
    needing ``cosh``/``sinh`` instead.  Should complex placement ever land,
    cos/sin would need a complex guard or a fallback registration, since they no
    longer appear in ``register_fallback_default``.
    """
    _, result_dtype = elementwise_dtypes(
        input, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    return torch.promote_types(result_dtype, torch.float32), result_dtype


@register_spyre_decompositions([torch.ops.aten.cos.default])
def spyre_cos(input: torch.Tensor) -> torch.Tensor:
    """cos(x) via Cody-Waite range reduction and degree-9 Horner polynomial.

    Serves every real dtype aten accepts, so no CPU fallback is needed: the body
    runs at fp32 or wider and the result carries aten's own dtype.  Complex is
    aten's one dtype this does not serve, and is unreachable on this backend
    rather than guarded against -- see ``_taylor_dtypes``.
    """
    compute_dtype, result_dtype = _taylor_dtypes(input)
    out = _taylor_cos(input.to(compute_dtype))
    return out if out.dtype == result_dtype else out.to(result_dtype)


@register_spyre_decompositions([torch.ops.aten.sin.default])
def spyre_sin(input: torch.Tensor) -> torch.Tensor:
    """sin(x) via Cody-Waite range reduction and degree-9 Horner polynomial.

    Serves every real dtype aten accepts, so no CPU fallback is needed: the body
    runs at fp32 or wider and the result carries aten's own dtype.  Complex is
    aten's one dtype this does not serve, and is unreachable on this backend
    rather than guarded against -- see ``_taylor_dtypes``.
    """
    compute_dtype, result_dtype = _taylor_dtypes(input)
    out = _taylor_sin(input.to(compute_dtype))
    return out if out.dtype == result_dtype else out.to(result_dtype)


@register_spyre_decompositions([torch.ops.aten.bitwise_not])
def bitwise_not(input: torch.Tensor) -> torch.Tensor:
    if input.dtype is torch.bool:
        return torch.logical_not(input)
    else:
        neg_one = torch.ops.aten.full_like(input, -1)
        return torch.ops.aten.bitwise_xor(input, neg_one)


@register_spyre_decompositions([torch.ops.aten.bitwise_and])
def bitwise_and(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if input1.dtype is torch.bool and input2.dtype is torch.bool:
        return torch.ops.aten.logical_and(input1, input2)
    else:
        return torch.ops.aten.bitwise_not(
            torch.ops.aten.bitwise_or(
                torch.ops.aten.bitwise_not(input1), torch.ops.aten.bitwise_not(input2)
            )
        )


#: Largest kernel tap (per spatial axis) the direct conv2d path accepts. A
#: dense (groups==1) conv contracts over C_in*kH*kW; that per-output-channel
#: weight working set grows with k**2 and, at k>3 with a stick-aligned C_in>=64,
#: exceeds the initial-chunk LX budget -- the backend aborts (the initial
#: chunk must fit in LX) because the kernel taps are pinned no-split
#: (ki/kj=1) and C_out=64 is a single stick, so
#: there is nothing left to tile. Depthwise conv escapes this (its contraction
#: is kH*kW only, no C_in), which is why depthwise supports k up to 9 and dense
#: does not. Until the backend can tile the C_in*kH*kW contraction for dense
#: conv, k>3 stays on the im2col+matmul decomposition.
_CONV_MAX_KERNEL = 3


def _is_direct_conv_supported(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    transposed: bool,
    output_padding: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> bool:
    """Cases the native conv2d direct lowering (lower_convolution) handles.

    Keep this in lock-step with the guards in lower_convolution so that whenever
    the decomposition defers here, the lowering is guaranteed to accept the op.
    Everything else stays on the im2col+matmul decomposition.  Excluded:
    - 1x1 kernel: only the 1x1 case has size-1 kernel taps on *both* axes
      (ki and kj), which the pipeline squeezes out so the emitted SDSC carries
      no window dims at all -- and the backend's conv path expects at least one
      windowed spatial dim, aborting in dimension-mapping (ddl_conversion.cpp
      "Unknown primary dimension kind for a window dimension").  A 1x1 conv is
      just a channel matmul, so it stays on the im2col+matmul path, which handles
      it exactly.  A 1xN / Nx1 kernel squeezes only one tap and keeps the other
      window dim, which the backend does accept (a 1-D conv) -- so those
      direct-lower and are covered by the test_conv2d_direct k1x3 / k3x1 cases;
    - non-zero padding: the backend zero-fill for a padded conv input is not wired
      for regular conv2d, so pad>0 stays on the im2col+matmul path (which pads
      correctly);
    - dilated conv: the windowed-input SDSC fields reuse the avgpool builder,
      which assumes dilation==1, so d>1 stays on the im2col+matmul path;
    - C_in not stick-aligned: Spyre stores C as the innermost (stick) dim, and
      the conv SDSC contracts over C_in with no partial-stick handling. A C_in
      that is not a whole multiple of the fp16 stick width (get_elem_in_stick,
      = 64) would need contraction-dim padding the direct path does not emit
      (known-broken), so it stays on the im2col+matmul path;
    - kernel tap > _CONV_MAX_KERNEL (3): the dense C_in*kH*kW contraction working
      set overflows the LX budget in the backend for k>3 and cannot be tiled
      (see _CONV_MAX_KERNEL), so it stays on the im2col+matmul path;
    - ragged input width under stride: when the strided windows do not exactly
      cover the input width -- (W_in - kW) % sW != 0 -- the fp16 conv opfunc's
      width tiling mis-accumulates the dangling partial column, so such convs
      stay on the im2col+matmul path. A ragged *height* is harmless (height is
      untiled) and stride==1 is never ragged, so this only excludes strided
      convs whose width does not divide evenly (all HW-verified).

    Why these gates run here (decomposition/routing time) rather than in layout
    propagation, where the device stick dim is actually assigned:

    - Declining here is what preserves the fallback. conv2d_via_bmm_decomp either
      defers (returns NotImplemented, leaving aten.convolution for
      lower_convolution to direct-lower) or expands into im2col+matmul -- and once
      it expands, the conv node is gone, replaced by a reshape+bmm subgraph.
      Inductor lowering is a single forward pass with no backtracking, so there is
      no way to un-decompose and re-route afterwards. By the time layout
      propagation runs, the graph is already committed to the direct path; a
      stick-alignment failure discovered there is a hard compile error, not a
      graceful fallback. So the decision has to be made before the branch, i.e.
      here.
    - Making it this early is correct because the stick-alignment gate is
      layout-invariant. C_in is logical dim 1 by the aten.convolution NCHW
      contract (guarded by input.dim() == 4), and ``C_in % stick == 0`` is a
      property of the channel *count*, which no layout choice changes -- layout
      propagation picks stick *placement*, not size. We are not assuming which
      host dim becomes the stick: the direct path itself forces channel-last (C on
      the stick) to feed the PE-array contraction, so this validates a
      precondition of the layout the path *will request*, not a guess about an
      assignment the solver is free to make differently. The stick width is
      get_elem_in_stick(torch.float16) == 64, derived from the dtype the fp16 gate
      above already pins -- not a hardcoded dim assumption.

    Assumption this routing decision rests on (documented, not enforced here):
    the gate reads C_in from logical dim 1 (guaranteed by the aten.convolution
    NCHW contract) and assumes the direct path will place C_in on the device
    stick. That stick placement is NOT decided here -- it is requested by the
    direct path and enforced downstream in propagate_layouts (_conv_layouts /
    find_stick_compatible_input_layout), which restickify the activation onto
    C_in or raise Unsupported if they cannot. So the ``C_in % stick == 0`` check
    below is a precondition of the channel-last layout the path *will request*,
    validated against the channel *count* (which no layout choice changes). The
    assumption is only that C_in-on-stick keeps being the layout a direct-conv
    node lands in; if that ever stops holding (e.g. a solver change assigns a
    different stick dim to a direct-conv node), this gate would be checking the
    wrong dimension and could route wrongly. It is documented here as an
    assumption rather than re-checked after layout assignment because by then
    the im2col+matmul fallback branch is gone (see above) -- a mismatch surfaces
    downstream as a hard Unsupported, not silent wrong numerics.
    """
    kH, kW = weight.shape[-2], weight.shape[-1]
    C_in = input.shape[1]
    eps = get_elem_in_stick(torch.float16)
    supported = (
        not transposed
        and all(op == 0 for op in output_padding)
        and all(p == 0 for p in padding)
        and all(d == 1 for d in dilation)
        and groups == 1
        and input.dim() == 4
        and input.dtype == torch.float16
        and not (kH == 1 and kW == 1)
        # Dense conv k>3 overflows the LX contraction budget in the backend.
        and kH <= _CONV_MAX_KERNEL
        and kW <= _CONV_MAX_KERNEL
        # isinstance guard: a dynamic-shape C_in (SymInt) is not statically known
        # to be stick-aligned, so fall back to the decomposition rather than
        # branching on a symbolic divisibility (which would add a shape guard).
        and isinstance(C_in, int)
        # Assumes the direct path lands C_in (logical dim 1) on the stick; that
        # is requested by the path and enforced in propagate_layouts, not here.
        # See the "Assumption this routing decision rests on" note above.
        and C_in % eps == 0
    )
    if not supported:
        return False
    # Ragged input width (see docstring): the fp16 opfunc tiles the output width
    # and mis-accumulates the dangling column when (W_in - kW) % sW != 0. Only
    # decidable for a static width; a dynamic (SymInt) width stays on the direct
    # path rather than adding a symbolic-remainder shape guard.
    W_in = input.shape[-1]
    sW = stride[-1]
    if isinstance(W_in, int) and (W_in - kW) % sW != 0:
        return False
    return True


@register_spyre_decompositions([torch.ops.aten.convolution.default])
def conv2d_via_bmm_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    """
    Decompose 2D convolution into batch matrix multiplication using torch.nn.unfold.
    torch.nn.unfold directly returns (N, C_in * K_h * K_w, H_out * W_out), avoiding
    intermediate reshape/view/unsqueeze operations.
    For depthwise convolutions (C_in = groups = C_out), invoke torch.spyre.conv2d directly.
    """
    # When the direct-lowering flag is on and the case is supported, decline the
    # decomposition (return NotImplemented) so aten.convolution.default survives
    # in the FX/AOT graph and reaches the Spyre lowering (lower_convolution),
    # which emits a native conv2d SDSC. Unsupported cases (grouped/transposed/
    # non-fp16) fall through and decompose to im2col+matmul as before. This is
    # the compile-path target; the flag defaults off so eager and default
    # compile behavior are unchanged.
    if config.conv2d_direct_lowering and _is_direct_conv_supported(
        input, weight, stride, transposed, output_padding, padding, dilation, groups
    ):
        return NotImplemented

    if transposed:
        raise Unsupported("conv2d_via_bmm: transposed convolution not supported")

    if any(op != 0 for op in output_padding):
        raise Unsupported("conv2d_via_bmm: output_padding not supported")

    if input.dim() != 4:
        raise Unsupported(f"conv2d_via_bmm: expected 4D input, got {input.dim()}D")

    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, K_h, K_w = weight.shape

    # For depthwise convolutions (C_in = groups = C_out), use torch.spyre.conv2d_with_bias
    if C_in == groups == C_out:
        return torch.ops.spyre.conv2d_with_bias(
            input, weight, bias, stride, padding, dilation, groups
        )

    stride_h, stride_w = stride[0], stride[1]
    pad_h, pad_w = padding[0], padding[1]
    dil_h, dil_w = dilation[0], dilation[1]

    if C_in != groups * C_in_per_group:
        raise Unsupported(
            f"conv2d_via_bmm: expect C_in == groups * C_in_per_group, got C_in: {C_in}, groups: {groups} C_in_per_group: {C_in_per_group}"
        )

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    patches = torch.ops.spyre.unfold(
        input,
        kernel_size=(K_h, K_w),
        dilation=(dil_h, dil_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
    )

    if groups == 1:
        # weight_2d = weight.reshape(C_out, C_in_per_group * K_h * K_w)
        weight_2d = torch.ops.spyre.reshape_via_cpu(
            weight, (C_out, C_in_per_group * K_h * K_w)
        )
        weight_2d_exp = weight_2d.unsqueeze(0).expand(N, -1, -1)
        weight_2d_exp_cln = weight_2d_exp.clone()
        # output = torch.matmul(weight_2d, patches)
        output = torch.matmul(weight_2d_exp_cln, patches)
    else:
        C_out_per_group = C_out // groups
        # patches = patches.reshape(N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        patches = torch.ops.spyre.reshape_via_cpu(
            patches, (N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        )
        # weight_grouped = weight.reshape(groups, C_out_per_group, C_in_per_group * K_h * K_w)
        weight_grouped = torch.ops.spyre.reshape_via_cpu(
            weight, (groups, C_out_per_group, C_in_per_group * K_h * K_w)
        )

        output = torch.matmul(
            weight_grouped.unsqueeze(0),
            patches,
        )
        output = output.reshape(N, C_out, H_out * W_out)

    if bias is not None:
        # Add bias while the output is still (N, C_out, H_out * W_out): the
        # matmul lays the trailing H_out*W_out dim on the stick, and for a
        # patch-embed conv (e.g. Prithvi's stride-16 kernel) that flat length
        # is a multiple of 64 even when H_out/W_out individually are not. If we
        # instead reshaped to (N, C_out, H_out, W_out) first and broadcast the
        # bias over a sub-stick spatial width (e.g. W_out == 32), the layout
        # solver rejects the resulting `w + 32*Mod(row, 2)` stick expression.
        bias_shaped = torch.ops.spyre.reshape_via_cpu(bias, (1, C_out, 1))
        output = output + bias_shaped

    output = output.reshape(N, C_out, H_out, W_out)

    return output


@register_spyre_decompositions([torch.ops.spyre.conv2d_with_bias.default])
def spyre_conv2d_with_bias_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """
    Decompose torch.ops.spyre.conv2d_with_bias into:
      1. torch.ops.spyre.conv2d without bias
      2. Expand bias to (N, C_out, H_out, W_out) for broadcasting
      3. Add the bias to the output

    This keeps the spyre.conv2d lowering simple while supporting conv2d with bias.
    """
    # Call spyre.conv2d without bias
    output = torch.ops.spyre.conv2d(input, weight, stride, padding, dilation, groups)

    # If bias is present, add it
    if bias is not None:
        # Get output shape
        N, C_out, H_out, W_out = output.shape
        # Reshape bias to (1, C_out, 1, 1) then expand to (N, C_out, H_out, W_out)
        # This avoids stick layout issues by expanding before adding
        bias_expanded = bias.reshape(1, C_out, 1, 1).expand(N, C_out, H_out, W_out)
        output = output + bias_expanded

    return output


# Register decomposition for custom spyre op (not aten, so use decomp.register_decomposition directly)
@register_spyre_decompositions([torch.ops.spyre.dequantize_fp8_with_scale])
def dequantize_fp8_with_scale_decomp(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Decompose dequantize_fp8_with_scale into:
    1. FP8→FP16 conversion using .to() (triggers fp8todl16 via dtype_ops)
    2. Multiply by scale

    This decomposition is executed during compilation and removes the custom op
    from the graph before lowering.
    """
    x_fp16 = input.to(torch.float16)
    return x_fp16 * scale


@register_spyre_decompositions([torch.ops.aten._scaled_mm.default])
def scaled_mm_decomp(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    bias: torch.Tensor = None,
    scale_result: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    Decompose _scaled_mm into:
    1. Raw FP8 matmul via spyre.scaled_mm (no scale/bias applied)
    2. Multiply by scale_a, if present
    3. Multiply by scale_b, if present
    4. Add bias, if present

    This decomposition is executed during compilation and keeps scale/bias
    arithmetic out of lower_scaled_mm's matmul lowering - the same
    separation dequantize_fp8_with_scale_decomp uses for its FP8->FP16
    conversion.
    """
    result = torch.ops.spyre.scaled_mm(mat1, mat2, out_dtype=out_dtype)

    if scale_a is not None:
        result = result * scale_a
    if scale_b is not None:
        result = result * scale_b
    if bias is not None:
        result = result + bias

    if scale_result is not None:
        logger.warning("scale_result parameter in _scaled_mm is not yet supported")
    if use_fast_accum:
        logger.warning("use_fast_accum parameter in _scaled_mm is not yet supported")

    return result


@register_spyre_decompositions([torch.ops.aten.where.ScalarOther])
def where_scalar_other_decomp(condition, self, other):
    other_t = torch.full_like(self, other)
    return torch.ops.aten.where.self(condition, self, other_t)


@register_spyre_decompositions([torch.ops.aten.where.ScalarSelf])
def where_scalar_self_decomp(condition, self, other):
    self_t = torch.full_like(other, self)
    return torch.ops.aten.where.self(condition, self_t, other)


@register_spyre_decompositions([torch.ops.aten.where.Scalar])
def where_scalar_decomp(condition, self, other):
    # Must use dtype float16 for spyre backend where3
    dtype = torch.float16

    # Use full.default instead of full_like to explicitly control dtype
    self_t = torch.ops.aten.full.default(
        list(condition.shape),
        self,
        dtype=dtype,
        device=condition.device,
    )
    other_t = torch.ops.aten.full.default(
        list(condition.shape),
        other,
        dtype=dtype,
        device=condition.device,
    )

    return torch.ops.aten.where.self(condition, self_t, other_t)


@register_spyre_decompositions([torch.ops.spyre.quantize_fp8_with_scale])
def spyre_quantize_fp8_with_scale(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    inv_scale = torch.reciprocal(scale)
    x_scaled = input * inv_scale
    x_clamped = torch.ops.spyre.clamp(x_scaled, FP8_E4M3FN_MIN, FP8_E4M3FN_MAX)
    return torch.ops.spyre.qfp8ch(x_clamped)


@register_spyre_decompositions([torch.ops.spyre.quantize_weight_fp8_with_scale])
def spyre_quantize_weight_fp8_with_scale(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    inv_scale = torch.reciprocal(scale)
    x_scaled = input * inv_scale
    x_clamped = torch.ops.spyre.clamp(x_scaled, FP8_E4M3FN_MIN, FP8_E4M3FN_MAX)
    return torch.ops.spyre.qfp8wt(x_clamped)


@register_spyre_decompositions([torch.ops.aten.flip.default])
def spyre_flip(input: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """Reverse ``input`` along each dim in ``dims`` using gathers.

    Inductor's default decomposition of ``aten.flip`` is ``prims.rev``, whose
    index expression walks the reversed dim backwards (``N - 1 - i``). Device
    coordinates can only ascend, so that form is not lowerable on Spyre —
    ``compute_coordinates`` rejects it. The same reversal expressed as an
    ``index_select`` with a descending index tensor is an ordinary gather,
    which the backend does support: the descending order lives in the index
    *values* rather than in the access pattern.

    Registering the aten op also installs the PrivateUse1 kernel, so this is
    what eager ``Tensor.flip`` dispatches to as well (there is no
    ``aten::flip`` kernel for Spyre otherwise).
    """
    # Replacing the op means aten.flip's own argument validation no longer
    # runs, so repeat it here rather than silently accepting a program aten
    # rejects. Out-of-range dims still raise from ``size()`` below.
    seen: set[int] = set()
    for dim in dims:
        normalized = dim + input.dim() if dim < 0 else dim
        if normalized in seen:
            raise RuntimeError(
                f"dim {normalized} appears multiple times in the list of dims"
            )
        seen.add(normalized)

    out = input
    reversed_any = False
    for dim in dims:
        # A 0-d tensor accepts flip(0) and is its own reversal; ``size(0)``
        # would raise on it, so skip before asking.
        size = 1 if input.dim() == 0 else out.size(dim)
        if size <= 1:
            # A dim of size 0 or 1 is its own reversal; index_select would
            # still work, but skipping avoids an empty/degenerate gather.
            continue
        index = torch.arange(size - 1, -1, -1, device=out.device, dtype=torch.int32)
        out = torch.index_select(out, dim, index)
        reversed_any = True
    # aten.flip always returns a fresh tensor; clone so the no-op case does
    # not alias its input.
    return out if reversed_any else out.clone()


@register_spyre_decompositions([torch.ops.aten.prod.dim_int])
def spyre_prod_dim_int(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    # int64 is converted to fp32 for now, so it stays on the decomposition
    # path below.
    if input.dtype != torch.int64:
        return torch.ops.spyre.prod_dim_int(input, dim, keepdim)

    if dim < 0:
        dim += input.ndim
    out_shape = list(input.shape)
    reduce_size = out_shape.pop(dim)
    acc = torch.ones(out_shape, dtype=input.dtype, device=input.device)
    for i in range(reduce_size):
        acc = acc * input.select(dim, i)

    if keepdim:
        acc = acc.unsqueeze(dim)

    return acc


@register_spyre_decompositions(
    [torch.ops.aten.all.default, torch.ops.aten.all.dim, torch.ops.aten.all.dims]
)
def spyre_all(
    input: torch.Tensor,
    dim=None,
    keepdim: bool = False,
) -> torch.Tensor:
    # Convert bool to float16 if needed
    if input.dtype is torch.bool:
        tmp = input.to(torch.float16)
    else:
        tmp = input

    tmp = torch.abs(tmp)
    result = torch.amin(tmp, dim=dim, keepdim=keepdim)

    return result.to(torch.bool)


def _masked_scatter_reject_reason(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> Optional[str]:
    """Why ``masked_scatter`` cannot use the row-level path here, or ``None`` if it can.

    The mask must select whole rows: constant across the last (column) dim, with
    every other dim matching ``self`` so there is exactly one mask bool per row.
    That covers both spellings PyTorch may hand us for a row-broadcast mask, which
    the body collapses identically via ``mask[..., 0]``:
      * un-expanded -- the last dim is literally ``1`` (e.g. ``[B, S, 1]``); or
      * expanded    -- the last dim is ``cols`` but broadcast (``stride(-1) == 0``).

    Checks are ordered cheapest/most-general to most-specific:
      1. Rank guards (no device_layout access needed).
      2. Leading-dim equality (one mask entry per row).
      3. Degenerate column guard (cols <= 1 is not a meaningful row).
      4. Source column alignment.
      5. Per-row last-dim check (the structural row-level requirement).
    """
    if self.dim() < 2 or source.dim() < 2 or mask.dim() != self.dim():
        return f"rank: self={self.dim()} mask={mask.dim()} source={source.dim()}"
    # One mask entry per row: every dim but the last must match self exactly, so
    # mask[..., 0] has exactly `rows` elements. (The last dim may differ: it is
    # either a literal 1 or a broadcast of cols -- checked below.)
    if tuple(mask.shape[:-1]) != tuple(self.shape[:-1]):
        return (
            f"mask leading dims {tuple(mask.shape[:-1])} != self "
            f"{tuple(self.shape[:-1])}"
        )
    cols = self.shape[-1]
    # cols <= 1 is a degenerate row that offers no block-per-row equivalence.
    if cols <= 1:
        return f"degenerate last dim: cols={cols}"
    if source.shape[-1] != cols:
        return f"source last dim {source.shape[-1]} != self last dim {cols}"
    # Per-row means the mask is constant across the last dim. Accept a literal
    # size-1 last dim (un-expanded) or a broadcast last dim (stride 0); reject a
    # genuinely per-element mask (last dim cols with a non-zero stride).
    if mask.shape[-1] != 1 and mask.stride(-1) != 0:
        return (
            f"mask is not per-row in its last dim (shape {tuple(mask.shape)}, "
            f"stride {tuple(mask.stride())})"
        )
    return None


@register_spyre_decompositions([torch.ops.aten.masked_scatter.default])
def spyre_masked_scatter(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """`masked_scatter` for a mask that is broadcast along the last dim.

    Such a mask (`stride(-1) == 0` -- e.g. an attention mask `[B, S]`
    expanded to `[B, S, C]`) selects *whole rows*: row `i`, if selected,
    consumes exactly one contiguous `C`-element block of `source`, i.e. one
    whole row of `source.reshape(-1, C)`. The gather is then
    `source_2d[row_idx]` -- a 1D index into the row dim of a 2D source, which
    is a plain stick gather.

    Any other mask makes this an element-level op, which Spyre cannot express:
    the index would have to address an element inside a *packed* 1D source, and
    a lane within a stick is not addressable. Exploding the source to one
    element per stick does not help -- that splits every stick 64 ways, an
    element scatter the backend cannot lower. So the generic form is rejected
    here rather than emitting a gather that fails deeper in layout propagation.
    """
    reason = _masked_scatter_reject_reason(self, mask, source)
    if reason is not None:
        raise Unsupported(
            f"masked_scatter needs a mask broadcast along the last dim so it "
            f"selects whole rows ({reason}). An element-level masked_scatter "
            f"would gather individual elements from a packed 1D source, and a "
            f"lane within a stick is not addressable on Spyre."
        )

    cols = self.shape[-1]
    rows = self.numel() // cols
    # Collapse the broadcast dim: one bool per row.
    mask_row = mask[..., 0].reshape(rows)
    source_2d = source.reshape(-1, cols)
    # masked_scatter requires source.numel() >= mask.sum(). For a whole-row
    # mask, source_2d needs at least as many rows as there are selected rows.
    torch._assert_async(
        mask_row.sum() <= source_2d.shape[0],
        "masked_scatter: source is too short -- it has fewer rows than the "
        "number of selected (True) mask rows.",
    )
    # Row i reads source row (prefix-count of selected rows - 1). Unselected
    # rows would get -1, so multiply by the mask to send them to row 0 instead
    # (in bounds, and discarded by the where). Done in fp32: Spyre has no usable
    # int `clip`, int32 sub/mul are unsupported, and fp16 loses large indices.
    pos = mask_row.cumsum(0).to(torch.float32) - 1.0
    row_idx = (pos * mask_row.to(torch.float32)).to(torch.int64)
    # The gather stays 2D (its args are indirect, so the pointwise dim_order
    # projection skips them), but the `where` must stay at `self`'s rank:
    # that projection computes `rank_diff = len(output) - len(arg)` and only
    # handles inputs of *lower* rank. A lower-rank output against the full-rank
    # ND mask gives rank_diff < 0, which shifts dims the wrong way and builds a
    # layout whose dim_order rank no longer matches host_size ("Incompatible
    # host_size and dim_order"). Reshaping back to ND keeps every `where`
    # input rank-aligned with the output.
    gathered = source_2d[row_idx].reshape(self.shape)
    return torch.where(mask, gathered, self)


@register_spyre_decompositions([torch.ops.aten.index_add.default])
def spyre_index_add(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    *,
    alpha: Union[int, float] = 1,
) -> torch.Tensor:
    """`index_add` as gather + add + overwrite-scatter, fully on device.

    `out.index_add_(dim, index, source * alpha)` is a read-modify-write:
    read the current values at the target slots, add the (scaled) source, and
    write them back, using primitives Spyre runs on the indirect-access engine:

      * `index_select`  -> on-device indirect gather
      * `index_put`     -> on-device indirect overwrite store

    PRECONDITION -- `index` must contain NO DUPLICATE values. A read-modify-
    write cannot sum colliding writes: every duplicate reads the same old value
    and the overwrite store keeps only the last writer, so duplicate indices are
    SILENTLY WRONG.
    """
    dim = dim % self.dim()
    if alpha != 1:
        source = source * alpha
    # Read current destination values, then add the source onto them.
    gathered = torch.index_select(self, dim, index)
    updated = gathered + source
    indices: list[Optional[torch.Tensor]] = [None] * dim + [index]
    return torch.index_put(self, indices, updated, accumulate=False)


@register_spyre_decompositions([torch.ops.aten.triu.default])
def spyre_triu(
    input: torch.Tensor,
    diagonal: int = 0,
) -> torch.Tensor:
    # Narrow integral dtypes have no on-device elementwise masking op (mul and
    # where are both rejected for SENUINT32/SENINT8/IEEE_INT32), so keep them
    # on the CPU fallback that aten.triu used before this decomposition existed.
    if not (input.dtype.is_floating_point or input.dtype == torch.bool):
        return torch.ops.spyre.triu_cpu(input, diagonal)

    h, w = input.shape[-2], input.shape[-1]

    # spyre::triu_mask builds [H,W] upper triangular mask on CPU and
    # transfers it to the input device. Wrapping this in a custom op keeps the
    # CPU-side construction opaque to torch.compile, so assert_functional_graph
    # is satisfied and the compiled graph sees only the resulting Spyre tensor.
    mask = torch.ops.spyre.triu_mask(h, w, diagonal, input.dtype, input.device)
    # Masking +/-inf this way works on device but would produce NaN under IEEE
    return input * mask
