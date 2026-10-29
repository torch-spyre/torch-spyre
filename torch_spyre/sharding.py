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

"""Decision-independent semantic shard geometry for equal-shard planning."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch.distributed.tensor import Shard

SUPPORTED_SHARD_COUNTS = (8, 16)


class ShardPlanningUnavailableError(RuntimeError):
    """The supported PyTorch shard-planning API is unavailable or incompatible."""


@dataclasses.dataclass(frozen=True, slots=True)
class ShardRange:
    """One immutable logical range along a tensor's sharded dimension."""

    shard_ordinal: int
    tensor_dim: int
    logical_offset: int
    logical_extent: int


def _require_exact_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a concrete int")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ShardChunkBinding:
    """One semantic shard bound to its backend placement slot and core.

    The physical placement of the slot is not recorded here; it comes back
    from the backend in the allocation receipt and is joined by shard
    ordinal.
    """

    shard_range: ShardRange
    placement_slot_id: int
    core_id: int


@dataclasses.dataclass(frozen=True, slots=True)
class ShardPlacementEntry:
    """One semantic shard's generation-neutral placement record.

    ``physical`` carries backend-specific placement fields opaquely (for
    example segment, region, or HBM-group identity). It is inspection-only
    and is never part of the shard's semantic identity.
    """

    shard_ordinal: int
    core_id: int
    byte_length: int
    is_device_addressable: bool
    physical: Mapping[str, int]


@dataclasses.dataclass(frozen=True, slots=True)
class ShardedReceipt:
    """Generation-neutral placement receipt for one sharded allocation."""

    num_shards: int
    shard_device_bytes: int
    total_bytes: int
    entries: tuple[ShardPlacementEntry, ...]

    def entry(self, shard_ordinal: int) -> ShardPlacementEntry:
        """Return the entry for this shard ordinal (not vector position)."""

        for candidate in self.entries:
            if candidate.shard_ordinal == shard_ordinal:
                return candidate
        raise KeyError(shard_ordinal)


def validate_sharded_receipt(
    receipt: ShardedReceipt, num_shards: int, shard_device_bytes: int
) -> None:
    """Check the neutral placement invariants every backend must uphold.

    One entry per shard with unique ordinals, one core per shard, equal
    device-addressable shard extents, and a total that covers the request.
    Anything backend-specific (physical fields) is not interpreted here.
    """

    num_shards = _require_exact_int("num_shards", num_shards)
    if num_shards <= 0:
        raise ValueError("placement receipt must describe at least one shard")
    if len(receipt.entries) != num_shards:
        raise ValueError("placement receipt must contain one entry per shard")
    if {entry.shard_ordinal for entry in receipt.entries} != set(range(num_shards)):
        raise ValueError("placement receipt shard ordinals must be exactly 0..N-1")
    if len({entry.core_id for entry in receipt.entries}) != num_shards:
        raise ValueError("placement receipt must bind one core per shard")
    for entry in receipt.entries:
        if entry.byte_length != shard_device_bytes:
            raise ValueError("placement receipt shards must be equal-sized")
        if entry.byte_length <= 0:
            raise ValueError("placement receipt shard byte_length must be positive")
        if not entry.is_device_addressable:
            raise ValueError(f"shard {entry.shard_ordinal} is not device addressable")
    if receipt.total_bytes < shard_device_bytes * num_shards:
        raise ValueError("placement receipt total must cover the requested bytes")


@dataclasses.dataclass(frozen=True, slots=True)
class SpyreShardingSpec:
    """Internal sharding layout: one ``Shard`` over N placement slots.

    Users call ``torch.spyre.distribute_tensor(x, Shard(d))``. This
    backend-owned record is kept so future hardware can attach topology /
    placement policy (for example a non-identity slot-to-core map) without
    changing the public API.
    """

    placement: Shard
    num_shards: int

    def __post_init__(self) -> None:
        from torch.distributed.tensor import Shard

        if type(self.placement) is not Shard:
            raise TypeError("placement must be exactly torch.distributed.tensor.Shard")
        _require_shard_count(self.num_shards)

    def plan(self, global_shape: Sequence[int]) -> tuple[ShardRange, ...]:
        """Return the admitted semantic shard plan for ``global_shape``."""

        return plan_equal_shards(global_shape, self.placement, self.num_shards)

    def bindings(self, global_shape: Sequence[int]) -> tuple[ShardChunkBinding, ...]:
        """Bind each semantic shard to its placement slot and core (identity)."""

        return tuple(
            ShardChunkBinding(
                shard_range=shard_range,
                placement_slot_id=shard_range.shard_ordinal,
                core_id=shard_range.shard_ordinal,
            )
            for shard_range in self.plan(global_shape)
        )


def _require_supported_helper():
    from torch.distributed.tensor import Shard

    helper = getattr(Shard, "local_shard_size_and_offset", None)
    if not callable(helper):
        raise ShardPlanningUnavailableError(
            "PyTorch Shard.local_shard_size_and_offset is unavailable"
        )
    return helper


def compute_shard_range(
    placement: Shard,
    tensor_rank: int,
    global_extent: int,
    shard_count: int,
    shard_ordinal: int,
) -> ShardRange:
    """Return one helper-derived logical range without runtime side effects."""

    from torch.distributed.tensor import Shard

    if type(placement) is not Shard:
        raise TypeError("placement must be exactly torch.distributed.tensor.Shard")
    tensor_rank = _require_exact_int("tensor_rank", tensor_rank)
    global_extent = _require_exact_int("global_extent", global_extent)
    shard_count = _require_exact_int("shard_count", shard_count)
    shard_ordinal = _require_exact_int("shard_ordinal", shard_ordinal)
    if tensor_rank <= 0:
        raise ValueError("tensor_rank must be positive")
    if global_extent < 0:
        raise ValueError("global_extent must be nonnegative")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_ordinal < shard_count:
        raise ValueError("shard_ordinal must be within shard_count")

    dim = placement.dim
    if type(dim) is not int:
        raise TypeError("Shard.dim must be a concrete int")
    if not -tensor_rank <= dim < tensor_rank:
        raise ValueError("Shard.dim is outside the tensor rank")
    normalized_dim = dim % tensor_rank

    helper = _require_supported_helper()
    try:
        result = helper(global_extent, shard_count, shard_ordinal)
    except Exception as error:
        raise ShardPlanningUnavailableError(
            "PyTorch shard range helper failed"
        ) from error
    if type(result) is not tuple or len(result) != 2:
        raise ShardPlanningUnavailableError(
            "PyTorch shard range helper returned an incompatible result"
        )
    logical_extent, logical_offset = result
    if type(logical_extent) is not int or type(logical_offset) is not int:
        raise ShardPlanningUnavailableError(
            "PyTorch shard range helper returned non-concrete geometry"
        )
    if logical_extent < 0 or logical_offset < 0:
        raise ShardPlanningUnavailableError(
            "PyTorch shard range helper returned negative geometry"
        )
    if logical_offset > global_extent or logical_extent > global_extent:
        raise ShardPlanningUnavailableError(
            "PyTorch shard range helper returned out-of-range geometry"
        )
    if logical_offset + logical_extent > global_extent:
        raise ShardPlanningUnavailableError(
            "PyTorch shard range helper returned a range past the global extent"
        )
    return ShardRange(
        shard_ordinal=shard_ordinal,
        tensor_dim=normalized_dim,
        logical_offset=logical_offset,
        logical_extent=logical_extent,
    )


def _require_shard_count(shard_count: object) -> int:
    shard_count = _require_exact_int("num_shards", shard_count)
    if shard_count not in SUPPORTED_SHARD_COUNTS:
        raise ValueError("num_shards must be 8 or 16")
    return shard_count


def plan_equal_shards(
    global_shape: Sequence[int],
    placement: Shard,
    shard_count: int,
) -> tuple[ShardRange, ...]:
    """Admit exactly ``shard_count`` (8 or 16) equal, non-empty shard ranges."""

    try:
        shape = tuple(global_shape)
    except TypeError as error:
        raise TypeError("global_shape must be a concrete sequence") from error
    if not shape:
        raise ValueError("scalar shapes are not supported")
    for extent in shape:
        if type(extent) is not int:
            raise TypeError("global_shape entries must be concrete ints")
        if extent <= 0:
            raise ValueError("global_shape entries must be positive")
    # Placement type is re-proven per ordinal by compute_shard_range, but
    # placement.dim is read below before that, so the type must hold here.
    # Only the dim range check is left to compute_shard_range.
    from torch.distributed.tensor import Shard

    if type(placement) is not Shard:
        raise TypeError("placement must be exactly torch.distributed.tensor.Shard")
    dim = placement.dim
    if type(dim) is not int:
        raise TypeError("Shard.dim must be a concrete int")
    normalized_dim = dim % len(shape)
    shard_count = _require_shard_count(shard_count)

    ranges = tuple(
        compute_shard_range(
            placement,
            len(shape),
            shape[normalized_dim],
            shard_count,
            shard_ordinal,
        )
        for shard_ordinal in range(shard_count)
    )
    expected_extent = ranges[0].logical_extent
    if expected_extent <= 0:
        raise ValueError("shards must be non-empty")
    expected_offset = 0
    for shard_range in ranges:
        if shard_range.tensor_dim != normalized_dim:
            raise ValueError("shard dimensions are inconsistent")
        if shard_range.logical_extent != expected_extent:
            raise ValueError("shards must be equal")
        if shard_range.logical_offset != expected_offset:
            raise ValueError("shards must be ordered and gap-free")
        expected_offset += shard_range.logical_extent
    if expected_offset != shape[normalized_dim]:
        raise ValueError("shards must cover the global extent exactly")
    return ranges
