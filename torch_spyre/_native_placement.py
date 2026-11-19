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

"""1p5-over-1p0 segment-emulated Flex placement receipts.

This module is the single generation-aware adapter in torch-spyre: it maps the
native Flex per-chunk placement records (produced by ``_C._spyre_sharding_info``)
to the generation-neutral ``ShardedReceipt`` that the rest of torch-spyre
consumes. Everything above it (sharding, the public API, the compiler) must be
able to run against any backend that honors the neutral receipt; moving this
adapter into Flex is planned; when that happens, this file is deleted
outright.

On Sentient 1p0 there is one physical HBM domain. Flex emulates the 1p5
chunk/core contract by placing each logical chunk into one of the six
dispatchable Tensor segments (1..6). Logical chunk/core identity is therefore
distinct from physical segment identity, and the receipt records both.
"""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from itertools import pairwise

from torch_spyre.sharding import (
    ShardedReceipt,
    ShardPlacementEntry,
    _require_exact_int,
    validate_sharded_receipt,
)

SUPPORTED_LOGICAL_CHUNK_COUNTS = (8, 16)
DISPATCHABLE_TENSOR_SEGMENTS = (1, 2, 3, 4, 5, 6)
DEVICE_ALIGNMENT_BYTES = 128


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a concrete bool")
    return value


def _field(record: Mapping[str, object], name: str, where: str) -> object:
    try:
        return record[name]
    except KeyError as error:
        raise ValueError(f"{where} is missing {name!r}") from error


def _require_logical_chunk_count(value: object) -> int:
    count = _require_exact_int("logical_chunks", value)
    if count not in SUPPORTED_LOGICAL_CHUNK_COUNTS:
        raise ValueError("segment emulation supports 8 or 16 logical chunks")
    return count


def _ensure_spyre_runtime_initialized() -> None:
    """Initialize the public torch.spyre runtime before native allocation."""

    import torch

    # torch.spyre is registered by this package's own autoload before any
    # code here can run, so both attributes exist unconditionally.
    if not torch.spyre.is_initialized():
        torch.spyre._lazy_init()


def sharded_receipt_from_native(
    native_chunks: object,
    num_shards: int,
    shard_device_bytes: int,
    total_bytes: int,
) -> ShardedReceipt:
    """Validate the native per-chunk placement records and build the receipt.

    Fails closed on any malformed, partial, or inconsistent placement. The
    per-segment overlap check is performed here only — the native allocator
    does not prove extents disjoint.
    """

    count = _require_logical_chunk_count(num_shards)
    if not isinstance(native_chunks, Sequence) or isinstance(native_chunks, str):
        raise TypeError("native chunks must be a concrete sequence")
    if len(native_chunks) != count:
        raise ValueError("receipt must contain exactly one record per logical chunk")

    # Identity coverage is proven up front (matching the historical check
    # order) so a mutated chunk id reports the coverage failure, not the
    # identity-map failure it would otherwise trigger below.
    ordinals = set()
    for native in native_chunks:
        if not isinstance(native, Mapping):
            raise TypeError("segment-emulated chunks must be mappings")
        ordinals.add(
            _require_exact_int(
                "logical_chunk_id",
                _field(native, "logical_chunk_id", "segment-emulated chunk"),
            )
        )
    if ordinals != set(range(count)):
        raise ValueError("receipt logical chunk IDs must be exactly 0..N-1")

    entries = []
    extents_by_segment: dict[int, list[tuple[int, int]]] = {}
    for position, native in enumerate(native_chunks):
        where = f"segment-emulated chunk at position {position}"
        if not isinstance(native, Mapping):
            raise TypeError(f"{where} must be a mapping")

        def integer(name: str) -> int:
            return _require_exact_int(name, _field(native, name, where))

        logical_chunk_id = integer("logical_chunk_id")
        logical_core_id = integer("logical_core_id")
        physical_domain_id = integer("physical_domain_id")
        region_id = integer("region_id")
        segment_id = integer("segment_id")
        segment_offset_bytes = integer("segment_offset_bytes")
        size_bytes = integer("size_bytes")
        is_device_addressable = _require_exact_bool(
            "is_device_addressable", _field(native, "is_device_addressable", where)
        )

        if logical_core_id != logical_chunk_id:
            raise ValueError("logical core ID must equal logical chunk ID")
        if not is_device_addressable:
            raise ValueError(f"chunk {logical_chunk_id} is not device addressable")
        if segment_id not in DISPATCHABLE_TENSOR_SEGMENTS:
            raise ValueError(
                f"chunk {logical_chunk_id} is not in a dispatchable Tensor segment"
            )
        if size_bytes <= 0 or size_bytes != shard_device_bytes:
            raise ValueError("segment-emulated chunks must be equal-sized")
        if size_bytes % DEVICE_ALIGNMENT_BYTES:
            raise ValueError(f"chunk {logical_chunk_id} size is not device aligned")
        if segment_offset_bytes < 0:
            raise ValueError(f"chunk {logical_chunk_id} has a negative segment offset")
        if segment_offset_bytes % DEVICE_ALIGNMENT_BYTES:
            raise ValueError(f"chunk {logical_chunk_id} offset is not device aligned")
        # The physical segment assignment is the Flex allocator's choice: with
        # plain per-shard allocations, shards co-locate across the dispatchable
        # segments however the free list dictates (e.g. 2,2,1,1,1,1 for eight
        # shards). Only disjointness within a segment is required, checked
        # below via the extent map.
        extents_by_segment.setdefault(segment_id, []).append(
            (segment_offset_bytes, segment_offset_bytes + size_bytes)
        )
        entries.append(
            ShardPlacementEntry(
                shard_ordinal=logical_chunk_id,
                core_id=logical_core_id,
                byte_length=size_bytes,
                is_device_addressable=is_device_addressable,
                # Read-only view: the receipt is documented as immutable.
                physical=types.MappingProxyType(
                    {
                        "physical_domain_id": physical_domain_id,
                        "region_id": region_id,
                        "segment_id": segment_id,
                        "segment_offset_bytes": segment_offset_bytes,
                    }
                ),
            )
        )

    for extents in extents_by_segment.values():
        extents.sort()
        for (_, prev_end), (start, _) in pairwise(extents):
            if start < prev_end:
                raise ValueError("chunks overlap within a physical segment")

    if total_bytes != shard_device_bytes * count:
        raise ValueError("total_bytes must equal the sum of chunk sizes")

    neutral = ShardedReceipt(
        num_shards=count,
        shard_device_bytes=shard_device_bytes,
        total_bytes=total_bytes,
        entries=tuple(entries),
    )
    validate_sharded_receipt(neutral, num_shards, shard_device_bytes)
    return neutral
