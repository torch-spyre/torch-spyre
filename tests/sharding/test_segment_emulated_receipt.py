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

"""Tests for 1p5-over-1p0 segment-emulated receipt validation."""

from __future__ import annotations

import copy

import pytest

from torch_spyre._native_placement import (
    DISPATCHABLE_TENSOR_SEGMENTS,
    sharded_receipt_from_native,
)

_CHUNK_BYTES = 256


def _native(count=8, chunk_bytes=_CHUNK_BYTES):
    chunks = []
    for chunk_id in range(count):
        segment = DISPATCHABLE_TENSOR_SEGMENTS[chunk_id % 6]
        chunks.append(
            {
                "logical_chunk_id": chunk_id,
                "logical_core_id": chunk_id,
                "physical_domain_id": 0,
                "region_id": 10 + segment,
                "segment_id": segment,
                "segment_offset_bytes": (chunk_id // 6) * chunk_bytes,
                "size_bytes": chunk_bytes,
                "is_device_addressable": True,
            }
        )
    return chunks


def _receipt(native, count=8, chunk_bytes=_CHUNK_BYTES):
    return sharded_receipt_from_native(native, count, chunk_bytes, count * chunk_bytes)


@pytest.mark.parametrize("count", [8, 16])
def test_valid_receipt_is_accepted(count):
    receipt = _receipt(_native(count), count)

    assert receipt.num_shards == count
    assert receipt.shard_device_bytes == _CHUNK_BYTES
    assert receipt.total_bytes == count * _CHUNK_BYTES
    assert [e.shard_ordinal for e in receipt.entries] == list(range(count))
    assert [e.physical["segment_id"] for e in receipt.entries] == [
        DISPATCHABLE_TENSOR_SEGMENTS[i % 6] for i in range(count)
    ]


def test_logical_identity_is_distinct_from_physical_segment():
    receipt = _receipt(_native(8))

    entry6 = receipt.entries[6].physical
    entry0 = receipt.entries[0].physical
    assert entry6["segment_id"] == entry0["segment_id"] == 1
    assert receipt.entries[6].core_id == 6
    assert entry6["segment_offset_bytes"] != entry0["segment_offset_bytes"]


def test_non_default_disjoint_packing_is_accepted():
    # The physical segment assignment is the Flex allocator's choice (plain
    # per-shard allocations co-locate as the free list dictates), so any
    # dispatchable segment is valid as long as extents stay disjoint.
    native = _native(8)
    native[1]["segment_id"] = 3
    native[1]["segment_offset_bytes"] = native[3]["segment_offset_bytes"] + _CHUNK_BYTES
    receipt = _receipt(native)

    assert receipt.entries[1].physical["segment_id"] == 3


def test_shard_ordinal_lookup_uses_logical_identity_not_position():
    native = _native(8)
    native.reverse()
    receipt = _receipt(native)

    assert receipt.entries[0].shard_ordinal == 7
    assert {e.shard_ordinal for e in receipt.entries} == set(range(8))


def _mutate(path, value):
    native = _native(8)
    target = native
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return native


@pytest.mark.parametrize(
    "native,error,message",
    [
        (_mutate((0, "logical_chunk_id"), 3), ValueError, "0..N-1"),
        (_mutate((2, "logical_core_id"), 0), ValueError, "core ID"),
        (_mutate((1, "segment_id"), 0), ValueError, "dispatchable"),
        (_mutate((1, "segment_id"), 7), ValueError, "dispatchable"),
        (_mutate((3, "is_device_addressable"), False), ValueError, "addressable"),
        (_mutate((3, "is_device_addressable"), 1), TypeError, "bool"),
        (_mutate((4, "size_bytes"), 128), ValueError, "equal-sized"),
        (_mutate((4, "segment_offset_bytes"), 64), ValueError, "aligned"),
        (_mutate((4, "segment_offset_bytes"), -128), ValueError, "negative"),
        (_mutate((6, "segment_offset_bytes"), 0), ValueError, "overlap"),
        (_mutate((5, "region_id"), "r"), TypeError, "concrete int"),
    ],
)
def test_malformed_receipts_fail_closed(native, error, message):
    with pytest.raises(error, match=message):
        _receipt(native)


def test_missing_fields_fail_closed():
    native = _native(8)
    del native[0]["segment_id"]
    with pytest.raises(ValueError, match="segment_id"):
        _receipt(native)


def test_wrong_chunk_count_fails_closed():
    native = _native(8)
    native.pop()
    with pytest.raises(ValueError, match="one record per logical chunk"):
        _receipt(native, count=8)


def test_non_sequence_chunks_fail_closed():
    with pytest.raises(TypeError, match="sequence"):
        _receipt("chunks")


def test_wrong_total_bytes_fails_closed():
    with pytest.raises(ValueError, match="total_bytes"):
        sharded_receipt_from_native(_native(8), 8, _CHUNK_BYTES, 8 * _CHUNK_BYTES + 1)


def test_native_chunks_are_not_mutated_by_validation():
    native = _native(8)
    snapshot = copy.deepcopy(native)
    _receipt(native)
    assert native == snapshot


# ---------------------------------------------------------------------------
# Mapping to the generation-neutral receipt.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [8, 16])
def test_neutral_receipt_maps_every_field(count):
    """The adapter must carry identity, size, and physical fields verbatim.

    A typo'd physical key or a swapped segment/region id here would otherwise
    be invisible to every CPU test (the device tests assert only the neutral
    identity fields).
    """

    native = _native(count)
    receipt = _receipt(native, count)

    assert receipt.num_shards == count
    assert receipt.shard_device_bytes == _CHUNK_BYTES
    assert receipt.total_bytes == count * _CHUNK_BYTES
    assert [e.shard_ordinal for e in receipt.entries] == list(range(count))
    assert [e.core_id for e in receipt.entries] == list(range(count))
    assert {e.byte_length for e in receipt.entries} == {_CHUNK_BYTES}
    for native_chunk, entry in zip(native, receipt.entries):
        assert entry.physical == {
            "physical_domain_id": native_chunk["physical_domain_id"],
            "region_id": native_chunk["region_id"],
            "segment_id": native_chunk["segment_id"],
            "segment_offset_bytes": native_chunk["segment_offset_bytes"],
        }
