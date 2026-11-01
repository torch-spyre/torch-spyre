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

"""Tests for the internal sharding layout and shard/slot bindings."""

from __future__ import annotations

import dataclasses

import pytest
from torch.distributed.tensor import Replicate, Shard

from torch_spyre.sharding import (
    ShardRange,
    SUPPORTED_SHARD_COUNTS,
    ShardChunkBinding,
    ShardedReceipt,
    ShardPlacementEntry,
    SpyreShardingSpec,
    plan_equal_shards,
    validate_sharded_receipt,
)


def test_supported_shard_counts_are_the_slot_core_shapes():
    assert SUPPORTED_SHARD_COUNTS == (8, 16)


def test_spec_is_immutable():
    spec = SpyreShardingSpec(Shard(0), 8)

    assert spec.placement == Shard(0)
    assert spec.num_shards == 8
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.num_shards = 16


@pytest.mark.parametrize("num_shards", SUPPORTED_SHARD_COUNTS)
def test_bindings_map_each_shard_to_matching_slot_and_core(num_shards):
    spec = SpyreShardingSpec(Shard(0), num_shards)
    bindings = spec.bindings((num_shards * 2, 3))

    assert type(bindings) is tuple
    assert len(bindings) == num_shards
    for ordinal, binding in enumerate(bindings):
        assert type(binding) is ShardChunkBinding
        assert type(binding.shard_range) is ShardRange
        assert binding.shard_range.shard_ordinal == ordinal
        assert binding.shard_range.logical_offset == ordinal * 2
        assert binding.shard_range.logical_extent == 2
        assert binding.placement_slot_id == ordinal
        assert binding.core_id == ordinal


def test_bindings_carry_no_physical_placement():
    fields = {field.name for field in dataclasses.fields(ShardChunkBinding)}

    assert fields == {"shard_range", "placement_slot_id", "core_id"}


@pytest.mark.parametrize("num_shards", SUPPORTED_SHARD_COUNTS)
def test_plan_matches_planner(num_shards):
    spec = SpyreShardingSpec(Shard(-1), num_shards)

    assert spec.plan((3, 32)) == plan_equal_shards((3, 32), Shard(-1), num_shards)


@pytest.mark.parametrize("placement", [Replicate(), object(), None])
def test_placement_must_be_exact_shard(placement):
    with pytest.raises(TypeError, match="Shard"):
        SpyreShardingSpec(placement, 8)


@pytest.mark.parametrize("num_shards", [8.0, "8", True, None])
def test_shard_count_must_be_concrete_int(num_shards):
    with pytest.raises(TypeError, match="concrete int"):
        SpyreShardingSpec(Shard(0), num_shards)


def test_sixteen_shards_require_divisible_extent():
    with pytest.raises(ValueError, match="equal|cover|non-empty"):
        SpyreShardingSpec(Shard(0), 16).plan((24,))


# ---------------------------------------------------------------------------
# Generation-neutral placement receipt.
#
# These tests document the neutral contract directly: the segment-emulated
# adapter pre-empts every one of these failure modes with stricter checks of
# its own, so a future backend is the caller that relies on them.
# ---------------------------------------------------------------------------


def _receipt(
    num_shards=8,
    shard_bytes=128,
    *,
    ordinals=None,
    cores=None,
    lengths=None,
    addressable=None,
    total_bytes=None,
):
    """Build a ShardedReceipt with per-entry overrides for failure cases.

    Note: ``zip`` truncates to the shortest override list; the entry-count
    test below relies on that (7-long ordinals/cores vs 8-long defaults).
    """

    ordinals = ordinals if ordinals is not None else list(range(num_shards))
    cores = cores if cores is not None else list(range(num_shards))
    lengths = lengths if lengths is not None else [shard_bytes] * num_shards
    addressable = addressable if addressable is not None else [True] * num_shards
    entries = tuple(
        ShardPlacementEntry(
            shard_ordinal=ordinal,
            core_id=core,
            byte_length=length,
            is_device_addressable=flag,
            physical={"segment_id": ordinal % 6 + 1},
        )
        for ordinal, core, length, flag in zip(ordinals, cores, lengths, addressable)
    )
    if total_bytes is None:
        total_bytes = shard_bytes * num_shards
    return ShardedReceipt(
        num_shards=num_shards,
        shard_device_bytes=shard_bytes,
        total_bytes=total_bytes,
        entries=entries,
    )


@pytest.mark.parametrize("num_shards", SUPPORTED_SHARD_COUNTS)
def test_valid_receipt_passes_neutral_validation(num_shards):
    receipt = _receipt(num_shards)

    validate_sharded_receipt(receipt, num_shards, 128)

    assert receipt.entry(0).shard_ordinal == 0
    assert receipt.entry(num_shards - 1).core_id == num_shards - 1
    with pytest.raises(KeyError):
        receipt.entry(num_shards)


def test_receipt_entry_count_must_match_shard_count():
    receipt = _receipt(ordinals=list(range(7)), cores=list(range(7)))

    with pytest.raises(ValueError, match="one entry per shard"):
        validate_sharded_receipt(receipt, 8, 128)


def test_receipt_ordinals_must_cover_zero_to_n():
    receipt = _receipt(ordinals=list(range(1, 9)), cores=list(range(8)))

    with pytest.raises(ValueError, match="0..N-1"):
        validate_sharded_receipt(receipt, 8, 128)


def test_receipt_needs_one_core_per_shard():
    receipt = _receipt(cores=[0] * 8)

    with pytest.raises(ValueError, match="one core per shard"):
        validate_sharded_receipt(receipt, 8, 128)


def test_receipt_shards_must_be_equal_sized():
    receipt = _receipt(lengths=[128] * 7 + [256])

    with pytest.raises(ValueError, match="equal-sized"):
        validate_sharded_receipt(receipt, 8, 128)


def test_receipt_shards_must_be_device_addressable():
    receipt = _receipt(addressable=[True] * 7 + [False])

    with pytest.raises(ValueError, match="device addressable"):
        validate_sharded_receipt(receipt, 8, 128)


def test_receipt_total_must_cover_the_request():
    receipt = _receipt(total_bytes=128 * 7)

    with pytest.raises(ValueError, match="cover the requested"):
        validate_sharded_receipt(receipt, 8, 128)


def test_receipt_rejects_zero_shards():
    with pytest.raises(ValueError, match="at least one shard"):
        validate_sharded_receipt(_receipt(0), 0, 128)


def test_receipt_rejects_nonpositive_shard_bytes():
    receipt = _receipt(shard_bytes=-128, lengths=[-128] * 8, total_bytes=-128 * 8)

    with pytest.raises(ValueError, match="positive"):
        validate_sharded_receipt(receipt, 8, -128)
