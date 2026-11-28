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

"""How placements combine in a pointwise op (no device needed).

``None`` is a plain unified tensor or a scalar. Only the single-copy
``Replicate()`` read path exists, so a replica with any other copy count must
fail closed instead of compiling as if it were one unified copy.
"""

from __future__ import annotations

import dataclasses

import pytest

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.sharded_hbm import (
    ReplicatedPlacement,
    ShardedPlacement,
    combine_placements,
)

S0 = ShardedPlacement(shard_dim=0, num_shards=8, shard_device_bytes=4096)
S1 = ShardedPlacement(shard_dim=1, num_shards=8, shard_device_bytes=4096)
S0_16 = ShardedPlacement(shard_dim=0, num_shards=16, shard_device_bytes=2048)
R = ReplicatedPlacement()


def test_replica_defaults_to_one_unified_copy():
    assert R.copies == 1


def test_replica_is_frozen_and_hashable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        R.copies = 8  # type: ignore[misc]
    assert hash(R) == hash(ReplicatedPlacement(copies=1))


@pytest.mark.parametrize("other", [None, R])
@pytest.mark.parametrize("flip", [False, True])
def test_shard_with_unsharded_operand_keeps_the_shard(other, flip):
    operands = [other, S0] if flip else [S0, other]
    assert combine_placements("buf", operands) == S0


def test_shard_with_replica_and_plain_and_scalar_keeps_the_shard():
    assert combine_placements("buf", [S0, R, None, None]) == S0


def test_same_shard_twice_is_that_shard():
    assert combine_placements("buf", [S0, S0]) == S0


@pytest.mark.parametrize("operands", [[], [None], [R], [R, R], [R, None], [None, R]])
def test_no_sharded_operand_is_an_ordinary_unified_op(operands):
    assert combine_placements("buf", operands) is None


def test_different_shard_dims_fail_closed_naming_both():
    with pytest.raises(Unsupported) as error:
        combine_placements("buf0", [S0, S1])
    message = str(error.value)
    assert "sharded differently" in message
    assert "Shard(0)" in message and "Shard(1)" in message


def test_different_shard_counts_fail_closed_naming_both():
    with pytest.raises(Unsupported, match="sharded differently") as error:
        combine_placements("buf0", [S0, S0_16])
    assert "8 shards" in str(error.value) and "16 shards" in str(error.value)


def test_replica_with_other_copy_count_fails_closed():
    with pytest.raises(Unsupported, match="only as one unified copy"):
        combine_placements("buf", [S0, ReplicatedPlacement(copies=8)])


def test_per_core_replica_without_a_sharded_operand_is_not_a_sharded_op():
    # Nothing sharded: no sharded kernel is planned, so nothing to reject here.
    assert combine_placements("buf", [ReplicatedPlacement(copies=8)]) is None
