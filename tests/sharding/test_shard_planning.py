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

"""Tests for decision-independent semantic shard geometry."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from side_effect_utils import forbid_distributed
from torch_spyre.sharding import (
    ShardPlanningUnavailableError,
    ShardRange,
    compute_shard_range,
    plan_equal_shards,
)


class _IntSubclass(int):
    pass


class _Indexable:
    def __index__(self):
        return 1


class _ShardSubclass(Shard):
    pass


@pytest.mark.parametrize("extent", [0, 3, 8, 10, 16])
@pytest.mark.parametrize("ordinal", range(8))
def test_adapter_matches_public_helper(extent, ordinal):
    placement = Shard(0)
    expected_extent, expected_offset = placement.local_shard_size_and_offset(
        extent, 8, ordinal
    )
    actual = compute_shard_range(placement, 2, extent, 8, ordinal)
    assert (actual.logical_extent, actual.logical_offset) == (
        expected_extent,
        expected_offset,
    )


@pytest.mark.parametrize("dim,normalized", [(0, 0), (1, 1), (-1, 1), (-2, 0)])
def test_dimension_normalization(dim, normalized):
    actual = compute_shard_range(Shard(dim), 2, 16, 8, 0)
    assert actual.tensor_dim == normalized


@pytest.mark.parametrize("dim", [-3])
def test_dimension_rejected_before_normalization(dim):
    with pytest.raises(ValueError, match="outside"):
        compute_shard_range(Shard(dim), 2, 16, 8, 0)


def test_scalar_rank_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        compute_shard_range(Shard(0), 0, 16, 8, 0)


@pytest.mark.parametrize(
    "argument,value",
    [
        ("tensor_rank", True),
        ("global_extent", False),
        ("shard_count", True),
        ("shard_ordinal", False),
        ("tensor_rank", _IntSubclass(2)),
        ("global_extent", _IntSubclass(16)),
        ("shard_count", _Indexable()),
        ("shard_ordinal", _Indexable()),
    ],
)
def test_adapter_rejects_non_exact_integers(argument, value):
    arguments = {
        "placement": Shard(0),
        "tensor_rank": 2,
        "global_extent": 16,
        "shard_count": 8,
        "shard_ordinal": 0,
    }
    arguments[argument] = value
    with pytest.raises(TypeError, match="concrete int"):
        compute_shard_range(**arguments)


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"global_extent": -1}, "nonnegative"),
        ({"shard_count": 0}, "positive"),
        ({"shard_count": -1}, "positive"),
        ({"shard_ordinal": -1}, "within"),
        ({"shard_ordinal": 8}, "within"),
    ],
)
def test_adapter_rejects_invalid_geometry(arguments, message):
    values = {
        "placement": Shard(0),
        "tensor_rank": 2,
        "global_extent": 16,
        "shard_count": 8,
        "shard_ordinal": 0,
    }
    values.update(arguments)
    with pytest.raises(ValueError, match=message):
        compute_shard_range(**values)


@pytest.mark.parametrize(
    "placement",
    [
        _ShardSubclass(0),
        _StridedShard(0, split_factor=2),
        Replicate(),
        Partial(),
        [Shard(0)],
        (Shard(0),),
    ],
)
def test_exact_shard_placement_is_required(placement):
    with pytest.raises(TypeError, match="exactly"):
        compute_shard_range(placement, 2, 16, 8, 0)
    with pytest.raises(TypeError, match="exactly"):
        plan_equal_shards((16, 2), placement, 8)


def test_shard_boolean_dimension_has_public_observable_value():
    placement = Shard(True)
    assert placement.dim == 1
    assert type(placement.dim) is int
    assert compute_shard_range(placement, 2, 16, 8, 0).tensor_dim == 1


def test_helper_is_called_once_with_exact_arguments(monkeypatch):
    original = Shard.local_shard_size_and_offset
    calls = []

    def recording_helper(extent, count, ordinal):
        calls.append((extent, count, ordinal))
        return original(extent, count, ordinal)

    monkeypatch.setattr(
        Shard, "local_shard_size_and_offset", staticmethod(recording_helper)
    )
    compute_shard_range(Shard(0), 2, 16, 8, 3)
    assert calls == [(16, 8, 3)]


def test_instance_helper_shadow_cannot_fabricate_geometry():
    placement = Shard(0)
    placement.local_shard_size_and_offset = lambda *args: (16, 0)
    actual = compute_shard_range(placement, 2, 16, 8, 3)
    assert (actual.logical_extent, actual.logical_offset) == (2, 6)


@pytest.mark.parametrize(
    "result",
    [
        None,
        [2, 0],
        (2,),
        (2, 0, 1),
        (True, 0),
        (2, False),
        (-1, 0),
        (2, -1),
        (17, 0),
        (2, 15),
    ],
)
def test_malformed_helper_result_fails_closed(monkeypatch, result):
    monkeypatch.setattr(
        Shard, "local_shard_size_and_offset", staticmethod(lambda *args: result)
    )
    with pytest.raises(ShardPlanningUnavailableError):
        compute_shard_range(Shard(0), 2, 16, 8, 0)


def test_missing_helper_fails_closed(monkeypatch):
    monkeypatch.setattr(Shard, "local_shard_size_and_offset", None)
    with pytest.raises(ShardPlanningUnavailableError, match="unavailable"):
        compute_shard_range(Shard(0), 2, 16, 8, 0)


def test_raising_helper_fails_closed_without_fallback(monkeypatch):
    def failing_helper(*args):
        raise RuntimeError("incompatible helper")

    monkeypatch.setattr(
        Shard, "local_shard_size_and_offset", staticmethod(failing_helper)
    )
    with pytest.raises(ShardPlanningUnavailableError, match="failed") as error:
        compute_shard_range(Shard(0), 2, 16, 8, 0)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_plan_accepts_exactly_eight_equal_nonempty_ranges():
    result = plan_equal_shards([3, 32], Shard(-1), 8)
    assert len(result) == 8
    assert tuple(r.shard_ordinal for r in result) == tuple(range(8))
    assert tuple(r.logical_offset for r in result) == tuple(range(0, 32, 4))
    assert {r.logical_extent for r in result} == {4}
    assert {r.tensor_dim for r in result} == {1}
    assert isinstance(result, tuple)
    assert all(isinstance(r, ShardRange) for r in result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result[0].logical_offset = 1
    assert not hasattr(result[0], "__dict__")


@pytest.mark.parametrize("shape", [(), [], (8, 0), (0, 8), (8, -1)])
def test_plan_rejects_scalar_zero_and_negative_shapes(shape):
    with pytest.raises(ValueError):
        plan_equal_shards(shape, Shard(0), 8)


@pytest.mark.parametrize(
    "shape",
    [
        (True, 8),
        (8, False),
        (_IntSubclass(8), 8),
        (8, _Indexable()),
        (torch.SymInt, 8),
    ],
)
def test_plan_rejects_nonconcrete_shape_entries(shape):
    with pytest.raises(TypeError, match="concrete ints"):
        plan_equal_shards(shape, Shard(0), 8)


@pytest.mark.parametrize("extent", [3, 7, 10, 15])
def test_plan_rejects_uneven_or_empty_tail_geometry(extent):
    with pytest.raises(ValueError):
        plan_equal_shards((extent, 2), Shard(0), 8)


def test_planner_does_not_mutate_mutable_input():
    accepted = [16, 3]
    before = accepted.copy()
    plan_equal_shards(accepted, Shard(0), 8)
    assert accepted == before

    rejected = [10, 3]
    before = rejected.copy()
    with pytest.raises(ValueError):
        plan_equal_shards(rejected, Shard(0), 8)
    assert rejected == before


def test_planner_calls_helper_once_for_each_ordinal(monkeypatch):
    original = Shard.local_shard_size_and_offset
    calls = []

    def recording_helper(extent, count, ordinal):
        calls.append((extent, count, ordinal))
        return original(extent, count, ordinal)

    monkeypatch.setattr(
        Shard, "local_shard_size_and_offset", staticmethod(recording_helper)
    )
    plan_equal_shards((24, 3), Shard(0), 8)
    assert calls == [(24, 8, ordinal) for ordinal in range(8)]


def test_planner_rejects_inconsistent_helper_geometry(monkeypatch):
    original = Shard.local_shard_size_and_offset

    def gapped_helper(extent, count, ordinal):
        local_extent, offset = original(extent, count, ordinal)
        return local_extent, offset + (1 if ordinal == 4 else 0)

    monkeypatch.setattr(
        Shard, "local_shard_size_and_offset", staticmethod(gapped_helper)
    )
    with pytest.raises(ValueError, match="gap-free"):
        plan_equal_shards((16, 2), Shard(0), 8)


def test_planning_has_no_distributed_side_effects(monkeypatch):
    assert not torch.distributed.is_initialized()
    forbid_distributed(monkeypatch)
    compute_shard_range(Shard(0), 2, 16, 8, 0)
    plan_equal_shards((16, 2), Shard(0), 8)
    assert not torch.distributed.is_initialized()
