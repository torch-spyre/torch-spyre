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

"""Tests for torch.spyre.distribute_tensor(x, Shard(d))."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.distributed.tensor import Partial, Replicate, Shard

import torch_spyre._sharded_api as sharded_api
from device_env import emulation_ready
from side_effect_utils import subprocess_import_side_effects
from torch_spyre._sharded_api import (
    distribute_tensor,
    full_tensor,
    is_sharded,
    sharding_info,
)
from torch_spyre.sharding import SpyreShardingSpec

_REAL_ACTIVE_CORE_COUNT = sharded_api._active_core_count


class _TensorSubclass(torch.Tensor):
    pass


def _source(shape=(16,), dtype=torch.float32):
    count = 1
    for extent in shape:
        count *= extent
    values = torch.arange(count, dtype=torch.int64)
    if dtype is torch.bool:
        values = values.remainder(2).to(dtype)
    else:
        values = values.to(dtype)
    return values.reshape(shape).clone()


class _NativeCalled(Exception):
    """Raised by the fake runtime: validation let a request through."""


@pytest.fixture(autouse=True)
def active_cores(monkeypatch):
    """Pin the active core count; tests that shard 16 ways set it to 16."""

    state = {"cores": 8}
    monkeypatch.setattr(sharded_api, "_active_core_count", lambda: state["cores"])
    return state


@pytest.fixture
def no_runtime(monkeypatch):
    """Fail if a request gets past validation to the runtime."""

    calls = []

    def init():
        calls.append("init")
        raise _NativeCalled

    monkeypatch.setattr(sharded_api, "_ensure_spyre_runtime_initialized", init)
    return calls


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _spyre_device_available() -> bool:
    try:
        return hasattr(torch, "spyre") and torch.spyre.device_count() > 0
    except (RuntimeError, ImportError):
        return False


_requires_spyre = pytest.mark.skipif(
    not (_spyre_device_available() and emulation_ready()),
    reason="requires a registered Spyre device and a Flex device mode that "
    "admits sharded allocations (see device_env.emulation_ready)",
)


def test_importing_sharded_api_has_no_backend_or_runtime_side_effects():
    effects = subprocess_import_side_effects("import torch_spyre._sharded_api")
    assert effects == {
        "extension": False,
        "distributed": False,
        "spyre_registered": False,
        "runtime": False,
    }


# ---------------------------------------------------------------------------
# Validation: every rejection happens before the runtime is touched (CPU only).
# ---------------------------------------------------------------------------


def test_valid_request_reaches_the_runtime(no_runtime):
    with pytest.raises(_NativeCalled):
        distribute_tensor(_source((16,)), Shard(0))
    assert no_runtime == ["init"]


@pytest.mark.parametrize("placement", [Replicate(), Partial(), object(), None, 0])
def test_placement_must_be_exact_shard(no_runtime, placement):
    with pytest.raises(TypeError, match="Shard"):
        distribute_tensor(_source(), placement)
    assert no_runtime == []


def test_spec_object_is_not_the_public_placement(no_runtime):
    # Construct a fully valid spec first: SpyreShardingSpec(Shard(0)) alone
    # raises a dataclass TypeError whose message names the class (which
    # contains "Shard"), so that form passes even if distribute_tensor
    # accepts spec objects. A valid spec isolates the placement rejection.
    spec = SpyreShardingSpec(Shard(0), 8)
    with pytest.raises(TypeError, match="placement must be exactly"):
        distribute_tensor(_source(), spec)
    assert no_runtime == []


def test_tp_style_single_element_list_placement_is_accepted(no_runtime):
    """``[Shard(0)]`` is the torch.distributed.tensor idiom; accept it."""

    with pytest.raises(_NativeCalled):
        distribute_tensor(_source((16,)), [Shard(0)])
    assert no_runtime == ["init"]


@pytest.mark.parametrize("placement", [[Shard(0), Shard(1)], [], [Shard(0), None]])
def test_placement_list_must_hold_exactly_one_entry(no_runtime, placement):
    with pytest.raises(TypeError, match="single Shard"):
        distribute_tensor(_source((16,)), placement)
    assert no_runtime == []


def test_num_shards_defaults_to_the_active_core_count(no_runtime, active_cores):
    active_cores["cores"] = 16

    with pytest.raises(_NativeCalled):
        distribute_tensor(_source((32,)), Shard(0))
    assert no_runtime == ["init"]


def test_default_count_must_be_a_supported_shard_count(no_runtime, active_cores):
    active_cores["cores"] = 32

    with pytest.raises(ValueError, match="8 or 16"):
        distribute_tensor(_source((32,)), Shard(0))
    assert no_runtime == []


def test_full_tensor_rejects_non_sharded_values():
    with pytest.raises(TypeError, match="sharded"):
        full_tensor(torch.empty(4))
    with pytest.raises(TypeError, match="sharded"):
        full_tensor(object())


@pytest.mark.parametrize("num_shards", [0, 32])
def test_unsupported_shard_counts_fail(no_runtime, num_shards):
    with pytest.raises(ValueError, match="8 or 16"):
        distribute_tensor(_source((32,)), Shard(0), num_shards=num_shards)
    assert no_runtime == []


def test_num_shards_is_keyword_only(no_runtime):
    with pytest.raises(TypeError):
        distribute_tensor(_source((16,)), Shard(0), 8)
    assert no_runtime == []


@pytest.mark.parametrize(
    "num_shards,cores", [(8, 16), (16, 8), (8, 32), (16, 32), (8, 1)]
)
def test_shard_count_must_equal_active_cores(
    no_runtime, active_cores, num_shards, cores
):
    active_cores["cores"] = cores

    with pytest.raises(ValueError, match=f"SENCORES={cores}"):
        distribute_tensor(_source((32,)), Shard(0), num_shards=num_shards)
    assert no_runtime == []


def test_active_core_count_reads_inductor_sencores(monkeypatch):
    from torch_spyre._inductor import config

    monkeypatch.setattr(config, "sencores", 16)

    assert _REAL_ACTIVE_CORE_COUNT() == 16


@pytest.mark.parametrize(
    "source,message",
    [
        (None, "torch.Tensor"),
        (object(), "torch.Tensor"),
        (_source(dtype=torch.int64), "dtype"),
        (torch.empty(16, device="meta"), "CPU tensor"),
        (torch.tensor(1.0), "scalar"),
        (torch.empty(0), "positive"),
        (torch.empty(16, requires_grad=True), "gradients"),
        (torch.empty(16).as_strided((8,), (1,), 1), "zero storage offset"),
        (torch.empty(4, 4).t(), "contiguous"),
        (torch.empty(32)[8:24], "zero storage offset|base-like"),
        (torch.arange(16.0).reshape(16), "base-like"),
    ],
)
def test_invalid_sources_fail_before_runtime(no_runtime, source, message):
    with pytest.raises((TypeError, ValueError), match=message):
        distribute_tensor(source, Shard(0))
    assert no_runtime == []


def test_tensor_subclass_fails_before_runtime(no_runtime):
    with pytest.raises(TypeError, match="torch.Tensor or torch.nn.Parameter"):
        distribute_tensor(_source().as_subclass(_TensorSubclass), Shard(0))
    assert no_runtime == []


def test_parameter_without_grad_reaches_runtime(no_runtime):
    source = torch.nn.Parameter(_source(), requires_grad=False)

    with pytest.raises(_NativeCalled):
        distribute_tensor(source, Shard(0))


@pytest.mark.parametrize(
    "shape,dim,num_shards,message",
    [
        ((3, 2), 0, 8, "non-empty|equal|cover"),
        ((10, 2), 0, 8, "equal|cover"),
        ((24,), 0, 16, "equal|cover"),
        ((16, 2), 2, 8, "outside"),
        ((16, 2), -3, 8, "outside"),
    ],
)
def test_planning_failures_precede_runtime(
    no_runtime, active_cores, shape, dim, num_shards, message
):
    active_cores["cores"] = num_shards
    with pytest.raises(ValueError, match=message):
        distribute_tensor(torch.empty(shape), Shard(dim), num_shards=num_shards)
    assert no_runtime == []


def test_is_sharded_is_false_for_non_spyre_values():
    assert not is_sharded(torch.empty(4))
    assert not is_sharded(object())
    with pytest.raises(TypeError, match="sharded"):
        sharding_info(torch.empty(4))


# ---------------------------------------------------------------------------
# Device: the result is a real sharded torch.Tensor (requires a Spyre card).
# ---------------------------------------------------------------------------


def _expected_round_trip(source):
    """The value an ordinary unsharded H2D/D2H round trip produces."""

    return source.to("spyre").cpu()


_DEVICE_CASES = [
    ((16, 64), 0, torch.float16),
    ((64, 256), 1, torch.float32),
    ((32, 128), 0, torch.bfloat16),
    ((2, 32, 64), 1, torch.float16),
    ((32,), 0, torch.float32),
]


@_requires_spyre
@pytest.mark.parametrize("num_shards", [8, 16])
@pytest.mark.parametrize("shape,dim,dtype", _DEVICE_CASES)
def test_device_result_is_a_sharded_tensor(active_cores, shape, dim, dtype, num_shards):
    active_cores["cores"] = num_shards
    shape = tuple(e * (2 if i == dim % len(shape) else 1) for i, e in enumerate(shape))
    source = torch.randn(shape).to(dtype)

    result = torch.spyre.distribute_tensor(source, Shard(dim), num_shards=num_shards)

    assert type(result) is torch.Tensor
    assert result.device.type == "spyre"
    assert result.shape == source.shape
    assert result.dtype == dtype
    assert torch.spyre.is_sharded(result)
    info = torch.spyre.sharding_info(result)
    assert info.placement == Shard(dim % len(shape))
    assert info.num_shards == num_shards
    # The public receipt is generation-neutral: identity, core binding, and
    # equal shard extents. The adapter's mapping of the opaque physical
    # fields is covered on CPU in test_segment_emulated_receipt; the device's
    # physical placement itself is not asserted here.
    assert [e.shard_ordinal for e in info.receipt.entries] == list(range(num_shards))
    assert [e.core_id for e in info.receipt.entries] == list(range(num_shards))
    assert {e.byte_length for e in info.receipt.entries} == {info.shard_device_bytes}
    assert torch.equal(result.to("cpu"), _expected_round_trip(source))


@_requires_spyre
@pytest.mark.parametrize(
    "gather",
    [
        lambda t: t.to("cpu"),
        lambda t: t.cpu(),
        lambda t: t.to(torch.device("cpu")),
        lambda t: t.to(device="cpu"),
        torch.spyre.full_tensor,
    ],
)
def test_device_gather_forms(gather):
    source = torch.randn(32, 64, dtype=torch.float32)
    result = torch.spyre.distribute_tensor(source, Shard(0))

    assert torch.equal(gather(result), source)


@_requires_spyre
@pytest.mark.parametrize("num_shards", [8, 16])
def test_device_shards_spread_round_robin_across_domains(active_cores, num_shards):
    """Each shard names its Flex memory domain: shard i lands in i % N."""

    from torch_spyre import _C

    active_cores["cores"] = num_shards
    source = torch.randn(num_shards * 4, 64, dtype=torch.float32)

    # distribute_tensor initializes the Flex runtime; num_memory_domains
    # reports the topology of the initialized allocator.
    result = torch.spyre.distribute_tensor(source, Shard(0), num_shards=num_shards)
    num_domains = _C.num_memory_domains()

    info = torch.spyre.sharding_info(result)
    expected = [i % num_domains for i in range(num_shards)]
    assert [e.physical["physical_domain_id"] for e in info.receipt.entries] == (
        expected
    )
    # Placement is an allocation-time fact: the round trip is unaffected.
    assert torch.equal(result.cpu(), source)


@_requires_spyre
def test_device_each_shard_is_in_its_own_chunk():
    source = torch.randn(32, 64, dtype=torch.float32)
    result = torch.spyre.distribute_tensor(source, Shard(0))
    from torch_spyre import _C

    # Overwrite shard 3 on the device through a fresh copy: only its rows move.
    replacement = source.clone()
    replacement[12:16] = 7.0
    _C.copy_tensor(replacement, result, non_blocking=False)
    assert torch.equal(result.cpu(), replacement)
    assert torch.equal(result.cpu()[:12], source[:12])


@_requires_spyre
def test_device_fill_fails_instead_of_writing_one_chunk(active_cores):
    # TODO: positive once fills are shard-aware.
    # fill_ takes one device address; the runtime guard rejects sharded
    # storage. Compiled ops on sharded tensors are covered by
    # test_sharded_add.py.
    result = torch.spyre.distribute_tensor(torch.randn(16, 64), Shard(0))

    with pytest.raises(
        RuntimeError, match="does not support sharded Spyre tensors yet"
    ):
        result.fill_(0)


test_device_fill_fails_instead_of_writing_one_chunk._expects_raise = True
