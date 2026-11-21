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
import os
from pathlib import Path

import torch

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


def test_coverage_guard_passes_when_device_tests_executed():
    from device_suite import device_coverage_verdict

    verdict, message = device_coverage_verdict(
        device_collected=36,
        device_executed=0,
        suite_collected=2,
        suite_executed=2,
        vf_capable=True,
        runtime_open=False,
        collect_only=False,
    )
    assert verdict == "pass"
    assert message is None


def test_coverage_guard_fails_when_vf_host_ran_nothing():
    from device_suite import device_coverage_verdict

    verdict, message = device_coverage_verdict(
        device_collected=36,
        device_executed=0,
        suite_collected=2,
        suite_executed=0,
        vf_capable=True,
        runtime_open=False,
        collect_only=False,
    )
    assert verdict == "fail"
    assert "bound VFs" in message


def test_coverage_guard_warns_when_no_device():
    from device_suite import device_coverage_verdict

    verdict, message = device_coverage_verdict(
        device_collected=36,
        device_executed=0,
        suite_collected=2,
        suite_executed=0,
        vf_capable=False,
        runtime_open=False,
        collect_only=False,
    )
    assert verdict == "warn"
    assert "0 of 36" in message


def test_coverage_guard_passes_on_collect_only_and_subsets():
    from device_suite import device_coverage_verdict

    # --collect-only: nothing was supposed to run.
    assert device_coverage_verdict(36, 0, 2, 0, True, False, True)[0] == "pass"
    # A -k selection that deselected the wrapper: the in-process skip of a
    # hand-picked device test is by design, not silent rot.
    assert device_coverage_verdict(1, 0, 0, 0, True, False, False)[0] == "pass"
    # Runtime already open: the wrapper skips by design (run files alone).
    verdict, _ = device_coverage_verdict(36, 0, 2, 0, True, True, False)
    assert verdict == "warn"


def test_vf_mode_is_promoted_to_emulation_before_the_runtime_loads(monkeypatch):
    from device_env import promote_vf_mode_to_emulation
    from torch_spyre._inductor import config as spyre_config

    if not hasattr(torch, "spyre"):
        pytest.skip(
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0: torch.spyre is not registered, "
            "so the runtime-open interplay this test pins cannot exist"
        )

    # The promotion must be a no-op once the runtime is open (the mode is
    # fixed then). Earlier tests in a combined run may have opened it, so
    # pin the state this test reasons about instead of inheriting it.
    monkeypatch.setattr(torch.spyre, "is_initialized", lambda: False)

    # promote() rewrites torch_spyre._inductor.config.sencores directly
    # when that module is already imported, and monkeypatch.setenv does
    # not undo that — the core count would leak to later tests in this
    # process (on a device-less runner the conftest never rewrote it).
    # patch() restores the attribute on exit, whatever the calls inside
    # did to it. (Not monkeypatch.setattr: pytest 9.2 regression.)
    with spyre_config.patch(sencores=spyre_config.sencores):
        # The whole suite runs in the emulation shape: a full-core VF
        # environment (SENCORES=32) is rewritten to EMUL1p5 with SENCORES
        # forced into the sharded-admitting shapes.
        monkeypatch.setenv("FLEX_DEVICE", "VF")
        monkeypatch.setenv("SENCORES", "8")
        assert promote_vf_mode_to_emulation() is True
        assert os.environ["FLEX_DEVICE"] == "EMUL1p5"
        assert os.environ["SENCORES"] == "8"

        monkeypatch.setenv("FLEX_DEVICE", "VF")
        monkeypatch.setenv("SENCORES", "32")
        assert promote_vf_mode_to_emulation() is True
        assert os.environ["FLEX_DEVICE"] == "EMUL1p5"
        assert os.environ["SENCORES"] == "8"

        monkeypatch.setenv("FLEX_DEVICE", "VF")
        monkeypatch.setenv("SENCORES", "16")
        assert promote_vf_mode_to_emulation() is True
        assert os.environ["SENCORES"] == "16"
        monkeypatch.delenv("SENCORES")

        # Other modes are left alone (PF stays PF; unset stays unset).
        monkeypatch.setenv("FLEX_DEVICE", "PF")
        assert promote_vf_mode_to_emulation() is False
        assert os.environ["FLEX_DEVICE"] == "PF"
        monkeypatch.delenv("FLEX_DEVICE")
        assert promote_vf_mode_to_emulation() is False
        assert "FLEX_DEVICE" not in os.environ

        # With the runtime already open, even VF is left alone.
        monkeypatch.setattr(torch.spyre, "is_initialized", lambda: True)
        monkeypatch.setenv("FLEX_DEVICE", "VF")
        monkeypatch.setenv("SENCORES", "8")
        assert promote_vf_mode_to_emulation() is False
        assert os.environ["FLEX_DEVICE"] == "VF"


def test_session_is_sharding_only():
    """The scoping gate for the VF -> emulation promotion (see conftest).

    Pure function of the session's path arguments and the invocation
    directory: the rewrite must apply only when nothing outside
    tests/sharding is collected. The repo layout is fixed, so real paths
    are used, resolved against a fake invocation directory.
    """
    from device_suite import session_is_sharding_only

    repo = Path(__file__).resolve().parents[2]
    sharding = repo / "tests" / "sharding"

    # A bare pytest (no args) collects from the invocation directory: inside
    # promotes, the repo root does not.
    assert session_is_sharding_only([], sharding) is True
    assert session_is_sharding_only([], repo) is False

    # Paths and node ids inside the directory.
    assert session_is_sharding_only(["tests/sharding"], repo) is True
    assert (
        session_is_sharding_only(["tests/sharding/test_sharded_add.py"], repo) is True
    )
    assert (
        session_is_sharding_only(
            ["tests/sharding/test_sharded_add.py::test_eager_add_on_sharded_inputs"],
            repo,
        )
        is True
    )
    assert (
        session_is_sharding_only([str(sharding / "test_sharding_spec.py")], repo)
        is True
    )

    # Relative to a different invocation directory (cd tests && pytest sharding).
    assert session_is_sharding_only(["sharding"], repo / "tests") is True

    # Anything outside, alone or mixed with an inside path.
    assert session_is_sharding_only(["tests"], repo) is False
    assert session_is_sharding_only(["tests/test_allocator_e2e.py"], repo) is False
    assert (
        session_is_sharding_only(
            ["tests/sharding/test_sharded_add.py", "tests/test_allocator_e2e.py"], repo
        )
        is False
    )
    assert session_is_sharding_only([".."], sharding) is False


def test_coverage_guard_never_probes_when_the_verdict_cannot_use_it():
    """The vf-capability probe opens the card in a subprocess; it must not
    run on a session whose verdict ignores it (review finding on #6206:
    --collect-only paid ~12 s for a probe whose answer was discarded, and
    a completed run probed while the parent still held the card)."""
    from device_suite import device_coverage_verdict

    def boom() -> bool:
        raise AssertionError("the probe must not run on this verdict path")

    # --collect-only.
    assert device_coverage_verdict(36, 0, 2, 0, boom, False, True)[0] == "pass"
    # The session executed its device tests (or wrappers).
    assert device_coverage_verdict(36, 5, 2, 1, boom, False, False)[0] == "pass"
    assert device_coverage_verdict(0, 0, 2, 2, boom, False, False)[0] == "pass"
    # Nothing guard-marked was selected at all.
    assert device_coverage_verdict(0, 0, 0, 0, boom, False, False)[0] == "pass"
    # The runtime is open: the wrapper skipped by design, no probe needed.
    assert device_coverage_verdict(36, 0, 2, 0, boom, True, False)[0] == "warn"

    # On the one path that needs it, the callable is consulted.
    calls = []

    def counting() -> bool:
        calls.append(1)
        return True

    assert device_coverage_verdict(36, 0, 2, 0, counting, False, False)[0] == "fail"
    assert calls == [1]
