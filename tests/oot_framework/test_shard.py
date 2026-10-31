"""Tests for oot_framework.utils.shard: every test runs in exactly one shard, deterministically."""
import json
import os
import random

import pytest

from oot_framework.utils.shard import (
    assign,
    load_durations,
    select_shard,
    shard_from_env,
)


class _Item:
    def __init__(self, nodeid):
        self.nodeid = nodeid


NODEIDS = [f"tests/inductor/test_x.py::TestX::test_{i}[case{j}]" for i in range(40) for j in range(3)]


@pytest.mark.parametrize("n", [2, 3, 4, 7])
@pytest.mark.parametrize("with_durations", [False, True])
def test_every_test_lands_in_exactly_one_shard(n, with_durations):
    rnd = random.Random(0)
    durations = {nid: rnd.uniform(0.1, 120) for nid in NODEIDS} if with_durations else {}
    items = [_Item(n_) for n_ in NODEIDS]
    seen = []
    for k in range(n):
        kept, dropped = select_shard(items, (k, n), durations)
        assert len(kept) + len(dropped) == len(items)
        seen += [i.nodeid for i in kept]
    assert sorted(seen) == sorted(NODEIDS)


@pytest.mark.parametrize("with_durations", [False, True])
def test_partition_ignores_collection_order(with_durations):
    rnd = random.Random(1)
    durations = {nid: rnd.uniform(0.1, 60) for nid in NODEIDS} if with_durations else {}
    shuffled = NODEIDS[:]
    random.Random(2).shuffle(shuffled)
    assert assign(NODEIDS, 4, durations) == assign(shuffled, 4, durations)


def test_durations_balance_beats_hash_on_uneven_suites():
    # Sliding-window shape: a few multi-minute cases among many short ones.
    durations = {nid: (120.0 if i % 20 == 0 else 5.0) for i, nid in enumerate(NODEIDS)}
    def longest(where):
        loads = {}
        for nid, k in where.items():
            loads[k] = loads.get(k, 0.0) + durations[nid]
        return max(loads.values())
    assert longest(assign(NODEIDS, 4, durations)) <= longest(assign(NODEIDS, 4, None))


def test_unknown_tests_get_the_median_weight():
    known = {NODEIDS[0]: 10.0, NODEIDS[1]: 30.0}
    where = assign(NODEIDS[:5], 2, known)
    assert set(where) == set(NODEIDS[:5])


@pytest.mark.parametrize("env,expected", [
    ({}, None),
    ({"SPYRE_TEST_SHARD_ID": "", "SPYRE_TEST_NUM_SHARDS": ""}, None),
    ({"SPYRE_TEST_SHARD_ID": "0", "SPYRE_TEST_NUM_SHARDS": "1"}, None),
    ({"SPYRE_TEST_SHARD_ID": "2", "SPYRE_TEST_NUM_SHARDS": "4"}, (2, 4)),
])
def test_shard_from_env(env, expected):
    assert shard_from_env(env) == expected


@pytest.mark.parametrize("env", [
    {"SPYRE_TEST_SHARD_ID": "1"},
    {"SPYRE_TEST_NUM_SHARDS": "3"},
    {"SPYRE_TEST_SHARD_ID": "3", "SPYRE_TEST_NUM_SHARDS": "3"},
    {"SPYRE_TEST_SHARD_ID": "-1", "SPYRE_TEST_NUM_SHARDS": "3"},
    {"SPYRE_TEST_SHARD_ID": "0", "SPYRE_TEST_NUM_SHARDS": "0"},
])
def test_shard_from_env_rejects_misconfiguration(env):
    with pytest.raises(ValueError):
        shard_from_env(env)


def test_load_durations(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"a": 1.5, "b": 2, "c": "x", "d": -1}))
    assert load_durations(str(p)) == {"a": 1.5, "b": 2.0}
    assert load_durations(str(tmp_path / "missing.json")) == {}
    assert load_durations(None) == {}


def test_filter_configs_expands_a_sharded_suite(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    cfgdir = tmp_path / "configs"
    (cfgdir / "inductor").mkdir(parents=True)
    (cfgdir / "inductor" / "test_big_config.yaml").write_text(
        "test_suite_config:\n  shards: 3\n  labels: [regression]\n  files: []\n")
    (cfgdir / "inductor" / "test_small_config.yaml").write_text(
        "test_suite_config:\n  labels: [regression]\n  files: []\n")
    script = Path(__file__).resolve().parent / "utils" / "filter_configs.py"
    out = subprocess.run([sys.executable, str(script), "--config-dir", str(cfgdir),
                          "--test-type", "regression", "--format", "matrix-json"],
                         capture_output=True, text=True, check=True).stdout
    suites = json.loads(out)["suite"]
    big = [s for s in suites if s["config"] == "inductor/test_big_config.yaml"]
    small = [s for s in suites if s["config"] == "inductor/test_small_config.yaml"]
    assert [(s["shard_id"], s["num_shards"]) for s in big] == [(0, 3), (1, 3), (2, 3)]
    assert [s["name"] for s in big] == [f"Inductor / Test Big ({k}/3)" for k in (1, 2, 3)]
    assert len(small) == 1 and "shard_id" not in small[0]
    paths = subprocess.run([sys.executable, str(script), "--config-dir", str(cfgdir),
                            "--test-type", "regression", "--format", "paths"],
                           capture_output=True, text=True, check=True).stdout.split()
    assert len(paths) == 2, "paths format lists each config once, unsharded"
