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

"""vllm-trace: synthetic bundles in the fleet and PVC layouts -> rows, identity, and the verdict views."""

import json
import shutil
from pathlib import Path

import pytest
from spyre_clickhouse_ingest import vllm_trace
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.identity import benchmark_id_for

DATA = Path(__file__).parent / "data" / "vllm_trace"
FLEET = DATA / "fleet"
PVC = DATA / "pvc"
PVC_RUN = "20261103T000544-b512"
AID = "aaaaaaaa-aaaa-5aaa-8aaa-aaaaaaaaaaaa"
TRACE_TYPES = ("trace_phase", "trace_step", "trace_kernel", "compile_pass")


def fleet_ctx(**kw):
    return vllm_trace.load(FLEET / "out" / "w1", FLEET / "rep", "w1", **kw)


def by_type(rows):
    out = {}
    for r in rows:
        out.setdefault(r["props"]["record_type"], []).append(r)
    return out


def bench_id(row):
    return benchmark_id_for(
        vllm_trace.COMPONENT, row["name"], row["tags"], row["disc"], row["disc_keys"]
    )


def test_fleet_bundle_rows_per_record_type():
    rows = by_type(vllm_trace.entries(fleet_ctx()))
    assert {k: len(v) for k, v in rows.items()} == {
        "trace_phase": 6,
        "trace_step": 2,
        "trace_kernel": 4,
        "compile_pass": 3,
    }
    phase = {r["props"]["phase"]: r for r in rows["trace_phase"]}["init"]
    assert set(phase["measurements"]) == {
        "wall_s", "cpu_s", "cpu_cores_mean", "cpu_cores_peak", "throttled_s",
        "nr_throttled", "rss_peak_b", "cgroup_peak_b", "dev_allocated_peak_b",
    }  # fmt: skip
    assert phase["measurements"]["cgroup_peak_b"] == [14.0]
    step = {r["props"]["phase"]: r for r in rows["trace_step"]}["decode"][
        "measurements"
    ]
    assert {"wall_ms", "busy_ms", "host_exposed_ms", "launches", "fallback_ms"} <= set(
        step
    )
    assert "host_sync_wait_ms" in step and "host_python_ms" in step
    assert step["launches"] == [100.0]


def test_kernel_rows_carry_family_floor_and_provenance():
    k = by_type(vllm_trace.entries(fleet_ctx()))["trace_kernel"][0]
    assert k["props"]["provenance_key"] == k["props"]["kernel_name"]
    assert k["props"]["family"] == "matmul"
    assert k["props"]["aten_ops"].startswith("aten._to_copy.default")
    m = k["measurements"]
    assert {"calls", "median_us", "total_ms", "floor_us", "meas_over_floor", "gbps",
            "hbm_bytes", "macs", "floor_valid"} <= set(m)  # fmt: skip
    assert m["floor_valid"] == [1.0]


def test_floor_below_one_is_flagged_invalid():
    ctx = fleet_ctx()
    ctx["roof"]["kernels"][0].update(median_us=10.0, floor_us=20.0)
    k = by_type(vllm_trace.entries(ctx))["trace_kernel"][0]
    assert k["measurements"]["floor_valid"] == [0.0]


def test_predicted_us_only_where_the_cost_sidecar_has_it():
    kernels = by_type(vllm_trace.entries(fleet_ctx()))["trace_kernel"]
    have = [k for k in kernels if "predicted_us" in k["measurements"]]
    assert len(have) == 1 and have[0]["measurements"]["predicted_us"] == [300.5]
    assert "predicted_us" not in kernels[-1]["measurements"]


def test_run_props_link_the_artifact_and_carry_env_and_compile():
    rows = vllm_trace.entries(fleet_ctx(), "https://x/bundle.tgz")
    p = rows[0]["run_props"]
    assert p["artifact_id12"] == "aaaaaaaaaaaa"
    assert p["trace_bundle_url"] == "https://x/bundle.tgz"
    assert p["compile_total_s"] == "123.4" and p["compile_cache_misses"] == "3"
    assert p["arch"] == "x86_64" and p["hardware_type"] == "IBM_Spyre"
    shared = {k: v for k, v in p.items() if k != "run_mode"}
    assert all(
        {k: v for k, v in r["run_props"].items() if k != "run_mode"} == shared
        for r in rows
        if r["props"]["record_type"] != "trace_kernel"
    )
    assert not any(k.startswith("res_") for k in p)


def test_run_props_carry_the_workload_and_known_engine_limits_only():
    rows = vllm_trace.entries(fleet_ctx())
    for r in rows:
        p = r["run_props"]
        for k in ("model", "tensor_parallel", "batch_size", "input_len", "output_len",
                  "max_num_seqs", "max_model_len", "run_mode"):  # fmt: skip
            assert p[k] == r["props"][k], k
        assert "max_num_batched_tokens" not in p
        assert "" not in p.values()


def test_run_props_never_change_the_benchmark_identity():
    ctx = fleet_ctx()
    base = [bench_id(r) for r in vllm_trace.entries(ctx)]
    ctx["workload"]["max_num_batched_tokens"] = "512"
    rows = vllm_trace.entries(ctx)
    assert all(r["run_props"]["max_num_batched_tokens"] == "512" for r in rows)
    for r in rows:
        r["run_props"] = {}
    ids = [bench_id(r) for r in rows]
    assert not set(base) & set(ids)
    assert ids == [bench_id(r) for r in vllm_trace.entries(ctx)]


def test_pvc_layout_reads_the_manifest_artifact_and_env():
    ctx = vllm_trace.load(
        PVC / PVC_RUN, PVC / "analyzed", PVC_RUN,
        model="/x/models--example-org--model-a/snapshots/abc",
        tp=1, batch_size=4, input_len=1984, output_len=32,
    )  # fmt: skip
    rows = vllm_trace.entries(ctx)
    p = rows[0]["run_props"]
    assert p["artifact_id12"] == "bbbbbbbbbbbb"
    assert p["env_cpu_model"] == "Example CPU"
    assert p["env_knob_FLEX_COMPUTE"] == "EXAMPLE"
    assert p["run_utc"] == "2026-11-03 00:05:45"
    assert (p["max_num_batched_tokens"], p["max_num_seqs"], p["max_model_len"]) == (
        "512",
        "4",
        "2048",
    )
    assert p["model"] == "example-org/model-a" and p["tensor_parallel"] == "1"
    assert rows[0]["props"]["model"] == "example-org/model-a"
    assert {r["props"]["record_type"] for r in rows} == set(TRACE_TYPES)


def test_engine_limits_come_from_the_log_and_separate_b512_from_b8192():
    kw = dict(model="m/n", tp=1, batch_size=4, input_len=1984, output_len=32)
    a = vllm_trace.load(PVC / PVC_RUN, PVC / "analyzed", PVC_RUN, **kw)
    assert {k: a["workload"][k] for k in vllm_trace.ENGINE_LIMITS} == {
        "max_num_batched_tokens": "512", "max_num_seqs": "4", "max_model_len": "2048"}  # fmt: skip
    b = vllm_trace.load(
        PVC / PVC_RUN, PVC / "analyzed", PVC_RUN, max_num_batched_tokens=8192, **kw
    )
    assert not {bench_id(r) for r in vllm_trace.entries(a)} & {
        bench_id(r) for r in vllm_trace.entries(b)
    }


def test_fleet_engine_limits_from_e2e_and_staging_rows_stays_out_of_identity():
    w = fleet_ctx()["workload"]
    assert (w["max_num_batched_tokens"], w["max_num_seqs"], w["max_model_len"]) == (
        "",
        "1",
        "1024",
    )
    assert "staging_rows" not in vllm_trace.ID_KEYS


def test_workload_is_missing_means_rejected_not_guessed():
    with pytest.raises(vllm_trace.TraceError, match="workload not known"):
        vllm_trace.load(PVC / PVC_RUN, PVC / "analyzed", PVC_RUN)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other/model"),
        ("tp", 4),
        ("batch_size", 8),
        ("input_len", 99),
        ("output_len", 7),
    ],
)
def test_every_workload_field_is_in_the_benchmark_identity(field, value):
    base = {bench_id(r) for r in vllm_trace.entries(fleet_ctx())}
    ctx = fleet_ctx()
    ctx["workload"][field] = value
    other = {bench_id(r) for r in vllm_trace.entries(ctx)}
    assert not base & other


def test_identity_is_stable_across_runs_and_unique_within_one():
    ids = [bench_id(r) for r in vllm_trace.entries(fleet_ctx())]
    assert len(ids) == len(set(ids)) == 15
    assert ids == [bench_id(r) for r in vllm_trace.entries(fleet_ctx())]


def test_the_same_kernel_in_prefill_and_decode_is_two_benchmarks():
    ctx = fleet_ctx()
    k = ctx["roof"]["kernels"][0]
    ctx["roof"]["kernels"] = [
        k,
        {**k, "phase": "prefill" if k["phase"] == "decode" else "decode"},
    ]
    ks = by_type(vllm_trace.entries(ctx))["trace_kernel"]
    assert len({bench_id(r) for r in ks}) == 2


def test_rejects_a_dir_with_no_analyzed_json(tmp_path):
    with pytest.raises(vllm_trace.TraceError, match="no w1"):
        vllm_trace.load(FLEET / "out" / "w1", tmp_path, "w1")


def test_cli_dry_run_prints_counts_and_writes_nothing(monkeypatch, capsys):
    monkeypatch.delenv("CLICKHOUSE_HOST", raising=False)
    rc = cli(["vllm-trace", str(FLEET / "out" / "w1"), "--analyzed", str(FLEET / "rep"),
              "--label", "w1", "--dry-run"])  # fmt: skip
    out = capsys.readouterr()
    report = json.loads(out.out.splitlines()[-1])
    assert rc == 0 and report["status"] == "would-ingest"
    assert report["rows"] == {
        "compile_pass": 3,
        "trace_kernel": 4,
        "trace_phase": 6,
        "trace_step": 2,
    }
    assert (
        "15 row(s) -> benchmark_runs" in out.err
        and "1 row(s) -> artifact_results" in out.err
    )


def test_cli_rejects_a_bundle_with_no_artifact(tmp_path, capsys):
    shutil.copytree(FLEET / "out" / "w1", tmp_path / "w1")
    (tmp_path / "w1" / "spyre_artifact.json").unlink()
    rc = cli(["vllm-trace", str(tmp_path / "w1"), "--analyzed", str(FLEET / "rep"),
              "--label", "w1", "--dry-run"])  # fmt: skip
    assert rc == 2 and "no artifact" in capsys.readouterr().out
