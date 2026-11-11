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
from spyre_clickhouse_ingest import benchmeta, vllm_trace
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.identity import benchmark_id_for

DATA = Path(__file__).parent / "data" / "vllm_trace"
FLEET = DATA / "fleet"
PVC = DATA / "pvc"
PVC_RUN = "20261103T000544-b512"
AID = "aaaaaaaa-aaaa-5aaa-8aaa-aaaaaaaaaaaa"
TRACE_TYPES = ("trace_phase", "trace_step", "trace_kernel", "compile_pass")
# Written by spyre-inference's run_vllm_benchmarks.py --profile-level full with vllm mocked (TP=2).
HARNESS = DATA / "harness" / "lat_tp2" / "trace"


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
    per_row = {"run_mode", "bound", "fallback_ops", "top_thread_groups"}
    shared = {k: v for k, v in p.items() if k not in per_row}
    assert all(
        {k: v for k, v in r["run_props"].items() if k not in per_row} == shared
        for r in rows
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


def test_run_props_carry_card_memory_compiles_dtype_task_and_series_files():
    p = vllm_trace.entries(fleet_ctx())[0]["run_props"]
    assert p["dev_mem_total_b"] == "1000000" and p["compile_n_compiles"] == "3"
    assert (p["dtype"], p["task"]) == ("float16", "generate")
    assert json.loads(p["series_files"]) == {
        "host": "rank0/samples.jsonl",
        "devmem": "rank0/devmem.jsonl",
        "step_trace": "rank0/traces/example.json",
    }


def test_pvc_dtype_comes_from_the_log_and_an_unknown_task_is_omitted():
    ctx = vllm_trace.load(
        PVC / PVC_RUN, PVC / "analyzed", PVC_RUN,
        model="m/n", tp=1, batch_size=4, input_len=1984, output_len=32,
    )  # fmt: skip
    p = vllm_trace.entries(ctx)[0]["run_props"]
    assert p["dtype"] == "float16" and "task" not in p and "series_files" not in p


def test_step_rows_carry_cpu_fallback_and_the_top_fallback_ops():
    steps = {
        r["props"]["phase"]: r
        for r in by_type(vllm_trace.entries(fleet_ctx()))["trace_step"]
    }
    pre, dec = steps["prefill"], steps["decode"]
    assert pre["measurements"]["host_cpu_fallback_ms"] == [7.5]
    assert dec["measurements"]["host_cpu_fallback_ms"] == [0.0]
    ops = json.loads(pre["run_props"]["fallback_ops"])
    assert len(ops) == vllm_trace.TOP_FALLBACK_OPS
    assert next(iter(ops)) == "aten::op_11" and "aten::op_00" not in ops
    assert ops["aten::op_11"] == {"calls_per_step": 1.0, "ms_per_step": 1.2}
    assert "fallback_ops" not in dec["run_props"]


def test_phase_rows_carry_the_top_thread_groups_and_kernels_their_cores():
    rows = by_type(vllm_trace.entries(fleet_ctx()))
    phases = {r["props"]["phase"]: r["run_props"] for r in rows["trace_phase"]}
    tg = json.loads(phases["init"]["top_thread_groups"])
    assert list(tg) == ["python", "worker-a", "worker-b", "worker-c", "worker-d"]
    assert "top_thread_groups" not in phases["decode"]
    assert all("cores" in k["measurements"] for k in rows["trace_kernel"])


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


def test_harness_bundle_label_comes_from_the_manifest_not_the_trace_dir():
    ctx = vllm_trace.load(HARNESS)
    assert ctx["label"] == "lat_tp2" and HARNESS.name == "trace"
    p = vllm_trace.entries(ctx)[0]["run_props"]
    assert p["trace_label"] == "lat_tp2" and p["source_file"] == "vllm-trace/lat_tp2"
    assert vllm_trace.load(HARNESS, label="other")["label"] == "other"


def test_harness_bundle_records_profile_level_and_rank_count():
    p = vllm_trace.entries(vllm_trace.load(HARNESS))[0]["run_props"]
    assert p["profile_level"] == "full" and p["n_ranks"] == "2"
    assert p["tensor_parallel"] == "2" and p["artifact_id12"] == "dddddddddddd"
    assert json.loads(p["series_files"]) == {
        "host": "samples.jsonl",
        "step_trace": "raw/dp0_pp0_tp0_rank0.123.pt.trace.json",
    }
    assert "profile_level" not in vllm_trace.entries(fleet_ctx())[0]["run_props"]


def harness_ctx_with_a_mixed_kernel():
    """The harness folds mixed steps into decode kernels; older roof.json had phase mixed."""
    ctx = vllm_trace.load(HARNESS)
    k = next(k for k in ctx["roof"]["kernels"] if k["phase"] == "decode")
    ctx["roof"]["kernels"].append({**k, "phase": "mixed"})
    return ctx


def test_harness_mixed_steps_and_kernels_read_from_rank0():
    rows = by_type(vllm_trace.entries(harness_ctx_with_a_mixed_kernel()))
    steps = {r["props"]["phase"]: r for r in rows["trace_step"]}
    assert set(steps) == {"prefill", "decode", "mixed"}
    assert steps["mixed"]["props"]["run_mode"] == "mixed"
    # Rank 1 has no spyre tags, so no mixed steps: only rank 0's counts are read.
    assert (steps["mixed"]["iterations"], steps["decode"]["iterations"]) == (2, 4)
    mixed = [k for k in rows["trace_kernel"] if k["props"]["phase"] == "mixed"]
    assert [k["props"]["run_mode"] for k in mixed] == ["mixed"]
    assert mixed[0]["run_props"]["run_mode"] == "mixed"


def test_mixed_rows_leave_prefill_and_decode_ids_unchanged():
    ctx = harness_ctx_with_a_mixed_kernel()
    with_mixed = {bench_id(r) for r in vllm_trace.entries(ctx)}
    ctx["host"].pop("mixed_summary")
    ctx["roof"]["kernels"] = [
        k for k in ctx["roof"]["kernels"] if k["phase"] != "mixed"
    ]
    without = {bench_id(r) for r in vllm_trace.entries(ctx)}
    assert without < with_mixed and len(with_mixed - without) == 2


def test_harness_compile_block_gives_pass_rows_and_compile_props():
    rows = vllm_trace.entries(vllm_trace.load(HARNESS))
    p = rows[0]["run_props"]
    assert (p["compile_total_s"], p["compile_n_compiles"]) == ("8.0", "2")
    assert (p["compile_cache_entries"], p["compile_cache_hits"]) == ("4", "2")
    passes = {
        r["props"]["pass"]: (r["measurements"]["self_s"], r["iterations"])
        for r in by_type(rows)["compile_pass"]
    }
    assert passes == {"Pre:a": ([4.0], 2), "Post:b": ([2.0], 2)}


def test_by_pass_as_bare_seconds_still_reads():
    ctx = vllm_trace.load(HARNESS)
    ctx["res"]["compile"]["by_pass"] = {"pass:Pre:a": 1.5}
    rows = by_type(vllm_trace.entries(ctx))["compile_pass"]
    assert [(r["props"]["pass"], r["measurements"]["self_s"], r["iterations"])
            for r in rows] == [("Pre:a", [1.5], 0)]  # fmt: skip


def _fleet_copy(tmp_path):
    run, ana = tmp_path / "w1", tmp_path / "rep"
    shutil.copytree(FLEET / "out" / "w1", run)
    shutil.copytree(FLEET / "rep", ana)
    return run, ana


def _counts(rows):
    return {k: len(v) for k, v in by_type(rows).items()}


@pytest.mark.parametrize(
    "path, text",
    [
        ("rep/w1.res.json", "[1, 2]"),
        ("rep/w1.res.json", "{broken"),
        ("rep/w1.roof.json", '{"kernels": 5}'),
        ("rep/w1.roof.json", '{"kernels": [5, null, {"key": 7, "phase": []}]}'),
        ("rep/w1.host.json", "5"),
        ("rep/w1.host.json", "[5]"),
        ("rep/w1.host.json", "[" * 200000),
        ("out/w1/manifest.json", "[1]"),
        ("out/w1/env.json", "[1]"),
        ("out/w1/rank0/compile_summary.json", '"x"'),
        ("out/w1/rank0/kernel_cost.jsonl", '5\n[1]\n{"cost": 3, "key": "k"}\n\xff'),
        ("out/w1/rank0/phases.jsonl", '{"t": 1e999}\n'),
        ("out/w1/rank0/phases.jsonl", "[1]"),
    ],
)
def test_a_wrong_shaped_optional_file_never_aborts_the_ingest(tmp_path, path, text):
    run, ana = _fleet_copy(tmp_path)
    target = tmp_path / path.replace("rep/", "rep/").replace("out/w1", "w1")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, errors="surrogateescape")
    ctx = vllm_trace.load(run, ana, "w1")
    assert vllm_trace.entries(ctx)


def test_wrong_typed_fields_drop_that_record_and_keep_the_rest():
    ctx = fleet_ctx()
    good = _counts(vllm_trace.entries(ctx))
    res, roof = ctx["res"], ctx["roof"]
    res["phases"]["init"] = 5
    res["phases"]["import"]["wall_s"] = 10**400
    res["phases"]["decode"]["top_thread_groups"] = [1]
    res["steps"] = {"prefill": [1], "decode": {"n": "x", "wall_ms": float("nan")}}
    res["compile"]["by_pass"]["bad"] = {"self_s": "x", "n": [1]}
    res["compile"]["by_pass"][3] = 4
    roof["kernels"][0]["aten"] = [1, None, "aten.mm.default"]
    roof["kernels"][1]["calls"] = "many"
    roof["kernels"][2]["median_us"] = float("inf")
    ctx["host"]["decode_summary"] = {
        "fallback_ops": {"op": 5, "ok": {"ms_per_step": "x"}}
    }
    rows = vllm_trace.entries(ctx)
    assert vllm_trace.run_props(ctx)
    c = _counts(rows)
    assert c["trace_kernel"] == good["trace_kernel"]
    assert c["trace_phase"] == good["trace_phase"] - 1
    for r in rows:
        for v in r["measurements"].values():
            assert all(x == x and abs(x) != float("inf") for x in v)


def test_benchmeta_flatten_swallows_malformed_data():
    assert benchmeta.flatten("env", {"cards": 5, "host": "n1"}) == {}
    assert benchmeta.flatten("compile_summary", {"total_s": 4}) == {
        "compile_total_s": "4"
    }
