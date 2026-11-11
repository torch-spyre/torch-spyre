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

"""env/phases/host_resources/outputs/compile_summary sidecars through the vLLM ingest."""

import json

import pytest
import regex as re

from spyre_clickhouse_ingest import benchmeta, vllm

TEST = "latency_granite8B_tp1_in64_out64"
ENV = {
    "host": "node-1",
    "arch": "x86_64",
    "cpu": {"model": "EPYC", "threads": 64},
    "cgroup": {"cpu_max": "32000 100000", "memory_max": "max"},
    "kernel": "5.14",
    "os": "RHEL 9",
    "cards": [{"pci": "0000:1a:00.0"}, {"pci": "0000:1b:00.0"}],
    "knobs": {"SPYRE_X": "1", "VLLM-Y": "2"},
    "artifact": {"id12": "abcdef123456"},
}


def _bench(d):
    (d / f"{TEST}.json").write_text(
        json.dumps({"avg_latency": 1.5, "latencies": [1.4, 1.6]})
    )


def _rows(d):
    return vllm.extract_rows(str(d), "main", "881a59d2", "", "0", "wf", 0)


def _run_props(d):
    (entry,) = vllm.bench_entries(_rows(d))
    return entry["run_props"]


def test_no_sidecars_leaves_run_props_unchanged(tmp_path):
    _bench(tmp_path)
    assert not any(k.startswith(("env_", "phase_")) for k in _run_props(tmp_path))
    assert benchmeta.props_for(str(tmp_path), TEST) == {}


def test_sidecars_are_not_read_as_benchmark_files(tmp_path):
    _bench(tmp_path)
    (tmp_path / "env.json").write_text(json.dumps(ENV))
    (tmp_path / "phases.json").write_text('{"phases": []}')
    assert len(_rows(tmp_path)) == 1


def test_env_becomes_run_props(tmp_path):
    _bench(tmp_path)
    (tmp_path / "env.json").write_text(json.dumps(ENV))
    p = _run_props(tmp_path)
    assert p["env_host"] == "node-1"
    assert p["env_cpu_model"] == "EPYC"
    assert p["env_card_ids"] == "0000:1a:00.0,0000:1b:00.0"
    assert p["artifact_id12"] == "abcdef123456"
    assert p["env_knob_SPYRE_X"] == "1"
    assert p["env_knob_VLLM_Y"] == "2"
    assert p["head_sha"] == "881a59d2"


def test_phases_hosts_outputs_compile(tmp_path):
    _bench(tmp_path)
    (tmp_path / "phases.json").write_text(
        json.dumps(
            {
                "server_startup_sec": 12.5,
                "phases": [
                    {"name": "load", "t_start": 1.0, "t_end": 4.5},
                    {"name": "bad", "t_start": 5, "t_end": 2},
                ],
                "clock": "monotonic",
            }
        )
    )
    (tmp_path / "host_resources.json").write_text(
        json.dumps(
            {
                "load": {"cpu_s": 3.5, "rss_peak_b": 100, "note": "x"},
                "series_file": "host_series.jsonl",
            }
        )
    )
    (tmp_path / "outputs.json").write_text(
        json.dumps({"prompts": 2, "token_sha256": ["a", "b"], "greedy": True})
    )
    (tmp_path / "compile_summary.json").write_text(
        json.dumps({"total_s": 9.0, "cache": {"entries": 3, "hits": 2, "misses": 1}})
    )
    p = _run_props(tmp_path)
    assert p["server_startup_sec"] == "12.5"
    assert p["phase_load_s"] == "3.500"
    assert "phase_bad_s" not in p
    assert p["res_load_cpu_s"] == "3.5"
    assert p["res_load_rss_peak_b"] == "100"
    assert "res_load_note" not in p
    assert p["outputs_prompts"] == "2"
    assert len(p["outputs_token_sha256"]) == 64
    assert p["compile_total_s"] == "9.0"
    assert p["compile_cache_hits"] == "2"


def test_per_test_file_wins_and_malformed_is_skipped(tmp_path):
    _bench(tmp_path)
    (tmp_path / "env.json").write_text(json.dumps(ENV))
    (tmp_path / f"{TEST}.env.json").write_text(json.dumps({"host": "other"}))
    (tmp_path / "phases.json").write_text("{not json")
    (tmp_path / "outputs.json").write_text("[1]")
    p = _run_props(tmp_path)
    assert p["env_host"] == "other"
    assert "env_cpu_model" not in p
    assert not any(k.startswith(("phase_", "outputs_")) for k in p)


def test_sampler_record_file_is_not_a_host_resources_summary(tmp_path):
    _bench(tmp_path)
    (tmp_path / f"{TEST}.host_resources.json").write_text(
        json.dumps({"test_name": TEST, "metric": "host_mem_mb", "value": 5}) + "\n"
    )
    assert not any(
        k.startswith("res_") for k in benchmeta.props_for(str(tmp_path), TEST)
    )


def test_per_test_subdir_is_read_first_and_repeats_ignored(tmp_path):
    _bench(tmp_path)
    (tmp_path / "env.json").write_text(json.dumps(ENV))
    sub = tmp_path / TEST
    sub.mkdir()
    (sub / "env.json").write_text(json.dumps({"host": "subdir"}))
    (sub / "host_resources.json").write_text(json.dumps({"load": {"cpu_s": 2}}))
    (sub / "compile_summary.json").write_text(json.dumps({"total_s": 4}))
    rep = tmp_path / "repeats" / "r1" / TEST
    rep.mkdir(parents=True)
    (rep / "outputs.json").write_text(json.dumps({"prompts": 9}))
    (rep / "env.json").write_text(json.dumps({"host": "repeat"}))
    p = _run_props(tmp_path)
    assert p["env_host"] == "subdir"
    assert p["res_load_cpu_s"] == "2"
    assert p["compile_total_s"] == "4"
    assert "outputs_prompts" not in p
    assert len(_rows(tmp_path)) == 1


@pytest.mark.parametrize(
    "name, doc",
    [
        ("env", {"cards": 5}),
        ("env", {"cards": [{"pci": ["x"]}], "knobs": {"A": 1}, "cpu": {"threads": 1}}),
        ("phases", {"phases": 5}),
        ("phases", {"phases": [{"name": "p", "t_start": 0, "t_end": 10**400}]}),
        ("host_resources", {"load": {"cpu_s": 10**400}}),
        ("host_resources", 5),
    ],
)
def test_wrong_shaped_sidecar_is_skipped_and_never_aborts(tmp_path, name, doc):
    _bench(tmp_path)
    (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    (tmp_path / "compile_summary.json").write_text(json.dumps({"total_s": 9.0}))
    p = _run_props(tmp_path)
    assert p["compile_total_s"] == "9.0"
    assert not any(k.startswith(("phase_", "res_")) for k in p)


def test_a_failing_sidecar_leaves_none_of_its_partial_props(tmp_path):
    _bench(tmp_path)
    (tmp_path / "env.json").write_text(json.dumps({"host": "n1", "cards": 5}))
    assert "env_host" not in _run_props(tmp_path)


def test_host_resources_keys_are_sanitized_and_capped(tmp_path):
    _bench(tmp_path)
    res = {f"p{i}": {"a b/c": 1, "d": 2} for i in range(500)}
    (tmp_path / "host_resources.json").write_text(json.dumps(res))
    got = {k: v for k, v in _run_props(tmp_path).items() if k.startswith("res_")}
    assert len(got) == 200
    assert "res_p0_a_b_c" in got
    assert all(re.fullmatch(r"res_[A-Za-z0-9_]+", k) for k in got)


@pytest.mark.parametrize("name", ["../x", "/etc/x", "a/../../x"])
def test_test_name_cannot_leave_the_results_dir(tmp_path, name):
    outside = tmp_path / "x"
    outside.mkdir()
    (outside / "env.json").write_text(json.dumps({"host": "leak"}))
    results = tmp_path / "results"
    results.mkdir()
    (results / "x").mkdir()
    (results / "x" / "env.json").write_text(json.dumps({"host": "leak"}))
    assert benchmeta.props_for(str(results), name) == {}


def test_a_deeply_nested_sidecar_is_skipped_with_a_warning(tmp_path, caplog):
    _bench(tmp_path)
    (tmp_path / "env.json").write_text("[" * 200000)
    (tmp_path / "compile_summary.json").write_text(json.dumps({"total_s": 9.0}))
    assert _run_props(tmp_path)["compile_total_s"] == "9.0"
    assert "skipped" in caplog.text
