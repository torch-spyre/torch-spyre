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

"""The host resource sampler's `<test_name>.host_resources.json` through the vLLM ingest."""

import json

import pytest

from spyre_clickhouse_ingest import offline, vllm

TEST = "latency_granite8B_tp1_in64_out64"
MODEL = "ibm-granite/granite-3.3-8b-instruct"


def _write(path, *records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _record(metric="host_cpu_pct_total_mean", value=42.1, test_name=TEST, **extra):
    return {"test_name": test_name, "metric": metric, "value": value, **extra}


def _rows(results_dir):
    return vllm.extract_rows(str(results_dir), "main", "881a59d2", "", "0", "wf", 0)


def _extras(rows):
    return [json.loads(r["extra"]) for r in rows]


def _pytorch(results_dir, test_name=TEST, model=MODEL):
    (results_dir / f"{test_name}.pytorch.json").write_text(
        json.dumps(
            [
                {
                    "benchmark": {"name": "latency", "model": model},
                    "metric": {"name": "avg_latency", "benchmark_values": [1.5]},
                }
            ]
        )
    )


@pytest.mark.parametrize(
    "metric, unit",
    [
        ("host_cpu_pct_total_mean", "%"),
        ("compile_host_cpu_pct_total_peak", "%"),
        ("host_mem_mb_p99", "MB"),
        ("compile_host_mem_mb_mean", "MB"),
        ("host_mem_mb", "MB"),
        ("avg_latency", "s"),
        ("mean_ttft_ms", "ms"),
        ("tokens_per_second", "tok/s"),
    ],
)
def test_metric_unit_knows_the_host_resource_units(metric, unit):
    assert vllm.metric_unit(metric) == unit


def test_extract_host_resource_metrics():
    assert vllm.extract_host_resource_metrics(_record(value=3)) == [
        ("host_cpu_pct_total_mean", 3.0)
    ]
    assert vllm.extract_host_resource_metrics({"metric": "m"}) == []
    assert vllm.extract_host_resource_metrics({"value": 1}) == []
    assert vllm.extract_host_resource_metrics(_record(value="n/a")) == []
    assert vllm.extract_host_resource_metrics(_record(value=None)) == []


def test_a_host_resources_file_yields_rows_keyed_by_the_record(tmp_path):
    _write(
        tmp_path / f"{TEST}.host_resources.json",
        _record("host_cpu_pct_total_mean", 42.1, model=MODEL),
        _record("compile_host_mem_mb_peak", 1000.0, model=MODEL),
    )
    rows = _rows(tmp_path)
    assert {r["metric"]: r["actual"] for r in rows} == {
        "host_cpu_pct_total_mean": 42.1,
        "compile_host_mem_mb_peak": 1000.0,
    }
    for extra in _extras(rows):
        assert (extra["test_name"], extra["model"]) == (TEST, MODEL)
        assert "iterations" not in extra


def test_the_test_name_comes_from_the_record_not_the_filename(tmp_path):
    """A dot in the stem would otherwise leave `.host_resources` in the test name."""
    name = "latency_org.model_tp1_in64_out64"
    _write(
        tmp_path / f"{name}.host_resources.json", _record(test_name=name, model=MODEL)
    )
    (extra,) = _extras(_rows(tmp_path))
    assert extra["test_name"] == name


def test_a_record_without_a_test_name_falls_back_to_the_filename(tmp_path):
    record = _record(model=MODEL)
    del record["test_name"]
    _write(tmp_path / f"{TEST}.host_resources.json", record)
    (extra,) = _extras(_rows(tmp_path))
    assert extra["test_name"] == TEST


def test_a_record_without_a_model_takes_it_from_the_pytorch_sibling(tmp_path):
    _pytorch(tmp_path)
    _write(tmp_path / f"{TEST}.host_resources.json", _record())
    host = [
        e
        for r, e in zip(_rows(tmp_path), _extras(_rows(tmp_path)))
        if r["metric"].startswith("host_")
    ]
    assert [e["model"] for e in host] == [MODEL]


def test_a_record_without_a_model_or_sibling_falls_back_to_the_test_name(tmp_path):
    _write(tmp_path / f"{TEST}.host_resources.json", _record())
    (extra,) = _extras(_rows(tmp_path))
    assert extra["model"] == TEST


def test_a_host_resources_file_is_not_also_read_as_a_native_file(tmp_path):
    _write(tmp_path / f"{TEST}.host_resources.json", _record())
    assert len(_rows(tmp_path)) == 1


def test_host_resource_rows_neither_hide_nor_get_hidden_by_pytorch_metrics(tmp_path):
    """The dedup set is per test: a metric the sibling .pytorch.json carries must not drop
    a host-resource row of the same name, and the sibling's n is not lost."""
    _pytorch(tmp_path)
    _write(
        tmp_path / f"{TEST}.json",
        {"avg_latency": 1.5, "latencies": [1.4, 1.6], "percentiles": {}},
    )
    _write(tmp_path / f"{TEST}.host_resources.json", _record(model=MODEL))
    rows = _rows(tmp_path)
    assert sorted(r["metric"] for r in rows) == [
        "avg_latency",
        "host_cpu_pct_total_mean",
    ]
    extras = {r["metric"]: e for r, e in zip(rows, _extras(rows))}
    assert extras["avg_latency"]["model"] == extras["host_cpu_pct_total_mean"]["model"]
    assert "iterations" not in extras["host_cpu_pct_total_mean"]


def test_host_resource_rows_become_benchmark_entries_with_units(tmp_path):
    _write(tmp_path / f"{TEST}.host_resources.json", _record(model=MODEL))
    (entry,) = vllm.bench_entries(_rows(tmp_path))
    assert entry["name"] == TEST
    assert entry["props"]["run_mode"] == "latency"
    assert entry["measurements"] == {"host_cpu_pct_total_mean": [42.1]}
    assert entry["run_props"]["unit.host_cpu_pct_total_mean"] == "%"
    assert entry["iterations"] == 0


# --- offline bundles --------------------------------------------------------------------------


def _bundle_root(tmp_path, name, *records):
    root = tmp_path / "bundle"
    (root / "results").mkdir(parents=True)
    _write(root / "results" / name, *records)
    return root


def test_check_vllm_accepts_a_host_resources_file(tmp_path):
    name = f"{TEST}.host_resources.json"
    root = _bundle_root(tmp_path, name, _record())
    offline.check_vllm(root, {}, [f"results/{name}"])


@pytest.mark.parametrize(
    "records, why",
    [
        ([{"test_name": TEST}], "no host resource metric"),
        ([{"metric": "m"}], "no host resource metric"),
        ([], "no host resource metric"),
    ],
)
def test_check_vllm_rejects_a_host_resources_file_without_a_metric(
    tmp_path, records, why
):
    name = f"{TEST}.host_resources.json"
    root = _bundle_root(tmp_path, name, *records)
    with pytest.raises(offline.BundleError, match=why):
        offline.check_vllm(root, {}, [f"results/{name}"])


def test_check_vllm_checks_the_name_of_a_host_resources_file(tmp_path):
    name = "decode_tp1.host_resources.json"
    root = _bundle_root(tmp_path, name, _record())
    with pytest.raises(offline.BundleError, match="must start with one of"):
        offline.check_vllm(root, {}, [f"results/{name}"])


def test_check_vllm_reads_the_shapes_off_a_host_resources_name(tmp_path):
    """The suffix is stripped first, so `.host_resources` is no shape token's tail."""
    name = f"{TEST}.host_resources.json"
    root = _bundle_root(tmp_path, name, _record())
    with pytest.raises(offline.BundleError, match="differ from perf"):
        offline.check_vllm(
            root, {"perf": {"tensor_parallel": "4"}}, [f"results/{name}"]
        )
