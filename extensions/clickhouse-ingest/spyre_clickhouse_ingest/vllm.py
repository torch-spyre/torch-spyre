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

"""vLLM bench JSON -> spyre_v2 benchmarks / benchmark_runs (report_kind 'vllm').

The one shaping of spyre-inference's perf results, used by its live ingest
(.github/scripts/ingest_vllm_benchmarks.py) and by offline results bundles, so a bundled run's
rows cannot differ from a live one's. The runner writes native `<test_name>.json` (latency /
throughput / serve schemas) and, with SAVE_TO_PYTORCH_BENCHMARK_FORMAT set, vLLM's
`<test_name>.pytorch.json`. test_name is `<run_mode>_<model>_tp<N>_in<N>_out<N>`.
"""

import glob
import json
import logging
import os
import time
from json import JSONDecodeError
from typing import Any

from .client import tables_present
from .writer import benchmarks_already_ingested, insert_benchmarks
from . import schema

log = logging.getLogger(__name__)

BENCH_COMPONENT = "spyre-inference"
REPORT_KIND = "vllm"
RUN_MODES = ("latency", "throughput", "serve")
# In the hash, not merely in props: mode and the input shapes are what separate two runs of
# the same model. Positional: reordering re-keys every benchmark.
BENCH_ID_KEYS = (
    "record_type",
    "run_mode",
    "tensor_parallel",
    "input_len",
    "output_len",
)

# Scalar metrics per vLLM bench schema; list fields (latencies, itls, ttfts, ...) are the raw
# samples behind these aggregates and are skipped.
_LATENCY_METRICS = ("avg_latency",)
_THROUGHPUT_METRICS = ("elapsed_time", "requests_per_second", "tokens_per_second")
_SERVE_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)
_UNITS = {
    "elapsed_time": "s",
    "requests_per_second": "req/s",
    "request_throughput": "req/s",
    "tokens_per_second": "tok/s",
    "output_throughput": "tok/s",
    "total_token_throughput": "tok/s",
}
# The HUD view cannot see the CI coordinates, so it reads them off benchmark_runs.props.
_RUN_PROP_COLUMNS = ("repo", "head_branch", "workflow_id", "run_attempt", "job_id")
_RUN_PROP_EXTRA_KEYS = ("head_sha", "arch", "hardware_type")
_BENCH_TABLES = (schema.BENCHMARKS, schema.BENCHMARK_RUNS)


def read_benchmark_results(filepath: str) -> list[dict[str, Any]]:
    """The records of one JSON file, standard or JSONEachRow."""
    results: list = []
    with open(filepath) as f:
        try:
            r = json.load(f)
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, list):
                results = r
        except JSONDecodeError:
            f.seek(0)
            for line in f:
                try:
                    r = json.loads(line)
                    if isinstance(r, dict):
                        results.append(r)
                    elif isinstance(r, list):
                        results.extend(r)
                except JSONDecodeError:
                    pass
    return results


def extract_vllm_metrics(record: dict[str, Any]) -> list[tuple[str, float]]:
    """(metric, value) pairs of one vLLM-native record.

    The three schemas are disjoint on their signature keys, so a record maps to exactly one.
    `percentiles` (latency) is flattened to `p{percentile}_latency`.
    """
    pairs: list[tuple[str, float]] = []

    def _add(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        pairs.append((name, float(value)))

    if "avg_latency" in record:
        for key in _LATENCY_METRICS:
            if key in record:
                _add(key, record[key])
        percentiles = record.get("percentiles")
        if isinstance(percentiles, dict):
            for pct, value in percentiles.items():
                _add(f"p{pct}_latency", value)
    elif "requests_per_second" in record or "tokens_per_second" in record:
        for key in _THROUGHPUT_METRICS:
            if key in record:
                _add(key, record[key])
    elif "request_throughput" in record or "output_throughput" in record:
        for key in _SERVE_METRICS:
            if key in record:
                _add(key, record[key])
    return pairs


def sample_count(record: dict[str, Any]) -> int:
    """The n behind a native record's aggregates: latency iterations, serve requests, or the
    one timed pass of a throughput run. 0 when the record does not say."""
    if "avg_latency" in record:
        latencies = record.get("latencies")
        return len(latencies) if isinstance(latencies, list) else 0
    if "requests_per_second" in record or "tokens_per_second" in record:
        return 1
    if "request_throughput" in record or "output_throughput" in record:
        completed = record.get("completed")
        return (
            completed
            if isinstance(completed, int) and not isinstance(completed, bool)
            else 0
        )
    return 0


def metric_unit(metric: str) -> str:
    """vLLM's own unit for a metric; its names encode ms but not seconds (latency, p99_latency)."""
    if metric.endswith("_ms"):
        return "ms"
    if metric.endswith("latency"):
        return "s"
    return _UNITS.get(metric, "")


def extract_pytorch_metrics(record: dict[str, Any]) -> list[tuple[str, float]]:
    """(metric, value) pairs of one `*.pytorch.json` record."""
    if "benchmark" not in record or "metric" not in record:
        return []
    metric = record["metric"]
    metric_name = metric.get("name", "unknown")
    return [(metric_name, float(v)) for v in metric.get("benchmark_values", [])]


def bench_name(filename: str) -> str:
    return filename.removesuffix(".pytorch.json").removesuffix(".json")


def model_from_record(record: dict[str, Any], filename: str) -> str:
    """Best-effort model name from a record, falling back to the file's test name."""
    raw_model = record.get("model")
    if isinstance(raw_model, str) and raw_model:
        return raw_model
    benchmark = record.get("benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    model_info = raw_model if isinstance(raw_model, dict) else {}
    return (
        benchmark.get("model")
        or benchmark.get("model_name")
        or model_info.get("name")
        or record.get("model_id")
        or bench_name(filename)
    )


def extract_rows(
    results_dir: str,
    branch: str,
    sha: str,
    run_id: str,
    job_id: str,
    workflow: str,
    pr_number: int,
    arch: str = "x86_64",
    model: str = "",
) -> list[dict[str, Any]]:
    """The flat (results_v3-shaped) rows of every benchmark JSON in `results_dir`.

    The `.pytorch.json` file is the source for every metric it carries (its names are what
    the HUD reads); the native file adds only the metrics it lacks, so no measurement is
    stored twice. `model` names a benchmark whose records name none (a native file with no
    `.pytorch.json` sibling).
    """
    rows = []
    ts = int(time.time() * 1000)
    all_json = set(glob.glob(f"{results_dir}/*.json"))
    pytorch_files = set(glob.glob(f"{results_dir}/*.pytorch.json"))
    native_files = sorted(all_json - pytorch_files)
    log.info(
        "Found %d vLLM-native and %d PyTorch-format benchmark JSON files in %s",
        len(native_files),
        len(pytorch_files),
        results_dir,
    )

    def _emit(
        filename: str, model: str, metric_name: str, value: float, n: int = 0
    ) -> None:
        info = {
            "device": "spyre",
            "arch": arch,
            "hardware_type": "IBM_Spyre",
            "model": model,
            "test_name": bench_name(filename),
            "head_sha": sha,
            "pr_number": pr_number,
            "value": value,
        }
        if n:
            info["iterations"] = n
        rows.append(
            {
                "timestamp": ts,
                "schema_version": "v3",
                "name": "spyre_e2e_benchmark",
                "metric": metric_name,
                "actual": float(value),
                "target": 0.0,
                "repo": "spyre-inference",
                "head_branch": branch,
                "workflow_id": int(run_id) if run_id.isdigit() else 0,
                "job_id": int(job_id) if job_id.isdigit() else 0,
                "run_attempt": 1,
                "extra": json.dumps(info),
            }
        )

    # Native latency/throughput JSON has no model key: take it from the sibling .pytorch.json.
    model_by_test: dict[str, str] = {}
    pytorch_metrics: dict[str, set[str]] = {}
    # Kept apart from the rows: a native record whose metrics all dedup away still has an n.
    samples_by_test: dict[str, int] = {}

    for file, extractor in [
        *[(f, extract_pytorch_metrics) for f in sorted(pytorch_files)],
        *[(f, extract_vllm_metrics) for f in native_files],
    ]:
        filename = os.path.basename(file)
        test_name = bench_name(filename)
        try:
            records = read_benchmark_results(file)
        except Exception:
            log.exception("Failed to read benchmark results from %s", filename)
            continue
        if not records:
            log.warning("No results in %s", filename)
            continue
        before_rows = len(rows)
        for record in records:
            if not isinstance(record, dict):
                continue
            record_model = model_from_record(record, filename)
            is_pytorch = filename.endswith(".pytorch.json")
            if is_pytorch:
                if record_model != test_name:
                    model_by_test[test_name] = record_model
            elif record_model == test_name:
                record_model = model_by_test.get(test_name) or model or record_model
            covered = pytorch_metrics.setdefault(test_name, set())
            n = 0 if is_pytorch else sample_count(record)
            if n:
                samples_by_test.setdefault(test_name, n)
            for metric_name, value in extractor(record):
                if is_pytorch:
                    covered.add(metric_name)
                elif metric_name in covered:
                    continue
                _emit(filename, record_model, metric_name, value, n)
        extracted = len(rows) - before_rows
        if extracted:
            log.info("Extracted %d rows from %s", extracted, filename)
        else:
            log.warning("No usable metrics in %s", filename)

    counted = set()
    for r in rows:
        extra = json.loads(r["extra"])
        if extra.get("iterations"):
            counted.add(extra.get("test_name"))
    for r in rows:
        extra = json.loads(r["extra"])
        name = extra.get("test_name")
        if name in samples_by_test and name not in counted:
            extra["iterations"] = samples_by_test[name]
            r["extra"] = json.dumps(extra)
            counted.add(name)

    log.info("Total rows extracted: %d", len(rows))
    return rows


def parse_input_shapes(test_name: str) -> dict[str, str]:
    """tp1_in64_out64 -> {tensor_parallel, input_len, output_len}: identity discriminators,
    without which tp1 and tp4 are one benchmark."""
    out: dict[str, str] = {}
    for token in (test_name or "").split("_"):
        for prefix, key in (
            ("tp", "tensor_parallel"),
            ("in", "input_len"),
            ("out", "output_len"),
        ):
            rest = token[len(prefix) :]
            if token.startswith(prefix) and rest.isdigit():
                out[key] = rest
    return out


def bench_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The flat rows in insert_benchmarks' entry shape: one entry per (benchmark, metric),
    merged by the writer into one benchmark_runs row per (benchmark, backend).

    iterations is set on ONE entry per benchmark: insert_benchmarks sums it across entries.
    """
    entries = []
    counted: set[str] = set()
    for r in rows:
        extra = json.loads(r["extra"])
        name = extra.get("test_name") or ""
        if not name:
            continue
        props = {"record_type": "model", "run_mode": name.split("_")[0]}
        props.update(parse_input_shapes(name))
        model = extra.get("model", "")
        # Kept only when it says something the name does not already.
        if model and model != name:
            props["model"] = model
        run_props = {k: str(r.get(k, "")) for k in _RUN_PROP_COLUMNS}
        run_props.update({k: str(extra.get(k, "")) for k in _RUN_PROP_EXTRA_KEYS})
        # One key per metric: the writer merges run_props by update.
        unit = metric_unit(r["metric"])
        if unit:
            run_props[f"unit.{r['metric']}"] = unit
        iterations = 0
        if extra.get("iterations") and name not in counted:
            counted.add(name)
            iterations = int(extra["iterations"])
        entries.append(
            {
                "name": name,
                "tags": [],
                # A column, never a hash input: the axis a cross-backend comparison pivots on.
                "backend": extra.get("device", ""),
                "props": props,
                "measurements": {r["metric"]: [float(r["actual"])]},
                "iterations": iterations,
                "run_props": run_props,
                "disc": props,
                "disc_keys": BENCH_ID_KEYS,
            }
        )
    return entries


def duration_s(rows: list[dict[str, Any]]) -> float:
    """The verdict's duration: each throughput run's own elapsed_time."""
    return sum(r["actual"] for r in rows if r.get("metric") == "elapsed_time")


def write_benchmarks(client, db: str, rows, run_id: str) -> int:
    """benchmarks + benchmark_runs for one leg; 0 when the tables are absent or the leg's
    rows already landed. Raises on a write error: the caller decides what it costs."""
    if not tables_present(client, db, tables=_BENCH_TABLES):
        log.info(
            "benchmarks/benchmark_runs absent or stale in %s — v2 perf rows skipped", db
        )
        return 0
    if benchmarks_already_ingested(client, db, run_id, BENCH_COMPONENT, REPORT_KIND):
        log.info("v2 perf rows already present for run_id=%s — skipping", run_id)
        return 0
    n = insert_benchmarks(
        client,
        db,
        BENCH_COMPONENT,
        run_id,
        bench_entries(rows),
        report_kind=REPORT_KIND,
    )
    log.info("Inserted %d benchmark_runs row(s) under run_id=%s", n, run_id)
    return n
