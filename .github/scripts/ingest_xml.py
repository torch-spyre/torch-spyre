#!/usr/bin/env python3
"""
Parses pytest JUnit XML files produced by the Spyre CI pipelines and
batch-inserts the results into ClickHouse.

Supports two XML types:
  1. Pytest JUnit Test-result XMLs  --> test_runs / test_cases / run_properties
  2. Performance benchmark XMLs (classname contains ".benchmark") --> benchmark_runs / perf_benchmarks

Usage (called by the GHA workflow):
    python3 ingest_xml.py \
        --xml-dir xml_artifacts \
        --workflow "model-module-tests" \
        --branch   "main" \
        --sha      "abcdef1234..." \
        --run-id   "12345678" \
        --triggered-at "2026-04-25T14:20:45Z" \
        --pr-number 2271
"""

import argparse
import os
import platform as _platform
import sys

import v2_schema
import uuid
import xml.etree.ElementTree as etree
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import clickhouse_connect
import regex as re

# ---------------------------------------------------------------------------
# Helpers shared by both pipelines
# ---------------------------------------------------------------------------


def _tag_props(tc_el) -> dict:
    """Return a flat dict of tag__ → value parsed from <properties>."""
    result = {}
    props_el = tc_el.find("properties")
    if props_el is None:
        return result
    for p in props_el.findall("property"):
        name = p.get("name", "").strip()
        value = p.get("value", "").strip()
        if name == "tag" and "__" in value:
            key, _, val = value.partition("__")
            result[key] = val
    return result


def _opt_float(d: dict, key: str):
    try:
        return float(d[key])
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
#  BENCHMARK XML detection & parsing
# ---------------------------------------------------------------------------

# Pattern:  perf_{op_name}_{metric}_{ms|MB}_{input_shapes}
# `compiler?` accepts both spellings the perf suite emits: op reports label the
# row "compiler_ms", Granite reports "compile_ms".
_PERF_NAME_RE = re.compile(
    r"^perf_(?P<op>.+?)"
    r"_(?P<metric>wall_clock|cpu|spyre|kernel|memory_transfer|runtime|compiler?|mem_size)"
    r"_(?:ms|MB)(?:_(?P<shapes>.+))?$"
)

_GRANITE_CONFIG_RE = re.compile(r"bs(?P<batch_size>\d+)(?:_pl(?P<prompt_length>\d+))?")


KERNEL_CLASSNAME = "kernel_benchmark"


def is_benchmark_xml(root) -> bool:
    """Return True if every testcase has classname containing 'benchmark'."""
    cases = root.findall(".//testcase")
    if not cases:
        return False
    return all("benchmark" in (tc.get("classname", "")) for tc in cases)


def is_kernel_benchmark_xml(root) -> bool:
    """Return True for spyre-perf-suite's per-kernel breakdown XMLs.

    Must be tested BEFORE is_benchmark_xml(), which also matches these — their
    classname contains 'benchmark' — and would parse them as op benchmarks,
    yielding a run row with zero measurements.
    """
    cases = root.findall(".//testcase")
    if not cases:
        return False
    return all(KERNEL_CLASSNAME in (tc.get("classname", "")) for tc in cases)


def parse_benchmark_xml(
    xml_path: Path, workflow: str = "", ci_run_id: str = "", platform: str = ""
):
    """
    Parse a performance-benchmark XML into (run_meta, list[benchmark_row]).

    Groups the per-op-shape metric cases into one perf_benchmarks row each,
    pivoting the metric values into the appropriate columns.

    Returns:
        run_meta  : dict  – data for benchmark_runs
        benchmarks: list[dict] – data for perf_benchmarks (one row per op+shape)
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    suite = root.find(".//testsuite")
    if suite is None:
        print(f"  [warn] No <testsuite> in {xml_path.name}", file=sys.stderr)
        return None, []

    ts_str = suite.get("timestamp", "")
    try:
        created_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        created_at = datetime.now(UTC)

    # ── extract testsuite-level version_info ───────────────────────────────
    version_info = None
    suite_props = suite.find("properties")
    if suite_props is not None:
        for p in suite_props.findall("property"):
            if p.get("name") == "version_info":
                version_info = p.get("value", "").strip() or None
                break

    # ── group cases by (op_name, input_shapes) ─────────────────────────────
    groups: dict[tuple, dict] = defaultdict(dict)  # (op, shapes) -> {metric: tc_el}

    for tc in suite.findall(".//testcase"):
        name = tc.get("name", "")
        m = _PERF_NAME_RE.match(name)
        if not m:
            print(
                f"  [warn] Unrecognised benchmark name pattern: {name}", file=sys.stderr
            )
            continue
        op = m.group("op")
        metric = m.group("metric")
        if metric == "compiler":  # normalise the op-report spelling to Granite's
            metric = "compile"
        shapes = m.group("shapes") or ""
        groups[(op, shapes)][metric] = tc

    # ── build one row per group ─────────────────────────────────────────────
    benchmarks = []
    for (op_name, shapes_str), metric_cases in groups.items():
        # Use the first available case to read shared tag props
        first_tc = next(iter(metric_cases.values()))
        tags = _tag_props(first_tc)

        # total_duration_ms: prefer wall_clock, fall back to cpu
        total_ms = None
        for preferred in ("wall_clock", "cpu"):
            if preferred in metric_cases:
                total_ms = float(metric_cases[preferred].get("time", 0) or 0)
                break

        cpu_ms = None
        if "cpu" in metric_cases:
            cpu_ms = float(metric_cases["cpu"].get("time", 0) or 0)

        spyre_ms = None
        if "spyre" in metric_cases:
            spyre_ms = float(metric_cases["spyre"].get("time", 0) or 0)

        kernel_ms = None
        if "kernel" in metric_cases:
            kernel_ms = float(metric_cases["kernel"].get("time", 0) or 0)

        mem_ms = None
        if "memory_transfer" in metric_cases:
            mem_ms = float(metric_cases["memory_transfer"].get("time", 0) or 0)

        compile_ms = None
        if "compile" in metric_cases:
            compile_ms = float(metric_cases["compile"].get("time", 0) or 0)

        runtime_ms = None
        if "runtime" in metric_cases:
            runtime_ms = float(metric_cases["runtime"].get("time", 0) or 0)

        # mem_size is a footprint in MB, not a duration, but it still travels in
        # the testcase `time` attribute like every other metric.
        mem_size_mb = None
        if "mem_size" in metric_cases:
            mem_size_mb = float(metric_cases["mem_size"].get("time", 0) or 0)

        # torch_spyre_ms lives in tags of individual cases
        torch_spyre_ms = _opt_float(tags, "torch_spyre_ms")
        ratio = _opt_float(tags, "ratio")

        # regression_status: read from kernel_ms testcase's tag properties
        regression_status = None
        if "kernel" in metric_cases:
            kernel_tags = _tag_props(metric_cases["kernel"])
            regression_status = kernel_tags.get("regression_status")
            if regression_status == "N/A":
                regression_status = None

        is_granite = op_name.startswith("granite_")
        if is_granite:
            config_m = _GRANITE_CONFIG_RE.search(op_name)
            batch_size = int(config_m.group("batch_size")) if config_m else None
            pl_raw = config_m.group("prompt_length") if config_m else None
            prompt_length = int(pl_raw) if pl_raw and pl_raw.isdigit() else None
            config_name = tags.get("config")
            run_mode = tags.get("mode")
            pt_util = _opt_float(tags, "pt_util")
            num_runs_val = _opt_float(tags, "num_runs")
            num_runs_int = int(num_runs_val) if num_runs_val is not None else None
        else:
            batch_size = None
            prompt_length = None
            config_name = None
            run_mode = "op_benchmark"
            pt_util = _opt_float(tags, "pt_util")
            num_runs_val = _opt_float(tags, "num_runs")
            num_runs_int = int(num_runs_val) if num_runs_val is not None else None

        benchmarks.append(
            {
                "benchmark_id": uuid.uuid4().int >> 64,
                "record_type": "model" if is_granite else "op",
                "operation_name": "granite" if is_granite else op_name,
                "config_name": config_name,
                "input_shapes": None if is_granite else (shapes_str or None),
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "run_mode": run_mode,
                "total_duration_ms": total_ms,
                "cpu_ms": cpu_ms,
                "spyre_ms": spyre_ms,
                "kernel_mean_ms": kernel_ms,
                "memory_transfer_mean_ms": mem_ms,
                "compile_ms": compile_ms,
                "runtime_ms": runtime_ms,
                "mem_size_mb": mem_size_mb,
                "pt_util_percent": pt_util,
                "num_runs": num_runs_int,
                "custom_op_file": None,
                "regression_status": regression_status,
                "created_at": created_at,
                "torch_spyre_ms": torch_spyre_ms,
                "ratio": ratio,
            }
        )

    # dedup key: bare basename collides across arches and nights. ci_run_id is
    # stable per run, so a re-ingest of the same file is still a no-op.
    run_key = ci_run_id or created_at.strftime("%Y%m%dT%H%M%SZ")
    source_file = "/".join(p for p in (workflow, run_key, xml_path.name) if p)

    run_meta = {
        "source_file": source_file,
        "created_at": created_at,
        "version_info": version_info,
        "workflow": workflow,
        "platform": platform,
    }
    return run_meta, benchmarks


def parse_kernel_xml(
    xml_path: Path, workflow: str = "", ci_run_id: str = "", platform: str = ""
):
    """Parse a per-kernel breakdown XML into (run_meta, list[kernel_row]).

    One testcase is already one row, so unlike parse_benchmark_xml there is no
    grouping or metric pivoting. Every field is read from the testcase's own
    tag properties rather than parsed out of its name.
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    suite = root.find(".//testsuite")
    if suite is None:
        print(f"  [warn] No <testsuite> in {xml_path.name}", file=sys.stderr)
        return None, []

    try:
        created_at = datetime.fromisoformat(
            suite.get("timestamp", "").replace("Z", "+00:00")
        )
    except ValueError:
        created_at = datetime.now(UTC)

    version_info = None
    suite_props = suite.find("properties")
    if suite_props is not None:
        for p in suite_props.findall("property"):
            if p.get("name") == "version_info":
                version_info = p.get("value", "").strip() or None
                break

    kernels = []
    for tc in suite.findall(".//testcase"):
        tags = _tag_props(tc)
        kernel_name = tags.get("kernel")
        if not kernel_name:
            print(
                f"  [warn] testcase without a kernel tag: {tc.get('name')}",
                file=sys.stderr,
            )
            continue

        operation_name = tags.get("op", "")
        is_granite = operation_name.startswith("granite_")
        num_runs = _opt_float(tags, "num_runs")

        # config__/mode__ carry only full_model|one_block and prefill|decode, so
        # without bs/pl two Granite configs would be indistinguishable here.
        batch_size = prompt_length = None
        if is_granite:
            config_m = _GRANITE_CONFIG_RE.search(operation_name)
            if config_m:
                batch_size = int(config_m.group("batch_size"))
                pl_raw = config_m.group("prompt_length")
                prompt_length = int(pl_raw) if pl_raw and pl_raw.isdigit() else None

        kernels.append(
            {
                "kernel_id": uuid.uuid4().int >> 64,
                "record_type": "model" if is_granite else "op",
                "operation_name": "granite" if is_granite else (operation_name or None),
                "kernel_name": kernel_name,
                # A section's Total is the sum of its siblings; flagged so
                # queries can aggregate without double-counting.
                "is_total": 1 if kernel_name == "Total" else 0,
                "metric": _null_tag(tags.get("metric")),
                "config_name": _null_tag(tags.get("config")),
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "run_mode": _null_tag(tags.get("mode"))
                or (None if is_granite else "op_benchmark"),
                "input_shapes": None
                if is_granite
                else _null_tag(tags.get("input_shape")),
                "duration_ms": _opt_float({"t": tc.get("time")}, "t"),
                "torch_spyre_ms": _opt_float(tags, "torch_spyre_ms"),
                "sendnn_ms": _opt_float(tags, "sendnn_ms"),
                "ratio": _opt_float(tags, "ratio"),
                "pt_util_percent": _opt_float(tags, "pt_util"),
                "num_runs": int(num_runs) if num_runs is not None else None,
                "created_at": created_at,
            }
        )

    run_key = ci_run_id or created_at.strftime("%Y%m%dT%H%M%SZ")
    source_file = "/".join(p for p in (workflow, run_key, xml_path.name) if p)

    run_meta = {
        "source_file": source_file,
        "created_at": created_at,
        "version_info": version_info,
        "workflow": workflow,
        "platform": platform,
        "run_type": "kernel",
    }
    return run_meta, kernels


def _null_tag(value):
    """The perf suite writes the literal 'null'/'N/A' for absent tag values."""
    return None if value in (None, "", "null", "N/A") else value


# ---------------------------------------------------------------------------
# ── BENCHMARK ClickHouse insertion ─────────────────────────────────────────
# ---------------------------------------------------------------------------
# schema-v2 benchmark write path. Same dimension+fact split as test_cases /
# test_case_runs, and the SAME derived run_id, which is what finally lets a perf
# number name the artifact it measured: v1 minted run_id = uuid4().int >> 64 per XML
# file, unrecomputable by anyone, and artifact_results.run_id consequently joined
# benchmark_runs.run_id in 0 of 34 rows.
# ---------------------------------------------------------------------------

# Identity discriminators, NOT measurements: these say which benchmark this is, so
# they belong in the dimension's props and in its hash. batch_size is set on 40/40
# model rows and 0/297 op rows -- a discriminator, not a number measured.
_V2_BENCH_PROP_KEYS = (
    "record_type",
    "config_name",
    "input_shapes",
    "run_mode",
    "kernel_name",
    "is_total",
    "batch_size",
    "prompt_length",
)
# Deliberately NOT here and NOT in the id hash: `metric`. It selects the backend, so
# the same kernel measured on cpu and on spyre is ONE benchmark with two backend
# rows -- putting it in the identity would split them and make the comparison a
# cross-identity join instead of a self-join.

# Everything the producer measured, keyed verbatim. A Map, not columns: the v1
# sparsity is per record_type (mem_size_mb 152/297 op vs 0/40 model, batch_size the
# inverse), so no wide column set fits and each new metric would need a DDL change.
_V2_BENCH_METRIC_KEYS = (
    "total_duration_ms",
    "cpu_ms",
    "spyre_ms",
    "kernel_mean_ms",
    "memory_transfer_mean_ms",
    "compile_ms",
    "runtime_ms",
    "mem_size_mb",
    "pt_util_percent",
    "duration_ms",
    "torch_spyre_ms",
    "sendnn_ms",
    "ratio",
)


# In the benchmark_id hash, not merely in props: one operation_name occurs at more
# than one record_type in prod (granite as model AND op, matmul/attention likewise),
# and the config keys separate the granite variants, so hashing name+tags alone
# merges genuinely different benchmarks into one identity.
_V2_BENCH_ID_KEYS = (
    "record_type",
    "config_name",
    "input_shapes",
    "run_mode",
    "kernel_name",
    "is_total",
)


def v2_benchmark_id(name: str, tags, disc=None) -> str:
    """uuid5 over name + sorted tags + the identity discriminators in
    _V2_BENCH_ID_KEYS. Same refuse-on-empty rule as v2_test_case_id: an empty name
    still hashes to a real uuid, so every unidentifiable benchmark would collide on
    ONE id rather than merely being orphaned."""
    if not _v2_norm(name):
        return ""
    tag_part = ",".join(sorted({_v2_norm(t) for t in (tags or []) if _v2_norm(t)}))
    disc = disc or {}
    disc_part = ",".join(f"{k}={_v2_norm(disc.get(k))}" for k in _V2_BENCH_ID_KEYS)
    return str(uuid.uuid5(V2_NAMESPACE, f"{_v2_norm(name)}|{tag_part}|{disc_part}"))


def v2_benchmark_tables_present(client) -> bool:
    return _table_exists(client, "benchmarks") and _table_exists(
        client, "benchmark_runs"
    )


def v2_benchmarks_already_ingested(client, run_id: str) -> bool:
    """benchmark_runs is a plain MergeTree with no dedup key, so a re-ingest
    doubles every measurement behind an average."""
    rows = client.query(
        "SELECT count() FROM benchmark_runs WHERE run_id = {run_id:UUID}",
        parameters={"run_id": run_id},
    ).result_rows
    return bool(rows and rows[0][0] > 0)


# perf_kernels.metric is the real backend axis: cpu_kernel_ms on 16,734 prod rows,
# spyre_kernel_ms on 3,475. Its torch_spyre_ms/sendnn_ms/ratio columns are NULL on all
# 20,209 rows, so the comparison v1 looks like it stores was never actually written.
_V2_BACKEND_BY_METRIC = {
    "cpu_kernel_ms": "cpu",
    "spyre_kernel_ms": "spyre",
    "sendnn_ms": "sendnn",
}


def _v2_bench_backend(rec: dict) -> str:
    """Which implementation produced these numbers, so the same benchmark measured on
    two backends compares by self-join instead of by a stored ratio that can disagree
    with its operands."""
    metric = (rec.get("metric") or "").strip()
    if metric in _V2_BACKEND_BY_METRIC:
        return _V2_BACKEND_BY_METRIC[metric]
    if rec.get("sendnn_ms") is not None and rec.get("torch_spyre_ms") is None:
        return "sendnn"
    return "torch-spyre"


def insert_benchmarks_v2(client, run_id: str, records: list) -> int:
    """Write benchmarks (identity) + benchmark_runs (measurements) for one run.

    Dropped from v2 deliberately: regression_status and ratio (verdicts with no
    recorded baseline -- derived in v_benchmark_regression / v_benchmark_backend_compare
    instead), and every run-context column (reached through run_id).
    """
    if not records:
        return 0
    ident_rows, fact_rows = {}, []
    skipped = 0
    for rec in records:
        name = rec.get("operation_name") or ""
        tags = sorted({t for t in (rec.get("tags") or []) if t})
        bid = v2_benchmark_id(name, tags, rec)
        if not bid:
            skipped += 1
            continue
        props = {
            k: str(rec[k])
            for k in _V2_BENCH_PROP_KEYS
            if rec.get(k) is not None and str(rec[k]) != ""
        }
        measurements = {
            k: float(rec[k]) for k in _V2_BENCH_METRIC_KEYS if rec.get(k) is not None
        }
        if not measurements:
            # chk_measurements refuses an empty map: a benchmark row that measured
            # nothing is a parse failure, not a result.
            skipped += 1
            continue
        ident_rows[bid] = [bid, name, tags, props]
        num_runs = rec.get("num_runs")
        fact_rows.append(
            [
                run_id,
                bid,
                _v2_bench_backend(rec),
                measurements,
                int(num_runs) if num_runs is not None else 0,
                {},
            ]
        )
    client.insert(
        "benchmarks",
        v2_new_identity_rows(client, "benchmarks", "benchmark_id", ident_rows),
        column_names=["benchmark_id", "name", "tags", "props"],
    )
    client.insert(
        "benchmark_runs",
        fact_rows,
        column_names=[
            "run_id",
            "benchmark_id",
            "backend",
            "measurements",
            "iterations",
            "props",
        ],
    )
    if skipped:
        print(
            f"  [warn] v2: {skipped} benchmark(s) skipped -- no derivable "
            f"benchmark_id or no measurements",
            file=sys.stderr,
        )
    return len(fact_rows)


# ---------------------------------------------------------------------------


def insert_benchmark_run(client, run_id: int, run_meta: dict) -> None:
    values = {
        "run_id": run_id,
        "source_file": run_meta["source_file"],
        "version_info": run_meta.get("version_info"),
        "created_at": run_meta["created_at"].replace(tzinfo=None),
        "workflow": run_meta.get("workflow", ""),
        "platform": run_meta.get("platform", ""),
        # Marks the two kernel rows so they don't read as runs that measured
        # nothing. Dropped when the migration adding it has not been applied.
        "run_type": run_meta.get("run_type", "benchmark"),
    }
    columns = list(values)
    if _absent_columns(client, "benchmark_runs", ("run_type",)):
        print(
            "  [warn] benchmark_runs has no run_type — storing this run without "
            "it. Apply the spyre-dashboard migration to capture it.",
            file=sys.stderr,
        )
        columns.remove("run_type")
    client.insert(
        "benchmark_runs",
        [[values[c] for c in columns]],
        column_names=columns,
    )


_PERF_BENCHMARK_COLUMNS = [
    "benchmark_id",
    "run_id",
    "record_type",
    "operation_name",
    "config_name",
    "input_shapes",
    "batch_size",
    "prompt_length",
    "run_mode",
    "total_duration_ms",
    "cpu_ms",
    "spyre_ms",
    "kernel_mean_ms",
    "memory_transfer_mean_ms",
    "compile_ms",
    "runtime_ms",
    "mem_size_mb",
    "pt_util_percent",
    "num_runs",
    "custom_op_file",
    "regression_status",
    "created_at",
]

# Added to perf_benchmarks by a spyre-dashboard migration, which deploys
# independently of this script. See insert_perf_benchmarks.
_PERF_BENCHMARK_OPTIONAL_COLUMNS = ("compile_ms", "runtime_ms", "mem_size_mb")


def _absent_columns(client, table: str, columns) -> set[str]:
    rows = client.query(
        "SELECT name FROM system.columns "
        "WHERE database = currentDatabase() AND table = {t:String}",
        parameters={"t": table},
    ).result_rows
    present = {r[0] for r in rows}
    return {c for c in columns if c not in present}


def _table_exists(client, table: str) -> bool:
    rows = client.query(
        "SELECT count() FROM system.tables "
        "WHERE database = currentDatabase() AND name = {t:String}",
        parameters={"t": table},
    ).result_rows
    return bool(rows and rows[0][0])


def insert_perf_benchmarks(client, run_id: int, benchmarks: list[dict]) -> None:
    if not benchmarks:
        return

    columns = list(_PERF_BENCHMARK_COLUMNS)

    # Drop the op-cost columns rather than failing when the migration adding them
    # has not been applied to this database. The benchmark_runs row is already
    # committed by now and the dedup check keys on it, so raising here would skip
    # the run on every retry and lose its metrics for good.
    absent = _absent_columns(
        client, "perf_benchmarks", _PERF_BENCHMARK_OPTIONAL_COLUMNS
    )
    if absent:
        print(
            f"  [warn] perf_benchmarks has no {', '.join(sorted(absent))} — "
            f"storing this run without them. Apply the spyre-dashboard migration "
            f"to capture them.",
            file=sys.stderr,
        )
        columns = [c for c in columns if c not in absent]

    def cell(b: dict, column: str):
        if column == "run_id":
            return run_id
        if column == "created_at":
            return b["created_at"].replace(tzinfo=None)
        return b[column]

    client.insert(
        "perf_benchmarks",
        [[cell(b, c) for c in columns] for b in benchmarks],
        column_names=columns,
    )


# perf_kernels and benchmark_runs.run_type come from a spyre-dashboard migration,
# not from this script. The two repos deploy independently, so the inserts below
# check what the target database actually has and degrade with a warning rather
# than raise: the benchmark_runs row is committed before the kernel insert and the
# dedup check keys on it, so raising would skip the run on every retry.
_PERF_KERNEL_COLUMNS = [
    "kernel_id",
    "run_id",
    "record_type",
    "operation_name",
    "kernel_name",
    "is_total",
    "metric",
    "config_name",
    "batch_size",
    "prompt_length",
    "run_mode",
    "input_shapes",
    "duration_ms",
    "torch_spyre_ms",
    "sendnn_ms",
    "ratio",
    "pt_util_percent",
    "num_runs",
    "created_at",
]


def insert_perf_kernels(client, run_id: int, kernels: list[dict]) -> None:
    if not kernels:
        return
    client.insert(
        "perf_kernels",
        [
            [
                k["kernel_id"],
                run_id,
                k["record_type"],
                k["operation_name"],
                k["kernel_name"],
                k["is_total"],
                k["metric"],
                k["config_name"],
                k["batch_size"],
                k["prompt_length"],
                k["run_mode"],
                k["input_shapes"],
                k["duration_ms"],
                k["torch_spyre_ms"],
                k["sendnn_ms"],
                k["ratio"],
                k["pt_util_percent"],
                k["num_runs"],
                k["created_at"].replace(tzinfo=None),
            ]
            for k in kernels
        ],
        column_names=_PERF_KERNEL_COLUMNS,
    )


# ---------------------------------------------------------------------------
# TEST-RESULT XML
# ---------------------------------------------------------------------------


def classify_testcase(tc_el):
    failure_el = tc_el.find("failure")
    error_el = tc_el.find("error")
    skipped_el = tc_el.find("skipped")

    if error_el is not None:
        msg = (error_el.get("message", "") + "\n" + (error_el.text or "")).strip()
        return "error", msg

    if failure_el is not None:
        ftype = (failure_el.get("type") or "").lower()
        msg = (failure_el.get("message", "") + "\n" + (failure_el.text or "")).strip()
        if "xfail" in ftype:
            return "xpass", msg
        return "failed", msg

    if skipped_el is not None:
        stype = (skipped_el.get("type") or "").lower()
        msg = (skipped_el.get("message") or skipped_el.text or "").strip()
        if "xfail" in stype:
            return "xfail", msg
        return "skipped", msg

    return "passed", ""


def extract_properties(tc_el):
    props = []
    props_el = tc_el.find("properties")
    if props_el is None:
        return props
    for p in props_el.findall("property"):
        name = p.get("name", "").strip()
        value = p.get("value", "").strip()
        if name:
            props.append((name, value))
    return props


def extract_op_dtype_platform(name: str, properties: list[tuple[str, str]]):
    op_name = ""
    dtype = ""
    platform = ""
    for pname, pvalue in properties:
        if pname.startswith("op__"):
            op_name = pname[4:]
        elif pname.startswith("dtype__"):
            dtype = pname[7:]
        elif pname.startswith("platform__"):
            platform = pname[10:]
        elif pname == "tag":
            if pvalue.startswith("op__"):
                op_name = pvalue[4:]
            elif pvalue.startswith("dtype__"):
                dtype = pvalue[7:]
            elif pvalue.startswith("platform__"):
                platform = pvalue[10:]

    if not dtype:
        for d in [
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
            "complex64",
            "complex128",
        ]:
            if d in name:
                dtype = d
                break
    return op_name, dtype, platform


def promote_xpass(raw_cases, suite_attrs):
    failures = int(suite_attrs.get("failures", 0))
    true_fail_raw = sum(1 for c in raw_cases if c["status"] in ("failed", "error"))
    strict_xpass_raw = sum(1 for c in raw_cases if c["status"] == "xpass")
    non_strict = max(0, failures - true_fail_raw - strict_xpass_raw)

    promoted = 0
    for c in raw_cases:
        if promoted >= non_strict:
            break
        if c["_is_bare"]:
            c["status"] = "xpass"
            promoted += 1


def parse_test_xml(xml_path: Path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    suites = root.findall(".//testsuite")
    if not suites:
        print(f"  [warn] No <testsuite> found in {xml_path.name}", file=sys.stderr)
        return None, []

    suite = suites[0]
    suite_attrs = suite.attrib

    ts_str = suite_attrs.get("timestamp", "")
    try:
        triggered_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        triggered_at = datetime.now(UTC)

    raw_cases = []
    for tc in suite.findall(".//testcase"):
        status, fail_msg = classify_testcase(tc)
        properties = extract_properties(tc)
        op_name, dtype, platform = extract_op_dtype_platform(
            tc.get("name", ""), properties
        )
        raw_cases.append(
            {
                "case_id": str(uuid.uuid4()),
                "classname": tc.get("classname", ""),
                "name": tc.get("name", ""),
                "op_name": op_name,
                "dtype": dtype,
                "platform": platform,
                "status": status,
                "duration_s": float(tc.get("time", 0) or 0),
                "fail_message": fail_msg,
                "properties": properties,
                "_is_bare": (status == "passed"),
                "triggered_at": triggered_at,
            }
        )

    promote_xpass(raw_cases, suite_attrs)

    counts = Counter(c["status"] for c in raw_cases)
    platform = next((c["platform"] for c in raw_cases if c["platform"]), "")
    run = {
        "suite_name": suite_attrs.get("name", xml_path.stem),
        "filename": xml_path.name,
        "platform": platform,
        "triggered_at": triggered_at,
        "total_tests": len(raw_cases),
        "passed": counts.get("passed", 0),
        # error is counted INSIDE failed on purpose, and `errors` below is a subset,
        # not an additional bucket: passed+failed+skipped+xfail+xpass must equal
        # total_tests, which holds on all 341,161 prod test_runs rows. Splitting error
        # out of failed would break that invariant for every consumer.
        "failed": counts.get("failed", 0) + counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "xfail": counts.get("xfail", 0),
        "errors": counts.get("error", 0),
        "xpass": counts.get("xpass", 0),
        "duration_s": float(suite_attrs.get("time", 0) or 0),
    }
    return run, raw_cases


# ---------------------------------------------------------------------------
# ── TEST-RESULT ClickHouse insertion (unchanged) ───────────────────────────
# ---------------------------------------------------------------------------


def get_client(database: str | None = None):
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=database or os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


def get_v2_client():
    """A SECOND connection, bound to the v2 database, or None when none is configured.

    v2 needs its own connection rather than sharing v1's: `benchmark_runs` exists in BOTH
    generations with incompatible shapes -- v1's has (run_id UInt64, source_file), v2's has
    (run_id UUID) and no source_file -- so the two dedup queries, both naming the table
    unqualified, cannot both resolve correctly through one `database=`. Verified on prod:
    spyre.benchmark_runs has source_file, spyre_v2.benchmark_runs does not.

    Returns None when CLICKHOUSE_DB_V2 is unset, which is what makes --schema v1 (the default)
    cost nothing: no second connection is opened.
    """
    db = os.environ.get("CLICKHOUSE_DB_V2", "").strip()
    if not db:
        return None
    return get_client(database=db)


def insert_run(client, run_id: str, run: dict, args):
    client.insert(
        "test_runs",
        [
            [
                run_id,
                args.workflow,
                run["suite_name"],
                run["filename"],
                run["platform"],
                args.branch,
                (args.sha or "").ljust(40)[:40],
                int(args.pr_number) if args.pr_number.strip() else 0,
                _runner_run_id(args, run_id),
                run["triggered_at"].replace(tzinfo=None),
                run["total_tests"],
                run["passed"],
                run["failed"],
                run["skipped"],
                run["xfail"],
                run["errors"],
                run["xpass"],
                run["duration_s"],
                getattr(args, "trigger_type", "") or "unknown",
            ]
        ],
        column_names=[
            "run_id",
            "workflow",
            "suite_name",
            "filename",
            "platform",
            "branch",
            "commit_sha",
            "pr_number",
            "runner_run_id",
            "triggered_at",
            "total_tests",
            "passed",
            "failed",
            "skipped",
            "xfail",
            "errors",
            "xpass",
            "duration_s",
            "test_type",
        ],
    )


def insert_cases(client, run_id: str, cases: list[dict], workflow: str = ""):
    if not cases:
        return
    client.insert(
        "test_cases",
        [
            [
                run_id,
                c["case_id"],
                c["classname"],
                c["name"],
                c["op_name"],
                c["dtype"],
                c["status"],
                c["duration_s"],
                c["fail_message"][:8192],
                c["triggered_at"].replace(tzinfo=None),
                workflow,
            ]
            for c in cases
        ],
        column_names=[
            "run_id",
            "case_id",
            "classname",
            "name",
            "op_name",
            "dtype",
            "status",
            "duration_s",
            "fail_message",
            "triggered_at",
            "workflow",
        ],
    )


def insert_properties(client, run_id: str, cases: list[dict]):
    rows = [
        {
            "run_id": run_id,
            "case_id": c["case_id"],
            "prop_name": pname,
            "prop_value": pvalue,
            "triggered_at": c["triggered_at"],
        }
        for c in cases
        for pname, pvalue in c["properties"]
    ]
    if rows:
        client.insert(
            "run_properties",
            [
                [
                    r["run_id"],
                    r["case_id"],
                    r["prop_name"],
                    r["prop_value"],
                    r["triggered_at"].replace(tzinfo=None),
                ]
                for r in rows
            ],
            column_names=[
                "run_id",
                "case_id",
                "prop_name",
                "prop_value",
                "triggered_at",
            ],
        )


# ---------------------------------------------------------------------------
# ── SCHEMA v2: test_cases + test_case_runs ─────────────────────────────────
#
# ADDITIVE. Everything above still writes the v1 tables exactly as before; this
# path writes the two v2 tables alongside them and is skipped entirely if they do
# not exist, so the script is safe to deploy before the v2 migration lands.
#
# Both ids are DERIVED, never minted. Four writers (this script, the two sibling
# product ingests, and the orchestrator's pushToClickhouse.pushArtifactResult)
# compute them independently with no threading contract -- which is the only thing
# that makes the tables joinable: v1 minted four unrelated identity schemes and
# artifact_results.run_id consequently joined test_runs.run_id in 2 of 1,266 rows.
#
# BYTE-EXACTNESS IS THE CONTRACT. Disagree about the namespace, the separator, the
# field order or the normalisation and you mint a different uuid for the same row --
# and an orphaned row is indistinguishable from "no tests ran", so the failure is
# silent. The reference implementation, its rule list and the golden values every
# port must reproduce live in spyre-frameworks pipelines/lib/run_identity.py and
# pipelines/lib/test_run_identity.py. Keep this block in sync with it.
# ---------------------------------------------------------------------------

# The product this script ingests for. Replaces v1's hf_/si_ table-name prefixes: one
# v2 table pair serves all three products, discriminated by this column. It is also a
# test_case_id hash input, so it cannot drift from the identity it is stamped on.
V2_COMPONENT = "torch-spyre"

V2_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")
V2_SEP = "|"


def _v2_norm(value) -> str:
    """Canonical scalar form. Lowercasing is not cosmetic: the same tier arrives as
    'Regression' from a Jenkins parameter and 'regression' from a GHA input."""
    return ("" if value is None else str(value)).strip().lower()


def v2_canonical_arch(arch) -> str:
    """amd64/x86/x86-64 all mean x86_64 -- a leg labelled 'amd64' by Jenkins and
    'x86_64' by GHA is ONE leg, and must hash as one."""
    a = _v2_norm(arch)
    return "x86_64" if a in ("amd64", "x86", "x86-64", "x86_64") else a


def v2_run_id(source: str, external_run_id: str, arch: str, test_type: str) -> str:
    """Identity of one TEST-EXECUTION LEG: (source, external_run_id, arch, test_type).

    arch and test_type are IN the key because the real execution grain measured
    (run, arch, tier) at 19,867 legs under 16,381 CI runs. external_run_id is a
    STRING: typed numerically, every Jenkins leg would be 0 and collide into one id.
    Returns '' when a field is missing -- an all-defaults hash is a real uuid that
    every incomplete leg would share, which is worse than a blank.
    """
    fields = (source, external_run_id, arch, test_type)
    if not all(_v2_norm(f) for f in fields):
        return ""
    return str(
        uuid.uuid5(
            V2_NAMESPACE,
            V2_SEP.join(
                (
                    _v2_norm(source),
                    _v2_norm(external_run_id),
                    v2_canonical_arch(arch),
                    _v2_norm(test_type),
                )
            ),
        )
    )


def v2_test_case_id(component: str, classname: str, name: str, tags) -> str:
    """Content identity of a TEST, so the same test reconciles across runs. v1 minted
    uuid4 per row: 37,322,701 identities for 58,711 distinct (classname, name) pairs.

    `tags` are deduped and SORTED -- they are a set and source order is incidental,
    so an unsorted join makes two writers disagree about the same test. They are
    INSIDE the hash, so re-tagging mints a new identity; trend queries must
    therefore group on (component, classname, name), never on test_case_id.
    """
    if not (_v2_norm(component) and _v2_norm(name)):
        # Same collision hazard as v2_run_id: an empty field still hashes to a real,
        # stable uuid that every other such case shares. The v2 table's CONSTRAINTs
        # reject component='' / name='' anyway. classname is legitimately empty for a
        # module-level test, so it is NOT required.
        return ""
    norm = sorted({t for t in (_v2_norm(x) for x in (tags or [])) if t})
    return str(
        uuid.uuid5(
            V2_NAMESPACE,
            V2_SEP.join(
                (
                    _v2_norm(component),
                    _v2_norm(classname),
                    _v2_norm(name),
                    ",".join(norm),
                )
            ),
        )
    )


def v2_tags_for_case(case: dict) -> list:
    """The case's tags as an ARRAY of `namespace__value` strings.

    Array, not Map: `testtype` carries up to 5 values on 91.7% of cases, so a Map
    would silently keep one and drop the rest. The v1 shape is a (prop_name,
    prop_value) list where the only prop_name is literally 'tag' and the real
    key is encoded inside the value -- so the VALUE is the tag.
    """
    tags = set()
    for pname, pvalue in case.get("properties", []) or []:
        if pname == "tag":
            if pvalue:
                tags.add(pvalue)
        elif "__" in pname:
            # Some emitters put the namespace__value in the property NAME instead.
            tags.add(pname)
    return sorted(tags)


def v2_source_and_external_run_id(args, run_id: str):
    """(source, external_run_id) for this leg, from whichever CI dispatched it.

    A numeric --gha-run-id means GHA dispatched it. Otherwise the leg is
    Jenkins-dispatched and its own externalizable id ('folder/job#123') is the run
    coordinate -- the SAME value the orchestrator hashes on its side of the join, so
    neither side has to thread a minted uuid.
    `source` is required precisely because a GHA run id and a Jenkins build number
    share a number space.
    """
    gha = (getattr(args, "gha_run_id", "") or "").strip()
    if gha:
        try:
            int(gha)
            return "gha", gha
        except (ValueError, TypeError):
            pass
    jenkins_key = (getattr(args, "jenkins_run_key", "") or "").strip()
    if jenkins_key:
        return "jenkins", jenkins_key
    # No CI coordinate at all: fall back to the run uuid so the rows are still
    # self-consistent and joinable WITHIN this ingest, just not to an artifact.
    return "local", run_id


def v2_tables_present(client) -> bool:
    """v2 write path is skipped unless BOTH tables exist, so this script can be
    deployed before the migration without erroring on every run."""
    return _table_exists(client, "test_case_runs") and _table_exists(
        client, "test_cases"
    )


def v2_already_ingested(client, run_id: str, component: str) -> bool:
    """test_case_runs is a plain MergeTree with no dedup key, so a double ingest of one
    leg DOUBLES its counts -- and the v2 schema dropped the stored counters precisely
    because they are derived from these rows. This check is what keeps that correct.
    Scoped by component as well as run_id to hit the ORDER BY prefix."""
    rows = client.query(
        "SELECT count() FROM test_case_runs "
        "WHERE component = {component:String} AND run_id = {run_id:UUID}",
        parameters={"component": component, "run_id": run_id},
    ).result_rows
    return bool(rows and rows[0][0] > 0)


def v2_new_identity_rows(client, table: str, id_col: str, ident_rows: dict) -> list:
    """Return only the identity rows this dimension does not already hold.

    Both dimensions are plain MergeTree, so re-inserting a known identity appends a
    duplicate row rather than collapsing it: one benchmark seen in 36 runs became 36
    rows, and every reader then has to remember to dedup. Deduping in-run is not
    enough because the collision is ACROSS runs.
    """
    if not ident_rows:
        return []
    ids = list(ident_rows)
    known = {
        r[0]
        for r in client.query(
            f"SELECT {id_col} FROM {table} WHERE {id_col} IN {{ids:Array(UUID)}}",
            parameters={"ids": ids},
        ).result_rows
    }
    return [
        row for i, row in ident_rows.items() if str(i) not in {str(k) for k in known}
    ]


def insert_v2(client, component: str, run_id: str, cases: list) -> int:
    """Write test_cases (identity) + test_case_runs (outcome) for one leg.

    Rows are built as dicts and ordered by v2_schema, so a field cannot be assigned to the
    wrong column and the column order lives in exactly one place.

    Dropped from v2 deliberately: filename, suite_name, runner_run_id, and every stored
    counter -- all derivable, and a stored counter invites drift.
    """
    if not cases:
        return 0
    ident_rows, run_rows = {}, []

    skipped_unidentifiable = 0
    for c in cases:
        tags = v2_tags_for_case(c)
        classname, name = c.get("classname", ""), c.get("name", "")
        tcid = v2_test_case_id(component, classname, name, tags)
        if not tcid:
            # Refused identity: writing the row anyway would collide it with every
            # other unidentifiable case rather than merely orphaning it.
            skipped_unidentifiable += 1
            continue
        # Keyed by id: identical identity rows within a leg are one fact.
        ident_rows[tcid] = {
            "test_case_id": tcid,
            "component": component,
            "classname": classname,
            "name": name,
            "tags": tags,
        }
        run_rows.append(
            {
                "run_id": run_id,
                "test_case_id": tcid,
                "component": component,
                "status": c.get("status", ""),
                "duration_s": float(c.get("duration_s", 0) or 0),
                "fail_message": (c.get("fail_message") or "")[:8192],
            }
        )
    # Cross-run dedup, not just in-leg: test_cases is a plain MergeTree, so re-inserting a
    # known identity appends a duplicate instead of collapsing it.
    v2_schema.insert_identities(client, v2_schema.TEST_CASES, ident_rows)
    v2_schema.insert(client, v2_schema.TEST_CASE_RUNS, run_rows)
    if skipped_unidentifiable:
        print(
            f"  [warn] v2: {skipped_unidentifiable} case(s) skipped -- identity not derivable",
            file=sys.stderr,
        )
    return len(run_rows)

# ---------------------------------------------------------------------------
# ── Main ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def _runner_run_id(args, run_id: str) -> str:
    """This leg's own run id: --gha-run-id when GHA-dispatched, else the same uuid as run_id."""
    raw = (getattr(args, "gha_run_id", "") or "").strip()
    if raw:
        try:
            int(raw)
            return raw
        except (ValueError, TypeError):
            pass
    return run_id


def _threaded_run_id(args) -> str:
    """--run-id when it is a real UUID, else "" so the caller mints one.

    The flag has always carried a Jenkins BUILD_NUMBER historically, which is not a UUID and
    must not land in test_runs.run_id (a UUID column). Only a well-formed uuid is honoured.
    """
    raw = (getattr(args, "run_id", "") or "").strip()
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=None)
    parser.add_argument("--xml-file", default=None)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--gha-run-id", default="")
    parser.add_argument("--triggered-at", default="")
    parser.add_argument("--pr-number", default="")
    parser.add_argument(
        "--jenkins-run-key",
        default="",
        help="This leg's own Jenkins externalizable id, e.g. 'Spyre/component-build#417'. "
        "Hashed into the schema-v2 run_id, which is how the orchestrator's "
        "artifact_results row and these per-case rows join without threading a uuid.",
    )
    parser.add_argument(
        "--trigger-type",
        default="",
        help="Suite tier that produced this run, e.g. regression | integration | unit | smoke",
    )
    parser.add_argument(
        "--platform",
        default=_platform.machine() or "",
        help="Hardware arch the run executed on (x86_64 | ppc64le | s390x). "
        "The benchmark XML carries no per-case platform tag, so the caller "
        "supplies it; defaults to the ingest host's arch.",
    )
    # Which schema generation to write. Defaults to v1 ONLY, so an un-updated caller keeps
    # behaving exactly as before -- this script runs from inside a BAKED image, so old images
    # and new ones coexist for as long as it takes every product image to be rebuilt.
    #
    # v1 is not a permanent home: test_runs, run_properties, perf_benchmarks and perf_kernels
    # have NO v2 equivalent because v2 replaces them outright -- run_properties becomes
    # test_cases.tags, test_runs is derivable from test_case_runs, and the two perf tables
    # collapse into benchmarks + benchmark_runs. The v2 DDL in spyre-frameworks deliberately
    # does not define them. Both is the migration window; v2 is the destination.
    parser.add_argument(
        "--schema",
        choices=["v1", "v2", "both"],
        default=os.environ.get("INGEST_SCHEMA", "v1"),
        help="Which schema generation to write: v1 (default, the legacy tables), v2 (the "
        "replacement tables only), or both (the migration window). Also settable via "
        "INGEST_SCHEMA so a workflow can set it once for every leg.",
    )
    args = parser.parse_args()
    # Resolved once here rather than re-tested at each call site, so the two paths cannot
    # drift into disagreeing about what was asked for.
    args.write_v1 = args.schema in ("v1", "both")
    args.write_v2 = args.schema in ("v2", "both")
    print(f"  schema={args.schema} (v1={args.write_v1} v2={args.write_v2})")

    if args.xml_file:
        xml_files = [Path(args.xml_file)]
    elif args.xml_dir:
        xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    else:
        print("Error: provide --xml-dir or --xml-file")
        sys.exit(1)

    if not xml_files:
        print("No XML files found — nothing to ingest.")
        sys.exit(0)

    print(
        f"Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', '443')} ..."
    )
    client = get_client()
    # Separate connection for the v2 tables -- see get_v2_client() for why sharing v1's
    # cannot work. None when CLICKHOUSE_DB_V2 is unset, which every v2 site treats as
    # "v2 not configured" and skips.
    v2client = get_v2_client() if args.write_v2 else None
    if args.write_v2 and v2client is None:
        print(
            "  WARN --schema asked for v2 but CLICKHOUSE_DB_V2 is unset — v2 rows skipped",
            file=sys.stderr,
        )
    client.command("SELECT 1")
    print("Connected.\n")

    # No schema mutation here, deliberately. This used to ALTER benchmark_runs on EVERY run to
    # add workflow/platform -- a migration in the wrong place: it demanded DDL rights on every
    # invocation, reshaped a table other producers share, and ran before any XML was read, so
    # under --schema v2 it failed the whole ingest with UNKNOWN_TABLE for a v1 table nothing
    # was going to write. Both columns have been live on prod for months, and the v2 tables
    # have neither and need neither. Schema changes belong in the DDL, not in the writer;
    # _absent_columns() below already degrades gracefully if a column really is missing.

    total_cases = 0
    total_benchmarks = 0
    total_kernels = 0

    for xml_path in xml_files:
        print(f"Processing: {xml_path.name}")

        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        # ── Dispatch: kernel breakdown vs benchmark vs test-result ─────────
        # Kernel first: is_benchmark_xml() also matches these.
        if is_kernel_benchmark_xml(root):
            print("  Detected: per-kernel breakdown XML")
            # Both halves of the migration are required, and it can land partially.
            # Without perf_kernels there is nowhere to put the kernels; without
            # run_type the run row cannot be marked as a kernel run. Either way the
            # result would be a benchmark_runs row with nothing behind it and no
            # marker — the "run that measured nothing" this branch exists to stop.
            # Checked before anything is written, so the skip stays retryable: no
            # source_file is recorded, and a later run re-ingests the file.
            missing = []
            if not _table_exists(client, "perf_kernels"):
                missing.append("no perf_kernels table")
            if _absent_columns(client, "benchmark_runs", ("run_type",)):
                missing.append("no benchmark_runs.run_type column")
            if missing:
                print(
                    f"  [warn] {' and '.join(missing)} — skipping this kernel XML. "
                    "Apply the spyre-dashboard migration to capture it.",
                    file=sys.stderr,
                )
                continue
            run_meta, kernels = parse_kernel_xml(
                xml_path, args.workflow, args.run_id, args.platform
            )
            if run_meta is None:
                continue

            # v1-table read, so it only applies when v1 is being written. The v2 path has its
            # own dedup (v2_benchmarks_already_ingested) against its own table.
            if args.write_v1:
                existing = client.query(
                    "SELECT count() FROM benchmark_runs WHERE source_file = {sf:String}",
                    parameters={"sf": run_meta["source_file"]},
                )
                if existing.result_rows[0][0] > 0:
                    print(
                        f"  Already ingested kernels — skipping {run_meta['source_file']}"
                    )
                    continue

            run_id = uuid.uuid4().int >> 64
            print(f"  run_id={run_id}  kernels={len(kernels)}")

            if args.write_v1:
                insert_benchmark_run(client, run_id, run_meta)
                insert_perf_kernels(client, run_id, kernels)

            # Additive v2 write: the same measurements under a DERIVED run_id, so a
            # perf number can name the artifact it measured. Guarded on both tables
            # existing so this deploys before the migration.
            if v2client is not None and v2_benchmark_tables_present(v2client):
                _src, _ext = v2_source_and_external_run_id(args, str(run_id))
                _v2_run_id = v2_run_id(_src, _ext, args.platform or "", "perf")
                if not _v2_run_id:
                    print(
                        "  [warn] v2 skipped: run_id not derivable "
                        f"(source={_src!r} external_run_id={_ext!r})",
                        file=sys.stderr,
                    )
                elif v2_benchmarks_already_ingested(v2client, _v2_run_id):
                    print(f"  v2: already ingested run_id={_v2_run_id} — skipping")
                else:
                    _n = insert_benchmarks_v2(v2client, _v2_run_id, kernels)
                    print(f"  v2: {_n} benchmark_runs under run_id={_v2_run_id}")

            total_kernels += len(kernels)
            print(f"  Inserted {len(kernels)} kernel rows")

        elif is_benchmark_xml(root):
            print("  Detected: performance benchmark XML")
            run_meta, benchmarks = parse_benchmark_xml(
                xml_path, args.workflow, args.run_id, args.platform
            )
            if run_meta is None:
                continue

            # A perf run uploads report.xml alongside the spyre/cpu kernel-report
            # XMLs. Those kernel reports are benchmark XMLs (classname carries
            # "benchmark") but their testcase names do not match _PERF_NAME_RE, so
            # they parse to zero rows. Inserting a run header for them creates an
            # empty benchmark_runs entry that shows as a "run" with 0 ops/models on
            # the dashboard. Skip the header when there is nothing to record; the
            # kernel timings are already folded into report.xml's kernel_mean_ms.
            if not benchmarks:
                print(f"  No benchmark records in {xml_path.name} — skipping header")
                continue

            # Deduplication: skip if source_file already in benchmark_runs
            # Same as the kernel path above: a v1-table read, gated on v1 being written.
            if args.write_v1:
                existing = client.query(
                    "SELECT count() FROM benchmark_runs WHERE source_file = {sf:String}",
                    parameters={"sf": run_meta["source_file"]},
                )
                if existing.result_rows[0][0] > 0:
                    print(
                        f"  Already ingested benchmark — skipping {run_meta['source_file']}"
                    )
                    continue

            # benchmark_runs.run_id is UInt64 — use a random 64-bit int
            run_id = uuid.uuid4().int >> 64  # positive 64-bit int
            print(f"  run_id={run_id}  benchmarks={len(benchmarks)}")

            if args.write_v1:
                insert_benchmark_run(client, run_id, run_meta)
                insert_perf_benchmarks(client, run_id, benchmarks)

            # Additive v2 write: the same measurements under a DERIVED run_id, so a
            # perf number can name the artifact it measured. Guarded on both tables
            # existing so this deploys before the migration.
            if v2client is not None and v2_benchmark_tables_present(v2client):
                _src, _ext = v2_source_and_external_run_id(args, str(run_id))
                _v2_run_id = v2_run_id(_src, _ext, args.platform or "", "perf")
                if not _v2_run_id:
                    print(
                        "  [warn] v2 skipped: run_id not derivable "
                        f"(source={_src!r} external_run_id={_ext!r})",
                        file=sys.stderr,
                    )
                elif v2_benchmarks_already_ingested(v2client, _v2_run_id):
                    print(f"  v2: already ingested run_id={_v2_run_id} — skipping")
                else:
                    _n = insert_benchmarks_v2(v2client, _v2_run_id, benchmarks)
                    print(f"  v2: {_n} benchmark_runs under run_id={_v2_run_id}")

            total_benchmarks += len(benchmarks)
            print(f"  Inserted {len(benchmarks)} benchmark rows")

        else:
            print("  Detected: test-result XML")
            run, cases = parse_test_xml(xml_path)
            if run is None:
                continue

            # One run_id per TEST RUN, not per XML file: the dispatching orchestrator
            # generates a uuid and threads it down as --run-id, and stamps the SAME value on
            # artifact_results, so the two tables join. `filename` stays the per-file
            # discriminator among the rows that share it.
            # Falls back to a fresh uuid4 when --run-id is absent or not a uuid (a standalone
            # or GHA-only run): the rows are still valid, just not linked to an artifact.
            # Resolved BEFORE dedup, which keys on it.
            run_id = _threaded_run_id(args) or str(uuid.uuid4())

            # Dedup on (run_id, filename): re-ingesting the SAME test run must be idempotent,
            # but two distinct runs must never collapse. runner_run_id mirrors run_id for a Jenkins/standalone leg, so it's only an independent signal for a GHA numeric id.
            runner_run_id = _runner_run_id(args, run_id)
            # v1-table reads, so gated on v1 being written. v2 dedups on its own table via
            # v2_already_ingested(run_id, component).
            if args.write_v1:
                existing = client.query(
                    "SELECT count() FROM test_runs "
                    "WHERE run_id = {run_id:String} AND filename = {filename:String}",
                    parameters={"run_id": run_id, "filename": run["filename"]},
                )
                if (
                    existing.result_rows[0][0] == 0
                    and runner_run_id
                    and runner_run_id != run_id
                ):
                    # A GHA re-ingest mints a fresh uuid4, so fall back to the numeric run id
                    # to keep that path idempotent.
                    existing = client.query(
                        "SELECT count() FROM test_runs WHERE "
                        "runner_run_id = {runner_run_id:String} AND filename = {filename:String}",
                        parameters={
                            "runner_run_id": runner_run_id,
                            "filename": run["filename"],
                        },
                    )
                if existing.result_rows[0][0] > 0:
                    print(f"  Already ingested — skipping {run['filename']}")
                    continue
            # `errors` is printed separately from `failed` even though it is a SUBSET of
            # it: a run whose outcomes are pytest errors could not start (bad import,
            # unloadable model), which is a different triage path from N regressions.
            # Observed reading as "failed=581" for 581 errors.
            print(
                f"  run_id={run_id}  tests={run['total_tests']}  "
                f"passed={run['passed']}  failed={run['failed']}"
                + (f" (of which errors={run['errors']})" if run["errors"] else "")
                + f"  xpass={run['xpass']}  xfail={run['xfail']}  skipped={run['skipped']}"
            )

            if args.write_v1:
                insert_run(client, run_id, run, args)

                # The (run_id, filename) dedup above already covers this file; a run_id-only recheck here would skip a second file sharing the same run_id.
                insert_cases(client, run_id, cases, workflow=args.workflow)
                insert_properties(client, run_id, cases)

            # v2 tables, alongside v1. Failure here must never cost a v1 row: v1 is still
            # authoritative, so the experimental write is contained rather than allowed to
            # abort the loop and drop every remaining file's v1 insert.
            try:
                if v2client is not None and v2_tables_present(v2client):
                    _v2_source, _v2_ext = v2_source_and_external_run_id(args, run_id)
                    _v2_tier = (getattr(args, "trigger_type", "") or "").strip()
                    _v2_run_id = v2_run_id(
                        _v2_source, _v2_ext, args.platform or run["platform"], _v2_tier
                    )
                    if not _v2_run_id:
                        # Loud, because a blank run_id means these cases reach v2 unjoinable
                        # to any artifact -- and that reads downstream as "no tests ran".
                        print(
                            f"  [warn] v2 skipped: run_id not derivable "
                            f"(source={_v2_source} ext={_v2_ext!r} "
                            f"arch={args.platform or run['platform']!r} tier={_v2_tier!r}); "
                            f"--trigger-type is the field usually missing",
                            file=sys.stderr,
                        )
                    elif v2_already_ingested(v2client, _v2_run_id, V2_COMPONENT):
                        print(f"  v2: already ingested run_id={_v2_run_id} — skipping")
                    else:
                        _n = insert_v2(v2client, V2_COMPONENT, _v2_run_id, cases)
                        print(f"  v2: {_n} test_case_runs under run_id={_v2_run_id}")
            except Exception as _v2_err:
                print(
                    f"  [warn] v2 write failed, v1 unaffected: {_v2_err!r}",
                    file=sys.stderr,
                )

            total_cases += len(cases)
            if args.write_v1:
                print(
                    f"  Inserted {len(cases)} test cases + "
                    f"{sum(len(c['properties']) for c in cases)} properties"
                )

    print(f"\nDone. {len(xml_files)} file(s) processed.")
    print(f"  Test cases ingested:  {total_cases}")
    print(f"  Benchmarks ingested:  {total_benchmarks}")
    print(f"  Kernels ingested:     {total_kernels}")


if __name__ == "__main__":
    main()
