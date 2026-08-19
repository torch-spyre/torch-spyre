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
import regex as re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import xml.etree.ElementTree as etree
import clickhouse_connect

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
        created_at = datetime.now(timezone.utc)

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
        created_at = datetime.now(timezone.utc)

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
        triggered_at = datetime.now(timezone.utc)

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


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


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
                int(args.run_id or 0),
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
            "gha_run_id",
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
# ── Main ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=None)
    parser.add_argument("--xml-file", default=None)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--triggered-at", default="")
    parser.add_argument("--pr-number", default="")
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
    args = parser.parse_args()

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
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("Connected.\n")

    # CREATE TABLE IF NOT EXISTS elsewhere won't add a column to an existing table
    client.command(
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS workflow String DEFAULT ''"
    )
    client.command(
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS platform String DEFAULT ''"
    )

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

            insert_benchmark_run(client, run_id, run_meta)
            insert_perf_kernels(client, run_id, kernels)

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

            insert_benchmark_run(client, run_id, run_meta)
            insert_perf_benchmarks(client, run_id, benchmarks)

            total_benchmarks += len(benchmarks)
            print(f"  Inserted {len(benchmarks)} benchmark rows")

        else:
            print("  Detected: test-result XML")
            run, cases = parse_test_xml(xml_path)
            if run is None:
                continue

            # Deduplication
            existing = client.query(
                "SELECT count() FROM test_runs "
                "WHERE gha_run_id = {gha_run_id:UInt64} AND filename = {filename:String}",
                parameters={
                    "gha_run_id": int(args.run_id or 0),
                    "filename": run["filename"],
                },
            )
            if existing.result_rows[0][0] > 0:
                print(f"  Already ingested — skipping {run['filename']}")
                continue

            run_id = str(uuid.uuid4())
            print(
                f"  run_id={run_id}  tests={run['total_tests']}  "
                f"passed={run['passed']}  failed={run['failed']}  "
                f"xpass={run['xpass']}  xfail={run['xfail']}  skipped={run['skipped']}"
            )

            insert_run(client, run_id, run, args)

            existing_cases = client.query(
                "SELECT count() FROM test_cases tc "
                "INNER JOIN test_runs tr ON tc.run_id = tr.run_id "
                "WHERE tr.gha_run_id = {gha_run_id:UInt64} AND tr.filename = {filename:String}",
                parameters={
                    "gha_run_id": int(args.run_id or 0),
                    "filename": run["filename"],
                },
            )
            if existing_cases.result_rows[0][0] > 0:
                print("  Cases already exist — skipping case+property inserts")
            else:
                insert_cases(client, run_id, cases, workflow=args.workflow)
                existing_props = client.query(
                    "SELECT count() FROM run_properties WHERE run_id = {run_id:String}",
                    parameters={"run_id": run_id},
                )
                if existing_props.result_rows[0][0] > 0:
                    print("  Properties already exist — skipping property insert")
                else:
                    insert_properties(client, run_id, cases)

            total_cases += len(cases)
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
