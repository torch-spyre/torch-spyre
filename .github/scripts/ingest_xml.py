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
import regex as re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree
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

# Pattern:  perf_{op_name}_{metric}_ms_{input_shapes}
_PERF_NAME_RE = re.compile(
    r"^perf_(?P<op>.+?)_(?P<metric>wall_clock|cpu|spyre|kernel|memory_transfer)_ms(?:_(?P<shapes>.+))?$"
)

_GRANITE_CONFIG_RE = re.compile(r"bs(?P<batch_size>\d+)(?:_pl(?P<prompt_length>\d+))?")


def is_benchmark_xml(root) -> bool:
    """Return True if every testcase has classname containing 'benchmark'."""
    cases = root.findall(".//testcase")
    if not cases:
        return False
    return all("benchmark" in (tc.get("classname", "")) for tc in cases)


def parse_benchmark_xml(xml_path: Path):
    """
    Parse a performance-benchmark XML into (run_meta, list[benchmark_row]).

    Groups the 5 per-op-shape metric cases into one perf_benchmarks row each,
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
                "pt_util_percent": pt_util,
                "num_runs": num_runs_int,
                "custom_op_file": None,
                "regression_status": regression_status,
                "created_at": created_at,
                "torch_spyre_ms": torch_spyre_ms,
                "ratio": ratio,
            }
        )

    run_meta = {
        "source_file": xml_path.name,
        "created_at": created_at,
        "version_info": version_info,
    }
    return run_meta, benchmarks


# ---------------------------------------------------------------------------
# ── BENCHMARK ClickHouse insertion ─────────────────────────────────────────
# ---------------------------------------------------------------------------


def insert_benchmark_run(client, run_id: int, run_meta: dict) -> None:
    client.insert(
        "benchmark_runs",
        [
            [
                run_id,
                run_meta["source_file"],
                run_meta.get("version_info"),
                run_meta["created_at"].replace(tzinfo=None),
            ]
        ],
        column_names=["run_id", "source_file", "version_info", "created_at"],
    )


def insert_perf_benchmarks(client, run_id: int, benchmarks: list[dict]) -> None:
    if not benchmarks:
        return
    client.insert(
        "perf_benchmarks",
        [
            [
                b["benchmark_id"],
                run_id,
                b["record_type"],
                b["operation_name"],
                b["config_name"],
                b["input_shapes"],
                b["batch_size"],
                b["prompt_length"],
                b["run_mode"],
                b["total_duration_ms"],
                b["cpu_ms"],
                b["spyre_ms"],
                b["kernel_mean_ms"],
                b["memory_transfer_mean_ms"],
                b["pt_util_percent"],
                b["num_runs"],
                b["custom_op_file"],
                b["regression_status"],
                b["created_at"].replace(tzinfo=None),
            ]
            for b in benchmarks
        ],
        column_names=[
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
            "pt_util_percent",
            "num_runs",
            "custom_op_file",
            "regression_status",
            "created_at",
        ],
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
            "trigger_type",
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

    total_cases = 0
    total_benchmarks = 0

    for xml_path in xml_files:
        print(f"Processing: {xml_path.name}")

        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        # ── Dispatch: benchmark vs test-result ─────────────────────────────
        if is_benchmark_xml(root):
            print("  Detected: performance benchmark XML")
            run_meta, benchmarks = parse_benchmark_xml(xml_path)
            if run_meta is None:
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


if __name__ == "__main__":
    main()
