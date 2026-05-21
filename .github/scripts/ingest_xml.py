#!/usr/bin/env python3
"""
Parses pytest JUnit XML files produced by the Spyre CI pipelines and
batch-inserts the results into ClickHouse.

Usage (called by the GHA workflow):
    python3 ingest_xml.py \
        --xml-dir xml_artifacts \
        --workflow "model-module-tests" \
        --branch   "main" \
        --sha      "abcdef1234..." \
        --run-id   "12345678" \
        --triggered-at "2026-04-25T14:20:45Z"
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from lxml import etree
import clickhouse_driver

# ---------------------------------------------------------------------------
# Status classification — uses the logic in dashboard.html parseXML()
# ---------------------------------------------------------------------------


def classify_testcase(tc_el):
    """
    Return (status, fail_message) for a single <testcase> element.

    pytest JUnit XML conventions:
      bare testcase (no children)                → passed
      <failure type="pytest.xfail">              → xpass (strict mode)
      <failure ...>                               → failed
      <error ...>                                 → error
      <skipped type="pytest.xfail">              → xfail
      <skipped type="pytest.skip" | "pytest.mark.skip"> → skipped
    """
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
    """
    Return list of (prop_name, prop_value) from <properties><property .../></properties>.

    Matches how dashboard.html reads them:
      - <property name="tag" value="model__granite3b"/> → stored as-is
      - <property name="granite" value="True"/>          → stored as ("granite", "True")
    """
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


def extract_op_dtype(name: str, properties: list[tuple[str, str]]):
    """
    Extract op_name and dtype from test case metadata.

    Priority: explicit op__/dtype__ properties → fall back to name parsing.
    """
    op_name = ""
    dtype = ""

    for pname, pvalue in properties:
        if pname.startswith("op__"):
            op_name = pname[4:]
        elif pname.startswith("dtype__"):
            dtype = pname[7:]
        elif pname == "tag":
            if pvalue.startswith("op__"):
                op_name = pvalue[4:]
            elif pvalue.startswith("dtype__"):
                dtype = pvalue[7:]

    # Name-based fallback for dtype
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

    return op_name, dtype


# ---------------------------------------------------------------------------
# Suite-level xpass promotion (uses dashboard.html second pass)
# ---------------------------------------------------------------------------


def promote_xpass(raw_cases, suite_attrs):
    """
    Promote bare 'passed' cases to 'xpass' when the suite attributes indicate
    the run had non-strict xpass entries counted in the failures counter.
    Mutates raw_cases in-place.
    """
    # total = int(suite_attrs.get("tests", 0))
    # skipped = int(suite_attrs.get("skipped", 0))
    failures = int(suite_attrs.get("failures", 0))
    # errors = int(suite_attrs.get("errors", 0))

    true_fail_raw = sum(1 for c in raw_cases if c["status"] in ("failed", "error"))
    strict_xpass_raw = sum(1 for c in raw_cases if c["status"] == "xpass")

    non_strict_in_failures = max(0, failures - true_fail_raw - strict_xpass_raw)

    promoted = 0
    for c in raw_cases:
        if promoted >= non_strict_in_failures:
            break
        if c["_is_bare"]:
            c["status"] = "xpass"
            promoted += 1


# -----------------------------
# Parse a single XML file
# -----------------------------


def parse_xml(xml_path: Path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # Support both <testsuites><testsuite> and bare <testsuite>
    suites = root.findall(".//testsuite")
    if not suites:
        print(f"  [warn] No <testsuite> found in {xml_path.name}", file=sys.stderr)
        return None, []

    suite = suites[0]
    suite_attrs = suite.attrib

    # Parse timestamp; fall back to now
    ts_str = suite_attrs.get("timestamp", "")
    try:
        triggered_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        triggered_at = datetime.now(timezone.utc)

    raw_cases = []
    for tc in suite.findall(".//testcase"):
        status, fail_msg = classify_testcase(tc)
        properties = extract_properties(tc)
        op_name, dtype = extract_op_dtype(tc.get("name", ""), properties)
        raw_cases.append(
            {
                "case_id": str(uuid.uuid4()),
                "classname": tc.get("classname", ""),
                "name": tc.get("name", ""),
                "op_name": op_name,
                "dtype": dtype,
                "status": status,
                "duration_s": float(tc.get("time", 0) or 0),
                "fail_message": fail_msg,
                "properties": properties,
                "_is_bare": (status == "passed"),  # may be promoted to xpass
                "triggered_at": triggered_at,
            }
        )

    promote_xpass(raw_cases, suite_attrs)

    # Build run-level counts from final statuses
    from collections import Counter

    counts = Counter(c["status"] for c in raw_cases)

    run = {
        "suite_name": suite_attrs.get("name", xml_path.stem),
        "filename": xml_path.name,
        "triggered_at": triggered_at,
        "total_tests": len(raw_cases),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0) + counts.get("error", 0),
        # "skipped": counts.get("skipped", 0) + counts.get("xfail", 0),
        "skipped": counts.get("skipped", 0),
        "xfail": counts.get("xfail", 0),
        "errors": counts.get("error", 0),
        "xpass": counts.get("xpass", 0),
        "duration_s": float(suite_attrs.get("time", 0) or 0),
    }
    return run, raw_cases


# ---------------------------------------------------------------------------
# ClickHouse insertion
# ---------------------------------------------------------------------------


def get_client():
    return clickhouse_driver.Client(
        host=os.environ["CLICKHOUSE_HOST"],
        # port=int(os.environ.get("CLICKHOUSE_PORT", 9440)),
        port=9440,  # native TCP+TLS port, always 9440 on ClickHouse Cloud
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


def insert_run(client, run_id: str, run: dict, args):
    client.execute(
        """
        INSERT INTO test_runs
            (run_id, workflow, suite_name, filename, branch, commit_sha,
             gha_run_id, triggered_at, total_tests, passed, failed,
             skipped, xfail, errors, xpass, duration_s)
        VALUES
        """,
        [
            {
                "run_id": run_id,
                "workflow": args.workflow,
                "suite_name": run["suite_name"],
                "filename": run["filename"],
                "branch": args.branch,
                "commit_sha": (args.sha or "").ljust(40)[:40],
                "gha_run_id": int(args.run_id or 0),
                "triggered_at": run["triggered_at"].replace(tzinfo=None),
                "total_tests": run["total_tests"],
                "passed": run["passed"],
                "failed": run["failed"],
                "skipped": run["skipped"],
                "xfail": run["xfail"],
                "errors": run["errors"],
                "xpass": run["xpass"],
                "duration_s": run["duration_s"],
            }
        ],
    )


def insert_cases(client, run_id: str, cases: list[dict], workflow: str = ""):
    if not cases:
        return
    rows = [
        {
            "run_id": run_id,
            "case_id": c["case_id"],
            "classname": c["classname"],
            "name": c["name"],
            "op_name": c["op_name"],
            "dtype": c["dtype"],
            "status": c["status"],
            "duration_s": c["duration_s"],
            "fail_message": c["fail_message"][:8192],  # cap very long traces
            "triggered_at": c["triggered_at"].replace(tzinfo=None),
            "workflow": workflow,
        }
        for c in cases
    ]
    client.execute("INSERT INTO test_cases VALUES", rows)


def insert_properties(client, run_id: str, cases: list[dict]):
    rows = []
    for c in cases:
        for pname, pvalue in c["properties"]:
            rows.append(
                {
                    "run_id": run_id,
                    "case_id": c["case_id"],
                    "prop_name": pname,
                    "prop_value": pvalue,
                    "triggered_at": c["triggered_at"].replace(tzinfo=None),
                }
            )
    if rows:
        client.execute("INSERT INTO run_properties VALUES", rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--triggered-at", default="")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    xml_files = sorted(xml_dir.glob("*.xml"))

    if not xml_files:
        print("No XML files found — nothing to ingest.")
        sys.exit(0)

    print(
        f"Connecting to ClickHouse at {os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 9000)} ..."
    )
    client = get_client()
    client.execute("SELECT 1")  # connectivity check
    print("Connected.")

    total_cases = 0
    for xml_path in xml_files:
        print(f"\nProcessing: {xml_path.name}")
        run, cases = parse_xml(xml_path)
        if run is None:
            continue

        # Deduplication check — skip if this exact GHA run + filename
        # has already been ingested
        existing = client.execute(
            """
            SELECT count() FROM test_runs
            WHERE gha_run_id = %(gha_run_id)s
            AND   filename   = %(filename)s
            """,
            {
                "gha_run_id": int(args.run_id or 0),
                "filename": run["filename"],
            },
        )
        if existing[0][0] > 0:
            print(f"  Already ingested — skipping {run['filename']}")
            continue

        run_id = str(uuid.uuid4())
        print(
            f"  run_id={run_id}  tests={run['total_tests']}  "
            f"passed={run['passed']}  failed={run['failed']}  "
            f"xpass={run['xpass']}  xfail={run['xfail']}  skipped={run['skipped']}"
        )

        insert_run(client, run_id, run, args)

        existing_cases_after_run = client.execute(
            """
            SELECT count() FROM test_cases tc
            INNER JOIN test_runs tr ON tc.run_id = tr.run_id
            WHERE tr.gha_run_id = %(gha_run_id)s
            AND   tr.filename   = %(filename)s
            """,
            {
                "gha_run_id": int(args.run_id or 0),
                "filename": run["filename"],
            },
        )
        if existing_cases_after_run[0][0] > 0:
            print("  Cases already exist — skipping case+property inserts")
        else:
            insert_cases(client, run_id, cases, workflow=args.workflow)
            existing_props = client.execute(
                "SELECT count() FROM run_properties WHERE run_id = %(run_id)s",
                {"run_id": run_id},
            )
            if existing_props[0][0] > 0:
                print("  Properties already exist — skipping property insert")
            else:
                insert_properties(client, run_id, cases)

        # insert_cases(client, run_id, cases, workflow=args.workflow)
        # insert_properties(client, run_id, cases)
        total_cases += len(cases)
        print(
            f"  Inserted {len(cases)} test cases + "
            f"{sum(len(c['properties']) for c in cases)} properties"
        )

    print(f"\nDone. {len(xml_files)} file(s), {total_cases} total cases ingested.")


if __name__ == "__main__":
    main()
