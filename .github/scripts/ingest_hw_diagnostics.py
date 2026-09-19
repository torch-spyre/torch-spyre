#!/usr/bin/env python3
"""
Reads the JSON produced by parse_hw_failures.py and batch-inserts the rows
into ClickHouse (hw_failure_diagnostics table).

Usage (called by the GHA workflow):
    python3 ingest_hw_diagnostics.py \
        --json-file hw_diagnostics_tests_74526099734.json \
        --workflow  "tests" \
        --branch    "main" \
        --sha       "abc123..." \
        --run-id    "74526099734" \
        --run-link  "https://github.com/org/repo/actions/runs/74526099734"

The parse/ingest logic lives in spyre_clickhouse_ingest (extensions/clickhouse-ingest) so the
product repos share one definition; this file is the CLI around it.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from spyre_clickhouse_ingest.client import client_summary, get_client
from spyre_clickhouse_ingest.hw_diagnostics import (
    RunContext,
    _str,
    build_row,
    filter_suite_records,
    insert_rows,
    load_records,
)
from spyre_clickhouse_ingest.hw_schema import (
    DEFAULT_TABLE,
    already_ingested,
    ensure_extra_columns,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest hw_diagnostics JSON → ClickHouse hw_failure_diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to JSON file produced by parse_hw_failures.py",
    )
    parser.add_argument(
        "--workflow", default="", help="Originating GHA workflow name (e.g. 'tests')"
    )
    parser.add_argument("--branch", default="", help="Git branch name (e.g. 'main')")
    parser.add_argument("--sha", default="", help="Git commit SHA")
    parser.add_argument(
        "--run-id",
        default="",
        help="GHA run ID — used as run_id if JSON records lack one",
    )
    parser.add_argument(
        "--run-link",
        default="",
        help="URL to the triggering GHA run (e.g. '<server>/<repo>/actions/runs/<id>')",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Target ClickHouse table (default: {DEFAULT_TABLE})",
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[error] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(json_path)
    if not records:
        print("[info] JSON file contains no records — nothing to ingest.")
        sys.exit(0)

    # Both emptiness checks are needed, and both belong BEFORE the connect: the filter can empty a
    # non-empty list, and the dedup step below indexes records[0]. Checking only the file left an
    # IndexError reachable after a ClickHouse session was already open.
    records = filter_suite_records(records)
    if not records:
        print("[info] No suite records after filtering — nothing to ingest.")
        sys.exit(0)

    print(f"[info] Loaded {len(records)} record(s) from {json_path.name}")

    print(f"[info] Connecting to ClickHouse at {client_summary()} ...")
    client = get_client()
    client.command("SELECT 1")
    print("[info] Connected.\n")

    ensure_extra_columns(client, table=args.table)

    # One JSON file is one run, so the first record's run_id represents the batch.
    run_id = _str(records[0].get("run_id") or args.run_id)
    workflow = _str(args.workflow)

    if already_ingested(client, run_id, workflow, table=args.table):
        print(
            f"[info] run_id={run_id!r} workflow={workflow!r} already ingested "
            f"— skipping. Re-run with a different table or clear the existing rows."
        )
        sys.exit(0)

    ctx = RunContext(
        run_id=args.run_id,
        workflow=args.workflow,
        branch=args.branch,
        sha=args.sha,
        run_link=args.run_link,
    )

    rows = []
    skipped = 0
    for rec in records:
        try:
            rows.append(build_row(rec, ctx))
        except Exception as exc:
            skipped += 1
            print(
                f"  [warn] Skipping record suite={rec.get('suite_name')!r} "
                f"attempt={rec.get('attempt')}: {exc}",
                file=sys.stderr,
            )

    if skipped:
        print(f"[warn] {skipped} record(s) skipped due to errors", file=sys.stderr)

    if not rows:
        print("[error] No valid rows to insert.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Inserting {len(rows)} row(s) into {args.table} ...")
    try:
        insert_rows(client, rows, table=args.table)
    except Exception as exc:
        print(f"[error] Insert failed: {exc}", file=sys.stderr)
        sys.exit(1)

    reasons: Counter = Counter(_str(r.get("failure_reason"), "none") for r in records)
    outcomes: Counter = Counter(_str(r.get("outcome"), "unknown") for r in records)

    print(f"\n[info] Successfully inserted {len(rows)} row(s) into {args.table}")
    print(f"[info]   run_id   : {run_id}")
    print(f"[info]   workflow : {workflow}")
    print(f"[info]   branch   : {args.branch}")
    print(f"[info]   sha      : {args.sha[:12]}")
    print()
    print("[info] Outcomes:")
    for outcome, n in sorted(outcomes.items()):
        print(f"[info]   {outcome:10}: {n}")
    print()
    print("[info] Failure reasons:")
    for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"[info]   {reason:35}: {n}")


if __name__ == "__main__":
    main()
