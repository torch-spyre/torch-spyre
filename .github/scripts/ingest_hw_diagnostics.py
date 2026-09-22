#!/usr/bin/env python3
"""
Reads the JSON produced by parse_hw_failures.py and batch-inserts the rows
into ClickHouse (hw_failure_diagnostics table).

Usage (called by the GHA workflow):
    python3 ingest_hw_diagnostics.py \
        --json-file    hw_diagnostics_tests_74526099734.json \
        --component    torch-spyre \
        --arch         x86_64 \
        --trigger-type regression \
        --gha-run-id   74526099734 \
        --run-url      "https://github.com/org/repo/actions/runs/74526099734"

The parse/ingest logic lives in spyre_clickhouse_ingest (extensions/clickhouse-ingest) so the
product repos share one definition; this file is the CLI around it.
"""

import argparse
import platform as _platform
import sys
from collections import Counter
from pathlib import Path

from spyre_clickhouse_ingest.client import client_summary, get_client
from spyre_clickhouse_ingest.hw_diagnostics import (
    NIL_UUID,
    RunContext,
    _str,
    build_row,
    filter_suite_records,
    insert_rows,
    load_records,
)
from spyre_clickhouse_ingest.identity import (
    canonical_arch,
    component_of,
    run_id_for,
)
from spyre_clickhouse_ingest.hw_schema import DEFAULT_TABLE, already_ingested


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
        "--table",
        default=DEFAULT_TABLE,
        help=f"Target ClickHouse table (default: {DEFAULT_TABLE})",
    )
    # run_id is DERIVED, never passed as a coordinate: --run-id carries the threaded uuid when
    # the orchestrator minted one, and the flags below supply the hash inputs when it did not.
    parser.add_argument(
        "--run-id",
        default="",
        help="Threaded run uuid from the orchestrator, when one was minted",
    )
    parser.add_argument(
        "--component",
        default="",
        help="Component whose suite this is (default: torch-spyre)",
    )
    parser.add_argument(
        "--arch",
        default=_platform.machine(),
        help="Arch of this leg; amd64 folds to x86_64 (default: this machine's)",
    )
    parser.add_argument(
        "--gha-run-id", default="", help="GHA run id, when GHA dispatched this leg"
    )
    parser.add_argument(
        "--jenkins-run-key",
        default="",
        help="Jenkins externalizable id ('folder/job#123'), when Jenkins dispatched this leg",
    )
    parser.add_argument(
        "--trigger-type",
        default="",
        help="Test tier of this leg (regression | trunk | perf | ...). Replaces --workflow, "
        "which carried the same value: it is the test_type run_id is hashed from.",
    )
    parser.add_argument(
        "--run-url",
        default="",
        help="URL of the CI run behind these rows; kept in props, not a column",
    )
    parser.add_argument(
        "--artifact-id",
        default="",
        help="artifact_id of the image this leg ran, read from its OCI label / in-image file",
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

    # One JSON file is one run, so the first record's coordinate represents the batch. It is a
    # hash INPUT, not the run_id: --gha-run-id is the flag form of the same value.
    external_run_id = _str(records[0].get("run_id") or args.gha_run_id)
    arch = canonical_arch(args.arch)
    component = component_of(args)
    # Same two-case rule as every other writer: the threaded uuid when one was supplied, else
    # the hash of this leg's own CI coordinate.
    run_id = run_id_for(args, external_run_id, arch, args.trigger_type)
    if not run_id:
        # Coercing this to a fixed sentinel (NIL_UUID) would make every underivable run
        # collide with every other one in the dedup check below, silently dropping runs
        # that never touched each other. Fail loudly instead.
        print(
            f"[error] run_id not derivable (external_run_id={external_run_id!r} "
            f"arch={arch!r} trigger_type={args.trigger_type!r}); "
            "--trigger-type is the flag usually missing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if already_ingested(client, run_id, component, table=args.table):
        print(
            f"[info] run_id={run_id} component={component!r} already ingested "
            f"— skipping. Re-run with a different table or clear the existing rows."
        )
        sys.exit(0)

    ctx = RunContext(
        run_id=run_id,
        artifact_id=_str(args.artifact_id) or NIL_UUID,
        component=component,
        arch=arch,
        external_run_id=external_run_id,
        run_url=args.run_url,
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
    print(f"[info]   run_id     : {ctx.run_id}")
    print(f"[info]   external   : {ctx.external_run_id} (tier {args.trigger_type!r})")
    print(f"[info]   component  : {ctx.component}  arch: {ctx.arch}")
    print(f"[info]   artifact_id: {ctx.artifact_id}")
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
