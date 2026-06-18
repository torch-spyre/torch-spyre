#!/usr/bin/env python3
"""
========================
Reads the JSON produced by parse_hardware_failures.py and batch-inserts
the rows into ClickHouse (hw_failure_diagnostics table).

Usage (called by the GHA workflow):
    python3 ingest_hw_diagnostics.py \
        --json-file hw_diagnostics_tests_74526099734.json \
        --workflow  "tests" \
        --branch    "main" \
        --sha       "abc123..." \
        --run-id    "74526099734"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import clickhouse_connect

# ---------------------------------------------------------------------------
# ClickHouse connection client
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


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _parse_ts(ts_str: str) -> datetime | None:
    """ISO-8601 string → naive UTC datetime, or None."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)  # ClickHouse DateTime64 wants naive
    except (ValueError, AttributeError):
        return None


def _str(val, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _detail_json(val) -> str:
    """Serialise failure_reason_detail dict → JSON string for ClickHouse."""
    if not val:
        return "{}"
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_row(rec: dict, args) -> list:
    """
    Map one JSON record → ordered list matching hw_failure_diagnostics columns.
    Column order must match COLUMN_NAMES below exactly.
    """
    return [
        # ── Identity ──────────────────────────────────────────────────────
        _str(rec.get("run_id") or args.run_id),
        _str(args.workflow),
        _str(args.branch),
        _str(args.sha)[:40].ljust(40)[:40],  # normalise to ≤40 chars
        _str(rec.get("suite_name")),
        _int(rec.get("attempt"), 1),
        _int(rec.get("total_attempts"), 1),
        _parse_ts(rec.get("ingested_at"))
        or datetime.now(timezone.utc).replace(tzinfo=None),
        # ── Outcome ───────────────────────────────────────────────────────
        _str(rec.get("outcome"), "unknown"),
        rec.get("exit_code"),  # Nullable(Int32) — keep None
        # ── Failure classification ────────────────────────────────────────
        _str(rec.get("failure_reason"), "none"),
        _str(rec.get("failure_phase")),
        _str(rec.get("retry_trigger")),
        _detail_json(rec.get("failure_reason_detail")),
        # ── Primary RAS event ─────────────────────────────────────────────
        _str(rec.get("ras_code")),
        _str(rec.get("ras_name")),
        _str(rec.get("ras_description")),
        _str(rec.get("ras_action")),
        _str(rec.get("ras_category")),
        _str(rec.get("ras_severity")),
        _str(rec.get("ras_message")),
        _str(rec.get("ras_events_json"), "[]"),
        # ── Hardware identifiers ──────────────────────────────────────────
        _str(rec.get("node_name")),
        _str(rec.get("pci_device")),
        _str(rec.get("aiu_world_rank0")),
        _str(rec.get("card_serial")),
        _str(rec.get("chip_ecid_raw")),
        _str(rec.get("chip_wafer_id")),
        _str(rec.get("chip_mfg_x")),
        _str(rec.get("chip_mfg_y")),
        _str(rec.get("chip_chipy")),
        _str(rec.get("chip_chipx")),
        # ── Timestamps ────────────────────────────────────────────────────
        _parse_ts(rec.get("first_error_ts")),  # Nullable(DateTime64)
        _parse_ts(rec.get("attempt_start_ts")),
        # ── Pytest statistics ─────────────────────────────────────────────
        _int(rec.get("tests_collected")),
        _int(rec.get("tests_passed")),
        _int(rec.get("tests_failed")),
        _int(rec.get("tests_error")),
        # ── Stall info ────────────────────────────────────────────────────
        _int(rec.get("stall_max_secs")),
    ]


# Column names — must match build_row() order and hw_failure_diagnostics DDL
COLUMN_NAMES = [
    # Identity
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "attempt",
    "total_attempts",
    "ingested_at",
    # Outcome
    "outcome",
    "exit_code",
    # Failure classification
    "failure_reason",
    "failure_phase",
    "retry_trigger",
    "failure_reason_detail",
    # RAS
    "ras_code",
    "ras_name",
    "ras_description",
    "ras_action",
    "ras_category",
    "ras_severity",
    "ras_message",
    "ras_events_json",
    # Hardware
    "node_name",
    "pci_device",
    "aiu_world_rank0",
    "card_serial",
    "chip_ecid_raw",
    "chip_wafer_id",
    "chip_mfg_x",
    "chip_mfg_y",
    "chip_chipy",
    "chip_chipx",
    # Timestamps
    "first_error_ts",
    "attempt_start_ts",
    # Pytest stats
    "tests_collected",
    "tests_passed",
    "tests_failed",
    "tests_error",
    # Stall
    "stall_max_secs",
]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def already_ingested(client, run_id: str, workflow: str) -> bool:
    """
    Return True if this (run_id, workflow) pair already has rows in the table.
    Prevents double-ingestion if the GHA job is re-run.
    """
    result = client.query(
        "SELECT count() FROM hw_failure_diagnostics "
        "WHERE run_id = {run_id:String} AND workflow = {workflow:String}",
        parameters={"run_id": run_id, "workflow": workflow},
    )
    return result.result_rows[0][0] > 0


# ---------------------------------------------------------------------------
# Schema update helper
# ---------------------------------------------------------------------------


def ensure_extra_columns(client) -> None:
    """
    Add columns that may not exist in older deployments of the schema.
    ALTER TABLE ADD COLUMN IF NOT EXISTS is idempotent in ClickHouse.
    """
    extras = [
        ("workflow", "LowCardinality(String) DEFAULT ''"),
        ("branch", "LowCardinality(String) DEFAULT ''"),
        ("commit_sha", "String DEFAULT ''"),
        ("failure_reason_detail", "String DEFAULT '{}'"),
        ("ras_category", "LowCardinality(String) DEFAULT ''"),
        ("ras_severity", "LowCardinality(String) DEFAULT ''"),
        ("ras_message", "String DEFAULT ''"),
        ("ras_events_json", "String DEFAULT '[]'"),
    ]
    for col_name, col_type in extras:
        try:
            client.command(
                f"ALTER TABLE hw_failure_diagnostics "
                f"ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
            )
        except Exception as exc:
            # Non-fatal: log and continue (column may already exist with right type)
            print(f"  [warn] Could not add column {col_name}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest hw_diagnostics JSON → ClickHouse hw_failure_diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to JSON file produced by parse_hardware_failures.py",
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
        "--table",
        default="hw_failure_diagnostics",
        help="Target ClickHouse table (default: hw_failure_diagnostics)",
    )
    args = parser.parse_args()

    # ── Load JSON ──────────────────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[error] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as fh:
        records = json.load(fh)

    if not records:
        print("[info] JSON file contains no records — nothing to ingest.")
        sys.exit(0)

    # Filter out any .DS_Store / meta records that slipped through
    records = [
        r
        for r in records
        if r.get("suite_name", "").strip() and not r["suite_name"].startswith(".")
    ]

    print(f"[info] Loaded {len(records)} record(s) from {json_path.name}")

    # ── Connect ────────────────────────────────────────────────────────────
    print(
        f"[info] Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("[info] Connected.\n")

    # ── Ensure schema is up to date ────────────────────────────────────────
    ensure_extra_columns(client)

    # ── Deduplication check ────────────────────────────────────────────────
    # Use the run_id from the first record (all records in one JSON share a run)
    run_id = _str(records[0].get("run_id") or args.run_id)
    workflow = _str(args.workflow)

    if already_ingested(client, run_id, workflow):
        print(
            f"[info] run_id={run_id!r} workflow={workflow!r} already ingested "
            f"— skipping. Re-run with a different table or clear the existing rows."
        )
        sys.exit(0)

    # ── Build rows ─────────────────────────────────────────────────────────
    rows = []
    skipped = 0
    for rec in records:
        try:
            rows.append(build_row(rec, args))
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

    # ── Insert ─────────────────────────────────────────────────────────────
    print(f"[info] Inserting {len(rows)} row(s) into {args.table} ...")
    try:
        client.insert(
            table=args.table,
            data=rows,
            column_names=COLUMN_NAMES,
        )
    except Exception as exc:
        print(f"[error] Insert failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────
    from collections import Counter

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
