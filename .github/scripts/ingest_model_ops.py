#!/usr/bin/env python3
"""
Reads the JSON produced by parse_model_ops_logs.py and batch-inserts the
rows into two ClickHouse tables:

  model_ops_suites  – one row per model suite per GHA run
                      (suite outcome + counts)

  model_ops_variants – one row per individual test variant
                       (operation, shapes, dtypes, XPASS/XFAIL/FALLBACK,
                        matching exactly the JSON schema shown in the dashboard)

Table lifecycle (every run):
  1. DROP TABLE IF EXISTS   model_ops_variants
  2. DROP TABLE IF EXISTS   model_ops_suites
  3. CREATE TABLE           model_ops_suites
  4. CREATE TABLE           model_ops_variants
  5. INSERT all suite rows
  6. INSERT all variant rows

This guarantees the dashboard always shows only the latest run's data —
no stale rows from previous runs are ever visible.

Usage (called by the GHA workflow):
    python3 ingest_model_ops.py \\
        --json-file model_ops_model-ops-tests_27674677047.json \\
        --workflow  "model-ops-tests" \\
        --branch    "main" \\
        --sha       "abc123..." \\
        --run-id    "27674677047"
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect

# ---------------------------------------------------------------------------
# ClickHouse DDL
# ---------------------------------------------------------------------------

_DROP_VARIANTS_SQL = "DROP TABLE IF EXISTS model_ops_variants"
_DROP_SUITES_SQL = "DROP TABLE IF EXISTS model_ops_suites"

_CREATE_SUITES_SQL = """
CREATE TABLE model_ops_suites
(
    -- ── Primary key ──────────────────────────────────────────────────────────
    -- SHA-256( gha_run_id || suite_name ) — one unique row per suite per run
    suite_id        FixedString(64),

    -- Identity / provenance
    gha_run_id      UInt64,
    run_id          String,
    workflow        LowCardinality(String) DEFAULT '',
    branch          LowCardinality(String) DEFAULT '',
    commit_sha      String DEFAULT '',

    -- Suite / config
    suite_name      String,
    model_name      LowCardinality(String) DEFAULT '',
    yaml_file       String DEFAULT '',

    -- Counts
    total_tests           UInt32 DEFAULT 0,
    spyre_enabled_count   UInt32 DEFAULT 0,
    not_implemented_count UInt32 DEFAULT 0,
    cpu_fallback_count    UInt32 DEFAULT 0,
    spyre_failed_count    UInt32 DEFAULT 0,

    -- Suite-level pytest stats
    suite_outcome   LowCardinality(String) DEFAULT 'unknown',
    suite_exit_code Nullable(Int32),
    tests_total     UInt32 DEFAULT 0,
    tests_passed    UInt32 DEFAULT 0,
    tests_failed    UInt32 DEFAULT 0,
    tests_skipped   UInt32 DEFAULT 0,
    tests_error     UInt32 DEFAULT 0,
    tests_xfail     UInt32 DEFAULT 0,
    tests_xpass     UInt32 DEFAULT 0,
    duration_s      Float32 DEFAULT 0,

    -- Timestamps
    triggered_at    DateTime64(3, 'UTC'),
    ingested_at     DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY suite_id          -- suite_id is the unique primary key
PARTITION BY toYYYYMM(triggered_at)
SETTINGS index_granularity = 8192
"""

_CREATE_VARIANTS_SQL = """
CREATE TABLE model_ops_variants
(
    -- ── Primary key ──────────────────────────────────────────────────────────
    -- SHA-256( gha_run_id || suite_name || operation || classification || test_name )
    -- Guarantees exactly one row per test variant per run, even when test_name=""
    variant_id      FixedString(64),

    -- Foreign key back to model_ops_suites.suite_id
    suite_id        FixedString(64),

    -- Provenance
    gha_run_id      UInt64,
    run_id          String,
    workflow        LowCardinality(String) DEFAULT '',
    branch          LowCardinality(String) DEFAULT '',
    commit_sha      String DEFAULT '',

    -- Suite / config
    suite_name      String,
    model_name      LowCardinality(String) DEFAULT '',
    yaml_file       String DEFAULT '',

    -- Variant identity
    operation       LowCardinality(String),
    classification  LowCardinality(String),   -- spyre_enabled | not_implemented | cpu_fallback
    test_name       String,
    status          LowCardinality(String),   -- XPASS | XFAIL | FALLBACK

    -- Tensor info (stored as JSON arrays serialised as strings)
    input_shapes    String DEFAULT '[]',      -- JSON array of shape strings
    input_strides   String DEFAULT '[]',
    input_dtypes    String DEFAULT '[]',

    -- Timestamps
    triggered_at    DateTime64(3, 'UTC'),
    ingested_at     DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY variant_id        -- variant_id is the unique primary key
PARTITION BY toYYYYMM(triggered_at)
SETTINGS index_granularity = 8192
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
        verify=False,
    )


def _parse_ts(ts_str: str) -> datetime:
    """ISO-8601 string → naive UTC datetime."""
    if not ts_str:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc).replace(tzinfo=None)


def _str(val, default: str = "") -> str:
    return str(val).strip() if val is not None else default


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _jstr(val) -> str:
    """Serialise a list → JSON string for ClickHouse String column."""
    if not val:
        return "[]"
    if isinstance(val, str):
        return val  # already serialised
    try:
        return json.dumps(val, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"


def _make_id(*parts: str) -> str:
    """Return a 64-character hex SHA-256 digest of the concatenated parts.

    Used to produce a single unique primary-key column for every row so
    ClickHouse's ReplacingMergeTree can deduplicate on a single column.

    Args:
        *parts: strings that together identify a row uniquely.
    """
    raw = "\x00".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()  # 64 hex chars


# ---------------------------------------------------------------------------
# Table lifecycle  — DROP → CREATE (called once per ingest run)
# ---------------------------------------------------------------------------


def recreate_tables(client) -> None:
    """
    Drop both tables (if they exist) then recreate them from scratch.
    This guarantees the dashboard always reflects only the latest run.
    """
    print("[info] Dropping existing tables (if any) ...")
    # Drop variants first — it has no dependants; suites may have dependants in views
    client.command(_DROP_VARIANTS_SQL)
    print("[info]   model_ops_variants  — dropped")
    client.command(_DROP_SUITES_SQL)
    print("[info]   model_ops_suites    — dropped")

    print("[info] Creating tables ...")
    client.command(_CREATE_SUITES_SQL)
    print("[info]   model_ops_suites    — created")
    client.command(_CREATE_VARIANTS_SQL)
    print("[info]   model_ops_variants  — created")
    print()


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

SUITE_COLS = [
    "suite_id",  # ← unique primary key (SHA-256)
    "gha_run_id",
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "model_name",
    "yaml_file",
    "total_tests",
    "spyre_enabled_count",
    "not_implemented_count",
    "cpu_fallback_count",
    "spyre_failed_count",
    "suite_outcome",
    "suite_exit_code",
    "tests_total",
    "tests_passed",
    "tests_failed",
    "tests_skipped",
    "tests_error",
    "tests_xfail",
    "tests_xpass",
    "duration_s",
    "triggered_at",
    "ingested_at",
]

VARIANT_COLS = [
    "variant_id",  # ← unique primary key (SHA-256)
    "suite_id",  # ← FK to model_ops_suites
    "gha_run_id",
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "suite_name",
    "model_name",
    "yaml_file",
    "operation",
    "classification",
    "test_name",
    "status",
    "input_shapes",
    "input_strides",
    "input_dtypes",
    "triggered_at",
    "ingested_at",
]


def build_suite_row(rec: dict, args, gha_run_id: int, now: datetime) -> list:
    summary = rec.get("summary", {})
    suite_name = _str(rec.get("suite_name"))
    suite_id = _make_id(gha_run_id, suite_name)  # unique PK
    return [
        suite_id,  # suite_id  (PK)
        gha_run_id,
        _str(rec.get("run_id") or args.run_id),
        _str(args.workflow),
        _str(args.branch),
        _str(args.sha)[:40].ljust(40)[:40],
        suite_name,
        _str(rec.get("model_name")),
        _str(rec.get("yaml_file")),
        _int(summary.get("total_tests")),
        _int(summary.get("spyre_enabled_count")),
        _int(summary.get("not_implemented_count")),
        _int(summary.get("cpu_fallback_count")),
        _int(summary.get("spyre_failed_count")),
        _str(rec.get("suite_outcome"), "unknown"),
        rec.get("suite_exit_code"),  # Nullable(Int32) — keep None
        _int(rec.get("suite_tests_total")),
        _int(rec.get("suite_tests_passed")),
        _int(rec.get("suite_tests_failed")),
        _int(rec.get("suite_tests_skipped")),
        _int(rec.get("suite_tests_error")),
        _int(rec.get("suite_tests_xfail")),
        _int(rec.get("suite_tests_xpass")),
        _float(rec.get("suite_duration_s")),
        _parse_ts(rec.get("triggered_at")),
        now,
    ]


def _build_variant_rows(
    rec: dict,
    args,
    gha_run_id: int,
    now: datetime,
) -> list[list]:
    """
    Flatten all per-variant records in a suite record into a list of rows
    for model_ops_variants.

    Covers: spyre_enabled, not_implemented variants (from operations.spyre_enabled
    / not_implemented / spyre_failed), and cpu_fallback entries.
    """
    suite_name = _str(rec.get("suite_name"))
    model_name = _str(rec.get("model_name"))
    yaml_file = _str(rec.get("yaml_file"))
    run_id = _str(rec.get("run_id") or args.run_id)
    triggered_at = _parse_ts(rec.get("triggered_at"))
    suite_id = _make_id(gha_run_id, suite_name)  # FK → model_ops_suites

    rows: list[list] = []
    ops = rec.get("operations", {})

    # Counter used to break ties when test_name is empty (cpu_fallback stubs)
    # so that every row still gets a unique variant_id.
    _seq = [0]

    def _base():
        return [
            gha_run_id,
            run_id,
            _str(args.workflow),
            _str(args.branch),
            _str(args.sha)[:40].ljust(40)[:40],
            suite_name,
            model_name,
            yaml_file,
        ]

    def _variant_row(v: dict, classification: str, status: str) -> list:
        operation = _str(v.get("operation"))
        test_name = _str(v.get("test_name"))
        # If test_name is empty (cpu_fallback stub with no variant data),
        # include a sequence counter so the SHA is still unique per row.
        _seq[0] += 1
        variant_id = _make_id(
            gha_run_id,
            suite_name,
            operation,
            classification,
            test_name,
            _seq[0] if not test_name else "",
        )
        return (
            [
                variant_id,  # variant_id (PK)
                suite_id,  # suite_id   (FK)
            ]
            + _base()
            + [
                operation,
                classification,
                test_name,
                status,
                _jstr(v.get("input_shapes", [])),
                _jstr(v.get("input_strides", [])),
                _jstr(v.get("input_dtypes", [])),
                triggered_at,
                now,
            ]
        )

    # ── spyre_enabled groups ────────────────────────────────────────────────
    for group in ops.get("spyre_enabled", []):
        for v in group.get("variants", []):
            rows.append(_variant_row(v, "spyre_enabled", "XPASS"))

    # ── not_implemented groups ──────────────────────────────────────────────
    for group in ops.get("not_implemented", []):
        for v in group.get("variants", []):
            rows.append(_variant_row(v, "not_implemented", "XFAIL"))

    # ── spyre_failed groups (mixed: some XPASS, some XFAIL) ────────────────
    for group in ops.get("spyre_failed", []):
        for v in group.get("xpass_variants", []):
            rows.append(_variant_row(v, "spyre_enabled", "XPASS"))
        for v in group.get("xfail_variants", []):
            rows.append(_variant_row(v, "not_implemented", "XFAIL"))

    # ── cpu_fallback ────────────────────────────────────────────────────────
    # Each entry has full variant data (same structure as spyre_enabled).
    # Insert every variant with classification="cpu_fallback" so the service
    # can read back shapes, dtypes, and test_names for fallback ops.
    # Fall back to a single stub row only when variants[] is absent/empty.
    for entry in ops.get("cpu_fallback", []):
        op = _str(entry.get("operation"))
        variants = entry.get("variants", [])
        if not op:
            continue
        if variants:
            for v in variants:
                rows.append(_variant_row(v, "cpu_fallback", "FALLBACK"))
        else:
            # Bare entry — only an operation name, no variant detail.
            # Use _variant_row via a minimal synthetic variant dict so
            # variant_id and suite_id are populated consistently.
            rows.append(
                _variant_row(
                    {"operation": op, "test_name": ""},
                    "cpu_fallback",
                    "FALLBACK",
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest model_ops JSON → ClickHouse (model_ops_suites + model_ops_variants).\n"
        "Tables are always dropped and recreated so the DB reflects only the latest run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json-file", required=True, help="Path to JSON from parse_model_ops_logs.py"
    )
    parser.add_argument(
        "--workflow", default="model-ops-tests", help="GHA workflow name"
    )
    parser.add_argument("--branch", default="", help="Git branch name")
    parser.add_argument("--sha", default="", help="Git commit SHA")
    parser.add_argument("--run-id", default="", help="GHA run ID (numeric string)")
    args = parser.parse_args()

    # ── Load JSON ────────────────────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[error] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as fh:
        records = json.load(fh)

    if not records:
        print("[info] JSON file contains no records — nothing to ingest.")
        sys.exit(0)
    # Filter out records without a valid suite_name
    records = [
        r
        for r in records
        if r.get("suite_name", "").strip() and not r["suite_name"].startswith(".")
    ]
    if not records:
        print("[info] No valid records after filtering — nothing to ingest.")
        sys.exit(0)

    print(f"[info] Loaded {len(records)} suite record(s) from {json_path.name}")

    # ── Connect ──────────────────────────────────────────────────────────────
    print(
        f"[info] Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("[info] Connected.\n")

    # ── Drop existing tables and recreate from scratch ────────────────────────
    recreate_tables(client)

    gha_run_id = _int(args.run_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── Build all rows up-front ───────────────────────────────────────────────
    all_suite_rows = []
    all_variant_rows = []

    for rec in records:
        suite_name = _str(rec.get("suite_name"))
        model_name = _str(rec.get("model_name"))

        try:
            suite_row = build_suite_row(rec, args, gha_run_id, now)
            all_suite_rows.append(suite_row)
            print(
                f"  [suite]    {suite_name!r}  model={model_name}  "
                f"outcome={rec.get('suite_outcome')}  "
                f"xpass={rec.get('summary', {}).get('spyre_enabled_count', 0)}  "
                f"xfail={rec.get('summary', {}).get('not_implemented_count', 0)}"
            )
        except Exception as exc:
            print(f"  [suite err]  {suite_name!r}: {exc}", file=sys.stderr)

        try:
            variant_rows = _build_variant_rows(rec, args, gha_run_id, now)
            all_variant_rows.extend(variant_rows)
            print(f"  [variants] {suite_name!r}: {len(variant_rows)} rows built")
        except Exception as exc:
            print(f"  [variants err] {suite_name!r}: {exc}", file=sys.stderr)

    # ── Batch insert ──────────────────────────────────────────────────────────
    print(f"\n[info] Inserting {len(all_suite_rows)} suite rows ...")
    if all_suite_rows:
        client.insert("model_ops_suites", all_suite_rows, column_names=SUITE_COLS)
        print(f"[info]   model_ops_suites    — {len(all_suite_rows)} rows inserted")

    print(f"[info] Inserting {len(all_variant_rows)} variant rows ...")
    if all_variant_rows:
        client.insert("model_ops_variants", all_variant_rows, column_names=VARIANT_COLS)
        print(f"[info]   model_ops_variants  — {len(all_variant_rows)} rows inserted")

    # ── Verify counts ─────────────────────────────────────────────────────────
    n_s = client.query("SELECT count() FROM model_ops_suites").result_rows[0][0]
    n_v = client.query("SELECT count() FROM model_ops_variants").result_rows[0][0]

    print("\n[info] ── Ingest complete ──────────────────────────────────────────")
    print(f"[info]   model_ops_suites   : {n_s} rows")
    print(f"[info]   model_ops_variants : {n_v} rows")
    print(f"[info]   gha_run_id         : {gha_run_id}")
    print(f"[info]   workflow           : {args.workflow}")
    print(f"[info]   branch             : {args.branch}")
    print(f"[info]   sha                : {args.sha[:12]}")


if __name__ == "__main__":
    main()
