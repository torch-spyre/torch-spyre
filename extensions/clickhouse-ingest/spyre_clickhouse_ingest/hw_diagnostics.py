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


"""Turn parsed hw-diagnostics records into hw_failure_diagnostics rows and insert them.

Takes a RunContext, not an argparse Namespace: the CLI shape belongs to the per-repo shim, and a
library that reads `args.sha` cannot be tested without building a parser.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .hw_schema import DEFAULT_TABLE, HW_COLUMN_NAMES


NIL_UUID = "00000000-0000-0000-0000-000000000000"


@dataclass(frozen=True)
class RunContext:
    """The run coordinates the parsed records do not carry themselves.

    `run_id` is the DERIVED uuid that joins artifact_results, not the producer's raw coordinate --
    that is `external_run_id`, a hash input, kept in props. artifact_id defaults to the nil UUID
    so an un-updated caller writes "not linked" rather than a hash that would join everything.

    No workflow/branch/sha/run_link: the first is the test_type run_id is already hashed from, the
    last is derivable from run_id, and the middle two are run-level facts that never varied per
    row.
    """

    run_id: str = NIL_UUID
    artifact_id: str = NIL_UUID
    component: str = ""
    arch: str = ""
    external_run_id: str = ""
    run_url: str = ""


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


def build_row(rec: dict, ctx: RunContext) -> list:
    """
    Map one JSON record → ordered list matching HW_COLUMN_NAMES.
    Order is HW_COLUMN_NAMES; insert_rows checks the arity.
    """
    return [
        # ── Identity ──────────────────────────────────────────────────────
        _str(ctx.run_id) or NIL_UUID,
        _str(ctx.artifact_id) or NIL_UUID,
        _str(ctx.component),
        _str(ctx.arch),
        _str(rec.get("suite_name")),
        _int(rec.get("attempt"), 1),
        _int(rec.get("total_attempts"), 1),
        _int(rec.get("pod_level_retry"), 0),
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
        # The run_id hash inputs, kept so a row stays traceable to the producer coordinate it
        # was derived from. The record's own run_id wins: one JSON file is one run, but a
        # re-ingest may be pointed at a file whose coordinate differs from the flag.
        {
            k: v
            for k, v in (
                ("external_run_id", _str(rec.get("run_id") or ctx.external_run_id)),
                ("run_url", _str(ctx.run_url)),
            )
            if v
        },
    ]


def load_records(json_path: Path) -> list:
    """The parse step's JSON output."""
    with open(json_path) as fh:
        return json.load(fh)


def filter_suite_records(records: list) -> list:
    """Drop .DS_Store / meta entries that are not suites.

    Separate from the emptiness check on purpose: this can empty a non-empty list, and the caller
    must re-test before indexing records[0] or connecting to ClickHouse.
    """
    return [
        r
        for r in records
        if r.get("suite_name", "").strip() and not r["suite_name"].startswith(".")
    ]


def insert_rows(client, rows: list, table: str = DEFAULT_TABLE) -> None:
    """Insert with explicit column names, so row order is checked rather than assumed.

    The arity guard is the one property worth borrowing from schema.Table: a column added to
    HW_COLUMN_NAMES but not to build_row would otherwise shift every later value one column left.
    """
    if rows and len(rows[0]) != len(HW_COLUMN_NAMES):
        raise ValueError(
            f"row has {len(rows[0])} values but {table} expects "
            f"{len(HW_COLUMN_NAMES)}: build_row and HW_COLUMN_NAMES disagree"
        )
    client.insert(table=table, data=rows, column_names=list(HW_COLUMN_NAMES))
