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

"""Turn parsed hw-diagnostics records into hw_failure_diagnostics rows, and insert."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .hw_schema import HwFailureDiagnostics

NIL_UUID = HwFailureDiagnostics.NIL_UUID


@dataclass(frozen=True)
class RunContext:
    """The run coordinates the parsed records do not carry themselves."""

    run_id: str = NIL_UUID
    artifact_id: str = NIL_UUID
    component: str = ""
    arch: str = ""
    external_run_id: str = ""
    run_url: str = ""


class Fields:
    """Scalar coercions shared by RowBuilder -- honest defaults over silent nulls."""

    @staticmethod
    def ts(ts_str: str | None) -> datetime | None:
        """ISO-8601 string → naive UTC datetime, or None."""
        if not ts_str:
            return None
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None)  # ClickHouse DateTime64 wants naive
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def text(val, default: str = "") -> str:
        # Named `text`, not `str`: a method named `str` shadows the builtin for every
        # annotation textually below it in this class body (class bodies resolve bare
        # names sequentially, unlike a function's), which silently corrupted every
        # later `-> str` return annotation in this class to point at this method.
        return default if val is None else str(val).strip()

    @staticmethod
    def number(val, default: int = 0) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def detail_json(val) -> str:
        """Serialise failure_reason_detail dict → JSON string for ClickHouse."""
        if not val:
            return "{}"
        if isinstance(val, str):
            return val
        try:
            return json.dumps(val, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"


class RowBuilder:
    """Maps one parsed record + its RunContext onto one hw_failure_diagnostics row."""

    F = Fields

    @classmethod
    def build(cls, rec: dict, ctx: RunContext) -> list:
        """One JSON record → ordered list matching HwFailureDiagnostics.COLUMN_NAMES."""
        return [
            # ── Identity ──
            cls.F.text(ctx.run_id) or NIL_UUID,
            cls.F.text(ctx.artifact_id) or NIL_UUID,
            cls.F.text(ctx.component),
            cls.F.text(ctx.arch),
            cls.F.text(rec.get("suite_name")),
            cls.F.number(rec.get("attempt"), 1),
            cls.F.number(rec.get("total_attempts"), 1),
            cls.F.number(rec.get("pod_level_retry"), 0),
            cls.F.ts(rec.get("ingested_at")) or datetime.now(UTC).replace(tzinfo=None),
            # ── Outcome ──
            cls.F.text(rec.get("outcome"), "unknown"),
            rec.get("exit_code"),  # Nullable(Int32) — keep None
            # ── Failure classification ──
            cls.F.text(rec.get("failure_reason"), "none"),
            cls.F.text(rec.get("failure_phase")),
            cls.F.text(rec.get("retry_trigger")),
            cls.F.detail_json(rec.get("failure_reason_detail")),
            # ── Primary RAS event ──
            cls.F.text(rec.get("ras_code")),
            cls.F.text(rec.get("ras_name")),
            cls.F.text(rec.get("ras_description")),
            cls.F.text(rec.get("ras_action")),
            cls.F.text(rec.get("ras_category")),
            cls.F.text(rec.get("ras_severity")),
            cls.F.text(rec.get("ras_message")),
            cls.F.text(rec.get("ras_events_json"), "[]"),
            # ── Hardware identifiers ──
            cls.F.text(rec.get("node_name")),
            cls.F.text(rec.get("pci_device")),
            cls.F.text(rec.get("aiu_world_rank0")),
            cls.F.text(rec.get("card_serial")),
            cls.F.text(rec.get("chip_ecid_raw")),
            cls.F.text(rec.get("chip_wafer_id")),
            cls.F.text(rec.get("chip_mfg_x")),
            cls.F.text(rec.get("chip_mfg_y")),
            cls.F.text(rec.get("chip_chipy")),
            cls.F.text(rec.get("chip_chipx")),
            # ── Timestamps ──
            cls.F.ts(rec.get("first_error_ts")),  # Nullable(DateTime64)
            cls.F.ts(rec.get("attempt_start_ts")),
            # ── Pytest statistics ──
            cls.F.number(rec.get("tests_collected")),
            cls.F.number(rec.get("tests_passed")),
            cls.F.number(rec.get("tests_failed")),
            cls.F.number(rec.get("tests_error")),
            # ── Stall info ──
            cls.F.number(rec.get("stall_max_secs")),
            # The run_id hash inputs, kept so a row stays traceable to the producer
            # coordinate it was derived from. The record's run_id wins: one JSON file
            # is one run, but a re-ingest may be pointed at a file whose coordinate
            # differs from the flag.
            {
                k: v
                for k, v in (
                    (
                        "external_run_id",
                        cls.F.text(rec.get("run_id") or ctx.external_run_id),
                    ),
                    ("run_url", cls.F.text(ctx.run_url)),
                )
                if v
            },
        ]


class RecordLoader:
    """Reads and filters the parse step's JSON output."""

    @staticmethod
    def load(json_path: Path) -> list:
        """The parse step's JSON output."""
        with open(json_path) as fh:
            return json.load(fh)

    @staticmethod
    def filter_suites(records: list) -> list:
        """Drop .DS_Store / meta entries; caller re-tests the result for emptiness."""
        return [
            r
            for r in records
            if r.get("suite_name", "").strip() and not r["suite_name"].startswith(".")
        ]

    @staticmethod
    def insert(client, rows: list, table: str = "") -> None:
        """Insert with explicit column names, so row order is checked, not assumed."""
        table = table or HwFailureDiagnostics.DEFAULT_TABLE
        cols = HwFailureDiagnostics.COLUMN_NAMES
        if rows and len(rows[0]) != len(cols):
            raise ValueError(
                f"row has {len(rows[0])} values but {table} expects "
                f"{len(cols)}: build_row and HW_COLUMN_NAMES disagree"
            )
        client.insert(table=table, data=rows, column_names=list(cols))


# Function API, kept so installed consumers import one definition, not a copy.
build_row = RowBuilder.build
load_records = RecordLoader.load
filter_suite_records = RecordLoader.filter_suites
insert_rows = RecordLoader.insert
_str = Fields.text
