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

"""hw_failure_diagnostics: v1-generation, self-migrating -- NOT a schema.Table."""

import sys


class HwFailureDiagnostics:
    """Column order, self-migration and dedup check for the self-migrating table."""

    DEFAULT_TABLE = "hw_failure_diagnostics"
    NIL_UUID = "00000000-0000-0000-0000-000000000000"

    # Must match build_row() order and the live DDL; the table name is a parameter
    # everywhere, since hardcoding it here while honouring --table elsewhere would make
    # that flag a half-truth.
    COLUMN_NAMES = (
        "run_id",
        "artifact_id",
        "component",
        "arch",
        "suite_name",
        "attempt",
        "total_attempts",
        "pod_level_retry",
        "ingested_at",
        "outcome",
        "exit_code",
        "failure_reason",
        "failure_phase",
        "retry_trigger",
        "failure_reason_detail",
        "ras_code",
        "ras_name",
        "ras_description",
        "ras_action",
        "ras_category",
        "ras_severity",
        "ras_message",
        "ras_events_json",
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
        "first_error_ts",
        "attempt_start_ts",
        "tests_collected",
        "tests_passed",
        "tests_failed",
        "tests_error",
        "stall_max_secs",
        "props",  # external_run_id (the raw run_id hash input) and run_url
    )

    # Columns absent from older deployments of this table (types match schema/45-hw-diagnostics
    # .sql, whose CREATE TABLE IF NOT EXISTS is a no-op against an already-existing table).
    # ADD COLUMN IF NOT EXISTS is idempotent, so this runs on every ingest and is the only
    # migration path this table has.
    EXTRA_COLUMNS = (
        (
            "artifact_id",
            "UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000')",
        ),
        ("component", "LowCardinality(String) DEFAULT ''"),
        ("arch", "LowCardinality(String) DEFAULT ''"),
        ("props", "Map(LowCardinality(String), String)"),
    )

    @classmethod
    def ensure_extra_columns(cls, client, table: str = "") -> None:
        """Add any missing EXTRA_COLUMNS. Non-fatal per column: the usual cause is that it
        already exists, and a failure here must not cost the run its rows."""
        table = table or cls.DEFAULT_TABLE
        for col_name, col_type in cls.EXTRA_COLUMNS:
            try:
                client.command(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                )
            except Exception as exc:
                print(
                    f"  [warn] Could not add column {col_name}: {exc}", file=sys.stderr
                )

    @classmethod
    def already_ingested(
        cls, client, run_id: str, component: str, table: str = ""
    ) -> bool:
        """True when this (component, run_id) already has rows -- guards against double-insert."""
        result = client.query(
            f"SELECT count() FROM {table or cls.DEFAULT_TABLE} "
            "WHERE component = {component:String} AND run_id = {run_id:UUID}",
            parameters={"run_id": run_id, "component": component},
        )
        return result.result_rows[0][0] > 0


# Constant/function API, kept so installed consumers import one definition, not a copy.
HW_COLUMN_NAMES = HwFailureDiagnostics.COLUMN_NAMES
NIL_UUID = HwFailureDiagnostics.NIL_UUID
DEFAULT_TABLE = HwFailureDiagnostics.DEFAULT_TABLE
EXTRA_COLUMNS = HwFailureDiagnostics.EXTRA_COLUMNS
ensure_extra_columns = HwFailureDiagnostics.ensure_extra_columns
already_ingested = HwFailureDiagnostics.already_ingested
