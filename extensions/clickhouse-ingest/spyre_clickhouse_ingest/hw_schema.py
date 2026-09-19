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


"""The hw_failure_diagnostics table: its column order and its self-migration.

Deliberately NOT a schema.Table. That model is for the four v2 tables: it rejects any row whose
keys do not match `columns` exactly and validates a `status` column against STATUS_VALUES. This
table is v1-generation and discovers its own columns at runtime via ADD COLUMN IF NOT EXISTS
(tolerated failures), and its verdict column is `outcome`, a different vocabulary -- so a frozen
Table would either validate nothing or turn a tolerated ALTER into a hard row-build error.

The table name is a parameter everywhere: the ingest exposes --table, and hardcoding it in the
dedup query while honouring the flag elsewhere made that flag a half-truth.
"""

import sys

# Column names — must match build_row() order and hw_failure_diagnostics DDL
HW_COLUMN_NAMES = (
    # Identity
    "run_id",
    "workflow",
    "branch",
    "commit_sha",
    "run_link",
    "suite_name",
    "attempt",
    "total_attempts",
    "pod_level_retry",
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
)

# Columns absent from older deployments of this table. ADD COLUMN IF NOT EXISTS is idempotent,
# so this runs on every ingest and is the only migration path this table has.
EXTRA_COLUMNS = (
    ("workflow", "LowCardinality(String) DEFAULT ''"),
    ("branch", "LowCardinality(String) DEFAULT ''"),
    ("commit_sha", "String DEFAULT ''"),
    ("run_link", "String DEFAULT ''"),
    ("failure_reason_detail", "String DEFAULT '{}'"),
    ("ras_category", "LowCardinality(String) DEFAULT ''"),
    ("ras_severity", "LowCardinality(String) DEFAULT ''"),
    ("ras_message", "String DEFAULT ''"),
    ("ras_events_json", "String DEFAULT '[]'"),
    # True when the row came from a pod-level-retry job (a fresh-pod re-run), not the original.
    ("pod_level_retry", "Bool DEFAULT false"),
)

DEFAULT_TABLE = "hw_failure_diagnostics"


def already_ingested(
    client, run_id: str, workflow: str, table: str = DEFAULT_TABLE
) -> bool:
    """True when this (run_id, workflow) pair already has rows, so a re-run does not double-insert."""
    result = client.query(
        f"SELECT count() FROM {table} "
        "WHERE run_id = {run_id:String} AND workflow = {workflow:String}",
        parameters={"run_id": run_id, "workflow": workflow},
    )
    return result.result_rows[0][0] > 0


def ensure_extra_columns(client, table: str = DEFAULT_TABLE) -> None:
    """Add any missing EXTRA_COLUMNS. Non-fatal per column: the usual cause is that it already
    exists, and a failure here must not cost the run its rows."""
    for col_name, col_type in EXTRA_COLUMNS:
        try:
            client.command(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
            )
        except Exception as exc:
            print(f"  [warn] Could not add column {col_name}: {exc}", file=sys.stderr)
