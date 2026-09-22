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

# Column names — must match build_row() order and hw_failure_diagnostics DDL
HW_COLUMN_NAMES = (
    # Identity. No workflow/branch/commit_sha/run_link: the first IS the test_type (a run_id
    # hash input), the last is derivable from run_id, and the middle two are run-level facts
    # that never varied per row (0 of 6,090 prod run_ids carried two values of any of them).
    "run_id",
    "artifact_id",
    "component",
    "arch",
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
    # external_run_id (the raw coordinate run_id is hashed from) and run_url.
    "props",
)

NIL_UUID = "00000000-0000-0000-0000-000000000000"

DEFAULT_TABLE = "hw_failure_diagnostics"


def already_ingested(
    client, run_id: str, component: str, table: str = DEFAULT_TABLE
) -> bool:
    """True when this (component, run_id) pair already has rows, so a re-run does not
    double-insert.

    Keyed on the sort-key prefix, which is what makes it cheap: the v1 table was created
    unsorted and this one query read every row (measured 840,032 of 840,032 on prod). It
    previously filtered (run_id, workflow); workflow is gone, being the test_type that run_id
    is already hashed from.
    """
    result = client.query(
        f"SELECT count() FROM {table} "
        "WHERE component = {component:String} AND run_id = {run_id:UUID}",
        parameters={"run_id": run_id, "component": component},
    )
    return result.result_rows[0][0] > 0
