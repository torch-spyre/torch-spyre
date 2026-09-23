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

"""Pins hw_failure_diagnostics' self-migration.

hw_failure_diagnostics has no CI-run apply path (schema/45-hw-diagnostics.sql is CREATE TABLE IF
NOT EXISTS, a no-op against an already-existing table): ensure_extra_columns is the only thing
that ever brings a deployed table's columns up to date with HwFailureDiagnostics.COLUMN_NAMES.
These tests pin that every EXTRA_COLUMNS entry is actually issued as an ALTER, and that one
column already existing (a tolerated failure) does not stop the rest from being added.
"""

from spyre_clickhouse_ingest.hw_schema import HwFailureDiagnostics


class FakeClient:
    """Records every ALTER TABLE statement `command` is called with."""

    def __init__(self, fail_columns=()):
        self.fail_columns = set(fail_columns)
        self.commands = []

    def command(self, sql):
        self.commands.append(sql)
        if any(f"ADD COLUMN IF NOT EXISTS {c} " in sql for c in self.fail_columns):
            raise RuntimeError("column already exists")


def test_ensure_extra_columns_issues_one_alter_per_column():
    client = FakeClient()
    HwFailureDiagnostics.ensure_extra_columns(client, table="hw_failure_diagnostics")
    assert len(client.commands) == len(HwFailureDiagnostics.EXTRA_COLUMNS)
    for col_name, col_type in HwFailureDiagnostics.EXTRA_COLUMNS:
        expected = (
            f"ALTER TABLE hw_failure_diagnostics ADD COLUMN IF NOT EXISTS "
            f"{col_name} {col_type}"
        )
        assert expected in client.commands


def test_ensure_extra_columns_tolerates_a_failing_column():
    client = FakeClient(fail_columns={"component"})
    HwFailureDiagnostics.ensure_extra_columns(client, table="hw_failure_diagnostics")
    # component's ALTER failed, but every other column was still attempted.
    assert len(client.commands) == len(HwFailureDiagnostics.EXTRA_COLUMNS)


def test_ensure_extra_columns_defaults_to_default_table():
    client = FakeClient()
    HwFailureDiagnostics.ensure_extra_columns(client)
    assert all(
        sql.startswith(f"ALTER TABLE {HwFailureDiagnostics.DEFAULT_TABLE} ")
        for sql in client.commands
    )
