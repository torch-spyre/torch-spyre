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

"""Static checks on schema/57; the views' behaviour is proven on a sandbox (verify_schema)."""

import re

from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

FILE = SCHEMA_DIR / "57-failure-forensics-views.sql"
VIEWS = {
    "v_result_run_context",
    "v_case_run_history",
    "v_failure_recurrence",
    "v_failure_first_seen",
}


def test_declares_only_the_forensics_views_idempotently():
    stmts = SchemaApplier.statements(FILE.read_text())
    names = {re.search(r"VIEW IF NOT EXISTS (\w+)", s).group(1) for s in stmts}
    assert names == VIEWS and len(stmts) == len(VIEWS)


def test_readme_lists_the_file_and_every_view():
    row = next(
        ln
        for ln in (SCHEMA_DIR / "README.md").read_text().splitlines()
        if "57-failure-forensics-views.sql" in ln
    )
    assert all(v in row for v in VIEWS)
