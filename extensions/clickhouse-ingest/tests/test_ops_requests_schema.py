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

"""ops_requests.status: queued when a request launches a build, then set to its outcome."""

import json

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

DDL = (SCHEMA_DIR / "41-ops-requests.sql").read_text()
MIGRATION = (SCHEMA_DIR / "migrations" / "024_ops_requests_status.sql").read_text()


class Chdb:
    def __init__(self):
        self.s = session.Session()

    def command(self, sql):
        return str(self.s.query(sql, "TSV")).rstrip("\n")

    def rows(self, sql):
        raw = str(self.s.query(sql, "JSONEachRow"))
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, *rows):
        self.command(
            "INSERT INTO ops_requests FORMAT JSONEachRow\n"
            + "\n".join(json.dumps(r) for r in rows)
        )

    def create_query(self):
        return self.command(
            "SELECT create_table_query FROM system.tables WHERE name = 'ops_requests'"
        )


@pytest.fixture
def db():
    c = Chdb()
    yield c
    c.s.close()


def _row(state, **over):
    # The writer's audit_row() today: no status, so the column's DEFAULT fills it.
    row = {"requested_at": "2026-11-05 10:00:00.000", "requested_by": "alice@ibm.com",
           "request": "run preset=perf-only", "verb": "run", "params": {}, "state": state}  # fmt: skip
    return {**row, **over}


def _before_024():
    (stmt,) = SchemaApplier.statements(DDL)
    stmt = re.sub(r"(?m)^\s*(?:status|CONSTRAINT chk_status)\s[^\n]*\n", "", stmt)
    return re.sub(r",(\s*\))", r"\1", stmt)


def test_a_launched_request_starts_queued(db):
    db.command(DDL)
    db.insert(
        _row("launched", job_run="https://j/trigger/1", build_url="https://j/orch/7"),
        _row("rejected", job_run="https://j/trigger/2"),
        _row("dry_run", job_run="https://j/trigger/3"),
    )
    got = db.rows("SELECT state, status FROM ops_requests ORDER BY job_run")
    assert got == [
        {"state": "launched", "status": "queued"},
        {"state": "rejected", "status": ""},
        {"state": "dry_run", "status": ""},
    ]


def test_status_is_validated(db):
    db.command(DDL)
    db.insert(_row("launched", status="passed"))
    with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
        db.insert(_row("launched", status="running"))


@pytest.mark.parametrize("outcome", ["passed", "failed", "error"])
def test_the_outcome_is_set_in_place(db, outcome):
    db.command(DDL)
    db.insert(_row("launched", build_url="https://j/orch/7"), _row("rejected"))
    db.command(
        f"ALTER TABLE ops_requests UPDATE status = '{outcome}' "
        "WHERE build_url = 'https://j/orch/7' SETTINGS mutations_sync = 2"
    )
    got = db.rows("SELECT state, status FROM ops_requests ORDER BY state")
    assert got == [
        {"state": "launched", "status": outcome},
        {"state": "rejected", "status": ""},
    ]


def test_migration_reaches_the_file_and_leaves_older_launches_unqueued(db):
    db.command(DDL)
    want = db.create_query()
    db.command("DROP TABLE ops_requests")

    db.command(_before_024())
    assert "status" not in db.create_query()
    db.insert(_row("launched", job_run="https://j/trigger/old"))
    SchemaApplier.run_migration(db, MIGRATION)
    assert db.create_query() == want

    db.insert(_row("launched", job_run="https://j/trigger/new"))
    got = db.rows("SELECT job_run, status FROM ops_requests ORDER BY job_run")
    assert got == [
        {"job_run": "https://j/trigger/new", "status": "queued"},
        {"job_run": "https://j/trigger/old", "status": ""},
    ]
