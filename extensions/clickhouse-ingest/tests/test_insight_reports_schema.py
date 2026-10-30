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

"""insight_reports is an append-only event log; v_insight_reports is the latest state per report."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

RID = "018f0000-0000-7000-8000-000000000001"
NIL = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "65-insight-reports.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.close()


def _event(**over):
    row = {"report_id": RID, "state": "saved", "actor": "alice@ibm.com",
           "visibility": "private", "title": "Pass rate", "question": "pass rate by arch?",
           "template": "headline_chart", "charts": "[]"}  # fmt: skip
    return {**row, **over}


def test_rows_are_validated(chdb_session):
    s = chdb_session
    for bad in (
        {"report_id": NIL},
        {"state": "archived"},
        {"visibility": "public"},
    ):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            s.query(
                "INSERT INTO insight_reports FORMAT JSONEachRow\n"
                + json.dumps(_event(**bad))
            )


def test_the_view_keeps_the_owner_and_the_latest_saved_content(chdb_session):
    s = chdb_session

    def insert(row):
        s.query("INSERT INTO insight_reports FORMAT JSONEachRow\n" + json.dumps(row))

    insert(_event(ts="2026-10-30 10:00:00.000"))
    insert(_event(ts="2026-10-30 11:00:00.000", actor="bob@ibm.com",
                  visibility="shared", title="Renamed"))  # fmt: skip
    insert(_event(ts="2026-10-30 12:00:00.000", state="deleted", actor="bob@ibm.com",
                  title="", question="", template="", charts=""))  # fmt: skip
    got = json.loads(
        s.query(
            "SELECT state, owner, visibility, title, question FROM v_insight_reports",
            "JSONEachRow",
        ).data()
    )
    assert got == {"state": "deleted", "owner": "alice@ibm.com", "visibility": "shared",
                   "title": "Renamed", "question": "pass rate by arch?"}  # fmt: skip
