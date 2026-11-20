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

"""ci_access_audit: append-only record of cross-user views and cancels of CI requests."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

RID = "018f0000-0000-7000-8000-000000000001"
NIL = "00000000-0000-0000-0000-000000000000"


def _session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "73-ci-access-audit.sql").read_text()
    ):
        s.query(stmt)
    return s


def _row(**over):
    row = {"action": "cancel", "ci_request_id": RID, "actor": "bob@ibm.com", "target": "alice@ibm.com",
           "result": "ok", "permission": "jobs:cancel-any"}  # fmt: skip
    return {**row, **over}


def test_rows_are_validated_and_defaults_fill_in():
    s = _session()

    def insert(row):
        s.query("INSERT INTO ci_access_audit FORMAT JSONEachRow\n" + json.dumps(row))

    for bad in (
        {"ci_request_id": NIL},
        {"action": "delete"},
        {"result": "cancelled"},
        {"actor_kind": "bot"},
        {"channel": "slack"},
    ):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            insert(_row(**bad))
    insert(_row())
    got = json.loads(
        s.query(
            "SELECT channel, actor_kind, reason FROM ci_access_audit", "JSONEachRow"
        ).data()
    )
    assert got == {"channel": "ui", "actor_kind": "human", "reason": ""}


def test_there_is_no_ttl_and_the_view_has_one_row_per_event():
    s = _session()
    assert "TTL" not in (SCHEMA_DIR / "73-ci-access-audit.sql").read_text()
    for event_id in (
        "018f0000-0000-7000-8000-0000000000a1",
        "018f0000-0000-7000-8000-0000000000a2",
    ):
        s.query(
            "INSERT INTO ci_access_audit FORMAT JSONEachRow\n"
            + json.dumps(_row(event_id=event_id))
        )
    got = s.query("SELECT count() AS n FROM v_ci_access_audit", "JSONEachRow").data()
    assert json.loads(got) == {"n": 2}
