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

"""ops_audit_events is an append-only record of control-route attempts."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

EID = "018f0000-0000-7000-8000-000000000001"
NIL = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "66-ops-audit-events.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.close()


def _event(**over):
    row = {"event_id": EID, "actor": "alice@ibm.com", "route": "/api/cache/invalidate",
           "method": "POST", "action": "cache.invalidate", "result": "ok", "status": 200}  # fmt: skip
    return {**row, **over}


def _insert(s, row):
    s.query("INSERT INTO ops_audit_events FORMAT JSONEachRow\n" + json.dumps(row))


def test_rows_are_validated(chdb_session):
    for bad in ({"event_id": NIL}, {"result": "denied"}):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            _insert(chdb_session, _event(**bad))


def test_a_refused_attempt_may_have_no_actor(chdb_session):
    s = chdb_session
    _insert(s, _event(actor="", result="refused", status=401, reason="no login"))
    _insert(s, _event(event_id="018f0000-0000-7000-8000-000000000002"))
    got = json.loads(
        "[" + ",".join(
            s.query(
                "SELECT actor, result, status FROM ops_audit_events ORDER BY event_id",
                "JSONEachRow",
            ).data().decode().split("\n")[:-1]
        ) + "]"
    )  # fmt: skip
    assert got == [
        {"actor": "", "result": "refused", "status": 401},
        {"actor": "alice@ibm.com", "result": "ok", "status": 200},
    ]
