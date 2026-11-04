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

"""effect_status is append-only triage events; v_effect_status is the latest state per signature."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

NIL = "00000000-0000-0000-0000-000000000000"
SIG = "test_unbind_* assertionerror"


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "67-effect-status.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.close()


def _event(n, ts, **over):
    row = {"event_id": f"018f0000-0000-7000-8000-{n:012d}", "ts": ts,
           "signature": SIG, "actor": "alice@ibm.com"}  # fmt: skip
    return {**row, **over}


def _insert(s, *rows):
    s.query(
        "INSERT INTO effect_status FORMAT JSONEachRow\n"
        + "\n".join(json.dumps(r) for r in rows)
    )


def _rows(s, sql):
    raw = s.query(sql, "JSONEachRow").data()
    raw = raw.decode() if isinstance(raw, bytes) else raw
    return [json.loads(line) for line in raw.splitlines()]


def test_rows_are_validated(chdb_session):
    for bad in (
        {"event_id": NIL},
        {"signature": ""},
        {"state": "wontfix"},
        {"prior_state": "wontfix"},
        {"actor_kind": "root"},
        {"source": "analyst"},
        {"kind": "delete"},
    ):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            _insert(chdb_session, _event(1, "2026-10-30 10:00:00", **bad))


def test_the_view_keeps_the_latest_state_and_the_first_taker(chdb_session):
    s = chdb_session
    _insert(
        s,
        _event(1, "2026-10-30 10:00:00", state="acknowledged", prior_state="untriaged",
               owner="alice@ibm.com", bug_url="https://example.test/bug/1"),
        _event(2, "2026-10-30 11:00:00", state="invalid", prior_state="acknowledged",
               actor="bob@ibm.com", reason="infra blip"),
        _event(3, "2026-10-30 12:00:00", kind="note", actor="carol@ibm.com"),
    )  # fmt: skip
    got = _rows(
        s,
        "SELECT state, owner, bug_url, reason, last_actor, taken_by, events "
        "FROM v_effect_status",
    )
    assert got == [
        {"state": "invalid", "owner": "alice@ibm.com", "bug_url": "https://example.test/bug/1",
         "reason": "infra blip", "last_actor": "carol@ibm.com", "taken_by": "alice@ibm.com", "events": 3}
    ]  # fmt: skip


def test_one_row_per_signature(chdb_session):
    s = chdb_session
    _insert(
        s,
        _event(1, "2026-10-30 10:00:00", state="acknowledged"),
        _event(2, "2026-10-30 10:00:00", signature="other", state="downstream"),
    )
    got = _rows(s, "SELECT signature, state FROM v_effect_status ORDER BY signature")
    assert got == [
        {"signature": "other", "state": "downstream"},
        {"signature": SIG, "state": "acknowledged"},
    ]
