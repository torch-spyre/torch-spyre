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

"""model_priority_events is append-only; v_model_priority is the newest event per use case."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

NIL = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "74-model-priority-events.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.close()


def _event(n, **over):
    row = {"event_id": f"018f0000-0000-7000-8000-00000000000{n}", "ts": f"2026-10-31 0{n}:00:00",
           "use_case": "chat", "action": "set", "model_order": ["cb", "gemma"], "actor": "amy@ibm.com"}  # fmt: skip
    return {**row, **over}


def _insert(s, row):
    s.query("INSERT INTO model_priority_events FORMAT JSONEachRow\n" + json.dumps(row))


def _rows(s, sql):
    raw = s.query(sql, "JSONEachRow").data()
    raw = raw.decode() if isinstance(raw, bytes) else raw
    return [json.loads(line) for line in raw.splitlines()]


def test_rows_are_validated(chdb_session):
    for bad in (
        {"event_id": NIL},
        {"action": "edit"},
        {"actor": ""},
        {"actor_kind": "root"},
        {"use_case": "Chat; DROP"},
        {"model_order": []},
    ):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            _insert(chdb_session, _event(1, **bad))


def test_a_reset_needs_no_order(chdb_session):
    _insert(chdb_session, _event(1, action="reset", model_order=[]))
    assert _rows(chdb_session, "SELECT action FROM v_model_priority") == [{"action": "reset"}]


def test_the_newest_event_per_use_case_wins_and_older_ones_stay(chdb_session):
    s = chdb_session
    _insert(s, _event(1))
    _insert(s, _event(2, model_order=["gemma", "cb"], reason="cb quota"))
    _insert(s, _event(3, use_case="slack", model_order=["gemma"]))
    got = _rows(s, "SELECT use_case, model_order, reason FROM v_model_priority ORDER BY use_case")
    assert got == [
        {"use_case": "chat", "model_order": ["gemma", "cb"], "reason": "cb quota"},
        {"use_case": "slack", "model_order": ["gemma"], "reason": ""},
    ]
    assert _rows(s, "SELECT count() AS n FROM model_priority_events") == [{"n": 3}]
