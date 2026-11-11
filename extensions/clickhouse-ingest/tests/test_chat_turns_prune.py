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

"""The scheduled prune deletes by (turn_id, actor): one person's feedback or purge can never delete another person's turn."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

PRUNE = SCHEMA_DIR.parent / "maintenance" / "prune-chat-turns.sql"
OLD = "2020-01-01 00:00:00.000"
V_TURN = "018f0000-0000-7000-8000-00000000000a"
X_TURN = "018f0000-0000-7000-8000-00000000000b"


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "66-chat-turns.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.close()


def _now(**delta):
    when = datetime.now(timezone.utc) + timedelta(**delta)
    return when.strftime("%Y-%m-%d %H:%M:%S.000")


def _row(turn_id, actor, ts, event="turn"):
    row = {"turn_id": turn_id, "event": event, "surface": "chat", "actor": actor,
           "actor_kind": "human", "question": "q", "ts": ts}  # fmt: skip
    return {**row, "thumbs": "up"} if event == "feedback" else row


def _add(s, *rows):
    for row in rows:
        s.query("INSERT INTO chat_turns FORMAT JSONEachRow\n" + json.dumps(row))


def _setting(s, actor, ts, **fields):
    row = {"actor": actor, "set_by": actor, "ts": ts, **fields}
    s.query("INSERT INTO chat_turn_retention FORMAT JSONEachRow\n" + json.dumps(row))


def _prune(s):
    for stmt in SchemaApplier.statements(PRUNE.read_text()):
        s.query(stmt)


def _left(s):
    out = s.query(
        "SELECT turn_id, actor, event FROM chat_turns ORDER BY turn_id, actor, event",
        "JSONEachRow",
    ).data()
    return [(r["turn_id"], r["actor"], r["event"]) for r in map(json.loads, out.splitlines())]  # fmt: skip


def test_feedback_cannot_delete_another_persons_turn(chdb_session):
    s = chdb_session
    _add(
        s,
        _row(V_TURN, "victim@ibm.com", OLD),
        _row(V_TURN, "victim@ibm.com", OLD, "feedback"),
        _row(V_TURN, "mallory@ibm.com", OLD, "feedback"),
    )
    _setting(s, "mallory@ibm.com", _now(minutes=-1), kind="purge")
    _prune(s)
    assert _left(s) == [
        (V_TURN, "victim@ibm.com", "feedback"),
        (V_TURN, "victim@ibm.com", "turn"),
    ]


def test_feedback_goes_with_its_own_turn(chdb_session):
    s = chdb_session
    _add(
        s,
        _row(V_TURN, "a@ibm.com", OLD),
        _row(V_TURN, "a@ibm.com", _now(), "feedback"),
    )
    _setting(s, "a@ibm.com", _now(minutes=-1), kind="retention", retention_days=7)
    _prune(s)
    assert _left(s) == []


def test_retention_and_purge_cases(chdb_session):
    s = chdb_session
    keep = "018f0000-0000-7000-8000-0000000000c"
    _add(
        s,
        _row(keep + "1", "nosetting@ibm.com", OLD),
        _row(keep + "2", "forever@ibm.com", OLD),
        _row(keep + "3", "recent@ibm.com", _now(days=-1)),
        _row(keep + "4", "purger@ibm.com", _now(minutes=-30)),
        _row(keep + "5", "purger@ibm.com", OLD),
    )
    _setting(s, "forever@ibm.com", _now(minutes=-1), kind="retention", retention_days=0)
    _setting(s, "recent@ibm.com", _now(minutes=-1), kind="retention", retention_days=7)
    _setting(s, "purger@ibm.com", _now(minutes=-10), kind="purge")
    _prune(s)
    assert [t for t, _, _ in _left(s)] == [keep + "1", keep + "2", keep + "3"]


def test_an_empty_retention_table_deletes_nothing(chdb_session):
    s = chdb_session
    _add(s, _row(V_TURN, "a@ibm.com", OLD), _row(X_TURN, "b@ibm.com", _now()))
    _prune(s)
    assert len(_left(s)) == 2
