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

"""chat_turns is an append-only turn/feedback log; v_chat_turns is one row per turn with its latest thumbs."""

import json

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

TID = "018f0000-0000-7000-8000-000000000001"
NIL = "00000000-0000-0000-0000-000000000000"


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


def _event(**over):
    row = {"turn_id": TID, "event": "turn", "surface": "chat", "actor": "alice@ibm.com",
           "actor_kind": "user", "question": "q", "tool_calls": 2, "latency_ms": 100}  # fmt: skip
    return {**row, **over}


def _insert(s, row):
    s.query("INSERT INTO chat_turns FORMAT JSONEachRow\n" + json.dumps(row))


def test_rows_are_kept_forever_unless_a_ttl_is_set():
    sql = (SCHEMA_DIR / "66-chat-turns.sql").read_text()
    assert not re.search(r"^TTL ", sql, re.M)


def test_rows_are_validated(chdb_session):
    for bad in (
        {"turn_id": NIL},
        {"event": "edit"},
        {"surface": "email"},
        {"actor_kind": "robot"},
        {"thumbs": "sideways"},
        {"actor_kind": "anonymous", "actor": "alice@ibm.com"},
        {"actor_kind": "user", "actor": "anon:abc"},
    ):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            _insert(chdb_session, _event(**bad))


def test_the_view_joins_the_latest_thumbs_to_the_turn(chdb_session):
    s = chdb_session
    _insert(s, _event(ts="2026-10-30 10:00:00.000"))
    _insert(
        s,
        _event(
            ts="2026-10-30 10:01:00.000",
            event="feedback",
            thumbs="up",
            actor_kind="user",
        ),
    )
    _insert(
        s,
        _event(
            ts="2026-10-30 10:02:00.000",
            event="feedback",
            thumbs="down",
            reason="wrong",
        ),
    )
    _insert(
        s,
        _event(
            turn_id="018f0000-0000-7000-8000-000000000002",
            event="feedback",
            thumbs="up",
        ),
    )
    got = json.loads(
        s.query(
            "SELECT thumbs, thumbs_reason, actor, tool_calls FROM v_chat_turns",
            "JSONEachRow",
        ).data()
    )
    assert got == {"thumbs": "down", "thumbs_reason": "wrong", "actor": "alice@ibm.com", "tool_calls": 2}  # fmt: skip


def test_a_turn_without_feedback_has_none(chdb_session):
    _insert(chdb_session, _event())
    got = json.loads(
        chdb_session.query("SELECT thumbs FROM v_chat_turns", "JSONEachRow").data()
    )
    assert got == {"thumbs": "none"}
