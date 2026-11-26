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

"""ch_mgmt: the login registry (latest row wins) and its append-only event log."""

import json
import regex as re

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

CH_MGMT_DIR = SCHEMA_DIR.parent / "schema-ch-mgmt"
LOGIN = "app_nightly"


def _tables():
    return {
        o.name: o
        for path, text in SchemaApplier.selected_files(CH_MGMT_DIR)
        for o in SchemaApplier.objects(path, text)
    }


def _request(**over):
    row = {
        "login": LOGIN,
        "version": "2026-11-26 10:00:00.000",
        "requester": "alice@ibm.com",
        "owner": "alice@ibm.com",
        "tier": 1,
        "grants": json.dumps([["SELECT", "spyre_v2", "benchmarks"]]),
        "status": "pending",
        "needs_approval": 0,
        "password_issued": 0,
        "created_at": "2026-11-26 10:00:00.000",
        "updated_by": "alice@ibm.com",
    }
    return {**row, **over}


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for obj in _tables().values():
        s.query(obj.sql)
    yield s
    s.close()


def _rows(s, sql):
    raw = s.query(sql, "JSONEachRow").data()
    raw = raw.decode() if isinstance(raw, bytes) else raw
    return [json.loads(line) for line in raw.splitlines()]


def _insert(s, table, row):
    s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n" + json.dumps(row))


def test_ch_mgmt_is_separate_from_spyre_v2_schema():
    assert set(_tables()) == {"login_requests", "login_events"}
    v2 = {
        o.name
        for path, text in SchemaApplier.selected_files(SCHEMA_DIR)
        for o in SchemaApplier.objects(path, text)
    }
    assert not v2 & {"login_requests", "login_events"}
    assert not {p.name for p in CH_MGMT_DIR.glob("*.sql")} & {
        p.name for p in SCHEMA_DIR.glob("*.sql")
    }


REQUEST_COLUMNS = {
    "login", "version", "requester", "owner", "tier", "grants", "status", "approver",
    "reason", "host", "expires_at", "needs_approval", "password_issued",
    "last_rotated_at", "old_password_valid_until", "created_at", "updated_by",
}  # fmt: skip
EVENT_COLUMNS = {"event_id", "ts", "login", "actor", "action", "detail", "reason"}


def test_the_columns_are_exactly_the_contract_so_no_secret_can_be_added(chdb_session):
    got: dict = {}
    for c in _rows(
        chdb_session,
        "SELECT table, name FROM system.columns WHERE database = currentDatabase()",
    ):
        got.setdefault(c["table"], set()).add(c["name"])
    assert got == {"login_requests": REQUEST_COLUMNS, "login_events": EVENT_COLUMNS}


def test_engines_and_sort_keys(chdb_session):
    got = {
        r["name"]: r
        for r in _rows(
            chdb_session,
            "SELECT name, engine, engine_full, sorting_key FROM system.tables "
            "WHERE database = currentDatabase()",
        )
    }
    req, ev = got["login_requests"], got["login_events"]
    assert req["engine"] == "ReplacingMergeTree"
    assert "ReplacingMergeTree(version)" in req["engine_full"]
    assert req["sorting_key"] == "login"
    assert ev["engine"] == "MergeTree"
    assert ev["sorting_key"] == "login, ts"


def _grants():
    out = set()
    text = (CH_MGMT_DIR / "roles" / "chlogins-roles.sql").read_text()
    for s in SchemaApplier.statements(text):
        if s.upper().startswith("CREATE"):
            continue
        m = re.fullmatch(r"GRANT\s+([\w\s,]+?)\s+ON\s+(\S+)\s+TO\s+(\w+)", s, re.I)
        assert m, s
        for priv in m.group(1).split(","):
            out.add((m.group(3), priv.strip().upper(), m.group(2)))
    return out


def test_roles_file_is_not_applied_and_grants_exactly_the_contract():
    assert "chlogins-roles.sql" not in {p.name for p in CH_MGMT_DIR.glob("*.sql")}
    tables = ("ch_mgmt.login_requests", "ch_mgmt.login_events")
    want = {
        ("dashboard_chlogins_writer", p, t)
        for p in ("SELECT", "INSERT")
        for t in tables
    }
    want |= {("dashboard_chlogins_reader", "SELECT", t) for t in tables}
    assert _grants() == want
    stmts = SchemaApplier.statements(
        (CH_MGMT_DIR / "roles" / "chlogins-roles.sql").read_text()
    )
    assert {s.split()[0] for s in stmts} == {"CREATE", "GRANT"}


def test_the_latest_version_of_a_login_wins_whatever_the_insert_order(chdb_session):
    s = chdb_session
    _insert(
        s,
        "login_requests",
        _request(version="2026-11-26 11:00:00.000", status="active", approver="bob"),
    )
    _insert(s, "login_requests", _request())
    got = _rows(
        s, "SELECT login, status, approver FROM login_requests FINAL ORDER BY login"
    )
    assert got == [{"login": LOGIN, "status": "active", "approver": "bob"}]


def test_never_expires_is_null_and_host_defaults_to_any(chdb_session):
    s = chdb_session
    _insert(s, "login_requests", _request(expires_at=None))
    got = _rows(s, "SELECT expires_at IS NULL AS never, host FROM login_requests")
    assert got == [{"never": 1, "host": "ANY"}]


def test_status_and_tier_accept_exactly_the_contract_values(chdb_session):
    s = chdb_session
    for n, status in enumerate(("pending", "active", "revoked", "deleted", "rejected")):
        _insert(s, "login_requests", _request(login=f"app_{n}", status=status))
    for n, tier in enumerate((1, 2)):
        _insert(s, "login_requests", _request(login=f"app_t{n}", tier=tier))
    for bad in ({"status": "expired"}, {"status": ""}, {"tier": 0}, {"tier": 3}):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            _insert(s, "login_requests", _request(login="app_bad", **bad))


def test_an_event_needs_only_login_actor_and_action(chdb_session):
    s = chdb_session
    _insert(s, "login_events", {"login": LOGIN, "actor": "alice", "action": "request"})
    got = _rows(
        s,
        "SELECT detail, reason, event_id != toUUIDOrZero('') AS has_id FROM login_events",
    )
    assert got == [{"detail": "", "reason": "", "has_id": 1}]
