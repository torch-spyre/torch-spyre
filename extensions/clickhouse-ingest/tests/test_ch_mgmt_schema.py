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
    assert not {p.name for p in CH_MGMT_DIR.glob("*.sql")} & {
        p.name for p in SCHEMA_DIR.glob("*.sql")
    }


def test_roles_file_is_not_applied_and_grants_no_alter_or_delete():
    roles = (CH_MGMT_DIR / "roles" / "chlogins-roles.sql").read_text()
    assert "chlogins-roles.sql" not in {p.name for p in CH_MGMT_DIR.glob("*.sql")}
    stmts = SchemaApplier.statements(roles)
    assert stmts and all(s.split()[0] in ("CREATE", "GRANT") for s in stmts)
    assert not any(w in s.upper() for s in stmts for w in ("ALTER", "DELETE", "DROP"))


def test_the_latest_version_of_a_login_wins(chdb_session):
    s = chdb_session
    _insert(s, "login_requests", _request())
    _insert(
        s,
        "login_requests",
        _request(version="2026-11-26 11:00:00.000", status="active", approver="bob"),
    )
    got = _rows(
        s, "SELECT login, status, approver FROM login_requests FINAL ORDER BY login"
    )
    assert got == [{"login": LOGIN, "status": "active", "approver": "bob"}]


def test_never_expires_is_null_and_host_defaults_to_any(chdb_session):
    s = chdb_session
    _insert(s, "login_requests", _request(expires_at=None))
    got = _rows(s, "SELECT expires_at IS NULL AS never, host FROM login_requests")
    assert got == [{"never": 1, "host": "ANY"}]


def test_a_request_with_an_unknown_status_or_tier_is_refused(chdb_session):
    for bad in ({"status": "expired"}, {"tier": 3}):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            _insert(chdb_session, "login_requests", _request(**bad))


def test_events_are_appended_in_login_order(chdb_session):
    s = chdb_session
    for i, action in enumerate(("request", "approve", "claim")):
        _insert(
            s,
            "login_events",
            {
                "login": LOGIN,
                "ts": f"2026-11-26 10:0{i}:00.000",
                "actor": "alice@ibm.com",
                "action": action,
            },
        )
    got = _rows(s, "SELECT action, detail, reason FROM login_events ORDER BY ts")
    assert [r["action"] for r in got] == ["request", "approve", "claim"]
    assert {(r["detail"], r["reason"]) for r in got} == {("", "")}
