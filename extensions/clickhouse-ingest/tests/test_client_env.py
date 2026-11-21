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

"""Pins the connection factory's handling of BLANK env vars.

GitHub Actions exports an unset secret as the empty string, not as an absent variable, so every
`os.environ.get(name, default)` default in this factory used to be unreachable and a missing
CLICKHOUSE_PORT surfaced as `int('')`.
"""

import clickhouse_connect
import pytest
from spyre_clickhouse_ingest import client

REQUIRED = {
    "CLICKHOUSE_HOST": "ch.example.com",
    "CLICKHOUSE_USER": "writer",
    "CLICKHOUSE_PASS": "secret",
}


@pytest.fixture
def captured(monkeypatch):
    """Capture the kwargs get_client would connect with."""
    seen = {}

    def fake_get_client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    for name in (
        "CLICKHOUSE_HOST",
        "CLICKHOUSE_PORT",
        "CLICKHOUSE_USER",
        "CLICKHOUSE_PASS",
        "CLICKHOUSE_PASSWORD",
        "CLICKHOUSE_SCHEMA_USER",
        "CLICKHOUSE_SCHEMA_PASSWORD",
        "CLICKHOUSE_DB",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    return seen


def test_blank_optional_vars_fall_back_to_the_documented_defaults(
    captured, monkeypatch
):
    for name in ("CLICKHOUSE_PORT", "CLICKHOUSE_DB"):
        monkeypatch.setenv(name, "")
    client.get_client()
    assert captured["port"] == 443
    assert captured["user"] == "writer"
    assert captured["database"] == "spyre"


def test_absent_optional_vars_use_the_same_defaults(captured):
    client.get_client()
    assert (captured["port"], captured["user"], captured["database"]) == (
        443,
        "writer",
        "spyre",
    )


def test_values_are_honoured_and_stripped(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_PORT", " 8443 ")
    monkeypatch.setenv("CLICKHOUSE_DB", " other ")
    client.get_client()
    assert captured["port"] == 8443
    assert captured["database"] == "other"


def test_verify_defaults_true_and_can_be_relaxed(captured):
    client.get_client()
    assert captured["verify"] is True
    captured.clear()
    client.get_client(verify=False)
    assert captured["verify"] is False


@pytest.mark.parametrize(
    "name", ["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASS"]
)
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_required_var_names_itself(captured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=name):
        client.get_client()


def test_non_numeric_port_says_so(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_PORT", "not-a-port")
    with pytest.raises(SystemExit, match="CLICKHOUSE_PORT"):
        client.get_client()


def test_client_summary_reports_the_resolved_values(captured, monkeypatch):
    # Read through the same resolver as the connection, so a log banner cannot claim a port the
    # client did not use.
    monkeypatch.setenv("CLICKHOUSE_PORT", "")
    assert client.client_summary() == "ch.example.com:443/spyre"


# ── credential roles ────────────────────────────────────────────────────────────────────


def test_no_default_user_is_assumed_on_a_secure_connection(captured, monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_USER")
    with pytest.raises(SystemExit, match="CLICKHOUSE_USER"):
        client.get_client()


def test_an_insecure_local_box_may_omit_the_user(captured, monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_USER")
    monkeypatch.delenv("CLICKHOUSE_PASS")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "0")
    client.get_client()
    assert "user" not in captured


def test_password_alias_is_accepted(captured, monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_PASS")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "alias")
    client.get_client()
    assert captured["password"] == "alias"


def test_schema_role_uses_its_own_login(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_USER", "ddl")
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_PASSWORD", "ddl-pw")
    client.ClickHouse.connect(role="schema")
    assert (captured["user"], captured["password"]) == ("ddl", "ddl-pw")
    captured.clear()
    client.ClickHouse.connect()
    assert (captured["user"], captured["password"]) == ("writer", "secret")


def test_schema_role_falls_back_once_with_a_warning(captured, capsys, monkeypatch):
    monkeypatch.setattr(client.ClickHouse, "_warned_schema_fallback", False)
    client.ClickHouse.connect(role="schema")
    client.ClickHouse.connect(role="schema")
    assert (captured["user"], captured["password"]) == ("writer", "secret")
    assert capsys.readouterr().err.count("CLICKHOUSE_SCHEMA_USER") == 1


def test_schema_role_with_half_a_pair_names_the_missing_var(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_USER", "ddl")
    with pytest.raises(SystemExit, match="CLICKHOUSE_SCHEMA_PASSWORD"):
        client.ClickHouse.connect(role="schema")


class Recorder:
    database = "db"

    def __init__(self):
        self.seen = []

    def command(self, cmd, *a, **k):
        self.seen.append(cmd)
        return 1

    def query(self, q=None, *a, **k):
        self.seen.append(q)
        return "rows"

    def insert(self, *a, **k):
        self.seen.append("insert")


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t (a Int8) ENGINE = Memory",
        "  alter table t add column b Int8",
        "DROP VIEW IF EXISTS v",
        "TRUNCATE TABLE t",
        "RENAME TABLE a TO b",
        "-- note\nDROP TABLE t",
        "/* c */ CREATE OR REPLACE VIEW v AS SELECT 1",
        "GRANT SELECT ON db.* TO u",
    ],
)
def test_a_writer_connection_never_sends_schema_statements(sql):
    inner = Recorder()
    w = client.WriterClient(inner)
    for call in (w.command, w.query):
        with pytest.raises(client.SchemaStatementOnWriter):
            call(sql)
    assert inner.seen == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "EXISTS TABLE t",
        "DESCRIBE TABLE t",
        "DELETE FROM t WHERE a = 1",
        "INSERT INTO t SELECT * FROM s",
        "SELECT 'CREATE TABLE' AS x",
    ],
)
def test_a_writer_connection_passes_data_statements(sql):
    inner = Recorder()
    w = client.WriterClient(inner)
    w.command(sql)
    assert w.query(sql) == "rows"
    w.insert("t", [], column_names=[])
    assert w.database == "db"
    assert inner.seen == [sql, sql, "insert"]


def test_connect_hands_out_a_guarded_client_by_default(captured, monkeypatch):
    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **k: Recorder())
    assert isinstance(client.ClickHouse.connect(), client.WriterClient)
    assert not isinstance(client.ClickHouse.connect(role="schema"), client.WriterClient)
