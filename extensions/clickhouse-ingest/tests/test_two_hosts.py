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

"""Legacy host for v1 tables, v2 host for the Replicated databases: routing, credentials, spool."""

import clickhouse_connect
import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import client, writes
from spyre_clickhouse_ingest.batching import BatchParked, insert_chunks
from spyre_clickhouse_ingest.resilience import Spool

ENV = {
    "CLICKHOUSE_HOST": "legacy.example",
    "CLICKHOUSE_USER": "writer",
    "CLICKHOUSE_PASS": "w-pw",
    "CLICKHOUSE_DB": "spyre",
    "CLICKHOUSE_DB_V2": "spyre_v2",
    "CLICKHOUSE_V2_HOST": "ha.example",
}


class Backend:
    """Records every statement; answers the engine lookup like the host's own tables."""

    def __init__(self, host, database, engine, script=()):
        self.ch_host, self.database, self.engine = host, database, engine
        self.script, self.calls = list(script), []

    def insert(self, table, data, column_names=None, database=None, settings=None):
        self.calls.append(("insert", table, database, dict(settings or {})))
        if self.script:
            raise self.script.pop(0)

    def command(self, cmd, *a, **k):
        self.calls.append(("command", cmd))
        return 1

    def query(self, q=None, *a, **k):
        self.calls.append(("query", q))

        class R:
            result_rows = [("x", 0)]

        R.result_rows = (
            [(1000,)]  # replicated_deduplication_window
            if "merge_tree_settings" in q
            else [(self.engine, 0)]
            if "system.tables" in q
            else [(0,)]
        )
        return R()


@pytest.fixture
def env(monkeypatch):
    for name in list(ENV) + [
        "CLICKHOUSE_V2_PORT", "CLICKHOUSE_V2_SECURE", "CLICKHOUSE_V2_USER",
        "CLICKHOUSE_V2_PASSWORD", "CLICKHOUSE_V2_SCHEMA_USER", "CLICKHOUSE_V2_SCHEMA_PASSWORD",
        "CLICKHOUSE_SCHEMA_USER", "CLICKHOUSE_SCHEMA_PASSWORD", "CH_V2_DATABASES",
        "CLICKHOUSE_PORT", "CLICKHOUSE_SECURE", "CH_WRITE_SPOOL", "CH_WRITE_DEADLINE_S",
    ]:  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(client.ClickHouse, "_warned_schema_fallback", True)
    opened = []

    def fake_get_client(**kw):
        opened.append(kw)
        return Backend(kw["host"], kw["database"], "MergeTree")

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    return opened


def test_without_a_v2_host_there_is_one_plain_client(env, monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_V2_HOST")
    c = client.ClickHouse.connect()
    assert isinstance(c, client.WriterClient) and len(env) == 1
    assert env[0]["host"] == "legacy.example" and env[0]["database"] == "spyre"


def test_v2_client_is_opened_lazily_and_never_defaults_to_the_v1_database(env):
    c = client.ClickHouse.connect()
    assert isinstance(c, client.RoutedClient) and len(env) == 1
    c.command("SELECT 1")
    c.insert("test_runs", [[1]], column_names=["a"])
    assert len(env) == 1  # a v1-only run never touches the v2 host
    c.command("EXISTS TABLE spyre_v2.test_case_runs")
    assert len(env) == 2
    assert (env[1]["host"], env[1]["database"]) == ("ha.example", "spyre_v2")


def test_statements_route_by_the_database_they_name(env):
    c = client.ClickHouse.connect()
    legacy, v2 = c.legacy, c.v2
    c.insert("test_runs", [[1]], column_names=["a"])
    c.insert("test_case_runs", [[1]], column_names=["a"], database="spyre_v2")
    c.insert("vigil_ops.events", [[1]], column_names=["a"])
    c.insert("hw_failure_diagnostics", [[1]], column_names=["a"], database="spyre")
    c.command("EXISTS TABLE spyre_v2.t")
    c.command("DELETE FROM optimize_ops.t WHERE 1", parameters={})
    c.query(
        "SELECT name FROM system.columns WHERE database = {db:String}",
        parameters={"db": "spyre_kb"},
    )
    c.command("SELECT version()")
    c.query("SELECT count() FROM test_runs WHERE run_id = 1")
    c.command("EXISTS TABLE spyre.not_spyre_v2_table")

    def kinds(b):
        return [x[1] for x in b.calls if x[0] == "insert"]

    assert kinds(legacy) == ["test_runs", "hw_failure_diagnostics"]
    assert kinds(v2) == ["test_case_runs", "vigil_ops.events"]
    sql_v2 = [x[1] for x in v2.calls if x[0] != "insert"]
    assert sql_v2 == [
        "EXISTS TABLE spyre_v2.t",
        "DELETE FROM optimize_ops.t WHERE 1",
        "SELECT name FROM system.columns WHERE database = {db:String}",
    ]
    assert "SELECT version()" in [x[1] for x in legacy.calls]


def test_a_connection_to_a_replicated_database_goes_straight_to_the_v2_host(env):
    c = client.ClickHouse.connect(database="spyre_v2")
    assert isinstance(c, client.WriterClient) and c.ch_host == "ha.example"
    assert len(env) == 1 and env[0]["database"] == "spyre_v2"


def test_the_replicated_database_list_is_configurable(env, monkeypatch):
    monkeypatch.setenv("CH_V2_DATABASES", "mydb, other")
    monkeypatch.delenv("CLICKHOUSE_DB_V2")
    c = client.ClickHouse.connect(database="mydb")
    assert c.ch_host == "ha.example"
    d = client.ClickHouse.connect()
    d.command("EXISTS TABLE spyre_v2.t")
    assert d.legacy.calls  # no longer on the list: legacy host


def test_v2_port_secure_and_login_fall_back_to_the_legacy_settings(env, monkeypatch):
    client.ClickHouse.connect(database="spyre_v2")
    assert env[0]["port"] == 443 and env[0]["secure"] is True
    assert (env[0]["user"], env[0]["password"]) == ("writer", "w-pw")
    monkeypatch.setenv("CLICKHOUSE_V2_PORT", "8443")
    monkeypatch.setenv("CLICKHOUSE_V2_SECURE", "0")
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "ha-writer")
    monkeypatch.setenv("CLICKHOUSE_V2_PASSWORD", "ha-pw")
    client.ClickHouse.connect(database="spyre_v2")
    assert (env[1]["port"], env[1]["secure"]) == (8443, False)
    assert (env[1]["user"], env[1]["password"]) == ("ha-writer", "ha-pw")
    client.ClickHouse.connect()  # the legacy leg keeps the writer pair
    assert (env[2]["user"], env[2]["host"]) == ("writer", "legacy.example")


def test_half_a_v2_login_is_an_error(env, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "ha-writer")
    with pytest.raises(SystemExit, match="CLICKHOUSE_V2_PASSWORD"):
        client.ClickHouse.connect(database="spyre_v2")


def test_schema_role_reaches_the_v2_host_with_its_own_login(env, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_USER", "ddl")
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_PASSWORD", "ddl-pw")
    c = client.ClickHouse.connect(database="spyre_v2", role="schema")
    assert c.ch_host == "ha.example" and not isinstance(c, client.WriterClient)
    assert (env[0]["user"], env[0]["password"]) == ("ddl", "ddl-pw")
    monkeypatch.setenv("CLICKHOUSE_V2_SCHEMA_USER", "ha-ddl")
    monkeypatch.setenv("CLICKHOUSE_V2_SCHEMA_PASSWORD", "ha-ddl-pw")
    client.ClickHouse.connect(database="spyre_v2", role="schema")
    assert (env[1]["user"], env[1]["password"]) == ("ha-ddl", "ha-ddl-pw")
    r = client.ClickHouse.connect(role="schema")
    r.command("CREATE TABLE spyre_v2.t (a Int8) ENGINE = Memory")
    r.command("ALTER TABLE test_runs ADD COLUMN c Int8")
    assert r.v2.calls[-1][1].startswith("CREATE") and r.legacy.calls[-1][1].startswith(
        "ALTER"
    )


def test_a_writer_on_the_v2_host_still_refuses_ddl(env):
    r = client.ClickHouse.connect()
    with pytest.raises(client.SchemaStatementOnWriter):
        r.command("DROP TABLE spyre_v2.t")
    assert r._v2 is None or not any(x[0] == "command" for x in r._v2.calls)


# ── engine detection, tokens and the spool are per client ───────────────────────


@pytest.fixture
def routed(monkeypatch, tmp_path):
    monkeypatch.setattr(writes, "_spools", {})
    monkeypatch.setattr(writes, "_parked", [])
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0")
    legacy = Backend("legacy.example", "spyre", "MergeTree")
    v2 = Backend("ha.example", "spyre_v2", "ReplicatedMergeTree")
    return (
        client.RoutedClient(legacy, lambda: v2, client.ClickHouseEnv.V2_DATABASES),
        legacy,
        v2,
    )


def inserts(b):
    return [c for c in b.calls if c[0] == "insert"]


def test_only_the_v2_table_gets_a_token_and_each_host_is_asked_for_its_engine(routed):
    c, legacy, v2 = routed
    insert_chunks(c, "test_runs", [[1]], ["a"])
    insert_chunks(c, "test_case_runs", [[1]], ["a"], database="spyre_v2")
    insert_chunks(c, "spyre_v2.artifacts", [[2]], ["a"])
    assert [i[3] for i in inserts(legacy)] == [{}]
    assert (
        all("insert_deduplication_token" in i[3] for i in inserts(v2))
        and len(inserts(v2)) == 2
    )
    assert [i[1] for i in inserts(v2)] == ["test_case_runs", "artifacts"]
    assert (
        inserts(v2)[1][2] == "spyre_v2"
    )  # qualified name is split, not sent as a table

    def engine_asks(b):
        return [
            x for x in b.calls
            if x[0] == "query" and "dependencies_table" in x[1]
        ]  # fmt: skip

    assert len(engine_asks(legacy)) == 1 and len(engine_asks(v2)) == 2


def test_a_parked_batch_remembers_which_host_it_was_for(routed, monkeypatch, tmp_path):
    c, legacy, v2 = routed
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    v2.script = [DatabaseError("Code: 285. x", code=285, name="X")] * 5
    with pytest.raises(BatchParked):
        insert_chunks(c, "test_case_runs", [[1]], ["a"], database="spyre_v2")
    (item,) = Spool(tmp_path / "sp").items()
    assert item["host"] == "ha.example" and item["table"] == "test_case_runs"
    assert inserts(legacy) == []


def test_the_probe_runs_against_the_v2_host(env, monkeypatch):
    from spyre_clickhouse_ingest import probe

    seen = {}

    def fake_probe(c, table, **kw):
        seen["host"], seen["table"] = c.ch_host, table
        return {"ok": True}

    monkeypatch.setattr(probe, "probe", fake_probe)
    assert probe.main([]) == 0
    assert seen == {"host": "ha.example", "table": "spyre_v2.zz_write_probe"}
