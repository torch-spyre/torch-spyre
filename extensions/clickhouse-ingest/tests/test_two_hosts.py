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

"""Legacy host for v1 tables, optional v2 host for the v2 databases: connect and routing."""

import threading

import clickhouse_connect
import pytest

from spyre_clickhouse_ingest import client

DBS = ("spyre_v2", "ops_db", "kb_db")
ENV = {
    "CLICKHOUSE_HOST": "legacy.example.com",
    "CLICKHOUSE_USER": "writer",
    "CLICKHOUSE_PASS": "w-pw",
    "CLICKHOUSE_DB": "spyre",
    "CLICKHOUSE_DB_V2": "spyre_v2",
    "CH_V2_DATABASES": ",".join(DBS),
    "CLICKHOUSE_V2_HOST": "v2.example.com",
}
CLEARED = (
    "CLICKHOUSE_PORT", "CLICKHOUSE_SECURE", "CLICKHOUSE_V2_PORT", "CLICKHOUSE_V2_SECURE",
    "CLICKHOUSE_V2_USER", "CLICKHOUSE_V2_PASSWORD", "CLICKHOUSE_V2_SCHEMA_USER",
    "CLICKHOUSE_V2_SCHEMA_PASSWORD", "CLICKHOUSE_SCHEMA_USER", "CLICKHOUSE_SCHEMA_PASSWORD",
    "CH_REQUIRE_V2_HOST",
)  # fmt: skip


class Rec:
    def __init__(self, host="", database=""):
        self.host, self.database, self.calls, self.settings = host, database, [], {}

    def __getattr__(self, attr):
        def call(*a, **k):
            self.calls.append((attr, a[0] if a else None, k))
            return attr

        return call

    def set_client_setting(self, key, value):
        self.settings[key] = value


@pytest.fixture
def env(monkeypatch):
    for name in CLEARED:
        monkeypatch.delenv(name, raising=False)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(client.ClickHouse, "_warned_schema_fallback", True)
    opened = []

    def fake_get_client(**kw):
        opened.append(kw)
        return Rec(kw["host"], kw["database"])

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    return opened


@pytest.fixture
def routed():
    legacy, v2 = Rec("legacy"), Rec("v2")
    return client.RoutedClient(legacy, lambda: v2, DBS), legacy, v2


def backend(c):
    return c._inner if isinstance(c, client.WriterClient) else c


# ── connect ─────────────────────────────────────────────────────────────────────


def test_without_a_v2_host_there_is_one_plain_client(env, monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_V2_HOST")
    for db in ("", "spyre_v2"):
        c = client.ClickHouse.connect(database=db)
        assert isinstance(c, client.WriterClient)
    assert [o["host"] for o in env] == ["legacy.example.com"] * 2
    assert [o["database"] for o in env] == ["spyre", "spyre_v2"]


def test_v2_client_is_opened_lazily_and_never_defaults_to_the_v1_database(env):
    c = client.ClickHouse.connect()
    assert isinstance(c, client.RoutedClient) and len(env) == 1
    c.command("SELECT 1")
    c.insert("test_runs", [[1]], column_names=["a"])
    assert len(env) == 1  # a v1-only run never touches the v2 host
    c.command("EXISTS TABLE spyre_v2.test_case_runs")
    assert (env[1]["host"], env[1]["database"]) == ("v2.example.com", "spyre_v2")


def test_statements_route_by_the_database_they_name(env):
    c = client.ClickHouse.connect()
    legacy, v2 = backend(c.legacy), backend(c.v2)
    c.insert("test_runs", [[1]], column_names=["a"])
    c.insert("test_case_runs", [[1]], column_names=["a"], database="spyre_v2")
    c.insert("ops_db.events", [[1]], column_names=["a"])
    c.insert("hw_failure_diagnostics", [[1]], column_names=["a"], database="spyre")
    c.command("EXISTS TABLE spyre_v2.t")
    c.command("DELETE FROM ops_db.t WHERE 1", parameters={})
    c.query(
        "SELECT name FROM system.columns WHERE database = {db:String}",
        parameters={"db": "kb_db"},
    )
    c.command("SELECT version()")
    c.query("SELECT count() FROM test_runs WHERE run_id = 1")
    c.command("EXISTS TABLE spyre.not_spyre_v2_table")

    def inserts(b):
        return [x[1] for x in b.calls if x[0] == "insert"]

    assert inserts(legacy) == ["test_runs", "hw_failure_diagnostics"]
    assert inserts(v2) == ["test_case_runs", "ops_db.events"]
    assert [x[1] for x in v2.calls if x[0] != "insert"] == [
        "EXISTS TABLE spyre_v2.t",
        "DELETE FROM ops_db.t WHERE 1",
        "SELECT name FROM system.columns WHERE database = {db:String}",
    ]


def test_a_connection_to_a_v2_database_goes_straight_to_the_v2_host(env):
    c = client.ClickHouse.connect(database="spyre_v2")
    assert isinstance(c, client.WriterClient)
    assert [(o["host"], o["database"]) for o in env] == [("v2.example.com", "spyre_v2")]


def test_the_v2_database_list_is_configurable(env, monkeypatch):
    monkeypatch.setenv("CH_V2_DATABASES", "mydb, other")
    monkeypatch.delenv("CLICKHOUSE_DB_V2")
    client.ClickHouse.connect(database="mydb")
    assert env[-1]["host"] == "v2.example.com"
    d = client.ClickHouse.connect()
    d.command("EXISTS TABLE spyre_v2.t")
    assert backend(d.legacy).calls  # no longer listed: legacy host


def test_default_v2_databases_and_an_unlisted_target_database(monkeypatch):
    monkeypatch.delenv("CH_V2_DATABASES", raising=False)
    monkeypatch.delenv("CLICKHOUSE_DB_V2", raising=False)
    assert client.ClickHouseEnv.v2_databases() == ("spyre_v2",)
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2_dev")
    assert "spyre_v2_dev" in client.ClickHouseEnv.v2_databases()
    assert client.ClickHouseEnv.v2_default_database() == "spyre_v2_dev"


def test_v2_port_secure_and_login_fall_back_to_the_legacy_settings(env, monkeypatch):
    client.ClickHouse.connect(database="spyre_v2")
    assert env[0]["port"] == 443 and env[0]["secure"] is True
    assert (env[0]["user"], env[0]["password"]) == ("writer", "w-pw")
    monkeypatch.setenv("CLICKHOUSE_V2_PORT", "8443")
    monkeypatch.setenv("CLICKHOUSE_V2_SECURE", "0")
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "v2-writer")
    monkeypatch.setenv("CLICKHOUSE_V2_PASSWORD", "v2-pw")
    client.ClickHouse.connect(database="spyre_v2")
    assert (env[1]["port"], env[1]["secure"]) == (8443, False)
    assert (env[1]["user"], env[1]["password"]) == ("v2-writer", "v2-pw")
    client.ClickHouse.connect()  # the legacy leg keeps the writer pair
    assert (env[2]["user"], env[2]["host"]) == ("writer", "legacy.example.com")


def test_half_a_v2_login_is_an_error(env, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "v2-writer")
    with pytest.raises(SystemExit, match="CLICKHOUSE_V2_PASSWORD"):
        client.ClickHouse.connect(database="spyre_v2")


def test_schema_role_reaches_the_v2_host_with_its_own_login(env, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_USER", "ddl")
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_PASSWORD", "ddl-pw")
    c = client.ClickHouse.connect(database="spyre_v2", role="schema")
    assert not isinstance(c, client.WriterClient) and c.host == "v2.example.com"
    assert (env[0]["user"], env[0]["password"]) == ("ddl", "ddl-pw")
    monkeypatch.setenv("CLICKHOUSE_V2_SCHEMA_USER", "v2-ddl")
    monkeypatch.setenv("CLICKHOUSE_V2_SCHEMA_PASSWORD", "v2-ddl-pw")
    client.ClickHouse.connect(database="spyre_v2", role="schema")
    assert (env[1]["user"], env[1]["password"]) == ("v2-ddl", "v2-ddl-pw")
    r = client.ClickHouse.connect(role="schema")
    r.command("CREATE TABLE spyre_v2.t (a Int8) ENGINE = Memory")
    r.command("ALTER TABLE test_runs ADD COLUMN c Int8")
    assert r.v2.calls[-1][1].startswith("CREATE")
    assert r.legacy.calls[-1][1].startswith("ALTER")


def test_v2_schema_role_prefers_the_v2_writer_login_over_the_legacy_one(
    env, monkeypatch
):
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "v2-w")
    monkeypatch.setenv("CLICKHOUSE_V2_PASSWORD", "v2-p")
    assert client.ClickHouse.credentials("schema", v2=True) == ("v2-w", "v2-p")
    assert client.ClickHouse.credentials("schema", v2=False) == ("writer", "w-pw")


def test_a_writer_on_the_v2_host_still_refuses_ddl(env):
    r = client.ClickHouse.connect()
    with pytest.raises(client.SchemaStatementOnWriter):
        r.command("DROP TABLE spyre_v2.t")
    assert r._v2 is None or not backend(r._v2).calls


def test_require_flag_refuses_a_v2_target_without_a_v2_host(env, monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_V2_HOST")
    monkeypatch.setenv("CH_REQUIRE_V2_HOST", "1")
    with pytest.raises(SystemExit, match="CLICKHOUSE_V2_HOST"):
        client.ClickHouse.connect()
    monkeypatch.delenv("CLICKHOUSE_DB_V2")
    client.ClickHouse.connect()  # nothing v2 configured: the legacy host is fine
    with pytest.raises(SystemExit, match="CLICKHOUSE_V2_HOST"):
        client.ClickHouse.connect(database="kb_db")


@pytest.mark.parametrize("flag", ["FALSE", " No ", "0", ""])
def test_require_flag_off_values(env, monkeypatch, flag):
    monkeypatch.delenv("CLICKHOUSE_V2_HOST")
    monkeypatch.setenv("CH_REQUIRE_V2_HOST", flag)
    assert isinstance(client.ClickHouse.connect(), client.WriterClient)


# ── routing: tokenised, not substring-matched ───────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        'INSERT INTO "spyre_v2"."t" VALUES (1)',
        "INSERT INTO spyre_v2 . t VALUES (1)",
        "SHOW TABLES FROM spyre_v2",
        "USE spyre_v2",
        "SELECT name FROM system.tables WHERE database = 'spyre_v2'",
        "SELECT 1 FROM ops_db.events /* spyre.x */",
        "SELECT r.tags FROM spyre_v2.runs AS r ARRAY JOIN r.tags AS tg",
        "SELECT r.tags FROM spyre_v2.runs AS r LEFT ARRAY JOIN r.tags AS tg",
        "SELECT x FROM spyre_v2.a, spyre_v2.b",
        "SELECT a FROM (SELECT a, t.col FROM spyre_v2.x AS t) WHERE a = 1",
        "INSERT INTO spyre_v2.y SELECT a, t.col, u.z FROM spyre_v2.x AS t "
        "JOIN spyre_v2.w AS u ON 1",
        "SELECT f(a, spyre.b) FROM spyre_v2.t",
        "SELECT * FROM system.tables WHERE database = 'kb_db'",
        "SELECT 1 FROM system.columns WHERE table_schema IN ('ops_db')",
        "SELECT * FROM system.tables WHERE 'spyre_v2' = database",
        "SELECT * FROM system.tables WHERE database LIKE 'spyre_v2'",
        "SELECT * FROM merge(spyre_v2, '^t')",
        "SELECT * FROM merge('spyre_v2', '^t')",
        "SELECT * FROM cluster('c', spyre_v2.t)",
    ],
)
def test_statements_naming_a_v2_database_go_to_v2(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is v2


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 -- spyre_v2.t",
        "SELECT 1 # spyre_v2.t",
        "SELECT /* spyre_v2.t */ 1",
        "SELECT 'see spyre_v2.t for details'",
        "SELECT * FROM test_runs WHERE note = 'my spyre_v2 note'",
        "SELECT * FROM spyre.test_runs",
        "SELECT spyre_v2_like FROM t",
        "INSERT INTO SPYRE_V2.t VALUES (1)",  # names are case-sensitive in ClickHouse
        "INSERT INTO `Spyre_V2`.`t` VALUES (1)",
        "SELECT * FROM spyre.test_runs WHERE name = 'kb_db'",
    ],
)
def test_mentions_in_comments_literals_and_lookalikes_stay_on_legacy(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is legacy


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO spyre.t SELECT a FROM spyre_v2.u",
        "INSERT INTO spyre_v2.t SELECT a FROM spyre.u",
        "SELECT * FROM spyre_v2.t1, spyre.t2",
        "SELECT * FROM spyre.t2, spyre_v2.t1",
        "CREATE TABLE spyre_v2.a AS spyre.b",
        "SELECT * FROM spyre_v2.a JOIN spyre.b ON 1",
        "SELECT * FROM (SELECT a FROM spyre_v2.t) q, spyre.u",
        "SELECT * FROM spyre_v2.t, (SELECT a FROM spyre.u) q",
        "SELECT * FROM (SELECT 1) q, spyre.u WHERE x IN (SELECT a FROM spyre_v2.t)",
        "SELECT * FROM spyre_v2.a WHERE k IN spyre.b",
        "SELECT * FROM spyre_v2.a WHERE k GLOBAL IN spyre.b",
        "EXCHANGE TABLES spyre_v2.a AND spyre.b",
        "RENAME TABLE spyre.a TO spyre_v2.a",
        "SELECT * FROM remote('h', spyre.b) JOIN spyre_v2.a USING k",
    ],
)
def test_a_statement_spanning_both_hosts_is_refused(routed, sql):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement):
        c.route(sql=sql)


def test_same_host_copies_are_not_cross_host(routed):
    c, legacy, v2 = routed
    assert c.route(sql="INSERT INTO spyre_v2.t SELECT a FROM spyre_v2.u") is v2


def test_table_arguments_with_quotes_route(routed):
    c, legacy, v2 = routed
    assert c.route(table="`spyre_v2`.t") is v2
    assert c.route(table='"ops_db"."t"') is v2
    assert c.route(table="test_runs") is legacy


def test_other_statement_methods_are_routed_not_forwarded_to_legacy(routed):
    c, legacy, v2 = routed
    c.raw_query("SELECT 1 FROM spyre_v2.t")
    c.query_df("SELECT 1 FROM ops_db.t")
    c.insert_df("kb_db.t", None)
    c.insert_df("test_runs", None)
    c.raw_query(query="SELECT 1")
    assert [x[0] for x in v2.calls] == ["raw_query", "query_df", "insert_df"]
    assert [x[0] for x in legacy.calls] == ["insert_df", "raw_query"]


def test_settings_set_before_the_v2_client_opens_are_replayed(routed):
    c, legacy, v2 = routed
    c.set_client_setting("readonly", "2")
    assert legacy.settings == {"readonly": "2"} and v2.settings == {}
    c.command("EXISTS TABLE spyre_v2.t")  # opens v2
    assert v2.settings == {"readonly": "2"}
    c.set_client_setting("x", "1")
    assert v2.settings["x"] == "1"


def test_concurrent_first_use_opens_one_v2_client():
    opened = []

    def factory():
        opened.append(1)
        return Rec("v2")

    c = client.RoutedClient(Rec("legacy"), factory, DBS)
    threads = [threading.Thread(target=lambda: c.v2) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(opened) == 1


def test_close_reaches_v2_even_when_legacy_close_fails(routed):
    c, legacy, v2 = routed
    c.v2  # open it

    def boom(*a, **k):
        raise RuntimeError("down")

    legacy.close = boom
    with pytest.raises(RuntimeError):
        c.close()
    assert [x[0] for x in v2.calls] == ["close"]


@pytest.mark.parametrize(
    "params,sql",
    [
        (
            {"v2db": "spyre_v2"},
            "SELECT * FROM system.tables WHERE database = {v2db:String}",
        ),
        ({"x": "spyre_v2"}, "SELECT count() FROM {x:Identifier}.t"),
        (
            {"target_database": "ops_db"},
            "SELECT 1 WHERE database = {target_database:String}",
        ),
    ],
)
def test_database_like_parameters_route_to_v2(routed, params, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql, parameters=params) is v2


def test_a_data_parameter_equal_to_a_database_name_stays_on_legacy(routed):
    c, legacy, v2 = routed
    sql = "SELECT * FROM test_runs WHERE note = {note:String}"
    assert c.route(sql=sql, parameters={"note": "spyre_v2"}) is legacy


# ── a bad v2 setting fails at connect, and a v2 host that fails later spares v1 ──


@pytest.mark.parametrize(
    "bad,var",
    [
        ({"CLICKHOUSE_V2_USER": "v2-writer"}, "CLICKHOUSE_V2_PASSWORD"),
        ({"CLICKHOUSE_V2_PORT": "eighty"}, "CLICKHOUSE_V2_PORT"),
        ({"CLICKHOUSE_V2_SCHEMA_PASSWORD": "x"}, "CLICKHOUSE_V2_SCHEMA_USER"),
    ],
)
def test_a_bad_v2_setting_fails_at_connect_before_anything_runs(
    env, monkeypatch, bad, var
):
    for k, v in bad.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(SystemExit, match=var):
        client.ClickHouse.connect(role="schema" if "SCHEMA" in var else "writer")
    assert env == []  # no connection was opened


def test_a_v2_host_that_fails_on_first_use_raises_a_normal_exception(env, monkeypatch):
    def fake_get_client(**kw):
        if kw["host"] == "v2.example.com":
            raise SystemExit("stand-in that exits")
        env.append(kw)
        return Rec(kw["host"], kw["database"])

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    c = client.ClickHouse.connect()
    with pytest.raises(client.V2Unavailable):
        c.command("EXISTS TABLE spyre_v2.t")
    assert c.command("SELECT 1") == "command"  # the legacy host still works


def test_the_v1_database_may_not_be_listed_as_a_v2_one(env, monkeypatch):
    monkeypatch.setenv("CH_V2_DATABASES", "spyre_v2,spyre")
    with pytest.raises(SystemExit, match="CLICKHOUSE_DB"):
        client.ClickHouse.connect()
    monkeypatch.setenv(
        "CLICKHOUSE_DB", "spyre_v2"
    )  # a v2-only process: its own db is v2
    client.ClickHouse.connect()
    assert env[-1]["host"] == "v2.example.com"


@pytest.mark.parametrize("raw", ["False", "OFF", " no ", "0"])
def test_v2_secure_off_values_are_case_insensitive(monkeypatch, raw):
    monkeypatch.setenv("CLICKHOUSE_V2_SECURE", raw)
    assert client.ClickHouseEnv.v2_secure() is False


def _results(tmp_path, monkeypatch, v2_get_client):
    from spyre_clickhouse_ingest import results
    from test_dry_run import RUN, WHEEL, XML, LiveClient

    for n in range(2):
        (tmp_path / f"junit-{n}.xml").write_text(XML.replace("test_", f"test{n}_"))
    legacy = LiveClient()

    def fake_get_client(**kw):
        return v2_get_client(kw) if kw["host"] == "v2.example.com" else legacy

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    results.main(
        ["--xml-dir", str(tmp_path), "--schema", "both", "--run-id", RUN,
         "--trigger-type", "regression", "--arch", "x86_64", "--artifact", WHEEL,
         "--repository", "torch-spyre/torch-spyre", "--sha", "ab12"]
    )  # fmt: skip
    return legacy


def test_a_v2_host_that_is_down_spares_every_files_v1_rows(env, tmp_path, monkeypatch):
    def down(kw):
        raise ConnectionRefusedError("v2 host down")

    legacy = _results(tmp_path, monkeypatch, down)
    v1 = [t for db, t, _ in legacy.inserts if db is None and t == "test_runs"]
    assert len(v1) == 2  # both files' v1 rows landed


def test_a_bad_v2_setting_stops_the_ingest_before_any_write(env, tmp_path, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "v2-writer")
    with pytest.raises(SystemExit, match="CLICKHOUSE_V2_PASSWORD"):
        _results(tmp_path, monkeypatch, lambda kw: None)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE VIEW spyre.v AS SELECT * FROM spyre_v2.t",
        "CREATE MATERIALIZED VIEW spyre_v2.mv TO spyre.t AS SELECT * FROM spyre_v2.s",
        "SELECT * FROM t JOIN spyre_v2.u USING k",
        "SELECT * FROM spyre_v2.u WHERE k IN (SELECT k FROM t)",
        "INSERT INTO spyre_v2.x SELECT * FROM test_runs",
        "SELECT * FROM spyre_v2.t /* /* nested */ */ JOIN spyre.u USING k",
        "SELECT * FROM clusterAllReplicas('c', 'spyre_v2.t') JOIN spyre.u USING k",
    ],
)
def test_views_mv_targets_unqualified_tables_and_nested_comments_cross_hosts(
    routed, sql
):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement):
        c.route(sql=sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM clusterAllReplicas('c', 'spyre_v2.t')",
        "SELECT * FROM spyre_v2.t /* /* */ JOIN spyre.u USING k */",
        "WITH x AS (SELECT 1) SELECT * FROM spyre_v2.a JOIN x USING k",
        "SELECT * FROM spyre_v2.a, numbers(3)",
        "SELECT $$ it's $$ FROM spyre_v2.t",
    ],
)
def test_string_table_arguments_ctes_and_heredocs_stay_on_v2(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is v2


def test_a_v2_parameter_in_a_statement_on_legacy_tables_is_refused(routed):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement):
        c.route(
            sql="SELECT * FROM spyre.t WHERE x = {db:String}",
            parameters={"db": "spyre_v2"},
        )


@pytest.mark.parametrize(
    "raw,secure", [("0", False), ("false", False), ("no", False), ("False", True), ("OFF", True),
                   ("off", True), ("No", True), ("1", True)],
)  # fmt: skip
def test_the_legacy_secure_setting_keeps_its_exact_values(monkeypatch, raw, secure):
    monkeypatch.setenv("CLICKHOUSE_SECURE", raw)
    monkeypatch.delenv("CLICKHOUSE_V2_SECURE", raising=False)
    assert client.ClickHouseEnv.secure() is secure
    assert client.ClickHouseEnv.v2_secure() is secure  # unset: the legacy value


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT extract(DAY FROM ts) FROM spyre_v2.t",
        "SELECT trim(BOTH ' ' FROM name) FROM spyre_v2.t",
        "SELECT a IS NOT DISTINCT FROM b FROM spyre_v2.t",
        "SELECT a IS DISTINCT FROM b FROM spyre_v2.t",
        "SELECT * FROM spyre_v2.t ORDER BY a WITH FILL FROM 1 TO 9",
        "SHOW COLUMNS FROM t FROM spyre_v2",
    ],
)
def test_from_keywords_that_are_not_a_from_list_stay_on_v2(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is v2


def test_a_comma_listed_unqualified_table_beside_a_v2_one_is_named(routed):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement, match="'t' unqualified"):
        c.route(sql="SELECT * FROM spyre_v2.a, t")


@pytest.mark.parametrize(
    "sql,kept",
    [
        ("SELECT 'a\\'b -- not a comment' FROM t", "'a\\'b -- not a comment'"),
        ('SELECT "x\\"y /* not */" FROM t', '"x\\"y /* not */"'),
    ],
)
def test_backslash_escapes_keep_a_quote_inside_its_literal(sql, kept):
    assert kept in client._strip_comments(sql)


def test_a_heredoc_hides_what_looks_like_a_legacy_table(routed):
    c, legacy, v2 = routed
    assert client._strip_comments("SELECT $$ FROM spyre.u $$ x") == "SELECT '' x"
    assert c.route(sql="SELECT $$ JOIN spyre.u $$ FROM spyre_v2.t") is v2
