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

"""Router tokenising, settings replay, replay of parked batches, attempt replacement, exit hook."""

import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import client, spool_cli, writes
from spyre_clickhouse_ingest.resilience import Spool, batch_token

ROOT = Path(__file__).resolve().parents[1]
DBS = ("spyre_v2", "vigil_ops", "spyre_kb", "optimize_ops")


class Rec:
    def __init__(self, name):
        self.name, self.calls, self.settings = name, [], {}

    def __getattr__(self, attr):
        def call(*a, **k):
            self.calls.append((attr, a, k))
            return attr

        return call

    def set_client_setting(self, key, value):
        self.settings[key] = value


@pytest.fixture
def routed():
    legacy, v2 = Rec("legacy"), Rec("v2")
    return client.RoutedClient(legacy, lambda: v2, DBS), legacy, v2


# ── router: tokenised, not substring-matched ────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        'INSERT INTO "spyre_v2"."t" VALUES (1)',
        "INSERT INTO spyre_v2 . t VALUES (1)",
        "INSERT INTO SPYRE_V2.t VALUES (1)",
        "INSERT INTO `Spyre_V2`.`t` VALUES (1)",
        "SHOW TABLES FROM spyre_v2",
        "USE spyre_v2",
        "SELECT name FROM system.tables WHERE database = 'spyre_v2'",
        "SELECT 1 FROM vigil_ops.events /* spyre.x */",
    ],
)
def test_statements_naming_a_replicated_database_go_to_v2(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is v2


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 -- spyre_v2.t",
        "SELECT 1 # spyre_v2.t",
        "SELECT /* spyre_v2.t */ 1",
        "SELECT 'see spyre_v2.t for details'",
        "SELECT * FROM test_runs WHERE note = 'spyre_v2'  AND x = 1 OR 'a' = 'b'".replace(
            "'spyre_v2'", "'my spyre_v2 note'"
        ),
        "SELECT * FROM spyre.test_runs",
        "SELECT spyre_v2_like FROM t",
    ],
)
def test_mentions_in_comments_literals_and_lookalikes_stay_on_legacy(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is legacy


def test_a_statement_spanning_both_hosts_is_refused(routed):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement):
        c.route(sql="INSERT INTO spyre.t SELECT a FROM spyre_v2.u")
    with pytest.raises(client.CrossHostStatement):
        c.route(sql="INSERT INTO spyre_v2.t SELECT a FROM spyre.u")
    # same-host copies and system reads are fine
    assert c.route(sql="INSERT INTO spyre_v2.t SELECT a FROM spyre_v2.u") is c.v2
    assert (
        c.route(sql="SELECT * FROM system.tables WHERE database = 'spyre_v2'") is c.v2
    )


def test_table_arguments_with_quotes_route(routed):
    c, legacy, v2 = routed
    assert c.route(table="`spyre_v2`.t") is v2
    assert c.route(table='"vigil_ops"."t"') is v2
    assert c.route(table="test_runs") is legacy


def test_other_statement_methods_are_routed_not_forwarded_to_legacy(routed):
    c, legacy, v2 = routed
    c.raw_query("SELECT 1 FROM spyre_v2.t")
    c.query_df("SELECT 1 FROM vigil_ops.t")
    c.insert_df("spyre_kb.t", None)
    c.insert_df("test_runs", None)
    c.raw_query(query="SELECT 1")
    assert [x[0] for x in v2.calls] == ["raw_query", "query_df", "insert_df"]
    assert [x[0] for x in legacy.calls] == ["insert_df", "raw_query"]


def test_an_unlisted_target_database_is_v2_by_definition(monkeypatch):
    monkeypatch.setenv("CH_V2_DATABASES", "")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2_dev")
    assert "spyre_v2_dev" in client.ClickHouseEnv.v2_databases()
    assert client.ClickHouseEnv.v2_default_database() == "spyre_v2_dev"


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


# ── credentials and the opt-in guard ────────────────────────────────────────────


def test_v2_schema_role_prefers_the_ha_logins_before_legacy_writer(monkeypatch):
    for k in (
        "CLICKHOUSE_SCHEMA_USER",
        "CLICKHOUSE_SCHEMA_PASSWORD",
        "CLICKHOUSE_V2_SCHEMA_USER",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CLICKHOUSE_USER", "legacy-w")
    monkeypatch.setenv("CLICKHOUSE_PASS", "p")
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "ha-w")
    monkeypatch.setenv("CLICKHOUSE_V2_PASSWORD", "hp")
    monkeypatch.setattr(client.ClickHouse, "_warned_schema_fallback", True)
    assert client.ClickHouse.credentials("schema", v2=True)[:2] == ("ha-w", "hp")
    assert client.ClickHouse.credentials("schema", v2=False)[:2] == ("legacy-w", "p")


def test_require_flag_refuses_a_replicated_target_without_a_v2_host(monkeypatch):
    for k in ("CLICKHOUSE_V2_HOST",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CLICKHOUSE_HOST", "legacy")
    monkeypatch.setenv("CLICKHOUSE_USER", "w")
    monkeypatch.setenv("CLICKHOUSE_PASS", "p")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setenv("CH_REQUIRE_V2_HOST", "1")
    with pytest.raises(SystemExit, match="CLICKHOUSE_V2_HOST"):
        client.ClickHouse.connect()
    monkeypatch.delenv("CH_REQUIRE_V2_HOST")
    import clickhouse_connect

    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **k: Rec("x"))
    client.ClickHouse.connect()  # single-host setups keep working


def test_a_comment_hash_does_not_hide_ddl_from_the_writer_guard():
    with pytest.raises(client.SchemaStatementOnWriter):
        client.WriterClient(Rec("x")).command("# note\nDROP TABLE t")


# ── replaying parked batches ────────────────────────────────────────────────────


class Cluster:
    database = "d"
    ch_host = "h"

    def __init__(self, engine="ReplicatedMergeTree"):
        self.engine, self.inserts, self.fail = engine, [], 0

    def query(self, sql, parameters=None):
        class R:
            pass

        R.result_rows = (
            [(1000,)] if "merge_tree_settings" in sql else [(self.engine, 0)]
        )
        return R()

    def insert(self, table, data, column_names=None, database=None, settings=None):
        if self.fail:
            self.fail -= 1
            raise DatabaseError("Code: 285. x", code=285, name="X")
        self.inserts.append((table, len(data), dict(settings or {})))


def parked(tmp_path, table="t", rows=3):
    spool = Spool(tmp_path / "sp")
    tok = batch_token(table, "scope", [[i] for i in range(rows)], 2)
    spool.put(
        table,
        ["a"],
        [[i] for i in range(rows)],
        tok,
        "d",
        scope="scope",
        ambiguous=True,
    )
    return spool, tok


def test_replay_resends_under_the_original_token_and_removes_the_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0")
    spool, tok = parked(tmp_path)
    c = Cluster()
    assert spool_cli.replay_items(spool, None, connect=lambda database="": c) == 0
    assert c.inserts[0][2]["insert_deduplication_token"] == tok
    assert spool.items() == []


def test_replay_is_idempotent_across_two_runs(tmp_path):
    spool, tok = parked(tmp_path)
    c = Cluster()
    spool_cli.replay_items(spool, tok, connect=lambda database="": c)
    spool2, _ = parked(tmp_path / "again")
    spool_cli.replay_items(spool2, tok, connect=lambda database="": c)
    tokens = {i[2]["insert_deduplication_token"] for i in c.inserts}
    assert tokens == {tok}  # same token both times: the server drops the second


def test_replay_keeps_the_file_when_the_batch_cannot_land(tmp_path, monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    spool, tok = parked(tmp_path)
    c = Cluster()
    c.fail = 9
    assert spool_cli.replay_items(spool, None, connect=lambda database="": c) == 1
    assert len(spool.items()) == 1


def test_replay_never_touches_a_plain_table(tmp_path):
    spool, tok = parked(tmp_path)
    c = Cluster(engine="MergeTree")
    assert spool_cli.replay_items(spool, None, connect=lambda database="": c) == 1
    assert c.inserts == [] and len(spool.items()) == 1


def test_list_never_prints_rows(tmp_path, monkeypatch, capsys):
    spool, tok = parked(tmp_path)
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    spool_cli.main(["list"])
    out = capsys.readouterr().out
    assert tok in out and '"rows": 3' in out and "[[0]" not in out


# ── exit status ─────────────────────────────────────────────────────────────────

PARK = textwrap.dedent(
    """
    import sys, threading
    from clickhouse_connect.driver.exceptions import DatabaseError
    from spyre_clickhouse_ingest import writes
    from spyre_clickhouse_ingest.batching import insert_chunks

    writes.install_exit_hook()

    class C:
        database = "d"
        def query(self, sql, parameters=None):
            class R: result_rows = [("ReplicatedMergeTree", 0)]
            return R()
        def insert(self, *a, **k):
            raise DatabaseError("Code: 285. x", code=285, name="X")

    try:
        insert_chunks(C(), "t", [[1]], ["a"])
    except Exception:
        pass
    {tail}
    """
)


def status(tmp_path, tail):
    env = {
        "PATH": "/usr/bin:/bin",
        "CH_WRITE_SPOOL": str(tmp_path / "sp"),
        "CH_WRITE_DEADLINE_S": "0.001",
        "PYTHONPATH": str(ROOT),
    }
    p = subprocess.run(
        [sys.executable, "-c", PARK.format(tail=tail)],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )  # fmt: skip
    return p.returncode, p.stderr


def test_string_exit_after_a_park_keeps_failing(tmp_path):
    rc, err = status(tmp_path, "sys.exit('boom')")
    assert rc == 1 and "boom" in err


def test_a_crashed_thread_after_a_park_still_exits_75(tmp_path):
    tail = "t = threading.Thread(target=lambda: 1/0); t.start(); t.join()"
    assert status(tmp_path, tail)[0] == 75


def test_system_exit_raised_inside_main_after_a_park_keeps_its_status(tmp_path):
    tail = (
        "from spyre_clickhouse_ingest import __main__ as m, spool_cli\n"
        "def boom(argv): raise SystemExit(1)\n"
        "spool_cli.main = boom\n"
        "m.main(['spool'])"
    )
    assert status(tmp_path, tail)[0] == 1


def test_a_bare_system_exit_outside_main_is_the_documented_75(tmp_path):
    assert status(tmp_path, "raise SystemExit(1)")[0] == 75


def test_a_caught_system_exit_does_not_hide_a_park(tmp_path):
    tail = (
        "from spyre_clickhouse_ingest import writes\n"
        "try:\n    sys.exit(1)\nexcept SystemExit:\n    writes.clear_exit()"
    )
    assert status(tmp_path, tail)[0] == 75


def test_connect_errors_after_a_park_keep_their_status(tmp_path):
    tail = "from spyre_clickhouse_ingest.client import ClickHouse; ClickHouse.connect()"
    rc, err = status(tmp_path, tail)
    assert rc == 1 and "CLICKHOUSE_HOST" in err


# ── misc ────────────────────────────────────────────────────────────────────────


def test_a_refused_insert_quorum_is_fatal_not_dropped():
    from guard_helpers import guarded as resilient_insert
    from spyre_clickhouse_ingest.breaker import SettingRefused

    class C:
        calls = 0

        def insert(self, *a, **k):
            C.calls += 1
            raise DatabaseError(
                "Code: 452. Setting insert_quorum is constrained", code=452, name="X"
            )

    with pytest.raises(SettingRefused):
        resilient_insert(
            C(), "t", [[1]], ["a"], scope="s", replicated=True, sleep=lambda s: None,
            log=lambda m: None,
        )  # fmt: skip
    assert C.calls == 1


def test_assume_star_only_covers_replicated_databases(monkeypatch):
    monkeypatch.setenv("CH_WRITE_ASSUME_REPLICATED", "*")
    monkeypatch.delenv("CH_V2_DATABASES", raising=False)
    monkeypatch.delenv("CLICKHOUSE_DB_V2", raising=False)
    assert writes._assumed("spyre_v2", "t") and not writes._assumed(
        "spyre", "test_runs"
    )


def test_parked_rows_is_the_base_of_batch_parked():
    from spyre_clickhouse_ingest.batching import BatchParked

    assert issubclass(BatchParked, writes.ParkedRows)
    assert json.dumps({}) == "{}"


# ── an unreadable engine on a Replicated database never becomes an untokened write ─────


class LookupDown(Cluster):
    def __init__(self, database, error):
        super().__init__()
        self.database, self.error = database, error

    def query(self, sql, parameters=None):
        raise self.error


def test_a_retriable_lookup_failure_on_a_replicated_database_refuses_to_write(
    monkeypatch,
):
    from spyre_clickhouse_ingest.batching import BatchFailed, insert_chunks

    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.delenv("CH_WRITE_ASSUME_REPLICATED", raising=False)
    c = LookupDown("spyre_v2", ConnectionRefusedError("down"))
    with pytest.raises(BatchFailed):
        insert_chunks(c, "t", [[1]], ["a"])
    with pytest.raises(
        BatchFailed
    ):  # the cached failure refuses too, without waiting again
        insert_chunks(c, "t", [[1]], ["a"])
    assert c.inserts == []


def test_a_denied_lookup_fails_closed_in_a_replicated_database_and_a_v1_one_stays_plain(
    monkeypatch,
):
    from spyre_clickhouse_ingest.batching import BatchFailed, insert_chunks

    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")

    denied = LookupDown("spyre_v2", DatabaseError("Code: 497. x", code=497, name="X"))
    with pytest.raises(
        BatchFailed, match="engine lookup failed"
    ):  # never tokenless into a Replicated db
        insert_chunks(denied, "t", [[1]], ["a"])
    assert denied.inserts == []
    v1 = LookupDown("spyre", ConnectionRefusedError("down"))
    insert_chunks(v1, "t", [[1]], ["a"])
    assert len(v1.inserts) == 1


# ── replay safety ───────────────────────────────────────────────────────────────


def put(spool, token="a-token", host=None, token_sent=True, **kw):
    meta = {"host": host, "token_sent": token_sent}
    spool.put("t", ["a"], [[1]], token, "d", scope="s", meta=meta, **kw)
    return next(spool.pending())


def test_an_ambiguous_tokenless_batch_is_never_replayed(tmp_path):
    item = put(Spool(tmp_path / "sp"), ambiguous=True, token_sent=False)
    assert "without a token" in writes.replay(Cluster(), item)


def test_an_ambiguous_batch_past_the_dedup_window_is_refused(tmp_path, monkeypatch):
    item = put(Spool(tmp_path / "sp"), ambiguous=True, created=1.0)
    assert "past the" in writes.replay(Cluster(), item)
    monkeypatch.setenv("CH_WRITE_REPLAY_MAX_AGE_S", str(10**12))
    assert writes.replay(Cluster(), item) == "sent"


def test_a_batch_refused_before_any_write_replays_at_any_age(tmp_path):
    item = put(Spool(tmp_path / "sp"), ambiguous=False, created=1.0)
    assert writes.replay(Cluster(), item) == "sent"


def test_a_batch_for_another_host_is_not_sent_here(tmp_path):
    item = put(Spool(tmp_path / "sp"), host="ha.example")
    assert "connected to h" in writes.replay(Cluster(), item)  # Cluster.ch_host == "h"


def test_a_file_whose_name_and_token_disagree_is_set_aside(tmp_path):
    spool = Spool(tmp_path / "sp")
    spool.put("t", ["a"], [[1]], "real-token", "d")
    (path,) = spool.files()
    doc = json.loads(path.read_text())
    doc["token"] = "other-token"
    path.write_text(json.dumps(doc))
    assert (
        list(spool.pending()) == [] and path.with_name(path.name + ".corrupt").exists()
    )


def test_typed_values_survive_the_spool(tmp_path):
    import datetime as dt
    import decimal
    import uuid

    row = [
        dt.datetime(2026, 1, 2, 3, 4, 5, 678000), dt.date(2026, 1, 2), decimal.Decimal("1.50"),
        uuid.UUID(int=7), {"k": [1, 2]}, b"\x00\xff", None, "s", 3, 2.5,
    ]  # fmt: skip
    spool = Spool(tmp_path / "sp")
    spool.put("t", list("abcdefghij"), [row], "typed")
    assert next(spool.pending()).rows == [row]


# ── router: the remaining shapes ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT r.tags FROM spyre_v2.runs AS r ARRAY JOIN r.tags AS tg",
        "SELECT r.tags FROM spyre_v2.runs AS r LEFT ARRAY JOIN r.tags AS tg",
        "SELECT x FROM spyre_v2.a, spyre_v2.b",
    ],
)
def test_array_join_and_same_host_comma_joins_are_not_cross_host(routed, sql):
    c, legacy, v2 = routed
    assert c.route(sql=sql) is v2


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM spyre_v2.t1, spyre.t2",
        "SELECT * FROM spyre.t2, spyre_v2.t1",
        "CREATE TABLE spyre_v2.a AS spyre.b",
        "SELECT * FROM spyre_v2.a JOIN spyre.b ON 1",
    ],
)
def test_comma_join_and_create_as_across_hosts_are_refused(routed, sql):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement):
        c.route(sql=sql)


def test_a_database_name_as_data_is_not_a_database_reference(routed):
    c, legacy, v2 = routed
    assert (
        c.route(sql="SELECT * FROM spyre.test_runs WHERE name = 'spyre_kb'") is legacy
    )
    assert c.route(sql="SELECT * FROM system.tables WHERE database = 'spyre_kb'") is v2
    assert (
        c.route(sql="SELECT 1 FROM system.columns WHERE table_schema IN ('vigil_ops')")
        is v2
    )
    assert c.route(sql="SELECT 1", parameters={"name": "spyre_kb"}) is legacy
    assert c.route(sql="SELECT 1", parameters={"db": "spyre_kb"}) is v2
    assert c.route(sql="SELECT 1", parameters={"source_database": "spyre_kb"}) is v2


# ── the action's manifest step ──────────────────────────────────────────────────


def run_report_step(tmp_path, persistent, files):
    import os
    import yaml

    action = yaml.safe_load(
        (
            ROOT.parents[1] / ".github/actions/ingest-xml-to-clickhouse/action.yml"
        ).read_text()
    )
    step = next(s for s in action["runs"]["steps"] if s.get("id") == "ch-spool")
    spool = tmp_path / "temp" / "ch-spool"
    spool.mkdir(parents=True)
    for name, text in files.items():
        (spool / name).write_text(text)
    out, summary = tmp_path / "out", tmp_path / "summary"
    out.write_text("")
    summary.write_text("")
    env = {
        "PATH": os.environ["PATH"], "RUNNER_TEMP": str(tmp_path / "temp"),
        "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary),
    }  # fmt: skip
    if persistent:
        env["CH_WRITE_SPOOL_PERSISTENT"] = "1"
    p = subprocess.run(
        ["bash", "-c", step["run"]], env=env, capture_output=True, text=True
    )
    manifest = tmp_path / "temp" / "ch-spool-manifest.jsonl"
    return p, out.read_text(), summary.read_text(), manifest


GOOD = json.dumps(
    {"table": "t", "columns": ["a"], "rows": [["SECRET-ROW-VALUE"]], "token": "tok",
     "created": __import__("time").time(), "ambiguous": False}
)  # fmt: skip


def test_the_report_step_fails_loudly_on_an_ephemeral_spool_and_never_leaks_rows(
    tmp_path,
):
    p, out, summary, manifest = run_report_step(
        tmp_path, False,
        {"0000000000001-tok.json": GOOD, "0000000000002-bad.json": "{not json",
         ".0000000000003-x.tmp": "partial", "0000000000004-old.json.corrupt": "x"},
    )  # fmt: skip
    assert p.returncode == 1 and "parked=true" in out
    assert "LOST when this job ends" in summary and "1 unreadable" in summary
    assert "SECRET-ROW-VALUE" not in summary + manifest.read_text()
    assert json.loads(manifest.read_text().splitlines()[0])["rows"] == 1


def test_the_report_step_passes_on_a_persistent_spool(tmp_path):
    p, out, summary, _ = run_report_step(
        tmp_path, True, {"0000000000001-tok.json": GOOD}
    )
    assert p.returncode == 0 and "parked=true" in out and "spool replay" in summary


def test_the_report_step_is_silent_for_leftovers_only(tmp_path):
    p, out, summary, _ = run_report_step(
        tmp_path, False, {".0000000000003-x.tmp": "partial", "1-x.json.corrupt": "x"}
    )
    assert p.returncode == 0 and out == "" and summary == ""


def test_a_nested_select_list_is_not_mistaken_for_a_comma_join(routed):
    c, legacy, v2 = routed
    sql = "SELECT a FROM (SELECT a, t.col FROM spyre_v2.x AS t) WHERE a = 1"
    assert c.route(sql=sql) is v2
    sql = "INSERT INTO spyre_v2.y SELECT a, t.col, u.z FROM spyre_v2.x AS t JOIN spyre_v2.w AS u ON 1"
    assert c.route(sql=sql) is v2


# ── malformed spool files never abort the listing ───────────────────────────────

MALFORMED = {
    "bad-decimal": {"table": "t", "columns": ["a"], "rows": [[{"$dec": "abc"}]]},
    "bad-map-key": {"table": "t", "columns": ["a"], "rows": [[{"$map": [[[1], 2]]}]]},
    "bad-map-shape": {"table": "t", "columns": ["a"], "rows": [[{"$map": 5}]]},
    "bad-created": {"table": "t", "columns": ["a"], "rows": [[1]], "created": "soon"},
    "bad-uuid": {"table": "t", "columns": ["a"], "rows": [[{"$uuid": "nope"}]]},
}


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_spool_file_is_set_aside_and_the_rest_still_list(tmp_path, name):
    spool = Spool(tmp_path / "sp")
    spool.put("t", ["a"], [[1]], "good-token")
    doc = {"token": f"{name}", "ambiguous": False, **MALFORMED[name]}
    (tmp_path / "sp" / f"0000000000009-{name}.json").write_text(json.dumps(doc))
    (tmp_path / "sp" / "0000000000008-list.json").write_text("[1, 2]")
    assert [i["token"] for i in spool.items()] == ["good-token"]
    assert list((tmp_path / "sp").glob("*.corrupt"))


def test_the_report_step_survives_the_same_malformed_files(tmp_path):
    files = {"0000000000001-tok.json": GOOD, "0000000000008-list.json": "[1, 2]"}
    for i, (name, doc) in enumerate(MALFORMED.items()):
        files[f"00000000001{i}-{name}.json"] = json.dumps({"token": name, **doc})
    p, out, summary, manifest = run_report_step(tmp_path, False, files)
    assert p.returncode == 1 and "parked=true" in out
    assert len(manifest.read_text().splitlines()) >= 1
    assert "unreadable" in summary


# ── every workflow that runs an ingest action exports the spool/v2 settings ─────────


def test_every_workflow_using_an_ingest_action_exports_the_ingest_env():
    import yaml

    wf_dir = ROOT.parents[1] / ".github/workflows"
    needed = {"CH_WRITE_SPOOL", "CH_WRITE_SPOOL_PERSISTENT", "CH_REQUIRE_V2_HOST"}
    checked = 0
    for path in sorted(wf_dir.glob("*.y*ml")):
        doc = yaml.safe_load(path.read_text())
        for name, job in (doc.get("jobs") or {}).items():
            steps = job.get("steps") or []
            uses = [
                s.get("uses", "") for s in steps
                if "/actions/ingest-" in s.get("uses", "") or "/actions/v2-artifact" in s.get("uses", "")
            ]  # fmt: skip
            if not uses:
                continue
            checked += 1
            have = set(doc.get("env") or {}) | set(job.get("env") or {})
            assert needed <= have, (
                f"{path.name}:{name} runs {uses} without {needed - have}"
            )
    assert checked >= 5


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM (SELECT a FROM spyre_v2.t) q, spyre.u",
        "SELECT * FROM spyre_v2.t, (SELECT a FROM spyre.u) q",
        "SELECT * FROM (SELECT 1) q, spyre.u WHERE x IN (SELECT a FROM spyre_v2.t)",
    ],
)
def test_a_comma_list_after_a_subquery_is_still_a_from_list(routed, sql):
    c, *_ = routed
    with pytest.raises(client.CrossHostStatement):
        c.route(sql=sql)


def test_a_function_call_in_select_does_not_open_a_from_list(routed):
    c, legacy, v2 = routed
    assert c.route(sql="SELECT f(a, spyre.b) FROM spyre_v2.t") is v2


@pytest.mark.parametrize("flag", ["FALSE", " No ", "0"])
def test_require_flag_off_values_are_case_and_space_insensitive(monkeypatch, flag):
    monkeypatch.delenv("CLICKHOUSE_V2_HOST", raising=False)
    monkeypatch.setenv("CLICKHOUSE_HOST", "legacy")
    monkeypatch.setenv("CLICKHOUSE_USER", "w")
    monkeypatch.setenv("CLICKHOUSE_PASS", "p")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setenv("CH_REQUIRE_V2_HOST", flag)
    import clickhouse_connect

    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **k: Rec("x"))
    client.ClickHouse.connect()


@pytest.mark.parametrize("verb", ["REPLACE", "BACKUP", "RESTORE", "MOVE"])
def test_writer_refuses_the_state_changing_verbs_too(verb):
    with pytest.raises(client.SchemaStatementOnWriter):
        client.WriterClient.check(f"{verb} TABLE a TO b")


# ── replay and the breaker ──────────────────────────────────────────────────────


def test_a_replay_that_retries_keeps_its_probe_slot_and_hands_it_back(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0.01")
    c = Cluster()
    c.fail = 2
    breaker = writes.breaker_for(c, "t", "d")
    breaker.trip(0.0, 1.0)  # open long ago: the next caller probes
    item = put(Spool(tmp_path / "sp"), ambiguous=False)
    assert writes.replay(c, item) == "sent"
    assert not breaker.tripped and not breaker.probing


def test_a_failed_replay_trips_the_breaker_and_frees_the_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    c = Cluster()
    c.fail = 99
    item = put(Spool(tmp_path / "sp"), ambiguous=False)
    assert writes.replay(c, item).startswith("not stored")
    breaker = writes.breaker_for(c, "t", "d")
    assert breaker.tripped and not breaker.probing


def test_an_ambiguous_batch_is_not_replayed_when_the_dedup_window_is_unreadable(
    tmp_path,
):
    class NoWindow(Cluster):
        def query(self, sql, parameters=None):
            if "merge_tree_settings" in sql:
                raise DatabaseError("Code: 497. x", code=497, name="X")
            return super().query(sql, parameters)

    item = put(Spool(tmp_path / "sp"), ambiguous=True)
    assert "unreadable" in writes.replay(NoWindow(), item)
    assert (
        writes.replay(
            NoWindow(), put(Spool(tmp_path / "sp2"), "b-token", ambiguous=False)
        )
        == "sent"
    )


def test_replay_does_not_work_around_a_refused_dedup_token(tmp_path):
    class Constrained(Cluster):
        def insert(self, *a, **k):
            raise DatabaseError(
                "Code: 452. Setting insert_deduplication_token is constrained",
                code=452,
                name="X",
            )

    item = put(Spool(tmp_path / "sp"), ambiguous=False)
    assert "SettingRefused" in writes.replay(Constrained(), item)


def test_the_cells_the_token_accepts_survive_the_spool(tmp_path):
    import datetime as dt
    import ipaddress

    row = [
        dt.time(1, 2, 3),
        ipaddress.ip_address("10.0.0.1"),
        ipaddress.ip_address("::1"),
    ]
    spool = Spool(tmp_path / "sp")
    spool.put("t", ["a", "b", "c"], [row], "t-1")
    assert next(spool.pending()).rows == [row]


def test_an_enum_and_numpy_scalars_survive_the_spool(tmp_path):
    import enum

    class Color(enum.Enum):
        RED = "red"

    class Scalar:
        __module__ = "numpy"

        def __init__(self, v):
            self.v = v

        def item(self):
            return self.v

    spool = Spool(tmp_path / "sp")
    spool.put(
        "t", list("abcd"), [[Color.RED, Scalar(7), Scalar(1.5), Scalar(True)]], "t-1"
    )
    assert next(spool.pending()).rows == [["red", 7, 1.5, True]]


def test_a_dead_host_is_not_asked_for_the_dedup_window(tmp_path):
    from spyre_clickhouse_ingest.breaker import Breaker, guarded_insert
    from spyre_clickhouse_ingest.resilience import Policy

    breaker = Breaker()
    breaker.trip(0.0, 10**9)
    asked = []
    guarded_insert(
        Cluster(), "t", [[1]], ["a"], scope="s", replicated=True,
        dedup_window=lambda: asked.append(1), breaker=breaker,
        spool=Spool(tmp_path / "sp"), policy=Policy(), clock=lambda: 5.0,
        log=lambda m: None,
    )  # fmt: skip
    assert asked == []
