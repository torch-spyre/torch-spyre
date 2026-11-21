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

"""The write path against a real server with Replicated tables.

A one-node ClickHouse container with an embedded Keeper stands in for the cluster: insert dedup by
token, the real TOO_FEW_LIVE_REPLICAS refusal (insert_quorum=2 with one replica) and the real
engine lookup. Runs when podman or docker is on PATH and the image is local (or CH_TEST_PULL=1);
CH_TEST_IMAGE overrides the image.
"""

import os
import shutil
import socket
import subprocess
import time
import uuid

import clickhouse_connect
import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import probe, resilience, writes
from spyre_clickhouse_ingest.batching import BatchFailed, insert_chunks
from spyre_clickhouse_ingest.client import SchemaStatementOnWriter, WriterClient
from guard_helpers import guarded
from spyre_clickhouse_ingest.resilience import (
    Policy,
    Spool,
    SpoolFull,
    resilient_insert,
)

IMAGE = os.environ.get("CH_TEST_IMAGE", "docker.io/clickhouse/clickhouse-server:26.3")
KEEPER = """<clickhouse>
 <listen_host>0.0.0.0</listen_host>
 <keeper_server><tcp_port>9181</tcp_port><server_id>1</server_id>
  <log_storage_path>/var/lib/clickhouse/coordination/log</log_storage_path>
  <snapshot_storage_path>/var/lib/clickhouse/coordination/snapshots</snapshot_storage_path>
  <raft_configuration><server><id>1</id><hostname>127.0.0.1</hostname><port>9234</port></server></raft_configuration>
 </keeper_server>
 <zookeeper><node><host>127.0.0.1</host><port>9181</port></node></zookeeper>
 <macros><shard>01</shard><replica>r1</replica></macros>
</clickhouse>
"""


def _runtime():
    rt = shutil.which("podman") or shutil.which("docker")
    if not rt:
        return None
    local = subprocess.run([rt, "image", "exists", IMAGE], capture_output=True)
    if rt.endswith("podman") and local.returncode == 0:
        return rt
    if not rt.endswith("podman"):
        have = subprocess.run([rt, "image", "inspect", IMAGE], capture_output=True)
        if have.returncode == 0:
            return rt
    return rt if os.environ.get("CH_TEST_PULL") else None


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    rt = _runtime()
    if not rt:
        pytest.skip("no container runtime with a ClickHouse image")
    cfg = tmp_path_factory.mktemp("chcfg") / "keeper.xml"
    cfg.write_text(KEEPER)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    name = f"spyre-ingest-test-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        [
            rt, "run", "-d", "--rm", "--name", name,
            "-p", f"127.0.0.1:{port}:8123",
            "-e", "CLICKHOUSE_USER=writer", "-e", "CLICKHOUSE_PASSWORD=pw",
            "-e", "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1",
            "-v", f"{cfg}:/etc/clickhouse-server/config.d/keeper.xml:ro,Z",
            IMAGE,
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode:
        pytest.skip(f"container did not start: {run.stderr[:200]}")
    try:
        client = None
        for _ in range(60):
            try:
                client = clickhouse_connect.get_client(
                    host="127.0.0.1", port=port, username="writer", password="pw"
                )
                client.command(
                    "CREATE TABLE IF NOT EXISTS default.boot (a Int8) "
                    "ENGINE = ReplicatedMergeTree('/boot/{shard}', '{replica}') ORDER BY a"
                )
                break
            except Exception:  # noqa: BLE001 -- the server is still starting
                time.sleep(1)
        else:
            pytest.skip("container never became ready")
        yield client
    finally:
        subprocess.run([rt, "rm", "-f", name], capture_output=True)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "CH_WRITE_RESILIENCE",
        "CH_WRITE_SPOOL",
        "RUNNER_TEMP",
        "WORKSPACE",
        "CH_WRITE_QUORUM",
        "CH_WRITE_DEADLINE_S",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0")
    monkeypatch.setattr(writes, "_parked", [])
    monkeypatch.setattr(writes, "_spools", {})


@pytest.fixture
def table(server):
    name = f"t_{uuid.uuid4().hex[:8]}"
    server.command(
        f"CREATE TABLE default.{name} (a UInt32, b String) "
        f"ENGINE = ReplicatedMergeTree('/t/{name}/{{shard}}', '{{replica}}') ORDER BY a"
    )
    yield name
    server.command(f"DROP TABLE IF EXISTS default.{name} SYNC")


ROWS = [[i, f"r{i}"] for i in range(10)]
FAST = dict(sleep=lambda s: None, log=lambda m: None)


def count(server, table):
    return server.command(f"SELECT count() FROM default.{table}")


def test_a_retried_token_lands_once(server, table):
    kw = dict(
        scope="run-1", replicated=True, database="default", feeds_views=False, **FAST
    )
    first = resilient_insert(server, table, ROWS, ["a", "b"], **kw)
    again = resilient_insert(server, table, ROWS, ["a", "b"], **kw)
    assert first.stored == again.stored == 10
    assert count(server, table) == 10


def test_a_different_scope_is_a_different_batch(server, table):
    kw = dict(replicated=True, database="default", feeds_views=False, **FAST)
    resilient_insert(server, table, ROWS, ["a", "b"], scope="run-1", **kw)
    resilient_insert(server, table, ROWS, ["a", "b"], scope="run-2", **kw)
    assert count(server, table) == 20


class LostResponse:
    """Delegates to the real client, but the first call stores the block and then errors."""

    def __init__(self, inner, error):
        self.inner, self.error, self.left = inner, error, 1

    def insert(self, *a, **k):
        self.inner.insert(*a, **k)
        if self.left:
            self.left -= 1
            raise self.error


def test_an_ambiguous_error_after_the_block_landed_does_not_double_it(server, table):
    lost = LostResponse(server, DatabaseError("Code: 159. timeout", code=159, name="T"))
    out = resilient_insert(
        lost, table, ROWS, ["a", "b"], scope="s", replicated=True,
        database="default", policy=Policy(batch_rows=4), **FAST,
    )  # fmt: skip
    assert out.stored == 10 and out.spooled == 0
    assert count(server, table) == 10


def test_an_ambiguous_error_before_the_block_landed_still_stores_it(server, table):
    class Dropped:
        left = 1

        def insert(self, *a, **k):
            if self.left:
                self.left -= 1
                raise OSError("connection reset")
            server.insert(*a, **k)

    out = resilient_insert(
        Dropped(), table, ROWS, ["a", "b"], scope="s", replicated=True,
        database="default", **FAST,
    )  # fmt: skip
    assert out.stored == 10 and count(server, table) == 10


def test_unmet_quorum_is_real_285_and_parks_after_the_deadline(server, table, tmp_path):
    spool = Spool(tmp_path / "sp")
    ticks = iter(range(0, 10_000, 100))
    out = resilient_insert(
        server, table, ROWS, ["a", "b"], scope="s", replicated=True,
        database="default", feeds_views=False, spool=spool,
        policy=Policy(quorum="2", deadline_s=250, backoff_s=1),
        clock=lambda: next(ticks), **FAST,
    )  # fmt: skip
    assert out.stored == 0 and out.spooled == 10
    assert count(server, table) == 0
    item = spool.items()[0]
    assert item["rows"] == 10 and item["held"] is False


def test_a_full_spool_raises_and_nothing_lands(server, table, tmp_path):
    ticks = iter(range(0, 10_000, 100))
    with pytest.raises(SpoolFull):
        resilient_insert(
            server, table, ROWS, ["a", "b"], scope="s", replicated=True,
            database="default", feeds_views=False,
            spool=Spool(tmp_path / "sp", max_bytes=20),
            policy=Policy(quorum="2", deadline_s=250, backoff_s=1),
            clock=lambda: next(ticks), **FAST,
        )  # fmt: skip
    assert count(server, table) == 0


def test_insert_chunks_finds_the_engine_and_dedups_a_rerun(server, table, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    client = WriterClient(server)
    assert (
        insert_chunks(client, table, ROWS, ["a", "b"], database="default", size=4) == 10
    )
    assert (
        insert_chunks(client, table, ROWS, ["a", "b"], database="default", size=4) == 10
    )
    assert count(server, table) == 10


def test_insert_chunks_fails_with_batch_failed_when_quorum_cannot_be_met(
    server, table, monkeypatch
):
    monkeypatch.setenv("CH_WRITE_QUORUM", "2")
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    with pytest.raises(BatchFailed):
        insert_chunks(WriterClient(server), table, ROWS, ["a", "b"], database="default")
    assert count(server, table) == 0


def test_a_dependent_view_is_detected(server, table):
    server.command(
        f"CREATE MATERIALIZED VIEW default.{table}_mv ENGINE = ReplicatedMergeTree("
        f"'/t/{table}_mv/{{shard}}', '{{replica}}') ORDER BY a AS SELECT a, b FROM default.{table}"
    )
    try:
        found = writes.target(server, table, "default")
        assert found.replicated and found.feeds_views
    finally:
        server.command(f"DROP VIEW default.{table}_mv SYNC")


def test_the_writer_connection_refuses_ddl_before_the_server_sees_it(server, table):
    w = WriterClient(server)
    for sql in (f"DROP TABLE default.{table}", f"TRUNCATE TABLE default.{table}",
                f"ALTER TABLE default.{table} ADD COLUMN c Int8",
                "CREATE TABLE default.zz (a Int8) ENGINE = Memory"):  # fmt: skip
        with pytest.raises(SchemaStatementOnWriter):
            w.command(sql)
    assert server.command(f"EXISTS TABLE default.{table}") == 1
    assert server.command("EXISTS TABLE default.zz") == 0


def test_probe_round_trip_on_a_replicated_table(server):
    server.command(
        probe.DDL.format(
            table="default.zz_write_probe",
            engine="ReplicatedMergeTree('/t/zz_write_probe/{shard}', '{replica}')",
        )
    )
    r = probe.probe(WriterClient(server), "default.zz_write_probe", rows=3)
    assert r["ok"] and r["replicated"] and r["dedup"] == "held" and r["count"] == 3


def test_a_rebuild_after_a_lightweight_delete_is_not_dropped_as_a_duplicate(
    server, table
):
    # The counters rebuild is DELETE then INSERT ... SELECT with an identical result.
    server.command(f"INSERT INTO default.{table} VALUES (1, 'x'), (2, 'y')")
    server.command(
        f"DELETE FROM default.{table} WHERE a >= 0 SETTINGS mutations_sync = 2"
    )
    assert count(server, table) == 0
    server.command(
        f"INSERT INTO default.{table} SELECT number + 1, ['x', 'y'][number + 1] FROM numbers(2)"
    )
    assert count(server, table) == 2
    server.command(
        f"DELETE FROM default.{table} WHERE a >= 0 SETTINGS mutations_sync = 2"
    )
    server.command(
        f"INSERT INTO default.{table} SELECT number + 1, ['x', 'y'][number + 1] FROM numbers(2)"
    )
    assert count(server, table) == 2


def test_a_constrained_profile_that_rejects_an_optional_setting_still_lands_the_batch(
    server, table
):
    server.command("DROP USER IF EXISTS lim")
    server.command(
        "CREATE USER lim IDENTIFIED BY 'pw' SETTINGS "
        "deduplicate_blocks_in_dependent_materialized_views = 0 CONST"
    )
    server.command("GRANT SELECT, INSERT ON default.* TO lim")
    limited = clickhouse_connect.get_client(
        host=server.url.split("//")[1].split(":")[0],
        port=int(server.url.rsplit(":", 1)[1]),
        username="lim",
        password="pw",
    )
    logged = []
    out = guarded(
        limited, table, ROWS, ["a", "b"], scope="s", replicated=True,
        database="default", feeds_views=True, sleep=lambda s: None, log=logged.append,
    )  # fmt: skip
    assert out.stored == 10 and count(server, table) == 10
    assert any(
        "deduplicate_blocks_in_dependent_materialized_views" in m for m in logged
    )
    server.command("DROP USER lim")


def test_a_table_the_login_cannot_see_fails_closed_in_a_replicated_database(
    server, table, monkeypatch
):
    server.command("DROP USER IF EXISTS blind")
    server.command("CREATE USER blind IDENTIFIED BY 'pw'")
    server.command(f"GRANT INSERT ON default.{table}_other TO blind")
    server.command(
        f"CREATE TABLE IF NOT EXISTS default.{table}_other (a UInt32, b String) "
        f"ENGINE = ReplicatedMergeTree('/t/{table}_other/{{shard}}', '{{replica}}') ORDER BY a"
    )
    host, port = server.url.split("//")[1].rsplit(":", 1)
    blind = clickhouse_connect.get_client(
        host=host, port=int(port), username="blind", password="pw"
    )
    try:
        assert writes.target(blind, f"{table}_other", "default").replicated
        monkeypatch.setenv("CH_V2_DATABASES", "default")
        with pytest.raises(resilience.WriteUnavailable):
            writes.target(blind, table, "default")  # exists, but system.tables hides it
        assert count(server, table) == 0
    finally:
        server.command("DROP USER blind")
        server.command(f"DROP TABLE IF EXISTS default.{table}_other SYNC")


def test_the_counters_rebuild_on_replicated_tables_is_not_deduplicated(server):
    from spyre_clickhouse_ingest.writer import TestResultWriter

    server.command("CREATE DATABASE IF NOT EXISTS repl_ctr")
    server.command("DROP TABLE IF EXISTS repl_ctr.test_case_runs SYNC")
    server.command("DROP TABLE IF EXISTS repl_ctr.run_case_counters SYNC")
    server.command(
        "CREATE TABLE repl_ctr.test_case_runs (run_id UUID, component String, "
        "status String, props Map(String, String)) ENGINE = ReplicatedMergeTree("
        "'/t/repl_ctr_f/{shard}', '{replica}') ORDER BY run_id"
    )
    server.command(
        "CREATE TABLE repl_ctr.run_case_counters (run_id UUID, component String, "
        "total_tests UInt64, passed UInt64, failed UInt64, errors UInt64, skipped UInt64, "
        "xfail UInt64, xpass UInt64, recovered UInt64) ENGINE = ReplicatedSummingMergeTree("
        "'/t/repl_ctr_c/{shard}', '{replica}') ORDER BY (run_id, component)"
    )
    run = "00000000-0000-4000-8000-0000000000c1"
    server.command(
        "INSERT INTO repl_ctr.test_case_runs VALUES "
        f"('{run}', 'c', 'passed', map()), ('{run}', 'c', 'failed', map())"
    )

    def counted():
        return int(
            server.command(
                f"SELECT sum(total_tests) FROM repl_ctr.run_case_counters WHERE run_id = '{run}'"
            )
            or 0
        )

    for _ in range(3):  # identical blocks, each after the previous delete
        TestResultWriter._recount(server, "repl_ctr", run, "c")
        assert counted() == 2
    assert not TestResultWriter._counters_stale(server, "repl_ctr", run, "c")


# ── re-running a file converges to one attempt, whatever failed after the insert ─────────

INGEST_RUN = "00000000-0000-4000-8000-0000000000aa"


@pytest.fixture(scope="module")
def ingest_db(server):
    from pathlib import Path

    from spyre_clickhouse_ingest.apply_schema import SchemaApplier

    server.command("CREATE DATABASE IF NOT EXISTS ing")
    host, port = server.url.split("//")[1].rsplit(":", 1)
    ing = clickhouse_connect.get_client(
        host=host, port=int(port), username="writer", password="pw", database="ing"
    )
    schema_dir = Path(__file__).resolve().parents[1] / "schema"
    version = SchemaApplier.version_tuple(ing.command("SELECT version()"))
    files = SchemaApplier.selected_files(schema_dir, [], version)
    SchemaApplier.apply(ing, "ing", files, SchemaApplier.migration_files(schema_dir))
    return "ing"


def case_rows(n, status="passed", prefix="t"):
    return [
        {
            "classname": "k", "name": f"{prefix}{i}", "status": status,
            "duration_s": 1.0, "fail_message": "", "properties": [],
        }
        for i in range(n)
    ]  # fmt: skip


class Interrupted(BaseException):
    """Stands in for a killed process."""


class Failing:
    """Delegates to the real client; raises once when a statement matches."""

    def __init__(
        self,
        inner,
        command=None,
        table=None,
        after=0,
        error=RuntimeError,
        after_commands=0,
    ):
        self.inner, self.command_match, self.table, self.after, self.error = (
            inner, command, table, after, error,
        )  # fmt: skip
        self.after_commands = after_commands
        self.armed = True

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def command(self, sql, *a, **k):
        if self.armed and self.command_match and self.command_match in sql:
            if self.after_commands:
                self.after_commands -= 1
            else:
                self.armed = False
                raise self.error("injected")
        return self.inner.command(sql, *a, **k)

    def insert(self, table, *a, **k):
        if self.armed and self.table == table:
            if self.after:
                self.after -= 1
            else:
                self.armed = False
                raise self.error("injected")
        return self.inner.insert(table, *a, **k)


def ingest(client, db, run, cases, attempt, source="f.xml"):
    """What results.py does for one file."""
    from spyre_clickhouse_ingest.writer import TestResultWriter as W

    if W.already_ingested(client, db, run, "comp", source, attempt=attempt):
        W.drop_older_attempts(client, db, run, "comp", source, attempt)
    else:
        W.replace_attempt(
            client, db, "comp", run, cases, source, attempt=attempt, arch="x86_64"
        )


def state(server, db, run):
    attempts = {
        r[0]: r[1]
        for r in server.query(
            f"SELECT toUInt32OrZero(props['run_attempt']) a, count() FROM {db}.test_case_runs "
            f"WHERE run_id = '{run}' GROUP BY a"
        ).result_rows
    }
    counted = server.command(
        f"SELECT sum(total_tests) FROM {db}.run_case_counters WHERE run_id = '{run}'"
    )
    return attempts, int(counted or 0)


def test_the_convergence_tests_run_against_plain_engines(server, ingest_db):
    engines = {
        r[0]
        for r in server.query(
            "SELECT engine FROM system.tables WHERE database = 'ing' "
            "AND name IN ('test_case_runs', 'run_case_counters')"
        ).result_rows
    }
    assert not any(e.startswith("Replicated") for e in engines)


@pytest.mark.parametrize(
    "inject",
    [
        {"command": "DELETE FROM ing.test_case_runs"},
        {"command": "ing.run_case_counters"},
        {"command": "DELETE FROM ing.test_case_runs", "error": Interrupted},
        {"command": "INSERT INTO ing.run_case_counters", "after_commands": 1},
    ],
    ids=[
        "delete raises",
        "counter rebuild raises",
        "process killed before the delete",
        "rebuild raises after the old rows are gone",
    ],
)
def test_a_failure_after_the_insert_converges_on_the_next_run(
    server, ingest_db, inject
):
    run = f"00000000-0000-4000-8000-{abs(hash(str(inject))) % 10**12:012d}"
    ingest(WriterClient(server), ingest_db, run, case_rows(6), 1)
    assert state(server, ingest_db, run) == ({1: 6}, 6)
    flaky = Failing(server, **inject)
    with pytest.raises(BaseException):
        ingest(WriterClient(flaky), ingest_db, run, case_rows(5, "failed"), 2)
    attempts, _ = state(server, ingest_db, run)
    assert 2 in attempts  # the new rows landed, the old ones may still be there
    ingest(WriterClient(server), ingest_db, run, case_rows(5, "failed"), 2)
    assert state(server, ingest_db, run) == ({2: 5}, 5)
    ingest(WriterClient(server), ingest_db, run, case_rows(5, "failed"), 2)  # and again
    assert state(server, ingest_db, run) == ({2: 5}, 5)


def test_a_later_batch_that_fails_is_finished_by_the_rerun(server, ingest_db):
    run = "00000000-0000-4000-8000-0000000000b1"
    ingest(WriterClient(server), ingest_db, run, case_rows(5003, prefix="old"), 1)
    flaky = Failing(server, table="test_case_runs", after=1)  # second batch fails
    with pytest.raises(Exception):
        ingest(WriterClient(flaky), ingest_db, run, case_rows(5003, prefix="new"), 2)
    attempts, _ = state(server, ingest_db, run)
    assert attempts.get(1) == 5003 and 0 < attempts.get(2, 0) < 5003
    ingest(WriterClient(server), ingest_db, run, case_rows(5003, prefix="new"), 2)
    assert state(server, ingest_db, run) == ({2: 5003}, 5003)


def test_a_repeat_of_the_same_attempt_adds_nothing(server, ingest_db):
    run = "00000000-0000-4000-8000-0000000000b2"
    for _ in range(3):
        ingest(WriterClient(server), ingest_db, run, case_rows(4), 3)
    assert state(server, ingest_db, run) == ({3: 4}, 4)


def test_a_failed_insert_leaves_the_previous_attempt_whole(server, ingest_db):
    run = "00000000-0000-4000-8000-0000000000b3"
    ingest(WriterClient(server), ingest_db, run, case_rows(4), 1)
    flaky = Failing(server, table="test_case_runs")
    with pytest.raises(RuntimeError):
        ingest(WriterClient(flaky), ingest_db, run, case_rows(4, "failed"), 2)
    assert state(server, ingest_db, run) == ({1: 4}, 4)


# ── a parked batch replays with its types intact ─────────────────────────────────────


def test_a_typed_batch_replays_from_the_spool(server, tmp_path, monkeypatch):
    import datetime as dt
    import decimal
    import uuid as uuidlib

    from spyre_clickhouse_ingest import spool_cli

    name = f"typed_{uuidlib.uuid4().hex[:8]}"
    server.command(
        f"CREATE TABLE default.{name} (id UUID, at DateTime64(3), d Date, amt Decimal(12, 4), "
        f"m Map(String, UInt32), n Nullable(String), t Array(String)) "
        f"ENGINE = ReplicatedMergeTree('/t/{name}/{{shard}}', '{{replica}}') ORDER BY id"
    )
    row = [
        uuidlib.UUID("00000000-0000-4000-8000-000000000042"),
        dt.datetime(2026, 11, 13, 1, 2, 3, 456000),
        dt.date(2026, 11, 13),
        decimal.Decimal("12.3456"),
        {"a": 1, "b": 2},
        None,
        ["x", "y"],
    ]
    cols = ["id", "at", "d", "amt", "m", "n", "t"]
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0")
    spool = Spool(tmp_path / "sp")
    tok = "typed-batch-token"
    spool.put(
        name,
        cols,
        [row],
        tok,
        "default",
        scope="s",
        ambiguous=True,
        meta={"host": None, "token_sent": True},
    )
    clients = lambda database="": WriterClient(server)  # noqa: E731
    for _ in range(2):  # the second replay is the dedup no-op
        spool_cli.replay_items(spool, tok, connect=clients)
        spool.put(name, cols, [row], tok, "default", scope="s", ambiguous=True)
    assert server.command(f"SELECT count() FROM default.{name}") == 1
    control = f"{name}_control"
    server.command(
        f"CREATE TABLE default.{control} (id UUID, at DateTime64(3), d Date, amt Decimal(12, 4), "
        "m Map(String, UInt32), n Nullable(String), t Array(String)) ENGINE = MergeTree ORDER BY id"
    )
    server.insert(
        f"default.{control}", [row], column_names=cols
    )  # the same row, sent directly
    read = "SELECT id, at, d, amt, m, n, t FROM default.%s"
    assert (
        server.query(read % name).result_rows
        == server.query(read % control).result_rows
    )
