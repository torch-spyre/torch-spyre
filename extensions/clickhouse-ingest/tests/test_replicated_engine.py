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

A one-node ClickHouse container with an embedded Keeper stands in for the cluster: token dedup,
the real TOO_FEW_LIVE_REPLICAS refusal (insert_quorum=2 with one replica), the engine lookup and
a typed replay. Runs when podman or docker is on PATH and the image is local (or CH_TEST_PULL=1);
CH_TEST_IMAGE overrides the image.
"""

import datetime as dt
import decimal
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import clickhouse_connect
import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import schema, spool_cli, writes
from spyre_clickhouse_ingest.batching import BatchParked
from spyre_clickhouse_ingest.client import SchemaStatementOnWriter, WriterClient
from spyre_clickhouse_ingest.resilience import Spool

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
    probe = "exists" if rt.endswith("podman") else "inspect"
    if subprocess.run([rt, "image", probe, IMAGE], capture_output=True).returncode == 0:
        return rt
    return rt if os.environ.get("CH_TEST_PULL") else None


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    rt = _runtime()
    if not rt:
        pytest.skip("no container runtime with a ClickHouse image")
    cfg = tmp_path_factory.mktemp("chcfg") / "keeper.xml"
    cfg.write_text(KEEPER)
    # Rootless podman maps the server's own uid: it must be able to read the mounted file.
    cfg.chmod(0o644)
    cfg.parent.chmod(0o755)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    name = f"spyre-ingest-test-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        [
            rt, "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8123",
            "-e", "CLICKHOUSE_USER=writer", "-e", "CLICKHOUSE_PASSWORD=pw",
            "-e", "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1",
            "-v", f"{cfg}:/etc/clickhouse-server/config.d/keeper.xml:ro,Z", IMAGE,
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode:
        pytest.skip(f"container did not start: {run.stderr[:200]}")
    try:
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
def clean_env(monkeypatch, tmp_path):
    for name in (
        "CH_WRITE_RESILIENCE",
        "CH_WRITE_QUORUM",
        "CH_WRITE_DEADLINE_S",
        "RUNNER_TEMP",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "spool"))
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0.01")
    monkeypatch.setenv("CH_WRITE_BACKOFF_MAX_S", "0.01")
    writes.reset()


def replicated(server, name, columns="a UInt32, b String", order="a"):
    server.command(
        f"CREATE TABLE default.{name} ({columns}) "
        f"ENGINE = ReplicatedMergeTree('/t/{name}/{{shard}}', '{{replica}}') ORDER BY {order}"
    )
    return name


@pytest.fixture
def table(server):
    name = replicated(server, f"t_{uuid.uuid4().hex[:8]}")
    yield name
    server.command(f"DROP TABLE IF EXISTS default.{name} SYNC")


ROWS = [[i, f"r{i}"] for i in range(10)]


def count(server, table):
    return server.command(f"SELECT count() FROM default.{table}")


class LostResponse:
    """The real client, but the first insert stores the block and then reports a timeout."""

    def __init__(self, inner):
        self.inner, self.left = inner, 1

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def insert(self, *a, **k):
        self.inner.insert(*a, **k)
        if self.left:
            self.left -= 1
            raise DatabaseError("Code: 159. timeout", code=159, name="T")


def test_a_resend_after_a_lost_response_does_not_double_the_batch(server, table):
    assert writes.insert(LostResponse(server), table, ROWS, ["a", "b"], "default") == 10
    assert count(server, table) == 10


def test_identical_rows_of_two_calls_both_land(server, table):
    for _ in range(2):
        writes.insert(WriterClient(server), table, ROWS, ["a", "b"], "default")
    assert count(server, table) == 20


def test_a_plain_table_is_left_to_the_callers_path(server):
    server.command(
        "CREATE TABLE default.plain (a UInt32) ENGINE = MergeTree ORDER BY a"
    )
    try:
        assert writes.insert(server, "plain", [[1]], ["a"], "default") is None
    finally:
        server.command("DROP TABLE default.plain SYNC")


def test_a_dependent_view_is_detected(server, table):
    server.command(
        f"CREATE MATERIALIZED VIEW default.{table}_mv ENGINE = ReplicatedMergeTree("
        f"'/t/{table}_mv/{{shard}}', '{{replica}}') ORDER BY a AS SELECT a, b FROM default.{table}"
    )
    try:
        assert writes.target(server, table, "default").feeds_views
    finally:
        server.command(f"DROP VIEW default.{table}_mv SYNC")


def test_unmet_quorum_parks_then_replay_lands_it_once(server, table, monkeypatch):
    monkeypatch.setenv("CH_WRITE_QUORUM", "2")  # one replica: a real 285 refusal
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.2")
    with pytest.raises(BatchParked):
        writes.insert(WriterClient(server), table, ROWS, ["a", "b"], "default")
    assert count(server, table) == 0
    spool = Spool(os.environ["CH_WRITE_SPOOL"])
    (item,) = spool.items()
    assert item["rows"] == 10 and item["held"] is False
    monkeypatch.delenv("CH_WRITE_QUORUM")
    connect = lambda database="": WriterClient(server)  # noqa: E731
    assert spool_cli.replay(spool, item["token"], connect=connect) == 0
    assert count(server, table) == 10 and spool.items() == []
    spool.put(table, ["a", "b"], ROWS, item["token"], "default", ambiguous=True)
    assert spool_cli.replay(spool, item["token"], connect=connect) == 0
    assert (
        count(server, table) == 10
    )  # the original token: the second replay is a no-op


def test_a_typed_batch_replays_from_the_spool_as_a_direct_insert_would_store_it(
    server, monkeypatch
):
    cols = ["id", "at", "d", "amt", "m", "n", "t"]
    ddl = (
        "id UUID, at DateTime64(3), d Date, amt Decimal(12, 4), m Map(String, UInt32), "
        "n Nullable(String), t Array(String)"
    )
    name = replicated(server, f"typed_{uuid.uuid4().hex[:8]}", ddl, "id")
    server.command(
        f"CREATE TABLE default.{name}_control ({ddl}) ENGINE = MergeTree ORDER BY id"
    )
    row = [
        uuid.UUID("00000000-0000-4000-8000-000000000042"),
        dt.datetime(2026, 11, 13, 1, 2, 3, 456000),
        dt.date(2026, 11, 13),
        decimal.Decimal("12.3456"),
        {"a": 1, "b": 2},
        None,
        ["x", "y"],
    ]
    monkeypatch.setenv("CH_WRITE_QUORUM", "2")
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.2")
    with pytest.raises(BatchParked):
        writes.insert(WriterClient(server), name, [row], cols, "default")
    monkeypatch.delenv("CH_WRITE_QUORUM")
    spool = Spool(os.environ["CH_WRITE_SPOOL"])
    connect = lambda database="": WriterClient(server)  # noqa: E731
    assert spool_cli.replay(spool, None, connect=connect) == 0
    server.insert(f"default.{name}_control", [row], column_names=cols)
    read = "SELECT id, at, d, amt, m, n, t FROM default.%s"
    stored = server.query(read % name).result_rows
    assert stored == server.query(read % f"{name}_control").result_rows and stored


def test_the_writer_connection_refuses_ddl_before_the_server_sees_it(server, table):
    w = WriterClient(server)
    for sql in (f"DROP TABLE default.{table}", f"TRUNCATE TABLE default.{table}",
                f"ALTER TABLE default.{table} ADD COLUMN c Int8",
                "CREATE TABLE default.zz (a Int8) ENGINE = Memory"):  # fmt: skip
        with pytest.raises(SchemaStatementOnWriter):
            w.command(sql)
    assert server.command(f"EXISTS TABLE default.{table}") == 1
    assert server.command("EXISTS TABLE default.zz") == 0


def test_table_insert_tokens_a_v2_table(server, monkeypatch):
    name = replicated(server, "artifact_tags_r", "tag String, tag_family String, "
                      "artifact_id String, refs Array(Tuple(String, String, String, String)), "
                      "published_refs Array(String), props Map(String, String)", "tag")  # fmt: skip
    monkeypatch.setattr(schema.ArtifactTags, "name", name)
    row = {"tag": "t", "tag_family": "f", "artifact_id": "a", "refs": [("1", "2", "3", "4")],
           "published_refs": ["p"], "props": {"k": "v"}}  # fmt: skip
    assert schema.ArtifactTags.insert(LostResponse(server), [row], db="default") == 1
    assert count(server, name) == 1


# ── re-running a file converges to one attempt, whatever failed after the insert ─────


@pytest.fixture(scope="module")
def ingest_db(server):
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
        {"classname": "k", "name": f"{prefix}{i}", "status": status, "duration_s": 1.0,
         "fail_message": "", "properties": []}
        for i in range(n)
    ]  # fmt: skip


class Interrupted(BaseException):
    """Stands in for a killed process."""


class Failing:
    """The real client; raises once when a statement or an insert matches."""

    def __init__(self, inner, command=None, table=None, after=0, error=RuntimeError):
        self.inner, self.match, self.table, self.after, self.error = (
            inner, command, table, after, error,
        )  # fmt: skip
        self.armed = True

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def command(self, sql, *a, **k):
        if self.armed and self.match and self.match in sql:
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
    """What results.py does for one file: always the replace path."""
    from spyre_clickhouse_ingest.writer import TestResultWriter as W

    W.replace_attempt(
        client, db, "comp", run, cases, source, attempt=attempt, arch="x86_64"
    )


def state(server, db, run):
    attempts = dict(
        server.query(
            f"SELECT toUInt32OrZero(props['run_attempt']) a, count() FROM {db}.test_case_runs "
            f"WHERE run_id = '{run}' GROUP BY a"
        ).result_rows
    )
    counted = server.command(
        f"SELECT sum(total_tests) FROM {db}.run_case_counters WHERE run_id = '{run}'"
    )
    return attempts, int(counted or 0)


@pytest.mark.parametrize(
    "inject",
    [
        {"command": "DELETE FROM ing.test_case_runs"},
        {"command": "DELETE FROM ing.test_case_runs", "error": Interrupted},
    ],
    ids=["delete raises", "process killed before the delete"],
)
def test_a_failure_after_the_insert_converges_on_the_next_run(
    server, ingest_db, inject
):
    run = f"00000000-0000-4000-8000-{abs(hash(str(inject))) % 10**12:012d}"
    ingest(WriterClient(server), ingest_db, run, case_rows(6), 1)
    assert state(server, ingest_db, run) == ({1: 6}, 6)
    with pytest.raises(BaseException):
        ingest(
            WriterClient(Failing(server, **inject)),
            ingest_db,
            run,
            case_rows(5, "failed"),
            2,
        )
    assert (
        2 in state(server, ingest_db, run)[0]
    )  # the new rows landed, the old ones remain
    for _ in range(2):
        ingest(WriterClient(server), ingest_db, run, case_rows(5, "failed"), 2)
        assert state(server, ingest_db, run) == ({2: 5}, 5)


def test_a_later_batch_that_fails_is_finished_by_the_rerun(server, ingest_db):
    run = "00000000-0000-4000-8000-0000000000b1"
    ingest(WriterClient(server), ingest_db, run, case_rows(5003, prefix="old"), 1)
    flaky = Failing(server, table="test_case_runs", after=1)  # the second batch fails
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
    with pytest.raises(RuntimeError):
        ingest(WriterClient(Failing(server, table="test_case_runs")), ingest_db, run,
               case_rows(4, "failed"), 2)  # fmt: skip
    assert state(server, ingest_db, run) == ({1: 4}, 4)


@pytest.mark.parametrize("attempt", [0, 1])
def test_a_first_attempt_that_failed_part_way_is_finished_by_the_rerun(
    server, ingest_db, attempt
):
    run = f"00000000-0000-4000-8000-0000000000c{attempt}"
    flaky = Failing(server, table="test_case_runs", after=1)  # the second batch fails
    with pytest.raises(Exception):
        ingest(WriterClient(flaky), ingest_db, run, case_rows(5003), attempt)
    assert 0 < sum(state(server, ingest_db, run)[0].values()) < 5003
    ingest(WriterClient(server), ingest_db, run, case_rows(5003), attempt)
    assert state(server, ingest_db, run) == ({attempt: 5003}, 5003)


def test_a_replay_after_a_rerun_is_refused_and_doubles_nothing(server, monkeypatch):
    server.command("CREATE DATABASE IF NOT EXISTS rr")
    server.command("DROP TABLE IF EXISTS rr.test_case_runs SYNC")
    server.command(
        "CREATE TABLE rr.test_case_runs (a UInt32, b String) ENGINE = "
        "ReplicatedMergeTree('/t/rr_tcr/{shard}', '{replica}') ORDER BY a"
    )
    monkeypatch.setenv("CH_WRITE_QUORUM", "2")
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.2")
    with pytest.raises(BatchParked):
        writes.insert(WriterClient(server), "test_case_runs", ROWS, ["a", "b"], "rr")
    monkeypatch.delenv("CH_WRITE_QUORUM")
    writes.reset()
    writes.insert(
        WriterClient(server), "test_case_runs", ROWS, ["a", "b"], "rr"
    )  # the re-run
    spool = Spool(os.environ["CH_WRITE_SPOOL"])
    connect = lambda database="": WriterClient(server)  # noqa: E731
    assert spool_cli.replay(spool, None, connect=connect) == 1
    assert server.command("SELECT count() FROM rr.test_case_runs") == 10
    assert len(spool.items()) == 1  # kept for a person, with the reason printed


def test_a_naive_datetime_lands_as_a_direct_insert_would_store_it(server):
    name = replicated(
        server, f"when_{uuid.uuid4().hex[:8]}", "id UInt32, at DateTime64(3)", "id"
    )
    server.command(
        f"CREATE TABLE default.{name}_control (id UInt32, at DateTime64(3)) ENGINE = MergeTree ORDER BY id"
    )
    row = [1, dt.datetime(2026, 11, 13, 1, 2, 3, 456000)]
    writes.insert(WriterClient(server), name, [row], ["id", "at"], "default")
    server.insert(f"default.{name}_control", [row], column_names=["id", "at"])
    read = "SELECT id, at FROM default.%s"
    assert (
        server.query(read % name).result_rows
        == server.query(read % f"{name}_control").result_rows
    )
