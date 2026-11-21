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

"""Wiring of resilient_insert into every ingest write path, with fake clients."""

import os
import stat

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import hw_diagnostics, probe, results, writes
from spyre_clickhouse_ingest.batching import BatchFailed, BatchParked, insert_chunks
from spyre_clickhouse_ingest.resilience import Spool
from spyre_clickhouse_ingest.schema import TEST_CASES


def err(code):
    return DatabaseError(f"Code: {code}. x", code=code, name="X")


class Result:
    def __init__(self, rows):
        self.result_rows = rows


class Fake:
    """Answers the engine lookup; stores inserts once per token, like a Replicated table."""

    database = "db"

    def __init__(self, engine="ReplicatedMergeTree", views=0, script=()):
        self.engine, self.views, self.script = engine, views, list(script)
        self.inserts, self.lookups, self.rows, self.tokens = [], 0, [], set()
        self.commands = []

    def command(self, sql, parameters=None, **kw):
        self.commands.append(sql)
        return 0

    def query(self, sql, parameters=None):
        if "system.tables" in sql:
            self.lookups += 1
            return Result([(self.engine, self.views)])
        if "count()" in sql:
            return Result([(len(self.rows),)])
        return Result([])

    def insert(self, table, data, column_names=None, database=None, settings=None):
        self.inserts.append((table, len(data), dict(settings or {})))
        if self.script:
            raise self.script.pop(0)
        token = (settings or {}).get("insert_deduplication_token")
        if token and token in self.tokens:
            return
        self.tokens.add(token)
        self.rows.extend(data)


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for name in (
        "CH_WRITE_RESILIENCE",
        "CH_WRITE_SPOOL",
        "CH_WRITE_SCOPE",
        "CH_WRITE_DEGRADE_TO",
        "CH_WRITE_DEADLINE_S",
        "RUNNER_TEMP",
        "WORKSPACE",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "CH_WRITE_LEG",
        "CH_WRITE_ASSUME_REPLICATED",
        "BUILD_TAG",
        "GITHUB_STEP_SUMMARY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0")
    monkeypatch.setattr(writes, "_parked", [])
    monkeypatch.setattr(writes, "_spools", {})
    monkeypatch.setattr(writes, "_process_scope", "")
    monkeypatch.setattr("time.sleep", lambda s: None)


ROWS = [[i, f"r{i}"] for i in range(7)]


def test_replicated_target_gets_a_token_per_batch():
    c = Fake()
    assert insert_chunks(c, "t", ROWS, ["a", "b"], size=3) == 7
    assert len(c.inserts) == 3
    toks = {s["insert_deduplication_token"] for _, _, s in c.inserts}
    assert len(toks) == 3


def test_plain_target_keeps_the_unchanged_path():
    c = Fake(engine="MergeTree")
    insert_chunks(c, "t", ROWS, ["a", "b"], size=3)
    assert all(s == {} for _, _, s in c.inserts)


def test_unreadable_engine_keeps_the_unchanged_path():
    class Blind(Fake):
        def query(self, sql, parameters=None):
            raise RuntimeError("denied")

    c = Blind()
    insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.inserts[0][2] == {}


def test_switch_off_skips_even_the_lookup(monkeypatch):
    monkeypatch.setenv("CH_WRITE_RESILIENCE", "off")
    c = Fake()
    insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.lookups == 0 and c.inserts[0][2] == {}


def test_engine_is_looked_up_once_per_table():
    c = Fake()
    for _ in range(3):
        insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.lookups == 2  # the engine facts and the dedup window, once each


def test_dependent_view_turns_on_view_dedup_setting():
    plain = Fake(views=0)
    with_view = Fake(views=1)
    insert_chunks(plain, "t", ROWS, ["a", "b"])
    insert_chunks(with_view, "t", ROWS, ["a", "b"])
    key = "deduplicate_blocks_in_dependent_materialized_views"
    assert key not in plain.inserts[0][2]
    assert with_view.inserts[0][2][key] == 1


def test_same_scope_and_rows_send_the_same_token_across_calls(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    a, b = Fake(), Fake()
    insert_chunks(a, "t", ROWS, ["a", "b"])
    insert_chunks(b, "t", ROWS, ["a", "b"])
    assert a.inserts[0][2] == b.inserts[0][2]
    monkeypatch.setenv("GITHUB_RUN_ID", "43")
    c = Fake()
    insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.inserts[0][2] != a.inserts[0][2]


def test_retry_then_success_sends_one_token():
    c = Fake(script=[err(285), err(285)])
    assert insert_chunks(c, "t", ROWS, ["a", "b"]) == 7
    assert len({s["insert_deduplication_token"] for _, _, s in c.inserts}) == 1
    assert len(c.rows) == 7


def test_deadline_without_spool_raises_batch_failed(monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    c = Fake(script=[err(285)] * 5)
    with pytest.raises(BatchFailed):
        insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.rows == []


def test_deadline_with_spool_parks_and_exit_code_is_three(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    c = Fake(script=[err(285)] * 5)
    with pytest.raises(BatchParked) as e:
        insert_chunks(c, "t", ROWS, ["a", "b"])
    assert (e.value.landed, e.value.parked) == (0, 7)
    assert writes.parked() == [("t", 7)]
    assert len(Spool(tmp_path / "sp").items()) == 1
    assert writes.exit_code(0) == writes.EXIT_PARKED
    assert writes.exit_code(1) == 1


def test_full_spool_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    monkeypatch.setitem(
        writes._spools, str(tmp_path / "sp"), Spool(tmp_path / "sp", max_bytes=10)
    )
    c = Fake(script=[err(285)] * 5)
    with pytest.raises(BatchFailed, match="spool"):
        insert_chunks(c, "t", ROWS, ["a", "b"])
    assert writes.parked() == []


def test_fatal_error_surfaces_as_batch_failed_with_the_stored_count():
    c = Fake(script=[None, err(60)])  # second batch is fatal
    c.script = []

    class Second(Fake):
        def insert(self, table, data, column_names=None, database=None, settings=None):
            if self.inserts:
                raise err(60)
            super().insert(table, data, column_names, database, settings)

    c = Second()
    with pytest.raises(BatchFailed) as e:
        insert_chunks(c, "t", ROWS, ["a", "b"], size=3)
    assert e.value.landed == 3 and not isinstance(e.value, BatchParked)
    assert "60" in str(e.value) and "r0" not in str(e.value)


def test_only_degradable_tables_may_lower_the_quorum(monkeypatch):
    monkeypatch.setenv("CH_WRITE_QUORUM", "2")  # lowering needs a quorum to lower from
    monkeypatch.setenv("CH_WRITE_DEGRADE_TO", "1")
    monkeypatch.setenv("CH_WRITE_DEGRADE_AFTER_S", "0")
    for name, lowered in (("pipeline_runs", True), ("test_case_runs", False)):
        c = Fake()
        insert_chunks(c, name, ROWS, ["a", "b"])
        assert (c.inserts[0][2].get("insert_quorum") == "1") is lowered, name


def test_strict_table_sends_no_quorum_unless_configured():
    c = Fake()
    insert_chunks(c, "artifact_results", ROWS, ["a", "b"])
    assert "insert_quorum" not in c.inserts[0][2]


def test_table_insert_and_direct_v1_helpers_share_the_path():
    c = Fake()
    TEST_CASES.insert(c, [], db="db")
    row = {k: ("x" if k != "tags" else []) for k in TEST_CASES.columns}
    row["test_case_id"] = "00000000-0000-0000-0000-000000000001"
    TEST_CASES.insert(c, [row], db="db")
    assert c.inserts[0][2]["insert_deduplication_token"]
    d = Fake()
    hw_diagnostics.RecordLoader.insert(d, [])  # empty: nothing sent
    writes.insert(d, "perf_kernels", [[1]], ["a"])
    assert d.inserts[-1][2]["insert_deduplication_token"]


class NoDirectInsert(Fake):
    """Replicated engine whose only legal write door is a resilient (settings-bearing) insert."""

    def insert(self, table, data, column_names=None, database=None, settings=None):
        assert settings and "insert_deduplication_token" in settings, "direct insert"
        super().insert(table, data, column_names, database, settings)


def test_direct_v1_sites_never_insert_directly_on_a_replicated_table():
    from datetime import datetime

    c = NoDirectInsert()
    results.writes.insert(c, "perf_kernels", [[1]], ["a"])
    results.writes.insert(c, "test_runs", [[1]], ["a"])
    hw_diagnostics.RecordLoader.insert(
        c, [[None] * len(hw_diagnostics.HwFailureDiagnostics.COLUMN_NAMES)]
    )
    assert len(c.inserts) == 3 and datetime  # each went through the token path


def test_writes_insert_on_plain_table_is_the_plain_call():
    p = Fake(engine="MergeTree")
    writes.insert(p, "test_runs", [[1]], ["a"])
    assert p.inserts == [("test_runs", 1, {})]


# ── spool location and permissions ──────────────────────────────────────────────


def test_spool_location_order(monkeypatch, tmp_path):
    assert writes.spool_dir() == ""
    monkeypatch.setenv("WORKSPACE", str(tmp_path / "ws"))
    assert writes.spool_dir() == str(tmp_path / "ws" / ".ch-spool")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "rt"))
    assert writes.spool_dir() == str(tmp_path / "rt" / "ch-spool")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "x"))
    assert writes.spool_dir() == str(tmp_path / "x")


def test_spool_is_private(tmp_path):
    s = Spool(tmp_path / "sp")
    f = s.put("t", ["a"], [[1]], "tok")
    assert stat.S_IMODE(os.stat(tmp_path / "sp").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600


def test_spool_refuses_a_symlinked_directory(tmp_path):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "link")
    with pytest.raises(OSError):
        Spool(tmp_path / "link")


def test_unusable_spool_degrades_to_no_spool(monkeypatch, tmp_path, capsys):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "link")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "link"))
    assert writes.spool_from_env() is None
    assert "parking disabled" in capsys.readouterr().err


def test_report_names_tables_and_writes_the_step_summary(monkeypatch, tmp_path, capsys):
    summary = tmp_path / "sum.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    writes._parked.extend([("t", 3), ("t", 4), ("u", 1)])
    writes.report_parked()
    err_text = capsys.readouterr().err
    assert "t: 7 row(s)" in err_text and "u: 1 row(s)" in err_text
    assert "t: 7 row(s)" in summary.read_text()
    assert writes.parked() == []


# ── probe ───────────────────────────────────────────────────────────────────────


class ProbeFake(Fake):
    def query(self, sql, parameters=None):
        if "probe_id" in sql:
            return Result([(len(self.rows),)])
        return super().query(sql, parameters)


def test_probe_replicated_round_trip_holds_the_token():
    c = ProbeFake()
    r = probe.probe(c, "db.zz_write_probe", rows=3, sleep=lambda s: None)
    assert r["ok"] and r["dedup"] == "held" and r["count"] == 3
    assert len(c.inserts) == 2
    assert (
        c.inserts[0][2]["insert_deduplication_token"]
        == c.inserts[1][2]["insert_deduplication_token"]
    )


def test_probe_reports_a_doubled_resend():
    class Doubling(ProbeFake):
        def insert(self, table, data, column_names=None, database=None, settings=None):
            self.rows.extend(data)

    r = probe.probe(Doubling(), "db.zz_write_probe", rows=2, sleep=lambda s: None)
    assert not r["ok"] and r["dedup"] == "doubled"


def test_probe_on_a_plain_table_skips_the_resend():
    c = ProbeFake(engine="MergeTree")
    r = probe.probe(c, "db.zz_write_probe", rows=2, sleep=lambda s: None)
    assert r["ok"] and r["dedup"] == "skipped" and len(c.inserts) == 1


def test_probe_reports_a_missing_table_without_raising():
    class Missing(Fake):
        def query(self, sql, parameters=None):
            return Result([])

    r = probe.probe(Missing(), "db.nope", sleep=lambda s: None)
    assert not r["ok"] and "error" in r


def test_probe_times_out_when_the_read_never_catches_up():
    class Lagging(ProbeFake):
        def query(self, sql, parameters=None):
            if "probe_id" in sql:
                return Result([(0,)])
            return super().query(sql, parameters)

    ticks = iter(range(0, 1000, 20))
    r = probe.probe(
        Lagging(), "db.p", rows=2, read_timeout_s=30, sleep=lambda s: None,
        clock=lambda: next(ticks),
    )  # fmt: skip
    assert not r["ok"] and r["count"] == 0


# ── token scope ─────────────────────────────────────────────────────────────────


def test_scope_names_attempt_and_job(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_JOB", "unit")
    a = writes.ambient_scope()
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    b = writes.ambient_scope()
    monkeypatch.setenv("CH_WRITE_LEG", "py312")
    c = writes.ambient_scope()
    assert len({a, b, c}) == 3


def test_empty_scope_is_per_process_not_empty(monkeypatch):
    first = writes.ambient_scope()
    assert first and writes.ambient_scope() == first
    monkeypatch.setattr(writes, "_process_scope", "")
    assert writes.ambient_scope() != first


# ── engine lookup under failure ─────────────────────────────────────────────────


class Flaky(Fake):
    def __init__(self, fail, **kw):
        super().__init__(**kw)
        self.fail = fail

    def query(self, sql, parameters=None):
        if "system.tables" in sql and self.fail:
            self.fail -= 1
            raise ConnectionRefusedError("down")
        return super().query(sql, parameters)


def test_lookup_retries_a_connection_failure_instead_of_going_plain():
    c = Flaky(fail=2)
    insert_chunks(c, "t", ROWS, ["a", "b"])
    assert "insert_deduplication_token" in c.inserts[0][2]


def test_lookup_that_never_answers_goes_plain_unless_assumed(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "5")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "2")
    c = Flaky(fail=99)
    insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.inserts[0][2] == {}
    monkeypatch.setenv("CH_WRITE_ASSUME_REPLICATED", "db.t")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    d = Flaky(fail=99)
    d.ch_host = "other-host"  # c's host is down; d's breaker is its own
    with pytest.raises(
        BatchParked
    ):  # assumed Replicated, and the cluster is down: parked
        insert_chunks(d, "t", ROWS, ["a", "b"])
    assert d.inserts == [] and Spool(tmp_path / "sp").items()[0]["table"] == "t"
    asked = d.lookups_failed = d.fail
    with pytest.raises(BatchParked):
        insert_chunks(d, "t", ROWS, ["a", "b"])
    assert d.fail == asked  # a recent failed lookup is not retried until its TTL passes


def test_fatal_lookup_error_is_not_retried(monkeypatch):
    class Denied(Fake):
        def query(self, sql, parameters=None):
            self.lookups += 1
            raise err(497)

    c = Denied()
    insert_chunks(c, "t", ROWS, ["a", "b"])
    assert c.lookups == 1 and c.inserts[0][2] == {}


# ── parked rows must stop the paths that would delete or count ──────────────────

UUID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def parking(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))


def test_artifact_verdict_keeps_the_old_attempt_when_the_new_one_parks(parking):
    from spyre_clickhouse_ingest.writer import ArtifactWriter

    kw = dict(
        artifact_id=UUID, run_id=UUID, test_type="unit", state="passed", arch="x86_64",
        attempt=2,
    )  # fmt: skip
    bad = Fake(script=[err(285)] * 5)
    with pytest.raises(BatchParked):
        ArtifactWriter.insert_result(bad, "db", **kw)
    assert not [c for c in bad.commands if c.startswith("DELETE")]
    good = Fake()
    ArtifactWriter.insert_result(good, "db", **kw)
    assert [c.split()[0] for c in good.commands] == ["DELETE"] and good.inserts


def test_superseded_delete_and_counter_rebuild_are_skipped_when_rows_park(
    parking, monkeypatch
):
    from spyre_clickhouse_ingest.writer import TestResultWriter

    monkeypatch.setattr(
        TestResultWriter, "_one_per_case", lambda self, *a, **k: (a[-1], ["u-1"]),
        raising=False,
    )  # fmt: skip
    case = {
        "classname": "k", "name": "n", "status": "passed", "duration_s": 1.0,
        "fail_message": "", "properties": [],
    }  # fmt: skip
    kw = dict(arch="x86_64")
    bad = Fake(script=[err(285)] * 9)
    with pytest.raises(BatchParked):
        TestResultWriter.insert(
            bad, "db", "comp", UUID, [case], "f.xml", attempt=1, **kw
        )
    assert bad.commands == []
    good = Fake()
    TestResultWriter.insert(good, "db", "comp", UUID, [case], "f.xml", attempt=1, **kw)
    assert good.commands and good.commands[0].startswith("DELETE")
    assert any(c.startswith("INSERT INTO db.run_case_counters") for c in good.commands)


def test_leg_discriminator_separates_matrix_legs_of_one_job(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_JOB", "ingest")
    monkeypatch.setenv("CH_WRITE_LEG", "aaaa")
    a = writes.ambient_scope()
    monkeypatch.setenv("CH_WRITE_LEG", "bbbb")
    assert writes.ambient_scope() != a


def test_the_window_lookup_is_cached_and_a_failure_is_remembered_briefly():
    class Q:
        database = "d"
        asked = 0

        def query(self, sql, parameters=None):
            Q.asked += 1
            raise err(210)

    q = Q()
    assert writes.window_of(q, "t", "d") is None
    assert writes.window_of(q, "t", "d") is None
    assert Q.asked == 1

    class Plain:
        database = "d"

        def query(self, sql, parameters=None):
            class R:
                result_rows = [("MergeTree", "")]

            return R()

    from spyre_clickhouse_ingest.resilience import dedup_window

    assert dedup_window(Plain(), "t", "d") == 0
