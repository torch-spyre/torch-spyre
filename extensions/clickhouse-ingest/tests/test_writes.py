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

"""Replicated inserts through classify/retry/park; plain and v1 inserts keep their path."""

import datetime
import time

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import __main__ as cli
from spyre_clickhouse_ingest import client as chclient
from spyre_clickhouse_ingest import schema, spool_cli, writes
from spyre_clickhouse_ingest.batching import BatchFailed, BatchParked
from spyre_clickhouse_ingest.hw_diagnostics import RecordLoader
from spyre_clickhouse_ingest.hw_schema import HwFailureDiagnostics
from spyre_clickhouse_ingest.resilience import Spool


def err(code):
    return DatabaseError(f"Code: {code}. DB::Exception: x", code=code, name="X")


class Fake:
    """Answers the engine and window lookups; records inserts with their settings."""

    def __init__(self, engine="ReplicatedMergeTree", views=0, window=1000, script=()):
        self.engine, self.views, self.window = engine, views, window
        self.script, self.inserts, self.queries, self.raw = list(script), [], [], []
        self.database = "db"
        self.columns = [("a", "UInt32"), ("b", "String")]

    def query(self, sql, parameters=None):
        self.queries.append(sql)
        if "length(dependencies_table)" in sql:
            rows = [(self.engine, self.views)] if self.engine else []
        elif "engine_full" in sql:
            full = f"SETTINGS replicated_deduplication_window = {self.window}"
            rows = [(self.engine, full)]
        elif "system.columns" in sql:
            rows = list(self.columns)
        else:
            rows = []

        class R:
            result_rows = rows

        return R()

    def insert(self, table, data, column_names=None, database=None, settings=None):
        self.inserts.append((database, table, len(data), dict(settings or {})))
        if self.script:
            raise self.script.pop(0)


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for name in ("CH_WRITE_RESILIENCE", "CH_WRITE_SPOOL", "RUNNER_TEMP", "GITHUB_STEP_SUMMARY",
                 "CH_WRITE_DEADLINE_S", "CH_WRITE_QUORUM", "CH_WRITE_REPLAY_MAX_AGE_S"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "0.001")
    monkeypatch.setenv("CH_WRITE_BACKOFF_MAX_S", "0.001")
    monkeypatch.setattr(writes, "_parked", [])
    monkeypatch.setattr(writes, "_spools", {})
    monkeypatch.setattr(writes, "_calls", {})


ROWS = [[i, f"r{i}"] for i in range(7)]


def test_a_replicated_target_gets_a_token_per_batch():
    c = Fake()
    assert writes.insert(c, "t", ROWS, ["a", "b"], "db", batch_rows=3) == 7
    tokens = [s["insert_deduplication_token"] for *_, s in c.inserts]
    assert len(tokens) == 3 and len(set(tokens)) == 3


def test_a_plain_or_unreadable_target_is_not_handled():
    assert writes.insert(Fake(engine="MergeTree"), "t", ROWS, ["a", "b"], "db") is None
    assert writes.insert(Fake(engine=""), "t", ROWS, ["a", "b"], "db") is None

    class Broken(Fake):
        def query(self, sql, parameters=None):
            raise err(497)

    assert writes.insert(Broken(), "t", ROWS, ["a", "b"], "db") is None


def test_switching_it_off_skips_even_the_lookup(monkeypatch):
    monkeypatch.setenv("CH_WRITE_RESILIENCE", "off")
    c = Fake()
    assert writes.insert(c, "t", ROWS, ["a", "b"], "db") is None
    assert c.queries == [] and c.inserts == []


def test_the_engine_is_looked_up_once_per_client_and_table():
    c = Fake()
    for _ in range(3):
        writes.insert(c, "t", ROWS, ["a", "b"], "db")
    assert sum("length(dependencies_table)" in q for q in c.queries) == 1


def test_a_dependent_view_turns_on_view_dedup():
    c = Fake(views=1)
    writes.insert(c, "t", ROWS, ["a", "b"], "db")
    assert c.inserts[0][3]["deduplicate_blocks_in_dependent_materialized_views"] == 1
    c = Fake(views=0)
    writes.insert(c, "t", ROWS, ["a", "b"], "db")
    assert "deduplicate_blocks_in_dependent_materialized_views" not in c.inserts[0][3]


def test_identical_rows_of_two_calls_are_never_merged_by_their_token():
    c = Fake()
    writes.insert(c, "t", ROWS, ["a", "b"], "db")
    writes.insert(c, "t", ROWS, ["a", "b"], "db")
    tokens = [s["insert_deduplication_token"] for *_, s in c.inserts]
    assert tokens[0] != tokens[1]


def test_a_retry_resends_the_same_token():
    c = Fake(script=[err(285)])
    assert writes.insert(c, "t", ROWS, ["a", "b"], "db") == 7
    first, second = (s["insert_deduplication_token"] for *_, s in c.inserts)
    assert first == second


def test_a_qualified_table_name_is_split():
    c = Fake()
    writes.insert(c, "spyre_v2.hw", ROWS, ["a", "b"])
    assert c.inserts[0][:2] == ("spyre_v2", "hw")


def test_past_the_deadline_without_a_spool_the_batch_fails(monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    with pytest.raises(BatchFailed) as e:
        writes.insert(Fake(script=[err(285)] * 50), "t", ROWS, ["a", "b"], "db")
    assert not isinstance(e.value, BatchParked)
    assert writes.parked() == []


def test_past_the_deadline_with_a_spool_rows_park_and_the_cli_exits_75(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    with pytest.raises(BatchParked) as e:
        writes.insert(Fake(script=[err(285)] * 50), "t", ROWS, ["a", "b"], "db")
    assert e.value.parked == 7 and e.value.landed == 0
    assert Spool(tmp_path / "ch-spool").items()[0]["rows"] == 7
    assert writes.finish(0) == writes.EXIT_PARKED
    assert "db.t: 7 row(s) parked" in summary.read_text()
    assert "r3" not in summary.read_text() + capsys.readouterr().err
    assert writes.finish(0) == 0  # reported once, then forgotten


def test_a_worse_exit_status_is_kept():
    writes._parked.append(("t", 1))
    assert writes.finish(2) == 2


def test_a_full_spool_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    monkeypatch.setattr(
        writes, "_spools", {str(tmp_path / "sp"): Spool(tmp_path / "sp", max_bytes=10)}
    )
    with pytest.raises(BatchFailed):
        writes.insert(Fake(script=[err(285)] * 50), "t", ROWS, ["a", "b"], "db")


def test_a_fatal_error_is_raised_as_it_is():
    with pytest.raises(DatabaseError):
        writes.insert(Fake(script=[err(60)]), "t", ROWS, ["a", "b"], "db")


def test_an_unusable_spool_means_no_parking(monkeypatch, tmp_path, capsys):
    target = tmp_path / "file"
    target.write_text("x")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(target))
    assert writes.spool_from_env() is None
    assert "unusable" in capsys.readouterr().err


def test_table_insert_takes_the_resilient_path_only_for_a_replicated_target():
    rows = [{c: "x" for c in schema.ArtifactTags.columns}]
    c = Fake()
    schema.ArtifactTags.insert(c, rows, db="db")
    assert "insert_deduplication_token" in c.inserts[0][3]
    c = Fake(engine="MergeTree")
    schema.ArtifactTags.insert(c, rows, db="db")
    assert c.inserts[0][3] == {}


def test_the_lookup_and_the_insert_go_to_the_host_that_serves_the_table():
    legacy, v2 = Fake(engine="MergeTree"), Fake()
    routed = chclient.RoutedClient(legacy, lambda: v2, ("spyre_v2",))
    writes.insert(routed, "t", ROWS, ["a", "b"], "spyre_v2")
    assert legacy.queries == [] and legacy.inserts == []
    assert v2.inserts and "insert_deduplication_token" in v2.inserts[0][3]


def test_the_v1_hw_insert_stays_a_plain_insert():
    c = Fake()
    row = [None] * len(HwFailureDiagnostics.COLUMN_NAMES)
    RecordLoader.insert(c, [row], table="hw_failure_diagnostics")
    assert c.queries == [] and c.inserts[0][3] == {}
    RecordLoader.insert(
        c, [row], table="spyre_v2.hw_failure_diagnostics", resilient=True
    )
    assert "insert_deduplication_token" in c.inserts[1][3]


# ── the CLI's exit status ───────────────────────────────────────────────────────


def test_the_cli_turns_a_clean_run_that_parked_rows_into_75(monkeypatch):
    def run(argv):
        writes._parked.append(("t", 1))

    monkeypatch.setattr("spyre_clickhouse_ingest.ci_run_timings.main", run)
    assert cli.main(["ci-run-timings"]) == writes.EXIT_PARKED


def test_the_cli_maps_a_clean_system_exit_after_a_park(monkeypatch):
    def run(argv):
        writes._parked.append(("t", 1))
        raise SystemExit(0)

    monkeypatch.setattr("spyre_clickhouse_ingest.ci_run_timings.main", run)
    with pytest.raises(SystemExit) as e:
        cli.main(["ci-run-timings"])
    assert e.value.code == writes.EXIT_PARKED


def test_the_cli_keeps_its_status_when_nothing_parked(monkeypatch):
    monkeypatch.setattr(
        "spyre_clickhouse_ingest.ci_run_timings.main", lambda argv: None
    )
    assert cli.main(["ci-run-timings"]) is None

    def fail(argv):
        raise SystemExit("boom")

    monkeypatch.setattr("spyre_clickhouse_ingest.ci_run_timings.main", fail)
    with pytest.raises(SystemExit, match="boom"):
        cli.main(["ci-run-timings"])


# ── spool replay ────────────────────────────────────────────────────────────────


def park(tmp_path, ambiguous=False, created=None, rows=None):
    spool = Spool(tmp_path / "sp")
    spool.put(
        "t", ["a", "b"], rows or ROWS, "t-tok", "db",
        ambiguous=ambiguous, created=created, scope="s",
    )  # fmt: skip
    return spool


def test_replay_resends_under_the_original_token_and_drops_the_file(tmp_path):
    spool, c = park(tmp_path), Fake()
    assert spool_cli.replay(spool, "t-tok", connect=lambda database="": c) == 0
    ((database, table, n, settings),) = c.inserts
    assert (database, table, n) == ("db", "t", 7)
    assert settings["insert_deduplication_token"] == "t-tok"
    assert spool.items() == []


def test_replay_restores_the_cell_types_json_flattened(tmp_path):
    when = datetime.datetime(2026, 11, 13, 1, 2, 3, 456000)
    spool = park(tmp_path, rows=[[1, when], [2, None]])
    item = next(spool.pending())
    assert item.rows[0][1] == str(when)
    fake = Fake()
    fake.columns = [("a", "UInt32"), ("b", "Nullable(DateTime64(3))")]
    assert writes._typed(fake, item) == [[1, when], [2, None]]


def test_replay_keeps_the_file_when_the_batch_does_not_land(tmp_path, capsys):
    spool, c = park(tmp_path), Fake(script=[err(285)])
    assert spool_cli.replay(spool, None, connect=lambda database="": c) == 1
    assert len(spool.items()) == 1
    assert "not stored" in capsys.readouterr().out


@pytest.mark.parametrize(
    "fake,ambiguous,age,refused",
    [
        (Fake(engine="MergeTree"), False, 0, "not a Replicated table"),
        (Fake(window=0), False, 0, "window 0"),
        (Fake(window=0), True, 0, "window 0"),
        (Fake(), True, 7200, "h old"),
    ],
)
def test_replay_refuses_a_resend_that_could_double(
    tmp_path, fake, ambiguous, age, refused
):
    spool = park(tmp_path, ambiguous=ambiguous, created=time.time() - age)
    assert writes.replay(fake, next(spool.pending())).find(refused) >= 0
    assert fake.inserts == []


def test_an_ambiguous_batch_is_refused_when_the_window_is_unreadable(tmp_path):
    class NoWindow(Fake):
        def query(self, sql, parameters=None):
            if "engine_full" in sql:
                raise err(497)
            return super().query(sql, parameters)

    spool = park(tmp_path, ambiguous=True)
    assert "unreadable" in writes.replay(NoWindow(), next(spool.pending()))


def test_a_batch_refused_before_any_write_replays_at_any_age(tmp_path):
    spool, c = park(tmp_path, created=time.time() - 10 * 86400), Fake()
    assert writes.replay(c, next(spool.pending())) == "sent"


def test_spool_list_never_prints_rows(tmp_path, monkeypatch, capsys):
    park(tmp_path, rows=[[1, "secret-value"]])
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    assert spool_cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "t-tok" in out and "secret-value" not in out


def test_the_spool_defaults_to_the_runner_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    assert writes.spool_dir() == str(tmp_path / "ch-spool")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "x"))
    assert writes.spool_dir() == str(tmp_path / "x")
