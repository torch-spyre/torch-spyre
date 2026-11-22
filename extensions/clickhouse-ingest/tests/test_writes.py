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
import sys
import time
from pathlib import Path

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
    writes.reset()


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
    assert writes.parked() == {}


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
    writes._parked["t"] = 1
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
        writes._parked["t"] = 1

    monkeypatch.setattr("spyre_clickhouse_ingest.ci_run_timings.main", run)
    assert cli.main(["ci-run-timings"]) == writes.EXIT_PARKED


def test_the_cli_maps_a_clean_system_exit_after_a_park(monkeypatch):
    def run(argv):
        writes._parked["t"] = 1
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
    assert writes._typed(fake, item) == ([[1, when], [2, None]], "")


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


class NoWindow(Fake):
    """A login that may not read system.merge_tree_settings (497), as prod read users."""

    def query(self, sql, parameters=None):
        if "engine_full" in sql:
            self.queries.append(sql)
            raise err(497)
        return super().query(sql, parameters)


def test_an_unreadable_window_is_the_server_default_read_once(tmp_path, capsys):
    c = NoWindow()
    for _ in range(2):
        writes.insert(c, "t", ROWS, ["a", "b"], "db")
    assert "insert_deduplication_token" in c.inserts[0][3]
    assert sum("engine_full" in q for q in c.queries) == 1
    assert capsys.readouterr().err.count("assuming the server default") == 1
    spool = park(tmp_path, ambiguous=True)
    assert writes.replay(NoWindow(), next(spool.pending())) == "sent"


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


# ── one deadline per process ────────────────────────────────────────────────────


def test_after_one_call_spends_its_deadline_later_calls_get_one_attempt(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.05")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    first = Fake(script=[err(285)] * 1000)
    with pytest.raises(BatchParked):
        writes.insert(first, "t", ROWS, ["a", "b"], "db")
    assert len(first.inserts) > 1
    second = Fake(script=[err(285)] * 1000)
    with pytest.raises(BatchParked):
        writes.insert(second, "t", ROWS, ["a", "b"], "db")
    assert len(second.inserts) == 1
    healthy = Fake()
    assert (
        writes.insert(healthy, "v", ROWS, ["a", "b"], "db") == 7
    )  # one try still lands
    writes.reset()
    third = Fake(script=[err(285)] * 1000)
    with pytest.raises(BatchParked):
        writes.insert(third, "t", ROWS, ["a", "b"], "db")
    assert len(third.inserts) > 1


def test_a_failure_without_a_spool_also_spends_the_deadline(monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.05")
    with pytest.raises(BatchFailed):
        writes.insert(Fake(script=[err(285)] * 1000), "t", ROWS, ["a", "b"], "db")
    second = Fake(script=[err(285)] * 1000)
    with pytest.raises(BatchFailed):
        writes.insert(second, "t", ROWS, ["a", "b"], "db")
    assert len(second.inserts) == 1


def test_parked_rows_are_one_entry_per_table(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    for _ in range(3):
        with pytest.raises(BatchParked):
            writes.insert(Fake(script=[err(285)] * 9), "t", ROWS, ["a", "b"], "db")
    assert writes.parked() == {"db.t": 21}


# ── exit status policy ──────────────────────────────────────────────────────────


def test_a_park_beside_written_v1_rows_warns_and_exits_clean(capsys, monkeypatch):
    writes.note_v1_written()
    writes._parked["t"] = 1
    assert writes.finish(0) == 0
    assert "::warning" in capsys.readouterr().out
    writes._parked["t"] = 1
    monkeypatch.setenv("CH_WRITE_STRICT", "1")
    assert writes.finish(0) == writes.EXIT_PARKED


def test_a_v2_only_run_that_parked_exits_75():
    writes._parked["t"] = 1
    assert writes.finish(None) == writes.EXIT_PARKED


@pytest.mark.parametrize("command", ["relay", "bench-verdicts"])
def test_the_relay_and_bench_verdicts_keep_their_own_status(monkeypatch, command):
    module = {"relay": "relay", "bench-verdicts": "bench_verdicts"}[command]

    def run(argv):
        writes._parked["t"] = 1
        return 0

    monkeypatch.setattr(f"spyre_clickhouse_ingest.{module}.main", run)
    assert cli.main([command]) == 0


def test_the_hw_script_exits_75_when_its_v2_rows_are_parked(monkeypatch, tmp_path):
    import runpy

    from spyre_clickhouse_ingest import client as ch
    from spyre_clickhouse_ingest import hw_diagnostics as hw

    script = (
        Path(__file__).resolve().parents[3] / ".github/scripts/ingest_hw_diagnostics.py"
    )

    class Live:
        def command(self, sql, *a, **k):
            return 1

        def query(self, sql, *a, **k):
            class R:
                result_rows = [(0,)]

            return R()

        def insert(self, *a, **k):
            pass

    def parks(client, rows, table="", resilient=False):
        if resilient:
            writes._parked[table] = len(rows)
            raise BatchParked("parked", 0, len(rows))

    monkeypatch.setattr(ch, "get_client", lambda **k: Live())
    monkeypatch.setattr(hw, "load_records", lambda p: [{"suite_name": "s"}])
    monkeypatch.setattr(hw, "filter_suite_records", lambda r: r)
    monkeypatch.setattr(hw, "build_row", lambda rec, ctx: ["row"])
    monkeypatch.setattr(hw, "insert_rows", parks)
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    run = "00000000-0000-4000-8000-000000000001"
    (tmp_path / "x.json").write_text("[]")
    writes.reset()
    monkeypatch.setattr(sys, "argv", [str(script), "--json-file", str(tmp_path / "x.json"),
                        "--run-id", run])  # fmt: skip
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(script), run_name="__main__")
    assert e.value.code == writes.EXIT_PARKED


# ── what replay refuses or restores ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "table", ["test_case_runs", "capability_runs", "artifact_results"]
)
def test_tables_whose_writers_do_more_than_insert_are_never_replayed(tmp_path, table):
    spool = Spool(tmp_path / "sp")
    spool.put(table, ["a"], [[1]], f"{table}-tok", "db")
    c = Fake()
    assert "re-run the ingest" in writes.replay(c, next(spool.pending()))
    assert c.inserts == [] and c.queries == []


def test_a_nested_typed_column_is_refused_not_resent_as_text(tmp_path):
    spool = park(tmp_path)
    fake = Fake()
    fake.columns = [("a", "UInt32"), ("b", "Array(DateTime64(3))")]
    assert "nests Array(DateTime64(3))" in writes.replay(fake, next(spool.pending()))
    assert fake.inserts == []


def test_bytes_the_spool_wrote_as_text_are_refused(tmp_path):
    spool = park(tmp_path, rows=[[1, str(b"\x00raw")]])
    fake = Fake()
    assert "bytes" in writes.replay(fake, next(spool.pending()))
    assert fake.inserts == []


def test_nested_wrappers_are_unwrapped():
    assert writes._unwrap("LowCardinality(Nullable(DateTime64(3, 'UTC')))") == (
        "DateTime64(3, 'UTC')"
    )


def test_a_naive_datetime_is_parked_with_its_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    when = datetime.datetime(2026, 11, 13, 1, 2, 3)
    with pytest.raises(BatchParked):
        writes.insert(Fake(script=[err(285)] * 9), "t", [[1, when]], ["a", "b"], "db")
    item = next(Spool(tmp_path / "ch-spool").pending())
    fake = Fake()
    fake.columns = [("a", "UInt32"), ("b", "DateTime")]
    restored = writes._typed(fake, item)[0][0][1]
    assert restored.tzinfo is not None and restored == when.astimezone()


def test_the_database_of_an_unqualified_insert_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    with pytest.raises(BatchParked):
        writes.insert(
            Fake(script=[err(285)] * 9), "result_invalidations", ROWS, ["a", "b"]
        )
    assert next(Spool(tmp_path / "ch-spool").pending()).database == "db"


def test_the_engine_cache_is_per_database():
    class TwoDbs(Fake):
        def query(self, sql, parameters=None):
            if "length(dependencies_table)" in sql:
                self.engine = (
                    "ReplicatedMergeTree" if parameters["db"] == "v2" else "MergeTree"
                )
            return super().query(sql, parameters)

    c = TwoDbs()
    assert writes.insert(c, "t", ROWS, ["a", "b"], "v2") == 7
    assert writes.insert(c, "t", ROWS, ["a", "b"], "v1") is None


def test_a_stored_insert_ends_the_outage_for_the_rest_of_the_process(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.05")
    with pytest.raises(BatchFailed):
        writes.insert(Fake(script=[err(285)] * 1000), "t", ROWS, ["a", "b"], "db")
    assert writes.insert(Fake(), "t", ROWS, ["a", "b"], "db") == 7  # recovered
    flaky = Fake(script=[err(285)] * 2)
    assert writes.insert(flaky, "t", ROWS, ["a", "b"], "db") == 7  # full retries again
    assert len(flaky.inserts) == 3


def test_a_spent_deadline_expires_after_one_deadline(monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.05")
    with pytest.raises(BatchFailed):
        writes.insert(Fake(script=[err(285)] * 1000), "t", ROWS, ["a", "b"], "db")
    time.sleep(0.06)
    flaky = Fake(script=[err(285)])
    assert writes.insert(flaky, "t", ROWS, ["a", "b"], "db") == 7
    assert len(flaky.inserts) == 2


def test_a_long_lived_threaded_process_recovers_after_an_outage(monkeypatch):
    import threading

    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.05")
    down = {"on": True}

    class Cluster(Fake):
        def insert(self, table, data, column_names=None, database=None, settings=None):
            self.inserts.append(settings)
            if down["on"]:
                raise err(285)

    def request(results):
        try:
            results.append(
                writes.insert(Cluster(), "dispatch_requests", [[1]], ["a"], "db")
            )
        except BatchFailed:
            results.append("failed")

    during: list = []
    threads = [threading.Thread(target=request, args=(during,)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert during == ["failed"] * 8
    down["on"] = False
    after: list = []
    threads = [threading.Thread(target=request, args=(after,)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert after == [1] * 8 and writes._state["spent"] == {}
    tokens = writes._calls[("db", "dispatch_requests")]
    assert tokens == 16  # scope numbers are unique across threads


def test_the_window_is_read_through_get_merge_tree_setting_when_the_table_does_not_say(
    capsys,
):
    class Setting(NoWindow):
        def command(self, sql, *a, **k):
            assert "getMergeTreeSetting('replicated_deduplication_window')" in sql
            return "0"

    assert writes.window(Setting(), "t", "db") == 0
    assert capsys.readouterr().err == ""


def test_bytes_inside_an_array_of_strings_are_refused(tmp_path):
    spool = park(tmp_path, rows=[[1, ["ok", str(b"\x00raw")]]])
    fake = Fake()
    fake.columns = [("a", "UInt32"), ("b", "Array(String)")]
    assert "bytes" in writes.replay(fake, next(spool.pending()))
    assert fake.inserts == []


def test_an_artifact_row_that_parks_fails_the_artifacts_command(monkeypatch):
    from spyre_clickhouse_ingest import artifacts
    from spyre_clickhouse_ingest import client as ch

    def parks(*a, **k):
        writes._parked["spyre_v2.artifacts"] = 1
        raise BatchParked("artifacts: 1 row(s) parked, 0 stored", 0, 1)

    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(ch.ClickHouse, "connect", classmethod(lambda cls, **k: Fake()))
    monkeypatch.setattr(artifacts, "ensure", parks)
    # Not 75: the v2-artifact action reports it through its "not recorded" branch.
    with pytest.raises(BatchParked):
        cli.main(
            ["artifacts", "ensure", "--artifact", "wheel:pkg==1.0", "--arch", "x86_64"]
        )


def test_a_failing_table_does_not_slow_a_healthy_one_and_stays_cheap(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.05")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    sick = Fake(script=[err(252)] * 10000)
    with pytest.raises(BatchParked):
        writes.insert(sick, "sick", ROWS, ["a", "b"], "db")
    for _ in range(5):  # interleaved: healthy stores never reset the sick table's state
        assert writes.insert(Fake(), "healthy", ROWS, ["a", "b"], "db") == 7
        before = len(sick.inserts)
        with pytest.raises(BatchParked):
            writes.insert(sick, "sick", ROWS, ["a", "b"], "db")
        assert len(sick.inserts) - before == 1
    flaky = Fake(script=[err(285)] * 2)
    assert writes.insert(flaky, "healthy", ROWS, ["a", "b"], "db") == 7
    assert len(flaky.inserts) == 3  # the healthy table still gets full retries


def test_bytes_inside_a_map_value_are_refused(tmp_path):
    spool = park(tmp_path, rows=[[1, {"k": str(b"\x00raw")}]])
    fake = Fake()
    fake.columns = [("a", "UInt32"), ("b", "Map(String, String)")]
    assert "bytes" in writes.replay(fake, next(spool.pending()))
    assert fake.inserts == []


def test_the_get_merge_tree_setting_answer_is_cached(capsys):
    class Setting(NoWindow):
        calls = 0

        def command(self, sql, *a, **k):
            Setting.calls += 1
            return "1000"

    c = Setting()
    assert writes.window(c, "t", "db") == writes.window(c, "t", "db") == 1000
    assert Setting.calls == 1
