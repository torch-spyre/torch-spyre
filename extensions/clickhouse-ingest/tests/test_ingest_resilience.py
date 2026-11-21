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

"""A whole results ingest: v1 statements unchanged, v2 tokens, attempt order, withheld verdicts."""

import pytest

from spyre_clickhouse_ingest import results, writes
from spyre_clickhouse_ingest.batching import BatchParked

from test_dry_run import RUN, WHEEL, XML, LiveClient


class Logged(LiveClient):
    """Every statement in order, inserts with their settings; tables answer with ENGINE."""

    ENGINE = "MergeTree"
    database = "spyre"

    def __init__(self):
        super().__init__()
        self.log = []

    def command(self, sql, parameters=None):
        self.log.append(("command", sql))
        return super().command(sql, parameters)

    def query(self, sql, parameters=None):
        self.log.append(("query", sql))
        if "length(dependencies_table)" in sql:

            class R:
                result_rows = [(self.ENGINE, 0)]

            return R()
        return super().query(sql, parameters)

    def insert(self, table, rows, column_names=None, database=None, settings=None):
        self.log.append(("insert", database, table, len(rows), dict(settings or {})))
        super().insert(table, rows, column_names, database)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("CH_WRITE_RESILIENCE", "CH_WRITE_SPOOL", "RUNNER_TEMP"):
        monkeypatch.delenv(name, raising=False)
    writes.reset()


def ingest(monkeypatch, tmp_path, schema_gen, engine="MergeTree", *extra, files=1):
    for n in range(files):
        (tmp_path / f"junit-{n}.xml").write_text(XML.replace("test_", f"test{n}_"))
    live = Logged()
    live.ENGINE = engine
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: live)
    results.main(
        ["--xml-dir", str(tmp_path), "--schema", schema_gen, "--run-id", RUN,
         "--run-attempt", "2", "--trigger-type", "regression", "--arch", "x86_64",
         "--artifact", WHEEL, "--repository", "torch-spyre/torch-spyre", "--sha", "ab12",
         *extra]
    )  # fmt: skip
    return live


def v1_only(log):
    v2 = (
        "spyre_v2",
        "FROM system.tables",
        "merge_tree_settings",
        "getMergeTreeSetting",
    )
    return [e for e in log if not any(m in str(e) for m in v2)]


@pytest.mark.parametrize("engine", ["MergeTree", "ReplicatedMergeTree"])
def test_the_v1_statements_are_those_of_the_path_with_resilience_off(
    monkeypatch, tmp_path, engine
):
    on = ingest(monkeypatch, tmp_path, "v1", engine)
    assert not any("system.tables" in str(e) for e in on.log)
    assert all(e[-1] == {} for e in on.log if e[0] == "insert")
    monkeypatch.setenv("CH_WRITE_RESILIENCE", "off")
    off = ingest(monkeypatch, tmp_path, "v1", engine)
    assert on.log == off.log
    both = ingest(monkeypatch, tmp_path, "both", engine)
    monkeypatch.delenv("CH_WRITE_RESILIENCE")
    both_on = ingest(monkeypatch, tmp_path, "both", engine)
    assert v1_only(both_on.log) == v1_only(both.log)


def test_on_replicated_tables_the_v2_inserts_carry_dedup_tokens(monkeypatch, tmp_path):
    live = ingest(monkeypatch, tmp_path, "both", "ReplicatedMergeTree")
    sent = {e[2]: e[4] for e in live.log if e[0] == "insert" and e[1] == "spyre_v2"}
    for table in ("test_case_runs", "artifact_results"):
        assert "insert_deduplication_token" in sent[table]
    assert all(e[4] == {} for e in live.log if e[0] == "insert" and e[1] != "spyre_v2")


def test_a_rerun_inserts_the_new_attempt_before_it_drops_the_old_one(
    monkeypatch, tmp_path
):
    live = ingest(monkeypatch, tmp_path, "v2")
    order = [
        "insert"
        if e[0] == "insert" and e[2] == "test_case_runs"
        else e[1].split(" (")[0]
        for e in live.log
        if (e[0] == "insert" and e[2] == "test_case_runs")
        or (
            e[0] == "command" and e[1].startswith("DELETE FROM spyre_v2.test_case_runs")
        )
    ]
    assert order and order[0] == "insert" and any(o.startswith("DELETE") for o in order)


def second_file(monkeypatch, error):
    real, seen = results.replace_case_attempt, []

    def flaky(*a, **k):
        seen.append(1)
        if len(seen) == 2:
            raise error
        return real(*a, **k)

    monkeypatch.setattr(results, "replace_case_attempt", flaky)


def test_a_parked_file_withholds_the_runs_verdict(monkeypatch, tmp_path, capsys):
    second_file(monkeypatch, BatchParked("test_case_runs: 2 parked, 0 stored", 0, 2))
    live = ingest(monkeypatch, tmp_path, "both", files=2)
    assert not any(e[0] == "insert" and e[2] == "artifact_results" for e in live.log)
    err = capsys.readouterr()
    assert "verdict withheld" in err.err and "PARKED for 1 file(s)" in err.out


def test_a_failed_file_withholds_the_verdict_and_fails_a_strict_run(
    monkeypatch, tmp_path
):
    second_file(monkeypatch, RuntimeError("down"))
    with pytest.raises(SystemExit):
        ingest(monkeypatch, tmp_path, "both", "MergeTree", "--strict", files=2)


def test_without_a_problem_the_verdict_is_written(monkeypatch, tmp_path):
    live = ingest(monkeypatch, tmp_path, "both", "MergeTree", "--strict", files=2)
    assert any(e[0] == "insert" and e[2] == "artifact_results" for e in live.log)


def test_a_file_with_rows_already_stored_still_takes_the_replace_path(
    monkeypatch, tmp_path
):
    import spyre_clickhouse_ingest as pkg

    calls = []
    monkeypatch.setattr(pkg, "cases_already_ingested", lambda *a, **k: True)
    monkeypatch.setattr(
        results, "replace_case_attempt", lambda *a, **k: calls.append(a) or 0
    )
    ingest(monkeypatch, tmp_path, "v2", files=2)
    assert len(calls) == 2  # a part-stored file is finished, not skipped


PERF_XML = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase classname="benchmark" name="perf_op-alpha_wall_clock_ms_1_8" time="1.0"/>'
    "</testsuite></testsuites>"
)


def perf_ingest(
    monkeypatch, tmp_path, insert_benchmarks, xml=PERF_XML, name="report.xml"
):
    (tmp_path / name).write_text(xml)
    live = Logged()
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: live)
    monkeypatch.setattr(results, "insert_benchmarks", insert_benchmarks)
    results.main(
        ["--xml-dir", str(tmp_path), "--schema", "v2", "--run-id", RUN, "--trigger-type", "perf",
         "--arch", "x86_64", "--artifact", WHEEL, "--repository", "torch-spyre/torch-spyre",
         "--sha", "ab12"]
    )  # fmt: skip
    return live


def test_a_parked_benchmark_report_withholds_the_perf_verdict(
    monkeypatch, tmp_path, capsys
):
    def parks(*a, **k):
        raise BatchParked("benchmark_runs: 1 parked, 0 stored", 0, 1)

    live = perf_ingest(monkeypatch, tmp_path, parks)
    assert not any(e[0] == "insert" and e[2] == "artifact_results" for e in live.log)
    assert "verdict withheld" in capsys.readouterr().err
    live = perf_ingest(monkeypatch, tmp_path, lambda *a, **k: 1)
    assert any(e[0] == "insert" and e[2] == "artifact_results" for e in live.log)


def test_a_both_schema_cli_run_that_parks_exits_clean_with_a_warning(
    monkeypatch, tmp_path, capsys
):
    from spyre_clickhouse_ingest import __main__ as cli

    def parks(*a, **k):
        writes._parked["spyre_v2.test_case_runs"] = 2
        raise BatchParked("test_case_runs: 2 parked, 0 stored", 0, 2)

    monkeypatch.setattr(results, "replace_case_attempt", parks)
    (tmp_path / "junit.xml").write_text(XML)
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: Logged())
    args = ["--xml-dir", str(tmp_path), "--run-id", RUN, "--trigger-type", "regression",
            "--arch", "x86_64"]  # fmt: skip
    assert cli.main(["results", "--schema", "both", *args]) in (0, None)
    assert "::warning" in capsys.readouterr().out
    monkeypatch.setattr(results, "replace_case_attempt", parks)
    assert cli.main(["results", "--schema", "v2", *args]) == writes.EXIT_PARKED


def test_without_an_attempt_a_complete_file_is_skipped_and_a_partial_one_finished(
    monkeypatch, tmp_path
):
    from spyre_clickhouse_ingest.writer import TestResultWriter

    calls = []
    monkeypatch.setattr(
        results, "replace_case_attempt", lambda *a, **k: calls.append(a) or 0
    )
    for stored, replaced in ((2, 0), (1, 1)):  # XML holds two cases
        calls.clear()
        monkeypatch.setattr(
            TestResultWriter, "rows_of_file", classmethod(lambda cls, *a: stored)
        )
        (tmp_path / "junit-0.xml").unlink(missing_ok=True)
        live = Logged()
        (tmp_path / "junit.xml").write_text(XML)
        monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
        monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
        monkeypatch.setattr(results, "get_client", lambda: live)
        results.main(["--xml-dir", str(tmp_path), "--schema", "v2", "--run-id", RUN,
                      "--trigger-type", "regression", "--arch", "x86_64"])  # fmt: skip
        assert len(calls) == replaced, stored


KERNEL_XML = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase classname="kernel_benchmark" name="k" time="1.0"><properties>'
    '<property name="tag" value="kernel__sdpa"/><property name="tag" value="op__op-alpha"/>'
    "</properties></testcase></testsuite></testsuites>"
)


def test_a_parked_kernel_report_withholds_the_perf_verdict(
    monkeypatch, tmp_path, capsys
):
    def parks(*a, **k):
        raise BatchParked("benchmark_runs: 1 parked, 0 stored", 0, 1)

    monkeypatch.setattr(results, "_exit_if_perf_zero", lambda *a: None)
    monkeypatch.setattr(results, "_table_exists", lambda *a, **k: True)
    monkeypatch.setattr(results, "_absent_columns", lambda *a, **k: set())
    live = perf_ingest(monkeypatch, tmp_path, parks, xml=KERNEL_XML, name="kernels.xml")
    out = capsys.readouterr()
    assert "per-kernel" in out.out and "verdict withheld" in out.err
    assert not any(e[0] == "insert" and e[2] == "artifact_results" for e in live.log)


def test_rows_of_file_counts_every_attempt_of_one_file():
    from spyre_clickhouse_ingest.writer import TestResultWriter

    seen = []

    class Count:
        def query(self, sql, parameters=None):
            seen.append((sql, parameters))

            class R:
                result_rows = [(3,)]

            return R()

    assert TestResultWriter.rows_of_file(Count(), "db", RUN, "c", "f.xml") == 3
    ((sql, params),) = seen
    assert "props['source_file'] = {sf:String}" in sql and "run_attempt" not in sql
    assert params == {"component": "c", "run_id": RUN, "sf": "f.xml"}
