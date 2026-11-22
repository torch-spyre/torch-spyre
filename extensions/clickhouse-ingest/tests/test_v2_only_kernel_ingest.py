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

"""A v2-only kernel report is written without the v1 kernel tables."""

import pytest

from test_bench_purpose_main import JUNIT, PERF, Client, ingest

from spyre_clickhouse_ingest import results
from spyre_clickhouse_ingest.schema import BENCHMARK_RUNS

KERNEL = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase classname="kernel_benchmark" name="k" time="1.0"><properties>'
    '<property name="tag" value="kernel__matmul"/><property name="tag" value="op__mm"/>'
    "</properties></testcase></testsuite></testsuites>"
)


def _written(client):
    return sum(
        len(rows) for table, rows in client.inserts if table == BENCHMARK_RUNS.name
    )


def test_a_kernel_report_lands_in_v2_when_the_v1_kernel_tables_are_absent(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(results, "_table_exists", lambda *a: False)
    monkeypatch.setattr(results, "_absent_columns", lambda *a: ("run_type",))
    c = Client()
    ingest(monkeypatch, tmp_path, KERNEL, c, "--schema", "v2", tier="regression")
    assert _written(c) == 1


def test_a_v1_kernel_write_still_needs_the_v1_kernel_tables(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(results, "_table_exists", lambda *a: False)
    monkeypatch.setattr(results, "_absent_columns", lambda *a: ())
    c = Client()
    ingest(monkeypatch, tmp_path, KERNEL, c, "--schema", "both", tier="regression")
    assert _written(c) == 0 and "no perf_kernels table" in capsys.readouterr().err


class NoProbe(Client):
    """A login with no SHOW TABLES grant: EXISTS TABLE is denied, system.columns lists nothing."""

    def __init__(self):
        super().__init__(tables=False)

    def command(self, sql, parameters=None):
        if sql.startswith("EXISTS"):
            raise RuntimeError("ACCESS_DENIED")
        return super().command(sql, parameters)


@pytest.mark.parametrize("strict, code", [(["--strict"], 1), ([], 0)])
def test_a_denied_benchmark_probe_fails_a_v2_only_perf_report_under_strict(
    monkeypatch, tmp_path, capsys, strict, code
):
    monkeypatch.delenv("CLICKHOUSE_V2_HOST", raising=False)
    try:
        ingest(monkeypatch, tmp_path, PERF, NoProbe(), *strict, "--schema", "v2")
        rc = 0
    except SystemExit as e:
        rc = e.code
    err = capsys.readouterr().err
    assert rc == code and "benchmark gate failed" in err
    assert "not written" in err


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_a_failed_v2_write_exits_only_under_strict(
    monkeypatch, tmp_path, capsys, flags, code
):
    def boom(*a, **k):
        raise RuntimeError("denied")

    monkeypatch.setattr(results, "replace_case_attempt", boom)
    try:
        ingest(
            monkeypatch,
            tmp_path,
            JUNIT,
            Client(),
            *flags,
            "--schema",
            "v2",
            tier="regression",
        )
        rc = 0
    except SystemExit as e:
        rc = e.code
    assert rc == code
    assert "v2 write failed" in capsys.readouterr().err


def test_the_default_schema_is_v2(monkeypatch):
    monkeypatch.delenv("INGEST_SCHEMA", raising=False)
    assert results.build_parser().parse_args(["--xml-dir", "x"]).schema == "v2"


def _code(monkeypatch, tmp_path, xml, client, *args, **kw):
    try:
        ingest(monkeypatch, tmp_path, xml, client, *args, **kw)
        return 0
    except SystemExit as e:
        return e.code


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_missing_functional_tables_are_a_failed_v2_write_not_a_silent_skip(
    monkeypatch, tmp_path, capsys, flags, code
):
    rc = _code(
        monkeypatch,
        tmp_path,
        JUNIT,
        Client(tables=False),
        *flags,
        "--schema",
        "v2",
        tier="regression",
    )
    assert rc == code
    assert "test_cases, test_case_runs) tables are absent" in capsys.readouterr().err


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_an_underivable_run_id_is_a_failed_v2_write(
    monkeypatch, tmp_path, capsys, flags, code
):
    monkeypatch.setattr(results, "run_id_for", lambda *a, **k: "")
    rc = _code(
        monkeypatch,
        tmp_path,
        JUNIT,
        Client(),
        *flags,
        "--schema",
        "v2",
        tier="regression",
    )
    assert rc == code
    assert "run_id not derivable" in capsys.readouterr().err


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_schema_v2_without_a_v2_database_fails_under_strict(
    monkeypatch, tmp_path, flags, code
):
    rc = _code(
        monkeypatch,
        tmp_path,
        JUNIT,
        Client(),
        *flags,
        "--schema",
        "v2",
        tier="regression",
        db_v2="",
    )
    assert rc == code


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_a_refused_benchmark_insert_is_a_failed_v2_write(
    monkeypatch, tmp_path, capsys, flags, code
):
    from spyre_clickhouse_ingest.batching import BatchFailed

    def refuse(*a, **k):
        raise BatchFailed("refused", 0, 1)

    monkeypatch.setattr(results, "insert_benchmarks", refuse)
    rc = _code(monkeypatch, tmp_path, PERF, Client(), *flags, "--schema", "v2")
    assert rc == code
    assert "v2 write failed" in capsys.readouterr().err


@pytest.mark.parametrize("report", [KERNEL, PERF], ids=["kernel", "benchmark"])
@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_a_failed_dedup_read_is_a_failed_v2_write_not_a_crash(
    monkeypatch, tmp_path, capsys, report, flags, code
):
    def unreadable(*a, **k):
        raise RuntimeError("read denied")

    monkeypatch.setattr(results, "benchmarks_already_ingested", unreadable)
    rc = _code(
        monkeypatch, tmp_path, report, Client(), *flags, "--schema", "v2",
        tier="regression" if report is KERNEL else "perf",
    )  # fmt: skip
    cap = capsys.readouterr()
    assert rc == code and "v2 dedup read failed" in cap.err
    assert "Inserted" not in cap.out
    assert ("::warning title=ClickHouse::v2 write failed" in cap.out) is (code == 0)


@pytest.mark.parametrize("report", [KERNEL, PERF], ids=["kernel", "benchmark"])
def test_an_unavailable_v2_prints_no_inserted_line(
    monkeypatch, tmp_path, capsys, report
):
    monkeypatch.setattr(results, "benchmark_tables_present", lambda *a: False)
    _code(
        monkeypatch, tmp_path, report, Client(), "--schema", "v2",
        tier="regression" if report is KERNEL else "perf",
    )  # fmt: skip
    cap = capsys.readouterr()
    assert "Inserted" not in cap.out and "rows stored" in cap.out


@pytest.mark.parametrize(
    "xml,tier,kind",
    [(KERNEL, "regression", "kernel"), (PERF, "perf", "benchmark")],
)
def test_a_stored_v2_only_report_is_counted_as_stored(
    monkeypatch, tmp_path, capsys, xml, tier, kind
):
    c = Client()
    ingest(monkeypatch, tmp_path, xml, c, "--schema", "v2", tier=tier)
    out = capsys.readouterr().out
    assert _written(c) == 1
    assert f"Inserted 1 {kind} rows" in out
    assert f"No {kind} rows stored" not in out
