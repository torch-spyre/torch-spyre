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

"""`results --purpose` end to end through main(): recorded, or refused/warned, never silently dropped."""

import pytest
from spyre_clickhouse_ingest import results, schema
from spyre_clickhouse_ingest.schema import BENCHMARK_RUNS

PERF = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase classname="op_benchmark" name="perf_op-alpha_wall_clock_ms_1_8" time="1.0"/></testsuite></testsuites>'
)
JUNIT = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" tests="1" timestamp="2026-01-01T00:00:00">'
    '<testcase classname="T" name="test_a" time="0.1"/></testsuite></testsuites>'
)
RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"


def _tables(cls=schema.Table):
    for sub in cls.__subclasses__():
        yield sub
        yield from _tables(sub)


class Client:
    """A v2 database (or one without the benchmark tables) that records writes."""

    def __init__(self, tables=True, ingested=0):
        self.cols = [(c,) for t in _tables() for c in t.columns] if tables else []
        self.tables, self.ingested = tables, ingested
        self.inserts = []

    def command(self, sql, parameters=None):
        return int(self.tables) if sql.split()[0] in ("SELECT", "EXISTS") else None

    def query(self, sql, parameters=None):
        rows = (
            self.cols
            if "system.columns" in sql
            else [(self.ingested,)]
            if "count()" in sql
            else []
        )

        class R:
            result_rows = rows

        return R()

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows))


def ingest(monkeypatch, tmp_path, xml, client, *args, db_v2="spyre_v2", tier="perf"):
    (tmp_path / "r.xml").write_text(xml)
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.delenv("INGEST_SCHEMA", raising=False)
    if db_v2:
        monkeypatch.setenv("CLICKHOUSE_DB_V2", db_v2)
    else:
        monkeypatch.delenv("CLICKHOUSE_DB_V2", raising=False)
    monkeypatch.setattr(results, "get_client", lambda: client)
    argv = [
        "--xml-dir",
        str(tmp_path),
        "--run-id",
        RUN,
        "--arch",
        "x86_64",
        "--trigger-type",
        tier,
        *args,
    ]
    return results.main(argv)


def purposes(client):
    out = []
    for table, rows in client.inserts:
        if table == BENCHMARK_RUNS.name:
            out += [
                r[BENCHMARK_RUNS.columns.index("props")].get("run.purpose")
                for r in rows
            ]
    return out


def test_a_perf_report_records_the_purpose(monkeypatch, tmp_path):
    c = Client()
    ingest(monkeypatch, tmp_path, PERF, c, "--purpose", "Nightly", "--schema", "v2")
    assert purposes(c) == ["nightly"]


@pytest.mark.parametrize(
    "case, kwargs, extra",
    [
        ("explicit v1", {}, ["--schema", "v1"]),
        ("CLICKHOUSE_DB_V2 unset", {"db_v2": ""}, ["--schema", "v2"]),
        (
            "v2 benchmark tables absent",
            {"client": Client(tables=False)},
            ["--schema", "v2"],
        ),
    ],
)
def test_a_purpose_that_cannot_reach_the_rows_exits_2(
    monkeypatch, tmp_path, capsys, case, kwargs, extra
):
    c = kwargs.pop("client", Client())
    with pytest.raises(SystemExit) as err:
        ingest(monkeypatch, tmp_path, PERF, c, "--purpose", "nightly", *extra, **kwargs)
    assert err.value.code == 2, case
    assert "--purpose cannot be recorded" in capsys.readouterr().err
    assert c.inserts == []


def test_a_plain_test_report_warns_that_the_purpose_was_not_recorded(
    monkeypatch, tmp_path, capsys
):
    c = Client()
    ingest(
        monkeypatch,
        tmp_path,
        JUNIT,
        c,
        "--purpose",
        "nightly",
        "--schema",
        "v2",
        tier="regression",
    )
    assert "not recorded for this file" in capsys.readouterr().err
    assert purposes(c) == []


def test_an_already_ingested_report_warns_that_the_purpose_was_not_recorded(
    monkeypatch, tmp_path, capsys
):
    c = Client(ingested=1)
    ingest(monkeypatch, tmp_path, PERF, c, "--purpose", "nightly", "--schema", "v2")
    err = capsys.readouterr().err
    assert "not recorded for this file" in err and "already ingested" in err
    assert purposes(c) == []


def test_without_a_purpose_nothing_is_refused_or_warned(monkeypatch, tmp_path, capsys):
    c = Client()
    ingest(monkeypatch, tmp_path, PERF, c, "--schema", "v2")
    assert "--purpose" not in capsys.readouterr().err
    assert purposes(c) == [None]


EMPTY_PERF = '<?xml version="1.0"?><testsuites><testsuite name="spyre-perf-suite" timestamp="2026-01-01T00:00:00Z"/></testsuites>'


def test_a_directory_whose_files_write_no_perf_row_exits_2(
    monkeypatch, tmp_path, capsys
):
    c = Client()
    with pytest.raises(SystemExit) as err:
        ingest(
            monkeypatch,
            tmp_path,
            EMPTY_PERF,
            c,
            "--purpose",
            "nightly",
            "--schema",
            "v2",
            tier="regression",
        )
    assert err.value.code == 2
    assert "recorded nowhere" in capsys.readouterr().err
    assert purposes(c) == []


@pytest.mark.parametrize("flag", ["--bench-dir", "--reference-dir"])
def test_a_purpose_is_refused_for_the_bench_and_reference_inputs(
    monkeypatch, tmp_path, flag
):
    with pytest.raises(SystemExit) as err:
        ingest(
            monkeypatch,
            tmp_path,
            PERF,
            Client(),
            "--purpose",
            "nightly",
            flag,
            str(tmp_path),
        )
    assert "refusing to drop it" in str(err.value)


def test_a_reference_dir_is_not_ingested_directly(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as err:
        ingest(monkeypatch, tmp_path, PERF, Client(), "--reference-dir", str(tmp_path))
    assert "recorded at approval" in str(err.value)
