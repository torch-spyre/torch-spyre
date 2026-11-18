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

"""`results --dry-run` reads the database and writes nothing to it, v1 or v2."""

import pytest

from spyre_clickhouse_ingest import results, schema
from spyre_clickhouse_ingest.writer import DryRunClient

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"
WHEEL = "wheel:apache-tvm-ffi==0.1.14.post1+146f67a53e78"
XML = """<?xml version="1.0"?>
<testsuites><testsuite name="s" tests="2" failures="1" timestamp="2026-10-08T00:00:00">
<testcase classname="T" name="test_a" time="0.1"/>
<testcase classname="T" name="test_b" time="0.2"><failure message="boom"/></testcase>
</testsuite></testsuites>
"""


def _tables(cls=schema.Table):
    for sub in cls.__subclasses__():
        yield sub
        yield from _tables(sub)


class LiveClient:
    """Answers reads as a v2 database holding an older attempt would; records every write."""

    COLUMNS = [(c,) for t in _tables() for c in t.columns]

    def __init__(self):
        self.inserts, self.writes = [], []

    def command(self, sql, parameters=None):
        if sql.split()[0] in ("SELECT", "EXISTS"):
            return 1
        self.writes.append(sql)

    def query(self, sql, parameters=None):
        rows = []
        if "system.columns" in sql:
            rows = self.COLUMNS
        elif "count()" in sql:
            rows = [(int("< {attempt" in sql),)]
        elif not sql.lstrip().startswith(("SELECT", "WITH")):
            self.writes.append(sql)

        class R:
            result_rows = rows

        return R()

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((database, table, len(rows)))


def _run(monkeypatch, tmp_path, schema_gen, *extra):
    xml = tmp_path / "junit.xml"
    xml.write_text(XML)
    live = LiveClient()
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: live)
    # A re-run attempt reaches drop_older_case_attempts' DELETEs and counter rebuild.
    results.main(
        ["--xml-file", str(xml), "--schema", schema_gen, "--strict", "--run-id", RUN,
         "--run-attempt", "2", "--trigger-type", "regression", "--arch", "x86_64",
         "--artifact", WHEEL, "--repository", "torch-spyre/torch-spyre", "--sha", "ab12",
         *extra]
    )  # fmt: skip
    return live


@pytest.mark.parametrize("schema_gen", ["v1", "v2", "both"])
def test_a_dry_run_writes_nothing(monkeypatch, tmp_path, capsys, schema_gen):
    live = _run(monkeypatch, tmp_path, schema_gen, "--dry-run")
    assert (live.inserts, live.writes) == ([], [])
    out = capsys.readouterr().out
    assert "dry run -- nothing written; would write:" in out
    if schema_gen != "v1":
        for table in ("test_cases", "test_case_runs", "artifacts", "artifact_results"):
            assert f"row(s) -> spyre_v2.{table}\n" in out
        assert "DELETE FROM spyre_v2.test_case_runs ..." in out
        assert "INSERT INTO spyre_v2.run_case_counters ..." in out
    if schema_gen != "v2":
        assert "2 row(s) -> test_cases\n" in out


def test_without_dry_run_the_same_ingest_writes(monkeypatch, tmp_path):
    live = _run(monkeypatch, tmp_path, "both")
    tables = {(db, t) for db, t, _ in live.inserts}
    assert {(None, "test_runs"), ("spyre_v2", "test_case_runs")} <= tables
    assert ("spyre_v2", "artifact_results") in tables
    assert any(w.startswith("DELETE FROM spyre_v2.test_case_runs") for w in live.writes)


def test_the_dry_run_client_passes_reads_and_keeps_writes():
    live = LiveClient()
    dry = DryRunClient(live)
    assert dry.command("EXISTS TABLE spyre_v2.test_cases") == 1
    dry.command("ALTER TABLE t DELETE WHERE 1")
    dry.query("INSERT INTO t SELECT 1")
    dry.insert("t", [(1,)], column_names=["a"], database="db")
    assert (live.inserts, live.writes) == ([], [])
    assert dry.rows == {"t": [{"a": 1}]}
    assert dry.counts == {"db.t": 1}
    assert len(dry.statements) == 2
