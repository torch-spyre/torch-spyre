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

"""Pins the applier's convergence rules: create what is missing, run each migration once,
recreate a changed view, and refuse to touch a drifted table."""

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SchemaApplier, SchemaDrift

DB = "db"


class FakeServer:
    """Holds CREATE statements by name; formatting is whitespace-collapsing."""

    NAME = re.compile(r"^CREATE\s+(?:MATERIALIZED\s+VIEW|TABLE|VIEW)\s+(\w+)", re.I)

    def __init__(self, live=None):
        self.live = dict(live or {})
        self.ledger = {}
        self.log = []

    def command(self, sql):
        self.log.append(sql)
        if sql.startswith("SELECT version()"):
            return "26.3.12.3"
        if sql.startswith("EXISTS TABLE"):
            return int(sql.split(".")[-1] in self.live)
        if sql.startswith("DROP VIEW IF EXISTS"):
            self.live.pop(sql.split()[-1], None)
            return None
        stmt = re.sub(r"IF\s+NOT\s+EXISTS\s+", "", sql, count=1, flags=re.I)
        m = self.NAME.match(stmt)
        if m and m.group(1) not in self.live:
            self.live[m.group(1)] = stmt
        return None

    def query(self, sql, parameters=None):
        if "formatQuerySingleLine" in sql:
            rows = [(" ".join(parameters["s"].split()),)]
        elif "system.tables" in sql:
            rows = list(self.live.items())
        else:
            rows = list(self.ledger.items())

        class R:
            result_rows = rows

        return R()

    def insert(self, table, rows, column_names, database):
        for mid, chk in rows:
            self.ledger[mid] = chk


def _schema(tmp_path, files, migrations=None):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    (tmp_path / "migrations").mkdir()
    for name, text in (migrations or {}).items():
        (tmp_path / "migrations" / name).write_text(text)
    return tmp_path


def _run(server, schema_dir, include=()):
    files = SchemaApplier.selected_files(schema_dir, include)
    migs = SchemaApplier.migration_files(schema_dir)
    return SchemaApplier.apply(server, DB, files, migs)


TABLE = "CREATE TABLE IF NOT EXISTS t (a UInt8) ENGINE = MergeTree ORDER BY a"
VIEW = "CREATE VIEW IF NOT EXISTS v AS SELECT a FROM t"


def test_fresh_database_gets_tables_then_migrations_then_views(tmp_path):
    d = _schema(
        tmp_path,
        {"10-t.sql": TABLE, "50-v.sql": VIEW},
        {"001_x.sql": "INSERT INTO t VALUES (1)"},
    )
    server = FakeServer()
    steps = _run(server, d)
    assert [(a, n) for a, n, _ in steps] == [
        ("create", "t"),
        ("migrate", "001_x.sql"),
        ("create", "v"),
    ]
    assert "001_x.sql" in server.ledger


def test_second_apply_is_a_no_op(tmp_path):
    d = _schema(
        tmp_path, {"10-t.sql": TABLE, "50-v.sql": VIEW}, {"001_x.sql": "SELECT 1"}
    )
    server = FakeServer()
    _run(server, d)
    server.log.clear()
    assert _run(server, d) == []
    assert not any(s.startswith(("DROP", "SELECT 1")) for s in server.log)


def test_changed_view_is_dropped_and_recreated(tmp_path):
    d = _schema(tmp_path, {"10-t.sql": TABLE, "50-v.sql": VIEW})
    server = FakeServer()
    _run(server, d)
    (d / "50-v.sql").write_text(
        "CREATE VIEW IF NOT EXISTS v AS SELECT a + 1 AS a FROM t"
    )
    steps = _run(server, d)
    assert [(a, n) for a, n, _ in steps] == [("recreate", "v")]
    assert "a + 1" in server.live["v"]


def test_drifted_table_fails_without_altering(tmp_path):
    d = _schema(tmp_path, {"10-t.sql": TABLE})
    server = FakeServer(
        {"t": "CREATE TABLE t (a UInt16) ENGINE = MergeTree ORDER BY a"}
    )
    with pytest.raises(SchemaDrift, match="t:"):
        _run(server, d)
    assert not any(s.startswith(("ALTER", "DROP")) for s in server.log)


def test_migration_can_resolve_drift_before_the_check(tmp_path):
    d = _schema(
        tmp_path,
        {"10-t.sql": TABLE},
        {"001_widen.sql": "CREATE TABLE IF NOT EXISTS placeholder (x UInt8)"},
    )
    server = FakeServer(
        {"t": "CREATE TABLE t (a UInt16) ENGINE = MergeTree ORDER BY a"}
    )
    orig = server.command

    def fixing(sql):
        if "placeholder" in sql:
            server.live["t"] = TABLE.replace("IF NOT EXISTS ", "")
        return orig(sql)

    server.command = fixing
    steps = _run(server, d)
    assert ("migrate", "001_widen.sql") in [(a, n) for a, n, _ in steps]


def test_non_create_statement_in_schema_is_rejected(tmp_path):
    d = _schema(tmp_path, {"10-t.sql": TABLE + ";\nALTER TABLE t ADD COLUMN b UInt8"})
    with pytest.raises(ValueError, match="migrations/"):
        _run(FakeServer(), d)


def test_explicit_file_applies_only_when_included(tmp_path):
    d = _schema(
        tmp_path, {"10-t.sql": TABLE, "80-o.sql": "-- APPLY: explicit\n" + VIEW}
    )
    assert [p.name for p, _ in SchemaApplier.selected_files(d)] == ["10-t.sql"]
    included = SchemaApplier.selected_files(d, include=["80-o.sql"])
    assert [p.name for p, _ in included] == ["10-t.sql", "80-o.sql"]


def test_plan_reports_without_executing(tmp_path):
    d = _schema(
        tmp_path, {"10-t.sql": TABLE, "50-v.sql": VIEW}, {"001_x.sql": "SELECT 1"}
    )
    server = FakeServer()
    files = SchemaApplier.selected_files(d)
    steps = SchemaApplier.plan(server, DB, files, SchemaApplier.migration_files(d))
    assert [a for a, _, _ in steps] == ["create", "migrate", "create"]
    assert server.live == {} and server.ledger == {}


def test_repo_schema_parses_as_create_only():
    d = SchemaApplier.schema_dir()
    for path, text in SchemaApplier.selected_files(d, include=["80-otel.sql"]):
        assert SchemaApplier.objects(path, text)
