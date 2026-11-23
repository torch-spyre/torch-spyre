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

"""The applier on a real engine (chdb): the repo schema converges and a Replicated-looking stored
definition compares equal to its file."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SchemaApplier, SchemaObject

session_mod = pytest.importorskip("chdb.session")


class ChdbClient:
    """The slice of the client the applier uses, over an in-process engine."""

    def __init__(self, session, db):
        self.s, self.db = session, db

    def command(self, sql):
        out = self.s.query(sql, "TabSeparated").bytes().decode().strip()
        return int(out) if out.isdigit() else out

    def query(self, sql, parameters=None):
        for k, v in (parameters or {}).items():
            sql = sql.replace(
                f"%({k})s", "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"
            )
        rows = self.s.query(sql, "JSONCompact").data()

        class R:
            result_rows = [tuple(r) for r in json.loads(rows)["data"]]

        return R()

    def insert(self, table, rows, column_names, database):
        for r in rows:
            vals = ", ".join("'" + str(v) + "'" for v in r)
            self.s.query(
                f"INSERT INTO {database}.{table} ({', '.join(column_names)}) VALUES ({vals})"
            )


@pytest.fixture
def real(tmp_path):
    s = session_mod.Session(str(tmp_path / "chdb"))
    s.query("CREATE DATABASE appdb")
    s.query("USE appdb")
    yield ChdbClient(s, "appdb")
    s.close()


def test_the_repo_schema_applies_twice_on_a_real_engine_with_no_change(real):
    d = SchemaApplier.schema_dir()
    files = SchemaApplier.selected_files(d, (), (26, 3))
    migs = SchemaApplier.migration_files(d)
    first = SchemaApplier.apply(real, "appdb", files, migs)
    assert first
    assert SchemaApplier.apply(real, "appdb", files, migs) == []


def test_a_real_table_recreated_with_its_replicated_looking_ddl_compares_equal(real):
    plain = "CREATE TABLE t (`a` UInt8, `u` DateTime) ENGINE = ReplacingMergeTree(u) PARTITION BY toYYYYMM(u) ORDER BY a"
    obj = SchemaObject("table", "t", plain, "x.sql")
    stored = plain.replace(
        "ReplacingMergeTree(u)",
        "ReplicatedReplacingMergeTree('/p/{shard}/appdb/t/g1', '{replica}', u)",
    )
    assert SchemaApplier.differs(real, obj, stored, "appdb") == ""
