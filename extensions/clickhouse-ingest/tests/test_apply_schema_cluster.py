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

"""Replicated-table support in the applier: a converted table is not drift, real drift still is,
and cluster mode puts DDL on every replica exactly once."""

from pathlib import Path

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SchemaApplier, SchemaDrift

FIX = Path(__file__).parent / "fixtures" / "live_ddl"
CLUSTER = "test_cluster"
PATH = "/clickhouse/test_cluster/tables/{shard}"


def live_ddl(name: str) -> str:
    return (FIX / f"{name}.sql").read_text().strip()


def replicated(ddl: str, db: str, table: str, suffix: str = "") -> str:
    """What a conversion does to a stored CREATE: engine swapped, keeper path and replica added."""
    m = re.search(r"ENGINE = (\w+)(?:\(([^)]*)\))?", ddl)
    args = m.group(2)
    new = f"'/ch/tables/{{shard}}/{db}/{table}{suffix}', '{{replica}}'" + (
        f", {args}" if args else ""
    )
    return ddl[: m.start()] + f"ENGINE = Replicated{m.group(1)}({new})" + ddl[m.end() :]


class FormatOnly:
    """Just enough of a client for canonical(): whitespace-collapsing, like the other tests."""

    def query(self, sql, parameters=None):
        class R:
            result_rows = [(" ".join(parameters["s"].split()),)]

        return R()


TABLES = [
    ("spyre_v2", "test_case_runs"),  # MergeTree, partitioned, index, constraint
    ("spyre_v2", "pipeline_runs"),  # ReplacingMergeTree(version), partitioned
    ("spyre_v2", "run_case_counters"),  # SummingMergeTree, no partition
    ("spyre_v2", "artifact_results"),  # MergeTree
    ("vigil_ops", "metrics_latest"),  # AggregatingMergeTree, TTL
    ("vigil_ops", "freshness_checks"),  # ReplacingMergeTree(version), TTL
    ("vigil_ops", "events"),  # MergeTree
]


def _obj(ddl):
    from spyre_clickhouse_ingest.apply_schema import SchemaObject

    return SchemaObject("table", "t", ddl, "x.sql")


@pytest.mark.parametrize("db,table", TABLES)
def test_a_converted_table_is_not_drift(db, table):
    plain = live_ddl(table)
    converted = replicated(plain, db, table)
    assert SchemaApplier.differs(FormatOnly(), _obj(plain), converted, db) == ""


@pytest.mark.parametrize("db,table", TABLES)
def test_a_table_converted_with_a_generation_suffix_is_not_drift(db, table):
    plain = live_ddl(table)
    converted = replicated(plain, db, table, suffix="/g20261110")
    assert SchemaApplier.differs(FormatOnly(), _obj(plain), converted, db) == ""


def test_the_conversion_marker_comment_is_not_drift():
    plain = live_ddl("events")
    converted = replicated(plain, "vigil_ops", "events") + " COMMENT 'ha-twin'"
    assert (
        SchemaApplier.differs(FormatOnly(), _obj(plain), converted, "vigil_ops") == ""
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("`status` LowCardinality(String)", "`status` String"),
        lambda s: s.replace(
            "ORDER BY (component, run_id, test_case_id)", "ORDER BY (run_id)"
        ),
        lambda s: s.replace("PARTITION BY toYYYYMM(ts)", "PARTITION BY toYear(ts)"),
        lambda s: s.replace("'xpass')", "'xpass', 'new')"),
        lambda s: s.replace("duration_s` Float32", "duration_s` Float64"),
    ],
    ids=["column-type", "order-by", "partition", "constraint", "float-width"],
)
def test_real_drift_is_still_caught_on_a_replicated_table(mutate):
    plain = live_ddl("test_case_runs")
    changed = mutate(replicated(plain, "spyre_v2", "test_case_runs"))
    assert changed != replicated(plain, "spyre_v2", "test_case_runs")
    assert SchemaApplier.differs(FormatOnly(), _obj(plain), changed, "spyre_v2")


def test_a_changed_engine_argument_is_drift():
    plain = live_ddl("pipeline_runs")
    other = replicated(plain, "spyre_v2", "pipeline_runs").replace(
        ", updated_at)", ", started_at)"
    )
    assert "started_at)" in other
    assert SchemaApplier.differs(FormatOnly(), _obj(plain), other, "spyre_v2")


def test_a_different_engine_family_is_drift():
    plain = live_ddl("events")
    other = replicated(plain, "vigil_ops", "events").replace(
        "ReplicatedMergeTree", "ReplicatedReplacingMergeTree"
    )
    assert SchemaApplier.differs(FormatOnly(), _obj(plain), other, "vigil_ops")


def test_replicated_engine_text_inside_a_string_literal_is_left_alone():
    sql = "CREATE TABLE t (`a` String DEFAULT 'ENGINE = ReplicatedMergeTree(x)') ENGINE = MergeTree ORDER BY a"
    assert "ReplicatedMergeTree(x)" in SchemaApplier.plain_engine(sql)


# ---- cluster mode: SQL rewriting -------------------------------------------------


def test_cluster_create_table_gets_on_cluster_and_a_replicated_engine():
    sql = SchemaApplier.clustered(
        live_ddl("test_case_runs"), "table", CLUSTER, "spyre_v2", "test_case_runs"
    )
    assert re.search(
        rf"^CREATE TABLE spyre_v2\.test_case_runs ON CLUSTER {CLUSTER}\b", sql
    )
    assert (
        "ENGINE = ReplicatedMergeTree('/clickhouse/test_cluster/tables/{shard}/spyre_v2/test_case_runs', '{replica}')"
        in sql
    )
    assert sql.count("ENGINE") == 1


def test_cluster_keeps_engine_arguments_in_order():
    sql = SchemaApplier.clustered(
        live_ddl("pipeline_runs"), "table", CLUSTER, "spyre_v2", "pipeline_runs"
    )
    assert (
        "ReplicatedReplacingMergeTree('/clickhouse/test_cluster/tables/{shard}/spyre_v2/pipeline_runs', '{replica}', updated_at)"
        in sql
    )


def test_cluster_summing_merge_tree_with_empty_parens():
    sql = SchemaApplier.clustered(
        "CREATE TABLE IF NOT EXISTS t (a UInt8) ENGINE = SummingMergeTree() ORDER BY a",
        "table",
        CLUSTER,
        "db",
        "t",
    )
    assert (
        "ReplicatedSummingMergeTree('/clickhouse/test_cluster/tables/{shard}/db/t', '{replica}')"
        in sql
    )


def test_cluster_view_and_mv_get_on_cluster_after_the_name():
    v = SchemaApplier.clustered(
        "CREATE VIEW IF NOT EXISTS v AS SELECT 1", "view", CLUSTER, "db", "v"
    )
    assert v.startswith(f"CREATE VIEW IF NOT EXISTS v ON CLUSTER {CLUSTER} AS SELECT 1")
    mv = SchemaApplier.clustered(
        "CREATE MATERIALIZED VIEW m TO t AS SELECT 1", "mv", CLUSTER, "db", "m"
    )
    assert mv.startswith(f"CREATE MATERIALIZED VIEW m ON CLUSTER {CLUSTER} TO t")


def test_a_refreshable_mv_is_never_created_on_cluster():
    sql = "CREATE MATERIALIZED VIEW r REFRESH EVERY 15 MINUTE APPEND TO t AS SELECT 1"
    assert SchemaApplier.clustered(sql, "refresh", CLUSTER, "db", "r") == sql


def test_without_a_cluster_nothing_is_rewritten():
    sql = live_ddl("events")
    assert SchemaApplier.clustered(sql, "table", "", "vigil_ops", "events") == sql


def test_an_unknown_engine_is_refused_in_cluster_mode():
    with pytest.raises(ValueError, match="engine"):
        SchemaApplier.clustered(
            "CREATE TABLE t (a UInt8) ENGINE = Log", "table", CLUSTER, "db", "t"
        )


def test_migration_ddl_gets_on_cluster_but_data_statements_do_not():
    f = lambda s: SchemaApplier.clustered_migration(s, CLUSTER)  # noqa: E731
    assert (
        f("ALTER TABLE t ADD COLUMN IF NOT EXISTS a UInt8")
        == f"ALTER TABLE t ON CLUSTER {CLUSTER} ADD COLUMN IF NOT EXISTS a UInt8"
    )
    assert f("DROP VIEW IF EXISTS v") == f"DROP VIEW IF EXISTS v ON CLUSTER {CLUSTER}"
    assert (
        f("DROP TABLE IF EXISTS t")
        == f"DROP TABLE IF EXISTS t ON CLUSTER {CLUSTER} SYNC"
    )
    assert f("TRUNCATE TABLE t") == f"TRUNCATE TABLE t ON CLUSTER {CLUSTER}"
    assert f("INSERT INTO t SELECT 1") == "INSERT INTO t SELECT 1"
    assert f("DELETE FROM t WHERE 1") == f"DELETE FROM t ON CLUSTER {CLUSTER} WHERE 1"
    assert (
        f("ALTER TABLE db.t ADD INDEX i a TYPE minmax")
        == f"ALTER TABLE db.t ON CLUSTER {CLUSTER} ADD INDEX i a TYPE minmax"
    )


def test_an_on_cluster_already_present_is_not_doubled():
    s = f"ALTER TABLE t ON CLUSTER {CLUSTER} ADD COLUMN a UInt8"
    assert SchemaApplier.clustered_migration(s, CLUSTER) == s


# ---- cluster mode: what apply() sends ---------------------------------------------

from tests.test_apply_schema import DB, FakeServer, _schema  # noqa: E402


def _apply(tmp_path, files, migrations=None, live=None, cluster=CLUSTER):
    server = FakeServer(live)
    d = _schema(tmp_path, files, migrations)
    f = SchemaApplier.selected_files(d)
    m = SchemaApplier.migration_files(d)
    steps = SchemaApplier.apply(server, DB, f, m, cluster)
    return server, steps


T = "CREATE TABLE a (x UInt8) ENGINE = MergeTree ORDER BY x"
V = "CREATE VIEW v AS SELECT x FROM a"
R = "CREATE MATERIALIZED VIEW r REFRESH EVERY 5 MINUTE APPEND TO a AS SELECT 1 AS x"


def test_cluster_apply_sends_on_cluster_ddl_and_a_replicated_ledger(tmp_path):
    server, _ = _apply(tmp_path, {"10-a.sql": T + ";\n" + V + ";"})
    sent = [c for c in server.log if c.startswith("CREATE")]
    assert any(
        "schema_migrations ON CLUSTER" in c and "ReplicatedMergeTree" in c for c in sent
    )
    table = [c for c in sent if " a " in c][0]
    assert f"ON CLUSTER {CLUSTER}" in table and "ReplicatedMergeTree(" in table
    view = [c for c in sent if c.startswith("CREATE VIEW")][0]
    assert f"v ON CLUSTER {CLUSTER}" in view


def test_cluster_apply_never_sends_a_refreshable_mv_on_cluster(tmp_path, capsys):
    server, steps = _apply(tmp_path, {"10-a.sql": T + ";\n" + R + ";"})
    refresh = [c for c in server.log if "REFRESH" in c]
    assert refresh and all("ON CLUSTER" not in c for c in refresh)
    assert "this server only" in capsys.readouterr().out


def test_cluster_migrations_run_ddl_on_every_replica_and_the_ledger_once(tmp_path):
    server, _ = _apply(
        tmp_path,
        {"10-a.sql": T + ";"},
        {
            "001-x.sql": "ALTER TABLE a ADD COLUMN IF NOT EXISTS y UInt8;\nINSERT INTO a SELECT 1;"
        },
    )
    assert (
        f"ALTER TABLE a ON CLUSTER {CLUSTER} ADD COLUMN IF NOT EXISTS y UInt8"
        in server.log
    )
    assert "INSERT INTO a SELECT 1" in server.log
    assert list(server.ledger) == ["001-x.sql"]


def test_cluster_recreated_view_is_dropped_and_created_on_every_replica(tmp_path):
    old = "CREATE VIEW v AS SELECT 0 AS x"
    server, steps = _apply(
        tmp_path, {"10-a.sql": T + ";\n" + V + ";"}, live={"a": T, "v": old}
    )
    assert ("recreate", "v", "10-a.sql") in steps
    assert f"DROP VIEW IF EXISTS v ON CLUSTER {CLUSTER}" in server.log
    assert any(c.startswith(f"CREATE VIEW v ON CLUSTER {CLUSTER}") for c in server.log)


def test_cluster_apply_accepts_a_converted_table_without_drift(tmp_path):
    converted = replicated(T, DB, "a")
    server, steps = _apply(tmp_path, {"10-a.sql": T + ";"}, live={"a": converted})
    assert ("create", "a", "10-a.sql") not in steps


def test_cluster_apply_still_fails_on_real_drift(tmp_path):
    drifted = replicated(
        "CREATE TABLE a (x UInt16) ENGINE = MergeTree ORDER BY x", DB, "a"
    )
    with pytest.raises(SchemaDrift):
        _apply(tmp_path, {"10-a.sql": T + ";"}, live={"a": drifted})


def test_without_a_cluster_the_default_path_is_byte_for_byte_unchanged(tmp_path):
    server, _ = _apply(tmp_path, {"10-a.sql": T + ";\n" + V + ";"}, cluster="")
    assert server.log[0] == SchemaApplier.LEDGER_DDL
    assert all("ON CLUSTER" not in c and "Replicated" not in c for c in server.log)


def test_the_cluster_name_must_be_a_plain_identifier(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_CLUSTER", "x; DROP TABLE t")
    with pytest.raises(ValueError):
        SchemaApplier.cluster()
    monkeypatch.setenv("CLICKHOUSE_CLUSTER", "prod_cluster")
    assert SchemaApplier.cluster() == "prod_cluster"
    monkeypatch.delenv("CLICKHOUSE_CLUSTER")
    assert SchemaApplier.cluster() == ""


def test_every_table_in_the_repo_schema_can_be_made_replicated():
    for path, text in SchemaApplier.selected_files(SchemaApplier.schema_dir()):
        for o in SchemaApplier.objects(path, text):
            if o.kind == "table":
                out = SchemaApplier.clustered(o.sql, o.kind, CLUSTER, "db", o.name)
                assert "ReplicatedMergeTree(" in out or re.search(
                    r"Replicated\w+MergeTree\(", out
                ), o.name


def test_the_keeper_path_can_be_set_by_a_template(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_KEEPER_PATH", "/zk/{shard}/{db}.{table}")
    sql = SchemaApplier.clustered(
        live_ddl("events"), "table", CLUSTER, "vigil_ops", "events"
    )
    assert "ReplicatedMergeTree('/zk/{shard}/vigil_ops.events', '{replica}')" in sql
