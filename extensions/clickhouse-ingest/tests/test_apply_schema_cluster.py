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


# ---- cluster mode: migrations, refreshable MVs, validation ------------------------

SCRATCH = "CREATE TABLE scratch (a UInt8) ENGINE = MergeTree ORDER BY a"


def test_a_migration_scratch_table_is_replicated_and_synced_before_it_is_read():
    mig = (
        f"{SCRATCH};\nINSERT INTO scratch SELECT 1;\n"
        "DELETE FROM t WHERE a IN (SELECT a FROM scratch);\nDROP TABLE scratch;"
    )
    sent = [s for _, s in SchemaApplier.migration_statements(mig, CLUSTER, "db")]
    assert f"CREATE TABLE scratch ON CLUSTER {CLUSTER}" in sent[0]
    assert (
        "ReplicatedMergeTree('/clickhouse/test_cluster/tables/{shard}/db/scratch'"
        in sent[0]
    )


def test_a_migration_syncs_replicas_after_filling_a_scratch_table(tmp_path):
    server, _ = _apply(
        tmp_path,
        {"10-a.sql": T + ";"},
        {
            "001-x.sql": f"{SCRATCH};\nINSERT INTO scratch SELECT 1;\nDELETE FROM a WHERE x IN (SELECT a FROM scratch);"
        },
    )
    log = server.log
    sync = f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} scratch"
    assert sync in log
    assert (
        log.index("INSERT INTO scratch SELECT 1")
        < log.index(sync)
        < next(i for i, c in enumerate(log) if c.startswith("DELETE FROM a"))
    )


def test_a_migration_syncs_an_existing_table_it_inserted_into_before_reading_it(
    tmp_path,
):
    mig = (
        "INSERT INTO a SELECT 1 WHERE 1 NOT IN (SELECT x FROM a);\n"
        "DELETE FROM a WHERE x IN (SELECT x FROM a WHERE x > 5);"
    )
    server, _ = _apply(tmp_path, {"10-a.sql": T + ";"}, {"001-x.sql": mig})
    log = server.log
    sync = f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} a"
    assert log.count(sync) == 1
    assert log.index(sync) < next(
        i for i, c in enumerate(log) if c.startswith("DELETE FROM a")
    )
    assert log.index(sync) > next(
        i for i, c in enumerate(log) if c.startswith("INSERT")
    )


def test_a_migration_does_not_sync_a_table_nothing_reads_afterwards(tmp_path):
    server, _ = _apply(
        tmp_path, {"10-a.sql": T + ";"}, {"001-x.sql": "INSERT INTO a SELECT 1;"}
    )
    assert f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} a" not in server.log


def test_a_migration_does_not_sync_a_table_it_is_about_to_drop(tmp_path):
    server, _ = _apply(
        tmp_path,
        {"10-a.sql": T + ";"},
        {
            "001-x.sql": f"{SCRATCH};\nINSERT INTO scratch SELECT 1;\nDROP TABLE scratch;"
        },
    )
    assert not any(c.startswith("SYSTEM SYNC REPLICA ON CLUSTER") for c in server.log)


def test_the_ledger_is_synced_before_it_is_read_in_cluster_mode(tmp_path):
    server, _ = _apply(tmp_path, {"10-a.sql": T + ";"}, {"001-x.sql": "SELECT 1;"})
    sync = f"SYSTEM SYNC REPLICA {DB}.schema_migrations"
    assert sync in server.log


def test_a_plain_apply_sends_no_sync(tmp_path):
    server, _ = _apply(
        tmp_path, {"10-a.sql": T + ";"}, {"001-x.sql": "SELECT 1;"}, cluster=""
    )
    assert not any("SYSTEM SYNC" in c for c in server.log)


def test_a_plain_scratch_table_with_on_cluster_is_refused():
    with pytest.raises(ValueError, match="Replicated"):
        SchemaApplier.clustered_migration(
            f"CREATE TABLE s ON CLUSTER {CLUSTER} (a UInt8) ENGINE = MergeTree ORDER BY a",
            CLUSTER,
        )


def test_a_scratch_table_on_an_unsupported_engine_is_refused():
    with pytest.raises(ValueError, match="engine"):
        SchemaApplier.clustered_migration(
            "CREATE TABLE s (a UInt8) ENGINE = Memory", CLUSTER
        )


def test_rename_and_exchange_get_on_cluster_at_the_end():
    f = lambda s: SchemaApplier.clustered_migration(s, CLUSTER)  # noqa: E731
    assert f("RENAME TABLE a TO b") == f"RENAME TABLE a TO b ON CLUSTER {CLUSTER}"
    assert (
        f("RENAME TABLE a TO b, c TO d")
        == f"RENAME TABLE a TO b, c TO d ON CLUSTER {CLUSTER}"
    )
    assert (
        f("EXCHANGE TABLES a AND b") == f"EXCHANGE TABLES a AND b ON CLUSTER {CLUSTER}"
    )


@pytest.mark.parametrize(
    "stmt",
    [
        "CREATE OR REPLACE TABLE t (a UInt8) ENGINE = MergeTree ORDER BY a",
        "CREATE DICTIONARY d (a UInt8) PRIMARY KEY a SOURCE(NULL()) LAYOUT(FLAT()) LIFETIME(0)",
        "OPTIMIZE TABLE t FINAL",
        "ATTACH TABLE t",
        "DETACH TABLE t",
        "SYSTEM STOP MERGES t",
    ],
)
def test_ddl_with_no_cluster_rewrite_is_refused_not_run_on_one_server(stmt):
    with pytest.raises(ValueError, match="cannot run on a cluster"):
        SchemaApplier.clustered_migration(stmt, CLUSTER)
    assert SchemaApplier.clustered_migration(stmt, "") == stmt


def test_a_refreshable_mv_in_a_migration_stays_on_the_connected_server():
    sql = "CREATE MATERIALIZED VIEW r REFRESH EVERY 15 MINUTE APPEND TO t AS SELECT 1"
    assert SchemaApplier.clustered_migration(sql, CLUSTER) == sql


def test_a_refreshable_mv_another_replica_owns_is_not_created_again(tmp_path):
    files = {"10-a.sql": T + ";\n" + R + ";"}
    server = FakeServer({"a": T})
    server.elsewhere = {"r"}
    d = _schema(tmp_path, files)
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    steps = SchemaApplier.apply(server, DB, f, m, CLUSTER)
    assert not any("REFRESH" in c for c in server.log)
    assert ("create", "r", "10-a.sql") not in steps
    assert SchemaApplier.plan(server, DB, f, m, CLUSTER) == []


def test_a_refreshable_mv_nobody_owns_is_created_once(tmp_path):
    server, steps = _apply(tmp_path, {"10-a.sql": T + ";\n" + R + ";"})
    assert ("create", "r", "10-a.sql") in steps


def test_without_a_cluster_no_replica_query_is_made(tmp_path):
    server = FakeServer()
    d = _schema(tmp_path, {"10-a.sql": T + ";\n" + R + ";"})
    SchemaApplier.apply(
        server, DB, SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    )
    assert SchemaApplier.refresh_mvs_on_cluster(server, DB, "") == set()


def test_plan_refuses_an_unsupported_engine_before_anything_is_created(tmp_path):
    bad = "CREATE TABLE b (a UInt8) ENGINE = Log"
    server = FakeServer()
    d = _schema(tmp_path, {"10-a.sql": T + ";\n" + bad + ";"})
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    with pytest.raises(ValueError, match="engine"):
        SchemaApplier.plan(server, DB, f, m, CLUSTER)
    with pytest.raises(ValueError, match="engine"):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)
    assert not [c for c in server.log if c.startswith("CREATE TABLE a")]


def test_apply_refuses_an_unclusterable_migration_before_creating_tables(tmp_path):
    server = FakeServer()
    d = _schema(
        tmp_path, {"10-a.sql": T + ";"}, {"001-x.sql": "OPTIMIZE TABLE a FINAL;"}
    )
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    with pytest.raises(ValueError, match="001-x.sql"):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)
    assert not [c for c in server.log if c.startswith("CREATE TABLE a")]
    assert not server.ledger


def test_every_repo_migration_can_run_on_a_cluster():
    for path in SchemaApplier.migration_files(SchemaApplier.schema_dir()):
        SchemaApplier.migration_statements(path.read_text(), CLUSTER, "db")


def test_the_repo_migrations_are_unchanged_without_a_cluster():
    for path in SchemaApplier.migration_files(SchemaApplier.schema_dir()):
        got = [s for _, s in SchemaApplier.migration_statements(path.read_text())]
        want = [
            s.rpartition("\0")[2].strip()
            for s in SchemaApplier.statements(
                SchemaApplier.IF_TABLE.sub(
                    lambda m: f"\0{m.group(1)}\0", path.read_text()
                )
            )
        ]
        assert got == want, path.name


# ---- cluster mode: names, rejected forms ---------------------------------------------

Q = f"ON CLUSTER {CLUSTER}"


@pytest.mark.parametrize(
    "stmt,want",
    [
        ("DROP VIEW IF EXISTS `v`", f"DROP VIEW IF EXISTS `v` {Q}"),
        ("DROP VIEW IF EXISTS db.v", f"DROP VIEW IF EXISTS db.v {Q}"),
        (
            "ALTER TABLE db.`t` ADD COLUMN a UInt8",
            f"ALTER TABLE db.`t` {Q} ADD COLUMN a UInt8",
        ),
        (
            'ALTER TABLE "d"."t" ADD COLUMN a UInt8',
            f'ALTER TABLE "d"."t" {Q} ADD COLUMN a UInt8',
        ),
        ("TRUNCATE TABLE IF EXISTS `t`", f"TRUNCATE TABLE IF EXISTS `t` {Q}"),
        ("DROP TABLE `t`", f"DROP TABLE `t` {Q} SYNC"),
        ("DELETE FROM `db`.`t` WHERE 1", f"DELETE FROM `db`.`t` {Q} WHERE 1"),
    ],
)
def test_quoted_and_qualified_names_get_on_cluster_in_the_right_place(stmt, want):
    assert SchemaApplier.clustered_migration(stmt, CLUSTER) == want


def test_a_quoted_create_table_gets_on_cluster_and_a_replicated_engine():
    for name, bare in [("`db`.t", "t"), ("`my-t`", "my-t"), ('"my t"', "my t")]:
        sql = (
            f"CREATE TABLE IF NOT EXISTS {name} (a UInt8) ENGINE = MergeTree ORDER BY a"
        )
        out = SchemaApplier.clustered(sql, "table", CLUSTER, "db", bare)
        assert out.startswith(f"CREATE TABLE IF NOT EXISTS {name} {Q} (a UInt8)"), out
        assert f"db/{bare}'" in out and "ReplicatedMergeTree(" in out
    out = SchemaApplier.clustered_migration(
        "CREATE TABLE `my-t` (a UInt8) ENGINE = MergeTree ORDER BY a", CLUSTER, "db"
    )
    assert f"`my-t` {Q}" in out and "ReplicatedMergeTree(" in out


def test_a_create_view_with_a_quoted_name_gets_on_cluster():
    assert (
        SchemaApplier.clustered(
            "CREATE VIEW IF NOT EXISTS `db`.`v` AS SELECT 1", "view", CLUSTER, "db", "v"
        )
        == f"CREATE VIEW IF NOT EXISTS `db`.`v` {Q} AS SELECT 1"
    )


def test_a_head_that_cannot_take_on_cluster_raises():
    with pytest.raises(ValueError, match="ON CLUSTER"):
        SchemaApplier.clustered(
            "CREATE VIEW (x) AS SELECT 1", "view", CLUSTER, "db", "v"
        )
    with pytest.raises(ValueError, match="ON CLUSTER"):
        SchemaApplier.clustered_migration("ALTER TABLE (x) ADD COLUMN a UInt8", CLUSTER)


@pytest.mark.parametrize(
    "stmt",
    [
        "DROP TABLE a, b",
        "RENAME TABLE a TO b SETTINGS x = 1",
        "EXCHANGE TABLES a AND b SETTINGS x = 1",
        "GRANT SELECT ON db.t TO u",
        "REVOKE SELECT ON db.t FROM u",
        "CREATE TABLE t ENGINE = MergeTree ORDER BY a AS SELECT 1 AS a",
        "CREATE TABLE t AS other",
        "CREATE TABLE t AS other ENGINE = MergeTree ORDER BY a",
        "CREATE MATERIALIZED VIEW m ENGINE = MergeTree ORDER BY a AS SELECT 1 AS a",
    ],
)
def test_forms_the_cluster_rewrite_cannot_do_correctly_raise(stmt):
    with pytest.raises(ValueError):
        SchemaApplier.clustered_migration(stmt, CLUSTER, "db")
    assert SchemaApplier.clustered_migration(stmt, "", "db") == stmt


def test_create_table_as_other_has_a_proper_message():
    with pytest.raises(ValueError, match="AS <table or SELECT>"):
        SchemaApplier.clustered("CREATE TABLE t AS other", "table", CLUSTER, "db", "t")


def test_a_mv_with_to_is_accepted_and_one_with_its_own_engine_is_not():
    ok = "CREATE MATERIALIZED VIEW m TO t AS SELECT 1 AS a"
    assert f"m {Q} TO t" in SchemaApplier.clustered(ok, "mv", CLUSTER, "db", "m")
    with pytest.raises(ValueError, match="TO <table>"):
        SchemaApplier.clustered(
            "CREATE MATERIALIZED VIEW m ENGINE = MergeTree ORDER BY a AS SELECT 1 AS a",
            "mv",
            CLUSTER,
            "db",
            "m",
        )


def test_a_keeper_path_without_the_table_placeholder_is_refused(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_KEEPER_PATH", "/zk/{shard}/{db}")
    with pytest.raises(ValueError, match="KEEPER_PATH"):
        SchemaApplier.clustered(T, "table", CLUSTER, "db", "a")
    monkeypatch.setenv("CLICKHOUSE_KEEPER_PATH", "/zk/{shard}/{uuid}")
    assert "/zk/{shard}/{uuid}" in SchemaApplier.clustered(
        T, "table", CLUSTER, "db", "a"
    )


# ---- cluster mode: reading the other replicas ---------------------------------------


class Blind(FakeServer):
    """Remote replicas that cannot see system.tables rows, or cannot be queried at all."""

    def __init__(self, mode, *a):
        super().__init__(*a)
        self.mode = mode

    def query(self, sql, parameters=None):
        if "clusterAllReplicas" in sql and self.mode == "denied":
            raise RuntimeError("ACCESS_DENIED")
        if (
            "clusterAllReplicas" in sql
            and "system.one" not in sql
            and self.mode == "blind"
        ):
            return type("R", (), {"result_rows": [("h1", 1, []), ("h2", 0, [])]})()
        if "system.one" in sql and self.mode == "blind":
            return type("R", (), {"result_rows": [("h1",), ("h2",)]})()
        return super().query(sql, parameters)


@pytest.mark.parametrize("mode", ["denied", "blind"])
def test_an_unreadable_replica_stops_apply_before_anything_is_created(tmp_path, mode):
    server = Blind(mode)
    d = _schema(tmp_path, {"10-a.sql": T + ";\n" + R + ";"})
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    with pytest.raises(RuntimeError, match="rerun"):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)
    assert not any(c.startswith("CREATE TABLE a") for c in server.log)
    assert not any("REFRESH" in c for c in server.log)
