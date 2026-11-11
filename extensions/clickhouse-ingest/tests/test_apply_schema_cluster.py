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
    sync = f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} `db`.scratch"
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
    sync = f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} `{DB}`.a"
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
    assert not any(c.startswith("SYSTEM SYNC REPLICA ON CLUSTER") for c in server.log)


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
            return type(
                "R", (), {"result_rows": [("h1", 9000, 1, []), ("h2", 9000, 0, [])]}
            )()
        if "system.one" in sql and self.mode == "blind":
            return type("R", (), {"result_rows": [("h1", 9000), ("h2", 9000)]})()
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


# ---- cluster mode: literals, anchoring, prerequisites --------------------------------


@pytest.mark.parametrize(
    "stmt,want",
    [
        (
            "ALTER TABLE t MODIFY COLUMN x String COMMENT 'rerun ON CLUSTER later'",
            f"ALTER TABLE t {Q} MODIFY COLUMN x String COMMENT 'rerun ON CLUSTER later'",
        ),
        (
            "ALTER TABLE t UPDATE x = 'ON CLUSTER' WHERE 1",
            f"ALTER TABLE t {Q} UPDATE x = 'ON CLUSTER' WHERE 1",
        ),
        (
            "DELETE FROM t WHERE x = 'on cluster'",
            f"DELETE FROM t {Q} WHERE x = 'on cluster'",
        ),
        (
            "RENAME TABLE a TO `ON CLUSTER x`",
            f"RENAME TABLE a TO `ON CLUSTER x` {Q}",
        ),
    ],
)
def test_on_cluster_inside_a_literal_does_not_count_as_present(stmt, want):
    assert SchemaApplier.clustered_migration(stmt, CLUSTER) == want


@pytest.mark.parametrize(
    "stmt",
    [
        f"ALTER TABLE t {Q} ADD COLUMN a UInt8",
        f"DELETE FROM t {Q} WHERE 1",
        f"DROP TABLE t {Q} SYNC",
        f"RENAME TABLE a TO b {Q}",
        f"EXCHANGE TABLES a AND b {Q}",
        f"CREATE VIEW v {Q} AS SELECT 1",
        f"CREATE MATERIALIZED VIEW m {Q} TO t AS SELECT 1 AS a",
    ],
)
def test_a_real_on_cluster_is_left_alone(stmt):
    assert SchemaApplier.clustered_migration(stmt, CLUSTER) == stmt


def test_literals_never_hide_or_fake_an_engine_or_as_select():
    sql = (
        "CREATE TABLE t (a String DEFAULT 'AS SELECT 1' COMMENT 'ENGINE = Log(x)') "
        "ENGINE = MergeTree ORDER BY a"
    )
    out = SchemaApplier.clustered(sql, "table", CLUSTER, "db", "t")
    assert "ReplicatedMergeTree('/clickhouse/test_cluster/tables/{shard}/db/t'" in out
    assert "COMMENT 'ENGINE = Log(x)'" in out and "DEFAULT 'AS SELECT 1'" in out


@pytest.mark.parametrize(
    "engine,args",
    [
        ("SummingMergeTree((a, b))", "(a, b)"),
        ("ReplacingMergeTree(toInt8(v))", "toInt8(v)"),
        ("ReplacingMergeTree(f(g(v), 1))", "f(g(v), 1)"),
    ],
)
def test_nested_parentheses_in_engine_arguments_are_kept(engine, args):
    out = SchemaApplier.clustered(
        f"CREATE TABLE t (a UInt8, b UInt8, v UInt8) ENGINE = {engine} ORDER BY a",
        "table",
        CLUSTER,
        "db",
        "t",
    )
    assert f"'{{replica}}', {args}) ORDER BY a" in out


def test_drop_table_no_delay_is_not_given_sync():
    out = SchemaApplier.clustered_migration("DROP TABLE t NO DELAY", CLUSTER)
    assert out == f"DROP TABLE t {Q} NO DELAY"


def test_an_escaped_backtick_in_a_name_is_handled():
    out = SchemaApplier.clustered_migration("DROP VIEW IF EXISTS `a``b`", CLUSTER)
    assert out == f"DROP VIEW IF EXISTS `a``b` {Q}"
    assert SchemaApplier.bare("`a``b`") == "a`b"


@pytest.mark.parametrize(
    "stmt,token",
    [
        ("INSERT INTO t SELECT 1", "t"),
        ("INSERT INTO TABLE t SELECT 1", "t"),
        ("INSERT INTO other.`t x` SELECT 1", "other.`t x`"),
        ("WITH 1 AS one INSERT INTO t SELECT one", "t"),
        ("INSERT INTO table (a) VALUES (1)", "table"),
        ("SELECT 'INSERT INTO t'", ""),
    ],
)
def test_insert_targets_are_tokenised_like_names(stmt, token):
    assert SchemaApplier.insert_target(stmt) == token


def test_the_sync_target_is_quoted_qualified_and_has_a_long_timeout(tmp_path):
    server = FakeServer({"a": replicated(T, DB, "a"), "t x": replicated(T, DB, "t x")})
    sent = []
    orig = server.command
    server.command = lambda sql, settings=None: (
        sent.append((sql, settings)),
        orig(sql),
    )[1]
    d = _schema(
        tmp_path,
        {"10-a.sql": T + ";"},
        {
            "001-x.sql": "INSERT INTO `t x` SELECT 1;\nINSERT INTO db.a SELECT 1;\nDELETE FROM a WHERE 1 IN (SELECT x FROM `t x`);\nSELECT 1 FROM a;"
        },
    )
    SchemaApplier.apply(
        server,
        DB,
        SchemaApplier.selected_files(d),
        SchemaApplier.migration_files(d),
        CLUSTER,
    )
    syncs = {c: st for c, st in sent if c.startswith("SYSTEM SYNC REPLICA ON CLUSTER")}
    assert f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} `{DB}`.`t x`" in syncs
    assert f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} db.a" in syncs
    assert all(st == SchemaApplier.SYNC_SETTINGS for st in syncs.values())


def test_two_replicas_on_one_host_are_told_apart(tmp_path):
    class TwoPorts(FakeServer):
        def query(self, sql, parameters=None):
            if "clusterAllReplicas" in sql and "system.one" in sql:
                return type("R", (), {"result_rows": [("h", 9000), ("h", 9001)]})()
            if "clusterAllReplicas" in sql:
                return type("R", (), {"result_rows": [("h", 9000, 1, [])]})()
            return super().query(sql, parameters)

    d = _schema(tmp_path, {"10-a.sql": T + ";"})
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    with pytest.raises(RuntimeError, match="9001"):
        SchemaApplier.apply(TwoPorts(), DB, f, m, CLUSTER)


def test_a_replica_that_cannot_be_queried_reports_the_real_error(tmp_path):
    server = Blind("denied")
    d = _schema(tmp_path, {"10-a.sql": T + ";"})
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    with pytest.raises(RuntimeError, match="ACCESS_DENIED.*down or unreachable"):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)


PLAIN_LEDGER = {"schema_migrations": SchemaApplier.LEDGER_DDL}


def test_an_unconverted_ledger_stops_apply_and_plan_before_any_write(tmp_path):
    server = FakeServer(dict(PLAIN_LEDGER))
    d = _schema(tmp_path, {"10-a.sql": T + ";"})
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    for call in (SchemaApplier.apply, SchemaApplier.plan):
        with pytest.raises(ValueError, match="schema_migrations"):
            call(server, DB, f, m, CLUSTER)
    assert not any(c.startswith(("CREATE", "SYSTEM", "INSERT")) for c in server.log)


def test_a_migration_inserting_into_an_unconverted_table_stops_before_any_write(
    tmp_path,
):
    server = FakeServer({"a": T})
    d = _schema(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "INSERT INTO a SELECT 1;\nDELETE FROM a WHERE 1;"},
    )
    f, m = SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)
    with pytest.raises(ValueError, match="not Replicated.*: a "):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)
    with pytest.raises(ValueError, match="not Replicated"):
        SchemaApplier.plan(server, DB, f, m, CLUSTER)
    assert not any(
        c.startswith(("CREATE", "INSERT", "DELETE", "SYSTEM")) for c in server.log
    )
    assert not server.ledger


def test_an_unconverted_table_no_migration_inserts_into_is_not_a_prerequisite(tmp_path):
    server, steps = _apply(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "ALTER TABLE a ADD COLUMN y UInt8;"},
        live={"a": replicated(T, DB, "a")},
    )
    assert ("migrate", "001-x.sql", "") in steps


def test_no_preflight_query_is_made_without_a_cluster(tmp_path):
    server, _ = _apply(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "INSERT INTO a SELECT 1;"},
        cluster="",
    )
    assert not any("SYSTEM" in c for c in server.log)
    assert not any("has(" in q or "clusterAllReplicas" in q for q in server.queries)


# ---- cluster mode: reruns, carry-over, visibility; non-cluster drift ------------------


def _mig(tmp_path, files, migrations, live):
    server = FakeServer(live)
    d = _schema(tmp_path, files, migrations)
    return server, SchemaApplier.selected_files(d), SchemaApplier.migration_files(d)


def test_rerun_checks_tables_are_replicated_before_running_anything(tmp_path):
    server = FakeServer({"a": T, "schema_migrations": SchemaApplier.LEDGER_DDL})
    p = tmp_path / "001-x.sql"
    p.write_text("-- RERUNNABLE\nINSERT INTO a SELECT 1;\nDELETE FROM a WHERE 1;")
    with pytest.raises(ValueError, match="not Replicated.*: a "):
        SchemaApplier.rerun(server, p, CLUSTER, DB)
    assert server.log == []
    server = FakeServer({"a": replicated(T, DB, "a")})
    SchemaApplier.rerun(server, p, CLUSTER, DB)
    assert f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} `{DB}`.a" in server.log


def test_rows_left_unsynced_by_one_migration_are_synced_before_the_next_reads_them(
    tmp_path,
):
    server, f, m = _mig(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "INSERT INTO a SELECT 1;", "002-y.sql": "DELETE FROM a WHERE 1;"},
        {},
    )
    SchemaApplier.apply(server, DB, f, m, CLUSTER)
    log = server.log
    sync = f"SYSTEM SYNC REPLICA ON CLUSTER {CLUSTER} `{DB}`.a"
    assert (
        log.index("INSERT INTO a SELECT 1")
        < log.index(sync)
        < next(i for i, c in enumerate(log) if c.startswith("DELETE FROM a"))
    )


def test_insert_into_function_is_not_a_sync_target():
    assert (
        SchemaApplier.insert_target("INSERT INTO FUNCTION remote('h', db, t) SELECT 1")
        == ""
    )


def test_an_insert_into_another_database_is_refused(tmp_path):
    server, f, m = _mig(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "INSERT INTO other.a SELECT 1;"},
        {},
    )
    with pytest.raises(ValueError, match="outside"):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)
    assert server.log == [] or not any(c.startswith("CREATE") for c in server.log)


def test_an_insert_into_a_table_the_login_cannot_see_is_refused(tmp_path):
    server, f, m = _mig(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "INSERT INTO ghost SELECT 1;"},
        {},
    )
    with pytest.raises(ValueError, match="ghost.*not visible"):
        SchemaApplier.apply(server, DB, f, m, CLUSTER)
    assert not any(c.startswith("CREATE") for c in server.log)


def test_an_if_table_exists_guarded_insert_is_not_refused(tmp_path):
    server, f, m = _mig(
        tmp_path,
        {"10-a.sql": T + ";"},
        {"001-x.sql": "-- IF TABLE EXISTS: ghost\nINSERT INTO ghost SELECT 1;"},
        {},
    )
    SchemaApplier.apply(server, DB, f, m, CLUSTER)


def test_a_comment_only_change_is_still_drift_without_a_cluster():
    plain = "CREATE TABLE t (a UInt8) ENGINE = MergeTree ORDER BY a"
    obj = _obj(plain)
    for stored in (plain + " COMMENT 'x'", plain + " COMMENT 'y'"):
        assert SchemaApplier.differs(FormatOnly(), obj, stored, "db") != ""
    commented = _obj(plain + " COMMENT 'x'")
    assert SchemaApplier.differs(FormatOnly(), commented, plain, "db") != ""
    assert (
        SchemaApplier.differs(FormatOnly(), commented, plain + " COMMENT 'x'", "db")
        == ""
    )


def test_a_comment_on_a_converted_table_is_still_ignored():
    plain = "CREATE TABLE t (a UInt8) ENGINE = MergeTree ORDER BY a"
    conv = replicated(plain, "db", "t") + " COMMENT 'converted'"
    assert SchemaApplier.differs(FormatOnly(), _obj(plain), conv, "db") == ""


# ---- cluster mode: the command sequence ------------------------------------------------


class Rec:
    def __init__(self):
        self.sent = []

    def command(self, sql, settings=None):
        self.sent.append(sql)
        return 1


def _run(migs, db="appdb"):
    rec, un = Rec(), {}
    for m in migs:
        SchemaApplier.run_migration(rec, m, CLUSTER, db, un)
    return rec.sent, un


S_CREATE = "CREATE TABLE s (a UInt8) ENGINE = MergeTree ORDER BY a"


def test_a_dropped_table_leaves_no_pending_sync_for_a_later_table_of_that_name():
    sent, un = _run(
        [f"{S_CREATE};\nINSERT INTO s VALUES (1);\nDROP TABLE s;", f"{S_CREATE};"]
    )
    assert not any(c.startswith("SYSTEM SYNC") for c in sent)
    assert un == {}


def test_a_dropped_view_or_if_exists_table_is_forgotten_too():
    sent, un = _run(
        [
            f"{S_CREATE};\nINSERT INTO s VALUES (1);\nDROP TABLE IF EXISTS s;",
            f"{S_CREATE};",
        ]
    )
    assert un == {} and not any(c.startswith("SYSTEM SYNC") for c in sent)


def test_a_truncated_table_keeps_its_pending_sync_until_read():
    sent, un = _run([f"{S_CREATE};\nINSERT INTO s VALUES (1);\nTRUNCATE TABLE s;"])
    assert "s" in un and not any(c.startswith("SYSTEM SYNC") for c in sent)


def test_rename_and_exchange_sync_the_tables_they_read_first():
    sent, un = _run([f"{S_CREATE};\nINSERT INTO s VALUES (1);\nRENAME TABLE s TO s2;"])
    assert sent.index(
        "SYSTEM SYNC REPLICA ON CLUSTER test_cluster appdb.s".replace(
            "appdb.s", "`appdb`.s"
        )
    ) < next(i for i, c in enumerate(sent) if c.startswith("RENAME"))
    assert "s" not in un
    sent, un = _run(
        [f"{S_CREATE};\nINSERT INTO s VALUES (1);\nEXCHANGE TABLES s AND t;"]
    )
    assert any(c.startswith("SYSTEM SYNC") for c in sent[:-1]) and un.get("s") is None


def test_migration_commands_include_syncs_and_guards():
    text = (
        f"{S_CREATE};\nINSERT INTO s VALUES (1);\n"
        "-- IF TABLE EXISTS: ghost\nDELETE FROM s WHERE a IN (SELECT a FROM ghost);\n"
        "DELETE FROM s WHERE 1;"
    )
    seen = []
    cmds = [
        c
        for c, _ in SchemaApplier.migration_commands(
            text, CLUSTER, "appdb", exists=lambda t: seen.append(t) or False
        )
    ]
    assert seen == ["ghost"]
    assert not any("ghost" in c for c in cmds)
    assert sum(c.startswith("SYSTEM SYNC") for c in cmds) == 1
