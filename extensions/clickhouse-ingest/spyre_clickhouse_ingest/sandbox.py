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

"""Per-person v2 sandbox databases on a dev server: built by the applier from a schema/ dir,
seeded with a connected sample of prod, owned by a login scoped to that one database."""

import argparse
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

import regex as re

from .apply_schema import SCHEMA_DIR, SchemaApplier, SchemaDrift


@dataclass(frozen=True)
class SeedFilter:
    """Which prod runs a sandbox gets; every table is then cut to those runs."""

    days: int = 14
    runs_per_component: int = 200
    components: tuple = ()
    arches: tuple = ()
    tags: tuple = ()
    run_ids: tuple = ()


class Sandbox:
    """Creates, seeds, diffs and drops `sandbox_<name>` and its `sandbox_<name>_admin` login."""

    PREFIX = "sandbox_"
    NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
    # Created once per dev server by `bootstrap`; holds prod's read-only login so no query
    # text (and so no query_log row) carries the password.
    SOURCE = "prod_v2"
    RUNS = "_seed_runs"
    # (table, timestamp column) whose run_ids are candidates for the sample.
    RUN_SOURCES = (
        ("test_case_runs", "ts"),
        ("benchmark_runs", "ts"),
        ("capability_runs", "ts"),
        ("artifact_results", "ts"),
        ("hw_failure_diagnostics", "ingested_at"),
    )
    # (table, predicate, key). Insert order matters: oss_ci_benchmark_v3_mv joins benchmarks at
    # insert time, so each dimension precedes its fact. MV targets and the ledger are filled by
    # the schema itself. A key the sandbox already holds is skipped, so a re-seed adds no
    # duplicate rows and the MVs count each run once.
    SEED = (
        (
            "test_cases",
            "test_case_id GLOBAL IN (SELECT test_case_id FROM {src:test_case_runs} WHERE {runs})",
            "test_case_id",
        ),
        ("test_case_runs", "{runs}", "run_id"),
        (
            "benchmarks",
            "benchmark_id GLOBAL IN (SELECT benchmark_id FROM {src:benchmark_runs} WHERE {runs})",
            "benchmark_id",
        ),
        ("benchmark_runs", "{runs}", "run_id"),
        (
            "capabilities",
            "capability_id GLOBAL IN (SELECT capability_id FROM {src:capability_runs} WHERE {runs})",
            "capability_id",
        ),
        ("capability_runs", "{runs}", "run_id"),
        (
            "artifacts",
            (
                "artifact_id GLOBAL IN (SELECT artifact_id FROM {src:artifact_results} WHERE {runs} "
                "UNION ALL SELECT artifact_id FROM {src:hw_failure_diagnostics} WHERE {runs})"
            ),
            "artifact_id",
        ),
        (
            "artifact_refs",
            "artifact_id GLOBAL IN (SELECT artifact_id FROM {db}.artifacts)",
            "artifact_id",
        ),
        (
            "artifact_tags",
            "artifact_id GLOBAL IN (SELECT artifact_id FROM {db}.artifacts)",
            "artifact_id",
        ),
        ("artifact_results", "{runs}", "run_id"),
        ("hw_failure_diagnostics", "{runs}", "run_id"),
        ("jenkins_agents", "ts >= now() - INTERVAL {days} DAY", "(node, ts)"),
        # By time, not {runs}: a run that failed before publishing has no run_id to be picked by.
        ("pipeline_runs", "started_at >= now() - INTERVAL {days} DAY", "run_key"),
        (
            "pipeline_run_legs",
            "run_key GLOBAL IN (SELECT run_key FROM {db}.pipeline_runs)",
            "(run_key, arch, component, image, kind)",
        ),
    )

    @classmethod
    def database(cls, name: str) -> str:
        if not cls.NAME.match(name):
            raise SystemExit(
                f"[error] sandbox name {name!r} must match {cls.NAME.pattern}"
            )
        return cls.PREFIX + name

    @classmethod
    def user(cls, name: str) -> str:
        return cls.database(name) + "_admin"

    @classmethod
    def src(cls, table: str) -> str:
        return f"remote({cls.SOURCE}, table='{table}')"

    @staticmethod
    def quoted(values) -> str:
        return ", ".join(
            "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'" for v in values
        )

    @classmethod
    def render(cls, table: str, template: str, key: str, db: str, f: SeedFilter) -> str:
        """A SEED predicate with its placeholders filled, minus keys the sandbox already holds."""
        sql = re.sub(r"\{src:(\w+)\}", lambda m: cls.src(m.group(1)), template)
        runs = f"run_id GLOBAL IN (SELECT run_id FROM {db}.{cls.RUNS})"
        sql = (
            sql.replace("{runs}", runs)
            .replace("{db}", db)
            .replace("{days}", str(f.days))
        )
        return f"({sql}) AND {key} GLOBAL NOT IN (SELECT {key.strip('()')} FROM {db}.{table})"

    @classmethod
    def run_selects(cls, f: SeedFilter, columns: dict) -> list:
        """One SELECT run_id per source that carries every filtered column; others sit out."""
        tagged = (
            f" AND run_id GLOBAL IN (SELECT run_id FROM {cls.src('artifact_results')} WHERE "
            f"artifact_id GLOBAL IN (SELECT artifact_id FROM {cls.src('artifact_tags')} "
            f"WHERE tag IN ({cls.quoted(f.tags)})))"
            if f.tags
            else ""
        )
        out = []
        for table, ts in cls.RUN_SOURCES:
            have = columns.get(table, set())
            if (
                not have
                or (f.components and "component" not in have)
                or (f.arches and "arch" not in have)
            ):
                continue
            where = f"{ts} >= now() - INTERVAL {f.days} DAY"
            if f.components:
                where += f" AND component IN ({cls.quoted(f.components)})"
            if f.arches:
                where += f" AND arch IN ({cls.quoted(f.arches)})"
            by = ", component" if "component" in have else ""
            out.append(
                f"SELECT run_id FROM (SELECT run_id{by} FROM {cls.src(table)} WHERE {where}{tagged} "
                f"GROUP BY run_id{by} ORDER BY max({ts}) DESC LIMIT {f.runs_per_component}"
                f"{' BY component' if by else ''})"
            )
        return out

    @staticmethod
    def columns(client, table_expr: str) -> list:
        return [r[0] for r in client.query(f"DESCRIBE TABLE {table_expr}").result_rows]

    @classmethod
    def seed(cls, client, db: str, f: SeedFilter) -> list:
        """(table, rows) copied from prod; only columns both sides have, so an edited table seeds."""
        local = {
            t
            for (t,) in client.query(
                "SELECT name FROM system.tables WHERE database = %(db)s AND engine LIKE '%%MergeTree'",
                parameters={"db": db},
            ).result_rows
        }
        source = {t: set(cls.columns(client, cls.src(t))) for t, _ in cls.RUN_SOURCES}
        client.command(f"DROP TABLE IF EXISTS {db}.{cls.RUNS}")
        client.command(f"CREATE TABLE {db}.{cls.RUNS} (run_id UUID) ENGINE = Memory")
        selects = cls.run_selects(f, source)
        if selects:
            client.command(
                f"INSERT INTO {db}.{cls.RUNS} " + " UNION DISTINCT ".join(selects)
            )
        if f.run_ids:
            client.command(
                f"INSERT INTO {db}.{cls.RUNS} SELECT toUUID(arrayJoin([{cls.quoted(f.run_ids)}]))"
            )
        out = []
        for table, template, key in cls.SEED:
            if table not in local:
                continue
            mine = cls.columns(client, f"{db}.{table}")
            theirs = set(cls.columns(client, cls.src(table)))
            cols = ", ".join(f"`{c}`" for c in mine if c in theirs)
            client.command(
                f"INSERT INTO {db}.{table} ({cols}) SELECT {cols} FROM {cls.src(table)} "
                f"WHERE {cls.render(table, template, key, db, f)}"
            )
            out.append((table, client.command(f"SELECT count() FROM {db}.{table}")))
        client.command(f"DROP TABLE {db}.{cls.RUNS}")
        return out

    @classmethod
    def grant(cls, client, name: str, password: str) -> None:
        """Full rights on its own database, read-only on dev v2, and read of prod via SOURCE."""
        db, user = cls.database(name), cls.user(name)
        client.command(
            f"CREATE USER OR REPLACE {user} IDENTIFIED WITH sha256_password BY %(pw)s "
            f"DEFAULT DATABASE {db}",
            parameters={"pw": password},
        )
        client.command(f"GRANT ALL ON {db}.* TO {user}")
        client.command(f"GRANT SELECT ON spyre_v2.* TO {user}")
        client.command(f"GRANT CREATE TEMPORARY TABLE, REMOTE ON *.* TO {user}")
        client.command(f"GRANT NAMED COLLECTION ON {cls.SOURCE} TO {user}")

    @classmethod
    def build(
        cls, admin, connect, name: str, schema_dir: Path, comment: str = ""
    ) -> list:
        """Create sandbox_<name> and converge it on schema_dir; a drifted apply leaves nothing."""
        db = cls.database(name)
        admin.command(f"DROP DATABASE IF EXISTS {db} SYNC")
        admin.command(f"CREATE DATABASE {db} COMMENT %(c)s", parameters={"c": comment})
        target = connect(db)
        server = SchemaApplier.version_tuple(target.command("SELECT version()"))
        files = SchemaApplier.selected_files(schema_dir, server=server)
        try:
            return SchemaApplier.apply(
                target, db, files, SchemaApplier.migration_files(schema_dir)
            )
        except SchemaDrift:
            admin.command(f"DROP DATABASE IF EXISTS {db} SYNC")
            raise

    @classmethod
    def drop(cls, admin, name: str) -> None:
        admin.command(f"DROP DATABASE IF EXISTS {cls.database(name)} SYNC")
        admin.command(f"DROP USER IF EXISTS {cls.user(name)}")

    @classmethod
    def diff(cls, target, db: str, schema_dir: Path) -> tuple:
        """(applier plan steps, objects only the sandbox has) between db and schema_dir."""
        server = SchemaApplier.version_tuple(target.command("SELECT version()"))
        files = SchemaApplier.selected_files(schema_dir, server=server)
        steps = SchemaApplier.plan(
            target, db, files, SchemaApplier.migration_files(schema_dir)
        )
        return steps, cls.extra_objects(target, db, files)

    @classmethod
    def extra_objects(cls, client, db: str, files: list) -> list:
        """(name, CREATE) for objects the sandbox has and the selected schema files do not declare."""
        declared = {
            o.name for path, text in files for o in SchemaApplier.objects(path, text)
        } | {SchemaApplier.LEDGER, cls.RUNS}
        out = []
        for n, ddl in sorted(SchemaApplier.live(client, db).items()):
            if n in declared:
                continue
            for default in SchemaApplier.SERVER_DEFAULTS:
                ddl = ddl.replace(default, "")
            out.append((n, ddl.replace(f"{db}.", "")))
        return out


def main() -> None:
    from .client import ClickHouse, ClickHouseEnv

    parser = argparse.ArgumentParser(
        description="Per-person v2 sandbox databases on dev"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    boot = sub.add_parser(
        "bootstrap", help="Create the prod_v2 named collection (once per server)"
    )
    boot.add_argument(
        "--source-host", default="spyre-dashboard-clickhouse.clickhouse.svc"
    )
    boot.add_argument("--source-port", default="9000")
    boot.add_argument("--source-user", default="spyre_user")
    boot.add_argument("--source-database", default="spyre_v2")

    def seed_args(p):
        p.add_argument("--days", type=int, default=SeedFilter.days)
        p.add_argument(
            "--runs-per-component",
            type=int,
            default=SeedFilter.runs_per_component,
            help="Per component where the source has one; artifact_results has none, so its "
            "runs are capped in total",
        )
        p.add_argument("--component", action="append", default=[])
        p.add_argument("--arch", action="append", default=[])
        p.add_argument(
            "--tag",
            action="append",
            default=[],
            help="Runs of artifacts carrying this tag",
        )
        p.add_argument(
            "--run-id", action="append", default=[], help="Always include this run"
        )

    for cmd, desc in (
        ("create", "Create sandbox_<name>, apply schema/, seed it, print its login"),
        ("seed", "Append another prod sample to an existing sandbox"),
        ("diff", "Show how the sandbox differs from schema/ (what a PR must carry)"),
        ("drop", "Drop sandbox_<name> and its login"),
    ):
        p = sub.add_parser(cmd, help=desc)
        p.add_argument("--name", required=True)
        if cmd in ("create", "diff"):
            p.add_argument("--schema-dir", type=Path, default=SCHEMA_DIR)
        if cmd in ("create", "seed"):
            seed_args(p)
        if cmd == "create":
            p.add_argument("--no-seed", action="store_true")
            p.add_argument(
                "--replace", action="store_true", help="Drop an existing sandbox first"
            )
    sub.add_parser("list", help="List sandboxes on the server")
    args = parser.parse_args()

    # diff needs no rights outside the sandbox, so its own login can run it.
    if args.cmd == "diff":
        db = Sandbox.database(args.name)
        steps, extra = Sandbox.diff(
            ClickHouse.connect(database=db), db, args.schema_dir
        )
        for action, name, detail in steps:
            label = {
                "create": "removed",
                "drift": "changed",
                "recreate": "view-changed",
            }.get(action, action)
            print(f"  {label:12} {name}" + (f"\n{detail}" if action == "drift" else ""))
        for name, ddl in extra:
            print(f"  {'added':12} {name}\n    {ddl}")
        print(
            f"[diff] {len(steps) + len(extra)} difference(s) between {db} and {args.schema_dir}"
        )
        sys.exit(1 if steps or extra else 0)

    client = ClickHouse.connect(database="default")
    if args.cmd == "bootstrap":
        pw = ClickHouseEnv.get("SANDBOX_SOURCE_PASS")
        if not pw:
            raise SystemExit(
                "[error] SANDBOX_SOURCE_PASS (prod read-only password) is unset"
            )
        # NOT OVERRIDABLE: otherwise a sandbox login could pass host= to remote() and receive
        # the prod password. ALTER after CREATE so a rotated password replaces the stored one.
        keys = (
            "host = %(h)s NOT OVERRIDABLE, port = %(p)s NOT OVERRIDABLE, "
            "user = %(u)s NOT OVERRIDABLE, password = %(pw)s NOT OVERRIDABLE, "
            "database = %(d)s NOT OVERRIDABLE"
        )
        params = {
            "h": args.source_host,
            "p": int(args.source_port),
            "u": args.source_user,
            "pw": pw,
            "d": args.source_database,
        }
        client.command(
            f"CREATE NAMED COLLECTION IF NOT EXISTS {Sandbox.SOURCE} AS {keys}",
            parameters=params,
        )
        client.command(
            f"ALTER NAMED COLLECTION {Sandbox.SOURCE} SET {keys}", parameters=params
        )
        n = client.command(f"SELECT count() FROM {Sandbox.src('artifacts')}")
        print(f"[info] {Sandbox.SOURCE} reaches {args.source_host}: {n} artifacts")
        return
    if args.cmd == "list":
        rows = client.query(
            "SELECT database, sum(total_rows), formatReadableSize(sum(total_bytes)) FROM system.tables "
            f"WHERE startsWith(database, '{Sandbox.PREFIX}') GROUP BY database ORDER BY database"
        ).result_rows
        for db, n, size in rows:
            print(f"  {db:40} {n or 0:>12} rows {size}")
        return

    db = Sandbox.database(args.name)
    f = (
        SeedFilter(
            args.days,
            args.runs_per_component,
            tuple(args.component),
            tuple(args.arch),
            tuple(args.tag),
            tuple(args.run_id),
        )
        if args.cmd in ("create", "seed")
        else None
    )
    exists = bool(client.command(f"EXISTS DATABASE {db}"))

    if args.cmd == "drop":
        Sandbox.drop(client, args.name)
        print(f"[info] dropped {db}")
        return
    if args.cmd == "seed" and not exists:
        raise SystemExit(f"[error] {db} does not exist -- run `create` first")

    if args.cmd == "create":
        if exists and not args.replace:
            raise SystemExit(f"[error] {db} exists -- pass --replace to rebuild it")
        try:
            steps = Sandbox.build(
                client,
                lambda d: ClickHouse.connect(database=d),
                args.name,
                args.schema_dir,
            )
        except SchemaDrift as e:
            raise SystemExit(f"[error] {e}") from None
        print(f"[info] {len(steps)} object(s) from {args.schema_dir} applied to {db}")
        password = secrets.token_urlsafe(24)
        Sandbox.grant(client, args.name, password)
        if not args.no_seed:
            for table, n in Sandbox.seed(client, db, f):
                print(f"  {table:28} {n:>10} rows")
        print(
            f"[login] host={ClickHouseEnv.host()} database={db} user={Sandbox.user(args.name)}"
        )
        print(f"[login] password={password}")
        return

    for table, n in Sandbox.seed(client, db, f):
        print(f"  {table:28} {n:>10} rows")


if __name__ == "__main__":
    main()
