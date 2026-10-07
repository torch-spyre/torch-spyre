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

"""Converges a database on schema/: creates what is missing, runs schema/migrations/ once
each, recreates changed views, and fails on table drift rather than ALTERing."""

import argparse
import difflib
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import regex as re


@dataclass(frozen=True)
class SchemaObject:
    """One CREATE statement: kind is 'table', 'mv' or 'view'."""

    kind: str
    name: str
    sql: str
    file: str


class SchemaDrift(RuntimeError):
    """A live table or MV differs from its file; only a migration may change one."""


class SchemaApplier:
    """Plans and applies schema/*.sql and schema/migrations/*.sql against one database."""

    # A trailing ';' is optional, and two files carry ';' inside comment prose, so
    # statements are split after stripping comments, not on every ';' in the text.
    LINE_COMMENT = re.compile(r"--[^\n]*")
    BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
    # Minimum server version per file, from a `-- NEEDS CLICKHOUSE >= X.Y` header line,
    # declared in the file so the requirement travels with the DDL that has it.
    NEEDS_VERSION = re.compile(
        r"--\s*NEEDS CLICKHOUSE >=\s*(\d+)\.(\d+)", re.IGNORECASE
    )
    # A file that belongs in a different database (e.g. otel) applies only when named.
    EXPLICIT = re.compile(r"^--\s*APPLY:\s*explicit\b", re.IGNORECASE | re.MULTILINE)
    # A migration written to be repeated, e.g. to re-key rows that stale writers keep producing.
    RERUNNABLE = re.compile(r"^--\s*RERUNNABLE\b", re.IGNORECASE | re.MULTILINE)
    # A migration statement after `-- IF TABLE EXISTS: <name>` runs only when that table exists,
    # so a migration can fix a table another schema file may not have created yet.
    IF_TABLE = re.compile(
        r"^--\s*IF TABLE EXISTS:\s*(\w+)\s*$", re.IGNORECASE | re.MULTILINE
    )
    CREATE = re.compile(
        r"^CREATE\s+(MATERIALIZED\s+VIEW|TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
        re.IGNORECASE,
    )
    # The server stores a view with its resolved column list and adds default settings;
    # both are stripped so a stored definition compares equal to the file that made it.
    VIEW_COLUMNS = re.compile(
        r"^(CREATE (?:MATERIALIZED )?VIEW \S+(?: TO \S+)?) \(.*?\) AS (SELECT|WITH)\b"
    )
    SERVER_DEFAULTS = (" SETTINGS index_granularity = 8192",)
    ALTER = re.compile(r"^ALTER\s+TABLE\s+(?:\w+\.)?(\w+)", re.IGNORECASE)
    ADDS = re.compile(
        r"\bADD\s+(?:COLUMN|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?", re.IGNORECASE
    )
    # A migration replaces a CHECK by DROP + ADD CONSTRAINT; the live table still has the old one.
    ADDS_CONSTRAINT = re.compile(
        r"\bADD\s+CONSTRAINT\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?", re.IGNORECASE
    )
    LEDGER = "schema_migrations"
    LEDGER_DDL = (
        "CREATE TABLE IF NOT EXISTS schema_migrations (migration_id String, "
        "checksum String, applied_at DateTime DEFAULT now()) "
        "ENGINE = MergeTree ORDER BY migration_id"
    )

    @staticmethod
    def schema_dir() -> Path:
        """Where the .sql files live: a checkout's sibling dir, else the install."""
        here = Path(__file__).resolve().parent
        sibling = here.parent / "schema"
        return sibling if sibling.is_dir() else here / "schema"

    @classmethod
    def sql_files(cls, schema_dir: Path) -> list:
        """The DDL files in apply order: filename order (prefix encodes dependency)."""
        return sorted(schema_dir.glob("*.sql"))

    @classmethod
    def migration_files(cls, schema_dir: Path) -> list:
        """One-shot migrations in apply order; each runs once per database, ever."""
        return sorted((schema_dir / "migrations").glob("*.sql"))

    @classmethod
    def statements(cls, text: str) -> list:
        """The executable statements in one file, comments removed."""
        stripped = cls.LINE_COMMENT.sub("", cls.BLOCK_COMMENT.sub("", text))
        return [s.strip() for s in stripped.split(";") if s.strip()]

    @classmethod
    def run_migration(cls, client, text: str) -> None:
        """Run a migration's statements, skipping each one whose IF_TABLE is absent."""
        marked = cls.IF_TABLE.sub(lambda m: f"\0{m.group(1)}\0", text)
        for stmt in cls.statements(marked):
            table, _, sql = stmt.rpartition("\0")
            table = table.strip("\0")
            if table and not client.command(f"EXISTS TABLE {table}"):
                continue
            client.command(sql.strip())

    @classmethod
    def required_version(cls, text: str) -> tuple:
        """The (major, minor) floor this file declares, or () when it declares none."""
        m = cls.NEEDS_VERSION.search(text)
        return (int(m.group(1)), int(m.group(2))) if m else ()

    @staticmethod
    def version_tuple(text: str) -> tuple:
        return tuple(int(p) for p in re.findall(r"\d+", text)[:2])

    @staticmethod
    def checksum(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @classmethod
    def objects(cls, path: Path, text: str) -> list:
        """The CREATE statements of one schema file; anything else belongs in migrations/."""
        out = []
        for stmt in cls.statements(text):
            m = cls.CREATE.match(stmt)
            if not m:
                raise ValueError(
                    f"{path.name}: only CREATE statements belong in schema/ -- move "
                    f"'{stmt.splitlines()[0][:80]}' to schema/migrations/"
                )
            kind = {"table": "table", "view": "view"}.get(m.group(1).lower(), "mv")
            out.append(SchemaObject(kind, m.group(2), stmt, path.name))
        return out

    @classmethod
    def selected_files(cls, schema_dir: Path, include=(), server: tuple = ()) -> list:
        """(path, text) for every file this run applies, in order."""
        chosen = []
        for path in cls.sql_files(schema_dir):
            text = path.read_text()
            if cls.EXPLICIT.search(text) and path.name not in include:
                print(f"  {path.name:34} skipped -- APPLY: explicit, not --include'd")
                continue
            need = cls.required_version(text)
            if need and server and server < need:
                print(
                    f"  {path.name:34} SKIPPED -- needs ClickHouse >= "
                    f"{need[0]}.{need[1]}, server is {server[0]}.{server[1]}"
                )
                continue
            chosen.append((path, text))
        return chosen

    @classmethod
    def canonical(cls, client, sql: str, db: str) -> str:
        """One definition in the server's own formatting, database-unqualified."""
        # Client-side binding: a DDL-sized value overflows a server-side URL parameter.
        s = client.query(
            "SELECT formatQuerySingleLine(%(s)s)", parameters={"s": sql}
        ).result_rows[0][0]
        s = s.replace(" IF NOT EXISTS", "", 1).replace(f"{db}.", "")
        for default in cls.SERVER_DEFAULTS:
            s = s.replace(default, "")
        return cls.VIEW_COLUMNS.sub(r"\1 AS \2", s, count=1)

    @classmethod
    def live(cls, client, db: str) -> dict:
        """name -> stored CREATE statement, for every object in `db`."""
        rows = client.query(
            "SELECT name, create_table_query FROM system.tables "
            "WHERE database = %(db)s",
            parameters={"db": db},
        ).result_rows
        return {name: ddl for name, ddl in rows}

    @classmethod
    def applied(cls, client, db: str) -> dict:
        """migration_id -> checksum already recorded; {} before the ledger exists."""
        if not bool(client.command(f"EXISTS TABLE {db}.{cls.LEDGER}")):
            return {}
        rows = client.query(
            f"SELECT migration_id, checksum FROM {db}.{cls.LEDGER}"
        ).result_rows
        return {mid: chk for mid, chk in rows}

    @staticmethod
    def without_constraints(sql: str, names: set) -> str:
        """A single-line CREATE with the named CONSTRAINT ... CHECK clauses removed."""
        for n in names:
            sql = re.sub(
                rf",\s*CONSTRAINT\s+`?{re.escape(n)}`?\s+CHECK\s.*?"
                r"(?=,\s*(?:CONSTRAINT|INDEX|PROJECTION)\s|\)\s*ENGINE\b)",
                "",
                sql,
            )
        return sql

    @classmethod
    def differs(
        cls, client, obj: SchemaObject, stored: str, db: str, ignore=frozenset()
    ) -> str:
        """'' when `stored` matches the file (less `ignore` constraints), else a short diff."""
        want = cls.without_constraints(cls.canonical(client, obj.sql, db), ignore)
        have = cls.without_constraints(cls.canonical(client, stored, db), ignore)
        if want == have:
            return ""
        split = lambda s: s.replace(", ", ",\n").splitlines()  # noqa: E731
        diff = difflib.unified_diff(
            split(have), split(want), "live", obj.file, lineterm="", n=0
        )
        return "\n".join(list(diff)[:40])

    @classmethod
    def pending_adds(cls, pending: list, pattern=None) -> dict:
        """table -> column/index (or, given ADDS_CONSTRAINT, constraint) names pending migrations ADD."""
        out: dict = {}
        for p in pending:
            for stmt in cls.statements(p.read_text()):
                m = cls.ALTER.match(stmt)
                if m:
                    out.setdefault(m.group(1), set()).update(
                        (pattern or cls.ADDS).findall(stmt)
                    )
        return out

    @classmethod
    def without(cls, obj: SchemaObject, names: set) -> SchemaObject:
        """obj with the column/index definitions named in `names` removed (obj itself if none)."""
        if not names:
            return obj
        alt = "|".join(re.escape(n) for n in names)
        sql = re.sub(rf"(?m)^\s*(?:INDEX\s+)?`?(?:{alt})`?\s[^\n]*\n", "", obj.sql)
        sql = re.sub(r",(\s*\))", r"\1", sql)
        return SchemaObject(obj.kind, obj.name, sql, obj.file)

    @classmethod
    def plan(cls, client, db: str, files: list, migrations: list) -> list:
        """(action, name, detail) for every change an apply would make, in order."""
        live = cls.live(client, db)
        done = cls.applied(client, db)
        objs = [o for path, text in files for o in cls.objects(path, text)]
        pending = [p for p in migrations if p.name not in done]
        added = cls.pending_adds(pending)
        replaced = cls.pending_adds(pending, cls.ADDS_CONSTRAINT)
        # An MV a pending migration drops and recreates matches its file once that migration runs.
        recreated = {
            m.group(2)
            for p in pending
            for stmt in cls.statements(p.read_text())
            if (m := cls.CREATE.match(stmt))
        }
        steps = []
        for o in objs:
            if o.kind == "view":
                continue
            if o.name not in live:
                steps.append(("create", o.name, o.file))
            else:
                diff = cls.differs(client, o, live[o.name], db)
                if diff:
                    # Expected only if the table matches once what a pending migration adds (and
                    # the live table lacks: the ADDs are IF NOT EXISTS) is left out.
                    stored = live[o.name]
                    missing = {
                        n
                        for n in added.get(o.name, set())
                        if not re.search(rf"\b{n}\b", stored)
                    }
                    before = cls.without(o, missing)
                    ignore = replaced.get(o.name, set())
                    resolved = o.name in recreated or (
                        (before is not o or bool(ignore))
                        and not cls.differs(client, before, stored, db, ignore)
                    )
                    steps.append(("migrates" if resolved else "drift", o.name, diff))
        steps += [("migrate", p.name, "") for p in pending]
        for o in objs:
            if o.kind != "view":
                continue
            if o.name not in live:
                steps.append(("create", o.name, o.file))
            elif cls.differs(client, o, live[o.name], db):
                steps.append(("recreate", o.name, o.file))
        return steps

    @classmethod
    def apply(cls, client, db: str, files: list, migrations: list) -> list:
        """Converge `db` on the files. Tables/MVs, then migrations, then views."""
        client.command(cls.LEDGER_DDL)
        objs = [o for path, text in files for o in cls.objects(path, text)]
        steps = []
        live = cls.live(client, db)
        for o in objs:
            if o.kind != "view" and o.name not in live:
                client.command(o.sql)
                steps.append(("create", o.name, o.file))
        done = cls.applied(client, db)
        for path in migrations:
            text = path.read_text()
            if path.name in done:
                if done[path.name] != cls.checksum(text):
                    print(f"  [warn] {path.name} changed after it was applied")
                continue
            cls.run_migration(client, text)
            client.insert(
                cls.LEDGER,
                [[path.name, cls.checksum(text)]],
                column_names=["migration_id", "checksum"],
                database=db,
            )
            steps.append(("migrate", path.name, ""))
        live = cls.live(client, db)
        drift = [
            (o.name, d)
            for o in objs
            if o.kind != "view" and (d := cls.differs(client, o, live[o.name], db))
        ]
        if drift:
            raise SchemaDrift(
                "live definition differs from schema/ -- add a migration:\n"
                + "\n".join(f"{name}:\n{d}" for name, d in drift)
            )
        for o in objs:
            if o.kind != "view":
                continue
            if o.name not in live:
                client.command(o.sql)
                steps.append(("create", o.name, o.file))
            elif cls.differs(client, o, live[o.name], db):
                # A plain view holds no data; the server rejects CREATE OR REPLACE.
                client.command(f"DROP VIEW IF EXISTS {o.name}")
                client.command(o.sql)
                steps.append(("recreate", o.name, o.file))
        return steps

    @classmethod
    def rerun(cls, client, path: Path) -> None:
        """Run a migration again; refused unless it is marked `-- RERUNNABLE`."""
        text = path.read_text()
        if not cls.RERUNNABLE.search(text):
            raise ValueError(f"{path.name} is not marked -- RERUNNABLE")
        cls.run_migration(client, text)


SCHEMA_DIR = SchemaApplier.schema_dir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converge a ClickHouse database on the v2 schema DDL"
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=SCHEMA_DIR,
        help=f"Directory of *.sql files (default: {SCHEMA_DIR})",
    )
    parser.add_argument(
        "--database",
        default="",
        help="Target database (default: $CLICKHOUSE_DB_V2, else $CLICKHOUSE_DB)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Also apply this APPLY: explicit file (repeatable)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="List the statements, without connecting",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Print what an apply would change; exit 1 if anything would, 2 on drift",
    )
    mode.add_argument(
        "--rerun",
        metavar="MIGRATION",
        help="Run one -- RERUNNABLE migration again (e.g. 007_case_id_keep_name_case.sql)",
    )
    args = parser.parse_args()

    if not SchemaApplier.sql_files(args.schema_dir):
        print(f"[error] No .sql files in {args.schema_dir}", file=sys.stderr)
        sys.exit(1)
    migrations = SchemaApplier.migration_files(args.schema_dir)

    if args.dry_run:
        print(f"[dry-run] {args.schema_dir}:")
        for path, text in SchemaApplier.selected_files(args.schema_dir, args.include):
            for o in SchemaApplier.objects(path, text):
                print(f"  {path.name:34} {o.kind:5} {o.name}")
        for path in migrations:
            print(f"  migrations/{path.name}")
        return

    from .client import ClickHouse, ClickHouseEnv

    db = args.database or ClickHouseEnv.target_database() or ClickHouseEnv.database()
    client = ClickHouse.connect(database=db)
    server = SchemaApplier.version_tuple(client.command("SELECT version()"))
    files = SchemaApplier.selected_files(args.schema_dir, args.include, server)
    print(f"[info] {ClickHouseEnv.host()}/{db}, ClickHouse {server[0]}.{server[1]}")

    if args.rerun:
        SchemaApplier.rerun(client, args.schema_dir / "migrations" / args.rerun)
        print(f"[info] reran {args.rerun} on {db}")
        return

    if args.check:
        steps = SchemaApplier.plan(client, db, files, migrations)
        for action, name, detail in steps:
            print(f"  {action:8} {name}" + (f"\n{detail}" if action == "drift" else ""))
        print(f"[check] {len(steps)} change(s) pending")
        if any(action == "drift" for action, _, _ in steps):
            sys.exit(2)
        sys.exit(1 if steps else 0)

    try:
        steps = SchemaApplier.apply(client, db, files, migrations)
    except SchemaDrift as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
    for action, name, _ in steps:
        print(f"  {action:8} {name}")
    print(f"[info] {len(steps)} change(s) applied to {db}")


if __name__ == "__main__":
    main()
