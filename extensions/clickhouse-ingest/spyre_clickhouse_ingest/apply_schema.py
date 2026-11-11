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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import regex as re


@dataclass(frozen=True)
class SchemaObject:
    """One CREATE statement: kind is 'table', 'mv', 'refresh' (a refreshable MV) or 'view'."""

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
    # A refreshable MV runs a query, so unlike an insert-triggered one it may read views; it is
    # created after them.
    REFRESHABLE = re.compile(
        r"^CREATE\s+MATERIALIZED\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+REFRESH\b",
        re.IGNORECASE,
    )
    # The server stores a view with its resolved column list (a refreshable MV also with its
    # default DEFINER) and adds default settings; all are stripped so a stored definition
    # compares equal to the file that made it.
    VIEW_COLUMNS = re.compile(
        r"^(CREATE (?:MATERIALIZED )?VIEW \S+(?: REFRESH [^(]*?)?(?: TO \S+)?) \(.*?\)"
        r"(?: DEFINER = \S+ SQL SECURITY DEFINER)? AS (SELECT|WITH)\b"
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

    # Engines the applier can make Replicated; a table on any other engine is refused in cluster mode.
    REPLICATED = {
        "MergeTree": "ReplicatedMergeTree",
        "ReplacingMergeTree": "ReplicatedReplacingMergeTree",
        "SummingMergeTree": "ReplicatedSummingMergeTree",
        "AggregatingMergeTree": "ReplicatedAggregatingMergeTree",
    }
    # Where a table made by the applier keeps its replication metadata: CLICKHOUSE_KEEPER_PATH, a
    # template with {shard}, else /clickhouse/<cluster>/tables/{shard}/<db>/<table>.
    KEEPER_PATH = "/clickhouse/%s/tables/{shard}/%s/%s"
    # The stored form of a Replicated engine: 'path', 'replica' and then the plain engine's own arguments.
    REPLICATED_ENGINE = re.compile(
        r"\bENGINE\s*=\s*Replicated(\w*MergeTree)\s*\(\s*'[^']*'\s*,\s*'[^']*'\s*(?:,\s*([^()]*?))?\s*\)"
    )
    # A conversion marks its table with a comment; it is not part of the file's definition.
    COMMENT = re.compile(r"\s+COMMENT\s+'[^']*'\s*$")
    # Statements in a migration that change structure on every replica, and the ones that only move data.
    MIGRATION_DDL = re.compile(
        r"^(ALTER\s+TABLE|DROP\s+TABLE|DROP\s+VIEW|TRUNCATE\s+TABLE|CREATE\s+(?:MATERIALIZED\s+)?VIEW|"
        r"CREATE\s+TABLE|RENAME\s+TABLE|EXCHANGE\s+TABLES)\b",
        re.IGNORECASE,
    )
    # Anything that changes structure but that cluster mode has no rewrite for is refused rather
    # than run on the connected server alone.
    UNCLUSTERABLE = re.compile(
        r"^(?:CREATE|ALTER|DROP|RENAME|EXCHANGE|TRUNCATE|ATTACH|DETACH|UNDROP|OPTIMIZE|SYSTEM|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )
    # An identifier, plain or `quoted` / "quoted", optionally db-qualified.
    IDENT = r'(?:`(?:[^`\\]|``|\\.)+`|"(?:[^"\\]|""|\\.)+"|\w+)'
    QNAME = rf"(?:{IDENT}\.)?{IDENT}"
    MIGRATION_CREATE_TABLE = re.compile(
        rf"^CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?({QNAME})", re.IGNORECASE
    )
    INSERT_INTO = re.compile(
        rf"^(?:WITH\b.*?\b)?INSERT\s+INTO\s+(?!FUNCTION\b)(?:TABLE\s+)?({QNAME})",
        re.IGNORECASE | re.DOTALL,
    )
    LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'")
    # Long enough for a large table to catch up; the default (180 s) fails the migration while
    # the sync carries on in the background.
    SYNC_SETTINGS = {"distributed_ddl_task_timeout": 3600}
    CREATE_AS_SELECT = re.compile(r"\bAS\s+(?:SELECT|WITH)\b", re.IGNORECASE)

    @classmethod
    def cluster(cls) -> str:
        """The cluster DDL runs on: CLICKHOUSE_CLUSTER, or '' for a single server (the default)."""
        name = os.environ.get("CLICKHOUSE_CLUSTER", "").strip()
        if name and not re.fullmatch(r"\w+", name):
            raise ValueError("CLICKHOUSE_CLUSTER must be a plain identifier")
        return name

    @classmethod
    def plain_engine(cls, sql: str) -> str:
        """`sql` with a Replicated*MergeTree engine written as its plain engine, so the two compare equal."""
        out, last = [], 0
        for m in cls.REPLICATED_ENGINE.finditer(sql):
            if cls._in_literal(sql, m.start()):
                continue
            args = m.group(2)
            out.append(sql[last : m.start()])
            out.append(f"ENGINE = {m.group(1)}" + (f"({args.strip()})" if args else ""))
            last = m.end()
        out.append(sql[last:])
        text = "".join(out)
        # The conversion marker is only dropped from a converted table, so a plain table's
        # COMMENT still counts as drift.
        return cls.COMMENT.sub("", text) if last else text

    @staticmethod
    def _in_literal(sql: str, pos: int) -> bool:
        """True when `pos` falls inside a single-quoted string literal."""
        return len(re.findall(r"(?<!\\)'", sql[:pos])) % 2 == 1

    @classmethod
    def mask(cls, sql: str) -> str:
        """`sql` with the inside of every string literal blanked (same length), so a keyword or
        parenthesis in a literal is never matched."""
        return cls.LITERAL.sub(lambda m: "'" + "_" * (len(m.group()) - 2) + "'", sql)

    @classmethod
    def bare(cls, qname: str) -> str:
        """The table part of a possibly qualified or quoted name."""
        last = re.findall(cls.IDENT, qname)[-1]
        if last[0] in '`"':
            return last[1:-1].replace(last[0] * 2, last[0])
        return last

    @staticmethod
    def quote(name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    @classmethod
    def sync_target(cls, token: str, db: str) -> str:
        """`token` as written, qualified with `db` when it is not already."""
        if re.fullmatch(rf"{cls.IDENT}\.{cls.IDENT}", token) or not db:
            return token
        return f"{cls.quote(db)}.{token}"

    @classmethod
    def insert_target(cls, stmt: str) -> str:
        """The table token an INSERT statement writes to, as written, else ''."""
        m = cls.INSERT_INTO.match(cls.mask(stmt))
        return m.group(1) if m else ""

    @classmethod
    def mentions(cls, masked: str, name: str) -> bool:
        forms = (name, cls.quote(name), '"' + name.replace('"', '""') + '"')
        return any(
            re.search(rf"(?<![\w`\"]){re.escape(q)}(?![\w`\"])", masked) for q in forms
        )

    @classmethod
    def engine_span(cls, masked: str):
        """(start, end, name, args) of the ENGINE clause in `masked`, args balanced; None if absent."""
        m = re.search(r"\bENGINE\s*=\s*(\w+)", masked, re.I)
        if not m:
            return None
        k = m.end()
        while masked[k : k + 1].isspace():
            k += 1
        if masked[k : k + 1] != "(":
            return m.start(), m.end(), m.group(1), ""
        depth = 0
        for i in range(k, len(masked)):
            depth += masked[i] == "("
            depth -= masked[i] == ")"
            if depth == 0:
                return m.start(), i + 1, m.group(1), (k, i)
        raise ValueError(
            f"unbalanced parentheses in ENGINE clause: '{masked[m.start() :][:80]}'"
        )

    @classmethod
    def clustered(cls, sql: str, kind: str, cluster: str, db: str, name: str) -> str:
        """A schema file's CREATE for cluster mode: ON CLUSTER, and a Replicated engine for a table.

        A refreshable MV is returned unchanged: it would append once per replica, so it is created on
        one server only. Anything it cannot place on every replica raises."""
        if not cluster or kind == "refresh":
            return sql
        head = re.compile(
            rf"^(CREATE\s+(?:MATERIALIZED\s+VIEW|TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?{cls.QNAME})",
            re.I,
        )
        h = head.match(cls.mask(sql))
        if not h:
            raise ValueError(f"{name}: cannot place ON CLUSTER in '{sql[:80]}'")
        if re.match(r"\s+ON\s+CLUSTER\b", cls.mask(sql)[h.end() :], re.I):
            return sql
        sql = sql[: h.end()] + f" ON CLUSTER {cluster}" + sql[h.end() :]
        mk = cls.mask(sql)
        if kind == "mv":
            asq = cls.CREATE_AS_SELECT.search(mk)
            prefix = mk[: asq.start()] if asq else mk
            if re.search(r"\bENGINE\s*=", prefix) and not re.search(
                r"\bTO\s+\S", prefix
            ):
                raise ValueError(
                    f"{name}: a materialized view with its own ENGINE is not supported "
                    "in cluster mode; use TO <table>"
                )
        if kind != "table":
            return sql
        bare_mk = mk.replace(f" ON CLUSTER {cluster}", "", 1)
        if cls.CREATE_AS_SELECT.search(mk) or re.match(
            rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{cls.QNAME}[^()]*?\bAS\s+{cls.QNAME}",
            bare_mk,
            re.I,
        ):
            raise ValueError(
                f"{name}: CREATE TABLE ... AS <table or SELECT> is not supported in cluster mode"
            )
        eng = cls.engine_span(mk)
        if not eng or eng[2] not in cls.REPLICATED:
            raise ValueError(
                f"{name}: cannot make engine {eng[2] if eng else '?'} Replicated"
            )
        start, end, engine, span = eng
        args = sql[span[0] + 1 : span[1]].strip() if span else ""
        path = os.environ.get("CLICKHOUSE_KEEPER_PATH", "").strip()
        if path and "{table}" not in path and "{uuid}" not in path:
            raise ValueError(
                "CLICKHOUSE_KEEPER_PATH must contain {table} (or {uuid}), else every table shares one path"
            )
        path = (
            path.replace("{db}", db).replace("{table}", name)
            if path
            else cls.KEEPER_PATH % (cluster, db, name)
        )
        new = f"'{path}', '{{replica}}'" + (f", {args}" if args else "")
        tail = sql[end:]
        return (
            sql[:start]
            + f"ENGINE = {cls.REPLICATED[engine]}({new})"
            + ("" if tail[:1].isspace() or not tail else " ")
            + tail
        )

    @classmethod
    def clustered_migration(cls, stmt: str, cluster: str, db: str = "") -> str:
        """A migration statement that changes structure, run on every replica; data statements stay as they are.

        A table a migration creates is made Replicated like a schema table, so a scratch table holds
        the same rows on every replica. A refreshable MV stays on the connected server (it appends
        on a schedule). Structural DDL with no rewrite raises instead of running on one server.
        Matching ignores string literals, and ON CLUSTER counts only where the statement puts it."""
        if not cluster:
            return stmt
        mk = cls.mask(stmt)
        if cls.REFRESHABLE.match(mk):
            return stmt

        def has_cluster(end: int) -> bool:
            return bool(re.match(r"\s+ON\s+CLUSTER\b", mk[end:], re.I))

        ct = cls.MIGRATION_CREATE_TABLE.match(mk)
        if ct:
            if cls.CREATE_AS_SELECT.search(mk):
                raise ValueError(
                    f"{cls.bare(ct.group(1))}: CREATE TABLE ... AS SELECT is not supported in cluster mode"
                )
            if not re.search(r"\bENGINE\s*=\s*Replicated", mk, re.I):
                if has_cluster(ct.end()):
                    raise ValueError(
                        f"{cls.bare(ct.group(1))}: ON CLUSTER needs a Replicated engine"
                    )
                return cls.clustered(stmt, "table", cluster, db, cls.bare(ct.group(1)))
        cmv = re.match(
            rf"^CREATE\s+MATERIALIZED\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?({cls.QNAME})",
            mk,
            re.I,
        )
        if cmv:
            if has_cluster(cmv.end()):
                return stmt
            return cls.clustered(stmt, "mv", cluster, db, cls.bare(cmv.group(1)))
        m = cls.MIGRATION_DDL.match(mk)
        if not m:
            dm = re.match(rf"^DELETE\s+FROM\s+{cls.QNAME}", mk, re.I)
            if dm:
                if has_cluster(dm.end()):
                    return stmt
                return stmt[: dm.end()] + f" ON CLUSTER {cluster}" + stmt[dm.end() :]
            if cls.UNCLUSTERABLE.match(mk):
                raise ValueError(
                    f"cannot run on a cluster: '{stmt.splitlines()[0][:80]}'"
                )
            return stmt
        verb = re.sub(r"\s+", " ", m.group(1).upper())
        if verb in ("RENAME TABLE", "EXCHANGE TABLES"):
            if re.search(r"\bSETTINGS\b", mk, re.I):
                raise ValueError(
                    f"cannot run on a cluster (SETTINGS after {verb}): '{stmt[:80]}'"
                )
            if re.search(rf"\bON\s+CLUSTER\s+{cls.IDENT}\s*$", mk, re.I):
                return stmt
            return f"{stmt} ON CLUSTER {cluster}"
        target = rf"(?:IF\s+(?:NOT\s+)?EXISTS\s+)?{cls.QNAME}"
        pat = {
            "ALTER TABLE": rf"^(ALTER\s+TABLE\s+{target})",
            "TRUNCATE TABLE": rf"^(TRUNCATE\s+TABLE\s+{target})",
            "DROP TABLE": rf"^(DROP\s+TABLE\s+{target})",
            "DROP VIEW": rf"^(DROP\s+VIEW\s+{target})",
        }.get(verb)
        if verb.startswith("CREATE"):
            pat = rf"^(CREATE\s+(?:MATERIALIZED\s+)?(?:VIEW|TABLE)\s+{target})"
        tm = re.match(pat, mk, re.I)
        if not tm:
            raise ValueError(f"cannot place ON CLUSTER in '{stmt[:80]}'")
        if has_cluster(tm.end()):
            return stmt
        if verb in ("DROP TABLE", "DROP VIEW") and re.match(r"\s*,", mk[tm.end() :]):
            raise ValueError(f"cannot run on a cluster (several tables): '{stmt[:80]}'")
        out = stmt[: tm.end()] + f" ON CLUSTER {cluster}" + stmt[tm.end() :]
        if verb == "DROP TABLE" and not re.search(r"\b(?:SYNC|NO\s+DELAY)\b", mk, re.I):
            out += " SYNC"
        return out

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
    def migration_statements(cls, text: str, cluster: str = "", db: str = "") -> list:
        """(IF_TABLE name or '', statement as it will run) for each statement of a migration."""
        marked = cls.IF_TABLE.sub(lambda m: f"\0{m.group(1)}\0", text)
        out = []
        for stmt in cls.statements(marked):
            table, _, sql = stmt.rpartition("\0")
            out.append(
                (table.strip("\0"), cls.clustered_migration(sql.strip(), cluster, db))
            )
        return out

    @classmethod
    def run_migration(
        cls, client, text: str, cluster: str = "", db: str = "", unsynced=None
    ) -> None:
        """Run a migration's statements, skipping each one whose IF_TABLE is absent.

        With a cluster, structural statements run on every replica. Other data statements run once, on
        the connected server, and reach the others by replication; a table inserted into is synced
        on every replica before a later statement reads it. `unsynced` carries that across migrations."""
        unsynced = {} if unsynced is None else unsynced
        for table, sql in cls.migration_statements(text, cluster, db):
            if table and not client.command(f"EXISTS TABLE {table}"):
                continue
            mk = cls.mask(sql)
            if cluster and not re.match(r"(?:DROP|TRUNCATE)\b", mk, re.I):
                for name in sorted(unsynced):
                    if cls.mentions(mk, name):
                        for target in sorted(unsynced.pop(name)):
                            client.command(
                                f"SYSTEM SYNC REPLICA ON CLUSTER {cluster} {target}",
                                settings=cls.SYNC_SETTINGS,
                            )
            client.command(sql)
            token = cls.insert_target(sql) if cluster else ""
            if token:
                unsynced.setdefault(cls.bare(token), set()).add(
                    cls.sync_target(token, db)
                )

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
            if cls.REFRESHABLE.match(stmt):
                kind = "refresh"
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
        s = cls.plain_engine(s)
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
    def refresh_mvs_on_cluster(
        cls, client, db: str, cluster: str, ledger_ready: bool = True
    ) -> set:
        """Names of materialized views present on ANY replica; empty without a cluster.

        A refreshable MV exists on one replica only, so the connected server alone cannot tell
        whether another already owns it. system.tables lists only what the querying user may see,
        and remote replicas run this as the cluster's inter-server user, so a replica that cannot
        see the ledger table would hide its MVs too: that raises rather than report "none"."""
        if not cluster or not ledger_ready:
            return set()
        try:
            hosts = {
                tuple(r)
                for r in client.query(
                    f"SELECT hostName(), tcpPort() FROM clusterAllReplicas('{cluster}', system.one)"
                ).result_rows
            }
            rows = client.query(
                "SELECT hostName(), tcpPort(), countIf(name = %(ledger)s), "
                "groupArrayIf(name, engine = 'MaterializedView') "
                f"FROM clusterAllReplicas('{cluster}', system.tables) "
                "WHERE database = %(db)s GROUP BY hostName(), tcpPort()",
                parameters={"db": db, "ledger": cls.LEDGER},
            ).result_rows
        except Exception as e:
            raise RuntimeError(
                f"reading system.tables on every replica of cluster {cluster} failed: {e} -- "
                "a replica may be down or unreachable, or lack SELECT on system.tables; nothing "
                "past this point was applied, rerun once it is fixed"
            ) from e
        blind = hosts - {tuple(r[:2]) for r in rows if r[2]}
        if blind:
            raise RuntimeError(
                f"replica(s) {sorted(blind)} of cluster {cluster} do not show {cls.LEDGER} in "
                "system.tables: a missing grant, missing cluster secret or lagging replica would "
                "hide a refreshable MV; fix that and rerun"
            )
        return {n for r in rows for n in r[3]}

    @classmethod
    def require_replicated(cls, client, db: str, names) -> None:
        """Raise if any of `names` exists in `db` on a non-Replicated engine.

        Cluster mode syncs the ledger and tables a migration inserts into, which a plain table
        refuses part-way through a migration. Checked on the connected server before any write."""
        names = sorted(set(names))
        if not names:
            return
        rows = client.query(
            "SELECT name, engine FROM system.tables WHERE database = %(db)s AND has(%(names)s, name)",
            parameters={"db": db, "names": names},
        ).result_rows
        plain = sorted(n for n, e in rows if not e.startswith("Replicated"))
        if plain:
            raise ValueError(
                f"cluster mode needs every table converted to a Replicated engine first; "
                f"not Replicated in {db}: {', '.join(plain)} (see schema/README.md, "
                "'Applying on a Replicated cluster')"
            )

    @classmethod
    def validate_cluster(
        cls, client, objs: list, live: dict, pending: list, cluster: str, db: str
    ) -> None:
        """Raise before anything is written if a new table or pending migration cannot run on a
        cluster, or a migration would insert into an existing table that is not Replicated."""
        if not cluster:
            return
        for o in objs:
            if o.kind in ("table", "mv") and o.name not in live:
                cls.clustered(o.sql, o.kind, cluster, db, o.name)
        inserted: set = set()
        created = {o.name for o in objs}
        stmts = {}
        for p in pending:
            try:
                stmts[p.name] = cls.migration_statements(p.read_text(), cluster, db)
            except ValueError as e:
                raise ValueError(f"{p.name}: {e}") from None
            for _, sql in stmts[p.name]:
                ct = cls.MIGRATION_CREATE_TABLE.match(cls.mask(sql))
                if ct:
                    created.add(cls.bare(ct.group(1)))
        for name, items in stmts.items():
            for guard, sql in items:
                token = cls.insert_target(sql)
                if not token:
                    continue
                qual = re.fullmatch(rf"({cls.IDENT})\.{cls.IDENT}", token)
                if qual and cls.bare(qual.group(1)) != db:
                    raise ValueError(
                        f"{name}: INSERT into {token} is outside {db}; cluster mode syncs only {db}"
                    )
                t = cls.bare(token)
                if t in live:
                    inserted.add(t)
                elif t not in created and not guard:
                    raise ValueError(
                        f"{name}: INSERT target {token} is not visible in {db}'s system.tables "
                        "(missing or no SELECT grant), so it cannot be checked or synced"
                    )
        cls.require_replicated(client, db, inserted)

    @classmethod
    def ledger_ddl(cls, cluster: str, db: str) -> str:
        """The migrations ledger's CREATE: plain on one server, Replicated ON CLUSTER in cluster mode."""
        if not cluster:
            return cls.LEDGER_DDL
        return cls.clustered(cls.LEDGER_DDL, "table", cluster, db, cls.LEDGER)

    @classmethod
    def applied(cls, client, db: str, cluster: str = "") -> dict:
        """migration_id -> checksum already recorded; {} before the ledger exists.

        In cluster mode this replica first catches up with the ledger, so a row inserted through
        another replica is not missed."""
        if not bool(client.command(f"EXISTS TABLE {db}.{cls.LEDGER}")):
            return {}
        if cluster:
            client.command(f"SYSTEM SYNC REPLICA {db}.{cls.LEDGER}")
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
    def plan(
        cls, client, db: str, files: list, migrations: list, cluster: str = ""
    ) -> list:
        """(action, name, detail) for every change an apply would make, in order.

        A Replicated table compares equal to its plain file, so a converted table is never drift;
        `cluster` only decides what the apply would do, not what the plan reports."""
        live = cls.live(client, db)
        if cluster:
            cls.require_replicated(client, db, [cls.LEDGER])
        done = cls.applied(client, db, cluster)
        objs = [o for path, text in files for o in cls.objects(path, text)]
        pending = [p for p in migrations if p.name not in done]
        cls.validate_cluster(client, objs, live, pending, cluster, db)
        elsewhere = cls.refresh_mvs_on_cluster(
            client,
            db,
            cluster,
            bool(cluster) and bool(client.command(f"EXISTS TABLE {db}.{cls.LEDGER}")),
        )
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
            if o.kind == "view" or (o.kind == "refresh" and o.name not in live):
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
        steps += [
            ("create", o.name, o.file)
            for o in objs
            if o.kind == "refresh" and o.name not in live and o.name not in elsewhere
        ]
        return steps

    @classmethod
    def apply(
        cls, client, db: str, files: list, migrations: list, cluster: str = ""
    ) -> list:
        """Converge `db` on the files. Tables/MVs, then migrations, then views, then refreshable MVs.

        `cluster` (default: none, one server) runs DDL ON CLUSTER and creates new tables on a
        Replicated engine. The migrations ledger is written once, by the connected server: it is a
        Replicated table in cluster mode, so the other replicas receive the row."""
        objs = [o for path, text in files for o in cls.objects(path, text)]
        steps = []
        if cluster:
            cls.require_replicated(client, db, [cls.LEDGER])
            done = cls.applied(client, db, cluster)
            cls.validate_cluster(
                client,
                objs,
                cls.live(client, db),
                [p for p in migrations if p.name not in done],
                cluster,
                db,
            )
        client.command(cls.ledger_ddl(cluster, db))
        unsynced: dict = {}
        cls.refresh_mvs_on_cluster(client, db, cluster)
        live = cls.live(client, db)
        for o in objs:
            if o.kind in ("table", "mv") and o.name not in live:
                client.command(cls.clustered(o.sql, o.kind, cluster, db, o.name))
                steps.append(("create", o.name, o.file))
        done = cls.applied(client, db, cluster)
        for path in migrations:
            text = path.read_text()
            if path.name in done:
                if done[path.name] != cls.checksum(text):
                    print(f"  [warn] {path.name} changed after it was applied")
                continue
            cls.run_migration(client, text, cluster, db, unsynced)
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
            if o.kind != "view"
            and o.name in live
            and (d := cls.differs(client, o, live[o.name], db))
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
                client.command(cls.clustered(o.sql, o.kind, cluster, db, o.name))
                steps.append(("create", o.name, o.file))
            elif cls.differs(client, o, live[o.name], db):
                # A plain view holds no data; the server rejects CREATE OR REPLACE.
                oc = f" ON CLUSTER {cluster}" if cluster else ""
                client.command(f"DROP VIEW IF EXISTS {o.name}{oc}")
                client.command(cls.clustered(o.sql, o.kind, cluster, db, o.name))
                steps.append(("recreate", o.name, o.file))
        elsewhere = cls.refresh_mvs_on_cluster(client, db, cluster)
        for o in objs:
            if o.kind == "refresh" and o.name not in live:
                if o.name in elsewhere:
                    continue
                # Never ON CLUSTER: it appends on a schedule, so one server must own it.
                if cluster:
                    print(
                        f"  [info] {o.name}: refreshable MV created on this server only"
                    )
                client.command(o.sql)
                steps.append(("create", o.name, o.file))
        return steps

    @classmethod
    def rerun(cls, client, path: Path, cluster: str = "", db: str = "") -> None:
        """Run a migration again; refused unless it is marked `-- RERUNNABLE`."""
        text = path.read_text()
        if not cls.RERUNNABLE.search(text):
            raise ValueError(f"{path.name} is not marked -- RERUNNABLE")
        if cluster:
            cls.validate_cluster(client, [], cls.live(client, db), [path], cluster, db)
        cls.run_migration(client, text, cluster, db)


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
    parser.add_argument(
        "--cluster",
        default="",
        help="Run DDL ON CLUSTER <name> and create tables on Replicated engines "
        "(default: $CLICKHOUSE_CLUSTER, else one server)",
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

    if args.cluster:
        os.environ["CLICKHOUSE_CLUSTER"] = args.cluster
    cluster = SchemaApplier.cluster()

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
    print(
        f"[info] {ClickHouseEnv.host()}/{db}, ClickHouse {server[0]}.{server[1]}"
        + (f", cluster {cluster}" if cluster else "")
    )

    if args.rerun:
        SchemaApplier.rerun(
            client, args.schema_dir / "migrations" / args.rerun, cluster, db
        )
        print(f"[info] reran {args.rerun} on {db}")
        return

    if args.check:
        steps = SchemaApplier.plan(client, db, files, migrations, cluster)
        for action, name, detail in steps:
            print(f"  {action:8} {name}" + (f"\n{detail}" if action == "drift" else ""))
        print(f"[check] {len(steps)} change(s) pending")
        if any(action == "drift" for action, _, _ in steps):
            sys.exit(2)
        sys.exit(1 if steps else 0)

    try:
        steps = SchemaApplier.apply(client, db, files, migrations, cluster)
    except SchemaDrift as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
    for action, name, _ in steps:
        print(f"  {action:8} {name}")
    print(f"[info] {len(steps)} change(s) applied to {db}")


if __name__ == "__main__":
    main()
