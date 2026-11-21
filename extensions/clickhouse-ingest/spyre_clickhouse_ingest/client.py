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

"""ClickHouse connection factory, the writer/schema logins, the v2-host router and the v2
table-presence gate."""

import os
import sys
import threading

import clickhouse_connect
import regex as re

from . import schema


class ClickHouseEnv:
    """The connection settings, read from the environment through one resolver."""

    DEFAULT_PORT = "443"
    DEFAULT_USER = "default"
    DEFAULT_DB = "spyre"
    DEFAULT_V2_DATABASES = ("spyre_v2",)

    @staticmethod
    def get(name: str, default: str = "") -> str:
        """An env var, treating BLANK as absent -- GHA exports an unset secret as ''."""
        return (os.environ.get(name) or "").strip() or default

    @classmethod
    def flag(cls, name: str) -> bool:
        return cls.get(name).lower() not in ("", "0", "false", "no")

    @classmethod
    def seconds(cls, name: str, default: int) -> int:
        """A timeout in seconds from the environment; a bad value falls back to the default."""
        try:
            return max(1, int(cls.get(name, str(default))))
        except ValueError:
            return default

    @classmethod
    def host(cls) -> str:
        """CLICKHOUSE_HOST, which has no default."""
        return cls.get("CLICKHOUSE_HOST")

    @classmethod
    def port(cls) -> str:
        """CLICKHOUSE_PORT as a raw string, so a non-numeric value can name itself."""
        return cls.get("CLICKHOUSE_PORT", cls.DEFAULT_PORT)

    @classmethod
    def database(cls) -> str:
        """CLICKHOUSE_DB, the connection's own database."""
        return cls.get("CLICKHOUSE_DB", cls.DEFAULT_DB)

    @classmethod
    def secure(cls) -> bool:
        """CLICKHOUSE_SECURE=0 drops to plain HTTP, to reach a local box with no TLS."""
        return cls.get("CLICKHOUSE_SECURE", "1") not in ("0", "false", "no")

    @classmethod
    def target_database(cls) -> str:
        """CLICKHOUSE_DB_V2: the v2 database NAME, or '' when v2 is not configured."""
        return os.environ.get("CLICKHOUSE_DB_V2", "").strip()

    @classmethod
    def v2_host(cls) -> str:
        """CLICKHOUSE_V2_HOST: the endpoint for the v2 databases; '' keeps one host."""
        return cls.get("CLICKHOUSE_V2_HOST")

    @classmethod
    def v2_port(cls) -> str:
        return cls.get("CLICKHOUSE_V2_PORT") or cls.port()

    @classmethod
    def v2_secure(cls) -> bool:
        raw = cls.get("CLICKHOUSE_V2_SECURE")
        return cls.secure() if not raw else raw not in ("0", "false", "no")

    @classmethod
    def v2_databases(cls) -> tuple:
        """CH_V2_DATABASES (comma list) plus CLICKHOUSE_DB_V2: the databases the v2 host serves."""
        listed = tuple(
            x.strip() for x in cls.get("CH_V2_DATABASES").split(",") if x.strip()
        )
        listed = listed or cls.DEFAULT_V2_DATABASES
        extra = cls.target_database()
        return listed + ((extra,) if extra and extra not in listed else ())

    @classmethod
    def v2_default_database(cls) -> str:
        """The v2 client's own database; never the v1 one."""
        return cls.target_database() or cls.v2_databases()[0]

    @classmethod
    def summary(cls) -> str:
        """A host:port/database string, resolved exactly as the connection is."""
        return f"{cls.host()}:{cls.port()}/{cls.database()}"


class SchemaStatementOnWriter(RuntimeError):
    """A writer connection was asked to change the schema; that is the schema login's job."""


class WriterClient:
    """A data-write client that refuses schema statements before they reach the server.

    A guard against mistakes, not a boundary: the writer login's grants are. Reads, inserts,
    lightweight `DELETE FROM` and `INSERT ... SELECT` pass; every ALTER is refused.
    """

    SCHEMA_VERBS = frozenset(
        "CREATE ALTER DROP TRUNCATE RENAME EXCHANGE ATTACH DETACH UNDROP GRANT REVOKE".split()
    )
    _LEAD = re.compile(
        r"\s*(?:(?:--[^\n]*(?:\n|$)|#[^\n]*(?:\n|$)|/\*.*?\*/)\s*)*([A-Za-z]+)",
        re.DOTALL,
    )
    # Every client call that takes statement text as its first argument.
    _GUARDED = frozenset(
        "command query raw_query raw_stream query_df query_np query_arrow "
        "query_column_block_stream query_row_block_stream query_rows_stream query_df_stream "
        "query_np_stream query_arrow_stream query_df_arrow query_df_arrow_stream".split()
    )

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def check(cls, sql) -> None:
        if sql is None:
            return
        if not isinstance(sql, str):
            raise SchemaStatementOnWriter("statement is not text; refused on a writer")
        m = cls._LEAD.match(sql)
        if m and m.group(1).upper() in cls.SCHEMA_VERBS:
            raise SchemaStatementOnWriter(
                f"{m.group(1).upper()} on a writer connection; use "
                "ClickHouse.connect(role='schema') (CLICKHOUSE_SCHEMA_USER/PASSWORD)"
            )

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name not in self._GUARDED or not callable(attr):
            return attr

        def guarded(*args, **kwargs):
            self.check(args[0] if args else kwargs.get("query", kwargs.get("cmd")))
            return attr(*args, **kwargs)

        return guarded


class CrossHostStatement(RuntimeError):
    """One statement names tables on both the legacy and the v2 host; neither can run it."""


_SQL_TOKEN = re.compile(
    r"""--[^\n]*|\#[^\n]*|/\*.*?\*/|'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|"")*"|`[^`]*`|"""
    r"""[A-Za-z_][\w$]*|\d+|\s+|.""",
    re.DOTALL,
)
_DB_KEYWORDS = frozenset("FROM USE DATABASE JOIN INTO TABLE EXISTS IN".split())


def _sql_tokens(sql: str) -> list[tuple[str, str]]:
    """(kind, text) with comments and blanks dropped, quoted identifiers unquoted."""
    out = []
    for m in _SQL_TOKEN.finditer(sql):
        t = m.group(0)
        if t.isspace() or t.startswith(("--", "#", "/*")):
            continue
        if t[0] == "'":
            out.append(("str", t[1:-1]))
        elif t[0] in '"`':
            out.append(("word", t[1:-1]))
        elif t[0].isalpha() or t[0] == "_":
            out.append(("word", t))
        else:
            out.append(("sym", t))
    return out


class RoutedClient:
    """The legacy-host client plus a lazily opened v2-host client, chosen per statement.

    A statement goes to the v2 host when, ignoring comments and string literals, it names a v2
    database: `db.table` (any quoting or case), a bare `FROM/USE/DATABASE db`, or a string that is
    exactly a database name compared with a database column (`WHERE database = 'db'`). One that
    also names a table on the legacy host raises CrossHostStatement. Everything else, unqualified
    and system statements included, stays on the legacy host.
    """

    _ROUTED_BY_SQL = WriterClient._GUARDED - {"command", "query"}
    _ROUTED_BY_TABLE = frozenset({"insert_df", "insert_arrow", "raw_insert"})
    _ENDS_FROM = frozenset(
        "WHERE GROUP ORDER LIMIT HAVING SETTINGS UNION FORMAT PREWHERE WINDOW QUALIFY".split()
    )
    _DB_COLUMNS = frozenset({"database", "table_schema", "db", "db_name"})

    def __init__(self, legacy, v2_factory, v2_databases):
        self.legacy, self._factory, self._v2 = legacy, v2_factory, None
        self.v2_databases = tuple(v2_databases)
        self._names = {d.lower() for d in self.v2_databases}
        self._settings: dict = {}
        self._lock = threading.Lock()

    @property
    def v2(self):
        with self._lock:
            if self._v2 is None:
                v2 = self._factory()
                for key, value in self._settings.items():
                    v2.set_client_setting(key, value)
                self._v2 = v2
        return self._v2

    def set_client_setting(self, key, value):
        self._settings[key] = value
        self.legacy.set_client_setting(key, value)
        if self._v2 is not None:
            self._v2.set_client_setting(key, value)

    def _is_db_literal(self, toks, i) -> bool:
        j = i - 1
        while j >= 0 and toks[j] == ("sym", "("):
            j -= 1
        if j >= 0 and (
            toks[j] == ("sym", "=")
            or (toks[j][0] == "word" and toks[j][1].upper() == "IN")
        ):
            j -= 1
            return (
                j >= 0
                and toks[j][0] == "word"
                and toks[j][1].lower() in self._DB_COLUMNS
            )
        return False

    def _hits(self, sql: str) -> tuple[bool, bool]:
        """(names a v2 database, names a qualified table elsewhere) for one statement."""
        toks = _sql_tokens(sql)
        first = toks[0][1].upper() if toks and toks[0][0] == "word" else ""
        v2 = other = False
        in_from: dict[int, bool] = {}  # paren depth -> inside that depth's FROM list
        depth = 0

        def word(k):
            ok = 0 <= k < len(toks) and toks[k][0] == "word"
            return toks[k][1].upper() if ok else ""

        for i, (kind, text) in enumerate(toks):
            nxt = toks[i + 1 : i + 3]
            qualified = len(nxt) == 2 and nxt[0] == ("sym", ".") and nxt[1][0] == "word"
            prev, prev2 = word(i - 1), word(i - 2)
            if kind == "sym" and text == "(":
                depth += 1
                in_from[depth] = False
            elif kind == "sym" and text == ")":
                in_from.pop(depth, None)
                depth = max(0, depth - 1)
            if kind == "word":
                up = text.upper()
                if up == "FROM":
                    in_from[depth] = True
                elif up == "SELECT" or up in self._ENDS_FROM:
                    in_from[depth] = False
            if kind == "str":
                if text.lower() in self._names and self._is_db_literal(toks, i):
                    v2 = True
            elif kind == "word" and text.lower() in self._names:
                if qualified or prev in _DB_KEYWORDS:
                    v2 = True
            elif kind == "word" and qualified and text.lower() != "system":
                after_join = prev == "JOIN" and prev2 != "ARRAY"
                listed = in_from.get(depth, False) and toks[i - 1] == ("sym", ",")
                create_as = first == "CREATE" and prev == "AS"
                if (
                    prev in ("INTO", "TABLE", "FROM")
                    or after_join
                    or listed
                    or create_as
                ):
                    other = True
        return v2, other

    def route(self, table=None, database=None, sql=None, parameters=None):
        """The backend client for one statement."""
        if isinstance(database, str) and database.lower() in self._names:
            return self.v2
        if isinstance(table, str) and self._hits(f"INTO {table}")[0]:
            return self.v2
        if isinstance(sql, str):
            v2, other = self._hits(sql)
            if v2 and other:
                raise CrossHostStatement(
                    "statement names tables on both hosts; copy through the client instead"
                )
            if v2:
                return self.v2
        if isinstance(parameters, dict) and any(
            isinstance(v, str)
            and v.lower() in self._names
            and (k in self._DB_COLUMNS or k.endswith("_database"))
            for k, v in parameters.items()
        ):
            return self.v2
        return self.legacy

    def insert(self, table, *args, **kwargs):
        backend = self.route(table=table, database=kwargs.get("database"))
        return backend.insert(table, *args, **kwargs)

    def command(self, cmd, *args, **kwargs):
        params = args[0] if args else kwargs.get("parameters")
        return self.route(sql=cmd, parameters=params).command(cmd, *args, **kwargs)

    def query(self, query=None, *args, **kwargs):
        params = args[0] if args else kwargs.get("parameters")
        return self.route(sql=query, parameters=params).query(query, *args, **kwargs)

    def close(self):
        try:
            self.legacy.close()
        finally:
            if self._v2 is not None:
                self._v2.close()

    def __getattr__(self, name):
        if name in self._ROUTED_BY_SQL:

            def by_sql(*args, **kwargs):
                sql = args[0] if args else kwargs.get("query")
                backend = self.route(sql=sql, parameters=kwargs.get("parameters"))
                return getattr(backend, name)(*args, **kwargs)

            return by_sql
        if name in self._ROUTED_BY_TABLE:

            def by_table(*args, **kwargs):
                table = args[0] if args else kwargs.get("table")
                backend = self.route(table=table, database=kwargs.get("database"))
                return getattr(backend, name)(*args, **kwargs)

            return by_table
        return getattr(self.legacy, name)


class ClickHouse:
    """The one connection factory for every ingest, plus the v2 write gate."""

    ENV = ClickHouseEnv
    WRITER, SCHEMA = "writer", "schema"
    _warned_schema_fallback = False

    @classmethod
    def _pair(cls, user_var: str, pass_var: str) -> tuple[str, str]:
        user, password = cls.ENV.get(user_var), cls.ENV.get(pass_var)
        if bool(user) != bool(password):
            missing = pass_var if user else user_var
            raise SystemExit(
                f"{missing} is unset or empty -- set both {user_var} and {pass_var}, or neither"
            )
        return user, password

    @classmethod
    def credentials(cls, role: str, v2: bool = False) -> tuple[str, str]:
        """(user, password) for `role` on the legacy or the v2 host, most specific pair first.

        schema: CLICKHOUSE_V2_SCHEMA_* (v2 only), then CLICKHOUSE_SCHEMA_*; v2: CLICKHOUSE_V2_*;
        then the legacy writer pair. Half a pair is an error; the schema role falling back to a
        writer login warns once.
        """
        tiers = []
        if role == cls.SCHEMA:
            if v2:
                tiers.append(
                    ("CLICKHOUSE_V2_SCHEMA_USER", "CLICKHOUSE_V2_SCHEMA_PASSWORD")
                )
            tiers.append(("CLICKHOUSE_SCHEMA_USER", "CLICKHOUSE_SCHEMA_PASSWORD"))
        if v2:
            tiers.append(("CLICKHOUSE_V2_USER", "CLICKHOUSE_V2_PASSWORD"))
        for user_var, pass_var in tiers:
            user, password = cls._pair(user_var, pass_var)
            if user:
                if role == cls.SCHEMA and "SCHEMA" not in user_var:
                    cls._warn_schema_fallback()
                return user, password
        if role == cls.SCHEMA:
            cls._warn_schema_fallback()
        return (
            cls.ENV.get("CLICKHOUSE_USER", cls.ENV.DEFAULT_USER),
            cls.ENV.get("CLICKHOUSE_PASS"),
        )

    @classmethod
    def _warn_schema_fallback(cls) -> None:
        if not cls._warned_schema_fallback:
            cls._warned_schema_fallback = True
            print(
                "[warn] CLICKHOUSE_SCHEMA_USER/PASSWORD unset: schema statements use the "
                "writer login",
                file=sys.stderr,
            )

    @classmethod
    def _open(cls, role: str, v2: bool, database: str, verify: bool):
        host = cls.ENV.v2_host() if v2 else cls.ENV.host()
        if not host:
            raise SystemExit(
                "CLICKHOUSE_HOST is unset or empty -- check the secrets mapping"
            )
        secure = cls.ENV.v2_secure() if v2 else cls.ENV.secure()
        user, password = cls.credentials(role, v2)
        if not password and secure:
            raise SystemExit(
                ("CLICKHOUSE_V2_PASSWORD" if v2 else "CLICKHOUSE_PASS")
                + " is unset or empty -- check the secrets mapping"
            )
        port_raw = cls.ENV.v2_port() if v2 else cls.ENV.port()
        try:
            port = int(port_raw)
        except ValueError:
            raise SystemExit(f"CLICKHOUSE_PORT is not a number: {port_raw!r}") from None
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            secure=secure,
            verify=verify,
            # No session: a timed-out request keeps its session locked server-side, so every
            # retry or probe on the same client would fail with SESSION_IS_LOCKED.
            autogenerate_session_id=False,
            connect_timeout=cls.ENV.seconds("CLICKHOUSE_CONNECT_TIMEOUT", 30),
            send_receive_timeout=cls.ENV.seconds(
                "CLICKHOUSE_SEND_RECEIVE_TIMEOUT", 300
            ),
        )
        return WriterClient(client) if role == cls.WRITER else client

    @classmethod
    def connect(cls, *, verify: bool = True, database: str = "", role: str = WRITER):
        """Connect with the resolved settings; `verify=False` for an unverified cert.

        `role='writer'` (default) refuses schema statements; `role='schema'` is for the applier,
        ensure-table steps and sandbox administration. With CLICKHOUSE_V2_HOST set, a connection
        to a v2 database goes to that host and any other one is a RoutedClient.
        """
        if role not in (cls.WRITER, cls.SCHEMA):
            raise ValueError(f"unknown role {role!r}")
        database = database or cls.ENV.database()
        if not cls.ENV.v2_host():
            wants_v2 = cls.ENV.target_database() or database in cls.ENV.v2_databases()
            if wants_v2 and cls.ENV.flag("CH_REQUIRE_V2_HOST"):
                raise SystemExit(
                    "CH_REQUIRE_V2_HOST is set and a v2 database is configured, but "
                    "CLICKHOUSE_V2_HOST is unset or empty -- the write would go to the legacy host"
                )
            return cls._open(role, False, database, verify)
        if database in cls.ENV.v2_databases():
            return cls._open(role, True, database, verify)
        return RoutedClient(
            cls._open(role, False, database, verify),
            lambda: cls._open(role, True, cls.ENV.v2_default_database(), verify),
            cls.ENV.v2_databases(),
        )

    @staticmethod
    def tables_present(
        client, db: str, tables=None, check_columns: bool = True
    ) -> bool:
        """The v2 write gate: every table (default: functional pair) has its columns."""
        return all(
            t.present(client, db, check_columns=check_columns)
            for t in (tables or (schema.TestCases, schema.TestCaseRuns))
        )


# Function API, kept so installed consumers import one definition, not a copy.
_env = ClickHouseEnv.get
get_client = ClickHouse.connect
client_summary = ClickHouseEnv.summary
target_database = ClickHouseEnv.target_database
tables_present = ClickHouse.tables_present
