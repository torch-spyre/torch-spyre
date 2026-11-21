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

"""ClickHouse connection factory and the v2 table-presence gate."""

import os
import regex as re
import sys
import threading

import clickhouse_connect

from . import schema


class ClickHouseEnv:
    """The connection settings, read from the environment through one resolver."""

    DEFAULT_PORT = "443"
    DEFAULT_DB = "spyre"

    @staticmethod
    def get(name: str, default: str = "") -> str:
        """An env var, treating BLANK as absent -- GHA exports an unset secret as ''."""
        return (os.environ.get(name) or "").strip() or default

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

    V2_DATABASES = ("spyre_v2", "vigil_ops", "spyre_kb", "optimize_ops")

    @classmethod
    def v2_host(cls) -> str:
        """CLICKHOUSE_V2_HOST: the endpoint for the Replicated databases; '' = one host only."""
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
        """CH_V2_DATABASES: the databases served by the v2 host (comma list)."""
        listed = (
            tuple(x.strip() for x in cls.get("CH_V2_DATABASES").split(",") if x.strip())
            or cls.V2_DATABASES
        )
        extra = (
            cls.target_database()
        )  # a renamed or dev v2 database is v2 by definition
        return listed + ((extra,) if extra and extra not in listed else ())

    @classmethod
    def v2_default_database(cls) -> str:
        """The v2 client's own default database; never the v1 one."""
        return cls.target_database() or cls.v2_databases()[0]

    @classmethod
    def summary(cls) -> str:
        """A host:port/database string, resolved exactly as the connection is."""
        return f"{cls.host()}:{cls.port()}/{cls.database()}"


class SchemaStatementOnWriter(RuntimeError):
    """A writer connection was asked to change the schema; that is the schema identity's job."""


class WriterClient:
    """A client for data writes that refuses statements which change schema, grants or state.

    This is a guard against mistakes, not a security boundary: the writer login's server-side
    grants are the boundary. Inserts, lightweight `DELETE FROM`, `INSERT ... SELECT` and reads
    pass; `ALTER TABLE ... DELETE/UPDATE` is refused with every other ALTER.
    """

    SCHEMA_VERBS = frozenset(
        "CREATE ALTER DROP TRUNCATE RENAME EXCHANGE ATTACH DETACH GRANT REVOKE "
        "OPTIMIZE SYSTEM KILL UNDROP SET REPLACE BACKUP RESTORE MOVE".split()
    )
    _LEAD = re.compile(
        r"\s*(?:(?:--[^\n]*\n|#[^\n]*\n|/\*.*?\*/)\s*)*([A-Za-z]+)", re.DOTALL
    )
    # Every call that takes statement text as its first argument.
    _GUARDED = (
        "command query raw_query raw_stream query_df query_np query_arrow query_column_block_stream "
        "query_row_block_stream query_rows_stream query_df_stream query_np_stream "
        "query_arrow_stream query_df_arrow query_df_arrow_stream"
    ).split()

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
    """One statement names tables on both the legacy and the v2 host; it cannot run on either."""


_TOKEN = re.compile(
    r"""--[^\n]*|\#[^\n]*|/\*.*?\*/|'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|"")*"|`[^`]*`|"""
    r"""[A-Za-z_][\w$]*|\d+|\s+|.""",
    re.DOTALL,
)
_DB_KEYWORDS = frozenset("FROM USE DATABASE JOIN INTO TABLE EXISTS IN".split())


def _tokens(sql: str) -> list[tuple[str, str]]:
    """(kind, text) with comments and blanks dropped, quoted identifiers unquoted."""
    out = []
    for m in _TOKEN.finditer(sql):
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

    A statement goes to the v2 host when, ignoring comments and string literals, it names a
    Replicated database: `db.table` (any quoting or case), a bare `FROM/USE/DATABASE db`, or a
    string literal that is exactly a database name (`WHERE database = 'db'`). A statement that
    reads from one host's tables and writes the other's raises CrossHostStatement. Everything
    else, including unqualified and system statements, stays on the legacy host.
    """

    _FANOUT = ("close",)
    _ROUTED_BY_SQL = (
        "raw_query raw_stream query_df query_np query_arrow query_column_block_stream "
        "query_row_block_stream query_rows_stream query_df_stream query_np_stream "
        "query_arrow_stream query_df_arrow query_df_arrow_stream"
    ).split()
    _ROUTED_BY_TABLE = ("insert_df", "insert_arrow", "raw_insert")

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

    _ENDS_FROM = frozenset(
        "WHERE GROUP ORDER LIMIT HAVING SETTINGS UNION FORMAT PREWHERE WINDOW QUALIFY".split()
    )
    _DB_COLUMNS = frozenset({"database", "table_schema", "db", "db_name"})

    def _is_db_literal(self, toks, i) -> bool:
        """A string that is exactly a database name, compared with a database column."""
        j = i - 1
        while j >= 0 and toks[j] == ("sym", "("):
            j -= 1
        if j < 0:
            return False
        if toks[j] == ("sym", "=") or (
            toks[j][0] == "word" and toks[j][1].upper() == "IN"
        ):
            j -= 1
            return (
                j >= 0
                and toks[j][0] == "word"
                and toks[j][1].lower() in self._DB_COLUMNS
            )
        return False

    def _hits(self, sql: str) -> tuple[bool, bool]:
        """(names a v2 database, names a table elsewhere) for one statement."""
        toks = _tokens(sql)
        first = toks[0][1].upper() if toks and toks[0][0] == "word" else ""
        v2 = other = False
        in_from: dict[int, bool] = {}  # paren depth -> inside that depth's FROM list
        depth = 0

        def word(k):
            return (
                toks[k][1].upper()
                if 0 <= k < len(toks) and toks[k][0] == "word"
                else ""
            )

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
        return self.route(table=table, database=kwargs.get("database")).insert(
            table, *args, **kwargs
        )

    def command(self, cmd, *args, **kwargs):
        params = args[0] if args else kwargs.get("parameters")
        return self.route(sql=cmd, parameters=params).command(cmd, *args, **kwargs)

    def query(self, query=None, *args, **kwargs):
        params = args[0] if args else kwargs.get("parameters")
        return self.route(sql=query, parameters=params).query(query, *args, **kwargs)

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
        if name in self._FANOUT:

            def both(*args, **kwargs):
                try:
                    return getattr(self.legacy, name)(*args, **kwargs)
                finally:
                    if self._v2 is not None:
                        getattr(self._v2, name)(*args, **kwargs)

            return both
        return getattr(self.legacy, name)


class ClickHouse:
    """The one connection factory for every ingest, plus the v2 write gate."""

    WRITER, SCHEMA = "writer", "schema"
    _warned_schema_fallback = False

    ENV = ClickHouseEnv

    @classmethod
    def _pair(cls, user_var: str, pass_vars: tuple) -> tuple[str, str]:
        user = cls.ENV.get(user_var)
        password = next((cls.ENV.get(v) for v in pass_vars if cls.ENV.get(v)), "")
        if bool(user) != bool(password):
            missing = pass_vars[0] if user else user_var
            sys.exit(
                f"{missing} is unset or empty -- set both {user_var} and {pass_vars[0]}, or neither"
            )
        return user, password

    @classmethod
    def credentials(cls, role: str, v2: bool = False) -> tuple[str, str, str, str]:
        """(user, password, user_var, password_var) for `role` on the legacy or the v2 host.

        Most specific pair first: CLICKHOUSE_V2_SCHEMA_* then CLICKHOUSE_SCHEMA_* for the schema
        role; CLICKHOUSE_V2_USER/PASSWORD for v2 writes; the writer pair last. A pair with only
        one half set is an error. The schema role falling back to the writer pair warns once.
        """
        tiers = []
        if role == cls.SCHEMA:
            if v2:
                tiers.append(
                    ("CLICKHOUSE_V2_SCHEMA_USER", ("CLICKHOUSE_V2_SCHEMA_PASSWORD",))
                )
            tiers.append(("CLICKHOUSE_SCHEMA_USER", ("CLICKHOUSE_SCHEMA_PASSWORD",)))
        if v2:  # the HA login precedes the legacy writer pair, for either role
            tiers.append(("CLICKHOUSE_V2_USER", ("CLICKHOUSE_V2_PASSWORD",)))
        for user_var, pass_vars in tiers:
            user, password = cls._pair(user_var, pass_vars)
            if user:
                return user, password, user_var, pass_vars[0]
        if role == cls.SCHEMA and not cls._warned_schema_fallback:
            cls._warned_schema_fallback = True
            print(
                "[warn] CLICKHOUSE_SCHEMA_USER/PASSWORD unset: schema statements use the "
                "writer login",
                file=sys.stderr,
            )
        user, password = (
            cls.ENV.get("CLICKHOUSE_USER"),
            cls.ENV.get("CLICKHOUSE_PASS") or cls.ENV.get("CLICKHOUSE_PASSWORD"),
        )
        return (
            user,
            password,
            "CLICKHOUSE_USER",
            "CLICKHOUSE_PASS (or CLICKHOUSE_PASSWORD)",
        )

    @classmethod
    def _open(cls, role, v2, database, verify):
        host = cls.ENV.v2_host() if v2 else cls.ENV.host()
        if not host:
            sys.exit("CLICKHOUSE_HOST is unset or empty -- check the secrets mapping")
        secure = cls.ENV.v2_secure() if v2 else cls.ENV.secure()
        user, password, user_var, password_var = cls.credentials(role, v2)
        if secure and not user:
            sys.exit(f"{user_var} is unset or empty -- check the secrets mapping")
        if not password and secure:
            sys.exit(f"{password_var} is unset or empty -- check the secrets mapping")
        port_raw = cls.ENV.v2_port() if v2 else cls.ENV.port()
        try:
            port = int(port_raw)
        except ValueError:
            sys.exit(f"CLICKHOUSE_PORT is not a number: {port_raw!r}")
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            **({"user": user} if user else {}),
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
        client = WriterClient(client) if role == cls.WRITER else client
        try:
            client.ch_host = host
        except AttributeError:  # a stand-in client that takes no attributes
            pass
        return client

    @classmethod
    def connect(cls, *, verify: bool = True, database: str = "", role: str = WRITER):
        """Connect with the resolved settings; `verify=False` for an unverified cert.

        `role='writer'` (default) returns a client that refuses schema statements;
        `role='schema'` is for the applier, ensure-table and sandbox administration.

        With CLICKHOUSE_V2_HOST set, a connection to one of the Replicated databases
        (CH_V2_DATABASES) goes to that host, and any other connection returns a RoutedClient:
        the legacy host by default, the v2 host for statements that name a Replicated database.
        """
        if role not in (cls.WRITER, cls.SCHEMA):
            raise ValueError(f"unknown role {role!r}")
        if not cls.ENV.v2_host():
            require = cls.ENV.get("CH_REQUIRE_V2_HOST").lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            if require and (
                cls.ENV.target_database() or database in cls.ENV.v2_databases()
            ):
                sys.exit(
                    "CH_REQUIRE_V2_HOST is set and a Replicated database is configured "
                    f"({cls.ENV.target_database() or database}), but CLICKHOUSE_V2_HOST is "
                    "unset or empty -- the write would go to the legacy host"
                )
            return cls._open(role, False, database or cls.ENV.database(), verify)
        if database in cls.ENV.v2_databases():
            return cls._open(role, True, database, verify)
        return RoutedClient(
            cls._open(role, False, database or cls.ENV.database(), verify),
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
