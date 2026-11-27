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
        """CLICKHOUSE_V2_SECURE (0/false/no/off in any case is off), else the legacy setting."""
        raw = cls.get("CLICKHOUSE_V2_SECURE").lower()
        return cls.secure() if not raw else raw not in ("0", "false", "no", "off")

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
    lightweight `DELETE FROM` and `INSERT ... SELECT` pass; every ALTER and REPLACE is refused.
    """

    SCHEMA_VERBS = frozenset(
        "CREATE ALTER DROP TRUNCATE RENAME EXCHANGE ATTACH DETACH UNDROP REPLACE RESTORE GRANT "
        "REVOKE".split()
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
        lead = _strip_comments(sql).lstrip().lstrip("\ufeff").lstrip()
        words = [w.upper() for w in re.findall(r"[A-Za-z]+", lead)[:2]]
        verb = (
            " ".join(words)
            if words == ["SYSTEM", "RESTORE"]
            else words[0]
            if words
            else ""
        )
        if lead[:1].isalpha() and verb in cls.SCHEMA_VERBS | {"SYSTEM RESTORE"}:
            raise SchemaStatementOnWriter(
                f"{verb} on a writer connection; use "
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


class V2Unavailable(RuntimeError):
    """The v2 client could not be opened when a statement first needed it."""


class CrossHostStatement(RuntimeError):
    """One statement names tables on both the legacy and the v2 host; neither can run it."""


_SQL_TOKEN = re.compile(
    r"""--[^\n]*|\#[^\n]*|/\*.*?\*/|'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|"")*"|`[^`]*`|"""
    r"""[A-Za-z_][\w$]*|\d+|\s+|.""",
    re.DOTALL,
)
# Words after which a bare v2 database name is the database itself (`SHOW TABLES FROM db`, `USE db`).
_DB_KEYWORDS = frozenset("FROM USE DATABASE INTO TABLE EXISTS IN".split())


def _strip_comments(sql: str) -> str:
    """`sql` with comments blanked (`/* */` nests in ClickHouse; `--`, `#`) and `$$..$$` heredocs
    turned into empty literals; quoted text is kept as it is."""
    out, i, n = [], 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("$$", i):
            end = sql.find("$$", i + 2)
            out.append("''")
            i = n if end < 0 else end + 2
        elif c in "'\"`":
            j = i + 1
            while j < n and sql[j] != c:
                j += 2 if sql[j] == "\\" else 1
            out.append(sql[i : j + 1])
            i = j + 1
        elif sql.startswith("--", i) or c == "#":
            end = sql.find("\n", i)
            out.append(" ")
            i = n if end < 0 else end
        elif sql.startswith("/*", i):
            depth, j = 0, i
            while j < n:
                if sql.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif sql.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                    if depth == 0:
                        break
                else:
                    j += 1
            out.append(" ")
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _sql_tokens(sql: str) -> list[tuple[str, str]]:
    """(kind, text) with comments and blanks dropped, quoted identifiers unquoted."""
    out = []
    for m in _SQL_TOKEN.finditer(_strip_comments(sql)):
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
    database: `db.table` (any quoting; names are case-sensitive, as in ClickHouse), a bare
    `FROM/USE/DATABASE db`, a table-function argument (`merge(db, ...)`), a string that is exactly
    a database name compared with a database column (`database = 'db'`, either side, `IN`,
    `LIKE`), or a database-like parameter (`{db:String}`, `{x:Identifier}`). One that also names a
    table on the legacy host, or reads an unqualified table, raises CrossHostStatement. Everything
    else, unqualified and system statements included, stays on the legacy host.
    """

    _ROUTED_BY_SQL = WriterClient._GUARDED - {"command", "query"}
    _ROUTED_BY_TABLE = frozenset({"insert_df", "insert_arrow", "raw_insert"})
    _ENDS_FROM = frozenset(
        "WHERE GROUP ORDER LIMIT HAVING SETTINGS UNION FORMAT PREWHERE WINDOW QUALIFY".split()
    )
    _DB_COLUMNS = frozenset({"database", "table_schema", "db", "db_name"})
    _COMPARE = frozenset({"IN", "LIKE", "ILIKE"})
    # Functions whose arguments use the keyword FROM: extract(DAY FROM ts), trim(BOTH ' ' FROM x).
    _FROM_FUNCTIONS = frozenset({"EXTRACT", "TRIM", "SUBSTRING", "POSITION", "OVERLAY"})
    _TABLE_FUNCTIONS = frozenset(
        "MERGE REMOTE REMOTESECURE CLUSTER CLUSTERALLREPLICAS".split()
    )

    def __init__(self, legacy, v2_factory, v2_databases):
        self.legacy, self._factory, self._v2 = legacy, v2_factory, None
        self.v2_databases = tuple(v2_databases)
        self._names = set(self.v2_databases)
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

    def _is_compare(self, tok) -> bool:
        return tok == ("sym", "=") or (
            tok[0] == "word" and tok[1].upper() in self._COMPARE
        )

    def _is_db_column(self, tok) -> bool:
        return tok[0] == "word" and tok[1].lower() in self._DB_COLUMNS

    def _is_db_literal(self, toks, i) -> bool:
        """A string compared with a database column, on either side."""
        j = i - 1
        while j >= 0 and toks[j] == ("sym", "("):
            j -= 1
        if j >= 1 and self._is_compare(toks[j]) and self._is_db_column(toks[j - 1]):
            return True
        after = toks[i + 1 : i + 3]
        return (
            len(after) == 2
            and self._is_compare(after[0])
            and self._is_db_column(after[1])
        )

    def _hits(self, sql: str) -> tuple[bool, bool, str]:
        """(names a v2 database, names a qualified table elsewhere, reads an unqualified
        table) for one statement."""
        toks = _sql_tokens(sql)
        first = toks[0][1].upper() if toks and toks[0][0] == "word" else ""
        v2 = other = False
        bare = ""
        not_clause: set[int] = set()  # FROM tokens that do not start a FROM list
        ctes = {
            toks[k][1]
            for k in range(len(toks) - 2)
            if toks[k][0] == "word"
            and toks[k + 1][0] == "word"
            and toks[k + 1][1].upper() == "AS"
            and toks[k + 2] == ("sym", "(")
        }
        in_from: dict[int, bool] = {}  # paren depth -> inside that depth's FROM list
        func: dict[int, str] = {}  # paren depth -> the function that opened it
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
                in_from[depth], func[depth] = False, prev
            elif kind == "sym" and text == ")":
                in_from.pop(depth, None)
                func.pop(depth, None)
                depth = max(0, depth - 1)
            table_arg = (
                func.get(depth, "") in self._TABLE_FUNCTIONS
                and i > 0
                and toks[i - 1] in (("sym", "("), ("sym", ","))
            )
            if kind == "word":
                up = text.upper()
                if up == "FROM" and (
                    prev in ("FILL", "DISTINCT")
                    or func.get(depth, "") in self._FROM_FUNCTIONS
                ):
                    not_clause.add(
                        i
                    )  # WITH FILL FROM, IS [NOT] DISTINCT FROM, extract(.. FROM ..)
                elif up == "FROM":
                    in_from[depth] = True
                elif up == "SELECT" or up in self._ENDS_FROM:
                    in_from[depth] = False
            after_join = prev == "JOIN" and prev2 != "ARRAY"
            listed = in_from.get(depth, False) and i > 0 and toks[i - 1] == ("sym", ",")
            if kind == "str":
                db_of = text.split(".", 1)[0]
                if text in self._names and (table_arg or self._is_db_literal(toks, i)):
                    v2 = True
                elif table_arg and "." in text:
                    v2, other = (
                        v2 or db_of in self._names,
                        other or db_of not in self._names,
                    )
            elif (
                kind == "word"
                and not qualified
                and first != "SHOW"
                and (
                    (prev == "FROM" and i - 1 not in not_clause) or after_join or listed
                )
                and text not in self._names
                and text not in ctes
                and nxt[:1] != [("sym", "(")]
                and text.upper() not in ("SELECT", "INFILE")
            ):
                bare = bare or text
            elif kind == "word" and text in self._names:
                if qualified or table_arg or prev in _DB_KEYWORDS:
                    v2 = True
            elif kind == "word" and qualified and text.lower() != "system":
                create_as = first == "CREATE" and prev == "AS"
                renamed = (
                    first in ("RENAME", "EXCHANGE") and prev in ("TO", "AND")
                ) or (first == "CREATE" and prev == "TO")
                if (
                    prev in ("INTO", "TABLE", "FROM", "IN", "VIEW", "DICTIONARY")
                    or after_join
                    or listed
                    or create_as
                    or renamed
                    or table_arg
                ):
                    other = True
        return v2, other, bare

    def route(self, table=None, database=None, sql=None, parameters=None):
        """The backend client for one statement."""
        if isinstance(database, str) and database in self._names:
            return self.v2
        if isinstance(table, str) and self._hits(f"INTO {table}")[0]:
            return self.v2
        v2 = other = False
        if isinstance(sql, str):
            v2, other, bare = self._hits(sql)
            if v2 and other:
                raise CrossHostStatement(
                    "statement names tables on both hosts; copy through the client instead"
                )
            if v2 and bare:
                raise CrossHostStatement(
                    f"statement names a v2 database and reads {bare!r} unqualified, which "
                    "resolves in the connection's own (legacy) database; qualify it as db.table"
                )
            if v2:
                return self.v2
        idents = (
            set(re.findall(r"\{(\w+):Identifier\}", sql))
            if isinstance(sql, str)
            else set()
        )
        if isinstance(parameters, dict) and any(
            isinstance(v, str)
            and v in self._names
            and (
                k in self._DB_COLUMNS or k.endswith(("_database", "db")) or k in idents
            )
            for k, v in parameters.items()
        ):
            if other:
                raise CrossHostStatement(
                    "a parameter names a v2 database in a statement on legacy tables"
                )
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
    def _params(cls, role: str, v2: bool, database: str) -> dict:
        """The connection arguments for one host; SystemExit names a bad setting."""
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
        port_var = (
            "CLICKHOUSE_V2_PORT"
            if v2 and cls.ENV.get("CLICKHOUSE_V2_PORT")
            else "CLICKHOUSE_PORT"
        )
        port_raw = cls.ENV.v2_port() if v2 else cls.ENV.port()
        try:
            port = int(port_raw)
        except ValueError:
            raise SystemExit(f"{port_var} is not a number: {port_raw!r}") from None
        return dict(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            secure=secure,
        )

    @classmethod
    def _open(cls, role: str, params: dict, verify: bool):
        client = clickhouse_connect.get_client(
            **params,
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
        to a v2 database goes to that host and any other one is a RoutedClient. Every v2 setting
        is checked here, before anything is written; the v2 client itself opens on first use.
        """
        if role not in (cls.WRITER, cls.SCHEMA):
            raise ValueError(f"unknown role {role!r}")
        database = database or cls.ENV.database()
        v2_dbs = cls.ENV.v2_databases()
        if not cls.ENV.v2_host():
            wants_v2 = cls.ENV.target_database() or database in v2_dbs
            if wants_v2 and cls.ENV.flag("CH_REQUIRE_V2_HOST"):
                raise SystemExit(
                    "CH_REQUIRE_V2_HOST is set and a v2 database is configured, but "
                    "CLICKHOUSE_V2_HOST is unset or empty -- the write would go to the legacy host"
                )
            return cls._open(role, cls._params(role, False, database), verify)
        v1 = cls.ENV.database()
        if v1 in v2_dbs and v1 != cls.ENV.target_database():
            raise SystemExit(
                f"CH_V2_DATABASES names CLICKHOUSE_DB ({v1}): every v1 statement would go to "
                "the v2 host"
            )
        if database in v2_dbs:
            return cls._open(role, cls._params(role, True, database), verify)
        v2_params = cls._params(role, True, cls.ENV.v2_default_database())

        def open_v2():
            try:
                return cls._open(role, v2_params, verify)
            except (
                SystemExit
            ) as stop:  # a SystemExit here would skip the caller's handlers
                raise V2Unavailable(str(stop)) from None

        return RoutedClient(
            cls._open(role, cls._params(role, False, database), verify), open_v2, v2_dbs
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
