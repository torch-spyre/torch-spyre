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

"""Writer and schema logins: the writer refuses DDL, and no ingest sends DDL on a writer."""

import ast
import textwrap
from pathlib import Path

import clickhouse_connect
import pytest

from spyre_clickhouse_ingest import client
from spyre_clickhouse_ingest.hw_schema import HwFailureDiagnostics

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_get_client(**kwargs):
        seen.update(kwargs)
        return Recorder()

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    for name in (
        "CLICKHOUSE_PORT", "CLICKHOUSE_DB", "CLICKHOUSE_DB_V2", "CLICKHOUSE_V2_HOST",
        "CLICKHOUSE_SCHEMA_USER", "CLICKHOUSE_SCHEMA_PASSWORD", "CH_REQUIRE_V2_HOST",
    ):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.example.com")
    monkeypatch.setenv("CLICKHOUSE_USER", "writer")
    monkeypatch.setenv("CLICKHOUSE_PASS", "secret")
    monkeypatch.setattr(client.ClickHouse, "_warned_schema_fallback", False)
    return seen


class Recorder:
    database = "db"

    def __init__(self):
        self.seen = []

    def command(self, cmd, *a, **k):
        self.seen.append(cmd)
        return 1

    def query(self, q=None, *a, **k):
        self.seen.append(q)
        return "rows"

    def insert(self, *a, **k):
        self.seen.append("insert")

    def __getattr__(self, name):
        def call(sql=None, *a, **k):
            self.seen.append(sql)
            return name

        return call


# ── roles ───────────────────────────────────────────────────────────────────────


def test_schema_role_uses_its_own_login(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_USER", "ddl")
    monkeypatch.setenv("CLICKHOUSE_SCHEMA_PASSWORD", "ddl-pw")
    client.ClickHouse.connect(role="schema")
    assert (captured["user"], captured["password"]) == ("ddl", "ddl-pw")
    client.ClickHouse.connect()
    assert (captured["user"], captured["password"]) == ("writer", "secret")


def test_schema_role_falls_back_to_the_writer_once_with_a_warning(captured, capsys):
    client.ClickHouse.connect(role="schema")
    client.ClickHouse.connect(role="schema")
    assert (captured["user"], captured["password"]) == ("writer", "secret")
    assert capsys.readouterr().err.count("CLICKHOUSE_SCHEMA_USER") == 1


def test_the_writer_role_never_warns(captured, capsys):
    client.ClickHouse.connect()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "present,missing",
    [
        ("CLICKHOUSE_SCHEMA_USER", "CLICKHOUSE_SCHEMA_PASSWORD"),
        ("CLICKHOUSE_SCHEMA_PASSWORD", "CLICKHOUSE_SCHEMA_USER"),
    ],
)
def test_half_a_schema_pair_names_the_missing_var(
    captured, monkeypatch, present, missing
):
    monkeypatch.setenv(present, "x")
    with pytest.raises(SystemExit, match=missing):
        client.ClickHouse.connect(role="schema")


def test_an_unknown_role_is_refused(captured):
    with pytest.raises(ValueError):
        client.ClickHouse.connect(role="admin")


def test_connect_hands_out_a_guarded_client_by_default(captured):
    assert isinstance(client.ClickHouse.connect(), client.WriterClient)
    assert not isinstance(client.ClickHouse.connect(role="schema"), client.WriterClient)


# ── the writer guard ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t (a Int8) ENGINE = Memory",
        "  alter table t add column b Int8",
        "ALTER TABLE t DELETE WHERE 1",
        "DROP VIEW IF EXISTS v",
        "TRUNCATE TABLE t",
        "RENAME TABLE a TO b",
        "-- note\nDROP TABLE t",
        "# note\nDROP TABLE t",
        "/* c */ CREATE OR REPLACE VIEW v AS SELECT 1",
        "GRANT SELECT ON db.* TO u",
        "REPLACE TABLE t AS SELECT 1",
        "\ufeffCREATE TABLE t (a Int8)",
        "#! note\nCREATE TABLE t (a Int8)",
        "/* /* */ SELECT */ DROP TABLE t",
        "\xa0DROP TABLE t",
        "\x0bDROP TABLE t",
        "\x0cDROP TABLE t",
        "\x85DROP TABLE t",
        "\u2003DROP TABLE t",
        "\u3000DROP TABLE t",
        "\ufeff\u2003/* c */ DROP TABLE t",
        "RESTORE TABLE t FROM Disk('b', 'f')",
        "SYSTEM RESTORE REPLICA t",
    ],
)
def test_a_writer_connection_never_sends_schema_statements(sql):
    inner = Recorder()
    w = client.WriterClient(inner)
    for call in (w.command, w.query, w.raw_query, w.query_df):
        with pytest.raises(client.SchemaStatementOnWriter):
            call(sql)
    with pytest.raises(client.SchemaStatementOnWriter):
        w.command(cmd=sql)
    assert inner.seen == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "EXISTS TABLE t",
        "DESCRIBE TABLE t",
        "DELETE FROM t WHERE a = 1",
        "INSERT INTO t SELECT * FROM s",
        "SELECT 'CREATE TABLE' AS x",
    ],
)
def test_a_writer_connection_passes_data_statements(sql):
    inner = Recorder()
    w = client.WriterClient(inner)
    w.command(sql)
    assert w.query(sql) == "rows"
    w.insert("t", [], column_names=[])
    assert w.database == "db"
    assert inner.seen == [sql, sql, "insert"]


def test_statements_that_are_not_text_are_refused():
    with pytest.raises(client.SchemaStatementOnWriter):
        client.WriterClient(Recorder()).command(b"DROP TABLE t")


def test_ensure_extra_columns_surfaces_a_writer_refusal():
    with pytest.raises(client.SchemaStatementOnWriter):
        HwFailureDiagnostics.ensure_extra_columns(client.WriterClient(Recorder()))


# ── no schema statement on a writer client anywhere in the repo's ingest code ────

VERBS = client.WriterClient.SCHEMA_VERBS
METHODS = client.WriterClient._GUARDED
# Functions whose first parameter receives DDL; their callers must hand them a schema client.
SCHEMA_PARAMS = {"ensure_tables", "ensure_extra_columns"}
# Modules whose purpose is schema administration: every connection they open is checked.
SCHEMA_MODULES = {
    "apply_schema.py",
    "sandbox.py",
    "sandbox_server.py",
    "perf_fixture.py",
}
# sandbox `diff` reads the sandbox with its own login.
WRITER_CONNECTS_ALLOWED = {"sandbox.py": 1}


def static_text(node, consts):
    """The leading text of a string expression: literals, f-strings, +, .format(), dedent."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return static_text(node.values[0], consts) or "" if node.values else ""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return static_text(node.left, consts)
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("format", "strip", "lstrip"):
            return static_text(f.value, consts)
        if node.args and ast.unparse(f).endswith("dedent"):
            return static_text(node.args[0], consts)
    return None


def has_schema_role(call):
    return isinstance(call, ast.Call) and any(
        k.arg == "role"
        and isinstance(k.value, ast.Constant)
        and k.value.value == "schema"
        for k in call.keywords
    )


def scopes(tree):
    yield tree, None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n, n


def own_nodes(scope):
    """Nodes of one scope, not descending into nested functions."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(n))


def schema_names(scope):
    return {
        t.id
        for n in own_nodes(scope)
        if isinstance(n, ast.Assign) and has_schema_role(n.value)
        for t in n.targets
        if isinstance(t, ast.Name)
    }


def check_tree(path, tree):
    bad, consts = [], {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            text = static_text(n.value, consts)
            if text is not None:
                consts.update(
                    {t.id: text for t in n.targets if isinstance(t, ast.Name)}
                )
    for scope, fn in scopes(tree):
        ok = schema_names(scope)
        params = {a.arg for a in fn.args.args} if fn else set()
        for n in own_nodes(scope):
            if not isinstance(n, ast.Call):
                continue
            name = (
                n.func.attr
                if isinstance(n.func, ast.Attribute)
                else getattr(n.func, "id", "")
            )
            if name in SCHEMA_PARAMS and n.args:
                arg = n.args[0]
                if not (
                    (isinstance(arg, ast.Name) and arg.id in ok) or has_schema_role(arg)
                ):
                    bad.append(f"{path}:{n.lineno} {name}({ast.unparse(arg)})")
            if not (isinstance(n.func, ast.Attribute) and name in METHODS and n.args):
                continue
            text = (static_text(n.args[0], consts) or "").lstrip().upper()
            head = text.split(None, 1)[0] if text else ""
            if head not in VERBS:
                continue
            recv = n.func.value
            good = (
                (isinstance(recv, ast.Name) and recv.id in ok)
                or has_schema_role(recv)
                or (
                    isinstance(recv, ast.Name)
                    and recv.id in params
                    and fn.name in SCHEMA_PARAMS
                )
            )
            if not good:
                bad.append(f"{path}:{n.lineno} {head} on {ast.unparse(recv)}")
    return bad


def scan():
    bad = []
    roots = (
        REPO / ".github/scripts",
        REPO / "tests/scripts",
        ROOT / "spyre_clickhouse_ingest",
    )
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            rel = str(path.relative_to(REPO))
            if path.name not in SCHEMA_MODULES:
                bad += check_tree(rel, tree)
                continue
            writers = [
                n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) in ("ClickHouse.connect", "get_client")
                and not has_schema_role(n)
            ]  # fmt: skip
            if len(writers) > WRITER_CONNECTS_ALLOWED.get(path.name, 0):
                bad.append(f"{rel}: writer connection in a schema module")
    return bad


def test_no_schema_statement_is_sent_on_a_client_not_built_with_the_schema_role():
    assert scan() == []


def test_the_scan_catches_the_shapes_it_claims_to():
    src = textwrap.dedent(
        """
        from textwrap import dedent
        SQL = "ALTER TABLE t " + "ADD COLUMN c Int8"
        FMT = "CREATE TABLE {t} (a Int8)"
        BLOCK = dedent('''
            GRANT SELECT ON db.* TO u
        ''')
        def f(get_client):
            schema_client = get_client()
            schema_client.command(SQL)
            c = get_client()
            c.raw_query(FMT.format(t="x"))
            c.query_df("TRUNCATE TABLE t")
            c.command(BLOCK)
            c.command(f"DROP TABLE {x}")
            ensure_tables(c)
            ok = get_client(role="schema")
            ok.command(SQL)
            ensure_tables(ok)
            ensure_extra_columns(get_client(role="schema"))
        """
    )
    found = check_tree("x.py", ast.parse(src))
    assert len(found) == 6, found
    assert not any(" on ok" in f or "(ok)" in f for f in found)
