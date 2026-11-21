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

"""Refused settings, the writer guard's reach, spool tokens, exit status, and DDL on writers."""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import client as chclient
from guard_helpers import guarded as resilient_insert
from spyre_clickhouse_ingest.breaker import SettingRefused
from spyre_clickhouse_ingest.resilience import Policy, Spool, batch_token, classify

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def err(code, text=""):
    return DatabaseError(f"Code: {code}. {text}", code=code, name="X")


class Rec:
    def __init__(self, script=()):
        self.script, self.calls = list(script), []

    def insert(self, table, data, column_names=None, database=None, settings=None):
        self.calls.append(dict(settings or {}))
        if self.script:
            raise self.script.pop(0)


FAST = dict(sleep=lambda s: None, log=lambda m: None)


def run(c, **kw):
    return resilient_insert(
        c, "t", [[1]], ["a"], scope="s", replicated=True, feeds_views=True, **FAST, **kw
    )


# ── a constrained profile ───────────────────────────────────────────────────────


def test_a_user_in_readonly_mode_is_fatal():
    assert classify(err(164, "Cannot execute query in readonly mode")) == "fatal"


def test_a_refused_optional_setting_is_dropped_once_and_the_batch_lands():
    c = Rec(
        [
            err(
                452,
                "Setting deduplicate_blocks_in_dependent_materialized_views is constrained",
            )
        ]
    )
    out = run(c)
    assert out.stored == 1 and len(c.calls) == 2
    assert "deduplicate_blocks_in_dependent_materialized_views" in c.calls[0]
    assert "deduplicate_blocks_in_dependent_materialized_views" not in c.calls[1]
    assert "insert_deduplication_token" in c.calls[1]


def test_a_refused_dedup_token_is_fatal_not_dropped():
    c = Rec([err(452, "Setting insert_deduplication_token is constrained")])
    with pytest.raises(SettingRefused):
        run(c)
    assert len(c.calls) == 1


def test_an_unnamed_constraint_violation_is_fatal_and_does_not_burn_the_deadline():
    c = Rec([err(452, "Some setting is constrained")] * 3)
    with pytest.raises(SettingRefused):
        run(c)
    assert len(c.calls) == 1


def test_a_batch_that_may_be_stored_never_loses_its_view_dedup(tmp_path):
    view = "deduplicate_blocks_in_dependent_materialized_views"
    c = Rec([err(999, "timeout"), err(452, f"Setting {view} is constrained")] * 40)
    spool = Spool(tmp_path / "sp")
    out = run(c, spool=spool, policy=Policy(deadline_s=5, backoff_s=1))
    assert out.spooled == 1 and [i["held"] for i in spool.items()] == [True]


# ── tokens ──────────────────────────────────────────────────────────────────────


def test_spool_put_refuses_a_bad_token(tmp_path):
    with pytest.raises(ValueError):
        Spool(tmp_path / "sp").put("t", ["a"], [[1]], "../evil")


def test_drop_needs_the_full_token(tmp_path):
    s = Spool(tmp_path / "sp")
    tok = batch_token("t", "s", [[1]])
    s.put("t", ["a"], [[1]], tok)
    assert s.drop(tok.split("-", 1)[1]) == 0
    assert s.drop(tok[-12:]) == 0
    assert s.drop(tok) == 1


# ── the writer guard ────────────────────────────────────────────────────────────


class Inner:
    def __init__(self):
        self.seen = []

    def __getattr__(self, name):
        def call(sql=None, *a, **k):
            self.seen.append((name, sql))
            return name

        return call


@pytest.mark.parametrize(
    "method",
    ["command", "query", "raw_query", "raw_stream", "query_df", "query_np", "query_arrow",
     "query_column_block_stream", "query_row_block_stream", "query_df_stream"],
)  # fmt: skip
def test_every_statement_taking_call_is_guarded(method):
    inner = Inner()
    w = chclient.WriterClient(inner)
    with pytest.raises(chclient.SchemaStatementOnWriter):
        getattr(w, method)("DROP TABLE t")
    assert inner.seen == []
    assert getattr(w, method)("SELECT 1") == method


@pytest.mark.parametrize(
    "sql",
    ["OPTIMIZE TABLE t FINAL", "SYSTEM STOP MERGES", "KILL QUERY WHERE 1", "UNDROP TABLE t",
     "SET max_threads = 1", "ALTER TABLE t DELETE WHERE 1", "ALTER TABLE t UPDATE a = 1 WHERE 1"],
)  # fmt: skip
def test_state_changing_statements_are_refused(sql):
    with pytest.raises(chclient.SchemaStatementOnWriter):
        chclient.WriterClient(Inner()).command(sql)


def test_bytes_statements_are_refused():
    with pytest.raises(chclient.SchemaStatementOnWriter):
        chclient.WriterClient(Inner()).command(b"DROP TABLE t")


def test_lightweight_delete_and_insert_select_pass():
    w = chclient.WriterClient(Inner())
    assert w.command("DELETE FROM t WHERE a = 1") == "command"
    assert w.command("INSERT INTO t SELECT * FROM s") == "command"


# ── exit status ─────────────────────────────────────────────────────────────────

PARK = textwrap.dedent(
    """
    import sys
    from clickhouse_connect.driver.exceptions import DatabaseError
    from spyre_clickhouse_ingest import writes
    from spyre_clickhouse_ingest.batching import insert_chunks

    writes.install_exit_hook()

    class C:
        database = "d"
        def query(self, sql, parameters=None):
            class R: result_rows = [("ReplicatedMergeTree", 0)]
            return R()
        def insert(self, *a, **k):
            raise DatabaseError("Code: 285. x", code=285, name="X")

    try:
        insert_chunks(C(), "t", [[1]], ["a"])
    except Exception as e:
        if "{swallow}" == "no":
            raise
    {tail}
    """
)


def exit_status(tmp_path, swallow, tail):
    env = {
        "PATH": "/usr/bin:/bin",
        "CH_WRITE_SPOOL": str(tmp_path / "sp"),
        "CH_WRITE_DEADLINE_S": "0.001",
        "PYTHONPATH": str(ROOT),
    }
    code = PARK.format(swallow=swallow, tail=tail)
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT
    )
    return p.returncode, p.stderr


def test_a_swallowed_park_turns_a_clean_exit_into_75(tmp_path):
    rc, errtxt = exit_status(tmp_path, "yes", "sys.exit(0)")
    assert rc == 75 and "[PARKED]" in errtxt


def test_falling_off_the_end_after_a_park_is_75(tmp_path):
    assert exit_status(tmp_path, "yes", "pass")[0] == 75


def test_an_uncaught_park_is_75(tmp_path):
    assert exit_status(tmp_path, "no", "pass")[0] == 75


def test_a_worse_failure_keeps_its_status(tmp_path):
    assert exit_status(tmp_path, "yes", "sys.exit(1)")[0] == 1
    assert exit_status(tmp_path, "yes", "raise RuntimeError('boom')")[0] == 1


# ── no schema statement on a writer client anywhere in the repo's ingest code ────

VERBS = tuple(chclient.WriterClient.SCHEMA_VERBS)
METHODS = set(chclient.WriterClient._GUARDED)
# Modules whose whole purpose is schema administration; their connections are checked in scan().
SCHEMA_MODULES = {
    "apply_schema.py",
    "sandbox.py",
    "sandbox_server.py",
    "perf_fixture.py",
}


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
    names = set()
    for n in own_nodes(scope):
        if isinstance(n, ast.Assign) and has_schema_role(n.value):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return names


def check_tree(path, tree, schema_params):
    bad = []
    consts = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            text = static_text(n.value, consts)
            if text is not None:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = text
    for scope, fn in scopes(tree):
        ok = schema_names(scope)
        params = {a.arg for a in fn.args.args} if fn else set()
        for n in own_nodes(scope):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            if n.func.attr in METHODS and n.args:
                text = (static_text(n.args[0], consts) or "").lstrip().upper()
                head = text.split(None, 1)[0] if text else ""
                if head in VERBS:
                    recv = n.func.value
                    good = (
                        (isinstance(recv, ast.Name) and recv.id in ok)
                        or has_schema_role(recv)
                        or (
                            isinstance(recv, ast.Name)
                            and recv.id in params
                            and fn is not None
                            and fn.name in schema_params
                        )
                    )
                    if not good:
                        bad.append(f"{path}:{n.lineno} {head} on {ast.unparse(recv)}")
    return bad


def check_callers(path, tree, schema_params):
    """Every call to a function whose first parameter takes DDL must pass a schema-role client."""
    bad = []
    for scope, _ in scopes(tree):
        ok = schema_names(scope)
        for n in own_nodes(scope):
            if not isinstance(n, ast.Call):
                continue
            name = (
                n.func.attr
                if isinstance(n.func, ast.Attribute)
                else getattr(n.func, "id", "")
            )
            if name in schema_params and n.args:
                arg = n.args[0]
                if not (
                    (isinstance(arg, ast.Name) and arg.id in ok) or has_schema_role(arg)
                ):
                    bad.append(f"{path}:{n.lineno} {name}({ast.unparse(arg)})")
    return bad


def scan():
    roots = [
        REPO / ".github/scripts",
        REPO / "tests/scripts",
        ROOT / "spyre_clickhouse_ingest",
    ]
    schema_params = {"ensure_tables", "ensure_extra_columns"}
    bad = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            rel = str(path.relative_to(REPO))
            if path.name in SCHEMA_MODULES:
                unmarked = [
                    n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func) in ("ClickHouse.connect", "get_client")
                    and not has_schema_role(n)
                ]  # fmt: skip
                # sandbox `diff` reads with the caller's own login; nothing else connects as a writer
                if len(unmarked) > (1 if path.name == "sandbox.py" else 0):
                    bad.append(f"{rel}: writer connection in a schema module")
                continue
            bad += check_tree(rel, tree, schema_params)
            bad += check_callers(rel, tree, schema_params)
    return bad


def test_no_schema_statement_is_sent_on_a_client_not_built_with_the_schema_role():
    assert scan() == []


def test_the_scan_catches_the_shapes_the_first_version_missed():
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
            c.query_df("OPTIMIZE TABLE t")
            c.command(BLOCK)
            c.command("SYSTEM STOP MERGES")
            c.command("SET max_threads = 1")
            ok = get_client(role="schema")
            ok.command(SQL)
        """
    )
    found = check_tree("x.py", ast.parse(src), set())
    assert len(found) == 6, found
    assert not any(f.endswith(" on ok") for f in found)
