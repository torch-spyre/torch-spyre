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

"""A reference written by the real writer on chdb: the views read it back, and the measurement readers do not."""

import json
import regex as re

import pytest
from spyre_clickhouse_ingest import reference
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests", "20-artifacts", "30-benchmarks", "49-result-uploads", "50-artifact-views",
    "60-benchmark-views", "86-bench-references",
)  # fmt: skip
UP1, UP2 = (
    "01a13151-2096-79bf-bd9c-311b8c8bc701",
    "01a13151-2096-79bf-bd9c-311b8c8bc702",
)
CASE = {
    "component": "synth-comp",
    "name": "decode/latency",
    "tags": [],
    "disc": {"model": "m1", "tp": "1"},
}
CASE["benchmark_id"] = reference.case_id(CASE)
KEY = "perf.synth.decode.refsys.latency"


def doc(value=11.0, **over):
    base = {
        "apiVersion": reference.API_VERSION, "kind": reference.KIND, "key": KEY, "origin": "generated", "class": "perf",
        "label": "ref-sys-a", "terms": "internal-generated", "tools": {"torch": "2.9.1"},
        "model": {"repo": "org/m1", "revision": "abc123"}, "system": {"backend": "cpu"}, "clean": False,
        "warnings": ["not_clean"], "pinning": "none",
        "comparability": {"harness": "synth@1", "harness_match": "yes", "precision": "fp32", "dataset_subset": "full",
                          "dataset_match": "same", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64},
        "values": [{"case": CASE, "backend": "cpu", "measurements": {"llm.itl.median": [value, value + 0.5]}, "iterations": 2}],
    }  # fmt: skip
    return {**base, **over}


class Rows:
    def __init__(self, rows):
        self.result_rows = rows


class Chdb:
    """The slice of the clickhouse_connect client the writers use, over a chdb session."""

    DB = "refdb"

    def __init__(self):
        self.s = session.Session()
        self.command(f"CREATE DATABASE {self.DB}")
        self.command(f"USE {self.DB}")
        objs = [
            o
            for n in FILES
            for o in SchemaApplier.objects(
                SCHEMA_DIR / f"{n}.sql", (SCHEMA_DIR / f"{n}.sql").read_text()
            )
        ]
        for kinds in (("table", "mv"), ("view",)):
            for o in objs:
                if o.kind in kinds:
                    self.command(o.sql)

    def command(self, sql, parameters=None):
        return str(self.s.query(self._bind(sql, parameters), "TSV")).rstrip("\n")

    @staticmethod
    def _lit(v):
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(Chdb._lit(x) for x in v) + "]"
        return (
            str(v)
            if isinstance(v, (int, float))
            else "'" + str(v).replace("'", "\\'") + "'"
        )

    def _bind(self, sql, parameters):
        return re.sub(
            r"\{(\w+):[^}]+\}", lambda m: self._lit((parameters or {})[m.group(1)]), sql
        )

    def query(self, sql, parameters=None, **kw):
        out = str(self.s.query(self._bind(sql, parameters), "JSONCompactEachRow"))
        return Rows([json.loads(x) for x in out.splitlines()])

    def rows(self, sql):
        return [
            json.loads(x) for x in str(self.s.query(sql, "JSONEachRow")).splitlines()
        ]

    def insert(self, table, rows, column_names=None, database=None, **kw):
        body = "\n".join(
            json.dumps(dict(zip(column_names, r)), default=str) for r in rows
        )
        self.command(
            f"INSERT INTO {database + '.' if database else ''}{table} FORMAT JSONEachRow\n{body}"
        )

    def event(self, d, upload, state, actor, ts, **kw):
        ident = reference.identity(d)
        ev = reference.upload_event(upload_id=upload, state=state, actor=actor, artifact_id=ident.artifact_id,
                                    run_id=reference.run_id_for(d, "amy", upload), doc=d, **kw)  # fmt: skip
        self.insert(
            "result_uploads",
            [[ts, *[ev[k] for k in ev]]],
            ["ts", *ev],
            database=self.DB,
        )


@pytest.fixture
def db():
    c = Chdb()
    yield c
    c.s.close()


def approve(
    c, d, upload, approver="bob", submitter="amy", ts="2026-11-01 00:00:00.000", **over
):
    ctx = reference.WriteContext(submitter=submitter, upload_id=upload, bundle_sha256="ab" * 32, register_missing=True,
                                 approval=reference.decide_approval(submitter=submitter, approver=approver, is_admin=False,
                                                                    approvers={approver}).props, **over)  # fmt: skip
    c.event(d, upload, "pending", submitter, ts)
    out = reference.write(c, Chdb.DB, d, ctx)
    if out["status"] == "ingested":
        c.event(d, upload, "published", approver, ts.replace("00:00:00", "00:01:00"))
    return out


def states(c):
    return [
        (r["version"], r["state"])
        for r in c.rows(
            "SELECT version, state FROM v_bench_references ORDER BY toInt32(version)"
        )
    ]


def test_an_approved_reference_reads_back_and_registers_its_case(db):
    out = approve(db, doc(), UP1)
    assert (out["version"], out["registered_now"]) == (1, [CASE["benchmark_id"]])
    assert states(db) == [("1", "current")]
    (row,) = db.rows("SELECT * FROM v_bench_references")
    assert (
        row["arch"],
        row["origin"],
        row["ref_class"],
        row["clean"],
        row["warnings"],
        row["pinning"],
    ) == ("noarch", "generated", "perf", "0", "not_clean", "none")
    assert (
        row["submitted_by"],
        row["approved_by"],
        row["approval_override"],
        row["approval_self"],
    ) == ("amy", "bob", 0, 0)
    (val,) = db.rows("SELECT * FROM v_bench_reference_values")
    assert (val["ref_key"], val["state"], val["backend"], val["benchmark_id"]) == (
        KEY,
        "current",
        "cpu",
        CASE["benchmark_id"],
    )
    assert (
        val["measurements"] == {"llm.itl.median": [11.0, 11.5]}
        and val["row_props"]["run.ref_only"] == "1"
    )
    assert db.rows("SELECT count() AS n FROM benchmarks")[0]["n"] == 1


def test_no_measurement_reader_sees_a_reference(db):
    approve(db, doc(), UP1)
    assert db.rows("SELECT count() AS n FROM benchmark_runs")[0]["n"] == 1
    for view in ("v_benchmark_results_enriched", "v_benchmark_run_artifacts"):
        assert db.rows(f"SELECT count() AS n FROM {view}")[0]["n"] == 0, view
    assert db.rows("SELECT count() AS n FROM v_tag_resolution")[0]["n"] == 0


def test_a_new_version_supersedes_and_rollback_and_retire_move_the_slot(db):
    approve(db, doc(11.0), UP1)
    out = approve(db, doc(12.0), UP2, ts="2026-11-02 00:00:00.000")
    assert out["version"] == 2 and states(db) == [("1", "superseded"), ("2", "current")]
    v1 = reference.identity(doc(11.0)).artifact_id
    reference.set_tag(db, Chdb.DB, reference.TAG_CURRENT, v1, KEY, "root", "rollback")
    assert states(db) == [("1", "current"), ("2", "superseded")]
    reference.set_tag(db, Chdb.DB, reference.TAG_RETIRED, v1, KEY, "root", "retire")
    assert states(db) == [("1", "retired"), ("2", "superseded")]


def test_a_re_send_of_the_same_upload_is_a_duplicate_and_another_upload_is_refused(db):
    approve(db, doc(), UP1)
    count = "SELECT (SELECT count() FROM artifacts) a, (SELECT count() FROM benchmark_runs) b, (SELECT count() FROM artifact_tags) t"
    before = db.rows(count)
    ctx = reference.WriteContext(
        submitter="amy",
        upload_id=UP1,
        bundle_sha256="cd" * 32,
        approval={"ref.approved_by": "bob"},
        register_missing=True,
    )
    again = reference.write(db, Chdb.DB, doc(label="other-label"), ctx)
    assert again["status"] == "duplicate" and again["version"] == 1
    assert db.rows(count) == before
    other = reference.WriteContext(
        submitter="amy",
        upload_id=UP2,
        bundle_sha256="cd" * 32,
        approval={"ref.approved_by": "bob"},
        register_missing=True,
    )
    with pytest.raises(reference.ReferenceError, match="already version 1"):
        reference.write(db, Chdb.DB, doc(), other)
    assert db.rows(count) == before


def test_retracting_the_current_version_leaves_the_previous_one_current_and_a_retire_survives_it(
    db,
):
    approve(db, doc(11.0), UP1)
    approve(db, doc(12.0), UP2, ts="2026-11-02 00:00:00.000")
    v1, v2 = (
        reference.identity(doc(11.0)).artifact_id,
        reference.identity(doc(12.0)).artifact_id,
    )
    db.event(doc(12.0), UP2, "retracted", "amy", "2026-11-03 00:00:00.000")
    assert states(db) == [
        ("1", "current")
    ]  # derived by the view: no write after a retract
    reference.set_tag(db, Chdb.DB, reference.TAG_RETIRED, v1, KEY, "root", "retire")
    assert states(db) == [("1", "retired")]
    db.event(doc(11.0), UP1, "retracted", "amy", "2026-11-04 00:00:00.000")
    assert states(db) == []
    assert v2 != v1


def test_a_duplicate_upload_withdrawn_never_moves_the_live_version(db):
    approve(db, doc(11.0), UP1)
    approve(db, doc(12.0), UP2, ts="2026-11-02 00:00:00.000")
    dup = "01a13151-2096-79bf-bd9c-311b8c8bc799"
    db.event(doc(12.0), dup, "pending", "cy", "2026-11-03 00:00:00.000")
    db.event(doc(12.0), dup, "retracted", "cy", "2026-11-03 00:01:00.000")
    assert states(db) == [("1", "superseded"), ("2", "current")]


def test_an_admin_self_approval_is_listed_by_the_view(db):
    d = doc()
    props = reference.decide_approval(
        submitter="amy", approver="amy", is_admin=True, approvers=set()
    ).props
    db.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    reference.write(
        db,
        Chdb.DB,
        d,
        reference.WriteContext(
            submitter="amy",
            upload_id=UP1,
            bundle_sha256="ab" * 32,
            approval=props,
            register_missing=True,
        ),
    )
    db.event(d, UP1, "published", "amy", "2026-11-01 00:01:00.000")
    (row,) = db.rows(
        "SELECT approval_override, approval_self, approved_by FROM v_bench_references"
    )
    assert (row["approval_override"], row["approval_self"], row["approved_by"]) == (
        1,
        1,
        "amy",
    )


def test_an_unregistered_case_is_refused_without_register_now(db):
    ctx = reference.WriteContext(
        submitter="amy",
        upload_id=UP1,
        bundle_sha256="ab" * 32,
        approval={"ref.approved_by": "bob"},
    )
    with pytest.raises(reference.ReferenceError, match="not registered"):
        reference.write(db, Chdb.DB, doc(), ctx)
    assert db.rows("SELECT count() AS n FROM artifacts")[0]["n"] == 0


def test_the_upload_formats_and_the_migration(db):
    ev = {"upload_id": UP1, "state": "pending", "actor": "a", "artifact_id": "2b397099-6200-52fb-98c4-b603961a0582",
          "run_id": "e4b1dc2a-985b-5295-9aaf-38cc48cfe00e", "result_kind": "performance", "test_type": "perf", "arch": "noarch",
          "filename": "r", "size_bytes": 1, "content_sha256": "c", "bundle_url": ""}  # fmt: skip
    for fmt in ("reference", "bench", "junit"):
        db.insert(
            "result_uploads",
            [list({**ev, "format": fmt}.values())],
            list({**ev, "format": fmt}),
            database=Chdb.DB,
        )
    with pytest.raises(Exception, match="chk_format|CHECK|Constraint"):
        db.insert(
            "result_uploads",
            [list({**ev, "format": "bogus"}.values())],
            list({**ev, "format": "bogus"}),
            database=Chdb.DB,
        )
    # Migration 026 turns the previous constraint into the new one on an existing table.
    old = (
        (SCHEMA_DIR / "49-result-uploads.sql")
        .read_text()
        .replace("'tables', 'reference', 'bench'", "'tables'")
    )
    c = Chdb.__new__(Chdb)
    c.s = db.s
    db.command("CREATE DATABASE old")
    db.command("USE old")
    for stmt in SchemaApplier.statements(old):
        if "CREATE TABLE" in stmt:
            db.command(stmt)
    with pytest.raises(Exception):
        db.command(
            "INSERT INTO old.result_uploads (upload_id, state, actor, artifact_id, run_id, format, result_kind, test_type, arch, filename, size_bytes, content_sha256, bundle_url) "
            "VALUES (generateUUIDv4(), 'pending', 'a', '2b397099-6200-52fb-98c4-b603961a0582', 'e4b1dc2a-985b-5295-9aaf-38cc48cfe00e', 'reference', 'performance', 'perf', 'noarch', 'f', 1, 'c', '')"
        )
    for stmt in SchemaApplier.statements(
        (
            SCHEMA_DIR / "migrations" / "026_result_uploads_bench_reference_formats.sql"
        ).read_text()
    ):
        db.command(stmt)
    db.command(
        "INSERT INTO old.result_uploads (upload_id, state, actor, artifact_id, run_id, format, result_kind, test_type, arch, filename, size_bytes, content_sha256, bundle_url) "
        "VALUES (generateUUIDv4(), 'pending', 'a', '2b397099-6200-52fb-98c4-b603961a0582', 'e4b1dc2a-985b-5295-9aaf-38cc48cfe00e', 'reference', 'performance', 'perf', 'noarch', 'f', 1, 'c', '')"
    )


def test_a_version_that_is_not_published_yet_never_holds_the_slot(db):
    approve(db, doc(11.0), UP1)
    d2 = doc(12.0)
    db.event(d2, UP2, "pending", "amy", "2026-11-02 00:00:00.000")
    ctx = reference.WriteContext(
        submitter="amy",
        upload_id=UP2,
        bundle_sha256="cd" * 32,
        approval={"ref.approved_by": "bob"},
    )
    reference.write(
        db, Chdb.DB, d2, ctx
    )  # written and tagged, but the upload is still pending
    assert states(db) == [("1", "current")]
    db.event(d2, UP2, "published", "bob", "2026-11-02 00:01:00.000")
    assert states(db) == [("1", "superseded"), ("2", "current")]


class Flaky(Chdb):
    """A client whose next insert(s) into a table raise, for failure injection."""

    def __init__(self):
        super().__init__()
        self.fail = {}
        self.skip = {}  # let this many inserts into a table through before the failure applies

    def insert(self, table, rows, column_names=None, database=None, **kw):
        if self.skip.get(table, 0) > 0:
            self.skip[table] -= 1
        elif self.fail.get(table, 0) > 0:
            self.fail[table] -= 1
            raise RuntimeError(f"boom {table}")
        super().insert(table, rows, column_names=column_names, database=database, **kw)


@pytest.fixture
def flaky():
    c = Flaky()
    yield c
    c.s.close()


def wctx(upload, url="", approver="bob", **kw):
    approval = reference.decide_approval(
        submitter="amy", approver=approver, is_admin=False, approvers={approver}
    ).props
    return reference.WriteContext(submitter="amy", upload_id=upload, bundle_sha256="ab" * 32, bundle_url=url,
                                  register_missing=True, approval=approval, **kw)  # fmt: skip


@pytest.mark.parametrize(
    "table",
    [
        "artifacts",
        "artifact_refs",
        "benchmark_runs",
        "artifact_tags",
        "artifact_results",
    ],
)
def test_a_failure_at_each_insert_resumes_with_the_first_address_and_approver(
    flaky, table
):
    d = doc()
    flaky.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.fail[table] = 1
    with pytest.raises(reference.ReferenceError) as err:
        reference.write(flaky, Chdb.DB, d, wctx(UP1, "https://x.invalid/one.tgz"))
    assert err.value.retry
    first = (
        "https://x.invalid/one.tgz"
        if table != "artifacts"
        else "https://x.invalid/two.tgz"
    )
    reference.write(
        flaky, Chdb.DB, d, wctx(UP1, "https://x.invalid/two.tgz", approver="carol")
    )
    one = lambda sql: flaky.rows(sql)  # noqa: E731
    assert (
        len(one("SELECT 1 FROM artifacts")) == 1
        and len(one("SELECT 1 FROM benchmark_runs")) == 1
    )
    assert len(one("SELECT 1 FROM artifact_results")) == 1
    assert [r["ref"] for r in one("SELECT ref FROM artifact_refs")] == [first]
    if table != "artifacts":
        assert one("SELECT props['ref.approved_by'] a FROM artifacts") == [{"a": "bob"}]
        assert {
            r["p"] for r in one("SELECT props['promoted_by'] p FROM artifact_tags")
        } == {"bob"}
    flaky.event(d, UP1, "published", "bob", "2026-11-01 00:01:00.000")
    assert states(flaky) == [("1", "current")]


def test_an_older_resumed_write_leaves_a_current_version_after_the_newer_one_is_retracted(
    flaky,
):
    da, dbb = doc(11.0), doc(12.0)
    flaky.event(da, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.fail["artifact_tags"] = 1
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, da, wctx(UP1))
    approve(flaky, dbb, UP2, ts="2026-11-02 00:00:00.000")
    assert states(flaky) == [("2", "current")]
    flaky.event(dbb, UP2, "retracted", "amy", "2026-11-03 00:00:00.000")
    out = reference.write(flaky, Chdb.DB, da, wctx(UP1))
    flaky.event(da, UP1, "published", "bob", "2026-11-04 00:00:00.000")
    assert out["status"] == "resumed" and states(flaky) == [("1", "current")]


# ---- randomised interleavings: the invariants hold after every step ----

import random  # noqa: E402

KEYS = (KEY, "perf.synth.decode.refsys.other")


def model_states(c, key):
    """The view's rule restated from the event log: the newest tag row among ref-retired rows and ref-current
    rows of artifacts whose upload is published decides; a version is current/retired when it is the holder."""
    pub = {
        r["artifact_id"]
        for r in c.rows(
            "SELECT artifact_id FROM v_result_uploads WHERE state = 'published' AND NOT is_round"
        )
    }
    tags = c.rows(
        f"SELECT tag, toString(artifact_id) a FROM artifact_tags WHERE tag_family = 'reference' AND props['ref.key'] = '{key}' ORDER BY ts, audit_timestamp"
    )
    live = [t for t in tags if t["tag"] == reference.TAG_RETIRED or t["a"] in pub]
    return live[-1] if live else None


def check_one_number_everywhere(c, trail):
    """The merged artifact version is the number on its verdict, its value rows and its tag rows."""
    for r in c.rows(
        "SELECT toString(artifact_id) a, props['ref.version'] v FROM v_artifacts WHERE component = 'reference' AND artifact_id IN (SELECT artifact_id FROM artifact_results)"
    ):
        seen = {
            "verdict": c.rows(f"SELECT groupUniqArray(props['ref.version']) x FROM artifact_results WHERE artifact_id = toUUID('{r['a']}')")[0]["x"],
            "values": c.rows(f"SELECT groupUniqArray(props['ref.version']) x FROM benchmark_runs WHERE props['run.ref_only'] = '1' AND run_id IN (SELECT run_id FROM artifact_results WHERE artifact_id = toUUID('{r['a']}'))")[0]["x"],
            "tags": [v for v in c.rows(f"SELECT groupUniqArray(props['ref.version']) x FROM artifact_tags WHERE artifact_id = toUUID('{r['a']}')")[0]["x"] if v],
        }  # fmt: skip
        for where, vs in seen.items():
            assert set(vs) <= {r["v"]}, ("number differs", where, vs, r["v"], trail)


def check_invariants(c, trail):
    rows = c.rows(
        "SELECT ref_key, version, state, toString(artifact_id) a FROM v_bench_references"
    )
    for key in KEYS:
        mine = [r for r in rows if r["ref_key"] == key]
        assert sum(r["state"] == "current" for r in mine) <= 1, ("two current", trail)
        holder = model_states(c, key)
        want = {
            r["a"]: (
                "current"
                if holder and holder["tag"] == "ref-current" and holder["a"] == r["a"]
                else "retired"
                if holder and holder["tag"] == "ref-retired" and holder["a"] == r["a"]
                else "superseded"
                if holder
                else "untagged"
            )
            for r in mine
        }
        assert {r["a"]: r["state"] for r in mine} == want, (
            "view differs from the rule",
            trail,
        )
    versions = c.rows(
        "SELECT k, v, count() n FROM (SELECT props['ref.key'] k, argMax(props['ref.version'], audit_timestamp) v "
        "FROM artifacts WHERE component = 'reference' AND artifact_id IN (SELECT artifact_id FROM artifact_results) "
        "GROUP BY artifact_id, k) GROUP BY k, v HAVING n > 1"
    )
    assert versions == [], ("a version number was reused", trail)
    check_one_number_everywhere(c, trail)
    for r in rows:
        done = c.rows(
            f"SELECT count() n FROM artifact_results WHERE artifact_id = toUUID('{r['a']}')"
        )[0]["n"]
        if done:
            assert (
                c.rows(
                    f"SELECT count() n FROM v_bench_reference_values WHERE artifact_id = toUUID('{r['a']}')"
                )[0]["n"]
                >= 1
            ), ("completed version without values", trail)


def run_seed(seed, steps=14):
    rnd = random.Random(seed)
    c = Flaky()
    clock = [0]

    def tick():
        clock[0] += 1
        return f"2026-11-01 00:{clock[0] // 60:02d}:{clock[0] % 60:02d}.000"

    orig = c.insert

    def insert(table, rows, column_names=None, database=None, **kw):
        if (
            table in ("artifact_tags", "artifacts")
            and column_names
            and "audit_timestamp" not in column_names
        ):
            column_names = [*column_names, "audit_timestamp"] + (
                ["ts"] if table == "artifact_tags" else []
            )
            rows = [
                [*r, tick()] + ([tick()[:19]] if table == "artifact_tags" else [])
                for r in rows
            ]
        return orig(table, rows, column_names=column_names, database=database, **kw)

    c.insert = insert
    uploads, trail, n_up = {}, [], [0]
    contents = [[doc(10.0 + i, key=KEYS[i % 2]) for i in range(4)]][0]
    try:
        for _ in range(steps):
            op = rnd.choice(
                [
                    "new",
                    "retry",
                    "publish",
                    "retract",
                    "dup",
                    "rollback",
                    "retire",
                    "new",
                    "publish",
                ]
            )
            trail.append(op)
            if op == "new":
                i = rnd.randrange(4)
                n_up[0] += 1
                up = f"01a13151-2096-79bf-bd9c-311b8c8b{n_up[0]:04d}"
                uploads[up] = {
                    "doc": contents[i],
                    "done": False,
                    "published": False,
                    "retracted": False,
                }
                c.event(contents[i], up, "pending", "amy", tick())
                if rnd.random() < 0.5:
                    c.fail[
                        rnd.choice(
                            [
                                "artifacts",
                                "artifact_refs",
                                "benchmark_runs",
                                "artifact_tags",
                                "artifact_results",
                            ]
                        )
                    ] = 1
                op = "retry"
            if op == "retry":
                todo = [
                    u
                    for u, v in uploads.items()
                    if not v["done"] and not v["retracted"]
                ]
                if todo:
                    u = rnd.choice(todo)
                    try:
                        out = reference.write(c, Chdb.DB, uploads[u]["doc"], wctx(u))
                        uploads[u]["done"] = out["status"] in (
                            "ingested",
                            "resumed",
                            "duplicate",
                        )
                    except reference.ReferenceError as err:
                        if not err.retry:
                            uploads[u]["retracted"] = True
                    c.fail.clear()
            elif op == "publish":
                todo = [
                    u
                    for u, v in uploads.items()
                    if v["done"] and not v["published"] and not v["retracted"]
                ]
                if todo:
                    u = rnd.choice(todo)
                    c.event(uploads[u]["doc"], u, "published", "bob", tick())
                    uploads[u]["published"] = True
            elif op == "retract":
                todo = [
                    u
                    for u, v in uploads.items()
                    if v["published"] and not v["retracted"]
                ]
                if todo:
                    u = rnd.choice(todo)
                    c.event(uploads[u]["doc"], u, "retracted", "amy", tick())
                    uploads[u]["retracted"] = True
            elif op == "dup":
                done = [v["doc"] for v in uploads.values() if v["done"]]
                if done:
                    n_up[0] += 1
                    up = f"01a13151-2096-79bf-bd9c-311b8c8b{n_up[0]:04d}"
                    c.event(rnd.choice(done), up, "pending", "cy", tick())
                    c.event(rnd.choice(done), up, "retracted", "cy", tick())
            elif op in ("rollback", "retire"):
                todo = [
                    u
                    for u, v in uploads.items()
                    if v["published"] and not v["retracted"]
                ]
                if todo:
                    u = rnd.choice(todo)
                    d = uploads[u]["doc"]
                    reference.set_tag(
                        c,
                        Chdb.DB,
                        reference.TAG_CURRENT
                        if op == "rollback"
                        else reference.TAG_RETIRED,
                        reference.identity(d).artifact_id,
                        d["key"],
                        "root",
                    )
            check_invariants(c, list(trail))
    finally:
        c.s.close()


@pytest.mark.parametrize("seed", range(30))
def test_random_interleavings_keep_the_invariants(seed):
    run_seed(seed)


def test_a_resumed_older_write_does_not_take_current_from_a_newer_published_version(
    flaky,
):
    da, dbb = doc(11.0), doc(12.0)
    flaky.event(da, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.fail["artifact_tags"] = 1
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, da, wctx(UP1))
    approve(flaky, dbb, UP2, ts="2026-11-02 00:00:00.000")
    out = reference.write(flaky, Chdb.DB, da, wctx(UP1))
    flaky.event(da, UP1, "published", "bob", "2026-11-03 00:00:00.000")
    assert out["current"] is False
    assert states(flaky) == [("1", "superseded"), ("2", "current")]


def test_promote_is_sticky_across_a_resume_on_chdb(flaky):
    d = doc()
    flaky.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.fail["artifact_results"] = 1
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, d, wctx(UP1, promote=False))
    out = reference.write(flaky, Chdb.DB, d, wctx(UP1, promote=True))
    flaky.event(d, UP1, "published", "bob", "2026-11-01 00:01:00.000")
    assert out["current"] is False and states(flaky) == [("1", "untagged")]


def claim(c, key, version):
    """A foreign claimant row: an artifact of the key that took this number long before."""
    c.command(
        "INSERT INTO refdb.artifacts (artifact_id, component, arch, kind, artifact_name, origin, identity_deps, context_deps, sources, props, audit_timestamp) "
        f"VALUES (generateUUIDv4(), 'reference', 'noarch', 'generic', 'x', 'built', [], [], [], map('ref.key', '{key}', 'ref.version', '{version}'), '2000-01-01 00:00:00.000')"
    )


def test_a_resumed_artifact_that_loses_its_number_is_renumbered_on_every_row(flaky):
    d = doc()
    flaky.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.fail["benchmark_runs"] = 1
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, d, wctx(UP1))
    claim(flaky, KEY, 1)  # a concurrent approval got number 1 first
    flaky.fail["artifact_results"] = (
        1  # and the renumbered attempt fails again before its verdict
    )
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, d, wctx(UP1))
    out = reference.write(
        flaky, Chdb.DB, d, wctx(UP1)
    )  # the third attempt reads the merged number
    flaky.event(d, UP1, "published", "bob", "2026-11-01 00:01:00.000")
    assert out["version"] == 2
    assert flaky.rows(
        "SELECT props['ref.version'] v FROM v_artifacts WHERE artifact_id = toUUID('%s')"
        % reference.identity(d).artifact_id
    ) == [{"v": "2"}]
    check_one_number_everywhere(flaky, ["resume"])
    assert states(flaky) == [("2", "current")]
    assert len(flaky.rows("SELECT 1 FROM v_bench_reference_values")) == 1


def test_a_loser_after_inserting_is_renumbered_on_every_row(flaky):
    d = doc()
    flaky.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    real = flaky.insert

    def insert(table, rows, column_names=None, database=None, **kw):
        real(table, rows, column_names=column_names, database=database, **kw)
        if table == "artifacts" and not getattr(flaky, "raced", False):
            flaky.raced = True
            claim(
                flaky, KEY, 1
            )  # taken by someone else between our check and our insert, earlier

    flaky.insert = insert
    out = reference.write(flaky, Chdb.DB, d, wctx(UP1))
    flaky.event(d, UP1, "published", "bob", "2026-11-01 00:01:00.000")
    assert out["version"] == 2
    check_one_number_everywhere(flaky, ["post"])
    assert states(flaky) == [("2", "current")]


def second_component(d):
    """The same reference with a second case in another component, so the rows land in two inserts."""
    case = {
        "component": "synth-comp-b",
        "name": "decode/latency-b",
        "tags": [],
        "disc": {"model": "m1"},
    }
    case["benchmark_id"] = reference.case_id(case)
    return {
        **d,
        "values": [
            *d["values"],
            {
                "case": case,
                "backend": "cpu",
                "measurements": {"m": [1.0]},
                "iterations": 1,
            },
        ],
    }


def test_a_resumed_artifact_with_rows_already_written_is_not_renumbered(flaky):
    d = second_component(doc())
    flaky.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.skip["benchmark_runs"], flaky.fail["benchmark_runs"] = (
        1,
        1,
    )  # component one lands, component two fails
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, d, wctx(UP1))
    claim(flaky, KEY, 1)
    with pytest.raises(reference.ReferenceError, match="already wrote rows") as err:
        reference.write(flaky, Chdb.DB, d, wctx(UP1))
    assert not err.value.retry
    assert flaky.rows(
        "SELECT props['ref.version'] v FROM v_artifacts WHERE artifact_id = toUUID('%s')"
        % reference.identity(d).artifact_id
    ) == [{"v": "1"}]


def test_a_renumbered_resume_does_not_take_current_from_a_newer_published_version(
    flaky,
):
    da, dbb = doc(11.0), doc(12.0)
    flaky.event(da, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    flaky.fail["benchmark_runs"] = 1
    with pytest.raises(reference.ReferenceError):
        reference.write(
            flaky, Chdb.DB, da, wctx(UP1)
        )  # A holds number 1 and has written no rows
    approve(flaky, dbb, UP2, ts="2026-11-02 00:00:00.000")  # B is number 2 and current
    claim(flaky, KEY, 1)  # and someone else took 1 before A resumes: A becomes 3
    out = reference.write(flaky, Chdb.DB, da, wctx(UP1))
    flaky.event(da, UP1, "published", "bob", "2026-11-03 00:00:00.000")
    assert out["version"] == 3 and out["current"] is False
    assert states(flaky) == [("2", "current"), ("3", "superseded")]


def test_a_renumber_row_is_later_than_the_artifact_even_when_the_clock_is_behind(flaky):
    d = doc()
    flaky.event(d, UP1, "pending", "amy", "2026-11-01 00:00:00.000")
    real = flaky.insert

    def insert(table, rows, column_names=None, database=None, **kw):
        if table == "artifacts" and "audit_timestamp" not in (column_names or []):
            column_names = [*column_names, "audit_timestamp"]
            rows = [
                [*r, "2099-01-01 00:00:00.000"] for r in rows
            ]  # the first row is stamped far ahead of the clock
        real(table, rows, column_names=column_names, database=database, **kw)

    flaky.insert = insert
    flaky.fail["benchmark_runs"] = 1
    with pytest.raises(reference.ReferenceError):
        reference.write(flaky, Chdb.DB, d, wctx(UP1))
    claim(flaky, KEY, 1)
    out = reference.write(flaky, Chdb.DB, d, wctx(UP1))
    assert out["version"] == 2
    assert flaky.rows(
        "SELECT props['ref.version'] v FROM v_artifacts WHERE artifact_id = toUUID('%s')"
        % reference.identity(d).artifact_id
    ) == [{"v": "2"}]
