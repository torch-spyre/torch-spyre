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

"""UploadLog against the real result_uploads DDL on chdb: the relay's rows through the views the
dashboard's retract and the published-results views read. Skipped without chdb and the ingest package."""

import json
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import relay
from test_relay import AID, NOW, REPORT, RID, ROOT, FakeArtifactory, _args, _dir

session = pytest.importorskip("chdb.session")
apply_schema = pytest.importorskip("spyre_clickhouse_ingest.apply_schema")
CI_RUN = "5eae88e4-fb62-5686-a23c-51a7ff0d1da3"


def literal(typ, value):
    if typ == "UInt64":
        return str(int(value))
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


class ChdbClient:
    """The two clickhouse_connect calls UploadLog makes, on a chdb session (server-side parameters inlined)."""

    def __init__(self, s):
        self.s = s

    def bind(self, sql, parameters):
        for k, v in (parameters or {}).items():
            for typ in ("UUID", "String", "UInt64"):
                if f"{{{k}:{typ}}}" in sql:
                    sql = sql.replace(f"{{{k}:{typ}}}", literal(typ, v))
        return sql

    def query(self, sql, parameters=None):
        out = self.s.query(
            self.bind(sql, parameters)
            + " SETTINGS output_format_json_quote_64bit_integers = 0",
            "JSONCompact",
        )
        return SimpleNamespace(result_rows=json.loads(out.data())["data"])

    def command(self, sql, parameters=None, settings=None):
        self.s.query(self.bind(sql, parameters))


@pytest.fixture
def s():
    sess = session.Session()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "49-result-uploads.sql",
    ):
        for stmt in apply_schema.SchemaApplier.statements(
            (apply_schema.SCHEMA_DIR / name).read_text()
        ):
            sess.query(stmt)
    for rid in (RID, CI_RUN):  # the verdicts the ingest leaves; one is a CI run
        sess.query(
            "INSERT INTO artifact_results (artifact_id, run_id, result_kind, test_type, state, arch, props) "
            f"VALUES ('{AID}', '{rid}', 'functional', 'fvt', 'passed', 's390x', map('bundle_kind', 'junit'))"
        )
    yield sess
    sess.cleanup()


def run_relay(s, files, tmp_path, outcome=(0, REPORT)):
    art = FakeArtifactory(files)
    uploads = relay.UploadLog(ChdbClient(s), "default")
    entries = relay.Relay(
        art, _args(), ingest=lambda local, argv: outcome, now=NOW, uploads=uploads
    ).run(tmp_path)
    return art, entries


def upload_rows(s):
    return json.loads(
        "["
        + ",".join(
            s.query("SELECT * FROM result_uploads ORDER BY ts", "JSONEachRow")
            .data()
            .split()
        )
        + "]"
    )


def published(s):
    return sorted(
        json.loads(
            s.query(
                "SELECT groupArray(toString(run_id)) FROM v_artifact_results_published",
                "JSONCompact",
            ).data()
        )["data"][0][0]
    )


def retract(s, upload_id, who, reason):
    """The dashboard's route: read the upload's latest state, append `retracted` carrying its linkage."""
    carried = ("artifact_id", "run_id", "format", "result_kind", "test_type", "arch", "filename", "size_bytes",
               "content_sha256", "bundle_url")  # fmt: skip
    latest = json.loads(s.query(f"SELECT {', '.join(carried)} FROM v_result_uploads WHERE upload_id = '{upload_id}'",
                                "JSONEachRow").data())  # fmt: skip
    row = {
        **latest,
        "upload_id": upload_id,
        "actor": who,
        "state": "retracted",
        "reason": reason,
    }
    s.query("INSERT INTO result_uploads FORMAT JSONEachRow\n" + json.dumps(row))


def test_a_relay_upload_is_retractable_through_the_dashboards_view(s, tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    _, entries = run_relay(s, _dir(folder), tmp_path / "1")
    assert entries[0]["upload"] == "recorded" and published(s) == sorted([RID, CI_RUN])
    (row,) = upload_rows(s)
    got = json.loads(s.query("SELECT upload_id, state, uploaded_by, content_sha256, run_id, format "
                             "FROM v_result_uploads", "JSONEachRow").data())  # fmt: skip
    assert got == {"upload_id": relay.upload_id_of(RID), "state": "published", "uploaded_by": "jdoe",
                   "content_sha256": REPORT["bundle_sha256"], "run_id": RID, "format": "junit"}  # fmt: skip
    assert row["props"]["channel"] == "relay" and row["result_kind"] == "functional"

    retract(s, relay.upload_id_of(RID), "admin@example.com", "wrong image")
    state = json.loads(
        s.query(
            "SELECT state, actor, uploaded_by, reason FROM v_result_uploads",
            "JSONEachRow",
        ).data()
    )
    assert state == {
        "state": "retracted",
        "actor": "admin@example.com",
        "uploaded_by": "jdoe",
        "reason": "wrong image",
    }
    assert published(s) == [CI_RUN]


def test_a_repeat_or_duplicate_writes_no_row_even_after_a_retract(s, tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    run_relay(s, _dir(folder), tmp_path / "1")
    run_relay(s, _dir(folder), tmp_path / "2", (0, {**REPORT, "status": "duplicate"}))
    assert len(upload_rows(s)) == 1
    retract(s, relay.upload_id_of(RID), "jdoe", "oops")
    run_relay(s, _dir(folder), tmp_path / "3", (0, {**REPORT, "status": "duplicate"}))
    assert [r["state"] for r in upload_rows(s)] == [
        "published",
        "retracted",
    ] and published(s) == [CI_RUN]


def test_a_rejected_bundle_writes_no_row(s, tmp_path):
    _, entries = run_relay(
        s,
        _dir(f"{ROOT}/inbox/{AID}/b-fvt"),
        tmp_path,
        (2, {"status": "rejected", "reason": "x"}),
    )
    assert entries[0]["outcome"] == "rejected" and upload_rows(s) == []


def test_the_table_accepts_every_format_the_verdict_can_name(s, tmp_path):
    """chk_format: a bundle_kind outside it fails the write, so the bundle waits (see the README)."""
    for n, kind in enumerate(("vllm", "sendnn", "benchmarks", "tables")):
        rid = f"00000000-0000-5000-8000-00000000000{n + 1}"
        s.query(
            "INSERT INTO artifact_results (artifact_id, run_id, result_kind, test_type, state, arch, props) "
            f"VALUES ('{AID}', '{rid}', 'performance', 'fvt', 'passed', 's390x', map('bundle_kind', '{kind}'))"
        )
        run_relay(
            s,
            _dir(f"{ROOT}/inbox/{AID}/b-{kind}"),
            tmp_path / kind,
            (0, {**REPORT, "run_id": rid}),
        )
    assert [(r["format"], r["result_kind"]) for r in upload_rows(s)] == [
        (k, "performance") for k in ("vllm", "sendnn", "benchmarks", "tables")
    ]


def test_a_stale_has_cannot_publish_a_retracted_run_again(s, tmp_path, monkeypatch):
    """has() answers no (the check ran before a retract landed); the insert checks again, in one statement."""
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    run_relay(s, _dir(folder), tmp_path / "1")
    retract(s, relay.upload_id_of(RID), "admin@example.com", "wrong image")
    monkeypatch.setattr(relay.UploadLog, "has", lambda self, run_id: False)
    _, entries = run_relay(
        s, _dir(folder), tmp_path / "2", (0, {**REPORT, "status": "duplicate"})
    )
    assert [r["state"] for r in upload_rows(s)] == [
        "published",
        "retracted",
    ] and published(s) == [CI_RUN]
    assert entries[0]["outcome"] == "duplicate"


def test_the_newest_verdict_names_the_format_and_result_kind(s):
    for ts, kind, result in (
        ("2020-01-01 00:00:00", "junit", "functional"),
        ("2099-01-02 00:00:00", "vllm", "performance"),
    ):
        s.query(
            "INSERT INTO artifact_results (ts, artifact_id, run_id, result_kind, test_type, state, arch, props) "
            f"VALUES ('{ts}', '{AID}', '{CI_RUN}', '{result}', 'fvt', 'passed', 's390x', map('bundle_kind', '{kind}'))"
        )
    got = relay.UploadLog(ChdbClient(s), "default").kinds(AID, CI_RUN, "fvt")
    assert got == ("performance", "vllm")
