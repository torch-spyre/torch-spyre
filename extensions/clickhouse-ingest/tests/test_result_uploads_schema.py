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

"""result_uploads' vocabularies are artifact_results' and the bundle schema's, not its own."""

import json

import pytest
import regex as re
from spyre_clickhouse_ingest import offline
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier


def _check(file: str, name: str) -> list:
    sql = (SCHEMA_DIR / file).read_text()
    m = re.search(rf"CONSTRAINT\s+{name}\s+CHECK\s+\w+\s+IN\s*\(([^)]*)\)", sql)
    assert m, f"{name} not in {file}"
    return re.findall(r"'([^']+)'", m.group(1))


def test_result_kind_and_test_type_match_artifact_results():
    for name in ("chk_result_kind", "chk_test_type"):
        assert _check("49-result-uploads.sql", name) == _check("20-artifacts.sql", name)


def test_format_matches_the_bundle_kinds():
    kinds = offline.schema()["properties"]["kind"]["enum"]
    assert _check("49-result-uploads.sql", "chk_format") == kinds


NIL = "00000000-0000-0000-0000-000000000000"
AID, UID = (
    "2b397099-6200-52fb-98c4-b603961a0582",
    "01a13151-2096-79bf-bd9c-311b8c8bc729",
)
RUN, CI_RUN = (
    "e4b1dc2a-985b-5295-9aaf-38cc48cfe00e",
    "5eae88e4-fb62-5686-a23c-51a7ff0d1da3",
)


def _upload(**over):
    row = {"upload_id": UID, "state": "published", "actor": "alice@ibm.com", "artifact_id": AID, "run_id": RUN,
           "format": "junit", "result_kind": "functional", "test_type": "unit", "arch": "x86_64",
           "filename": "a.xml", "size_bytes": 10, "content_sha256": "c", "bundle_url": "u"}  # fmt: skip
    return {**row, **over}


def test_rows_must_be_linked_and_the_view_names_the_uploader():
    session = pytest.importorskip("chdb.session")

    s = session.Session()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "49-result-uploads.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)

    def insert(row):
        s.query("INSERT INTO result_uploads FORMAT JSONEachRow\n" + json.dumps(row))

    for bad in ({"artifact_id": NIL}, {"run_id": NIL}, {"format": "xml"}, {"state": "rejected"},
                {"test_type": "nightly"}, {"result_kind": "tests"}):  # fmt: skip
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            insert(_upload(**bad))
    insert(_upload())
    insert(_upload(state="retracted", actor="admin@ibm.com", reason="wrong image"))
    got = s.query(
        "SELECT state, actor, uploaded_by, reason FROM v_result_uploads", "JSONEachRow"
    ).data()
    assert json.loads(got) == {"state": "retracted", "actor": "admin@ibm.com",
                               "uploaded_by": "alice@ibm.com", "reason": "wrong image"}  # fmt: skip
    for rid in (RUN, CI_RUN):
        s.query(
            "INSERT INTO artifact_results (artifact_id, run_id, result_kind, test_type, state, arch) "
            f"VALUES ('{AID}', '{rid}', 'functional', 'unit', 'passed', 'x86_64')"
        )
    left = s.query(
        "SELECT toString(run_id) FROM v_artifact_results_published", "CSV"
    ).data()
    assert left.split() == [f'"{CI_RUN}"']


def test_a_retracted_upload_leaves_the_benchmark_coverage_and_case_views():
    session = pytest.importorskip("chdb.session")

    s = session.Session()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "30-benchmarks.sql",
        "49-result-uploads.sql",
        "50-artifact-views.sql",
        "51-functional-views.sql",
        "52-cross-views.sql",
        "60-benchmark-views.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)
    for state in ("published", "retracted"):
        s.query(
            "INSERT INTO result_uploads FORMAT JSONEachRow\n"
            + json.dumps(_upload(state=state))
        )
    bench, case = (
        "8f6a2f53-0d5b-5d1e-9a51-6c3a0f3c0b11",
        "c0de0000-0000-5000-8000-000000000001",
    )
    s.query(
        f"INSERT INTO benchmarks (benchmark_id, component, name) VALUES ('{bench}', 'torch-spyre', 'matmul')"
    )
    s.query(
        f"INSERT INTO test_cases (test_case_id, component, classname, name) VALUES ('{case}', 'torch-spyre', 'T', 't')"
    )
    for rid in (RUN, CI_RUN):
        s.query(
            "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements) "
            f"VALUES ('{rid}', '{bench}', 'torch-spyre', 'sendnn', map('kernel_ms', [1.0]))"
        )
        s.query(
            "INSERT INTO test_case_runs (run_id, test_case_id, component, status) "
            f"VALUES ('{rid}', '{case}', 'torch-spyre', 'passed')"
        )

    def one(sql):
        return s.query(sql, "CSV").data().split()

    assert one(
        "SELECT DISTINCT toString(run_id) FROM v_benchmark_results_enriched"
    ) == [f'"{CI_RUN}"']
    assert one("SELECT sum(runs), sum(runs_missing_result) FROM v_run_coverage") == [
        "1,1"
    ]
    assert one("SELECT sum(runs) FROM v_case_trend") == ["1"]


def test_a_retracted_upload_leaves_the_capability_views():
    session = pytest.importorskip("chdb.session")

    s = session.Session()
    for name in (
        "20-artifacts.sql",
        "46-capabilities.sql",
        "49-result-uploads.sql",
        "61-capability-views.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)
    upload = _upload(format="junit", result_kind="capability", test_type="model_ops")
    for state in ("published", "retracted"):
        s.query(
            "INSERT INTO result_uploads FORMAT JSONEachRow\n"
            + json.dumps({**upload, "state": state})
        )
    cap = "ca9a0000-0000-5000-8000-000000000001"
    s.query(
        "INSERT INTO capabilities (capability_id, component, test_type, subject, name) "
        f"VALUES ('{cap}', 'torch-spyre', 'model_ops', 'some-model', 'torch.add')"
    )
    for rid in (RUN, CI_RUN):
        s.query(
            "INSERT INTO capability_runs (run_id, capability_id, component, test_type, arch, status, backend) "
            f"VALUES ('{rid}', '{cap}', 'torch-spyre', 'model_ops', 'x86_64', 'passed', 'spyre')"
        )
    got = (
        s.query("SELECT DISTINCT toString(run_id) FROM v_capability_results", "CSV")
        .data()
        .split()
    )
    assert got == [f'"{CI_RUN}"']
    got = (
        s.query(
            "SELECT DISTINCT toString(run_id) FROM v_capability_run_counters", "CSV"
        )
        .data()
        .split()
    )
    assert got == [f'"{CI_RUN}"']
