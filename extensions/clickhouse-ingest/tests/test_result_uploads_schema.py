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
