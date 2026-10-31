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

"""ci_requests is the dashboard's launch log; ops_requests carries the same channel and plan keys."""

import json

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

RID = "018f0000-0000-7000-8000-000000000001"
NIL = "00000000-0000-0000-0000-000000000000"
ARTIFACT = "11111111-1111-1111-1111-111111111111"


def _in_list(file: str, name: str) -> list:
    sql = (SCHEMA_DIR / file).read_text()
    m = re.search(rf"CONSTRAINT\s+{name}\s+CHECK\s+\w+\s+IN\s*\(([^)]*)\)", sql)
    assert m, f"{name} not in {file}"
    return re.findall(r"'([^']+)'", m.group(1))


def test_ops_requests_carries_the_dashboard_join_keys():
    sql = (SCHEMA_DIR / "41-ops-requests.sql").read_text()
    for col in ("channel", "on_behalf_of", "plan_id"):
        assert re.search(rf"^\s+{col}\s", sql, re.M), col


def test_ci_requests_states_include_the_trigger_job_outcomes():
    # A launched/rejected/failed request is recorded under the same word on both sides.
    ops = set(_in_list("41-ops-requests.sql", "chk_state"))
    ci = set(_in_list("42-ci-requests.sql", "chk_state"))
    assert {"rejected", "failed"} <= ci
    assert "launched" in ci and "dry_run" in ops and "previewed" in ci


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "49-result-uploads.sql",
        "42-ci-requests.sql",
        "50-artifact-views.sql",
        "59-ci-request-views.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)
    yield s
    s.close()


def _request(**over):
    row = {"request_id": RID, "plan_id": "p1", "state": "previewed", "actor": "alice@ibm.com",
           "channel": "ui", "preset": "perf-only", "arches": ["x86_64"], "modes": ["perf"]}  # fmt: skip
    return {**row, **over}


def test_rows_are_validated_and_the_view_keeps_the_first_requester(chdb_session):
    s = chdb_session

    def insert(row):
        s.query("INSERT INTO ci_requests FORMAT JSONEachRow\n" + json.dumps(row))

    for bad in ({"request_id": NIL}, {"state": "dry_run"}, {"channel": "slack"}):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            insert(_request(**bad))
    insert(_request())
    insert(
        _request(
            state="launched",
            actor="svc-dashboard",
            job_run="https://j/1",
            build_url="https://j/2",
        )
    )
    got = s.query(
        "SELECT state, requested_by, job_run, build_url FROM v_ci_requests",
        "JSONEachRow",
    ).data()
    assert json.loads(got) == {"state": "launched", "requested_by": "alice@ibm.com",
                               "job_run": "https://j/1", "build_url": "https://j/2"}  # fmt: skip


def test_cell_durations_use_passed_nonzero_legs_and_one_arch_spelling(chdb_session):
    s = chdb_session
    s.query(
        "INSERT INTO artifacts (artifact_id, component, arch, kind, artifact_name, origin) "
        f"VALUES ('{ARTIFACT}', 'torch-spyre', 'x86_64', 'image', 'torch-spyre-dev', 'built')"
    )
    for i, (state, arch, dur) in enumerate(
        [
            ("passed", "amd64", 100),
            ("passed", "x86_64", 300),
            ("failed", "x86_64", 5),
            ("passed", "x86_64", 0),
        ]
    ):
        s.query(
            "INSERT INTO artifact_results (artifact_id, run_id, result_kind, test_type, state, arch, duration_s) "
            f"VALUES ('{ARTIFACT}', '2222222{i}-2222-2222-2222-222222222222', 'functional', 'smoke', "
            f"'{state}', '{arch}', {dur})"
        )
    got = json.loads(
        s.query(
            "SELECT component, test_type, arch, n FROM v_ci_cell_durations",
            "JSONEachRow",
        ).data()
    )
    assert got == {
        "component": "torch-spyre",
        "test_type": "smoke",
        "arch": "x86_64",
        "n": 2,
    }
