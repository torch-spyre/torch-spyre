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

"""artifact_sweep_runs is append-only; its CHECKs pin the vocabularies and the mode/status pairing."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

DIGEST = "ab" * 32


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "75-artifact-sweep-runs.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.close()


def _row(**over):
    row = {"run_id": "ci/artifactory-sweep#12", "mode": "plan", "status": "planned",
           "repos": ["rpm", "generic"], "days": 90, "keep": 3, "plan_digest": DIGEST}  # fmt: skip
    return {**row, **over}


def _insert(s, row):
    s.query("INSERT INTO artifact_sweep_runs FORMAT JSONEachRow\n" + json.dumps(row))


def _rows(s, sql):
    raw = s.query(sql, "JSONEachRow").data()
    raw = raw.decode() if isinstance(raw, bytes) else raw
    return [json.loads(line) for line in raw.splitlines()]


def test_every_column_round_trips(chdb_session):
    s = chdb_session
    _insert(
        s,
        _row(
            ts="2026-11-12 01:02:03.456",
            mode="apply",
            status="partial",
            selected_files=10,
            selected_bytes=2**40,
            deleted_files=9,
            failed_files=1,
            reason="1 delete failed",
            job_run="https://jenkins.example/job/x/13/",
            props={"plan_build": "12", "live_allowed": "true"},
        ),
    )
    got = _rows(
        s, "SELECT * EXCEPT (audit_uuid, audit_timestamp) FROM artifact_sweep_runs"
    )
    assert got == [
        {
            "ts": "2026-11-12 01:02:03.456",
            "run_id": "ci/artifactory-sweep#12",
            "mode": "apply",
            "status": "partial",
            "repos": ["rpm", "generic"],
            "days": 90,
            "keep": 3,
            "plan_digest": DIGEST,
            "selected_files": 10,
            "selected_bytes": 2**40,
            "deleted_files": 9,
            "failed_files": 1,
            "reason": "1 delete failed",
            "job_run": "https://jenkins.example/job/x/13/",
            "props": {"plan_build": "12", "live_allowed": "true"},
        }
    ]


LEGIT = [
    {"status": "planned", "selected_files": 5, "selected_bytes": 50},
    {"mode": "apply", "status": "started", "selected_files": 5, "selected_bytes": 50},
    {"mode": "apply", "status": "applied", "selected_files": 5, "deleted_files": 5},
    {"mode": "apply", "status": "partial", "deleted_files": 3, "failed_files": 2, "reason": "2 failed"},
    {"status": "skipped", "plan_digest": "", "reason": "kill switch"},
    {"mode": "apply", "status": "skipped", "plan_digest": "", "reason": "not due"},
    {"status": "refused", "plan_digest": "", "reason": "over ceiling"},
    {"mode": "apply", "status": "refused", "reason": "digest mismatch"},
    {"status": "failed", "plan_digest": "", "reason": "listing error"},
    {"mode": "apply", "status": "failed", "deleted_files": 3, "failed_files": 2},
]  # fmt: skip


@pytest.mark.parametrize(
    "row", LEGIT, ids=lambda r: f"{r.get('mode', 'plan')}-{r['status']}"
)
def test_every_legitimate_mode_status_pair_is_accepted(chdb_session, row):
    _insert(chdb_session, _row(**row))
    assert _rows(chdb_session, "SELECT count() AS n FROM artifact_sweep_runs") == [
        {"n": 1}
    ]


def test_the_range_edges_are_accepted(chdb_session):
    _insert(chdb_session, _row(days=3650, keep=1000))


def test_apply_once_lookup_reads_the_latest_row_per_digest(chdb_session):
    s = chdb_session
    other = "cd" * 32
    _insert(s, _row(ts="2026-11-12 01:00:00"))
    _insert(s, _row(ts="2026-11-12 02:00:00", plan_digest=other, run_id="j#2"))
    _insert(s, _row(ts="2026-11-12 03:00:00", mode="apply", status="started", run_id="j#3"))  # fmt: skip
    _insert(s, _row(ts="2026-11-12 04:00:00", mode="apply", status="applied", run_id="j#3", deleted_files=4))  # fmt: skip
    got = _rows(
        s,
        "SELECT plan_digest, argMax(status, ts) AS latest FROM artifact_sweep_runs"
        " GROUP BY plan_digest ORDER BY plan_digest",
    )
    assert got == [
        {"plan_digest": DIGEST, "latest": "applied"},
        {"plan_digest": other, "latest": "planned"},
    ]


APPLY_ONCE = (
    "SELECT countIf(mode = 'apply' AND status IN ('started', 'applied', 'partial', 'failed')) AS n"
    " FROM artifact_sweep_runs WHERE plan_digest = '{d}'"
)


def test_apply_once_counts_only_claiming_apply_rows(chdb_session):
    s = chdb_session
    _insert(s, _row())
    _insert(s, _row(mode="apply", status="refused", reason="mismatch", run_id="j#2"))
    _insert(s, _row(mode="apply", status="skipped", reason="not due", run_id="j#3"))
    assert _rows(s, APPLY_ONCE.format(d=DIGEST)) == [{"n": 0}]
    for n, status in enumerate(("started", "applied", "failed"), start=1):
        _insert(s, _row(mode="apply", status=status, run_id=f"c#{n}"))
        assert _rows(s, APPLY_ONCE.format(d=DIGEST)) == [{"n": n}]
    _insert(s, _row(mode="apply", status="partial", run_id="c#4", failed_files=1))
    assert _rows(s, APPLY_ONCE.format(d=DIGEST)) == [{"n": 4}]
    assert _rows(s, APPLY_ONCE.format(d="ef" * 32)) == [{"n": 0}]


def test_a_skipped_run_needs_no_plan(chdb_session):
    _insert(
        chdb_session,
        _row(mode="plan", status="skipped", plan_digest="", reason="kill switch"),
    )
    assert _rows(
        chdb_session, "SELECT plan_digest, selected_files FROM artifact_sweep_runs"
    ) == [{"plan_digest": "", "selected_files": 0}]


@pytest.mark.parametrize(
    "bad",
    [
        {"run_id": ""},
        {"mode": "delete", "status": "skipped", "reason": "x"},
        {"status": "done"},
        {"mode": "apply", "status": "planned"},
        {"mode": "plan", "status": "applied"},
        {"mode": "plan", "status": "partial"},
        {"plan_digest": "AB" * 32},
        {"plan_digest": "ab" * 31},
        {"plan_digest": ""},
        {"run_id": "   "},
        {"mode": "apply", "status": "started", "plan_digest": ""},
        {"mode": "plan", "status": "started"},
        {"mode": "apply", "status": "started", "deleted_files": 1},
        {"mode": "apply", "status": "started", "failed_files": 1},
        {"mode": "apply", "status": "applied", "plan_digest": ""},
        {"mode": "apply", "status": "partial", "plan_digest": "", "failed_files": 1},
        {"days": 3651},
        {"days": 70000},
        {"keep": 1001},
        {"keep": 65536},
        {"deleted_files": 1},
        {"failed_files": 1},
        {"mode": "apply", "status": "applied", "failed_files": 1},
        {"mode": "apply", "status": "partial", "failed_files": 0},
        {"mode": "apply", "status": "refused", "reason": ""},
        {"status": "skipped", "plan_digest": "", "reason": ""},
    ],
)
def test_rows_are_validated(chdb_session, bad):
    with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
        _insert(chdb_session, _row(**bad))


def test_rows_are_never_rewritten(chdb_session):
    s = chdb_session
    _insert(s, _row())
    _insert(s, _row(mode="apply", status="applied", deleted_files=5))
    assert _rows(s, "SELECT count() AS n FROM artifact_sweep_runs") == [{"n": 2}]
