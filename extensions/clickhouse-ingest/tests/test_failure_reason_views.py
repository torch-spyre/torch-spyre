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

"""Runs schema/50's failure_* columns on chdb: which source explains a non-passing verdict, and
that a reasons row overrides the verdict's own props only at a higher confidence."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

AID = "2b397099-6200-52fb-98c4-b603961a0582"
DIAG = json.dumps(
    {"category": "infra_network", "evidence": "checkout retries", "is_infra": True}
)
# run_id suffix -> (state, props); each run is one verdict on AID.
VERDICTS = {
    "01": ("failed", {"diagnosis": DIAG}),
    "02": ("error", {"closed_reason": "parent_superseded"}),
    "03": ("failed", {"run_url": "https://ci/3"}),
    "04": ("error", {}),
    "05": ("passed", {"diagnosis": DIAG}),
    "06": ("error", {}),
    "07": ("error", {"failure_reason": "aborted", "failure_subreason": "user"}),
    "08": ("error", {"diagnosis": "infra_result_lost"}),
    "09": ("error", {"closed_reason": "parent_hung_jenkins_restart"}),
    "10": ("failed", {"closed_reason": "parent_manual_abort"}),
    "11": (
        "error",
        {
            "failure_reason": "ingest_error",
            "failure_subreason": "no_cases",
            "failure_confidence": "1",
        },
    ),
    "12": ("error", {}),
    "13": ("error", {"runner_died": "true"}),
}


def _rid(n):
    return f"00000000-0000-0000-0000-0000000000{n}"


@pytest.fixture(scope="module")
def db():
    s = session.Session()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "50-artifact-views.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)
    rows = [
        {"artifact_id": AID, "run_id": _rid(n), "result_kind": "functional",
         "test_type": "unit", "state": st, "arch": "s390x", "duration_s": 1, "props": p}
        for n, (st, p) in VERDICTS.items()
    ]  # fmt: skip
    s.query("INSERT INTO artifact_results FORMAT JSONEachRow\n"
            + "\n".join(map(json.dumps, rows)))  # fmt: skip
    s.query(
        "INSERT INTO run_case_counters (run_id, component, total_tests, passed, failed) "
        f"VALUES ('{_rid('03')}', 'torch-spyre', 5, 3, 2), ('{_rid('10')}', 'torch-spyre', 12, 1, 11)"
    )
    reasons = [
        (_rid("06"), "infra_timeout", "card_lock", 2, "backfill-console", "10:00"),
        (_rid("07"), "infra_env", "", 2, "backfill-console", "10:00"),
        (_rid("11"), "infra_timeout", "inner", 3, "jenkins", "10:00"),
        # One source re-classifying: its later, lower-confidence row is its answer.
        (_rid("12"), "infra_hardware", "", 3, "collector", "10:00"),
        (_rid("12"), "infra_network", "", 1, "collector", "11:00"),
    ]
    for rid, reason, sub, conf, src, at in reasons:
        s.query(
            "INSERT INTO artifact_result_reasons (updated_at, artifact_id, run_id, result_kind, "
            "test_type, failure_reason, failure_subreason, failure_wait_s, confidence, source) "
            f"VALUES ('2026-10-10 {at}:00', '{AID}', '{rid}', 'functional', 'unit', '{reason}', "
            f"'{sub}', 18000, {conf}, '{src}')"
        )
    yield {
        r["run_id"][-2:]: r
        for r in map(
            json.loads,
            str(
                s.query(
                    "SELECT run_id, failure_reason, failure_subreason, failure_detail, "
                    "failure_log_url, failure_wait_s, failure_source "
                    "FROM v_artifact_results_enriched",
                    "JSONEachRow",
                )
            ).splitlines(),
        )
    }
    s.close()


def _why(db, n):
    r = db[n]
    return r["failure_reason"], r["failure_subreason"], r["failure_source"]


def test_the_writers_diagnosis_names_the_reason_and_its_evidence(db):
    assert _why(db, "01") == ("infra_network", "", "writer")
    assert db["01"]["failure_detail"] == "checkout retries"


def test_the_stale_leg_cleanup_reads_as_superseded(db):
    assert _why(db, "02") == ("superseded", "by_newer_run", "writer")


def test_failed_cases_explain_a_verdict_nothing_else_does(db):
    assert _why(db, "03") == ("test_failure", "", "derived")
    assert db["03"]["failure_detail"] == "2 of 5 cases failed"
    assert db["03"]["failure_log_url"] == "https://ci/3"


def test_an_unexplained_failure_says_unknown_and_a_pass_says_nothing(db):
    assert _why(db, "04") == ("unknown", "", "derived")
    assert _why(db, "05") == ("", "", "")


def test_a_reasons_row_outranks_an_unexplained_verdict(db):
    assert _why(db, "06") == ("infra_timeout", "card_lock", "backfill-console")
    assert int(db["06"]["failure_wait_s"]) == 18000


def test_the_writers_own_reason_outranks_a_backfill_guess(db):
    assert _why(db, "07") == ("aborted", "user", "writer")


def test_the_pre_taxonomy_result_lost_reads_as_ingest_error(db):
    assert _why(db, "08") == ("ingest_error", "result_lost", "writer")


def test_a_jenkins_restart_reads_as_infra(db):
    assert _why(db, "09") == ("infra_capacity", "jenkins_restart", "writer")


def test_a_failed_close_is_explained_by_its_cases_not_the_cleanup(db):
    assert _why(db, "10") == ("test_failure", "", "derived")
    assert db["10"]["failure_detail"] == "11 of 12 cases failed"


def test_a_later_collector_outranks_the_clis_default(db):
    assert _why(db, "11") == ("infra_timeout", "inner", "jenkins")


def test_a_sources_latest_row_is_its_answer_whatever_its_confidence(db):
    assert _why(db, "12") == ("infra_network", "", "collector")


def test_groovys_runner_died_true_reads_as_infra(db):
    assert _why(db, "13") == ("infra_capacity", "runner_died", "writer")
