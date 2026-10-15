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

"""Runs schema/54's views on chdb over rows the real writer builds: one run per lane, a
superseded rerun and a re-written run, each folded to one row in its own lane."""

import json
from datetime import datetime, timezone

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
from spyre_clickhouse_ingest.ci_run_timings import build_rows

session = pytest.importorskip("chdb.session")

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _ms(hms: str) -> int:
    """Epoch ms of 2026-10-08T<hms>Z, the form the Jenkins side sends."""
    at = datetime.fromisoformat(f"2026-10-08T{hms}").replace(tzinfo=timezone.utc)
    return int(at.timestamp() * 1000)


def _run(n, source, **over):
    """A finished run: one build that built, one dropped, one GHA test cell."""
    run = {
        "run_key": f"Spyre/orchestrator#{n}",
        "trigger_kind": "upstream",
        "trigger_source": source,
        "preset": "trigger-pr-validation",
        "build_mode": "pr",
        "trigger_pr": f"github.ibm.com/ai-chip-toolchain/deeptools#{n}",
        "base_ref": "main",
        "verdict": "green",
        "result": "SUCCESS",
        "scheduled_at": _ms("10:00:00"),
        "started_at": _ms("10:00:30"),
        "ended_at": _ms("11:00:00"),
        **over.pop("run", {}),
    }
    return {
        "run": run,
        "builds": [
            {
                "component": "deeptools",
                "arch": "amd64",
                "state": "built",
                "queued_at": _ms("10:01:00"),
                "started_at": _ms("10:02:00"),
                "ended_at": _ms("10:20:00"),
            },
            {"component": "flex", "arch": "amd64", "state": "dropped"},
        ],
        "tests": [
            {
                "component": "torch-spyre",
                "arch": "amd64",
                "modes": "smoke",
                "result": "SUCCESS",
                "queued_at": _ms("10:21:00"),
                "started_at": _ms("10:21:10"),
                "ended_at": _ms("10:55:00"),
                "exec": {
                    "executor": "gha-ephemeral",
                    "dispatched_at": _ms("10:22:00"),
                    "started_at": _ms("10:26:00"),
                    "ended_at": _ms("10:50:00"),
                },
            }
        ],
        **over,
    }


SPYRE_TEST = _run(
    1,
    "spyre-test",
    run={
        "pickup_path": "webhook",
        "comment_at": "2026-10-08T09:59:50Z",
        "picked_up_at": _ms("09:59:55"),
        "pr_queued_at": _ms("10:00:40"),
        "pr_running_at": _ms("10:02:00"),
    },
)
# The same PR again, cancelled by a newer /spyre-test.
SUPERSEDED = _run(
    2, "spyre-test", run={"superseded": True, "verdict": "", "result": "ABORTED"}
)
MERGE_QUEUE = _run(3, "merge-queue")
# What main-push-build sends: the merge commit's time, its own start, the repo and merged PR.
MERGED = {
    "trigger_pr": "",
    "preset": "main-build-and-test",
    "build_mode": "reuse",
    "repo": "torch-spyre",
    "comment_at": "2026-10-08T09:58:00Z",
    "picked_up_at": _ms("09:59:00"),
    "pickup_path": "webhook",
}
MAIN_PUSH = _run(
    4,
    "main-push",
    run={**MERGED, "pr_number": "5294", "verdict": "red", "result": "FAILURE"},
)
MAIN_PUSH["builds"].append(
    {
        "component": "spyre-comms",
        "arch": "amd64",
        "state": "failed",
        "failed_stage": "Build",
        "failure_reason": "compile_error",
    }
)
MAIN_PUSH["tests"][0].update(
    result="FAILURE", failed_stage="Test", failure_reason="runner_lost"
)
MAIN_PUSH_GREEN = _run(5, "main-push", run={**MERGED, "pr_number": "5295"})


class Chdb:
    def __init__(self):
        self.s = session.Session()

    def command(self, sql):
        return str(self.s.query(sql, "TSV")).rstrip("\n")

    def rows(self, sql):
        return [
            json.loads(x) for x in str(self.s.query(sql, "JSONEachRow")).splitlines()
        ]

    def write(self, batch, now=NOW):
        def cell(v):
            return (
                v.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if isinstance(v, datetime)
                else v
            )

        body = "\n".join(
            json.dumps({k: cell(v) for k, v in r.items()})
            for r in build_rows(batch, now)
        )
        self.command(f"INSERT INTO ci_run_timings FORMAT JSONEachRow\n{body}")


@pytest.fixture
def db():
    c = Chdb()
    for name in ("48-ci-run-timings.sql", "54-ci-run-timing-views.sql"):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            c.command(stmt)
    for batch in (SPYRE_TEST, SUPERSEDED, MERGE_QUEUE, MAIN_PUSH, MAIN_PUSH_GREEN):
        c.write(batch)
    yield c
    c.s.close()


def test_entry_view_keeps_the_latest_write_and_the_materialized_spans(db):
    # Re-writing a run replaces its rows: FINAL in the view, no duplicate entries.
    db.write(_run(3, "merge-queue", run={"verdict": "red"}), now=NOW.replace(hour=13))
    rows = db.rows(
        "SELECT entry, verdict, queue_ms, exec_queue_ms FROM v_ci_run_timings "
        "WHERE run_key = 'Spyre/orchestrator#3' ORDER BY entry, component"
    )
    assert [(r["entry"], r["verdict"]) for r in rows] == [
        ("build", "red"),
        ("build", "red"),
        ("test", "red"),
    ]
    assert rows[0]["queue_ms"] == 60000 and rows[2]["exec_queue_ms"] == 240000


def test_one_row_per_run_with_its_rollups(db):
    runs = {r["run_key"]: r for r in db.rows("SELECT * FROM v_ci_runs")}
    assert len(runs) == 5
    r = runs["Spyre/orchestrator#3"]
    assert (r["outcome"], r["failure_summary"]) == ("passed", "")
    assert (r["builds"], r["builds_built"], r["builds_dropped"]) == (2, 1, 1)
    assert (r["test_cells"], r["tests_passed"]) == (1, 1)
    assert r["jenkins_queue_ms"] == 30000
    assert r["build_queue_max_ms"] == 60000
    assert r["build_max_ms"] == 18 * 60000
    assert r["build_phase_ms"] == 19 * 60000
    assert r["test_phase_ms"] == 34 * 60000
    assert r["exec_queue_max_ms"] == 4 * 60000
    assert r["exec_max_ms"] == 24 * 60000
    assert r["run_ms"] == 3570000


def test_each_lane_view_holds_only_its_lane(db):
    def keys(view):
        return sorted(r["run_key"] for r in db.rows(f"SELECT run_key FROM {view}"))

    assert keys("v_ci_runs_spyre_test") == [
        "Spyre/orchestrator#1",
        "Spyre/orchestrator#2",
    ]
    assert keys("v_ci_runs_merge_queue") == ["Spyre/orchestrator#3"]
    assert keys("v_ci_runs_main_push") == [
        "Spyre/orchestrator#4",
        "Spyre/orchestrator#5",
    ]


def test_spyre_test_is_a_timeline_from_comment_to_verdict(db):
    (r,) = db.rows(
        "SELECT * FROM v_ci_runs_spyre_test WHERE run_key = 'Spyre/orchestrator#1'"
    )
    assert (r["repo"], r["pr_number"], r["pickup_path"]) == ("deeptools", 1, "webhook")
    assert r["outcome"] == "passed"
    assert r["build_first_started_at"].startswith("2026-10-08 10:02:00")
    assert r["test_first_started_at"].startswith("2026-10-08 10:21:10")
    assert r["comment_to_pickup_ms"] == 5000
    assert r["pickup_to_scheduled_ms"] == 5000
    assert r["jenkins_queue_ms"] == 30000
    assert r["comment_to_queued_ms"] == 50000
    assert r["queued_to_running_ms"] == 80000
    assert r["comment_to_end_ms"] == 3610000


def test_an_aborted_rerun_reads_superseded(db):
    (r,) = db.rows(
        "SELECT outcome, failure_summary FROM v_ci_runs_spyre_test "
        "WHERE run_key = 'Spyre/orchestrator#2'"
    )
    assert (r["outcome"], r["failure_summary"]) == ("superseded", "")


def test_a_green_main_push_runs_from_merge_to_green(db):
    (r,) = db.rows(
        "SELECT * FROM v_ci_runs_main_push WHERE run_key = 'Spyre/orchestrator#5'"
    )
    assert (r["repo"], r["pr_number"], r["outcome"]) == ("torch-spyre", 5295, "passed")
    assert r["merged_at"].startswith("2026-10-08 09:58:00")
    assert r["merge_to_trigger_ms"] == 60000
    assert r["trigger_to_scheduled_ms"] == 60000
    assert r["merge_to_end_ms"] == r["merge_to_green_ms"] == 62 * 60000
    assert r["scheduled_to_end_ms"] == 60 * 60000
    assert r["built_components"] == ["deeptools"]
    assert "comment_to_queued_ms" not in r


def test_a_red_main_push_says_what_broke_instead_of_a_green_time(db):
    (r,) = db.rows(
        "SELECT * FROM v_ci_runs_main_push WHERE run_key = 'Spyre/orchestrator#4'"
    )
    assert r["outcome"] == "failed"
    assert r["merge_to_green_ms"] is None and r["merge_to_end_ms"] == 62 * 60000
    assert r["failed_components"] == ["spyre-comms"]
    assert r["failure_summary"] == (
        "spyre-comms x86_64 build failed in Build: compile_error | "
        "torch-spyre x86_64 [smoke] test failed in Test: runner_lost"
    )
    assert r["failed_builds"][0][:5] == [
        "spyre-comms",
        "x86_64",
        1,
        "Build",
        "compile_error",
    ]


def test_a_failed_run_with_no_failed_entry_still_says_so(db):
    db.write(_run(6, "main-push", run={"verdict": "red", "result": "FAILURE"}))
    (r,) = db.rows(
        "SELECT outcome, failure_summary, merge_to_end_ms FROM v_ci_runs_main_push "
        "WHERE run_key = 'Spyre/orchestrator#6'"
    )
    assert r == {
        "outcome": "failed",
        "failure_summary": "run failure with no failed build or test",
        # main-push-build has not sent the merge time: unknown, not 0.
        "merge_to_end_ms": None,
    }


def test_daily_leaves_superseded_runs_out_of_rates_and_spans(db):
    days = {r["trigger_source"]: r for r in db.rows("SELECT * FROM v_ci_lane_daily")}
    assert set(days) == {"spyre-test", "merge-queue", "main-push"}
    st = days["spyre-test"]
    assert (st["runs"], st["superseded_runs"], st["green"], st["pass_rate"]) == (
        2,
        1,
        1,
        1,
    )
    assert st["comment_to_queued_p50_min"] == pytest.approx(50000 / 60000)
    # The merge queue has no trigger event: the span is NULL, not a 0 that drags a median down.
    assert days["merge-queue"]["comment_to_end_p50_min"] is None
    assert days["main-push"]["comment_to_end_p50_min"] == pytest.approx(62)
    assert days["merge-queue"]["jenkins_queue_p50_min"] == pytest.approx(0.5)


def test_a_date_filter_prunes_partitions_through_final_and_group_by(db):
    old = _run(9, "merge-queue")
    old["run"]["started_at"] -= 40 * 86400 * 1000
    db.write(old)
    plan = db.command(
        "EXPLAIN indexes = 1 SELECT count() FROM v_ci_runs_merge_queue "
        "WHERE run_started_at >= '2026-10-01'"
    )
    assert "Condition: (toYYYYMM(run_started_at) in [202610, +Inf))" in plan
