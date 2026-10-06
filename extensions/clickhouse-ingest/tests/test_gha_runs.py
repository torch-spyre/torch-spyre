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

from datetime import datetime, timedelta, timezone
from pathlib import Path

from spyre_clickhouse_ingest import gha_runs
from spyre_clickhouse_ingest.schema import PipelineRuns

REPO = "torch-spyre/torch-spyre"
DDL = Path(__file__).resolve().parents[1] / "schema" / "47-pipeline-runs.sql"


def _run(**kw):
    run = {
        "id": 900001,
        "run_attempt": 1,
        "run_number": 777,
        "name": "tests",
        "path": ".github/workflows/tests.yml",
        "event": "pull_request",
        "head_branch": "feature-x",
        "head_sha": "abc123",
        "status": "completed",
        "conclusion": "failure",
        "html_url": "https://github.com/torch-spyre/torch-spyre/actions/runs/900001",
        "created_at": "2026-10-05T10:00:00Z",
        "run_started_at": "2026-10-05T10:02:00Z",
        "updated_at": "2026-10-05T10:40:00Z",
        "pull_requests": [{"number": 5190}],
    }
    run.update(kw)
    return run


def _job(**kw):
    job = {
        "id": 11,
        "run_attempt": 1,
        "name": "run-tests (x86_64)",
        "status": "completed",
        "conclusion": "failure",
        "created_at": "2026-10-05T10:02:00Z",
        "started_at": "2026-10-05T10:05:00Z",
        "completed_at": "2026-10-05T10:39:00Z",
        "runner_name": "spyre-pf-x1-abc",
        "labels": ["spyre_pf_x1"],
        "html_url": "https://github.com/x/job/11",
        "steps": [
            {"name": "checkout", "conclusion": "success"},
            {"name": "Run tests", "conclusion": "failure"},
        ],
    }
    job.update(kw)
    return job


def test_model_columns_match_the_ddl_order():
    body = (
        DDL.read_text()
        .split("CREATE TABLE IF NOT EXISTS pipeline_runs", 1)[1]
        .split("ENGINE", 1)[0]
    )
    cols = [
        line.split()[0]
        for line in body.splitlines()
        if line.startswith("    ")
        and not line.startswith("     ")
        and line.split()
        and not line.lstrip().startswith(("--", "(", ")"))
    ]
    assert list(PipelineRuns.columns) == [
        c for c in cols if c not in ("CONSTRAINT", "audit_uuid", "audit_timestamp")
    ]


def test_lane_follows_the_jenkins_names():
    assert gha_runs.lane({"event": "merge_group"}) == "merge-queue"
    assert gha_runs.lane({"event": "pull_request"}) == "spyre-test"
    assert gha_runs.lane({"event": "push", "head_branch": "main"}) == "main-push"
    assert gha_runs.lane({"event": "push", "head_branch": "release-0.5"}) == "push"
    assert gha_runs.lane({"event": "workflow_run"}) == "chained"
    assert gha_runs.lane({"event": "repository_dispatch"}) == "repository_dispatch"


def test_pr_number_from_link_or_merge_queue_branch():
    assert gha_runs.pr_number(_run()) == 5190
    mq = _run(
        pull_requests=[], head_branch="gh-readonly-queue/main/pr-5170-7a5f4f75ea4e"
    )
    assert gha_runs.pr_number(mq) == 5170
    assert gha_runs.pr_number(_run(pull_requests=[], head_branch="main")) == 0


def test_job_arch_from_runner_labels():
    assert gha_runs.job_arch(["spyre_pf_x1"]) == "x86_64"
    assert gha_runs.job_arch(["self-hosted", "spyre-s390x"]) == "s390x"
    assert gha_runs.job_arch(["ppc64le-runner"]) == "ppc64le"


def test_workflow_row_is_a_valid_attempt_row():
    row = gha_runs.workflow_row(REPO, _run(), [_job()])
    PipelineRuns.row(row)
    assert row["run_key"] == "gha:torch-spyre/torch-spyre/900001#1"
    assert row["job_name"] == "torch-spyre/torch-spyre/.github/workflows/tests.yml"
    assert (row["source"], row["pipeline_type"], row["state"]) == (
        "gha",
        "gha-workflow",
        "finished",
    )
    assert (row["repo"], row["pr_number"], row["trigger_source"]) == (
        "torch-spyre",
        5190,
        "spyre-test",
    )
    assert row["queue_ms"] == 120_000 and row["duration_ms"] == 38 * 60_000
    assert row["build_url"].endswith("/attempts/1")
    assert row["failed_stage"] == "run-tests (x86_64)" and row["arches"] == ["x86_64"]


def test_job_row_links_to_its_attempt_and_names_the_failed_step():
    row = gha_runs.job_row(REPO, _run(), _job())
    PipelineRuns.row(row)
    assert row["run_key"] == "gha:torch-spyre/torch-spyre/900001#1/11"
    assert row["parent_run_key"] == "gha:torch-spyre/torch-spyre/900001#1"
    assert (row["agent"], row["queue_ms"], row["failed_stage"]) == (
        "spyre-pf-x1-abc",
        180_000,
        "Run tests",
    )


def test_running_and_timed_out_rows():
    running = gha_runs.workflow_row(
        REPO, _run(status="in_progress", conclusion=None), []
    )
    assert (running["state"], running["result"], running["ended_at"]) == (
        "running",
        "",
        None,
    )
    timed_out = gha_runs.job_row(REPO, _run(), _job(conclusion="timed_out", steps=[]))
    assert (timed_out["result"], timed_out["failure_reason"]) == (
        "timed_out",
        "infra_timeout",
    )


def test_a_job_that_never_started_has_no_row():
    assert gha_runs.job_row(REPO, _run(), _job(started_at=None)) is None


class FakeGitHub:
    def __init__(self, runs):
        self._runs, self.calls = runs, []

    def runs(self, repo, start, end):
        yield from self._runs

    def attempt(self, repo, run_id, attempt):
        self.calls.append(("attempt", attempt))
        return _run(id=run_id, run_attempt=attempt, conclusion="failure")

    def jobs(self, repo, run_id, attempt):
        self.calls.append(("jobs", attempt))
        return [_job(id=10 + attempt, run_attempt=attempt)]


class FakeClient:
    def __init__(self, stored_rows=()):
        self.stored_rows, self.inserted = list(stored_rows), []

    def query(self, sql, parameters=None):
        return type("R", (), {"result_rows": self.stored_rows})()

    def insert(self, table, rows, column_names=None, database=None):
        self.inserted += [dict(zip(column_names, r)) for r in rows]


def _window():
    end = datetime(2026, 10, 6, tzinfo=timezone.utc)
    return end - timedelta(hours=8), end


def test_poll_writes_every_attempt_of_a_rerun():
    gh, client = FakeGitHub([_run(run_attempt=2, conclusion="success")]), FakeClient()
    attempts, rows = gha_runs.poll(gh, client, "db", REPO, *_window())
    keys = sorted(
        r["run_key"] for r in client.inserted if r["pipeline_type"] == "gha-workflow"
    )
    assert keys == [
        "gha:torch-spyre/torch-spyre/900001#1",
        "gha:torch-spyre/torch-spyre/900001#2",
    ]
    assert (attempts, rows) == (2, 4) and ("attempt", 1) in gh.calls


def test_poll_skips_finished_attempts_it_already_holds():
    updated = datetime(2026, 10, 5, 10, 40, tzinfo=timezone.utc)
    held = [
        ("gha:torch-spyre/torch-spyre/900001#1", updated, "finished"),
        ("gha:torch-spyre/torch-spyre/900001#2", updated, "finished"),
    ]
    gh = FakeGitHub([_run(run_attempt=2, conclusion="success")])
    assert gha_runs.poll(gh, FakeClient(held), "db", REPO, *_window()) == (0, 0)
    assert gh.calls == []


def test_poll_rewrites_an_attempt_held_as_running():
    held = [
        (
            "gha:torch-spyre/torch-spyre/900001#1",
            datetime(2026, 10, 5, 10, 10, tzinfo=timezone.utc),
            "running",
        )
    ]
    client = FakeClient(held)
    assert gha_runs.poll(FakeGitHub([_run()]), client, "db", REPO, *_window()) == (1, 2)
    assert client.inserted[0]["state"] == "finished"


def test_runs_splits_a_window_over_the_list_cap():
    class Paged(gha_runs.GitHub):
        def __init__(self):
            super().__init__("t")
            self.windows = []

        def get(self, path, **q):
            lo, hi = (
                datetime.fromisoformat(x.replace("Z", "+00:00"))
                for x in q["created"].split("..")
            )
            if q["page"] == 1:
                self.windows.append((lo, hi))
            total = 1500 if hi - lo > timedelta(hours=4) else 10
            return {
                "total_count": total,
                "workflow_runs": [{"id": hash((lo, hi))}] if q["page"] == 1 else [],
            }

    gh = Paged()
    start, end = _window()
    runs = list(gh.runs(REPO, start, end))
    assert len(runs) == 2 and all(
        hi - lo <= timedelta(hours=4) for lo, hi in gh.windows[1:]
    )


def test_backoff_waits_out_the_rate_limit_and_gives_up_on_client_errors():
    import time
    import urllib.error

    def err(code, **headers):
        return urllib.error.HTTPError("u", code, "x", headers, None)

    reset = str(int(time.time()) + 30)
    assert (
        25
        < gha_runs.GitHub._backoff(
            err(403, **{"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}), 0
        )
        <= 32
    )
    assert gha_runs.GitHub._backoff(err(502), 2) == 4
    assert gha_runs.GitHub._backoff(err(404), 0) is None
    assert (
        gha_runs.GitHub._backoff(err(403, **{"X-RateLimit-Remaining": "12"}), 0) is None
    )
