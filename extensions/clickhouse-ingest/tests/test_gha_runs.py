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

import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
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
    assert (
        timed_out["result"],
        timed_out["failure_reason"],
        timed_out["failure_is_infra"],
    ) == (
        "timed_out",
        "infra_timeout",
        True,
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
    attempts, rows, _ = gha_runs.poll(gh, client, "db", REPO, *_window())
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
    assert gha_runs.poll(gh, FakeClient(held), "db", REPO, *_window()) == (0, 0, "")
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
    assert gha_runs.poll(FakeGitHub([_run()]), client, "db", REPO, *_window()) == (
        1,
        2,
        "",
    )
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


def test_get_retries_a_truncated_response(monkeypatch):
    import http.client
    import io

    calls = []

    class Resp(io.BytesIO):
        headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"{", 10)
        return Resp(b'{"ok": true}')

    monkeypatch.setattr(gha_runs.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gha_runs.time, "sleep", lambda s: None)
    assert gha_runs.GitHub("t").get("/x") == {"ok": True} and len(calls) == 2


NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


def _created(i):
    return f"{NOW - timedelta(hours=7) + timedelta(minutes=10 * i):%Y-%m-%dT%H:%M:%SZ}"


class FakeAPI(gha_runs.GitHub):
    """The Actions endpoints over a list of runs; every counted call costs 1 quota, 1 minute.

    Listing takes 2 calls (a full page, then an empty one); each run then costs 1 (its jobs).
    """

    def __init__(self, runs, limit=100, remaining=100, clock=None, **kw):
        super().__init__("t", **kw)
        self.all, self.quota, self.left = runs, limit, remaining
        self.clock, self.paths = clock, []
        self.reset_at = clock.now + 3600

    def _fetch(self, req):
        url = urllib.parse.urlparse(req.full_url)
        q = dict(urllib.parse.parse_qsl(url.query))
        if self.clock.now >= self.reset_at:
            self.left, self.reset_at = self.quota, self.reset_at + 3600
        headers = {
            "X-RateLimit-Limit": str(self.quota),
            "X-RateLimit-Remaining": str(self.left),
            "X-RateLimit-Reset": str(self.reset_at),
        }
        if url.path == "/rate_limit":
            core = {"limit": self.quota, "remaining": self.left, "reset": 0}
            return {"resources": {"core": core}}, headers
        self.left -= 1
        self.clock.now += 60
        headers["X-RateLimit-Remaining"] = str(self.left)
        self.paths.append(url.path)
        if url.path.endswith("/actions/runs"):
            lo, hi = q["created"].split("..")
            hit = sorted(
                (r for r in self.all if lo <= r["created_at"] <= hi),
                key=lambda r: r["created_at"],
                reverse=True,
            )
            page = int(q["page"])
            return {
                "total_count": len(hit),
                "workflow_runs": hit[(page - 1) * 100 : page * 100],
            }, headers
        run_id, attempt = (int(x) for x in url.path.split("/")[6:9:2])
        if url.path.endswith("/jobs"):
            return {"total_count": 1, "jobs": [_job(id=run_id * 10)]}, headers
        return next(r for r in self.all if r["id"] == run_id), headers


class Clock:
    def __init__(self):
        self.now = NOW.timestamp()
        self.slept = []

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(gha_runs.time, "time", lambda: c.now)
    monkeypatch.setattr(gha_runs.time, "sleep", c.sleep)
    return c


class Store:
    """A pipeline_runs stand-in answering the poller's stored() and watermark() reads."""

    def __init__(self):
        self.rows = []

    def insert(self, table, rows, column_names=None, database=None):
        self.rows += [dict(zip(column_names, r)) for r in rows]

    def query(self, sql, parameters=None):
        wf = [r for r in self.rows if r["pipeline_type"] == "gha-workflow"]
        if "count()" in sql:
            first = [
                r["started_at"] - timedelta(milliseconds=r["queue_ms"])
                for r in wf
                if r["run_key"].endswith("#1")
            ]
            rows = [(len(first), max(first) if first else None)]
        else:
            rows = [(r["run_key"], r["updated_at"], r["state"]) for r in wf]
        return type("R", (), {"result_rows": rows})()

    def keys(self):
        return [r["run_key"] for r in self.rows if r["pipeline_type"] == "gha-workflow"]


def _runs(n):
    return [
        _run(
            id=1000 + i,
            created_at=_created(i),
            run_started_at=_created(i),
            updated_at=_created(i),
        )
        for i in range(n)
    ]


def _key(i):
    return f"gha:{REPO}/{1000 + i}#1"


def _poll(api, store, start=None):
    api.rate()
    return gha_runs.poll(api, store, "db", REPO, start or NOW - timedelta(hours=8), NOW)


def test_poll_stops_at_the_reserve_and_keeps_what_it_finished(clock):
    # 60 left with half reserved leaves 10 calls: the listing, then 8 runs.
    api, store = FakeAPI(_runs(20), remaining=60, clock=clock), Store()
    attempts, rows, reason = _poll(api, store)
    assert reason == "budget reached at 50 remaining"
    assert api.left == 50 and attempts == 8
    assert store.keys() == [_key(i) for i in range(8)]


def test_poll_stops_at_the_deadline(clock):
    api = FakeAPI(_runs(20), clock=clock, deadline=clock.now + 7 * 60)
    attempts, _, reason = _poll(api, Store())
    assert reason == "deadline reached" and attempts == 5


def test_poll_works_oldest_first_and_resumes_without_a_hole(clock):
    runs, store = _runs(20), Store()
    api = FakeAPI(runs, remaining=60, clock=clock)
    assert _poll(api, store)[2]
    # The next run gets a fresh quota and starts from the watermark.
    mark = gha_runs.watermark(store, "db", REPO)
    assert mark == gha_runs._ts(_created(7))
    start, resumed = gha_runs.window_start(
        NOW, timedelta(minutes=30), mark, timedelta(hours=6), timedelta(days=90)
    )
    assert resumed
    attempts, _, reason = _poll(FakeAPI(runs, clock=clock), store, start)
    assert reason == "" and attempts == 12
    assert store.keys() == [_key(i) for i in range(20)]


def test_the_watermark_extends_the_window_after_an_outage():
    day = timedelta(days=1)
    args = (timedelta(hours=6), timedelta(days=90))
    # Polling as usual: the watermark is recent and --hours already reaches past it.
    assert gha_runs.window_start(
        NOW, timedelta(hours=8), NOW - timedelta(minutes=15), *args
    ) == (NOW - timedelta(hours=8), False)
    assert gha_runs.window_start(NOW, timedelta(hours=8), NOW - 3 * day, *args) == (
        NOW - 3 * day - timedelta(hours=6),
        True,
    )
    assert gha_runs.window_start(NOW, timedelta(hours=8), None, *args) == (
        NOW - timedelta(hours=8),
        False,
    )


def test_catch_up_is_capped_but_an_explicit_backfill_is_not():
    day = timedelta(days=1)
    args = (timedelta(hours=6), timedelta(days=90))
    assert gha_runs.window_start(NOW, timedelta(hours=8), NOW - 200 * day, *args) == (
        NOW - 90 * day,
        True,
    )
    assert gha_runs.window_start(NOW, 120 * day, NOW - 200 * day, *args) == (
        NOW - 120 * day,
        False,
    )


def test_wait_for_reset_sleeps_instead_of_stopping(clock):
    api = FakeAPI(_runs(20), remaining=60, clock=clock, wait_for_reset=True)
    store = Store()
    attempts, _, reason = _poll(api, store)
    assert reason == "" and attempts == 20
    # 10 calls at 1 minute each, then a sleep to the hourly reset.
    assert clock.slept == [3600 - 10 * 60 + 1]
    assert store.keys() == [_key(i) for i in range(20)]


def test_wait_for_reset_still_honours_the_deadline(clock):
    api = FakeAPI(
        _runs(20),
        remaining=60,
        clock=clock,
        wait_for_reset=True,
        deadline=clock.now + 30 * 60,
    )
    assert _poll(api, Store())[2] == "deadline reached"
    assert clock.slept == []


def test_cli_ends_a_stopped_poll_with_a_notice_not_an_error(clock, monkeypatch, capsys):
    now = datetime.now(timezone.utc)
    runs = [
        _run(
            id=1000 + i,
            created_at=f"{now - timedelta(minutes=60 - i):%Y-%m-%dT%H:%M:%SZ}",
        )
        for i in range(5)
    ]
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(
        gha_runs,
        "GitHub",
        lambda token, **kw: FakeAPI(runs, remaining=53, clock=clock, **kw),
    )
    gha_runs.main(
        ["poll", "--repo", REPO, "--repo", "torch-spyre/hf-adapters", "--dry-run"]
    )
    err = capsys.readouterr().err
    assert "[info] rate limit 53/100 at start" in err
    assert "[notice] budget reached at 50 remaining; resume next run" in err
    assert "hf-adapters" not in err and "rate limit 50/100 at end" in err
