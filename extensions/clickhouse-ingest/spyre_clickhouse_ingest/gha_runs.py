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

"""Poll GitHub Actions runs and their jobs into pipeline_runs (source='gha').

    python -m spyre_clickhouse_ingest.gha_runs poll --repo torch-spyre/torch-spyre --hours 8
    python -m spyre_clickhouse_ingest.gha_runs poll --repo ... --days 90 --wait-for-reset

Each poll lists the runs CREATED in the window (the API has no updated-since filter), oldest
first, and writes every attempt of each run plus its jobs, skipping a run whose stored row is
already finished at the same updated_at. So what is stored is always a prefix of the window,
and the window starts at the earlier of now - --hours and the repo's watermark (newest stored
run's created_at) minus --overlap-hours: a poll that stopped early, or an outage, is resumed
from where it left off, back to at most --max-catchup-days. A re-run keeps its run's created_at,
so late attempts of older runs need a periodic longer window. Rows replace by (run, attempt).

The token is shared with other jobs, so a poll stops cleanly (flush, notice, exit 0) when the
rate limit's remaining falls to --reserve of it. --deadline-minutes is split evenly over the
repos still to poll, so one busy repo cannot starve the rest; time a repo leaves unused passes
on. The next poll continues either way. --wait-for-reset sleeps to the reset instead, for a
manual backfill. A token that expires mid-poll (an App installation token lives an hour)
stops it the same way. A poll that finished exits 0; one that stopped early exits 3 (deadline
or token: continue now) or 4 (rate reserve: continue in a later run).
Needs GITHUB_TOKEN.
"""

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from .schema import PipelineRuns

API = "https://api.github.com"
# The runs list stops at 1000 results per query; a fuller window is split until it fits.
LIST_CAP = 1000
PAGE = 100
RETRIES = 6
# Exit status of a poll that stopped early with more to do: at its deadline or an expired token
# (a caller may continue at once), or at the rate reserve (wait for a later run). 0 = done.
EXIT_MORE = 3
EXIT_BUDGET = 4

# Lane names match the Jenkins trigger_source values, so one gate view covers both systems.
# pull_request is the PR-validation lane, which Jenkins calls spyre-test.
EVENT_LANES = {
    "merge_group": "merge-queue",
    "pull_request": "spyre-test",
    "pull_request_target": "spyre-test",
    "schedule": "scheduled",
    "workflow_dispatch": "dispatch",
    "workflow_run": "chained",
}
MAIN_BRANCHES = ("main", "master")
ARCH_LABELS = (("s390x", "s390x"), ("ppc64le", "ppc64le"), ("arm64", "aarch64"))


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _ms(start: datetime | None, end: datetime | None) -> int:
    return max(0, int((end - start).total_seconds() * 1000)) if start and end else 0


def lane(run: Mapping[str, Any]) -> str:
    """The CI lane a run serves, from its event (and branch, for a push)."""
    event = run.get("event") or ""
    if event == "push":
        return "main-push" if run.get("head_branch") in MAIN_BRANCHES else "push"
    return EVENT_LANES.get(event, event)


def pr_number(run: Mapping[str, Any]) -> int:
    """The PR a run is for: its pull_requests link, or the merge-queue branch name."""
    prs = run.get("pull_requests") or []
    if prs:
        return int(prs[0].get("number") or 0)
    # A merge-queue run's branch is gh-readonly-queue/<base>/pr-<n>-<sha>.
    tail = (run.get("head_branch") or "").rsplit("/pr-", 1)
    num = tail[1].split("-", 1)[0] if len(tail) == 2 else ""
    return int(num) if num.isdigit() else 0


def job_arch(labels: list[str]) -> str:
    """Canonical arch of a runner, from its labels; x86_64 unless a label names another."""
    joined = " ".join(labels).lower()
    for needle, arch in ARCH_LABELS:
        if needle in joined:
            return arch
    return "x86_64"


def _blank() -> dict[str, Any]:
    """Every pipeline_runs column at its empty value, so a row names only what it knows."""
    return {
        **{c: "" for c in PipelineRuns.columns},
        "build_number": 0,
        "attempt": 1,
        "ended_at": None,
        **{
            c: 0
            for c in (
                "queue_ms",
                "duration_ms",
                "build_ms",
                "test_ms",
                "pr_number",
                "nodes_built",
                "nodes_reused",
                "nodes_dropped",
                "tests_total",
                "tests_failed",
                "ch_write_failures",
            )
        },
        **{
            c: False
            for c in ("superseded", "reached_normal_completion", "failure_is_infra")
        },
        "arches": [],
        "lane_results": {},
        "props": {},
    }


def _outcome(
    row: dict[str, Any],
    status: str,
    conclusion: str | None,
    start: datetime | None,
    end: datetime | None,
) -> None:
    finished = status == "completed"
    row["state"] = "finished" if finished else "running"
    row["result"] = (conclusion or "").lower() if finished else ""
    row["ended_at"] = end if finished else None
    row["duration_ms"] = _ms(start, end) if finished else 0
    # The one GHA conclusion diagnose_failure.py's taxonomy covers without reading a log.
    if row["result"] == "timed_out":
        row["failure_reason"] = "infra_timeout"
        row["failure_is_infra"] = True


def run_key(repo: str, run_id: int, attempt: int) -> str:
    return f"gha:{repo}/{run_id}#{attempt}"


def workflow_row(
    repo: str, run: Mapping[str, Any], jobs: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """The gha-workflow row for one run attempt."""
    attempt = int(run.get("run_attempt") or 1)
    created, started, updated = (
        _ts(run.get(k)) for k in ("created_at", "run_started_at", "updated_at")
    )
    row = _blank()
    row.update(
        run_key=run_key(repo, run["id"], attempt),
        updated_at=updated or started,
        source="gha",
        pipeline_type="gha-workflow",
        job_name=f"{repo}/{run.get('path') or run.get('name') or ''}",
        build_number=int(run.get("run_number") or 0),
        attempt=attempt,
        build_url=f"{run.get('html_url', '')}/attempts/{attempt}",
        started_at=started or created,
        queue_ms=_ms(created, started) if attempt == 1 else 0,
        trigger_kind=run.get("event") or "",
        trigger_source=lane(run),
        repo=repo.split("/")[-1],
        pr_number=pr_number(run),
        sha=run.get("head_sha") or "",
        component=repo.split("/")[-1],
        arches=sorted({job_arch(j.get("labels") or []) for j in jobs}),
        props={
            k: str(v)
            for k, v in (
                ("workflow", run.get("name")),
                ("head_branch", run.get("head_branch")),
                ("run_id", run.get("id")),
            )
            if v
        },
    )
    # A run has no completed_at; its updated_at is when it concluded.
    _outcome(
        row, run.get("status") or "", run.get("conclusion"), row["started_at"], updated
    )
    failed = [
        j for j in jobs if (j.get("conclusion") or "") in ("failure", "timed_out")
    ]
    if failed and not row["failure_reason"]:
        row["failed_stage"] = failed[0].get("name") or ""
    return row


def job_row(
    repo: str, run: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The gha-job row for one job of a run attempt; None for a job that never started."""
    started = _ts(job.get("started_at"))
    if started is None:
        return None
    attempt = int(job.get("run_attempt") or run.get("run_attempt") or 1)
    parent = run_key(repo, run["id"], attempt)
    completed = _ts(job.get("completed_at"))
    row = _blank()
    row.update(
        run_key=f"{parent}/{job['id']}",
        updated_at=completed or started,
        source="gha",
        pipeline_type="gha-job",
        job_name=f"{repo}/{run.get('path') or run.get('name') or ''}/{job.get('name') or ''}",
        build_number=int(run.get("run_number") or 0),
        attempt=attempt,
        build_url=job.get("html_url") or "",
        agent=job.get("runner_name") or "",
        parent_run_key=parent,
        started_at=started,
        queue_ms=_ms(_ts(job.get("created_at")), started),
        trigger_kind=run.get("event") or "",
        trigger_source=lane(run),
        repo=repo.split("/")[-1],
        pr_number=pr_number(run),
        sha=job.get("head_sha") or run.get("head_sha") or "",
        component=repo.split("/")[-1],
        arches=[job_arch(job.get("labels") or [])],
        props={
            k: str(v)
            for k, v in (
                ("runner_group", job.get("runner_group_name")),
                ("labels", ",".join(job.get("labels") or [])),
            )
            if v
        },
    )
    _outcome(row, job.get("status") or "", job.get("conclusion"), started, completed)
    failed_step = next(
        (
            s
            for s in job.get("steps") or []
            if (s.get("conclusion") or "") in ("failure", "timed_out")
        ),
        None,
    )
    if failed_step:
        row["failed_stage"] = failed_step.get("name") or ""
    return row


class Stop(Exception):
    """The poll must end here, cleanly: the budget, the deadline or the token is used up."""


class GitHub:
    """The few Actions API reads the poller needs, over stdlib HTTP, within a rate budget."""

    def __init__(
        self,
        token: str,
        reserve: float = 0.5,
        deadline: float | None = None,
        wait_for_reset: bool = False,
    ):
        self.token = token
        self.reserve = reserve
        self.deadline = deadline
        self.wait_for_reset = wait_for_reset
        self.limit = self.remaining = self.reset = None
        self.requests, self.first = 0, ""
        self.authed = self.expired = False

    def get(self, path: str, **query: Any) -> dict[str, Any]:
        url = f"{API}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        for attempt in range(RETRIES):
            self._guard()
            self.requests += 1
            try:
                body, headers = self._fetch(req)
                self._note(headers)
                self.authed = True
                return body
            except urllib.error.HTTPError as err:
                # An installation token lives an hour, so a long poll outlives it; a 401
                # before any success is a bad credential and stays an error.
                if err.code == 401 and self.authed:
                    self.expired = True
                    raise Stop("token expired") from err
                self._note(err.headers)
                wait = self._backoff(err, attempt)
                if wait is None:
                    raise
            # A truncated body or dropped connection is transient, like a 5xx.
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
            ):
                wait = 2**attempt
            self._sleep(wait)
        raise SystemExit(
            f"[error] GitHub API still failing after {RETRIES} tries: {url}"
        )

    @staticmethod
    def _fetch(req) -> tuple[dict[str, Any], Mapping[str, str]]:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp), resp.headers

    def _note(self, headers) -> None:
        """Track the rate limit from a response's headers."""
        if headers is None or headers.get("X-RateLimit-Limit") is None:
            return
        self.limit = int(headers["X-RateLimit-Limit"])
        self.remaining = int(headers.get("X-RateLimit-Remaining", self.limit))
        self.reset = float(headers.get("X-RateLimit-Reset", 0))
        self.first = self.first or self.rate()

    def rate(self) -> str:
        """remaining/limit as the last response reported it; GET /rate_limit lags behind."""
        return f"{self.remaining}/{self.limit}"

    def exhausted(self) -> bool:
        """True once a request would dip into the reserve."""
        return self.limit is not None and self.remaining <= self.reserve * self.limit

    def _guard(self) -> None:
        """Stop, or wait for the reset, before a request that would dip into the reserve."""
        if self.deadline is not None and time.time() >= self.deadline:
            raise Stop("deadline reached")
        if not self.exhausted():
            return
        if not self.wait_for_reset:
            raise Stop(f"budget reached at {self.remaining} remaining")
        self._sleep(max(1.0, (self.reset or 0) - time.time() + 1))
        self.remaining = self.limit

    def _sleep(self, seconds: float) -> None:
        if self.deadline is not None and time.time() + seconds > self.deadline:
            raise Stop("deadline reached")
        time.sleep(seconds)

    @staticmethod
    def _backoff(err, attempt: int) -> float | None:
        """Seconds to wait before retrying, or None for an error retrying cannot fix."""
        if err.code in (403, 429) and err.headers.get("X-RateLimit-Remaining") == "0":
            return max(
                1.0, float(err.headers.get("X-RateLimit-Reset", "0")) - time.time() + 1
            )
        if err.code == 429 or err.code >= 500:
            return float(err.headers.get("Retry-After") or 2**attempt)
        return None

    def runs(
        self, repo: str, start: datetime, end: datetime
    ) -> Iterator[dict[str, Any]]:
        """Every run created in [start, end], oldest first, listed one sub-window at a time.

        A window over the list cap is split, older half first; each part is listed whole and
        sorted, so a poll stopped mid-window has paid only for the part it was in.
        """
        created = f"{start:%Y-%m-%dT%H:%M:%SZ}..{end:%Y-%m-%dT%H:%M:%SZ}"
        first = self.get(
            f"/repos/{repo}/actions/runs", created=created, per_page=PAGE, page=1
        )
        if first.get("total_count", 0) > LIST_CAP and end - start > timedelta(
            minutes=1
        ):
            mid = start + (end - start) / 2
            yield from self.runs(repo, start, mid)
            yield from self.runs(repo, mid + timedelta(seconds=1), end)
            return
        page, batch, listed = 1, first, []
        while batch.get("workflow_runs"):
            listed += batch["workflow_runs"]
            page += 1
            batch = self.get(
                f"/repos/{repo}/actions/runs", created=created, per_page=PAGE, page=page
            )
        yield from sorted(listed, key=lambda r: (r.get("created_at") or "", r["id"]))

    def attempt(self, repo: str, run_id: int, attempt: int) -> dict[str, Any]:
        return self.get(f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}")

    def jobs(self, repo: str, run_id: int, attempt: int) -> list[dict[str, Any]]:
        out, page = [], 1
        while True:
            batch = self.get(
                f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs",
                per_page=PAGE,
                page=page,
            )
            out += batch.get("jobs") or []
            if len(out) >= batch.get("total_count", 0) or not batch.get("jobs"):
                return out
            page += 1


def stored(
    client, db: str, repo: str, since: datetime
) -> dict[str, tuple[datetime, str]]:
    """run_key -> (updated_at, state) of the gha-workflow rows already written for repo."""
    res = client.query(
        f"SELECT run_key, updated_at, state FROM {db}.pipeline_runs FINAL "
        "WHERE source = 'gha' AND pipeline_type = 'gha-workflow' "
        "AND repo = {repo:String} AND started_at >= {since:DateTime64(3)}",
        parameters={"repo": repo.split("/")[-1], "since": since},
    )
    return {
        k: (u.replace(tzinfo=timezone.utc) if u.tzinfo is None else u, s)
        for k, u, s in res.result_rows
    }


def watermark(client, db: str, repo: str) -> datetime | None:
    """created_at of the newest run stored for repo, or None when it has none.

    An attempt-1 row's started_at - queue_ms is its run's created_at, the key runs are
    listed and processed in.
    """
    n, newest = client.query(
        f"SELECT count(), max(started_at - toIntervalMillisecond(queue_ms)) "
        f"FROM {db}.pipeline_runs "
        "WHERE source = 'gha' AND pipeline_type = 'gha-workflow' "
        "AND startsWith(run_key, {prefix:String}) AND endsWith(run_key, '#1')",
        parameters={"prefix": f"gha:{repo}/"},
    ).result_rows[0]
    if not n:
        return None
    return newest.replace(tzinfo=timezone.utc) if newest.tzinfo is None else newest


def window_start(
    now: datetime,
    base: timedelta,
    mark: datetime | None,
    overlap: timedelta,
    max_catchup: timedelta,
) -> tuple[datetime, bool]:
    """(start, from_watermark): now - base, extended back to mark - overlap after a gap.

    The extension stops at now - max_catchup; an explicit longer base is a backfill and wins.
    """
    start = now - base
    if mark is None:
        return start, False
    resume = max(mark - overlap, now - max_catchup)
    return (resume, True) if resume < start else (start, False)


def poll(
    gh: GitHub,
    client,
    db: str | None,
    repo: str,
    start: datetime,
    end: datetime,
    dry_run: bool = False,
) -> tuple[int, int, str]:
    """Write every new or changed run attempt created in the window, oldest first.

    Returns (attempts, rows, stop reason or ''). A stop keeps every attempt completed so far,
    so what is stored stays a prefix of the window.
    """
    known = (
        {}
        if client is None
        else stored(client, db or "", repo, start - timedelta(days=1))
    )
    attempts = rows = 0
    reason = ""
    batch: list[dict[str, Any]] = []
    try:
        for latest in gh.runs(repo, start, end):
            n = int(latest.get("run_attempt") or 1)
            for a in range(1, n + 1):
                key = run_key(repo, latest["id"], a)
                have = known.get(key)
                run = latest if a == n else None
                if (
                    have
                    and have[1] == "finished"
                    and (a < n or have[0] == _ts(latest.get("updated_at")))
                ):
                    continue
                run = run or gh.attempt(repo, latest["id"], a)
                jobs = gh.jobs(repo, latest["id"], a)
                batch.append(workflow_row(repo, run, jobs))
                batch += [r for r in (job_row(repo, run, j) for j in jobs) if r]
                attempts += 1
            if len(batch) >= 500:
                rows += _flush(client, db, batch, dry_run)
    except Stop as stop:
        reason = str(stop)
    rows += _flush(client, db, batch, dry_run)
    return attempts, rows, reason


def _flush(client, db: str | None, batch: list[dict[str, Any]], dry_run: bool) -> int:
    n = len(batch)
    if n and not dry_run:
        PipelineRuns.insert(client, batch, db=db)
    elif n:
        for r in batch:
            print(json.dumps(r, default=str))
    batch.clear()
    return n


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll", help="runs created in the last --hours / --days")
    p.add_argument(
        "--repo", action="append", required=True, help="owner/name; repeatable"
    )
    window = p.add_mutually_exclusive_group()
    window.add_argument("--hours", type=float, default=8.0)
    window.add_argument("--days", type=float)
    p.add_argument(
        "--overlap-hours",
        type=float,
        default=6.0,
        help="re-read this much before the watermark (runs that started late)",
    )
    p.add_argument("--max-catchup-days", type=float, default=90.0)
    p.add_argument(
        "--reserve",
        type=float,
        default=0.5,
        help="fraction of the rate limit to leave for other jobs",
    )
    p.add_argument(
        "--deadline-minutes", type=float, default=12.0, help="0 = no deadline"
    )
    p.add_argument(
        "--wait-for-reset",
        action="store_true",
        help="at the reserve, sleep to the rate-limit reset instead of stopping",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print rows instead of writing"
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("[error] GITHUB_TOKEN is unset")
    now = datetime.now(timezone.utc)
    base = timedelta(days=args.days) if args.days else timedelta(hours=args.hours)

    client, db = None, None
    if not args.dry_run:
        from .client import ClickHouse, ClickHouseEnv

        db = args.database or ClickHouseEnv.target_database()
        if not db:
            sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
        client = ClickHouse.connect(database=db)
    gh = GitHub(
        token,
        reserve=args.reserve,
        deadline=time.time() + args.deadline_minutes * 60
        if args.deadline_minutes
        else None,
        wait_for_reset=args.wait_for_reset,
    )
    end_all, stopped = gh.deadline, False
    for i, repo in enumerate(args.repo):
        # An even share of the time left; what a repo leaves unused passes on.
        if end_all is not None:
            gh.deadline = time.time() + (end_all - time.time()) / (len(args.repo) - i)
        start, resumed = window_start(
            now,
            base,
            watermark(client, db or "", repo) if client is not None else None,
            timedelta(hours=args.overlap_hours),
            timedelta(days=args.max_catchup_days),
        )
        print(
            f"[info] {repo}: from {start:%Y-%m-%dT%H:%MZ}"
            + (" (watermark - overlap)" if resumed else ""),
            file=sys.stderr,
        )
        attempts, rows, reason = poll(gh, client, db, repo, start, now, args.dry_run)
        print(
            f"[info] {repo}: {attempts} run attempt(s), {rows} row(s)", file=sys.stderr
        )
        if reason:
            stopped = True
            print(f"[notice] {repo}: {reason}; resume next run", file=sys.stderr)
            # The budget and the token are shared by every repo; a deadline share is not.
            if gh.expired or (gh.exhausted() and not gh.wait_for_reset):
                break
    print(
        f"[info] rate limit {gh.first or 'unseen'} at the first response, "
        f"{gh.rate()} at the last, {gh.requests} request(s)",
        file=sys.stderr,
    )
    if stopped:
        sys.exit(EXIT_BUDGET if gh.exhausted() and not gh.wait_for_reset else EXIT_MORE)


if __name__ == "__main__":
    main()
