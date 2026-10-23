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

"""Record a failed GitHub Actions run as one incident_events row. Fail-open: never exits non-zero."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import regex as re

from .gha_runs import GitHub
from .schema import IncidentEvents

FAILED = ("failure", "timed_out")
MAX_JOBS, MAX_STEPS, MAX_SUMMARY, MAX_PAGES = 5, 3, 500, 3
# Anything shaped like a credential, in any text that reaches a row.
SECRET = re.compile(
    r"(?i)(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+"
    r"|(?:bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{12,}|://[^/\s:@]+:[^/\s@]+@)"
)


def clean(text: Any, limit: int = 120) -> str:
    """A job or step name made safe for a row: no control characters, no secret shapes, capped."""
    out = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()
    return SECRET.sub("***", out)[:limit]


def failed_jobs(jobs: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """(job name, failed step names) for each job that did not succeed; names only, never log text."""
    out = []
    for job in jobs:
        if job.get("conclusion") not in FAILED:
            continue
        steps = [
            clean(s.get("name"))
            for s in job.get("steps") or []
            if s.get("conclusion") in FAILED
        ]
        out.append((clean(job.get("name")), steps[:MAX_STEPS]))
    return out


def build_row(
    repo: str,
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """The incident_events row for a failed run, or None when the run did not fail."""
    conclusion = run.get("conclusion")
    if run.get("status") != "completed" or conclusion not in FAILED:
        return None
    workflow = clean(run.get("name") or "workflow")
    branch = clean(run.get("head_branch"), 80)
    failing = failed_jobs(jobs)
    names = [n for n, _ in failing[:MAX_JOBS]]
    steps = sorted({s for _, ss in failing[:MAX_JOBS] for s in ss})[:MAX_STEPS]
    summary = f"{workflow} {conclusion} on {branch}"
    if names:
        summary += f"; failed jobs: {', '.join(names)}"
    if steps:
        summary += f"; failed steps: {', '.join(steps)}"
    run_id = str(run.get("id", ""))
    url = f"https://github.com/{repo}/actions/runs/{run_id}"
    return {
        "ts": now or datetime.now(timezone.utc),
        "source": "gha",
        "subject": f"{repo}/{workflow}"[:200],
        # A push to main means main is red; a scheduled or manual run is a warning.
        "severity": "critical" if run.get("event") == "push" else "warning",
        "signature": "",
        "cause_class": "unknown",
        "summary": summary[:MAX_SUMMARY],
        "evidence_url": url,
        "dedup_key": "|".join([repo, workflow, branch, names[0] if names else ""]),
        "incident_id": "",
        "job_run": url,
        "props": {
            "run_id": run_id,
            "run_attempt": str(run.get("run_attempt", 1)),
            "event": clean(run.get("event"), 40),
            "conclusion": str(conclusion),
            "head_sha": clean(run.get("head_sha"), 12),
        },
    }


def read_jobs(gh: Any, repo: str, run_id: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        body = gh.get(
            f"/repos/{repo}/actions/runs/{run_id}/jobs",
            per_page=100,
            page=page,
            filter="latest",
        )
        got = body.get("jobs") or []
        jobs += got
        if len(got) < 100:
            break
    return jobs


def record(gh: Any, repo: str, run_id: int, client: Any, db: str, dry_run: bool) -> str:
    """One line saying what happened; the run is read from the API, not from the event payload."""
    run = gh.get(f"/repos/{repo}/actions/runs/{run_id}")
    row = build_row(repo, run, read_jobs(gh, repo, run_id) if run else [])
    if row is None:
        return f"run {run_id} did not fail ({run.get('conclusion')}); nothing recorded"
    values = IncidentEvents.row(row)
    if dry_run:
        return json.dumps(dict(zip(IncidentEvents.columns, values)), default=str)
    if not IncidentEvents.present(client, db):
        return f"{db} has no incident_events table yet; nothing recorded"
    IncidentEvents.insert(client, [row], db=db)
    return f"recorded {row['subject']} ({row['severity']}) in {db}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gha-failure", help="one failed workflow run")
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            raise RuntimeError("GITHUB_TOKEN is unset")
        client, db = None, ""
        if not args.dry_run:
            from .client import ClickHouse, ClickHouseEnv

            db = args.database or ClickHouseEnv.target_database()
            if not db:
                raise RuntimeError("no v2 database (CLICKHOUSE_DB_V2 is unset)")
            client = ClickHouse.connect(database=db)
        print(
            "[incidents] "
            + record(GitHub(token), args.repo, args.run_id, client, db, args.dry_run)
        )
    except (Exception, SystemExit) as err:  # recording must never fail the caller
        print(f"[incidents] not recorded: {clean(f'{type(err).__name__}: {err}', 200)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
