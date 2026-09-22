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

"""Fetching GitHub Actions job logs and run artifacts through `gh`.

Lived as an inline heredoc in each repo's ingest workflow, where it could not be tested and where
the pre-commit hooks could not see it -- so the retry hardening added to one copy never reached
the checked-in downloader script. One definition, on disk, testable.

stdlib + the `gh` CLI only: this also runs in matrix jobs that install no Python packages.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import regex as re

TRANSIENT_HTTP_CODES = ("502", "503", "504")


def escape_sequence_flag() -> list:
    """`--allow-escape-sequences` when this `gh` understands it, else [].

    gh 2.97.0 began refusing to emit a response body containing terminal escape sequences without
    the flag, and Actions job logs are full of them: without it every download exits 1 with an
    empty body and the ingest is silently skipped. Probed rather than version-pinned because the
    flag does not exist on older gh.
    """
    help_text = subprocess.run(
        ["gh", "api", "--help"], capture_output=True, text=True
    ).stdout
    return (
        ["--allow-escape-sequences"] if "--allow-escape-sequences" in help_text else []
    )


def run_with_retry(args: list, max_attempts: int = 5, base_delay: int = 2):
    """Run `gh`, retrying only transient 5xx with exponential backoff.

    A 404 (a job with no logs) is returned immediately: retrying a permanent answer just burns
    the budget that a real 502 needs.
    """
    result = subprocess.run(args, capture_output=True, env={**os.environ})
    for attempt in range(1, max_attempts + 1):
        if result.returncode == 0:
            return result
        err = result.stderr.decode("utf-8", "replace")
        if (
            not any(code in err for code in TRANSIENT_HTTP_CODES)
            or attempt == max_attempts
        ):
            return result
        delay = base_delay * (2 ** (attempt - 1))
        print(
            f"  [retry] attempt {attempt}/{max_attempts} failed "
            f"(rc={result.returncode}), retrying in {delay}s: {err.strip()}"
        )
        time.sleep(delay)
        result = subprocess.run(args, capture_output=True, env={**os.environ})
    return result


def dedupe_jobs(jobs: list) -> list:
    """Drop repeated job ids, so each job's log is downloaded and parsed once even if the API
    lists it twice."""
    seen: set = set()
    out = []
    for job in jobs:
        if job["id"] not in seen:
            seen.add(job["id"])
            out.append(job)
    if len(out) != len(jobs):
        print(f"[warn] jobs list had {len(jobs) - len(out)} duplicate id(s), deduped")
    return out


def load_jobs(path) -> list:
    """Jobs from a JSONL listing, as written by `gh api ... --jq`."""
    with open(path) as fh:
        return dedupe_jobs([json.loads(line) for line in fh if line.strip()])


def download_job_logs(jobs: list, repo: str, out_dir) -> int:
    """Write each job's log to out_dir; returns how many were fetched."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    esc_flag = escape_sequence_flag()
    downloaded = 0
    for idx, job in enumerate(jobs):
        safe = re.sub(r"[^\w\s\-]", "", job["name"]).strip()
        target = out_path / f"{idx}_{safe}.txt"
        result = run_with_retry(
            ["gh", "api", *esc_flag, f"/repos/{repo}/actions/jobs/{job['id']}/logs"]
        )
        if result.returncode == 0 and result.stdout:
            target.write_bytes(result.stdout)
            downloaded += 1
            print(f"  OK  {target}")
        else:
            err = result.stderr.decode("utf-8", "replace").strip()
            print(f"  SKIP  {job['name']} (rc={result.returncode}): {err}")
    print(f"[info] downloaded {downloaded}/{len(jobs)} job logs")
    return downloaded


def download_job_logs_or_die(jobs: list, repo: str, out_dir) -> int:
    """As download_job_logs, but a total failure is fatal.

    Every completed run has at least one job with a log, so zero downloads means the download
    mechanism broke -- not that the run produced nothing. Failing here is deliberate: the
    empty-data guard downstream would skip the ingest and still report success, which is how one
    such outage stayed green for days.
    """
    downloaded = download_job_logs(jobs, repo, out_dir)
    if jobs and downloaded == 0:
        print(f"::error::downloaded 0 of {len(jobs)} job logs -- aborting")
        sys.exit(1)
    return downloaded
