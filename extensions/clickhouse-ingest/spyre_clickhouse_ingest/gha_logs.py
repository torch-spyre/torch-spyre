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

"""Fetching GitHub Actions job logs through `gh` -- stdlib + `gh` CLI only."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import regex as re

TRANSIENT_HTTP_CODES = ("502", "503", "504")


class GhCli:
    """Probing and invoking the `gh` binary."""

    @staticmethod
    def escape_sequence_flag() -> list:
        """`--allow-escape-sequences` when this `gh` supports it (probed), else []."""
        help_text = subprocess.run(
            ["gh", "api", "--help"], capture_output=True, text=True
        ).stdout
        return (
            ["--allow-escape-sequences"]
            if "--allow-escape-sequences" in help_text
            else []
        )

    @staticmethod
    def run_with_retry(args: list, max_attempts: int = 5, base_delay: int = 2):
        """Run `gh`, retrying transient 5xx with backoff; a 404 returns immediately."""
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


class JobLogs:
    """Listing, deduping and downloading GHA job logs."""

    @staticmethod
    def dedupe(jobs: list) -> list:
        """Drop repeated job ids, so each job's log is downloaded and parsed once."""
        seen: set = set()
        out = []
        for job in jobs:
            if job["id"] not in seen:
                seen.add(job["id"])
                out.append(job)
        if len(out) != len(jobs):
            print(
                f"[warn] jobs list had {len(jobs) - len(out)} duplicate id(s), deduped"
            )
        return out

    @classmethod
    def load(cls, path) -> list:
        """Jobs from a JSONL listing, as written by `gh api ... --jq`."""
        with open(path) as fh:
            return cls.dedupe([json.loads(line) for line in fh if line.strip()])

    @staticmethod
    def download(jobs: list, repo: str, out_dir) -> int:
        """Write each job's log to out_dir; returns how many were fetched."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Resolved via the module global, not GhCli directly, so a caller patching the
        # module-level `escape_sequence_flag` alias (no live `gh`) is honoured.
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

    @classmethod
    def download_or_die(cls, jobs: list, repo: str, out_dir) -> int:
        """As `download`, but zero downloads for a non-empty job list is fatal."""
        downloaded = cls.download(jobs, repo, out_dir)
        if jobs and downloaded == 0:
            print(f"::error::downloaded 0 of {len(jobs)} job logs -- aborting")
            sys.exit(1)
        return downloaded


# Function API, kept so installed consumers import one definition, not a copy.
escape_sequence_flag = GhCli.escape_sequence_flag
run_with_retry = GhCli.run_with_retry
dedupe_jobs = JobLogs.dedupe
load_jobs = JobLogs.load
download_job_logs = JobLogs.download
download_job_logs_or_die = JobLogs.download_or_die
