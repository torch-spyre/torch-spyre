#!/usr/bin/env python3
"""
Downloads every GitHub Actions artifact whose name ends in .xml from a
workflow run into a local directory (via `gh api` + unzip).

Used by build_test_pytorch_source.yaml's `report` job to gather the
per-suite JUnit XML reports the upstream-beta test matrix uploaded (one
artifact per suite, instead of a single combined report).

Requires GH_TOKEN (or GITHUB_TOKEN) in the environment for `gh api`.

Usage (called by the GHA workflow):
    python3 download_run_xml_artifacts.py \
        --repo "$GITHUB_REPOSITORY" \
        --run-id "$GITHUB_RUN_ID" \
        --output-dir xml_artifacts
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Deliberately stdlib-only, with no spyre_clickhouse_ingest import: the calling job
# (build_test_pytorch_source.yaml's report job) runs bare `python3` with no venv and installs no
# packages, so importing the shared library here would break it.
TRANSIENT_HTTP_CODES = ("502", "503", "504")


def _gh_with_retry(args: list, max_attempts: int = 5, base_delay: int = 2):
    """Retry only transient 5xx: a 502 on one artifact used to abort the whole report job, while
    retrying a permanent 404 just burns the budget a real 502 needs."""
    result = subprocess.run(args, capture_output=True)
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
            f"[retry] attempt {attempt}/{max_attempts}, retrying in {delay}s",
            file=sys.stderr,
        )
        time.sleep(delay)
        result = subprocess.run(args, capture_output=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--run-id", required=True, help="Workflow run ID")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to unzip artifacts into"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    listing = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{args.repo}/actions/runs/{args.run_id}/artifacts",
            "--jq",
            '.artifacts[] | select(.name | endswith(".xml")) | {id: .id, name: .name}',
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    artifacts = [
        json.loads(line) for line in listing.stdout.splitlines() if line.strip()
    ]

    for artifact in artifacts:
        zip_path = out_dir / f"{artifact['name']}.zip"
        result = _gh_with_retry(
            ["gh", "api", f"/repos/{args.repo}/actions/artifacts/{artifact['id']}/zip"]
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", "replace").strip()
            print(
                f"::error::could not download {artifact['name']}: {err}",
                file=sys.stderr,
            )
            sys.exit(1)
        zip_path.write_bytes(result.stdout)
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)], check=True
        )

    print(f"Downloaded {len(artifacts)} artifact(s).")


if __name__ == "__main__":
    main()
