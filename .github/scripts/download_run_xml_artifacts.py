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
from pathlib import Path


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
        result = subprocess.run(
            ["gh", "api", f"/repos/{args.repo}/actions/artifacts/{artifact['id']}/zip"],
            capture_output=True,
            check=True,
        )
        zip_path.write_bytes(result.stdout)
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)], check=True
        )

    print(f"Downloaded {len(artifacts)} artifact(s).")


if __name__ == "__main__":
    main()
