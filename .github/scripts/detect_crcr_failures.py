#!/usr/bin/env python3
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

"""Detect newly-failing upstream-beta tests on a completed CRCR run.

Reads the merged JUnit XML reports (the ``upstream-beta-reports-<sha>``
artifact of a push-pytorch-dispatch-to-clickhouse run), extracts every
failed/errored testcase with its full failure text, and de-duplicates
against open ``crcr-auto-skip`` PRs (hidden digest markers in their bodies)
so the downstream Claude step only ever triages failures not already
covered by a pending skip PR.

The watched run's conclusion is NOT a signal: upstream-beta tests run with
``mode: validate``, so a run full of failures still completes 'success'.
The JUnit XMLs are the only authoritative per-test source. Totals are
computed by parse_junit_xml_counts.count_testcases -- the same code the
CRCR reporting path uses -- so the counts here cannot drift from what was
reported.

Writes a JSON summary (--output-file) consumed by the notify and
auto-recover jobs, plus ``$GITHUB_OUTPUT`` keys: has_failures,
failure_count, total_tests, new_failure_count, mass_failure, skip_needed.

Requires GH_TOKEN in the environment for ``gh pr list`` and the
third-party ``regex`` module (CI runs this via ``uv run --with regex``).
If ``gh pr list`` fails, the script exits nonzero (fail-closed): without
coverage information, running auto-recover could file a duplicate skip PR,
so the whole triage run is abandoned and retried on the next CRCR run.

Usage (called by the crcr-regression-triage workflow):
    uv run --with regex detect_crcr_failures.py \\
        --xml-dir xml_artifacts --repo torch-spyre/torch-spyre \\
        --run-url URL --matrix-run-url URL --pytorch-sha SHA \\
        --torch-spyre-sha SHA --report-artifact NAME \\
        --output-file failures.json
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import regex as re

# Marker/digest namespace for skip-PR coverage. Deliberately distinct from
# report_test_failures_issue.py's "crcr-test" digests (sha1 over
# "<bucket>|<test_id>"): the two layers dedupe different things (pending
# skip PRs vs. tracking-issue history) and must never collide.
SKIP_MARKER_FMT = "<!-- crcr-skip:{digest} -->"
SKIP_MARKER_RE = re.compile(r"<!-- crcr-skip:([0-9a-f]{12}) -->")
# Full traceback text kept per test; enough to classify the failure without
# blowing up the JSON handed to the LLM step.
MAX_TEXT = 4000

DEFAULT_PR_LABEL = "crcr-auto-skip"
DEFAULT_MAX_FAILURES = 50
DEFAULT_MAX_FAILURE_RATIO = 0.10


def _load_count_testcases():
    """Loads count_testcases from the sibling parse_junit_xml_counts.py.

    Import-by-path (not a package import) so this script reuses exactly the
    counting code the CRCR reporting path runs, with no sys.path games.
    """
    sibling = Path(__file__).resolve().parent / "parse_junit_xml_counts.py"
    spec = importlib.util.spec_from_file_location("parse_junit_xml_counts", sibling)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count_testcases


def skip_digest(test_id: str) -> str:
    return hashlib.sha1(f"crcr-skip|{test_id}".encode()).hexdigest()[:12]


def collect_totals(xml_dir: Path, count_testcases) -> dict:
    totals = {col: 0 for col in (
        "total", "passed", "failed", "errors", "skipped", "xfailed", "xpassed"
    )}  # fmt: skip
    for xml_path in sorted(xml_dir.rglob("*.xml")):
        file_totals = count_testcases(xml_path)
        for col in totals:
            totals[col] += file_totals[col]
    return totals


def collect_failures(xml_dir: Path) -> dict:
    """Returns {test_id: failure record}, first occurrence winning.

    Overlapping bundles (per-suite artifacts plus a merged copy) and
    pytest-rerun repeats can list the same testcase more than once; sorted
    path order + first-wins keeps the result deterministic.
    """
    failures: dict = {}
    for xml_path in sorted(xml_dir.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            print(f"::warning::skipping unparsable {xml_path}: {exc}", file=sys.stderr)
            continue
        suite = xml_path.stem
        for case in root.iter("testcase"):
            # NB: ElementTree elements are falsy when childless -- compare
            # against None, never use them as booleans.
            node = case.find("failure")
            kind = "failure"
            if node is None:
                node = case.find("error")
                kind = "error"
            if node is None:
                continue
            test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
            if test_id in failures:
                continue
            message_lines = (node.get("message") or "").strip().splitlines()
            failures[test_id] = {
                "test_id": test_id,
                "suite": suite,
                "classname": case.get("classname", ""),
                "name": case.get("name", ""),
                "kind": kind,
                "message": (message_lines[0] if message_lines else "")[:300],
                "text": (node.text or "").strip()[:MAX_TEXT],
                "digest": skip_digest(test_id),
            }
    return failures


def open_skip_coverage(repo: str, label: str):
    """Returns (PR info of the newest open skip PR, covered digest set).

    Digests are unioned across every open labeled PR; the PR handed to the
    recovery step for appending is the newest (highest number).
    """
    res = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "-R",
            repo,
            "--state",
            "open",
            "--label",
            label,
            "--limit",
            "10",
            "--json",
            "number,url,headRefName,body",
        ],  # fmt: skip
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(
            f"::error::gh pr list failed ({res.stderr.strip()}); without "
            "skip-PR coverage info, auto-recover is unsafe this run",
            file=sys.stderr,
        )
        sys.exit(1)
    prs = sorted(json.loads(res.stdout or "[]"), key=lambda pr: pr["number"])
    covered = set()
    for pr in prs:
        covered.update(SKIP_MARKER_RE.findall(pr.get("body") or ""))
    newest = prs[-1] if prs else {}
    return (
        {
            "number": newest.get("number", 0),
            "url": newest.get("url", ""),
            "head_branch": newest.get("headRefName", ""),
        },
        covered,
    )


def write_github_output(payload: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for key in (
            "has_failures",
            "failure_count",
            "total_tests",
            "new_failure_count",
            "mass_failure",
            "skip_needed",
        ):
            f.write(f"{key}={str(payload[key]).lower()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--run-url", default="", help="Watched workflow run URL")
    parser.add_argument("--matrix-run-url", default="", help="Matrix run URL")
    parser.add_argument("--pytorch-sha", default="")
    parser.add_argument("--torch-spyre-sha", default="")
    parser.add_argument("--report-artifact", default="")
    parser.add_argument(
        "--pr-label",
        default=DEFAULT_PR_LABEL,
        help="Label of open PRs whose bodies carry skip-coverage digests",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=DEFAULT_MAX_FAILURES,
        help="Above this many failures, treat the run as a mass failure "
        "(infra breakage): notify only, never auto-skip",
    )
    parser.add_argument(
        "--max-failure-ratio",
        type=float,
        default=DEFAULT_MAX_FAILURE_RATIO,
        help="Mass-failure threshold as a fraction of total tests",
    )
    parser.add_argument("--output-file", default="", help="JSON summary path")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    count_testcases = _load_count_testcases()
    totals = collect_totals(xml_dir, count_testcases)
    failures = collect_failures(xml_dir)
    failure_count = len(failures)

    open_skip_pr, covered = open_skip_coverage(args.repo, args.pr_label)
    tests = []
    for test_id in sorted(failures):
        record = dict(failures[test_id])
        record["already_in_skip_pr"] = record["digest"] in covered
        tests.append(record)
    new_failure_count = sum(1 for t in tests if not t["already_in_skip_pr"])

    mass_failure = failure_count > args.max_failures or (
        totals["total"] > 0 and failure_count / totals["total"] > args.max_failure_ratio
    )
    skip_needed = new_failure_count > 0 and not mass_failure

    if failure_count and failure_count != totals["failed"] + totals["errors"]:
        # Duplicate testcase entries (reruns, overlapping bundles) -- the
        # unique list is what gets triaged, so this is informational.
        print(
            f"::warning::{failure_count} unique failing tests vs. "
            f"{totals['failed'] + totals['errors']} failing testcase entries "
            "(reruns/overlapping bundles deduplicated)"
        )

    payload = {
        "context": {
            "repo": args.repo,
            "run_url": args.run_url,
            "matrix_run_url": args.matrix_run_url,
            "pytorch_sha": args.pytorch_sha,
            "torch_spyre_sha": args.torch_spyre_sha,
            "report_artifact": args.report_artifact,
        },
        "totals": totals,
        "mass_failure": mass_failure,
        "open_skip_pr": open_skip_pr,
        "tests": tests,
    }
    outputs = {
        "has_failures": failure_count > 0,
        "failure_count": failure_count,
        "total_tests": totals["total"],
        "new_failure_count": new_failure_count,
        "mass_failure": mass_failure,
        "skip_needed": skip_needed,
    }
    payload["outputs"] = outputs

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(payload, indent=2) + "\n")
    write_github_output(outputs)

    print(f"Test totals: {totals}")
    print(
        f"Failures: {failure_count} ({new_failure_count} not covered by an "
        f"open skip PR); mass_failure={mass_failure}; skip_needed={skip_needed}"
    )
    if open_skip_pr["number"]:
        print(f"Open skip PR: {open_skip_pr['url']}")


if __name__ == "__main__":
    main()
