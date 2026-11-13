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

#!/usr/bin/env python3
"""Open (or update) GitHub issues for failed upstream-beta tests.

Reads the per-suite JUnit XML reports gathered by
download_run_xml_artifacts.py, groups every failed/errored testcase into
buckets (by suite family, see BUCKETS), and files ONE issue per bucket that
tags that bucket's maintainers (resolved from CODEOWNERS) and links to the run, the dispatched test
matrix run, and the log/report artifacts.

De-duplication: the workflow is triggered by CRCR, so an unfixed regression
re-fails on every dispatch. Each bucket owns at most one *open* issue, found
by a hidden marker and a ``crcr-bucket:<name>`` label. On a later run:
  - no new failures vs. the open issue's list -> nothing is posted;
  - some new failures                         -> one comment with only those;
  - no open issue (never filed, or closed)    -> a new issue is created.
Failures already recorded in the issue (body or earlier comments) are never
reported twice.

Requires GH_TOKEN with ``issues: write``; uses the ``gh`` CLI.

Usage:
    python3 report_test_failures_issue.py \
        --xml-dir xml_artifacts --repo torch-spyre/torch-spyre \
        --run-url URL --matrix-run-url URL --pytorch-sha SHA \
        --torch-spyre-sha SHA --report-artifact NAME --codeowners CODEOWNERS [--dry-run]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import regex as re

BASE_LABEL = "crcr-regression"
MARKER_FMT = "<!-- crcr-bucket:{bucket} -->"
TEST_MARKER_FMT = "<!-- crcr-test:{digest} -->"
# GitHub rejects bodies/comments over 65536 chars; stay well under.
MAX_BODY = 60000
MAX_MESSAGE = 300

# (bucket name, title, report-name prefix, repo path the suite exercises).
# First matching prefix wins. Owners are NOT listed here: they are resolved
# from CODEOWNERS for the representative path (see resolve_owners), so they
# follow CODEOWNERS changes automatically.
BUCKETS = [
    ("inductor", "Inductor", "inductor-", "torch_spyre/_inductor/__init__.py"),
    ("profiler", "Profiler", "profiler-", "torch_spyre/profiler/__init__.py"),
    ("nn", "torch.nn", "nn-", "torch_spyre/ops/__init__.py"),
    ("optim", "torch.optim", "optim-", "torch_spyre/ops/__init__.py"),
]
# Everything else (test_indexing, test_linalg, test_torch, ...): eager ops.
DEFAULT_BUCKET = (
    "eager-ops",
    "Eager ops (test_*.py)",
    "",
    "torch_spyre/ops/__init__.py",
)


def _pattern_regex(pattern):
    """Compiles a CODEOWNERS (gitignore-style) pattern to a regex.

    Matches the path itself or anything beneath it, so a directory pattern
    owns every file inside it.
    """
    dir_only = pattern.endswith("/")
    pat = pattern.strip("/")
    anchored = pattern.startswith("/") or "/" in pat
    out = []
    i = 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    body = "".join(out)
    prefix = "" if anchored else "(?:.*/)?"
    tail = "/.+" if dir_only else "(?:/.*)?"
    return re.compile(f"^{prefix}{body}{tail}$")


def resolve_owners(codeowners_path, path):
    """Returns the owners of `path` per CODEOWNERS (last matching rule wins)."""
    try:
        lines = Path(codeowners_path).read_text().splitlines()
    except OSError as exc:
        print(f"::warning::cannot read {codeowners_path}: {exc}", file=sys.stderr)
        return []
    owners = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        pattern, *rule_owners = line.split()
        if _pattern_regex(pattern).match(path):
            owners = rule_owners
    return owners


def run_gh(args, check=True):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


def bucket_for(report_name):
    for bucket in BUCKETS:
        if report_name.startswith(bucket[2]):
            return bucket
    return DEFAULT_BUCKET


def collect_failures(xml_dir):
    """Returns {bucket_name: {test_id: (suite, kind, message)}} plus bucket meta."""
    failures = defaultdict(dict)
    meta = {}
    for xml_path in sorted(Path(xml_dir).rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            print(f"::warning::skipping unparsable {xml_path}: {exc}", file=sys.stderr)
            continue
        suite = xml_path.stem
        bucket = bucket_for(suite)
        for case in root.iter("testcase"):
            node = case.find("failure")
            kind = "failure"
            if node is None:
                node = case.find("error")
                kind = "error"
            if node is None:
                continue
            test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
            msg = (node.get("message") or node.text or "").strip().splitlines()
            msg = (msg[0] if msg else "")[:MAX_MESSAGE]
            failures[bucket[0]][test_id] = (suite, kind, msg)
            meta[bucket[0]] = bucket
    return failures, meta


def digest(bucket, test_id):
    return hashlib.sha1(f"{bucket}|{test_id}".encode()).hexdigest()[:12]


def md_escape_cell(text):
    return text.replace("|", "\\|").replace("`", "'")


def render_table(bucket, tests, budget):
    rows = ["| Test | Suite | Kind | Message |", "|---|---|---|---|"]
    shown = 0
    for test_id, (suite, kind, msg) in sorted(tests.items()):
        row = (
            f"| `{test_id}` {TEST_MARKER_FMT.format(digest=digest(bucket, test_id))}"
            f" | {suite} | {kind} | {md_escape_cell(msg)} |"
        )
        if sum(len(r) + 1 for r in rows) + len(row) > budget:
            break
        rows.append(row)
        shown += 1
    out = "\n".join(rows)
    if shown < len(tests):
        out += (
            f"\n\n_{len(tests) - shown} more failing test(s) omitted; "
            "see the attached JUnit reports._"
        )
    return out


def links_section(args):
    lines = [
        f"- Workflow run (logs, `Download log archive` under the run's gear menu): {args.run_url}",
    ]
    if args.matrix_run_url:
        lines.append(
            f"- Dispatched upstream-beta matrix run (per-suite job logs): {args.matrix_run_url}"
        )
        lines.append(
            f"- Per-suite JUnit XML artifacts: {args.matrix_run_url}#artifacts"
        )
    lines.append(
        f"- Merged JUnit reports artifact `{args.report_artifact}`: {args.run_url}#artifacts"
    )
    run_id = args.run_url.rstrip("/").rsplit("/", 1)[-1]
    lines.append(
        f"- CLI: `gh run download {run_id} -R {args.repo} -n {args.report_artifact}` "
        f"/ `gh run view {run_id} -R {args.repo} --log-failed`"
    )
    return "\n".join(lines)


def run_info(args):
    return (
        f"- PyTorch commit: [`{args.pytorch_sha}`]"
        f"(https://github.com/pytorch/pytorch/commit/{args.pytorch_sha})\n"
        f"- torch-spyre commit: [`{args.torch_spyre_sha}`]"
        f"(https://github.com/{args.repo}/commit/{args.torch_spyre_sha})"
    )


def find_open_issue(repo, bucket):
    res = run_gh(
        [
            "issue", "list", "-R", repo, "--state", "open",
            "--label", f"crcr-bucket:{bucket}", "--limit", "5",
            "--json", "number,body,url",
        ]
    )  # fmt: skip
    for issue in json.loads(res.stdout or "[]"):
        if MARKER_FMT.format(bucket=bucket) in (issue.get("body") or ""):
            return issue
    return None


def known_digests(repo, issue):
    """Digests already reported in the issue body or any comment."""
    res = run_gh(
        ["issue", "view", str(issue["number"]), "-R", repo, "--json", "body,comments"]
    )
    data = json.loads(res.stdout)
    text = data.get("body") or ""
    for comment in data.get("comments", []):
        text += "\n" + (comment.get("body") or "")
    return set(re.findall(r"<!-- crcr-test:([0-9a-f]{12}) -->", text))


def ensure_labels(repo, bucket):
    for name, color in ((BASE_LABEL, "d93f0b"), (f"crcr-bucket:{bucket}", "fbca04")):
        run_gh(["label", "create", name, "-R", repo, "--color", color, "--force"],
               check=False)  # fmt: skip


def create_issue(args, bucket, title, owners, tests):
    head = (
        f"{MARKER_FMT.format(bucket=bucket)}\n"
        f"cc {' '.join(owners)}\n\n"
        f"The CRCR-triggered upstream-beta run reported **{len(tests)} failing "
        f"test(s)** in the **{title}** bucket.\n\n"
        f"### Run info\n{run_info(args)}\n\n### Logs and reports\n{links_section(args)}\n\n"
        "### Failing tests\n"
    )
    foot = (
        "\n\n---\n_Filed automatically by `push-pytorch-dispatch-to-clickhouse`. "
        "While this issue stays open, later runs only comment with **new** "
        "failures; close it once fixed (or excluded in "
        "`tests/configs/upstream_tests_beta/`)._"
    )
    body = head + render_table(bucket, tests, MAX_BODY - len(head) - len(foot)) + foot
    cmd = ["issue", "create", "-R", args.repo,
           "--title", f"[CRCR] {title}: {len(tests)} upstream-beta test failure(s) "
                      f"@ pytorch {args.pytorch_sha[:10]}",
           "--body", body,
           "--label", BASE_LABEL, "--label", f"crcr-bucket:{bucket}"]  # fmt: skip
    return run_gh(cmd).stdout.strip()


def comment_new(args, issue, owners, new_tests, bucket):
    head = (
        f"{' '.join(owners)} — **{len(new_tests)} new failing test(s)** in a later "
        "CRCR run (earlier failures in this issue are not repeated).\n\n"
        f"{run_info(args)}\n\n{links_section(args)}\n\n"
    )
    body = head + render_table(bucket, new_tests, MAX_BODY - len(head))
    run_gh(["issue", "comment", str(issue["number"]), "-R", args.repo, "--body", body])
    return issue["url"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--run-url", required=True)
    p.add_argument("--matrix-run-url", default="")
    p.add_argument("--pytorch-sha", required=True)
    p.add_argument("--torch-spyre-sha", required=True)
    p.add_argument("--report-artifact", required=True)
    p.add_argument(
        "--codeowners",
        default="CODEOWNERS",
        help="Path to the CODEOWNERS file used to tag each bucket's maintainers",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    failures, meta = collect_failures(args.xml_dir)
    if not failures:
        print("No failed tests; nothing to report.")
        return

    for bucket, tests in sorted(failures.items()):
        _, title, _, owner_path = meta[bucket]
        owners = resolve_owners(args.codeowners, owner_path)
        if args.dry_run:
            print(f"[dry-run] {bucket}: {len(tests)} failures, owners {owners}")
            continue
        ensure_labels(args.repo, bucket)
        issue = find_open_issue(args.repo, bucket)
        if issue is None:
            print(
                f"{bucket}: created {create_issue(args, bucket, title, owners, tests)}"
            )
            continue
        seen = known_digests(args.repo, issue)
        new = {t: v for t, v in tests.items() if digest(bucket, t) not in seen}
        if not new:
            print(f"{bucket}: all {len(tests)} failures already in {issue['url']}")
        else:
            print(
                f"{bucket}: commented on {comment_new(args, issue, owners, new, bucket)}"
            )


if __name__ == "__main__":
    main()
