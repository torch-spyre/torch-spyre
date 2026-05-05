# Copyright 2025 The Torch-Spyre Authors.
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

"""List open PRs grouped by CODEOWNERS area.

Usage:
    python tools/list_prs_by_codeowner.py [--repo OWNER/REPO] [--limit N]

Requires the `gh` CLI to be authenticated.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def load_codeowners(repo_root: Path) -> list[tuple[str, list[str]]]:
    """Parse CODEOWNERS into an ordered list of (pattern, owners) pairs."""
    path = repo_root / "CODEOWNERS"
    rules: list[tuple[str, list[str]]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern, owners = parts[0], parts[1:]
        rules.append((pattern, owners))
    return rules


def owners_for_file(path: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    """Return owners for a file path using last-match-wins CODEOWNERS semantics."""
    matched: list[str] = []
    for pattern, owners in rules:
        # Strip leading slash for matching
        pat = pattern.lstrip("/")
        # Directory patterns (ending with /) match everything beneath
        if pattern.endswith("/"):
            if fnmatch.fnmatch(path, pat + "*") or path.startswith(pat):
                matched = owners
        elif fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat + "/*"):
            matched = owners
        elif pat == "*":
            matched = owners
    return matched


def area_label(pattern: str, owners: list[str]) -> str:
    """Derive a human-readable area name from a CODEOWNERS pattern."""
    pat = pattern.strip("/").rstrip("/").rstrip("*")
    if pat in ("", "*"):
        return "Default / Build / Infrastructure"
    return pat or pattern


def dominant_area(
    files: list[str],
    rules: list[tuple[str, list[str]]],
) -> tuple[str, list[str]]:
    """Return the (area_label, owners) that owns the most files in this PR."""
    area_counts: dict[str, int] = defaultdict(int)
    area_owners: dict[str, list[str]] = {}

    for f in files:
        for pattern, owners in reversed(rules):
            pat = pattern.lstrip("/")
            matched = False
            if pattern.endswith("/"):
                matched = f.startswith(pat) or fnmatch.fnmatch(f, pat + "*")
            elif fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(f, pat + "/*"):
                matched = True
            elif pat == "*":
                matched = True
            if matched:
                label = area_label(pattern, owners)
                area_counts[label] += 1
                area_owners[label] = owners
                break

    if not area_counts:
        default_owners = rules[0][1] if rules else []
        return ("Default / Build / Infrastructure", default_owners)

    best = max(area_counts, key=lambda k: area_counts[k])
    return best, area_owners[best]


def fetch_prs(repo: str, limit: int) -> list[dict]:
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--json",
        "number,title,author,files,url",
        "--limit",
        str(limit),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def format_markdown(
    grouped: dict[str, list[dict]], area_owners: dict[str, list[str]]
) -> str:
    lines: list[str] = []
    for area in sorted(grouped):
        owners_str = " ".join(area_owners.get(area, []))
        lines.append(f"### {area}")
        if owners_str:
            lines.append(f"*Owners: {owners_str}*")
        lines.append("")
        lines.append("| # | Title | Author |")
        lines.append("|---|---|---|")
        for pr in sorted(grouped[area], key=lambda p: p["number"], reverse=True):
            num = pr["number"]
            title = pr["title"]
            url = pr["url"]
            author = pr["author"]["login"]
            lines.append(f"| [#{num}]({url}) | {title} | {author} |")
        lines.append("")
    return "\n".join(lines)


def format_text(
    grouped: dict[str, list[dict]], area_owners: dict[str, list[str]]
) -> str:
    lines: list[str] = []
    for area in sorted(grouped):
        owners_str = " ".join(area_owners.get(area, []))
        lines.append(f"{'=' * 60}")
        lines.append(f"{area}")
        if owners_str:
            lines.append(f"Owners: {owners_str}")
        lines.append("")
        for pr in sorted(grouped[area], key=lambda p: p["number"], reverse=True):
            author = pr["author"]["login"]
            lines.append(f"  #{pr['number']:5d}  [{author}]  {pr['title']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="",
        help="GitHub repo as OWNER/REPO (default: detected from git remote)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of PRs to fetch (default: 100)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--codeowners",
        default="",
        help="Path to CODEOWNERS file (default: auto-detect from git root)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="",
        help="Filter to PRs that touch files under this repo path (e.g. torch_spyre/_inductor/)",
    )
    args = parser.parse_args()

    # Resolve repo root for CODEOWNERS
    if args.codeowners:
        rules = load_codeowners(Path(args.codeowners).parent)
    else:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_root = Path(result.stdout.strip())
        rules = load_codeowners(repo_root)

    # Resolve GitHub repo slug
    repo = args.repo
    if not repo:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            capture_output=True,
            text=True,
            check=True,
        )
        repo = result.stdout.strip()

    print(f"Fetching open PRs from {repo}...", file=sys.stderr)
    prs = fetch_prs(repo, args.limit)
    print(f"  {len(prs)} open PRs found", file=sys.stderr)

    if args.path:
        filter_prefix = args.path.lstrip("/").rstrip("/") + "/"
        # Also match exact file names at the given path (no trailing slash needed)
        filter_exact = args.path.lstrip("/").rstrip("/")
        prs = [
            pr
            for pr in prs
            if any(
                f["path"].startswith(filter_prefix) or f["path"] == filter_exact
                for f in pr.get("files", [])
            )
        ]
        print(f"  {len(prs)} PRs touch '{args.path}'", file=sys.stderr)

    grouped: dict[str, list[dict]] = defaultdict(list)
    area_owners_map: dict[str, list[str]] = {}

    for pr in prs:
        file_paths = [f["path"] for f in pr.get("files", [])]
        area, owners = dominant_area(file_paths, rules)
        grouped[area].append(pr)
        area_owners_map[area] = owners

    if args.format == "markdown":
        print(format_markdown(grouped, area_owners_map))
    else:
        print(format_text(grouped, area_owners_map))


if __name__ == "__main__":
    main()
