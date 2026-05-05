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

"""List open issues with a given label from torch-spyre/torch-spyre.

Usage:
  list_inductor_issues.py <label> [--split-label LABEL]

Arguments:
  label               Primary label to filter issues (e.g. "inductor")
  --split-label LABEL Optional label used to split the table into three
                      sections: epics, non-epics without split-label,
                      non-epics with split-label.  If omitted, a single
                      table is printed.

Sort order within each section: earliest month tag, then issue number.

Requires: gh CLI authenticated, or GH_TOKEN set.
"""

import argparse
import json
import regex as re
import subprocess
import sys


REPO = "torch-spyre/torch-spyre"

_MONTH_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)(\d{4})$"
)

_MONTH_ORDER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def month_label_sort_key(label: str) -> tuple[int, int]:
    """Return (year, month_number) for a month label, for chronological sorting."""
    m = _MONTH_RE.match(label)
    if not m:
        return (9999, 99)
    return (int(m.group(2)), _MONTH_ORDER[m.group(1)])


def fetch_issues(label: str) -> list[dict]:
    cmd = [
        "gh",
        "issue",
        "list",
        "--repo",
        REPO,
        "--label",
        label,
        "--state",
        "open",
        "--limit",
        "500",
        "--json",
        "number,title,labels,url,assignees",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching issues:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def label_names(issue: dict) -> set[str]:
    return {lbl["name"] for lbl in issue["labels"]}


def is_month_label(name: str) -> bool:
    return bool(_MONTH_RE.match(name))


def month_tags(issue: dict) -> str:
    months = sorted(
        (lbl["name"] for lbl in issue["labels"] if is_month_label(lbl["name"])),
        key=month_label_sort_key,
    )
    return ", ".join(months) if months else ""


def earliest_month_key(issue: dict) -> tuple[int, int]:
    """Return the sort key for the earliest month label, or a large sentinel if none."""
    month_labels = [
        lbl["name"] for lbl in issue["labels"] if is_month_label(lbl["name"])
    ]
    if not month_labels:
        return (9999, 99)
    return min(month_label_sort_key(m) for m in month_labels)


def make_sort_key(split_label: str | None):
    def sort_key(issue: dict) -> tuple:
        labels = label_names(issue)
        is_epic = "epic" in labels
        has_split = split_label is not None and any(
            lbl.startswith(split_label) for lbl in labels
        )
        if is_epic:
            group = 0
        elif not has_split:
            group = 1
        else:
            group = 2
        year, month = earliest_month_key(issue)
        return (group, year, month, issue["number"])

    return sort_key


def format_table(issues: list[dict], primary_label: str) -> str:
    col_num = "#"
    col_title = "Title"
    col_labels = "Labels"
    col_months = "Months"
    col_assignee = "Assignee"

    rows = []
    for issue in issues:
        labels = label_names(issue)
        display_labels = ", ".join(
            sorted(
                lbl
                for lbl in labels
                if lbl != primary_label and not is_month_label(lbl)
            )
        )
        title = issue["title"]
        if len(title) > 80:
            title = title[:79] + "…"
        assignee = ", ".join(a["login"] for a in issue.get("assignees", []))
        rows.append(
            {
                "num": str(issue["number"]),
                "title": title,
                "labels": display_labels,
                "months": month_tags(issue),
                "assignee": assignee,
            }
        )

    w_num = max(len(col_num), *(len(r["num"]) for r in rows))
    w_title = max(len(col_title), *(len(r["title"]) for r in rows))
    w_labels = max(len(col_labels), *(len(r["labels"]) for r in rows))
    w_months = max(len(col_months), *(len(r["months"]) for r in rows))
    w_assignee = max(len(col_assignee), *(len(r["assignee"]) for r in rows))

    def row_line(num, title, months, assignee, labels):
        return (
            f"| {num:<{w_num}} "
            f"| {title:<{w_title}} "
            f"| {months:<{w_months}} "
            f"| {assignee:<{w_assignee}} "
            f"| {labels:<{w_labels}} |"
        )

    sep = (
        f"|-{'-' * w_num}-"
        f"|-{'-' * w_title}-"
        f"|-{'-' * w_months}-"
        f"|-{'-' * w_assignee}-"
        f"|-{'-' * w_labels}-|"
    )

    lines = [
        row_line(col_num, col_title, col_months, col_assignee, col_labels),
        sep,
    ]
    for r in rows:
        lines.append(
            row_line(r["num"], r["title"], r["months"], r["assignee"], r["labels"])
        )
    return "\n".join(lines)


def group_of(issue: dict, split_label: str | None) -> int:
    labels = label_names(issue)
    if "epic" in labels:
        return 0
    if split_label is not None and any(lbl.startswith(split_label) for lbl in labels):
        return 2
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="List open issues with a given label from torch-spyre/torch-spyre."
    )
    parser.add_argument(
        "label", help='Primary label to filter issues (e.g. "inductor")'
    )
    parser.add_argument(
        "--split-label",
        metavar="LABEL",
        help=(
            "Label used to further split non-epics into two sections: "
            "non-epics without LABEL, and non-epics with LABEL. "
            "If omitted, all non-epics appear in one table."
        ),
    )
    args = parser.parse_args()

    issues = fetch_issues(args.label)
    issues.sort(key=make_sort_key(args.split_label))

    if not issues:
        print(f"No open issues found with label '{args.label}'.")
        return

    total = len(issues)
    print(f"Open '{args.label}' issues in {REPO} ({total} total)\n")

    by_group: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for issue in issues:
        by_group[group_of(issue, args.split_label)].append(issue)

    if args.split_label is None:
        sections = [
            (0, "Epics"),
            (1, "Non-epics"),
        ]
    else:
        sections = [
            (0, "Epics"),
            (1, f"Non-epics (no '{args.split_label}' label)"),
            (2, f"Non-epics (with '{args.split_label}' label)"),
        ]

    for group_id, section_title in sections:
        group_issues = by_group[group_id]
        print(f"## {section_title} ({len(group_issues)})\n")
        if group_issues:
            print(format_table(group_issues, args.label))
        else:
            print("(none)")
        print()


if __name__ == "__main__":
    main()
