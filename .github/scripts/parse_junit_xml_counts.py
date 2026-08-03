#!/usr/bin/env python3
"""
Aggregates passed/failed/skipped/error/xfailed/xpassed counts from a pytest
JUnit XML report and reports them to the calling GHA step.

xfailed is reliable: pytest's junitxml plugin emits
<skipped type="pytest.xfail"> for expected failures. xpassed is best-effort
only: a *non-strict* xpass (torch-spyre does not set xfail_strict) is
written as an ordinary empty <testcase/>, indistinguishable from a plain
pass — only a strict xpass would show up here (as a <skipped> with a
"passes unexpectedly" message).

Writes the counts as a Markdown table to $GITHUB_STEP_SUMMARY (if set) and
as `test_results_json=<json>` to $GITHUB_OUTPUT (if set), so a later step
can read `steps.<id>.outputs.test_results_json`.

Usage (called by the GHA workflow):
    python3 parse_junit_xml_counts.py \
        --junit-xml test_upstream_beta_config.xml \
        --summary-title "Upstream Beta test results"
"""

import argparse
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

_COLUMNS = ["total", "passed", "failed", "errors", "skipped", "xfailed", "xpassed"]


def count_testcases(xml_path: Path) -> dict:
    totals = {col: 0 for col in _COLUMNS}
    if not xml_path.is_file():
        return totals

    root = ET.parse(xml_path).getroot()
    cases = root.findall(".//testcase")
    totals["total"] = len(cases)
    for case in cases:
        skipped = case.find("skipped")
        if case.find("failure") is not None:
            totals["failed"] += 1
        elif case.find("error") is not None:
            totals["errors"] += 1
        elif skipped is not None:
            if skipped.get("type") == "pytest.xfail":
                totals["xfailed"] += 1
            elif "passes unexpectedly" in (skipped.get("message") or ""):
                totals["xpassed"] += 1
            else:
                totals["skipped"] += 1
        else:
            totals["passed"] += 1
    return totals


def write_step_summary(summary_path: str, title: str, totals: dict) -> None:
    if not summary_path:
        return
    header = " | ".join(col.capitalize() for col in _COLUMNS)
    separator = "|".join(["---"] * len(_COLUMNS))
    row = " | ".join(str(totals[col]) for col in _COLUMNS)
    with open(summary_path, "a") as f:
        f.write(f"### {title}\n\n| {header} |\n|{separator}|\n| {row} |\n")


def write_output(output_path: str, totals: dict) -> None:
    if not output_path:
        return
    with open(output_path, "a") as f:
        f.write(f"test_results_json={json.dumps(totals)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit-xml", required=True, help="Path to the merged JUnit XML report"
    )
    parser.add_argument(
        "--summary-title",
        default="Test results",
        help="Heading for the $GITHUB_STEP_SUMMARY table",
    )
    args = parser.parse_args()

    totals = count_testcases(Path(args.junit_xml))
    print(f"Test totals: {totals}")

    write_step_summary(
        os.environ.get("GITHUB_STEP_SUMMARY", ""), args.summary_title, totals
    )
    write_output(os.environ.get("GITHUB_OUTPUT", ""), totals)


if __name__ == "__main__":
    main()
