#!/usr/bin/env python3
"""
Aggregates passed/failed/skipped/error/xfailed/xpassed counts (and per-suite
durations) from one or more pytest JUnit XML reports and reports them to the
calling GHA step.

Usage (called by the GHA workflow):
    python3 parse_junit_xml_counts.py \
        --junit-xml test_upstream_beta_config.xml \
        --summary-title "Upstream Beta test results"

    # Multiple reports (e.g. one per matrix-job suite) are summed together;
    # the reported duration is the MAX across reports, not the sum, since
    # matrix jobs run in parallel and wall-clock is dominated by the
    # slowest one.
    python3 parse_junit_xml_counts.py \
        --junit-xml test_indexing.xml --junit-xml test_linalg.xml \
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


def suite_duration_seconds(xml_path: Path) -> float:
    """Sum of every <testcase time="..."> in the report.

    Summing testcase times (rather than trusting the <testsuite time="...">
    header) is the same defensive read _test_matrix.yaml's empty-suite check
    uses for testcase counts -- pytest-xdist writes unreliable suite-level
    attributes in some configurations.
    """
    if not xml_path.is_file():
        return 0.0
    root = ET.parse(xml_path).getroot()
    total = 0.0
    for case in root.findall(".//testcase"):
        try:
            total += float(case.get("time") or 0.0)
        except ValueError:
            pass
    return total


def write_step_summary(
    summary_path: str, title: str, totals: dict, durations: dict
) -> None:
    if not summary_path:
        return
    header = " | ".join(col.capitalize() for col in _COLUMNS)
    separator = "|".join(["---"] * len(_COLUMNS))
    row = " | ".join(str(totals[col]) for col in _COLUMNS)
    lines = [f"### {title}\n", f"| {header} |", f"|{separator}|", f"| {row} |", ""]

    if len(durations) > 1:
        lines.append("### Per-suite durations\n")
        lines.append("| Suite | Seconds |")
        lines.append("|---|---|")
        for name, seconds in sorted(durations.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{name}` | {seconds:.1f} |")
        lines.append("")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def write_output(output_path: str, totals: dict, max_duration_seconds: float) -> None:
    if not output_path:
        return
    with open(output_path, "a") as f:
        f.write(f"test_results_json={json.dumps(totals)}\n")
        f.write(f"max_duration_seconds={int(round(max_duration_seconds))}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit-xml",
        action="append",
        default=[],
        dest="junit_xml",
        help="Path to a JUnit XML report. Repeatable — counts are summed, "
        "duration is the max across all reports. May be omitted entirely "
        "(e.g. a build failure upstream meant no suite ever ran) — reports "
        "all-zero totals rather than erroring.",
    )
    parser.add_argument(
        "--summary-title",
        default="Test results",
        help="Heading for the $GITHUB_STEP_SUMMARY table",
    )
    args = parser.parse_args()

    totals = {col: 0 for col in _COLUMNS}
    durations = {}
    for raw_path in args.junit_xml:
        xml_path = Path(raw_path)
        file_totals = count_testcases(xml_path)
        for col in _COLUMNS:
            totals[col] += file_totals[col]
        durations[xml_path.stem] = suite_duration_seconds(xml_path)

    max_duration = max(durations.values(), default=0.0)
    print(f"Test totals: {totals}")
    print(f"Per-suite durations (seconds): {durations}")
    print(f"Max duration: {max_duration:.1f}s")

    write_step_summary(
        os.environ.get("GITHUB_STEP_SUMMARY", ""), args.summary_title, totals, durations
    )
    write_output(os.environ.get("GITHUB_OUTPUT", ""), totals, max_duration)


if __name__ == "__main__":
    main()
