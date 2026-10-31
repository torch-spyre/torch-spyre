#!/usr/bin/env python3
"""
Parses GitHub Actions test log output and produces a structured JSON report
suitable for ingestion into ClickHouse (hw_failure_diagnostics table).

Usage
-----
  # Single log file:
  python3 parse_hardware_failures.py \\
      --log-file run.log \\
      --run-id 27674677047 \\
      [--suite "Inductor Ops Reductions Scalar"]

  # Directory of *.log / *.txt files (one per suite):
  python3 parse_hardware_failures.py \\
      --log-dir ./logs/ \\
      --run-id 27674677047

Please note:
- Hardware identifiers (node name, PCI device, card serial, chip info) only
  appear in the env-var dump on the FINAL attempt (when DTLOG_LEVEL=Info is
  enabled).  The parser back-fills those values onto all earlier attempts of
  the same suite so every record is fully attributed.
- Multiple RAS errors in one attempt are all captured in `ras_events` (a JSON
  array). The `ras_*` top-level fields reflect the FIRST (earliest) event.
- The `failure_reason` is derived from the first RAS event's `name` field, so
  different hardware error types produce different reason codes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from spyre_clickhouse_ingest.hw_parse import (
    _pick_files_from_dir,
    parse_log,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse GHA hardware failure logs → ClickHouse JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--log-file",
        metavar="FILE",
        help="Path to a single log file (use '-' for stdin)",
    )
    src.add_argument(
        "--log-dir",
        metavar="DIR",
        help=(
            "Directory of log files to scan. "
            "Accepts the exact folder structure produced by GHA log downloads: "
            "files named like '24_run-tests _ Inductor Ops Reductions Scalar.txt'. "
            "Meta/gate jobs (Detect changed files, Run Spyre unit tests) are "
            "skipped automatically. Duplicate numbered/.txt vs unnumbered files "
            "are deduplicated automatically."
        ),
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions run ID shown in the Actions URL, e.g. 27674677047",
    )
    p.add_argument(
        "--suite",
        default="",
        help=(
            "Suite name to use when parsing a single --log-file that has no "
            "'=== Attempt N/M: Suite Name ===' banner. "
            "Ignored when --log-dir is used (names are derived from filenames)."
        ),
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation). Smaller files, faster ingest.",
    )
    p.add_argument(
        "--out", metavar="FILE", help="Write JSON output to FILE instead of stdout."
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    all_records: list[dict[str, Any]] = []

    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"[error] Not a directory: {log_dir}", file=sys.stderr)
            sys.exit(1)

        file_suite_pairs = _pick_files_from_dir(log_dir)
        if not file_suite_pairs:
            print(
                f"[warn] No recognised test-suite log files found in {log_dir}",
                file=sys.stderr,
            )

        print(
            f"[info] Found {len(file_suite_pairs)} suite log file(s) to parse",
            file=sys.stderr,
        )

        for fpath, suite_name, is_pod_level_retry in file_suite_pairs:
            text = fpath.read_text(errors="replace")
            recs = parse_log(
                text,
                run_id=args.run_id,
                suite_hint=suite_name,
                is_pod_level_retry=is_pod_level_retry,
            )
            all_records.extend(recs)
            outcomes = [r["outcome"] for r in recs]
            reasons = [
                r["failure_reason"] for r in recs if r["failure_reason"] != "none"
            ]
            print(
                f"[info]  {fpath.name}",
                file=sys.stderr,
            )
            print(
                f"         suite={suite_name!r}  pod_level_retry={is_pod_level_retry}"
                f"  attempts={len(recs)}"
                f"  outcomes={outcomes}  reasons={reasons or ['(none)']!r}",
                file=sys.stderr,
            )

    elif args.log_file == "-" or (not args.log_file and not sys.stdin.isatty()):
        text = sys.stdin.read()
        all_records = parse_log(text, run_id=args.run_id, suite_hint=args.suite)

    elif args.log_file:
        text = Path(args.log_file).read_text(errors="replace")
        all_records = parse_log(text, run_id=args.run_id, suite_hint=args.suite)

    else:
        print(
            "[error] Provide --log-file, --log-dir, or pipe via stdin.", file=sys.stderr
        )
        sys.exit(1)

    # Summary to stderr
    if all_records:
        total = len(all_records)
        failed = sum(1 for r in all_records if r["outcome"] == "failed")
        passed = sum(1 for r in all_records if r["outcome"] == "passed")
        by_reason: dict[str, int] = {}
        for r in all_records:
            k = r["failure_reason"]
            by_reason[k] = by_reason.get(k, 0) + 1
        print("\n[info] ── Summary ──────────────────────────────────", file=sys.stderr)
        print(f"[info]  Total attempt records : {total}", file=sys.stderr)
        print(f"[info]  Passed                : {passed}", file=sys.stderr)
        print(f"[info]  Failed                : {failed}", file=sys.stderr)
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"[info]    {reason:35s}: {count}", file=sys.stderr)

    output = json.dumps(all_records, indent=None if args.compact else 2)

    if args.out:
        Path(args.out).write_text(output)
        print(f"[info]  Output written to: {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
