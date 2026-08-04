#!/usr/bin/env python3
"""
Builds the `result_payload` JSON blob attached to the ClickHouse
pytorch_ci_dispatches row's `completed` stage report (see
ingest_pytorch_dispatch.py --result-payload-file).

Merges: the test-counts JSON produced by parse_junit_xml_counts.py, the
job's conclusion, and any number of stage durations computed from pairs of
Unix-epoch-second environment variables written earlier in the job via
`echo "VAR=$(date +%s)" >> "$GITHUB_ENV"`. A duration is omitted (null) if
either boundary env var is unset, e.g. because that stage never ran.

Usage (called by the GHA workflow):
    python3 build_stage_result_payload.py \
        --test-results-json "$TEST_RESULTS_JSON" \
        --conclusion success \
        --duration pytorch_build_seconds=T_TORCH_BUILD_START:T_TORCH_SPYRE_BUILD_START \
        --duration torch_spyre_build_seconds=T_TORCH_SPYRE_BUILD_START:T_RUNNING_TESTS_START \
        --duration tests_seconds=T_RUNNING_TESTS_START:T_RUNNING_TESTS_END \
        --output-file result_payload.json
"""

import argparse
import json
import os
import sys


def parse_duration_spec(spec: str) -> tuple[str, str, str]:
    """Parse "name=START_ENV:END_ENV" into (name, start_env, end_env)."""
    name, _, bounds = spec.partition("=")
    start_env, _, end_env = bounds.partition(":")
    if not name or not start_env or not end_env:
        raise argparse.ArgumentTypeError(
            f"--duration must look like 'name=START_ENV:END_ENV', got: {spec!r}"
        )
    return name, start_env, end_env


def elapsed_seconds(start_env: str, end_env: str) -> int | None:
    start, end = os.environ.get(start_env), os.environ.get(end_env)
    if not start or not end:
        return None
    return int(end) - int(start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-results-json",
        default="{}",
        help="JSON string of test counts (e.g. from parse_junit_xml_counts.py)",
    )
    parser.add_argument(
        "--conclusion", required=True, help="Job conclusion, e.g. success | failure"
    )
    parser.add_argument(
        "--duration",
        action="append",
        default=[],
        type=parse_duration_spec,
        metavar="NAME=START_ENV:END_ENV",
        help=(
            "Repeatable. Adds payload[NAME] = int($END_ENV) - int($START_ENV), "
            "or null if either env var is unset."
        ),
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to write the merged JSON payload to"
    )
    args = parser.parse_args()

    try:
        payload = json.loads(args.test_results_json or "{}")
    except json.JSONDecodeError as exc:
        print(f"[error] Could not parse --test-results-json: {exc}", file=sys.stderr)
        sys.exit(1)

    payload["conclusion"] = args.conclusion
    for name, start_env, end_env in args.duration:
        payload[name] = elapsed_seconds(start_env, end_env)

    with open(args.output_file, "w") as f:
        json.dump(payload, f)

    print(f"Wrote result payload to {args.output_file}: {payload}")


if __name__ == "__main__":
    main()
