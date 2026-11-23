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

"""`python -m spyre_clickhouse_ingest {artifacts|results|ci-run-timings|dispatch|spool|relay} ...` (also `spyre-clickhouse-ingest`)."""

import sys

# `results` modes that never reach ClickHouse; they run on the standard library alone.
OFFLINE_FLAGS = ("--offline", "--validate-only", "--upload", "--check")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = (
        "artifacts",
        "results",
        "ci-run-timings",
        "dispatch",
        "spool",
        "relay",
        "bench-verdicts",
    )
    if not argv or argv[0] not in commands:
        sys.exit(
            f"usage: python -m spyre_clickhouse_ingest {{{'|'.join(commands)}}} ..."
        )
    command, rest = argv[0], argv[1:]
    if command == "artifacts":
        from .artifacts import main as run
    elif command == "ci-run-timings":
        from .ci_run_timings import main as run
    elif command == "dispatch":
        from .dispatch import main as run
    elif command == "spool":
        from .spool_cli import main as run
    elif command == "relay":
        from .relay import main as run
    elif command == "bench-verdicts":
        from .bench_verdicts import main as run
    elif any(a.split("=")[0] in OFFLINE_FLAGS for a in rest):
        from .offline import main as run
    else:
        from .results import main as run
    if command in ("relay", "bench-verdicts"):
        return run(
            rest
        )  # a park there is a failed write, retried by their own loop or caller
    try:
        rc = run(rest)
    except SystemExit as stop:
        code = _with_parked(stop.code)
        if code is stop.code:
            raise
        raise SystemExit(code) from None
    return _with_parked(rc)


def _with_parked(rc):
    """EXIT_PARKED for a clean run that parked rows (possible only once `writes` is imported)."""
    writes = sys.modules.get(f"{__package__}.writes")
    return writes.finish(rc) if writes else rc


if __name__ == "__main__":
    sys.exit(main())
