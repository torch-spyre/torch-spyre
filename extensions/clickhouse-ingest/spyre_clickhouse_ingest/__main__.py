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

"""`python -m spyre_clickhouse_ingest {artifacts|results|ci-run-timings|dispatch|spool|relay|bench-verdicts|probe} ...` (also `spyre-clickhouse-ingest`)."""

import sys

# `results` modes that never reach ClickHouse; they run on the standard library alone, so they
# never import `writes` (its retry module needs the `regex` package).
OFFLINE_FLAGS = ("--offline", "--validate-only", "--upload", "--check")


def _offline(argv) -> bool:
    return (
        bool(argv)
        and argv[0] == "results"
        and any(a.split("=")[0] in OFFLINE_FLAGS for a in argv[1:])
    )


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
        "probe",
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
    elif command == "probe":
        from .probe import main as run
    elif _offline(argv):
        from .offline import main as run
    else:
        from .results import main as run
    if _offline(argv):
        return run(rest)
    from . import writes

    try:
        rc = run(rest)
    except SystemExit as stop:
        if isinstance(stop.code, str):
            raise
        raise SystemExit(writes.exit_code(stop.code)) from None
    return writes.exit_code(rc)


def entry():
    """The command-line entry: `main` plus the exit-75-on-parked-rows hook."""
    if not _offline(sys.argv[1:]):
        from . import writes

        writes.install_exit_hook()
    return main()


if __name__ == "__main__":
    sys.exit(entry())
