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

"""Unit tests for the parse internals. The behavioural cases live in the root suite
(tests/test_hw_diagnostics_scripts.py), which CI actually runs."""

import pytest
from spyre_clickhouse_ingest import hw_parse

RAS_LINE = (
    'ERRR 15.09.2026 10:25:09.123456 [ras_base.hpp: 74] {"Device":"/dev/vfio/1",'
    '"action":"information","category":"configuration","code":"0xf40a",'
    '"message":"timeout","name":"RAS::CBRB::ResponseTimeout","severity":"ERROR"}'
)


@pytest.mark.parametrize(
    "line",
    [
        "==================== 12 passed in 8.11s ====================",
        "============ 4 failed, 96 passed in 120.5s ============",
        "==== 5 passed, 2 xfailed in 9.00s ====",
        "7 passed, 1 failed in 3.2s",
    ],
)
def test_summary_shapes_are_recognised(line):
    assert hw_parse._pytest_summary_line([line]) == line


@pytest.mark.parametrize(
    "line",
    [
        "INFO 15.09.2026 10:00:00.1 [dt] retry queue: 3 errors drained",
        "tests/test_foo.py ....  4 passed",
        "[stall-watcher] No new output for 600s",
        "collected 231 items",
    ],
)
def test_non_summary_lines_are_ignored(line):
    # Each of these contains a count that must not be mistaken for the run's verdict.
    assert hw_parse._pytest_summary_line([line]) == ""


def test_the_last_summary_wins():
    # With reruns a chunk holds several; the final one is the attempt's verdict.
    first = "==== 1 passed in 0.1s ===="
    last = "==== 4 failed, 96 passed in 120.5s ===="
    assert hw_parse._pytest_summary_line([first, "noise", last]) == last


def test_crash_detail_is_skipped_when_ras_events_exist(monkeypatch):
    # Asserts the SKIP, not just the output: the guard exists so the signal/heap scan and the
    # backtrace walk do not run on the common hardware-failure path.
    def boom(*args, **kwargs):
        raise AssertionError(
            "_extract_crash_detail ran despite RAS events being present"
        )

    monkeypatch.setattr(hw_parse, "_extract_crash_detail", boom)
    records = hw_parse.parse_log(
        f"=== Attempt 1/2: S ===\n{RAS_LINE}\n=== Attempt 1 FAILED (exit=1)\n",
        run_id="1",
        suite_hint="S",
    )
    assert records[0]["failure_reason"] == "hardware_ras_timeout"


def test_crash_detail_still_runs_without_ras_events(monkeypatch):
    calls = []
    real = hw_parse._extract_crash_detail

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hw_parse, "_extract_crash_detail", spy)
    hw_parse.parse_log(
        "=== Attempt 1/1: S ===\nSegmentation fault\n=== Attempt 1 FAILED (exit=139)\n",
        run_id="1",
        suite_hint="S",
    )
    assert calls, "crash detail must still be computed when there are no RAS events"


def test_counts_are_read_through_the_module_level_patterns(monkeypatch):
    # Single-sourcing, behaviourally: neutralising the shared pattern must change what the
    # parser reports. A re-inlined copy of the regex would keep reporting 96.
    monkeypatch.setattr(
        hw_parse, "RE_PY_PASSED", hw_parse.re.compile(r"(?P<n>\d+) zzz")
    )
    records = hw_parse.parse_log(
        "=== Attempt 1/1: S ===\n==== 4 failed, 96 passed in 1.0s ====\n",
        run_id="1",
        suite_hint="S",
    )
    assert records[0]["tests_passed"] == 0
    assert records[0]["tests_failed"] == 4
