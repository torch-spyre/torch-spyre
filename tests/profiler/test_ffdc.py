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

import json
import tempfile
from pathlib import Path

from torch_spyre.profiler._ffdc import (
    CATEGORY_COMPILE,
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    CATEGORY_UNKNOWN,
    _MAX_REPORTS,
    _prune_old_reports,
    REQUIRED_FIELDS,
    collect,
    get_diagnostic_report,
)


class TestFfdcCollect:
    def _collect_to_tmpdir(self, exc=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(exc, output_dir=tmp, **kwargs)
            # verify the JSON file was written and is valid
            path = report.get("_report_path")
            assert path is not None
            with open(path) as f:
                on_disk = json.load(f)
            assert on_disk["failure"]["category"] == report["failure"]["category"]
        return report

    def test_collect_with_exception_is_complete(self):
        try:
            raise ValueError("test failure")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        assert report["collector"]["completeness_pct"] == 100.0
        assert report["collector"]["missing_fields"] == []
        assert report["collector"]["success"] is True

    def test_failure_fields_populated(self):
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_COMPILE)

        assert report["failure"]["category"] == CATEGORY_COMPILE
        assert report["failure"]["exception_type"] == "RuntimeError"
        assert "something went wrong" in report["failure"]["message"]
        assert isinstance(report["failure"]["traceback"], str)
        assert "RuntimeError" in report["failure"]["traceback"]

    def test_traceback_is_joined_string(self):
        try:
            raise TypeError("bad type")
        except TypeError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        tb = report["failure"]["traceback"]
        assert isinstance(tb, str)
        assert len(tb.splitlines()) > 1

    def test_runtime_context_passed_through(self):
        try:
            raise RuntimeError("kernel failed")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(
                exc,
                failure_category=CATEGORY_RUNTIME_LAUNCH,
                kernel_name="my_kernel",
                code_dir="/tmp/code",
            )

        assert report["runtime"]["kernel_name"] == "my_kernel"
        assert report["runtime"]["code_dir"] == "/tmp/code"

    def test_runtime_context_absent_is_none(self):
        try:
            raise RuntimeError("unimplemented")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(
                exc, failure_category=CATEGORY_UNIMPLEMENTED
            )

        assert report["runtime"]["kernel_name"] is None
        assert report["runtime"]["code_dir"] is None

    def test_collect_never_raises(self):
        # collect() must be best-effort; write failures must not propagate.
        # Use a plain file as output_dir so mkdir() raises NotADirectoryError —
        # a reliably unwritable path on every platform without root access.
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not_a_dir"
            blocker.write_text("")  # create a file where a directory is expected
            report = collect(
                None,
                failure_category=CATEGORY_UNKNOWN,
                output_dir=str(blocker / "subdir"),
            )
        assert report is not None
        assert report["_report_path"] is None
        assert report["collector"]["success"] is False

    def test_category_constants_match_report(self):
        for category in (
            CATEGORY_COMPILE,
            CATEGORY_RUNTIME_LAUNCH,
            CATEGORY_UNIMPLEMENTED,
            CATEGORY_UNKNOWN,
        ):
            try:
                raise ValueError("x")
            except ValueError as exc:
                report = self._collect_to_tmpdir(exc, failure_category=category)
            assert report["failure"]["category"] == category

    def test_report_filename_contains_category(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            with tempfile.TemporaryDirectory() as tmp:
                report = collect(exc, failure_category=CATEGORY_COMPILE, output_dir=tmp)
                fname = Path(report["_report_path"]).name
        assert fname.startswith("ffdc_compile_")
        assert ".json" in fname

    def test_completeness_pct_reflects_missing_fields(self):
        # Without an exception, failure.exception_type and failure.traceback are
        # None, so they appear in missing_fields.  This verifies that
        # completeness_pct is driven by REQUIRED_FIELDS programmatically:
        # any drift between the two would show up here as a wrong percentage.
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)

        missing = report["collector"]["missing_fields"]
        assert "failure.exception_type" in missing
        assert "failure.traceback" in missing
        # REQUIRED_FIELDS has 11 entries; exc=None leaves exception_type and
        # traceback as None (2 missing, 9 present).
        # round(100 * 9 / 11, 1) == 81.8  — hardcoded to catch formula regressions.
        assert len(REQUIRED_FIELDS) == 11, (
            "Update the expected_pct below if REQUIRED_FIELDS changes"
        )
        assert report["collector"]["completeness_pct"] == 81.8
        assert report["collector"]["completeness_pct"] < 100.0

    def test_metadata_fields_present(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        meta = report["metadata"]
        for key in (
            "timestamp",
            "host",
            "pid",
            "python_version",
            "torch_version",
            "platform",
        ):
            assert key in meta

    def test_environment_keys_captured(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        env = report["environment"]
        for key in ("TORCH_COMPILE_DEBUG", "TORCH_SPYRE_DEBUG", "SPYRE_INDUCTOR_LOG"):
            assert key in env

    def test_capture_latency_is_positive(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        assert report["collector"]["capture_latency_ms"] > 0

    def test_get_diagnostic_report_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert get_diagnostic_report(output_dir=tmp) is None

    def test_get_diagnostic_report_returns_latest(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("first")
            except RuntimeError as exc:
                r1 = collect(exc, failure_category=CATEGORY_COMPILE, output_dir=tmp)
            # Pin the first file's mtime to epoch so the second is unambiguously newer.
            os.utime(r1["_report_path"], (0, 0))
            try:
                raise RuntimeError("second")
            except RuntimeError as exc:
                collect(exc, failure_category=CATEGORY_RUNTIME_LAUNCH, output_dir=tmp)

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert "failure" in result
            assert result["failure"]["category"] == CATEGORY_RUNTIME_LAUNCH

    def test_get_diagnostic_report_returns_latest_across_categories(self):
        # A fresh compile report must win over a stale unknown report.
        # With name-sort, unknown > compile lexically so the stale unknown
        # would be returned instead.
        import os

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            stale_unknown = d / "ffdc_unknown_20250101T000000_000000_1.json"
            fresh_compile = d / "ffdc_compile_20250101T000001_000000_1.json"
            stale_unknown.write_text('{"failure": {"category": "unknown"}}')
            fresh_compile.write_text('{"failure": {"category": "compile"}}')
            os.utime(stale_unknown, (0, 0))  # mtime: epoch
            os.utime(fresh_compile, (100, 100))  # mtime: 100 s later

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["failure"]["category"] == "compile"

    def test_prune_old_reports_removes_oldest(self):
        # _prune_old_reports keeps the newest `keep` files by mtime, not by name.
        # compile sorts first lexically, so use compile as the NEWEST category —
        # a name-sort regression would evict these and wrongly keep the older files.
        import os

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # oldest → newest by mtime (index = mtime in seconds since epoch)
            files = [
                "ffdc_unknown_20250101T000000_000000_1.json",  # mtime 0 - oldest
                "ffdc_runtime_launch_20250101T000001_000000_1.json",  # mtime 1
                "ffdc_compile_20250101T000002_000000_1.json",  # mtime 2
                "ffdc_compile_20250101T000003_000000_1.json",  # mtime 3
                "ffdc_compile_20250101T000004_000000_1.json",  # mtime 4 - newest
            ]
            for i, name in enumerate(files):
                p = d / name
                p.write_text("{}")
                os.utime(p, (i, i))  # mtime = i seconds since epoch
            _prune_old_reports(d, keep=3)
            remaining = sorted(d.glob("ffdc_*.json"), key=lambda p: p.stat().st_mtime)
            assert len(remaining) == 3
            # The three newest by mtime must survive — all three are compile files
            # even though compile sorts first by name.
            assert [p.name for p in remaining] == [
                "ffdc_compile_20250101T000002_000000_1.json",
                "ffdc_compile_20250101T000003_000000_1.json",
                "ffdc_compile_20250101T000004_000000_1.json",
            ]

    def test_collect_prunes_beyond_max_reports(self):
        # After writing, collect() must not leave more than _MAX_REPORTS files.
        with tempfile.TemporaryDirectory() as tmp:
            # Pre-seed the directory with _MAX_REPORTS files so the next write
            # would exceed the cap.
            d = Path(tmp)
            for i in range(_MAX_REPORTS):
                (d / f"ffdc_unknown_20240101T{i:06d}_000000_1.json").write_text("{}")
            try:
                raise ValueError("x")
            except ValueError as exc:
                collect(exc, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
            assert len(list(d.glob("ffdc_*.json"))) <= _MAX_REPORTS
