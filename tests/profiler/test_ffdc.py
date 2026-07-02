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

import pytest
import torch
import torch_spyre  # noqa: F401

from torch_spyre import make_spyre_module  # type: ignore[attr-defined]
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.profiler import get_diagnostic_report as profiler_get_diagnostic_report
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


@pytest.fixture(scope="module", autouse=True)
def register_torch_spyre_public_api():
    if not hasattr(torch, "spyre"):
        torch.utils.rename_privateuse1_backend(DEVICE_NAME)
        torch._register_device_module(DEVICE_NAME, make_spyre_module())


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

    def test_collect_filename_parses_for_timestamp_sort_key(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            with tempfile.TemporaryDirectory() as tmp:
                report = collect(
                    exc, failure_category=CATEGORY_RUNTIME_LAUNCH, output_dir=tmp
                )
                path = Path(report["_report_path"])

        parts = path.stem.rsplit("_", 3)
        assert len(parts) == 4
        assert parts[0] == "ffdc_runtime_launch"
        assert parts[1].startswith("20") and "T" in parts[1]
        assert parts[2].isdigit()
        assert parts[3].isdigit()
        sort_key = f"{parts[1]}_{parts[2]}"
        assert len(sort_key) > 0

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
            assert result["_report_path"].endswith(".json")

    def test_get_diagnostic_report_skips_corrupted_newest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            corrupt = d / "ffdc_compile_20250101T000002_000000_1.json"
            valid = d / "ffdc_unknown_20250101T000001_000000_1.json"
            corrupt.write_text("{not valid json")
            valid.write_text('{"failure": {"category": "unknown"}}')

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["failure"]["category"] == "unknown"
            assert result["_report_path"] == str(valid)

    def test_get_diagnostic_report_returns_none_when_all_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "ffdc_unknown_20250101T000000_000000_1.json").write_text("{bad json")

            assert get_diagnostic_report(output_dir=tmp) is None

    def test_get_diagnostic_report_includes_report_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("path test")
            except RuntimeError as exc:
                collect(exc, failure_category=CATEGORY_COMPILE, output_dir=tmp)

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert Path(result["_report_path"]).is_file()
            assert result["_report_path"].startswith(tmp)

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


class TestFfdcPublicApi:
    def test_torch_spyre_exposes_get_diagnostic_report(self):
        assert hasattr(torch.spyre, "get_diagnostic_report")
        assert callable(torch.spyre.get_diagnostic_report)

    def test_profiler_package_reexports_get_diagnostic_report(self):
        assert profiler_get_diagnostic_report is not None
        assert callable(profiler_get_diagnostic_report)

    def test_torch_spyre_get_diagnostic_report_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert torch.spyre.get_diagnostic_report(output_dir=tmp) is None

    def test_torch_spyre_get_diagnostic_report_returns_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("public api")
            except RuntimeError as exc:
                collect(exc, failure_category=CATEGORY_RUNTIME_LAUNCH, output_dir=tmp)

            result = torch.spyre.get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["failure"]["category"] == CATEGORY_RUNTIME_LAUNCH
            assert result["_report_path"].endswith(".json")
