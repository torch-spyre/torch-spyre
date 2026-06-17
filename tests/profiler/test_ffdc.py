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
import unittest
from pathlib import Path

from torch_spyre.profiler._ffdc import (
    CATEGORY_COMPILE,
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    CATEGORY_UNKNOWN,
    collect,
    get_diagnostic_report,
)


class TestFfdcCollect(unittest.TestCase):
    def _collect_to_tmpdir(self, exc=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(exc, output_dir=tmp, **kwargs)
            # verify the JSON file was written and is valid
            path = report.get("_report_path")
            self.assertIsNotNone(path)
            with open(path) as f:
                on_disk = json.load(f)
            self.assertEqual(
                on_disk["failure"]["category"], report["failure"]["category"]
            )
        return report

    def test_collect_with_exception_is_complete(self):
        try:
            raise ValueError("test failure")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        self.assertEqual(report["collector"]["completeness_pct"], 100.0)
        self.assertEqual(report["collector"]["missing_fields"], [])
        self.assertTrue(report["collector"]["success"])

    def test_failure_fields_populated(self):
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_COMPILE)

        self.assertEqual(report["failure"]["category"], CATEGORY_COMPILE)
        self.assertEqual(report["failure"]["exception_type"], "RuntimeError")
        self.assertIn("something went wrong", report["failure"]["message"])
        self.assertIsInstance(report["failure"]["traceback"], str)
        self.assertIn("RuntimeError", report["failure"]["traceback"])

    def test_traceback_is_joined_string(self):
        try:
            raise TypeError("bad type")
        except TypeError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        tb = report["failure"]["traceback"]
        self.assertIsInstance(tb, str)
        self.assertGreater(len(tb.splitlines()), 1)

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

        self.assertEqual(report["runtime"]["kernel_name"], "my_kernel")
        self.assertEqual(report["runtime"]["code_dir"], "/tmp/code")

    def test_runtime_context_absent_is_none(self):
        try:
            raise RuntimeError("unimplemented")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(
                exc, failure_category=CATEGORY_UNIMPLEMENTED
            )

        self.assertIsNone(report["runtime"]["kernel_name"])
        self.assertIsNone(report["runtime"]["code_dir"])

    def test_collect_never_raises(self):
        # collect() must be best-effort; exceptions inside must not propagate
        report = collect(
            None, failure_category=CATEGORY_UNKNOWN, output_dir="/nonexistent/path/xyz"
        )
        self.assertIsNotNone(report)

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
            self.assertEqual(report["failure"]["category"], category)

    def test_report_filename_contains_category(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            with tempfile.TemporaryDirectory() as tmp:
                report = collect(exc, failure_category=CATEGORY_COMPILE, output_dir=tmp)
                fname = Path(report["_report_path"]).name
        self.assertTrue(fname.startswith("ffdc_compile_"))
        self.assertIn(".json", fname)

    def test_required_fields_coverage(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        self.assertEqual(report["collector"]["missing_fields"], [])

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
            self.assertIn(key, meta)

    def test_environment_keys_captured(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        env = report["environment"]
        for key in ("TORCH_COMPILE_DEBUG", "TORCH_SPYRE_DEBUG", "SPYRE_INDUCTOR_LOG"):
            self.assertIn(key, env)

    def test_capture_latency_is_positive(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        self.assertGreater(report["collector"]["capture_latency_ms"], 0)

    def test_get_diagnostic_report_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(get_diagnostic_report(output_dir=tmp))

    def test_get_diagnostic_report_returns_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("first")
            except RuntimeError as exc:
                collect(exc, failure_category=CATEGORY_COMPILE, output_dir=tmp)
            try:
                raise RuntimeError("second")
            except RuntimeError as exc:
                collect(exc, failure_category=CATEGORY_RUNTIME_LAUNCH, output_dir=tmp)

            result = get_diagnostic_report(output_dir=tmp)
            self.assertIsNotNone(result)
            self.assertIn("failure", result)
            self.assertEqual(result["failure"]["category"], CATEGORY_RUNTIME_LAUNCH)
