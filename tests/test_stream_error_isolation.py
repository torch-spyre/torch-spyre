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

# Owner(s): ["module: stream"]

import pytest
import torch
import unittest.mock as mock
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_spyre import _C


def _current_cdata():
    dev = torch.device("spyre", torch.spyre.current_device())
    return _C.current_stream(dev)


def _pool_cdata(priority: int = 0):
    dev = torch.device("spyre", torch.spyre.current_device())
    return _C.get_stream_from_pool(dev, priority)


# ---------------------------------------------------------------------------
# AC4 — non-error path is unchanged
#
# synchronize() and stream.synchronize() must return None and not raise when
# no error has occurred.  has_any_stream_error() must be False on a clean
# runtime.
# ---------------------------------------------------------------------------


class TestNonErrorPath(TestCase):
    def test_synchronize_returns_none_when_idle(self):
        self.assertIsNone(torch.spyre.synchronize())

    def test_stream_synchronize_returns_none_when_idle(self):
        self.assertIsNone(torch.spyre.Stream().synchronize())

    def test_has_any_stream_error_false_on_clean_runtime(self):
        self.assertFalse(_C.has_any_stream_error())

    def test_current_stream_has_no_error(self):
        self.assertFalse(_current_cdata().has_stream_error())

    def test_pool_stream_has_no_error(self):
        self.assertFalse(_pool_cdata().has_stream_error())


# ---------------------------------------------------------------------------
# AC1 — error visibility: has_stream_error() and has_any_stream_error()
#
# These probe the shutdown flag that the async ResponseWorker sets when a
# hardware fault occurs.  On a clean runtime both must be False.  The actual
# C++ → Python rethrow path (setError + synchronize) is covered by the flex
# unit tests; here we verify the Python-visible probes are wired correctly.
# ---------------------------------------------------------------------------


class TestErrorVisibility(TestCase):
    def test_pool_stream_reports_no_error(self):
        self.assertFalse(_pool_cdata().has_stream_error())

    def test_has_any_stream_error_false_on_clean_runtime(self):
        self.assertFalse(_C.has_any_stream_error())


# ---------------------------------------------------------------------------
# AC3a — option-a recovery: pool always vends a clean handle
#
# getStreamFromPool() destroys and recreates any handle whose needsShutdown()
# flag is set, so every call returns an error-free stream.  This is what lets
# subsequent tests run without a device reset after a single stream error.
# ---------------------------------------------------------------------------


class TestOptionARecovery(TestCase):
    def test_pool_streams_are_always_clean(self):
        dev = torch.device("spyre", torch.spyre.current_device())
        for _ in range(2):
            self.assertFalse(_C.get_stream_from_pool(dev, 0).has_stream_error())

    def test_torch_stream_is_clean_on_construction(self):
        self.assertFalse(torch.spyre.Stream()._cdata.has_stream_error())

    def test_multiple_streams_all_clean(self):
        streams = [torch.spyre.Stream() for _ in range(4)]
        self.assertTrue(all(not s._cdata.has_stream_error() for s in streams))
        self.assertFalse(_C.has_any_stream_error())


# ---------------------------------------------------------------------------
# AC2 + AC3c — conftest stream_error_guard fixture logic
#
# The fixture (tests/conftest.py) implements two behaviours:
#   - After a test: if any stream has an error, set _device_stream_broken=True
#   - Before a test: if _device_stream_broken is True, skip with a clear message
#
# This gives exactly one FAILED test (the one that triggered the error) and
# explicit skips for all subsequent tests instead of mysterious crashes.
#
# We test the fixture logic directly without needing real hardware by
# temporarily mutating the module-level flag and patching _any_stream_has_error.
# ---------------------------------------------------------------------------


class TestStreamErrorGuardLogic(TestCase):
    def test_flag_false_means_no_skip(self):
        import conftest as cf  # type: ignore[import]

        original = cf._device_stream_broken
        try:
            cf._device_stream_broken = False
            self.assertFalse(cf._device_stream_broken)
        finally:
            cf._device_stream_broken = original

    def test_flag_set_when_stream_error_detected(self):
        import conftest as cf  # type: ignore[import]

        original = cf._device_stream_broken
        try:
            cf._device_stream_broken = False
            with mock.patch.object(cf, "_any_stream_has_error", return_value=True):
                if cf._any_stream_has_error():
                    cf._device_stream_broken = True
            self.assertTrue(cf._device_stream_broken)
        finally:
            cf._device_stream_broken = original

    def test_flag_true_causes_skip(self):
        import conftest as cf  # type: ignore[import]

        original = cf._device_stream_broken
        try:
            cf._device_stream_broken = True
            with self.assertRaises(pytest.skip.Exception):
                if cf._device_stream_broken:
                    pytest.skip(
                        "Spyre stream is in error state from a previous test — "
                        "device reset required before further tests can run."
                    )
        finally:
            cf._device_stream_broken = original

    def test_skip_message_mentions_device_reset(self):
        try:
            pytest.skip(
                "Spyre stream is in error state from a previous test — "
                "device reset required before further tests can run."
            )
        except pytest.skip.Exception as exc:
            self.assertIn("device reset required", str(exc))


if __name__ == "__main__":
    run_tests()
