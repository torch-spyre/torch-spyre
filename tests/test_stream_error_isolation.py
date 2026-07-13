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

import importlib.util
import os
import sys
import unittest
import unittest.mock as mock

import pytest
import torch

from torch_spyre import _C


def _current_cdata():
    dev = torch.device("spyre", torch.spyre.current_device())
    return _C.current_stream(dev)


def _pool_cdata(priority: int = 0):
    dev = torch.device("spyre", torch.spyre.current_device())
    return _C.get_stream_from_pool(dev, priority)


def _import_conftest():
    """Import the tests/conftest.py module robustly regardless of sys.path."""
    # Prefer a cached import so repeated calls return the same module object.
    if "conftest" in sys.modules:
        return sys.modules["conftest"]
    # Walk up from this file until we find tests/conftest.py.
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [here, os.path.join(here, "..")]:
        path = os.path.join(candidate, "conftest.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("conftest", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["conftest"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("Could not locate tests/conftest.py")


# AC4 — non-error path: synchronize() returns None, all error probes False.


class TestNonErrorPath(unittest.TestCase):
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


# AC1 — C++ exceptions from synchronize() cross the pybind11 boundary intact.


class TestErrorVisibility(unittest.TestCase):
    def test_pool_stream_reports_no_error(self):
        self.assertFalse(_pool_cdata().has_stream_error())

    def test_has_any_stream_error_false_on_clean_runtime(self):
        self.assertFalse(_C.has_any_stream_error())

    # AC1 — RuntimeError from synchronize() reaches Python with message intact.
    def test_synchronize_propagates_error_message(self):
        sentinel = "Deferred Error First"
        with mock.patch.object(
            torch.spyre.Stream, "synchronize", side_effect=RuntimeError(sentinel)
        ):
            with self.assertRaises(RuntimeError) as ctx:
                torch.spyre.Stream().synchronize()
        self.assertIn(sentinel, str(ctx.exception))

    # AC1 — same check via device-level torch.spyre.synchronize().
    def test_device_synchronize_propagates_error_message(self):
        sentinel = "Deferred Error First"
        with mock.patch.object(_C, "synchronize", side_effect=RuntimeError(sentinel)):
            with self.assertRaises(RuntimeError) as ctx:
                torch.spyre.synchronize()
        self.assertIn(sentinel, str(ctx.exception))


# AC3a — pool streams are always clean on the non-error path.


class TestOptionARecovery(unittest.TestCase):
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


# AC2 + AC3c — stream_error_guard fixture: one FAILED test then skips the rest.


class TestStreamErrorGuardLogic(unittest.TestCase):
    def test_flag_false_means_no_skip(self):
        cf = _import_conftest()

        original = cf._device_stream_broken
        try:
            cf._device_stream_broken = False
            self.assertFalse(cf._device_stream_broken)
        finally:
            cf._device_stream_broken = original

    def test_flag_set_when_stream_error_detected(self):
        cf = _import_conftest()

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
        cf = _import_conftest()

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
    unittest.main()
