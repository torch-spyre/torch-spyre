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

"""
Tests for the device-error-state skip mechanism.

- TestStreamErrorBindings: unit-tests the typed _C.SpyreStreamError /
  _C.SpyreDeviceState enums and the associated query functions.
- TestDeviceErrorSkipIntegration: calls the conftest hook directly to verify
  skip behaviour without spawning subprocesses.

Usage: ``python test_device_error_skip.py`` or ``pytest test_device_error_skip.py``
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre import _C

# Load tests/conftest.py by explicit path so we always get the right module
# regardless of sys.path ordering or the presence of a root-level conftest.py.
_CONFTEST_PATH = Path(__file__).parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("tests.conftest", _CONFTEST_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"Could not load conftest from {_CONFTEST_PATH}"
)
_tests_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tests_conftest)  # type: ignore[union-attr]
pytest_runtest_setup = _tests_conftest.pytest_runtest_setup


class TestStreamErrorBindings(TestCase):
    """Unit tests for the typed SpyreStreamError / SpyreDeviceState bindings."""

    # Testing SpyreStreamError

    def test_stream_error_enum_members_exist(self):
        """SpyreStreamError must expose Success and Shutdown members."""
        self.assertIsInstance(_C.SpyreStreamError.Success, _C.SpyreStreamError)
        self.assertIsInstance(_C.SpyreStreamError.Shutdown, _C.SpyreStreamError)

    def test_stream_error_integer_values(self):
        """SpyreStreamError values must match the documented ABI (Success=0, Shutdown=1)."""
        self.assertEqual(int(_C.SpyreStreamError.Success), 0)
        self.assertEqual(int(_C.SpyreStreamError.Shutdown), 1)

    def test_stream_error_names(self):
        """SpyreStreamError .name must return the enum member's string name."""
        self.assertEqual(_C.SpyreStreamError.Success.name, "Success")
        self.assertEqual(_C.SpyreStreamError.Shutdown.name, "Shutdown")

    # Testing SpyreDeviceState

    def test_device_state_enum_members_exist(self):
        """SpyreDeviceState must expose Ok, NotInitialized, and StreamError."""
        self.assertIsInstance(_C.SpyreDeviceState.Ok, _C.SpyreDeviceState)
        self.assertIsInstance(_C.SpyreDeviceState.NotInitialized, _C.SpyreDeviceState)
        self.assertIsInstance(_C.SpyreDeviceState.StreamError, _C.SpyreDeviceState)

    def test_device_state_integer_values(self):
        """SpyreDeviceState values must match the ABI (Ok=0, NotInitialized=1, StreamError=2)."""
        self.assertEqual(int(_C.SpyreDeviceState.Ok), 0)
        self.assertEqual(int(_C.SpyreDeviceState.NotInitialized), 1)
        self.assertEqual(int(_C.SpyreDeviceState.StreamError), 2)

    def test_device_state_names(self):
        """SpyreDeviceState .name must return the enum member's string name."""
        self.assertEqual(_C.SpyreDeviceState.Ok.name, "Ok")
        self.assertEqual(_C.SpyreDeviceState.NotInitialized.name, "NotInitialized")
        self.assertEqual(_C.SpyreDeviceState.StreamError.name, "StreamError")

    # Testing get_device_state()

    def test_get_device_state_returns_device_state(self):
        """get_device_state() must be importable and return a SpyreDeviceState."""
        result = _C.get_device_state()
        self.assertIsInstance(result, _C.SpyreDeviceState)

    def test_get_device_state_healthy(self):
        """get_device_state() returns Ok when mocked healthy."""
        with patch.object(_C, "get_device_state", return_value=_C.SpyreDeviceState.Ok):
            self.assertEqual(_C.get_device_state(), _C.SpyreDeviceState.Ok)

    def test_get_device_state_faulted(self):
        """get_device_state() returns StreamError when mocked faulted."""
        with patch.object(
            _C, "get_device_state", return_value=_C.SpyreDeviceState.StreamError
        ):
            self.assertEqual(_C.get_device_state(), _C.SpyreDeviceState.StreamError)

    def test_get_device_state_not_initialized(self):
        """get_device_state() returns NotInitialized when mocked pre-init."""
        with patch.object(
            _C,
            "get_device_state",
            return_value=_C.SpyreDeviceState.NotInitialized,
        ):
            self.assertEqual(_C.get_device_state(), _C.SpyreDeviceState.NotInitialized)

    def test_get_device_state_not_cached(self):
        """Consecutive calls reflect live state, not a cached value."""
        states = [
            _C.SpyreDeviceState.Ok,
            _C.SpyreDeviceState.StreamError,
            _C.SpyreDeviceState.Ok,
        ]
        with patch.object(_C, "get_device_state", side_effect=states):
            self.assertEqual(_C.get_device_state(), _C.SpyreDeviceState.Ok)
            self.assertEqual(_C.get_device_state(), _C.SpyreDeviceState.StreamError)
            self.assertEqual(_C.get_device_state(), _C.SpyreDeviceState.Ok)

    # Testing stream_get_error() / stream_get_error_string()

    def test_stream_get_error_returns_stream_error(self):
        """stream_get_error() must return a SpyreStreamError."""
        mock_stream = MagicMock()
        with patch.object(
            _C, "stream_get_error", return_value=_C.SpyreStreamError.Success
        ):
            result = _C.stream_get_error(mock_stream)
        self.assertIsInstance(result, _C.SpyreStreamError)

    def test_error_string_success(self):
        """stream_get_error_string(Success) == 'Success'."""
        self.assertEqual(
            _C.stream_get_error_string(_C.SpyreStreamError.Success), "Success"
        )

    def test_error_string_shutdown(self):
        """stream_get_error_string(Shutdown) == 'Shutdown'."""
        self.assertEqual(
            _C.stream_get_error_string(_C.SpyreStreamError.Shutdown), "Shutdown"
        )


class TestDeviceErrorSkipIntegration(TestCase):
    """
    Calls pytest_runtest_setup() directly with a mock item to verify the
    skip hook.
    """

    def _make_item(self, keywords=()):
        """Return a minimal mock pytest.Item with the given keyword names."""
        item = MagicMock(spec=pytest.Item)
        item.keywords = set(keywords)
        return item

    def test_healthy_device_does_not_skip(self):
        """When device state is Ok the hook must not skip."""
        with patch.object(_C, "get_device_state", return_value=_C.SpyreDeviceState.Ok):
            # Should complete without raising pytest.skip.Exception
            pytest_runtest_setup(self._make_item())

    def test_not_initialized_does_not_skip(self):
        """When device state is NotInitialized the hook must not skip (proceed)."""
        with patch.object(
            _C,
            "get_device_state",
            return_value=_C.SpyreDeviceState.NotInitialized,
        ):
            pytest_runtest_setup(self._make_item())

    def test_faulted_device_skips(self):
        """When device state is StreamError every hook call must skip."""
        with patch.object(
            _C, "get_device_state", return_value=_C.SpyreDeviceState.StreamError
        ):
            for _ in range(3):
                with self.assertRaises(pytest.skip.Exception):
                    pytest_runtest_setup(self._make_item())

    def test_skip_message_contains_device_is_in_error_state(self):
        """The skip reason must contain 'Device is in error state'."""
        with patch.object(
            _C, "get_device_state", return_value=_C.SpyreDeviceState.StreamError
        ):
            with self.assertRaises(pytest.skip.Exception) as ctx:
                pytest_runtest_setup(self._make_item())
        self.assertIn("Device is in error state", str(ctx.exception))

    def test_skip_message_contains_error_class_name(self):
        """The skip reason must name the error class (e.g. 'StreamError')."""
        with patch.object(
            _C, "get_device_state", return_value=_C.SpyreDeviceState.StreamError
        ):
            with self.assertRaises(pytest.skip.Exception) as ctx:
                pytest_runtest_setup(self._make_item())
        self.assertIn("StreamError", str(ctx.exception))

    def test_import_error_does_not_block_test(self):
        """If torch_spyre._C is not importable the hook must silently pass."""
        with patch.dict(sys.modules, {"torch_spyre._C": None}):
            pytest_runtest_setup(self._make_item())


if __name__ == "__main__":
    run_tests()
