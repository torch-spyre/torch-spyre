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
Tests for the device-error-state mechanism.

- TestStreamErrorBindings: unit-tests the typed _C.SpyreStreamError /
  _C.SpyreDeviceState enums and the associated query functions.
- TestDeviceFaultSetup: calls the setup hook and _is_device_fault directly.
- TestDeviceFaultSession: runs the real hooks in a child pytest session, pinning
  report outcomes, the xfail interplay and the exit code end to end.

Usage: ``python test_device_error_skip.py`` or ``pytest test_device_error_skip.py``
"""

import importlib.util
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
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
SpyreDeviceFault = _tests_conftest.SpyreDeviceFault
_is_device_fault = _tests_conftest._is_device_fault


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


class TestDeviceFaultSetup(TestCase):
    """
    Calls pytest_runtest_setup() and _is_device_fault() directly with mocks.
    """

    def _make_item(self, keywords=()):
        """Return a minimal mock pytest.Item with the given keyword names."""
        item = MagicMock(spec=pytest.Item)
        item.keywords = set(keywords)
        return item

    def test_healthy_device_proceeds(self):
        """When device state is Ok the hook must let the test run."""
        with patch.object(_C, "get_device_state", return_value=_C.SpyreDeviceState.Ok):
            pytest_runtest_setup(self._make_item())

    def test_not_initialized_proceeds(self):
        """When device state is NotInitialized the hook must let the test run."""
        with patch.object(
            _C,
            "get_device_state",
            return_value=_C.SpyreDeviceState.NotInitialized,
        ):
            pytest_runtest_setup(self._make_item())

    def test_faulted_device_errors_instead_of_skipping(self):
        """When device state is StreamError every hook call must error, never skip."""
        with patch.object(
            _C, "get_device_state", return_value=_C.SpyreDeviceState.StreamError
        ):
            for _ in range(3):
                with self.assertRaises(SpyreDeviceFault):
                    pytest_runtest_setup(self._make_item())

    def test_fault_message_names_the_state(self):
        """The error must say the device is in error state and name StreamError."""
        with patch.object(
            _C, "get_device_state", return_value=_C.SpyreDeviceState.StreamError
        ):
            with self.assertRaises(SpyreDeviceFault) as ctx:
                pytest_runtest_setup(self._make_item())
        self.assertIn("Device is in error state", str(ctx.exception))
        self.assertIn("StreamError", str(ctx.exception))

    def test_import_error_does_not_block_test(self):
        """If torch_spyre._C is not importable the hook must silently pass."""
        with patch.dict(sys.modules, {"torch_spyre._C": None}):
            pytest_runtest_setup(self._make_item())

    def test_broken_state_query_does_not_block_test(self):
        """A _C that raises (not just a missing one) must not escape into pytest."""
        with patch.object(_C, "get_device_state", side_effect=RuntimeError("boom")):
            pytest_runtest_setup(self._make_item())

    def _excinfo(self, exc):
        try:
            raise exc
        except type(exc):
            return pytest.ExceptionInfo.from_current()

    def test_ras_hardware_error_is_a_device_fault(self):
        """A RAS hardware error must never be absorbed by an xfail."""
        ras = RuntimeError(_RAS)
        self.assertTrue(_is_device_fault(self._excinfo(ras)))

    def test_text_mentioning_a_ras_error_is_not_a_device_fault(self):
        """Only the runtime's own record counts, not an assertion quoting it."""
        quoted = AssertionError(f"expected no fault, got {_RAS}")
        self.assertFalse(_is_device_fault(self._excinfo(quoted)))
        self.assertFalse(_is_device_fault(self._excinfo(RuntimeError(f"saw {_RAS}"))))

    def test_ordinary_failure_is_not_a_device_fault(self):
        """An unsupported-op error stays eligible for xfail."""
        self.assertFalse(
            _is_device_fault(self._excinfo(RuntimeError("Unsupported: flip")))
        )


_RAS = (
    '{"action":"information","category":"hardware","code":"0x7b1b",'
    '"name":"RAS::RUNTIMESCHEDULER::ComputeHardwareError"}'
)

# The real conftest hooks in a child pytest session, with get_device_state() driven by the
# tests themselves. Appends (not prepends) this dir so a stubbed path can shadow it.
_SESSION_CONFTEST = f"""
import importlib.util, sys
sys.path.append({str(_CONFTEST_PATH.parent)!r})
# torch first: importing torch_spyre first makes torch's backend autoload re-enter it half-initialized.
import torch
from torch_spyre import _C
STATE = {{"s": _C.SpyreDeviceState.Ok}}
_C.get_device_state = lambda: STATE["s"]
_spec = importlib.util.spec_from_file_location("real_conftest", {str(_CONFTEST_PATH)!r})
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
pytest_runtest_setup = _real.pytest_runtest_setup
pytest_runtest_makereport = _real.pytest_runtest_makereport
pytest_sessionfinish = _real.pytest_sessionfinish
"""

_SESSION_TESTS = f"""
import unittest, pytest
from torch_spyre import _C
from conftest import STATE
RAS = {_RAS!r}

@pytest.mark.xfail(reason="unsupported op")
def test_a_ordinary_xfail():
    raise RuntimeError("Unsupported: flip")

@pytest.mark.xfail(reason="unsupported op")
def test_b_ras_under_pytest_xfail():
    raise RuntimeError(RAS)

@pytest.mark.xfail(reason="unsupported op", strict=True)
def test_c_ras_under_strict_xfail():
    raise RuntimeError(RAS)

class TestD(unittest.TestCase):
    def test_d_ras_under_oot_xfail(self):
        raise RuntimeError(RAS)
    test_d_ras_under_oot_xfail.pytestmark = [pytest.mark.xfail(reason="oot")]

@pytest.mark.xfail(reason="expected to fail")
def test_e_xpass_while_the_device_faults():
    STATE["s"] = _C.SpyreDeviceState.StreamError

def test_f_after_the_fault():
    pass
"""

_LAST_TEST_FAULTS = """
import pytest
from torch_spyre import _C
from conftest import STATE

@pytest.fixture(scope="session", autouse=True)
def fault_after_the_last_report():
    yield
    STATE["s"] = _C.SpyreDeviceState.StreamError

def test_only():
    pass
"""


class TestDeviceFaultSession(TestCase):
    """Runs the real hooks inside pytest: xfail rewrites, report outcomes and exit code."""

    def _run(self, tests):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "conftest.py").write_text(_SESSION_CONFTEST)
            Path(d, "test_session.py").write_text(tests)
            xml = Path(d, "out.xml")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-rA",
                    "--import-mode=append",
                    f"--junit-xml={xml}",
                    d,
                ],
                capture_output=True,
                text=True,
                cwd=d,
            )
            if not xml.exists():
                self.fail(
                    f"child pytest exited {proc.returncode} without writing "
                    f"{xml.name}\n--- stdout ---\n{proc.stdout[-4000:]}"
                    f"\n--- stderr ---\n{proc.stderr[-4000:]}"
                )
            outcomes = {}
            for case in ET.parse(xml).getroot().iter("testcase"):
                kinds = [
                    c.tag for c in case if c.tag in ("failure", "error", "skipped")
                ]
                outcomes[case.get("name")] = (kinds[0] if kinds else "passed", case)
        return proc, outcomes

    def test_faults_fail_the_session_and_never_read_as_xfail(self):
        proc, outcomes = self._run(_SESSION_TESTS)
        status = {name: kind for name, (kind, _) in outcomes.items()}
        self.assertEqual(
            status,
            {
                "test_a_ordinary_xfail": "skipped",  # a genuine XFAIL
                "test_b_ras_under_pytest_xfail": "failure",
                "test_c_ras_under_strict_xfail": "failure",
                "test_d_ras_under_oot_xfail": "failure",
                "test_e_xpass_while_the_device_faults": "failure",
                "test_f_after_the_fault": "error",
            },
            proc.stdout,
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        # The XPASS carried no exception, so its failure needs a readable message.
        _, xpass = outcomes["test_e_xpass_while_the_device_faults"]
        self.assertIn("StreamError", xpass.find("failure").get("message", ""))

    def test_fault_after_the_last_test_fails_the_session(self):
        proc, outcomes = self._run(_LAST_TEST_FAULTS)
        self.assertEqual(outcomes["test_only"][0], "passed")
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    run_tests()
