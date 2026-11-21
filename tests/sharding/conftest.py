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

"""Suite-wide guard for the sharding device tests (see device_suite.py).

Counts the tests marked ``sharding_device`` / ``sharding_device_suite`` and,
at the end of the session, checks that they actually executed. On a host
with bound VFs a plain ``pytest tests/sharding/`` runs them through the
subprocess wrappers; if that does not happen the session fails. On a host
that cannot run them (device-less, or PF mode without VFs) the session
stays green but says so in the terminal summary, so the skip is visible in
every log instead of silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from device_env import device_available, mode_usable, promote_vf_mode_to_emulation
from device_suite import (
    DEVICE_MODES,
    device_coverage_verdict,
    session_is_sharding_only,
)

_GUARD_MARKERS = ("sharding_device", "sharding_device_suite")


# Node ids of the guard-marked tests this session collected and executed,
# per marker (the device tests themselves, and the subprocess wrappers).
# A module-level registry is safe here: one pytest process runs one session.
_device_collected: set[str] = set()
_device_executed: set[str] = set()
_suite_collected: set[str] = set()
_suite_executed: set[str] = set()


def pytest_configure(config):
    for marker in _GUARD_MARKERS:
        config.addinivalue_line(
            "markers",
            f"{marker}: internal to tests/sharding — tags tests the "
            f"coverage guard counts (see device_suite.py)",
        )
    # Before collection imports any test module (and so before anything
    # loads the Flex runtime): a sharding-only session on a VF-mode host
    # runs the emulation device instead. In a mixed session (e.g.
    # ``pytest tests/``) the rewrite is skipped — other directories keep
    # the ambient environment, and the wrapper subprocesses still force
    # their own shape for the device tests.
    if session_is_sharding_only(
        config.args, Path(config.invocation_params.dir).resolve()
    ):
        promote_vf_mode_to_emulation()


def pytest_collection_modifyitems(items):
    for item in items:
        if "sharding_device" in item.keywords:
            _device_collected.add(item.nodeid)
        if "sharding_device_suite" in item.keywords:
            _suite_collected.add(item.nodeid)


def pytest_deselected(items):
    # A -k/-m selection that filters tests out is user intent, not silent
    # rot: the guard only judges the tests the session actually selected.
    for item in items:
        _device_collected.discard(item.nodeid)
        _suite_collected.discard(item.nodeid)


def pytest_runtest_logreport(report):
    # Count executions, not passes: a session whose device tests all
    # failed has real failures to show; the guard's "none executed"
    # message would only mislead next to them.
    if report.when == "call" and report.outcome in ("passed", "failed"):
        if report.nodeid in _device_collected:
            _device_executed.add(report.nodeid)
        if report.nodeid in _suite_collected:
            _suite_executed.add(report.nodeid)


def _vf_capable() -> bool:
    """Whether this host could run the device suite right now.

    Mirrors the wrapper's own gate: a device is present and the EMUL1p5
    mode can open it (bound VFs, not a PF-only host). Probed lazily, only
    when the session executed no device tests, because it opens the card
    in a throwaway process (mode_usable caches one probe per shape).
    """
    if not device_available():
        return False
    return mode_usable(DEVICE_MODES["EMUL1p5"], 8)[0]


def pytest_sessionfinish(session, exitstatus):
    import torch

    # Under pytest-xdist every worker collects the whole suite but executes
    # only its share, so a worker that drew no wrapper test would look like
    # "nothing executed" and falsely fail. The controller process runs no
    # tests, so it cannot judge either: the guard is a no-op under xdist.
    # (CI's OOT --parallel is per-file pytest processes across cards, not
    # xdist, so the guard does apply there.)
    if hasattr(session.config, "workerinput"):
        return

    # TORCH_DEVICE_BACKEND_AUTOLOAD=0 (a supported mode, see device_env)
    # leaves torch.spyre unregistered; is_initialized would raise there.
    try:
        runtime_open = torch.spyre.is_initialized()
    except AttributeError:
        runtime_open = False

    verdict, message = device_coverage_verdict(
        device_collected=len(_device_collected),
        device_executed=len(_device_executed),
        suite_collected=len(_suite_collected),
        suite_executed=len(_suite_executed),
        # Lazy: the probe opens the card in a subprocess, so it must not
        # run on a session whose verdict cannot use it (--collect-only,
        # everything executed, an empty selection, or an open runtime —
        # the parent may still hold the card).
        vf_capable=_vf_capable,
        runtime_open=runtime_open,
        collect_only=session.config.option.collectonly,
    )
    if verdict == "pass":
        return
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is not None:
        bold = verdict == "fail"
        terminalreporter.write_sep(
            "=" if bold else "-", message.upper(), red=bold, bold=bold
        )
    if verdict == "fail":
        # Not an assertion in a test (the session is over): fail the run
        # itself, with the message above as the only trace of why.
        # wrap_session returns session.exitstatus after this hook, so the
        # assignment reaches the process exit code.
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
