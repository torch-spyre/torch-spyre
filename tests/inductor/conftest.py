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

"""Inductor test suite conftest — ensures OpSpec validation for all tests.

Validation is on by default, but we set the env var explicitly here so tests
remain covered even if the production default changes in the future.  Disable
via SPYRE_VALIDATE_OP_SPECS=0 if profiling test-suite runtime.
"""

import gc
import os

import pytest
import torch
from lx_finalizer_parity import LXFinalizerParity


@pytest.fixture
def lx_finalizer_parity():
    """Audit successful real compiles through both finalizer call sites."""

    with LXFinalizerParity() as audit:
        yield audit


@pytest.fixture(autouse=True)
def _audit_lx_finalizer_corpus(request):
    """Optional whole-suite audit; direct finalizer unit calls are out of scope.

    Set ``SPYRE_AUDIT_LX_FINALIZER_INPUTS=1`` when running the Inductor corpus.
    Tests that instantiate OpSpec/finalizer objects directly have no scheduler
    node and are intentionally not recorded. Real codegen calls are required to
    match a full-node preflight exactly.
    """

    if os.environ.get("SPYRE_AUDIT_LX_FINALIZER_INPUTS") != "1":
        yield
        return
    audit = request.getfixturevalue("lx_finalizer_parity")
    yield
    # A test can intentionally stop compilation before codegen. In that case
    # there is no later call for a successful scheduler preflight to match.
    # Still reject every codegen call that lacked an identical preflight; only
    # the reverse coverage proof is conditional on the test reaching its end.
    audit.assert_codegen_covered()
    fallback_report = audit.fallback_report()
    terminal = request.config.pluginmanager.getplugin("terminalreporter")
    if terminal is not None:
        for line in fallback_report:
            terminal.write_line(line)
    report = getattr(request.node, "lx_finalizer_call_report", None)
    if report is not None and report.passed:
        audit.assert_complete()
        expected = request.node.get_closest_marker("lx_finalizer_fallback_expected")
        if expected is None:
            assert not fallback_report, "\n".join(fallback_report)
        else:
            assert len(expected.args) == 1 and isinstance(expected.args[0], str), (
                "lx_finalizer_fallback_expected requires one reason string"
            )
            expected_count = expected.kwargs.get("count", 1)
            assert isinstance(expected_count, int) and expected_count > 0, (
                "lx_finalizer_fallback_expected count must be a positive integer"
            )
            assert len(fallback_report) == expected_count, (
                "LX finalizer fallback count changed: "
                f"expected {expected_count}, got {len(fallback_report)}; "
                f"records={fallback_report}"
            )
            assert all(expected.args[0] in line for line in fallback_report), (
                "LX finalizer fallback reason changed: "
                f"expected substring {expected.args[0]!r}; records={fallback_report}"
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Expose the call outcome to the finalizer-audit fixture teardown."""

    outcome = yield
    if call.when == "call":
        item.lx_finalizer_call_report = outcome.get_result()


def pytest_runtest_logreport(report):
    """Make long corpus-run failures visible before the final pytest summary."""

    if os.environ.get("SPYRE_AUDIT_LX_FINALIZER_INPUTS") == "1" and report.failed:
        print(f"\nLX_AUDIT_FAILURE {report.nodeid} [{report.when}]\n{report.longrepr}")


def pytest_configure(config):
    """Ensure OpSpec validation is enabled for the inductor test suite."""
    os.environ["SPYRE_VALIDATE_OP_SPECS"] = "1"
    config.addinivalue_line(
        "markers",
        "lx_finalizer_fallback_expected(reason, count=1): this test deliberately "
        "exercises the named number of scheduler LX fallbacks",
    )


_POISON_VALUE = 1234.0

# Several tensors of varied sizes (not one big blob) so the allocator's
# best-fit free-list ends up with poisoned blocks of many different sizes
# scattered across segments -- maximizing the chance that a subsequent
# test's real allocation reuses a poisoned block instead of a lucky
# never-touched (all-zero) one. Total is a few GB: small relative to the
# ~96 GiB of Tensor-usable device HBM, but comfortably larger than any
# individual test's working set in this file.
_POISON_SHAPES_BYTES = [
    1 * 1024**3,
    512 * 1024**2,
    512 * 1024**2,
    256 * 1024**2,
    256 * 1024**2,
    128 * 1024**2,
    128 * 1024**2,
    64 * 1024**2,
]


@pytest.fixture(scope="session", autouse=True)
def _poison_device_hbm():
    """Poison device HBM with non-zero sentinel values before any test runs.

    Defeats the "virgin device HBM reads back as zero" failure mode: a
    kernel bug that reads uninitialized HBM instead of its intended operand
    silently produces a zero-padded (and often coincidentally correct-
    looking) result on a freshly-initialized device, masking the bug until
    some other test happens to leave nonzero data behind first. Poisoning
    once here means every test -- including one run alone -- sees non-zero
    garbage instead, making such bugs fail deterministically. See issue
    #3613 and test_unsqueeze_broadcast_matmul_tile_E_poisoned_correct in
    test_coarse_tile_e2e.py for the specific bug class this guards against.
    """
    poison_tensors = [
        torch.full((nbytes // 2,), _POISON_VALUE, dtype=torch.float16, device="spyre")
        for nbytes in _POISON_SHAPES_BYTES
    ]
    del poison_tensors
    gc.collect()
    yield
