# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Shared fixtures and helpers for the indirect_access test suite.

All helpers are plain functions imported explicitly by test modules.

SENCORES is an Inductor compiled work-division knob. Eager does not use it.

Collection (pytest_generate_tests):
  - execution_mode only → [eager], [compiled] (default inductor config, no patch)
  - execution_mode + patch_sencores/sencores → [eager] (no patch),
    [1-compiled], [<chip-max>-compiled]
  - sencores only (compile-only modules) → [1], [<chip-max>]

Eager is never collected as [1-eager] / [32-eager]. Request patch_sencores
only when the compiled dest can split across cores.

A test whose compiled rows only record a gap (rejection, xfail or
``dynamic=True`` refusal) with the same error at every core count is tagged
``@gap_one_core``: it is collected for 1 core only, because the chip-max row
would repeat the 1-core row.
"""

import functools
import os
import regex as re

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

from utils_inductor import compare_with_cpu, expects_raise, strict_xfail

try:
    from torch_spyre._inductor import config as _spyre_config

    _max_sencores = _spyre_config.sencores
except ImportError:
    _max_sencores = int(os.getenv("SENCORES", "32"))

# Single-core and chip-max; MULTICORE_SENCORES subset.
MULTICORE_SENCORES = (1, _max_sencores)


@pytest.fixture(scope="module", autouse=True)
def _raised_dynamo_recompile_limits():
    """Raise Dynamo's recompile limits for this directory only, then restore them.

    Many shape and dtype variants compile in one process, which exceeds the
    production limit. ``config.patch`` puts the old values back when the module
    ends, so a run such as ``pytest tests/inductor/`` does not change the
    settings of other tests. OpSpec validation needs no setting here:
    ``tests/inductor/conftest.py`` already enables it for this directory.
    """
    with torch._dynamo.config.patch(
        recompile_limit=65536, accumulated_recompile_limit=65536
    ):
        yield


def _parametrized_argnames(metafunc):
    names = set()
    for mark in metafunc.definition.iter_markers("parametrize"):
        argnames = mark.args[0]
        if isinstance(argnames, str):
            names.update(a.strip() for a in argnames.split(","))
        else:
            names.update(argnames)
    return names


def gap_one_core(test):
    """Collect a gap test for 1 core only.

    Use it when every compiled row of the test records the same gap at 1 core
    and at chip-max (same rejection or xfail, same message), so the chip-max row
    adds nothing. Once the gap is fixed at 1 core, remove the tag so the test
    runs at both core counts again.
    """
    test._gap_one_core = True
    return test


def pytest_generate_tests(metafunc):
    """Eager: one record, default config. Compiled multi-core: 1 and chip-max."""
    cores = (
        (1,)
        if getattr(metafunc.function, "_gap_one_core", False)
        else MULTICORE_SENCORES
    )
    existing = _parametrized_argnames(metafunc)
    has_mode = "execution_mode" in metafunc.fixturenames
    has_sc = (
        "sencores" in metafunc.fixturenames or "patch_sencores" in metafunc.fixturenames
    )
    if "execution_mode" in existing or "sencores" in existing:
        return

    if has_mode and has_sc:
        metafunc.parametrize(
            "execution_mode,sencores",
            [
                pytest.param("eager", None, id="eager"),
                *(pytest.param("compiled", n, id=f"{n}-compiled") for n in cores),
            ],
        )
    elif has_mode:
        metafunc.parametrize(
            "execution_mode",
            [
                pytest.param("eager", id="eager"),
                pytest.param("compiled", id="compiled"),
            ],
        )
    elif has_sc:
        metafunc.parametrize(
            "sencores",
            [pytest.param(n, id=str(n)) for n in cores],
        )


@pytest.fixture
def patch_sencores(sencores):
    """Patch inductor SENCORES for compiled multi-core cases only.

    sencores is None on eager — yield without config.patch so eager uses the
    process default and is not a SENCORES variant.
    """
    if sencores is None:
        yield None
        return
    from torch_spyre._inductor import config

    with config.patch({"sencores": sencores}):
        yield sencores


_current_request = None


@pytest.fixture(autouse=True)
def _track_current_test(request):
    """Track the current test request so xfail_existing can use strict_xfail."""
    global _current_request
    _current_request = request
    yield
    _current_request = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Strict error validation: an xfail only counts if the message matches.

    strict_xfail (AGENTS.md Test Invariants) checks the exception *type* via
    raises=. That still lets a different failure of the same type hide behind
    the xfail. xfail_existing records the expected message pattern on the item;
    here, a failure whose message does not match is turned into a real FAILED
    so the stale or wrong reason is noticed. XPASS stays a strict failure.

    Why a hook and not ``@expects_raise`` with ``pytest.raises(match=...)``
    (AGENTS.md): every test here runs in eager, 1-core compiled and chip-max
    compiled mode, and the expected outcome differs per mode. Eager can raise
    ``Could not run 'aten::X'`` while the compiled rows pass, or the reverse. One
    ``xfail_existing(...)`` call at the top of the test declares the expectation
    for each mode, and this hook applies it without wrapping the body in
    ``pytest.raises`` under an ``if execution_mode`` at about 120 call sites. The
    semantics match ``pytest.raises(match=...)``: the test passes only on the
    documented message, and the failure text says when the gap is fixed.
    ``expect_dynamic_true_refused`` has one expectation for every mode, so it
    uses the standard form (``pytest.raises`` plus ``expects_raise``).
    """
    outcome = yield
    rep = outcome.get_result()
    reject = getattr(item, "_reject_match", None)
    if (
        call.when == "call"
        and reject is not None
        and rep.outcome != "skipped"
        and not (call.excinfo and call.excinfo.errisinstance(pytest.skip.Exception))
    ):
        if call.excinfo is None:
            rep.outcome = "failed"
            rep.longrepr = (
                "Strict rejection: the test no longer raises.\n"
                f"  documented rejection: {item._reject_reason}\n"
                f"  expected message to match: {reject!r}\n"
                "  The gap looks fixed: turn this into a positive test and remove "
                "the rejection."
            )
        else:
            message = str(call.excinfo.value)
            if re.search(reject, message, re.DOTALL):
                rep.outcome = "passed"
                rep.longrepr = None
            else:
                rep.outcome = "failed"
                rep.longrepr = (
                    "Strict rejection: the test raised, but not the error its "
                    f"reason documents.\n  expected message to match: {reject!r}\n"
                    f"  documented rejection: {item._reject_reason}\n"
                    f"  actual: {type(call.excinfo.value).__name__}: {message[:600]}"
                )
        return
    pattern = getattr(item, "_xfail_match", None)
    if (
        call.when != "call"
        or pattern is None
        or call.excinfo is None
        or not hasattr(rep, "wasxfail")
    ):
        return
    message = str(call.excinfo.value)
    if re.search(pattern, message, re.DOTALL):
        return
    del rep.wasxfail
    rep.outcome = "failed"
    rep.longrepr = (
        "Strict error validation: the test failed, but not with the error its "
        f"xfail reason documents.\n  expected message to match: {pattern!r}\n"
        f"  xfail reason: {item._xfail_reason}\n"
        f"  actual: {type(call.excinfo.value).__name__}: {message[:600]}"
    )


class _Rejection(tuple):
    """An (issue, reason, match) spec for a stable rejection, not a bug.

    AGENTS.md: a missing feature that fails with a stable error (``Could not run
    'aten::X'``, ``Unsupported``, a layout rejection) is an assertion, not an
    xfail. ``xfail_existing`` registers a ``_Rejection`` spec as one: the test
    passes only if the body raises an error whose message matches, and fails if it
    no longer raises (the gap is fixed: make the test positive) or raises
    something else. Wrong values and dbo-opt crashes stay plain strict xfails.
    """

    __slots__ = ()


def rejection(issue, reason, match):
    """Build the spec of a stable rejection; see ``_Rejection``."""
    return _Rejection((issue, reason, match))


def xfail_existing(
    execution_mode,
    *,
    patch_sencores=None,
    eager=None,
    compiled=None,
    compiled_32=None,
    compiled_1=None,
    raises=(NotImplementedError, RuntimeError, AssertionError),
):
    """Register expected failure using torch-spyre's built-in strict_xfail.

    eager/compiled/compiled_32/compiled_1 are (issue, reason, match) or None.
    ``issue`` is an open GitHub issue number. ``match`` is a regex that must
    match the exception message; a different failure of an allowed type is
    reported as FAILED (see pytest_runtest_makereport above), and a pass is a
    strict XPASS failure.

    A spec built with ``rejection(...)`` is asserted instead: it passes when the
    body raises a matching error and fails when it does not raise.
    """
    global _current_request
    if execution_mode == "eager":
        spec = eager
    elif patch_sencores == _max_sencores and compiled_32 is not None:
        spec = compiled_32
    elif patch_sencores == 1 and compiled_1 is not None:
        spec = compiled_1
    else:
        spec = compiled

    if spec is not None and _current_request is not None:
        issue, reason, match = spec
        issue_str = (
            f"See issue #{issue}."
            if isinstance(issue, int)
            else (f"({issue})" if issue else "")
        )
        reason_msg = f"{reason} {issue_str}".strip()
        if isinstance(spec, _Rejection):
            _current_request.node._reject_match = match
            _current_request.node._reject_reason = reason_msg
            return
        _current_request.node._xfail_match = match
        _current_request.node._xfail_reason = reason_msg
        strict_xfail(
            _current_request,
            raises=raises,
            reason=reason_msg,
        )


_xfail_existing = xfail_existing

# torch.compile(dynamic=True) is refused on Spyre (#5578, _refuse_dynamic_true in
# torch_spyre/_inductor/patches.py), before any lowering runs.
_DYNAMIC_TRUE_REFUSAL = r"does not support: torch\.compile\(dynamic=True\)"


def expect_dynamic_true_refused(test):
    """Assert that a test which compiles with ``dynamic=True`` is refused.

    The refusal is deliberate, so it is an assertion and not an xfail (AGENTS.md,
    "Say exactly what is expected"). The body stays the positive test it would be
    if the refusal were lifted: the first compile must raise
    ``BackendCompilerFailed`` with the refusal text, and a different error, or no
    error, fails the test. The supported route for a varying dimension is
    ``x.to("spyre", dynamic={dim: dict(min=..., max=...)})`` with the default
    ``dynamic=None``.
    """

    @functools.wraps(test)
    def refused(*args, **kwargs):
        with pytest.raises(BackendCompilerFailed, match=_DYNAMIC_TRUE_REFUSAL):
            test(*args, **kwargs)

    return expects_raise(refused)


_DL16_MAX = 65472.0  # 1023/1024 * 2**16, largest finite dl16 value


def dl16_snap(x):
    """Round fp16 values to what a Spyre ``.to(device).cpu()`` round trip returns.

    fp16 is stored on device as dl16 (SEN169_FP16), which keeps one mantissa bit
    fewer than IEEE fp16 (9 stored vs 10), so a transfer rounds each value to 10
    significant bits.  Ties round away from zero (measured on device over every
    finite fp16 value; ties-to-even matches only 75.8%).  This is the "transfer
    floor" in docs/source/user_guide/debugging/op_spec_lab.md, not a bug.

    Rounding saturates at 65472, the largest dl16 value: the device returns
    +-65472 for +-65504 (measured), where plain rounding would give 65536.

    The function is idempotent, leaves 0, inf and nan unchanged, and commutes
    with data movement: snap(x)[i] == snap(x[i]).

    Tensors of any other dtype (bf16, fp32, int, bool) round-trip losslessly
    and are returned unchanged.  Tuples and lists are handled element-wise.
    """
    if isinstance(x, (tuple, list)):
        return type(x)(dl16_snap(e) for e in x)
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float16:
        return x
    mant, exp = torch.frexp(x.float())
    scaled = mant * 1024
    rounded = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    snapped = torch.ldexp(rounded / 1024, exp)
    snapped = snapped.clamp(-_DL16_MAX, _DL16_MAX)
    return torch.where(torch.isfinite(x), snapped, x.float()).to(torch.float16)


def compare_mode(execution_mode, fn, *args, atol, rtol):
    """Run fn in exactly one execution mode and compare result with CPU.

    ``atol``/``rtol`` are required: use 0 for pure data movement (gather, take,
    index_select, embedding lookup, scatter without arithmetic), and the dtype
    tolerance only for compute that follows the gather.

    With ``atol == rtol == 0`` the comparison stays bit-exact.  fp16 inputs are
    first snapped to the dl16 transfer floor (see ``dl16_snap``), because an
    fp16 value cannot survive the host/device round trip unchanged; with
    snapped inputs a correct gather returns bit-identical values on CPU and
    device, while a wrong row, a dropped or a duplicated element still differs
    by far more than the floor and fails.

    Executes the full graph on Spyre hardware without manual try/except.
    Torch-Spyre's strict_xfail manages the expected failure validation.
    """
    if atol == 0 and rtol == 0:
        args = dl16_snap(args)
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def compiled_code(fn, *args):
    """Compile fn with Inductor, run it, and return (result, code_string)."""
    from torch._inductor.utils import run_and_get_code

    compiled_fn = torch.compile(fn, backend="inductor")
    result, code_list = run_and_get_code(compiled_fn, *args)
    return result, "\n".join(code_list)
