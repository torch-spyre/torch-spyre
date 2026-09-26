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
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import compare_with_cpu  # noqa: E402

try:
    from torch_spyre._inductor import config as _spyre_config

    _max_sencores = _spyre_config.sencores
except ImportError:
    _max_sencores = int(os.getenv("SENCORES", "32"))

# Single-core and chip-max; MULTICORE_SENCORES subset.
MULTICORE_SENCORES = (1, _max_sencores)


def _parametrized_argnames(metafunc):
    names = set()
    for mark in metafunc.definition.iter_markers("parametrize"):
        argnames = mark.args[0]
        if isinstance(argnames, str):
            names.update(a.strip() for a in argnames.split(","))
        else:
            names.update(argnames)
    return names


def pytest_generate_tests(metafunc):
    """Eager: one record, default config. Compiled multi-core: 1 and chip-max."""
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
                pytest.param("compiled", 1, id="1-compiled"),
                pytest.param("compiled", _max_sencores, id=f"{_max_sencores}-compiled"),
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
            [pytest.param(n, id=str(n)) for n in MULTICORE_SENCORES],
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


def xfail_existing(
    execution_mode,
    *,
    patch_sencores=None,
    eager=None,
    compiled=None,
    compiled_32=None,
    compiled_1=None,
):
    """Xfail this mode when the abort already has a GitHub issue.

    eager/compiled/compiled_32/compiled_1 are (issue_number, reason) or None.
    """
    if execution_mode == "eager":
        spec = eager
    elif patch_sencores == _max_sencores and compiled_32 is not None:
        spec = compiled_32
    elif patch_sencores == 1 and compiled_1 is not None:
        spec = compiled_1
    else:
        spec = compiled

    if spec is None:
        return
    issue, reason = spec
    pytest.xfail(reason=f"{reason} See issue #{issue}.")


_xfail_existing = xfail_existing


def compare_mode(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    """Run fn in exactly one execution mode and compare result with CPU.

    Pair with an ``execution_mode`` argument (pytest_generate_tests) so each
    mode is a separate test record — a failure in eager does not prevent
    compiled from running and vice versa.
    """
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
