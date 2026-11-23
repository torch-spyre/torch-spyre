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

"""Shared machinery for the sharding device-test files.

A device test file (``test_sharded_add.py``, ``test_sharded_reduction.py``)
gates its device tests with :func:`requires_emulation` and defines its
``test_device_suite_in_emulation_shape`` with :func:`make_device_suite_test`
instead of repeating that logic. The wrapper re-runs the file in a subprocess
per (mode, shards) shape, so on a host with bound VFs the device tests run
from a plain ``pytest tests/sharding/`` — no extra environment needed.

The suite-wide guard in ``conftest.py`` then checks that they actually did:
a run that collects device tests but executes none cannot pass silently.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

import pytest

from device_env import device_available, emulation_ready, mode_usable

# The Flex device mode that admits sharded allocations (the scheduler
# factory routes it to the multi-chunk scheduler): the EMUL1p5 device
# always admits. "EMUL1p5-small-dma" also lowers the DMA slice size, which
# Flex reads once at runtime start, so each shard's copy becomes several
# DMA slices (256 MiB by default: a test shard would take one).
#
# The small-dma mode reruns the whole file on purpose, not only the
# multi-slice test: every other test whose shards are larger than one slice
# is then also checked with its copies split into several DMA slices.
SMALL_DMA_SLICE_BYTES = 4096
DEVICE_MODES = {
    "EMUL1p5": {"FLEX_DEVICE": "EMUL1p5"},
    "EMUL1p5-small-dma": {
        "FLEX_DEVICE": "EMUL1p5",
        "FLEX_DATA_TRANSFER_BUFFER_SIZE": str(SMALL_DMA_SLICE_BYTES),
    },
}

# Extra child pytest arguments per mode. With the default slice size the
# multi-slice test would only skip, and the wrapper allows no skips in a
# child besides its own tests, so it is deselected there instead.
MODE_PYTEST_ARGS = {
    "EMUL1p5": ("-k", "not test_shards_larger_than_one_dma_slice_round_trip"),
}

_SHARD_SHAPES = (8, 16)


def requires_emulation(test):
    """Gate a device test and tag it for the coverage guard.

    The test skips unless the process itself started in a sharding shape
    (``SENCORES`` in (8, 16) and ``FLEX_DEVICE=EMUL1p5``); the
    ``sharding_device`` marker lets ``conftest.py`` count it, so a run
    where every such test skipped is visible instead of silent.
    """
    test = pytest.mark.sharding_device(
        pytest.mark.skipif(
            not emulation_ready(),
            reason="needs SENCORES in (8, 16) and a sharded-admitting Flex "
            "device mode (FLEX_DEVICE=EMUL1p5) at process start; "
            "test_device_suite_in_emulation_shape runs these in that shape",
        )(test)
    )
    return test


def make_device_suite_test(source_file):
    """Build the file's ``test_device_suite_in_emulation_shape``.

    The wrapper skips when this process cannot host the child runs (no
    device, or the runtime is already open, or the mode cannot open the
    card — a PF-only host has no VFs). On a VF-mode host a plain pytest
    run executes it, and it fails unless the child suites actually ran.
    """

    @pytest.mark.sharding_device_suite
    @pytest.mark.skipif(
        emulation_ready() or not device_available(),
        reason="needs a Spyre device; in an emulation shape the tests below "
        "run directly",
    )
    @pytest.mark.parametrize(
        ("mode", "shards"),
        [(m, s) for m in sorted(DEVICE_MODES) for s in _SHARD_SHAPES],
    )
    def test_device_suite_in_emulation_shape(mode, shards, tmp_path):
        import torch

        # The child opens the card itself, so this process must not hold it.
        if torch.spyre.is_initialized():
            pytest.skip(
                "the Spyre runtime is already open in this process; run "
                f"{source_file} on its own"
            )
        env = dict(os.environ)
        env["SENCORES"] = str(shards)
        env.update(DEVICE_MODES[mode])
        env.pop("PYTEST_ADDOPTS", None)
        # A device being present is not enough: the mode must be able to open
        # it (a PF-only host has no VFs, so EMUL1p5 children die opening the
        # PCI path). Probe in a throwaway process and skip rather than fail.
        usable, reason = mode_usable(DEVICE_MODES[mode], shards)
        if not usable:
            pytest.skip(
                f"Flex device mode {mode} cannot open the card on this "
                f"machine ({reason}); needs bound VFs"
            )
        junit = tmp_path / "child.xml"
        # Output is streamed (not captured) so CI's no-output watchdog sees
        # progress; results are read from the JUnit report.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-p",
                "no:cacheprovider",
                f"--junit-xml={junit}",
                *MODE_PYTEST_ARGS.get(mode, ()),
                source_file,
            ],
            env=env,
            timeout=1200,
        )
        # Parse the report before judging the exit code, so a nonzero exit
        # can name the failing child cases (with their messages) instead of
        # showing only "assert 1 == 0". Child stdout was streamed to this
        # process's output and is in the log above.
        outcomes = {}
        failures = []
        if junit.exists():
            for case in ElementTree.parse(junit).getroot().iter("testcase"):
                children = [c.tag for c in case]
                outcome = next(
                    (t for t in ("failure", "error", "skipped") if t in children),
                    "passed",
                )
                outcomes[case.get("name")] = outcome
                for tag in ("failure", "error"):
                    node = case.find(tag)
                    if node is not None:
                        message = (node.get("message") or "").splitlines()
                        failures.append(
                            f"{case.get('name')}: {message[0] if message else tag}"
                        )
        if result.returncode != 0:
            detail = (
                "; ".join(failures)
                if failures
                else "no failing case in the JUnit report (collection or setup error)"
            )
            raise AssertionError(
                f"child pytest failed for {shards} shards in {mode} mode "
                f"(exit {result.returncode}). {detail}"
            )
        wrappers = {
            f"test_device_suite_in_emulation_shape[{m}-{s}]"
            for m in sorted(DEVICE_MODES)
            for s in _SHARD_SHAPES
        }
        assert {k for k, v in outcomes.items() if v == "skipped"} == wrappers, outcomes
        ran = {k for k, v in outcomes.items() if v == "passed"}
        assert ran and ran | wrappers == set(outcomes), outcomes

    return test_device_suite_in_emulation_shape


def device_coverage_verdict(
    device_collected: int,
    device_executed: int,
    suite_collected: int,
    suite_executed: int,
    vf_capable: bool | Callable[[], bool],
    runtime_open: bool,
    collect_only: bool,
) -> tuple[str, str | None]:
    """What the coverage guard should do for one finished session.

    The ``device_*`` counts are the tests marked ``sharding_device`` (they
    skip in a plain process; the wrapper runs them in children), the
    ``suite_*`` counts the wrappers marked ``sharding_device_suite``.
    ``runtime_open`` says the Spyre runtime is already open in this
    process, in which case the wrapper skips by design (run the file on
    its own).

    ``vf_capable`` says whether this host could run the device suite
    (device present and the EMUL1p5 mode can open it). It may be a
    callable, and if so it is resolved **only on the verdict path that
    needs it**: probing opens the card in a subprocess (up to 180 s, and
    it can contend with a parent that holds the card), so
    ``--collect-only``, sessions that executed their device tests, empty
    device selections, and an open runtime must not pay for it.

    Returns ``("pass", None)`` or ``("warn" | "fail", message)``.

    - ``fail`` — the wrapper was collected on a VF-capable host, nothing
      stopped it, and it still did not execute: the silent-green case the
      guard exists to catch (a broken wrapper or gating).
    - ``warn`` — the host could not run the device tests (no device, or
      PF mode without VFs), or the runtime is already open: the session
      stays green but the skip is stated, so device-less CI logs carry it.
    - ``pass`` — device tests executed, none were collected (a subset
      run), or nothing was supposed to run (``--collect-only``, or a
      ``-k``/``-m`` selection that deselected the wrapper: the in-process
      skip of a hand-picked device test is by design, not silent rot).
    """
    if collect_only:
        return "pass", None
    if device_executed > 0 or suite_executed > 0:
        return "pass", None
    if device_collected == 0 and suite_collected == 0:
        return "pass", None
    if runtime_open:
        return (
            "warn",
            "sharding device tests: none executed (the Spyre runtime "
            "is already open in this process; run the device test "
            "files on their own)",
        )
    host_capable = vf_capable() if callable(vf_capable) else vf_capable
    if host_capable:
        if suite_collected > 0:
            return (
                "fail",
                "sharding device tests: none executed although this host "
                "admits the emulation shape (bound VFs) and the device "
                "suite wrapper was collected — fix the wrapper or the "
                "gating instead of skipping",
            )
        return "pass", None
    if device_collected > 0:
        return (
            "warn",
            f"sharding device tests: 0 of {device_collected} executed (no "
            "Spyre device with bound VFs on this runner; the suite only "
            "ran its device-less tests)",
        )
    return "pass", None


def session_is_sharding_only(args: list[str], invocation_dir: Path) -> bool:
    """Whether a pytest session with these args collects nothing outside
    ``tests/sharding``.

    The VF -> emulation rewrite is only safe when it cannot reach another
    directory's tests: a session that also collects e.g. ``tests/`` as a
    whole would run every test in it emulated at 8 cores. Judge by the
    invocation's paths, resolved against the invocation directory; empty
    args (a bare ``pytest``) collect from the invocation directory itself.
    Collection has not run yet at configure time, so args are the contract:
    a session whose args all name files or directories inside
    ``tests/sharding`` is sharding-only. A node id
    (``tests/sharding/test_x.py::test_y``) counts as inside — its path
    prefix is inside.
    """
    sharding_dir = Path(__file__).resolve().parent
    paths = [Path(a) for a in args] or [invocation_dir]
    for arg in paths:
        resolved = (
            arg.resolve() if arg.is_absolute() else (invocation_dir / arg).resolve()
        )
        if resolved != sharding_dir and sharding_dir not in resolved.parents:
            return False
    return True
