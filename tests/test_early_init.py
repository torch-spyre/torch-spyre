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

"""TORCH_SPYRE_EARLY_INIT starts the device runtime on a background thread."""

import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import torch_spyre


class _FakeImpl:
    def __init__(self, initialized=False, bad_fork=False, fail=False):
        self._initialized = initialized
        self._bad_fork = bad_fork
        self._fail = fail
        self.started = threading.Event()
        self.thread_name = None

    def _is_in_bad_fork(self):
        return self._bad_fork

    def _lazy_init(self):
        self.thread_name = threading.current_thread().name
        self.started.set()
        if self._fail:
            raise RuntimeError("no device")


def _run(env_value, impl):
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_EARLY_INIT": env_value}),
        mock.patch.object(torch, "spyre", SimpleNamespace(_impl=impl)),
    ):
        torch_spyre._maybe_start_runtime_early()


def test_switch_on_starts_the_runtime_on_a_background_thread():
    impl = _FakeImpl()
    _run("1", impl)
    assert impl.started.wait(10)
    assert impl.thread_name == "spyre-early-init"


@pytest.mark.parametrize(
    "env_value,impl",
    [
        ("0", _FakeImpl()),
        ("", _FakeImpl()),
        ("1", _FakeImpl(initialized=True)),
        ("1", _FakeImpl(bad_fork=True)),
    ],
    ids=["off", "empty", "already-started", "forked-child"],
)
def test_no_start(env_value, impl):
    _run(env_value, impl)
    assert not impl.started.wait(0.5)


def test_a_failed_background_start_only_warns(caplog):
    impl = _FakeImpl(fail=True)
    _run("1", impl)
    assert impl.started.wait(10)
    for thread in threading.enumerate():
        if thread.name == "spyre-early-init":
            thread.join(10)
    assert "TORCH_SPYRE_EARLY_INIT" in caplog.text


@pytest.mark.parametrize(
    "process_name,starts",
    [
        ("EngineCore", True),
        ("MainProcess", False),
        ("ApiServer_0", False),
        ("EngineCore_DP0", False),
    ],
)
def test_a_process_name_starts_only_that_process(process_name, starts):
    impl = _FakeImpl()
    with mock.patch(
        "multiprocessing.current_process",
        return_value=SimpleNamespace(name=process_name),
    ):
        _run("EngineCore", impl)
    assert impl.started.wait(10 if starts else 0.5) == starts


_SPAWN_CHECK = textwrap.dedent(
    """
    import multiprocessing
    import os

    import torch
    import torch_spyre

    # Set only after import, so no process here opens the device.
    os.environ["TORCH_SPYRE_EARLY_INIT"] = "EngineCore"
    AT_IMPORT = torch_spyre._early_init_requested()
    del os.environ["TORCH_SPYRE_EARLY_INIT"]


    def report(queue):
        queue.put((multiprocessing.current_process().name, AT_IMPORT))


    if __name__ == "__main__":
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        for name in ("EngineCore", "ApiServer_0"):
            p = ctx.Process(target=report, args=(queue,), name=name)
            p.start()
            print(*queue.get())
            p.join()
        print("MainProcess", AT_IMPORT)
    """
)


def test_a_spawned_child_has_its_name_while_it_imports(tmp_path):
    script = tmp_path / "spawn_check.py"
    script.write_text(_SPAWN_CHECK)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = proc.stdout.split("\n")
    assert "EngineCore True" in out
    assert "ApiServer_0 False" in out
    assert "MainProcess False" in out
