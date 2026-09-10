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

"""Device-free tests for parallel DXP compilation."""

from concurrent.futures import Future
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import CodeCacheFuture
from torch._inductor.async_compile import shutdown_compile_workers

from torch_spyre._inductor import config as spyre_config
from torch_spyre.execution import async_compile as async_compile_mod


class _RecordingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.futures: list[Future[str]] = []

    def submit(self, fn, *args):
        future = Future[str]()
        self.calls.append((fn, args))
        self.futures.append(future)
        return future


def _runner(name, code_dir, kernel_provenance=None):
    return name, code_dir, kernel_provenance


def test_sdsc_submits_all_dxp_jobs_before_wait():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()
    events = []

    def generate_bundle(name, output_dir, specs, pool_size=0):
        events.append(("bundle", name))

    real_submit = pool.submit

    def submit(fn, *args):
        events.append(("submit", args[0]))
        return real_submit(fn, *args)

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"async_dxp_compile": True, "spyre_kernel_cache": False}),
        patch.object(compiler, "wait_pool_ready"),
        patch.object(compiler, "use_process_pool", return_value=True),
        patch.object(compiler, "process_pool", return_value=pool),
        patch.object(pool, "submit", side_effect=submit),
        patch.object(
            async_compile_mod,
            "get_output_dir",
            side_effect=["/tmp/k0", "/tmp/k1"],
        ),
        patch.object(async_compile_mod, "generate_bundle", side_effect=generate_bundle),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(
            async_compile_mod, "SpyreSDSCKernelRunner", side_effect=_runner
        ) as runner_type,
    ):
        scope = {
            "kernel0": compiler.sdsc("sdsc_0", []),
            "kernel1": compiler.sdsc("sdsc_1", []),
        }

        assert all(isinstance(value, CodeCacheFuture) for value in scope.values())
        assert events == [
            ("bundle", "sdsc_0"),
            ("submit", "sdsc_0"),
            ("bundle", "sdsc_1"),
            ("submit", "sdsc_1"),
        ]
        runner_type.assert_not_called()

        for future in pool.futures:
            future.set_result("compiled")
        compiler.wait(scope)

    assert scope == {
        "kernel0": ("sdsc_0", "/tmp/k0", None),
        "kernel1": ("sdsc_1", "/tmp/k1", None),
    }


def test_async_cache_commit_is_deferred_until_wait():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"async_dxp_compile": True, "spyre_kernel_cache": True}),
        patch.object(compiler, "wait_pool_ready"),
        patch.object(compiler, "use_process_pool", return_value=True),
        patch.object(compiler, "process_pool", return_value=pool),
        patch.object(async_compile_mod, "compute_specs_hash", return_value="key"),
        patch.object(async_compile_mod, "get_cached_kernel_dir", return_value=None),
        patch.object(
            async_compile_mod, "allocate_compile_dir", return_value="/tmp/key.tmp"
        ),
        patch.object(
            async_compile_mod, "commit_compile_dir", return_value="/cache/key"
        ) as commit,
        patch.object(async_compile_mod, "generate_bundle"),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(async_compile_mod, "SpyreSDSCKernelRunner", side_effect=_runner),
    ):
        scope = {"kernel": compiler.sdsc("sdsc_0", [])}
        commit.assert_not_called()

        pool.futures[0].set_result("compiled")
        compiler.wait(scope)

    commit.assert_called_once_with("/tmp/key.tmp", "key")
    assert scope["kernel"] == ("sdsc_0", "/cache/key", None)


def test_async_compile_failure_moves_cache_entry_at_wait():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"async_dxp_compile": True, "spyre_kernel_cache": True}),
        patch.object(compiler, "wait_pool_ready"),
        patch.object(compiler, "use_process_pool", return_value=True),
        patch.object(compiler, "process_pool", return_value=pool),
        patch.object(async_compile_mod, "compute_specs_hash", return_value="key"),
        patch.object(async_compile_mod, "get_cached_kernel_dir", return_value=None),
        patch.object(
            async_compile_mod, "allocate_compile_dir", return_value="/tmp/key.tmp"
        ),
        patch.object(async_compile_mod, "generate_bundle"),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(async_compile_mod, "_move_to_failed_dir") as move_failed,
    ):
        scope = {"kernel": compiler.sdsc("sdsc_0", [])}
        pool.futures[0].set_exception(RuntimeError("DXP failed"))

        with pytest.raises(RuntimeError, match="DXP failed"):
            compiler.wait(scope)

    move_failed.assert_called_once_with("/tmp/key.tmp")


def test_real_subprocess_pool_runs_dxp_jobs_concurrently(tmp_path: Path):
    """Two DXP jobs must overlap rather than running serially in the parent."""
    bin_dir = tmp_path / "bin"
    marker_dir = tmp_path / "markers"
    compile_dirs = [tmp_path / "kernel0", tmp_path / "kernel1"]
    bin_dir.mkdir()
    marker_dir.mkdir()
    for compile_dir in compile_dirs:
        compile_dir.mkdir()

    fake_dxp = bin_dir / "dxp_standalone"
    fake_dxp.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "marker_dir = Path(os.environ['FAKE_DXP_MARKER_DIR'])\n"
        "(marker_dir / Path(sys.argv[2]).name).touch()\n"
        "deadline = time.monotonic() + 10\n"
        "while len(list(marker_dir.iterdir())) < 2:\n"
        "    if time.monotonic() >= deadline:\n"
        "        raise SystemExit('DXP jobs did not overlap')\n"
        "    time.sleep(0.05)\n"
    )
    fake_dxp.chmod(0o755)

    shutdown_compile_workers()
    try:
        with (
            torch._inductor.config.patch(
                {"compile_threads": 2, "worker_start_method": "subprocess"}
            ),
            spyre_config.patch(  # type: ignore[attr-defined]
                {"async_dxp_compile": True}
            ),
            patch.dict(
                os.environ,
                {
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "FAKE_DXP_MARKER_DIR": str(marker_dir),
                },
            ),
        ):
            compiler = async_compile_mod.SpyreAsyncCompile()
            tasks = [
                compiler._submit_dxp(f"sdsc_{index}", str(compile_dir))
                for index, compile_dir in enumerate(compile_dirs)
            ]
            assert all(task is not None for task in tasks)
            for task in tasks:
                assert task is not None
                task.result(timeout=30)
    finally:
        shutdown_compile_workers()

    assert {path.name for path in marker_dir.iterdir()} == {"kernel0", "kernel1"}
