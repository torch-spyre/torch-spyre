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

"""Device-free tests for parallel backend compilation."""

from concurrent.futures import Future
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import CodeCacheFuture
from torch._inductor.async_compile import shutdown_compile_workers

from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre.execution import async_compile as async_compile_mod
from torch_spyre.execution import kernel_cache


class _RecordingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.futures: list[Future[str]] = []

    def submit(self, fn, *args):
        future = Future[str]()
        self.calls.append((fn, args))
        self.futures.append(future)
        return future


def _runner(name, code_dir, kernel_provenance=None, symbol_kinds=None):
    return name, code_dir, kernel_provenance, symbol_kinds


def _load_config(monkeypatch):
    # Exercise import-time environment resolution without replacing the config
    # module used by other tests or launching a fresh Python for every spelling.
    name = "_test_loop_unroll_config"
    spec = importlib.util.spec_from_file_location(name, spyre_config.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("0", False),
        ("false", False),
        (" NO ", False),
        ("1", True),
        ("TRUE", True),
        ("yes", True),
    ],
)
def test_loop_unroll_environment_and_cache_key(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SPYRE_BACKEND_LOOP_UNROLL", raising=False)
    else:
        monkeypatch.setenv("SPYRE_BACKEND_LOOP_UNROLL", value)
    config = _load_config(monkeypatch)
    assert config.backend_loop_unroll is expected

    with (
        patch.object(
            kernel_cache, "_get_backend_compiler_version", return_value="test"
        ),
        patch.object(kernel_cache, "_get_torch_spyre_version", return_value="test"),
        spyre_config.patch(backend_loop_unroll=config.backend_loop_unroll),
    ):
        resolved = kernel_cache.compute_specs_hash([])
        assert resolved == kernel_cache.compute_specs_hash(
            [], backend_loop_unroll=expected
        )
        assert resolved != kernel_cache.compute_specs_hash(
            [], backend_loop_unroll=not expected
        )


@pytest.mark.parametrize("value", ["", "2", "invalid"])
def test_loop_unroll_rejects_invalid_environment(monkeypatch, value):
    monkeypatch.setenv("SPYRE_BACKEND_LOOP_UNROLL", value)
    with pytest.raises(ValueError, match="SPYRE_BACKEND_LOOP_UNROLL must be"):
        _load_config(monkeypatch)


@pytest.mark.parametrize("unroll", [False, True])
def test_backend_compiler_forwards_loop_unroll_control(tmp_path, unroll):
    code_dir = tmp_path / "spyreCodeDir"
    code_dir.mkdir()
    (code_dir / "spyrecode.json").write_text("{}")
    # A worker's environment and imported config may disagree with the parent.
    env = {"SPYRE_BACKEND_LOOP_UNROLL": str(int(not unroll))}
    with (
        spyre_config.patch(backend_loop_unroll=not unroll),
        patch.object(async_compile_mod, "_check_backend_compiler_on_path"),
        patch.object(async_compile_mod.subprocess, "run") as run,
    ):
        assert async_compile_mod._run_backend_compiler(
            "kernel", str(tmp_path), env, unroll
        ) == str(tmp_path)
    args = run.call_args.args[0]
    flags = [arg for arg in args if arg.startswith("--enable-loop-unroll=")]
    assert flags == [f"--enable-loop-unroll={int(unroll)}"]
    assert run.call_args.kwargs["env"] == env


@pytest.mark.parametrize("unroll", [False, True])
def test_single_worker_compiles_inline_without_starting_pool(unroll):
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        torch._inductor.config.patch({"compile_threads": 1}),
        spyre_config.patch(backend_loop_unroll=unroll),
        patch.object(compiler, "wait_pool_ready") as wait_ready,
        patch.object(compiler, "process_pool") as pool,
        patch.object(async_compile_mod, "_run_backend_compiler") as compile_backend,
    ):
        task = compiler._submit_backend_compile("sdsc_0", "/tmp/kernel")

    assert task is None
    wait_ready.assert_not_called()
    pool.assert_not_called()
    compile_backend.assert_called_once_with(
        "sdsc_0", "/tmp/kernel", dict(os.environ), unroll
    )


def test_sdsc_submits_all_backend_jobs_before_wait():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()
    events = []
    provenance = object()

    def generate_bundle(name, output_dir, specs, pool_size=0):
        events.append(("bundle", name))
        return []

    real_submit = pool.submit

    def submit(fn, *args):
        events.append(("submit", args[0]))
        return real_submit(fn, *args)

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"spyre_kernel_cache": False}),
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
            async_compile_mod,
            "build_kernel_provenance_descriptor",
            return_value=provenance,
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
        "kernel0": ("sdsc_0", "/tmp/k0", provenance, []),
        "kernel1": ("sdsc_1", "/tmp/k1", provenance, []),
    }


def test_async_cache_commit_is_deferred_until_wait():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()
    fake_symbol_kinds = [SymbolKind.kernel(0), SymbolKind.kernel(1)]

    def generate_bundle(*args, **kwargs):
        # Changing config after hashing must not change the submitted setting.
        spyre_config.backend_loop_unroll = True
        return fake_symbol_kinds

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"spyre_kernel_cache": True, "backend_loop_unroll": False}),
        patch.object(compiler, "wait_pool_ready"),
        patch.object(compiler, "use_process_pool", return_value=True),
        patch.object(compiler, "process_pool", return_value=pool),
        patch.object(
            async_compile_mod, "compute_specs_hash", return_value="key"
        ) as hash_specs,
        patch.object(async_compile_mod, "get_cached_kernel_dir", return_value=None),
        patch.object(
            async_compile_mod, "allocate_compile_dir", return_value="/tmp/key.tmp"
        ),
        patch.object(
            async_compile_mod, "commit_compile_dir", return_value="/cache/key"
        ) as commit,
        patch.object(async_compile_mod, "generate_bundle", side_effect=generate_bundle),
        patch.object(async_compile_mod, "save_symbol_kinds"),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(async_compile_mod, "SpyreSDSCKernelRunner", side_effect=_runner),
    ):
        scope = {"kernel": compiler.sdsc("sdsc_0", [])}
        assert hash_specs.call_args.kwargs["backend_loop_unroll"] is False
        assert pool.calls[0][1][-1] is False
        commit.assert_not_called()

        pool.futures[0].set_result("compiled")
        compiler.wait(scope)

    commit.assert_called_once_with("/tmp/key.tmp", "key")
    assert scope["kernel"] == ("sdsc_0", "/cache/key", None, fake_symbol_kinds)


def test_cache_hit_reloads_symbol_kinds_from_miss(tmp_path: Path):
    compiler = async_compile_mod.SpyreAsyncCompile()
    compile_dir = str(tmp_path / "key.tmp")
    Path(compile_dir).mkdir()
    fake_symbol_kinds = [SymbolKind.kernel(0), SymbolKind.kernel(2)]

    with (
        torch._inductor.config.patch({"compile_threads": 1}),
        spyre_config.patch({"spyre_kernel_cache": True}),  # type: ignore[attr-defined]
        patch.object(async_compile_mod, "compute_specs_hash", return_value="key"),
        patch.object(
            async_compile_mod,
            "get_cached_kernel_dir",
            side_effect=[None, compile_dir],
        ),
        patch.object(
            async_compile_mod, "allocate_compile_dir", return_value=compile_dir
        ),
        patch.object(async_compile_mod, "commit_compile_dir", return_value=compile_dir),
        patch.object(
            async_compile_mod, "generate_bundle", return_value=fake_symbol_kinds
        ) as generate_bundle,
        patch.object(async_compile_mod, "save_symbol_kinds"),
        patch.object(
            async_compile_mod,
            "load_symbol_kinds",
            return_value=fake_symbol_kinds,
        ),
        patch.object(async_compile_mod, "_run_backend_compiler"),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(async_compile_mod, "SpyreSDSCKernelRunner", side_effect=_runner),
    ):
        miss_runner = compiler.sdsc("sdsc_0", [])
        hit_runner = compiler.sdsc("sdsc_0", [])

    generate_bundle.assert_called_once()
    assert miss_runner[3] == fake_symbol_kinds
    assert hit_runner[3] == fake_symbol_kinds


def test_async_compile_failure_moves_cache_entry_at_wait():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"spyre_kernel_cache": True}),
        patch.object(compiler, "wait_pool_ready"),
        patch.object(compiler, "use_process_pool", return_value=True),
        patch.object(compiler, "process_pool", return_value=pool),
        patch.object(async_compile_mod, "compute_specs_hash", return_value="key"),
        patch.object(async_compile_mod, "get_cached_kernel_dir", return_value=None),
        patch.object(
            async_compile_mod, "allocate_compile_dir", return_value="/tmp/key.tmp"
        ),
        patch.object(async_compile_mod, "generate_bundle", return_value=[]),
        patch.object(async_compile_mod, "save_symbol_kinds"),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(async_compile_mod, "_move_to_failed_dir") as move_failed,
    ):
        scope = {"kernel": compiler.sdsc("sdsc_0", [])}
        pool.futures[0].set_exception(RuntimeError("backend compile failed"))

        with pytest.raises(RuntimeError, match="backend compile failed"):
            compiler.wait(scope)

    move_failed.assert_called_once_with("/tmp/key.tmp")


def test_wait_drains_remaining_spyre_futures_after_failure():
    pool = _RecordingPool()
    compiler = async_compile_mod.SpyreAsyncCompile()
    fake_symbol_kinds = [SymbolKind.kernel(0), SymbolKind.kernel(1)]

    with (
        torch._inductor.config.patch({"compile_threads": 2}),
        spyre_config.patch({"spyre_kernel_cache": True}),
        patch.object(compiler, "wait_pool_ready"),
        patch.object(compiler, "use_process_pool", return_value=True),
        patch.object(compiler, "process_pool", return_value=pool),
        patch.object(
            async_compile_mod,
            "compute_specs_hash",
            side_effect=["key0", "key1", "key2"],
        ),
        patch.object(async_compile_mod, "get_cached_kernel_dir", return_value=None),
        patch.object(
            async_compile_mod,
            "allocate_compile_dir",
            side_effect=["/tmp/key0.tmp", "/tmp/key1.tmp", "/tmp/key2.tmp"],
        ),
        patch.object(
            async_compile_mod, "commit_compile_dir", return_value="/cache/key1"
        ) as commit,
        patch.object(
            async_compile_mod, "generate_bundle", return_value=fake_symbol_kinds
        ),
        patch.object(async_compile_mod, "save_symbol_kinds"),
        patch.object(async_compile_mod, "find_unimplemented", return_value=None),
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
        patch.object(async_compile_mod, "SpyreSDSCKernelRunner", side_effect=_runner),
        patch.object(async_compile_mod, "_move_to_failed_dir") as move_failed,
    ):
        scope = {
            f"kernel{index}": compiler.sdsc(f"sdsc_{index}", []) for index in range(3)
        }
        pool.futures[0].set_exception(RuntimeError("first backend failure"))
        pool.futures[1].set_result("compiled")
        pool.futures[2].set_exception(RuntimeError("later backend failure"))

        with pytest.raises(RuntimeError, match="first backend failure"):
            compiler.wait(scope)

        commit.assert_called_once_with("/tmp/key1.tmp", "key1")
        assert scope["kernel1"].result() == (
            "sdsc_1",
            "/cache/key1",
            None,
            fake_symbol_kinds,
        )
        assert [call.args[0] for call in move_failed.call_args_list] == [
            "/tmp/key0.tmp",
            "/tmp/key2.tmp",
        ]


def test_compile_to_dir_rejects_dimension_symbols(tmp_path: Path):
    """_compile_to_dir must raise NotImplementedError when generate_bundle returns
    dimension symbols, before any backend-compiler artifact is produced."""
    fake_symbol_kinds = [SymbolKind.dimension(16, 128, "s0"), SymbolKind.kernel(0)]

    with (
        patch.object(
            async_compile_mod, "generate_bundle", return_value=fake_symbol_kinds
        ),
        pytest.raises(NotImplementedError, match="kDimension"),
    ):
        async_compile_mod._compile_to_dir("test_kernel", str(tmp_path), [], 0)


def test_real_subprocess_pool_runs_backend_jobs_concurrently(tmp_path: Path):
    """Two backend compiles must overlap, not run serially in the parent.

    Each fake compiler touches a marker named for its own compile dir, then
    blocks until it can see two markers.  That is a mutual deadlock unless both
    processes are running at once: a serialized pool leaves the first job waiting
    for a marker the second cannot yet write, and it exits non-zero at the
    deadline.  Both finish only if the pool really ran them concurrently."""
    bin_dir = tmp_path / "bin"
    marker_dir = tmp_path / "markers"
    compile_dirs = [tmp_path / "kernel0", tmp_path / "kernel1"]
    bin_dir.mkdir()
    marker_dir.mkdir()
    for compile_dir in compile_dirs:
        compile_dir.mkdir()

    fake_compiler = bin_dir / "dbo-opt"
    fake_compiler.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "marker_dir = Path(os.environ['FAKE_BACKEND_COMPILER_MARKER_DIR'])\n"
        # Read the compile dir off --export-dir rather than a positional index:
        # the argv layout depends on whether --device is passed.
        "export_dir = next(\n"
        "    a.split('=', 1)[1] for a in sys.argv[1:]\n"
        "    if a.startswith('--export-dir=')\n"
        ")\n"
        "(marker_dir / Path(export_dir).name).write_text('\\n'.join(sys.argv[1:]))\n"
        # The caller treats a missing spyrecode.json as a failure even on exit 0.
        "code_dir = Path(export_dir) / 'spyreCodeDir'\n"
        "code_dir.mkdir(parents=True, exist_ok=True)\n"
        "(code_dir / 'spyrecode.json').write_text('{}')\n"
        "deadline = time.monotonic() + 10\n"
        "while len(list(marker_dir.iterdir())) < 2:\n"
        "    if time.monotonic() >= deadline:\n"
        "        raise SystemExit('backend compile jobs did not overlap')\n"
        "    time.sleep(0.05)\n"
    )
    fake_compiler.chmod(0o755)

    shutdown_compile_workers()
    try:
        with (
            torch._inductor.config.patch(
                {"compile_threads": 2, "worker_start_method": "subprocess"}
            ),
            patch.dict(
                os.environ,
                {
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "FAKE_BACKEND_COMPILER_MARKER_DIR": str(marker_dir),
                },
            ),
        ):
            compiler = async_compile_mod.SpyreAsyncCompile()
            tasks = []
            for index, compile_dir in enumerate(compile_dirs):
                with spyre_config.patch(backend_loop_unroll=bool(index)):  # type: ignore[attr-defined]
                    tasks.append(
                        compiler._submit_backend_compile(
                            f"sdsc_{index}", str(compile_dir)
                        )
                    )
            assert all(task is not None for task in tasks)
            for task in tasks:
                assert task is not None
                task.result(timeout=30)
    finally:
        shutdown_compile_workers()

    assert {path.name for path in marker_dir.iterdir()} == {"kernel0", "kernel1"}
    for index in range(2):
        args = (marker_dir / f"kernel{index}").read_text().splitlines()
        assert f"--enable-loop-unroll={index}" in args
