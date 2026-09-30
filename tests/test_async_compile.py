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
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest
import sympy
import torch
from torch._inductor.codecache import CodeCacheFuture
from torch._inductor.async_compile import shutdown_compile_workers

from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre.execution import async_compile as async_compile_mod
from torch_spyre._inductor.op_spec import LoopSpec
from torch_spyre.execution import kernel_runner as kernel_runner_mod
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner


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


def test_single_worker_compiles_inline_without_starting_pool():
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        torch._inductor.config.patch({"compile_threads": 1}),
        patch.object(compiler, "wait_pool_ready") as wait_ready,
        patch.object(compiler, "process_pool") as pool,
        patch.object(async_compile_mod, "_run_backend_compiler") as compile_backend,
    ):
        task = compiler._submit_backend_compile("sdsc_0", "/tmp/kernel")

    assert task is None
    wait_ready.assert_not_called()
    pool.assert_not_called()
    compile_backend.assert_called_once_with("sdsc_0", "/tmp/kernel", dict(os.environ))


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
        patch.object(
            async_compile_mod, "commit_compile_dir", return_value="/cache/key"
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
    ):
        scope = {"kernel": compiler.sdsc("sdsc_0", [])}
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
        "(marker_dir / Path(export_dir).name).touch()\n"
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
            tasks = [
                compiler._submit_backend_compile(f"sdsc_{index}", str(compile_dir))
                for index, compile_dir in enumerate(compile_dirs)
            ]
            assert all(task is not None for task in tasks)
            for task in tasks:
                assert task is not None
                task.result(timeout=30)
    finally:
        shutdown_compile_workers()

    assert {path.name for path in marker_dir.iterdir()} == {"kernel0", "kernel1"}


def test_symbolic_loop_returns_lazy_runner_without_generating_bundle():
    compiler = async_compile_mod.SpyreAsyncCompile()
    count = sympy.Symbol("s0", integer=True, positive=True)
    specs = [LoopSpec(count=count, body=[], max_count=8)]

    with (
        patch.object(async_compile_mod, "generate_bundle") as generate,
        patch.object(
            async_compile_mod, "build_kernel_provenance_descriptor", return_value=None
        ),
    ):
        runner = compiler.sdsc("dynamic", specs)

    assert isinstance(runner, SpyreSDSCKernelRunner)
    assert runner.code_dir is None
    generate.assert_not_called()


def test_specialize_loop_count_replaces_only_count_and_checks_bound():
    count = sympy.Symbol("s0", integer=True, positive=True)
    body = object()
    specs = [LoopSpec(count=count, body=[body], max_count=8)]

    result = async_compile_mod.specialize_loop_count(specs, 3)

    assert result[0].count == 3
    assert result[0].max_count == 8
    assert result[0].body[0] is body
    assert specs[0].count == count
    with pytest.raises(ValueError, match="exceeds traced maximum"):
        async_compile_mod.specialize_loop_count(specs, 9)


def test_variant_runner_caches_one_concrete_runner_per_count():
    count = sympy.Symbol("s0", integer=True, positive=True)
    runner = SpyreSDSCKernelRunner(
        "dynamic",
        None,
        specs=[LoopSpec(count=count, body=[], max_count=8)],
    )
    concrete = object()
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        patch.object(
            async_compile_mod,
            "SpyreAsyncCompile",
            return_value=compiler,
        ),
        patch.object(compiler, "_sdsc_concrete", return_value=concrete) as compile_one,
    ):
        assert runner._variant_runner(3) is concrete
        assert runner._variant_runner(3) is concrete

    compile_one.assert_called_once()


def test_variant_runner_single_flight_for_concurrent_same_count():
    count = sympy.Symbol("s0", integer=True, positive=True)
    runner = SpyreSDSCKernelRunner(
        "dynamic",
        None,
        specs=[LoopSpec(count=count, body=[], max_count=8)],
    )
    concrete = object()
    compiler = async_compile_mod.SpyreAsyncCompile()
    entered = threading.Event()
    release = threading.Event()

    def compile_one(*_args):
        entered.set()
        assert release.wait(timeout=10)
        return concrete

    results = []
    errors = []

    def request():
        try:
            results.append(runner._variant_runner(4))
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    with (
        patch.object(
            async_compile_mod,
            "SpyreAsyncCompile",
            return_value=compiler,
        ),
        patch.object(
            compiler, "_sdsc_concrete", side_effect=compile_one
        ) as compile_call,
    ):
        first = threading.Thread(target=request)
        second = threading.Thread(target=request)
        first.start()
        assert entered.wait(timeout=10)
        second.start()
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)

    assert not errors
    assert results == [concrete, concrete]
    compile_call.assert_called_once()


def test_symbolic_runner_has_no_jobplan_of_its_own():
    """A symbolic runner owns no compiled code, only the specs to specialize.

    Reading .jobplan on one is a caller that skipped run(loop_count=...); say so,
    rather than reporting a missing code directory as if compilation had failed.
    """
    count = sympy.Symbol("s0", integer=True, positive=True)
    runner = SpyreSDSCKernelRunner(
        "dynamic",
        None,
        specs=[LoopSpec(count=count, body=[], max_count=8)],
    )

    with pytest.raises(RuntimeError, match="symbolic loop count"):
        runner.jobplan


def test_jobplan_is_prepared_once_for_concurrent_first_launches():
    """prepare_kernel builds device-side state, so two threads must not both run it.

    The first launch of a kernel is exactly when two engine threads can arrive
    together, and a second JobPlan for the same code dir is a leak at best.
    """
    runner = SpyreSDSCKernelRunner("concrete", "/tmp/nonexistent-code-dir")
    plan = object()
    entered = threading.Event()
    release = threading.Event()

    def prepare_kernel(_spyrecode_dir):
        entered.set()
        assert release.wait(timeout=10)
        return plan

    results = []
    errors = []

    def request():
        try:
            results.append(runner.jobplan)
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    with (
        patch.object(torch.spyre._impl, "_lazy_init"),
        patch.object(
            kernel_runner_mod, "prepare_kernel", side_effect=prepare_kernel
        ) as prepare,
    ):
        first = threading.Thread(target=request)
        second = threading.Thread(target=request)
        first.start()
        assert entered.wait(timeout=10)
        second.start()
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)

    assert not errors
    assert results == [plan, plan]
    prepare.assert_called_once()


def test_ktir_rejects_a_symbolic_loop_count_before_emitting():
    """The KTIR emitter has no per-count variant path, so say so up front.

    Failing here, rather than inside codegen, names both the emitter and the flag
    that selected it, and leaves no half-written artifact behind.
    """
    compiler = async_compile_mod.SpyreAsyncCompile()
    count = sympy.Symbol("s0", integer=True, positive=True)
    specs = [LoopSpec(count=count, body=[], max_count=8)]

    with (
        patch.object(async_compile_mod, "_check_ktir_device_prerequisites") as check,
        patch.object(async_compile_mod, "get_output_dir") as output_dir,
    ):
        with pytest.raises(NotImplementedError, match="TORCH_SPYRE_KTIR=0"):
            compiler.ktir("dynamic", specs)

    check.assert_not_called()
    output_dir.assert_not_called()


def test_compile_timeout_env_overrides_the_stage_default():
    with patch.dict(os.environ):
        os.environ.pop(async_compile_mod._TIMEOUT_ENV, None)
        assert async_compile_mod.compile_timeout_s(30.0) == 30.0

    with patch.dict(os.environ, {async_compile_mod._TIMEOUT_ENV: "5"}):
        assert async_compile_mod.compile_timeout_s(30.0) == 5.0

    with patch.dict(os.environ, {async_compile_mod._TIMEOUT_ENV: "0"}):
        assert async_compile_mod.compile_timeout_s(30.0) is None
        assert async_compile_mod.variant_wait_timeout_s() is None

    with patch.dict(os.environ, {async_compile_mod._TIMEOUT_ENV: "soon"}):
        with pytest.raises(ValueError, match="is not a number"):
            async_compile_mod.compile_timeout_s(30.0)


def test_variant_wait_outlasts_the_compile_it_waits_for():
    """A waiter that gives up first would report a hang the compiler is not in."""
    with patch.dict(os.environ, {async_compile_mod._TIMEOUT_ENV: "5"}):
        wait = async_compile_mod.variant_wait_timeout_s()
        compile_bound = async_compile_mod.compile_timeout_s(
            async_compile_mod._COMPILE_TIMEOUT_S
        )

    assert wait is not None and compile_bound is not None
    assert wait > compile_bound


def test_run_reaped_kills_grandchildren_on_timeout(tmp_path):
    """Killing only the child leaves the compiler's own children holding the card."""
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time;"
        " time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        async_compile_mod._run_reaped([sys.executable, "-c", script], timeout=2.0)

    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} survived the timeout")


def test_pickled_symbolic_runner_recompiles_its_variants():
    """Compiled variants, jobplans and locks are process-local by construction.

    A runner crosses a process boundary as identity only; the receiver rebuilds
    what binds to its own RuntimeContext.
    """
    count = sympy.Symbol("s0", integer=True, positive=True)
    runner = SpyreSDSCKernelRunner(
        "dynamic",
        None,
        specs=[LoopSpec(count=count, body=[], max_count=8)],
    )
    concrete = object()
    compiler = async_compile_mod.SpyreAsyncCompile()

    with (
        patch.object(async_compile_mod, "SpyreAsyncCompile", return_value=compiler),
        patch.object(compiler, "_sdsc_concrete", return_value=concrete) as compile_one,
    ):
        assert runner._variant_runner(3) is concrete
        copy = pickle.loads(pickle.dumps(runner))
        assert copy._variant_futures == {}
        assert copy._jobplan is None
        assert copy._jobplan_lock is not None
        assert copy._variant_runner(3) is concrete

    assert compile_one.call_count == 2


def test_run_rejects_an_unexpected_keyword_argument():
    """Launch args are positional and index-bound to the op specs.

    A keyword arriving here is a generated-wrapper/runner mismatch; absorbing it
    would drop a real argument silently.
    """
    runner = SpyreSDSCKernelRunner("concrete", None)

    with pytest.raises(TypeError, match="unexpected keyword argument 'stream'"):
        runner.run(object(), stream=0)
