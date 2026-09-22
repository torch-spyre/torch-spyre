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

import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from typing import Any, cast

import torch
from torch._inductor.async_compile import AsyncCompile, get_compile_threads
from torch._inductor.codecache import CodeCacheFuture
from torch._inductor.compile_worker.subproc_pool import SubprocException
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.kernel_provenance import (
    build_kernel_provenance_descriptor,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre.profiler._ffdc import CATEGORY_COMPILE_BACKEND, try_collect
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner
from .kernel_cache import (
    allocate_compile_dir,
    commit_compile_dir,
    compute_specs_hash,
    get_cached_kernel_dir,
    get_kernel_registry,
    load_symbol_kinds,
    save_symbol_kinds,
    _move_to_failed_dir,
)

logger = get_inductor_logger("sdsc_compile")

# Wall-clock ceiling on ONE backend-compiler invocation, applied to dbo-opt on
# both the bundle and the KTIR path. It bounds a wedged compiler -- which would
# otherwise block torch.compile forever with no diagnostic -- rather than
# policing slowness: both finish in well under a second on a small kernel.
# Raise it if a large bundle legitimately needs longer.
_COMPILE_TIMEOUT_S = 60.0


def _check_ktir_device_prerequisites() -> None:
    """Raise unless the environment can compile emitted KTIR for the device.

    Names everything missing at once, so a first run does not turn one
    misconfiguration into a sequence of unrelated-looking failures.
    """
    missing = []

    if not _spyre_config.ktir_device_mlir:
        missing.append("set KTIR_DEVICE_MLIR to a .mlir declaring the target device")

    if shutil.which("dbo-opt") is None:
        missing.append("put dbo-opt on PATH")

    if missing:
        raise RuntimeError(
            "OpSpec->KTIR: cannot compile for the device:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


# Linux caps a single path component at NAME_MAX bytes (255 on ext4/xfs/tmpfs).
# Deeply fused kernels (e.g. Gemma-4 MoE decode) produce kernel names >250
# chars; left whole they overflow the per-kernel dir/file name and mkdtemp
# raises [Errno 36] ENAMETOOLONG. Truncate to a readable head: the uuid digest
# below and mkdtemp's random suffix already guarantee uniqueness, so the tail is
# only a human-readable aid.
_NAME_MAX = 255
# Budget leaves room for the longest wrapper around the name:
#   dir:  f"{digest(8)}_{name}_" + mkdtemp's 8 random chars  => name + 18
#   file: f"{name}.ktir"                                     => name + 5
# 18 is the larger fixed overhead; keep a safety margin.
_KERNEL_NAME_BUDGET = _NAME_MAX - 24


def _safe_kernel_name(kernel_name: str) -> str:
    """Truncate a kernel name so it fits a single NAME_MAX path component."""
    if len(kernel_name) <= _KERNEL_NAME_BUDGET:
        return kernel_name
    return kernel_name[:_KERNEL_NAME_BUDGET]


def _check_backend_compiler_on_path() -> None:
    """Raise unless ``dbo-opt`` can be found, before a bundle is emitted.

    Deliberately narrower than ``_check_ktir_device_prerequisites``: the bundle
    path treats ``KTIR_DEVICE_MLIR`` as optional (dbo-opt falls back to the
    spyre_dd2_basic under DEEPTOOLS_PATH), so requiring it here would reject a
    working setup.

    ``dxp_standalone`` was invoked with no such check, so a missing binary
    surfaced as a bare FileNotFoundError from subprocess.run. Now that dbo-opt
    is on the path of every compile, say what to do about it instead.
    """
    if shutil.which("dbo-opt") is None:
        raise RuntimeError(
            "cannot compile the bundle: dbo-opt not found.\n  - put dbo-opt on PATH"
        )


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    digest = uuid.uuid4().hex[:8]
    safe_name = _safe_kernel_name(kernel_name)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{digest}_{safe_name}_")
    return kernel_output_dir


def _compile_to_dir(
    kernel_name: str,
    compile_dir: str,
    specs,
    pool_size: int,
):
    """Run generate_bundle for ``specs`` into ``compile_dir``.

    Shared by the cache-miss path and the no-cache path so that any change to
    the compilation sequence is applied in both places automatically.

    Returns:
        The list of ``SymbolKind`` values produced by ``generate_bundle``,
        describing the kind (address symbol vs. dimension argument) of each
        symbol in the compiled bundle.

    Raises:
        NotImplementedError: if any dimension symbol is present, because the
            runtime kDimension payload is not yet implemented and submitting
            such a bundle to the backend compiler would produce a mismatched
            inputSym_ slot count.
    """
    symbol_kinds = generate_bundle(kernel_name, compile_dir, specs, pool_size=pool_size)
    if any(sk.is_dimension for sk in symbol_kinds):
        raise NotImplementedError(
            "SDSC bundle dimension symbols require runtime kDimension support"
        )
    return symbol_kinds


def _run_backend_compiler(
    kernel_name: str, compile_dir: str, env: dict[str, str]
) -> str:
    """Compile one materialized bundle with dbo-opt and return its directory.

    This function is module-level and its arguments are intentionally simple so
    Inductor's subprocess compile pool can pickle and execute it.  Bundle
    generation remains in the parent process; workers only invoke the backend
    compiler and never construct a runner or touch the device runtime.

    ``env`` is the parent's os.environ snapshot, so PATH reaches the worker and
    the dbo-opt lookup below resolves the same binary the parent would.
    """
    _check_backend_compiler_on_path()

    cmd = ["dbo-opt"]
    # Only when configured: with no --device the dataflow scheduler falls back
    # to the spyre_dd2_basic under DEEPTOOLS_PATH, so passing an empty value
    # would be worse than omitting the flag.  KTIR_DEVICE_MLIR is reused rather
    # than given a bundle-path spelling of its own.
    if _spyre_config.ktir_device_mlir:
        cmd.append(f"--device={_spyre_config.ktir_device_mlir}")
    cmd += [
        f"--export-dir={compile_dir}",
        "-kEmitSpyreCode",
        os.path.join(compile_dir, "bundle.mlir"),
    ]

    with torch.profiler.record_function(f"dbo-opt:{kernel_name}"):
        try:
            # capture_output: dbo-opt is an MLIR *-opt tool, so it prints the
            # transformed module to stdout as a matter of course.
            # ``dxp_standalone -d`` did not, which is why nothing captured it
            # before -- left uncaptured, every kernel dumps its whole bundle
            # module into the user's terminal.  Capturing also makes the
            # compiler's own diagnostics available to the error paths below.
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                timeout=_COMPILE_TIMEOUT_S,
            )
            # The KTIR path (#3651) reports that dbo-opt can exit 0 having
            # written nothing, so treat the artifact -- not the return code --
            # as the success condition. Not independently confirmed for the
            # bundle frontend, but the check is cheap either way.
            spyrecode = os.path.join(compile_dir, "spyreCodeDir", "spyrecode.json")
            if not os.path.exists(spyrecode):
                raise RuntimeError(
                    f"dbo-opt exited 0 but wrote no {spyrecode}.\n"
                    f"command: {' '.join(cmd)}\n"
                    f"stderr:\n{proc.stderr}"
                )
        except subprocess.TimeoutExpired as exc:
            # Would otherwise land in the broad handler below, which collects
            # correctly but re-raises a TimeoutExpired whose message says
            # nothing about which knob relaxes it.
            try_collect(
                exc,
                logger=logger,
                failure_category=CATEGORY_COMPILE_BACKEND,
                kernel_name=kernel_name,
                code_dir=compile_dir,
            )
            raise RuntimeError(
                f"dbo-opt timed out after {_COMPILE_TIMEOUT_S}s "
                f"(_COMPILE_TIMEOUT_S).\ncommand: {' '.join(cmd)}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            try_collect(
                exc,
                logger=logger,
                failure_category=CATEGORY_COMPILE_BACKEND,
                kernel_name=kernel_name,
                code_dir=compile_dir,
            )
            # Re-raised as RuntimeError rather than bare: with capture_output
            # the compiler's diagnostics no longer reach the terminal on their
            # own, so a bare CalledProcessError would report only an exit code.
            raise RuntimeError(
                f"dbo-opt failed with exit code {exc.returncode}.\n"
                f"command: {' '.join(cmd)}\nstderr:\n{exc.stderr}"
            ) from exc
        except Exception as exc:
            try_collect(
                exc,
                logger=logger,
                failure_category=CATEGORY_COMPILE_BACKEND,
                kernel_name=kernel_name,
                code_dir=compile_dir,
            )
            raise
    return compile_dir


class _SpyreCompileFuture(CodeCacheFuture):
    """Resolve one backend compile task and build its runner in the parent."""

    def __init__(
        self,
        task: Future[str],
        kernel_name: str,
        compile_dir: str,
        kernel_provenance,
        symbol_kinds,
        cache_key: str | None = None,
    ) -> None:
        self._task = task
        self._kernel_name = kernel_name
        self._compile_dir = compile_dir
        self._kernel_provenance = kernel_provenance
        self._symbol_kinds = symbol_kinds
        self._cache_key = cache_key
        self._runner: SpyreSDSCKernelRunner | None = None
        self._failure_dir_moved = False

    def result(self, timeout: float | None = None):
        if self._runner is not None:
            return self._runner

        try:
            self._task.result(timeout=timeout)
        except FuturesTimeoutError:
            # The worker is still active; do not move its directory out from
            # underneath it.  AsyncCompile.wait() adds the timeout diagnostic.
            raise
        except SubprocException as exc:
            self._move_failed_cache_entry()
            raise exc.with_name(self._kernel_name) from exc
        except Exception:
            self._move_failed_cache_entry()
            raise

        code_dir = self._compile_dir
        if self._cache_key is not None:
            code_dir = commit_compile_dir(self._compile_dir, self._cache_key)
        self._runner = SpyreSDSCKernelRunner(
            self._kernel_name,
            code_dir,
            kernel_provenance=self._kernel_provenance,
            symbol_kinds=self._symbol_kinds,
        )
        return self._runner

    def _move_failed_cache_entry(self) -> None:
        if self._cache_key is not None and not self._failure_dir_moved:
            _move_to_failed_dir(self._compile_dir)
            self._failure_dir_moved = True


class SpyreAsyncCompile(AsyncCompile):
    """Spyre kernel compilation (`sdsc`), plus the upstream AsyncCompile.

    A graph mixing Spyre and CPU work emits `async_compile.cpp_pybinding(...)`
    against this same object, so we inherit AsyncCompile for `cpp_pybinding`/
    `wait` rather than stubbing them -- a no-op `wait()` alone can't compile a
    CPU kernel it was never given.

    """

    def __init__(self):
        super().__init__()
        self._provenance_attempt_count = 0
        self._provenance_failure_count = 0
        self._pending_spyre_futures: list[_SpyreCompileFuture] = []

    def triton(self, *args, **kwargs):
        raise NotImplementedError(
            "SpyreAsyncCompile does not support Triton kernels; only "
            "cpp_pybinding (CPU) and sdsc (Spyre) are validated."
        )

    def cpp(self, *args, **kwargs):
        raise NotImplementedError(
            "SpyreAsyncCompile does not support the cpp() path; CPU kernels "
            "go through cpp_pybinding (cpu_backend='cpp')."
        )

    def _submit_backend_compile(
        self, kernel_name: str, compile_dir: str
    ) -> Future[str] | None:
        """Submit the backend compile to Inductor's pool, or compile inline."""
        if _spyre_config.async_backend_compile and get_compile_threads() > 1:
            # The first use creates the pool and submits its readiness probe.
            # Waiting for that short probe guarantees the first Spyre kernel is
            # parallel too, rather than accidentally compiling it inline.
            self.wait_pool_ready()
            if self.use_process_pool():
                return self.process_pool().submit(
                    _run_backend_compiler,
                    kernel_name,
                    compile_dir,
                    dict(os.environ),
                )

        _run_backend_compiler(kernel_name, compile_dir, dict(os.environ))
        return None

    def _compile_future(
        self,
        task: Future[str],
        kernel_name: str,
        compile_dir: str,
        kernel_provenance,
        symbol_kinds,
        cache_key: str | None = None,
    ) -> _SpyreCompileFuture:
        future = _SpyreCompileFuture(
            task,
            kernel_name,
            compile_dir,
            kernel_provenance,
            symbol_kinds,
            cache_key=cache_key,
        )
        self._pending_spyre_futures.append(future)
        return future

    def sdsc(
        self,
        kernel_name: str,
        specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
        pool_size: int = 0,
    ):
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                "WARNING: Compiling unimplemented %s to runtime exception", unimp.op
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        self._provenance_attempt_count += 1
        try:
            # This is the common fresh-compile/cache-reload boundary: generated
            # wrappers have reconstructed the finalized OpSpecs before calling
            # sdsc(). Derive the transport-neutral identity here without changing
            # the generated wrapper call ABI.
            finalized_specs = cast(Sequence[OpSpec | LoopSpec], specs)
            kernel_provenance = build_kernel_provenance_descriptor(finalized_specs)
        except Exception:  # noqa: BLE001 - provenance must never fail the build
            # Keep canonicalization strict rather than issuing an ambiguous
            # fallback key. Log the first traceback, then report the complete
            # failure count at the generated wrapper's wait() boundary.
            self._provenance_failure_count += 1
            if self._provenance_failure_count == 1:
                logger.warning(
                    "kernel provenance descriptor construction failed for kernel "
                    "%s; continuing without kernel provenance; additional "
                    "failures in this compilation will be summarized",
                    kernel_name,
                    exc_info=True,
                )
            kernel_provenance = None

        use_cache = (
            _spyre_config.spyre_kernel_cache
            and not torch._inductor.config.force_disable_caches
        )

        if use_cache:
            # Hash the specs in-memory BEFORE any disk I/O.  On a cache hit
            # neither generate_bundle nor the backend compiler runs at all.
            try:
                cache_key = compute_specs_hash(
                    specs, kernel_name=kernel_name, pool_size=pool_size
                )
            except RuntimeError as e:
                logger.warning(
                    "Kernel cache disabled for %s: could not compute cache key: %s. "
                    "Set SPYRE_KERNEL_CACHE=0 to suppress this warning.",
                    kernel_name,
                    e,
                )
            else:
                logger.debug("Bundle cache key: %s", cache_key)

                cached_dir = get_cached_kernel_dir(cache_key)
                if cached_dir is not None:
                    logger.debug("Cache HIT: Using cached kernel from: %s", cached_dir)
                    get_kernel_registry().record_hit(cache_key)
                    return SpyreSDSCKernelRunner(
                        kernel_name,
                        cached_dir,
                        kernel_provenance=kernel_provenance,
                        symbol_kinds=load_symbol_kinds(cached_dir),
                    )

                logger.debug("Cache MISS: Compiling kernel")
                get_kernel_registry().record_miss(cache_key)

                # Allocate a temp dir INSIDE the cache root (same filesystem)
                # so the rename in commit_compile_dir is atomic on POSIX.
                compile_dir: str = allocate_compile_dir(cache_key)
                try:
                    symbol_kinds = _compile_to_dir(
                        kernel_name, compile_dir, specs, pool_size
                    )
                    save_symbol_kinds(compile_dir, symbol_kinds)
                    task = self._submit_backend_compile(kernel_name, compile_dir)
                    if task is not None:
                        return self._compile_future(
                            task,
                            kernel_name,
                            compile_dir,
                            kernel_provenance,
                            symbol_kinds,
                            cache_key=cache_key,
                        )
                    cached_dir = commit_compile_dir(compile_dir, cache_key)
                    logger.debug("Kernel compiled and cached at: %s", cached_dir)
                    return SpyreSDSCKernelRunner(
                        kernel_name,
                        cached_dir,
                        kernel_provenance=kernel_provenance,
                        symbol_kinds=symbol_kinds,
                    )
                except Exception:  # subprocess.CalledProcessError:
                    # Move the failed dir to failed/ for manual debugging
                    # rather than leaving .tmp. dirs accumulating in the root.
                    _move_to_failed_dir(compile_dir)
                    raise

        # Caching disabled (SPYRE_KERNEL_CACHE=0 or force_disable_caches).
        # Compile into a throw-away temp dir that lives for this process only.
        output_dir = get_output_dir(kernel_name)
        symbol_kinds = _compile_to_dir(kernel_name, output_dir, specs, pool_size)
        task = self._submit_backend_compile(kernel_name, output_dir)
        if task is not None:
            return self._compile_future(
                task,
                kernel_name,
                output_dir,
                kernel_provenance,
                symbol_kinds,
            )
        return SpyreSDSCKernelRunner(
            kernel_name,
            output_dir,
            kernel_provenance=kernel_provenance,
            symbol_kinds=symbol_kinds,
        )

    def ktir(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        """Emit KTDP-dialect MLIR for ``specs`` (OpSpec->KTIR path).

        Mirrors ``sdsc`` but emits KTIR directly instead of an SDSC bundle: the
        emitted KTIR is persisted to disk for inspection and then compiled by
        ``dbo-opt``, which writes a ``spyreCodeDir`` in the same layout the
        bundle path produces, so the result is loaded and launched by the same
        ``SpyreSDSCKernelRunner``.
        """
        # Upfront, before anything is emitted: what device execution needs is a
        # matter of configuration, so there is no reason to emit first.
        _check_ktir_device_prerequisites()

        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                "WARNING: Compiling unimplemented %s to runtime exception", unimp.op
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        from torch_spyre._inductor.codegen.ktir import generate_ktir

        # Emit before opening the file: if generate_ktir raises we must not
        # leave a truncated/empty .ktir behind.
        #
        # Canonical KTIR spells base addresses as func arguments.  dbo-opt needs
        # them baked into constants (dataflow-scheduler#65), so this path -- the
        # one that runs dbo-opt -- asks for that form; the emitter itself has no
        # opinion about the backend.  Drop the argument when #65 is fixed.
        #
        # The same predicate ``call_kernel`` passes a pool tensor by, so the
        # signature opens with a matching slot.  Read from config here: the
        # emitter reads no config.
        ktir_text = generate_ktir(
            kernel_name,
            specs,
            bake_addresses=not _spyre_config.bundle_symbolic_args,
            frontend_pool_allocation=_spyre_config.pool_allocated_by_frontend(),
        )

        # Persist the emitted KTIR as a text file in the same per-kernel output
        # dir as sdsc's bundle.
        output_dir = get_output_dir(kernel_name)
        ktir_path = os.path.join(output_dir, f"{_safe_kernel_name(kernel_name)}.ktir")
        with open(ktir_path, "w") as fh:
            fh.write(ktir_text)
        logger.debug("OpSpec->KTIR: wrote %s", ktir_path)

        return self._compile_ktir_with_dbo(kernel_name, ktir_path, output_dir)

    def _compile_ktir_with_dbo(self, kernel_name: str, ktir_path: str, output_dir: str):
        """Compile ``ktir_path`` with ``dbo-opt`` and return a runner for it.

        ``--export-dir`` receives the per-kernel output dir, under which dbo-opt
        writes ``spyreCodeDir/{spyrecode.json, init_binary.bin}`` -- exactly the
        layout ``prepare_kernel`` loads, so no new runner is needed.
        """
        # Re-checked here, not only in ``ktir``: this is also reached directly
        # (tests, callers compiling a .ktir off disk), and the check is a cheap
        # idempotent read of config plus one PATH lookup.
        _check_ktir_device_prerequisites()

        cmd = [
            "dbo-opt",
            "--from-ktir",
            f"--device={_spyre_config.ktir_device_mlir}",
            f"--export-dir={output_dir}",
            "--kEmitSpyreCode",
            ktir_path,
        ]

        # No environment override: dbo-opt inherits ours, so whatever library
        # search path was exported for this process is what it resolves against.
        # A build that cannot find its own libraries that way is a deployment
        # problem to fix in the shell, not something to paper over per-child --
        # and a child-only path stopped being separable once a process commits
        # to one backend for its lifetime via ``ktir_emitter``.
        with torch.profiler.record_function(f"dbo-opt:{kernel_name}"):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=_COMPILE_TIMEOUT_S,
                )
                # dbo-opt can exit 0 having written nothing, so the artifact
                # itself -- not the return code -- is the success condition.
                spyrecode = os.path.join(output_dir, "spyreCodeDir", "spyrecode.json")
                if not os.path.exists(spyrecode):
                    raise RuntimeError(
                        "OpSpec->KTIR: dbo-opt exited 0 but wrote no "
                        f"{spyrecode}.\ncommand: {' '.join(cmd)}\n"
                        f"stderr:\n{proc.stderr}"
                    )
            except subprocess.TimeoutExpired as exc:
                # Would otherwise land in the broad handler below, which collects
                # correctly but re-raises a TimeoutExpired whose message says
                # nothing about which knob relaxes it.
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise RuntimeError(
                    f"OpSpec->KTIR: dbo-opt timed out after "
                    f"{_COMPILE_TIMEOUT_S}s (_COMPILE_TIMEOUT_S).\n"
                    f"command: {' '.join(cmd)}"
                ) from exc
            except subprocess.CalledProcessError as exc:
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise RuntimeError(
                    f"OpSpec->KTIR: dbo-opt failed with exit code "
                    f"{exc.returncode}.\ncommand: {' '.join(cmd)}\n"
                    f"stderr:\n{exc.stderr}"
                ) from exc
            except Exception as exc:
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        try:
            super().wait(scope)
        except Exception:
            # Upstream stops at the first failed future. Resolve every submitted
            # Spyre future so successful cache entries are committed and every
            # other failed .tmp entry is moved aside before preserving the first
            # exception for the caller. Track the underlying futures directly:
            # debugging helpers may wrap them in another CodeCacheFuture.
            wait_timeout = torch._inductor.config.compile_worker_wait_timeout or None
            for future in self._pending_spyre_futures:
                try:
                    future.result(timeout=wait_timeout)
                except Exception:
                    pass
            raise
        finally:
            self._pending_spyre_futures.clear()
            if self._provenance_failure_count:
                logger.warning(
                    "kernel provenance disabled for %d/%d compiled Spyre kernels",
                    self._provenance_failure_count,
                    self._provenance_attempt_count,
                )
            self._provenance_attempt_count = 0
            self._provenance_failure_count = 0
