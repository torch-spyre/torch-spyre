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
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Sequence

import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor import timing_recorder
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
from .kernel_runner import (
    SpyreFrontendOnlyRunner,
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)
from .kernel_cache import (
    allocate_compile_dir,
    commit_compile_dir,
    compute_specs_hash,
    get_cached_kernel_dir,
    get_kernel_registry,
    _move_to_failed_dir,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.kernel_provenance import KernelProvenanceDescriptor

logger = get_inductor_logger("sdsc_compile")

# Wall-clock ceiling on ONE backend-compiler invocation, only used for dbo-opt
# on the KTIR path. It bounds a wedged compiler -- which would otherwise block
# torch.compile forever with no diagnostic -- rather than policing slowness:
# both finish in well under a second on a small kernel.
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


# One event name for every backend invocation, whichever emitter selected it, so
# a frontend total stays a single subtraction as emitters come and go. The tool
# is in the event's meta.
_BACKEND_STAGE = "stage:SpyreAsyncCompile:backend_compile"

# The frontend-only warning is per process, not per kernel: a large model emits
# hundreds of kernels and the record already names each skipped one.
_warned_frontend_only = False


def _skip_backend(kernel_name: str, output_dir: str, tool: str):
    """Frontend-only boundary: record the skip and hand back a raising stub.

    Shared by every emitter so the mode means one thing, and so a new backend
    cannot quietly bypass it -- the marker region times nothing, it only fixes
    where the boundary fell in the timeline.
    """
    global _warned_frontend_only
    if not _warned_frontend_only:
        logger.warning(
            "TORCH_SPYRE_FRONTEND_ONLY=1: skipping %s for every kernel; this "
            "process produces no runnable kernels. Skipped kernels are named in "
            "the timing record.",
            tool,
        )
        _warned_frontend_only = True
    with timing_recorder.stage(
        "stage:SpyreAsyncCompile:backend_skipped", kernel=kernel_name, tool=tool
    ):
        pass
    timing_recorder.append_run_meta("backend_skipped_kernels", kernel_name)
    return SpyreFrontendOnlyRunner(kernel_name, output_dir)


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    digest = uuid.uuid4().hex[:8]
    kernel_output_dir = tempfile.mkdtemp(
        dir=spyre_dir, prefix=f"{digest}_{kernel_name}_"
    )
    return kernel_output_dir


def _compile_to_dir(
    kernel_name: str,
    compile_dir: str,
    specs,
    pool_size: int,
) -> bool:
    """Run generate_bundle then dxp_standalone for ``specs`` into ``compile_dir``.

    Shared by the cache-miss path and the no-cache path so that any change to
    the compilation sequence is applied in both places automatically.

    Returns False when frontend-only mode stopped before the backend, leaving
    ``compile_dir`` holding a bundle and none of dxp_standalone's output.
    """
    with timing_recorder.stage(
        "stage:SpyreAsyncCompile:generate_bundle",
        kernel=kernel_name,
        specs=len(specs),
    ):
        generate_bundle(kernel_name, compile_dir, specs, pool_size=pool_size)

    # Backend input is complete. Everything above is frontend work; the
    # subprocess below is the whole of the backend for this kernel, which is
    # what makes this one of the two places a frontend-only compile stops. The
    # boundary sits inside the sequence both callers share, so a later change to
    # that sequence cannot route around it.
    if _spyre_config.frontend_only:
        return False

    with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
        try:
            with timing_recorder.stage(
                _BACKEND_STAGE, kernel=kernel_name, tool="dxp_standalone"
            ):
                subprocess.run(
                    ["dxp_standalone", "-d", compile_dir],
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            try_collect(
                exc,
                logger=logger,
                failure_category=CATEGORY_COMPILE_BACKEND,
                kernel_name=kernel_name,
                code_dir=compile_dir,
            )
            raise
    return True


def _prepare_kernel(
    kernel_name: str,
    output_dir: str,
    kernel_provenance: "KernelProvenanceDescriptor | None",
) -> SpyreSDSCKernelRunner:
    with timing_recorder.stage(
        "stage:SpyreAsyncCompile:prepare_kernel", kernel=kernel_name
    ):
        return SpyreSDSCKernelRunner(
            kernel_name, output_dir, kernel_provenance=kernel_provenance
        )


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
            with timing_recorder.stage(
                "stage:SpyreAsyncCompile:kernel_provenance", kernel=kernel_name
            ):
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
            # A cache hit runs neither generate_bundle nor dxp_standalone, so a
            # frontend-only process would measure no frontend work and still hand
            # back a runnable kernel. Committing a bundle-only dir is worse:
            # commit_compile_dir treats an existing dir as a lost race, so that
            # key would discard every later complete compile.
            and not _spyre_config.frontend_only
        )

        if use_cache:
            # Hash the specs in-memory BEFORE any disk I/O.  On a cache hit
            # neither generate_bundle nor dxp_standalone runs at all.
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
                    return _prepare_kernel(kernel_name, cached_dir, kernel_provenance)

                logger.debug("Cache MISS: Compiling kernel")
                get_kernel_registry().record_miss(cache_key)

                # Allocate a temp dir INSIDE the cache root (same filesystem)
                # so the rename in commit_compile_dir is atomic on POSIX.
                compile_dir: str = allocate_compile_dir(cache_key)
                try:
                    completed = _compile_to_dir(
                        kernel_name, compile_dir, specs, pool_size
                    )
                    # frontend_only is excluded from use_cache above, so the
                    # backend always ran here.
                    assert completed, "frontend-only must not reach the kernel cache"
                    cached_dir = commit_compile_dir(compile_dir, cache_key)
                    logger.debug("Kernel compiled and cached at: %s", cached_dir)
                    return _prepare_kernel(kernel_name, cached_dir, kernel_provenance)
                except Exception:  # subprocess.CalledProcessError:
                    # Move the failed dir to failed/ for manual debugging
                    # rather than leaving .tmp. dirs accumulating in the root.
                    _move_to_failed_dir(compile_dir)
                    raise

        # Caching disabled (SPYRE_KERNEL_CACHE=0, force_disable_caches, or
        # frontend-only). Compile into a throw-away temp dir that lives for this
        # process only.
        output_dir = get_output_dir(kernel_name)
        if not _compile_to_dir(kernel_name, output_dir, specs, pool_size):
            return _skip_backend(kernel_name, output_dir, "dxp_standalone")
        return _prepare_kernel(kernel_name, output_dir, kernel_provenance)

    def ktir(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        """Emit KTDP-dialect MLIR for ``specs`` (OpSpec->KTIR path).

        Mirrors ``sdsc`` but emits KTIR directly instead of an SDSC bundle: the
        emitted KTIR is persisted to disk for inspection and then compiled by
        ``dbo-opt``, which writes a ``spyreCodeDir`` in the same layout
        ``dxp_standalone`` produces, so the result is loaded and launched by the
        same ``SpyreSDSCKernelRunner``.
        """
        # Upfront, before anything is emitted: what device execution needs is a
        # matter of configuration, so there is no reason to emit first. Skipped
        # under frontend_only: these are the backend's prerequisites, and a
        # frontend measurement must not require a toolchain it never invokes.
        if not _spyre_config.frontend_only:
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
        with timing_recorder.stage(
            "stage:SpyreAsyncCompile:generate_ktir",
            kernel=kernel_name,
            specs=len(specs),
        ):
            ktir_text = generate_ktir(
                kernel_name,
                specs,
                bake_addresses=not _spyre_config.bundle_symbolic_args,
            )

        # Persist the emitted KTIR as a text file in the same per-kernel output
        # dir as sdsc's bundle.
        output_dir = get_output_dir(kernel_name)
        ktir_path = os.path.join(output_dir, f"{kernel_name}.ktir")
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
        # dbo-opt is this emitter's backend, so the boundary is here -- ahead of
        # its prerequisites, which only the invocation needs. Both emitters are
        # guarded rather than rejecting the combination at config read, so the
        # mode means the same thing whichever one is selected.
        if _spyre_config.frontend_only:
            return _skip_backend(kernel_name, output_dir, "dbo-opt")

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
                with timing_recorder.stage(
                    _BACKEND_STAGE, kernel=kernel_name, tool="dbo-opt"
                ):
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
        super().wait(scope)
        if self._provenance_failure_count:
            logger.warning(
                "kernel provenance disabled for %d/%d compiled Spyre kernels",
                self._provenance_failure_count,
                self._provenance_attempt_count,
            )
        self._provenance_attempt_count = 0
        self._provenance_failure_count = 0
