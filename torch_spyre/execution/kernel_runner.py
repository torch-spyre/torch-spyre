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

import threading
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError

import torch
from torch_spyre._C import (
    SymbolicArg,
    SymbolicArgKind,
    launch_jobplan,
    prepare_kernel,
    register_kernel_provenance,
)
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.kernel_provenance import KernelProvenanceDescriptor
from torch_spyre._inductor.profiler_event import (
    format_kernel_provenance_event_name,
)
from torch_spyre.profiler._ffdc import (
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    with_ffdc,
)

logger = get_inductor_logger("kernel_runner")


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    @with_ffdc(CATEGORY_UNIMPLEMENTED, logger, code_dir_attr=None)
    def run(self, *args, **kwargs):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains"
            f" unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    """Kernel runner for a compiled SDSC bundle.

    The jobplan handle is initialised lazily on first call to :meth:`run`.
    This avoids calling ``prepare_kernel`` (which requires a live C++
    RuntimeContext) in the compiling process; the context is only guaranteed
    to be available on the process that actually launches the kernel.

    Everything lazily derived here -- the jobplan, the per-count variant
    runners, and the locks guarding both -- is process-local: a jobplan binds to
    one process's RuntimeContext and a lock cannot cross a fork or a pickle at
    all. Pickling therefore carries only the identity (name, code dir, specs,
    provenance), and the receiving process re-derives the rest on first use.
    """

    def __init__(
        self,
        name: str,
        code_dir: str | None,
        kernel_provenance: KernelProvenanceDescriptor | None = None,
        symbol_kinds: list[SymbolKind] | None = None,
        specs=None,
        pool_size: int = 0,
    ):
        self.kernel_name = name
        self.code_dir = code_dir
        self.kernel_provenance = kernel_provenance
        # Canonical symbol order returned by generate_bundle()
        self.symbol_kinds: list[SymbolKind] = (
            symbol_kinds if symbol_kinds is not None else []
        )
        self.profiler_event_name: str | None
        self._jobplan = None  # initialised lazily, not pickled
        self._jobplan_lock = threading.Lock()
        self._specs = specs
        self._pool_size = pool_size
        self._variant_lock = threading.Lock()
        self._variant_futures: dict[int, Future] = {}

        if kernel_provenance is None:
            self.profiler_event_name = None
        else:
            self.profiler_event_name = format_kernel_provenance_event_name(
                kernel_provenance
            )
            # Rejection is intentionally fail-open: C++ warns and counts
            # conflicts while the key-bearing name remains the compatibility
            # join.
            register_kernel_provenance(
                self.profiler_event_name,
                list(kernel_provenance.debug_handle_ids),
            )

        # Build the SymbolicArg payload from the canonical symbol order
        # that generate_bundle() returned and stored on this runner.
        # symbol_kinds matches the MLIR input_arg slot order: pool first
        # (when frontend_pool_allocation is active), then kernel tensor
        # args in arg_index order. The payload is invariant across launches.
        if self.symbol_kinds:
            if self.symbol_kinds[0].is_pool:
                # call_kernel prepends the pool tensor to args, so it sits at
                # args[0].  Kernel tensor arg_indices are 0-based among kernel
                # tensors only, so add 1 to account for the pool.
                self._symbolic_args: list[SymbolicArg] | None = (
                    [SymbolicArg(kind=SymbolicArgKind.kAddress, tensor_id=0)]
                ) + (
                    [
                        SymbolicArg(
                            kind=SymbolicArgKind.kAddress,
                            tensor_id=sk.arg_index + 1,
                        )
                        for sk in self.symbol_kinds[1:]
                    ]
                )
            else:
                # No pool param — arg_index maps directly to args position.
                self._symbolic_args = [
                    SymbolicArg(kind=SymbolicArgKind.kAddress, tensor_id=sk.arg_index)
                    for sk in self.symbol_kinds
                ]
        else:
            self._symbolic_args = None

    @property
    def jobplan(self):
        if self._specs is not None:
            raise RuntimeError(
                f"{self.kernel_name} has a symbolic loop count and no jobplan of "
                "its own; launch it as run(..., loop_count=n) so the variant for "
                "n is compiled, and take that runner's jobplan"
            )
        if self.code_dir is None:
            raise RuntimeError(f"{self.kernel_name} has no concrete code directory")
        if self._jobplan is not None:
            return self._jobplan
        # Single-flight: prepare_kernel builds device-side state, so two threads
        # first launching the same kernel must not each build their own. The lock
        # is never held together with _variant_lock -- a variant runner is a
        # different object with its own jobplan -- so the two cannot deadlock.
        with self._jobplan_lock:
            if self._jobplan is None:
                logger.debug(
                    "Initialising jobplan for %s from %s",
                    self.kernel_name,
                    self.code_dir,
                )
                # _lazy_init() ensures the C++ RuntimeContext is initialised
                # before prepare_kernel(), which calls into
                # JobPlanBuilder/getDefaultStream().
                torch.spyre._impl._lazy_init()
                spyrecode_dir = self.code_dir + "/spyreCodeDir"
                if self.profiler_event_name is None:
                    self._jobplan = prepare_kernel(spyrecode_dir)
                else:
                    with torch.profiler.record_function(
                        f"prepare_kernel:{self.kernel_name}"
                    ):
                        self._jobplan = prepare_kernel(
                            spyrecode_dir,
                            profiler_name=self.profiler_event_name,
                        )
            return self._jobplan

    def _variant_runner(self, loop_count: int):
        if isinstance(loop_count, bool) or not isinstance(loop_count, int):
            raise TypeError(f"loop_count must be a positive int, got {loop_count!r}")
        if loop_count <= 0:
            raise ValueError(f"loop_count must be positive, got {loop_count}")

        with self._variant_lock:
            future = self._variant_futures.get(loop_count)
            if future is None:
                future = Future()
                self._variant_futures[loop_count] = future
                owner = True
            else:
                owner = False

        from torch_spyre.execution.async_compile import (
            SpyreAsyncCompile,
            specialize_loop_count,
            variant_wait_timeout_s,
        )

        if owner:
            try:
                compiler = SpyreAsyncCompile()
                specs = specialize_loop_count(self._specs, loop_count)
                runner = compiler._sdsc_concrete(
                    self.kernel_name,
                    specs,
                    self._pool_size,
                    self.kernel_provenance,
                )
                if hasattr(runner, "result"):
                    runner = runner.result()
                future.set_result(runner)
            except BaseException as exc:
                future.set_exception(exc)
                with self._variant_lock:
                    if self._variant_futures.get(loop_count) is future:
                        del self._variant_futures[loop_count]
                raise
            # A failed compile is retryable: the future was evicted above, so the
            # next dispatch of this count owns a fresh one.
            return future.result()

        # Only a non-owner waits, and it is bounded: if the owning thread dies
        # without resolving the future, an unbounded wait would hang the engine
        # with nothing to report. The future is deliberately left in place -- the
        # owner may still be making progress, and evicting it here would start a
        # duplicate compile of the same count.
        timeout = variant_wait_timeout_s()
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise RuntimeError(
                f"{self.kernel_name}: waited {timeout}s for another thread to "
                f"compile the loop_count={loop_count} variant; raise or disable "
                "the bound with SPYRE_BACKEND_COMPILE_TIMEOUT_S (0 disables it)"
            ) from exc

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_jobplan"] = None
        state["_jobplan_lock"] = None
        state["_variant_lock"] = None
        state["_variant_futures"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._jobplan = None
        self._jobplan_lock = threading.Lock()
        self._variant_lock = threading.Lock()
        self._variant_futures = {}

    @with_ffdc(CATEGORY_RUNTIME_LAUNCH, logger)
    def run(self, *args, loop_count: int | None = None):
        if self._specs is not None:
            if loop_count is None:
                raise TypeError(f"{self.kernel_name}.run() requires loop_count=")
            return self._variant_runner(loop_count).run(*args)
        if loop_count is not None:
            raise TypeError(f"{self.kernel_name} has no symbolic loop count")
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)
        with torch.profiler.record_function(f"launch_jobplan:{self.kernel_name}"):
            if self._symbolic_args is not None:
                launch_jobplan(self.jobplan, args, self._symbolic_args)
            else:
                launch_jobplan(self.jobplan, args)
