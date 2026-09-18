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
from concurrent.futures import Future

import torch
from torch_spyre._C import (
    SymbolicArg,
    launch_jobplan,
    prepare_kernel,
    register_kernel_provenance,
)
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
    def run(self, *args, **kw_args):
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
    """

    def __init__(
        self,
        name: str,
        code_dir: str | None,
        kernel_provenance: KernelProvenanceDescriptor | None = None,
        specs=None,
        pool_size: int = 0,
    ):
        self.kernel_name = name
        self.code_dir = code_dir
        self.kernel_provenance = kernel_provenance
        self.profiler_event_name: str | None
        self._jobplan = None  # initialised lazily, not pickled
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

    @property
    def jobplan(self):
        if self.code_dir is None:
            raise RuntimeError(f"{self.kernel_name} has no concrete code directory")
        if self._jobplan is None:
            logger.debug(
                "Initialising jobplan for %s from %s", self.kernel_name, self.code_dir
            )
            # _lazy_init() ensures the C++ RuntimeContext is initialised before
            # prepare_kernel(), which calls into JobPlanBuilder/getDefaultStream().
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

        if owner:
            try:
                from torch_spyre.execution.async_compile import (
                    SpyreAsyncCompile,
                    specialize_loop_count,
                )

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
        return future.result()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_jobplan"] = None
        state["_variant_lock"] = None
        state["_variant_futures"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._jobplan = None
        self._variant_lock = threading.Lock()
        self._variant_futures = {}

    @with_ffdc(CATEGORY_RUNTIME_LAUNCH, logger)
    def run(
        self,
        *args,
        symbolic_args: list[SymbolicArg] | None = None,
        loop_count: int | None = None,
        **kw_args,
    ):
        if self._specs is not None:
            if loop_count is None:
                raise TypeError(f"{self.kernel_name}.run() requires loop_count=")
            return self._variant_runner(loop_count).run(
                *args, symbolic_args=symbolic_args
            )
        if loop_count is not None:
            raise TypeError(f"{self.kernel_name} has no symbolic loop count")
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)
        with torch.profiler.record_function(f"launch_jobplan:{self.kernel_name}"):
            if symbolic_args:
                launch_jobplan(self.jobplan, args, symbolic_args)
            else:
                launch_jobplan(self.jobplan, args)
