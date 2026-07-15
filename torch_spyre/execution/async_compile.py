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

import tempfile
from collections.abc import Sequence
import os
import subprocess
import torch

from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile(AsyncCompile):
    """Spyre kernel compilation (`sdsc`), plus the upstream AsyncCompile.

    A graph mixing Spyre and CPU work emits `async_compile.cpp_pybinding(...)`
    against this same object, so we inherit AsyncCompile for `cpp_pybinding`/
    `wait` rather than stubbing them -- a no-op `wait()` alone can't compile a
    CPU kernel it was never given.

    """

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
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        # Generate SDSC Bundle from OpSpecs
        output_dir = get_output_dir(kernel_name)
        generate_bundle(kernel_name, output_dir, specs)

        # Invoke backend compiler of SDSC Bundle
        with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
            subprocess.run(["dxp_standalone", "-d", output_dir], check=True)

        return SpyreSDSCKernelRunner(kernel_name, output_dir)
