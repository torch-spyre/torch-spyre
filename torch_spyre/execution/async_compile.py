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
from typing import Any
import os
import subprocess

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import LoopSpec, UnimplementedOp
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.unroll import unroll_loop_specs
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


def _find_unimplemented(specs: list):
    """Return the first UnimplementedOp in specs (recursing into LoopSpec), or None."""
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            return entry
        if isinstance(entry, LoopSpec):
            found = _find_unimplemented(entry.body)
            if found is not None:
                return found
    return None


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, specs: list):
        unimp = _find_unimplemented(specs)
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        # Generate SDSC Bundle from specs (may contain LoopSpec entries).
        output_dir = get_output_dir(kernel_name)
        if not _spyre_config.bundle_hbm_symbols:
            specs = unroll_loop_specs(specs)
        generate_bundle(
            kernel_name,
            output_dir,
            specs,
            use_symbols=_spyre_config.bundle_hbm_symbols,
        )

        # Invoke backend compiler of SDSC Bundle
        subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
