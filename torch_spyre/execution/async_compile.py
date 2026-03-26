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

import json
import tempfile
from typing import Any, Union
import os
import subprocess

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._C import convert_artifacts
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.logging_utils import get_inductor_logger, _get_env_bool
from torch_spyre._inductor.op_spec import OpSpec, UnimplementedOp
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")

_SDSC_BUNDLE = _get_env_bool("SPYRE_SUPERDSC_BUNDLE", True)


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, specs: list[Union[OpSpec | UnimplementedOp]]):
        # 1. Generate SDSC.json for each OpSpec
        sdscs = []
        arg_mappings = []
        for ks in specs:
            if isinstance(ks, UnimplementedOp):
                print(f"WARNING: Compiling unimplemented {ks.op} to runtime exception")
                return SpyreUnimplementedRunner(kernel_name, ks.op)

            dt_sdsc, arg_map = compile_op_spec(kernel_name, ks)
            sdscs.append(dt_sdsc)
            arg_mappings.append(arg_map)

        # Write SDSCs to file system, invoke backend compiler, and return KernelRunner
        kernel_output_dir = get_output_dir(kernel_name)
        if _SDSC_BUNDLE:
            for idx, sdsc in enumerate(sdscs):
                with open(
                    os.path.join(kernel_output_dir, f"sdsc_{idx}.json"), "w"
                ) as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(sdsc, file, indent=2)
            with open(os.path.join(kernel_output_dir, "bundle.mlir"), "w") as file:
                logger.info(f"Generating {file.name}")
                file.write("module {\n")
                file.write("\tfunc.func @sdsc_bundle() {\n")
                for i in range(len(sdscs)):
                    file.write(
                        '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_'
                        + f"{i}"
                        + '.json"}\n'
                    )
                file.write("\t\treturn\n")
                file.write("\t}\n")
                file.write("}\n")

            subprocess.run(
                ["dxp_standalone", "--bundle", "-d", kernel_output_dir], check=True
            )
            convert_artifacts(kernel_output_dir)

            return SpyreSDSCKernelRunner(kernel_name, [kernel_output_dir], arg_mappings)
        else:
            # Process each SuperDSC separately
            sdsc_dirs = []
            for sdsc in enumerate(sdscs):
                kernel_output_dir = get_output_dir(kernel_name)
                subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
                os.makedirs(subdir, exist_ok=True)
                with open(os.path.join(subdir, "sdsc.json"), "w") as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(dt_sdsc, file, indent=2)
                sdsc_dirs.append(kernel_output_dir)
                arg_mappings.append(arg_map)

            for dir in sdsc_dirs:
                subprocess.run(["dxp_standalone", "-d", dir], check=True)
                convert_artifacts(dir)

            return SpyreSDSCKernelRunner(kernel_name, sdsc_dirs, arg_mappings)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
