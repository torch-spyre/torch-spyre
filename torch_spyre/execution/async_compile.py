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
from torch_spyre._inductor.codegen.superdsc import generate_sdsc
from torch_spyre._inductor.constants import SEGMENT_OFFSETS
from torch_spyre._inductor.logging_utils import get_inductor_logger
from . import ConstantArg, OpSpec, UnimplementedOp
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")

_argument_names = ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]


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
        sdsc_dirs = []
        arg_mappings = []
        for ks in specs:
            if isinstance(ks, UnimplementedOp):
                print(f"WARNING: Compiling unimplemented {ks.op} to runtime exception")
                return SpyreUnimplementedRunner(kernel_name, ks.op)

            inputs = []
            outputs = []
            arg_map = []
            for index, ts in enumerate(ks.args):
                # use node seq (idx in nodes) to verify whether to reuse lx for this buffer,
                # in case same Op used twice in sequence and only want pin 1 of them
                lx_addr = None
                for k, addr in getattr(ts, "allocation", {}).items():
                    if kernel_name.split("_")[-1] == k.replace("lx:", ""):
                        lx_addr = addr

                if isinstance(ts, ConstantArg):
                    raise RuntimeError("TOOO: implement SDSC generation for constants")
                elif ts.is_input:
                    inputs.append(
                        {
                            "name": _argument_names[index],
                            "it_dim_map": ts.it_dim_map,
                            "device_layout": ts.device_layout,
                            "lx_addr": lx_addr,
                        }
                    )
                    arg_map.append(ts.arg_index)
                else:
                    outputs.append(
                        {
                            "name": _argument_names[index],
                            "it_dim_map": ts.it_dim_map,
                            "device_layout": ts.device_layout,
                            "lx_addr": lx_addr,
                        }
                    )
                    arg_map.append(ts.arg_index)
            kernel_descriptor = {
                "name": kernel_name,
                "reduction": ks.is_reduction,
                "op": ks.op,
                "dimensions": ks.iteration_space,
                "inputs": inputs,
                "outputs": outputs,
            }
            if ks.op_info is not None:
                kernel_descriptor["op_info"] = ks.op_info
            pointers = dict(zip(_argument_names, SEGMENT_OFFSETS))
            dt_sdsc = generate_sdsc(pointers, **kernel_descriptor)
            kernel_output_dir = get_output_dir(kernel_name)
            subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, "sdsc.json"), "w") as file:
                logger.info(f"Generating {file.name}")
                json.dump(dt_sdsc, file, indent=2)
            sdsc_dirs.append(kernel_output_dir)
            arg_mappings.append(arg_map)

        # 2. Invoke the backend compiler on each sdsc.json
        for dir in sdsc_dirs:
            subprocess.run(["dxp_standalone", "-d", dir], check=True)
            convert_artifacts(dir)

        # 3. Construct the KernelRunner
        return SpyreSDSCKernelRunner(kernel_name, sdsc_dirs, arg_mappings)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
