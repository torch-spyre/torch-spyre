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

import os
from torch_spyre._C import launch_kernel
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre.tensor import SpyreTensor

logger = get_inductor_logger("kernel_runner")


def _unwrap_spyre_tensor(arg):
    return arg._t if isinstance(arg, SpyreTensor) else arg


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    def __init__(self, name: str, code_dirs: list[str], arg_mappings: list[list[int]]):
        self.kernel_name = name
        self.code_dirs = code_dirs
        self.arg_mappings = arg_mappings

    def run(self, *args, **kw_args):
        for i in range(len(self.code_dirs)):
            g2 = os.path.join(self.code_dirs[i], "g2.graph.cbor")
            logger.info(f"RUN: {self.kernel_name}_{i} {g2}")
            actuals = [_unwrap_spyre_tensor(args[i]) for i in self.arg_mappings[i]]
            launch_kernel(g2, actuals)
