# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This example file demonstrates how to run a sequence of op_specs on Spyre.

from sympy import sympify

import torch

from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, UnimplementedOp, TensorArg


def execute_ops(ops: list[OpSpec | UnimplementedOp], tensors: list[torch.Tensor]):
    """
    Execute a sequence of op_specs provided a list of input/output tensors.
    """
    # copy tensors to device
    dev_tensors = [t.to(torch.device("spyre")) for t in tensors]

    # compile op_specs and invoke sdsc bundle
    SpyreAsyncCompile().sdsc("test_sdsc", ops).run(*dev_tensors)

    # copy device tensors back
    for t, dt in zip(tensors, dev_tensors):
        t[:] = dt.cpu()


# input/output tensors

torch.manual_seed(0xAFFE)

x = torch.rand(128, 256, dtype=torch.float16)
y = torch.rand(128, 256, dtype=torch.float16)
z = torch.rand(128, 256, dtype=torch.float16)

# sequence of op_specs

ops: list[OpSpec | UnimplementedOp] = [
    OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={
            sympify("c0"): (sympify("128"), 32),
            sympify("c1"): (sympify("256"), 1),
        },
        op_info={},
        args=[
            TensorArg(
                is_input=True,
                arg_index=0,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[4, 128, 64],
                device_coordinates=[
                    sympify("floor(c1/64)"),
                    sympify("c0"),
                    sympify("Mod(c1, 64)"),
                ],
                allocation={},
            ),
            TensorArg(
                is_input=True,
                arg_index=1,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[4, 128, 64],
                device_coordinates=[
                    sympify("floor(c1/64)"),
                    sympify("c0"),
                    sympify("Mod(c1, 64)"),
                ],
                allocation={},
            ),
            TensorArg(
                is_input=False,
                arg_index=2,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[4, 128, 64],
                device_coordinates=[
                    sympify("floor(c1/64)"),
                    sympify("c0"),
                    sympify("Mod(c1, 64)"),
                ],
                allocation={},
            ),
        ],
    )
]

# invoke sequence

execute_ops(ops, [x, y, z])

# print tensors

print(x)
print(y)
print(z)
