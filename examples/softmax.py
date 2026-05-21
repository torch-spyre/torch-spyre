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

import torch
import torch_spyre._inductor.passes as passes

import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

R, C = 512, 1024
x = torch.rand(R, C, dtype=torch.float16)


def func(a):
    return torch.softmax(a, dim=0)


cpu_result = func(x)

x_device = x.to(DEVICE)

declare_real_dim("R", R)
declare_real_dim("C", C)

annotate_real_dims(x_device, ["R", "C"])

eager_result = func(x_device).cpu()

compiled_result = torch.compile(func)(x_device).cpu()

device_delta = torch.abs(eager_result - compiled_result).max()
cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
print(f"Max delta Compiled Spyre vs. Eager Spyre: {device_delta}")
