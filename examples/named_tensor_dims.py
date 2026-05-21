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
declare_tensor_dim = prd.declare_tensor_dim
name_tensor_dims = prd.name_tensor_dims

torch.manual_seed(0xAFFE)

A, B, C = 64, 128, 256
x = torch.rand(A, B, dtype=torch.float16) * 0.01
y = torch.rand(B, C, dtype=torch.float16) * 0.01
z = torch.rand(A, C, dtype=torch.float16) * 0.01
v = torch.rand(A * C, dtype=torch.float16) * 0.01  # flat 1D, logically [A, C]


def f(x, y, z, v):
    return (x @ y + z + v.view(A, C)).sum(0)


r = f(x, y, z, v)
x_dev = x.to("spyre")
y_dev = y.to("spyre")
z_dev = z.to("spyre")
v_dev = v.to("spyre")

declare_tensor_dim("A", A)
declare_tensor_dim("B", B)
declare_tensor_dim("C", C)

name_tensor_dims(x_dev, ["A", "B"])
name_tensor_dims(y_dev, ["B", "C"])
name_tensor_dims(z_dev, ["A", "C"])
name_tensor_dims(v_dev, ["A", "C"])  # 1D flat annotated with its logical shape

result = torch.compile(f)(x_dev, y_dev, z_dev, v_dev).cpu()

print(r)
print(result)
print(torch.abs(r - result).amax())
