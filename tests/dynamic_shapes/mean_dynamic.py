# Copyright 2026 The Torch-Spyre Authors.
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

# Dynamic shape version of mean.py
# Right now this is a work in progress — setting dynamic=True lets Dynamo
# trace with symbolic shapes, but the SDSC we generate still ends up fully
# concrete. Getting symbolic shapes all the way into the SDSC is tracked
# in issues #220, #1371, #1372, #1373.

import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)


def mean_fn(a):
    return torch.mean(a, dim=0)


# Compile with dynamic=True
compiled_fn = torch.compile(mean_fn, dynamic=True)

x = torch.rand(128, 64, dtype=torch.float16)

cpu_result = mean_fn(x)

x_device = x.to(DEVICE)
compiled_result = compiled_fn(x_device).cpu()

# Compare results
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
