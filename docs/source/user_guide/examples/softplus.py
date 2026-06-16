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

import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

# Create input tensor
x = torch.rand(512, 1024, dtype=torch.float16)

# Compute softplus on the cpu
cpu_result = F.softplus(x, beta=1.0, threshold=20.0)

# Send input tensor to device
x_device = x.to(DEVICE)

# Compute softplus on the device in compiled mode and get the result back to the host
compiled_sp = torch.compile(lambda a, b, c: F.softplus(a, b, c))
compiled_result = compiled_sp(x_device, 1.0, 20.0).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
