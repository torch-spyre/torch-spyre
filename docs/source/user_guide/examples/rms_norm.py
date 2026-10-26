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
"""RMS normalization on Spyre.

This script normalizes along dimension 0 to respect the reduction-dimension
restriction, so the activation and the weight are laid out as [D, T] and the
scalar epsilon is broadcast per column. It compares the compiled Spyre result
against a CPU reference. It requires a Spyre device to run.
"""

import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

EPS = 1e-6
T = 128
D = 256

# Build inputs on CPU. The weight is expanded to [D, T] to work around the
# on-device broadcast limitation, and eps is a broadcast scalar per column.
activation = torch.randn(D, T, dtype=torch.float16)
weight = torch.randn(D, dtype=torch.float16)
weight_expanded = weight.reshape(D, 1).expand(D, T).contiguous()
eps = torch.full([T], EPS, dtype=torch.float16)


def rms_norm(x, weight, eps):
    x_sq = x * x
    x_mean_sq = x_sq.mean(dim=0)
    return x * torch.rsqrt(x_mean_sq + eps)[None, :] * weight


# Compute the reference on CPU
cpu_result = rms_norm(activation, weight_expanded, eps)

# Send inputs to device
activation_device = activation.to(DEVICE)
weight_device = weight_expanded.to(DEVICE)
eps_device = eps.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(rms_norm)
compiled_result = compiled(activation_device, weight_device, eps_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
torch.testing.assert_close(compiled_result, cpu_result, atol=0.1, rtol=0.1)
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
