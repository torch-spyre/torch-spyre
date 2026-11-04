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
"""Scaled dot-product attention assembled from matmul and softmax.

This script computes a single attention block as q @ k^T scaled, softmax, then
@ v, and compares the compiled Spyre result against a CPU reference. It is
compile-only: the eager matmul path is unsupported here because mm on Spyre
tensors needs the hardware session that torch.compile sets up. It requires a
Spyre device to run.
"""

import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

H = 4  # heads per group
Q = 64  # query length
L = 256  # key/value length
D = 128  # head dimension

# Build inputs on CPU
q = torch.randn(H * Q, D, dtype=torch.float16)
k = torch.randn(L, D, dtype=torch.float16)
v = torch.randn(L, D, dtype=torch.float16)
# The softmax scale is one scalar, repeated into a per-column vector so it
# broadcasts against the qk scores without a scalar-tensor broadcast.
sm_scale = torch.tensor(1 / (D**0.5), dtype=torch.float16).repeat(k.shape[0])


def attn(q, k, v, sm_scale):
    qk = q @ k.transpose(-1, -2).contiguous()
    qk = qk * sm_scale
    p = qk.softmax(dim=-1)
    return p @ v


# Compute the reference on CPU
cpu_result = attn(q, k, v, sm_scale)

# Send inputs to device
q_device = q.to(DEVICE)
k_device = k.to(DEVICE)
v_device = v.to(DEVICE)
sm_scale_device = sm_scale.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(attn)
compiled_result = compiled(q_device, k_device, v_device, sm_scale_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
