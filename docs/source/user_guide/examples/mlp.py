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
"""SwiGLU feed-forward block on Spyre.

This script computes a SwiGLU MLP, silu(x @ gate) * (x @ up) @ down, and
compares the compiled Spyre result against a CPU reference. It requires a
Spyre device to run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

seq_len = 256
emb_dim = 1024

# Build inputs on CPU
x = torch.randn(seq_len, emb_dim, dtype=torch.float16)
gate_proj_weight = torch.empty(emb_dim, 4 * emb_dim, dtype=torch.float16)
up_proj_weight = torch.empty(emb_dim, 4 * emb_dim, dtype=torch.float16)
down_proj_weight = torch.empty(4 * emb_dim, emb_dim, dtype=torch.float16)
nn.init.kaiming_uniform_(gate_proj_weight)
nn.init.kaiming_uniform_(up_proj_weight)
nn.init.kaiming_uniform_(down_proj_weight)


def mlp(x, gate, up, down):
    gate_out = x @ gate
    up_out = x @ up
    swiglu_out = up_out * F.silu(gate_out)
    return swiglu_out @ down


# Compute the reference on CPU
cpu_result = mlp(x, gate_proj_weight, up_proj_weight, down_proj_weight)

# Send inputs to device
x_device = x.to(DEVICE)
gate_device = gate_proj_weight.to(DEVICE)
up_device = up_proj_weight.to(DEVICE)
down_device = down_proj_weight.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(mlp)
compiled_result = compiled(x_device, gate_device, up_device, down_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
