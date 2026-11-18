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
"""Flash attention with online softmax on Spyre.

This script computes attention block by block, carrying a running maximum and
denominator so the softmax never materializes the full score matrix. It
compares the compiled Spyre result against PyTorch's native scaled
dot-product attention on CPU. It requires a Spyre device to run.
"""

import math

import torch
import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

B, H, L, D = 1, 8, 256, 64
block_size = 128

# Build inputs on CPU
Q = torch.randn(B, H, L, D, dtype=torch.float16)
K = torch.randn(B, H, L, D, dtype=torch.float16)
V = torch.randn(B, H, L, D, dtype=torch.float16)


def flash(Q, K, V, block_size):
    output = torch.zeros_like(Q)
    M = torch.full((B, H, L), float("-inf"), device=Q.device, dtype=torch.float16)
    denominator = torch.zeros((B, H, L), device=Q.device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)

    for start in range(0, L, block_size):
        end = start + block_size
        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]
        K_block_T = K_block.transpose(-1, -2).contiguous()

        scores = torch.matmul(Q, K_block_T) * scale  # B, H, L, Block
        scores = scores.transpose(-1, -2).contiguous()  # avoid stick reduction
        block_max = torch.amax(scores, dim=-2)
        max_running = torch.maximum(M, block_max)

        exp_scores = torch.exp(scores - max_running.unsqueeze(-2))  # B, H, Block, L
        correction = torch.exp(M - max_running)

        denominator = denominator * correction + exp_scores.sum(dim=-2)
        output = output * correction.unsqueeze(-1) + torch.bmm(
            exp_scores.transpose(-1, -2).flatten(0, 1), V_block.flatten(0, 1)
        ).unflatten(0, (B, H))

        M = max_running

    return output / denominator.unsqueeze(-1)


# Compute the reference on CPU with PyTorch's native scaled dot-product attention
cpu_result = F.scaled_dot_product_attention(Q, K, V)

# Send inputs to device
Q_device = Q.to(DEVICE)
K_device = K.to(DEVICE)
V_device = V.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(flash)
compiled_result = compiled(Q_device, K_device, V_device, block_size).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
torch.testing.assert_close(compiled_result, cpu_result, atol=0.1, rtol=0.1)
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
