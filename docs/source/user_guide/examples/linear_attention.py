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
"""Causal linear attention on Spyre with ``torch.ops.spyre.linear_attention``.

Linear attention replaces softmax with a feature map ``phi`` so attention can be
written as a recurrence over a fixed-size state ``S`` (``[Dk, Dv]`` per head)
instead of an ``L x L`` score matrix::

    S_t = S_{t-1} + phi(k_t)^T v_t
    z_t = z_{t-1} + phi(k_t)             # normalizer
    o_t = (phi(q_t) S_t) / (phi(q_t) z_t)

The op computes the unnormalized part of this (``S`` and ``phi(q) S``) in
chunks. This script shows the three things a caller owns:

1. the feature map (``elu(x) + 1``, applied on the host here because ``elu`` is
   not a listed Spyre op),
2. the normalizer, computed in the *same* op call by appending a block of ones
   to ``V`` so the last ``STICK`` columns of the output are ``phi(q) z``, and
3. continuing a sequence by feeding one call's ``final_state`` into the next,
   which is how chunked prefill and token-by-token decode work.

It compares the compiled Spyre result against a float64 quadratic-form
reference on CPU. It requires a Spyre device to run.
"""

import torch
import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

B, H, L, D = 1, 4, 200, 64  # L is not a multiple of chunk_size: the op pads it.
CHUNK_SIZE = 64
STICK = 64  # fp16 elements per 128-byte stick
EPS = 1e-3


def feature_map(x):
    """Positive feature map from Katharopoulos et al. (2020)."""
    return F.elu(x) + 1.0


def normalized_linear_attention(q, k, v, initial_state=None):
    """Compiled-region body: one op call yields the output and the normalizer.

    ``v`` is extended with ``STICK`` columns of ones, so the state becomes
    ``[Dk, Dv + STICK]`` and its last block accumulates ``sum_t phi(k_t)``. The
    extra width keeps every dimension stick-aligned.
    """
    ones = torch.ones(*v.shape[:-1], STICK, dtype=v.dtype, device=v.device)
    v_aug = torch.cat([v, ones], dim=-1)
    out_aug, state = torch.ops.spyre.linear_attention(
        q, k, v_aug, initial_state, CHUNK_SIZE, True, 1.0
    )
    numerator = out_aug[..., : v.size(-1)]
    denominator = out_aug[..., v.size(-1) : v.size(-1) + 1]  # every column is equal
    return numerator / (denominator + EPS), state


def reference(q, k, v):
    """Quadratic-form normalized causal linear attention in float64."""
    q, k, v = q.double(), k.double(), v.double()
    scores = torch.einsum("bhlk,bhmk->bhlm", q, k)
    scores = scores * torch.tril(torch.ones(q.size(2), q.size(2), dtype=torch.double))
    return (scores @ v) / (scores.sum(-1, keepdim=True) + EPS)


# Build inputs on CPU. Keys are scaled so the fp16 state stays well in range.
Q = feature_map(torch.randn(B, H, L, D)).half()
K = (feature_map(torch.randn(B, H, L, D)) * D**-0.5).half()
V = torch.randn(B, H, L, D, dtype=torch.float16)

# Compute the reference on CPU
cpu_result = reference(Q, K, V).half()

# Send inputs to device
Q_device, K_device, V_device = Q.to(DEVICE), K.to(DEVICE), V.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host.
# The op has no eager implementation on Spyre, so it must run under torch.compile.
compiled = torch.compile(normalized_linear_attention, backend="inductor")
compiled_result, final_state = compiled(Q_device, K_device, V_device)
compiled_result = compiled_result.cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
torch.testing.assert_close(compiled_result, cpu_result, atol=0.1, rtol=0.1)
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")

# Continue the sequence: process the first 128 tokens, then the rest with the
# state the first call returned. The state has a fixed size no matter how many
# tokens it has absorbed, unlike a KV cache.
split = 128
head_out, head_state = compiled(
    Q_device[:, :, :split], K_device[:, :, :split], V_device[:, :, :split]
)
tail_out, _ = compiled(
    Q_device[:, :, split:], K_device[:, :, split:], V_device[:, :, split:], head_state
)
resumed = torch.cat([head_out, tail_out], dim=2).cpu()
torch.testing.assert_close(resumed, cpu_result, atol=0.1, rtol=0.1)
print(f"Max delta resumed vs. CPU: {torch.abs(resumed - cpu_result).max()}")
