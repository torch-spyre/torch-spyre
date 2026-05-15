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

"""Baby step: fp32 input for non-linear LayerNorm, fp16 weight/bias."""

import os

import torch
import torch_spyre  # noqa: F401

HIDDEN_DIM = 4096
BATCH = 32

torch._dynamo.reset()
gen = torch.Generator().manual_seed(42)

# Input is fp32 — the non-linear reduction part (exx2, layernormscale) will see fp32
x = torch.randn(BATCH, HIDDEN_DIM, generator=gen, dtype=torch.float32)
# Weight/bias also fp32 — avoids mixed-dtype issues in layernormnorm
weight = torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32)
bias = torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32)

# CPU fp32 baseline
baseline = torch.nn.functional.layer_norm(x, [HIDDEN_DIM], weight=weight, bias=bias)

device = torch.device("spyre")
x_dev = x.to(device)  # fp32 on Spyre
w_dev = weight.to(device)  # fp32 on Spyre
b_dev = bias.to(device)  # fp32 on Spyre


@torch.compile(backend="inductor")
def compiled_fn(x, w, b):
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=w, bias=b)


print(f"x dtype: {x_dev.dtype}, weight dtype: {w_dev.dtype}, bias dtype: {b_dev.dtype}")
print(f"SENCORES={os.environ.get('SENCORES', '32')}")

out = compiled_fn(x_dev, w_dev, b_dev)
result = out.cpu().float()

diff = (result - baseline).abs()
print(f"max_abs={diff.max().item():.6e}  mean_abs={diff.mean().item():.6e}")
print("PASS" if diff.max().item() < 1e-5 else "FAIL")
