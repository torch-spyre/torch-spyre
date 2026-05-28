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

import math

import torch
from torch_spyre._inductor import config, spyre_hint
import torch_spyre._inductor.propagate_named_dims as pnd

declare_tensor_dim = pnd.declare_tensor_dim
name_tensor_dims = pnd.name_tensor_dims

torch.manual_seed(0xAFFE)

B, H, Lq, Lk, D = 1, 8, 256, 256, 64
block_size = 128

queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)


def flash(queries, keys, values):
    output = torch.zeros_like(queries)
    M = torch.full(
        (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
    )
    denominator = torch.zeros((B, H, Lq), device=queries.device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(math.sqrt(D))

#    with spyre_hint(slices={"B": 1}):
#        with spyre_hint(slices={"H": 1}):
    with spyre_hint(slices={"Lk": Lk // block_size}):
        keys_T = keys.transpose(-1, -2).contiguous()

        scores = torch.matmul(queries * scale, keys_T * scale)  # B, H, Lq, Lk
        scores = scores.transpose(-1, -2).contiguous()  # avoid stick reduction: B, H, Lk, Lq
        block_max = torch.amax(scores, dim=-2)  # B, H, Lq
        max_running = torch.maximum(M, block_max)  # B, H, Lq

        exp_scores = torch.exp(
            scores - max_running.unsqueeze(-2)
        )  # B, H, Lk, Lq
        correction = torch.exp(M - max_running)  # B, H, Lq

        denominator = denominator * correction + exp_scores.sum(dim=-2)  # B, H, Lq
        output = (
            output * correction.unsqueeze(-1)
            + torch.matmul(exp_scores.transpose(-1, -2), values)  # B, H, Lq, D
        )
        M = max_running

    return output / denominator.unsqueeze(-1)


# CPU reference
ref = flash(queries_t, keys_t, values_t)

# Device run
queries_dev = queries_t.to("spyre")
keys_dev = keys_t.to("spyre")
values_dev = values_t.to("spyre")

declare_tensor_dim("B", B)
declare_tensor_dim("H", H)
declare_tensor_dim("Lq", Lq)
declare_tensor_dim("Lk", Lk)
declare_tensor_dim("D", D)

name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
# name_tensor_dims(values_dev, ["B", "H2", "Lk", "D"])  # H2 instead of H to avoid tiling second input of matmul on line 60
name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])  

config.coarse_tiling = True

result = torch.compile(flash)(queries_dev, keys_dev, values_dev).cpu()

print("ref:   ", ref.flatten()[:8])
print("result:", result.flatten()[:8])
print("max abs diff:", torch.abs(ref - result).amax().item())

torch.testing.assert_close(
    result,
    ref,
    equal_nan=True,
    atol=1.0,
    rtol=0.1,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("PASSED")
