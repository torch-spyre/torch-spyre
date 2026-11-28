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

import torch
import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor.wsr.for_each_tile import Slice, for_each_tile


def decode(tokens, weights, count):
    count = count.abs()  # count must be made on Spyre (one small op)

    def one_expert(total, ops):
        (my_weight,) = ops  # [1, 128, 128]: this expert's weight
        return total + tokens @ my_weight[
            0
        ], None  # [32, 128] @ [128, 128] -> [32, 128]

    total, _ = for_each_tile(
        one_expert,
        (weights,),
        dims=(Slice(0, 1),),
        tile_size=1,
        init=torch.zeros_like(tokens),  # running sum, starts at 0
        active_count=count,
    )  # chip runs only `count` experts
    return total  # [32, 128]


tokens = torch.randn(32, 128, dtype=torch.float16)  # 32 tokens (one decode step)
weights = (
    torch.randn(4, 128, 128, dtype=torch.float16) * 0.1
)  # 4 experts, already in picked order

compiled = torch.compile(decode, backend="inductor", fullgraph=True)

for picked in [1, 2, 4]:  # how many experts the batch picked
    count = torch.full((1, 32), float(picked))  # [1, 32]: one stick, chip reads col 0
    out = compiled(tokens.to("spyre"), weights.to("spyre"), count.to("spyre")).cpu()
    cpu = sum(
        tokens.float() @ weights[e].float() for e in range(picked)
    )  # first `picked` experts
    ok = torch.allclose(out.float(), cpu, atol=0.5, rtol=0.05)
    print(f"count {picked} -> sum of first {picked} experts: {ok}")
