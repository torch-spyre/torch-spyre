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

# Real-world decode (Gemma-like): 32 tokens, 128 experts, hidden 2816, expert width 704
# the loop can run up to 128 experts; the chip runs only the experts the batch picked


def decode(tokens, up, down, count):
    count = count.abs()  # count must be made on Spyre (one small op)

    def one_expert(total, ops):
        my_up, my_down = ops  # [1, 2816, 704] | [1, 704, 2816]: this expert's weights
        h = tokens @ my_up[0]  # [32, 2816] @ [2816, 704] -> [32, 704]
        return total + h @ my_down[0], None  # [32, 704] @ [704, 2816] -> [32, 2816]

    total, _ = for_each_tile(
        one_expert,
        (up, down),
        dims=(Slice(0, 1), Slice(0, 1)),
        tile_size=1,
        init=torch.zeros_like(tokens),  # running sum, starts at 0
        active_count=count,
    )  # chip runs only `count` experts
    return total  # [32, 2816]


tokens = torch.randn(32, 2816, dtype=torch.float16)  # 32 tokens (one decode step)
up = (
    torch.randn(128, 2816, 704, dtype=torch.float16) * 0.02
)  # 128 experts, already in picked order
down = torch.randn(128, 704, 2816, dtype=torch.float16) * 0.02

compiled = torch.compile(decode, backend="inductor", fullgraph=True)
tokens_d, up_d, down_d = tokens.to("spyre"), up.to("spyre"), down.to("spyre")

for picked in [1, 8, 64, 128]:  # how many experts the batch picked
    count = torch.full((1, 32), float(picked))  # [1, 32]: one stick, chip reads col 0
    out = compiled(tokens_d, up_d, down_d, count.to("spyre")).cpu()
    cpu = sum(
        (tokens.float() @ up[e].float()) @ down[e].float() for e in range(picked)
    )  # first `picked` experts
    err = ((out.float() - cpu).norm() / cpu.norm()).item()
    print(
        f"count {picked:3d} -> sum of first {picked} experts, rel err {err:.4f}",
        "PASS" if err < 0.02 else "FAIL",
    )
