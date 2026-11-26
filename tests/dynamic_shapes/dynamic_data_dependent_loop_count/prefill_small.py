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


def prefill(tokens, weights, counts):
    counts = counts.abs()  # count must be made on Spyre (one small op)

    def one_expert(_, ops):
        my_tokens, my_weight, my_count = ops  # [256, 128] | [1, 128, 64] | [1, 32]

        def one_tile(_, tile):
            return None, tile[0] @ my_weight[0]  # [64, 128] @ [128, 64] -> [64, 64]

        _, out = for_each_tile(
            one_tile,
            (my_tokens,),
            dims=(0,),
            tile_size=64,
            out_dim=0,
            active_count=my_count,
        )  # chip runs only my_count tiles
        return None, out  # [256, 64]

    _, out = for_each_tile(
        one_expert,
        (tokens, weights, counts),
        dims=(
            Slice(0, 256),
            Slice(0, 1),
            Slice(0, 1),
        ),  # each expert: 256 rows, 1 weight, 1 count
        tile_size=1,
        out_dim=0,
    )
    return out  # [1024, 64]


tokens = torch.randn(1024, 128, dtype=torch.float16)  # 4 experts x 256 tokens
weights = torch.randn(4, 128, 64, dtype=torch.float16) * 0.1  # 1 weight per expert
counts = (
    torch.tensor([[1.0], [3.0], [0.0], [4.0]]).expand(4, 32).contiguous()
)  # tiles per expert

compiled = torch.compile(prefill, backend="inductor", fullgraph=True)
out = compiled(tokens.to("spyre"), weights.to("spyre"), counts.to("spyre")).cpu()

for e in range(4):
    done = []
    for t in range(4):
        rows = slice(e * 256 + t * 64, e * 256 + t * 64 + 64)  # tile t of expert e
        cpu = tokens[rows].float() @ weights[e].float()
        done.append(torch.allclose(out[rows].float(), cpu, atol=0.5, rtol=0.05))
    print(f"expert {e}: count {int(counts[e, 0])} -> tiles done {done}")
