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

# Real-world prefill: 8192 tokens, tile of 512 tokens -> an expert can get up to 8192 / 512 = 16 tiles
# 128 experts, each with room for 16 tiles = 8192 rows


def prefill(tokens, weights, counts):
    counts = counts.abs()  # count must be made on Spyre (one small op)

    def one_expert(_, ops):
        my_tokens, my_weight, my_count = ops  # [8192, 128] | [1, 128, 64] | [1, 32]

        def one_tile(_, tile):
            return None, tile[0] @ my_weight[0]  # [512, 128] @ [128, 64] -> [512, 64]

        _, out = for_each_tile(
            one_tile,
            (my_tokens,),
            dims=(0,),
            tile_size=512,
            out_dim=0,
            active_count=my_count,
        )  # chip runs only my_count tiles
        return None, out  # [8192, 64]

    _, out = for_each_tile(
        one_expert,
        (tokens, weights, counts),
        dims=(
            Slice(0, 8192),
            Slice(0, 1),
            Slice(0, 1),
        ),  # each expert: 8192 rows, 1 weight, 1 count
        tile_size=1,
        out_dim=0,
    )
    return out  # [128 * 8192, 64]


tokens = torch.randn(128 * 8192, 128, dtype=torch.float16)  # 128 experts x 8192 rows
weights = torch.randn(128, 128, 64, dtype=torch.float16) * 0.1  # 1 weight per expert
tiles = torch.randint(0, 17, (128,))  # tiles per expert: 0..16
counts = tiles.float().view(128, 1).expand(128, 32).contiguous()  # [128, 32]

compiled = torch.compile(prefill, backend="inductor", fullgraph=True)
out = compiled(tokens.to("spyre"), weights.to("spyre"), counts.to("spyre")).cpu()

bad = 0
for e in range(128):
    want = min(max(int(tiles[e]), 1), 16)  # 0 still runs 1
    done = 0
    for t in range(16):
        rows = slice(e * 8192 + t * 512, e * 8192 + t * 512 + 512)  # tile t of expert e
        cpu = tokens[rows].float() @ weights[e].float()
        done += torch.allclose(out[rows].float(), cpu, atol=0.5, rtol=0.05)
    bad += done != want
    if e < 5:
        print(
            f"expert {e}: count {int(tiles[e])} -> tiles done {done} (expected {want})"
        )
print(
    f"... all 128 experts checked, {128 - bad} exact, {bad} wrong ->",
    "PASS" if bad == 0 else "FAIL",
)
