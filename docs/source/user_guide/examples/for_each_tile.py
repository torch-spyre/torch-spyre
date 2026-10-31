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

import torch

from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

torch.manual_seed(0xAFFE)

a = torch.rand(64, 128, dtype=torch.float16)  # M K
b = torch.rand(64, 128, dtype=torch.float16)  # M K
c = torch.rand(128, 256, dtype=torch.float16)  # K N
d = torch.rand(64, 256, dtype=torch.float16)  # M N


def f(a, b, c, d):
    # x = a + b, in four tiles along M (a "map": each trip produces a slice of x).
    def add_body(_, tiles):
        a_tile, b_tile = tiles
        return None, a_tile + b_tile

    _, x = for_each_tile(add_body, (a, b), dims=(0, 0), tile_size=16, out_dim=0)

    # y = x @ c, in two tiles along K (a "carry": each trip adds a partial product).
    def matmul_body(acc, tiles):
        x_tile, c_tile = tiles
        return acc + x_tile @ c_tile, None

    y, _ = for_each_tile(
        matmul_body,
        (x, c),
        dims=(1, 0),
        tile_size=64,
        init=torch.zeros(64, 256, dtype=x.dtype, device=x.device),
    )
    return y + d


def reference(a, b, c, d):
    return (a + b) @ c + d


cpu = reference(a, b, c, d)

a_dev = a.to("spyre")
b_dev = b.to("spyre")
c_dev = c.to("spyre")
d_dev = d.to("spyre")

# for_each_tile needs the whole function captured: fullgraph=True.
aiu = torch.compile(f, fullgraph=True)(a_dev, b_dev, c_dev, d_dev).cpu()

print(cpu)
print(aiu)

print(torch.abs(aiu - cpu).amax())
