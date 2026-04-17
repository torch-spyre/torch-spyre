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

# Demonstrates work_division_hint to keep mm output and bias-add splits
# compatible: both ops split M by 4 and N by 2 (total 8 cores).
#
# F.linear(x, w, b) decomposes into mm + add. The hint [4, 2, 1] targets
# the mm (M, N, K order). The bias add is a pointwise op over the mm output
# shape (M, N), so using the same [4, 2] hint keeps splits aligned.

import torch

from torch_spyre._inductor.work_division_hint import work_division_hint

DEVICE = "spyre"

M, N, K = 512, 256, 128

x = torch.randn([M, K], dtype=torch.float16).to(DEVICE)
w = torch.randn([N, K], dtype=torch.float16).to(DEVICE)
b = torch.randn([N], dtype=torch.float16).to(DEVICE)


@torch.compile
def linear_with_bias(x, w, b):
    # mm hint: [M_split, N_split, K_split] = [4, 2, 1]  -> 8 cores on M x N
    # add hint: [M_split, N_split]         = [4, 2]     -> same 8-core split
    # Both hints are attached to every node in the block; the mm node matches
    # length 3 and the add node matches length 2, so each gets the right hint.
    with work_division_hint([4, 2, 1]):
        mm_out = x @ w.T
    with work_division_hint([4, 2]):
        out = mm_out + b
    return out


out = linear_with_bias(x, w, b)
print(out)
