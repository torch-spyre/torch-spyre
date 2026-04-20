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

# Demonstrates work_division_hint to keep mm and bias-add splits aligned:
# the matmul splits M by 4 and N by 2, and the add uses the same M/N split.
#
# epilogue_fusion is not needed here.  Even if Inductor fuses mm + add into
# addmm, Spyre's addmm decomposition (decompositions.py) breaks it back into
# separate mm and add ComputedBuffers that each go through core division
# planning independently.

import torch
from torch_spyre._inductor.work_division_hint import work_division_hint

torch.manual_seed(0xAFFE)
DEVICE = "spyre"

M, N, K = 2048, 2048, 65536

x = torch.randn([M, K], dtype=torch.float16).to(DEVICE)

w = torch.randn([N, K], dtype=torch.float16)
torch.nn.init.xavier_uniform_(w)
w = w.to(DEVICE)

b = torch.randn([N], dtype=torch.float16).to(DEVICE)


@torch.compile
def linear_with_bias(x, w, b):
    with work_division_hint([4, 2, 1]):
        mm_out = x @ w.T  # matmul iteration space: [M, N, K]
    with work_division_hint([4, 2]):
        out = mm_out + b  # pointwise iteration space: [M, N]
    return out


out = linear_with_bias(x, w, b)
print(out)
