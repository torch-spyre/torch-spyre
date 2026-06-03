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

from torch_spyre._inductor import spyre_hint

torch.manual_seed(0xAFFE)

a = torch.rand(64, 128, dtype=torch.float16)  # M K
b = torch.rand(64, 128, dtype=torch.float16)  # M K
c = torch.rand(128, 256, dtype=torch.float16)  # K N
d = torch.rand(64, 256, dtype=torch.float16)  # M N


def f(a, b, c, d):
    with spyre_hint(tiles={"K": 2}):
        with spyre_hint(tiles={"M": 4}):
            x = a + b
        y = x @ c
    return y + d


cpu = f(a, b, c, d)

a_dev = a.to("spyre")
b_dev = b.to("spyre")
c_dev = c.to("spyre")
d_dev = d.to("spyre")

aiu = torch.compile(f)(a_dev, b_dev, c_dev, d_dev).cpu()

print(cpu)
print(aiu)

print(torch.abs(aiu - cpu).amax())
