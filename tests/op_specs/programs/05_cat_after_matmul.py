# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Cat after a matmul: the worked example for comparing two runs.

49 tokens on a 7x7 grid go through a matmul, and an eager cat puts a class token
in front of them, as a vision transformer's patch embedding does. With
SPYRE_NDO_MATMUL_OUTPUT_REORDER=1 the matmul's output gets a layout that splits
the grid, and the cat writes every token one row early, over the class token.
"Comparing two runs" in the OpSpec Lab doc finds it. Once that is fixed, both
runs print a small difference and the example has nothing left to find.
"""

import torch

import torch_spyre  # noqa: F401  -- registers the "spyre" device

torch.manual_seed(0xAFFE)

patches = torch.rand(1, 7, 7, 64, dtype=torch.float16)
weight = torch.rand(64, 768, dtype=torch.float16)
cls = torch.rand(1, 1, 768, dtype=torch.float16)

# Compiled, so the flag lays out the matmul's output; the cat runs eagerly.
tokens = torch.compile(lambda p, w: p @ w)(patches.to("spyre"), weight.to("spyre"))
got = torch.cat([cls.to("spyre"), tokens.reshape(1, 49, 768)], dim=1).cpu()

expected = torch.cat([cls, (patches @ weight).reshape(1, 49, 768)], dim=1)
# fp16 accumulation over K=64 puts this near 0.1; a misplaced row is near 20.
print(f"max|got - expected| = {(got - expected).abs().max().item()}")
