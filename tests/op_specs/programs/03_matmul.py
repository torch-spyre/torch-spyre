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

"""Matmul: one OpSpec, on the matrix engine rather than the vector unit."""

import torch

import torch_spyre  # noqa: F401  -- registers the "spyre" device

torch.manual_seed(0xAFFE)

a = torch.rand(64, 128, dtype=torch.float16)
b = torch.rand(128, 256, dtype=torch.float16)

expected = a @ b
got = torch.compile(lambda x, y: x @ y)(a.to("spyre"), b.to("spyre")).cpu()

# fp16 accumulation over K=128, so the tolerance here is the dtype, not a bug.
print(f"max|got - expected| = {(got - expected).abs().max().item()}")
