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

"""Softmax: one aten op that becomes six fused OpSpecs."""

import torch

import torch_spyre  # noqa: F401  -- registers the "spyre" device

torch.manual_seed(0xAFFE)

x = torch.rand(64, 256, dtype=torch.float16)

expected = torch.softmax(x, dim=-1)
got = torch.compile(lambda t: torch.softmax(t, dim=-1))(x.to("spyre")).cpu()

print(f"max|got - expected| = {(got - expected).abs().max().item()}")
