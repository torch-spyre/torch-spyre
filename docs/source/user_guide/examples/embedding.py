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
"""Embedding lookup on Spyre.

This script looks up rows of an embedding table with F.embedding. The table's
padding row is zeroed the way nn.Embedding zeroes it at construction, and the
input references that row, so the lookup returns zeros there. It compares the
compiled Spyre result against a CPU reference. It requires a Spyre device to
run.
"""

import torch
import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

padding_idx = 0

# Build inputs on CPU. Zero the padding row in the table, as nn.Embedding does
# when it is built with padding_idx, and reference it from the input.
indices = torch.tensor([[1, 2, 0, 5], [4, 0, 2, 9]], dtype=torch.int64)
weight = torch.rand(10, 3, dtype=torch.float16)
weight[padding_idx] = 0.0


def embedding(indices, weight):
    return F.embedding(indices, weight, padding_idx=padding_idx)


# Compute the reference on CPU
cpu_result = embedding(indices, weight)

# Send inputs to device
indices_device = indices.to(DEVICE)
weight_device = weight.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(embedding)
compiled_result = compiled(indices_device, weight_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
torch.testing.assert_close(compiled_result, cpu_result, atol=0.1, rtol=0.1)
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
