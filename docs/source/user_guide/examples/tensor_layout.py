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
"""Moving a tensor to Spyre with an explicit on-device layout.

This script builds a SpyreTensorLayout with a non-default dimension order and
moves an RMSNorm input to the device under that layout, so the sequence
dimension becomes the device stick. It compares the compiled Spyre result
against a CPU reference. It requires a Spyre device to run.
"""

import torch

from torch_spyre._C import SpyreTensorLayout

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

B, S, H = 1, 64, 1536
eps = 1e-6
dtype = torch.float16

# Build inputs on CPU
hidden = torch.randn(B, S, H, dtype=dtype)
weight = torch.randn(H, dtype=dtype)


def rms_norm(hidden, weight):
    x = hidden.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    return weight * normed.to(dtype)


# Compute the reference on CPU
cpu_result = rms_norm(hidden, weight)

# Build an explicit on-device layout that places the sequence dimension as the
# device stick, then move the input to the device under that layout.
hidden_layout = SpyreTensorLayout(
    hidden.size(), hidden.stride(), hidden.dtype, [0, 2, 1]
)
hidden_device = hidden.to(device_layout=hidden_layout)
weight_device = weight.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(rms_norm)
compiled_result = compiled(hidden_device, weight_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
torch.testing.assert_close(compiled_result, cpu_result, atol=0.1, rtol=0.1)
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
