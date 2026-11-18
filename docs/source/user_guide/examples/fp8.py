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
"""FP8 channel-wise quantization roundtrip on Spyre.

This script converts an FP16 tensor to FP8 E4M3 with torch.ops.spyre.qfp8ch,
then back to FP16 with torch.ops.spyre.dequantize_fp8_with_scale using an
identity scale, and compares the compiled Spyre result against a CPU reference
that clamps, casts to FP8, and casts back. It requires a Spyre device to run.
"""

import torch

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX, FP8_E4M3FN_MIN

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

# Build inputs on CPU
x = torch.randn((1, 2, 8), dtype=torch.float16)
scale = torch.ones((1, 2, 1), dtype=torch.float16)


def quantize_dequantize(x, scale):
    x_fp8 = torch.ops.spyre.qfp8ch(x)
    return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)


def cpu_reference(x, scale):
    clamped = (x / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX)
    return clamped.to(torch.float8_e4m3fn).to(torch.float16) * scale


# Compute the reference on CPU
cpu_result = cpu_reference(x, scale)

# Send inputs to device
x_device = x.to(DEVICE)
scale_device = scale.to(DEVICE)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(quantize_dequantize)
compiled_result = compiled(x_device, scale_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
