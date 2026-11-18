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
"""Registering a custom op with a Spyre lowering.

This script defines a custom clone_identity op with CPU and Meta
implementations, registers a Spyre lowering for it with
register_spyre_lowering, and compiles a function that calls it on a permuted
input. It compares the compiled Spyre result against a CPU reference. It
requires a Spyre device to run.
"""

import torch

from torch_spyre._inductor.lowering import (
    clone as spyre_clone_lowering,
    register_spyre_lowering,
)

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

# Define a custom op with CPU and Meta implementations.
lib = torch.library.Library("spyre_example", "FRAGMENT")
lib.define("clone_identity(Tensor x) -> Tensor")
op = torch.ops.spyre_example.clone_identity.default
lib.impl("clone_identity", lambda x: x.clone(), "CPU")
lib.impl("clone_identity", lambda x: torch.empty_like(x), "Meta")


# Register a Spyre lowering so the op compiles on the device.
@register_spyre_lowering(op, type_promotion_kind=None)
def _lower_clone_identity(x):
    return spyre_clone_lowering(x)


def fn(a):
    return torch.ops.spyre_example.clone_identity(a.permute(1, 0))


# Build the input on CPU
x = torch.randn(128, 192, dtype=torch.float16)

# Compute the reference on CPU
cpu_result = fn(x)

# Compute on the device in compiled mode and bring the result back to the host
compiled = torch.compile(fn, backend="inductor")
compiled_result = compiled(x.to(DEVICE)).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
torch.testing.assert_close(compiled_result, cpu_result, atol=1e-3, rtol=1e-3)
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
