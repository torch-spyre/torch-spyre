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

# this import will start the runtime
import torch

# Integer condition (non-bool)
condition_int = torch.tensor(
    [[1, 0],
     [0, 2]],
    device="spyre"
)

# Explicitly convert to boolean
condition = condition_int != 0

# Inputs
a = torch.tensor(
    [[10, 10],
     [10, 10]],
    dtype=torch.float16,
    device="spyre"
)

b = torch.tensor(
    [[20, 20],
     [20, 20]],
    dtype=torch.float16,
    device="spyre"
)

# torch.where with boolean condition
out = torch.where(condition, a, b)

print("condition_int:")
print(condition_int)

print("condition (bool):")
print(condition)

print("output:")
print(out)

print("With CPU:")

condition_int_cpu = torch.tensor(
    [[1, 0],
     [0, 2]],
    device="cpu"
)

# Inputs
a_cpu = torch.tensor(
    [[10, 10],
     [10, 10]],
    dtype=torch.float16,
    device="cpu"
)

b_cpu = torch.tensor(
    [[20, 20],
     [20, 20]],
    dtype=torch.float16,
    device="cpu"
)

condition_cpu = condition_int_cpu != 0
# torch.where with boolean condition
out_cpu = torch.where(condition_cpu, a_cpu, b_cpu)

print("condition_int_cpu:")
print(condition_int_cpu)

print("condition (bool):")
print(condition_cpu)

print("output:")
print(out_cpu)