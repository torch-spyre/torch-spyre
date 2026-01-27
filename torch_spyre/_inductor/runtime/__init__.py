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

import dataclasses
from typing import Any, Sequence, Union
import torch
from torch_spyre._C import SpyreTensorLayout


@dataclasses.dataclass
class TensorArg:
    is_input: bool
    arg_index: int
    dtype: torch.dtype
    host_size: torch.Size
    allocation: Any
    device_layout: SpyreTensorLayout


@dataclasses.dataclass
class ConstantArg:
    value: Union[bool, float, int]
    dtype: torch.dtype


@dataclasses.dataclass
class KernelSpec:
    op: str
    is_reduction: bool
    dimensions: list[int]
    args: Sequence[TensorArg | ConstantArg]
    scales: list[list[int]]
    op_info: dict[str, Any]


@dataclasses.dataclass
class UnimplementedOp:
    op: str
