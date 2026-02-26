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
    """
    A class representing a Tensor argument to a KernelSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        dtype: The PyTorch (host) dtype of the tensor elements.
        host_size: The PyTorch (host) size of the Tensor.
        allocation: If present, the offset in scratchpad memory assigned to the Tensor.
        device_layout: The SpyreTensorLayout describe the on device shape of the Tensor.
    """

    is_input: bool
    arg_index: int
    dtype: torch.dtype
    host_size: torch.Size
    allocation: Any
    device_layout: SpyreTensorLayout


@dataclasses.dataclass
class ConstantArg:
    """
    A class representing a Constant argument to a KernelSpec

    Attributes:
        value: The value of the constant.
        dtype: The PyTorch (host) dtype of the constant.
    """

    value: Union[bool, float, int]
    dtype: torch.dtype


@dataclasses.dataclass
class KernelSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        dimensions: The iteration space of the operation.
        args: The input and output arguments to the operation.
        scales: A mapping between the op's dimensions and the (host) dimensions of each Tensor.
                Scales is a list[list[int]] such that len(scales) == len(args) and for each i, len(scales[i]) == len(dimensions).
                The integer values are interpreted as follows:
                    -1 indicates the the op dimension is a broadcast or reduction dimension for the Tensor
                    a non-negative value is the (host) dimension of the Tensor that corresponds to the op dimension
        op_info: A dictionary of auxiliary information whose content is operation-specific.
    """

    op: str
    is_reduction: bool
    dimensions: list[int]
    args: Sequence[TensorArg | ConstantArg]
    scales: list[list[int]]
    op_info: dict[str, Any]


@dataclasses.dataclass
class UnimplementedOp:
    op: str
