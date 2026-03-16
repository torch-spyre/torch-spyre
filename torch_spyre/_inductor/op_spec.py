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
from typing import Any, Sequence
import torch
from torch_spyre._C import SpyreTensorLayout


@dataclasses.dataclass
class TensorArg:
    """
    A class representing a Tensor argument to an OpSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        dtype: The PyTorch (host) dtype of the tensor elements.
        it_dim_map: A mapping between the op's iteration_space and the PyTorch (host) dimensions of the Tensor.
            it_dim_map[d] is an integer that is interpreted as follows:
                -1 indicates the the d-th dimension of ks.iteration_space is a broadcast or reduction dimension for this Tensor.
                A non-negative value is the PyTorch (host) dimension of the Tensor that corresponds to the d-th dimension of ks.iteration_space.
        allocation: If present, the offset in scratchpad memory assigned to the Tensor.
        device_layout: The SpyreTensorLayout describe the device shape of the Tensor.
    """

    is_input: bool
    arg_index: int
    dtype: torch.dtype
    it_dim_map: list[int]
    allocation: Any
    device_layout: SpyreTensorLayout


@dataclasses.dataclass
class OpSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        iteration_space: The iteration space of the operation.
        args: The input and output arguments to the operation.
        op_info: A dictionary of auxiliary information whose content is operation-specific.
    """

    op: str
    is_reduction: bool
    iteration_space: list[int]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]


@dataclasses.dataclass
class UnimplementedOp:
    op: str
