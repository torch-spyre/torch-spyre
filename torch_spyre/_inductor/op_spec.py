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


from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from sympy import Symbol, Expr
from torch_spyre._C import DataFormats
import torch


@dataclasses.dataclass
class TensorArg:
    """
    A class representing a Tensor argument to an OpSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        device_dtype: The device dtype of the tensor elements.
        device_size: The device size (as per SpyreTensorLayout) of the Tensor
        device_coordinates: The sympy Exprs that describe how elements in the Tensor are accessed.
                Free variables in device_coordinates refer to entries in the OpSpec's iteration_space.
        allocation: If present, the offset in scratchpad memory assigned to the Tensor.
    """

    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: Any
    stride_map: list[int] | None = None
    per_tile_fixed: bool = False


@dataclasses.dataclass
class OpSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        iteration_space: The iteration space of the operation. The values are tuples of (range, work_division).
        args: The input and output arguments to the operation.
        op_info: A dictionary of auxiliary information whose content is operation-specific.
        tiled_symbols: Iteration-space symbols divided by the enclosing loop's count.
            Empty for ops that are not inside a LoopSpec.  The runtime computes the
            per-iteration tensor base offset for symbol ``s`` as
            ``loop_var * iteration_space[s].range``.
    """

    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]
    tiled_symbols: list[Symbol] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class UnimplementedOp:
    op: str


@dataclasses.dataclass
class LoopSpec:
    """A counted loop whose body is a sequence of ops, possibly nested.

    Attributes:
        count: Trip count of the loop. May be a symbolic shape expression.
        body: The operations to execute each iteration. Each element may be
            an OpSpec, UnimplementedOp, or a nested LoopSpec.
        tiled_symbols: The iteration-space symbols divided by ``count`` at
            *this* loop level.  Used by the unroller to advance HBM base
            addresses by exactly the right stride for each nesting level.
            Empty for LoopSpecs that do not carry per-level tiling info
            (legacy path; falls back to OpSpec.tiled_symbols).
    """

    count: Expr
    # list[OpSpec | UnimplementedOp | LoopSpec], typed as Any to accommodate
    # the two distinct UnimplementedOp types (op_spec vs spyre_kernel).
    body: list[Any]
    tiled_symbols: list[Symbol] = dataclasses.field(default_factory=list)


def spyre_constant_tensor(const_val, device, dtype=torch.float16):
    return torch.tensor(const_val, dtype=dtype).to(device)


def find_unimplemented(specs: list) -> UnimplementedOp | None:
    """Return the first UnimplementedOp in specs (recursing into LoopSpec), or None."""
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            return entry
        if isinstance(entry, LoopSpec):
            found = find_unimplemented(entry.body)
            if found is not None:
                return found
    return None
