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

from typing import Sequence, Tuple, Union
from enum import Enum
import dataclasses

import torch
from sympy import Expr
from torch._inductor.ir import FixedLayout
from torch.fx.experimental.symbolic_shapes import (
    guard_size_oblivious,
    is_nested_int,
)

from . import Unsupported


BYTES_IN_STICK = 128


def elems_per_stick(dtype: torch.dtype) -> int:
    return BYTES_IN_STICK // dtype.itemsize


class StickFormat(Enum):
    DENSE = 1
    SPARSE = 2
    SPARSE_MULTI = 3


@dataclasses.dataclass
class SpyreDCI:
    """
    FIXME: This will be part of csrc with a pybind to be placed in TensorImpl
    NOTE: dim_order follows the PyTorch conventions.
    dim_order[-1] is the dimenson with the smallest stride;
    dim_order[0] is the dimension with the largest stride.
    dim_order[-num_stick_dims:] are the stick dimensions
    """

    dim_order: list[int]
    num_stick_dims: int = 1
    format: StickFormat = StickFormat.DENSE

    def __post_init__(self):
        assert self.num_stick_dims == 1, "currently limited to one stick dimension"

    @staticmethod
    def generic_stick_dci(t: torch.Tensor):
        dim_order = list(range(len(t.size())))
        return SpyreDCI(dim_order)

    def get_stick_dims(self) -> list[int]:
        return self.dim_order[-self.num_stick_dims :]

    def is_stick_dim(self, dim: Union[int | list[int]]) -> bool:
        stick_dims = self.get_stick_dims()
        if isinstance(dim, int):
            return dim in stick_dims
        else:
            for d in dim:
                if d in stick_dims:
                    return True
            return False

    def is_stick_reduction(self, axis: list[int]) -> bool:
        stick = False
        non_stick = False
        stick_dims = self.get_stick_dims()
        for d in axis:
            if d in stick_dims:
                stick = True
            else:
                non_stick = True
        if stick and non_stick:
            raise Unsupported(
                f"reduction on both stick and non-stick dimensions {axis}"
            )
        return stick

    def spyre_strides(
        self, size: torch.Size, dtype: torch.dtype
    ) -> list[int | torch.SymInt]:
        cur_stride = 1 if self.format == StickFormat.DENSE else elems_per_stick(dtype)
        strides: list[int | torch.SymInt] = [-1] * len(size)
        for d in reversed(self.dim_order):
            strides[d] = cur_stride
            cur_stride = cur_stride * size[d]
        return strides

    def spyre_layout(
        self, device: torch.device, size: torch.Size, dtype: torch.dtype
    ) -> FixedLayout:
        stride = self.spyre_strides(size, dtype)
        return SpyreFixedLayout(device, dtype, list(size), stride, self)


class SpyreFixedLayout(FixedLayout):
    dci: SpyreDCI

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: list[Expr],
        dci: SpyreDCI,
    ) -> None:
        super().__init__(device, dtype, size, stride)
        self.dci = dci

    def __str__(self) -> str:
        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}, dci={self.dci})"
        )

    def get_allocation_size(self) -> list[Expr]:
        # TODO: Eventually this will include padding, etc.
        return self.size

    __repr__ = __str__


def tensor_get_dci(self: torch.Tensor) -> SpyreDCI:
    if not hasattr(self, "spyre_dci"):
        print(f"Warning: {self} lacks spyre_dci; assuming generic stick layout")
        self.spyre_dci = SpyreDCI.generic_stick_dci(self)
    return self.spyre_dci


def spyre_matmul_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreDCI]:
    x_dci: SpyreDCI = x.get_dci()
    y_dci: SpyreDCI = y.get_dci()
    if x_dci.format != StickFormat.DENSE or y_dci.format != StickFormat.DENSE:
        raise Unsupported(f"matmul on non-dense tensors {x_dci} {y_dci}")
    if x_dci.dim_order != y_dci.dim_order:
        raise Unsupported(f"matmul stick dimensions mismatch {x_dci} {y_dci}")
    res_dci = SpyreDCI(list(x_dci.dim_order))
    res_size = [x.size()[0], y.size()[1]]
    return res_size, res_dci


def spyre_reduction_result_shape(
    x: torch.Tensor, axis: Union[int, list[int]], keepdims: bool = False
) -> Tuple[Sequence[int], SpyreDCI]:
    # Normalize axis
    x_size = x.size()
    if isinstance(axis, int):
        axis = [axis]
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(x_size) if len(x_size) else 1

    # Compute result shape + DCI
    x_dci: SpyreDCI = x.get_dci()
    is_stick_reduction = x_dci.is_stick_reduction(axis)
    res_size = list(x_size)
    res_order = list(x_dci.dim_order)
    for d in axis:
        if keepdims:
            res_size[d] = 1
        else:
            res_size[d] = -1
            res_order[d] = -1
            res_order = [rd if rd < d else rd - 1 for rd in res_order]
    res_size = [rs for rs in res_size if rs >= 0]
    res_order = [rd for rd in res_order if rd >= 0]
    result_dci = SpyreDCI(
        res_order,
        format=StickFormat.SPARSE if is_stick_reduction else StickFormat.DENSE,
    )
    return res_size, result_dci


def spyre_pointwise_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreDCI]:
    """
    Compute the shape of the result of a pointwise binary operation.
    The code is based torch.broadcast_shapes with Spyre enhancements.
    """
    x_size = x.size()
    y_size = y.size()
    res_size = [1] * max(len(x_size), len(y_size))
    x_broadcasted = [False] * len(res_size)
    y_broadcasted = [False] * len(res_size)
    for i in range(-1, -1 - len(x_size), -1):
        res_size[i] = x_size[i]

    for i in range(-1, -1 - len(y_size), -1):
        # NB: handle nested ints specially to avoid invalid guarding on Ne(j0, 1).
        if is_nested_int(y_size[i]):
            # Broadcasting is allowed for (j0, 1) or (j0, j0);
            # not (j0, j1), (j0, 5), etc.
            if is_nested_int(res_size[i]) and guard_size_oblivious(
                y_size[i] == res_size[i]
            ):
                continue
        else:
            if guard_size_oblivious(y_size[i] == res_size[i]):
                continue
            if guard_size_oblivious(y_size[i] == 1) and not guard_size_oblivious(
                res_size[i] == 1
            ):
                y_broadcasted[i] = True
                continue

        if res_size[i] != 1:
            raise RuntimeError(
                "Shape mismatch: objects cannot be broadcast to a single shape"
            )
        res_size[i] = y_size[i]
        x_broadcasted[i] = True

    x_dci = x.get_dci()
    y_dci = y.get_dci()
    if x_dci.format == y_dci.format:
        res_format = x_dci.format
    elif x_dci.format == StickFormat.DENSE and y_broadcasted[x_dci.get_stick_dims()[0]]:
        res_format = StickFormat.DENSE
    elif y_dci.format == StickFormat.DENSE and x_broadcasted[y_dci.get_stick_dims()[0]]:
        res_format = StickFormat.DENSE
    else:
        raise Unsupported(
            f"binop with incompatible DCIs: {x_dci} {y_dci} {x_broadcasted} {y_broadcasted}"
        )

    dim_order = list(range(len(res_size)))
    return res_size, SpyreDCI(dim_order, format=res_format)
