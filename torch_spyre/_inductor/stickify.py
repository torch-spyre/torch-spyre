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

import torch
from sympy import Expr
from torch._inductor.ir import FixedLayout
from torch.fx.experimental.symbolic_shapes import (
    guard_size_oblivious,
    is_nested_int,
)

from torch_spyre._C import SpyreTensorLayout
from . import Unsupported


def stl_host_dim_order(self: SpyreTensorLayout) -> list[int]:
    return self.dim_map[1:]


def stl_stick_dim(self: SpyreTensorLayout) -> int:
    return self.dim_map[-1]


def stl_is_stick_reduction(self: SpyreTensorLayout, axis: list[int]) -> bool:
    stick_dim = self.stick_dim()
    if stick_dim in axis:
        if len(axis) > 1:
            Unsupported(f"reduction on both stick and non-stick dimensions {axis}")
        return True
    else:
        return False


def stl_spyre_fixed_layout(
    self: SpyreTensorLayout, device: torch.device, size: torch.Size, dtype: torch.dtype
):
    cur_stride = (
        1
        if self.format == SpyreTensorLayout.StickFormat.Dense
        else 128 // dtype.itemsize  # TODO - get from self.device_strides?
    )
    stride: list[int | torch.SymInt] = [-1] * len(size)
    for d in reversed(self.host_dim_order()):
        stride[d] = cur_stride
        cur_stride = cur_stride * size[d]
    return SpyreFixedLayout(device, dtype, list(size), stride, self)


setattr(SpyreTensorLayout, "host_dim_order", stl_host_dim_order)
setattr(SpyreTensorLayout, "stick_dim", stl_stick_dim)
setattr(SpyreTensorLayout, "is_stick_reduction", stl_is_stick_reduction)
setattr(SpyreTensorLayout, "spyre_fixed_layout", stl_spyre_fixed_layout)


class SpyreFixedLayout(FixedLayout):
    device_layout: SpyreTensorLayout

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: list[Expr],
        device_layout: SpyreTensorLayout,
    ) -> None:
        super().__init__(device, dtype, size, stride)
        self.device_layout = device_layout

    def __str__(self) -> str:
        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}, device_layout={self.device_layout})"
        )

    def get_allocation_size(self) -> list[Expr]:
        # TODO: Eventually this will include padding, etc.
        return self.size

    __repr__ = __str__


def tensor_get_spyre_layout(self: torch.Tensor) -> SpyreTensorLayout:
    if not hasattr(self, "spyre_layout"):
        print(f"Warning: {self} lacks spyre_layout; assuming generic stick layout")
        self.spyre_layout = SpyreTensorLayout(self.size(), self.dtype)
    return self.spyre_layout


def spyre_matmul_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    x_dci: SpyreTensorLayout = x.get_spyre_layout()
    y_dci: SpyreTensorLayout = y.get_spyre_layout()
    if (
        x_dci.format != SpyreTensorLayout.StickFormat.Dense
        or y_dci.format != SpyreTensorLayout.StickFormat.Dense
    ):
        raise Unsupported(f"matmul on non-dense tensors {x_dci} {y_dci}")
    if x_dci.dim_order != y_dci.dim_order:
        raise Unsupported(f"matmul stick dimensions mismatch {x_dci} {y_dci}")
    res_size = [x.size()[0], y.size()[1]]
    res_dci = SpyreTensorLayout(res_size, x_dci.dtype)  # TODO: forcing generic stick
    return res_size, res_dci


def spyre_reduction_result_shape(
    x: torch.Tensor, axis: Union[int, list[int]], keepdims: bool = False
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    # Normalize axis
    x_size = x.size()
    if isinstance(axis, int):
        axis = [axis]
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(x_size) if len(x_size) else 1

    # Compute result shape + DCI
    x_dci: SpyreTensorLayout = x.get_spyre_layout()
    is_stick_reduction = x_dci.is_stick_reduction(axis)
    res_size = list(x_size)
    res_order = x_dci.host_dim_order()
    for d in axis:
        if keepdims:
            res_size[d] = 1
        else:
            res_size[d] = -1
            res_order[d] = -1
            res_order = [rd if rd < d else rd - 1 for rd in res_order]
    res_size = [rs for rs in res_size if rs >= 0]
    res_order = [rd for rd in res_order if rd >= 0]
    res_format = (
        SpyreTensorLayout.StickFormat.Sparse
        if is_stick_reduction
        else SpyreTensorLayout.StickFormat.Dense
    )
    result_dci = SpyreTensorLayout(res_size, x.dtype, res_order, format=res_format)
    return res_size, result_dci


def spyre_pointwise_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    """
    Compute the shape of the result of a pointwise binary operation.
    The code is based on torch.broadcast_shapes with Spyre enhancements.
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

    x_dci = x.get_spyre_layout()
    y_dci = y.get_spyre_layout()
    if x_dci.format == y_dci.format:
        res_format = x_dci.format
    elif (
        x_dci.format == SpyreTensorLayout.StickFormat.Dense
        and y_broadcasted[x_dci.stick_dim()]
    ):
        res_format = SpyreTensorLayout.StickFormat.Dense
    elif (
        y_dci.format == SpyreTensorLayout.StickFormat.Dense
        and x_broadcasted[y_dci.stick_dim]
    ):
        res_format = SpyreTensorLayout.StickFormat.Dense
    else:
        raise Unsupported(
            f"binop with incompatible DCIs: {x_dci} {y_dci} {x_broadcasted} {y_broadcasted}"
        )

    # TODO: Forcing generic stick dimension order
    dim_order = list(range(len(res_size)))
    return res_size, SpyreTensorLayout(res_size, x.dtype, dim_order, format=res_format)
