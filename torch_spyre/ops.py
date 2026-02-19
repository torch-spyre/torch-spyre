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

import torch
import torch_spyre.fallbacks  # noqa: F401
from typing import Optional, Union


def maybe_wrap_dim(dim, ndims):
    if dim < 0:
        return dim + ndims
    return dim


@torch.library.register_kernel("aten::mm", ["spyre"])
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)


@torch.library.register_kernel("aten::mm.out", ["spyre"])
def spyre__mm_out(
    self: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2, out=out)


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])
def spyre__fill_scalar(
    self: torch.Tensor, other: Union[int, float, bool, complex]
) -> torch.Tensor:
    tmp = torch.ones(self.size(), dtype=self.dtype) * other
    self.copy_(tmp)
    return self


@torch.library.register_kernel("aten::transpose.int", ["spyre"])
def spyre__transpose_int(self: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    ndims = self.dim()
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)

    # Transpose of a tensor is a view operation.
    if dim0 == dim1:
        return torch.ops.aten.alias(self)

    sizes = list(self.shape)
    sizes[dim0], sizes[dim1] = sizes[dim1], sizes[dim0]
    strides = list(self.stride())
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
    prev_stl = self.device_tensor_layout()
    dim_map = prev_stl.dim_map
    for idx, dim in enumerate(dim_map):
        if dim == dim0:
            dim_map[idx] = dim1
        elif dim == dim1:
            dim_map[idx] = dim0
    new_stl = torch_spyre._C.SpyreTensorLayout(
        prev_stl.device_size, dim_map, prev_stl.device_dtype
    )

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


# derived from https://github.com/pytorch/pytorch/blob/f91f262275e12bd6249a2bcd2c3c06e0c78e20ee/aten/src/ATen/native/TensorShape.cpp#L3897
# with changes specific for spyre
def infer_squeeze_geometry(
    tensor: torch.Tensor, dims: Optional[int | list[int]] = None
):
    sizes = []
    strides = []
    current_stl = tensor.device_tensor_layout()
    stick_dim = current_stl.host_stick_dim()
    dim_map = current_stl.dim_map

    for idx in range(tensor.dim()):
        dim_check = False

        # Handle the cases where dims is set
        if isinstance(dims, int):
            dim_check = idx != dims
        elif isinstance(dims, list):
            dim_check = idx not in dims

        # Keep any dim > 1 that fulfills the dim_check
        if dim_check or tensor.size(idx) != 1:
            sizes.append(tensor.size(idx))
            strides.append(tensor.stride(idx))
        elif idx == stick_dim:
            # We cannot squeeze the stick dimension!
            raise ValueError("The stick dimension cannot be squeezed")
        else:
            # For the squeezed dimensions, correct the dim_map by
            # lowering the dimensions after the squeezed one
            for dim_idx in range(len(dim_map)):
                if dim_map[dim_idx] >= idx:
                    dim_map[dim_idx] -= 1

    new_stl = torch_spyre._C.SpyreTensorLayout(
        current_stl.device_size, dim_map, current_stl.device_dtype
    )

    return sizes, strides, new_stl


@torch.library.register_kernel("aten::squeeze", ["spyre"])
def spyre__squeeze(self: torch.Tensor) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::squeeze.dim", ["spyre"])
def spyre__squeeze_dim(self: torch.Tensor, dim: int) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self, dim)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::squeeze.dims", ["spyre"])
def spyre__squeeze_dims(self: torch.Tensor, dim: list[int]) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self, dim)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


# derived from https://github.com/pytorch/pytorch/blob/f91f262275e12bd6249a2bcd2c3c06e0c78e20ee/aten/src/ATen/native/TensorShape.cpp#L3943
# with changes specific to Spyre
def infer_unsqueeze_geometry(tensor: torch.Tensor, dim: int):
    sizes = list(tensor.size())
    strides = list(tensor.stride())

    new_stride = 1
    if dim < tensor.dim():
        new_stride = sizes[dim] * strides[dim]

    sizes.insert(dim, 1)
    strides.insert(dim, new_stride)

    current_stl = tensor.device_tensor_layout()
    dim_map = current_stl.dim_map

    for dim_idx in range(len(dim_map)):
        if dim_map[dim_idx] >= dim:
            dim_map[dim_idx] += 1

    new_stl = torch_spyre._C.SpyreTensorLayout(
        current_stl.device_size, dim_map, current_stl.device_dtype
    )

    return sizes, strides, new_stl


@torch.library.register_kernel("aten::unsqueeze", ["spyre"])
def spyre__unsqueeze(self: torch.Tensor, dim: int) -> torch.Tensor:
    sizes, strides, new_stl = infer_unsqueeze_geometry(self, dim)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


# INSERT_CODEGEN_HERE
