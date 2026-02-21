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
from torch_spyre._C import SpyreTensorLayout, as_strided_with_layout
import torch_spyre.fallbacks  # noqa: F401


def maybe_wrap_dim(dim: int, ndims: int) -> int:
    if dim < 0:
        return dim + ndims
    return dim


@torch.library.register_kernel("aten::mm", ["spyre"])  # type:ignore
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)


@torch.library.register_kernel("aten::mm.out", ["spyre"])  # type:ignore
def spyre__mm_out(
    self: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2, out=out)


@torch.library.register_kernel("aten::addmm", ["spyre"])  # type:ignore
def spyre__addmm_default(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta: int | float | bool | complex = 1,
    alpha: int | float | bool | complex = 1,
) -> torch.Tensor:
    # TODO: Add support for beta when constants work
    # TODO: Use inductor decomp when available
    mm_result = torch.ops.aten.mm(mat1, mat2)
    return torch.ops.aten.add.Tensor(mm_result, self, alpha=alpha)


@torch.library.register_kernel("aten::addmm.out", ["spyre"])  # type:ignore
def spyre__addmm_out(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta: int | float | bool | complex = 1,
    alpha: int | float | bool | complex = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # TODO: Add support for beta when constants work
    # TODO: Use inductor decomp when available
    mm_result = torch.ops.aten.mm(mat1, mat2)
    return torch.ops.aten.add.out(mm_result, self, alpha=alpha, out=out)


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])  # type:ignore
def spyre__fill_scalar(
    self: torch.Tensor, other: int | float | bool | complex
) -> torch.Tensor:
    tmp = torch.ones(self.size(), dtype=self.dtype) * other
    self.copy_(tmp)
    return self


@torch.library.register_kernel("aten::permute", ["spyre"])  # type:ignore
def spyre__permute(self: torch.Tensor, dims: list[int]) -> torch.Tensor:
    ndims = self.dim()
    dims = [maybe_wrap_dim(d, ndims) for d in dims]

    sizes = list(self.shape)
    strides = list(self.stride())
    new_sizes = [sizes[d] for d in dims]
    new_strides = [strides[d] for d in dims]

    prev_stl: SpyreTensorLayout = self.device_tensor_layout()  # type:ignore
    assert isinstance(prev_stl, SpyreTensorLayout)
    inv_perm = [0] * ndims
    for new_pos, old_pos in enumerate(dims):
        inv_perm[old_pos] = new_pos

    new_dim_map = [inv_perm[dim] for dim in prev_stl.dim_map]

    new_stl = SpyreTensorLayout(
        prev_stl.device_size, new_dim_map, prev_stl.device_dtype
    )

    result = as_strided_with_layout(
        self, tuple(new_sizes), tuple(new_strides), self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::transpose.int", ["spyre"])  # type:ignore
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
    prev_stl: SpyreTensorLayout = self.device_tensor_layout()  # type:ignore
    assert isinstance(prev_stl, SpyreTensorLayout)
    dim_map = prev_stl.dim_map
    for idx, dim in enumerate(dim_map):
        if dim == dim0:
            dim_map[idx] = dim1
        elif dim == dim1:
            dim_map[idx] = dim0
    new_stl = SpyreTensorLayout(prev_stl.device_size, dim_map, prev_stl.device_dtype)

    result = as_strided_with_layout(
        self, tuple(sizes), tuple(strides), self.storage_offset(), new_stl
    )
    return result


# derived from https://github.com/pytorch/pytorch/blob/f91f262275e12bd6249a2bcd2c3c06e0c78e20ee/aten/src/ATen/native/TensorShape.cpp#L3897
# with changes specific for spyre
def infer_squeeze_geometry(
    tensor: torch.Tensor, dims: int | list[int] | None = None
) -> tuple[tuple[int, ...], tuple[int, ...], SpyreTensorLayout]:
    sizes: list[int] = []
    strides: list[int] = []
    current_stl: SpyreTensorLayout = tensor.device_tensor_layout()  # type:ignore
    assert isinstance(current_stl, SpyreTensorLayout)
    stick_dim = current_stl.host_stick_dim()
    if stick_dim is None:
        raise ValueError("Squeezing of sparse tensors not implemented")

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

    new_stl = torch_spyre._C.compute_view_layout(tensor.size(), sizes, current_stl)

    return tuple(sizes), tuple(strides), new_stl


@torch.library.register_kernel("aten::squeeze", ["spyre"])  # type:ignore
def spyre__squeeze(self: torch.Tensor) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self)

    result = as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::squeeze.dim", ["spyre"])  # type:ignore
def spyre__squeeze_dim(self: torch.Tensor, dim: int) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self, dim)

    result = as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::squeeze.dims", ["spyre"])  # type:ignore
def spyre__squeeze_dims(self: torch.Tensor, dim: list[int]) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self, dim)

    result = as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


# derived from https://github.com/pytorch/pytorch/blob/f91f262275e12bd6249a2bcd2c3c06e0c78e20ee/aten/src/ATen/native/TensorShape.cpp#L3943
# with changes specific to Spyre
def infer_unsqueeze_geometry(
    tensor: torch.Tensor, dim: int
) -> tuple[tuple[int, ...], tuple[int, ...], SpyreTensorLayout]:
    sizes = list(tensor.size())
    strides = list(tensor.stride())

    new_stride = 1
    if dim < tensor.dim():
        new_stride = sizes[dim] * strides[dim]

    sizes.insert(dim, 1)
    strides.insert(dim, new_stride)

    current_stl = tensor.device_tensor_layout()
    stick_dim = current_stl.host_stick_dim()
    if stick_dim is None:
        raise ValueError("Unsqueezing of sparse tensors not implemented")

    new_stl = torch_spyre._C.compute_view_layout(tensor.size(), sizes, current_stl)

    return tuple(sizes), tuple(strides), new_stl


@torch.library.register_kernel("aten::unsqueeze", ["spyre"])  # type:ignore
def spyre__unsqueeze(self: torch.Tensor, dim: int) -> torch.Tensor:
    sizes, strides, new_stl = infer_unsqueeze_geometry(self, dim)

    result = as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::zero_", ["spyre"])  # type:ignore
def spyre__zero_(self: torch.Tensor) -> torch.Tensor:
    """Zero out the tensor in-place."""
    # Create zeros on CPU
    tmp = torch.zeros(self.size(), dtype=self.dtype, device="cpu")
    # Copy to device
    self.copy_(tmp)
    # TODO: Can we zero out tensors in-place without copy
    return self


@torch.library.register_kernel("aten::silu.out", ["spyre"])
def spyre__silu_out(self: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Out variant
    compiled_silu = torch.compile(torch.ops.aten.silu.out, dynamic=False)
    return compiled_silu(self, out=out)


@torch.library.register_kernel("aten::mish.out", ["spyre"])
def spyre__mish_out(self: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Out variant
    compiled_mish = torch.compile(torch.ops.aten.mish.out, dynamic=False)
    return compiled_mish(self, out=out)


# INSERT_CODEGEN_HERE
