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
import torch_spyre.ops.fallbacks  # noqa: F401
import torch_spyre._C as _C
import warnings


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


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])  # type:ignore
def spyre__fill_scalar(
    self: torch.Tensor, other: int | float | bool | complex
) -> torch.Tensor:
    tmp = torch.ones(self.size(), dtype=self.dtype) * other
    self.copy_(tmp)
    return self


@torch.library.register_kernel("aten::normal_", ["spyre"])  # type:ignore
def spyre__normal_(self, mean=0.0, std=1.0, *, generator=None):
    # "normal_" generates a random tensor, thus copying
    # "self" back from SPYRE to CPU is not needed.
    # cpu_tmp = self.to("cpu")

    # Create a new tensor on cpu itself to avoid unnecessary data copy.
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)
    cpu_tmp.normal_(mean, std, generator=generator)
    self.copy_(cpu_tmp)
    return self


@torch.library.register_kernel("aten::zero_", ["spyre"])  # type:ignore
def spyre__zero_(self: torch.Tensor) -> torch.Tensor:
    """Zero out the tensor in-place."""
    # Create zeros on CPU
    tmp = torch.zeros(self.size(), dtype=self.dtype, device="cpu")
    # Copy to device
    self.copy_(tmp)
    # TODO: Can we zero out tensors in-place without copy
    return self


@torch.library.register_kernel("aten::silu.out", ["spyre"])  # type:ignore
def spyre__silu_out(self: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Out variant
    compiled_silu = torch.compile(torch.ops.aten.silu.out, dynamic=False)
    return compiled_silu(self, out=out)


@torch.library.register_kernel("aten::mish.out", ["spyre"])  # type:ignore
def spyre__mish_out(self: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Out variant
    compiled_mish = torch.compile(torch.ops.aten.mish.out, dynamic=False)
    return compiled_mish(self, out=out)


@torch.library.register_kernel("aten::uniform_", "spyre")  # type:ignore
def spyre__uniform_(self, from_=0.0, to=1.0, generator=None):
    # Create a new tensor on cpu
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)

    # Fill the CPU tensor with uniform random values
    cpu_tmp.uniform_(from_, to, generator=generator)

    # Copy the CPU tensor back to the spyre device
    self.copy_(cpu_tmp)

    return self


@torch.library.register_kernel("aten::_local_scalar_dense", "spyre")
def spyre__local_scalar_dense(self):
    return self.cpu().item()


@torch.library.register_kernel("aten::_copy_from", ["spyre"])
def spyre__copy_from(self, dst, non_blocking=False):
    # Check if views of same data
    if (
        self.data_ptr() == dst.data_ptr()
        and self.storage_offset() == dst.storage_offset()
        and self.strides().equals(dst.strides())
        and self.sizes().equals(dst.sizes())
        and self.scalar_type() == dst.scalar_type()
        and self.is_conj() == dst.is_conj()
        and self.is_neg() == dst.is_neg()
    ):
        return dst

    if self.numel() == 0:
        return dst

    if self.device.type == "cpu" and dst.device.type == "spyre":
        _C.copy_host_to_device(self, dst)
        return dst
    elif self.device.type == "spyre" and dst.device.type == "cpu":
        _C.copy_device_to_host(self, dst)
        return dst
    elif self.device.type == "spyre" and self.device == dst.device:
        torch.ops.spyre.copy_from_d2d(self, dst)
        return dst
    else:
        if non_blocking:
            warnings.warn(
                f"non_blocking is set to {non_blocking}", UserWarning, stacklevel=2
            )

        torch.ops.aten._copy_from.default(self, dst, non_blocking)
        return dst


# INSERT_CODEGEN_HERE
