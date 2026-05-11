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
from .fallbacks import _get_op_overloads
import torch_spyre._C as _C
import warnings
import functools
import inspect
import operator


aten = torch.ops.aten


# Decorator to keep track of compiled variant
def compile_once(op, **compile_kwargs):
    def decorator(fn):
        compiled = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled
            nonlocal op
            if compiled is None:
                if isinstance(op, str):
                    op = operator.attrgetter(op)(torch.ops)
                compiled = torch.compile(op, **compile_kwargs)
            return fn(*args, compiled=compiled, **kwargs)

        # We remove the `compiled` arg from the signature to have
        # a clean signature.
        old_signature = inspect.signature(fn)
        params = dict(old_signature.parameters)
        params.pop("compiled", None)
        new_signature = old_signature.replace(parameters=params.values())
        wrapper.__signature__ = new_signature

        return wrapper

    return decorator


def maybe_wrap_dim(dim: int, ndims: int) -> int:
    if dim < 0:
        return dim + ndims
    return dim


def dispatch_to_torch_compile(*args, compiled=None, **kwargs):
    return compiled(*args, **kwargs)


def register_torch_compile_kernel(ops):
    for op in _get_op_overloads(ops):
        if "Tensor" not in str(op._schema):
            # there are some ops that do not take in Tensors
            # like aten.sum.int
            continue
        if "dtype" in op.name():
            # ops that change dtype are not supported yet
            continue
        compiled_kernel = compile_once(op, dynamic=False)(dispatch_to_torch_compile)
        torch.library.register_kernel(op.name(), ["spyre"])(compiled_kernel)


register_torch_compile_kernel(
    [
        aten.mm,
        aten.silu.out,
        aten.mish.out,
        aten.abs,
        aten.add,
        aten.bitwise_not,
        aten.logical_not,
        aten.bmm,
        aten.cat,
        aten.div,
        aten.exp,
        aten.floor,
        aten.log,
        aten.mean,
        aten.mul,
        aten.reciprocal,
        aten.neg,
        aten.relu,
        aten.rsqrt,
        aten.sigmoid,
        aten._softmax,
        aten.stack,
        aten.sum,
        aten.sqrt,
        aten.tanh,
        aten.sub,
        aten.addmm,
        aten.eq,
        aten.ge,
        aten.gt,
        aten.lt,
        aten.maximum,
        aten.pow,
        aten.linalg_vector_norm,
    ]
)


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
    if self.numel() == 0:
        return dst

    # Check if views of same data
    if (
        self.data_ptr() == dst.data_ptr()
        and self.storage_offset() == dst.storage_offset()
        and self.stride() == dst.stride()
        and self.size() == dst.size()
        and self.dtype == dst.dtype
        and self.is_conj() == dst.is_conj()
        and self.is_neg() == dst.is_neg()
    ):
        return dst

    if (self.device.type == "cpu" and dst.device.type == "spyre") or (
        self.device.type == "spyre" and dst.device.type == "cpu"
    ):
        _C.copy_tensor(self, dst, non_blocking)
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
