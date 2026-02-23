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

from contextlib import contextmanager

from typing import Optional, Sequence, Union, Callable, TypeVar
from typing_extensions import ParamSpec

import torch
from torch.utils import _pytree as pytree
from torch._inductor.decomposition import register_decomposition

from .constants import DEVICE_NAME

# Dict for Spyre-specific decompositions to be registered via DispatchKey
spyre_decomposition_via_dispatchkey: dict = {}

_T = TypeVar("_T")
_P = ParamSpec("_P")

def register_spyre_decomposition_via_dispatchkey(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for Spyre device.
    These will only be active when compiling for the Spyre device.
    """
    def decomposition_decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        orig_fn = fn

        def register(op):
            spyre_decomposition_via_dispatchkey[op] = fn

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, ops)
        return orig_fn
    
    return decomposition_decorator

@contextmanager
def enable_spyre_decomposition_via_dispatchkey():
    """
    Context manager to temporarily register custom spyre implementations.
    
    This allows you to register device-specific implementations that are only
    active within the context, and automatically cleaned up when exiting.
    
    Example:
        >>> with enable_spyre_decomposition_via_dispatchkey():
        ...     output = torch.nn.functional.layer_norm(input, [512])
        ...     # Custom spyre implementation is used here
        >>> # Custom implementation is cleaned up here
    """
    # Store the original kernels so we can restore them
    original_kernels = {}
    newly_registered = []
    
    try:
        for op, fn in spyre_decomposition_via_dispatchkey.items():
            # Check if already registered for PrivateUse1
            if torch._C.DispatchKey.PrivateUse1 in op.py_kernels:
                # Already registered, store it so we can restore later
                original_kernels[op] = op.py_kernels[torch._C.DispatchKey.PrivateUse1]
                # Skip re-registration to avoid "Trying to override" error
                continue
            
            # Register the custom implementation for PrivateUse1 (spyre)
            op.py_impl(torch._C.DispatchKey.PrivateUse1)(fn)
            newly_registered.append(op)
            
            # Clear dispatch cache to ensure new implementation is used
            if hasattr(op, '_dispatch_cache'):
                op._dispatch_cache.clear()
        
        yield
        
    finally:
        # Clean up: restore or remove the registered implementations
        for op in newly_registered:
            try:
                # Remove our custom kernel (only for newly registered ones)
                if torch._C.DispatchKey.PrivateUse1 in op.py_kernels:
                    del op.py_kernels[torch._C.DispatchKey.PrivateUse1]
                
                # Clear dispatch cache again
                if hasattr(op, '_dispatch_cache'):
                    op._dispatch_cache.clear()
            except:
                pass
        
        # Restore original kernels
        for op, original_kernel in original_kernels.items():
            try:
                op.py_kernels[torch._C.DispatchKey.PrivateUse1] = original_kernel
                if hasattr(op, '_dispatch_cache'):
                    op._dispatch_cache.clear()
            except:
                pass

# @register_decomposition([torch.ops.spyre.compact])
# def compact_decomp(x: torch.Tensor) -> torch.Tensor:
#     return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


# @register_decomposition([torch.ops.spyre.layer_norm])
# def layernorm_decomp(
#     input: torch.Tensor,
#     normalized_shape: list[int],
#     weight: Optional[torch.Tensor] = None,
#     bias: Optional[torch.Tensor] = None,
#     eps: float = 1e-5,
# ) -> torch.Tensor:
#     mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
#     norm_mean = torch.ops.spyre.layernormscale(mean, eps)
#     return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


# # TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# # a one-element Spyre tensor. This currently fails because Spyre does not handle
# # single-element tensors well.
# # Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
# #
# # To avoid constant folding, we introduce a custom op `spyre::full` that runs
# # torch.full on CPU and copies the result to Spyre. Remove this workaround once
# # Spyre supports one-element tensors.
# @register_decomposition([torch.ops.aten.full])
# def full_decomp(
#     size: list[Union[int, torch.SymInt]],
#     fill_value: torch.types.Number,
#     dtype: Optional[torch.dtype] = None,
#     layout: Optional[torch.layout] = None,
#     device: Optional[torch.device] = None,
#     pin_memory: Optional[bool] = None,
# ) -> torch.Tensor:
#     assert layout in (torch.strided, None), f"doesn't support layout={layout}"
#     assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
#     return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


"""
Hook torch.nn.functional.layer_norm to select spyre optimized version where applicable
"""
# Store original layer_norm function for fallback
orig_layer_norm = torch.nn.functional.layer_norm

# Register the native_layer_norm operator for spyre device
@register_spyre_decomposition_via_dispatchkey([torch.ops.aten.native_layer_norm.default])
def spyre_native_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom implementation of native_layer_norm for spyre device.
    Returns: (output, mean, rstd)
    """
    if input.device.type == "spyre" and len(normalized_shape) == 1:
        # Use spyre's optimized layer_norm
        output = torch.ops.spyre.layer_norm(input, normalized_shape, weight, bias, eps)
        # native_layer_norm returns (output, mean, rstd)
        # For now, compute mean and rstd on CPU as fallback
        input_cpu = input.cpu()
        mean = input_cpu.mean(dim=-1, keepdim=True)
        var = input_cpu.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        return output, mean.to(input.device), rstd.to(input.device)
    else:
        # Fallback to default implementation
        return torch.ops.aten.native_layer_norm.default(input, normalized_shape, weight, bias, eps)

# CRITICAL WORKAROUND: torch.layer_norm incorrectly dispatches to native_batch_norm for PrivateUse1
# This appears to be a bug in PyTorch's dispatch system for custom backends.
# Register native_batch_norm to redirect to layer_norm as a workaround.
@register_spyre_decomposition_via_dispatchkey([torch.ops.aten.native_batch_norm.default])
def spyre_native_batch_norm_redirect(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    WORKAROUND: torch.layer_norm incorrectly calls native_batch_norm for PrivateUse1.
    Redirect to layer_norm implementation.
    """
    # Infer normalized_shape from weight if available, otherwise use last dimension
    if weight is not None:
        normalized_shape = list(weight.shape)
    else:
        normalized_shape = [input.shape[-1]]
    
    # Call our layer_norm implementation
    return spyre_native_layer_norm(input, normalized_shape, weight, bias, eps)


# torch.nn.functional.layer_norm = spyre_layer_norm

# Register GELU for spyre device
# torch.nn.functional.gelu calls torch.ops.aten.gelu.default
@register_spyre_decomposition_via_dispatchkey([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    """
    Custom implementation of GELU for spyre device.
    """
    if input.device.type == "spyre":
        return torch.ops.spyre.gelu(input, approximate)
    else:
        # Fallback to default implementation
        return torch.ops.aten.gelu.default(input, approximate=approximate)


# Register softplus for spyre device
# torch.nn.functional.softplus calls torch.ops.aten.softplus
@register_spyre_decomposition_via_dispatchkey([
    torch.ops.aten.softplus.default,
    torch.ops.aten.softplus.out
])
def spyre_softplus(
    input: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Custom implementation of softplus for spyre device.
    Handles both default and out variants.
    """
    if input.device.type == "spyre":
        result = torch.ops.spyre.softplus(input, beta, threshold)
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        # Fallback to default implementation
        if out is not None:
            return torch.ops.aten.softplus.out(input, beta, threshold, out=out)
        return torch.ops.aten.softplus.default(input, beta, threshold)


# @register_decomposition([torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Tensor_out])
# def gt_decomp(
#     input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
# ) -> torch.Tensor:
#     # TODO: Implement greaterthan in the backend compiler
#     out_ge = torch.ge(input, other).to(dtype=torch.float16)
#     out_ne = torch.ne(input, other).to(dtype=torch.float16)
#     return torch.mul(out_ge, out_ne, out=out).to(dtype=torch.bool)


# @register_decomposition([torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Tensor_out])
# def lt_decomp(
#     input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
# ) -> torch.Tensor:
#     # TODO: Implement lessthan in the backend compiler
#     out_le = torch.le(input, other).to(dtype=torch.float16)
#     out_ne = torch.ne(input, other).to(dtype=torch.float16)
#     return torch.mul(out_le, out_ne, out=out).to(dtype=torch.bool)
