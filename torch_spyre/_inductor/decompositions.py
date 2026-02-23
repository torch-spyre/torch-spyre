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
        class OPWrapper:
            def __init__(self, op, spyre_fn):
                self.op = op
                self.spyre_fn = spyre_fn
                self._spyre_enabled = True

            @property
            def spyre_enabled(self):
                return self._spyre_enabled

            @spyre_enabled.setter
            def spyre_enabled(self, value):
                self._spyre_enabled = bool(value)

            def __call__(self, *args, **kwargs):
                if not self.spyre_enabled:
                    # This codepath should with high probability never be hit!
                    # If it is, it means that the Wrapper for the spyre decomposition
                    # is dispatched for in a non-spyre scenario.
                    # If this case is intended, try to dispatch the next higher key except PrivateUse1
                    # Note: We should probably either raise an Exception or at least a warning?
                    with torch._C._ExcludeDispatchKeyGuard(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
                    ):
                        return self.op(*args, **kwargs)

                return self.spyre_fn(*args, **kwargs)

        def register(op):
            spyre_decomposition_via_dispatchkey[op] = OPWrapper(op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, ops)
        return fn

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
    from torch.library import Library, fallthrough_kernel

    autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    lib = Library("aten", "IMPL", "PrivateUse1")

    for op, wrapper_cls in spyre_decomposition_via_dispatchkey.items():
        # Ensure that the spyre_fn is enabled in the wrapper
        wrapper_cls.spyre_enabled = True

        # Register a fallthrough kernel for the Autograd
        autograd_lib.impl(op._name, fallthrough_kernel)

        # Register the custom spyre kernel for the DispatchKey PrivateUse1
        lib.impl(op._name, wrapper_cls)

    try:
        yield
    finally:
        # Clean up: restore or remove the registered implementations
        for op, wrapper_cls in spyre_decomposition_via_dispatchkey.items():
            # Ensure that the spyre_fn is enabled in the wrapper
            wrapper_cls.spyre_enabled = False
        pass


@register_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


@register_decomposition([torch.ops.spyre.layer_norm])
def layernorm_decomp(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_decomposition([torch.ops.aten.full])
def full_decomp(
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


"""
Hook torch.nn.functional.layer_norm to select spyre optimized version where applicable
"""


# Register the spyre_layer_norm as Dispatcher backend for the spyre device
@register_spyre_decomposition_via_dispatchkey(
    [torch.ops.aten.native_layer_norm.default, torch.ops.aten.native_batch_norm.default]
)
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if input.device.type == "spyre" and len(normalized_shape) == 1:
        return torch.ops.spyre.layer_norm(input, normalized_shape, weight, bias, eps)
    else:
        # This should not happen, as this kernel should only dispatch for the spyre device
        raise Exception("This should not happen!")
        # return orig_layer_norm(input, normalized_shape, weight, bias, eps)


orig_gelu = torch.nn.functional.gelu


def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    if input.device.type == "spyre":
        return torch.ops.spyre.gelu(input, approximate)
    else:
        return orig_gelu(input, approximate=approximate)


torch.nn.functional.gelu = spyre_gelu


orig_softplus = torch.nn.functional.softplus


def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    if input.device.type == "spyre":
        return torch.ops.spyre.softplus(input, beta, threshold)
    else:
        return orig_softplus(input, beta, threshold)


torch.nn.functional.softplus = spyre_softplus


@register_decomposition([torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Tensor_out])
def gt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement greaterthan in the backend compiler
    out_ge = torch.ge(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_ge, out_ne, out=out).to(dtype=torch.bool)


@register_decomposition([torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Tensor_out])
def lt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement lessthan in the backend compiler
    out_le = torch.le(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_le, out_ne, out=out).to(dtype=torch.bool)
