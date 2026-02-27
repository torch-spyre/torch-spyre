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
import torch._decomp as decomp
from .errors import Unsupported

import threading

# A module-level lock + nesting counter to make the CM reentrant/thread-safe
_decompositions_lock = threading.RLock()
_decompositions_nesting = 0

# Dictionary for Spyre-specific decompositions
spyre_decompositions: dict = {}

# Exclude specific Inductor default decompositions on Spyre.
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
spyre_decompositions_to_exclude = [
    # The default decomposition for torch.new_ones (defined in pytorch/torch/refs/__init__.py)
    # uses torch.full, which is not yet supported in Spyre eager mode.
    # See: https://github.com/torch-spyre/torch-spyre/issues/128#issuecomment-3576168221
    torch.ops.aten.new_ones,
]

# A module-level lock + nesting counter to make the CM reentrant/thread-safe
_decompositions_via_dispatchkey_lock = threading.RLock()
_decompositions_via_dispatchkey_nesting = 0

# Dict for Spyre-specific decompositions to be registered via DispatchKey
spyre_decompositions_via_dispatchkey: dict = {}

_T = TypeVar("_T")
_P = ParamSpec("_P")
    
def register_spyre_decomposition(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for Spyre device.
    These will only be active when compiling for the Spyre device.
    """
    return decomp.register_decomposition(ops, spyre_decompositions)

# Context manager that enables spyre specific decompositions in addition to PyTorch in-tree decompositions
@contextmanager
def enable_spyre_decompositions():
    """
    CM that enables Spyre decompositions:
      - Temporarily adds relevant Spyre decompositions to global decompositions dictionary
      - Restore original decompositions on exit

    This CM is reentrant and safe under nested usage.
    """
    global _decompositions_nesting
    with _decompositions_lock:
        first_enter = (_decompositions_nesting == 0)  # fmt: skip
        _decompositions_nesting += 1

        if first_enter:
            from torch_spyre.fallbacks import fallback_ops
            from torch._inductor.decomposition import decompositions
            from torch._ops import OpOverload, OpOverloadPacket

            # Helper function to remove ops from decompositions
            def _fetch_and_remove_op(ops):
                _removed = {}
                for op in ops:
                    if isinstance(op, OpOverloadPacket):
                        for overload_name in op.overloads():
                            opo = getattr(op, overload_name)
                            op_ret = decompositions.pop(opo, None)
                            if op_ret is not None:
                                _removed[opo] = op_ret
                    elif isinstance(op, OpOverload):
                        op_ret = decompositions.pop(op, None)
                        if op_ret is not None:
                            _removed[op] = op_ret
                return _removed

            # 1. Add/override spyre-specific decompositions
            saved_intree_decompositions = {}
            for (
                spyre_decompositions_op,
                spyre_decompositions_impl,
            ) in spyre_decompositions.items():
                if spyre_decompositions_op in decompositions:
                    saved_intree_decompositions[spyre_decompositions_op] = (
                        decompositions[spyre_decompositions_op]
                    )
                decompositions[spyre_decompositions_op] = spyre_decompositions_impl

            # Attach to the function so we can restore on last exit
            enable_spyre_decompositions._saved_decompositions = (
                saved_intree_decompositions
            )

            # 2. Remove selected decompositions from Inductor's registry for spyre
            _removed_decompositions_to_exclude = _fetch_and_remove_op(
                spyre_decompositions_to_exclude
            )

            # Attach to the function so we can restore on last exit
            enable_spyre_decompositions._removed_decompositions_to_exclude = (
                _removed_decompositions_to_exclude
            )

            # 3. Remove selected decompositions for fallback ops defined in fallbacks.py
            _removed_decompositions_fallback_ops = _fetch_and_remove_op(fallback_ops)

            # Attach to the function so we can restore on last exit
            enable_spyre_decompositions._removed_decompositions_fallback_ops = (
                _removed_decompositions_fallback_ops
            )

        try:
            yield
        finally:
            _decompositions_nesting -= 1
            last_exit = (_decompositions_nesting == 0)  # fmt: skip
            if last_exit:
                # Inverse order compared to when entering the context manager

                # 1. Revert selected decompositions that have been marked for fallback ops
                removed_decompositions_fallback_ops = getattr(
                    enable_spyre_decompositions,
                    "_removed_decompositions_fallback_ops",
                    {},
                )
                [
                    torch._decomp._add_op_to_registry(decompositions, op, fn)
                    for op, fn in removed_decompositions_fallback_ops.items()
                ]

                # 2. Revert selected decompositions that have been removed from Inductor's registry for spyre
                removed_decompositions_to_exclude = getattr(
                    enable_spyre_decompositions,
                    "_removed_decompositions_to_exclude",
                    {},
                )
                [
                    torch._decomp._add_op_to_registry(decompositions, op, fn)
                    for op, fn in removed_decompositions_to_exclude.items()
                ]

                # 3. Reset the saved in-tree lowerings if needed
                saved_intree_decompositions = getattr(
                    enable_spyre_decompositions, "_saved_decompositions", {}
                )
                for (
                    spyre_decompositions_op,
                    spyre_decompositions_impl,
                ) in spyre_decompositions.items():
                    if spyre_decompositions_op in saved_intree_decompositions:
                        decompositions[spyre_decompositions_op] = (
                            saved_intree_decompositions[spyre_decompositions_op]
                        )
                    else:
                        decompositions.pop(spyre_decompositions_op, None)

                # Clean up
                enable_spyre_decompositions._saved_decompositions = {}
                enable_spyre_decompositions._removed_decompositions_to_exclude = {}
                enable_spyre_decompositions._removed_decompositions_fallback_ops = {}


def register_spyre_decompositions_via_dispatchkey(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for Spyre device via the PyTorch dispatcher
    This replaces the need for global patching of operations in order to enable them for 
    eager mode.
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
            spyre_decompositions_via_dispatchkey[op] = OPWrapper(op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, ops)
        return fn

    return decomposition_decorator
    
@contextmanager
def enable_spyre_decompositions_via_dispatchkey():
    """
    Context manager to temporarily register custom spyre implementations.

    This allows you to register device-specific implementations that are only
    active within the context, and automatically cleaned up when exiting.

    Example:
        >>> with enable_spyre_decompositions_via_dispatchkey():
        ...     output = torch.nn.functional.layer_norm(input, [512])
        ...     # Custom spyre implementation is used here
        >>> # Custom implementation is cleaned up here
    """
    from torch.library import Library, fallthrough_kernel
    
    global _decompositions_via_dispatchkey_nesting
    with _decompositions_via_dispatchkey_lock:
        first_enter = (_decompositions_via_dispatchkey_nesting == 0)  # fmt: skip
        _decompositions_via_dispatchkey_nesting += 1

        autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
        lib = Library("aten", "IMPL", "PrivateUse1")

        for op, wrapper_cls in spyre_decompositions_via_dispatchkey.items():
            # Ensure that the spyre_fn is enabled in the wrapper
            wrapper_cls.spyre_enabled = True

            # Add the new kernels just once
            if first_enter:
                # Register a fallthrough kernel for the Autograd
                autograd_lib.impl(op._name, fallthrough_kernel)

                # Register the custom spyre kernel for the DispatchKey PrivateUse1
                lib.impl(op._name, wrapper_cls)

    try:
        yield
    finally:
        _decompositions_via_dispatchkey_nesting -= 1
        last_exit = (_decompositions_via_dispatchkey_nesting == 0)  # fmt: skip
        
        # Keep the spyre implementations enabled until the last context manager exits
        if last_exit:
            # Clean up: restore or remove the registered implementations
            for op, wrapper_cls in spyre_decompositions_via_dispatchkey.items():
                # Ensure that the spyre_fn is enabled in the wrapper
                wrapper_cls.spyre_enabled = False


@register_spyre_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


@register_spyre_decomposition([torch.ops.spyre.layer_norm])
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


@register_spyre_decomposition([torch.ops.spyre.rms_norm])
def rmsnorm_decomp(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    # TODO: limitation with mean on dim=-1, transpose for now to avoid
    # https://github.com/torch-spyre/torch-spyre/issues/632
    input = input.transpose(-1, -2).contiguous()
    eps = torch.ops.spyre.full(input.shape, eps, dtype=torch.float16, device="spyre")
    rsqrt_inp = torch.rsqrt(torch.mean(input * input, dim=-2, keepdim=True)) + eps
    output = (input * rsqrt_inp).transpose(-1, -2).contiguous()
    if weight is not None:
        output = output * weight
    return output


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_spyre_decomposition([torch.ops.aten.full])
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
@register_spyre_decompositions_via_dispatchkey(
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

orig_rms_norm = torch.nn.functional.rms_norm


def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> torch.Tensor:
    if input.device.type == "spyre" and len(normalized_shape) == 1:
        return torch.ops.spyre.rms_norm(input, normalized_shape, weight, eps)
    elif input.device.type == "spyre" and len(normalized_shape) != 1:
        raise Unsupported("RMSNorm reducing more than 1 dimension")
    else:
        return orig_rms_norm(input, normalized_shape, weight, eps)


torch.nn.functional.rms_norm = spyre_rms_norm


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


@register_spyre_decomposition([torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Tensor_out])
def gt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement greaterthan in the backend compiler
    out_ge = torch.ge(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_ge, out_ne, out=out).to(dtype=torch.bool)


@register_spyre_decomposition([torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Tensor_out])
def lt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement lessthan in the backend compiler
    out_le = torch.le(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_le, out_ne, out=out).to(dtype=torch.bool)
