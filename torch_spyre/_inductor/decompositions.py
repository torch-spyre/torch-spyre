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

from typing import Optional, Union, Sequence, Callable, TypeVar
from typing_extensions import ParamSpec
import torch
from torch.utils import _pytree as pytree
import torch._decomp as decomp

from .constants import DEVICE_NAME
from .errors import Unsupported
from . import customops  # noqa: F401

import threading

# A module-level lock to make the CM thread-safe
_decompositions_lock = threading.RLock()

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

# Dict for Spyre-specific decompositions to be registered via DispatchKey
spyre_decompositions_via_dispatchkey: dict = {}

# Module-level Library objects kept alive permanently so that the registered
# PrivateUse1 / AutogradPrivateUse1 kernels are never unregistered by garbage collector.
# (torch.library.Library uses weakref.finalize → m.reset() on GC, which would
# silently remove the kernels from the C++ dispatcher.)
_spyre_autograd_lib = None
_spyre_lib = None
_dispatchkey_kernels_registered = False

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
def enable_spyre_decompositions(
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    CM that enables Spyre decompositions:
      - Temporarily adds relevant Spyre decompositions to provided decomposition table `decomps`
      - Restore original decompositions table on exit

    This CM is reentrant and safe under nested usage.

    Args:
        decomps: Decomposition table to modify. Maps operator overloads to their
            decomposition implementations. Defaults to PyTorch Inductor's global
            decomposition registry (torch._inductor.decomposition.decompositions).
    """
    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    with _decompositions_lock:
        from torch_spyre.fallbacks import fallback_ops
        from torch._ops import OpOverload, OpOverloadPacket

        # Helper function to remove ops from decompositions
        def _fetch_and_remove_op(ops):
            _removed = {}
            for op in ops:
                if isinstance(op, OpOverloadPacket):
                    for overload_name in op.overloads():
                        opo = getattr(op, overload_name)
                        op_ret = decomps.pop(opo, None)
                        if op_ret is not None:
                            _removed[opo] = op_ret
                elif isinstance(op, OpOverload):
                    op_ret = decomps.pop(op, None)
                    if op_ret is not None:
                        _removed[op] = op_ret
            return _removed

        # 1. Add/override spyre-specific decompositions
        saved_intree_decompositions = {}
        for (
            spyre_decompositions_op,
            spyre_decompositions_impl,
        ) in spyre_decompositions.items():
            if spyre_decompositions_op in decomps:
                saved_intree_decompositions[spyre_decompositions_op] = decomps[
                    spyre_decompositions_op
                ]
            decomps[spyre_decompositions_op] = spyre_decompositions_impl

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._saved_decompositions = saved_intree_decompositions

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
            yield decomps
        finally:
            # Inverse order compared to when entering the context manager

            # 1. Revert selected decompositions that have been marked for fallback ops
            removed_decompositions_fallback_ops = getattr(
                enable_spyre_decompositions,
                "_removed_decompositions_fallback_ops",
                {},
            )
            [
                torch._decomp._add_op_to_registry(decomps, op, fn)
                for op, fn in removed_decompositions_fallback_ops.items()
            ]

            # 2. Revert selected decompositions that have been removed from Inductor's registry for spyre
            removed_decompositions_to_exclude = getattr(
                enable_spyre_decompositions,
                "_removed_decompositions_to_exclude",
                {},
            )
            [
                torch._decomp._add_op_to_registry(decomps, op, fn)
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
                    decomps[spyre_decompositions_op] = saved_intree_decompositions[
                        spyre_decompositions_op
                    ]
                else:
                    decomps.pop(spyre_decompositions_op, None)

            # Clean up
            enable_spyre_decompositions._saved_decompositions = {}
            enable_spyre_decompositions._removed_decompositions_to_exclude = {}
            enable_spyre_decompositions._removed_decompositions_fallback_ops = {}


def _register_spyre_dispatchkey_kernels_permanently():
    """
    Permanently register PrivateUse1 / AutogradPrivateUse1 kernels for all ops
    in ``spyre_decompositions_via_dispatchkey``.

    This must be called once before any eager-mode dispatch can reach the Spyre
    kernels (typically from ``_SpyreImpl._lazy_init()``).  It is idempotent:
    subsequent calls are no-ops.

    The ``Library`` objects are stored in module-level globals so they are never
    garbage-collected (and therefore never unregistered from the C++ dispatcher).

    After registration, ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
    to route dispatch: inside a ``torch.compile`` context the Spyre function is called
    directly; outside (eager mode) the pre-compiled wrapper is used.
    """
    global _spyre_autograd_lib, _spyre_lib, _dispatchkey_kernels_registered

    if _dispatchkey_kernels_registered:
        return

    from torch.library import Library, fallthrough_kernel

    _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    _spyre_lib = Library("aten", "IMPL", "PrivateUse1")

    for op, wrapper_cls in spyre_decompositions_via_dispatchkey.items():
        # Autograd key: fall through so that the PrivateUse1 kernel is reached.
        _spyre_autograd_lib.impl(op._name, fallthrough_kernel)
        # PrivateUse1 key: the OPWrapper dispatches to spyre_fn.
        _spyre_lib.impl(op._name, wrapper_cls)

    _dispatchkey_kernels_registered = True


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
                # Pre-compile once so that repeated eager-mode calls reuse the
                # same compiled entry point rather than constructing a new
                # torch.compile wrapper on every invocation.
                self._compiled_fn = torch.compile(spyre_fn)

            def __call__(self, *args, **kwargs):
                # We are about to execute the op on spyre, hence the inputs are expected to be on spyre
                if any(
                    isinstance(x, torch.Tensor)
                    and getattr(x.device, "type", None) != DEVICE_NAME
                    for x in (pytree.tree_leaves(args) + pytree.tree_leaves(kwargs))
                ):
                    raise RuntimeError(
                        "Spyre decomposition function called with inputs being on a different device!"
                    )

                # Inside a torch.compile context (make_fx tracing, Inductor
                # lowering, etc.) call the function directly — wrapping it in
                # another torch.compile call would be incorrect.
                if torch.compiler.is_compiling():
                    return self.spyre_fn(*args, **kwargs)
                else:
                    # Eager mode: use the pre-compiled wrapper.
                    return self._compiled_fn(*args, **kwargs)

        def register(op):
            spyre_decompositions_via_dispatchkey[op] = OPWrapper(op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, ops)
        return fn

    return decomposition_decorator


@contextmanager
def enable_spyre_decompositions_via_dispatchkey():
    """
    Context manager that ensures the Spyre PrivateUse1 kernels are registered
    for the duration of a ``torch.compile`` call.

    Kernels are registered permanently in the C++ dispatcher by
    ``_register_spyre_dispatchkey_kernels_permanently()`` (idempotent).
    Once registered, ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
    to route dispatch: inside a ``torch.compile`` context the Spyre function is
    called directly; outside (eager mode) the pre-compiled wrapper is used.

    The CM is reentrant.
    """
    _register_spyre_dispatchkey_kernels_permanently()
    yield


@decomp.register_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


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


@register_spyre_decomposition([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


###############################################################################################
##                           Functions requiring dispatch keys                               ##
###############################################################################################
# The ops below require BOTH decorators:
#
# @register_spyre_decompositions_via_dispatchkey  — registers the PrivateUse1 kernel so that
#     eager-mode dispatch on a Spyre tensor reaches the Spyre implementation.
#
# @register_spyre_decomposition  — inserts the function into the Spyre decomposition table
#     so that make_fx (inside AOT Autograd) uses the Spyre implementation when tracing.
#     This is necessary because these ops are CompositeImplicitAutograd (CIA): make_fx
#     normally expands them via their default PyTorch implementation regardless of the
#     decompositions dict.  Explicitly registering a decomposition overrides that CIA
#     expansion and ensures the Spyre-specific path is traced instead.
@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.rms_norm.default])
@register_spyre_decomposition([torch.ops.aten.rms_norm.default])
def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_rms_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )

    # TODO: limitation with mean on dim=-1, transpose for now to avoid
    # https://github.com/torch-spyre/torch-spyre/issues/632
    input = input.transpose(-1, -2).contiguous()
    eps_tensor = torch.ops.spyre.full(
        input.shape, eps, dtype=torch.float16, device="spyre"
    )
    rsqrt_inp = (
        torch.rsqrt(torch.mean(input * input, dim=-2, keepdim=True)) + eps_tensor
    )
    output = (input * rsqrt_inp).transpose(-1, -2).contiguous()
    if weight is not None:
        output = output * weight
    return output


@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.layer_norm.default])
@register_spyre_decomposition([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_layer_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.gelu.default])
@register_spyre_decomposition([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.softplus.default])
@register_spyre_decomposition([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    return torch.ops.spyre.softplus(input, beta, threshold)
