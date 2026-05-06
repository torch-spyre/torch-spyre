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

import math
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
    torch.ops.aten.triu,
    torch.ops.aten.tril,
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

    For ``aten`` ops, this also registers a PrivateUse1 dispatch kernel
    (via ``register_spyre_decompositions_via_dispatchkey``) so that
    eager-mode dispatch on a Spyre tensor reaches the Spyre implementation.
    This is necessary for ops with CompositeImplicitAutograd (CIA) in
    upstream PyTorch, and harmless for non-CIA ops.
    """

    def decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # 1. Register in the Spyre decomposition table (for compile mode / make_fx)
        decomp.register_decomposition(ops, spyre_decompositions)(fn)

        # 2. For aten ops, also register via PrivateUse1 dispatch key (for eager mode).
        #    Non-aten ops (e.g. spyre::compact) are custom Spyre ops that don't need
        #    PrivateUse1 kernel registration.
        #    Skip ops that already have a PrivateUse1 kernel (e.g. from eager.py) to
        #    avoid registration conflicts.
        ops_list = ops if isinstance(ops, list) else [ops]
        aten_ops = [
            op
            for op in ops_list
            if getattr(op, "namespace", None) == "aten"
            and not torch._C._dispatch_has_kernel_for_dispatch_key(
                op._name, "PrivateUse1"
            )
        ]
        if aten_ops:
            register_spyre_decompositions_via_dispatchkey(aten_ops)(fn)

        return fn

    return decorator


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
        from torch_spyre.ops.fallbacks import fallback_ops
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
                self._compiled_fn = torch.compile(spyre_fn, dynamic=False)

            def __call__(self, *args, **kwargs):
                # We are about to execute the op on spyre, hence the inputs are expected to be on spyre
                if any(
                    isinstance(x, torch.Tensor)
                    and getattr(x.device, "type", None) != DEVICE_NAME
                    for x in (pytree.tree_leaves(args) + pytree.tree_leaves(kwargs))
                ):
                    args_device = [
                        x.device if isinstance(x, torch.Tensor) else None
                        for x in (pytree.tree_leaves(args) + pytree.tree_leaves(kwargs))
                    ]
                    raise RuntimeError(
                        f"Spyre decomposition function called with inputs being on a different device! Args devices: {args_device=}"
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


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# Implement ones via identity broadcast: create a size-1 tensor (ones_scalar), expand to
# target size, then clone (identity) to materialize. Clone op with identity is merged.
@register_spyre_decomposition([torch.ops.aten.ones.default])
def ones_decomp(
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    scalar = torch.ops.spyre.ones_scalar(device, dtype=dtype)
    expanded = scalar.expand(size)
    return expanded.clone()


@register_spyre_decomposition([torch.ops.aten.new_ones.default])
def new_ones_decomp(
    self: torch.Tensor,
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    dev = device if device is not None else self.device
    dt = dtype if dtype is not None else self.dtype
    scalar = torch.ops.spyre.ones_scalar(dev, dtype=dt)
    expanded = scalar.expand(size)
    return expanded.clone()


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


@register_spyre_decomposition([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


@register_spyre_decomposition([torch.ops.aten.addmm.default, torch.ops.aten.addmm.out])
def addmm_decomp(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: Union[int, float] = 1,
    alpha: Union[int, float] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Decompose addmm into basic operations: out = beta * input + alpha * (mat1 @ mat2)
    """
    # Compute matrix multiplication using matmul to handle batched tensors
    mm_result = mat1 @ mat2

    # Apply alpha scaling if needed
    if alpha != 1:
        mm_result = alpha * mm_result

    # Apply beta scaling and add input if needed
    if beta == 0:
        result = mm_result
    elif beta == 1:
        result = input + mm_result
    else:
        result = beta * input + mm_result

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result


###############################################################################################
##                           Spyre decompositions for aten ops                               ##
###############################################################################################
# For aten ops, ``register_spyre_decomposition`` automatically registers both a
# decomposition table entry (for compile mode / make_fx) and a PrivateUse1
# dispatch kernel (for eager mode).  The latter is essential for ops with
# CompositeImplicitAutograd (CIA) in upstream PyTorch (e.g. rms_norm,
# layer_norm), and harmless for non-CIA ops (e.g. gelu, softplus).
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

    mean = torch.mean(input * input, dim=-1, keepdim=True)
    eps_tensor = torch.ops.spyre.full((1,), eps, dtype=torch.float16, device="spyre")
    rsqrt_inp = torch.rsqrt(mean + eps_tensor)
    output = input * rsqrt_inp
    if weight is not None:
        output = output * weight
    return output


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


@register_spyre_decomposition([torch.ops.aten.topk])
def spyre_topk(
    input: torch.Tensor,
    k: int,
    dim: Optional[int] = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k > 4:
        raise Unsupported("Topk is not supported for this config")
    return torch.ops.spyre.topkvalue(input, k, dim), torch.ops.spyre.topkindex(
        input, k, dim
    )


@register_spyre_decomposition([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decomposition([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    return torch.ops.spyre.softplus(input, beta, threshold)


@register_spyre_decomposition([torch.ops.aten.linear.default])
def spyre_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    weight = weight.transpose(-1, -2)
    while weight.dim() < input.dim():
        weight = torch.unsqueeze(weight, 0)
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


@register_spyre_decomposition(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default]
)
def spyre__sdpa_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = query.size(0)
    num_heads = query.size(1)
    num_kvheads = key.size(1)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)

    query = query.clone(memory_format=torch.contiguous_format)
    key = key.clone(memory_format=torch.contiguous_format)
    value = value.clone(memory_format=torch.contiguous_format)

    scaling_factor = scale
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(query.shape[-1])
    scaling_factor = math.sqrt(scaling_factor)

    # TODO (aviros): Figure why this broadcast doesn't work
    scaling_factor_q = torch.full_like(query, scaling_factor)
    scaling_factor_k = torch.full_like(key, scaling_factor)

    query = query * scaling_factor_q
    key = key * scaling_factor_k

    expansion = num_heads // num_kvheads
    if expansion != 1:
        key = key.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        value = value.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
    key_t = key.transpose(-2, -1).clone(memory_format=torch.contiguous_format)

    attn = torch.matmul(query, key_t)

    if is_causal:
        assert attn_bias is None
        attn_bias = torch.full_like(attn, float("-inf"))
        attn_bias = attn_bias.triu(diagonal=1)

    if attn_bias is not None:
        attn = attn + attn_bias

    # TODO (aviros): Switch to _safe_softmax
    attn = torch.softmax(attn, -1)

    if dropout_p > 0.0:
        # TODO(aviros): Implement
        raise Unsupported("Attention dropout not implemented for Spyre")

    # Unused for now
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    # B, H, S, E
    out = torch.matmul(attn, value)

    # B, S, H, E
    # This is needed to maintain the API promise from SDPA (attn needs to have same size+stride as q)
    out = out.transpose(1, 2).clone(memory_format=torch.contiguous_format)

    # Returns (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
    return (
        out.transpose(1, 2),
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


## TODO(imaihal): Need to fix scalar tensor shape mismatch during Spyre-to-CPU transfer.
## See: https://github.com/torch-spyre/torch-spyre/issues/1172
## This will be enabled after solving this.
# @register_spyre_decomposition([torch.ops.aten.max.default])
# def spyre_max_default_decomp(input):
#    """
#    Decompose torch.max(input) with conditional CPU fallback for int64.
#
#    For int64 tensors, use custom op spyre::max_default_int64_fallback which has
#    a CPU fallback registered in fallbacks.py.
#    For other dtypes (float16, float32, etc.), use amax.
#    """
#    if input.dtype == torch.int64:
#        # Use custom op with CPU fallback to avoid recursive decomposition
#        # Returns a scalar (0D) tensor
#        return torch.ops.spyre.max_default_int64_fallback(input)
#    else:
#        # Use amax for supported dtypes (can run on Spyre)
#        # Returns a scalar (0D) tensor
#        return torch.ops.aten.amax(input)


@register_spyre_decomposition([torch.ops.aten.max.dim])
def spyre_max_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.max(input, dim) with conditional CPU fallback for int64.

    For int64 tensors, use custom op spyre::max_dim_int64_fallback which has
    a CPU fallback registered in fallbacks.py.
    For other dtypes (float16, float32, etc.), decompose into amax and argmax operations.

    Returns a named tuple (values, indices) as expected by torch.max.

    # TODO (imaihal): Decomposed into torch.topk with k=1 to obtain both values and indices,
    #  or implement argmax in the backend compiler to get indices
    """
    if input.dtype == torch.int64:
        # Use custom op with CPU fallback to avoid recursive decomposition
        return torch.ops.spyre.max_dim_int64_fallback(input, dim, keepdim)
    else:
        # Use amax and argmax for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amax(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input, dim=dim, keepdim=keepdim)
        return torch.return_types.max((values, indices))


@register_spyre_decomposition([torch.ops.aten.cat.default])
def decompose_cat(
    tensors: list[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    orig_decomp = torch._inductor.decomposition.cat(tensors, dim)
    if orig_decomp == NotImplemented:
        expanded_size = 0
        for t in tensors:
            expanded_size += t.size(dim)
        output_size = list(tensors[0].size())
        output_size[dim] = expanded_size
        output = tensors[0].new_empty(output_size)
        offset = 0
        for input in tensors:
            output = torch.ops.spyre.overwrite_f(
                input=input, output=output, dims=[dim], offsets=[offset]
            )
            offset += input.size(dim)
        return output
    else:
        return orig_decomp


@register_spyre_decomposition([torch.ops.aten.constant_pad_nd.default])
def pad_decomp(
    input: torch.Tensor,
    pad: list[int],
    value: float = 0,
) -> torch.Tensor:
    # pad is in reverse dim order: (left_last, right_last, left_2nd_last, right_2nd_last, ...)
    n_dims_padded = len(pad) // 2

    # Negative pad values (cropping) require reading from a non-zero storage
    # offset or a sub-stick position, neither of which the SFP supports.
    if any(p < 0 for p in pad):
        raise Unsupported(
            f"constant_pad_nd: negative padding (cropping) is not supported on "
            f"Spyre (pad={pad})"
        )

    # Left-padding on the last (stick) dimension shifts the output start address
    # by `left` elements. The hardware can only express this in whole sticks, so
    # `left` must be a multiple of the stick size (64 fp16 elements).
    # Sub-stick left-padding on the last dimension is tracked in:
    # https://github.com/torch-spyre/torch-spyre/issues/1464
    last_dim_left = pad[0]
    if last_dim_left > 0:
        elems_per_stick = 128 // input.element_size()
        if last_dim_left % elems_per_stick != 0:
            raise Unsupported(
                f"constant_pad_nd: sub-stick left-padding on the last dimension is "
                f"not supported on Spyre (pad={pad}, left={last_dim_left}, "
                f"stick_size={elems_per_stick})"
            )

    # Build the padded output shape and collect which dimensions need padding.
    scalar = torch.ops.spyre.full([1], value, input.device, dtype=input.dtype)
    output_size = list(input.size())
    dims: list[int] = []
    offsets: list[int] = []
    for i in range(n_dims_padded - 1, -1, -1):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        if left + right == 0:
            continue
        dim = input.dim() - 1 - i
        output_size[dim] += left + right
        dims.append(dim)
        offsets.append(left)

    if not dims:
        return input

    output = scalar.expand(output_size).clone()
    output = torch.ops.spyre.overwrite_f(
        input=input, output=output, dims=dims, offsets=offsets
    )
    return output


@register_spyre_decomposition([torch.ops.aten.bitwise_not])
def bitwise_not(input: torch.Tensor) -> torch.Tensor:
    if input.dtype is torch.bool:
        return torch.logical_not(input)
    else:
        neg_one = torch.ops.aten.full_like(input, -1)
        return torch.ops.aten.bitwise_xor(input, neg_one)


@register_spyre_decomposition([torch.ops.aten.bitwise_and])
def bitwise_and(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if input1.dtype is torch.bool and input2.dtype is torch.bool:
        return torch.ops.aten.logical_and(input1, input2)
    else:
        return torch.ops.aten.bitwise_not(
            torch.ops.aten.bitwise_or(
                torch.ops.aten.bitwise_not(input1), torch.ops.aten.bitwise_not(input2)
            )
        )


###############################################################################################
##                           Register custom kernels for Spyre.                              ##
###############################################################################################
# Kernels are registered permanently in the C++ dispatcher by
# ``_register_spyre_dispatchkey_kernels_permanently()`` (idempotent).
# Once registered, ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
# to route dispatch: inside a ``torch.compile`` context the Spyre function is
# called directly; outside (eager mode) the pre-compiled wrapper is used.
# Note: This has to stay at the end of the file.
_register_spyre_dispatchkey_kernels_permanently()
