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

from .constants import DEVICE_NAME, FP8_E4M3_MAX
from .errors import Unsupported
from . import customops  # noqa: F401
from . import spyre_hint
from torch_spyre._C import DataFormats, get_device_dtype

import threading


# Determine the float dtype for bool at module load time (not during tracing)
_BOOL_FLOAT_DTYPE = None


def _get_float_dtype_for_bool() -> torch.dtype:
    """
    Get the appropriate float dtype to convert boolean tensors on Spyre.
    Boolean tensors are stored as either FP16 or FP32 on the device.
    This is determined once at module load time to avoid tracing issues.
    """
    global _BOOL_FLOAT_DTYPE
    if _BOOL_FLOAT_DTYPE is None:
        device_dtype = get_device_dtype(torch.bool)
        # Map DataFormats to torch.dtype, defaulting to float16
        if device_dtype == DataFormats.IEEE_FP32:
            _BOOL_FLOAT_DTYPE = torch.float32
        else:
            _BOOL_FLOAT_DTYPE = torch.float16
    return _BOOL_FLOAT_DTYPE


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
    return torch.ops.aten.full(size, 1, dtype=dtype, layout=layout, device=device)


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
    return torch.ops.aten.full(
        size,
        1,
        dtype=dtype if dtype is not None else self.dtype,
        layout=layout,
        device=device if device is not None else self.device,
    )


@register_spyre_decomposition([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


@register_spyre_decomposition([torch.ops.aten.sign.default])
def spyre_sign(input: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(input)
    return torch.where(
        torch.gt(input, zero),
        torch.ones_like(input),
        torch.where(torch.lt(input, zero), -torch.ones_like(input), zero),
    )


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
    rsqrt_inp = torch.rsqrt(mean + eps)
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
    # F.layer_norm treats weight=None as identity and bias=None as zero;
    # spyre.layernormnorm doesn't handle missing args, so substitute defaults.
    if weight is None:
        weight = input.new_ones(normalized_shape)
    if bias is None:
        bias = input.new_zeros(normalized_shape)
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decomposition([torch.ops.aten.silu.default])
def silu(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.silu(input)


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
    head_dim = query.size(3)

    scaling_factor = scale
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(math.sqrt(head_dim))
    else:
        scaling_factor = math.sqrt(scaling_factor)

    if dropout_p > 0.0:
        raise Unsupported("Attention dropout not implemented for Spyre")

    expansion = num_heads // num_kvheads
    if expansion != 1:
        key = key.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        value = value.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)

    kv_block_size = 64
    q_block_size = 64

    output = torch.zeros_like(query)

    # FIXME: create a sparse M tensor via reduction
    M_reduced = torch.full(
        (batch_size, num_heads, max_seqlen_q, 64),
        float("-inf"),
        device=query.device,
        dtype=query.dtype,
    )
    M = M_reduced.amax(dim=-1)  # batch_size, num_heads, max_seqlen_q sparse

    # FIXME: create a sparse denominator tensor via reduction
    denominator_reduced = torch.zeros(
        (batch_size, num_heads, max_seqlen_q, 64),
        device=query.device,
        dtype=query.dtype,
    )
    denominator = denominator_reduced.amax(
        dim=-1
    )  # batch_size, num_heads, max_seqlen_q sparse

    # Precompute the causal additive mask once before entering the tiled loops.
    # Shape [1, 1, max_seqlen_q, max_seqlen_kv]: 0.0 = keep, -inf = masked.
    #
    # spyre::causal_mask builds the mask on CPU (tril + masked_fill_) and
    # transfers it to the query device. Wrapping this in a custom op makes the
    # CPU-side in-place ops opaque to torch.compile, so assert_functional_graph
    # is satisfied and the compiled graph sees only the resulting Spyre tensor.
    if is_causal:
        causal_mask = torch.ops.spyre.causal_mask(
            max_seqlen_q, max_seqlen_kv, query.dtype, query.device
        )

    with spyre_hint(tiles={"batch_size": max(1, batch_size // 2)}):
        with spyre_hint(tiles={"num_heads": max(1, num_heads // 4)}):
            with spyre_hint(
                tiles={"max_seqlen_q": max(1, max_seqlen_q // q_block_size)}
            ):
                with spyre_hint(
                    tiles={"max_seqlen_kv": max(1, max_seqlen_kv // kv_block_size)}
                ):
                    with spyre_hint(
                        work_div={"num_heads": 4, "max_seqlen_q": 8, "max_seqlen_kv": 8}
                    ):
                        scaled_keys = (
                            key * scaling_factor
                        )  # batch_size, num_heads, max_seqlen_kv, head_dim
                        keys_T = scaled_keys.transpose(
                            -1, -2
                        )  # batch_size, num_heads, head_dim, max_seqlen_kv
                        scores = torch.matmul(
                            query * scaling_factor, keys_T
                        )  # batch_size, num_heads, max_seqlen_q, max_seqlen_kv

                        if is_causal:
                            scores = scores + causal_mask

                        if attn_bias is not None:
                            scores = scores + attn_bias

                        block_max = torch.amax(
                            scores, dim=-1
                        )  # batch_size, num_heads, max_seqlen_q sparse
                        max_running = torch.maximum(
                            M, block_max
                        )  # batch_size, num_heads, max_seqlen_q sparse

                        exp_scores = torch.exp(
                            scores - max_running.unsqueeze(-1)
                        )  # batch_size, num_heads, max_seqlen_q, max_seqlen_kv
                        correction = torch.exp(
                            M - max_running
                        )  # batch_size, num_heads, max_seqlen_q sparse

                        denominator = torch.ops.spyre.copy_f(
                            denominator * correction + exp_scores.sum(dim=-1),
                            denominator,
                        )  # batch_size, num_heads, max_seqlen_q sparse
                        output = torch.ops.spyre.copy_f(
                            output * correction.unsqueeze(-1)
                            + torch.matmul(exp_scores, value),
                            output,
                        )  # batch_size, num_heads, max_seqlen_q, head_dim

                        M = torch.ops.spyre.copy_f(
                            max_running,
                            M,
                        )  # batch_size, num_heads, max_seqlen_q sparse

    output = torch.ops.spyre.copy_f(output / denominator.unsqueeze(-1), output)

    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    return (
        output.clone(memory_format=torch.contiguous_format),
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


@register_spyre_decomposition([torch.ops.aten.max.default])
def spyre_max_default_decomp(input):
    """
    Decompose torch.max(input) with conditional CPU fallback for int64.

    For int64 tensors, use custom op spyre::max_default_int64_fallback which has
    a CPU fallback registered in fallbacks.py.
    For other dtypes (float16, float32, etc.), use amax.
    """
    if input.dtype == torch.int64:
        # Use custom op with CPU fallback to avoid recursive decomposition
        # Returns a scalar (0D) tensor
        return torch.ops.spyre.max_default_int64_fallback(input)
    else:
        # Use amax for supported dtypes (can run on Spyre)
        # Returns a scalar (0D) tensor
        return torch.ops.aten.amax(input)


@register_spyre_decomposition([torch.ops.aten.max.dim])
def spyre_max_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.max(input, dim) with conditional handling for bool and int64.
    For bool: convert to float16, perform max, convert back (bool stored as fp16 on Spyre).
    For int64: use CPU fallback custom op (not supported on Spyre).
    For other dtypes: use default PyTorch decomposition (amax + argmax).
    """
    if input.dtype == torch.bool:
        # Reinterpret bool as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
        float_dtype = _get_float_dtype_for_bool()
        input_float = torch.ops.prims.convert_element_type(input, float_dtype)
        values_float = torch.ops.aten.amax(input_float, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input_float, dim=dim, keepdim=keepdim)
        values = torch.ops.prims.convert_element_type(values_float, torch.bool)
        return torch.return_types.max((values, indices))
    elif input.dtype == torch.int64:
        # Use CPU fallback custom op for int64
        return torch.ops.spyre.max_dim_int64_fallback(input, dim=dim, keepdim=keepdim)
    else:
        # Use amax and argmax for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amax(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input, dim=dim, keepdim=keepdim)
        return torch.return_types.max((values, indices))


@register_spyre_decomposition([torch.ops.aten.min.dim])
def spyre_min_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.min(input, dim) with conditional handling for bool and int64.
    For bool: convert to float16, perform min, convert back (bool stored as fp16 on Spyre).
    For int64: use CPU fallback custom op (not supported on Spyre).
    For other dtypes: use default PyTorch decomposition (amin + argmin).
    """
    if input.dtype == torch.bool:
        # Reinterpret bool as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
        float_dtype = _get_float_dtype_for_bool()
        input_float = torch.ops.prims.convert_element_type(input, float_dtype)
        values_float = torch.ops.aten.amin(input_float, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input_float, dim=dim, keepdim=keepdim)
        values = torch.ops.prims.convert_element_type(values_float, torch.bool)
        return torch.return_types.min((values, indices))
    elif input.dtype == torch.int64:
        # Use CPU fallback custom op for int64
        return torch.ops.spyre.min_dim_int64_fallback(input, dim=dim, keepdim=keepdim)
    else:
        # Use amin and argmin for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amin(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input, dim=dim, keepdim=keepdim)
        return torch.return_types.min((values, indices))


@register_spyre_decomposition([torch.ops.aten.amax.default])
def spyre_amax_decomp(
    input: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    """
    Decompose torch.amax for boolean tensors.
    For bool tensors: convert to float16, perform amax, convert back (bool stored as fp16 on Spyre).
    For other dtypes: return NotImplemented to use default behavior.
    """
    if input.dtype != torch.bool:
        # For non-bool types, don't decompose - use default lowering
        return NotImplemented

    # For bool tensors: reinterpret as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
    float_dtype = _get_float_dtype_for_bool()
    input_float = torch.ops.prims.convert_element_type(input, float_dtype)
    if dim is None:
        result_float = torch.ops.aten.amax(input_float, keepdim=keepdim)
    else:
        result_float = torch.ops.aten.amax(input_float, dim=dim, keepdim=keepdim)
    return torch.ops.prims.convert_element_type(result_float, torch.bool)


@register_spyre_decomposition([torch.ops.aten.amin.default])
def spyre_amin_decomp(
    input: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    """
    Decompose torch.amin for boolean tensors.
    For bool tensors: convert to float16, perform amin, convert back (bool stored as fp16 on Spyre).
    For other dtypes: return NotImplemented to use default behavior.
    """
    if input.dtype != torch.bool:
        # For non-bool types, don't decompose - use default lowering
        return NotImplemented

    # For bool tensors: reinterpret as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
    float_dtype = _get_float_dtype_for_bool()
    input_float = torch.ops.prims.convert_element_type(input, float_dtype)
    if dim is None:
        result_float = torch.ops.aten.amin(input_float, keepdim=keepdim)
    else:
        result_float = torch.ops.aten.amin(input_float, dim=dim, keepdim=keepdim)
    return torch.ops.prims.convert_element_type(result_float, torch.bool)


@register_spyre_decomposition([torch.ops.aten.ceil.default])
def spyre_ceil(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.neg.default(
        torch.ops.aten.floor.default(torch.ops.aten.neg.default(input))
    )


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


@register_spyre_decomposition([torch.ops.aten.convolution.default])
def conv2d_via_bmm_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    """
    Decompose 2D convolution into batch matrix multiplication using torch.nn.unfold.
    torch.nn.unfold directly returns (N, C_in * K_h * K_w, H_out * W_out), avoiding
    intermediate reshape/view/unsqueeze operations.
    """
    if transposed:
        raise Unsupported("conv2d_via_bmm: transposed convolution not supported")

    if any(op != 0 for op in output_padding):
        raise Unsupported("conv2d_via_bmm: output_padding not supported")

    if input.dim() != 4:
        raise Unsupported(f"conv2d_via_bmm: expected 4D input, got {input.dim()}D")

    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, K_h, K_w = weight.shape

    stride_h, stride_w = stride[0], stride[1]
    pad_h, pad_w = padding[0], padding[1]
    dil_h, dil_w = dilation[0], dilation[1]

    if C_in != groups * C_in_per_group:
        raise Unsupported(
            f"conv2d_via_bmm: expect C_in == groups * C_in_per_group, got C_in: {C_in}, groups: {groups} C_in_per_group: {C_in_per_group}"
        )

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    patches = torch.ops.spyre.unfold(
        input,
        kernel_size=(K_h, K_w),
        dilation=(dil_h, dil_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
    )

    if groups == 1:
        # weight_2d = weight.reshape(C_out, C_in_per_group * K_h * K_w)
        weight_2d = torch.ops.spyre.reshape_via_cpu(
            weight, (C_out, C_in_per_group * K_h * K_w)
        )
        weight_2d_exp = weight_2d.unsqueeze(0).expand(N, -1, -1)
        weight_2d_exp_cln = weight_2d_exp.clone()
        # output = torch.matmul(weight_2d, patches)
        output = torch.matmul(weight_2d_exp_cln, patches)
    else:
        C_out_per_group = C_out // groups
        # patches = patches.reshape(N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        patches = torch.ops.spyre.reshape_via_cpu(
            patches, (N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        )
        # weight_grouped = weight.reshape(groups, C_out_per_group, C_in_per_group * K_h * K_w)
        weight_grouped = torch.ops.spyre.reshape_via_cpu(
            weight, (groups, C_out_per_group, C_in_per_group * K_h * K_w)
        )

        output = torch.matmul(
            weight_grouped.unsqueeze(0),
            patches,
        )
        output = output.reshape(N, C_out, H_out * W_out)

    output = output.reshape(N, C_out, H_out, W_out)

    if bias is not None:
        # To ensure stick compatibility: reshape bias via reshape_via_cpu to (1, C_out, 1, 1).
        # The resulting tensor has a layout compatible with broadcasting to (N, C_out, H_out, W_out).
        bias_shaped = torch.ops.spyre.reshape_via_cpu(bias, (1, C_out, 1, 1))
        output = output + bias_shaped

    return output


# Register decomposition for custom spyre op (not aten, so use decomp.register_decomposition directly)
@decomp.register_decomposition(
    [torch.ops.spyre.dequantize_fp8_with_scale], spyre_decompositions
)
def dequantize_fp8_with_scale_decomp(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Decompose dequantize_fp8_with_scale into:
    1. FP8→FP16 conversion using .to() (triggers fp8todl16 via dtype_ops)
    2. Multiply by scale

    This decomposition is executed during compilation and removes the custom op
    from the graph before lowering.
    """
    x_fp16 = input.to(torch.float16)
    return x_fp16 * scale


@register_spyre_decomposition([torch.ops.aten.where.ScalarOther])
def where_scalar_other_decomp(condition, self, other):
    other_t = torch.full_like(self, other)
    return torch.ops.aten.where.self(condition, self, other_t)


@register_spyre_decomposition([torch.ops.aten.where.ScalarSelf])
def where_scalar_self_decomp(condition, self, other):
    self_t = torch.full_like(other, self)
    return torch.ops.aten.where.self(condition, self_t, other)


@register_spyre_decomposition([torch.ops.aten.where.Scalar])
def where_scalar_decomp(condition, self, other):
    # Must use dtype float16 for spyre backend where3
    dtype = torch.float16

    # Use full.default instead of full_like to explicitly control dtype
    self_t = torch.ops.aten.full.default(
        list(condition.shape),
        self,
        dtype=dtype,
        device=condition.device,
    )
    other_t = torch.ops.aten.full.default(
        list(condition.shape),
        other,
        dtype=dtype,
        device=condition.device,
    )

    return torch.ops.aten.where.self(condition, self_t, other_t)


@register_spyre_decomposition([torch.ops.spyre.quantize_fp8_with_scale])
def spyre_quantize_fp8_with_scale(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    inv_scale = torch.reciprocal(scale)
    x_scaled = input * inv_scale
    x_clamped = torch.ops.spyre.clamp(x_scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return torch.ops.spyre.qfp8ch(x_clamped)


@register_spyre_decomposition([torch.ops.aten.prod.dim_int])
def spyre_prod_dim_int(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    # Currently, restickify does not support fp32 (int64 is also converted to fp32
    # for now, so it is unsupported as well).
    # Use decomposition in these cases as a safe fallback, even if restickify
    # might not be needed in the end.
    if input.dtype != torch.float32 and input.dtype != torch.int64:
        return torch.ops.spyre.prod_dim_int(input, dim, keepdim)

    if dim < 0:
        dim += input.ndim
    out_shape = list(input.shape)
    reduce_size = out_shape.pop(dim)
    acc = torch.ones(out_shape, dtype=input.dtype, device=input.device)
    for i in range(reduce_size):
        acc = acc * input.select(dim, i)

    if keepdim:
        acc = acc.unsqueeze(dim)

    return acc


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
