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
from warnings import warn

import torch

from torch._inductor.ir import Reduction, Pointwise, StorageBox
import torch._inductor.lowering as lowering
import torch._inductor.ir as ir
from typing import Any, Callable, Union

from .constants import (
    BATCH_MATMUL_OP,
    COPY_BACK_CANDIDATE_ATTR,
    BATCH_MATMUL_FP8_OP,
    SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
    SHARED_WEIGHT_UNIT_BMM_INFO_KEY,
)
import torch_spyre._inductor.customops  # noqa: F401
from torch_spyre.ops.fallbacks import fallback_ops
from .ir import SpyreReduction, SpyreConstantFallback, SpyreEmptyFallback
from torch_spyre._C import get_elem_in_stick
from torch._inductor.virtualized import V
from .errors import Unsupported
import threading
from .logging_utils import get_inductor_logger
import logging

logger = get_inductor_logger("lowering")

# A module-level lock + nesting counter to make the CM reentrant/thread-safe
_lowerings_lock = threading.RLock()
_lowerings_nesting = 0

# The specific spyre lowerings will be registered into this dictionary
# and merged with the in-tree lowerings when needed
spyre_lowerings: dict[Union[Callable[..., Any], str], Callable[..., Any]] = {}


def _current_fx_custom_meta() -> dict[str, Any]:
    node = V.get_current_node()
    meta = getattr(node, "meta", None)
    if not isinstance(meta, dict):
        return {}
    custom = meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def register_spyre_lowering(
    op,
    name=None,
    broadcast=False,
    type_promotion_kind=lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    override_return_dtype=None,
    convert_input_to_bool=False,
    lowering_dict=spyre_lowerings,
):
    name = name or op.__name__

    ensure_default_handler(name)

    lowering.register_op_dtype_propagation_rules(
        name=name,
        type_promotion_kind=type_promotion_kind,
        override_return_dtype=override_return_dtype,
    )
    return lowering.register_lowering(
        op,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
        lowering_dict=lowering_dict,
    )


# Implicit fallback to an eager op does not become effective when lowering of
# the op is registered by default. Here, we unregister ops that are falling back
# to eager ops
# Note: If an op has a decomposition defined, a lowering is not registered
def unregister_lowerings(fallback_ops, lowering_dict, allow_missing=False):
    saved_overloads = {}
    # Pass 1: Pre-check for exception safety (Fail-fast)
    if not allow_missing:
        missing = [
            overload
            for op in fallback_ops
            for overload in lowering.get_overloads(op)
            if overload not in lowering_dict
        ]
        if missing:
            raise RuntimeError(f"Cannot unregister. Missing lowerings for: {missing}")

    # Pass 2: Safely remove and store
    for op in fallback_ops:
        saved_overloads[op] = {}
        for overload in lowering.get_overloads(op):
            if overload in lowering_dict:
                # .pop() grabs the function and
                # deletes the key in one atomic step
                # if all overloads are unique then the op
                # key is not needed here.
                saved_overloads[op][overload] = lowering_dict.pop(overload)
    return saved_overloads


def restore_lowerings(saved_overloads, lowering_dict):
    for _, op_stored_overloads in saved_overloads.items():
        for overload, func in op_stored_overloads.items():
            lowering_dict[overload] = func


# Overload names for aten.clamp
_CLAMP_FUNC_OVS = ["default", "Tensor", "Tensor_minmax"]


# Context manager that enables spyre specific lowerings in addition to PyTorch in-tree lowerings
@contextmanager
def enable_spyre_lowerings():
    """
    CM that enables Spyre lowerings:
      - Temporarily redirect relevant aten ops → Spyre lowering
      - Restore original aten lowerings on exit

    This CM is reentrant and safe under nested usage.
    """
    global _lowerings_nesting
    with _lowerings_lock:
        first_enter = (_lowerings_nesting == 0)  # fmt: skip
        _lowerings_nesting += 1

        if first_enter:
            enable_spyre_lowerings._removed_fallbacks = {}
            enable_spyre_lowerings._removed_fallbacks = unregister_lowerings(
                fallback_ops, lowering.lowerings, allow_missing=True
            )
            saved_intree_lowerings = {}
            for spyre_lowering_op, spyre_lowering_impl in spyre_lowerings.items():
                if spyre_lowering_op in lowering.lowerings:
                    saved_intree_lowerings[spyre_lowering_op] = lowering.lowerings[
                        spyre_lowering_op
                    ]
                lowering.lowerings[spyre_lowering_op] = spyre_lowering_impl

            # Build adapters that call your Spyre lowering
            def _impl_lower_aten_clamp(x, min=None, max=None):
                return lower_clamp(x, min=min, max=max)

            def _impl_lower_aten_clamp_min(x, min):
                return lower_clamp(x, min=min, max=None)

            def _impl_lower_aten_clamp_max(x, max):
                return lower_clamp(x, min=None, max=max)

            # Collect overload handles
            clamp_ovs = [
                getattr(torch.ops.aten.clamp, name, None) for name in _CLAMP_FUNC_OVS
            ]
            clamp_min_ov = getattr(torch.ops.aten.clamp_min, "default", None)
            clamp_max_ov = getattr(torch.ops.aten.clamp_max, "default", None)

            # Save originals and patch — keep references in function attribute
            saved = {}

            def _save_set(ov, fn):
                if ov is None:
                    return
                saved[ov] = lowering.lowerings.get(ov)
                lowering.lowerings[ov] = fn

            for ov in clamp_ovs:
                _save_set(ov, _impl_lower_aten_clamp)
            _save_set(clamp_min_ov, _impl_lower_aten_clamp_min)
            _save_set(clamp_max_ov, _impl_lower_aten_clamp_max)

            # Attach to the function so we can restore on last exit
            enable_spyre_lowerings._saved_aten_lowerings = saved
            enable_spyre_lowerings._saved_lowerings = saved_intree_lowerings

        try:
            yield
        finally:
            _lowerings_nesting -= 1
            last_exit = (_lowerings_nesting == 0)  # fmt: skip
            if last_exit:
                # Restore on final exit
                saved = getattr(enable_spyre_lowerings, "_saved_aten_lowerings", {})
                for ov, prev in saved.items():
                    if prev is None:
                        lowering.lowerings.pop(ov, None)
                    else:
                        lowering.lowerings[ov] = prev
                # Clean up
                enable_spyre_lowerings._saved_aten_lowerings = {}
                # Reset the saved in-tree lowerings if needed
                saved_intree_lowerings = getattr(
                    enable_spyre_lowerings, "_saved_lowerings", {}
                )
                for spyre_lowering_op, spyre_lowering_impl in spyre_lowerings.items():
                    if spyre_lowering_op in saved_intree_lowerings:
                        lowering.lowerings[spyre_lowering_op] = saved_intree_lowerings[
                            spyre_lowering_op
                        ]
                    else:
                        lowering.lowerings.pop(spyre_lowering_op, None)
                restore_lowerings(
                    enable_spyre_lowerings._removed_fallbacks, lowering.lowerings
                )

                # Clean up
                enable_spyre_lowerings._saved_lowerings = {}
                enable_spyre_lowerings._removed_fallbacks = {}


def ensure_default_handler(op_name):
    """
    Install a default handler for a custom operator in DefaultHandler.

    DefaultHandler defines handlers for built‑in operators but does not
    automatically create one for custom ops, which leads to warnings like:

      UserWarning: undefined OpHandler.<op_name>, please add missing op schema

    This helper registers a fallback handler to suppress that warning.

    Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/ops_handler.py#L745

    TODO: Remove once the handler registration issue is resolved.
    """

    cls = torch._inductor.ops_handler.DefaultHandler
    if op_name not in cls.__dict__:
        method = cls._call_default(op_name)
        setattr(cls, op_name, method)


def eager_fallback(op, *args, **kwargs):
    handler = lowering.fallback_handler(op, add_to_fallback_set=False)
    return handler(*args, **kwargs)


# TODO:This is just place holder now; Real implementation will follow
@register_spyre_lowering(torch.ops.aten._scaled_mm.default)
def lower_scaled_mm(
    mat1,
    mat2,
    scale_a=None,
    scale_b=None,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    if scale_a is not None:
        raise Unsupported("scale_a parameter in _scaled_mm is not yet supported")
    if scale_b is not None:
        raise Unsupported("scale_b parameter in _scaled_mm is not yet supported")
    if bias is not None:
        raise Unsupported("bias parameter in _scaled_mm is not yet supported")
    if scale_result is not None:
        raise Unsupported("scale_result parameter in _scaled_mm is not yet supported")
    if use_fast_accum:
        raise Unsupported("use_fast_accum parameter in _scaled_mm is not yet supported")

    mat1.realize()
    mat2.realize()
    mat1_loader = mat1.make_loader()
    mat2_loader = mat2.make_loader()

    mat1_size = mat1.get_size()
    mat2_size = mat2.get_size()
    mat1_ndim = len(mat1_size)
    mat2_ndim = len(mat2_size)

    mat1_dtype = mat1.get_dtype()
    mat2_dtype = mat2.get_dtype()

    if mat1_dtype not in [torch.float8_e4m3fn]:
        raise ValueError(f"Expected FP8 input for mat1, got {mat1_dtype}")
    if mat2_dtype not in [torch.float8_e4m3fn]:
        raise ValueError(f"Expected FP8 input for mat2, got {mat2_dtype}")

    output_dtype = out_dtype if out_dtype is not None else torch.float16
    reduction_numel = mat1_size[-1]

    if mat1_ndim == 2 and mat2_ndim == 2:
        # [M, K] × [K, N] → [M, N]
        ranges = [mat1_size[0], mat2_size[1]]

        def inner_fn(index, reduction_index):
            i0, i1 = index
            (r0,) = reduction_index
            return (mat1_loader([i0, r0]), mat2_loader([r0, i1]))

    elif mat1_ndim == 3 and mat2_ndim == 2:
        # [B, M, K] × [K, N] → [B, M, N]
        ranges = [mat1_size[0], mat1_size[1], mat2_size[1]]

        def inner_fn(index, reduction_index):
            i0, i1, i2 = index
            (r0,) = reduction_index
            return (mat1_loader([i0, i1, r0]), mat2_loader([r0, i2]))

    elif mat1_ndim == 3 and mat2_ndim == 3:
        # [B, M, K] × [B, K, N] → [B, M, N]
        ranges = [mat1_size[0], mat1_size[1], mat2_size[2]]

        def inner_fn(index, reduction_index):
            i0, i1, i2 = index
            (r0,) = reduction_index
            return (mat1_loader([i0, i1, r0]), mat2_loader([i0, r0, i2]))

    else:
        raise Unsupported(
            f"_scaled_mm with shapes {mat1_size} and {mat2_size} not supported"
        )

    result = Reduction.create(
        reduction_type=BATCH_MATMUL_FP8_OP,
        input_node=[mat1, mat2],
        device=mat1.get_device(),
        dst_dtype=output_dtype,
        src_dtype=mat1_dtype,
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=[reduction_numel],
    )

    result.realize()

    if logger.isEnabledFor(logging.DEBUG):
        result_buf = V.graph.get_buffer(result.get_name())
        logger.debug(
            f"_scaled_mm (FP8): mat1{[int(s) for s in mat1_size]} @ mat2{[int(s) for s in mat2_size]} "
            f"-> {[int(s) for s in result_buf.get_size()]}, "
            f"mat1_dtype={mat1_dtype}, mat2_dtype={mat2_dtype}, out_dtype={output_dtype}"
        )

    return result


@register_spyre_lowering(torch.ops.aten.mm.default)
def lower_mm(x, y):
    x.realize()
    y.realize()
    x_loader = x.make_loader()
    y_loader = y.make_loader()

    x_size = x.get_size()
    y_size = y.get_size()
    x_ndim = len(x_size)
    y_ndim = len(y_size)

    reduction_numel = x_size[-1]  # K

    # Handle 3D input with 2D weight (batched matmul)
    if x_ndim == 3 and y_ndim == 2:
        ranges = [x_size[0], x_size[1], y_size[1]]  # [B, M, N]

        def inner_fn(index, reduction_index):
            i0, i1, i2 = index  # batch, row, col
            (r0,) = reduction_index
            return (x_loader([i0, i1, r0]), y_loader([r0, i2]))
    elif x_ndim == 2 and y_ndim == 2:
        ranges = [x_size[0], y_size[1]]

        def inner_fn(index, reduction_index):
            i0, i1 = index
            (r0,) = reduction_index
            return (x_loader([i0, r0]), y_loader([r0, i1]))
    else:
        raise ValueError(
            f"Unsupported tensor dimensions for mm: x.shape={x_size}, y.shape={y_size}. "
            f"Expected (2D, 2D) or (3D, 2D), got ({x_ndim}D, {y_ndim}D)"
        )

    if reduction_numel == 1:
        # Reduction degenerates to a pointwise mul
        result = lowering.mul(x, y)
    else:
        result = Reduction.create(
            reduction_type=BATCH_MATMUL_OP,
            input_node=[x, y],
            device=x.get_device(),
            dst_dtype=x.get_dtype(),
            src_dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=[reduction_numel],
        )

    result.realize()

    if logger.isEnabledFor(logging.DEBUG):
        result_buf = V.graph.get_buffer(result.get_name())
        logger.debug(
            f"mm: x{list(x_size)} @ y{list(y_size)} -> {list(result_buf.get_size())}, "
            f"x_layout={x.get_layout()}, y_layout={y.get_layout()}, out_layout={result_buf.get_layout()}"
        )

    return result


@register_spyre_lowering(torch.ops.spyre.batched_matmul.default)
@register_spyre_lowering(torch.ops.aten.bmm.default)
def lower_bmm(x, y):
    x.realize()
    y.realize()
    x_loader = x.make_loader()
    y_loader = y.make_loader()

    x_size = x.get_size()
    y_size = y.get_size()
    x_ndim = len(x_size)
    y_ndim = len(y_size)

    reduction_numel = x_size[-1]  # K

    if x_ndim == 3 and y_ndim == 3:
        ranges = [x_size[0], x_size[1], y_size[2]]  # B, M, N

        def inner_fn(index, reduction_index):
            i0, i1, i2 = index
            (r0,) = reduction_index
            tmp1 = x_loader([i0, i1, r0])
            tmp2 = y_loader([i0, r0, i2])
            return (tmp1, tmp2)
    elif x_ndim == 4 and y_ndim == 4:
        ranges = [x_size[0], x_size[1], x_size[2], y_size[-1]]

        def inner_fn(index, reduction_index):
            i0, i1, i2, i3 = index
            (r0,) = reduction_index
            tmp1 = x_loader([i0, i1, i2, r0])
            tmp2 = y_loader([i0, i1, r0, i3])
            return (tmp1, tmp2)
    elif x_ndim == 3 and y_ndim == 2:
        ranges = [x_size[0], x_size[1], y_size[1]]  # B, M, N

        def inner_fn(index, reduction_index):
            i0, i1, i2 = index
            (r0,) = reduction_index
            tmp1 = x_loader([i0, i1, r0])
            tmp2 = y_loader([r0, i2])
            return (tmp1, tmp2)
    else:
        raise Unsupported(f"BMM with input shapes {x.get_size()} and {y.get_size()}")

    custom_meta = _current_fx_custom_meta()
    op_info = {}
    if SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY in custom_meta:
        op_info[SHARED_WEIGHT_UNIT_BMM_INFO_KEY] = custom_meta[
            SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY
        ]

    if reduction_numel == 1:
        # Reduction degenerates to a pointwise mul
        result = lowering.mul(x, y)
    else:
        reduction_kwargs = dict(
            reduction_type=BATCH_MATMUL_OP,
            input_node=[x, y],
            device=x.get_device(),
            dst_dtype=x.get_dtype(),
            src_dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=[reduction_numel],
        )
        if op_info:
            result = SpyreReduction.create(op_info=op_info, **reduction_kwargs)
        else:
            result = Reduction.create(**reduction_kwargs)

    result.realize()

    if logger.isEnabledFor(logging.DEBUG):
        result_buf = V.graph.get_buffer(result.get_name())
        logger.debug(
            f"bmm: x{list(x_size)} @ y{list(y_size)} -> {list(result_buf.get_size())}"
        )

    return result


@register_spyre_lowering(torch.ops.spyre.exx2)
def lower_exx2(x, exx2Scale, useZeroMean):
    kwargs = lowering._make_reduction_inner(
        x, axis=[-1], keepdims=True, dtype=x.dtype, override_return_dtype=None
    )
    op_info = {
        "constants": {
            "exx2scale": exx2Scale,
            "useZeroMean": useZeroMean,
        }
    }
    result = SpyreReduction.create(
        reduction_type="exx2",
        input_node=x,
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=kwargs["inner_fn"],
        ranges=x.get_size()[:-1] + [1],
        reduction_ranges=kwargs["reduction_ranges"],
        op_info=op_info,
    )
    result.realize()
    return result


@register_spyre_lowering(torch.ops.spyre.layernormnorm)
def lower_layernormnorm(x, mean, norm_mean, weight, bias):
    fn = lowering.ops_wrapper(torch.ops.spyre.layernormnorm.__name__)

    def inner_fn(index):
        loaded_inputs = [
            x.make_loader()(index),
            mean.make_loader()(index),
            norm_mean.make_loader()(index),
        ]
        if weight is not None:
            loaded_inputs.append(weight.make_loader()(index[-1:]))
        if bias is not None:
            loaded_inputs.append(bias.make_loader()(index[-1:]))
        return fn(*loaded_inputs)

    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.spyre.layernormscale)
def lower_layernormscale(x, eps):
    fn = lowering.ops_wrapper(torch.ops.spyre.layernormscale.__name__)

    def inner_fn(index):
        return fn(x.make_loader()(index), eps)

    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.spyre.topkvalue)
def lower_topkvalue(x, k, dim):
    x_size = x.get_size()
    ndim = len(x_size)
    # Normalize dim to a positive index.
    norm_dim = dim % ndim
    loader = x.make_loader()

    if norm_dim == ndim - 1:
        # dim=-1 (or last dim): input shape [mb, n_in], reduce along n_in.
        # ranges=[mb, k]: index=[mb_idx, k_idx], rindex=[n_in_idx].
        mb = x_size[0]
        n_in = x_size[1]

        def inner_fn(index, rindex):
            return loader([index[0], rindex[0]])

        ranges = [mb, k]
        reduction_ranges = [n_in]
    else:
        # dim=0: input shape [n_in, mb], reduce along n_in (dim 0).
        # ranges=[k, mb]: index=[k_idx, mb_idx], rindex=[n_in_idx].
        mb = x_size[1]

        def inner_fn(index, rindex):
            # index = [k_idx, mb_idx], rindex = [n_in_idx]
            # Load from input at (n_in_idx, mb_idx); k_idx is the output row.
            return loader([rindex[0], index[1]])

        ranges = [k, mb]
        reduction_ranges = x_size[:1]

    result = Reduction.create(
        reduction_type="topkvalue",
        input_node=x,
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
    )
    result.realize()
    return result


@register_spyre_lowering(torch.ops.spyre.topkindex)
def lower_topkindex(x, k, dim):
    x_size = x.get_size()
    ndim = len(x_size)
    # Normalize dim to a positive index.
    norm_dim = dim % ndim
    loader = x.make_loader()

    if norm_dim == ndim - 1:
        # dim=-1 (or last dim): input shape [mb, n_in], reduce along n_in.
        # ranges=[mb, k]: index=[mb_idx, k_idx], rindex=[n_in_idx].
        mb = x_size[0]
        n_in = x_size[1]

        def inner_fn(index, rindex):
            return loader([index[0], rindex[0]])

        ranges = [mb, k]
        reduction_ranges = [n_in]
    else:
        # dim=0: input shape [n_in, mb], reduce along n_in (dim 0).
        # ranges=[k, mb]: index=[k_idx, mb_idx], rindex=[n_in_idx].
        mb = x_size[1]

        def inner_fn(index, rindex):
            # index = [k_idx, mb_idx], rindex = [n_in_idx]
            # Load from input at (n_in_idx, mb_idx); k_idx is the output row.
            return loader([rindex[0], index[1]])

        ranges = [k, mb]
        reduction_ranges = x_size[:1]

    result = Reduction.create(
        reduction_type="topkindex",
        input_node=x,
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
    )
    result.realize()
    return result


@register_spyre_lowering(torch.ops.aten.mean.dim)
def lower_mean(x, axis=None, keepdim=False, *, dtype=None):
    kwargs = lowering._make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=x.dtype, override_return_dtype=None
    )
    size = x.get_size()
    denom = torch._inductor.utils.sympy_product(size[i] for i in axis)
    scaling_factor = 1.0 / denom
    op_info = {"constants": {"scaling_factor": scaling_factor}}
    result = SpyreReduction.create(
        reduction_type="mean", input_node=x, op_info=op_info, **kwargs
    )
    result.realize()
    return result


@register_spyre_lowering(torch.ops.aten.mean.default)
def lower_mean_default(x, *, dtype=None):
    axis = list(range(len(x.get_size())))
    return lower_mean(x, axis=axis, keepdim=False, dtype=dtype)


@register_spyre_lowering(torch.ops.spyre.gelu)
def lower_gelu(x, approximate="none"):
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda index: lowering.ops_wrapper(torch.ops.spyre.gelu.__name__)(
            x.make_loader()(index)
        ),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.spyre.silu)
def lower_silu(x):
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda index: lowering.ops_wrapper(torch.ops.spyre.silu.__name__)(
            x.make_loader()(index)
        ),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.spyre.softplus)
def lower_softplus(x, beta=1.0, threshold=20.0):
    fn = lowering.ops_wrapper(torch.ops.spyre.softplus.__name__)

    def inner_fn(index):
        return fn(x.make_loader()(index), beta, threshold)

    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.spyre.clamp)
def lower_clamp(x, min=None, max=None):
    if min is None:
        min = torch.finfo(torch.float16).min
    if max is None:
        max = torch.finfo(torch.float16).max
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda index: lowering.ops_wrapper(torch.ops.spyre.clamp.__name__)(
            x.make_loader()(index), min, max
        ),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.aten.clone.default, type_promotion_kind=None)
def clone(x, *, memory_format=None):
    from torch._inductor.ir import FlexibleLayout, get_stride_order
    from torch._inductor.lowering import clone as clone_lowering

    result = clone_lowering(x, memory_format=memory_format)
    # Upstream Inductor ignores memory_format (TODO in clone lowering).
    # The output gets a FlexibleLayout whose stride order is inferred from
    # the input's strides via ComputedBuffer.get_fill_order(). When the
    # input is a non-contiguous view (e.g. a permute), the clone output
    # inherits those strides instead of the requested memory format.
    # This causes index/stride mismatches during Spyre's stickify pass.
    # Fix: freeze the layout to the requested stride order so that
    # decide_layout() respects the memory_format contract.
    if memory_format is not None and memory_format != torch.preserve_format:
        stride_order = get_stride_order(
            FlexibleLayout.stride_ordered_for_memory_format(
                result.get_size(), memory_format
            )
        )
        result.realize()
        result.freeze_layout_with_stride_order(stride_order)
    return result


@register_spyre_lowering(torch.ops.spyre.copy_from_d2d)
def lower_spyre_from_d2d(src, dst):
    lowering.mutate_to(dst, src)


@register_spyre_lowering(torch.ops.spyre.overwrite)
def lower_overwrite(input, output, dims, offsets):
    depr_msg = """torch.ops.spyre.overwrite is deprecated. Use standard PyTorch operations like \
output[indices] = input or output[indices].copy_(input). Please report any incompatibilities."""
    warn(depr_msg, FutureWarning, stacklevel=1)

    sliced_output = output
    for dim, offset in zip(dims, offsets):
        input_size_at_dim = input.get_size()[dim]
        sliced_output = ir.SliceView.create(
            sliced_output, dim, offset, offset + input_size_at_dim
        )
    lowering.mutate_to(sliced_output, input)
    return output


@register_spyre_lowering(torch.ops.spyre.restickify)
def lower_restickify(x):
    # Restickify must operate on base tensors, so we need
    # to unwrap any views.
    base = x
    while not isinstance(base, StorageBox):
        base = base.data

    # Force realization so base has a buffer name and make_loader() emits
    # ops.load(name, ...) rather than inlining the producer's inner_fn.
    # Without this, ComputedBuffer.make_loader() may inline when num_reads()==0,
    # capturing a closure that later resolves to the restickify buffer itself
    # (after pw.realize() assigns the name), creating a self-dependency cycle.
    base.realize()

    loader = base.make_loader()

    def inner_fn(index):
        return loader(index)

    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=base.get_size(),
        origin_node=V.get_current_node(),
        traceback=x.get_traceback(),
    )

    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.aten.full.default, type_promotion_kind=None)
def lower_full(size, fill_value, dtype=None, layout=None, device=None, pin_memory=None):
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype not in (torch.float16, torch.float32):
        return ir.TensorBox.create(
            ir.FallbackKernel.create(
                torch.ops.aten.full.default,
                size,
                fill_value,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
            )
        )
    scalar = ir.TensorBox.create(
        SpyreConstantFallback(
            torch.ops.spyre.constant.default, float(fill_value), dtype, device
        )
    )
    scalar_loader = scalar.make_loader()

    def inner_fn(index):
        return scalar_loader([])

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )


@register_spyre_lowering(torch.ops.spyre.constant.default, type_promotion_kind=None)
def lower_constant(value, dtype, device):
    op_overload = getattr(
        torch.ops.spyre.constant, V.graph.current_node.target._overloadname
    )
    return ir.TensorBox.create(SpyreConstantFallback(op_overload, value, dtype, device))


@register_spyre_lowering(torch.ops.spyre.empty.default, type_promotion_kind=None)
def lower_empty(size, device, dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    op_overload = getattr(
        torch.ops.spyre.empty, V.graph.current_node.target._overloadname
    )
    return ir.TensorBox.create(
        SpyreEmptyFallback(op_overload, list(size), device, dtype)
    )


def _peel(node):
    """Unwrap TensorBox/StorageBox/MutableBox layers to reach the underlying Buffer."""
    while isinstance(node, ir.MutableBox):
        node = node.data
    while isinstance(node, ir.StorageBox):
        node = node.data
    return node


def _copy_back_candidate(dst, src) -> bool:
    """Whether ``copy_(dst, src)`` is worth checking after layout propagation.

    Lowering only identifies the structural pattern.  Layout propagation later
    proves the full safety condition and either removes the copy or leaves this
    normal ``copy_`` mutation op intact.
    """
    dst_buf = _peel(dst)
    if not isinstance(dst_buf, ir.InputBuffer):
        return False
    if dst_buf.get_name() not in V.graph.graph_input_names:
        return False

    if dst.get_device() != src.get_device():
        return False
    if dst.get_dtype() != src.get_dtype():
        return False
    if tuple(dst.get_size()) != tuple(src.get_size()):
        return False

    src_buf = _peel(src)
    if not isinstance(src_buf, ir.ComputedBuffer):
        return False
    if not isinstance(src_buf.layout, ir.FlexibleLayout):
        return False
    return tuple(dst_buf.layout.stride) == tuple(src_buf.layout.stride)


def _mark_copy_back_candidate(first_new_op: int, dst) -> None:
    dst_name = _peel(dst).get_name()
    for op in V.graph.operations[first_new_op:]:
        layout = getattr(op, "layout", None)
        if not isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            continue
        if layout.get_buffer().get_name() == dst_name:
            setattr(op, COPY_BACK_CANDIDATE_ATTR, True)


@register_spyre_lowering(torch.ops.aten.copy_.default, type_promotion_kind=None)
def spyre_copy_(dst, src, non_blocking=False):
    """Lower ``copy_`` and mark graph-input copy-back candidates.

    Do not alias at lowering time.  Candidate marking keeps the structural
    connection to ``copy_`` while letting layout propagation make the final,
    feasibility-aware decision after producer layouts are known.
    """
    if dst is src:
        return dst

    candidate = _copy_back_candidate(dst, src)
    src = lowering.to_device(src, dst.get_device())
    src = lowering.to_dtype(src, dst.get_dtype())
    src = lowering.expand(src, dst.get_size())

    first_new_op = len(V.graph.operations)
    result = lowering.mutate_to(dst, src)
    if candidate:
        _mark_copy_back_candidate(first_new_op, dst)
    return result


@register_spyre_lowering(torch.ops.aten.cat.default, type_promotion_kind=None)
def lower_cat(inputs, dim=0):
    output_size = list(inputs[0].get_size())
    output_size[dim] = sum(x.get_size()[dim] for x in inputs)

    dtype = inputs[0].get_dtype()
    device = inputs[0].get_device()
    output = lowering.empty(output_size, dtype=dtype, device=device)

    offset = 0
    for input_tensor in inputs:
        sliced_output = ir.SliceView.create(
            output, dim, offset, offset + input_tensor.get_size()[dim]
        )
        lowering.mutate_to(sliced_output, input_tensor)
        offset += input_tensor.get_size()[dim]

    return output


@register_spyre_lowering(
    torch.ops.aten.constant_pad_nd.default, type_promotion_kind=None
)
def lower_constant_pad_nd(input, pad, value=0, align_to_stick=False):
    # pad is in reverse dim order: (left_last, right_last, left_2nd_last, right_2nd_last, ...)
    bounds = list(reversed(list(zip(pad[::2], pad[1::2]))))
    sizes = input.get_size()
    n = len(sizes) - len(bounds)

    # Apply cropping (negative padding) if needed
    cropped_input = input
    for i, (left, right) in enumerate(bounds):
        if left < 0 or right < 0:
            dim = n + i
            size = sizes[n + i] if i == 0 else cropped_input.get_size()[dim]
            start = max(0, -left)
            end = size - max(0, -right)
            cropped_input = ir.SliceView.create(cropped_input, dim, start, end)

    # Apply positive padding
    cropped_sizes = cropped_input.get_size()
    output_size = list(cropped_sizes[:n])
    dims: list[int] = []
    offsets: list[int] = []

    for (left, right), size in zip(bounds, cropped_sizes[n:]):
        pad_left = max(0, left)
        pad_right = max(0, right)

        if pad_left + pad_right == 0:
            output_size.append(size)
            continue

        dim = len(output_size)
        output_size.append(size + pad_left + pad_right)
        dims.append(dim)
        offsets.append(pad_left)

    if not dims:
        return clone(cropped_input)

    dtype = input.get_dtype()
    device = input.get_device()
    output = lowering.empty(output_size, dtype=dtype, device=device)
    pad_constant = lower_constant(value, dtype, device)

    # Fill padding regions. If align_to_stick is enabled, use stick-aligned offsets.
    # Extra padding from alignment will be overwritten by input.
    stick_size = get_elem_in_stick(dtype)
    for (left, right), dim in zip(bounds, range(n, len(output_size))):
        pad_left = max(0, left)
        pad_right = max(0, right)
        if pad_left + pad_right == 0:
            continue

        def fill_padding(count, offset):
            if align_to_stick:
                count += offset % stick_size
                offset = (offset // stick_size) * stick_size

            pad_size = list(output_size)
            pad_size[dim] = count
            pad_view = lowering.expand(pad_constant, pad_size)
            sliced_output = ir.SliceView.create(output, dim, offset, offset + count)
            lowering.mutate_to(sliced_output, pad_view)

        if pad_left > 0:
            fill_padding(pad_left, 0)
        if pad_right > 0:
            fill_padding(pad_right, output_size[dim] - pad_right)

    # Copy cropped input into the output at the correct offsets
    sliced_output = output
    for i, dim in enumerate(dims):
        sliced_output = ir.SliceView.create(
            sliced_output, dim, offsets[i], offsets[i] + cropped_input.get_size()[dim]
        )

    # Mutate the slice to contain the cropped input data
    lowering.mutate_to(sliced_output, cropped_input)

    return output


@register_spyre_lowering(
    torch.ops.prims.convert_element_type.default,
    type_promotion_kind=None,
)
def to_dtype(x, dst_dtype):
    from torch_spyre._inductor.dtype_ops import DtypeOpTable

    src_dtype = x.get_dtype()

    if src_dtype == dst_dtype:
        return clone(x)

    # Check if conversion is supported by backend
    if DtypeOpTable.get_operator(src_dtype, dst_dtype) is None:
        # Unsupported conversion - fall back to CPU
        op = torch.ops.spyre.to_dtype_cpu.default
        return eager_fallback(op, x, dst_dtype)

    return lowering.to_dtype(x, dst_dtype, copy=True)


def with_int64_fallback(fn, *args, convert_output=True):
    """
    Helper to handle int64 operations by converting to fp32.

    Args:
        fn: The lowering function to call
        *args: Arguments to pass to fn
        convert_output: If True, convert output back to int64.
                       Set to False for operations like div that should return float.
    """
    # Skip constants (int/float literals) that don't have get_dtype()
    has_int64 = False
    for x in args:
        if isinstance(x, (int, float)):
            continue
        if hasattr(x, "get_dtype") and x.get_dtype() == torch.int64:
            has_int64 = True
            break

    if not has_int64:
        return fn(*args)

    # Convert args, skipping constants
    converted_args = []
    for x in args:
        if isinstance(x, (int, float)):
            converted_args.append(x)
        else:
            converted_args.append(to_dtype(x, torch.float32))

    output = fn(*converted_args)

    if convert_output:
        return to_dtype(output, torch.int64)

    return output


@register_spyre_lowering(
    torch.ops.aten.add.Tensor,
    type_promotion_kind=None,
)
def lower_add(x, y, *, alpha=1):
    if alpha != 1:
        alpha_tensor = lower_full(
            y.get_size(),
            float(alpha),
            dtype=y.get_dtype(),
            device=y.get_device(),
        )
        alpha_tensor.realize()
        y = with_int64_fallback(lowering.mul, y, alpha_tensor)
        y.realize()
    return with_int64_fallback(lowering.add, x, y)


@register_spyre_lowering(
    torch.ops.aten.mul.Tensor,
    type_promotion_kind=None,
)
def lower_mul(x, y):
    return with_int64_fallback(lowering.mul, x, y)


@register_spyre_lowering(
    torch.ops.aten.sub.Tensor,
    type_promotion_kind=None,
)
def lower_sub(x, y, *, alpha=1):
    if alpha != 1:
        alpha_tensor = lower_full(
            y.get_size(),
            float(alpha),
            dtype=y.get_dtype(),
            device=y.get_device(),
        )
        alpha_tensor.realize()
        y = with_int64_fallback(lowering.mul, y, alpha_tensor)
        y.realize()
    return with_int64_fallback(lowering.sub, x, y)


@register_spyre_lowering(
    torch.ops.aten.minimum.default,
    type_promotion_kind=None,
)
def lower_minimum(x, y):
    return with_int64_fallback(lowering.minimum, x, y)


@register_spyre_lowering(
    torch.ops.aten.maximum.default,
    type_promotion_kind=None,
)
def lower_maximum(x, y):
    return with_int64_fallback(lowering.maximum, x, y)


@register_spyre_lowering(torch.ops.spyre.qfp8ch)
def lower_qfp8ch(x):
    """
    Lower qfp8ch operation - channel-wise FP8 format conversion.

    Pointwise format conversion only (no scaling).
    """

    fn = lowering.ops_wrapper(torch.ops.spyre.qfp8ch.__name__)
    x_loader = x.make_loader()

    def inner_fn(index):
        return fn(x_loader(index))

    pw = Pointwise.create(
        device=x.get_device(),
        dtype=torch.float8_e4m3fn,
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw
