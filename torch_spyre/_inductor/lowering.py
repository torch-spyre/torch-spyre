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

import torch

from torch._inductor.ir import ComputedBuffer, Reduction, Pointwise, Scatter, StorageBox
import torch._inductor.lowering as lowering
import torch._inductor.ir as ir
from .ir import SpyreConstantFallback

from typing import Any, Callable, Union

from .constants import BATCH_MATMUL_OP
import torch_spyre._inductor.customops  # noqa: F401
from torch_spyre.ops.fallbacks import fallback_ops
from .ir import SpyreReduction
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
    fn = lowering.ops_wrapper(torch.ops.spyre.overwrite.__name__)

    def inner_fn(index):
        return fn(input.make_loader()(index))

    def output_indexer(index):
        out_index = [*index]
        for dim, offset in zip(dims, offsets):
            out_index[dim] += offset
        return out_index

    inp = Scatter(
        device=input.get_device(),
        dtype=input.get_dtype(),
        inner_fn=inner_fn,
        ranges=input.get_size(),
        output_indexer=output_indexer,
    )

    output.realize()

    try:
        from torch._inductor.ir import MutationLayoutSHOULDREMOVE
    except ImportError:
        raise RuntimeError(
            "spyre::overwrite lowering: MutationLayoutSHOULDREMOVE is not available. "
            "Upstream likely removed/renamed it."
        )

    buffer = ComputedBuffer(
        name=None,
        layout=MutationLayoutSHOULDREMOVE(output),
        data=inp,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

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


@register_spyre_lowering(torch.ops.aten.slice.Tensor, type_promotion_kind=None)
def lower_slice(x, dim=0, start=None, end=None, step=1):
    result = lowering.slice_(x, dim=dim, start=start, end=end, step=step)
    return clone(result, memory_format=torch.contiguous_format)


@register_spyre_lowering(torch.ops.spyre.constant.default, type_promotion_kind=None)
def lower_constant(value, dtype, device):
    op_overload = getattr(
        torch.ops.spyre.constant, V.graph.current_node.target._overloadname
    )
    return ir.TensorBox.create(SpyreConstantFallback(op_overload, value, dtype, device))
