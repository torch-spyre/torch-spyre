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

import sympy
import torch

from torch._inductor.ir import Reduction, Pointwise, StorageBox
import torch._inductor.lowering as lowering
import torch._inductor.ir as ir
from typing import Any, Callable, Union

from .constants import (
    AVGPOOL2D_OP,
    BATCH_MATMUL_OP,
    CONV2D_FWD_OP,
    COPY_BACK_CANDIDATE_ATTR,
    BATCH_MATMUL_FP8_OP,
    DEPTHWISE_CONV2D_OP,
    SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
    SHARED_WEIGHT_UNIT_BMM_INFO_KEY,
)
from . import config
import torch_spyre._inductor.customops  # noqa: F401
from torch_spyre.ops.fallbacks import fallback_ops
from .ir import (
    SpyreReduction,
    SpyreConstantFallback,
    SpyreEmptyFallback,
    BroadcastAsyncFallback,
    WaitWorkFallback,
    AllGatherAsyncFallback,
    AllReduceAsyncFallback,
)
from torch_spyre._C import get_elem_in_stick
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
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


def register_fallback_over_decomp(fallback_ops):
    """Register explicit fallback lowerings for fallback ops that also carry an
    in-tree Inductor decomposition, returning the overloads newly added.

    PT 2.12 added core-aten decompositions for several ops Spyre falls back to
    (arange, tril, triu, isin, index_copy.out, ...). When such an op has no
    registered lowering, Inductor's graph lowering auto-invokes
    ``make_fallback(op)`` *without* ``override_decomp``, which asserts
    ``both a fallback and a decomp for same op``. Pre-registering the fallback
    with ``override_decomp=True`` installs a lowering so that auto-path — and
    its assertion — is never reached.

    Only overloads that are in ``lowering.decompositions`` and currently lack a
    lowering are touched, so this composes with ``unregister_lowerings`` (which
    runs first) and does not clobber Spyre's own lowerings.
    """
    added = []
    for op in fallback_ops:
        for overload in lowering.get_overloads(op):
            if (
                overload in lowering.decompositions
                and overload not in lowering.lowerings
            ):
                lowering.make_fallback(overload, override_decomp=True)
                added.append(overload)
    return added


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
            # Install explicit fallbacks for ops that also have an in-tree
            # decomposition (PT 2.12), so Inductor never hits the asserting
            # auto-``make_fallback`` path for them.
            enable_spyre_lowerings._added_fallbacks = register_fallback_over_decomp(
                fallback_ops
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
                # Remove the fallback lowerings we added for decomp-carrying
                # ops before restoring the originally-removed ones.
                for overload in getattr(enable_spyre_lowerings, "_added_fallbacks", []):
                    lowering.lowerings.pop(overload, None)
                    lowering.fallbacks.discard(overload)
                enable_spyre_lowerings._added_fallbacks = []
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


def _ensure_synthetic_origin(result, target, args: tuple) -> None:
    """Give a lowering result a synthetic ``target`` origin FX node, so Spyre
    layout passes (which key off ``op.data.origins[].target``) recognize it even
    when the lowering was called directly, without an FX node of its own. No-op
    if a ``target`` origin already exists.
    """

    def _realized_buffer(node):
        while not isinstance(node, StorageBox):
            node = node.data
        return node.data

    origins = getattr(_realized_buffer(result), "origins", None)
    if origins and any(o.target == target for o in origins):
        return

    fx_graph = V.graph.graph
    first_compute_node = next(n for n in fx_graph.nodes if n.op != "placeholder")
    with fx_graph.inserting_before(first_compute_node):
        fx_node = fx_graph.create_node("call_function", target, args)

    # Realize so the origin lands on a stable ComputedBuffer, not a Pointwise.
    result.realize()
    buf = _realized_buffer(result)
    # buf.data is a frozen Loops; override its origins via object.__setattr__.
    object.__setattr__(buf.data, "origins", OrderedSet([fx_node]))
    buf.origins = OrderedSet([fx_node])


@register_spyre_lowering(torch.ops.spyre.scaled_mm.default)
def lower_scaled_mm(
    mat1,
    mat2,
    out_dtype=None,
):
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
            f"scaled_mm: mat1{[int(s) for s in mat1_size]} @ mat2{[int(s) for s in mat2_size]} "
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


@register_spyre_lowering(torch.ops.spyre.conv2d.default)
def lower_depthwise_conv2d(x, w, stride, padding, dilation, groups):
    x = V.graph.get_buffer(x.realize())
    w = V.graph.get_buffer(w.realize())
    x_loader = x.make_loader()
    w_loader = w.make_loader()

    # Input / weight shapes
    N, C_in, H_in, W_in = x.get_size()
    C_out, G, K_h, K_w = w.get_size()

    H_in_padded = H_in + 2 * padding[0]
    W_in_padded = W_in + 2 * padding[1]

    if C_out != C_in or C_in != groups:
        raise Unsupported(
            f"Input and output channels and groups should all be equal for depthwiseconv2d: {C_in}, {C_out}, {groups}"
        )

    if tuple(padding) != (0, 0):
        raise Unsupported(
            f"Depthwise conv2d currently only supports zero padding; got padding={padding}. "
            "Support for non-zero padding requires changes to the Spyre runtime to handle "
            "non-zero tensor allocation addresses."
        )

    if tuple(dilation) != (1, 1):
        raise Unsupported(
            f"Depthwise conv2d currently only supports dilation=1; got dilation={dilation}. "
            "Support for dilation > 1 requires backend changes."
        )

    # Output spatial sizes
    H_out = (H_in + 2 * padding[0] - K_h) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - K_w) // stride[1] + 1

    def inner_fn(index, reduction_index):
        # Output indices
        n, c, ho, wo = index
        # Reduction indices: may be [kh, kw] or [kh, kw, g] depending on whether G is 1
        kh = reduction_index[0]
        kw = reduction_index[1]
        g = reduction_index[2] if len(reduction_index) > 2 else 0

        x_val = x_loader([n, c, ho, wo])

        # Depthwise filter: one filter per input channel
        w_val = w_loader([c, g, kh, kw])

        return (x_val, w_val)

    op_info = {
        "conv_params": {
            "stride_i": stride[0],
            "stride_j": stride[1],
            "pad_i": padding[0],
            "pad_j": padding[1],
            "dilation_i": dilation[0],
            "dilation_j": dilation[1],
            "total_size_i": H_in_padded,
            "total_size_j": W_in_padded,
            "kernel_h": K_h,
            "kernel_w": K_w,
            "pad_type": "padded_fullspan_wunneeded"
            if (padding[0] == 0 and padding[1] == 0)
            else "padded_nozeropad",
        }
    }
    # Only include G in reduction_ranges if it's not 1 (size-1 dims get simplified away anyway)
    red_ranges = [K_h, K_w]
    if G != 1:
        red_ranges.append(G)

    result = SpyreReduction.create(
        reduction_type=DEPTHWISE_CONV2D_OP,
        input_node=[x, w],
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=[N, C_out, H_out, W_out],
        reduction_ranges=red_ranges,
        op_info=op_info,
    )

    result.realize()
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


def _topk_reduction_kwargs(x, k, dim):
    """Build Reduction.create kwargs for topk along an arbitrary dim/rank.

    Unlike a normal reduction, topk keeps `k` elements along the reduced
    dim rather than collapsing it to size 1, so lowering._make_reduction_inner
    can't be reused directly.
    """
    x_size = x.get_size()
    ndim = len(x_size)
    norm_dim = dim % ndim
    loader = x.make_loader()

    ranges = list(x_size)
    ranges[norm_dim] = k
    reduction_ranges = [x_size[norm_dim]]

    def inner_fn(index, rindex):
        full_index = list(index)
        full_index[norm_dim] = rindex[0]
        return loader(full_index)

    return dict(
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
    )


@register_spyre_lowering(torch.ops.spyre.topkvalue)
def lower_topkvalue(x, k, dim):
    result = Reduction.create(
        reduction_type="topkvalue",
        input_node=x,
        **_topk_reduction_kwargs(x, k, dim),
    )
    result.realize()
    return result


@register_spyre_lowering(torch.ops.spyre.topkindex)
def lower_topkindex(x, k, dim):
    result = Reduction.create(
        reduction_type="topkindex",
        input_node=x,
        **_topk_reduction_kwargs(x, k, dim),
    )
    result.realize()
    return result


@register_spyre_lowering(torch.ops.aten.avg_pool2d.default)
def lower_avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if ceil_mode:
        raise Unsupported("avg_pool2d ceil_mode not supported")
    if not count_include_pad:
        raise Unsupported("avg_pool2d count_include_pad=False not supported")
    if divisor_override is not None:
        raise Unsupported("avg_pool2d divisor_override not supported")

    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = stride if stride is not None else kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    padding = padding if padding else 0
    pH, pW = (padding, padding) if isinstance(padding, int) else padding

    if pH > 0 or pW > 0:
        raise Unsupported("avg_pool2d with non-zero padding not yet supported")

    if kH == 1 or kW == 1:
        # avgpoolfwd is a windowed reduction; a 1-wide kernel has no pooling
        # window along that axis (it is an identity or a strided subsample),
        # which the pool datapath cannot express — the DDL rejects a windowless
        # pool ("Unknown primary dimension kind ... for a window dimension").
        # Spyre also has no eager avg_pool2d kernel to fall back to.  So delegate
        # to the in-tree Inductor lowering, which decomposes avg_pool2d into
        # pointwise/reduction ops the Spyre backend does support and keeps k=1
        # on-device.
        return lowering.avg_pool2d(
            x,
            [kH, kW],
            [sH, sW],
            [pH, pW],
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    N, C, H_in, W_in = x.get_size()

    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1

    op_info = {
        "constants": {
            "kernel_h": kH,
            "kernel_w": kW,
            "stride_h": sH,
            "stride_w": sW,
            "pad_h": pH,
            "pad_w": pW,
            "scaling_factor": 1.0 / (kH * kW),
        },
    }

    def inner_fn(index, reduction_index):
        n, c, ho, wo = index
        kh, kw = reduction_index
        hi = ho * sH - pH + kh
        wi = wo * sW - pW + kw
        return x.make_loader()([n, c, hi, wi])

    result = SpyreReduction.create(
        reduction_type=AVGPOOL2D_OP,
        input_node=x,
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=[N, C, H_out, W_out],
        reduction_ranges=[kH, kW],
        op_info=op_info,
    )
    # Realize so the pool becomes its own ComputedBuffer: codegen's
    # kernel_store_reduction reads op_info (pool constants) off node.data.op_info,
    # which only exists when the SpyreReduction is realized rather than fused into
    # a consumer.
    result.realize()
    return result


@register_spyre_lowering(torch.ops.aten.convolution.default)
def lower_convolution(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """Direct lowering of fp16 conv2d to a native ``conv2d`` SDSC (PE array).

    conv2d is a two-input reduction (activation + weight, weight->KERNEL layout,
    ``pt`` execution unit, contraction over the input-channel dim ``in``) with
    windowed spatial dims (``ki``/``kj`` mapping to output ``i``/``j`` via
    stride/pad/dilation) -- a hybrid of the matmul (lower_mm/lower_bmm) and
    avgpool (lower_avg_pool2d) patterns. Selected via config.conv2d_direct_lowering;
    when off, aten.convolution decomposes to im2col+matmul (conv2d_via_bmm_decomp).

    v1 scope: fp16, groups==1, non-transposed, 4D input. Bias (if present) is a
    separate pointwise add, not a fused biasadd computeOp (follow-up).
    """
    # Lock-step invariant with the decomposition: conv2d_via_bmm_decomp only
    # declines (returns NotImplemented, letting aten.convolution survive to this
    # lowering) when `config.conv2d_direct_lowering and _is_direct_conv_supported`
    # (see decompositions.py). So reaching here with the flag off means the
    # decomposition failed to run -- an internal wiring bug, not an unsupported
    # user op -- so assert rather than raise Unsupported (which would masquerade
    # as a graceful fallback that can no longer happen post-decomposition). The
    # guards below mirror _is_direct_conv_supported in IR-node terms (x/weight
    # expose get_size()/get_dtype(), not the torch.Tensor shape/dtype the
    # predicate reads), keeping the two in lock-step.
    assert config.conv2d_direct_lowering, (
        "lower_convolution reached with conv2d_direct_lowering off; "
        "conv2d_via_bmm_decomp should have decomposed the op"
    )
    if transposed:
        raise Unsupported("conv2d direct lowering: transposed convolution")
    if any(op != 0 for op in output_padding):
        raise Unsupported("conv2d direct lowering: output_padding")
    if groups != 1:
        raise Unsupported(f"conv2d direct lowering: groups={groups} (only groups==1)")
    if any(p != 0 for p in padding):
        raise Unsupported(f"conv2d direct lowering: padding={padding} (only 0)")
    if any(d != 1 for d in dilation):
        raise Unsupported(f"conv2d direct lowering: dilation={dilation} (only 1)")
    if len(x.get_size()) != 4:
        raise Unsupported(
            f"conv2d direct lowering: expected 4D input, got {len(x.get_size())}D"
        )
    if x.get_dtype() != torch.float16:
        raise Unsupported(f"conv2d direct lowering: dtype {x.get_dtype()} (fp16 only)")
    if weight.get_size()[-2] == 1 and weight.get_size()[-1] == 1:
        # Only a 1x1 kernel squeezes *both* ki and kj to size-1, leaving the
        # conv SDSC with no window dims -- which the backend's conv path rejects in
        # dimension-mapping. A 1x1 conv is a channel matmul; it stays on the
        # im2col+matmul decomposition (see _is_direct_conv_supported). A 1xN /
        # Nx1 kernel keeps one window dim and direct-lowers fine (a 1-D conv;
        # verified by the test_conv2d_direct k1x3 / k3x1 cases).
        raise Unsupported("conv2d direct lowering: 1x1 kernel (use decomposition)")
    kh_w, kw_w = int(weight.get_size()[-2]), int(weight.get_size()[-1])
    if kh_w > 3 or kw_w > 3:
        # k>3 overflows the dense C_in*kH*kW contraction's LX budget in the backend;
        # mirrors the _CONV_MAX_KERNEL exclusion in _is_direct_conv_supported.
        raise Unsupported(
            f"conv2d direct lowering: kernel {kh_w}x{kw_w} > 3 (use decomposition)"
        )
    C_in_size = x.get_size()[1]
    eps = get_elem_in_stick(x.get_dtype())
    if not (isinstance(C_in_size, (int, sympy.Integer)) and int(C_in_size) % eps == 0):
        # Spyre sticks C as the innermost dim and the conv SDSC contracts over
        # C_in with no partial-stick handling, so a C_in that is not a whole
        # multiple of the fp16 stick width (get_elem_in_stick == 64) would need
        # contraction-dim padding this path does not emit. Mirrors the
        # stick-alignment guard in _is_direct_conv_supported.
        raise Unsupported(
            f"conv2d direct lowering: C_in={C_in_size} not a multiple of the "
            f"stick width {eps} (use decomposition)"
        )

    x.realize()
    weight.realize()
    x_loader = x.make_loader()
    weight_loader = weight.make_loader()

    N, C_in, H_in, W_in = x.get_size()
    C_out, C_in_per_group, kH, kW = weight.get_size()

    sH, sW = stride[0], stride[1]
    pH, pW = padding[0], padding[1]
    dilH, dilW = dilation[0], dilation[1]

    H_out = (H_in + 2 * pH - dilH * (kH - 1) - 1) // sH + 1
    W_out = (W_in + 2 * pW - dilW * (kW - 1) - 1) // sW + 1

    # Ragged input width (W_in - kW) % sW != 0: the fp16 opfunc's width tiling
    # mis-accumulates the dangling column; mirrors _is_direct_conv_supported.
    if isinstance(W_in, (int, sympy.Integer)) and (int(W_in) - int(kW)) % int(sW) != 0:
        raise Unsupported(
            f"conv2d direct lowering: ragged input width "
            f"(W_in={int(W_in)} - kW={int(kW)}) % sW={int(sW)} != 0 (use decomposition)"
        )

    op_info = {
        # opConsts_ in the emitted SDSC must stay {} for the plain fp16 conv op
        # (fused epilog scalars live on separate computeOps). Conv geometry goes
        # under conv_params, which superdsc reads to build padding_sizes/window
        # fields -- it is NOT copied into opConsts.
        "constants": {},
        "conv_params": {
            "kernel_h": kH,
            "kernel_w": kW,
            "stride_h": sH,
            "stride_w": sW,
            "pad_h": pH,
            "pad_w": pW,
            "dil_h": dilH,
            "dil_w": dilW,
        },
        # NOTE: conv iteration-space dim roles are NOT snapshotted here.  Codegen
        # recovers each dim's role (channel / in_channel / win_h / win_w /
        # out_h / out_w / batch) structurally from the args' access expressions
        # -- set membership and co-occurrence in device_coordinates, never sizes
        # or positions (see _match_labels_by_structure in codegen/superdsc.py) --
        # so views and device-layout assignment stay authoritative.
    }

    def inner_fn(index, reduction_index):
        n, co, ho, wo = index
        r_in, r_ki, r_kj = reduction_index
        # Unclamped windowed input coordinates; zero-padding is expressed at the
        # SDSC level via padFront_/padBack_ in padding_sizes (see superdsc
        # _conv_sdsc_fields), mirroring the avgpool window mechanism.
        hi = ho * sH - pH + r_ki * dilH
        wi = wo * sW - pW + r_kj * dilW
        act = x_loader([n, r_in, hi, wi])
        # weight is [C_out, C_in_per_group, kH, kW]; groups==1 so C_in_per_group == C_in.
        ker = weight_loader([co, r_in, r_ki, r_kj])
        return (act, ker)

    result = SpyreReduction.create(
        reduction_type=CONV2D_FWD_OP,
        input_node=[x, weight],
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=[N, C_out, H_out, W_out],
        reduction_ranges=[C_in, kH, kW],
        op_info=op_info,
    )
    result.realize()

    if bias is not None:
        # v1: separate pointwise add (broadcast bias [C_out] over NCHW) rather
        # than a fused biasadd computeOp. Reshape to (1, C_out, 1, 1) for the
        # channel-wise broadcast.
        bias_reshaped = lowering.view(bias, [1, C_out, 1, 1])
        result = lowering.add(result, bias_reshaped)

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
    result = lowering.clone(x, memory_format=memory_format)

    # A clone called directly from another lowering has no aten.clone FX node,
    # so it inherits the caller's origin and the layout pass skips clone's
    # layout propagation enforcement. Inject a synthetic clone origin so it fires.
    args: tuple = ()
    if isinstance(x, ir.IRNode) and (n := x.get_origin_node()) is not None:
        args = (n,)
    _ensure_synthetic_origin(result, torch.ops.aten.clone.default, args)

    # Upstream Inductor ignores memory_format (TODO in clone lowering).
    # The output gets a FlexibleLayout whose stride order is inferred from
    # the input's strides via ComputedBuffer.get_fill_order(). When the
    # input is a non-contiguous view (e.g. a permute), the clone output
    # inherits those strides instead of the requested memory format.
    # This causes index/stride mismatches during Spyre's stickify pass.
    # Fix: freeze the layout to the requested stride order so that
    # decide_layout() respects the memory_format contract.
    if memory_format is not None and memory_format != torch.preserve_format:
        stride_order = ir.get_stride_order(
            ir.FlexibleLayout.stride_ordered_for_memory_format(
                result.get_size(), memory_format
            )
        )
        result.realize()
        result.freeze_layout_with_stride_order(stride_order)
    return result


def _reoffset(node, offset):
    """Re-introduce a storage_offset onto a graph-input node in-graph.

    A tensor that was sliced OUTSIDE the compiled region (e.g. a
    ``x.narrow(0, 2, 1)`` handed to a standalone-compiled op) reaches the
    lowering as a placeholder whose FixedLayout carries the right size and
    stride but ``offset == 0`` — upstream Inductor's placeholder path reads
    ``static_sizes_strides`` only and drops ``storage_offset`` (see
    torch/_inductor/graph.py). The Spyre SpyreTensorLayout likewise has no
    offset field, so the compiled kernel binds the storage BASE pointer
    (job_plan.cpp / spyre_stream.cpp use ``storage().data_ptr()``) and reads
    from element 0 regardless of the view's true offset.

    To make the offset survive into the SDSC binary it must live in the
    in-graph coordinate: superdsc.py bakes a per-dim byte offset from the
    coordinate's constant term (``as_coeff_Add()[0]``). We therefore rebuild
    the node as a ReinterpretView over the same storage with the offset
    installed on the layout — the same mechanism aten.slice / SliceView use.
    Size and stride are preserved from the input's own layout.

    REQUIREMENT — the offset is a single *host-space* flat scalar (in host
    elements). ``compute_coordinates`` (views.py) distributes it across dims
    against the *device* ``stride_map``; the innermost (stick) dim holds
    ``elems_per_stick`` elements (64 at fp16, 128 at fp8), and the backend has
    no mechanism to bake an offset that lands *inside* a stick. So a re-injected
    offset is only representable when it is a whole multiple of
    ``elems_per_stick`` — i.e. it steps by complete sticks. This was measured
    directly (see ``sweep_d2d_offsets.py`` / ``tests/inductor/
    test_copy_from_d2d_offsets.py``): across contiguous ``narrow`` and
    ``select`` views of every inner size, the copy is correct iff
    ``offset % elems_per_stick == 0`` and otherwise the restickify pass raises
    "no mechanism to resolve stick incompatibility". Notably this depends on the
    OFFSET, not the inner-dim size: e.g. an inner dim of 96 (not a stick
    multiple) still copies correctly at offset 192 (== 3 sticks) but errors at
    offset 96 (== 1.5 sticks).

    ``_validate_reoffset_supported`` converts the unaligned case into a clean,
    actionable ``Unsupported`` at lowering time instead of the cryptic
    restickify error deeper in the pipeline. It does NOT guard the separate
    offset-*within*-the-stick-dim case (a column narrow, e.g.
    ``x.narrow(1, 64, 64)``), which the backend still miscompiles silently — see
    ``test_column_slice_inner_offset`` (tracked for the follow-up PR).
    """
    if offset == 0:
        return node
    storage, old_layout = ir.as_storage_and_layout(node)
    _validate_reoffset_supported(old_layout, offset)
    new_layout = ir.FixedLayout(
        old_layout.device,
        old_layout.dtype,
        list(old_layout.size),
        list(old_layout.stride),
        sympy.expand(offset),
    )
    return ir.TensorBox(ir.ReinterpretView(data=storage, layout=new_layout))


def _validate_reoffset_supported(layout, offset) -> None:
    """Fail fast when a D2D-copy offset is not a whole number of sticks.

    A re-injected storage_offset must step by complete sticks: the innermost
    device dim holds ``elems_per_stick`` elements and the backend cannot bake an
    offset that lands inside a stick. Measured behavior (``sweep_d2d_offsets.py``)
    is unambiguous — across contiguous ``narrow`` / ``select`` views of every
    inner size, the copy is correct iff ``offset % elems_per_stick == 0``; every
    unaligned offset instead raises deep in the restickify pass ("no mechanism
    to resolve stick incompatibility"). This check surfaces the same rejection
    earlier with an actionable message rather than silently miscompiling or
    emitting a cryptic downstream error.

    Scope — this rule is keyed on the OFFSET alone, so it holds regardless of
    rank reduction (``select`` drops the outer dim but the flat offset is
    unchanged) and does not need the device layout, which is not attached to the
    placeholder yet. It deliberately does NOT cover an offset that falls *inside*
    the stick dimension itself (a column narrow): that case is
    ``offset % elems_per_stick == 0`` yet still miscompiles, and detecting it
    would require the base-storage layout that is unavailable here. It stays a
    documented known limitation (``test_column_slice_inner_offset``).
    """
    try:
        off = int(offset)
    except (TypeError, ValueError):
        # Symbolic offset: cannot check statically, let it through
        # (specialize_int bakes concrete offsets, so this is the rare path).
        return
    if off == 0:
        return
    eps = get_elem_in_stick(layout.dtype)
    if off % eps != 0:
        raise Unsupported(
            "spyre::copy_from_d2d of a sliced view requires a stick-aligned "
            f"storage_offset: {off} is not a multiple of elems_per_stick={eps} "
            f"(dtype {layout.dtype}). The offset must step by whole sticks; an "
            "offset landing inside a stick cannot be baked into the kernel "
            "coordinate. Re-slice on a stick-aligned boundary (a multiple of "
            f"{eps} elements), or copy the full (unsliced) tensor."
        )


@register_spyre_lowering(torch.ops.spyre.copy_from_d2d)
def lower_spyre_from_d2d(src, dst, src_off, dst_off):
    # A sliced src/dst reaches us as a graph input whose storage_offset was
    # dropped (offset==0 on its layout). Re-introduce the offsets in-graph so
    # they land in the coordinate that superdsc bakes into the kernel; without
    # this the kernel reads/writes from the storage base and every offset
    # silently returns the first call's data. See _reoffset above.
    src = _reoffset(src, src_off)
    dst = _reoffset(dst, dst_off)
    lowering.mutate_to(dst, src)


def _build_mutation_lowering(src, dst):
    # Shared lowering body for copy_forced and opaque_copy_: builds an
    # explicit MutationLayoutSHOULDREMOVE buffer so the mutation into dst
    # survives regardless of what the scheduler would otherwise decide.
    # mutate_to() has multiple code paths and does not always mutate, so
    # the buffer is constructed by hand here instead.
    src = lowering.to_dtype(src, dst.get_dtype())
    src = lowering.expand(src, dst.get_size())

    pw = Pointwise.create(
        device=dst.get_device(),
        dtype=dst.get_dtype(),
        inner_fn=src.make_loader(),
        ranges=list(dst.get_size()),
    )

    dst.realize()

    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.MutationLayoutSHOULDREMOVE(dst),
        data=pw.data.data,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    return dst


@register_spyre_lowering(torch.ops.spyre.copy_forced)
def lower_spyre_copy_forced(src, dst):
    return _build_mutation_lowering(src, dst)


@register_spyre_lowering(torch.ops.spyre.opaque_copy_)
def lower_spyre_opaque_copy_(value, acc):
    # opaque_copy_ is functional at the FX/AOTAutograd level (see customops.py)
    # so that assert_functional_graph never sees a mutation. The real
    # mutating write into acc is introduced here, at lowering time, via the
    # same MutationLayoutSHOULDREMOVE(acc) buffer that lower_spyre_copy_forced
    # builds for copy_forced. Everything downstream that keys off
    # MutationLayoutSHOULDREMOVE (e.g. wsr/coarse_tile.py) treats this
    # identically to a copy_forced write.
    return _build_mutation_lowering(value, acc)


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


@register_spyre_lowering(torch.ops.aten.slice_scatter.default, type_promotion_kind=None)
def lower_slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    size = self.get_size()
    dim = dim % len(size)

    if step != 1:
        # Only a unit step maps to a single SliceView.
        raise Unsupported(
            f"slice_scatter with step={step} is not supported on Spyre "
            f"(only unit step maps to a SliceView mutation)"
        )

    start = 0 if start is None else start
    end = size[dim] if end is None else end

    output = lowering.clone(self)
    output.realize()
    sliced_output = ir.SliceView.create(output, dim, start, end)
    lowering.mutate_to(sliced_output, src)
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


def _peel_through_views(node):
    """Like _peel, but also unwraps views (reshape/expand/slice/etc.).

    A view (e.g. ir.ReinterpretView) is not a MutableBox/StorageBox, so plain
    _peel stops at the view instead of reaching the Buffer underneath it.
    """
    node = _peel(node)
    while isinstance(node, ir.BaseView):
        node = _peel(node.data)
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
        return lowering.clone(cropped_input)

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
def to_dtype(x, dst_dtype, use_compute_types=True):
    # PT 2.12 passes a ``use_compute_types`` kwarg to registered dtype-conversion
    # lowerings; accept and forward it to the in-tree lowering.
    from torch_spyre._inductor.dtype_ops import DtypeOpTable

    src_dtype = x.get_dtype()

    if src_dtype == dst_dtype:
        return lowering.clone(x)

    # Check if conversion is supported by backend
    if not DtypeOpTable.is_supported(src_dtype, dst_dtype):
        # Unsupported conversion - fall back to CPU
        op = torch.ops.spyre.to_dtype_cpu.default
        return eager_fallback(op, x, dst_dtype)

    # DMA-copied bool InputBuffers have a different HBM element ordering than
    # computation-kernel outputs, so a stick-reordering typecast (e.g.
    # DL16TOFP32 for bool → float32) reads them back shuffled (28/64 elements).
    # Fall back to CPU for host bools whose conversion reorders sticks; an
    # IDENTITY byte-copy (e.g. bool → float16) is safe. On-device computed bools
    # use the native path. Peel through views (reshape/expand) too, since a
    # viewed host bool is still backed by the same DMA-copied InputBuffer.
    if src_dtype == torch.bool and DtypeOpTable.bool_host_conversion_reorders_sticks(
        dst_dtype
    ):
        if isinstance(_peel_through_views(x), ir.InputBuffer):
            op = torch.ops.spyre.to_dtype_cpu.default
            return eager_fallback(op, x, dst_dtype)

    return lowering.to_dtype(
        x, dst_dtype, copy=True, use_compute_types=use_compute_types
    )


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
    broadcast=True,
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
    broadcast=True,
)
def lower_mul(x, y):
    return with_int64_fallback(lowering.mul, x, y)


@register_spyre_lowering(
    torch.ops.aten.sub.Tensor,
    type_promotion_kind=None,
    broadcast=True,
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
    broadcast=True,
)
def lower_minimum(x, y):
    return with_int64_fallback(lowering.minimum, x, y)


@register_spyre_lowering(
    torch.ops.aten.maximum.default,
    type_promotion_kind=None,
    broadcast=True,
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


@register_spyre_lowering(torch.ops.spyre.qfp8wt)
def lower_qfp8wt(x):
    """
    Lower qfp8wt operation - weight FP8 format conversion.

    Pointwise format conversion only (no scaling).
    """

    fn = lowering.ops_wrapper(torch.ops.spyre.qfp8wt.__name__)
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


@register_spyre_lowering(
    torch.ops.spyre.prod_dim_int,
    type_promotion_kind=None,
)
def lower_prod_dim(x, dim, keepdim=False):
    def _prod_dim_impl(x):
        kwargs = lowering._make_reduction_inner(
            x, axis=[dim], keepdims=keepdim, dtype=x.dtype, override_return_dtype=None
        )
        result = Reduction.create(reduction_type="prod", input_node=x, **kwargs)
        result.realize()
        return result

    return with_int64_fallback(_prod_dim_impl, x)


# ============================================================================
# Direct c10d Lowerings
# ============================================================================
@register_spyre_lowering(torch.ops._c10d_functional.broadcast.default)
def lower_c10d_broadcast_async(tensor, src_rank, group_name):
    """
    Direct lowering for _c10d_functional.broadcast using ASYNC pattern.

    Creates an async broadcast operation that returns immediately without blocking.
    This provides the infrastructure for potential communication-compute overlap,
    though actual overlap depends on scheduler decisions.

    Flow:
      _c10d_functional.broadcast → This lowering → BroadcastAsyncFallback
      → Generated code: torch.ops.spyre.broadcast_async() → C++ → spyre-comms (non-blocking)
    """
    logger.debug(
        "Lowering _c10d_functional.broadcast to BroadcastAsyncFallback "
        "(src_rank=%s, group_name='%s')",
        src_rank,
        group_name,
    )

    tensor.realize()
    return ir.TensorBox.create(
        BroadcastAsyncFallback(
            torch.ops.spyre.broadcast_async.default,
            tensor,
            src_rank,
            group_name,
        )
    )


@register_spyre_lowering(torch.ops._c10d_functional.wait_tensor.default)
def lower_c10d_wait_tensor_async(tensor):
    """
    Direct lowering for _c10d_functional.wait_tensor using ASYNC pattern.

    Synchronizes on the async broadcast operation, blocking until communication completes.

    Flow:
      _c10d_functional.wait_tensor → This lowering → WaitWorkFallback
      → Generated code: torch.ops.spyre.wait_work() → C++ → work->wait()
    """
    logger.debug("Lowering _c10d_functional.wait_tensor to WaitWorkFallback")

    tensor.realize()
    return ir.TensorBox.create(
        WaitWorkFallback(
            torch.ops.spyre.wait_work.default,
            tensor,
        )
    )


@register_spyre_lowering(torch.ops._c10d_functional.all_gather_into_tensor.default)
def lower_c10d_all_gather_async(tensor, group_size, group_name):
    """
    Direct lowering for _c10d_functional.all_gather_into_tensor using ASYNC pattern.

    Creates an async all_gather operation that returns immediately without blocking.
    Output tensor has shape[0] = input.shape[0] * group_size (concatenation of all ranks).

    Flow:
      _c10d_functional.all_gather_into_tensor → This lowering
      → AllGatherAsyncFallback → Generated code:
      torch.ops.spyre.all_gather_async() → C++ → spyre-comms (non-blocking)
    """
    logger.info(
        "Lowering _c10d_functional.all_gather_into_tensor to "
        "SpyreAllGatherAsyncFallback (group_size=%s, group_name='%s')",
        group_size,
        group_name,
    )

    tensor.realize()
    return ir.TensorBox.create(
        AllGatherAsyncFallback(
            torch.ops.spyre.all_gather_async.default,
            tensor,
            group_size,
            group_name,
        )
    )


@register_spyre_lowering(torch.ops._c10d_functional.all_reduce.default)
def lower_c10d_all_reduce_async(tensor, reduce_op, group_name):
    """
    Direct lowering for _c10d_functional.all_reduce using ASYNC pattern.

    Creates an async all_reduce operation that returns immediately without blocking.
    Output tensor has shape[0] = input.shape[0].
    """
    tensor.realize()
    logger.debug(
        "Lowering _c10d_functional.all_reduce to AllReduceAsyncFallback "
        "(reduce_op=%s, group_name='%s')",
        reduce_op,
        group_name,
    )
    return ir.TensorBox.create(
        AllReduceAsyncFallback(
            torch.ops.spyre.all_reduce_async.default,
            tensor,
            reduce_op,
            group_name,
        )
    )


@register_spyre_lowering(torch.ops._c10d_functional.all_reduce_.default)
def lower_c10d_all_reduce_inplace(tensor, reduce_op, group_name):
    """
    Lowering for _c10d_functional.all_reduce_ (in-place variant).

    Inductor's reinplace pass converts the functional all_reduce to the in-place
    all_reduce_ when the output shape matches the input. This lowering catches
    that case and emits the same Spyre all_reduce op (always in-place on device).
    """
    tensor.realize()
    logger.debug(
        "Lowering _c10d_functional.all_reduce_ to AllReduceAsyncFallback "
        "(reduce_op=%s, group_name='%s')",
        reduce_op,
        group_name,
    )
    return ir.TensorBox.create(
        AllReduceAsyncFallback(
            torch.ops._c10d_functional.all_reduce_.default,
            tensor,
            reduce_op,
            group_name,
        )
    )
