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

from torch._inductor.ir import Reduction, Pointwise
import torch._inductor.lowering as lowering

from typing import Any, Callable, Union

from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
import torch_spyre._inductor.customops  # noqa: F401
from torch_spyre._C import get_elem_in_stick
from torch_spyre.fallbacks import fallback_ops
from .ir import SpyreReduction
from torch._inductor.virtualized import V

import threading

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
def unregister_lowering(op, lowering_dict=lowering.lowerings, allow_missing=False):
    for overload in lowering.get_overloads(op):
        if overload in lowering_dict:
            del lowering_dict[overload]
        elif not allow_missing:
            raise RuntimeError(f"lowering of {overload} is not registered")


for op in fallback_ops:
    unregister_lowering(op, allow_missing=True)

# Overload names for aten.clamp
_CLAMP_FUNC_OVS = ["default", "Tensor", "Tensor_minmax"]


# Context manager that enables spyre specific lowerings in addition to PyTorch in-tree lowerings
@contextmanager
def enable_spyre_lowerings():
    """
    Existing CM that enables Spyre lowerings:
      - Temporarily redirect relevant aten ops → Spyre lowering
      - Restore original aten lowerings on exit

    This CM is reentrant and safe under nested usage.
    """
    global _lowerings_nesting
    with _lowerings_lock:
        first_enter = (_lowerings_nesting == 0)  # fmt: skip
        _lowerings_nesting += 1

        if first_enter:
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
                # Clean up
                enable_spyre_lowerings._saved_lowerings = {}


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
    def inner_fn(index, reduction_index):
        i0, i1 = index
        (r0,) = reduction_index
        return (x_loader([i0, r0]), y_loader([r0, i1]))

    x = V.graph.get_buffer(x.realize())
    y = V.graph.get_buffer(y.realize())
    x_loader = x.make_loader()
    y_loader = y.make_loader()

    result = Reduction.create(
        reduction_type=MATMUL_REDUCTION_OP,
        input_node=[x, y],
        device=x.get_device(),
        dst_dtype=x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=[x.get_size()[0], y.get_size()[1]],
        reduction_ranges=[x.get_size()[1]],
    )

    result.realize()

    return result


@register_spyre_lowering(torch.ops.aten.bmm.default)
def lower_bmm(x, y):
    x = V.graph.get_buffer(x.realize())
    y = V.graph.get_buffer(y.realize())
    x_loader = x.make_loader()
    y_loader = y.make_loader()
    d3 = len(x.get_size()) == 3
    if d3:

        def inner_fn(index, reduction_index):
            i0, i1, i2 = index
            (r0,) = reduction_index
            tmp1 = x_loader([i0, i1, r0])
            tmp2 = y_loader([i0, r0, i2])
            return (tmp1, tmp2)

        result = Reduction.create(
            reduction_type=BATCH_MATMUL_OP,
            input_node=[x, y],
            device=x.get_device(),
            dst_dtype=x.get_dtype(),
            src_dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=[x.get_size()[0], x.get_size()[1], y.get_size()[2]],  # B, M, N
            reduction_ranges=[x.get_size()[2]],  # K
        )
    else:  # 4d

        def inner_fn(index, reduction_index):
            i0, i1, i2, i3 = index
            (r0,) = reduction_index
            tmp1 = x_loader([i0, i1, i2, r0])
            tmp2 = y_loader([i0, i1, r0, i3])
            return (tmp1, tmp2)

        result = Reduction.create(
            reduction_type=BATCH_MATMUL_OP,
            input_node=[x, y],
            device=x.get_device(),
            dst_dtype=x.get_dtype(),
            src_dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=[
                x.get_size()[0],
                x.get_size()[1],
                x.get_size()[2],
                y.get_size()[-1],
            ],
            reduction_ranges=[x.get_size()[-1]],
        )

    result.realize()
    return result


@register_spyre_lowering(torch.ops.spyre.swap)
def lower_swap(x):
    fn = lowering.ops_wrapper(torch.ops.spyre.swap.__name__)

    def inner_fn(index):
        return fn(x.make_loader()(index))

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


@register_spyre_lowering(torch.ops.spyre.slice)
def lower_slice(x):
    fn = lowering.ops_wrapper(torch.ops.spyre.slice.__name__)

    def inner_fn(index):
        return fn(x.make_loader()(index))

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
        ranges=x.get_size()[:-1] + [get_elem_in_stick(x.get_dtype())],
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
