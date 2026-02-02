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

import torch

from torch._inductor.ir import Reduction, Pointwise
from torch._inductor.virtualized import ops
import torch._inductor.lowering as lowering

from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from torch_spyre._C import get_elem_in_stick
from torch_spyre.fallbacks import fallback_ops
from .ir import SpyreReduction


def register_spyre_lowering(
    op,
    name=None,
    broadcast=False,
    type_promotion_kind=lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    override_return_dtype=None,
    convert_input_to_bool=False,
    lowering_dict=lowering.lowerings,
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
        tmp1 = ops.load(x.get_name(), x.get_size()[1] * i0 + r0)
        tmp2 = ops.load(y.get_name(), i1 + y.get_size()[1] * r0)
        return (tmp1, tmp2)

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
    def inner_fn(index, reduction_index):
        i0, i1, i2 = index
        (r0,) = reduction_index
        tmp1 = ops.load(
            x.get_name(),
            x.get_size()[2] * x.get_size()[1] * i0 + x.get_size()[1] * i1 + r0,
        )
        tmp2 = ops.load(
            y.get_name(),
            y.get_size()[2] * y.get_size()[1] * i0 + y.get_size()[1] * r0 + i2,
        )
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

    result.realize()

    return result


@register_spyre_lowering(torch.ops.spyre.swap)
def lower_swap(x):
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw


@register_spyre_lowering(torch.ops.spyre.slice)
def lower_slice(x):
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
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
