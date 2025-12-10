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
from .ir import SpyrePointwise, SpyreReduction


@lowering.register_lowering(torch.ops.aten.mm.default)
def lower_mm(x, y):
    def inner_fn(index, reduction_index):
        i0, i1 = index
        (r0,) = reduction_index
        tmp1 = ops.load(x.get_name(), x.get_layout().stride[0] * i0 + r0)
        tmp2 = ops.load(y.get_name(), i1 + y.get_layout().stride[0] * r0)
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


@lowering.register_lowering(torch.ops.aten.bmm.default)
def lower_bmm(x, y):
    def inner_fn(index, reduction_index):
        i0, i1, i2 = index
        x_layout = x.get_layout()
        y_layout = y.get_layout()
        (r0,) = reduction_index
        tmp1 = ops.load(
            x.get_name(), x_layout.stride[0] * i0 + x_layout.stride[1] * i1 + r0
        )
        tmp2 = ops.load(
            y.get_name(), y_layout.stride[0] * i0 + y_layout.stride[1] * r0 + i2
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


@lowering.register_lowering(torch.ops.spyre.swap)
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


@lowering.register_lowering(torch.ops.spyre.slice)
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


@lowering.register_lowering(torch.ops.spyre.exx2)
def lower_exx2(x, exx2Scale, useZeroMean):
    kwargs = lowering._make_reduction_inner(
        x, axis=[-1], keepdims=False, dtype=x.dtype, override_return_dtype=None
    )
    op_info = {
        "constants": {
            "exx2scale": exx2Scale,
            "useZeroMean": useZeroMean,
        }
    }
    result = SpyreReduction.create(
        reduction_type="exx2", input_node=x, op_info=op_info, **kwargs
    )
    result.realize()
    return result


# TODO: Put this inside a register_spyre_lowering decorator??
lowering.register_op_dtype_propagation_rules(
    "layernormnorm", lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None
)


@lowering.register_lowering(torch.ops.spyre.layernormnorm)
def lower_layernormnorm(x, mean, norm_mean, weight, bias):
    fn = lowering.ops_wrapper("layernormnorm")

    def inner_fn(index):
        loaded_inputs = [
            x.make_loader()(index),
            mean.make_loader()(index[-1:]),
            norm_mean.make_loader()(index[-1:]),
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


# TODO: Put this inside a register_spyre_lowering decorator??
lowering.register_op_dtype_propagation_rules(
    "layernormscale", lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None
)


@lowering.register_lowering(torch.ops.spyre.layernormscale)
def lower_layernormscale(x, eps):
    fn = lowering.ops_wrapper(torch.ops.spyre.layernormscale.__name__)

    def inner_fn(index):
        return fn(x.make_loader()(index), eps)

    pw = SpyrePointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
        op_info={"constants": {"eps": eps}},
    )
    pw.realize()
    return pw


@lowering.register_lowering(torch.ops.aten.mean.dim)
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


lowering.register_op_dtype_propagation_rules(
    "gelu", lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None
)


@lowering.register_lowering(torch.ops.spyre.gelu)
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


lowering.register_op_dtype_propagation_rules(
    "softplus", lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None
)


@lowering.register_lowering(torch.ops.spyre.softplus)
def lower_softplus(x, beta=1.0, threshold=20.0):
    op_info = {
        "constants": {
            "softplusBeta": beta,
            "softplusThresh": threshold,
        }
    }

    fn = lowering.ops_wrapper(torch.ops.spyre.softplus.__name__)

    def inner_fn(index):
        return fn(x.make_loader()(index), beta, threshold)

    pw = SpyrePointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
        op_info=op_info,
    )
    pw.realize()
    return pw
