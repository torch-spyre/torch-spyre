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
from torch._dynamo.backends.common import AotAutograd
from .stickify import (
    spyre_matmul_result_shape,
    spyre_reduction_result_shape,
    spyre_pointwise_result_shape,
    SpyreDCI,
)

DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]


aten = torch.ops.aten


def spyre_matmul(x, y):
    res_size, res_dci = spyre_matmul_result_shape(x, y)
    res = x.new_empty(res_size)
    res.spyre_dci = res_dci
    return res


def spyre_amax(x, dim, keepdim=False):
    res_size, res_dci = spyre_reduction_result_shape(x, dim, keepdim)
    res = x.new_empty(res_size)
    res.spyre_dci = res_dci
    return res


def spyre_amin(x, dim, keepdim=False):
    res_size, res_dci = spyre_reduction_result_shape(x, dim, keepdim)
    res = x.new_empty(res_size)
    res.spyre_dci = res_dci
    return res


def spyre_max(x, dim, keepdim=False):
    res_size, res_dci = spyre_reduction_result_shape(x, dim, keepdim)
    values = x.new_empty(res_size)
    values.spyre_dci = res_dci
    indices = x.new_empty(res_size, dtype=torch.int64)
    indices.spyre_dci = res_dci
    return torch.return_types.max(sequence=(values, indices))


def spyre_min(x, dim, keepdim=False):
    res_size, res_dci = spyre_reduction_result_shape(x, dim, keepdim)
    values = x.new_empty(res_size)
    values.spyre_dci = res_dci
    indices = x.new_empty(res_size, dtype=torch.int64)
    indices.spyre_dci = res_dci
    return torch.return_types.min(sequence=(values, indices))


def spyre_sum(x, axis, keepdims=False):
    res_size, res_dci = spyre_reduction_result_shape(x, axis, keepdims)
    res = x.new_empty(res_size)
    res.spyre_dci = res_dci
    return res


def spyre_pointwise_binary(x, y):
    if isinstance(y, torch.Tensor):
        res_size, res_dci = spyre_pointwise_result_shape(x, y)
        res = x.new_empty(res_size)
        res.spyre_dci = res_dci
        return res
    else:
        return spyre_pointwise_unary(x)


def spyre_pointwise_unary(x):
    res = x.new_empty(x.size())
    res.spyre_dci = x.get_dci()
    return res


def spyre_unsqueeze(x, dim):
    x_size = x.size()
    x_dci = x.get_dci()
    if dim < 0:
        dim += len(x_size) + 1
    res_shape = list(x_size)
    res_shape.insert(dim, 1)
    res_dim_order = []
    for d in x_dci.dim_order:
        if d < dim:
            res_dim_order.append(d)
        elif d > dim:
            res_dim_order.append(d + 1)
        else:
            res_dim_order.append(d)
            res_dim_order.append(d + 1)
    res = x.new_empty(res_shape)
    res.spyre_dci = SpyreDCI(res_dim_order, x_dci.num_stick_dims, x_dci.format)
    return res


def spyre_where(cond, x, y):
    # TODO: check op validity and generalize
    res = x.new_empty(x.size())
    res.spyre_dci = x.get_dci()
    return res


def spyre_fresh_tensor_constructor_wrapper(orig_fn, *args, **kwargs):
    """Creating tensor fresh from size/stride.  Assume generic stick"""
    res = orig_fn(*args, **kwargs)
    res.spyre_dci = SpyreDCI.generic_stick_dci(res)
    return res


def spyre_like_tensor_constructor_wrapper(orig_fn, input, *args, **kwargs):
    """Creating a new tensor with same shape as input.  Propagate SpyreDCI if present."""
    res = orig_fn(input, *args, **kwargs)
    if hasattr(input, "spyre_dci"):
        res.spyre_dci = input.spyre_dci
    else:
        print(
            f"Warning: like_tensor constructor given {input} that lacks spyre_dci; assuming generic stick layout"
        )
        res.spyre_dci = SpyreDCI.generic_stick_dci(res)
    return res


_meta_ops = {
    # Reductions
    aten.amax.default: spyre_amax,
    aten.amin.default: spyre_amin,
    aten.max.dim: spyre_max,
    aten.min.dim: spyre_min,
    aten.mm.default: spyre_matmul,
    aten.sum.dim_IntList: spyre_sum,
    # Pointwise binary
    aten.add.Tensor: spyre_pointwise_binary,
    aten.div.Tensor: spyre_pointwise_binary,
    aten.eq.Tensor: spyre_pointwise_binary,
    aten.ge.Tensor: spyre_pointwise_binary,
    aten.mul.Tensor: spyre_pointwise_binary,
    aten.sub.Tensor: spyre_pointwise_binary,
    # Pointwise unary
    aten.abs.default: spyre_pointwise_unary,
    aten.exp.default: spyre_pointwise_unary,
    aten.log.default: spyre_pointwise_unary,
    aten.neg.default: spyre_pointwise_unary,
    aten.reciprocal.default: spyre_pointwise_unary,
    aten.relu.default: spyre_pointwise_unary,
    aten.rsqrt.default: spyre_pointwise_unary,
    aten.sigmoid.default: spyre_pointwise_unary,
    aten.sqrt.default: spyre_pointwise_unary,
    aten.tanh.default: spyre_pointwise_unary,
    # Other ops
    aten.detach.default: spyre_pointwise_unary,
    aten.unsqueeze.default: spyre_unsqueeze,
    aten.where.self: spyre_where,
}

_like_tensor_constructors = [
    aten.empty_like.default,
    aten.zeros_like.default,
    aten.ones_like.default,
]

_fresh_tensor_constructors = [
    aten.new_empty.default,
    aten.new_empty_strided.default,
    aten.new_full.default,
    aten.new_zeros.default,
    aten.new_ones.default,
]


@contextmanager
def spyre_meta_ops():
    orig = {}
    for op, fn in _meta_ops.items():
        orig[op] = op.py_kernels.copy()
        op.py_kernels.pop(DispatchKey.Meta)
        op.py_impl(DispatchKey.Meta)(fn)
    for op in _fresh_tensor_constructors:
        orig[op] = op.py_kernels.copy()
        orig_fn = op.py_kernels.pop(DispatchKey.Meta)
        op.py_impl(DispatchKey.Meta)(
            lambda *args,
            captured_orig_fn=orig_fn,
            **kwargs: spyre_fresh_tensor_constructor_wrapper(
                captured_orig_fn, *args, **kwargs
            )
        )
    for op in _like_tensor_constructors:
        orig[op] = op.py_kernels.copy()
        orig_fn = op.py_kernels.pop(DispatchKey.Meta)
        op.py_impl(DispatchKey.Meta)(
            lambda input,
            *args,
            captured_orig_fn=orig_fn,
            **kwargs: spyre_like_tensor_constructor_wrapper(
                captured_orig_fn, input, *args, **kwargs
            )
        )

    torch._subclasses.FakeTensorMode.cache_clear()
    try:
        yield
    finally:
        torch._subclasses.FakeTensorMode.cache_clear()
        for op, prior in orig.items():
            op.py_kernels.clear()
            op.py_kernels.update(prior)
            op._dispatch_cache.clear()


@contextmanager
def spyre_data_types():
    saved = torch._prims_common._computation_dtype_map
    torch._prims_common._computation_dtype_map = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.complex32: torch.complex32,
    }
    try:
        yield
    finally:
        torch._prims_common._computation_dtype_map = saved


class SpyreAotAutograd(AotAutograd):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs, **kwargs):
        if any(
            isinstance(t, torch.Tensor) and t.device.type == "spyre"
            for t in example_inputs
        ):
            with spyre_meta_ops(), spyre_data_types():
                return super().__call__(gm, example_inputs, **kwargs)
        else:
            return super().__call__(gm, example_inputs, **kwargs)
