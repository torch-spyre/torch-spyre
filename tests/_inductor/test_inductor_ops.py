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

import pytest
import unittest
import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
    make_param_dict,
)
from utils_inductor import compare, compare_with_cpu

POINTWISE_UNARY_OPS_DICT = {
    "abs": torch.abs,
    "cos": torch.cos,
    "exp": torch.exp,
    "neg": torch.neg,
    "reciprocal": torch.reciprocal,
    "relu": torch.relu,
    "sin": torch.sin,
    "tanh": torch.tanh,
}

POINTWISE_BINARY_OPS_DICT = {
    "add": torch.add,
    "mul": torch.mul,
    "sub": torch.sub,
    "div": torch.div,
}

REDUCTION_OPS_DICT = {
    "sum": torch.sum,
    "max": torch.max,
}

FP32_EPS = torch.finfo(torch.float32).eps  # 1.1920928955078125e-07
FP16_EPS = torch.finfo(torch.float16).eps  # 0.0009765625


class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)
    # Define parameter sets for each base test method
    # If parameterized, the base test method will not be invoked
    # The test methods that are not parameterized will be invoked
    # as usual (i.e. no change in their behaviors)
    # If using unittest.skip decorator on a base function that is
    # parameterized, the parameterized functions are skipped too
    # See utils.py for more details.
    PARAMS = {
        (
            "test_sqrt",
            "test_unary_op",
        ): {
            "ops_dict": {
                "sqrt": torch.sqrt,  # undefined for negative input
            },
            "param_sets": {
                "1d_abs": (cached_randn((64,), abs=True),),
                "2d_abs": (cached_randn((67, 256), abs=True),),
            },
        },
        (
            "test_rsqrt",
            "test_unary_op",
        ): {
            "ops_dict": {
                "rsqrt": torch.rsqrt,  # undefined for zero or negative input
            },
            "param_sets": {
                "1d_abs_nz": (cached_randn((64,), abs=True) + FP16_EPS,),
                "2d_abs_nz": (cached_randn((67, 256), abs=True) + FP16_EPS,),
            },
        },
        (
            "test_log",
            "test_unary_op",
        ): {
            "ops_dict": {
                "log": torch.log,  # undefined for zero or negative input
            },
            "param_sets": {
                "1d_abs_nz": (cached_randn((64,), abs=True) + FP16_EPS,),
                "2d_abs_nz": (cached_randn((67, 256), abs=True) + FP16_EPS,),
            },
        },
        (
            "test_pointwise_unary_op",
            "test_unary_op",
        ): {
            "ops_dict": POINTWISE_UNARY_OPS_DICT,
            "param_sets": make_param_dict(
                [
                    ((256,),),
                    ((67, 256),),
                    ((67, 71, 256),),
                ]
            ),
        },
        (
            "test_pointwise_binary_op",
            "test_binary_op",
        ): {
            "ops_dict": POINTWISE_BINARY_OPS_DICT,
            "param_sets": make_param_dict(
                [
                    ((256,),) * 2,
                    ((67, 256),) * 2,
                    ((67, 71, 256),) * 2,
                ]
            ),
        },
        ("test_add_broadcast", "test_add_broadcast"): {
            "param_sets": make_param_dict(
                [
                    ((256,), (67, 256)),
                ]
            )
        },
        ("test_add_broadcast_cpu", "test_add_broadcast_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((256,), (67, 256)),
                ]
            )
        },
        ("test_mm", "test_binary_op"): {
            "ops_dict": {
                "mm": torch.mm,
                "matmul": torch.matmul,
                # "einsum": lambda a, b: torch.einsum('mk, kn -> mn', a, b),  # bmm not supported yet
            },
            "param_sets": make_param_dict(
                [
                    ((67, 256), (256, 128)),
                ]
            ),
        },
        ("test_bmm", "test_binary_op"): {
            "ops_dict": {
                "bmm": torch.bmm,
            },
            "param_sets": make_param_dict(
                [
                    ((3, 17, 256), (3, 256, 128)),
                ]
            ),
        },
        # ("test_reduce_2d", "test_reduce"): {
        #     "ops_dict": REDUCTION_OPS_DICT,
        #     "param_sets": {
        #         "dim_0": (0, cached_randn((67, 256))),
        #         # Skip: `cpu()` on sparse tensor doesn't work in eager mode yet
        #         # "dim_1": (1, cached_randn((67, 256))),
        #     },
        # },
        ("test_max_sub_broadcast_cpu", "test_max_sub_broadcast_cpu"): {
            "param_sets": {
                "dim_0": (0, cached_randn((128, 256))),
                "dim_1": (1, cached_randn((128, 256))),
            },
        },
        (
            "test_alias_operands",
            "test_unary_op",
        ): {
            "ops_dict": {
                "double": lambda x: x + x,
                "square": lambda x: x * x,
                "cube": lambda x: x * x * x,
                "triple": lambda x: x + x + x,
            },
            "param_sets": make_param_dict(
                [
                    ((256,),),
                    ((67, 256),),
                    # ((67, 71, 256),), # 3d input causes eager timeout
                ]
            ),
        },
        (
            "test_alias_operands_cpu",
            "test_unary_op_cpu",
        ): {
            "ops_dict": {
                "pow": lambda x: torch.pow(x, 2),
            },
            "param_sets": make_param_dict(
                [
                    ((256,),),
                    ((67, 256),),
                ]
            ),
        },
        # Compare with cpu for now to avoid hitting eager mode coverage issue
        ("test_max_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": {
                "sum": torch.max,
            },
            "param_sets": {
                "2d_dim_0": (0, cached_randn((67, 256))),
                # "2d_dim_1": (1, cached_randn((67, 256))), # `cpu()` on sparse tensor doesn't work in eager mode yet
                # "3d_dim_0": (0, cached_randn((67, 71, 256))), # layout needs repermutation
                "3d_dim_1": (1, cached_randn((67, 71, 256))),
                # "3d_dim_2": (2, cached_randn((67, 71, 256))), # sparse tensor output
            },
        },
        ("test_max_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": {
                "sum": torch.max,
            },
            "param_sets": {
                "2d_dim_0": (0, cached_randn((67, 256))),
                "2d_dim_1": (1, cached_randn((67, 256))),  # sparse tensor output
                "3d_dim_0": (0, cached_randn((67, 71, 256))),
                "3d_dim_1": (1, cached_randn((67, 71, 256))),
                "3d_dim_2": (2, cached_randn((67, 71, 256))),  # sparse tensor output
            },
        },
        ("test_sum_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": {
                "sum": torch.sum,
            },
            "param_sets": {
                "2d_dim_0": (0, cached_randn((67, 256))),
                # "2d_dim_1": (1, cached_randn((67, 256))), # `cpu()` on sparse tensor doesn't work in eager mode yet
                # "2d_dim_01": ([0, 1], cached_randn((67, 256))), # spyre scalar represented as 1d instead of 0d
                # "3d_dim_0": (0, cached_randn((67, 71, 256), scale=0.01)), # layout needs repermutation
                "3d_dim_1": (1, cached_randn((67, 71, 256), scale=0.01)),
                # "3d_dim_2": (2, cached_randn((67, 71, 256), scale=0.01)), # sparse tensor output
                "3d_dim_01": ([0, 1], cached_randn((67, 71, 256), scale=0.01)),
                # "3d_dim_012": ([0, 1, 2], cached_randn((67, 71, 256), scale=0.01)), # spyre scalar represented as 1d instead of 0d
            },
        },
        ("test_sum_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": {
                "sum": torch.sum,
            },
            "param_sets": {
                "2d_dim_0": (0, cached_randn((67, 256))),
                "2d_dim_1": (1, cached_randn((67, 256))),  # sparse tensor output
                "2d_dim_01": ([0, 1], cached_randn((67, 256))),
                "3d_dim_0": (0, cached_randn((3, 5, 256), scale=0.1)),
                "3d_dim_1": (1, cached_randn((67, 71, 256), scale=0.1)),
                "3d_dim_2": (
                    2,
                    cached_randn((67, 71, 256), scale=0.1),
                ),  # sparse tensor output
                "3d_dim_01": ([0, 1], cached_randn((67, 71, 256), scale=0.1)),
                "3d_dim_012": ([0, 1, 2], cached_randn((67, 71, 256), scale=0.1)),
            },
        },
        ("test_transpose_2d_cpu", "test_transpose_2d_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((1088, 320),),
                    ((320, 320),),
                ]
            ),
        },
        ("test_transpose_3d_cpu", "test_transpose_3d_cpu"): {
            "param_sets": {
                "dim_0_2": (
                    0,
                    2,
                    cached_randn((512, 256, 128), abs=True),
                ),
                "dim_1_2": (
                    1,
                    2,
                    cached_randn((512, 256, 128), abs=True),
                ),
                "dim_0_2_same_dim": (
                    0,
                    2,
                    cached_randn((128, 128, 128), abs=True),
                ),
            }
        },
        ("test_transpose_4d_cpu", "test_transpose_4d_cpu"): {
            "param_sets": {
                "dim_0_3": (
                    0,
                    3,
                    cached_randn((256, 3, 17, 64), abs=True),
                ),
                # skipping these - not working yet
                # "dim_1_3": (
                #     1,
                #     3,
                #     cached_randn((3, 256, 17, 64), abs=True),
                # ),
                # "dim_2_3": (
                #     2,
                #     3,
                #     cached_randn((3, 17, 256, 64), abs=True),
                # ),
            }
        },
        ("test_cmp", "test_binary_op_cpu"): {
            "ops_dict": {
                "eq": torch.eq,
                "ne": torch.ne,
                "ge": torch.ge,
                "le": torch.le,
                "gt": torch.gt,
                "lt": torch.lt,
            },
            "param_sets": {
                "1d": (
                    torch.ceil(cached_randn((256,), abs=True, scale=10.0)).to(
                        dtype=torch.float16
                    ),
                    torch.ceil(cached_randn((256,), abs=True, scale=9.9)).to(
                        dtype=torch.float16
                    ),
                ),
                "2d": (
                    torch.ceil(cached_randn((64, 128), abs=True, scale=10.0)).to(
                        dtype=torch.float16
                    ),
                    torch.ceil(cached_randn((64, 128), abs=True, scale=9.9)).to(
                        dtype=torch.float16
                    ),
                ),
                "3d": (
                    torch.ceil(cached_randn((2, 32, 128), abs=True, scale=10.0)).to(
                        dtype=torch.float16
                    ),
                    torch.ceil(cached_randn((2, 32, 128), abs=True, scale=9.9)).to(
                        dtype=torch.float16
                    ),
                ),
                "broadcast": (
                    torch.ceil(cached_randn((256, 256), abs=True, scale=10.0)).to(
                        dtype=torch.float16
                    ),
                    torch.ceil(cached_randn((256,), abs=True, scale=9.9)).to(
                        dtype=torch.float16
                    ),
                ),
            },
        },
        (
            "test_where",
            "test_where_cpu",
        ): {
            "ops_dict": {
                "eq": lambda x, y: x == y,
                "ne": lambda x, y: x != y,
                "ge": lambda x, y: x >= y,
                "le": lambda x, y: x <= y,
                "gt": lambda x, y: x > y,
                "lt": lambda x, y: x < y,
            },
            "param_sets": {
                "1d256": (
                    torch.ceil(cached_randn((256,), abs=True, scale=10.0)).to(
                        dtype=torch.float16
                    ),
                    torch.ceil(cached_randn((256,), abs=True, scale=9.9)).to(
                        dtype=torch.float16
                    ),
                ),
            },
        },
        (
            "test_pointwise_binary_op_fp32",
            "test_binary_op",
        ): {
            "ops_dict": POINTWISE_BINARY_OPS_DICT,
            "param_sets": {
                "fp32": (
                    cached_randn((67, 256), dtype=torch.float32),
                    cached_randn((67, 256), dtype=torch.float32),
                ),
            },
        },
        (
            "test_pointwise_range_op",
            "test_range_op",
        ): {
            "ops_dict": {
                "clamp": torch.clamp,
            },
            "param_sets": {
                "fp16": (
                    cached_randn((128, 256), dtype=torch.float16),
                    0.1,
                    0.9,
                    FP16_EPS,
                ),
            },
        },
        (
            "test_activation_cls",
            "test_activation_cls",
        ): {
            "ops_dict": {
                "gelu": torch.nn.GELU,
            },
            "param_sets": {
                "fp16": (
                    cached_randn((128, 128), dtype=torch.float16),
                    {
                        "approximate": "tanh",
                    },
                    0.01,
                ),
            },
        },
        (
            "test_activation_fn",
            "test_activation_fn",
        ): {
            "ops_dict": {
                "silu": torch.nn.functional.silu,
            },
            "param_sets": {
                "fp16": (
                    cached_randn((128, 128), dtype=torch.float16),
                    0.01,
                ),
            },
        },
        (
            "test_clone",
            "test_clone",
        ): {
            "param_sets": {
                "1d": (cached_randn((128,), dtype=torch.float16),),
                "2d": (cached_randn((256, 128), dtype=torch.float16),),
                "3d": (cached_randn((8, 16, 256), dtype=torch.float16),),
            },
        },
        (
            "test_fallback",
            "test_fallback_cpu",
        ): {
            "param_sets": {
                "1d": (cached_randn((128,), dtype=torch.float16),),
                "2d": (cached_randn((256, 128), dtype=torch.float16),),
                "3d": (cached_randn((8, 16, 256), dtype=torch.float16),),
            },
        },
        (
            "test_arange",
            "test_arange_cpu",
        ): {
            "param_sets": {
                "end": (64.0,),
                "start_end": (64.0, 128.0),
                "start_end_step": (0.0, 128.0, 2.0),
            },
        },
        (
            "test_new_ones",
            "test_new_ones_cpu",
        ): {
            "param_sets": {
                "size_1": (
                    cached_randn((64, 256)),
                    ([64, 256]),
                ),
            },
        },
        (
            "test_numel",
            "test_numel_cpu",
        ): {
            "param_sets": {
                "size_1": {
                    cached_randn(
                        (
                            64,
                            128,
                        )
                    ),
                },
            },
        },
        (
            "test_full",
            "test_full_cpu",
        ): {
            "param_sets": {
                "value_1": (
                    ([64, 128]),
                    -65472.0,
                ),
                "value_2": (
                    ([64, 128]),
                    -65504.0,
                ),
                "tuple": (
                    ((64, 64)),
                    1024.0,
                ),
                "size": (
                    torch.Size([64, 128]),
                    1024.0,
                ),
            },
        },
        (
            "test_dropout_functional",
            "test_dropout_functional",
        ): {
            "param_sets": {
                "value_3d": (
                    cached_randn((64, 11, 2048)),
                    {
                        "p": 0.5,
                        "training": False,
                        "inplace": False,
                    },
                ),
                "value_4d": (
                    cached_randn((1, 64, 11, 2048)),
                    {
                        "p": 0.0,
                        "training": False,
                        "inplace": False,
                    },
                ),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.mark.filterwarnings("ignore::torch_spyre.fallbacks.FallbackWarning")
    def test_unary_op(self, op, x):
        if op == torch.reciprocal:
            # TODO: Division by 0 or near-zero differs on Spyre from CPU, sidestep for now.
            tiny_value_mask = torch.abs(x) < FP16_EPS
            x[tiny_value_mask] = FP16_EPS

        cpu_ops = {
            torch.cos,  # CPU fallback
            torch.exp,  # TODO: eager / sendnn results are radically differ from CPU. deeptools bug?
            torch.sin,  # CPU fallback
        }
        if op in cpu_ops:
            compare_with_cpu(op, x)
        elif op == torch.neg:
            compare_with_cpu(op, x)
        else:
            compare(op, x)

    def test_bool(self):
        # torch._dynamo.config.dynamic_shapes = False
        dtype = torch.bool
        x = torch.randint(0, 2, (2, 64), dtype=dtype)
        x_spyre = x.to("spyre")
        y = torch.randint(0, 2, (2, 64), dtype=dtype)
        y_spyre = y.to("spyre")
        result = torch.compile(torch.eq, dynamic=False)(x_spyre, y_spyre).cpu()
        torch.testing.assert_close(result, torch.eq(x, y))

    def test_unary_op_cpu(self, op, x):
        compare_with_cpu(op, x)

    def test_binary_op(self, op, a, b):
        if op == torch.div:
            # TODO: Division by 0 or near-zero differs on Spyre from CPU, sidestep for now.
            tiny_value_mask = torch.abs(b) < FP16_EPS
            b[tiny_value_mask] = FP16_EPS

        if a.dtype == torch.float32:
            compare_with_cpu(op, a, b)
        elif op == torch.bmm:
            # TODO: Eager mode mismatch causing cryptic error, sidestep for now.
            compare_with_cpu(op, a, b)
        else:
            compare(op, a, b)

    def test_binary_op_cpu(self, op, x, y):
        compare_with_cpu(op, x, y)

    @unittest.skip("deeptools: error")
    def test_add_broadcast(self, x, y):
        compare(lambda x, y: torch.add(x[None, :], y), x, y)

    # Example where base function is not parameterized
    def test_add_broadcast_cpu(self, x, y):
        compare_with_cpu(lambda x, y: torch.add(x[None, :], y), x, y)

    # @unittest.skip("eager mode crashes")
    # def test_reduce(self, op, dim: int, x):
    #     if op == torch.max:
    #         compare(lambda x: op(x, dim=dim)[0], x)
    #     else:
    #         compare(lambda x: op(x, dim=dim), x)

    def test_reduce_keepdim0_cpu(self, op, dim: int, x):
        if op == torch.max:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=False)[0], x)
        else:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=False), x)

    def test_reduce_keepdim1_cpu(self, op, dim: int, x):
        if op == torch.max:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=True)[0], x)
        else:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=True), x)

    def test_max_sub_broadcast_cpu(self, dim: int, x):
        def fn(x):
            x_max = torch.max(x, dim=dim)[0]
            z = x - torch.unsqueeze(x_max, dim=dim)
            return z

        compare_with_cpu(fn, x)  # eager mode crashes

    def test_transpose_2d_cpu(self, x):
        compare_with_cpu(lambda x: x.t().contiguous(), x)

    def test_transpose_3d_cpu(self, dim0: int, dim1: int, x):
        compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1).contiguous(), x)

    def test_transpose_4d_cpu(self, dim0: int, dim1: int, x):
        compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1).contiguous(), x)

    def test_where_cpu(self, cond_op, x, y):
        compare_with_cpu(lambda x, y: torch.where(cond_op(x, y), x, y), x, y)

    def test_range_op(self, op, input, min, max, err):
        compare_with_cpu(lambda x: op(x, min, max), input, atol=err, rtol=err)

    def test_activation_cls(self, op, input, kwargs, err):
        compare_with_cpu(lambda x: op(**kwargs)(x), input, atol=err, rtol=err)

    def test_activation_fn(self, op, input, err):
        compare_with_cpu(lambda x: op(x), input, atol=err, rtol=err)

    def test_clone(self, x):
        compare_with_cpu(lambda a: torch.clone(a).contiguous(), x)

    def test_dropout_functional(self, input, kwargs):
        compare_with_cpu(lambda a: torch.nn.functional.dropout(a, **kwargs), input)

    @pytest.mark.filterwarnings("ignore::torch_spyre.fallbacks.FallbackWarning")
    def test_fallback_cpu(self, x):
        def fn(t):
            t = torch.exp(t)  # compiled op
            t = torch.sin(t)  # fallback op
            t = torch.exp(t)  # compiled op
            return t

        compare_with_cpu(fn, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.fallbacks.FallbackWarning")
    def test_arange_cpu(self, *args):
        def fn(device=None):
            return torch.arange(*args, dtype=torch.float16, device=device)

        compare_with_cpu(fn, needs_device=True)

    def test_new_ones_cpu(self, x, y):
        compare_with_cpu(lambda x: x.new_ones((x.size())), x)

    def test_numel_cpu(self, x):
        compare_with_cpu(lambda x: torch.numel(x), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.fallbacks.FallbackWarning")
    def test_full_cpu(self, *args):
        def fn(device=None):
            return torch.full(*args, dtype=torch.float16, device=device)

        compare_with_cpu(fn, needs_device=True)


if __name__ == "__main__":
    unittest.main()
