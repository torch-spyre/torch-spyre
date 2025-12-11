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
    "exp": torch.exp,
    "reciprocal": torch.reciprocal,
    "relu": torch.relu,
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
        ("test_reduce_2d", "test_reduce"): {
            "ops_dict": REDUCTION_OPS_DICT,
            "param_sets": {
                "dim_0": (0, cached_randn((67, 256))),
                # Skip: `cpu()` on sparse tensor doesn't work in eager mode yet
                # "dim_1": (1, cached_randn((67, 256))),
            },
        },
        ("test_reduce_2d_cpu", "test_reduce_cpu"): {
            "ops_dict": REDUCTION_OPS_DICT,
            "param_sets": {
                "dim_0": (0, cached_randn((67, 256))),
                # Skip: `cpu()` on sparse tensor doesn't work in eager mode yet
                # "dim_1": (1, cached_randn((67, 256))),
            },
        },
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
        # skipping these - not working yet
        # ("test_reduce_3d",
        #  "test_reduce"
        # ):{
        #     "ops_dict": REDUCTION_OPS_DICT,
        #     "param_sets": {
        #         "dim_0": (0, cached_randn((67, 71, 256))),
        #         "dim_1": (1, cached_randn((67, 71, 256))),
        #         "dim_2": (2, cached_randn((67, 71, 256))),
        #     }
        # },
        ("test_transpose_2d_cpu", "test_transpose_2d_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((1088, 320),),
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
            }
        },
        (
            "test_where",
            "test_where_cpu",
        ): {
            "param_sets": {
                "eq": (
                    lambda x, y: x == y,
                    cached_randn((256,)),
                    cached_randn((256,)),
                ),
                "ge": (
                    lambda x, y: x >= y,
                    cached_randn((256,)),
                    cached_randn((256,)),
                ),
                "ne": (
                    lambda x, y: x != y,
                    cached_randn((256,)),
                    cached_randn((256,)),
                ),
            }
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
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_unary_op(self, op, x):
        if op == torch.reciprocal:
            # TODO: Division by 0 or near-zero differs on Spyre from CPU, sidestep for now.
            tiny_value_mask = torch.abs(x) < FP16_EPS
            x[tiny_value_mask] = FP16_EPS

        if op == torch.exp:
            # TODO: eager / sendnn results are radically differ from CPU. deeptools bug?
            compare_with_cpu(op, x)
        else:
            compare(op, x)

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

    @unittest.skip("deeptools: error")
    def test_add_broadcast(self, x, y):
        compare(lambda x, y: torch.add(x[None, :], y), x, y)

    # Example where base function is not parameterized
    def test_add_broadcast_cpu(self, x, y):
        compare_with_cpu(lambda x, y: torch.add(x[None, :], y), x, y)

    @unittest.skip("eager mode crashes")
    def test_reduce(self, op, dim: int, x):
        if op == torch.max:
            compare(lambda x: op(x, dim=dim)[0], x)
        else:
            compare(lambda x: op(x, dim=dim), x)

    def test_reduce_cpu(self, op, dim: int, x):
        if op == torch.max:
            compare_with_cpu(lambda x: op(x, dim=dim)[0], x)
        else:
            compare_with_cpu(lambda x: op(x, dim=dim), x)

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

    def test_where_cpu(self, cond_op, x, y):
        compare_with_cpu(lambda x, y: torch.where(cond_op(x, y), x, y), x, y)


if __name__ == "__main__":
    unittest.main()
